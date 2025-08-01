import warnings
warnings.filterwarnings("ignore", category=UserWarning, message=r"Environment variable TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD detected.*", module=r"e3nn\.o3\._wigner")

from typing import Sequence

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from ase.build import bulk
from numpy.random import Generator
from pymatgen.io.ase import AseAtomsAdaptor
from smol.cofe.space.domain import get_allowed_species
from smol.moca import Ensemble, Sampler
from tc.sampler_data import SamplerData
import tc.dataset

import plotly.graph_objects as go
import math
from joblib import Parallel, delayed
import os

# =====================================================================
# Public API
# =====================================================================

def make_sampler_filepath(composition: dict[str, float], supercell_size: int, *, lattice_relaxed: bool) -> str:
    comp_str_list = []
    for el, ratio in composition.items():
        ratio_for_str = round(1000 * ratio)
        if abs(ratio_for_str / 1000 - ratio) > 1e-6:
            raise ValueError(f"Ratio {ratio} is not a multiple of 0.001; cannot convert to filename.")
        comp_str_list.append(f"{el}{ratio_for_str}")
    comp_str = "-".join(sorted(comp_str_list))
    return os.path.join(tc.dataset.run_folderpath(list(composition.keys()), supercell_size, lattice_relaxed=lattice_relaxed), f"{comp_str}.npz")


def Tc_from_Cv(temperatures_K: np.ndarray, Cv: np.ndarray) -> tuple[float, float]:
    i = int(np.argmax(Cv))
    Tc = temperatures_K[i]

    # parabolic refinement if interior point
    if 0 < i < len(Cv)-1:
        x1,x2,x3 = temperatures_K[i-1:i+2]
        y1,y2,y3 = Cv[i-1:i+2]
        denom = (y1 - 2*y2 + y3)
        if denom != 0:
            Tc = x2 + 0.5 * (y1 - y3) * (x3 - x1) / denom

    # simple HWHM uncertainty
    half_max  = 0.5 * Cv[i]
    j = i
    while j >= 0:
        if Cv[j] <= half_max:
            left = temperatures_K[j]
            break
        j -= 1
    else:
        raise ValueError("Cv does not cross half-maximum on the low end.")
    j = i
    while j < len(Cv):
        if Cv[j] <= half_max:
            right = temperatures_K[j]
            break
        j += 1
    else:
        raise ValueError("Cv does not cross half-maximum on the high end.")

    dTc = 0.5 * (right - left)
    return Tc, dTc

def initialize_wl_sampler(
    ensemble: Ensemble,
    *,
    num_bins: int,
    flatness: float = 0.8,
    seeds: list[int],
    window: tuple[float, float],
) -> Sampler:
    """Run a Wang-Landau sampler and show a three-panel diagnostic figure.

    The figure contains:
    1. Modification-factor convergence.
    2. Density of states within the WL energy window.
    3. Heat-capacity curve, **computed in this function** and passed to the
       plotting helper.
    """


    # ------------------------------------------------------------
    # 1) Define the WL energy window (per primitive cell)
    # ------------------------------------------------------------
    min_E, max_E = window
    bin_size = (max_E - min_E) / num_bins
    max_E = min_E + num_bins * bin_size - 1e-8*(max_E - min_E)/num_bins  # avoid floating-point rounding issues
    actual_num_bins = len(np.arange(min_E, max_E, bin_size))
    if actual_num_bins != num_bins:
        raise ValueError(f"Wang-Landau would create {actual_num_bins} bins, but you requested {num_bins}.  "
            "This is due to floating-point rounding; adjust max_E (e.g. np.nextafter) or bin_size so the counts match."
        )

    # ------------------------------------------------------------
    # 2) Build the Wang-Landau sampler
    # ------------------------------------------------------------
    sampler = Sampler.from_ensemble(
        ensemble,
        kernel_type="Wang-Landau",
        bin_size=bin_size,
        step_type="swap",
        flatness=flatness,
        min_enthalpy=min_E,
        max_enthalpy=max_E,
        seeds=list(seeds),
    )
    return sampler

def generate_wl_plots(
    data: SamplerData,
    temperatures: np.ndarray,
) -> None:
    """
    Plot WL diagnostics using a *serialised* sampler snapshot.

    Parameters
    ----------
    data : SamplerData
        Output of ``load_sampler_data``.
    temperatures, Cv
        Same arrays you previously fed into the old function.
    """

    # ------------------------------------------------------------------
    # figure layout
    # ------------------------------------------------------------------
    _, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=False, constrained_layout=True)
    ax_conv, ax_dos   = axes[0]
    ax_cv,   ax_hist  = axes[1]

    # WL convergence  (top-left)
    ax_conv.semilogy(data.mod_factor_trace, ".-")
    ax_conv.set(
        xlabel="Iteration",
        ylabel="Modification factor (ln f)",
        title="WL convergence",
    )

    # Density of states  (top-right)
    entropy     = data.entropy
    nbins       = entropy.size
    bin_centers = data.min_E + (np.arange(nbins) + 0.5) * data.bin_size

    mask    = entropy > 0
    S_shift = entropy[mask] - entropy[mask].min()
    dos     = np.exp(S_shift - S_shift.max())
    dos    /= dos.sum()

    ax_dos.semilogy(bin_centers[mask], dos, ".-")
    ax_dos.set(
        xlabel=r"$E$ (eV / supercell)",
        ylabel="Density of states",
        title="WL DOS estimate",
        xlim=(data.min_E, data.max_E),
    )

    # Heat capacity  (bottom-left)
    Cv = compute_thermodynamics(data, temperatures)
    ax_cv.plot(temperatures, Cv, lw=2)
    ax_cv.set(
        xlabel="Temperature (K)",
        ylabel=r"$C_v$ per supercell (eV K$^{-1}$)",
        title=r"Wang-Landau $C_v(T)$",
    )

    # Histogram of kept configs  (bottom-right)
    # counts = compute_histogram_per_bin(data)
    counts = data.histogram
    ax_hist.bar(
        bin_centers,
        counts,
        width=0.9 * data.bin_size,
        align="center",
        edgecolor="k",
    )
    ax_hist.set(
        xlabel=r"$E$ (eV / supercell)",
        ylabel="# kept configs",
        title="Trace occupancy per energy bin",
        xlim=(data.min_E, data.max_E),
    )

    plt.show()

# ---------------------------------------------------------------
def initialize_supercell_occupancy(
    ensemble      : Ensemble,
    rng           : Generator,
    composition   : dict[str, float],           # e.g. {"Co":0.25, "Mn":0.75}
) -> np.ndarray:
    """
    Build a random supercell consistent with `composition` on the cation
    sub-lattice, then return the integer occupancy encoding expected by SMOL.

    Notes
    -----
    * Fractions must sum to 1 (±1 e-6).  
    * Elements in `composition` must all replace `replace_element`.
    """
    # ── 0 · sanity ------------------------------------------------------
    if not math.isclose(sum(composition.values()), 1.0, abs_tol=1e-6):
        raise ValueError("Fractions in `composition` must sum to 1.")

    elem2idx = get_cation_index_map(ensemble)
    unknown  = set(composition) - set(elem2idx)
    if unknown:
        raise KeyError(f"Elements {unknown} not present in the ensemble site-space")

    # ── 1 · make an MgO prototype of identical size --------------------
    n_sc = round(ensemble.processor.size ** (1/3))
    if n_sc**3 != ensemble.processor.size:
        raise ValueError("Ensemble supercell is not cubic.")
    sc_tuple = (n_sc, n_sc, n_sc)
    snapshot = bulk("MgO", crystalstructure="rocksalt", a=4.3, cubic=True) * sc_tuple

    # indices of the cation sites to be replaced
    repl_idx = [i for i, at in enumerate(snapshot) if at.symbol == "Mg"]
    rng.shuffle(repl_idx)

    # ── 2 · translate fractions → exact integer counts -----------------
    n_cations      = len(repl_idx)
    target_counts  = {el: round(frac * n_cations) for el, frac in composition.items()}

    # fix rounding drift so total matches exactly
    delta = n_cations - sum(target_counts.values())
    if delta:
        for el in list(target_counts)[:abs(delta)]:
            target_counts[el] += int(math.copysign(1, delta))

    # sanity
    assert sum(target_counts.values()) == n_cations

    # ── 3 · assign elements to sites -----------------------------------
    start = 0
    for el, n_el in target_counts.items():
        end = start + n_el
        snapshot.symbols[repl_idx[start:end]] = el
        start = end

    # ── 4 · encode to SMOL occupancy -----------------------------------
    sc_mat = np.diag(sc_tuple)
    pmg_struct = AseAtomsAdaptor.get_structure(snapshot)
    occ_enc = ensemble.processor.cluster_subspace.occupancy_from_structure(pmg_struct, scmatrix=sc_mat, encode=True)
    return occ_enc.astype(np.int32)


def compute_thermodynamics(
    sampler: SamplerData,
    temperatures_K: np.ndarray,
) -> np.ndarray:
    """Compute heat capacity from Wang-Landau sampler results.
    
    Parameters
    ----------
    sampler : Sampler
        Wang-Landau sampler with entropy and energy data
    temperatures_K : np.ndarray
        Temperature array (K)
        
    Returns
    -------
    Cv : np.ndarray
        Heat capacity per primitive cell (eV K^-1)
    """
    # Extract density of states from entropy
    entropy = sampler.entropy
    mask = entropy > 0
    ent_ref = entropy[mask] - entropy[mask].min()
    dos_levels = np.exp(ent_ref - ent_ref.max())
    dos_levels /= dos_levels.sum()
    energy_levels = sampler.energy_levels[mask]

    # Compute thermodynamic properties
    k_B = 8.617333262e-5  # eV / K
    E_rr = energy_levels - energy_levels.min()

    Z = np.array([np.sum(dos_levels * np.exp(-E_rr / (k_B * T))) for T in temperatures_K])
    U = np.array([np.sum(dos_levels * energy_levels * np.exp(-E_rr / (k_B * T))) for T in temperatures_K]) / Z
    U2 = np.array([np.sum(dos_levels * energy_levels ** 2 * np.exp(-E_rr / (k_B * T))) for T in temperatures_K]) / Z
    Cv = (U2 - U ** 2) / (k_B * temperatures_K ** 2)  # eV / K / atom
    return Cv

# ────────────────────────────────────────────────────────────────────
def plot_cv_surface(
    compositions: list[dict[str, float]],  # e.g. [{"Co":0.3, "Mn":0.5, "Fe":0.2}, ...]
    temperatures_K: np.ndarray,
    Cv_matrix: np.ndarray,
    *,
    mode: str = "surface",              # "surface" | "pcolormesh" | "scatter"
    cmap: str = "viridis",
    **plotly_kwargs                     # forwarded to go.Figure()
):
    """
    Interactive/-static visualisation of C_v(T, x).

    Parameters
    ----------
    ratios, temperatures_K, Cv_matrix
        Same meaning as before.
    mode
        "surface"        - a mesh surface (3-D)
        "pcolormesh"     - 2-D colour-map
        **"scatter"**    - coloured points at the grid nodes
    engine
        "mpl"     → Matplotlib backend
        "plotly"  → Plotly backend (interactive)
    plotly_kwargs
        Extra keyword args passed to go.Figure(**plotly_kwargs)
    """

    elem0, ratios = binary_comps_to_ratios(compositions)
    temperatures_K = np.asarray(temperatures_K)
    Cv_matrix = np.asarray(Cv_matrix)

    if Cv_matrix.shape != (ratios.size, temperatures_K.size):
        raise ValueError("Cv_matrix shape must be (len(ratios), len(T)).")
    # ----------------------------------------------------------------
    # PLOTLY (interactive)
    # ----------------------------------------------------------------
    # helper: convert a Matplotlib colormap → Plotly colourscale
    def _mpl_to_plotly(cm, n=256):
        return [[i/(n-1), f'rgb({int(r*255)},{int(g*255)},{int(b*255)})']
                for i, (r, g, b, _) in enumerate(cm(np.linspace(0, 1, n)))]
    cscale = _mpl_to_plotly(plt.get_cmap(cmap))

    if mode == "pcolormesh":
        T, X = np.meshgrid(temperatures_K, ratios)
        trace = go.Heatmap(x=T, y=X, z=Cv_matrix,
                            colorscale=cscale, colorbar_title=r"$C_v$")
    elif mode == "surface":
        T, X = np.meshgrid(temperatures_K, ratios)
        trace = go.Surface(x=T, y=X, z=Cv_matrix,
                            colorscale=cscale, colorbar_title=r"$C_v$")
    elif mode == "scatter":
        T, X = np.meshgrid(temperatures_K, ratios)
        trace = go.Scatter3d(
            x=T.ravel(), y=X.ravel(), z=Cv_matrix.ravel(),
            mode="markers",
            marker=dict(size=3, color=Cv_matrix.ravel(),
                        colorscale=cscale, colorbar=dict(title=r"$C_v$"))
        )
    else:
        raise ValueError("mode must be 'surface', 'pcolormesh' or 'scatter'")

    fig = go.Figure(data=[trace], **plotly_kwargs)
    fig.update_layout(
        scene=dict(
            xaxis_title="Temperature (K)",
            yaxis_title=f"{elem0} fraction x",
            zaxis_title=r"$C_v$ (eV K$^{-1}$ / supercell)",
        ),
        title=r"$C_v(T,x)$ - Wang-Landau",
        margin=dict(l=0, r=0, t=40, b=0)
    )
    return fig

def binary_comps_to_ratios(compositions):
    if any(len(comp) != 2 or set(comp.keys()) != set(compositions[0].keys()) for comp in compositions):
        raise ValueError("Compositions must be a list of 2 dicts with the same keys (elements).")
    elem0 = next(iter(compositions[0].keys()))

    ratios = np.asarray([comp[elem0] for comp in compositions])
    return elem0,ratios


# ═════════════════════════════════════════════════════════════════════════════
# 2)  Convenience - crunch a *list* of samplers in one go
# ═════════════════════════════════════════════════════════════════════════════
def compute_dos_matrix(
    data_list: Sequence[SamplerData],  # one WL sampler per composition
) -> np.ndarray:
    """
    Stack the **normalised** DOS of many samplers into a 2-D array
    with shape  (n_ratios, nbins).

    All samplers *must* share an identical energy window & bin size.
    """
    if len(data_list) == 0:
        raise ValueError("Need at least one sampler.")
    nbins = min([data.nbins for data in data_list])

    dos_rows = []
    for data in data_list:
        if nbins != data.nbins:
            raise ValueError("All samplers must have the same number of bins.")
        dos_rows.append(data.normalized_dos[:nbins])
    return np.vstack(dos_rows) # shape: (n_ratios, nbins)


# ═════════════════════════════════════════════════════════════════════════════
# 3)  2-D/3-D visualiser  (matplotlib OR plotly)
# ═════════════════════════════════════════════════════════════════════════════
def plot_dos_surface(
    compositions: list[dict[str, float]], # shape (n_ratios,)
    dos_matrix: np.ndarray, # shape (n_ratios, nbins)
    *,
    mode: str="surface",     # "surface" | "pcolormesh" | "scatter"
    cmap: str="viridis",
    **plotly_kwargs
):
    """
    Display DOS(ratio, bin_index) on a log₁₀ scale.  Zero-DOS cells are
    masked (shown transparent / skipped) so the logarithm is well-behaved.
    """

    elem0, ratios = binary_comps_to_ratios(compositions)
    dos_matrix = np.asarray(dos_matrix)
    if dos_matrix.shape[0] != ratios.size:
        raise ValueError("dos_matrix rows must equal len(ratios)")

    # ------------------------------------------------------------------
    # prepare grid & masked-log data
    # ------------------------------------------------------------------
    n_bins          = dos_matrix.shape[1]
    bin_indices     = np.arange(n_bins)                      # x-axis
    B, R            = np.meshgrid(bin_indices, ratios)       # 2-D grids
    dos_masked      = np.where(dos_matrix > 0, dos_matrix, np.nan)
    Zlog            = np.log10(dos_masked)                   # still has nans

    # ------------------------------------------------------------------
    # PLOTLY (interactive)
    # ------------------------------------------------------------------
    # helper: mpl->plotly colourscale
    def _mpl_to_plotly(cm, n=256):
        return [[i/(n-1),
                    f'rgb({int(r*255)},{int(g*255)},{int(b*255)})']
                for i, (r,g,b,_) in enumerate(cm(np.linspace(0,1,n)))]
    cscale = _mpl_to_plotly(plt.get_cmap(cmap))

    if mode == "pcolormesh":
        trace = go.Heatmap(
            x=bin_indices, y=ratios, z=Zlog,
            colorscale=cscale, colorbar_title="log₁₀ DOS"
        )
    elif mode == "surface":
        trace = go.Surface(
            x=bin_indices, y=ratios, z=Zlog,
            colorscale=cscale, colorbar_title="log₁₀ DOS"
        )
    elif mode == "scatter":
        trace = go.Scatter3d(
            x=B.ravel(), y=R.ravel(), z=Zlog.ravel(),
            mode="markers",
            marker=dict(size=3, color=Zlog.ravel(),
                        colorscale=cscale, colorbar=dict(title="log₁₀ DOS"))
        )
    else:
        raise ValueError("mode must be 'surface', 'pcolormesh' or 'scatter'")

    fig = go.Figure(data=[trace], **plotly_kwargs)
    fig.update_layout(
        scene=dict(
            xaxis_title="Bin index",
            yaxis_title=f"{elem0} fraction x",
            zaxis_title="log₁₀ DOS",
        ),
        title="Density-of-states  log₁₀DOS(idx, x)",
        margin=dict(l=0, r=0, t=40, b=0)
    )
    return fig


# ──────────────────────────────────────────────────────────────────────
# 1)  Per-sampler helper (already shown earlier; keep if you have it)
# ──────────────────────────────────────────────────────────────────────
def compute_histogram_per_bin(data: SamplerData) -> np.ndarray:
    counts  = np.zeros(data.nbins, dtype=int)
    # use the ENTIRE enthalpy trace, not the grid
    idx = ((data.enthalpy_trace - data.min_E) // data.bin_size).astype(int)
    idx = idx[(0 <= idx) & (idx < data.nbins)]
    np.add.at(counts, idx, 1)
    return counts


def compute_histogram_matrix(data_list: Sequence[SamplerData]):
    """Area is the non-zero entropy counts, value is the convergence exponent.
    shape (n_ratios, nbins)."""
    if len(data_list) == 0:
        raise ValueError("Need at least one sampler.")
    nbins = data_list[0].nbins

    rows = []
    for data in data_list:
        if data.nbins != nbins:
            raise ValueError("All samplers must share the same bin grid.")
        # Add 1e-6 so that log10(1) is not 0, which doesn't show up in the plot.
        rows.append((1e-6 - np.log10(data.mod_factor_trace[-1])) * (data.histogram > 0))
    return np.vstack(rows) # (n_ratios, nbins)


# ──────────────────────────────────────────────────────────────────────
# 2-bis)  2-D heat-map of *raw counts* (no log scale)
# ──────────────────────────────────────────────────────────────────────
def plot_hist_heatmap(
    compositions: list[dict[str, float]],  # e.g. [{"Co":0.5, "Mn":0.5}, ...]
    hist_matrix,
    *,
    cmap="viridis",
    mask_zeros=True,
    clim=(0, 8),                       # <‑‑ add a colour‑limit kwarg
):
    """
    Show the raw number of kept configurations in each WL energy bin.
    …

    clim : (low, high)
        Fix colour scale to this range; pass None for default autoscaling.
    """
    elem0, ratios = binary_comps_to_ratios(compositions)
    hist_matrix = np.asarray(hist_matrix)
    if hist_matrix.shape[0] != ratios.size:
        raise ValueError("hist_matrix rows must equal len(ratios)")

    B, R = np.meshgrid(np.arange(hist_matrix.shape[1]), ratios)

    Z = hist_matrix.astype(float)
    if mask_zeros:
        Z = np.where(Z > 0, Z, np.nan)        # leave zeros transparent

    # ---------------- MATPLOTLIB ----------------
    fig, ax = plt.subplots(figsize=(7, 4))

    # build a Normalizer or fall back to auto range
    norm = None
    if clim is not None:
        vmin, vmax = clim
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    pcm = ax.pcolormesh(B, R, Z, shading="auto", cmap=cmap, norm=norm)             # <‑‑ fixed colour limits
    fig.colorbar(pcm, ax=ax, label="count / bin")
    ax.set(
        xlabel="Bin index",
        ylabel=f"{elem0} fraction $x$",
        title="Convergence histogram",
    )
    fig.tight_layout()
    return fig

def get_cation_index_map(ensemble) -> dict[str, int]:
    """
    Return {element_symbol: integer_index} in the *same order* that SMOL
    uses to encode the occupancy vector, but without relying on the
    newer `ClusterSubspace.site_spaces` attribute.
    """
    struct = ensemble.processor.structure

    # `get_allowed_species(struct)` returns a tuple for every crystallographic
    # site.  The first tuple with >1 entries is our cation sub‑lattice.
    for species_tuple in get_allowed_species(struct):
        if len(species_tuple) > 1:
            return {str(sp): idx for idx, sp in enumerate(species_tuple)}

    raise RuntimeError("No multicomponent site-space found in this ensemble.")

def sample_configs_fast(
    ensemble      : Ensemble,
    rng           : Generator,
    composition   : dict[str, float], # e.g. {"Co":0.3, "Mn":0.5, "Fe":0.2}
    *,
    n_samples : int = 100,
) -> np.ndarray:
    """
    Draw random occupancies matching the requested composition and return the
    corresponding CE energies.  Works for any number of species on the cation
    sub-lattice.
    """
    elem2idx = get_cation_index_map(ensemble)

    # ----- basic sanity --------------------------------------------------
    if not math.isclose(sum(composition.values()), 1.0, abs_tol=1e-6):
        raise ValueError("Fractions must sum to 1.")
    if set(composition) - set(elem2idx):
        raise KeyError("Composition contains elements not present in the site-space.")

    # ----- figure out how many sites of each element we need -------------
    cat_idx = np.array(
        [i for i, sp in enumerate(get_allowed_species(ensemble.processor.structure)) if len(sp) > 1], dtype=np.int32,
    )
    n_cations     = cat_idx.size
    target_counts = {el: round(frac * n_cations) for el, frac in composition.items()}

    # correct rounding so the total matches exactly
    delta = n_cations - sum(target_counts.values())
    if delta:                                   # distribute leftovers
        # give the last |delta| elements +1 (or −1) until the sum is right
        for el in list(target_counts)[:abs(delta)]:
            target_counts[el] += int(math.copysign(1, delta))

    if sum(target_counts.values()) != n_cations:
        raise RuntimeError("Target counts do not match the number of cation sites.")

    # ----- sampling loop -------------------------------------------------
    n_sites = ensemble.num_sites
    ce_E = []

    for _ in range(n_samples):
        occ = np.full(n_sites, -1, dtype=np.int32)   # -1 = untouched
        free_idx = cat_idx.copy()

        # place each element in turn
        for el, n_el in target_counts.items():
            chosen = rng.choice(free_idx, n_el, replace=False)
            occ[chosen] = elem2idx[el]
            free_idx = free_idx[~np.in1d(free_idx, chosen)]

        # defensive check
        if (occ[cat_idx] < 0).any():
            raise RuntimeError("Not all cation sites were assigned.")

        # --- CE energy
        corr = ensemble.compute_feature_vector(occ)
        E_sup = float(ensemble.natural_parameters @ corr)
        ce_E.append(E_sup)

    return np.asarray(ce_E)


def init_worker():
    import sys
    sys.stdout.reconfigure(line_buffering=True) # type: ignore


def determine_wl_window(
    *,
    composition: dict[str, float], # e.g. {"Co":0.3, "Mn":0.5, "Fe":0.2}
    n_samples_per_site: int,
    snapshot_counts: int,
    half_window: int,
    seed: int,
    rng: Generator,
    E_bin_per_supercell_eV: float,
    ensemble: Ensemble,
    minimum_bins: int = 100,
) -> Sampler:
    """
    Find an energy window that contains at least `minimum_bins` visited
    WL bins and is safely away from the edges.  Returns the converged
    Wang-Landau `Sampler`.
    """
    # ── initial mu and symmetric window ────────────────────────────────
    rand_E = sample_configs_fast(ensemble, rng, composition=composition, n_samples=10_000)
    mu = float(rand_E.mean())
    half_W_eV = half_window * E_bin_per_supercell_eV
    window = (mu - half_W_eV, mu + half_W_eV)

    local_rng = np.random.default_rng(seed)

    # ── adaptive shrink-until-converged loop ──────────────────────────
    while True:
        # ----- (a) initialise and run one WL pass ---------------------
        sampler = initialize_wl_sampler(ensemble, num_bins=2*half_window, seeds=[int(local_rng.integers(1<<32))], window=window)
        occ_enc = initialize_supercell_occupancy(ensemble, local_rng, composition)
        nsamples = int(n_samples_per_site * ensemble.num_sites)
        thin_by = max(1, math.ceil(nsamples / snapshot_counts))
        sampler.run(nsamples, occ_enc, thin_by=thin_by, progress=False)

        # ----- (b) coverage analysis ---------------------------------
        hist = sampler.samples.get_trace_value("histogram")[-1]
        mask = hist > 0
        first = np.argmax(mask)
        last = len(mask) - np.argmax(mask[::-1]) - 1
        active = last - first + 1

        if first < 10 or last > len(mask) - 10:
            raise ValueError(f"{tc.dataset.comp_str(composition)}  [{first},{last}] window too narrow")

        if active < minimum_bins:
            print(f"{tc.dataset.comp_str(composition)}  [{first},{last}] {active} bins -> shrink window.")
            half_W_eV *= 0.7
            window     = (mu - half_W_eV, mu + half_W_eV)
            continue

        print(f"{tc.dataset.comp_str(composition)}  [{first},{last}] {active} bins -> converged.")
        return sampler


def determine_wl_windows(
    n_samples_per_site: int,
    snapshot_counts: int,
    half_window: int,
    nprocs: int,
    rng: Generator,
    compositions: list[dict[str, float]],
    E_bin_per_supercell_eV: float,
    ensemble: Ensemble,
    *,
    minimum_bins: int = 100,
):
    """
    Run `determine_wl_window` for every composition in `ratios`
    using joblib for parallelism.  Returns a list of converged samplers
    in the same order as `ratios`.
    """
    print(f"Starting Wang-Landau window search on {nprocs} processes …")

    root_seq = np.random.SeedSequence(42)
    seeds = root_seq.generate_state(len(compositions))

    samplers = Parallel(n_jobs=nprocs, backend="loky", initializer=init_worker)(
        delayed(determine_wl_window)(
            composition=composition,
            n_samples_per_site=n_samples_per_site,
            snapshot_counts=snapshot_counts,
            half_window=half_window,
            seed=int(seed),
            rng=rng,
            E_bin_per_supercell_eV=E_bin_per_supercell_eV,
            ensemble=ensemble,
            minimum_bins=minimum_bins,
        )
        for seed, composition in zip(seeds, compositions)
    )

    return samplers
