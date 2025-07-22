from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from ase.build import bulk
from numpy.random import Generator
from pymatgen.io.ase import AseAtomsAdaptor
from smol.cofe import ClusterSubspace
from smol.cofe.space.domain import get_allowed_species
from smol.moca import Ensemble, Sampler
from tc.sampler_data import SamplerData

import plotly.graph_objects as go
import math
from joblib import Parallel, delayed

# =====================================================================
# Public API
# =====================================================================

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
    rng: Generator,
    ratio: float,
    num_bins: int,
    flatness: float = 0.8,
    seeds: Sequence[int],
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

def initialize_supercell_occupancy(
    ensemble: Ensemble,
    rng: Generator,
    replace_element: str,
    new_elements: tuple[str, str],
    ratio: float,
) -> np.ndarray:
    """Encode the initial random snapshot at the requested composition."""

    n_sc = round(ensemble.processor.size ** (1 / 3))
    if n_sc ** 3 != ensemble.processor.size:
        raise ValueError("Ensemble supercell is not cubic.")

    sc_tuple = (n_sc, n_sc, n_sc)
    sc_mat = np.diag(sc_tuple)

    conv_cell = bulk("MgO", crystalstructure="rocksalt", a=4.2, cubic=True)
    snapshot = conv_cell * sc_tuple

    # Populate the cation sub-lattice
    repl_idx = [i for i, at in enumerate(snapshot) if at.symbol == replace_element]
    rng.shuffle(repl_idx)
    n_A = int(round(ratio * len(repl_idx)))
    A, B = new_elements
    snapshot.symbols[repl_idx] = B
    snapshot.symbols[repl_idx[:n_A]] = A

    subspace: ClusterSubspace = ensemble.processor.cluster_subspace
    pmg_struct = AseAtomsAdaptor.get_structure(snapshot)
    occ_enc = subspace.occupancy_from_structure(pmg_struct, scmatrix=sc_mat, encode=True)

    return occ_enc.astype(np.int32) # type: ignore


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
    ratios: np.ndarray,
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
    ratios         = np.asarray(ratios)
    temperatures_K = np.asarray(temperatures_K)
    Cv_matrix      = np.asarray(Cv_matrix)

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
            yaxis_title="Mg fraction x",
            zaxis_title=r"$C_v$ (eV K$^{-1}$ / supercell)",
        ),
        title=r"$C_v(T,x)$ – Wang-Landau",
        margin=dict(l=0, r=0, t=40, b=0)
    )
    return fig


# ═════════════════════════════════════════════════════════════════════════════
# 2)  Convenience – crunch a *list* of samplers in one go
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
    ratios          : np.ndarray,            # shape (n_ratios,)
    dos_matrix      : np.ndarray,            # shape (n_ratios, nbins)
    *,
    mode            : str   = "surface",     # "surface" | "pcolormesh" | "scatter"
    cmap            : str   = "viridis",
    **plotly_kwargs                       # forwarded to go.Figure()
):
    """
    Display DOS(ratio, bin_index) on a log₁₀ scale.  Zero-DOS cells are
    masked (shown transparent / skipped) so the logarithm is well-behaved.
    """

    ratios = np.asarray(ratios)
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
            yaxis_title="Mg fraction x",
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
    """Stack per-bin counts → shape (n_ratios, nbins)."""
    if len(data_list) == 0:
        raise ValueError("Need at least one sampler.")
    nbins = data_list[0].nbins

    rows = []
    for data in data_list:
        if data.nbins != nbins:
            raise ValueError("All samplers must share the same bin grid.")
        rows.append(data.histogram)
    return np.vstack(rows) # (n_ratios, nbins)


# ──────────────────────────────────────────────────────────────────────
# 2-bis)  2-D heat-map of *raw counts* (no log scale)
# ──────────────────────────────────────────────────────────────────────
def plot_hist_heatmap(
    ratios, hist_matrix,
    *,
    cmap="viridis",
    mask_zeros=True,           # set False if you want zeros coloured
):
    """
    Show the **raw number** of kept configurations in each WL energy bin.

    Parameters
    ----------
    ratios : 1-D array
        Composition values (one per WL sampler).
    hist_matrix : 2-D int array
        Rows: ratios;  columns: bin indices.
    engine : {"mpl", "plotly"}
        Matplotlib (static) or Plotly (interactive) backend.
    cmap : str
        Matplotlib colormap name.
    mask_zeros : bool
        If True (default) bins with 0 counts are transparent / NaN.
    """
    ratios      = np.asarray(ratios)
    hist_matrix = np.asarray(hist_matrix)
    if hist_matrix.shape[0] != ratios.size:
        raise ValueError("hist_matrix rows must equal len(ratios)")

    n_bins      = hist_matrix.shape[1]
    bin_idx     = np.arange(n_bins)
    B, R        = np.meshgrid(bin_idx, ratios)

    Z = hist_matrix.astype(float)
    if mask_zeros:
        Z = np.where(Z > 0, Z, np.nan)       # leave zeros transparent

    # ---------------- MATPLOTLIB ----------------
    fig, ax = plt.subplots(figsize=(7, 4))
    pcm = ax.pcolormesh(B, R, Z, shading="auto", cmap=cmap)
    fig.colorbar(pcm, ax=ax, label="# kept configs")
    ax.set(xlabel="Bin index", ylabel="Mg fraction $x$", title="# kept configurations (raw counts)")
    plt.tight_layout()
    return fig


def sample_configs_fast(
    ensemble: Ensemble,
    rng: Generator,
    ratio: float,
    *,
    n_samples: int = 100,
) -> np.ndarray:
    """
    Compare CE (via Ensemble.processor.compute_property) to MACE on random configs.
    Follows the 'WL sanity check' recipe exactly.
    """

    N_sc = round(ensemble.processor.size ** (1/3))
    if N_sc**3 != ensemble.processor.size:
        raise ValueError(f"Supercell size {ensemble.processor.size} is not a perfect cube.")
    cat_idx = np.array(
        [
            i for i, sp in enumerate(get_allowed_species(ensemble.processor.structure))
            if len(sp) > 1 # type: ignore # >1 allowed species ⇒ cation site
        ],
        dtype=np.int32,
    )
    n_cations = cat_idx.size
    n_A = round(n_cations * ratio)
    n_B = n_cations - n_A
    n_sites   = ensemble.num_sites

    # print(f"Sampling {n_samples} configurations with {n_A} Li and {n_B} Mn...")
    ce_E = []
    for _ in range(n_samples):
        occ = np.ones(n_sites, dtype=np.int32)
        li_sites = rng.choice(cat_idx, round(n_cations * ratio), replace=False)
        occ[li_sites] = 0

        # ---- sanity checks ---------------------------------------
        n_A_actual = (occ[cat_idx] == 0).sum()
        n_B_actual = (occ[cat_idx] == 1).sum()
        if (n_A != n_A_actual) or (n_B != n_B_actual):
            raise ValueError(f"Expected {n_A} Li and {n_B} Mn, but got {n_A_actual} Li and {n_B_actual} Mn.")

        # ---- CE energy ------------------------------------------
        corr  = ensemble.compute_feature_vector(occ)              # raw counts
        E_sup = float(ensemble.natural_parameters @ corr)         # eV / cell
        ce_E.append(E_sup)

    # print the mean, std dev, min, and max of the CE energies
    ce_E = np.array(ce_E)
    # print(f"ratio: {ratio} CE energies: mean = {1000 * ce_E.mean():8.2f} meV, std = {1000 * ce_E.std():8.2f} meV, min = {1000 * ce_E.min():8.2f} meV, max = {1000 * ce_E.max():8.2f} meV")
    return ce_E


def _single_wl(
        ratio: float,
        seed: int,
        *,
        ensemble: Ensemble,
        num_bins: int,
        window: tuple[float, float],
        replace_element: str,
        new_elements: tuple[str, str],
        snapshot_counts: int,
        n_samples_per_site: int,
):
    rng = np.random.default_rng(seed)
    sampler = initialize_wl_sampler(ensemble, rng=rng, ratio=ratio, num_bins=num_bins, seeds=[seed], window=window)
    occ_enc = initialize_supercell_occupancy(ensemble, rng, replace_element, new_elements, ratio)
    nsamples = int(n_samples_per_site * ensemble.num_sites)
    thin_by  = max(1, math.ceil(nsamples / snapshot_counts))
    sampler.run(nsamples, occ_enc, thin_by=thin_by, progress=False)
    return sampler

def determine_wl_window(n_samples_per_site, snapshot_counts, half_window, nprocs, rng, replace_element, new_elements, ratios, E_bin_per_supercell_eV, ensemble):
    print(f"Starting Wang-Landau window searching over {nprocs} processes…")
    windows = []
    for ratio in ratios:
        random_samples = sample_configs_fast(ensemble, rng, n_samples=10_000, ratio=ratio)
        mu = random_samples.mean()
        windows.append((mu - half_window * E_bin_per_supercell_eV,
                        mu + half_window * E_bin_per_supercell_eV))

    seed_root  = np.random.SeedSequence(42)
    child_seeds = [int(s) for s in seed_root.generate_state(len(ratios))]
    samplers   = [None] * len(ratios)
    while any(s is None for s in samplers):
        todo_idx = [i for i, s in enumerate(samplers) if s is None]

        print(f"WL level: running {len(todo_idx)} / {len(ratios)} ratios …")

        # --- 1. launch only the missing jobs -------------------------
        new_samplers = Parallel(n_jobs=nprocs, backend="loky", max_nbytes=None)(
            delayed(_single_wl)(
                ratios[i],
                child_seeds[i],
                ensemble = ensemble,
                num_bins = half_window * 2,  # or better: derive from windows[i]
                window = windows[i],
                replace_element = replace_element,
                new_elements = new_elements,
                snapshot_counts = snapshot_counts,
                n_samples_per_site = n_samples_per_site
            )
            for i in todo_idx
        )

        # --- 2. analyse and decide which ones are done ---------------
        for local_j, i in enumerate(todo_idx):
            sampler = new_samplers[local_j] #type: ignore
            entropy = sampler.samples.get_trace_value("entropy")[-1] # type: ignore

            first = np.argmax(entropy > 0)
            last  = len(entropy) - np.argmax(entropy[::-1] > 0) - 1
            active = last - first + 1

            if first < 10 or last >= len(entropy) - 10:
                raise ValueError(f"Energy window too narrow for x={ratios[i]:.3f}")

            if active < 50:
                print(f" x={ratios[i]:.3f}: only {active} active bins → shrink window.")
                w0, w1 = windows[i]
                width = w1 - w0
                center = 0.5*(w0 + w1)
                factor = 0.5
                windows[i] = (center - 0.5*width*factor, center + 0.5*width*factor)
            else:
                print(f" x={ratios[i]:.3f}: {active} bins, converged.")
                samplers[i] = sampler
    return samplers
