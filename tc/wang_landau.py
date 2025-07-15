import math
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from ase.build import bulk
from numpy.random import Generator
from pymatgen.io.ase import AseAtomsAdaptor
from smol.cofe import ClusterSubspace
from smol.moca import Ensemble, Sampler
from smol.cofe.space.domain import get_allowed_species

import tc

# =====================================================================
# Public API
# =====================================================================

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

    random_samples = sample_configs_fast(ensemble, rng, n_samples=10_000, ratio=ratio)

    # ------------------------------------------------------------
    # 1) Define the WL energy window (per primitive cell)
    # ------------------------------------------------------------
    mu, sigma = random_samples.mean(), random_samples.std()
    window_low, window_high = window
    min_E = mu - window_low * sigma
    max_E = mu + window_high * sigma
    bin_size = (max_E - min_E) / num_bins
    max_E = min_E + num_bins * bin_size  # ensure max_E is exact to avoid rounding issues
    print(f"Energy window : [{min_E:.3f}, {max_E:.3f}] eV ({num_bins} bins, {bin_size:.4f} eV each)")

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

# =====================================================================
# Helper functions
# =====================================================================

def generate_wl_plots(
    sampler: Sampler,
    temperatures: np.ndarray,
    Cv: np.ndarray,
) -> None:
    """Create a 4-panel figure (2 × 2)."""

    # ------------------------------------------------------------------
    # Figure / axes layout
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(
        2, 2, figsize=(10, 8), sharex=False, constrained_layout=True
    )
    ax_conv, ax_dos   = axes[0]        # first row
    ax_cv,   ax_hist  = axes[1]        # second row

    # ==================================================================
    # (1) WL convergence  (top-left)
    # ==================================================================
    mod_factor = sampler.samples.get_trace_value("mod_factor")
    ax_conv.semilogy(mod_factor, ".-")
    ax_conv.set_xlabel("Iteration")
    ax_conv.set_ylabel("Modification factor")
    ax_conv.set_title("WL convergence")

    # ==================================================================
    # (2) Density of states          (top-right)
    # ==================================================================
    entropy = sampler.samples.get_trace_value("entropy")[-1]
    nbins   = entropy.size
    kernel = sampler.mckernels[0]
    bin_centers = kernel.spec.min_enthalpy + (np.arange(nbins) + 0.5) * kernel.bin_size

    mask = entropy > 0
    S_shift = entropy[mask] - entropy[mask].min()
    dos = np.exp(S_shift - S_shift.max())
    dos /= dos.sum()

    ax_dos.semilogy(bin_centers[mask], dos, ".-")
    ax_dos.set_xlabel(r"$E$ (eV / supercell)")
    ax_dos.set_ylabel("Density of states")
    ax_dos.set_title("WL DOS estimate")
    ax_dos.set_xlim(kernel.spec.min_enthalpy, kernel.spec.max_enthalpy)
    ax_dos.legend()

    # ==================================================================
    # (3) Heat-capacity curve        (bottom-left)
    # ==================================================================
    ax_cv.plot(temperatures, Cv, lw=2)
    ax_cv.set_xlabel("Temperature (K)")
    ax_cv.set_ylabel(r"$C_v$ per supercell (eV K$^{-1}$)")
    ax_cv.set_title("Wang-Landau $C_v(T)$")

    # ==================================================================
    # (4) Histogram of *kept* configs (bottom-right)
    # ==================================================================
    # 4a) obtain energies of trace & map to bins
    energies = sampler.samples.get_trace_value("enthalpy")

    # helper: convert an enthalpy value to its bin number
    idx = lambda E: int((E - kernel.spec.min_enthalpy) // kernel.bin_size)

    counts   = np.zeros(nbins, dtype=int)
    for E in energies:
        b = idx(E)
        if 0 <= b < nbins:          # ignore any value outside the WL window
            counts[b] += 1

    ax_hist.bar(bin_centers, counts, width=0.9*kernel.bin_size, align="center", edgecolor="k")
    ax_hist.set_xlabel(r"$E$ (eV / supercell)")
    ax_hist.set_ylabel("# kept configs")
    ax_hist.set_title("Trace occupancy per energy bin")
    ax_hist.set_xlim(kernel.spec.min_enthalpy, kernel.spec.max_enthalpy)

    # annotate: unique occupancies vs. total kept
    occs      = sampler.samples.get_trace_value("occupancy")
    n_unique  = len({bytes(o) for o in occs})
    ax_hist.annotate(
        f"unique configs: {n_unique}/{len(occs)}",
        xy=(0.98, 0.95), xycoords="axes fraction",
        ha="right", va="top", fontsize=10,
        bbox=dict(boxstyle="round,pad=0.25", fc="w")
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
    sampler: Sampler,
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
    entropy = sampler.samples.get_trace_value("entropy")[-1]
    mask = entropy > 0
    ent_ref = entropy[mask] - entropy[mask].min()
    dos_levels = np.exp(ent_ref - ent_ref.max())
    dos_levels /= dos_levels.sum()
    energy_levels = sampler.mckernels[0].levels

    # Compute thermodynamic properties
    k_B = 8.617333262e-5  # eV / K
    E_rr = energy_levels - energy_levels.min()

    Z = np.array([np.sum(dos_levels * np.exp(-E_rr / (k_B * T))) for T in temperatures_K])
    U = np.array([np.sum(dos_levels * energy_levels * np.exp(-E_rr / (k_B * T))) for T in temperatures_K]) / Z
    U2 = np.array([np.sum(dos_levels * energy_levels ** 2 * np.exp(-E_rr / (k_B * T))) for T in temperatures_K]) / Z
    Cv = (U2 - U ** 2) / (k_B * temperatures_K ** 2)  # eV / K / atom
    return Cv

# ── tc/wang_landau.py  (add near the bottom) ───────────────────────────
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# ────────────────────────────────────────────────────────────────────
def plot_cv_surface(
    ratios: np.ndarray,
    temperatures_K: np.ndarray,
    Cv_matrix: np.ndarray,
    *,
    mode: str = "surface",              # "surface" | "pcolormesh" | "scatter"
    cmap: str = "viridis",
    engine: str = "mpl",                # "mpl" | "plotly"
    **plotly_kwargs                     # forwarded to go.Figure()
):
    """
    Interactive/-static visualisation of C_v(T, x).

    Parameters
    ----------
    ratios, temperatures_K, Cv_matrix
        Same meaning as before.
    mode
        "surface"        – a mesh surface (3-D)
        "pcolormesh"     – 2-D colour-map
        **"scatter"**    – coloured points at the grid nodes
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
    # MATPLOTLIB
    # ----------------------------------------------------------------
    if engine == "mpl":
        if mode == "pcolormesh":
            fig, ax = plt.subplots(figsize=(7, 4))
            T, X = np.meshgrid(temperatures_K, ratios)
            pcm = ax.pcolormesh(T, X, Cv_matrix, cmap=cmap, shading="auto")
            fig.colorbar(pcm, ax=ax, label=r"$C_v$ (eV K$^{-1}$ / supercell)")
            ax.set(xlabel="Temperature (K)", ylabel="Mg fraction $x$",
                   title=r"$C_v(T,x)$ – Wang-Landau")
        elif mode == "surface":
            fig = plt.figure(figsize=(7, 5))
            ax  = fig.add_subplot(111, projection="3d")
            T, X = np.meshgrid(temperatures_K, ratios)
            ax.plot_surface(T, X, Cv_matrix, cmap=cmap,
                            rstride=1, cstride=1, antialiased=True)
            ax.set(xlabel="Temperature (K)", ylabel="Mg fraction $x$",
                   zlabel=r"$C_v$ (eV K$^{-1}$ / supercell)",
                   title=r"$C_v(T,x)$ – Wang-Landau")
            ax.view_init(elev=25, azim=-60)
        elif mode == "scatter":
            fig = plt.figure(figsize=(7, 5))
            ax  = fig.add_subplot(111, projection="3d")
            T, X = np.meshgrid(temperatures_K, ratios)
            sc = ax.scatter(T.ravel(), X.ravel(), Cv_matrix.ravel(),
                            c=Cv_matrix.ravel(), cmap=cmap, s=15)
            fig.colorbar(sc, ax=ax, label=r"$C_v$ (eV K$^{-1}$ / supercell)")
            ax.set(xlabel="Temperature (K)", ylabel="Mg fraction $x$",
                   zlabel=r"$C_v$ (eV K$^{-1}$ / supercell)",
                   title=r"$C_v(T,x)$ – Wang-Landau (scatter)")
            ax.view_init(elev=25, azim=-60)
        else:
            raise ValueError("mode must be 'surface', 'pcolormesh' or 'scatter'")
        plt.tight_layout()
        return fig  # caller can `.show()` or further tweak

    # ----------------------------------------------------------------
    # PLOTLY (interactive)
    # ----------------------------------------------------------------
    if engine == "plotly":
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

    raise ValueError("engine must be 'mpl' or 'plotly'")

# ═════════════════════════════════════════════════════════════════════════════
# 1)  Per-sampler helper – extract a normalised DOS *per bin index*
# ═════════════════════════════════════════════════════════════════════════════
def compute_dos_per_bin(sampler) -> np.ndarray:
    """
    Return the *normalised* DOS for a single Wang–Landau `sampler`,
    laid out on the *bin index* grid (length == nbins).

    Any zero–entropy bins are left at exactly zero so they may be
    masked later for log plotting.
    """
    entropy = sampler.samples.get_trace_value("entropy")[-1]
    # zero/negative entropy → unvisited bin
    mask = entropy > 0
    S_shift       = entropy[mask] - entropy[mask].min()
    dos_normed    = np.zeros_like(entropy, dtype=float)
    dos_normed[mask] = np.exp(S_shift - S_shift.max())
    # global normalisation (Σ DOS = 1)
    dos_normed   /= dos_normed.sum()
    return dos_normed


# ═════════════════════════════════════════════════════════════════════════════
# 2)  Convenience – crunch a *list* of samplers in one go
# ═════════════════════════════════════════════════════════════════════════════
def compute_dos_matrix(
    samplers: Sequence,              # one WL sampler per composition
) -> np.ndarray:
    """
    Stack the **normalised** DOS of many samplers into a 2-D array
    with shape  (n_ratios, nbins).

    All samplers *must* share an identical energy window & bin size.
    """
    if len(samplers) == 0:
        raise ValueError("Need at least one sampler.")
    nbins = samplers[0].samples.get_trace_value("entropy")[-1].size

    dos_rows = []
    for i, s in enumerate(samplers):
        ent = s.samples.get_trace_value("entropy")[-1]
        if ent.size != nbins:
            raise ValueError("All samplers must have the same number of bins.")
        dos_rows.append(compute_dos_per_bin(s))
    return np.vstack(dos_rows)                # shape: (n_ratios, nbins)


# ═════════════════════════════════════════════════════════════════════════════
# 3)  2-D/3-D visualiser  (matplotlib OR plotly)
# ═════════════════════════════════════════════════════════════════════════════
def plot_dos_surface(
    ratios          : np.ndarray,            # shape (n_ratios,)
    dos_matrix      : np.ndarray,            # shape (n_ratios, nbins)
    *,
    mode            : str   = "surface",     # "surface" | "pcolormesh" | "scatter"
    engine          : str   = "mpl",         # "mpl" | "plotly"
    cmap            : str   = "viridis",
    **plotly_kwargs                       # forwarded to go.Figure()
):
    """
    Display DOS(ratio, bin_index) on a log₁₀ scale.  Zero-DOS cells are
    masked (shown transparent / skipped) so the logarithm is well-behaved.
    """

    ratios     = np.asarray(ratios)
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
    # MATPLOTLIB
    # ------------------------------------------------------------------
    if engine == "mpl":
        if mode == "pcolormesh":
            fig, ax = plt.subplots(figsize=(7, 4))
            pcm = ax.pcolormesh(B, R, Zlog, cmap=cmap, shading="auto")
            cb  = fig.colorbar(pcm, ax=ax, label=r"log$_{10}$ DOS")
            ax.set(xlabel="Bin index", ylabel="Mg fraction $x$",
                   title="Density-of-states  log₁₀DOS(idx, x)")
            plt.tight_layout()
            return fig

        # 3-D variants
        fig = plt.figure(figsize=(7, 5))
        ax  = fig.add_subplot(111, projection="3d")

        if mode == "surface":
            ax.plot_surface(B, R, Zlog, cmap=cmap,
                            rstride=1, cstride=1, antialiased=True)
        elif mode == "scatter":
            ax.scatter(B.ravel(), R.ravel(), Zlog.ravel(), c=Zlog.ravel(),
                       cmap=cmap, s=10)
        else:
            raise ValueError("mode must be 'surface', 'pcolormesh' or 'scatter'")

        ax.set(xlabel="Bin index",
               ylabel="Mg fraction $x$",
               zlabel=r"log$_{10}$ DOS",
               title="Density-of-states  log₁₀DOS(idx, x)")
        ax.view_init(elev=25, azim=-60)
        plt.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # PLOTLY (interactive)
    # ------------------------------------------------------------------
    if engine == "plotly":
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

    raise ValueError("engine must be 'mpl' or 'plotly'")


# ──────────────────────────────────────────────────────────────────────
# 1)  Per-sampler helper (already shown earlier; keep if you have it)
# ──────────────────────────────────────────────────────────────────────
def compute_histogram_per_bin(sampler) -> np.ndarray:
    kernel      = sampler.mckernels[0]
    min_E       = kernel.spec.min_enthalpy
    bin_size    = kernel.bin_size
    nbins       = kernel._entropy.size

    counts      = np.zeros(nbins, dtype=int)
    energies    = sampler.samples.get_trace_value("enthalpy")
    idx         = ((energies - min_E) // bin_size).astype(int)
    idx         = idx[(0 <= idx) & (idx < nbins)]
    np.add.at(counts, idx, 1)
    return counts


def compute_histogram_matrix(samplers):
    """Stack per-bin counts → shape (n_ratios, nbins)."""
    if len(samplers) == 0:
        raise ValueError("Need at least one sampler.")
    nbins = samplers[0].mckernels[0]._entropy.size

    rows = []
    for i, s in enumerate(samplers):
        if s.mckernels[0]._entropy.size != nbins:
            raise ValueError("All samplers must share the same bin grid.")
        rows.append(compute_histogram_per_bin(s))
    return np.vstack(rows)                           # (n_ratios, nbins)


# ──────────────────────────────────────────────────────────────────────
# 2-bis)  2-D heat-map of *raw counts* (no log scale)
# ──────────────────────────────────────────────────────────────────────
def plot_hist_heatmap(
    ratios, hist_matrix,
    *,
    engine="mpl",              # "mpl" | "plotly"
    cmap="viridis",
    mask_zeros=True,           # set False if you want zeros coloured
    **plotly_kwargs
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
    import numpy as np, matplotlib.pyplot as plt
    import plotly.graph_objects as go

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
    if engine == "mpl":
        fig, ax = plt.subplots(figsize=(7, 4))
        pcm = ax.pcolormesh(B, R, Z, shading="auto", cmap=cmap)
        fig.colorbar(pcm, ax=ax, label="# kept configs")
        ax.set(xlabel="Bin index", ylabel="Mg fraction $x$",
               title="# kept configurations (raw counts)")
        plt.tight_layout()
        return fig

    # ---------------- PLOTLY ----------------
    if engine == "plotly":
        def _mpl_to_plotly(cm, n=256):
            return [[i/(n-1),
                     f'rgb({int(r*255)},{int(g*255)},{int(b*255)})']
                    for i,(r,g,b,_) in enumerate(cm(np.linspace(0,1,n)))]
        cscale = _mpl_to_plotly(plt.get_cmap(cmap))

        trace = go.Heatmap(
            x=bin_idx, y=ratios, z=Z,
            colorscale=cscale, colorbar_title="# kept configs"
        )
        fig = go.Figure(data=[trace], **plotly_kwargs)
        fig.update_layout(
            xaxis_title="Bin index",
            yaxis_title="Mg fraction x",
            title="# kept configurations (raw counts)",
            margin=dict(l=0, r=0, t=40, b=0)
        )
        return fig

    raise ValueError("engine must be 'mpl' or 'plotly'")

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
            if len(sp) > 1                          # >1 allowed species ⇒ cation site
        ],
        dtype=np.int32,
    )
    n_cations = cat_idx.size
    n_A = round(n_cations * ratio)
    n_B = n_cations - n_A
    n_sites   = ensemble.num_sites

    print(f"Sampling {n_samples} configurations with {n_A} Li and {n_B} Mn...")
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
    print(f"ratio: {ratio} CE energies: mean = {1000 * ce_E.mean():8.2f} meV, std = {1000 * ce_E.std():8.2f} meV, min = {1000 * ce_E.min():8.2f} meV, max = {1000 * ce_E.max():8.2f} meV")
    return ce_E