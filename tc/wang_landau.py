import math
from typing import Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from ase.build import bulk
from numpy.random import Generator
from pymatgen.io.ase import AseAtomsAdaptor
from smol.cofe import ClusterSubspace
from smol.moca import Ensemble, Sampler

# =====================================================================
# Public API
# =====================================================================

def run_wang_landau(
    ensemble: Ensemble,
    samples: np.ndarray,
    rng: Generator,
    *,
    replace_element: str = "Mg",
    new_elements: Tuple[str, str] = ("Mg", "Fe"),
    ratio: float = 0.50,
    num_bins: int = 100,
    flatness: float = 0.8,
    seeds: Sequence[int] = (123,),
    thin_target: int = 500,
    n_samples_per_site: int = 100_000,
    window_width_factor: tuple[float, float] = (25, 35),
    progress: bool = True,
) -> tuple[Sampler, float, float, float, float]:
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
    mu, sigma = samples.mean(), samples.std()
    window_low, window_high = window_width_factor
    min_E = mu - window_low * sigma
    max_E = mu + window_high * sigma
    bin_size = (max_E - min_E) / num_bins
    print(
        f"Energy window : [{min_E:.3f}, {max_E:.3f}] eV "
        f"({num_bins} bins, {bin_size:.4f} eV each)"
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

    # ------------------------------------------------------------
    # 3) Generate an initial configuration at the desired composition
    # ------------------------------------------------------------
    occ_enc = _initialize_supercell_occupancy(
        ensemble, rng, replace_element, new_elements, ratio
    )

    # ------------------------------------------------------------
    # 4) Run Wang-Landau MC
    # ------------------------------------------------------------
    nsamples = int(n_samples_per_site * ensemble.num_sites)
    thin_by = max(1, math.ceil(nsamples / thin_target))
    sampler.run(nsamples, occ_enc, thin_by=thin_by, progress=progress)

    return sampler, mu, min_E, max_E, bin_size


# =====================================================================
# Helper functions
# =====================================================================

def generate_wl_plots(
    mu: float,
    min_E: float,
    max_E: float,
    bin_size: float,
    sampler: Sampler,
    temperatures: np.ndarray,
    Cv: np.ndarray,
) -> None:
    """Create a 3-panel figure using data pre-computed in `run_wang_landau`."""

    _, (ax_top, ax_mid, ax_bot) = plt.subplots(3, 1, figsize=(6, 9), sharex=False, constrained_layout=True)

    # (1) WL convergence ---------------------------------------------------
    mod_factor = sampler.samples.get_trace_value("mod_factor")
    ax_top.semilogy(mod_factor, ".-")
    ax_top.set_xlabel("Iteration")
    ax_top.set_ylabel("Modification factor")
    ax_top.set_title("WL convergence")

    # (2) Density of states -------------------------------------------------
    entropy = sampler.samples.get_trace_value("entropy")[-1]
    nbins = entropy.size
    bin_centers = min_E + (np.arange(nbins) + 0.5) * bin_size
    mask = entropy > 0
    S_shift = entropy[mask] - entropy[mask].min()
    dos = np.exp(S_shift - S_shift.max())
    dos /= dos.sum()

    ax_mid.semilogy(bin_centers[mask], dos, ".-")
    ax_mid.axvline(mu, color="red", ls="--", label="mean train energy")
    ax_mid.set_xlabel(r"$E$ (eV / supercell)")
    ax_mid.set_ylabel("Density of states")
    ax_mid.set_title("WL DOS estimate")
    ax_mid.set_xlim(min_E, max_E)
    ax_mid.legend()

    # (3) Heat capacity -----------------------------------------------------
    ax_bot.plot(temperatures, Cv, lw=2)
    ax_bot.set_xlabel("Temperature (K)")
    ax_bot.set_ylabel(r"$C_v$ per supercell (eV K$^{-1}$)")
    ax_bot.set_title("Wang-Landau $C_v(T)$")

    plt.show()


def _initialize_supercell_occupancy(
    ensemble: Ensemble,
    rng: Generator,
    replace_element: str,
    new_elements: Tuple[str, str],
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

    # Print energy diagnostics
    print("min(raw) =", energy_levels.min(), "eV")
    print("max(raw) =", energy_levels.max(), "eV")
    print("ΔE =", energy_levels.max() - energy_levels.min(), "eV")

    # Compute thermodynamic properties
    k_B = 8.617333262e-5  # eV / K
    E_rr = energy_levels - energy_levels.min()

    Z = np.array([np.sum(dos_levels * np.exp(-E_rr / (k_B * T))) for T in temperatures_K])
    U = np.array([np.sum(dos_levels * energy_levels * np.exp(-E_rr / (k_B * T))) for T in temperatures_K]) / Z
    U2 = np.array([np.sum(dos_levels * energy_levels ** 2 * np.exp(-E_rr / (k_B * T))) for T in temperatures_K]) / Z
    Cv = (U2 - U ** 2) / (k_B * temperatures_K ** 2)  # eV / K / atom
    return Cv

# ── tc/wang_landau.py  (add near the bottom) ───────────────────────────
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers 3-D proj)
import numpy as np


def plot_cv_surface(
    ratios: np.ndarray,
    temperatures_K: np.ndarray,
    Cv_matrix: np.ndarray,
    *,
    mode: str = "surface",        # "surface" | "pcolormesh"
    cmap: str = "viridis",
) -> None:
    """
    Plot a 3-D heat-capacity surface C_v(T, x).

    Parameters
    ----------
    ratios : 1-D np.ndarray
        Composition axis (x = N_Mg / N_total), shape (M,)
    temperatures_K : 1-D np.ndarray
        Temperature axis (K), shape (N,)
    Cv_matrix : 2-D np.ndarray
        Heat capacity values, shape (M, N).  Cv_matrix[i, j] = Cv(T_j, x_i)
    mode : str
        "surface"  –  3-D surface plot (default)  
        "pcolormesh" –  2-D colour map in the (T, x) plane
    cmap : str
        Matplotlib colormap
    """
    ratios = np.asarray(ratios)
    temperatures_K = np.asarray(temperatures_K)
    Cv_matrix = np.asarray(Cv_matrix)

    if Cv_matrix.shape != (ratios.size, temperatures_K.size):
        raise ValueError("Cv_matrix shape must be (len(ratios), len(T)).")

    if mode == "pcolormesh":
        plt.figure(figsize=(7, 4))
        T_grid, X_grid = np.meshgrid(temperatures_K, ratios)
        pcm = plt.pcolormesh(
            T_grid,
            X_grid,
            Cv_matrix,
            shading="auto",
            cmap=cmap,
        )
        plt.colorbar(pcm, label=r"$C_v$ (eV K$^{-1}$ per supercell)")
        plt.xlabel("Temperature (K)")
        plt.ylabel("Mg fraction $x$")
        plt.title(r"$C_v(T,x)$ from Wang–Landau")
        plt.tight_layout()
        plt.show()

    elif mode == "surface":
        fig = plt.figure(figsize=(7, 5))
        ax = fig.add_subplot(111, projection="3d")
        T_grid, X_grid = np.meshgrid(temperatures_K, ratios)
        ax.plot_surface(
            T_grid,
            X_grid,
            Cv_matrix,
            rstride=1,
            cstride=1,
            cmap=cmap,
            linewidth=0,
            antialiased=True,
        )
        ax.set_xlabel("Temperature (K)")
        ax.set_ylabel("Mg fraction $x$")
        ax.set_zlabel(r"$C_v$ (eV K$^{-1}$ per supercell)")
        ax.set_title(r"$C_v(T,x)$ from Wang–Landau")
        ax.view_init(elev=25, azim=-60)
        plt.tight_layout()
        plt.show()
    else:
        raise ValueError("mode must be 'surface' or 'pcolormesh'")
