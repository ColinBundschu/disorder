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
    n_samples_per_site: int = 1000,
    window_width_factor: float = 5.0,
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
    mu, sigma = samples.mean(), samples.std()
    min_E, max_E = mu - window_width_factor * sigma, mu + window_width_factor * sigma
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
    sampler.run(nsamples, occ_enc, thin_by=thin_by, progress=True)

    temperatures_K = np.linspace(1.0, 20.0, 10_000)
    Cv = _compute_thermodynamics(sampler, ensemble, temperatures_K)
    _generate_wl_plots( mu, min_E, max_E, bin_size, sampler, temperatures_K, Cv)

    return sampler


# =====================================================================
# Helper functions
# =====================================================================

def _generate_wl_plots(
    mu: float,
    min_E: float,
    max_E: float,
    bin_size: float,
    sampler: Sampler,
    temperatures: np.ndarray,
    Cv: np.ndarray,
) -> None:
    """Create a 3-panel figure using data pre-computed in `run_wang_landau`."""

    fig, (ax_top, ax_mid, ax_bot) = plt.subplots(
        3, 1, figsize=(6, 9), sharex=False, constrained_layout=True
    )

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
    ax_mid.set_xlabel(r"$E$ (eV / cell)")
    ax_mid.set_ylabel("Density of states")
    ax_mid.set_title("WL DOS estimate")
    ax_mid.set_xlim(min_E, max_E)
    ax_mid.legend()

    # (3) Heat capacity -----------------------------------------------------
    ax_bot.plot(temperatures, Cv, lw=2)
    ax_bot.set_xlabel("Temperature (K)")
    ax_bot.set_ylabel(r"$C_v$ per atom (eV K$^{-1}$)")
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

    return occ_enc.astype(np.int32)


def _compute_thermodynamics(
    sampler: Sampler,
    ensemble: Ensemble,
    temperatures_K: np.ndarray,
) -> np.ndarray:
    """Compute heat capacity from Wang-Landau sampler results.
    
    Parameters
    ----------
    sampler : Sampler
        Wang-Landau sampler with entropy and energy data
    ensemble : Ensemble
        Ensemble object containing number of sites
    temperatures_K : np.ndarray
        Temperature array (K)
        
    Returns
    -------
    Cv : np.ndarray
        Heat capacity per atom (eV K^-1)
    """
    # Extract density of states from entropy
    entropy = sampler.samples.get_trace_value("entropy")[-1]
    mask = entropy > 0
    ent_ref = entropy[mask] - entropy[mask].min()
    dos_levels = np.exp(ent_ref - ent_ref.max())
    dos_levels /= dos_levels.sum()

    # Get energy levels per atom
    N_sites = ensemble.num_sites
    energy_levels = sampler.mckernels[0].levels / N_sites  # eV per atom

    # Print energy diagnostics
    print("min(raw) =", 1000 * energy_levels.min(), "meV")
    print("max(raw) =", 1000 * energy_levels.max(), "meV")
    print("ΔE =", 1000 * (energy_levels.max() - energy_levels.min()), "meV")

    # Compute thermodynamic properties
    k_B = 8.617333262e-5  # eV / K
    E_rr = energy_levels - energy_levels.min()

    Z = np.array(
        [np.sum(dos_levels * np.exp(-E_rr / (k_B * T))) for T in temperatures_K]
    )
    U = (
        np.array(
            [np.sum(dos_levels * energy_levels * np.exp(-E_rr / (k_B * T))) for T in temperatures_K]
        )
        / Z
    )
    U2 = (
        np.array(
            [np.sum(dos_levels * energy_levels ** 2 * np.exp(-E_rr / (k_B * T))) for T in temperatures_K]
        )
        / Z
    )
    Cv = (U2 - U ** 2) / (k_B * temperatures_K ** 2)  # eV / K / atom

    return Cv
