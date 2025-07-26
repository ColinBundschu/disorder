"""
Utility helpers to benchmark Wang-Landau Tc predictions with
straight-forward Metropolis Monte-Carlo sampling using *smol*.

The public API exposes two call-sites that mirror the WL helpers already
in the repo:

• ``run_metropolis_heat_capacity`` - run a Metropolis sampler at a single
  temperature and return the heat capacity *Cv* (per primitive cell).

• ``plot_cv_curve`` - convenience wrapper that takes an array of
  temperatures and the corresponding *Cv* values and produces a simple
  line plot.

Typical usage
-------------
>>> Ts = np.linspace(50, 800, 30)              # K
>>> Cv = [
...     run_metropolis_heat_capacity(
...         ensemble, T, rng, ratio=0.50,
...         replace_element="Mg", new_elements=("Mg", "Fe"),
...     )
...     for T in Ts
... ]
>>> plot_cv_curve(Ts, Cv)

The code purposefully mirrors the structure of *wang_landau.py* so that
both approaches are easy to compare side-by-side.
"""

import math
from typing import Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from ase.build import bulk    # only used for dummy default in helper
from numpy.random import Generator
from pymatgen.io.ase import AseAtomsAdaptor
from smol.cofe import ClusterSubspace
from smol.moca import Ensemble, Sampler

# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

def run_metropolis_heat_capacity(
    ensemble: Ensemble,
    temperature_K: float,
    rng: Generator,
    *,
    replace_element: str = "Mg",
    new_elements: Tuple[str, str] = ("Mg", "Fe"),
    ratio: float = 0.50,
    # n_steps_per_site: int = 2_000,
    n_steps_per_site: int = 10,
    thin_target: int = 1_000,
    seeds: Sequence[int] = (123,),
    equil_fraction: float = 0.50,
) -> float:
    """Estimate *Cv* at a single *T* using Metropolis MC with swap moves.

    Parameters
    ----------
    ensemble
        *smol* ``Ensemble`` object to be sampled.
    temperature_K
        Target temperature in Kelvin.
    rng
        Numpy random generator for reproducible shuffles.
    replace_element / new_elements / ratio
        Same semantics as in ``wang_landau.run_wang_landau``.
    n_steps_per_site
        MC propagation length — *per lattice site*.
    thin_target
        Desired number of stored samples.  Energies are thinned on the fly
        so that roughly this many frames are analysed.
    seeds
        Seeds for the underlying *smol* ``Sampler`` (one per replica).
    equil_fraction
        First ``equil_fraction`` of collected samples are discarded when
        estimating averages.

    Returns
    -------
    Cv
        Heat capacity (eV / K) *per primitive cell* at the requested *T*.
    """

    k_B = 8.617333262e-5  # eV / K

    # ------------------------------------------------------------------
    # 1) Build the Metropolis sampler - *smol* has this out-of-the-box
    # ------------------------------------------------------------------
    sampler = Sampler.from_ensemble(
        ensemble,
        kernel_type="Metropolis",    # → detailed-balance MC
        temperature=temperature_K,    # in Kelvin
        step_type="swap",            # preserves overall composition
        seeds=list(seeds),
    )

    # ------------------------------------------------------------------
    # 2) Generate an initial configuration at the desired composition
    # ------------------------------------------------------------------
    occ_enc = initialize_supercell_occupancy(
        ensemble, rng, replace_element, new_elements, ratio
    )

    # ------------------------------------------------------------------
    # 3) Run Metropolis MC
    # ------------------------------------------------------------------
    nsamples = int(n_steps_per_site * ensemble.num_sites)
    thin_by = max(1, math.ceil(nsamples / thin_target))
    sampler.run(nsamples, occ_enc, thin_by=thin_by, progress=False)

    # ------------------------------------------------------------------
    # 4) Extract energies and compute Cv
    # ------------------------------------------------------------------
    energies = sampler.samples.get_energies()                      # eV / supercell
    discard  = int(len(energies) * equil_fraction)                 # burn-in
    energies = energies[discard:]

    E_mean = energies.mean()
    E2_mean = (energies ** 2).mean()

    Cv = (E2_mean - E_mean**2) / (k_B * temperature_K**2)
    return float(Cv)


def plot_cv_curve(temperatures: np.ndarray, Cv: np.ndarray) -> None:
    """Simple helper - plot *Cv(T)* on a fresh Matplotlib figure."""

    _, ax = plt.subplots(figsize=(5, 4))
    ax.plot(temperatures, Cv, marker="o", lw=2)
    ax.set_xlabel("Temperature (K)")
    ax.set_ylabel(r"$C_v$ per unit cell (eV K$^{-1}$)")
    ax.set_title("Metropolis $C_v(T)$")
    plt.show()

# -----------------------------------------------------------------------------
# Internal helpers (copied / adapted from wang_landau.py)
# -----------------------------------------------------------------------------

def initialize_supercell_occupancy(
    ensemble: Ensemble,
    rng: Generator,
    replace_element: str,
    new_elements: Tuple[str, str],
    ratio: float,
) -> np.ndarray:
    """Encode a random snapshot at the requested composition.

    This is a near verbatim copy of the WL helper so that both samplers
    start from statistically equivalent configurations.
    """

    n_sc = round(ensemble.processor.size ** (1 / 3))
    if n_sc ** 3 != ensemble.processor.size:
        raise ValueError("Ensemble supercell is not cubic.")

    sc_tuple = (n_sc, n_sc, n_sc)
    sc_mat = np.diag(sc_tuple)

    # Dummy conventional cell - only topology matters here.  If *conv_cell*
    # is available upstream you can pass it in instead to avoid the import.
    conv_cell = bulk("MgO", crystalstructure="rocksalt", a=4.3, cubic=True)
    snapshot = conv_cell * sc_tuple

    # Populate the cation sub-lattice ----------------------------------
    repl_idx = [i for i, at in enumerate(snapshot) if at.symbol == replace_element]
    rng.shuffle(repl_idx)
    n_A = int(round(ratio * len(repl_idx)))
    A, B = new_elements
    snapshot.symbols[repl_idx] = B
    snapshot.symbols[repl_idx[: n_A]] = A

    subspace: ClusterSubspace = ensemble.processor.cluster_subspace
    pmg_struct = AseAtomsAdaptor.get_structure(snapshot)
    occ_enc = subspace.occupancy_from_structure(pmg_struct, scmatrix=sc_mat, encode=True)

    return occ_enc.astype(np.int32) # type: ignore
