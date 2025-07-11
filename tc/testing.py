from collections import defaultdict
from dataclasses import dataclass

import numpy as np
from ase.atoms import Atoms
from mace.calculators import MACECalculator
from numpy.random import Generator
from pymatgen.io.ase import AseAtomsAdaptor
from smol.moca import Ensemble
from tqdm.auto import tqdm

import tc.dataset
from smol.cofe.space.domain import get_allowed_species


@dataclass
class ErrorStats:
    rmse_meV: float
    max_abs_meV: float
    per_comp_meV: dict[float, tuple[float, float]]


def evaluate_ensemble_vs_mace(
    ensemble: Ensemble,
    calc: MACECalculator,
    conv_cell: Atoms,
    rng: Generator,
    endpoint_energies: list[float],
    *,
    replace_element: str,
    new_elements: tuple[str, str],
    n_test: int = 30,
    comps: tuple[float, ...] = (0, 0.2, 0.5, 0.8, 1),
) -> ErrorStats:
    """
    Compare CE (via Ensemble.processor.compute_property) to MACE on random configs.
    Follows the 'WL sanity check' recipe exactly.
    """
    # ------------------------------------------------------------
    # derived constants
    # ------------------------------------------------------------
    N_sc = round(ensemble.processor.size ** (1/3))
    if N_sc**3 != ensemble.processor.size:
        raise ValueError(f"Supercell size {ensemble.processor.size} is not a perfect cube.")
    sc_tuple = (N_sc, N_sc, N_sc)
    sc_mat   = np.diag(sc_tuple)
    subspace = ensemble.processor.cluster_subspace

    proto       = conv_cell * sc_tuple
    repl_idx    = [i for i, at in enumerate(proto) if at.symbol == replace_element]
    n_replace   = len(repl_idx)
    A, B        = new_elements

    mace_Es, ce_E, x_list = [], [], []

    # ------------------------------------------------------------
    # loop
    # ------------------------------------------------------------
    for _ in tqdm(range(n_test), desc="CE vs MACE"):
        x   = float(rng.choice(comps))
        n_A = int(round(x * n_replace))

        snap = proto.copy()
        rng.shuffle(repl_idx)
        snap.symbols[repl_idx]       = B
        snap.symbols[repl_idx[:n_A]] = A

        # -- MACE -------------------------------------------------
        mace_E = tc.dataset.calculate_mace_energy(calc, snap, new_elements, endpoint_energies)
        mace_Es.append(mace_E)

        # -- CE via WL path --------------------------------------
        pmg_s = AseAtomsAdaptor.get_structure(snap)  # pyright: ignore[reportArgumentType]
        occ_enc = subspace.occupancy_from_structure(pmg_s, scmatrix=sc_mat, encode=True).astype(np.int32)
        E_ce_wl = float(ensemble.processor.compute_property(occ_enc))
        ce_E.append(E_ce_wl)
        x_list.append(x)

        print(f" x_Li = {x:4.2f} → CE = {1000*E_ce_wl:8.2f} meV   MACE = {1000* mace_E:8.2f} meV")

    # ------------------------------------------------------------
    # stats
    # ------------------------------------------------------------
    err = 1_000 * (np.array(mace_Es) - np.array(ce_E))       # meV
    rmse = float(np.sqrt(np.mean(err**2)))
    mabs = float(np.max(np.abs(err)))

    bucket: dict[float, list[float]] = defaultdict(list)
    for c, e in zip(x_list, err, strict=True):
        bucket[c].append(e)

    per_stats = {
        c: (float(np.sqrt(np.mean(np.square(v)))), float(np.max(np.abs(v))))
        for c, v in bucket.items()
    }

    # ------------------------------------------------------------
    # report
    # ------------------------------------------------------------
    print("\n------------ CE  vs  MACE  ------------")
    for c in sorted(per_stats):
        r, m = per_stats[c]
        print(f" x_Li = {c:4.2f} → RMSE = {r:6.2f} meV   |err|_max = {m:6.2f} meV")
    print("----------------------------------------")
    print(f" overall       RMSE = {rmse:6.2f} meV   |err|_max = {mabs:6.2f} meV")

    return ErrorStats(rmse, mabs, per_stats)

def sample_configs_slow(
    ensemble: Ensemble,
    conv_cell: Atoms,
    rng: Generator,
    *,
    replace_element: str,
    new_elements: tuple[str, str],
    n_samples: int = 100,
    ratio: float = 0.5,
) -> np.ndarray:
    """
    Compare CE (via Ensemble.processor.compute_property) to MACE on random configs.
    Follows the 'WL sanity check' recipe exactly.
    """
    # ------------------------------------------------------------
    # derived constants
    # ------------------------------------------------------------
    N_sc = round(ensemble.processor.size ** (1/3))
    if N_sc**3 != ensemble.processor.size:
        raise ValueError(f"Supercell size {ensemble.processor.size} is not a perfect cube.")
    sc_tuple = (N_sc, N_sc, N_sc)
    sc_mat   = np.diag(sc_tuple)
    subspace = ensemble.processor.cluster_subspace

    proto       = conv_cell * sc_tuple
    repl_idx    = [i for i, at in enumerate(proto) if at.symbol == replace_element]
    n_replace   = len(repl_idx)
    A, B        = new_elements
    n_A = int(round(ratio * n_replace))

    ce_E = []
    for _ in tqdm(range(n_samples), desc="CE samples"):
        snap = proto.copy()
        rng.shuffle(repl_idx)
        snap.symbols[repl_idx]       = B
        snap.symbols[repl_idx[:n_A]] = A

        # -- CE via WL path --------------------------------------
        pmg_s = AseAtomsAdaptor.get_structure(snap)  # pyright: ignore[reportArgumentType]
        occ_enc = subspace.occupancy_from_structure(pmg_s, scmatrix=sc_mat, encode=True).astype(np.int32)
        E_ce_wl = float(ensemble.processor.compute_property(occ_enc))
        ce_E.append(E_ce_wl)

    # print the mean, std dev, min, and max of the CE energies
    ce_E = np.array(ce_E)
    print(f"CE energies: mean = {1000 * ce_E.mean():8.2f} meV, std = {1000 * ce_E.std():8.2f} meV, min = {1000 * ce_E.min():8.2f} meV, max = {1000 * ce_E.max():8.2f} meV")
    return ce_E


def sample_configs_fast(
    ensemble: Ensemble,
    rng: Generator,
    *,
    n_samples: int = 100,
    ratio: float = 0.5,
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

    ce_E = []
    for step in tqdm(range(n_samples)):
        occ = np.zeros(n_sites, dtype=np.int32)
        li_sites = rng.choice(cat_idx, round(n_cations * ratio), replace=False)
        occ[li_sites] = 1

        # ---- sanity checks ---------------------------------------
        n_A_actual = (occ[cat_idx] == 1).sum()
        n_B_actual = (occ[cat_idx] == 0).sum()
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

def check_sampler_energies(
    sampler,
    ensemble,
    *,
    atol: float = 1e-6,
    rtol: float = 1e-8,
) -> None:
    """
    Re-compute the energy of *every* kept configuration with
    `ensemble.compute_feature_vector` and compare to the enthalpies
    recorded in the Wang–Landau trace.

    Parameters
    ----------
    sampler : smol.moca.Sampler
        A sampler that has already been run.
    ensemble : smol.moca.Ensemble
        The ensemble used by the sampler (needed for the feature → energy map).
    atol, rtol : float, optional
        Absolute / relative tolerances passed to ``np.allclose``.

    Prints
    ------
    * number of samples checked
    * max |ΔE|  between trace and recomputed energies
    * root-mean-square error (RMSE)
    * “PASS” / “FAIL” message
    """
    # --- pull everything out of the trace (no thinning / discarding) -----------
    occus       = sampler.samples.get_trace_value("occupancy", flat=True)
    enthalpies  = sampler.samples.get_trace_value("enthalpy",  flat=True).ravel()

    # --- recompute energies ----------------------------------------------------
    feats   = np.stack([ensemble.compute_feature_vector(o) for o in occus])
    e_comp  = feats @ ensemble.natural_parameters          # dot product
    diff    = e_comp - enthalpies

    # --- summary ---------------------------------------------------------------
    n   = len(occus)
    rms = np.sqrt(np.mean(diff**2))
    mx  = np.max(np.abs(diff))

    print(f"Checked {n} configurations")
    print(f"max |ΔE|  = {mx: .3e} eV")
    print(f"RMSE      = {rms:.3e} eV")

    if np.allclose(e_comp, enthalpies, atol=atol, rtol=rtol):
        print("✅  PASS: trace enthalpies match recomputed energies "
              f"(atol={atol}, rtol={rtol})")
    else:
        num_bad = np.count_nonzero(~np.isclose(e_comp, enthalpies,
                                               atol=atol, rtol=rtol))
        print(f"⚠️  FAIL: {num_bad} / {n} samples differ beyond tolerance "
              f"(atol={atol}, rtol={rtol})")


def compare_sampler_and_mace(
    sampler,
    ensemble: Ensemble,
    calc: MACECalculator,
    *,
    endpoint_eVpc: tuple[float, float],          # (E_A, E_B)  eV per cation
    cation_elements: tuple[str, str] = ("Mg", "Fe"),
) -> None:
    """
    For the first `n_compare` (thinned / discarded) samples in `sampler`,
    compute MACE energies and print a side-by-side comparison with the
    enthalpies stored in the WL trace.
    """
    occus = sampler.samples.get_trace_value("occupancy")
    wl_energies = sampler.samples.get_trace_value("enthalpy")

    # Compute the number of configurations to compare
    m = len(wl_energies)
    mace_energies = np.array([tc.dataset.mace_E_from_occ(ensemble, occus[i], calc, cation_elements, endpoint_eVpc) for i in tqdm(range(m), desc="Computing MACE energies")])

    # ---- pretty print & quick stats ---------------------------------------
    print(" idx |     MACE (eV) | WL trace (eV) |  Δ = MACE − WL (meV)")
    print("-----+---------------+---------------+----------------------")
    for i in range(m):
        delta = 1e3 * (mace_energies[i] - wl_energies[i])   # meV
        print(f"{i:4d} | {mace_energies[i]:13.6f} | {wl_energies[i]:13.6f} | {delta:10.2f}")

    rms_meV = 1e3 * np.sqrt(np.mean((mace_energies - wl_energies)**2))
    max_meV = 1e3 * np.max(np.abs(mace_energies - wl_energies))
    print("\nSummary:  RMS = {:.2f} meV   |Δ|_max = {:.2f} meV over {} configs".format(rms_meV, max_meV, m))
