from collections import defaultdict
from dataclasses import dataclass

import numpy as np
from ase.atoms import Atoms
from mace.calculators import MACECalculator
from numpy.random import Generator
from pymatgen.io.ase import AseAtomsAdaptor
from smol.moca import Ensemble
from tqdm.auto import tqdm


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
    *,
    replace_element: str,
    new_elements: tuple[str, str],
    n_test: int = 10,
    comps: tuple[float, ...] = (0.25, 0.40, 0.50, 0.60, 0.75),
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

    mace_E, ce_E, x_list = [], [], []

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
        snap.calc = calc
        mace_E.append(float(snap.get_potential_energy()))

        # -- CE via WL path --------------------------------------
        pmg_s = AseAtomsAdaptor.get_structure(snap)  # pyright: ignore[reportArgumentType]
        occ_enc = subspace.occupancy_from_structure(pmg_s, scmatrix=sc_mat, encode=True).astype(np.int32)
        E_ce_wl = float(ensemble.processor.compute_property(occ_enc))
        ce_E.append(E_ce_wl)
        x_list.append(x)

    # ------------------------------------------------------------
    # stats
    # ------------------------------------------------------------
    err = 1_000 * (np.array(mace_E) - np.array(ce_E))       # meV
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
