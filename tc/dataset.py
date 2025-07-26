from collections.abc import Iterable
import numpy as np
from ase.atoms import Atoms
from mace.calculators import MACECalculator
from numpy.random import Generator
from pymatgen.core import Composition, Element, Structure
from pymatgen.entries.computed_entries import ComputedStructureEntry
from pymatgen.io.ase import AseAtomsAdaptor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import max_error, mean_squared_error
from sklearn.model_selection import KFold
from smol.cofe import ClusterExpansion, ClusterSubspace, StructureWrangler
from smol.moca import Ensemble
from ase.optimize import FIRE
from ase.filters import UnitCellFilter
import os
import math

def run_folderpath(new_elements: list[str], supercell_size: int, *, lattice_relaxed: bool):
    relaxed_str = "LR" if lattice_relaxed else "F"
    elem_str = '-'.join(sorted(new_elements))
    # RSO for Rock Salt Oxide
    return os.path.join("/mnt", "z", "disorder", "RSO", f"{elem_str}_{supercell_size}_{relaxed_str}")

def make_ensemble_filepath(new_elements: list[str], supercell_size: int, *, lattice_relaxed: bool) -> str:
    return os.path.join(run_folderpath(new_elements, supercell_size, lattice_relaxed=lattice_relaxed), "ensemble.json.gz")


def create_canonical_ensemble(
    conv_cell: Atoms,
    calc: MACECalculator,
    replace_element: str,
    new_elements: list[str],
    ensemble_size: int,
    endpoint_energies: list[float],
    supercell_diag: tuple[int, int, int],
    snapshots: list[Atoms],
    *,
    relax_lattice: bool,
) -> Ensemble:
    # Create a cluster expansion from the provided snapshots
    pmg_structs = []
    for i, snapshot in enumerate(snapshots):
        pmg_struct = AseAtomsAdaptor.get_structure(snapshot) # pyright: ignore[reportArgumentType]
        pmg_struct.energy = calculate_mace_energy(calc, snapshot, new_elements, endpoint_energies, relax_lattice=relax_lattice)
        pmg_structs.append(pmg_struct)
        # print every 10% of the snapshots
        if i % (len(snapshots) // 10) == 0 or i == len(snapshots) - 1:
            print(f"{round(i / len(snapshots) * 100)}% of snapshots processed")
    orbit_cutoffs = {1: 100, 2: 10.0, 3: 8.0, 4: 6.0}
    print(f"Creating cluster expansion with cutoffs {orbit_cutoffs}")
    ce = cluster_expansion_from_pmg_structs(conv_cell, orbit_cutoffs, supercell_diag, pmg_structs, replace_element, new_elements)

    # Create a canonical ensemble
    ensemble = Ensemble.from_cluster_expansion(ce, np.diag((ensemble_size, ensemble_size, ensemble_size)))
    return ensemble

def calculate_endpoint_energies(
        conv_cell: Atoms,
        calc: MACECalculator,
        replace_element: str,
        new_elements: tuple[str, ...],
        *,
        relax_lattice: bool,
) -> list[float]:
    replace_idx = [i for i, at in enumerate(conv_cell) if at.symbol == replace_element] # type: ignore
    endpoint_energies = []
    for elem in new_elements:
        prim = conv_cell.copy()
        prim.symbols[replace_idx] = elem
        prim.calc = calc

        if relax_lattice:
            dyn = FIRE(UnitCellFilter(prim), logfile=None) # type: ignore
            dyn.run(fmax=0.02, steps=200)
            dyn_str= f"iters={dyn.nsteps:3d}"
        else:
            dyn_str = "Fixed-lattice"

        E_mace = prim.get_potential_energy() / len(replace_idx)
        endpoint_energies.append(E_mace)

        # lattice constants (Å) after relaxation
        a, b, c = prim.cell.lengths()
        print(f"{elem:2s}  a={a:6.3f} Å  b={b:6.3f} Å  c={c:6.3f} Å  E={E_mace:7.4f} eV/cation  {dyn_str}")

    return endpoint_energies

def calculate_mace_energy(
        calc: MACECalculator,
        snapshot: Atoms,
        cation_elements: Iterable[str],
        endpoint_energies_per_cation: Iterable[float],
        *,
        relax_lattice: bool,
        ) -> float:
    cation_counts = [snapshot.symbols.count(elem) for elem in cation_elements]
    snapshot.calc = calc
    if relax_lattice:
        FIRE(UnitCellFilter(snapshot), logfile=None).run(fmax=0.02, steps=200) # type: ignore
    return snapshot.get_potential_energy() - np.dot(cation_counts, np.array(endpoint_energies_per_cation))

def cluster_expansion_from_pmg_structs(
        conv_cell: Atoms,
        cutoffs,
        supercell_diag: tuple[int, int, int],
        pmg_structs: list[Structure],
        replace_element: str,
        new_elements: list[str],
        )-> ClusterExpansion:
    # Count how many cations are in the conv_cell
    n_cations_per_prim = sum(1 for at in conv_cell if at.symbol == replace_element) # type: ignore

    prim_cfg = AseAtomsAdaptor.get_structure(conv_cell) # pyright: ignore[reportArgumentType]
    composition = Composition({Element(elem): 0.5 for elem in new_elements})
    prim_cfg.replace_species({Element(replace_element): composition})

    print(f"Primitive cell: {prim_cfg.composition.reduced_formula} with {n_cations_per_prim} cations")
    subspace = ClusterSubspace.from_cutoffs(
        structure = prim_cfg,
        cutoffs   = cutoffs,
        basis     = "indicator"
    )
    print(f"Number of orbits: {subspace.num_orbits}")

    entries = [ComputedStructureEntry(structure=s, energy=s.energy) for s in pmg_structs]
    wrangler = StructureWrangler(subspace)
    supercell_matrix = np.diag(supercell_diag)
    site_map = None
    for ent in entries:
        wrangler.add_entry(ent, supercell_matrix=supercell_matrix, site_mapping=site_map, verbose=False)
        if site_map is None:
            site_map = wrangler.entries[-1].data["site_mapping"]

    print(f"Matched structures: {wrangler.num_structures}/{len(entries)}")

    X = wrangler.feature_matrix
    y = wrangler.get_property_vector("energy")

    reg = LinearRegression(fit_intercept=False)
    reg.fit(X, y)
    coefs = reg.coef_
    print("rank =", np.linalg.matrix_rank(X), "of", X.shape[1], "columns")

    rmse = np.sqrt(mean_squared_error(y, X @ coefs))
    mex  = max_error(y, X @ coefs)
    print(f"RMSE  {1e3*rmse/n_cations_per_prim:7.2f} meV   MAX  {1e3*mex/n_cations_per_prim:7.2f} meV")

    kf = KFold(n_splits=5, shuffle=True, random_state=0)
    cv_rmse = []
    for train, test in kf.split(X):
        reg = LinearRegression(fit_intercept=False).fit(X[train], y[train])
        cv_rmse.append(np.sqrt(mean_squared_error(y[test], reg.predict(X[test]))))
    print("5-fold CV RMSE:", 1e3*np.mean(cv_rmse)/n_cations_per_prim, "meV")

    ce = ClusterExpansion(subspace, coefs)
    return ce

def make_random_snapshots(
    conv_cell      : Atoms,
    supercell_diag : tuple[int,int,int],
    rng            : Generator,
    *,
    replace_element: str,
    compositions   : list[dict[str,float]],   # list of {element: fraction}
    count          : int,
) -> list[Atoms]:
    """
    Generate `count` random snapshots **per composition**.

    Parameters
    ----------
    conv_cell        : primitive MgO‐like rock‑salt cell
    supercell_diag   : (nx,ny,nz) replication tuple
    replace_element  : element in `conv_cell` to be replaced
    compositions     : list of dictionaries, each summing to 1.0
    count            : how many distinct random snapshots per composition
    """
    proto       = conv_cell * supercell_diag
    repl_idx    = [i for i, at in enumerate(proto) if at.symbol == replace_element]
    n_replace   = len(repl_idx)
    snapshots: list[Atoms] = []

    for comp in compositions:
        # ── sanity & integer site counts ───────────────────────────────
        if not math.isclose(sum(comp.values()), 1.0, abs_tol=1e-6):
            raise ValueError(f"Fractions must sum to 1; got {comp}")
        counts = {el: round(frac * n_replace) for el, frac in comp.items()}

        # fix rounding so totals match exactly
        delta = n_replace - sum(counts.values())
        if delta:
            for el in list(counts)[:abs(delta)]:
                counts[el] += int(math.copysign(1, delta))

        # ── generate `count` unique snapshots ──────────────────────────
        existing_configs = set()        # track uniqueness by a hashable key
        while len(existing_configs) < count:
            rng.shuffle(repl_idx)       # random permutation
            snapshot = proto.copy()

            start = 0
            key_parts = []
            for el, n_el in counts.items():
                end = start + n_el
                idx_slice = repl_idx[start:end]
                snapshot.symbols[idx_slice] = el
                key_parts.append(tuple(sorted(idx_slice)))
                start = end
            key = tuple(key_parts)      # composite key for all elements

            if key in existing_configs:
                continue
            existing_configs.add(key)
            snapshots.append(snapshot)

    return snapshots

def mace_E_from_occ(
    ensemble       : Ensemble,
    occupancy      : np.ndarray,
    calc           : MACECalculator,
    cation_elements: tuple[str, str],
    endpoint_eVpc  : list[float],      # (E_Mg, E_Fe)  eV / cation
    *,
    relax_lattice: bool,
) -> float:
    """
    Return the reference-shifted MACE energy (eV) of a configuration encoded
    by `occupancy`.
    """
    struct = ensemble.processor.structure_from_occupancy(occupancy)
    snapshot: Atoms = AseAtomsAdaptor.get_atoms(struct) # type: ignore
    return calculate_mace_energy(calc, snapshot, cation_elements, endpoint_eVpc, relax_lattice=relax_lattice)
