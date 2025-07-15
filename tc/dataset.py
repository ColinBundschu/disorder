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

def create_canonical_ensemble(conv_cell, calc, replace_element, new_elements, ensemble_size, endpoint_energies, supercell_diag, snapshots, reuse_site_map):
    # Create a cluster expansion from the provided snapshots
    pmg_structs = []
    for snapshot in snapshots:
        pmg_struct = AseAtomsAdaptor.get_structure(snapshot) # pyright: ignore[reportArgumentType]
        pmg_struct.energy = calculate_mace_energy(calc, snapshot, new_elements, endpoint_energies)
        pmg_structs.append(pmg_struct)
    ce = cluster_expansion_from_pmg_structs(conv_cell, {1: 100, 2: 10.0, 3: 8.0, 4: 6.0}, supercell_diag, pmg_structs, replace_element, new_elements, reuse_site_map)

    # Create a canonical ensemble
    ensemble = Ensemble.from_cluster_expansion(ce, np.diag((ensemble_size, ensemble_size, ensemble_size)))
    return ensemble

def calculate_endpoint_energies(conv_cell, calc, replace_element, new_elements):
    replace_idx = [i for i, at in enumerate(conv_cell) if at.symbol == replace_element] # type: ignore
    endpoint_energies = []
    for elem in new_elements:
        prim = conv_cell.copy()
        prim.symbols[replace_idx] = elem
        prim.calc = calc
        E_mace = prim.get_potential_energy() / len(replace_idx)
        endpoint_energies.append(E_mace)
    return endpoint_energies

def calculate_mace_energy(
        calc: MACECalculator,
        snapshot: Atoms,
        cation_elements: tuple[str, str],
        endpoint_energies_per_cation: list[float],
        ) -> float:
    cation_counts = [snapshot.symbols.count(elem) for elem in cation_elements]
    snapshot.calc = calc
    return snapshot.get_potential_energy() - np.dot(cation_counts, endpoint_energies_per_cation)

def cluster_expansion_from_pmg_structs(
        conv_cell: Atoms,
        cutoffs,
        supercell_diag: tuple[int, int, int],
        pmg_structs: list[Structure],
        replace_element: str,
        new_elements: tuple[str, ...],
        reuse_site_map: bool,
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
        if site_map is None and reuse_site_map:
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

def make_snapshots(
        conv_cell: Atoms,
        supercell_diag: tuple[int, int, int],
        rng: Generator,
        replace_element: str,
        new_elements: tuple[str, str],
        count: int,
        ratios: list[float],
        ) -> list[Atoms]:
    if len(new_elements) != 2:
        raise NotImplementedError("Only two new elements are supported for replacement.")
    A, B = new_elements

    proto = conv_cell * supercell_diag
    replace_idx = [i for i, at in enumerate(proto) if at.symbol == replace_element]
    n_replace  = len(replace_idx)

    snapshots: list[Atoms] = []
    for ratio in ratios:
        existing_configs = set()
        n_A = int(round(ratio * n_replace))
        for _ in range(count):
            snapshot = proto.copy()
            A_sites = sorted(rng.choice(replace_idx, size=n_A, replace=False))
            if tuple(A_sites) in existing_configs:
                continue
            existing_configs.add(tuple(A_sites))
            snapshot.symbols[replace_idx] = B # Set all to B first
            snapshot.symbols[A_sites] = A # Set selected sites to A
            snapshots.append(snapshot)
    return snapshots

def mace_E_from_occ(
    ensemble       : Ensemble,
    occupancy      : np.ndarray,
    calc           : MACECalculator,
    cation_elements: tuple[str, str],
    endpoint_eVpc  : tuple[float, float],      # (E_Mg, E_Fe)  eV / cation
) -> float:
    """
    Return the reference-shifted MACE energy (eV) of a configuration encoded
    by `occupancy`.
    """
    struct = ensemble.processor.structure_from_occupancy(occupancy)
    snapshot: Atoms = AseAtomsAdaptor.get_atoms(struct)
    return calculate_mace_energy(calc, snapshot, cation_elements, list(endpoint_eVpc))
