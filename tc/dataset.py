import math

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
from tqdm.auto import tqdm
from monty.serialization import dumpfn

def make_ce_ensembles_from_mace(
        conv_cell: Atoms,
        rng: Generator,
        calc: MACECalculator,
        replace_element: str,
        new_elements: tuple[str, str],
        ratio: float,
        *,
        ensemble_sizes: tuple[int, ...] = (4, 6, 8, 10, 12),
        supercell_size: int = 6,
        bin_counts: int = 200,
        ) -> list[Ensemble]:

    replace_idx = [i for i, at in enumerate(conv_cell) if at.symbol == replace_element]
    endpoint_energies = calculate_endpoint_energies(conv_cell, calc, new_elements, replace_idx)

    supercell_diag = (supercell_size, supercell_size, supercell_size)
    snapshots = make_snapshots(conv_cell, supercell_diag, rng, replace_element, new_elements, bin_counts, ratio)
    n_cations = len(replace_idx) * supercell_size**3
    print(f"Total snapshots: {len(snapshots)} for {n_cations} cations")
    pmg_structs = calculate_mace_energies(calc, snapshots, new_elements, endpoint_energies, n_cations)
    ce = cluster_expansion_from_pmg_structs(conv_cell, supercell_diag, pmg_structs, replace_element, new_elements)

    # Create a canonical ensemble
    ensembles  = []
    for n in ensemble_sizes:
        ensemble = Ensemble.from_cluster_expansion(ce, np.diag((n, n, n)))
        ensembles.append(ensemble)
        dumpfn(ensemble, f"{''.join(new_elements)}O_ensemble{n}.json.gz", indent=2)

    return ensembles

def calculate_endpoint_energies(conv_cell, calc, new_elements, replace_idx):
    endpoint_energies = []
    for elem in new_elements:
        prim = conv_cell.copy()
        prim.symbols[replace_idx] = elem
        prim.calc = calc
        E_mace = prim.get_potential_energy() / len(replace_idx)
        endpoint_energies.append(E_mace)
    return endpoint_energies

def calculate_mace_energies(
        calc: MACECalculator,
        snapshots: list[Atoms],
        cation_elements: tuple[str, str],
        endpoint_energies_per_cation: list[float],
        n_cations: int,
        *,
        loading_bar: bool = True,
        ) -> list[Structure]:
    pmg_structs = []
    for snapshot in tqdm(snapshots, desc="MACE energies", disable=not loading_bar):
        snapshot.calc = calc
        pmg_struct = AseAtomsAdaptor.get_structure(snapshot) # pyright: ignore[reportArgumentType]
        cation_counts = []
        for elem in cation_elements:
            cation_counts.append(snapshot.symbols.count(elem))
        if sum(cation_counts) != n_cations:
            raise ValueError(f"Snapshot has {sum(cation_counts)} cations, expected {n_cations}.")
        pmg_struct.energy = snapshot.get_potential_energy() - np.dot(cation_counts, endpoint_energies_per_cation)
        pmg_structs.append(pmg_struct)
    return pmg_structs

def cluster_expansion_from_pmg_structs(
        conv_cell: Atoms,
        supercell_diag: tuple[int, int, int],
        pmg_structs: list[Structure],
        replace_element: str,
        new_elements: tuple[str, ...],
        )-> ClusterExpansion:
    # Count how many cations are in the conv_cell
    n_cations_per_prim = sum(1 for at in conv_cell if at.symbol == replace_element)

    prim_cfg = AseAtomsAdaptor.get_structure(conv_cell) # pyright: ignore[reportArgumentType]
    composition = Composition({Element(elem): 0.5 for elem in new_elements})
    prim_cfg.replace_species({Element(replace_element): composition})

    subspace = ClusterSubspace.from_cutoffs(
        structure = prim_cfg,
        cutoffs   = {2: 10.0, 3: 8.0},
        basis     = "indicator"
    )

    entries   = [ComputedStructureEntry(structure=s, energy=s.energy) for s in pmg_structs]
    wrangler = StructureWrangler(subspace)
    supercell_matrix = np.diag(supercell_diag)
    N_prims = math.prod(supercell_diag)
    for ent in tqdm(entries, desc="Adding"):
        wrangler.add_entry(ent, supercell_matrix=supercell_matrix, verbose=False)

    print(f"Matched structures: {wrangler.num_structures}/{len(entries)}")

    X = wrangler.feature_matrix
    y = wrangler.get_property_vector("energy")

    reg = LinearRegression(fit_intercept=False)
    reg.fit(X, y)
    coefs = reg.coef_

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
        bin_counts: int,
        ratio: float,
        ) -> list[Atoms]:
    if len(new_elements) != 2:
        raise NotImplementedError("Only two new elements are supported for replacement.")
    A, B = new_elements

    proto = conv_cell * supercell_diag
    replace_idx = [i for i, at in enumerate(proto) if at.symbol == replace_element]
    n_replace  = len(replace_idx)

    # ---------- Stratified random sampling ----------
    composition_bins = {
        ratio: bin_counts,
    }

    snapshots: list[Atoms] = []
    for A_frac, count in composition_bins.items():
        for _ in range(count):
            snapshot = proto.copy()
            n_A = int(round(A_frac * n_replace))
            A_sites = rng.choice(replace_idx, size=n_A, replace=False)
            snapshot.symbols[replace_idx] = B # Set all to B first
            snapshot.symbols[A_sites] = A # Set selected sites to A
            snapshots.append(snapshot)

    return snapshots
