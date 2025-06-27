import numpy as np
from ase.atoms import Atoms
from numpy.random import Generator
from pymatgen.core import Structure, Composition, Element
from pymatgen.entries.computed_entries import ComputedStructureEntry
from pymatgen.io.ase import AseAtomsAdaptor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import max_error, mean_squared_error
from sklearn.model_selection import KFold
from smol.cofe import ClusterExpansion, ClusterSubspace, StructureWrangler
from smol.moca import Ensemble
from tqdm.auto import tqdm
from mace.calculators import MACECalculator

def make_ce_ensemble_from_mace(
        conv_cell: Atoms,
        rng: Generator,
        calc: MACECalculator,
        replace_element: str,
        new_elements: tuple[str, str],
        *,
        ensemble_size: int = 6,
        supercell_size: int = 4,
        bin_counts: int = 15,
        ) -> Ensemble:

    supercell_diag = (supercell_size, supercell_size, supercell_size)
    snapshots = make_snapshots(conv_cell, supercell_diag, rng, replace_element, new_elements, bin_counts)
    print(f"Total snapshots: {len(snapshots)}")
    pmg_structs = calculate_mace_energies(calc, snapshots)
    ce = cluster_expansion_from_pmg_structs(conv_cell, supercell_diag, pmg_structs, replace_element, new_elements)

    # Create a canonical ensemble
    ensemble_matrix = np.diag((ensemble_size, ensemble_size, ensemble_size))
    ensemble  = Ensemble.from_cluster_expansion(ce, ensemble_matrix)
    return ensemble

def calculate_mace_energies(calc: MACECalculator, snapshots: list[Atoms]) -> list[Structure]:
    pmg_structs = []
    for snapshot in tqdm(snapshots, desc="MACE energies"):
        snapshot.calc = calc
        pmg_struct = AseAtomsAdaptor.get_structure(snapshot) # pyright: ignore[reportArgumentType]
        pmg_struct.energy = snapshot.get_potential_energy()
        pmg_structs.append(pmg_struct)
    return pmg_structs

def cluster_expansion_from_pmg_structs(
        conv_cell: Atoms,
        supercell_diag: tuple[int, int, int],
        pmg_structs: list[Structure],
        replace_element: str,
        new_elements: tuple[str, ...],
        )-> ClusterExpansion:
    prim_cfg = AseAtomsAdaptor.get_structure(conv_cell) # pyright: ignore[reportArgumentType]
    composition = Composition({Element(elem): 0.5 for elem in new_elements})
    prim_cfg.replace_species({Element(replace_element): composition})

    subspace = ClusterSubspace.from_cutoffs(
        structure = prim_cfg,
        cutoffs   = {2: 8.0, 3: 6.0, 4:4.0},
        basis     = "indicator"
    )

    entries   = [ComputedStructureEntry(structure=s, energy=s.energy) for s in pmg_structs]
    wrangler = StructureWrangler(subspace)
    supercell_matrix = np.diag(supercell_diag)
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
    print(f"RMSE  {1e3*rmse:7.2f} meV   MAX  {1e3*mex:7.2f} meV")

    kf = KFold(n_splits=5, shuffle=True, random_state=0)
    cv_rmse = []
    for train, test in kf.split(X):
        reg = LinearRegression(fit_intercept=False).fit(X[train], y[train])
        cv_rmse.append(np.sqrt(mean_squared_error(y[test], reg.predict(X[test]))))
    print("5-fold CV RMSE:", 1e3*np.mean(cv_rmse), "meV")

    ce = ClusterExpansion(subspace, coefs)
    return ce

def make_snapshots(
        conv_cell: Atoms,
        supercell_diag: tuple[int, int, int],
        rng: Generator,
        replace_element: str,
        new_elements: tuple[str, str],
        bin_counts: int,
        ) -> list[Atoms]:
    if len(new_elements) != 2:
        raise NotImplementedError("Only two new elements are supported for replacement.")
    A, B = new_elements

    proto = conv_cell * supercell_diag
    replace_idx = [i for i, at in enumerate(proto) if at.symbol == replace_element]
    n_replace  = len(replace_idx)

    # ---------- Stratified random sampling ----------
    composition_bins = {
        0.25: bin_counts,
        0.40: bin_counts,
        0.60: bin_counts,
        0.75: bin_counts,
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
