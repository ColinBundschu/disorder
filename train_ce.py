#!/usr/bin/env python

'''
This program trains a cluster expansion using Wang-Landau sampling with MACE as the ground truth.
Initially we train a CE using the MACE energies of randomly generated supercells with varying cation ratios.
This gives us a starting point for the CE, but random sampling misses low and high energy configurations.
To generate a more uniform training set across the accessible energy range, we run Wang-Landau sampling
to generate a set of configurations that roughly uniformly cover the full energy range of the system.
The final CE is then trained on the MACE energies of these Wang-Landau sampled configurations.
'''

import argparse

import numpy as np
from mace.calculators import mace_mp
from pymatgen.io.ase import AseAtomsAdaptor
from ase.build import bulk
from monty.serialization import dumpfn

import tc.dataset
import tc.testing
import tc.wang_landau
import pathlib


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--nprocs", type=int, default=17, help="number of parallel processes (default 17)")
    p.add_argument("--supercell_size", type=int, default=4, help="super-cell size (default 4x4x4)")
    p.add_argument("--snapshot_counts",  type=int, default=50, help="number of random snapshots per ratio")
    p.add_argument("--n_samples_per_site",  type=int, default=5_000, help="number of Wang-Landau samples per site")
    p.add_argument("--half_window", type=int, default=250, help="Half the Wang-Landau energy window in bins")
    p.add_argument("--E_bin_per_prim_eV", type=float, default=0.004, help="Energy bin width for Wang-Landau sampling (default 0.004 eV)")
    p.add_argument("--new_elements", type=str, default="Mg,Fe", help="Comma-separated list of new elements to be added")
    p.add_argument("--relax_lattice", action="store_true", help="relax the lattice of the supercell during MACE calculations")
    p.add_argument("--debug",  action="store_true", help="run extra MACE/ensemble sanity tests")
    args = p.parse_args(argv)

    # Set up the initial conditions
    supercell_diag = (args.supercell_size, args.supercell_size, args.supercell_size)
    conv_cell = bulk("MgO", crystalstructure="rocksalt", a=4.3, cubic=True)
    replace_element = "Mg"
    calc = mace_mp(model="large", device="cuda", default_dtype="float64")
    rng = np.random.default_rng(123)
    new_elements = args.new_elements.split(",")
    el0, el1 = args.new_elements.split(",")
    compositions = [{el0: ratio, el1: 1 - ratio} for ratio in np.linspace(0.1, 0.9, 17, endpoint=True)]
    E_bin_per_supercell_eV = np.prod(supercell_diag) * args.E_bin_per_prim_eV

    print(f"Using rock salt supercell with {args.supercell_size}x{args.supercell_size}x{args.supercell_size} supercell and new elements ({', '.join(new_elements)})…")
    print(f"Creating initial random snapshot ensemble ({args.snapshot_counts * len(compositions)} snapshots)…")
    endpoint_energies = tc.dataset.calculate_endpoint_energies(conv_cell, calc, replace_element, new_elements, relax_lattice=args.relax_lattice)
    snapshots = tc.dataset.make_random_snapshots(conv_cell, supercell_diag, rng, replace_element=replace_element, compositions=compositions, count=args.snapshot_counts)
    ensemble = tc.dataset.create_canonical_ensemble(conv_cell, calc, replace_element, new_elements, args.supercell_size,
                                                    endpoint_energies, supercell_diag, snapshots, relax_lattice=args.relax_lattice)

    if args.debug:
        tc.testing.evaluate_ensemble_vs_mace(ensemble, calc, conv_cell, rng, endpoint_energies, replace_element=replace_element,
                                             new_elements=new_elements, ratios=ratios, relax_lattice=args.relax_lattice)

    print("Initial ensemble created with", len(snapshots), "snapshots and", ensemble.num_sites, "sites per supercell.")
    samplers = tc.wang_landau.determine_wl_windows(args.n_samples_per_site, args.snapshot_counts, args.half_window, args.nprocs,
                                                  rng, compositions, E_bin_per_supercell_eV, ensemble)

    print("Computing final CE from Wang-Landau sampled configurations…")
    wl_occupancies = [occ for sampler in samplers for occ in sampler.samples.get_trace_value("occupancy")] # type: ignore
    snapshots = [AseAtomsAdaptor.get_atoms(ensemble.processor.structure_from_occupancy(occ)) for occ in wl_occupancies]
    ensemble = tc.dataset.create_canonical_ensemble(conv_cell, calc, replace_element, new_elements, args.supercell_size,
                                                    endpoint_energies, supercell_diag, snapshots, relax_lattice=args.relax_lattice) # type: ignore

    if args.debug:
        tc.testing.evaluate_ensemble_vs_mace(ensemble, calc, conv_cell, rng, endpoint_energies, replace_element=replace_element,
                                             new_elements=new_elements, ratios=ratios, relax_lattice=args.relax_lattice)

    filepath = tc.dataset.make_ensemble_filepath(new_elements, args.supercell_size, lattice_relaxed=args.relax_lattice)
    pathlib.Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    dumpfn((ensemble, endpoint_energies), filepath, indent=2)
    print("Done - results written to", filepath)


if __name__ == "__main__":
    main()
