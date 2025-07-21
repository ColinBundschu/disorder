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
import math

import numpy as np
from joblib import Parallel, delayed
from mace.calculators import mace_mp
from pymatgen.io.ase import AseAtomsAdaptor
from ase.build import bulk
from monty.serialization import dumpfn

import tc.dataset
import tc.testing
import tc.wang_landau


def _single_wl(ratio, seed, *, ensemble, num_bins, window, replace_element, new_elements, snapshot_counts, n_samples_per_site):
    rng = np.random.default_rng(seed)
    sampler = tc.wang_landau.initialize_wl_sampler(ensemble, rng=rng, ratio=ratio, num_bins=num_bins, seeds=[seed], window=window)
    occ_enc = tc.wang_landau.initialize_supercell_occupancy(ensemble, rng, replace_element, new_elements, ratio)
    nsamples = int(n_samples_per_site * ensemble.num_sites)
    thin_by  = max(1, math.ceil(nsamples / snapshot_counts))
    sampler.run(nsamples, occ_enc, thin_by=thin_by, progress=False)
    return sampler


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--nprocs", type=int, default=17, help="number of parallel processes (default 17)")
    p.add_argument("--supercell_size", type=int, default=6, help="super-cell size (default 6x6x6)")
    p.add_argument("--num_wl_bins", type=int, default=250, help="number of Wang-Landau bins (default 250)")
    p.add_argument("--snapshot_counts",  type=int, default=100, help="number of random snapshots per ratio")
    p.add_argument("--n_samples_per_site",  type=int, default=5_000, help="number of Wang-Landau samples per site")
    p.add_argument("--initial_window", type=int, default=20, help="Initial Wang-Landau energy window in stddevs of random")
    p.add_argument("--relax_lattice", action="store_true", help="relax the lattice of the supercell during MACE calculations")
    p.add_argument("--debug",  action="store_true", help="run extra MACE/ensemble sanity tests")
    args = p.parse_args(argv)

    # Set up the initial conditions
    supercell_diag = (args.supercell_size, args.supercell_size, args.supercell_size)
    conv_cell = bulk("MgO", crystalstructure="rocksalt", a=4.2, cubic=True)
    calc = mace_mp(model="large", device="cuda", default_dtype="float64")
    rng = np.random.default_rng(123)
    replace_element = "Mg"
    new_elements=("Mg", "Fe")
    ratios = list(np.linspace(0.1, 0.9, 17, endpoint=True))
    window=(args.initial_window, args.initial_window)

    print(f"Creating initial random snapshot ensemble ({args.snapshot_counts} snapshots)…")
    endpoint_energies = tc.dataset.calculate_endpoint_energies(conv_cell, calc, replace_element, new_elements, relax_lattice=args.relax_lattice)
    snapshots = tc.dataset.make_snapshots(conv_cell, supercell_diag, rng, replace_element, new_elements, args.snapshot_counts, ratios=ratios)
    ensemble = tc.dataset.create_canonical_ensemble(conv_cell, calc, replace_element, new_elements, args.supercell_size,
                                                    endpoint_energies, supercell_diag, snapshots, relax_lattice=args.relax_lattice)

    if args.debug:
        tc.testing.evaluate_ensemble_vs_mace(ensemble, calc, conv_cell, rng, endpoint_energies, replace_element=replace_element,
                                             new_elements=new_elements, comps=ratios, relax_lattice=args.relax_lattice)

    print(f"Starting Wang-Landau sampling over {args.nprocs} processes and initial window {window}…")
    seed_root  = np.random.SeedSequence(42)
    child_seeds = [int(x) for x in seed_root.generate_state(len(ratios)).tolist()]

    while True:
        print(f"Sampling {len(ratios)} cation ratios with {args.n_samples_per_site} samples per site…")
        samplers = Parallel(n_jobs=args.nprocs, backend="loky", max_nbytes=None)(
            delayed(_single_wl)(
                r, s,
                ensemble=ensemble,
                num_bins=args.num_wl_bins,
                window=window,
                replace_element=replace_element,
                new_elements=new_elements,
                snapshot_counts=args.snapshot_counts,
                n_samples_per_site=args.n_samples_per_site
            )
            for r, s in zip(ratios, child_seeds)
        )
        widen_low = False
        widen_high = False
        for sampler in samplers:
            entropy = sampler.samples.get_trace_value("entropy")[-1]
            if entropy[0] > 0:
                widen_low = True
            if entropy[-1] > 0:
                widen_high = True
        
        if not widen_low and not widen_high:
            print(f"Finished sampling with window {window}. Increasing to {window[0] + 5, window[1] + 5} to add buffer.")
            window = (window[0] + 5, window[1] + 5)
            break # The window is sufficiently large, exit the loop
        
        if widen_low:
            print(f"Low end of window {window} is not sufficiently sampled, increasing to {window[0] + 5, window[1]}.")
            window = (window[0] + 5, window[1])
        if widen_high:
            print(f"High end of window {window} is not sufficiently sampled, increasing to {window[0], window[1] + 5}.")
            window = (window[0], window[1] + 5)

    print("Computing final CE from Wang-Landau sampled configurations…")
    wl_occupancies = [occ for sampler in samplers for occ in sampler.samples.get_trace_value("occupancy")]
    snapshots = [AseAtomsAdaptor.get_atoms(ensemble.processor.structure_from_occupancy(occ)) for occ in wl_occupancies]
    ensemble = tc.dataset.create_canonical_ensemble(conv_cell, calc, replace_element, new_elements, args.supercell_size,
                                                    endpoint_energies, supercell_diag, snapshots, relax_lattice=args.relax_lattice)

    if args.debug:
        tc.testing.evaluate_ensemble_vs_mace(ensemble, calc, conv_cell, rng, endpoint_energies, replace_element=replace_element,
                                             new_elements=new_elements, comps=ratios, relax_lattice=args.relax_lattice)

    filename = f"{''.join(new_elements)}O_ensemble{args.supercell_size}.json.gz"
    dumpfn((ensemble, endpoint_energies), filename, indent=2)
    print("Done - results written to", filename)


if __name__ == "__main__":
    main()
