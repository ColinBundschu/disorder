#!/usr/bin/env python

import argparse
import math
import pathlib
import numpy as np
from joblib import Parallel, delayed
from monty.serialization import loadfn

import tc.wang_landau
import tc.sampler_data
import tc.dataset


def _run_wl_to_convergence(
        seed: int,  # random seed for this run
        ensemble,  # ensemble of structures to sample from
        num_bins: int,  # number of Wang-Landau bins
        window: tuple[float, float],  # Wang-Landau energy window
        composition: dict[str, float],  # new elements to be added to the supercell
        snapshots_per_loop: int,  # number of random snapshots per ratio
        n_samples_per_site: int,  # number of Wang-Landau samples per site
        supercell_size: int,  # size of the supercell (e.g., 6 for 6x6x6 supercell)
        *,
        lattice_relaxed: bool,  # whether the lattice of the supercell is relaxed during MACE calculations
        mod_factor_threshold: float = 1e-8,  # convergence threshold for the modification factor
        max_loops: int = 200,  # maximum number of loops to run
):
    rng = np.random.default_rng(seed)
    sampler = tc.wang_landau.initialize_wl_sampler(ensemble, num_bins=num_bins, seeds=[seed], window=window)
    occ_enc = tc.wang_landau.initialize_supercell_occupancy(ensemble, rng, composition)
    mod_factor = 1

    nsamples_per_loop = int(n_samples_per_site * ensemble.num_sites / max_loops)
    thin_by = max(1, math.ceil(nsamples_per_loop / snapshots_per_loop))
    loop_count = 0
    while mod_factor > mod_factor_threshold and loop_count < max_loops:
        print(f"{tc.dataset.comp_str(composition)}  loop {loop_count}  ln f={mod_factor:8.2e}")
        sampler.run(nsamples_per_loop, occ_enc, thin_by=thin_by, progress=False)
        filepath = tc.wang_landau.make_sampler_filepath(composition, supercell_size, lattice_relaxed=lattice_relaxed)
        pathlib.Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        data = tc.sampler_data.dump_sampler_data(sampler, filepath)
        mod_factor = data.mod_factor_trace[-1]
        occ_enc = None
        loop_count += 1

    status = "CONVERGED" if mod_factor <= mod_factor_threshold else "INCOMPLETE"
    print(f"{tc.dataset.comp_str(composition)}  {status} after {loop_count} loop(s); ln f={mod_factor:8.2e}")

def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--nprocs", type=int, default=33, help="number of parallel processes")
    p.add_argument("--supercell_size", type=int, default=4, help="super-cell size")
    p.add_argument("--half_window", type=int, default=250, help="Half the Wang-Landau energy window in bins")
    p.add_argument("--E_bin_per_prim_eV", type=float, default=0.004, help="Energy bin width for Wang-Landau sampling")
    p.add_argument("--new_elements", type=str, default="Mg,Fe", help="Comma-separated list of new elements to be added")
    p.add_argument("--snapshots_per_loop",  type=int, default=100, help="number of random snapshots per ratio")
    p.add_argument("--n_samples_per_site",  type=int, default=10_000_000, help="number of Wang-Landau samples per site")
    p.add_argument("--relax_lattice", action="store_true", help="relax the lattice of the supercell during MACE calculations")

    args = p.parse_args(argv)

    # ── input parameters ────────────────────────────────────────────────
    seed_root = np.random.SeedSequence(42) # master seed
    el0, el1 = args.new_elements.split(",")
    compositions = [{el0: ratio, el1: 1 - ratio} for ratio in np.linspace(0.1, 0.9, 33, endpoint=True)]
    new_elements = args.new_elements.split(",")
    filepath = tc.dataset.make_ensemble_filepath(new_elements, args.supercell_size, lattice_relaxed=args.relax_lattice)
    E_bin_per_supercell_eV = args.supercell_size ** 3 * args.E_bin_per_prim_eV

    print(f"Using rock salt supercell with {args.supercell_size}x{args.supercell_size}x{args.supercell_size} supercell and new elements ({', '.join(new_elements)})…")
    print(f"Loading ensemble from {filepath}...")
    ensemble, _ = loadfn(filepath)
    print("Ensemble loaded.")

    rng = np.random.default_rng(123)
    samplers = tc.wang_landau.determine_wl_windows(5_000, args.snapshots_per_loop, args.half_window, args.nprocs,
                                                  rng, compositions, E_bin_per_supercell_eV, ensemble)
    windows = [(sampler.mckernels[0].spec.min_enthalpy, sampler.mckernels[0].spec.max_enthalpy) for sampler in samplers] # type: ignore

    child_seeds = [int(x) for x in seed_root.generate_state(len(compositions)).tolist()]
    print(f"Using {args.nprocs} workers for parallel sampling.")
    Parallel(n_jobs=args.nprocs, backend="loky", initializer=tc.wang_landau.init_worker)(
        delayed(_run_wl_to_convergence)(seed, ensemble, 2*args.half_window, window, composition, args.snapshots_per_loop,
                                        args.n_samples_per_site, args.supercell_size, lattice_relaxed=args.relax_lattice,
                                        ) for composition, seed, window in zip(compositions, child_seeds, windows)
    )

if __name__ == "__main__":
    main()
