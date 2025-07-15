#!/usr/bin/env python

import argparse
import math
import os

import numpy as np
from joblib import Parallel, delayed
from monty.serialization import loadfn

import tc.wang_landau
import tc.sampler_data


def _run_wl_to_convergence(ratio, seed, ensemble, num_bins, window, replace_element, new_elements, snapshots_per_loop,
                           n_samples_per_site, mod_factor_threshold=1e-6, max_loops=200):
    rng = np.random.default_rng(seed)
    sampler = tc.wang_landau.initialize_wl_sampler(
        ensemble, rng=rng, ratio=ratio, num_bins=num_bins, seeds=[seed], window=window)
    occ_enc = tc.wang_landau.initialize_supercell_occupancy(ensemble, rng, replace_element, new_elements, ratio)
    mod_factor = 1

    nsamples_per_loop = int(n_samples_per_site * ensemble.num_sites / max_loops)
    thin_by = max(1, math.ceil(nsamples_per_loop / snapshots_per_loop))
    loop_count = 0
    while mod_factor > mod_factor_threshold and loop_count < max_loops:
        print(f"[p={ratio:4.2f}]  loop {loop_count}  ln f={mod_factor:8.2e}")
        sampler.run(nsamples_per_loop, occ_enc, thin_by=thin_by, progress=False)
        filepath = os.path.join('samplers', f"sampler_{round(1000 * ratio)}.npz")
        data = tc.sampler_data.dump_sampler_data(sampler, filepath)
        mod_factor = data.mod_factor_trace[-1]
        occ_enc = None
        loop_count += 1

    status = "converged" if mod_factor <= mod_factor_threshold else "GAVE UP"
    print(f"[p={ratio:4.2f}] {status} after {loop_count} loop(s); ln f={mod_factor:8.2e}")

def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--nprocs", type=int, default=33, help="number of parallel processes (default 17)")
    p.add_argument("--supercell_size", type=int, default=6, help="super-cell size (default 6x6x6)")
    p.add_argument("--num_wl_bins", type=int, default=200, help="number of Wang-Landau bins (default 200)")
    p.add_argument("--snapshots_per_loop",  type=int, default=100, help="number of random snapshots per ratio")
    p.add_argument("--n_samples_per_site",  type=int, default=10_000_000, help="number of Wang-Landau samples per site")
    args = p.parse_args(argv)

    # ── input parameters ────────────────────────────────────────────────
    seed_root = np.random.SeedSequence(42) # master seed
    replace_element = "Mg"
    ratios = list(np.linspace(0.1, 0.9, 33, endpoint=True))
    new_elements=("Mg", "Fe")
    filename = f"{''.join(new_elements)}O_ensemble{args.supercell_size}.json.gz"
    ensemble, _ = loadfn(filename)

    child_seeds = [int(x) for x in seed_root.generate_state(len(ratios)).tolist()]
    print(f"Using {args.nprocs} workers for parallel sampling.")
    Parallel(n_jobs=args.nprocs, backend="loky")(
        delayed(_run_wl_to_convergence)(ratio, seed, ensemble, args.num_wl_bins, args.window, replace_element,
                                        new_elements, args.snapshots_per_loop, args.n_samples_per_site
                                        ) for ratio, seed in zip(ratios, child_seeds)
    )

if __name__ == "__main__":
    main()
