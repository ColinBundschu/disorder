from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import numpy as np
from smol.moca import Sampler


@dataclass
class SamplerData:
    """Minimal snapshot of a finished Wang-Landau sampler (full precision)."""
    entropy: np.ndarray   # log g(E)  (float64)
    histogram: np.ndarray   # final histogram (int64)
    energy_levels: np.ndarray   # bin centre energies (float64)
    mod_factor_trace: np.ndarray   # ln f per iteration (float64)
    bin_size: float
    min_E: float
    max_E: float

    @property
    def nbins(self) -> int:
        return self.entropy.size

    @property
    def normalized_dos(self) -> np.ndarray:
        """
        Return the *normalised* density of states (DOS).
        Any zero-entropy bins are left at exactly zero so they may be
        masked later for log plotting.
        """
        mask = self.entropy > 0 # zero/negative entropy → unvisited bin
        S_shift = self.entropy[mask] - self.entropy[mask].min()
        dos_normed = np.zeros_like(self.entropy, dtype=float)
        dos_normed[mask] = np.exp(S_shift - S_shift.max())
        dos_normed /= dos_normed.sum() # global normalisation (Σ DOS = 1)
        return dos_normed


def dump_sampler_data(sampler: Sampler, path: str | Path) -> SamplerData:
    """
    Extract entropy, histogram, ln f trace, energy grid, and window meta
    from `sampler`, save them loss-lessly to <path>.npz, and return the
    in-memory `SamplerData` object.
    """
    kernel = sampler.mckernels[0]
    data = SamplerData(
        entropy=sampler.samples.get_trace_value("entropy")[-1],
        histogram=sampler.samples.get_trace_value("histogram")[-1],
        energy_levels=kernel.levels,
        mod_factor_trace=sampler.samples.get_trace_value("mod_factor"),
        bin_size=float(kernel.bin_size),
        min_E=float(kernel.spec.min_enthalpy),
        max_E=float(kernel.spec.max_enthalpy),
    )

    path = Path(path).with_suffix(".npz")
    np.savez_compressed(path, **asdict(data))      # binary, no precision loss
    print(f"[dump_sampler_data] wrote {path}  "
          f"({path.stat().st_size/1024:.1f} kB)")
    return data


def load_sampler_data(path: str | Path) -> SamplerData:
    """
    Load the file created by `dump_sampler_data` and rebuild a `SamplerData`
    instance with the same dtypes.
    """
    path = Path(path).with_suffix(".npz")
    with np.load(path, allow_pickle=False) as npz:
        kwargs: dict[str, Any] = {k: npz[k] for k in npz.files}

    # Scalars (bin_size, min_E, max_E) are stored as 0-d arrays → convert
    for key in ("bin_size", "min_E", "max_E"):
        if isinstance(kwargs[key], np.ndarray):
            kwargs[key] = float(kwargs[key])

    return SamplerData(**kwargs)
