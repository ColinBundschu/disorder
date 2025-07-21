from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import numpy as np
from smol.moca import Sampler


@dataclass
class SamplerData:
    entropy:            np.ndarray   # full window
    histogram:          np.ndarray
    energy_levels:      np.ndarray   # full window (same length)
    mod_factor_trace:   np.ndarray
    bin_size: float
    min_E:   float
    max_E:   float

    @property
    def nbins(self) -> int:
        return self.entropy.size

    @property
    def visited(self) -> np.ndarray:
        """Boolean mask of bins that were visited at least once."""
        return self.entropy > 0

    @property
    def normalized_dos(self) -> np.ndarray:
        """Σ DOS = 1 on the full grid (unvisited bins = 0)."""
        mask = self.visited
        S = self.entropy[mask] - self.entropy[mask].min()
        g = np.zeros_like(self.entropy, dtype=float)
        g[mask] = np.exp(S - S.max())
        g /= g.sum()
        return g

    @property
    def visited_energy(self) -> np.ndarray:
        """Energy grid restricted to visited bins."""
        return self.energy_levels[self.visited]



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
        energy_levels=kernel._levels,
        mod_factor_trace=sampler.samples.get_trace_value("mod_factor"),
        enthalpy_trace=sampler.samples.get_trace_value("enthalpy").astype(np.float64),
        bin_size=float(kernel.bin_size),
        min_E=float(kernel.spec.min_enthalpy),
        max_E=float(kernel.spec.max_enthalpy),
    )

    path = Path(path).with_suffix(".npz")
    np.savez_compressed(path, **asdict(data))      # binary, no precision loss
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
