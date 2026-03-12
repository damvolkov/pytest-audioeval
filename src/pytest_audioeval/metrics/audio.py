"""Perceptual audio quality via PESQ."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Self

import numpy as np
from pesq import pesq as pesq_score


@dataclass(slots=True, frozen=True)
class AudioMetrics:
    """PESQ MOS (1-5 scale)."""

    mos: float
    sample_rate: int

    @classmethod
    def compute(cls, reference: np.ndarray, hypothesis: np.ndarray, *, sample_rate: int = 16_000) -> Self:
        """PESQ wideband comparison."""
        score = pesq_score(sample_rate, reference, hypothesis, "wb")
        return cls(mos=float(score), sample_rate=sample_rate)

    def assert_quality(self, *, min_mos: float = 3.0) -> None:
        """Raise AssertionError if MOS below threshold."""
        if self.mos < min_mos:
            raise AssertionError(f"PESQ MOS {self.mos:.2f} < {min_mos}")
