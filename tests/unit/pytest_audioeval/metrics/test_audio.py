"""Tests for AudioMetrics."""

from __future__ import annotations

import numpy as np
import pytest

from pytest_audioeval.metrics.audio import AudioMetrics


async def test_AudioMetrics_compute_identical_signals(dummy_audio_f32: np.ndarray) -> None:
    """Identical signals yield high MOS."""
    m = AudioMetrics.compute(dummy_audio_f32, dummy_audio_f32, sample_rate=16_000)
    assert m.mos >= 4.0
    assert m.sample_rate == 16_000


async def test_AudioMetrics_compute_noisy_signal(dummy_audio_f32: np.ndarray) -> None:
    """Noisy signal yields lower MOS."""
    rng = np.random.default_rng(42)
    noisy = dummy_audio_f32 + rng.normal(0, 0.3, size=dummy_audio_f32.shape).astype(np.float32)
    m = AudioMetrics.compute(dummy_audio_f32, noisy, sample_rate=16_000)
    assert m.mos < 4.0


async def test_AudioMetrics_assert_quality_passes() -> None:
    """No error when MOS above threshold."""
    m = AudioMetrics(mos=3.5, sample_rate=16_000)
    m.assert_quality(min_mos=3.0)


async def test_AudioMetrics_assert_quality_fails() -> None:
    """Raises AssertionError when MOS below threshold."""
    m = AudioMetrics(mos=2.0, sample_rate=16_000)
    with pytest.raises(AssertionError, match="PESQ MOS 2.00 < 3.0"):
        m.assert_quality(min_mos=3.0)


async def test_AudioMetrics_frozen() -> None:
    """AudioMetrics is immutable."""
    m = AudioMetrics(mos=4.0, sample_rate=16_000)
    with pytest.raises(AttributeError):
        m.mos = 1.0  # type: ignore[misc]
