"""Tests for TextMetrics."""

from __future__ import annotations

import pytest

from pytest_audioeval.metrics.text import TextMetrics


async def test_TextMetrics_compute_identical_strings() -> None:
    """Perfect match yields zero error rates."""
    m = TextMetrics.compute("hello world", "hello world")
    assert m.wer == 0.0
    assert m.cer == 0.0
    assert m.substitutions == 0
    assert m.insertions == 0
    assert m.deletions == 0


async def test_TextMetrics_compute_with_substitution() -> None:
    """Single word substitution yields expected WER."""
    m = TextMetrics.compute("the cat sat", "the dog sat")
    assert m.wer == pytest.approx(1 / 3, abs=0.01)
    assert m.substitutions == 1


async def test_TextMetrics_compute_with_insertion() -> None:
    """Extra word yields insertion."""
    m = TextMetrics.compute("hello world", "hello big world")
    assert m.insertions >= 1


async def test_TextMetrics_compute_with_deletion() -> None:
    """Missing word yields deletion."""
    m = TextMetrics.compute("hello big world", "hello world")
    assert m.deletions >= 1


async def test_TextMetrics_assert_quality_passes_within_threshold() -> None:
    """No error when within thresholds."""
    m = TextMetrics(wer=0.1, cer=0.05, substitutions=1, insertions=0, deletions=0)
    m.assert_quality(max_wer=0.2, max_cer=0.15)


async def test_TextMetrics_assert_quality_fails_on_wer() -> None:
    """Raises AssertionError when WER exceeds threshold."""
    m = TextMetrics(wer=0.5, cer=0.05, substitutions=2, insertions=0, deletions=0)
    with pytest.raises(AssertionError, match="WER 0.500 > 0.2"):
        m.assert_quality(max_wer=0.2, max_cer=0.15)


async def test_TextMetrics_assert_quality_fails_on_cer() -> None:
    """Raises AssertionError when CER exceeds threshold."""
    m = TextMetrics(wer=0.1, cer=0.3, substitutions=0, insertions=1, deletions=0)
    with pytest.raises(AssertionError, match="CER 0.300 > 0.15"):
        m.assert_quality(max_wer=0.2, max_cer=0.15)


@pytest.mark.parametrize(
    ("reference", "hypothesis", "max_wer"),
    [
        ("one two three", "one two three", 0.0),
        ("one two three", "one two four", 0.4),
        ("hello", "goodbye", 1.0),
    ],
    ids=["perfect", "one-sub", "total-miss"],
)
async def test_TextMetrics_compute_parametrized(reference: str, hypothesis: str, max_wer: float) -> None:
    """Parametrized WER boundary checks."""
    m = TextMetrics.compute(reference, hypothesis)
    assert m.wer <= max_wer + 0.01


async def test_TextMetrics_frozen() -> None:
    """TextMetrics is immutable."""
    m = TextMetrics(wer=0.1, cer=0.05, substitutions=1, insertions=0, deletions=0)
    with pytest.raises(AttributeError):
        m.wer = 0.5  # type: ignore[misc]
