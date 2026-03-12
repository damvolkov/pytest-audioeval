"""Root conftest — shared fixtures."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pytest_audioeval.samples.registry import AudioSample, SampleLang

_SAMPLES_DIR = Path(__file__).resolve().parent.parent / "src" / "pytest_audioeval" / "samples" / "audio"


@pytest.fixture(scope="session")
def samples_dir() -> Path:
    """Path to embedded audio samples."""
    return _SAMPLES_DIR


@pytest.fixture
def sample_hello_world() -> AudioSample:
    """Hello world audio sample fixture."""
    return AudioSample(
        name="hello_world",
        lang=SampleLang.EN,
        reference_text="Hello world.",
        audio_path=_SAMPLES_DIR / "en" / "hello_world.wav",
        sample_rate=16_000,
        duration_ms=1143,
    )


@pytest.fixture
def dummy_audio_f32() -> np.ndarray:
    """1 second of 440Hz sine wave at 16kHz, float32."""
    t = np.linspace(0, 1, 16_000, dtype=np.float32)
    return np.sin(2 * np.pi * 440 * t).astype(np.float32)
