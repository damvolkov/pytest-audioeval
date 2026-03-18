"""Tests for SampleRegistry."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pytest_audioeval.samples.registry import AudioSample, SampleLang, SampleRegistry


async def test_SampleRegistry_discovers_embedded_samples() -> None:
    """Registry auto-discovers samples from audio directory."""
    registry = SampleRegistry()
    assert len(registry) >= 3
    assert "en_hello_world" in registry
    assert "en_quick_brown_fox" in registry
    assert "en_counting" in registry


async def test_SampleRegistry_getattr_access() -> None:
    """Attribute-style access returns correct sample."""
    registry = SampleRegistry()
    sample = registry.en_hello_world
    assert sample.lang == SampleLang.EN
    assert sample.name == "hello_world"
    assert "Hello world" in sample.reference_text


async def test_SampleRegistry_getattr_missing_raises() -> None:
    """Missing sample raises AttributeError with available list."""
    registry = SampleRegistry()
    with pytest.raises(AttributeError, match="not found"):
        _ = registry.nonexistent_sample


async def test_SampleRegistry_by_lang() -> None:
    """Filter by language returns correct subset."""
    registry = SampleRegistry()
    en_samples = registry.by_lang(SampleLang.EN)
    assert len(en_samples) >= 3
    assert all(s.lang == SampleLang.EN for s in en_samples)


async def test_SampleRegistry_register_custom(tmp_path: Path) -> None:
    """Custom sample registration works."""
    registry = SampleRegistry()
    initial_count = len(registry)

    wav_path = tmp_path / "test.wav"
    wav_path.write_bytes(b"fake")

    custom = AudioSample(
        name="custom_test",
        lang=SampleLang.EN,
        reference_text="custom test",
        audio_path=wav_path,
    )
    registry.register(custom)
    assert len(registry) == initial_count + 1
    assert "en_custom_test" in registry


async def test_SampleRegistry_repr() -> None:
    """Repr shows sample count."""
    registry = SampleRegistry()
    assert "SampleRegistry(" in repr(registry)


async def test_AudioSample_audio_numpy(sample_hello_world: AudioSample) -> None:
    """Audio loads as float32 numpy array."""
    data = sample_hello_world.audio_numpy()
    assert isinstance(data, np.ndarray)
    assert data.dtype == np.float32
    assert len(data) > 0


async def test_AudioSample_chunks(sample_hello_world: AudioSample) -> None:
    """Chunks split audio into correct sizes."""
    chunks = sample_hello_world.chunks(chunk_ms=200)
    assert len(chunks) >= 1
    samples_per_chunk = (16_000 * 200) // 1000
    bytes_per_chunk = samples_per_chunk * 4  # float32
    assert len(chunks[0]) == bytes_per_chunk


async def test_AudioSample_chunks_pcm16(sample_hello_world: AudioSample) -> None:
    """chunks_pcm16() returns int16 PCM chunks."""
    chunks = sample_hello_world.chunks_pcm16(chunk_ms=200)
    assert len(chunks) >= 1
    samples_per_chunk = (16_000 * 200) // 1000
    bytes_per_chunk = samples_per_chunk * 2  # int16 = 2 bytes
    assert len(chunks[0]) == bytes_per_chunk


async def test_AudioSample_audio_bytes(sample_hello_world: AudioSample) -> None:
    """Raw file bytes are non-empty."""
    raw = sample_hello_world.audio_bytes()
    assert len(raw) > 0


async def test_AudioSample_frozen() -> None:
    """AudioSample is immutable."""
    s = AudioSample(name="x", lang=SampleLang.EN, reference_text="x", audio_path=Path("/tmp/x.wav"))
    with pytest.raises(AttributeError):
        s.name = "y"  # type: ignore[misc]


async def test_SampleRegistry_all() -> None:
    """all() returns list of all samples."""
    registry = SampleRegistry()
    samples = registry.all()
    assert isinstance(samples, list)
    assert len(samples) == len(registry)
    assert all(isinstance(s, AudioSample) for s in samples)
