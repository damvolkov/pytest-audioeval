"""Tests for AudioEval facade."""

from __future__ import annotations

from pytest_audioeval.client import AudioEval
from pytest_audioeval.samples.registry import SampleRegistry
from pytest_audioeval.stt import STTClient
from pytest_audioeval.tts import TTSClient

##### CONSTRUCTION #####


async def test_AudioEval_with_urls() -> None:
    """AudioEval creates STT and TTS clients from URLs."""
    ae = AudioEval(stt_url="ws://fake:9090", tts_url="http://fake:8880")
    assert isinstance(ae.stt, STTClient)
    assert isinstance(ae.tts, TTSClient)
    assert isinstance(ae.samples, SampleRegistry)
    await ae.aclose()


async def test_AudioEval_without_urls() -> None:
    """AudioEval with no URLs has None clients."""
    ae = AudioEval()
    assert ae.stt is None
    assert ae.tts is None
    assert isinstance(ae.samples, SampleRegistry)
    await ae.aclose()


async def test_AudioEval_partial_urls() -> None:
    """AudioEval with only one URL."""
    ae = AudioEval(stt_url="ws://fake:9090")
    assert isinstance(ae.stt, STTClient)
    assert ae.tts is None
    await ae.aclose()


async def test_AudioEval_samples_populated() -> None:
    """AudioEval samples are auto-discovered."""
    ae = AudioEval()
    assert len(ae.samples) >= 3
    await ae.aclose()


##### ACLOSE #####


async def test_AudioEval_aclose_with_none_clients() -> None:
    """aclose() handles None clients gracefully."""
    ae = AudioEval()
    await ae.aclose()


async def test_AudioEval_aclose_suppresses_runtime_error() -> None:
    """aclose() suppresses RuntimeError from closed loops."""
    ae = AudioEval(stt_url="ws://fake:9090")
    # Force close so second close triggers error path
    await ae.aclose()
