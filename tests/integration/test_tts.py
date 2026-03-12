"""Integration tests for TTS (Kokoro) service via httpx."""

from __future__ import annotations

import io

import httpx
import pytest
import soundfile as sf


async def test_tts_health(tts_base_url: str) -> None:
    """TTS service health endpoint responds."""
    async with httpx.AsyncClient() as client:
        r = await client.get(f"{tts_base_url}/health")
        assert r.status_code == 200


async def test_tts_generates_wav(tts_speech_url: str) -> None:
    """TTS generates valid WAV audio."""
    payload = {
        "input": "Hello world.",
        "model": "kokoro",
        "voice": "af_heart",
        "response_format": "wav",
        "stream": False,
    }
    async with httpx.AsyncClient(timeout=30.0) as client:
        r = await client.post(tts_speech_url, json=payload)
    r.raise_for_status()
    data, rate = sf.read(io.BytesIO(r.content), dtype="float32")
    assert rate == 24_000
    assert len(data) > 0


async def test_tts_generates_pcm(tts_speech_url: str) -> None:
    """TTS generates raw PCM audio."""
    payload = {
        "input": "Testing PCM output.",
        "model": "kokoro",
        "voice": "af_heart",
        "response_format": "pcm",
        "stream": False,
    }
    async with httpx.AsyncClient(timeout=30.0) as client:
        r = await client.post(tts_speech_url, json=payload)
    r.raise_for_status()
    assert len(r.content) > 0


async def test_tts_streaming(tts_speech_url: str) -> None:
    """TTS streams audio chunks."""
    payload = {
        "input": "Streaming test.",
        "model": "kokoro",
        "voice": "af_heart",
        "response_format": "wav",
        "stream": True,
    }
    chunks: list[bytes] = []
    async with (
        httpx.AsyncClient(timeout=30.0) as client,
        client.stream(
            "POST",
            tts_speech_url,
            json=payload,
        ) as response,
    ):
        response.raise_for_status()
        async for chunk in response.aiter_bytes():
            chunks.append(chunk)
    assert len(chunks) > 0


_VOICE_CASES = [
    ("af_heart", "female-us"),
    ("am_adam", "male-us"),
    ("bf_emma", "female-uk"),
]


@pytest.mark.parametrize(
    ("voice", "label"),
    _VOICE_CASES,
    ids=[c[1] for c in _VOICE_CASES],
)
async def test_tts_multiple_voices(tts_speech_url: str, voice: str, label: str) -> None:
    """TTS generates audio for different voices."""
    payload = {
        "input": f"Voice test for {label}.",
        "model": "kokoro",
        "voice": voice,
        "response_format": "wav",
        "stream": False,
    }
    async with httpx.AsyncClient(timeout=30.0) as client:
        r = await client.post(tts_speech_url, json=payload)
    r.raise_for_status()
    data, rate = sf.read(io.BytesIO(r.content), dtype="float32")
    assert rate == 24_000
    assert len(data) > 0
