"""Integration tests for STT (WhisperLive) service via httpx-ws."""

from __future__ import annotations

import asyncio
import uuid

import httpx
import orjson as json
from httpx_ws import AsyncWebSocketClient

from pytest_audioeval.metrics.text import TextMetrics
from pytest_audioeval.samples.registry import SampleRegistry

_registry = SampleRegistry()


async def _stt_transcribe(ws_url: str, audio_bytes: bytes, *, language: str = "en") -> str:
    """Send audio to STT via httpx-ws, return last known transcription."""
    uid = str(uuid.uuid4())
    last_segments: list[dict[str, str | bool]] = []

    async with httpx.AsyncClient() as client:
        ws_client = AsyncWebSocketClient(client, keepalive_ping_interval_seconds=None)
        async with ws_client.connect(ws_url) as ws:
            config = json.dumps(
                {"uid": uid, "language": language, "task": "transcribe", "model": "large-v3-turbo", "use_vad": True}
            ).decode()
            await ws.send_text(config)

            ready_msg = json.loads(await ws.receive_text())
            assert ready_msg["message"] == "SERVER_READY"

            chunk_size = 4096 * 4
            for i in range(0, len(audio_bytes), chunk_size):
                await ws.send_bytes(audio_bytes[i : i + chunk_size])
                await asyncio.sleep(0.01)

            await asyncio.sleep(2)
            await ws.send_bytes(b"END_OF_AUDIO")

            while True:
                try:
                    raw = await ws.receive_text(timeout=5.0)
                    msg = json.loads(raw)
                    if "segments" in msg:
                        last_segments = msg["segments"]
                except Exception:
                    break

    return " ".join(seg.get("text", "").strip() for seg in last_segments if seg.get("text", "").strip())


async def test_stt_server_ready(stt_ws_url: str) -> None:
    """STT server accepts connection and sends SERVER_READY."""
    uid = str(uuid.uuid4())
    async with httpx.AsyncClient() as client:
        ws_client = AsyncWebSocketClient(client, keepalive_ping_interval_seconds=None)
        async with ws_client.connect(stt_ws_url) as ws:
            config = json.dumps(
                {"uid": uid, "language": "en", "task": "transcribe", "model": "large-v3-turbo", "use_vad": True}
            ).decode()
            await ws.send_text(config)
            msg = json.loads(await ws.receive_text())
            assert msg["message"] == "SERVER_READY"


async def test_stt_transcribes_hello_world(stt_ws_url: str) -> None:
    """STT transcribes 'Hello world.' with acceptable WER."""
    sample = _registry.en_hello_world
    transcript = await _stt_transcribe(stt_ws_url, sample.audio_numpy().tobytes())
    assert len(transcript) > 0

    metrics = TextMetrics.compute(sample.reference_text.lower(), transcript.lower())
    assert metrics.wer <= 0.5, f"WER too high: {metrics.wer:.3f}, got: {transcript!r}"


async def test_stt_transcribes_quick_brown_fox(stt_ws_url: str) -> None:
    """STT transcribes pangram with acceptable WER."""
    sample = _registry.en_quick_brown_fox
    transcript = await _stt_transcribe(stt_ws_url, sample.audio_numpy().tobytes())
    assert len(transcript) > 0

    metrics = TextMetrics.compute(sample.reference_text.lower(), transcript.lower())
    assert metrics.wer <= 0.3, f"WER too high: {metrics.wer:.3f}, got: {transcript!r}"


_DIGIT_TO_WORD = {
    "1": "one",
    "2": "two",
    "3": "three",
    "4": "four",
    "5": "five",
    "6": "six",
    "7": "seven",
    "8": "eight",
    "9": "nine",
    "10": "ten",
}


def _normalize_digits(text: str) -> str:
    """Replace digit strings with word equivalents for fair WER comparison."""
    words = text.replace(",", "").replace(".", "").split()
    return " ".join(_DIGIT_TO_WORD.get(w, w) for w in words)


async def test_stt_transcribes_counting(stt_ws_url: str) -> None:
    """STT transcribes counting sequence (digits normalized to words)."""
    sample = _registry.en_counting
    transcript = await _stt_transcribe(stt_ws_url, sample.audio_numpy().tobytes())
    assert len(transcript) > 0

    normalized = _normalize_digits(transcript.lower())
    reference = sample.reference_text.lower().replace(".", "")
    metrics = TextMetrics.compute(reference, normalized)
    assert metrics.wer <= 0.3, f"WER too high: {metrics.wer:.3f}, got: {transcript!r} -> {normalized!r}"
