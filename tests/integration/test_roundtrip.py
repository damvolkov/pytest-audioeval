"""Integration tests for TTS-to-STT roundtrip evaluation."""

from __future__ import annotations

import asyncio
import io
import uuid

import httpx
import numpy as np
import orjson as json
import pytest
import soundfile as sf
from httpx_ws import AsyncWebSocketClient

from pytest_audioeval.metrics.text import TextMetrics
from pytest_audioeval.samples.registry import SampleRegistry

_registry = SampleRegistry()

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
    """Replace digit strings with word equivalents."""
    words = text.replace(",", "").replace(".", "").split()
    return " ".join(_DIGIT_TO_WORD.get(w, w) for w in words)


def _resample(audio: np.ndarray, from_rate: int, to_rate: int) -> np.ndarray:
    """Linear interpolation resampling. O(n)."""
    if from_rate == to_rate:
        return audio
    duration = len(audio) / from_rate
    new_len = int(duration * to_rate)
    indices = np.linspace(0, len(audio) - 1, new_len)
    return np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)


_ROUNDTRIP_CASES = [
    ("en_hello_world", 0.5),
    ("en_quick_brown_fox", 0.3),
    ("en_counting", 0.4),
]


@pytest.mark.parametrize(
    ("sample_key", "max_wer"),
    _ROUNDTRIP_CASES,
    ids=["hello", "pangram", "counting"],
)
async def test_tts_stt_roundtrip(tts_speech_url: str, stt_ws_url: str, sample_key: str, max_wer: float) -> None:
    """TTS generates audio → resample → STT transcribes → WER check."""
    sample = getattr(_registry, sample_key)

    # TTS: generate audio
    async with httpx.AsyncClient(timeout=30.0) as client:
        payload = {
            "input": sample.reference_text,
            "model": "kokoro",
            "voice": "af_heart",
            "response_format": "wav",
            "stream": False,
        }
        r = await client.post(tts_speech_url, json=payload)
    r.raise_for_status()
    tts_audio, tts_rate = sf.read(io.BytesIO(r.content), dtype="float32")

    # Resample 24kHz → 16kHz for STT
    audio_16k = _resample(tts_audio, tts_rate, 16_000)
    audio_bytes = audio_16k.tobytes()

    # STT: transcribe via httpx-ws
    uid = str(uuid.uuid4())
    last_segments: list[dict[str, str | bool]] = []

    async with httpx.AsyncClient() as client:
        ws_client = AsyncWebSocketClient(client, keepalive_ping_interval_seconds=None)
        async with ws_client.connect(stt_ws_url) as ws:
            config = json.dumps(
                {"uid": uid, "language": "en", "task": "transcribe", "model": "large-v3-turbo", "use_vad": True}
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

    transcript = " ".join(seg.get("text", "").strip() for seg in last_segments if seg.get("text", "").strip())
    assert len(transcript) > 0, f"Empty transcript for {sample_key}"

    # Normalize for fair comparison
    normalized = _normalize_digits(transcript.lower())
    reference = sample.reference_text.lower().replace(".", "").replace(",", "")

    metrics = TextMetrics.compute(reference, normalized)
    assert metrics.wer <= max_wer, (
        f"Roundtrip WER {metrics.wer:.3f} > {max_wer} for {sample_key}\n"
        f"  reference:  {reference!r}\n"
        f"  transcript: {normalized!r}"
    )
