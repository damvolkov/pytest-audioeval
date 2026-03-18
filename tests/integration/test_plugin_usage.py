"""Plugin usage showcase — every capability a user can test with pytest-audioeval.

Run with:
  pytest tests/integration/test_plugin_usage.py \
    --stt-url=ws://localhost:45120 \
    --tts-url=http://localhost:45130/v1/audio/speech
"""

from __future__ import annotations

import asyncio
import io
import uuid

import numpy as np
import orjson as json
import pytest
import soundfile as sf

from pytest_audioeval.client import AudioEval
from pytest_audioeval.metrics.text import TextMetrics
from pytest_audioeval.samples.registry import SampleLang
from pytest_audioeval.stt import AudioEncoding, STTResult

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


##### STT — BATCH (HTTP POST) #####


async def test_user_stt_batch() -> None:
    """audioeval.stt.post() — send audio via HTTP POST, get transcript back.

    For HTTP-based STT APIs (e.g. OpenAI Whisper API, faster-whisper-server).
    WhisperLive is WS-only, so this is skipped here.
    """
    pytest.skip("WhisperLive does not expose an HTTP POST endpoint — use stt.ws() instead")


##### STT — CHUNKED STREAMING #####


async def test_user_stt_streaming() -> None:
    """audioeval.stt.stream() — chunked HTTP streaming for STT.

    For STT servers that return transcription as chunked HTTP responses.
    WhisperLive is WS-only, so this is skipped here.
    """
    pytest.skip("WhisperLive does not expose an HTTP streaming endpoint — use stt.ws() instead")


##### STT — SSE #####


async def test_user_stt_sse() -> None:
    """audioeval.stt.sse() — Server-Sent Events streaming for STT.

    For STT servers that return transcription as SSE events.
    WhisperLive is WS-only, so this is skipped here.
    """
    pytest.skip("WhisperLive does not expose an SSE endpoint — use stt.ws() instead")


##### STT — WEBSOCKET #####


async def test_user_stt_ws(audioeval: AudioEval) -> None:
    """audioeval.stt.ws() — stream audio over WebSocket, get transcript, check WER."""
    assert audioeval.stt is not None, "--stt-url required"
    sample = audioeval.samples.en_quick_brown_fox

    async with audioeval.stt.ws(sample=sample) as session:
        config = json.dumps(
            {
                "uid": str(uuid.uuid4()),
                "language": "en",
                "task": "transcribe",
                "model": "large-v3-turbo",
                "use_vad": True,
            }
        ).decode()
        await session.send_text(config)

        ready = await session.receive_text()
        assert "SERVER_READY" in ready

        await session.send_sample(sample, chunk_ms=200, encoding=AudioEncoding.PCM16_BASE64)
        await asyncio.sleep(2)
        await session.send_text("END_OF_AUDIO")

        last_segments: list[dict[str, str | bool]] = []
        while True:
            try:
                raw = await session.receive_text(timeout=5.0)
                msg = json.loads(raw)
                if "segments" in msg:
                    last_segments = msg["segments"]
            except Exception:
                break

    transcript = " ".join(seg.get("text", "").strip() for seg in last_segments if seg.get("text", "").strip())
    assert len(transcript) > 0, "STT returned empty transcript"

    metrics = TextMetrics.compute(sample.reference_text.lower(), transcript.lower())
    assert metrics.wer <= 0.3, (
        f"WER {metrics.wer:.3f} > 0.3\n  reference:  {sample.reference_text!r}\n  transcript: {transcript!r}"
    )


##### STT — RESULT + CHAINABLE ASSERTIONS #####


async def test_user_stt_result() -> None:
    """STTResult — compute metrics and chain quality assertions."""
    result = STTResult(hypothesis_text="Hello world.")
    result.compute_metrics("Hello world.")
    result.assert_quality(max_wer=0.2, max_cer=0.15)

    assert result.text_metrics is not None
    assert result.text_metrics.wer == 0.0


##### TTS — BATCH (HTTP POST) #####


async def test_user_tts_batch(audioeval: AudioEval) -> None:
    """audioeval.tts.post() — single POST, get WAV back."""
    assert audioeval.tts is not None, "--tts-url required"

    payload = {
        "input": "Hello world.",
        "model": "kokoro",
        "voice": "af_heart",
        "response_format": "wav",
        "stream": False,
    }
    response = await audioeval.tts.post(json=payload)
    data, rate = sf.read(io.BytesIO(response.content), dtype="float32")
    assert rate == 24_000
    assert len(data) > 0


##### TTS — CHUNKED STREAMING #####


async def test_user_tts_streaming(audioeval: AudioEval) -> None:
    """audioeval.tts.stream() — chunked HTTP streaming."""
    assert audioeval.tts is not None, "--tts-url required"

    payload = {
        "input": "Streaming test.",
        "model": "kokoro",
        "voice": "af_heart",
        "response_format": "wav",
        "stream": True,
    }
    chunks: list[bytes] = []
    async with audioeval.tts.stream(json=payload) as response:
        async for chunk in response.aiter_bytes():
            chunks.append(chunk)
    assert len(chunks) > 0


##### TTS — SSE #####


async def test_user_tts_sse() -> None:
    """audioeval.tts.sse() — Server-Sent Events streaming.

    Kokoro returns audio/wav, not text/event-stream, so this is skipped here.
    For TTS servers that support SSE (e.g. custom OpenAI-compatible endpoints).
    """
    pytest.skip("Kokoro does not support SSE — returns audio/wav instead of text/event-stream")


##### TTS — WEBSOCKET #####


async def test_user_tts_ws() -> None:
    """audioeval.tts.ws() — WebSocket session for TTS streaming.

    For TTS servers that accept text over WebSocket and stream audio back.
    Kokoro does not expose a WebSocket endpoint, so this is skipped here.
    """
    pytest.skip("Kokoro does not expose a WebSocket endpoint — use tts.post() or tts.stream() instead")


##### ROUNDTRIP — TTS → STT #####


async def test_user_roundtrip(audioeval: AudioEval) -> None:
    """TTS → resample 24kHz→16kHz → STT → WER check."""
    assert audioeval.tts is not None, "--tts-url required"
    assert audioeval.stt is not None, "--stt-url required"

    sample = audioeval.samples.en_hello_world

    # TTS: generate audio
    payload = {
        "input": sample.reference_text,
        "model": "kokoro",
        "voice": "af_heart",
        "response_format": "wav",
        "stream": False,
    }
    response = await audioeval.tts.post(json=payload)
    tts_audio, tts_rate = sf.read(io.BytesIO(response.content), dtype="float32")
    audio_16k = _resample(tts_audio, tts_rate, 16_000)
    audio_bytes = audio_16k.tobytes()

    # STT: transcribe
    last_segments: list[dict[str, str | bool]] = []
    async with audioeval.stt.ws() as session:
        config = json.dumps(
            {
                "uid": str(uuid.uuid4()),
                "language": "en",
                "task": "transcribe",
                "model": "large-v3-turbo",
                "use_vad": True,
            }
        ).decode()
        await session.send_text(config)

        ready = await session.receive_text()
        assert "SERVER_READY" in ready

        chunk_size = 4096 * 4
        for i in range(0, len(audio_bytes), chunk_size):
            await session.send_bytes(audio_bytes[i : i + chunk_size])
            await asyncio.sleep(0.01)

        await asyncio.sleep(2)
        await session.send_bytes(b"END_OF_AUDIO")

        while True:
            try:
                raw = await session.receive_text(timeout=5.0)
                msg = json.loads(raw)
                if "segments" in msg:
                    last_segments = msg["segments"]
            except Exception:
                break

    transcript = " ".join(seg.get("text", "").strip() for seg in last_segments if seg.get("text", "").strip())
    assert len(transcript) > 0, "Roundtrip returned empty transcript"

    reference = sample.reference_text.lower().replace(".", "").replace(",", "")
    normalized = _normalize_digits(transcript.lower())
    metrics = TextMetrics.compute(reference, normalized)
    assert metrics.wer <= 0.5, (
        f"Roundtrip WER {metrics.wer:.3f} > 0.5\n  reference:  {reference!r}\n  transcript: {normalized!r}"
    )


##### METRICS — TEXT #####


async def test_user_metrics_text() -> None:
    """TextMetrics.compute() — WER, CER, substitutions."""
    reference = "the quick brown fox jumps over the lazy dog"
    hypothesis = "the quick brown fox jumps over the lazy dock"

    metrics = TextMetrics.compute(reference, hypothesis)
    assert metrics.wer < 0.15
    assert metrics.cer < 0.05
    assert metrics.substitutions == 1


##### SAMPLES — REGISTRY #####


async def test_user_samples_browse(audioeval: AudioEval) -> None:
    """Browse and filter embedded audio samples."""
    assert len(audioeval.samples) >= 3

    en_samples = audioeval.samples.by_lang(SampleLang.EN)
    assert len(en_samples) >= 3

    sample = audioeval.samples.en_hello_world
    assert sample.reference_text == "Hello world."
    assert sample.sample_rate == 16_000
    assert sample.duration_ms > 0


async def test_user_samples_audio(audioeval: AudioEval) -> None:
    """Access sample audio as numpy, raw bytes, or chunks."""
    sample = audioeval.samples.en_hello_world

    audio = sample.audio_numpy()
    assert len(audio) > 0

    raw = sample.audio_bytes()
    assert len(raw) > 0

    chunks = sample.chunks(chunk_ms=100)
    assert len(chunks) >= 1


##### THRESHOLDS — CLI-DRIVEN #####


async def test_user_thresholds(audioeval_thresholds: dict[str, float]) -> None:
    """CLI-driven thresholds from --audioeval-wer/cer/mos."""
    assert audioeval_thresholds["max_wer"] == 0.2
    assert audioeval_thresholds["max_cer"] == 0.15
    assert audioeval_thresholds["min_mos"] == 3.0
