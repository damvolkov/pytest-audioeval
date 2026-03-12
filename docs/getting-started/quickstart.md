# Quick Start

This guide walks you through writing your first STT and TTS tests with pytest-audioeval.

## Minimal pytest configuration

```toml title="pyproject.toml"
[tool.pytest.ini_options]
asyncio_mode = "auto"
```

All tests use the `audioeval` fixture, which is session-scoped and auto-configured from CLI options.

## STT — WebSocket streaming

The most common STT pattern: stream audio over WebSocket and check the transcript.

```python title="tests/test_stt.py"
import asyncio
import uuid

import orjson as json

from pytest_audioeval.client import AudioEval
from pytest_audioeval.metrics.text import TextMetrics


async def test_stt_websocket(audioeval: AudioEval):
    """Stream audio over WebSocket, get transcript, check WER."""
    assert audioeval.stt is not None, "--stt-url required"
    sample = audioeval.samples.en_hello_world

    async with audioeval.stt.ws(sample=sample) as session:
        # Send config (WhisperLive protocol)
        config = json.dumps({
            "uid": str(uuid.uuid4()),
            "language": "en",
            "task": "transcribe",
            "model": "large-v3-turbo",
            "use_vad": True,
        }).decode()
        await session.send_text(config)

        # Wait for server ready
        ready = await session.receive_text()
        assert "SERVER_READY" in ready

        # Stream audio in chunks with realistic pacing
        await session.send_sample(sample, chunk_ms=200)
        await asyncio.sleep(2)
        await session.send_text("END_OF_AUDIO")

        # Collect transcript segments
        segments = []
        while True:
            try:
                raw = await session.receive_text(timeout=5.0)
                msg = json.loads(raw)
                if "segments" in msg:
                    segments = msg["segments"]
            except Exception:
                break

    transcript = " ".join(
        seg.get("text", "").strip()
        for seg in segments
        if seg.get("text", "").strip()
    )
    assert len(transcript) > 0

    metrics = TextMetrics.compute(
        sample.reference_text.lower(),
        transcript.lower(),
    )
    assert metrics.wer <= 0.3
```

Run it:

```bash
pytest tests/test_stt.py --stt-url=ws://localhost:45120
```

## STT — HTTP batch

For STT services with an HTTP POST endpoint (e.g. OpenAI Whisper API):

```python title="tests/test_stt_batch.py"
async def test_stt_batch(audioeval: AudioEval):
    """Send audio via HTTP POST, get transcript back."""
    assert audioeval.stt is not None
    sample = audioeval.samples.en_hello_world

    response = await audioeval.stt.post(
        data=sample.audio_bytes(),
        headers={"Content-Type": "audio/wav"},
    )
    assert response.status_code == 200
    transcript = response.json()["text"]
    assert len(transcript) > 0
```

## TTS — HTTP batch

Send text, get audio back as a single response:

```python title="tests/test_tts.py"
import io
import soundfile as sf

from pytest_audioeval.client import AudioEval


async def test_tts_batch(audioeval: AudioEval):
    """POST text, get WAV audio back."""
    assert audioeval.tts is not None

    response = await audioeval.tts.post(json={
        "input": "Hello world.",
        "model": "kokoro",
        "voice": "af_heart",
        "response_format": "wav",
        "stream": False,
    })
    data, rate = sf.read(io.BytesIO(response.content), dtype="float32")
    assert rate == 24_000
    assert len(data) > 0
```

## TTS — Chunked streaming

For TTS services that stream audio in chunks:

```python title="tests/test_tts_streaming.py"
async def test_tts_streaming(audioeval: AudioEval):
    """Receive audio in chunks via HTTP streaming."""
    assert audioeval.tts is not None

    chunks = []
    async with audioeval.tts.stream(json={
        "input": "Streaming test.",
        "model": "kokoro",
        "voice": "af_heart",
        "response_format": "wav",
        "stream": True,
    }) as response:
        async for chunk in response.aiter_bytes():
            chunks.append(chunk)

    assert len(chunks) > 0
```

Run it:

```bash
pytest tests/test_tts.py --tts-url=http://localhost:45130/v1/audio/speech
```

## Using embedded samples

The plugin ships with ground-truth audio samples:

```python
async def test_browse_samples(audioeval: AudioEval):
    # Access by name
    sample = audioeval.samples.en_hello_world
    assert sample.reference_text == "Hello world."

    # Access audio data
    audio = sample.audio_numpy()      # float32 numpy array
    raw = sample.audio_bytes()         # raw WAV bytes
    chunks = sample.chunks(chunk_ms=100)  # PCM chunks for streaming

    # Browse catalog
    all_samples = audioeval.samples.all()
    en_samples = audioeval.samples.by_lang(SampleLang.EN)
```

## Standalone metrics

You can use metrics directly, without the `audioeval` fixture:

```python
from pytest_audioeval.metrics.text import TextMetrics

async def test_text_quality():
    metrics = TextMetrics.compute(
        reference="the quick brown fox",
        hypothesis="the quick brown dock",
    )
    assert metrics.wer < 0.3
    assert metrics.cer < 0.1
    assert metrics.substitutions == 1
```

## Next steps

- [Configuration](../guides/configuration.md) — CLI options, thresholds, fixtures
- [Advanced Usage](../guides/advanced-usage.md) — Roundtrips, custom samples, SSE, WebSocket TTS
- [API Reference](../reference/) — Full class and method documentation
