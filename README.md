# pytest-audioeval

Pytest plugin for STT/TTS integration testing. Built on the httpx ecosystem (`httpx`, `httpx-ws`, `httpx-sse`) with built-in metrics, embedded ground-truth audio samples, and chainable assertions.

## Features

- **STT via WebSocket** — `audioeval.stt.ws()` streams audio, collects transcription
- **TTS via HTTP** — `audioeval.tts.post()` batch, `.stream()` chunked, `.sse()` Server-Sent Events
- **Text metrics** — WER, CER, substitutions, insertions, deletions (via `jiwer`)
- **Audio metrics** — PESQ MOS 1–5 scale (via `pesq`)
- **Embedded samples** — ground-truth audio + reference text pairs, multi-language ready
- **Chainable assertions** — `result.compute_metrics(ref).assert_quality(max_wer=0.2)`
- **CLI thresholds** — `--audioeval-wer`, `--audioeval-cer`, `--audioeval-mos`

## Install

```bash
uv add pytest-audioeval
```

## Quick Start

### STT — WebSocket

```python
import asyncio
import uuid
import orjson as json
from pytest_audioeval.client import AudioEval


async def test_user_stt_ws(audioeval: AudioEval) -> None:
    sample = audioeval.samples.en_hello_world

    async with audioeval.stt.ws(sample=sample) as session:
        config = json.dumps(
            {"uid": str(uuid.uuid4()), "language": "en", "task": "transcribe",
             "model": "large-v3-turbo", "use_vad": True}
        ).decode()
        await session.send_text(config)

        ready = await session.receive_text()
        assert "SERVER_READY" in ready

        await session.send_sample(sample, chunk_ms=200)
        await asyncio.sleep(2)
        await session.send_text("END_OF_AUDIO")

        # Collect transcription segments...
```

### TTS — Batch POST

```python
import io
import soundfile as sf
from pytest_audioeval.client import AudioEval


async def test_user_tts_batch(audioeval: AudioEval) -> None:
    response = await audioeval.tts.post(
        json={"input": "Hello world.", "model": "kokoro",
              "voice": "af_heart", "response_format": "wav", "stream": False},
    )
    data, rate = sf.read(io.BytesIO(response.content), dtype="float32")
    assert rate == 24_000
    assert len(data) > 0
```

### TTS — Chunked Streaming

```python
async def test_user_tts_streaming(audioeval: AudioEval) -> None:
    chunks = []
    async with audioeval.tts.stream(json={"input": "Hello.", ...}) as response:
        async for chunk in response.aiter_bytes():
            chunks.append(chunk)
    assert len(chunks) > 0
```

### TTS — Server-Sent Events

```python
async def test_user_tts_sse(audioeval: AudioEval) -> None:
    async with audioeval.tts.sse(json={"input": "Hello.", ...}) as event_source:
        async for sse in event_source.aiter_sse():
            print(sse.data)
```

### Text Metrics

```python
from pytest_audioeval.metrics.text import TextMetrics


async def test_user_metrics_text() -> None:
    metrics = TextMetrics.compute(
        reference="the quick brown fox jumps over the lazy dog",
        hypothesis="the quick brown fox jumps over the lazy dock",
    )
    assert metrics.wer < 0.15
    assert metrics.substitutions == 1
```

### STT Result — Chainable Assertions

```python
from pytest_audioeval.stt import STTResult


async def test_user_stt_result() -> None:
    result = STTResult(hypothesis_text="Hello world.")
    result.compute_metrics("Hello world.")
    result.assert_quality(max_wer=0.2, max_cer=0.15)
```

### Sample Registry

```python
from pytest_audioeval.samples.registry import SampleLang


async def test_user_samples_browse(audioeval: AudioEval) -> None:
    # All samples
    assert len(audioeval.samples) >= 3

    # Filter by language
    en_samples = audioeval.samples.by_lang(SampleLang.EN)

    # Attribute access: {lang}_{name}
    sample = audioeval.samples.en_hello_world
    assert sample.reference_text == "Hello world."

    # Audio access
    audio_f32 = sample.audio_numpy()        # numpy float32 array
    audio_raw = sample.audio_bytes()         # raw bytes
    chunks = sample.chunks(chunk_ms=200)     # chunked for streaming
```

### CLI Thresholds

```python
async def test_user_thresholds(audioeval_thresholds: dict[str, float]) -> None:
    assert audioeval_thresholds["max_wer"] == 0.2
    assert audioeval_thresholds["max_cer"] == 0.15
    assert audioeval_thresholds["min_mos"] == 3.0
```

## CLI Options

```bash
pytest --stt-url=ws://localhost:45120 --tts-url=http://localhost:45130/v1/audio/speech
pytest --audioeval-wer=0.15 --audioeval-cer=0.10 --audioeval-mos=3.5
```

| Option | Default | Description |
|---|---|---|
| `--stt-url` | `None` | STT service WebSocket URL |
| `--tts-url` | `None` | TTS service HTTP URL |
| `--audioeval-wer` | `0.2` | Max WER threshold |
| `--audioeval-cer` | `0.15` | Max CER threshold |
| `--audioeval-mos` | `3.0` | Min PESQ MOS threshold |

## Fixtures

| Fixture | Scope | Type | Description |
|---|---|---|---|
| `audioeval` | session | `AudioEval` | Main facade — `audioeval.stt`, `audioeval.tts`, `audioeval.samples` |
| `audioeval_thresholds` | function | `dict[str, float]` | CLI-driven threshold dict |

## Architecture

```
src/pytest_audioeval/
├── plugin.py              # pytest entry point (fixtures, CLI options)
├── client.py              # AudioEval facade
├── stt.py                 # STTClient (httpx-ws), STTSession, STTResult
├── tts.py                 # TTSClient (httpx + httpx-sse)
├── metrics/
│   ├── text.py            # TextMetrics — WER, CER via jiwer
│   └── audio.py           # AudioMetrics — PESQ MOS via pesq
└── samples/
    ├── registry.py        # SampleRegistry + AudioSample + SampleLang
    └── audio/en/          # Embedded ground-truth WAV + TXT pairs
```

### Clients

| Client | Transport | Methods |
|---|---|---|
| `STTClient` | `httpx-ws` | `.ws()` — WebSocket context manager yielding `STTSession` |
| `TTSClient` | `httpx` + `httpx-sse` | `.post()` batch, `.stream()` chunked, `.sse()` SSE |

### Metrics

| Metric | Class | Source | Range |
|---|---|---|---|
| Word Error Rate (WER) | `TextMetrics` | `jiwer` | 0.0 – 1.0+ |
| Character Error Rate (CER) | `TextMetrics` | `jiwer` | 0.0 – 1.0+ |
| Substitutions / Insertions / Deletions | `TextMetrics` | `jiwer` | 0 – N |
| PESQ MOS | `AudioMetrics` | `pesq` | 1.0 – 5.0 |

### Samples

Embedded ground-truth audio with reference transcriptions:

```
samples/audio/
└── en/                    # English (16kHz, float32)
    ├── hello_world.wav    # "Hello world."
    ├── quick_brown_fox.wav
    └── counting.wav       # "One, two, three, four, five."
```

Access: `audioeval.samples.en_hello_world`, `audioeval.samples.en_counting`, etc.

## Infrastructure

Integration tests require GPU-accelerated TTS/STT services:

```bash
make infra-up       # Start TTS (Kokoro) + STT (WhisperLive)
make infra-status   # Check health
make infra-logs     # View logs
make infra-down     # Stop services
```

| Service | Image | Port | Protocol |
|---|---|---|---|
| TTS (Kokoro) | `ghcr.io/remsky/kokoro-fastapi-gpu` | `45130` | HTTP |
| STT (WhisperLive) | `ghcr.io/collabora/whisperlive-gpu` | `45120` | WebSocket |

## Development

```bash
make install            # uv sync --dev
make lint               # ruff check + format
make test-unit          # unit tests (no services)
make test-integration   # integration tests (requires services)
make coverage           # coverage report (>90%)
```

## Requirements

- Python >= 3.13
- NVIDIA GPU + Docker with nvidia-container-toolkit (for integration tests)

## License

MIT
