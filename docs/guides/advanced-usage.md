# Advanced Usage

## Transport Protocols

Both `STTClient` and `TTSClient` support four transport protocols. Pick the one that matches your service.

### HTTP POST — `.post()`

Single request/response. Best for batch processing.

=== "STT"

    ```python
    async def test_stt_post(audioeval: AudioEval):
        sample = audioeval.samples.en_hello_world
        response = await audioeval.stt.post(
            data=sample.audio_bytes(),
            headers={"Content-Type": "audio/wav"},
        )
        transcript = response.json()["text"]
    ```

=== "TTS"

    ```python
    async def test_tts_post(audioeval: AudioEval):
        response = await audioeval.tts.post(json={
            "input": "Hello world.",
            "model": "kokoro",
            "voice": "af_heart",
            "response_format": "wav",
        })
        audio_bytes = response.content
    ```

### Chunked Streaming — `.stream()`

HTTP streaming for large responses. Yields `httpx.Response` for `aiter_bytes()` / `aiter_lines()`.

=== "STT"

    ```python
    async def test_stt_stream(audioeval: AudioEval):
        sample = audioeval.samples.en_hello_world
        async with audioeval.stt.stream(data=sample.audio_bytes()) as response:
            async for chunk in response.aiter_lines():
                partial = json.loads(chunk)
                print(partial["text"])
    ```

=== "TTS"

    ```python
    async def test_tts_stream(audioeval: AudioEval):
        chunks = []
        async with audioeval.tts.stream(json={
            "input": "Streaming test.",
            "model": "kokoro",
            "stream": True,
        }) as response:
            async for chunk in response.aiter_bytes():
                chunks.append(chunk)
    ```

### Server-Sent Events — `.sse()`

For SSE-compatible endpoints. Yields `httpx_sse.EventSource`.

=== "STT"

    ```python
    async def test_stt_sse(audioeval: AudioEval):
        sample = audioeval.samples.en_hello_world
        async with audioeval.stt.sse(data=sample.audio_bytes()) as event_source:
            async for event in event_source.aiter_sse():
                if event.event == "transcript":
                    print(event.data)
    ```

=== "TTS"

    ```python
    async def test_tts_sse(audioeval: AudioEval):
        async with audioeval.tts.sse(json={
            "input": "Hello via SSE.",
            "model": "kokoro",
        }) as event_source:
            async for event in event_source.aiter_sse():
                if event.event == "audio":
                    audio_chunk = base64.b64decode(event.data)
    ```

!!! note "Accept header"
    `.sse()` automatically sets `Accept: text/event-stream`. You don't need to add it manually.

### WebSocket — `.ws()`

Full-duplex streaming. STT returns an `STTSession` with built-in metrics; TTS returns a raw `AsyncWebSocketSession`.

=== "STT"

    ```python
    async def test_stt_ws(audioeval: AudioEval):
        sample = audioeval.samples.en_hello_world

        async with audioeval.stt.ws(sample=sample) as session:
            await session.send_sample(sample, chunk_ms=200)
            transcript = await session.receive_text(timeout=5.0)

        result = session.result()
        result.assert_quality(max_wer=0.2)
    ```

=== "TTS"

    ```python
    async def test_tts_ws(audioeval: AudioEval):
        async with audioeval.tts.ws() as session:
            await session.send_text(json.dumps({
                "text": "Hello via WebSocket.",
                "voice": "af_heart",
            }))
            audio = await session.receive_bytes()
    ```

## STTSession — Built-in Metrics

When using `stt.ws()`, you get an `STTSession` that tracks latency, accumulates fragments, and auto-computes metrics:

```python
async with audioeval.stt.ws(sample=sample) as session:
    await session.send_sample(sample)
    await session.receive_text()
    await session.receive_text()

result = session.result()
# result.hypothesis_text  — joined fragments
# result.latency_ms       — wall-clock time
# result.chunks_received  — number of receive_text calls
# result.fragments        — raw received texts
# result.text_metrics     — auto-computed if sample provided
```

### Chainable assertions

Both `compute_metrics()` and `assert_quality()` return `Self`:

```python
result = STTResult(hypothesis_text="hello world")
result.compute_metrics("hello world").assert_quality(max_wer=0.2, max_cer=0.15)
```

## TTS → STT Roundtrip

Test the full pipeline: synthesize audio, then transcribe it back and compare:

```python
import io
import numpy as np
import soundfile as sf

async def test_roundtrip(audioeval: AudioEval):
    assert audioeval.tts is not None
    assert audioeval.stt is not None

    sample = audioeval.samples.en_hello_world

    # 1. TTS: generate audio
    response = await audioeval.tts.post(json={
        "input": sample.reference_text,
        "model": "kokoro",
        "voice": "af_heart",
        "response_format": "wav",
    })
    tts_audio, tts_rate = sf.read(
        io.BytesIO(response.content), dtype="float32",
    )

    # 2. Resample if needed (e.g. 24kHz TTS → 16kHz STT)
    if tts_rate != 16_000:
        duration = len(tts_audio) / tts_rate
        new_len = int(duration * 16_000)
        indices = np.linspace(0, len(tts_audio) - 1, new_len)
        tts_audio = np.interp(
            indices, np.arange(len(tts_audio)), tts_audio,
        ).astype(np.float32)

    # 3. STT: transcribe
    async with audioeval.stt.ws() as session:
        chunk_size = 4096 * 4
        audio_bytes = tts_audio.tobytes()
        for i in range(0, len(audio_bytes), chunk_size):
            await session.send_bytes(audio_bytes[i:i + chunk_size])

    # 4. Verify
    from pytest_audioeval.metrics.text import TextMetrics
    metrics = TextMetrics.compute(
        sample.reference_text.lower(),
        transcript.lower(),
    )
    assert metrics.wer <= 0.5
```

## Audio Quality — PESQ MOS

Compare synthesized audio against reference using PESQ:

```python
from pytest_audioeval.metrics.audio import AudioMetrics

async def test_audio_quality():
    reference = np.random.randn(16_000).astype(np.float32)
    hypothesis = reference + 0.01 * np.random.randn(16_000).astype(np.float32)

    metrics = AudioMetrics.compute(
        reference, hypothesis, sample_rate=16_000,
    )
    metrics.assert_quality(min_mos=3.0)
    print(f"PESQ MOS: {metrics.mos:.2f}")
```

!!! info "PESQ requirements"
    PESQ operates on 16kHz or 8kHz audio. Both arrays must have the same length and sample rate.

## Custom Samples

Register your own ground-truth samples:

```python title="conftest.py"
from pathlib import Path
import pytest
from pytest_audioeval.client import AudioEval
from pytest_audioeval.samples.registry import AudioSample, SampleLang

@pytest.fixture(scope="session")
def audioeval_with_custom(audioeval: AudioEval) -> AudioEval:
    audioeval.samples.register(AudioSample(
        name="medical_term",
        lang=SampleLang.EN,
        reference_text="The patient presents with dyspnea.",
        audio_path=Path("tests/resources/medical_term.wav"),
        sample_rate=16_000,
    ))
    return audioeval
```

Then use in tests:

```python
async def test_medical_stt(audioeval_with_custom):
    sample = audioeval_with_custom.samples.en_medical_term
    # ... test with domain-specific audio
```

### Sample directory convention

Place WAV + TXT pairs in the directory structure:

```
samples/audio/
├── en/
│   ├── hello_world.wav
│   ├── hello_world.txt    # "Hello world."
│   ├── custom_phrase.wav
│   └── custom_phrase.txt  # "Your custom phrase."
└── es/
    ├── hola_mundo.wav
    └── hola_mundo.txt     # "Hola mundo."
```

The registry auto-discovers these on initialization. Access via `samples.en_hello_world`, `samples.es_hola_mundo`, etc.

## Embedded Samples

The plugin ships with three English ground-truth samples:

| Key | Reference Text | Duration |
|---|---|---|
| `en_hello_world` | "Hello world." | ~1.5s |
| `en_quick_brown_fox` | "The quick brown fox jumps over the lazy dog." | ~2.5s |
| `en_counting` | "One two three four five six seven eight nine ten." | ~2.0s |

All recorded at 16kHz, mono, float32.

```python
async def test_sample_access(audioeval: AudioEval):
    sample = audioeval.samples.en_quick_brown_fox

    sample.name            # "quick_brown_fox"
    sample.lang            # SampleLang.EN
    sample.reference_text  # "The quick brown fox..."
    sample.sample_rate     # 16_000
    sample.duration_ms     # ~2500

    sample.audio_numpy()   # np.ndarray (float32)
    sample.audio_bytes()   # raw WAV bytes
    sample.chunks(200)     # list[bytes] — 200ms PCM chunks
```

## Filtering samples

```python
from pytest_audioeval.samples.registry import SampleLang

# All English samples
en = audioeval.samples.by_lang(SampleLang.EN)

# Total count
total = len(audioeval.samples)

# Membership check
exists = "en_hello_world" in audioeval.samples
```
