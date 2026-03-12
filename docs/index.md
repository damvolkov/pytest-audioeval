# pytest-audioeval

**Pytest plugin for STT/TTS integration testing with httpx, metrics, and embedded audio samples.**

pytest-audioeval gives you a single `audioeval` fixture that exposes STT and TTS clients over four transport protocols (HTTP POST, chunked streaming, SSE, WebSocket), built-in quality metrics (WER, CER, PESQ MOS), and a catalog of ground-truth audio samples — all wired through pytest CLI options.

---

## At a Glance

```python
async def test_stt_transcription(audioeval):
    sample = audioeval.samples.en_hello_world

    async with audioeval.stt.ws(sample=sample) as session:
        await session.send_sample(sample)
        transcript = await session.receive_text()

    result = session.result()
    result.assert_quality(max_wer=0.2)
```

```python
async def test_tts_synthesis(audioeval):
    response = await audioeval.tts.post(json={
        "input": "Hello world.",
        "model": "kokoro",
        "voice": "af_heart",
        "response_format": "wav",
    })
    assert response.status_code == 200
    assert len(response.content) > 0
```

---

## Features

| Feature | Description |
|---|---|
| **4 transport protocols** | HTTP POST, chunked streaming, SSE, WebSocket — for both STT and TTS |
| **Text metrics** | WER and CER via [jiwer](https://github.com/jitsi/jiwer) with chainable assertions |
| **Audio metrics** | PESQ MOS via [pesq](https://github.com/ludlows/PESQ) for perceptual quality scoring |
| **Embedded samples** | Ground-truth WAV + transcription pairs, auto-discovered from `samples/audio/` |
| **CLI-driven thresholds** | `--audioeval-wer`, `--audioeval-cer`, `--audioeval-mos` configure quality gates |
| **Pure httpx** | Built on [httpx](https://www.python-httpx.org/), [httpx-ws](https://github.com/frankie567/httpx-ws), and [httpx-sse](https://github.com/florimondmanca/httpx-sse) |
| **Async-native** | All I/O is `async/await`, integrates with `pytest-asyncio` |
| **Typed** | Full PEP 561 type stubs, `py.typed` marker included |

---

## Protocol Matrix

Both `STTClient` and `TTSClient` expose the same four transport methods:

| Method | Protocol | Use case |
|---|---|---|
| `.post()` | HTTP POST | Batch request/response (e.g. Whisper API, Kokoro) |
| `.stream()` | Chunked HTTP | Streaming responses via `aiter_bytes()` / `aiter_lines()` |
| `.sse()` | Server-Sent Events | SSE-compatible endpoints via `EventSource` |
| `.ws()` | WebSocket | Full-duplex streaming (e.g. WhisperLive, real-time TTS) |

---

## Quick Links

<div class="grid cards" markdown>

- :material-download: **[Installation](getting-started/installation.md)** — Install with pip or uv
- :material-rocket-launch: **[Quick Start](getting-started/quickstart.md)** — Write your first audio test in 5 minutes
- :material-cog: **[Configuration](guides/configuration.md)** — CLI options, fixtures, thresholds
- :material-school: **[Advanced Usage](guides/advanced-usage.md)** — Roundtrips, custom samples, metrics
- :material-api: **[API Reference](reference/)** — Auto-generated from source

</div>
