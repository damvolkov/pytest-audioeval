# Installation

## Requirements

- Python **>= 3.13**
- A running STT and/or TTS service to test against

## Install the plugin

=== "uv"

    ```bash
    uv add --group dev pytest-audioeval
    ```

=== "pip"

    ```bash
    pip install pytest-audioeval
    ```

This pulls in the core dependencies:

| Package | Purpose |
|---|---|
| `pytest >= 8.0` | Test framework |
| `httpx >= 0.27` | Async HTTP client |
| `httpx-ws >= 0.8.2` | WebSocket transport |
| `httpx-sse >= 0.4.3` | Server-Sent Events transport |
| `jiwer >= 3.0` | WER/CER computation |
| `pesq >= 0.0.4` | PESQ MOS audio quality |
| `numpy >= 2.0` | Audio array operations |
| `soundfile >= 0.12` | WAV file I/O |

## Verify installation

After installing, verify the plugin is registered:

```bash
pytest --co -q
```

You should see the `audioeval` options in:

```bash
pytest --help | grep audioeval
```

```text
Audio evaluation options:
  --stt-url=STT_URL     STT service WebSocket URL
  --tts-url=TTS_URL     TTS service HTTP URL
  --audioeval-wer=AUDIOEVAL_WER
                        Max WER threshold
  --audioeval-cer=AUDIOEVAL_CER
                        Max CER threshold
  --audioeval-mos=AUDIOEVAL_MOS
                        Min PESQ MOS threshold
```

## Recommended dev dependencies

For development with pytest-audioeval, add the async test runner:

=== "uv"

    ```bash
    uv add --group dev pytest-asyncio
    ```

=== "pip"

    ```bash
    pip install pytest-asyncio
    ```

And configure pytest for async:

```toml title="pyproject.toml"
[tool.pytest.ini_options]
asyncio_mode = "auto"
```

## Infrastructure setup

pytest-audioeval tests against real STT/TTS services. A typical setup uses Docker Compose:

```yaml title="compose.integration.yml"
services:
  whisper-live:
    image: ghcr.io/collabora/whisperlive-gpu:latest
    ports:
      - "45120:9090"
    deploy:
      resources:
        reservations:
          devices:
            - capabilities: [gpu]

  kokoro-tts:
    image: ghcr.io/remsky/kokoro-fastapi:latest
    ports:
      - "45130:8880"
    deploy:
      resources:
        reservations:
          devices:
            - capabilities: [gpu]
```

```bash
docker compose -f compose.integration.yml up -d
```

Then run your tests pointing at the services:

```bash
pytest tests/ \
  --stt-url=ws://localhost:45120 \
  --tts-url=http://localhost:45130/v1/audio/speech
```
