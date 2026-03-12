# Configuration

## CLI Options

All configuration is passed via pytest CLI options. No config files needed.

```bash
pytest tests/ \
  --stt-url=ws://localhost:45120 \
  --tts-url=http://localhost:45130/v1/audio/speech \
  --audioeval-wer=0.15 \
  --audioeval-cer=0.10 \
  --audioeval-mos=3.5
```

| Option | Type | Default | Description |
|---|---|---|---|
| `--stt-url` | `str` | `None` | STT service URL (HTTP or WebSocket) |
| `--tts-url` | `str` | `None` | TTS service URL (HTTP or WebSocket) |
| `--audioeval-wer` | `float` | `0.2` | Maximum Word Error Rate threshold |
| `--audioeval-cer` | `float` | `0.15` | Maximum Character Error Rate threshold |
| `--audioeval-mos` | `float` | `3.0` | Minimum PESQ MOS threshold (1.0–5.0 scale) |

!!! tip "Optional services"
    Both `--stt-url` and `--tts-url` are optional. If omitted, `audioeval.stt` or `audioeval.tts` will be `None`. Guard your tests accordingly:

    ```python
    async def test_stt(audioeval: AudioEval):
        assert audioeval.stt is not None, "--stt-url required"
    ```

## Fixtures

The plugin registers two fixtures:

### `audioeval`

**Scope:** `session`

The main entry point. Provides access to STT/TTS clients and the sample registry.

```python
async def test_example(audioeval: AudioEval):
    audioeval.stt      # STTClient | None
    audioeval.tts      # TTSClient | None
    audioeval.samples  # SampleRegistry (always available)
```

The fixture creates clients based on CLI options and tears them down at the end of the session.

### `audioeval_thresholds`

**Scope:** `function`

A dictionary of CLI-driven quality thresholds:

```python
async def test_thresholds(audioeval_thresholds: dict[str, float]):
    max_wer = audioeval_thresholds["max_wer"]   # from --audioeval-wer
    max_cer = audioeval_thresholds["max_cer"]   # from --audioeval-cer
    min_mos = audioeval_thresholds["min_mos"]   # from --audioeval-mos
```

Use it to build configurable quality gates:

```python
async def test_stt_quality(audioeval, audioeval_thresholds):
    result = STTResult(hypothesis_text="hello world")
    result.compute_metrics("hello world")
    result.assert_quality(
        max_wer=audioeval_thresholds["max_wer"],
        max_cer=audioeval_thresholds["max_cer"],
    )
```

## pytest configuration

### Async mode

pytest-audioeval is fully async. Configure pytest-asyncio:

```toml title="pyproject.toml"
[tool.pytest.ini_options]
asyncio_mode = "auto"
asyncio_default_fixture_loop_scope = "session"
```

Setting `asyncio_default_fixture_loop_scope = "session"` is required for the session-scoped `audioeval` fixture to share the event loop with your tests.

### Import mode

If your test directory has files with the same name across subdirectories (e.g. `tests/unit/test_stt.py` and `tests/integration/test_stt.py`), use importlib mode:

```toml title="pyproject.toml"
[tool.pytest.ini_options]
addopts = ["--import-mode=importlib"]
```

## Environment patterns

### CI/CD

In CI, pass URLs as environment variables and forward them to pytest:

```yaml title=".github/workflows/test.yml"
jobs:
  integration:
    services:
      whisper:
        image: ghcr.io/collabora/whisperlive-gpu:latest
        ports: ["45120:9090"]
      kokoro:
        image: ghcr.io/remsky/kokoro-fastapi:latest
        ports: ["45130:8880"]
    steps:
      - run: |
          pytest tests/integration/ \
            --stt-url=ws://localhost:45120 \
            --tts-url=http://localhost:45130/v1/audio/speech
```

### Conditional tests

Skip tests when a service is unavailable:

```python
async def test_stt_only_when_available(audioeval: AudioEval):
    if audioeval.stt is None:
        pytest.skip("STT service not configured")
    # ... test logic
```

### Makefile targets

```makefile
test-integration:
	@pytest tests/integration/ \
		--stt-url=ws://localhost:45120 \
		--tts-url=http://localhost:45130/v1/audio/speech
```
