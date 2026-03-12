# Changelog

## 0.1.0

Initial release.

- **STTClient** with 4 transport protocols: `post()`, `stream()`, `sse()`, `ws()`
- **TTSClient** with 4 transport protocols: `post()`, `stream()`, `sse()`, `ws()`
- **STTSession** with built-in latency tracking, fragment accumulation, and auto-metrics
- **STTResult** with chainable `compute_metrics()` and `assert_quality()`
- **TextMetrics** — WER/CER via jiwer with detailed edit distance breakdown
- **AudioMetrics** — PESQ MOS via pesq for perceptual audio quality
- **SampleRegistry** — auto-discovery of ground-truth WAV + TXT pairs
- **Embedded samples** — 3 English ground-truth recordings (hello_world, quick_brown_fox, counting)
- **`audioeval` fixture** — session-scoped, CLI-configured facade
- **`audioeval_thresholds` fixture** — per-test quality gates from CLI options
- **CLI options**: `--stt-url`, `--tts-url`, `--audioeval-wer`, `--audioeval-cer`, `--audioeval-mos`
- Full PEP 561 typing with `py.typed` marker
