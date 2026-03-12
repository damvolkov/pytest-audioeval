"""Generate embedded audio samples from TTS service."""
from __future__ import annotations

import asyncio
import io
from pathlib import Path

import httpx
import numpy as np
import soundfile as sf

_TTS_URL = "http://localhost:45130/v1/audio/speech"
_OUTPUT_DIR = Path(__file__).resolve().parent.parent / "src" / "pytest_audioeval" / "samples" / "audio"
_TARGET_RATE = 16_000
_TTS_RATE = 24_000

_SAMPLES: dict[str, dict[str, str]] = {
    "en": {
        "hello_world": "Hello world.",
        "quick_brown_fox": "The quick brown fox jumps over the lazy dog.",
        "counting": "One two three four five six seven eight nine ten.",
    },
}


def _resample(data: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """Linear interpolation resample. O(n)."""
    if src_rate == dst_rate:
        return data
    ratio = dst_rate / src_rate
    new_length = int(len(data) * ratio)
    return np.interp(np.linspace(0, len(data) - 1, new_length), np.arange(len(data)), data).astype(np.float32)


async def _generate() -> None:
    total = 0
    async with httpx.AsyncClient(timeout=30.0) as client:
        for lang, samples in _SAMPLES.items():
            lang_dir = _OUTPUT_DIR / lang
            lang_dir.mkdir(parents=True, exist_ok=True)

            for name, text in samples.items():
                wav_path = lang_dir / f"{name}.wav"
                txt_path = lang_dir / f"{name}.txt"

                response = await client.post(
                    _TTS_URL,
                    json={
                        "input": text,
                        "model": "kokoro",
                        "voice": "af_heart",
                        "response_format": "wav",
                        "stream": False,
                    },
                )
                response.raise_for_status()

                data, rate = sf.read(io.BytesIO(response.content), dtype="float32")
                resampled = _resample(data, rate, _TARGET_RATE)
                sf.write(wav_path, resampled, _TARGET_RATE, subtype="FLOAT")

                txt_path.write_text(text)
                total += 1
                print(f"  {lang}/{name}: {len(resampled)} samples, {len(resampled) / _TARGET_RATE:.2f}s")

    print(f"Generated {total} samples in {_OUTPUT_DIR}")


if __name__ == "__main__":
    asyncio.run(_generate())
