"""AudioEval facade — main fixture interface."""

from __future__ import annotations

import contextlib
import io
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from pytest_audioeval.samples.registry import AudioSample, SampleLang, SampleRegistry
from pytest_audioeval.stt import STTClient
from pytest_audioeval.tts import TTSClient


class AudioEval:
    """Pytest fixture facade: audioeval.stt / audioeval.tts / audioeval.samples."""

    __slots__ = ("_tmpdir", "samples", "stt", "tts")

    def __init__(self, *, stt_url: str | None = None, tts_url: str | None = None) -> None:
        self.samples = SampleRegistry()
        self.stt: STTClient | None = STTClient(url=stt_url) if stt_url else None
        self.tts: TTSClient | None = TTSClient(url=tts_url) if tts_url else None
        self._tmpdir: Path | None = None

    async def create_sample(
        self,
        text: str,
        *,
        lang: SampleLang = SampleLang.EN,
        name: str | None = None,
        target_rate: int = 16_000,
        tts_json: dict[str, Any] | None = None,
    ) -> AudioSample:
        """Generate an AudioSample on-the-fly using the TTS service.

        Args:
            text: Text to synthesize.
            lang: Language tag for the sample.
            name: Sample name (auto-generated from text if omitted).
            target_rate: Target sample rate in Hz (resampled from TTS output).
            tts_json: Full TTS payload override. If omitted, uses sensible defaults.

        Returns:
            AudioSample ready for STT testing, auto-registered in the catalog.
        """
        if self.tts is None:
            raise RuntimeError("create_sample() requires --tts-url to be set")

        if self._tmpdir is None:
            self._tmpdir = Path(tempfile.mkdtemp(prefix="audioeval_"))

        sample_name = name or text.lower().replace(" ", "_").replace(".", "")[:40]
        wav_path = self._tmpdir / f"{lang}_{sample_name}.wav"

        payload = tts_json or {
            "input": text,
            "model": "kokoro",
            "voice": "af_heart",
            "response_format": "wav",
            "stream": False,
        }
        if "input" not in payload:
            payload["input"] = text

        response = await self.tts.post(json=payload)
        data, rate = sf.read(io.BytesIO(response.content), dtype="float32")

        if rate != target_rate:
            new_len = int(len(data) * target_rate / rate)
            data = np.interp(np.linspace(0, len(data) - 1, new_len), np.arange(len(data)), data).astype(np.float32)

        sf.write(wav_path, data, target_rate, subtype="FLOAT")

        sample = AudioSample(
            name=sample_name,
            lang=lang,
            reference_text=text,
            audio_path=wav_path,
            sample_rate=target_rate,
            duration_ms=int(len(data) / target_rate * 1000),
        )
        self.samples.register(sample)
        return sample

    async def aclose(self) -> None:
        """Cleanup clients and temp files."""
        for client in (self.stt, self.tts):
            if client is not None:
                with contextlib.suppress(RuntimeError):
                    await client.aclose()
        if self._tmpdir is not None:
            shutil.rmtree(self._tmpdir, ignore_errors=True)
