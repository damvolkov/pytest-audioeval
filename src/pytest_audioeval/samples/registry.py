"""Sample catalog and ground-truth registry."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum, auto
from pathlib import Path

import numpy as np
import soundfile as sf


class SampleLang(StrEnum):
    """Supported sample languages."""

    EN = auto()
    ES = auto()
    FR = auto()
    DE = auto()
    PT = auto()
    IT = auto()
    RU = auto()
    ZH = auto()
    JA = auto()
    KO = auto()
    AR = auto()
    HI = auto()
    NL = auto()
    PL = auto()
    TR = auto()


@dataclass(slots=True, frozen=True)
class AudioSample:
    """Ground-truth pair: audio + expected transcription."""

    name: str
    lang: SampleLang
    reference_text: str
    audio_path: Path
    sample_rate: int = 16_000
    duration_ms: int = 0

    def audio_bytes(self) -> bytes:
        """Raw file bytes."""
        return self.audio_path.read_bytes()

    def audio_numpy(self) -> np.ndarray:
        """Load as float32 numpy array."""
        data, _ = sf.read(self.audio_path, dtype="float32")
        return data

    def chunks(self, chunk_ms: int = 200) -> list[bytes]:
        """Split audio into float32 PCM chunks for streaming. O(n/chunk_size)."""
        data = self.audio_numpy()
        samples_per_chunk = (self.sample_rate * chunk_ms) // 1000
        return [data[i : i + samples_per_chunk].tobytes() for i in range(0, len(data), samples_per_chunk)]


_SAMPLES_DIR = Path(__file__).resolve().parent / "audio"


class SampleRegistry:
    """Catalog of embedded audio fixtures. O(1) lookup by name."""

    __slots__ = ("_catalog",)

    def __init__(self) -> None:
        self._catalog: dict[str, AudioSample] = {}
        self._discover()

    def _discover(self) -> None:
        """Auto-register samples from directory convention: {lang}/{name}.wav + .txt."""
        if not _SAMPLES_DIR.exists():
            return
        for lang_dir in sorted(_SAMPLES_DIR.iterdir()):
            if not lang_dir.is_dir():
                continue
            lang_code = lang_dir.name
            if lang_code not in SampleLang.__members__.values():
                continue
            lang = SampleLang(lang_code)
            for wav_path in sorted(lang_dir.glob("*.wav")):
                txt_path = wav_path.with_suffix(".txt")
                if not txt_path.exists():
                    continue
                name = wav_path.stem
                key = f"{lang_code}_{name}"
                info = sf.info(wav_path)
                self._catalog[key] = AudioSample(
                    name=name,
                    lang=lang,
                    reference_text=txt_path.read_text().strip(),
                    audio_path=wav_path,
                    sample_rate=int(info.samplerate),
                    duration_ms=int(info.duration * 1000),
                )

    def __getattr__(self, name: str) -> AudioSample:
        """Attribute-style access: samples.en_hello_world."""
        if name in self._catalog:
            return self._catalog[name]
        available = ", ".join(sorted(self._catalog))
        raise AttributeError(f"Sample '{name}' not found. Available: {available}")

    def all(self) -> list[AudioSample]:
        """All registered samples."""
        return list(self._catalog.values())

    def by_lang(self, lang: SampleLang) -> list[AudioSample]:
        """Filter samples by language. O(n)."""
        return [s for s in self._catalog.values() if s.lang == lang]

    def register(self, sample: AudioSample) -> None:
        """Register custom project-specific samples."""
        self._catalog[f"{sample.lang}_{sample.name}"] = sample

    def __len__(self) -> int:
        return len(self._catalog)

    def __contains__(self, name: str) -> bool:
        return name in self._catalog

    def __repr__(self) -> str:
        return f"SampleRegistry({len(self._catalog)} samples)"
