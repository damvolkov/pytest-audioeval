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

_SAMPLES: dict[str, dict[str, str]] = {
    "en": {
        "hello_world": "Hello world.",
        "quick_brown_fox": "The quick brown fox jumps over the lazy dog.",
        "counting": "One two three four five six seven eight nine ten.",
    },
    "es": {
        "hola_mundo": "Hola mundo.",
        "pangrama": "El veloz murciélago hindú comía feliz cardillo y kiwi.",
        "conteo": "Uno dos tres cuatro cinco seis siete ocho nueve diez.",
    },
    "fr": {
        "bonjour_monde": "Bonjour le monde.",
        "pangrama": "Portez ce vieux whisky au juge blond qui fume.",
        "comptage": "Un deux trois quatre cinq six sept huit neuf dix.",
    },
    "pt": {
        "ola_mundo": "Olá mundo.",
        "pangrama": "A rápida raposa marrom salta sobre o cachorro preguiçoso.",
        "contagem": "Um dois três quatro cinco seis sete oito nove dez.",
    },
    "it": {
        "ciao_mondo": "Ciao mondo.",
        "pangrama": "Quel vituperabile fez sbagliato ci provoca danni.",
        "conteggio": "Uno due tre quattro cinque sei sette otto nove dieci.",
    },
    "zh": {
        "nihao_shijie": "你好世界。",
        "pangrama": "天地玄黄宇宙洪荒日月盈昃辰宿列张。",
        "jishu": "一二三四五六七八九十。",
    },
    "ja": {
        "konnichiwa_sekai": "こんにちは世界。",
        "pangrama": "いろはにほへと散りぬるを我が世誰ぞ常ならん。",
        "kazoeru": "一二三四五六七八九十。",
    },
    "hi": {
        "namaste_duniya": "नमस्ते दुनिया.",
        "pangrama": "ऊँट के मुँह में जीरा कहावत बहुत पुरानी और मशहूर है।",
        "ginti": "एक दो तीन चार पांच छह सात आठ नौ दस.",
    },
    "pl": {
        "witaj_swiecie": "Witaj świecie.",
        "pangrama": "Pchnąć w tę łódź jeża lub ośm skrzyń fig.",
        "liczenie": "Jeden dwa trzy cztery pięć sześć siedem osiem dziewięć dziesięć.",
    },
}

_VOICES: dict[str, str] = {
    "en": "af_heart",
    "es": "ef_dora",
    "fr": "ff_siwis",
    "pt": "pf_dora",
    "it": "if_sara",
    "zh": "zf_xiaobei",
    "ja": "jf_alpha",
    "hi": "hf_alpha",
    "pl": "pf_dora",
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
    failed = 0
    async with httpx.AsyncClient(timeout=30.0) as client:
        for lang, samples in _SAMPLES.items():
            lang_dir = _OUTPUT_DIR / lang
            lang_dir.mkdir(parents=True, exist_ok=True)
            voice = _VOICES.get(lang, "af_heart")

            for name, text in samples.items():
                wav_path = lang_dir / f"{name}.wav"
                txt_path = lang_dir / f"{name}.txt"

                try:
                    response = await client.post(
                        _TTS_URL,
                        json={
                            "input": text,
                            "model": "kokoro",
                            "voice": voice,
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
                except Exception as exc:
                    failed += 1
                    print(f"  {lang}/{name}: FAILED — {exc}")

    print(f"\nGenerated {total} samples ({failed} failed) in {_OUTPUT_DIR}")


if __name__ == "__main__":
    asyncio.run(_generate())
