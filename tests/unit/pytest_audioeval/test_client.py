"""Tests for AudioEval facade."""

from __future__ import annotations

import io

import httpx
import pytest
from pytest_mock import MockerFixture

from pytest_audioeval.client import AudioEval
from pytest_audioeval.samples.registry import SampleLang, SampleRegistry
from pytest_audioeval.stt import STTClient
from pytest_audioeval.tts import TTSClient

##### CONSTRUCTION #####


async def test_AudioEval_with_urls() -> None:
    """AudioEval creates STT and TTS clients from URLs."""
    ae = AudioEval(stt_url="ws://fake:9090", tts_url="http://fake:8880")
    assert isinstance(ae.stt, STTClient)
    assert isinstance(ae.tts, TTSClient)
    assert isinstance(ae.samples, SampleRegistry)
    await ae.aclose()


async def test_AudioEval_without_urls() -> None:
    """AudioEval with no URLs has None clients."""
    ae = AudioEval()
    assert ae.stt is None
    assert ae.tts is None
    assert isinstance(ae.samples, SampleRegistry)
    await ae.aclose()


async def test_AudioEval_partial_urls() -> None:
    """AudioEval with only one URL."""
    ae = AudioEval(stt_url="ws://fake:9090")
    assert isinstance(ae.stt, STTClient)
    assert ae.tts is None
    await ae.aclose()


async def test_AudioEval_samples_populated() -> None:
    """AudioEval samples are auto-discovered."""
    ae = AudioEval()
    assert len(ae.samples) >= 3
    await ae.aclose()


##### ACLOSE #####


async def test_AudioEval_aclose_with_none_clients() -> None:
    """aclose() handles None clients gracefully."""
    ae = AudioEval()
    await ae.aclose()


async def test_AudioEval_aclose_suppresses_runtime_error() -> None:
    """aclose() suppresses RuntimeError from closed loops."""
    ae = AudioEval(stt_url="ws://fake:9090")
    # Force close so second close triggers error path
    await ae.aclose()


async def test_AudioEval_aclose_cleans_tmpdir(mocker: MockerFixture) -> None:
    """aclose() removes temp directory if created."""
    import soundfile as sf

    ae = AudioEval(tts_url="http://fake:8880")

    # Generate a fake WAV response
    buf = io.BytesIO()
    import numpy as np

    sf.write(buf, np.zeros(16_000, dtype="float32"), 24_000, format="WAV", subtype="FLOAT")
    wav_bytes = buf.getvalue()

    mock_response = mocker.MagicMock(spec=httpx.Response)
    mock_response.content = wav_bytes
    mock_response.raise_for_status = mocker.MagicMock()
    mocker.patch.object(httpx.AsyncClient, "post", return_value=mock_response)

    sample = await ae.create_sample("test phrase", lang=SampleLang.EN)
    assert ae._tmpdir is not None
    tmpdir = ae._tmpdir
    assert tmpdir.exists()
    assert sample.audio_path.exists()

    await ae.aclose()
    assert not tmpdir.exists()


##### CREATE SAMPLE #####


async def test_AudioEval_create_sample_generates_audio(mocker: MockerFixture) -> None:
    """create_sample() generates WAV and returns AudioSample."""
    import soundfile as sf

    ae = AudioEval(tts_url="http://fake:8880")

    buf = io.BytesIO()
    import numpy as np

    sf.write(buf, np.zeros(24_000, dtype="float32"), 24_000, format="WAV", subtype="FLOAT")

    mock_response = mocker.MagicMock(spec=httpx.Response)
    mock_response.content = buf.getvalue()
    mock_response.raise_for_status = mocker.MagicMock()
    mocker.patch.object(httpx.AsyncClient, "post", return_value=mock_response)

    sample = await ae.create_sample("Hello test", lang=SampleLang.EN, name="my_test")

    assert sample.name == "my_test"
    assert sample.lang == SampleLang.EN
    assert sample.reference_text == "Hello test"
    assert sample.audio_path.exists()
    assert sample.sample_rate == 16_000
    assert sample.duration_ms > 0
    assert "en_my_test" in ae.samples

    await ae.aclose()


async def test_AudioEval_create_sample_auto_name(mocker: MockerFixture) -> None:
    """create_sample() auto-generates name from text."""
    import soundfile as sf

    ae = AudioEval(tts_url="http://fake:8880")

    buf = io.BytesIO()
    import numpy as np

    sf.write(buf, np.zeros(16_000, dtype="float32"), 16_000, format="WAV", subtype="FLOAT")

    mock_response = mocker.MagicMock(spec=httpx.Response)
    mock_response.content = buf.getvalue()
    mock_response.raise_for_status = mocker.MagicMock()
    mocker.patch.object(httpx.AsyncClient, "post", return_value=mock_response)

    sample = await ae.create_sample("Hola mundo.")
    assert sample.name == "hola_mundo"

    await ae.aclose()


async def test_AudioEval_create_sample_requires_tts() -> None:
    """create_sample() raises without TTS configured."""
    ae = AudioEval()
    with pytest.raises(RuntimeError, match="--tts-url"):
        await ae.create_sample("hello")
    await ae.aclose()


async def test_AudioEval_create_sample_custom_payload(mocker: MockerFixture) -> None:
    """create_sample() accepts custom tts_json override."""
    import soundfile as sf

    ae = AudioEval(tts_url="http://fake:8880")

    buf = io.BytesIO()
    import numpy as np

    sf.write(buf, np.zeros(16_000, dtype="float32"), 16_000, format="WAV", subtype="FLOAT")

    mock_response = mocker.MagicMock(spec=httpx.Response)
    mock_response.content = buf.getvalue()
    mock_response.raise_for_status = mocker.MagicMock()
    mock_post = mocker.patch.object(httpx.AsyncClient, "post", return_value=mock_response)

    custom = {"model": "custom-model", "voice": "custom-voice", "response_format": "wav"}
    await ae.create_sample("custom test", tts_json=custom)

    _, kwargs = mock_post.call_args
    assert kwargs["json"]["model"] == "custom-model"
    assert kwargs["json"]["input"] == "custom test"

    await ae.aclose()
