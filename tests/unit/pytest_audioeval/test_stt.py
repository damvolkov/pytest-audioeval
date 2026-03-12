"""Tests for STTClient and STTSession."""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest
from pytest_mock import MockerFixture

from pytest_audioeval.samples.registry import AudioSample, SampleLang, SampleRegistry
from pytest_audioeval.stt import STTClient, STTResult, STTSession

##### STT RESULT #####


async def test_STTResult_defaults() -> None:
    """STTResult has sane defaults."""
    r = STTResult()
    assert r.hypothesis_text == ""
    assert r.text_metrics is None
    assert r.latency_ms == 0.0
    assert r.chunks_received == 0
    assert r.fragments == []


async def test_STTResult_compute_metrics() -> None:
    """compute_metrics() sets text_metrics. Chainable."""
    r = STTResult(hypothesis_text="hello world")
    returned = r.compute_metrics("hello world")
    assert returned is r
    assert r.text_metrics is not None
    assert r.text_metrics.wer == 0.0


async def test_STTResult_compute_metrics_with_error() -> None:
    """compute_metrics() detects substitution errors."""
    r = STTResult(hypothesis_text="hello dock")
    r.compute_metrics("hello world")
    assert r.text_metrics is not None
    assert r.text_metrics.wer > 0


async def test_STTResult_assert_quality_passes() -> None:
    """assert_quality() passes with good metrics. Chainable."""
    r = STTResult(hypothesis_text="hello world")
    r.compute_metrics("hello world")
    returned = r.assert_quality(max_wer=0.2)
    assert returned is r


async def test_STTResult_assert_quality_fails_no_metrics() -> None:
    """assert_quality() raises without metrics."""
    r = STTResult()
    with pytest.raises(AssertionError, match="No text metrics"):
        r.assert_quality()


async def test_STTResult_assert_quality_fails_bad_wer() -> None:
    """assert_quality() raises on high WER."""
    r = STTResult(hypothesis_text="goodbye world")
    r.compute_metrics("hello world")
    with pytest.raises(AssertionError, match="WER"):
        r.assert_quality(max_wer=0.1)


##### STT SESSION #####


async def test_STTSession_send_bytes(mocker: MockerFixture) -> None:
    """send_bytes() delegates to ws session."""
    mock_ws = mocker.AsyncMock()
    session = STTSession(session=mock_ws, sample=None)
    await session.send_bytes(b"audio-chunk")
    mock_ws.send_bytes.assert_awaited_once_with(b"audio-chunk")


async def test_STTSession_send_text(mocker: MockerFixture) -> None:
    """send_text() delegates to ws session."""
    mock_ws = mocker.AsyncMock()
    session = STTSession(session=mock_ws, sample=None)
    await session.send_text("config json")
    mock_ws.send_text.assert_awaited_once_with("config json")


async def test_STTSession_receive_text_accumulates(mocker: MockerFixture) -> None:
    """receive_text() accumulates fragments."""
    mock_ws = mocker.AsyncMock()
    mock_ws.receive_text = mocker.AsyncMock(side_effect=["hello", "world"])

    session = STTSession(session=mock_ws, sample=None)
    f1 = await session.receive_text()
    f2 = await session.receive_text()

    assert f1 == "hello"
    assert f2 == "world"
    assert session._result.chunks_received == 2
    assert session._result.fragments == ["hello", "world"]


async def test_STTSession_receive_bytes(mocker: MockerFixture) -> None:
    """receive_bytes() returns raw bytes."""
    mock_ws = mocker.AsyncMock()
    mock_ws.receive_bytes = mocker.AsyncMock(return_value=b"binary-data")

    session = STTSession(session=mock_ws, sample=None)
    data = await session.receive_bytes()
    assert data == b"binary-data"


async def test_STTSession_send_sample(mocker: MockerFixture) -> None:
    """send_sample() streams chunks with pacing."""
    mock_ws = mocker.AsyncMock()
    sample = AudioSample(
        name="test",
        lang=SampleLang.EN,
        reference_text="test",
        audio_path=SampleRegistry().en_hello_world.audio_path,
    )

    session = STTSession(session=mock_ws, sample=None)
    mocker.patch("pytest_audioeval.stt.asyncio.sleep", return_value=None)
    await session.send_sample(sample, chunk_ms=200)

    assert mock_ws.send_bytes.await_count > 0


async def test_STTSession_result_no_sample(mocker: MockerFixture) -> None:
    """result() without sample skips metrics."""
    mock_ws = mocker.AsyncMock()
    session = STTSession(session=mock_ws, sample=None)
    session._result.fragments = ["hello", "world"]

    result = session.result()
    assert result.hypothesis_text == "hello world"
    assert result.text_metrics is None
    assert result.latency_ms > 0


async def test_STTSession_result_with_sample(mocker: MockerFixture) -> None:
    """result() with sample auto-computes metrics."""
    mock_ws = mocker.AsyncMock()
    sample = AudioSample(
        name="test",
        lang=SampleLang.EN,
        reference_text="hello world",
        audio_path=Path("/tmp/fake.wav"),
    )

    session = STTSession(session=mock_ws, sample=sample)
    session._result.fragments = ["hello world"]

    result = session.result()
    assert result.text_metrics is not None
    assert result.text_metrics.wer == 0.0


##### STT CLIENT — POST #####


async def test_STTClient_post_returns_response(mocker: MockerFixture) -> None:
    """post() returns httpx.Response."""
    mock_response = mocker.MagicMock(spec=httpx.Response)
    mock_response.status_code = 200
    mock_response.raise_for_status = mocker.MagicMock()

    mocker.patch.object(httpx.AsyncClient, "post", return_value=mock_response)

    client = STTClient(url="http://fake:9090/v1/audio/transcriptions")
    response = await client.post(data=b"audio-bytes")

    assert response is mock_response


async def test_STTClient_post_raises_on_error(mocker: MockerFixture) -> None:
    """post() propagates HTTP errors."""
    mock_response = mocker.MagicMock(spec=httpx.Response)
    mock_response.raise_for_status = mocker.MagicMock(
        side_effect=httpx.HTTPStatusError("422", request=mocker.MagicMock(), response=mock_response),
    )

    mocker.patch.object(httpx.AsyncClient, "post", return_value=mock_response)

    client = STTClient(url="http://fake:9090")
    with pytest.raises(httpx.HTTPStatusError):
        await client.post(data=b"bad-audio")


##### STT CLIENT — STREAM #####


async def test_STTClient_stream_yields_response(mocker: MockerFixture) -> None:
    """stream() yields httpx.Response for chunked reading."""
    chunks = [b"chunk1", b"chunk2"]

    mock_response = mocker.AsyncMock(spec=httpx.Response)
    mock_response.raise_for_status = mocker.MagicMock()

    async def aiter_bytes():
        for c in chunks:
            yield c

    mock_response.aiter_bytes = aiter_bytes

    mock_cm = mocker.AsyncMock()
    mock_cm.__aenter__ = mocker.AsyncMock(return_value=mock_response)
    mock_cm.__aexit__ = mocker.AsyncMock(return_value=False)

    mocker.patch.object(httpx.AsyncClient, "stream", return_value=mock_cm)

    client = STTClient(url="http://fake:9090")
    received: list[bytes] = []
    async with client.stream(data=b"audio") as response:
        async for chunk in response.aiter_bytes():
            received.append(chunk)

    assert received == chunks


##### STT CLIENT — SSE #####


async def test_STTClient_sse_yields_event_source(mocker: MockerFixture) -> None:
    """sse() yields EventSource wrapping the streaming response."""
    mock_response = mocker.AsyncMock(spec=httpx.Response)
    mock_response.raise_for_status = mocker.MagicMock()

    mock_cm = mocker.AsyncMock()
    mock_cm.__aenter__ = mocker.AsyncMock(return_value=mock_response)
    mock_cm.__aexit__ = mocker.AsyncMock(return_value=False)

    mock_stream = mocker.patch.object(httpx.AsyncClient, "stream", return_value=mock_cm)

    client = STTClient(url="http://fake:9090")
    async with client.sse(data=b"audio") as event_source:
        assert event_source is not None

    # Verify Accept header was set
    _, kwargs = mock_stream.call_args
    assert kwargs["headers"]["Accept"] == "text/event-stream"


##### STT CLIENT — WEBSOCKET #####


async def test_STTClient_ws_context(mocker: MockerFixture) -> None:
    """ws() yields STTSession."""
    mock_ws_session = mocker.AsyncMock()
    mock_cm = mocker.AsyncMock()
    mock_cm.__aenter__ = mocker.AsyncMock(return_value=mock_ws_session)
    mock_cm.__aexit__ = mocker.AsyncMock(return_value=False)

    mocker.patch("pytest_audioeval.stt.AsyncWebSocketClient.connect", return_value=mock_cm)

    client = STTClient(url="ws://fake:9090")
    async with client.ws() as session:
        assert isinstance(session, STTSession)


async def test_STTClient_ws_with_sample(mocker: MockerFixture) -> None:
    """ws() passes sample to session."""
    mock_ws_session = mocker.AsyncMock()
    mock_cm = mocker.AsyncMock()
    mock_cm.__aenter__ = mocker.AsyncMock(return_value=mock_ws_session)
    mock_cm.__aexit__ = mocker.AsyncMock(return_value=False)

    mocker.patch("pytest_audioeval.stt.AsyncWebSocketClient.connect", return_value=mock_cm)

    sample = AudioSample(name="test", lang=SampleLang.EN, reference_text="hi", audio_path=Path("/tmp/x.wav"))

    client = STTClient(url="ws://fake:9090")
    async with client.ws(sample=sample) as session:
        assert session._sample is sample


##### STT CLIENT — ACLOSE #####


async def test_STTClient_aclose_is_noop() -> None:
    """aclose() is a no-op (clients created per-call)."""
    client = STTClient(url="ws://fake:9090")
    await client.aclose()


async def test_STTClient_slots() -> None:
    """STTClient uses __slots__."""
    client = STTClient(url="ws://fake:9090")
    assert hasattr(client, "__slots__")
    with pytest.raises(AttributeError):
        client.nonexistent = True  # type: ignore[attr-defined]
