"""Tests for TTSClient."""

from __future__ import annotations

import httpx
import pytest
from pytest_mock import MockerFixture

from pytest_audioeval.tts import TTSClient

##### TTS POST #####


async def test_TTSClient_post_returns_response(mocker: MockerFixture) -> None:
    """post() returns httpx.Response."""
    mock_response = mocker.MagicMock(spec=httpx.Response)
    mock_response.status_code = 200
    mock_response.raise_for_status = mocker.MagicMock()

    mocker.patch.object(httpx.AsyncClient, "post", return_value=mock_response)

    client = TTSClient(url="http://fake:8880/v1/audio/speech")
    response = await client.post(json={"input": "hello", "model": "kokoro"})

    assert response is mock_response
    await client.aclose()


async def test_TTSClient_post_raises_on_error(mocker: MockerFixture) -> None:
    """post() propagates HTTP errors."""
    mock_response = mocker.MagicMock(spec=httpx.Response)
    mock_response.raise_for_status = mocker.MagicMock(
        side_effect=httpx.HTTPStatusError("422", request=mocker.MagicMock(), response=mock_response),
    )

    mocker.patch.object(httpx.AsyncClient, "post", return_value=mock_response)

    client = TTSClient(url="http://fake:8880")
    with pytest.raises(httpx.HTTPStatusError):
        await client.post(json={"bad": "payload"})
    await client.aclose()


##### TTS STREAM #####


async def test_TTSClient_stream_yields_response(mocker: MockerFixture) -> None:
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

    client = TTSClient(url="http://fake:8880")
    received: list[bytes] = []
    async with client.stream(json={"input": "hello"}) as response:
        async for chunk in response.aiter_bytes():
            received.append(chunk)

    assert received == chunks
    await client.aclose()


##### TTS SSE #####


async def test_TTSClient_sse_yields_event_source(mocker: MockerFixture) -> None:
    """sse() yields EventSource wrapping the streaming response."""
    mock_response = mocker.AsyncMock(spec=httpx.Response)
    mock_response.raise_for_status = mocker.MagicMock()

    mock_cm = mocker.AsyncMock()
    mock_cm.__aenter__ = mocker.AsyncMock(return_value=mock_response)
    mock_cm.__aexit__ = mocker.AsyncMock(return_value=False)

    mock_stream = mocker.patch.object(httpx.AsyncClient, "stream", return_value=mock_cm)

    client = TTSClient(url="http://fake:8880")
    async with client.sse(json={"input": "hello"}) as event_source:
        assert event_source is not None

    # Verify Accept header was set
    _, kwargs = mock_stream.call_args
    assert kwargs["headers"]["Accept"] == "text/event-stream"
    await client.aclose()


##### TTS ACLOSE #####


async def test_TTSClient_aclose(mocker: MockerFixture) -> None:
    """aclose() closes httpx client."""
    mock_aclose = mocker.patch.object(httpx.AsyncClient, "aclose", return_value=None)
    client = TTSClient(url="http://fake:8880")
    await client.aclose()
    mock_aclose.assert_awaited_once()


async def test_TTSClient_slots() -> None:
    """TTSClient uses __slots__."""
    client = TTSClient(url="http://fake:8880")
    assert hasattr(client, "__slots__")
    with pytest.raises(AttributeError):
        client.nonexistent = True  # type: ignore[attr-defined]
    await client.aclose()
