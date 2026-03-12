"""TTS evaluation client — httpx + httpx-sse + httpx-ws under the hood."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import httpx
from httpx_sse import EventSource
from httpx_ws import AsyncWebSocketClient, AsyncWebSocketSession


class TTSClient:
    """TTS evaluation client — HTTP batch, streaming, and SSE."""

    __slots__ = ("_timeout", "_url")

    def __init__(self, *, url: str, timeout: float = 30.0) -> None:
        self._url = url
        self._timeout = timeout

    async def post(self, *, json: dict[str, Any] | None = None, **kwargs: Any) -> httpx.Response:
        """Batch POST to TTS endpoint. Returns raw httpx.Response."""
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.post(self._url, json=json, **kwargs)
        response.raise_for_status()
        return response

    @asynccontextmanager
    async def stream(self, *, json: dict[str, Any] | None = None, **kwargs: Any) -> AsyncIterator[httpx.Response]:
        """Chunked streaming POST. Yields httpx.Response for aiter_bytes/aiter_lines."""
        async with (
            httpx.AsyncClient(timeout=self._timeout) as client,
            client.stream("POST", self._url, json=json, **kwargs) as response,
        ):
            response.raise_for_status()
            yield response

    @asynccontextmanager
    async def sse(self, *, json: dict[str, Any] | None = None, **kwargs: Any) -> AsyncIterator[EventSource]:
        """SSE streaming POST. Yields EventSource for aiter_sse()."""
        headers = kwargs.pop("headers", {})
        headers["Accept"] = "text/event-stream"
        async with (
            httpx.AsyncClient(timeout=self._timeout) as client,
            client.stream("POST", self._url, json=json, headers=headers, **kwargs) as response,
        ):
            response.raise_for_status()
            yield EventSource(response)

    @asynccontextmanager
    async def ws(self, **kwargs: Any) -> AsyncIterator[AsyncWebSocketSession]:
        """Open WebSocket session for TTS streaming (e.g. WebSocket-based TTS servers)."""
        async with httpx.AsyncClient() as client:
            ws_client = AsyncWebSocketClient(client, keepalive_ping_interval_seconds=None)
            async with ws_client.connect(self._url, **kwargs) as session:
                yield session

    async def aclose(self) -> None:
        """No-op — clients are created per-call."""
