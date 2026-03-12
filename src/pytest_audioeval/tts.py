"""TTS evaluation client — httpx + httpx-sse under the hood."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import httpx
from httpx_sse import EventSource


class TTSClient:
    """TTS evaluation client — HTTP batch, streaming, and SSE."""

    __slots__ = ("_client", "_url")

    def __init__(self, *, url: str, timeout: float = 30.0) -> None:
        self._url = url
        self._client = httpx.AsyncClient(timeout=timeout)

    async def post(self, *, json: dict[str, Any] | None = None, **kwargs: Any) -> httpx.Response:
        """Batch POST to TTS endpoint. Returns raw httpx.Response."""
        response = await self._client.post(self._url, json=json, **kwargs)
        response.raise_for_status()
        return response

    @asynccontextmanager
    async def stream(self, *, json: dict[str, Any] | None = None, **kwargs: Any) -> AsyncIterator[httpx.Response]:
        """Chunked streaming POST. Yields httpx.Response for aiter_bytes/aiter_lines."""
        async with self._client.stream("POST", self._url, json=json, **kwargs) as response:
            response.raise_for_status()
            yield response

    @asynccontextmanager
    async def sse(self, *, json: dict[str, Any] | None = None, **kwargs: Any) -> AsyncIterator[EventSource]:
        """SSE streaming POST. Yields EventSource for aiter_sse()."""
        headers = kwargs.pop("headers", {})
        headers["Accept"] = "text/event-stream"
        async with self._client.stream("POST", self._url, json=json, headers=headers, **kwargs) as response:
            response.raise_for_status()
            yield EventSource(response)

    async def aclose(self) -> None:
        """Cleanup HTTP client."""
        await self._client.aclose()
