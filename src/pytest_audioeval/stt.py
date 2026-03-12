"""STT evaluation client — httpx-ws under the hood."""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, Self

import httpx
from httpx_ws import AsyncWebSocketClient, AsyncWebSocketSession

from pytest_audioeval.metrics.text import TextMetrics
from pytest_audioeval.samples.registry import AudioSample


@dataclass(slots=True)
class STTResult:
    """STT evaluation result with optional metrics."""

    hypothesis_text: str = ""
    text_metrics: TextMetrics | None = None
    latency_ms: float = 0.0
    chunks_received: int = 0
    fragments: list[str] = field(default_factory=list)

    def assert_quality(self, *, max_wer: float = 0.2, max_cer: float = 0.15) -> Self:
        """Assert STT quality. Chainable."""
        if self.text_metrics is None:
            raise AssertionError("No text metrics — call compute_metrics() first or provide a sample")
        self.text_metrics.assert_quality(max_wer=max_wer, max_cer=max_cer)
        return self

    def compute_metrics(self, reference: str) -> Self:
        """Compute WER/CER against reference. Chainable."""
        self.text_metrics = TextMetrics.compute(reference, self.hypothesis_text)
        return self


class STTSession:
    """Active WebSocket session for STT evaluation."""

    __slots__ = ("_result", "_sample", "_session", "_t0")

    def __init__(self, *, session: AsyncWebSocketSession, sample: AudioSample | None) -> None:
        self._session = session
        self._sample = sample
        self._t0 = time.perf_counter()
        self._result = STTResult()

    async def send_bytes(self, data: bytes) -> None:
        """Send binary audio data."""
        await self._session.send_bytes(data)

    async def send_text(self, data: str) -> None:
        """Send text (JSON config, END_OF_AUDIO, etc.)."""
        await self._session.send_text(data)

    async def send_sample(self, sample: AudioSample, *, chunk_ms: int = 200) -> None:
        """Stream sample in chunks with realistic pacing."""
        for chunk in sample.chunks(chunk_ms):
            await self._session.send_bytes(chunk)
            await asyncio.sleep(chunk_ms / 1000)

    async def receive_text(self, *, timeout: float | None = None) -> str:
        """Receive text frame and accumulate as fragment."""
        text = await self._session.receive_text(timeout=timeout)
        self._result.fragments.append(text)
        self._result.chunks_received += 1
        return text

    async def receive_bytes(self, *, timeout: float | None = None) -> bytes:
        """Receive binary frame."""
        return await self._session.receive_bytes(timeout=timeout)

    def result(self) -> STTResult:
        """Build STTResult from accumulated fragments."""
        self._result.latency_ms = (time.perf_counter() - self._t0) * 1000
        self._result.hypothesis_text = " ".join(self._result.fragments)
        if self._sample and self._result.hypothesis_text:
            self._result.compute_metrics(self._sample.reference_text)
        return self._result


class STTClient:
    """STT evaluation client — wraps httpx-ws."""

    __slots__ = ("_client", "_url", "_ws_client")

    def __init__(self, *, url: str) -> None:
        self._url = url
        self._client = httpx.AsyncClient()
        self._ws_client = AsyncWebSocketClient(self._client, keepalive_ping_interval_seconds=None)

    @asynccontextmanager
    async def ws(self, *, sample: AudioSample | None = None, **kwargs: Any) -> AsyncIterator[STTSession]:
        """Open WebSocket session for STT streaming."""
        async with self._ws_client.connect(self._url, **kwargs) as session:
            yield STTSession(session=session, sample=sample)

    async def aclose(self) -> None:
        """Cleanup HTTP client."""
        await self._client.aclose()
