"""AudioEval facade — main fixture interface."""

from __future__ import annotations

import contextlib

from pytest_audioeval.samples.registry import SampleRegistry
from pytest_audioeval.stt import STTClient
from pytest_audioeval.tts import TTSClient


class AudioEval:
    """Pytest fixture facade: audioeval.stt / audioeval.tts / audioeval.samples."""

    __slots__ = ("samples", "stt", "tts")

    def __init__(self, *, stt_url: str | None = None, tts_url: str | None = None) -> None:
        self.samples = SampleRegistry()
        self.stt: STTClient | None = STTClient(url=stt_url) if stt_url else None
        self.tts: TTSClient | None = TTSClient(url=tts_url) if tts_url else None

    async def aclose(self) -> None:
        """Cleanup clients."""
        for client in (self.stt, self.tts):
            if client is not None:
                with contextlib.suppress(RuntimeError):
                    await client.aclose()
