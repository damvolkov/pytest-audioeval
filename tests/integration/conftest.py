"""Integration test conftest — real service fixtures."""

from __future__ import annotations

import contextlib

import httpx
import pytest

_TTS_URL = "http://localhost:45130"
_STT_URL = "ws://localhost:45120"


@pytest.fixture(scope="session")
def tts_base_url() -> str:
    """TTS service base URL."""
    return _TTS_URL


@pytest.fixture(scope="session")
def tts_speech_url() -> str:
    """TTS speech endpoint."""
    return f"{_TTS_URL}/v1/audio/speech"


@pytest.fixture(scope="session")
def stt_ws_url() -> str:
    """STT WebSocket URL."""
    return _STT_URL


@pytest.fixture(scope="session", autouse=True)
async def _check_services() -> None:
    """Skip all integration tests if services are unavailable."""
    tts_ok = False
    stt_ok = False

    with contextlib.suppress(Exception):
        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.get(f"{_TTS_URL}/health")
            tts_ok = r.status_code == 200

    with contextlib.suppress(Exception):
        async with httpx.AsyncClient(timeout=5.0) as client:
            from httpx_ws import AsyncWebSocketClient

            ws_client = AsyncWebSocketClient(client, keepalive_ping_interval_seconds=None)
            async with ws_client.connect(_STT_URL):
                stt_ok = True

    if not tts_ok:
        pytest.skip("TTS service unavailable at localhost:45130")
    if not stt_ok:
        pytest.skip("STT service unavailable at localhost:45120")
