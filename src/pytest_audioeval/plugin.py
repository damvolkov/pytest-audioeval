"""pytest entry point — fixtures and hooks."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import pytest

from pytest_audioeval.client import AudioEval


def pytest_addoption(parser: Any) -> None:
    """Register CLI options for audioeval."""
    group = parser.getgroup("audioeval", "Audio evaluation options")
    group.addoption("--stt-url", default=None, help="STT service WebSocket URL")
    group.addoption("--tts-url", default=None, help="TTS service HTTP URL")
    group.addoption("--audioeval-wer", type=float, default=0.2, help="Max WER threshold")
    group.addoption("--audioeval-cer", type=float, default=0.15, help="Max CER threshold")
    group.addoption("--audioeval-mos", type=float, default=3.0, help="Min PESQ MOS threshold")


@pytest.fixture(scope="session")
async def audioeval(request: pytest.FixtureRequest) -> AsyncIterator[AudioEval]:
    """Session-scoped AudioEval fixture with CLI-driven configuration."""
    stt_url = request.config.getoption("--stt-url")
    tts_url = request.config.getoption("--tts-url")

    ae = AudioEval(stt_url=stt_url, tts_url=tts_url)
    yield ae
    await ae.aclose()


@pytest.fixture
def audioeval_thresholds(request: pytest.FixtureRequest) -> dict[str, float]:
    """Per-test threshold overrides from CLI."""
    return {
        "max_wer": request.config.getoption("--audioeval-wer"),
        "max_cer": request.config.getoption("--audioeval-cer"),
        "min_mos": request.config.getoption("--audioeval-mos"),
    }
