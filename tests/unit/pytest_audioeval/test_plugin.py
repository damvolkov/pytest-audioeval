"""Tests for plugin registration."""

from __future__ import annotations

import pytest


async def test_plugin_registered(pytestconfig: pytest.Config) -> None:
    """Plugin is registered."""
    pm = pytestconfig.pluginmanager
    assert pm.has_plugin("pytest_audioeval.plugin") or pm.has_plugin("audioeval")


async def test_cli_options_registered(pytestconfig: pytest.Config) -> None:
    """CLI options are registered with correct defaults."""
    assert pytestconfig.getoption("--audioeval-wer") == 0.2
    assert pytestconfig.getoption("--audioeval-cer") == 0.15
    assert pytestconfig.getoption("--audioeval-mos") == 3.0
