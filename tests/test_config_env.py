"""Fail-closed ``${ENV}`` expansion for user configuration (2026-10-18 sync)."""

from __future__ import annotations

from pathlib import Path

import pytest

from pes2ts_core.config_loader import ConfigError, load_config


def _write(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "user.yaml"
    path.write_text(body, encoding="utf-8")
    return path


def test_env_placeholder_is_expanded(monkeypatch, tmp_path):
    monkeypatch.setenv("PES2TS_TEST_ROOT", "/opt/acp")
    path = _write(
        tmp_path,
        "acp:\n"
        "  root: ${PES2TS_TEST_ROOT}\n"
        "  nested:\n"
        "    python: ${PES2TS_TEST_ROOT}/bin/python\n",
    )
    cfg = load_config(path)
    assert cfg["acp"]["root"] == "/opt/acp"
    assert cfg["acp"]["nested"]["python"] == "/opt/acp/bin/python"


def test_missing_env_is_typed_and_names_the_variable(monkeypatch, tmp_path):
    monkeypatch.delenv("PES2TS_TEST_MISSING", raising=False)
    path = _write(tmp_path, "root: ${PES2TS_TEST_MISSING}\n")
    with pytest.raises(ConfigError) as exc:
        load_config(path)
    assert "PES2TS_TEST_MISSING" in str(exc.value)


def test_shell_style_dollar_is_literal(tmp_path):
    path = _write(tmp_path, "literal: $NOT_AN_ENV_PLACEHOLDER\n")
    cfg = load_config(path)
    assert cfg["literal"] == "$NOT_AN_ENV_PLACEHOLDER"


def test_defaults_still_load_without_a_user_file():
    cfg = load_config(None)
    assert isinstance(cfg, dict)
    assert cfg  # bundled defaults are non-empty
