"""Unit tests for the G2 contract layer: package constants, statuses,
failure codes, precedence, and the default configuration block."""

from __future__ import annotations

from pathlib import Path

import pytest

from pes2ts_core.config_loader import load_config
from pes2ts_core.g0.rejections import RejectionCode
from pes2ts_core.generation.execution.xtb_path import (
    EXIT_G2_FAILED,
    G2_DIRNAME,
    G2_PATHS_DIRNAME,
    SCHEMA_COVERAGE,
    SCHEMA_MANIFEST,
    SCHEMA_PATH,
)
from pes2ts_core.generation.execution.xtb_path.status import (
    DEFAULT_REVERSE_RETRY_TRIGGERS,
    FAILURE_PRECEDENCE,
    G2_FAILURE_CODES,
    G2_STATUSES,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

#: The twelve G2 rejection codes in the deterministic plan order.
G2_CODES: tuple[str, ...] = (
    "G2_NOT_ELIGIBLE",
    "G2_MISSING_G1_DOC",
    "G2_G1_NOT_VALID",
    "G2_ENDPOINT_MISMATCH",
    "G2_ASSEMBLY_FAILED",
    "G2_ASSEMBLY_COLLISION",
    "G2_XTB_FAILED",
    "G2_ENDPOINT_NOT_REACHED",
    "G2_TOPOLOGY_DRIFT",
    "G2_ENERGY_INCOMPLETE",
    "G2_PATH_DISCONTINUOUS",
    "G2_COLLISION",
)


def test_g2_package_contract_constants() -> None:
    assert G2_DIRNAME == "g2"
    assert G2_PATHS_DIRNAME == "paths"
    assert SCHEMA_PATH == "g2_path_v1"
    assert SCHEMA_MANIFEST == "g2_manifest_v1"
    assert SCHEMA_COVERAGE == "g2_coverage_v1"
    assert EXIT_G2_FAILED == 24


def test_rejection_code_accepts_every_g2_code() -> None:
    for name in G2_CODES:
        code = RejectionCode(name)
        assert code.value == name


def test_g2_failure_codes_cover_the_twelve_codes_in_order() -> None:
    assert tuple(code.value for code in G2_FAILURE_CODES) == G2_CODES
    assert set(G2_FAILURE_CODES) == {RejectionCode(name) for name in G2_CODES}


def test_g2_statuses_are_exactly_valid_and_failed() -> None:
    assert G2_STATUSES == ("valid", "failed")


def test_failure_precedence_is_the_contract_order() -> None:
    assert FAILURE_PRECEDENCE == (
        RejectionCode.G2_XTB_FAILED,
        RejectionCode.G2_ENDPOINT_NOT_REACHED,
        RejectionCode.G2_TOPOLOGY_DRIFT,
        RejectionCode.G2_ENERGY_INCOMPLETE,
        RejectionCode.G2_PATH_DISCONTINUOUS,
        RejectionCode.G2_COLLISION,
    )


def test_default_reverse_retry_triggers_are_exact() -> None:
    assert DEFAULT_REVERSE_RETRY_TRIGGERS == (
        RejectionCode.G2_ENDPOINT_NOT_REACHED,
        RejectionCode.G2_TOPOLOGY_DRIFT,
        RejectionCode.G2_ENERGY_INCOMPLETE,
        RejectionCode.G2_PATH_DISCONTINUOUS,
        RejectionCode.G2_COLLISION,
    )
    assert (
        set(DEFAULT_REVERSE_RETRY_TRIGGERS)
        == set(FAILURE_PRECEDENCE) - {RejectionCode.G2_XTB_FAILED}
    )


def test_g2_config_defaults_load_with_contract_values() -> None:
    config = load_config()
    g2 = config["g2"]
    assert g2["eligible_path"] is None
    assert g2["shard_size"] == 1000
    assert "keep_trials" not in g2
    # Local-runner keys were removed with runner.py (ADR-0002 X2'-C): xTB PATH
    # executes through ACP, wired by the acp.* section.
    assert "executable" not in g2["xtb"]
    assert "fallback_paths" not in g2["xtb"]
    assert g2["xtb"]["threads"] == 4
    assert g2["xtb"]["timeout_seconds"] == 1800
    assert g2["xtb"]["seed"] == 42
    assert config["acp"] == {
        "root": None, "python": None, "config_path": None, "register": True,
    }
    assert g2["path"] == {
        "nrun": 1,
        "npoint": 50,
        "anopt": 10,
        "kpush": 0.003,
        "kpull": -0.015,
        "ppull": 0.05,
        "alp": 0.5,
    }
    assert g2["assembly"] == {
        "min_anchor_maps": 3,
        "anchor_tolerance": 0.001,
        "forming_min_distance": 2.0,
        "forming_target_distance": 3.0,
        "bond_tolerance": 0.45,
    }
    assert g2["validity"] == {
        "endpoint_rmsd_max": 0.5,
        "max_frame_step": 4.0,
        "min_frames": 8,
        "collision_min_distance": 0.8,
    }
    assert g2["reverse_retry"]["enabled"] is True
    assert g2["reverse_retry"]["trigger_codes"] == [
        code.value for code in DEFAULT_REVERSE_RETRY_TRIGGERS
    ]


def test_user_config_deep_merges_over_g2_defaults(tmp_path: Path) -> None:
    override = tmp_path / "override.yaml"
    override.write_text("g2:\n  xtb:\n    threads: 2\n", encoding="utf-8")
    g2 = load_config(override)["g2"]
    assert g2["xtb"]["threads"] == 2
    assert g2["path"]["npoint"] == 50


def test_g2_contract_modules_carry_no_truth_references() -> None:
    for relative in ("pes2ts_core/generation/execution/xtb_path/__init__.py", "pes2ts_core/generation/execution/xtb_path/status.py"):
        text = (PROJECT_ROOT / relative).read_text(encoding="utf-8")
        assert "ground_truth" not in text, relative
        assert "truth_sources" not in text, relative


def test_unknown_rejection_code_raises_value_error() -> None:
    with pytest.raises(ValueError):
        RejectionCode("NOPE")


def test_non_member_string_is_not_in_failure_precedence() -> None:
    assert "NOPE" not in FAILURE_PRECEDENCE
    assert RejectionCode.G2_NOT_ELIGIBLE not in FAILURE_PRECEDENCE


def test_reverse_retry_triggers_exclude_xtb_failed() -> None:
    assert RejectionCode.G2_XTB_FAILED not in DEFAULT_REVERSE_RETRY_TRIGGERS
