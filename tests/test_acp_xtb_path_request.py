"""Frozen-request builder tests for the ACP ``XtbPathSearch`` workflow.

Locks the ``pes2ts_xtb_path_request_v1`` payload contract:

* ``build_path_inp_text`` renders the literal ``$path`` control block from the
  frozen parameter order/defaults (the RecipeEquivalence anchor; the local
  runner these defaults were copied from was deleted by ADR-0002 X2'-C, so
  the explicit expected blocks below are the sole byte-level lock);
* ``build_path_request`` shape, ``uhf`` mapping, and provenance;
* ``request_sha256`` determinism and provenance exclusion;
* ``validate_path_request`` typed rejections — never silent defaults.
"""

from __future__ import annotations

import hashlib
import json

import pytest

from pes2ts_core.integration.acp.xtb_path_request import (
    ADAPTER_VERSION,
    REQUEST_SCHEMA_VERSION,
    build_path_inp_text,
    build_path_request,
    request_sha256,
    validate_path_request,
)
from pes2ts_core.utils.hashing import stable_json_dumps

START_XYZ = "2\nPES2TS start\nH 0.000000 0.000000 0.000000\nH 0.740000 0.000000 0.000000\n"
END_XYZ = "2\nPES2TS end\nH 0.000000 0.000000 0.000000\nH 1.500000 0.000000 0.000000\n"


def _request(**overrides):
    kwargs = {
        "reaction_id": "RXN_0000000001",
        "start_xyz_text": START_XYZ,
        "end_xyz_text": END_XYZ,
        "charge": 0,
        "multiplicity": 1,
        "path_config": {},
    }
    kwargs.update(overrides)
    return build_path_request(**kwargs)


# ---------------------------------------------------------------------------
# RecipeEquivalence: literal $path block rendering.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "overrides, expected_lines",
    [
        ({}, [
            "$path", "   nrun=1", "   npoint=50", "   anopt=10", "   kpush=0.003",
            "   kpull=-0.015", "   ppull=0.05", "   alp=0.5", "$end",
        ]),
        ({"nrun": 2}, [
            "$path", "   nrun=2", "   npoint=50", "   anopt=10", "   kpush=0.003",
            "   kpull=-0.015", "   ppull=0.05", "   alp=0.5", "$end",
        ]),
        ({"npoint": 25}, [
            "$path", "   nrun=1", "   npoint=25", "   anopt=10", "   kpush=0.003",
            "   kpull=-0.015", "   ppull=0.05", "   alp=0.5", "$end",
        ]),
        ({"anopt": 5, "npoint": 100}, [
            "$path", "   nrun=1", "   npoint=100", "   anopt=5", "   kpush=0.003",
            "   kpull=-0.015", "   ppull=0.05", "   alp=0.5", "$end",
        ]),
        ({"kpush": 0.01, "kpull": -0.02}, [
            "$path", "   nrun=1", "   npoint=50", "   anopt=10", "   kpush=0.01",
            "   kpull=-0.02", "   ppull=0.05", "   alp=0.5", "$end",
        ]),
        ({"ppull": 0.1, "alp": 1.2}, [
            "$path", "   nrun=1", "   npoint=50", "   anopt=10", "   kpush=0.003",
            "   kpull=-0.015", "   ppull=0.1", "   alp=1.2", "$end",
        ]),
        ({"nrun": 3, "npoint": 75, "anopt": 15, "kpush": 0.005,
          "kpull": -0.01, "ppull": 0.02, "alp": 0.4}, [
            "$path", "   nrun=3", "   npoint=75", "   anopt=15", "   kpush=0.005",
            "   kpull=-0.01", "   ppull=0.02", "   alp=0.4", "$end",
        ]),
        ({"nrun": 1, "npoint": 50, "anopt": 10, "kpush": 0.003,
          "kpull": -0.015, "ppull": 0.05, "alp": 0.5}, [
            "$path", "   nrun=1", "   npoint=50", "   anopt=10", "   kpush=0.003",
            "   kpull=-0.015", "   ppull=0.05", "   alp=0.5", "$end",
        ]),
        ({"unknown_key": 99, "nrun": 4}, [
            "$path", "   nrun=4", "   npoint=50", "   anopt=10", "   kpush=0.003",
            "   kpull=-0.015", "   ppull=0.05", "   alp=0.5", "$end",
        ]),
    ],
)
def test_build_path_inp_text_renders_frozen_contract_block(
    overrides, expected_lines,
) -> None:
    text = build_path_inp_text(overrides)
    assert text.splitlines() == expected_lines
    assert text.endswith("$end\n")


def test_build_path_inp_text_default_block_literal() -> None:
    """The frozen default block matches plan appendix A byte for byte."""
    assert build_path_inp_text({}) == (
        "$path\n"
        "   nrun=1\n"
        "   npoint=50\n"
        "   anopt=10\n"
        "   kpush=0.003\n"
        "   kpull=-0.015\n"
        "   ppull=0.05\n"
        "   alp=0.5\n"
        "$end\n"
    )


def test_build_path_inp_text_rejects_non_mapping() -> None:
    with pytest.raises(ValueError, match="mapping"):
        build_path_inp_text([("nrun", 1)])  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# build_path_request shape + uhf + provenance.
# ---------------------------------------------------------------------------
def test_build_path_request_shape_and_uhf_mapping() -> None:
    request = _request(multiplicity=3, charge=-1, path_config={"nrun": 2},
                       gfn_level=2, threads=8, timeout_seconds=900, seed=42,
                       extra_args=("--ALPHA", "0.5"), plan_sha256="a" * 64,
                       config_digest="cfg-digest")
    assert request["schema_version"] == REQUEST_SCHEMA_VERSION
    assert request["reaction_id"] == "RXN_0000000001"
    assert request["source"] == {
        "source_type": "xyz_text_pair",
        "start_xyz": START_XYZ,
        "end_xyz": END_XYZ,
        "charge": -1,
        "multiplicity": 3,
    }
    assert request["recipe"]["path_inp_text"] == build_path_inp_text({"nrun": 2})
    assert request["recipe"]["gfn_level"] == 2
    assert request["recipe"]["uhf"] == 2
    assert request["recipe"]["threads"] == 8
    assert request["recipe"]["timeout_seconds"] == 900
    assert request["recipe"]["seed"] == 42
    assert request["recipe"]["extra_args"] == ["--ALPHA", "0.5"]
    assert request["provenance"] == {
        "plan_sha256": "a" * 64,
        "config_digest": "cfg-digest",
        "request_sha256": request_sha256(request),
        "adapter_version": ADAPTER_VERSION,
    }
    validate_path_request(request)


@pytest.mark.parametrize(
    ("multiplicity", "expected_uhf"),
    [(1, 0), (2, 1), (3, 2), (5, 4)],
)
def test_uhf_is_max_zero_multiplicity_minus_one(multiplicity, expected_uhf) -> None:
    request = _request(multiplicity=multiplicity)
    assert request["recipe"]["uhf"] == expected_uhf


def test_build_path_request_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="reaction_id"):
        _request(reaction_id="  ")
    with pytest.raises(ValueError, match="start_xyz_text"):
        _request(start_xyz_text="")
    with pytest.raises(ValueError, match="end_xyz_text"):
        _request(end_xyz_text="   ")
    with pytest.raises(ValueError, match="charge"):
        _request(charge=True)
    with pytest.raises(ValueError, match="charge"):
        _request(charge="0")
    with pytest.raises(ValueError, match="multiplicity"):
        _request(multiplicity=1.5)
    with pytest.raises(ValueError, match="multiplicity must be >= 1"):
        _request(multiplicity=0)
    with pytest.raises(ValueError, match="threads"):
        _request(threads=0)
    with pytest.raises(ValueError, match="timeout_seconds"):
        _request(timeout_seconds=0)
    with pytest.raises(ValueError, match="extra_args"):
        _request(extra_args="--ALPHA")
    with pytest.raises(ValueError, match="extra_args"):
        _request(extra_args=("",))


# ---------------------------------------------------------------------------
# request_sha256: canonical, deterministic, provenance-excluded.
# ---------------------------------------------------------------------------
def test_request_sha256_deterministic_across_runs() -> None:
    first = _request()
    second = _request()
    assert request_sha256(first) == request_sha256(second)
    assert first["provenance"]["request_sha256"] == second["provenance"]["request_sha256"]


def test_request_sha256_excludes_provenance_and_matches_canonical_core() -> None:
    request = _request()
    mutated = json.loads(json.dumps(request))
    mutated["provenance"]["config_digest"] = "changed"
    mutated["provenance"]["plan_sha256"] = "f" * 64
    mutated["provenance"]["request_sha256"] = "0" * 64
    assert request_sha256(mutated) == request_sha256(request)
    core = {
        "schema_version": request["schema_version"],
        "reaction_id": request["reaction_id"],
        "source": request["source"],
        "recipe": request["recipe"],
    }
    expected = hashlib.sha256(stable_json_dumps(core).encode("utf-8")).hexdigest()
    assert request_sha256(request) == expected


def test_request_sha256_is_key_order_independent() -> None:
    request = _request()
    shuffled = {
        "recipe": request["recipe"],
        "provenance": request["provenance"],
        "reaction_id": request["reaction_id"],
        "schema_version": request["schema_version"],
        "source": request["source"],
    }
    assert request_sha256(shuffled) == request_sha256(request)


def test_request_sha256_changes_with_content() -> None:
    base = request_sha256(_request())
    assert request_sha256(_request(charge=1)) != base
    assert request_sha256(_request(multiplicity=3)) != base
    assert request_sha256(_request(path_config={"nrun": 2})) != base
    assert request_sha256(_request(seed=7)) != base


# ---------------------------------------------------------------------------
# validate_path_request: typed rejections, no silent defaults.
# ---------------------------------------------------------------------------
def test_validate_path_request_accepts_built_payload() -> None:
    validate_path_request(_request())


def test_validate_path_request_rejects_wrong_schema_version() -> None:
    bad = _request()
    bad["schema_version"] = "pes2ts_xtb_path_request_v0"
    with pytest.raises(ValueError, match="schema_version"):
        validate_path_request(bad)


def test_validate_path_request_rejects_empty_xyz() -> None:
    for field in ("start_xyz", "end_xyz"):
        bad = _request()
        bad["source"][field] = ""
        with pytest.raises(ValueError, match=field):
            validate_path_request(bad)
        bad["source"][field] = "   \n"
        with pytest.raises(ValueError, match=field):
            validate_path_request(bad)


def test_validate_path_request_rejects_non_int_charge_and_multiplicity() -> None:
    for field in ("charge", "multiplicity"):
        for value in ("0", 1.0, True, None):
            bad = _request()
            bad["source"][field] = value
            with pytest.raises(ValueError, match=field):
                validate_path_request(bad)
    bad = _request()
    bad["source"]["multiplicity"] = 0
    with pytest.raises(ValueError, match="multiplicity"):
        validate_path_request(bad)


def test_validate_path_request_rejects_missing_path_inp_text() -> None:
    for recipe in ({}, {"path_inp_text": ""}, {"path_inp_text": "nrun=1"},
                   {"path_inp_text": 42}, None, []):
        bad = _request()
        bad["recipe"] = recipe
        with pytest.raises(ValueError):
            validate_path_request(bad)


def test_validate_path_request_rejects_structural_damage() -> None:
    with pytest.raises(ValueError, match="mapping"):
        validate_path_request(["not", "a", "request"])  # type: ignore[arg-type]
    for missing in ("source", "recipe", "reaction_id", "schema_version"):
        bad = _request()
        del bad[missing]
        with pytest.raises(ValueError):
            validate_path_request(bad)
    bad = _request()
    bad["source"]["source_type"] = "xyz_pair"
    with pytest.raises(ValueError, match="source_type"):
        validate_path_request(bad)
