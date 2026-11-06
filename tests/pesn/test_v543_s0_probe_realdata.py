"""V5.4.3 WP3 — REAL-DATA S0-B / S0-C certificate.

Marked ``realdata``: ``pytest.ini``'s default ``addopts`` EXCLUDE this file
(``-m "not realdata ..."``), so it only runs when explicitly requested, inside
the GPU host (WSL ``pes2ts-gpu``)::

    PYTHONPATH=. python -m pytest -m realdata \
        tests/pesn/test_v543_s0_probe_realdata.py -q -s \
        -p no:cacheprovider --basetemp=.pytest_tmp_wp3/base

This is the plan §7.1 real-data leg of WP3:

* S0-B Latent Sufficiency Probe (:614–630) — the frozen A0 encoder of every
  campaign seed, the frozen Bridge context, four tiny head probes fitted on
  inner-train rows with the epoch chosen on a curve-level probe-val cut, and a
  single inner-val evaluation;
* S0-C Vector Basis Coverage (:632–642) — rank / singular values / condition
  number / projection residual (with and without channel 0) / teacher
  coefficients / leave-one-channel-out, per key atom of the inner-val frames,
  stratified by stratum and by breaking/forming/both.

The heavy lifting is ``scripts/agent_ops/v543_s0_probe.py`` (the same code path
the WP3 evidence report is produced with); this test drives it at a REDUCED
budget so it stays a certificate rather than a duplicate run.  Money quotes:

* the campaign's three seeds come from the config (``source.seeds``) and the
  checkpoints are SHA-verified against the pinned digests — a mismatch is a
  typed ``V543_PROBE_CKPT`` refusal, never a warning;
* ``bridge_spec_sha256`` must still be ``97f8d9af…`` (ADR-0034 §10 v1.1 ③: the
  BOND_SCALE freeze does NOT bump the spec);
* the plan's thresholds are REPORTED (10 % / 100 / 0.5) and the report carries
  no verdict — this test asserts that structure, it does not rule either;
* if the frozen data view or a pinned checkpoint is absent, the test SKIPS
  with an explicit note instead of pretending the diagnostic ran.

Set ``V543_S0_PROBE_OUT=<path>`` to persist the reduced-budget report (the
full-budget report of record is written by the CLI).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from pes2ts_core.pesn.models.v543 import LatentBridge  # noqa: E402
from pes2ts_core.pesn.offline.evaluation.v543_s0_probe import (  # noqa: E402
    PROBE_ORDER,
    S0_B_GO_RELATIVE_IMPROVEMENT,
    S0_C_CONDITION_THRESHOLD,
    S0_C_PROJECTION_RESIDUAL_THRESHOLD,
    STRATA_ORDER,
    TARGET_DELTA_U,
    V543ProbeError,
)

pytestmark = pytest.mark.realdata

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_VIEW = Path("/pes2ts_data/v53mp_50k_data")
#: ADR-0034 §10 v1.1 ③ — the frozen Bridge spec digest (must not move).
EXPECTED_BRIDGE_SPEC_SHA256 = (
    "97f8d9af0c6a0c58e4dd256c3cb0ee6c073e4512aecce7c5e4cb9ec552739a24")
#: Reduced budget of the certificate (the CLI's defaults are the report's).
CERT_BUDGET = {
    "train_frames": 1024,
    "probe_val_frames": 384,
    "eval_frames": 768,
    "init_seeds": [0],
    "max_epochs": 30,
}


def _require_assets() -> None:
    for name in ("data_manifest.json", "teacher_manifest.json",
                 "split_manifest_v53mp.json"):
        if not (DATA_VIEW / name).is_file():
            pytest.skip("frozen 50K data view unavailable in this environment "
                        f"({DATA_VIEW / name})")
    try:
        from scripts.agent_ops.v543_s0_probe import (
            CAMPAIGN_CONFIG,
            load_campaign_config,
            resolve_checkpoints,
        )

        campaign = load_campaign_config(PROJECT_ROOT / CAMPAIGN_CONFIG)
        resolve_checkpoints(campaign, campaign["source"]["seeds"])
    except V543ProbeError as error:
        pytest.skip(f"pinned A0 checkpoints unavailable ({error})")
    except Exception as error:  # noqa: BLE001 - reported in the skip reason
        pytest.skip(f"cannot pin the campaign assets ({type(error).__name__}: "
                    f"{error})")


@pytest.fixture(scope="module")
def report() -> dict:
    _require_assets()
    from scripts.agent_ops import v543_s0_probe as runner

    class Args:
        out = str(Path(os.environ.get("V543_S0_PROBE_OUT", "/tmp/unused.json")))
        seeds = None                     # the campaign's own seeds
        train_frames = CERT_BUDGET["train_frames"]
        probe_val_frames = CERT_BUDGET["probe_val_frames"]
        eval_frames = CERT_BUDGET["eval_frames"]
        init_seeds = CERT_BUDGET["init_seeds"]
        max_epochs = CERT_BUDGET["max_epochs"]

    return runner.run(Args())


def test_report_schema_and_frozen_identity(report: dict) -> None:
    assert report["schema"] == "pesn_v543_s0_probe_report_v1"
    assert report["ok"] is True
    assert report["bridge_spec_sha256"] == EXPECTED_BRIDGE_SPEC_SHA256
    assert report["bridge_spec_sha256"] == LatentBridge().bridge_spec_sha256()
    assert report["seeds"] and len(report["seeds"]) == len(
        report["checkpoints"])
    for seed, entry in report["checkpoints"].items():
        assert Path(entry["path"]).is_file()
        assert len(entry["sha256"]) == 64
    # the plan's thresholds are quoted, and the report says who decides
    assert report["thresholds"]["b2_vs_b4_relative_improvement_go"] == \
        S0_B_GO_RELATIVE_IMPROVEMENT
    assert report["thresholds"]["condition_number_rework"] == \
        S0_C_CONDITION_THRESHOLD
    assert report["thresholds"]["projection_residual_median_rework"] == \
        S0_C_PROJECTION_RESIDUAL_THRESHOLD
    assert "no_verdict" in report["thresholds"]
    # the BOND_SCALE freeze of ADR-0034 §10 v1.1 is recorded, not assumed
    assert report["bond_scale_note"]["bond_scale"] == 3.0


def test_probe_table_and_row_counts(report: dict) -> None:
    assert sorted(report["s0_b"]) == sorted(PROBE_ORDER)
    for probe in PROBE_ORDER:
        block = report["s0_b"][probe]
        assert block["purpose"]
        assert block["features"]
    for probe in ("B2", "B3", "B4"):
        block = report["s0_b"][probe]
        assert block["target"] == TARGET_DELTA_U
        pooled = block["metrics_pooled_inner_val"]
        assert pooled["n_rows"] > 0
        assert pooled["mae_atom_angstrom"] > 0
        assert block["baseline_m0_pooled"]["m0_mae_atom_angstrom"] > 0
        assert len(block["metrics_by_seed"]) == len(report["seeds"])
        for seed_metrics in block["metrics_by_seed"]:
            assert seed_metrics["n_train_rows"] > 0
            assert seed_metrics["n_val_rows"] > 0
            assert seed_metrics["selected_epoch"]
    # B1 carries both PROCESS targets and the δu reference for its clause
    b1 = report["s0_b"]["B1"]
    assert len(b1["targets"]) == 2
    assert b1["delta_u_m0_baseline"]["m0_mae_atom_angstrom"] > 0


def test_go_comparison_is_measurement_only(report: dict) -> None:
    go = report["s0_b_go_comparison"]
    assert go["go_threshold_relative_improvement"] == \
        S0_B_GO_RELATIVE_IMPROVEMENT
    assert "thresholds_only_no_verdict" in go
    improvement = go["relative_improvement_b2_vs_b4"]
    assert isinstance(improvement, float)
    assert go["mae_b2_angstrom"] == pytest.approx(
        report["s0_b"]["B2"]["metrics_pooled_inner_val"][
            "mae_atom_angstrom"])
    assert go["mae_b4_angstrom"] == pytest.approx(
        report["s0_b"]["B4"]["metrics_pooled_inner_val"][
            "mae_atom_angstrom"])
    expected = ((go["mae_b4_angstrom"] - go["mae_b2_angstrom"])
                / go["mae_b4_angstrom"])
    assert improvement == pytest.approx(expected)


def test_basis_coverage_reports_the_f7_bundle(report: dict) -> None:
    s0_c = report["s0_c"]
    pooled = s0_c["pooled_inner_val"]
    assert pooled["n_valid_rows"] > 0
    for key in ("rank", "singular_values", "condition_number",
                "effective_rank", "stable_rank", "coverage",
                "coverage_no_channel0", "projection_residual_ratio",
                "projection_residual_ratio_no_channel0",
                "teacher_coefficients", "leave_one_channel_out",
                "by_stratum", "by_intent_class", "thresholds_no_verdict"):
        assert key in pooled, key
    # §2 F7: a coverage number alone is not allowed to stand
    bundle = s0_c["f7_bundle"]
    for key in ("coverage_median", "condition_number_median",
                "effective_rank_median", "teacher_coefficient_abs_median",
                "leave_one_out_max_median_drop"):
        assert key in bundle
    assert sorted(pooled["by_stratum"]) == sorted(STRATA_ORDER)
    # rank is the plan's per-key quantity and cannot exceed the 3 available dims
    assert 0 <= pooled["rank"]["mean"] <= 3.0
    assert pooled["condition_number"]["n_finite"] >= 0
    thresholds = pooled["thresholds_no_verdict"]
    assert thresholds["condition_number_threshold"] == \
        S0_C_CONDITION_THRESHOLD
    assert thresholds["projection_residual_median_threshold"] == \
        S0_C_PROJECTION_RESIDUAL_THRESHOLD
    assert len(pooled["leave_one_channel_out"]["per_channel"]) == 8


def test_eligibility_evidence_is_present(report: dict) -> None:
    """The MOVE mask removes frames — the report must SHOW which ones."""
    eligibility = report["eligibility"]
    assert "by_seed" in eligibility
    for seed, block in eligibility["by_seed"].items():
        crosstab = block["inner_val"]
        assert crosstab["n_frames"] > 0
        assert crosstab["by_label_status"]
        assert block["n_key_rows_eligible"]["inner_val"] <= (
            block["n_key_rows_all"]["inner_val"])


def test_report_can_be_persisted(report: dict) -> None:
    out = os.environ.get("V543_S0_PROBE_OUT")
    if not out:
        pytest.skip("V543_S0_PROBE_OUT not set; nothing to persist")
    target = Path(out)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2, sort_keys=True,
                                 ensure_ascii=False), encoding="utf-8")
    assert target.is_file()
    loaded = json.loads(target.read_text(encoding="utf-8"))
    assert loaded["schema"] == report["schema"]
