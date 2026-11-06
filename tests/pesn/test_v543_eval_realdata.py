"""V5.4.3 — REAL-DATA certificate of the frozen V543 evaluator M0/P0 baseline.

Marked ``realdata`` (``pytest.ini``'s default ``addopts`` exclude it), so it only
runs inside the GPU host (WSL ``pes2ts-gpu``)::

    PYTHONPATH=. python -m pytest -m realdata \
        tests/pesn/test_v543_eval_realdata.py -q -s \
        -p no:cacheprovider --basetemp=.pytest_tmp_wp5/realdata

What it certifies (plan §9.5 :1035–1041):

* the three frozen A0 checkpoints load STRICT and their SHA matches the pinned
  campaign config (a mismatch is a typed refusal, never a warning);
* M0/P0 (the zero residual of §4.3 :226) evaluates on the REAL frozen inner-val
  slice, with the frozen word of §9.1 metrics all present and FINITE;
* the document carries the evaluator code SHA, the evaluator spec SHA, the
  Bridge spec SHA and the three checkpoint SHAs (the §9.5 freeze);
* the zero-residual identity checks hold (``u == u0``, ``q == tanh(q0_raw)``,
  ``e == e0``) and the MOVE eligibility evidence shows that
  ``label_status != "ok"`` frames were kept out of the MOVE denominators;
* a report can be re-persisted and re-read (JSON round-trip).

The runner is driven at a REDUCED budget (``CERT_BUDGET``) so this stays a
certificate, not a duplicate of the report of record (whose budget is the CLI
default).  If the frozen data view or a pinned checkpoint is absent the test
SKIPS with an explicit note instead of pretending the measurement ran.

Set ``V543_EVAL_OUT=<path>`` to persist the reduced-budget report.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path

import pytest

pytest.importorskip("torch")

from pes2ts_core.pesn.models.v543 import LatentBridge  # noqa: E402
from pes2ts_core.pesn.offline.evaluation import v543_eval as evaluator  # noqa: E402
from pes2ts_core.pesn.offline.evaluation.v543_eval import (  # noqa: E402
    V543EvalError,
)

pytestmark = pytest.mark.realdata

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_VIEW = Path("/pes2ts_data/v53mp_50k_data")
#: ADR-0034 §10 v1.1 ③ — the frozen Bridge spec digest (must not move).
EXPECTED_BRIDGE_SPEC_SHA256 = (
    "97f8d9af0c6a0c58e4dd256c3cb0ee6c073e4512aecce7c5e4cb9ec552739a24")
#: Reduced budget of the certificate (the CLI's default is the report's).
#: 32 curves x 4 frames keeps the multi-frame structure the §9.4 rules need
#: (>=3 q points per curve, ordering pairs within a curve) at certificate cost.
CERT_BUDGET = {"eval_curves": 32, "frames_per_curve": 4}


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
    except V543EvalError as error:
        pytest.skip(f"pinned A0 checkpoints unavailable ({error})")
    except Exception as error:  # noqa: BLE001 - reported in the skip reason
        pytest.skip(f"cannot pin the campaign assets ({type(error).__name__}: "
                    f"{error})")


@pytest.fixture(scope="module")
def report() -> dict:
    _require_assets()
    from scripts.agent_ops import v543_eval as runner

    class Args:
        out = str(Path(os.environ.get("V543_EVAL_OUT", "/tmp/unused.json")))
        seeds = None                     # the campaign's own three seeds
        eval_curves = CERT_BUDGET["eval_curves"]
        frames_per_curve = CERT_BUDGET["frames_per_curve"]
        no_basis = False
        no_strata = False

    return runner.run(Args())


def _walk_numbers(node, path: str = ""):
    if isinstance(node, dict):
        for key, value in node.items():
            yield from _walk_numbers(value, f"{path}.{key}")
    elif isinstance(node, list):
        for index, value in enumerate(node):
            yield from _walk_numbers(value, f"{path}[{index}]")
    elif isinstance(node, (int, float)) and not isinstance(node, bool):
        yield path, float(node)


def test_report_identity_and_frozen_shas(report: dict) -> None:
    assert report["schema"] == "pesn_v543_m0p0_baseline_v1"
    assert report["ok"] is True
    assert report["seeds"] == [17, 29, 43]
    # checkpoint SHAs are the pinned ones and the files exist
    for seed, entry in report["checkpoints"].items():
        assert Path(entry["path"]).is_file()
        assert len(entry["sha256"]) == 64
    # the Bridge spec digest is the frozen one (ADR-0034 §10 v1.1 ③)
    assert report["bridge_spec_sha256"] == EXPECTED_BRIDGE_SPEC_SHA256
    assert report["bridge_spec_sha256"] == LatentBridge().bridge_spec_sha256()
    # §9.5 freeze: code SHA (the actual file) + spec SHA (the formula block)
    module_path = PROJECT_ROOT / "pes2ts_core/pesn/offline/evaluation/v543_eval.py"
    assert report["evaluator"]["code_sha256"] == hashlib.sha256(
        module_path.read_bytes()).hexdigest()
    assert report["evaluator"]["spec_sha256"] == evaluator.evaluator_spec_sha256()
    assert report["evaluator"]["report_schema"] == "pesn_v543_eval_report_v1"
    # per-seed document SHAs are recorded for every seed
    assert sorted(report["per_seed_document_sha256"]) == sorted(
        str(seed) for seed in report["seeds"])
    for digest in report["per_seed_document_sha256"].values():
        assert len(digest) == 64


def test_every_seed_carries_a_full_eval_document(report: dict) -> None:
    for seed in report["seeds"]:
        document = report["per_seed"][str(seed)]
        assert document["schema"] == "pesn_v543_eval_report_v1"
        assert document["tau_end"] is None            # §1.5 :105
        assert document["per_curve"], "no per-curve rows"
        pooled = document["pooled"]
        for key in ("move", "process"):
            assert key in pooled
        # the §9.1 leaf table is complete for the H2/H3 metrics
        move = pooled["move"]
        for key in ("rmse_zero", "sign_accuracy_cluster_median",
                    "dead_zone_fraction", "atom_dead_fraction",
                    "fixed_coverage_direction_at_3",
                    "fixed_coverage_topk_overlap_at_3", "pair_delta_d_mae",
                    "cos_eff_median"):
            assert key in move, key
        assert move["fixed_coverage"]["k"] == 3
        assert move["fixed_coverage"]["aggregation"].startswith("frame mean")
        process = pooled["process"]
        for key in ("mae_curve_median", "n_eval_curves",
                    "median_points_per_curve", "frac_curves_ge3",
                    "frac_curves_ge5", "corr_curve_median",
                    "constant_q_baseline_mae"):
            assert key in process["q"], key
        for key in ("pairwise_accuracy_full", "pairwise_accuracy_cross_ts"):
            assert key in process["ordering"], key
        for key in ("auprc", "auroc", "brier", "ece",
                    "recall_at_fpr_0p05", "fpr_at_threshold"):
            assert key in process["endpoint"], key


def test_new_direction_metric_is_measured_and_distinct(report: dict) -> None:
    """The §9.2 metric is a real number and is NOT the V542 sign leaf."""
    for seed in report["seeds"]:
        move = report["per_seed"][str(seed)]["pooled"]["move"]
        direction = move["fixed_coverage_direction_at_3"]
        assert direction is not None
        assert 0.0 <= direction <= 1.0
        assert 0.0 <= move["fixed_coverage_topk_overlap_at_3"] <= 1.0
        assert 0.0 <= move["sign_accuracy_cluster_median"] <= 1.0
        assert direction != pytest.approx(move["sign_accuracy_cluster_median"])


def test_move_eligibility_evidence(report: dict) -> None:
    for seed in report["seeds"]:
        evidence = report["label_status_evidence"][str(seed)]
        assert evidence["n_frames"] > 0
        assert evidence["by_label_status"]
        document = report["per_seed"][str(seed)]
        validity = document["validity"]["move"]
        assert validity["n_frames"] == evidence["n_frames"]
        # the excluded frames are exactly the non-"ok" ones
        excluded = validity["n_frames_excluded_by_label_status"]
        expected = evidence["n_frames"] - evidence["by_label_status"].get(
            "ok", 0)
        assert excluded == expected
        assert validity["n_frames_move_eligible"] == \
            evidence["by_label_status"].get("ok", 0)


def test_zero_residual_identity_checks(report: dict) -> None:
    checks = report["identity_checks"]
    assert checks["all_ok"] is True
    for seed, entry in checks["by_seed"].items():
        assert entry["u_equals_u0"] is True
        assert entry["q_equals_tanh_q0_raw"] is True
        assert entry["endpoint_logit_equals_endpoint0"] is True
        assert entry["n_frames"] > 0


def test_bootstrap_inputs_are_paired_by_curve(report: dict) -> None:
    inputs = report["bootstrap_inputs"]
    assert inputs["draws"] == 10000                 # 附录 B :1608
    assert inputs["seed_root"] == 20261102          # 附录 B :1609
    for metric, block in inputs["metrics"].items():
        per_seed = block["per_seed_curve_values"]
        assert sorted(per_seed) == sorted(str(s) for s in report["seeds"])
        if metric == "move.fixed_coverage.direction_at_3":
            for seed, values in per_seed.items():
                assert values, f"no curve values for {metric} seed {seed}"
                assert all(0.0 <= v <= 1.0 for v in values.values())
    # the candidate will be bootstrapped against THESE tables
    direction = inputs["metrics"]["move.fixed_coverage.direction_at_3"]
    assert direction["higher_is_better"] is True
    rmse = inputs["metrics"]["move.rmse_zero"]
    assert rmse["higher_is_better"] is False


def test_curve_structure_keeps_the_statistics_defined(report: dict) -> None:
    """The §9.4 rules must be SATISFIABLE by the selected frame set.

    A frame set with one frame per curve would leave ``q.corr`` undefined
    (§9.4 :1031) and ordering without pairs (§9.1 :892); this test fails if the
    selection ever regresses to that.
    """
    for seed in report["seeds"]:
        document = report["per_seed"][str(seed)]
        positions = report["positions"][str(seed)]
        rule = positions["rule"]
        assert rule["n_curves_used"] >= 2
        assert rule["frames_per_curve_min"] >= 1
        assert rule["frames_per_curve_max"] <= rule[
            "frames_per_curve_requested"]
        # every selected inner-val curve has hundreds of frames, so the budget
        # is met exactly
        assert rule["n_positions_used"] == (
            rule["n_curves_used"] * rule["frames_per_curve_requested"])
        q = document["pooled"]["process"]["q"]
        assert q["n_eval_curves"] >= 2
        assert q["median_points_per_curve"] >= 1
        # with >=3 q points per curve the correlation is DEFINED and the
        # validity flag is not forced on
        if q["frac_curves_ge3"] and q["frac_curves_ge3"] > 0.0:
            assert q["corr_curve_median"] is not None
        # ordering pairs exist inside (curve, direction) groups
        ordering = document["pooled"]["process"]["ordering"]
        assert ordering["n_pairs_full"] > 0
        assert 0.0 <= ordering["pairwise_accuracy_full"] <= 1.0


def test_bootstrap_inputs_are_consumable_by_the_confirmatory_mode(
        report: dict) -> None:
    """The stats lane must be able to run the multi-seed mode from this report.

    Drives ``paired_curve_bootstrap_multiseed`` end-to-end on the real per-seed
    curve tables of the baseline (candidate == baseline here, so the paired
    improvement is identically zero and the p value is 1) — this proves the
    table plumbing, the seed order and the mode, not a scientific result.
    """
    from pes2ts_core.pesn.offline.evaluation.v543_eval import (
        paired_curve_bootstrap_multiseed,
    )

    inputs = report["bootstrap_inputs"]
    assert inputs["seed_order"] == report["seeds"]
    assert sorted(inputs["modes"]) == ["per_seed_independent",
                                      "shared_intersection"]
    assert inputs["modes"]["per_seed_independent"]["status"].startswith(
        "CONFIRMATORY")
    metric = "move.fixed_coverage.direction_at_3"
    block = inputs["metrics"][metric]
    tables = [block["per_seed_curve_values"][str(seed)]
              for seed in inputs["seed_order"]]
    assert all(tables), "a seed carried no curve values"
    assert block["n_curves_per_seed"] == {
        str(seed): len(table) for seed, table in zip(inputs["seed_order"],
                                                     tables)}
    # the collapsed intersection that motivated the fix is visible as data
    assert block["n_curves_shared_across_all_seeds"] <= min(
        block["n_curves_per_seed"].values())
    result = paired_curve_bootstrap_multiseed(
        tables, tables, higher_is_better=block["higher_is_better"],
        metric=metric, draws=200, seed_root=inputs["seed_root"])
    assert result["mode"] == "per_seed_independent"
    assert result["n_curves_per_seed"] == [
        len(table) for table in tables]
    assert result["n_curves_total"] == sum(len(t) for t in tables)
    # candidate == baseline → every paired improvement is exactly 0
    assert result["point_estimate"] == pytest.approx(0.0)
    assert result["ci95"] == pytest.approx([0.0, 0.0])
    assert result["p_one_sided"] == pytest.approx(1.0)


def test_revision_block_records_the_bootstrap_fix(report: dict) -> None:
    revision = report["revision"]
    assert revision["evaluator_code_sha256"] == \
        report["evaluator"]["code_sha256"]
    assert revision["evaluator_spec_sha256"] == \
        report["evaluator"]["spec_sha256"]
    assert "paired_curve_bootstrap_multiseed" in revision["bump_reason"]
    rationale = revision["multiseed_rationale"]
    for key in ("problem", "plan_reading", "fix", "compatibility"):
        assert rationale[key]
    assert "37" in rationale["problem"]


def test_all_measured_numbers_are_finite(report: dict) -> None:
    bad: list[str] = []
    for seed in report["seeds"]:
        for path, value in _walk_numbers(report["per_seed"][str(seed)]):
            if not math.isfinite(value):
                bad.append(f"seed {seed}: {path} = {value}")
    assert not bad, bad[:10]


def test_report_can_be_persisted(report: dict) -> None:
    out = os.environ.get("V543_EVAL_OUT")
    if not out:
        pytest.skip("V543_EVAL_OUT not set; nothing to persist")
    target = Path(out)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2, sort_keys=True,
                                 ensure_ascii=False), encoding="utf-8")
    assert target.is_file()
    loaded = json.loads(target.read_text(encoding="utf-8"))
    assert loaded["schema"] == report["schema"]
    assert loaded["evaluator"]["code_sha256"] == \
        report["evaluator"]["code_sha256"]
