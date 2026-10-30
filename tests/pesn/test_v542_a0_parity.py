"""V5.4.2 WP1 lane A — A0 parity certificate on the SIX pinned checkpoints
(ADR-0033 §2.2; config ``backbone_contract.a0_migration.parity.assets``).

Per pinned asset (3 V541 E0 S17/S29/S43 ``runs/`` 37,900-update finals +
3 V53MP C S17/S29/S43 ``outputs/pesn/v53mp/`` S3 formal 15,400-update):

* path-exists + SHA-256 == pinned value (mismatch/unpinned ⇒ hard fail,
  never a silent skip);
* strict-audit load of ``V542Model`` (A0) from ``payload["model"]``;
* >= 8 REAL inner-val batches from the frozen 50K data view (the
  existing ``ColumnarKeyAtomDatasetV3`` + ``v53mp_pool_split`` +
  ``_batch_from_positions`` loader — no new feature pipeline);
* CPU bitwise (``torch.equal``) equality of all five outputs AND
  ``forward_move`` {u, z} against the legacy reference
  (``StarEGNNV3`` + ``ReactionProcessHeadV53MP`` direct forward on the
  same batch); ``q is order_score`` re-asserted;
* GPU tolerance leg (skipped cleanly without CUDA): hard caps
  atol=1e-6/rtol=1e-5, TF32 off; run-to-run floor recorded as
  diagnostics (floor>cap ⇒ inconclusive, CPU bitwise stays decisive);
* EVALUATOR parity leg (config ``parity.evaluator_parity =
  recompute_H2_H3_vs_V541_archive_bitwise_or_le_1e-9``): the frozen
  ``evaluate_v53mp`` runs over the same inner-val positions on BOTH the
  V542 wrapper and the legacy reference; every metric leaf must be
  bitwise-equal or within 1e-9 absolute (floats; ints/bools/strings/
  None exact) — a mismatch hard-fails with key + values + max abs delta.

Environment skips (pytest): a checkpoint absent on disk OR the frozen
data view unavailable ⇒ ``skip`` with an explicit reason — this is the
documented per-test environment behavior; a checkpoint that IS present
but fails parity (or whose SHA drifted) fails HARD (no tolerance
loosening, no adapter "fixing").

Certificate scope (module-level, same as the CLI): forward + evaluator
migration ONLY — NOT evidence of E0 trainability reuse."""

from __future__ import annotations

import copy
import importlib.util
from typing import Any, Dict, List, Mapping, Tuple

import pytest

torch = pytest.importorskip("torch")

from pathlib import Path  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]

from scripts.agent_ops.v542_a0_parity import (  # noqa: E402
    DEFAULT_CONFIG,
    EVALUATOR_ATOL,
    EVALUATOR_TOP_KEYS,
    GPU_ATOL,
    GPU_RTOL,
    MEASURED_ONLY_COVERAGE_CATEGORIES,
    MIN_INNER_VAL_BATCHES,
    REPORT_SCHEMA,
    REQUIRED_COVERAGE_CATEGORIES,
    V542ParityError,
    _asset_load_audit_record,
    _evaluator_noop_problems,
    compare_evaluator_metrics,
    load_pinned_assets,
    run_cpu_parity,
    run_parity,
    sha256_file,
    verify_assets,
)

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("yaml") is None,
    reason="PyYAML required to read the frozen campaign config")

_CONFIG_PATH = PROJECT_ROOT / DEFAULT_CONFIG

#: The six pinned asset ids (frozen; the parametrization source).
_ASSET_IDS = ("V541_E0_S17", "V541_E0_S29", "V541_E0_S43",
              "V53MP_C_S17", "V53MP_C_S29", "V53MP_C_S43")


@pytest.fixture(scope="module")
def pinned() -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """The frozen parity block from the campaign config (fail-closed)."""
    try:
        return load_pinned_assets(_CONFIG_PATH)
    except V542ParityError as exc:  # pragma: no cover - config is frozen
        pytest.skip(f"frozen config unavailable: {exc}")


@pytest.fixture(scope="module")
def assets_by_id(pinned) -> Dict[str, Dict[str, Any]]:
    _cfg, assets = pinned
    return {str(a["id"]): dict(a) for a in assets}


def _require_asset(asset: Mapping[str, Any]) -> Path:
    """Resolve one pinned asset; SKIP only when the file is absent."""
    path = PROJECT_ROOT / str(asset["path"])
    if not path.is_file():
        pytest.skip(f"pinned checkpoint absent in this environment: {path}")
    actual = sha256_file(path)
    if actual != str(asset["sha256"]):
        # A present checkpoint with drifted bytes is a HARD failure —
        # never a skip (fix-5 C3: smoke/non-pinned/sha-drift forbidden).
        pytest.fail(
            f"pinned SHA mismatch for {asset['id']}: {actual} != "
            f"{asset['sha256']} ({path})")
    return path


def _require_data_view() -> None:
    tables = Path("/pes2ts_data/v53mp_50k_data")
    if not (tables / "data_manifest.json").is_file() or \
            not (tables / "split_manifest_v53mp.json").is_file():
        pytest.skip("frozen 50K data view unavailable in this environment "
                    f"({tables})")


def _load_pair(path: Path):
    """Strict-load BOTH models from one checkpoint (hard-fails on audit)."""
    from pes2ts_core.pesn.models.key_atom_heads_v3 import KeyAtomV53MPModel
    from pes2ts_core.pesn.models.v542 import V542Model, load_a0_strict

    payload = torch.load(str(path), map_location="cpu", weights_only=False)
    state = payload["model"] if "model" in payload else payload
    assert isinstance(state, Mapping)
    legacy = KeyAtomV53MPModel()
    legacy.load_state_dict(state, strict=True)
    v542 = V542Model()
    load_a0_strict(v542, {"model": state})
    assert v542.backbone.w_v is v542.move_decoder.w_v
    return v542, legacy


# ---------------------------------------------------------------------------
# config-level certificate (no checkpoint/data access)
# ---------------------------------------------------------------------------
def test_config_lists_exactly_six_required_pinned_assets(pinned) -> None:
    _cfg, assets = pinned
    assert len(assets) == 6
    assert all(a["required"] is True for a in assets)
    ids = sorted(a["id"] for a in assets)
    assert ids == sorted(_ASSET_IDS)
    tiers = {a["id"]: (a["source_tier"], int(a["training_updates"]))
             for a in assets}
    for sid in ("V541_E0_S17", "V541_E0_S29", "V541_E0_S43"):
        assert tiers[sid] == ("v541_e0_runs_final_37900", 37900)
        assert "/runs/" in str(next(a for a in assets
                                    if a["id"] == sid)["path"])
    for sid in ("V53MP_C_S17", "V53MP_C_S29", "V53MP_C_S43"):
        assert tiers[sid] == ("v53mp_20k_s3_formal_15400_auxiliary", 15400)
        path = str(next(a for a in assets if a["id"] == sid)["path"])
        assert path.startswith("outputs/pesn/v53mp/")
        assert "/smoke" not in path and "/runs/" not in path
    for a in assets:
        assert len(str(a["sha256"])) == 64
        assert "smoke" not in str(a["path"]).split("/")


def test_verify_assets_refuses_drifted_bytes(pinned, tmp_path) -> None:
    _cfg, assets = pinned
    bad = [dict(a) for a in assets]
    doctor = tmp_path / "checkpoint_best.pt"
    doctor.write_bytes(b"not-a-real-checkpoint")
    bad[0]["path"] = str(doctor)
    with pytest.raises(V542ParityError) as exc:
        verify_assets(bad, root=PROJECT_ROOT)
    assert exc.value.code == "V542_PARITY_SHA_MISMATCH"


def test_verify_assets_refuses_missing_file(pinned, tmp_path) -> None:
    _cfg, assets = pinned
    bad = [dict(a) for a in assets]
    bad[0]["path"] = str(tmp_path / "absent.pt")
    with pytest.raises(V542ParityError) as exc:
        verify_assets(bad, root=PROJECT_ROOT)
    assert exc.value.code == "V542_PARITY_ASSET_MISSING"


# ---------------------------------------------------------------------------
# per-asset certificate (real checkpoints + real inner-val batches)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("asset_id", _ASSET_IDS)
def test_asset_cpu_bitwise_parity(assets_by_id, asset_id: str) -> None:
    """STRICT per-asset certificate: present ⇒ fail-hard on any drift."""
    asset = assets_by_id[asset_id]
    path = _require_asset(asset)          # skip only if absent; SHA hard
    _require_data_view()                  # skip only if data view absent
    from scripts.agent_ops.v542_a0_parity import inner_val_batches

    cfg, _assets = load_pinned_assets(_CONFIG_PATH)
    v542, legacy = _load_pair(path)       # strict-audit load (hard)
    seed = int(str(asset_id).rsplit("_S", 1)[1])
    batches = inner_val_batches(cfg, seed)
    assert len(batches) >= MIN_INNER_VAL_BATCHES

    cpu = run_cpu_parity(v542, legacy, batches)
    assert cpu["n_batches"] >= MIN_INNER_VAL_BATCHES
    if not cpu["all_bitwise"]:
        bad = {k: v for k, v in cpu["per_key"].items() if not v["bitwise"]}
        pytest.fail(f"{asset_id} CPU parity BREAK on keys {bad}")
    assert cpu["all_bitwise"] is True
    assert cpu["forward_move_bitwise"] is True
    assert cpu["q_is_order_score"] is True
    for key, row in cpu["per_key"].items():
        assert row["max_abs_diff"] == 0.0, (asset_id, key)


@pytest.mark.parametrize("asset_id", _ASSET_IDS)
def test_asset_gpu_tolerance_caps(assets_by_id, asset_id: str) -> None:
    """GPU leg: hard caps 1e-6/1e-5, TF32 off, floor diagnostic only."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable — GPU tolerance leg skipped "
                    "(CPU bitwise remains decisive)")
    asset = assets_by_id[asset_id]
    path = _require_asset(asset)
    _require_data_view()
    from scripts.agent_ops.v542_a0_parity import inner_val_batches, run_gpu_tolerance

    cfg, _assets = load_pinned_assets(_CONFIG_PATH)
    v542, legacy = _load_pair(path)
    seed = int(str(asset_id).rsplit("_S", 1)[1])
    batches = inner_val_batches(cfg, seed)

    gpu = run_gpu_tolerance(v542, legacy, batches)
    assert gpu["atol"] == GPU_ATOL and gpu["rtol"] == GPU_RTOL
    assert gpu["tf32"] is False
    assert gpu["status"] in ("pass", "inconclusive"), \
        f"{asset_id} GPU parity FAIL (caps never loosened): {gpu}"
    if gpu["status"] == "inconclusive":
        # floor > cap: recorded diagnostics, CPU bitwise stays decisive
        assert gpu["run_to_run_floor_all_within_cap"] is False


# ---------------------------------------------------------------------------
# evaluator parity leg (config parity.evaluator_parity =
# recompute_H2_H3_vs_V541_archive_bitwise_or_le_1e-9)
# ---------------------------------------------------------------------------
def test_evaluator_metric_comparator_bitwise_and_within_tol_pass() -> None:
    """Identical metric dicts pass; a <=1e-9 float drift also passes."""
    base = {
        "schema": "v53mp-eval",
        "move": {
            "key_rmse_over_zero": {"median_rmse_over_zero": 0.91,
                                    "n_curves": 12},
            "sre_and_sign": {"sign_accuracy_cluster_median": 0.93,
                              "dead_zone_fraction": 0.01},
        },
        "q_head": {"corr_curve_median": 0.397, "n_eval_curves": 10},
        "endpoint_head": {"auprc": 0.2907, "tau_end": None},
        "run_outcome": "ok",
        "n_dev_samples": 64,
    }
    same = compare_evaluator_metrics(base, copy.deepcopy(base))
    assert same["all_equal"] is True
    assert same["mismatches"] == []
    assert same["max_abs_diff"] == 0.0
    assert same["n_float_leaves"] > 0 and same["n_exact_leaves"] > 0
    assert same["tolerance"] == EVALUATOR_ATOL

    near = copy.deepcopy(base)
    near["endpoint_head"]["auprc"] = base["endpoint_head"]["auprc"] + 5e-10
    within = compare_evaluator_metrics(base, near)
    assert within["all_equal"] is True
    assert 0.0 < within["max_abs_diff"] <= EVALUATOR_ATOL


def test_evaluator_metric_comparator_rejects_perturbed_metric() -> None:
    """A deliberately perturbed metric dict is REJECTED (hard fail)."""
    base = {
        "q_head": {"corr_curve_median": 0.397, "mae_curve_median": 0.21},
        "endpoint_head": {"auprc": 0.2907},
    }
    perturbed = copy.deepcopy(base)
    perturbed["q_head"]["corr_curve_median"] = 0.397 + 1e-6

    report = compare_evaluator_metrics(base, perturbed)
    assert report["all_equal"] is False
    assert len(report["mismatches"]) == 1
    miss = report["mismatches"][0]
    assert miss["key"] == "q_head.corr_curve_median"
    assert miss["kind"] == "float"
    assert abs(miss["abs_diff"] - 1e-6) < 1e-15
    assert miss["reference"] == 0.397
    assert miss["candidate"] == 0.397 + 1e-6
    assert miss["abs_diff"] > EVALUATOR_ATOL
    assert report["max_abs_diff"] == miss["abs_diff"]


def test_evaluator_metric_comparator_rejects_structural_drift() -> None:
    """Missing keys, changed counts and type drift are all rejected."""
    base = {"move": {"n_val_samples": 64, "run_flag": True},
            "ordering": {"n_pairs_full": 120}}
    # 1) a float-exact-equal count change is still a hard fail (ints exact)
    count_drift = copy.deepcopy(base)
    count_drift["ordering"]["n_pairs_full"] = 121
    r1 = compare_evaluator_metrics(base, count_drift)
    assert r1["all_equal"] is False
    assert r1["mismatches"][0]["key"] == "ordering.n_pairs_full"
    assert r1["mismatches"][0]["kind"] == "exact"
    # 2) a key present on one side only
    dropped = copy.deepcopy(base)
    del dropped["move"]["run_flag"]
    r2 = compare_evaluator_metrics(base, dropped)
    assert r2["all_equal"] is False
    assert r2["mismatches"][0]["kind"] == "missing_in_candidate"
    # 3) None vs float
    nulled = copy.deepcopy(base)
    nulled["move"]["n_val_samples"] = None
    r3 = compare_evaluator_metrics(base, nulled)
    assert r3["all_equal"] is False


def test_evaluator_top_keys_documented_surface() -> None:
    """The compared surface is the full frozen-evaluator output."""
    assert "move" in EVALUATOR_TOP_KEYS
    assert "q_head" in EVALUATOR_TOP_KEYS
    assert "endpoint_head" in EVALUATOR_TOP_KEYS
    assert EVALUATOR_ATOL == 1e-9


@pytest.mark.parametrize("asset_id", _ASSET_IDS)
def test_asset_evaluator_parity(assets_by_id, asset_id: str) -> None:
    """STRICT per-asset evaluator parity: present ⇒ fail-hard on drift.

    Runs the frozen ``evaluate_v53mp`` on the V542 wrapper and the legacy
    reference over the same inner-val positions; every metric leaf must
    be bitwise-equal or within 1e-9 absolute.
    """
    asset = assets_by_id[asset_id]
    path = _require_asset(asset)
    _require_data_view()
    from scripts.agent_ops.v542_a0_parity import (
        evaluator_context,
        run_evaluator_parity,
    )

    cfg, _assets = load_pinned_assets(_CONFIG_PATH)
    v542, legacy = _load_pair(path)
    seed = int(str(asset_id).rsplit("_S", 1)[1])

    ev = run_evaluator_parity(v542, legacy, cfg, seed)
    assert ev["tolerance"] == EVALUATOR_ATOL
    assert set(ev["compared_top_keys"]) == set(EVALUATOR_TOP_KEYS)
    assert ev["n_positions"] >= MIN_INNER_VAL_BATCHES * 8
    ctx = evaluator_context(cfg, seed)
    assert ev["n_positions"] == len(ctx["positions"])
    if ev["status"] != "pass":
        pytest.fail(f"{asset_id} evaluator parity BREAK: {ev['error']}")
    assert ev["status"] == "pass"
    cmp = ev["comparison"]
    assert cmp["all_equal"] is True
    assert cmp["mismatches"] == []
    assert cmp["max_abs_diff"] <= EVALUATOR_ATOL


def test_evaluator_parity_typed_unavailable_name() -> None:
    """The unavailability code is typed and names its dependency."""
    # Drive evaluator_context against a config whose split manifest is
    # absent => the typed failure must be EVALUATOR_PARITY_UNAVAILABLE
    # (or the pre-existing data-view refusal) — never a silent skip.
    bad_cfg = {"campaign_id": "parity-typed-unavailable-probe",
               "data_contract": {"tables_dir": "/nonexistent/pes2ts_tables"},
               "split_contract": {
                   "split_manifest": "/nonexistent/split_manifest.json"}}
    with pytest.raises(V542ParityError) as exc:
        from scripts.agent_ops.v542_a0_parity import evaluator_context
        evaluator_context(bad_cfg, 17)
    assert exc.value.code in ("EVALUATOR_PARITY_UNAVAILABLE",
                              "V542_PARITY_DATA")
    assert exc.value.code != ""


# ---------------------------------------------------------------------------
# gate-1 A7: the evaluator leg can never pass as a no-op
# ---------------------------------------------------------------------------
def test_evaluator_noop_guard_flags_empty_comparison() -> None:
    """Two empty dicts (or missing top keys / zero float leaves) FAIL."""
    problems = _evaluator_noop_problems({}, {}, 0)
    assert problems, "empty evaluator outputs must be flagged as no-op"
    joined = " ".join(problems)
    assert "n_float_leaves == 0" in joined
    assert "reference" in joined and "candidate" in joined
    # a mapping missing documented top-level keys is flagged even with
    # float leaves present
    partial = {"q_head": {"corr_curve_median": 0.4}}
    problems = _evaluator_noop_problems(partial, partial, 1)
    assert any("top-level" in p for p in problems)
    # non-mapping output is flagged
    problems = _evaluator_noop_problems(None, None, 1)
    assert any("not a mapping" in p for p in problems)
    # the REAL frozen-evaluator surface passes the guard
    full_ref = {k: {} for k in EVALUATOR_TOP_KEYS}
    assert _evaluator_noop_problems(full_ref, full_ref, 63) == []


# ---------------------------------------------------------------------------
# gate-1 C1: the report carries the strict-load audit + template mapping
# ---------------------------------------------------------------------------
def test_load_audit_record_covers_template_record_fields(tmp_path) -> None:
    """Every field the template's load_audit.record requires exists."""
    import json

    from pes2ts_core.pesn.models.key_atom_heads_v3 import KeyAtomV53MPModel
    from pes2ts_core.pesn.models.v542 import V542Model, load_a0_strict

    template = json.loads(
        (PROJECT_ROOT / "config/campaigns/v542_templates/"
                        "parity_certificate.template.json"
         ).read_text(encoding="utf-8"))
    record_fields = template["load_audit"]["record"]
    assert set(record_fields) == {
        "source_sha256", "mapping_table_sha256", "target_key_set",
        "audit_json_path"}

    # real strict-load audit from a real (fixture-seeded) model pair
    torch.manual_seed(17)
    legacy = KeyAtomV53MPModel()
    v542 = V542Model()
    audit = load_a0_strict(v542, {"model": legacy.state_dict()})

    doctor = tmp_path / "synthetic_checkpoint.pt"
    doctor.write_bytes(b"synthetic-bytes-for-digest-only")
    fake_asset = {"id": "SYNTHETIC", "path": "synthetic/checkpoint.pt"}
    record = _asset_load_audit_record(fake_asset, doctor, audit)

    # C1: the full loader audit surface is persisted
    assert record["ok"] is True and record["strict"] is True
    assert record["source_path"] == "synthetic/checkpoint.pt"
    assert record["source_sha256"] == sha256_file(doctor)
    # gate-0 A3: the mapping digest is now canonical (64-hex, no placeholder)
    digest = str(record["mapping_table_sha256"])
    assert len(digest) == 64 and not digest.startswith("LOCAL_PLACEHOLDER:")
    assert record["n_source_keys"] == 43
    assert record["n_mapped_targets"] == 45
    assert set(record["dual_write_aliases"]) == {
        "backbone.w_v.weight", "backbone.w_v.bias"}
    assert len(record["target_key_set"]) == 45
    assert record["torch_equal_verified"] == 45
    assert record["missing"] == [] and record["unexpected"] == []
    assert record["shape_mismatch"] == [] and record["dtype_mismatch"] == []
    # every template load_audit.record field is present in the report
    # record (audit_json_path carries an explicit mapping note)
    for field in record_fields:
        assert field in record, field
    assert record["audit_json_path"] is None
    assert "embedded" in record["audit_json_path_mapping"]


def test_report_schema_relationship_to_certificate_template(monkeypatch,
                                                            pinned) -> None:
    """The report names its exact relationship + field mapping to the
    certificate template pes2ts_v542_parity_certificate_v1 (C1: no
    silent divergence)."""
    import json

    import scripts.agent_ops.v542_a0_parity as parity_mod

    cfg, assets = pinned

    def _fake_run_asset(asset, path, cfg, *, skip_gpu=False):
        return {"id": str(asset["id"]), "passed": True, "error": None,
                "source_tier": str(asset["source_tier"]),
                "training_updates": int(asset["training_updates"]),
                "gpu": {"status": "skipped_by_flag" if skip_gpu else "pass"},
                "cpu": {"all_bitwise": True}, "evaluator": {"status": "pass"},
                "load_audit": {"ok": True}}

    monkeypatch.setattr(parity_mod, "run_asset", _fake_run_asset)
    paths = {str(a["id"]): PROJECT_ROOT / str(a["path"]) for a in assets}
    report = run_parity(cfg, assets, paths)
    note = report["report_schema_note"]
    assert note["report_schema"] == REPORT_SCHEMA
    assert note["certificate_template_schema"] == \
        "pes2ts_v542_parity_certificate_v1"
    template = json.loads(
        (PROJECT_ROOT / "config/campaigns/v542_templates/"
                        "parity_certificate.template.json"
         ).read_text(encoding="utf-8"))
    mapping = note["template_field_mapping"]
    for field in template["load_audit"]["record"]:
        assert field in mapping, field
    assert "superset" in note["relationship"]
    # the template itself carries the mirrored note
    assert "pes2ts_v542_a0_parity_report_v1" in \
        template["load_audit"]["report_schema_note"]


# ---------------------------------------------------------------------------
# gate-1 C2: partial runs are DIAGNOSTIC, never a certificate
# ---------------------------------------------------------------------------
def _fake_passing_entry(asset, *, gpu_status):
    return {"id": str(asset["id"]), "passed": True, "error": None,
            "source_tier": str(asset["source_tier"]),
            "training_updates": int(asset["training_updates"]),
            "gpu": {"status": gpu_status},
            "cpu": {"all_bitwise": True}, "evaluator": {"status": "pass"},
            "load_audit": {"ok": True}}


def test_one_asset_run_is_not_certificate(monkeypatch, pinned) -> None:
    import scripts.agent_ops.v542_a0_parity as parity_mod

    cfg, assets = pinned
    monkeypatch.setattr(
        parity_mod, "run_asset",
        lambda asset, path, cfg, *, skip_gpu=False: _fake_passing_entry(
            asset, gpu_status="skipped_by_flag" if skip_gpu else "pass"))
    paths = {str(a["id"]): PROJECT_ROOT / str(a["path"]) for a in assets}
    only = [str(assets[0]["id"])]
    report = run_parity(cfg, assets, paths, only=only)
    assert report["certificate_complete"] is False
    assert report["only"] == only
    assert report["skip_gpu"] is False
    assert len(report["entries"]) == 1
    # even though the selected asset passed, the CERTIFICATE must not
    assert report["passed"] is False


def test_skip_gpu_run_is_not_certificate(monkeypatch, pinned) -> None:
    import scripts.agent_ops.v542_a0_parity as parity_mod

    cfg, assets = pinned
    monkeypatch.setattr(
        parity_mod, "run_asset",
        lambda asset, path, cfg, *, skip_gpu=False: _fake_passing_entry(
            asset, gpu_status="skipped_by_flag" if skip_gpu else "pass"))
    paths = {str(a["id"]): PROJECT_ROOT / str(a["path"]) for a in assets}
    report = run_parity(cfg, assets, paths, skip_gpu=True)
    assert report["certificate_complete"] is False
    assert report["skip_gpu"] is True
    assert report["gpu_disposition"] == "skipped_by_flag"
    assert len(report["entries"]) == 6
    assert report["passed"] is False


def test_full_six_asset_run_is_certificate(monkeypatch, pinned) -> None:
    import scripts.agent_ops.v542_a0_parity as parity_mod

    cfg, assets = pinned
    paths = {str(a["id"]): PROJECT_ROOT / str(a["path"]) for a in assets}

    # all-pass + GPU pass => certificate PASS
    monkeypatch.setattr(
        parity_mod, "run_asset",
        lambda asset, path, cfg, *, skip_gpu=False: _fake_passing_entry(
            asset, gpu_status="pass"))
    report = run_parity(cfg, assets, paths)
    assert report["certificate_complete"] is True
    assert report["only"] is None and report["skip_gpu"] is False
    assert report["gpu_disposition"] == "pass"
    assert report["passed"] is True

    # 6/6 but GPU FAIL => complete certificate, not PASS
    def _gpu_fail(asset, path, cfg, *, skip_gpu=False):
        e = _fake_passing_entry(asset, gpu_status="fail")
        e["passed"] = False
        return e

    monkeypatch.setattr(parity_mod, "run_asset", _gpu_fail)
    report = run_parity(cfg, assets, paths)
    assert report["certificate_complete"] is True
    assert report["gpu_disposition"] == "fail"
    assert report["passed"] is False

    # 6/6, GPU inconclusive (floor>cap) => certificate PASS stays allowed
    monkeypatch.setattr(
        parity_mod, "run_asset",
        lambda asset, path, cfg, *, skip_gpu=False: _fake_passing_entry(
            asset, gpu_status="inconclusive"))
    report = run_parity(cfg, assets, paths)
    assert report["certificate_complete"] is True
    assert report["gpu_disposition"] == "inconclusive"
    assert report["passed"] is True


# ---------------------------------------------------------------------------
# gate-1 C3: real-batch coverage is machine-verified per asset
# ---------------------------------------------------------------------------
def test_inner_val_selection_covers_required_categories() -> None:
    """Deterministic selection covers every required category (real
    frozen 50K data view; clean skip when the view is absent)."""
    _require_data_view()
    from scripts.agent_ops.v542_a0_parity import inner_val_selection

    cfg, _assets = load_pinned_assets(_CONFIG_PATH)
    view = inner_val_selection(cfg, 17)
    coverage = view["coverage"]
    assert coverage["required_categories"] == list(REQUIRED_COVERAGE_CATEGORIES)
    assert coverage["measured_only_categories"] == \
        list(MEASURED_ONLY_COVERAGE_CATEGORIES)
    for cat in REQUIRED_COVERAGE_CATEGORIES:
        assert coverage["present"][cat] is True, (cat, coverage["counts"])
        assert coverage["counts"][cat] >= 1
    assert coverage["missing_required"] == []
    assert coverage["n_positions"] >= MIN_INNER_VAL_BATCHES * 8
    assert len(view["batches"]) >= MIN_INNER_VAL_BATCHES
    # measured-only structural signals are RECORDED (may be zero — the
    # 50K pool molecules are compact; the count is the evidence)
    graph = coverage["graph_structure"]
    assert graph["isolated_atom_count"] >= 0
    assert graph["zero_in_degree_node_count"] >= 0
    assert graph["isolated_atom_present"] == (
        graph["isolated_atom_count"] > 0)
    assert graph["zero_in_degree_node_present"] == (
        graph["zero_in_degree_node_count"] > 0)
