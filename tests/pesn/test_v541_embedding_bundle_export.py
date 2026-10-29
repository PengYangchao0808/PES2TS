"""V5.4.1 WP7 — embedding champion bundle exporter (plan §12.1/§12.3).

Headline guarantees under test:

* ``export_embedding_bundle`` writes EXACTLY the eight-file whitelist
  ``EMBEDDING_BUNDLE_FILES`` and then re-audits whitelist + every SHA
  (``verify_embedding_bundle``);
* E0 exports the EMPTY-branch unified schema (no ``alpha``/``w_out``/
  ``embedding.*`` keys; the legacy backbone+head mapping is present) while
  E1–E5 exports carry the branch + gate + ``W`` AND the immutable full
  V541 checkpoint reference (path + sha256 never lost);
* ``feature_schema.json`` fixes node 48 / legacy edge 22 / chem+spatial+
  angle recipes / tie / mask / eps / cache fingerprint;
* ``provenance.json`` requires the §12.2 identity SHAs + §12.3 evidence
  blocks (three seeds, screen, dev exposure, UNREAD confirm, cost,
  failures/censoring, unclosed capacity diagnostics) and re-binds the
  champion checkpoint sha;
* ``downstream_handoff.json`` hands off frozen-E* as read-only
  (``requires_grad=false``) with a strict-load entry, and marks
  joint-finetune as a SEPARATE experiment;
* ``model_card.md`` states common-core warmstart, three seeds, dev
  exposure, unread confirm, cost, failures/censoring, unclosed capacity
  diagnostics and the explicit non-claims (no full original-paper model /
  no strict TS / no deployment qualification);
* typed refusals (:class:`EmbeddingBundleError`) for unknown/diagnostic
  ids, E0-with-branch, missing gate/W/branch, incomplete provenance,
  winner mismatch and any drift;
* the LEGACY exporters (B6 ``export_bundle`` / V5.3-MP
  ``export_move_process_bundle``) are untouched.

Plus the §13 evidence row 「下游可 strict-load」: a CLEAN-PROCESS script
loads the bundle, strict-loads a ``V541Model``, runs a small batch forward
and performs read-only checks — E0 outputs are bitwise-identical to the
V541 checkpoint model and to the legacy ``KeyAtomV53MPModel``.

All fixtures are SYNTHETIC (tmp_path); no real 50K data, no dev/confirm
access, no GPU.  The real export runs only after the WP7 dev evaluation,
executed by the main agent.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

import pytest

torch = pytest.importorskip("torch")

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from pes2ts_core.pesn.offline.publishing.export import (  # noqa: E402
    EMBEDDING_BUNDLE_FILES,
    EMBEDDING_BUNDLE_SCHEMA,
    EMBEDDING_FEATURE_SCHEMA_ID,
    EMBEDDING_LEGACY_EDGE_DIM,
    EMBEDDING_NODE_DIM,
    EMBEDDING_REQUIRED_PROVENANCE,
    MOVE_PROCESS_BUNDLE_FILES,
    EmbeddingBundleError,
    build_downstream_handoff,
    build_embedding_feature_schema,
    export_bundle,
    export_embedding_bundle,
    export_move_process_bundle,
    verify_embedding_bundle,
)
from pes2ts_core.pesn.models.v541.model import V541Model  # noqa: E402

SEED = 17
SEEDS = (17, 29, 43)


# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------


def _make_champion(run_dir: Path, embedding_id: str = "E0", *,
                   seed: int = SEED) -> Path:
    """Synthetic V541 champion checkpoint (trainer 'model' contract)."""
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    if embedding_id == "E0":
        model = V541Model()
    else:
        model = V541Model(
            embedding_id=embedding_id,
            embedding_init_seed=1_000_003 + int(seed))
    ckpt = run_dir / "checkpoint_best.pt"
    torch.save({"model": model.state_dict(), "update": 37_900}, ckpt)
    return ckpt


def _provenance(checkpoint_sha256: str | None = None) -> Dict[str, Any]:
    """A complete §12.2 + §12.3 provenance block (synthetic digests)."""
    doc: Dict[str, Any] = {
        "config_sha256": "a" * 64,
        "common_protocol_sha256": "b" * 64,
        "embedding_spec_sha256": "c" * 64,
        "feature_recipe_sha256": "d" * 64,
        "code_sha256": "e" * 64,
        "environment_sha256": "f" * 64,
        "git_commit": "0" * 40,
        "seeds": {
            17: {"status": "done", "outcome": "done"},
            29: {"status": "done", "outcome": "done"},
            43: {"status": "done", "outcome": "done"},
        },
        "screen": {"split": "inner-val", "top2": ["E0", "E3"]},
        "dev_exposure": {
            "wp7_candidate_evaluation": "one-shot frozen dev eval",
            "access_count": 1,
        },
        "confirm_read": False,
        "cost": {"gpu_hours": 12.5, "new_qc_calls": 0},
        "failures_and_censoring": {
            "E1": "scientific_failure (H2 MOVE guard)",
            "E4": "gate_skipped (cost ratio > 2.0)",
        },
        "capacity_diagnostics": {
            "T4": "gate_skipped(no_new_embedding_gain)",
            "unclosed": ["capacity/depth confounds not exercised"],
        },
    }
    if checkpoint_sha256 is not None:
        doc["checkpoint_sha256"] = checkpoint_sha256
    return doc


def _benchmark(embedding_id: str = "E0") -> Dict[str, Any]:
    """A complete WP7 benchmark summary (synthetic; winner matches)."""
    return {
        "winner_embedding_id": embedding_id,
        "candidates": {
            "E0": {"seeds": [17, 29, 43], "h2": "pass"},
            "E1": {"seeds": [17], "h2": "fail"},
            "E2": {"seeds": [17], "h2": "fail"},
            "E3": {"seeds": [17], "h2": "fail"},
            "E4": {"seeds": [], "gate": "cost_ratio_exceeds_2.0"},
            "E5": {"seeds": [], "gate": "parent_gate_skipped"},
        },
        "stages": {"T3": "gate_skipped", "T4": "gate_skipped",
                   "E5": "gate_skipped"},
        "h3": {"passed": False, "reason": "no new embedding winner"},
        "confirm_read": False,
    }


@pytest.fixture()
def champion_e0(tmp_path: Path) -> Path:
    """The champion RUN dir (containing checkpoint_best.pt)."""
    _make_champion(tmp_path / "runs" / "V541_T0R_E0_S17", "E0")
    return tmp_path / "runs" / "V541_T0R_E0_S17"


def _export(tmp_path: Path, champion_dir: Path, embedding_id: str,
            **overrides: Any) -> Path:
    args: Dict[str, Any] = {
        "run_dir": champion_dir,
        "output_dir": tmp_path / "bundle",
        "model_id": "pes2ts-v541-{}-s17".format(embedding_id.lower()),
        "embedding_id": embedding_id,
        "embedding_config": {
            "schema": "pes2ts_v541_campaign_v1",
            "registry_row": {"family": "legacy"},
            "residual": {"alpha0": 0.0, "w_out_dim": 48},
        },
        "feature_schema": build_embedding_feature_schema(embedding_id),
        "provenance": _provenance(),
        "benchmark_summary": _benchmark(embedding_id),
    }
    args.update(overrides)
    return export_embedding_bundle(**args)


# ---------------------------------------------------------------------------
# happy path: whitelist, weights payload, schema, provenance, handoff, card
# ---------------------------------------------------------------------------
def test_e0_export_writes_exactly_the_whitelist(
        tmp_path: Path, champion_e0: Path) -> None:
    bundle = _export(tmp_path, champion_e0, "E0")
    present = {p.name for p in bundle.iterdir() if p.is_file()}
    assert present == set(EMBEDDING_BUNDLE_FILES)
    assert present == {
        "embedding_config.yaml", "embedding_weights.pt", "feature_schema.json",
        "provenance.json", "checksums.json", "benchmark_summary.json",
        "model_card.md", "downstream_handoff.json",
    }
    checksums = verify_embedding_bundle(bundle)
    assert set(checksums) == set(EMBEDDING_BUNDLE_FILES) - {"checksums.json"}


def test_e0_weights_use_empty_branch_unified_schema(
        tmp_path: Path, champion_e0: Path) -> None:
    bundle = _export(tmp_path, champion_e0, "E0")
    w = torch.load(bundle / "embedding_weights.pt", map_location="cpu",
                   weights_only=False)
    assert w["schema"] == "pes2ts_embedding_weights_v1"
    assert w["embedding_id"] == "E0"
    assert w["has_branch"] is False
    assert w["branch_state"] == {}
    legacy = w["legacy_state"]
    assert legacy, "legacy backbone+head mapping must be present"
    assert all(k.startswith(("backbone.", "process_head.")) for k in legacy)
    # immutable full-V541 checkpoint reference is never lost (§12.3)
    ref = w["checkpoint_ref"]
    assert ref["path"] == str(champion_e0 / "checkpoint_best.pt")
    assert len(ref["sha256"]) == 64
    from pes2ts_core.utils.hashing import sha256_file

    assert ref["sha256"] == sha256_file(champion_e0 / "checkpoint_best.pt")
    # no optimizer/teacher state leaks into the weights payload
    src = torch.load(champion_e0 / "checkpoint_best.pt", map_location="cpu",
                     weights_only=False)
    assert set(legacy) == set(src["model"])


def test_e0_export_is_consistent_with_v541_and_legacy_state(
        tmp_path: Path, champion_e0: Path) -> None:
    """E0 weights == the V541 checkpoint state == the legacy state (§5.3)."""
    bundle = _export(tmp_path, champion_e0, "E0")
    w = torch.load(bundle / "embedding_weights.pt", map_location="cpu",
                   weights_only=False)
    src = torch.load(champion_e0 / "checkpoint_best.pt", map_location="cpu",
                     weights_only=False)["model"]
    for key, tensor in w["legacy_state"].items():
        assert torch.equal(tensor, src[key]), key


def test_branch_export_carries_gate_w_and_branch(
        tmp_path: Path) -> None:
    champ_dir = tmp_path / "runs" / "V541_E1_S17"
    _make_champion(champ_dir, "E1")
    bundle = _export(tmp_path, champ_dir, "E1")
    w = torch.load(bundle / "embedding_weights.pt", map_location="cpu",
                   weights_only=False)
    assert w["has_branch"] is True
    branch = w["branch_state"]
    assert "alpha" in branch and "w_out.weight" in branch
    assert any(k.startswith("embedding.") for k in branch)
    assert set(w["legacy_state"]) and all(
        k.startswith(("backbone.", "process_head."))
        for k in w["legacy_state"])
    assert len(w["checkpoint_ref"]["sha256"]) == 64
    # strict-load round trip: merged mapping loads strict=True
    model = V541Model(embedding_id="E1", embedding_init_seed=424242)
    model.load_state_dict({**w["legacy_state"], **w["branch_state"]},
                          strict=True)


def test_feature_schema_fixes_dims_recipes_tie_mask_eps_fingerprint(
        tmp_path: Path, champion_e0: Path) -> None:
    bundle = _export(tmp_path, champion_e0, "E0")
    doc = json.loads((bundle / "feature_schema.json").read_text("utf-8"))
    assert doc["schema_version"] == EMBEDDING_FEATURE_SCHEMA_ID
    assert doc["node_dim"] == EMBEDDING_NODE_DIM == 48
    assert doc["legacy_edge_dim"] == EMBEDDING_LEGACY_EDGE_DIM == 22
    assert doc["edge_channel_order"] == "legacy"
    assert doc["chem_recipe"]["dim"] == 8
    assert doc["spatial_recipe"]["rbf_n"] == 12
    assert doc["angle_recipe"]["max_neighbors_per_center"] == 12
    assert doc["tie"]["angle_tie_policy"] == "reject_boundary_tie_shell"
    assert doc["mask"]["node_vector_consumed"] is False
    assert doc["eps"]["cos_denom_eps"] > 0
    assert len(doc["cache_fingerprint"]) == 64
    # the fingerprint binds the WHOLE recipe body
    body = {k: v for k, v in doc.items() if k != "cache_fingerprint"}
    from pes2ts_core.pesn.offline.experiments.runs import sha256_text
    from pes2ts_core.utils.hashing import stable_json_dumps

    assert doc["cache_fingerprint"] == sha256_text(stable_json_dumps(body))


def test_provenance_carries_evidence_blocks_and_checkpoint_binding(
        tmp_path: Path, champion_e0: Path) -> None:
    bundle = _export(tmp_path, champion_e0, "E0")
    prov = json.loads((bundle / "provenance.json").read_text("utf-8"))
    for key in EMBEDDING_REQUIRED_PROVENANCE:
        assert key in prov, key
    assert prov["confirm_read"] is False
    assert set(map(int, prov["seeds"])) == {17, 29, 43}
    assert prov["screen"] and prov["dev_exposure"]
    assert prov["cost"] and prov["failures_and_censoring"]
    assert prov["capacity_diagnostics"]
    assert prov["export_schema"] == EMBEDDING_BUNDLE_SCHEMA
    assert len(prov["checkpoint_sha256"]) == 64
    assert prov["checkpoint_path"].endswith("checkpoint_best.pt")


def test_downstream_handoff_frozen_readonly_strict_load_and_separate_ft(
        tmp_path: Path, champion_e0: Path) -> None:
    bundle = _export(tmp_path, champion_e0, "E0")
    hand = json.loads((bundle / "downstream_handoff.json").read_text("utf-8"))
    frozen = hand["frozen"]
    assert frozen["mode"] == "frozen-E0"
    assert frozen["read_only"] is True
    assert frozen["requires_grad"] is False
    assert frozen["strict_load"] is True
    assert "load_state_dict" in frozen["strict_load_entry"]
    joint = hand["joint_finetune"]
    assert joint["requires_separate_experiment"] is True
    assert joint["not_implied_by_this_bundle"] is True
    assert len(hand["reference_checkpoint"]["sha256"]) == 64


def test_model_card_states_every_required_disclosure(
        tmp_path: Path, champion_e0: Path) -> None:
    bundle = _export(tmp_path, champion_e0, "E0")
    card = (bundle / "model_card.md").read_text("utf-8")
    for probe in (
        "winner embedding: E0",
        "common-core warmstart",
        "three-seed ledger",
        "confirm curves were NEVER read",
        "Resource cost",
        "Failures and censoring",
        "Capacity/depth diagnostics",
        "full original-paper GNN",
        "strict/verified TS",
        "NOT deployment qualification",
        "joint-finetune-E*",
        "requires_grad=false",
        "research_only: true",
    ):
        assert probe in card, probe
    # per-candidate ledger is rendered from evidence, incl. the H2 failures
    # of E1–E3 (seed17 only) and the E4 cost gate / T3-T4-E5 skips
    for eid in ("E0", "E1", "E2", "E3", "E4", "E5"):
        assert "- {}:".format(eid) in card, eid
    assert "scientific_failure (H2 MOVE guard)" in card
    assert "gate_skipped (cost ratio > 2.0)" in card
    assert "- T3: gate_skipped" in card
    assert "- T4: gate_skipped" in card
    assert "- E5: gate_skipped" in card


# ---------------------------------------------------------------------------
# typed refusals (fail-closed)
# ---------------------------------------------------------------------------
def test_unknown_embedding_id_is_refused(
        tmp_path: Path, champion_e0: Path) -> None:
    with pytest.raises(EmbeddingBundleError) as excinfo:
        _export(tmp_path, champion_e0, "E9")
    assert excinfo.value.code == "V541_EMBEDDING_UNKNOWN"


def test_diagnostic_e6_can_never_be_a_winner(
        tmp_path: Path, champion_e0: Path) -> None:
    with pytest.raises(EmbeddingBundleError) as excinfo:
        _export(tmp_path, champion_e0, "E6")
    assert excinfo.value.code == "V541_EMBEDDING_DIAGNOSTIC"


def test_e0_refuses_a_branch_checkpoint(
        tmp_path: Path) -> None:
    champ_dir = tmp_path / "runs" / "V541_E1_S17"
    _make_champion(champ_dir, "E1")
    with pytest.raises(EmbeddingBundleError) as excinfo:
        _export(tmp_path, champ_dir, "E0")
    assert excinfo.value.code == "V541_WEIGHTS_E0_BRANCH"


def test_candidate_refuses_state_without_gate(
        tmp_path: Path, champion_e0: Path) -> None:
    """An E0-shaped checkpoint exported as E1 lacks alpha/W/branch."""
    with pytest.raises(EmbeddingBundleError) as excinfo:
        _export(tmp_path, champion_e0, "E1")
    assert excinfo.value.code == "V541_WEIGHTS_GATE_MISSING"


def test_unexpected_state_key_is_refused(
        tmp_path: Path, champion_e0: Path) -> None:
    src = torch.load(champion_e0 / "checkpoint_best.pt", map_location="cpu",
                     weights_only=False)
    src["model"]["rogue.weight"] = torch.zeros(3)
    torch.save(src, champion_e0 / "checkpoint_best.pt")
    with pytest.raises(EmbeddingBundleError) as excinfo:
        _export(tmp_path, champion_e0, "E0")
    assert excinfo.value.code == "V541_WEIGHTS_UNEXPECTED_KEY"


@pytest.mark.parametrize("mutate,code", [
    (lambda p: p.pop("cost"), "V541_PROVENANCE_MISSING"),
    (lambda p: p.update({"confirm_read": True}), "V541_PROVENANCE_CONFIRM"),
    (lambda p: p.update({"seeds": {17: {}, 29: {}}}), "V541_PROVENANCE_SEEDS"),
    (lambda p: p.update({"screen": None}), "V541_PROVENANCE_NULL"),
    (lambda p: p.update({"checkpoint_sha256": "0" * 64}),
     "V541_PROVENANCE_CHECKPOINT_DRIFT"),
])
def test_provenance_refusals(
        tmp_path: Path, champion_e0: Path, mutate: Any, code: str) -> None:
    prov = _provenance()
    mutate(prov)
    with pytest.raises(EmbeddingBundleError) as excinfo:
        _export(tmp_path, champion_e0, "E0", provenance=prov)
    assert excinfo.value.code == code


def test_winner_mismatch_is_refused(
        tmp_path: Path, champion_e0: Path) -> None:
    with pytest.raises(EmbeddingBundleError) as excinfo:
        _export(tmp_path, champion_e0, "E0", benchmark_summary=_benchmark("E1"))
    assert excinfo.value.code == "V541_BENCHMARK_WINNER"


def test_missing_checkpoint_is_refused(
        tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "V541_T0R_E0_S17"
    run_dir.mkdir(parents=True)
    with pytest.raises(EmbeddingBundleError) as excinfo:
        _export(tmp_path, run_dir, "E0")
    assert excinfo.value.code == "V541_CHECKPOINT_MISSING"


def test_feature_schema_dimension_drift_is_refused(
        tmp_path: Path, champion_e0: Path) -> None:
    schema = build_embedding_feature_schema("E0")
    schema["node_dim"] = 32
    with pytest.raises(EmbeddingBundleError) as excinfo:
        _export(tmp_path, champion_e0, "E0", feature_schema=schema)
    assert excinfo.value.code == "V541_FEATURE_SCHEMA_DIMS"


def test_feature_schema_missing_block_is_refused(
        tmp_path: Path, champion_e0: Path) -> None:
    schema = build_embedding_feature_schema("E0")
    schema.pop("cache_fingerprint")
    with pytest.raises(EmbeddingBundleError) as excinfo:
        _export(tmp_path, champion_e0, "E0", feature_schema=schema)
    assert excinfo.value.code == "V541_FEATURE_SCHEMA_MISSING"


# ---------------------------------------------------------------------------
# post-hoc whitelist + SHA audit (verify_embedding_bundle)
# ---------------------------------------------------------------------------
def test_verify_detects_extra_file(
        tmp_path: Path, champion_e0: Path) -> None:
    bundle = _export(tmp_path, champion_e0, "E0")
    (bundle / "rogue_optimizer.pt").write_bytes(b"nope")
    with pytest.raises(EmbeddingBundleError) as excinfo:
        verify_embedding_bundle(bundle)
    assert excinfo.value.code == "V541_BUNDLE_WHITELIST_DRIFT"
    assert "rogue_optimizer.pt" in str(excinfo.value)


def test_verify_detects_missing_file(
        tmp_path: Path, champion_e0: Path) -> None:
    bundle = _export(tmp_path, champion_e0, "E0")
    (bundle / "model_card.md").unlink()
    with pytest.raises(EmbeddingBundleError) as excinfo:
        verify_embedding_bundle(bundle)
    assert excinfo.value.code == "V541_BUNDLE_WHITELIST_DRIFT"
    assert "model_card.md" in str(excinfo.value)


def test_verify_detects_tampered_bytes(
        tmp_path: Path, champion_e0: Path) -> None:
    bundle = _export(tmp_path, champion_e0, "E0")
    card = bundle / "model_card.md"
    card.write_text(card.read_text("utf-8") + "\ntampered\n", encoding="utf-8")
    with pytest.raises(EmbeddingBundleError) as excinfo:
        verify_embedding_bundle(bundle)
    assert excinfo.value.code == "V541_BUNDLE_CHECKSUM_MISMATCH"
    assert "model_card.md" in str(excinfo.value)


def test_verify_detects_checksum_coverage_drift(
        tmp_path: Path, champion_e0: Path) -> None:
    bundle = _export(tmp_path, champion_e0, "E0")
    path = bundle / "checksums.json"
    checksums = json.loads(path.read_text("utf-8"))
    checksums.pop("model_card.md")
    path.write_text(json.dumps(checksums), encoding="utf-8")
    with pytest.raises(EmbeddingBundleError) as excinfo:
        verify_embedding_bundle(bundle)
    assert excinfo.value.code == "V541_BUNDLE_CHECKSUMS_COVERAGE"


# ---------------------------------------------------------------------------
# legacy exporters are UNTOUCHED (旧 bundle 不变)
# ---------------------------------------------------------------------------
def test_legacy_whitelist_constants_unchanged() -> None:
    assert MOVE_PROCESS_BUNDLE_FILES == (
        "weights.pt", "tau_end.json", "metrics.json", "provenance.json",
        "model_card.md", "model_bundle.json", "checksums.json",
    )
    # the whitelists may share generic names (provenance/model_card/
    # checksums) but the EMBEDDING-specific payloads never collide with the
    # move-process payload names, and the two bundles live in separate dirs
    assert "embedding_weights.pt" not in MOVE_PROCESS_BUNDLE_FILES
    assert "weights.pt" not in EMBEDDING_BUNDLE_FILES


def test_legacy_move_process_export_still_behaves(
        tmp_path: Path) -> None:
    """The V5.3-MP exporter keeps its exact 7-file behaviour."""
    from pes2ts_core.pesn.models.key_atom_heads_v3 import KeyAtomV53MPModel

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    torch.manual_seed(SEED)
    torch.save({"model": KeyAtomV53MPModel().state_dict()},
               run_dir / "checkpoint_best.pt")
    out = export_move_process_bundle(
        run_dir=run_dir, output_dir=tmp_path / "mp_bundle",
        model_id="pes2ts-v53mp-C-20k-s17", tau_end=0.87,
        metrics={"schema": "pesn_v53mp_metrics_v1"},
        provenance={
            "config_sha256": "a" * 64, "data_manifest_sha256": "b" * 64,
            "teacher_manifest_sha256": "c" * 64, "split_sha256": "d" * 64,
            "label_spec_sha256": "e" * 64, "code_commit": "f" * 40,
        })
    present = {p.name for p in out.iterdir() if p.is_file()}
    assert present == set(MOVE_PROCESS_BUNDLE_FILES)


def test_b6_export_bundle_importable_and_distinct() -> None:
    assert export_bundle is not export_move_process_bundle
    assert export_bundle is not export_embedding_bundle


# ---------------------------------------------------------------------------
# §13 evidence row: clean-process strict-load + small-batch forward +
# read-only checks (E0 outputs == V541 == legacy, bitwise)
# ---------------------------------------------------------------------------
_CLEAN_PROCESS_SCRIPT = r"""
import json, sys, torch
from pathlib import Path

bundle_dir = Path(sys.argv[1])
ckpt_path = Path(sys.argv[2])
report_path = Path(sys.argv[3])

sys.path.insert(0, __REPO__)
from pes2ts_core.pesn.models.v541.model import V541Model
from pes2ts_core.pesn.models.key_atom_heads_v3 import KeyAtomV53MPModel
from pes2ts_core.pesn.contracts.key_atom_v3 import forward_view_v3
from scripts.agent_ops.feature_gate.v3_fixtures import make_v3_batch
from pes2ts_core.utils.hashing import sha256_file

w = torch.load(bundle_dir / "embedding_weights.pt", map_location="cpu",
               weights_only=False)
state = {**w["legacy_state"], **w["branch_state"]}
eid = w["embedding_id"]

# 1. strict-load into a freshly built V541Model (downstream_handoff recipe)
if eid == "E0":
    model = V541Model()
else:
    model = V541Model(embedding_id=eid, embedding_init_seed=0)
missing, unexpected = [], []
result = model.load_state_dict(state, strict=True)
missing = list(result.missing_keys)
unexpected = list(result.unexpected_keys)
model.eval()

# 2. frozen-E* read-only contract: requires_grad=False everywhere
for p in model.parameters():
    p.requires_grad_(False)
assert all(not p.requires_grad for p in model.parameters())

# 3. small-batch forward, all five outputs finite
batch = forward_view_v3(make_v3_batch(n_graphs=2))
with torch.no_grad():
    out = model(batch)
assert set(out) == {"u", "z", "q", "endpoint_logit", "order_score"}
finite = all(bool(torch.isfinite(v).all()) for v in out.values())

# 4. outputs == the V541 checkpoint model (bitwise); for E0 this also
#    equals the legacy KeyAtomV53MPModel (same state, same math, §5.3)
ref_state = torch.load(ckpt_path, map_location="cpu",
                       weights_only=False)["model"]
if eid == "E0":
    v541 = V541Model()
else:
    v541 = V541Model(embedding_id=eid, embedding_init_seed=0)
v541.load_state_dict(ref_state, strict=True)
v541.eval()
with torch.no_grad():
    ref_out = v541(batch)
v541_bitwise = all(torch.equal(out[k], ref_out[k]) for k in out)

legacy_bitwise = None
if eid == "E0":
    legacy = KeyAtomV53MPModel()
    legacy.load_state_dict(ref_state, strict=True)
    legacy.eval()
    with torch.no_grad():
        legacy_out = legacy(batch)
    legacy_bitwise = all(torch.equal(out[k], legacy_out[k]) for k in out)

# 5. read-only: loading never mutates the bundle bytes
after = sha256_file(bundle_dir / "embedding_weights.pt")

json.dump({
    "embedding_id": eid,
    "missing": missing,
    "unexpected": unexpected,
    "finite": finite,
    "requires_grad_any": any(p.requires_grad for p in model.parameters()),
    "v541_bitwise": v541_bitwise,
    "legacy_bitwise": legacy_bitwise,
    "weights_sha_after": after,
}, open(report_path, "w"))
"""


def _run_clean_process(bundle: Path, checkpoint: Path,
                       tmp_path: Path) -> Dict[str, Any]:
    script = tmp_path / "strict_load_check.py"
    script.write_text(
        _CLEAN_PROCESS_SCRIPT.replace("__REPO__", repr(str(REPO))),
        encoding="utf-8")
    report = tmp_path / "report.json"
    proc = subprocess.run(
        [sys.executable, str(script), str(bundle), str(checkpoint),
         str(report)],
        capture_output=True, text=True, cwd=str(REPO), timeout=600)
    assert proc.returncode == 0, proc.stderr
    return json.loads(report.read_text("utf-8"))


def test_clean_process_strict_load_forward_and_readonly_e0(
        tmp_path: Path, champion_e0: Path) -> None:
    from pes2ts_core.utils.hashing import sha256_file

    bundle = _export(tmp_path, champion_e0, "E0")
    before = sha256_file(bundle / "embedding_weights.pt")
    report = _run_clean_process(bundle, champion_e0 / "checkpoint_best.pt",
                                tmp_path)
    assert report["embedding_id"] == "E0"
    assert report["missing"] == [] and report["unexpected"] == []
    assert report["finite"] is True
    assert report["requires_grad_any"] is False
    assert report["v541_bitwise"] is True
    assert report["legacy_bitwise"] is True
    assert report["weights_sha_after"] == before
    verify_embedding_bundle(bundle)  # still whitelisted + hashed after load


def test_clean_process_strict_load_forward_e1(
        tmp_path: Path) -> None:
    champ_dir = tmp_path / "runs" / "V541_E1_S17"
    _make_champion(champ_dir, "E1")
    bundle = _export(tmp_path, champ_dir, "E1")
    report = _run_clean_process(bundle, champ_dir / "checkpoint_best.pt",
                                tmp_path)
    assert report["embedding_id"] == "E1"
    assert report["missing"] == [] and report["unexpected"] == []
    assert report["finite"] is True
    assert report["requires_grad_any"] is False
    assert report["v541_bitwise"] is True
    assert report["legacy_bitwise"] is None  # E1 != legacy by construction
    verify_embedding_bundle(bundle)


# ---------------------------------------------------------------------------
# helper contract: build_downstream_handoff defaults
# ---------------------------------------------------------------------------
def test_handoff_builder_defaults() -> None:
    hand = build_downstream_handoff(
        model_id="m", embedding_id="E0", checkpoint_path="/x/checkpoint_best.pt",
        checkpoint_sha256="a" * 64)
    assert hand["schema_version"] == "pes2ts_v541_downstream_handoff_v1"
    assert hand["frozen"]["requires_grad"] is False
    assert hand["frozen"]["read_only"] is True
    assert hand["joint_finetune"]["requires_separate_experiment"] is True
    assert hand["reference_checkpoint"]["path"] == "/x/checkpoint_best.pt"
