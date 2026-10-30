"""V5.4.2 WP3 lane P3-B — assembly + trainer + crash-safe resume tests.

CPU-fast gates for ``pes2ts_core.pesn.models.v542.assembly`` and
``pes2ts_core.pesn.offline.training.v542_train`` (ADR-0033 §2.1/§2.3/§2.6,
config ``training_contract``, plan §6 / §8.2):

* **assembly** — exact active/joint counts for A0-equivalent, A1, A5 and
  A5-C, by-``id()`` param groups (core 1e-4 / head 3e-4, every unique
  parameter exactly once), the frozen five-key forward with
  ``order_score IS q``, the frozen anchor-seed rule ({17, 29, 43} only),
  the construction-order/seed-recipe audit, the step-6 training-seed
  reset (verified), identity recording (lenient by default, strict only
  under ``require_identity=True``) and no ``torch.compile`` anywhere;
* **phases** — W0 freezes the core (``requires_grad=False`` only —
  gradients ``None``, values untouched, head live) and W1 updates all;
  the shared ``w_v`` (and the A5/A5-C shared GRU cell) is updated
  EXACTLY once per step — proven with the closed-form single AdamW step,
  not just structurally;
* **resume round-trip** — ``N`` steps + checkpoint + ``M`` steps is
  BITWISE equal to an uninterrupted ``N+M`` run (model tensors,
  optimizer state, schedule chain, inner-val losses, content digest);
  the observed max abs parameter diff must be exactly ``0.0`` on CPU;
* **typed refusal** — an identity / seed / λ mismatch on resume raises
  ``RESUME_IDENTITY_MISMATCH`` (outcome ``invalid_run``) BEFORE any
  write, leaving the checkpoint byte-identical and the manifests
  untouched;
* **records** — ``run_records.jsonl`` is append-only with monotonically
  non-decreasing steps and exactly one row per checkpoint/eval carrying
  step, inner-val total loss, wall/core seconds, seed and the identity
  SHA fields; selection is by inner-val total loss ONLY (no H2/H3 keys
  exist anywhere in the record surface);
* **smoke-sized run** — a 2-step run completes and writes
  ``checkpoint_last.pt`` / ``checkpoint_best.pt`` + ``metrics.json``.

The fixture environment is the REAL six-table path reused from the
frozen V5.3-MP/V541 trainer tests (``_build_env`` → records → features →
real collate → ``forward_view``), so every batch here goes through the
same collate/forward-view code the formal runs use.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, Mapping

import numpy as np
import pytest

torch = pytest.importorskip("torch")

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from pes2ts_core.pesn.contracts.key_atom_v3 import (  # noqa: E402
    forward_view_v3,
)
from pes2ts_core.pesn.models.v542.assembly import (  # noqa: E402
    JOINT_PARAM_MAX_V542,
    LR_CORE_V542,
    LR_HEAD_V542,
    AssembledV542Model,
    V542AssemblyError,
    assemble_v542_model,
    identity_sha256,
)
from pes2ts_core.pesn.models.v542.candidates import (  # noqa: E402
    CANDIDATE_PARAM_COUNTS,
    CONSTRUCTION_ORDER,
    V542InitMapError,
    V542InitSeedError,
    load_r3_backbone_state,
    resolve_init_source_id,
    seed_init,
)
from pes2ts_core.pesn.models.v542.interfaces import (  # noqa: E402
    A0_ACTIVE_PARAMS_V542,
    A0_JOINT_PARAMS_V542,
    N_DECODER_PARAMS_V542,
    N_PROCESS_PARAMS_V542,
    V542InitError,
    assert_no_torch_compile,
)
from pes2ts_core.pesn.offline.training import v542_train  # noqa: E402
from pes2ts_core.pesn.offline.training.key_atom_v53mp_train_v3 import (  # noqa: E402
    FEATURE_LEVEL_V53MP,
    V53MPTrainerError,
    _FeatureLRU,
    _batch_from_positions,
    _loss_on_batch,
    _update_rng,
)
from pes2ts_core.pesn.offline.training.v541_identity import (  # noqa: E402
    read_manifest,
    run_outcome_for_exception,
    sha256_json,
)
from pes2ts_core.pesn.offline.training.v542_train import (  # noqa: E402
    OPERATIONAL_OUTCOMES_V542,
    RUN_OUTCOMES_V542,
    SCIENTIFIC_OUTCOMES_V542,
    V542_TRAIN_SCHEMA,
    V542Outcome,
    V542Trainer,
    V542TrainerConfig,
    V542TrainerError,
    gate_skipped,
    identity_block_v542,
    invalid_run,
    is_formal_schedule_v542,
    make_optimizer_v542,
    outcome_for_exception,
    resource_failure,
    set_training_stage_v542,
    train_run_v542,
)
from scripts.agent_ops.feature_gate.v3_fixtures import (  # noqa: E402
    make_v3_gate_batch,
)
from tests.pesn.test_r3_v53mp_trainer import (  # noqa: E402
    DEFAULT_FOLDS,
    _build_env,
)

SEED = 17
#: The exact frozen candidate counts asserted by the assembly tests.
EXPECTED_ACTIVE = {
    "A0": A0_ACTIVE_PARAMS_V542,
    "A1": CANDIDATE_PARAM_COUNTS["A1"],
    "A5": CANDIDATE_PARAM_COUNTS["A5"],
    "A5-C": CANDIDATE_PARAM_COUNTS["A5-C"],
}
#: A small but plausible run-identity SHA block (the runner lane owns the
#: real one; the trainer records/compares whatever it is given).
IDENTITY = {
    "source_manifest_sha256": "a" * 64,
    "config_sha256": "b" * 64,
    "common_protocol_sha256": "c" * 64,
}


# ---------------------------------------------------------------------------
# fixtures / helpers
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def r3_state():
    return load_r3_backbone_state()


@pytest.fixture()
def env(tmp_path):
    """The REAL fixture env: six-table CSVs → KeyAtomDatasetV3 → labels."""
    return _build_env(tmp_path, DEFAULT_FOLDS)


def _cfg(**over: Any) -> V542TrainerConfig:
    base: Dict[str, Any] = dict(
        seed=SEED, freeze_encoder_updates=2, joint_updates=4, val_every=1,
        device="cpu", inner_val_max_samples=None)
    base.update(over)
    return V542TrainerConfig(**base)


def _trainer(env, cfg: V542TrainerConfig, out: Path, **kw: Any) -> V542Trainer:
    trainer = V542Trainer(
        config=cfg, out_dir=out, dataset=env.ds, labels_v52=env.labels,
        endpoint_labels=env.endpoint, dev_positions=env.dev_pos,
        identity=IDENTITY, **kw)
    return trainer


def _run(env, cfg: V542TrainerConfig, out: Path, **kw: Any) -> Dict[str, Any]:
    return train_run_v542(env.ds, env.labels, env.endpoint, cfg, out,
                          dev_positions=env.dev_pos, identity=IDENTITY, **kw)


def _load(path: Path) -> Dict[str, Any]:
    return torch.load(str(path), map_location="cpu", weights_only=False)


def _read_jsonl(path: Path) -> list:
    if not path.is_file():
        return []
    return [json.loads(line) for line in
            path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _train_rows(out: Path) -> Dict[int, dict]:
    return {row["update"]: row for row in _read_jsonl(out / "train_log.jsonl")
            if row.get("event") == "train"}


def _records(out: Path) -> list:
    return _read_jsonl(out / "run_records.jsonl")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _opt_equal(x: Any, y: Any, path: str = "") -> None:
    if isinstance(x, dict):
        assert set(x) == set(y), path
        for key in x:
            _opt_equal(x[key], y[key], path + "/" + str(key))
    elif isinstance(x, (list, tuple)):
        assert len(x) == len(y), path
        for i, (u, v) in enumerate(zip(x, y)):
            _opt_equal(u, v, "{}[{}]".format(path, i))
    elif isinstance(x, torch.Tensor):
        assert torch.equal(x, y), path
    elif hasattr(x, "shape") and hasattr(x, "dtype"):
        assert np.array_equal(x, y), path
    else:
        assert x == y, path


def _state_equal(a: Mapping[str, Any], b: Mapping[str, Any]) -> None:
    assert set(a) == set(b)
    for key in a:
        assert torch.equal(a[key], b[key]), key


# ---------------------------------------------------------------------------
# ASSEMBLY: exact counts, by-id param groups, frozen protocol
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("candidate_id", ["A0", "A1", "A5", "A5-C"])
def test_assembly_exact_counts_and_by_id_param_groups(candidate_id, r3_state):
    assembled = assemble_v542_model(
        candidate_id, SEED, device="cpu", identity=IDENTITY,
        require_identity=True)
    assert isinstance(assembled, AssembledV542Model)

    # --- exact counts (registry value for candidates; A0 invariant) -------
    active = EXPECTED_ACTIVE[candidate_id]
    assert assembled.active_params == active
    assert assembled.backbone.active_param_count == active
    joint = active + N_DECODER_PARAMS_V542 + N_PROCESS_PARAMS_V542
    assert assembled.joint_params == joint
    assert joint <= JOINT_PARAM_MAX_V542
    if candidate_id == "A0":
        assert joint == A0_JOINT_PARAMS_V542 == 57_619

    # --- param groups BY id(): names, frozen lrs, exact partition ---------
    assert [entry["name"] for entry in assembled.param_groups] == \
        ["core", "head"]
    assert [entry["lr"] for entry in assembled.param_groups] == \
        [LR_CORE_V542, LR_HEAD_V542] == [1.0e-4, 3.0e-4]
    core_n = sum(int(p.numel()) for p in assembled.core_params)
    head_n = sum(int(p.numel()) for p in assembled.head_params)
    assert core_n == active + N_DECODER_PARAMS_V542
    assert head_n == N_PROCESS_PARAMS_V542
    seen = [id(p) for entry in assembled.param_groups
            for p in entry["params"]]
    assert len(seen) == len(set(seen)), "duplicate param id in a group"
    model_ids = {id(p) for p in assembled.model.parameters()}
    assert set(seen) == model_ids, "every unique param must be grouped once"
    assert len(seen) == len(model_ids)

    # --- shared w_v: R3-exact and (when the backbone aliases it) shared ---
    assert torch.equal(assembled.move_decoder.w_v.weight,
                       r3_state["w_v.weight"])
    assert torch.equal(assembled.move_decoder.w_v.bias, r3_state["w_v.bias"])
    if hasattr(assembled.backbone, "w_v"):
        assert assembled.backbone.w_v is assembled.move_decoder.w_v
    else:
        assert candidate_id in ("A5", "A5-C")

    # --- frozen construction protocol record ------------------------------
    built = assembled.built
    assert built.candidate_id == candidate_id
    assert built.executed_steps == CONSTRUCTION_ORDER[:4]
    assert built.deferred_steps == CONSTRUCTION_ORDER[4:]
    assert built.construction_order == CONSTRUCTION_ORDER
    assert built.seed_init == seed_init(
        SEED, resolve_init_source_id(candidate_id))
    assert assembled.seed_init == built.seed_init
    assert assembled.init_source_id == resolve_init_source_id(candidate_id)
    assert assembled.anchor_seed == assembled.run_seed == SEED
    audit = built.mapping_audit
    assert len(audit.source_keys) == 37
    assert audit.missing == () and audit.unexpected == ()
    assert audit.shape_mismatch == () and audit.dtype_mismatch == ()
    assert set(audit.torch_equal_verified) == set(audit.target_keys)

    # --- identity recorded verbatim + frozen surface ----------------------
    assert assembled.identity == IDENTITY
    assert assembled.identity_sha256 == sha256_json(IDENTITY) == \
        identity_sha256(IDENTITY)
    assert_no_torch_compile(assembled.model)


def test_assembly_five_key_forward_and_order_score_is_q():
    assembled = assemble_v542_model("A1", SEED, device="cpu")
    batch = forward_view_v3(make_v3_gate_batch(n_graphs=2))
    out = assembled.model(batch)
    n_graphs = int(batch["cv_target_delta"].shape[0])
    n_keys = int(batch["key_index"].shape[0])
    assert set(out) == {"u", "z", "q", "endpoint_logit", "order_score"}
    assert out["order_score"] is out["q"]        # frozen shared scalar
    assert tuple(out["z"].shape) == (n_graphs, 64)
    assert tuple(out["q"].shape) == (n_graphs,)
    assert tuple(out["endpoint_logit"].shape) == (n_graphs,)
    assert tuple(out["u"].shape) == (n_keys, 3)
    move = assembled.model.forward_move(batch)
    assert set(move) == {"u", "z"}


@pytest.mark.parametrize("run_seed", [18, 7, 0, 1, 43 + 1, "17", True])
def test_assembly_refuses_any_non_frozen_anchor_seed(run_seed):
    """anchor_seed IS the run seed and only {17, 29, 43} are frozen."""
    with pytest.raises(V542InitSeedError) as excinfo:
        assemble_v542_model("A1", run_seed, device="cpu")
    assert excinfo.value.code == "V542_ANCHOR_SEED"


@pytest.mark.parametrize("run_seed", [17, 29, 43])
def test_assembly_accepts_exactly_the_three_frozen_seeds(run_seed):
    assembled = assemble_v542_model("A1", run_seed, device="cpu")
    assert assembled.anchor_seed == assembled.run_seed == run_seed
    # the anchored head really is seed-specific (step 5 replay)
    other = 29 if run_seed != 29 else 17
    head_other = assemble_v542_model("A1", other, device="cpu")
    if other != run_seed:
        a = assembled.process_head.state_dict()
        b = head_other.process_head.state_dict()
        assert any(not torch.equal(a[k], b[k]) for k in a)


def test_assembly_refuses_unknown_candidate():
    with pytest.raises(V542InitMapError) as excinfo:
        assemble_v542_model("A9", SEED, device="cpu")
    assert excinfo.value.code == "V542_CANDIDATE_UNKNOWN"


def test_assembly_resets_and_verifies_the_training_seed():
    """Step 6: the global torch RNG is reset to the run seed (verified)."""
    torch.manual_seed(12345)
    _ = torch.rand(3)                       # perturb the stream first
    assemble_v542_model("A1", SEED, device="cpu")
    assert torch.initial_seed() == SEED
    actual = torch.rand(5)                  # the stream post-assembly …
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(SEED)
        expected = torch.rand(5)            # … == fresh run-seed stream
    assert torch.equal(actual, expected)


def test_assembly_identity_lenient_by_default_strict_on_demand():
    # lenient: whatever dict is given is recorded (absent ⇒ {})
    lenient = assemble_v542_model("A1", SEED, device="cpu")
    assert lenient.identity == {}
    assert lenient.identity_sha256 == sha256_json({})
    dirty = assemble_v542_model("A1", SEED, device="cpu",
                                identity={"x": None})
    assert dirty.identity == {"x": None}     # recorded VERBATIM
    # strict: absent / dirty are typed refusals
    with pytest.raises(V542AssemblyError) as exc_absent:
        assemble_v542_model("A1", SEED, device="cpu", require_identity=True)
    assert exc_absent.value.code == "V542_IDENTITY_ABSENT"
    # the assembly refusal taxonomy stays inside the V5.4.2 init errors
    assert isinstance(exc_absent.value, V542InitError)
    with pytest.raises(V542AssemblyError) as exc_empty:
        assemble_v542_model("A1", SEED, device="cpu", identity={},
                            require_identity=True)
    assert exc_empty.value.code == "V542_IDENTITY_ABSENT"
    with pytest.raises(V542AssemblyError) as exc_dirty:
        assemble_v542_model("A1", SEED, device="cpu",
                            identity={"source_manifest_sha256": ""},
                            require_identity=True)
    assert exc_dirty.value.code == "V542_IDENTITY_DIRTY"
    with pytest.raises(V542AssemblyError) as exc_none:
        assemble_v542_model("A1", SEED, device="cpu", identity=None,
                            require_identity=True)
    assert exc_none.value.code == "V542_IDENTITY_ABSENT"
    # … and the strict happy path still records the clean block
    clean = assemble_v542_model("A1", SEED, device="cpu",
                                identity=IDENTITY, require_identity=True)
    assert clean.identity == IDENTITY
    assert clean.identity_sha256 == sha256_json(IDENTITY)


def test_assembly_never_calls_torch_compile(monkeypatch):
    def _boom(*_args, **_kwargs):
        raise AssertionError("torch.compile is forbidden (ADR-0033 §2.1)")

    monkeypatch.setattr(torch, "compile", _boom)
    assembled = assemble_v542_model("A0", SEED, device="cpu")
    assert_no_torch_compile(assembled.model)
    batch = forward_view_v3(make_v3_gate_batch(n_graphs=1))
    out = assembled.model(batch)             # forward must also be clean
    assert set(out) == {"u", "z", "q", "endpoint_logit", "order_score"}


def test_assembly_is_deterministic_per_seed():
    a = assemble_v542_model("A1", SEED, device="cpu")
    b = assemble_v542_model("A1", SEED, device="cpu")
    _state_equal(a.model.state_dict(), b.model.state_dict())
    assert a.seed_init == b.seed_init
    c = assemble_v542_model("A1", 29, device="cpu")
    # R3-mapped w_v is seed-independent; fresh init + anchored head differ
    assert torch.equal(a.move_decoder.w_v.weight, c.move_decoder.w_v.weight)
    sa, sc = a.model.state_dict(), c.model.state_dict()
    assert any(not torch.equal(sa[k], sc[k]) for k in sa)
    assert any(not torch.equal(sa[k], sc[k]) for k in sa
               if k.startswith("process_head."))


# ---------------------------------------------------------------------------
# CONFIG: frozen schedule / seeds / lrs / objective
# ---------------------------------------------------------------------------
def test_config_freezes_the_frozen_surface():
    cfg = V542TrainerConfig(candidate_id="A1")
    assert cfg.arm == "C"
    assert cfg.objective == "unified_move_process"
    assert (cfg.lambda_rank, cfg.lambda_q, cfg.lambda_endpoint) == \
        (0.5, 0.5, 0.012385049119773344)
    assert cfg.w0_updates == 400 and cfg.w1_updates == 37500
    assert cfg.total_updates == 37900
    assert (cfg.lr_encoder, cfg.lr_head) == (LR_CORE_V542, LR_HEAD_V542)
    assert cfg.weight_decay == 1.0e-4 and cfg.grad_clip == 1.0
    assert cfg.val_every == 100 and cfg.eval_scope == "inner_val_only"
    assert is_formal_schedule_v542(cfg) is True
    assert is_formal_schedule_v542(
        V542TrainerConfig(candidate_id="A1", smoke_updates=2)) is False

    with pytest.raises(V542TrainerError) as exc_seed:
        V542TrainerConfig(candidate_id="A1", seed=7)
    assert exc_seed.value.code == "V542_SEED_FROZEN"
    with pytest.raises(V542TrainerError) as exc_lr:
        V542TrainerConfig(candidate_id="A1", lr_encoder=2.0e-4)
    assert exc_lr.value.code == "V542_LR_FROZEN"
    with pytest.raises(V542TrainerError) as exc_lr2:
        V542TrainerConfig(candidate_id="A1", lr_head=1.0e-4)
    assert exc_lr2.value.code == "V542_LR_FROZEN"
    with pytest.raises(V542TrainerError) as exc_init:
        V542TrainerConfig(candidate_id="A1", backbone_init="scratch")
    assert exc_init.value.code == "V542_BACKBONE_INIT"
    with pytest.raises(V542TrainerError) as exc_obj:
        V542TrainerConfig(candidate_id="A1", objective="three_arm_matrix")
    assert exc_obj.value.code == "OBJECTIVE"
    # the frozen loss arm still refuses λ drift (V5.3-MP rules, reused)
    with pytest.raises(V53MPTrainerError) as exc_lambda:
        V542TrainerConfig(candidate_id="A1", lambda_q=0.0)
    assert exc_lambda.value.code == "LAMBDA_ARM_MISMATCH"
    # seeds 29 and 43 are legal
    assert V542TrainerConfig(candidate_id="A1", seed=29).seed == 29
    assert V542TrainerConfig(candidate_id="A5", seed=43).seed == 43


# ---------------------------------------------------------------------------
# typed outcomes: operational vs scientific
# ---------------------------------------------------------------------------
def test_operational_outcomes_are_typed_and_distinct_from_results():
    assert RUN_OUTCOMES_V542 == (
        "done", "invalid_run", "scientific_failure", "resource_failure",
        "budget_censored", "gate_skipped")
    assert set(OPERATIONAL_OUTCOMES_V542) == {
        "invalid_run", "resource_failure", "budget_censored", "gate_skipped"}
    assert set(SCIENTIFIC_OUTCOMES_V542) == {"done", "scientific_failure"}

    for build, name in ((gate_skipped, "gate_skipped"),
                        (invalid_run, "invalid_run"),
                        (resource_failure, "resource_failure")):
        outcome = build("reason_code")
        assert outcome.outcome == name
        assert outcome.is_scientific_result is False
    # gate_skipped must always carry a reason code (plan §12)
    with pytest.raises(V542TrainerError) as exc_reason:
        gate_skipped("   ")
    assert exc_reason.value.code == "OUTCOME_REASON"
    with pytest.raises(V542TrainerError) as exc_vocab:
        V542Outcome("totally_not_an_outcome")
    assert exc_vocab.value.code == "OUTCOME_VOCABULARY"
    # results vs operational
    assert V542Outcome("done").is_scientific_result is True
    assert V542Outcome("scientific_failure", "H2 breach"). \
        is_scientific_result is True
    # exceptions map into the operational class, never into a result
    err = V542TrainerError("RESUME_IDENTITY_MISMATCH", "seed drift")
    mapped = outcome_for_exception(err)
    assert mapped.outcome == "invalid_run"
    assert mapped.is_scientific_result is False
    assert run_outcome_for_exception(err) == "invalid_run"
    assert outcome_for_exception(
        MemoryError("boom")).outcome == "resource_failure"
    assert err.outcome == "invalid_run"


# ---------------------------------------------------------------------------
# PHASES: W0 freezes the core / W1 updates all / w_v updates exactly once
# ---------------------------------------------------------------------------
def test_w0_freezes_core_and_updates_head(env, tmp_path):
    cfg = _cfg(candidate_id="A1", freeze_encoder_updates=1,
               smoke_updates=1, val_every=1)
    out = tmp_path / "w0"
    trainer = _trainer(env, cfg, out)
    metrics = trainer.run()
    assert metrics["run_outcome"] == "done"

    ref = assemble_v542_model("A1", SEED, device="cpu").model.state_dict()
    payload = _load(out / "checkpoint_last.pt")
    assert payload["update"] == 1 and payload["phase"] == "w0"
    core_keys = [k for k in payload["model"]
                 if not k.startswith("process_head.")]
    head_keys = [k for k in payload["model"]
                 if k.startswith("process_head.")]
    for key in core_keys:
        assert torch.equal(payload["model"][key], ref[key]), key
    assert any(not torch.equal(payload["model"][key], ref[key])
               for key in head_keys), "the head must train in W0"

    # frozen == requires_grad=False ONLY (grads stay None, never written)
    assembled = trainer.assembled
    assert all(p.grad is None for p in assembled.core_params)
    assert any(p.grad is not None for p in assembled.head_params)
    assert all(not p.requires_grad for p in assembled.core_params)
    assert all(p.requires_grad for p in assembled.head_params)
    rows = _train_rows(out)
    assert rows[1]["core_trainable"] is False
    assert rows[1]["head_trainable"] is True
    # W0 evidence: the shared w_v did NOT move either
    assert torch.equal(payload["model"]["move_decoder.w_v.weight"],
                       ref["move_decoder.w_v.weight"])


def test_w1_updates_all_and_the_stage_switch_is_evidenced(env, tmp_path):
    cfg = _cfg(candidate_id="A1", freeze_encoder_updates=1,
               smoke_updates=3, val_every=100)   # single eval at the end
    out = tmp_path / "w1"
    metrics = _run(env, cfg, out)
    assert metrics["run_outcome"] == "done"
    assert metrics["schedule"]["updates_run"] == 3

    ref = assemble_v542_model("A1", SEED, device="cpu").model.state_dict()
    payload = _load(out / "checkpoint_last.pt")
    assert payload["update"] == 3 and payload["phase"] == "w1"
    assert any(not torch.equal(payload["model"][k], ref[k])
               for k in payload["model"] if not k.startswith("process_head.")
               ), "the core must train in W1"
    assert any(not torch.equal(payload["model"][k], ref[k])
               for k in payload["model"] if k.startswith("process_head."))

    rows = _read_jsonl(out / "train_log.jsonl")
    switches = [r for r in rows if r.get("event") == "stage_switch"]
    assert len(switches) == 1
    sw = switches[0]
    assert sw["update"] == 2 and sw["phase"] == "w1"
    assert sw["core_requires_grad"] is True
    assert sw["core_reachable"] is True
    assert sw["n_core_params_with_grad"] >= 1
    assert sw["core_grad_max_abs"] and sw["core_grad_max_abs"] > 0.0
    by_update = _train_rows(out)
    assert by_update[1]["core_trainable"] is False
    assert by_update[2]["core_trainable"] is True
    assert by_update[3]["core_trainable"] is True


def _single_adamw_step(env, assembled, cfg, positions):
    """ONE trainer-shaped step (frozen loss, clip, AdamW) that returns the
    pre-step values + grads of the tensors under test."""
    set_training_stage_v542(assembled, core_trainable=True)
    optimizer = make_optimizer_v542(assembled, cfg)
    lru = _FeatureLRU(env.ds, capacity=max(len(positions), 1) + 64,
                      level=FEATURE_LEVEL_V53MP, intent_off=False)
    batch, records = _batch_from_positions(env.ds, lru, positions,
                                           cfg.device)
    optimizer.zero_grad(set_to_none=True)
    # endpoint_pos_weight is a loss SCALE only — irrelevant to the
    # update-count property under test; the loss itself is frozen.
    loss_out = _loss_on_batch(
        assembled.model, batch, records, env.labels, env.endpoint, cfg,
        endpoint_pos_weight=1.0,
        pair_rng=_update_rng(cfg.seed, 1, salt=1))
    loss_out["loss"].backward()
    trainable = [p for p in assembled.model.parameters() if p.requires_grad]
    torch.nn.utils.clip_grad_norm_(trainable, float(cfg.grad_clip))
    watched = [assembled.move_decoder.w_v]
    cell = getattr(assembled.backbone, "gru_cells", None)
    if cell:                                   # A5 / A5-C shared cell
        watched.append(cell[0])
    snapshots = []
    for module in watched:
        for param in module.parameters():
            assert param.grad is not None, "missing grad for a watched param"
            snapshots.append((param, param.detach().clone(),
                              param.grad.detach().clone()))
    optimizer.step()
    return optimizer, snapshots


def _assert_single_update(param, before, grad, *, lr, wd, eps=1.0e-8):
    """Closed-form first AdamW step: exactly ONE decoupled decay + ONE
    normalized step.

    CPU fp32 tolerance: the closed form and ``AdamW.addcdiv_`` round in a
    different order, so agreement is asserted within 8 float32 ULP of the
    parameter magnitude (observed max diff ≈ 7.5e-9 = ONE ULP at
    |p| ≈ 0.06).  A double update deviates by ~lr (1e-4 ⇒ ≫ tol) and a
    skipped update deviates by ~|step| (also ≫ tol), so the bound still
    proves exactly one optimizer application.
    """
    after = param.detach()
    expected = before * (1.0 - lr * wd) - lr * grad / (grad.abs() + eps)
    diff = float((after - expected).abs().max())
    eps32 = 1.1920928955078125e-07
    magnitude = max(float(after.abs().max()), float(expected.abs().max()))
    tol = max(8.0 * eps32 * magnitude, 1.0e-9)
    assert diff <= tol, (
        "{}: single-step closed form missed by {} (> {} ULP tol {}) — the "
        "shared parameter was not updated EXACTLY once".format(
            tuple(param.shape), diff, 8, tol))
    # negative control: a double update would be ~lr away, never ~1 ULP
    # (guarded: only meaningful when the single step actually MOVED the
    # parameter by more than the control's threshold)
    step_mag = float((after - before).abs().max())
    double = (before * (1.0 - lr * wd) * (1.0 - lr * wd)
              - 2.0 * lr * grad / (grad.abs() + eps))
    double_diff = float((after - double).abs().max())
    if step_mag > 1.0e-5:
        assert double_diff > max(1.0e-6, 50.0 * tol), (
            "a double update is indistinguishable from the observed step")
    return diff, tol


@pytest.mark.parametrize("candidate_id", ["A1", "A5", "A5-C"])
def test_shared_params_update_exactly_once(env, tmp_path, candidate_id):
    """The shared ``w_v`` (and A5/A5-C's shared GRU cell) is updated
    EXACTLY once per optimizer step — no double update from the shared
    module, and never a duplicate optimizer group entry."""
    cfg = V542TrainerConfig(candidate_id=candidate_id, seed=SEED,
                            device="cpu")
    assembled = assemble_v542_model(candidate_id, SEED, device="cpu",
                                    identity=IDENTITY, require_identity=True)
    optimizer, snapshots = _single_adamw_step(
        env, assembled, cfg, positions=list(env.pool_pos[:8]))

    # structural: exactly ONE group owns each watched tensor (AdamW itself
    # refuses a parameter listed in two groups — construction succeeded)
    for param, _before, _grad in snapshots:
        owners = [g["name"] for g in optimizer.param_groups
                  if any(p is param for p in g["params"])]
        assert owners == ["core"], owners

    # numerical: exactly one AdamW application (tolerance: ≤ 8 float32
    # ULP of the parameter magnitude; observed ≈ 1 ULP)
    for param, before, grad in snapshots:
        diff, tol = _assert_single_update(param, before, grad,
                                          lr=float(cfg.lr_encoder),
                                          wd=float(cfg.weight_decay))
        assert diff <= tol

    # named_parameters() must mention the shared w_v exactly once
    wv_names = [n for n, _ in assembled.model.named_parameters()
                if n.endswith("w_v.weight")]
    assert len(wv_names) == 1, wv_names
    # dual-share wiring: an aliasing backbone shares the SAME module
    # object (one tensor ⇒ one optimizer slot), a non-aliasing one
    # (A5/A5-C) leaves w_v owned solely by the decoder.
    if hasattr(assembled.backbone, "w_v"):
        assert assembled.backbone.w_v is assembled.move_decoder.w_v
    else:
        assert candidate_id in ("A5", "A5-C")
        assert wv_names == ["move_decoder.w_v.weight"]


# ---------------------------------------------------------------------------
# RESUME: bitwise round-trip, identity refusal, append-only records
# ---------------------------------------------------------------------------
def test_resume_bitwise_roundtrip_n_plus_m(env, tmp_path):
    """3 + 3 resumed == 6 uninterrupted (CPU bitwise: max diff 0.0)."""
    resumed_dir = tmp_path / "resumed"
    fresh_dir = tmp_path / "fresh"
    common = dict(candidate_id="A1", freeze_encoder_updates=2,
                  joint_updates=4, val_every=1, device="cpu",
                  inner_val_max_samples=None)

    leg1 = _run(env, V542TrainerConfig(smoke_updates=3, **common),
                resumed_dir, resume=False)
    leg2 = _run(env, V542TrainerConfig(smoke_updates=6, **common),
                resumed_dir, resume=True)
    fresh = _run(env, V542TrainerConfig(smoke_updates=6, **common),
                 fresh_dir, resume=False)
    assert leg1["schedule"]["updates_run"] == 3
    assert leg2["schedule"]["updates_run"] == 3
    assert fresh["schedule"]["updates_run"] == 6

    # --- model weights: BITWISE ------------------------------------------
    a = _load(resumed_dir / "checkpoint_last.pt")
    b = _load(fresh_dir / "checkpoint_last.pt")
    assert a["update"] == b["update"] == 6
    assert a["phase"] == b["phase"] == "w1"
    max_abs = 0.0
    for key in a["model"]:
        assert torch.equal(a["model"][key], b["model"][key]), key
        max_abs = max(max_abs,
                      float((a["model"][key] - b["model"][key]).abs().max()))
    assert max_abs == 0.0, "observed max abs parameter diff {}".format(max_abs)

    # --- optimizer state + RNG + schedule + digest -------------------------
    _opt_equal(a["optimizer"], b["optimizer"], "optimizer")
    assert a["schedule"] == b["schedule"]
    assert a["content_digest_sha256"] == b["content_digest_sha256"]
    assert torch.equal(a["rng"]["torch"], b["rng"]["torch"])
    assert a["identity_sha256"] == b["identity_sha256"] == \
        sha256_json(a["identity"])
    assert a["sampler_counters"] == b["sampler_counters"]
    # RNG was restored on resume (python/numpy/torch states match)
    assert a["rng"]["numpy"][1].tolist() == b["rng"]["numpy"][1].tolist()
    assert a["rng"]["python"] == b["rng"]["python"]

    # --- best checkpoint ---------------------------------------------------
    ba = _load(resumed_dir / "checkpoint_best.pt")
    bb = _load(fresh_dir / "checkpoint_best.pt")
    assert ba["best_val"] == bb["best_val"]
    assert ba["best_update"] == bb["best_update"]
    assert ba["content_digest_sha256"] == bb["content_digest_sha256"]
    for key in ba["model"]:
        assert torch.equal(ba["model"][key], bb["model"][key]), key

    # --- per-step train rows identical apart from wall-clock fields -------
    def strip(row):
        return {k: v for k, v in row.items()
                if k not in ("wall_s", "core_s", "elapsed_s",
                             "phase_timing", "resources")}

    rows_r, rows_f = _train_rows(resumed_dir), _train_rows(fresh_dir)
    assert set(rows_r) == set(rows_f) == set(range(1, 7))
    for update in rows_f:
        assert strip(rows_r[update]) == strip(rows_f[update]), update

    # --- inner-val losses per step ----------------------------------------
    def iv_by_step(out):
        return {r["step"]: r["inner_val_total_loss"]
                for r in _records(out) if r["event"] == "inner_val"}

    assert iv_by_step(resumed_dir) == iv_by_step(fresh_dir)
    assert leg2["best_inner_val_loss"] == fresh["best_inner_val_loss"]
    assert leg2["selected_update"] == fresh["selected_update"]
    assert leg2["best_update"] == fresh["best_update"]
    assert leg2["last_train_loss"] == fresh["last_train_loss"]
    assert leg2["sampler"]["batch_schedule_sha256"] == \
        fresh["sampler"]["batch_schedule_sha256"]
    # leg-1 records are preserved byte-for-byte (append-only)
    manifest_r = read_manifest(resumed_dir / "manifest.jsonl")
    assert [r["event"] for r in manifest_r].count("resume") == 1
    assert [r["event"] for r in manifest_r].count("run_complete") == 2


@pytest.mark.parametrize("over, identity_over, field", [
    (dict(), {"source_manifest_sha256": "f" * 64},
     "source_manifest_sha256"),
    (dict(seed=29), {}, "seed"),
    (dict(lambda_q=0.4), {}, "lambda_q"),
])
def test_resume_identity_mismatch_is_a_typed_refusal_leaving_the_ckpt(
        env, tmp_path, over, identity_over, field):
    out = tmp_path / "mismatch"
    identity = dict(IDENTITY)
    identity.update(identity_over)
    _run(env, _cfg(candidate_id="A1", smoke_updates=3), out, resume=False)

    ckpt = out / "checkpoint_last.pt"
    records_path = out / "run_records.jsonl"
    manifest_path = out / "manifest.jsonl"
    before_ckpt = _sha256(ckpt)
    before_records = records_path.read_bytes()
    before_manifest = manifest_path.read_bytes()

    cfg = _cfg(candidate_id="A1", smoke_updates=6, **over)
    with pytest.raises(V542TrainerError) as excinfo:
        train_run_v542(env.ds, env.labels, env.endpoint, cfg, out,
                       dev_positions=env.dev_pos, identity=identity,
                       resume=True)
    exc = excinfo.value
    assert exc.code == "RESUME_IDENTITY_MISMATCH"
    assert field in exc.fields
    assert exc.outcome == "invalid_run"
    assert run_outcome_for_exception(exc) == "invalid_run"
    assert outcome_for_exception(exc).is_scientific_result is False

    # NOTHING was written: checkpoint, records and manifest are untouched
    assert _sha256(ckpt) == before_ckpt
    assert records_path.read_bytes() == before_records
    assert manifest_path.read_bytes() == before_manifest
    assert "resume" not in [r["event"] for r in read_manifest(manifest_path)]
    # and the crashed run never switched seeds
    assert _load(ckpt)["identity"]["seed"] == SEED


def test_resume_refuses_tampered_content_digest(env, tmp_path):
    from pes2ts_core.pesn.offline.training.v541_identity import \
        content_digest_of

    out = tmp_path / "tamper"
    _run(env, _cfg(candidate_id="A1", smoke_updates=3), out, resume=False)
    path = out / "checkpoint_last.pt"
    payload = _load(path)
    assert content_digest_of(payload) == payload["content_digest_sha256"]
    key = next(iter(payload["model"]))
    payload["model"][key] = payload["model"][key] + 1.0
    torch.save(payload, str(path))
    before = _sha256(path)
    with pytest.raises(V542TrainerError) as excinfo:
        _run(env, _cfg(candidate_id="A1", smoke_updates=6), out,
             resume=True)
    assert excinfo.value.code == "RESUME_CONTENT_DIGEST"
    assert excinfo.value.fields == ["content_digest_sha256"]
    assert _sha256(path) == before          # refusal before any write


def test_records_are_append_only_monotone_and_inner_val_only(env, tmp_path):
    out = tmp_path / "records"
    _run(env, _cfg(candidate_id="A1", smoke_updates=3), out, resume=False)
    first_bytes = (out / "run_records.jsonl").read_bytes()
    first_manifest = (out / "manifest.jsonl").read_bytes()
    _run(env, _cfg(candidate_id="A1", smoke_updates=6), out, resume=True)
    records_path = out / "run_records.jsonl"
    assert records_path.read_bytes().startswith(first_bytes)   # APPEND-only
    assert (out / "manifest.jsonl").read_bytes().startswith(first_manifest)

    rows = _records(out)
    assert rows, "run records must exist"
    steps = [r["step"] for r in rows]
    assert steps == sorted(steps), "steps must never decrease (append-only)"

    # exactly one row per checkpoint / eval event, all required fields
    required = {"step", "inner_val_total_loss", "wall_s", "core_s", "seed",
                "identity", "identity_sha256", "phase", "event",
                "grad_finite_all_finite"}
    allowed = required | {"objective", "candidate_id", "best_update",
                          "best_inner_val_total_loss", "selection_metric",
                          "checkpoint", "grad_nonfinite_params"}
    iv_steps = [r["step"] for r in rows if r["event"] == "inner_val"]
    last_steps = [r["step"] for r in rows if r["event"] == "checkpoint_last"]
    assert iv_steps == last_steps == [1, 2, 3, 4, 5, 6]
    for row in rows:
        assert required <= set(row), sorted(row)
        assert set(row) <= allowed, sorted(set(row) - allowed)
        assert row["seed"] == SEED
        assert row["identity_sha256"] == sha256_json(row["identity"])
        assert row["identity"]["config_sha256"] == \
            sha256_json(v542_train.config_identity_dict(
                _cfg(candidate_id="A1", smoke_updates=6)))
        assert isinstance(row["wall_s"], (int, float))
        assert isinstance(row["core_s"], (int, float))
        # C3: a healthy run records the trainer-side gradient screen on
        # EVERY record row (frozen field name the harness consumes)
        assert row["grad_finite_all_finite"] is True
        assert "grad_nonfinite_params" not in row

    # selection: checkpoint_best at the FIRST inner-val minimum, nowhere else
    losses = {r["step"]: r["inner_val_total_loss"]
              for r in rows if r["event"] == "inner_val"}
    best_step = min(losses, key=lambda step: losses[step])
    best_steps = [r["step"] for r in rows if r["event"] == "checkpoint_best"]
    assert best_steps == sorted(set(best_steps))
    payload = _load(out / "checkpoint_best.pt")
    assert payload["best_update"] == best_step
    assert payload["best_val"] == losses[best_step]
    metrics = json.loads((out / "metrics.json").read_text())
    assert metrics["best_update"] == best_step
    assert metrics["selection_metric"] == "inner-val total loss"
    assert metrics["h3_metric_selection_forbidden"] is True

    # NO H2/H3 metric can appear in the record surface (selection leak test)
    forbidden = ("auprc", "q_corr", "rmse", "fixed_coverage",
                 "dead_fraction", "h2_breach", "endpoint_auprc")
    for row in rows:
        for key in row:
            assert not any(token in key for token in forbidden), key
    assert "endpoint_auprc" not in metrics and "q_corr" not in metrics


def test_two_step_smoke_run_writes_last_and_best(env, tmp_path):
    out = tmp_path / "smoke2"
    cfg = _cfg(candidate_id="A5", freeze_encoder_updates=1,
               smoke_updates=2, val_every=1)
    metrics = _run(env, cfg, out)
    assert metrics["run_outcome"] == "done"
    assert metrics["is_formal_schedule"] is False
    assert metrics["schedule"]["updates_run"] == 2

    assert (out / "metrics.json").is_file()
    last = _load(out / "checkpoint_last.pt")
    best = _load(out / "checkpoint_best.pt")
    assert last["update"] == 2 and last["phase"] == "w1"
    assert best["update"] in (1, 2)
    # payload completeness: model / optimizer / phase / step / RNG /
    # sampler counter / identity digest / schedule / seeds
    for key in ("schema", "model", "optimizer", "update", "phase",
                "best_val", "best_update", "config", "config_full_sha256",
                "identity", "identity_sha256", "provenance", "seeds",
                "rng", "sampler", "schedule", "sampler_counters",
                "data_position", "git", "env", "content_digest_sha256"):
        assert key in last, key
    assert last["schema"] == V542_TRAIN_SCHEMA
    assert last["identity_sha256"] == sha256_json(last["identity"])
    assert last["identity"]["candidate_id"] == "A5"
    assert set(last["rng"]) >= {"python", "numpy", "torch"}
    assert last["sampler"]["sampler_id"] == "v541_curve_stratified_v1"
    assert len(last["sampler_counters"]) == 3      # forward/reverse/total
    assert last["seeds"]["run_seed"] == SEED
    assert last["seeds"]["anchor_seed"] == SEED
    assert last["param_counts"] == {
        "backbone_active": EXPECTED_ACTIVE["A5"],
        "joint": EXPECTED_ACTIVE["A5"] + 49 + 2178,
    }
    events = [r["event"] for r in read_manifest(out / "manifest.jsonl")]
    assert "checkpoint_last" in events and "checkpoint_best" in events
    assert "run_complete" in events
    assert metrics["param_groups"][0]["lr"] == LR_CORE_V542
    assert metrics["param_groups"][1]["lr"] == LR_HEAD_V542


def test_trainer_never_calls_torch_compile(env, tmp_path, monkeypatch):
    def _boom(*_args, **_kwargs):
        raise AssertionError("torch.compile is forbidden (ADR-0033 §2.1)")

    monkeypatch.setattr(torch, "compile", _boom)
    out = tmp_path / "nocompile"
    metrics = _run(env, _cfg(candidate_id="A5", freeze_encoder_updates=1,
                             smoke_updates=2), out)
    assert metrics["run_outcome"] == "done"
    assert metrics["schedule"]["torch_compile"] is False


def test_sampler_binding_drift_is_a_typed_refusal(env, tmp_path, monkeypatch):
    monkeypatch.setattr(v542_train, "sampler_spec_sha256",
                        lambda: "0" * 64)
    with pytest.raises(V542TrainerError) as excinfo:
        _run(env, _cfg(candidate_id="A1", smoke_updates=2),
             tmp_path / "drift")
    assert excinfo.value.code == "V542_SAMPLER_DRIFT"


def test_identity_block_carries_the_canonical_run_fields():
    cfg = _cfg(candidate_id="A1")
    block = identity_block_v542(cfg, IDENTITY, pool_sha256="d" * 64)
    assert block["schema"] == V542_TRAIN_SCHEMA
    assert block["candidate_id"] == "A1"
    assert block["seed"] == block["anchor_seed"] == SEED
    assert block["sampler_pool_sha256"] == "d" * 64
    assert len(block["config_sha256"]) == 64
    assert block["source_manifest_sha256"] == IDENTITY[
        "source_manifest_sha256"]
    # canonical fields win: a caller cannot smuggle a foreign seed
    poisoned = identity_block_v542(cfg, {"seed": 99}, pool_sha256=None)
    assert poisoned["seed"] == SEED


# ---------------------------------------------------------------------------
# C3 — trainer-side gradient finiteness on every run-record row
# ---------------------------------------------------------------------------
def test_grad_finite_state_reports_nonfinite_params():
    """Unit gate for the frozen screen: None grads are vacuously finite;
    a single inf/NaN gradient flips the flag and NAMES the offender."""
    model = torch.nn.Linear(2, 1)
    # no grads yet → vacuously finite
    assert v542_train.grad_finite_state(model) == {
        "grad_finite_all_finite": True}
    model.weight.grad = torch.zeros_like(model.weight)
    assert v542_train.grad_finite_state(model) == {
        "grad_finite_all_finite": True}
    # an inf gradient flips the flag and lists the parameter
    model.bias.grad = torch.full_like(model.bias, float("inf"))
    report = v542_train.grad_finite_state(model)
    assert report["grad_finite_all_finite"] is False
    assert report["grad_nonfinite_params"] == ["bias"]
    # a NaN gradient flips it too; the list is sorted by name
    model.bias.grad = torch.zeros_like(model.bias)
    model.weight.grad = torch.full_like(model.weight, float("nan"))
    report = v542_train.grad_finite_state(model)
    assert report["grad_finite_all_finite"] is False
    assert report["grad_nonfinite_params"] == ["weight"]


def test_nonfinite_gradient_flips_the_run_record_field(env, tmp_path):
    """A synthetic non-finite gradient (injected into a live head
    parameter via a backward hook) flips ``grad_finite_all_finite`` on
    EVERY run-record row and names the poisoned parameter — the frozen
    field the campaign diagnostic harness consumes (oracle gate-3 C3)."""
    cfg = _cfg(candidate_id="A1", freeze_encoder_updates=1,
               smoke_updates=1, val_every=1)
    out = tmp_path / "gradnan"
    trainer = _trainer(env, cfg, out)
    st = trainer._prepare()
    # inject a NaN gradient into exactly ONE head parameter's backward …
    target = st.assembled.head_params[0]
    target_name = next(n for n, p in st.model.named_parameters()
                       if p is target)
    target.register_hook(lambda g: g * float("nan"))
    # … and keep the poisoned gradient OUT of the weights (no-op step),
    # so the synthetic non-finiteness stays observable ONLY through the
    # frozen record field instead of poisoning the whole model
    st.optimizer.step = lambda *args, **kwargs: None
    loop = trainer._train_loop(st)
    metrics = trainer._finalise(st, loop)
    assert metrics["run_outcome"] == "done"

    rows = _records(out)
    assert rows, "the run must still record its eval/checkpoint rows"
    for row in rows:
        assert row["grad_finite_all_finite"] is False
        # the poisoned parameter is NAMED (plus any NaN-propagated
        # downstream params — the injected one must be present)
        assert isinstance(row["grad_nonfinite_params"], list)
        assert target_name in row["grad_nonfinite_params"]
    # the screen is measured on the RAW backward output (pre-clip), so
    # the injected parameter is the one named first
    assert rows[0]["grad_nonfinite_params"] == [target_name]


def test_normal_run_records_grad_finite_true_on_every_row(env, tmp_path):
    """A healthy tiny run records ``grad_finite_all_finite: true`` on
    EVERY run-record row (and never lists ``grad_nonfinite_params``)."""
    out = tmp_path / "gradok"
    metrics = _run(env, _cfg(candidate_id="A1", freeze_encoder_updates=1,
                             smoke_updates=2, val_every=1), out)
    assert metrics["run_outcome"] == "done"
    rows = _records(out)
    assert rows
    for row in rows:
        assert row["grad_finite_all_finite"] is True
        assert "grad_nonfinite_params" not in row
