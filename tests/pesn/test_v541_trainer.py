"""V5.4.1 WP4b — unified trainer tests (``v541_train.py``, route E).

CPU-fast gates for the NEW unified MOVE+PROCESS trainer (plan §7 / §7.5):

* config: internal ``arm=C`` loss only + external ``objective`` frozen,
  identity SHAs (common/embedding/feature/config) and the run-length
  exclusion rule, typed refusals (eval scope, batch shape, independent
  init seed, profile window);
* THREE explicit parameter groups with their lrs — E0's ``{param: lr}``
  map is identical to the legacy arm-C optimizer, candidates get
  ``embedding`` at ``lr_encoder`` (never ``lr_head``), and an
  unclassifiable parameter is a typed refusal;
* E0 model parity vs the legacy build (state dict + five outputs) and R3
  strict warmstart;
* W0/W1: backbone frozen with ``requires_grad=False`` only (forward NOT
  under ``no_grad``), core reachable at the W0→W1 switch;
* the NEW stratified sampler: 8×4 batches, four strata per curve with
  2+2 direction balance, positions identical to the pure
  ``(seed, update, 0)`` draw;
* dev firewall: dev positions are used ONLY to exclude dev curves — a
  poisoned dataset/label table proves zero dev feature/label/eval calls,
  and metrics carry ``evaluation_scope="inner_val"`` /
  ``dev_eval_done=false``;
* monitoring: monitor-step toggling of ``diagnostics_enabled``, alpha
  learning evidence (grad-based, NOT AdamW decay), ``null + reason`` for
  an unsupervised term, per-group grad norms;
* ``profile_only`` short window marked as NOT a formal run;
* the independent §7.4 run-outcome vocabulary and the "old ledger
  untouched" guarantee.

The real E1–E5 candidates are registered by WP3 — this file uses a
test-only dummy embedding (WP2 discipline) and never fakes a candidate
implementation.  Resume/bitwise behaviour lives in
``test_v541_trainer_resume.py``.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from pes2ts_core.pesn.contracts.key_atom_v3 import forward_view_v3  # noqa: E402
from pes2ts_core.pesn.models.key_atom_egnn_v3 import StarEGNNV3  # noqa: E402
from pes2ts_core.pesn.models.key_atom_heads_v3 import (  # noqa: E402
    KeyAtomV53MPModel,
)
from pes2ts_core.pesn.models.v541 import (  # noqa: E402
    EMBEDDING_ID_E0,
    HIDDEN_V541,
    NodeDeltaEmbedding,
    V541Model,
    register_embedding,
    registry_restore,
    registry_snapshot,
)
from pes2ts_core.pesn.offline.training import key_atom_v53mp_train_v3 as tr  # noqa: E402
from pes2ts_core.pesn.offline.training.key_atom_bc_v3 import (  # noqa: E402
    set_seed,
)
from pes2ts_core.pesn.offline.training.v541_identity import (  # noqa: E402
    RUN_OUTCOMES_V541,
    V541RunBudgetError,
    run_outcome_for_exception,
)
from pes2ts_core.pesn.offline.training.v541_monitor import (  # noqa: E402
    embedding_diagnostics,
    set_diagnostics_enabled_v541,
)
from pes2ts_core.pesn.offline.training.v541_strata import (  # noqa: E402
    BATCH_CURVES_V541,
    FRAMES_PER_CURVE_V541,
    StrataIndex,
    batch_rng_v541,
    curve_batch_positions_v541,
    sampler_spec_sha256,
    strata_of_position_v541,
)
from pes2ts_core.pesn.offline.training.v541_train import (  # noqa: E402
    V541_FORMAL_TOTAL_UPDATES,
    V541_LAMBDA_ENDPOINT,
    V541_LAMBDA_Q,
    V541_LAMBDA_RANK,
    V541_TRAIN_SCHEMA,
    V541_W0_UPDATES,
    V541_W1_UPDATES,
    V541TrainerConfig,
    V541TrainerError,
    _edge_triplet_block,
    config_full_sha256,
    config_identity_sha256,
    is_formal_schedule,
    load_model_v541,
    named_param_groups_v541,
    param_groups_v541,
    set_training_stage_v541,
    train_run_v541,
)
from scripts.agent_ops.feature_gate.v3_fixtures import make_v3_batch  # noqa: E402
from tests.pesn.test_r3_v53mp_trainer import (  # noqa: E402
    DEFAULT_FOLDS,
    _build_env,
)

SEED = 17
INIT_SEED = 777
TEST_ID = "TEST_TRAINER_MLP"
COUNT_ID = "TEST_TRAINER_COUNT_MLP"
EMPTY_ID = "TEST_TRAINER_EMPTY_MLP"
N_NODES = 7


# ---------------------------------------------------------------------------
# test-only dummy embeddings (E(3)-safe; WP3 registers the real candidates)
# ---------------------------------------------------------------------------
class _DummyEmbedding(NodeDeltaEmbedding):
    """No diagnostics surface at all — the pre-WP3 candidate shape."""

    def __init__(self, hidden: int = HIDDEN_V541) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(hidden, hidden), torch.nn.SiLU(),
            torch.nn.Linear(hidden, hidden))

    def forward(self, h_legacy, batch):        # batch deliberately unread
        return self.net(h_legacy)


class _CountingEmbedding(_DummyEmbedding):
    """WP3-shaped candidate: own toggle + ``last_diagnostics`` counts."""

    def __init__(self, hidden: int = HIDDEN_V541) -> None:
        super().__init__(hidden)
        self.diagnostics_enabled: bool = True
        self.last_diagnostics: dict = {}

    def forward(self, h_legacy, batch):
        out = self.net(h_legacy)
        if self.diagnostics_enabled:
            self.last_diagnostics = {
                "n_atoms": float(h_legacy.shape[0]),
                "n_triplets": 17.0,
                "zero_neighbor_frac": 0.125,
                "zero_angle_frac": 0.0625,
            }
        else:
            # WP3 provider contract under diagnostics=False: counts == {}
            self.last_diagnostics = {}
        return out


class _EmptyDiagEmbedding(_CountingEmbedding):
    """Exposes the attribute but never fills it (providers diagnostics=False)."""

    def forward(self, h_legacy, batch):
        out = self.net(h_legacy)
        self.last_diagnostics = {}
        return out


@pytest.fixture(autouse=True)
def _registry():
    snap = registry_snapshot()
    register_embedding(TEST_ID, lambda: _DummyEmbedding())
    register_embedding(COUNT_ID, lambda: _CountingEmbedding())
    register_embedding(EMPTY_ID, lambda: _EmptyDiagEmbedding())
    yield
    registry_restore(snap)


@pytest.fixture()
def env(tmp_path):
    return _build_env(tmp_path, DEFAULT_FOLDS)


def _cfg(**over) -> V541TrainerConfig:
    base = dict(
        seed=SEED, backbone_init="scratch",
        freeze_encoder_updates=2, joint_updates=4, smoke_updates=6,
        val_every=2, grad_monitor_every=2, device="cpu")
    base.update(over)
    return V541TrainerConfig(**base)


def _read_log(out: Path) -> list:
    path = out / "train_log.jsonl"
    if not path.is_file():
        return []
    return [json.loads(ln) for ln in
            path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def _train_rows(out: Path) -> dict:
    return {e["update"]: e for e in _read_log(out)
            if e.get("event") == "train"}


# ---------------------------------------------------------------------------
# config / identity
# ---------------------------------------------------------------------------


def test_config_defaults_are_the_frozen_contract() -> None:
    cfg = V541TrainerConfig()
    assert cfg.arm == "C"                       # INTERNAL loss arm only
    assert cfg.objective == "unified_move_process"
    assert cfg.eval_scope == "inner_val_only"
    assert cfg.lambda_rank == V541_LAMBDA_RANK == 0.5
    assert cfg.lambda_q == V541_LAMBDA_Q == 0.5
    assert cfg.lambda_endpoint == V541_LAMBDA_ENDPOINT
    assert cfg.w0_updates == V541_W0_UPDATES == 400
    assert cfg.w1_updates == V541_W1_UPDATES == 37500
    assert cfg.total_updates == 37900
    assert cfg.inner_val_max_samples == 8192
    assert cfg.frames_per_curve == FRAMES_PER_CURVE_V541 == 4
    assert cfg.batch_curves == BATCH_CURVES_V541 == 8
    assert cfg.val_every == 100 and cfg.grad_monitor_every == 100
    assert cfg.lr_head == 3.0e-4 and cfg.lr_encoder == 1.0e-4
    assert cfg.weight_decay == 1.0e-4 and cfg.grad_clip == 1.0
    assert cfg.embedding_id == EMBEDDING_ID_E0 and cfg.init_seed is None
    assert cfg.common_protocol_sha256 is None
    assert cfg.embedding_spec_sha256 is None
    assert cfg.feature_recipe_sha256 is None
    assert cfg.profile_only is False and cfg.host_limit_gib is None


@pytest.mark.parametrize("over,code", [
    ({"objective": "three_arm_matrix"}, "OBJECTIVE"),
    ({"eval_scope": "dev"}, "EVAL_SCOPE"),
    ({"frames_per_curve": 2}, "BATCH_SHAPE"),
    ({"profile_only": True}, "PROFILE_WINDOW"),
    ({"embedding_id": "E1"}, "INIT_SEED_REQUIRED"),
    ({"embedding_id": "E1", "init_seed": SEED}, "INIT_SEED_NOT_INDEPENDENT"),
    ({"arm": "A"}, "LAMBDA_ARM_MISMATCH"),
    ({"inner_val_max_samples": 0}, "INNER_VAL_MAX"),
])
def test_config_typed_refusals(over, code) -> None:
    # the legacy arm rules raise the BASE class; everything V541-specific
    # raises the subclass — one assertion covers both.
    with pytest.raises(tr.V53MPTrainerError) as exc:
        V541TrainerConfig(**over)
    assert exc.value.code == code


def test_candidate_config_accepts_independent_init_seed() -> None:
    cfg = V541TrainerConfig(embedding_id=TEST_ID, init_seed=INIT_SEED)
    assert cfg.init_seed == INIT_SEED != cfg.seed
    assert cfg.embedding_id == TEST_ID
    with pytest.raises(V541TrainerError) as exc:
        V541TrainerConfig(embedding_id=TEST_ID, init_seed=None)
    assert exc.value.code == "INIT_SEED_REQUIRED"


def test_identity_config_sha_excludes_only_run_length_and_resource() -> None:
    cfg20 = _cfg(smoke_updates=20)
    cfg40 = _cfg(smoke_updates=40)
    # run-length knobs never change what ONE update computes
    assert config_identity_sha256(cfg20) == config_identity_sha256(cfg40)
    assert config_full_sha256(cfg20) != config_full_sha256(cfg40)
    # numerics DO change identity
    assert config_identity_sha256(_cfg(seed=29)) != \
        config_identity_sha256(_cfg(seed=SEED))
    assert config_identity_sha256(_cfg(lambda_q=0.4)) != \
        config_identity_sha256(_cfg())
    assert config_identity_sha256(_cfg(order_min_gap=0.5)) != \
        config_identity_sha256(_cfg())
    assert config_identity_sha256(_cfg(device="cuda")) != \
        config_identity_sha256(_cfg())
    # resource / profile knobs are outside the identity too
    assert config_identity_sha256(_cfg(host_limit_gib=12.0)) == \
        config_identity_sha256(_cfg())
    assert config_identity_sha256(
        _cfg(profile_only=True, profile_w0_updates=1,
             profile_w1_updates=1)) == config_identity_sha256(_cfg())


# ---------------------------------------------------------------------------
# parameter groups (§7.1)
# ---------------------------------------------------------------------------


def _lr_by_param_name(model, groups) -> dict:
    names = {id(p): n for n, p in model.named_parameters()}
    out = {}
    for grp in groups:
        for p in grp["params"]:
            out[names[id(p)]] = float(grp["lr"])
    return out


def test_e0_param_groups_identical_to_legacy_arm_c() -> None:
    set_seed(SEED)
    v541 = V541Model()
    cfg = V541TrainerConfig(seed=SEED)
    groups = param_groups_v541(v541, cfg)
    assert [g["name"] for g in groups] == ["backbone", "process_head"]

    set_seed(SEED)
    legacy = KeyAtomV53MPModel()
    legacy_cfg = tr.V53MPTrainerConfig(
        arm="C", seed=SEED, lambda_rank=V541_LAMBDA_RANK,
        lambda_q=V541_LAMBDA_Q, lambda_endpoint=V541_LAMBDA_ENDPOINT)
    legacy_opt = tr._make_optimizer(legacy, legacy_cfg)

    assert _lr_by_param_name(v541, groups) == \
        _lr_by_param_name(legacy, legacy_opt.param_groups)
    # every parameter is classified, nothing falls through
    named = named_param_groups_v541(v541, cfg)
    assert sum(len(v) for v in named.values()) == \
        len(list(v541.named_parameters()))
    assert named["embedding"] == []


def test_candidate_groups_put_branch_at_encoder_lr_not_head_lr() -> None:
    model = V541Model(embedding_id=TEST_ID, embedding_init_seed=INIT_SEED)
    cfg = V541TrainerConfig(embedding_id=TEST_ID, init_seed=INIT_SEED)
    groups = param_groups_v541(model, cfg)
    assert [g["name"] for g in groups] == [
        "backbone", "process_head", "embedding"]
    by_name = {g["name"]: g for g in groups}
    assert by_name["backbone"]["lr"] == cfg.lr_encoder == 1.0e-4
    assert by_name["process_head"]["lr"] == cfg.lr_head == 3.0e-4
    assert by_name["embedding"]["lr"] == cfg.lr_encoder == 1.0e-4
    assert by_name["embedding"]["lr"] != by_name["process_head"]["lr"]
    # the branch really carries the three §7.1 anchors
    branch_names = {n for n, _p in model.named_parameters()
                    if n.startswith(("embedding.", "w_out.")) or n == "alpha"}
    assert {"alpha", "w_out.weight"} <= branch_names
    assert len(by_name["embedding"]["params"]) == len(branch_names)


def test_unclassified_parameter_is_a_typed_refusal() -> None:
    model = V541Model()
    model.stray = torch.nn.Parameter(torch.zeros(3))     # no group prefix
    with pytest.raises(V541TrainerError) as exc:
        param_groups_v541(model, V541TrainerConfig())
    assert exc.value.code == "PARAM_GROUP_UNCLASSIFIED"
    assert "stray" in str(exc.value)


def test_stage_helper_w0_freezes_backbone_only() -> None:
    model = V541Model(embedding_id=TEST_ID, embedding_init_seed=INIT_SEED)
    set_training_stage_v541(model, backbone_trainable=False)
    assert all(not p.requires_grad for p in model.backbone.parameters())
    assert all(p.requires_grad for p in model.process_head.parameters())
    assert all(p.requires_grad for p in model.embedding.parameters())
    assert model.alpha.requires_grad and model.w_out.weight.requires_grad
    set_training_stage_v541(model, backbone_trainable=True)
    assert all(p.requires_grad for p in model.parameters())


# ---------------------------------------------------------------------------
# E0 model parity + R3 strict warmstart
# ---------------------------------------------------------------------------


def test_e0_model_parity_with_legacy_build() -> None:
    legacy_cfg = tr.V53MPTrainerConfig(
        arm="C", seed=SEED, backbone_init="scratch",
        lambda_rank=V541_LAMBDA_RANK,
        lambda_q=V541_LAMBDA_Q, lambda_endpoint=V541_LAMBDA_ENDPOINT)
    set_seed(SEED)
    legacy = tr.load_model_v53mp(legacy_cfg, device="cpu")
    set_seed(SEED)
    v541 = load_model_v541(
        V541TrainerConfig(seed=SEED, backbone_init="scratch"),
        device="cpu")

    assert isinstance(v541, V541Model) and v541.is_e0
    assert not v541.has_branch
    assert set(v541.state_dict()) == set(legacy.state_dict())
    for key, tensor in legacy.state_dict().items():
        assert torch.equal(v541.state_dict()[key], tensor), key

    batch = forward_view_v3(make_v3_batch(n_graphs=2))
    v541.eval()
    legacy.eval()
    with torch.no_grad():
        out_a = v541(batch)
        out_b = legacy(batch)
    assert set(out_a) == set(out_b) == {
        "u", "z", "q", "endpoint_logit", "order_score"}
    for key in out_a:
        assert torch.equal(out_a[key], out_b[key]), key
    assert torch.equal(out_a["q"], out_a["order_score"])


def test_r3_strict_warmstart_matches_legacy(tmp_path) -> None:
    torch.manual_seed(5)
    ref = StarEGNNV3()
    ckpt = tmp_path / "r3.pt"
    torch.save({"model": ref.state_dict()}, str(ckpt))

    set_seed(SEED)
    v541 = load_model_v541(
        V541TrainerConfig(seed=SEED, backbone_init="r3"), r3_ckpt=ckpt)
    set_seed(SEED)
    legacy = tr.load_model_v53mp(
        tr.V53MPTrainerConfig(arm="C", seed=SEED, backbone_init="r3",
                              lambda_rank=V541_LAMBDA_RANK,
                              lambda_q=V541_LAMBDA_Q,
                              lambda_endpoint=V541_LAMBDA_ENDPOINT),
        r3_ckpt=ckpt, device="cpu")
    for key, tensor in ref.state_dict().items():
        assert torch.equal(v541.backbone.state_dict()[key], tensor), key
    assert set(v541.state_dict()) == set(legacy.state_dict())
    for key, tensor in legacy.state_dict().items():
        assert torch.equal(v541.state_dict()[key], tensor), key

    with pytest.raises(V541TrainerError) as exc:
        load_model_v541(V541TrainerConfig(seed=SEED, backbone_init="r3"))
    assert exc.value.code == "R3_CHECKPOINT_MISSING"


def test_branch_construction_keeps_legacy_rng_sequence() -> None:
    """Branch init runs under fork_rng — the legacy stream is untouched."""
    set_seed(SEED)
    e0_a = V541Model()
    set_seed(SEED)
    cand = V541Model(embedding_id=TEST_ID, embedding_init_seed=INIT_SEED)
    set_seed(SEED)
    e0_b = V541Model()
    for key, tensor in e0_a.state_dict().items():
        assert torch.equal(e0_b.state_dict()[key], tensor), key
    # ... and the candidate shares the SAME legacy weights as E0
    for key, tensor in e0_a.state_dict().items():
        assert torch.equal(cand.state_dict()[key], tensor), key
    # same independent seed ⇒ identical branch weights
    set_seed(SEED)
    cand2 = V541Model(embedding_id=TEST_ID, embedding_init_seed=INIT_SEED)
    assert torch.equal(cand.w_out.weight, cand2.w_out.weight)
    assert torch.equal(cand.alpha, cand2.alpha)


# ---------------------------------------------------------------------------
# W0 / W1 stage, gradient reachability (§7.1 / §7.5)
# ---------------------------------------------------------------------------


def test_w0_w1_stage_reachability_and_grads(env, tmp_path) -> None:
    out = tmp_path / "stage"
    cfg = _cfg(freeze_encoder_updates=2, joint_updates=4, smoke_updates=6,
               grad_monitor_every=2, val_every=3)
    metrics = train_run_v541(env.ds, env.labels, env.endpoint, cfg, out,
                             dev_positions=env.dev_pos, resume=False)
    rows = _train_rows(out)
    events = _read_log(out)

    # W0: backbone frozen, but the loss still has a grad path — the
    # forward was NEVER wrapped in no_grad() (§7.5).
    assert not any(e.get("event") == "skipped_no_grad" for e in events)
    w0 = rows[1]
    assert w0["phase"] == "w0" and w0["encoder_trainable"] is False
    assert w0["grad_monitor"]["encoder_frozen"] is True
    assert w0["param_group_grads"]["norms"]["backbone"] is None
    assert w0["param_group_grads"]["reasons"]["backbone"] == \
        "group_empty_or_frozen"
    # process head IS learning in W0
    assert w0["param_group_grads"]["norms"]["process_head"] > 0.0

    # W0 -> W1 switch: core enters the optimizer AND receives gradients
    switches = [e for e in events if e.get("event") == "stage_switch"]
    assert len(switches) == 1
    sw = switches[0]
    assert sw["update"] == cfg.effective_freeze_updates + 1 == 3
    assert sw["phase"] == "w1"
    assert sw["core_requires_grad"] is True
    assert sw["n_core_params_with_grad"] == sw["n_core_params"] > 0
    assert sw["core_reachable"] is True
    assert sw["core_grad_max_abs"] > 0.0

    # W1: backbone trains and its grad norm is a real number
    w1 = rows[4]
    assert w1["phase"] == "w1" and w1["encoder_trainable"] is True
    assert w1["grad_monitor"]["encoder_frozen"] is False
    assert w1["param_group_grads"]["norms"]["backbone"] > 0.0
    assert metrics["schedule"]["w0_updates"] == 2
    assert metrics["schedule"]["w1_updates"] == 4
    assert metrics["schedule"]["updates_run"] == 6


def test_w0_branch_keeps_graph_alive(env, tmp_path) -> None:
    """Frozen core still differentiates the branch (no no_grad forward)."""
    out = tmp_path / "branchgraph"
    cfg = _cfg(embedding_id=TEST_ID, init_seed=INIT_SEED,
               freeze_encoder_updates=4, joint_updates=1, smoke_updates=4,
               grad_monitor_every=2, val_every=2)
    train_run_v541(env.ds, env.labels, env.endpoint, cfg, out,
                   dev_positions=env.dev_pos, resume=False)
    rows = _train_rows(out)
    assert rows[1]["phase"] == "w0"
    assert rows[1]["encoder_trainable"] is False
    assert rows[1]["branch"]["present"] is True
    assert rows[1]["param_group_grads"]["norms"]["embedding"] is not None
    assert rows[1]["grad_monitor"]["encoder_frozen"] is True


# ---------------------------------------------------------------------------
# stratified sampler + schedule hash (§7.2)
# ---------------------------------------------------------------------------


def _reference_index(env) -> StrataIndex:
    """Rebuild the trainer's inner-train index from the frozen inputs."""
    samples = env.ds.samples
    dev_curves = {str(samples[p].curve_id) for p in env.dev_pos}
    pool = [i for i in range(len(samples))
            if str(samples[i].curve_id) not in dev_curves]
    pool_curves = sorted({str(samples[i].curve_id) for i in pool})
    train_pos, _iv = tr.v53mp_pool_split(
        samples, pool_curve_ids=pool_curves, seed=SEED,
        inner_val_fraction=0.2)
    return StrataIndex.build(samples, train_pos, dataset=env.ds)


def test_batches_are_the_frozen_four_strata_draw(env, tmp_path) -> None:
    out = tmp_path / "strata"
    cfg = _cfg(freeze_encoder_updates=1, joint_updates=3, smoke_updates=4,
               grad_monitor_every=1, val_every=4)
    metrics = train_run_v541(env.ds, env.labels, env.endpoint, cfg, out,
                             dev_positions=env.dev_pos, resume=False)
    index = _reference_index(env)
    samples = env.ds.samples
    rows = _train_rows(out)
    assert sorted(rows) == [1, 2, 3, 4]

    for update, row in sorted(rows.items()):
        positions = row["batch_positions"]
        assert len(positions) == BATCH_CURVES_V541 * FRAMES_PER_CURVE_V541
        # EXACTLY the pure (seed, update, salt=0) stratified draw
        expected = curve_batch_positions_v541(
            index, rng=batch_rng_v541(SEED, update),
            batch_curves=BATCH_CURVES_V541)
        assert positions == expected
        # 8 curve groups x 4 slots: four strata, direction balance 2+2
        for g in range(BATCH_CURVES_V541):
            chunk = positions[4 * g:4 * g + 4]
            assert len(chunk) == 4
            info = [strata_of_position_v541(samples, index, p)
                    for p in chunk]
            assert len({cid for cid, _d, _s in info}) == 1
            assert [_d for _c, _d, _s in info] == [
                "forward", "forward", "reverse", "reverse"]
            fwd = {_s for _c, _d, _s in info[:2]}
            rev = {_s for _c, _d, _s in info[2:]}
            assert fwd.isdisjoint(rev)
            assert fwd | rev == {"start", "pre_TS", "post_TS", "endpoint"}

    assert metrics["sampler"]["sampler_id"] == "v541_curve_stratified_v1"
    assert metrics["sampler"]["spec_sha256"] == sampler_spec_sha256()
    assert metrics["sampler"]["pool_sha256"] == index.pool_sha256
    assert metrics["sampler"]["schedule_updates"] == 4
    assert metrics["schedule"]["batch_curves"] == 8
    assert metrics["schedule"]["frames_per_curve"] == 4
    # dev curves never enter the sampler pool
    dev_curves = {str(samples[p].curve_id) for p in env.dev_pos}
    assert not (set(index.curves) & dev_curves)


# ---------------------------------------------------------------------------
# dev firewall: ZERO dev feature / label / eval calls
# ---------------------------------------------------------------------------


class _PoisonMap:
    """``by_sample`` view that raises the moment a dev id is read."""

    def __init__(self, inner, bad) -> None:
        self._inner = inner
        self._bad = set(bad)

    def _check(self, key) -> None:
        if key in self._bad:
            raise AssertionError(
                "DEV LABEL READ: {!r} — dev labels must never be touched"
                .format(key))

    def __contains__(self, key):
        self._check(key)
        return key in self._inner

    def __getitem__(self, key):
        self._check(key)
        return self._inner[key]

    def __len__(self):
        return len(self._inner)

    def __iter__(self):
        return iter(self._inner)


class _PoisonTable:
    def __init__(self, inner, bad) -> None:
        object.__setattr__(self, "_inner", inner)
        object.__setattr__(self, "by_sample", _PoisonMap(inner.by_sample, bad))

    def __getattr__(self, name):
        return getattr(self._inner, name)


class _PoisonDataset:
    """``records()`` refuses ANY dev position; structural metadata stays."""

    def __init__(self, inner, dev_pos) -> None:
        object.__setattr__(self, "_inner", inner)
        object.__setattr__(self, "_dev", set(int(p) for p in dev_pos))

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def records(self, indices=None):
        if indices is None:
            raise AssertionError(
                "DEV RECORD MIRROR: full materialisation would include dev")
        bad = sorted(self._dev & {int(i) for i in indices})
        if bad:
            raise AssertionError(
                "DEV RECORD READ: positions {}".format(bad[:5]))
        return self._inner.records(indices)


def test_dev_positions_only_exclude_dev_curves(env, tmp_path) -> None:
    samples = env.ds.samples
    dev_ids = {str(samples[p].sample_id) for p in env.dev_pos}
    assert dev_ids
    dataset = _PoisonDataset(env.ds, env.dev_pos)
    labels = _PoisonTable(env.labels, dev_ids)
    endpoint = _PoisonTable(env.endpoint, dev_ids)

    out = tmp_path / "devfirewall"
    before = {p for p in tmp_path.rglob("*") if p.is_file()}
    metrics = train_run_v541(
        dataset, labels, endpoint,
        _cfg(freeze_encoder_updates=1, joint_updates=3, smoke_updates=4,
             grad_monitor_every=2, val_every=2),
        out, dev_positions=env.dev_pos, resume=False)
    after = {p for p in tmp_path.rglob("*") if p.is_file()}
    # the run wrote ONLY inside its own out_dir (no ledger, no dev pack)
    assert (after - before) <= set(out.rglob("*"))

    assert metrics["evaluation_scope"] == "inner_val"
    assert metrics["dev_eval_done"] is False
    assert metrics["dev_positions_used_for"] == "curve_exclusion_only"
    assert metrics["confirm_read"] is False
    assert metrics["n_eval_samples"] == metrics["n_inner_val_samples"] > 0
    assert metrics["n_eval_samples"] <= 8192
    dev_curves = {str(samples[p].curve_id) for p in env.dev_pos}
    assert metrics["n_dev_curves_excluded"] == len(dev_curves)
    assert set(metrics["dev_curve_ids"]) == dev_curves
    assert set(metrics["dev_curve_ids"]).isdisjoint(metrics["train_curves"])
    assert set(metrics["dev_curve_ids"]).isdisjoint(
        metrics["inner_val_curves"])
    # the MOVE guard ran against the R3 reference on INNER-VAL positions
    guard = metrics["move_guard_vs_r3"]
    assert guard["positions_scope"] == "inner_val"
    assert "dev is never evaluated" in guard["baseline"]


# ---------------------------------------------------------------------------
# monitoring: toggling, alpha learning, null+reason (§7.5)
# ---------------------------------------------------------------------------


def test_diagnostics_toggle_only_on_monitor_steps(env, tmp_path) -> None:
    out = tmp_path / "toggle"
    cfg = _cfg(embedding_id=TEST_ID, init_seed=INIT_SEED,
               freeze_encoder_updates=1, joint_updates=5, smoke_updates=6,
               grad_monitor_every=3, val_every=6)
    train_run_v541(env.ds, env.labels, env.endpoint, cfg, out,
                   dev_positions=env.dev_pos, resume=False)
    events = _read_log(out)
    flags = {e["update"]: e for e in events if e.get("event") == "diag_flag"}
    rows = _train_rows(out)
    # monitor steps = 1, 3, 6 (every 3, plus first and last)
    assert sorted(rows) == [1, 3, 6]
    assert sorted(flags) == [2, 4, 5]
    for row in rows.values():
        assert row["diagnostics_enabled"] is True
        # diagnostics actually produced by the branch forward
        assert row["diagnostics"]["alpha"] is not None
        assert row["diagnostics"]["n_edges"] > 0
    for flag in flags.values():
        assert flag["diagnostics_enabled"] is False
        assert flag["event"] == "diag_flag"


def test_alpha_learning_evidence_is_grad_based(env, tmp_path) -> None:
    out = tmp_path / "alpha"
    cfg = _cfg(embedding_id=TEST_ID, init_seed=INIT_SEED,
               freeze_encoder_updates=1, joint_updates=5, smoke_updates=6,
               grad_monitor_every=2, val_every=6)
    train_run_v541(env.ds, env.labels, env.endpoint, cfg, out,
                   dev_positions=env.dev_pos, resume=False)
    rows = _train_rows(out)
    first = rows[1]
    # cold start: alpha gets a FINITE NONZERO gradient, W/f do not
    assert first["alpha_before_step"] == 0.0
    assert first["alpha_after_step"] is not None
    assert np.isfinite(first["alpha_after_step"])
    assert first["alpha_after_step"] != 0.0
    grads = first["branch"]["grad_norms"]
    assert first["branch"]["alpha_grad"] not in (None, 0.0)
    assert np.isfinite(first["branch"]["alpha_grad"])
    assert grads["embedding"] == 0.0 and grads["w_out"] == 0.0
    assert first["branch"]["all_zero"] is False
    # first monitor has no previous snapshot -> null, not a fake 0.0
    assert first["branch_param_delta_norm"] is None

    # after alpha left zero the WHOLE branch receives task gradients
    later = rows[4]
    assert later["alpha_before_step"] != 0.0
    assert later["branch"]["grad_norms"]["embedding"] > 0.0
    assert later["branch"]["grad_norms"]["w_out"] > 0.0
    assert later["branch"]["finite"] is True
    assert later["branch_param_delta_norm"] is not None
    assert later["branch_param_delta_norm"] > 0.0
    # evidence is the GRADIENT, not AdamW decay: grads are nonzero
    assert later["branch"]["all_zero"] is False
    # residual strength reported at BOTH scales with a fixed-eps ratio
    assert later["rms_h_legacy"] > 0.0
    assert later["rms_alpha_W_f"] > 0.0
    assert later["rms_ratio"] > 0.0


def test_unsupervised_term_reports_null_plus_reason(env, tmp_path) -> None:
    out = tmp_path / "null"
    cfg = _cfg(order_min_gap=10.0, freeze_encoder_updates=0,
               joint_updates=1, smoke_updates=2, grad_monitor_every=1,
               val_every=2)
    train_run_v541(env.ds, env.labels, env.endpoint, cfg, out,
                   dev_positions=env.dev_pos, resume=False)
    rows = _train_rows(out)
    row = rows[1]
    assert row["n_rank_pairs"] == 0
    gm = row["grad_monitor"]
    assert gm["grad_norms"]["rank"] is None          # NOT a fake 0.0
    assert gm["null_reasons"]["rank"] == "no_valid_ordering_pairs"
    assert not any((k.startswith("rank_") or k.endswith("_rank"))
                   for k, v in gm["cosines"].items() if v is not None)
    # supervised terms are unaffected
    assert row["n_q_rows"] > 0 and gm["grad_norms"]["q"] is not None
    assert row["n_endpoint_rows"] > 0
    assert gm["grad_norms"]["endpoint"] is not None


# ---------------------------------------------------------------------------
# diagnostics wiring: model flag -> embedding flag -> merged counts (§7.5)
# ---------------------------------------------------------------------------


def test_set_diagnostics_enabled_propagates_to_embedding() -> None:
    model = V541Model(embedding_id=COUNT_ID, embedding_init_seed=INIT_SEED)
    batch = forward_view_v3(make_v3_batch(n_graphs=2))
    model.eval()

    set_diagnostics_enabled_v541(model, False)
    assert model.diagnostics_enabled is False
    assert model.embedding.diagnostics_enabled is False
    with torch.no_grad():
        model(batch)
    # flag off => the candidate must NOT pay the count sync (counts == {})
    assert model.embedding.last_diagnostics == {}
    assert embedding_diagnostics(model) == ({}, "empty")

    set_diagnostics_enabled_v541(model, True)
    assert model.diagnostics_enabled is True
    assert model.embedding.diagnostics_enabled is True
    with torch.no_grad():
        model(batch)
    diag, status = embedding_diagnostics(model)
    assert status == "ok"
    assert diag["n_triplets"] == 17.0
    # E0 has no branch: the helper must be a no-op, never raise
    e0 = V541Model()
    set_diagnostics_enabled_v541(e0, False)
    assert e0.diagnostics_enabled is False
    assert embedding_diagnostics(e0) == ({}, "absent_no_branch")


@pytest.mark.parametrize("model_kind,expect_status,expect_reason", [
    ("e0", "absent_no_branch", "e0_no_branch"),
    ("plain", "attr_absent", "embedding_diagnostics_unavailable"),
    ("empty", "empty", "embedding_diagnostics_empty"),
])
def test_edge_triplet_block_null_plus_reason_per_status(
        model_kind, expect_status, expect_reason) -> None:
    """No provider field ⇒ ``null + reason`` — never a fabricated 0."""
    if model_kind == "e0":
        model = V541Model()
    elif model_kind == "plain":
        model = V541Model(embedding_id=TEST_ID,
                          embedding_init_seed=INIT_SEED)
    else:
        model = V541Model(embedding_id=EMPTY_ID,
                          embedding_init_seed=INIT_SEED)
    emb_diag, status = embedding_diagnostics(model)
    assert status == expect_status
    block = _edge_triplet_block({"n_nodes": 8.0, "n_edges": 12.0},
                                emb_diag, status)
    assert block["n_nodes"] == 8.0 and block["n_edges"] == 12.0
    for key in ("n_triplets", "zero_neighbor_frac", "zero_angle_frac"):
        assert block[key] is None
        assert block["reasons"][key] == expect_reason
    assert block["embedding_diagnostics_status"] == expect_status


def test_embedding_diagnostics_merged_into_monitor_log(env, tmp_path) -> None:
    out = tmp_path / "embdiag"
    cfg = _cfg(embedding_id=COUNT_ID, init_seed=INIT_SEED,
               freeze_encoder_updates=1, joint_updates=3, smoke_updates=4,
               grad_monitor_every=2, val_every=4)
    train_run_v541(env.ds, env.labels, env.endpoint, cfg, out,
                   dev_positions=env.dev_pos, resume=False)
    events = _read_log(out)
    rows = _train_rows(out)
    flags = {e["update"]: e for e in events if e.get("event") == "diag_flag"}
    assert sorted(rows) == [1, 2, 4]
    assert sorted(flags) == [3]

    for row in rows.values():
        # the toggle reached BOTH the model and the candidate
        assert row["diagnostics_enabled"] is True
        assert row["embedding_diagnostics_enabled"] is True
        assert row["embedding_diagnostics_status"] == "ok"
        assert row["embedding_diagnostics"]["n_triplets"] == 17.0
        # ... and the counts are merged into the §7.5 block
        block = row["edge_triplet_zero_neighbor"]
        assert block["n_edges"] > 0.0          # model-level diagnostics
        assert block["n_triplets"] == 17.0     # WP3 provider counts
        assert block["zero_neighbor_frac"] == 0.125
        assert block["zero_angle_frac"] == 0.0625
        assert block["reasons"] == {}
    for flag in flags.values():
        assert flag["diagnostics_enabled"] is False
        assert flag["embedding_diagnostics_enabled"] is False


def test_embedding_without_diagnostics_stays_null_in_log(env, tmp_path) -> None:
    """A candidate that exposes no diagnostics ⇒ null + reason, no fake 0."""
    out = tmp_path / "nodiag"
    cfg = _cfg(embedding_id=TEST_ID, init_seed=INIT_SEED,
               freeze_encoder_updates=1, joint_updates=1, smoke_updates=2,
               grad_monitor_every=1, val_every=2)
    train_run_v541(env.ds, env.labels, env.endpoint, cfg, out,
                   dev_positions=env.dev_pos, resume=False)
    row = _train_rows(out)[1]
    assert row["embedding_diagnostics_enabled"] is None
    assert row["embedding_diagnostics_status"] == "attr_absent"
    block = row["edge_triplet_zero_neighbor"]
    assert block["n_edges"] > 0.0
    assert block["n_triplets"] is None
    assert block["reasons"]["n_triplets"] == "embedding_diagnostics_unavailable"
    assert block["reasons"]["zero_neighbor_frac"] == \
        "embedding_diagnostics_unavailable"
    assert block["reasons"]["zero_angle_frac"] == \
        "embedding_diagnostics_unavailable"


# ---------------------------------------------------------------------------
# profile-only window (§8.3)
# ---------------------------------------------------------------------------


def test_profile_only_window_is_marked_and_never_formal(env, tmp_path) -> None:
    out = tmp_path / "profile"
    cfg = _cfg(profile_only=True, profile_w0_updates=2,
               profile_w1_updates=3, smoke_updates=None,
               freeze_encoder_updates=400, joint_updates=37500,
               grad_monitor_every=2, val_every=100)
    metrics = train_run_v541(env.ds, env.labels, env.endpoint, cfg, out,
                             dev_positions=env.dev_pos, resume=False)
    assert metrics["profile_only"] is True
    assert metrics["is_formal_run"] is False
    assert metrics["run_outcome"] is None
    assert "profile_only" in metrics["run_outcome_reason"]
    assert metrics["profile_window"] == {
        "profile_w0_updates": 2, "profile_w1_updates": 3, "total_updates": 5}
    assert metrics["formal_schedule_untouched"]["total_updates"] == 37900
    assert metrics["formal_schedule_untouched"]["w0_updates"] == 400
    assert metrics["formal_schedule_untouched"]["w1_updates"] == 37500
    assert set(metrics["timing"]) == {"profile_w0", "profile_w1", "startup"}
    assert metrics["timing"]["profile_w0"]["updates"] == 2
    assert metrics["timing"]["profile_w1"]["updates"] == 3
    assert metrics["dev_eval_done"] is False
    # a profile is NOT a formal run: no checkpoints, no manifest, no τ
    assert not list(out.glob("checkpoint*.pt"))
    assert not (out / "manifest.jsonl").is_file()
    assert (out / "profile.json").is_file()
    assert (out / "metrics.json").is_file()
    assert json.loads((out / "metrics.json").read_text())["profile_only"]


def test_profile_window_refuses_missing_phase_length() -> None:
    with pytest.raises(V541TrainerError) as exc:
        V541TrainerConfig(profile_only=True, profile_w0_updates=2)
    assert exc.value.code == "PROFILE_WINDOW"


# ---------------------------------------------------------------------------
# run outcome vocabulary + legacy ledger untouched (§7.4)
# ---------------------------------------------------------------------------


def test_run_outcome_vocabulary_is_the_frozen_six() -> None:
    assert tuple(RUN_OUTCOMES_V541) == (
        "done", "invalid_run", "scientific_failure", "resource_failure",
        "budget_censored", "gate_skipped")
    assert len(set(RUN_OUTCOMES_V541)) == 6


def test_run_outcome_for_exception_mapping() -> None:
    assert run_outcome_for_exception(
        V541TrainerError("X", "y")) == "invalid_run"
    assert run_outcome_for_exception(RuntimeError("boom")) == "invalid_run"
    assert run_outcome_for_exception(MemoryError("oom")) == "resource_failure"
    assert run_outcome_for_exception(
        V541RunBudgetError("over budget")) == "budget_censored"


def test_short_run_is_not_labelled_formal_but_still_done(
        env, tmp_path) -> None:
    out = tmp_path / "outcome"
    metrics = train_run_v541(
        env.ds, env.labels, env.endpoint,
        _cfg(freeze_encoder_updates=1, joint_updates=2, smoke_updates=3,
             grad_monitor_every=1, val_every=3),
        out, dev_positions=env.dev_pos, resume=False)
    assert not any((tmp_path / n).exists() for n in
                   ("run_ledger", "run_ledger.jsonl", "ledger.jsonl"))
    assert metrics["run_outcome"] == "done"
    assert tuple(metrics["run_outcome_vocabulary"]) == RUN_OUTCOMES_V541
    # a short fixture window is a REAL run but NEVER a formal attempt
    assert metrics["is_formal_run"] is False
    assert metrics["schedule"]["formal_schedule"] is False
    assert metrics["profile_only"] is False
    assert metrics["ledger_status_written"] is False
    assert metrics["schema"] == V541_TRAIN_SCHEMA
    assert metrics["objective"] == "unified_move_process"
    # the internal loss arm never leaks as the run's objective
    assert "arm_internal_loss_only" not in metrics
    assert "arm" not in json.dumps(metrics["schedule"]).lower()
    assert metrics["schedule"]["objective_mapping"] == \
        "single unified MOVE+PROCESS loss"
    assert "arm" not in json.dumps(_read_log(out)).lower()
    assert "arm" not in (out / "metrics.json").read_text().lower()


def test_formal_schedule_predicate() -> None:
    # the frozen preregistered schedule
    assert is_formal_schedule(V541TrainerConfig()) is True
    assert V541TrainerConfig().total_updates == V541_FORMAL_TOTAL_UPDATES
    # ... and NOTHING else
    assert is_formal_schedule(_cfg(smoke_updates=6)) is False
    assert is_formal_schedule(_cfg(smoke_updates=200)) is False
    assert is_formal_schedule(
        _cfg(smoke_updates=None, freeze_encoder_updates=400,
             joint_updates=100)) is False
    # smoke_updates=37900 alone must not smuggle a run in as formal
    assert is_formal_schedule(
        _cfg(smoke_updates=37900, freeze_encoder_updates=1,
             joint_updates=1)) is False
    # an explicit runner flag wins and is recorded in the identity
    assert is_formal_schedule(
        _cfg(smoke_updates=200, formal_run=True)) is True
    assert is_formal_schedule(
        V541TrainerConfig(formal_run=False)) is False
    assert config_identity_sha256(
        _cfg(smoke_updates=200, formal_run=True)) != \
        config_identity_sha256(_cfg(smoke_updates=200))


def test_metrics_carry_required_identity_and_resource_fields(
        env, tmp_path) -> None:
    out = tmp_path / "fields"
    cfg = _cfg(freeze_encoder_updates=1, joint_updates=2, smoke_updates=3,
               grad_monitor_every=1, val_every=3)
    metrics = train_run_v541(
        env.ds, env.labels, env.endpoint, cfg, out,
        dev_positions=env.dev_pos, resume=False,
        provenance={"tier": "50k", "data_manifest_sha256": "d" * 64,
                    "split_manifest_sha256": "s" * 64,
                    "shard_path": "/pes2ts_data/v53mp_50k_data"})
    ident = metrics["identity"]
    for key in ("common_protocol_sha256", "embedding_spec_sha256",
                "feature_recipe_sha256", "config_sha256",
                "sampler_spec_sha256", "sampler_pool_sha256"):
        assert key in ident
    assert ident["config_sha256"] == config_identity_sha256(cfg)
    assert ident["tier"] == "50k"
    assert metrics["config_full_sha256"] == config_full_sha256(cfg)
    assert metrics["git"]["commit"] is None or len(
        metrics["git"]["commit"]) == 40
    assert metrics["env"]["torch"] == torch.__version__
    assert metrics["resources"]["host_peak_rss_bytes"] > 0
    assert metrics["resources"]["gpu_max_memory_allocated_bytes"] == 0
    assert metrics["timing"]["startup_wall_s"] >= 0.0
    assert metrics["cache"]["capacity"] == 8192
    assert metrics["param_groups"][0]["name"] == "backbone"
    assert metrics["checkpoint_best_sha256"]
    assert Path(metrics["manifest_path"]).is_file()
    assert metrics["provenance"]["data_manifest_sha256"] == "d" * 64
    assert metrics["provenance"]["shard_path"] == \
        "/pes2ts_data/v53mp_50k_data"
