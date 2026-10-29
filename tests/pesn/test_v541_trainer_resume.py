"""V5.4.1 WP4b — checkpoint / resume / manifest tests (``v541_train.py``).

Route-E resume guarantees of plan §7.3:

* **20+20 vs 40 bitwise** — a run stopped after 20 updates and resumed to
  40 is bit-for-bit the uninterrupted 40-update run: model tensors,
  optimizer state, batch schedule chain, inner-val losses per update,
  best/selected update AND the checkpoint ``content_digest_sha256``;
* **typed identity refusal** — resume is refused (with the offending key
  names on ``exc.fields``) when any SHA (common / embedding / feature /
  config / sampler pool), the embedding id, the independent init seed,
  the run seed, the scale tier or any λ differs;
* **payload completeness** — model / optimizer / update / best / config /
  provenance / τ, all RNG states, strata pool+spec SHA,
  ``BatchScheduleHasher`` state, data+shard position, the four identity
  SHAs, git + env and a self-verifying content digest;
* **manifest APPEND** — every leg only ever appends; earlier lines are
  preserved byte-for-byte;
* **schedule-hash continuity** — the chain equals an independent
  recomputation of the pure ``(seed, update, 0)`` draws.

Kill/resume inside a live process is deliberately left to the real
200-update smoke (§7.3 test 2) — this file exercises the trainer's own
two-invocation resume path on a small fixture.
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

from pes2ts_core.pesn.models.v541 import (  # noqa: E402
    HIDDEN_V541,
    NodeDeltaEmbedding,
    register_embedding,
    registry_restore,
    registry_snapshot,
)
from pes2ts_core.pesn.offline.training import key_atom_v53mp_train_v3 as tr  # noqa: E402
from pes2ts_core.pesn.offline.training.v541_identity import (  # noqa: E402
    content_digest_of,
    read_manifest,
)
from pes2ts_core.pesn.offline.training.v541_strata import (  # noqa: E402
    BatchScheduleHasher,
    StrataIndex,
    batch_rng_v541,
    curve_batch_positions_v541,
    sampler_spec_sha256,
)
from pes2ts_core.pesn.offline.training.v541_train import (  # noqa: E402
    V541_TRAIN_SCHEMA,
    V541TrainerConfig,
    V541TrainerError,
    train_run_v541,
)
from tests.pesn.test_r3_v53mp_trainer import (  # noqa: E402
    DEFAULT_FOLDS,
    _build_env,
)

SEED = 17
TEST_ID = "TEST_TRAINER_MLP"
COMMON = dict(seed=SEED, backbone_init="scratch",
              freeze_encoder_updates=4, joint_updates=36,
              val_every=5, grad_monitor_every=5, device="cpu",
              inner_val_max_samples=None)


class _DummyEmbedding(NodeDeltaEmbedding):
    def __init__(self, hidden: int = HIDDEN_V541) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(hidden, hidden), torch.nn.SiLU(),
            torch.nn.Linear(hidden, hidden))

    def forward(self, h_legacy, batch):
        return self.net(h_legacy)


@pytest.fixture(autouse=True)
def _registry():
    snap = registry_snapshot()
    register_embedding(TEST_ID, lambda: _DummyEmbedding())
    yield
    registry_restore(snap)


@pytest.fixture()
def env(tmp_path):
    return _build_env(tmp_path, DEFAULT_FOLDS)


def _cfg(**over) -> V541TrainerConfig:
    base = dict(COMMON)
    base.update(over)
    return V541TrainerConfig(**base)


def _run(env, cfg, out, *, resume: bool, **kw) -> dict:
    return train_run_v541(env.ds, env.labels, env.endpoint, cfg, out,
                          dev_positions=env.dev_pos, resume=resume, **kw)


def _read_log(out: Path) -> list:
    path = out / "train_log.jsonl"
    if not path.is_file():
        return []
    return [json.loads(ln) for ln in
            path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def _train_rows(out: Path) -> dict:
    return {e["update"]: e for e in _read_log(out)
            if e.get("event") == "train"}


def _inner_val_by_update(out: Path) -> dict:
    return {e["update"]: e["inner_val_loss"] for e in _read_log(out)
            if e.get("event") == "inner_val"}


def _strip_timing(row: dict) -> dict:
    return {k: v for k, v in row.items()
            if k not in ("elapsed_s", "phase_timing", "resources")}


def _load(path: Path) -> dict:
    return torch.load(str(path), map_location="cpu", weights_only=False)


# ---------------------------------------------------------------------------
# §7.3 test 1: 20+20 vs 40 bitwise
# ---------------------------------------------------------------------------


def test_20_plus_20_equals_40_bitwise(env, tmp_path) -> None:
    resumed_dir = tmp_path / "resumed"
    fresh_dir = tmp_path / "fresh"

    leg1 = _run(env, _cfg(smoke_updates=20), resumed_dir, resume=False)
    assert leg1["selected_update"] in range(1, 21)
    leg2 = _run(env, _cfg(smoke_updates=40), resumed_dir, resume=True)
    fresh = _run(env, _cfg(smoke_updates=40), fresh_dir, resume=False)

    # --- model weights bitwise ------------------------------------------
    a = _load(resumed_dir / "checkpoint_last.pt")
    b = _load(fresh_dir / "checkpoint_last.pt")
    assert a["update"] == b["update"] == 40
    assert set(a["model"]) == set(b["model"])
    for key in a["model"]:
        assert torch.equal(a["model"][key], b["model"][key]), key

    # --- optimizer state bitwise ----------------------------------------
    def _opt_equal(x, y, path=""):
        if isinstance(x, dict):
            assert set(x) == set(y), path
            for k in x:
                _opt_equal(x[k], y[k], path + "/" + str(k))
        elif isinstance(x, (list, tuple)):
            assert len(x) == len(y), path
            for i, (u, v) in enumerate(zip(x, y)):
                _opt_equal(u, v, path + "[%d]" % i)
        elif isinstance(x, torch.Tensor):
            assert torch.equal(x, y), path
        elif hasattr(x, "shape") and hasattr(x, "dtype"):
            assert np.array_equal(x, y), path
        else:
            assert x == y, path

    _opt_equal(a["optimizer"], b["optimizer"], "optimizer")

    # --- batch schedule chain bitwise ------------------------------------
    assert a["schedule"] == b["schedule"]
    assert a["content_digest_sha256"] == b["content_digest_sha256"]

    # --- best checkpoint too ---------------------------------------------
    ba = _load(resumed_dir / "checkpoint_best.pt")
    bb = _load(fresh_dir / "checkpoint_best.pt")
    assert ba["best_val"] == bb["best_val"]
    assert ba["best_update"] == bb["best_update"]
    assert ba["content_digest_sha256"] == bb["content_digest_sha256"]
    for key in ba["model"]:
        assert torch.equal(ba["model"][key], bb["model"][key]), key

    # --- metrics / per-update indicators ---------------------------------
    assert leg2["best_inner_val_loss"] == fresh["best_inner_val_loss"]
    assert leg2["selected_update"] == fresh["selected_update"]
    assert leg2["best_update"] == fresh["best_update"]
    assert leg2["last_train_loss"] == fresh["last_train_loss"]
    assert leg2["sampler"]["batch_schedule_sha256"] == \
        fresh["sampler"]["batch_schedule_sha256"]
    assert leg2["tau_end_calibration"] == fresh["tau_end_calibration"]

    val_r = _inner_val_by_update(resumed_dir)
    val_f = _inner_val_by_update(fresh_dir)
    assert set(val_r) == set(val_f) == set(range(5, 41, 5)) | {40}
    assert val_r == val_f

    # --- train_log rows are identical apart from wall-clock fields -------
    rows_r = _train_rows(resumed_dir)
    rows_f = _train_rows(fresh_dir)
    assert set(rows_r) == set(rows_f)
    for update, row in rows_f.items():
        assert _strip_timing(rows_r[update]) == _strip_timing(row), update
    # the resumed dir additionally carries leg-1's rows (they are the same
    # rows the fresh run produced for updates 1..20 — asserted above)
    assert min(rows_r) == 1 and max(rows_r) == 40


# ---------------------------------------------------------------------------
# resume identity refusal (§7.3)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("over,prov_over,field", [
    (dict(seed=29), {}, "seed"),
    (dict(lambda_q=0.4), {}, "lambda_q"),
    (dict(lambda_rank=0.3), {}, "lambda_rank"),
    (dict(lambda_endpoint=0.01), {}, "lambda_endpoint"),
    (dict(common_protocol_sha256="c" * 64), {}, "common_protocol_sha256"),
    (dict(embedding_spec_sha256="e" * 64), {}, "embedding_spec_sha256"),
    (dict(feature_recipe_sha256="f" * 64), {}, "feature_recipe_sha256"),
    (dict(order_min_gap=0.9), {}, "config_sha256"),
    (dict(inner_val_fraction=0.33), {}, "config_sha256"),
    (dict(lr_encoder=2.0e-4), {}, "config_sha256"),
    (dict(grad_clip=0.5), {}, "config_sha256"),
])
def test_resume_refuses_identity_mismatch(env, tmp_path, over, prov_over,
                                          field) -> None:
    out = tmp_path / "dir"
    _run(env, _cfg(smoke_updates=4), out, resume=False)
    cfg = _cfg(smoke_updates=8, **over)
    with pytest.raises(V541TrainerError) as exc:
        _run(env, cfg, out, resume=True, provenance=prov_over or None)
    assert exc.value.code == "RESUME_IDENTITY_MISMATCH"
    assert field in exc.value.fields


def test_resume_refuses_tier_and_embedding_mismatch(env, tmp_path) -> None:
    out = tmp_path / "dir"
    _run(env, _cfg(smoke_updates=4), out, resume=False,
         provenance={"tier": "50k"})
    # different scale tier
    with pytest.raises(V541TrainerError) as exc:
        _run(env, _cfg(smoke_updates=8), out, resume=True,
             provenance={"tier": "20k"})
    assert exc.value.code == "RESUME_IDENTITY_MISMATCH"
    assert "tier" in exc.value.fields
    # different embedding id (candidate needs its own independent seed)
    with pytest.raises(V541TrainerError) as exc:
        _run(env, _cfg(smoke_updates=8, embedding_id="TEST_TRAINER_MLP",
                       init_seed=777), out, resume=True,
             provenance={"tier": "50k"})
    assert exc.value.code == "RESUME_IDENTITY_MISMATCH"
    assert "embedding_id" in exc.value.fields


def test_resume_refuses_foreign_schema(env, tmp_path) -> None:
    out = tmp_path / "dir"
    _run(env, _cfg(smoke_updates=4), out, resume=False)
    path = out / "checkpoint_last.pt"
    payload = _load(path)
    payload["schema"] = "some_other_schema"
    torch.save(payload, str(path))
    with pytest.raises(V541TrainerError) as exc:
        _run(env, _cfg(smoke_updates=8), out, resume=True)
    assert exc.value.code == "RESUME_IDENTITY_MISMATCH"
    assert exc.value.fields == ["schema"]


def test_resume_refuses_tampered_content_digest(env, tmp_path) -> None:
    """A payload edited after it was written never reaches the model."""
    out = tmp_path / "dir"
    _run(env, _cfg(smoke_updates=4), out, resume=False)
    path = out / "checkpoint_last.pt"
    payload = _load(path)
    assert content_digest_of(payload) == payload["content_digest_sha256"]

    key = next(iter(payload["model"]))
    payload["model"][key] = payload["model"][key] + 1.0      # tamper
    torch.save(payload, str(path))

    with pytest.raises(V541TrainerError) as exc:
        _run(env, _cfg(smoke_updates=8), out, resume=True)
    assert exc.value.code == "RESUME_CONTENT_DIGEST"
    assert exc.value.fields == ["content_digest_sha256"]
    # the refusal happens BEFORE any load: file untouched, no resume line
    assert _load(path)["update"] == 4
    assert "resume" not in [e["event"] for e in
                            read_manifest(out / "manifest.jsonl")]


def test_resume_refuses_missing_content_digest(env, tmp_path) -> None:
    out = tmp_path / "dir"
    _run(env, _cfg(smoke_updates=4), out, resume=False)
    path = out / "checkpoint_last.pt"
    payload = _load(path)
    payload.pop("content_digest_sha256")
    torch.save(payload, str(path))
    with pytest.raises(V541TrainerError) as exc:
        _run(env, _cfg(smoke_updates=8), out, resume=True)
    assert exc.value.code == "RESUME_CONTENT_DIGEST"
    assert exc.value.fields == ["content_digest_sha256"]


def test_intact_resume_passes_the_content_digest(env, tmp_path) -> None:
    """The happy path still verifies (and only then loads)."""
    out = tmp_path / "dir"
    _run(env, _cfg(smoke_updates=4), out, resume=False)
    payload = _load(out / "checkpoint_last.pt")
    assert content_digest_of(payload) == payload["content_digest_sha256"]
    metrics = _run(env, _cfg(smoke_updates=8), out, resume=True)
    assert metrics["schedule"]["updates_run"] == 4
    assert "resume" in [e["event"] for e in
                        read_manifest(out / "manifest.jsonl")]


def test_matching_identity_resumes(env, tmp_path) -> None:
    out = tmp_path / "dir"
    _run(env, _cfg(smoke_updates=4), out, resume=False)
    metrics = _run(env, _cfg(smoke_updates=8), out, resume=True)
    # leg 2 only executed updates 5..8 — 1..4 came from checkpoint_last
    assert metrics["schedule"]["updates_run"] == 4
    assert _load(out / "checkpoint_last.pt")["update"] == 8
    assert read_manifest(out / "manifest.jsonl")[0]["event"].startswith(
        "checkpoint_")


# ---------------------------------------------------------------------------
# payload completeness + content digest (§7.3 / §12.2)
# ---------------------------------------------------------------------------


def test_checkpoint_payload_carries_every_required_field(env, tmp_path) -> None:
    out = tmp_path / "payload"
    _run(env, _cfg(smoke_updates=4), out, resume=False,
         provenance={"tier": "50k", "data_manifest_sha256": "d" * 64,
                     "split_manifest_sha256": "s" * 64,
                     "teacher_manifest_sha256": "t" * 64,
                     "shard_path": "/pes2ts_data/v53mp_50k_data"})
    payload = _load(out / "checkpoint_last.pt")

    for key in ("schema", "model", "optimizer", "update", "best_val",
                "best_update", "config", "config_full_sha256", "identity",
                "provenance", "seeds", "rng", "sampler", "schedule",
                "sampler_counters", "monitor_state", "data_position",
                "git", "env", "tau_end_calibration",
                "content_digest_sha256"):
        assert key in payload, key
    assert payload["schema"] == V541_TRAIN_SCHEMA

    # all RNG states
    assert set(payload["rng"]) >= {"python", "numpy", "torch"}
    assert isinstance(payload["rng"]["torch"], torch.Tensor)
    # strata pool + spec SHA
    assert payload["sampler"]["sampler_id"] == "v541_curve_stratified_v1"
    assert payload["sampler"]["spec_sha256"] == sampler_spec_sha256()
    assert len(payload["sampler"]["pool_sha256"]) == 64
    # schedule hasher state (round-trips through from_state)
    restored = BatchScheduleHasher.from_state(payload["schedule"])
    assert restored.chain_sha256 == payload["schedule"]["chain_sha256"]
    assert restored.records == payload["schedule"]["records"]
    # data / shard position
    dp = payload["data_position"]
    assert dp["n_train_positions"] > 0 and dp["n_inner_val_positions"] > 0
    assert len(dp["train_positions_sha256"]) == 64
    assert len(dp["inner_val_positions_sha256"]) == 64
    assert dp["data_manifest_sha256"] == "d" * 64
    assert dp["split_manifest_sha256"] == "s" * 64
    assert dp["shard"] == "/pes2ts_data/v53mp_50k_data"
    assert dp["dev_curve_ids"] and "ONLY to exclude dev curves" in dp["note"]
    # the four identity SHAs
    ident = payload["identity"]
    for key in ("common_protocol_sha256", "embedding_spec_sha256",
                "feature_recipe_sha256", "config_sha256"):
        assert key in ident
    assert ident["config_sha256"] == payload["identity"]["config_sha256"]
    # git / env
    assert payload["git"]["commit"] is None or len(
        payload["git"]["commit"]) == 40
    assert payload["env"]["torch"] == torch.__version__
    # τ was calibrated and frozen into the payload
    assert isinstance(payload["tau_end_calibration"], dict)
    assert "tau_end" in payload["tau_end_calibration"]
    # content digest verifies against the payload itself
    assert content_digest_of(payload) == payload["content_digest_sha256"]


def test_content_digest_changes_with_model_weights(env, tmp_path) -> None:
    out = tmp_path / "digest"
    _run(env, _cfg(smoke_updates=2), out, resume=False)
    payload = _load(out / "checkpoint_last.pt")
    original = payload["content_digest_sha256"]
    assert content_digest_of(payload) == original
    key = next(iter(payload["model"]))
    payload["model"][key] = payload["model"][key] + 1.0
    assert content_digest_of(payload) != original


# ---------------------------------------------------------------------------
# manifest APPEND (§7.3: 追加而非覆盖)
# ---------------------------------------------------------------------------


def test_manifest_appends_never_overwrites(env, tmp_path) -> None:
    out = tmp_path / "manifest"
    _run(env, _cfg(smoke_updates=4), out, resume=False)
    first = read_manifest(out / "manifest.jsonl")
    assert first, "manifest must exist after leg 1"
    first_text = (out / "manifest.jsonl").read_text(encoding="utf-8")
    first_events = [e["event"] for e in first]
    assert "checkpoint_last" in first_events
    assert "checkpoint_best" in first_events
    assert "tau_calibration" in first_events
    assert "run_complete" in first_events

    _run(env, _cfg(smoke_updates=8), out, resume=True)
    second_text = (out / "manifest.jsonl").read_text(encoding="utf-8")
    second = read_manifest(out / "manifest.jsonl")
    # byte-for-byte prefix preservation: nothing was rewritten
    assert second_text.startswith(first_text)
    assert len(second) > len(first)
    assert "resume" in [e["event"] for e in second]
    assert second[:len(first)] == first

    # a third leg keeps appending (never truncating)
    _run(env, _cfg(smoke_updates=12), out, resume=True)
    third = read_manifest(out / "manifest.jsonl")
    assert len(third) > len(second)
    assert third[:len(second)] == second
    assert [e["event"] for e in third].count("resume") == 2
    # every attempt keeps its own line (§12.2): leg1 ended at update 4,
    # leg2 resumed from 4 and ended at 8, leg3 resumed from 8 and ended
    # at 12 — two resume records, in order, nothing overwritten.
    resumes = [e for e in third if e["event"] == "resume"]
    assert [e["from_update"] for e in resumes] == [4, 8]
    assert [e["next_update"] for e in resumes] == [5, 9]


def test_manifest_records_digest_and_sha_per_checkpoint(env, tmp_path) -> None:
    out = tmp_path / "manifest2"
    _run(env, _cfg(smoke_updates=4), out, resume=False)
    entries = read_manifest(out / "manifest.jsonl")
    saves = [e for e in entries if e["event"].startswith("checkpoint_")]
    assert saves
    for entry in saves:
        assert len(entry["file_sha256"]) == 64
        assert len(entry["content_digest_sha256"]) == 64
        assert len(entry["config_sha256"]) == 64
        assert entry["update"] >= 1
        assert Path(entry["path"]).is_file()
    complete = [e for e in entries if e["event"] == "run_complete"][0]
    assert complete["run_outcome"] == "done"
    assert complete["ledger_status_written"] is False


# ---------------------------------------------------------------------------
# schedule-hash continuity (§7.2)
# ---------------------------------------------------------------------------


def _reference_chain(env, *, seed: int, n_updates: int) -> str:
    samples = env.ds.samples
    dev_curves = {str(samples[p].curve_id) for p in env.dev_pos}
    pool = [i for i in range(len(samples))
            if str(samples[i].curve_id) not in dev_curves]
    pool_curves = sorted({str(samples[i].curve_id) for i in pool})
    train_pos, _iv = tr.v53mp_pool_split(
        samples, pool_curve_ids=pool_curves, seed=seed,
        inner_val_fraction=0.2)
    index = StrataIndex.build(samples, train_pos, dataset=env.ds)
    hasher = BatchScheduleHasher()
    for update in range(1, n_updates + 1):
        positions = curve_batch_positions_v541(
            index, rng=batch_rng_v541(seed, update), batch_curves=8)
        hasher.record(update, positions)
    return hasher.chain_sha256


def test_schedule_hash_continuity_across_resume(env, tmp_path) -> None:
    resumed_dir = tmp_path / "resumed"
    fresh_dir = tmp_path / "fresh"
    _run(env, _cfg(smoke_updates=20), resumed_dir, resume=False)
    m2 = _run(env, _cfg(smoke_updates=40), resumed_dir, resume=True)
    m3 = _run(env, _cfg(smoke_updates=40), fresh_dir, resume=False)

    expected = _reference_chain(env, seed=SEED, n_updates=40)
    assert m2["sampler"]["batch_schedule_sha256"] == expected
    assert m3["sampler"]["batch_schedule_sha256"] == expected
    assert m2["sampler"]["schedule_updates"] == 40
    assert m3["sampler"]["schedule_updates"] == 40
    # the checkpoint carries the SAME chain
    payload = _load(resumed_dir / "checkpoint_last.pt")
    assert payload["schedule"]["chain_sha256"] == expected
    assert payload["schedule"]["last_update"] == 40
    # a resumed leg never re-records already-hashed updates
    assert payload["schedule"]["records"] == 40


def test_schedule_hasher_refuses_non_contiguous_and_foreign_state() -> None:
    hasher = BatchScheduleHasher()
    hasher.record(1, [1, 2, 3])
    with pytest.raises(Exception) as exc:
        hasher.record(3, [1, 2, 3])
    assert getattr(exc.value, "code", "") == "SCHEDULE_UPDATE_NONCONTIGUOUS"
    state = hasher.state_dict()
    state["spec_sha256"] = "0" * 64
    with pytest.raises(Exception) as exc:
        BatchScheduleHasher.from_state(state)
    assert getattr(exc.value, "code", "") == "SCHEDULE_STATE_INVALID"
