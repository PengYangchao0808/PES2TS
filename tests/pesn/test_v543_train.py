"""V5.4.3 WP4 — S1 trainer CPU certificate (plan §7.2 :651–733, §7.3 :735–746,
§7.4 :748–773, §8.1 :797–809, §11.2 ``v543_train.py``).

Pinned here (the WP4 acceptance list of the WP checklist W3):

* the frozen schedule: 37,900 flat + 400 warmup, LR 3e-4 / AdamW wd 1e-4 /
  clip 1.0 / fp32 / batch 8×4 / val every 100 / inner-val cap 8,192 — every
  deviation is a typed refusal, and ``--stop-after`` never shrinks the dose;
* the frozen loss coefficients, EXACTLY: 0.5 / 0.02 / 0.5 / 0.5 /
  0.012385049119773344, and the assembled loss really is
  ``L_atom + 0.5·L_pair + 0.02·L_cos`` / ``0.5·L_rank + 0.5·L_q +
  0.012385049119773344·L_endpoint``;
* **the PROCESS frame mask is the endpoint-label mask over ALL statuses**
  (plan :707, the F5 correction): a ``terminal`` frame with a valid endpoint
  label TRAINS the PROCESS head and is EXCLUDED from MOVE — the two masks are
  asserted to disagree on exactly that row;
* gradient isolation on both tracks (§7.2 :713–726), encoder = exactly one
  forward per update (§4.4), and the cache path skips the encoder entirely;
* resume identity (16 keys, 14 plan lines) matches key-by-key + digest, and a
  drift is a typed refusal; 3+3 updates reproduce 6 updates bit-for-bit;
* a batch with no valid supervision records ``skipped_no_grad`` (:711)."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from pes2ts_core.pesn.contracts.key_atom_v3 import forward_view_v3  # noqa: E402
from pes2ts_core.pesn.models.v543.contracts import PRECISION_VERSION_V543  # noqa: E402
from pes2ts_core.pesn.offline.training.v542_train import (  # noqa: E402
    V542_LAMBDA_ENDPOINT,
    V542_LAMBDA_Q,
    V542_LAMBDA_RANK,
)
from pes2ts_core.pesn.offline.training.v543_train import (  # noqa: E402
    LOSS_COEFFICIENTS_V543,
    MOVE_FRAME_MASK_RULE,
    PROCESS_FRAME_MASK_RULE,
    RESUME_IDENTITY_FIELDS_V543,
    RESUME_IDENTITY_GROUPS_V543,
    SKIPPED_NO_GRAD,
    TrainBatch,
    V543_BATCH_CURVES,
    V543_BATCH_FRAMES,
    V543_CHECKPOINT_EVERY,
    V543_FROZEN_SEEDS,
    V543_GRAD_CLIP,
    V543_INNER_VAL_MAX_SAMPLES,
    V543_LAMBDA_COS,
    V543_LAMBDA_ENDPOINT,
    V543_LAMBDA_PAIR,
    V543_LAMBDA_Q,
    V543_LAMBDA_RANK,
    V543_LR,
    V543_SAMPLER_ID,
    V543_SAMPLER_SPEC_SHA256_PINNED,
    V543_TOTAL_UPDATES,
    V543_TRAIN_SCHEMA,
    V543_VAL_EVERY,
    V543_WARMUP_UPDATES,
    V543_WEIGHT_DECAY,
    V543Trainer,
    V543TrainerConfig,
    V543TrainerError,
    _supervision_for_batch,
    assert_frozen_loss_coefficients,
    assert_resume_identity_v543,
    identity_block_v543,
    lr_at_update,
    move_frame_mask_v543,
    move_track_loss,
    process_frame_mask_v543,
    process_track_loss,
    track_has_supervision,
    train_run_v543,
)
from tests.pesn._v543_fixtures_wp4 import (  # noqa: E402
    SynthMoveRefiner,
    SynthProcessRefiner,
    SynthSource,
    batch_with_status,
    make_encoder,
    make_model,
    synthetic_tables,
)


# ---------------------------------------------------------------------------
# the frozen schedule / coefficients
# ---------------------------------------------------------------------------
def test_frozen_schedule_constants() -> None:
    assert V543_TOTAL_UPDATES == 37_900
    assert V543_WARMUP_UPDATES == 400
    assert V543_LR == 3.0e-4
    assert V543_WEIGHT_DECAY == 1.0e-4
    assert V543_GRAD_CLIP == 1.0
    assert V543_BATCH_CURVES == 8
    assert V543_BATCH_FRAMES == 32
    assert V543_VAL_EVERY == 100
    assert V543_INNER_VAL_MAX_SAMPLES == 8_192
    assert V543_CHECKPOINT_EVERY == 500
    assert V543_FROZEN_SEEDS == (17, 29, 43)
    assert V543_SAMPLER_ID == "v541_curve_stratified_v1"
    assert V543_SAMPLER_SPEC_SHA256_PINNED == (
        "0769817ef01dca3abbb1d67f5405e98138b4707e74b690240d3f9b6821b65d9e")
    assert PRECISION_VERSION_V543 == "fp32_v1"


def test_frozen_loss_coefficients_are_exact() -> None:
    """plan §7.2 :691/:703 — the numbers, to the last digit."""
    assert LOSS_COEFFICIENTS_V543 == {
        "lambda_pair": 0.5,
        "lambda_cos": 0.02,
        "lambda_strong": 0.0,
        "lambda_dead": 0.0,
        "lambda_rank": 0.5,
        "lambda_q": 0.5,
        "lambda_endpoint": 0.012385049119773344,
    }
    assert V543_LAMBDA_PAIR == 0.5
    assert V543_LAMBDA_COS == 0.02
    assert V543_LAMBDA_RANK == 0.5
    assert V543_LAMBDA_Q == 0.5
    assert V543_LAMBDA_ENDPOINT == 0.012385049119773344
    # the V5.4.2 frozen constants are the SAME numbers (reuse, not re-tuning)
    assert V542_LAMBDA_RANK == V543_LAMBDA_RANK
    assert V542_LAMBDA_Q == V543_LAMBDA_Q
    assert V542_LAMBDA_ENDPOINT == V543_LAMBDA_ENDPOINT
    assert assert_frozen_loss_coefficients()["lambda_endpoint"] == \
        0.012385049119773344


def test_loss_coefficient_drift_is_refused(monkeypatch) -> None:
    import pes2ts_core.pesn.offline.training.v543_train as module

    monkeypatch.setattr(module, "V543_LAMBDA_Q", 0.4, raising=True)
    with pytest.raises(V543TrainerError) as error:
        module.assert_frozen_loss_coefficients()
    assert error.value.code == "V543_LOSS_COEFFICIENT_DRIFT"


@pytest.mark.parametrize("update,expected", [
    (1, V543_LR / 400.0),
    (200, V543_LR / 2.0),
    (400, V543_LR),
    (401, V543_LR),
    (37_900, V543_LR),
])
def test_warmup_is_linear_then_constant(update: int, expected: float) -> None:
    assert lr_at_update(update) == pytest.approx(expected, rel=0, abs=1e-18)


# ---------------------------------------------------------------------------
# config refusals
# ---------------------------------------------------------------------------
def test_config_accepts_the_frozen_schedule() -> None:
    config = V543TrainerConfig(track="move", decoder_id="M0", seed=17,
                               stop_after=5)
    assert config.total_updates == 5
    assert config.is_formal_schedule is False
    assert config.cache_enabled is True
    described = config.describe()
    assert described["updates"] == 37_900
    assert described["batch"] == "8x4"
    assert described["loss_coefficients"]["lambda_endpoint"] == \
        0.012385049119773344
    assert described["frame_masks"] == {"move": MOVE_FRAME_MASK_RULE,
                                        "process": PROCESS_FRAME_MASK_RULE}


@pytest.mark.parametrize("kwargs,code", [
    ({"updates": 30_000}, "V543_DOSE"),
    ({"warmup_updates": 300}, "V543_WARMUP"),
    ({"lr": 1e-3}, "V543_OPTIMIZER_FROZEN"),
    ({"weight_decay": 0.0}, "V543_OPTIMIZER_FROZEN"),
    ({"grad_clip": 5.0}, "V543_OPTIMIZER_FROZEN"),
    ({"batch_curves": 4}, "V543_BATCH_FROZEN"),
    ({"frames_per_curve": 8}, "V543_BATCH_FROZEN"),
    ({"val_every": 50}, "V543_VAL_EVERY"),
    ({"inner_val_max_samples": 4096}, "V543_INNER_VAL_CAP"),
    ({"precision": "fp16"}, "V543_PRECISION"),
    ({"torch_compile": True}, "V543_TORCH_COMPILE"),
    ({"sampler": "uniform"}, "V543_SAMPLER_DRIFT"),
    ({"sampler_spec_sha256": "0" * 64}, "V543_SAMPLER_DRIFT"),
    ({"lambda_pair": 0.4}, "V543_LOSS_FROZEN"),
    ({"lambda_endpoint": 0.02}, "V543_LOSS_FROZEN"),
    ({"seed": 7}, "V543_SEED_FROZEN"),
    ({"track": "joint"}, "V543_TRACK"),
    ({"stop_after": 0}, "V543_STOP_AFTER"),
    ({"cache_mode": "streaming"}, "V543_CACHE_MODE"),
])
def test_config_refusals(kwargs: dict, code: str) -> None:
    with pytest.raises(V543TrainerError) as error:
        V543TrainerConfig(**kwargs)
    assert error.value.code == code


def test_formal_schedule_flag_needs_the_full_dose() -> None:
    assert V543TrainerConfig().is_formal_schedule is True
    assert V543TrainerConfig(stop_after=200).is_formal_schedule is False


def test_identity_block_is_the_plan_list() -> None:
    """plan §7.4 :750–767 — 14 lines, flattened to 16 keys."""
    assert len(RESUME_IDENTITY_GROUPS_V543) == 14
    assert len(RESUME_IDENTITY_FIELDS_V543) == 16
    assert RESUME_IDENTITY_FIELDS_V543[:4] == (
        "encoder_ckpt_sha256", "bridge_spec_sha256", "decoder_id", "track")
    assert "cache_identity" in RESUME_IDENTITY_FIELDS_V543


# ---------------------------------------------------------------------------
# the PROCESS frame mask (plan :707 — the F5 correction)
# ---------------------------------------------------------------------------
TERMINAL_STATUSES = ("terminal", "masked", "conflict", "ok")


@pytest.fixture(scope="module")
def mask_batch() -> dict:
    return batch_with_status(TERMINAL_STATUSES, jitter_seed=11)


def test_process_mask_covers_terminal_rows(mask_batch: dict) -> None:
    """``terminal``/``masked``/``conflict`` frames with a valid endpoint label
    are IN the PROCESS mask and OUT of the MOVE mask."""
    tables = synthetic_tables(mask_batch, endpoint_labels=(1, 0, 1, 0))
    device = "cpu"
    process = process_frame_mask_v543(tables.sample_ids, tables.endpoints,
                                      device)
    move = move_frame_mask_v543(TERMINAL_STATUSES, device)
    assert tuple(process["process_frame_mask"].tolist()) == \
        (True, True, True, True)
    assert tuple(move["move_frame_mask"].tolist()) == \
        (False, False, False, True)
    assert process["rule"] == "endpoint_label_mask_all_statuses"
    assert move["rule"] == "label_status_ok_only"
    # the two masks disagree on exactly the non-ok rows
    disagreement = process["process_frame_mask"] & ~move["move_frame_mask"]
    assert int(disagreement.sum()) == 3


def test_invalid_endpoint_labels_are_out_of_the_process_mask(
        mask_batch: dict) -> None:
    tables = synthetic_tables(mask_batch, endpoint_labels=(-1, 0, -1, 1))
    process = process_frame_mask_v543(tables.sample_ids, tables.endpoints,
                                      "cpu")
    assert tuple(process["process_frame_mask"].tolist()) == \
        (False, True, False, True)


def test_process_loss_trains_terminal_rows(mask_batch: dict) -> None:
    """The terminal row really participates in L_P (and not in L_M)."""
    tables = synthetic_tables(mask_batch, endpoint_labels=(1, 0, 1, 0))
    config = V543TrainerConfig(track="process", decoder_id="P0", seed=17,
                               stop_after=1)
    model = make_model(track="process")
    view = forward_view_v3(dict(mask_batch))
    packet = model.assembly.encoder(view)
    out = model.forward_context(
        model.assembly.bridge.build_context(packet, view))
    tb = TrainBatch(
        positions=(0, 1, 2, 3), batch=mask_batch,
        sample_ids=tables.sample_ids,
        curve_ids=tuple(str(c) for c in mask_batch["curve_ids"]),
        label_status=TERMINAL_STATUSES, split="inner_train")
    supervision = _supervision_for_batch(tb, tables.labels, tables.endpoints,
                                        config)
    assert int(supervision["process_frame_mask"].sum()) == 4
    assert int(supervision["move_frame_mask"].sum()) == 1
    assert int(supervision["q_mask"].sum()) == 4
    process = process_track_loss(out, tb.batch, supervision, config,
                                 endpoint_pos_weight=1.0)
    assert process["n_endpoint_rows"] == 4      # all four, terminal included
    assert math.isfinite(float(process["loss"]))
    move = move_track_loss(out, tb.batch, supervision, config)
    assert move["n_frames"] == 1                # MOVE sees the ok row only
    # and the assembled losses are the frozen formulas
    assert float(move["loss"]) == pytest.approx(
        float(move["l_atom"]) + 0.5 * float(move["l_pair"])
        + 0.02 * float(move["l_cos"]), rel=1e-5, abs=1e-7)
    assert float(process["loss"]) == pytest.approx(
        0.5 * float(process["l_rank"]) + 0.5 * float(process["l_q"])
        + 0.012385049119773344 * float(process["l_endpoint"]),
        rel=1e-5, abs=1e-7)


def test_no_move_eligible_frame_is_skipped(mask_batch: dict) -> None:
    tables = synthetic_tables(mask_batch, endpoint_labels=(1, 0, 1, 0))
    config = V543TrainerConfig(track="move", decoder_id="M0", seed=17,
                               stop_after=1)
    tb = TrainBatch(
        positions=(0, 1, 2, 3), batch=mask_batch,
        sample_ids=tables.sample_ids,
        curve_ids=tuple(str(c) for c in mask_batch["curve_ids"]),
        label_status=("terminal", "masked", "conflict", "terminal"),
        split="inner_train")
    supervision = _supervision_for_batch(tb, tables.labels, tables.endpoints,
                                        config)
    assert track_has_supervision(supervision, "move") is False
    assert track_has_supervision(supervision, "process") is True


# ---------------------------------------------------------------------------
# end-to-end runs (M0/P0 baselines and a trainable stand-in)
# ---------------------------------------------------------------------------
def _source(*, status=("ok",) * 4, endpoints=(1, 0, 1, 0), repeat=False,
            **kwargs) -> SynthSource:
    batch = batch_with_status(status, jitter_seed=17)
    tables = synthetic_tables(batch, endpoint_labels=endpoints)
    return SynthSource(tables, label_status=status, repeat_batch=repeat,
                       **kwargs)


def _run(tmp_path: Path, *, track: str, decoder=None, decoder_id: str,
         updates: int, source: SynthSource, seed: int = 17,
         resume_from=None, use_cache: bool = True,
         encoder_state=None) -> dict:
    model = make_model(track=track, decoder=decoder, decoder_id=decoder_id,
                       encoder=(None if encoder_state is None
                                else make_encoder(encoder_state)))
    config = V543TrainerConfig(track=track, decoder_id=decoder_id, seed=seed,
                               stop_after=updates, use_cache=use_cache)
    return train_run_v543(
        model, config, source, labels_v52=source.tables.labels,
        endpoint_labels=source.tables.endpoints, out_dir=tmp_path,
        resume_from=resume_from)


@pytest.mark.parametrize("track,decoder_id", [("move", "M0"),
                                              ("process", "P0")])
def test_baseline_run_writes_the_acceptance_artifacts(
        tmp_path: Path, track: str, decoder_id: str) -> None:
    source = _source()
    metrics = _run(tmp_path, track=track, decoder_id=decoder_id,
                   updates=2, source=source)
    assert metrics["schema"] == V543_TRAIN_SCHEMA
    assert metrics["outcome"] == "done"
    assert metrics["updates"] == 2
    assert metrics["is_formal_schedule"] is False
    assert metrics["best_val_loss"] is not None
    assert math.isfinite(float(metrics["best_val_loss"]))
    assert metrics["identity_sha256"] == metrics["identity_sha256"]
    for key in RESUME_IDENTITY_FIELDS_V543:
        assert key in metrics["identity"], key
    for name in ("checkpoint_best.pt", "checkpoint_last.pt", "metrics.json",
                 "train_log.jsonl", "run_records.jsonl"):
        assert (tmp_path / name).is_file(), name
    rows = [json.loads(line) for line in
            (tmp_path / "train_log.jsonl").read_text().splitlines()]
    assert len(rows) == 2
    assert all(row["event"] == "train" for row in rows)
    assert all(row["grad_finite"] is None for row in rows)   # no parameter
    records = [json.loads(line) for line in
               (tmp_path / "run_records.jsonl").read_text().splitlines()]
    assert records and records[-1]["event"] == "validate"
    checkpoint = torch.load(str(tmp_path / "checkpoint_last.pt"),
                            map_location="cpu", weights_only=False)
    assert checkpoint["schema"] == V543_TRAIN_SCHEMA
    assert checkpoint["identity_sha256"] == metrics["identity_sha256"]
    assert checkpoint["model"] == {}                     # zero-parameter
    assert checkpoint["content_digest_sha256"]


def test_trainable_candidate_runs_with_gradient_isolation(
        tmp_path: Path) -> None:
    source = _source()
    refiner = SynthMoveRefiner()
    model = make_model(track="move", decoder=refiner, decoder_id="M1")
    config = V543TrainerConfig(track="move", decoder_id="M1", seed=17,
                               stop_after=3)
    trainer = V543Trainer(
        model=model, config=config, source=source,
        labels_v52=source.tables.labels,
        endpoint_labels=source.tables.endpoints, out_dir=tmp_path)
    before = [p.detach().clone() for p in refiner.parameters()]
    metrics = trainer.run()
    after = [p.detach().clone() for p in refiner.parameters()]
    assert any(not torch.equal(a, b) for a, b in zip(before, after)), \
        "the active refiner did not move"
    for parameter in model.assembly.encoder.parameters():
        assert parameter.grad is None
    assert all(p.grad is None
               for p in model.assembly.process.decoder.parameters())
    rows = [json.loads(line) for line in
            (tmp_path / "train_log.jsonl").read_text().splitlines()]
    assert all(row["grad_finite"] for row in rows)
    # warmup is real: update u gets 3e-4 · u / 400 (plan §7.2 :660)
    assert [row["lr"] for row in rows] == pytest.approx(
        [V543_LR * update / V543_WARMUP_UPDATES for update in (1, 2, 3)])
    assert metrics["gradient_isolation"]["active_trainable_params"] == 3
    checkpoint = torch.load(str(tmp_path / "checkpoint_last.pt"),
                            map_location="cpu", weights_only=False)
    assert checkpoint["model"]


def test_encoder_runs_once_per_update_and_cache_hits_skip_it(
        tmp_path: Path) -> None:
    source = _source(repeat=True)
    model = make_model(track="move")
    config = V543TrainerConfig(track="move", decoder_id="M0", seed=17,
                               stop_after=2)
    captured = {"counts": []}
    encoder = model.assembly.encoder
    real_forward = encoder.forward

    def counting_forward(batch):  # noqa: ANN001 - instrumentation
        captured["counts"].append(1)
        return real_forward(batch)

    encoder.forward = counting_forward            # type: ignore[assignment]
    trainer = V543Trainer(
        model=model, config=config, source=source,
        labels_v52=source.tables.labels,
        endpoint_labels=source.tables.endpoints, out_dir=tmp_path)
    metrics = trainer.run()
    # 4 batches in the run (2 updates + 2 inner-val batches); update 2 is a
    # FULL cache hit, so exactly 3 encoder forwards happen (plan §7.3)
    assert len(captured["counts"]) == 3
    assert metrics["cache"]["hits"] >= 1
    # the final validation re-reads the same frames -> hits, no encoder calls
    before = len(captured["counts"])
    trainer.validate(99)
    assert len(captured["counts"]) == before
    encoder.forward = real_forward               # type: ignore[assignment]
    assert metrics["cache"]["hits"] > 0


def test_skipped_no_grad_is_recorded(tmp_path: Path) -> None:
    source = _source(status=("terminal",) * 4, endpoints=(1, 0, 1, 0))
    model = make_model(track="move")
    config = V543TrainerConfig(track="move", decoder_id="M0", seed=17,
                               stop_after=2)
    trainer = V543Trainer(
        model=model, config=config, source=source,
        labels_v52=source.tables.labels,
        endpoint_labels=source.tables.endpoints, out_dir=tmp_path)
    metrics = trainer.run()
    assert metrics["skipped_updates"] == 2
    rows = [json.loads(line) for line in
            (tmp_path / "train_log.jsonl").read_text().splitlines()]
    assert [row["event"] for row in rows] == [SKIPPED_NO_GRAD] * 2


def test_process_track_runs_with_a_trainable_refiner(tmp_path: Path) -> None:
    source = _source()
    refiner = SynthProcessRefiner()
    model = make_model(track="process", decoder=refiner, decoder_id="P1")
    config = V543TrainerConfig(track="process", decoder_id="P1", seed=29,
                               stop_after=2)
    trainer = V543Trainer(
        model=model, config=config, source=source,
        labels_v52=source.tables.labels,
        endpoint_labels=source.tables.endpoints, out_dir=tmp_path)
    metrics = trainer.run()
    rows = [json.loads(line) for line in
            (tmp_path / "train_log.jsonl").read_text().splitlines()]
    assert rows and all(row["event"] == "train" for row in rows)
    assert all("l_rank" in row and "l_q" in row and "l_endpoint" in row
               for row in rows)
    assert all("l_move" not in row for row in rows), \
        "the PROCESS track must not call the MOVE loss (plan :706)"
    assert metrics["endpoint_pos_weight"]["pos_weight"] == 1.0
    for parameter in model.assembly.encoder.parameters():
        assert parameter.grad is None


def test_resume_identity_matches_then_refuses_drift(tmp_path: Path) -> None:
    source = _source()
    metrics = _run(tmp_path, track="move", decoder=SynthMoveRefiner(),
                   decoder_id="M1", updates=2, source=source)
    state = torch.load(str(tmp_path / "checkpoint_last.pt"),
                       map_location="cpu", weights_only=False)
    # a matching identity is returned unchanged
    assert assert_resume_identity_v543(state, metrics["identity"]) == \
        state["identity"]
    drifted = dict(metrics["identity"])
    drifted["seed"] = 29
    with pytest.raises(V543TrainerError) as error:
        assert_resume_identity_v543(state, drifted)
    assert error.value.code == "V543_RESUME_IDENTITY"
    assert "seed" in error.value.detail
    other = dict(metrics["identity"])
    other["encoder_ckpt_sha256"] = "0" * 64
    with pytest.raises(V543TrainerError):
        assert_resume_identity_v543(state, other)
    broken = dict(state)
    broken["identity_sha256"] = "0" * 64
    with pytest.raises(V543TrainerError):
        assert_resume_identity_v543(broken, metrics["identity"])
    with pytest.raises(V543TrainerError) as error:
        assert_resume_identity_v543({"schema": "other"}, metrics["identity"])
    assert error.value.code == "V543_RESUME_IDENTITY"


def test_exact_resume_reproduces_the_uninterrupted_run(tmp_path: Path) -> None:
    """plan §7.4 :773 — 3 + 3 updates == 6 updates, bit-for-bit."""
    straight = tmp_path / "straight"
    split = tmp_path / "split"
    # ONE encoder state and ONE decoder init for every leg: the §7.4 identity
    # measures the encoder digest, and a resumed leg must start from the
    # checkpoint's decoder weights, not from a fresh random init.
    encoder_state = make_encoder().state_dict()
    decoder_init = SynthMoveRefiner().state_dict()

    def fresh_decoder() -> SynthMoveRefiner:
        refiner = SynthMoveRefiner()
        refiner.load_state_dict(dict(decoder_init))
        return refiner

    straight_metrics = _run(straight, track="move",
                            decoder=fresh_decoder(), decoder_id="M1",
                            updates=6, source=_source(),
                            encoder_state=encoder_state)
    _run(split, track="move", decoder=fresh_decoder(), decoder_id="M1",
         updates=3, source=_source(), encoder_state=encoder_state)
    resumed_metrics = _run(split, track="move", decoder=fresh_decoder(),
                           decoder_id="M1", updates=6, source=_source(),
                           resume_from=split / "checkpoint_last.pt",
                           encoder_state=encoder_state)
    straight_state = torch.load(str(straight / "checkpoint_last.pt"),
                                map_location="cpu", weights_only=False)
    resumed_state = torch.load(str(split / "checkpoint_last.pt"),
                               map_location="cpu", weights_only=False)
    assert straight_state["identity_sha256"] == \
        resumed_state["identity_sha256"]
    assert set(straight_state["model"]) == set(resumed_state["model"])
    for key in straight_state["model"]:
        assert torch.equal(straight_state["model"][key],
                           resumed_state["model"][key]), key
    assert straight_metrics["best_val_loss"] == \
        resumed_metrics["best_val_loss"]
    straight_rows = (straight / "train_log.jsonl").read_text().splitlines()
    split_rows = (split / "train_log.jsonl").read_text().splitlines()
    assert len(straight_rows) == 6
    assert len(split_rows) == 6
    for left, right in zip(straight_rows, split_rows):
        assert json.loads(left)["loss"] == json.loads(right)["loss"]


def test_resume_refuses_a_foreign_checkpoint(tmp_path: Path) -> None:
    _run(tmp_path / "a", track="move", decoder_id="M0", updates=1,
         source=_source(), seed=17)

    source = _source()
    model = make_model(track="move")
    config = V543TrainerConfig(track="move", decoder_id="M0", seed=29,
                               stop_after=2)
    trainer = V543Trainer(
        model=model, config=config, source=source,
        labels_v52=source.tables.labels,
        endpoint_labels=source.tables.endpoints, out_dir=tmp_path / "b")
    with pytest.raises(V543TrainerError) as error:
        trainer.resume_from(tmp_path / "a" / "checkpoint_last.pt")
    assert error.value.code == "V543_RESUME_IDENTITY"
    assert "seed" in error.value.detail
    with pytest.raises(V543TrainerError) as error:
        trainer.resume_from(tmp_path / "missing.pt")
    assert error.value.code == "V543_RESUME_MISSING"


def test_identity_block_reports_the_live_environment(tmp_path: Path) -> None:
    source = _source()
    model = make_model(track="move")
    config = V543TrainerConfig(track="move", decoder_id="M0", seed=17,
                               stop_after=1)
    block = identity_block_v543(
        model=model, config=config, source=source, cache=None,
        encoder_ckpt_sha256="a" * 64)
    assert tuple(block) == RESUME_IDENTITY_FIELDS_V543
    assert block["encoder_ckpt_sha256"] == "a" * 64
    assert block["track"] == "move"
    assert block["decoder_id"] == "M0"
    assert block["contract_version"] == "decoder_context_v1"
    assert block["cache_identity"] == "disabled"
    assert block["sampler_spec_sha256"] == V543_SAMPLER_SPEC_SHA256_PINNED
    assert len(block["environment_sha256"]) == 64
    assert len(block["code_fingerprint"]) == 64

