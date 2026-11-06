"""V5.4.3 WP4 — REAL-DATA S1 smoke: M0/P0 short run through the whole pipeline.

Marked ``realdata``: ``pytest.ini``'s default ``addopts`` EXCLUDE this file
(``-m "not realdata ..."``), so it only runs inside the GPU host (WSL
``pes2ts-gpu``)::

    PYTHONPATH=. python -m pytest -m realdata \
        tests/pesn/test_v543_train_realdata.py -q -s \
        -p no:cacheprovider --basetemp=.pytest_tmp_wp4/realdata

What this certifies (the WP4 acceptance items that need real data):

* the frozen A0 encoder strict-loads for the campaign seed, the frozen Bridge
  builds a real ``DecoderContextV1``, and the S1 loop runs 20 updates for the
  MOVE track (M0) and 20 for the PROCESS track (P0) — both losses finite, both
  writing ``train_log`` / ``run_records`` / ``checkpoint_*`` / ``metrics``;
* the encoder runs EXACTLY ONCE per batch (hook counters = 1, cumulative
  forward count = number of encoded batches), never per frame (§4.4);
* the §7.3 cache serves a repeated inner-val pass with ZERO extra encoder
  forwards and a non-zero hit count;
* ``checkpoint_last.pt`` restores into a fresh trainer after the §7.4 identity
  check, and a drifted identity is refused;
* the data path is the S0-probe / V5.3-MP path verbatim (frozen
  ``v53mp_pool_split`` + ``v541_curve_stratified_v1`` + ``_batch_from_positions``);
* without the frozen data view / pinned checkpoints the test SKIPS with an
  explicit reason (never a fake pass).

The evaluation budget of the smoke is deliberately tiny (2 inner-val batches =
64 frames): ``training.inner_val_max_samples = 8,192`` is an UPPER bound
(campaign YAML + plan §7.2 :668), and a smoke must not spend the scientific
validation budget.  Every consumed number is reported in the metrics payload.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from pes2ts_core.pesn.models.v543 import CoarseEncoder  # noqa: E402
from pes2ts_core.pesn.models.v543.assembly import assemble_v543  # noqa: E402
from pes2ts_core.pesn.models.v543.contracts import (  # noqa: E402
    HOOK_LABEL_EDGE_MSG,
    HOOK_LABEL_NODE_H,
    HOOK_LABEL_Q0_RAW,
)
from pes2ts_core.pesn.models.v543.model import V543Model  # noqa: E402
from pes2ts_core.pesn.models.v543.registry import (  # noqa: E402
    TRACK_MOVE,
    TRACK_PROCESS,
)
from pes2ts_core.pesn.offline.training.v543_train import (  # noqa: E402
    RealDataSource,
    V543_SAMPLER_SPEC_SHA256_PINNED,
    V543Trainer,
    V543TrainerConfig,
)

pytestmark = pytest.mark.realdata

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_VIEW = Path("/pes2ts_data/v53mp_50k_data")
#: Smoke dose: 20 updates per track (plan §8.1 T1 uses 200; a WP4 smoke is
#: shorter and is NOT a scientific attempt).
SMOKE_UPDATES = 20
#: Inner-val batches used by the smoke (see the module docstring).
SMOKE_VAL_BATCHES = 2
SEED = 17


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
        resolve_checkpoints(campaign, [SEED])
    except Exception as error:  # noqa: BLE001 - reported in the skip reason
        pytest.skip(f"pinned A0 checkpoints unavailable "
                    f"({type(error).__name__}: {error})")


class SmokeDataSource(RealDataSource):
    """The real frozen data path with a smoke-sized inner-val slice.

    Only :meth:`inner_val_batches` is narrowed (the trainer's 8,192 is an upper
    bound); the TRAIN path is untouched — same ``v53mp_pool_split``, same
    ``v541_curve_stratified_v1`` sampler, same ``_batch_from_positions``.
    """

    def inner_val_batches(self, limit: int):  # noqa: D102 - see class docstring
        return super().inner_val_batches(
            min(int(limit), SMOKE_VAL_BATCHES * 32))


@pytest.fixture(scope="module")
def real_context():
    """The frozen dataset, labels, pool split and the strict-loaded encoder."""
    _require_assets()
    from scripts.agent_ops.v542_a0_parity import (
        _open_frozen_dataset,
        load_pinned_assets,
    )
    from scripts.agent_ops.v543_s0_probe import (
        CAMPAIGN_CONFIG,
        DATA_VIEW_CONFIG,
        load_campaign_config,
        resolve_checkpoints,
    )
    from scripts.campaigns.r3_v53mp_run import _tier_pool_curve_ids

    campaign = load_campaign_config(PROJECT_ROOT / CAMPAIGN_CONFIG)
    checkpoints = resolve_checkpoints(campaign, [SEED])
    data_cfg, _assets = load_pinned_assets(PROJECT_ROOT / DATA_VIEW_CONFIG)
    dataset = _open_frozen_dataset(data_cfg)
    labels_v52, endpoint_labels = dataset.label_tables()
    split_path = str((data_cfg.get("split_contract") or {}).get(
        "split_manifest")
        or "/pes2ts_data/v53mp_50k_data/split_manifest_v53mp.json")
    split_doc = json.loads(Path(split_path).read_text(encoding="utf-8"))
    pool_ids = _tier_pool_curve_ids(split_doc, "50k")
    return {
        "dataset": dataset,
        "labels_v52": labels_v52,
        "endpoint_labels": endpoint_labels,
        "pool_ids": pool_ids,
        "checkpoint": checkpoints[SEED],
    }


def _run_track(real_context, tmp_path: Path, track: str, decoder_id: str):
    from pes2ts_core.pesn.offline.data.v543_feature_cache import CacheIdentity
    from pes2ts_core.pesn.offline.training.v543_train import (
        encoder_digest_v543,
    )

    encoder = CoarseEncoder.load(str(real_context["checkpoint"]["path"]), SEED)
    assembly = assemble_v543(encoder, active_track=track)
    model = V543Model(assembly)
    # The §5.7 frame keys carry the encoder digest: the source must use the
    # SAME digest the trainer measures, otherwise every lookup would miss and
    # the cache would be silently useless (the trainer refuses that mismatch).
    source = SmokeDataSource(
        real_context["dataset"], seed=SEED,
        pool_curve_ids=real_context["pool_ids"],
        cache_identity=CacheIdentity(
            encoder_ckpt_sha256=encoder_digest_v543(encoder)),
        feature_cache_capacity=4_096)
    config = V543TrainerConfig(track=track, decoder_id=decoder_id, seed=SEED,
                               stop_after=SMOKE_UPDATES, device="cpu")
    trainer = V543Trainer(
        model=model, config=config, source=source,
        labels_v52=real_context["labels_v52"],
        endpoint_labels=real_context["endpoint_labels"],
        out_dir=tmp_path)
    return trainer, model, encoder, config


@pytest.mark.parametrize("track,decoder_id", [
    (TRACK_MOVE, "M0"), (TRACK_PROCESS, "P0")])
def test_real_data_smoke_run(real_context, tmp_path: Path, track: str,
                             decoder_id: str) -> None:
    trainer, model, encoder, config = _run_track(
        real_context, tmp_path, track, decoder_id)
    encoded_before = int(encoder.forward_count)
    metrics = trainer.run()
    # -- the run is a real (short) run, not a formal attempt ---------------
    assert metrics["schema"] == "pesn_v543_train_v1"
    assert metrics["outcome"] == "done"
    assert metrics["updates"] == SMOKE_UPDATES
    assert metrics["is_formal_schedule"] is False
    assert metrics["track"] == track
    assert metrics["decoder_id"] == decoder_id
    assert metrics["identity"]["sampler_spec_sha256"] == \
        V543_SAMPLER_SPEC_SHA256_PINNED
    assert len(metrics["identity"]["encoder_ckpt_sha256"]) == 64
    # -- losses are finite and the frozen formula is reported --------------
    rows = [json.loads(line) for line in
            (tmp_path / "train_log.jsonl").read_text().splitlines()]
    assert len(rows) == SMOKE_UPDATES
    for row in rows:
        assert row["event"] == "train"
        assert row["loss"] is not None and row["loss"] == row["loss"]
        assert row["n_graphs"] > 0
    if track == TRACK_MOVE:
        assert all("l_atom" in row for row in rows)
        assert all("l_rank" not in row for row in rows)
    else:
        assert all("l_rank" in row for row in rows)
        assert all("l_atom" not in row for row in rows), \
            "the PROCESS track must not call the MOVE loss (plan :706)"
        assert metrics["endpoint_pos_weight"]["scope"] == "inner_train_only"
    # -- §4.4: the encoder runs exactly once per encoded batch -------------
    encoded = int(encoder.forward_count) - encoded_before
    frames = sum(int(row["n_graphs"]) for row in rows)
    assert encoded <= SMOKE_UPDATES + SMOKE_VAL_BATCHES + 1, encoded
    assert frames == metrics["updates"] * 32
    for label in (HOOK_LABEL_EDGE_MSG, HOOK_LABEL_NODE_H, HOOK_LABEL_Q0_RAW):
        assert encoder.last_hook_fires[label] == 1
        assert encoder.cumulative_hook_fires[label] == int(
            encoder.forward_count)
    # -- the cache serves a REPEATED inner-val pass for free ---------------
    before = int(encoder.forward_count)
    hits_before = 0 if trainer.cache is None else trainer.cache.hits
    trainer.validate(SMOKE_UPDATES + 1)
    assert int(encoder.forward_count) == before, \
        "a repeated inner-val pass must be served by the cache"
    assert trainer.cache is not None
    assert trainer.cache.hits > hits_before
    assert trainer.cache.stats()["hit_rate"] > 0.0
    # -- checkpoints exist and restore, identity enforced ------------------
    for name in ("checkpoint_best.pt", "checkpoint_last.pt", "metrics.json",
                 "run_records.jsonl"):
        assert (tmp_path / name).is_file(), name
    restored, restored_model, _encoder, _cfg = _run_track(
        real_context, tmp_path / "restored", track, decoder_id)
    stored = restored.resume_from(tmp_path / "checkpoint_last.pt")
    assert stored["track"] == track
    assert stored["decoder_id"] == decoder_id
    assert stored["seed"] == SEED
    # a drifted identity is refused (never a silent fresh start)
    from pes2ts_core.pesn.offline.training.v543_train import (
        V543TrainerError,
        assert_resume_identity_v543,
    )

    payload = torch.load(str(tmp_path / "checkpoint_last.pt"),
                         map_location="cpu", weights_only=False)
    drifted = dict(metrics["identity"])
    drifted["seed"] = 29
    with pytest.raises(V543TrainerError) as error:
        assert_resume_identity_v543(payload, drifted)
    assert error.value.code == "V543_RESUME_IDENTITY"
    print(f"[wp4-realdata] {track}/{decoder_id}: updates={SMOKE_UPDATES} "
          f"encoder_forwards={encoded} frames={frames} "
          f"best_val={metrics['best_val_loss']} "
          f"cache={metrics['cache']['hits']}/"
          f"{metrics['cache']['hits'] + metrics['cache']['misses']}",
          file=sys.stderr, flush=True)


def test_gradient_isolation_is_not_needed_for_the_baselines(
        real_context, tmp_path: Path) -> None:
    """M0/P0 carry no parameter: the isolation assertion is trivially clean."""
    trainer, model, encoder, _config = _run_track(
        real_context, tmp_path, TRACK_MOVE, "M0")
    trainer.run()
    for parameter in encoder.parameters():
        assert parameter.grad is None
        assert parameter.requires_grad is False
    report = model.assert_gradient_isolation(require_active_grad=False)
    assert report["active_trainable_params"] == 0
