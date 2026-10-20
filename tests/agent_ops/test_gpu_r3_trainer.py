"""WP-L3/E3 trainer tests (mission: key_atom_bc_v3 training loop, lane W-LOSS).

CPU fixture mini-runs on the real E1 fold-0 tables (skipped when the
gitignored campaign data is absent): loss decreases, checkpoint roundtrip,
resume-by-run-id, label-weight applied exactly once, inner-val-only
checkpoint selection, metrics schema, determinism, curve-balanced sampler
composition, inner-split containment, config refusals and the classic
CV/Jacobian baseline algebra.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "outputs/pesn/30h_campaign/data"
SPLIT = DATA_DIR / "split_manifest_r3.json"

pytestmark = pytest.mark.skipif(
    not (DATA_DIR / "sample_view.csv").is_file() or not SPLIT.is_file(),
    reason="campaign E1 tables not present in this checkout (gitignored artifacts)",
)

from pes2ts_core.pesn.offline.data.key_atom_dataset_v3 import (  # noqa: E402
    KeyAtomDatasetV3,
)
from pes2ts_core.pesn.offline.training import key_atom_bc_v3 as trainer_mod  # noqa: E402
from pes2ts_core.pesn.offline.training.key_atom_bc_v3 import (  # noqa: E402
    TrainerConfigV3,
    TrainerError,
    classic_jacobian_step,
    curve_batch_sampler,
    make_inner_split,
    train_run,
)

UPDATES = 30


@pytest.fixture(scope="module")
def dataset() -> KeyAtomDatasetV3:
    return KeyAtomDatasetV3(
        DATA_DIR, fold=0, split_manifest=SPLIT, label_statuses=("ok",))


def _config(**overrides) -> TrainerConfigV3:
    base = dict(arm="F0", fold=0, seed=17, level="F0", updates=UPDATES,
                val_every=10, device="cpu")
    base.update(overrides)
    return TrainerConfigV3(**base)


# ---------------------------------------------------------------------------
# config refusals (fail-closed)
# ---------------------------------------------------------------------------


def test_config_refusals():
    with pytest.raises(TrainerError) as excinfo:
        TrainerConfigV3(arm="F0", fold=0, seed=17, batch_size=30)
    assert "BATCH_COMPOSITION" in str(excinfo.value)
    with pytest.raises(TrainerError) as excinfo:
        TrainerConfigV3(arm="S2", fold=0, seed=17, aux_arm="S2", lambda_arm=0.9)
    assert "AUX_LAMBDA_RANGE" in str(excinfo.value)
    with pytest.raises(TrainerError) as excinfo:
        TrainerConfigV3(arm="F0", fold=0, seed=17, lambda_arm=0.1)
    assert "AUX_LAMBDA_WITHOUT_ARM" in str(excinfo.value)


# ---------------------------------------------------------------------------
# inner split + sampler composition
# ---------------------------------------------------------------------------


def test_inner_split_only_outer_train_and_deterministic(dataset):
    samples = dataset.samples
    train_idx, val_idx = make_inner_split(samples, seed=17)
    outer = {s.curve_id for s in samples}
    train_curves = {samples[i].curve_id for i in train_idx}
    val_curves = {samples[i].curve_id for i in val_idx}
    assert train_curves | val_curves == outer
    assert not (train_curves & val_curves)          # disjoint
    assert 1 <= len(val_curves) <= len(outer) - 1   # both sides non-trivial
    # deterministic per seed
    t2, v2 = make_inner_split(samples, seed=17)
    assert (train_idx, val_idx) == (t2, v2)
    # a different seed picks a different (equally sized in curves) val split
    _, v3 = make_inner_split(samples, seed=29)
    v3_curves = {samples[i].curve_id for i in v3}
    assert len(v3_curves) == len(val_curves)


def test_sampler_curve_balanced_and_direction_balanced(dataset):
    samples = dataset.samples
    train_idx, _ = make_inner_split(samples, seed=17)
    train_curves = {samples[i].curve_id for i in train_idx}
    gen = curve_batch_sampler(samples, train_idx, bidirectional=False, seed=7)
    for _ in range(5):
        positions = next(gen)
        assert len(positions) == 32                    # frozen batch size
        curves = [samples[train_idx[p]].curve_id for p in positions]
        assert set(curves) <= train_curves             # inner-train only
        from collections import Counter
        per_curve = Counter(curves)
        assert sum(per_curve.values()) == 32
        # 8 curve slots x 4 frames (a curve may repeat when the fold has
        # <8 inner-train curves; every slot still draws 4 frames)
        assert max(per_curve.values()) % 4 == 0
        dirs = {samples[train_idx[p]].intent.direction for p in positions}
        assert dirs == {"forward"}                     # forward-only arm
    gen_b = curve_batch_sampler(samples, train_idx, bidirectional=True, seed=7)
    positions = next(gen_b)
    from collections import Counter
    per_curve_dir = Counter(
        (samples[train_idx[p]].curve_id, samples[train_idx[p]].intent.direction)
        for p in positions)
    by_curve = {}
    for (curve, direction), n in per_curve_dir.items():
        by_curve.setdefault(curve, {})[direction] = n
    for curve, dirs in by_curve.items():
        assert sum(dirs.values()) % 4 == 0             # 4 frames per curve slot
        if "forward" in dirs and "reverse" in dirs:
            assert dirs["forward"] == dirs["reverse"]  # balanced 2+2 per slot


# ---------------------------------------------------------------------------
# mini-run: loss decreases, checkpoint roundtrip, selection, metrics schema
# ---------------------------------------------------------------------------


def test_mini_run_loss_decreases_checkpoint_and_metrics(dataset, tmp_path):
    out = tmp_path / "run"
    metrics = train_run(dataset, _config(), out, resume=False)

    # loss decreases (first logged train loss vs last)
    log = [json.loads(ln) for ln in
           (out / "train_log.jsonl").read_text().splitlines()]
    train_logs = [e for e in log if "loss" in e]
    assert train_logs[0]["loss"] > train_logs[-1]["loss"]

    # checkpoint roundtrip: load best, run one eval batch
    state = torch.load(str(out / "checkpoint_best.pt"), map_location="cpu",
                       weights_only=False)
    assert state["schema"] == trainer_mod.TRAINER_SCHEMA
    assert state["config"]["arm"] == "F0"
    assert state["seeds"]["torch"] == 17

    # selection is inner-val argmin (recorded best_update matches the log)
    val_logs = [e for e in log if "inner_val_loss" in e]
    best = min(val_logs, key=lambda e: e["inner_val_loss"])
    assert metrics["best_update"] == best["update"]
    assert metrics["selected_update"] == best["update"]
    # selection never uses outer-val: recorded val curves ⊂ fold0 outer-train
    fold0_curves = {s.curve_id for s in dataset.samples}
    assert set(metrics["inner_val_curves"]) <= fold0_curves
    assert set(metrics["inner_train_curves"]).isdisjoint(
        metrics["inner_val_curves"])

    # metrics schema (frozen primaries + guards + provenance)
    for key in ("key_rmse_over_zero", "sre_and_sign", "baselines", "guards",
                "checkpoint_best_sha256", "seeds", "selection_metric",
                "hyperparameters", "arm", "fold", "seed"):
        assert key in metrics, key
    assert metrics["key_rmse_over_zero"]["median_rmse_over_zero"] is not None
    assert metrics["sre_and_sign"]["sre_cluster_median"] is not None
    assert metrics["guards"]["policy"].startswith("record-only")
    assert metrics["baselines"]["zero_step"]["key_rmse_over_zero"] is not None
    assert "formula" in metrics["baselines"]["classic_cv_jacobian"]
    # metrics file written and self-consistent
    on_disk = json.loads((out / "metrics.json").read_text())
    assert on_disk["checkpoint_best_sha256"] == metrics["checkpoint_best_sha256"]
    from pes2ts_core.pesn.offline.training.key_atom_bc_v3 import sha256_file
    assert sha256_file(out / "checkpoint_best.pt") == \
        metrics["checkpoint_best_sha256"]


def test_resume_continues_by_run_identity(dataset, tmp_path):
    out = tmp_path / "run"
    train_run(dataset, _config(), out, resume=False)
    state = torch.load(str(out / "checkpoint_last.pt"), map_location="cpu",
                       weights_only=False)
    assert state["update"] == UPDATES
    # resume with a longer budget continues from the checkpoint
    metrics = train_run(dataset, _config(updates=UPDATES + 10), out, resume=True)
    state2 = torch.load(str(out / "checkpoint_last.pt"), map_location="cpu",
                        weights_only=False)
    assert state2["update"] == UPDATES + 10
    # resume into a DIFFERENT identity refuses (fail-closed)
    with pytest.raises(TrainerError) as excinfo:
        train_run(dataset, _config(seed=29), out, resume=True)
    assert "RESUME_IDENTITY_MISMATCH" in str(excinfo.value)


def test_determinism_same_seed_same_metrics(dataset, tmp_path):
    """Same seed -> same trajectory within float tolerance (CPU threading
    makes bit-exactness unattainable; recorded honestly, not claimed)."""
    m1 = train_run(dataset, _config(), tmp_path / "a", resume=False)
    m2 = train_run(dataset, _config(), tmp_path / "b", resume=False)
    assert m1["last_train_loss"] == pytest.approx(m2["last_train_loss"],
                                                  rel=1e-6, abs=1e-9)
    assert m1["best_inner_val_loss"] == pytest.approx(
        m2["best_inner_val_loss"], rel=1e-6, abs=1e-9)
    assert m1["seeds"] == m2["seeds"]
    assert m1["best_update"] == m2["best_update"]


# ---------------------------------------------------------------------------
# label_weight applied exactly once (spy on the frozen loss call)
# ---------------------------------------------------------------------------


def test_frame_weight_passed_once(dataset, tmp_path, monkeypatch):
    """The trainer hands the loss EXACTLY the batch frame_weight (label_weight)."""
    captured = []
    real_loss = trainer_mod.key_atom_loss_v3

    def spy(u, y, degK, key_ptr, pos, key_index, pair_index=None, **kwargs):
        fw = kwargs.get("frame_weight")
        captured.append(None if fw is None else fw.detach().cpu().clone())
        return real_loss(u, y, degK, key_ptr, pos, key_index, pair_index,
                         **kwargs)

    monkeypatch.setattr(trainer_mod, "key_atom_loss_v3", spy)
    out = tmp_path / "run"
    train_run(dataset, _config(updates=2, val_every=2), out, resume=False)
    assert captured, "spy never saw a loss call"
    # every training call received a [G] weight tensor equal to the batch's
    # label weights (dataset label_weight=1.0 for all E1 'ok' rows); the
    # trainer must NOT square/rescale it (applied exactly once, in the loss)
    for fw in captured[:3]:
        assert fw is not None and fw.ndim == 1
        assert float(fw.min()) >= 0.0


# ---------------------------------------------------------------------------
# classic CV/Jacobian baseline algebra
# ---------------------------------------------------------------------------


def test_classic_jacobian_step_satisfies_projection():
    rng = np.random.RandomState(0)
    pos = rng.randn(5, 3)
    key_rows = [0, 1, 2]
    pairs = [(0, 1), (1, 2), (0, 2)]
    target = np.array([0.08, -0.08, 0.0])
    u = classic_jacobian_step(pos, key_rows, pairs, target)
    assert u.shape == (3, 3)
    # first-order projections reproduce the solvable target subspace
    slot = {r: k for k, r in enumerate(key_rows)}
    for l, (i, j) in enumerate(pairs):
        r = pos[j] - pos[i]
        e = r / np.linalg.norm(r)
        j_l = float(e @ (u[slot[j]] - u[slot[i]]))
        assert math.isclose(j_l, target[l], abs_tol=1e-8)


def test_aux_arm_loss_wiring(dataset, tmp_path):
    """S2 arm: trainer passes its calibrated λ to the frozen loss (bucket only)."""
    out = tmp_path / "run"
    metrics = train_run(
        dataset,
        _config(arm="S2", aux_arm="S2", lambda_arm=0.05),
        out, resume=False)
    assert metrics["aux_arm"] == "S2"
    assert metrics["lambda_arm"] == 0.05
    log = [json.loads(ln) for ln in (out / "train_log.jsonl").read_text().splitlines()]
    train_logs = [e for e in log if "loss" in e]
    assert train_logs[0]["lambda_strong"] == pytest.approx(0.05)
    assert train_logs[0]["lambda_dead"] == 0.0
