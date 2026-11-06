"""V5.4.3 WP9 — dev T7 loader/runner tests (``scripts/agent_ops/
v543_dev_t7.py``).

Fast, synthetic tests by default (no data, no torch forward, no pack read):

* the frame-selection wrapper is the frozen ``inner_val_positions`` rule and is
  deterministic (same positions, same rule block, same digests);
* the loader contract is asserted key-by-key and the shape validator rejects
  malformed frames / missing keys / bad positions;
* the ``DevPackCache`` adapter implements the ``cache[position]`` protocol the
  frozen ``_batch_from_positions`` needs;
* constructing the loader performs NO I/O (no pack open, no ledger write), and
  the execute ``argparse.Namespace`` mirrors ``v543_rollout.main``'s defaults
  with ``mode=dev`` + ``dev_onetime=True`` + the injected loader;
* ``--execute`` and ``--dry-run-loader`` are mutually exclusive (CLI refusal).

The heavy leg is marked ``realdata`` and runs the actual dry run on a frozen
pool slice — it proves the real frame path works and that the dev ledger stays
untouched and the dev pack stays unread::

    PYTHONPATH=. python -m pytest -m realdata \
        tests/agent_ops/test_v543_dev_t7.py -q -s \
        -p no:cacheprovider --basetemp=.pytest_tmp_wp9/dev_t7
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
numpy = pytest.importorskip("numpy")

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.agent_ops import v543_dev_t7 as dev  # noqa: E402
from scripts.agent_ops import v543_rollout as rollout  # noqa: E402

np = numpy


# ---------------------------------------------------------------------------
# synthetic frozen slice
# ---------------------------------------------------------------------------
@dataclass
class _Intent:
    direction: str


@dataclass
class _Sample:
    curve_id: str
    intent: _Intent


class _FakeDataset:
    """Just enough of ``dataset.samples`` for ``inner_val_positions``."""

    def __init__(self, samples):
        self.samples = list(samples)


def _fake_slice(n_curves: int = 3, per_direction: int = 6) -> _FakeDataset:
    samples = []
    for index in range(n_curves):
        for direction in ("forward", "reverse"):
            for _ in range(per_direction):
                samples.append(_Sample(curve_id=f"c{index}",
                                       intent=_Intent(direction)))
    return _FakeDataset(samples)


# ---------------------------------------------------------------------------
# selection rule
# ---------------------------------------------------------------------------
def test_selection_reuses_the_frozen_inner_val_rule() -> None:
    dataset = _fake_slice(n_curves=3, per_direction=6)
    positions = list(range(len(dataset.samples)))
    selected, rule = dev.select_frames(
        dataset, positions, eval_curves=3, frames_per_curve=4,
        split_tag="pool_slice")
    # 3 curves x 2 directions x (4 // 2 = 2 frames per direction) = 12
    assert len(selected) == 12
    assert len({str(dataset.samples[p].curve_id) for p in selected}) == 3
    assert rule["rule_source"] == "scripts.agent_ops.v543_eval.inner_val_positions"
    assert rule["split"] == "pool_slice"
    assert "inner_val_positions" in str(rule["rule"])
    assert int(rule["frames_per_curve_requested"]) == 4
    assert int(rule["n_direction_groups"]) == 6
    # deterministic: a second call is bit-identical
    again, rule_again = dev.select_frames(
        dataset, positions, eval_curves=3, frames_per_curve=4,
        split_tag="pool_slice")
    assert selected == again
    assert rule == rule_again
    assert rollout.canonical_sha256(selected) == \
        rollout.canonical_sha256(again)


def test_selection_refuses_a_zero_budget() -> None:
    dataset = _fake_slice()
    with pytest.raises(dev.DevT7Error) as error:
        dev.select_frames(dataset, [0, 1], eval_curves=0, frames_per_curve=4,
                          split_tag="pool_slice")
    assert error.value.code == "V543_DEV_T7_BUDGET"


# ---------------------------------------------------------------------------
# pack cache adapter
# ---------------------------------------------------------------------------
class _FakeView:
    def __init__(self, rows):
        self.rows = dict(rows)
        self.n_positions = len(self.rows)

    def __getitem__(self, position):
        return self.rows[int(position)]


def test_pack_cache_implements_the_lru_get_protocol() -> None:
    row = {"y_teacher": np.zeros((2, 3), np.float32)}
    view = _FakeView({7: row})
    cache = dev.DevPackCache(view)
    assert cache.n_positions == 1
    assert cache.get(7) is row
    assert cache[7] is row
    assert cache.contains(7) is True
    assert cache.lookups == 2
    # the adapter is what _batch_from_positions needs: cache.get(int(p))
    assert cache.get(7)["y_teacher"].shape == (2, 3)


# ---------------------------------------------------------------------------
# loader construction is I/O free; the execute namespace mirrors the parser
# ---------------------------------------------------------------------------
def _args(**overrides):
    base = dict(
        out=dev.DEFAULT_DEV_OUT, dryrun_out=dev.DEFAULT_DRYRUN_OUT,
        config=rollout.DEFAULT_CONFIG,
        select_report=f"{rollout.CAMPAIGN_ROOT}/{rollout.DEFAULT_SELECT_REL}",
        baseline=None, seeds=None, seed=17,
        dev_ledger=rollout.DEFAULT_DEV_LEDGER_REL, eval_curves=dev.DEV_CURVES_PLAN,
        frames_per_curve=rollout.DEFAULT_FRAMES_PER_CURVE,
        max_steps=rollout.DEFAULT_MAX_STEPS, no_basis=False, no_strata=False,
        actor="v543_dev_t7", pack_dir=None, slice_curves=2,
        no_pack_path_check=False)
    base.update(overrides)
    return argparse.Namespace(**base)


def test_loader_construction_does_not_read_the_pack_or_touch_the_ledger(
        tmp_path: Path) -> None:
    ledger_path = tmp_path / "dev_ledger.jsonl"
    before = int(dev.PACK_OPENS["n"])
    loader = dev.DevFrameLoader(eval_curves=dev.DEV_CURVES_PLAN,
                                frames_per_curve=8)
    assert loader.pack_open_count == 0
    assert int(dev.PACK_OPENS["n"]) == before
    assert not ledger_path.exists()
    assert rollout.DevAccessLedger(ledger_path).already_used() is False


def test_execute_namespace_mirrors_the_rollout_parser_defaults() -> None:
    def loader(**kwargs):                       # pragma: no cover - identity only
        return {}

    namespace = dev.execute_namespace(_args(), loader, seeds=[17])
    assert namespace.mode == rollout.MODE_DEV
    assert namespace.dev_onetime is True
    assert namespace.dev_frame_loader is loader
    assert namespace.seeds == [17]
    assert namespace.eval_curves == dev.DEV_CURVES_PLAN
    assert namespace.frames_per_curve == rollout.DEFAULT_FRAMES_PER_CURVE
    assert namespace.max_steps == rollout.DEFAULT_MAX_STEPS
    assert namespace.select_report == (
        f"{rollout.CAMPAIGN_ROOT}/{rollout.DEFAULT_SELECT_REL}")
    # the §9.5 gate accepts exactly this mode/flag pair
    assert rollout.assert_mode_allowed(namespace.mode,
                                       dev_onetime=namespace.dev_onetime) \
        == rollout.MODE_DEV
    # and the loader the namespace carries is the one we built
    assert rollout.require_dev_frame_loader(
        namespace.dev_frame_loader) is loader


def test_cli_refuses_execute_and_dry_run_together(capsys) -> None:
    code = dev.main(["--execute", "--dry-run-loader"])
    assert code == 2
    assert "mutually exclusive" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# execute wrapper (the rollout is stubbed: no claim, no data)
# ---------------------------------------------------------------------------
def _dev_report(**overrides):
    report = {
        "schema": rollout.ROLLOUT_SCHEMA, "mode": rollout.MODE_DEV,
        "dev_access": {"dev_accessed": True, "ledger_entries_before": 0,
                       "ledger_entries_after": 1, "confirm_read": False},
        "data": {"mode": rollout.MODE_DEV, "dev_accessed": True},
        "per_seed": {},
    }
    report.update(overrides)
    return report


def test_run_execute_passes_the_dev_namespace_and_stamps_the_runner(
        monkeypatch) -> None:
    captured = {}

    def fake_run(namespace):
        captured["namespace"] = namespace
        return _dev_report()

    monkeypatch.setattr(dev.rollout, "run", fake_run)
    report = dev.run_execute(_args())
    namespace = captured["namespace"]
    assert namespace.mode == rollout.MODE_DEV
    assert namespace.dev_onetime is True
    assert isinstance(namespace.dev_frame_loader, dev.DevFrameLoader)
    assert namespace.eval_curves == dev.DEV_CURVES_PLAN
    runner = report["dev_t7_runner"]
    assert runner["schema"] == dev.RUNNER_SCHEMA
    assert runner["loader"]["pack_open_count"] == 0
    assert runner["confirm_read"] is False
    assert runner["new_QC_calls"] == 0
    assert "eval_curves=589" in runner["budget_note"]


def test_run_execute_refuses_a_non_dev_report(monkeypatch) -> None:
    monkeypatch.setattr(dev.rollout, "run",
                        lambda namespace: _dev_report(mode=rollout.MODE_INNER_VAL))
    with pytest.raises(dev.DevT7Error) as error:
        dev.run_execute(_args())
    assert error.value.code == "V543_DEV_T7_REPORT"


def test_run_execute_refuses_a_report_without_the_claim(monkeypatch) -> None:
    monkeypatch.setattr(dev.rollout, "run", lambda namespace: _dev_report(
        dev_access={"dev_accessed": False, "ledger_entries_before": 0,
                    "ledger_entries_after": 0}))
    with pytest.raises(dev.DevT7Error) as error:
        dev.run_execute(_args())
    assert error.value.code == "V543_DEV_T7_REPORT"


# ---------------------------------------------------------------------------
# contract validation (synthetic frames)
# ---------------------------------------------------------------------------
def _eval_frame(*, curve="c0", direction="forward"):
    from pes2ts_core.pesn.offline.evaluation.v543_eval import EvalFrame

    u = np.zeros((3, 3), dtype=np.float64)
    return EvalFrame(
        curve_id=curve, u=u, y=u.copy(),
        pair_slots=np.zeros((1, 2), dtype=np.int64),
        pair_offsets=np.zeros((1, 3), dtype=np.float64),
        direction=direction, stratum="pre_TS", intent_class="both",
        q_pred=0.1, q_true=0.2, q_mask=True, endpoint_logit=0.3,
        endpoint_label=1, endpoint_mask=True, is_start_row=False,
        is_ts_row=False, move_eligible=True, u0=u.copy(), q0_raw=0.1,
        endpoint0_logit=0.3,
        key_v=np.zeros((3, 8, 3), dtype=np.float64))


class _FakeLru:
    def get(self, position):
        return {}


def _valid_payload(positions=(5, 9)):
    frames = [_eval_frame() for _ in positions]
    return {
        "frames": frames, "positions": list(positions),
        "rule": {"rule": "synthetic"}, "dataset": _FakeDataset([_Sample(
            "c0", _Intent("forward"))]),
        "lru": _FakeLru(), "train_pos": [1, 2],
        "labels_v52": object(), "endpoint_labels": object(),
        "positions_sha256": rollout.canonical_sha256(list(positions)),
        "n_frames": len(frames),
    }


def test_validate_payload_accepts_the_contract() -> None:
    summary = dev.validate_payload(_valid_payload())
    assert summary["passed"] is True
    assert summary["n_frames"] == 2 and summary["n_positions"] == 2
    assert summary["directions"] == {"forward": 2}
    assert summary["positions_sha256"] == rollout.canonical_sha256([5, 9])
    assert list(summary["contract_keys_present"]) == list(dev.CONTRACT_KEYS)


def test_validate_payload_rejects_a_missing_contract_key() -> None:
    payload = _valid_payload()
    del payload["endpoint_labels"]
    with pytest.raises(dev.DevT7Error) as error:
        dev.validate_payload(payload)
    assert error.value.code == "V543_DEV_T7_CONTRACT"
    assert "endpoint_labels" in str(error.value)


def test_validate_payload_rejects_unsorted_positions() -> None:
    payload = _valid_payload(positions=(9, 5))
    payload["positions_sha256"] = rollout.canonical_sha256([9, 5])
    with pytest.raises(dev.DevT7Error) as error:
        dev.validate_payload(payload)
    assert error.value.code == "V543_DEV_T7_CONTRACT"
    assert "ascending" in str(error.value)


def test_validate_payload_rejects_a_nonfinite_frame() -> None:
    payload = _valid_payload()
    payload["frames"][0].u[0, 0] = float("nan")
    with pytest.raises(dev.DevT7Error) as error:
        dev.validate_payload(payload)
    assert error.value.code == "V543_DEV_T7_FRAME"


# ---------------------------------------------------------------------------
# realdata: the actual dry run on a frozen pool slice
# ---------------------------------------------------------------------------
def _require_data_view() -> None:
    data = Path("/pes2ts_data/v53mp_50k_data")
    for name in ("data_manifest.json", "teacher_manifest.json",
                 "split_manifest_v53mp.json"):
        if not (data / name).is_file():
            pytest.skip(f"frozen 50K data view unavailable ({data / name})")


@pytest.mark.realdata
def test_dry_run_loads_frames_on_a_pool_slice_without_dev_access(
        tmp_path: Path) -> None:
    _require_data_view()
    ledger_path = tmp_path / "dev_ledger.jsonl"
    before_pack_opens = int(dev.PACK_OPENS["n"])
    args = _args(dev_ledger=str(ledger_path), slice_curves=2,
                 frames_per_curve=4, no_pack_path_check=False)
    report = dev.run_dry_run(args)
    assert report["ok"] is True
    assert report["dev_pack_read"] is False
    assert report["dev_ledger_touched"] is False
    assert int(dev.PACK_OPENS["n"]) == before_pack_opens
    assert not ledger_path.exists()
    assert report["dev_ledger"]["entries_before"] == 0
    assert report["dev_ledger"]["entries_after"] == 0
    validation = report["validation"]
    assert validation["passed"] is True
    assert validation["n_frames"] == validation["n_positions"] == (
        report["pool_slice"]["n_selected_frames"])
    assert validation["n_frames"] > 0
    assert validation["directions"]  # forward and/or reverse, but present
    assert validation["strata"]       # the v541 strata rule ran
    determinism = report["determinism"]
    assert determinism["positions_equal"] is True
    assert determinism["rule_equal"] is True
    assert determinism["positions_sha256"] == \
        determinism["positions_sha256_again"]
    pack_check = report["dev_pack_path_check"]
    assert pack_check is not None
    assert pack_check["files_read"] == []
    assert pack_check["loader_pack_opens"] == 0
    # the live dev split must resolve to the frozen pack directory identity
    assert pack_check["n_dev_curves"] == dev.DEV_CURVES_PLAN
    assert len(str(pack_check["positions_sha256"])) == 64
    assert len(str(pack_check["features_sha256"])) == 64
    # a persisted dry-run document round-trips as JSON
    text = json.dumps(report, sort_keys=True)
    assert json.loads(text)["schema"] == dev.DRYRUN_SCHEMA
