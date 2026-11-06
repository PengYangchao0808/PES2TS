#!/usr/bin/env python3
"""V5.4.3 WP9 — T7 INTEGRATION ROLLOUT (plan §10 :1044–1082, §9.5 :1035–1041).

WHAT THIS IS
------------
Plan §10.1 (:1047–1061) freezes the four T7 combinations::

    | 组合 | MOVE   | PROCESS | 目的                                    |
    | B0   | M0     | P0      | 双基线                                  |
    | BM   | M*     | P0      | MOVE 改进是否带来连续路径收益           |
    | BP   | M0     | P*      | PROCESS 改进是否改善端点判断            |
    | BMP  | M*     | P*      | 双专家组合是否出现意外错误停止/几何退化 |

and §4.3 :226 fixes the assembly rule this module implements: "单 Track
Benchmark 时，未参与当前 Track 的 Refiner 固定为零残差（M0/P0）… 只有 T7
集成才同时启用两个冠军。"  So B0/BM/BP are the ordinary single-active-track
assemblies (the inactive Refiner stays the zero residual) and ONLY BMP binds two
non-baseline experts at once — which the single-active-track
``assemble_v543`` refuses by construction (§7.2 / ``V543_ASSEMBLY_INACTIVE``).
:func:`build_rollout_assembly` therefore builds the assembly directly, binds
each decoder through the frozen §6.1 protocol check, and re-asserts the §6.4
budget (per-candidate band + global 150,000 cap) without pretending there is
one active track.

The outer ``ReactionController`` is §4.6 (:250–263) and §10.3 (:1073–1080),
transcribed verbatim in :func:`move_flag`:

* TS rows ALWAYS keep moving (§4.3 :227 "TS 行必须继续 MOVE");
* the ONLY legal No-Move is "p_end >= tau_end AND chemical_state_ok";
* ``q`` never participates (§4.3 :227 "q 阈值、接近 TS、路径长度均不得触发
  No-Move"; §10.2 :1071 "``q`` 不得出现在 No-Move 决策日志的触发原因中"), and
  :func:`controller_step` records the observed ``q`` under a field name that
  says so instead of using it;
* ``tau_end`` is calibrated on inner-val ONCE with target precision 0.9 (§9.5
  :1040) and never on dev.  This lane does not calibrate: ``tau_end`` defaults
  to ``None`` (§1.5 :105) and a ``None`` tau makes No-Move IMPOSSIBLE — it is
  reported as ``tau_end_uncalibrated_no_move_impossible``, never silently
  treated as "0.5" or "reached".

DATA MODES (§9.5 :1038–1039, §10.2)
-----------------------------------
``inner-val``  (default) the frozen inner-val frame set of ``v543_eval`` — the
               same positions, the same ordering, the same evaluator.  This is
               the only mode that may run during development, and it is what
               the realdata leg of this lane uses.
``dev``        the 589 dev curves, ONCE for the whole campaign (§9.5 :1038
               "只在 T6 冻结冠军后、T7 集成时一次性访问 589 曲线；不用于选择、
               不调阈值、不生成缓存特征；访问账本 append-only").  This mode is
               REFUSED unless ``--dev-onetime`` is passed AND the one-shot
               ledger is still empty.  The first visit is written to the
               append-only ledger with ``flock`` BEFORE any dev payload is
               read (fail-closed: a partially finished dev read still consumes
               the one shot), and a second visit is the typed
               ``V543_ROLLOUT_DEV_ALREADY_USED`` refusal.
``confirm``    NEVER.  §9.5 :1039 "**confirm**：永封存；``confirm_read=false``；
               任何读取都是 ``invalid_run``" — the mode exists only so the
               refusal is testable, and it refuses unconditionally.

ROLLOUT SEMANTICS (§10.2 :1063–1071)
------------------------------------
The plan forbids physical relaxation in this stage ("物理松弛：**不做**；与
reference-normalized rollout 分开报告").  The rollout implemented here is the
REFERENCE-NORMALIZED one: per curve/direction the frozen reference frame
sequence is the walk path, the stage model is queried on each reference frame
(one encoder forward per configuration — §4.4 boundary 1 is never violated by
re-encoding the same frame), and the controller decides whether the walk
continues.  Consequently ``collision`` has no measurement (there is no physics
step) and is reported as such, and endpoint arrival is ``not_applicable`` while
``tau_end`` is ``None``.  A number that was not measured is never invented.

OUTPUT
------
``pesn_v543_rollout_report_v1``: per stage the frozen evaluator's FULL
``pesn_v543_eval_report_v1`` document (``evaluate_v543`` on the same frame set,
so a candidate/baseline pair is frame-for-frame pairable, §9.4 :1019), the
rollout walk metrics of §10.2, the controller log summary, the champion
resolution block (select report path + SHA, per track champion-or-retained,
checkpoint digests), the §10.1 integration-gate block, and the dev-access
bookkeeping (``confirm_read=false``, ``new_QC_calls=0`` always).

B0 CONSISTENCY ASSERTION
------------------------
B0 IS the frozen M0/P0 baseline.  :func:`assert_b0_matches_frozen_baseline`
compares B0's per-seed ``per_curve``/``pooled``/``validity`` blocks against
``audit/m0_p0_baseline_v543_eval.json`` and refuses with
``V543_ROLLOUT_B0_BASELINE_MISMATCH`` on any drift; when the budget differs
from the frozen document's, the check is reported as ``not_checked`` with the
reason instead of being faked.

CLI::

    PYTHONPATH=. python scripts/agent_ops/v543_rollout.py \
        --out outputs/pesn/v543_dual_decoder_50k/reports/t7_rollout.json

    # the dev one-shot (WP9 only; NOT executed by this lane)
    PYTHONPATH=. python scripts/agent_ops/v543_rollout.py --mode dev --dev-onetime

Discipline: no V5.4.2 / V5.3-MP file is modified (plan §11.3), nothing under
``runs/``, ``run_ledger.jsonl`` or ``plan/runplan*.json`` is written, no
training is started, no ``torch.compile`` (§11.3 禁止), every constant is
``Final`` and every refusal is typed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Final, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:            # pragma: no cover - import aid
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np                                              # noqa: E402
import torch                                                    # noqa: E402

from pes2ts_core.pesn.models.v543.contracts import (            # noqa: E402
    ENCODER_FORWARD_CALLS_PER_VIEW,
    V543Error,
)
from pes2ts_core.pesn.models.v543.registry import (             # noqa: E402
    BASELINE_DECODER_BY_TRACK,
    PARAM_GLOBAL_CAP_V543,
    TRACK_MOVE,
    TRACK_PROCESS,
    TRACKS_V543,
    assert_budget,
    build_baseline,
)
from pes2ts_core.pesn.models.v543.assembly import (             # noqa: E402
    DecoderBinding,
    V543Assembly,
    bind_decoder,
    check_move_output,
    check_process_output,
    unpermute_to_key_rows,
)

# ---------------------------------------------------------------------------
# frozen identity of THIS document
# ---------------------------------------------------------------------------
#: Schema of the rollout report.
ROLLOUT_SCHEMA: Final[str] = "pesn_v543_rollout_report_v1"
#: Registered version of the rollout lane.
ROLLOUT_VERSION: Final[str] = "pesn_v543_rollout_v1"
#: The plan lines this document implements (quoted, never paraphrased away).
PLAN_SECTION: Final[str] = (
    "§10.1 :1047-1061 (T7 四组合 B0/BM/BP/BMP 与集成门) / §10.2 :1063-1071 "
    "(rollout 协议：起点集/预算/tau/异常/无物理松弛/指标/No-Move 日志) / "
    "§10.3 :1073-1080 (Controller 验收 test_v543_no_move_contract) / "
    "§4.6 :250-263 (move_flag 语义) / §4.3 :226-227 (T7 同时启用两冠军；"
    "No-Move 只能由校准端点+化学状态一致触发) / §9.5 :1035-1041 "
    "(dev 一次性/confirm 永封/QC=0)")

#: The four §10.1 combinations, in the plan's own order.
STAGES: Final[tuple[str, ...]] = ("B0", "BM", "BP", "BMP")
#: Which track is the CHAMPION side of each combination (§10.1 table).
STAGE_CHAMPION_TRACKS: Final[Mapping[str, tuple[str, ...]]] = {
    "B0": (),
    "BM": (TRACK_MOVE,),
    "BP": (TRACK_PROCESS,),
    "BMP": (TRACK_MOVE, TRACK_PROCESS),
}
#: §10.1 :1049–1054 — the purpose column, carried into the report.
STAGE_PURPOSE: Final[Mapping[str, str]] = {
    "B0": "double baseline (M0 + P0)",
    "BM": "does the MOVE gain buy continuous-path benefit",
    "BP": "does the PROCESS gain improve endpoint judgement",
    "BMP": "do two experts together produce an unexpected misstop / "
           "geometric degradation",
}
#: §10.1 :1059 — the integration gate BMP must satisfy.
INTEGRATION_GATE_RULES: Final[tuple[str, ...]] = (
    "h2_not_degraded_on_either_track",
    "misstop_start_or_ts_not_increased",
    "collision_not_increased",
    "p95_latency_le_2x_A0",
)
#: §10.1 :1059 — the p95 latency cap relative to A0 (B0 here).
MAX_P95_LATENCY_RATIO: Final[float] = 2.0

#: Data modes (§9.5 :1038–1039).
MODE_INNER_VAL: Final[str] = "inner_val"
MODE_DEV: Final[str] = "dev"
MODE_CONFIRM: Final[str] = "confirm"
MODES: Final[tuple[str, ...]] = (MODE_INNER_VAL, MODE_DEV, MODE_CONFIRM)

#: Campaign root (shared with v543_eval / v543_stats / v543_formal).
CAMPAIGN_ROOT: Final[str] = "outputs/pesn/v543_dual_decoder_50k"
#: The frozen M0/P0 baseline document (§9.5 :1037; same path as v543_stats).
DEFAULT_BASELINE_REL: Final[str] = "audit/m0_p0_baseline_v543_eval.json"
#: The T6 select report (v543_stats ``select`` product).
DEFAULT_SELECT_REL: Final[str] = "reports/track_champions.json"
#: The append-only ONE-SHOT dev ledger of this lane (never ``run_ledger.jsonl``).
DEFAULT_DEV_LEDGER_REL: Final[str] = "audit/dev_rollout_access_ledger.jsonl"
#: Default report path.
DEFAULT_OUT: Final[str] = "reports/t7_rollout.json"
#: The campaign config (pinned checkpoint digests live there).
DEFAULT_CONFIG: Final[str] = "config/campaigns/pes2ts_v543_dual_decoder_50k.yaml"
#: Run directory template of a formal candidate run (§8.5 archiving layout).
RUN_ID_TEMPLATE: Final[str] = "V543_{decoder_id}_S{seed}"
#: Where the champion's checkpoint lives inside its run directory.
CHAMPION_CHECKPOINT_NAME: Final[str] = "checkpoint_best.pt"
#: Inner-val budget defaults — the §1.4 :91 cap, identical to ``v543_eval``.
DEFAULT_EVAL_CURVES: Final[int] = 1024
DEFAULT_FRAMES_PER_CURVE: Final[int] = 8
FRAMES_PER_BATCH: Final[int] = 32
INNER_VAL_CAP: Final[int] = 8192

#: Reference-normalized walk budget (§10.2 :1066 "固定步数/固定 wall").
DEFAULT_MAX_STEPS: Final[int] = 8
MAX_STEPS_CAP: Final[int] = 64

#: §4.6 :253 — every legal No-Move reason.  ``q`` is absent BY CONSTRUCTION
#: (§10.2 :1071) and :func:`controller_step` asserts it.
NO_MOVE_REASONS: Final[tuple[str, ...]] = (
    "endpoint_probability_above_tau_and_chemical_state_consistent",)
#: §10.3 :1074–1080 acceptance vocabulary of ``test_v543_no_move_contract``.
CONTROLLER_ACCEPTANCE_CASES: Final[tuple[str, ...]] = (
    "q_positive_low_p_end_must_move",
    "q_negative_high_p_end_consistent_state_may_no_move",
    "ts_row_always_moves",
    "high_p_end_inconsistent_state_must_move",
)

# ---------------------------------------------------------------------------
# typed refusal codes (every refusal is machine-classifiable)
# ---------------------------------------------------------------------------
CODE_MODE: Final[str] = "V543_ROLLOUT_MODE"
CODE_DEV_FLAG_REQUIRED: Final[str] = "V543_ROLLOUT_DEV_FLAG_REQUIRED"
CODE_DEV_ALREADY_USED: Final[str] = "V543_ROLLOUT_DEV_ALREADY_USED"
CODE_DEV_LEDGER: Final[str] = "V543_ROLLOUT_DEV_LEDGER"
CODE_DEV_LOADER_UNAVAILABLE: Final[str] = "V543_ROLLOUT_DEV_LOADER_UNAVAILABLE"
CODE_CONFIRM_SEALED: Final[str] = "V543_ROLLOUT_CONFIRM_SEALED"
CODE_SELECT_MISSING: Final[str] = "V543_ROLLOUT_SELECT_MISSING"
CODE_SELECT_SCHEMA: Final[str] = "V543_ROLLOUT_SELECT_SCHEMA"
CODE_CHAMPION_CHECKPOINT: Final[str] = "V543_ROLLOUT_CHAMPION_CHECKPOINT"
CODE_CHAMPION_CHECKPOINT_SHA: Final[str] = "V543_ROLLOUT_CHAMPION_CHECKPOINT_SHA"
CODE_CHAMPION_STATE: Final[str] = "V543_ROLLOUT_CHAMPION_STATE"
CODE_STAGE: Final[str] = "V543_ROLLOUT_STAGE"
CODE_ASSEMBLY: Final[str] = "V543_ROLLOUT_ASSEMBLY"
CODE_BUDGET: Final[str] = "V543_ROLLOUT_BUDGET"
CODE_FRAMES: Final[str] = "V543_ROLLOUT_FRAMES"
CODE_BASELINE_MISSING: Final[str] = "V543_ROLLOUT_BASELINE_MISSING"
CODE_B0_BASELINE_MISMATCH: Final[str] = "V543_ROLLOUT_B0_BASELINE_MISMATCH"
CODE_CONTROLLER: Final[str] = "V543_ROLLOUT_CONTROLLER"


class RolloutError(V543Error):
    """T7 rollout refusal (mode gate, champion resolution, frames, B0 drift)."""


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(obj: Any) -> str:
    """The frozen identity recipe (``v542.identity.canonical_sha256``)."""
    from pes2ts_core.pesn.models.v542.identity import canonical_sha256 as _c

    return _c(obj)


def _utc_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path, *, what: str, code: str) -> dict[str, Any]:
    if not path.is_file():
        raise RolloutError(code, f"{what} not found: {path}")
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except Exception as error:                    # noqa: BLE001 - typed
        raise RolloutError(code, f"{what} unreadable ({path}): {error}") from error
    if not isinstance(document, dict):
        raise RolloutError(code, f"{what} is not a JSON object: {path}")
    return document


def _sigmoid(logit: Any) -> float | None:
    """``p_end = σ(e)`` (§4.3 :224) — computed, never a second sigmoid."""
    if logit is None:
        return None
    value = float(logit)
    if not math.isfinite(value):
        return None
    return float(1.0 / (1.0 + math.exp(-value)))


# ---------------------------------------------------------------------------
# §4.6 / §10.3 — the non-learned ReactionController
# ---------------------------------------------------------------------------
def move_flag(p_end: float | None, tau_end: float | None,
              chemical_state_ok: bool, is_ts: bool) -> bool:
    """The plan §4.6 :253–259 function, transcribed VERBATIM.

    ``if is_ts: return True`` / ``if p_end >= tau_end and chemical_state_ok:
    return False`` / ``return True  # q 不参与``.

    ``tau_end=None`` means "not calibrated" (§1.5 :105): the comparison cannot
    be made, so the function refuses to invent a threshold and returns ``True``
    (keep moving).  ``p_end=None`` (a non-finite logit) behaves the same way.
    """
    if is_ts:
        return True
    if tau_end is None or p_end is None:
        return True
    if not math.isfinite(float(tau_end)):
        return True
    if float(p_end) >= float(tau_end) and bool(chemical_state_ok):
        return False
    return True


def controller_step(*, p_end: float | None, tau_end: float | None,
                    chemical_state_ok: bool, is_ts: bool,
                    q_observed: float | None = None) -> dict[str, Any]:
    """One controller decision + the §10.2 :1071 audit record.

    The returned record ALWAYS carries the three fields §10.2 :1071 demands for
    any No-Move (``p_end``, ``chemical_state_ok``, ``is_ts``), the frozen
    ``tau_end`` it used, a reason drawn from a closed vocabulary, and the
    observed ``q`` under a field whose NAME states it is not a trigger
    (``q_observed_not_a_trigger``).  A No-Move whose reason mentions ``q`` is a
    hard refusal (``V543_ROLLOUT_CONTROLLER``), so the §10.2 discipline is
    structural, not a comment.
    """
    move = move_flag(p_end, tau_end, bool(chemical_state_ok), bool(is_ts))
    if is_ts:
        reason = "ts_row_must_move"
        triggered_by = ["is_ts"]
    elif tau_end is None:
        reason = "tau_end_uncalibrated_no_move_impossible"
        triggered_by = ["tau_end"]
    elif p_end is None:
        reason = "endpoint_probability_unavailable_must_move"
        triggered_by = ["p_end"]
    elif not move:
        reason = NO_MOVE_REASONS[0]
        triggered_by = ["p_end", "tau_end", "chemical_state_ok"]
    elif bool(chemical_state_ok):
        reason = "endpoint_probability_below_tau_continue_moving"
        triggered_by = ["p_end", "tau_end"]
    else:
        # §10.3 :1079 — "p_end 高但化学状态不一致 → 必须 MOVE"
        reason = "chemical_state_not_consistent_must_move"
        triggered_by = ["chemical_state_ok"]
    if not move and reason not in NO_MOVE_REASONS:
        raise RolloutError(
            CODE_CONTROLLER,
            f"a No-Move must use one of {list(NO_MOVE_REASONS)}, got "
            f"{reason!r} (plan §4.6 :256)")
    if not move and "q" in reason:
        raise RolloutError(
            CODE_CONTROLLER,
            f"the No-Move reason {reason!r} mentions q; plan §10.2 :1071 "
            "forbids q in a No-Move trigger reason")
    return {
        "move": bool(move),
        "move_flag": bool(move),
        "reason": reason,
        "triggered_by": list(triggered_by),
        "inputs": {
            "p_end": None if p_end is None else float(p_end),
            "tau_end": None if tau_end is None else float(tau_end),
            "chemical_state_ok": bool(chemical_state_ok),
            "is_ts": bool(is_ts),
        },
        "q_observed_not_a_trigger": (None if q_observed is None
                                     else float(q_observed)),
        "q_used_as_trigger": False,
        "plan_lines": "§4.6 :253-259 / §4.3 :227 / §10.2 :1071 / §10.3",
    }


def default_chemical_state_check(frame: Any) -> tuple[bool, str]:
    """The §4.6 :262 outer chemical verification — UNAVAILABLE in this lane.

    "``chemical_state_ok`` 由外层合法端点/键级核验给出，不由模型 ``q`` 给出."
    No legal-endpoint / bond-level verifier exists in V5.4.3 (it belongs to the
    V6 physics loop), so the honest answer is ``False`` with a recorded reason:
    a No-Move is then impossible, which is exactly what an unverified chemical
    state must imply.  A caller that owns a real verifier injects it through
    ``chemical_state_check=`` and its provenance lands in the report.
    """
    return False, "unavailable_no_legal_endpoint_or_bond_verifier_in_v543"


# ---------------------------------------------------------------------------
# stage assembly (§10.1) — the champion/baseline matrix
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class StagePlan:
    """One §10.1 combination: which track uses the champion."""

    stage: str
    champion_tracks: tuple[str, ...]

    def uses_champion(self, track: str) -> bool:
        return str(track) in self.champion_tracks

    def describe(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "champion_tracks": list(self.champion_tracks),
            "purpose": STAGE_PURPOSE[self.stage],
            "move": "M*" if self.uses_champion(TRACK_MOVE) else "M0",
            "process": "P*" if self.uses_champion(TRACK_PROCESS) else "P0",
        }


def stage_plans() -> dict[str, StagePlan]:
    """The four §10.1 combinations (order fixed by :data:`STAGES`)."""
    return {stage: StagePlan(stage=stage,
                             champion_tracks=STAGE_CHAMPION_TRACKS[stage])
            for stage in STAGES}


def stage_plan(stage: str) -> StagePlan:
    if str(stage) not in STAGES:
        raise RolloutError(
            CODE_STAGE, f"stage must be one of {list(STAGES)}, got {stage!r} "
                        "(plan §10.1 :1047-1061)")
    return stage_plans()[str(stage)]


# ---------------------------------------------------------------------------
# champion resolution (select report -> decoder id + checkpoint)
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class ChampionResolution:
    """One track's T7 side: the champion, or the retained M0/P0 baseline."""

    track: str
    decoder_id: str
    source: str                       # "champion" | "retained_baseline"
    checkpoint_path: str | None
    checkpoint_sha256: str | None
    select_report: str | None
    select_report_sha256: str | None
    seed: int

    @property
    def is_baseline(self) -> bool:
        return self.source == "retained_baseline"

    def describe(self) -> dict[str, Any]:
        return {
            "track": self.track,
            "decoder_id": self.decoder_id,
            "source": self.source,
            "checkpoint_path": self.checkpoint_path,
            "checkpoint_sha256": self.checkpoint_sha256,
            "select_report": self.select_report,
            "select_report_sha256": self.select_report_sha256,
            "seed": int(self.seed),
            "is_baseline": bool(self.is_baseline),
        }


def load_select_report(path: Any) -> dict[str, Any]:
    """Read the T6 select report (``pesn_v543_stats_select_report_v1``)."""
    target = Path(str(path))
    if not target.is_absolute():
        target = PROJECT_ROOT / target
    document = _read_json(target, what="select report", code=CODE_SELECT_MISSING)
    schema = str(document.get("schema", ""))
    if schema != "pesn_v543_stats_select_report_v1":
        raise RolloutError(
            CODE_SELECT_SCHEMA,
            f"select report schema {schema!r} is not "
            "'pesn_v543_stats_select_report_v1' (the v543_stats select "
            f"product): {target}")
    tracks = document.get("tracks")
    if not isinstance(tracks, Mapping):
        raise RolloutError(
            CODE_SELECT_SCHEMA,
            f"select report {target} carries no 'tracks' mapping; champions "
            "cannot be resolved (plan §10.1 :1049)")
    return {"document": document, "path": target,
            "sha256": sha256_file(target)}


def _champion_id(entry: Any) -> str | None:
    """The champion decoder id of one track entry (``None`` = retained base)."""
    if entry is None:
        return None
    if isinstance(entry, str):
        return entry or None
    if isinstance(entry, Mapping):
        for key in ("decoder_id", "champion", "id"):
            value = entry.get(key)
            if isinstance(value, str) and value:
                return value
        return None
    raise RolloutError(
        CODE_SELECT_SCHEMA,
        f"a track champion entry must be a string/mapping/null, got "
        f"{type(entry).__name__}")


def resolve_champion(select: Mapping[str, Any], track: str, *, seed: int,
                     campaign_root: Any = CAMPAIGN_ROOT,
                     override_decoder: str | None = None,
                     override_checkpoint: Any | None = None
                     ) -> ChampionResolution:
    """Resolve one track's T7 decoder (champion or retained baseline).

    A track whose select entry carries ``champion: null`` retains its M0/P0
    baseline (§8.2 :816) — that is a normal negative result and is what the
    committed select report currently says.  When a champion exists its
    checkpoint is the formal run's ``runs/V543_<id>_S<seed>/checkpoint_best.pt``
    and its digest is recorded (a missing checkpoint is the typed
    ``V543_ROLLOUT_CHAMPION_CHECKPOINT`` refusal — a champion without weights
    cannot be rolled out).
    """
    if str(track) not in TRACKS_V543:
        raise RolloutError(CODE_STAGE, f"unknown track {track!r}")
    document = select["document"]
    tracks = document.get("tracks") or {}
    entry = tracks.get(str(track))
    if not isinstance(entry, Mapping) and entry is not None:
        raise RolloutError(
            CODE_SELECT_SCHEMA, f"tracks.{track} is not a mapping: {entry!r}")
    entry = entry or {}
    champion_id = _champion_id(entry.get("champion"))
    if override_decoder is not None:
        champion_id = str(override_decoder)
        source = "override"
    else:
        source = "champion" if champion_id else "retained_baseline"
    baseline_id = BASELINE_DECODER_BY_TRACK[str(track)]
    if not champion_id:
        return ChampionResolution(
            track=str(track), decoder_id=baseline_id, source="retained_baseline",
            checkpoint_path=None, checkpoint_sha256=None,
            select_report=str(select["path"]),
            select_report_sha256=str(select["sha256"]), seed=int(seed))
    if str(champion_id) == baseline_id:
        # a select report that names M0/P0 explicitly is the retained baseline
        return ChampionResolution(
            track=str(track), decoder_id=baseline_id,
            source="retained_baseline", checkpoint_path=None,
            checkpoint_sha256=None, select_report=str(select["path"]),
            select_report_sha256=str(select["sha256"]), seed=int(seed))

    if override_checkpoint is not None:
        path = Path(str(override_checkpoint))
        if not path.is_absolute():
            path = PROJECT_ROOT / path
    else:
        run_id = RUN_ID_TEMPLATE.format(decoder_id=champion_id, seed=int(seed))
        path = (Path(str(campaign_root)) / "runs" / run_id /
                CHAMPION_CHECKPOINT_NAME)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
    if not path.is_file():
        raise RolloutError(
            CODE_CHAMPION_CHECKPOINT,
            f"tracks.{track}.champion={champion_id!r} is selected but its "
            f"checkpoint is missing ({path}); T7 cannot roll out a champion "
            "without weights (plan §10.1 :1049-1054)")
    return ChampionResolution(
        track=str(track), decoder_id=str(champion_id), source=str(source),
        checkpoint_path=str(path), checkpoint_sha256=sha256_file(path),
        select_report=str(select["path"]),
        select_report_sha256=str(select["sha256"]), seed=int(seed))


# ---------------------------------------------------------------------------
# decoder construction / loading
# ---------------------------------------------------------------------------
def candidate_registry() -> Any:
    """A FRESH registry with M0/P0 + the seven formal candidates (§11.1)."""
    from scripts.agent_ops.v543_campaign import candidate_registry as _fresh

    return _fresh()


def build_champion_decoder(resolution: ChampionResolution, *, registry: Any = None
                           ) -> Any:
    """Construct the decoder for ``resolution`` (baseline ids: M0/P0)."""
    reg = registry or candidate_registry()
    decoder = reg.build(str(resolution.decoder_id), track=str(resolution.track))
    if resolution.checkpoint_path is None:
        return decoder
    load_decoder_checkpoint(decoder, resolution.checkpoint_path,
                            decoder_id=str(resolution.decoder_id),
                            expected_sha256=resolution.checkpoint_sha256)
    return decoder


def load_decoder_checkpoint(decoder: Any, path: Any, *, decoder_id: str,
                            expected_sha256: str | None = None
                            ) -> dict[str, Any]:
    """Strict-load the decoder ``state_dict`` of a formal run checkpoint.

    The trainer payload (``v543_train.V543_TRAIN_SCHEMA``) stores the ACTIVE
    decoder's ``state_dict`` under ``payload["model"]`` (see
    ``v543_train.V543Trainer._checkpoint_payload``).  The load is
    ``strict=True`` and every key is re-verified with ``torch.equal`` — no
    ``strict=False`` anywhere (plan §11.3 禁止).
    """
    target = Path(str(path))
    if not target.is_file():
        raise RolloutError(CODE_CHAMPION_CHECKPOINT,
                           f"champion checkpoint not found: {target}")
    actual_sha = sha256_file(target)
    if expected_sha256 and str(expected_sha256) != actual_sha:
        raise RolloutError(
            CODE_CHAMPION_CHECKPOINT_SHA,
            f"{decoder_id}: checkpoint sha256 {actual_sha} != resolved "
            f"{expected_sha256} ({target}); the file changed between "
            "resolution and load")
    payload = torch.load(str(target), map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise RolloutError(CODE_CHAMPION_STATE,
                           f"{decoder_id}: checkpoint payload is not a mapping")
    schema = str(payload.get("schema", ""))
    if schema != "pesn_v543_train_v1":
        raise RolloutError(
            CODE_CHAMPION_STATE,
            f"{decoder_id}: checkpoint schema {schema!r} is not "
            "'pesn_v543_train_v1' ({target})")
    recorded_id = None
    identity = payload.get("identity")
    if isinstance(identity, Mapping):
        recorded_id = identity.get("decoder_id")
    if recorded_id is not None and str(recorded_id) != str(decoder_id):
        raise RolloutError(
            CODE_CHAMPION_STATE,
            f"{decoder_id}: checkpoint records decoder_id={recorded_id!r} "
            f"({target}); refusing a cross-decoder load")
    state = payload.get("model")
    if not isinstance(state, Mapping):
        raise RolloutError(CODE_CHAMPION_STATE,
                           f"{decoder_id}: checkpoint has no 'model' state dict")
    decoder.load_state_dict(dict(state), strict=True)
    target_keys = set(decoder.state_dict())
    equal_checked = 0
    for key, tensor in decoder.state_dict().items():
        if not bool(torch.equal(tensor, state[key])):
            raise RolloutError(
                CODE_CHAMPION_STATE,
                f"{decoder_id}: key {key!r} differs from the checkpoint after "
                "strict load")
        equal_checked += 1
    return {
        "checkpoint_path": str(target),
        "checkpoint_sha256": actual_sha,
        "schema": schema,
        "decoder_id": str(decoder_id),
        "n_state_keys": len(target_keys),
        "torch_equal_verified": int(equal_checked),
        "strict": True,
        "update": int(payload.get("update", -1)),
        "best_val_loss": payload.get("best_val_loss"),
        "identity_sha256": payload.get("identity_sha256"),
    }


# ---------------------------------------------------------------------------
# the T7 assembly (§4.3 :226 — only T7 enables BOTH champions)
# ---------------------------------------------------------------------------
def build_rollout_assembly(encoder: Any, *, bridge: Any,
                           move: Any, move_id: str,
                           process: Any, process_id: str) -> V543Assembly:
    """Bind encoder + Bridge + the two stage Refiners for a T7 combination.

    ``assemble_v543`` is deliberately NOT used: it asserts that the inactive
    track IS its zero-residual baseline (§4.3 :226 / ``V543_ASSEMBLY_INACTIVE``)
    because a *benchmark* run must never let a random second decoder pollute the
    measured track.  T7 is the one place the plan turns both champions ON
    ("只有 T7 集成才同时启用两个冠军"), so this function performs the same
    boundary checks that still apply — §6.1 protocol, measured §6.4 budget, the
    encoder freeze and the "one encoder forward per configuration" counter —
    and records a non-learning role per decoder.
    """
    from pes2ts_core.pesn.models.v543.coarse_encoder import CoarseEncoder
    from pes2ts_core.pesn.models.v543.latent_bridge import LatentBridge

    if not isinstance(encoder, CoarseEncoder):
        raise RolloutError(
            CODE_ASSEMBLY,
            f"encoder must be a CoarseEncoder, got {type(encoder).__name__}")
    if not isinstance(bridge, LatentBridge):
        raise RolloutError(
            CODE_ASSEMBLY, f"bridge must be a LatentBridge, got "
                           f"{type(bridge).__name__}")
    move_binding = bind_decoder(move, decoder_id=str(move_id),
                                track=TRACK_MOVE, role="active")
    process_binding = bind_decoder(process, decoder_id=str(process_id),
                                   track=TRACK_PROCESS, role="active")
    assembly = V543Assembly(encoder, bridge, move=move_binding,
                            process=process_binding,
                            active_track=TRACK_MOVE)
    assembly.freeze_encoder()
    assembly.assert_encoder_frozen()
    _assert_rollout_budget(assembly)
    return assembly


def _assert_rollout_budget(assembly: V543Assembly) -> dict[str, Any]:
    """§6.4 budget for a TWO-expert T7 assembly (no single active track)."""
    encoder_joint = int(sum(p.numel() for p in assembly.encoder.parameters()))
    bridge_params = int(sum(p.numel() for p in assembly.bridge.parameters()))
    move_params = int(assembly.move.n_trainable_params)
    process_params = int(assembly.process.n_trainable_params)
    total = encoder_joint + move_params + process_params
    if bridge_params != 0:
        raise RolloutError(
            CODE_BUDGET,
            f"the Bridge must be parameter free (plan §4.1 :185), found "
            f"{bridge_params} trainable parameters")
    for binding in (assembly.move, assembly.process):
        try:
            assert_budget(binding.n_trainable_params,
                          decoder_id=binding.decoder_id, track=binding.track)
        except V543Error as error:
            raise RolloutError(
                getattr(error, "code", CODE_BUDGET),
                f"{binding.decoder_id}: {getattr(error, 'detail', error)}"
            ) from error
    if total > PARAM_GLOBAL_CAP_V543:
        raise RolloutError(
            CODE_BUDGET,
            f"total model parameters {total} exceed the global hard cap "
            f"{PARAM_GLOBAL_CAP_V543} (plan §6.4 :568)")
    return {
        "encoder_joint": encoder_joint,
        "bridge_trainable": bridge_params,
        "move_params": move_params,
        "process_params": process_params,
        "total_model_params": total,
        "global_hard_cap": PARAM_GLOBAL_CAP_V543,
        "note": ("T7 enables BOTH refiners (plan §4.3 :226), so there is no "
                 "'inactive zero-residual' slot; the per-candidate band and the "
                 "global cap are still asserted with the measured counts "
                 "(plan §6.4 :569)"),
    }


# ---------------------------------------------------------------------------
# dev one-shot gate (§9.5 :1038-1039) — mirrors the v541 dev_access ledger
# ---------------------------------------------------------------------------
#: Schema of one ledger line (append-only JSONL, byte-comparable per entry).
DEV_LEDGER_SCHEMA: Final[str] = "v543_dev_rollout_access_ledger_v1"


def assert_mode_allowed(mode: str, *, dev_onetime: bool) -> str:
    """The §9.5 :1038–1039 mode gate (typed refusals, no side effect).

    * ``confirm`` → ALWAYS ``V543_ROLLOUT_CONFIRM_SEALED``;
    * ``dev`` without ``--dev-onetime`` → ``V543_ROLLOUT_DEV_FLAG_REQUIRED``;
    * anything else → ``V543_ROLLOUT_MODE``.
    """
    text = str(mode)
    if text not in MODES:
        raise RolloutError(
            CODE_MODE, f"mode must be one of {list(MODES)}, got {text!r}")
    if text == MODE_CONFIRM:
        raise RolloutError(
            CODE_CONFIRM_SEALED,
            "confirm is sealed forever (plan §9.5 :1039 'confirm 永封存; "
            "confirm_read=false; 任何读取都是 invalid_run') — no flag opens it")
    if text == MODE_DEV and not dev_onetime:
        raise RolloutError(
            CODE_DEV_FLAG_REQUIRED,
            "--mode dev requires the explicit --dev-onetime flag: T7 accesses "
            "the 589 dev curves ONCE for the whole campaign (plan §9.5 :1038)")
    return text


class DevAccessLedger:
    """The append-only ONE-SHOT dev access ledger of the T7 rollout.

    Mirrors ``scripts/campaigns/v541_dev_evaluate.py`` (:1002/:1027–1037): one
    JSONL line per access, written under ``fcntl.flock`` and ``fsync``, never
    rewritten.  The file is a NEW file
    (``audit/dev_rollout_access_ledger.jsonl``) — ``run_ledger.jsonl`` is owned
    by the formal runner and is never touched by this lane.
    """

    def __init__(self, path: Any) -> None:
        self.path = Path(str(path))
        if not self.path.is_absolute():
            self.path = PROJECT_ROOT / self.path

    # -- reading ------------------------------------------------------------
    def entries(self) -> list[dict[str, Any]]:
        if not self.path.is_file():
            return []
        rows: list[dict[str, Any]] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            text = line.strip()
            if not text:
                continue
            try:
                rows.append(json.loads(text))
            except Exception as error:            # noqa: BLE001 - typed
                raise RolloutError(
                    CODE_DEV_LEDGER,
                    f"ledger line is not valid JSON ({self.path}): {error}"
                ) from error
        return rows

    def dev_uses(self) -> list[dict[str, Any]]:
        return [row for row in self.entries()
                if str(row.get("split")) == MODE_DEV]

    def already_used(self) -> bool:
        return bool(self.dev_uses())

    # -- writing ------------------------------------------------------------
    def claim(self, *, actor: str, purpose: str, seed: int, mode: str,
              budget: Mapping[str, Any], extra: Mapping[str, Any] | None = None
              ) -> dict[str, Any]:
        """Append the one-shot claim (fail-closed under an exclusive lock).

        The check for an existing dev use and the append happen INSIDE the
        same lock, so two concurrent rollouts cannot both consume the single
        allowed access.
        """
        entry = {
            "schema": DEV_LEDGER_SCHEMA,
            "entry_id": f"v543_rollout_dev_{int(time.time() * 1000)}",
            "ts": _utc_now(),
            "actor": str(actor),
            "wp": "WP9",
            "gate": "T7 integration rollout",
            "campaign": "V5.4.3",
            "split": MODE_DEV,
            "mode": str(mode),
            "purpose": str(purpose),
            "seed": int(seed),
            "one_shot": True,
            "confirm_read": False,
            "new_QC_calls": 0,
            "n_dev_curves_plan": 589,
            "budget": dict(budget),
            "plan_lines": "§9.5 :1038-1039 / §10.2 :1063-1071",
        }
        if extra:
            entry.update(dict(extra))
        entry["entry_sha256"] = canonical_sha256(entry)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # ``a+`` keeps the file when it exists; the read+append pair is atomic
        # only under the file lock (POSIX), which is where the real gate runs.
        with open(self.path, "a+", encoding="utf-8") as handle:
            locked = False
            try:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
                locked = True
            except Exception:                      # pragma: no cover - non-POSIX
                locked = False
            try:
                handle.seek(0)
                existing = [line for line in handle.read().splitlines()
                            if line.strip()]
                for line in existing:
                    if str(json.loads(line).get("split")) == MODE_DEV:
                        raise RolloutError(
                            CODE_DEV_ALREADY_USED,
                            "the 589 dev curves were ALREADY accessed once "
                            f"({self.path}); T7 dev access is one-shot for the "
                            "whole campaign (plan §9.5 :1038) — a second visit "
                            "is an audit event, not a retry")
                handle.seek(0, os.SEEK_END)
                handle.write(json.dumps(entry, sort_keys=True,
                                        ensure_ascii=False) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            finally:
                if locked:                         # pragma: no cover - POSIX
                    import fcntl

                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        return entry


def require_dev_frame_loader(loader: Callable[..., Any] | None) -> Callable[..., Any]:
    """Refuse a dev rollout that has no way to read the dev pack.

    This runs BEFORE :meth:`DevAccessLedger.claim` on purpose: a missing loader
    must never burn the one shot.
    """
    if loader is None:
        raise RolloutError(
            CODE_DEV_LOADER_UNAVAILABLE,
            "no dev frame loader is configured: the 589-curve dev pack read "
            "belongs to WP9 (plan §9.5 :1038) and is intentionally NOT wired "
            "into this lane; pass a loader to run the dev mode for real")
    if not callable(loader):
        raise RolloutError(CODE_DEV_LOADER_UNAVAILABLE,
                           "dev frame loader is not callable")
    return loader


# ---------------------------------------------------------------------------
# frame prediction (the decoder half of §4.2, for ONE stage)
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class FramePrediction:
    """One frame's stage prediction (rows already in dataset key order)."""

    u_rows: np.ndarray
    delta_q_raw: float
    delta_e: float
    n_keys: int


@dataclass
class StageFrames:
    """The stage's patched ``EvalFrame`` list + the measurement provenance."""

    stage: str
    frames: list[Any]
    n_batches: int
    encoder_forward_count: int
    wall_seconds: float
    n_frames: int
    max_abs_delta_q_raw: float
    max_abs_delta_e: float
    max_abs_delta_u: float
    identity_vs_zero_residual: bool

    def describe(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "n_frames": int(self.n_frames),
            "n_batches": int(self.n_batches),
            "encoder_forward_count": int(self.encoder_forward_count),
            "wall_seconds": float(self.wall_seconds),
            "max_abs_delta_q_raw": float(self.max_abs_delta_q_raw),
            "max_abs_delta_e": float(self.max_abs_delta_e),
            "max_abs_delta_u": float(self.max_abs_delta_u),
            "identity_vs_zero_residual": bool(self.identity_vs_zero_residual),
        }


def stage_predictions(
        encoder: Any, bridge: Any, *, move_decoder: Any,
        process_decoder: Any, dataset: Any, lru: Any,
        positions: Sequence[int], checkpoint_sha256: str,
        batch_size: int = FRAMES_PER_BATCH) -> tuple[list[FramePrediction],
                                                    dict[str, Any]]:
    """Run ONE stage over the frozen frames (one encoder forward per batch).

    The decoder half is the plan §4.2 flow: ``packet -> context -> δu/δq/δe``,
    ``u = u0 + δu`` (rows restored to ``key_index`` order through the Bridge
    permutation, §4.3 :222), ``q = tanh(q0_raw + δq)``, ``e = e0 + δe``.
    """
    from pes2ts_core.pesn.contracts.key_atom_v3 import forward_view_v3
    from pes2ts_core.pesn.offline.training.key_atom_v53mp_train_v3 import (
        _batch_from_positions,
    )

    started = time.perf_counter()
    predictions: list[FramePrediction] = []
    n_batches = 0
    max_dq = 0.0
    max_de = 0.0
    max_du = 0.0
    before = int(encoder.forward_count)
    for start in range(0, len(positions), int(batch_size)):
        chunk = [int(p) for p in positions[start:start + int(batch_size)]]
        batch, _records = _batch_from_positions(dataset, lru, chunk, "cpu")
        view = forward_view_v3(batch)
        with torch.no_grad():
            packet = encoder(view)
            context = bridge.build_context(
                packet, view, encoder_ckpt_sha256=str(checkpoint_sha256))
            n_keys = int(context.n_keys)
            move_delta = check_move_output(
                move_decoder(context.move_view()), n_keys=n_keys,
                decoder_id=str(getattr(move_decoder, "decoder_id", "MOVE")))
            residual = check_process_output(
                process_decoder(context.process_view()),
                n_graphs=int(context.n_graphs),
                decoder_id=str(getattr(process_decoder, "decoder_id",
                                       "PROCESS")))
        n_batches += 1
        permutation = torch.argsort(context.key_graph, stable=True)
        total_u = context.u0 + move_delta
        u_rows = unpermute_to_key_rows(total_u, permutation).detach().cpu(
        ).numpy().astype(np.float64)
        delta_q = residual.delta_q_raw.detach().cpu().numpy().astype(np.float64)
        delta_e = residual.delta_e.detach().cpu().numpy().astype(np.float64)
        base_u = unpermute_to_key_rows(context.u0, permutation).detach().cpu(
        ).numpy().astype(np.float64)
        key_ptr = context.key_ptr.detach().cpu().numpy().astype(np.int64)
        n_graphs = int(key_ptr.shape[0]) - 1
        if n_graphs != len(chunk):
            raise RolloutError(
                CODE_FRAMES,
                f"{n_graphs} graphs in the context for {len(chunk)} positions")
        max_dq = max(max_dq, float(np.max(np.abs(delta_q))) if delta_q.size
                     else 0.0)
        max_de = max(max_de, float(np.max(np.abs(delta_e))) if delta_e.size
                     else 0.0)
        max_du = max(max_du, float(np.max(np.abs(u_rows - base_u))) if u_rows.size
                     else 0.0)
        for g in range(n_graphs):
            k_lo, k_hi = int(key_ptr[g]), int(key_ptr[g + 1])
            predictions.append(FramePrediction(
                u_rows=np.asarray(u_rows[k_lo:k_hi], dtype=np.float64),
                delta_q_raw=float(delta_q[g]),
                delta_e=float(delta_e[g]),
                n_keys=int(k_hi - k_lo)))
    forwards = int(encoder.forward_count) - before
    if forwards != n_batches:
        raise RolloutError(
            CODE_FRAMES,
            f"{n_batches} batches performed {forwards} encoder forwards; one "
            f"configuration must encode exactly "
            f"{ENCODER_FORWARD_CALLS_PER_VIEW} time (plan §4.4 :231)")
    meta = {
        "n_batches": n_batches,
        "encoder_forward_count": forwards,
        "wall_seconds": float(time.perf_counter() - started),
        "max_abs_delta_q_raw": float(max_dq),
        "max_abs_delta_e": float(max_de),
        "max_abs_delta_u": float(max_du),
    }
    return predictions, meta


def apply_predictions(base_frames: Sequence[Any], predictions: Sequence[Any],
                      *, stage: str, meta: Mapping[str, Any]) -> StageFrames:
    """Patch the frozen frames with ONE stage's predictions (§4.3 :220–224).

    The frame set, the pair slots, the teacher values and every grouping field
    are the BASELINE's, so a stage report and the frozen M0/P0 report are
    pairable frame-for-frame (§9.4 :1019).  A stage whose predictions are the
    zero residual reproduces the baseline bit-for-bit — which is what
    ``identity_vs_zero_residual`` asserts and what makes B0 a real check.
    """
    if len(base_frames) != len(predictions):
        raise RolloutError(
            CODE_FRAMES,
            f"stage {stage}: {len(predictions)} predictions for "
            f"{len(base_frames)} frames (the frame set must not change)")
    patched: list[Any] = []
    identical = True
    for frame, prediction in zip(base_frames, predictions):
        u_rows = np.asarray(prediction.u_rows, dtype=np.float64)
        if u_rows.shape != np.asarray(frame.u).shape:
            raise RolloutError(
                CODE_FRAMES,
                f"stage {stage}: curve {frame.curve_id!r} key rows "
                f"{list(u_rows.shape)} != frame rows "
                f"{list(np.asarray(frame.u).shape)}")
        q0_raw = frame.q0_raw
        e0 = frame.endpoint0_logit
        if q0_raw is None or e0 is None:
            raise RolloutError(
                CODE_FRAMES,
                f"stage {stage}: curve {frame.curve_id!r} carries no "
                "q0_raw/endpoint0_logit; the stage prediction needs the "
                "zero-residual baseline quantities (plan §4.3 :226)")
        q_pred = (None if frame.q_pred is None
                  else float(math.tanh(float(q0_raw)
                                       + float(prediction.delta_q_raw))))
        endpoint_logit = (None if frame.endpoint_logit is None
                          else float(e0) + float(prediction.delta_e))
        patched.append(replace(
            frame, u=u_rows,
            q_pred=(float(math.tanh(float(q0_raw))) if q_pred is None
                    else q_pred),
            endpoint_logit=(float(e0) if endpoint_logit is None
                            else endpoint_logit)))
        if not (np.array_equal(u_rows, np.asarray(frame.u))
                and float(prediction.delta_q_raw) == 0.0
                and float(prediction.delta_e) == 0.0):
            identical = False
    return StageFrames(
        stage=str(stage), frames=patched,
        n_batches=int(meta["n_batches"]),
        encoder_forward_count=int(meta["encoder_forward_count"]),
        wall_seconds=float(meta["wall_seconds"]),
        n_frames=len(patched),
        max_abs_delta_q_raw=float(meta["max_abs_delta_q_raw"]),
        max_abs_delta_e=float(meta["max_abs_delta_e"]),
        max_abs_delta_u=float(meta["max_abs_delta_u"]),
        identity_vs_zero_residual=bool(identical))


# ---------------------------------------------------------------------------
# the reference-normalized walk (§10.2 :1063–1071)
# ---------------------------------------------------------------------------
@dataclass
class WalkResult:
    """One stage's walk metrics over the frozen frame groups."""

    stage: str
    n_groups: int
    max_steps: int
    n_frames_stepped: int
    n_no_move: int
    n_budget_exhausted: int
    n_terminal_frame_reached: int
    n_ts_row_reached: int
    n_passed_ts: int
    n_misstop_start: int
    n_misstop_ts: int
    steps_mean: float | None
    steps_median: float | None
    path_length_angstrom_mean: float | None
    success_rate: float
    wall_seconds: float
    controller_reasons: dict[str, int]
    n_controller_calls: int

    def describe(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "unit": "curve x direction reference group",
            "n_groups": int(self.n_groups),
            "max_steps": int(self.max_steps),
            "n_frames_stepped": int(self.n_frames_stepped),
            "n_no_move": int(self.n_no_move),
            "n_budget_exhausted": int(self.n_budget_exhausted),
            "n_terminal_frame_reached": int(self.n_terminal_frame_reached),
            "n_ts_row_reached": int(self.n_ts_row_reached),
            "n_passed_ts": int(self.n_passed_ts),
            "n_misstop_start": int(self.n_misstop_start),
            "n_misstop_ts": int(self.n_misstop_ts),
            "steps_mean": self.steps_mean,
            "steps_median": self.steps_median,
            "path_length_angstrom_mean": self.path_length_angstrom_mean,
            "success_rate": float(self.success_rate),
            "wall_seconds": float(self.wall_seconds),
            "controller_reasons": dict(self.controller_reasons),
            "n_controller_calls": int(self.n_controller_calls),
            "tau_end": None,
            "endpoint_arrival": {
                "status": "not_applicable_tau_end_uncalibrated",
                "terminal_frame_reached": int(self.n_terminal_frame_reached),
                "reason": ("plan §1.5 :105 tau_end is None and §9.5 :1040 "
                           "calibrates it on inner-val with target precision "
                           "0.9; no threshold is invented here"),
            },
            "ts_crossing": {
                "n_groups_reached_ts": int(self.n_ts_row_reached),
                "n_groups_passed_ts": int(self.n_passed_ts),
                "rule": ("plan §4.3 :227 / §10.2 :1067 — a TS row must keep "
                         "moving, so 'passed' means a step AFTER the TS row "
                         "was executed"),
            },
            "collision": {
                "status": "not_measured",
                "value": None,
                "reason": ("plan §10.2 :1069 '物理松弛：不做' — the "
                           "reference-normalized walk performs no physics "
                           "step, so no collision can be detected or invented"),
            },
            "cost": {
                "wall_seconds": float(self.wall_seconds),
                "seconds_per_frame": (float(self.wall_seconds)
                                      / max(1, int(self.n_frames_stepped))),
            },
        }


def frame_groups(frames: Sequence[Any]) -> dict[tuple[str, str], list[int]]:
    """Group frame indices by ``(curve_id, direction)`` in frozen order."""
    groups: dict[tuple[str, str], list[int]] = {}
    for index, frame in enumerate(frames):
        groups.setdefault((str(frame.curve_id), str(frame.direction)),
                          []).append(int(index))
    return groups


def run_walk(stage_frames: StageFrames, *, max_steps: int = DEFAULT_MAX_STEPS,
             tau_end: float | None = None,
             chemical_state_check: Callable[[Any], tuple[bool, str]] | None = None
             ) -> WalkResult:
    """Walk the frozen reference sequences with the controller in the loop.

    One start per ``(curve, direction)`` group (the §10.2 :1065 "冻结起点集"),
    each walked once, at most ``max_steps`` frames, stopping on the first legal
    No-Move or when the group's frames are exhausted.
    """
    check = chemical_state_check or default_chemical_state_check
    if int(max_steps) < 1 or int(max_steps) > MAX_STEPS_CAP:
        raise RolloutError(
            CODE_BUDGET,
            f"max_steps must be in [1, {MAX_STEPS_CAP}], got {max_steps}")
    started = time.perf_counter()
    groups = frame_groups(stage_frames.frames)
    steps: list[int] = []
    path_lengths: list[float] = []
    reasons: dict[str, int] = {}
    n_no_move = 0
    n_exhausted = 0
    n_terminal = 0
    n_ts_reached = 0
    n_passed_ts = 0
    n_misstop_start = 0
    n_misstop_ts = 0
    n_frames_stepped = 0
    n_calls = 0
    for _key, indices in groups.items():
        if not indices:
            continue
        taken = indices[:int(max_steps)]
        group_steps = 0
        group_path = 0.0
        stopped = None
        passed_ts = False
        reached_ts = False
        for position, index in enumerate(taken):
            frame = stage_frames.frames[index]
            decision = controller_step(
                p_end=_sigmoid(frame.endpoint_logit),
                tau_end=tau_end,
                chemical_state_ok=bool(check(frame)[0]),
                is_ts=bool(frame.is_ts_row),
                q_observed=frame.q_pred)
            n_calls += 1
            group_steps += 1
            n_frames_stepped += 1
            reasons[decision["reason"]] = int(
                reasons.get(decision["reason"], 0)) + 1
            try:
                displacement = np.asarray(frame.u, dtype=np.float64) - np.asarray(
                    frame.u0 if frame.u0 is not None else frame.u,
                    dtype=np.float64)
                group_path += float(np.linalg.norm(displacement))
            except Exception:                       # pragma: no cover - guard
                pass
            if frame.is_ts_row:
                reached_ts = True
                if position + 1 < len(taken):
                    passed_ts = True
            if not decision["move"]:
                n_no_move += 1
                if frame.is_start_row:
                    n_misstop_start += 1
                if frame.is_ts_row:
                    n_misstop_ts += 1
                stopped = "no_move"
                break
        steps.append(group_steps)
        path_lengths.append(group_path)
        if reached_ts:
            n_ts_reached += 1
        if passed_ts:
            n_passed_ts += 1
        if stopped is None and taken and taken[-1] == indices[-1]:
            n_terminal += 1
        elif stopped is None:
            n_exhausted += 1
    n_groups = len(steps)
    return WalkResult(
        stage=stage_frames.stage, n_groups=n_groups, max_steps=int(max_steps),
        n_frames_stepped=n_frames_stepped, n_no_move=n_no_move,
        n_budget_exhausted=n_exhausted, n_terminal_frame_reached=n_terminal,
        n_ts_row_reached=n_ts_reached, n_passed_ts=n_passed_ts,
        n_misstop_start=n_misstop_start, n_misstop_ts=n_misstop_ts,
        steps_mean=(float(np.mean(steps)) if steps else None),
        steps_median=(float(np.median(steps)) if steps else None),
        path_length_angstrom_mean=(float(np.mean(path_lengths))
                                   if path_lengths else None),
        success_rate=(float(n_terminal) / float(n_groups) if n_groups else 0.0),
        wall_seconds=float(time.perf_counter() - started),
        controller_reasons=dict(sorted(reasons.items())),
        n_controller_calls=n_calls)


# ---------------------------------------------------------------------------
# B0 consistency (§9.5 :1037 — B0 IS the frozen baseline)
# ---------------------------------------------------------------------------
def load_frozen_baseline(path: Any = None) -> tuple[dict[str, Any], str]:
    target = Path(str(path)) if path is not None else (
        PROJECT_ROOT / CAMPAIGN_ROOT / DEFAULT_BASELINE_REL)
    if not Path(str(target)).is_absolute():
        target = PROJECT_ROOT / str(target)
    document = _read_json(target, what="frozen M0/P0 baseline",
                          code=CODE_BASELINE_MISSING)
    if str(document.get("schema")) != "pesn_v543_m0p0_baseline_v1":
        raise RolloutError(
            CODE_BASELINE_MISSING,
            f"{target} is not a pesn_v543_m0p0_baseline_v1 document")
    return document, sha256_file(target)


def assert_b0_matches_frozen_baseline(
        *, stage_report: Mapping[str, Any], baseline: Mapping[str, Any],
        seed: int, budget_matches: bool = True) -> dict[str, Any]:
    """Compare B0 with the frozen M0/P0 per-seed document (§9.5 :1037).

    ``per_curve``/``pooled``/``validity`` must be BIT-identical.  When the
    rollout budget differs from the frozen document's, the comparison is
    reported as ``not_checked`` (with the reason) — never as a pass.
    """
    entry = (baseline.get("per_seed") or {}).get(str(int(seed)))
    if not isinstance(entry, Mapping):
        raise RolloutError(
            CODE_BASELINE_MISSING,
            f"the frozen baseline carries no per_seed entry for seed {seed}")
    if not budget_matches:
        return {
            "checked": False,
            "identical": None,
            "reason": ("this rollout evaluated a DIFFERENT inner-val budget "
                       "than the frozen baseline document; B0 cannot be "
                       "compared bit-for-bit against a different frame set"),
            "plan_lines": "§9.5 :1037",
        }
    differing: list[str] = []
    for key in ("per_curve", "pooled", "validity"):
        if canonical_sha256(stage_report.get(key)) != canonical_sha256(
                entry.get(key)):
            differing.append(key)
    if differing:
        raise RolloutError(
            CODE_B0_BASELINE_MISMATCH,
            "stage B0 is not the frozen M0/P0 baseline: the blocks "
            f"{differing} differ from {DEFAULT_BASELINE_REL} (seed {seed}); "
            "the T7 baseline must reproduce the §9.5 :1037 recomputation "
            "exactly, otherwise the whole rollout is unpairable")
    return {
        "checked": True,
        "identical": True,
        "blocks": ["per_curve", "pooled", "validity"],
        "baseline_per_seed_document_sha256": canonical_sha256(entry),
        "b0_per_seed_document_sha256": canonical_sha256(
            {key: stage_report.get(key)
             for key in ("per_curve", "pooled", "validity")}),
        "plan_lines": "§9.5 :1035-1037",
    }


# ---------------------------------------------------------------------------
# the run
# ---------------------------------------------------------------------------
@dataclass
class SeedStageResults:
    """Everything measured for one seed: frames + per stage eval + walks."""

    seed: int
    base_frames: list[Any]
    positions_rule: Mapping[str, Any]
    positions_sha256: str
    train_q_mean: float | None
    stages: dict[str, StageFrames] = field(default_factory=dict)
    reports: dict[str, dict[str, Any]] = field(default_factory=dict)
    walks: dict[str, WalkResult] = field(default_factory=dict)
    decoders: dict[str, dict[str, Any]] = field(default_factory=dict)


def _stage_decoders(stage: str, *, move_resolution: ChampionResolution,
                    process_resolution: ChampionResolution,
                    registry: Any) -> tuple[Any, Any, dict[str, Any]]:
    """Construct the (move, process) decoder pair of one §10.1 combination."""
    plan = stage_plan(stage)
    if plan.uses_champion(TRACK_MOVE) and not move_resolution.is_baseline:
        move = build_champion_decoder(move_resolution, registry=registry)
        move_id = str(move_resolution.decoder_id)
        move_audit = {"source": "champion",
                      "checkpoint": move_resolution.checkpoint_path,
                      "sha256": move_resolution.checkpoint_sha256}
    else:
        move = build_baseline(TRACK_MOVE)
        move_id = BASELINE_DECODER_BY_TRACK[TRACK_MOVE]
        move_audit = {"source": "baseline_zero_residual", "checkpoint": None,
                      "sha256": None}
    if plan.uses_champion(TRACK_PROCESS) and not process_resolution.is_baseline:
        process = build_champion_decoder(process_resolution, registry=registry)
        process_id = str(process_resolution.decoder_id)
        process_audit = {"source": "champion",
                         "checkpoint": process_resolution.checkpoint_path,
                         "sha256": process_resolution.checkpoint_sha256}
    else:
        process = build_baseline(TRACK_PROCESS)
        process_id = BASELINE_DECODER_BY_TRACK[TRACK_PROCESS]
        process_audit = {"source": "baseline_zero_residual", "checkpoint": None,
                         "sha256": None}
    audit = {
        "stage": str(stage),
        "move": dict(move_audit, decoder_id=str(move_id)),
        "process": dict(process_audit, decoder_id=str(process_id)),
        "champion_tracks": list(plan.champion_tracks),
        "purpose": STAGE_PURPOSE[stage],
    }
    return move, process, audit


def run_seed(*, seed: int, encoder: Any, bridge: Any, dataset: Any,
             train_pos: Sequence[int], labels_v52: Any,
             base_frames: list[Any], positions_rule: Mapping[str, Any],
             positions_sha256: str, move_resolution: ChampionResolution,
             process_resolution: ChampionResolution, registry: Any,
             checkpoint_sha256: str, lru: Any, positions: Sequence[int],
             max_steps: int, tau_end: float | None,
             chemical_state_check: Callable[[Any], tuple[bool, str]] | None,
             stage_filter: Sequence[str] | None = None) -> SeedStageResults:
    """Measure all four §10.1 combinations for ONE seed on the frozen frames."""
    from pes2ts_core.pesn.offline.evaluation.v543_eval import evaluate_v543
    from pes2ts_core.pesn.offline.training.key_atom_v53mp_train_v3 import (
        _train_q_mean,
    )

    train_q_mean = _train_q_mean(dataset.samples, labels_v52,
                                 [int(p) for p in train_pos])
    wanted = list(stage_filter) if stage_filter else list(STAGES)
    result = SeedStageResults(
        seed=int(seed), base_frames=base_frames, positions_rule=dict(
            positions_rule), positions_sha256=str(positions_sha256),
        train_q_mean=train_q_mean)
    for stage in wanted:
        move, process, audit = _stage_decoders(
            stage, move_resolution=move_resolution,
            process_resolution=process_resolution, registry=registry)
        predictions, prediction_meta = stage_predictions(
            encoder, bridge, move_decoder=move, process_decoder=process,
            dataset=dataset, lru=lru, positions=positions,
            checkpoint_sha256=checkpoint_sha256)
        stage_frames = apply_predictions(base_frames, predictions, stage=stage,
                                        meta=prediction_meta)
        report = evaluate_v543(stage_frames.frames, train_q_mean=train_q_mean,
                               tau_end=None)
        walk = run_walk(stage_frames, max_steps=int(max_steps), tau_end=tau_end,
                        chemical_state_check=chemical_state_check)
        result.stages[stage] = stage_frames
        result.reports[stage] = report
        result.walks[stage] = walk
        result.decoders[stage] = dict(audit, assembly_measured=_assert_rollout_budget(
            build_rollout_assembly(
                encoder, bridge=bridge, move=move,
                move_id=str(getattr(move, "decoder_id", "M0")), process=process,
                process_id=str(getattr(process, "decoder_id", "P0")))))
        print(f"[v543rollout] seed {seed} stage {stage}: "
              f"{walk.n_frames_stepped} frames stepped, "
              f"no_move={walk.n_no_move}, ts_passed={walk.n_passed_ts}, "
              f"success={walk.success_rate:.3f}, "
              f"wall={stage_frames.wall_seconds:.1f}s", file=sys.stderr,
              flush=True)
    return result


def build_report(*, mode: str, seeds: Sequence[int], campaign: Mapping[str, Any],
                 checkpoints: Mapping[int, Mapping[str, Any]],
                 results: Sequence[SeedStageResults],
                 move_resolution_by_seed: Mapping[int, ChampionResolution],
                 process_resolution_by_seed: Mapping[int, ChampionResolution],
                 budget: Mapping[str, Any], data_block: Mapping[str, Any],
                 baseline_document: Mapping[str, Any] | None,
                 baseline_sha256: str | None,
                 dev_access: Mapping[str, Any] | None = None,
                 determinism_note: Mapping[str, Any] | None = None
                 ) -> dict[str, Any]:
    """Assemble the ``pesn_v543_rollout_report_v1`` document (measurements)."""
    per_seed: dict[str, Any] = {}
    baseline_budget_matches = _budget_matches_frozen(baseline_document, budget)
    for result in results:
        seed = str(int(result.seed))
        stage_block: dict[str, Any] = {}
        for stage in STAGES:
            if stage not in result.stages:
                stage_block[stage] = {"status": "not_run"}
                continue
            frames = result.stages[stage]
            report = result.reports[stage]
            walk = result.walks[stage]
            consistency = None
            if stage == "B0" and baseline_document is not None:
                consistency = assert_b0_matches_frozen_baseline(
                    stage_report=report, baseline=baseline_document,
                    seed=int(result.seed),
                    budget_matches=bool(baseline_budget_matches))
            stage_block[stage] = {
                "stage": stage,
                "decoders": dict(result.decoders.get(stage, {})),
                "frames": frames.describe(),
                "eval_report": report,
                "eval_report_sha256": canonical_sha256(report),
                "rollout": walk.describe(),
                "b0_vs_frozen_baseline": consistency,
            }
        per_seed[seed] = {
            "seed": int(result.seed),
            "encoder": dict(checkpoints[int(result.seed)]),
            "positions": {
                "rule": dict(result.positions_rule),
                "n_positions": len(result.base_frames),
                "positions_sha256": result.positions_sha256,
                "train_q_mean": result.train_q_mean,
            },
            "champion_resolution": {
                TRACK_MOVE: move_resolution_by_seed[int(result.seed)].describe(),
                TRACK_PROCESS:
                    process_resolution_by_seed[int(result.seed)].describe(),
            },
            "stages": stage_block,
        }
    stage_summary = _stage_summary(results)
    integration_gate = _integration_gate(
        per_seed=per_seed, summary=stage_summary)
    return {
        "schema": ROLLOUT_SCHEMA,
        "version": ROLLOUT_VERSION,
        "plan_section": PLAN_SECTION,
        "generator": _generator_identity(),
        "campaign_id": str(campaign.get("campaign_id", "")),
        "created_utc": _utc_now(),
        "mode": str(mode),
        "stages": list(STAGES),
        "stage_definitions": {stage: stage_plan(stage).describe()
                              for stage in STAGES},
        "seeds": [int(seed) for seed in seeds],
        "budget": dict(budget),
        "data": dict(data_block),
        "frozen_baseline": {
            "path": DEFAULT_BASELINE_REL,
            "sha256": baseline_sha256,
            "present": baseline_document is not None,
            "budget_matches_b0_check": bool(baseline_budget_matches),
            "role": ("B0 MUST reproduce this document (plan §9.5 :1035-1037); "
                     "it is the H2/H3 denominator and the A0 latency baseline"),
        },
        "controller": {
            "kind": "non_learned_reaction_controller",
            "parameter_count": 0,
            "move_flag": "plan §4.6 :253-259 (verbatim)",
            "no_move_reasons": list(NO_MOVE_REASONS),
            "q_used_as_trigger": False,
            "tau_end": (None if not budget else budget.get("tau_end", None)),
            "tau_end_note": ("§9.5 :1040 calibrates tau_end on inner-val once "
                             "(target precision 0.9); uncalibrated => No-Move "
                             "is impossible and is reported as such"),
            "chemical_state_check": (
                "injected" if budget and budget.get(
                    "chemical_state_check_injected") else
                "unavailable_no_legal_endpoint_or_bond_verifier_in_v543"),
            "not_in_model_forward": True,
            "acceptance_cases": list(CONTROLLER_ACCEPTANCE_CASES),
            "plan_lines": "§4.6 :250-263 / §10.3 :1073-1080",
        },
        "per_seed": per_seed,
        "stage_summary": stage_summary,
        "integration_gate": integration_gate,
        "dev_access": dict(dev_access or {
            "mode": str(mode),
            "dev_accessed": False,
            "confirm_read": False,
            "new_QC_calls": 0,
            "one_shot_ledger": DEFAULT_DEV_LEDGER_REL,
            "note": "inner-val mode: no dev/confirm payload was read",
        }),
        "determinism": dict(determinism_note or {}),
        "implementation_choices": _implementation_choices(),
        "ambiguities": _ambiguities(),
    }


def _generator_identity() -> dict[str, Any]:
    """The code identity of this document (§9.5 freeze discipline)."""
    from pes2ts_core.pesn.models.v543 import LatentBridge
    from pes2ts_core.pesn.offline.evaluation.v543_eval import (
        evaluator_spec_sha256,
    )

    module = Path(__file__).resolve()
    evaluator = PROJECT_ROOT / "pes2ts_core/pesn/offline/evaluation/v543_eval.py"
    return {
        "module": (str(module.relative_to(PROJECT_ROOT))
                   if module.is_relative_to(PROJECT_ROOT) else str(module)),
        "module_sha256": sha256_file(module),
        "evaluator_module": ("pes2ts_core/pesn/offline/evaluation/v543_eval.py"),
        "evaluator_code_sha256": (sha256_file(evaluator)
                                  if evaluator.is_file() else None),
        "evaluator_spec_sha256": evaluator_spec_sha256(),
        "bridge_spec_sha256": str(LatentBridge().bridge_spec_sha256()),
        "note": ("a rollout number is only reproducible together with the "
                 "generator SHA, the frozen evaluator code/spec SHA and the "
                 "Bridge spec SHA (plan §9.5 :1037 discipline)"),
    }


def _budget_matches_frozen(baseline: Mapping[str, Any] | None,
                           budget: Mapping[str, Any]) -> bool | None:
    """Does this rollout use the frozen baseline's inner-val budget?"""
    if not baseline:
        return None
    frozen = baseline.get("budget") or {}
    keys = ("eval_curves_requested", "frames_per_curve_requested",
            "frames_per_batch")
    for key in keys:
        if int(frozen.get(key, -1)) != int(budget.get(key, -2)):
            return False
    return True


def _pooled_primary(report: Mapping[str, Any]) -> dict[str, Any]:
    pooled = report.get("pooled") or {}
    move = pooled.get("move") or {}
    process = pooled.get("process") or {}
    q_block = process.get("q") or {}
    ordering = process.get("ordering") or {}
    endpoint = process.get("endpoint") or {}
    return {
        "move.rmse_zero": move.get("rmse_zero"),
        "move.sign_accuracy": move.get("sign_accuracy_cluster_median"),
        "move.fixed_coverage.direction_at_3":
            move.get("fixed_coverage_direction_at_3"),
        "move.dead_fraction": move.get("dead_zone_fraction"),
        "process.q.mae_curve_median": q_block.get("mae_curve_median"),
        "process.ordering.pairwise_accuracy_full":
            ordering.get("pairwise_accuracy_full"),
        "process.endpoint.auprc": endpoint.get("auprc"),
        "process.endpoint.brier": endpoint.get("brier"),
        "process.endpoint.ece": endpoint.get("ece"),
    }


def _stage_summary(results: Sequence[SeedStageResults]) -> dict[str, Any]:
    """Per-stage pooled metrics + walk counts, seed-averaged (readability)."""
    out: dict[str, Any] = {}
    for stage in STAGES:
        per_seed: dict[str, Any] = {}
        for result in results:
            if stage not in result.reports:
                continue
            per_seed[str(int(result.seed))] = {
                "pooled": _pooled_primary(result.reports[stage]),
                "walk": result.walks[stage].describe(),
                "frames": result.stages[stage].describe(),
            }
        keys = sorted({key for entry in per_seed.values()
                       for key in (entry["pooled"] or {})})
        metrics: dict[str, Any] = {}
        for key in keys:
            values = [entry["pooled"][key] for entry in per_seed.values()
                      if isinstance(entry["pooled"].get(key), (int, float))
                      and not isinstance(entry["pooled"].get(key), bool)]
            metrics[key] = {
                "per_seed": {seed: per_seed[seed]["pooled"].get(key)
                             for seed in per_seed},
                "mean": (float(np.mean(values)) if values else None),
                "n_seeds": len(values),
            }
        out[stage] = {
            "purpose": STAGE_PURPOSE[stage],
            "per_seed": per_seed,
            "metrics": metrics,
            "note": ("seed means are readability only; §9.4 :1023 averages "
                     "seeds per bootstrap draw inside the statistics lane"),
        }
    return out


def _integration_gate(*, per_seed: Mapping[str, Any],
                      summary: Mapping[str, Any]) -> dict[str, Any]:
    """§10.1 :1059 — the BMP integration gate (statuses, never invented)."""
    b0 = summary.get("B0") or {}
    bmp = summary.get("BMP") or {}
    latency = {}
    for stage in STAGES:
        entry = summary.get(stage) or {}
        walls = [block["frames"]["wall_seconds"]
                 for block in (entry.get("per_seed") or {}).values()
                 if isinstance(block, dict)]
        latency[stage] = float(np.max(walls)) if walls else None
    b0_wall = latency.get("B0")
    ratio = (None if not b0_wall or not latency.get("BMP")
             else float(latency["BMP"]) / float(b0_wall))
    misstop = {}
    for stage in STAGES:
        entry = summary.get(stage) or {}
        start = sum(int(block["walk"]["n_misstop_start"])
                    for block in (entry.get("per_seed") or {}).values()
                    if isinstance(block, dict))
        ts = sum(int(block["walk"]["n_misstop_ts"])
                 for block in (entry.get("per_seed") or {}).values()
                 if isinstance(block, dict))
        misstop[stage] = {"start": start, "ts": ts}
    checks = {
        "h2_not_degraded_on_either_track": {
            "status": "pending_h2_report",
            "reason": ("§10.1 :1059 needs the frozen H2 report; the rollout "
                       "lane reports the single-step metrics but never "
                       "declares H2 itself"),
        },
        "misstop_start_or_ts_not_increased": {
            "status": ("pass" if misstop.get("BMP", {}).get("start", 0) == 0
                       and misstop.get("BMP", {}).get("ts", 0) == 0
                       else "fail"),
            "bmp": misstop.get("BMP"),
            "b0": misstop.get("B0"),
        },
        "collision_not_increased": {
            "status": "not_measurable",
            "reason": "§10.2 :1069 — no physical relaxation in this stage",
        },
        "p95_latency_le_2x_A0": {
            "status": (None if ratio is None
                       else ("pass" if ratio <= MAX_P95_LATENCY_RATIO
                             else "fail")),
            "ratio_bmp_over_b0": ratio,
            "limit": MAX_P95_LATENCY_RATIO,
            "note": ("list of per-stage max wall seconds is NOT a p95 "
                     "estimator; it is the conservative upper bound available "
                     "in this lane and is labelled as such"),
            "wall_seconds_by_stage": latency,
            "estimator": "max_wall_seconds_upper_bound",
        },
    }
    decided = [check for check in checks.values()
               if check.get("status") in ("fail", "pass")]
    return {
        "rules": list(INTEGRATION_GATE_RULES),
        "checks": checks,
        "passed": (False if any(check.get("status") == "fail"
                                for check in checks.values()) else None),
        "status": ("fail" if any(check.get("status") == "fail"
                                 for check in checks.values())
                   else ("pending" if len(decided) < len(checks)
                         else "pass")),
        "selected_combination": None,
        "selection_note": ("§10.1 :1060 — the fallback order BM → BP → B0 is "
                           "pre-registered and is NOT chosen by looking at the "
                           "rollout; this document therefore records the gate "
                           "state and leaves the choice to WP9"),
        "plan_lines": "§10.1 :1059-1061",
        "b0_metrics": (b0.get("metrics") or {}),
        "bmp_metrics": (bmp.get("metrics") or {}),
        "n_seeds_compared": len(per_seed),
    }


def _implementation_choices() -> list[str]:
    return [
        "the rollout is the REFERENCE-NORMALIZED one: the frozen reference "
        "frame sequence of each (curve, direction) group is the walk path and "
        "the stage model is queried once per reference frame. Plan §10.2 :1069 "
        "forbids physical relaxation in this stage and requires the two to be "
        "reported separately; the physics-relaxed walk belongs to V6.",
        "one start per (curve, direction) reference group (§10.2 :1065 "
        "'冻结起点集'), each walked once, at most --max-steps frames, stopping "
        "on the first legal No-Move or at the end of the group.",
        "tau_end stays None (§1.5 :105): §9.5 :1040 calibrates it on inner-val "
        "with target precision 0.9, and this lane does not calibrate, so "
        "endpoint arrival is reported as not_applicable rather than computed "
        "against an invented threshold.",
        "chemical_state_ok has no V5.4.3 implementation (§4.6 :262 — the outer "
        "legal-endpoint/bond verification is the V6 physics loop's job), so the "
        "default checker returns False with a recorded reason: No-Move is then "
        "impossible, which is the conservative direction (a No-Move requires "
        "positive chemical evidence, never its absence).",
        "each stage report is the FULL frozen evaluator document "
        "(evaluate_v543) over the SAME frame set, so a stage and the frozen "
        "M0/P0 baseline are pairable frame-for-frame (§9.4 :1019) without any "
        "re-derivation.",
        "B0's per_curve/pooled/validity blocks are compared bit-for-bit with "
        "audit/m0_p0_baseline_v543_eval.json; a different --eval-curves/"
        "--frames-per-curve budget makes the check 'not_checked' instead of "
        "faking a pass.",
        "the two-expert assembly is built directly (not via assemble_v543) "
        "because that helper asserts a zero-residual inactive track for "
        "benchmark runs (§4.3 :226); T7 is the documented exception that turns "
        "both champions on.",
    ]


def _ambiguities() -> list[str]:
    return [
        "§11.1 lists the rollout as pes2ts_core/pesn/offline/evaluation/"
        "v543_rollout.py while the WP9 task brief names "
        "scripts/agent_ops/v543_rollout.py; this lane implements the "
        "agent_ops entry point (the campaign seam) and keeps every heavy "
        "primitive in the frozen packages it imports. §11.2's row "
        "('rollout、controller_step') is satisfied by this module.",
        "§10.2 does not define the walk STEP semantics beyond '推进预算：固定步数"
        "/固定 wall'. This lane advances one frozen reference frame per step "
        "(reference-normalized) and reports steps/path length per group.",
        "§10.2 :1070 lists 'endpoint arrival、TS crossing、collision、steps、path "
        "length、success、cost' but does not define 'success'. This lane defines "
        "it as reaching the group's terminal reference frame within the step "
        "budget and reports the count next to the rate.",
        "§10.1 :1059 requires 'p95 延迟 ≤2× A0' but this lane measures wall "
        "seconds per stage (not a p95 estimator); it reports the max-wall "
        "upper bound explicitly labelled as such.",
        "§9.5 :1038 says '只在 T6 冻结冠军后、T7 集成时一次性访问 589 曲线' "
        "without naming the ledger file. This lane uses its own append-only "
        "audit/dev_rollout_access_ledger.jsonl and never writes "
        "run_ledger.jsonl (owned by the formal runner).",
    ]


# ---------------------------------------------------------------------------
# data plumbing (inner-val; shared with v543_eval's frozen frame set)
# ---------------------------------------------------------------------------
def open_inner_val_seed(*, seed: int, campaign: Mapping[str, Any],
                        eval_curves: int, frames_per_curve: int,
                        with_basis: bool, with_strata: bool
                        ) -> dict[str, Any]:
    """Open the frozen inner-val slice of ONE seed and build its frame set.

    Delegates to ``scripts.agent_ops.v543_eval`` (``inner_val_positions`` +
    ``build_seed_frames``) so the frame set is byte-identical to the one the
    frozen M0/P0 baseline and every formal candidate used.
    """
    import json as _json

    from pes2ts_core.pesn.offline.training.key_atom_v53mp_train_v3 import (
        _FeatureLRU,
        v53mp_pool_split,
    )
    from scripts.agent_ops.v542_a0_parity import (
        DEFAULT_CONFIG as DATA_CONFIG,
        _open_frozen_dataset,
        load_pinned_assets,
    )
    from scripts.agent_ops.v543_eval import (
        FRAMES_PER_BATCH as EVAL_FRAMES_PER_BATCH,
        build_seed_frames,
        inner_val_positions,
    )
    from scripts.agent_ops.v543_s0_probe import positions_meta
    from scripts.campaigns.r3_v53mp_run import _tier_pool_curve_ids

    data_cfg, _assets = load_pinned_assets(PROJECT_ROOT / DATA_CONFIG)
    dataset = _open_frozen_dataset(data_cfg)
    split_path = str((data_cfg.get("split_contract") or {}).get(
        "split_manifest") or "/pes2ts_data/v53mp_50k_data/"
                            "split_manifest_v53mp.json")
    split_doc = _json.loads(Path(split_path).read_text(encoding="utf-8"))
    pool_ids = _tier_pool_curve_ids(split_doc, "50k")
    labels_v52, endpoint_labels = dataset.label_tables()
    train_pos, val_pos = v53mp_pool_split(dataset.samples,
                                          pool_curve_ids=pool_ids,
                                          seed=int(seed))
    positions, rule = inner_val_positions(
        val_pos, dataset, eval_curves=int(eval_curves),
        frames_per_curve=int(frames_per_curve))
    rule = dict(rule)
    rule["frames_per_batch"] = EVAL_FRAMES_PER_BATCH
    meta = (positions_meta(dataset, positions) if with_strata
            else [{"stratum": ""} for _ in positions])
    lru = _FeatureLRU(dataset, capacity=len(positions) + 128)
    return {
        "dataset": dataset, "lru": lru, "positions": positions,
        "train_pos": [int(p) for p in train_pos],
        "val_pos": [int(p) for p in val_pos],
        "labels_v52": labels_v52, "endpoint_labels": endpoint_labels,
        "meta": meta, "rule": rule,
        "positions_sha256": canonical_sha256([int(p) for p in positions]),
        "data_config": DATA_CONFIG, "split_manifest": split_path,
        "n_pool_curves": len(pool_ids),
        "build_seed_frames": build_seed_frames,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    """The full T7 measurement (inner-val by default; dev gated by §9.5)."""
    from pes2ts_core.pesn.models.v543 import CoarseEncoder, LatentBridge
    from pes2ts_core.pesn.offline.evaluation.v543_eval import zero_residual_frames
    from scripts.agent_ops.v543_s0_probe import (
        load_campaign_config,
        resolve_checkpoints,
    )

    mode = assert_mode_allowed(str(args.mode),
                              dev_onetime=bool(getattr(args, "dev_onetime",
                                                       False)))
    dev_loader = getattr(args, "dev_frame_loader", None)
    dev_ledger = DevAccessLedger(getattr(args, "dev_ledger",
                                         DEFAULT_DEV_LEDGER_REL))
    dev_access: dict[str, Any] = {
        "mode": mode,
        "dev_accessed": False,
        "confirm_read": False,
        "new_QC_calls": 0,
        "one_shot_ledger": str(dev_ledger.path),
        "ledger_entries_before": len(dev_ledger.entries()),
    }
    if mode == MODE_DEV:
        # §9.5 :1038 — refuse an unreadable/unusable dev path BEFORE claiming
        require_dev_frame_loader(dev_loader)
        if dev_ledger.already_used():
            raise RolloutError(
                CODE_DEV_ALREADY_USED,
                f"the 589 dev curves were already accessed once "
                f"({dev_ledger.path}); T7 dev access is one-shot (plan §9.5 "
                ":1038)")
    campaign = load_campaign_config(PROJECT_ROOT / str(args.config))
    seeds = [int(s) for s in (getattr(args, "seeds", None)
                              or campaign["source"]["seeds"])]
    if len(seeds) != 1 and mode != MODE_INNER_VAL:      # dev is one seed
        seeds = seeds[:1]
    checkpoints = resolve_checkpoints(campaign, seeds)
    select = load_select_report(getattr(args, "select_report",
                                        PROJECT_ROOT / CAMPAIGN_ROOT
                                        / DEFAULT_SELECT_REL))
    registry = candidate_registry()
    move_resolutions: dict[int, ChampionResolution] = {}
    process_resolutions: dict[int, ChampionResolution] = {}
    for seed in seeds:
        move_resolutions[int(seed)] = resolve_champion(
            select, TRACK_MOVE, seed=int(seed),
            campaign_root=PROJECT_ROOT / CAMPAIGN_ROOT,
            override_decoder=getattr(args, "move_champion", None),
            override_checkpoint=getattr(args, "move_champion_checkpoint", None))
        process_resolutions[int(seed)] = resolve_champion(
            select, TRACK_PROCESS, seed=int(seed),
            campaign_root=PROJECT_ROOT / CAMPAIGN_ROOT,
            override_decoder=getattr(args, "process_champion", None),
            override_checkpoint=getattr(args, "process_champion_checkpoint",
                                        None))

    budget = {
        "eval_curves_requested": int(getattr(args, "eval_curves",
                                             DEFAULT_EVAL_CURVES)),
        "frames_per_curve_requested": int(getattr(args, "frames_per_curve",
                                                  DEFAULT_FRAMES_PER_CURVE)),
        "frames_per_batch": FRAMES_PER_BATCH,
        "inner_val_cap_plan": INNER_VAL_CAP,
        "max_steps": int(getattr(args, "max_steps", DEFAULT_MAX_STEPS)),
        "tau_end": None,
        "with_basis_diagnostic": not bool(getattr(args, "no_basis", False)),
        "with_strata": not bool(getattr(args, "no_strata", False)),
        "chemical_state_check_injected": False,
        "training_runs_started": 0,
    }
    # The frozen M0/P0 baseline is the reference of BOTH modes (§9.5 :1037).
    baseline_document, baseline_sha = load_frozen_baseline(
        getattr(args, "baseline", None))

    results: list[SeedStageResults] = []
    data_block: dict[str, Any] = {
        "mode": mode,
        "dev_accessed": bool(mode == MODE_DEV),
        "confirm_accessed": False,
    }
    for seed in seeds:
        seed = int(seed)
        checkpoint = dict(checkpoints[seed])
        encoder = CoarseEncoder.load(str(checkpoint["path"]), seed)
        encoder.eval()
        bridge = LatentBridge(encoder_ckpt_sha256=str(checkpoint["sha256"]))
        if mode == MODE_INNER_VAL:
            opened = open_inner_val_seed(
                seed=seed, campaign=campaign,
                eval_curves=budget["eval_curves_requested"],
                frames_per_curve=budget["frames_per_curve_requested"],
                with_basis=budget["with_basis_diagnostic"],
                with_strata=budget["with_strata"])
            base_frames = opened["build_seed_frames"](
                encoder, bridge, opened["dataset"], opened["lru"],
                opened["positions"], checkpoint_sha256=checkpoint["sha256"],
                meta=opened["meta"], labels_v52=opened["labels_v52"],
                endpoint_labels=opened["endpoint_labels"],
                with_basis=budget["with_basis_diagnostic"]).frames
            positions = opened["positions"]
            positions_sha = opened["positions_sha256"]
            rule = dict(opened["rule"])
        else:                                           # pragma: no cover - WP9
            # The one-shot dev claim lands BEFORE the payload is read.
            claim = dev_ledger.claim(
                actor=str(getattr(args, "actor", "v543_rollout")),
                purpose="t7_integration_rollout",
                seed=seed, mode=mode, budget=budget,
                extra={"select_report": str(select["path"]),
                       "select_report_sha256": str(select["sha256"]),
                       "dev_onetime_flag": True})
            dev_access.update({"dev_accessed": True, "claim": claim,
                               "ledger_entries_after": len(
                                   dev_ledger.entries())})
            dev_payload = require_dev_frame_loader(dev_loader)(
                seed=seed, dataset=None, encoder=encoder, bridge=bridge)
            base_frames = list(dev_payload["frames"])
            positions = [int(p) for p in dev_payload["positions"]]
            positions_sha = canonical_sha256(positions)
            rule = dict(dev_payload.get("rule") or {})
            opened = {"dataset": dev_payload["dataset"],
                      "lru": dev_payload["lru"],
                      "train_pos": dev_payload["train_pos"],
                      "labels_v52": dev_payload["labels_v52"],
                      "endpoint_labels": dev_payload["endpoint_labels"]}
        # B0 must be the frozen baseline: the zero-residual frame set.
        zero_frames = zero_residual_frames(base_frames)
        result = run_seed(
            seed=seed, encoder=encoder, bridge=bridge,
            dataset=opened["dataset"], train_pos=opened["train_pos"],
            labels_v52=opened["labels_v52"],
            base_frames=zero_frames, positions_rule=rule,
            positions_sha256=positions_sha,
            move_resolution=move_resolutions[seed],
            process_resolution=process_resolutions[seed], registry=registry,
            checkpoint_sha256=str(checkpoint["sha256"]),
            lru=opened["lru"], positions=positions,
            max_steps=budget["max_steps"], tau_end=budget["tau_end"],
            chemical_state_check=None)
        results.append(result)
        data_block["checkpoint"] = dict(checkpoint)

    report = build_report(
        mode=mode, seeds=seeds, campaign=campaign, checkpoints=checkpoints,
        results=results, move_resolution_by_seed=move_resolutions,
        process_resolution_by_seed=process_resolutions, budget=budget,
        data_block=data_block, baseline_document=baseline_document,
        baseline_sha256=baseline_sha, dev_access=dev_access,
        determinism_note={
            "encoder_forwards_per_configuration": ENCODER_FORWARD_CALLS_PER_VIEW,
            "no_training_started": True,
            "torch_compile": False,
            "new_QC_calls": 0,
        })
    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_seeds(text: str) -> list[int]:
    seeds = [int(token) for token in str(text).replace(",", " ").split()]
    if not seeds:
        raise argparse.ArgumentTypeError("at least one seed is required")
    return seeds


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="V5.4.3 T7 integration rollout (plan §10) — four "
                    "combinations B0/BM/BP/BMP over the frozen frame set")
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--mode", default=MODE_INNER_VAL, choices=list(MODES),
                        help="inner_val (default) / dev (one-shot) / confirm "
                             "(sealed, always refused)")
    parser.add_argument("--dev-onetime", action="store_true",
                        help="the explicit one-shot flag required by §9.5 :1038")
    parser.add_argument("--dev-ledger", default=DEFAULT_DEV_LEDGER_REL)
    parser.add_argument("--select-report",
                        default=f"{CAMPAIGN_ROOT}/{DEFAULT_SELECT_REL}")
    parser.add_argument("--baseline", default=None)
    parser.add_argument("--seeds", type=_parse_seeds, default=None)
    parser.add_argument("--eval-curves", type=int, default=DEFAULT_EVAL_CURVES)
    parser.add_argument("--frames-per-curve", type=int,
                        default=DEFAULT_FRAMES_PER_CURVE)
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--no-basis", action="store_true")
    parser.add_argument("--no-strata", action="store_true")
    parser.add_argument("--move-champion", default=None,
                        help="override the select report's move champion id")
    parser.add_argument("--process-champion", default=None)
    parser.add_argument("--move-champion-checkpoint", default=None)
    parser.add_argument("--process-champion-checkpoint", default=None)
    parser.add_argument("--actor", default="v543_rollout")
    args = parser.parse_args(argv)
    try:
        report = run(args)
    except (RolloutError, V543Error) as error:
        print(f"[v543rollout] REFUSED {error}", file=sys.stderr)
        return 2
    target = Path(args.out)
    if not target.is_absolute():
        target = PROJECT_ROOT / target
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2, sort_keys=True,
                                 ensure_ascii=False), encoding="utf-8")
    print(f"[v543rollout] report -> {target}", file=sys.stderr)
    return 0


if __name__ == "__main__":                        # pragma: no cover
    raise SystemExit(main())
