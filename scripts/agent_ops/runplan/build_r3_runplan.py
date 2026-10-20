"""Build ``run_plan.jsonl`` for campaign PES2TS_30H_R3_V52_V53 (E0 artifact).

Emits the plan *exactly* per the 30h plan
(``docs/plans/PES2TS_30h_R3_V52_V53_分层训练与验收计划_v1.1``):

Section 4.2 -- R3, 66 new runs (``--mode full``, default):

===========  ==========================================  =========
substage     experiment                                  new runs
===========  ==========================================  =========
F-main       F0/F1/F2/F3, forward-only, fold0 x 3            12
F-bidir      F2 A->B + B->A, equal curve budget, fold0 x 3     3
F-ablate     F2 intent-off full mask (PD-6), fold0 x 3         3
B            winning-F B0-F, folds1-4 x 3 (fold0 reuse)       12
L            S2/S10/D2/D10, fold0 x 3                         12
V            top-2, folds1-2 x 3                              12
R            R1/R2 ramps, fold0 x 3 (R0 reuse)                 6
C            dev-positive top-1, folds3-4 x 3                  6
===========  ==========================================  =========

Section 5.5 -- V5.2, 33 new runs: Q 9 / T 9 / QT 9 (folds0-2 x 3) +
QT-confirm 6 (folds3-4 x 3); the M0 Move-only baseline is a *reuse* of the
frozen R3 checkpoint (0 new runs) and is recorded in per-run decision notes.

Section 4.2 degraded set -- ``--mode partial`` (PD-4, 21 runs): B fold0 x 3 +
F-bidir x 3 + F-ablate x 3 + L x 12; header ``campaign_mode: "partial"``,
every line ``exploratory_only: true``.

Two-stage identity binding (ORA-2 blocker 1): by default (stage ``e0``)
``config_sha256`` is REAL (sha256 of the frozen campaign yaml
``config/campaigns/pes2ts_30h_20261020.yaml``) while data/teacher/split carry
the yaml's ``PENDING_E1`` token and features the ``PENDING_E2`` token; at
stage ``g0``/``launch`` all five shas must be real 64-hex digests (rebuild
with ``--shas``).  The header records a per-sha ``sha_status`` map (never a
misleading single "provided_shas" flag) plus the ``two_stage_binding`` note.
Zero (all-zeros) shas are refused at build time.

Every line: ``run_id`` (sha256[:16] identity hash -- see
``run_ledger.compute_run_id(..., scheme="hashed")``), ``campaign_id``,
``phase``, ``substage``, ``arm``, ``fold``, ``seed``, ``variant``,
``reuses``, ``status`` (``planned``), ``confirm`` (explicit ORA-2/2 flag,
true only on C / QT-confirm fold3/4 lines), ``exploratory_only``, the five
identity shas and ``decision_note``.  Full-mode header carries the top-level
``pd3_note`` (PD-3 conditional); partial-mode header carries ``pd4_note``.

Writing refuses to overwrite an existing plan unless ``force`` (section 7.1
frozen file).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from scripts.agent_ops.ledger.run_ledger import HASH_RE, compute_run_id

__all__ = [
    "CAMPAIGN_ID",
    "DEFAULT_CAMPAIGN_YAML_SUBPATH",
    "DEFAULT_OUT_SUBPATH",
    "PD3_NOTE",
    "PD4_NOTE",
    "PARTIAL_SUBSTAGE_COUNTS",
    "PENDING_TOKENS",
    "SEEDS",
    "SUBSTAGE_COUNTS",
    "TWO_STAGE_BINDING_NOTE",
    "build_plan",
    "classify_sha",
    "main",
    "write_plan",
]

CAMPAIGN_ID = "PES2TS_30H_R3_V52_V53"
SEEDS = (17, 29, 43)
#: section 7.1: run_plan.jsonl lives beside the frozen protocol files
DEFAULT_OUT_SUBPATH = "outputs/pesn/30h_campaign/run_plan.jsonl"
#: the frozen campaign protocol = this yaml (ORA-2/4c) -> config identity sha
DEFAULT_CAMPAIGN_YAML_SUBPATH = "config/campaigns/pes2ts_30h_20261020.yaml"

#: stage-e0 pending tokens, mirroring the campaign yaml ``freeze_hashes`` block
PENDING_TOKENS = ("PENDING_E1", "PENDING_E2")

PD3_NOTE = (
    "extra-3 beyond 63 = F-bidir per 30h §4.2; matched forward-only control = "
    "F2-forward fold0×3 inside F-main (reuse, not new runs); flagged for human "
    "confirm at E0 sign-off"
)
PD4_NOTE = (
    "PD-4: the 21-run partial campaign is an independent exploratory batch only "
    "(21 runs: B fold0 + F-bidir + F-ablate + L); it never counts as a full R3 "
    "outcome and scale_controlled/V5.1b approval is forbidden from these runs"
)
TWO_STAGE_BINDING_NOTE = (
    "two-stage run_id binding: stage e0 = real config_sha256 (campaign yaml) + "
    "PENDING_E1/PENDING_E2 tokens for data/teacher/features/split; stage "
    "g0/launch = all five shas real 64-hex (G0/G2 freeze) -- zero shas are "
    "never valid at any stage"
)

SHA_FIELDS = (
    "config_sha256",
    "data_sha256",
    "teacher_sha256",
    "features_sha256",
    "split_sha256",
)

#: exact per-substage new-run counts, full campaign (30h §4.2 + §5.5)
SUBSTAGE_COUNTS = {
    "F-main": 12,
    "F-bidir": 3,
    "F-ablate": 3,
    "B": 12,
    "L": 12,
    "V": 12,
    "R": 6,
    "C": 6,
    "Q": 9,
    "T": 9,
    "QT": 9,
    "QT-confirm": 6,
}
#: PD-4 degraded set (30h §4.2 "若预算严格" block)
PARTIAL_SUBSTAGE_COUNTS = {"B": 3, "F-bidir": 3, "F-ablate": 3, "L": 12}
PHASE_COUNTS = {"R3": 66, "V5.2": 33}
PARTIAL_PHASE_COUNTS = {"R3": 21}


def classify_sha(value: Any) -> str:
    """Classify an identity sha: ``real`` | ``zero`` | ``PENDING_E1`` |
    ``PENDING_E2`` | ``invalid`` (used by both builder header and verifier)."""
    if value == "PENDING_E1" or value == "PENDING_E2":
        return str(value)
    if isinstance(value, str) and HASH_RE.match(value):
        return "zero" if value == "0" * 64 else "real"
    return "invalid"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(str(path), "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _default_shas(campaign_root: Path) -> Dict[str, str]:
    """Stage-e0 defaults: real config (campaign yaml) + yaml PENDING tokens."""
    yaml_path = campaign_root / DEFAULT_CAMPAIGN_YAML_SUBPATH
    if not yaml_path.is_file():
        raise BuildError(
            "CAMPAIGN_YAML_MISSING: cannot derive config_sha256 from {} "
            "(pass --shas with an explicit config_sha256 instead)".format(yaml_path)
        )
    return {
        "config_sha256": _sha256_file(yaml_path),
        "data_sha256": "PENDING_E1",
        "teacher_sha256": "PENDING_E1",
        "features_sha256": "PENDING_E2",
        "split_sha256": "PENDING_E1",
    }


def _sha256s(shas: Optional[Dict[str, str]], campaign_root: Path) -> Dict[str, str]:
    merged = _default_shas(campaign_root)
    if shas is not None:
        if not isinstance(shas, dict):
            raise BuildError("INVALID_SHAS: --shas must be a JSON object")
        for key, value in shas.items():
            if key not in SHA_FIELDS:
                raise BuildError("INVALID_SHAS: unknown field {!r}".format(key))
            status = classify_sha(value)
            if status == "invalid":
                raise BuildError(
                    "INVALID_SHAS: {} must be a lowercase 64-char sha256 hex digest "
                    "or one of {}".format(key, "/".join(PENDING_TOKENS))
                )
            if status == "zero":
                raise BuildError(
                    "INVALID_SHAS: {} is an all-zeros sha (zero shas are never "
                    "valid at any stage)".format(key)
                )
            merged[key] = value
    return merged


def _sha_status(shas: Dict[str, str]) -> Dict[str, str]:
    return {field: classify_sha(shas.get(field)) for field in SHA_FIELDS}


def _identity_source(statuses: Dict[str, str]) -> str:
    if all(v == "real" for v in statuses.values()):
        return "all_real"
    if any(v == "zero" for v in statuses.values()):
        return "zero_present"
    if statuses["config_sha256"] == "real" and all(
        statuses[f] in PENDING_TOKENS
        for f in ("data_sha256", "teacher_sha256", "features_sha256", "split_sha256")
    ):
        return "two_stage_pending"
    return "mixed"


class BuildError(Exception):
    """Builder input/output problem (always fail-closed)."""


def _run_line(
    *,
    phase: str,
    substage: str,
    arm: str,
    fold: int,
    seed: int,
    variant: Optional[str],
    reuses: List[str],
    decision_note: str,
    shas: Dict[str, str],
    confirm: bool,
    exploratory_only: bool,
) -> Dict[str, Any]:
    record: Dict[str, Any] = {
        "campaign_id": CAMPAIGN_ID,
        "phase": phase,
        "arm": arm,
        "fold": fold,
        "seed": seed,
        "variant": variant,
    }
    record.update(shas)
    line: Dict[str, Any] = {
        "run_id": compute_run_id(record, scheme="hashed"),
        "campaign_id": CAMPAIGN_ID,
        "phase": phase,
        "substage": substage,
        "arm": arm,
        "fold": fold,
        "seed": seed,
        "variant": variant,
        "reuses": list(reuses),
        "status": "planned",
        "confirm": bool(confirm),
        "exploratory_only": bool(exploratory_only),
        "decision_note": decision_note,
    }
    line.update(shas)
    return line


_NOTE_F0 = (
    "F0 forward-only fold0 screening (30h §4.2); fold0 is screening only, "
    "cannot claim generalization"
)
_NOTE_F1 = (
    "F1 forward-only fold0 screening (30h §4.2); representation pool for "
    "V5.2 is restricted to F2-intent-on and F3"
)
_NOTE_F2 = (
    "F2 forward-only fold0 screening (30h §4.2); F2-forward fold0×3 doubles as "
    "the matched forward-only control for F-bidir (PD-3: reuse, not new runs)"
)
_NOTE_F3 = (
    "F3 forward-only fold0 screening (30h §4.2); F3 (full directed intent) is "
    "in the V5.2 representation pool"
)
_NOTE_B = (
    "B0-F on winning F features, folds1-4×3 (30h §4.2); B-fold0→F-winner: the "
    "fold0 baseline reuses the F-main winner fold0 run (no new launch); folds3/4 "
    "metrics sealed until one-shot metric_unseal"
)
_NOTE_F_BIDIR = (
    "PD-3 conditional extra-3: F-bidir beyond the 63-run base per 30h §4.2; "
    "matched forward-only control = F2-forward fold0×3 inside F-main (reuse, not "
    "new runs); equal curve/fold/budget discipline; human confirm at E0 sign-off"
)
_NOTE_F_ABLATE = (
    "F2 intent-off full mask per PD-6 (ordered src/tgt + signed edit + intent_role "
    "+ directed CV + implicit direction encoding; legal undirected chemistry "
    "context kept); scientific negative control, fold0×3 (30h §4.2)"
)
_NOTE_L = (
    "aux arm fold0×3 (30h §4.2); λ calibrated on the new R3 B0-F checkpoint "
    "(§4.1), scale_unreachable if gradient share <1%, λ=0 must be bit-identical "
    "to the B0-F path"
)
_NOTE_V = (
    "top-2 dev selection, folds1-2×3 (30h §4.2); RMSE/SRE + tolerances frozen at "
    "the R3 protocol values; no confirm-fold tuning"
)
_NOTE_R = (
    "ramp diagnostic fold0×3 (30h §4.2); R-R0→main: the R0 baseline (main "
    "schedule) reuses the existing fold0 baseline run -- no new launch"
)
_NOTE_C = (
    "dev-positive top-1 confirm folds3-4×3 (30h §4.2); C runs are explicit "
    "confirm runs (confirm: true): one-shot BATCH unseal token "
    "(state/gates/unseal.json, pre-registered run set), one-shot metrics, no "
    "threshold feedback"
)
_NOTE_M0 = (
    "M0→R3 checkpoint: the Move-only M0 baseline (§5.5) reuses the frozen R3 "
    "checkpoint -- 0 new runs; identical curve data and training budget, head "
    "loss switches only"
)
_NOTE_Q = "Move+Progress head, folds0-2×3 (30h §5.5); " + _NOTE_M0
_NOTE_T = "Move+TSStop head, folds0-2×3 (30h §5.5); " + _NOTE_M0
_NOTE_QT = "Move+Progress+TSStop head, folds0-2×3 (30h §5.5); " + _NOTE_M0
_NOTE_QT_CONFIRM = (
    "preregistered dev-positive QT confirm folds3-4×3 (30h §5.5); explicit "
    "confirm run (confirm: true) inside the one-shot BATCH unseal; " + _NOTE_M0
)
_NOTE_PARTIAL_B = (
    "PD-4 partial: B0-F baseline fold0×3 standalone (exploratory only; no "
    "F-screening in the 21-run degraded set)"
)
_NOTE_PARTIAL_F_BIDIR = (
    "PD-4 partial: F-bidir fold0×3 exploratory (21-run degraded set; never a "
    "full-R3 outcome)"
)
_NOTE_PARTIAL_F_ABLATE = (
    "PD-4 partial: F2 intent-off full-mask (PD-6) fold0×3 exploratory "
    "(21-run degraded set)"
)
_NOTE_PARTIAL_L = (
    "PD-4 partial: S2/S10/D2/D10 fold0×3 exploratory (21-run degraded set)"
)


def build_plan(
    shas: Optional[Dict[str, str]] = None,
    mode: str = "full",
    campaign_root: Optional[Any] = None,
) -> List[Dict[str, Any]]:
    """Return the full plan: ``[plan_header, run_line, ...]``.

    ``mode`` is ``"full"`` (66 R3 + 33 V5.2 = 99 lines) or ``"partial"``
    (PD-4 21-run exploratory degraded set).
    """
    if mode not in ("full", "partial"):
        raise BuildError("INVALID_MODE: {!r} not in full/partial".format(mode))
    root = Path(campaign_root) if campaign_root is not None else Path(
        __file__
    ).resolve().parents[3]
    sha_map = _sha256s(shas, root)
    statuses = _sha_status(sha_map)
    partial = mode == "partial"

    header: Dict[str, Any] = {
        "record_type": "plan_header",
        "campaign_id": CAMPAIGN_ID,
        "campaign_mode": mode,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "budget": dict(PARTIAL_PHASE_COUNTS if partial else PHASE_COUNTS),
        "counting": "all launch attempts incl. retries (task_state budget.run_caps)",
        "seeds": list(SEEDS),
        "sha_status": statuses,
        "identity_sha_source": _identity_source(statuses),
        "two_stage_binding": TWO_STAGE_BINDING_NOTE,
    }
    if partial:
        header["matrix_ref"] = ["30h plan §4.2 degraded set (PD-4, partial=21)"]
        header["pd4_note"] = PD4_NOTE
    else:
        header["matrix_ref"] = [
            "30h plan §4.2 (R3=66)",
            "30h plan §5.5 (V5.2=33)",
        ]
        header["pd3_note"] = PD3_NOTE

    lines: List[Dict[str, Any]] = [header]

    def add(
        phase,
        substage,
        arm,
        fold,
        seed,
        variant,
        reuses,
        note,
        confirm=False,
    ):
        lines.append(
            _run_line(
                phase=phase,
                substage=substage,
                arm=arm,
                fold=fold,
                seed=seed,
                variant=variant,
                reuses=reuses,
                decision_note=note,
                shas=sha_map,
                confirm=confirm,
                exploratory_only=partial,
            )
        )

    if partial:
        # PD-4 21-run degraded set: B fold0×3 + F-bidir×3 + F-ablate×3 + L×12
        for seed in SEEDS:
            add("R3", "B", "B", 0, seed, None, [], _NOTE_PARTIAL_B)
        for seed in SEEDS:
            add("R3", "F-bidir", "F2-bidir", 0, seed, "bidir", [], _NOTE_PARTIAL_F_BIDIR)
        for seed in SEEDS:
            add(
                "R3",
                "F-ablate",
                "F2-intent-off",
                0,
                seed,
                "full_mask_PD6",
                [],
                _NOTE_PARTIAL_F_ABLATE,
            )
        for arm in ("S2", "S10", "D2", "D10"):
            for seed in SEEDS:
                add("R3", "L", arm, 0, seed, None, [], _NOTE_PARTIAL_L)
        return lines

    # ---- R3 (30h §4.2): 12 + 3 + 3 + 12 + 12 + 12 + 6 + 6 = 66 -------------
    f_notes = {"F0": _NOTE_F0, "F1": _NOTE_F1, "F2": _NOTE_F2, "F3": _NOTE_F3}
    for arm in ("F0", "F1", "F2", "F3"):  # F-main: 4 arms x fold0 x 3 seeds
        for seed in SEEDS:
            add("R3", "F-main", arm, 0, seed, "forward", [], f_notes[arm])
    for seed in SEEDS:  # F-bidir: F2 bidir fold0 x 3
        add("R3", "F-bidir", "F2-bidir", 0, seed, "bidir", [], _NOTE_F_BIDIR)
    for seed in SEEDS:  # F-ablate: F2 intent-off full-mask (PD-6) fold0 x 3
        add("R3", "F-ablate", "F2-intent-off", 0, seed, "full_mask_PD6", [], _NOTE_F_ABLATE)
    for fold in (1, 2, 3, 4):  # B: winner folds1-4 x 3, fold0 reuse link in note
        for seed in SEEDS:
            add("R3", "B", "B", fold, seed, None, ["substage:F-main"], _NOTE_B)
    for arm in ("S2", "S10", "D2", "D10"):  # L: 4 arms x fold0 x 3 seeds
        for seed in SEEDS:
            add("R3", "L", arm, 0, seed, None, ["substage:F-main"], _NOTE_L)
    for arm in ("V-top1", "V-top2"):  # V: top-2 folds1-2 x 3
        for fold in (1, 2):
            for seed in SEEDS:
                add("R3", "V", arm, fold, seed, None, ["substage:B"], _NOTE_V)
    for arm in ("R1", "R2"):  # R: ramps fold0 x 3; R0 reuse in note
        for seed in SEEDS:
            add("R3", "R", arm, 0, seed, None, ["substage:F-main"], _NOTE_R)
    for fold in (3, 4):  # C: dev-positive top-1 confirm folds3-4 x 3
        for seed in SEEDS:
            add("R3", "C", "C", fold, seed, None, ["substage:B"], _NOTE_C, confirm=True)

    # ---- V5.2 (30h §5.5): 9 + 9 + 9 + 6 = 33 -------------------------------
    for arm, note in (("Q", _NOTE_Q), ("T", _NOTE_T), ("QT", _NOTE_QT)):
        for fold in (0, 1, 2):
            for seed in SEEDS:
                add("V5.2", arm, arm, fold, seed, None, [], note)
    for fold in (3, 4):  # QT-confirm: preregistered dev-positive, batch unseal
        for seed in SEEDS:
            add(
                "V5.2",
                "QT-confirm",
                "QT",
                fold,
                seed,
                "confirm",
                ["substage:QT"],
                _NOTE_QT_CONFIRM,
                confirm=True,
            )
    return lines


def write_plan(
    out_path: Any,
    shas: Optional[Dict[str, str]] = None,
    force: bool = False,
    mode: str = "full",
    campaign_root: Optional[Any] = None,
) -> Dict[str, Any]:
    """Build and atomically write the plan; refuses to overwrite unless ``force``."""
    out = Path(out_path)
    if out.exists() and not force:
        raise BuildError(
            "PLAN_EXISTS: {} already exists (section 7.1 frozen artifact); "
            "pass force=True/--force to rebuild".format(out)
        )
    lines = build_plan(shas=shas, mode=mode, campaign_root=campaign_root)
    out.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(out.parent), prefix=out.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            for line in lines:
                handle.write(json.dumps(line, sort_keys=True, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, str(out))
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    run_lines = [ln for ln in lines if ln.get("record_type") != "plan_header"]
    header = lines[0]
    summary = {
        "out": str(out),
        "campaign_mode": mode,
        "runs": len(run_lines),
        "unique_run_ids": len({ln["run_id"] for ln in run_lines}),
        "sha_status": header["sha_status"],
        "identity_sha_source": header["identity_sha_source"],
    }
    for phase in ("R3", "V5.2"):
        count = sum(1 for ln in run_lines if ln["phase"] == phase)
        if count:
            summary[phase] = count
    summary["pd3_note"] = PD3_NOTE if mode == "full" else None
    if mode == "partial":
        summary["pd4_note"] = PD4_NOTE
    return summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="build_r3_runplan",
        description="Build the E0 run_plan.jsonl (full: 66 R3 + 33 V5.2 unique "
        "planned runs; partial: PD-4 21-run exploratory set)",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="output run_plan.jsonl (default: <root>/outputs/pesn/30h_campaign/run_plan.jsonl)",
    )
    parser.add_argument(
        "--root",
        default=str(Path(__file__).resolve().parents[3]),
        help="campaign root used to resolve the default --out path and the "
        "campaign yaml config_sha256",
    )
    parser.add_argument(
        "--shas",
        default=None,
        help="JSON file with identity shas (config/data/teacher/features/split); "
        "default: real campaign-yaml config sha + PENDING_E1/E2 tokens (stage e0)",
    )
    parser.add_argument(
        "--mode",
        choices=("full", "partial"),
        default="full",
        help="full = 99-run plan; partial = PD-4 21-run exploratory degraded set",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="overwrite an existing run_plan.jsonl (frozen-file rebuild)",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        shas = None
        if args.shas:
            try:
                shas = json.loads(Path(args.shas).read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise BuildError("INVALID_SHAS: cannot read {}: {}".format(args.shas, exc))
        out = Path(args.out) if args.out else Path(args.root) / DEFAULT_OUT_SUBPATH
        summary = write_plan(
            out, shas=shas, force=args.force, mode=args.mode, campaign_root=args.root
        )
    except BuildError as exc:
        sys.stderr.write("error: {}\n".format(exc))
        return 2
    print(json.dumps(summary, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
