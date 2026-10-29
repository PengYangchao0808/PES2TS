#!/usr/bin/env python
"""V5.4.1 WP7 — frozen dev evaluation, matched E0 x seeds 17/29/43 ONLY.

WHAT THIS IS
------------
The single registered WP7 ``candidate_evaluation`` dev access (plan §10.1).
The single-seed screen already adjudicated ``E* = E0``:

* E1/E2/E3  scientific_failure  (H2 MOVE guard)  — NEVER evaluated on dev;
* E4/E5/T3/T4 gate_skipped                        — NEVER evaluated on dev;
* E0 (T0-R matched retrain x 17/29/43) is the ONLY frozen set on dev.

ORDER OF OPERATIONS (fail-closed, plan §10.1.1-3)
-------------------------------------------------
1. ``freeze``  — writes ``reports/wp7_freeze.json`` binding: screen SHA, the
   three E0 ``checkpoint_best`` SHAs, the per-seed frozen inner-val tau,
   runplan/code/statistics-code SHAs, dev access type
   ``candidate_evaluation``, confirm=sealed.  NO dev access in this stage
   (only byte hashing + publicly readable audits: denominator COUNTS and
   ledger SHAs — never sealed science values).
2. ``run``     — FIRST re-verifies every freeze binding, THEN verifies the
   WP1 sealed baseline (manifest + 5 sealed files + evaluator report +
   ledger prefix) is byte-identical; only then may the dev corpus be opened
   (frozen dev feature pack, dual position-SHA check) and each E0 seed
   evaluated with ONE dev prediction pass each.
3. ``verify``  — read-only re-check of bindings, outputs and ledger.

PROTOCOL PER SEED (mirrors the WP1 dev stage tooling exactly)
-------------------------------------------------------------
* model:  ``load_model_v541`` (R3 strict warmstart) + ``strict=True``;
* tau:    READ-ONLY from ``checkpoint_best.pt#tau_end_calibration``
          (inner-val frozen; never recalibrated on dev);
* train_q_mean: ONLY from this seed's inner-train slice of the v53mp pool
  split (``v53mp_pool_split`` + ``_train_q_mean``) — the constant-q
  baseline source; never derived from dev;
* metrics: same ONE-pass chunk-64 composition, repaired pair tooling
  (WP1a ``assign_batch_pairs``), MOVE / PROCESS / denominators / sanity as
  ``v541_baseline_repair._eval_split``;
* R3 reference: REUSED from the WP1 sealed ``r3_reference_dev.json``
  (same dev positions, SHA-verified) — R3 is NOT re-run.

HARD BOUNDARIES
---------------
* read-only on every historical/WP1 artifact; WP1 ledger lines untouched
  (append-only with a separate ``purpose=candidate_evaluation``);
* NO confirm access, NO QC calls, NO training, NO E1-E6 forward passes;
* execution single-lane: ``cgroup_scope.sh 12G 10G`` + GPU lock;
* any mismatch → typed ``WP7Error`` + fail-closed record, never a skip.

Stages: ``freeze`` → ``run`` → ``verify``.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import sys
import time
import traceback
from dataclasses import fields as dc_fields
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# WP1 / runner tooling — IMPORT-ONLY surface.  Nothing in this list may be
# modified by WP7 (freeze protection; see design §2.3 / §10).
from scripts.campaigns import v541_baseline_repair as br   # noqa: E402
from scripts.campaigns.r3_v53mp_scale_paired_ci import (    # noqa: E402
    TIER_DEFAULTS,
    _build_tier,
    _ok_mask,
)

# ---------------------------------------------------------------------------
# frozen identities / paths
# ---------------------------------------------------------------------------
SCHEMA_FREEZE = "v541_wp7_freeze_v1"
SCHEMA_SEED = "v541_wp7_dev_seed_v1"
SCHEMA_EVAL = "v541_wp7_dev_frozen_evaluation_v1"
SCHEMA_FAILURE = "v541_wp7_failure_v1"
#: SAME schema string + byte format as the WP1 ledger lines (one file).
SCHEMA_LEDGER = "v541_dev_access_ledger_v1"

CAMPAIGN_DIR = Path("outputs/pesn/v541_embed_50k")
REPORTS_DIR = CAMPAIGN_DIR / "reports"
FREEZE_REL = REPORTS_DIR / "wp7_freeze.json"
FREEZE_FAILURE_REL = REPORTS_DIR / "wp7_freeze_failure.json"
EVAL_REPORT_REL = REPORTS_DIR / "dev_frozen_evaluation.json"
PER_SEED_REL_DIR = REPORTS_DIR / "dev_frozen"
AUDIT_DIR = CAMPAIGN_DIR / "audits"
SEALED_DIR = AUDIT_DIR / "dev_baseline_sealed"
SEALED_MANIFEST_REL = SEALED_DIR / "manifest.json"
LEDGER_REL = AUDIT_DIR / "dev_access_ledger.jsonl"
INNER_VAL_JSON_REL = AUDIT_DIR / "evaluator_before_after.json"
SCREEN_REL = REPORTS_DIR / "single_seed_screen.json"
RUNS_DIR = CAMPAIGN_DIR / "runs"
R3_INIT_REL = Path(
    "outputs/pesn/30h_campaign/r3_runs/5372d9b12807c3f0/checkpoint_best.pt")
R3_INIT_SHA256 = (
    "a2e46b6ad65cc18996d6c1660c5ae24c723397ca22526e52fa34e430afd0417f")

TIER = br.TIER                              # "50k"
SEEDS: Tuple[int, ...] = (17, 29, 43)
E0_RUN_ID = {s: "V541_T0R_E0_S%d" % s for s in SEEDS}
E0_MODEL_NAME = {s: "V541_E0_T0R_S%d" % s for s in SEEDS}
EMBEDDING_ID = "E0"
V541_CKPT_SCHEMA = "pesn_v541_train_v1"
ACTOR_SUFFIX = "wp7_dev_frozen_eval"

#: dev-path eval surface whose per-file SHA is part of the code freeze.
#: WP1's own list is reused verbatim and EXTENDED with the V541-only files.
V541_CODE_FILES: Tuple[str, ...] = (
    "pes2ts_core/pesn/offline/training/v541_train.py",
    "scripts/campaigns/v541_dev_evaluate.py",
)
V541_MODEL_DIR = Path("pes2ts_core/pesn/models/v541")
#: statistics tooling bound in the freeze (WP7 reports no new CI — the SHA
#: binds the frozen paired-CI/statistics surface so it cannot drift).
STATS_CODE_FILES: Tuple[str, ...] = (
    "scripts/campaigns/r3_v53mp_scale_paired_ci.py",
    "scripts/campaigns/r3_v53mp_run.py",
)


class WP7Error(RuntimeError):
    """Typed fail-closed refusal."""

    def __init__(self, code: str, detail: str = "") -> None:
        super().__init__("%s: %s" % (code, detail) if detail else code)
        self.code = code
        self.detail = detail


# ---------------------------------------------------------------------------
# paths (injectable root — fixture-testable)
# ---------------------------------------------------------------------------


class Wp7Paths:
    """All WP7 paths relative to a project root (real tree or fixture)."""

    def __init__(self, project_root: Path) -> None:
        self.project_root = Path(project_root)
        r = self.project_root
        self.campaign_dir = r / CAMPAIGN_DIR
        self.reports_dir = r / REPORTS_DIR
        self.freeze_path = r / FREEZE_REL
        self.freeze_failure_path = r / FREEZE_FAILURE_REL
        self.eval_report_path = r / EVAL_REPORT_REL
        self.per_seed_dir = r / PER_SEED_REL_DIR
        self.audit_dir = r / AUDIT_DIR
        self.sealed_dir = r / SEALED_DIR
        self.sealed_manifest = r / SEALED_MANIFEST_REL
        self.ledger_path = r / LEDGER_REL
        self.inner_val_json = r / INNER_VAL_JSON_REL
        self.screen_path = r / SCREEN_REL
        self.runs_dir = r / RUNS_DIR
        self.r3_init = r / R3_INIT_REL
        self.tables_dir = Path(TIER_DEFAULTS[TIER]["tables_dir"])
        self.split_path = self.tables_dir / "split_manifest_v53mp.json"

    def e0_run_dir(self, seed: int) -> Path:
        return self.runs_dir / E0_RUN_ID[int(seed)]

    def per_seed_file(self, seed: int) -> Path:
        return self.per_seed_dir / ("%s_dev.json" % E0_RUN_ID[int(seed)])


def _say(msg: str) -> None:
    print("[wp7] %s" % msg, flush=True)


def _sha256_file(path: Path) -> str:
    return br._sha256_file(path)


def _sha256_text(text: str) -> str:
    return br._sha256_text(text)


def _load_json(path: Path) -> Dict[str, Any]:
    if not Path(path).is_file():
        raise WP7Error("FILE_MISSING", str(path))
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _now() -> str:
    return br._now()


def _actor() -> str:
    return "%s#%s" % (br._actor().split("#")[0], ACTOR_SUFFIX)


def _jsonable(obj: Any) -> Any:
    return br._jsonable(obj)


def _write_json(path: Path, doc: Mapping[str, Any]) -> str:
    return br._write_json(Path(path), doc)


def _read_ledger_lines(paths: Wp7Paths) -> List[str]:
    if not paths.ledger_path.is_file():
        return []
    return [ln for ln in paths.ledger_path.read_text(
        encoding="utf-8").splitlines() if ln.strip()]


#: The REAL executing tree — code identity always resolves here (immune to
#: fixture PROJECT_ROOT monkeypatching in tests).
CODE_ROOT = Path(__file__).resolve().parents[2]


def _code_identity(_paths: Optional[Wp7Paths] = None) -> Dict[str, str]:
    """Per-file SHA of the whole WP7 eval-path + statistics surface.

    Code files ALWAYS resolve against the real executing tree
    (``CODE_ROOT``) — the code identity is a property of the code that
    runs, independent of any fixture artifact root.
    """
    files: List[str] = list(br.CODE_FILES) + list(V541_CODE_FILES)
    model_dir = CODE_ROOT / V541_MODEL_DIR
    if model_dir.is_dir():
        files.extend(str(p.relative_to(CODE_ROOT))
                     for p in sorted(model_dir.rglob("*.py")))
    files.extend(STATS_CODE_FILES)
    out: Dict[str, str] = {}
    for rel in dict.fromkeys(files):          # stable, deduped
        p = CODE_ROOT / rel
        if not p.is_file():
            raise WP7Error("CODE_FILE_MISSING", rel)
        out[rel] = _sha256_file(p)
    return out


def _code_identity_sha(identity: Mapping[str, str]) -> str:
    return _sha256_text(json.dumps(
        dict(identity), sort_keys=True, separators=(",", ":")))


# ---------------------------------------------------------------------------
# stage: freeze (NO dev access — byte hashing + readable audits only)
# ---------------------------------------------------------------------------


def screen_adjudication_gate(screen: Mapping[str, Any]) -> Dict[str, Any]:
    """E*=E0 is the ONLY legal WP7 state (plan §10.1 / screen ruling)."""
    if str(screen.get("E_star")) != EMBEDDING_ID:
        raise WP7Error(
            "SCREEN_ADJUDICATION_UNEXPECTED",
            "E_star=%r — WP7 evaluates matched E0 only" % screen.get("E_star"))
    if screen.get("selected") or int(screen.get("n_selected") or 0):
        raise WP7Error("SCREEN_ADJUDICATION_UNEXPECTED",
                       "selected candidates present — dev set is not E0-only")
    for cand in screen.get("candidates") or []:
        emb = str(cand.get("embedding_id"))
        if emb != EMBEDDING_ID:
            if cand.get("outcome") != "scientific_failure":
                raise WP7Error(
                    "SCREEN_ADJUDICATION_UNEXPECTED",
                    "%s outcome=%r is not scientific_failure" % (
                        emb, cand.get("outcome")))
            if cand.get("advance_to_T3"):
                raise WP7Error("SCREEN_ADJUDICATION_UNEXPECTED",
                               "%s advanced to T3 but E*=E0" % emb)
    for key in ("E4", "E5", "T3", "T4"):
        blk = screen.get(key)
        if isinstance(blk, Mapping) and blk.get("formal_run_started"):
            raise WP7Error("SCREEN_ADJUDICATION_UNEXPECTED",
                           "%s claims a formal run" % key)
    return {
        "e_star": EMBEDDING_ID,
        "selected": [],
        "n_selected": 0,
        "forbidden_candidates": ["E1", "E2", "E3", "E4", "E5"],
        "adjudication": {
            "E1": "scientific_failure", "E2": "scientific_failure",
            "E3": "scientific_failure", "E4": "gate_skipped",
            "E5": "gate_skipped", "T3": "gate_skipped",
            "T4": "gate_skipped",
        },
        "early_termination": bool(screen.get("early_termination")),
    }


def build_freeze(paths: Wp7Paths) -> Dict[str, Any]:
    """Assemble the WP7 freeze document.  NO sealed dev science is read:
    only file BYTE hashes, plus the publicly readable audit (denominator
    counts) and ledger (SHAs)."""
    screen = _load_json(paths.screen_path)
    screen_block = screen_adjudication_gate(screen)
    screen_block["path"] = str(paths.screen_path.relative_to(
        paths.project_root))
    screen_block["sha256"] = _sha256_file(paths.screen_path)
    screen_block["policy"] = str(screen.get("dev_policy") or "")

    # ---- frozen set: E0 T0-R x 3 seeds (checkpoint/metrics cross-check) --
    per_seed: Dict[str, Any] = {}
    tau_block: Dict[str, Any] = {}
    for seed in SEEDS:
        run_dir = paths.e0_run_dir(seed)
        ckpt, mpath = run_dir / "checkpoint_best.pt", run_dir / "metrics.json"
        if not ckpt.is_file() or not mpath.is_file():
            raise WP7Error("E0_ASSET_MISSING", "%s / %s" % (ckpt, mpath))
        ckpt_sha = _sha256_file(ckpt)
        m = _load_json(mpath)
        prov = m.get("provenance") or {}
        ident = m.get("identity") or {}
        checks = {
            "ckpt_eq_metrics_sha": ckpt_sha == m.get("checkpoint_best_sha256"),
            "schema_is_v541": m.get("schema") == V541_CKPT_SCHEMA,
            "embedding_is_e0": (m.get("embedding_id") == EMBEDDING_ID
                                and ident.get("embedding_id") == EMBEDDING_ID),
            "seed_match": int(m.get("seed", -1)) == int(seed)
            and int(ident.get("seed", -2)) == int(seed),
            "run_outcome_done": m.get("run_outcome") == "done",
            "is_formal_run": bool(m.get("is_formal_run")),
            "eval_scope_inner_val": m.get("eval_scope") == "inner_val_only",
            "confirm_read_false": prov.get("confirm_read") is False
            and m.get("confirm_read") is False,
            "r3_init_sha": prov.get("r3_init_sha256") == R3_INIT_SHA256,
            "tau_on_inner_val": m.get("tau_calibrated_on") == "inner_val",
        }
        bad = [k for k, v in checks.items() if not v]
        if bad:
            raise WP7Error("E0_FREEZE_CROSSCHECK_FAILED",
                           "seed %d failed: %s" % (seed, ", ".join(bad)))
        tau = m.get("tau_end_calibration") or {}
        if tau.get("tau_end") is None:
            raise WP7Error("TAU_MISSING_IN_METRICS", "seed %d" % seed)
        per_seed[str(seed)] = {
            "run_id": E0_RUN_ID[seed],
            "checkpoint_path": str(ckpt.relative_to(paths.project_root)),
            "checkpoint_best_sha256": ckpt_sha,
            "metrics_path": str(mpath.relative_to(paths.project_root)),
            "metrics_sha256": _sha256_file(mpath),
            "config_full_sha256": m.get("config_full_sha256"),
            "r3_init_sha256": prov.get("r3_init_sha256"),
            "schema": m.get("schema"),
            "run_outcome": m.get("run_outcome"),
            "is_formal_run": True,
            "embedding_id": EMBEDDING_ID,
            "n_train_samples_inner": m.get("n_train_samples"),
            "n_inner_val_samples": m.get("n_inner_val_samples"),
        }
        tau_block[str(seed)] = {
            "tau_end": tau.get("tau_end"),
            "target_precision": tau.get("target_precision"),
            "achieved_precision": tau.get("achieved_precision"),
            "achieved_recall": tau.get("achieved_recall"),
            "prevalence": tau.get("prevalence"),
            "n_inner_val_unmasked": tau.get("n_inner_val_unmasked"),
            "fallback": tau.get("fallback"),
            "source": "checkpoint_best.pt#tau_end_calibration",
            "calibrated_on": "inner_val",
            "recomputed_on_dev": False,
        }

    r3_sha = _sha256_file(paths.r3_init)
    if r3_sha != R3_INIT_SHA256:
        raise WP7Error("R3_INIT_DRIFT",
                       "%s sha %s" % (paths.r3_init, r3_sha[:16]))

    # ---- plan / runplan / statistics bindings ---------------------------
    pin_path = paths.campaign_dir / "plan" / "runplan.pin.json"
    pin = _load_json(pin_path)
    if list(pin.get("seeds") or []) != list(SEEDS):
        raise WP7Error("RUNPLAN_SEEDS_MISMATCH", str(pin.get("seeds")))
    plan_bindings = {
        "runplan_pin_path": str(pin_path.relative_to(paths.project_root)),
        "runplan_pin_sha256": _sha256_file(pin_path),
        "runplan_sha256": pin.get("runplan_sha256"),
        "code_sha256": pin.get("code_sha256"),
        "config_sha256": pin.get("config_sha256"),
        "budget_sha256": pin.get("budget_sha256"),
        "source_manifest_sha256": pin.get("source_manifest_sha256"),
        "common_protocol_sha256": pin.get("common_protocol_sha256"),
        "environment_sha256": pin.get("environment_sha256"),
        "code_commit": pin.get("code_commit"),
        "seeds": list(SEEDS),
    }
    identity = _code_identity(paths)
    code_bindings = {
        "eval_path_files": identity,
        "stats_code_files": {rel: identity[rel] for rel in STATS_CODE_FILES},
        "evaluator_code_sha256": _code_identity_sha(identity),
        "wp7_script_sha256": identity["scripts/campaigns/v541_dev_evaluate.py"],
    }

    # ---- WP1 seal bindings (byte hashes only) ---------------------------
    man_sha = _sha256_file(paths.sealed_manifest)
    man = _load_json(paths.sealed_manifest)
    files = dict(man.get("files") or {})
    if not files:
        raise WP7Error("WP1_SEAL_EMPTY", str(paths.sealed_manifest))
    lines = _read_ledger_lines(paths)
    if not lines:
        raise WP7Error("WP1_LEDGER_EMPTY", str(paths.ledger_path))
    # WP1段 = the maximal LEADING run of baseline_repair entries; any
    # following line must be a WP7 candidate_evaluation entry (append-only
    # supersession — never a rewrite of the WP1 block).
    wp1_entries: List[Dict[str, Any]] = []
    n_wp1 = 0
    for ln in lines:
        rec = json.loads(ln)
        if rec.get("purpose") == "baseline_repair":
            wp1_entries.append(rec)
            n_wp1 += 1
            continue
        break
    for ln in lines[n_wp1:]:
        if json.loads(ln).get("purpose") != "candidate_evaluation":
            raise WP7Error(
                "WP1_LEDGER_UNEXPECTED_LINE",
                "non-candidate_evaluation entry after the WP1 block: %r"
                % json.loads(ln).get("purpose"))
    if not wp1_entries:
        raise WP7Error("WP1_LEDGER_EMPTY", "no baseline_repair entries")
    last_seal = [r for r in wp1_entries if r.get("model") == "seal_manifest"]
    if not last_seal or last_seal[-1].get("output_sha256") != man_sha:
        raise WP7Error("WP1_SEAL_LEDGER_MISMATCH",
                       "ledger seal_manifest tail != current manifest sha")
    # WP1段 byte prefix (lines joined with their original newlines)
    raw = paths.ledger_path.read_bytes()
    prefix = b"".join(ln.encode("utf-8") + b"\n" for ln in lines[:n_wp1])
    if not raw.startswith(prefix):
        raise WP7Error("WP1_LEDGER_PREFIX_UNSTABLE",
                       "WP1 block is not a stable byte prefix")
    ev = _load_json(paths.inner_val_json)
    if ev.get("status") != "ok" or not (ev.get("dev_sealed_status") or {}).get(
            "pass"):
        raise WP7Error("WP1_EVALUATOR_NOT_OK",
                       "status=%r" % ev.get("status"))
    den = (ev.get("dev_sealed_status") or {}).get("denominator_sanity") or {}
    # devpack positions SHA + curve SHA come from the READABLE ledger (not
    # from sealed science values)
    ref_entry = next((r for r in wp1_entries
                      if r.get("model") == "r3_reference"), None)
    if ref_entry is None:
        raise WP7Error("WP1_LEDGER_NO_R3_ENTRY", "")
    wp1_seal = {
        "sealed_manifest_path": str(paths.sealed_manifest.relative_to(
            paths.project_root)),
        "sealed_manifest_sha256": man_sha,
        "files": files,
        "ledger_path": str(paths.ledger_path.relative_to(paths.project_root)),
        "ledger_prefix_sha256": _sha256_text(prefix.decode("utf-8")),
        "ledger_entries_wp1": n_wp1,
        "evaluator_report_path": str(paths.inner_val_json.relative_to(
            paths.project_root)),
        "evaluator_report_sha256": _sha256_file(paths.inner_val_json),
        "status": "ok",
        "dev_sealed_pass": True,
        "unseal_rule": str(man.get("unseal_rule") or ""),
    }
    dev_reference_binding = {
        "r3_reference_dev_sha256": files.get("r3_reference_dev.json"),
        "n_dev_positions": int(den.get("n_dev_positions") or -1),
        "n_move_ok_positions": int(den.get("n_move_ok_positions") or -1),
        "n_dev_curves": int(den.get("n_dev_curves") or -1),
        "devpack_positions_sha256": ref_entry.get("positions_sha256"),
        "curves_sha256": ref_entry.get("curve_set_sha256"),
        "source": ("readable audits only: evaluator_before_after.json "
                   "denominator COUNTS + dev_access_ledger.jsonl SHAs"),
    }

    git = {}
    try:
        git = {
            "git_head": br._git("rev-parse", "HEAD").strip(),
            "git_diff_sha256": _sha256_text(br._git("diff", "HEAD")),
        }
    except Exception:                                     # pragma: no cover
        git = {"git_head": None, "git_diff_sha256": None}

    return {
        "schema": SCHEMA_FREEZE,
        "created_at": _now(),
        "wp": "WP7",
        "dev_access": {
            "type": "candidate_evaluation",
            "purpose": "frozen_dev_evaluation_matched_e0_only",
            "split": "dev",
            "n_planned_evaluations": len(SEEDS),
            "policy": ("matched E0 x3 only; no candidate dev access; "
                       "confirm remains sealed"),
        },
        "screen": screen_block,
        "frozen_set": {"embedding_id": EMBEDDING_ID, "seeds": list(SEEDS),
                       "per_seed": per_seed},
        "inner_val_tau": {"calibrated_on": "inner_val", "per_seed": tau_block},
        "plan_bindings": plan_bindings,
        "code_bindings": code_bindings,
        "wp1_seal": wp1_seal,
        "dev_reference_binding": dev_reference_binding,
        "confirm": {"status": "sealed", "confirm_read": False,
                    "new_QC_calls": 0},
        "provenance": {"actor": _actor(), **git},
    }


def verify_freeze(paths: Wp7Paths, freeze: Mapping[str, Any]) -> None:
    """Recompute every binding and compare (fail-closed)."""
    if freeze.get("schema") != SCHEMA_FREEZE:
        raise WP7Error("FREEZE_SCHEMA", str(freeze.get("schema")))
    if freeze.get("dev_access", {}).get("type") != "candidate_evaluation":
        raise WP7Error("FREEZE_DEV_ACCESS_TYPE",
                       str(freeze.get("dev_access")))
    if freeze.get("confirm", {}).get("status") != "sealed" \
            or freeze.get("confirm", {}).get("confirm_read") is not False:
        raise WP7Error("FREEZE_CONFIRM_STATE", str(freeze.get("confirm")))
    current = build_freeze(paths)
    for key in ("screen", "frozen_set", "inner_val_tau", "plan_bindings",
                "code_bindings", "wp1_seal", "dev_reference_binding"):
        if current.get(key) != freeze.get(key):
            drifted = sorted(k for k in set(current.get(key) or {})
                             | set(freeze.get(key) or {})
                             if (current.get(key) or {}).get(k)
                             != (freeze.get(key) or {}).get(k))
            raise WP7Error("FREEZE_DRIFT_%s" % key.upper(),
                           "drifted keys: %s" % (drifted[:8] or key))
    _say("freeze bindings verified (%d code files)" % len(
        current["code_bindings"]["eval_path_files"]))


# ---------------------------------------------------------------------------
# stage: WP1 seal verification (dev-unseal precondition)
# ---------------------------------------------------------------------------


def verify_wp1_seal(paths: Wp7Paths, freeze: Mapping[str, Any]) -> Dict[
        str, Any]:
    recorded = freeze.get("wp1_seal") or {}
    man_sha = _sha256_file(paths.sealed_manifest)
    if man_sha != recorded.get("sealed_manifest_sha256"):
        raise WP7Error("WP1_SEAL_MANIFEST_DRIFT", man_sha[:16])
    man = _load_json(paths.sealed_manifest)
    files = dict(man.get("files") or {})
    if files != recorded.get("files"):
        raise WP7Error("WP1_SEAL_FILE_TABLE_DRIFT", "manifest.files changed")
    for name, want in files.items():
        p = paths.sealed_dir / name
        if not p.is_file():
            raise WP7Error("WP1_SEAL_FILE_MISSING", name)
        got = _sha256_file(p)
        if got != want:
            raise WP7Error("WP1_SEAL_FILE_DRIFT",
                           "%s %s != %s" % (name, got[:16], want[:16]))
    lines = _read_ledger_lines(paths)
    n_wp1 = int(recorded.get("ledger_entries_wp1") or 0)
    if len(lines) < n_wp1:
        raise WP7Error("WP1_LEDGER_TRUNCATED",
                       "%d < %d" % (len(lines), n_wp1))
    prefix = b"".join(ln.encode("utf-8") + b"\n" for ln in lines[:n_wp1])
    if _sha256_text(prefix.decode("utf-8")) != recorded.get(
            "ledger_prefix_sha256"):
        raise WP7Error("WP1_LEDGER_PREFIX_DRIFT", "WP1 lines changed")
    for ln in lines[:n_wp1]:
        if json.loads(ln).get("purpose") != "baseline_repair":
            raise WP7Error("WP1_LEDGER_PURPOSE_DRIFT", "")
    tail = [json.loads(ln) for ln in lines[n_wp1:]]
    bad = [r.get("purpose") for r in tail
           if r.get("purpose") != "candidate_evaluation"]
    if bad:
        raise WP7Error("LEDGER_POST_WP1_PURPOSE", str(bad[:3]))
    seal_tail = [r for r in lines[:n_wp1]
                 if json.loads(r).get("model") == "seal_manifest"]
    if not seal_tail or json.loads(
            seal_tail[-1]).get("output_sha256") != man_sha:
        raise WP7Error("WP1_SEAL_LEDGER_MISMATCH", "")
    ev_sha = _sha256_file(paths.inner_val_json)
    if ev_sha != recorded.get("evaluator_report_sha256"):
        raise WP7Error("WP1_EVALUATOR_DRIFT", ev_sha[:16])
    ev = _load_json(paths.inner_val_json)
    if ev.get("status") != "ok" or not (ev.get("dev_sealed_status") or {}).get(
            "pass"):
        raise WP7Error("WP1_EVALUATOR_NOT_OK", str(ev.get("status")))
    return {"sealed_manifest_sha256": man_sha, "n_wp1_entries": n_wp1,
            "n_entries_total": len(lines), "pass": True}


# ---------------------------------------------------------------------------
# model loading (E0 strict) + inner-train train_q_mean
# ---------------------------------------------------------------------------


def strict_load_check(model: Any, state_dict: Mapping[str, Any]) -> Dict[
        str, Any]:
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as exc:
        raise WP7Error("STRICT_LOAD_FAILED", str(exc)[:400]) from exc
    return {"mode": "strict", "ok": True, "n_missing": 0, "n_unexpected": 0}


def load_e0_model(paths: Wp7Paths, seed: int, device: str) -> Tuple[Any, Dict[
        str, Any], Dict[str, Any]]:
    import torch

    from pes2ts_core.pesn.offline.training.v541_train import (
        V541TrainerConfig,
        load_model_v541,
    )

    run_dir = paths.e0_run_dir(seed)
    ckpt_path = run_dir / "checkpoint_best.pt"
    metrics = _load_json(run_dir / "metrics.json")
    ckpt_sha = _sha256_file(ckpt_path)
    if ckpt_sha != metrics.get("checkpoint_best_sha256"):
        raise WP7Error("CKPT_METRICS_SHA_MISMATCH", "seed %d" % seed)
    state = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    if not (isinstance(state, Mapping) and "model" in state):
        raise WP7Error("CKPT_FORMAT", str(ckpt_path))
    if str(state.get("schema")) != V541_CKPT_SCHEMA:
        raise WP7Error("CKPT_SCHEMA", str(state.get("schema")))
    ident = state.get("identity") or {}
    if str(ident.get("embedding_id")) != EMBEDDING_ID:
        raise WP7Error("FORBIDDEN_CANDIDATE",
                       "refusing non-E0 checkpoint: %r"
                       % ident.get("embedding_id"))
    if int(ident.get("seed", -1)) != int(seed):
        raise WP7Error("CKPT_IDENTITY_MISMATCH",
                       "identity.seed=%r want %d" % (ident.get("seed"), seed))
    stored = state.get("config") or {}
    names = {f.name for f in dc_fields(V541TrainerConfig)}
    kwargs = {k: v for k, v in stored.items() if k in names}
    kwargs["device"] = str(device)
    cfg = V541TrainerConfig(**kwargs)
    if cfg.embedding_id != EMBEDDING_ID:
        raise WP7Error("FORBIDDEN_CANDIDATE",
                       "config embedding_id=%r" % cfg.embedding_id)
    if int(cfg.seed) != int(seed):
        raise WP7Error("CKPT_IDENTITY_MISMATCH",
                       "config.seed=%r want %d" % (cfg.seed, seed))
    model = load_model_v541(cfg, r3_ckpt=paths.r3_init, device=device)
    if hasattr(model, "is_e0") and not bool(model.is_e0):
        raise WP7Error("FORBIDDEN_CANDIDATE", "loaded model is not E0")
    load_info = strict_load_check(model, state["model"])
    tau = (state.get("tau_end_calibration") or {}).get("tau_end")
    if tau is None:
        raise WP7Error("TAU_MISSING_IN_CHECKPOINT", "seed %d" % seed)
    meta = {
        "checkpoint_path": str(ckpt_path),
        "checkpoint_sha256": ckpt_sha,
        "strict_load": load_info,
        "identity": {"embedding_id": ident.get("embedding_id"),
                     "seed": ident.get("seed"),
                     "eval_scope": ident.get("eval_scope")},
        "tau_end": float(tau),
        "tau_block": dict(state.get("tau_end_calibration") or {}),
    }
    return model, state, meta


def seed_train_q_mean(dataset: Any, labels: Any, split_doc: Mapping[str,
                                                                   Any],
                      seed: int, inner_val_fraction: float) -> Tuple[
                          Optional[float], Dict[str, Any]]:
    """Constant-q baseline source: THIS seed's inner-train slice ONLY."""
    from pes2ts_core.pesn.offline.training.key_atom_v53mp_train_v3 import (
        _train_q_mean,
        v53mp_pool_split,
    )
    from scripts.campaigns.r3_v53mp_run import _tier_pool_curve_ids

    pool_curves = _tier_pool_curve_ids(split_doc, TIER)
    train_pos, val_pos = v53mp_pool_split(
        dataset.samples, pool_curve_ids=pool_curves, seed=int(seed),
        inner_val_fraction=float(inner_val_fraction))
    try:
        n_rows = int(len(train_pos))
        value = _train_q_mean(dataset.samples, labels, train_pos)
        n_q_masked = None
        q_mask_col = getattr(labels, "q_mask_col", None)
        if q_mask_col is not None and n_rows > 0:
            n_q_masked = int(np.asarray(q_mask_col)[
                np.asarray(train_pos, dtype=np.int64)].sum())
    finally:
        del train_pos, val_pos
    info = {
        "value": None if value is None else float(value),
        "source": ("inner-train rows of the v53mp pool split "
                   "(seed=%d); never recomputed from dev" % seed),
        "n_rows_used": n_rows,
        "n_q_masked": n_q_masked,
        "inner_val_fraction": float(inner_val_fraction),
        "derived_from_dev": False,
    }
    return value, info


# ---------------------------------------------------------------------------
# metric blocks (constant-q baseline mirror of evaluate_v53mp)
# ---------------------------------------------------------------------------


def constant_q_block(preds: Sequence[Any],
                     train_q_mean: Optional[float]) -> Dict[str, Any]:
    """``evaluate_v53mp`` q-head constant-baseline block (verbatim formula,
    key_atom_v53mp_train_v3.py:1660-1676 — ``learnable_above_constant``
    compares the MEDIAN curve MAE)."""
    qp = np.asarray([p.q_pred for p in preds], dtype=np.float64)
    qt = np.asarray([p.q_true for p in preds], dtype=np.float64)
    qm = np.asarray([p.q_mask for p in preds], dtype=bool)
    per_curve: Dict[str, List[int]] = {}
    for i, p in enumerate(preds):
        per_curve.setdefault(str(p.curve_id), []).append(i)
    q_curve_mae: List[float] = []
    for _cid, idxs in sorted(per_curve.items()):
        ok = [i for i in idxs if qm[i]]
        if not ok:
            continue
        q_curve_mae.append(float(np.abs(qp[ok] - qt[ok]).mean()))
    const_mae = (float(np.abs(float(train_q_mean) - qt[qm]).mean())
                 if (qm.any() and train_q_mean is not None) else None)
    return {
        "constant_q_baseline_mae": const_mae,
        "constant_q_value": (None if train_q_mean is None
                             else float(train_q_mean)),
        "learnable_above_constant": (
            None if (const_mae is None or not q_curve_mae)
            else float(np.median(q_curve_mae)) < const_mae),
        "source": "inner-train mean q (per-seed); evaluated on dev q rows",
    }


def eval_split_with_preds(
    split: str,
    positions: Sequence[int],
    *,
    corpus: Mapping[str, Any],
    cache: Any,
    device: str,
    model: Any,
    tau_end: Optional[float],
    cost: Any,
) -> Tuple[Dict[str, Any], Dict[int, Any]]:
    """ONE dev prediction pass + MOVE/PROCESS/denominators/sanity.

    Mirrors ``v541_baseline_repair._eval_split`` EXACTLY (same helpers,
    same order) but also RETURNS the raw predictions so the constant-q
    baseline block can be merged in (WP7 increment; WP1 module untouched).
    """
    preds, stats = br._predict_once(
        model, positions, cache, device,
        collect_process=True, labels=corpus.get("labels"),
        endpoint=corpus.get("endpoint"))
    cost.note_forward(stats["forward_wall_s"], stats["n_forward_batches"],
                      stats["n_positions"])
    ok_mask = _ok_mask(corpus["dataset"], positions)
    move_positions = [int(p) for p, m in zip(positions, ok_mask) if m]
    if not move_positions:
        raise WP7Error("MOVE_EMPTY", "%s split has no ok positions" % split)
    legacy_mv, repaired_mv, mv_meta = br._build_move_variants(
        preds, move_positions)
    move = br._move_metrics(legacy_mv, repaired_mv, mv_meta)
    ordered = [preds[int(p)] for p in positions]
    process = br._process_metrics(ordered, tau_end=tau_end)
    n_curves_all = len({str(p.curve_id) for p in preds.values()})
    n_curves_move = len({preds[int(p)].curve_id for p in move_positions})
    res = {
        "denominators": {
            "n_positions": len(positions),
            "n_positions_ok": len(move_positions),
            "n_curves": n_curves_all,
            "n_curves_move": n_curves_move,
            "positions_sha256": br._canonical_positions_sha(list(positions)),
            "curves_sha256": br._canonical_curves_sha(
                sorted({str(p.curve_id) for p in preds.values()})),
            "n_forward_batches": stats["n_forward_batches"],
        },
        "move": move,
        "process": process,
        "sanity_gates": br._move_sanity(
            move, n_positions=len(positions),
            n_positions_ok=len(move_positions),
            n_curves_move=n_curves_move),
    }
    return res, preds


# ---------------------------------------------------------------------------
# stage: run (the single candidate_evaluation dev access)
# ---------------------------------------------------------------------------


def _denominator_gates(got: Mapping[str, Any], want: Mapping[str, Any]) -> List[
        Dict[str, Any]]:
    return [
        br._compare("n_dev_positions_exact", got.get("n_positions"),
                    want.get("n_dev_positions"), 0, "exact"),
        br._compare("n_move_ok_exact", got.get("n_positions_ok"),
                    want.get("n_move_ok_positions"), 0, "exact"),
        br._compare("n_dev_curves_exact", got.get("n_curves"),
                    want.get("n_dev_curves"), 0, "exact"),
        br._compare("canonical_positions_sha_exact",
                    got.get("positions_sha256"),
                    want.get("canonical_positions_sha256"), 0, "exact"),
        br._compare("devpack_positions_sha_exact",
                    got.get("devpack_positions_sha256"),
                    want.get("devpack_positions_sha256"), 0, "exact"),
        br._compare("curves_sha_exact", got.get("curves_sha256"),
                    want.get("curves_sha256"), 0, "exact"),
    ]


def _sealed_reference_values(paths: Wp7Paths,
                             freeze: Mapping[str, Any]) -> Dict[str, Any]:
    """Gate values — readable audits + ledger SHAs from the freeze, then
    CROSS-CHECKED against the sealed R3 reference (allowed now that the
    seal is verified)."""
    ref = dict(freeze.get("dev_reference_binding") or {})
    sealed = _load_json(paths.sealed_dir / "r3_reference_dev.json")
    den = sealed.get("denominators") or {}
    ref["canonical_positions_sha256"] = den.get("positions_sha256")
    ref["n_curves_move"] = den.get("n_curves_move")
    checks = {
        "n_dev_positions": den.get("n_positions") == ref.get(
            "n_dev_positions"),
        "n_dev_curves": den.get("n_curves") == ref.get("n_dev_curves"),
        "curves_sha": den.get("curves_sha256") == ref.get("curves_sha256"),
        "sanity_pass": bool(sealed.get("sanity_pass")),
    }
    bad = [k for k, v in checks.items() if not v]
    if bad:
        raise WP7Error("SEALED_R3_CROSSCHECK_FAILED", ", ".join(bad))
    return ref


def eval_seed(paths: Wp7Paths, seed: int, *, corpus: Mapping[str, Any],
              cache: Any, split_doc: Mapping[str, Any],
              dev_positions: Sequence[int], dev_curves: Sequence[str],
              devpack_sha: str, canon_sha: str, curves_sha: str,
              ref_values: Mapping[str, Any], device: str, cost: Any) -> Dict[
                  str, Any]:
    _say("=== dev E0 seed %d: load + ONE pass ===" % seed)
    model, state, meta = load_e0_model(paths, seed, device)
    try:
        tau = meta["tau_end"]
        mpath = paths.e0_run_dir(seed) / "metrics.json"
        m_tau = (_load_json(mpath).get("tau_end_calibration") or {}).get(
            "tau_end")
        if m_tau is None or abs(float(m_tau) - float(tau)) > 0.0:
            raise WP7Error("TAU_CKPT_METRICS_MISMATCH",
                           "seed %d ckpt=%r metrics=%r" % (seed, tau, m_tau))
        frac = float((state.get("config") or {}).get(
            "inner_val_fraction", 0.2))
        train_q_mean, tqm_info = seed_train_q_mean(
            corpus["dataset"], corpus["labels"], split_doc, seed, frac)
        res, preds = eval_split_with_preds(
            "dev", dev_positions, corpus=corpus, cache=cache,
            device=device, model=model, tau_end=tau, cost=cost)
    finally:
        del model, state
        br._free_cuda()

    ordered = [preds[int(p)] for p in dev_positions]
    res["process"]["q_head"].update(constant_q_block(ordered, train_q_mean))
    del preds, ordered

    # denominators (+ dual position SHA recorded)
    res["denominators"]["devpack_positions_sha256"] = devpack_sha
    res["denominators"]["canonical_positions_sha256"] = canon_sha
    res["denominators"]["curves_sha256_canonical"] = curves_sha
    den_gates = _denominator_gates(
        {**res["denominators"], "positions_sha256": canon_sha}, ref_values)
    if not br._pass_all(den_gates):
        raise WP7Error("DEV_DENOMINATOR_MISMATCH",
                       str(br._gate_flags(den_gates)))
    if not br._pass_all(res["sanity_gates"]):
        raise WP7Error("MOVE_SANITY_FAILED",
                       str(br._gate_flags(res["sanity_gates"])))

    # H2 descriptive guard vs the WP1 sealed R3 reference (REUSED, not
    # re-run); E0 is the baseline — recorded as control, never a gate.
    r3_move = (_load_json(paths.sealed_dir / "r3_reference_dev.json")
               .get("move_before_after") or {})
    h2 = [br._h2_move_guard(res["move"][variant], r3_move[variant],
                            split="dev", variant=variant, seed=seed)
          for variant in ("after_repaired", "before_legacy")]

    seed_doc = {
        "schema": SCHEMA_SEED,
        "split": "dev",
        "seed": int(seed),
        "model": E0_MODEL_NAME[seed],
        "run_id": E0_RUN_ID[seed],
        "created_at": _now(),
        "checkpoint_sha256": meta["checkpoint_sha256"],
        "strict_load": meta["strict_load"],
        "identity": meta["identity"],
        "tau": {**meta["tau_block"], "tau_end": tau,
                "source": "checkpoint_best.pt#tau_end_calibration",
                "calibrated_on": "inner_val", "recomputed": False},
        "train_q_mean": tqm_info,
        "denominators": res["denominators"],
        "denominator_gates": den_gates,
        "sanity_gates": res["sanity_gates"],
        "move": res["move"],
        "process": res["process"],
        "h2_reference_guard_dev": {
            "reference": ("WP1 sealed r3_reference_dev.json "
                          "(reused, SHA-verified; R3 not re-run)"),
            "guards": h2,
            "note": ("descriptive control only — E0 is the baseline; "
                     "no candidate gating"),
        },
        "controls": {"confirm_read": False, "new_QC_calls": 0,
                     "selection_or_tuning_performed": False,
                     "tau_recomputed": False},
        "seed_pass": True,
    }
    return seed_doc


def light_metrics(seed_doc: Mapping[str, Any]) -> Dict[str, Any]:
    """Top-line per-seed numbers for the aggregate (descriptive)."""
    a = seed_doc["move"]["after_repaired"]
    b = seed_doc["move"]["before_legacy"]
    q = seed_doc["process"]["q_head"]
    o = seed_doc["process"]["ordering"]
    e = seed_doc["process"]["endpoint"]
    return {
        "endpoint_auprc": e.get("auprc"),
        "endpoint_auroc": e.get("auroc"),
        "endpoint_prevalence": e.get("prevalence"),
        "endpoint_n_unmasked": e.get("n_unmasked"),
        "endpoint_n_pos": e.get("n_pos"),
        "endpoint_n_neg": e.get("n_neg"),
        "start_misstop_rate_at_tau": e.get("start_misstop_rate_at_tau"),
        "ts_misstop_rate_at_tau": e.get("ts_misstop_rate_at_tau"),
        "tau_end": e.get("tau_end"),
        "q_corr_curve_mean": q.get("corr_curve_mean"),
        "q_mae_curve_mean": q.get("mae_curve_mean"),
        "q_n_eval_curves": q.get("n_eval_curves"),
        "constant_q_baseline_mae": q.get("constant_q_baseline_mae"),
        "learnable_above_constant": q.get("learnable_above_constant"),
        "ordering_accuracy_full": o.get("pairwise_accuracy_full"),
        "ordering_n_pairs_full": o.get("n_pairs_full"),
        "ordering_accuracy_cross_ts": o.get("pairwise_accuracy_cross_ts"),
        "median_rmse": a["key_rmse_over_zero"].get("median_rmse"),
        "median_rmse_over_zero":
            a["key_rmse_over_zero"].get("median_rmse_over_zero"),
        "cos_eff_median": a["key_rmse_over_zero"].get("cos_eff_median"),
        "key_atom_coverage":
            a["fixed_coverage_direction"].get("key_atom_coverage"),
        "dead_fraction": a["sre_and_sign"].get("dead_zone_fraction"),
        "sign_accuracy_cluster_median":
            a["sre_and_sign"].get("sign_accuracy_cluster_median"),
        "n_sign_pairs": a["sre_and_sign"].get("n_sign_pairs"),
        "n_active_pairs": a["sre_and_sign"].get("n_active_pairs"),
        "sign_accuracy_cluster_median_legacy":
            b["sre_and_sign"].get("sign_accuracy_cluster_median"),
        "n_pairs_collected_after":
            seed_doc["move"]["pair_denominators"].get(
                "n_pairs_collected_after"),
    }


def aggregate_three_seed(per_seed_light: Mapping[str, Mapping[str, Any]]) ->\
        Dict[str, Any]:
    keys = sorted({k for blk in per_seed_light.values() for k in blk})
    out: Dict[str, Any] = {}
    for k in keys:
        per = {s: (blk.get(k) if isinstance(blk.get(k), (int, float,
                                                        bool, str))
                   or blk.get(k) is None else None)
               for s, blk in sorted(per_seed_light.items())}
        nums = [float(v) for v in per.values()
                if isinstance(v, (int, float)) and not isinstance(v, bool)]
        out[k] = {
            "per_seed": per,
            "mean": float(np.mean(nums)) if nums else None,
            "min": float(np.min(nums)) if nums else None,
            "max": float(np.max(nums)) if nums else None,
            "n_seeds_defined": len(nums),
        }
    return {"unit": "seed (descriptive only)", "metrics": out}


# ---------------------------------------------------------------------------
# ledger (append-only; SAME byte format as WP1 lines)
# ---------------------------------------------------------------------------


def make_ledger_entry(*, model: str, output_path: str, output_sha256: str,
                      freeze_sha256: str, code_sha256: str, cost: Mapping[
                          str, Any], positions_sha256: str,
                      devpack_positions_sha256: str, curve_set_sha256: str,
                      n_positions: int, n_curves: int,
                      sealed_reference_manifest_sha256: Optional[str] = None
                      ) -> Dict[str, Any]:
    return {
        "schema": SCHEMA_LEDGER,
        "entry_id": "dev_%s_%d" % (model, int(time.time() * 1000)),
        "actor": _actor(),
        "purpose": "candidate_evaluation",
        "wp": "WP7",
        "campaign": "V5.4.1",
        "split": "dev",
        "model": model,
        "ts": _now(),
        "curve_set_sha256": curve_set_sha256,
        "positions_sha256": positions_sha256,
        "devpack_positions_sha256": devpack_positions_sha256,
        "n_positions": int(n_positions),
        "n_curves": int(n_curves),
        "code_sha256": code_sha256,
        "output_path": output_path,
        "output_sha256": output_sha256,
        "freeze_sha256": freeze_sha256,
        "sealed_reference_manifest_sha256": sealed_reference_manifest_sha256,
        "new_QC_calls": 0,
        "confirm_read": False,
        "cost": dict(cost),
        "status": "done",
    }


def append_ledger(paths: Wp7Paths, entry: Mapping[str, Any]) -> None:
    paths.audit_dir.mkdir(parents=True, exist_ok=True)
    with open(paths.ledger_path, "a", encoding="utf-8") as fh:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        try:
            fh.write(json.dumps(_jsonable(entry), sort_keys=True,
                                ensure_ascii=False) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        finally:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)


# ---------------------------------------------------------------------------
# failure record
# ---------------------------------------------------------------------------


def record_failure(paths: Wp7Paths, stage: str, exc: BaseException,
                   completed: Optional[Mapping[str, Any]] = None) -> Path:
    code = getattr(exc, "code", exc.__class__.__name__)
    doc = {
        "schema": SCHEMA_FAILURE,
        "created_at": _now(),
        "wp": "WP7",
        "stage": stage,
        "status": "failed",
        "failure": {"code": str(code),
                    "detail": str(getattr(exc, "detail", exc))[:800]},
        "controls": {"confirm_read": False, "new_QC_calls": 0,
                     "training_runs_started": 0},
        "per_seed_files": dict(completed or {}),
        "traceback": traceback.format_exc()[-2000:],
    }
    path = paths.eval_report_path
    try:
        _write_json(path, doc)
    except Exception:
        path = paths.audit_dir / ("wp7_failure_%d.json" % int(time.time()))
        _write_json(path, doc)
    return path


# ---------------------------------------------------------------------------
# cmd: freeze
# ---------------------------------------------------------------------------


def cmd_freeze(_args: argparse.Namespace) -> int:
    paths = Wp7Paths(PROJECT_ROOT)
    try:
        freeze = build_freeze(paths)
        if paths.freeze_path.is_file():
            old = _load_json(paths.freeze_path)
            verify_freeze(paths, old)
            _say("freeze already present and still valid: %s"
                 % paths.freeze_path)
            return 0
        sha = _write_json(paths.freeze_path, freeze)
        verify_freeze(paths, _load_json(paths.freeze_path))
        _say("freeze written: %s (sha256=%s)" % (paths.freeze_path, sha[:16]))
        return 0
    except Exception as exc:
        out = record_failure(paths, "freeze", exc)
        _say("FREEZE FAILED (%s) -> %s" % (
            getattr(exc, "code", exc), out))
        return 1


# ---------------------------------------------------------------------------
# cmd: run
# ---------------------------------------------------------------------------


def cmd_run(args: argparse.Namespace) -> int:
    paths = Wp7Paths(PROJECT_ROOT)
    fixture_mode = os.environ.get("PES2TS_WP7_FIXTURE") == "1"
    if args.limit_positions is not None and not fixture_mode:
        err = WP7Error("FIXTURE_FLAG_REQUIRED",
                       "--limit-positions needs PES2TS_WP7_FIXTURE=1")
        out = record_failure(paths, "run", err)
        _say("REFUSED: %s -> %s" % (err, out))
        return 1
    seeds = tuple(int(s) for s in args.seeds.split(",") if s.strip())
    if seeds != SEEDS:
        err = WP7Error("SEED_SET_MISMATCH",
                       "frozen seeds are %s" % (SEEDS,))
        out = record_failure(paths, "run", err)
        _say("REFUSED: %s -> %s" % (err, out))
        return 1
    device = str(args.device)
    completed: Dict[str, Any] = {}
    stage = "preflight"
    cost = br._Cost()
    try:
        if not paths.freeze_path.is_file():
            raise WP7Error("FREEZE_MISSING",
                           "run `freeze` before `run` (%s)"
                           % paths.freeze_path)
        freeze = _load_json(paths.freeze_path)
        verify_freeze(paths, freeze)
        stage = "wp1_seal"
        seal = verify_wp1_seal(paths, freeze)
        _say("WP1 seal verified: %d WP1 ledger entries intact"
             % seal["n_wp1_entries"])
        stage = "screen_gate"
        screen_adjudication_gate(_load_json(paths.screen_path))
        stage = "corpus"
        ref_values = _sealed_reference_values(paths, freeze)
        from scripts.campaigns.r3_v53mp_run import _subset_curve_ids

        corpus_pack = _build_tier(
            TIER, paths.tables_dir, paths.split_path, device,
            int(TIER_DEFAULTS[TIER]["cache_curves"]), "auto")
        pack_info = corpus_pack[5]
        if not pack_info.get("used"):
            raise WP7Error("DEV_PACK_UNAVAILABLE", str(pack_info))
        dataset, labels, endpoint, dev_pos, cache = (
            corpus_pack[0], corpus_pack[1], corpus_pack[2], corpus_pack[3],
            corpus_pack[4])
        split_doc = _load_json(paths.split_path)
        dev_curves = sorted(_subset_curve_ids(split_doc, "dev"))
        from pes2ts_core.pesn.offline.data.key_atom_dev_pack_v53mp import (
            positions_sha256 as devpack_positions_sha256,
        )

        dev_positions = [int(p) for p in dev_pos]
        if args.limit_positions is not None:
            dev_positions = dev_positions[:int(args.limit_positions)]
            _say("FIXTURE MODE: truncated to %d positions"
                 % len(dev_positions))
        devpack_sha = str(devpack_positions_sha256(list(dev_positions)))
        canon_sha = br._canonical_positions_sha(dev_positions)
        curves_sha = br._canonical_curves_sha(dev_curves)
        ok_mask = _ok_mask(dataset, dev_positions)
        n_ok = int(np.sum(ok_mask))
        _say("dev: positions=%d ok=%d curves=%d pack=%s" % (
            len(dev_positions), n_ok, len(dev_curves), bool(pack_info.get(
                "used"))))
        corpus = {"dataset": dataset, "labels": labels, "endpoint": endpoint}
        freeze_sha = _sha256_file(paths.freeze_path)
        code_sha = freeze["code_bindings"]["evaluator_code_sha256"]
        ledger_common = {
            "curve_set_sha256": curves_sha,
            "positions_sha256": canon_sha,
            "devpack_positions_sha256": devpack_sha,
            "n_positions": len(dev_positions),
            "n_curves": len(dev_curves),
            "code_sha256": code_sha,
            "freeze_sha256": freeze_sha,
            "sealed_reference_manifest_sha256":
                freeze["wp1_seal"]["sealed_manifest_sha256"],
        }

        stage = "gpu_lock"
        lock = br._GpuLock()
        lock.acquire()
        try:
            paths.per_seed_dir.mkdir(parents=True, exist_ok=True)
            per_seed_light: Dict[str, Dict[str, Any]] = {}
            for seed in seeds:
                stage = "seed_%d" % seed
                seed_doc = eval_seed(
                    paths, seed, corpus=corpus, cache=cache,
                    split_doc=split_doc, dev_positions=dev_positions,
                    dev_curves=dev_curves, devpack_sha=devpack_sha,
                    canon_sha=canon_sha, curves_sha=curves_sha,
                    ref_values=ref_values, device=device, cost=cost)
                if fixture_mode:
                    seed_doc["fixture_mode"] = True
                out_path = paths.per_seed_file(seed)
                out_sha = _write_json(out_path, seed_doc)
                entry = make_ledger_entry(
                    model=E0_MODEL_NAME[seed],
                    output_path=str(out_path.relative_to(
                        paths.project_root)),
                    output_sha256=out_sha, cost=cost.snapshot(device),
                    **ledger_common)
                append_ledger(paths, entry)
                completed[str(seed)] = {
                    "path": str(out_path.relative_to(paths.project_root)),
                    "sha256": out_sha,
                    "ledger_entry_id": entry["entry_id"],
                }
                per_seed_light[str(seed)] = light_metrics(seed_doc)
                _say("seed %d done: denom ok | ledger %s" % (
                    seed, entry["entry_id"]))

            stage = "aggregate"
            agg = aggregate_three_seed(per_seed_light)
            denom_summary = {
                "n_dev_positions": len(dev_positions),
                "n_move_ok": n_ok,
                "n_dev_curves": len(dev_curves),
                "canonical_positions_sha256": canon_sha,
                "devpack_positions_sha256": devpack_sha,
                "curves_sha256": curves_sha,
                "n_invalid": 0,
            }
            report = {
                "schema": SCHEMA_EVAL,
                "created_at": _now(),
                "wp": "WP7",
                "dev_access": {"type": "candidate_evaluation",
                               "split": "dev",
                               "n_evaluations": len(seeds)},
                "freeze": {"path": str(paths.freeze_path.relative_to(
                    paths.project_root)), "sha256": freeze_sha,
                    "verified": True},
                "wp1_seal_verified": True,
                "per_seed_files": completed,
                "aggregate": agg,
                "denominator_summary": denom_summary,
                "adjudication_context": {
                    "e_star": EMBEDDING_ID,
                    "candidates_evaluated_on_dev": 0,
                    "forbidden_candidates": ["E1", "E2", "E3", "E4", "E5"],
                    "dev_veto_applicable": False,
                    "reason": ("E*=E0 with zero surviving candidates; "
                               "plan §10.1.4 veto path has no upgrade "
                               "to veto"),
                    "selection_or_tuning_performed": False,
                    "tau_recomputed": False,
                    "runner_up_considered": False,
                },
                "controls": {"confirm_read": False, "new_QC_calls": 0,
                             "qc_calls_breakdown": {},
                             "training_runs_started": 0},
                "status": "ok",
                "cost": cost.snapshot(device),
                "fixture_mode": fixture_mode,
                "ledger_entries_this_run": [
                    completed[s]["ledger_entry_id"] for s in sorted(completed)],
            }
            if fixture_mode:
                report["fixture_mode"] = True
            report_sha = _write_json(paths.eval_report_path, report)
            entry = make_ledger_entry(
                model="aggregate",
                output_path=str(paths.eval_report_path.relative_to(
                    paths.project_root)),
                output_sha256=report_sha, cost=cost.snapshot(device),
                **ledger_common)
            append_ledger(paths, entry)
            _say("dev frozen evaluation written: %s (sha256=%s)" % (
                paths.eval_report_path, report_sha[:16]))
            return 0
        finally:
            lock.release()
    except Exception as exc:
        out = record_failure(paths, stage, exc, completed)
        _say("RUN FAILED at stage=%s (%s) -> %s" % (
            stage, getattr(exc, "code", exc), out))
        return 1


# ---------------------------------------------------------------------------
# cmd: verify (read-only; no dev access)
# ---------------------------------------------------------------------------


def cmd_verify(_args: argparse.Namespace) -> int:
    paths = Wp7Paths(PROJECT_ROOT)
    ok = True

    def check(name: str, cond: bool, detail: str = "") -> None:
        nonlocal ok
        ok = ok and bool(cond)
        _say("verify %-34s %s%s" % (name, "PASS" if cond else "FAIL",
                                    (" | " + detail) if detail else ""))

    try:
        freeze = _load_json(paths.freeze_path)
        verify_freeze(paths, freeze)
        check("freeze_bindings", True)
        seal = verify_wp1_seal(paths, freeze)
        check("wp1_seal", True, "%d WP1 entries" % seal["n_wp1_entries"])
        report = _load_json(paths.eval_report_path)
        check("eval_report_status_ok", report.get("status") == "ok",
              str(report.get("status")))
        for seed, blk in sorted((report.get("per_seed_files") or {}).items()):
            p = paths.project_root / blk["path"]
            check("per_seed_%s_sha" % seed,
                  p.is_file() and _sha256_file(p) == blk["sha256"],
                  blk["path"])
        lines = _read_ledger_lines(paths)
        n_wp1 = seal["n_wp1_entries"]
        new = [json.loads(ln) for ln in lines[n_wp1:]]
        check("ledger_new_purposes",
              all(r.get("purpose") == "candidate_evaluation" for r in new),
              "%d new entries" % len(new))
        # supersession-aware output check: a superseded entry (an earlier
        # attempt for the same model) is valid iff the CURRENT file at its
        # recorded path matches SOME (later) entry for that model.
        shas_by_model: Dict[str, set] = {}
        for rec in new:
            shas_by_model.setdefault(str(rec.get("model")), set()).add(
                str(rec.get("output_sha256")))
        for rec in new:
            p = paths.project_root / str(rec.get("output_path"))
            got = _sha256_file(p) if p.is_file() else None
            superseded_ok = (got is not None
                             and got in shas_by_model.get(
                                 str(rec.get("model")), set()))
            check("ledger_output_%s_%s" % (rec.get("model"),
                                           str(rec.get("entry_id"))[-8:]),
                  got == rec.get("output_sha256") or superseded_ok,
                  str(rec.get("output_path")))
        ctx = report.get("adjudication_context") or {}
        check("no_selection_or_tuning",
              ctx.get("selection_or_tuning_performed") is False
              and ctx.get("dev_veto_applicable") is False
              and ctx.get("candidates_evaluated_on_dev") == 0)
        ctrl = report.get("controls") or {}
        check("controls_clean", ctrl.get("confirm_read") is False
              and ctrl.get("new_QC_calls") == 0
              and ctrl.get("training_runs_started") == 0)
        _say("verify %s" % ("PASS" if ok else "FAIL"))
        return 0 if ok else 1
    except Exception as exc:
        _say("VERIFY FAILED (%s): %s" % (
            getattr(exc, "code", exc), getattr(exc, "detail", exc)))
        return 1


# ---------------------------------------------------------------------------
# cli
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="v541_dev_evaluate.py",
        description="V5.4.1 WP7 frozen dev evaluation (matched E0 x3 only)")
    sub = p.add_subparsers(dest="stage", required=True)
    sub.add_parser("freeze", help="write/verify reports/wp7_freeze.json")
    r = sub.add_parser("run", help="the single candidate_evaluation access")
    r.add_argument("--seeds", default="17,29,43")
    r.add_argument("--device", default="cuda")
    r.add_argument("--limit-positions", type=int, default=None,
                   help="FIXTURE ONLY (needs PES2TS_WP7_FIXTURE=1)")
    sub.add_parser("verify", help="read-only integrity re-check")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.stage == "freeze":
        return cmd_freeze(args)
    if args.stage == "run":
        return cmd_run(args)
    if args.stage == "verify":
        return cmd_verify(args)
    return 2


if __name__ == "__main__":
    sys.exit(main())
