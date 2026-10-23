"""V5.3-MP run driver (campaign plan §5 / §8.1; MoveProcessContract v1).

One planned V5.3-MP run per invocation:

1. campaign-root assertion (the driver writes ONLY under
   ``outputs/pesn/v53mp/**``; sealed prefixes refuse fail-closed);
2. run_id parse + plan lookup (exact run_id; per-tier plan
   ``run_plan_v53mp_<tier>.jsonl``) + WP-L3-style plan-pin verification
   (pin sidecar written by ``plan-init``; ANY byte change after plan freeze
   refuses);
3. **authorization gate**: the plan line's stage (S1/S2/S3) must carry its
   ``*_authorized: true`` bit in the campaign yaml — otherwise a refusal
   JSON is recorded under ``<root>/refusals/`` and the driver exits 1
   BEFORE touching the ledger or the GPU lock;
4. λ calibration gate: arms B/C refuse while the gradient-share probe
   values are still ``null`` in the config (fail-closed);
5. five-sha binding re-verification (config / data / teacher / features /
   split artifacts on disk must still hash to the frozen plan-line shas);
6. stage binding (``assert_stage_binding``: every planned line carries real
   64-hex identity shas) + ledger reconcile + plan-count attempt caps;
7. S1 GPU-seconds budget (≤3600 s cumulative; refuse BEFORE launch);
8. GPU flock with metadata, then ``monitor.attempt_launch`` (never a raw
   ledger launch);
9. train via ``train_run_v53mp`` (Pinned Lane-C API) on the sharded loader
   with FULL label statuses; τ/q/endpoint λ from the calibrated probe;
10. ``done`` transition with ``checkpoint_sha256`` + ``metrics_path`` +
    ``gpu_seconds`` under ``<root>/<run_id>/``.

Fixed backbone init (plan §3, first round frozen architecture): the R3
F2-bidir winner fold0×seed17 checkpoint ``5372d9b12807c3f0`` — ALL V5.3-MP
runs start from this single frozen init (data scale is the only primary
variable).

DEVIATION (documented): the ledger is intentionally NOT bound to the plan
(``plan_path=None``) because V5.3-MP run_ids use the human-readable
``v53mp_{tier}_{arm}_s{seed}`` scheme, while ``RunLedger`` recomputes run_id
via the legacy/hashed schemes and would refuse the launch with
``RUN_ID_MISMATCH``.  Plan discipline is instead enforced IN this driver:
plan membership, plan-pin byte verification, five-sha re-verification,
``assert_stage_binding``, and per-phase attempt caps mirrored from the plan
counts (S1=3, S2=9, S3=9).  Ledger records carry the plan run_id in
``plan_run_id`` for traceability.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pes2ts_core.utils.hashing import sha256_file  # noqa: E402
from scripts.agent_ops.gpu_queue.gpu_lock import acquire as gpu_acquire  # noqa: E402
from scripts.agent_ops.ledger.run_ledger import (  # noqa: E402
    LedgerError,
    RunLedger,
    compute_run_id,
    read_run_plan,
    verify_plan_pin,
)
from scripts.agent_ops.runplan.build_r3_runplan import (  # noqa: E402
    SHA_FIELDS,
    classify_sha,
)
from scripts.agent_ops.supervisor.campaign_root import (  # noqa: E402
    CampaignRootError,
    SEALED_PREFIXES,
    normalize_root,
)
from scripts.agent_ops.supervisor.monitor import attempt_launch  # noqa: E402
from scripts.agent_ops.supervisor.stage_bind import (  # noqa: E402
    DEFAULT_TASK_STATE_PATH,
    assert_stage_binding,
)

DEFAULT_V53MP_ROOT = "outputs/pesn/v53mp"
V53MP_ROOT_REL = "v53mp"          # rel to outputs/pesn/
DEFAULT_CONFIG = "config/campaigns/pes2ts_v53mp_20261021.yaml"
DEFAULT_TABLES_DIR = DEFAULT_V53MP_ROOT + "/data"
DEFAULT_SPLIT_BASENAME = "split_manifest_v53mp.json"
GPU_LOCK = "/tmp/opencode/pes2ts_gpu.lock"

CAMPAIGN_ID = "PES2TS_V53MP_MOVE_PROCESS"
SCHEMA_ID = "pes2ts_v53mp_campaign_v1"
#: R3 F2-bidir winner (fold0 x seed17) — the single frozen V5.3-MP init.
R3_WINNER_RUN_ID = "5372d9b12807c3f0"
R3_WINNER_CKPT = (
    "outputs/pesn/30h_campaign/r3_runs/{}/checkpoint_best.pt".format(R3_WINNER_RUN_ID)
)
#: Frozen input-feature builder (contract: no feature change S1–S3); its
#: file sha is bound into the plan as features_sha256.
FEATURES_MODULE_PATH = "pes2ts_core/pesn/features/key_atom_graph_v3.py"

TIER_PHASE = {"2k": "S1", "5k": "S2", "20k": "S3"}
TIER_LIMITS = {"2k": 2000, "5k": 5000, "20k": 20000}
#: Geometry LRU bound passed to ``ShardedKeyAtomDatasetV3``: covers the whole
#: tier pool plus the frozen dev margin, so a curve's shard geometry is
#: decompressed at most once per run (measured: ~0.13 s/update of redundant
#: npz reads at 5k with the old 512-curve default).
GEOMETRY_CACHE_CURVES = {"2k": 3500, "5k": 6500, "20k": 26000}
ARMS = ("A", "B", "C")
SEEDS_S1 = (17,)
SEEDS_FORMAL = (17, 29, 43)
#: per-phase launch-attempt caps == plan counts (all attempts incl. retries)
#: S1 = 4: 3 planned arms + 1 consumed by the 2026-10-22 λ-wiring bug
#: (LAMBDA_ARM_MISMATCH, fixed in the driver the same day; the failure was a
#: real recorded attempt, so the cap is bumped by exactly that count — never
#: silently raised).
#: S2 = 10: 9 planned runs + 1 consumed by the 2026-10-21T23:35:39 pre-plan
#: ``interrupted`` A_s17 attempt (recorded before the S2 plan freeze); the
#: failure was real and recorded, so the cap is bumped by exactly that count.
#: S3 = 14: 9 planned runs + 4 consumed by the two 2026-10-22 host-OOM
#:      restarts (see S3-old-13 below) + 1 consumed by the 2026-10-23
#:      external termination of B_s43 mid-run (~update 800; a real recorded
#:      attempt, reconciled to ``interrupted`` and retried by the ledger —
#:      the cap is bumped by exactly that count, never silently).
#:      Concurrency stays 1 for the rest of S3.
#: S3-old-13 = 13: 9 planned runs + 2 consumed by the first 2026-10-22 WSL OOM
#:      restart + 2 consumed by the second one (the 11:16 two-way attempt):
#:      the GUEST cgroup never OOM-ed (memory.events oom_kill=0, anon
#:      37.0 GiB when the VM vanished) — the HOST ran out, which a guest
#:      cgroup cannot prevent.
#:      Concurrency is therefore 1 for the rest of S3.
#: S3-old = 11: 9 planned runs + 2 consumed by the 2026-10-22 WSL OOM restart
#: (host memory exhaustion killed the A_s17/A_s29 driver processes mid-run;
#: both were real recorded attempts, reconciled to ``interrupted`` and
#: retried, so the cap is bumped by exactly that count — never silently).
PHASE_CAPS = {"S1": 4, "S2": 10, "S3": 14}
STAGE_AUTH_BIT = {"S1": "s1_authorized", "S2": "s2_authorized", "S3": "s3_authorized"}
STAGE_GATE_ID = {
    "S1": "V53MP-S1-smoke",
    "S2": "V53MP-S2-scale",
    "S3": "V53MP-S3-formal",
}
#: λ required per arm before launch (probe values; null => refuse)
ARM_REQUIRED_LAMBDAS = {
    "A": (),
    "B": ("lambda_rank", "lambda_q"),
    "C": ("lambda_rank", "lambda_q", "lambda_endpoint"),
}
DEFAULT_S1_GPU_SECONDS_CAP = 3600
DEFAULT_S1_GPU_SECONDS_PER_RUN = 1200


class V53MPConfigError(LedgerError):
    """Campaign config missing / malformed / drifted from the frozen plan."""


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def _assert_v53mp_root(path: Any) -> str:
    """Campaign-root assertion for the v53mp lane (fail-closed).

    Mirrors ``campaign_root.assert_campaign_root`` discipline but for the
    V5.3-MP root: garbage / non-``outputs/pesn`` paths and the sealed
    ``v5_1`` / ``v5_1_r1`` prefixes are refused; the path must normalize to
    ``outputs/pesn/v53mp`` or anything below it.
    """
    normalized = normalize_root(path)
    rel = normalized[len("outputs/pesn/"):]
    for sealed in SEALED_PREFIXES:
        sealed_rel = sealed[len("outputs/pesn/"):]
        if rel == sealed_rel or rel.startswith(sealed_rel + "/"):
            raise CampaignRootError(
                "SEALED_ROOT_REFUSED: {} is a sealed read-only prefix ({})".format(
                    normalized, sealed
                )
            )
    if rel != V53MP_ROOT_REL and not rel.startswith(V53MP_ROOT_REL + "/"):
        raise CampaignRootError(
            "OUTSIDE_CAMPAIGN_ROOT: {} is outside the V5.3-MP campaign root "
            "{}/...".format(normalized, DEFAULT_V53MP_ROOT)
        )
    return normalized


def _parse_run_id(run_id: str) -> Tuple[str, str, int]:
    """``v53mp_{tier}_{arm}_s{seed}`` -> (tier, arm, seed); ValueError on drift."""
    parts = str(run_id).split("_")
    if len(parts) != 4 or parts[0] != "v53mp":
        raise ValueError(
            "run_id must look like v53mp_<2k|5k|20k>_<A|B|C>_s<seed>, got {!r}".format(
                run_id
            )
        )
    tier, arm, seed_token = parts[1], parts[2], parts[3]
    if tier not in TIER_PHASE:
        raise ValueError("unknown tier {!r}".format(tier))
    if arm not in ARMS:
        raise ValueError("unknown arm {!r}".format(arm))
    if not seed_token.startswith("s") or not seed_token[1:].isdigit():
        raise ValueError("bad seed token {!r}".format(seed_token))
    return tier, arm, int(seed_token[1:])


def _load_campaign_config(config_path: Any) -> Dict[str, Any]:
    """Load + schema-check the campaign yaml (fail-closed)."""
    import yaml  # PyYAML is an existing repo dependency

    path = Path(config_path)
    if not path.is_file():
        raise V53MPConfigError("CONFIG_MISSING: {}".format(path))
    try:
        doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception as exc:  # yaml.YAMLError and friends
        raise V53MPConfigError("CONFIG_UNREADABLE: {}: {}".format(path, exc)) from exc
    if not isinstance(doc, dict):
        raise V53MPConfigError("CONFIG_CORRUPT: {} is not a mapping".format(path))
    if doc.get("schema") != SCHEMA_ID:
        raise V53MPConfigError(
            "CONFIG_SCHEMA: expected schema {!r}, got {!r}".format(
                SCHEMA_ID, doc.get("schema")
            )
        )
    if doc.get("campaign_id") != CAMPAIGN_ID:
        raise V53MPConfigError(
            "CONFIG_CAMPAIGN: expected campaign_id {!r}, got {!r}".format(
                CAMPAIGN_ID, doc.get("campaign_id")
            )
        )
    return doc


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _record_refusal(out_root: Path, run_id: str, payload: Dict[str, Any]) -> None:
    """Persist a refusal JSON under ``<root>/refusals/`` (auditable, fail-closed)."""
    target = out_root / "refusals"
    target.mkdir(parents=True, exist_ok=True)
    (target / "{}.json".format(run_id)).write_text(
        json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _git_code_commit() -> str:
    """Current repo commit (fail-closed: must resolve to a 40-hex sha1)."""
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise V53MPConfigError("GIT_COMMIT_UNAVAILABLE: {}".format(exc)) from exc
    commit = proc.stdout.strip()
    if len(commit) != 40 or any(c not in "0123456789abcdef" for c in commit):
        raise V53MPConfigError("GIT_COMMIT_INVALID: {!r}".format(commit))
    return commit


def _v53mp_record(
    line: Mapping[str, Any], *, attempt: int = 1, parent: Optional[str] = None
) -> Dict[str, Any]:
    """Ledger record for one planned V5.3-MP run (all five shas from the plan)."""
    rec: Dict[str, Any] = {
        "campaign_id": line["campaign_id"],
        "phase": line["phase"],
        "arm": line["arm"],
        "fold": int(line["fold"]),
        "seed": int(line["seed"]),
        "variant": line.get("variant"),
        "n_updates": int(line["n_updates"]),
        "data_sha256": line["data_sha256"],
        "teacher_sha256": line["teacher_sha256"],
        "features_sha256": line["features_sha256"],
        "split_sha256": line["split_sha256"],
        "config_sha256": line["config_sha256"],
        "checkpoint_update": None,
        "checkpoint_sha256": None,
        "status": "planned",
        "metrics_path": None,
        "gate_id": STAGE_GATE_ID.get(line["phase"], "V53MP"),
        "admissibility": "not_admissible_until_gate_pass",
        "evidence_sha": None,
        "attempt": int(attempt),
        "parent_run_id": parent,
        "confirm": bool(line.get("confirm", False)),
        # v53mp traceability (non-identity, non-content fields)
        "plan_run_id": line["run_id"],
        "tier": line.get("tier"),
    }
    return rec


def _resolve_plan_path(args: argparse.Namespace, tier: str) -> Path:
    if args.plan:
        return Path(args.plan)
    return Path(args.output_root) / "run_plan_v53mp_{}.jsonl".format(tier)


def _resolve_ledger_path(args: argparse.Namespace) -> Path:
    if args.ledger:
        return Path(args.ledger)
    return Path(args.output_root) / "run_records.jsonl"


def _verify_plan_pin_sidecar(plan_path: Path) -> Dict[str, Any]:
    """WP-L3-style pin: the plan bytes must match the ``plan-init`` freeze."""
    pin_path = Path(str(plan_path) + ".pin.json")
    if not pin_path.is_file():
        raise LedgerError(
            "PLAN_PIN_MISSING: {} not found (re-freeze via plan-init --force; "
            "an un-pinned plan is never launchable)".format(pin_path)
        )
    try:
        pin = json.loads(pin_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LedgerError("PLAN_PIN_CORRUPT: {}: {}".format(pin_path, exc)) from exc
    return verify_plan_pin(plan_path, pin)


def _verify_five_sha_binding(
    line: Mapping[str, Any], config_path: Path, tables_dir: Path, split_path: Path
) -> None:
    """On-disk artifacts must still hash to the frozen plan-line shas."""
    actual = {
        "config_sha256": sha256_file(config_path),
        "data_sha256": sha256_file(tables_dir / "data_manifest.json"),
        "teacher_sha256": sha256_file(tables_dir / "teacher_manifest.json"),
        "features_sha256": sha256_file(PROJECT_ROOT / FEATURES_MODULE_PATH),
        "split_sha256": sha256_file(split_path),
    }
    for field, digest in actual.items():
        planned = line.get(field)
        if digest != planned:
            raise LedgerError(
                "SHA_BINDING_DRIFT: {} on disk is {} but the frozen plan line "
                "carries {} -- rebuild the plan (plan-init --force) or restore "
                "the artifact".format(field, digest[:12], str(planned)[:12])
            )


def _tier_pool_curve_ids(split_doc: Mapping[str, Any], tier: str) -> List[str]:
    """Pool curve_ids for a nested tier (2k ⊂ 5k ⊂ 20k by rank)."""
    subsets = split_doc.get("subsets")
    if not isinstance(subsets, dict) or "pool" not in subsets:
        raise LedgerError("SPLIT_SCHEMA: split manifest has no subsets.pool")
    pool = subsets["pool"]
    if isinstance(pool, dict):
        if tier in pool:
            ids = pool[tier]
        elif isinstance(pool.get("tiers"), dict) and tier in pool["tiers"]:
            ids = pool["tiers"][tier]
        else:
            raise LedgerError(
                "SPLIT_SCHEMA: subsets.pool carries no tier {!r}".format(tier)
            )
        if not isinstance(ids, (list, tuple)):
            raise LedgerError("SPLIT_SCHEMA: pool tier {!r} is not a list".format(tier))
        return [str(x) for x in ids]
    if isinstance(pool, list):
        limit = TIER_LIMITS[tier]
        ids = []
        for entry in pool:
            if isinstance(entry, str):
                ids.append(entry)
            elif isinstance(entry, Mapping) and "curve_id" in entry:
                if int(entry.get("rank", 0)) <= limit:
                    ids.append(str(entry["curve_id"]))
            else:
                raise LedgerError(
                    "SPLIT_SCHEMA: pool entry {!r} is not a curve_id/record".format(
                        entry
                    )
                )
        if all(isinstance(entry, str) for entry in pool):
            ids = ids[:limit]  # flat ordered list: prefix = nested tier
        return ids
    raise LedgerError("SPLIT_SCHEMA: subsets.pool must be a list or tier mapping")


def _subset_curve_ids(split_doc: Mapping[str, Any], subset: str) -> List[str]:
    """curve_ids of one frozen split subset (``dev`` / ``confirm``)."""
    subsets = split_doc.get("subsets")
    if not isinstance(subsets, dict) or subset not in subsets:
        raise LedgerError(
            "SPLIT_SCHEMA: split manifest has no subsets.{}".format(subset)
        )
    entries = subsets[subset]
    if not isinstance(entries, (list, tuple)):
        raise LedgerError(
            "SPLIT_SCHEMA: subsets.{} is not a list".format(subset)
        )
    ids: List[str] = []
    for entry in entries:
        if isinstance(entry, str):
            ids.append(entry)
        elif isinstance(entry, Mapping) and "curve_id" in entry:
            ids.append(str(entry["curve_id"]))
        else:
            raise LedgerError(
                "SPLIT_SCHEMA: subsets.{} entry {!r} is not a curve_id/record"
                .format(subset, entry)
            )
    return ids


def _label_spec_sha256(config: Mapping[str, Any]) -> str:
    """label_spec_sha256 := sha256 of the frozen endpoint-label spec section.

    ``endpoint_label_spec_v1`` is frozen IN the campaign config
    (``data_contract.endpoint_labels``); hashing that section binds the label
    spec into provenance and into the exported bundle.
    """
    from pes2ts_core.utils.hashing import stable_json_dumps

    section = dict((config.get("data_contract") or {}).get("endpoint_labels") or {})
    if not section:
        raise V53MPConfigError(
            "LABEL_SPEC_MISSING: data_contract.endpoint_labels is empty"
        )
    import hashlib

    return hashlib.sha256(stable_json_dumps(section).encode("utf-8")).hexdigest()


def _build_training_provenance(
    config: Mapping[str, Any],
    config_path: Path,
    tables_dir: Path,
    split_path: Path,
    *,
    run_id: str,
    tier: str,
) -> Dict[str, str]:
    """Provenance superset: trainer keys (campaign_run_id/tier/
    split_manifest_sha256/endpoint_label_spec_sha256) + contract/export keys
    (config/split/label_spec/code_commit). Both spellings carry the same
    digests."""
    data_manifest = tables_dir / "data_manifest.json"
    teacher_manifest = tables_dir / "teacher_manifest.json"
    for path in (data_manifest, teacher_manifest, split_path):
        if not path.is_file():
            raise V53MPConfigError("PROVENANCE_ARTIFACT_MISSING: {}".format(path))
    split_sha = sha256_file(split_path)
    label_spec_sha = _label_spec_sha256(config)
    return {
        "campaign_run_id": run_id,
        "tier": tier,
        "config_sha256": sha256_file(config_path),
        "code_commit": _git_code_commit(),
        "data_manifest_sha256": sha256_file(data_manifest),
        "teacher_manifest_sha256": sha256_file(teacher_manifest),
        # trainer-spelled keys
        "split_manifest_sha256": split_sha,
        "endpoint_label_spec_sha256": label_spec_sha,
        # contract/export-spelled keys (same digests)
        "split_sha256": split_sha,
        "label_spec_sha256": label_spec_sha,
    }


# --------------------------------------------------------------------------
# plan-init
# --------------------------------------------------------------------------


def _tier_joint_updates(schedule: Mapping[str, Any], tier: str) -> int:
    """Density-preserving joint updates: base 2k value scaled by tier size
    (campaign yaml ``joint_updates_scaling: density_preserving_per_tier`` —
    constant per-sample coverage across 2k/5k/20k so data scale is the only
    primary variable)."""
    base = int(schedule.get("joint_updates_base_2k", 1500))
    return base * max(1, TIER_LIMITS[tier] // TIER_LIMITS["2k"])


def cmd_plan_init(args: argparse.Namespace) -> int:
    tier = args.tier
    phase = TIER_PHASE[tier]
    _assert_v53mp_root(args.output_root)
    out_root = Path(args.output_root)
    plan_path = _resolve_plan_path(args, tier)
    if plan_path.exists() and not args.force:
        sys.stderr.write(
            "error: PLAN_EXISTS: {} (plan rebuild requires --force + "
            "re-signoff)\n".format(plan_path)
        )
        return 1

    config = _load_campaign_config(args.config)
    schedule = dict((config.get("training_contract") or {}).get("schedule") or {})
    freeze_updates = int(schedule.get("freeze_encoder_updates", 400))
    joint_updates = _tier_joint_updates(schedule, tier)
    smoke_updates = int(
        (config.get("training_contract") or {}).get("smoke_updates") or 0
    ) or None

    tables_dir = Path(args.tables_dir)
    split_path = Path(args.split) if args.split else tables_dir / DEFAULT_SPLIT_BASENAME
    artifacts = {
        "config_sha256": Path(args.config),
        "data_sha256": tables_dir / "data_manifest.json",
        "teacher_sha256": tables_dir / "teacher_manifest.json",
        "features_sha256": PROJECT_ROOT / FEATURES_MODULE_PATH,
        "split_sha256": split_path,
    }
    missing = [str(p) for p in artifacts.values() if not p.is_file()]
    if missing:
        sys.stderr.write(
            "error: PLAN_ARTIFACT_MISSING (fail-closed): {}\n".format(
                ", ".join(sorted(missing))
            )
        )
        return 1
    shas = {field: sha256_file(path) for field, path in artifacts.items()}
    bad = [f for f in SHA_FIELDS if classify_sha(shas.get(f)) != "real"]
    if bad:
        sys.stderr.write("error: PLAN_SHA_NOT_REAL: {}\n".format(", ".join(bad)))
        return 1

    seeds = SEEDS_S1 if tier == "2k" else SEEDS_FORMAL
    n_updates = smoke_updates if tier == "2k" else (freeze_updates + joint_updates)
    if n_updates is None or n_updates <= 0:
        sys.stderr.write(
            "error: CONFIG_UPDATES: no usable n_updates for tier {} "
            "(training_contract)\n".format(tier)
        )
        return 1

    header: Dict[str, Any] = {
        "record_type": "plan_header",
        "campaign_id": CAMPAIGN_ID,
        "campaign_mode": "full",
        "generated_at": _utc_now_iso(),
        "tier": tier,
        "phase": phase,
        "budget": dict(PHASE_CAPS),
        "counting": "all launch attempts incl. retries (per-phase caps)",
        "seeds": list(seeds),
        "sha_status": {field: classify_sha(shas[field]) for field in SHA_FIELDS},
        "config_path": str(args.config),
        "matrix_ref": ["V5.3-MP plan §5 stage {}".format(phase)],
    }
    lines: List[Dict[str, Any]] = [header]
    run_ids: List[str] = []
    for arm in ARMS:
        for seed in seeds:
            run_id = "v53mp_{}_{}_s{}".format(tier, arm, seed)
            run_ids.append(run_id)
            line: Dict[str, Any] = {
                "run_id": run_id,
                "campaign_id": CAMPAIGN_ID,
                "phase": phase,
                "substage": "v53mp-{}".format(tier),
                "tier": tier,
                "arm": arm,
                "fold": 0,
                "seed": int(seed),
                "variant": None,
                "n_updates": int(n_updates),
                "smoke_updates": smoke_updates if tier == "2k" else None,
                "status": "planned",
                "confirm": False,
                "r3_init": R3_WINNER_RUN_ID,
            }
            line.update(shas)
            lines.append(line)

    plan_path.parent.mkdir(parents=True, exist_ok=True)
    plan_path.write_text(
        "".join(json.dumps(ln, sort_keys=True, ensure_ascii=False) + "\n" for ln in lines),
        encoding="utf-8",
    )
    pin = {"sha256": sha256_file(plan_path), "campaign_mode": "full"}
    Path(str(plan_path) + ".pin.json").write_text(
        json.dumps(pin, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "action": "plan_init",
        "plan": str(plan_path),
        "pin": str(plan_path) + ".pin.json",
        "tier": tier,
        "phase": phase,
        "count": len(run_ids),
        "run_ids": run_ids,
        "plan_sha256": pin["sha256"],
    }, sort_keys=True))
    return 0


# --------------------------------------------------------------------------
# gate-check (S0-F readiness, plan §4.6 / §8.3)
# --------------------------------------------------------------------------


def _gate_item_import(attr: str, module: str) -> Tuple[str, str]:
    import importlib

    try:
        mod = importlib.import_module(module)
    except Exception as exc:  # noqa: BLE001 - fail-closed: ANY import error fails
        return "fail", "import {} failed: {}".format(module, exc)
    if not hasattr(mod, attr):
        return "fail", "{} has no attribute {!r}".format(module, attr)
    return "pass", "{}.{}".format(module, attr)


def _gate_item_file_import(rel_path: str, attr: str, module: str) -> Tuple[str, str]:
    path = PROJECT_ROOT / rel_path
    if not path.is_file():
        return "fail", "{} missing".format(rel_path)
    return _gate_item_import(attr, module)


def _verify_manifest_shas(tables_dir: Path, name: str) -> Tuple[str, str]:
    path = tables_dir / name
    if not path.is_file():
        return "fail", "{} missing".format(name)
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return "fail", "{} unreadable: {}".format(name, exc)
    files = doc.get("files") if isinstance(doc, dict) else None
    if isinstance(files, list):
        entries = {str(e.get("path")): str(e.get("sha256")) for e in files}
    elif isinstance(files, dict):
        entries = {str(k): str(v) for k, v in files.items()}
    else:
        entries = None
    if entries is None:
        return "pass", "{} present (no file list to verify)".format(name)
    for rel, sha in entries.items():
        target = tables_dir / rel
        if not target.is_file():
            return "fail", "{}: listed file {} missing".format(name, rel)
        actual = sha256_file(target)
        if actual != sha:
            return "fail", "{}: {} sha mismatch".format(name, rel)
    return "pass", "{} verified ({} file(s))".format(name, len(entries))


def cmd_gate_check(args: argparse.Namespace) -> int:
    items: Dict[str, str] = {}
    details: Dict[str, str] = {}

    def record(name: str, result: Tuple[str, str]) -> None:
        items[name], details[name] = result

    # -- S0-F readiness items 1-7 -----------------------------------------
    config_path = Path(args.config)
    if not config_path.is_file():
        record("config_yaml", ("fail", "{} missing".format(config_path)))
    else:
        try:
            config = _load_campaign_config(config_path)
            record(
                "config_yaml",
                ("pass", "sha256 {} (schema {})".format(
                    sha256_file(config_path)[:12], config.get("schema"))),
            )
        except V53MPConfigError as exc:
            record("config_yaml", ("fail", str(exc)))
    record("loss_key_atom_v53mp", _gate_loss_item())
    record("trainer_key_atom_v53mp", _gate_item_file_import(
        "pes2ts_core/pesn/offline/training/key_atom_v53mp_train_v3.py",
        "train_run_v53mp",
        "pes2ts_core.pesn.offline.training.key_atom_v53mp_train_v3",
    ))
    record("run_driver_r3_v53mp", (
        "pass" if Path(__file__).is_file() else "fail", str(Path(__file__))
    ))
    record("sharded_loader", _gate_item_import(
        "ShardedKeyAtomDatasetV3",
        "pes2ts_core.pesn.offline.data.key_atom_dataset_v3",
    ))
    record("endpoint_label_spec_v1", _gate_item_import(
        "build_endpoint_labels_spec_v1",
        "pes2ts_core.pesn.offline.data.key_atom_endpoint_labels_v1",
    ))
    record("export_move_process_bundle", _gate_item_import(
        "export_move_process_bundle",
        "pes2ts_core.pesn.offline.publishing.export",
    ))

    # -- data gates (S0-B/C/D artifacts) -----------------------------------
    tables_dir = Path(args.tables_dir)
    from pes2ts_core.pesn.offline.data.key_atom_dataset_v3 import TABLE_FILENAMES

    missing_tables = [
        filename
        for filename in TABLE_FILENAMES.values()
        if not (tables_dir / filename).is_file()
    ]
    if missing_tables:
        record("data_tables_six", (
            "fail", "missing in {}: {}".format(tables_dir, ", ".join(missing_tables)),
        ))
    else:
        record("data_tables_six", ("pass", "six tables present in {}".format(tables_dir)))
    record("data_manifest", _verify_manifest_shas(tables_dir, "data_manifest.json"))
    record("teacher_manifest", _verify_manifest_shas(tables_dir, "teacher_manifest.json"))

    split_path = Path(args.split) if args.split else tables_dir / DEFAULT_SPLIT_BASENAME
    if not split_path.is_file():
        record("split_manifest", ("fail", "{} missing".format(split_path)))
    else:
        try:
            split_doc = json.loads(split_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            record("split_manifest", ("fail", "unreadable: {}".format(exc)))
        else:
            schema = None
            if isinstance(split_doc, dict):
                schema = split_doc.get("schema") or split_doc.get("schema_version")
            if schema != "v53mp_split_v1":
                record("split_manifest", (
                    "fail", "schema {!r} != 'v53mp_split_v1'".format(schema),
                ))
            else:
                record("split_manifest", ("pass", "v53mp_split_v1 @ {}".format(split_path)))

    truth_log = Path(args.truth_log) if args.truth_log else (
        tables_dir / "truth_access_log.jsonl"
    )
    if not truth_log.is_file():
        record("truth_access_log", ("fail", "{} missing".format(truth_log)))
    else:
        try:
            log_lines = [
                ln for ln in truth_log.read_text(encoding="utf-8").splitlines() if ln.strip()
            ]
            for lineno, ln in enumerate(log_lines, start=1):
                parsed = json.loads(ln)
                if not isinstance(parsed, dict):
                    raise ValueError("line {} is not an object".format(lineno))
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            record("truth_access_log", ("fail", "invalid log: {}".format(exc)))
        else:
            record("truth_access_log", (
                "pass", "{} valid access record(s)".format(len(log_lines)),
            ))

    ok = all(v == "pass" for v in items.values())
    print(json.dumps({"items": items, "details": details, "ok": ok}, sort_keys=True))
    return 0 if ok else 1


def _gate_loss_item() -> Tuple[str, str]:
    """S0-F item 2: loss module exists and imports (entry-point presence)."""
    rel = "pes2ts_core/pesn/models/key_atom_loss_v53mp_v3.py"
    path = PROJECT_ROOT / rel
    if not path.is_file():
        return "fail", "{} missing".format(rel)
    import importlib

    try:
        mod = importlib.import_module("pes2ts_core.pesn.models.key_atom_loss_v53mp_v3")
    except Exception as exc:  # noqa: BLE001 - fail-closed
        return "fail", "import failed: {}".format(exc)
    symbols = [n for n in dir(mod) if "loss" in n.lower() or "v53mp" in n.lower()]
    if not symbols:
        return "fail", "module imports but exposes no v53mp loss symbol"
    return "pass", "imported {}".format(rel)


# --------------------------------------------------------------------------
# run
# --------------------------------------------------------------------------


def cmd_run(args: argparse.Namespace) -> int:
    _assert_v53mp_root(args.output_root)
    out_root = Path(args.output_root)
    run_id = args.run_id
    try:
        tier, arm, seed = _parse_run_id(run_id)
    except ValueError as exc:
        sys.stderr.write("error: RUN_ID_INVALID: {}\n".format(exc))
        return 2
    phase = TIER_PHASE[tier]

    plan_path = _resolve_plan_path(args, tier)
    plan = read_run_plan(plan_path)  # LedgerError -> main() exit 2
    line = plan["by_run_id"].get(run_id)
    if line is None:
        sys.stderr.write("error: RUN_NOT_IN_PLAN: {}\n".format(run_id))
        return 2
    if (
        line.get("campaign_id") != CAMPAIGN_ID
        or line.get("arm") != arm
        or int(line.get("seed", -1)) != seed
        or line.get("tier") != tier
    ):
        sys.stderr.write("error: PLAN_LINE_DRIFT: {} does not match plan line\n".format(run_id))
        return 2
    _verify_plan_pin_sidecar(plan_path)

    config_path = Path(args.config)
    config = _load_campaign_config(config_path)

    # -- authorization gate (BEFORE ledger / GPU) --------------------------
    auth = dict((config.get("governance") or {}).get("authorization") or {})
    bit = STAGE_AUTH_BIT[phase]
    if auth.get(bit) is not True:
        refusal = {
            "action": "refused",
            "reason": "authorization",
            "run_id": run_id,
            "plan_run_id": line["run_id"],
            "phase": phase,
            "tier": tier,
            "arm": arm,
            "seed": seed,
            "authorization_bit": bit,
            "value": auth.get(bit),
            "note": "S0 gates未全通过前不得启动训练；翻转授权位=重新冻结 config",
            "ts": _utc_now_iso(),
        }
        _record_refusal(out_root, run_id, refusal)
        print(json.dumps(refusal, sort_keys=True))
        return 1

    # -- λ calibration gate (arms B/C) -------------------------------------
    values = dict(
        ((config.get("training_contract") or {}).get("lambda_probe") or {}).get("values")
        or {}
    )
    missing_lambdas = [
        key for key in ARM_REQUIRED_LAMBDAS[arm] if values.get(key) is None
    ]
    if missing_lambdas:
        refusal = {
            "action": "refused",
            "reason": "lambda_not_calibrated",
            "run_id": run_id,
            "phase": phase,
            "arm": arm,
            "missing": missing_lambdas,
            "note": "λ 未由 gradient-share probe 回填（2–10%）；回填 config 并重建 plan 后再启动",
            "ts": _utc_now_iso(),
        }
        _record_refusal(out_root, run_id, refusal)
        print(json.dumps(refusal, sort_keys=True))
        return 1

    tables_dir = Path(args.tables_dir)
    split_path = Path(args.split) if args.split else tables_dir / DEFAULT_SPLIT_BASENAME
    _verify_five_sha_binding(line, config_path, tables_dir, split_path)

    r3_ckpt = Path(args.r3_ckpt)
    if not r3_ckpt.is_file():
        sys.stderr.write("error: R3_INIT_MISSING: {}\n".format(r3_ckpt))
        return 2

    assert_stage_binding(plan_path, stage="g0", run_id=run_id)

    ledger = RunLedger(_resolve_ledger_path(args), caps=dict(PHASE_CAPS))
    ledger.reconcile()

    record = _v53mp_record(line, attempt=1)
    rid = compute_run_id(record, "auto")
    record["run_id"] = rid
    entry = ledger.status(rid)
    if entry is not None and entry.get("status") == "done":
        print(json.dumps({"run_id": run_id, "ledger_run_id": rid,
                          "action": "already_done"}, sort_keys=True))
        return 0
    attempt = 1 if (entry is None or entry.get("status") == "planned") else \
        2 + int(entry.get("retry_count", 0) or 0)
    record = _v53mp_record(line, attempt=attempt,
                           parent=rid if attempt > 1 else None)
    record["run_id"] = rid

    # -- S1 GPU-seconds budget (refuse BEFORE any launch) -------------------
    if phase == "S1":
        budget_cfg = dict(config.get("budget") or {})
        cap = int(budget_cfg.get("s1_gpu_seconds_cap", DEFAULT_S1_GPU_SECONDS_CAP))
        estimate = int(budget_cfg.get(
            "s1_gpu_seconds_per_run_estimate", DEFAULT_S1_GPU_SECONDS_PER_RUN
        ))
        spent = sum(
            int(e.get("gpu_seconds") or 0)
            for e in ledger.entries(campaign_id=CAMPAIGN_ID, phase="S1")
        )
        if spent + estimate > cap:
            refusal = {
                "action": "refused",
                "reason": "s1_gpu_budget_exceeded",
                "run_id": run_id,
                "phase": phase,
                "gpu_seconds_spent": spent,
                "gpu_seconds_estimate": estimate,
                "gpu_seconds_cap": cap,
                "note": "S1 ≤1 GPU-h 是资源上限；记录 budget_censored 并申请调整，不得突破",
                "ts": _utc_now_iso(),
            }
            _record_refusal(out_root, run_id, refusal)
            print(json.dumps(refusal, sort_keys=True))
            return 1

    if entry is None:
        ledger.append(record)

    handle = gpu_acquire(args.gpu_lock, lane="V53MP-{}".format(phase),
                         run_id=rid, cuda=True)
    if handle is None:
        sys.stderr.write("error: GPU lock busy; run stays {}\n".format(
            entry.get("status") if entry is not None else "planned"))
        return 1
    out_dir = out_root / run_id
    try:
        report = attempt_launch(ledger, record, stage="g0",
                                task_state=args.task_state)
        if report["state"] != "running":
            print(json.dumps(report, sort_keys=True))
            return 3
        try:
            from pes2ts_core.pesn.offline.data.key_atom_dataset_v3 import (
                LABEL_STATUSES,
                ShardedKeyAtomDatasetV3,
            )
            from pes2ts_core.pesn.offline.data.key_atom_endpoint_labels_v1 import (
                build_endpoint_labels_spec_v1,
            )
            from pes2ts_core.pesn.offline.training.key_atom_v53mp_train_v3 import (
                V53MPTrainerConfig,
                train_run_v53mp,
                v53mp_pool_split,
            )

            provenance = _build_training_provenance(
                config, config_path, tables_dir, split_path,
                run_id=run_id, tier=tier,
            )
            dataset = ShardedKeyAtomDatasetV3(
                args.tables_dir,
                split_manifest=str(split_path),
                label_statuses=LABEL_STATUSES,   # ok/masked/terminal/conflict
                light_cache=True,   # V5.3-MP: manifest-bound light-index cache
                # Geometry LRU sized to the tier pool (+ dev margin): each
                # curve's shard geometry is decompressed at most ONCE per
                # run instead of ~6x at 5k / ~6x at 20k (measured ~40 KB
                # per curve → ≤1 GB at 20k).  Shard sha256 is pre-verified
                # at init, so the loop never hashes.
                cache_curves=GEOMETRY_CACHE_CURVES[tier],
            )
            samples = dataset.samples
            # Label cache (V5.3-MP): the v52 + endpoint label tables are pure
            # functions of the frozen tables; reuse the cache written by the
            # S0 finalize, fail-closed on staleness (never silently recompute
            # against a drifted dataset).
            from pes2ts_core.pesn.offline.data.key_atom_label_cache_v53mp import (
                LabelCacheError,
                load_label_cache,
                save_label_cache,
            )
            try:
                labels, endpoint_labels = load_label_cache(
                    args.tables_dir,
                    dataset_manifest_sha256=dataset.manifest_sha256,
                )
            except LabelCacheError:
                labels = dataset.v52_labels()
                endpoint_labels = build_endpoint_labels_spec_v1(dataset, labels)
                save_label_cache(
                    args.tables_dir, labels, endpoint_labels,
                    dataset_manifest_sha256=dataset.manifest_sha256,
                )
            split_doc = json.loads(split_path.read_text(encoding="utf-8"))
            pool_curve_ids = _tier_pool_curve_ids(split_doc, tier)
            # Preflight: typed refusals on empty/confirm-contaminated pools.
            train_pos, inner_val_pos = v53mp_pool_split(
                samples, pool_curve_ids=pool_curve_ids, seed=seed
            )
            # Trainer contract: train_positions = the FULL inner-train pool
            # (the trainer re-derives the identical curve-level split);
            # dev_positions = the FROZEN dev subset of the split manifest.
            train_positions = sorted(
                {int(p) for p in list(train_pos) + list(inner_val_pos)}
            )
            dev_curve_ids = set(_subset_curve_ids(split_doc, "dev"))
            dev_positions = [
                i for i, s in enumerate(samples)
                if str(s.curve_id) in dev_curve_ids
            ]
            if not dev_positions:
                raise LedgerError(
                    "SPLIT_SCHEMA: frozen dev subset has no loaded samples"
                )
            schedule = dict((config.get("training_contract") or {}).get("schedule") or {})
            # per-arm λ gating (the trainer validates arm/λ consistency):
            # A = move-only (all 0); B = +rank +q; C = +endpoint too
            lam_rank = float(values.get("lambda_rank") or 0.0)
            lam_q = float(values.get("lambda_q") or 0.0)
            lam_ep = float(values.get("lambda_endpoint") or 0.0)
            if arm == "A":
                lam_rank = lam_q = lam_ep = 0.0
            elif arm == "B":
                lam_ep = 0.0
            trainer_config = V53MPTrainerConfig(
                arm=arm,
                seed=seed,
                backbone_init="r3",
                lambda_rank=lam_rank,
                lambda_q=lam_q,
                lambda_endpoint=lam_ep,
                freeze_encoder_updates=int(schedule.get("freeze_encoder_updates", 400)),
                joint_updates=_tier_joint_updates(schedule, tier),
                lr_head=float(schedule.get("lr_head", 3e-4)),
                lr_encoder=float(schedule.get("lr_encoder", 1e-4)),
                weight_decay=float(schedule.get("weight_decay", 1e-4)),
                grad_clip=float(schedule.get("grad_clip", 1.0)),
                smoke_updates=(
                    int(line["smoke_updates"]) if tier == "2k" and line.get("smoke_updates")
                    else None
                ),
                eval_max_samples=(
                    int((config.get("training_contract") or {})
                        .get("s1_eval_max_samples"))
                    if tier == "2k"
                    and (config.get("training_contract") or {})
                        .get("s1_eval_max_samples")
                    else None
                ),
                inner_val_max_samples=(
                    int((config.get("training_contract") or {})
                        .get("inner_val_max_samples"))
                    if (config.get("training_contract") or {})
                        .get("inner_val_max_samples")
                    else None
                ),
                device=args.device,
            )
            metrics = train_run_v53mp(
                dataset, labels, endpoint_labels, trainer_config, out_dir,
                r3_ckpt=r3_ckpt, r3_eval_ckpt=r3_ckpt,
                dev_positions=dev_positions, train_positions=train_positions,
                resume=True, provenance=provenance,
            )
        except BaseException as exc:
            fail = dict(record)
            fail["status"] = "failed"
            try:
                ledger.append(fail)
            except LedgerError as ledger_exc:
                sys.stderr.write(
                    "note: failed-record append refused: {}\n".format(ledger_exc))
            sys.stderr.write("error: run {} failed: {}\n".format(run_id, exc))
            return 1
    finally:
        handle.release()

    ckpt_sha = metrics.get("checkpoint_best_sha256")
    selected = metrics.get("selected_update")
    if not (isinstance(ckpt_sha, str) and len(ckpt_sha) == 64):
        fail = dict(record)
        fail["status"] = "failed"
        try:
            ledger.append(fail)
        except LedgerError as ledger_exc:
            sys.stderr.write(
                "note: failed-record append refused: {}\n".format(ledger_exc))
        sys.stderr.write(
            "error: NO_CHECKPOINT_EVIDENCE: {}\n".format(run_id))
        return 1
    done = dict(record)
    done.update({
        "status": "done",
        "checkpoint_update": int(selected if selected is not None else 0),
        "checkpoint_sha256": ckpt_sha,
        "metrics_path": str(out_dir / "metrics.json"),
        "gpu_seconds": int(float(metrics.get("wall_clock_s") or 0)),
    })
    outcome = ledger.append(done)
    print(json.dumps({
        "run_id": run_id,
        "ledger_run_id": rid,
        "action": outcome.action,
        "arm": arm,
        "tier": tier,
        "phase": phase,
        "seed": seed,
        "backbone_run_id": R3_WINNER_RUN_ID,
        "n_updates": int(line["n_updates"]),
        "selected_update": int(selected if selected is not None else 0),
        "gpu_seconds": done["gpu_seconds"],
        "checkpoint_sha256": ckpt_sha,
        "metrics_path": done["metrics_path"],
    }, sort_keys=True))
    return 0


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="r3_v53mp_run",
        description="V5.3-MP MOVE+PROCESS planned run driver "
                    "(plan-init / gate-check / run)",
    )
    parser.add_argument("--output-root", default=DEFAULT_V53MP_ROOT)
    sub = parser.add_subparsers(dest="cmd", required=True)

    init = sub.add_parser("plan-init", help="freeze one tier's run plan + pin")
    init.add_argument("--tier", required=True, choices=sorted(TIER_PHASE))
    init.add_argument("--plan", default=None)
    init.add_argument("--config", default=DEFAULT_CONFIG)
    init.add_argument("--tables-dir", default=DEFAULT_TABLES_DIR)
    init.add_argument("--split", default=None)
    init.add_argument("--force", action="store_true",
                      help="rebuild an existing plan (requires re-signoff)")

    gate = sub.add_parser("gate-check",
                          help="S0-F readiness 7/7 + data-gate verification")
    gate.add_argument("--config", default=DEFAULT_CONFIG)
    gate.add_argument("--tables-dir", default=DEFAULT_TABLES_DIR)
    gate.add_argument("--split", default=None)
    gate.add_argument("--truth-log", default=None)

    run = sub.add_parser("run", help="one planned V5.3-MP run")
    run.add_argument("--run-id", required=True)
    run.add_argument("--plan", default=None,
                     help="default: <output-root>/run_plan_v53mp_<tier>.jsonl")
    run.add_argument("--ledger", default=None,
                     help="default: <output-root>/run_records.jsonl")
    run.add_argument("--config", default=DEFAULT_CONFIG)
    run.add_argument("--tables-dir", default=DEFAULT_TABLES_DIR)
    run.add_argument("--split", default=None)
    run.add_argument("--task-state", default=DEFAULT_TASK_STATE_PATH)
    run.add_argument("--r3-ckpt", default=R3_WINNER_CKPT)
    run.add_argument("--gpu-lock", default=GPU_LOCK)
    run.add_argument("--device", default="cuda")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        if args.cmd == "plan-init":
            return cmd_plan_init(args)
        if args.cmd == "gate-check":
            return cmd_gate_check(args)
        if args.cmd == "run":
            return cmd_run(args)
        return 2
    except CampaignRootError as exc:
        sys.stderr.write("error: {}\n".format(exc))
        return 1
    except V53MPConfigError as exc:
        sys.stderr.write("error: {}\n".format(exc))
        return 2
    except (LedgerError, OSError) as exc:
        sys.stderr.write("error: {}\n".format(exc))
        return 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
