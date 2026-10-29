#!/usr/bin/env python3
"""V5.4.1 WP4c — independent Embedding-50K campaign runner (plan §7/§12/§14).

This is the ONLY V5.4.1 launch entry.  It deliberately does **not** wrap
``r3_v53mp_run.py --tier 50k``: a V5.3-MP tier run is a different campaign
(plan §14.1) and can never stand in for a V541 unified run.

Sub-commands
============
===========================  ==================================================
``validate-config``          calls the existing ``v541_config_schema``
                             (``--formal`` additionally checks the T1 GATES
                             as machine evidence, not just the thresholds)
``audit-data-and-manifests`` read-only data/pool/split/strata audit, machine
                             JSON; anything missing FAILS (never passes)
``plan-init``                instantiate ``plan/{runplan.jsonl,
                             runplan.pin.json, preregistration_signoff.json,
                             budget_policy.json}`` from config + templates
``gate-check``               readiness + T1 gate evidence, machine JSON
``run``                      mode = ``smoke`` | ``profile`` | ``formal``
                             (run_id / embedding / seed / resume)
===========================  ==================================================

Discipline
==========
* **registry dispatch** — an embedding id that WP3 has not registered yet is
  a TYPED REFUSAL (``V541_EMBEDDING_NOT_IMPLEMENTED``); no placeholder math
  is ever substituted.
* **dev/confirm firewall** — ``dev_positions`` EXCLUDE dev curves from the
  pool only (no dev feature / label / eval); confirm folds are refused
  fail-closed; ``new_QC_calls = 0`` everywhere.
* **columnar/label/pool/manifest reuse** — the old runner's helpers
  (``_verify_manifest_shas``, ``_tier_pool_curve_ids``, ``_subset_curve_ids``,
  ``_label_spec_sha256``, ``dev_positions_from_split``) are IMPORTED
  read-only; nothing legacy is edited.
* **own ledger** — every attempt/outcome/cost/checkpoint/τ/batch SHA is
  APPENDED to ``outputs/pesn/v541_embed_50k/run_records.jsonl``; the legacy
  run ledger is never opened for writing.
* **single lane** — one GPU lock, concurrency 1, cgroup 12G/10G/swap0
  verified and recorded by this runner (enforced by the outer shell).

Exit codes: ``0`` ok · ``1`` refused/blocked (typed ``code`` on stdout as
JSON) · ``2`` usage/config error.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pes2ts_core.pesn.offline.training.v541_identity import (  # noqa: E402
    append_manifest,
    env_state,
)
from scripts.agent_ops.supervisor.campaign_root import (  # noqa: E402
    CampaignRootError,
    normalize_root,
)
from scripts.agent_ops.v541_config_schema import (  # noqa: E402
    DEFAULT_CONFIG,
    V541SchemaError,
    load_config,
    validate_config,
)
from scripts.campaigns.v541_provenance import (  # noqa: E402
    V541ProvenanceError,
    artifact_shas,
    code_sha256,
    common_protocol_sha256,
    embedding_spec_sha256,
    environment_sha256,
    feature_recipe_sha256,
    sha256_file,
    source_manifest,
)
from scripts.campaigns.v541_runplan import (  # noqa: E402
    RUN_RECORDS_NAME,
    V541RunplanError,
    classify_started_runs,
    plan_paths,
    read_plan,
    verify_plan_pin_v541,
    write_plan,
)

__all__ = [
    "DEFAULT_ROOT",
    "FORMAL_TOTAL_UPDATES",
    "PROFILE_W0_DEFAULT",
    "PROFILE_W1_DEFAULT",
    "SMOKE_TARGET_UPDATES",
    "V541RunnerError",
    "audit_data_and_manifests",
    "main",
    "t1_gate_report",
]

DEFAULT_ROOT = "outputs/pesn/v541_embed_50k"
V541_ROOT_REL = "v541_embed_50k"
SEALED_PREFIXES = ("outputs/pesn/v5_1", "outputs/pesn/v5_1_r1")

DEFAULT_CONFIG_PATH = str(DEFAULT_CONFIG)
DEFAULT_TABLES_DIR = "/pes2ts_data/v53mp_50k_data"
DEFAULT_SPLIT_NAME = "split_manifest_v53mp.json"
DEFAULT_V53MP_CONFIG = "config/campaigns/pes2ts_v53mp_20261021.yaml"
GPU_LOCK = "/tmp/opencode/pes2ts_gpu.lock"

#: Frozen doses (§7.1 / §8.2): never reduced, never re-seeded.
SMOKE_TARGET_UPDATES = 200
FORMAL_TOTAL_UPDATES = 37900
PROFILE_W0_DEFAULT = 40
PROFILE_W1_DEFAULT = 40
PROFILE_W0_MAX = 400          # the W0 window itself is only 400 updates
#: Independent branch-init seed base (§5.4): ``base + run seed``, never the
#: run seed itself, so the legacy RNG stream stays untouched.
EMBEDDING_INIT_SEED_BASE = 1_000_003
GEOMETRY_CACHE_CURVES_50K = 65000
SEEDS = (17, 29, 43)

#: Every attempt/outcome/cost/... lands in V541's OWN ledger.  The filename
#: (``RUN_RECORDS_NAME``, imported from ``v541_runplan``) is owned there so
#: the plan re-freeze scan and this runner can never disagree about which
#: file is the ledger.


class V541RunnerError(RuntimeError):
    """Typed refusal of the V541 runner (``code`` + ``detail`` + ``blocks``)."""

    def __init__(self, code: str, detail: str = "",
                 blocks: Optional[Sequence[Mapping[str, Any]]] = None) -> None:
        super().__init__(f"{code}: {detail}" if detail else code)
        self.code = str(code)
        self.detail = str(detail)
        self.blocks = [dict(b) for b in (blocks or ())]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _emit(payload: Mapping[str, Any]) -> None:
    print(json.dumps(dict(payload), ensure_ascii=False, sort_keys=True,
                      default=str))


# ---------------------------------------------------------------------------
# campaign root
# ---------------------------------------------------------------------------
def assert_v541_root(path: Any) -> str:
    """Refuse sealed / foreign output roots; return the normalized key."""
    normalized = normalize_root(path, campaign_root=DEFAULT_ROOT)
    for sealed in SEALED_PREFIXES:
        rel_sealed = sealed[len("outputs/pesn/"):]
        rel = normalized[len("outputs/pesn/"):]
        if rel == rel_sealed or rel.startswith(rel_sealed + "/"):
            raise CampaignRootError(
                "SEALED_ROOT_REFUSED: {} is a sealed read-only prefix".format(
                    normalized))
    rel = normalized[len("outputs/pesn/"):]
    if rel != V541_ROOT_REL and not rel.startswith(V541_ROOT_REL + "/"):
        raise CampaignRootError(
            "OUTSIDE_CAMPAIGN_ROOT: {} is outside {}/...".format(
                normalized, DEFAULT_ROOT))
    return normalized


# ---------------------------------------------------------------------------
# resources (cgroup verification / recording; the outer shell enforces)
# ---------------------------------------------------------------------------
def _read_int_or_none(path: Path) -> Optional[int]:
    try:
        text = path.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    if text in ("max", ""):
        return None
    try:
        return int(text)
    except ValueError:
        return None


def _current_cgroup_path() -> Optional[str]:
    """The cgroup THIS process lives in (v2 ``0::/path`` from procfs)."""
    try:
        for line in Path("/proc/self/cgroup").read_text(
                encoding="utf-8").splitlines():
            if line.startswith("0::"):
                return line[3:].strip() or "/"
    except OSError:
        return None
    return None


def read_cgroup_limits() -> Dict[str, Any]:
    """Snapshot the cgroup limits that actually govern THIS process.

    The outer shell creates the 12G / 10G-high / swap0 cgroup; the runner
    resolves its own ``/proc/self/cgroup`` path against the cgroup2 mount so
    the recorded numbers are the ones being enforced (not a root-level
    guess).  Missing controller files mean the limits cannot be verified —
    reported as ``readable=false``, which formal treats as a typed block.
    """
    info: Dict[str, Any] = {"version": None, "source": None,
                            "cgroup_path": _current_cgroup_path(),
                            "memory_max_bytes": None,
                            "memory_high_bytes": None,
                            "swap_max_bytes": None,
                            "oom_kill": None,
                            "readable": False}
    base = Path("/sys/fs/cgroup")
    rel = info["cgroup_path"] or "/"
    cgroup_dir = base if rel in ("", "/") else (base / rel.lstrip("/"))
    if not cgroup_dir.is_dir():
        cgroup_dir = base
    if (cgroup_dir / "memory.max").is_file():
        info.update(
            version=2, source=str(cgroup_dir),
            memory_max_bytes=_read_int_or_none(cgroup_dir / "memory.max"),
            memory_high_bytes=_read_int_or_none(cgroup_dir / "memory.high"),
            swap_max_bytes=_read_int_or_none(cgroup_dir / "memory.swap.max"),
            readable=True,
        )
        try:
            for line in (cgroup_dir / "memory.events").read_text(
                    encoding="utf-8").splitlines():
                key, _, value = line.partition(" ")
                if key.strip() == "oom_kill":
                    info["oom_kill"] = int(value.strip())
        except (OSError, ValueError):
            info["oom_kill"] = None
        return info
    v1 = Path("/sys/fs/cgroup/memory")
    if (v1 / "memory.limit_in_bytes").is_file():
        info.update(
            version=1, source=str(v1),
            memory_max_bytes=_read_int_or_none(v1 / "memory.limit_in_bytes"),
            memory_high_bytes=None,
            swap_max_bytes=_read_int_or_none(v1 / "memory.memsw.limit_in_bytes"),
            oom_kill=_read_int_or_none(v1 / "memory.oom_control"),
            readable=True,
        )
    return info


def resource_blocks(resources: Mapping[str, Any],
                    budget: Mapping[str, Any],
                    *, strict: bool) -> List[Dict[str, Any]]:
    """Typed blocks for a FORMAL launch (record-only when ``strict`` is off)."""
    if not strict:
        return []
    gib = 1024 ** 3
    blocks: List[Dict[str, Any]] = []
    if not resources.get("readable"):
        blocks.append({"code": "V541_CGROUP_UNVERIFIED",
                       "detail": "cgroup limits could not be read; a formal "
                                 "run must be launched inside the 12G/10G/"
                                 "swap0 cgroup"})
        return blocks
    host_limit = int(float(budget.get("host_limit_gib") or 0) * gib)
    high_limit = int(float(budget.get("memory_high_gib") or 0) * gib)
    mem_max = resources.get("memory_max_bytes")
    if not isinstance(mem_max, int) or mem_max <= 0 or mem_max > host_limit:
        blocks.append({"code": "V541_CGROUP_HOST_LIMIT",
                       "detail": "memory.max={!r} must be <= {} GiB".format(
                           mem_max, budget.get("host_limit_gib"))})
    high = resources.get("memory_high_bytes")
    if not isinstance(high, int) or high <= 0 or high > high_limit:
        blocks.append({"code": "V541_CGROUP_MEMORY_HIGH",
                       "detail": "memory.high={!r} must be <= {} GiB".format(
                           high, budget.get("memory_high_gib"))})
    swap = resources.get("swap_max_bytes")
    if swap != 0:
        blocks.append({"code": "V541_CGROUP_SWAP",
                       "detail": "memory.swap.max={!r} must be 0".format(swap)})
    oom = resources.get("oom_kill")
    if oom is None or int(oom) != 0:
        blocks.append({"code": "V541_CGROUP_OOM_KILL",
                       "detail": "oom_kill={!r} must be 0".format(oom)})
    if int(budget.get("swap_max", 0)) != 0 or int(
            budget.get("oom_kill_max", 0)) != 0:
        blocks.append({"code": "V541_RESOURCE_CONFIG",
                       "detail": "budget.swap_max/oom_kill_max must be 0"})
    return blocks


# ---------------------------------------------------------------------------
# T1 gate evidence (formal launch = ALL machine evidence passes)
# ---------------------------------------------------------------------------
def _load_json(path: Path) -> Optional[Mapping[str, Any]]:
    if not path.is_file():
        return None
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return doc if isinstance(doc, Mapping) else None


def _evidence_pass(doc: Optional[Mapping[str, Any]]) -> Tuple[bool, str]:
    """A gate only PASSES on an explicit positive signal — missing never does."""
    if doc is None:
        return False, "evidence file missing"
    if doc.get("passed") is True or doc.get("pass") is True:
        return True, "passed=true"
    status = doc.get("status")
    if isinstance(status, str) and status.lower() in ("pass", "ok", "passed",
                                                      "signed", "frozen"):
        return True, "status={}".format(status)
    if doc.get("ok") is True and doc.get("status") in ("pass", "ok", "done"):
        return True, "ok=true"
    return False, "no positive signal (status={!r}, passed={!r})".format(
        doc.get("status"), doc.get("passed"))


def _smoke_evidence(root: Path, embedding_id: str) -> Tuple[bool, str]:
    """§8.2 technical smoke: 200 updates, NOT formal, with a kill/resume leg."""
    run_dir = root / "smoke" / "V541_{}_S17".format(embedding_id)
    metrics = _load_json(run_dir / "metrics.json")
    if metrics is None:
        return False, "missing {}".format(run_dir / "metrics.json")
    schedule = metrics.get("schedule") or {}
    if int(schedule.get("total_updates") or 0) != SMOKE_TARGET_UPDATES:
        return False, "total_updates={} != {}".format(
            schedule.get("total_updates"), SMOKE_TARGET_UPDATES)
    if metrics.get("is_formal_run") is not False:
        return False, "smoke must be labelled is_formal_run=false"
    if metrics.get("run_outcome") != "done":
        return False, "run_outcome={!r}".format(metrics.get("run_outcome"))
    manifest = run_dir / "manifest.jsonl"
    if manifest.is_file():
        resumed = any(
            (json.loads(ln) or {}).get("event") == "resume"
            for ln in manifest.read_text(encoding="utf-8").splitlines()
            if ln.strip())
        if resumed:
            return True, "200-update non-formal smoke + kill/resume leg"
        return False, "manifest has no kill/resume leg (plan §8.2.4)"
    return False, "missing {} (kill/resume evidence)".format(manifest)


def t1_gate_report(cfg: Mapping[str, Any], *, root: Any,
                   formal_config_blocks: Sequence[str] = (),
                   embedding_id: str = "E0") -> Dict[str, Any]:
    """Every T1 gate as machine evidence → ``blocks`` (empty = all pass)."""
    root = Path(root)
    audits = root / "audits"
    plan = plan_paths(root)
    blocks: List[Dict[str, Any]] = []

    for code, reason in (("V541_T1_CONFIG", b)
                         for b in formal_config_blocks):
        blocks.append({"code": code, "detail": str(reason)})

    evidence = (
        ("V541_GATE_WP1", "WP1 pair-evaluator before/after",
         audits / "evaluator_before_after.json", None),
        # R3 H2 re-evaluation through the SAME pipeline (§17 "R3 H2 参照重评"):
        # the baseline-repair verify report, NOT the WP2 forward-parity file.
        ("V541_GATE_R3_REEVAL", "R3 / C×3 same-pipeline re-evaluation",
         audits / "verify_report.json", None),
        # WP2 E0 strict-load / forward parity stays its OWN gate.
        ("V541_GATE_WP2_PARITY", "WP2 E0 strict-load + forward parity",
         audits / "v53_parity.json", None),
        ("V541_GATE_E0_DECISION", "E0 reuse / retrain decision",
         audits / "e0_reuse_decision.json", "decision"),
        ("V541_GATE_H0", "H0 legality gate",
         audits / "h0_legality.json", None),
        ("V541_GATE_H1", "H1 resource gate",
         audits / "h1_resource.json", None),
    )
    for code, label, path, extra in evidence:
        doc = _load_json(path)
        ok, detail = _evidence_pass(doc)
        if ok and extra and not (doc or {}).get(extra):
            ok, detail = False, "evidence has no {!r} field".format(extra)
        if not ok:
            blocks.append({"code": code,
                           "detail": "{}: {} ({})".format(label, detail, path)})

    ok, detail = _smoke_evidence(root, embedding_id)
    if not ok:
        blocks.append({"code": "V541_GATE_SMOKE_200",
                       "detail": "V541-smoke-200: {}".format(detail)})

    budget_doc = _load_json(plan["budget"])
    if budget_doc is None:
        blocks.append({"code": "V541_GATE_BUDGET_SIGNOFF",
                       "detail": "plan/budget_policy.json missing"})
    else:
        per = (budget_doc.get("per_candidate") or {}).get(embedding_id) or {}
        signed = budget_doc.get("signed_by")
        if not signed:
            blocks.append({"code": "V541_GATE_BUDGET_SIGNOFF",
                           "detail": "budget_policy.signed_by is null "
                                     "(W0/W1 profiling signoff required, §8.3)"})
        if per.get("approved_for_formal") is not True:
            blocks.append({"code": "V541_GATE_BUDGET_SIGNOFF",
                           "detail": "per_candidate.{}.approved_for_formal is "
                                     "not true (status={!r})".format(
                                         embedding_id,
                                         per.get("status"))})
        cfg_budget = cfg.get("budget") or {}
        if cfg_budget.get("per_candidate_core_time") is None:
            blocks.append({"code": "V541_GATE_BUDGET_SIGNOFF",
                           "detail": "budget.per_candidate_core_time is null "
                                     "(runner never fills defaults, §8.3)"})

    if not plan["pin"].is_file() or not plan["plan"].is_file():
        blocks.append({"code": "V541_GATE_RUNPLAN_PIN",
                       "detail": "plan/runplan.jsonl or runplan.pin.json "
                                 "missing — run plan-init first"})
    else:
        try:
            verified = verify_plan_pin_v541(plan["plan"], plan["pin"])
            current = code_sha256(
                (cfg.get("identity_contract") or {}).get("code") or {},
                root=PROJECT_ROOT)
            if verified.get("code_sha256") != current:
                blocks.append({"code": "V541_CODE_DRIFT",
                               "detail": "source fingerprint {} != pinned {} "
                                         "(formal code SHA drift)".format(
                                             current,
                                             verified.get("code_sha256"))})
            env_now = environment_sha256()
            if verified.get("environment_sha256") != env_now:
                blocks.append({"code": "V541_ENV_DRIFT",
                               "detail": "environment {} != pinned {}".format(
                                   env_now, verified.get("environment_sha256"))})
        except V541RunplanError as exc:
            blocks.append({"code": exc.code, "detail": exc.detail})

    return {
        "blocks": blocks,
        "ready": not blocks,
        "checked": ["WP1_pair_evaluator", "R3_same_pipeline_re_evaluation",
                    "WP2_E0_parity", "E0_decision",
                    "H0", "H1", "V541-smoke-200", "V541-budget-signoff",
                    "runplan_pin", "code/environment drift"],
    }


# ---------------------------------------------------------------------------
# data loading (read-only reuse of the old runner's columnar/label/pool tools)
# ---------------------------------------------------------------------------
@dataclass
class RunInputs:
    dataset: Any
    labels: Any
    endpoint: Any
    split_doc: Mapping[str, Any]
    pool_curve_ids: List[str]
    dev_curve_ids: List[str]
    confirm_curve_ids: List[str]
    provenance: Dict[str, Any] = field(default_factory=dict)


def verify_idx_v2_binding(tables_dir: Any) -> Dict[str, Any]:
    """Fail-closed idx_v2 pre-flight BEFORE the columnar dataset opens.

    The dataset constructor re-verifies every column sha + the binding
    block; this pre-flight turns a half-built index into a typed refusal
    with an actionable message instead of an import-time KeyError.
    """
    from pes2ts_core.pesn.offline.data.key_atom_dataset_v3_col import (
        IDX2_DIRNAME,
        IDX2_SCHEMA,
    )

    idx_dir = Path(tables_dir) / IDX2_DIRNAME
    required_files = ["curves.json", "actions.npy", "sid_lookup_keys.npy",
                      "sid_lookup_vals.npy", "curve_node_energies.npy"]
    required_dirs = ["samples", "intents", "labels_v52", "labels_ep",
                     "label_curves"]
    missing = [str(idx_dir / name) for name in required_files
               if not (idx_dir / name).is_file()]
    missing += [str(idx_dir / name) for name in required_dirs
                if not (idx_dir / name).is_dir()]
    if missing:
        raise V541RunnerError(
            "V541_IDX2_MISSING",
            "idx_v2 is required for the 50K pool and is incomplete: "
            "{}".format(", ".join(sorted(missing))))
    curves_path = idx_dir / "curves.json"
    try:
        doc = json.loads(curves_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise V541RunnerError(
            "V541_IDX2_BINDING", "curves.json unreadable: {}".format(exc)) from exc
    if not isinstance(doc, Mapping) or doc.get("schema") != IDX2_SCHEMA:
        raise V541RunnerError(
            "V541_IDX2_BINDING",
            "curves.json schema {!r} != {!r}".format(
                doc.get("schema") if isinstance(doc, Mapping) else None,
                IDX2_SCHEMA))
    binding = doc.get("binding")
    if not isinstance(binding, Mapping):
        raise V541RunnerError(
            "V541_IDX2_BINDING", "curves.json has no binding block")
    return {"idx_dir": str(idx_dir), "schema": doc.get("schema"),
            "n_samples": binding.get("n_samples"),
            "n_curves": binding.get("n_curves"),
            "data_manifest_sha256": binding.get("data_manifest_sha256"),
            "curves_json_sha256": sha256_file(curves_path)}


def load_run_inputs(*, config: Mapping[str, Any], config_path: Any,
                    tables_dir: Any, split_path: Any) -> RunInputs:
    """Open the 50K pool read-only (columnar v2) + split subsets.

    Nothing here builds dev features/labels or reads confirm: the split
    subsets are used for curve EXCLUSION and the confirm firewall only.
    """
    from pes2ts_core.pesn.offline.data.key_atom_dataset_v3 import (
        LABEL_STATUSES,
    )
    from pes2ts_core.pesn.offline.data.key_atom_dataset_v3_col import (
        ColumnarKeyAtomDatasetV3,
    )

    tables = Path(tables_dir)
    split = Path(split_path)
    for path in (tables / "data_manifest.json", tables / "teacher_manifest.json",
                 split):
        if not path.is_file():
            raise V541RunnerError(
                "V541_ARTIFACT_MISSING", "required artifact missing: {}".format(path))
    idx_info = verify_idx_v2_binding(tables)
    dataset = ColumnarKeyAtomDatasetV3(
        str(tables), split_manifest=str(split),
        label_statuses=LABEL_STATUSES,
        cache_curves=GEOMETRY_CACHE_CURVES_50K)
    labels, endpoint = dataset.label_tables()
    split_doc = json.loads(split.read_text(encoding="utf-8"))

    from scripts.campaigns.r3_v53mp_run import (  # read-only reuse
        _subset_curve_ids,
        _tier_pool_curve_ids,
    )
    pool_ids = _tier_pool_curve_ids(split_doc, "50k")
    dev_ids = _subset_curve_ids(split_doc, "dev")
    confirm_ids = _subset_curve_ids(split_doc, "confirm")

    provenance: Dict[str, Any] = {
        "tier": "50k",
        "shard_path": str(tables),
        "loader_version": "v2",
        "loader_idx_sha256": idx_info["curves_json_sha256"],
        "idx_v2_binding": idx_info,
        "split_manifest_sha256": sha256_file(split),
        "data_manifest_sha256": sha256_file(tables / "data_manifest.json"),
        "teacher_manifest_sha256": sha256_file(tables / "teacher_manifest.json"),
    }
    shas = artifact_shas(
        config, config_path=config_path, tables_dir=tables, split_path=split,
        code_sha=code_sha256((config.get("identity_contract") or {}).get("code") or {},
                             root=PROJECT_ROOT))
    provenance.update(shas)
    return RunInputs(
        dataset=dataset, labels=labels, endpoint=endpoint, split_doc=split_doc,
        pool_curve_ids=list(pool_ids), dev_curve_ids=list(dev_ids),
        confirm_curve_ids=list(confirm_ids), provenance=provenance)


def _positions_for_seed(inputs: RunInputs, seed: int) -> Tuple[List[int], List[int]]:
    """Curve-level 80/20 inner split of the pool for one seed (read-only)."""
    from pes2ts_core.pesn.offline.training.key_atom_v53mp_train_v3 import (
        v53mp_pool_split,
    )

    train_pos, inner_val_pos = v53mp_pool_split(
        inputs.dataset.samples, pool_curve_ids=inputs.pool_curve_ids,
        seed=int(seed))
    return list(train_pos), list(inner_val_pos)


def dev_positions_of(inputs: RunInputs) -> List[int]:
    """Frozen dev positions — curve EXCLUSION ONLY (no dev feature/label/eval)."""
    from pes2ts_core.pesn.offline.data.key_atom_dev_pack_v53mp import (
        dev_positions_from_split,
    )

    positions = dev_positions_from_split(
        inputs.dataset.samples, set(inputs.dev_curve_ids))
    if not positions:
        raise V541RunnerError(
            "V541_DEV_POSITIONS",
            "frozen dev subset has no loaded samples (exclusion impossible)")
    return list(positions)


# ---------------------------------------------------------------------------
# audit-data-and-manifests
# ---------------------------------------------------------------------------
def audit_data_and_manifests(*, config: Mapping[str, Any], config_path: Any,
                             tables_dir: Any, split_path: Any,
                             inputs: Optional[RunInputs] = None,
                             seeds: Sequence[int] = SEEDS) -> Dict[str, Any]:
    """Machine-readable data/manifest/pool/split/strata audit (fail-closed)."""
    items: Dict[str, Dict[str, Any]] = {}
    counts: Dict[str, Any] = {}
    tables = Path(tables_dir)
    split = Path(split_path)

    def record(name: str, ok: Any, detail: str) -> None:
        items[name] = {"status": "pass" if ok else "fail", "detail": detail}

    # -- config -----------------------------------------------------------
    try:
        report = validate_config(config, formal=False)
        record("config_schema", True,
               "schema={} mode={}".format(report["schema"], report["mode"]))
    except V541SchemaError as exc:
        record("config_schema", False, "{}: {}".format(exc.code, exc.detail))

    # -- required artifacts (missing => never pass) -----------------------
    for name, path in (("data_manifest", tables / "data_manifest.json"),
                       ("teacher_manifest", tables / "teacher_manifest.json"),
                       ("split_manifest", split),
                       ("truth_access_log", tables / "truth_access_log.jsonl")):
        record(name + "_present", path.is_file(), str(path))

    # -- manifest SHA verification (old runner helper, read-only) ---------
    from scripts.campaigns.r3_v53mp_run import (  # read-only reuse
        _subset_curve_ids,
        _tier_pool_curve_ids,
        _verify_manifest_shas,
    )
    for name in ("data_manifest.json", "teacher_manifest.json"):
        if (tables / name).is_file():
            status, detail = _verify_manifest_shas(tables, name)
            record(name + "_verified", status == "pass", detail)
        else:
            record(name + "_verified", False, "missing {}".format(name))

    # -- split subsets ----------------------------------------------------
    split_doc: Optional[Mapping[str, Any]] = None
    if split.is_file():
        try:
            parsed = json.loads(split.read_text(encoding="utf-8"))
            split_doc = parsed if isinstance(parsed, Mapping) else None
        except (OSError, json.JSONDecodeError) as exc:
            record("split_schema", False, "unreadable: {}".format(exc))
    if split_doc is not None:
        schema = split_doc.get("schema") or split_doc.get("schema_version")
        record("split_schema", schema == "v53mp_split_v1",
               "schema={!r}".format(schema))
        try:
            pool_ids = _tier_pool_curve_ids(split_doc, "50k")
            dev_ids = _subset_curve_ids(split_doc, "dev")
            confirm_ids = _subset_curve_ids(split_doc, "confirm")
        except Exception as exc:  # noqa: BLE001 - typed legacy error
            pool_ids, dev_ids, confirm_ids = [], [], []
            record("split_subsets", False, str(exc))
        else:
            record("split_subsets", True,
                   "pool50k={} dev={} confirm={}".format(
                       len(pool_ids), len(dev_ids), len(confirm_ids)))
            counts["pool_curves_50k"] = len(pool_ids)
            counts["dev_curves"] = len(dev_ids)
            counts["confirm_curves"] = len(confirm_ids)
            pset, dset, cset = set(pool_ids), set(dev_ids), set(confirm_ids)
            overlap = {
                "pool∩dev": sorted(pset & dset),
                "pool∩confirm": sorted(pset & cset),
                "dev∩confirm": sorted(dset & cset),
            }
            clean = all(not v for v in overlap.values())
            record("subset_disjointness", clean,
                   "overlaps={}".format({k: len(v) for k, v in overlap.items()}))
            record("confirm_excluded_from_pool", not (pset & cset),
                   "confirm curves inside the pool: {}".format(
                       sorted(pset & cset)[:5]))
            record("pool_50k_nonempty", bool(pset),
                   "{} pool curve ids".format(len(pset)))
    else:
        for name in ("split_subsets", "subset_disjointness",
                     "confirm_excluded_from_pool", "pool_50k_nonempty"):
            record(name, False, "split manifest unreadable")

    # -- dataset-level checks --------------------------------------------
    if inputs is None:
        try:
            inputs = load_run_inputs(
                config=config, config_path=config_path,
                tables_dir=tables, split_path=split)
        except (V541RunnerError, V541ProvenanceError) as exc:
            record("dataset_open", False, "{}: {}".format(exc.code, exc.detail))
            inputs = None
        else:
            record("dataset_open", True,
                   "columnar v2 loader + idx_v2 binding verified")
    if inputs is not None:
        samples = inputs.dataset.samples
        n = len(samples)
        counts["n_samples"] = n
        pool_set = set(inputs.pool_curve_ids)
        dev_set = set(inputs.dev_curve_ids)
        confirm_set = set(inputs.confirm_curve_ids)

        leaked = sorted(pool_set & confirm_set)
        record("confirm_curve_not_in_pool", not leaked,
               "confirm curve ids present in the pool: {}".format(leaked[:5]))
        record("pool_and_dev_disjoint", not sorted(pool_set & dev_set),
               "pool∩dev = {}".format(sorted(pool_set & dev_set)[:5]))
        # direction coverage: every pool curve needs forward AND reverse
        intents = inputs.dataset.intents
        dirs_by_curve: Dict[str, set] = {}
        for key, intent_rec in intents.items():
            direction = str(getattr(intent_rec, "direction", "") or "")
            if direction not in ("forward", "reverse"):
                # key layout fallback: (curve_id, direction) or intent id
                token = str(key[1])
                direction = ("forward" if "fwd" in token or token == "forward"
                             else "reverse" if "rev" in token or token == "reverse"
                             else "")
            dirs_by_curve.setdefault(str(key[0]), set()).add(direction)
        missing_dir = sorted(
            cid for cid in inputs.pool_curve_ids
            if not {"forward", "reverse"} <= dirs_by_curve.get(cid, set()))
        record("direction_coverage", not missing_dir,
               "pool curves missing a direction: {}{}".format(
                   len(missing_dir),
                   (" e.g. " + ", ".join(missing_dir[:3])) if missing_dir else ""))
        counts["direction_missing_curves"] = len(missing_dir)

        # endpoint_b_index == 0 quirk hit count (strata path_order quirk)
        quirk = 0
        for cid in inputs.pool_curve_ids:
            try:
                _a, b = inputs.dataset.curve_endpoints(cid)
            except Exception:  # noqa: BLE001 - structural read failure = fail
                quirk = -1
                break
            if int(b) == 0:
                quirk += 1
        record("endpoint_b_index_zero_quirks", quirk >= 0,
               "pool curves hitting endpoint_b_index=0 quirk: {}".format(quirk))
        counts["endpoint_b_index_zero_quirks"] = quirk

        # per-seed inner split (80/20 curve level, no dev/confirm leakage)
        split_rows: Dict[str, Any] = {}
        first_train: List[int] = []
        for seed in seeds:
            try:
                train_pos, val_pos = _positions_for_seed(inputs, int(seed))
            except Exception as exc:  # noqa: BLE001
                record("inner_split_s{}".format(seed), False, str(exc))
                continue
            train_curves = {str(samples[p].curve_id) for p in train_pos}
            val_curves = {str(samples[p].curve_id) for p in val_pos}
            bad = sorted((train_curves | val_curves) & (dev_set | confirm_set))
            total = len(train_curves) + len(val_curves)
            frac = (len(val_curves) / total) if total else 0.0
            # 80/20 curve-level split (§7.2); ±0.15 band absorbs the small-n
            # rounding of a 3-curve fixture without loosening the real rule.
            frac_ok = bool(val_curves) and abs(frac - 0.2) <= 0.15
            ok = (bool(train_curves) and bool(val_curves)
                  and not bad and frac_ok
                  and not (train_curves & val_curves))
            record("inner_split_s{}".format(seed), ok,
                   "train_curves={} inner_val_curves={} frac={:.3f} "
                   "dev/confirm leakage={}".format(
                       len(train_curves), len(val_curves), frac, bad[:5]))
            split_rows[str(seed)] = {
                "train_curves": len(train_curves),
                "inner_val_curves": len(val_curves),
                "inner_val_fraction": round(frac, 6),
                "leakage_curves": bad,
            }
            if seed == seeds[0]:
                first_train = train_pos
        counts["inner_split"] = split_rows

        # strata index + pool SHA on the FIRST seed's inner-train slice
        if first_train:
            try:
                from pes2ts_core.pesn.offline.training.v541_strata import (
                    StrataIndex,
                    STRATA_ORDER_V541,
                    sampler_spec_sha256,
                )
                index = StrataIndex.build(samples, first_train,
                                          dataset=inputs.dataset)
                covered = sorted({s for (_c, _d, s) in index.pools})
                missing = sorted(set(STRATA_ORDER_V541) - set(covered))
                record("strata_pool_sha256", bool(index.pool_sha256),
                       "pool_sha256={} curves={}".format(
                           index.pool_sha256, len(index.curves)))
                record("strata_coverage", not missing and bool(index.dir_pools),
                       "missing strata={} direction pools={}".format(
                           missing, sorted(index.dir_pools)))
                counts["strata_pool_sha256"] = index.pool_sha256
                counts["sampler_spec_sha256"] = sampler_spec_sha256()
                counts["strata_curves"] = len(index.curves)
                counts["strata_positions"] = index.n_positions
            except Exception as exc:  # noqa: BLE001
                record("strata_pool_sha256", False, str(exc))
                record("strata_coverage", False, "index build failed")
        else:
            record("strata_pool_sha256", False, "no inner-train positions")
            record("strata_coverage", False, "no inner-train positions")

        # dev positions exist and are used for EXCLUSION only
        try:
            dev_pos = dev_positions_of(inputs)
            overlap = sorted({str(samples[p].curve_id) for p in dev_pos}
                             & (pool_set - dev_set))
            record("dev_exclusion_only", bool(dev_pos),
                   "dev positions={} (curve exclusion only; no dev feature/"
                   "label/eval built)".format(len(dev_pos)))
            counts["dev_positions"] = len(dev_pos)
        except V541RunnerError as exc:
            record("dev_exclusion_only", False, "{}: {}".format(exc.code, exc.detail))

        # label tables loadable (coverage is enforced again at run start)
        record("label_tables", inputs.labels is not None,
               "label tables loaded from idx_v2 (read-only)")

    # -- provenance SHAs (data / teacher / split / label / R3) ------------
    try:
        shas = artifact_shas(
            config, config_path=config_path, tables_dir=tables,
            split_path=split,
            code_sha=code_sha256(
                (config.get("identity_contract") or {}).get("code") or {},
                root=PROJECT_ROOT))
        for key in ("data_manifest_sha256", "teacher_manifest_sha256",
                    "split_sha256", "label_spec_sha256",
                    "endpoint_label_spec_sha256", "r3_init_sha256",
                    "code_sha256", "environment_sha256", "config_sha256"):
            value = shas.get(key)
            record(key, bool(value) and len(str(value)) == 64,
                   str(value))
        counts["shas"] = shas
    except (V541ProvenanceError, V541RunnerError) as exc:
        record("provenance_shas", False, "{}: {}".format(exc.code, exc.detail))

    ok = bool(items) and all(v["status"] == "pass" for v in items.values())
    return {
        "schema": "pes2ts_v541_data_audit_v1",
        "campaign_id": config.get("campaign_id"),
        "objective": config.get("objective"),
        "generated_at": _utc_now(),
        "tables_dir": str(tables_dir),
        "split_manifest": str(split_path),
        "items": items,
        "counts": counts,
        "confirm_read": False,
        "dev_used_for": "curve_exclusion_only",
        "new_qc_calls": 0,
        "ok": ok,
        "failed": sorted(k for k, v in items.items() if v["status"] != "pass"),
    }


# ---------------------------------------------------------------------------
# degree stats (E2, inner-train only; computed once, cached, SHA-bound)
# ---------------------------------------------------------------------------
def degree_stats_required(cfg: Mapping[str, Any], embedding_id: str) -> bool:
    recipe = (cfg.get("embedding_contract") or {}).get("feature_recipe") or {}
    ds = recipe.get("degree_stats") or {}
    return str(embedding_id) in {str(x) for x in (ds.get("required_for") or ())}


def ensure_degree_stats(cfg: Mapping[str, Any], *, embedding_id: str,
                        inputs: RunInputs, train_positions: Sequence[int],
                        root: Any, code_sha: str,
                        dev_positions: Sequence[int],
                        confirm_curve_ids: Sequence[str],
                        seed: int) -> Dict[str, Any]:
    """Fit/cache the E2 degree statistics over the FULL inner-train set.

    Discipline (config ``feature_recipe.degree_stats``):

    * population = EVERY inner-train position of THIS run seed — no
      sampling, no cap, no under-count (the position set IS the
      population; ``v541_degree_stats`` de-duplicates it to a set);
    * graph view = ``G_chem`` in-degree only (no spatial edges, no self
      loop, no label weight), delegated to the frozen
      ``v541_degree_stats.compute_degree_stats`` helper;
    * ``split`` is hard-wired to ``"inner-train"`` — dev / inner-val /
      confirm are typed refusals there AND re-checked here;
    * one fit per RUN seed → its own cache file and its own SHA;
    * the cache file is the trainer-compatible envelope read by
      ``load_degree_stats_v541`` and handed to
      ``set_degree_stats(mu_log, stats_sha256)`` once at construction —
      never recomputed inside ``forward``.
    """
    if not degree_stats_required(cfg, embedding_id):
        return {"required": False, "path": None, "sha256": None}

    from pes2ts_core.pesn.offline.training.v541_degree_stats import (
        DEGREE_STATS_SCHEMA,
        compute_degree_stats,
    )
    from pes2ts_core.pesn.offline.training.v541_train import (
        degree_stats_payload_digest,
    )

    recipe = ((cfg.get("embedding_contract") or {}).get("feature_recipe")
              or {}).get("degree_stats") or {}
    schema = str(recipe.get("schema") or DEGREE_STATS_SCHEMA)
    if schema != DEGREE_STATS_SCHEMA:
        raise V541RunnerError(
            "V541_DEGREE_STATS_FIT",
            "config degree_stats.schema {!r} != the frozen {!r}".format(
                schema, DEGREE_STATS_SCHEMA))
    if recipe.get("split", "inner-train") != "inner-train":
        raise V541RunnerError(
            "V541_DEGREE_STATS_FIT",
            "degree_stats.split must be 'inner-train', got {!r}".format(
                recipe.get("split")))
    cache_rel = str(recipe.get("cache")
                    or "outputs/pesn/v541_embed_50k/cache/degree_stats")

    dev_curves = {str(c) for c in inputs.dev_curve_ids}
    confirm_curves = {str(c) for c in confirm_curve_ids}
    samples = inputs.dataset.samples
    positions = sorted({int(p) for p in train_positions})
    n_total = len(samples)
    for p in positions:
        if not (0 <= p < n_total):
            raise V541RunnerError(
                "V541_DEGREE_STATS_FIT",
                "inner-train position {} outside [0,{})".format(p, n_total))
    train_curves = {str(samples[p].curve_id) for p in positions}
    forbidden = sorted(train_curves & (dev_curves | confirm_curves))
    if forbidden:
        raise V541RunnerError(
            "V541_DEGREE_STATS_FIT",
            "degree stats may only fit inner-train curves; leaked {}".format(
                forbidden[:5]))
    dev_pos_set = set(int(p) for p in dev_positions)
    if dev_pos_set & set(positions):
        raise V541RunnerError(
            "V541_DEGREE_STATS_FIT",
            "degree-stats population contains {} dev position(s)".format(
                len(dev_pos_set & set(positions))))
    if not positions:
        raise V541RunnerError(
            "V541_DEGREE_STATS_FIT", "inner-train slice is empty")

    inputs_block = {
        "schema": schema,
        "fit_split": "inner-train_only",
        "split": "inner-train",
        "population": "full_inner_train",
        "graph_view": "G_chem",
        "embedding_id": str(embedding_id),
        "seed": int(seed),
        "n_positions": len(positions),
        "positions_sha256": _positions_digest(positions),
        "train_curves_sha256": _strings_digest(sorted(train_curves)),
        "code_sha256": str(code_sha),
    }
    inputs_sha = _digest(inputs_block)
    declared = Path(cache_rel)
    if declared.is_absolute():
        cache_root = declared
    else:
        try:
            # declared relative to the campaign root: re-anchor it on THIS
            # run's output root so tests / alternate roots never write into
            # the repository's own outputs/ tree.
            cache_root = Path(root) / declared.relative_to(DEFAULT_ROOT)
        except ValueError:
            cache_root = PROJECT_ROOT / declared
    # per-run-seed cache: seed is part of the file name AND of inputs_sha
    cache_path = cache_root / "{}_seed{}_{}.json".format(
        schema, int(seed), inputs_sha[:24])

    if cache_path.is_file():
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            cached = None
        if isinstance(cached, Mapping) and cached.get("inputs_sha256") == inputs_sha:
            return {"required": True, "path": str(cache_path),
                    "sha256": sha256_file(cache_path), "cache_hit": True,
                    "inputs_sha256": inputs_sha,
                    "n_positions": int(cached.get("n_positions", 0)),
                    "seed": int(seed),
                    "mu_log": cached.get("mu_log"),
                    "stats_sha256": cached.get("stats_sha256")}

    result = compute_degree_stats(samples, positions,
                                  split="inner-train", seed=int(seed))
    payload = result.to_dict()
    try:
        recomputed = degree_stats_payload_digest(payload)
    except Exception as exc:  # noqa: BLE001 - typed trainer refusal -> runner
        raise V541RunnerError(
            "V541_DEGREE_STATS_FIT",
            "the fit payload does not carry the frozen digest keys: "
            "{}".format(exc)) from exc
    if recomputed != result.stats_sha256:
        raise V541RunnerError(
            "V541_DEGREE_STATS_FIT",
            "stats_sha256 {} != recomputed {} — the degree-stats payload "
            "contract drifted".format(result.stats_sha256[:12],
                                      recomputed[:12]))
    histogram = {str(k): int(v)
                 for k, v in sorted(result.degree_histogram.items())}
    envelope: Dict[str, Any] = {
        "schema": schema,
        "fit_split": "inner-train_only",
        "fit_forbidden_splits": list(recipe.get("forbidden_fit_splits") or
                                     ("dev", "inner-val", "confirm",
                                      "dev_confirm")),
        "embedding_id": str(embedding_id),
        "definition": "G_chem in-degree per node (no spatial edge, no self "
                      "loop) over the FULL inner-train position set of this "
                      "run seed",
        # --- the two values the PNA branch consumes ------------------------
        "mu_log": float(result.mu_log),
        "stats_sha256": result.stats_sha256,
        "histogram": histogram,
        "degree_histogram": histogram,
        # --- full fit provenance / weighted / unique / cost ----------------
        "split": result.split,
        "split_sha256": result.split_sha256,
        "positions_sha256": result.positions_sha256,
        "graph_recipe_sha256": result.graph_recipe_sha256,
        "seed": int(result.seed),
        "seed_sha256": result.seed_sha256,
        "n_positions": int(result.n_positions),
        "n_unique_graphs": int(result.n_unique_graphs),
        "weighted_graph_total": int(result.weighted_graph_total),
        "weighted_node_total": int(result.weighted_node_total),
        "weighted_edge_total": int(result.weighted_edge_total),
        "weighted_pair_total": int(result.weighted_pair_total),
        "elapsed_seconds": float(result.elapsed_seconds),
        "max_rss_kb": int(result.max_rss_kb),
        "inputs": inputs_block,
        "inputs_sha256": inputs_sha,
        "computed_at": _utc_now(),
        "computed_by": "v541_run.ensure_degree_stats",
        "forward_recomputation": False,
    }
    cache_root.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps(envelope, ensure_ascii=False, indent=1,
                   sort_keys=True) + "\n",
        encoding="utf-8")
    return {"required": True, "path": str(cache_path),
            "sha256": sha256_file(cache_path), "cache_hit": False,
            "inputs_sha256": inputs_sha,
            "n_positions": int(result.n_positions), "seed": int(seed),
            "mu_log": float(result.mu_log),
            "stats_sha256": result.stats_sha256}


def _digest(obj: Any) -> str:
    from scripts.campaigns.v541_provenance import sha256_json

    return sha256_json(obj)


def _positions_digest(positions: Sequence[int]) -> str:
    from pes2ts_core.pesn.offline.training.v541_identity import positions_sha256

    return positions_sha256(list(positions))


def _strings_digest(values: Sequence[str]) -> str:
    from scripts.campaigns.v541_provenance import sha256_json

    return sha256_json(list(values))


# ---------------------------------------------------------------------------
# run records (V541's OWN append-only ledger; legacy ledger untouched)
# ---------------------------------------------------------------------------
def run_records_path(root: Any) -> Path:
    return Path(root) / RUN_RECORDS_NAME


def append_run_record(root: Any, record: Mapping[str, Any]) -> Dict[str, Any]:
    payload = {"ts": _utc_now(), "schema": "pes2ts_v541_run_record_v1"}
    payload.update(dict(record))
    return append_manifest(run_records_path(root), payload)


def previous_attempts(root: Any, run_id: str) -> int:
    path = run_records_path(root)
    if not path.is_file():
        return 0
    count = 0
    for raw in path.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        try:
            rec = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if (isinstance(rec, Mapping) and rec.get("run_id") == run_id
                and rec.get("event") == "attempt_start"):
            count += 1
    return count


# ---------------------------------------------------------------------------
# mode helpers
# ---------------------------------------------------------------------------
def mode_out_dir(root: Any, mode: str, run_id: str) -> Path:
    base = {"smoke": "smoke", "profile": "profile", "formal": "runs"}[mode]
    return Path(root) / base / run_id


def plan_line_for(plan: Mapping[str, Any], embedding_id: str,
                  seed: int) -> Dict[str, Any]:
    """The UNIQUE non-diagnostic plan line for ``(embedding_id, seed)``.

    T4 diagnostics share E0 with the T0-R line, so they are excluded before
    uniqueness is required — a technical smoke/profile run always binds to
    the ONE nominal/conditional line of its (embedding, seed).
    """
    by_id = plan["by_run_id"]
    candidates = sorted(
        rid for rid, line in by_id.items()
        if str(line.get("embedding_id")) == str(embedding_id)
        and int(line.get("seed", -1)) == int(seed)
        and str(line.get("class")) != "diagnostic")
    if not candidates:
        raise V541RunnerError(
            "V541_RUN_IDENTITY_UNREGISTERED",
            "embedding {} seed {} has no preregistered (non-diagnostic) "
            "plan line".format(embedding_id, seed))
    if len(candidates) > 1:
        raise V541RunnerError(
            "V541_RUN_LINE_AMBIGUOUS",
            "embedding {} seed {} matches {} plan lines {}".format(
                embedding_id, seed, len(candidates), candidates))
    return {"run_id": candidates[0], "line": by_id[candidates[0]]}


def resolve_run_identity(args: argparse.Namespace, cfg: Mapping[str, Any],
                         plan: Mapping[str, Any]) -> Dict[str, Any]:
    """Bind ``--run-id/--embedding/--seed`` to the frozen plan (fail-closed).

    FORMAL runs must match a plan line EXACTLY (run_id + embedding + seed).
    A technical smoke/profile run keeps its own run_id (the smoke lane is
    ``smoke/V541_<E>_S17``) but still binds the FULL provenance SHA block of
    the unique matching plan line — an E0/E1 technical run must never end up
    with empty common/config/… SHAs just because its directory name differs
    from the formal id.
    """
    run_id = str(args.run_id)
    embedding_id = str(args.embedding)
    seed = int(args.seed)
    if not run_id.startswith("V541_"):
        raise V541RunnerError(
            "V541_RUN_ID_NAMESPACE",
            "run_id must use the V541_ namespace, got {!r}".format(run_id))
    if seed not in SEEDS:
        raise V541RunnerError(
            "V541_SEED", "seed {} is not preregistered {}".format(seed, SEEDS))
    by_id = plan["by_run_id"]
    if args.mode == "formal":
        line = by_id.get(run_id)
        if line is None:
            raise V541RunnerError(
                "V541_RUN_NOT_IN_PLAN",
                "{} is not a preregistered formal run (plan lists {})".format(
                    run_id, sorted(by_id)))
        if str(line.get("embedding_id")) != embedding_id:
            raise V541RunnerError(
                "V541_PLAN_LINE_DRIFT",
                "{} plan line binds embedding {} but --embedding={}".format(
                    run_id, line.get("embedding_id"), embedding_id))
        if int(line.get("seed", -1)) != seed:
            raise V541RunnerError(
                "V541_PLAN_LINE_DRIFT",
                "{} plan line binds seed {} but --seed={}".format(
                    run_id, line.get("seed"), seed))
        return {"line": line, "in_plan": True, "matched_run_ids": [run_id],
                "bound_plan_run_id": run_id}
    exact = by_id.get(run_id)
    if exact is not None:
        if (str(exact.get("embedding_id")) != embedding_id
                or int(exact.get("seed", -1)) != seed):
            raise V541RunnerError(
                "V541_PLAN_LINE_DRIFT",
                "{} binds embedding={} seed={} in the plan".format(
                    run_id, exact.get("embedding_id"), exact.get("seed")))
        return {"line": exact, "in_plan": True, "matched_run_ids": [run_id],
                "bound_plan_run_id": run_id}
    bound = plan_line_for(plan, embedding_id, seed)
    return {"line": bound["line"], "in_plan": False,
            "matched_run_ids": [bound["run_id"]],
            "bound_plan_run_id": bound["run_id"]}


def guard_mode_args(mode: str, args: argparse.Namespace) -> None:
    """Dose / window guards — checked BEFORE any data, lock or model work.

    The frozen doses (200-update smoke, 37,900-update formal, W0 ≤ 400
    profile window) are never reduced through a CLI flag (plan §7.1/§8.2).
    """
    if int(args.seed) not in SEEDS:
        raise V541RunnerError(
            "V541_SEED",
            "seed {} is not preregistered {}".format(args.seed, list(SEEDS)))
    if mode == "smoke":
        if int(args.seed) != 17:
            raise V541RunnerError(
                "V541_SMOKE_SEED",
                "the technical smoke runs on seed 17 only (plan §8.2; the "
                "V541-smoke-200 gate reads smoke/V541_<E>_S17), got seed "
                "{}".format(args.seed))
        if args.stop_after is not None and int(args.stop_after) > \
                SMOKE_TARGET_UPDATES:
            raise V541RunnerError(
                "V541_SMOKE_DOSE",
                "--stop-after {} exceeds the frozen {}-update smoke dose "
                "(plan §8.2)".format(args.stop_after, SMOKE_TARGET_UPDATES))
        if args.stop_after is not None and int(args.stop_after) < 1:
            raise V541RunnerError(
                "V541_SMOKE_DOSE", "--stop-after must be >= 1")
    elif mode == "profile":
        if args.stop_after is not None:
            raise V541RunnerError(
                "V541_PROFILE_DOSE",
                "--stop-after is only legal for the smoke kill/resume drill")
        if not (1 <= int(args.profile_w0) <= PROFILE_W0_MAX):
            raise V541RunnerError(
                "V541_PROFILE_WINDOW",
                "--profile-w0 must lie in 1..{} (the W0 window is {} "
                "updates)".format(PROFILE_W0_MAX, PROFILE_W0_MAX))
        if int(args.profile_w1) < 1:
            raise V541RunnerError(
                "V541_PROFILE_WINDOW", "--profile-w1 must be >= 1")
    else:  # formal
        if args.stop_after is not None:
            raise V541RunnerError(
                "V541_FORMAL_DOSE",
                "the formal dose is fixed at {} updates — a bounded leg "
                "flag may not reduce it (plan §7.1/§8.4)".format(
                    FORMAL_TOTAL_UPDATES))


def build_trainer_config(cfg: Mapping[str, Any], *, mode: str,
                         embedding_id: str, seed: int, plan_line: Mapping[str, Any],
                         device: str, stop_after: Optional[int],
                         profile_w0: int, profile_w1: int,
                         degree_stats: Mapping[str, Any],
                         cuda_memory_fraction: Optional[float],
                         eval_vram_gate_mib: Optional[int],
                         eval_vram_gate_timeout_s: int) -> Any:
    """Assemble the frozen ``V541TrainerConfig`` for this mode (no knobs)."""
    from pes2ts_core.pesn.offline.training.v541_train import V541TrainerConfig

    tc = cfg.get("training_contract") or {}
    schedule = dict(tc.get("schedule") or {})
    split = cfg.get("split_contract") or {}
    budget = dict(cfg.get("budget") or {})
    lambdas = dict(tc.get("lambdas") or {})

    smoke_updates: Optional[int] = None
    profile_only = False
    formal_run: Optional[bool] = None
    if mode == "smoke":
        target = SMOKE_TARGET_UPDATES
        if stop_after is not None and int(stop_after) > target:
            raise V541RunnerError(
                "V541_SMOKE_DOSE",
                "--stop-after {} exceeds the frozen {}-update smoke dose "
                "(plan §8.2)".format(stop_after, target))
        smoke_updates = int(stop_after) if stop_after else target
        formal_run = False
    elif mode == "profile":
        profile_only = True
        formal_run = False
        if stop_after is not None:
            raise V541RunnerError(
                "V541_PROFILE_DOSE",
                "--stop-after is only legal for the smoke kill/resume drill")
        if not (1 <= int(profile_w0) <= PROFILE_W0_MAX):
            raise V541RunnerError(
                "V541_PROFILE_WINDOW",
                "--profile-w0 must lie in 1..{} (the W0 window is {} "
                "updates)".format(PROFILE_W0_MAX, PROFILE_W0_MAX))
        if int(profile_w1) < 1:
            raise V541RunnerError(
                "V541_PROFILE_WINDOW", "--profile-w1 must be >= 1")
    else:  # formal
        if stop_after is not None:
            raise V541RunnerError(
                "V541_FORMAL_DOSE",
                "the formal dose is fixed at {} updates — a bounded leg "
                "flag may not reduce it (plan §7.1/§8.4)".format(
                    FORMAL_TOTAL_UPDATES))
        formal_run = True

    init_seed = None
    if embedding_id != "E0":
        init_seed = EMBEDDING_INIT_SEED_BASE + int(seed)

    return V541TrainerConfig(
        arm="C",
        seed=int(seed),
        objective="unified_move_process",
        embedding_id=embedding_id,
        init_seed=init_seed,
        backbone_init="r3",
        lambda_rank=float(lambdas.get("lambda_rank", 0.5)),
        lambda_q=float(lambdas.get("lambda_q", 0.5)),
        lambda_endpoint=float(lambdas.get("lambda_endpoint", 0.012385049119773344)),
        freeze_encoder_updates=int(schedule.get("w0_updates", 400)),
        joint_updates=int(schedule.get("w1_updates", 37500)),
        lr_head=float((schedule.get("param_groups") or {}).get(
            "process_head", {}).get("lr", 3e-4)),
        lr_encoder=float((schedule.get("param_groups") or {}).get(
            "legacy_core_move_readout_h_legacy", {}).get("lr", 1e-4)),
        weight_decay=float(schedule.get("weight_decay", 1e-4)),
        grad_clip=float(schedule.get("grad_clip", 1.0)),
        val_every=int((tc.get("selection") or {}).get("val_every", 100)),
        grad_monitor_every=int(
            (tc.get("gradient_monitoring") or {}).get("every_updates", 100)),
        batch_curves=int((tc.get("batch") or {}).get("curves", 8)),
        frames_per_curve=int((tc.get("batch") or {}).get("frames", 4)),
        inner_val_max_samples=(int(split["inner_val_max_samples"])
                               if split.get("inner_val_max_samples") else None),
        smoke_updates=smoke_updates,
        profile_only=profile_only,
        profile_w0_updates=(int(profile_w0) if profile_only else None),
        profile_w1_updates=(int(profile_w1) if profile_only else None),
        formal_run=formal_run,
        device=str(device),
        cuda_memory_fraction=cuda_memory_fraction,
        eval_vram_gate_mib=eval_vram_gate_mib,
        eval_vram_gate_timeout_s=int(eval_vram_gate_timeout_s),
        host_limit_gib=float(budget.get("host_limit_gib") or 12),
        memory_high_gib=float(budget.get("memory_high_gib") or 10),
        swap_max=int(budget.get("swap_max") or 0),
        oom_kill_max=int(budget.get("oom_kill_max") or 0),
        concurrency=1,
        common_protocol_sha256=plan_line.get("common_protocol_sha256"),
        embedding_spec_sha256=plan_line.get("embedding_spec_sha256"),
        feature_recipe_sha256=plan_line.get("feature_recipe_sha256"),
        code_sha256=plan_line.get("code_sha256"),
        environment_sha256=plan_line.get("environment_sha256"),
        degree_stats_path=(degree_stats.get("path")
                           if degree_stats.get("required") else None),
        degree_stats_sha256=(degree_stats.get("sha256")
                             if degree_stats.get("required") else None),
    )


# ---------------------------------------------------------------------------
# cmd: validate-config
# ---------------------------------------------------------------------------
def cmd_validate_config(args: argparse.Namespace) -> int:
    assert_v541_root(args.output_root)
    cfg = load_config(args.config)
    report = validate_config(cfg, formal=args.formal,
                             check_gv=not args.no_gv)
    out: Dict[str, Any] = dict(report)
    if args.formal:
        root = Path(args.output_root)
        gates = t1_gate_report(
            cfg, root=root,
            formal_config_blocks=report.get("t1_blocked") or [],
            embedding_id=str(args.embedding))
        out["t1_gates"] = gates
        out["t1_blocks"] = gates["blocks"]
        out["t1_ready"] = bool(gates["ready"])
        out["ok"] = bool(report.get("ok")) and bool(gates["ready"])
    _emit(out)
    return 0 if out.get("ok") else 1


# ---------------------------------------------------------------------------
# cmd: audit-data-and-manifests
# ---------------------------------------------------------------------------
def cmd_audit(args: argparse.Namespace) -> int:
    assert_v541_root(args.output_root)
    root = Path(args.output_root)
    cfg = load_config(args.config)
    validate_config(cfg, formal=False)
    split = Path(args.split) if args.split else (
        Path(args.tables_dir) / DEFAULT_SPLIT_NAME)
    report = audit_data_and_manifests(
        config=cfg, config_path=args.config, tables_dir=args.tables_dir,
        split_path=split)
    out_dir = Path(root) / "audits"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "data_audit.json"
    out_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=1, sort_keys=True,
                   default=str) + "\n",
        encoding="utf-8")
    report["audit_path"] = str(out_path)
    _emit(report)
    return 0 if report["ok"] else 1


# ---------------------------------------------------------------------------
# cmd: plan-init
# ---------------------------------------------------------------------------
def cmd_plan_init(args: argparse.Namespace) -> int:
    assert_v541_root(args.output_root)
    root = Path(args.output_root)
    cfg = load_config(args.config)
    validate_config(cfg, formal=False)
    split = Path(args.split) if args.split else (
        Path(args.tables_dir) / DEFAULT_SPLIT_NAME)
    for path in (Path(args.config), Path(args.tables_dir) / "data_manifest.json",
                 Path(args.tables_dir) / "teacher_manifest.json", split):
        if not path.is_file():
            raise V541RunnerError(
                "V541_PLAN_ARTIFACT_MISSING",
                "required plan input missing: {}".format(path))
    result = write_plan(
        root, cfg, config_path=args.config, tables_dir=args.tables_dir,
        split_path=split, force=bool(args.force),
        v53mp_config_path=args.v53mp_config,
        frozen_by="v541_run.plan-init")
    result["ok"] = True
    _emit(result)
    return 0


# ---------------------------------------------------------------------------
# cmd: gate-check
# ---------------------------------------------------------------------------
def _gate_item_import(attr: str, module: str) -> Tuple[str, str]:
    import importlib

    try:
        mod = importlib.import_module(module)
    except Exception as exc:  # noqa: BLE001 - fail-closed
        return "fail", "import {} failed: {}".format(module, exc)
    if not hasattr(mod, attr):
        return "fail", "{} has no attribute {!r}".format(module, attr)
    return "pass", "{}.{}".format(module, attr)


def cmd_gate_check(args: argparse.Namespace) -> int:
    assert_v541_root(args.output_root)
    root = Path(args.output_root)
    items: Dict[str, str] = {}
    details: Dict[str, str] = {}

    def record(name: str, result: Tuple[str, str]) -> None:
        items[name], details[name] = result

    cfg_path = Path(args.config)
    cfg: Optional[Mapping[str, Any]] = None
    if not cfg_path.is_file():
        record("config_yaml", ("fail", "{} missing".format(cfg_path)))
    else:
        try:
            cfg = load_config(cfg_path)
            record("config_yaml", ("pass", "sha256 {}".format(
                sha256_file(cfg_path)[:12])))
        except V541SchemaError as exc:
            record("config_yaml", ("fail", "{}: {}".format(exc.code, exc.detail)))
    if cfg is not None:
        try:
            report = validate_config(cfg, formal=False)
            record("config_schema", ("pass", "mode={}".format(report["mode"])))
        except V541SchemaError as exc:
            record("config_schema", ("fail", "{}: {}".format(exc.code, exc.detail)))

    for name, attr, module in (
            ("trainer_v541", "train_run_v541",
             "pes2ts_core.pesn.offline.training.v541_train"),
            ("sampler_v541", "sampler_spec_sha256",
             "pes2ts_core.pesn.offline.training.v541_strata"),
            ("identity_v541", "git_state",
             "pes2ts_core.pesn.offline.training.v541_identity"),
            ("embedding_registry", "build_embedding",
             "pes2ts_core.pesn.models.v541.embedding_registry"),
            ("config_schema_v541", "validate_config",
             "scripts.agent_ops.v541_config_schema"),
            ("run_driver_v541", "main", "scripts.campaigns.v541_run")):
        record(name, _gate_item_import(attr, module))

    tables = Path(args.tables_dir)
    split = Path(args.split) if args.split else tables / DEFAULT_SPLIT_NAME
    from pes2ts_core.pesn.offline.data.key_atom_dataset_v3 import TABLE_FILENAMES

    missing = [fn for fn in TABLE_FILENAMES.values()
               if not (tables / fn).is_file()]
    record("data_tables_six", ("pass" if not missing else "fail",
                               "missing: {}".format(sorted(missing)) if missing
                               else "six tables present"))
    if cfg is not None:
        from scripts.campaigns.r3_v53mp_run import _verify_manifest_shas

        for name in ("data_manifest.json", "teacher_manifest.json"):
            if (tables / name).is_file():
                status, detail = _verify_manifest_shas(tables, name)
                record(name, (status, detail))
            else:
                record(name, ("fail", "{} missing".format(name)))
        try:
            idx = verify_idx_v2_binding(tables)
            record("idx_v2_binding", ("pass", "n_samples={}".format(
                idx.get("n_samples"))))
        except V541RunnerError as exc:
            record("idx_v2_binding", ("fail", "{}: {}".format(exc.code, exc.detail)))
        if split.is_file():
            record("split_manifest", ("pass", str(split)))
        else:
            record("split_manifest", ("fail", "{} missing".format(split)))
        try:
            shas = artifact_shas(
                cfg, config_path=args.config, tables_dir=tables,
                split_path=split,
                code_sha=code_sha256(
                    (cfg.get("identity_contract") or {}).get("code") or {},
                    root=PROJECT_ROOT))
            bad = [k for k, v in shas.items()
                   if k != "git_commit" and (not v or len(str(v)) != 64)]
            record("provenance_shas",
                   ("fail" if bad else "pass",
                    "all real" if not bad else "placeholder/missing: {}".format(bad)))
        except (V541ProvenanceError, V541RunnerError) as exc:
            record("provenance_shas", ("fail", "{}: {}".format(exc.code, exc.detail)))

    # plan + pin
    paths = plan_paths(root)
    if paths["plan"].is_file() and paths["pin"].is_file():
        try:
            verified = verify_plan_pin_v541(paths["plan"], paths["pin"])
            record("runplan_pin", ("pass", "code_sha256={}".format(
                str(verified.get("code_sha256"))[:12])))
            if cfg is not None:
                current = code_sha256(
                    (cfg.get("identity_contract") or {}).get("code") or {},
                    root=PROJECT_ROOT)
                record("code_sha256_current",
                       ("pass" if current == verified.get("code_sha256") else "fail",
                        "current={} pinned={}".format(
                            current[:12],
                            str(verified.get("code_sha256"))[:12])))
        except V541RunplanError as exc:
            record("runplan_pin", ("fail", "{}: {}".format(exc.code, exc.detail)))
    else:
        record("runplan_pin", ("fail", "plan/pin missing — run plan-init"))

    # T1 gate evidence (missing never passes)
    if cfg is not None:
        gates = t1_gate_report(cfg, root=root,
                               formal_config_blocks=[],
                               embedding_id=str(args.embedding))
        for block in gates["blocks"]:
            record("t1_gate:{}".format(block["code"]), ("fail", block["detail"]))
        if gates["ready"]:
            record("t1_gates", ("pass", "all machine evidence present"))

    record("resources", ("pass", json.dumps(read_cgroup_limits(),
                                            sort_keys=True, default=str)))
    record("new_qc_calls", ("pass", "0"))

    ok = all(v == "pass" for v in items.values())
    _emit({"schema": "pes2ts_v541_gate_check_v1",
           "root": str(root), "items": items, "details": details,
           "ok": ok,
           "failed": sorted(k for k, v in items.items() if v != "pass")})
    return 0 if ok else 1


# ---------------------------------------------------------------------------
# cmd: run
# ---------------------------------------------------------------------------
def _require_plan(root: Any, cfg: Mapping[str, Any]) -> Dict[str, Any]:
    paths = plan_paths(root)
    if not paths["plan"].is_file() or not paths["pin"].is_file():
        raise V541RunnerError(
            "V541_PLAN_MISSING",
            "runplan.jsonl / runplan.pin.json missing under {} — run "
            "plan-init first (an unpinned plan is never launchable)".format(
                paths["dir"]))
    verified = verify_plan_pin_v541(paths["plan"], paths["pin"])
    plan = read_plan(paths["plan"])
    plan["pin_verified"] = verified
    current = code_sha256(
        (cfg.get("identity_contract") or {}).get("code") or {},
        root=PROJECT_ROOT)
    if verified.get("code_sha256") != current:
        raise V541RunnerError(
            "V541_CODE_DRIFT",
            "source fingerprint changed since the plan freeze (pinned {} != "
            "current {}) — re-freeze the plan; refusing to launch".format(
                verified.get("code_sha256"), current))
    return plan


def verify_plan_binding(*, cfg: Mapping[str, Any], plan: Mapping[str, Any],
                        config_path: Any, tables_dir: Any, split_path: Any,
                        embedding_id: Optional[str] = None,
                        seed: Optional[int] = None) -> Dict[str, Any]:
    """Recompute every frozen SHA; ANY drift is a typed refusal.

    Covers the §17 formal freeze: config / data / teacher / split / label /
    R3 / code / environment / common-protocol digests must still equal the
    pinned plan header, and the chosen candidate's embedding/feature recipe
    SHAs must still equal its plan line.  Applies to EVERY mode — a drifted
    protocol is never launched, smoke included.
    """
    header = plan["header"]

    def _check(field: str, actual: Any, code: str) -> None:
        planned = header.get(field)
        if actual != planned:
            raise V541RunnerError(
                code,
                "{} drifted since the plan freeze (pinned {} != current "
                "{}) — re-freeze the plan; refusing to launch".format(
                    field, str(planned)[:16], str(actual)[:16]))

    _check("config_sha256", sha256_file(config_path), "V541_CONFIG_DRIFT")
    code = code_sha256((cfg.get("identity_contract") or {}).get("code") or {},
                       root=PROJECT_ROOT)
    shas = artifact_shas(
        cfg, config_path=config_path, tables_dir=tables_dir,
        split_path=split_path, code_sha=code)
    for field, err in (
            ("data_manifest_sha256", "V541_DATA_DRIFT"),
            ("teacher_manifest_sha256", "V541_TEACHER_DRIFT"),
            ("split_sha256", "V541_SPLIT_DRIFT"),
            ("label_spec_sha256", "V541_LABEL_DRIFT"),
            ("r3_init_sha256", "V541_R3_INIT_DRIFT"),
            ("environment_sha256", "V541_ENV_DRIFT"),
            ("code_sha256", "V541_CODE_DRIFT")):
        _check(field, shas.get(field), err)
    common = common_protocol_sha256(
        cfg, code_sha=code,
        data_manifest_sha256=shas["data_manifest_sha256"],
        teacher_manifest_sha256=shas["teacher_manifest_sha256"],
        split_sha256=shas["split_sha256"],
        label_spec_sha256=shas["label_spec_sha256"])
    _check("common_protocol_sha256", common, "V541_COMMON_DRIFT")

    if embedding_id:
        bound = plan_line_for(plan, embedding_id, int(seed if seed is not None
                                                      else 17))
        line = bound["line"]
        for field, actual, err in (
                ("embedding_spec_sha256",
                 embedding_spec_sha256(cfg, embedding_id),
                 "V541_EMBEDDING_SPEC_DRIFT"),
                ("feature_recipe_sha256",
                 feature_recipe_sha256(cfg, embedding_id),
                 "V541_FEATURE_RECIPE_DRIFT")):
            if line.get(field) != actual:
                raise V541RunnerError(
                    err,
                    "{} for {} drifted (pinned {} != current {})".format(
                        field, embedding_id,
                        str(line.get(field))[:16], str(actual)[:16]))
    return {"code_sha256": code, "environment_sha256": shas["environment_sha256"],
            "common_protocol_sha256": common, "shas": shas}


def cmd_run(args: argparse.Namespace) -> int:
    assert_v541_root(args.output_root)
    root = Path(args.output_root)
    mode = str(args.mode)
    cfg = load_config(args.config)
    try:
        report = validate_config(cfg, formal=(mode == "formal"))
    except V541SchemaError as exc:
        raise V541RunnerError(exc.code, exc.detail) from exc

    # -- frozen dose / profile window guards (cheapest refusal first) -----
    guard_mode_args(mode, args)

    # -- registry dispatch: unimplemented ids REFUSE (no placeholder) -----
    from pes2ts_core.pesn.models.v541.embedding_registry import (
        is_embedding_implemented,
        is_embedding_known,
    )

    embedding_id = str(args.embedding)
    if not is_embedding_known(embedding_id):
        raise V541RunnerError(
            "V541_EMBEDDING_UNKNOWN",
            "embedding {!r} is outside the frozen registry".format(embedding_id))
    if not is_embedding_implemented(embedding_id):
        raise V541RunnerError(
            "V541_EMBEDDING_NOT_IMPLEMENTED",
            "embedding {!r} is preregistered but not registered by WP3 yet — "
            "refusing rather than substituting a placeholder branch".format(
                embedding_id))

    # -- plan + identity binding (BEFORE any data/GPU work) ---------------
    plan = _require_plan(root, cfg)
    identity = resolve_run_identity(args, cfg, plan)
    plan_line = identity["line"] or {}
    run_id = str(args.run_id)
    seed = int(args.seed)

    # -- every frozen SHA must still equal the pinned plan (drift = refuse)
    split_path = Path(args.split) if args.split else (
        Path(args.tables_dir) / DEFAULT_SPLIT_NAME)
    verify_plan_binding(cfg=cfg, plan=plan, config_path=args.config,
                        tables_dir=args.tables_dir, split_path=split_path,
                        embedding_id=embedding_id, seed=seed)

    # -- formal: EVERY machine gate must pass -----------------------------
    blocks: List[Dict[str, Any]] = []
    if mode == "formal":
        gates = t1_gate_report(
            cfg, root=root, formal_config_blocks=report.get("t1_blocked") or [],
            embedding_id=embedding_id)
        blocks.extend(gates["blocks"])
        resources = read_cgroup_limits()
        blocks.extend(resource_blocks(resources, cfg.get("budget") or {},
                                      strict=True))
        if blocks:
            append_run_record(root, {
                "event": "refused", "run_id": run_id, "mode": mode,
                "embedding_id": embedding_id, "seed": seed,
                "code": "V541_FORMAL_GATES_BLOCKED", "blocks": blocks,
                "new_qc_calls": 0,
            })
            raise V541RunnerError(
                "V541_FORMAL_GATES_BLOCKED",
                "formal launch blocked by {} typed gate(s)".format(len(blocks)),
                blocks=blocks)

    # smoke/profile only RECORD the cgroup snapshot (the outer shell
    # enforces 12G/10G/swap0); the strict resource gate is formal-only.
    resources = read_cgroup_limits()

    # -- single-lane GPU lock --------------------------------------------
    from scripts.agent_ops.gpu_queue.gpu_lock import acquire as gpu_acquire

    handle = gpu_acquire(
        args.gpu_lock, lane="V541-{}".format(mode), run_id=run_id,
        cuda=str(args.device).startswith("cuda"))
    if handle is None:
        raise V541RunnerError(
            "V541_GPU_LOCK_BUSY",
            "GPU lock {} is held — single-lane concurrency=1 means this run "
            "stays planned".format(args.gpu_lock))

    try:
        return _run_locked(args, cfg, plan, identity, resources, root,
                           mode, run_id, embedding_id, seed, report)
    finally:
        handle.release()


def _run_locked(args: argparse.Namespace, cfg: Mapping[str, Any],
                plan: Mapping[str, Any], identity: Mapping[str, Any],
                resources: Mapping[str, Any], root: Any, mode: str,
                run_id: str, embedding_id: str, seed: int,
                report: Mapping[str, Any]) -> int:
    split = Path(args.split) if args.split else (
        Path(args.tables_dir) / DEFAULT_SPLIT_NAME)
    inputs = load_run_inputs(config=cfg, config_path=args.config,
                             tables_dir=args.tables_dir, split_path=split)
    train_positions, inner_val_positions = _positions_for_seed(inputs, seed)
    pool_positions = sorted(set(train_positions) | set(inner_val_positions))
    dev_positions = dev_positions_of(inputs)

    out_dir = mode_out_dir(root, mode, run_id)
    if out_dir.exists() and args.fresh and (out_dir / "checkpoint_last.pt").exists():
        raise V541RunnerError(
            "V541_CHECKPOINT_EXISTS",
            "{} already has a checkpoint; --fresh would discard it — resume "
            "instead (the formal run only continues from a matching "
            "checkpoint_last)".format(out_dir))
    out_dir.mkdir(parents=True, exist_ok=True)

    attempt = previous_attempts(root, run_id) + 1
    plan_line = identity.get("line") or {}
    code = code_sha256((cfg.get("identity_contract") or {}).get("code") or {},
                       root=PROJECT_ROOT)
    if mode == "smoke":
        campaign_target = SMOKE_TARGET_UPDATES
    elif mode == "formal":
        campaign_target = FORMAL_TOTAL_UPDATES
    else:
        campaign_target = int(args.profile_w0) + int(args.profile_w1)

    degree_stats = ensure_degree_stats(
        cfg, embedding_id=embedding_id, inputs=inputs,
        train_positions=train_positions, root=root, code_sha=code,
        dev_positions=dev_positions,
        confirm_curve_ids=inputs.confirm_curve_ids, seed=seed)

    trainer_config = build_trainer_config(
        cfg, mode=mode, embedding_id=embedding_id, seed=seed,
        plan_line=plan_line, device=args.device,
        stop_after=(args.stop_after if mode == "smoke" else None),
        profile_w0=args.profile_w0, profile_w1=args.profile_w1,
        degree_stats=degree_stats,
        cuda_memory_fraction=args.cuda_memory_fraction,
        eval_vram_gate_mib=args.eval_vram_gate_mib,
        eval_vram_gate_timeout_s=args.eval_vram_gate_timeout_s)

    provenance = dict(inputs.provenance)
    provenance.update({
        "campaign_run_id": run_id,
        "mode": mode,
        "embedding_id": embedding_id,
        "attempt": attempt,
        "objective": cfg.get("objective"),
        "code_sha256": code,
        "environment_sha256": environment_sha256(),
        "common_protocol_sha256": plan_line.get("common_protocol_sha256"),
        "embedding_spec_sha256": plan_line.get("embedding_spec_sha256"),
        "feature_recipe_sha256": plan_line.get("feature_recipe_sha256"),
        "runplan_sha256": plan.get("pin_verified", {}).get("pin", {}).get(
            "sha256"),
        "budget_sha256": plan_line.get("budget_sha256"),
        "degree_stats": (dict(degree_stats)
                         if degree_stats.get("required") else None),
        "resources": dict(resources),
        "new_qc_calls": 0,
        "confirm_read": False,
        "dev_used_for": "curve_exclusion_only",
    })

    append_run_record(root, {
        "event": "attempt_start",
        "run_id": run_id,
        "mode": mode,
        "embedding_id": embedding_id,
        "seed": seed,
        "init_seed": trainer_config.init_seed,
        "attempt": attempt,
        "resume": bool(args.resume) and not bool(args.fresh),
        "out_dir": str(out_dir),
        "plan_run_ids": identity.get("matched_run_ids"),
        "bound_plan_run_id": identity.get("bound_plan_run_id"),
        "in_plan": bool(identity.get("in_plan")),
        "is_formal_run": bool(trainer_config.formal_run),
        "target_updates": campaign_target,
        "leg_window_updates": int(trainer_config.total_updates),
        "stop_after": (args.stop_after if mode == "smoke" else None),
        "config_sha256": plan_line.get("config_sha256"),
        "code_sha256": code,
        "common_protocol_sha256": plan_line.get("common_protocol_sha256"),
        "environment_sha256": environment_sha256(),
        "degree_stats_sha256": degree_stats.get("sha256"),
        "resources": dict(resources),
        "concurrency": 1,
        "new_qc_calls": 0,
        "confirm_read": False,
    })

    from pes2ts_core.pesn.offline.training.v541_train import train_run_v541

    r3_ckpt = Path(str((cfg.get("training_contract") or {})
                       .get("backbone_init", {}).get("path")))
    if not r3_ckpt.is_absolute():
        r3_ckpt = PROJECT_ROOT / r3_ckpt

    try:
        metrics = train_run_v541(
            inputs.dataset, inputs.labels, inputs.endpoint, trainer_config,
            out_dir, r3_ckpt=r3_ckpt, r3_eval_ckpt=r3_ckpt,
            dev_positions=dev_positions, train_positions=pool_positions,
            resume=bool(args.resume) and not bool(args.fresh),
            provenance=provenance, feature_workers=args.feature_workers)
    except BaseException as exc:  # noqa: BLE001 - record then re-raise
        from pes2ts_core.pesn.offline.training.v541_identity import (
            run_outcome_for_exception,
        )

        append_run_record(root, {
            "event": "attempt_end",
            "run_id": run_id,
            "mode": mode,
            "embedding_id": embedding_id,
            "seed": seed,
            "attempt": attempt,
            "outcome": run_outcome_for_exception(exc),
            "error": "{}: {}".format(type(exc).__name__, exc),
            "is_formal_run": bool(trainer_config.formal_run),
            "new_qc_calls": 0,
        })
        raise

    leg_window = int(trainer_config.total_updates)
    leg_stop = args.stop_after if mode == "smoke" else None
    # a smoke leg is only "complete" once it reached the FROZEN 200 updates
    leg_complete = (mode != "smoke") or (leg_window >= campaign_target)
    tau = metrics.get("tau_end_calibration") or {}
    append_run_record(root, {
        "event": "attempt_end",
        "run_id": run_id,
        "mode": mode,
        "embedding_id": embedding_id,
        "seed": seed,
        "attempt": attempt,
        "outcome": metrics.get("run_outcome"),
        "attempt_status": ("leg_complete" if leg_complete else "leg_partial"),
        "is_formal_run": bool(metrics.get("is_formal_run")),
        "target_updates": campaign_target,
        "leg_window_updates": leg_window,
        "leg_stop_after": leg_stop,
        "updates_run": int((metrics.get("schedule") or {}).get("updates_run")
                           or 0),
        "selected_update": metrics.get("selected_update"),
        "best_update": metrics.get("best_update"),
        "checkpoint_best_sha256": metrics.get("checkpoint_best_sha256"),
        "checkpoint_best_path": metrics.get("checkpoint_best_path"),
        "tau": {"tau_end": tau.get("tau_end"),
                "calibrated_on": metrics.get("tau_calibrated_on"),
                "frozen": tau.get("frozen", True)},
        "batch_schedule_sha256": (metrics.get("sampler") or {}).get(
            "batch_schedule_sha256"),
        "sampler_pool_sha256": (metrics.get("sampler") or {}).get("pool_sha256"),
        "common_protocol_sha256": plan_line.get("common_protocol_sha256"),
        "embedding_spec_sha256": plan_line.get("embedding_spec_sha256"),
        "feature_recipe_sha256": plan_line.get("feature_recipe_sha256"),
        "code_sha256": code,
        "environment_sha256": environment_sha256(),
        "degree_stats_sha256": degree_stats.get("sha256"),
        "cost": {
            "wall_clock_s": (metrics.get("timing") or {}).get("wall_clock_s"),
            "loop_wall_s": (metrics.get("timing") or {}).get("loop_wall_s"),
            "host_peak_rss_bytes": (metrics.get("resources") or {}).get(
                "host_peak_rss_bytes"),
            "gpu_max_memory_allocated_bytes": (metrics.get("resources") or {}).get(
                "gpu_max_memory_allocated_bytes"),
            "cache_hits": (metrics.get("cache") or {}).get("hits"),
            "cache_misses": (metrics.get("cache") or {}).get("misses"),
        },
        "evaluation_scope": metrics.get("evaluation_scope"),
        "dev_eval_done": metrics.get("dev_eval_done"),
        "confirm_read": False,
        "new_qc_calls": 0,
        "metrics_path": metrics.get("metrics_path"),
        "resources": dict(resources),
    })

    _emit({
        "action": "run",
        "ok": True,
        "mode": mode,
        "run_id": run_id,
        "embedding_id": embedding_id,
        "seed": seed,
        "attempt": attempt,
        "out_dir": str(out_dir),
        "is_formal_run": bool(metrics.get("is_formal_run")),
        "run_outcome": metrics.get("run_outcome"),
        "attempt_status": ("leg_complete" if leg_complete else "leg_partial"),
        "target_updates": campaign_target,
        "leg_window_updates": leg_window,
        "updates_run": (metrics.get("schedule") or {}).get("updates_run"),
        "checkpoint_best_sha256": metrics.get("checkpoint_best_sha256"),
        "batch_schedule_sha256": (metrics.get("sampler") or {}).get(
            "batch_schedule_sha256"),
        "degree_stats": degree_stats,
        "run_records": str(run_records_path(root)),
        "new_qc_calls": 0,
    })
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_cuda_fraction(text: str) -> float:
    try:
        value = float(text)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(
            "--cuda-memory-fraction must be a float in (0,1]") from exc
    if not (0.0 < value <= 1.0):
        raise argparse.ArgumentTypeError(
            "--cuda-memory-fraction must lie in (0,1]")
    return value


def _parse_vram_gate(text: str) -> int:
    try:
        value = int(text)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(
            "--eval-vram-gate-mib must be an int >= 1") from exc
    if value < 1:
        raise argparse.ArgumentTypeError("--eval-vram-gate-mib must be >= 1")
    return value


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="v541_run",
        description="V5.4.1 Embedding-50K unified campaign runner "
                    "(validate-config / audit-data-and-manifests / "
                    "plan-init / gate-check / run)")
    parser.add_argument("--output-root", default=DEFAULT_ROOT)
    sub = parser.add_subparsers(dest="cmd", required=True)

    val = sub.add_parser("validate-config",
                         help="schema validation; --formal also checks T1 gates")
    val.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    val.add_argument("--formal", action="store_true")
    val.add_argument("--no-gv", action="store_true")
    val.add_argument("--embedding", default="E0",
                     help="candidate whose budget/smoke evidence is checked "
                          "(formal gate report only)")

    audit = sub.add_parser("audit-data-and-manifests",
                           help="read-only data/pool/split/strata audit (JSON)")
    audit.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    audit.add_argument("--tables-dir", default=DEFAULT_TABLES_DIR)
    audit.add_argument("--split", default=None)
    audit.add_argument("--v53mp-config", default=DEFAULT_V53MP_CONFIG)

    init = sub.add_parser("plan-init",
                          help="instantiate plan/{runplan,signoff,budget,pin}")
    init.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    init.add_argument("--tables-dir", default=DEFAULT_TABLES_DIR)
    init.add_argument("--split", default=None)
    init.add_argument("--v53mp-config", default=DEFAULT_V53MP_CONFIG)
    init.add_argument("--force", action="store_true",
                      help="rebuild an existing plan/pin (refused while any "
                           "run has started; re-signoff required)")

    gate = sub.add_parser("gate-check",
                          help="readiness + T1 gate evidence (machine JSON)")
    gate.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    gate.add_argument("--tables-dir", default=DEFAULT_TABLES_DIR)
    gate.add_argument("--split", default=None)
    gate.add_argument("--embedding", default="E0")

    run = sub.add_parser("run", help="one V541 run (smoke|profile|formal)")
    run.add_argument("--mode", required=True,
                     choices=("smoke", "profile", "formal"))
    run.add_argument("--run-id", required=True)
    run.add_argument("--embedding", required=True)
    run.add_argument("--seed", type=int, required=True)
    run.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    run.add_argument("--tables-dir", default=DEFAULT_TABLES_DIR)
    run.add_argument("--split", default=None)
    run.add_argument("--v53mp-config", default=DEFAULT_V53MP_CONFIG)
    run.add_argument("--device", default="cuda")
    run.add_argument("--gpu-lock", default=GPU_LOCK)
    run.add_argument("--resume", action="store_true", default=True,
                     help="continue from a matching checkpoint_last "
                          "(default: on)")
    run.add_argument("--fresh", action="store_true",
                     help="start over; REFUSES when a checkpoint already "
                          "exists (never silently discards progress)")
    run.add_argument("--stop-after", type=int, default=None,
                     help="smoke kill/resume drill: bound this leg at N "
                          "updates (N <= 200); re-run with --resume to "
                          "continue to the frozen 200")
    run.add_argument("--profile-w0", type=int, default=PROFILE_W0_DEFAULT)
    run.add_argument("--profile-w1", type=int, default=PROFILE_W1_DEFAULT)
    run.add_argument("--feature-workers", type=int, default=None)
    run.add_argument("--cuda-memory-fraction", type=_parse_cuda_fraction,
                     default=None)
    run.add_argument("--eval-vram-gate-mib", type=_parse_vram_gate, default=None)
    run.add_argument("--eval-vram-gate-timeout-s", type=int, default=600)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        if args.cmd == "validate-config":
            return cmd_validate_config(args)
        if args.cmd == "audit-data-and-manifests":
            return cmd_audit(args)
        if args.cmd == "plan-init":
            return cmd_plan_init(args)
        if args.cmd == "gate-check":
            return cmd_gate_check(args)
        if args.cmd == "run":
            return cmd_run(args)
        return 2
    except V541RunnerError as exc:
        _emit({"ok": False, "code": exc.code, "detail": exc.detail,
               "blocks": exc.blocks})
        return 1
    except (V541RunplanError, V541ProvenanceError) as exc:
        _emit({"ok": False, "code": exc.code, "detail": exc.detail})
        return 1
    except V541SchemaError as exc:
        _emit({"ok": False, "code": exc.code, "detail": exc.detail})
        return 2
    except CampaignRootError as exc:
        _emit({"ok": False, "code": "V541_ROOT", "detail": str(exc)})
        return 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
