"""Unified control RL round runner: preflight -> collect -> train -> freeze
-> execute (three policy groups) -> report.

Implements the closed loop of the ADR-0017 plan §7 package C on the frozen
E_native environment, extended per the Demo24 real-RL plan (§4/§5):

* **preflight** — per-case qualification audit over the frozen 24-case
  execution list (identity, snapshot hash, review status, legacy/repaired
  plan presence, finite start geometry, control compile, residual spec,
  adapter).  No QC runs here; every case gets a typed verdict.
* **collect** — rule-zero baseline + seeded bounded exploration over the
  train-qualified selection; one ``pes2ts_control_transition_v2`` macro step
  per decision; single-case typed failures never abort the batch; finalized
  cases are skipped on resume without re-billing; a round-level cost ledger
  records every episode.
* **train** — offline conservative-Q + restricted-imitation (``cql``) or
  explicit BC-only (``bc_only``) training on qualified transitions, with
  reaction-group train/dev split and typed rejection statistics.
* **freeze** — export the ``pesn_model_bundle_v2`` research bundle per
  training mode; heads open only on recorded evidence (residual needs
  imitation support; Q stays closed for BC).
* **execute** — re-run the SAME environment and budget with one of three
  frozen policy groups (rule_zero / bc_residual / residual_q) over the full
  execution list; environment drift, bundle applicability, and per-case
  identity are verified before any scan; the model residual is actually
  wired into candidate generation; Q scores are gated by the recorded
  action-support registry.
* **report** — full-denominator per-case comparison across groups.

The runner is explicitly research tooling: no production default changes, no
ORCA rescue (E_hybrid stays gated on the AB4 four-point gate), and every
phase writes typed artifacts under the round directory.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

from pes2ts_core.generation.campaign import load_campaign_manifest
from pes2ts_core.generation.planning.chemical_control import (
    compile_chemical_control,
    load_plan,
)
from pes2ts_core.generation.planning.classic_scan import NativeScanPolicy
from pes2ts_core.generation.planning.control_step import (
    freeze_control_environment,
    persist_control_outputs,
    run_control_scan,
)
from pes2ts_core.generation.planning.policy_adapter import (
    ACTION_SOURCE_RULE_ZERO,
    CandidatePolicyConfig,
    ControlCaseAdapter,
    EpsilonExplorePolicy,
    ModelResidualPolicy,
    QGreedyPolicy,
    RuleZeroPolicy,
    build_residual_spec,
)
from pes2ts_core.generation.planning.synchronized_path import digest
from pes2ts_core.pesn.contracts.control import (
    CONTROL_CONTEXT_DIM,
    DEFAULT_HARD_FAILURE_CODES,
    PolicyRewardSpec,
    ResidualControlSpec,
)
from pes2ts_core.pesn.offline.transitions import (
    TRANSITIONS_FILE_NAME,
    append_transitions,
    count_episodes_for_reaction,
    load_transition_set,
    register_episode,
    transition_store_summary,
    write_case_sidecar,
)

CONTROL_ROUND_SUMMARY_SCHEMA = "pesn_control_round_summary_v1"
CONTROL_SELECTION_SCHEMA = "pesn_control_round_selection_v1"
COST_LEDGER_NAME = "cost_ledger.jsonl"
POLICY_GROUPS = ("rule_zero", "bc_residual", "residual_q")

_ALLOWED_TOP = frozenset(
    {"schema_version", "seed", "paths", "environment", "residual", "candidates", "reward", "control_train", "model", "bundle"}
)
_ALLOWED_KEYS = {
    "paths": frozenset({"rounds_root", "artifacts_root"}),
    "environment": frozenset(NativeScanPolicy().__dataclass_fields__.keys()),
    "residual": frozenset({"epsilon_angstrom", "allow_sign", "range_slack_angstrom"}),
    "candidates": frozenset(
        {"model_scales", "n_single_coord_perturbs", "max_candidates",
         "exploration_probability", "exploration_seed", "dedup_tolerance"}
    ),
    "reward": frozenset(
        {"candidate_bonus", "cost_reference_seconds", "hard_failure_penalty",
         "step_cost_reference_seconds"}
    ),
    "control_train": frozenset(
        {"batch_transitions", "max_epochs", "patience", "lr", "weight_decay", "clip",
         "cql_alpha", "imitation_weight", "target_sync_epochs", "freeze_encoder",
         "encoder_init_bundle", "training_mode", "dev_fraction"}
    ),
    "model": frozenset(
        {"hidden", "n_layers", "n_basis", "use_coordinate_attention", "context_dim"}
    ),
    "bundle": frozenset({"model_id", "bc_model_id"}),
}


class ControlRoundError(RuntimeError):
    """Typed control-round failure."""


def load_control_round_config(path: str | Path) -> dict[str, Any]:
    from pes2ts_core.config_loader import load_config_mapping

    config_path = Path(path)
    if not config_path.is_file():
        raise ControlRoundError(f"config file missing: {config_path}")
    document = load_config_mapping(config_path)
    if not isinstance(document, dict):
        raise ControlRoundError(f"config is not a mapping: {config_path}")
    unknown = set(document) - _ALLOWED_TOP
    if unknown:
        raise ControlRoundError(f"unknown config sections: {sorted(unknown)}")
    for section, allowed in _ALLOWED_KEYS.items():
        value = document.get(section)
        if value is None:
            continue
        if not isinstance(value, dict):
            raise ControlRoundError(f"config section {section!r} must be a mapping")
        unknown_keys = set(value) - allowed
        if unknown_keys:
            raise ControlRoundError(
                f"unknown keys in section {section!r}: {sorted(unknown_keys)}"
            )
    if str(document.get("schema_version")) != "pesn_control_round_config_v1":
        raise ControlRoundError(
            f"config schema_version {document.get('schema_version')!r} unsupported"
        )
    return document


def build_candidate_config(config: Mapping[str, Any]) -> CandidatePolicyConfig:
    section = dict(config.get("candidates", {}))
    return CandidatePolicyConfig(
        model_scales=tuple(float(s) for s in section.get("model_scales", (0.5, 1.0))),
        n_single_coord_perturbs=int(section.get("n_single_coord_perturbs", 2)),
        max_candidates=int(section.get("max_candidates", 8)),
        exploration_probability=float(section.get("exploration_probability", 0.25)),
        exploration_seed=int(section.get("exploration_seed", config.get("seed", 42))),
        dedup_tolerance=float(section.get("dedup_tolerance", 1e-6)),
    )


def build_reward_spec(config: Mapping[str, Any]) -> PolicyRewardSpec:
    section = dict(config.get("reward", {}))
    spec = PolicyRewardSpec(
        schema_version="pes2ts_policy_reward_spec_v2",
        candidate_bonus=float(section.get("candidate_bonus", 1.0)),
        cost_reference_seconds=float(section.get("cost_reference_seconds", 900.0)),
        hard_failure_penalty=float(section.get("hard_failure_penalty", 0.25)),
        hard_failure_codes=DEFAULT_HARD_FAILURE_CODES,
    )
    spec.validate()
    return spec


def build_scan_policy(config: Mapping[str, Any]) -> NativeScanPolicy:
    overrides = dict(config.get("environment", {}))
    return NativeScanPolicy(**overrides) if overrides else NativeScanPolicy()


# ---------------------------------------------------------------------------
# Case loading shared by all scan phases
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LoadedCase:
    reaction_id: str
    control: Any
    spec: ResidualControlSpec
    adapter: ControlCaseAdapter
    origin_geometry: list[list[float]]
    plan_source: str


def _find_case_plan(reaction_id: str, legacy_root: Path, repaired_root: Path | None):
    for label, root in (("legacy", legacy_root), ("repaired", repaired_root)):
        if root is None:
            continue
        plan_path = root / reaction_id / "plan.json"
        if plan_path.is_file():
            return label, plan_path
    return None, None


def _load_round_case(
    *,
    reaction_id: str,
    manifest: Mapping[str, Any],
    manifest_path: Path,
    legacy_root: Path,
    config: Mapping[str, Any],
    scan_policy: NativeScanPolicy,
    repaired_root: Path | None = None,
) -> LoadedCase:
    case_entries = {case["reaction_id"]: case for case in manifest["cases"]}
    if reaction_id not in case_entries:
        raise ControlRoundError(f"reaction {reaction_id} not in the campaign manifest")
    plan_source, plan_path = _find_case_plan(reaction_id, legacy_root, repaired_root)
    if plan_path is None:
        raise ControlRoundError(
            f"{reaction_id}: no legacy or repaired plan.json under the given roots"
        )
    plan = load_plan(plan_path)

    snapshot_ref = Path(case_entries[reaction_id]["snapshot_path"])
    snapshot_path = (
        snapshot_ref if snapshot_ref.is_absolute() else Path.cwd() / snapshot_ref
    )
    if not snapshot_path.is_file():
        manifest_anchored = manifest_path.resolve().parent.parent / snapshot_ref
        if manifest_anchored.is_file():
            snapshot_path = manifest_anchored
    endpoint_r = endpoint_p = None
    if snapshot_path.is_file():
        snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
        endpoint_r = snapshot.get("r_coordinates")
        endpoint_p = snapshot.get("p_coordinates")

    control = compile_chemical_control(
        plan,
        reaction_id=reaction_id,
        n_grid_points=scan_policy.n_grid_points,
        endpoint_geometry_r=endpoint_r,
        endpoint_geometry_p=endpoint_p,
    )
    origin_geometry = plan.get("start_geometry")
    if not isinstance(origin_geometry, list) or len(origin_geometry) != len(control.elements):
        raise ControlRoundError(f"{reaction_id}: plan lacks a usable start_geometry")

    residual_cfg = dict(config.get("residual", {}))
    spec = build_residual_spec(
        control,
        epsilon_angstrom=float(residual_cfg.get("epsilon_angstrom", 0.10)),
        allow_sign=int(residual_cfg.get("allow_sign", 0)),
        range_slack_angstrom=float(residual_cfg.get("range_slack_angstrom", 0.0)),
    )
    if endpoint_r is None or endpoint_p is None:
        raise ControlRoundError(
            f"{reaction_id}: case snapshot endpoints are required for the adapter"
        )
    adapter = ControlCaseAdapter.from_case(
        control=control,
        plan=plan,
        endpoint_geometry_r=endpoint_r,
        endpoint_geometry_p=endpoint_p,
    )
    return LoadedCase(
        reaction_id=reaction_id,
        control=control,
        spec=spec,
        adapter=adapter,
        origin_geometry=list(origin_geometry),
        plan_source=plan_source,
    )


def _case_sidecar_payload(case: LoadedCase) -> dict[str, Any]:
    return {
        "reaction_id": case.reaction_id,
        "task_graph": case.adapter.task_graph_payload,
        "coordinate_spec": case.adapter.coordinate_spec_payload,
        "charge": case.adapter.charge,
        "multiplicity": case.adapter.multiplicity,
        "atomic_numbers": list(case.adapter.atomic_numbers),
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2),
        encoding="utf-8",
    )


def _append_cost_ledger(round_dir: Path, entry: Mapping[str, Any]) -> None:
    path = Path(round_dir) / COST_LEDGER_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    row = dict(entry)
    row["ts"] = datetime.now(UTC).isoformat(timespec="seconds")
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def _case_stats(result: Mapping[str, Any]) -> dict[str, Any]:
    decisions = result.get("decisions", [])
    transitions = result.get("_transitions", [])
    sources: dict[str, int] = {}
    for entry in decisions:
        source = str(entry.get("source"))
        sources[source] = sources.get(source, 0) + 1
    nonzero = sum(count for source, count in sources.items()
                  if source != ACTION_SOURCE_RULE_ZERO)
    refusals = sum(len(t.get("refusal_reasons", ())) for t in transitions)
    return {
        "n_decisions": len(decisions),
        "n_nonzero_executed": nonzero,
        "executed_action_sources": sources,
        "n_refusals": refusals,
    }


def _case_row(
    *,
    reaction_id: str,
    result: Mapping[str, Any] | None,
    status: str,
    failure_code: str | None = None,
    failure_detail: str | None = None,
    resumed: bool = False,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "reaction_id": reaction_id,
        "status": status,
        "failure_code": failure_code,
        "failure_detail": failure_detail,
        "resumed": resumed,
    }
    if result is not None:
        row.update(_case_stats(result))
        row["completed_interval"] = result.get("completed_interval")
        row["last_accepted_lambda"] = result.get("last_accepted_lambda")
        row["cost"] = result.get("cost")
    else:
        row.update({
            "n_decisions": 0, "n_nonzero_executed": 0,
            "executed_action_sources": {}, "n_refusals": 0,
            "completed_interval": None, "last_accepted_lambda": None,
            "cost": None,
        })
    return row


def _case_failure_row(reaction_id: str, exc: Exception) -> dict[str, Any]:
    detail = f"{type(exc).__name__}: {exc}"
    code = getattr(exc, "code", None) if not isinstance(exc, ControlRoundError) else None
    return _case_row(
        reaction_id=reaction_id,
        result=None,
        status="rejected_input",
        failure_code=f"CASE_FAILURE:{code or type(exc).__name__}",
        failure_detail=detail[:500],
    )


# ---------------------------------------------------------------------------
# Phase: preflight (qualification audit, no QC)
# ---------------------------------------------------------------------------


def run_control_preflight(
    *,
    round_dir: str | Path,
    manifest_path: str | Path,
    legacy_root: str | Path,
    selection_path: str | Path,
    config: Mapping[str, Any],
    repaired_root: str | Path | None = None,
) -> dict[str, Any]:
    root = Path(round_dir)
    manifest = load_campaign_manifest(Path(manifest_path))
    selection = _load_selection(selection_path)
    scan_policy = build_scan_policy(config)
    reward_spec = build_reward_spec(config)
    environment = freeze_control_environment(
        policy=scan_policy, reward_spec=reward_spec
    )
    review_status = dict(manifest.get("review_status", {}))
    import numpy as np

    rows: list[dict[str, Any]] = []
    for reaction_id in selection["reaction_ids"]:
        row: dict[str, Any] = {"reaction_id": reaction_id}
        entry = next(
            (c for c in manifest["cases"] if c["reaction_id"] == reaction_id), None
        )
        if entry is None:
            row.update(status="rejected", failure_code="NOT_IN_CAMPAIGN_MANIFEST")
            rows.append(row)
            continue
        row["review_status"] = review_status.get(reaction_id)
        snapshot_path = Path(entry["snapshot_path"])
        if not snapshot_path.is_file():
            manifest_anchored = (
                Path(manifest_path).resolve().parent.parent / snapshot_path
            )
            if manifest_anchored.is_file():
                snapshot_path = manifest_anchored
        payload_hash = None
        if snapshot_path.is_file():
            payload_hash = hashlib.sha256(
                snapshot_path.read_text(encoding="utf-8").encode("utf-8")
            ).hexdigest()
        if payload_hash != entry.get("snapshot_sha256"):
            row.update(status="rejected", failure_code="SNAPSHOT_HASH_MISMATCH")
            rows.append(row)
            continue
        try:
            case = _load_round_case(
                reaction_id=reaction_id,
                manifest=manifest,
                manifest_path=Path(manifest_path),
                legacy_root=Path(legacy_root),
                config=config,
                scan_policy=scan_policy,
                repaired_root=Path(repaired_root) if repaired_root else None,
            )
        except Exception as exc:  # noqa: BLE001 - typed per-case audit verdict
            row.update(
                status="rejected",
                failure_code=f"CASE_LOAD_FAILED:{type(exc).__name__}",
                failure_detail=str(exc)[:400],
            )
            rows.append(row)
            continue
        geometry_finite = bool(
            np.isfinite(np.asarray(case.origin_geometry, dtype=float)).all()
        )
        if not geometry_finite:
            row.update(status="rejected", failure_code="INPUT_GEOMETRY_INVALID")
            rows.append(row)
            continue
        row.update({
            "status": "qualified",
            "plan_source": case.plan_source,
            "n_drivers": len(case.spec.driver_ids),
            "epsilon_angstrom": sorted(set(case.spec.epsilon)),
            "charge": case.adapter.charge,
            "multiplicity": case.adapter.multiplicity,
            "n_atoms": len(case.adapter.atomic_numbers),
        })
        rows.append(row)

    counts: dict[str, int] = {}
    for row in rows:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    summary = {
        "schema_version": CONTROL_ROUND_SUMMARY_SCHEMA,
        "phase": "preflight",
        "n_cases": len(rows),
        "status_counts": counts,
        "environment_id": environment["environment_id"],
        "environment_hash": environment["environment_hash"],
        "cases": rows,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    _write_json(root / "preflight" / "preflight_summary.json", summary)
    return summary


# ---------------------------------------------------------------------------
# Phase: collection (rule + bounded exploration on E_native)
# ---------------------------------------------------------------------------


def run_control_collection(
    *,
    round_dir: str | Path,
    manifest_path: str | Path,
    legacy_root: str | Path,
    selection_path: str | Path,
    config: Mapping[str, Any],
    transport,
    max_cases: int | None = None,
    repaired_root: str | Path | None = None,
    resume: bool = True,
) -> dict[str, Any]:
    root = Path(round_dir)
    selection = _load_selection(selection_path)
    reaction_ids = list(selection["reaction_ids"])
    if max_cases is not None:
        reaction_ids = reaction_ids[: int(max_cases)]

    manifest = load_campaign_manifest(Path(manifest_path))
    scan_policy = build_scan_policy(config)
    reward_spec = build_reward_spec(config)
    candidate_config = build_candidate_config(config)
    environment = freeze_control_environment(
        policy=scan_policy, reward_spec=reward_spec
    )
    _validate_round_environment(root, environment)
    policy = EpsilonExplorePolicy(candidate_config)

    _write_json(root / "environment_freeze.json", environment)
    _write_json(
        root / "collection_policy.json",
        {
            "policy_id": policy.policy_id,
            "policy_hash": policy.policy_hash,
            "config": candidate_config.to_payload(),
            "environment_hash": environment["environment_hash"],
        },
    )

    transitions_path = root / "transitions" / TRANSITIONS_FILE_NAME
    started = time.monotonic()
    case_rows: list[dict[str, Any]] = []
    append_stats_total = {"appended": 0, "duplicates_skipped": 0}
    for reaction_id in reaction_ids:
        case_dir = root / "cases" / reaction_id
        finalized = case_dir / "control" / "control_scan_result.json"
        if resume and finalized.is_file():
            try:
                prior = json.loads(finalized.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                prior = None
            if prior is not None:
                row = _case_row(
                    reaction_id=reaction_id,
                    result=prior,
                    status=str(prior.get("status", "resumed_unknown")),
                    failure_code=prior.get("failure_code"),
                    resumed=True,
                )
                case_rows.append(row)
                _append_cost_ledger(root, {
                    "phase": "collect", "reaction_id": reaction_id,
                    "status": row["status"], "resumed": True,
                    "wall_seconds": 0.0, "host_requests": 0,
                    "environment_hash": environment["environment_hash"],
                })
                continue
        try:
            case = _load_round_case(
                reaction_id=reaction_id,
                manifest=manifest,
                manifest_path=Path(manifest_path),
                legacy_root=Path(legacy_root),
                config=config,
                scan_policy=scan_policy,
                repaired_root=Path(repaired_root) if repaired_root else None,
            )
            write_case_sidecar(root, _case_sidecar_payload(case))
            episode_seq = count_episodes_for_reaction(root, case.reaction_id)
            episode_tag = f"e{episode_seq:03d}"
            result = run_control_scan(
                control=case.control,
                origin_geometry=case.origin_geometry,
                transport=transport,
                save_dir=case_dir,
                spec=case.spec,
                reward_spec=reward_spec,
                decide=policy.decide,
                policy=scan_policy,
                reaction_id=case.reaction_id,
                episode_tag=episode_tag,
                round_id=root.name,
                snapshot_dir=case_dir / "control" / "snapshots",
            )
            stats = _case_stats(result)
            transitions_payload = result.pop("_transitions", [])
            persist_control_outputs(case_dir, result)
            append_stats = append_transitions(transitions_path, transitions_payload)
            episode_entry = None
            if transitions_payload:
                episode_entry = register_episode(
                    root,
                    episode_id=f"{root.name}:{case.reaction_id}:{episode_tag}",
                    round_id=root.name,
                    reaction_id=case.reaction_id,
                    transition_docs=transitions_payload,
                    policy_id=policy.policy_id,
                    policy_hash=policy.policy_hash,
                )
            for key in append_stats_total:
                append_stats_total[key] += append_stats.get(key, 0)
            cost = dict(result.get("cost", {}))
            case_rows.append({
                "reaction_id": case.reaction_id,
                "episode_id": (
                    episode_entry["episode_id"] if episode_entry else None
                ),
                "status": result["status"],
                "failure_code": result.get("failure_code"),
                "failure_detail": result.get("failure_detail"),
                "resumed": False,
                "plan_source": case.plan_source,
                **stats,
                "completed_interval": result["completed_interval"],
                "cost": cost,
            })
            _append_cost_ledger(root, {
                "phase": "collect", "reaction_id": case.reaction_id,
                "status": result["status"], "resumed": False,
                "wall_seconds": cost.get("wall_seconds"),
                "host_requests": cost.get("host_requests"),
                "n_optimization_cycles": cost.get("n_optimization_cycles_logged"),
                "environment_hash": environment["environment_hash"],
            })
        except Exception as exc:  # noqa: BLE001 - one bad case never stops the 24
            row = _case_failure_row(reaction_id, exc)
            case_rows.append(row)
            _append_cost_ledger(root, {
                "phase": "collect", "reaction_id": reaction_id,
                "status": row["status"], "resumed": False,
                "failure_code": row["failure_code"],
                "environment_hash": environment["environment_hash"],
            })

    summary = {
        "schema_version": CONTROL_ROUND_SUMMARY_SCHEMA,
        "phase": "collect",
        "environment_id": environment["environment_id"],
        "environment_hash": environment["environment_hash"],
        "policy_id": policy.policy_id,
        "policy_hash": policy.policy_hash,
        "selection_manifest_sha256": selection.get("content_sha256"),
        "n_cases": len(reaction_ids),
        "n_executed": sum(1 for r in case_rows if not r.get("resumed")),
        "n_resumed": sum(1 for r in case_rows if r.get("resumed")),
        "transition_append_stats": append_stats_total,
        "cases": case_rows,
        "wall_seconds": time.monotonic() - started,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    _write_json(root / "collect_summary.json", summary)
    return summary


def _validate_round_environment(root: Path, environment: Mapping[str, Any]) -> None:
    freeze_path = Path(root) / "environment_freeze.json"
    if not freeze_path.is_file():
        return
    stored = json.loads(freeze_path.read_text(encoding="utf-8"))
    if stored.get("environment_hash") != environment["environment_hash"]:
        raise ControlRoundError(
            "environment drift: the round directory was frozen under a different "
            f"environment hash ({stored.get('environment_hash', '?')[:12]} != "
            f"{environment['environment_hash'][:12]}); refuse to mix environments"
        )


# ---------------------------------------------------------------------------
# Phase: training + freeze
# ---------------------------------------------------------------------------


def _resolve_training_mode(
    param_mode: str | None, config: Mapping[str, Any]
) -> str:
    from pes2ts_core.pesn.offline.control_trainer import TRAINING_MODES

    config_mode = str(
        (config.get("control_train") or {}).get("training_mode", "cql")
    )
    if config_mode not in TRAINING_MODES:
        raise ControlRoundError(
            f"config training_mode {config_mode!r} not in {list(TRAINING_MODES)}"
        )
    # An explicit CLI mode overrides the config default; the effective mode is
    # embedded in the run_id, checkpoint payload, and mode-labelled summaries.
    mode = param_mode or config_mode
    if mode not in TRAINING_MODES:
        raise ControlRoundError(f"training_mode {mode!r} not in {list(TRAINING_MODES)}")
    return mode


def effective_training_config(
    config: Mapping[str, Any], mode: str
) -> dict[str, Any]:
    """Deep-copy ``config`` with the effective training mode written back.

    Round0 defect (RL repair plan §4): the CLI override was resolved for the
    run_id/labels but never reached ``ControlTrainSettings``, so a ``bc``
    invocation trained (and checkpointed) as CQL.  Every downstream consumer
    (settings, config digest, RunRecord, checkpoints, summaries, freeze) must
    consume the returned effective config, never the raw one.
    """
    import copy

    effective = copy.deepcopy(dict(config))
    train_section = dict(effective.get("control_train") or {})
    train_section["training_mode"] = mode
    effective["control_train"] = train_section
    return effective


def _load_td_support_registry(
    round_dir: str | Path,
    config: Mapping[str, Any],
    freeze_doc: Mapping[str, Any],
    settings: "ControlTrainSettings",
    transitions: Sequence[Any],
):
    """Build the TD-side registry gate from the whitelisted FIT transitions.

    Review F2: the input must be the same selection-filtered qualified
    transition list the trainer consumes.  This function never re-reads the
    transition store and never widens the training-use allow-list; the fit
    subset comes from the same deterministic reaction-group split the trainer
    applies, so TD, calibration, and the exported registry share one support
    domain (plan §5.1/§5.2).
    """
    from pes2ts_core.pesn.inference.action_support import ActionSupportRegistry
    from pes2ts_core.pesn.offline.action_support import build_action_support_registry
    from pes2ts_core.pesn.offline.control_trainer import split_fit_dev_reactions

    if not transitions:
        return None
    fit_reactions, _dev = split_fit_dev_reactions(
        transitions,
        dev_fraction=settings.dev_fraction,
        seed=settings.seed,
    )
    fit_set = set(fit_reactions)
    fit_transitions = [t for t in transitions if t.reaction_id in fit_set]
    environment_hash = str(freeze_doc["environment_hash"])
    reward_spec = build_reward_spec(config)
    doc = build_action_support_registry(
        fit_transitions,
        environment_hash=environment_hash,
        reward_spec_sha256=reward_spec.content_hash(),
    )
    registry = ActionSupportRegistry(doc)
    if registry.reward_spec_sha256 != reward_spec.content_hash():
        raise ControlRoundError("TD support registry reward-spec mismatch")
    if registry.environment_hash != environment_hash:
        raise ControlRoundError("TD support registry environment mismatch")
    return registry


def run_control_training(
    *,
    round_dir: str | Path,
    config: Mapping[str, Any],
    selection_path: str | Path,
    training_mode: str | None = None,
) -> dict[str, Any]:
    from pes2ts_core.pesn.offline.control_trainer import (
        ControlTrainer,
        ControlTrainSettings,
    )
    from pes2ts_core.pesn.offline.experiments.runs import (
        RunRecord,
        config_content_sha256,
    )

    mode = _resolve_training_mode(training_mode, config)
    mode_tag = "bc" if mode == "bc_only" else "cql"
    called_config_sha = config_content_sha256(dict(config))
    config = effective_training_config(config, mode)
    root = Path(round_dir)
    freeze_doc = json.loads((root / "environment_freeze.json").read_text(encoding="utf-8"))
    selection = _load_selection(selection_path)
    load = load_transition_set(
        root,
        environment_hash=str(freeze_doc["environment_hash"]),
        allowed_reactions=list(selection["reaction_ids"]),
    )
    transitions = load.qualified
    if not transitions:
        raise ControlRoundError(
            "no qualified transitions: run the collect phase first (or check the "
            "environment hash / selection allow-list); rejected="
            f"{json.dumps(load.rejected, sort_keys=True)}"
        )
    digest_of_transitions = hashlib.sha256(
        "\n".join(
            json.dumps(t.doc, sort_keys=True) for t in transitions
        ).encode("utf-8")
    ).hexdigest()

    settings = ControlTrainSettings.from_config(dict(config))
    config_sha = config_content_sha256(dict(config))
    td_registry = _load_td_support_registry(
        root, config, freeze_doc, settings, transitions
    )
    td_registry_sha = (
        hashlib.sha256(
            json.dumps(td_registry.document, sort_keys=True).encode("utf-8")
        ).hexdigest()
        if td_registry is not None
        else None
    )
    runs_root = root / "runs"
    attempt = 1
    date = datetime.now(UTC).strftime("%Y%m%d")
    run_id = f"control-{date}-{config_sha[:8]}-seed{settings.seed}-{mode_tag}-r{attempt}"
    while (runs_root / run_id).exists():
        attempt += 1
        run_id = (
            f"control-{date}-{config_sha[:8]}-seed{settings.seed}-{mode_tag}-r{attempt}"
        )
    run_record = RunRecord.start(
        runs_root=runs_root,
        run_id=run_id,
        config=dict(config),
        config_sha256=config_sha,
        dataset_manifest_sha256=digest_of_transitions,
        split_manifest_sha256=str(selection.get("content_sha256", "")),
        feature_spec_version="pesn_feature_spec_v1.0",
        label_spec_sha256="",
        seed=settings.seed,
        n_parameters=0,
        backend_versions={},
        training_mode=mode,
        called_config_sha256=called_config_sha,
    )
    trainer = ControlTrainer(
        settings=settings,
        run_record=run_record,
        config_sha256=config_sha,
        transitions_digest=digest_of_transitions,
        support_registry=td_registry,
    )
    run_record.update_status(
        "running",
        n_parameters=trainer.model.n_parameters(),
        n_parameters_total=trainer.n_parameters_total,
        n_parameters_frozen=trainer.n_parameters_frozen,
        n_parameters_trainable=trainer.n_parameters_trainable,
        training_mode=mode,
    )
    try:
        summary = trainer.train(transitions)
    except Exception as exc:
        run_record.record_failure("control_train", str(exc))
        run_record.finalize("failed")
        raise ControlRoundError(f"control training failed: {exc}") from exc
    run_record.finalize("completed")

    from pes2ts_core.pesn.offline.action_support import (
        build_action_support_registry,
    )

    fit_reactions = set(summary["fit_reactions"])
    fit_transitions = [t for t in transitions if t.reaction_id in fit_reactions]
    registry_doc = build_action_support_registry(
        fit_transitions,
        environment_hash=str(freeze_doc["environment_hash"]),
        reward_spec_sha256=build_reward_spec(config).content_hash(),
    )
    registry_path = run_record.run_dir / "action_support_registry.json"
    registry_sha = hashlib.sha256(
        json.dumps(registry_doc, sort_keys=True).encode("utf-8")
    ).hexdigest()
    if td_registry_sha is not None and registry_sha != td_registry_sha:
        raise ControlRoundError(
            "exported support registry differs from the TD-training registry "
            f"({registry_sha[:12]} != {td_registry_sha[:12]}); training and "
            "export must share one support domain"
        )
    _write_json(registry_path, registry_doc)
    run_record.append_event(
        "action_support_registry_built",
        path=str(registry_path),
        content_sha256=registry_sha,
        n_fit_transitions=len(fit_transitions),
        calibration=registry_doc["calibration"],
    )

    store_summary = transition_store_summary(transitions)
    payload = {
        "schema_version": CONTROL_ROUND_SUMMARY_SCHEMA,
        "phase": "train",
        "training_mode": mode,
        "effective_config_sha256": config_sha,
        "transitions_digest": digest_of_transitions,
        "action_support_registry_path": str(registry_path),
        "action_support_registry_sha256": registry_sha,
        "td_support_registry_sha256": td_registry_sha,
        "support_registry_td_export_consistent": (
            td_registry_sha is None or td_registry_sha == registry_sha
        ),
        "run_id": run_id,
        "n_parameters": summary["n_parameters"],
        "n_parameters_total": summary.get("n_parameters_total", summary["n_parameters"]),
        "n_parameters_frozen": summary.get("n_parameters_frozen", 0),
        "n_parameters_trainable": summary.get("n_parameters_trainable", summary["n_parameters"]),
        "optimizer_param_names_sha256": summary.get("optimizer_param_names_sha256"),
        "q_params_sha256_before": summary.get("q_params_sha256_before"),
        "q_params_sha256_after": summary.get("q_params_sha256_after"),
        "encoder_params_sha256_before": summary.get("encoder_params_sha256_before"),
        "encoder_params_sha256_after": summary.get("encoder_params_sha256_after"),
        "q_params_changed": summary.get("q_params_changed"),
        "encoder_params_changed": summary.get("encoder_params_changed"),
        "n_imitation_distinct_samples": summary.get("n_imitation_distinct_samples", 0),
        "n_imitation_epoch_uses": summary.get("n_imitation_epoch_uses", summary["n_imitation_total"]),
        "encoder_frozen": summary["encoder_frozen"],
        "best_epoch": summary["best_epoch"],
        "best_loss": summary["best_loss"],
        "selection_basis": summary["selection_basis"],
        "fit_reactions": summary["fit_reactions"],
        "dev_reactions": summary["dev_reactions"],
        "n_imitation_total": summary["n_imitation_total"],
        "n_bootstrap_inconsistent": summary["n_bootstrap_inconsistent"],
        "bootstrap_support": summary.get("bootstrap_support", {
            "gate": "legacy_recorded_candidates",
            "n_cancelled_no_supported_successor_action": 0,
            "unsupported_reasons": {},
        }),
        "transition_store": store_summary,
        "transition_rejections": load.rejected,
    }
    _write_json(root / f"train_summary_{mode_tag}.json", payload)
    return {"run_id": run_id, **payload}


def run_control_freeze(
    *,
    round_dir: str | Path,
    config: Mapping[str, Any],
    selection_path: str | Path,
    training_mode: str | None = None,
) -> dict[str, Any]:
    from pes2ts_core.pesn.offline.experiments.runs import config_content_sha256
    from pes2ts_core.pesn.offline.publishing.control_export import export_control_bundle

    mode = _resolve_training_mode(training_mode, config)
    mode_tag = "bc" if mode == "bc_only" else "cql"
    config = effective_training_config(config, mode)
    root = Path(round_dir)
    freeze_doc = json.loads((root / "environment_freeze.json").read_text(encoding="utf-8"))
    reward_spec = build_reward_spec(config)
    selection = _load_selection(selection_path)
    train_summary_path = root / f"train_summary_{mode_tag}.json"
    if not train_summary_path.is_file():
        raise ControlRoundError(
            f"no train_summary_{mode_tag}.json; run the train phase in {mode!r} mode"
        )
    train_summary = json.loads(train_summary_path.read_text(encoding="utf-8"))
    run_id = str(train_summary.get("run_id") or "")
    if not run_id:
        raise ControlRoundError(
            f"train_summary_{mode_tag}.json carries no run_id; cannot bind freeze"
        )
    run_dir = root / "runs" / run_id
    if not run_dir.is_dir():
        raise ControlRoundError(
            f"train summary run {run_id!r} has no run directory under "
            f"{root / 'runs'}; freeze binds the explicit run, never a guess"
        )

    bundle_cfg = dict(config.get("bundle") or {})
    base_model_id = str(bundle_cfg.get("model_id", "pesn-control-v0.1.0-rc.1"))
    model_id = (
        str(bundle_cfg.get("bc_model_id", f"{base_model_id}-bc"))
        if mode == "bc_only"
        else base_model_id
    )
    artifacts_root = Path(
        str((config.get("paths") or {}).get("artifacts_root", "artifacts/pesn"))
    )
    control_spec_sha = _round_control_spec_sha(root)

    effective_config_sha = str(train_summary.get("effective_config_sha256") or "")
    current_config_sha = config_content_sha256(dict(config))
    if not effective_config_sha:
        raise ControlRoundError(
            "train summary predates the effective-config fix (no "
            "effective_config_sha256); retrain before freezing"
        )
    if effective_config_sha != current_config_sha:
        raise ControlRoundError(
            "freeze config differs from the trained run's effective config "
            f"({current_config_sha[:12]} != {effective_config_sha[:12]}); "
            "refusing to package a run trained under another configuration"
        )
    recorded_mode = str(train_summary.get("training_mode") or "")
    if recorded_mode != mode:
        raise ControlRoundError(
            f"train summary mode {recorded_mode!r} != freeze mode {mode!r}; "
            "refusing to mix runs"
        )

    registry_path_str = str(train_summary.get("action_support_registry_path") or "")
    registry_sha_recorded = str(
        train_summary.get("action_support_registry_sha256") or ""
    )
    if not registry_path_str or not registry_sha_recorded:
        raise ControlRoundError(
            "train summary carries no action-support registry identity; new "
            "rounds must bind the training registry into the package "
            "(retrain with the current pipeline; only explicitly legacy "
            "bundles may ship without one)"
        )
    registry_path = Path(registry_path_str)
    if not registry_path.is_file():
        raise ControlRoundError(
            f"training support registry missing: {registry_path}"
        )
    registry_doc = json.loads(registry_path.read_text(encoding="utf-8"))
    registry_sha = hashlib.sha256(
        json.dumps(registry_doc, sort_keys=True).encode("utf-8")
    ).hexdigest()
    if registry_sha != registry_sha_recorded:
        raise ControlRoundError(
            "training support registry digest mismatch "
            f"({registry_sha[:12]} != {registry_sha_recorded[:12]}); refusing "
            "to bind a modified registry"
        )

    n_imitation_total = int(train_summary.get("n_imitation_total", 0))
    open_residual = n_imitation_total > 0
    open_q_proxy = mode == "cql"
    executed_sources = dict(
        train_summary.get("transition_store", {}).get("executed_action_sources", {})
    )
    supported_sources = sorted({ACTION_SOURCE_RULE_ZERO, *executed_sources.keys()})
    metrics = {
        "status": "research_only_first_round",
        "training_mode": mode,
        "environment_hash": str(freeze_doc["environment_hash"]),
        "n_imitation_total": n_imitation_total,
        "executed_action_sources": executed_sources,
        "supported_action_sources": supported_sources,
        "transition_rejections": train_summary.get("transition_rejections", {}),
        "residual_opened_on_evidence": open_residual,
        "q_proxy_opened_on_evidence": open_q_proxy,
    }
    bundle_path = export_control_bundle(
        run_dir=run_dir,
        output_dir=artifacts_root / model_id,
        model_id=model_id,
        metrics=metrics,
        run_summary={
            "run_id": run_dir.name,
            "selection_manifest_sha256": selection.get("content_sha256"),
            "training_mode": mode,
            "effective_config_sha256": effective_config_sha,
            "transitions_digest": str(train_summary.get("transitions_digest") or ""),
        },
        model_config={
            "hidden": int((config.get("model") or {}).get("hidden", 64)),
            "n_layers": int((config.get("model") or {}).get("n_layers", 3)),
            "n_basis": int((config.get("model") or {}).get("n_basis", 16)),
            "context_dim": CONTROL_CONTEXT_DIM,
            "use_coordinate_attention": bool(
                (config.get("model") or {}).get("use_coordinate_attention", False)
            ),
        },
        control_spec_sha256=control_spec_sha,
        reward_spec_sha256=reward_spec.content_hash(),
        environment_id=str(freeze_doc["environment_id"]),
        open_residual=open_residual,
        open_q_proxy=open_q_proxy,
        support_registry_doc=registry_doc,
    )
    _write_json(
        root / f"freeze_summary_{mode_tag}.json",
        {
            "schema_version": CONTROL_ROUND_SUMMARY_SCHEMA,
            "phase": "freeze",
            "training_mode": mode,
            "run_id": run_dir.name,
            "bundle_path": str(bundle_path),
            "model_id": model_id,
            "open_residual": open_residual,
            "open_q_proxy": open_q_proxy,
            "supported_action_sources": supported_sources,
            "support_registry": {
                "bound": True,
                "sha256": registry_sha,
                "path": str(registry_path),
                "environment_hash": str(registry_doc["environment_hash"]),
                "reward_spec_sha256": str(registry_doc["reward_spec_sha256"]),
            },
        },
    )
    return {
        "bundle_path": str(bundle_path),
        "model_id": model_id,
        "training_mode": mode,
        "run_id": run_dir.name,
        "open_residual": open_residual,
        "open_q_proxy": open_q_proxy,
        "support_registry": {
            "bound": True,
            "sha256": registry_sha,
        },
    }


def _round_control_spec_sha(round_dir: Path) -> str:
    from pes2ts_core.pesn.offline.transitions import load_case_sidecars

    sidecars = load_case_sidecars(round_dir)
    if not sidecars:
        raise ControlRoundError("no case sidecars found in the round directory")
    payload = sorted(
        (rid, sidecar["coordinate_spec"]) for rid, sidecar in sidecars.items()
    )
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()


# ---------------------------------------------------------------------------
# Phase: execution with the frozen policy (same environment, same budget)
# ---------------------------------------------------------------------------


def _validate_execute_bundle(
    bundle_dir: Path,
    *,
    environment: Mapping[str, Any],
    reward_spec: PolicyRewardSpec,
) -> dict[str, Any]:
    manifest_path = Path(bundle_dir) / "model_bundle.json"
    if not manifest_path.is_file():
        raise ControlRoundError(f"bundle manifest missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("environment_id") != environment["environment_id"]:
        raise ControlRoundError(
            f"bundle environment {manifest.get('environment_id')!r} != round "
            f"environment {environment['environment_id']!r}"
        )
    if manifest.get("reward_spec_sha256") != reward_spec.content_hash():
        raise ControlRoundError(
            "bundle reward spec hash does not match the round's frozen reward spec"
        )
    metrics_path = Path(bundle_dir) / "metrics.json"
    metrics: dict[str, Any] = {}
    if metrics_path.is_file():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    bundle_env_hash = metrics.get("environment_hash")
    if bundle_env_hash is not None and bundle_env_hash != environment["environment_hash"]:
        raise ControlRoundError(
            "bundle was trained under a different environment hash; refusing to "
            "execute it in this round"
        )
    return {"manifest": manifest, "metrics": metrics}


def run_control_execution(
    *,
    round_dir: str | Path,
    manifest_path: str | Path,
    legacy_root: str | Path,
    selection_path: str | Path,
    config: Mapping[str, Any],
    bundle_dir: str | Path,
    transport,
    max_cases: int | None = None,
    repaired_root: str | Path | None = None,
    resume: bool = True,
    policy_group: str = "residual_q",
    execute_selection_path: str | Path | None = None,
    support_registry_path: str | Path | None = None,
) -> dict[str, Any]:
    from pes2ts_core.pesn.inference.control_predictor import ControlPredictor

    if policy_group not in POLICY_GROUPS:
        raise ControlRoundError(
            f"policy_group {policy_group!r} not in {list(POLICY_GROUPS)}"
        )
    if bundle_dir is None:
        raise ControlRoundError(
            "execute requires the frozen bundle directory (pass --bundle or run "
            "the freeze phase in the same invocation)"
        )
    root = Path(round_dir)
    freeze_doc = json.loads((root / "environment_freeze.json").read_text(encoding="utf-8"))
    selection = _load_selection(execute_selection_path or selection_path)
    reaction_ids = list(selection["reaction_ids"])
    denominator = len(reaction_ids)
    if max_cases is not None:
        reaction_ids = reaction_ids[: int(max_cases)]

    manifest = load_campaign_manifest(Path(manifest_path))
    scan_policy = build_scan_policy(config)
    reward_spec = build_reward_spec(config)
    candidate_config = build_candidate_config(config)
    environment = freeze_control_environment(
        policy=scan_policy, reward_spec=reward_spec
    )
    if environment["environment_hash"] != freeze_doc["environment_hash"]:
        raise ControlRoundError(
            "environment drift between the collect freeze and this execution "
            f"({environment['environment_hash'][:12]} != "
            f"{freeze_doc['environment_hash'][:12]}); the comparison groups must "
            "share one frozen environment"
        )
    bundle_info = _validate_execute_bundle(
        Path(bundle_dir), environment=environment, reward_spec=reward_spec
    )
    metrics = bundle_info["metrics"]
    predictor = ControlPredictor.from_bundle(bundle_dir)

    adapters: dict[str, ControlCaseAdapter] = {}
    specs: dict[str, ResidualControlSpec] = {}

    def control_input_for(state):
        adapter = adapters[state.control.reaction_id]
        spec = specs[state.control.reaction_id]
        return adapter.build_input(
            geometry=state.parent_geometry,
            context=_context_of(state),
            spec=spec,
            base_targets=state.control.targets_at(state.target_lambda),
        )

    def model_residual_fn(state):
        available, proposal, _reason = predictor.propose_residual(
            control_input_for(state)
        )
        if not available or proposal is None:
            return None
        return [float(v) for v in proposal]

    q_support_stats = {"scored": 0, "unsupported": 0, "finite": 0,
                       "unsupported_reasons": {}}

    def score(state, candidates):
        prediction = predictor.predict(control_input_for(state), candidates)
        values: list[float | None] = []
        for estimate in prediction.q_proxy:
            q_support_stats["scored"] += 1
            if estimate.available:
                q_support_stats["finite"] += 1
                values.append(estimate.value)
            else:
                q_support_stats["unsupported"] += 1
                values.append(None)
        return values

    registry = None
    support_gate_kind = "legacy_source_labels"
    registry_sha = None
    registry_source = None
    bundle_binding = bundle_info["manifest"].get("support_registry")
    if bundle_binding is not None:
        from pes2ts_core.pesn.contracts.control import (
            CONTROL_SUPPORT_REGISTRY_FILE,
            validate_support_registry_binding,
        )
        from pes2ts_core.pesn.inference.action_support import ActionSupportRegistry

        validate_support_registry_binding(bundle_binding)
        bound_path = Path(bundle_dir) / CONTROL_SUPPORT_REGISTRY_FILE
        if not bound_path.is_file():
            raise ControlRoundError(
                "registry-bound bundle lacks its action_support_registry.json; "
                "refusing to execute without the bound support domain"
            )
        bound_doc = json.loads(bound_path.read_text(encoding="utf-8"))
        registry_content_sha = hashlib.sha256(
            json.dumps(bound_doc, sort_keys=True).encode("utf-8")
        ).hexdigest()
        if registry_content_sha != str(bundle_binding["sha256"]):
            raise ControlRoundError(
                "bundle-bound support registry hash mismatch "
                f"({registry_content_sha[:12]} != "
                f"{str(bundle_binding['sha256'])[:12]}); the registry file was "
                "tampered with or replaced"
            )
        registry = ActionSupportRegistry(bound_doc)
        if registry.environment_hash != environment["environment_hash"]:
            raise ControlRoundError(
                "bound support registry environment hash does not match this "
                "round's frozen environment; refusing to gate execution with it"
            )
        if registry.reward_spec_sha256 != reward_spec.content_hash():
            raise ControlRoundError(
                "bound support registry reward-spec hash does not match this "
                "round's frozen reward spec; refusing to gate execution with it"
            )
        registry_sha = registry_content_sha
        support_gate_kind = "registry_v1_bound"
        registry_source = "bundle_binding"
        if support_registry_path is not None:
            explicit_doc = json.loads(
                Path(support_registry_path).read_text(encoding="utf-8")
            )
            explicit_sha = hashlib.sha256(
                json.dumps(explicit_doc, sort_keys=True).encode("utf-8")
            ).hexdigest()
            if explicit_sha != registry_sha:
                raise ControlRoundError(
                    "--support-registry differs from the registry bound into "
                    "the bundle; a bound package must execute with its own "
                    "support domain"
                )
    elif support_registry_path is not None:
        from pes2ts_core.pesn.inference.action_support import ActionSupportRegistry

        registry = ActionSupportRegistry.from_json(support_registry_path)
        if registry.environment_hash != environment["environment_hash"]:
            raise ControlRoundError(
                "support registry environment hash does not match this round's "
                "frozen environment; refusing to gate execution with it"
            )
        if registry.reward_spec_sha256 != reward_spec.content_hash():
            raise ControlRoundError(
                "support registry reward-spec hash does not match this round's "
                "frozen reward spec; refusing to gate execution with it"
            )
        import hashlib as _hashlib

        registry_sha = _hashlib.sha256(
            Path(support_registry_path).read_bytes()
        ).hexdigest()
        support_gate_kind = "registry_v1_manual"
        registry_source = "cli_flag"

    sidecar_digests: dict[str, str] = {}

    def _coordinate_spec_sha_for(reaction_id: str) -> str:
        if reaction_id not in sidecar_digests:
            from pes2ts_core.pesn.offline.action_support import (
                coordinate_spec_sha256,
            )
            from pes2ts_core.pesn.offline.transitions import load_case_sidecars

            sidecars = load_case_sidecars(root)
            if reaction_id not in sidecars:
                raise ControlRoundError(
                    f"{reaction_id}: no case sidecar for support lookup"
                )
            sidecar_digests[reaction_id] = coordinate_spec_sha256(
                sidecars[reaction_id]
            )
            sidecar_digests.update({
                rid: coordinate_spec_sha256(doc)
                for rid, doc in sidecars.items()
            })
        return sidecar_digests[reaction_id]

    def registry_support_gate(state, candidates, _sources):
        spec = specs[state.control.reaction_id]
        verdicts = registry.query_actions(
            reaction_id=state.control.reaction_id,
            driver_ids=spec.driver_ids,
            epsilon=spec.epsilon,
            environment_hash=environment["environment_hash"],
            reward_spec_sha256=reward_spec.content_hash(),
            coordinate_spec_sha256=_coordinate_spec_sha_for(
                state.control.reaction_id
            ),
            geometry=[[float(x) for x in row] for row in state.parent_geometry],
            context_vector=list(_context_of(state).to_vector()),
            candidate_actions=[list(c) for c in candidates],
        )
        for verdict in verdicts:
            if not verdict.supported and verdict.reason:
                reasons = q_support_stats["unsupported_reasons"]
                reasons[verdict.reason] = reasons.get(verdict.reason, 0) + 1
        return [verdict.supported for verdict in verdicts]

    supported_sources = set(metrics.get("supported_action_sources", []))
    supported_sources.add(ACTION_SOURCE_RULE_ZERO)

    def support_gate(_state, _candidates, sources):
        return [str(source) in supported_sources for source in sources]

    if registry is not None:
        execution_support_gate = registry_support_gate
    else:
        execution_support_gate = support_gate

    if policy_group == "rule_zero":
        policy = RuleZeroPolicy(candidate_config)
    elif policy_group == "bc_residual":
        policy = ModelResidualPolicy(candidate_config, model_residual_fn)
    else:
        if predictor.capabilities().get("q_proxy", "disabled") == "disabled":
            raise ControlRoundError(
                "frozen bundle has no research q_proxy support; refusing to run "
                "the residual_q execution group with an untrained head"
            )
        policy = QGreedyPolicy(
            candidate_config, score,
            model_residual_fn=model_residual_fn,
            support_fn=execution_support_gate,
        )

    execute_root = root / "execute" / policy_group
    started = time.monotonic()
    case_rows: list[dict[str, Any]] = []
    executed_ids = set(reaction_ids)
    for reaction_id in reaction_ids:
        case_dir = execute_root / reaction_id
        finalized = case_dir / "control" / "control_scan_result.json"
        if resume and finalized.is_file():
            try:
                prior = json.loads(finalized.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                prior = None
            if prior is not None:
                row = _case_row(
                    reaction_id=reaction_id,
                    result=prior,
                    status=str(prior.get("status", "resumed_unknown")),
                    failure_code=prior.get("failure_code"),
                    resumed=True,
                )
                row["policy_group"] = policy_group
                case_rows.append(row)
                _append_cost_ledger(root, {
                    "phase": "execute", "policy_group": policy_group,
                    "reaction_id": reaction_id, "status": row["status"],
                    "resumed": True, "wall_seconds": 0.0, "host_requests": 0,
                    "environment_hash": environment["environment_hash"],
                })
                continue
        try:
            case = _load_round_case(
                reaction_id=reaction_id,
                manifest=manifest,
                manifest_path=Path(manifest_path),
                legacy_root=Path(legacy_root),
                config=config,
                scan_policy=scan_policy,
                repaired_root=Path(repaired_root) if repaired_root else None,
            )
            adapters[case.reaction_id] = case.adapter
            specs[case.reaction_id] = case.spec
            result = run_control_scan(
                control=case.control,
                origin_geometry=case.origin_geometry,
                transport=transport,
                save_dir=case_dir,
                spec=case.spec,
                reward_spec=reward_spec,
                decide=policy.decide,
                policy=scan_policy,
                reaction_id=case.reaction_id,
                round_id=f"{root.name}:{policy_group}",
                snapshot_dir=case_dir / "control" / "snapshots",
            )
            stats = _case_stats(result)
            persist_control_outputs(case_dir, result)
            cost = dict(result.get("cost", {}))
            row = {
                "reaction_id": case.reaction_id,
                "policy_group": policy_group,
                "status": result["status"],
                "failure_code": result.get("failure_code"),
                "failure_detail": result.get("failure_detail"),
                "resumed": False,
                "plan_source": case.plan_source,
                **stats,
                "completed_interval": result["completed_interval"],
                "cost": cost,
            }
            case_rows.append(row)
            _append_cost_ledger(root, {
                "phase": "execute", "policy_group": policy_group,
                "reaction_id": case.reaction_id, "status": result["status"],
                "resumed": False,
                "wall_seconds": cost.get("wall_seconds"),
                "host_requests": cost.get("host_requests"),
                "n_optimization_cycles": cost.get("n_optimization_cycles_logged"),
                "environment_hash": environment["environment_hash"],
            })
        except Exception as exc:  # noqa: BLE001 - one bad case never stops the 24
            row = _case_failure_row(reaction_id, exc)
            row["policy_group"] = policy_group
            case_rows.append(row)
            _append_cost_ledger(root, {
                "phase": "execute", "policy_group": policy_group,
                "reaction_id": reaction_id, "status": row["status"],
                "resumed": False, "failure_code": row["failure_code"],
                "environment_hash": environment["environment_hash"],
            })

    not_attempted = [
        {"reaction_id": rid, "policy_group": policy_group, "status": "not_attempted"}
        for rid in selection["reaction_ids"]
        if rid not in executed_ids
    ]
    summary = {
        "schema_version": CONTROL_ROUND_SUMMARY_SCHEMA,
        "phase": "execute",
        "policy_group": policy_group,
        "environment_hash": freeze_doc["environment_hash"],
        "policy_id": policy.policy_id,
        "policy_hash": policy.policy_hash,
        "bundle_dir": str(bundle_dir),
        "residual_capability": predictor.capabilities().get("residual_head"),
        "q_capability": predictor.capabilities().get("q_proxy"),
        "supported_action_sources": sorted(supported_sources),
        "support_gate": {
            "kind": support_gate_kind,
            "legacy_support": registry is None,
            "registry_path": (
                str(support_registry_path) if support_registry_path else None
            ),
            "registry_sha256": registry_sha,
            "registry_source": registry_source,
        },
        "q_support_stats": q_support_stats,
        "n_denominator": denominator,
        "n_cases": len(case_rows),
        "n_executed": sum(1 for r in case_rows if not r.get("resumed")),
        "n_resumed": sum(1 for r in case_rows if r.get("resumed")),
        "cases": case_rows + not_attempted,
        "wall_seconds": time.monotonic() - started,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    _write_json(execute_root / "execute_summary.json", summary)
    return summary


def _context_of(state):
    from pes2ts_core.pesn.contracts.control import ControlContext

    context = ControlContext(
        nominal_lambda=float(state.last_accepted_lambda),
        next_target_lambda=float(state.target_lambda),
        remaining_wall_fraction=float(state.remaining_wall_fraction),
        remaining_request_fraction=float(state.remaining_request_fraction),
        recent_rejections=int(state.recent_rejections),
        rescue_quota_remaining=0,
        candidate_already_awarded=bool(state.candidate_already_awarded),
    )
    context.validate()
    return context


# ---------------------------------------------------------------------------
# Phase: report (full-denominator comparison across groups)
# ---------------------------------------------------------------------------


def run_control_report(
    *,
    round_dir: str | Path,
    execute_selection_path: str | Path,
    groups: Sequence[str] = POLICY_GROUPS,
) -> dict[str, Any]:
    root = Path(round_dir)
    selection = _load_selection(execute_selection_path)
    roles = dict(selection.get("reaction_roles") or {})
    reaction_ids = list(selection["reaction_ids"])
    rows: list[dict[str, Any]] = []
    for reaction_id in reaction_ids:
        for group in groups:
            summary_path = root / "execute" / group / "execute_summary.json"
            entry: dict[str, Any] = {
                "reaction_id": reaction_id,
                "policy_group": group,
                "role": roles.get(reaction_id),
                "status": "not_run",
            }
            if summary_path.is_file():
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                found = next(
                    (c for c in summary.get("cases", [])
                     if c.get("reaction_id") == reaction_id),
                    None,
                )
                if found is not None:
                    entry = {
                        "reaction_id": reaction_id,
                        "policy_group": group,
                        "role": roles.get(reaction_id),
                        **{
                            k: v for k, v in found.items()
                            if k != "reaction_id"
                        },
                    }
            rows.append(entry)

    by_group: dict[str, dict[str, int]] = {}
    for group in groups:
        entries = [r for r in rows if r["policy_group"] == group]
        by_group[group] = {
            "n_denominator": len(reaction_ids),
            "n_rows": len(entries),
            "n_complete_interval": sum(
                1 for r in entries if r.get("completed_interval") is True
            ),
            "n_partial_prefix": sum(
                1 for r in entries if r.get("status") == "partial_prefix"
            ),
            "n_rejected": sum(
                1 for r in entries if r.get("status") in ("rejected_input", "rejected")
            ),
            "n_budget_exhausted": sum(
                1 for r in entries if r.get("status") == "budget_exhausted"
            ),
            "n_not_run_or_attempted": sum(
                1 for r in entries if r.get("status") in ("not_run", "not_attempted")
            ),
            "n_nonzero_executed_total": sum(
                int(r.get("n_nonzero_executed") or 0) for r in entries
            ),
            "total_wall_seconds": sum(
                float((r.get("cost") or {}).get("wall_seconds") or 0.0)
                for r in entries
            ),
            "total_host_requests": sum(
                int((r.get("cost") or {}).get("host_requests") or 0)
                for r in entries
            ),
        }
    report = {
        "schema_version": "pesn_control_compare_report_v1",
        "n_denominator": len(reaction_ids),
        "groups": list(groups),
        "by_group": by_group,
        "cases": rows,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    _write_json(root / "reports" / "compare_report_v1.json", report)
    return report


# ---------------------------------------------------------------------------
# Selection manifest
# ---------------------------------------------------------------------------


def build_selection_manifest(
    *,
    reaction_ids: Sequence[str],
    campaign_manifest_path: str | Path,
    output_path: str | Path,
    note: str = "",
    purpose: str = "train_collect",
    reaction_roles: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Freeze a hash-bound reaction list for one round purpose.

    ``purpose`` separates the train-qualified collection list
    (``train_collect``) from the 24-case execution list (``execute24``): an
    execute list is never a training permission, and the train allow-list is
    enforced again at transition-load time.
    """
    manifest = load_campaign_manifest(Path(campaign_manifest_path))
    known = {case["reaction_id"] for case in manifest["cases"]}
    unknown = set(reaction_ids) - known
    if unknown:
        raise ControlRoundError(
            f"selection contains reactions outside the campaign manifest: {sorted(unknown)}"
        )
    campaign_sha = manifest.get("content_sha256") or digest(
        {k: v for k, v in manifest.items() if k != "content_sha256"}
    )
    roles = {rid: dict(reaction_roles[rid]) for rid in reaction_ids} if reaction_roles else {}
    payload = {
        "schema_version": CONTROL_SELECTION_SCHEMA,
        "document_kind": "control_round_selection",
        "purpose": purpose,
        "status": "frozen_selection_not_executed",
        "reaction_ids": sorted(set(reaction_ids)),
        "reaction_roles": roles,
        "campaign_manifest_sha256": campaign_sha,
        "note": note
        or (
            "reactions must be individually re-verified for train qualification, "
            "group/split, and development exposure before any training claim"
        ),
    }
    canonical = json.dumps(
        {k: v for k, v in payload.items() if k != "content_sha256"},
        sort_keys=True,
    )
    payload["content_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    _write_json(Path(output_path), payload)
    return payload


def _load_selection(path: str | Path) -> dict[str, Any]:
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    if document.get("schema_version") != CONTROL_SELECTION_SCHEMA:
        raise ControlRoundError(
            f"selection manifest schema {document.get('schema_version')!r} unsupported"
        )
    if not isinstance(document.get("reaction_ids"), list) or not document["reaction_ids"]:
        raise ControlRoundError("selection manifest carries no reaction ids")
    return document


__all__ = [
    "CONTROL_ROUND_SUMMARY_SCHEMA",
    "CONTROL_SELECTION_SCHEMA",
    "COST_LEDGER_NAME",
    "POLICY_GROUPS",
    "ControlRoundError",
    "build_candidate_config",
    "build_reward_spec",
    "build_scan_policy",
    "build_selection_manifest",
    "effective_training_config",
    "load_control_round_config",
    "run_control_collection",
    "run_control_execution",
    "run_control_freeze",
    "run_control_preflight",
    "run_control_report",
    "run_control_training",
]
