"""WP-0 audit of the Demo24 round0 RL evidence (RL repair plan §3).

Freezes the round0 evidence in read-only inventories, re-audits the 258
recorded transitions (environment / receipt / reward / action / successor
identity), rebuilds the episode identity index on
``(round_id, episode_id, decision_index)``, and re-derives the canonical
reaction-group / split reference for the parent train16 set.

Outputs (never overwriting any round0 artifact):

* ``audit_round0.json``  — evidence inventory + live-verified corrective
  annotations (the old ``bc`` package was trained with the ``cql`` pipeline).
* ``source_hashes.json`` — relative path -> sha256 for every evidence file.
* ``qualified_import.json`` — typed re-audit of every transition with the
  frozen reusable count (actual, never forced to the historical 258).
* ``episode_index.json`` — unique episode/decision identities with
  successor-compatibility verdicts.
* ``split_reference.json`` — canonical reaction-group integrity and the
  reproduced fit/dev reaction split.

The script writes ONLY under the audit output root; historical JSON and
weights are never modified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from pes2ts_core.pesn.offline.transitions import (  # noqa: E402
    TRANSITIONS_FILE_NAME,
    validate_transition_doc,
)

AUDIT_SCHEMA = "pesn_rl_repair_audit_v1"
ROUND0_DEFAULT = REPO_ROOT / "outputs/pesn/demo24_rl_v1_20261007/round0"
AUDIT_ROOT_DEFAULT = REPO_ROOT / "outputs/pesn/rl_repair_v1_20261008/audit"

_EVIDENCE_SUFFIXES = (".json", ".jsonl", ".pt")
_EVIDENCE_NAMES = {
    "collect_summary.json",
    "collection_policy.json",
    "environment_freeze.json",
    "cost_ledger.jsonl",
    "train_summary_bc.json",
    "train_summary_cql.json",
    "freeze_summary_bc.json",
    "freeze_summary_cql.json",
    "control_transitions.jsonl",
    "run.json",
    "events.jsonl",
    "checkpoint_best.pt",
    "checkpoint_last.pt",
    "model_bundle.json",
    "weights.pt",
    "metrics.json",
    "checksums.json",
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2),
        encoding="utf-8",
    )


def _collect_evidence(round_root: Path) -> dict[str, str]:
    """Hash every frozen evidence file under the round + published bundles."""
    targets: list[Path] = []
    for name in (
        "collect_summary.json",
        "collection_policy.json",
        "environment_freeze.json",
        "cost_ledger.jsonl",
        "train_summary_bc.json",
        "train_summary_cql.json",
        "freeze_summary_bc.json",
        "freeze_summary_cql.json",
    ):
        candidate = round_root / name
        if candidate.is_file():
            targets.append(candidate)
    transitions = round_root / "transitions" / TRANSITIONS_FILE_NAME
    if transitions.is_file():
        targets.append(transitions)
    for sidecar in sorted((round_root / "cases").glob("*.json")):
        targets.append(sidecar)
    for run_dir in sorted((round_root / "runs").iterdir()) if (round_root / "runs").is_dir() else []:
        for name in ("run.json", "events.jsonl", "checkpoint_best.pt", "checkpoint_last.pt"):
            candidate = run_dir / name
            if candidate.is_file():
                targets.append(candidate)
    for freeze_name in ("freeze_summary_bc.json", "freeze_summary_cql.json"):
        freeze_path = round_root / freeze_name
        if not freeze_path.is_file():
            continue
        try:
            bundle_path = Path(json.loads(freeze_path.read_text())["bundle_path"])
        except (ValueError, KeyError):
            continue
        if not bundle_path.is_absolute():
            bundle_path = REPO_ROOT / bundle_path
        if bundle_path.is_dir():
            for entry in sorted(bundle_path.iterdir()):
                if entry.is_file():
                    targets.append(entry)
    hashes: dict[str, str] = {}
    for target in targets:
        key = str(target.resolve().relative_to(REPO_ROOT))
        hashes[key] = _sha256_file(target)
    return hashes


def _audit_training_mode_defect(round_root: Path) -> dict:
    """Live-verify the round0 BC/CQL control-group defect."""
    annotations: dict = {"verified_at": datetime.now(UTC).isoformat(timespec="seconds")}
    findings: list[dict] = []
    weights_hashes: dict[str, str] = {}
    try:
        import torch
    except ImportError:  # pragma: no cover - audit environment
        annotations["checkpoint_inspection"] = "unavailable: torch missing"
        return annotations

    for tag in ("bc", "cql"):
        summary_path = round_root / f"train_summary_{tag}.json"
        if not summary_path.is_file():
            continue
        summary = json.loads(summary_path.read_text())
        run_id = summary.get("run_id")
        checkpoints: dict[str, str | None] = {}
        if run_id:
            for name in ("checkpoint_best.pt", "checkpoint_last.pt"):
                path = round_root / "runs" / run_id / name
                if path.is_file():
                    payload = torch.load(path, map_location="cpu", weights_only=False)
                    checkpoints[name] = payload.get("training_mode")
        weights_path = None
        freeze_path = round_root / f"freeze_summary_{tag}.json"
        if freeze_path.is_file():
            bundle = json.loads(freeze_path.read_text()).get("bundle_path")
            if bundle:
                weights_path = Path(bundle)
                if not weights_path.is_absolute():
                    weights_path = REPO_ROOT / weights_path
                weights_path = weights_path / "weights.pt"
                if weights_path.is_file():
                    weights_hashes[tag] = _sha256_file(weights_path)
        mode_consistent = all(
            value == summary.get("training_mode") for value in checkpoints.values()
        ) if checkpoints else None
        findings.append({
            "record_name": f"train_summary_{tag}.json",
            "recorded_training_mode": summary.get("training_mode"),
            "checkpoint_training_modes": checkpoints,
            "mode_consistent": mode_consistent,
            "published_weights_sha256": weights_hashes.get(tag),
        })
    same_weights = (
        "bc" in weights_hashes and "cql" in weights_hashes
        and weights_hashes["bc"] == weights_hashes["cql"]
    )
    annotations["per_record"] = findings
    annotations["published_weights_identical_bc_cql"] = same_weights
    annotations["corrective_statement"] = (
        "The round0 record named 'bc' was trained through the CQL pipeline "
        "(the effective CLI mode never reached ControlTrainSettings); both "
        "published packages therefore carry the same weights. The historical "
        "record names are kept verbatim; any future report must describe that "
        "group as 'trained-as-CQL model-residual group', never as pure BC."
    )
    return annotations


def _audit_transitions(round_root: Path, environment_hash: str) -> tuple[dict, dict]:
    """Typed re-audit of every recorded transition."""
    transitions_path = round_root / "transitions" / TRANSITIONS_FILE_NAME
    rejected: dict[str, list[str]] = {}
    qualified: list[dict] = []

    def reject(code: str, transition_id: str) -> None:
        rejected.setdefault(code, []).append(transition_id)

    seen_ids: set[str] = set()
    episode_chains: dict[tuple[str, str], dict[int, dict]] = {}
    n_docs = 0
    for line in transitions_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        n_docs += 1
        doc = json.loads(line)
        transition_id = str(doc.get("transition_id", ""))
        if transition_id in seen_ids:
            reject("duplicate_transition_id", transition_id)
            continue
        seen_ids.add(transition_id)
        try:
            validate_transition_doc(doc)
        except Exception as exc:  # noqa: BLE001 - typed audit verdict
            reject(f"structure_invalid:{type(exc).__name__}", transition_id)
            continue
        if doc["environment_hash"] != environment_hash:
            reject("environment_mismatch", transition_id)
            continue
        # receipt presence: an executed macro step must carry its attempt refs
        if not doc.get("child_attempt_refs"):
            reject("receipt_missing", transition_id)
            continue
        reward = doc.get("reward")
        if not isinstance(reward, (int, float)) or reward != reward:
            reject("reward_not_finite", transition_id)
            continue
        action = doc.get("action") or {}
        residual = action.get("residual_u")
        n_drivers = len(action.get("driver_ids", []))
        if not isinstance(residual, list) or (n_drivers and len(residual) != n_drivers):
            reject("action_driver_dimension_mismatch", transition_id)
            continue
        executed_index = int(doc["executed_candidate_index"])
        candidates = doc["candidates"]
        executed = candidates[executed_index]
        if len(executed) != n_drivers:
            reject("executed_candidate_dimension_mismatch", transition_id)
            continue
        executed_matches_action = all(
            abs(float(a) - float(b)) <= 1e-9
            for a, b in zip(executed, residual)
        ) and len(executed) == len(residual)
        if not executed_matches_action:
            reject("executed_candidate_not_action_residual", transition_id)
            continue
        qualified.append(doc)
        episode_key = (str(doc["reaction_id"]), "ep0")
        episode_chains.setdefault(episode_key, {})[int(doc["decision_index"])] = doc

    summary = {
        "schema_version": AUDIT_SCHEMA,
        "source": str(transitions_path),
        "n_documents": n_docs,
        "n_qualified_for_corrective_reuse": len(qualified),
        "n_rejected": n_docs - len(qualified),
        "rejections": {
            code: {"count": len(ids), "transition_ids": ids}
            for code, ids in sorted(rejected.items())
        },
        "notes": (
            "Qualified = schema-valid, environment-matching, receipt-bearing, "
            "finite-reward, action/candidate-consistent transitions. The count "
            "is whatever the audit finds; the historical 258 is not a target."
        ),
    }
    return summary, {"qualified": summary, "chains": episode_chains}


def _build_episode_index(
    round_id: str, chains: dict[tuple[str, str], dict[int, dict]]
) -> dict:
    entries: list[dict] = []
    n_link_ok = n_link_broken = 0
    for (reaction_id, local_episode), chain in sorted(chains.items()):
        episode_id = f"{round_id}:{reaction_id}:{local_episode}"
        indices = sorted(chain)
        contiguous = all(b == a + 1 for a, b in zip(indices, indices[1:]))
        links_ok = True
        for current_index, next_index in zip(indices, indices[1:]):
            state_after = chain[current_index]["state_after_geometry"]
            next_start = chain[next_index]["decision_start_geometry"]
            compatible = state_after == next_start
            links_ok = links_ok and compatible
            if compatible:
                n_link_ok += 1
            else:
                n_link_broken += 1
        entries.append({
            "episode_id": episode_id,
            "round_id": round_id,
            "reaction_id": reaction_id,
            "decision_indices": indices,
            "contiguous": contiguous,
            "successor_links_compatible": links_ok,
            "terminal_status": chain[indices[-1]]["episode_status"],
            "n_decisions": len(indices),
        })
    duplicate_identities = len(entries) - len({e["episode_id"] for e in entries})
    return {
        "schema_version": AUDIT_SCHEMA,
        "round_id": round_id,
        "n_episodes": len(entries),
        "duplicate_episode_identities": duplicate_identities,
        "successor_links": {"compatible": n_link_ok, "incompatible": n_link_broken},
        "episodes": entries,
        "notes": (
            "round0 recorded one contiguous episode per reaction; multi-round or "
            "branched data must register new episode ids here instead of "
            "concatenating (reaction_id, decision_index) pairs."
        ),
    }


def _build_split_reference(reactions: list[str]) -> dict:
    """Canonical group integrity + reproduced fit/dev split for train16.

    Review F4: a necessary evidence source that cannot be read is a typed
    ``unknown`` verdict — ``group_integrity_ok`` is only ever True when both
    sources were actually read and produced no findings; a missing source
    never inherits the split.
    """
    group_findings: dict[str, list[str]] = {"in_duplicate_group": [], "not_train_split": []}
    source_status: dict[str, dict[str, object]] = {
        "duplicate_groups": {"available": False, "validated": False},
        "split_assignment": {"available": False, "validated": False},
    }
    unavailable: list[str] = []
    try:
        dup_doc = json.loads(
            (REPO_ROOT / "data/interim/duplicate_groups.json").read_text()
        )
        members: dict[str, list[str]] = {}
        for group in dup_doc.get("groups", []):
            for reaction_id in group["member_reaction_ids"]:
                members.setdefault(reaction_id, []).append(group["identity"])
        for reaction_id in reactions:
            if reaction_id in members:
                group_findings["in_duplicate_group"].append(reaction_id)
        source_status["duplicate_groups"]["available"] = True
        source_status["duplicate_groups"]["validated"] = True
    except (OSError, ValueError) as exc:
        unavailable.append(f"duplicate_groups: {type(exc).__name__}")
    try:
        import pandas as pd

        assignment = pd.read_parquet(REPO_ROOT / "data/interim/split_assignment.parquet")
        split_of = dict(zip(assignment.reaction_id, assignment.split))
        for reaction_id in reactions:
            if split_of.get(reaction_id) != "train":
                group_findings["not_train_split"].append(reaction_id)
        source_status["split_assignment"]["available"] = True
        source_status["split_assignment"]["validated"] = True
    except (OSError, ValueError) as exc:
        unavailable.append(f"split_assignment: {type(exc).__name__}")

    from pes2ts_core.pesn.offline.control_trainer import split_fit_dev_reactions

    class _Doc:
        __slots__ = ("reaction_id",)

        def __init__(self, reaction_id: str) -> None:
            self.reaction_id = reaction_id

    fit, dev = split_fit_dev_reactions(
        [_Doc(rid) for rid in reactions], dev_fraction=0.25, seed=42
    )
    if unavailable:
        integrity_status = "unknown_missing_evidence_sources"
    elif group_findings["in_duplicate_group"] or group_findings["not_train_split"]:
        integrity_status = "violations_found"
    else:
        integrity_status = "ok"
    integrity_ok = integrity_status == "ok"
    return {
        "schema_version": AUDIT_SCHEMA,
        "reactions": sorted(reactions),
        "canonical_group_audit": group_findings,
        "source_status": source_status,
        "unavailable_sources": unavailable,
        "integrity_status": integrity_status,
        "group_integrity_ok": integrity_ok,
        "inherited_fit_dev_split": {
            "basis": "round0 split_fit_dev_reactions(seed=42, dev_fraction=0.25)",
            "fit_reactions": fit,
            "dev_reactions": dev,
            "inherit": integrity_ok,
        },
        "teacher_encoder_split": {
            "bundle": "artifacts/pesn/pesn-irc-v0.1.0-rc.1",
            "subsets_used": ["train_fit", "train_dev"],
            "note": "encoder-only transfer; sealed valid/test contributed zero samples",
        },
        "notes": (
            "Old 12 fit / 4 dev inherited only when group integrity holds AND "
            "both evidence sources were read; any missing source returns "
            "integrity_status=unknown_missing_evidence_sources and the split "
            "is not inherited. On any conflict the whole group is isolated "
            "and a new split registered."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--round-dir", default=str(ROUND0_DEFAULT))
    parser.add_argument("--round-id", default="demo24_rl_v1_20261007/round0")
    parser.add_argument("--output-root", default=str(AUDIT_ROOT_DEFAULT))
    parser.add_argument("--force", action="store_true", help="rewrite audit outputs")
    args = parser.parse_args()

    round_root = Path(args.round_dir)
    if not round_root.is_dir():
        print(f"round directory missing: {round_root}", file=sys.stderr)
        return 2
    out_root = Path(args.output_root)
    if out_root.exists() and any(out_root.iterdir()) and not args.force:
        print(
            f"audit output root not empty: {out_root} (use --force to rewrite)",
            file=sys.stderr,
        )
        return 2

    environment = json.loads(
        (round_root / "environment_freeze.json").read_text(encoding="utf-8")
    )

    hashes = _collect_evidence(round_root)
    _write_json(out_root / "source_hashes.json", {
        "schema_version": AUDIT_SCHEMA,
        "evidence_files": hashes,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
    })

    audit_doc = {
        "schema_version": AUDIT_SCHEMA,
        "round_id": args.round_id,
        "round_root": str(round_root),
        "environment": {
            "environment_id": environment.get("environment_id"),
            "environment_hash": environment.get("environment_hash"),
        },
        "n_evidence_files": len(hashes),
        "corrective_annotations": _audit_training_mode_defect(round_root),
        "read_only_statement": (
            "This audit only reads round0 artifacts and writes under its own "
            "output root; historical JSON/weights are never modified."
        ),
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    _write_json(out_root / "audit_round0.json", audit_doc)

    import_summary, extra = _audit_transitions(
        round_root, str(environment["environment_hash"])
    )
    import_summary["purpose"] = "corrective_retraining_reuse"
    import_summary["environment_hash"] = environment["environment_hash"]
    _write_json(out_root / "qualified_import.json", import_summary)

    episode_index = _build_episode_index(args.round_id, extra["chains"])
    _write_json(out_root / "episode_index.json", episode_index)

    reactions = sorted({entry["reaction_id"] for entry in episode_index["episodes"]})
    _write_json(out_root / "split_reference.json", _build_split_reference(reactions))

    print(f"audit: {import_summary['n_qualified_for_corrective_reuse']}/"
          f"{import_summary['n_documents']} transitions qualified; "
          f"{episode_index['n_episodes']} episodes; {len(hashes)} evidence files hashed")
    if import_summary["n_rejected"]:
        print(f"rejections: { {k: v['count'] for k, v in import_summary['rejections'].items()} }")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
