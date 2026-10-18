"""Offline control-transition storage, qualification, and training view.

Real transitions come only from executed G2/ACP macro steps (plan §1): IRC
teacher frames have NO environment-transition qualification and are never
mixed in.  The store appends typed JSONL records; qualification checks the
frozen environment hash, the reaction allow-list, and the executed-action
support before any tensor is built.

Training view: each qualified transition provides

* ``state_features`` — GraphFeatures of the decision-start geometry,
* ``context`` — the execution-context vector recorded at decision time
  (recomputed from the typed context stored per transition),
* ``candidates`` — the frozen candidate list (for conservative Q over the
  action set, including unexecuted legal candidates),
* ``executed_index`` / ``reward`` / ``next_state_features`` /
  ``next_candidates`` / ``bootstrap_mask`` / ``terminal``.

Unexecuted candidates are never relabelled as executed; counterfactual
rewards are not fabricated (plan §4).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence


from pes2ts_core.pesn.contracts.control import (
    CONTROL_TRANSITION_SCHEMA_VERSION,
)

TRANSITIONS_FILE_NAME = "control_transitions.jsonl"
CASES_DIR_NAME = "cases"
EPISODE_INDEX_FILE_NAME = "episode_index.json"
EPISODE_INDEX_SCHEMA_VERSION = "pesn_control_episode_index_v1"

_QUALIFIED_EPISODE_STATUSES = {
    "ongoing",
    "terminal_complete",
    "terminal_step_limit",
    "truncated_budget",
}


class TransitionStoreError(RuntimeError):
    """Typed transition-store failure."""


def validate_transition_doc(doc: Mapping[str, Any]) -> None:
    if doc.get("schema_version") != CONTROL_TRANSITION_SCHEMA_VERSION:
        raise TransitionStoreError(
            f"transition schema {doc.get('schema_version')!r} unsupported"
        )
    required = (
        "transition_id",
        "environment_id",
        "environment_hash",
        "reaction_id",
        "decision_index",
        "action",
        "candidates",
        "executed_candidate_index",
        "macro_cost",
        "episode_status",
        "reward",
        "bootstrap_mask",
        "decision_start_geometry",
        "state_after_geometry",
    )
    missing = [key for key in required if key not in doc]
    if missing:
        raise TransitionStoreError(f"transition missing keys: {missing}")
    if doc["episode_status"] not in _QUALIFIED_EPISODE_STATUSES:
        raise TransitionStoreError(
            f"episode_status {doc['episode_status']!r} not qualified"
        )
    cost = doc["macro_cost"]
    if not isinstance(cost, Mapping) or "wall_seconds" not in cost:
        raise TransitionStoreError("macro_cost malformed")
    geometry = doc["decision_start_geometry"]
    if not isinstance(geometry, list) or not geometry:
        raise TransitionStoreError("decision_start_geometry missing")
    index = int(doc["executed_candidate_index"])
    if not 0 <= index < len(doc["candidates"]):
        raise TransitionStoreError("executed candidate index out of range")


def append_transitions(
    path: str | Path, payloads: Sequence[Mapping[str, Any]]
) -> dict[str, int]:
    """Append transitions idempotently by ``transition_id``.

    A resumed or re-run case must never double-bill the training ledger:
    records whose ``transition_id`` already exists in the file are counted
    and skipped, never re-written.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    existing: set[str] = set()
    for doc in iter_transition_docs(target):
        transition_id = doc.get("transition_id")
        if isinstance(transition_id, str):
            existing.add(transition_id)
    appended = 0
    duplicates = 0
    with target.open("a", encoding="utf-8") as handle:
        for payload in payloads:
            transition_id = payload.get("transition_id")
            if isinstance(transition_id, str) and transition_id in existing:
                duplicates += 1
                continue
            handle.write(json.dumps(dict(payload), sort_keys=True) + "\n")
            if isinstance(transition_id, str):
                existing.add(transition_id)
            appended += 1
    return {"appended": appended, "duplicates_skipped": duplicates}


def iter_transition_docs(path: str | Path) -> Iterator[dict[str, Any]]:
    source = Path(path)
    if not source.is_file():
        return
    with source.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def write_case_sidecar(round_dir: str | Path, case: Mapping[str, Any]) -> Path:
    """Persist per-case static data for offline feature rebuilding."""
    cases_root = Path(round_dir) / CASES_DIR_NAME
    cases_root.mkdir(parents=True, exist_ok=True)
    reaction_id = str(case["reaction_id"])
    path = cases_root / f"{reaction_id}.json"
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != dict(case):
            raise TransitionStoreError(
                f"case sidecar conflict for {reaction_id}: refusing to overwrite"
            )
        return path
    path.write_text(
        json.dumps(dict(case), sort_keys=True, indent=2), encoding="utf-8"
    )
    return path


def load_case_sidecars(round_dir: str | Path) -> dict[str, dict[str, Any]]:
    cases_root = Path(round_dir) / CASES_DIR_NAME
    sidecars: dict[str, dict[str, Any]] = {}
    if not cases_root.is_dir():
        return sidecars
    for path in sorted(cases_root.glob("*.json")):
        sidecars[path.stem] = json.loads(path.read_text(encoding="utf-8"))
    return sidecars


def _episode_index_path(round_dir: str | Path) -> Path:
    return Path(round_dir) / "transitions" / EPISODE_INDEX_FILE_NAME


def _episode_entry_from_docs(
    *,
    episode_id: str,
    round_id: str,
    reaction_id: str,
    transition_docs: Sequence[Mapping[str, Any]],
    policy_id: str,
    policy_hash: str,
    branch: str | None,
) -> dict[str, Any]:
    ordered = sorted(transition_docs, key=lambda d: int(d["decision_index"]))
    indices = [int(d["decision_index"]) for d in ordered]
    if indices != list(range(len(indices))):
        raise TransitionStoreError(
            f"episode {episode_id}: decision indices {indices} are not "
            "contiguous from zero; refusing to register a broken chain"
        )
    compatible_links: list[list[int]] = []
    for current, following in zip(ordered, ordered[1:]):
        if (
            current["state_after_geometry"]
            == following["decision_start_geometry"]
        ):
            compatible_links.append(
                [int(current["decision_index"]), int(following["decision_index"])]
            )
    return {
        "episode_id": episode_id,
        "round_id": round_id,
        "reaction_id": reaction_id,
        "branch": branch,
        "policy_id": policy_id,
        "policy_hash": policy_hash,
        "transition_ids": [str(d["transition_id"]) for d in ordered],
        "decision_indices": indices,
        "compatible_links": compatible_links,
        "n_decisions": len(ordered),
        "terminal_status": str(ordered[-1]["episode_status"]),
    }


def register_episode(
    round_dir: str | Path,
    *,
    episode_id: str,
    round_id: str,
    reaction_id: str,
    transition_docs: Sequence[Mapping[str, Any]],
    policy_id: str,
    policy_hash: str,
    branch: str | None = None,
) -> dict[str, Any]:
    """Append one episode to the round's episode-index sidecar (idempotent).

    The index is the training-side episode identity: successors link only
    inside one episode, a repeated registration is a no-op, and a
    ``transition_id`` may never belong to two episodes.
    """
    entry = _episode_entry_from_docs(
        episode_id=episode_id,
        round_id=round_id,
        reaction_id=reaction_id,
        transition_docs=transition_docs,
        policy_id=policy_id,
        policy_hash=policy_hash,
        branch=branch,
    )
    path = _episode_index_path(round_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    document: dict[str, Any] = {
        "schema_version": EPISODE_INDEX_SCHEMA_VERSION,
        "episodes": [],
    }
    if path.is_file():
        document = json.loads(path.read_text(encoding="utf-8"))
        if document.get("schema_version") != EPISODE_INDEX_SCHEMA_VERSION:
            raise TransitionStoreError(
                f"episode index schema {document.get('schema_version')!r} "
                f"unsupported at {path}"
            )
    episodes: list[dict[str, Any]] = list(document.get("episodes", []))
    known_ids = {str(e["episode_id"]) for e in episodes}
    if episode_id in known_ids:
        return entry
    owner: dict[str, set[str]] = {}
    for existing in episodes:
        for tid in existing.get("transition_ids", []):
            owner.setdefault(str(tid), set()).add(str(existing["episode_id"]))
    for tid in entry["transition_ids"]:
        if tid in owner:
            raise TransitionStoreError(
                f"transition {tid} already belongs to episode(s) "
                f"{sorted(owner[tid])}; episodes must never share transitions"
            )
    episodes.append(entry)
    document["episodes"] = episodes
    tmp = path.with_suffix(path.suffix + f".part-{os.getpid()}")
    tmp.write_text(
        json.dumps(document, sort_keys=True, indent=2), encoding="utf-8"
    )
    os.replace(tmp, path)
    return entry


def count_episodes_for_reaction(round_dir: str | Path, reaction_id: str) -> int:
    path = _episode_index_path(round_dir)
    if not path.is_file():
        return 0
    document = json.loads(path.read_text(encoding="utf-8"))
    return sum(
        1
        for entry in document.get("episodes", [])
        if str(entry.get("reaction_id")) == str(reaction_id)
    )


def load_episode_transition_map(
    round_dir: str | Path,
) -> dict[str, dict[str, Any]]:
    """``transition_id`` -> trainer-facing episode metadata.

    Each value carries the unique ``episode_id`` and whether the incoming
    successor link (previous decision inside the same episode) is
    geometrically compatible; incompatible links must not be bootstrapped.
    """
    path = _episode_index_path(round_dir)
    if not path.is_file():
        return {}
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("schema_version") != EPISODE_INDEX_SCHEMA_VERSION:
        raise TransitionStoreError(
            f"episode index schema {document.get('schema_version')!r} "
            f"unsupported at {path}"
        )
    mapping: dict[str, dict[str, Any]] = {}
    for entry in document.get("episodes", []):
        links = [
            (int(from_idx), int(to_idx))
            for from_idx, to_idx in entry.get("compatible_links", [])
        ]
        compatible_sources = {from_idx for from_idx, _to in links}
        compatible_targets = {to for _from, to in links}
        for position, tid in enumerate(entry.get("transition_ids", [])):
            if tid in mapping:
                raise TransitionStoreError(
                    f"transition {tid} appears in multiple episodes in {path}"
                )
            mapping[tid] = {
                "episode_id": str(entry["episode_id"]),
                "round_id": str(entry.get("round_id", "")),
                "branch": entry.get("branch"),
                "position": position,
                "incoming_link_compatible": (
                    position == 0 or position in compatible_targets
                ),
                "successor_indices": (
                    {position + 1} if position in compatible_sources else set()
                ),
            }
    return mapping


@dataclass(frozen=True)
class QualifiedTransition:
    """One training-ready transition plus its case sidecar.

    ``episode`` carries the episode-index metadata when the round store has
    an :data:`EPISODE_INDEX_FILE_NAME` sidecar covering this transition:
    ``episode_id`` (globally unique), the position inside the episode, and
    whether the incoming successor link (previous decision -> this one) is
    geometrically compatible.  Legacy stores without an index leave it
    ``None`` and keep the historical same-reaction successor semantics.
    """

    doc: dict[str, Any]
    case: dict[str, Any]
    episode: Mapping[str, Any] | None = None

    @property
    def reaction_id(self) -> str:
        return str(self.doc["reaction_id"])

    @property
    def episode_id(self) -> str | None:
        if self.episode is None:
            return None
        return str(self.episode["episode_id"])

    @property
    def incoming_link_compatible(self) -> bool:
        if self.episode is None:
            return True
        return bool(self.episode["incoming_link_compatible"])

    @property
    def episode_successor_key_base(self) -> str:
        return self.episode_id if self.episode_id is not None else self.reaction_id

    @property
    def candidates(self) -> tuple[tuple[float, ...], ...]:
        return tuple(
            tuple(float(v) for v in candidate) for candidate in self.doc["candidates"]
        )

    @property
    def executed_index(self) -> int:
        return int(self.doc["executed_candidate_index"])

    @property
    def reward(self) -> float:
        return float(self.doc["reward"])

    @property
    def terminal(self) -> bool:
        return str(self.doc["episode_status"]).startswith("terminal")

    @property
    def bootstrap_mask(self) -> bool:
        return bool(self.doc["bootstrap_mask"])

    @property
    def next_candidates(self) -> tuple[tuple[float, ...], ...]:
        return tuple(
            tuple(float(v) for v in candidate)
            for candidate in self.doc.get("next_candidates", [])
        )

    def state_input(self) -> Any:
        from pes2ts_core.pesn.contracts.inputs import (
            build_pesn_input,
        )
        from pes2ts_core.generation.planning.policy_adapter import (
            coordinate_spec_from_payload,
            task_graph_from_payload,
        )

        sidecar = self.case
        task_graph = task_graph_from_payload(sidecar["task_graph"])
        coordinate_spec = coordinate_spec_from_payload(sidecar["coordinate_spec"])
        return build_pesn_input(
            input_id=f"{self.reaction_id}#d{self.doc['decision_index']}",
            task_graph=task_graph,
            coordinate_spec=coordinate_spec,
            coordinates=self.doc["decision_start_geometry"],
            charge=sidecar.get("charge"),
            multiplicity=sidecar.get("multiplicity"),
            method="gfn2_xtb_e_native",
        )


def load_transition_set(
    round_dir: str | Path,
    *,
    environment_hash: str,
    allowed_reactions: Sequence[str] | None = None,
) -> "TransitionLoadResult":
    """Load + qualify transitions from one round directory, with rejection stats.

    Qualification: schema/structure validity, environment-hash match (data
    from a different frozen environment is never silently mixed), the
    reaction allow-list (train-qualified cases only — a valid-mixing attempt
    is typed as ``reaction_not_allowed``), and per-case sidecar presence.
    Manually censored episodes are typed ``censored_manual`` and never
    fabricated into training data.
    """
    root = Path(round_dir)
    transitions_path = root / "transitions" / TRANSITIONS_FILE_NAME
    sidecars = load_case_sidecars(root)
    episode_map = load_episode_transition_map(root)
    allowed = set(allowed_reactions) if allowed_reactions is not None else None
    qualified: list[QualifiedTransition] = []
    rejected: dict[str, int] = {}
    seen_ids: set[str] = set()

    def reject(code: str) -> None:
        rejected[code] = rejected.get(code, 0) + 1

    for doc in iter_transition_docs(transitions_path):
        transition_id = doc.get("transition_id")
        if isinstance(transition_id, str):
            if transition_id in seen_ids:
                reject("duplicate_transition_id")
                continue
            seen_ids.add(transition_id)
        try:
            validate_transition_doc(doc)
        except TransitionStoreError:
            status = doc.get("episode_status")
            if status == "censored_manual":
                reject("censored_manual")
            else:
                reject("structure_invalid")
            continue
        if doc["environment_hash"] != environment_hash:
            reject("environment_mismatch")
            continue
        if allowed is not None and doc["reaction_id"] not in allowed:
            reject("reaction_not_allowed")
            continue
        case = sidecars.get(str(doc["reaction_id"]))
        if case is None:
            reject("case_sidecar_missing")
            continue
        episode = episode_map.get(str(transition_id))
        qualified.append(
            QualifiedTransition(
                doc=doc, case=case, episode=_freeze_episode_meta(episode)
            )
        )
    return TransitionLoadResult(qualified=qualified, rejected=rejected)


def _freeze_episode_meta(
    episode: Mapping[str, Any] | None,
) -> Mapping[str, Any] | None:
    if episode is None:
        return None
    return {
        "episode_id": str(episode["episode_id"]),
        "round_id": str(episode.get("round_id", "")),
        "branch": episode.get("branch"),
        "position": int(episode["position"]),
        "incoming_link_compatible": bool(episode["incoming_link_compatible"]),
        "successor_indices": frozenset(
            int(idx) for idx in episode.get("successor_indices", ())
        ),
    }


@dataclass(frozen=True)
class TransitionLoadResult:
    """Qualified transitions plus the typed rejection histogram."""

    qualified: list[QualifiedTransition]
    rejected: dict[str, int]


def load_qualified_transitions(
    round_dir: str | Path,
    *,
    environment_hash: str,
    allowed_reactions: Sequence[str] | None = None,
) -> list[QualifiedTransition]:
    """Backward-compatible list view of :func:`load_transition_set`."""
    return load_transition_set(
        round_dir,
        environment_hash=environment_hash,
        allowed_reactions=allowed_reactions,
    ).qualified


def transition_store_summary(
    transitions: Sequence[QualifiedTransition],
) -> dict[str, Any]:
    """Honest coverage/support summary for a qualified transition set."""
    n = len(transitions)
    executed_sources: dict[str, int] = {}
    n_bootstrap = 0
    n_terminal = 0
    n_candidate_support = 0
    for transition in transitions:
        source = str(transition.doc["action"]["source"])
        executed_sources[source] = executed_sources.get(source, 0) + 1
        n_bootstrap += int(transition.bootstrap_mask)
        n_terminal += int(transition.terminal)
        n_candidate_support += int(len(transition.candidates) > 1)
    return {
        "n_transitions": n,
        "n_reactions": len({t.reaction_id for t in transitions}),
        "executed_action_sources": executed_sources,
        "n_bootstrap_mask": n_bootstrap,
        "n_terminal": n_terminal,
        "n_with_multi_candidate_support": n_candidate_support,
    }


def context_vector_from_doc(doc: Mapping[str, Any]) -> tuple[float, ...]:
    """Rebuild the execution-context vector recorded with the transition."""
    from pes2ts_core.pesn.contracts.control import CONTROL_CONTEXT_DIM

    context = doc.get("decision_context")
    if isinstance(context, Mapping) and "to_vector" not in context:
        values = tuple(float(v) for v in context["vector"])
    elif isinstance(context, Sequence):
        values = tuple(float(v) for v in context)
    else:
        values = tuple(0.0 for _ in range(CONTROL_CONTEXT_DIM))
    if len(values) != CONTROL_CONTEXT_DIM:
        raise TransitionStoreError("context vector dimension mismatch")
    return values


__all__ = [
    "CASES_DIR_NAME",
    "EPISODE_INDEX_FILE_NAME",
    "EPISODE_INDEX_SCHEMA_VERSION",
    "TRANSITIONS_FILE_NAME",
    "QualifiedTransition",
    "TransitionLoadResult",
    "TransitionStoreError",
    "append_transitions",
    "context_vector_from_doc",
    "count_episodes_for_reaction",
    "iter_transition_docs",
    "load_case_sidecars",
    "load_episode_transition_map",
    "load_qualified_transitions",
    "load_transition_set",
    "register_episode",
    "transition_store_summary",
    "validate_transition_doc",
    "write_case_sidecar",
]
