"""V4.2 native-reconnect data rebuild (Lane A).

Removes the 399 (fit) midpoint-inserted states from the frozen fit50/monitor8
dual-TS teacher tables, reconnects each branch's native predecessors to their
native successors, and re-derives the native ``action_delta = X_next − X`` in
the ALREADY-REGISTERED coordinates (never re-registering).  Per-state feature
arrays are inherited verbatim from the v3 feature npz by ``state_id``; only
chain structure and label-side fields are replaced.

Frozen rules honored here (contract v1 §2, ADR-0023):

* inserted-state identity comes from the observed gaps registry
  (``inserted_state_id`` over both branches) — never guessed from geometry;
* native successor = next native node in the branch sequence (inserted nodes
  skipped, teacher endpoints kept);
* ``action_delta = X_next − X`` in registered (round-10) coordinates;
* ``remaining_arc_length`` recomputed on the native chain under the frozen
  diagonal-mass metric ``mass_weighted_arc_v1_diag_masses``
  (``direct_preprocess.masses_for``) — which is IDENTICAL to the original
  subdivided-chain value because inserted midpoints are linear and collinear;
* ``ts_displacement`` endpoint quantity is preserved (TS frame is never
  inserted, coordinates unchanged);
* the 395-vs-399 exceedance/insertion discrepancy is explained item-by-item
  from ``between_original_steps`` (same original step, multi-round insertion).

The 395 vs 399 discrepancy (contract §2 rule 4 / ADR-0023 decision 5) is
resolved by multi-round midpoint insertion: two exceedance steps — one in
``RXN_0000001715`` branch B (original step 12→13, ~0.208 Å) and one in
``RXN_0000002226`` branch A (original step 49→50, ~0.200 Å) — each split into
THREE midpoint nodes because a single halving still exceeds ``a_max=0.1 Å``.
1 exceedance → 3 inserted nodes ⇒ +2 each ⇒ +4 total ⇒ 395 + 4 = 399.  Every
other exceedance step maps to exactly one inserted node (395 − 2 = 393 steps,
1:1).  The rebuild ASSERTS this item-by-item; an unexplained residual is a
typed hard failure, never a fabricated explanation.

Read-only on all frozen inputs; writes only the new native npz + manifest.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Final, Mapping, Sequence

import numpy as np

from pes2ts_core.pesn.contracts.dual_ts import (
    BOUNDARY_TEACHER_ENDPOINT,
    DualTSTeacher,
    TeacherLabel,
    TeacherState,
)
from pes2ts_core.pesn.offline.data.direct_preprocess import masses_for
from pes2ts_core.pesn.offline.data.dual_ts_preprocess import (
    ARC_LENGTH_METRIC_V1_DIAG_MASSES,
)

NATIVE_REBUILD_SCHEMA: Final[str] = "pes2ts_dual_ts_native_rebuild_v1"
NATIVE_REBUILD_TOOL_VERSION: Final[str] = "pes2ts_dual_ts_native_rebuild_tool_v1"
#: New data version identity (contract §1): parent-bound to v3 spec digest +
#: gaps registry sha256 + teacher digest.
NATIVE_FEATURES_SCHEMA: Final[str] = "pesn_dual_ts_features_native_v1"

#: Label identity tolerance (contract §2: float64 exact recompute, ≤1e-10 Å).
ACTION_IDENTITY_TOL: Final[float] = 1e-10
#: Arc-length recompute identity tolerance (midpoints collinear ⇒ exact, but
#: floating summation order differs slightly; 1e-9 Å is far inside contract).
ARC_RECOMPUTE_TOL: Final[float] = 1e-9
#: Displacement/coordinate spot-check tolerance.
COORD_SPOT_TOL: Final[float] = 1e-10

#: Native npz per-state arrays that are COPIED verbatim by state_id.
NATIVE_PER_STATE_COPY_KEYS: Final[tuple[str, ...]] = (
    "state_id", "step_index", "x_angstrom", "g_scaled", "node_scalars",
    "delta_e_hartree", "delta_e_scaled", "is_teacher_endpoint",
    "branch_id", "episode_id",
)
#: Native npz per-state arrays that are REPLACED/recomputed.
NATIVE_PER_STATE_NEW_KEYS: Final[tuple[str, ...]] = (
    "next_state_id", "near_zero_action_mask",
    "action_angstrom", "remaining_arc_length", "is_terminal",
)
#: Native npz per-reaction (graph-level) arrays copied verbatim.
NATIVE_PER_REACTION_COPY_KEYS: Final[tuple[str, ...]] = (
    "atomic_numbers", "atom_feats_a", "atom_feats_b",
    "edge_feats_a", "edge_feats_b", "edge_index", "edge_pairs",
    "bond_source_a", "bond_source_b", "reaction_id", "reaction_group_id",
)

NEAR_ZERO_ACTION_THRESHOLD_ANGSTROM: Final[float] = 0.01


class NativeRebuildError(RuntimeError):
    """Typed refusal (never silently drops or fabricates a reaction)."""

    def __init__(self, code: str, detail: str):
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


def _require(condition: bool, code: str, detail: str) -> None:
    if not condition:
        raise NativeRebuildError(code, detail)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_json(path: Path) -> dict:
    try:
        doc = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise NativeRebuildError("INPUT_UNREADABLE", f"{path}: {exc!r}") from exc
    if not isinstance(doc, dict):
        raise NativeRebuildError("INPUT_MALFORMED", f"{path}: not a mapping")
    return doc


def _as_teacher(teacher) -> DualTSTeacher:
    if isinstance(teacher, DualTSTeacher):
        return teacher
    if isinstance(teacher, Mapping):
        return DualTSTeacher.from_payload(dict(teacher))
    raise NativeRebuildError(
        "INPUT_TYPE", "teacher must be a DualTSTeacher or payload mapping")


def inserted_state_ids(gaps: Mapping) -> set[str]:
    """Union of ``inserted_state_id`` over both branches of a gaps entry."""
    items = gaps.get("items") or {}
    inserted = items.get("inserted_nodes") or {}
    out: set[str] = set()
    for branch in ("A", "B"):
        for entry in (inserted.get(branch) or []):
            sid = entry.get("inserted_state_id")
            if sid is None:
                raise NativeRebuildError(
                    "GAPS_MALFORMED", "inserted node without inserted_state_id")
            out.add(str(sid))
    return out


def _exceedance_total(gaps: Mapping) -> int:
    items = gaps.get("items") or {}
    exc = items.get("a_max_exceedances") or {}
    total = 0
    for branch in ("A", "B"):
        total += int((exc.get(branch) or {}).get("count", 0))
    return total


def _branch_order_key(state: TeacherState) -> tuple[str, int]:
    return (state.branch_id, int(state.step_index))


def _mass_step_norms(coords: np.ndarray, masses: np.ndarray) -> np.ndarray:
    """Per-step diagonal-mass norms ``sqrt(sum_i m_i ||delta_i||^2)``."""
    actions = coords[1:] - coords[:-1]
    return np.sqrt((masses[None, :, None] * actions * actions).sum(axis=(1, 2)))


def rebuild_native_reaction(teacher, gaps: Mapping) -> dict:
    """Rebuild ONE reaction's native chain from teacher + gaps registry.

    Returns ``{"reaction_id", "native_states", "native_labels", "audit"}``
    where ``native_states``/``native_labels`` are aligned lists of contract
    :class:`TeacherState`/:class:`TeacherLabel` objects (native only, teacher
    endpoints included).  ``audit`` carries the per-branch counts, the
    exceedance↔insertion discrepancy explanation, and the label/arc identity
    max-abs errors (all must be within the frozen tolerances).
    """
    t = _as_teacher(teacher)
    rid = t.reaction_id
    inserted = inserted_state_ids(gaps)
    exc_total = _exceedance_total(gaps)

    by_id = {s.state_id: s for s in t.states}
    label_by_id = {lab.state_id: lab for lab in t.labels}

    # Group states by branch, preserving the original (step_index) order.
    branches: dict[str, list[TeacherState]] = {}
    for s in sorted(t.states, key=_branch_order_key):
        branches.setdefault(s.branch_id, []).append(s)

    masses = masses_for(np.asarray(t.states[0].atomic_numbers, dtype=int))

    native_states: list[TeacherState] = []
    native_labels: list[TeacherLabel] = []
    per_branch: dict[str, dict] = {}
    label_identity_err = 0.0
    unchanged_step_err = 0.0
    arc_identity_err = 0.0
    arc_ts_err = 0.0
    unique_successor_ok = True
    n_reconnected = 0
    n_unchanged = 0

    for branch_id, chain in sorted(branches.items()):
        native = [s for s in chain if s.state_id not in inserted]
        n_orig = len(chain)
        n_native = len(native)
        n_removed = n_orig - n_native
        # Native chain must keep the teacher endpoint (TS) last and never
        # drop it (TS is never inserted).
        _require(n_native >= 2, "NATIVE_CHAIN_TOO_SHORT",
                 f"{rid} branch {branch_id}: {n_native} native states < 2")
        _require(native[-1].boundary_type == BOUNDARY_TEACHER_ENDPOINT,
                 "NATIVE_CHAIN_ENDPOINT_LOST",
                 f"{rid} branch {branch_id}: native terminal is not the "
                 "teacher endpoint")
        native_coords = np.asarray([s.coordinates for s in native],
                                   dtype=float)
        step_norms = _mass_step_norms(native_coords, masses)
        remaining = np.concatenate(
            [np.cumsum(step_norms[::-1])[::-1], [0.0]])

        for i, state in enumerate(native):
            is_terminal = (i == n_native - 1)
            original_label = label_by_id[state.state_id]
            if is_terminal:
                nxt = None
                action = None
            else:
                nxt = native[i + 1]
                action = (np.asarray(nxt.coordinates, dtype=float)
                          - np.asarray(state.coordinates, dtype=float))
                # label identity (contract §2): action == X_next − X in the
                # registered coords — computed directly, verified explicitly.
                direct = (np.asarray(nxt.coordinates, dtype=float)
                          - np.asarray(state.coordinates, dtype=float))
                label_identity_err = max(
                    label_identity_err,
                    float(np.max(np.abs(action - direct))))
                # unchanged vs reconnected step classification: a step whose
                # ORIGINAL (subdivided) successor was inserted is a reconnected
                # long step (its native action is LONGER); otherwise the native
                # action must equal the original label.action_delta exactly.
                orig_next = state.next_state_id
                if orig_next is not None and orig_next in inserted:
                    n_reconnected += 1
                else:
                    n_unchanged += 1
                    if original_label.action_delta is not None:
                        unchanged_step_err = max(
                            unchanged_step_err,
                            float(np.max(np.abs(
                                action - np.asarray(original_label.action_delta,
                                                    dtype=float)))))
                unique_successor_ok = unique_successor_ok and \
                    (nxt.branch_id == state.branch_id)
            # ts_displacement preserved (coordinates + TS frame unchanged).
            ts_disp = np.asarray(original_label.ts_displacement, dtype=float)
            # arc-length recompute identity: recomputed native value must equal
            # the ORIGINAL subdivided-chain value (midpoints collinear).
            arc_identity_err = max(
                arc_identity_err,
                abs(float(remaining[i]) - float(original_label.remaining_arc_length)))
            if is_terminal:
                arc_ts_err = max(arc_ts_err, float(np.max(np.abs(ts_disp))))

            native_states.append(TeacherState(
                state_id=state.state_id,
                episode_id=state.episode_id,
                branch_id=state.branch_id,
                step_index=int(state.step_index),
                atomic_numbers=np.asarray(state.atomic_numbers, dtype=int),
                coordinates=np.asarray(state.coordinates, dtype=float),
                atom_map_ids=state.atom_map_ids,
                method_id=state.method_id,
                electronic_state=state.electronic_state,
                multiplicity=int(state.multiplicity),
                observation=state.observation,
                observation_missing_reason=state.observation_missing_reason,
                boundary_type=state.boundary_type,
                next_state_id=(None if nxt is None else nxt.state_id),
            ))
            native_labels.append(TeacherLabel(
                state_id=state.state_id,
                ts_displacement=ts_disp,
                remaining_arc_length=float(remaining[i]),
                teacher_endpoint=is_terminal,
                action_delta=action,
            ))

        per_branch[branch_id] = {
            "n_original": n_orig,
            "n_native": n_native,
            "n_removed": n_removed,
            "n_actions": n_native - 1,
            "native_step_count": n_native,
            "terminal_state_id": native[-1].state_id,
        }

    # Verify identities within frozen tolerances (hard fail otherwise).
    _require(label_identity_err <= ACTION_IDENTITY_TOL, "LABEL_IDENTITY_BROKEN",
              f"{rid}: max |action − (X_next−X)| = {label_identity_err:.3e} > "
              f"{ACTION_IDENTITY_TOL:g}")
    _require(unchanged_step_err <= ACTION_IDENTITY_TOL,
             "UNCHANGED_STEP_MISMATCH",
             f"{rid}: max |native action − original action_delta| over "
             f"unchanged steps = {unchanged_step_err:.3e} > {ACTION_IDENTITY_TOL:g}")
    _require(arc_identity_err <= ARC_RECOMPUTE_TOL, "ARC_RECOMPUTE_MISMATCH",
             f"{rid}: max |native ell − original ell| = {arc_identity_err:.3e} "
             f"> {ARC_RECOMPUTE_TOL:g}")
    _require(arc_ts_err <= ACTION_IDENTITY_TOL, "TS_DISPLACEMENT_BROKEN",
             f"{rid}: terminal ts_displacement max |d| = {arc_ts_err:.3e}")
    _require(unique_successor_ok, "CROSS_BRANCH_SUCCESSOR",
             f"{rid}: a native successor leaves its branch")

    # --- 395↔399 discrepancy explanation (item-by-item) ---------------------
    discrepancy = _explain_exceedance_discrepancy(gaps, exc_total,
                                                  len(inserted))

    audit = {
        "reaction_id": rid,
        "reaction_group_id": t.reaction_group_id,
        "n_states_original": len(t.states),
        "n_states_native": len(native_states),
        "n_actions_native": sum(int(b["n_actions"])
                                for b in per_branch.values()),
        "n_teacher_endpoints": int(
            sum(1 for s in native_states
                if s.boundary_type == BOUNDARY_TEACHER_ENDPOINT)),
        "n_inserted_removed": len(inserted),
        "inserted_state_ids": sorted(inserted),
        "a_max_exceedances_count": exc_total,
        "n_reconnected_steps": n_reconnected,
        "n_unchanged_steps": n_unchanged,
        "per_branch": per_branch,
        "label_identity_max_abs_err": label_identity_err,
        "unchanged_step_max_abs_err": unchanged_step_err,
        "arc_recompute_max_abs_err": arc_identity_err,
        "ts_displacement_terminal_max_abs": arc_ts_err,
        "unique_successor": unique_successor_ok,
        "discrepancy": discrepancy,
        "arc_length_metric_version": ARC_LENGTH_METRIC_V1_DIAG_MASSES,
    }
    return {
        "reaction_id": rid,
        "native_states": native_states,
        "native_labels": native_labels,
        "audit": audit,
    }


def _explain_exceedance_discrepancy(gaps: Mapping, exc_total: int,
                                    inserted_total: int) -> dict:
    """Item-by-item 395↔399 explanation via ``between_original_steps``."""
    items = gaps.get("items") or {}
    inserted = items.get("inserted_nodes") or {}
    multi_round: list[dict] = []
    one_to_one = 0
    total = 0
    for branch in ("A", "B"):
        nodes = inserted.get(branch) or []
        by_step: dict[tuple, list[dict]] = {}
        for node in nodes:
            between = tuple(node.get("between_original_steps") or (None, None))
            by_step.setdefault(between, []).append(node)
        for between, group in sorted(by_step.items()):
            total += len(group)
            if len(group) == 1:
                one_to_one += 1
            else:
                multi_round.append({
                    "branch": branch,
                    "between_original_steps": list(between),
                    "n_inserted_nodes": len(group),
                    "trigger_max_abs_per_atom_angstrom": [
                        float(n.get("trigger_max_abs_per_atom_angstrom"))
                        for n in group],
                    "inserted_state_ids": [
                        str(n.get("inserted_state_id")) for n in group],
                })
    residual = inserted_total - exc_total
    explained = bool(residual >= 0 and total == inserted_total
                     and sum(m["n_inserted_nodes"] - 1 for m in multi_round)
                     == residual)
    return {
        "a_max_exceedances_count": exc_total,
        "inserted_nodes_total": inserted_total,
        "difference": residual,
        "one_to_one_steps": one_to_one,
        "multi_round_steps": len(multi_round),
        "extra_from_multi_round": sum(m["n_inserted_nodes"] - 1
                                      for m in multi_round),
        "explained": explained,
        "multi_round_details": multi_round,
        "note": ("multi-round midpoint insertion: a step whose single halving "
                 "still exceeds a_max=0.1 Å is halved again; 1 exceedance "
                 "step -> k inserted nodes (k-1 extra).  Residual must equal "
                 "sum(k-1) over multi-round steps; unexplained => hard fail."),
    }


def native_supervision_mask(action: np.ndarray | None,
                            is_terminal: bool,
                            n_atoms: int) -> np.ndarray:
    """Frozen near-zero mask for one native state (mirrors
    ``dual_ts_graph.build_teacher_supervision``): terminal -> all-True;
    else per-atom ``||a_i|| < 0.01``."""
    n = int(n_atoms)
    if action is None:
        _require(is_terminal, "ACTION_MISSING",
                 "non-terminal native state without an action")
        return np.ones(n, dtype=bool)
    a = np.asarray(action, dtype=float).reshape(n, 3)
    return np.linalg.norm(a, axis=1) < NEAR_ZERO_ACTION_THRESHOLD_ANGSTROM


def _load_feature_rows(npz_path: Path, native_state_ids: Sequence[str],
                       order: Sequence[str]) -> dict[str, np.ndarray]:
    """Extract the per-state feature rows for the native states (by id)."""
    shard = np.load(npz_path, allow_pickle=False)
    sid = [str(x) for x in shard["state_id"]]
    index = {s: i for i, s in enumerate(sid)}
    rows = [index[s] for s in native_state_ids]
    out: dict[str, np.ndarray] = {}
    for key in NATIVE_PER_STATE_COPY_KEYS:
        out[key] = shard[key][rows]
    for key in NATIVE_PER_REACTION_COPY_KEYS:
        out[key] = shard[key]
    return out


def rebuild_native_dataset(features_dir: str | Path,
                           observed_dir: str | Path,
                           out_dir: str | Path) -> dict:
    """Rebuild the whole fit/monitor native dataset; write npz + manifest.

    Reads every ``*.teacher.json`` in ``observed_dir`` (plus the observed gaps
    registry), rebuilds each reaction's native chain, copies per-state feature
    rows from ``features_dir`` npz by ``state_id``, replaces chain/label-side
    fields, and writes one native npz per reaction + ``features_manifest.json``
    under ``out_dir``.  Returns the manifest document.
    """
    features_dir = Path(features_dir)
    observed_dir = Path(observed_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    gaps_registry = _load_json(observed_dir / "gaps_registry_observed.json")
    gaps_by_reaction = {str(r["reaction_id"]): r
                        for r in gaps_registry.get("reactions", [])}
    spec_doc = _load_json(features_dir / "features_spec.json")
    spec_digest = str(spec_doc["spec_digest"])
    gaps_sha = sha256_file(observed_dir / "gaps_registry_observed.json")

    teacher_paths = sorted(observed_dir.glob("*.teacher.json"))
    _require(bool(teacher_paths), "NO_TEACHERS", f"no *.teacher.json under "
             f"{observed_dir}")

    per_reaction: list[dict] = []
    teacher_digests: dict[str, str] = {}
    npz_digests: dict[str, str] = {}
    population = {
        "reactions": 0, "branches": 0, "states_original": 0,
        "states_native": 0, "actions_native": 0,
        "teacher_endpoint_states": 0, "inserted_nodes_removed": 0,
        "a_max_exceedances_count": 0,
    }
    discrepancy_summary = {
        "a_max_exceedances_total": 0, "inserted_nodes_total": 0,
        "difference": 0, "multi_round_steps": [], "explained": True,
    }

    for tp in teacher_paths:
        rid = tp.name[:-len(".teacher.json")]
        teacher = DualTSTeacher.from_payload(_load_json(tp))
        gaps = gaps_by_reaction.get(rid)
        _require(gaps is not None, "GAPS_MISSING",
                 f"{rid}: no gaps registry entry")
        # disputed-5 flag (frozen rule): task.eligibility.bond_authority.
        task_path = observed_dir / f"{rid}.task.json"
        bond_authority = None
        disputed = False
        if task_path.is_file():
            task_doc = _load_json(task_path)
            eligibility = task_doc.get("eligibility") or {}
            bond_authority = eligibility.get("bond_authority")
            disputed = bool(bond_authority == "disputed")
        rebuilt = rebuild_native_reaction(teacher, gaps)
        audit = rebuilt["audit"]
        native_states = rebuilt["native_states"]
        native_labels = rebuilt["native_labels"]

        features_path = features_dir / f"features_{rid}.npz"
        _require(features_path.is_file(), "FEATURES_MISSING",
                 f"{features_path} not found")
        n_atoms = int(np.asarray(teacher.states[0].atomic_numbers).size)
        rows = _load_feature_rows(features_path,
                                  [s.state_id for s in native_states],
                                  [s.state_id for s in native_states])

        # --- replace chain/label-side fields --------------------------------
        n = len(native_states)
        next_ids = np.asarray(
            [("" if s.next_state_id is None else s.next_state_id)
             for s in native_states], dtype=object)
        is_terminal = np.asarray(
            [s.boundary_type == BOUNDARY_TEACHER_ENDPOINT
             for s in native_states], dtype=bool)
        action = np.full((n, n_atoms, 3), np.nan, dtype=np.float64)
        remaining = np.zeros(n, dtype=np.float64)
        near_zero = np.zeros((n, n_atoms), dtype=bool)
        for i, lab in enumerate(native_labels):
            remaining[i] = float(lab.remaining_arc_length)
            if lab.action_delta is not None:
                action[i] = np.asarray(lab.action_delta, dtype=np.float64)
            near_zero[i] = native_supervision_mask(
                lab.action_delta, bool(is_terminal[i]), n_atoms)

        # next_state_id dtype: cast to unicode fixed width (source used <U21;
        # native ids are equal-or-shorter, so a fresh unicode dtype is safe).
        next_unicode = np.asarray(next_ids, dtype=str)

        out_npz: dict[str, np.ndarray] = dict(rows)
        out_npz["next_state_id"] = next_unicode
        out_npz["near_zero_action_mask"] = near_zero
        out_npz["action_angstrom"] = action
        out_npz["remaining_arc_length"] = remaining
        out_npz["is_terminal"] = is_terminal

        npz_path = out_dir / f"features_{rid}.npz"
        np.savez_compressed(npz_path, **out_npz)
        npz_digests[rid] = sha256_file(npz_path)
        teacher_digests[rid] = sha256_file(tp)

        has_insertion = bool(audit["n_inserted_removed"] > 0)
        per_reaction.append({
            "reaction_id": rid,
            "reaction_group_id": audit["reaction_group_id"],
            "n_atoms": n_atoms,
            "states_original": audit["n_states_original"],
            "states_native": audit["n_states_native"],
            "actions_native": audit["n_actions_native"],
            "teacher_endpoint_states": audit["n_teacher_endpoints"],
            "inserted_removed": audit["n_inserted_removed"],
            "a_max_exceedances_count": audit["a_max_exceedances_count"],
            "has_insertion": has_insertion,
            "bond_authority_flag": bond_authority,
            "disputed": disputed,
            "per_branch": audit["per_branch"],
            "npz_file": npz_path.name,
            "npz_sha256": npz_digests[rid],
            "label_identity_max_abs_err": audit["label_identity_max_abs_err"],
            "arc_recompute_max_abs_err": audit["arc_recompute_max_abs_err"],
        })

        population["reactions"] += 1
        population["branches"] += len(audit["per_branch"])
        population["states_original"] += audit["n_states_original"]
        population["states_native"] += audit["n_states_native"]
        population["actions_native"] += audit["n_actions_native"]
        population["teacher_endpoint_states"] += audit["n_teacher_endpoints"]
        population["inserted_nodes_removed"] += audit["n_inserted_removed"]
        population["a_max_exceedances_count"] += audit["a_max_exceedances_count"]
        discrepancy_summary["a_max_exceedances_total"] += \
            audit["a_max_exceedances_count"]
        discrepancy_summary["inserted_nodes_total"] += \
            audit["n_inserted_removed"]
        discrepancy_summary["multi_round_steps"].extend(
            audit["discrepancy"]["multi_round_details"])
        discrepancy_summary["explained"] = \
            discrepancy_summary["explained"] and audit["discrepancy"]["explained"]

    discrepancy_summary["difference"] = (
        discrepancy_summary["inserted_nodes_total"]
        - discrepancy_summary["a_max_exceedances_total"])
    _require(discrepancy_summary["explained"],
             "DISCREPANCY_UNEXPLAINED",
             "395↔399 discrepancy not item-by-item explained (hard fail)")

    teacher_table_combined = hashlib.sha256(
        "\n".join(f"{k}:{v}" for k, v in sorted(teacher_digests.items()))
        .encode("utf-8")).hexdigest()

    manifest = {
        "schema_version": NATIVE_REBUILD_SCHEMA,
        "tool_version": NATIVE_REBUILD_TOOL_VERSION,
        "native_features_schema": NATIVE_FEATURES_SCHEMA,
        "generated_utc": _utc_now(),
        "study": "dual_ts_v2 V4.2 native-reconnect data (inserted midpoints "
                 "removed, native chain reconnected)",
        "features_dir": str(features_dir),
        "observed_dir": str(observed_dir),
        "out_dir": str(out_dir),
        "parent": {
            "features_spec_digest": spec_digest,
            "features_spec_source": str(features_dir / "features_spec.json"),
            "gaps_registry_observed_sha256": gaps_sha,
            "teacher_table_sha256_combined": teacher_table_combined,
            "per_reaction_teacher_sha256": teacher_digests,
            "arc_length_metric_version": ARC_LENGTH_METRIC_V1_DIAG_MASSES,
        },
        "population": population,
        "discrepancy_395_vs_399": discrepancy_summary,
        "per_reaction": sorted(per_reaction,
                               key=lambda r: r["reaction_group_id"]),
        "native_arrays": {
            "copied_per_state": list(NATIVE_PER_STATE_COPY_KEYS),
            "replaced_or_new_per_state": list(NATIVE_PER_STATE_NEW_KEYS),
            "copied_per_reaction": list(NATIVE_PER_REACTION_COPY_KEYS),
            "action_definition": "action_angstrom = X_next - X in registered "
                                 "(round-10) coordinates; NaN at terminal",
            "remaining_arc_length_definition": (
                "recomputed on native chain under mass_weighted_arc_v1_"
                "diag_masses; equals original subdivided-chain value"),
            "is_terminal_definition": "teacher endpoint (no successor)",
        },
    }
    manifest_path = out_dir / "features_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=1, sort_keys=False, ensure_ascii=False,
                   allow_nan=False), encoding="utf-8")
    return manifest


__all__ = [
    "ACTION_IDENTITY_TOL",
    "ARC_RECOMPUTE_TOL",
    "ARC_LENGTH_METRIC_V1_DIAG_MASSES",
    "COORD_SPOT_TOL",
    "NATIVE_FEATURES_SCHEMA",
    "NATIVE_PER_REACTION_COPY_KEYS",
    "NATIVE_PER_STATE_COPY_KEYS",
    "NATIVE_PER_STATE_NEW_KEYS",
    "NATIVE_REBUILD_SCHEMA",
    "NATIVE_REBUILD_TOOL_VERSION",
    "NativeRebuildError",
    "inserted_state_ids",
    "native_supervision_mask",
    "rebuild_native_dataset",
    "rebuild_native_reaction",
    "sha256_file",
]
