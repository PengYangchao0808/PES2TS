"""R3 v3 fixtures: synthetic Reaction-Intent chemistry + six-table writer.

Deterministic fixtures for the W-FEAT E2 math-test battery and the G1
feature-gate runs (real data arrives from E1 in parallel).  The chemistry is
a tiny SN2-like ``CH3–Br + O → CH3–O + Br`` system (6 atoms) with:

* ``G_src``: C–Br + 3× C–H; ``G_tgt``: C–O + 3× C–H;
* K = {C, Br, O} (changed-bond atoms), CVs = (d(C–Br), d(C–O)) with directed
  target increments ``[+0.05, −0.05, 0, 0]`` Å for the forward intent;
* forward/reverse intents forming a mutual-reverse pair (label involution).

Every geometry is seeded; graph separation is a pure x-shift so no spatial
edge ever crosses graphs.  Shapes are chosen so the atom count never equals
the edge count, the key count or the graph count (permutation-gather safety).
"""

from __future__ import annotations

import csv
import hashlib
import json
import struct
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import torch

from pes2ts_core.pesn.contracts.key_atom_v3 import (
    CV_MAX,
    INTENT_OFF_ALL_V3,
    IntentRecordV3,
    KeyAtomSampleV3,
)
from pes2ts_core.pesn.features.key_atom_graph_v3 import (
    apply_intent_off,
    build_sample_features_v3,
    collate_v3,
    to_torch_batch,
)

ATOMIC_NUMBERS_SN2: tuple[int, ...] = (6, 35, 8, 1, 1, 1)  # C Br O H H H
BASE_XYZ_SN2: tuple[tuple[float, float, float], ...] = (
    (0.0, 0.0, 0.0),        # C
    (1.9, 0.05, 0.0),       # Br
    (-1.6, 0.3, 0.1),       # O
    (0.2, 1.0, 0.2),        # H
    (0.2, -0.9, 0.5),       # H
    (-0.3, -0.2, -1.0),     # H
)
G_SRC_BONDS: tuple[tuple[int, int, float], ...] = (
    (0, 1, 1.0), (0, 3, 1.0), (0, 4, 1.0), (0, 5, 1.0))
G_TGT_BONDS: tuple[tuple[int, int, float], ...] = (
    (0, 2, 1.0), (0, 3, 1.0), (0, 4, 1.0), (0, 5, 1.0))
#: directed Δξ = (+Δd(C–Br), −Δd(C–O), 0, 0) for the FORWARD intent
CV_DELTAS_FORWARD: tuple[float, float, float, float] = (0.05, -0.05, 0.0, 0.0)

#: forward teacher displacements (Å) for K = (C, Br, O)
Y_FORWARD: tuple[tuple[float, float, float], ...] = (
    (0.004, 0.000, 0.000),
    (0.028, 0.002, 0.000),
    (-0.022, 0.003, 0.001),
)


def make_sn2_intent(
    direction: str = "forward",
    *,
    curve_id: str = "curve_001",
    jitter_seed: int | None = None,
) -> IntentRecordV3:
    """The fixture intent pair (mutual reverses; ids stable per direction)."""
    fwd = IntentRecordV3(
        intent_id=f"{curve_id}::fwd",
        curve_id=curve_id,
        direction="forward",
        reverse_of_intent_id=f"{curve_id}::rev",
        intent_pair_id=f"{curve_id}::pair",
        g_src_bonds=G_SRC_BONDS,
        g_tgt_bonds=G_TGT_BONDS,
        cv_deltas=CV_DELTAS_FORWARD,
        intent_version="v3",
    )
    rev = fwd.reversed_intent()
    if direction == "forward":
        return fwd
    if direction == "reverse":
        return rev
    raise ValueError(f"direction must be forward|reverse, got {direction!r}")


def make_sn2_xyz(graph_index: int = 0, *, jitter_seed: int | None = None) -> tuple:
    """Seeded current-frame geometry; graphs separated by +12 Å in x."""
    coords = np.asarray(BASE_XYZ_SN2, dtype=np.float64)
    if jitter_seed is not None:
        rng = np.random.default_rng(jitter_seed)
        coords = coords + rng.normal(scale=0.03, size=coords.shape)
    coords = coords + np.array([12.0 * graph_index, 0.0, 0.0])
    return tuple(tuple(float(v) for v in row) for row in coords)


def make_sn2_teacher(direction: str = "forward", *, asymmetric: bool = False):
    """First-contact teacher targets for K = (C, Br, O).

    ``asymmetric=True`` builds the REVERSE target as an approximate (not
    exact) negative of the forward one — mirroring the contract's
    "small-step tangents approximately opposite, NOT exact negatives at
    finite step" (ReactionIntent Contract v1 §3.2).
    """
    base = np.asarray(Y_FORWARD, dtype=np.float64)
    if direction == "forward":
        return tuple(tuple(float(v) for v in row) for row in base)
    if not asymmetric:
        return tuple(tuple(float(v) for v in row) for row in -base)
    rng = np.random.default_rng(7)
    flipped = -base * 1.02 + rng.normal(scale=1e-4, size=base.shape)
    return tuple(tuple(float(v) for v in row) for row in flipped)


def make_sn2_sample(
    direction: str = "forward",
    *,
    node_index: int = 2,
    graph_index: int = 0,
    curve_id: str = "curve_001",
    d0: float = 0.08,
    frame_weight: float = 1.0,
    endpoint_names: tuple[str, str] = ("A", "B"),
    asymmetric_reverse: bool = False,
    jitter_seed: int | None = None,
) -> KeyAtomSampleV3:
    """One buildable fixture sample."""
    intent = make_sn2_intent(direction, curve_id=curve_id)
    return KeyAtomSampleV3(
        curve_id=curve_id,
        node_index=node_index,
        intent=intent,
        atomic_numbers=ATOMIC_NUMBERS_SN2,
        xyz=make_sn2_xyz(graph_index, jitter_seed=jitter_seed),
        d0=d0,
        endpoint_names=endpoint_names,
        y_teacher=make_sn2_teacher(
            direction, asymmetric=asymmetric_reverse and direction == "reverse"),
        frame_weight=frame_weight,
        sample_id=f"{curve_id}:{node_index}:{intent.intent_id}:{d0}",
    )


def build_fixture_features(
    direction: str = "forward",
    *,
    level: str = "F3",
    intent_off: bool | Sequence[str] = False,
    graph_index: int = 0,
    endpoint_names: tuple[str, str] = ("A", "B"),
    **kwargs,
) -> dict:
    """Built numpy features for one fixture sample."""
    return build_sample_features_v3(
        make_sn2_sample(direction, graph_index=graph_index,
                        endpoint_names=endpoint_names, **kwargs),
        level=level, intent_off=intent_off)


def make_v3_batch(
    *,
    n_graphs: int = 2,
    level: str = "F3",
    intent_off: bool | Sequence[str] = False,
    bidirectional: bool = True,
    with_teacher: bool = True,
    device: str = "cpu",
    jitter_seed: int | None = None,
) -> dict:
    """Collated torch fixture batch (one or more graphs, fp32/int64).

    With ``bidirectional=True`` the batch mixes forward and reverse intents
    on the same curve family (the F-bidir composition), otherwise every
    graph is forward.
    """
    samples = []
    for g in range(n_graphs):
        directions = ("forward", "reverse") if bidirectional else ("forward",)
        for d in directions:
            s = make_sn2_sample(
                d, node_index=g % 5, graph_index=g,
                jitter_seed=None if jitter_seed is None
                else jitter_seed + 100 * g + (0 if d == "forward" else 1))
            samples.append(s)
    built = [build_sample_features_v3(s, level=level, intent_off=intent_off)
             for s in samples]
    if not with_teacher:
        for b in built:
            b.pop("y_teacher", None)
    return to_torch_batch(collate_v3(built), device=device)


def make_v3_gate_batch(**kwargs) -> dict:
    """Forward-view-only fixture batch — what ``model.forward`` and the
    feature-gate harness consume (supervision stripped)."""
    from pes2ts_core.pesn.contracts.key_atom_v3 import forward_view_v3

    return forward_view_v3(make_v3_batch(**kwargs))


def make_intent_off_batch(
    batch_builder_args: Mapping[str, object] | None = None,
    *,
    mode: str = "full",
    unmask_classes: Iterable[str] = (),
) -> dict:
    """PD-6 intent-off variant of a fixture batch (harness ``intent_off=``).

    modes:

    * ``"full"``   — legal ablation arm: all five classes masked;
    * ``"partial"``— all classes EXCEPT ``unmask_classes`` masked (each
      unmasked class must be detected by ``check_intent_off_mask``);
    * ``"nuke"``   — five classes masked AND the legal undirected context
      destroyed (deliberate PD-6 violation).
    """
    args = dict(batch_builder_args or {})
    if mode == "full":
        return make_v3_batch(intent_off=True, **args)
    if mode == "partial":
        from pes2ts_core.pesn.contracts.key_atom_v3 import INTENT_OFF_CLASSES_V3

        keep = tuple(unmask_classes)
        known = {cls for cls, _ in INTENT_OFF_CLASSES_V3}
        unknown = set(keep) - known
        if unknown:
            raise ValueError(f"unknown PD-6 classes {sorted(unknown)}")
        masked = tuple(c for c in INTENT_OFF_ALL_V3 if c not in keep)
        return make_v3_batch(intent_off=masked, **args)
    if mode == "nuke":
        batch = make_v3_batch(intent_off=True, **args)
        for key in ("z", "edge_index", "is_key", "degK", "abs_delta",
                    "is_changed", "edge_attr"):
            if key in batch and isinstance(batch[key], torch.Tensor):
                batch[key] = torch.zeros_like(batch[key])
        return batch
    raise ValueError(f"unknown intent-off mode {mode!r}")


# ---------------------------------------------------------------------------
# Six-table fixture writer (PD-1 layout; E1 delivers the real tables later)
# ---------------------------------------------------------------------------
def _write_csv(path: Path, header: Sequence[str], rows: Iterable[Sequence[object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(list(header))
        for row in rows:
            writer.writerow(list(row))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_fixture_tables(
    tables_dir: str | Path,
    *,
    curve_id: str = "curve_001",
    n_frames: int = 5,
    n_curves: int = 1,
    fold: int = 0,
    label_status: str = "ok",
) -> dict:
    """Write the six-table PD-1 layout + manifests under ``tables_dir``.

    Conventions (see ``key_atom_dataset_v3`` interface note): per-frame full
    ``[N,3]`` coordinates live as a JSON payload in ``irc_node.x``; the
    sample weights satisfy Σ ``curve_weight`` = 1.0 per ``(curve, d0)``;
    ``target_action_u`` carries the teacher displacement for K.
    """
    out = Path(tables_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    atomic_json = json.dumps(list(ATOMIC_NUMBERS_SN2))
    fwd = make_sn2_intent("forward", curve_id=curve_id)
    rev = fwd.reversed_intent()
    intents = (fwd, rev)

    for c in range(n_curves):
        cid = curve_id if n_curves == 1 else f"{curve_id}_{c:03d}"
        fwd_c = make_sn2_intent("forward", curve_id=cid)
        rev_c = fwd_c.reversed_intent()
        frames = []
        for frame in range(n_frames):
            frames.append(np.asarray(
                make_sn2_xyz(0, jitter_seed=1000 * c + frame)))
        n_samples = n_frames * 2
        cw = 1.0 / n_samples

        # irc_curve
        p = out / "irc_curve.csv"
        _write_csv(p, (
            "curve_id", "canonical_irc_id", "n_nodes", "n_edges", "ts_index",
            "endpoint_a_index", "endpoint_b_index"),
            [(cid, f"canon::{cid}", n_frames, 2 * 4, n_frames // 2,
              0, n_frames - 1)])
        # irc_node (JSON [N,3] frame payload in x; y/z kept numeric-empty)
        p = out / "irc_node.csv"
        rows = []
        for frame, xyz in enumerate(frames):
            rows.append((cid, frame, json.dumps(xyz.tolist()), 0.0, 0.0,
                         -1.0 + 2.0 * frame / max(n_frames - 1, 1),
                         frame == n_frames // 2,
                         frame in (0, n_frames - 1), atomic_json))
        _write_csv(p, (
            "curve_id", "node_index", "x", "y", "z", "progress", "is_ts",
            "is_endpoint", "atomic_numbers"), rows)
        # reaction_intent (mutual reverses)
        p = out / "reaction_intent.csv"
        rows = []
        for it in (fwd_c, rev_c):
            rows.append((
                it.intent_id, cid, it.direction, it.reverse_of_intent_id,
                it.intent_pair_id, "false",
                json.dumps([list(b) for b in it.g_src_bonds]),
                json.dumps([list(b) for b in it.g_tgt_bonds]),
                "[]", json.dumps({"target_deltas": list(it.cv_deltas)}),
                "", "", "v3"))
        _write_csv(p, (
            "intent_id", "curve_id", "direction", "reverse_of_intent_id",
            "intent_pair_id", "intent_underdetermined", "g_src", "g_tgt",
            "delta_b_intent", "cv_spec", "intent_role", "rules",
            "intent_version"), rows)
        # irc_edge (PD-9 flat encoding; one row per intent edge)
        p = out / "irc_edge.csv"
        rows = []
        for edge_index, (src, dst) in enumerate(((0, 1), (1, 2), (2, 3), (3, 4))):
            for it in (fwd_c, rev_c):
                flat = list(np.asarray(
                    make_sn2_teacher(it.direction), dtype=np.float64).ravel())
                packed = struct.pack(f"<{len(flat)}d", *flat)
                rows.append((
                    cid, it.intent_id, edge_index, src, dst,
                    json.dumps(flat), "[6,3]", f"[0,{len(flat)})",
                    hashlib.sha256(b"fixture-atom-map").hexdigest(),
                    hashlib.sha256(packed).hexdigest(), "angstrom", "true"))
        _write_csv(p, (
            "curve_id", "intent_id", "edge_index", "src_node_index",
            "dst_node_index", "delta_xyz_flat", "delta_xyz_shape",
            "delta_xyz_offsets", "atom_map_sha256", "delta_xyz_sha256",
            "units", "same_ref_frame"), rows)
        # curve_quality (provenance only)
        p = out / "curve_quality.csv"
        _write_csv(p, ("curve_id", "quality_status"),
                   [(cid, label_status)])
        # sample_view
        p = out / "sample_view.csv"
        rows = []
        for frame in range(n_frames):
            for it in (fwd_c, rev_c):
                target = json.dumps(
                    [list(r) for r in make_sn2_teacher(it.direction)])
                rows.append((
                    f"{cid}:{frame}:{it.intent_id}:0.08", cid, frame,
                    it.intent_id, 0.08, 0.0, 0.0, 0.0,
                    1.0, cw, 1.0, label_status,
                    hashlib.sha256(b"feature").hexdigest(),
                    hashlib.sha256(b"label").hexdigest(),
                    "false", "local_frame",
                    len(it.key_rows()) and it.key_rows()[-1], target, 0.0, 0,
                    "false", fold))
        _write_csv(p, (
            "sample_id", "curve_id", "node_index", "intent_id", "d0",
            "x", "y", "z", "label_weight", "curve_weight", "sample_weight",
            "label_status", "feature_sha", "label_sha", "sensitivity_only",
            "gauge_ref", "target_node_index", "target_action_u",
            "target_progress_q", "stop_label", "mask", "fold"), rows)

    # manifests
    table_names = ("irc_curve.csv", "irc_node.csv", "irc_edge.csv",
                   "reaction_intent.csv", "curve_quality.csv",
                   "sample_view.csv")
    manifest = {"files": [
        {"path": name, "sha256": _sha256(out / name)} for name in table_names]}
    (out / "data_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out / "teacher_manifest.json").write_text(
        json.dumps({"files": []}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    curve_ids = [curve_id if n_curves == 1 else f"{curve_id}_{c:03d}"
                 for c in range(n_curves)]
    (out / "split_manifest_r3.json").write_text(
        json.dumps({"curve_folds": {cid: fold for cid in curve_ids}},
                   indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"tables_dir": out, "curve_ids": tuple(curve_ids),
            "n_frames": n_frames, "fold": fold}


__all__ = [
    "ATOMIC_NUMBERS_SN2",
    "BASE_XYZ_SN2",
    "CV_DELTAS_FORWARD",
    "G_SRC_BONDS",
    "G_TGT_BONDS",
    "Y_FORWARD",
    "build_fixture_features",
    "make_intent_off_batch",
    "make_sn2_intent",
    "make_sn2_sample",
    "make_sn2_teacher",
    "make_sn2_xyz",
    "make_v3_batch",
    "make_v3_gate_batch",
    "write_fixture_tables",
]
