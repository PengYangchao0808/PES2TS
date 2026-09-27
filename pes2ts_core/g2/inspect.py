"""Per-frame path metrics and the validity verdict (plan task 7).

Frame-order contract: :func:`frame_metrics` and :func:`evaluate_validity`
accept ONLY R→P-ordered frames — frame 0 is the assembled reactant-side start
and the last frame is the product-side end.  The pipeline must normalize the
direction (reverse the frame order and recalibrate ``energy_rel_kcal``) BEFORE
calling; reversed input is judged against the wrong endpoints, so it must
never reach this module silently.

Atom-order contract: path frames carry atoms in ascending reaction-map order
(the endpoint writer emits map-sorted XYZ and xTB preserves atom order), so
frame atom ``i`` corresponds to the ``i``-th smallest map of
``reactant_coords``; the frame element sequence is verified against
``elements``.

No truth references, no xTB invocation, no raw-frame mutation; stdlib + numpy
+ repo imports only.
"""

# allow: SIZE_OK -- the plan contract fixes the frame-metric columns, the
# event-distance key scheme, all six validity predicates with their exact
# thresholds, and the Verdict record in this single module; task 8 consumes
# the metric columns and task 9 the verdict directly.

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import combinations
from collections.abc import Collection
from typing import Any, Final, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from pes2ts_core.g0.rejections import RejectionCode
from pes2ts_core.g0.rp_checks import ELEMENT_SYMBOLS
from pes2ts_core.g1.index_map import COVALENT_RADII
from pes2ts_core.g2.status import FAILURE_PRECEDENCE, STATUS_FAILED, STATUS_VALID
from pes2ts_core.g2.xtb_output import Frame
from pes2ts_core.utils.hashing import JSONValue

#: Plan ``g2.validity.endpoint_rmsd_max`` fallback.
DEFAULT_ENDPOINT_RMSD_MAX: Final[float] = 0.5
#: Plan ``g2.validity.max_frame_step`` fallback.
DEFAULT_MAX_FRAME_STEP: Final[float] = 4.0
#: Plan ``g2.validity.min_frames`` fallback.
DEFAULT_MIN_FRAMES: Final[int] = 8
#: Plan ``g2.validity.collision_min_distance`` fallback.
DEFAULT_COLLISION_MIN_DISTANCE: Final[float] = 0.8
#: Plan ``g2.assembly.bond_tolerance`` fallback (element-aware drift threshold).
DEFAULT_BOND_TOLERANCE: Final[float] = 0.45

MapPair = tuple[int, int]
ArrayF64 = NDArray[np.float64]
Metric = dict[str, JSONValue]


@dataclass(frozen=True, slots=True)
class Verdict:
    """Terminal validity result of one R→P-ordered path."""

    status: str
    failure_code: RejectionCode | None
    detail: str | None
    summary: dict[str, JSONValue]


def _radius(element: str) -> float:
    """Cordero radius via the element→Z table (same source as endpoints)."""
    try:
        return COVALENT_RADII[ELEMENT_SYMBOLS.index(element) + 1]
    except ValueError:
        raise ValueError(f"no covalent radius for element {element!r}") from None


def _config_section(config: Mapping[str, Any] | None, name: str) -> Mapping[str, Any]:
    g2 = (config or {}).get("g2", {})
    block = g2.get(name, {}) if isinstance(g2, Mapping) else {}
    return block if isinstance(block, Mapping) else {}


def _thresholds(config: Mapping[str, Any] | None) -> tuple[float, float, int, float, float]:
    validity = _config_section(config, "validity")
    assembly = _config_section(config, "assembly")
    return (
        float(validity.get("endpoint_rmsd_max", DEFAULT_ENDPOINT_RMSD_MAX)),
        float(validity.get("max_frame_step", DEFAULT_MAX_FRAME_STEP)),
        int(validity.get("min_frames", DEFAULT_MIN_FRAMES)),
        float(validity.get("collision_min_distance", DEFAULT_COLLISION_MIN_DISTANCE)),
        float(assembly.get("bond_tolerance", DEFAULT_BOND_TOLERANCE)),
    )


def _event_entries(events: Mapping[str, Sequence[Mapping[str, Any]]]) -> list[tuple[str, MapPair]]:
    """Deterministic (key, map-pair) list for every evaluated event half.

    formed/broken/order_changed carry one pair each; hydrogen_migration
    contributes one entry per non-null partner (``from``/``to`` halves).
    """
    entries: list[tuple[str, MapPair]] = []
    for kind in ("formed", "broken", "order_changed"):
        for record in events.get(kind, ()):
            first, second = sorted(int(value) for value in record["atoms"])
            entries.append((f"{kind}:{first}-{second}", (first, second)))
    for record in events.get("hydrogen_migration", ()):
        hydrogen = int(record["h"])
        origin = record.get("from")
        target = record.get("to")
        if origin is not None:
            entries.append((f"h_migration:{hydrogen}:from={int(origin)}", (hydrogen, int(origin))))
        if target is not None:
            entries.append((f"h_migration:{hydrogen}:to={int(target)}", (hydrogen, int(target))))
    return entries


def _canonical_pairs(pairs: Collection[Sequence[int]]) -> frozenset[MapPair]:
    return frozenset((min(pair), max(pair)) for pair in pairs)


def _check_frames(
    frames: Sequence[Frame], maps: tuple[int, ...], elements: Mapping[int, str]
) -> None:
    if not frames:
        raise ValueError("no frames to inspect")
    expected = tuple(elements[map_] for map_ in maps)
    for index, frame in enumerate(frames):
        if len(frame.coordinates) != len(maps):
            raise ValueError(f"frame {index}: {len(frame.coordinates)} atoms, expected {len(maps)}")
        if frame.elements != expected:
            raise ValueError(
                f"frame {index}: element sequence {frame.elements} does not match the "
                f"ascending map order {expected}"
            )


def _rmsd(delta: ArrayF64) -> float:
    return float(np.sqrt(np.mean(np.sum(delta * delta, axis=1))))


def frame_metrics(
    frames: Sequence[Frame],
    *,
    reaction_id: str,
    reactant_coords: Mapping[int, Sequence[float]],
    product_coords: Mapping[int, Sequence[float]],
    elements: Mapping[int, str],
    r_pairs: Collection[Sequence[int]],
    p_pairs: Collection[Sequence[int]],
    events: Mapping[str, Sequence[Mapping[str, Any]]],
    config: Mapping[str, Any] | None = None,
) -> list[Metric]:
    """One metric record per R→P frame (columns consumed by task 8's parquet).

    ``rmsd_to_start``/``rmsd_to_end`` are in-place RMSDs against the map-keyed
    endpoint coordinates (no superposition; frame 0 is the assembly start).
    ``step_max``/``step_rmsd`` compare each frame with its predecessor
    (frame 0 → 0.0).  ``min_nonbonded_distance`` spans pairs bonded in neither
    R nor P (``None`` when no such pair exists).  ``event_distances`` holds the
    deterministic per-frame distance of every evaluated event half.
    """
    maps = tuple(sorted(reactant_coords))
    if set(product_coords) != set(maps):
        raise ValueError("product_coords map set differs from reactant_coords map set")
    _check_frames(frames, maps, elements)
    index_of_map = {map_: index for index, map_ in enumerate(maps)}
    reactant_array = np.array([reactant_coords[map_] for map_ in maps], dtype=np.float64)
    product_array = np.array([product_coords[map_] for map_ in maps], dtype=np.float64)
    event_entries = _event_entries(events)
    bonded = _canonical_pairs(r_pairs) | _canonical_pairs(p_pairs)
    collision_pairs = [pair for pair in combinations(maps, 2) if pair not in bonded]
    metrics: list[Metric] = []
    previous: ArrayF64 | None = None
    for index, frame in enumerate(frames):
        array = np.array([frame.coordinates[index_of_map[map_]] for map_ in maps], dtype=np.float64)
        if previous is None:
            step_max, step_rmsd = 0.0, 0.0
        else:
            norms = np.sqrt(np.sum((array - previous) ** 2, axis=1))
            step_max = float(np.max(norms))
            step_rmsd = float(np.sqrt(np.mean(norms * norms)))
        event_distances = {
            key: float(np.linalg.norm(array[index_of_map[first]] - array[index_of_map[second]]))
            for key, (first, second) in event_entries
        }
        nonbonded = [
            float(np.linalg.norm(array[index_of_map[first]] - array[index_of_map[second]]))
            for first, second in collision_pairs
        ]
        metrics.append(
            {
                "reaction_id": str(reaction_id),
                "frame_index": index,
                "energy_rel_kcal": None if frame.energy is None else float(frame.energy),
                "rmsd_to_start": _rmsd(array - reactant_array),
                "rmsd_to_end": _rmsd(array - product_array),
                "step_max": step_max,
                "step_rmsd": step_rmsd,
                "min_nonbonded_distance": min(nonbonded) if nonbonded else None,
                "event_distances": event_distances,
            }
        )
        previous = array
    return metrics


def _expected_for(kind: str, threshold: float) -> str:
    if kind == "formed":
        return f"first>={threshold:.6f} and last<{threshold:.6f}"
    if kind == "broken":
        return f"first<{threshold:.6f} and last>={threshold:.6f}"
    return f"first<{threshold:.6f} and last<{threshold:.6f}"


def _topology_violations(
    first_distances: Mapping[str, float],
    last_distances: Mapping[str, float],
    events: Mapping[str, Sequence[Mapping[str, Any]]],
    elements: Mapping[int, str],
    bond_tolerance: float,
) -> list[str]:
    """PASS-predicate violations per event; empty list = no drift.

    G1 semantics list a bond-order change as ``formed`` + ``broken`` +
    ``order_changed`` (the pair is absent from one side's bond list and carries
    two different orders).  The formed/broken predicates are mutually
    contradictory for such a pair (one demands ``first >= threshold``, the
    other ``first < threshold``), so a pair present in ``order_changed`` is
    judged only by the dedicated order-changed predicate (bonded throughout).

    Element-aware threshold = Cordero radius sum + ``bond_tolerance``.  A null
    h_migration partner makes that half vacuous (documented interpretation).
    """
    violations: list[str] = []
    order_changed_pairs = {
        tuple(sorted(int(value) for value in record["atoms"]))
        for record in events.get("order_changed", ())
    }
    for kind in ("formed", "broken", "order_changed"):
        for record in events.get(kind, ()):
            first, second = sorted(int(value) for value in record["atoms"])
            if kind in ("formed", "broken") and (first, second) in order_changed_pairs:
                continue  # an order change is judged by its own predicate only
            threshold = _radius(elements[first]) + _radius(elements[second]) + bond_tolerance
            key = f"{kind}:{first}-{second}"
            d_first, d_last = first_distances[key], last_distances[key]
            passed = (
                d_first >= threshold and d_last < threshold
                if kind == "formed"
                else d_first < threshold and d_last >= threshold
                if kind == "broken"
                else d_first < threshold and d_last < threshold
            )
            if not passed:
                violations.append(
                    f"{key}: first={d_first:.6f} last={d_last:.6f} "
                    f"({_expected_for(kind, threshold)})"
                )
    for record in events.get("hydrogen_migration", ()):
        hydrogen = int(record["h"])
        halves: list[str] = []
        passed = True
        origin = record.get("from")
        target = record.get("to")
        if origin is not None:
            threshold = _radius(elements[hydrogen]) + _radius(elements[int(origin)]) + bond_tolerance
            distance = first_distances[f"h_migration:{hydrogen}:from={int(origin)}"]
            passed = passed and distance < threshold
            halves.append(f"from d_first={distance:.6f} th={threshold:.6f}")
        if target is not None:
            threshold = _radius(elements[hydrogen]) + _radius(elements[int(target)]) + bond_tolerance
            distance = last_distances[f"h_migration:{hydrogen}:to={int(target)}"]
            passed = passed and distance < threshold
            halves.append(f"to d_last={distance:.6f} th={threshold:.6f}")
        if not passed:
            violations.append(f"h_migration:{hydrogen}: " + ", ".join(halves))
    return violations


def evaluate_validity(
    frames: Sequence[Frame],
    *,
    reaction_id: str,
    reactant_coords: Mapping[int, Sequence[float]],
    product_coords: Mapping[int, Sequence[float]],
    elements: Mapping[int, str],
    r_pairs: Collection[Sequence[int]],
    p_pairs: Collection[Sequence[int]],
    events: Mapping[str, Sequence[Mapping[str, Any]]],
    xtb_failure: str | None = None,
    config: Mapping[str, Any] | None = None,
) -> Verdict:
    """Judge an R→P-ordered path; first violated predicate in precedence wins.

    The pipeline must normalize direction (reverse + recalibrate) BEFORE
    calling — this function accepts no reversed frames.
    An xTB failure short-circuits everything: a crashed run (non-zero exit /
    timeout / missing key artifacts) may have produced no parseable frames at
    all, so the verdict is returned before any frame validation.
    """
    if xtb_failure is not None:
        return Verdict(
            STATUS_FAILED,
            RejectionCode.G2_XTB_FAILED,
            str(xtb_failure),
            {
                "reaction_id": str(reaction_id),
                "n_frames": len(frames),
                "xtb_failure": str(xtb_failure),
            },
        )
    endpoint_max, max_step, min_frames, collision_min, bond_tolerance = _thresholds(config)
    metrics = frame_metrics(
        frames,
        reaction_id=reaction_id,
        reactant_coords=reactant_coords,
        product_coords=product_coords,
        elements=elements,
        r_pairs=r_pairs,
        p_pairs=p_pairs,
        events=events,
        config=config,
    )
    first, last = metrics[0], metrics[-1]
    first_vs_r = float(first["rmsd_to_start"])
    last_vs_p = float(last["rmsd_to_end"])
    violations: dict[RejectionCode, str] = {}
    if first_vs_r > endpoint_max or last_vs_p > endpoint_max:
        violations[RejectionCode.G2_ENDPOINT_NOT_REACHED] = (
            f"first_vs_R={first_vs_r:.6f} last_vs_P={last_vs_p:.6f} "
            f"exceed endpoint_rmsd_max={endpoint_max:.6f}"
        )
    drift = _topology_violations(
        first["event_distances"], last["event_distances"], events, elements, bond_tolerance
    )
    if drift:
        violations[RejectionCode.G2_TOPOLOGY_DRIFT] = "; ".join(drift)
    bad_energy = [
        record["frame_index"]
        for record in metrics
        if record["energy_rel_kcal"] is None or not math.isfinite(float(record["energy_rel_kcal"]))
    ]
    if bad_energy:
        violations[RejectionCode.G2_ENERGY_INCOMPLETE] = (
            f"frames with missing/non-finite energy: {bad_energy}"
        )
    stepped = [record["frame_index"] for record in metrics if float(record["step_max"]) > max_step]
    if stepped or len(frames) < min_frames:
        parts: list[str] = []
        if stepped:
            parts.append(f"single-atom steps exceed max_frame_step={max_step} at frames {stepped}")
        if len(frames) < min_frames:
            parts.append(f"n_frames={len(frames)} < min_frames={min_frames}")
        violations[RejectionCode.G2_PATH_DISCONTINUOUS] = "; ".join(parts)
    collisions = [
        (record["frame_index"], float(record["min_nonbonded_distance"]))
        for record in metrics
        if record["min_nonbonded_distance"] is not None
        and float(record["min_nonbonded_distance"]) < collision_min
    ]
    if collisions:
        worst_frame, worst_distance = min(collisions, key=lambda item: item[1])
        violations[RejectionCode.G2_COLLISION] = (
            f"non-bonded pair below collision_min_distance={collision_min}: "
            f"worst frame {worst_frame} at {worst_distance:.6f} A"
        )
    first_distances: Mapping[str, float] = first["event_distances"]
    last_distances: Mapping[str, float] = last["event_distances"]
    nonbonded_values = [
        float(record["min_nonbonded_distance"])
        for record in metrics
        if record["min_nonbonded_distance"] is not None
    ]
    summary: dict[str, JSONValue] = {
        "reaction_id": str(reaction_id),
        "n_frames": len(metrics),
        "first_vs_r_rmsd": first_vs_r,
        "last_vs_p_rmsd": last_vs_p,
        "worst_step_max": max(float(record["step_max"]) for record in metrics),
        "min_nonbonded_distance": min(nonbonded_values) if nonbonded_values else None,
        "event_distances": {
            key: {"first": first_distances[key], "last": last_distances[key]}
            for key in sorted(first_distances)
        },
    }
    code = next((candidate for candidate in FAILURE_PRECEDENCE if candidate in violations), None)
    if code is None:
        return Verdict(STATUS_VALID, None, None, summary)
    return Verdict(STATUS_FAILED, code, violations[code], summary)


__all__ = ["Verdict", "evaluate_validity", "frame_metrics"]
