"""Candidate generation, legality, and behavior policies (ADR-0017 §4).

The candidate list at one decision is finite, pre-registered, and frozen
within a round: the zero residual (the Classic baseline), the model residual
with its 0.5/1.0 scalings, and a small seeded set of bounded single-coordinate
+/- perturbations.  Candidates are deduplicated, capped, and judged against
the frozen :class:`ResidualControlSpec` (the A(H) side of legality); illegal
candidates are dropped with a typed reason, never clamped silently.

Selection policies:

* :class:`RuleZeroPolicy` — always executes the zero residual (Classic
  baseline behaviour policy for collection and comparison);
* :class:`EpsilonExplorePolicy` — epsilon-greedy over the legal candidates
  with a frozen seed (the bounded-exploration collector of the first round);
* :class:`QGreedyPolicy` — argmax Q_proxy among candidates with training
  support; falls back to the frozen baseline when support is missing
  (an untrained Q never claims decision capability).

The module also adapts a G2 case into PES-N inputs (task graph from the
frozen plan edits + endpoint covalent graphs, coordinate spec from the
compiled drivers) so a predictor-backed policy can score states without the
generation side importing any model code.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from pes2ts_core.generation.planning.chemical_control import ChemicalControl
from pes2ts_core.generation.planning.control_step import (
    ControlDecision,
    ControlScanState,
)
from pes2ts_core.pesn.contracts.control import (
    ACTION_SOURCE_EXPLORE,
    ACTION_SOURCE_MODEL_RESIDUAL,
    ACTION_SOURCE_PERTURB,
    ACTION_SOURCE_RULE_ZERO,
    ACTION_SOURCE_SCALED,
    CONTROL_CONTEXT_DIM,
    CONTROL_INPUT_SCHEMA_VERSION,
    ControlContext,
    ControlInput,
    ResidualControlSpec,
)

_COVALENT_RADII: dict[str, float] = {
    "H": 0.31, "B": 0.84, "C": 0.76, "N": 0.71, "O": 0.66, "F": 0.57,
    "Si": 1.11, "P": 1.07, "S": 1.05, "Cl": 1.02, "Br": 1.20, "I": 1.39,
}
_BOND_TOLERANCE = 0.45


class PolicyAdapterError(ValueError):
    """Typed candidate/policy contract violation."""


# ---------------------------------------------------------------------------
# Frozen candidate configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CandidatePolicyConfig:
    """Pre-registered candidate generation + exploration parameters."""

    model_scales: tuple[float, ...] = (0.5, 1.0)
    n_single_coord_perturbs: int = 2
    max_candidates: int = 8
    exploration_probability: float = 0.25
    exploration_seed: int = 42
    dedup_tolerance: float = 1e-6

    def to_payload(self) -> dict[str, Any]:
        return {
            "model_scales": list(self.model_scales),
            "n_single_coord_perturbs": int(self.n_single_coord_perturbs),
            "max_candidates": int(self.max_candidates),
            "exploration_probability": float(self.exploration_probability),
            "exploration_seed": int(self.exploration_seed),
            "dedup_tolerance": float(self.dedup_tolerance),
        }

    def content_hash(self) -> str:
        return hashlib.sha256(
            json.dumps(self.to_payload(), sort_keys=True, separators=(",", ":")).encode(
                "utf-8"
            )
        ).hexdigest()


def build_residual_spec(
    control: ChemicalControl,
    *,
    epsilon_angstrom: float,
    allow_sign: int = 0,
    range_slack_angstrom: float = 0.0,
) -> ResidualControlSpec:
    """Derive the frozen per-driver residual spec from a compiled control.

    The legal command range per driver is its frozen endpoint schedule widened
    by ``epsilon + slack`` (distance drivers only in v1, mirroring the feature
    layer's distance-only support).
    """
    if epsilon_angstrom <= 0.0:
        raise PolicyAdapterError("epsilon must be positive")
    if allow_sign not in (-1, 0, 1):
        raise PolicyAdapterError("allow_sign must be -1/0/+1")
    eps: list[float] = []
    mins: list[float] = []
    maxs: list[float] = []
    signs: list[int] = []
    for driver in control.drivers:
        if driver.kind != "distance":
            raise PolicyAdapterError(
                f"driver {driver.driver_id} is {driver.kind!r}; v1 residuals "
                "support distance drivers only"
            )
        slack = epsilon_angstrom + range_slack_angstrom
        low = max(0.05, min(driver.values) - slack)
        high = max(driver.values) + slack
        eps.append(epsilon_angstrom)
        mins.append(low)
        maxs.append(high)
        signs.append(allow_sign)
    spec = ResidualControlSpec(
        schema_version="pesn_residual_control_spec_v1",
        driver_ids=tuple(driver.driver_id for driver in control.drivers),
        epsilon=tuple(eps),
        allowed_sign=tuple(signs),
        min_target=tuple(mins),
        max_target=tuple(maxs),
    )
    spec.validate()
    return spec


# ---------------------------------------------------------------------------
# Candidate generation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CandidateList:
    """Legal, deduplicated, capped candidate set for one decision."""

    candidates: tuple[tuple[float, ...], ...]
    sources: tuple[str, ...]
    dropped: tuple[str, ...]


def _round_for_dedup(u: Sequence[float], tolerance: float) -> tuple[float, ...]:
    return tuple(round(float(v) / tolerance) * tolerance for v in u)


def generate_candidates(
    spec: ResidualControlSpec,
    base_targets: Sequence[float],
    *,
    config: CandidatePolicyConfig,
    model_u: Sequence[float] | None = None,
    rng: np.random.Generator | None = None,
) -> CandidateList:
    """Zero residual + model scalings + seeded bounded perturbations.

    The zero residual is always first (the frozen baseline).  Illegal
    candidates (out of range or wrong sign) are dropped with typed reasons.
    """
    n = len(spec.driver_ids)
    zero = tuple(0.0 for _ in range(n))
    pool: list[tuple[tuple[float, ...], str]] = [(zero, ACTION_SOURCE_RULE_ZERO)]
    dropped: list[str] = []

    if model_u is not None and len(model_u) == n and any(abs(v) > 1e-9 for v in model_u):
        for scale in config.model_scales:
            scaled = tuple(float(v) * scale for v in model_u)
            legal, reason = spec.residual_is_legal(scaled, base_targets)
            if legal:
                source = (
                    ACTION_SOURCE_MODEL_RESIDUAL
                    if abs(scale - 1.0) < 1e-12
                    else ACTION_SOURCE_SCALED
                )
                pool.append((scaled, source))
            else:
                dropped.append(f"scaled:{reason}")

    generator = rng if rng is not None else np.random.default_rng(config.exploration_seed)
    n_perturbs = int(config.n_single_coord_perturbs)
    if n_perturbs > 0 and n > 0:
        # Deterministic per-call exploration of the coordinate/direction grid,
        # seeded by the frozen exploration seed (pre-registered in train).
        total_directions = 2 * n
        start = int(generator.integers(0, total_directions))
        emitted = 0
        cursor = 0
        while emitted < n_perturbs and cursor < 4 * total_directions:
            slot = (start + cursor) % total_directions
            cursor += 1
            coordinate = slot // 2
            sign = 1.0 if slot % 2 == 0 else -1.0
            u = [0.0] * n
            u[coordinate] = sign
            candidate = tuple(float(v) for v in u)
            if any(
                _round_for_dedup(existing, config.dedup_tolerance)
                == _round_for_dedup(candidate, config.dedup_tolerance)
                for existing, _ in pool
            ):
                continue
            legal, reason = spec.residual_is_legal(candidate, base_targets)
            if not legal:
                dropped.append(f"perturb:{reason}")
                continue
            pool.append((candidate, ACTION_SOURCE_PERTURB))
            emitted += 1

    deduped: list[tuple[tuple[float, ...], str]] = []
    seen: set[tuple[float, ...]] = set()
    for candidate, source in pool:
        key = _round_for_dedup(candidate, config.dedup_tolerance)
        if key in seen:
            continue
        seen.add(key)
        deduped.append((candidate, source))
        if len(deduped) >= config.max_candidates:
            break

    return CandidateList(
        candidates=tuple(candidate for candidate, _ in deduped),
        sources=tuple(source for _, source in deduped),
        dropped=tuple(dropped),
    )


# ---------------------------------------------------------------------------
# Behavior policies
# ---------------------------------------------------------------------------


class RuleZeroPolicy:
    """Frozen baseline behaviour: always execute the zero residual."""

    policy_id = "rule_zero_v1"

    def __init__(self, config: CandidatePolicyConfig) -> None:
        self.config = config
        self.policy_hash = self._content_hash()

    def _content_hash(self) -> str:
        payload = {"policy_id": self.policy_id, "config": self.config.to_payload()}
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

    def decide(self, state: ControlScanState) -> ControlDecision:
        listing = generate_candidates(
            state.spec,
            state.control.targets_at(state.target_lambda),
            config=self.config,
        )
        return ControlDecision(
            candidates=listing.candidates,
            candidate_sources=listing.sources,
            executed_candidate_index=0,
            policy_id=self.policy_id,
            policy_hash=self.policy_hash,
        )


class EpsilonExplorePolicy:
    """Rule-zero baseline with epsilon-greedy bounded exploration.

    The exploration draw uses a frozen-seed RNG owned by the policy, so the
    collection behaviour is reproducible from the round configuration.  This
    is the first-round data-collection behaviour policy; it never claims
    learned decision capability.
    """

    policy_id = "epsilon_explore_v1"

    def __init__(
        self,
        config: CandidatePolicyConfig,
        *,
        model_residual_fn: Callable[[ControlScanState], Sequence[float] | None] | None = None,
    ) -> None:
        self.config = config
        self.model_residual_fn = model_residual_fn
        self.rng = np.random.default_rng(config.exploration_seed)
        self.policy_hash = self._content_hash()

    def _content_hash(self) -> str:
        payload = {
            "policy_id": self.policy_id,
            "config": self.config.to_payload(),
            "uses_model_residual": self.model_residual_fn is not None,
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

    def decide(self, state: ControlScanState) -> ControlDecision:
        base_targets = state.control.targets_at(state.target_lambda)
        model_u = self.model_residual_fn(state) if self.model_residual_fn else None
        listing = generate_candidates(
            state.spec, base_targets, config=self.config, model_u=model_u,
            rng=self.rng,
        )
        if float(self.rng.random()) < self.config.exploration_probability:
            index = int(self.rng.integers(0, len(listing.candidates)))
            source = listing.sources[index]
            if source == ACTION_SOURCE_RULE_ZERO:
                # exploration already covers the baseline as candidate 0;
                # a drawn zero stays the baseline
                index = 0
            elif source in (ACTION_SOURCE_MODEL_RESIDUAL, ACTION_SOURCE_SCALED,
                            ACTION_SOURCE_PERTURB):
                # exploration executes the drawn candidate as an exploration action
                listing = CandidateList(
                    candidates=listing.candidates,
                    sources=tuple(
                        ACTION_SOURCE_EXPLORE if position == index else src
                        for position, src in enumerate(listing.sources)
                    ),
                    dropped=listing.dropped,
                )
        else:
            index = 0
        return ControlDecision(
            candidates=listing.candidates,
            candidate_sources=listing.sources,
            executed_candidate_index=index,
            policy_id=self.policy_id,
            policy_hash=self.policy_hash,
        )


class ModelResidualPolicy:
    """Frozen BC-residual control group (Demo24 plan §7).

    Executes the frozen residual head's own proposal (the scale-1.0 model
    residual candidate) after the legality/support gate; with no available
    proposal, an illegal proposal, or a disabled head, the policy falls back
    to the zero residual.  No Q participates in this control group.
    """

    policy_id = "bc_residual_v1"

    def __init__(
        self,
        config: CandidatePolicyConfig,
        model_residual_fn: Callable[[ControlScanState], Sequence[float] | None],
    ) -> None:
        self.config = config
        self.model_residual_fn = model_residual_fn
        self.policy_hash = self._content_hash()

    def _content_hash(self) -> str:
        payload = {
            "policy_id": self.policy_id,
            "config": self.config.to_payload(),
            "uses_model_residual": True,
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

    def decide(self, state: ControlScanState) -> ControlDecision:
        base_targets = state.control.targets_at(state.target_lambda)
        model_u = self.model_residual_fn(state)
        listing = generate_candidates(
            state.spec, base_targets, config=self.config, model_u=model_u
        )
        index = 0
        if model_u is not None:
            for position, source in enumerate(listing.sources):
                if source == ACTION_SOURCE_MODEL_RESIDUAL:
                    index = position
                    break
        return ControlDecision(
            candidates=listing.candidates,
            candidate_sources=listing.sources,
            executed_candidate_index=index,
            policy_id=self.policy_id,
            policy_hash=self.policy_hash,
        )


class QGreedyPolicy:
    """Argmax Q_proxy among supported candidates; baseline fallback.

    ``score_fn(state, candidates) -> Sequence[float | None]`` returns one
    Q_proxy per candidate (None = no training support).  With no support at
    all the policy takes the frozen zero-residual baseline — an unsupervised
    Q never drives execution (plan §4).
    """

    policy_id = "q_greedy_v1"

    def __init__(
        self,
        config: CandidatePolicyConfig,
        score_fn: Callable[[ControlScanState, Sequence[Sequence[float]]], Sequence[float | None]],
        *,
        model_residual_fn: Callable[[ControlScanState], Sequence[float] | None] | None = None,
        support_fn: Callable[
            [ControlScanState, Sequence[Sequence[float]], Sequence[str]],
            Sequence[bool],
        ]
        | None = None,
    ) -> None:
        self.config = config
        self.score_fn = score_fn
        self.model_residual_fn = model_residual_fn
        self.support_fn = support_fn
        self.policy_hash = self._content_hash()

    def _content_hash(self) -> str:
        payload = {
            "policy_id": self.policy_id,
            "config": self.config.to_payload(),
            "uses_model_residual": self.model_residual_fn is not None,
            "uses_support_gate": self.support_fn is not None,
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

    def decide(self, state: ControlScanState) -> ControlDecision:
        base_targets = state.control.targets_at(state.target_lambda)
        model_u = self.model_residual_fn(state) if self.model_residual_fn else None
        listing = generate_candidates(
            state.spec, base_targets, config=self.config, model_u=model_u
        )
        scores = list(self.score_fn(state, listing.candidates))
        if len(scores) != len(listing.candidates):
            raise PolicyAdapterError("score_fn output not parallel to candidates")
        if self.support_fn is not None:
            mask = list(self.support_fn(state, listing.candidates, listing.sources))
            if len(mask) != len(listing.candidates):
                raise PolicyAdapterError("support_fn output not parallel to candidates")
            scores = [
                None if not supported else value
                for supported, value in zip(mask, scores)
            ]
        best_index = 0
        best_value: float | None = None
        for index, value in enumerate(scores):
            if value is None or not np.isfinite(value):
                continue
            if best_value is None or value > best_value:
                best_value = float(value)
                best_index = index
        return ControlDecision(
            candidates=listing.candidates,
            candidate_sources=listing.sources,
            executed_candidate_index=best_index,
            policy_id=self.policy_id,
            policy_hash=self.policy_hash,
        )


# ---------------------------------------------------------------------------
# Case adapter: G2 control state -> PES-N control input
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ControlCaseAdapter:
    """Builds :class:`ControlInput` observations for one compiled case.

    The task graph is derived ONCE per case from the frozen plan: R/P bond
    lists come from the endpoint covalent graphs (order 1.0, plan-frozen atom
    order), task edits from the plan's active edits.  The coordinate spec is
    the compiled driver list (distance drivers, scale 1 A, ref 0).  No truth
    source is consulted anywhere in this adapter.
    """

    reaction_id: str
    atomic_numbers: tuple[int, ...]
    task_graph_payload: dict[str, Any]
    coordinate_spec_payload: dict[str, Any]
    charge: int | None
    multiplicity: int | None

    @classmethod
    def from_case(
        cls,
        *,
        control: ChemicalControl,
        plan: dict[str, Any],
        endpoint_geometry_r: Sequence[Sequence[float]],
        endpoint_geometry_p: Sequence[Sequence[float]],
    ) -> "ControlCaseAdapter":
        from pes2ts_core.pesn.contracts.inputs import (
            CoordinateDef,
            CoordinateSpec,
            TaskEdit,
            TaskGraph,
        )

        n_atoms = len(control.elements)
        atomic_numbers = _element_to_z(control.elements)
        r_bonds = _covalent_terms(
            control.elements, np.asarray(endpoint_geometry_r, dtype=float)
        )
        p_bonds = _covalent_terms(
            control.elements, np.asarray(endpoint_geometry_p, dtype=float)
        )
        edits: list[TaskEdit] = []
        position_by_map = {int(m): i for i, m in enumerate(control.atom_map_order)}
        for edit in plan.get("active_edits", []):
            maps = edit.get("maps")
            if not isinstance(maps, list) or len(maps) != 2:
                continue
            i = position_by_map[int(maps[0])]
            j = position_by_map[int(maps[1])]
            kind = str(edit.get("edit_kind", "broken"))
            if kind not in ("formed", "broken", "order_changed"):
                kind = "broken"
            edits.append(TaskEdit(kind, i, j, 1.0, 0.0 if kind == "broken" else 1.0))
        task_graph = TaskGraph(
            schema_version="pesn_task_graph_v1",
            atomic_numbers=atomic_numbers,
            atom_map=tuple(int(m) for m in control.atom_map_order),
            reactant_bonds=tuple(r_bonds),
            product_bonds=tuple(p_bonds),
            edits=tuple(edits),
            source=f"control_case:{control.plan_sha256[:12]}",
            provenance={"plan_sha256": control.plan_sha256},
        )
        task_graph.validate()
        coords = tuple(
            CoordinateDef(
                kind="distance",
                atoms=tuple(driver.atoms),
                units="angstrom",
                role="driver",
                scale_a=1.0,
                ref_q=0.0,
                label=driver.driver_id,
            )
            for driver in control.drivers
            if driver.kind == "distance"
        )
        spec = CoordinateSpec(
            schema_version="pesn_coordinate_spec_v1",
            coords=coords,
            spec_version=f"control:{control.plan_sha256[:12]}",
        )
        spec.validate(n_atoms)
        return cls(
            reaction_id=control.reaction_id,
            atomic_numbers=atomic_numbers,
            task_graph_payload=task_graph_content(task_graph),
            coordinate_spec_payload=spec_content(spec),
            charge=control.charge,
            multiplicity=control.multiplicity,
        )

    def build_input(
        self,
        *,
        geometry: Sequence[Sequence[float]],
        context: ControlContext,
        spec: ResidualControlSpec,
        base_targets: Sequence[float],
    ) -> ControlInput:
        from pes2ts_core.pesn.contracts.inputs import (
            build_pesn_input,
        )

        task_graph = task_graph_from_payload(self.task_graph_payload)
        coordinate_spec = coordinate_spec_from_payload(self.coordinate_spec_payload)
        base = build_pesn_input(
            input_id=f"{self.reaction_id}#control",
            task_graph=task_graph,
            coordinate_spec=coordinate_spec,
            coordinates=geometry,
            charge=self.charge,
            multiplicity=self.multiplicity,
            method="gfn2_xtb_e_native",
        )
        if len(context.to_vector()) != CONTROL_CONTEXT_DIM:
            raise PolicyAdapterError("context vector dimension drifted")
        control_input = ControlInput(
            schema_version=CONTROL_INPUT_SCHEMA_VERSION,
            base_input=base,
            context=context,
            spec=spec,
            base_targets=tuple(float(v) for v in base_targets),
        )
        control_input.validate()
        return control_input


def task_graph_content(task_graph) -> dict[str, Any]:
    """Serializable TaskGraph payload (atoms, bonds, edits, provenance)."""
    return {
        "schema_version": task_graph.schema_version,
        "atomic_numbers": list(task_graph.atomic_numbers),
        "atom_map": list(task_graph.atom_map),
        "reactant_bonds": [[b.i, b.j, b.order] for b in task_graph.reactant_bonds],
        "product_bonds": [[b.i, b.j, b.order] for b in task_graph.product_bonds],
        "edits": [
            [e.kind, e.i, e.j, e.reactant_order, e.product_order]
            for e in task_graph.edits
        ],
        "source": task_graph.source,
        "provenance": dict(task_graph.provenance),
    }


def task_graph_from_payload(payload: Mapping[str, Any]):
    """Rebuild a typed TaskGraph from its serializable payload."""
    from pes2ts_core.pesn.contracts.inputs import BondTerm, TaskEdit, TaskGraph

    return TaskGraph(
        schema_version=payload["schema_version"],
        atomic_numbers=tuple(payload["atomic_numbers"]),
        atom_map=tuple(payload["atom_map"]),
        reactant_bonds=tuple(
            BondTerm(b[0], b[1], b[2]) for b in payload["reactant_bonds"]
        ),
        product_bonds=tuple(
            BondTerm(b[0], b[1], b[2]) for b in payload["product_bonds"]
        ),
        edits=tuple(
            TaskEdit(e[0], e[1], e[2], e[3], e[4]) for e in payload["edits"]
        ),
        source=payload["source"],
        provenance=dict(payload.get("provenance", {})),
    )


def coordinate_spec_from_payload(payload: Mapping[str, Any]):
    """Rebuild a typed CoordinateSpec from its serializable payload."""
    from pes2ts_core.pesn.contracts.inputs import CoordinateDef, CoordinateSpec

    return CoordinateSpec(
        schema_version=payload["schema_version"],
        coords=tuple(
            CoordinateDef(
                kind=c["kind"],
                atoms=tuple(c["atoms"]),
                units=c["units"],
                role=c["role"],
                scale_a=c["scale_a"],
                ref_q=c["ref_q"],
                label=c["label"],
            )
            for c in payload["coords"]
        ),
        spec_version=payload["spec_version"],
    )


def spec_content(spec) -> dict[str, Any]:
    return {
        "schema_version": spec.schema_version,
        "coords": [
            {
                "kind": c.kind,
                "atoms": list(c.atoms),
                "units": c.units,
                "role": c.role,
                "scale_a": c.scale_a,
                "ref_q": c.ref_q,
                "label": c.label,
            }
            for c in spec.coords
        ],
        "spec_version": spec.spec_version,
    }


_Z_BY_SYMBOL: dict[str, int] = {
    "H": 1, "B": 5, "C": 6, "N": 7, "O": 8, "F": 9, "Si": 14, "P": 15,
    "S": 16, "Cl": 17, "Br": 35, "I": 53,
}


def _element_to_z(elements: Sequence[str]) -> tuple[int, ...]:
    numbers: list[int] = []
    for symbol in elements:
        z = _Z_BY_SYMBOL.get(symbol)
        if z is None:
            raise PolicyAdapterError(f"unsupported element {symbol!r} in control case")
        numbers.append(z)
    return tuple(numbers)


def _covalent_terms(elements, geometry) -> list[Any]:
    from pes2ts_core.pesn.contracts.inputs import BondTerm

    n = len(elements)
    terms: list[BondTerm] = []
    for i in range(n):
        for j in range(i + 1, n):
            ri = _COVALENT_RADII.get(elements[i])
            rj = _COVALENT_RADII.get(elements[j])
            if ri is None or rj is None:
                continue
            if float(np.linalg.norm(geometry[i] - geometry[j])) <= ri + rj + _BOND_TOLERANCE:
                terms.append(BondTerm(i, j, 1.0))
    return terms


__all__ = [
    "CandidateList",
    "CandidatePolicyConfig",
    "ControlCaseAdapter",
    "EpsilonExplorePolicy",
    "ModelResidualPolicy",
    "PolicyAdapterError",
    "QGreedyPolicy",
    "RuleZeroPolicy",
    "build_residual_spec",
    "generate_candidates",
]
