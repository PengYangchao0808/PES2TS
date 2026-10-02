"""Endpoint-informed plans that use only complete connectivity changes."""
from __future__ import annotations

from dataclasses import asdict
import numpy as np
from rdkit import Chem

from pes2ts_core.g1.reaction_edit_graph import build_reaction_edit_graph
from pes2ts_core.generation.planning.continuation import ContinuationPolicy, validate_plan
from pes2ts_core.generation.planning.synchronized_path import digest, value


def input_screen(x, elements, bond_indices):
    periodic = Chem.GetPeriodicTable()
    radii = [periodic.GetRcovalent(periodic.GetAtomicNumber(e)) for e in elements]
    bonds = {tuple(sorted(p)) for p in bond_indices}
    failures = []
    for i in range(len(x)):
        for j in range(i+1, len(x)):
            d = float(np.linalg.norm(np.asarray(x[i])-x[j]))
            total = radii[i] + radii[j]
            if (i, j) in bonds:
                if d < .45*total or d > total+.8:
                    failures.append({"reason": "BONDED_GEOMETRY_INVALID", "atoms": [i, j], "distance": d})
            elif d < .55*total:
                failures.append({"reason": "NONBONDED_COLLISION", "atoms": [i, j], "distance": d})
    return failures


def choose_origin(snapshot, bundle):
    graph = build_reaction_edit_graph(bundle)
    active = [e for e in graph.edits if e.edit_kind in {"formed", "broken"}]
    if not active:
        return {"status": "OUT_OF_SCOPE_NO_CONNECTIVITY_EDIT"}
    order = {m: i for i, m in enumerate(snapshot["maps"])}
    rows = []
    for side, g in (("R", bundle.r_graph), ("P", bundle.p_graph)):
        x = np.asarray(snapshot["r_coordinates" if side == "R" else "p_coordinates"], float)
        bonds = [[order[e.map_a], order[e.map_b]] for e in g.edges]
        failures = input_screen(x, snapshot["elements"], bonds)
        count = sum((e.r_bond_order if side == "R" else e.p_bond_order) is not None for e in active)
        active_maps = {m for e in active for m in e.pair}
        components = sum(bool(active_maps & set(c.map_ids)) for c in g.components)
        rows.append({"side": side, "failures": failures,
                     "rank": (components, -count, 0 if side == "P" else 1), "bond_indices": bonds})
    valid = [r for r in rows if not r["failures"]]
    if not valid:
        return {"status": "INPUT_GEOMETRY_INVALID", "sides": rows}
    chosen = min(valid, key=lambda r: r["rank"])
    electronic = snapshot["endpoint_electronic"]
    states = [(electronic[s]["charge"], electronic[s]["multiplicity"]) for s in ("reactant", "product")]
    if states[0] != states[1]:
        return {"status": "ELECTRONIC_STATE_UNRESOLVED", "sides": rows}
    return {"status": "ready", "side": chosen["side"], "sides": rows,
            "bond_indices": chosen["bond_indices"], "charge": states[0][0], "multiplicity": states[0][1]}


def build_plan(snapshot, bundle, origin, prepared_geometry, *, reference_plan=None, policy=None, phase_profile=None, guards=()):
    """Freeze targets after preparing the origin; teacher use is explicit.

    reference_plan contributes target distances only, never execution seeds.
    Normal mode uses endpoint distances for intracomponent changes, and a
    radius-based separation target when the opposite atoms are fragments.
    """
    if origin["status"] != "ready":
        raise ValueError(origin["status"])
    periodic = Chem.GetPeriodicTable()
    graph = build_reaction_edit_graph(bundle)
    active = [e for e in graph.edits if e.edit_kind in {"formed", "broken"}]
    index = {m: i for i, m in enumerate(snapshot["maps"])}
    x = np.asarray(prepared_geometry, float)
    side = origin["side"]
    other_g = bundle.p_graph if side == "R" else bundle.r_graph
    opposite = np.asarray(snapshot["p_coordinates" if side == "R" else "r_coordinates"])
    radii = [periodic.GetRcovalent(periodic.GetAtomicNumber(e)) for e in snapshot["elements"]]
    drivers = []
    for edit in active:
        atoms = [index[m] for m in edit.pair]
        a, b = atoms
        q0 = value(x, "distance", atoms)
        bonded_at_start = (edit.r_bond_order if side == "R" else edit.p_bond_order) is not None
        if (q0 <= radii[a]+radii[b]+.45) != bonded_at_start:
            raise ValueError("ORIGIN_CONNECTIVITY_CHANGED")
        if reference_plan is not None:
            old = next(d for d in reference_plan["drivers"] if tuple(sorted(d["maps"])) == edit.pair)
            grid = list(reference_plan["lambda_values"])
            targets = list(old["values"])
            if side == "P":
                grid = [1-v for v in reversed(grid)]
                targets.reverse()
            # Preparation is a separate stage and this rebasing is frozen explicitly.
            targets[0] = q0
        else:
            same = any(set(edit.pair) <= set(c.map_ids) for c in other_g.components)
            q1 = value(opposite, "distance", atoms) if same else max(3., radii[a]+radii[b]+1.8)
            if not bonded_at_start:
                q1 = value(opposite, "distance", atoms)
            grid = [0., 1.]
            targets = [q0, q1]
        drivers.append({"id": "B_"+"_".join(map(str, edit.pair)), "kind": "distance",
                        "role": "driver", "edit_kind": edit.edit_kind, "maps": list(edit.pair),
                        "atoms": atoms, "unit": "angstrom", "lambda_values": grid, "values": targets})
    if phase_profile is not None:
        if reference_plan is not None or len(phase_profile) != len(drivers):
            raise ValueError("INVALID_PHASE_PROFILE")
        for d, phase in zip(drivers, phase_profile):
            if not np.isfinite(phase) or not 0 < phase < 1:
                raise ValueError("INVALID_PHASE_PROFILE")
            grid = np.linspace(0., 1., 101)
            sigma = .15
            curve = 1/(1+np.exp(-(grid-phase)/sigma))
            curve = (curve-curve[0])/(curve[-1]-curve[0])
            first, last = d["values"][0], d["values"][-1]
            d["lambda_values"] = grid.tolist()
            d["values"] = (first+(last-first)*curve).tolist()
            d["schedule"] = {"kind": "endpoint_normalized_sigmoid", "phase": phase, "sigma": sigma}
    bonded = {tuple(sorted(p)) for p in origin["bond_indices"]}
    fb_atoms = {tuple(sorted(d["atoms"])) for d in drivers}
    centres = {m for e in active for m in e.pair}
    supports = set(centres)
    for g in (bundle.r_graph, bundle.p_graph):
        for e in g.edges:
            if e.map_a in centres or e.map_b in centres:
                supports.update((e.map_a, e.map_b))
    checks = [{"atoms": [a, b], "minimum_distance": .55*(radii[a]+radii[b])}
              for a in range(len(x)) for b in range(a+1, len(x))
              if (a, b) not in bonded | fb_atoms]
    plan = {"schema_version": "pes2ts_connectivity_continuation_plan_v1",
            "scope": "connectivity_only", "parameter_dimension": 1,
            "reaction_id": snapshot["reaction_id"], "atom_map_order": snapshot["maps"],
            "elements": snapshot["elements"], "start_geometry": x.tolist(),
            "masses": [periodic.GetAtomicWeight(e) for e in snapshot["elements"]],
            "origin_endpoint": side, "execution_direction": "R_to_P" if side == "R" else "P_to_R",
            "charge": origin["charge"], "multiplicity": origin["multiplicity"],
            "method": "GFN2-xTB", "origin_source_sha256": digest(snapshot),
            "active_edits": [{"maps": list(e.pair), "edit_kind": e.edit_kind} for e in active],
            "derived_changes_archive": [e.to_record() for e in graph.edits if e.edit_kind == "order_changed"],
            "drivers": drivers, "guards": list(guards), "local_support_maps": sorted(supports), "collision_checks": checks,
            "policy": asdict(policy or ContinuationPolicy()),
            "schedule_source": "reference_target_control" if reference_plan is not None else "endpoint_graph_and_geometry",
            "reference_seed_used": False, "fixed_target_geometry_required": False,
            "blind_prediction_claimed": False,
            "reference_target_plan_sha256": digest(reference_plan) if reference_plan is not None else None}
    plan["phase_profile"] = phase_profile
    validate_plan(plan)
    plan["content_sha256"] = digest(plan)
    return plan
