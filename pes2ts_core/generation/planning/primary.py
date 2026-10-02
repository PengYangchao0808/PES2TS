"""Legacy single-bond baseline and one-primary candidate selection.

New full-boundary paths use synchronized_path and never primary_stretch.
Selection uses endpoint graphs and geometries only. Backend capability and
human authorization are separate execution gates, never inferred here.
"""
from __future__ import annotations

from collections.abc import Mapping
import math
from typing import Any

from rdkit import Chem

from pes2ts_core.g1.endpoint_graph import EndpointGraphBundle
from pes2ts_core.generation.planning.coordinate_pool import EndpointMaterials
from pes2ts_core.generation.planning.connectivity import required_pairs


def primary_stretch(
    bundle: EndpointGraphBundle,
    materials: EndpointMaterials,
    *,
    anchor: Mapping[str, Any] | None = None,
    max_step: float = 0.15,
    max_points: int = 40,
) -> dict[str, Any]:
    """Return exactly one primary hypothesis, or an explicit blocked outcome.

An optional caller-selected anchor is checked against the graph and start
geometry. Its provenance belongs to the caller's development-run manifest;
this function never reads TS/IRC or treats that anchor as a chemistry review.
"""
    if not math.isfinite(max_step) or max_step <= 0 or max_points < 2:
        raise ValueError("invalid primary scan point budget")
    graphs = {"R": bundle.r_graph, "P": bundle.p_graph}
    xyz = {"R": materials.r_coordinates, "P": materials.p_coordinates}
    orders = {side: {(e.map_a, e.map_b): e.bond_order for e in graph.edges} for side, graph in graphs.items()}
    elements = {node.map_id: node.element for node in bundle.r_graph.nodes}
    periodic = Chem.GetPeriodicTable()
    choices = []
    for side, other in (("R", "P"), ("P", "R")):
        for pair in sorted(set(orders[side])-set(orders[other])):
            if any(m not in xyz[side] for m in pair):
                continue
            start = math.dist(xyz[side][pair[0]], xyz[side][pair[1]])
            radii = sum(periodic.GetRcovalent(periodic.GetAtomicNumber(elements[m])) for m in pair)
            if not math.isfinite(start) or start < 0.5*radii or start > radii+0.45:
                continue
            # Different source components have independent coordinate frames.
            # Prefer a complete, connected bonded side over an association side.
            components = len(graphs[side].components)
            contains_h = any(elements[m] == "H" for m in pair)
            end = max(start+(1.6 if contains_h else 2.0), radii+(1.8 if contains_h else 2.2))
            # Intracomponent endpoint distances are meaningful; intercomponent
            # distances are deliberately not used as a scan target.
            same_component = any(set(pair)<=set(c.map_ids) for c in graphs[other].components)
            if same_component and all(m in xyz[other] for m in pair):
                opposite = math.dist(xyz[other][pair[0]], xyz[other][pair[1]])
                if math.isfinite(opposite) and opposite <= 5.0:
                    end = max(end, opposite+0.3)
            choices.append({"maps":list(pair), "start_endpoint":side, "start":start, "end":end,
                            "rank":(components, 0 if contains_h else 1, end-start, tuple(elements[m] for m in pair), pair, side)})
    if anchor is not None:
        pair=sorted(int(m) for m in anchor.get("maps", []))
        side=str(anchor.get("start_endpoint", ""))
        choices=[c for c in choices if c["maps"]==pair and c["start_endpoint"]==side]
        if not choices:
            raise ValueError("selected primary anchor is not a valid changed bond at its start endpoint")
    if not choices:
        return {"status":"blocked", "reason":"NO_VALID_BONDED_STRETCH_ANCHOR", "primary":None}
    selected=min(choices,key=lambda c:c["rank"])
    n_points=max(16,1+math.ceil((selected["end"]-selected["start"])/max_step))
    if n_points>max_points:
        return {"status":"blocked", "reason":"PRIMARY_POINT_BUDGET_EXCEEDED", "primary":None}
    maps=selected["maps"]
    start_side=selected['start_endpoint']
    geometry={int(m):tuple(x) for m,x in xyz[start_side].items()}
    components=sorted(graphs[start_side].components,
                      key=lambda c:(not set(maps)<=set(c.map_ids),c.component_id))
    # Independently optimized components do not share an origin. Keep the
    # primary bonded component intact and translate other components rigidly
    # into a separated common frame. Do not use their old relative distances.
    placed=list(components[0].map_ids)
    placements=[]
    for component in components[1:]:
        left=max(geometry[m][0] for m in placed)+6.0
        shift=left-min(geometry[m][0] for m in component.map_ids)
        for m in component.map_ids:
            x=geometry[m]
            geometry[m]=(x[0]+shift,x[1],x[2])
        placed.extend(component.map_ids)
        placements.append({'component_id':component.component_id,'translation':[shift,0.0,0.0]})
    for edge in graphs[start_side].edges:
        distance=math.dist(geometry[edge.map_a],geometry[edge.map_b])
        radii=sum(periodic.GetRcovalent(periodic.GetAtomicNumber(elements[m])) for m in (edge.map_a,edge.map_b))
        if distance>radii+0.45 or distance<0.5*radii:
            return {"status":"blocked","reason":"PRIMARY_ENDPOINT_BOND_GEOMETRY_INVALID","primary":None}
    return {"status":"hypothesis", "reason":None, "primary":{
        "candidate_id":"primary-0000", "role":"primary", "mode":"SINGLE_1D",
        "start_endpoint":selected["start_endpoint"],
        "direction":"R_to_P" if selected["start_endpoint"]=="R" else "P_to_R",
        "kind":"B", "maps":maps, "unit":"angstrom", "start":selected["start"],
        "end":selected["end"], "n_points":n_points, "schedule_kind":"linear",
        "geometry":{str(m):list(x) for m,x in geometry.items()},
        "component_placements":placements,
        "monitored_edits":[list(p) for p in sorted(set(orders['R'])|set(orders['P']))
                           if orders['R'].get(p)!=orders['P'].get(p) and list(p)!=maps],
        "selection_rule":"bonded_changed_anchor_positive_stretch_connected_start_v1",
        "execution_eligible":False,
    }}


def select_primary_candidate(candidates: list[dict[str, Any]], bundle: EndpointGraphBundle, *, connectivity_only: bool = False) -> dict[str, Any]:
    """Promote one archived candidate as a hypothesis, never a ready plan."""
    orders=[{tuple(sorted((e.map_a,e.map_b))):e.bond_order for e in graph.edges}
            for graph in (bundle.r_graph,bundle.p_graph)]
    required=required_pairs(bundle)
    if not required:
        return {"status":"blocked", "reason":"OUT_OF_SCOPE_NO_CONNECTIVITY_EDIT", "primary":None}
    synchronized=[]
    for candidate in candidates:
        drivers=candidate.get('drivers',[])
        controlled={tuple(sorted(d['maps'])) for d in drivers if d.get('kind')=='B'}
        if drivers and all(d.get('kind')=='B' for d in drivers) and required==controlled and len(drivers)==len(controlled):
            synchronized.append(candidate)
    if synchronized:
        selected=min(synchronized,key=lambda c:(len(c.get('failure_reasons',[])),c['candidate_id']))
        return {'status':'hypothesis','reason':None,'primary':{
            'candidate_id':selected['candidate_id'],'source_candidate_id':selected['candidate_id'],
            'role':'primary','mode':'SYNCHRONIZED_1D','parameter_dimension':1,
            'drivers':selected['drivers'],'execution_eligible':False,
            'boundary_verification_required':True}}
    if connectivity_only:
        return {'status': 'blocked', 'reason': 'INCOMPLETE_CONNECTIVITY_DRIVER_COVERAGE', 'primary': None}
    options=[]
    graphs={"R":bundle.r_graph,"P":bundle.p_graph}
    bonds={s:{(e.map_a,e.map_b) for e in g.edges} for s,g in graphs.items()}
    for c in candidates:
        ds=c.get("drivers", [])
        if len(ds)!=1 or ds[0].get("kind")!="B" or c.get("schedule_kind")!="linear":
            continue
        side=c["start_endpoint"]
        pair=tuple(sorted(ds[0]["maps"]))
        other="P" if side=="R" else "R"
        if pair not in bonds[side] or pair in bonds[other]:
            continue
        coords=c.get("extensions",{}).get("assembled_endpoint_coordinates")
        if not coords:
            continue
        mats=EndpointMaterials({int(k):tuple(v) for k,v in coords['R'].items()},
                              {int(k):tuple(v) for k,v in coords['P'].items()})
        try:
            selection=primary_stretch(bundle,mats,anchor={"maps":pair,"start_endpoint":side})
        except ValueError:
            continue
        if selection["primary"]:
            p=selection["primary"]
            hard={x["code"] for x in c["failure_reasons"]} - {
                "PRUNED_BY_BUDGET","ROUTE_REVIEW_REQUIRED","DIRECTION_NOT_READY",
                "CAPABILITY_UNPROBED","BACKEND_SMOKE_REQUIRED","BACKEND_CAPABILITY_MISSING"}
            options.append(((bool(hard),len(graphs[side].components),p['end']-p['start'],pair,side),c['candidate_id'],p))
    if not options:
        return {"status":"blocked","reason":"NO_PRIMARY_STRETCH_CANDIDATE","primary":None}
    _,source_id,p=min(options,key=lambda x:x[0])
    p['source_candidate_id']=source_id
    return {"status":"hypothesis","reason":None,"primary":p}
