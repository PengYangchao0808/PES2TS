"""Runtime ``predict`` subcommand: load a bundle, predict one JSON input."""

from __future__ import annotations

import json
from pathlib import Path

from pes2ts_core.pesn.contracts.inputs import (
    BondTerm,
    CoordinateDef,
    CoordinateSpec,
    TaskEdit,
    TaskGraph,
    build_pesn_input,
)


def _build_input_from_json(document: dict) -> object:
    task_graph = TaskGraph(
        schema_version=str(document["task_graph"]["schema_version"]),
        atomic_numbers=tuple(int(z) for z in document["task_graph"]["atomic_numbers"]),
        atom_map=tuple(int(m) for m in document["task_graph"]["atom_map"]),
        reactant_bonds=tuple(
            BondTerm(int(i), int(j), float(o))
            for i, j, o in document["task_graph"].get("reactant_bonds", [])
        ),
        product_bonds=tuple(
            BondTerm(int(i), int(j), float(o))
            for i, j, o in document["task_graph"].get("product_bonds", [])
        ),
        edits=tuple(
            TaskEdit(str(k), int(i), int(j), float(ro), float(po))
            for k, i, j, ro, po in document["task_graph"].get("edits", [])
        ),
        source=str(document["task_graph"]["source"]),
        provenance=dict(document["task_graph"].get("provenance", {})),
    )
    coordinate_spec = CoordinateSpec(
        schema_version=str(document["coordinate_spec"]["schema_version"]),
        coords=tuple(
            CoordinateDef(
                kind=str(c["kind"]),
                atoms=tuple(int(a) for a in c["atoms"]),
                units=str(c["units"]),
                role=str(c["role"]),
                scale_a=float(c["scale_a"]),
                ref_q=float(c["ref_q"]),
                label=str(c["label"]),
            )
            for c in document["coordinate_spec"]["coords"]
        ),
        spec_version=str(document["coordinate_spec"]["spec_version"]),
    )
    return build_pesn_input(
        input_id=str(document["input_id"]),
        task_graph=task_graph,
        coordinate_spec=coordinate_spec,
        coordinates=document["coordinates"],
        charge=document.get("charge"),
        multiplicity=document.get("multiplicity"),
        method=document.get("method"),
    )


def cmd_predict(args) -> int:
    from pes2ts_core.pesn.api import load_bundle

    try:
        predictor = load_bundle(args.bundle)
        prediction_input = _build_input_from_json(
            json.loads(Path(args.input).read_text(encoding="utf-8"))
        )
        prediction = predictor.predict_point(prediction_input)
    except Exception as exc:
        print(f"predict failed: {type(exc).__name__}: {exc}")
        return 30
    payload = {
        "schema_version": prediction.schema_version,
        "input_hash": prediction.input_hash,
        "model_id": prediction.model_id,
        "model_hash": prediction.model_hash,
        "signed_progress": {
            "available": prediction.signed_progress.available,
            "value": prediction.signed_progress.value,
            "reason": prediction.signed_progress.reason,
        },
        "path_proximity": {
            "available": prediction.path_proximity.available,
            "value": prediction.path_proximity.value,
        },
        "path_score": {
            "available": prediction.path_score.available,
            "value": prediction.path_score.value,
        },
        "teacher_direction": {
            "available": prediction.teacher_direction.available,
            "values": list(prediction.teacher_direction.values),
            "reason": prediction.teacher_direction.reason,
        },
        "coordinate_activity": {
            "available": prediction.coordinate_activity.available,
            "values": list(prediction.coordinate_activity.values),
            "reason": prediction.coordinate_activity.reason,
        },
        "capabilities": predictor.capabilities(),
        "research_only": prediction.research_only,
    }
    text = json.dumps(payload, indent=2)
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
    else:
        print(text)
    return 0


__all__ = ["cmd_predict"]
