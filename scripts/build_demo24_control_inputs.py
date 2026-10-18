"""Build the Demo24 control-round inputs (Demo24 real-RL plan §3/§4 step 0).

Freezes, under ``outputs/pesn/demo24_rl_v1_20261007/inputs/``:

* a repaired case snapshot for ``RXN_0000155302`` — the historical
  ``INPUT_GEOMETRY_INVALID`` endpoint collisions (C-Cl 0.535 A on R, two
  Cl-H contacts on P) are separated by rigid per-component translation
  (``repair_component_overlap``), producing a NEW snapshot identity whose
  provenance binds the original source geometry hashes
  (``apply_input_repair``); the original staging snapshot is never touched;
* a new frozen campaign manifest (same cases; only RXN_0000155302's
  snapshot_path/sha256 point at the repaired snapshot);
* the train_collect16 and execute24 selection manifests (execute roles
  carry split/stratum from the frozen candidate CSV);
* the frozen origin-repair recipe for the two historical startpoint cases
  (registered BEFORE any origin execution; the recipe is a new frozen
  identity, not a silent budget raise of the historical campaign).

No QC runs here and no case is replaced; the roster stays the original 24.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from pes2ts_core.generation.planning.connectivity_plan import choose_origin, input_screen  # noqa: E402
from pes2ts_core.generation.planning.graph_rebuild import (  # noqa: E402
    load_endpoint_materials_from_export,
    rebuild_endpoint_graphs,
)
from pes2ts_core.generation.planning.origin_preparation import (  # noqa: E402
    apply_input_repair,
    repair_component_overlap,
)
from pes2ts_core.generation.planning.synchronized_path import digest  # noqa: E402

CANDIDATE_CSV = REPO_ROOT / "data/manifests/pes2ts_demo24_candidates_v1.csv"
CAMPAIGN_MANIFEST = REPO_ROOT / "data/manifests/pes2ts_g2t_campaign_s3_full.json"
ENDPOINT_COLLISION_CASE = "RXN_0000155302"
ORIGIN_RECIPE = {
    "schema_version": "demo24_origin_repair_recipe_v1",
    "method": "GFN2-xTB",
    "nproc": 4,
    "timeout_seconds": 1800.0,
    "capability_probe": True,
    "note": (
        "New frozen recipe for the two historical startpoint cases "
        "(RXN_0000155302 endpoint repair + origin free-opt; RXN_0000171675 "
        "origin free-opt retry). Registered before execution; the historical "
        "campaign recipe (nproc=2, 600 s) is not silently relaxed and this "
        "preparation cost is billed separately from the 22 h scan envelope."
    ),
}

#: Host-pinned ACP executables (mirrors ~/.cccp.yaml, whose ORCA-6.1.1 pin is
#: the working host configuration: ORCA 5.0.4 segfaults under PAL4+openmpi418
#: per the host owner's note).  Written into an explicit round ACP config so
#: the optimizer identity is versioned and hash-bound instead of depending on
#: root's home config; a bare --config defaults.yaml would deep-merge into the
#: broken 5.0.4-binary-plus-6.1.1-libs combination.
ACP_EXECUTABLE_PINS = {
    "orca": {
        "path": "/opt/orca_6_1_1/orca",
        "ld_library_path": "/opt/orca_6_1_1:/opt/openmpi418/lib",
    },
    "crest": {"path": "/opt/software/crest/crest"},
    "xtb": {"path": "/opt/software/xtb/bin/xtb"},
    "shermo": {"path": "/opt/software/shermo/Shermo"},
}


def _write_round_acp_config(out_dir: Path) -> Path:
    import yaml

    acp_root = Path(
        json.loads(CAMPAIGN_MANIFEST.read_text(encoding="utf-8"))["acp_wiring"]["root"]
    )
    source = acp_root / "config" / "defaults.yaml"
    config = yaml.safe_load(source.read_text(encoding="utf-8"))
    executables = dict(config.get("executables") or {})
    for name, pins in ACP_EXECUTABLE_PINS.items():
        executables[name] = {**(executables.get(name) or {}), **pins}
    config["executables"] = executables
    target = out_dir / "acp_config_demo24.yaml"
    target.write_text(yaml.safe_dump(config, sort_keys=True), encoding="utf-8")
    return target


def _read_csv_rows() -> list[dict[str, str]]:
    with CANDIDATE_CSV.open(encoding="utf-8-sig") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def repair_collision_snapshot(snapshot: dict) -> dict:
    """Separate colliding endpoint components; new identity, bound provenance."""
    bundle = rebuild_endpoint_graphs(
        snapshot["reaction_smiles"], load_endpoint_materials_from_export(snapshot)
    )
    order = {int(m): i for i, m in enumerate(snapshot["maps"])}
    repaired = dict(snapshot)
    provenance: dict[str, dict] = {}
    for side, graph in (("r", bundle.r_graph), ("p", bundle.p_graph)):
        key = f"{side}_coordinates"
        geometry = np.asarray(snapshot[key], float)
        bonds = [[order[e.map_a], order[e.map_b]] for e in graph.edges]
        failures = input_screen(geometry, snapshot["elements"], bonds)
        if not failures:
            continue
        components = [
            sorted(order[m] for m in component.map_ids)
            for component in graph.components
        ]
        separated, report = repair_component_overlap(
            geometry, snapshot["elements"], components
        )
        if separated is None or report.get("status") not in ("repaired", "unchanged"):
            raise SystemExit(
                f"{ENDPOINT_COLLISION_CASE}: {side}-side component separation failed: {report}"
            )
        residual_failures = input_screen(separated, snapshot["elements"], bonds)
        if residual_failures:
            raise SystemExit(
                f"{ENDPOINT_COLLISION_CASE}: {side} still fails after repair: "
                f"{residual_failures}"
            )
        repair = apply_input_repair(
            geometry, separated, "component_overlap_separation",
            {"screen_failures_before": failures, "separation_report": report},
        )
        repaired[key] = repair["geometry"]
        provenance[key] = repair["provenance"]
    if not provenance:
        raise SystemExit(f"{ENDPOINT_COLLISION_CASE}: nothing to repair?")
    repaired["startpoint_repair_provenance"] = {
        "schema_version": "demo24_startpoint_repair_v1",
        "case": ENDPOINT_COLLISION_CASE,
        "repair_kind": "endpoint_component_separation",
        "sides": provenance,
        "original_snapshot_unchanged": True,
    }
    return repaired


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root", type=Path,
        default=REPO_ROOT / "outputs/pesn/demo24_rl_v1_20261007/inputs",
    )
    args = parser.parse_args()
    out = args.output_root
    out.mkdir(parents=True, exist_ok=True)

    rows = _read_csv_rows()
    roles = {
        row["reaction_id"]: {"split": row["split"], "stratum": row["stratum"]}
        for row in rows
    }
    train_ids = sorted(r for r, role in roles.items() if role["split"] == "train")
    valid_ids = sorted(r for r, role in roles.items() if role["split"] == "valid")
    assert len(train_ids) == 16 and len(valid_ids) == 8, (len(train_ids), len(valid_ids))

    manifest = json.loads(CAMPAIGN_MANIFEST.read_text(encoding="utf-8"))
    case_entries = {case["reaction_id"]: dict(case) for case in manifest["cases"]}

    repaired_dir = out / "snapshots"
    repaired_dir.mkdir(parents=True, exist_ok=True)
    for case in case_entries.values():
        if case["reaction_id"] != ENDPOINT_COLLISION_CASE:
            continue
        original_path = (
            Path(case["snapshot_path"])
            if Path(case["snapshot_path"]).is_absolute()
            else REPO_ROOT / case["snapshot_path"]
        )
        snapshot = json.loads(original_path.read_text(encoding="utf-8"))
        repaired_snapshot = repair_collision_snapshot(snapshot)
        repaired_path = repaired_dir / f"{case['reaction_id']}.json"
        repaired_path.write_text(
            json.dumps(repaired_snapshot, ensure_ascii=False, sort_keys=True, indent=2),
            encoding="utf-8",
        )
        case["snapshot_path"] = str(repaired_path)
        case["snapshot_sha256"] = _sha256_file(repaired_path)
        print(
            f"repaired snapshot: {repaired_path} "
            f"(sha256 {case['snapshot_sha256'][:16]}...)"
        )

    manifest["cases"] = [case_entries[rid] for rid in sorted(case_entries)]
    round_acp_config = _write_round_acp_config(out)
    manifest["acp_wiring"] = {
        **manifest["acp_wiring"],
        "config_path": str(round_acp_config),
    }
    manifest["content_sha256"] = digest(
        {k: v for k, v in manifest.items() if k != "content_sha256"}
    )
    manifest_path = out / "campaign_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2),
        encoding="utf-8",
    )

    from pes2ts_core.generation.control_round import build_selection_manifest

    build_selection_manifest(
        reaction_ids=train_ids,
        campaign_manifest_path=manifest_path,
        output_path=out / "selection_train16.json",
        purpose="train_collect",
        note="Demo24 train-qualified collection list (12 fit / 4 dev by reaction group)",
        reaction_roles={rid: roles[rid] for rid in train_ids},
    )
    build_selection_manifest(
        reaction_ids=sorted(roles),
        campaign_manifest_path=manifest_path,
        output_path=out / "selection_execute24.json",
        purpose="execute24",
        note=(
            "Demo24 full execution list; an execute list is NEVER a training "
            "permission - train transitions are gated again at load time"
        ),
        reaction_roles=roles,
    )
    recipe_path = out / "origin_repair_recipe_v1.json"
    recipe_path.write_text(
        json.dumps(ORIGIN_RECIPE, ensure_ascii=False, sort_keys=True, indent=2),
        encoding="utf-8",
    )

    summary = {
        "schema_version": "demo24_control_inputs_v1",
        "campaign_manifest": str(manifest_path),
        "campaign_manifest_sha256": _sha256_file(manifest_path),
        "selection_train16": str(out / "selection_train16.json"),
        "selection_execute24": str(out / "selection_execute24.json"),
        "origin_repair_recipe": str(recipe_path),
        "repaired_cases": [ENDPOINT_COLLISION_CASE],
        "candidate_csv_sha256": _sha256_file(CANDIDATE_CSV),
        "n_train": len(train_ids),
        "n_valid": len(valid_ids),
    }
    (out / "input_manifest.json").write_text(
        json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
