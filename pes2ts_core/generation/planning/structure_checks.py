"""Explicit spectator checks; fractional reaction-region bonds are allowed."""
from __future__ import annotations

import numpy as np


def add_online_checks(plan, snapshot, bundle):
    """Freeze spectator requirements without adding chemical drivers."""
    import copy
    from rdkit import Chem
    from pes2ts_core.generation.planning.synchronized_path import digest
    result = copy.deepcopy(plan)
    table = Chem.GetPeriodicTable()
    radii = [table.GetRcovalent(e) for e in plan["elements"]]
    index = {m: i for i, m in enumerate(plan["atom_map_order"])}
    common = ({tuple(sorted((e.map_a, e.map_b))) for e in bundle.r_graph.edges}
              & {tuple(sorted((e.map_a, e.map_b))) for e in bundle.p_graph.edges})
    active = {tuple(sorted(d["maps"])) for d in plan["drivers"]}
    checks = []
    for pair in sorted(common-active):
        a, b = (index[m] for m in pair)
        total = radii[a]+radii[b]
        checks.append({"maps": list(pair), "atoms": [a, b], "minimum_distance": .45*total,
                       "maximum_distance": total+.8})
    result["common_bond_checks"] = checks
    params = Chem.SmilesParserParams()
    params.removeHs = False
    sides = snapshot["reaction_smiles"].split(">")
    reactant, product = (Chem.MolFromSmiles(sides[i], params) for i in (0, -1))
    stereo = []
    centres = {m for d in plan["drivers"] for m in d["maps"]}
    support = set(plan.get("local_support_maps", [])) | centres
    for graph in (bundle.r_graph,bundle.p_graph):
        for edge in graph.edges:
            if edge.map_a in centres or edge.map_b in centres:
                support.update((edge.map_a,edge.map_b))
    result["local_support_maps"] = sorted(support)
    product_atoms = {a.GetAtomMapNum(): a for a in product.GetAtoms()}
    x = np.asarray(plan["start_geometry"])
    for atom in reactant.GetAtoms():
        m = atom.GetAtomMapNum()
        other = product_atoms.get(m)
        if (m in support or other is None or atom.GetChiralTag() == Chem.ChiralType.CHI_UNSPECIFIED
                or other.GetChiralTag() == Chem.ChiralType.CHI_UNSPECIFIED):
            continue
        neighbours = sorted(n.GetAtomMapNum() for n in atom.GetNeighbors())
        if len(neighbours) < 3 or neighbours != sorted(n.GetAtomMapNum() for n in other.GetNeighbors()):
            continue
        maps = [m]+neighbours[:3]
        atoms = [index[k] for k in maps]
        volume = float(np.linalg.det(x[atoms[1:]]-x[atoms[0]]))
        if abs(volume) > .05:
            stereo.append({"maps": maps, "atoms": atoms, "reference_sign": float(np.sign(volume)),
                           "minimum_volume": .01})
    result["spectator_stereo_checks"] = stereo
    result["online_identity_scope"] = "common_bond_geometry_and_nonreactive_mapped_stereocentres"
    result["content_sha256"] = digest({k:v for k,v in result.items() if k != "content_sha256"})
    return result


def structure_issues(geometry, plan):
    x = np.asarray(geometry, float)
    issues = []
    for check in plan.get("common_bond_checks", []):
        a, b = check["atoms"]
        distance = float(np.linalg.norm(x[a]-x[b]))
        if not check["minimum_distance"] <= distance <= check["maximum_distance"]:
            issues.append({"reason": "SPECTATOR_BOND_GEOMETRY", "atoms": [a, b],
                           "maps": check.get("maps"), "distance": distance,
                           "minimum_distance": check["minimum_distance"],
                           "maximum_distance": check["maximum_distance"]})
    for check in plan.get("spectator_stereo_checks", []):
        centre, a, b, c = check["atoms"]
        volume = float(np.linalg.det(x[[a, b, c]]-x[centre]))
        if volume*check["reference_sign"] <= check["minimum_volume"]:
            issues.append({"reason": "SPECTATOR_STEREO_CHANGED", "maps": check["maps"],
                           "signed_volume_angstrom3": volume})
    return issues


def endpoint_geometry_identity(geometry, elements, bond_indices):
    """Full adjacency screen only; does not certify bond orders/spin/minimum."""
    from rdkit import Chem
    x = np.asarray(geometry, float)
    table = Chem.GetPeriodicTable()
    radii = [table.GetRcovalent(e) for e in elements]
    expected = {tuple(sorted(pair)) for pair in bond_indices}
    issues = []
    for a in range(len(x)):
        for b in range(a+1, len(x)):
            distance = float(np.linalg.norm(x[a]-x[b]))
            threshold = radii[a]+radii[b]+.45
            observed = distance <= threshold
            if observed != ((a, b) in expected):
                issues.append({"atoms": [a, b], "distance": distance,
                               "expected_bonded": (a, b) in expected, "observed_bonded": observed})
    return {"adjacency_screen_passed": not issues, "issues": issues,
            "complete_chemical_identity_verified": False}
