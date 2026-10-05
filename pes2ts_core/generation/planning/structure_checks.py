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


#: Landing-identity tiers (G2-AB1 WP-5).  A geometric tier NEVER proves
#: chemical identity: only ACP OptTS/frequency/two-way IRC evidence can
#: (R2), and ``full_identity_verified`` is unreachable from geometry alone.
LANDING_IDENTITY_TIERS = ("target_fb_pattern", "adjacency_screen_only",
                          "full_identity_verified", "unknown")


def landing_identity(geometry, plan, *, target_bond_indices=None,
                     acp_validation_evidence=None):
    """Tiered landing identity for a (final) continuation frame.

    Tier semantics (a higher tier subsumes the lower ones' checks):

    - ``target_fb_pattern`` — every active driver distance matches its
      edit-kind's TARGET bonded/unbonded pattern under the covalent-radius
      threshold (distance mode only; a distance pattern, never an identity).
    - ``adjacency_screen_only`` — additionally the FULL endpoint adjacency
      screen (:func:`endpoint_geometry_identity`) passes against
      ``target_bond_indices``.  Still a distance screen: bond orders,
      stereo assignment, and electronic state are NOT verified.
    - ``full_identity_verified`` — ONLY when ``acp_validation_evidence``
      records a converged OptTS, a passed single-imaginary frequency, and
      matched two-way IRC.  Without it this tier is unreachable by design.
    - ``unknown`` — no tier is claimable; a mismatch is never an auto-pass
      into a weaker tier.

    Stereo semantics: ``spectator_stereo_checks`` guard only centres whose
    chirality is SPECIFIED on the endpoint; unspecified stereocentres cannot
    be guarded by signed volumes and are reported as unspecified.  Atom order
    follows ``plan["atom_map_order"]``; rows are never reordered.
    """
    from rdkit import Chem
    x = np.asarray(geometry, float)
    table = Chem.GetPeriodicTable()
    radii = [table.GetRcovalent(e) for e in plan["elements"]]
    driver_rows = []
    pattern_matched = True
    for driver in plan.get("drivers", []):
        a, b = driver["atoms"]
        distance = float(np.linalg.norm(x[a]-x[b]))
        observed_bonded = distance <= radii[a]+radii[b]+.45
        wants_bonded = driver["edit_kind"] == "formed"
        matched = observed_bonded == wants_bonded
        pattern_matched = pattern_matched and matched
        driver_rows.append({"atoms": [a, b], "maps": driver.get("maps"),
                            "edit_kind": driver["edit_kind"], "distance": distance,
                            "observed_bonded": observed_bonded,
                            "target_bonded": wants_bonded, "matched": matched})
    status = "target_fb_pattern" if pattern_matched and driver_rows else "unknown"
    scope = "active_fb_distance_pattern"
    adjacency = None
    if status == "target_fb_pattern" and target_bond_indices is not None:
        adjacency = endpoint_geometry_identity(x, plan["elements"], target_bond_indices)
        if adjacency["adjacency_screen_passed"]:
            status = "adjacency_screen_only"
            scope = "full_covalent_adjacency_screen"
        else:
            status = "unknown"
            scope = "full_covalent_adjacency_screen"
    if status in {"target_fb_pattern", "adjacency_screen_only"} and isinstance(
            acp_validation_evidence, dict):
        verified = (acp_validation_evidence.get("optts") == "converged"
                    and acp_validation_evidence.get("frequency") == "passed"
                    and acp_validation_evidence.get("irc_forward") == "matched"
                    and acp_validation_evidence.get("irc_reverse") == "matched")
        if verified:
            status = "full_identity_verified"
            scope = "acp_optts_frequency_two_way_irc"
    return {"landing_identity_status": status, "landing_identity_scope": scope,
            "driver_pattern": driver_rows,
            "adjacency_screen": adjacency,
            "target_fb_pattern_is_chemical_success": False,
            "complete_chemical_identity_verified": status == "full_identity_verified"}
