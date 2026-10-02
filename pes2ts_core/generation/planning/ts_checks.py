"""Independent TS mode and mapped endpoint identity checks."""
from __future__ import annotations

import numpy as np

from pes2ts_core.generation.planning.continuation import _jacobian


def assess_mode(geometry, drivers, frequencies, modes, *, threshold=-30.):
    meaningful = [index for index, frequency in frequencies.items() if frequency < threshold]
    record = {"frequencies_cm1": frequencies, "significance_threshold_cm1": threshold,
              "meaningful_imaginary_mode_indices": meaningful, "single_meaningful_imaginary": len(meaningful)==1,
              "small_negative_modes": {i:f for i,f in frequencies.items() if threshold <= f < 0},
              "target_mode_screen_passed": False, "irc_connection_verified": False}
    if len(meaningful) != 1:
        return record
    mode = modes.get(meaningful[0])
    if mode is None:
        record["mode_status"] = "missing"
        return record
    mode = np.asarray(mode, float)
    if mode.shape != np.asarray(geometry).shape or not np.isfinite(mode).all() or np.linalg.norm(mode) < 1e-12:
        record["mode_status"] = "invalid"
        return record
    mode /= np.linalg.norm(mode)
    changes = _jacobian(np.asarray(geometry), drivers)@mode.ravel()
    record["active_distance_mode_derivatives"] = changes.tolist()
    record["mode_status"] = "bound"
    # A permissive relevance screen; IRC, not this scalar, establishes identity.
    record["target_mode_screen_passed"] = bool(np.linalg.norm(changes) > .05)
    return record


def mapped_endpoint_identity(geometry, snapshot, side):
    """Infer bonds independently, compare mapped order/charge/radical/stereo."""
    from rdkit import Chem
    from rdkit.Chem import rdDetermineBonds
    params = Chem.SmilesParserParams(); params.removeHs = False
    text = snapshot["reaction_smiles"].split(">")[0 if side == "R" else -1]
    expected = Chem.MolFromSmiles(text, params)
    xyz = "\n".join([str(len(snapshot["elements"])), "endpoint"]+
        [e+" "+" ".join(f"{v:.12f}" for v in row) for e,row in zip(snapshot["elements"], geometry)])
    raw = Chem.MolFromXYZBlock(xyz)
    state = snapshot["endpoint_electronic"]["reactant" if side == "R" else "product"]
    if expected is None or raw is None:
        return {"matched": False, "reason": "IDENTITY_PARSE_FAILED"}
    def signature(mol):
        Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
        atoms = {a.GetAtomMapNum(): (a.GetAtomicNum(), a.GetFormalCharge(), a.GetNumRadicalElectrons()) for a in mol.GetAtoms()}
        bonds = {tuple(sorted((b.GetBeginAtom().GetAtomMapNum(), b.GetEndAtom().GetAtomMapNum()))):
                 float(b.GetBondTypeAsDouble()) for b in mol.GetBonds()}
        stereo = {a.GetAtomMapNum(): a.GetProp("_CIPCode") for a in mol.GetAtoms() if a.HasProp("_CIPCode")}
        bond_stereo = {tuple(sorted((b.GetBeginAtom().GetAtomMapNum(),b.GetEndAtom().GetAtomMapNum()))):str(b.GetStereo())
                       for b in mol.GetBonds() if b.GetStereo() in {Chem.BondStereo.STEREOE,Chem.BondStereo.STEREOZ}}
        return atoms, bonds, stereo, bond_stereo
    ea, eb, es, ebs = signature(expected)
    diagnostics = []
    for charged in (True, False):
        observed = Chem.Mol(raw)
        for atom, mapped in zip(observed.GetAtoms(), snapshot["maps"]):
            atom.SetAtomMapNum(mapped)
        try:
            rdDetermineBonds.DetermineBonds(observed, charge=state["charge"], allowChargedFragments=charged,
                                            embedChiral=True)
            oa, ob, os, obs = signature(observed)
        except (ValueError, RuntimeError) as exc:
            diagnostics.append({"allow_charged_fragments": charged, "reason": str(exc)})
            continue
        atom_match, bond_match = oa == ea, ob == eb
        stereo_match = (all(os.get(m) == stereo for m,stereo in es.items())
                        and all(obs.get(pair)==stereo for pair,stereo in ebs.items()))
        matched = atom_match and bond_match and stereo_match
        diagnostics.append({"allow_charged_fragments": charged, "atom_charge_radical_match": atom_match,
                            "bond_order_match": bond_match, "specified_stereo_match": stereo_match,
                            "inferred_mapped_smiles": Chem.MolToSmiles(observed)})
        if matched:
            return {"matched": True, "side": side, "charge": state["charge"],
                    "multiplicity": state["multiplicity"], "checks": diagnostics,
                    "scf_stability_verified": False}
    return {"matched": False, "side": side, "reason": "MAPPED_CHEMICAL_IDENTITY_MISMATCH", "checks": diagnostics}
