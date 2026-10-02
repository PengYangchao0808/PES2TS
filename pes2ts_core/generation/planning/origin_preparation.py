"""Versioned, topology-preserving input repairs and torsional hypotheses."""
from __future__ import annotations

import numpy as np


def repair_component_overlap(geometry, elements, components, *, max_iterations=200):
    from rdkit import Chem
    x = np.asarray(geometry,float).copy()
    original = x.copy()
    table=Chem.GetPeriodicTable()
    covalent=[table.GetRcovalent(e) for e in elements]
    vdw=[table.GetRvdw(e) for e in elements]
    groups=[list(group) for group in components]
    if sorted(i for group in groups for i in group) != list(range(len(x))):
        raise ValueError("INVALID_COMPONENT_PARTITION")
    translations=np.zeros((len(groups),3))
    for iteration in range(max_iterations):
        worst=None
        for a in range(len(groups)):
            for b in range(a+1,len(groups)):
                for i in groups[a]:
                    for j in groups[b]:
                        separation=max(covalent[i]+covalent[j]+.8,.75*(vdw[i]+vdw[j]))
                        distance=np.linalg.norm(x[j]-x[i])
                        shortage=separation-distance
                        if shortage > 1e-6 and (worst is None or shortage>worst[0]):
                            worst=(shortage,a,b,i,j,distance)
        if worst is None:
            return x,{"status":"repaired" if np.any(translations) else "unchanged",
                      "iterations":iteration,"component_translations_angstrom":translations.tolist(),
                      "internal_geometry_preserved":all(np.allclose(
                          x[group]-x[group[0]],original[group]-original[group[0]]) for group in groups)}
        shortage,a,b,i,j,distance=worst
        direction=(x[j]-x[i])/distance if distance>1e-10 else np.eye(3)[(a+b)%3]
        movement=(shortage+.001)*direction
        x[groups[b]]+=movement
        translations[b]+=movement
    return None,{"status":"component_placement_failed","iterations":max_iterations}


def torsion_hypotheses(geometry, bonds, *, maximum=2, rotatable_bonds=None):
    """Rotate a disconnected side of a non-ring internal bond; no atom jitter."""
    x=np.asarray(geometry,float)
    neighbours={i:set() for i in range(len(x))}
    for a,b in bonds:
        neighbours[a].add(b);neighbours[b].add(a)
    rows=[]
    eligible = {tuple(sorted(pair)) for pair in (rotatable_bonds if rotatable_bonds is not None else bonds)}
    for a,b in sorted(tuple(sorted(pair)) for pair in bonds):
        if (a,b) not in eligible:
            continue
        if len(neighbours[a])<2 or len(neighbours[b])<2:
            continue
        visited={b};queue=[b]
        for node in queue:
            for other in neighbours[node]:
                if {node,other}=={a,b} or other in visited:
                    continue
                visited.add(other);queue.append(other)
        if a in visited:
            continue
        axis=x[b]-x[a];norm=np.linalg.norm(axis)
        if norm<1e-10:
            continue
        axis/=norm
        for degrees in (60.,-60.):
            angle=np.radians(degrees)
            ids=sorted(visited)
            v=x[ids]-x[a]
            rotated=v*np.cos(angle)+np.cross(np.tile(axis,(len(ids),1)),v)*np.sin(angle)+np.outer(v@axis,axis)*(1-np.cos(angle))
            trial=x.copy();trial[ids]=rotated+x[a]
            rows.append((trial,{"rotation_bond_atoms":[a,b],"rotation_degrees":degrees,"rotated_atoms":ids}))
            if len(rows)>=maximum:
                return rows
    return rows
