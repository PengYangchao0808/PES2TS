"""Prepare reviewable alternative whole-system boundaries; never replace originals."""
import json,sys
from pathlib import Path
import numpy as np
from rdkit import Chem
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from run_demo24_synchronized import OUT,read,xyztext
from pes2ts_core.generation.planning.synchronized_path import align,digest
from pes2ts_core.integration.acp.scheduler_backend import save_receipt
rid='RXN_0000155302';source=read(ROOT/f'tests/fixtures/p0_demo24/records/{rid}.json')
base=read(OUT/rid/'PathPlan_v1.json');seed=np.asarray(base['reference_geometries'])
folder=OUT/rid/'boundary_proposal';folder.mkdir(exist_ok=True)
parser=Chem.SmilesParserParams();parser.removeHs=False
records=[]
for side,smiles,reference,key in zip(('R','P'),source['reaction_smiles'].split('>>'),(seed[4],seed[-5]),('r_coordinates','p_coordinates')):
    mol=Chem.MolFromSmiles(smiles,parser);original=np.asarray(source[key]);proposal=original.copy()
    rows={m:i for i,m in enumerate(source['maps'])};components=[]
    for fragment in Chem.GetMolFrags(mol):
        maps=[mol.GetAtomWithIdx(i).GetAtomMapNum() for i in fragment];indices=[rows[m] for m in maps]
        proposal[indices]=align(original[indices],reference[indices])
        components.append(indices)
    errors=[];closest=[]
    for i,indices in enumerate(components):
        a=original[indices];b=proposal[indices]
        errors.append(float(np.max(np.abs(np.linalg.norm(a[:,None]-a[None,:],axis=2)-np.linalg.norm(b[:,None]-b[None,:],axis=2)))))
        for js in components[i+1:]:
            d=np.linalg.norm(proposal[indices,None]-proposal[js][None,:],axis=2)
            left,right=np.unravel_index(np.argmin(d),d.shape)
            closest.append({'maps':[source['maps'][indices[left]],source['maps'][js[right]]],'distance_angstrom':float(d[left,right])})
    (folder/f'{side}_proposed.xyz').write_text(xyztext(proposal,source['elements'],f'{rid} {side} PROPOSAL ONLY; original full coordinates unchanged'))
    records.append({'endpoint':side,'original_sha256':digest(original.tolist()),'proposed_sha256':digest(proposal.tolist()),
                    'max_internal_pair_distance_change':max(errors),'cross_component_minimum_distances':closest,
                    'all_atom_aligned_rmsd_to_original':float(np.sqrt(np.mean((align(proposal,original)-original)**2)*3)),
                    'geometry':proposal.tolist()})
save_receipt(folder/'proposal.json',{'reaction_id':rid,'status':'review_only_not_used_for_calculations',
    'method':'rigid component placement against verified IRC-end reference; preserve every component internal geometry',
    'original_boundary_modified':False,'records':records})
print(json.dumps([{k:v for k,v in r.items() if k!='geometry'} for r in records],indent=2))
