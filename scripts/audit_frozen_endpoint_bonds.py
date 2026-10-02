"""Chemistry sanity audit of immutable Demo24 endpoints, without repairing them."""
import json,sys
from pathlib import Path
import numpy as np
from rdkit import Chem
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from pes2ts_core.integration.acp.scheduler_backend import save_receipt
periodic=Chem.GetPeriodicTable();records=[]
parser=Chem.SmilesParserParams();parser.removeHs=False
for path in sorted((ROOT/'tests/fixtures/p0_demo24/records').glob('RXN_*.json')):
    s=json.loads(path.read_text());row={m:i for i,m in enumerate(s['maps'])};sides=[]
    for side,smiles in zip(('R','P'),s['reaction_smiles'].split('>>')):
        xyz=np.asarray(s['r_coordinates' if side=='R' else 'p_coordinates']);mol=Chem.MolFromSmiles(smiles,parser)
        failures=[]
        for b in mol.GetBonds():
            a,c=b.GetBeginAtom(),b.GetEndAtom();maps=[a.GetAtomMapNum(),c.GetAtomMapNum()]
            q=float(np.linalg.norm(xyz[row[maps[0]]]-xyz[row[maps[1]]]))
            radii=sum(periodic.GetRcovalent(atom.GetAtomicNum()) for atom in (a,c))
            if q<.5*radii or q>radii+.45:
                failures.append({'maps':maps,'elements':[a.GetSymbol(),c.GetSymbol()],
                                 'distance_angstrom':q,'covalent_radii_sum':radii})
        component={m:i for i,fragment in enumerate(Chem.GetMolFrags(mol))
                   for m in [mol.GetAtomWithIdx(j).GetAtomMapNum() for j in fragment]}
        contacts=[]
        for i,a in enumerate(s['maps']):
            for j in range(i+1,len(s['maps'])):
                c=s['maps'][j]
                if component[a]==component[c]:continue
                q=float(np.linalg.norm(xyz[i]-xyz[j]));radii=sum(periodic.GetRcovalent(periodic.GetAtomicNumber(s['elements'][k])) for k in (i,j))
                if q<.5*radii:contacts.append({'maps':[a,c],'elements':[s['elements'][i],s['elements'][j]],'distance_angstrom':q})
        sides.append({'endpoint':side,'bond_geometry_ok':not failures,'abnormal_bonds':failures,
                      'cross_component_close_contacts':contacts,'complete_endpoint_geometry_ok':not failures and not contacts})
    records.append({'reaction_id':s['reaction_id'],'endpoints':sides,'both_endpoints_bond_geometry_ok':all(x['bond_geometry_ok'] for x in sides),
                    'both_complete_endpoints_geometry_ok':all(x['complete_endpoint_geometry_ok'] for x in sides)})
out=ROOT/'outputs/PES2TS_Demo24_synchronized_reference_20261002/endpoint_bond_audit.json'
save_receipt(out,{'criterion':'provisional sanity screen: 0.5*sum(covalent radii) <= bonded distance <= sum(radii)+0.45 A',
                  'geometry_modified':False,'records':records})
print('Both endpoints pass:',sum(r['both_endpoints_bond_geometry_ok'] for r in records),'/24')
print('Whole endpoint contacts pass:',sum(r['both_complete_endpoints_geometry_ok'] for r in records),'/24')
for r in records:
    if not r['both_complete_endpoints_geometry_ok']:print(json.dumps(r),flush=True)
