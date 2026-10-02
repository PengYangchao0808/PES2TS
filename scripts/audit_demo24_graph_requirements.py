"""Read-only reconstruction of Demo24 edit and coupling graphs for reporting."""
import json
import sys
from collections import Counter
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from pes2ts_core.g1.endpoint_context import build_endpoint_context
from pes2ts_core.g1.event_coupling import build_event_coupling_graph, aromatic_regions_from_bundle
from pes2ts_core.g1.reaction_edit_graph import build_reaction_edit_graph
from pes2ts_core.generation.planning.graph_rebuild import rebuild_endpoint_graphs, load_endpoint_materials_from_export
from pes2ts_core.integration.acp.scheduler_backend import save_receipt

OUT=ROOT/'outputs/demo24_acp_geometry_audit_20261001'
records=[]
for source in sorted((ROOT/'tests/fixtures/p0_demo24/records').glob('RXN_*.json')):
    s=json.loads(source.read_text(encoding='utf-8'))
    rid=s['reaction_id']
    b=rebuild_endpoint_graphs(s['reaction_smiles'],load_endpoint_materials_from_export(s))
    delta=build_reaction_edit_graph(b,aromatic_regions=aromatic_regions_from_bundle(b))
    coords={side:{m:tuple(s[key][i]) for i,m in enumerate(s['maps'])}
            for side,key in [('R','r_coordinates'),('P','p_coordinates')]}
    context=build_endpoint_context(b,delta,r_coordinates=coords['R'],p_coordinates=coords['P'])
    h=build_event_coupling_graph(b,delta,context)
    p=json.loads((ROOT/f'outputs/PES2TS_Demo24_primary_20261001/{rid}/PrimaryPlan.json').read_text(encoding='utf-8'))['primary']
    doc={'reaction_id':rid,'delta_graph':delta.to_doc(),'coupling_graph':h.to_doc(),
         'executed_driver_maps':[p['maps']],'executed_mode':p['mode'],
         'executed_direction':p['direction'],'n_actual_drivers':1}
    save_receipt(OUT/f'{rid}_graph.json',doc)
    records.append({'reaction_id':rid,'n_edits':len(delta.edits),
                    'edit_kinds':dict(Counter(e.edit_kind for e in delta.edits)),
                    'n_events':len(h.events),'event_types':dict(Counter(e.event_type for e in h.events)),
                    'n_strong_components':len(h.strong_components),
                    'actual_drivers':[p['maps']]})
save_receipt(OUT/'graph_requirement_summary.json',{'n_cases':len(records),'records':records})
print(json.dumps({'n_cases':len(records),'total_events':sum(x['n_events'] for x in records),
                  'event_types':dict(sum((Counter(x['event_types']) for x in records),Counter()))}))
