"""Regression tests for scientific planning bugs found by the Demo24 audit."""
import hashlib
import json
from pathlib import Path

import pytest

from pes2ts_core.config_loader import load_config
from pes2ts_core.generation.planning.coordinate_pool import EndpointMaterials
from pes2ts_core.generation.planning.graph_rebuild import load_endpoint_materials_from_export, rebuild_endpoint_graphs
from pes2ts_core.generation.planning.primary import primary_stretch
from pes2ts_core.generation.planning.selector import propose_strategies

FIX=Path(__file__).parent/'fixtures/p0_demo24/records'


def _input(path):
    s=json.loads(path.read_text(encoding='utf-8'))
    b=rebuild_endpoint_graphs(s['reaction_smiles'],load_endpoint_materials_from_export(s))
    m={side:{map_id:s[side][i] for i,map_id in enumerate(s['maps'])} for side in ('r_coordinates','p_coordinates')}
    m['endpoint_electronic']=s['endpoint_electronic']
    return s,b,m


def test_all_24_have_one_bonded_positive_primary_hypothesis():
    results=[]
    for path in sorted(FIX.glob('RXN_*.json')):
        s,b,m=_input(path)
        r=primary_stretch(b,EndpointMaterials(m['r_coordinates'],m['p_coordinates']))
        assert r['status']=='hypothesis',s['reaction_id']
        p=r['primary']
        assert p['end']>p['start']
        assert p['n_points']<=40
        assert not p['execution_eligible']
        graph=b.r_graph if p['start_endpoint']=='R' else b.p_graph
        assert tuple(p['maps']) in {(e.map_a,e.map_b) for e in graph.edges}
        assert any(set(p['maps'])<=set(c.map_ids) for c in graph.components)
        results.append(s['reaction_id'])
    assert len(results)==len(set(results))==24


def test_selected_anchor_cannot_be_a_nonbonded_compression_start():
    _,b,m=_input(FIX/'RXN_0000077619.json')
    with pytest.raises(ValueError,match='not a valid changed bond'):
        primary_stretch(b,EndpointMaterials(m['r_coordinates'],m['p_coordinates']),
                        anchor={'maps':[2,5],'start_endpoint':'R'})


def test_success_notes_are_not_failures_and_monitor_targets_preserve_measurement():
    s,b,m=_input(FIX/'RXN_0000007104.json')
    p=propose_strategies(b,m,load_config(),reaction_id=s['reaction_id'])
    assert p['extensions']['primary_selection']['primary'] is not None
    for c in p['candidates']:
        assert 'SCHEDULE_INTEGRITY' not in {r['code'] for r in c['failure_reasons']}
        assert c['extensions']['schedule_integrity_notes']
        for monitor in c['monitors']:
            assert monitor['target_test']=='target:'+monitor['measurement']


def test_cli_populates_explicit_hash_bound_inputs(tmp_path):
    from pes2ts_core.generation.planning.cli import scan_plan_proposals, verify_scan_proposals
    source=FIX/'RXN_0000077619.json'
    snapshot=tmp_path/'snapshot.json'
    snapshot.write_bytes(source.read_bytes())
    manifest=tmp_path/'input.json'
    manifest.write_text(json.dumps({'schema_version':'g1_scan_input_manifest_v1','records':[
        {'reaction_id':'RXN_0000077619','snapshot':'snapshot.json','sha256':hashlib.sha256(snapshot.read_bytes()).hexdigest()}]}))
    cfg=load_config()
    cfg['paths']['interim']=str(tmp_path/'interim')
    cfg['paths']['manifests']=str(tmp_path/'manifests')
    cfg['scan_strategy']['input_manifest']=str(manifest)
    result=scan_plan_proposals(cfg)
    assert result.n_written==1
    assert not verify_scan_proposals(cfg).problems
    snapshot.write_text('{}')
    with pytest.raises(ValueError,match='digest mismatch'):
        scan_plan_proposals(cfg)
