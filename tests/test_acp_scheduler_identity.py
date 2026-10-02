"""Workbench renames task displays; frozen identities must remain idempotent."""
import pytest

from pes2ts_core.integration.acp.scheduler_backend import ACPWorkbench


def payload():
    return {'name': 'frozen-name', 'project_id': 'final', 'input': {
        'source': {'source_type': 'xyz_text', 'xyz_text': 'example'},
        'selection': {'pes2ts': {'batch_id': 'batch', 'reaction_id': 'rxn',
                                 'plan_sha256': 'digest'}}}}


def job(identity='job1', project='final'):
    spec=payload()
    spec.update(name='ACP-generated-display-name', project_id=project)
    return {'id': identity, 'spec': spec}


def test_recovery_survives_workbench_display_rename(monkeypatch):
    api=ACPWorkbench()
    monkeypatch.setattr(api,'request',lambda path: {'jobs': [job(),job('pilot','audit')]})
    assert api.submit_once(payload()) == {'job_id': 'job1', 'recovered': True}


def test_duplicate_scientific_identity_is_rejected(monkeypatch):
    api=ACPWorkbench()
    monkeypatch.setattr(api,'request',lambda path: {'jobs': [job(),job('job2')]})
    with pytest.raises(RuntimeError,match='duplicate'):
        api.submit_once(payload())


def test_changed_geometry_is_not_reused(monkeypatch):
    api=ACPWorkbench()
    existing=job()
    existing['spec']['input']['source']['xyz_text']='different'
    monkeypatch.setattr(api,'request',lambda path: {'jobs': [existing]})
    with pytest.raises(RuntimeError,match='differs'):
        api.submit_once(payload())
