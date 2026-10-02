"""Submit PES2TS's frozen primary batch to the existing ACP Workbench API."""
from __future__ import annotations

import json
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen


class ACPWorkbench:
    def __init__(self, base_url: str = 'http://127.0.0.1:8765'):
        self.base_url=base_url.rstrip('/')

    def request(self, path: str, payload: dict | None = None) -> dict:
        body=None if payload is None else json.dumps(payload,ensure_ascii=False).encode('utf-8')
        req=Request(self.base_url+path,data=body,headers={'Content-Type':'application/json'},method='GET' if body is None else 'POST')
        try:
            with urlopen(req,timeout=60) as response:
                return json.load(response)
        except HTTPError as exc:
            raise RuntimeError(f'ACP {path}: HTTP {exc.code}: {exc.read().decode("utf-8",errors="replace")}') from exc

    def ensure_project(self, name: str, description: str) -> str:
        matches=[p for p in self.request('/api/v1/projects')['projects'] if p['name']==name]
        if len(matches)>1:
            raise RuntimeError('ambiguous ACP project name')
        if matches:
            return matches[0]['project_id']
        result=self.request('/api/v1/projects',{'name':name,'description':description,'tags':['PES2TS','Demo24']})
        return result['project_id']

    def submit_once(self, payload: dict) -> dict:
        """Recover by the frozen batch/task identity; never blind-retry a POST."""
        jobs=self.request('/api/v1/jobs?limit=1000')['jobs']
        expected=payload['input']['selection']['pes2ts']
        # Workbench renames display names from molecule/task/remark. Use the
        # persisted scientific identity and project, never that mutable name.
        matches=[j for j in jobs if
                 j['spec']['input'].get('selection',{}).get('pes2ts',{}).get('batch_id')==expected['batch_id']
                 and j['spec']['input'].get('selection',{}).get('pes2ts',{}).get('reaction_id')==expected['reaction_id']
                 and j['spec'].get('project_id')==payload.get('project_id')]
        if len(matches)>1:
            raise RuntimeError('duplicate ACP primary tasks already exist')
        if matches:
            job=matches[0]
            # Identity must survive ACP's scan_request materialization.
            existing=job['spec']['input'].get('selection',{}).get('pes2ts',{})
            if existing!=expected or job['spec']['input'].get('source')!=payload['input']['source']:
                raise RuntimeError('existing ACP task identity/input differs from frozen primary plan')
            return {'job_id':job['id'],'recovered':True}
        return self.request('/api/v1/jobs',payload)


def save_receipt(path: Path, data: dict) -> None:
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
    temporary.replace(path)
