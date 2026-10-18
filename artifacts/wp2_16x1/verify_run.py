"""Read-only receipt checks plus a timestamped performance snapshot."""
import json
from pathlib import Path
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
from compute_dual_ts_observations import validate_receipt

out = ROOT / 'outputs/pesn/dual_ts_v2_20261015/wp2_observations'
rows = [json.loads(line) for line in (out / 'progress.jsonl').read_text().splitlines() if line.strip()]
run_id = rows[-1]['run_id']
events = [r for r in rows if r['run_id'] == run_id]
success = [r for r in events if r['status'] == 'computed']
start = max(datetime.fromisoformat(r['ts']) for r in events if r['status'] == 'started')
last = datetime.fromisoformat(success[-1]['ts'])
checks = []
for event in success[:16]:
    node = event['node_digest']
    folder = out / node[:2] / node
    request = json.loads((folder / 'request.json').read_text())
    result = json.loads((folder / 'result.json').read_text())
    n_atoms = len(result['gradient_hartree_per_angstrom'])
    valid, reason = validate_receipt(request, result, n_atoms)
    nproc = request['pes2ts_settings']['nproc']
    assert valid and nproc == 1, (node, valid, reason, nproc)
    checks.append({'node_digest': node, 'valid': valid, 'nproc': nproc})
snapshot = {
    'timestamp_utc': datetime.now(timezone.utc).isoformat(), 'run_id': run_id,
    'computed': len(success), 'failures': sum(r['status'] == 'failed' for r in events),
    'resumed': sum(r['status'] == 'skipped_resume' for r in events),
    'pending_at_launch': sum(r['status'] == 'started' for r in events),
    'seconds_since_dispatch': (last - start).total_seconds(),
    'nodes_per_minute': 60 * len(success) / (last - start).total_seconds(),
    'reference_2x2_nodes_per_minute': 38.72,
    'comparison_limit': 'Short live windows on different nodes, not a controlled benchmark.',
    'receipt_checks': checks,
}
Path(__file__).with_name('performance.json').write_text(json.dumps(snapshot, indent=2))
print(json.dumps({k: v for k, v in snapshot.items() if k != 'receipt_checks'}, indent=2))
print(f'{len(checks)} real receipts validated with nproc=1')
