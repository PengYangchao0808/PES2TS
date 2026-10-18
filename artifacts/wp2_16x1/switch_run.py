"""One-time, audited WP2 resource switch; never interrupt an active QC child."""
import json
import os
from pathlib import Path
import signal
import subprocess
import time
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/pesn/dual_ts_v2_20261015/wp2_observations'
AUDIT = Path(__file__).resolve().parent
SCRIPT = 'scripts/compute_dual_ts_observations.py'


def processes():
    found = {}
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():
            continue
        try:
            fields = (proc / 'stat').read_text().rsplit(')', 1)[1].split()
            found[int(proc.name)] = {
                'ppid': int(fields[1]), 'state': fields[0],
                'cmd': (proc / 'cmdline').read_bytes().replace(b'\0', b' ').decode(errors='replace'),
                'cwd': str((proc / 'cwd').resolve()),
                'wchan': (proc / 'wchan').read_text().strip(),
            }
        except (OSError, ProcessLookupError):
            pass
    return found


def descendants(table, pid):
    result = set()
    pending = [pid]
    while pending:
        parent = pending.pop()
        children = {p for p, info in table.items() if info['ppid'] == parent}
        result.update(children)
        pending.extend(children)
    return result


def save(record):
    (AUDIT / 'transition.json').write_text(json.dumps(record, indent=2), encoding='utf-8')


def main():
    os.chdir(ROOT)
    table = processes()
    matches = {pid for pid, info in table.items()
               if SCRIPT in info['cmd'] and info['cmd'].split()[0].split('/')[-1].startswith('python')
               and info['cwd'] == str(ROOT)}
    parents = [pid for pid in matches if table[pid]['ppid'] not in matches]
    if len(parents) != 1:
        raise RuntimeError(f'Expected one live WP2 parent, found {parents}')
    parent = parents[0]
    workers = matches - {parent}
    record = {'requested_by': 'user', 'stage': 'dual_ts_v2/WP2',
              'started_utc': datetime.now(timezone.utc).isoformat(),
              'old_processes': {str(p): table[p] for p in matches},
              'new_workers': 16, 'new_nproc': 1,
              'status': 'draining', 'existing_receipts_preserved': True}
    save(record)
    # Pausing the producer also pauses its executor feeder thread. Workers
    # finish the small bounded IPC queue, including their current QC jobs.
    os.kill(parent, signal.SIGSTOP)
    drained = False
    try:
        deadline = time.monotonic() + 45
        stable = 0
        while time.monotonic() < deadline:
            table = processes()
            active = descendants(table, parent) - workers
            # A completed worker can block sending its finished result while
            # the producer is paused; its receipt is already persisted.
            idle = all(p in table and table[p]['wchan'] in ('anon_pipe_read', 'pipe_read', 'futex_wait_queue_me', 'anon_pipe_write', 'pipe_write') for p in workers)
            stable = stable + 1 if not active and idle else 0
            if stable >= 3:
                drained = True
                break
            time.sleep(1)
        if not drained:
            raise RuntimeError('Workers did not drain; restoring the original run')
    finally:
        if not drained:
            os.kill(parent, signal.SIGCONT)
            record['status'] = 'aborted_original_resumed'
            save(record)
    # No worker has an active child or pending work at this point.
    record['drained_workers'] = {str(p): table[p] for p in workers}
    for worker in workers:
        os.kill(worker, signal.SIGTERM)
    os.kill(parent, signal.SIGTERM)
    os.kill(parent, signal.SIGCONT)
    for _ in range(50):
        table = processes()
        if not any(p in table and table[p]['state'] != 'Z' for p in matches):
            break
        time.sleep(.1)
    else:
        raise RuntimeError('Old process still present; new run not launched')
    rows = [json.loads(line) for line in (OUT / 'progress.jsonl').read_text().splitlines() if line.strip()]
    old_run = rows[-1]['run_id']
    events = [r for r in rows if r['run_id'] == old_run]
    completed = [r for r in events if r['status'] == 'computed']
    receipts = list(OUT.glob('*/*/result.json'))
    record.update(old_run_id=old_run, old_logged_computed=len(completed),
                  old_logged_seconds_sum=sum(r.get('seconds', 0) for r in completed),
                  receipt_files_at_switch=len(receipts),
                  stopped_utc=datetime.now(timezone.utc).isoformat(),
                  old_run_note='Externally stopped after draining. Main driver ledger may lack this run; raw receipts and progress retained. Drain results may not be logged by the paused parent.')
    env = os.environ.copy()
    env.update({key: '1' for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'BLIS_NUM_THREADS')})
    cmd = ['/root/miniforge3/bin/python3', '-u', SCRIPT, '--workers', '16',
           '--acp-config', 'config/pesn/dual_ts_observations_16x1_v1.yaml']
    log = AUDIT / 'compute_16x1.log'
    with log.open('ab', buffering=0) as handle:
        proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
    record.update(status='launched', new_pid=proc.pid, command=cmd,
                  thread_environment={k: env[k] for k in env if k.endswith('NUM_THREADS')},
                  log=str(log), launched_utc=datetime.now(timezone.utc).isoformat())
    save(record)
    print(json.dumps(record, indent=2), flush=True)


if __name__ == '__main__':
    main()
