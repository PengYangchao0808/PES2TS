"""Audit existing qualified feature shards/records; no QC, training or truth reads."""
from pathlib import Path
from datetime import datetime, timezone
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'outputs/pesn/dual_ts_v2_20261015'
stats = dict(groups=0, nonterminal_states=0, affected_states=0,
             atom_rows=0, mismatched_atom_rows=0, starts=0,
             starts_with_different_mask=0, input_channel_equals_target_mask=True)
group_rows = []
for path in sorted((DATA / 'wp3_features').glob('features_*.npz')):
    with np.load(path, allow_pickle=False) as shard:
        scalars = shard['node_scalars']
        masks = shard['near_zero_action_mask'].astype(bool)
        terminal = shard['is_teacher_endpoint'].astype(bool)
        # GUARD: channel 6 is the old 7-channel v2 near_zero_action_mask
        # slice.  This script is pinned to historical `wp3_features` (7-ch)
        # shards ONLY — it must NEVER be pointed at v41 5-channel shards,
        # where index 6 would silently read `multiplicity`.
        stats['input_channel_equals_target_mask'] &= bool(np.array_equal(scalars[:, :, 6].astype(bool), masks))
        groups = {}
        for idx, branch in enumerate(shard['branch_id']):
            groups.setdefault(str(branch), []).append(idx)
        local = dict(group=path.stem.removeprefix('features_'), states=0, affected_states=0, mismatched_atoms=0)
        for indices in groups.values():
            indices.sort(key=lambda i: int(shard['step_index'][i]))
            previous = np.ones(masks.shape[1], dtype=bool)
            for step, idx in enumerate(indices):
                if not terminal[idx]:
                    delta = masks[idx] != previous
                    stats['nonterminal_states'] += 1
                    stats['affected_states'] += int(delta.any())
                    stats['atom_rows'] += len(delta)
                    stats['mismatched_atom_rows'] += int(delta.sum())
                    local['states'] += 1
                    local['affected_states'] += int(delta.any())
                    local['mismatched_atoms'] += int(delta.sum())
                    if step == 0:
                        stats['starts'] += 1
                        stats['starts_with_different_mask'] += int(delta.any())
                previous = masks[idx]
        stats['groups'] += 1
        group_rows.append(local)

closure = json.loads((DATA / 'wp4_bc/g5_closure_record.json').read_text())
arms = closure.get('arms', closure.get('runs', {}))
audit = {
    'audited_utc': datetime.now(timezone.utc).isoformat(),
    'scope': 'Read-only fit50 frozen feature shards; no inference effect or causality claim.',
    'mask_semantics': {'offline': 'leaving teacher action', 'freerun': 'arriving executed action; all true at start'},
    'mask_comparison': stats, 'per_group': group_rows,
    'note': 'On the teacher path, the arriving-action mask at t equals the leaving-action mask at t-1. Differences here are input semantic differences, not a measured effect on policy output.',
    'closure_top_level_keys': list(closure),
}
out = Path(__file__).with_name('evidence_audit.json')
out.write_text(json.dumps(audit, indent=2), encoding='utf-8')
print(json.dumps(stats, indent=2))
print('closure keys:', list(closure))
if isinstance(arms, dict):
    for name, value in arms.items():
        print(name, value.get('summary', {}))
