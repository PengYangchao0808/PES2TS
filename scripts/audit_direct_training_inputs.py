"""Audit direct-training candidate metadata; no truth geometry, QC, or training.

This planning utility reads existing membership and class JSON documents.
Its output is a draft inventory, never an authorization or qualified split.
Run with --check to verify the saved inventory without modifying it.
"""
import argparse
import hashlib
import json
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'outputs/pesn/scale20k_rl2000_v1_20261008'
def read(path):
    return json.loads(path.read_text(encoding='utf-8'))

irc = read(BASE / 'cohorts/irc_fit_cohorts.json')
rl = read(BASE / 'cohorts/rl_cohorts.json')
sealed = read(BASE / 'cohorts/sealed_cohorts.json')
replacements = read(BASE / 'cohorts/origin_gap_replacements.json')
exclusions = read(BASE / 'cohorts/cross_task_exclusions.json')
teacher = read(BASE / 'irc_5k/teacher_dataset/dataset_manifest.json')
dup = read(ROOT / 'data/interim/duplicate_groups.json')
canonical = {}
members = {}
for group in dup['groups']:
    gid = group['canonical_reaction_id']
    members[gid] = group['member_reaction_ids']
    for rid in group['member_reaction_ids']:
        if rid in canonical and canonical[rid] != gid:
            raise ValueError(f'Conflicting canonical groups for {rid}')
        canonical[rid] = gid
def group_set(ids):
    return {canonical.get(rid, rid) for rid in ids}

sets = {
    'legacy_fit5000': set(irc['irc_fit_5000']),
    'legacy_fit2000': set(irc['irc_fit_2000_verified']),
    'monitor250': set(irc['irc_dev_250']),
    'dev_original200': set(rl['rl_dev_200']),
    'dev_effective50': set(replacements['effective_dev50']),
    'sealed_iid100': set(sealed['sealed_iid_100']),
    'sealed_family100': set(sealed['sealed_family_100']),
    'demo24': set(exclusions['demo24_dev_exposure']),
}
sets['dev_exposure_union'] = sets['dev_original200'] | sets['dev_effective50'] | {x['reaction_id'] for x in replacements['replacements']['dev']}
exclude_groups = group_set(set().union(*(sets[k] for k in ['monitor250', 'dev_exposure_union', 'sealed_iid100', 'sealed_family100', 'demo24'])))
candidate = sets['legacy_fit5000'] - {r for r in sets['legacy_fit5000'] if canonical.get(r, r) in exclude_groups}
families = {}
missing_classes = []
class_hashes = {}
all_ids = set().union(*sets.values())
all_ids |= set().union(*(set(members.get(g, [g])) for g in group_set(all_ids)))
for rid in sorted(all_ids):
    p = ROOT / 'data/interim/g1_v2/classes' / f'{int(rid.split("_")[-1]) // 1000:05d}' / f'{rid}.json'
    if not p.is_file():
        missing_classes.append(rid)
        continue
    raw = p.read_bytes()
    d = json.loads(raw)
    family = d.get('levels', {}).get('l0u_undirected_family', {}).get('cluster_id')
    if not family:
        missing_classes.append(rid)
        continue
    families[rid] = family
    class_hashes[rid] = hashlib.sha256(raw).hexdigest()
family_sealed = {families[r] for r in sets['sealed_family100'] if r in families}
family_conflicts = sorted(r for r in candidate if families.get(r) in family_sealed)
candidate -= set(family_conflicts)
by_group = defaultdict(list)
for rid in candidate:
    by_group[canonical.get(rid, rid)].append(rid)
rank = lambda gid: (hashlib.sha256(f'42:{gid}'.encode()).hexdigest(), gid)
representatives = [min(by_group[g]) for g in sorted(by_group, key=rank)]
mixed = {g: sorted({families[r] for r in members.get(g, [g]) if r in families}) for g in sorted(group_set(all_ids))}
mixed = {g: fs for g, fs in mixed.items() if len(fs) > 1}
overlaps = {}
for a, b in [('legacy_fit5000','dev_exposure_union'), ('legacy_fit5000','demo24'), ('legacy_fit5000','sealed_iid100'), ('legacy_fit5000','sealed_family100'), ('monitor250','dev_exposure_union'), ('legacy_fit2000','legacy_fit5000')]:
    overlaps[f'{a}__{b}'] = {'reaction_ids': sorted(sets[a] & sets[b]), 'group_ids': sorted(group_set(sets[a]) & group_set(sets[b]))}
report = {
    'schema_version': 'pes2ts_direct_training_input_audit_draft_v1',
    'as_of_local_date': '2026-10-15',
    'status': 'metadata_audited_not_training_qualified',
    'truth_geometry_read': False,
    'new_qc_or_training_executed': False,
    'source_files': {str(p.relative_to(ROOT)).replace('\\','/'): hashlib.sha256(p.read_bytes()).hexdigest() for p in [BASE/'cohorts/irc_fit_cohorts.json',BASE/'cohorts/rl_cohorts.json',BASE/'cohorts/sealed_cohorts.json',BASE/'cohorts/origin_gap_replacements.json',BASE/'cohorts/cross_task_exclusions.json',BASE/'irc_5k/teacher_dataset/dataset_manifest.json',ROOT/'data/interim/duplicate_groups.json',ROOT/'data/manifests/source_manifest.json']},
    'legacy_counts': {k: {'reaction_ids':len(v),'canonical_groups':len(group_set(v)),'l0u_families':len({families[r] for r in v if r in families})} for k,v in sets.items()},
    'legacy_teacher': {'n_reactions':teacher['n_reactions'],'n_samples':teacher['n_samples'],'n_shards':len(teacher['shards']),'source_method':teacher['source_method']},
    'overlaps':overlaps,
    'class_audit': {'missing_ids':missing_classes,'mixed_family_groups':mixed,'l0u_source_digest':hashlib.sha256(json.dumps(class_hashes,sort_keys=True).encode()).hexdigest(),'scope':'selected metadata plus duplicate-group members; no claim about whole-library family closure or unseen historical exposure'},
    'candidate_pool': {'n_reaction_ids':len(candidate),'n_independent_groups':len(representatives),'excluded_reaction_ids':sorted(sets['legacy_fit5000']-candidate),'family_overlap_exclusions':family_conflicts,'representative_rule':'lexicographically smallest eligible reaction_id per existing canonical group; sha256(42:group_id) ordering','reaction_ids_ranked':representatives},
    'planned_prefixes': {'fit50':[0,50], 'fit500':[0,500], 'fit2000':[0,2000], 'fit_expansion_available':[0,len(representatives)]},
    'prefix_semantics':'zero-based half-open slices of candidate_pool.reaction_ids_ranked; candidates only, re-freeze after group/family and eligibility gates; do not silently replace post-QC failures',
    'training_qualification_pending':['new direct purpose and schemas','full historical exposure and family-component closure','source truth access scope','target/electronic-state/atom-map eligibility','ACP E/g preparation','shared reducer replay and terminal/budget reconstruction','runtime start protocol and strict-chain capability'],
}
out=ROOT/'docs/plans/fixtures/pes2ts_direct_training_input_audit_20261015.json'
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--check',action='store_true')
args=parser.parse_args()
if args.check:
    if read(out) != report:
        raise SystemExit('Saved input audit differs from current metadata; inspect changes before replacing it.')
else:
    out.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'legacy_counts':report['legacy_counts'],'overlap_counts':{k:{m:len(vv) for m,vv in v.items()} for k,v in overlaps.items()},'excluded':report['candidate_pool']['excluded_reaction_ids'],'candidate_groups':len(representatives),'missing_classes':len(missing_classes),'mixed_groups':len(mixed),'first_4':representatives[:4]},ensure_ascii=False,indent=2))
