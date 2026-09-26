# PES2TS

PES2TS (Potential Energy Surface to Transition State) builds a traceable,
leakage-audited data foundation on top of the public Reaction-QM dataset. Stage
**G0** ("data entry and split") downloads and MD5-verifies the B3LYP-D3/TZVP
artifacts from Zenodo, turns them into a transition-state-free reactant/product
inventory with a typed rejection ledger, quarantines all transition-state and IRC
ground truth behind a single allow-listed accessor, adopts the authors' own
reaction-level train/valid/test split with an independent DRFP near-duplicate
audit, and emits deterministic trial and stratified cohort manifests. Stage
**G1** ("universal S0") then derives the reaction bond-change record for every
inventory reaction — mapped-graph formed/broken/order-changed bonds and hydrogen
migrations, the reaction center, and a unified map→local→global index table with
explicit symmetric-ambiguity recording — from the transition-state-free
inventory alone. Every manifest is versioned and reproducible from a fixed seed.

## Scope boundaries

G0 delivers **data entry, isolation, and split freezing only**. It deliberately
does **not**:

- generate any reaction path (no GFN2-xTB, no `xtb`/`ORCA`/`Gaussian`/`crest`
  execution);
- run any quantum-chemistry or QC validation work (no OptTS/frequency/IRC
  validation);
- produce the authoritative G1 bond-change table — G0 only publishes a clearly
  labelled `authoritative=false` preview (see *Strata preview*; the authoritative
  table is G1's deliverable);
- produce G2 path manifests, G3 labels, or oracle-gap data;
- train any model;
- interpret IRC energies as physics (the index records shapes only).

G1 delivers **the universal S0 bond-change and index layer, plus (as clearly
segregated truth-assisted annotation layers) the P1 TS/IRC reference mapping
and the P2 reaction taxonomy**. It deliberately does **not**:

- generate any reaction path (no GFN2-xTB, no `xtb`/`ORCA`/`Gaussian`/`crest`
  execution);
- run any quantum-chemistry or QC validation work (no OptTS/frequency/IRC
  validation);
- read anything but the transition-state-free inventory and cohort files on the
  endpoint-only G1 build path (no raw HDF5, no split CSVs);
- touch the quarantined DFT/IRC truth on the endpoint-only path — the truth
  guard covers `pes2ts_core/g1/` with exactly one scoped exception: the P1
  annotation module `pes2ts_core/g1/p1_truth.py`, which reads only through the
  audited accessor behind an explicit `--allow-truth`, logs every read, and
  persists no truth geometry; G2 path-generation inputs still exclude
  TS/IRC-derived fields;
- produce G2 path manifests, G3 labels, or oracle-gap data;
- train any model.

No record is ever dropped silently: every reaction that does not enter the
inventory or the frozen split carries a typed `RejectionCode`, a stage, a detail,
and a source pointer in the unified rejection ledger.

## Environment

Dedicated conda environment, `pip` only (no `uv`, no `pyproject.toml`):

```bash
conda create -n pes2ts python=3.12 -y
conda run -n pes2ts pip install -r requirements.txt
conda run -n pes2ts python bin/pes2ts g0 --help
```

`requirements.txt` pins a minimum version for most dependencies and an exact
version for the DRFP fingerprint package:

```
numpy>=2.2.2
pandas>=2.2
rdkit>=2025.9
h5py>=3.11
pyarrow>=16
pyyaml>=6
pytest>=8
drfp==0.3.7
```

Installation note: `drfp==0.3.7` transitively pulls `xgboost`, which on Linux
additionally installs the ~305 MB `nvidia-nccl-cu13` wheel, plus `openpyxl` and
`pre-commit`. These extras (and regular transitive dependencies such as `scipy`
and `tqdm`) are accepted for reproducibility and are not used directly by the
pipeline code. No `scikit-learn` is pulled.

Versions resolved in the reference environment (`pes2ts`, Python 3.12): numpy
2.5.3, pandas 3.0.6, rdkit 2026.3.6, h5py 3.16.0, pyarrow 25.0.1, pyyaml 6.0.3,
pytest 9.1.1. Because pandas 3.x is installed, readers do not rely on pandas 2.x
semantics.

## Data acquisition

G0 uses Zenodo record **18551029** (DOI `10.5281/zenodo.18551029`, revision 1).
Exactly six files (~12.1 GB / 11.3 GiB) are required:

| File | Size (bytes) | Zenodo MD5 |
| --- | ---: | --- |
| `B3LYPD3_TZVP.h5` | 2,054,981,016 | `2c572a68849e805bc0dbb257a53a1fd2` |
| `B3LYPD3_TZVP_IRC.h5` | 9,947,957,486 | `782a4e5e8099de8f2b8e0e90128028cb` |
| `B3LYPD3_TZVP_reaction_info.csv` | 71,468,002 | `6bbf509808a5193770a2093874fb7ad8` |
| `B3LYP-RXN_train.csv` | 57,152,608 | `222059ce2ec2585a385518af393ecaae` |
| `B3LYP-RXN_valid.csv` | 7,158,908 | `0174d304143089c26a93dc89632f5317` |
| `B3LYP-RXN_test.csv` | 7,156,620 | `359332ff1ccb289751ef43bb544f2373` |

The ~21 GB `GFN2_xTB.h5` is deliberately skipped (G2 recomputes GFN2-xTB
locally). Run:

```bash
conda run -n pes2ts python bin/pes2ts g0 fetch
conda run -n pes2ts python bin/pes2ts g0 fetch --force   # force re-download
```

Downloads are streamed (1 MiB chunks) to `<name>.part-<pid>` and only
`os.replace`d into place after the MD5 matches, so a partial or corrupt transfer
never leaves a usable file. A file whose local MD5 already matches is skipped; a
re-run therefore downloads zero bytes. `g0 fetch` writes `source_manifest.json`
only after every configured file is verified, and aborts with exit code 3 on any
MD5 mismatch (leaving no manifest and no `.part` residue).

## CLI reference

All commands share the global options `--config PATH` (a user YAML deep-merged
over `config/defaults.yaml`), `--log-level LEVEL` (default `INFO`), and
`--log-file PATH`. `--help` works for every command and exits 0.

| Command | Extra arguments | What it does |
| --- | --- | --- |
| `g0 fetch` | `--force` | Download and MD5-verify the six sources; write `source_manifest.json`. |
| `g0 inventory` | — | Join the reaction-info CSV with the HDF5 R/P species; write `inventory.parquet` + `inventory_manifest.json`. |
| `g0 quarantine` | — | Extract TS geometries and the IRC index, relocate the two answer-bearing HDF5 files, write `truth_manifest.json`. |
| `g0 dedup` | — | Exact-duplicate and written-reverse detection; write `duplicate_groups.json` + `duplicate_ledger.json`. |
| `g0 audit` | — | Full DRFP + blocked-Tanimoto cross-split leak audit; write `fingerprints.parquet` + `leak_audit.json`. |
| `g0 split` | — | Adopt the official split; write `split_assignment.parquet` + a `leak_status="pending"` `split_manifest.json`. |
| `g0 freeze` | `--remediation {none,exclude-leaky,rebuild}` | Apply the leak decision policy and finalise `split_manifest.json`. |
| `g0 cohorts` | — | Stratification preview and deterministic cohorts; write `strata_report.json`, `cohort_trial.json`, `cohort_stratified.json`. |
| `g0 run-all` | `--data-root PATH`, `--force`, `--skip-fetch`, `--remediation {none,exclude-leaky,rebuild}` | Run the whole idempotent pipeline; write `g0_run_report.json`. |
| `g0 truth-index` | `--reaction-id RXN_ID` | Audited read of the quarantined IRC index (or one reaction's IRC frames). |

`g0 run-all --data-root PATH` re-roots every configured path under `PATH`
(a fresh fixture tree, for example). `g0 run-all --force` re-runs every stage
instead of skipping; `--skip-fetch` verifies existing sources instead of
downloading. Because `g0 quarantine` physically relocates the two HDF5 archives
out of the raw tree, `run-all --force` on an already-quarantined tree cannot
re-run `inventory` (its raw input is gone) and fails loudly at that stage;
forcing a transfer is `g0 fetch --force` on a fresh tree.

### Exit codes

| Code | Constant | Meaning |
| ---: | --- | --- |
| 0 | — | Success. |
| 2 | `EXIT_CONFIG_ERROR` | Configuration could not be loaded (e.g. a user config path does not exist). |
| 3 | `EXIT_CHECKSUM_MISMATCH` | A source MD5 does not match, or a required upstream artifact/manifest is missing. |
| 4 | `EXIT_QUARANTINE_ERROR` | Quarantine failed (HDF5/OS error). |
| 5 | `EXIT_SPLIT_SCHEMA_ERROR` | An official split CSV is malformed (missing id column, bad/duplicate id, an id in two splits). |
| 6 | `EXIT_AUDIT_BUDGET_EXCEEDED` | The leak audit exceeded `near_dup.max_seconds`; the audit is marked incomplete and never claims a clean split. |
| 7 | `EXIT_LEAK_FOUND` | Cross-split leakage found and no remediation was requested (the default HALT). |
| 8 | `EXIT_AUDIT_INCOMPLETE` | Freeze refused because the audit is not `complete=true`. |
| 9 | `EXIT_REMEDIATION_FAILED` | A requested remediation did not reach a clean audit; the pre-remediation state is restored. |
| 20 | `EXIT_AUTHORITATIVE_STRATA_REQUIRED` | `cohorts.require_authoritative_strata=true` but only the non-authoritative preview exists. |
| 21 | `EXIT_PIPELINE_FAILED` | `g0 run-all` did not complete every stage (the run report names the failing stage). |
| 22 | `EXIT_G1_BUILD_FAILED` | A `g1` stage failed (e.g. a document that cannot be re-read by `g1 verify`, or an authoritative strata rebuild requested from a partial build). |
| 23 | `EXIT_TRUTH_FLAG_REQUIRED` | A truth-assisted `g1` subcommand (`join-audit`, `resolve-map`) was invoked without the explicit `--allow-truth` flag, or the audited accessor refused the read. |

## Configuration keys

Defaults live in `config/defaults.yaml`; override any subset with `--config`.

| Key | Default | Purpose |
| --- | --- | --- |
| `source.zenodo_record` | `18551029` | Zenodo record id. |
| `source.zenodo_doi` | `10.5281/zenodo.18551029` | Record DOI (recorded in manifests). |
| `source.zenodo_revision` | `"1"` | Zenodo revision. |
| `source.zenodo_modified` | `"2025-09-01"` | Record modification date. |
| `source.files.<key>.filename` / `.url` / `.md5` | six entries | Per-file download URL and expected MD5. |
| `paths.data_root` | `data` | Root for every derived path. |
| `paths.raw` | `data/raw/reaction_qm` | Downloaded raw sources. |
| `paths.interim` | `data/interim` | Parquet/JSON intermediate artifacts. |
| `paths.ground_truth` | `data/ground_truth` | Quarantined TS/IRC artifacts. |
| `paths.truth_sources` | `data/ground_truth/sources` | Relocated truth-bearing HDF5 files. |
| `paths.manifests` | `data/manifests` | All JSON manifests and ledgers. |
| `split.seed` | `42` | Seed for cohort selection and rebuild determinism. |
| `near_dup.threshold` | `0.90` | Tanimoto threshold for a cross-split near-duplicate. |
| `near_dup.cluster_threshold` | `0.80` | Butina cluster boundary (cutoff = `1 - this`). |
| `near_dup.fp_size` | `2048` | DRFP folded length. |
| `near_dup.block_size` | `1000` | Reference block size for the blocked Tanimoto scan. |
| `near_dup.max_seconds` | `7200` | Audit wall-clock budget; exceeding it aborts instead of sampling. |
| `near_dup.max_pairs` | `10000` | Cap on stored top offenses (the exact count is always reported). |
| `cohorts.trial` | `200` | Trial cohort size. |
| `cohorts.stratified` | `1000` | Stratified cohort size. |
| `cohorts.require_authoritative_strata` | `false` | When true, `g0 cohorts` halts with exit 20 until G1 supplies authoritative strata. |

## Artifact contracts

All G0 JSON manifests carry `schema_version = "g0_manifest_v1"` (except the
JSONL logs, which are line records), and every artifact carries a
`dataset_version` of the form `zenodo-18551029-rev1`. The rewritten
`strata_report.json` retains `g0_manifest_v1`; only the G1 reaction-change
documents and G1 side manifests use their own `g1_change_v1` /
`g1_manifest_v1` schema versions (see
*G1 — universal S0: bond changes and unified index*). Determinism guarantee: two
runs over the same inputs produce **byte-identical** artifacts after removing the
volatile keys `VOLATILE_KEYS = {generated_at, downloaded_at, duration_seconds}`.
Parquet artifacts are written with a stable column order; ledgers are
append-only.

| Artifact | Path | Produced by | Key fields / columns |
| --- | --- | --- | --- |
| `source_manifest.json` | `data/manifests/` | `g0 fetch` | `zenodo_doi`, `zenodo_revision`, `zenodo_modified`, `files[{filename,url,zenodo_md5,sha256,size_bytes,downloaded_at}]` |
| `inventory.parquet` | `data/interim/` | `g0 inventory` | `reaction_id`, `dataset_version`, `reaction_smiles`, `n_reactant_components`, `n_product_components`, `elements`, `total_atoms_reactants`, `total_atoms_products`, `charge_total_reactants`, `charge_total_products`, `multiplicity_max`, `components` |
| `inventory_manifest.json` | `data/manifests/` | `g0 inventory` | `n_rows`, `n_reactions_in_csv`, `n_reactions_in_h5`, `rejection_histogram` |
| `truth_manifest.json` | `data/manifests/` | `g0 quarantine` | `ts_parquet{path,sha256,n_rows}`, `irc_index{path,sha256,n_rows}`, `sources[{filename,original_path,relocated_path,sha256,size_bytes}]`, `n_reactions_without_ts` |
| `truth_access_log.jsonl` | `data/manifests/` | allow-listed truth reads | `{ts, accessor_module, function, reaction_id}` per line |
| `duplicate_groups.json` | `data/interim/` | `g0 dedup` | `groups[{identity, member_reaction_ids, canonical_reaction_id}]` |
| `duplicate_ledger.json` | `data/manifests/` | `g0 dedup` | `n_inventory_rows`, `n_unique_identities`, `n_duplicate_groups`, `n_exact_duplicates`, `n_reverse_pairs`, `groups`, `pairs[{kind,canonical_reaction_id,other_reaction_id}]` |
| `fingerprints.parquet` | `data/interim/` | `g0 audit` | `reaction_id`, `fp_bits` (exactly `fp_size` 0/1 characters) |
| `leak_audit.json` | `data/manifests/` | `g0 audit` / freeze re-audit | `method="DRFP"`, `params{fp_size,radius,rings,threshold}`, `complete`, `n_probes`, `n_references`, `n_comparisons`, `n_pairs_over_threshold`, `max_similarity`, `top_offenses`, `n_cross_split_known_duplicates` |
| `split_assignment.parquet` | `data/interim/` | `g0 split` (rewritten by `freeze` remediation) | `reaction_id`, `split` ∈ {`train`,`valid`,`test`}, sorted by id |
| `split_manifest.json` | `data/manifests/` | `g0 split` (pending) then `g0 freeze` | adoption: `strategy`, `seed`, `source_sha256`, `counts`, `covered`, `n_inventory`, `n_not_in_split`, `n_split_only`, `split_only_examples`; freeze adds `frozen`, `leak_status`, `threshold`, `offending_tuples`, `remediation`, `audit` |
| `strata_report.json` | `data/manifests/` | `g0 cohorts` | `strata_source="preview"`, `authoritative=false`, `g1_obligation`, `seed`, `seed_stratified`, `strata[{key,n_inventory,n_cohort_trial,n_cohort_stratified}]`, `totals` |
| `cohort_trial.json` | `data/interim/` | `g0 cohorts` | `cohort="trial"`, `size`, `members`, `strata` |
| `cohort_stratified.json` | `data/interim/` | `g0 cohorts` | `cohort="stratified"`, `size`, `members`, `strata` |
| `rejection_ledger.jsonl` | `data/manifests/` | every stage (append-only) | `{reaction_id, stage, code, detail, source_pointer}` per line |
| `rejection_summary.json` | `data/manifests/` | every stage | `total`, `by_code` |
| `g0_stage_state.json` | `data/manifests/` | `g0 run-all` | `stages{name: {inputs_digest, outputs{path: sha256}}}` |
| `g0_run_report.json` | `data/manifests/` | `g0 run-all` | `stages[]`, `totals`, `leak_status`, `rejection_histogram`, `config_digest`, `data_root` |
| `ts.parquet` | `data/ground_truth/` | `g0 quarantine` | `reaction_id`, `atomic_numbers`, `coordinates`, `EHG`, `charge`, `multiplicity`, `reaction_smiles` |
| `irc_index.parquet` | `data/ground_truth/` | `g0 quarantine` | `reaction_id`, `n_atoms`, `n_frames`, `has_forces`, `ts_index` |
| `g1_p1_join_audit.json` | `data/manifests/` | `g1 join-audit` / `resolve-map` | `n_inventory`, `n_ts`, `n_irc_index`, `n_joined_all`, `by_reason` (typed join reasons with capped examples), extra-id counts, source digests |
| `p1_mapping/<shard>/<reaction_id>.json` | `data/interim/g1_truth/` | `g1 resolve-map` | `schema_version="g1_p1_truth_v1"`, `truth_assisted=true`, `sources` (dataset/inventory/truth-manifest/algorithm digests), `status` (8-valued enum), `mapping.map_to_atoms[{map, element, r/p component+local, ts_irc_index}]` + class statistics, `bond_events` (map-space edits), `irc_validation{ts_frame_index, layout, orientation, orientation_basis, orientation_bond_scores, endpoint_match, event_support[], synchrony, quality_flags}`, `reaction_center`, `graph` (map-space R/P bond lists), `validation`. No coordinate/energy arrays. |
| `g1_p1_summary.parquet` | `data/interim/` | `g1 resolve-map` | One row per resolved reaction: `status`, `g2_eligible`, layout/orientation/endpoint columns, combination/class counts, event and support counts |
| `g1_p1_manifest.json` / `g1_p1_coverage.json` | `data/manifests/` | `g1 resolve-map` | Status histogram, eligible count, config + source digests / per-status coverage with deterministic manual-check samples and layout/orientation histograms |
| `reaction_classes/<shard>/<reaction_id>.json` | `data/interim/g1_truth/` | `g1 classify` | `schema_version="g1_p2_class_v1"`, `status`, `levels{l0..l3 cluster_id+signature}`, `family_labels`, `pathway_labels` (separate from structural clusters), center/context sizes, `flags` |
| `g1_reaction_class_summary.parquet` | `data/interim/` | `g1 classify` | One row per P1 reaction: cluster ids per level, family labels, pathway labels, flags |
| `g1_p2_manifest.json` / `g1_p2_coverage.json` / `g1_cluster_report.json` | `data/manifests/` | `g1 classify` | Status counts + digests / coverage fractions / per-level cluster statistics, size histograms, singleton fractions, family-label distributions, P1-status cross-tabs, cross-split cluster distribution (report only) |
| `g1_gate.json` / `g2_eligible.json` | `data/manifests/` + `data/interim/` | `g1 gate` | Denominator, eligible count and fraction, exclusion breakdown, verification refusals / the pure reaction-id list G2 may consume |

The two truth-bearing archives `B3LYPD3_TZVP.h5` and `B3LYPD3_TZVP_IRC.h5` are
relocated out of the raw tree into `data/ground_truth/sources/` by
`g0 quarantine`; the reaction-info CSV and the three split CSVs remain in
`data/raw/reaction_qm/`.

## Ground-truth isolation

The DFT transition-state geometry lives in the *same* combined HDF5 archive as
the reactants and products, so isolation is structural rather than advisory:

1. **Extraction.** `g0 quarantine` extracts every transition-state species into
   `data/ground_truth/ts.parquet` and builds a shape-only IRC index at
   `data/ground_truth/irc_index.parquet` (`reaction_id`, `n_atoms`, `n_frames`,
   `has_forces`, `ts_index`); the ~9.3 GB of IRC frames are never copied.
2. **Relocation.** The two truth-bearing HDF5 archives are physically moved from
   the raw tree into `data/ground_truth/sources/`, so the path-generation side no
   longer has the answer file in its accessible raw tree.
3. **Single allow-listed accessor.** `pes2ts_core.g0.truth.truth_reader` is the
   only module that reads the quarantined artifacts. Every accessor requires
   `allow_truth=True` (otherwise it raises `PermissionError`) and appends an entry
   to `truth_access_log.jsonl` on each successful read (bulk accessors
   `load_ts_table` and `iter_irc_trajectories` record the calling module).
   `g0 truth-index` is the audited CLI entry point.
4. **Static AST guard.** `pes2ts_core.utils.truth_guard.assert_no_truth_access()`
   parses every `.py` under `pes2ts_core/` and `bin/` and fails if any module
   outside the allowlist contains a string constant with `ground_truth` or
   `truth_sources`, names `B3LYPD3_TZVP.h5` or `B3LYPD3_TZVP_IRC.h5`, imports the
   truth package or `truth_reader`, or uses `subprocess`/`importlib`/`eval`/`exec`
   (`DYNAMIC_EXEC_RISK`, flagged for review). The allowlist is
   `pes2ts_core/g0/truth_quarantine.py`, `pes2ts_core/g0/truth/truth_reader.py`,
   `pes2ts_core/utils/truth_guard.py`, and the single scoped P1 annotation module
   `pes2ts_core/g1/p1_truth.py` (which reads exclusively through the audited
   accessors), plus `cli.py` **only** inside the
   `truth_index` handler. The default test suite asserts the shipped package is
   clean.

### Residual limits

- The guard is **static AST analysis**. It cannot catch a path string built
  dynamically at runtime (for example by string concatenation or decoding), so
  it is a strong code-level control, not a proof. Dynamic-execution constructs
  (`subprocess`, `importlib`, `eval`, `exec`) are flagged for manual review
  precisely because they can bypass static inspection.
- **Filesystem permissions do not block direct reads.** The guard constrains the
  code path, not the operating system; a process that already knows the
  relocated path could still read it. Isolation here is enforced by relocation,
  a single audited accessor, and the AST guard.
- The access log records **deliberate access through the allow-listed accessor
  only**. Out-of-band reads of the relocated files are not observable by the log.

## Strata preview

`g0 cohorts` publishes a **non-authoritative** stratification preview. Its
`strata_report.json` carries `strata_source="preview"`, `authoritative=false`,
and a `g1_obligation` note:

> G1 must recompute bond-change strata authoritatively via
> `pes2ts_core.g0.strata.compute_bond_changes_preview` import and re-derive
> cohorts.

G1 must import `pes2ts_core.g0.strata.compute_bond_changes_preview` rather than
reimplement the bond-change computation. The preview computes formed/broken/
bond-order-change counts from a mapped RDKit graph difference; when a reaction
lacks usable atom maps or cannot be parsed, the three counters are the sentinel
value `-1` (never an exception, never a silent zero). Stratum keys are
`(element_set, n_components, heavy_atom_bucket)` rendered as e.g. `C,H,O|2|small`,
where the bucket is derived from the reactant atom count including hydrogens
(`small` ≤ 8, `medium` 9–16, `large` 17–28, `xl` > 28). Cohort selection is
deterministic: quotas use the largest-remainder method and members are ranked by
`sha256(f"{seed}:{reaction_id}")`; the trial cohort is a nested subset of the
stratified cohort. When `cohorts.require_authoritative_strata=true`, selection
halts with exit code 20 until G1 supplies authoritative strata.

`g1 strata` (after a full `g1 build`) performs this recomputation: it overwrites
the report with `strata_source="g1_authoritative"` and `authoritative=true` (the
`g1_obligation` key disappears) and re-exports both cohort files — see
*Authoritative strata*.

## Split policy

G0 **adopts the authors' official reaction-level split** from
`B3LYP-RXN_train.csv` / `B3LYP-RXN_valid.csv` / `B3LYP-RXN_test.csv`, validating
full inventory coverage and pairwise disjointness. Adoption writes
`leak_status="pending"`; `g0 freeze` then runs the leak decision policy.

The independent DRFP audit (`g0 audit`) fingerprints every reaction (folded
length 2048, radius 3, rings) and computes a blocked Tanimoto similarity of every
valid/test probe against **all** train references — never a random sample — with
the top offenses capped but the exact over-threshold count always reported. A
`clean` verdict is only possible when the audit is `complete=true`.

Default policy is **HALT**: on `LEAK_FOUND`, `g0 freeze` exits 7, writes
`leak_status="LEAK_FOUND"`, and leaves the assignment byte-unchanged.
Remediation happens only through an explicit flag:

- `--remediation exclude-leaky` removes the held-out member of every offending
  pair from the split, records one `LEAK_EXCLUDED` rejection per victim, and
  re-runs the audit; it commits only after a clean re-audit.
- `--remediation rebuild` discards the authors' split and rebuilds an
  **80/10/10 Butina cluster split** over the whole inventory: DRFP fingerprints
  are clustered with a Tanimoto distance cutoff of
  `1 - near_dup.cluster_threshold` (= 0.20 by default) using the fixed
  `split.seed`, whole clusters are assigned greedily to meet the 80/10/10 quotas,
  and no cluster ever straddles two splits. The result is recorded as
  `strategy="rebuilt_butina_v1"` while the authors' `source_sha256` values are
  preserved for provenance.

Both remediations commit atomically: the candidate assignment is snapshotted,
re-audited, and restored (assignment and audit manifest) on any exception or
still-leaking result, so the split is never left in a mixed state. Rebuild builds
an **O(N²) condensed distance matrix** (≈160 GB for the full 199,890-reaction
inventory), so it is intended for fixtures/moderate sizes; on the full dataset the
default excluded/rebuild decision must be made deliberately.

## G1 — universal S0: bond changes and unified index

Stage **G1** ("universal S0") turns every reaction in the TS-free inventory
(199,217 real rows) into a versioned *bond-change record*. For each reaction it
parses both sides with explicit hydrogens retained, diffs the mapped RDKit
graphs, derives the hydrogen migrations and the reaction center, and matches
every reactant/product component against the inventory components to build a
unified **map → local → global index table**.

- **Bond changes.** `formed`/`broken`/`order_changed` are computed on mapped
  graph edges with `GetBondTypeAsDouble()` (aromatics stay un-kekulized at 1.5,
  both sides parsed by the same code path); `hydrogen_migration` reports one
  entry per mapped hydrogen whose bonded partner differs between the sides
  (`from`/`to`, `null` = isolated). The three bond lists follow the G0 preview
  semantics and are deliberately **not** mutually exclusive (one pair can be both
  `broken` and `formed` when its bond order changes). Every detail count is
  cross-checked against the imported G0 preview and recorded as `preview_match`.
- **Reaction center.** `core` is every endpoint of the changed bonds;
  `with_shell` adds `g1.neighborhood_shell` rings over the reactant∪product graph
  union.
- **Unified index.** Component matching is stereo-, bond-order- and
  charge-insensitive (flattened-skeleton mol-vs-mol subgraph match, never a
  SMILES round-trip). Each component block carries
  `rows[{local_index,map,element,global_index}]`, the `element_check`
  cross-validation (map `k` ↔ `atomic_numbers[k-1]`), the geometry annotation
   (`geometry_check_ok` / `geometry_worst`), and the full ambiguity state: up to
   `g1.max_candidates` stored `(map, local_index)` bijections, the first of
   which fills `rows` (alternatives = `n_candidates − 1`), and `truncated`
   marks a match that hit `g1.match_cap`.
- **Typed failures.** Every rejected reaction still gets a document
  (`validation.status="rejected"`, machine-readable `failure_code`, empty
  `reactants`/`products` — no usable index table) plus one ledger entry with
  `stage="g1_build"`. The codes are `G1_MAP_ERROR`, `G1_COMPONENT_MISMATCH`,
  `G1_INDEX_MISMATCH`, `G1_BOND_GEOMETRY`, `G1_PREVIEW_CONFLICT`, and
  `G1_NO_BOND_CHANGE`.
- **Categories.** `categories` flags the seven change classes (`pure_formed`,
  `pure_broken`, `both`, `order_change_only`, `has_order_change`, `h_migration`,
  `multi_component`); `g1_coverage.json` reports per-category coverage plus the
  deterministic manual-check sample used by `g1 sample`.

G1 reads only the transition-state-free inventory and cohort files — it generates
no path, runs no QC, and never reads the quarantined DFT/IRC truth; the static
AST guard covers the `pes2ts_core/g1/` modules like every other module.

### G1 CLI reference

| Command | Extra arguments | What it does |
| --- | --- | --- |
| `g1 build` | `--cohort {trial,stratified,all}`, `--limit N` | Build the per-reaction documents and write `g1_reaction_change_summary.parquet`, `g1_manifest.json`, `g1_coverage.json`. Default: every inventory row; `--cohort` restricts to a cohort file; `--limit` truncates the sorted cohort. |
| `g1 sample` | `--category NAME`, `--n N` | Print per-category bond-change counts (formed/broken/order_changed/h_migration) plus index/pairing status from the summary parquet (documents are not opened); `--n` is the per-category sample size. |
| `g1 strata` | — | Re-derive the authoritative strata and cohorts from a **full** build; write `g1_strata_manifest.json` and rewrite `strata_report.json` + both cohort files. |
| `g1 verify` | `--stage {build,p1,p2}` | Re-read every written document of the chosen tree and reconcile counts and the summary checksum with the manifest (default `build`, the classic G1 documents). |
| `g1 join-audit` | `--allow-truth` | Write `g1_p1_join_audit.json`: the inventory/TS/IRC-index/G1-summary ID join with typed reason buckets and source digests. Refuses with exit 23 without `--allow-truth`. |
| `g1 resolve-map` | `--allow-truth`, `--cohort {trial,stratified,all}`, `--limit N` | Build the truth-assisted P1 mapping documents (`interim/g1_truth/p1_mapping/`), `g1_p1_summary.parquet`, `g1_p1_manifest.json`, `g1_p1_coverage.json`, and the join audit; append typed ledger entries for non-eligible statuses. |
| `g1 classify` | — | Classify the P1 documents into the hierarchical taxonomy; write `interim/g1_truth/reaction_classes/`, `g1_reaction_class_summary.parquet`, `g1_p2_manifest.json`, `g1_p2_coverage.json`, `g1_cluster_report.json`. |
| `g1 gate` | — | Verify P1+P2, then write `g1_gate.json` and the `g2_eligible.json` id list (exit 22 on any verification problem). |

### G1 artifacts

| Artifact | Path | Produced by | Key fields / columns |
| --- | --- | --- | --- |
| `reaction_change/<shard>/<reaction_id>.json` | `data/interim/g1/` | `g1 build` | One document per reaction, `schema_version="g1_change_v1"`: `mapping`, `bond_changes{formed,broken,order_changed,hydrogen_migration,preview_counters,preview_match}`, `reaction_center{core,with_shell}`, `categories`, `reactants[]`/`products[]` index blocks, `ambiguity`, `validation{status,failure_code,failure_detail}`. Shard = numeric id // `shard_size` (5-digit zero-padded). |
| `g1_reaction_change_summary.parquet` | `data/interim/` | `g1 build` | One row per built reaction: id, seven category booleans, bond-change counts, `index_status`/`pairing_status`, atom counts. |
| `g1_manifest.json` | `data/manifests/` | `g1 build` | `n_total`, `n_valid`, `n_rejected`, `by_code`, `n_files`, `summary_sha256`, config summary. |
| `g1_coverage.json` | `data/manifests/` | `g1 build` | `overall`, per-category coverage, `index_status`/`pairing_status` histograms, `by_code`, and the deterministic `manual_sample` (`seed`, `seed_source`, `sample_size`). |
| `g1_strata_manifest.json` | `data/manifests/` | `g1 strata` | `basis{n_inventory,n_built,n_valid,n_rejected,n_unbuilt,n_counter_agreements}`, `seed`, `seed_stratified`, `n_strata`, artifact paths. |
| `strata_report.json` | `data/manifests/` | `g1 strata` | Rewritten with `strata_source="g1_authoritative"`, `authoritative=true`; `totals` and per-stratum counts come from the authoritative build. |
| `cohort_trial.json` / `cohort_stratified.json` | `data/interim/` | `g1 strata` | Re-exported cohorts (`cohort`, `size`, `members`, `strata`); members stay identical to the preview for the same records and seed. |

A component index block contains `tag`, `index_base`, `n_atoms`,
`rows[{local_index,map,element,global_index}]`, `status`
(`unique` / `symmetric_ambiguous` / `truncated`), `n_candidates`, `candidates`
(capped `{local_index,map}` pairs), `element_check`, `geometry_check_ok`,
`geometry_worst`, and `truncated`.

### G1 configuration keys

| Key | Default | Purpose |
| --- | --- | --- |
| `g1.shard_size` | `1000` | Reactions per `reaction_change` shard directory. |
| `g1.match_cap` | `10000` | `GetSubstructMatches` `maxMatches` cap; hitting it sets `truncated`. |
| `g1.max_candidates` | `64` | Up to N stored `(map, local_index)` bijections per component, the first of which fills `rows` (alternatives = `n_candidates − 1`). |
| `g1.bond_tolerance` | `0.45` | Å tolerance over the Cordero covalent-radius sum for the geometry annotation. |
| `g1.neighborhood_shell` | `1` | Ring count added to the reaction-center core for `with_shell` (`0` = core only). |
| `g1.sample_size` | `20` | Per-category deterministic manual-check sample size. |
| `g1.sample_seed` | `42` | Sample seed; falls back to `split.seed` when unset (`seed_source` records which was used). |
| `g1_truth.shard_size` | `1000` | Reactions per P1/P2 shard directory. |
| `g1_truth.max_combinations` | `1024` | Candidate-combination cap per side; beyond it the status is `unresolved_truncated`. |
| `g1_truth.rmsd_class_tolerance` | `0.05` | Width (Å) of the absolute Kabsch-RMSD buckets that fold equivalent candidates. |
| `g1_truth.endpoint_rmsd_pass` / `.weak` | `1.0` / `3.0` | Branch-endpoint match quality thresholds (Å, aligned); the weak bound is calibrated on the real archive's IRC-vs-optimized conformer distances. |
| `g1_truth.ts_frame_tolerance` | `1e-4` | Aligned RMSD below which an IRC frame is identified with the quarantined TS (the trajectory identity proof). |
| `g1_truth.splice_factor` | `5.0` | A splice step must exceed this factor times the median consecutive-frame step. |
| `g1_truth.bond_distance_tolerance` | `0.45` | Covalent-radius-sum tolerance for event bonded-range checks. |
| `g1_truth.permutation_budget` | `64` | Search budget of the TS-row permutation fallback. |
| `g1_truth.per_reaction_lookup_threshold` | `512` | Selection sizes at/below this fetch IRC frames per reaction; larger selections stream the archive once. |
| `g1_class.taxonomy_version` | `"g1p2_taxonomy_v1"` | Version embedded in every P2 cluster id. |
| `g1_class.canonical_budget` | `2000` | Individualization budget of the canonical labeler; exhaustion is flagged, never silent. |

### Authoritative strata

`g1 strata` requires a **full** build (every inventory row must have been built);
a partial build exits 22. It then replaces the `-1` preview counters with the
authoritative detail counts from the summary (rejected/unbuilt rows keep the `-1`
sentinel), cross-checks its own counts against the imported G0 preview
(`basis.n_counter_agreements`) and refuses on any disagreement, writes
`g1_strata_manifest.json` with the full `basis`, and rewrites
`strata_report.json` + both cohort files as described above.

### G1 residuals

- **Constitutional mismatches ≈0.3–0.7% of reactions (probe bound; the exact
  count is in `g1_manifest.json` by_code after a full build).** The reaction SMILES
  and the
  inventory components are occasionally genuinely different molecules (different
  ring size or radical count). These become typed `G1_COMPONENT_MISMATCH`
  rejections — never silent drops. One known sub-class is the radical-count blind
  spot: the flattened skeletons are identical but one side carries unpaired
  electrons the other does not (`RXN_0000100283`-class); the recipe deliberately
  does not normalize radical electrons.
- **Geometry is a soft annotation.** Isolated long next-valence bonds (S–O/P–O
  2.2–2.7 Å) occur in the real components. `geometry_check_ok=false` with the
  worst violation detail is recorded; only a component with more than half its
  bonds outside `g1.bond_tolerance` (systematic corruption) is a hard
  `G1_BOND_GEOMETRY` rejection.
- **Symmetric ambiguity is recorded, not resolved.** The first RDKit isomorphism
  (deterministic enumeration order) supplies `rows`; all further candidates up to
  the cap are persisted, and the per-component `status` plus the per-reaction
  `ambiguity.index` distinguish `unique` / `ambiguous` / `truncated` (likewise
  `ambiguity.pairing` for multiple same-skeleton components on one side).

## G1 truth-assisted layers: P1 TS/IRC mapping and P2 reaction taxonomy

The plan document `G1_TS_IRC_映射与反应归簇实现方案.md` extends G1 with two
truth-assisted annotation layers. They are deliberately **separate** from the
endpoint-only G1 build: every truth byte flows through the audited accessor,
every truth-assisted artifact carries `truth_assisted=true` plus source
digests, and the G2 path-generation inputs still exclude anything derived
from TS/IRC coordinates or energies (the mapping, bond edits, and class
labels are reaction metadata; geometry is not).

### P1 — canonical TS/IRC atom mapping (`g1 resolve-map --allow-truth`)

For every inventory reaction P1 solves the R/P ↔ TS/IRC atom correspondence,
validates the mapped bond events against the IRC geometry, and freezes a
versioned per-reaction document. The solver is anchored on two
archive-invariants that are **re-verified per reaction, never assumed**:

- TS/IRC atom rows follow the reaction map-number order (row `i` ↔ map
  `i+1`); the identity proposal must pass the element-sequence hard check,
  otherwise a budget-capped element-preserving permutation search runs and
  ambiguous outcomes become `unresolved_reactive_center`.
- The IRC trajectory is `[TS, branch_1 → end A, branch_2 → end B]` with the
  two branches spliced where one consecutive-frame step is a
  `splice_factor`-median outlier; the stored `ts_index=0` constant is
  confirmed by locating the frame that matches the quarantined TS geometry
  (rigid-aligned: a minority of trajectories live in a different Cartesian
  frame). A trajectory with no TS frame is `ts_irc_endpoint_mismatch`.

The R/P side assignment reuses the G1 candidate bijections — never a fresh
"first graph match": every stored candidate combination, including
permutations of identical-SMILES component geometries, is scored by the
Kabsch residual of the TS geometry against the map-ordered side coordinates,
folded into absolute RMSD buckets (`round(rmsd/tolerance)`, so input order
can never change the classes), and the best bucket's lexicographic
representative wins. Ties collapse to `resolved_symmetry_collapsed`;
enumeration beyond `max_combinations` marks `unresolved_truncated`. Endpoint
RMSDs are recorded as `pass|weak|fail` quality (IRC termini routinely sit
1.5–3 Å from the optimized species — conformer differences), while the exact
TS-frame equality is the identity proof. The branch-to-side **orientation**
is decided by the mapped bond patterns at the two branch termini (at the
R-side terminus the R bond set is fully satisfied while the P set misses
exactly the reaction's edits): local pair distances are immune to the
conformer noise that demonstrably mislabels a whole-molecule-RMSD decision,
which survives only as the fallback for bond-set-identical (pure
order-change) reactions — `orientation_basis` records which signal decided
and `orientation_bond_scores` the pattern evidence. Every mapped bond event
(formed/broken/order_changed/hydrogen_migration) is checked against its
branch distance curve with a typed `support|weak|mismatch` verdict — and a
`mismatch` requires a known orientation: under an unresolved one,
contradiction verdicts are capped to `weak`. A single-frame distance never
overrides the graph.

Statuses (all typed, all persisted): `resolved_unique`,
`resolved_symmetry_collapsed`, `unresolved_reactive_center`,
`unresolved_truncated`, `ts_irc_endpoint_mismatch`,
`atom_or_element_mismatch`, `missing_truth_join`, `source_structure_mismatch`.
Only the first two are G2-eligible.

### P2 — hierarchical reaction taxonomy (`g1 classify`)

P2 reads **only** P1 artifacts (never truth geometry) and classifies every
G2-eligible reaction into four levels: `l0_edit_family` (event-count
composition), `l1_center_template` (labeled reaction-center graph),
`l2_context_r1`/`l3_context_r2` (one/two-shell neighborhood templates).
Cluster ids are `taxonomy_version + level + sha256(canonical signature)`
where the signature comes from an individualization-refinement canonical
labeling whose result is the lexicographic minimum of the template and its
R↔P reversal — so map relabeling, component reordering, candidate
reshuffling, or writing the reaction backwards can never split one
transformation into two clusters (unit-locked). Rule-based multi-labels
(`addition`, `substitution`, `elimination`, `rearrangement`, `ring_closure`,
`ring_opening`, `fragmentation`, `h_transfer`, `other`) and the IRC-derived
pathway labels (orientation, endpoint match, synchrony, support histogram)
are stored as separate columns, never mixed into the structural cluster ids.
A canonicalization budget exhaustion is flagged on the document, never
silent. The cluster report adds per-level cluster counts, size histograms,
singleton fractions, family-label distributions, P1-status cross-tabs, IRC
evidence quality, representatives, and the frozen-G0-split cross
distribution (reported only — the split is never reordered).

### G2 eligibility gate (`g1 gate`)

The gate re-runs both verifications (`g1 verify --stage p1`/`--stage p2`
re-read every written document, recompute the P2 classifications, and
reconcile counts and digests), refuses on any problem, and writes
`g1_gate.json` (denominator, eligible count, eligible fraction, exclusion
breakdown) plus `g2_eligible.json` — the pure reaction-id list G2 may
consume. Every reaction is either eligible or carries an exclusion reason;
nothing is dropped silently.

### Truth-isolation exception

`pes2ts_core/g1/p1_truth.py` is the **only** G1 module allowed to touch the
quarantined truth (static guard allowlist), and it does so exclusively
through the audited accessors — including two bulk accessors added for
annotation runs: `load_ts_table` (one audited table read instead of 200k
filtered reads) and `iter_irc_trajectories` (one streaming pass over the
IRC archive, one audit line per reaction, with the calling module
recorded). P1/P2 artifacts persist mappings, event verdicts, cluster ids,
and audit digests — never TS/IRC coordinates or energy arrays (the P1
verifier rejects any document containing them). CLI truth reads demand an
explicit `--allow-truth` (exit 23 otherwise).

## Testing

The default test suite requires **no network and no large data download**:

```bash
conda run -n pes2ts python -m pytest -q
```

It builds synthetic HDF5/CSV fixtures at test time and covers HDF5/CSV schema
validation, reaction-id normalization, map/direction/stereo identity invariance,
exact-duplicate and reverse detection, the DRFP audit including the budget abort,
official split adoption, split freeze plus both remediations and the leak halt,
the strata preview and deterministic cohorts, the ground-truth guard (import/path
/file/dynamic-exec detection, relocation, and the access log), and an end-to-end
fixture pipeline run. The G1 suite adds synthetic fixtures for all seven change
categories (including pure-formed/pure-broken/order-change-only, which are
near-absent in the real inventory), component matching and index
ambiguity/truncation, the preview cross-check, the typed rejection paths,
document/summary determinism, and the authoritative strata re-export. The
truth-layer suite adds join-audit reason fixtures, IRC layout detection
(ts-first/ts-last/single-branch/rotated-trajectory/interior TS), the
candidate solver (unique best, symmetric collapse, order invariance,
identical-geometry permutations, truncation), per-event IRC verdicts
(support/weak/mismatch, non-finite curves, synchrony), P2 invariance
(map relabeling, direction reversal, component reordering, context
splitting, budget exhaustion), the audited bulk truth accessors, and a
synthetic end-to-end P1→P2→verify→gate pipeline with CLI contracts.
Tally: **355 passed, 3 deselected**.

Checks that need Zenodo or the ~12 GB download are marked `realdata` and excluded
by default via `pytest.ini` (`addopts = -m "not realdata"`). List them with:

```bash
conda run -n pes2ts python -m pytest -q -m realdata --collect-only
```

Three gated tests exist: `test_fetch.py::test_realdata_manifest_and_no_redownload`,
`test_dedup.py::test_real_inventory_full_pass_within_budget`, and
`test_neardup.py::test_real_data_audit_completes`.

Determinism philosophy: every JSON/Parquet artifact is reproducible from the
inputs and `split.seed`; the only permitted variation across identical runs is
the timestamp-valued `VOLATILE_KEYS` (`generated_at`, `downloaded_at`,
`duration_seconds`). `g0 run-all` is idempotent — a second run over an unchanged
tree skips every stage and rewrites only the run report.

Per-todo command transcripts and outputs are recorded under
`.omo/evidence/task-*-pes2ts-g0-data-entry-and-split.txt` (G0) and
`.omo/evidence/task-*-pes2ts-g1-general-s0.txt` (G1).
