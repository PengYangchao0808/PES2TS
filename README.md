# PES2TS

PES2TS (Potential Energy Surface to Transition State) builds a traceable,
leakage-audited data foundation on top of the public Reaction-QM dataset. Stage
**G0** ("data entry and split") downloads and MD5-verifies the B3LYP-D3/TZVP
artifacts from Zenodo, turns them into a transition-state-free reactant/product
inventory with a typed rejection ledger, quarantines all transition-state and IRC
ground truth behind a single allow-listed accessor, adopts the authors' own
reaction-level train/valid/test split with an independent DRFP near-duplicate
audit, and emits deterministic trial and stratified cohort manifests. Every
manifest is versioned and reproducible from a fixed seed.

## Scope boundaries

G0 delivers **data entry, isolation, and split freezing only**. It deliberately
does **not**:

- generate any reaction path (no GFN2-xTB, no `xtb`/`ORCA`/`Gaussian`/`crest`
  execution);
- run any quantum-chemistry or QC validation work (no OptTS/frequency/IRC
  validation);
- produce the authoritative G1 bond-change table — it only publishes a clearly
  labelled `authoritative=false` preview (see *Strata preview*);
- produce G2 path manifests, G3 labels, or oracle-gap data;
- train any model;
- interpret IRC energies as physics (the index records shapes only).

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

All JSON manifests carry `schema_version = "g0_manifest_v1"` (except the JSONL
logs, which are line records) and a `dataset_version` of the form
`zenodo-18551029-rev1`. Determinism guarantee: two runs over the same inputs
produce **byte-identical** artifacts after removing the volatile keys
`VOLATILE_KEYS = {generated_at, downloaded_at, duration_seconds}`. Parquet
artifacts are written with a stable column order; ledgers are append-only.

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
   to `truth_access_log.jsonl` on each successful read. `g0 truth-index` is the
   audited CLI entry point.
4. **Static AST guard.** `pes2ts_core.utils.truth_guard.assert_no_truth_access()`
   parses every `.py` under `pes2ts_core/` and `bin/` and fails if any module
   outside the allowlist contains a string constant with `ground_truth` or
   `truth_sources`, names `B3LYPD3_TZVP.h5` or `B3LYPD3_TZVP_IRC.h5`, imports the
   truth package or `truth_reader`, or uses `subprocess`/`importlib`/`eval`/`exec`
   (`DYNAMIC_EXEC_RISK`, flagged for review). The allowlist is
   `pes2ts_core/g0/truth_quarantine.py`, `pes2ts_core/g0/truth/truth_reader.py`,
   and `pes2ts_core/utils/truth_guard.py`, plus `cli.py` **only** inside the
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
fixture pipeline run. Tally: **200 passed, 3 deselected**.

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
`.omo/evidence/task-*-pes2ts-g0-data-entry-and-split.txt`.
