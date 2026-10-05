# PES2TS

> **开发宪法（规范根）**：本仓全部开发行为受 [docs/PES2TS_开发宪法_v1.md](docs/PES2TS_开发宪法_v1.md) 约束；[AGENT.md](AGENT.md) 是其执行投影。任何规范冲突以宪法为准。

## Current project direction

PES2TS is the integration of two separately testable projects: **PES generation**
(G1 plans and G2 executes auditable, approximate reaction paths) and **PES
ranking** (validates path usability and ranks calculated frames as TS seeds).
The strict outcome is a target TS confirmed by OptTS, frequency, and two-way
IRC; a path or high-energy frame alone is an intermediate result. The
repository now includes v1 shared data contracts, endpoint-only ReactionCase
conversion, a minimal one-coordinate ScanPlan, ACP request/result projections,
a local ACP CLI attempt runner, and a replaceable synthetic path ranker with an
offline energy/structure viewer. It also has a read-only collector for ACP
BatchOptimize TS/frequency and IRC result artifacts; scheduler-managed task
registration, stage execution orchestration, and validation against a real
reaction remain later-stage integrations.

See the [current development plan](docs/plans/PES2TS_双项目开发与Demo方案.md), the
[unified stage acceptance order](docs/plans/PES2TS_统一开发顺序与阶段验收.md), the
[contracts v1 field dictionary](docs/contracts/PES2TS_contracts_v1.md), the
[ACP v1 field mapping](docs/contracts/PES2TS_ACP字段映射_v1.md), the
[24-reaction Demo candidate set](docs/demo/PES2TS_demo24_反应挑选与复核方案.md), the
[Demo evaluation metrics v1](docs/demo/PES2TS_Demo评估指标_v1.md), and the
[G1 completion plan](docs/design/G1_v2_补全实施总方案.md). The older
[step-by-step plan](docs/plans/PES2TS_逐步实施与验证方案.md) is retained as historical context.
The [stage report](docs/reports/PES2TS_开发阶段性报告_20260930.md) and the
[scan-planning lineage merge plan](docs/plans/PES2TS_扫描规划谱系合并方案_20260930.md)
record the 2026-09-30 checkpoint.

Run `python bin/pes2ts demo` to write the synthetic seven-object bundle and
offline viewer to `examples/contracts_v1`. The energy curve demonstrates that
highest-energy and internal-peak rules can select different frames without
recomputing the path; the viewer links each selected frame to its structure.
The bundle also includes a rejected plan, a failed attempt followed by a retry,
a missing energy, and an unrun validation result. To view any PathBundle, run
`python bin/pes2ts view-path --path <PathBundle.json> --output <viewer.html>`.
The first real-input preflight plan is under `examples/m1_first_plan`; it is not
an ACP submission or a calculated path.
It is a hash-bound preflight snapshot. After the pending chemistry review is
accepted, regenerate the plan from the approved ReactionCase snapshot before
any ACP submission; the current review cases have updated source-spin evidence.

## One local ACP CLI attempt

`acp-run` launches the ACP `PESsearch` CLI in a separate process using a frozen
ScanPlan. It requires an accepted `ReviewRecord` linked to the exact ready
ReactionCase produced by the review importer. Each invocation has a unique
`execution_id` and immutable `attempt_id`; an interrupted attempt is recovered
from its receipt when ACP already finalized the result, while a retry uses new
IDs. The command applies the plan wall-time budget, captures the ACP log and
exit status, verifies the ACP v2 result manifest and products, reads frame XYZ
files from `WORK`, and publishes the same atom-ordered geometries under
`RESULT/pes2ts/frames/`, registered as hash-bearing products in ACP's v2
manifest. The collected `PathBundle.geometry_ref` points to those `RESULT`
files, while the native ACP profile and `WORK` frames remain intact. It writes
PES2TS `ExecutionRecord`, `PathBundle`, and path quality files beside the ACP
attempt directory. CLI mode does not create an ACP scheduler task ID or claim
a TS has passed physical validation. CPU time stays unknown unless ACP reports
it.

```powershell
python bin/pes2ts acp-run `
  --case <accepted-ReactionCase.json> `
  --review-record <accepted-ReviewRecord.json> `
  --plan <ScanPlan.json> `
  --acp-root E:/Calculations/Common_Script/Auto_Calc_Platform/ACP_V1_20260811 `
  --python <ACP-environment-python.exe> `
  --acp-config <ACP-config.yaml> `
  --output-root outputs/acp_cli `
  --execution-id <unique-execution-id> `
  --attempt-id <unique-attempt-id>
```

This command starts a real local calculation. The CLI backend has automated
regression coverage with an isolated fake ACP process, but no accepted real
reaction has yet been submitted through it.

## Train-first cohort execution

`acp-demo-run` executes a frozen cohort manifest sequentially. The manifest
lists every ReactionCase snapshot under `cohort_cases` and exactly one `runs`
row for each case; each row may point to an accepted ReviewRecord. Train cases
run by default. Valid cases remain in the denominator and are marked held out
unless `--include-valid` is explicitly supplied. Each run writes its plan,
execution record, collected path, quality decision, ranking proposals, and
offline viewer where those stages succeed, followed by a cohort index, run
bundle, and split-aware metrics. Metrics report measured wall time from each
attempt receipt and keep CPU cost unknown when ACP does not provide it. Use a
new/empty output root for each immutable cohort run.

```powershell
python bin/pes2ts acp-demo-run `
  --manifest <DemoExecutionManifest.json> `
  --acp-root E:/Calculations/Common_Script/Auto_Calc_Platform/ACP_V1_20260811 `
  --python <ACP-environment-python.exe> `
  --output-root outputs/demo24_train_run
```

The manifest contains `schema_version: "pes2ts_demo_execution_manifest_v1"`,
bundle-relative JSON references in `cohort_cases` and `runs`, and optional
frozen `experiment_id`, `method`, `n_points`, `budget`, and `ranking_labels` values.
Ranking labels are a separate JSON array with a source SHA256 and are never
inferred from generated paths. A `budget` object freezes `max_attempts`,
`max_cpu_hours`, and `max_wall_seconds`; each retry gets a new attempt ID, and
all retries share the enforced wall-time limit. ACP CLI does not expose a
measured CPU counter, so `max_cpu_hours` is recorded but cannot be certified as
a hard runtime limit. Supplying
`--include-valid` enables reviewed valid cases for the one-time evaluation
run. This runner has only been exercised against a fake ACP CLI; the 24 sample
cases still require human chemistry review before any real run.

After separately completing ACP BatchOptimize (TS-tagged structure plus
frequency) and two-way IRC tasks, `acp-collect-validation` verifies their ACP v2
manifests, file hashes, method/basis, selected proposal geometry, normal modes,
and endpoint matches, then writes a `ValidationResult`. It only collects and
checks existing products; it does not launch those calculations. A passed
result therefore requires genuine ACP products from the same reviewed case and
proposal. Use `python bin/pes2ts acp-collect-validation --help` for the required
artifact arguments.

`acp-validate-run` executes those two ACP stages in sequence. It checks the
accepted review and proposal binding, runs a TS-tagged `opt_freq` BatchOptimize,
checks its actual method and optimized geometry, then writes ACP's required
hash-bound TS provenance and starts a two-way IRC. Each stage gets a separate
execution/attempt ID, receipt, log, and timeout. After both stages, it writes a
ValidationResult from the ACP manifests and products. Failed BatchOptimize,
bad TS/frequency evidence, failed IRC, or rejected IRC products still produce a
failed ValidationResult with the failed stage and its attempt identity recorded;
IRC is skipped unless frequency establishes a first-order saddle. The result
also carries receipt references, manifest hashes, return codes, and wall-time
per attempt; ACP does not currently report CPU time, so CPU cost remains
explicitly unavailable.

```powershell
python bin/pes2ts acp-validate-run `
  --case <accepted-ReactionCase.json> `
  --review-record <accepted-ReviewRecord.json> `
  --path <usable-PathBundle.json> `
  --proposal <accepted-SeedProposal.json> `
  --source-frame-id <selected-frame-id> `
  --acp-root E:/Calculations/Common_Script/Auto_Calc_Platform/ACP_V1_20260811 `
  --python <ACP-environment-python.exe> `
  --output-root outputs/acp_validation `
  --batch-execution-id <batch-execution-id> `
  --batch-attempt-id <batch-attempt-id> `
  --irc-execution-id <irc-execution-id> `
  --irc-attempt-id <irc-attempt-id> `
  --method <frozen-method> --basis <frozen-basis> `
  --batch-timeout 7200 --irc-timeout 7200 `
  --validation-id <validation-id> --output <ValidationResult.json>
```

This starts real quantum-chemistry calculations. No calculation has been
launched through this sequence because the 24 sample reviews are still pending.

For the 24-reaction S1 review packet, run `python scripts/audit_demo24_endpoints.py`
then `python scripts/build_demo24_reaction_cases.py`, and run
`python scripts/audit_demo24_spin_sources.py` for the spin-source ledger. The
builder writes standard
endpoint-only cases under `data/interim/g1_v2/reaction_cases_review_v1` and a
provenance manifest under `data/manifests`. All 24 cases remain `needs_review`.
R/P spin states are now resolved from sanitized source component records when
total spin is unambiguous, with evidence recorded in each case and in
[`demo24_spin_source_audit_v1.json`](data/manifests/demo24_spin_source_audit_v1.json).
This does not replace independent chemistry review; reviewers must still accept
the endpoints and confirm spin before the cases can become ScanPlans.

The two-reviewer review workbook is
[`outputs/01a0ed78-73eb-7833-b3c1-4a6c0733a397/Demo24_双人化学复核.xlsx`](outputs/01a0ed78-73eb-7833-b3c1-4a6c0733a397/Demo24_双人化学复核.xlsx).
It contains independent reviewer sheets, an adjudication sheet, candidate
summaries, mapped reaction SMILES, and review instructions. Review decisions
must still be transferred to auditable `ReviewRecord`/`ReactionCase` records;
blank cells are not approvals.

After both reviewers and the adjudicator complete all 24 rows, import the
workbook with `python scripts/import_demo24_review_workbook.py <completed.xlsx>`.
The importer verifies row identities against the frozen ReactionCase manifest,
requires distinct reviewers and matching adjudication, and promotes a case to
`ready` only when both reviews confirm every dimension, agree on 1D feasibility,
and enter matching explicit R/P multiplicities. It writes a new reviewed-case
snapshot and review-record manifest without overwriting the review packet.

After a successful import, freeze the 16/8 execution input bundle with
`python scripts/build_demo24_execution_bundle.py --reviewed-cases <reviewed-root> --reviewed-manifest <review-records.json> --output <new-input-bundle>`.
The packer reconciles each reviewed case and ReviewRecord against the original
24-case IDs/splits, copies the exact snapshots, and writes a hash receipt. It
does not create approvals; accepted/rejected decisions must come from the
completed workbook. Run `acp-demo-run` on the resulting
`DemoExecutionManifest.json` first without `--include-valid`. After reviewing
and locking the train run, pass its `DemoExecutionIndex.json` with
`--include-valid --train-index <train-index.json>` for the valid evaluation.
The runner verifies the same frozen manifest hash, complete cohort accounting,
the same input/config/ACP Python-source fingerprint, and held-out valid rows
before starting that run. It carries the train artifacts into the final bundle
and calculates only valid cases, so train work is not submitted again.

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
| `g2 prepare` | `--reaction RXN_ID` (repeatable), `--cohort {trial,stratified,all}` (default `trial`), `--limit N`, `--force` | Assemble the deterministic R/P endpoints for the selected eligible reactions; write `endpoints.json`, `R.xyz`, `P.xyz`. |
| `g2 run` | `--reaction RXN_ID` (repeatable), `--cohort {trial,stratified,all}` (default `trial`), `--limit N`, `--force` | Run the GFN2-xTB PATH step (with a swapped-side retry when triggered) and write the per-reaction path artifacts plus the summary, manifest, and coverage. |
| `g2 verify` | — | Re-read the whole G2 tree and reconcile it against the summary and manifest; exit 24 on any inconsistency. |

`g2 prepare` and `g2 run` default to the `trial` cohort; `--cohort all` is the
explicit full-set switch. A completed G2 batch exits 0 even when individual
reactions failed (each failure is typed into the ledger and the manifests);
exit 24 is for infrastructure errors and `g2 verify` inconsistencies only.

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
| 24 | `EXIT_G2_FAILED` | A `g2` stage hit an infrastructure error (missing inventory/eligible/cohort/G1 inputs, missing ACP execution wiring), or `g2 verify` found any inconsistency. |

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
5. **Dynamic-execution exemptions.** xTB PATH execution now goes through the
   ACP CLI (ADR-0002): the reviewed ACP adapters
   (`pes2ts_core/integration/acp/cli_backend.py`, `stage_cli.py`, and the
   XtbPathSearch transport `xtb_path_transport.py`) sit on the narrow
   `SUBPROCESS_IMPORT_ALLOWLIST` and may import `subprocess` only; every other
   dynamic-execution primitive and every truth check still applies to them in
   full. The broader `DYNAMIC_EXEC_ALLOWLIST` — which used to exempt the local
   xTB runner — is now **empty** since that runner was deleted (X2′-C); no
   shipped module needs a dynamic-execution exemption, and the default test
   suite asserts the shipped package is clean.

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

## G1 v2 — audit and repair layer (`g1 v2-build` / `v2-classify` / `v2-verify` / `v2-gate`)

The L0 population counts and earlier routing analysis are documented in [G1 v2 L0 到 G2 扫描策略设计](docs/design/G1_v2_L0到G2扫描策略设计.md). The current responsibility split and G1-owned ORCA scan-mode plan are specified in [G1 v2 ORCA 扫描模式识别与计划](docs/design/G1_v2_ORCA扫描模式识别与计划.md).
The one-dimensional-first path strategy and the xTB versus 2D-grid cost analysis are in [G1 v2 一维优先与 xTB 二维成本策略](docs/design/G1_v2_一维优先与xTB二维成本策略.md).
The earlier fixed-product-side reverse-scan analysis and H-transfer shortcut are in [G1 v2 产物成键逆向扫描优先策略](docs/design/G1_v2_产物成键逆向扫描优先策略.md). The direction-independent more-bond anchor baseline is in [G1 v2 方向无关成键锚点与双向扫描策略](docs/design/G1_v2_方向无关成键锚点与双向扫描策略.md). The current recommended hybrid selector, comparing more-bond and fewer-bond anchors, is in [G1 v2 多成键与少成键方向对比](docs/design/G1_v2_多成键与少成键方向对比.md).
For team discussion, open the self-contained [G1 v2 成断键扫描方向选择可视化](docs/design/G1_v2_成断键扫描方向选择可视化.html) in a browser; it includes an interactive F/B selector and the G1→G2 handoff.
The integrated implementation contract and acceptance gates are in [G1 v2 补全实施总方案](docs/design/G1_v2_补全实施总方案.md).

The v1 bond-edit semantics were deliberately **not mutually exclusive** (a
single→double bond change produced one `formed` key, one `broken` key, *and*
one `order_changed` pair for the same atom pair; 165,165 of 191,148 admitted
records carried bond-order changes under that reading), the P2 L0 family and
the rule labels consumed those overlapping counters, the gate released
records carrying IRC event mismatches as ordinary eligible records, and the
canonicalization fallback serialized by map-number insertion order so the
budget-exhausted records (868) changed cluster ids under pure map relabeling.
The v2 layer repairs all of this in **independent artifact paths** — v1
trees, the frozen split, and the v1 gate outputs are never overwritten.

### v2 bond edits (`g1 v2-build`)

Every admitted P1 reaction gets one v2 document
(`data/interim/g1_v2/edits/<shard>/<rid>.json`, schema `g1_v2_edits_v1`)
holding:

- **Mutually exclusive edits.** One record per unordered map pair with
  `edit_kind` ∈ {`formed` (R-absent/P-present), `broken` (R-present/P-absent),
  `order_changed` (both present, different order)} and both original bond
  orders preserved (aromatic stays `1.5`). Edited aromatic bonds carry their
  **conjugated-region id** (`aromatic_regions` groups a whole re-kekulizing
  ring into one region, so G2 never scans six independent coordinates for one
  ring event).
- **Hydrogen partner changes** with an exhaustive partner-state taxonomy:
  `transfer` (heavy→heavy), `release`/`capture` (bound⇄free),
  `to_hh`/`from_hh`/`hh_release`/`hh_form_free`/`hh_swap` (H–H events). H₂
  formation is never labelled an H transfer.
- The embedded map-space graph, the v2 reaction center, per-kind edit
  counts, and the complete audit verdict.

The build first freezes the v1 baseline (`data/manifests/g1_v2_freeze.json`:
the 199,217 denominator, 191,148 eligible, 638 v1 L0 clusters, status
histograms, digests), then audits every record along three separated
dimensions with typed issue codes:

| Dimension | Issue codes | Meaning |
| --- | --- | --- |
| `structural_validity` | `map_bijection_broken`, `element_sequence_mismatch`, `component_assignment_mismatch`, `legacy_event_inconsistency`, `multi_bond_pair`, `edit_contract_violation` | the map table is a bijection onto the component rows, elements agree at every (component, row), the v1 `bond_events` are exactly derivable from the same graph, and the v2 block satisfies its own contract |
| `mapping_determinism` | `collapse_edit_ambiguous`, `collapse_audit_truncated`, `candidate_search_truncated` | symmetry-collapse ties and truncated G1 candidate searches |
| `irc_evidence_quality` | `irc_event_mismatch`, `orientation_unresolved` | IRC evidence conflicts (previously silently eligible) |

**Symmetry-collapse equivalence proof.** The v2 edits, reaction center, and
classifications are functions of the mapped reaction-SMILES graphs alone;
candidate bijections and same-skeleton pairing permutations only reassign
which inventory geometry row backs each map number, so a
`resolved_symmetry_collapsed` tie provably cannot change the recorded
chemistry. The build verifies every stored G1 alternative is
element-consistent with that proof (`collapse_state` = `invariant`, with the
verified-alternative count as evidence); a candidate that breaks element
consistency is flagged `collapse_edit_ambiguous`. Same-skeleton pairing
permutations are accepted by the component check (per-tag map sets must
partition the maps into component map sets), not misread as tag/ordinal
mismatches.

Outputs: `g1_v2_summary.parquet` (one row per reaction: audit status, issue
list, dimensions, exclusive counts, H-change counts by class, aromatic
counts, collapse state), `g1_v2_manifest.json` (status/issue/dimension
histograms, collapse-state histogram, digests), and the per-reaction issue
ledger `data/manifests/g1_v2_issue_ledger.jsonl`.

### v2 taxonomy (`g1 v2-classify`)

Five levels per record (`data/interim/g1_v2/classes/<shard>/<rid>.json`,
schema `g1_v2_class_v1`, taxonomy `g1p2_taxonomy_v2`): the **directional**
`l0_edit_family` (`F/B/O/H` exclusive counts — the scan-facing layer), the
**direction-invariant** `l0u_undirected_family` (F/B swap-normalized — the
chemical family), and `l1`–`l3` center/context templates over exclusive-edit
roles. Two v1 defects are fixed: the individualization search branches on the
smallest-*color* cell (colour ranks are canonical; v1 branched by map-number
lexicographic order, making leaves map-dependent), and a budget exhaustion
aborts the whole search into a **group-based WL-stable serialization**
(colour classes + connection-label multisets; no map numbers), so the id is
mapping-number-invariant by construction and still flagged
`canonical_budget_exhausted`. Rule labels are recomputed from exclusive
counts (H₂ events get no `h_transfer`; a pure H₂ formation is an `addition`).
`g1_v2_migration.json` records the v1→v2 family-label transition matrix, the
L0 cluster counts, and per-split L0u distributions.

### v2 verification and gate (`g1 v2-verify` / `g1 v2-gate`)

`v2-verify` **recomputes** every edit block, center, and classification from
the persisted documents (not just field/digest reconciliation) and enforces
the export blacklist (`ts_irc_index`, IRC frames, coordinates of truth,
energies may never appear). `v2-gate` verifies first, refuses on any problem,
then splits the population three ways — **`scan_ready`** / **`needs_review`**
(with typed reasons and dimensions, exported to `g2_needs_review.json` as the
adjudication queue; IRC-mismatch and truncated-search records land here, no
longer inside a single eligible number) / **`excluded`** (`p1:<status>`) —
and writes, for every scan-ready reaction, a **whitelisted G2 export**
(`data/interim/g1_v2/export/<shard>/<rid>.json`, schema `g1_v2_export_v1`):
map-ascending `r/p_atomic_numbers` + `r/p_coordinates` (assembled through
the P1 bijection from the inventory endpoint components — endpoint data, not
truth), the component bijection table, charges/spins, the exclusive edits,
aromatic regions, reaction center, v2 classification, and
`mapping_provenance="truth_assisted_p1"`. The gate manifest
(`g1_v2_gate.json`) reports per-reason and per-dimension breakdowns plus the
scan-ready split histogram; `g2_scan_ready.json` is the pure reaction-id
list G2 may consume.

Configuration (`g1_v2` in `config/defaults.yaml`): `shard_size` 1000,
`collapse_audit_budget` 256, `center_shell` 1, `taxonomy_version`
`g1p2_taxonomy_v2`, `canonical_budget` 2000.

## G2: R/P endpoint assembly and cheap GFN2-xTB paths

Stage **G2** turns the eligible reaction list (`g2_eligible.json`, 191,148 ids
in the reference tree) into one versioned reaction path per reaction, entirely
**truth-free**. It reads the transition-state-free inventory (component
coordinates, charge, multiplicity), the cohort files, and the G1 change
documents; it never opens the quarantined TS/IRC truth, and its inputs carry no
TS-derived fields. Per reaction, G2 first **assembles the R/P endpoints** into
a single map-ordered frame (rigid placement of multi-component sides plus
cross-component clash separation), then runs the **GFN2-xTB PATH**
metadynamics on the assembled `start.xyz`/`end.xyz`, parses the per-frame
energies and geometries, and applies the validity predicates below. The
endpoint-reach judgement is measured after optimal full-atom superposition:
xTB's output frames drift as whole-molecule rigid rotations, and xTB's own
`product-end path RMSD` (and RPH v4.0.1's product reference RMSD) are
likewise superposed metrics. Every selected reaction ends with a terminal path
document; failures carry a typed `failure_code` and one entry in the unified
rejection ledger.

### G2 inputs and artifacts

Selection is a strict intersection of the eligible id list, the cohort (or the
whole inventory for `--cohort all`), and any explicit `--reaction` ids; the
result is sorted and truncated by `--limit`. An explicit `--reaction` does not
bypass the cohort filter.

Inputs: `data/interim/g2_eligible.json` (the id list written by `g1 gate`, or
`g2.eligible_path` when set), `data/interim/inventory.parquet`,
`data/interim/cohort_trial.json` / `cohort_stratified.json`, and the G1
documents `data/interim/g1/reaction_change/<shard>/<reaction_id>.json`.

| Artifact | Path | Produced by | Key fields / columns |
| --- | --- | --- | --- |
| `endpoints.json` | `data/interim/g2/paths/<shard>/<reaction_id>/` | `g2 prepare` | Assembly record: `groups`, `placements`, `separations`, `metrics`, `candidates`, the map-ordered atom tables, `multiplicity_basis`. |
| `R.xyz` / `P.xyz` | same | `g2 prepare` | Map-ordered endpoint frames (`element x y z`, 6 decimals); the comment line carries the reaction id and the direction marker. |
| `run/` | same | `g2 run` | `start.xyz`, `end.xyz`, `path.inp`, `xtb_path.log`, `xtbpath.xyz`, `xtbpath_ts.xyz`; trial segments (`xtbpath_<n>.xyz`) are ACP-owned RESULT artifacts — never parsed, never deleted by PES2TS, and `xtbpath.xyz` is the sole parsing authority. A reverse retry lives in `run_reverse/` with the same layout. |
| `frames.parquet` | same | `g2 run` | One row per path frame: `reaction_id`, `frame_index`, `energy_rel_kcal`, `energy_rel_kcal_raw`, `rmsd_to_start`, `rmsd_to_end`, `step_max`, `step_rmsd`, `min_nonbonded_distance`, `event_distances` (JSON string). |
| `reaction_path.json` | same | `g2 run` | `schema_version="g2_path_v1"`: status, failure code, direction and `direction_recovered`, scalar frame summary, validity verdict, attempt history, and source digests. No coordinate or energy arrays. |
| `g2_summary.parquet` | `data/interim/` | `g2 run` | One row per selected reaction: `reaction_id`, `status`, `failure_code`, `direction`, `direction_recovered`, `n_frames`, `n_attempts`, `energy_min`, `energy_max`. |
| `g2_path_manifest.json` | `data/manifests/` | `g2 run` | `schema_version="g2_manifest_v1"`: `n_total`/`n_valid`/`n_failed`, `by_code`, `reverse_recovery_rate`, the `run` counter block (`n_selected`/`n_attempted`/`n_skipped`), `summary_sha256`, the xTB fingerprint (executable `sha256`, version line, full `argv`, `OMP_NUM_THREADS`, seed support and seed — from the last attempt of the last attempted reaction, or `null` when no attempt ran — plus the `path.inp` text), `config_digest`, `generated_at`. |
| `g2_coverage.json` | `data/manifests/` | `g2 run` | `schema_version="g2_coverage_v1"`: overall counts, the seven G1 change categories crossed with valid/failed, `current_by_code`, and `historical_failed_by_code` (every `stage="g2_path"` ledger entry, so `--force` re-runs keep earlier failures visible). |

`<shard>` is the numeric reaction id divided by `g2.shard_size` (default
`1000`), five-digit zero-padded, the same shard naming as the G1 documents.
`g2 verify` re-reads the entire tree and reconciles schema, counts, `sha256`
digests, frame row counts, and forbidden keys; it never reads the ledger, so a
historical failure entry cannot fail verification.

### Endpoint assembly

Each side's atoms are ordered by reaction-global map and validated
element-by-element against the inventory; only Q=0 / M=1 reactions are
accepted (`multiplicity_basis="g1_valid_invariant"`). Because the reactant and
product coordinates are stored in different Cartesian frames, multi-component
assembly is rigid rather than a coordinate concatenation:

- **Three deterministic candidates.** C1 anchors the largest reactant
  component, C2 the second-largest reactant component (when the side has at
  least two), C3 the largest product component (when the product side has at
  least two). The anchor component keeps its stored coordinates and defines the
  candidate frame.
- **Cross-side bipartite BFS.** Components are nodes `(side, tag)`, edges are
  shared reaction-global maps, and expansion is deterministic (shared-map count
  descending, R before P, tag ascending). Each new component is placed with the
  Kabsch transform onto the already-placed component it shares the most maps
  with. The placement basis is tried in order: shared maps outside
  `reaction_center.with_shell` (at least `g2.assembly.min_anchor_maps` = 3 and
  non-degenerate), all shared maps (at least 3 and non-degenerate), then
  translation only (shared-map centroid alignment). A set is non-degenerate
  when the centered anchor points pass the rank gate (both singular values
  above `g2.assembly.anchor_tolerance`, 1e-3 Å), which excludes collinear and
  coincident sets.
- **Disconnected spectator groups.** A bipartite component with no geometric
  constraint to the candidate frame keeps its reactant stored coordinates for
  both sides, is flagged `assembly_basis="stored_frame_per_group"` and
  `frame_ambiguous=true`, and still has to pass the collision check; an
  overlap is a terminal `G2_ASSEMBLY_COLLISION`, never a silent acceptance.
- **Changed-pair separation.** On the R assembly every `formed` pair, and on
  the P assembly every `broken` pair, whose two atoms sit in different placed
  components closer than `max(2.0 Å, radius_sum + 0.45 Å)` is pushed apart to
  `max(3.0 Å, radius_sum + 0.45 Å + 0.5)`. The element-aware threshold uses the
  Cordero covalent radii; every evaluated pair is recorded
  (`separation_evaluated`, `triggered`, `d_before`, `d_after` or `d_placed`).
- **Scoring.** The candidates are ranked by the tuple `(n_severe_contacts,
  -min_nonbonded_distance, formed_excess, candidate_index)`, smallest wins;
  "nonbonded" means bonded in neither the R nor the P bond graph, and
  `n_severe_contacts` counts nonbonded pairs below
  `g2.validity.collision_min_distance`. All candidate scores are persisted in
  `endpoints.json`. A chosen assembly that still has a nonbonded pair below
  0.8 Å is a terminal `G2_ASSEMBLY_COLLISION`: no xTB run and no reverse retry.

### Validity predicates and failure codes

Each path is judged by the first violated predicate in this precedence order:

1. `G2_XTB_FAILED`: xTB exited non-zero, timed out, or a key output
   (`xtbpath.xyz`, `xtbpath_ts.xyz`, `xtb_path.log`) is missing or malformed.
2. `G2_ENDPOINT_NOT_REACHED`: the frame-0-vs-reactant (`first_vs_R`) or
   last-frame-vs-product (`last_vs_P`) RMSD **after optimal Kabsch
   superposition** exceeds 0.5 Å (OR semantics; equality passes). The predicate
   reads the superposed summary values `first_vs_r_rmsd_aligned` /
   `last_vs_p_rmsd_aligned`; the raw audit values `first_vs_r_rmsd` /
   `last_vs_p_rmsd` remain in the summary and the failure detail, and the
   `frames.parquet` columns retain the raw in-place per-frame metrics.
3. `G2_TOPOLOGY_DRIFT`: any event pair violates its PASS predicate (formed:
   first frame at or beyond the covalent-radius sum + 0.45 Å and last frame
   bonded; broken: the reverse; order_changed: bonded at both ends;
   h_migration: the H-from contact bonded at the first frame and the H-to
   contact bonded at the last).
4. `G2_ENERGY_INCOMPLETE`: any frame energy is missing or non-finite.
5. `G2_PATH_DISCONTINUOUS`: a single-atom step above 4.0 Å between adjacent
   frames, or fewer than 8 frames.
6. `G2_COLLISION`: any frame has a nonbonded pair closer than 0.8 Å.

The four `g2.validity` scalars are `endpoint_rmsd_max=0.5`,
`max_frame_step=4.0`, `min_frames=8`, and `collision_min_distance=0.8`; all
four were retained after the two-reaction pilot
(`.omo/evidence/task-15-g2-pilot-report.md`). `endpoint_rmsd_max=0.5` is
applied to the superposed endpoint values and is retained unchanged.
**Order-change nuance:** G1 lists
an order change under `formed`, `broken`, *and* `order_changed`; the formed and
broken PASS predicates contradict each other on such a pair, so a pair present
in `order_changed` is judged only by the order-changed predicate (bonded at
both ends).

| Code | Stage | Meaning |
| --- | --- | --- |
| `G2_NOT_ELIGIBLE` | prepare | The selected id is not in the eligible list. |
| `G2_MISSING_G1_DOC` | prepare | The reaction's G1 change document is missing. |
| `G2_G1_NOT_VALID` | prepare | The G1 document has `validation.status != "valid"`. |
| `G2_ENDPOINT_MISMATCH` | prepare | Side validation failed (map set, element sequence, or charge/multiplicity invariant). |
| `G2_ASSEMBLY_FAILED` | prepare | A malformed document, a missing component row, or a non-finite placement transform. |
| `G2_ASSEMBLY_COLLISION` | prepare | The chosen assembly still has a nonbonded pair below 0.8 Å. |
| `G2_XTB_FAILED` | run | xTB exited non-zero or timed out, or a key output is missing or malformed. Never reverse-retried. |
| `G2_ENDPOINT_NOT_REACHED` | run | Aligned `first_vs_R` or `last_vs_P` exceeds 0.5 Å; the raw in-place values are carried alongside in the failure detail (the summary and `frames.parquet` retain them). |
| `G2_TOPOLOGY_DRIFT` | run | An event pair violates its PASS predicate. |
| `G2_ENERGY_INCOMPLETE` | run | A frame energy is missing or non-finite. |
| `G2_PATH_DISCONTINUOUS` | run | A step above 4.0 Å, or fewer than 8 frames. |
| `G2_COLLISION` | run | A frame has a nonbonded pair below 0.8 Å. |

Every selected reaction ends with one terminal document and, if failed, one
`stage="g2_path"` ledger record. Re-running the same failure adds no duplicate
record (the ledger signature is idempotent), so the history stays append-only
without growing on retries.

### Reverse retry and direction normalization

When a forward run terminates with a code in `g2.reverse_retry.trigger_codes`
(the five validity codes above, and only when `g2.reverse_retry.enabled=true`),
G2 retries once with the sides swapped (`start=P.xyz`, `end=R.xyz`) in
`run_reverse/`. `G2_XTB_FAILED` is deliberately absent: an infrastructure
failure is not retried. A successful reverse run is normalized back to R→P
order before it is judged:

- the frame sequence is reversed;
- `energy_rel_kcal` is recalibrated so the new frame 0 (the R endpoint) is
  zero, while the raw xTB values stay in `energy_rel_kcal_raw`;
- the frame metrics are recomputed, and the document records
  `direction="reverse"` with `direction_recovered=true`.

The retry verdict is terminal: the pipeline does not compare forward and
reverse endpoint quality, so a reaction whose retry fails is a failure even
when the forward attempt was closer (both attempts remain in `attempts`). If
both directions fail, `status="failed"` and `failure_code` is the reverse
terminal code. A reaction with a terminal document is skipped on rerun unless
`--force` is passed; a fully skipped batch makes zero xTB calls and leaves the
terminal artifacts byte-identical.

### Determinism boundaries

- `g2 prepare` is byte-identical across runs on the same inputs (it always
  rewrites; there is no prepare-side resume).
- The derived `g2 run` artifacts (`reaction_path.json`, `frames.parquet`,
  summary, manifest, coverage) are byte-identical given the same raw xTB
  output.
- The raw xTB PATH runs are **not** byte-reproducible (metadynamics). The
  manifest records the full argv, the `OMP_NUM_THREADS` value, the executable
  `sha256` and version line, and seed support; `g2.xtb.seed=42` is passed only
  when the binary advertises a seed flag (xTB 6.7.1 does not, so
  `seed_supported=false` is recorded).
- `generated_at` is the only volatile JSON key. The `run` counter block varies
  with the call (`n_selected`/`n_attempted`/`n_skipped`) and is excluded from
  cross-call byte comparison. Parquet bytes depend on row order, so
  determinism comparisons use identical row lists.

### Truth isolation

G2 modules carry no truth imports, no `ground_truth`/`truth_sources` strings,
and no truth-file names; the static AST guard scans them like every other
module. xTB PATH execution goes through the ACP CLI adapters
(`pes2ts_core/integration/acp/`), which sit on the narrow
`SUBPROCESS_IMPORT_ALLOWLIST` (subprocess import only; every truth check stays
active); the former local runner and its `DYNAMIC_EXEC_ALLOWLIST` entry were
deleted per [ADR-0002](docs/design/decisions/ADR-0002-计算后端统一经ACP执行.md)
(X2′-C).

### G2 configuration keys

| Key | Default | Purpose |
| --- | --- | --- |
| `acp.root` | `null` | ACP checkout root (e.g. `.../ACP_V1_20260811`); mandatory for `g2 run` — a null/empty value exits 24 (no local xTB fallback). |
| `acp.python` | `null` | Python interpreter of the ACP environment (the one that provides xTB for ACP to launch). |
| `acp.config_path` | `null` | Optional ACP config YAML, passed to the ACP CLI as `--config`. |
| `acp.register` | `true` | Append `--register` to the ACP CLI so a completed run registers in the ACP jobs store (`acp_jobs.db` under `ACP_RUN_ROOT`, shared with the ACP server) and resolves in the Workbench (`/api/v1/jobs/{id}/s2/profile`). ACP registers only after workflow success, so a non-zero exit with a valid RESULT means registration failed: the attempt fails loudly with a typed `--register`/`ACP_RUN_ROOT` error (RESULT stays on disk), never a silent "completed" the Workbench cannot show. `false` = headless runs with no jobs store. |
| `g2.eligible_path` | `null` | Eligible id list; `null` means `<paths.interim>/g2_eligible.json`. |
| `g2.shard_size` | `1000` | Reactions per `paths/<shard>/` directory (same shard naming as G1). |
| `g2.xtb.threads` | `4` | Threads requested per ACP `XtbPathSearch` attempt; feeds the frozen `pes2ts_xtb_path_request_v1` recipe via `acp_backend.py`/`pipeline.py` (`-P`/`OMP_NUM_THREADS` inside ACP). |
| `g2.xtb.timeout_seconds` | `1800` | Per-attempt wall-clock budget; a timeout kills the ACP CLI process group and records `timed_out=true`. |
| `g2.xtb.seed` | `42` | Seed in the frozen request recipe; passed only when the xTB binary advertises a seed flag, seed support is recorded either way. |
| `g2.path.nrun` | `1` | `$path` block: number of PATH runs. |
| `g2.path.npoint` | `50` | `$path` block: interpolation points (raised from 25 after the pilot). |
| `g2.path.anopt` | `10` | `$path` block: anchor optimization cycles. |
| `g2.path.kpush` | `0.003` | `$path` block: push force constant. |
| `g2.path.kpull` | `-0.015` | `$path` block: pull force constant. |
| `g2.path.ppull` | `0.05` | `$path` block: pull pressure. |
| `g2.path.alp` | `0.5` | `$path` block: alpha (lowered from 1.2 after the pilot; 1.2 made real paths fail inside xTB with "No product"). |
| `g2.assembly.min_anchor_maps` | `3` | Minimum shared maps for the anchor-map Kabsch basis. |
| `g2.assembly.anchor_tolerance` | `0.001` | Å; singular-value threshold below which an anchor set is degenerate (collinear or coincident). |
| `g2.assembly.forming_min_distance` | `2.0` | Å; separation floor for a cross-component formed/broken pair. |
| `g2.assembly.forming_target_distance` | `3.0` | Å; separation target floor. |
| `g2.assembly.bond_tolerance` | `0.45` | Å; Cordero-radius-sum tolerance for bonded/unbonded classification and the separation thresholds. |
| `g2.validity.endpoint_rmsd_max` | `0.5` | Å; bound on the superposed `first_vs_R`/`last_vs_P` endpoint RMSD (retained from the pilot). |
| `g2.validity.max_frame_step` | `4.0` | Å; largest allowed single-atom step between adjacent frames. |
| `g2.validity.min_frames` | `8` | Minimum parsed frame count. |
| `g2.validity.collision_min_distance` | `0.8` | Å; nonbonded threshold used by the assembly collision check and `G2_COLLISION`. |
| `g2.reverse_retry.enabled` | `true` | Try the swapped start/end once after a triggerable forward failure. |
| `g2.reverse_retry.trigger_codes` | `G2_ENDPOINT_NOT_REACHED`, `G2_TOPOLOGY_DRIFT`, `G2_ENERGY_INCOMPLETE`, `G2_PATH_DISCONTINUOUS`, `G2_COLLISION` | Forward codes that trigger the swap; `G2_XTB_FAILED` is never retried. |

### Scale-up runbook

The full run is gated on the heavy tree, not on G2 itself. Restore the
truth-free inputs first: `inventory.parquet`, the cohort files, and the
**complete** G1 document tree (`g1 build` over the full inventory). The
committed `g2_eligible.json` holds 191,148 ids, but a missing G1 document is a
typed `G2_MISSING_G1_DOC` ledger entry, not an abort, so a partial document
tree produces a batch of typed failures instead of a usable path set. On a
tree that has not been quarantined yet, run `g0 quarantine` before any path
generation: it physically relocates the two truth-bearing HDF5 archives out of
the accessible raw tree. The pilot deliberately skipped that step, but a full
setup should not.

```bash
conda run -n pes2ts python bin/pes2ts g2 prepare --cohort all
conda run -n pes2ts python bin/pes2ts g2 run --cohort all
conda run -n pes2ts python bin/pes2ts g2 verify
```

Failure classification: read `g2_coverage.json` for the current batch (overall
counts, per-category valid/failed, `current_by_code`) together with
`rejection_ledger.jsonl`, where every failed reaction has one
`stage="g2_path"` record; `historical_failed_by_code` keeps earlier failures
visible after a `--force` re-run, and `g2 verify` re-checks the tree
independently (exit 24 on any inconsistency) without reading the ledger.

Parallelism: **v1 is serial**. The CLI runs one reaction at a time; per-shard
multiprocess execution is future work and does not exist today.

Cost: the pilot measured about 10 s per xTB attempt on the 18-20-atom pilot
systems (10.55 s for one forward run; 8.45 s + 9.61 s for a forward/reverse
pair). A serial full run over 191,148 reactions is on the order of three weeks
of wall time before retries, and larger reactions cost more; every triggered
failure adds another attempt.

### B97-3c: future optional layer (not implemented in this stage)

B97-3c single points are a planned optional refinement layer. The intended
shape, frozen for a later stage: a **stratified subset** of reactions gets a
B97-3c energy for **every valid frame** of its path (not only candidate
frames), computed with ORCA. The integration point is a nullable
`energy_b973c` column added to `frames.parquet` plus the corresponding manifest
field; the validity predicates and thresholds above stay unchanged. Nothing of
this exists in the current stage: no `energy_b973c` column, no manifest field,
and no ORCA invocation.

### ReactionProfileHunter reference note

The reference implementation for the xTB PATH step is the latest GitHub
[`PengYangchao0808/ReactionProfileHunter`](https://github.com/PengYangchao0808/ReactionProfileHunter)
**v4.0.1 (commit `3abbaec`)**. G2 follows the same command shape:

```bash
xtb start.xyz --path end.xyz --input path.inp -P <threads> --gfn 2 --chrg <q> --uhf <u>
```

([`rph_core/utils/xtb_runner.py` L859-L975](https://github.com/PengYangchao0808/ReactionProfileHunter/blob/3abbaecdd0b3c8cad6c4106c6e3ea07b6071e437/rph_core/utils/xtb_runner.py#L859-L975)).
Two RPH interfaces are reference points for a later stage, not current G2
features: the split of the multi-frame `xtbpath.xyz` into per-frame
`path_frames/` files (`split_multixyz`,
[`rph_core/utils/file_io.py` L158-L173](https://github.com/PengYangchao0808/ReactionProfileHunter/blob/3abbaecdd0b3c8cad6c4106c6e3ea07b6071e437/rph_core/utils/file_io.py#L158-L173)),
and the `ts_guess.xyz` / `scan_profile.json` candidate dictionary
(`{frame_index, xyz, confidence}`,
[`rph_core/steps/step2_retro/path_selector.py` L630-L648](https://github.com/PengYangchao0808/ReactionProfileHunter/blob/3abbaecdd0b3c8cad6c4106c6e3ea07b6071e437/rph_core/steps/step2_retro/path_selector.py#L630-L648)).
One deliberate difference: **G2 takes each frame energy from the path output's
own comment** (relative kcal/mol in `xtbpath.xyz`) and does **not** recompute a
single point per frame the way RPH does. All product reference RMSDs compared
in G2 (and in the RPH reference) are Kabsch-superposed. All line references
above are GitHub permalinks at the v4.0.1 commit; older v3.0.0 checkouts have
different line numbers and are not the reference.

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
synthetic end-to-end P1→P2→verify→gate pipeline with CLI contracts. The
G2 suite adds endpoint-assembly placement/separation/collision fixtures, the
strict xTB output parser against real pinned PATH fixtures, frame metrics and
validity predicates (including the superposed-endpoint regression), the ACP
attempt seam (fake-ACP helper driving the pipeline, plus the fake-ACP
transport/backend contracts), and the reverse-retry direction
normalization. Tally: see the latest CI/local run (the G2 runner tests were
replaced by ACP-seam coverage when the local runner was deleted, ADR-0002
X2′-C).

Checks that need Zenodo or the ~12 GB download are marked `realdata`, the
real-ACP xTB PATH smoke is marked `acp`, and any remaining real-binary checks
are marked `xtb`/`orca`; all four families are excluded by default via
`pytest.ini`
(`addopts = -m "not realdata and not xtb and not acp and not orca"`). List
them with:

```bash
conda run -n pes2ts python -m pytest -q -m "realdata or xtb or acp or orca" --collect-only
```

The gated real-input families include `test_fetch.py::test_realdata_manifest_and_no_redownload`,
`test_dedup.py::test_real_inventory_full_pass_within_budget`,
`test_neardup.py::test_real_data_audit_completes`, and
`test_g2_xtb_smoke.py::test_real_acp_xtb_path_smoke`. The ACP-gated smoke
test drives
`pes2ts_core.generation.execution.xtb_path.acp_backend.run_xtb_path_acp_attempt`
on a real ACP checkout (`XtbPathSearch` workflow launching GFN2-xTB) against
the small C7OH8 fixture and asserts a valid path verdict; it resolves the
checkout from the `PES2TS_ACP_ROOT` environment variable and the interpreter
from `PES2TS_ACP_PYTHON` (optional `PES2TS_ACP_CONFIG` for the ACP config
YAML), and **fails with an explicit message naming the missing variables when
neither resolves — it never skips**, so a missing ACP cannot masquerade as a
pass. Run it with:

```bash
PES2TS_ACP_ROOT=/path/to/ACP_V1_20260811 \
PES2TS_ACP_PYTHON=/path/to/acp-env/python \
conda run -n pes2ts python -m pytest -q -m acp
```

Determinism philosophy: every JSON/Parquet artifact is reproducible from the
inputs and `split.seed`; the only permitted variation across identical runs is
the timestamp-valued `VOLATILE_KEYS` (`generated_at`, `downloaded_at`,
`duration_seconds`). `g0 run-all` is idempotent — a second run over an unchanged
tree skips every stage and rewrites only the run report.

Per-todo command transcripts and outputs are recorded under
`.omo/evidence/task-*-pes2ts-g0-data-entry-and-split.txt` (G0) and
`.omo/evidence/task-*-pes2ts-g1-general-s0.txt` (G1).
