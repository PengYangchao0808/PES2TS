# outputs/ storage policy

`outputs/` holds local run artifacts. It is **ignored by Git** and is not part
of the repository. Do not use `git add -f` on this directory.

## What lives here

| Path | Content | Size today |
|---|---|---|
| `outputs/g2t_campaign_*` | frozen G2.T campaign outputs | ~1.3 GB total |
| `outputs/pesn/**` | PES-N training / rollout / ORCA-xTB artifacts | local |
| `outputs/checks/` | small versioned check evidence (whitelisted files only) | small |
| `outputs/01a0ed78-.../` | Demo24 review workbook (versioned deliverable) | 2.6 MB |
| `outputs/demo24_review_images/` | review PNGs (versioned deliverable) | small |
| `outputs/pes_generation_strategy_design_v1/` | design-time strategy evidence (versioned) | small |

## Current decision (2026-10-18)

- `outputs/` is **not uploaded** to GitHub for now.
- The two 668 MB campaign directories remain local until an external archive
  location is agreed; record their SHA-256 before deleting anything.
- V5/V6 campaign outputs go to `outputs/pesn/v5/**` and stay local. Only small
  gate reports / model cards / checksums are committed.

## Do not commit

- checkpoints, `weights.pt`, ORCA `.gbw`/`.out`, xTB logs, trajectories;
- raw rollout JSONL/parquet;
- any file larger than ~5 MB.

A pre-commit sanity check should refuse staged paths under `outputs/` (except
the explicit whitelist above) and any `*.pt` / `*.ckpt`.
