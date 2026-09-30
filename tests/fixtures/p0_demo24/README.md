# P0 Demo24 frozen fixtures

Version-controlled inputs for the P0 golden acceptance test
`tests/test_p0_demo24_golden.py`. A clean clone can recompute the P0 feature
layer against
`outputs/pes_generation_strategy_design_v1/demo24_strategy_evidence.json`
(tracked in git) **without** the `data/` tree.

## Contents

| Path | What |
| --- | --- |
| `pes2ts_demo24_candidates_v1.csv` | Byte copy of the 24-reaction candidate CSV. SHA256 must equal `manifest.json.source_csv_sha256` and the evidence JSON `source_csv_sha256`. |
| `manifest.json` | Fixture manifest: source CSV/manifest sha256, RDKit pin, per-record `export_sha256` + mapped reaction SMILES. |
| `records/<reaction_id>.json` | Minimal per-reaction snapshot derived from the sanitized export contract `data/interim/g1_v2/export_contracts_v1/<shard>/<rid>.json` plus endpoint electronic metadata from the review case. |

## Snapshot fields (minimal P0 whitelist)

- graph rebuild (`load_endpoint_materials_from_export`): `maps`, `elements`,
  `r_coordinates`, `p_coordinates`, `atom_rows` (map/element/component table)
- context/events: `aromatic_regions`, `hydrogen_partner_changes`, `components`
- endpoints charge/multiplicity: `endpoint_electronic`
  (source: `reaction_cases_review_v1` case docs — endpoint-only metadata;
  Demo24 charges/multiplicities are cross-checked against the export totals
  at fixture build time)
- binding: `export_sha256` (sha256 of the source export-contract JSON),
  `source_export_path`, `reaction_smiles` (from the candidate CSV),
  `mapping_provenance` (`truth_assisted_p1`)

Full export documents, review cases, and the `data/` tree are deliberately
**not** committed.

## RDKit pin

`manifest.json.rdkit_version = "2026.03.6"` — the repo reference conda env
(`pes2ts`). The evidence JSON was built under `2026.03.3`
(`manifest.json.evidence_rdkit_version`); the P0 graph features are stable
across the two pins (verified by todos 7/9 and re-verified by the golden test).
The golden test `pytest.fail`s when the running RDKit differs from the pin.

## Regeneration

Requires the local data tree + the tracked evidence JSON (not needed to *run*
the golden test — only to rebuild fixtures):

```bash
conda run -n pes2ts python /tmp/opencode/build_p0_fixtures.py
```

The generator verifies the candidate CSV sha256 against the evidence JSON
before copying, verifies each export `export_sha256` against the evidence
records, and refuses to write fixtures on any mismatch. After regeneration,
re-run:

```bash
conda run -n pes2ts python -m pytest -q tests/test_p0_demo24_golden.py
```

## Note on evidence `proposal` fields

The evidence JSON `records[].proposal` blocks (and the `PROPOSALS` table in
`outputs/pes_generation_strategy_design_v1/build_evidence.py`) are
**design-time hypotheses**, not selector outputs. The strategy selector
arrives in a later plan todo; the P0 golden test never treats proposals as
pipeline products.
