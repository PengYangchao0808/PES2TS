# Provenance — `tests/fixtures/g2/real_path/`

Real GFN2-xTB PATH outputs used as the authoritative parse fixtures for
`pes2ts_core/g2/xtb_output.py` (plan task 6) and the real-xTB smoke test
(plan task 17).

## Origin

- Source directory: `/tmp/opencode/xtb_path_test/` (read-only scratch output of a
  manual run; copied verbatim on 2026-09-28, byte-identical).
- Generator: xTB 6.7.1 (`xtb: 6.7.1 (edcfbbe)`), binary
  `/home/xieningke/xtb-dist/bin/xtb`, run on 2026-09-27 20:25 on this machine.
- System: 15-atom C7OH8 endpoints (`start.xyz` → `end.xyz`), single component,
  neutral closed shell.
- Command form: `xtb start.xyz --path end.xyz --input path.inp --gfn 2`
  (stdout+stderr merged into `xtb_path.log`); `path.inp` is the `$path` block
  with `nrun=1, npoint=25, anopt=10, kpush=0.003, kpull=-0.015, ppull=0.05,
  alp=1.2` (see `path.inp`).
- The two trial segment files (`xtbpath_0.xyz`, `xtbpath_1.xyz`,
  `xtbpath_2.xyz`) and other scratch outputs (`charges`, `wbo`, `xtbopt.log`,
  `xtbrestart`, `xtbtopo.mol`) are deliberately **not** copied: trial segments
  are never an authoritative parse source (`xtbpath.xyz` is), and the rest are
  unused by the parser.

## Pinned constants (locked by `tests/test_g2_output.py`)

| Constant | Value |
| --- | --- |
| `xtbpath.xyz` frame count | 22 |
| First frame `energy:` comment | `-0.000000` kcal/mol (file literal `0.000000000000`) |
| Last frame `energy:` comment | `-25.074971387031` kcal/mol |
| `xtbpath_ts.xyz` energy comment | `-20.902910236446` kcal/mol |
| Log `forward  barrier (kcal)` | `10.364` |
| Log `backward barrier (kcal)` | `35.439` |
| Log `reaction energy  (kcal)` | `-25.075` |
| Log `npath` token | absent (xTB 6.7.1 does not print it; parser yields `None`) |

Frame energies are **relative kcal/mol** from the xTB PATH metadynamics run
(first frame ≈ 0), never absolute total energies. The comment line format is
` energy: <value> xtb: 6.7.1 (edcfbbe)`.

## File checksums (at copy time)

```
6aeec563acdd76dc9df3c0b6c3e3cd7832b587c6bf1a8c3292ebea40a06c0b3c  end.xyz
5932fe20a55e06e9d79883f6b72d989fedab2921403770caf904a94bf8b082a1  path.inp
8e0bd0192fba0c26ddd1ea27ec7987fea751e9f21accb14f631a3f6733e932c4  start.xyz
33959b90db19c4547953a69e214969dbeae2eddd97a83b2014f317fe189836bb  xtb_path.log
83bca0ec91a3cc71993f5e6f98ac6047d1fcf6868e0edd9954d775b3243e11eb  xtbpath_ts.xyz
7bef73032ddce6f0c407fac45f182c2fbede0d35ed1dc9782c9e58546324caab  xtbpath.xyz
```

The repo `.gitignore` globally ignores `*.xyz`/`*.log`; two negation lines
(`!tests/fixtures/g2/**/*.xyz`, `!tests/fixtures/g2/**/*.log`) keep exactly
these fixtures versioned.
