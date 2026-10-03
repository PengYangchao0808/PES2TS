# scripts/archive — experiment-layer code retired by ADR-0002 X4′-B

Nothing here is imported by `pes2ts_core/`, the default CLI, or the default
test suite.  Files are archived (not deleted) so historical Demo24/round-2/3
experiment receipts remain reproducible against the code that produced them.

## Why these modules left `pes2ts_core/`

| Archived module | Decision | Missing ACP capability |
| --- | --- | --- |
| `continuation_backend.py` | **Retired** (option a) | ACP has no constrained-continuation single-point / constrained-optimize workflow (`cccp` `constrained_optimize` + constraint objects). No truthful minimal mapping exists on current ACP primitives. |
| `gradient_backend_cccp.py` | Historical copy of the pre-X4′-B cccp in-process gradient backend | Replaced in `pes2ts_core` by the ACP `OrcaGradient` CLI transport (`orca_gradient_request.py` + `orca_gradient_transport.py` + rewritten `gradient_backend.py`). |
| `seed_validation.py` | **Retired** (option a) | Beyond the gradient stage, the harness drives in-process cccp `ORCAInterface.transition_state_opt` / `.frequency` / `.irc` / `.optimize` and parses raw ORCA logs with `cccp.qc.interfaces.orca_ts.parse_ts_frequency_map` / `parse_ts_mode_vectors`. ACP exposes contract-level `BatchOptimize`/`irc` workflows (already used by `stage_cli.py`) but no drop-in equivalent for this experiment harness's call shape, budgets, and raw-log parsers; those parsers have no existing PES2TS/ACP replacement. |

`pes2ts_core/` is now free of in-process `cccp` imports (`grep -rn "cccp"
pes2ts_core/` is empty outside `__pycache__`).

## Archived experiment consumers

These one-off scripts imported the retired modules and are kept beside their
backends.  Imports were rewired to the archived siblings
(`continuation_backend`, `gradient_backend_cccp`, `seed_validation`); `ROOT`
now resolves to the repository root from `scripts/archive/`.

- `run_demo24_continuation_round2.py`, `run_demo24_round3.py`
- `report_demo24_continuation_round2.py`, `prepare_round3_origins.py`
- `probe_round3_gradient.py`, `probe_round3_curvature.py`,
  `probe_round3_local_guard.py`
- `validate_round2_candidates.py`, `validate_round3_candidates.py`
- `probe_acp_synchronized.py`, `probe_fixed_endpoint_scc.py` (direct cccp
  importers; experiment-layer capability probes)

`tests/test_acp_synchronized_values.py` stays in `tests/`: it does **not**
import the retired modules (it `importorskip`s `cccp`/`acp` directly and
tests ACP scan contracts); it remains skip-gated outside an ACP+cccp
environment and is unaffected by this retirement.

Do not import anything from this directory in shipped code.
