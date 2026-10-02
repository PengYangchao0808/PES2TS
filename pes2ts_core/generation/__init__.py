"""PES generation (PES2TS-G): plan a reaction into an auditable path.

This package is the umbrella for the G2 stage. Its target layout (ADR-0001,
整体开发与发布方案 v2 §5) is::

    generation/
      planning/    # graph selector / StrategyProposal / GenerationPlanV2 / compile
      assembly/    # deterministic endpoint assembly (shared, truth-free)
      execution/   # ExecutionBackend protocol + backends (XTB_PATH / ORCA / NEB)
      quality/     # per-frame recovery + target-path quality layers

Migration status (code sync, additive): only ``execution`` exists so far; it
holds the X0 seam only. ``planning``/``assembly``/``quality`` are introduced in
later sync tiers by relocating the existing :mod:`pes2ts_core.generation.planning`
planning modules and the :mod:`pes2ts_core.generation.execution.xtb_path` assembly/runner, with
re-export shims left at the old paths. No existing module is imported here, so
this package has zero effect on current behaviour.
"""

from __future__ import annotations

__all__: list[str] = []
