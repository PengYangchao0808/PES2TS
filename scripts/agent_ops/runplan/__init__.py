"""E0 run-plan builder/verifier package (WP-L2; supervisor-owned planning artifacts).

* ``build_r3_runplan`` emits ``run_plan.jsonl`` for campaign
  ``PES2TS_30H_R3_V52_V53``: R3 = 66 runs (30h plan section 4.2 matrix) plus
  V5.2 = 33 runs (section 5.5) -- 99 planned lines + one plan header carrying
  the PD-3 conditional note.
* ``verify`` re-checks uniqueness, exact counts, substage counts, reuse
  targets, legal seeds/folds and cross-arm duplicates; exit 0 iff all pass.
"""
