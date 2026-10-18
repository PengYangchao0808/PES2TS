# pesn-control-scale-cql-v0.1.0-rl50 — research control model card (draft)

- generated: 2026-10-08T22:13:37+00:00
- research package: bounded per-coordinate residual proposals and
  action-conditioned Q_proxy estimates only; Q_proxy is not Q_MOVE,
  not a success probability, and production control stays disabled.
- environment domain: E_native (frozen identity
  hash in the manifest); policies must be compared on the same
  environment and budget.
- data provenance: REAL G2/ACP macro-step transitions collected under
  the frozen E_native environment on train-qualified reactions; IRC
  teacher frames contributed zero transition records.

## Frozen metrics (control run)
```json
{"environment_hash": "cfeea21a73fcb36ba627ac535913d085ca07f1e654d5001fc59e1f2bb8212737", "executed_action_sources": {"bounded_exploration": 101, "rule_zero": 504}, "n_imitation_total": 825, "q_proxy_opened_on_evidence": true, "residual_opened_on_evidence": true, "status": "research_only_first_round", "supported_action_sources": ["bounded_exploration", "rule_zero"], "training_mode": "cql", "transition_rejections": {}}
```

## Run summary
```json
{"effective_config_sha256": "0f6cb8ffd04db971a73dad9e17fbc24adf83e0cb039578a0da8eee03aad0da79", "run_id": "control-20261008-0f6cb8ff-seed42-cql-r1", "selection_manifest_sha256": "e7c3a0e9af1065f66dbd05cd6ebc60f1fcf92e4decb8efc885c665d9a3964fe3", "training_mode": "cql", "transitions_digest": "70ede7edaa13089f2adb94208fe48c19d48111faa387bfed90f9f19191a848d2"}
```

## Capabilities
```json
{"c_full": "disabled", "coordinate_activity": "disabled", "encoder": "enabled", "p_seed": "disabled", "path_proximity": "disabled", "production_control": "disabled", "progress": "disabled", "q_move": "disabled", "q_proxy": "research_only", "q_stop": "disabled", "residual_head": "research_only", "teacher_direction": "disabled"}
```

No claim is made about TS success, refinement savings, or transfer
beyond the frozen environment.
