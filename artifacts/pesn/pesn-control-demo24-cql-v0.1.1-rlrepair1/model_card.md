# pesn-control-demo24-cql-v0.1.1-rlrepair1 — research control model card (draft)

- generated: 2026-10-07T16:47:52+00:00
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
{"environment_hash": "cfeea21a73fcb36ba627ac535913d085ca07f1e654d5001fc59e1f2bb8212737", "executed_action_sources": {"bounded_exploration": 51, "rule_zero": 207}, "n_imitation_total": 984, "q_proxy_opened_on_evidence": true, "residual_opened_on_evidence": true, "status": "research_only_first_round", "supported_action_sources": ["bounded_exploration", "rule_zero"], "training_mode": "cql", "transition_rejections": {}}
```

## Run summary
```json
{"effective_config_sha256": "be73dcb31ba30354f3286dfb1109f6d35374ffc68af12b80cd4cb12d266fe72c", "run_id": "control-20261007-be73dcb3-seed42-cql-r1", "selection_manifest_sha256": "f7d94cd091148a3279a124376b57b36c6eb885087f1e53577b46a704a48266cf", "training_mode": "cql", "transitions_digest": "7543d2037db4668d540a73c96ae77726f06d2a679ba7c6a3a4b288cfb6d9e7a5"}
```

## Capabilities
```json
{"c_full": "disabled", "coordinate_activity": "disabled", "encoder": "enabled", "p_seed": "disabled", "path_proximity": "disabled", "production_control": "disabled", "progress": "disabled", "q_move": "disabled", "q_proxy": "research_only", "q_stop": "disabled", "residual_head": "research_only", "teacher_direction": "disabled"}
```

No claim is made about TS success, refinement savings, or transfer
beyond the frozen environment.
