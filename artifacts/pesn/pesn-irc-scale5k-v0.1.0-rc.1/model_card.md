# pesn-irc-scale5k-v0.1.0-rc.1 — research model card (draft)

- generated: 2026-10-08T07:47:53+00:00
- research package: predictions are IRC-teacher research outputs only;
  STOP/MOVE/Q heads and production control are disabled by contract.
- data provenance: B3LYP-D3/TZVP IRC trajectories (quarantined source,
  audited scope-checked bridge reads); truth-assisted P1 atom mapping;
  endpoint graphs from the G1 authority. Not end-to-end truth-free.

## Frozen metrics (initial run)
```json
{"baseline_constant":{"activity_f1":NaN,"activity_pr_auc":NaN,"direction_cosine_mean":NaN,"direction_frame_cosine_mean":NaN,"localization_top1_mean_d":0.018943611659109592,"localization_top3_mean_d":0.018943611659109592,"mae_d":0.51292226825219,"mae_s":0.51292226825219,"n_activity_labels":0,"n_pairs":1529066,"n_reactions_direction":0,"n_reactions_localization":250,"n_reactions_pairs":250,"n_reactions_s":250,"n_valid_frames":0,"pair_concordance":0.0},"baseline_rule":{"activity_f1":NaN,"activity_pr_auc":NaN,"direction_cosine_mean":NaN,"direction_frame_cosine_mean":NaN,"localization_top1_mean_d":0.13747172912955283,"localization_top3_mean_d":0.1225013230368495,"mae_d":0.20767713622713474,"mae_s":0.2187572359230797,"n_activity_labels":0,"n_pairs":1529066,"n_reactions_direction":0,"n_reactions_localization":250,"n_reactions_pairs":250,"n_reactions_s":250,"n_valid_frames":0,"pair_concordance":0.49537328730483177},"eval_name":"train_dev@g3p1-20261008-1e5b6828-seed42-r1","model":{"activity_f1":0.6047309675516751,"activity_positive_rate":0.36759059982261477,"activity_pr_auc":0.6256094342596168,"activity_pr_auc_reaction_mean":0.6555355524728954,"direction_cosine_mean":0.42436317300214293,"direction_frame_cosine_mean":0.42496167881299235,"localization_top1_mean_d":0.09973985496908426,"localization_top3_mean_d":0.08613224391639232,"mae_d":0.14639543698659827,"mae_s":0.15407132239123766,"n_activity_labels":161231,"n_pairs":1529066,"n_reactions_direction":250,"n_reactions_localization":250,"n_reactions_pairs":250,"n_reactions_s":250,"n_valid_frames":28150,"pair_concordance":0.8891792152445752},"n_reactions":250,"progress_mae_bootstrap":{"ci95_high":0.16635791796587948,"ci95_low":0.14436437732119117,"mean":0.15407132239123766,"n":250},"schema_version":"pesn_benchmark_report_v1"}
```

## Run summary
```json
{"best_epoch":0.160407,"n_parameters":117572,"run_id":"g3p1-20261008-1e5b6828-seed42-r1","status":"completed"}
```

## Capabilities
```json
{"c_full":"disabled","coordinate_activity":"research_only","encoder":"enabled","p_seed":"disabled","path_proximity":"enabled","production_control":"disabled","progress":"enabled","q_move":"disabled","q_stop":"disabled","teacher_direction":"disabled"}
```

No claim is made about refinement success, off-path transfer, or QC cost.
