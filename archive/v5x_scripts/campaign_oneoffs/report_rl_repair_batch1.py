"""Assemble the first-batch verification report (RL repair plan §14).

Reads the WP-0 audit, the r1_fix corrective runs, the frozen bundles, and
the fit registry, verifies every promotion gate LIVE (no hardcoded
verdicts), and writes ``reports/repair_batch1_report.json`` plus a short
human summary.  Read-only over all inputs.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

R1 = REPO_ROOT / "outputs/pesn/rl_repair_v1_20261008/r1_fix"
AUDIT = REPO_ROOT / "outputs/pesn/rl_repair_v1_20261008/audit"
REPORTS = REPO_ROOT / "outputs/pesn/rl_repair_v1_20261008/reports"
BUNDLES = {
    "bc": REPO_ROOT / "artifacts/pesn/pesn-control-demo24-bc-v0.1.1-rlrepair1",
    "cql": REPO_ROOT / "artifacts/pesn/pesn-control-demo24-cql-v0.1.1-rlrepair1",
}
ROUND0_BUNDLES = {
    "bc": REPO_ROOT / "artifacts/pesn/pesn-control-demo24-bc-v0.1.0",
    "cql": REPO_ROOT / "artifacts/pesn/pesn-control-demo24-v0.1.0",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check(name: str, ok: bool, detail) -> dict:
    return {"gate": name, "pass": bool(ok), "detail": detail}


def main() -> int:
    gates: list[dict] = []
    evidence: dict = {}

    audit = json.loads((AUDIT / "audit_round0.json").read_text())
    qualified = json.loads((AUDIT / "qualified_import.json").read_text())
    episodes = json.loads((AUDIT / "episode_index.json").read_text())
    split_ref = json.loads((AUDIT / "split_reference.json").read_text())
    gates.append(check(
        "wp0_evidence_hashed",
        audit["n_evidence_files"] > 0,
        {"n_evidence_files": audit["n_evidence_files"]},
    ))
    gates.append(check(
        "wp0_qualified_reusable_transitions",
        qualified["n_rejected"] == 0,
        {
            "n_documents": qualified["n_documents"],
            "n_qualified": qualified["n_qualified_for_corrective_reuse"],
            "n_rejected": qualified["n_rejected"],
        },
    ))
    gates.append(check(
        "wp0_episode_identity_unique",
        episodes["duplicate_episode_identities"] == 0
        and episodes["successor_links"]["incompatible"] == 0,
        episodes["successor_links"],
    ))
    gates.append(check(
        "wp0_split_group_integrity",
        split_ref["group_integrity_ok"]
        and split_ref["inherited_fit_dev_split"]["inherit"],
        {
            "fit": len(split_ref["inherited_fit_dev_split"]["fit_reactions"]),
            "dev": len(split_ref["inherited_fit_dev_split"]["dev_reactions"]),
        },
    ))

    import torch

    per_mode = {}
    for tag in ("bc", "cql"):
        summary = json.loads((R1 / f"train_summary_{tag}.json").read_text())
        run_id = summary["run_id"]
        run_doc = json.loads((R1 / "runs" / run_id / "run.json").read_text())
        checkpoints = {}
        for name in ("checkpoint_best.pt", "checkpoint_last.pt"):
            payload = torch.load(
                R1 / "runs" / run_id / name, map_location="cpu", weights_only=False
            )
            checkpoints[name] = str(payload.get("training_mode"))
        expected = summary["training_mode"]
        mode_consistent = (
            run_doc.get("training_mode") == expected
            and run_doc["config"]["control_train"]["training_mode"] == expected
            and all(value == expected for value in checkpoints.values())
        )
        gates.append(check(
            f"wp1_mode_consistency_{tag}",
            mode_consistent,
            {"summary": expected, "run.json": run_doc.get("training_mode"),
             "config": run_doc["config"]["control_train"]["training_mode"],
             "checkpoints": checkpoints, "run_id": run_id},
        ))
        per_mode[tag] = summary

    bc, cql = per_mode["bc"], per_mode["cql"]
    gates.append(check(
        "wp1_bc_q_params_changed_zero",
        bc["q_params_changed"] is False and bc["q_params_sha256_before"] == bc["q_params_sha256_after"],
        {"before": bc["q_params_sha256_before"][:16], "after": bc["q_params_sha256_after"][:16]},
    ))
    gates.append(check(
        "wp1_cql_q_params_updated",
        cql["q_params_changed"] is True,
        {"before": cql["q_params_sha256_before"][:16], "after": cql["q_params_sha256_after"][:16]},
    ))
    gates.append(check(
        "wp1_frozen_encoder_unchanged_both_modes",
        bc["encoder_params_changed"] is False and cql["encoder_params_changed"] is False,
        {"bc": bc["encoder_params_sha256_after"][:16], "cql": cql["encoder_params_sha256_after"][:16]},
    ))
    gates.append(check(
        "wp1_bootstrap_errors_zero",
        bc["n_bootstrap_inconsistent"] == 0 and cql["n_bootstrap_inconsistent"] == 0,
        {"bc": bc["n_bootstrap_inconsistent"], "cql": cql["n_bootstrap_inconsistent"]},
    ))
    gates.append(check(
        "wp1_distinct_control_groups",
        bc["effective_config_sha256"] != cql["effective_config_sha256"]
        and sha(R1 / "runs" / bc["run_id"] / "checkpoint_best.pt")
        != sha(R1 / "runs" / cql["run_id"] / "checkpoint_best.pt"),
        {"note": "distinct weights are necessary, not sufficient; parameter-change "
                 "evidence above is the primary proof"},
    ))
    gates.append(check(
        "wp1_imitation_counts_honest",
        bc["n_imitation_distinct_samples"] == 41 and bc["n_imitation_epoch_uses"] == 984
        and cql["n_imitation_distinct_samples"] == 41,
        {"distinct": bc["n_imitation_distinct_samples"],
         "epoch_uses": bc["n_imitation_epoch_uses"]},
    ))
    gates.append(check(
        "wp1_cql_repair_reproduces_round0_cql",
        sha(BUNDLES["cql"] / "weights.pt") == sha(ROUND0_BUNDLES["cql"] / "weights.pt"),
        {"note": "same seed/config/data => byte-identical CQL; confirms round0's "
                 "'bc' package was the same CQL training"},
    ))
    gates.append(check(
        "wp1_bc_bundle_q_disabled",
        json.loads((BUNDLES["bc"] / "model_bundle.json").read_text())["capabilities"]["q_proxy"] == "disabled"
        and json.loads((BUNDLES["cql"] / "model_bundle.json").read_text())["capabilities"]["q_proxy"] != "disabled",
        None,
    ))
    gates.append(check(
        "wp1_bc_bundle_weights_differ_from_round0_defect",
        sha(BUNDLES["bc"] / "weights.pt") != sha(ROUND0_BUNDLES["bc"] / "weights.pt"),
        None,
    ))

    registry_path = R1 / "action_support_registry_fit.json"
    registry = json.loads(registry_path.read_text())
    gates.append(check(
        "wp2_registry_fit_only",
        registry["built_from"]["n_fit_transitions"] == 201
        and len(registry["built_from"]["dev_reactions_excluded"]) == 4,
        {"n_fit": registry["built_from"]["n_fit_transitions"],
         "dev_excluded": registry["built_from"]["dev_reactions_excluded"],
         "calibration": registry["calibration"],
         "sha256": sha(registry_path)},
    ))

    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q",
         "tests/test_control_round.py", "tests/test_control_round_mode_repair.py",
         "tests/test_action_support.py"],
        capture_output=True, text=True, cwd=REPO_ROOT,
        env={"PYTHONPATH": "", "PATH": "/usr/bin:/bin"},
    )
    last = [line for line in result.stdout.splitlines() if line.strip()][-1] if result.stdout else "no output"
    gates.append(check("wp1_wp2_targeted_tests", result.returncode == 0, last))

    evidence = {
        "corrective_runs": {
            tag: {
                "run_id": summary["run_id"],
                "training_mode": summary["training_mode"],
                "best_epoch": summary["best_epoch"],
                "n_parameters_total": summary["n_parameters_total"],
                "n_parameters_frozen": summary["n_parameters_frozen"],
                "n_parameters_trainable": summary["n_parameters_trainable"],
                "optimizer_param_names_sha256": summary["optimizer_param_names_sha256"],
                "bootstrap_support_gate": "legacy_recorded_candidates (corrective batch; registry gate ships for subsequent rounds)",
            }
            for tag, summary in per_mode.items()
        },
        "bundles": {
            tag: {
                "model_id": json.loads((d / "model_bundle.json").read_text())["model_id"],
                "weights_sha256": sha(d / "weights.pt"),
                "capabilities": json.loads((d / "model_bundle.json").read_text())["capabilities"],
            }
            for tag, d in BUNDLES.items()
        },
    }

    all_pass = all(gate["pass"] for gate in gates)
    report = {
        "schema_version": "pesn_rl_repair_batch1_report_v1",
        "batch": "WP-0 + WP-1 (+WP-1 §4.3 corrective models) + WP-2 code",
        "all_gates_pass": all_pass,
        "gates": gates,
        "evidence": evidence,
        "environment": {
            "python": sys.executable,
            "torch": torch.__version__,
        },
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "repair_batch1_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    for gate in gates:
        mark = "PASS" if gate["pass"] else "FAIL"
        print(f"[{mark}] {gate['gate']}")
    print(f"\nall_gates_pass={all_pass} -> {REPORTS / 'repair_batch1_report.json'}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
