#!/usr/bin/env python3
"""V5.4.2 run-identity CLI (gate-0 advisory A3; ADR-0033 §2.6, plan §6).

Assembles the frozen 14-field run identity for one (candidate, run-seed)
pair via :func:`pes2ts_core.pesn.models.v542.identity.assemble_run_identity`
and prints the identity JSON.  Exit codes:

* 0 — identity assembled; the JSON record is printed (and written via
  ``--out`` when requested);
* 2 — REFUSED: any typed :class:`V542IdentityError` (missing/None field,
  unknown candidate, bad/out-of-set seed — only the frozen campaign
  seeds {17, 29, 43} are accepted, consistent with assembly/trainer —
  file drift, dirty tree under ``--require-clean``) AND any unexpected
  exception (printed as a typed ``V542_IDENTITY_UNEXPECTED`` refusal,
  never a traceback) exits non-zero WITHOUT printing an identity.

Input sources (all frozen; nothing is invented here):

* config: ``--config`` (default: the frozen V5.4.2 campaign config);
* runplan pin record: ``--runplan`` (default: the WP0 template
  ``config/campaigns/v542_templates/runplan_pin.template.json`` — the
  formal instance at ``outputs/pesn/v542_backbone_50k/plan/runplan.pin.json``
  supersedes it once frozen);
* budget policy record: ``--budget`` (default: the WP0 template
  ``config/campaigns/v542_templates/budget_policy.template.json`` — budget
  VALUES stay null until the same-env profile; the record digest binds the
  frozen null state);
* source manifest: ``--source-manifest`` (optional; when omitted the
  digest is derived from the live source-tree fingerprint and equals the
  future manifest-file digest).

The printed record is ``{"schema": ..., "identity": {...}, "payloads":
{...}}`` — ``identity`` carries the 14 frozen fields + ``code_commit``;
``payloads`` carries each builder's canonical payload for later SHA
reconstruction (config/plan/oracle digests are file-bytes digests and their
reconstruction records name the pinned paths).

CLI::

    python scripts/agent_ops/v542_identity.py --candidate A1 --run-seed 17 \
        [--out identity.json] [--require-clean] \
        [--config CONFIG] [--runplan RUNPLAN] [--budget BUDGET] \
        [--source-manifest MANIFEST]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pes2ts_core.pesn.models.v542.assembly import (  # noqa: E402
    ANCHOR_SEEDS_V542,
)
from pes2ts_core.pesn.models.v542.identity import (  # noqa: E402
    DEFAULT_CAMPAIGN_CONFIG,
    RUN_IDENTITY_SCHEMA,
    V542IdentityError,
    assemble_run_identity,
    backbone_spec_sha256,
    budget_sha256,
    common_protocol_sha256,
    config_sha256,
    environment_sha256,
    init_mapping_sha256,
    oracle_design_source_sha256,
    plan_sha256,
    process_head_anchor_sha256,
    runplan_sha256,
    source_manifest_sha256,
    source_tree_fingerprint,
    statistics_contract_sha256,
)

DEFAULT_RUNPLAN = "config/campaigns/v542_templates/runplan_pin.template.json"
DEFAULT_BUDGET = "config/campaigns/v542_templates/budget_policy.template.json"

EXIT_PASS, EXIT_REFUSED = 0, 2


def _load_json(path: Path) -> Mapping[str, Any]:
    if not path.is_file():
        raise V542IdentityError(
            "V542_IDENTITY_FIELD_MISSING",
            f"record file not found: {path}")
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise V542IdentityError(
            "V542_IDENTITY_FIELD_MISSING",
            f"record file unreadable: {path}: {exc}") from exc
    if not isinstance(record, Mapping):
        raise V542IdentityError(
            "V542_IDENTITY_FIELD_MISSING",
            f"record file is not a JSON object: {path}")
    return record


def build_identity_record(
    candidate_id: str,
    run_seed: int,
    *,
    config_path: Path,
    runplan_path: Path,
    budget_path: Path,
    source_manifest_path: Optional[Path],
    require_clean: bool,
    root: Path = PROJECT_ROOT,
) -> dict[str, Any]:
    """Full identity record with per-field reconstruction payloads.

    ``run_seed`` must be one of the frozen campaign seeds
    ``ANCHOR_SEEDS_V542`` = {17, 29, 43} (the same set the assembler and
    the trainer enforce): an out-of-set seed is the typed refusal
    ``V542_IDENTITY_SEED`` — this CLI never invents an identity for a
    seed the campaign cannot run (oracle gate-3 C5).
    """
    import yaml

    if not isinstance(run_seed, int) or isinstance(run_seed, bool) \
            or run_seed not in ANCHOR_SEEDS_V542:
        raise V542IdentityError(
            "V542_IDENTITY_SEED",
            "run_seed must be one of the frozen campaign seeds "
            f"{list(ANCHOR_SEEDS_V542)} (config "
            "process_head_anchor.seeds; assembly/trainer enforce the same "
            f"set), got {run_seed!r}")
    if not config_path.is_file():
        raise V542IdentityError(
            "V542_IDENTITY_FILE_MISSING", f"config not found: {config_path}")
    try:
        cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 - typed conversion
        raise V542IdentityError(
            "V542_IDENTITY_CONFIG", f"config unreadable: {exc}") from exc
    if not isinstance(cfg, Mapping):
        raise V542IdentityError(
            "V542_IDENTITY_CONFIG", "config is not a mapping")
    runplan_record = _load_json(runplan_path)
    budget_block = _load_json(budget_path)

    identity = assemble_run_identity(
        cfg,
        candidate_id=candidate_id,
        run_seed=run_seed,
        runplan_record=runplan_record,
        budget_block=budget_block,
        root=root,
        config_path=config_path,
        source_manifest_path=source_manifest_path,
        require_clean=require_clean,
    )

    # Reconstruction payloads for every builder (persisted with the digest
    # so each SHA can be re-derived later).
    fingerprint_sha, fingerprint = source_tree_fingerprint(root)
    config_sha, config_record = config_sha256(config_path, root=root)
    common_sha, common_record = common_protocol_sha256(cfg)
    environment_digest, environment_record = environment_sha256()
    backbone_sha, backbone_record = backbone_spec_sha256(cfg, candidate_id)
    init_mapping_digest, init_mapping_record = init_mapping_sha256(cfg)
    anchor_sha, anchor_record = process_head_anchor_sha256(run_seed)
    statistics_sha, statistics_record = statistics_contract_sha256(cfg)
    plan_digest, plan_record = plan_sha256(root=root)
    oracle_sha, oracle_record = oracle_design_source_sha256(root=root)
    runplan_digest, runplan_record_payload = runplan_sha256(runplan_record)
    budget_digest, budget_record = budget_sha256(budget_block)
    if source_manifest_path is not None:
        manifest_sha, manifest_record = source_manifest_sha256(
            source_manifest_path, root=root)
    else:
        manifest_sha, manifest_record = fingerprint_sha, {
            "recipe": "derived_from_source_tree_fingerprint",
            "note": ("the frozen manifest file is defined as the canonical "
                     "bytes of this fingerprint payload; its file-bytes "
                     "digest equals fingerprint_sha"),
            "fingerprint": fingerprint,
        }

    # cross-check: every assembled field equals the builder it came from
    expected_pairs = {
        "source_manifest_sha256": (identity["source_manifest_sha256"],
                                   manifest_sha),
        "config_sha256": (identity["config_sha256"], config_sha),
        "common_protocol_sha256": (identity["common_protocol_sha256"],
                                   common_sha),
        "environment_sha256": (identity["environment_sha256"],
                               environment_digest),
        "backbone_spec_sha256": (identity["backbone_spec_sha256"],
                                 backbone_sha),
        "init_mapping_sha256": (identity["init_mapping_sha256"],
                                init_mapping_digest),
        "process_head_anchor_sha256": (identity["process_head_anchor_sha256"],
                                       anchor_sha),
        "statistics_contract_sha256": (identity["statistics_contract_sha256"],
                                       statistics_sha),
        "plan_sha256": (identity["plan_sha256"], plan_digest),
        "oracle_design_source_sha256": (identity["oracle_design_source_sha256"],
                                        oracle_sha),
        "runplan_sha256": (identity["runplan_sha256"], runplan_digest),
        "budget_policy_sha256": (identity["budget_policy_sha256"],
                                 budget_digest),
    }
    for field, (left, right) in expected_pairs.items():
        if left != right:  # pragma: no cover - construction guard
            raise V542IdentityError(
                "V542_IDENTITY_FIELD_CONFLICT",
                f"assembled {field} != builder digest ({left} != {right})")

    return {
        "schema": RUN_IDENTITY_SCHEMA,
        "candidate_id": candidate_id,
        "run_seed": int(run_seed),
        "identity": identity,
        "payloads": {
            "source_tree_fingerprint": manifest_record,
            "config": {"sha256": config_sha, "record": config_record},
            "common_protocol": {"sha256": common_sha, "record": common_record},
            "environment": {"sha256": environment_digest,
                            "record": environment_record},
            "backbone_spec": {"sha256": backbone_sha,
                              "record": backbone_record},
            "init_mapping": {"sha256": init_mapping_digest,
                             "record": init_mapping_record},
            "process_head_anchor": {"sha256": anchor_sha,
                                    "record": anchor_record},
            "statistics_contract": {"sha256": statistics_sha,
                                    "record": statistics_record},
            "plan": {"sha256": plan_digest, "record": plan_record},
            "oracle_design_source": {"sha256": oracle_sha,
                                     "record": oracle_record},
            "runplan": {"sha256": runplan_digest,
                        "record": runplan_record_payload},
            "budget_policy": {"sha256": budget_digest,
                              "record": budget_record},
        },
    }


def _main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Assemble the frozen V5.4.2 run identity (14 required "
                    "fields + record-only code_commit) for one "
                    "(candidate, run-seed); REFUSES (exit 2) on any typed "
                    "identity error")
    parser.add_argument("--candidate", required=True, metavar="ID",
                        help="candidate id (A0, A1, A2, A3, A4, A5, A1-C, "
                             "A5-C)")
    parser.add_argument("--run-seed", required=True, type=int, metavar="N",
                        help="run seed (frozen run seeds: 17, 29, 43)")
    parser.add_argument("--out", default=None, metavar="PATH",
                        help="also write the identity record JSON to PATH "
                             "(always printed to stdout)")
    parser.add_argument("--require-clean", action="store_true",
                        help="refuse (exit 2) when the frozen fingerprint "
                             "scope is dirty (pes2ts_core, "
                             "config/campaigns/v542_templates, scripts/campaigns, "
                             "scripts/agent_ops, scripts/infra, the campaign "
                             "config and the requirements*.txt files)")
    parser.add_argument("--config", default=DEFAULT_CAMPAIGN_CONFIG,
                        metavar="PATH",
                        help="campaign config (default: %(default)s)")
    parser.add_argument("--runplan", default=DEFAULT_RUNPLAN, metavar="PATH",
                        help="frozen runplan pin record JSON (default: the "
                             "WP0 template; the formal pin instance "
                             "supersedes it)")
    parser.add_argument("--budget", default=DEFAULT_BUDGET, metavar="PATH",
                        help="budget policy record JSON (default: the WP0 "
                             "template; null budget values stay null)")
    parser.add_argument("--source-manifest", default=None, metavar="PATH",
                        help="frozen source manifest file (default: derived "
                             "from the live source-tree fingerprint)")
    args = parser.parse_args(argv)

    def _resolve(path_str: str) -> Path:
        path = Path(path_str)
        return path if path.is_absolute() else PROJECT_ROOT / path

    try:
        record = build_identity_record(
            args.candidate, args.run_seed,
            config_path=_resolve(args.config),
            runplan_path=_resolve(args.runplan),
            budget_path=_resolve(args.budget),
            source_manifest_path=(_resolve(args.source_manifest)
                                  if args.source_manifest else None),
            require_clean=bool(args.require_clean),
        )
        text = json.dumps(record, indent=2, sort_keys=True,
                          ensure_ascii=False)
        print(text)
        if args.out:
            out_path = Path(args.out)
            if out_path.is_dir():
                print(f"REFUSED: --out is a directory: {out_path}",
                      file=sys.stderr)
                return EXIT_REFUSED
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(text + "\n", encoding="utf-8")
    except V542IdentityError as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return EXIT_REFUSED
    except Exception as exc:  # noqa: BLE001 - typed CLI catch-all (C5)
        # An UNEXPECTED failure must still be a typed refusal (exit 2,
        # no traceback): the CLI either prints the identity JSON and
        # exits 0, or prints one REFUSED line and exits 2 — never a
        # stack trace on stdout/stderr.
        print(f"REFUSED: V542_IDENTITY_UNEXPECTED: "
              f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return EXIT_REFUSED
    return EXIT_PASS


def main(argv: Optional[Sequence[str]] = None) -> int:
    return _main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
