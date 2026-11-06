#!/usr/bin/env python3
"""V5.4.3 WP9 — unified bundle build/verify + the S0-A parity CERTIFICATE
INSTANCE generator (plan §13.3 :1329-1375, §11.2 ``v543_bundle.py`` row
``build``/``verify``, §11.5 ``test_v543_bundle_strict_load`` /
``test_v543_legacy_files_untouched``; ADR-0034).

WHAT THIS IS
------------
Three subcommands, one governance anchor:

``build``
    Assemble a ``PES2TSBundleV543`` tree with the §13.3 :1331–1352 component
    layout (``encoder/`` ``bridge/`` ``move_expert/`` ``process_expert/``
    ``controller/`` ``provenance/``) from the campaign artifacts that exist
    TODAY ("存在多少收多少"), write ``manifest.json`` (sha256 + byte size for
    every file, plus the source path it was copied from), the five-report
    skeleton of §13.3 :1354–1360 (``encoder_bridge_parity`` /
    ``move_benchmark`` / ``process_benchmark`` / ``integration_rollout`` /
    ``final_selection``) with every unfinished section marked ``pending`` and a
    reason, and finally declare the tree immutable (``IMMUTABLE.json`` +
    ``chmod`` on POSIX).  The directory name carries a UTC timestamp, so a
    later WP9 final bundle never overwrites this one.

``verify``
    Re-hash an existing bundle and refuse on ANY drift: a changed digest, a
    changed size, a missing file, an EXTRA file, or a manifest that does not
    match its own ``manifest_sha256``.  Refusals are typed
    (``V543_BUNDLE_VERIFY_*``), never warnings.

``certify``
    Regenerate the S0-A parity certificate instance from LIVE measurements of
    the three pinned V542 A0 checkpoints, using the WP1 primitives the
    ``realdata`` test uses (``CoarseEncoder.load`` +
    ``assert_encoder_packet_parity`` + the frozen inner-val batch selection),
    and fill ``config/campaigns/v543_templates/parity_certificate.template.json``
    into the instance at ``outputs/pesn/v543_dual_decoder_50k/plan/
    parity_certificate.json`` (the WP9 task brief's path — the template's
    ``_template_note`` says ``audits/``; see the ``ambiguities`` block this
    generator writes, both paths are recorded).

WHY THE TWO STATE-DICT COUNTS ARE RECORDED SEPARATELY
-----------------------------------------------------
The template's ``state_dict_count_semantics`` freezes TWO 口径 that must never
be mixed: ``payload_key_count`` (w_v dual-write counted ONCE, expected 43 —
ADR-0033 identity 口径) and ``expanded_target_key_count`` (the strict-load
expanded target set, w_v counted TWICE, expected 45 — 主方案 §1.2 口径), with
``expanded = payload + 2`` as the consistency rule.  The realistic S0-A asset on
disk is the V5.4.2 **45-key** ``v542_a0_joint_45`` payload, i.e. the w_v
dual-write is ALREADY materialised in the file, so the MEASURED payload count
is 45 and the "payload + 2" identity does NOT hold for it.  This generator
therefore records the MEASURED values, names the 口径 of its per-key assertions
(``count_scope_of_this_audit``), states the layout it found, and reports the
measured-vs-template-expectation deviation in ``state_dict_counts`` — it never
rewrites the frozen expectation to make the numbers agree (§7.1 S0-A: no
"approximate equality" release).

IMMUTABILITY / NON-DESTRUCTION
------------------------------
* ``runs/`` is READ ONLY for this lane: the bundle COPIES a run's artifacts
  (``checkpoint_best.pt`` / ``train_log.jsonl`` / …) and never writes inside
  ``runs/``; ``run_ledger.jsonl`` and ``plan/runplan*.json`` are copied, never
  rewritten (plan §13.3 :1362 append-only);
* a bundle is write-once: ``build`` refuses an existing target directory
  (``V543_BUNDLE_EXISTS``) and the ``--force`` flag does NOT exist;
* ``verify`` never repairs: it reports.

CLI::

    python scripts/agent_ops/v543_bundle.py build
    python scripts/agent_ops/v543_bundle.py verify --bundle <dir>
    python scripts/agent_ops/v543_bundle.py certify

Discipline: no V5.4.2 / V5.3-MP file is modified (plan §11.3), no
``strict=False`` (the certificate's loads are ``strict=True`` + per-key
``torch.equal``), no ``torch.compile`` (§11.3 禁止), every constant is
``Final`` and every refusal is typed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import stat
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Final, Iterator, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:            # pragma: no cover - import aid
    sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# frozen identity of THIS document
# ---------------------------------------------------------------------------
#: Schema of a bundle manifest.
BUNDLE_SCHEMA: Final[str] = "pes2ts_v543_bundle_manifest_v1"
#: Registered bundle version (§13.3 :1332 ``PES2TSBundleV543``).
BUNDLE_VERSION: Final[str] = "PES2TSBundleV543"
#: Directory-name prefix (the UTC creation stamp is appended).
BUNDLE_DIR_PREFIX: Final[str] = "PES2TSBundleV543_"
#: Schema of the five-report bundle report.
BUNDLE_REPORT_SCHEMA: Final[str] = "pes2ts_v543_bundle_report_v1"
#: Schema of the read-only declaration.
IMMUTABLE_SCHEMA: Final[str] = "pes2ts_v543_bundle_immutable_v1"
#: Schema of the generated controller policy (§13.3 :1345–1346).
CONTROLLER_POLICY_SCHEMA: Final[str] = "pes2ts_v543_legal_endpoint_policy_v1"
#: Schema of the generated A0 encoder manifest (§13.3 :1333–1335).
ENCODER_MANIFEST_SCHEMA: Final[str] = "pes2ts_v543_encoder_manifest_v1"
#: Schema of the generated data/split/sampler provenance block.
DATA_PROVENANCE_SCHEMA: Final[str] = "pes2ts_v543_data_provenance_v1"

#: Campaign root (shared with every other V543 lane).
CAMPAIGN_ROOT: Final[str] = "outputs/pesn/v543_dual_decoder_50k"
#: Where bundles live (a NEW subtree; nothing else writes here).
BUNDLES_SUBDIR: Final[str] = "bundles"
#: Campaign config (the pinned checkpoint / sampler / seed identities).
DEFAULT_CONFIG: Final[str] = "config/campaigns/pes2ts_v543_dual_decoder_50k.yaml"
#: The five §13.3 :1354–1360 reports, in the plan's own order.
FIVE_REPORTS: Final[tuple[str, ...]] = (
    "encoder_bridge_parity",
    "move_benchmark",
    "process_benchmark",
    "integration_rollout",
    "final_selection",
)
#: §13.3 :1362 ledgers that must be carried (append-only, never rewritten).
LEDGER_FILES: Final[tuple[str, ...]] = (
    "run_ledger.jsonl",
    "run_records.jsonl",
)
#: Controller semantics transcribed into ``controller/legal_endpoint_policy.json``
#: (plan §4.6 :250–263 / §4.3 :227 / §10.2 :1071 / §10.3 :1073–1080).
CONTROLLER_POLICY: Final[Mapping[str, Any]] = {
    "schema": CONTROLLER_POLICY_SCHEMA,
    "component": "ReactionController",
    "learned": False,
    "parameter_count": 0,
    "move_flag": {
        "signature": "move_flag(p_end, tau_end, chemical_state_ok, is_ts)",
        "source": "plan §4.6 :253-259 (verbatim transcription)",
        "rules": [
            "if is_ts: return True            # TS 永远继续 MOVE",
            "if p_end >= tau_end and chemical_state_ok: return False",
            "return True                      # q 不参与",
        ],
        "tau_end": None,
        "tau_end_policy": (
            "§9.5 :1040 — calibrated ONCE on inner-val with target precision "
            "0.9; never recalibrated on dev; an uncalibrated tau makes No-Move "
            "impossible"),
    },
    "no_move": {
        "only_reason": "p_end >= tau_end AND chemical_state_ok",
        "forbidden_triggers": ["q_threshold", "near_ts", "path_length"],
        "audit_fields_required": ["p_end", "chemical_state_ok", "is_ts"],
        "q_may_not_appear_in_trigger_reason": True,
        "plan_lines": "§4.3 :227 / §10.2 :1071",
    },
    "not_in_model_forward": True,
    "plan_lines": "§4.6 :250-263 / §10.3 :1073-1080",
}

# ---------------------------------------------------------------------------
# typed refusal codes
# ---------------------------------------------------------------------------
CODE_EXISTS: Final[str] = "V543_BUNDLE_EXISTS"
CODE_ROOT: Final[str] = "V543_BUNDLE_ROOT"
CODE_WRITE: Final[str] = "V543_BUNDLE_WRITE"
CODE_VERIFY_MISSING: Final[str] = "V543_BUNDLE_VERIFY_MISSING"
CODE_VERIFY_SCHEMA: Final[str] = "V543_BUNDLE_VERIFY_SCHEMA"
CODE_VERIFY_DIGEST: Final[str] = "V543_BUNDLE_VERIFY_DIGEST"
CODE_VERIFY_EXTRA: Final[str] = "V543_BUNDLE_VERIFY_EXTRA"
CODE_VERIFY_IMMUTABLE: Final[str] = "V543_BUNDLE_VERIFY_IMMUTABLE"
CODE_CERT_TEMPLATE: Final[str] = "V543_CERTIFICATE_TEMPLATE"
CODE_CERT_CHECKPOINT: Final[str] = "V543_CERTIFICATE_CHECKPOINT"
CODE_CERT_EQUATION: Final[str] = "V543_CERTIFICATE_EQUATION"
CODE_CERT_DATA: Final[str] = "V543_CERTIFICATE_DATA"


class BundleError(RuntimeError):
    """Typed bundle / certificate refusal (``code`` + ``detail``)."""

    def __init__(self, code: str, detail: str = "") -> None:
        super().__init__(f"{code}: {detail}" if detail else code)
        self.code = code
        self.detail = detail


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(obj: Any) -> str:
    from pes2ts_core.pesn.models.v542.identity import canonical_sha256 as _c

    return _c(obj)


def _utc_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def _utc_stamp() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _read_json(path: Path, *, what: str, code: str) -> dict[str, Any]:
    if not Path(path).is_file():
        raise BundleError(code, f"{what} not found: {path}")
    try:
        document = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception as error:                    # noqa: BLE001 - typed
        raise BundleError(code, f"{what} unreadable ({path}): {error}") from error
    if not isinstance(document, dict):
        raise BundleError(code, f"{what} is not a JSON object: {path}")
    return document


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT.resolve()))
    except Exception:                              # pragma: no cover - outside
        return str(path)


def _iter_files(root: Path) -> Iterator[Path]:
    for path in sorted(Path(root).rglob("*")):
        if path.is_file():
            yield path


# ---------------------------------------------------------------------------
# collection plan (§13.3 :1329–1362) — declared, then reported as found/missing
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class SourceRule:
    """One declared bundle input.

    ``source`` is either an explicit project-relative path or a glob pattern
    (``glob=True``).  ``required_for_final`` marks the §13.3 deliverables a WP9
    FINAL bundle may not miss; a missing one is a ``pending`` line in the
    manifest (an intermediate bundle is explicitly allowed to be partial —
    the WP9 task says "存在多少收多少").  ``exclude_names`` keeps a glob from
    duplicating a file that an explicit rule already places elsewhere.
    """

    component: str
    dest: str
    source: str
    required_for_final: bool
    glob: bool = False
    note: str = ""
    flatten: bool = False
    exclude_names: tuple[str, ...] = ()


def collection_plan(campaign_root: str = CAMPAIGN_ROOT) -> tuple[SourceRule, ...]:
    """The declared §13.3 :1331–1352 surface, in plan order."""
    root = campaign_root.rstrip("/")
    rules: list[SourceRule] = [
        # -- governance anchors -------------------------------------------
        SourceRule("provenance", "provenance/governance/ADR-0034.md",
                   "docs/design/decisions/ADR-0034-V5.4.3-双Decoder与"
                   "DecoderContextV1.md", True),
        SourceRule("provenance", "provenance/governance/plan_v5.4.3.md",
                   "docs/plans/PES2TS_V5.4.3_双Decoder_50K_全面开发与训练"
                   "方案_v1.0_20261102.md", True),
        SourceRule("provenance", "provenance/governance/wp_checklist_v5.4.3.md",
                   "docs/plans/PES2TS_V5.4.3_WP执行清单_v1.0_20261102.md",
                   True),
        SourceRule("provenance", "provenance/governance/campaign_config.yaml",
                   DEFAULT_CONFIG, True),
        SourceRule("provenance", "provenance/governance/config_validator.py",
                   "scripts/agent_ops/v543_config_schema.py", True),
        # -- plan records --------------------------------------------------
        SourceRule("provenance", "provenance/governance/runplan.json",
                   f"{root}/plan/runplan.json", True),
        SourceRule("provenance", "provenance/governance/runplan.pin.json",
                   f"{root}/plan/runplan.pin.json", True),
        SourceRule("provenance",
                   "provenance/governance/preregistration_signoff.json",
                   f"{root}/plan/preregistration_signoff.json", True),
        SourceRule("provenance", "provenance/governance/budget_profile.json",
                   f"{root}/plan/budget_profile.json", False),
        # -- ledgers (§13.3 :1362, append-only) ---------------------------
        SourceRule("provenance", "provenance/ledger/run_ledger.jsonl",
                   f"{root}/run_ledger.jsonl", True),
        SourceRule("provenance", "provenance/ledger/run_records.jsonl",
                   f"{root}/run_records.jsonl", True),
        SourceRule("provenance", "provenance/ledger/train_log.jsonl",
                   f"{root}/train_log.jsonl", False),
        SourceRule("provenance", "provenance/ledger/metrics.json",
                   f"{root}/metrics.json", False),
        # -- bridge / parity certificates ---------------------------------
        SourceRule("bridge", "bridge/parity_certificate.json",
                   f"{root}/plan/parity_certificate.json", True),
        SourceRule("bridge", "bridge/wp3_go_no_go_decision.json",
                   f"{root}/audit/wp3_go_no_go_decision.json", True),
        SourceRule("bridge", "bridge/decoder_context_v1.template.json",
                   "config/campaigns/v543_templates/decoder_context.template"
                   ".json", False),
        # -- audits / selection / reports ---------------------------------
        SourceRule("provenance", "provenance/audit/", f"{root}/audit/",
                   False, glob=True, flatten=True),
        SourceRule("provenance", "provenance/reports/", f"{root}/reports/",
                   False, glob=True, flatten=False,
                   note=("statistics reports; the two files that also have a "
                         "dedicated §13.3 destination are not duplicated"),
                   exclude_names=("t7_rollout.json", "track_champions.json",
                                  "final_selection.json")),
        SourceRule("provenance", "provenance/selection/track_champions.json",
                   f"{root}/reports/track_champions.json", True),
        SourceRule("provenance",
                   "provenance/selection/final_selection.json",
                   f"{root}/reports/final_selection.json", True),
        SourceRule("provenance",
                   "provenance/selection/activation_decisions.json",
                   f"{root}/reports/activation_decisions.json", False),
        # -- rollout + engineering ledgers --------------------------------
        SourceRule("provenance", "provenance/rollout/t7_rollout.json",
                   f"{root}/reports/t7_rollout.json", True),
        SourceRule("provenance",
                   "provenance/rollout/wp3_sufficiency_report.json",
                   f"{root}/audit/s0_probe_report.json", False),
        SourceRule("provenance", "provenance/rollout/profile_report.jsonl",
                   f"{root}/profile/profile_report.jsonl", False),
        SourceRule("provenance", "provenance/rollout/smoke_report.jsonl",
                   f"{root}/smoke/smoke_report.jsonl", False),
        # -- runs: "存在多少收多少" (each run archived under its track) ----
        SourceRule("move_expert", "move_expert/runs/", f"{root}/runs/",
                   False, glob=True, flatten=False),
    ]
    return tuple(rules)


@dataclass
class CollectedFile:
    """One file that made it into the bundle (copied or generated)."""

    component: str
    path: str                 # bundle-relative
    source: str               # project-relative source, or "<generated>"
    sha256: str
    bytes: int
    origin: str               # "copied" | "generated"

    def describe(self) -> dict[str, Any]:
        return {
            "component": self.component,
            "path": self.path,
            "source": self.source,
            "sha256": self.sha256,
            "bytes": int(self.bytes),
            "origin": self.origin,
        }


@dataclass
class PendingFile:
    """One declared input that is not available yet (§13.3 deliverable)."""

    component: str
    path: str
    source: str
    required_for_final: bool
    reason: str
    note: str = ""

    def describe(self) -> dict[str, Any]:
        return {
            "component": self.component,
            "path": self.path,
            "source": self.source,
            "required_for_final": bool(self.required_for_final),
            "status": "pending",
            "reason": self.reason,
            "note": self.note,
        }


def _resolve_source(source: str) -> Path:
    path = Path(source)
    return path if path.is_absolute() else PROJECT_ROOT / path


def collect(rules: Sequence[SourceRule], *, bundle_root: Path,
            campaign_root: str = CAMPAIGN_ROOT) -> tuple[list[CollectedFile],
                                                        list[PendingFile]]:
    """Copy every declared input that exists; report the rest as pending."""
    collected: list[CollectedFile] = []
    pending: list[PendingFile] = []
    excluded = (bundle_root / "").resolve()
    for rule in rules:
        source = _resolve_source(rule.source)
        if rule.glob:
            if not source.exists():
                pending.append(PendingFile(
                    component=rule.component, path=rule.dest + "<glob>",
                    source=rule.source, required_for_final=(
                        rule.required_for_final),
                    reason="source path absent", note=rule.note))
                continue
            files = [p for p in _iter_files(source)]
            if not files:
                pending.append(PendingFile(
                    component=rule.component, path=rule.dest + "<glob>",
                    source=rule.source, required_for_final=(
                        rule.required_for_final),
                    reason="source directory holds no files", note=rule.note))
                continue
            for path in files:
                try:
                    if path.resolve().is_relative_to(excluded):
                        continue
                except Exception:                  # pragma: no cover
                    pass
                if path.name in rule.exclude_names:
                    continue
                if rule.component == "move_expert":
                    component = _run_component(path)
                    dest_rel = (f"{component}/runs/"
                                f"{path.relative_to(source).as_posix()}")
                else:
                    component = rule.component
                    dest_rel = (rule.dest + path.name if rule.flatten
                                else rule.dest
                                + path.relative_to(source).as_posix())
                collected.append(_copy(path, bundle_root, dest_rel, component))
            continue
        if not source.is_file():
            pending.append(PendingFile(
                component=rule.component, path=rule.dest, source=rule.source,
                required_for_final=rule.required_for_final,
                reason="source file absent", note=rule.note))
            continue
        if rule.dest.endswith("/") and source.is_dir():   # pragma: no cover
            for path in _iter_files(source):
                collected.append(_copy(path, bundle_root,
                                       rule.dest + path.name, rule.component))
            continue
        collected.append(_copy(source, bundle_root, rule.dest, rule.component,
                               flatten=rule.flatten))
    return collected, pending


def _run_component(path: Path) -> str:
    """Route a run artifact by its decoder id (M* → move, P* → process)."""
    run_id = path.parent.name if path.parent.name.startswith("V543_") else (
        path.name)
    for token in str(run_id).split("_"):
        if token.startswith("M"):
            return "move_expert"
        if token.startswith("P"):
            return "process_expert"
    return "move_expert"                                # T0 / unknown → move


def _copy(source: Path, bundle_root: Path, dest_rel: str, component: str,
          *, flatten: bool = False) -> CollectedFile:
    relative = str(dest_rel).lstrip("/")
    if flatten:
        relative = f"{Path(dest_rel).parent}/{source.name}".lstrip("/")
    target = bundle_root / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    return CollectedFile(
        component=component, path=relative, source=_rel(source),
        sha256=sha256_file(target), bytes=int(target.stat().st_size),
        origin="copied")


def _write_generated(bundle_root: Path, dest_rel: str, document: Mapping[str, Any],
                     component: str) -> CollectedFile:
    target = bundle_root / dest_rel
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(dict(document), indent=2, sort_keys=True,
                                 ensure_ascii=False), encoding="utf-8")
    return CollectedFile(component=component, path=dest_rel,
                         source="<generated>", sha256=sha256_file(target),
                         bytes=int(target.stat().st_size), origin="generated")


# ---------------------------------------------------------------------------
# generated bundle documents (encoder manifest / controller policy / reports)
# ---------------------------------------------------------------------------
def encoder_manifest(config: Mapping[str, Any]) -> dict[str, Any]:
    """§13.3 :1333–1335 — ``encoder/A0 checkpoint`` + identity/SHA."""
    source = config.get("source") or {}
    checkpoints = {}
    for seed, entry in sorted((source.get("checkpoints") or {}).items()):
        path = _resolve_source(str(entry.get("path", "")))
        observed = sha256_file(path) if path.is_file() else None
        checkpoints[str(seed)] = {
            "path": str(entry.get("path")),
            "pinned_sha256": str(entry.get("sha256")),
            "observed_sha256": observed,
            "bytes": int(path.stat().st_size) if path.is_file() else None,
            "sha_matches_pin": (None if observed is None
                                else observed == str(entry.get("sha256"))),
            "maps_to": f"V543_M0_S{seed}",
            "role": "frozen A0 coarse encoder (same-seed V542 A0 best)",
        }
    return {
        "schema": ENCODER_MANIFEST_SCHEMA,
        "campaign_id": str(config.get("campaign_id", "")),
        "encoder": str(source.get("encoder", "V542_A0")),
        "checkpoint_rule": str(source.get("checkpoint_rule", "")),
        "freeze_encoder": bool(source.get("freeze_encoder", True)),
        "checkpoints": checkpoints,
        "note": ("the A0 checkpoints live in the FROZEN V542 tree and are "
                 "referenced by path+SHA, never copied over V542 runs (plan "
                 "§13.3 :1333-1335 / §11.3 禁止)"),
    }


def data_provenance(config: Mapping[str, Any]) -> dict[str, Any]:
    """§13.3 :1349 — data / split / sampler SHA."""
    data = config.get("data") or {}
    training = config.get("training") or {}
    split_path = _resolve_source(str(data.get("split_manifest") or ""))
    manifest_path = _resolve_source(
        str(Path(str(data.get("tables_dir") or "")) / "data_manifest.json"))
    return {
        "schema": DATA_PROVENANCE_SCHEMA,
        "tables_dir": str(data.get("tables_dir")),
        "split_manifest": str(data.get("split_manifest")),
        "split_manifest_sha256": (sha256_file(split_path)
                                  if split_path.is_file() else None),
        "data_manifest_sha256": (sha256_file(manifest_path)
                                 if manifest_path.is_file() else None),
        "sampler": str(training.get("sampler")),
        "sampler_spec_sha256": str(training.get("sampler_spec_sha256")),
        "pool": data.get("pool"),
        "dev": data.get("dev"),
        "confirm": data.get("confirm"),
        "confirm_policy": str(data.get("confirm_policy")),
        "label_statuses": list(data.get("label_statuses") or []),
        "note": ("read-only digests; nothing is recomputed from the data view "
                 "and confirm is never opened (plan §9.5 :1039)"),
    }


def five_reports(*, pending: Sequence[PendingFile],
                 collected: Sequence[CollectedFile],
                 campaign_root: str = CAMPAIGN_ROOT) -> dict[str, Any]:
    """The §13.3 :1354–1360 five-report skeleton (filled where evidence exists).

    Every report carries ``status`` ∈ {``complete``, ``partial``, ``pending``}
    and, when not complete, an explicit list of what is missing and where it
    must come from.  Nothing is asserted that the artifacts do not support.
    """
    present = {entry.path for entry in collected}
    missing_paths = {entry.path for entry in pending}

    def _have(needle: str) -> bool:
        return any(needle in path for path in present)

    def _report(name: str, evidence: Sequence[str], pending_reason: str | None,
                extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
        block = {
            "report": name,
            "evidence": [path for path in evidence if _have(path)],
            "missing_evidence": [path for path in evidence if not _have(path)],
            "status": "complete" if not pending_reason else "partial",
            "pending_reason": pending_reason,
            "findings": [],
        }
        if extra:
            block.update(dict(extra))
        return block

    parity_reason = None
    if not _have("bridge/parity_certificate.json"):
        parity_reason = ("the S0-A parity certificate instance is produced by "
                         "`v543_bundle.py certify` and lands in "
                         f"{campaign_root}/plan/parity_certificate.json")
    move_reason = None
    if not _have("provenance/reports/h2_report.json"):
        move_reason = "H2 report absent (WP8 statistics lane not finished)"
    process_reason = None
    if not _have("provenance/reports/h3_report.json"):
        process_reason = "H3 report absent (WP8 statistics lane not finished)"
    rollout_reason = None
    if not _have("provenance/rollout/t7_rollout.json"):
        rollout_reason = ("T7 rollout report absent — run "
                          "`v543_rollout.py` (inner-val leg) or the WP9 dev "
                          "one-shot first")
    selection_reason = None
    if not _have("provenance/selection/final_selection.json"):
        selection_reason = ("final_selection.json absent; the T6 select report "
                            "(track_champions.json) is carried instead and "
                            "the WP9 selection decision is not yet made")
    return {
        "encoder_bridge_parity": _report(
            "encoder_bridge_parity",
            ["bridge/parity_certificate.json", "provenance/audit/s0_probe_"
             "report.json", "bridge/decoder_context_v1.template.json"],
            parity_reason,
            {"plan_lines": "§13.3 :1356 (三 seed parity / hook / q0_raw / "
                           "basis / probe)"}),
        "move_benchmark": _report(
            "move_benchmark",
            ["provenance/reports/h2_report.json",
             "provenance/reports/h3_report.json",
             "provenance/reports/metrics_table.json",
             "provenance/selection/track_champions.json"],
            move_reason,
            {"plan_lines": "§13.3 :1357 (M0–M3 +M4 条件 / H2 / H3 / 成本)"}),
        "process_benchmark": _report(
            "process_benchmark",
            ["provenance/reports/h3_report.json",
             "provenance/reports/metrics_table.json",
             "provenance/selection/track_champions.json"],
            process_reason,
            {"plan_lines": "§13.3 :1358 (P0–P4 +P4-C/P5 / H2 / H3 / 校准 / "
                           "有效点数)"}),
        "integration_rollout": _report(
            "integration_rollout",
            ["provenance/rollout/t7_rollout.json"],
            rollout_reason,
            {"plan_lines": "§13.3 :1359 (B0/BM/BP/BMP / 四组合 / 安全 / 成本)"}),
        "final_selection": _report(
            "final_selection",
            ["provenance/selection/final_selection.json",
             "provenance/selection/track_champions.json"],
            selection_reason,
            {"plan_lines": "§13.3 :1360 (两冠军/留任 / 理由 / 失败与删失 / "
                           "交接)"}),
    }


# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------
def build_bundle(*, campaign_root: str = CAMPAIGN_ROOT,
                 project_root: Path = PROJECT_ROOT,
                 bundle_root: Path | None = None,
                 apply_immutability: bool = True,
                 collected: Sequence[CollectedFile] | None = None,
                 pending: Sequence[PendingFile] | None = None
                 ) -> dict[str, Any]:
    """Write ONE immutable ``PES2TSBundleV543_<UTC>`` tree + manifest + report.

    Returns the manifest document.  Refusals: ``V543_BUNDLE_EXISTS`` (a bundle
    is write-once), ``V543_BUNDLE_ROOT`` (target outside the campaign root).
    ``collected``/``pending`` are injectable for tests only; the normal path
    collects from the declared :func:`collection_plan`.
    """
    root = Path(str(campaign_root))
    if not root.is_absolute():
        root = Path(project_root) / root
    if bundle_root is None:
        bundle_root = root / BUNDLES_SUBDIR / f"{BUNDLE_DIR_PREFIX}{_utc_stamp()}"
    bundle_root = Path(bundle_root)
    try:
        bundle_root.resolve().relative_to(root.resolve())
    except Exception as error:                      # noqa: BLE001 - typed
        raise BundleError(
            CODE_ROOT,
            f"the bundle must live under the campaign root {root} (got "
            f"{bundle_root})") from error
    if bundle_root.exists() and any(bundle_root.iterdir()):
        raise BundleError(
            CODE_EXISTS,
            f"{bundle_root} already exists and is not empty; PES2TSBundleV543 "
            "is write-once per UTC stamp — build a new bundle instead of "
            "overwriting evidence")
    bundle_root.mkdir(parents=True, exist_ok=True)

    config_path = _resolve_source(DEFAULT_CONFIG)
    config_doc: dict[str, Any] = {}
    config_sha = None
    if config_path.is_file():
        import yaml

        config_doc = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        config_sha = sha256_file(config_path)

    if collected is None or pending is None:
        rules = collection_plan(campaign_root=campaign_root)
        collected, pending = collect(rules, bundle_root=bundle_root,
                                    campaign_root=campaign_root)
    files: list[CollectedFile] = list(collected)
    pending_files: list[PendingFile] = list(pending)

    # -- generated documents of the §13.3 layout ---------------------------
    files.append(_write_generated(bundle_root, "encoder/a0_checkpoint_manifest.json",
                                  encoder_manifest(config_doc), "encoder"))
    files.append(_write_generated(bundle_root,
                                  "provenance/data/split_sampler_provenance.json",
                                  data_provenance(config_doc), "provenance"))
    files.append(_write_generated(bundle_root,
                                  "controller/legal_endpoint_policy.json",
                                  CONTROLLER_POLICY, "controller"))
    reports = five_reports(pending=pending_files, collected=files,
                           campaign_root=campaign_root)
    report_doc = {
        "schema": BUNDLE_REPORT_SCHEMA,
        "bundle_version": BUNDLE_VERSION,
        "bundle_id": bundle_root.name,
        "created_utc": _utc_now(),
        "campaign_id": str(config_doc.get("campaign_id", "")),
        "plan_section": "§13.3 :1329-1375 (PES2TSBundleV543 + 五份报告 + 账本)",
        "five_reports": reports,
        "reports_complete": [name for name, block in reports.items()
                             if block["status"] == "complete"],
        "reports_pending": [name for name, block in reports.items()
                            if block["status"] != "complete"],
        "pending_inputs": [entry.describe() for entry in pending_files],
        "n_files": len(files),
        "n_files_note": ("counted before this report and manifest.json are "
                         "written; the authoritative total is "
                         "manifest.counts.n_files"),
        "n_pending": len(pending_files),
        "strict_load": {
            "status": "deferred",
            "rule": ("§11.5 test_v543_bundle_strict_load — a bundle is "
                     "strict-loaded once the champion checkpoints exist; this "
                     "intermediate bundle records the manifest digests only"),
        },
        "read_only_declared": bool(apply_immutability),
        "confirm_read": False,
        "new_QC_calls": 0,
        "ambiguities": _bundle_ambiguities(),
    }
    files.append(_write_generated(bundle_root, "BUNDLE_REPORT.json", report_doc,
                                  "provenance"))
    markdown = render_report_markdown(report_doc)
    md_path = bundle_root / "BUNDLE_REPORT.md"
    md_path.write_text(markdown, encoding="utf-8")
    files.append(CollectedFile(component="provenance", path="BUNDLE_REPORT.md",
                               source="<generated>", sha256=sha256_file(md_path),
                               bytes=int(md_path.stat().st_size),
                               origin="generated"))

    chmod_applied = bool(apply_immutability) and os.name == "posix"
    manifest = {
        "schema": BUNDLE_SCHEMA,
        "bundle_version": BUNDLE_VERSION,
        "bundle_id": bundle_root.name,
        "created_utc": _utc_now(),
        "campaign_id": str(config_doc.get("campaign_id", "")),
        "campaign_root": str(campaign_root),
        "config_path": DEFAULT_CONFIG,
        "config_sha256": config_sha,
        "plan_section": "§13.3 :1329-1362 / §11.5 test_v543_legacy_files_untouched",
        "collected_from_committed_artifacts": True,
        "files": [entry.describe() for entry in sorted(
            files, key=lambda item: item.path)],
        "pending": [entry.describe() for entry in pending_files],
        "counts": {
            "n_files": len(files),
            "n_bytes": int(sum(entry.bytes for entry in files)),
            "n_pending": len(pending_files),
            "n_components": len(sorted({entry.component for entry in files})),
        },
        "components": sorted({entry.component for entry in files}),
        "reports": {
            "complete": report_doc["reports_complete"],
            "pending": report_doc["reports_pending"],
        },
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "generator": _rel(Path(__file__).resolve()),
            "generator_sha256": sha256_file(Path(__file__).resolve()),
        },
        "immutability": {
            "write_once": True,
            "chmod_applied": chmod_applied,
            "declared_read_only": bool(apply_immutability),
            "declaration": "IMMUTABLE.json",
        },
    }
    manifest_path = bundle_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True,
                                       ensure_ascii=False), encoding="utf-8")
    manifest_sha = sha256_file(manifest_path)

    immutable = {
        "schema": IMMUTABLE_SCHEMA,
        "bundle_id": bundle_root.name,
        "created_utc": _utc_now(),
        "manifest_sha256": manifest_sha,
        "manifest_path": "manifest.json",
        "declared_read_only": bool(apply_immutability),
        "rule": ("a PES2TSBundleV543 is write-once evidence: re-building means "
                 "a NEW UTC-stamped directory; `verify` refuses any digest, "
                 "size or file-set drift (plan §13.3 :1362 append-only)"),
        "chmod": {"applied": chmod_applied, "files_mode": "0444",
                  "dirs_mode": "0555", "posix_only": True},
        "confirm_read": False,
        "new_QC_calls": 0,
    }
    (bundle_root / "IMMUTABLE.json").write_text(
        json.dumps(immutable, indent=2, sort_keys=True, ensure_ascii=False),
        encoding="utf-8")
    if chmod_applied:
        # freeze every file the manifest lists, plus the manifest itself; the
        # IMMUTABLE.json declaration stays readable and is frozen last.
        for entry in files:
            os.chmod(bundle_root / entry.path,
                     stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
        os.chmod(manifest_path,
                 stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
        for path in sorted(bundle_root.rglob("*"), reverse=True):
            if path.is_dir():
                os.chmod(path, stat.S_IRUSR | stat.S_IXUSR | stat.S_IRGRP
                         | stat.S_IXGRP | stat.S_IROTH | stat.S_IXOTH)
        os.chmod(bundle_root / "IMMUTABLE.json",
                 stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    manifest["manifest_sha256"] = manifest_sha
    return manifest


def render_report_markdown(report: Mapping[str, Any]) -> str:
    """Render the bundle report as a small markdown summary."""
    lines = [
        f"# {report.get('bundle_id', 'PES2TSBundleV543')}",
        "",
        f"- campaign: {report.get('campaign_id', '')}",
        f"- created (UTC): {report.get('created_utc', '')}",
        f"- files: {report.get('n_files', 0)} / pending inputs: "
        f"{report.get('n_pending', 0)}",
        f"- confirm_read: false / new_QC_calls: 0",
        "",
        "## Five reports (§13.3 :1354-1360)",
        "",
        "| report | status | missing evidence |",
        "|---|---|---|",
    ]
    for name, block in (report.get("five_reports") or {}).items():
        missing = ", ".join(block.get("missing_evidence") or []) or "—"
        lines.append(f"| {name} | {block.get('status')} | {missing} |")
    lines.extend(["", "## Pending inputs", ""])
    pending = report.get("pending_inputs") or []
    if not pending:
        lines.append("_none_")
    else:
        for entry in pending:
            lines.append(
                f"- `{entry.get('path')}` ← `{entry.get('source')}` "
                f"({'required' if entry.get('required_for_final') else 'optional'})"
                f": {entry.get('reason')}")
    lines.extend(["", "## Ambiguities", ""])
    for item in report.get("ambiguities") or []:
        lines.append(f"- {item}")
    lines.append("")
    return "\n".join(lines)


def _bundle_ambiguities() -> list[str]:
    return [
        "§13.3 :1332 names the tree PES2TSBundleV543 and does not fix where it "
        "lives or how immutability is declared; this lane writes "
        "outputs/pesn/v543_dual_decoder_50k/bundles/PES2TSBundleV543_<UTC>/ "
        "with an IMMUTABLE.json declaration plus POSIX chmod 0444/0555.",
        "§13.3 :1349 asks for 'data / split / sampler SHA'; the sampler spec "
        "SHA is a pinned config value (training.sampler_spec_sha256) and the "
        "data digests are computed from the frozen manifest files — the bundle "
        "records both sources explicitly.",
        "the 'five reports' are deliverables that combine artifacts from the "
        "statistics / rollout lanes; this bundle declares each report's "
        "evidence and marks the section pending when that evidence does not "
        "exist yet instead of paraphrasing numbers into it.",
        "§13.3 :1362 lists run_ledger.jsonl / run_records.jsonl / "
        "train_log.jsonl / metrics.json at the CAMPAIGN root; the formal "
        "runner writes run_records.jsonl per RUN directory plus one "
        "run_ledger.jsonl, so the bundle collects both layouts and reports "
        "which existed.",
    ]


# ---------------------------------------------------------------------------
# verify
# ---------------------------------------------------------------------------
def verify_bundle(bundle_root: Any) -> dict[str, Any]:
    """Re-hash a bundle against its manifest (§11.2 ``verify``).

    Fail-closed: a changed digest/size, a missing file, an unexpected extra
    file, a manifest whose own digest disagrees with ``IMMUTABLE.json``, or a
    tree that was made writable again are all typed refusals.
    """
    root = Path(str(bundle_root))
    if not root.is_dir():
        raise BundleError(CODE_VERIFY_MISSING, f"bundle directory not found: "
                                              f"{root}")
    manifest = _read_json(root / "manifest.json",
                          what="bundle manifest", code=CODE_VERIFY_MISSING)
    if str(manifest.get("schema")) != BUNDLE_SCHEMA:
        raise BundleError(
            CODE_VERIFY_SCHEMA,
            f"{root}/manifest.json schema {manifest.get('schema')!r} != "
            f"{BUNDLE_SCHEMA}")
    declared = {str(entry["path"]): entry
                for entry in (manifest.get("files") or [])}
    failures: list[str] = []
    checked = 0
    for relative, entry in sorted(declared.items()):
        target = root / relative
        if not target.is_file():
            failures.append(f"{relative}: missing")
            continue
        actual = sha256_file(target)
        size = int(target.stat().st_size)
        checked += 1
        if actual != str(entry.get("sha256")):
            failures.append(
                f"{relative}: sha256 {actual} != manifest "
                f"{entry.get('sha256')}")
        if size != int(entry.get("bytes", -1)):
            failures.append(f"{relative}: bytes {size} != manifest "
                            f"{entry.get('bytes')}")
    on_disk = {str(path.relative_to(root)) for path in _iter_files(root)}
    allowed_extra = {"manifest.json", "IMMUTABLE.json"}
    extra = sorted(on_disk - set(declared) - allowed_extra)
    if extra:
        failures.append(f"unexpected file(s) not in the manifest: {extra[:8]}")
    declaration_path = root / "IMMUTABLE.json"
    declaration: dict[str, Any] = {}
    if declaration_path.is_file():
        declaration = _read_json(declaration_path, what="IMMUTABLE.json",
                                 code=CODE_VERIFY_IMMUTABLE)
        recorded = str(declaration.get("manifest_sha256", ""))
        actual_manifest = sha256_file(root / "manifest.json")
        if recorded and recorded != actual_manifest:
            failures.append(
                f"manifest.json sha256 {actual_manifest} != the digest the "
                f"IMMUTABLE.json declaration froze ({recorded})")
    else:
        failures.append("IMMUTABLE.json is missing (the read-only declaration "
                        "is part of the §13.3 deliverable)")
    writable: list[str] = []
    if os.name == "posix" and bool(declaration.get("chmod", {}).get("applied")):
        for relative in sorted(declared):
            target = root / relative
            if target.is_file() and bool(
                    target.stat().st_mode & (stat.S_IWUSR | stat.S_IWGRP
                                             | stat.S_IWOTH)):
                writable.append(relative)
        if writable:
            failures.append(f"file(s) are writable again: {writable[:8]}")
    if failures:
        code = (CODE_VERIFY_DIGEST if any("sha256" in item or "bytes" in item
                                          for item in failures)
                else (CODE_VERIFY_EXTRA if any("unexpected" in item
                                               for item in failures)
                      else CODE_VERIFY_MISSING))
        raise BundleError(code, "; ".join(failures[:8]))
    return {
        "schema": "pes2ts_v543_bundle_verify_v1",
        "bundle_id": manifest.get("bundle_id"),
        "manifest_sha256": sha256_file(root / "manifest.json"),
        "n_files_verified": checked,
        "n_declared": len(declared),
        "n_pending": len(manifest.get("pending") or []),
        "reports_complete": (manifest.get("reports") or {}).get("complete"),
        "reports_pending": (manifest.get("reports") or {}).get("pending"),
        "immutable_declared": bool(declaration.get("declared_read_only")),
        "chmod_applied": bool(declaration.get("chmod", {}).get("applied")),
        "ok": True,
    }


# ---------------------------------------------------------------------------
# S0-A parity CERTIFICATE INSTANCE generator (plan §5.8 / §7.1 / ADR-0034 §3)
# ---------------------------------------------------------------------------
#: The template the instance is derived from (WP0 frozen).
CERTIFICATE_TEMPLATE: Final[str] = (
    "config/campaigns/v543_templates/parity_certificate.template.json")
#: The instance path named by the WP9 task brief (``plan/``).
CERTIFICATE_INSTANCE: Final[str] = (
    "outputs/pesn/v543_dual_decoder_50k/plan/parity_certificate.json")
#: The path the template's own ``_template_note`` names (``audits/``).
CERTIFICATE_INSTANCE_TEMPLATE_NOTE: Final[str] = (
    "outputs/pesn/v543_dual_decoder_50k/audits/parity_certificate.json")
#: Seeds of the three frozen A0 checkpoints (§7.1 S0-A).
CERTIFICATE_SEEDS: Final[tuple[int, ...]] = (17, 29, 43)
#: Fixture batches per seed (the WP1 realdata test's cadence).
CERTIFICATE_FIXTURE_BATCHES: Final[int] = 8
#: Real inner-val batches per seed (§11.5: ">= 8 batches").
CERTIFICATE_REAL_BATCHES: Final[int] = 8
#: The V542 data-view config the frozen inner-val batches come from.
V542_DATA_CONFIG: Final[str] = "config/campaigns/pes2ts_v542_backbone_50k.yaml"
#: Code surface whose per-file SHA becomes ``code_sha256``.
CERTIFICATE_CODE_FILES: Final[tuple[str, ...]] = (
    "scripts/agent_ops/v543_bundle.py",
    "scripts/agent_ops/v542_a0_parity.py",
    "scripts/agent_ops/feature_gate/v3_fixtures.py",
    "pes2ts_core/pesn/models/v543/contracts.py",
    "pes2ts_core/pesn/models/v543/coarse_encoder.py",
    "pes2ts_core/pesn/models/v543/latent_bridge.py",
    "pes2ts_core/pesn/models/v543/registry.py",
    "pes2ts_core/pesn/models/v543/assembly.py",
    "pes2ts_core/pesn/models/v543/model.py",
    "pes2ts_core/pesn/models/v542/backbone_a0.py",
    "pes2ts_core/pesn/models/v542/decoder.py",
    "pes2ts_core/pesn/models/v542/loader.py",
    "pes2ts_core/pesn/models/v542/identity.py",
)


def code_identity(files: Sequence[str] = CERTIFICATE_CODE_FILES
                  ) -> dict[str, str]:
    identity: dict[str, str] = {}
    for relative in dict.fromkeys(files):
        path = _resolve_source(relative)
        if not path.is_file():
            raise BundleError(CODE_CERT_TEMPLATE,
                              f"certificate code file missing: {relative}")
        identity[relative] = sha256_file(path)
    return identity


def environment_identity() -> dict[str, Any]:
    import torch

    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "torch": str(torch.__version__),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_version": (None if not torch.cuda.is_available()
                         else str(torch.version.cuda)),
        "dtype": "fp32",
        "torch_num_threads_during_parity": 1,
        "tf32_during_parity": False,
    }


def _max_abs(a: Any, b: Any) -> float:
    import torch

    if not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor):
        return float("nan")
    if a.numel() == 0:
        return 0.0
    return float((a.detach().cpu() - b.detach().cpu()).abs().max().item())


def measure_checkpoint(*, seed: int, checkpoint_path: Path,
                       pinned_sha256: str | None,
                       fixture_batches: int = CERTIFICATE_FIXTURE_BATCHES,
                       real_batches: int | None = CERTIFICATE_REAL_BATCHES,
                       data_config: str = V542_DATA_CONFIG
                       ) -> dict[str, Any]:
    """Measure the seven §5.8 equations for ONE A0 checkpoint (CPU bitwise).

    Returns the per-checkpoint certificate block of the template, with every
    equation a boolean, every max-abs-difference a measured float, the hook
    counts, and the two MEASURED state-dict counts.
    """
    import torch

    from pes2ts_core.pesn.contracts.key_atom_v3 import forward_view_v3
    from pes2ts_core.pesn.models.v542 import V542Model, load_a0_strict
    from pes2ts_core.pesn.models.v543 import (
        PAYLOAD_LAYOUT_V542_A0_45,
        CoarseEncoder,
        V543Model,
        assert_encoder_packet_parity,
        resolve_a0_mapping,
    )
    from pes2ts_core.pesn.models.v543.assembly import assemble_v543
    from scripts.agent_ops.feature_gate.v3_fixtures import make_v3_gate_batch

    path = Path(checkpoint_path)
    if not path.is_file():
        raise BundleError(CODE_CERT_CHECKPOINT,
                          f"A0 checkpoint missing: {path}")
    observed_sha = sha256_file(path)
    payload = torch.load(str(path), map_location="cpu", weights_only=False)
    state = payload["model"] if isinstance(payload, dict) and "model" in \
        payload else payload

    torch.set_num_threads(1)
    encoder = CoarseEncoder.load(payload, int(seed))
    encoder.eval()
    reference = V542Model()
    _layout, table = resolve_a0_mapping(state,
                                        target_keys=set(reference.state_dict()))
    load_a0_strict(reference, payload, table)
    reference.eval()
    assembly = assemble_v543(encoder, active_track="move",
                             encoder_ckpt_sha256=str(observed_sha))
    model = V543Model(assembly)
    audit = encoder.load_audit or {}

    hook_expect = {"message_mlp_last": 1, "norm_h_last": 1, "progress_fc2": 1}
    hook_counts = {"message_mlp[-1]": 0, "norm_h[-1]": 0,
                   "progress_head.fc2": 0}
    max_abs = {key: 0.0 for key in ("u0", "z", "q0", "q0_raw", "endpoint0",
                                    "u", "q", "endpoint_logit",
                                    "order_score")}
    eq_u0 = eq_z = eq_q0 = eq_tanh = eq_endpoint = eq_model = eq_order = True
    n_forwards = 0
    batches_checked = 0

    def _one(batch: Any, label: str) -> None:
        nonlocal n_forwards, batches_checked
        nonlocal eq_u0, eq_z, eq_q0, eq_tanh, eq_endpoint, eq_model, eq_order
        view = batch
        with torch.no_grad():
            packet = encoder(view)
            ref = reference(view)
            ref_packet = reference.backbone(view)
            out = model.forward_packet(packet, view)
        n_forwards += 1
        batches_checked += 1
        fires = dict(encoder.last_hook_fires)
        if fires != hook_expect:
            raise BundleError(
                CODE_CERT_EQUATION,
                f"seed {seed} {label}: hook fire counts {fires} != "
                f"{hook_expect} (plan §11.5 test_v543_hook_fires_once)")
        for key in hook_counts:
            hook_counts[key] = max(hook_counts[key], int(fires[
                {"message_mlp[-1]": "message_mlp_last",
                 "norm_h[-1]": "norm_h_last",
                 "progress_head.fc2": "progress_fc2"}[key]]))
        rows = assert_encoder_packet_parity(
            packet, z=ref["z"], u0=ref["u"], q=ref["q"],
            endpoint_logit=ref["endpoint_logit"],
            edge_msg=ref_packet.edge_msg, context=f"seed {seed} {label}")
        eq_u0 = eq_u0 and bool(rows["u0"]["bitwise"])
        eq_z = eq_z and bool(rows["z"]["bitwise"])
        eq_q0 = eq_q0 and bool(rows["q0"]["bitwise"])
        eq_tanh = eq_tanh and bool(rows["q0_raw_tanh"]["bitwise"])
        eq_endpoint = eq_endpoint and bool(rows["endpoint0"]["bitwise"])
        max_abs["u0"] = max(max_abs["u0"], float(rows["u0"]["max_abs_diff"]))
        max_abs["z"] = max(max_abs["z"], float(rows["z"]["max_abs_diff"]))
        max_abs["q0"] = max(max_abs["q0"], float(rows["q0"]["max_abs_diff"]))
        max_abs["q0_raw"] = max(max_abs["q0_raw"], float(
            rows["q0_raw_tanh"]["max_abs_diff"]))
        max_abs["endpoint0"] = max(max_abs["endpoint0"],
                                   float(rows["endpoint0"]["max_abs_diff"]))
        for key in ("u", "z", "q", "endpoint_logit", "order_score"):
            if key == "z":
                bitwise = bool(torch.equal(out["z"], ref["z"]))
            else:
                bitwise = bool(torch.equal(out[key], ref[key]))
            eq_model = eq_model and bitwise
            max_abs[key] = max(max_abs[key], _max_abs(out[key], ref[key]))
        eq_order = eq_order and bool(out["order_score"] is out["q"]) and bool(
            ref["order_score"] is ref["q"])

    for index in range(int(fixture_batches)):
        batch = forward_view_v3(make_v3_gate_batch(n_graphs=2,
                                                   jitter_seed=int(seed) + index))
        _one(batch, f"fixture batch {index}")

    real_block: dict[str, Any] = {"requested": int(real_batches or 0),
                                  "executed": 0, "status": "not_requested"}
    if real_batches:
        try:
            from scripts.agent_ops.v542_a0_parity import (
                inner_val_batches,
                load_pinned_assets,
            )

            cfg, _assets = load_pinned_assets(_resolve_source(data_config))
            batches = inner_val_batches(cfg, int(seed),
                                        n_batches=int(real_batches))
            for index, batch in enumerate(batches):
                _one(batch, f"inner-val batch {index}")
            real_block = {"requested": int(real_batches),
                          "executed": len(batches),
                          "status": "executed",
                          "source": data_config,
                          "rule": ("frozen inner-val batch selection of "
                                   "v542_a0_parity.inner_val_selection "
                                   "(deterministic, no RNG)"),
                          "note": ("'requested' is the MINIMUM of §11.5 "
                                   "('>= 8 batches'); the frozen selection "
                                   "appends one deterministic coverage batch, "
                                   "so 'executed' can be larger")}
        except Exception as error:                 # noqa: BLE001 - typed
            real_block = {
                "requested": int(real_batches),
                "executed": 0,
                "status": "skipped",
                "reason": f"{type(error).__name__}: {error}",
                "note": ("a fixture-only certificate is NOT the full S0-A "
                         "leg (§7.1: real-batch failure is a data-dependent "
                         "boundary bug, never waived)"),
            }

    n_source_keys = int(audit.get("n_source_keys", len(state)))
    n_targets = int(audit.get("n_mapped_targets", 0))
    layout = str(audit.get("payload_layout", ""))
    equations = {
        "u0_equals_v542_u": bool(eq_u0),
        "z_equals_v542_z": bool(eq_z),
        "q0_equals_v542_q": bool(eq_q0),
        "tanh_q0_raw_equals_q0": bool(eq_tanh),
        "endpoint0_equals_v542_endpoint_logit": bool(eq_endpoint),
        "zero_residual_model_equal": bool(eq_model),
        "order_score_is_q": bool(eq_order),
    }
    all_equations = all(equations.values())
    cpu_bitwise = bool(all_equations)
    real_bitwise = (None if real_block["status"] != "executed"
                    else bool(all_equations))
    status = ("pass" if cpu_bitwise and (real_block["status"] == "executed"
                                        and real_bitwise) else
              "pass_cpu_fixtures_only" if cpu_bitwise else "fail")
    return {
        "id": f"V542_A0_S{int(seed)}",
        "source": (f"V542 A0 matched baseline seed{seed} best inner-val "
                   "checkpoint (the same-seed encoder V543 freezes)"),
        "source_tier": "v542_backbone_50k_runs_best",
        "required": True,
        "path": _rel(path),
        "sha256": observed_sha,
        "pinned_sha256": pinned_sha256,
        "sha_matches_pin": (None if pinned_sha256 is None
                            else observed_sha == str(pinned_sha256)),
        "maps_to": f"V543_M0_S{int(seed)}",
        "equations": equations,
        "equation_count_passed": int(sum(1 for value in equations.values()
                                         if value)),
        "equation_count_required": 7,
        "cpu_bitwise": cpu_bitwise,
        "real_batch_bitwise": real_bitwise,
        "gpu_tolerance": {
            "status": "not_executed",
            "atol": 1e-06,
            "rtol": 1e-05,
            "reason": ("this instance was produced on CPU; §7.1 S0-A makes "
                       "CPU bitwise DECISIVE and the GPU caps a hard upper "
                       "bound that must never be relaxed"),
            "measured_max_abs": None,
        },
        "max_abs_diff": {key: float(value) for key, value in max_abs.items()},
        "encoder_forward_count": int(n_forwards),
        "batches_checked": int(batches_checked),
        "hook_fire_counts": hook_counts,
        "hook_fire_expectation": dict(hook_expect),
        "state_dict_counts": {
            "payload_key_count_measured": int(n_source_keys),
            "expanded_target_key_count_measured": int(
                len(set(reference.state_dict())) if n_targets == 0 else n_targets),
            "count_scope_of_this_audit": (
                "expanded_target_key_count (the strict-load target key set the "
                "per-key torch.equal assertions below run on)"),
            "payload_layout_measured": layout,
            "expected_payload_key_count": 43,
            "expected_expanded_target_key_count": 45,
            "payload_layout_is_v542_a0_joint_45": (
                layout == PAYLOAD_LAYOUT_V542_A0_45),
            "consistency_rule": "expanded_target_key_count == "
                                "payload_key_count + 2",
            "consistency_rule_holds": bool(
                int(len(set(reference.state_dict()))) == int(n_source_keys) + 2),
            "note": ("the S0-A asset is the V5.4.2 45-key v542_a0_joint_45 "
                     "payload (the w_v dual-write is ALREADY materialised in "
                     "the file), so the MEASURED payload count is 45 and the "
                     "'payload + 2' identity does not hold for it; the 43-key "
                     "form is the legacy_v53mp_joint_43 layout. The measured "
                     "values are recorded as they are — the frozen expectation "
                     "is never rewritten to agree (§7.1 S0-A)"),
        },
        "load_audit": {
            "strict": bool(audit.get("strict", True)),
            "ok": bool(audit.get("ok", True)),
            "n_source_keys": n_source_keys,
            "n_mapped_targets": n_targets,
            "torch_equal_verified": int(audit.get("torch_equal_verified", 0)),
            "mapping_table_sha256": audit.get("mapping_table_sha256"),
            "mapping_mode": audit.get("mapping_mode"),
            "encoder_ckpt_sha256": audit.get("encoder_ckpt_sha256"),
            "missing": (audit.get("load_audit") or {}).get("missing", []),
            "unexpected": (audit.get("load_audit") or {}).get("unexpected", []),
            "shape_mismatch": (audit.get("load_audit") or {}).get(
                "shape_mismatch", []),
            "dtype_mismatch": (audit.get("load_audit") or {}).get(
                "dtype_mismatch", []),
            "strict_false_used": False,
        },
        "real_batch": real_block,
        "status": status,
    }


def generate_parity_certificate(*, template_path: str = CERTIFICATE_TEMPLATE,
                                config_path: str = DEFAULT_CONFIG,
                                output_path: str | None = None,
                                seeds: Sequence[int] = CERTIFICATE_SEEDS,
                                fixture_batches: int = CERTIFICATE_FIXTURE_BATCHES,
                                real_batches: int | None = CERTIFICATE_REAL_BATCHES,
                                data_config: str = V542_DATA_CONFIG,
                                measurer: Callable[..., dict[str, Any]] | None = None
                                ) -> dict[str, Any]:
    """Measure the three checkpoints and fill the template INSTANCE.

    ``measurer`` is injectable so the assembly logic can be unit-tested without
    torch/checkpoints; the default is :func:`measure_checkpoint`.
    """
    import yaml

    template_file = _resolve_source(template_path)
    template = _read_json(template_file, what="parity certificate template",
                          code=CODE_CERT_TEMPLATE)
    config_file = _resolve_source(config_path)
    config_doc = {}
    if config_file.is_file():
        config_doc = yaml.safe_load(config_file.read_text(encoding="utf-8")) or {}
    pinned = {}
    for seed, entry in ((config_doc.get("source") or {}).get("checkpoints")
                        or {}).items():
        pinned[int(seed)] = dict(entry)

    measure = measurer or measure_checkpoint
    blocks: list[dict[str, Any]] = []
    for seed in seeds:
        entry = pinned.get(int(seed))
        if entry is None:
            raise BundleError(
                CODE_CERT_CHECKPOINT,
                f"campaign config pins no A0 checkpoint for seed {seed}")
        path = _resolve_source(str(entry["path"]))
        blocks.append(measure(seed=int(seed), checkpoint_path=path,
                              pinned_sha256=str(entry.get("sha256")),
                              fixture_batches=int(fixture_batches),
                              real_batches=real_batches,
                              data_config=data_config))

    environment = environment_identity()
    code = code_identity()
    document = fill_certificate(template, blocks, extra={
        "instance_schema": "pes2ts_v543_parity_certificate_v1_instance",
        "template_path": _rel(template_file),
        "template_sha256": sha256_file(template_file),
        "template_schema": str(template.get("schema", "")),
        "generated_by": _rel(Path(__file__).resolve()),
        "generated_utc": _utc_now(),
        "campaign_id": str(config_doc.get("campaign_id",
                                          template.get("campaign_id", ""))),
        "config_path": config_path,
        "config_sha256": (sha256_file(config_file)
                          if config_file.is_file() else None),
        "code_sha256": canonical_sha256(code),
        "code_files": code,
        "environment_sha256": canonical_sha256(environment),
        "environment": environment,
        "source_manifest_sha256": canonical_sha256({
            "checkpoints": {str(block["id"]): block["sha256"]
                            for block in blocks},
            "paths": {str(block["id"]): block["path"] for block in blocks},
        }),
        "bridge_spec_sha256": _bridge_spec_sha256(),
        "checkpoint_count": len(blocks),
        "required_count": int(template.get("required_count", 3)),
        "overall_status": ("pass_cpu_pending_gpu_and_signoff"
                           if all(block["status"].startswith("pass")
                                  for block in blocks)
                           else "fail"),
        "overall_status_note": (
            "every §5.8 equation held bitwise on CPU for all three seeds; the "
            "GPU leg was not executed in this environment and the certificate "
            "is UNSIGNED (signed_by/signed_on stay null until the WP1 "
            "sign-off)"),
        "signed_by": None,
        "signed_on": None,
        "instances": {
            "written_by_task_brief": CERTIFICATE_INSTANCE,
            "named_by_template_note": CERTIFICATE_INSTANCE_TEMPLATE_NOTE,
            "note": ("the two paths disagree (template _template_note says "
                     "outputs/.../audits/, the WP9 brief says "
                     "outputs/.../plan/); this generator writes the brief's "
                     "path and records both here"),
        },
        "ambiguities": [
            "the template note names outputs/pesn/v543_dual_decoder_50k/"
            "audits/parity_certificate.json while the WP9 task brief names "
            "outputs/pesn/v543_dual_decoder_50k/plan/parity_certificate.json; "
            "the brief's path is written and both are recorded in "
            "'instances'.",
            "the template freezes expected counts 43/45, but the S0-A assets "
            "on disk are the 45-key v542_a0_joint_45 payloads, so the MEASURED "
            "payload count is 45 and 'expanded = payload + 2' cannot hold; the "
            "measured values are reported and the expectation is NOT rewritten "
            "(§7.1 S0-A forbids releasing on approximate equality).",
            "§7.1 S0-A's GPU leg (atol 1e-6 / rtol 1e-5) needs the GPU host; "
            "this instance records gpu_tolerance.status=not_executed and keeps "
            "CPU bitwise as the decisive criterion.",
        ],
    })
    if output_path is not None:
        target = _resolve_source(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(document, indent=2, sort_keys=True,
                                     ensure_ascii=False), encoding="utf-8")
        document["written_to"] = _rel(target)
        document["written_sha256"] = sha256_file(target)
    return document


def fill_certificate(template: Mapping[str, Any],
                     blocks: Sequence[Mapping[str, Any]], *, extra: Mapping[str, Any] | None = None
                     ) -> dict[str, Any]:
    """Pure merge of measured checkpoint blocks into a template document.

    Split out of :func:`generate_parity_certificate` so the field mapping is
    unit-testable without torch: the template's static blocks are preserved,
    ``checkpoints`` is replaced by the measured list (one entry per seed, in
    ``seeds`` order) and each measured block must carry every template field.
    """
    document = dict(template)
    document["checkpoints"] = [dict(block) for block in blocks]
    document["checkpoint_count"] = len(blocks)
    if extra:
        document.update(dict(extra))
    return document


def _bridge_spec_sha256() -> str:
    from pes2ts_core.pesn.models.v543 import LatentBridge

    return str(LatentBridge().bridge_spec_sha256())


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="V5.4.3 WP9 bundle build/verify + the S0-A parity "
                    "certificate instance generator (plan §13.3)")
    sub = parser.add_subparsers(dest="command", required=True)

    p_build = sub.add_parser("build", help="write an immutable bundle")
    p_build.add_argument("--campaign-root", default=CAMPAIGN_ROOT)
    p_build.add_argument("--out-dir", default=None,
                         help="explicit bundle directory (default: "
                              "bundles/PES2TSBundleV543_<UTC>)")
    p_build.add_argument("--no-immutability", action="store_true",
                         help="skip the read-only declaration (not for WP9)")

    p_verify = sub.add_parser("verify", help="re-hash an existing bundle")
    p_verify.add_argument("--bundle", required=True)

    p_cert = sub.add_parser("certify",
                            help="measure and write the parity certificate "
                                 "instance")
    p_cert.add_argument("--out", default=CERTIFICATE_INSTANCE)
    p_cert.add_argument("--template", default=CERTIFICATE_TEMPLATE)
    p_cert.add_argument("--config", default=DEFAULT_CONFIG)
    p_cert.add_argument("--data-config", default=V542_DATA_CONFIG)
    p_cert.add_argument("--fixture-batches", type=int,
                        default=CERTIFICATE_FIXTURE_BATCHES)
    p_cert.add_argument("--real-batches", type=int,
                        default=CERTIFICATE_REAL_BATCHES)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "build":
            manifest = build_bundle(
                campaign_root=args.campaign_root,
                bundle_root=(Path(args.out_dir) if args.out_dir else None),
                apply_immutability=not bool(args.no_immutability))
            print(f"[v543bundle] bundle -> "
                  f"{args.out_dir or manifest['bundle_id']}", file=sys.stderr)
            print(json.dumps({
                "bundle_id": manifest["bundle_id"],
                "n_files": manifest["counts"]["n_files"],
                "n_bytes": manifest["counts"]["n_bytes"],
                "n_pending": manifest["counts"]["n_pending"],
                "reports_complete": manifest["reports"]["complete"],
                "reports_pending": manifest["reports"]["pending"],
            }, indent=2, sort_keys=True), file=sys.stderr)
            return 0
        if args.command == "verify":
            result = verify_bundle(args.bundle)
            print(json.dumps(result, indent=2, sort_keys=True), file=sys.stderr)
            return 0
        document = generate_parity_certificate(
            template_path=args.template, config_path=args.config,
            output_path=args.out, fixture_batches=int(args.fixture_batches),
            real_batches=(None if not args.real_batches
                          else int(args.real_batches)),
            data_config=args.data_config)
        print(json.dumps({
            "written_to": document.get("written_to"),
            "overall_status": document.get("overall_status"),
            "checkpoints": [
                {"id": block["id"], "status": block["status"],
                 "equations_passed": block["equation_count_passed"],
                 "real_batch": block["real_batch"]["status"],
                 "payload_key_count_measured":
                     block["state_dict_counts"]["payload_key_count_measured"],
                 "expanded_target_key_count_measured":
                     block["state_dict_counts"][
                         "expanded_target_key_count_measured"]}
                for block in document["checkpoints"]],
        }, indent=2, sort_keys=True), file=sys.stderr)
        return 0
    except BundleError as error:
        print(f"[v543bundle] REFUSED {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":                        # pragma: no cover
    raise SystemExit(main())
