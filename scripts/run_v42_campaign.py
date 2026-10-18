"""V4.2–V4.7 overnight campaign driver (Lane C; contract §8/§11).

Single-GPU sequential orchestration of the frozen arm table:

* precheck (product existence + data counts + H1 gate) → per-architecture
  timing calibration → staged execution → promotion-based parent resolution
  (§9.1) → V4.7 FINAL three-seed re-fit (with §9.2 40k gate + a_s@fit50
  re-estimation) → hand-off to the terminal review (out of scope).

Each formal run is a SUBPROCESS
(``python -m pes2ts_core.pesn.offline.training.dual_ts_campaign``) so a
crashed arm cannot take down the driver.  The driver is crash-restart safe:
it rebuilds its full state from ``campaign_manifest.json`` +
``stage_decisions.jsonl`` + the per-run ``run_record.json`` files, never
re-running a completed run.  A hard deadline saves-then-stops the current arm
and records ``budget_censored`` (never a performance failure).  Overruns
prune complete double-seed arms in the frozen order W128 → V2 → E2.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Final

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pes2ts_core.pesn.offline.training.dual_ts_campaign import (  # noqa: E402
    CampaignArmConfig,
    CampaignDatasetSpec,
    RUN_RECORD_V3_SCHEMA,
    compute_a_s,
    load_campaign_states,
)
from pes2ts_core.utils.jsonio import write_json  # noqa: E402

#: Python executable for the single-GPU training environment.
PYTHON: Final[str] = os.environ.get("PES2TS_GPU_PYTHON", sys.executable)

#: Campaign output root (contract §9).
OUT_ROOT: Final[Path] = PROJECT_ROOT / (
    "outputs/pesn/dual_ts_v2_20261015/wp42_campaign")

#: Data product paths (Lane A products).
DEV_SPLIT_PATH: Final[Path] = OUT_ROOT / "dev_split.json"
PHYSICAL_PRIOR_PATH: Final[Path] = OUT_ROOT / "physical_prior_spec.json"
NATIVE_FEATURES_DIR: Final[Path] = OUT_ROOT / "native_features"
NATIVE_MANIFEST_PATH: Final[Path] = NATIVE_FEATURES_DIR / "features_manifest.json"
NATIVE_MONITOR_MANIFEST: Final[Path] = OUT_ROOT / "native_monitor_features" \
    / "features_manifest.json"
LEGACY_LABELS_DIR: Final[Path] = OUT_ROOT / "legacy_labels"
LEGACY_LABELS_AUDIT: Final[Path] = LEGACY_LABELS_DIR / "legacy_labels_audit.json"
LEGACY_FEATURES_DIR: Final[Path] = PROJECT_ROOT / (
    "outputs/pesn/dual_ts_v2_20261015/wp3_features_v41")
NATIVE_RECONNECT_AUDIT: Final[Path] = OUT_ROOT / "native_reconnect_audit.json"
GRADIENT_BASELINE_AUDIT: Final[Path] = OUT_ROOT / "gradient_baseline_audit.json"
PROTOCOL_FREEZE_PATH: Final[Path] = OUT_ROOT / "protocol_freeze.json"
LAMBDA_CANARY_PATH: Final[Path] = OUT_ROOT / "lambda_canary.json"
#: v4 (E1) 7-channel edge features (materialized by build_v4_edge_features.py).
V4_FEATURES_DIR: Final[Path] = OUT_ROOT / "features_v4"
V4_NATIVE_FIT_DIR: Final[Path] = V4_FEATURES_DIR / "native_fit"

#: Screening seeds and V4.7 final seeds (contract §8).
SCREEN_SEEDS: Final[tuple[int, ...]] = (20261017, 20261018)
FINAL_SEEDS: Final[tuple[int, ...]] = (20261017, 20261018, 20261019)

#: Stage time envelopes (minutes from T0; contract §8).
STAGES: Final[tuple[dict, ...]] = (
    {"id": "PRECHECK", "start": 0, "end": 35},
    {"id": "V4.2", "start": 35, "end": 120},
    {"id": "V4.3", "start": 120, "end": 205},
    {"id": "V4.4", "start": 205, "end": 290},
    {"id": "V4.5", "start": 290, "end": 385},
    {"id": "V4.6", "start": 385, "end": 455},
    {"id": "V4.7", "start": 455, "end": 550},
    {"id": "REVIEW", "start": 550, "end": 600},
)

#: Frozen prune order (delete whole double-seed arms; contract §8).
PRUNE_ORDER: Final[tuple[str, ...]] = ("W128", "V2", "E2")

MANIFEST_SCHEMA: Final[str] = "pesn_v42_campaign_manifest_v1"

#: Lane-D promotion metric names -> my dev-summary key.
LANE_D_KEYS: Final[dict[str, str]] = {
    "overall_cos": "direction_cosine_median",
    "near_ts_cos": "near_ts_cosine_median",
    "group_median_ratio": "mse_ratio_median",
    "better_than_zero_groups": "n_groups_ratio_below_one",
    "pooled_ratio": "pooled_mse_ratio",
    "near_zero_rmse_A": "near_zero_rmse_pooled_A",
}

# ---------------------------------------------------------------------------
# Arm registry (frozen; pre-registered upper bound, contract §8)
# ---------------------------------------------------------------------------
ARM_REGISTRY: Final[tuple[dict, ...]] = (
    {"arm_id": "D0", "stage": "V4.2", "seeds": SCREEN_SEEDS, "prunable": False,
     "config": {"dataset_kind": "legacy", "head": "bc_nll_legacy",
                "sampler": "group_balanced_v41", "preset": "legacy_v41"}},
    {"arm_id": "D1", "stage": "V4.2", "seeds": SCREEN_SEEDS, "prunable": False,
     "config": {"dataset_kind": "legacy", "head": "bc_nll_legacy",
                "sampler": "epoch_traversal_v41", "preset": "legacy_v41"}},
    {"arm_id": "D2", "stage": "V4.2", "seeds": SCREEN_SEEDS, "prunable": False,
     "config": {"dataset_kind": "legacy", "head": "displacement",
                "preset": "campaign_v1", "sampler_dev_winner": True}},
    {"arm_id": "D3", "stage": "V4.2", "seeds": SCREEN_SEEDS, "prunable": False,
     "config": {"dataset_kind": "native", "head": "displacement",
                "preset": "campaign_v1", "sampler_dev_winner": True}},
    {"arm_id": "H0", "stage": "V4.3", "seeds": (), "prunable": False,
     "reuse": "D3"},
    {"arm_id": "H1", "stage": "V4.3", "seeds": SCREEN_SEEDS, "prunable": False,
     "config": {"dataset_kind": "native", "head": "displacement",
                "preset": "campaign_v1", "use_prior_skip": True,
                "sampler_dev_winner": True}},
    {"arm_id": "H2", "stage": "V4.3", "seeds": SCREEN_SEEDS, "prunable": False,
     "parent": {"winner_of": ["H0", "H1"]},
     "config": {"lambda_dir": "CALIBRATED", "lambda_mag": "CALIBRATED"}},
    {"arm_id": "E0", "stage": "V4.4", "seeds": (), "prunable": False,
     "reuse": "V4.3_WINNER"},
    {"arm_id": "E1", "stage": "V4.4", "seeds": SCREEN_SEEDS, "prunable": False,
     "parent": {"winner_of": ["H0", "H1", "H2"]},
     "config": {"features_version": "v4"}},
    {"arm_id": "E2", "stage": "V4.4", "seeds": SCREEN_SEEDS, "prunable": True,
     "parent": "E1", "config": {"keypair_condition": True}},
    {"arm_id": "V0", "stage": "V4.5", "seeds": (), "prunable": False,
     "reuse": "V4.4_WINNER"},
    {"arm_id": "V1", "stage": "V4.5", "seeds": SCREEN_SEEDS, "prunable": False,
     "parent": {"winner_of": ["E1", "E2"]},
     "config": {"encoder": "painn", "vector_channels": 16}},
    {"arm_id": "V2", "stage": "V4.5", "seeds": SCREEN_SEEDS, "prunable": True,
     "parent": "V1", "config": {"contact_edges": True}},
    {"arm_id": "W64", "stage": "V4.6", "seeds": (), "prunable": False,
     "reuse": "V4.5_WINNER"},
    {"arm_id": "W96", "stage": "V4.6", "seeds": SCREEN_SEEDS, "prunable": False,
     "parent": {"winner_of": ["V1", "V2"]}, "config": {"hidden": 96}},
    {"arm_id": "W128", "stage": "V4.6", "seeds": SCREEN_SEEDS, "prunable": True,
     "parent": {"winner_of": ["V1", "V2"]}, "config": {"hidden": 128}},
    {"arm_id": "FINAL", "stage": "V4.7", "seeds": FINAL_SEEDS,
     "prunable": False,
     "parent": {"winner_of": ["W64", "W96", "W128"]},
     "config": {"dataset_kind": "native", "final_fit50": True}},
)

ARMS: Final[dict[str, dict]] = {a["arm_id"]: a for a in ARM_REGISTRY}


#: Representative architecture set for timing calibration.  One entry per
#: distinct architectural FORM (D0/D1 legacy heads are skipped: historical
#: 346–427 s receipts already bound their wall time).  Each entry is D3's base
#: (native + displacement + campaign_v1) plus an explicit increment; the
#: sampler is provisionally fixed to group_balanced_v41 (batch shape is
#: sampler-independent, so wall time is unaffected by the D0/D1 promotion).
CALIBRATION_ARCHS: Final[tuple[dict, ...]] = (
    {"key": "egnn_disp", "features_version": "v3", "encoder": "egnn",
     "contact_edges": False, "keypair_condition": False, "hidden": 64},
    {"key": "v4_7ch", "features_version": "v4", "encoder": "egnn",
     "contact_edges": False, "keypair_condition": False, "hidden": 64},
    {"key": "keypair", "features_version": "v4", "encoder": "egnn",
     "contact_edges": False, "keypair_condition": True, "hidden": 64},
    {"key": "painn", "features_version": "v4", "encoder": "painn",
     "contact_edges": False, "keypair_condition": False, "hidden": 64},
    {"key": "painn_contact", "features_version": "v4", "encoder": "painn",
     "contact_edges": True, "keypair_condition": False, "hidden": 64},
    {"key": "hidden96", "features_version": "v4", "encoder": "painn",
     "contact_edges": False, "keypair_condition": False, "hidden": 96},
    {"key": "hidden128", "features_version": "v4", "encoder": "painn",
     "contact_edges": False, "keypair_condition": False, "hidden": 128},
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _dataset_spec(kind: str, features_version: str = "v3") -> CampaignDatasetSpec:
    if kind == "legacy":
        return CampaignDatasetSpec(
            kind="legacy", features_dir=str(LEGACY_FEATURES_DIR),
            labels_dir=str(LEGACY_LABELS_DIR), split_path=str(DEV_SPLIT_PATH))
    # E1+ arms (features_version="v4") consume the materialized 7-channel
    # edge shards; D3/H0/H1/H2 (v3) consume the native reconnected shards.
    features_dir = (V4_NATIVE_FIT_DIR if features_version == "v4"
                    else NATIVE_FEATURES_DIR)
    return CampaignDatasetSpec(
        kind="native", features_dir=str(features_dir),
        labels_dir="", split_path=str(DEV_SPLIT_PATH))


def _merge_config(base: dict, overrides: dict) -> dict:
    merged = dict(base)
    for key, value in overrides.items():
        if value != "CALIBRATED":
            merged[key] = value
    return merged


def _resolve_parent(arm_id: str, parent, decisions: dict) -> str:
    if isinstance(parent, str):
        return parent
    winners = parent.get("winner_of", []) if isinstance(parent, dict) else []
    if not winners:
        raise SystemExit(f"REFUSED: arm {arm_id} has an unresolved parent")
    winner = decisions.get("winner_of", {}).get(arm_id)
    if winner is None:
        raise SystemExit(
            f"REFUSED: arm {arm_id} requires a promotion winner over "
            f"{winners} before it can be resolved")
    return winner


def _flat_config(arm: CampaignArmConfig) -> dict:
    return {
        "dataset_kind": arm.dataset.kind,
        "head": arm.head,
        "use_prior_skip": arm.use_prior_skip,
        "lambda_dir": arm.lambda_dir,
        "lambda_mag": arm.lambda_mag,
        "features_version": arm.features_version,
        "encoder": arm.encoder,
        "vector_channels": arm.vector_channels,
        "contact_edges": arm.contact_edges,
        "hidden": arm.hidden,
        "n_layers": arm.n_layers,
        "sampler": arm.sampler,
        "preset": arm.preset,
        "ema_read": arm.ema_read,
        "keypair_condition": arm.keypair_condition,
        "updates": arm.updates,
        "final_fit50": arm.final_fit50,
    }


def _load_pruned_arms() -> set[str]:
    """Arms manually pruned per the pre-registered order (contract §8).

    ``pruned_arms.json`` (list under ``pruned``) is an auditable operator
    action used only when the timing projection shows an arm cannot fit its
    envelope; it never encodes performance verdicts.
    """
    path = OUT_ROOT / "pruned_arms.json"
    if not path.is_file():
        return set()
    try:
        payload = json.loads(path.read_text())
        return set(payload.get("pruned", []))
    except Exception:
        return set()


def _configs_key(key: str, decisions: dict) -> str:
    """Map a parent/reuse key to a CONCRETE arm id that has a configs entry.

    Reuse aliases and stage-winner labels (E0, V4.3_WINNER, ...) never hold a
    ``configs`` line of their own; resolving them through the run-arm alias
    chain lands on the concrete arm (e.g. E0 -> H0 -> D3) whose frozen config
    is the correct base.  Falls back to the raw key when no alias resolves.
    """
    if key in decisions.get("configs", {}):
        return key
    try:
        return _resolve_run_arm(key, decisions)
    except SystemExit:
        return key


def _resolve_arm(entry: dict, decisions: dict, *, seed: int) -> CampaignArmConfig:
    cfg = dict(entry.get("config") or {})
    reuse = entry.get("reuse")
    parent = entry.get("parent")
    base: dict = {}
    if reuse is not None:
        base = decisions.get("configs", {}).get(
            _configs_key(reuse, decisions), {})
    elif parent is not None:
        parent_id = _resolve_parent(entry["arm_id"], parent, decisions)
        base = decisions.get("configs", {}).get(
            _configs_key(parent_id, decisions), {})
    merged = _merge_config(base, cfg)
    # Direct driver-injected config for THIS arm (e.g. H2's calibrated
    # lambdas, set in run_campaign) takes precedence over the registry/base
    # merge — the registry H2 slot carries "CALIBRATED" placeholders that
    # _merge_config skips, so without this overlay H2 would resolve with
    # lambda_dir=lambda_mag=None (a silent H0 replica).
    own = decisions.get("configs", {}).get(entry["arm_id"])
    if own:
        merged.update(dict(own))
    merged.setdefault("dataset_kind", "legacy")
    merged.setdefault("head", "displacement")
    merged.setdefault("sampler", "group_balanced_v41")
    merged.setdefault("preset", "campaign_v1")
    if cfg.get("sampler_dev_winner"):
        # ``sampler_winner`` is None until the D0/D1 promotion; ``.get(k, d)``
        # returns None (the key exists), so use ``or`` to fall back to the
        # group-balanced default.
        merged["sampler"] = decisions.get("sampler_winner") \
            or "group_balanced_v41"
    return CampaignArmConfig(
        arm_id=entry["arm_id"],
        dataset=_dataset_spec(merged["dataset_kind"],
                              features_version=merged.get("features_version", "v3")),
        head=merged["head"],
        use_prior_skip=bool(merged.get("use_prior_skip", False)),
        lambda_dir=merged.get("lambda_dir"),
        lambda_mag=merged.get("lambda_mag"),
        features_version=merged.get("features_version", "v3"),
        encoder=merged.get("encoder", "egnn"),
        vector_channels=int(merged.get("vector_channels", 16)),
        contact_edges=bool(merged.get("contact_edges", False)),
        hidden=int(merged.get("hidden", 64)),
        n_layers=int(merged.get("n_layers", 3)),
        sampler=merged["sampler"],
        preset=merged["preset"],
        seed=seed,
        ema_read=merged.get("ema_read", "raw"),
        keypair_condition=bool(merged.get("keypair_condition", False)),
        final_fit50=bool(merged.get("final_fit50", False)),
    )


def _run_dir(arm_id: str, seed: int) -> Path:
    return OUT_ROOT / "run_records" / f"{arm_id}_seed{seed}"


#: Stage-winner reuse labels -> the concrete arm whose ``winner_of`` slot
#: carries that stage's winner (the arm that INHERITS the stage winner).
_STAGE_WINNER_OF: Final[dict[str, str]] = {
    "V4.3_WINNER": "E1",
    "V4.4_WINNER": "V1",
    "V4.5_WINNER": "W96",
}


def _resolve_run_arm(arm_id: str, decisions: dict,
                     registry: dict = ARMS) -> str:
    """Resolve an arm id (or a reuse label) to the CONCRETE arm whose run
    directories hold the dev metrics.

    Follows ``reuse`` (a literal arm id or a ``V4.x_WINNER`` stage label) and
    ``winner_of`` indirection with cycle protection, so reuse arms (H0 → D3,
    E0/V0/W64 → their stage winners) resolve to the arm that actually has
    ``run_records/<arm>_seed*/``.  Raises ``SystemExit`` with the full chain
    on any unresolved reference.
    """
    chain: list[str] = []
    cur = arm_id
    while True:
        if cur in chain:
            raise SystemExit(
                f"REFUSED: reuse/winner cycle while resolving {arm_id!r}: "
                f"chain={chain + [cur]!r}")
        chain.append(cur)
        entry = registry.get(cur)
        if entry is None:
            # a stage-winner label maps to the arm that stores that winner
            holder = _STAGE_WINNER_OF.get(cur)
            if holder is None:
                raise SystemExit(
                    f"REFUSED: cannot resolve {arm_id!r}: unknown arm id or "
                    f"stage label {cur!r}; chain={chain!r}")
            winner = decisions.get("winner_of", {}).get(holder)
            if winner is None:
                raise SystemExit(
                    f"REFUSED: cannot resolve {arm_id!r}: stage label "
                    f"{cur!r} (holder {holder!r}) has no winner recorded; "
                    f"chain={chain!r}")
            cur = winner
            continue
        reuse = entry.get("reuse")
        if reuse is not None:
            cur = reuse  # literal arm id, or a stage label (handled next loop)
            continue
        if entry.get("seeds"):
            return cur  # concrete arm with its own run directories
        raise SystemExit(
            f"REFUSED: cannot resolve {arm_id!r}: arm {cur!r} has no runs and "
            f"no reuse target; chain={chain!r}")


def _calibration_arm(spec: dict) -> CampaignArmConfig:
    """One timing-probe arm from a representative spec (D3 base + increment).

    The sampler is provisionally ``group_balanced_v41`` (declared via the
    ``provisional_fields`` marker on the calibration record); the architecture
    fields (features_version/encoder/contact_edges/keypair/hidden) are the
    explicit registry increments so the probe exercises exactly the target
    compute graph.
    """
    features_version = str(spec.get("features_version", "v3"))
    return CampaignArmConfig(
        arm_id=f"TIMING_{spec['key']}",
        dataset=_dataset_spec("native", features_version=features_version),
        head="displacement",
        use_prior_skip=False,
        lambda_dir=None, lambda_mag=None,
        features_version=features_version,
        encoder=str(spec.get("encoder", "egnn")),
        vector_channels=16,
        contact_edges=bool(spec.get("contact_edges", False)),
        hidden=int(spec.get("hidden", 64)),
        n_layers=3,
        sampler="group_balanced_v41",   # provisional (timing-only)
        preset="campaign_v1",
        seed=SCREEN_SEEDS[0],
        ema_read="raw",
        keypair_condition=bool(spec.get("keypair_condition", False)),
    )


def _is_complete(record: dict, arm: CampaignArmConfig) -> bool:
    if record.get("schema_version") != RUN_RECORD_V3_SCHEMA:
        return False
    if record.get("timing_only"):
        return False
    if (record.get("budget_state") or {}).get("status") == "budget_censored":
        return True
    return int(record.get("accounting", {}).get("updates_run", 0)) >= arm.updates


def _env_payload() -> dict:
    torch_version = "unavailable"
    cuda = "unavailable"
    gpu_name = "unavailable"
    try:
        import torch
        torch_version = torch.__version__
        cuda = torch.version.cuda or "cpu"
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
    except Exception:
        pass
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch_version,
        "cuda": cuda,
        "gpu_name": gpu_name,
    }


def _data_hashes() -> dict:
    hashes = {}
    for name, path in (("dev_split", DEV_SPLIT_PATH),
                       ("physical_prior", PHYSICAL_PRIOR_PATH),
                       ("native_manifest", NATIVE_MANIFEST_PATH),
                       ("native_monitor_manifest", NATIVE_MONITOR_MANIFEST),
                       ("legacy_labels_audit", LEGACY_LABELS_AUDIT),
                       ("native_reconnect_audit", NATIVE_RECONNECT_AUDIT),
                       ("gradient_baseline_audit", GRADIENT_BASELINE_AUDIT),
                       ("protocol_freeze", PROTOCOL_FREEZE_PATH),
                       ("lambda_canary", LAMBDA_CANARY_PATH)):
        hashes[name] = sha256_file(path) if Path(path).is_file() else None
    return hashes


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

class CampaignDriver:
    def __init__(self, *, out_root: Path = OUT_ROOT) -> None:
        self.out_root = out_root
        self.out_root.mkdir(parents=True, exist_ok=True)
        self.manifest_path = out_root / "campaign_manifest.json"
        self.decisions_path = out_root / "stage_decisions.jsonl"
        self.calibration_path = out_root / "timing_calibration.json"
        self.decisions: dict = {"sampler_winner": None,
                                "winner_of": {}, "configs": {}}
        # one-shot decision events already recorded (e.g. prior_not_supported,
        # forty_k_gate) — used for cross-restart replay dedup.
        self._recorded_events: set = set()
        self.manifest: dict | None = None
        self._load_state()

    # -- state (crash-restart) ----------------------------------------------

    def _load_state(self) -> None:
        if self.manifest_path.is_file():
            self.manifest = json.loads(self.manifest_path.read_text())
        if self.decisions_path.is_file():
            for line in self.decisions_path.read_text().splitlines():
                if line.strip():
                    self._ingest_decision(json.loads(line))

    def _ingest_decision(self, decision: dict) -> None:
        if decision.get("sampler_winner"):
            self.decisions["sampler_winner"] = decision["sampler_winner"]
        for key in ("configs", "winner_of"):
            for k, v in (decision.get(key) or {}).items():
                self.decisions[key][k] = v
        if decision.get("stage_winner") and decision.get("winner"):
            self.decisions["configs"][
                f"{decision['stage']}_WINNER"] = decision["winner_config"]
        if "decision" in decision and "arm" in decision:
            self._recorded_events.add(
                (str(decision["arm"]), str(decision["decision"])))

    def _decision_already_applied(self, decision: dict) -> bool:
        """True iff every semantic key of ``decision`` is already recorded.

        Dedup is on the idempotent keys only (sampler_winner / winner_of /
        configs / stage winner / one-shot decision events); volatile per-run
        detail (promotion clauses, forty_k_gate numbers) is ignored for
        identity so a replay never changes the recorded semantics.
        """
        checks: list[bool] = []
        if "sampler_winner" in decision:
            checks.append(
                self.decisions["sampler_winner"] == decision["sampler_winner"])
        for k, v in (decision.get("winner_of") or {}).items():
            checks.append(self.decisions["winner_of"].get(k) == v)
        for k, v in (decision.get("configs") or {}).items():
            checks.append(self.decisions["configs"].get(k) == v)
        if decision.get("stage_winner") and decision.get("stage"):
            checks.append(
                self.decisions["configs"].get(f"{decision['stage']}_WINNER")
                == decision.get("winner_config"))
        if "decision" in decision and "arm" in decision:
            checks.append(
                (str(decision["arm"]), str(decision["decision"]))
                in self._recorded_events)
        if not checks:
            return False  # nothing semantic to dedup -> always write
        return all(checks)

    def _append_decision(self, decision: dict) -> bool:
        """Record a decision idempotently; returns True if written, False if
        it was a cross-restart replay of an already-applied decision."""
        if self._decision_already_applied(decision):
            return False
        with self.decisions_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(decision, sort_keys=True) + "\n")
        self._ingest_decision(decision)
        return True

    # -- manifest ------------------------------------------------------------

    def ensure_manifest(self, *, contract_sha256: str = "",
                        protocol_sha256: str = "") -> dict:
        if self.manifest is None:
            self.manifest = {
                "schema_version": MANIFEST_SCHEMA,
                "T0_unix": time.time(),
                "T0_iso": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()),
                "protocol_sha256": protocol_sha256,
                "contract_sha256": contract_sha256,
                "seeds": {"screening": list(SCREEN_SEEDS),
                          "final": list(FINAL_SEEDS)},
                "arm_table": [a["arm_id"] for a in ARM_REGISTRY],
                "prune_order": list(PRUNE_ORDER),
                "environment": _env_payload(),
                "data_hashes": _data_hashes(),
            }
            write_json(self.manifest_path, self.manifest)
        return self.manifest

    # -- precheck (product existence + counts + H1 gate) --------------------

    def precheck(self) -> dict:
        missing = [name for name, path in
                   (("dev_split", DEV_SPLIT_PATH),
                    ("physical_prior", PHYSICAL_PRIOR_PATH),
                    ("native_manifest", NATIVE_MANIFEST_PATH),
                    ("native_monitor_manifest", NATIVE_MONITOR_MANIFEST),
                    ("legacy_labels", LEGACY_LABELS_DIR),
                    ("native_reconnect_audit", NATIVE_RECONNECT_AUDIT),
                    ("gradient_baseline_audit", GRADIENT_BASELINE_AUDIT),
                    ("protocol_freeze", PROTOCOL_FREEZE_PATH))
                   if not Path(path).exists()]
        counts = {}
        problems = list(missing)

        # -- native fit counts (5668/5568/100) -------------------------------
        if NATIVE_MANIFEST_PATH.is_file():
            fm = json.loads(NATIVE_MANIFEST_PATH.read_text())
            pop = fm.get("population", {})
            counts["native_fit"] = {
                "states": int(pop.get("states_native", -1)),
                "actions": int(pop.get("actions_native", -1)),
                "endpoints": int(pop.get("teacher_endpoint_states", -1)),
            }
            if (counts["native_fit"]["states"], counts["native_fit"]["actions"],
                    counts["native_fit"]["endpoints"]) != (5668, 5568, 100):
                problems.append("native_fit_count_mismatch")
        # -- native monitor counts (931/915/16) ------------------------------
        if NATIVE_MONITOR_MANIFEST.is_file():
            mm = json.loads(NATIVE_MONITOR_MANIFEST.read_text())
            pop = mm.get("population", {})
            counts["native_monitor"] = {
                "states": int(pop.get("states_native", -1)),
                "actions": int(pop.get("actions_native", -1)),
                "endpoints": int(pop.get("teacher_endpoint_states", -1)),
            }
            if (counts["native_monitor"]["states"],
                    counts["native_monitor"]["actions"],
                    counts["native_monitor"]["endpoints"]) != (931, 915, 16):
                problems.append("native_monitor_count_mismatch")
        # -- legacy labels (5967 actions) ------------------------------------
        if LEGACY_LABELS_AUDIT.is_file():
            la = json.loads(LEGACY_LABELS_AUDIT.read_text())
            counts["legacy_actions"] = int(sum(
                int(r.get("n_actions", 0)) for r in la.get("per_reaction", [])))
            if counts["legacy_actions"] != 5967:
                problems.append("legacy_actions_count_mismatch")
        # -- dev split (5 groups) --------------------------------------------
        if DEV_SPLIT_PATH.is_file():
            ds = json.loads(DEV_SPLIT_PATH.read_text())
            counts["dev_groups"] = len(ds.get("dev_group_ids", []))
            if counts["dev_groups"] != 5:
                problems.append("dev_split_group_count_mismatch")
        # -- H1 release gate --------------------------------------------------
        h1_released = None
        if GRADIENT_BASELINE_AUDIT.is_file():
            ga = json.loads(GRADIENT_BASELINE_AUDIT.read_text())
            h1 = ga.get("h1_release") or {}
            h1_released = bool(h1.get("released", False))
            counts["h1_release"] = h1.get("value", h1_released)

        return {"ok": not problems, "missing": missing,
                "problems": problems, "counts": counts,
                "h1_released": h1_released,
                "hashes": _data_hashes()}

    # -- timing calibration --------------------------------------------------

    def calibrate(self, *, specs: tuple[dict, ...] = CALIBRATION_ARCHS) -> dict:
        calibration = (json.loads(self.calibration_path.read_text())
                       if self.calibration_path.is_file() else {})
        for spec in specs:
            key = str(spec["key"])
            if key in calibration:
                continue  # already calibrated on a prior (crashed) run
            arm = _calibration_arm(spec)
            record = self._spawn(arm, _run_dir(f"TIMING_{key}", arm.seed),
                                 timing_only=True)
            estimate = dict(record.get("estimate", {}))
            estimate["provisional_fields"] = ["sampler"]
            calibration[key] = estimate
            # incremental write: a crash mid-calibrate preserves completed
            # specs and leaves a valid (partial) timing_calibration.json.
            write_json(self.calibration_path, calibration)
        return calibration

    # -- subprocess ----------------------------------------------------------

    def _spawn(self, arm: CampaignArmConfig, run_dir: Path, *,
               timing_only: bool = False, deadline: float | None = None,
               prior_spec: Path | None = None) -> dict:
        run_dir.mkdir(parents=True, exist_ok=True)
        config_path = run_dir / "arm_config.json"
        write_json(config_path, arm.to_payload())
        cmd = [PYTHON, "-m",
               "pes2ts_core.pesn.offline.training.dual_ts_campaign",
               "--arm-config", str(config_path), "--out", str(run_dir)]
        if timing_only:
            cmd.append("--timing-only")
        if deadline is not None:
            cmd += ["--deadline", repr(float(deadline))]
        if prior_spec is not None:
            cmd += ["--prior-spec", str(prior_spec)]
        proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
        if proc.returncode != 0:
            raise SystemExit(
                f"subprocess failed for arm {arm.arm_id} seed {arm.seed} "
                f"(rc={proc.returncode}): {' '.join(cmd)}")
        record_path = run_dir / "run_record.json"
        if not record_path.is_file():
            raise SystemExit(f"subprocess produced no run_record for "
                             f"{arm.arm_id} seed {arm.seed}")
        return json.loads(record_path.read_text())

    def _run_or_skip(self, arm: CampaignArmConfig) -> dict:
        run_dir = _run_dir(arm.arm_id, arm.seed)
        rec_path = run_dir / "run_record.json"
        if rec_path.is_file():
            rec = json.loads(rec_path.read_text())
            if _is_complete(rec, arm):
                return rec
        return None

    def _spawn_if_needed(self, arm: CampaignArmConfig, *,
                         deadline: float | None = None,
                         prior_spec: Path | None = None) -> dict:
        done = self._run_or_skip(arm)
        if done is not None:
            return done
        return self._spawn(arm, _run_dir(arm.arm_id, arm.seed),
                           deadline=deadline, prior_spec=prior_spec)

    # -- promotion (§9.1, Lane D) --------------------------------------------

    def _lane_d_dev(self, arm_id: str) -> dict:
        """Per-seed Lane-D metrics dict for an arm (from its run records).

        Resolves reuse/winner-of aliases (H0→D3, E0/V0/W64→stage winners) to
        the concrete arm holding the run directories before reading them.
        """
        concrete = _resolve_run_arm(arm_id, self.decisions, ARMS)
        entry = ARMS[concrete]
        out = {}
        for seed in entry["seeds"]:
            rec_path = _run_dir(concrete, seed) / "run_record.json"
            if not rec_path.is_file():
                continue
            rec = json.loads(rec_path.read_text())
            last = list(rec.get("dev_evals", {}).values())
            if not last:
                continue
            summary = last[-1].get("summary", {})
            out[seed] = {ld: summary.get(my) for ld, my in LANE_D_KEYS.items()}
        return out

    def apply_promotion(self, parent_id: str, child_id: str,
                        parent_label: str) -> str:
        """§9.1 pairwise child-vs-parent; returns the retained arm id."""
        from pes2ts_core.pesn.offline.evaluation.dual_ts_promotion import (
            apply_promotion_rule)
        parent_dev = self._lane_d_dev(parent_id)
        child_dev = self._lane_d_dev(child_id)
        seeds = sorted(set(parent_dev) & set(child_dev))
        decision = apply_promotion_rule(parent_dev, child_dev, seeds)
        winner = child_id if decision.get("promote") else parent_id
        self._append_decision({
            "winner_of": {parent_label: winner},
            "configs": {winner: self._config_payload(winner)},
            "promotion": decision,
        })
        return winner

    def apply_sampler_promotion(self, candidate_ids: list[str]) -> str:
        from pes2ts_core.pesn.offline.evaluation.dual_ts_promotion import (
            apply_promotion_rule)
        parent_dev = self._lane_d_dev(candidate_ids[0])
        child_dev = self._lane_d_dev(candidate_ids[1])
        seeds = sorted(set(parent_dev) & set(child_dev))
        decision = apply_promotion_rule(parent_dev, child_dev, seeds)
        winner = candidate_ids[1] if decision.get("promote") \
            else candidate_ids[0]
        sampler = ARMS[winner]["config"]["sampler"]
        self._append_decision({"sampler_winner": sampler,
                               "sampler_promotion": decision})
        return sampler

    def _config_payload(self, arm_id: str) -> dict:
        entry = ARMS[arm_id]
        seed = entry["seeds"][0] if entry["seeds"] else SCREEN_SEEDS[0]
        return _flat_config(_resolve_arm(entry, self.decisions, seed=seed))

    def record_stage_winner(self, stage: str, winner: str) -> None:
        self._append_decision({
            "stage": stage, "winner": winner, "stage_winner": True,
            "winner_config": self._config_payload(winner),
        })

    def _record_config(self, arm_id: str, arm: CampaignArmConfig) -> None:
        if arm_id not in self.decisions["configs"]:
            self._append_decision({"configs": {arm_id: _flat_config(arm)}})

    # -- stage execution -----------------------------------------------------

    def run_arm_ids(self, arm_ids: list[str], *, deadline: float,
                    prior_spec: Path | None = None) -> list[dict]:
        records = []
        for arm_id in arm_ids:
            if arm_id in _load_pruned_arms():
                # Pre-registered prune (contract §8 order: W128 → V2 → E2),
                # executed when the timing projection shows an arm cannot
                # fit its envelope.  Typed, auditable, never a performance
                # verdict.
                self._append_decision({
                    "decision": "pruned_per_prereg_order",
                    "arm": arm_id,
                    "source": "pruned_arms.json",
                    "prune_order": list(PRUNE_ORDER),
                })
                print(f"[campaign] arm={arm_id} PRUNED per prereg order "
                      f"(pruned_arms.json)", flush=True)
                continue
            entry = ARMS[arm_id]
            for seed in entry["seeds"]:
                arm = _resolve_arm(entry, self.decisions, seed=seed)
                self._record_config(arm_id, arm)
                records.append(self._spawn_if_needed(
                    arm, deadline=deadline, prior_spec=prior_spec))
        return records


def _stage_deadline(t0: float, stage_id: str) -> float:
    for stage in STAGES:
        if stage["id"] == stage_id:
            return t0 + stage["end"] * 60.0
    raise KeyError(stage_id)


def _resolve_lambdas() -> tuple[float | None, float | None]:
    """Resolve H2's CALIBRATED lambdas from the canary artifact."""
    if not LAMBDA_CANARY_PATH.is_file():
        raise SystemExit("REFUSED: H2 requires lambda_canary.json "
                         "(run scripts/calibrate_v42_lambdas.py first)")
    canary = json.loads(LAMBDA_CANARY_PATH.read_text())
    return float(canary["lambda_dir"]), float(canary["lambda_mag"])


def _winner_dev_curves(winner_id: str, decisions: dict) -> dict:
    """{seed: {u10000: {...}, u20000: {...}}} Lane-D curves for the 40k gate."""
    concrete = _resolve_run_arm(winner_id, decisions, ARMS)
    entry = ARMS[concrete]
    curves = {}
    for seed in entry["seeds"]:
        rec_path = _run_dir(concrete, seed) / "run_record.json"
        if not rec_path.is_file():
            continue
        rec = json.loads(rec_path.read_text())
        dev_evals = rec.get("dev_evals", {})
        curve = {}
        for upd in ("u10000", "u20000"):
            if upd in dev_evals:
                summary = dev_evals[upd].get("summary", {})
                curve[upd] = {
                    ld: summary.get(my) for ld, my in LANE_D_KEYS.items()}
        if "u10000" in curve and "u20000" in curve:
            curves[seed] = curve
    return curves


def _forty_k_timing(calibration_key: str, calibration: dict) -> dict:
    """Estimate seed_40k_seconds from the timing calibration (best-effort)."""
    estimate = calibration.get(calibration_key) or {}
    steps_per_sec = float(estimate.get("steps_per_sec", 0.0) or 0.0)
    eval_seconds = float(estimate.get("eval_seconds", 0.0) or 0.0)
    if steps_per_sec > 0:
        seed_seconds = 40000.0 / steps_per_sec + 4.0 * eval_seconds
    else:
        seed_seconds = float("inf")
    return {"seed_40k_seconds": seed_seconds,
            "n_seeds": 3, "budget_minutes": 95.0}


def _calibration_key_for_arm(arm_id: str, decisions: dict) -> str:
    """Map a V4.6 winner arm to its representative calibration key."""
    if arm_id == "W96":
        return "hidden96"
    if arm_id == "W128":
        return "hidden128"
    # W64 reuses the V4.5 winner (V1=painn / V2=painn_contact)
    v45 = decisions.get("winner_of", {}).get("W96")
    return "painn_contact" if v45 == "V2" else "painn"


def _a_s_fit50() -> float:
    """Re-estimate a_s on the complete fit50 native population (contract §6)."""
    split = json.loads(DEV_SPLIT_PATH.read_text())
    all_groups = sorted(set(split.get("train_group_ids", []))
                        | set(split.get("dev_group_ids", [])))
    states, _meta = load_campaign_states(
        CampaignDatasetSpec(kind="native", features_dir=str(NATIVE_FEATURES_DIR)),
        group_ids=all_groups)
    return compute_a_s([s for s in states if not s["is_teacher"]])


def run_campaign(driver: CampaignDriver, *, contract_sha256: str,
                 protocol_sha256: str) -> None:
    # Resume detection: a pre-existing manifest (written on an earlier crash)
    # already froze T0 — reuse it verbatim, never reset the clock.
    resumed = driver.manifest is not None
    manifest = driver.ensure_manifest(contract_sha256=contract_sha256,
                                      protocol_sha256=protocol_sha256)
    if resumed:
        print(f"[campaign] RESUME: reusing manifest "
              f"T0={manifest.get('T0_iso')} "
              f"(unix={manifest.get('T0_unix')}); clock NOT reset", flush=True)
    precheck = driver.precheck()
    if not precheck["ok"]:
        raise SystemExit("REFUSED: precheck failed — problems: "
                         + ", ".join(precheck["problems"]))
    t0 = float(manifest["T0_unix"])

    # timing calibration for the distinct architectural forms (idempotent)
    driver.calibrate()

    # V4.2: sampler attribution + native/unbounded migration
    driver.run_arm_ids(["D0", "D1"],
                       deadline=_stage_deadline(t0, "V4.2"))
    driver.apply_sampler_promotion(["D0", "D1"])
    driver.run_arm_ids(["D2", "D3"],
                       deadline=_stage_deadline(t0, "V4.2"))

    # V4.3: physical-residual + auxiliary objectives
    driver.decisions["configs"]["H0"] = driver._config_payload("D3")
    h1_released = precheck.get("h1_released")
    if h1_released is False:
        # H1 pre-registered gate failed: prior_not_supported → skip H1 (two
        # runs, no budget); H2 inherits H0 directly.
        driver._append_decision({
            "decision": "prior_not_supported", "arm": "H1",
            "source": precheck["hashes"].get("gradient_baseline_audit"),
        })
        driver.decisions["winner_of"]["H2"] = "H0"
        v43_base = "H0"
        v43_candidates = ["H0", "H2"]
    else:
        driver.run_arm_ids(["H1"], deadline=_stage_deadline(t0, "V4.3"),
                           prior_spec=PHYSICAL_PRIOR_PATH)
        v43_base = driver.apply_promotion("H0", "H1", parent_label="H2")
        v43_candidates = ["H0", "H1", "H2"]

    # resolve H2 lambdas then run H2 (parent = v43_base)
    lam_dir, lam_mag = _resolve_lambdas()
    driver.decisions["configs"]["H2"] = dict(
        driver.decisions["configs"].get(v43_base, {}))
    driver.decisions["configs"]["H2"]["lambda_dir"] = lam_dir
    driver.decisions["configs"]["H2"]["lambda_mag"] = lam_mag
    # HARD GATE: H2 must carry the calibrated lambdas — never silently
    # mislabel an H0 replica as H2 (previous bug).  Fail loudly if λ is None
    # or does not match the canary.
    _h2_probe = _resolve_arm(ARMS["H2"], driver.decisions,
                             seed=SCREEN_SEEDS[0])
    if _h2_probe.lambda_dir is None or _h2_probe.lambda_mag is None:
        raise SystemExit(
            "REFUSED: H2 resolved with lambda_dir/lambda_mag=None — λ "
            "injection failed; aborting instead of silently mislabeling H0 "
            "as H2")
    if abs(_h2_probe.lambda_dir - lam_dir) > 1e-9 \
            or abs(_h2_probe.lambda_mag - lam_mag) > 1e-9:
        raise SystemExit(
            f"REFUSED: H2 lambda mismatch — resolved "
            f"({_h2_probe.lambda_dir}, {_h2_probe.lambda_mag}) != canary "
            f"({lam_dir}, {lam_mag})")
    driver.run_arm_ids(["H2"], deadline=_stage_deadline(t0, "V4.3"),
                       prior_spec=PHYSICAL_PRIOR_PATH)
    v43_winner = driver.apply_promotion(v43_base, "H2", parent_label="E1")
    driver.record_stage_winner("V4.3", v43_winner)

    # V4.4: physical inner-products + sparse keypair condition
    # §9.1 strict: each candidate must beat the CURRENT winner (starting
    # from the parent E0); children never win by pairwise head-to-head.
    driver.run_arm_ids(["E1"], deadline=_stage_deadline(t0, "V4.4"))
    driver.run_arm_ids(["E2"], deadline=_stage_deadline(t0, "V4.4"))
    v44_winner = driver.apply_promotion("E0", "E1", parent_label="V1")
    v44_winner = driver.apply_promotion(v44_winner, "E2", parent_label="V1")
    driver.record_stage_winner("V4.4", v44_winner)

    # V4.5: vector channels + contact edges (same sequential-parent rule)
    driver.run_arm_ids(["V1"], deadline=_stage_deadline(t0, "V4.5"))
    driver.run_arm_ids(["V2"], deadline=_stage_deadline(t0, "V4.5"))
    v45_winner = driver.apply_promotion(v44_winner, "V1", parent_label="W96")
    if "V2" not in _load_pruned_arms():
        v45_winner = driver.apply_promotion(v45_winner, "V2",
                                            parent_label="W96")
    driver.decisions["winner_of"]["W128"] = v45_winner
    driver.record_stage_winner("V4.5", v45_winner)

    # V4.6: width sweep
    driver.run_arm_ids(["W96", "W128"], deadline=_stage_deadline(t0, "V4.6"))
    v46_winner = driver.apply_promotion("W64", "W96", parent_label="FINAL")
    # W128 vs the running winner (width ordering: ratio → cos → params)
    if "W128" not in _load_pruned_arms():
        v46_winner = driver.apply_promotion(v46_winner, "W128",
                                            parent_label="FINAL")
    driver.record_stage_winner("V4.6", v46_winner)

    # V4.7: final three-seed re-fit on full fit50 native (contract §9.2)
    a_s_fit50 = _a_s_fit50()
    driver.manifest["a_s_fit50"] = a_s_fit50
    write_json(driver.manifest_path, driver.manifest)

    # §9.2 40k gate on the selected recipe's 10k→20k curves + timing
    calibration = (json.loads(driver.calibration_path.read_text())
                   if driver.calibration_path.is_file() else {})
    winner_curves = _winner_dev_curves(v46_winner, driver.decisions)
    calib_key = _calibration_key_for_arm(v46_winner, driver.decisions)
    timing = _forty_k_timing(calib_key, calibration)
    from pes2ts_core.pesn.offline.evaluation.dual_ts_promotion import (
        apply_40k_gate)
    gate = apply_40k_gate(winner_curves, timing) if winner_curves \
        else {"allow_40k": False, "decision": "STAY_20K",
              "reasons": ["no dev curves"]}
    allow_40k = bool(gate.get("allow_40k"))
    driver._append_decision({
        "decision": gate.get("decision"), "arm": "FINAL",
        "forty_k_gate": gate, "a_s_fit50": a_s_fit50,
    })
    final_entry = ARMS["FINAL"]
    # override updates / eval_points on the FINAL config
    driver.decisions["configs"]["FINAL"] = dict(
        driver.decisions["configs"].get(v46_winner, {}))
    driver.decisions["configs"]["FINAL"].update({
        "dataset_kind": "native", "final_fit50": True,
        "updates": 40000 if allow_40k else 20000,
    })
    if allow_40k:
        driver.decisions["_final_eval_points"] = (5000, 10000, 20000, 40000)
    else:
        driver.decisions["_final_eval_points"] = (5000, 10000, 20000)

    final_runs = []
    for seed in final_entry["seeds"]:
        arm = _resolve_arm(final_entry, driver.decisions, seed=seed)
        from dataclasses import replace
        arm = replace(
            arm,
            updates=driver.decisions["configs"]["FINAL"]["updates"],
            eval_points=tuple(driver.decisions["_final_eval_points"]))
        final_runs.append(driver._spawn_if_needed(
            arm, deadline=_stage_deadline(t0, "V4.7"),
            prior_spec=PHYSICAL_PRIOR_PATH))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=OUT_ROOT)
    ap.add_argument("--contract-sha256", default="")
    ap.add_argument("--protocol-sha256", default="")
    ap.add_argument("--dry-run", action="store_true",
                    help="precheck + manifest only; no training")
    args = ap.parse_args(argv)

    driver = CampaignDriver(out_root=args.out)
    if args.dry_run:
        # Dry-run MUST NOT freeze T0: report only, never write the manifest
        # (nor touch protocol_freeze/dev_split/etc.).  If a manifest already
        # exists (a prior crashed run), report its T0 unchanged.
        manifest = driver.manifest
        if manifest is None:
            print("[campaign] dry-run: no manifest yet (T0 NOT frozen)",
                  flush=True)
        else:
            print(f"[campaign] dry-run: existing manifest T0="
                  f"{manifest.get('T0_iso')} (unix={manifest.get('T0_unix')})",
                  flush=True)
        precheck = driver.precheck()
        print(f"[campaign] precheck_ok={precheck['ok']} "
              f"problems={precheck['problems']}", flush=True)
        for k, v in precheck["counts"].items():
            print(f"  count {k}={v}", flush=True)
        print(f"  h1_released={precheck['h1_released']}", flush=True)
        return 0
    run_campaign(driver, contract_sha256=args.contract_sha256,
                 protocol_sha256=args.protocol_sha256)
    print("[campaign] completed", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ARM_REGISTRY",
    "ARMS",
    "CALIBRATION_ARCHS",
    "CampaignDriver",
    "FINAL_SEEDS",
    "PRUNE_ORDER",
    "SCREEN_SEEDS",
    "STAGES",
    "run_campaign",
]
