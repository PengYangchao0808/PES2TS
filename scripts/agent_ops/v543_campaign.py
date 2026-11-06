#!/usr/bin/env python3
"""V5.4.3 WP7 campaign technical-measurement harness (smoke / diagnostic /
profile) — plan §8.1 T1/T2, §8.3 resource bounds, §8.5 circuit breakers,
§7.4 resume identity, §11.2 ``v543_campaign.py`` row, ADR-0034.

What this is (and is not)
-------------------------
One CLI, three TECHNICAL-MEASUREMENT subcommands.  They are ENGINEERING
measurements: never scientific results, never ranked, never matrix rows
(``rank_excluded=true`` / ``scientific_result=false`` on every record — a
violation is the typed refusal ``V543_CAMPAIGN_SCIENCE_LEAK``).

* ``smoke`` — plan §8.1 T1 (200-update smoke).  One candidate × one frozen
  seed × a real short run through the FROZEN trainer
  (``pes2ts_core.pesn.offline.training.v543_train.V543Trainer``), plus ONE
  kill-and-resume cycle: leg 1 is a bounded ``stop_after=kill_at`` dose that
  leaves ``checkpoint_last.pt`` behind exactly as a killed process would,
  leg 2 RE-BUILDS the run from scratch and resumes from that checkpoint
  (identity-checked by the trainer, plan §7.4 :771 "不匹配 = invalid_run"),
  then completes the smoke dose.  §8 pass criteria, all measured here:
  loss finite (no non-finite loss anywhere in the curve), gradient ALIVE
  (finite, non-zero grad norm on a parameterised decoder), cache WIRED
  (frozen-encoder packet cache consulted, ``builds>0``, lookups counted),
  resume PROVEN (both legs share one identity digest and leg 2 continues at
  ``kill_at+1``).
* ``profile`` — plan §8.1 T2 profile + the WP7 budget fill.  A fixed window
  (``warmup_updates=20`` skipped + ``steady_updates=40`` measured — the
  frozen ``config/campaigns/v543_templates/budget_policy.template.json``
  ``cost_model.profile_window``; the V542 "40 W0 + 40 W1" split does NOT
  apply because §2 difference 6 retires the two-phase schedule) is run for
  each of the seven formal decoders AND for the two same-environment cost
  denominators M0/P0.  The artifact
  ``outputs/pesn/v543_dual_decoder_50k/plan/budget_profile.json`` carries
  the exact per-candidate core time, the measurement-environment SHA, the
  M0/P0 denominator rule and the cost ratios — the ORCHESTRATOR pastes
  ``per_candidate_core_time`` into the campaign YAML and re-runs
  ``v543_config_schema.py --formal``; this harness NEVER writes the config
  or the budget table (typed refusal
  ``V543_CAMPAIGN_BUDGET_WRITE_FORBIDDEN``).
* ``diagnostic`` — plan §8.1 T2 (2,000 updates) with ONLY the frozen
  engineering screens computable from the trainer's own records
  (``screening.diagnostic_loss_drop_min_pct`` and the non-finite
  loss/gradient screen).  The evaluator-side screens (dead fraction, RMSE
  ratio, AUPRC drop) are explicitly recorded as
  ``screen_not_evaluated`` — never silently passed; they belong to the
  evaluator lane.  H3 effect thresholds are structurally unreachable here
  (typed refusal ``V543_DIAG_H3_GATES_FORBIDDEN`` on any attempt to read
  them).

Governance asserted on EVERY subcommand (``assert_operational_guards``)
------------------------------------------------------------------------
``governance.authorization``/``budget``/``resources`` must keep single lane,
``swap_allowed=false``, ``oom_kill_max=0``, confirm+sealed never authorized,
``new_qc_calls=0`` and ``host_limit_gib=12``.  Any drift is a typed
``V543_CAMPAIGN_GUARD`` refusal BEFORE a byte is trained or written.

Resource discipline (§8.3 / §8.5)
---------------------------------
* one GPU lane only: :class:`LaneLock` takes an OS-level exclusive lock on
  ``<campaign_root>/plan/lane.lock`` (``fcntl.flock`` on POSIX,
  ``msvcrt.locking`` on Windows); a second concurrent harness is a typed
  ``V543_CAMPAIGN_LANE_BUSY`` refusal, and the lock is held for the whole
  measurement;
* ``OOM``/``swap``/``host RSS > host_limit_gib`` and a wall clock above
  ``2 ×`` the §8.4 estimate are ``resource_failure`` (typed
  ``V543_CAMPAIGN_RESOURCE``), recorded with the numbers, never retried
  with a reduced batch/frame/dose (§8.5 "不得减 batch/帧数");
* non-finite loss/gradient is ``invalid_run`` (``V543_CAMPAIGN_NONFINITE``);
  the dose is never extended and the seed never swapped.

Frozen machinery is REUSED, never forked: encoder/model assembly via
``CoarseEncoder.load`` + ``assemble_v543`` (candidates register into a FRESH
``DecoderRegistry`` through ``register_m1/m2/m3`` and
``register_process_candidates`` — the module-global registry keeps M0/P0
only, plan §11.1), training via ``V543Trainer``/``V543TrainerConfig``
(37,900 flat dose + ``stop_after``; the dose is never reduced), the data
view via the S0-probe/V542 frozen loader.

CLI::

    python scripts/agent_ops/v543_campaign.py smoke  --decoder M1 --seed 17 \\
        --updates 200 --kill-at 12 --device cuda
    python scripts/agent_ops/v543_campaign.py profile [--decoders M1 P1 ...]
    python scripts/agent_ops/v543_campaign.py diagnostic --decoder P1
    python scripts/agent_ops/v543_campaign.py <mode> --dry-run   # plan only
    python scripts/agent_ops/v543_campaign.py <mode> --formal    # typed-blocked
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Final, Mapping, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import yaml  # PyYAML: existing repo dependency

# ---------------------------------------------------------------------------
# frozen constants — mirror the campaign config / the plan; nothing invented
# ---------------------------------------------------------------------------
#: Technical-measurement record schema (run record + append-only report rows).
CAMPAIGN_SCHEMA: Final[str] = "pes2ts_v543_campaign_harness_v1"
RUN_RECORD_SCHEMA: Final[str] = "pes2ts_v543_measurement_run_record_v1"
#: The WP7 budget-fill artifact (plan §12 WP7; budget_policy.template.json).
BUDGET_PROFILE_SCHEMA: Final[str] = "pes2ts_v543_budget_profile_v1"

DEFAULT_CONFIG: Final[Path] = (
    PROJECT_ROOT / "config/campaigns/pes2ts_v543_dual_decoder_50k.yaml")
#: ``pes2ts_v543_dual_decoder_50k.yaml`` has no ``plan_outputs`` block; the
#: campaign root below is the one every other V543 lane already writes under
#: (``outputs/pesn/v543_dual_decoder_50k/{audit,plan,reports,runs}``).
DEFAULT_CAMPAIGN_ROOT: Final[Path] = (
    PROJECT_ROOT / "outputs/pesn/v543_dual_decoder_50k")
#: Frozen data view for the 50K pool (the S0-probe / V542 pinned view).
DEFAULT_DATA_CONFIG: Final[Path] = (
    PROJECT_ROOT / "config/campaigns/pes2ts_v542_backbone_50k.yaml")

#: The seven formal decoders of the 18-run matrix (plan §8.1 / appendix C) —
#: the exact set ``budget.per_candidate_core_time`` must cover.
MOVE_FORMAL: Final[tuple[str, ...]] = ("M1", "M2", "M3")
PROCESS_FORMAL: Final[tuple[str, ...]] = ("P1", "P2", "P3", "P4")
FORMAL_DECODERS: Final[tuple[str, ...]] = MOVE_FORMAL + PROCESS_FORMAL
#: Same-environment cost denominators (M0 = MOVE, P0 = PROCESS).
MOVE_BASELINE: Final[str] = "M0"
PROCESS_BASELINE: Final[str] = "P0"
PROFILE_DECODERS: Final[tuple[str, ...]] = (
    MOVE_BASELINE, PROCESS_BASELINE) + FORMAL_DECODERS
#: Decoders this harness may ever construct (formal + baselines).  The
#: conditional / control ids (M4, M5, P5, P4-C) are outside the matrix
#: (§8.1 T8 "另计"; §6.3 P4-C is a diagnostic control) and are refused here.
KNOWN_MEASURABLE: Final[frozenset[str]] = frozenset(
    ("M0", "M1", "M2", "M3", "P0", "P1", "P2", "P3", "P4"))

#: Frozen seeds (plan §7.2 :670; config ``source.seeds``).
FROZEN_SEEDS: Final[tuple[int, ...]] = (17, 29, 43)
#: Technical-measurement seed (config ``screening``; smoke = the anchor seed).
SMOKE_SEED: Final[int] = 17
PROFILE_SEED: Final[int] = 17
DIAGNOSTIC_SEED: Final[int] = 17

#: Frozen smoke dose (config ``screening.smoke_updates``; plan §8.1 T1).
SMOKE_UPDATES_DEFAULT: Final[int] = 200
#: Frozen diagnostic dose (config ``screening.diagnostic_updates``; §8.1 T2).
DIAGNOSTIC_UPDATES_DEFAULT: Final[int] = 2000
#: The formal flat dose (plan §7.2 :659; never reduced — ``stop_after`` only).
FORMAL_DOSE: Final[int] = 37900

#: Profile window: ``budget_policy.template.json``
#: ``cost_model.profile_window = {warmup_updates: 20, steady_updates: 40}``.
PROFILE_WARMUP_UPDATES: Final[int] = 20
PROFILE_STEADY_UPDATES: Final[int] = 40
PROFILE_WINDOW_UPDATES: Final[int] = (
    PROFILE_WARMUP_UPDATES + PROFILE_STEADY_UPDATES)

#: Resource bounds (plan §8.3; config ``resources``/``budget``).
HOST_LIMIT_GIB: Final[int] = 12
MAX_CORE_TIME_RATIO: Final[float] = 2.0
#: §8.4 single formal run estimate 43–55 min (cache hit); §8.5 circuit
#: breaker at 2 × the estimate = 2.5 h.  A technical measurement never gets
#: near it — the bound is still enforced (a hung profile is a resource
#: failure, not a silent extension).
MAX_WALL_SECONDS: Final[float] = 2.5 * 3600.0

#: Reason codes (each record carries exactly one; frozen vocabulary).
REASON_SMOKE_OK: Final[str] = "smoke_pass_loss_finite_grad_alive_cache_resumable"
REASON_PROFILE_OK: Final[str] = "profile_ok_budget_fill_measured"
REASON_DIAG_PASS: Final[str] = "diag_engineering_screens_pass"
REASON_DIAG_LOSS_DROP: Final[str] = "diag_best_inner_val_drop_lt_5pct_vs_initial"
REASON_DIAG_NOT_EVALUATED: Final[str] = "screen_not_evaluated_evaluator_metrics_absent"

#: §8.1 T2 diagnostic screens that need the evaluator (never fabricated here).
EVALUATOR_SIDE_SCREENS: Final[tuple[str, ...]] = (
    "move_dead_fraction_margin", "move_rmse_rel_max", "process_auprc_drop_max")

#: The §7.4 identity slots this harness records on every run record
#: (the trainer's own 16-key block is embedded verbatim and digested).
IDENTITY_FIELD: Final[str] = "identity"
IDENTITY_SHA_FIELD: Final[str] = "identity_sha256"


class V543CampaignError(RuntimeError):
    """Typed refusal of the campaign harness (``code`` + ``detail``).

    ``outcome`` (when set) is the frozen six-word run outcome the refusal
    maps to (plan §7.4) so an operational failure is never mistakable for a
    scientific result.
    """

    def __init__(self, code: str, detail: str = "",
                 outcome: str | None = None) -> None:
        super().__init__(f"{code}: {detail}" if detail else str(code))
        self.code = str(code)
        self.detail = str(detail)
        self.outcome = outcome


def _require(cond: bool, code: str, detail: str,
             outcome: str | None = None) -> None:
    if not cond:
        raise V543CampaignError(code, detail, outcome=outcome)


# ---------------------------------------------------------------------------
# tiny io helpers (append-only records; canonical digests)
# ---------------------------------------------------------------------------
def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_sha256(obj: Any) -> str:
    """SHA-256 of the canonical JSON encoding (sorted keys, compact, UTF-8).

    The repo's frozen ``canonical_sha256`` recipe (``v542.identity``) is
    reused when available so every V543 record digests the same way.
    """
    try:
        from pes2ts_core.pesn.models.v542.identity import canonical_sha256 as _c

        return str(_c(obj))
    except Exception:  # pragma: no cover - v542 lane unavailable
        blob = json.dumps(obj, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=False, default=str).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()


def sha256_file(path: Any) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def append_jsonl(path: Any, row: Mapping[str, Any]) -> None:
    """APPEND one JSON row (the report files are audit logs, never rewritten)."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(row), sort_keys=True, default=str) + "\n")


def read_jsonl(path: Any) -> list[dict[str, Any]]:
    p = Path(path)
    if not p.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def write_json(path: Any, payload: Mapping[str, Any]) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(payload), indent=1, sort_keys=True,
                              ensure_ascii=False, default=str) + "\n",
                   encoding="utf-8")
    return out


# ---------------------------------------------------------------------------
# config + operational guards
# ---------------------------------------------------------------------------
def load_campaign_config(path: Any | None = None) -> dict[str, Any]:
    """Load the frozen V543 campaign config YAML (read-only)."""
    cfg_path = Path(path) if path is not None else DEFAULT_CONFIG
    _require(cfg_path.is_file(), "V543_CAMPAIGN_CONFIG_MISSING",
             f"campaign config not found: {cfg_path}")
    try:
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 - typed conversion
        raise V543CampaignError(
            "V543_CAMPAIGN_CONFIG_INVALID", f"config unreadable: {exc}") from exc
    _require(isinstance(cfg, Mapping), "V543_CAMPAIGN_CONFIG_INVALID",
             "config is not a mapping")
    return dict(cfg)


def assert_operational_guards(cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Fail-closed operational guards, asserted on EVERY subcommand.

    Reads the frozen ``governance.authorization`` / ``budget`` / ``resources``
    blocks; any drift is a typed ``V543_CAMPAIGN_GUARD`` refusal BEFORE
    anything is trained or written.  Returns the guard record every output
    row carries verbatim.
    """
    auth = ((cfg.get("governance") or {}).get("authorization") or {})
    budget = cfg.get("budget") or {}
    resources = cfg.get("resources") or {}
    _require(auth.get("single_lane") is True, "V543_CAMPAIGN_GUARD",
             "governance.authorization.single_lane must be true (single-lane "
             f"campaign, plan §8.3; got {auth.get('single_lane')!r})")
    _require(auth.get("swap_allowed") is False, "V543_CAMPAIGN_GUARD",
             "governance.authorization.swap_allowed must be false (swap=0, "
             f"plan §8.3; got {auth.get('swap_allowed')!r})")
    _require(auth.get("confirm_authorized") is False, "V543_CAMPAIGN_GUARD",
             "governance.authorization.confirm_authorized must be false — "
             "confirm is sealed forever (got "
             f"{auth.get('confirm_authorized')!r})")
    _require(auth.get("sealed_authorized") is False, "V543_CAMPAIGN_GUARD",
             "governance.authorization.sealed_authorized must be false (got "
             f"{auth.get('sealed_authorized')!r})")
    _require(int(auth.get("new_qc_calls", 0) or 0) == 0,
             "V543_CAMPAIGN_GUARD",
             "governance.authorization.new_qc_calls must be 0 (got "
             f"{auth.get('new_qc_calls')!r})")
    _require(budget.get("single_lane") is True, "V543_CAMPAIGN_GUARD",
             f"budget.single_lane must be true (got {budget.get('single_lane')!r})")
    _require(int(budget.get("swap_max", -1)) == 0, "V543_CAMPAIGN_GUARD",
             f"budget.swap_max must be 0 (got {budget.get('swap_max')!r})")
    _require(int(budget.get("oom_kill_max", -1)) == 0, "V543_CAMPAIGN_GUARD",
             "budget.oom_kill_max must be 0 (got "
             f"{budget.get('oom_kill_max')!r})")
    _require(int(budget.get("host_limit_gib", -1)) == HOST_LIMIT_GIB,
             "V543_CAMPAIGN_GUARD",
             f"budget.host_limit_gib must be {HOST_LIMIT_GIB} GiB (plan §8.3; "
             f"got {budget.get('host_limit_gib')!r})")
    _require(int(resources.get("host_limit_gib", -1)) == HOST_LIMIT_GIB,
             "V543_CAMPAIGN_GUARD",
             "resources.host_limit_gib must be "
             f"{HOST_LIMIT_GIB} GiB (plan §8.3; got "
             f"{resources.get('host_limit_gib')!r})")
    _require(resources.get("single_lane") is True, "V543_CAMPAIGN_GUARD",
             "resources.single_lane must be true (got "
             f"{resources.get('single_lane')!r})")
    _require(resources.get("swap_allowed") is False, "V543_CAMPAIGN_GUARD",
             "resources.swap_allowed must be false (got "
             f"{resources.get('swap_allowed')!r})")
    return {
        "single_lane": True,
        "swap_max": 0,
        "oom_kill_max": 0,
        "host_limit_gib": HOST_LIMIT_GIB,
        "confirm_read": False,
        "sealed_authorized": False,
        "new_qc_calls": 0,
    }


def track_of(decoder_id: str) -> str:
    """The track of a decoder id (M*=move, P*=process).  Typed otherwise."""
    did = str(decoder_id)
    if did.startswith("M"):
        return "move"
    if did.startswith("P"):
        return "process"
    raise V543CampaignError(
        "V543_CAMPAIGN_DECODER_UNKNOWN",
        f"decoder id {did!r} does not belong to a V543 track (M*=move, "
        "P*=process)")


def resolve_decoders(requested: Sequence[str] | None, *,
                     default: Sequence[str] = FORMAL_DECODERS
                     ) -> list[str]:
    """Validate a requested decoder set (unknown / conditional ⇒ refusal)."""
    cands = [str(c) for c in (requested or default)]
    _require(bool(cands), "V543_CAMPAIGN_DECODERS", "decoder set is empty")
    unknown = [c for c in cands if c not in KNOWN_MEASURABLE]
    _require(not unknown, "V543_CAMPAIGN_DECODER_UNKNOWN",
             f"unknown/unmeasurable decoder(s) {unknown}; this harness "
             f"measures {sorted(KNOWN_MEASURABLE)} — the conditional "
             "candidates (M4/M5/P5) and the P4-C control are outside the "
             "WP7 engineering screens (plan §6.3/§8.1 T8)")
    seen: dict[str, None] = {}
    for cand in cands:
        seen.setdefault(cand, None)
    return list(seen)


def assert_technical_result(row: Mapping[str, Any]) -> None:
    """Typed refusal if a technical measurement would be reported as science.

    Plan §8.1: smoke/diagnostic/profile are engineering screens — they never
    enter N_core, never rank a candidate and never gate H2/H3.
    """
    _require(row.get("rank_excluded") is True,
             "V543_CAMPAIGN_SCIENCE_LEAK",
             "a technical measurement must carry rank_excluded=true (got "
             f"{row.get('rank_excluded')!r})")
    _require(row.get("scientific_result") is False,
             "V543_CAMPAIGN_SCIENCE_LEAK",
             "a technical measurement must carry scientific_result=false "
             f"(got {row.get('scientific_result')!r})")


# ---------------------------------------------------------------------------
# environment / identity
# ---------------------------------------------------------------------------
def environment_record(device: str = "cpu") -> dict[str, Any]:
    """The same-environment measurement block (plan §8.3 / §7.4 env slot)."""
    import torch

    cuda_available = bool(torch.cuda.is_available())
    device_name = None
    vram_total_mb = None
    if cuda_available:
        try:
            device_name = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            vram_total_mb = round(props.total_memory / (1024.0 ** 2), 1)
        except Exception:  # pragma: no cover - driver edge
            device_name = "<unavailable>"
    return {
        "device": str(device),
        "device_name": device_name,
        "vram_total_mb": vram_total_mb,
        "torch_version": str(torch.__version__),
        "cuda_version": torch.version.cuda,
        "cuda_available": cuda_available,
        "torch_deterministic": False,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "pid": os.getpid(),
    }


def environment_sha256(environment: Mapping[str, Any]) -> str:
    """Digest of the measurement environment (volatile pid excluded)."""
    stable = {key: value for key, value in dict(environment).items()
              if key != "pid"}
    return canonical_sha256(stable)


def build_identity_record(cfg: Mapping[str, Any], decoder_id: str, seed: int,
                          *, encoder_ckpt_sha256: str,
                          campaign_config_path: Any) -> dict[str, Any]:
    """The harness-side identity preamble of one measurement (plan §7.4).

    The trainer emits the authoritative 16-key block (it is embedded
    verbatim in the run record); this preamble names WHICH campaign config /
    decoder / seed / encoder the measurement was requested with, so a record
    can never be read against a different campaign revision.
    """
    from pes2ts_core.pesn.models.v543.latent_bridge import bridge_spec_sha256

    return {
        "campaign_id": str(cfg.get("campaign_id", "")),
        "config_path": str(campaign_config_path),
        "config_sha256": sha256_file(campaign_config_path),
        "decoder_id": str(decoder_id),
        "track": track_of(decoder_id),
        "seed": int(seed),
        "encoder_ckpt_sha256": str(encoder_ckpt_sha256),
        "bridge_spec_sha256": str(bridge_spec_sha256()),
        "contract_version": "decoder_context_v1",
        "sampler_spec_sha256": str(
            (cfg.get("training") or {}).get("sampler_spec_sha256", "")),
        "dose": FORMAL_DOSE,
    }


# ---------------------------------------------------------------------------
# single-lane GPU lock (plan §8.3 "单路 GPU"/"single_lane")
# ---------------------------------------------------------------------------
class LaneLock:
    """OS-level exclusive lock for the single measurement lane.

    POSIX uses ``fcntl.flock``; Windows uses ``msvcrt.locking``.  A second
    concurrent acquisition (same or different process) is the typed
    ``V543_CAMPAIGN_LANE_BUSY`` refusal — the campaign never runs two lanes,
    so two measurements can never contend for the 8 GB card.
    """

    def __init__(self, campaign_root: Any) -> None:
        self.path = Path(campaign_root) / "plan" / "lane.lock"
        self._handle: Any = None
        self.acquired = False

    def __enter__(self) -> "LaneLock":
        self.acquire()
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.release()

    def acquire(self) -> "LaneLock":
        if self.acquired:
            return self
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle = self.path.open("a+", encoding="utf-8")
        try:
            _lock_handle(handle)
        except OSError as exc:
            handle.close()
            raise V543CampaignError(
                "V543_CAMPAIGN_LANE_BUSY",
                "another V543 lane already holds "
                f"{self.path} — the campaign is single-lane only (plan §8.3); "
                f"a second measurement may never run concurrently ({exc})",
                outcome="resource_failure") from exc
        handle.seek(0)
        handle.truncate()
        handle.write(f"pid={os.getpid()} since={_now()}\n")
        handle.flush()
        self._handle = handle
        self.acquired = True
        return self

    def release(self) -> None:
        if not self.acquired or self._handle is None:
            return
        try:
            _unlock_handle(self._handle)
        finally:
            self._handle.close()
            self._handle = None
            self.acquired = False


def _lock_handle(handle: Any) -> None:
    try:
        import fcntl
    except ImportError:  # pragma: no cover - Windows
        import msvcrt

        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        return
    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)


def _unlock_handle(handle: Any) -> None:
    try:
        import fcntl
    except ImportError:  # pragma: no cover - Windows
        import msvcrt

        try:
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        except OSError:
            pass
        return
    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


# ---------------------------------------------------------------------------
# resource monitoring / circuit breakers (plan §8.3 / §8.5)
# ---------------------------------------------------------------------------
def host_peak_rss_mb() -> float | None:
    """Peak RSS of this process in MiB (the plan §8.3 host bound)."""
    try:
        from pes2ts_core.pesn.offline.training.v541_monitor import (
            host_peak_rss_bytes,
        )

        return round(host_peak_rss_bytes() / (1024.0 ** 2), 1)
    except Exception:  # pragma: no cover - non-POSIX
        try:
            import resource

            return round(
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 1)
        except Exception:
            return None


def peak_vram_mb(device: str) -> float | None:
    """Peak allocated VRAM in MiB (CUDA only; ``None`` on CPU / no allocation).

    ``0.0`` is reported as ``None``: with the frozen stack the model lives on
    CPU, so an untouched CUDA allocator means "no GPU measurement happened" —
    never "a GPU run that used no memory".
    """
    import torch

    if not str(device).startswith("cuda") or not torch.cuda.is_available():
        return None
    try:
        value = round(torch.cuda.max_memory_allocated(device) / (1024.0 ** 2), 1)
    except Exception:  # pragma: no cover - driver edge
        return None
    return value if value > 0 else None


def reset_peak_vram(device: str) -> None:
    import torch

    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)


def assert_resources(*, host_peak_mb: float | None, oom_kills: int,
                     swap_used: int, wall_seconds: float | None,
                     host_limit_gib: int = HOST_LIMIT_GIB,
                     max_wall_seconds: float = MAX_WALL_SECONDS) -> dict[str, Any]:
    """Enforce the §8.3 / §8.5 resource bounds; refuse on any breach.

    A breach is ``resource_failure`` (plan §8.5 "OOM/swap →
    resource_failure"), never a silent extension and never a reduced dose.
    """
    checks = {
        "host_peak_rss_mb": host_peak_mb,
        "host_limit_mb": float(host_limit_gib) * 1024.0,
        "oom_kill_count": int(oom_kills),
        "swap_used": int(swap_used),
        "wall_seconds": wall_seconds,
        "max_wall_seconds": float(max_wall_seconds),
    }
    if host_peak_mb is not None:
        _require(float(host_peak_mb) <= float(host_limit_gib) * 1024.0,
                 "V543_CAMPAIGN_RESOURCE",
                 "host peak RSS "
                 f"{host_peak_mb} MiB exceeds the {host_limit_gib} GiB bound "
                 "(plan §8.3) — resource_failure, no reduced batch/frame",
                 outcome="resource_failure")
    _require(int(oom_kills) == 0, "V543_CAMPAIGN_RESOURCE",
             f"OOM/kill count {oom_kills} != 0 (plan §8.3/§8.5: OOM is a "
             "resource_failure, never retried with a smaller batch/frame)",
             outcome="resource_failure")
    _require(int(swap_used) == 0, "V543_CAMPAIGN_RESOURCE",
             f"swap used ({swap_used}) but swap_allowed=false (plan §8.3)",
             outcome="resource_failure")
    if wall_seconds is not None:
        _require(float(wall_seconds) <= float(max_wall_seconds),
                 "V543_CAMPAIGN_RESOURCE",
                 f"wall clock {wall_seconds:.0f}s exceeds the "
                 f"{max_wall_seconds:.0f}s circuit breaker (plan §8.5: 2x the "
                 "§8.4 estimate; the dose is never silently extended)",
                 outcome="resource_failure")
    return checks


# ---------------------------------------------------------------------------
# frozen data view + candidate context (test seams)
# ---------------------------------------------------------------------------
def load_campaign_env(cfg: Mapping[str, Any],
                      *, data_config: Any = DEFAULT_DATA_CONFIG) -> Any:
    """Open the FROZEN 50K pool view READ-ONLY (no dev, no confirm).

    Reuses the S0-probe / V542 trainer asset discovery verbatim so a WP7
    measurement cannot disagree with the S0 certificates about which data
    view it used.
    """
    from scripts.agent_ops.v542_a0_parity import (
        _open_frozen_dataset,
        load_pinned_assets,
    )
    from scripts.campaigns.r3_v53mp_run import _tier_pool_curve_ids

    data_cfg, _assets = load_pinned_assets(Path(data_config))
    dataset = _open_frozen_dataset(data_cfg)
    labels_v52, endpoint_labels = dataset.label_tables()
    split_path = Path(str(
        (data_cfg.get("split_contract") or {}).get("split_manifest")
        or "/pes2ts_data/v53mp_50k_data/split_manifest_v53mp.json"))
    _require(split_path.is_file(), "V543_CAMPAIGN_DATA_MISSING",
             f"frozen split manifest missing: {split_path}")
    split_doc = json.loads(split_path.read_text(encoding="utf-8"))
    pool_ids = _tier_pool_curve_ids(split_doc, "50k")
    _require(bool(pool_ids), "V543_CAMPAIGN_DATA_MISSING",
             "the frozen 50k pool tier has no curve ids")
    return SimpleNamespace(dataset=dataset, labels=labels_v52,
                           endpoint=endpoint_labels, pool_ids=list(pool_ids),
                           data_config_path=str(data_config),
                           split_manifest=str(split_path))


def candidate_registry():
    """A FRESH registry with M0/P0 + the seven formal candidates (plan §11.1).

    The module-global ``REGISTRY_V543`` intentionally keeps only the M0/P0
    baselines; campaign wiring registers M1–M3 via ``register_m1/m2/m3`` and
    P1–P4 via ``register_process_candidates`` so no WP4/WP5/WP6 file has to
    be edited by this lane.
    """
    from pes2ts_core.pesn.models.v543 import move as move_pkg
    from pes2ts_core.pesn.models.v543 import process as process_pkg
    from pes2ts_core.pesn.models.v543.registry import default_registry

    registry = default_registry()
    move_pkg.register_m1(registry)
    move_pkg.register_m2(registry)
    move_pkg.register_m3(registry)
    process_pkg.register_process_candidates(registry)
    return registry


def build_candidate_context(*, cfg: Mapping[str, Any], env: Any,
                            decoder_id: str, seed: int, device: str = "cpu",
                            campaign_config_path: Any = DEFAULT_CONFIG,
                            registry: Any = None) -> Any:
    """Build (encoder, model, source) for ONE decoder × seed measurement.

    Candidates are constructed through the fresh registry (so §6.4 parameter
    assertions run at build time) and bound to the frozen A0 encoder of the
    seed.  The M0/P0 baselines use the zero-residual assembly (plan §4.3
    :226), which is also the same-environment cost denominator.
    """
    from scripts.agent_ops.v543_s0_probe import resolve_checkpoints
    from pes2ts_core.pesn.models.v543 import CoarseEncoder, V543Model
    from pes2ts_core.pesn.models.v543.assembly import assemble_v543
    from pes2ts_core.pesn.offline.data.v543_feature_cache import CacheIdentity
    from pes2ts_core.pesn.offline.training.v543_train import (
        RealDataSource,
        encoder_digest_v543,
    )

    track = track_of(decoder_id)
    checkpoints = resolve_checkpoints(cfg, [int(seed)])
    checkpoint = dict(checkpoints[int(seed)])
    encoder = CoarseEncoder.load(str(checkpoint["path"]), int(seed))
    encoder_sha = encoder_digest_v543(encoder)

    if decoder_id in (MOVE_BASELINE, PROCESS_BASELINE):
        assembly = assemble_v543(
            encoder, active_track=track, encoder_ckpt_sha256=encoder_sha)
    else:
        reg = registry or candidate_registry()
        decoder = reg.build(decoder_id, track=track)
        kwargs: dict[str, Any] = {}
        if track == "move":
            kwargs["move_refiner"] = decoder
            kwargs["move_decoder_id"] = str(decoder_id)
        else:
            kwargs["process_refiner"] = decoder
            kwargs["process_decoder_id"] = str(decoder_id)
        assembly = assemble_v543(
            encoder, active_track=track, encoder_ckpt_sha256=encoder_sha,
            **kwargs)
    model = V543Model(assembly)
    # IMPORTANT (measured, not assumed): the frozen WP4 trainer never migrates
    # the assembled model.  ``V543TrainerConfig.device`` reaches only the
    # supervision tensors and the data source, so building the batch on CUDA
    # while the parameters stay on CPU fails inside the V542 A0 backbone
    # ("Expected all tensors to be on the same device").  A WP7 measurement is
    # therefore executed on CPU and records the requested device next to the
    # execution device, so a record can never pretend to be a GPU run.
    execution_device = "cpu"
    source = RealDataSource(
        env.dataset, seed=int(seed), pool_curve_ids=env.pool_ids,
        cache_identity=CacheIdentity(encoder_ckpt_sha256=encoder_sha),
        device=execution_device)
    return SimpleNamespace(
        encoder=encoder, model=model, source=source, decoder_id=str(decoder_id),
        track=track, seed=int(seed), device=str(device),
        requested_device=str(device), execution_device=execution_device,
        checkpoint=checkpoint, encoder_ckpt_sha256=str(encoder_sha),
        config_path=str(campaign_config_path),
        labels_v52=env.labels, endpoint_labels=env.endpoint,
        params=dict(assembly.param_report()))


def run_leg(ctx: Any, *, out_dir: Any, stop_after: Optional[int],
            device: str = "cpu", resume: bool = False) -> dict[str, Any]:
    """Train ONE leg through the frozen trainer (plan §7.2 / §7.4).

    The dose stays the frozen 37,900; ``stop_after`` is the honest
    short-run knob (``is_formal_schedule=false``).  ``stop_after=None``
    (2026-11-03 seam fix, used by the FORMAL lane only) runs the full
    frozen dose with no cap, so the trainer's ``is_formal_schedule``
    property is True; smoke/profile legs keep passing an int cap.
    ``resume=True`` restores from ``checkpoint_last.pt`` after the
    trainer's identity check (§7.4 :771: a mismatch is ``invalid_run``,
    the checkpoint stays untouched because the refusal is raised BEFORE
    any write).
    """
    from pes2ts_core.pesn.offline.training.v543_train import (
        V543Trainer,
        V543TrainerConfig,
    )

    started = time.perf_counter()
    execution_device = str(getattr(ctx, "execution_device", "cpu"))
    config = V543TrainerConfig(
        track=str(ctx.track), decoder_id=str(ctx.decoder_id),
        seed=int(ctx.seed),
        stop_after=(int(stop_after) if stop_after is not None else None),
        use_cache=True,
        device=execution_device)
    trainer = V543Trainer(
        model=ctx.model, config=config, source=ctx.source,
        labels_v52=ctx.labels_v52, endpoint_labels=ctx.endpoint_labels,
        out_dir=Path(out_dir))
    resume_info: dict[str, Any] = {"requested": bool(resume)}
    if resume:
        checkpoint_last = Path(out_dir) / "checkpoint_last.pt"
        _require(checkpoint_last.is_file(), "V543_CAMPAIGN_RESUME_MISSING",
                 f"resume requested but {checkpoint_last} does not exist — "
                 "a resume must continue from checkpoint_last.pt (plan §7.4)",
                 outcome="invalid_run")
        resume_info["checkpoint_last_path"] = str(checkpoint_last)
        resume_info["checkpoint_last_sha256"] = sha256_file(checkpoint_last)
        try:
            stored = trainer.resume_from(checkpoint_last)
        except Exception as exc:  # noqa: BLE001 - typed re-raise below
            from pes2ts_core.pesn.offline.training.v543_train import (
                V543TrainerError,
            )

            if isinstance(exc, V543TrainerError):
                raise V543CampaignError(
                    "V543_CAMPAIGN_RESUME_IDENTITY",
                    "the resume was refused by the frozen trainer "
                    f"({exc.code}): {exc.detail} — plan §7.4 :771 makes an "
                    "identity mismatch an invalid_run (never a silent fresh "
                    "start)", outcome="invalid_run") from exc
            raise
        resume_info["stored_identity_sha256"] = canonical_sha256(stored)
        resume_info["resumed_at_update"] = int(trainer._update)
    metrics = trainer.run()
    resume_info["identity_sha256"] = metrics.get("identity_sha256")
    resume_info["identity_matched"] = (
        None if not resume else
        resume_info.get("stored_identity_sha256")
        == resume_info.get("identity_sha256"))
    return {
        "metrics": metrics,
        "identity": dict(metrics.get("identity") or {}),
        "identity_sha256": metrics.get("identity_sha256"),
        "trainer_config": config.describe(),
        "resume": resume_info,
        "params": dict(metrics.get("params") or {}),
        "wall_seconds": round(time.perf_counter() - started, 3),
        "out_dir": str(out_dir),
    }


# ---------------------------------------------------------------------------
# record construction helpers
# ---------------------------------------------------------------------------
def _preamble(*, kind: str, run_id: str, cfg: Mapping[str, Any],
              decoder_id: str, seed: int, device: str,
              guards: Mapping[str, Any], cfg_path: Any) -> dict[str, Any]:
    """The identity/guard preamble every measurement record carries."""
    return {
        "schema": RUN_RECORD_SCHEMA,
        "harness_schema": CAMPAIGN_SCHEMA,
        "kind": str(kind),
        "run_id": str(run_id),
        "campaign_id": str(cfg.get("campaign_id", "")),
        "config_path": str(cfg_path),
        "config_sha256": sha256_file(cfg_path),
        "decoder_id": str(decoder_id),
        "track": track_of(decoder_id),
        "seed": int(seed),
        "device": str(device),
        "execution_device": "cpu",
        "execution_device_note": (
            "the frozen WP4 trainer never migrates the assembled model to a "
            "CUDA device (device only selects supervision-tensor placement and "
            "the data source), so a WP7 measurement executes on CPU; the "
            "requested --device is recorded beside it and is never presented "
            "as a GPU run"),
        "created_utc": _now(),
        "guards": dict(guards),
        "single_lane": True,
        "confirm_read": False,
        "new_qc_calls": 0,
        "torch_compile": False,
        "rank_excluded": True,
        "scientific_result": False,
        "is_formal_schedule": False,
        "formal_dose": FORMAL_DOSE,
        "dose_semantics": (
            "37,900 flat updates + 400 warmup (plan §7.2 :659); a technical "
            "measurement only ever shortens the run via stop_after and never "
            "reduces the frozen dose"),
    }


def _train_rows(out_dir: Any) -> list[dict[str, Any]]:
    return [row for row in read_jsonl(Path(out_dir) / "train_log.jsonl")
            if row.get("event") == "train"]


def _val_rows(out_dir: Any) -> list[dict[str, Any]]:
    return [row for row in read_jsonl(Path(out_dir) / "run_records.jsonl")
            if row.get("event") == "validate"]


def loss_curve_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Loss/gradient curve summary (the §8.1 T1 "loss finite" evidence)."""
    losses = [row.get("loss") for row in rows]
    finite = [float(v) for v in losses
              if isinstance(v, (int, float)) and math.isfinite(float(v))]
    grads = [row.get("grad_norm") for row in rows]
    finite_grads = [float(v) for v in grads
                    if isinstance(v, (int, float)) and math.isfinite(float(v))]
    positive_grads = [v for v in finite_grads if v > 0.0]
    return {
        "n_updates": len(rows),
        "loss_first": (finite[0] if finite else None),
        "loss_last": (finite[-1] if finite else None),
        "loss_min": (min(finite) if finite else None),
        "loss_max": (max(finite) if finite else None),
        "n_finite_loss": len(finite),
        "all_loss_finite": bool(len(finite) == len(rows) and rows),
        "grad_n_observed": len(finite_grads),
        "grad_n_positive": len(positive_grads),
        "grad_min": (min(finite_grads) if finite_grads else None),
        "grad_max": (max(finite_grads) if finite_grads else None),
        "grad_alive": bool(positive_grads),
        "n_nonfinite_grad": int(len(rows) - len(finite_grads)),
        "monotone_drop": (bool(finite[-1] < finite[0])
                          if len(finite) >= 2 else None),
    }


def _cache_summary(metrics: Mapping[str, Any]) -> dict[str, Any]:
    cache = dict(metrics.get("cache") or {})
    hits = int(cache.get("hits", 0) or 0)
    misses = int(cache.get("misses", 0) or 0)
    return {
        "enabled": bool(cache.get("enabled", False)),
        "mode": cache.get("mode"),
        "hits": hits,
        "misses": misses,
        "lookups": hits + misses,
        "builds": int(cache.get("builds", 0) or 0),
        "hit_rate": cache.get("hit_rate"),
        "hit_rate_target": cache.get("hit_rate_target"),
        "identity": cache.get("identity"),
    }


def _step_seconds(rows: Sequence[Mapping[str, Any]]) -> list[float]:
    out: list[float] = []
    for row in rows:
        value = row.get("seconds")
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            out.append(float(value))
    return out


def _p95(values: Sequence[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(float(v) for v in values)
    if len(ordered) == 1:
        return ordered[0]
    import numpy as np

    return float(np.percentile(np.asarray(ordered, dtype=float), 95))


def _median(values: Sequence[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(float(v) for v in values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


def _sha_of_checkpoints(out_dir: Any) -> dict[str, str | None]:
    out: dict[str, str | None] = {}
    for name in ("checkpoint_best.pt", "checkpoint_last.pt"):
        path = Path(out_dir) / name
        out[name.replace(".pt", "_sha256")] = (
            sha256_file(path) if path.is_file() else None)
    return out


# ---------------------------------------------------------------------------
# SMOKE — §8.1 T1: loss finite, gradient alive, cache wired, resume proven
# ---------------------------------------------------------------------------
def evaluate_smoke_criteria(*, leg1_rows: Sequence[Mapping[str, Any]],
                            leg1_identity_sha256: str | None,
                            leg2_rows: Sequence[Mapping[str, Any]],
                            leg2_metrics: Mapping[str, Any],
                            leg2_identity_sha256: str | None,
                            resumed_rows: Sequence[Mapping[str, Any]],
                            kill_at: int) -> dict[str, Any]:
    """The four §8.1 T1 smoke criteria, evaluated on measured evidence.

    * ``loss_finite`` — every logged update of both legs has a finite loss;
    * ``grad_alive`` — at least one finite, POSITIVE grad norm on the
      parameterised decoder (a zero-parameter baseline cannot satisfy this,
      so baselines are not smoke-gated);
    * ``cache_wired`` — the frozen-encoder packet cache is enabled, it was
      consulted (``hits+misses>0``) and it built packets (``builds>0``);
    * ``resume_proven`` — both legs share ONE identity digest, the resumed
      leg continued at ``kill_at+1`` and the identity check passed.
    """
    summary1 = loss_curve_summary(list(leg1_rows))
    summary = loss_curve_summary(list(leg2_rows))
    checks: dict[str, dict[str, Any]] = {}

    checks["loss_finite"] = {
        "passed": bool(summary["all_loss_finite"] and summary1["all_loss_finite"]),
        "value": {"leg1_all_finite": summary1["all_loss_finite"],
                  "leg2_all_finite": summary["all_loss_finite"]},
        "rule": "every logged update (both legs) has a finite loss (§8.1 T1)",
    }
    checks["grad_alive"] = {
        "passed": bool(summary["grad_alive"] and summary["n_nonfinite_grad"] == 0),
        "value": {"n_positive_grad": summary["grad_n_positive"],
                  "grad_min": summary["grad_min"],
                  "grad_max": summary["grad_max"],
                  "n_nonfinite_grad": summary["n_nonfinite_grad"]},
        "rule": ("at least one finite non-zero grad norm on the active "
                 "decoder, no non-finite gradient (§8.1 T1 / §8.5)"),
    }
    cache = _cache_summary(leg2_metrics)
    checks["cache_wired"] = {
        "passed": bool(cache["enabled"] and cache["lookups"] > 0
                       and cache["builds"] > 0),
        "value": {key: cache[key] for key in
                  ("enabled", "hits", "misses", "lookups", "builds",
                   "hit_rate", "hit_rate_target")},
        "rule": ("the §7.3 frozen-encoder packet cache is enabled, consulted "
                 "and building packets; the >=0.80 hit-rate TARGET is a "
                 "full-dose property and is reported, never faked at 25 "
                 "updates"),
    }
    first_resumed = (int(resumed_rows[0]["update"])
                     if resumed_rows else None)
    identity_ok = (leg1_identity_sha256 is not None
                   and leg1_identity_sha256 == leg2_identity_sha256)
    checks["resume_proven"] = {
        "passed": bool(identity_ok and int(kill_at) >= 1
                       and first_resumed == int(kill_at) + 1),
        "value": {"leg1_identity_sha256": leg1_identity_sha256,
                  "leg2_identity_sha256": leg2_identity_sha256,
                  "first_resumed_update": first_resumed,
                  "expected_first_resumed_update": int(kill_at) + 1},
        "rule": ("kill at N then resume continues at N+1 under ONE identity "
                 "digest (§7.4 :771-773)"),
    }
    failures = [name for name, entry in checks.items() if not entry["passed"]]
    return {
        "passed": not failures,
        "criteria": checks,
        "failures": failures,
        "loss_curve": summary,
        "loss_curve_leg1": summary1,
    }


def smoke_candidate(env: Any, decoder_id: str, *, cfg: Mapping[str, Any],
                    campaign_root: Any, seed: int = SMOKE_SEED,
                    updates: int = SMOKE_UPDATES_DEFAULT,
                    kill_at: int | None = None, device: str = "cpu",
                    guard_record: Mapping[str, Any] | None = None,
                    cfg_path: Any = DEFAULT_CONFIG,
                    context_builder: Any = None,
                    leg_runner: Any = None) -> dict[str, Any]:
    """ONE smoke run with a kill-and-resume cycle (plan §8.1 T1).

    Leg 1 runs the bounded ``kill_at`` dose (leaving ``checkpoint_last.pt``);
    leg 2 RE-BUILDS the run from scratch (a genuine restart, not a reused
    object graph) and resumes from that checkpoint.  ``context_builder`` /
    ``leg_runner`` are the CPU-test seams (they default to the real frozen
    machinery).
    """
    updates = int(updates)
    kill_at = int(kill_at) if kill_at is not None else max(1, updates // 2)
    _require(1 <= kill_at < updates, "V543_CAMPAIGN_KILL_AT",
             f"kill_at must satisfy 1 <= kill_at < updates ({updates}); got "
             f"{kill_at}")
    guards = dict(guard_record or assert_operational_guards(cfg))
    builder = context_builder or build_candidate_context
    runner = leg_runner or run_leg

    run_id = f"V543_SMOKE_{decoder_id}_S{int(seed)}"
    run_dir = Path(campaign_root) / "smoke" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    identity_preamble = build_identity_record(
        cfg, decoder_id, int(seed), encoder_ckpt_sha256="<measured>",
        campaign_config_path=cfg_path)

    reset_peak_vram(device)
    ctx1 = builder(cfg=cfg, env=env, decoder_id=decoder_id, seed=int(seed),
                   device=device, campaign_config_path=cfg_path)
    identity_preamble["encoder_ckpt_sha256"] = str(ctx1.encoder_ckpt_sha256)
    leg1 = runner(ctx1, out_dir=run_dir, stop_after=kill_at, device=device,
                  resume=False)
    rows_after_leg1 = _train_rows(run_dir)
    n_rows_after_leg1 = len(rows_after_leg1)

    ctx2 = builder(cfg=cfg, env=env, decoder_id=decoder_id, seed=int(seed),
                   device=device, campaign_config_path=cfg_path)
    leg2 = runner(ctx2, out_dir=run_dir, stop_after=updates, device=device,
                  resume=True)
    all_rows = _train_rows(run_dir)
    resumed_rows = all_rows[n_rows_after_leg1:]

    evidence = evaluate_smoke_criteria(
        leg1_rows=rows_after_leg1,
        leg1_identity_sha256=leg1.get("identity_sha256"),
        leg2_rows=all_rows,
        leg2_metrics=(leg2.get("metrics") or {}),
        leg2_identity_sha256=leg2.get("identity_sha256"),
        resumed_rows=resumed_rows, kill_at=kill_at)
    evidence["leg1_rows"] = n_rows_after_leg1
    evidence["leg2_rows"] = len(resumed_rows)

    vram = peak_vram_mb(device)
    host_peak = host_peak_rss_mb()
    wall = float(leg1.get("wall_seconds") or 0.0) + float(
        leg2.get("wall_seconds") or 0.0)
    resource_checks = assert_resources(
        host_peak_mb=host_peak, oom_kills=0, swap_used=0, wall_seconds=wall)

    row = _preamble(kind="smoke", run_id=run_id, cfg=cfg,
                    decoder_id=decoder_id, seed=int(seed), device=device,
                    guards=guards, cfg_path=cfg_path)
    row.update({
        "identity": identity_preamble,
        "identity_sha256": canonical_sha256(identity_preamble),
        "trainer_identity": dict(leg2.get("identity") or {}),
        "trainer_identity_sha256": leg2.get("identity_sha256"),
        "dose": {"updates": updates, "kill_at": kill_at,
                 "warmup_updates": int(
                     (cfg.get("training") or {}).get("warmup_updates", 400)),
                 "is_formal_schedule": bool(
                     (leg2.get("metrics") or {}).get("is_formal_schedule"))},
        "trainer_config": dict(leg2.get("trainer_config") or {}),
        "shas": {
            "encoder_ckpt_sha256": str(ctx2.encoder_ckpt_sha256),
            "bridge_spec_sha256": identity_preamble["bridge_spec_sha256"],
            "checkpoint_path": str(ctx2.checkpoint.get("path")),
            "checkpoint_source_sha256": str(ctx2.checkpoint.get("sha256")),
            **_sha_of_checkpoints(run_dir),
        },
        "legs": {
            "leg1": {"stop_after": kill_at,
                     "updates_run": (leg1.get("metrics") or {}).get("updates"),
                     "outcome": (leg1.get("metrics") or {}).get("outcome"),
                     "wall_seconds": leg1.get("wall_seconds"),
                     "resume": leg1.get("resume")},
            "leg2": {"stop_after": updates,
                     "updates_run": (leg2.get("metrics") or {}).get("updates"),
                     "outcome": (leg2.get("metrics") or {}).get("outcome"),
                     "wall_seconds": leg2.get("wall_seconds"),
                     "resume": leg2.get("resume")},
        },
        "resume": {
            "kill_at": kill_at,
            "identity_match": evidence["criteria"]["resume_proven"]["passed"],
            "first_resumed_update":
                evidence["criteria"]["resume_proven"]["value"][
                    "first_resumed_update"],
            "checkpoint_last_path":
                (leg2.get("resume") or {}).get("checkpoint_last_path"),
            "checkpoint_last_sha256":
                (leg2.get("resume") or {}).get("checkpoint_last_sha256"),
        },
        "loss_curve": evidence["loss_curve"],
        "checks": {"passed": evidence["passed"],
                   "criteria": evidence["criteria"],
                   "failures": evidence["failures"],
                   "source_rows": {"leg1": evidence["leg1_rows"],
                                   "leg2": evidence["leg2_rows"]}},
        "cache": _cache_summary(leg2.get("metrics") or {}),
        "encoder_forward_count": int(
            (leg2.get("metrics") or {}).get("encoder_forward_count") or 0),
        "params": dict(leg2.get("params") or {}),
        "resources": {"device": str(device), "peak_vram_mb": vram,
                      "host_peak_rss_mb": host_peak,
                      "host_limit_gib": HOST_LIMIT_GIB,
                      "oom_kill_count": 0, "swap_used": 0,
                      "wall_seconds": round(wall, 3),
                      "checks": resource_checks},
        "environment": environment_record(device),
        "run_dir": str(run_dir),
        "reason_code": (REASON_SMOKE_OK if evidence["passed"]
                        else "smoke_failed:" + ",".join(evidence["failures"])),
    })
    row["environment_sha256"] = environment_sha256(row["environment"])
    assert_technical_result(row)
    append_jsonl(Path(campaign_root) / "smoke" / "smoke_report.jsonl", row)
    write_json(run_dir / "run_record.json", row)
    if not evidence["passed"]:
        raise V543CampaignError(
            "V543_CAMPAIGN_SMOKE_FAILED",
            f"smoke for {decoder_id}@S{seed} failed the §8.1 T1 criteria "
            f"{evidence['failures']} (record: {run_dir / 'run_record.json'})",
            outcome="invalid_run")
    return row


def plan_smoke(*, cfg: Mapping[str, Any], campaign_root: Any,
               decoders: Sequence[str] | None = None, seed: int = SMOKE_SEED,
               updates: int = SMOKE_UPDATES_DEFAULT,
               kill_at: int | None = None, device: str = "cpu",
               cfg_path: Any = DEFAULT_CONFIG) -> dict[str, Any]:
    """The planned smoke steps (``--dry-run``; no training, no writes)."""
    guards = assert_operational_guards(cfg)
    resolved = resolve_decoders(decoders)
    resolved_kill = (int(kill_at) if kill_at is not None
                     else max(1, int(updates) // 2))
    _require(1 <= resolved_kill < int(updates), "V543_CAMPAIGN_KILL_AT",
             f"kill_at must satisfy 1 <= kill_at < updates ({updates}); got "
             f"{resolved_kill}")
    root = Path(campaign_root) / "smoke"
    return {
        "schema": CAMPAIGN_SCHEMA,
        "kind": "smoke",
        "dry_run": True,
        "training_performed": False,
        "device": str(device),
        "seed": int(seed),
        "dose": int(updates),
        "kill_at": resolved_kill,
        "criteria": ["loss_finite", "grad_alive", "cache_wired",
                     "resume_proven"],
        "steps": [{
            "step": index,
            "run_id": f"V543_SMOKE_{cand}_S{int(seed)}",
            "decoder_id": cand,
            "track": track_of(cand),
            "legs": [{"leg": 1, "stop_after": resolved_kill, "resume": False},
                     {"leg": 2, "stop_after": int(updates), "resume": True}],
            "run_dir": str(root / f"V543_SMOKE_{cand}_S{int(seed)}"),
        } for index, cand in enumerate(resolved, start=1)],
        "report_path": str(root / "smoke_report.jsonl"),
        "guards": guards,
        "rank_excluded": True,
        "scientific_result": False,
        "config_path": str(cfg_path),
    }


# ---------------------------------------------------------------------------
# PROFILE — fixed window, budget fill (§8.1 T2 / §12 WP7)
# ---------------------------------------------------------------------------
def profile_candidate(env: Any, decoder_id: str, *, cfg: Mapping[str, Any],
                      campaign_root: Any, seed: int = PROFILE_SEED,
                      warmup: int = PROFILE_WARMUP_UPDATES,
                      steady: int = PROFILE_STEADY_UPDATES,
                      device: str = "cpu",
                      guard_record: Mapping[str, Any] | None = None,
                      cfg_path: Any = DEFAULT_CONFIG,
                      context_builder: Any = None,
                      leg_runner: Any = None) -> dict[str, Any]:
    """Measure ONE decoder over the frozen profile window.

    ``warmup`` updates are discarded (allocator/compile-free ramp) and the
    next ``steady`` updates are measured: core time = the trainer's own
    per-update ``seconds`` summed over the steady window (no startup, no
    validation, no checkpoint time), plus the p95 step time, the per-update
    rate, the CUDA peak and the cache hit rate.
    """
    warmup, steady = int(warmup), int(steady)
    total = warmup + steady
    _require(warmup >= 0 and steady >= 1, "V543_CAMPAIGN_PROFILE_WINDOW",
             f"profile window needs steady>=1 and warmup>=0; got warmup="
             f"{warmup}, steady={steady}")
    guards = dict(guard_record or assert_operational_guards(cfg))
    builder = context_builder or build_candidate_context
    runner = leg_runner or run_leg

    run_id = f"V543_PROFILE_{decoder_id}_S{int(seed)}"
    run_dir = Path(campaign_root) / "profile" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    reset_peak_vram(device)
    ctx = builder(cfg=cfg, env=env, decoder_id=decoder_id, seed=int(seed),
                  device=device, campaign_config_path=cfg_path)
    leg = runner(ctx, out_dir=run_dir, stop_after=total, device=device,
                 resume=False)
    rows = _train_rows(run_dir)
    steady_rows = [row for row in rows if int(row.get("update", 0)) > warmup]
    steps = _step_seconds(steady_rows)
    core_seconds = round(sum(steps), 6)
    median_step = _median(steps)
    # A single host scheduling hiccup inside the window can dominate a 40-update
    # sum (see the run report's §"profile noise" note).  The frozen definition
    # stays the SUM; a median-based robust value is reported beside it so a
    # denominator cannot be inflated by one stalled update.
    robust_seconds = (round(median_step * len(steps), 6)
                      if median_step is not None else None)
    outliers = ([value for value in steps if value > 5.0 * median_step]
                if median_step else [])
    per_update = (core_seconds / len(steps)) if steps else None
    p95 = _p95(steps)
    updates_per_second = (round(1.0 / per_update, 4)
                          if per_update and per_update > 0 else None)
    vram = peak_vram_mb(device)
    host_peak = host_peak_rss_mb()
    wall = float(leg.get("wall_seconds") or 0.0)
    resource_checks = assert_resources(
        host_peak_mb=host_peak, oom_kills=0, swap_used=0, wall_seconds=wall)
    cache = _cache_summary(leg.get("metrics") or {})

    identity_preamble = build_identity_record(
        cfg, decoder_id, int(seed),
        encoder_ckpt_sha256=str(ctx.encoder_ckpt_sha256),
        campaign_config_path=cfg_path)
    row = _preamble(kind="profile", run_id=run_id, cfg=cfg,
                    decoder_id=decoder_id, seed=int(seed), device=device,
                    guards=guards, cfg_path=cfg_path)
    row.update({
        "identity": identity_preamble,
        "identity_sha256": canonical_sha256(identity_preamble),
        "trainer_identity": dict(leg.get("identity") or {}),
        "trainer_identity_sha256": leg.get("identity_sha256"),
        "window": {
            "warmup_updates": warmup,
            "steady_updates": steady,
            "total_updates": total,
            "measured_updates": len(steps),
            "warmup_rule": ("the frozen budget_policy.template.json "
                            "cost_model.profile_window "
                            "{warmup_updates:20, steady_updates:40}; the V542 "
                            "W0/W1 split does not apply (plan §2 difference 6)"),
        },
        "dose": {"updates": total, "stop_after": total,
                 "is_formal_schedule": bool(
                     (leg.get("metrics") or {}).get("is_formal_schedule"))},
        "core_time": {
            "core_seconds": core_seconds,
            "core_seconds_robust": robust_seconds,
            "core_seconds_per_update": (round(per_update, 6)
                                        if per_update is not None else None),
            "updates_per_second": updates_per_second,
            "p95_step_seconds": (round(p95, 6) if p95 is not None else None),
            "step_seconds_median": median_step,
            "step_seconds_max": (max(steps) if steps else None),
            "n_step_outliers_gt_5x_median": len(outliers),
            "outlier_note": (
                "core_seconds is the frozen definition (sum of the steady "
                "window); core_seconds_robust = median x measured_updates and "
                "is reported so one host scheduling hiccup cannot inflate a "
                "M0/P0 cost denominator"),
            "measured_updates": len(steps),
            "definition": ("sum of the frozen trainer's per-update `seconds` "
                           "over the steady window — excludes startup, "
                           "validation, checkpointing and evaluation"),
        },
        "resources": {"device": str(device), "peak_vram_mb": vram,
                      "host_peak_rss_mb": host_peak,
                      "host_limit_gib": HOST_LIMIT_GIB,
                      "oom_kill_count": 0, "swap_used": 0,
                      "wall_seconds": round(wall, 3),
                      "checks": resource_checks},
        "cache": cache,
        "encoder_forward_count": int(
            (leg.get("metrics") or {}).get("encoder_forward_count") or 0),
        "params": dict(leg.get("params") or {}),
        "trainer_config": dict(leg.get("trainer_config") or {}),
        "shas": {
            "encoder_ckpt_sha256": str(ctx.encoder_ckpt_sha256),
            "bridge_spec_sha256": identity_preamble["bridge_spec_sha256"],
            **_sha_of_checkpoints(run_dir),
        },
        "environment": environment_record(device),
        "budget_written": False,
        "budget_note": ("measurement only: the ORCHESTRATOR pastes "
                        "per_candidate_core_time into the campaign config and "
                        "re-runs the --formal gate; this harness never writes "
                        "the config or the budget table"),
        "run_dir": str(run_dir),
        "reason_code": REASON_PROFILE_OK,
    })
    row["environment_sha256"] = environment_sha256(row["environment"])
    _require("per_candidate_core_time" not in row,
             "V543_CAMPAIGN_BUDGET_WRITE_FORBIDDEN",
             "a profile row must never carry budget.per_candidate_core_time")
    assert_technical_result(row)
    append_jsonl(Path(campaign_root) / "profile" / "profile_report.jsonl", row)
    write_json(run_dir / "run_record.json", row)
    return row


def plan_profile(*, cfg: Mapping[str, Any], campaign_root: Any,
                 decoders: Sequence[str] | None = None,
                 seed: int = PROFILE_SEED,
                 warmup: int = PROFILE_WARMUP_UPDATES,
                 steady: int = PROFILE_STEADY_UPDATES,
                 device: str = "cpu",
                 cfg_path: Any = DEFAULT_CONFIG) -> dict[str, Any]:
    """The planned profile steps (``--dry-run``; no training, no writes)."""
    guards = assert_operational_guards(cfg)
    resolved = resolve_decoders(decoders, default=PROFILE_DECODERS)
    root = Path(campaign_root) / "profile"
    return {
        "schema": CAMPAIGN_SCHEMA,
        "kind": "profile",
        "dry_run": True,
        "training_performed": False,
        "device": str(device),
        "seed": int(seed),
        "window": {"warmup_updates": int(warmup),
                   "steady_updates": int(steady),
                   "total_updates": int(warmup) + int(steady)},
        "denominators": {"move": MOVE_BASELINE, "process": PROCESS_BASELINE},
        "steps": [{
            "step": index,
            "run_id": f"V543_PROFILE_{cand}_S{int(seed)}",
            "decoder_id": cand,
            "track": track_of(cand),
            "role": ("denominator" if cand in (MOVE_BASELINE, PROCESS_BASELINE)
                     else "formal_candidate"),
            "run_dir": str(root / f"V543_PROFILE_{cand}_S{int(seed)}"),
        } for index, cand in enumerate(resolved, start=1)],
        "report_path": str(root / "profile_report.jsonl"),
        "budget_artifact": str(Path(campaign_root) / "plan"
                               / "budget_profile.json"),
        "guards": guards,
        "rank_excluded": True,
        "scientific_result": False,
        "config_path": str(cfg_path),
    }


def build_budget_profile(rows: Sequence[Mapping[str, Any]], *,
                         cfg: Mapping[str, Any],
                         warmup: int, steady: int,
                         environment: Mapping[str, Any],
                         cfg_path: Any = DEFAULT_CONFIG) -> dict[str, Any]:
    """Assemble the WP7 budget-fill artifact from the profile rows.

    Every value is MEASURED; the M0/P0 denominators come from the same
    window in the same environment (``cost_acceptance.denominator`` of
    ``budget_policy.template.json``: "same V543 environment, same batch, same
    profile measured M0/P0"; the archived V541/V542 per-candidate core times
    are the FORBIDDEN denominator).
    """
    by_id: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        by_id[str(row.get("decoder_id"))] = row
    missing = [did for did in FORMAL_DECODERS if did not in by_id]
    _require(not missing, "V543_CAMPAIGN_BUDGET_PROFILE_INCOMPLETE",
             "the budget artifact needs all seven formal decoders; missing "
             f"{missing} — the measured rows are kept in "
             "profile/profile_report.jsonl, but no budget artifact is ever "
             "written from a partial set (a partial table would be "
             "V543_BUDGET_INVALID at the --formal gate)")
    for den in (MOVE_BASELINE, PROCESS_BASELINE):
        _require(den in by_id, "V543_CAMPAIGN_BUDGET_PROFILE_INCOMPLETE",
                 f"the budget artifact needs the same-env {den} denominator "
                 "(M0 = MOVE, P0 = PROCESS; archive numbers are forbidden)")

    def _core(did: str) -> float:
        entry = dict(by_id[did].get("core_time") or {})
        value = entry.get("core_seconds")
        _require(isinstance(value, (int, float))
                 and math.isfinite(float(value)) and float(value) > 0,
                 "V543_CAMPAIGN_BUDGET_PROFILE_INVALID",
                 f"{did}: core_seconds must be a finite POSITIVE measurement "
                 f"(got {value!r}) — the formal gate types zero/NaN as "
                 "V543_BUDGET_INVALID")
        return float(value)

    def _robust(did: str) -> float | None:
        value = dict(by_id[did].get("core_time") or {}).get(
            "core_seconds_robust")
        if isinstance(value, (int, float)) and math.isfinite(float(value)) \
                and float(value) > 0:
            return float(value)
        return None

    denominators = {
        "move": {"decoder_id": MOVE_BASELINE,
                 "core_seconds": _core(MOVE_BASELINE),
                 "core_seconds_robust": _robust(MOVE_BASELINE)},
        "process": {"decoder_id": PROCESS_BASELINE,
                    "core_seconds": _core(PROCESS_BASELINE),
                    "core_seconds_robust": _robust(PROCESS_BASELINE)},
    }
    per_candidate: dict[str, Any] = {}
    for did in FORMAL_DECODERS:
        track = track_of(did)
        core = _core(did)
        den = denominators[track]["core_seconds"]
        core_time = dict(by_id[did].get("core_time") or {})
        cache = dict(by_id[did].get("cache") or {})
        params = dict(by_id[did].get("params") or {})
        active = params.get("active")
        trainable = (active.get("n_trainable_params")
                     if isinstance(active, Mapping) else active)
        resources = dict(by_id[did].get("resources") or {})
        ratio = round(core / den, 6) if den > 0 else None
        robust = _robust(did)
        den_robust = denominators[track]["core_seconds_robust"]
        robust_ratio = (round(robust / den_robust, 6)
                        if robust and den_robust and den_robust > 0 else None)
        per_candidate[did] = {
            "core_seconds": round(core, 6),
            "core_seconds_robust": (None if robust is None
                                    else round(robust, 6)),
            "core_seconds_per_update": core_time.get("core_seconds_per_update"),
            "updates_per_second": core_time.get("updates_per_second"),
            "p95_step_seconds": core_time.get("p95_step_seconds"),
            "step_seconds_median": core_time.get("step_seconds_median"),
            "n_step_outliers_gt_5x_median":
                core_time.get("n_step_outliers_gt_5x_median"),
            "measured_updates": core_time.get("measured_updates"),
            "core_ratio_vs_denominator": ratio,
            "core_ratio_robust_vs_denominator": robust_ratio,
            "denominator_decoder_id": denominators[track]["decoder_id"],
            "cost_gate_pass": (None if ratio is None
                               else bool(ratio <= MAX_CORE_TIME_RATIO)),
            "cost_gate_pass_robust": (None if robust_ratio is None
                                      else bool(robust_ratio
                                                <= MAX_CORE_TIME_RATIO)),
            "peak_vram_mb": resources.get("peak_vram_mb"),
            "host_peak_rss_mb": resources.get("host_peak_rss_mb"),
            "cache_hit_rate": cache.get("hit_rate"),
            "encoder_forward_count": by_id[did].get("encoder_forward_count"),
            "trainable_params": trainable,
            "track": track,
        }
    measurements = {
        "profile_window": {"warmup_updates": int(warmup),
                           "steady_updates": int(steady),
                           "total_updates": int(warmup) + int(steady)},
        "environment_sha256": environment_sha256(environment),
        "environment": dict(environment),
        "denominators": denominators,
        "per_candidate_core_time": per_candidate,
        "cost_acceptance": {
            "max_core_time_ratio": MAX_CORE_TIME_RATIO,
            "ratio_expression":
                "candidate core_seconds / same-env track denominator "
                "core_seconds (M0 for MOVE, P0 for PROCESS)",
            "denominator": ("same V543 environment, same batch, same "
                            "20+40 profile window measured M0/P0"),
            "forbidden_denominator": (
                "v541/v542 archived per-candidate core time "
                "(budget_policy.template.json cost_acceptance)"),
            "on_exceed": ("gate_skipped_cost_ratio_exceeds_2 — the scientific "
                          "result is retained, the candidate is not promoted; "
                          "NOT a resource_failure"),
        },
    }
    return {
        "schema": BUDGET_PROFILE_SCHEMA,
        "kind": "wp7_budget_profile",
        "campaign_id": str(cfg.get("campaign_id", "")),
        "config_path": str(cfg_path),
        "config_sha256": sha256_file(cfg_path),
        "created_utc": _now(),
        "plan_section": ("plan §8.1 T2 profile + §12 WP7 'budget 签认'; "
                         "budget_policy.template.json cost_model"),
        "measurements": measurements,
        "measurements_sha256": canonical_sha256(measurements),
        "ready_for_budget_fill": True,
        "fill_instructions": (
            "copy measurements.per_candidate_core_time into the campaign "
            "YAML budget.per_candidate_core_time (the validator accepts "
            "'{decoder: {core_seconds: <positive>}}'; M0/P0 may be added as "
            "the same-env denominators), then re-run "
            "'python scripts/agent_ops/v543_config_schema.py --formal "
            "--config config/campaigns/pes2ts_v543_dual_decoder_50k.yaml'"),
        "discipline": (
            "this artifact is the ONLY WP7 budget input; it is never written "
            "into the config by the measuring harness and it never carries an "
            "archived V541/V542 cost number"),
        "rank_excluded": True,
        "scientific_result": False,
    }


def run_profile(env: Any, *, cfg: Mapping[str, Any], campaign_root: Any,
                decoders: Sequence[str] | None = None,
                seed: int = PROFILE_SEED,
                warmup: int = PROFILE_WARMUP_UPDATES,
                steady: int = PROFILE_STEADY_UPDATES,
                device: str = "cpu", cfg_path: Any = DEFAULT_CONFIG,
                context_builder: Any = None, leg_runner: Any = None
                ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Profile every requested decoder and write the budget-fill artifact."""
    guards = assert_operational_guards(cfg)
    resolved = resolve_decoders(decoders, default=PROFILE_DECODERS)
    rows: list[dict[str, Any]] = []
    for decoder_id in resolved:
        rows.append(profile_candidate(
            env, decoder_id, cfg=cfg, campaign_root=campaign_root,
            seed=seed, warmup=warmup, steady=steady, device=device,
            guard_record=guards, cfg_path=cfg_path,
            context_builder=context_builder, leg_runner=leg_runner))
    artifact = build_budget_profile(
        rows, cfg=cfg, warmup=warmup, steady=steady,
        environment=environment_record(device), cfg_path=cfg_path)
    write_json(Path(campaign_root) / "plan" / "budget_profile.json", artifact)
    return rows, artifact


# ---------------------------------------------------------------------------
# DIAGNOSTIC — §8.1 T2 2,000-update engineering screens
# ---------------------------------------------------------------------------
def diagnostic_screens(cfg: Mapping[str, Any]) -> dict[str, float]:
    """The frozen engineering-screen thresholds this harness may apply.

    An ALLOWLIST: only the trainer-record-computable screens are returned, so
    an H3 effect key (plan §9.3) is structurally impossible to smuggle into a
    2,000-update diagnostic (``V543_DIAG_H3_GATES_FORBIDDEN``).
    """
    screening = dict(cfg.get("screening") or {})
    allowed = {
        "diagnostic_loss_drop_min_pct": float(
            screening.get("diagnostic_loss_drop_min_pct", 5.0)),
    }
    leaked = [key for key in allowed
              if key.startswith(("move_h3", "process_h3", "h3_"))]
    _require(not leaked, "V543_DIAG_H3_GATES_FORBIDDEN",
             f"H3 effect thresholds {leaked} may never gate a diagnostic "
             "(plan §8.1 T2 / §9.3)")
    return allowed


def evaluate_diagnostic(*, val_rows: Sequence[Mapping[str, Any]],
                        train_rows: Sequence[Mapping[str, Any]],
                        loss_drop_min_pct: float) -> dict[str, Any]:
    """Apply ONLY the loss-drop + non-finite screens (§8.1 T2).

    The evaluator-side screens (dead fraction / RMSE ratio / AUPRC drop) are
    recorded ``screen_not_evaluated`` — never a silent pass.
    """
    finite_vals = [float(row["inner_val_loss"]) for row in val_rows
                   if isinstance(row.get("inner_val_loss"), (int, float))
                   and math.isfinite(float(row["inner_val_loss"]))]
    summary = loss_curve_summary(list(train_rows))
    first = finite_vals[0] if finite_vals else None
    best = min(finite_vals) if finite_vals else None
    drop_pct = (None if first in (None, 0.0) or best is None
                else round(100.0 * (first - best) / first, 4))
    nonfinite = not summary["all_loss_finite"] or summary["n_nonfinite_grad"] > 0
    checks = {
        "non_finite_loss_or_gradient": {"passed": not nonfinite,
                                        "value": summary},
        "loss_drop": {
            "passed": (drop_pct is not None and drop_pct >= loss_drop_min_pct),
            "value": {"first_inner_val_loss": first,
                      "best_inner_val_loss": best,
                      "drop_pct": drop_pct,
                      "threshold_pct": float(loss_drop_min_pct)},
            "rule": "best inner-val loss drop >= the frozen screening margin",
        },
        "evaluator_side_screens": {
            "passed": None,
            "value": {name: "not_evaluated" for name in EVALUATOR_SIDE_SCREENS},
            "rule": ("dead fraction / MOVE RMSE / PROCESS AUPRC need the "
                     "frozen V543 evaluator; the diagnostic harness records "
                     "them as not evaluated instead of passing them"),
            "reason_code": REASON_DIAG_NOT_EVALUATED,
        },
    }
    failures = [name for name, entry in checks.items()
                if entry["passed"] is False]
    return {"passed": not failures, "checks": checks, "failures": failures,
            "loss_drop_pct": drop_pct}


def diagnostic_candidate(env: Any, decoder_id: str, *,
                         cfg: Mapping[str, Any], campaign_root: Any,
                         seed: int = DIAGNOSTIC_SEED,
                         updates: int = DIAGNOSTIC_UPDATES_DEFAULT,
                         device: str = "cpu",
                         guard_record: Mapping[str, Any] | None = None,
                         cfg_path: Any = DEFAULT_CONFIG,
                         context_builder: Any = None,
                         leg_runner: Any = None) -> dict[str, Any]:
    """Run the 2,000-update diagnostic for ONE decoder (§8.1 T2)."""
    updates = int(updates)
    _require(updates >= 1, "V543_CAMPAIGN_DIAGNOSTIC_WINDOW",
             f"diagnostic updates must be >= 1, got {updates}")
    guards = dict(guard_record or assert_operational_guards(cfg))
    builder = context_builder or build_candidate_context
    runner = leg_runner or run_leg
    run_id = f"V543_DIAG_{decoder_id}_S{int(seed)}"
    run_dir = Path(campaign_root) / "diagnostic" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    reset_peak_vram(device)
    ctx = builder(cfg=cfg, env=env, decoder_id=decoder_id, seed=int(seed),
                  device=device, campaign_config_path=cfg_path)
    leg = runner(ctx, out_dir=run_dir, stop_after=updates, device=device,
                 resume=False)
    train_rows = _train_rows(run_dir)
    val_rows = _val_rows(run_dir)
    screens = diagnostic_screens(cfg)
    verdict = evaluate_diagnostic(
        val_rows=val_rows, train_rows=train_rows,
        loss_drop_min_pct=screens["diagnostic_loss_drop_min_pct"])
    vram = peak_vram_mb(device)
    host_peak = host_peak_rss_mb()
    wall = float(leg.get("wall_seconds") or 0.0)
    resource_checks = assert_resources(
        host_peak_mb=host_peak, oom_kills=0, swap_used=0, wall_seconds=wall)
    identity_preamble = build_identity_record(
        cfg, decoder_id, int(seed),
        encoder_ckpt_sha256=str(ctx.encoder_ckpt_sha256),
        campaign_config_path=cfg_path)
    row = _preamble(kind="diagnostic", run_id=run_id, cfg=cfg,
                    decoder_id=decoder_id, seed=int(seed), device=device,
                    guards=guards, cfg_path=cfg_path)
    row.update({
        "identity": identity_preamble,
        "identity_sha256": canonical_sha256(identity_preamble),
        "trainer_identity": dict(leg.get("identity") or {}),
        "trainer_identity_sha256": leg.get("identity_sha256"),
        "dose": {"updates": updates, "stop_after": updates,
                 "is_formal_schedule": bool(
                     (leg.get("metrics") or {}).get("is_formal_schedule"))},
        "screens": {"thresholds": screens, "checks": verdict["checks"],
                    "failures": verdict["failures"],
                    "passed": verdict["passed"],
                    "loss_drop_pct": verdict["loss_drop_pct"],
                    "evaluator_side_screens_not_evaluated": list(
                        EVALUATOR_SIDE_SCREENS)},
        "loss_curve": loss_curve_summary(train_rows),
        "validation": {"n_validations": len(val_rows),
                       "inner_val_losses": [
                           row_.get("inner_val_loss") for row_ in val_rows]},
        "cache": _cache_summary(leg.get("metrics") or {}),
        "encoder_forward_count": int(
            (leg.get("metrics") or {}).get("encoder_forward_count") or 0),
        "params": dict(leg.get("params") or {}),
        "resources": {"device": str(device), "peak_vram_mb": vram,
                      "host_peak_rss_mb": host_peak,
                      "host_limit_gib": HOST_LIMIT_GIB,
                      "oom_kill_count": 0, "swap_used": 0,
                      "wall_seconds": round(wall, 3),
                      "checks": resource_checks},
        "environment": environment_record(device),
        "run_dir": str(run_dir),
        "reason_code": (REASON_DIAG_PASS if verdict["passed"]
                        else (REASON_DIAG_LOSS_DROP if "loss_drop"
                              in verdict["failures"] else "diag_non_finite")),
    })
    row["environment_sha256"] = environment_sha256(row["environment"])
    assert_technical_result(row)
    append_jsonl(Path(campaign_root) / "diagnostic"
                 / "diagnostic_report.jsonl", row)
    write_json(run_dir / "run_record.json", row)
    return row


# ---------------------------------------------------------------------------
# formal gate (delegated to the frozen validator — never bypassed)
# ---------------------------------------------------------------------------
def formal_gate(config_path: Any = DEFAULT_CONFIG,
                *, record_root: Any = None) -> dict[str, Any]:
    """``--formal`` DELEGATES to the frozen config validator.

    While ``budget.per_candidate_core_time`` is null the validator refuses
    with ``V543_BUDGET_NULL`` — surfaced VERBATIM, so this technical harness
    can never launch a formal run and can never bypass the WP7 budget gate.
    """
    from scripts.agent_ops.v543_config_schema import (
        V543SchemaError,
        load_config,
        validate_config,
    )

    try:
        return validate_config(load_config(config_path), formal=True,
                               record_root=record_root)
    except V543SchemaError as exc:
        raise V543CampaignError(
            exc.code,
            "formal launch is typed-blocked by the frozen config validator: "
            + exc.detail) from exc


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def validate_cli_seed(seed: Any) -> int:
    """Typed validation of ``--seed`` (frozen seeds only)."""
    try:
        value = int(seed)
    except (TypeError, ValueError) as exc:
        raise V543CampaignError(
            "V543_CAMPAIGN_SEED_FROZEN",
            f"--seed must be an integer in {list(FROZEN_SEEDS)}, got "
            f"{seed!r}") from exc
    _require(value in FROZEN_SEEDS, "V543_CAMPAIGN_SEED_FROZEN",
             f"--seed must be one of the frozen seeds {list(FROZEN_SEEDS)} "
             f"(plan §7.2 :670), got {value!r}")
    return value


def validate_cli_device(device: Any) -> str:
    """Typed validation of ``--device`` (``cpu`` always; cuda must exist)."""
    dev = str(device)
    if dev == "cpu":
        return dev
    if dev.startswith("cuda"):
        import torch

        _require(bool(torch.cuda.is_available()),
                 "V543_CAMPAIGN_DEVICE_UNAVAILABLE",
                 f"device={dev!r} requested but CUDA is not available on this "
                 "host (cpu is fully supported)")
        if dev != "cuda":
            try:
                index = int(dev.split(":", 1)[1])
            except (IndexError, ValueError) as exc:
                raise V543CampaignError(
                    "V543_CAMPAIGN_DEVICE_UNKNOWN",
                    f"--device must be 'cpu', 'cuda' or 'cuda:N', got "
                    f"{dev!r}") from exc
            _require(0 <= index < torch.cuda.device_count(),
                     "V543_CAMPAIGN_DEVICE_UNAVAILABLE",
                     f"device={dev!r} requested but this host has "
                     f"{torch.cuda.device_count()} CUDA device(s)")
        return dev
    raise V543CampaignError(
        "V543_CAMPAIGN_DEVICE_UNKNOWN",
        f"--device must be 'cpu' or 'cuda[:N]', got {dev!r}")


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--campaign-root", default=str(DEFAULT_CAMPAIGN_ROOT))
    parser.add_argument("--data-config", default=str(DEFAULT_DATA_CONFIG))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true",
                        help="print the planned steps WITHOUT training "
                             "(no outputs, no lock)")
    parser.add_argument("--formal", action="store_true",
                        help="attempt a formal launch: DELEGATES to the frozen "
                             "validator (typed-blocked V543_BUDGET_NULL while "
                             "budget.per_candidate_core_time is null)")


def _campaign_refusal_from_exception(
        exc: BaseException) -> V543CampaignError | None:
    """Convert EXPECTED failure classes into typed campaign refusals.

    The frozen trainer (``V543TrainerError``), the frozen config validator
    (``V543SchemaError``) and a missing frozen artifact (``FileNotFoundError``)
    are expected: they surface as a typed refusal with a non-zero exit, never
    an uncaught traceback.  Anything else returns ``None`` and the caller
    re-raises (unexpected exceptions are never swallowed).
    """
    try:
        from pes2ts_core.pesn.offline.training.v543_train import (
            V543TrainerError,
        )

        if isinstance(exc, V543TrainerError):
            return V543CampaignError(
                "V543_CAMPAIGN_TRAINER_REFUSAL",
                f"frozen trainer refused (code "
                f"{getattr(exc, 'code', None)!r}, outcome "
                f"{getattr(exc, 'outcome', None)!r}): {exc}",
                outcome=getattr(exc, "outcome", None))
    except Exception:  # noqa: BLE001 - trainer import unavailable
        pass
    try:
        from scripts.agent_ops.v543_config_schema import V543SchemaError

        if isinstance(exc, V543SchemaError):
            return V543CampaignError(
                getattr(exc, "code", "V543_CAMPAIGN_VALIDATOR_REFUSAL"),
                f"frozen config validator refused: {exc}")
    except Exception:  # noqa: BLE001 - validator import unavailable
        pass
    if isinstance(exc, FileNotFoundError):
        return V543CampaignError(
            "V543_CAMPAIGN_DATA_MISSING",
            f"frozen data-view artifact missing: {exc}")
    return None


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="v543_campaign.py",
        description="V5.4.3 WP7 campaign technical-measurement harness "
                    "(smoke / diagnostic / profile; never formal, never "
                    "scientific results)")
    sub = parser.add_subparsers(dest="command", required=True)

    p_smoke = sub.add_parser(
        "smoke", help="200-update smoke with one kill-resume cycle (§8.1 T1)")
    _add_common_args(p_smoke)
    p_smoke.add_argument("--decoder", "--decoders", dest="decoders",
                         nargs="*", default=None)
    p_smoke.add_argument("--updates", type=int, default=SMOKE_UPDATES_DEFAULT)
    p_smoke.add_argument("--kill-at", type=int, default=None)

    p_profile = sub.add_parser(
        "profile", help="fixed-window profile + WP7 budget fill (§8.1 T2)")
    _add_common_args(p_profile)
    p_profile.add_argument("--decoder", "--decoders", dest="decoders",
                           nargs="*", default=None)
    p_profile.add_argument("--warmup", type=int,
                           default=PROFILE_WARMUP_UPDATES)
    p_profile.add_argument("--steady", type=int,
                           default=PROFILE_STEADY_UPDATES)

    p_diag = sub.add_parser(
        "diagnostic", help="2,000-update engineering-screen diagnostic")
    _add_common_args(p_diag)
    p_diag.add_argument("--decoder", "--decoders", dest="decoders",
                        nargs="*", default=None)
    p_diag.add_argument("--updates", type=int,
                        default=DIAGNOSTIC_UPDATES_DEFAULT)

    args = parser.parse_args(argv)
    try:
        if getattr(args, "formal", False):
            formal_gate(args.config)
            return 2  # pragma: no cover - formal_gate always raises

        cfg = load_campaign_config(args.config)
        seed_default = {"smoke": SMOKE_SEED, "profile": PROFILE_SEED,
                        "diagnostic": DIAGNOSTIC_SEED}[args.command]
        seed = validate_cli_seed(
            args.seed if args.seed is not None else seed_default)
        device = validate_cli_device(args.device)

        if args.dry_run:
            if args.command == "smoke":
                plan = plan_smoke(
                    cfg=cfg, campaign_root=args.campaign_root,
                    decoders=args.decoders, seed=seed, updates=args.updates,
                    kill_at=args.kill_at, device=device,
                    cfg_path=args.config)
            elif args.command == "profile":
                plan = plan_profile(
                    cfg=cfg, campaign_root=args.campaign_root,
                    decoders=args.decoders, seed=seed, warmup=args.warmup,
                    steady=args.steady, device=device, cfg_path=args.config)
            else:
                plan = plan_smoke(
                    cfg=cfg, campaign_root=args.campaign_root,
                    decoders=args.decoders, seed=seed, updates=args.updates,
                    kill_at=None, device=device, cfg_path=args.config)
                plan["kind"] = "diagnostic"
            print(json.dumps(plan, indent=1, sort_keys=True, default=str))
            return 0

        guards = assert_operational_guards(cfg)
        with LaneLock(args.campaign_root):
            env = load_campaign_env(cfg, data_config=args.data_config)
            if args.command == "profile":
                rows, artifact = run_profile(
                    env, cfg=cfg, campaign_root=args.campaign_root,
                    decoders=args.decoders, seed=seed, warmup=args.warmup,
                    steady=args.steady, device=device, cfg_path=args.config)
                summary = {
                    "kind": "profile",
                    "rows": len(rows),
                    "run_ids": [row["run_id"] for row in rows],
                    "budget_artifact": str(
                        Path(args.campaign_root) / "plan"
                        / "budget_profile.json"),
                    "per_candidate_core_time": {
                        did: entry["core_seconds"] for did, entry in
                        artifact["measurements"][
                            "per_candidate_core_time"].items()},
                    "denominators": artifact["measurements"]["denominators"],
                    "environment_sha256":
                        artifact["measurements"]["environment_sha256"],
                    "rank_excluded": True,
                    "scientific_result": False,
                }
            elif args.command == "smoke":
                resolved = resolve_decoders(args.decoders)
                rows = [smoke_candidate(
                    env, decoder_id, cfg=cfg,
                    campaign_root=args.campaign_root, seed=seed,
                    updates=args.updates, kill_at=args.kill_at, device=device,
                    guard_record=guards, cfg_path=args.config)
                    for decoder_id in resolved]
                summary = {
                    "kind": "smoke",
                    "rows": len(rows),
                    "run_ids": [row["run_id"] for row in rows],
                    "reason_codes": [row["reason_code"] for row in rows],
                    "checks": {row["run_id"]: row["checks"]["passed"]
                               for row in rows},
                    "rank_excluded": True,
                    "scientific_result": False,
                }
            else:
                resolved = resolve_decoders(args.decoders)
                rows = [diagnostic_candidate(
                    env, decoder_id, cfg=cfg,
                    campaign_root=args.campaign_root, seed=seed,
                    updates=args.updates, device=device,
                    guard_record=guards, cfg_path=args.config)
                    for decoder_id in resolved]
                summary = {
                    "kind": "diagnostic",
                    "rows": len(rows),
                    "run_ids": [row["run_id"] for row in rows],
                    "reason_codes": [row["reason_code"] for row in rows],
                    "evaluator_side_screens_not_evaluated": list(
                        EVALUATOR_SIDE_SCREENS),
                    "rank_excluded": True,
                    "scientific_result": False,
                }
        print(json.dumps(summary, indent=1, sort_keys=True, default=str))
        return 0
    except V543CampaignError as exc:
        print(f"V543 campaign refused: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # noqa: BLE001 - typed conversion, re-raise below
        typed = _campaign_refusal_from_exception(exc)
        if typed is None:
            raise
        print(f"V543 campaign refused: {typed}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
