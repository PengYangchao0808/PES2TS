"""Dual-TS free-run harness (WP4-B, plan v4 §G 自由运行门; G4 §6 BINDING).

FREE-RUN semantics (G4 §6, binding): roll out the SELECTED BC policy from
the held monitor endpoint states with NO reference next-frame correction
anywhere in the loop — the action is ``a = a_det(μ)`` (primary; sampled
variant behind a flag), ``X' = X + a``, and the state features are
RECOMPUTED at consume time with the frozen spec (scales + centroid rule
reused verbatim via :func:`pes2ts_core.pesn.features.dual_ts_graph.
build_state_features`).  The teacher path is loaded ONLY by
:func:`post_hoc_truth_metrics` for the deviation metric via the audited
reader pattern (:meth:`DualTSTeacher.from_payload` full contract
validation) — teacher geometry is offline label data, never online input.

Real per-step E/g: every visited state is observed through the WP2 ACP
xTB driver receipt machinery (content-addressed receipts, same
computation-identity scheme ``eg_single_point``), separately billed with a
pre-registered QC budget cap + hard abort (NO interpolated/fake
observations; declared observation mode primary = ``real_per_step_eg``;
the missing-mask path stays tested at fixture level only).

Independent action checks as guard rails (plan §I.3): numerical validity,
scale bounds (a_max), catastrophic overlap.  Any rejected action/state
→ ABORT with the reason and the offending geometry saved verbatim.

Metrics (per rollout + aggregate): cumulative geometric deviation vs the
teacher path where they overlap (registration = the SAME frozen
mass-weighted Kabsch rule, :func:`kabsch_register`), per-step geometry
quality (min interatomic distance, bond validity via the frozen covalent
rule, a_max clip/saturation rate), termination behavior, remaining-arc to
TS as ORACLE DIAGNOSTIC ONLY (recorded, never fed to the policy, never a
reward), and real cost (QC calls, receipts, wall time).

Artifacts: per-rollout records as versioned bundles — a validated
``pesn_dual_ts_bundle_v2`` binding the parent checkpoint digest + spec
digest + action version, plus an additive rollout record
(``pesn_wp4_freerun_rollout_v1``) with a content digest.  Deviations and
failures are saved verbatim.
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Final, Sequence

import numpy as np

from pes2ts_core.generation.planning.synchronized_path import digest
from pes2ts_core.generation.planning.gradient_evidence import BOHR_ANGSTROM
from pes2ts_core.integration.acp.gradient_backend import (
    ORCAGradientBackend,
    write_json as _atomic_write_json,
)
from pes2ts_core.integration.acp.direct_observation import symbols_for
from pes2ts_core.pesn.contracts.dual_ts import (
    BUNDLE_SCHEMA_VERSION,
    DualTSTeacher,
    FEATURES_SCHEMA_VERSION,
    content_digest_of,
    validate_record,
)
from pes2ts_core.pesn.features.dual_ts_graph import (
    COVALENT_BOND_TOL_ANGSTROM,
    build_state_features,
)
from pes2ts_core.pesn.features.dual_ts_spec import load_spec_payload
from pes2ts_core.pesn.features.geometry import covalent_radius
from pes2ts_core.pesn.models.dual_ts_policy import DualTSPolicy
from pes2ts_core.pesn.offline.data.direct_preprocess import (
    kabsch_register,
    masses_for,
)
from pes2ts_core.pesn.offline.data.dual_ts_fit50 import (
    atom_order_hash,
    computation_identity,
    geometry_content_digest,
)
from pes2ts_core.pesn.offline.training.dual_ts_bc import (
    DENSITY_TARGET_VERSION,
    load_spec_and_manifest,
    sha256_file,
)

FREERUN_VERSION: Final[str] = "pesn_dual_ts_freerun_v1.1"
ROLLOUT_RECORD_SCHEMA: Final[str] = "pesn_wp4_freerun_rollout_v1"
FREERUN_RUN_SCHEMA: Final[str] = "pesn_wp4_freerun_run_v1"
OBS_LEDGER_SCHEMA: Final[str] = "pes2ts_dual_ts_freerun_obs_ledger_v0"

#: Declared observation mode (G4 §6): real per-step E/g is PRIMARY; the
#: missing-mask path exists in the spec but stays tested at fixture level.
OBSERVATION_MODE_PRIMARY: Final[str] = "real_per_step_eg"

#: Coverage-node operation string (WP2 identity scheme) — reusing it makes
#: the rollout start-state receipts content-address to the SAME receipts the
#: monitor campaign already computed (no double billing for the start frame).
OPERATION: Final[str] = "eg_single_point"
DEFAULT_METHOD: Final[str] = "GFN2-xTB"

#: Independent action checks (plan §I.3) — guard-rail constants:
#: (i) catastrophic-overlap hard floor: any interatomic pair closer than
#: this is a nonphysical collapse → ABORT (typical bonds are ≥ 0.7 Å);
#: (ii) a_max scale-bound relative tolerance;
#: (iii) saturation band: atoms within (1 - sat_tol)·a_max count as clipped.
OVERLAP_ABORT_MIN_DISTANCE_ANGSTROM: Final[float] = 0.4
A_MAX_REL_TOL: Final[float] = 1e-6
A_MAX_SATURATION_REL: Final[float] = 1e-9

#: Rollout population: the monitor8 wave, branch A consistently (one branch
#: per group — the dispatched plan).
BRANCH_LABEL: Final[str] = "A"

LEDGER_NAME: Final[str] = "compute_ledger.json"


class FreerunError(RuntimeError):
    """Typed free-run refusal (carries a machine-readable code)."""

    def __init__(self, code: str, detail: str):
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


class QCBudgetExhausted(FreerunError):
    """Hard abort at the pre-registered QC budget cap (G4 §6)."""


# ---------------------------------------------------------------------------
# Receipt machinery (same contract as the WP2 driver; drift-guarded in tests)
# ---------------------------------------------------------------------------


def _finite_float(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def validate_receipt_pair(request: object, result: object,
                          n_atoms: int) -> tuple[bool, str | None]:
    """Validate one receipt (``request.json`` + ``result.json``) pair.

    Mirrors the WP2 driver's ``validate_receipt`` (request re-bind, result
    binding, success, finite energy, (n_atoms, 3) finite gradient); a test
    drift-guard asserts the two accept/reject identically.  Invalid receipts
    are never zero-filled — they are failures.
    """
    if not isinstance(request, dict):
        return False, "request_not_a_mapping"
    if not isinstance(result, dict):
        return False, "result_not_a_mapping"
    binding = str(request.get("request_sha256") or "")
    if not binding:
        return False, "missing_request_sha256"
    rebind_payload = {
        k: v for k, v in request.items()
        if k != "request_sha256" and k != "acp_request_sha256"
        and not str(k).startswith("pes2ts_")
    }
    try:
        rebind = digest(rebind_payload)
    except (TypeError, ValueError) as exc:
        return False, f"request_rebind_failed:{exc}"
    if rebind != binding:
        return False, f"request_binding_mismatch:stored={binding}:rebind={rebind}"
    if str(result.get("request_sha256") or "") != binding:
        return False, "result_request_sha256_mismatch"
    if result.get("success") is not True:
        return False, f"success_not_true:{result.get('success')!r}:" \
                      f"{result.get('failure_class')}"
    energy = _finite_float(result.get("energy"))
    if energy is None:
        return False, f"energy_not_finite:{result.get('energy')!r}"
    gradient = result.get("gradient_hartree_per_angstrom")
    if not isinstance(gradient, list) or len(gradient) != n_atoms:
        return False, (f"gradient_missing_or_wrong_shape:n_atoms={n_atoms}:got="
                       f"{len(gradient) if isinstance(gradient, list) else type(gradient).__name__}")
    for i, row in enumerate(gradient):
        if not isinstance(row, list) or len(row) != 3:
            return False, f"gradient_row_{i}_not_3_components"
        for j, v in enumerate(row):
            if _finite_float(v) is None:
                return False, f"gradient_[{i}][{j}]_not_finite:{v!r}"
    return True, None


def observation_settings_digest(acp: dict) -> str:
    """Settings digest over the observation identity (driver formula)."""
    payload = {
        "workflow": "OrcaGradient",
        "method": str(acp["method"]),
        "basis": "",
        "route_extras": ["EnGrad"],
        "extra_blocks": ['%xtb\n XTBINPUTSTRING "--iterations 2000"\nend'],
        "scf_convergence": "tight",
        "output_name": "grad",
        "timeout_seconds": float(acp["timeout_seconds"]),
        "nproc": int(acp["nproc"]),
        "acp_root": str(acp["root"]),
        "acp_python": str(acp["python"]),
        "acp_config_path": str(acp["config_path"]),
    }
    return digest(payload)


def receipt_dir(out_root: str | Path, digest_hex: str) -> Path:
    return Path(out_root) / digest_hex[:2] / digest_hex


def _load_receipt_pair(rd: Path) -> tuple[dict, dict]:
    request = json.loads((rd / "request.json").read_text(encoding="utf-8"))
    result = json.loads((rd / "result.json").read_text(encoding="utf-8"))
    return request, result


def _receipt_state(root: Path, identity_digest: str,
                   n_atoms: int) -> tuple[bool, str | None, dict, dict]:
    """``(valid, reason, request, result)``; absent → ``(False, None, {}, {})``."""
    rd = receipt_dir(root, identity_digest)
    if not (rd / "result.json").is_file():
        return False, None, {}, {}
    try:
        request, result = _load_receipt_pair(rd)
    except (OSError, json.JSONDecodeError) as exc:
        return False, f"receipt_unreadable:{exc}", {}, {}
    ok, reason = validate_receipt_pair(request, result, n_atoms)
    return ok, reason, request, result


def annotate_freerun_receipt(rd: Path, identity: dict, acp: dict,
                             node_label: str) -> None:
    """Additive identity + settings/units annotations on a fresh receipt."""
    settings = {
        "scheme": "pes2ts_dual_ts_freerun_settings_v0",
        "settings_digest": observation_settings_digest(acp),
        "workflow": "OrcaGradient",
        "method": str(acp["method"]),
        "timeout_seconds": float(acp["timeout_seconds"]),
        "nproc": int(acp["nproc"]),
        "acp_root": str(acp["root"]),
        "acp_python": str(acp["python"]),
        "acp_config_path": str(acp["config_path"]),
    }
    freerun_identity = {
        "scheme": "pes2ts_dual_ts_freerun_identity_v0",
        "computation_identity": identity["digest"],
        "geometry_content_digest": identity["geometry_content_digest"],
        "atom_order_hash": identity["atom_order_hash"],
        "method": identity["method"],
        "operation": identity["operation"],
        "charge": identity["charge"],
        "multiplicity": identity["multiplicity"],
        "node_label": node_label,
        "driver_version": FREERUN_VERSION,
    }
    units = {
        "scheme": "pes2ts_dual_ts_freerun_units_v0",
        "energy": "hartree",
        "gradient_hartree_per_angstrom": "hartree/angstrom",
        "gradient_native_field":
            "gradient_evidence.gradient_hartree_per_bohr",
        "gradient_native_unit": "hartree/bohr",
        "conversion": ("gradient_hartree_per_angstrom = "
                       "gradient_evidence.gradient_hartree_per_bohr / "
                       "BOHR_ANGSTROM"),
        "BOHR_ANGSTROM": BOHR_ANGSTROM,
        "recorded_by": FREERUN_VERSION,
    }
    request_path = rd / "request.json"
    request = json.loads(request_path.read_text(encoding="utf-8"))
    for key, value in (("pes2ts_computation_identity", freerun_identity),
                       ("pes2ts_settings", settings)):
        if key not in request:
            request[key] = value
    _atomic_write_json(request_path, request)
    result_path = rd / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    for key, value in (("pes2ts_computation_identity", freerun_identity),
                       ("pes2ts_units", units)):
        if key not in result:
            result[key] = value
    _atomic_write_json(result_path, result)


# ---------------------------------------------------------------------------
# Observer: real per-step E/g via the ACP receipt machinery
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ObservationCall:
    """Outcome of ONE observation request for a visited state."""

    digest: str
    status: str            # computed | resume_valid | reused_fallback | failed
    billed: bool
    energy: float | None
    gradient: np.ndarray | None
    backend_seconds: float | None
    receipt_dir: str | None
    provenance: str
    error: str | None


class FreerunObserver:
    """Content-addressed E/g observation source for rollout states.

    Lookup order: the freerun receipt root first, then read-only fallback
    roots (the monitor observation store — same computation identity = the
    same receipt, so the start frame costs zero new QC).  Only receipts that
    pass :func:`validate_receipt_pair` count as valid; anything invalid is
    recomputed (fresh compute bills against the pre-registered QC budget,
    with a HARD abort at the cap — G4 §6).
    """

    def __init__(
        self,
        *,
        out_root: str | Path,
        acp: dict,
        qc_budget: int,
        fallback_roots: Sequence[str | Path] = (),
        method: str = DEFAULT_METHOD,
        backend_factory: Callable[..., object] | None = None,
    ) -> None:
        self.out_root = Path(out_root)
        self.out_root.mkdir(parents=True, exist_ok=True)
        self.acp = dict(acp)
        self.acp.setdefault("method", method)
        self.qc_budget = int(qc_budget)
        if self.qc_budget <= 0:
            raise FreerunError("QC_BUDGET_INVALID", "must be positive")
        self.fallback_roots = [Path(p) for p in fallback_roots]
        self.method = str(self.acp.get("method", method))
        self._backend_factory = backend_factory
        self.billed_count = 0
        self.reused_count = 0
        self.failed_count = 0
        self.backend_seconds: list[float] = []
        self.events: list[dict] = []

    # -- receipt identity ----------------------------------------------------

    def identity_for(self, z: np.ndarray, coords: np.ndarray,
                     maps: Sequence[str]) -> dict:
        identity = computation_identity(
            geometry_content_digest(np.asarray(coords, dtype=float)),
            atom_order_hash(np.asarray(z, dtype=int), list(maps)),
            self.method, 0, 1, OPERATION)
        return identity

    # -- lookup + compute ----------------------------------------------------

    def observe(self, z: np.ndarray, coords: np.ndarray,
                maps: Sequence[str], *, charge: int, multiplicity: int,
                node_label: str) -> ObservationCall:
        identity = self.identity_for(z, coords, maps)
        d = str(identity["digest"])
        n_atoms = int(np.asarray(z).size)

        # 1) own receipt root (resume)
        ok, reason, request, result = _receipt_state(
            self.out_root, d, n_atoms)
        if ok:
            return self._reuse(d, result, self.out_root, "resume_valid")
        # 2) read-only fallback roots (e.g. the monitor observation store)
        for root in self.fallback_roots:
            ok, reason, request, result = _receipt_state(root, d, n_atoms)
            if ok:
                return self._reuse(d, result, root, "reused_fallback")
        # (invalid receipts are recorded then recomputed — never zero-filled)
        # 3) hard QC budget gate BEFORE any new billed compute (G4 §6)
        if self.billed_count >= self.qc_budget:
            raise QCBudgetExhausted(
                "QC_BUDGET_EXHAUSTED",
                f"pre-registered budget {self.qc_budget} reached before "
                f"observing {node_label} (digest {d[:12]}…) — hard abort")
        # 4) real ACP compute
        started = time.monotonic()
        try:
            if self._backend_factory is not None:
                backend = self._backend_factory(
                    charge=int(charge), multiplicity=int(multiplicity),
                    elements=symbols_for(z), folder=str(self.out_root),
                    method=self.method,
                    timeout_seconds=float(self.acp["timeout_seconds"]),
                    nproc=int(self.acp["nproc"]),
                    acp_wiring={"root": self.acp["root"],
                                "python": self.acp["python"],
                                "config_path": self.acp["config_path"]})
            else:
                backend = ORCAGradientBackend(
                    charge=int(charge), multiplicity=int(multiplicity),
                    elements=symbols_for(z), folder=str(self.out_root),
                    method=self.method,
                    timeout_seconds=float(self.acp["timeout_seconds"]),
                    nproc=int(self.acp["nproc"]),
                    acp_wiring={"root": self.acp["root"],
                                "python": self.acp["python"],
                                "config_path": self.acp["config_path"]})
            record = backend(np.asarray(coords, dtype=float),
                             f"{d[:2]}/{d}")
        except Exception as exc:  # transport/launch fault → registered failure
            self.failed_count += 1
            self.events.append({"node_label": node_label, "digest": d,
                                "status": "failed",
                                "error": f"backend_exception:"
                                         f"{type(exc).__name__}:{exc}",
                                "seconds": round(time.monotonic() - started, 4)})
            return ObservationCall(
                digest=d, status="failed", billed=False, energy=None,
                gradient=None, backend_seconds=None, receipt_dir=None,
                provenance="compute", error=self.events[-1]["error"])
        rd = receipt_dir(self.out_root, d)
        ok, reason = validate_receipt_pair(
            json.loads((rd / "request.json").read_text(encoding="utf-8"))
            if (rd / "request.json").is_file() else {},
            record, n_atoms)
        if not ok:
            self.failed_count += 1
            self.events.append({"node_label": node_label, "digest": d,
                                "status": "failed", "error": reason,
                                "seconds": round(time.monotonic() - started, 4)})
            return ObservationCall(
                digest=d, status="failed", billed=False, energy=None,
                gradient=None, backend_seconds=None,
                receipt_dir=str(rd), provenance="compute", error=reason)
        annotate_freerun_receipt(rd, identity, self.acp, node_label)
        gradient = np.asarray(record["gradient_hartree_per_angstrom"],
                              dtype=np.float64)
        backend_seconds = float(record.get("duration_seconds") or 0.0)
        self.billed_count += 1
        self.backend_seconds.append(backend_seconds)
        self.events.append({"node_label": node_label, "digest": d,
                            "status": "computed", "billed": True,
                            "backend_seconds": round(backend_seconds, 4),
                            "seconds": round(time.monotonic() - started, 4)})
        return ObservationCall(
            digest=d, status="computed", billed=True,
            energy=float(record["energy"]), gradient=gradient,
            backend_seconds=backend_seconds, receipt_dir=str(rd),
            provenance="compute", error=None)

    def _reuse(self, d: str, result: dict, root: Path,
               status: str) -> ObservationCall:
        gradient = np.asarray(result["gradient_hartree_per_angstrom"],
                              dtype=np.float64)
        self.reused_count += 1
        self.events.append({"digest": d, "status": status,
                            "root": str(root), "billed": False})
        return ObservationCall(
            digest=d, status=status, billed=False,
            energy=float(result["energy"]), gradient=gradient,
            backend_seconds=None, receipt_dir=str(receipt_dir(root, d)),
            provenance=str(root), error=None)

    # -- ledger ---------------------------------------------------------------

    def write_ledger(self, *, run_id: str, wall_seconds: float,
                     extra: dict | None = None) -> Path:
        path = self.out_root / LEDGER_NAME
        ledger = json.loads(path.read_text(encoding="utf-8")) \
            if path.is_file() else {"schema_version": OBS_LEDGER_SCHEMA,
                                    "runs": []}
        per_qc = sorted(self.backend_seconds)
        entry = {
            "run_id": run_id, "ended_utc": time.strftime(
                "%Y-%m-%dT%H:%M:%S", time.gmtime()),
            "driver_version": FREERUN_VERSION,
            "out_root": str(self.out_root),
            "fallback_roots": [str(p) for p in self.fallback_roots],
            "qc_budget": self.qc_budget,
            "billed": self.billed_count, "reused": self.reused_count,
            "failed": self.failed_count,
            "settings_digest": observation_settings_digest(self.acp),
            "acp": {k: self.acp[k] for k in
                    ("root", "python", "config_path", "method",
                     "timeout_seconds", "nproc") if k in self.acp},
            "backend_seconds_median": (
                float(np.median(per_qc)) if per_qc else None),
            "backend_seconds_total": round(sum(per_qc), 4),
            "per_receipt_backend_seconds": {
                e["digest"]: e["backend_seconds"] for e in self.events
                if e.get("billed")},
            "wall_seconds": round(wall_seconds, 3),
            **(extra or {}),
        }
        ledger["runs"].append(entry)
        _atomic_write_json(path, ledger)
        return path


# ---------------------------------------------------------------------------
# Identity gates: model loading with bundle/spec/action binding
# ---------------------------------------------------------------------------


def load_selected_policy(
    torch,
    *,
    checkpoint_path: str | Path,
    bundle_path: str | Path,
    features_dir: str | Path,
    selected_recipe: str | None = None,
) -> tuple[DualTSPolicy, dict]:
    """Load the SELECTED BC checkpoint behind hard identity gates.

    Gates (refusal on any mismatch, per the dispatch): (i) the bundle is a
    VALID ``pesn_dual_ts_bundle_v2`` whose ``bc_parent_checkpoint_digest``
    equals the sha256 of the checkpoint file bytes; (ii) the spec digest is
    the SAME across bundle / checkpoint / the (monitor) features manifest —
    recomputed from ``features_spec.json`` bytes, never trusted from text;
    (iii) the action version binds ``radial_tanh_v0:bc_target=
    teacher_latent_v1`` with the checkpoint's a_max/sigma range; (iv) when a
    selection record is supplied, the checkpoint recipe must be the SELECTED
    recipe.
    """
    checkpoint_path = Path(checkpoint_path)
    bundle_path = Path(bundle_path)
    if not checkpoint_path.is_file():
        raise FreerunError("CHECKPOINT_MISSING", str(checkpoint_path))
    if not bundle_path.is_file():
        raise FreerunError("BUNDLE_MISSING", str(bundle_path))
    ckpt_sha = sha256_file(checkpoint_path)
    bundle_payload = json.loads(bundle_path.read_text(encoding="utf-8"))
    if bundle_payload.get("schema_version") != BUNDLE_SCHEMA_VERSION:
        raise FreerunError(
            "BUNDLE_SCHEMA_UNKNOWN",
            str(bundle_payload.get("schema_version")))
    validate_record(BUNDLE_SCHEMA_VERSION, bundle_payload)
    if str(bundle_payload.get("bc_parent_checkpoint_digest")) != ckpt_sha:
        raise FreerunError(
            "CHECKPOINT_BUNDLE_MISMATCH",
            f"bundle bc_parent_checkpoint_digest "
            f"{bundle_payload.get('bc_parent_checkpoint_digest')[:12]}… != "
            f"checkpoint sha256 {ckpt_sha[:12]}…")
    ckpt = torch.load(checkpoint_path, map_location="cpu",
                      weights_only=False)
    if ckpt.get("schema_version") != "pesn_checkpoint_v1":
        raise FreerunError("CHECKPOINT_SCHEMA_UNKNOWN",
                           str(ckpt.get("schema_version")))
    extra = ckpt["extra"]
    settings = extra["settings"]
    # (ii) spec digest: bundle == checkpoint == recomputed manifest digest
    spec_digest, _manifest = load_spec_and_manifest(features_dir)
    bundle_spec = str(bundle_payload.get("input_eligibility", {})
                      .get("spec_digest", ""))
    if not (bundle_spec == str(extra["spec_digest"]) == spec_digest):
        raise FreerunError(
            "SPEC_DIGEST_MISMATCH",
            f"bundle {bundle_spec[:12]}… checkpoint "
            f"{str(extra['spec_digest'])[:12]}… recomputed "
            f"{spec_digest[:12]}…")
    # (iii) action version binding
    action_version = str(bundle_payload.get("action_version", ""))
    required = (f"{DENSITY_TARGET_VERSION}:a_max={float(settings['action_max'])}:"
                f"sigma_min={float(settings['sigma_min'])}")
    if action_version != required or \
            str(extra.get("density_target_version")) != DENSITY_TARGET_VERSION:
        raise FreerunError(
            "ACTION_VERSION_MISMATCH",
            f"bundle action_version {action_version!r} != required "
            f"{required!r}")
    # (iv) selection cross-check
    if selected_recipe is not None and str(settings["recipe"]) != \
            str(selected_recipe):
        raise FreerunError(
            "SELECTED_MODEL_MISMATCH",
            f"checkpoint recipe {settings['recipe']!r} is not the selected "
            f"recipe {selected_recipe!r}")
    from pes2ts_core.pesn.models.dual_egnn import DualTSModelConfig
    policy = DualTSPolicy(
        config=DualTSModelConfig(
            light=bool(settings["light"]),
            signed_vector_messages=bool(extra["signed_vector_messages"])),
        action_max=float(settings["action_max"]),
        sigma_range=(float(settings["sigma_min"]),
                     float(settings["sigma_max"])),
        signed_mu_gate=bool(extra["signed_mu_gate"]),
    )
    policy.load_state_dicts(ckpt["model_state"])
    identity = {
        "checkpoint_sha256": ckpt_sha,
        "checkpoint_file": str(checkpoint_path),
        "bundle_content_digest": content_digest_of(
            dict(bundle_payload)),
        "bundle_bc_parent_digest": ckpt_sha,
        "run_id": str(bundle_payload["run_id"]),
        "recipe": str(settings["recipe"]),
        "signed_mu_gate": bool(extra["signed_mu_gate"]),
        "signed_vector_messages": bool(extra["signed_vector_messages"]),
        "density_target_version": str(extra["density_target_version"]),
        "action_version": action_version,
        "spec_digest": spec_digest,
        "action_max": float(settings["action_max"]),
        "sigma_min": float(settings["sigma_min"]),
        "sigma_max": float(settings["sigma_max"]),
        "train_seed": int(settings["seed"]),
    }
    return policy, identity


# ---------------------------------------------------------------------------
# Rollout start state + frozen topology (monitor wave, branch A)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RolloutStart:
    """Initial condition + frozen per-reaction topology of ONE rollout.

    Holds ONLY the endpoint frame identity (coords/z/maps) and the frozen
    dual-topology blocks — never teacher labels/actions/remaining-arc
    (those are loaded exclusively by :func:`post_hoc_truth_metrics`).
    """

    group_id: str
    reaction_id: str
    branch_id: str
    episode_id: str
    start_state_id: str
    atomic_numbers: np.ndarray          # (N,) int
    coordinates: np.ndarray             # (N,3) float64, RAW exported X
    atom_map_ids: tuple[str, ...]
    charge: int
    multiplicity: int
    edge_pairs: np.ndarray              # (E,2) frozen union edges
    edge_index: np.ndarray              # (2,2E)
    atom_feats_a: np.ndarray            # (N,4)
    atom_feats_b: np.ndarray            # (N,4)
    edge_feats_a: np.ndarray            # (2E,4)
    edge_feats_b: np.ndarray            # (2E,4)
    bond_source_a: str
    bond_source_b: str

    @property
    def n_atoms(self) -> int:
        return int(self.atomic_numbers.size)


def load_rollout_start(
    *,
    features_dir: str | Path,
    observed_dir: str | Path,
    group_id: str,
    branch_label: str = BRANCH_LABEL,
    expected_spec_digest: str | None = None,
) -> RolloutStart:
    """Endpoint frame + frozen topology for one monitor group.

    The feature shard is re-verified (sha256 vs manifest; manifest spec
    digest recomputed) and supplies ONLY the frozen topology blocks.  The
    start frame comes from the observed teacher payload via the audited
    reader (:meth:`DualTSTeacher.from_payload` — full contract validation).
    """
    features_dir = Path(features_dir)
    observed_dir = Path(observed_dir)
    spec_digest, manifest = load_spec_and_manifest(features_dir)
    if expected_spec_digest is not None and spec_digest != \
            expected_spec_digest:
        raise FreerunError(
            "SPEC_DIGEST_MISMATCH",
            f"features manifest spec {spec_digest[:12]}… != expected "
            f"{expected_spec_digest[:12]}…")
    rows = [r for r in manifest["per_reaction"]
            if str(r["reaction_group_id"]) == str(group_id)]
    if not rows:
        raise FreerunError("GROUP_NOT_IN_WAVE",
                           f"group {group_id} not in the feature manifest")
    row = rows[0]
    shard_path = features_dir / row["npz_file"]
    if sha256_file(shard_path) != row["npz_sha256"]:
        raise FreerunError(
            "SHARD_DIGEST_MISMATCH",
            f"{shard_path.name} fails its manifest sha256")
    teacher = DualTSTeacher.from_payload(json.loads(
        (observed_dir / f"{row['reaction_id']}.teacher.json").read_text(
            encoding="utf-8")))
    branch = next((b for b in teacher.branches
                   if b.endpoint_label == str(branch_label)), None)
    if branch is None:
        raise FreerunError(
            "BRANCH_MISSING",
            f"group {group_id} has no endpoint-{branch_label} branch")
    episode = next(e for e in teacher.episodes
                   if e.branch_id == branch.branch_id)
    state = next(s for s in teacher.states
                 if s.state_id == episode.start_state_id)
    task_payload = json.loads(
        (observed_dir / f"{row['reaction_id']}.task.json").read_text(
            encoding="utf-8"))
    shard = np.load(shard_path)
    return RolloutStart(
        group_id=str(group_id),
        reaction_id=str(row["reaction_id"]),
        branch_id=branch.branch_id,
        episode_id=episode.episode_id,
        start_state_id=state.state_id,
        atomic_numbers=np.asarray(state.atomic_numbers, dtype=np.int64),
        coordinates=np.asarray(state.coordinates, dtype=np.float64),
        atom_map_ids=tuple(str(m) for m in state.atom_map_ids),
        charge=int(task_payload.get("charge", 0)),
        multiplicity=int(task_payload.get("multiplicity", 1)),
        edge_pairs=np.asarray(shard["edge_pairs"], dtype=np.int64),
        edge_index=np.asarray(shard["edge_index"], dtype=np.int64),
        atom_feats_a=np.asarray(shard["atom_feats_a"], dtype=np.float64),
        atom_feats_b=np.asarray(shard["atom_feats_b"], dtype=np.float64),
        edge_feats_a=np.asarray(shard["edge_feats_a"], dtype=np.float64),
        edge_feats_b=np.asarray(shard["edge_feats_b"], dtype=np.float64),
        bond_source_a=str(shard["bond_source_a"]),
        bond_source_b=str(shard["bond_source_b"]),
    )


def load_frozen_scales(features_dir: str | Path) -> dict:
    """The frozen spec scales, verified against the saved spec bytes."""
    spec, _payload = load_spec_payload(
        Path(features_dir) / "features_spec.json")
    return {
        "s_g": float(spec.frozen_scales["s_g"]),
        "s_delta_e": float(spec.frozen_scales["s_delta_e"]),
        "spec_digest": spec.content_digest(),
    }


# ---------------------------------------------------------------------------
# Consume-time state assembly (frozen spec, no truth access)
# ---------------------------------------------------------------------------


def build_rollout_state_inputs(
    *,
    start: RolloutStart,
    state_id: str,
    coordinates: np.ndarray,
    gradient_hartree_per_angstrom: np.ndarray,
    delta_e_hartree: float,
    scales: dict,
) -> tuple[dict, object]:
    """One visited state's model inputs, RECOMPUTED at consume time.

    Reuses :func:`build_state_features` verbatim (frozen scales + consume-time
    centroid removal); the raw coordinates are passed as ``x_angstrom`` and
    the collation layer applies the SAME frozen centroid rule the offline
    shards consumed (idempotent — byte-identical inputs).  Observation-only:
    NO teacher/reference/arriving-action data enters here by construction
    (v4.1 WP1 — the online path calls the SAME observation-only builder as
    offline, with observation args only).
    """
    features = build_state_features(
        state_id=state_id,
        atomic_numbers=start.atomic_numbers,
        coordinates=coordinates,
        gradient=gradient_hartree_per_angstrom,
        delta_e_hartree=float(delta_e_hartree),
        s_g=float(scales["s_g"]),
        s_delta_e=float(scales["s_delta_e"]),
        charge=start.charge,
        multiplicity=start.multiplicity,
    )
    inputs = {
        "atomic_numbers": start.atomic_numbers,
        "node_scalars": features.node_scalars,
        "x_angstrom": np.asarray(coordinates, dtype=np.float64),
        "g_scaled": features.g_scaled,
        "atom_feats_a": start.atom_feats_a,
        "atom_feats_b": start.atom_feats_b,
        "edge_feats_a": start.edge_feats_a,
        "edge_feats_b": start.edge_feats_b,
        "edge_index": start.edge_index,
    }
    return inputs, features


# ---------------------------------------------------------------------------
# Guard rails (plan §I.3 independent action checks)
# ---------------------------------------------------------------------------


def pairwise_min_distance(coords: np.ndarray) -> float:
    delta = coords[:, None, :] - coords[None, :, :]
    dmat = np.sqrt((delta * delta).sum(-1))
    np.fill_diagonal(dmat, np.inf)
    return float(dmat.min())


def bond_validity(coords: np.ndarray, z: np.ndarray,
                  edge_pairs: np.ndarray) -> dict:
    """Frozen covalent-rule validity over the frozen union edge set."""
    if edge_pairs.size == 0:
        return {"n_bonds": 0, "broken_fraction": 0.0, "max_excess_A": 0.0}
    excess = []
    for i, j in edge_pairs:
        i, j = int(i), int(j)
        d = float(np.linalg.norm(coords[i] - coords[j]))
        limit = covalent_radius(int(z[i])) + covalent_radius(int(z[j])) \
            + COVALENT_BOND_TOL_ANGSTROM
        excess.append(max(0.0, d - limit))
    excess_arr = np.asarray(excess, dtype=np.float64)
    return {
        "n_bonds": int(excess_arr.size),
        "broken_fraction": float((excess_arr > 0.0).mean()),
        "max_excess_A": float(excess_arr.max()),
    }


# ---------------------------------------------------------------------------
# Settings + rollout engine
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RolloutSettings:
    """One free-run campaign's preregistered execution settings."""

    max_steps: int = 40
    qc_budget: int = 320
    action_mode: str = "deterministic"       # | "sampled" (optional variant)
    seed: int = 20261017
    branch_label: str = BRANCH_LABEL
    overlap_min_distance_angstrom: float = OVERLAP_ABORT_MIN_DISTANCE_ANGSTROM
    record_inputs: bool = False              # test-only input probe toggle

    def __post_init__(self) -> None:
        if self.max_steps <= 0 or self.qc_budget <= 0:
            raise FreerunError("SETTINGS_INVALID", "non-positive steps/budget")
        if self.action_mode not in ("deterministic", "sampled"):
            raise FreerunError("SETTINGS_INVALID",
                               f"action_mode {self.action_mode!r} unknown")

    def to_payload(self) -> dict:
        return {
            "max_steps": int(self.max_steps),
            "qc_budget": int(self.qc_budget),
            "action_mode": self.action_mode,
            "seed": int(self.seed),
            "branch_label": self.branch_label,
            "overlap_min_distance_angstrom":
                float(self.overlap_min_distance_angstrom),
            "a_max_rel_tol": A_MAX_REL_TOL,
            "observation_mode": OBSERVATION_MODE_PRIMARY,
        }


def _round10(a: np.ndarray) -> list:
    return np.round(np.asarray(a, dtype=np.float64), 10).tolist()


def run_rollout(
    *,
    torch,
    model,
    observer: FreerunObserver,
    start: RolloutStart,
    scales: dict,
    identity: dict,
    settings: RolloutSettings,
    group_index: int = 0,
    input_probe: Callable[..., None] | None = None,
) -> dict:
    """FREE-RUN rollout from one held endpoint (G4 §6).

    NO reference next-frame correction anywhere: the carried state is
    exactly ``X' = X + a``; features are recomputed at consume time; the
    model sees ONLY the current state's inputs (``input_probe`` exists for
    tests).  Teacher data is not an input of this function at all.
    """
    device = next(model.parameters()).device
    model.train_mode(False)
    z = np.asarray(start.atomic_numbers, dtype=np.int64)
    x = np.asarray(start.coordinates, dtype=np.float64).copy()
    e_start: float | None = None
    a_max = float(identity["action_max"])
    steps: list[dict] = []
    billed_at_start = observer.billed_count
    t0 = time.perf_counter()
    generator = None
    if settings.action_mode == "sampled":
        seed = int(np.random.default_rng(
            [int(settings.seed), int(group_index)]).integers(0, 2**31 - 1))
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)

    def abort(reason: str, step: int, detail: str,
              offending_x=None, offending_a=None) -> dict:
        block = {
            "status": "aborted", "reason": reason, "step": int(step),
            "detail": detail,
        }
        if offending_x is not None:
            block["offending_geometry_round10"] = _round10(offending_x)
        if offending_a is not None:
            block["offending_action_round10"] = _round10(offending_a)
        return block

    termination: dict | None = None
    for t in range(int(settings.max_steps)):
        state_id = f"{start.group_id}|{settings.branch_label}|free{t:04d}"
        # -- real per-step observation (content-addressed; billed or reused)
        try:
            obs = observer.observe(
                z, x, start.atom_map_ids, charge=start.charge,
                multiplicity=start.multiplicity, node_label=state_id)
        except QCBudgetExhausted as exc:
            termination = abort("qc_budget_exhausted", t, exc.detail)
            break
        if obs.status == "failed":
            termination = abort("observation_failed", t, str(obs.error))
            break
        if e_start is None:
            e_start = float(obs.energy)
            delta_e = 0.0
        else:
            delta_e = float(obs.energy) - e_start
        # -- consume-time feature recomputation (frozen spec)
        inputs, features = build_rollout_state_inputs(
            start=start, state_id=state_id, coordinates=x,
            gradient_hartree_per_angstrom=obs.gradient,
            delta_e_hartree=delta_e, scales=scales)
        # -- policy forward on CURRENT state inputs only
        from pes2ts_core.pesn.models.dual_egnn import collate_dual_states
        batch = collate_dual_states(torch, [inputs])
        for key, val in batch.items():
            if torch.is_tensor(val):
                if val.dtype in (torch.float64, torch.float32):
                    val = val.to(dtype=torch.float32)
                batch[key] = val.to(device)
        with torch.no_grad():
            out = model(batch)
        mu = out["mu"]
        mu_np = mu.detach().cpu().numpy().astype(np.float64)
        sigma_np = out["sigma"].detach().cpu().numpy().astype(np.float64)
        if settings.action_mode == "sampled":
            a_t = model.sample(mu, out["log_sigma"],
                               generator=generator)["a"]
        else:
            a_t = model.deterministic(mu)
        a = a_t.detach().cpu().numpy().astype(np.float64)
        # -- guard (i): numerical validity of the action carrier
        if not np.isfinite(a).all():
            termination = abort("action_nonfinite", t,
                                "action carrier contains NaN/Inf",
                                offending_a=a)
            break
        # -- guard (ii): scale bounds (a_max)
        norms = np.linalg.norm(a, axis=1)
        bound = a_max * (1.0 + A_MAX_REL_TOL)
        if bool((norms > bound).any()):
            termination = abort(
                "action_scale_bound", t,
                f"max |a_i| = {float(norms.max()):.12f} A > "
                f"a_max*(1+{A_MAX_REL_TOL}) = {bound:.12f} A",
                offending_a=a)
            break
        clip_rate = float((norms >= a_max * (1.0 - A_MAX_SATURATION_REL))
                          .mean())
        # -- X' = X + a  (NO reference correction of any kind)
        x_next = x + a
        if not np.isfinite(x_next).all():
            termination = abort("geometry_nonfinite", t,
                                "X' = X + a contains NaN/Inf",
                                offending_x=x_next, offending_a=a)
            break
        # -- guard (iii): catastrophic overlap on the candidate geometry
        min_dist = pairwise_min_distance(x_next)
        if min_dist < float(settings.overlap_min_distance_angstrom):
            termination = abort(
                "catastrophic_overlap", t,
                f"min interatomic distance {min_dist:.6f} A < "
                f"{settings.overlap_min_distance_angstrom:.2f} A floor",
                offending_x=x_next, offending_a=a)
            break
        bond = bond_validity(x_next, z, start.edge_pairs)
        if input_probe is not None:
            # test-only probe: exposes the exact carried state (X' = X + a)
            input_probe(t, inputs, features, obs, delta_e, a, x_next)
        steps.append({
            "step": int(t),
            "state_id": state_id,
            "x_round10": _round10(x),
            "action_round10": _round10(a),
            "mu_norm_mean": float(np.linalg.norm(mu_np, axis=1).mean()),
            "mu_norm_max": float(np.linalg.norm(mu_np, axis=1).max()),
            "sigma_mean": float(sigma_np.mean()),
            "delta_e_hartree": float(delta_e),
            "observation": {
                "digest": obs.digest, "status": obs.status,
                "billed": bool(obs.billed),
                "backend_seconds": obs.backend_seconds,
                "receipt_dir": obs.receipt_dir,
                "provenance": obs.provenance,
                "energy_hartree": (None if obs.energy is None
                                   else float(obs.energy)),
            },
            "geometry_next": {
                "min_interatomic_distance_A": min_dist,
                **bond,
                "a_max_clip_rate": clip_rate,
                "max_action_norm_A": float(norms.max()),
                "a_max_violations": int((norms > bound).sum()),
            },
        })
        if settings.record_inputs:
            steps[-1]["model_inputs_round10"] = {
                "x": _round10(inputs["x_angstrom"]),
                "g_scaled": _round10(inputs["g_scaled"]),
                "node_scalars": _round10(inputs["node_scalars"]),
            }
        x = x_next
    if termination is None:
        termination = {"status": "completed_step_cap", "reason": None,
                       "step": int(settings.max_steps)}
    n_billed = observer.billed_count - billed_at_start
    n_reused = sum(1 for s in steps
                   if s["observation"]["status"] != "computed")
    wall = time.perf_counter() - t0
    record: dict = {
        "schema_version": ROLLOUT_RECORD_SCHEMA,
        "freerun_version": FREERUN_VERSION,
        "group_id": start.group_id,
        "reaction_id": start.reaction_id,
        "branch_id": start.branch_id,
        "branch_label": settings.branch_label,
        "episode_id": start.episode_id,
        "start_state_id": start.start_state_id,
        "settings": settings.to_payload(),
        "identity": dict(identity),
        "topology": {
            "n_atoms": start.n_atoms,
            "n_edges_union": int(start.edge_pairs.shape[0]),
            "bond_source_a": start.bond_source_a,
            "bond_source_b": start.bond_source_b,
        },
        "observation_mode": {
            "primary": OBSERVATION_MODE_PRIMARY,
            "note": "real per-step xTB E/g via ACP receipts; the "
                    "missing-mask path stays fixture-level only (G4 §6)",
        },
        "steps": steps,
        "n_steps": len(steps),
        "termination": termination,
        "cost": {
            "n_observations": len(steps),
            "n_billed": int(n_billed),
            "n_reused": int(n_reused),
            "backend_seconds": round(
                sum(s["observation"]["backend_seconds"] or 0.0
                    for s in steps), 4),
            "wall_seconds": round(wall, 3),
        },
    }
    return record


# ---------------------------------------------------------------------------
# POST-HOC truth metrics (deviation + oracle arc) — reporting ONLY
# ---------------------------------------------------------------------------


def post_hoc_truth_metrics(
    record: dict,
    *,
    observed_dir: str | Path,
) -> dict:
    """Append deviation-vs-teacher + oracle-arc diagnostics to a record.

    The ONLY place teacher data is touched: the monitor teacher payload is
    loaded via the audited reader (:meth:`DualTSTeacher.from_payload`) and
    consumed for REPORTING metrics that are never fed to the policy and
    never a reward.  Registration uses the SAME frozen mass-weighted Kabsch
    rule as the offline path build (:func:`kabsch_register`, v1 mass table).
    """
    observed_dir = Path(observed_dir)
    teacher = DualTSTeacher.from_payload(json.loads(
        (observed_dir / f"{record['reaction_id']}.teacher.json").read_text(
            encoding="utf-8")))
    branch = next(b for b in teacher.branches
                  if b.endpoint_label == str(record["branch_label"]))
    ordered = sorted(
        (s for s in teacher.states if s.branch_id == branch.branch_id),
        key=lambda s: s.step_index)
    labels = {lab.state_id: lab for lab in teacher.labels}
    ts_state = next(s for s in ordered
                    if s.boundary_type == "teacher_endpoint")
    z = np.asarray(ordered[0].atomic_numbers, dtype=np.int64)
    masses = masses_for(z)
    teacher_coords = np.asarray(
        [s.coordinates for s in ordered], dtype=np.float64)
    ts_coords = np.asarray(ts_state.coordinates, dtype=np.float64)
    dev_rms, dev_max, oracle_arc, teacher_arc = [], [], [], []
    for step_row in record["steps"]:
        t = int(step_row["step"])
        if t >= teacher_coords.shape[0]:
            break  # beyond the teacher path: no overlap, no metric
        x = np.asarray(step_row["x_round10"], dtype=np.float64)
        registered = kabsch_register(x, teacher_coords[t], masses)
        delta = registered - teacher_coords[t]
        dev_rms.append(float(np.sqrt((delta * delta).sum(1).mean())))
        dev_max.append(float(np.linalg.norm(delta, axis=1).max()))
        oracle_reg = kabsch_register(x, ts_coords, masses)
        oracle_arc.append(float(np.sqrt(
            (masses[:, None] * (oracle_reg - ts_coords) ** 2).sum()
            / masses.sum())))
        teacher_arc.append(
            float(labels[ordered[t].state_id].remaining_arc_length))
    post: dict = {
        "declaration": "post_hoc_truth_metrics: ORACLE DIAGNOSTIC ONLY — "
                       "recorded for reporting, never fed to the policy, "
                       "never a reward (G4 §6)",
        "registration_rule": "mass-weighted Kabsch (translation+rotation, "
                             "det+1) — the same frozen rule as the WP2 path "
                             "build (direct_preprocess.kabsch_register)",
        "mass_source": "v1 masses_for table (mass_weighted_arc_v1_diag_masses)",
        "n_overlap_steps": len(dev_rms),
        "per_step_deviation_rms_A": dev_rms,
        "per_step_deviation_max_atom_A": dev_max,
        "cumulative_deviation_mean_A": (
            float(np.mean(dev_rms)) if dev_rms else None),
        "cumulative_deviation_max_A": (
            float(np.max(dev_rms)) if dev_rms else None),
        "deviation_final_A": (dev_rms[-1] if dev_rms else None),
        "deviation_growth_A": (
            dev_rms[-1] - dev_rms[0] if len(dev_rms) >= 2 else None),
        "oracle_arc_to_ts_A": oracle_arc,
        "oracle_arc_start_A": (oracle_arc[0] if oracle_arc else None),
        "oracle_arc_end_A": (oracle_arc[-1] if oracle_arc else None),
        "teacher_label_remaining_arc_A": teacher_arc,
    }
    # start-frame integrity cross-check (post-hoc evidence, not control):
    # the reused start receipt's energy must equal the teacher-bound
    # observation of the SAME geometry.
    if record["steps"]:
        start_state = next(s for s in teacher.states
                           if s.state_id == record["start_state_id"])
        first_obs = record["steps"][0]["observation"]
        if start_state.observation is not None and \
                first_obs.get("energy_hartree") is not None:
            bound_e = float(start_state.observation.energy_hartree)
            receipt_e = float(first_obs["energy_hartree"])
            post["start_state_integrity"] = {
                "start_state_observation_digest_match": bool(
                    first_obs["status"] == "reused_fallback"),
                "teacher_bound_energy_hartree": bound_e,
                "receipt_energy_hartree": receipt_e,
                "delta_hartree": abs(receipt_e - bound_e),
            }
    record["post_hoc_truth_metrics"] = post
    return record


# ---------------------------------------------------------------------------
# Digest binding + artifact emission
# ---------------------------------------------------------------------------


RECORD_DIGEST_KEY: Final[str] = "record_content_digest"


def compute_record_digest(payload: dict) -> str:
    return content_digest_of(
        {k: v for k, v in payload.items() if k != RECORD_DIGEST_KEY})


def seal_record(payload: dict) -> dict:
    sealed = dict(payload)
    sealed[RECORD_DIGEST_KEY] = compute_record_digest(payload)
    return sealed


def verify_record_digest(payload: dict) -> bool:
    if RECORD_DIGEST_KEY not in payload:
        return False
    return compute_record_digest(payload) == payload[RECORD_DIGEST_KEY]


def build_rollout_bundle(
    *,
    record: dict,
    identity: dict,
    scales: dict,
    acp_settings_digest: str,
) -> dict:
    """Versioned ``pesn_dual_ts_bundle_v2`` for ONE rollout record."""
    payload = {
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "run_id": (
            f"pesn_dual_ts_wp4b_freerun_{record['group_id']}_"
            f"branch{str(record['branch_label']).lower()}_v1"),
        "bc_parent_checkpoint_digest": identity["checkpoint_sha256"],
        "seeds": {"torch": int(record["settings"]["seed"]),
                  "numpy": int(record["settings"]["seed"]),
                  "python": int(record["settings"]["seed"])},
        "hyperparams": {
            "freerun_version": FREERUN_VERSION,
            "rollout_record_schema": ROLLOUT_RECORD_SCHEMA,
            **record["settings"],
            "action_rule": "a = a_det(mu) = a_max*tanh(||mu||)mu/||mu|| "
                           "(primary); sampled variant behind action_mode",
            "feature_rule": "consume-time recompute via build_state_features "
                            "with the frozen spec (centroid rule + frozen "
                            "scales); NO reference next-frame correction",
            "guard_rails": "numerical validity; a_max scale bounds; "
                           "catastrophic overlap floor "
                           f"{record['settings']['overlap_min_distance_angstrom']} A",
            "n_steps": record["n_steps"],
            "termination": record["termination"]["status"],
            "termination_reason": record["termination"]["reason"],
        },
        "dataset_version": f"{FEATURES_SCHEMA_VERSION}@{identity['spec_digest']}",
        "norm_version": (
            f"dual_ts_frozen_scales_v0:s_g={scales['s_g']}_Ha/A:"
            f"s_delta_e={scales['s_delta_e']}_Ha"),
        "split_version": "dual_ts_split_lineage_v1:wave=monitor8",
        "action_version": identity["action_version"],
        "environment_version": (
            f"pesn_dual_ts_freerun_env_v0:acp_settings_digest="
            f"{acp_settings_digest}"),
        "reward_version": (
            "pesn_dual_ts_reward_v2:none_oracle_diagnostic_only_v0"),
        "input_eligibility": {
            "purpose": "free-run from held monitor endpoints (G4 §6 / C6)",
            "start_state_id": record["start_state_id"],
            "episode_id": record["episode_id"],
            "observation_mode": OBSERVATION_MODE_PRIMARY,
            "truth_access": "start-frame geometry read pre-loop as initial "
                            "condition (episode start_state_id coordinates, "
                            "Z, maps, charge/multiplicity only); labels/"
                            "actions/future path never read into inputs; "
                            "teacher path metrics computed post-hoc",
            "spec_digest": identity["spec_digest"],
        },
        "output_eligibility": {
            "purpose": "WP4-B free-run evidence; conclusions SEPARATE from "
                       "the teacher-fit monitor8 gate (which FAILED "
                       "honestly — wp4-monitor-report.md)",
            "production_decision": False,
            "oracle_arc_is_diagnostic_only": True,
        },
        "production_decision_open": False,
    }
    validate_record(BUNDLE_SCHEMA_VERSION, payload)
    return payload


def emit_rollout_artifacts(
    *,
    out_dir: str | Path,
    record: dict,
    identity: dict,
    scales: dict,
    acp_settings_digest: str,
) -> tuple[Path, Path, str]:
    """Write (rollout record, bundle) + return paths and the record digest."""
    out_dir = Path(out_dir)
    (out_dir / "rollouts").mkdir(parents=True, exist_ok=True)
    bundle_payload = build_rollout_bundle(
        record=record, identity=identity, scales=scales,
        acp_settings_digest=acp_settings_digest)
    bundle_digest = content_digest_of(bundle_payload)
    record_with_bundle = dict(record)
    record_with_bundle["bundle_content_digest"] = bundle_digest
    sealed = seal_record(record_with_bundle)
    record_path = out_dir / "rollouts" / (
        f"rollout_{record['group_id']}_branch"
        f"{str(record['branch_label']).lower()}.json")
    _atomic_write_json(record_path, sealed)
    bundle_path = out_dir / "rollouts" / (
        f"rollout_{record['group_id']}_branch"
        f"{str(record['branch_label']).lower()}.bundle.json")
    _atomic_write_json(bundle_path, bundle_payload)
    return record_path, bundle_path, str(sealed[RECORD_DIGEST_KEY])


# ---------------------------------------------------------------------------
# Aggregate summary
# ---------------------------------------------------------------------------


def summarize_rollouts(records: Sequence[dict], *,
                       qc_budget: int) -> dict:
    """Aggregate over per-rollout records (no pooling of geometries)."""
    def col(key: str) -> list:
        return [r["post_hoc_truth_metrics"][key] for r in records
                if "post_hoc_truth_metrics" in r
                and r["post_hoc_truth_metrics"].get(key) is not None]

    deviation_final = col("deviation_final_A")
    deviation_cum = col("cumulative_deviation_mean_A")
    min_dists = [
        min(s["geometry_next"]["min_interatomic_distance_A"]
            for s in r["steps"]) for r in records if r["steps"]]
    clip_rates = [
        float(np.mean([s["geometry_next"]["a_max_clip_rate"]
                       for s in r["steps"]])) for r in records if r["steps"]]
    broken = [
        max(s["geometry_next"]["broken_fraction"] for s in r["steps"])
        for r in records if r["steps"]]
    reasons: dict[str, int] = {}
    for r in records:
        key = r["termination"]["status"] + (
            f":{r['termination']['reason']}"
            if r["termination"]["reason"] else "")
        reasons[key] = reasons.get(key, 0) + 1
    billed = sum(r["cost"]["n_billed"] for r in records)
    reused = sum(r["cost"]["n_reused"] for r in records)
    failed = sum(
        1 for r in records if r["termination"]["reason"] == "observation_failed")
    backend = sorted(
        s["observation"]["backend_seconds"] for r in records
        for s in r["steps"]
        if s["observation"]["backend_seconds"] is not None)
    oracle = {r["group_id"]: [
        r["post_hoc_truth_metrics"]["oracle_arc_start_A"],
        r["post_hoc_truth_metrics"]["oracle_arc_end_A"]]
        for r in records
        if "post_hoc_truth_metrics" in r
        and r["post_hoc_truth_metrics"].get("oracle_arc_start_A") is not None}
    return {
        "n_groups": len(records),
        "termination_reasons": reasons,
        "steps_min_median_max": (
            [min(r["n_steps"] for r in records),
             float(np.median([r["n_steps"] for r in records])),
             max(r["n_steps"] for r in records)] if records else []),
        "deviation_final_A_per_group": {
            r["group_id"]: r["post_hoc_truth_metrics"]["deviation_final_A"]
            for r in records
            if "post_hoc_truth_metrics" in r
            and r["post_hoc_truth_metrics"].get("deviation_final_A")
            is not None},
        "deviation_final_median_A": (
            float(np.median(deviation_final)) if deviation_final else None),
        "cumulative_deviation_median_A": (
            float(np.median(deviation_cum)) if deviation_cum else None),
        "min_interatomic_distance_global_min_A": (
            float(np.min(min_dists)) if min_dists else None),
        "a_max_clip_rate_mean_max": (
            [float(np.mean(clip_rates)), float(np.max(clip_rates))]
            if clip_rates else None),
        "broken_bond_fraction_max": (
            float(np.max(broken)) if broken else None),
        "oracle_arc_start_end_A_per_group": oracle,
        "qc": {
            "budget": int(qc_budget),
            "billed_total": int(billed),
            "reused_total": int(reused),
            "billed_within_budget": bool(billed <= int(qc_budget)),
            "observation_failures": int(failed),
            "backend_seconds_median": (
                float(np.median(backend)) if backend else None),
            "backend_seconds_total": round(sum(backend), 4),
        },
    }


__all__ = [
    "A_MAX_REL_TOL",
    "A_MAX_SATURATION_REL",
    "BRANCH_LABEL",
    "FREERUN_RUN_SCHEMA",
    "FREERUN_VERSION",
    "OVERLAP_ABORT_MIN_DISTANCE_ANGSTROM",
    "OBSERVATION_MODE_PRIMARY",
    "ObservationCall",
    "FreerunError",
    "FreerunObserver",
    "QCBudgetExhausted",
    "ROLLOUT_RECORD_SCHEMA",
    "RolloutSettings",
    "RolloutStart",
    "build_rollout_bundle",
    "build_rollout_state_inputs",
    "bond_validity",
    "compute_record_digest",
    "emit_rollout_artifacts",
    "load_frozen_scales",
    "load_rollout_start",
    "load_selected_policy",
    "observation_settings_digest",
    "pairwise_min_distance",
    "post_hoc_truth_metrics",
    "run_rollout",
    "seal_record",
    "summarize_rollouts",
    "validate_receipt_pair",
    "verify_record_digest",
]
