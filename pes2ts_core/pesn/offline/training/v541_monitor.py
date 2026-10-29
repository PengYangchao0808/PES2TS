"""V5.4.1 WP4b — monitor rows for the unified trainer (route E).

Pure, additive monitoring helpers for plan §7.1 (per-100-update loss /
gradient logging) and §7.5 (new-branch learning + dual-task interference):

* :func:`annotated_grad_monitor` — REUSES the legacy
  ``key_atom_v53mp_train_v3._grad_monitor_row`` (read-only import, never
  monkeypatched) and rewrites terms with **no valid supervision** from a
  misleading ``0.0`` into ``null + reason`` (§7.5: 没有有效监督时记
  ``null+reason``, 不记零冲突);
* :func:`branch_grad_row` — gradient norms of the explicit
  ``embedding.* / w_out.* / alpha`` group (``autograd.grad`` with
  ``retain_graph``; never written into the optimizer, never a second
  training step — §7.5 诊断梯度不得写入 optimizer);
* :func:`branch_delta_norm` — parameter-change norm over one monitor
  interval (the ``param_delta_norm`` of the frozen config's
  ``gradient_monitoring.track`` list);
* :class:`PhaseTimer` / :func:`host_peak_rss_bytes` — wall/core/data/eval/
  checkpoint timing plus host peak RSS for §7.3/§8.3 cost accounting.

No dataset, no labels, no dev/confirm access — this module only inspects
tensors that the trainer already produced.
"""

from __future__ import annotations

import collections.abc as _cabc
import sys
import time
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import torch

from pes2ts_core.pesn.offline.training.key_atom_v53mp_train_v3 import (
    _grad_monitor_row,
)

__all__ = [
    "EMBEDDING_DIAG_ATTRS_V541",
    "PhaseTimer",
    "annotated_grad_monitor",
    "branch_delta_norm",
    "branch_grad_row",
    "branch_named_params",
    "branch_snapshot",
    "embedding_diagnostics",
    "group_grad_row",
    "host_peak_rss_bytes",
    "is_branch_param",
    "rms_ratio",
    "set_diagnostics_enabled_v541",
    "supervision_nulls",
]

#: Fixed denominator epsilon for the ``RMS(alpha*W*f) / RMS(h_legacy)``
#: ratio (§7.5: 分母 eps 固定 — never tuned, never per-model).
RMS_RATIO_EPS_V541: float = 1.0e-8

#: Explicit branch parameter-group prefixes (§7.1: 不再用「非 backbone
#: 全是 head」推断 — registration is by NAME, not by negation).
BRANCH_PREFIXES_V541: Tuple[str, ...] = ("embedding.", "w_out.")
BRANCH_EXACT_V541: Tuple[str, ...] = ("alpha",)

#: row-count key -> (loss term, reason) for the no-supervision annotation.
_NULL_ROW_KEYS: Dict[str, Tuple[str, str]] = {
    "n_rank_pairs": ("rank", "no_valid_ordering_pairs"),
    "n_q_rows": ("q", "no_valid_q_rows"),
    "n_endpoint_rows": ("endpoint", "no_valid_endpoint_rows"),
}


def is_branch_param(name: str) -> bool:
    """True for ``embedding.*`` / ``w_out.*`` / ``alpha`` (by NAME)."""
    return (str(name).startswith(BRANCH_PREFIXES_V541)
            or str(name) in BRANCH_EXACT_V541)


def branch_named_params(model: torch.nn.Module) -> list:
    """``[(name, param)]`` of the branch group, in ``named_parameters()`` order."""
    return [(n, p) for n, p in model.named_parameters() if is_branch_param(n)]


def branch_group(name: str) -> str:
    """``embedding`` | ``w_out`` | ``alpha`` for one branch parameter name."""
    if str(name) in BRANCH_EXACT_V541:
        return "alpha"
    if str(name).startswith("w_out."):
        return "w_out"
    return "embedding"


# ---------------------------------------------------------------------------
# diagnostics wiring (§7.5): model flag -> embedding flag, then merge
# ---------------------------------------------------------------------------

#: Attributes a WP3 embedding may expose its last-forward diagnostics on.
#: ``last_diagnostics`` is preferred (explicit "of the last forward");
#: ``diagnostics`` is the fallback name.  Both are read ONLY if present —
#: this module never touches the WP3 candidate files.
EMBEDDING_DIAG_ATTRS_V541: Tuple[str, ...] = ("last_diagnostics",
                                              "diagnostics")


def set_diagnostics_enabled_v541(model: torch.nn.Module,
                                 enabled: bool) -> None:
    """Propagate the monitor toggle to the model AND its embedding.

    ``V541Model.diagnostics_enabled`` only gates the model-level scalars
    (alpha / RMS / edge counts); a WP3 candidate that carries its own
    ``diagnostics_enabled`` must receive the SAME flag, otherwise the
    branch keeps paying (or never pays) the provider-side count syncs and
    the per-100-update log would silently disagree with the run.  Every
    attribute is optional — E0 and pre-WP3 candidates simply have none.
    """
    flag = bool(enabled)
    if hasattr(model, "diagnostics_enabled"):
        model.diagnostics_enabled = flag
    emb = getattr(model, "embedding", None)
    if emb is not None and hasattr(emb, "diagnostics_enabled"):
        emb.diagnostics_enabled = flag


def embedding_diagnostics(model: torch.nn.Module
                          ) -> Tuple[Dict[str, Any], str]:
    """``(diagnostics mapping, status)`` of the branch's last forward.

    ``status`` is one of:

    * ``absent_no_branch`` — E0, no ``model.embedding`` at all;
    * ``attr_absent`` / ``attr_not_a_mapping`` — the candidate exposes
      neither documented attribute (or a non-mapping value);
    * ``empty`` — the attribute exists but holds ``{}`` (exactly what the
      WP3 providers return under ``diagnostics=False``, plan §6.4);
    * ``ok`` — a non-empty mapping was captured.

    Never raises and never fabricates a value: an absent quantity stays
    ``None`` and the caller records a reason.
    """
    emb = getattr(model, "embedding", None)
    if emb is None:
        return {}, "absent_no_branch"
    for attr in EMBEDDING_DIAG_ATTRS_V541:
        try:
            present = hasattr(emb, attr)
            value = getattr(emb, attr) if present else None
        except Exception:                   # pragma: no cover - defensive
            present, value = False, None
        if not present:
            continue
        if not isinstance(value, _cabc.Mapping):
            return {}, "attr_not_a_mapping"
        return dict(value), ("empty" if not value else "ok")
    return {}, "attr_absent"


# ---------------------------------------------------------------------------
# no-supervision annotation (§7.5: null + reason, never a fake 0.0)
# ---------------------------------------------------------------------------


def supervision_nulls(loss_out: Mapping[str, Any], *,
                      move_rows: int) -> Dict[str, str]:
    """Terms with ZERO valid supervision rows in this batch → reason map.

    ``move_rows`` is the caller's count of MOVE-eligible rows (the
    ``frame_mask`` true count); the process terms read their own counters
    straight from the frozen loss output.  Only ACTIVE terms (λ != 0) can
    be null — an inactive term has no supervision obligation.
    """
    nulls: Dict[str, str] = {}
    if int(move_rows) <= 0:
        nulls["move"] = "no_valid_move_rows"
    for key, (term, reason) in _NULL_ROW_KEYS.items():
        if float(loss_out.get("lambda_" + term, 1.0) or 0.0) == 0.0:
            continue
        try:
            rows = int(loss_out.get(key, 0))
        except (TypeError, ValueError):
            rows = 0
        if rows <= 0:
            nulls[term] = reason
    return nulls


def annotated_grad_monitor(loss_out: Mapping[str, Any],
                           encoder_params: Sequence[torch.Parameter],
                           config: Any, *,
                           move_rows: int) -> Dict[str, Any]:
    """Legacy grad monitor + ``null + reason`` for unsupervised terms.

    The legacy :func:`_grad_monitor_row` is imported and CALLED (route E:
    reuse, no monkeypatch).  Terms without supervision are replaced by a
    detached zero tensor BEFORE the call, so they can neither contribute a
    gradient vector, a pairwise cosine, nor the ``encoder_share`` ratio;
    afterwards their norm is rewritten to ``None`` with an explicit
    ``null_reasons`` entry.
    """
    nulls = supervision_nulls(loss_out, move_rows=move_rows)
    filtered: Dict[str, Any] = dict(loss_out)
    for term in nulls:
        # the frozen loss exposes the terms as ``l_<term>``; replace THAT
        # key so the legacy monitor sees a detached constant instead of a
        # graph-connected zero (which would still count as "supervised").
        filtered["l_" + term] = torch.zeros((), dtype=torch.float32)
    row = _grad_monitor_row(filtered, encoder_params, config)
    row["null_reasons"] = dict(nulls)
    for term, reason in nulls.items():
        row.setdefault("grad_norms", {})[term] = None
        # drop cosines that referenced the nulled term
        for key in [k for k in row.get("cosines", {})
                    if k.startswith(term + "_") or k.endswith("_" + term)]:
            row["cosines"][key] = None
        if term == "move":
            row["encoder_share"] = None
            row["encoder_share_reason"] = reason
    return row


# ---------------------------------------------------------------------------
# branch gradients + parameter change (§7.5)
# ---------------------------------------------------------------------------


def group_grad_row(model: torch.nn.Module, loss: Any) -> Dict[str, Any]:
    """Total-loss gradient norm of EVERY explicit parameter group (§7.1).

    ``backbone`` / ``process_head`` / ``embedding`` are measured on the
    SAME loss and the same forward — the per-100-update evidence that each
    group is (or is deliberately not) in the learning path.  A frozen or
    absent group is reported as ``null + reason``, never as a fake
    ``0.0``.  Cross-group cosines are NOT computed here (the group vectors
    live in different parameter spaces); the MOVE↔PROCESS cosine the plan
    asks for is the legacy ``_grad_monitor_row`` cosine on the ONE shared
    parameter set (the backbone).  Diagnostic only: ``retain_graph=True``
    and no optimizer write.
    """
    if not (torch.is_tensor(loss) and bool(loss.requires_grad)):
        return {"present": False, "reason": "loss_has_no_grad_path",
                "norms": {}, "reasons": {}}
    buckets: Dict[str, list] = {"backbone": [], "process_head": [],
                                "embedding": []}
    for name, param in model.named_parameters():
        if not bool(param.requires_grad):
            continue
        if name.startswith("backbone."):
            buckets["backbone"].append(param)
        elif name.startswith("process_head."):
            buckets["process_head"].append(param)
        elif is_branch_param(name):
            buckets["embedding"].append(param)
    norms: Dict[str, Optional[float]] = {}
    reasons: Dict[str, str] = {}
    for name, params in buckets.items():
        if not params:
            norms[name] = None
            reasons[name] = "group_empty_or_frozen"
            continue
        grads = torch.autograd.grad(
            loss, params, retain_graph=True, allow_unused=True)
        if all(g is None for g in grads):
            norms[name] = None
            reasons[name] = "no_grad_path_from_loss"
            continue
        vec = torch.cat([
            torch.zeros_like(p).reshape(-1) if g is None else g.reshape(-1)
            for p, g in zip(params, grads)])
        norms[name] = float(vec.norm())
    return {"present": True, "norms": norms, "reasons": reasons,
            "cosine_note": ("MOVE<->PROCESS cosine lives in the legacy "
                            "grad_monitor row (shared backbone set)")}


def branch_grad_row(model: torch.nn.Module, loss: Any) -> Dict[str, Any]:
    """Gradient norms of the branch group for ONE loss (diagnostic only).

    Uses ``torch.autograd.grad(..., retain_graph=True, allow_unused=True)``
    — the caller still runs the single real ``loss.backward()`` afterwards,
    so this NEVER becomes a second optimizer step.  ``all_zero`` at update
    1 is the MATH expected cold start (W/f are multiplied by ``alpha==0``);
    it is NOT evidence that the branch left the learning path (§7.5).
    """
    named = branch_named_params(model)
    if not named:
        return {"present": False, "reason": "e0_no_branch"}
    if not all(bool(p.requires_grad) for _n, p in named):
        return {"present": True, "reason": "branch_params_frozen",
                "frozen": True}
    if not (torch.is_tensor(loss) and bool(loss.requires_grad)):
        return {"present": True, "reason": "loss_has_no_grad_path"}
    params = [p for _n, p in named]
    grads = torch.autograd.grad(
        loss, params, retain_graph=True, allow_unused=True)
    grouped: Dict[str, list] = {}
    total_parts: list = []
    alpha_grad: Optional[float] = None
    finite = True
    for (name, _p), grad in zip(named, grads):
        vec = torch.zeros_like(_p) if grad is None else grad
        if not bool(torch.isfinite(vec).all()):
            finite = False
        flat = vec.reshape(-1)
        grouped.setdefault(branch_group(name), []).append(flat)
        total_parts.append(flat)
        if name in BRANCH_EXACT_V541:
            alpha_grad = float(flat[0]) if flat.numel() else 0.0
    norms = {g: float(torch.cat(parts).norm())
             for g, parts in sorted(grouped.items())}
    total = float(torch.cat(total_parts).norm()) if total_parts else 0.0
    return {
        "present": True,
        "grad_norms": norms,
        "total_norm": total,
        "alpha_grad": alpha_grad,
        "finite": bool(finite),
        "all_zero": bool(total == 0.0),
        "note": ("cold start (alpha==0) makes W/f task gradients exactly "
                 "zero — expected on step 1; AdamW weight decay is NOT "
                 "learning evidence (plan §7.5)"),
    }


def branch_snapshot(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    """Detached CPU clone of the branch weights (post-step snapshot)."""
    return {n: p.detach().cpu().clone() for n, p in branch_named_params(model)}


def branch_delta_norm(model: torch.nn.Module,
                      previous: Optional[Mapping[str, torch.Tensor]]
                      ) -> Optional[float]:
    """``||theta_now - theta_prev||`` over the branch group since last monitor.

    ``None`` on the first monitor (no previous snapshot) — reported as a
    reason by the caller, never as a fake ``0.0``.
    """
    if previous is None:
        return None
    parts = []
    for name, param in branch_named_params(model):
        old = previous.get(name)
        if old is None:
            continue
        parts.append((param.detach().cpu() - old).reshape(-1))
    if not parts:
        return None
    return float(torch.cat(parts).norm())


def rms_ratio(numerator: Optional[float],
              denominator: Optional[float]) -> Optional[float]:
    """``RMS(alpha*W*f) / RMS(h_legacy)`` with the FIXED epsilon of §7.5."""
    if numerator is None or denominator is None:
        return None
    return float(numerator) / (float(denominator) + RMS_RATIO_EPS_V541)


# ---------------------------------------------------------------------------
# timing / resource accounting (§7.3, §8.3)
# ---------------------------------------------------------------------------


class PhaseTimer:
    """Per-phase wall / data / core / eval / checkpoint accumulators.

    Phases are ``startup`` / ``w0`` / ``w1`` (formal) or ``profile_w0`` /
    ``profile_w1`` (profile-only windows) — §8.3 forbids extrapolating a
    200-update W0 window onto W1, so the two are ALWAYS timed apart.
    """

    _KEYS = ("wall_s", "data_s", "core_s", "eval_s", "checkpoint_s",
             "updates")

    def __init__(self) -> None:
        self._phases: Dict[str, Dict[str, float]] = {}

    def bucket(self, phase: str) -> Dict[str, float]:
        return self._phases.setdefault(
            str(phase), {k: 0.0 for k in self._KEYS})

    def add(self, phase: str, key: str, seconds: float) -> None:
        if key not in self._KEYS:
            raise KeyError(f"unknown timer key {key!r}")
        self.bucket(phase)[key] += float(seconds)

    def count(self, phase: str) -> None:
        self.bucket(phase)["updates"] += 1.0

    def total(self, key: str) -> float:
        return float(sum(b.get(key, 0.0) for b in self._phases.values()))

    def as_dict(self) -> Dict[str, Dict[str, float]]:
        return {phase: {k: round(float(v), 6) for k, v in sorted(vals.items())}
                for phase, vals in sorted(self._phases.items())}


def host_peak_rss_bytes() -> int:
    """Peak resident set size of THIS process in bytes (Linux: ru_maxrss KiB)."""
    try:
        import resource

        usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    except Exception:                        # pragma: no cover - non-POSIX
        return 0
    usage = int(usage)
    if usage <= 0:
        return 0
    if sys.platform == "darwin":             # ru_maxrss already bytes
        return usage
    return usage * 1024


def now() -> float:
    """Monotonic-ish wall clock used by the trainer's phase timers."""
    return time.time()
