"""Feature-gate property-test harness (T1.1-T1.4), model-agnostic.

Lane B dry-run scaffold (branch ``ops/par-dryrun-b``).  Every check takes a
``model`` and a feature ``batch`` (a plain dict of tensors / metadata) and
returns a **report dict**; nothing here mutates the caller's batch.

Model contract expected by the harness (kept deliberately minimal so the real
StarEGNN v3 can be adapted later with a thin wrapper):

* input  — ``batch`` dict; coordinate-bearing tensors are float ``[*, 3]``;
  the harness only requires a ``"pos"`` tensor ``[N, 3]`` for geometric
  transforms.  Optional conventional keys: ``"z"`` ``[N]``, ``"edge_index"``
  ``[2, E]``, ``"key_index"`` ``[K]`` (rows the model outputs displacements
  for; defaults to all atoms), ``"graph_index"`` ``[N]`` (graph id per atom),
  ``"endpoint_names"`` (tuple of endpoint display names, used only by
  :func:`check_relabel_invariance`), ``"signed_intent"`` (scalar tensor,
  used only by :func:`check_reverse_sensitivity`).
* output — ``model(batch)`` returns either a tensor ``u [K, 3]`` or a dict
  with key ``"u"`` (equivariant vectors ``[K, 3]``) plus optional invariant
  scalar tensors (any other float tensor, e.g. ``"s"``).

Normative references
--------------------
* Typed-invalid on NaN / out-of-contract input:
  ``docs/contracts/PES2TS_GeometryPolicyContract_v1.md`` S4.3 (L121, L125,
  L170, L183): "online inference on out-of-contract input (missing topology,
  illegal K, unsolvable gauge, NaN) must return a typed invalid, never a
  fabricated action."
* Relabel invariance + tolerance preset "30h_v1.1":
  30h plan S3.3 T1.1 / ReactionIntent Contract v1 L321-343 — the referenced
  file ``docs/contracts/PES2TS_ReactionIntent_Contract_v1.md`` is **ABSENT**
  at worktree HEAD 4918442 (see OPEN DECISION note in
  :data:`PRESET_SOURCES`).
* Tolerance preset "r3_v3" (abs 1e-4): source document also not found in the
  worktree; recorded verbatim from the task spec.

OPEN DECISION (never silently defaulted): the two docs disagree on the
equivariance tolerance (rel 1e-5 + abs 1e-6 vs abs 1e-4).  Callers must pass
``preset=`` explicitly; a ``None`` preset emits a :class:`UserWarning` that
lists both options and falls back to ``"30h_v1.1"``.  Every report records
which preset was used and whether it was explicit.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping, MutableMapping, Sequence

import numpy as np
import torch

__all__ = [
    "PRESETS",
    "PRESET_SOURCES",
    "DEFAULT_PRESET",
    "RELABEL_ABS_TOL",
    "FORBIDDEN_FEATURE_PATTERNS",
    "IllegalFeatureError",
    "TypedInvalidError",
    "resolve_preset",
    "check_input_legality",
    "check_translation_invariance",
    "check_rotation_equivariance",
    "check_permutation_equivariance",
    "check_relabel_invariance",
    "check_reverse_sensitivity",
    "check_nan_typed_invalid",
    "run_feature_gates",
]


# ---------------------------------------------------------------------------
# Tolerance presets (OPEN DECISION — see module docstring)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Preset:
    name: str
    rel: float
    abs: float  # noqa: A003 - mirrors numpy/pytorch naming
    source: str


PRESET_SOURCES: dict[str, str] = {
    "30h_v1.1": (
        "30h plan S3.3 T1.1 / PES2TS_ReactionIntent_Contract_v1.md L321-343 "
        "(rel 1e-5 + abs 1e-6) — NOTE: docs/contracts/"
        "PES2TS_ReactionIntent_Contract_v1.md is ABSENT at worktree HEAD "
        "4918442; spec taken from task text"
    ),
    "r3_v3": (
        "r3_v3 equivariance tolerance (abs 1e-4) — source document not found "
        "in worktree; spec taken from task text"
    ),
}

PRESETS: dict[str, Preset] = {
    "30h_v1.1": Preset("30h_v1.1", rel=1e-5, abs=1e-6,
                       source=PRESET_SOURCES["30h_v1.1"]),
    "r3_v3": Preset("r3_v3", rel=0.0, abs=1e-4,
                    source=PRESET_SOURCES["r3_v3"]),
}

DEFAULT_PRESET = "30h_v1.1"

#: Hard bit-identity / abs threshold for the endpoint display-name swap
#: (30h plan S3.3 T1.1).  Independent of the preset.
RELABEL_ABS_TOL = 1e-6


class TypedInvalidError(RuntimeError):
    """Typed invalid raised by models for out-of-contract input (NaN etc.).

    Mirrors GeometryPolicyContract S4.3: never fabricate an action.
    """

    def __init__(self, code: str, detail: str = "") -> None:
        super().__init__(f"{code}: {detail}" if detail else code)
        self.code = code
        self.detail = detail


class IllegalFeatureError(ValueError):
    """Raised when the feature dict carries forbidden (label-leaking) keys."""

    def __init__(self, violations: Sequence[str]) -> None:
        self.violations = tuple(violations)
        super().__init__(
            "forbidden feature keys (label leakage): "
            + ", ".join(self.violations)
        )


def resolve_preset(preset: str | None) -> tuple[Preset, bool]:
    """Resolve a preset name; ``None`` warns and falls back (never silent)."""
    if preset is None:
        listing = "\n".join(
            f"  - {name}: rel={p.rel} abs={p.abs}  <- {p.source}"
            for name, p in PRESETS.items()
        )
        warnings.warn(
            "feature-gate tolerance preset not specified; the two source "
            "documents disagree (OPEN DECISION). Falling back to "
            f"'{DEFAULT_PRESET}'. Candidates:\n{listing}",
            UserWarning,
            stacklevel=2,
        )
        return PRESETS[DEFAULT_PRESET], False
    if preset not in PRESETS:
        raise KeyError(
            f"unknown tolerance preset {preset!r}; "
            f"known: {sorted(PRESETS)}"
        )
    return PRESETS[preset], True


# ---------------------------------------------------------------------------
# Forbidden-feature legality gate (ReactionIntent Contract v1 L321-343)
# ---------------------------------------------------------------------------

#: Substring patterns (lower-cased) that must never appear in a feature dict.
#: Categories: future frames; TS coordinates; true progress / remaining arc;
#: branch id; split / family; energy / gradient; near_ts / capped flags.
FORBIDDEN_FEATURE_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("future_frames", (
        "future", "next_frame", "pos_next", "frame_next", "fut_",
        "irc_next", "traj_future",
    )),
    ("ts_coords", (
        "pos_ts", "ts_coord", "ts_xyz", "x_ts", "true_ts", "ts_pos",
        "ts_frame",
    )),
    ("true_progress_arc", (
        "true_progress", "progress", "remaining_arc", "arc_remaining",
        "s_remaining", "frac_remaining", "true_arc", "arc_length",
        "true_s",
    )),
    ("branch_id", ("branch_id", "branch")),
    ("split_family", (
        "split", "family", "fold", "cluster_id", "group_ids",
        "reaction_ids", "scaffold",
    )),
    ("energy_gradient", (
        "energy", "grad", "force", "dft_", "e_react", "e_prod", "e_t",
        "delta_e",
    )),
    ("near_ts_capped_flags", (
        "near_ts", "capped", "cap_flag", "terminal",
    )),
)


def _violating_keys(batch: Mapping[str, Any]) -> list[tuple[str, str]]:
    hits: list[tuple[str, str]] = []
    for key in batch:
        low = str(key).lower()
        for category, patterns in FORBIDDEN_FEATURE_PATTERNS:
            if any(pat in low for pat in patterns):
                hits.append((str(key), category))
                break
    return hits


def check_input_legality(
    batch: Mapping[str, Any],
    *,
    raise_on_violation: bool = False,
    preset: str | None = None,
) -> dict[str, Any]:
    """Assert the feature dict carries no forbidden (label-leaking) keys."""
    preset_obj, explicit = resolve_preset(preset)
    hits = _violating_keys(batch)
    report: dict[str, Any] = {
        "check": "input_legality",
        "passed": not hits,
        "violations": [k for k, _ in hits],
        "violation_categories": sorted({c for _, c in hits}),
        "preset": preset_obj.name,
        "preset_explicit": explicit,
        "tolerance": {"rel": preset_obj.rel, "abs": preset_obj.abs},
    }
    if hits and raise_on_violation:
        raise IllegalFeatureError([k for k, _ in hits])
    return report


# ---------------------------------------------------------------------------
# Batch helpers (non-mutating)
# ---------------------------------------------------------------------------

def _copy_batch(batch: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in batch.items():
        out[k] = v.clone() if isinstance(v, torch.Tensor) else v
    return out


def _float3_tensors(batch: Mapping[str, Any]) -> list[str]:
    return [
        k for k, v in batch.items()
        if isinstance(v, torch.Tensor) and v.is_floating_point()
        and v.ndim >= 2 and v.shape[-1] == 3
    ]


#: Keys holding *absolute* coordinates (translated but not differenced).
ABS_POS_KEYS: tuple[str, ...] = ("pos", "pos0", "xyz", "coords")


def _as_output_dict(out: Any) -> dict[str, torch.Tensor]:
    if isinstance(out, torch.Tensor):
        return {"u": out}
    if isinstance(out, Mapping):
        return {k: v for k, v in out.items() if isinstance(v, torch.Tensor)}
    raise TypeError(
        f"model output must be a tensor or a mapping of tensors, got {type(out)}"
    )


def _call_model(model: Callable[..., Any], batch: Mapping[str, Any]):
    was_training = getattr(model, "training", None)
    if hasattr(model, "eval"):
        model.eval()
    try:
        with torch.no_grad():
            return model(batch)
    finally:
        if was_training and hasattr(model, "train"):
            model.train()


def _u_vecs(out: Mapping[str, torch.Tensor]) -> torch.Tensor:
    if "u" not in out:
        raise KeyError("model output dict must contain equivariant 'u' [K,3]")
    return out["u"]


def _stats(errors: torch.Tensor | np.ndarray) -> dict[str, float]:
    arr = np.asarray(
        errors.detach().cpu().numpy() if isinstance(errors, torch.Tensor)
        else errors, dtype=np.float64
    ).ravel()
    if arr.size == 0:
        return {"median": 0.0, "p99": 0.0, "max": 0.0}
    return {
        "median": float(np.median(arr)),
        "p99": float(np.percentile(arr, 99)),
        "max": float(np.max(arr)),
    }


def _threshold(preset: Preset, scale: float) -> float:
    return preset.abs + preset.rel * float(scale)


def _scale(u: torch.Tensor) -> float:
    return float(u.detach().abs().max().item()) if u.numel() else 0.0


def _report(
    check: str,
    errors: torch.Tensor,
    passed: bool,
    preset: Preset,
    explicit: bool,
    scale: float,
    **extra: Any,
) -> dict[str, Any]:
    rep: dict[str, Any] = {
        "check": check,
        "passed": bool(passed),
        "preset": preset.name,
        "preset_explicit": bool(explicit),
        "tolerance": {"rel": preset.rel, "abs": preset.abs},
        "scale": scale,
        "threshold": _threshold(preset, scale),
        **_stats(errors),
        **extra,
    }
    return rep


# ---------------------------------------------------------------------------
# T1.1 checks
# ---------------------------------------------------------------------------

def check_translation_invariance(
    model: Callable[..., Any],
    batch: Mapping[str, Any],
    *,
    preset: str | None = None,
    shift: torch.Tensor | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    """max|u(x+t) - u(x)|; absolute-coordinate keys are translated."""
    preset_obj, explicit = resolve_preset(preset)
    base = _copy_batch(batch)
    moved = _copy_batch(batch)
    if shift is None:
        gen = torch.Generator().manual_seed(seed)
        shift = (torch.rand(3, generator=gen) - 0.5) * 4.0  # ~[-2,2] A
    for key in ABS_POS_KEYS:
        if key in moved and isinstance(moved[key], torch.Tensor):
            moved[key] = moved[key] + shift.to(moved[key].dtype)

    out_a = _as_output_dict(_call_model(model, base))
    out_b = _as_output_dict(_call_model(model, moved))
    u_a, u_b = _u_vecs(out_a), _u_vecs(out_b)
    errors = (u_a - u_b).detach().abs()
    scale = max(_scale(u_a), _scale(u_b))
    thr = _threshold(preset_obj, scale)
    return _report(
        "translation_invariance", errors, bool(errors.max().item() <= thr),
        preset_obj, explicit, scale,
        max_abs_delta=float(errors.max().item()),
        shift=shift.tolist(),
    )


def _random_rotation(seed: int) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    q, r = torch.linalg.qr(torch.randn(3, 3, generator=gen, dtype=torch.float64))
    q = q @ torch.diag(torch.sign(torch.diag(r)))
    if torch.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q.float()


def check_rotation_equivariance(
    model: Callable[..., Any],
    batch: Mapping[str, Any],
    *,
    preset: str | None = None,
    rotation: torch.Tensor | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    """Per-atom |R u(x) - u(R x)| for a random proper rotation R."""
    preset_obj, explicit = resolve_preset(preset)
    R = rotation if rotation is not None else _random_rotation(seed)
    base = _copy_batch(batch)
    rot = _copy_batch(batch)
    for key in _float3_tensors(rot):
        rot[key] = rot[key] @ R.T.to(rot[key].dtype)

    out_a = _as_output_dict(_call_model(model, base))
    out_b = _as_output_dict(_call_model(model, rot))
    u_a, u_b = _u_vecs(out_a), _u_vecs(out_b)
    Rl = R.to(u_a.dtype)
    per_atom = ((u_a @ Rl.T) - u_b).detach().norm(dim=-1)  # [K]
    scale = max(_scale(u_a), _scale(u_b))
    thr = _threshold(preset_obj, scale)
    return _report(
        "rotation_equivariance", per_atom,
        bool(per_atom.numel() and per_atom.max().item() <= thr),
        preset_obj, explicit, scale,
        per_atom_errors=per_atom.tolist(),
        rotation=R.tolist(),
    )


def check_permutation_equivariance(
    model: Callable[..., Any],
    batch: Mapping[str, Any],
    *,
    preset: str | None = None,
    permutation: torch.Tensor | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    """|u(P x) un-permuted - u(x)| for a within-graph atom permutation P."""
    preset_obj, explicit = resolve_preset(preset)
    base = _copy_batch(batch)
    pos = base["pos"]
    n = int(pos.shape[0])

    if permutation is None:
        gen = torch.Generator().manual_seed(seed)
        graph = base.get("graph_index")
        perm = torch.empty(n, dtype=torch.long)
        if graph is None:
            perm = torch.randperm(n, generator=gen)
        else:
            graph = graph.long()
            for gid in torch.unique(graph):
                idx = torch.nonzero(graph == gid, as_tuple=False).ravel()
                perm[idx] = idx[torch.randperm(idx.numel(), generator=gen)]
    else:
        perm = permutation.long()
    inv = torch.empty_like(perm)
    inv[perm] = torch.arange(n, dtype=torch.long)

    permuted = _copy_batch(batch)
    for key, v in list(permuted.items()):
        if key == "key_index" or key == "edge_index":
            continue  # index tensors: remapped below, not gathered
        if isinstance(v, torch.Tensor) and v.ndim >= 1 and v.shape[0] == n:
            permuted[key] = v[perm]
    if "edge_index" in permuted and isinstance(permuted["edge_index"], torch.Tensor):
        permuted["edge_index"] = inv[permuted["edge_index"].long()]
    if "key_index" in permuted and isinstance(permuted["key_index"], torch.Tensor):
        # values re-point at the atoms' post-permutation rows; row ORDER of
        # the key list (hence of model outputs) is preserved
        permuted["key_index"] = inv[permuted["key_index"].long()]

    out_a = _as_output_dict(_call_model(model, base))
    out_b = _as_output_dict(_call_model(model, permuted))
    u_a = _u_vecs(out_a)
    u_b = _u_vecs(out_b)
    # Row alignment: when `key_index` is present the harness convention keeps
    # the key-list ORDER fixed (values re-pointed), so outputs align directly.
    # Without `key_index` outputs follow storage order and must be un-permuted
    # (new[i] = old[perm[i]]  =>  u_back[perm] = u_b).
    if "key_index" in batch:
        errors = (u_a - u_b).detach().abs()
    else:
        u_back = torch.empty_like(u_a)
        u_back[perm] = u_b
        errors = (u_a - u_back).detach().abs()
    scale = max(_scale(u_a), _scale(u_b))
    thr = _threshold(preset_obj, scale)
    return _report(
        "permutation_equivariance", errors,
        bool(errors.numel() and errors.max().item() <= thr),
        preset_obj, explicit, scale,
        max_abs_delta=float(errors.max().item()),
        permutation=perm.tolist(),
    )


def check_relabel_invariance(
    model: Callable[..., Any],
    batch: Mapping[str, Any],
    *,
    preset: str | None = None,
) -> dict[str, Any]:
    """Endpoint display-name swap: outputs must be bit-identical.

    30h plan S3.3 T1.1: swapping the two endpoint display names must leave
    the outputs **bit-identical**; hard threshold ``max|du| <= 1e-6``.
    """
    preset_obj, explicit = resolve_preset(preset)
    names = batch.get("endpoint_names")
    if names is None:
        raise KeyError(
            "check_relabel_invariance requires batch['endpoint_names'] "
            "(tuple of the two endpoint display names)"
        )
    base = _copy_batch(batch)
    swapped = _copy_batch(batch)
    swapped["endpoint_names"] = tuple(reversed(tuple(names)))

    out_a = _as_output_dict(_call_model(model, base))
    out_b = _as_output_dict(_call_model(model, swapped))
    u_a, u_b = _u_vecs(out_a), _u_vecs(out_b)
    errors = (u_a - u_b).detach().abs()
    max_delta = float(errors.max().item()) if errors.numel() else 0.0
    bit_identical = all(
        torch.equal(out_a[k], out_b[k]) for k in out_a
        if k in out_b and out_a[k].shape == out_b[k].shape
    ) and set(out_a) == set(out_b)
    return {
        "check": "relabel_invariance",
        "passed": bool(bit_identical and max_delta <= RELABEL_ABS_TOL),
        "preset": preset_obj.name,
        "preset_explicit": explicit,
        "tolerance": {"rel": preset_obj.rel, "abs": preset_obj.abs},
        "relabel_abs_tol": RELABEL_ABS_TOL,
        "bit_identical": bool(bit_identical),
        "max_abs_delta": max_delta,
        "scale": max(_scale(u_a), _scale(u_b)),
        **_stats(errors),
    }


def check_reverse_sensitivity(
    model: Callable[..., Any],
    batch: Mapping[str, Any],
    *,
    preset: str | None = None,
) -> dict[str, Any]:
    """Signed-intent flip (T1.2): outputs must CHANGE, no exact-negative req."""
    preset_obj, explicit = resolve_preset(preset)
    if "signed_intent" not in batch:
        raise KeyError(
            "check_reverse_sensitivity requires batch['signed_intent'] "
            "(scalar tensor; the harness flips its sign)"
        )
    base = _copy_batch(batch)
    flipped = _copy_batch(batch)
    flipped["signed_intent"] = -base["signed_intent"]

    out_a = _as_output_dict(_call_model(model, base))
    out_b = _as_output_dict(_call_model(model, flipped))
    u_a, u_b = _u_vecs(out_a), _u_vecs(out_b)
    errors = (u_a - u_b).detach().abs()
    scale = max(_scale(u_a), _scale(u_b))
    # "changed" must exceed numerical noise under the active preset;
    # exact negative (u_flip == -u) is NOT required (T1.2).
    min_change = max(preset_obj.abs, 1e-12)
    changed = bool(errors.numel() and errors.max().item() > min_change)
    exact_negative = bool(
        errors.numel()
        and torch.allclose(u_b, -u_a, atol=preset_obj.abs, rtol=0.0)
    )
    return _report(
        "reverse_sensitivity", errors, changed,
        preset_obj, explicit, scale,
        outputs_changed=changed,
        exact_negative_observed=exact_negative,
        exact_negative_required=False,
    )


# ---------------------------------------------------------------------------
# T1.3/T1.4: NaN -> typed invalid (GeometryPolicyContract S4.3)
# ---------------------------------------------------------------------------

def check_nan_typed_invalid(
    model: Callable[..., Any],
    batch: Mapping[str, Any],
    *,
    preset: str | None = None,
) -> dict[str, Any]:
    """NaN coordinates must yield a typed invalid, never a fabricated action."""
    preset_obj, explicit = resolve_preset(preset)
    nan_batch = _copy_batch(batch)
    key = "pos" if "pos" in nan_batch else _float3_tensors(nan_batch)[0]
    nan_batch[key] = torch.full_like(nan_batch[key], float("nan"))

    outcome = "fabricated_action"
    code: str | None = None
    try:
        out = _call_model(model, nan_batch)
    except Exception as exc:  # noqa: BLE001 - classified below
        typed = isinstance(exc, TypedInvalidError) or (
            "invalid" in type(exc).__name__.lower()
            and getattr(exc, "code", None)
        )
        outcome = "typed_invalid_raised" if typed else "untyped_exception"
        code = getattr(exc, "code", None) or type(exc).__name__
    else:
        if isinstance(out, Mapping) and (
            out.get("invalid") is True or out.get("status") == "invalid"
        ):
            outcome = "typed_invalid_returned"
            code = str(out.get("reason") or out.get("code") or "INVALID")
        else:
            tensors = _as_output_dict(out)
            nonfinite = any(
                not torch.isfinite(v).all().item()
                for v in tensors.values() if v.is_floating_point()
            )
            outcome = "non_finite_output" if nonfinite else "fabricated_action"

    return {
        "check": "nan_typed_invalid",
        "passed": outcome.startswith("typed_invalid"),
        "outcome": outcome,
        "code": code,
        "preset": preset_obj.name,
        "preset_explicit": explicit,
        "tolerance": {"rel": preset_obj.rel, "abs": preset_obj.abs},
        "contract": "GeometryPolicyContract_v1.md S4.3 (L121/L125/L170/L183)",
    }


# ---------------------------------------------------------------------------
# Aggregate runner
# ---------------------------------------------------------------------------

def run_feature_gates(
    model: Callable[..., Any],
    batch: Mapping[str, Any],
    *,
    preset: str | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    """Run every gate that the batch supports; aggregate pass/fail."""
    resolved, explicit = resolve_preset(preset)
    reports: dict[str, Any] = {
        "input_legality": check_input_legality(batch, preset=resolved.name),
        "nan_typed_invalid": check_nan_typed_invalid(
            model, batch, preset=resolved.name),
        "translation_invariance": check_translation_invariance(
            model, batch, preset=resolved.name, seed=seed),
        "rotation_equivariance": check_rotation_equivariance(
            model, batch, preset=resolved.name, seed=seed),
        "permutation_equivariance": check_permutation_equivariance(
            model, batch, preset=resolved.name, seed=seed),
    }
    if "endpoint_names" in batch:
        reports["relabel_invariance"] = check_relabel_invariance(
            model, batch, preset=resolved.name)
    if "signed_intent" in batch:
        reports["reverse_sensitivity"] = check_reverse_sensitivity(
            model, batch, preset=resolved.name)
    return {
        "preset": resolved.name,
        "preset_explicit": explicit,
        "preset_source": resolved.source,
        "passed": all(r["passed"] for r in reports.values()),
        "checks": reports,
    }


def _main(argv: Sequence[str] | None = None) -> int:
    import argparse
    import json

    from .dummy_equivariant_model import make_dummy_batch, make_dummy_model

    parser = argparse.ArgumentParser(
        prog="equivariance_harness",
        description="Run the feature-gate harness on the bundled dummy model.",
    )
    parser.add_argument(
        "--preset", choices=sorted(PRESETS), default=None,
        help="tolerance preset (OPEN DECISION: '30h_v1.1' = rel 1e-5 + abs "
             "1e-6 per 30h plan S3.3 T1.1; 'r3_v3' = abs 1e-4). Omitting "
             "emits a warning and falls back to '30h_v1.1'.",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    model = make_dummy_model(seed=args.seed)
    batch = make_dummy_batch(seed=args.seed)
    report = run_feature_gates(model, batch, preset=args.preset, seed=args.seed)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())
