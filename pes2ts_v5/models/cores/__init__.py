"""Memory-core registry (M01–M14) + parameter-budget resolution utility.

``resolve_core_config`` implements the *equal-Core-parameter* fairness axis
(plan §3.4): for a target budget (0.03M / 0.1M / 0.5M) it picks the hidden size
that makes the **Core-only** parameter count closest to the budget, so
architectures differ in inductive bias rather than in capacity.
"""

from __future__ import annotations

import math

import numpy as np

from pes2ts_v5.models.cores.base import CoreBase, count_params
from pes2ts_v5.models.cores.m01_stateless_mlp import StatelessMLP
from pes2ts_v5.models.cores.m02_window_mlp import WindowMLP
from pes2ts_v5.models.cores.m03_causal_tcn import CausalTCN
from pes2ts_v5.models.cores.m04_gru import GRUCore
from pes2ts_v5.models.cores.m05_time_aware_gru import TimeAwareGRU
from pes2ts_v5.models.cores.m06_lstm import LSTMCore
from pes2ts_v5.models.cores.m07_xlstm import XLSTMSmall
from pes2ts_v5.models.cores.m08_cfc import CfCSmall
from pes2ts_v5.models.cores.m09_ltc import LTCSmall
from pes2ts_v5.models.cores.m10_latent_neural_ode import LatentNeuralODE
from pes2ts_v5.models.cores.m11_tiny_causal_transformer import TinyCausalTransformer
from pes2ts_v5.models.cores.m12_diagonal_selective_ssm import DiagonalSelectiveSSM
from pes2ts_v5.models.cores.m13_s4d import S4D
from pes2ts_v5.models.cores.m14_mamba2_small import Mamba2Small

CORE_REGISTRY: dict[str, type[CoreBase]] = {
    "M01": StatelessMLP,
    "M02": WindowMLP,
    "M03": CausalTCN,
    "M04": GRUCore,
    "M05": TimeAwareGRU,
    "M06": LSTMCore,
    "M07": XLSTMSmall,
    "M08": CfCSmall,
    "M09": LTCSmall,
    "M10": LatentNeuralODE,
    "M11": TinyCausalTransformer,
    "M12": DiagonalSelectiveSSM,
    "M13": S4D,
    "M14": Mamba2Small,
}

#: architecture metadata for reports / model_selection (simplification is explicit)
CORE_META: dict[str, dict] = {
    "M01": {"title": "Stateless MLP", "paradigm": "feedforward", "stateful": False,
            "simplified": False, "history": "none (single frame)"},
    "M02": {"title": "Window-MLP", "paradigm": "feedforward", "stateful": False,
            "simplified": False, "history": "fixed causal window 4"},
    "M03": {"title": "Causal TCN", "paradigm": "cnn", "stateful": False,
            "simplified": False, "history": "dilated causal conv (1,2,4,8)"},
    "M04": {"title": "GRU", "paradigm": "rnn", "stateful": True,
            "simplified": False, "history": "full recurrent state"},
    "M05": {"title": "Time-aware GRU", "paradigm": "continuous-rnn", "stateful": True,
            "simplified": False, "history": "recurrent + log1p(delta_s)"},
    "M06": {"title": "LSTM", "paradigm": "rnn", "stateful": True,
            "simplified": False, "history": "full recurrent state"},
    "M07": {"title": "xLSTM (simplified)", "paradigm": "gated-memory", "stateful": True,
            "simplified": True,
            "simplification": "sLSTM-only exponential gating; no mLSTM matrix memory"},
    "M08": {"title": "CfC", "paradigm": "continuous-rnn", "stateful": True,
            "simplified": True,
            "simplification": "closed-form exponential gate; no mixed ODE solver"},
    "M09": {"title": "LTC (simplified)", "paradigm": "continuous-rnn", "stateful": True,
            "simplified": True,
            "simplification": "liquid time-constant cell with exponential-Euler step"},
    "M10": {"title": "Latent Neural ODE", "paradigm": "continuous-depth", "stateful": False,
            "simplified": True, "simplification": "fixed 2-step Euler; NFE recorded",
            "history": "latent flow per frame"},
    "M11": {"title": "Tiny Causal Transformer", "paradigm": "attention", "stateful": False,
            "simplified": False, "history": "full history (causal mask)"},
    "M12": {"title": "Diagonal Selective SSM", "paradigm": "ssm", "stateful": False,
            "simplified": True,
            "simplification": "python scan over full sequence; incremental state carry "
                             "not exposed (history-window feeding used in rollout)"},
    "M13": {"title": "S4D", "paradigm": "ssm", "stateful": False,
            "simplified": True,
            "simplification": "real-diagonal state dynamics, no complex rotation; "
                             "incremental state carry not exposed"},
    "M14": {"title": "Mamba-2 Small", "paradigm": "ssm", "stateful": False,
            "simplified": True,
            "simplification": "selective scan python loop + gated MLP; no chunked "
                             "parallel scan; incremental state carry not exposed"},
}

_SAFE_HIDDEN = tuple(range(16, 513, 8))


def build_core(
    name: str,
    *,
    hidden_dim: int,
    layers: int = 2,
    dropout: float = 0.0,
    **kwargs,
) -> CoreBase:
    key = str(name).strip().upper()
    if key not in CORE_REGISTRY:
        raise KeyError(f"unknown core {name!r}; choices={sorted(CORE_REGISTRY)}")
    return CORE_REGISTRY[key](hidden_dim=int(hidden_dim), layers=int(layers),
                              dropout=float(dropout), **kwargs)


def _param_count_at(name: str, hidden_dim: int, layers: int, cache: dict) -> int:
    key = (hidden_dim, layers)
    if key not in cache:
        cache[key] = count_params(build_core(name, hidden_dim=hidden_dim, layers=layers))
    return cache[key]


def resolve_core_config(
    name: str,
    target_params: int,
    *,
    layers: int = 2,
    candidates: tuple[int, ...] = _SAFE_HIDDEN,
) -> dict:
    """Pick the hidden size whose Core parameter count is closest to target.

    Uses a quadratic fit on three probe points, then a local exact-parameter
    refinement over the candidate grid.  Returns a dict with
    ``hidden_dim``, ``params``, ``target``, ``rel_error``.
    """
    target = int(target_params)
    if target <= 0:
        raise ValueError("target_params must be positive")
    cache: dict = {}
    probe = [32, 64, 128]
    xs = np.asarray(probe, dtype=np.float64)
    ys = np.asarray([_param_count_at(name, h, layers, cache) for h in probe],
                    dtype=np.float64)
    est = 128.0
    if np.all(ys > 0):
        try:
            coeffs = np.polyfit(xs, ys, 2)
            roots = np.roots([coeffs[0], coeffs[1], coeffs[2] - target])
            real = [float(r.real) for r in roots if abs(r.imag) < 1e-6 and r.real > 1.0]
            if real:
                est = min(real)
        except np.linalg.LinAlgError:  # pragma: no cover - defensive
            est = 128.0
    local = {int(max(16, min(512, round(est / 8.0) * 8.0)))}
    local.update(int(max(16, min(512, round(est) + d))) for d in (-48, -32, -16, -8, 0, 8, 16, 32, 48))
    grid = sorted(set(candidates) | local)
    best_h, best_p = None, None
    for h in grid:
        p = _param_count_at(name, h, layers, cache)
        if best_p is None or abs(p - target) < abs(best_p - target):
            best_h, best_p = h, p
    return {
        "hidden_dim": int(best_h),
        "params": int(best_p),
        "target": target,
        "rel_error": float(abs(best_p - target) / target),
        "layers": int(layers),
    }


def core_param_table(
    targets: tuple[int, ...] = (30_000, 100_000, 500_000),
    *,
    layers: int = 2,
) -> list[dict]:
    """Parameter table for all 14 cores across the three fairness budgets."""
    rows = []
    for name in sorted(CORE_REGISTRY):
        row = {"arch": name, "title": CORE_META[name]["title"],
               "simplified": CORE_META[name]["simplified"]}
        for target in targets:
            info = resolve_core_config(name, target, layers=layers)
            row[f"h_{target}"] = info["hidden_dim"]
            row[f"params_{target}"] = info["params"]
            row[f"rel_{target}"] = round(info["rel_error"], 4)
        rows.append(row)
    return rows


def expected_stateful(name: str) -> bool:
    return bool(CORE_META.get(str(name).upper(), {}).get("stateful", False))


__all__ = [
    "CORE_REGISTRY", "CORE_META", "CoreBase", "build_core", "count_params",
    "resolve_core_config", "core_param_table", "expected_stateful",
]

# keep math import used for linters that require it in older numpys
_ = math
