"""CPU deployment benchmark (V5.8 Figure A): batch=1 single-step latency.

Measured on the current process with ``torch.set_num_threads(1)``; reports
mean/median/p95 latency for a **single new frame** conditioned on a history
window, which is the controller's real inference pattern.
"""

from __future__ import annotations

import time

import numpy as np
import torch


def _make_step_input(model, dataset, *, history_len: int = 8, q_dim: int | None = None):
    q_dim = int(q_dim or dataset.q_dim)
    frame_feat_dim = int(dataset.frame_feat_dim)
    static_dim = int(dataset.static_feat_dim)
    rng = np.random.default_rng(17)
    frame = torch.from_numpy(
        rng.normal(0.0, 0.3, (1, history_len, frame_feat_dim)).astype(np.float32))
    static = torch.from_numpy(
        rng.normal(0.0, 0.3, (1, static_dim)).astype(np.float32))
    ds = torch.full((1, history_len), 1.0, dtype=torch.float32)
    return {"frame_feats": frame, "static_feats": static, "delta_s": ds}


@torch.no_grad()
def benchmark_cpu_inference(
    model,
    dataset,
    *,
    history_len: int = 8,
    n_warmup: int = 3,
    n_iters: int = 20,
    threads: int = 1,
) -> dict:
    model.eval()
    model.to("cpu")
    torch.set_num_threads(max(1, int(threads)))
    batch = _make_step_input(model, dataset, history_len=history_len)
    stateful = bool(getattr(model.core, "stateful", False))
    state = model.init_state(1, device="cpu")
    latencies = []
    for i in range(n_warmup + n_iters):
        batch_i = batch
        if stateful:
            batch_i = {k: v[:, -1:] if k in ("frame_feats", "delta_s") else v
                       for k, v in batch.items()}
            state = model.init_state(1, device="cpu")
        t0 = time.perf_counter()
        model(batch_i, state=state)
        latencies.append((time.perf_counter() - t0) * 1000.0)
    measured = np.asarray(latencies[n_warmup:], dtype=np.float64)
    return {
        "batch_size": 1,
        "history_len": int(history_len),
        "stateful": stateful,
        "threads": int(threads),
        "n_iters": int(n_iters),
        "latency_ms_mean": float(measured.mean()),
        "latency_ms_median": float(np.median(measured)),
        "latency_ms_p95": float(np.percentile(measured, 95)),
        "params_total": int(sum(p.numel() for p in model.parameters())),
        "note": "CPU batch=1 single-step latency; not a training-throughput metric",
    }
