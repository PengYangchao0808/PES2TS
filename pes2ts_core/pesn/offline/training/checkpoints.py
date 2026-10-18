"""Checkpoint save/restore with optimizer, RNG, and dependency hashes (B4).

Two checkpoint families live here:

* ``pesn_checkpoint_v1`` (:func:`save_checkpoint` / :func:`load_checkpoint`)
  — the historical generic checkpoint (kept byte-for-byte for compatibility;
  never used for new dual-TS BC runs).
* ``pesn_dual_ts_bc_checkpoint_v2`` (:func:`save_bc_checkpoint_v2`) — the
  v4.1 BC checkpoint carrying a full ``identity`` block (features spec digest,
  action version, seed, recipe, λ_a, λ_dir, sampler config, protocol sha256)
  plus a real ``content_digest``.  This is the only schema
  :func:`pes2ts_core.pesn.offline.training.dual_ts_iql.load_bc_actor` accepts.
"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Final

import numpy as np

from pes2ts_core.utils.hashing import JSONValue

CHECKPOINT_SCHEMA: Final[str] = "pesn_checkpoint_v1"
BC_CHECKPOINT_SCHEMA_V2: Final[str] = "pesn_dual_ts_bc_checkpoint_v2"


class CheckpointError(RuntimeError):
    """Typed checkpoint failure (mismatched config, corrupted state)."""


def save_checkpoint(
    path: str | Path,
    *,
    model,
    optimizer,
    epoch: int,
    best_metric: float,
    config_sha256: str,
    dataset_manifest_sha256: str,
    extra: dict[str, JSONValue] | None = None,
) -> None:
    torch = model.torch
    payload = {
        "schema_version": CHECKPOINT_SCHEMA,
        "epoch": epoch,
        "best_metric": best_metric,
        "config_sha256": config_sha256,
        "dataset_manifest_sha256": dataset_manifest_sha256,
        "model_state": model.state_dicts(),
        "optimizer_state": optimizer.state_dict(),
        "torch_rng_state": torch.get_rng_state(),
        "numpy_rng_state": np.random.get_state(),
        "python_rng_state": random.getstate(),
        "extra": extra or {},
    }
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, target)


def load_checkpoint(
    path: str | Path,
    *,
    model,
    optimizer,
    expected_config_sha256: str | None = None,
    expected_dataset_sha256: str | None = None,
) -> dict[str, object]:
    torch = model.torch
    source = Path(path)
    if not source.is_file():
        raise CheckpointError(f"checkpoint missing: {source}")
    payload = torch.load(source, map_location="cpu", weights_only=False)
    if payload.get("schema_version") != CHECKPOINT_SCHEMA:
        raise CheckpointError(
            f"checkpoint schema {payload.get('schema_version')!r} unsupported"
        )
    if (
        expected_config_sha256 is not None
        and payload["config_sha256"] != expected_config_sha256
    ):
        raise CheckpointError(
            "checkpoint config hash mismatch: a changed config requires a new run"
        )
    if (
        expected_dataset_sha256 is not None
        and payload["dataset_manifest_sha256"] != expected_dataset_sha256
    ):
        raise CheckpointError(
            "checkpoint dataset hash mismatch: a changed dataset requires a new run"
        )
    model.load_state_dicts(payload["model_state"])
    optimizer.load_state_dict(payload["optimizer_state"])
    torch.set_rng_state(payload["torch_rng_state"])
    np.random.set_state(payload["numpy_rng_state"])
    random.setstate(payload["python_rng_state"])
    return payload


def _canonical_for_digest(torch, obj):
    """Canonicalize a payload value for content-digest hashing.

    Tensors/arrays reduce to ``dtype + shape + byte sha256`` so the digest is
    byte-level deterministic across processes (never float-print dependent).
    """
    if isinstance(obj, dict):
        return {str(k): _canonical_for_digest(torch, v)
                for k, v in sorted(obj.items(), key=lambda kv: str(kv[0]))}
    if isinstance(obj, (list, tuple)):
        return [_canonical_for_digest(torch, v) for v in obj]
    if hasattr(obj, "detach") and hasattr(obj, "numpy"):
        t = obj.detach().to("cpu", copy=True).contiguous()
        return {"__tensor__": [str(t.dtype), list(t.shape),
                               hashlib.sha256(t.numpy().tobytes()).hexdigest()]}
    if isinstance(obj, np.ndarray):
        return {"__ndarray__": [
            str(obj.dtype), list(obj.shape),
            hashlib.sha256(np.ascontiguousarray(obj).tobytes()).hexdigest()]}
    if isinstance(obj, (np.integer, np.floating, np.bool_)):
        return obj.item()
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return repr(obj)


def checkpoint_content_digest(torch, payload) -> str:
    """Real content digest of a checkpoint payload (self-digest excluded)."""
    core = {k: v for k, v in payload.items() if k != "content_digest"}
    text = json.dumps(_canonical_for_digest(torch, core),
                      sort_keys=True, allow_nan=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def save_bc_checkpoint_v2(
    path: str | Path,
    *,
    policy,
    optimizer,
    epoch: int,
    best_metric: float,
    identity: dict[str, JSONValue],
    sampler_state: dict | None = None,
    extra: dict[str, JSONValue] | None = None,
) -> str:
    """Write a ``pesn_dual_ts_bc_checkpoint_v2`` with a real content digest.

    ``identity`` MUST carry (features_spec_digest, action_version, seed,
    recipe, lambda_a, lambda_dir, sampler_config, protocol_sha256) — the exact
    block :func:`dual_ts_iql.load_bc_actor` verifies.  ``model_state`` is the
    policy's ``state_dicts()`` (``{"encoder": ..., "policy": ...}``).  Returns
    the checkpoint's ``content_digest``.
    """
    torch = policy.torch
    payload = {
        "schema_version": BC_CHECKPOINT_SCHEMA_V2,
        "epoch": epoch,
        "best_metric": best_metric,
        "identity": dict(identity),
        "model_state": policy.state_dicts(),
        "optimizer_state": optimizer.state_dict(),
        "torch_rng_state": torch.get_rng_state(),
        "numpy_rng_state": np.random.get_state(),
        "python_rng_state": random.getstate(),
        "extra": extra or {},
    }
    if sampler_state is not None:
        payload["sampler_state"] = sampler_state
    payload["content_digest"] = checkpoint_content_digest(torch, payload)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, target)
    return str(payload["content_digest"])


__all__ = [
    "BC_CHECKPOINT_SCHEMA_V2",
    "CHECKPOINT_SCHEMA",
    "CheckpointError",
    "checkpoint_content_digest",
    "load_checkpoint",
    "save_bc_checkpoint_v2",
    "save_checkpoint",
]
