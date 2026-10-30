"""V5.4.2 strict-load audit for A0 parity/anchor loading (ADR-0033 §2.2).

The A0 mapping is FROZEN and explicit: identity over every key of the
source ``payload["model"]`` state dict (the legacy 43-key joint layout:
37 backbone + 6 process head), PLUS the one declared dual-write alias —
the source ``backbone.w_v.weight`` / ``backbone.w_v.bias`` entries are
written to BOTH ``backbone.w_v.*`` and ``move_decoder.w_v.*`` (the two
targets are the SAME shared module object, so the alias is a
state-dict-view duplication, not a value fork).  NO other aliasing exists.

Audit contract (config ``a0_migration.mapping: explicit_strict_audit``):

* ``missing`` / ``unexpected`` / ``shape_mismatch`` / ``dtype_mismatch``
  must ALL be empty — otherwise a TYPED :class:`V542LoadError` with the
  offending category in ``code`` (never a bare RuntimeError, never a
  silent ``strict=False`` fallback, never runtime shape heuristics);
* load runs ``strict=True`` only;
* every written key is verified with ``torch.equal`` AFTER the load;
* the audit does NOT reject the declared ``w_v`` dual-write (it is part
  of the frozen mapping table);
* the return value is an auditable JSON-able record.  The
  ``mapping_table_sha256`` field is the CANONICAL digest of the mapping
  table (gate-0 advisory A3 closed): the table is canonicalized with
  :func:`pes2ts_core.pesn.models.v542.identity.canonical_sha256` (compact
  sorted JSON, UTF-8) — the SAME recipe every other V5.4.2 identity field
  uses, so the digest is reproducible everywhere and no longer a local
  placeholder.

RNG discipline (gate-1 C4): building the frozen mapping table
constructs a throwaway ``KeyAtomV53MPModel`` which WOULD consume the
global torch RNG stream.  :func:`a0_mapping_table` therefore runs that
construction inside ``torch.random.fork_rng(devices=[])`` — the global
torch CPU RNG state (and, since no CUDA device is touched by the
construction, the CUDA RNG states too) is restored byte-identically on
exit.  ``load_a0_strict`` / ``a0_mapping_table`` can NEVER perturb the
training RNG stream (asserted by
``tests/pesn/test_v542_model.py::test_a0_loader_never_perturbs_global_rng``).
"""

from __future__ import annotations

from typing import Final, Mapping, Sequence

import torch
import torch.nn as nn

from pes2ts_core.pesn.models.key_atom_heads_v3 import KeyAtomV53MPModel
from pes2ts_core.pesn.models.v542.identity import canonical_sha256
from pes2ts_core.pesn.models.v542.interfaces import V542LoadError

#: Frozen A0 dual-write alias: the single ``w_v`` source entries map to
#: BOTH state-dict targets (shared module object — same tensor).
A0_WV_SOURCE_KEYS: Final[tuple[str, ...]] = (
    "backbone.w_v.weight", "backbone.w_v.bias")
A0_WV_TARGET_SUFFIXES: Final[tuple[str, ...]] = (
    "backbone.w_v.weight", "backbone.w_v.bias",
    "move_decoder.w_v.weight", "move_decoder.w_v.bias")

#: Audit record schema id.
A0_LOAD_AUDIT_SCHEMA: Final[str] = "pes2ts_v542_a0_strict_load_audit_v1"


def a0_mapping_table() -> dict[str, list[str]]:
    """The frozen A0 mapping: identity over all source keys + w_v dual-write.

    Built from the legacy 43-key layout (``KeyAtomV53MPModel`` state
    dict == V541 E0 state dict == V53MP C state dict).  Every source key
    maps to exactly one target (itself), except the two ``w_v`` entries
    which additionally map to the decoder's copies of the same tensors.

    The throwaway legacy construction runs inside
    ``torch.random.fork_rng(devices=[])`` so the GLOBAL torch RNG stream
    is never perturbed (gate-1 C4).
    """
    with torch.random.fork_rng(devices=[]):
        legacy_keys = sorted(KeyAtomV53MPModel().state_dict().keys())
    table: dict[str, list[str]] = {}
    for key in legacy_keys:
        targets = [key]
        if key in A0_WV_SOURCE_KEYS:
            targets.append(key.replace("backbone.", "move_decoder.", 1))
        table[key] = targets
    return table


def _mapping_table_digest(table: Mapping[str, Sequence[str]]) -> str:
    """Canonical digest of the mapping table (gate-0 advisory A3).

    The table is normalized to ``{source: [targets...]}`` and hashed with
    the ONE frozen V5.4.2 recipe
    :func:`pes2ts_core.pesn.models.v542.identity.canonical_sha256`
    (compact JSON, ``sort_keys=True``, UTF-8, SHA-256 hex) — identical
    semantics to the previous local digest (sorted-table compact JSON) but
    with no local placeholder prefix, so the value is reproducible from
    the table alone anywhere in the campaign.
    """
    return canonical_sha256({k: list(v) for k, v in table.items()})


def _extract_state_dict(payload: object) -> Mapping[str, torch.Tensor]:
    """Accept a raw state dict OR a trainer checkpoint payload."""
    if not isinstance(payload, Mapping):
        raise V542LoadError(
            "V542_LOAD_PAYLOAD",
            f"checkpoint payload must be a mapping, got {type(payload).__name__}")
    if "model" in payload and isinstance(payload["model"], Mapping):
        payload = payload["model"]
    for key, value in payload.items():
        if not isinstance(value, torch.Tensor):
            raise V542LoadError(
                "V542_LOAD_PAYLOAD",
                f"state entry {key!r} is {type(value).__name__}, expected a tensor")
    return payload  # type: ignore[return-value]


def load_a0_strict(
        model: nn.Module,
        payload: object,
        mapping_table: Mapping[str, Sequence[str]] | None = None,
) -> dict[str, object]:
    """Strict-audit load of a legacy joint checkpoint into a V542 A0 model.

    Args:
        model: the target :class:`~pes2ts_core.pesn.models.v542.model.V542Model`
            (or any module whose state dict is the A0 target layout).
        payload: raw state dict OR trainer checkpoint (``payload["model"]``).
        mapping_table: explicit source→targets mapping; ``None`` uses the
            frozen :func:`a0_mapping_table` (identity + w_v dual-write).

    Returns the auditable record::

        {"schema": ..., "ok": True, "missing": [], "unexpected": [],
         "shape_mismatch": [], "dtype_mismatch": [],
         "dual_write_aliases": {"backbone.w_v.*": "move_decoder.w_v.*"},
         "mapping_table": {...}, "mapping_table_sha256": "<canonical hex>",
         "loaded_keys": [...], "torch_equal_verified": [...],
         "strict": True}

    Raises :class:`V542LoadError` with codes ``V542_LOAD_MISSING``,
    ``V542_LOAD_UNEXPECTED``, ``V542_LOAD_SHAPE``, ``V542_LOAD_DTYPE``
    on any non-empty audit category.  ``strict=False`` is NOT an option.
    """
    table = (dict(mapping_table) if mapping_table is not None
             else a0_mapping_table())
    state = _extract_state_dict(payload)
    model_state = model.state_dict()

    # --- expand the mapping over the actual source keys -------------------
    # unexpected = source keys the frozen mapping does not consume.
    unexpected = sorted(set(state) - set(table))
    # missing = (a) source keys the mapping requires but the payload lacks,
    # and (b) target model keys no mapping entry writes.
    missing_source = sorted(k for k in table if k not in state)
    mapped_targets: list[str] = []
    for src in sorted(table):
        mapped_targets.extend(table[src])
    missing_target = sorted(set(model_state) - set(mapped_targets))
    missing = sorted(set(missing_source) | set(missing_target))

    # --- shape / dtype audit against the TARGET slots ----------------------
    shape_mismatch: list[tuple[str, tuple[int, ...], tuple[int, ...]]] = []
    dtype_mismatch: list[tuple[str, str, str]] = []
    for src in sorted(set(table) & set(state)):
        src_tensor = state[src]
        for tgt in table[src]:
            if tgt not in model_state:
                continue  # reported via missing_target above
            ref = model_state[tgt]
            if tuple(ref.shape) != tuple(src_tensor.shape):
                shape_mismatch.append(
                    (tgt, tuple(ref.shape), tuple(src_tensor.shape)))
            if ref.dtype != src_tensor.dtype:
                dtype_mismatch.append(
                    (tgt, str(ref.dtype), str(src_tensor.dtype)))

    if missing:
        raise V542LoadError(
            "V542_LOAD_MISSING",
            "strict audit: missing keys (source or target) — "
            f"missing_source={missing_source} missing_target={missing_target}")
    if unexpected:
        raise V542LoadError(
            "V542_LOAD_UNEXPECTED",
            f"strict audit: source keys outside the frozen mapping: {unexpected}")
    if shape_mismatch:
        raise V542LoadError(
            "V542_LOAD_SHAPE",
            "strict audit: shape mismatch — "
            + ", ".join(f"{k}: model{ms} != ckpt{cs}"
                        for k, ms, cs in shape_mismatch))
    if dtype_mismatch:
        raise V542LoadError(
            "V542_LOAD_DTYPE",
            "strict audit: dtype mismatch — "
            + ", ".join(f"{k}: model {md} != ckpt {cd}"
                        for k, md, cd in dtype_mismatch))

    # --- strict=True load through the expanded mapping ---------------------
    expanded: dict[str, torch.Tensor] = {}
    for src in sorted(table):
        for tgt in table[src]:
            # The dual-write targets are the same shared tensor object in
            # the live model; both slots receive the identical source value.
            expanded[tgt] = state[src]
    # strict=True refuses anything missing/unexpected at the torch level too.
    missing_torch, unexpected_torch = model.load_state_dict(
        expanded, strict=True)
    if list(missing_torch) or list(unexpected_torch):  # pragma: no cover - guarded
        raise V542LoadError(
            "V542_LOAD_MISSING",
            f"torch strict load reported missing={list(missing_torch)} "
            f"unexpected={list(unexpected_torch)}")

    # --- per-key torch.equal verification (post-load) ----------------------
    post = model.state_dict()
    torch_equal: list[str] = []
    for tgt in sorted(expanded):
        if not torch.equal(post[tgt], expanded[tgt]):
            raise V542LoadError(
                "V542_LOAD_VALUE",
                f"post-load torch.equal FAILED for {tgt}")
        torch_equal.append(tgt)
    # The declared dual-write must hold: the two w_v targets are EQUAL
    # (same shared module object ⇒ same tensor values by construction).
    for src in A0_WV_SOURCE_KEYS:
        tgt = src.replace("backbone.", "move_decoder.", 1)
        if not torch.equal(post[src], post[tgt]):  # pragma: no cover - invariant
            raise V542LoadError(
                "V542_LOAD_VALUE",
                f"w_v dual-write diverged: {src} != {tgt}")

    dual_write = {src: list(table[src]) for src in A0_WV_SOURCE_KEYS
                  if src in table and len(table[src]) > 1}
    return {
        "schema": A0_LOAD_AUDIT_SCHEMA,
        "ok": True,
        "strict": True,
        "missing": [],
        "unexpected": [],
        "shape_mismatch": [],
        "dtype_mismatch": [],
        "n_source_keys": len(state),
        "n_mapped_targets": len(expanded),
        "dual_write_aliases": dual_write,
        "mapping_table_sha256": _mapping_table_digest(table),
        "loaded_keys": sorted(expanded),
        "torch_equal_verified": torch_equal,
    }


__all__ = [
    "A0_LOAD_AUDIT_SCHEMA",
    "A0_WV_SOURCE_KEYS",
    "A0_WV_TARGET_SUFFIXES",
    "a0_mapping_table",
    "load_a0_strict",
]
