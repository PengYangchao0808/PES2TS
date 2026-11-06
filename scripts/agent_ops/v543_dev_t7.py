#!/usr/bin/env python3
"""V5.4.3 WP9 — the dev one-shot T7 runner and the dev frame loader.

WHY THIS FILE EXISTS
--------------------
``scripts/agent_ops/v543_rollout.py`` carries the §9.5 :1038–1039 one-shot gate
(``audit/dev_rollout_access_ledger.jsonl``) but deliberately does NOT wire a dev
frame loader: its dev branch calls ``require_dev_frame_loader`` BEFORE the claim
and, with no loader, refuses with ``V543_ROLLOUT_DEV_LOADER_UNAVAILABLE`` while
the one shot stays unburned.  This module is the missing loader plus the runner
that hands it to ``v543_rollout.run``.  Nothing in the rollout/eval/training
code is modified (plan §11.3) — the dev read is expressed with the SAME frozen
primitives the inner-val arm uses:

* frame selection ....... ``v543_eval.inner_val_positions`` (the frozen
  rule: ``eval_curves`` curves in sorted slice order, a deterministic uniform
  ``linspace`` spread inside every ``(curve, direction)`` group, no RNG);
* frame construction ..... ``v543_eval.build_seed_frames`` (encoder + Bridge,
  one forward per collated batch, ``FRAMES_PER_BATCH = 32``), so a dev frame is
  produced by the very code path that produced the inner-val frames;
* features ............... the frozen dev feature pack
  (``pes2ts_core/pesn/offline/data/key_atom_dev_pack_v53mp``) through the
  documented ``cache[position] -> built dict`` protocol, wrapped in the tiny
  ``DevPackCache`` adapter because ``_batch_from_positions`` calls
  ``lru.get(position)`` while the pack exposes ``view[position]``;
* supervision ........... ``dataset.label_tables()`` and the handled
  ``label_status`` records of the same columnar dataset (exactly what
  ``open_inner_val_seed`` returns for inner-val).

The loader returns the contract ``v543_rollout.run`` reads::

    {"frames", "positions", "rule", "dataset", "lru", "train_pos",
     "labels_v52", "endpoint_labels"}

plus audit-only extras (``positions_sha256``, ``n_frames``, ``dev_pack``).

THREE MODES
-----------
``--dry-run-loader`` (DEFAULT)
    Build the loader object and validate the EXACT frame-build code path on a
    ``(curve, direction)`` slice of the frozen pool (inner-val positions of one
    seed) with a real ``_FeatureLRU``.  It asserts the returned structure
    key-by-key, the determinism of ``rule`` / ``positions_sha256``, that the
    dev ledger length is unchanged, and that zero dev-pack files were opened.
    No pack file is read and no ledger claim is appended.

``--preflight``
    Read-only mirror of every ``v543_rollout.run`` step that happens BEFORE the
    §9.5 claim: the mode/flag gate, the loader gate, the one-shot ledger state,
    the campaign config, the pinned A0 checkpoints, the T6 select report, both
    champion checkpoints (path + SHA) and the frozen baseline.  It also reports
    whether the dev budget makes stage B0 ``not_checked`` and probes the pack
    path (stat only).  It never claims and never opens the pack.

``--execute``
    One-shot dev run.  Builds the loader, builds the ``argparse.Namespace``
    that mirrors ``v543_rollout.main``'s parser defaults (``mode=dev``,
    ``dev_onetime=True``, the frozen select report, the frozen baseline) and
    calls ``v543_rollout.run(args)``; the §9.5 claim happens inside
    ``run`` BEFORE the loader is invoked (claim → read), exactly as the gate
    requires.  MUST be run ONCE for the campaign: a second visit is the typed
    ``V543_ROLLOUT_DEV_ALREADY_USED`` refusal (audit event, not a retry).

WHY ``--eval-curves`` DEFAULTS TO 589 (AND NOT 1024)
----------------------------------------------------
The frozen inner-val baseline document was measured at 1024 curves × 8 frames.
The dev subset has exactly 589 curves, so the dev frame set is a DIFFERENT
frame set.  ``v543_rollout.build_report`` compares stage B0 with the frozen
baseline only when the requested budget matches; declaring the dev budget as
589 curves makes the comparison honestly ``not_checked`` (different frame set)
instead of tripping ``V543_ROLLOUT_B0_BASELINE_MISMATCH`` on a comparison that
was never meaningful.  ``frames_per_curve``/``max_steps`` keep the inner-val
defaults (8 / 8); inside each curve the 8 frames are split over the two
direction groups exactly as inner-val does (``8 // 2 = 4`` per direction).

CLI::

    # default: pool-slice dry run (no dev pack, no claim)
    PYTHONPATH=. python scripts/agent_ops/v543_dev_t7.py

    # the one-shot dev execution (WP9: the orchestrator runs this ONCE)
    PYTHONPATH=. python scripts/agent_ops/v543_dev_t7.py --execute

Typed refusals: :class:`DevT7Error` (own codes) plus ``RolloutError`` /
``V543Error`` / ``DevPackError`` propagated from the frozen code.  Every exit
is a typed code, never a bare assertion.  ``confirm`` is never referenced:
there is no code path in this module that can read it.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Final, Mapping, Sequence

import numpy as np

from scripts.agent_ops import v543_rollout as rollout

# ---------------------------------------------------------------------------
# frozen identity of this runner
# ---------------------------------------------------------------------------
#: Runner schema stamped on the dry-run document and on the rollout report.
RUNNER_SCHEMA: Final[str] = "pesn_v543_dev_t7_runner_v1"
#: Dry-run evidence schema.
DRYRUN_SCHEMA: Final[str] = "pesn_v543_dev_t7_dryrun_v1"
#: The frozen dev subset size (plan §9.5 :1038: 589 curves).
DEV_CURVES_PLAN: Final[int] = 589
#: Default dev report (separate from the inner-val ``t7_rollout_champions``).
DEFAULT_DEV_OUT: Final[str] = (
    "outputs/pesn/v543_dual_decoder_50k/reports/t7_rollout_dev.json")
#: Default dry-run evidence path.
DEFAULT_DRYRUN_OUT: Final[str] = (
    "outputs/pesn/v543_dual_decoder_50k/reports/wp9_dev_loader_dryrun.json")
#: Default preflight evidence path (the one-shot safety check).
DEFAULT_PREFLIGHT_OUT: Final[str] = (
    "outputs/pesn/v543_dual_decoder_50k/reports/wp9_dev_loader_preflight.json")
#: The exact loader contract keys ``v543_rollout.run`` reads.
CONTRACT_KEYS: Final[tuple[str, ...]] = (
    "frames", "positions", "rule", "dataset", "lru", "train_pos",
    "labels_v52", "endpoint_labels")
#: Audit-only extras this loader adds (the rollout ignores them).
EXTRA_KEYS: Final[tuple[str, ...]] = ("positions_sha256", "n_frames",
                                      "dev_pack")
#: Counter of REAL dev-pack opens (dry-run asserts it stays 0).
PACK_OPENS: dict[str, int] = {"n": 0}


class DevT7Error(RuntimeError):
    """Typed refusal of the dev T7 loader/runner (``code`` + ``detail``)."""

    def __init__(self, code: str, detail: str = "") -> None:
        super().__init__(f"{code}: {detail}" if detail else code)
        self.code = str(code)
        self.detail = str(detail)


def _require(cond: bool, code: str, detail: str) -> None:
    if not cond:
        raise DevT7Error(code, detail)


def _log(msg: str) -> None:
    print(f"[v543devt7] {msg}", file=sys.stderr, flush=True)


def _resolve(path: str | Path) -> Path:
    out = Path(str(path))
    return out if out.is_absolute() else rollout.PROJECT_ROOT / out


# ---------------------------------------------------------------------------
# frozen data view / split / dev positions (same paths as open_inner_val_seed)
# ---------------------------------------------------------------------------
def data_contract() -> dict[str, Any]:
    """Read the pinned data view (config + tables dir + split manifest)."""
    from scripts.agent_ops.v542_a0_parity import (
        DEFAULT_CONFIG as DATA_CONFIG,
        load_pinned_assets,
    )

    config, _assets = load_pinned_assets(rollout.PROJECT_ROOT / DATA_CONFIG)
    tables = str((config.get("data_contract") or {}).get("tables_dir")
                 or "/pes2ts_data/v53mp_50k_data")
    split = str((config.get("split_contract") or {}).get("split_manifest")
                or "/pes2ts_data/v53mp_50k_data/"
                   "split_manifest_v53mp.json")
    return {"config": config, "tables_dir": Path(tables),
            "split_manifest": Path(split), "data_config": DATA_CONFIG}


def open_dataset(data: Mapping[str, Any] | None = None) -> Any:
    """Open the frozen columnar dataset (read-only, process-cached)."""
    from scripts.agent_ops.v542_a0_parity import _open_frozen_dataset

    data = data or data_contract()
    return _open_frozen_dataset(data["config"])


def dev_positions(dataset: Any, split_doc: Mapping[str, Any]
                  ) -> tuple[list[int], list[str]]:
    """Frozen dev positions (driver rule; shared with the pack builder).

    ``dev_positions_from_split`` is the SINGLE implementation the pack builder
    and the run driver share, so the pack directory and this loader cannot
    diverge on position identity.
    """
    from pes2ts_core.pesn.offline.data.key_atom_dev_pack_v53mp import (
        dev_positions_from_split,
    )

    ids = split_doc.get("subsets", {}).get("dev")
    _require(isinstance(ids, (list, tuple)) and bool(ids),
             "V543_DEV_T7_SPLIT",
             "split manifest carries no non-empty subsets.dev list "
             f"({type(ids).__name__})")
    positions = [int(p) for p in dev_positions_from_split(dataset.samples,
                                                          set(ids))]
    _require(bool(positions), "V543_DEV_T7_DEV_EMPTY",
             "the frozen dev subset has no samples in this dataset view")
    return positions, [str(c) for c in ids]


def read_split(path: Path) -> dict[str, Any]:
    _require(Path(path).is_file(), "V543_DEV_T7_SPLIT_MISSING",
             f"split manifest not found: {path}")
    return json.loads(Path(path).read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# frame selection — the frozen inner-val rule, applied to any frozen slice
# ---------------------------------------------------------------------------
def select_frames(dataset: Any, positions: Sequence[int], *,
                  eval_curves: int, frames_per_curve: int,
                  split_tag: str) -> tuple[list[int], dict[str, Any]]:
    """Select frames with ``v543_eval.inner_val_positions`` (no RNG).

    The rule is reused verbatim; only the human-readable ``rule`` text and the
    ``split``/``rule_source`` audit fields are re-stamped so a reader cannot
    mistake a dev selection for the inner-val one.
    """
    from scripts.agent_ops.v543_eval import inner_val_positions

    _require(int(eval_curves) >= 1, "V543_DEV_T7_BUDGET",
             f"eval_curves must be >= 1, got {eval_curves!r}")
    _require(int(frames_per_curve) >= 1, "V543_DEV_T7_BUDGET",
             f"frames_per_curve must be >= 1, got {frames_per_curve!r}")
    selected, inner_rule = inner_val_positions(
        [int(p) for p in positions], dataset,
        eval_curves=int(eval_curves),
        frames_per_curve=int(frames_per_curve))
    selected = [int(p) for p in selected]
    rule = dict(inner_rule)
    rule.update({
        "rule": (
            "first N {tag} curves of the frozen sorted slice; inside each "
            "curve every (curve, direction) group is sampled with a "
            "deterministic uniform linspace spread of frames (no RNG) — the "
            "frozen v543_eval.inner_val_positions rule, reused verbatim"
        ).format(tag=split_tag),
        "split": str(split_tag),
        "rule_source": "scripts.agent_ops.v543_eval.inner_val_positions",
        "frames_per_batch": int(rollout.FRAMES_PER_BATCH),
        "why_reused": (
            "§9.4 :1018 makes the CURVE the bootstrap cluster and §9.4 :1031 "
            "needs >=3 q points per curve, so the inner-val frame set is "
            "built per (curve, direction) group; the dev frame set must "
            "follow the same construction to be a comparable starting-point "
            "set (and to keep the rollout stage arithmetic identical)"),
    })
    return selected, rule


# ---------------------------------------------------------------------------
# the dev pack cache adapter (the _FeatureLRU protocol)
# ---------------------------------------------------------------------------
class DevPackCache:
    """``cache[position] -> built dict`` over the frozen dev feature pack.

    ``key_atom_v53mp_train_v3._batch_from_positions`` calls
    ``lru.get(int(p))``; :class:`DevFeaturePackView` exposes
    ``view[int(p)]``.  This adapter is the whole glue: one blob decoded per
    access, no caching, no writes (the walk re-reads a bounded set).
    """

    def __init__(self, view: Any) -> None:
        self._view = view
        self.lookups = 0

    def get(self, position: int) -> Mapping[str, Any]:
        self.lookups += 1
        return self._view[int(position)]

    def __getitem__(self, position: int) -> Mapping[str, Any]:
        return self.get(position)

    def contains(self, position: int) -> bool:
        return True

    @property
    def n_positions(self) -> int:
        return int(self._view.n_positions)


# ---------------------------------------------------------------------------
# frame construction (the frozen encoder + Bridge path)
# ---------------------------------------------------------------------------
def bridge_checkpoint_sha(bridge: Any) -> str:
    """The A0 checkpoint digest the Bridge records in every context."""
    describe = getattr(bridge, "describe", None)
    sha = ""
    if callable(describe):
        sha = str((describe() or {}).get("encoder_ckpt_sha256") or "")
    _require(len(sha) == 64
             and all(c in "0123456789abcdef" for c in sha),
             "V543_DEV_T7_BRIDGE_SHA",
             "the Bridge does not carry a pinned 64-hex encoder checkpoint "
             f"digest (got {sha!r}); the dev run must pin the A0 digest")
    return sha


def build_frames(*, encoder: Any, bridge: Any, dataset: Any, cache: Any,
                 positions: Sequence[int],
                 labels_v52: Any, endpoint_labels: Any,
                 with_basis: bool, with_strata: bool
                 ) -> tuple[list[Any], Any]:
    """Encode ``positions`` through the frozen ``build_seed_frames`` path."""
    from scripts.agent_ops.v543_eval import build_seed_frames
    from scripts.agent_ops.v543_s0_probe import positions_meta

    meta = (positions_meta(dataset, positions) if with_strata
            else [{"stratum": ""} for _ in positions])
    seed_frames = build_seed_frames(
        encoder, bridge, dataset, cache, positions,
        checkpoint_sha256=bridge_checkpoint_sha(bridge),
        meta=meta, labels_v52=labels_v52,
        endpoint_labels=endpoint_labels, with_basis=bool(with_basis))
    frames = list(seed_frames.frames)
    _require(len(frames) == len(positions), "V543_DEV_T7_FRAMES",
             f"{len(frames)} frames for {len(positions)} positions")
    return frames, seed_frames


def make_payload(*, frames: Sequence[Any], positions: Sequence[int],
                 rule: Mapping[str, Any], dataset: Any, cache: Any,
                 train_pos: Sequence[int], labels_v52: Any,
                 endpoint_labels: Any) -> dict[str, Any]:
    """Assemble the loader contract dict (single implementation)."""
    pos = [int(p) for p in positions]
    payload = {
        "frames": list(frames),
        "positions": pos,
        "rule": dict(rule),
        "dataset": dataset,
        "lru": cache,
        "train_pos": [int(p) for p in train_pos],
        "labels_v52": labels_v52,
        "endpoint_labels": endpoint_labels,
        "positions_sha256": rollout.canonical_sha256(pos),
        "n_frames": int(len(frames)),
    }
    return payload


# ---------------------------------------------------------------------------
# contract validation (dry-run evidence and post-conditions)
# ---------------------------------------------------------------------------
def validate_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Assert the loader contract key-by-key; return a shape/flat summary."""
    missing = [key for key in CONTRACT_KEYS if key not in payload]
    _require(not missing, "V543_DEV_T7_CONTRACT",
             f"loader payload misses contract keys {missing}")
    frames = payload["frames"]
    positions = [int(p) for p in payload["positions"]]
    _require(isinstance(frames, list), "V543_DEV_T7_CONTRACT",
             "frames must be a list")
    _require(len(frames) == len(positions)
             and int(payload.get("n_frames", -1)) == len(frames),
             "V543_DEV_T7_CONTRACT",
             f"{len(frames)} frames vs {len(positions)} positions vs "
             f"n_frames={payload.get('n_frames')!r}")
    _require(bool(positions), "V543_DEV_T7_CONTRACT", "empty positions")
    _require(all(positions[i] < positions[i + 1]
                 for i in range(len(positions) - 1)),
             "V543_DEV_T7_CONTRACT", "positions must be strictly ascending")
    _require(str(payload.get("positions_sha256"))
             == rollout.canonical_sha256(positions), "V543_DEV_T7_CONTRACT",
             "positions_sha256 does not match the canonical position digest")
    _require(hasattr(payload["dataset"], "samples"), "V543_DEV_T7_CONTRACT",
             "dataset object has no samples")
    _require(callable(getattr(payload["lru"], "get", None)),
             "V543_DEV_T7_CONTRACT", "lru object has no .get(position)")
    train_pos = [int(p) for p in payload["train_pos"]]
    _require(bool(train_pos), "V543_DEV_T7_CONTRACT", "empty train_pos")
    _require(isinstance(payload["rule"], Mapping)
             and bool(payload["rule"].get("rule")), "V543_DEV_T7_CONTRACT",
             "rule block is empty")
    curves: set[str] = set()
    directions: dict[str, int] = {}
    strata: dict[str, int] = {}
    intents: dict[str, int] = {}
    n_key_max = 0
    n_key_total = 0
    for index, frame in enumerate(frames):
        for field in ("curve_id", "direction", "stratum", "u", "y", "u0",
                      "pair_slots", "pair_offsets", "q_pred", "q0_raw",
                      "endpoint_logit", "intent_class", "move_eligible"):
            _require(hasattr(frame, field), "V543_DEV_T7_FRAME",
                     f"frame {index} misses {field!r}")
        u = np.asarray(frame.u, dtype=np.float64)
        y = np.asarray(frame.y, dtype=np.float64)
        u0 = np.asarray(frame.u0, dtype=np.float64)
        _require(u.ndim == 2 and u.shape[1] == 3, "V543_DEV_T7_FRAME",
                 f"frame {index}: u shape {u.shape} != [K,3]")
        _require(y.shape == u.shape and u0.shape == u.shape,
                 "V543_DEV_T7_FRAME",
                 f"frame {index}: y/u0 disagree with u {u.shape}")
        _require(bool(np.isfinite(u).all()) and bool(np.isfinite(y).all()),
                 "V543_DEV_T7_FRAME", f"frame {index}: non-finite u/y")
        for field in ("q_pred", "q0_raw", "endpoint_logit"):
            value = float(getattr(frame, field))
            _require(np.isfinite(value), "V543_DEV_T7_FRAME",
                     f"frame {index}: non-finite {field}")
        slots = np.asarray(frame.pair_slots)
        offsets = np.asarray(frame.pair_offsets)
        _require(slots.ndim == 2 and slots.shape[1] == 2,
                 "V543_DEV_T7_FRAME",
                 f"frame {index}: pair_slots shape {slots.shape} != [P,2]")
        _require(offsets.shape == (slots.shape[0], 3), "V543_DEV_T7_FRAME",
                 f"frame {index}: pair_offsets {offsets.shape} != [P,3]")
        key_v = getattr(frame, "key_v", None)
        if key_v is not None:
            key_v = np.asarray(key_v)
            _require(key_v.ndim == 3 and key_v.shape[1] == 8
                     and key_v.shape[2] == 3 and key_v.shape[0] == u.shape[0],
                     "V543_DEV_T7_FRAME",
                     f"frame {index}: key_v shape {key_v.shape} != [K,8,3]")
        curves.add(str(frame.curve_id))
        direction = str(frame.direction)
        directions[direction] = directions.get(direction, 0) + 1
        stratum = str(frame.stratum)
        strata[stratum] = strata.get(stratum, 0) + 1
        intent = str(frame.intent_class)
        intents[intent] = intents.get(intent, 0) + 1
        n_key_max = max(n_key_max, int(u.shape[0]))
        n_key_total += int(u.shape[0])
    return {
        "passed": True,
        "n_frames": len(frames),
        "n_positions": len(positions),
        "n_curves": len(curves),
        "directions": dict(sorted(directions.items())),
        "strata": dict(sorted(strata.items())),
        "intent_classes": dict(sorted(intents.items())),
        "n_key_rows_total": n_key_total,
        "n_key_rows_max": n_key_max,
        "positions_sha256": str(payload["positions_sha256"]),
        "rule_sha256": rollout.canonical_sha256(payload["rule"]),
        "contract_keys_present": list(CONTRACT_KEYS),
    }


# ---------------------------------------------------------------------------
# the dev frame loader (the object handed to v543_rollout.run)
# ---------------------------------------------------------------------------
class DevFrameLoader:
    """Callable ``loader(seed=..., dataset=None, encoder=..., bridge=...)``.

    Construction performs NO I/O (so the dry run can build it and never touch
    the pack).  The dev pack is opened — and therefore read — only when the
    loader is CALLED, which happens inside ``run`` AFTER the §9.5 claim.
    """

    def __init__(self, *, eval_curves: int = DEV_CURVES_PLAN,
                 frames_per_curve: int = rollout.DEFAULT_FRAMES_PER_CURVE,
                 with_basis: bool = True, with_strata: bool = True,
                 pack_dir: str | Path | None = None) -> None:
        self.eval_curves = int(eval_curves)
        self.frames_per_curve = int(frames_per_curve)
        self.with_basis = bool(with_basis)
        self.with_strata = bool(with_strata)
        self.pack_dir = None if pack_dir is None else str(pack_dir)
        self.pack_open_count = 0
        self._data: dict[str, Any] | None = None
        self._dev_positions: list[int] | None = None
        self._dev_curve_ids: list[str] | None = None

    # -- metadata-only helpers (safe for the dry run) ----------------------
    def _contract(self) -> dict[str, Any]:
        if self._data is None:
            self._data = data_contract()
        return self._data

    def dev_positions(self, dataset: Any) -> tuple[list[int], list[str]]:
        if self._dev_positions is None or self._dev_curve_ids is None:
            split_doc = read_split(self._contract()["split_manifest"])
            positions, curve_ids = dev_positions(dataset, split_doc)
            self._dev_positions, self._dev_curve_ids = positions, curve_ids
        return list(self._dev_positions), list(self._dev_curve_ids)

    def resolve_pack_dir(self, dataset: Any
                         ) -> tuple[Path, list[int], str, str]:
        from pes2ts_core.pesn.offline.data.key_atom_dev_pack_v53mp import (
            DEV_PACKS_DIRNAME,
            features_sha256_of,
            positions_sha256,
        )

        positions, _curve_ids = self.dev_positions(dataset)
        fsha = features_sha256_of()
        psha = positions_sha256(positions)
        if self.pack_dir is not None:
            pack_dir = _resolve(self.pack_dir)
        else:
            pack_dir = (self._contract()["tables_dir"] / DEV_PACKS_DIRNAME
                        / fsha / psha)
        return Path(pack_dir), positions, fsha, psha

    def describe_pack_path(self, dataset: Any) -> dict[str, Any]:
        """Resolve the pack path WITHOUT opening any pack file (metadata)."""
        pack_dir, positions, fsha, psha = self.resolve_pack_dir(dataset)
        binding = pack_dir / f"{psha}.json"
        return {
            "pack_dir": str(pack_dir),
            "dir_exists": bool(pack_dir.is_dir()),
            "expected_binding": str(binding),
            "binding_exists": bool(binding.is_file()),
            "features_sha256": fsha,
            "positions_sha256": psha,
            "n_dev_positions": len(positions),
            "n_dev_curves": len(self._dev_curve_ids or []),
            "tables_dir": str(self._contract()["tables_dir"]),
            "files_read": [],
            "note": ("metadata only: existence is checked (stat), no pack "
                     "file is opened, read or hashed by this probe"),
        }

    # -- the one-shot call (reads the pack; only AFTER the claim) ----------
    def __call__(self, *, seed: int, dataset: Any = None, encoder: Any,
                 bridge: Any) -> dict[str, Any]:
        from pes2ts_core.pesn.offline.data.key_atom_dev_pack_v53mp import (
            DevFeaturePackView,
            positions_sha256,
        )
        from pes2ts_core.pesn.offline.training.key_atom_v53mp_train_v3 import (
            FEATURE_LEVEL_V53MP,
            v53mp_pool_split,
        )
        from scripts.campaigns.r3_v53mp_run import _tier_pool_curve_ids

        seed = int(seed)
        data = self._contract()
        ds = dataset if dataset is not None else open_dataset(data)
        dev_pos, dev_curve_ids = self.dev_positions(ds)
        _require(len(dev_curve_ids) == DEV_CURVES_PLAN, "V543_DEV_T7_CURVES",
                 f"the frozen dev subset has {len(dev_curve_ids)} curves, "
                 f"the §9.5 one-shot is defined for {DEV_CURVES_PLAN}")
        pack_dir, _positions, fsha, psha = self.resolve_pack_dir(ds)
        _log(f"opening frozen dev pack {pack_dir} "
             f"({len(dev_pos)} positions) — THE one-shot read")
        started = time.perf_counter()
        view = DevFeaturePackView.open(
            pack_dir, expected_positions=dev_pos,
            feature_level=FEATURE_LEVEL_V53MP, verify_files=True)
        PACK_OPENS["n"] = int(PACK_OPENS["n"]) + 1
        self.pack_open_count += 1
        binding = dict(view.binding)
        _require(str(binding.get("positions_sha256")) == psha,
                 "V543_DEV_T7_PACK_POSITIONS",
                 "the opened pack's positions_sha256 does not match the live "
                 f"dev split ({str(binding.get('positions_sha256'))[:12]}… vs "
                 f"{psha[:12]}…); refusing a pack for a different dataset")
        _require(int(binding.get("n_positions", -1)) == len(dev_pos),
                 "V543_DEV_T7_PACK_POSITIONS",
                 f"pack carries {binding.get('n_positions')} positions, the "
                 f"live dev split has {len(dev_pos)}")
        _require(str(binding.get("features_sha256")) == fsha,
                 "V543_DEV_T7_PACK_FEATURES",
                 "the pack's features_sha256 does not match the LIVE feature "
                 "module (stale pack)")
        pack_seconds = float(time.perf_counter() - started)
        cache = DevPackCache(view)
        positions, rule = select_frames(
            ds, dev_pos, eval_curves=self.eval_curves,
            frames_per_curve=self.frames_per_curve, split_tag="dev")
        rule["dev_pack"] = {
            "path": str(pack_dir),
            "features_sha256": fsha,
            "positions_sha256": psha,
            "n_positions": int(binding.get("n_positions", 0)),
            "feature_level": str(binding.get("feature_level")),
            "verify_files": True,
            "binding_sha256": rollout.canonical_sha256(binding),
            "open_seconds": pack_seconds,
            "opened_after_claim": True,
        }
        rule["dev_curves_planned"] = DEV_CURVES_PLAN
        rule["dev_curves_used"] = len(dev_curve_ids)
        rule["dev_positions_available"] = len(dev_pos)
        labels_v52, endpoint_labels = ds.label_tables()
        split_doc = read_split(data["split_manifest"])
        pool_ids = _tier_pool_curve_ids(split_doc, "50k")
        train_pos, val_pos = v53mp_pool_split(
            ds.samples, pool_curve_ids=pool_ids, seed=seed)
        _log(f"building {len(positions)} dev frames "
             f"({self.eval_curves} curves x {self.frames_per_curve} frames)")
        frames, _seed_frames = build_frames(
            encoder=encoder, bridge=bridge, dataset=ds, cache=cache,
            positions=positions, labels_v52=labels_v52,
            endpoint_labels=endpoint_labels, with_basis=self.with_basis,
            with_strata=self.with_strata)
        payload = make_payload(
            frames=frames, positions=positions, rule=rule, dataset=ds,
            cache=cache, train_pos=train_pos, labels_v52=labels_v52,
            endpoint_labels=endpoint_labels)
        payload["dev_pack"] = dict(rule["dev_pack"])
        payload["dev_pack"]["n_dev_curves"] = len(dev_curve_ids)
        _log(f"dev frames ready: {len(frames)} frames, "
             f"{len({f.curve_id for f in frames})} curves, "
             f"{len(dev_pos)} dev positions in pack")
        return payload


# ---------------------------------------------------------------------------
# the argparse.Namespace handed to v543_rollout.run (mirrors its parser)
# ---------------------------------------------------------------------------
def execute_namespace(args: argparse.Namespace, loader: Any,
                      *, seeds: Sequence[int] | None) -> argparse.Namespace:
    """Mirror ``v543_rollout.main``'s parser defaults + the dev switch."""
    return argparse.Namespace(
        out=str(args.out),
        config=str(args.config),
        mode=rollout.MODE_DEV,
        dev_onetime=True,
        dev_ledger=str(args.dev_ledger),
        select_report=str(args.select_report),
        baseline=None if args.baseline is None else str(args.baseline),
        seeds=None if seeds is None else [int(s) for s in seeds],
        eval_curves=int(args.eval_curves),
        frames_per_curve=int(args.frames_per_curve),
        max_steps=int(args.max_steps),
        no_basis=bool(args.no_basis),
        no_strata=bool(args.no_strata),
        move_champion=None,
        process_champion=None,
        move_champion_checkpoint=None,
        process_champion_checkpoint=None,
        actor=str(args.actor),
        dev_frame_loader=loader,
    )


# ---------------------------------------------------------------------------
# dry run (default): the exact code path on a pool slice, no dev pack, no claim
# ---------------------------------------------------------------------------
def _prefix_curve_positions(dataset: Any, positions: Sequence[int],
                            n_curves: int) -> list[int]:
    """First ``n_curves`` curves of a sorted position slice (cheap prefix)."""
    sorted_pos = np.sort(np.asarray([int(p) for p in positions],
                                    dtype=np.int64)).tolist()
    seen: set[str] = set()
    out: list[int] = []
    for position in sorted_pos:
        curve_id = str(dataset.samples[int(position)].curve_id)
        if curve_id not in seen:
            if len(seen) >= int(n_curves):
                break
            seen.add(curve_id)
        out.append(int(position))
    return out


def run_dry_run(args: argparse.Namespace) -> dict[str, Any]:
    """Validate the loader's frame path on a frozen POOL slice (never dev)."""
    from pes2ts_core.pesn.models.v543 import CoarseEncoder, LatentBridge
    from pes2ts_core.pesn.offline.training.key_atom_v53mp_train_v3 import (
        _FeatureLRU,
        v53mp_pool_split,
    )
    from scripts.agent_ops.v543_s0_probe import (
        load_campaign_config,
        resolve_checkpoints,
    )
    from scripts.campaigns.r3_v53mp_run import _tier_pool_curve_ids

    ledger = rollout.DevAccessLedger(str(args.dev_ledger))
    entries_before = len(ledger.entries())
    pack_opens_before = int(PACK_OPENS["n"])
    started = time.perf_counter()

    data = data_contract()
    dataset = open_dataset(data)
    split_doc = read_split(data["split_manifest"])
    campaign = load_campaign_config(
        rollout.PROJECT_ROOT / str(args.config))
    seeds = ([int(args.seed)] if args.seed is not None
             else [int(campaign["source"]["seeds"][0])])
    seed = int(seeds[0])
    checkpoints = resolve_checkpoints(campaign, seeds)
    checkpoint = dict(checkpoints[seed])
    encoder = CoarseEncoder.load(str(checkpoint["path"]), seed)
    encoder.eval()
    bridge = LatentBridge(encoder_ckpt_sha256=str(checkpoint["sha256"]))
    labels_v52, endpoint_labels = dataset.label_tables()

    pool_ids = _tier_pool_curve_ids(split_doc, "50k")
    train_pos, val_pos = v53mp_pool_split(
        dataset.samples, pool_curve_ids=pool_ids, seed=seed)
    slice_positions = _prefix_curve_positions(
        dataset, val_pos, int(args.slice_curves))
    _require(bool(slice_positions), "V543_DEV_T7_SLICE",
             "the frozen inner-val slice is empty")
    positions, rule = select_frames(
        dataset, slice_positions, eval_curves=int(args.slice_curves),
        frames_per_curve=int(args.frames_per_curve), split_tag="pool_slice")
    lru = _FeatureLRU(dataset, capacity=len(positions) + 128)
    frames, _seed_frames = build_frames(
        encoder=encoder, bridge=bridge, dataset=dataset, cache=lru,
        positions=positions, labels_v52=labels_v52,
        endpoint_labels=endpoint_labels, with_basis=not bool(args.no_basis),
        with_strata=not bool(args.no_strata))
    payload = make_payload(
        frames=frames, positions=positions, rule=rule, dataset=dataset,
        cache=lru, train_pos=train_pos, labels_v52=labels_v52,
        endpoint_labels=endpoint_labels)
    validation = validate_payload(payload)

    # determinism: same input -> same positions and same rule block
    positions_again, rule_again = select_frames(
        dataset, slice_positions, eval_curves=int(args.slice_curves),
        frames_per_curve=int(args.frames_per_curve), split_tag="pool_slice")
    determinism = {
        "positions_equal": positions == positions_again,
        "rule_equal": rule == rule_again,
        "positions_sha256": str(payload["positions_sha256"]),
        "positions_sha256_again": rollout.canonical_sha256(positions_again),
        "rule_sha256": rollout.canonical_sha256(rule),
        "rule_sha256_again": rollout.canonical_sha256(rule_again),
    }
    _require(determinism["positions_equal"] and determinism["rule_equal"]
             and determinism["positions_sha256"]
             == determinism["positions_sha256_again"],
             "V543_DEV_T7_DETERMINISM",
             "the frozen selection rule is not deterministic on the pool "
             "slice")

    # dev-pack PATH resolution only (metadata; construction performs no I/O)
    pack_path_check = None
    loader_opens = None
    if not bool(args.no_pack_path_check):
        probe = DevFrameLoader(
            eval_curves=int(args.eval_curves),
            frames_per_curve=int(args.frames_per_curve),
            with_basis=not bool(args.no_basis),
            with_strata=not bool(args.no_strata),
            pack_dir=args.pack_dir)
        pack_path_check = probe.describe_pack_path(dataset)
        loader_opens = int(probe.pack_open_count)
        pack_path_check["loader_pack_opens"] = loader_opens

    entries_after = len(ledger.entries())
    pack_opens_after = int(PACK_OPENS["n"])
    untouched = entries_after == entries_before
    _require(untouched, "V543_DEV_T7_LEDGER",
             f"the dry run changed the dev ledger "
             f"({entries_before} -> {entries_after} entries)")
    _require(pack_opens_after == pack_opens_before
             and (loader_opens in (None, 0)),
             "V543_DEV_T7_PACK_READ",
             "the dry run opened the dev pack; it must validate on the pool "
             "slice only")

    return {
        "schema": DRYRUN_SCHEMA,
        "runner_schema": RUNNER_SCHEMA,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "mode": "dry_run_loader",
        "ok": True,
        "dev_pack_read": False,
        "dev_ledger_touched": False,
        "confirm_read": False,
        "new_QC_calls": 0,
        "seed": seed,
        "checkpoint": checkpoint,
        "data": {
            "data_config": data["data_config"],
            "tables_dir": str(data["tables_dir"]),
            "split_manifest": str(data["split_manifest"]),
        },
        "pool_slice": {
            "source": "v53mp_pool_split(seed)['val'] (frozen inner-val pool)",
            "slice_curves_requested": int(args.slice_curves),
            "n_pool_val_positions": len(val_pos),
            "n_slice_positions": len(slice_positions),
            "n_selected_frames": len(positions),
            "selection": rule,
        },
        "validation": validation,
        "determinism": determinism,
        "dev_pack_path_check": pack_path_check,
        "dev_ledger": {
            "path": str(ledger.path),
            "entries_before": entries_before,
            "entries_after": entries_after,
            "untouched": bool(untouched),
        },
        "pack_opens": {"before": pack_opens_before, "after": pack_opens_after},
        "wall_seconds": round(time.perf_counter() - started, 2),
        "note": (
            "the dry run builds frames with a real _FeatureLRU over a frozen "
            "POOL slice; it never opens the dev pack and never claims the "
            "§9.5 dev ledger"),
    }


# ---------------------------------------------------------------------------
# preflight: every check run() performs BEFORE the claim (one-shot safety)
# ---------------------------------------------------------------------------
def run_preflight(args: argparse.Namespace) -> dict[str, Any]:
    """Exercise the pre-claim half of ``v543_rollout.run`` — WITHOUT claiming.

    ``run`` refuses a dev rollout before the claim only for the mode/flag and
    the loader; everything else (campaign, A0 checkpoints, select report,
    champion checkpoints, frozen baseline) is resolved AFTER the claim.  This
    preflight mirrors those steps read-only so a broken champion checkpoint or
    a missing select report is found BEFORE the §9.5 one shot is burned.
    """
    from scripts.agent_ops.v543_s0_probe import (
        load_campaign_config,
        resolve_checkpoints,
    )

    started = time.perf_counter()
    loader = DevFrameLoader(
        eval_curves=int(args.eval_curves),
        frames_per_curve=int(args.frames_per_curve),
        with_basis=not bool(args.no_basis),
        with_strata=not bool(args.no_strata),
        pack_dir=args.pack_dir)
    mode = rollout.assert_mode_allowed(rollout.MODE_DEV, dev_onetime=True)
    rollout.require_dev_frame_loader(loader)
    ledger = rollout.DevAccessLedger(str(args.dev_ledger))
    already_used = ledger.already_used()
    _require(not already_used, "V543_DEV_T7_ALREADY_USED",
             f"the dev ledger already carries a dev use ({ledger.path}); T7 "
             "dev access is one-shot for the campaign (plan §9.5 :1038) — "
             "do NOT run --execute again")
    campaign = load_campaign_config(rollout.PROJECT_ROOT / str(args.config))
    seeds = ([int(args.seed)] if args.seed is not None
             else [int(campaign["source"]["seeds"][0])])
    seed = int(seeds[0])
    checkpoints = resolve_checkpoints(campaign, seeds)
    select = rollout.load_select_report(args.select_report)
    registry = rollout.candidate_registry()
    resolutions = {
        track: rollout.resolve_champion(
            select, track, seed=seed,
            campaign_root=rollout.PROJECT_ROOT / rollout.CAMPAIGN_ROOT)
        for track in (rollout.TRACK_MOVE, rollout.TRACK_PROCESS)}
    baseline_doc, baseline_sha = rollout.load_frozen_baseline(args.baseline)
    budget = {
        "eval_curves_requested": int(args.eval_curves),
        "frames_per_curve_requested": int(args.frames_per_curve),
        "frames_per_batch": int(rollout.FRAMES_PER_BATCH),
    }
    baseline_budget_matches = rollout._budget_matches_frozen(baseline_doc,
                                                             budget)
    data = data_contract()
    dataset = open_dataset(data)
    pack_check = loader.describe_pack_path(dataset)
    return {
        "schema": "pesn_v543_dev_t7_preflight_v1",
        "runner_schema": RUNNER_SCHEMA,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "mode": "preflight",
        "ok": True,
        "dev_pack_read": False,
        "dev_ledger_claim": False,
        "confirm_read": False,
        "new_QC_calls": 0,
        "checks": {
            "mode": mode,
            "loader_callable": True,
            "dev_ledger_already_used": bool(already_used),
            "campaign_loaded": True,
            "a0_checkpoints": {str(k): dict(v) for k, v in checkpoints.items()},
            "select_report": {"path": str(select["path"]),
                              "sha256": str(select["sha256"])},
            "champions": {str(k): v.describe()
                          for k, v in resolutions.items()},
            "candidate_registry_built": bool(registry is not None),
            "frozen_baseline": {"sha256": baseline_sha,
                                "budget_matches_this_run":
                                    baseline_budget_matches},
        },
        "budget": budget,
        "budget_note": (
            "eval_curves_requested={} (dev curve count) != the frozen "
            "baseline's 1024, so stage B0 will be reported as "
            "not_checked/different-frame-set instead of a false mismatch"
        ).format(int(args.eval_curves)) if baseline_budget_matches is False
        else "budget matches the frozen baseline document",
        "dev_pack_path_check": pack_check,
        "dev_ledger": {"path": str(ledger.path), "already_used": already_used},
        "wall_seconds": round(time.perf_counter() - started, 2),
        "note": ("read-only preflight: no ledger claim, no dev pack read; run "
                 "the classified --execute command only if this is ok"),
    }


# ---------------------------------------------------------------------------
# execute: the one-shot dev run (claim -> read -> rollout)
# ---------------------------------------------------------------------------
def run_execute(args: argparse.Namespace) -> dict[str, Any]:
    seeds = None
    if args.seed is not None:
        seeds = [int(args.seed)]
    loader = DevFrameLoader(
        eval_curves=int(args.eval_curves),
        frames_per_curve=int(args.frames_per_curve),
        with_basis=not bool(args.no_basis),
        with_strata=not bool(args.no_strata),
        pack_dir=args.pack_dir)
    namespace = execute_namespace(args, loader, seeds=seeds)
    _log("EXECUTE: the §9.5 one-shot dev claim will be appended inside "
         "v543_rollout.run BEFORE the dev pack is read")
    started = time.perf_counter()
    report = rollout.run(namespace)          # raises typed RolloutError/V543Error
    wall = float(time.perf_counter() - started)
    dev_access = report.get("dev_access") or {}
    _require(str(report.get("mode")) == rollout.MODE_DEV,
             "V543_DEV_T7_REPORT", "the rollout report is not a dev report")
    _require(bool(dev_access.get("dev_accessed")), "V543_DEV_T7_REPORT",
             "the rollout report records dev_accessed=false")
    _require(int(dev_access.get("ledger_entries_after", 0))
             > int(dev_access.get("ledger_entries_before", 0)),
             "V543_DEV_T7_REPORT",
             "the rollout report does not show the §9.5 claim being appended")
    report["dev_t7_runner"] = {
        "schema": RUNNER_SCHEMA,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "execute_wall_seconds": round(wall, 2),
        "loader": {
            "class": "DevFrameLoader",
            "eval_curves": int(args.eval_curves),
            "frames_per_curve": int(args.frames_per_curve),
            "with_basis": not bool(args.no_basis),
            "with_strata": not bool(args.no_strata),
            "features": "frozen dev feature pack (verify_files=True)",
            "frame_rule": ("v543_eval.inner_val_positions applied to the "
                           "frozen dev subset"),
            "frame_builder": "v543_eval.build_seed_frames (encoder+Bridge)",
            "pack_open_count": int(loader.pack_open_count),
        },
        "dev_ledger": str(rollout.DevAccessLedger(str(args.dev_ledger)).path),
        "select_report": str(args.select_report),
        "baseline": (None if args.baseline is None
                     else str(args.baseline)),
        "budget_note": (
            "eval_curves=589 declares the dev curve count so the §9.5 :1037 "
            "B0-vs-frozen-baseline comparison is reported as not_checked "
            "instead of comparing two different frame sets"),
        "confirm_read": False,
        "new_QC_calls": 0,
    }
    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="v543_dev_t7",
        description="V5.4.3 WP9 — dev one-shot T7 runner + dev frame loader "
                    "(default: pool-slice dry run; --execute: the one shot)")
    parser.add_argument("--dry-run-loader", action="store_true",
                        help="(default) validate the loader on a pool slice; "
                             "no dev pack, no ledger claim")
    parser.add_argument("--preflight", action="store_true",
                        help="read-only pre-claim checks (campaign, A0 "
                             "checkpoints, select report, champion "
                             "checkpoints, baseline, pack path); no claim")
    parser.add_argument("--execute", action="store_true",
                        help="the §9.5 ONE-SHOT dev execution "
                             "(claim -> dev pack read -> rollout)")
    parser.add_argument("--out", default=None,
                        help=f"default: {DEFAULT_DRYRUN_OUT} (dry run) / "
                             f"{DEFAULT_DEV_OUT} (execute)")
    parser.add_argument("--config", default=rollout.DEFAULT_CONFIG)
    parser.add_argument("--select-report",
                        default=f"{rollout.CAMPAIGN_ROOT}/"
                                f"{rollout.DEFAULT_SELECT_REL}")
    parser.add_argument("--baseline", default=None)
    parser.add_argument("--seeds", default=None,
                        help="comma/space separated; dev is one seed (first)")
    parser.add_argument("--seed", type=int, default=None,
                        help="single dev seed (default: campaign seed[0])")
    parser.add_argument("--dev-ledger", default=rollout.DEFAULT_DEV_LEDGER_REL)
    parser.add_argument("--eval-curves", type=int, default=DEV_CURVES_PLAN,
                        help=f"dev curve cap (default {DEV_CURVES_PLAN} = the "
                             "frozen dev subset; see the module docstring)")
    parser.add_argument("--frames-per-curve", type=int,
                        default=rollout.DEFAULT_FRAMES_PER_CURVE)
    parser.add_argument("--max-steps", type=int, default=rollout.DEFAULT_MAX_STEPS)
    parser.add_argument("--no-basis", action="store_true")
    parser.add_argument("--no-strata", action="store_true")
    parser.add_argument("--actor", default="v543_dev_t7")
    parser.add_argument("--pack-dir", default=None,
                        help="explicit dev pack dir (default: "
                             "<tables_dir>/dev_packs/<features_sha>/<positions_sha>)")
    parser.add_argument("--slice-curves", type=int, default=8,
                        help="dry run only: pool curves in the validation slice")
    parser.add_argument("--no-pack-path-check", action="store_true",
                        help="dry run only: skip the metadata-only dev-pack "
                             "path probe")
    parser.add_argument("--dryrun-out", default=DEFAULT_DRYRUN_OUT)
    args = parser.parse_args(argv)

    if int(bool(args.execute)) + int(bool(args.dry_run_loader)) \
            + int(bool(args.preflight)) > 1:
        _log("refused: --execute / --preflight / --dry-run-loader are "
             "mutually exclusive")
        return 2
    if args.seeds is not None:
        text = str(args.seeds).replace(",", " ")
        seeds = [int(token) for token in text.split()]
        if not seeds:
            _log("refused: --seeds is empty")
            return 2
        args.seed = int(seeds[0])
    if args.out is None:
        if args.execute:
            args.out = DEFAULT_DEV_OUT
        elif args.preflight:
            args.out = DEFAULT_PREFLIGHT_OUT
        else:
            args.out = args.dryrun_out

    try:
        if args.execute:
            report = run_execute(args)
            target = _resolve(args.out)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(report, indent=2, sort_keys=True,
                                         ensure_ascii=False),
                              encoding="utf-8")
            _log(f"dev rollout report -> {target}")
            _log("the §9.5 one-shot dev ledger now carries ONE dev entry; "
                 "a second dev visit is V543_ROLLOUT_DEV_ALREADY_USED")
            return 0
        if args.preflight:
            report = run_preflight(args)
            target = _resolve(args.out)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(report, indent=2, sort_keys=True,
                                         ensure_ascii=False),
                              encoding="utf-8")
            _log(f"preflight evidence -> {target}")
            _log("OK: pre-claim checks passed; no dev pack read, no claim")
            return 0
        report = run_dry_run(args)
        target = _resolve(args.out)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(report, indent=2, sort_keys=True,
                                     ensure_ascii=False),
                          encoding="utf-8")
        _log(f"dry-run evidence -> {target}")
        _log(f"OK: {report['validation']['n_frames']} frames / "
             f"{report['validation']['n_curves']} curves on the pool slice; "
             f"dev_pack_read=False; ledger untouched "
             f"({report['dev_ledger']['entries_before']} -> "
             f"{report['dev_ledger']['entries_after']})")
        return 0
    except rollout.RolloutError as error:
        print(f"[v543devt7] REFUSED {error}", file=sys.stderr)
        return 2
    except DevT7Error as error:
        print(f"[v543devt7] REFUSED {error}", file=sys.stderr)
        return 2
    except Exception as error:                       # typed, never silent
        print(f"[v543devt7] FAILED {type(error).__name__}: {error}",
              file=sys.stderr)
        return 1


if __name__ == "__main__":                          # pragma: no cover
    raise SystemExit(main())
