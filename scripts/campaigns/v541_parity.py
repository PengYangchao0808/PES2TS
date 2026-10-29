#!/usr/bin/env python3
"""V5.4.1 E0 strict-load / forward parity CLI (WP2, plan §5.3 / ADR-0032).

Compares a legacy V5.3-MP checkpoint (``KeyAtomV53MPModel``) against the
E0 build of ``V541Model`` and writes a JSON audit report to a NEW path
under the caller's control.  This CLI never writes into — and never
modifies — any historical run directory: it only READS checkpoints and
writes the single report file named by ``--out``.

Per checkpoint entry (plan §5.3 certificate items 1–5):

1. explicit state-dict audit — ``missing`` / ``unexpected`` /
   ``shape_mismatch`` / ``dtype_mismatch`` must ALL be empty, then
   ``strict=True`` load into BOTH models;
2. after load, every shared tensor of the two models is bit-for-bit
   equal (torch.equal) — the R3 common-core weights are the SAME tensors;
3. ``model.eval()``, one synthetic representative fixture batch, same
   device/dtype — compare all five outputs
   ``u / z / q / endpoint_logit / order_score`` bitwise (``torch.equal``)
   plus a numeric ``max_abs_diff`` (CPU convention: 1e-7 tolerance band,
   recorded; PASS is decided by the bitwise flag — a numeric-only pass is
   NOT a pass);
4. ``q == order_score`` is re-asserted on both models (shared-scalar
   semantics unchanged);
5. ``forward_move`` (``{"u","z"}``) is compared the same way.

The fixture is the synthetic SN2 fixture batch
(``scripts/agent_ops/feature_gate/v3_fixtures.make_v3_gate_batch``) —
fixture-level evidence, ``not_admissible_until_gate_pass`` for any real
data claim.  Running against the three historical C checkpoints is the
main agent's job; this tool itself performs no training, no QC and no
dev/confirm access.

Exit codes: 0 = all entries passed; 1 = at least one entry failed;
2 = refused (output path exists without ``--force``, or usage error).

CLI::

    python scripts/campaigns/v541_parity.py --out <NEW_JSON_PATH> \\
        [--checkpoint PATH ...] [--fixture-seed 17] [--n-graphs 2] [--force]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pes2ts_core.pesn.models.key_atom_heads_v3 import (  # noqa: E402
    V53MP_HEADS_VERSION,
    KeyAtomV53MPModel,
)
from pes2ts_core.pesn.models.v541 import (  # noqa: E402
    V541_SCHEMA_VERSION,
    V541Model,
)
from scripts.agent_ops.feature_gate.v3_fixtures import make_v3_gate_batch  # noqa: E402

#: Report schema id (new V541 audit artifact; nothing historical).
REPORT_SCHEMA: str = "pes2ts_v541_parity_report_v1"

#: The five contract §3.2 outputs compared per entry.
OUTPUT_KEYS: Sequence[str] = (
    "u", "z", "q", "endpoint_logit", "order_score")

#: CPU numeric tolerance band recorded next to the bitwise flag (§5.3 item 4).
CPU_ATOL: float = 1e-7

#: Historical C checkpoints (READ-ONLY inputs; main agent runs these).
DEFAULT_CHECKPOINTS: Sequence[str] = (
    "outputs/pesn/v53mp/s4_50k/v53mp_50k_C_s17/checkpoint_best.pt",
    "outputs/pesn/v53mp/s4_50k/v53mp_50k_C_s29/checkpoint_best.pt",
    "outputs/pesn/v53mp/s4_50k/v53mp_50k_C_s43/checkpoint_best.pt",
)

EXIT_PASS, EXIT_FAIL, EXIT_REFUSED = 0, 1, 2


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    """Tuples → lists, tensors → shapes, recursively (JSON-safe audit)."""
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, torch.Tensor):  # pragma: no cover - defensive
        return list(value.shape)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def compare_outputs(
        reference: Mapping[str, torch.Tensor],
        candidate: Mapping[str, torch.Tensor],
        keys: Sequence[str] = OUTPUT_KEYS,
) -> Dict[str, Any]:
    """Per-key bitwise + numeric comparison of two forward outputs."""
    report: Dict[str, Any] = {}
    all_bitwise = True
    for key in keys:
        a, b = reference[key], candidate[key]
        bitwise = bool(torch.equal(a, b))
        max_abs = float((a - b).abs().max().item()) if a.numel() else 0.0
        report[key] = {
            "bitwise": bitwise,
            "max_abs_diff": max_abs,
            "within_cpu_atol_1e7": bool(max_abs <= CPU_ATOL),
        }
        all_bitwise = all_bitwise and bitwise
    report["all_bitwise"] = bool(all_bitwise)
    return report


def run_entry(checkpoint: Path, batch: Mapping[str, torch.Tensor]) -> Dict[str, Any]:
    """Full audit + parity for ONE checkpoint (any failure → entry dict)."""
    entry: Dict[str, Any] = {
        "checkpoint": str(checkpoint),
        "passed": False,
        "error": None,
    }
    if not checkpoint.is_file():
        entry["error"] = "checkpoint file not found (read-only input)"
        return entry
    entry["sha256"] = sha256_file(checkpoint)

    try:
        payload = torch.load(str(checkpoint), map_location="cpu",
                             weights_only=True)
        if isinstance(payload, Mapping) and "model" in payload:
            state = payload["model"]
            entry["payload_schema"] = str(payload.get("schema", ""))
            entry["payload_update"] = _jsonable(payload.get("update"))
        else:
            state = payload
        if not isinstance(state, Mapping):
            raise TypeError(
                f"checkpoint 'model' entry is {type(state).__name__}, "
                "expected a state-dict mapping")
    except Exception as exc:  # noqa: BLE001 - reported as typed entry failure
        entry["error"] = f"checkpoint load failed: {type(exc).__name__}: {exc}"
        return entry

    try:
        # --- 1. explicit strict-load audit (plan §5.3 item 1) -------------
        # Audit FIRST so a defect is always reported as a structured
        # load_audit (missing/unexpected/shape/dtype), not as an opaque
        # legacy load_state_dict traceback.
        v541 = V541Model()                      # E0: no branch, no new params
        audit = v541.audit_legacy_state_dict(state)
        entry["load_audit"] = _jsonable(audit)
        if not audit["ok"]:
            entry["error"] = ("V541 strict-load audit not clean "
                              "(missing/unexpected/shape/dtype)")
            return entry

        legacy_model = KeyAtomV53MPModel()
        legacy_model.load_state_dict(state, strict=True)
        load_report = v541.load_legacy_state_dict(state, strict=True)
        entry["load_strict"] = True
        entry["load_missing_keys"] = list(load_report["missing_keys"])
        entry["load_unexpected_keys"] = list(load_report["unexpected_keys"])

        # --- 2. per-tensor identity after load (item 2) -------------------
        legacy_state = legacy_model.state_dict()
        v541_state = v541.state_dict()
        weights_equal = (
            set(legacy_state) == set(v541_state)
            and all(torch.equal(legacy_state[k], v541_state[k])
                    for k in legacy_state))
        entry["weights_equal_legacy_vs_v541"] = bool(weights_equal)
        entry["state_keys_identical"] = bool(
            set(legacy_state) == set(v541_state))

        # --- 3-5. eval-mode forward parity on the representative fixture --
        legacy_model.eval()
        v541.eval()
        with torch.no_grad():
            out_legacy = legacy_model(batch)
            out_v541 = v541(batch)
        entry["outputs"] = compare_outputs(out_legacy, out_v541)
        entry["q_equals_order_score"] = {
            "legacy": bool(torch.equal(out_legacy["q"],
                                       out_legacy["order_score"])),
            "v541": bool(torch.equal(out_v541["q"], out_v541["order_score"])),
        }
        with torch.no_grad():
            move_legacy = legacy_model.forward_move(batch)
            move_v541 = v541.forward_move(batch)
        entry["forward_move"] = compare_outputs(
            move_legacy, move_v541, keys=("u", "z"))

        entry["n_parameters"] = {
            "legacy": legacy_model.n_parameters,
            "v541_e0": v541.n_parameters,
        }
        entry["passed"] = bool(
            entry["state_keys_identical"]
            and entry["weights_equal_legacy_vs_v541"]
            and entry["outputs"]["all_bitwise"]
            and entry["forward_move"]["all_bitwise"]
            and all(entry["q_equals_order_score"].values()))
        if not entry["passed"]:
            entry["error"] = "parity criteria not met (see report fields)"
    except Exception as exc:  # noqa: BLE001 - typed entry failure, no raise
        entry["error"] = f"{type(exc).__name__}: {exc}"
    return entry


def run_parity(
        checkpoints: Sequence[Path],
        *,
        fixture_seed: int = 17,
        n_graphs: int = 2,
) -> Dict[str, Any]:
    """Build the representative fixture and audit every checkpoint."""
    batch = make_v3_gate_batch(
        n_graphs=int(n_graphs), jitter_seed=int(fixture_seed))
    entries = [run_entry(ckpt, batch) for ckpt in checkpoints]
    report: Dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model_versions": {
            "legacy": V53MP_HEADS_VERSION,
            "v541": V541_SCHEMA_VERSION,
        },
        "embedding_id": "E0",
        "fixture": {
            "source": ("scripts/agent_ops/feature_gate/v3_fixtures."
                       "make_v3_gate_batch"),
            "synthetic": True,
            "n_graphs": int(n_graphs),
            "jitter_seed": int(fixture_seed),
            "n_nodes": int(batch["pos"].shape[0]),
            "n_edges": int(batch["edge_index"].shape[1]),
            "note": ("fixture-level evidence only "
                     "(not_admissible_until_gate_pass)"),
        },
        "tolerance": {
            "require_bitwise": True,
            "cpu_atol_record_only": CPU_ATOL,
            "note": ("PASS is decided by torch.equal bitwise identity; the "
                     "1e-7 band is recorded, never used to widen a pass "
                     "(plan §5.3 item 4)"),
        },
        "entries": entries,
        "passed": bool(entries) and all(e["passed"] for e in entries),
    }
    return report


def _main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="V5.4.1 E0 strict-load/forward parity audit "
                    "(read-only over historical checkpoints; writes ONE "
                    "new JSON report)")
    parser.add_argument(
        "--out", required=True,
        help="NEW path for the JSON audit report (refused if it exists "
             "unless --force); must not be inside a historical run dir "
             "you do not own")
    parser.add_argument(
        "--checkpoint", action="append", default=None, metavar="PATH",
        help="checkpoint to audit (repeatable; default: the three "
             "v53mp_50k_C_s{17,29,43} checkpoint_best.pt files)")
    parser.add_argument("--fixture-seed", type=int, default=17)
    parser.add_argument("--n-graphs", type=int, default=2)
    parser.add_argument(
        "--force", action="store_true",
        help="overwrite an existing --out report (still never touches "
             "historical files)")
    args = parser.parse_args(argv)

    out_path = Path(args.out)
    if out_path.exists() and not args.force:
        print(f"REFUSED: output report already exists: {out_path} "
              f"(pass --force to overwrite)", file=sys.stderr)
        return EXIT_REFUSED
    if out_path.is_dir():
        print(f"REFUSED: --out is a directory: {out_path}", file=sys.stderr)
        return EXIT_REFUSED

    raw_checkpoints = args.checkpoint or list(DEFAULT_CHECKPOINTS)
    checkpoints = [
        Path(p) if Path(p).is_absolute() else PROJECT_ROOT / p
        for p in raw_checkpoints]

    report = run_parity(
        checkpoints,
        fixture_seed=args.fixture_seed,
        n_graphs=args.n_graphs)
    report["checkpoints"] = [str(c) for c in checkpoints]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(_jsonable(report), indent=2, sort_keys=True) + "\n",
        encoding="utf-8")

    for entry in report["entries"]:
        status = "PASS" if entry["passed"] else "FAIL"
        print(f"[{status}] {entry['checkpoint']}"
              + (f" — {entry['error']}" if entry.get("error") else ""))
    print(f"overall: {'PASS' if report['passed'] else 'FAIL'} → {out_path}")
    return EXIT_PASS if report["passed"] else EXIT_FAIL


def main(argv: Optional[Sequence[str]] = None) -> int:
    return _main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
