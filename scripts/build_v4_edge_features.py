"""V4.2 E1 feature materialization (Lane C; contract §5 E1 / §8).

Builds the ``pesn_dual_ts_features_v4`` (7-channel gradient-augmented edge)
shards from three sources, one output directory each:

* ``native_features/``        -> ``features_v4/native_fit/``
* ``native_monitor_features/``-> ``features_v4/native_monitor/``
* ``wp4_monitor/features_v41/``-> ``features_v4/legacy_monitor/``

For every npz, ``edge_feats_a`` and ``edge_feats_b`` (4ch shared) are
augmented PER STATE via Lane B's ``augment_edge_features_gradient`` using that
state's ``x_angstrom`` and ``g_scaled`` (``ghat = g/‖g‖`` is scale-invariant,
so the frozen ``s_g`` does not enter).  Every other array is copied verbatim.
Each output directory gets a ``features_manifest.json`` (per-reaction rows +
parent hash + zero-norm-mask statistics) and a ``features_spec.json`` written
from Lane B's ``build_features_spec_v4`` payload.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pes2ts_core.pesn.features.dual_ts_graph_v4 import (  # noqa: E402
    build_features_spec_v4,
    augment_edge_features_gradient,
)
from pes2ts_core.pesn.features.dual_ts_spec import save_spec_payload  # noqa: E402

WP42 = PROJECT_ROOT / "outputs/pesn/dual_ts_v2_20261015/wp42_campaign"
OUT_ROOT = WP42 / "features_v4"

#: (output subdir, source features dir)
INPUTS: tuple[tuple[str, Path], ...] = (
    ("native_fit", WP42 / "native_features"),
    ("native_monitor", WP42 / "native_monitor_features"),
    ("legacy_monitor", PROJECT_ROOT
     / "outputs/pesn/dual_ts_v2_20261015/wp4_monitor/features_v41"),
)

#: The v3 spec is the single parent for every v4 build (native/manifests
#: reference it via features_spec_source).
V3_SPEC_PATH = PROJECT_ROOT / (
    "outputs/pesn/dual_ts_v2_20261015/wp3_features_v41/features_spec.json")

MANIFEST_SCHEMA = "pesn_dual_ts_features_v4_manifest_v1"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _augment_shard(src_npz: np.lib.npyio.NpzFile):
    """Return (out_dict, zero_norm_counts) for one shard.

    ``out_dict`` is the full output-array mapping (edge_feats_a/b replaced by
    the per-state 7-channel augmented blocks; every other array copied).
    """
    x = np.asarray(src_npz["x_angstrom"], dtype=np.float64)          # (S, N, 3)
    g = np.asarray(src_npz["g_scaled"], dtype=np.float64)            # (S, N, 3)
    edge_index = np.asarray(src_npz["edge_index"], dtype=np.int64)   # (2, E)
    edge_a = np.asarray(src_npz["edge_feats_a"], dtype=np.float64)   # (E, 4)
    edge_b = np.asarray(src_npz["edge_feats_b"], dtype=np.float64)   # (E, 4)

    n_states = int(x.shape[0])
    aug_a = np.empty((n_states,) + edge_a.shape[:-1] + (7,), dtype=np.float64)
    aug_b = np.empty((n_states,) + edge_b.shape[:-1] + (7,), dtype=np.float64)
    zero_norm_total = 0
    n_atoms_total = 0
    for r in range(n_states):
        a7, mask_a = augment_edge_features_gradient(
            edge_a, x[r], g[r], edge_index)
        b7, mask_b = augment_edge_features_gradient(
            edge_b, x[r], g[r], edge_index)
        aug_a[r] = a7
        aug_b[r] = b7
        zero_norm_total += int(mask_a.sum()) + int(mask_b.sum())
        n_atoms_total += 2 * int(mask_a.size)

    out = {}
    for key in src_npz.keys():
        out[key] = src_npz[key]
    out["edge_feats_a"] = aug_a
    out["edge_feats_b"] = aug_b
    return out, {"zero_norm_atoms": zero_norm_total, "n_atoms": n_atoms_total}


def build_one(name: str, src_dir: Path, out_dir: Path, *, v3_spec: dict) -> dict:
    src_manifest = json.loads((src_dir / "features_manifest.json").read_text())
    rows = sorted(src_manifest["per_reaction"],
                  key=lambda r: str(r["reaction_group_id"]))
    out_dir.mkdir(parents=True, exist_ok=True)

    zero_total = 0
    n_atoms_total = 0
    per_reaction = []
    for row in rows:
        npz_file = str(row["npz_file"])
        src_npz = np.load(src_dir / npz_file)
        arrays, stats = _augment_shard(src_npz)
        np.savez(out_dir / npz_file, **arrays)
        zero_total += stats["zero_norm_atoms"]
        n_atoms_total += stats["n_atoms"]
        per_reaction.append({
            "reaction_group_id": str(row["reaction_group_id"]),
            "npz_file": npz_file,
            "npz_sha256": sha256_file(out_dir / npz_file),
        })

    s_g = float(v3_spec["frozen_scales"]["s_g"])
    s_delta_e = float(v3_spec["frozen_scales"]["s_delta_e"])
    sd = v3_spec["support_domain"]
    _, spec_payload = build_features_spec_v4(
        s_g=s_g, s_delta_e=s_delta_e,
        element_vocab=list(sd["elements"]),
        population=dict(sd["population"]),
        n_atoms_range=tuple(sd["n_atoms_range"]))
    spec_digest = save_spec_payload(spec_payload, out_dir / "features_spec.json")

    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "tool_version": "pes2ts_dual_ts_features_v4_tool_v1",
        "name": name,
        "source_dir": str(src_dir),
        "source_manifest_sha256": sha256_file(src_dir / "features_manifest.json"),
        "parent_spec_digest": str(spec_payload.get("parent_spec_digest", "")),
        "v4_spec_digest": spec_digest,
        "zero_norm": {
            "zero_norm_atoms": int(zero_total),
            "n_atoms_scanned": int(n_atoms_total),
            "zero_norm_fraction": (
                float(zero_total / n_atoms_total) if n_atoms_total else 0.0),
        },
        "per_reaction": per_reaction,
    }
    (out_dir / "features_manifest.json").write_text(
        json.dumps(manifest, indent=1, sort_keys=False, ensure_ascii=False),
        encoding="utf-8")
    return manifest


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-root", type=Path, default=OUT_ROOT)
    args = ap.parse_args(argv)

    v3_doc = json.loads(V3_SPEC_PATH.read_text())
    v3_spec = v3_doc["spec"]

    for name, src_dir in INPUTS:
        out_dir = args.out_root / name
        manifest = build_one(name, src_dir, out_dir, v3_spec=v3_spec)
        zn = manifest["zero_norm"]
        print(f"[v4] {name}: {len(manifest['per_reaction'])} npz | "
              f"zero-norm atoms {zn['zero_norm_atoms']}/{zn['n_atoms_scanned']} "
              f"({zn['zero_norm_fraction']:.4%}) | "
              f"spec_digest {manifest['v4_spec_digest'][:12]}…", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["INPUTS", "OUT_ROOT", "build_one", "main"]
