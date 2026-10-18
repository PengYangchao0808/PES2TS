"""Lane D — evaluate a trained arm against the frozen three protocols.

Reads a run directory (``run_record.json`` v2/v3 + a checkpoint weight),
rebuilds the model purely from the recorded settings, forwards the selected
protocol population, scores it with :func:`evaluate_protocol`, applies the
four-gate with :func:`apply_gates`, and writes one JSON per protocol to
``wp42_campaign/protocol_evals/{arm_id}_{protocol}.json``.

Discipline (contract §7 / §10): a model-load failure or an incompatible /
legacy checkpoint is a TYPED skip recorded into the output JSON — never a
silent shape truncation or a fabricated result.  Both heads are supported:
the legacy radial-tanh head (v4.1 ``DualTSPolicy``) and the unbounded
displacement head (Lane B ``dual_ts_displacement``; rebuilt exactly as the
V4.2 campaign trainer built it, loaded with ``strict=True``).  Arms using
``use_prior_skip`` / ``keypair_condition`` are honest typed skips (their
forward requires ``a_phys`` / the key-pair encoder wrapper this script does
not rebuild).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402

from pes2ts_core.pesn.offline.evaluation.dual_ts_protocols import (  # noqa: E402
    PROTOCOL_NAMES,
    apply_gates,
    evaluate_protocol,
    load_protocol_reference,
    verify_freeze_payload,
)

# Contract §7 default product paths.
CAMPAIGN_DIR = PROJECT_ROOT / "outputs" / "pesn" / "dual_ts_v2_20261015" / \
    "wp42_campaign"
DEFAULT_LEGACY_DIR = PROJECT_ROOT / "outputs" / "pesn" / "dual_ts_v2_20261015" \
    / "wp4_monitor" / "features_v41"
DEFAULT_NATIVE_DIR = CAMPAIGN_DIR / "native_monitor_features"
DEFAULT_FREEZE_PATH = CAMPAIGN_DIR / "protocol_freeze.json"
DEFAULT_OUT_DIR = CAMPAIGN_DIR / "protocol_evals"

CHECKPOINT_CANDIDATES = ("u20000", "u20000", "checkpoint_last.pt")


def _json_round(obj):
    """JSON-safe conversion for numpy scalars inside nested structures."""
    if isinstance(obj, dict):
        return {k: _json_round(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_round(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def _skip(arm_id: str, protocol: str, code: str, detail: str) -> dict:
    return {
        "status": "skipped",
        "code": code,
        "detail": detail,
        "arm_id": arm_id,
        "protocol": protocol,
    }


# ---------------------------------------------------------------------------
# Model construction + forward (defensive; Lane B head imported lazily)
# ---------------------------------------------------------------------------

def _load_checkpoint_path(run_dir: Path, settings: dict) -> Path:
    """Resolve the checkpoint weight file for a run directory.

    V4.2 campaign layout: ``checkpoints/{u20000,u10000,u5000}/{raw,ema}.pt``
    (each is a DIRECTORY per update step containing raw/ema weight files);
    the weight kind follows ``settings["ema_read"]`` (default "raw").
    Falls back to the legacy flat layout (``checkpoints/u20000`` etc. as
    plain files) for pre-campaign runs.  Directories never match
    (``is_file()``), so they are skipped rather than raising.
    """
    weight = "ema" if str(settings.get("ema_read", "raw")) == "ema" else "raw"
    for step in ("u20000", "u10000", "u5000"):
        p = run_dir / "checkpoints" / step / f"{weight}.pt"
        if p.is_file():
            return p
    # Legacy flat layout (pre-campaign checkpoints as plain files).
    for name in ("u20000", "u10000", "checkpoint_last.pt"):
        p = run_dir / "checkpoints" / name
        if p.is_file():
            return p
    p = run_dir / "checkpoint_last.pt"
    if p.is_file():
        return p
    raise FileNotFoundError(f"no checkpoint found under {run_dir}")


def _legacy_policy(settings: dict):
    """Rebuild the v4.1 radial-tanh policy from recorded settings."""
    from pes2ts_core.pesn.models.dual_egnn import DualTSModelConfig
    from pes2ts_core.pesn.models.dual_ts_policy import DualTSPolicy
    return DualTSPolicy(
        config=DualTSModelConfig(
            light=bool(settings.get("light", True)),
            signed_vector_messages=bool(
                settings.get("signed_vector_messages", False)),
        ),
        action_max=float(settings.get("action_max", 0.1)),
        sigma_range=(float(settings.get("sigma_min", 0.02)),
                     float(settings.get("sigma_max", 1.0))),
        signed_mu_gate=bool(settings.get("signed_mu_gate", False)),
    )


def _displacement_policy(settings: dict):
    """Rebuild the unbounded displacement policy (Lane B; imported lazily).

    Mirrors the V4.2 campaign trainer's ``_build_model`` exactly (same
    config objects, sizes, and ``build_displacement_policy`` call) so the
    state dict loads with ``strict=True`` — never a silently different
    architecture.
    """
    from pes2ts_core.pesn.models.dual_egnn import DualTSModelConfig
    from pes2ts_core.pesn.models.dual_ts_displacement import (
        DisplacementHeadConfig,
        build_displacement_policy,
    )
    edge_feat_dim = (7 if settings.get("features_version") == "v4" else 4)
    model_config = DualTSModelConfig(
        d_model=32, d_element=16, max_z=32,
        n_basis=32, cutoff=5.0, light=False,
        edge_feat_dim=edge_feat_dim)
    head_config = DisplacementHeadConfig(
        readout_hidden=int(settings.get("readout_hidden", 64)),
        use_prior_skip=bool(settings.get("use_prior_skip", False)))
    return build_displacement_policy(
        model_config, head_config,
        encoder_kind=str(settings.get("encoder", "egnn")),
        vector_channels=int(settings.get("vector_channels", 16)),
        contact_edges=bool(settings.get("contact_edges", False)),
        hidden=int(settings.get("hidden", 64)),
        n_layers=int(settings.get("n_layers", 3)),
        edge_feat_dim=edge_feat_dim)


def _head_kind(settings: dict) -> str:
    hv = str(settings.get("head_version")
             or settings.get("action_version")
             or settings.get("density_target_version", ""))
    if "unbounded_displacement" in hv or "displacement" in hv:
        return "displacement"
    if "radial_tanh" in hv or settings.get("recipe") in (
            "bc_nll", "bc_a", "bc_d"):
        return "legacy"
    # V4.2 campaign run_record: explicit ``head`` field ("displacement" |
    # "bc_nll_legacy").
    head = str(settings.get("head", ""))
    if "displacement" in head:
        return "displacement"
    if head in ("bc_nll_legacy",) or "legacy" in head or "bc_nll" in head:
        return "legacy"
    return "unknown"


def _feature_dirs(features_version: str, legacy_dir: Path,
                  native_dir: Path) -> tuple[Path, Path]:
    """Resolve the (legacy, native) feature dirs matching the model version.

    v3 models (edge_feat_dim=4) consume the monitor shards as-is; v4 models
    (edge_feat_dim=7) consume the pre-augmented ``features_v4/legacy_monitor``
    and ``features_v4/native_monitor`` shards (same states, 7-channel edge
    blocks; built by ``scripts/build_v4_edge_features.py``).
    """
    if str(features_version) == "v4":
        return (CAMPAIGN_DIR / "features_v4" / "legacy_monitor",
                CAMPAIGN_DIR / "features_v4" / "native_monitor")
    return Path(legacy_dir), Path(native_dir)


def _attach_features(states: list[dict], features_dir: Path) -> None:
    """Merge shard model-input arrays onto raw reference records by state_id.

    The frozen protocol reference transfers carry only raw geometry
    (``atomic_numbers``/``x_angstrom``); the model consumes the shard arrays
    ``node_scalars``/``g_scaled``/``atom_feats_*``/``edge_feats_*``/
    ``edge_index`` AS-IS (exactly what the campaign trainer consumed for the
    same states).  Mutates each state dict in place; raises a typed
    ``RuntimeError`` on a missing state or an edge-block width mismatch
    (never a silent slice/pad).
    """
    manifest = json.loads((features_dir / "features_manifest.json").read_text())
    wanted = {str(s["state_id"]) for s in states}
    found: dict[str, dict] = {}
    rows = sorted(manifest.get("per_reaction", []),
                  key=lambda r: str(r.get("reaction_group_id")))
    for row in rows:
        shard = np.load(features_dir / row["npz_file"])
        z = np.asarray(shard["atomic_numbers"], dtype=np.int64)
        node_scalars = shard["node_scalars"]
        x = shard["x_angstrom"]
        g = shard["g_scaled"]
        edge_index = shard["edge_index"]
        atom_a, atom_b = shard["atom_feats_a"], shard["atom_feats_b"]
        edge_a, edge_b = shard["edge_feats_a"], shard["edge_feats_b"]
        # v3: shared (E, 4) block; v4: per-state (S, E, 7) blocks.
        edge_a_per_state = edge_a.ndim == 3
        edge_b_per_state = edge_b.ndim == 3
        for r, sid in enumerate(shard["state_id"]):
            sid = str(sid)
            if sid not in wanted or sid in found:
                continue
            ea = edge_a[r] if edge_a_per_state else edge_a
            eb = edge_b[r] if edge_b_per_state else edge_b
            if int(np.asarray(ea).shape[-1]) != \
                    int(np.asarray(eb).shape[-1]):
                raise RuntimeError(
                    f"SKIP_FEATURES_MALFORMED: state {sid} edge block "
                    f"widths {np.asarray(ea).shape[-1]} vs "
                    f"{np.asarray(eb).shape[-1]}")
            found[sid] = {
                "atomic_numbers": z,
                "node_scalars": node_scalars[r],
                "x_angstrom": x[r],
                "g_scaled": g[r],
                "atom_feats_a": atom_a,
                "atom_feats_b": atom_b,
                "edge_feats_a": ea,
                "edge_feats_b": eb,
                "edge_index": edge_index,
            }
    missing = sorted(wanted - set(found))
    if missing:
        raise RuntimeError(
            f"SKIP_FEATURES_MISSING: {len(missing)} protocol states absent "
            f"from {features_dir} (e.g. {missing[:3]})")
    for s in states:
        s.update(found[str(s["state_id"])])


def forward_actions(run_dir: Path, states: list[dict]) -> dict:
    """Return ``{state_id -> (N,3) ndarray}`` deterministic predicted actions.

    Raises a typed ``RuntimeError`` with a machine-readable prefix on any
    model-load / incompatibility issue (the caller records it as a skip).
    """
    record = json.loads((run_dir / "run_record.json").read_text())
    settings = record.get("settings", {})
    kind = _head_kind(settings)
    ckpt = _load_checkpoint_path(run_dir, settings)

    # Honest typed skips: never silently rebuild a different model than
    # the one that was trained.
    if bool(settings.get("use_prior_skip", False)):
        raise RuntimeError(
            "SKIP_PRIOR_SKIP_UNSUPPORTED: forward requires a_phys")
    if bool(settings.get("keypair_condition", False)):
        raise RuntimeError(
            "SKIP_KEYPAIR_WRAPPER_UNSUPPORTED: encoder wrapper not rebuilt")

    from pes2ts_core.pesn.models import require_torch
    torch = require_torch()
    device = "cpu"

    # v3 monitor shards carry 4-channel edge blocks, v4 models 7-channel;
    # a mismatch is a typed skip (never a silent slice).
    expect_edge = 7 if str(settings.get("features_version")) == "v4" else 4
    for s in states:
        width = int(np.asarray(s["edge_feats_a"]).shape[-1])
        if width != expect_edge:
            raise RuntimeError(
                f"SKIP_FEATURE_WIDTH_MISMATCH: state {s['state_id']} edge "
                f"block width {width} != model edge_feat_dim {expect_edge} "
                f"(features_version={settings.get('features_version')})")

    if kind == "legacy":
        policy = _legacy_policy(settings)
    elif kind == "displacement":
        policy = _displacement_policy(settings)
    else:
        raise RuntimeError(
            f"SKIP_UNKNOWN_HEAD: cannot determine model head from settings "
            f"{list(settings)}")

    policy.to(torch.float32).to(device)
    payload = torch.load(ckpt, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "model_state" not in payload:
        raise RuntimeError(
            "SKIP_INCOMPATIBLE_CHECKPOINT: checkpoint lacks model_state "
            "(legacy v1 checkpoint or non-BC payload); no silent shape "
            "truncation")
    try:
        if kind == "displacement":
            # DualTSDisplacementPolicy is a plain nn.Module: standard
            # strict load (no load_state_dicts wrapper).
            policy.load_state_dict(payload["model_state"], strict=True)
        else:
            policy.load_state_dicts(payload["model_state"])
    except Exception as exc:  # noqa: BLE001 — record the real reason
        raise RuntimeError(
            f"SKIP_INCOMPATIBLE_CHECKPOINT: state_dict load failed: {exc!r}")

    from pes2ts_core.pesn.models.dual_egnn import collate_dual_states

    if kind == "legacy":
        policy.train_mode(False)
    else:
        policy.eval()
    out: dict[str, np.ndarray] = {}
    with torch.no_grad():
        for i in range(0, len(states), 256):
            chunk = list(states[i:i + 256])
            n_atoms = [int(np.asarray(s["atomic_numbers"]).size) for s in chunk]
            inputs = collate_dual_states(torch, chunk)
            for k, v in inputs.items():
                if torch.is_tensor(v) and v.dtype in (torch.float64,
                                                      torch.float32):
                    inputs[k] = v.to(dtype=torch.float32)
            model_out = policy(inputs)
            if kind == "legacy":
                mu = model_out["mu"]
                mu_norm = mu.norm(dim=-1, keepdim=True)
                a = (float(settings.get("action_max", 0.1)) * torch.tanh(mu_norm)
                     * mu / mu_norm.clamp(min=1e-12))
            else:
                a = model_out["a_hat"]
            a_np = a.detach().cpu().numpy().astype(np.float64)
            offset = 0
            for s, n in zip(chunk, n_atoms):
                out[str(s["state_id"])] = a_np[offset:offset + n]
                offset += n
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", required=True,
                    help="run directory with run_record.json + checkpoints/")
    ap.add_argument("--protocol", required=True,
                    choices=("legacy", "native", "common", "all"))
    ap.add_argument("--legacy-dir", default=str(DEFAULT_LEGACY_DIR))
    ap.add_argument("--native-dir", default=str(DEFAULT_NATIVE_DIR))
    ap.add_argument("--freeze-path", default=str(DEFAULT_FREEZE_PATH))
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    record = json.loads((run_dir / "run_record.json").read_text())
    arm_id = str(record.get("arm_id")
                 or record.get("run_id")
                 or run_dir.name)
    # Include the seed so multi-seed stages (V4.7 FINAL) never overwrite each
    # other's protocol evidence; single-seed arms gain the same traceability.
    seed = (record.get("seed")
            or record.get("settings", {}).get("seed"))
    if seed is not None:
        arm_id = f"{arm_id}_seed{seed}"

    protocols = PROTOCOL_NAMES if args.protocol == "all" else (
        [f"{args.protocol}_full"] if args.protocol in ("legacy", "native")
        else ["common_unchanged"])

    # -- reference + freeze identity ----------------------------------------
    freeze_path = Path(args.freeze_path)
    if not freeze_path.is_file():
        print(f"REFUSING: freeze missing at {freeze_path} — run the protocol "
              f"freeze before any evaluation (contract §7)", file=sys.stderr)
        return 2
    disk_freeze = json.loads(freeze_path.read_text())
    if not verify_freeze_payload(disk_freeze)["ok"]:
        print("REFUSING: protocol freeze digest mismatch", file=sys.stderr)
        return 2
    frozen = load_protocol_reference(Path(args.legacy_dir),
                                     Path(args.native_dir))
    if frozen["payload_sha256"] != disk_freeze["payload_sha256"]:
        print("REFUSING: live reference digest != frozen digest "
              "(data changed after freeze)", file=sys.stderr)
        return 2

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # -- forward once over the union of the requested populations -----------
    needed_states: dict[str, list] = {}
    for protocol in protocols:
        for t in frozen["_reference"][protocol]:
            needed_states.setdefault(t["state_id"], t)

    preds: dict[str, np.ndarray] = {}
    load_error = None
    try:
        by_id = {str(t["state_id"]): dict(t)
                 for t in needed_states.values()}
        # The frozen transfers carry raw geometry only; merge the model
        # inputs from the SAME frozen feature shards per source library
        # (legacy_full/common_unchanged from the legacy library — the
        # common records ARE legacy records; native_full from the native
        # one; common states are byte-identical across the two libraries).
        feat_legacy, feat_native = _feature_dirs(
            record.get("settings", {}).get("features_version"),
            Path(args.legacy_dir), Path(args.native_dir))
        legacy_src: set = set()
        native_src: set = set()
        if "legacy_full" in protocols:
            legacy_src |= {str(t["state_id"])
                           for t in frozen["_reference"]["legacy_full"]}
        if "common_unchanged" in protocols:
            legacy_src |= {str(t["state_id"])
                           for t in frozen["_reference"]["common_unchanged"]}
        if "native_full" in protocols:
            native_src |= {str(t["state_id"])
                           for t in frozen["_reference"]["native_full"]}
        legacy_src &= by_id.keys()
        native_src &= by_id.keys()
        native_only = native_src - legacy_src
        if legacy_src:
            _attach_features([by_id[s] for s in sorted(legacy_src)],
                             feat_legacy)
        if native_only:
            _attach_features([by_id[s] for s in sorted(native_only)],
                             feat_native)
        preds = forward_actions(run_dir, list(by_id.values()))
    except Exception as exc:  # noqa: BLE001 — typed skip, recorded below
        load_error = exc

    for protocol in protocols:
        out_path = out_dir / f"{arm_id}_{protocol}.json"
        if load_error is not None:
            out_path.write_text(json.dumps(
                _skip(arm_id, protocol, "MODEL_LOAD_FAILED",
                      str(load_error)), indent=2) + "\n")
            continue
        try:
            metrics = evaluate_protocol(preds, protocol, frozen)
            gates = apply_gates(metrics)
            payload = {
                "arm_id": arm_id,
                "protocol": protocol,
                "freeze_sha256": disk_freeze["payload_sha256"],
                "status": "ok",
                "metrics": metrics,
                "gates": gates,
            }
        except Exception as exc:  # noqa: BLE001 — record, never fabricate
            payload = _skip(arm_id, protocol, "EVAL_FAILED", str(exc))
        out_path.write_text(json.dumps(_json_round(payload), indent=2) + "\n")
        print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
