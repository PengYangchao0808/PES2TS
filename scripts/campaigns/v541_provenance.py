"""V5.4.1 WP4c — provenance SHAs for the unified campaign runner.

Scope (plan §12.2 / §17 「formal-run code/common-protocol/runplan SHA
冻结」): every digest a V5.4.1 runplan line, checkpoint identity or resume
guard needs, computed WITHOUT touching any legacy V5.3-MP / R3 file.

What lives here
===============

``source_manifest`` / ``code_sha256``
    The worktree is DIRTY, so a git commit alone cannot bind the code.
    A deterministic **source-tree fingerprint** (per-file SHA256 over the
    config-declared ``identity_contract.code.includes`` set) is the
    authoritative code identity; ``v541_identity.git_state`` (no
    subprocess) still records the commit next to it.  The fingerprint is
    folded into the common-protocol SHA, so any source drift changes the
    formal code/common SHA and triggers a typed refusal.

``environment_sha256``
    Explicit digest of :func:`v541_identity.env_state` (Python / Torch /
    NumPy / CUDA / platform).

``common_protocol_sha256``
    The NUMERIC protocol only: schedule, λ, batch, sampler binding,
    selection, τ, split, R3 init and the data/teacher/split/label SHAs.
    ``embedding_id`` / ``feature_recipe`` / ``seed`` / ``run_id`` are
    excluded, so all candidates share ONE common SHA while
    ``embedding_spec_sha256`` / ``feature_recipe_sha256`` are computed
    PER CANDIDATE RECIPE (never one shared value).

``label_spec_shas``
    Reuses the old runner's ``_label_spec_sha256`` (endpoint-label spec)
    read-only and wraps it with the V5.4.1 teacher/process contract.

Typed refusals: :class:`V541ProvenanceError` (``code`` + ``detail``).
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pes2ts_core.pesn.models.v541.embedding_registry import (  # noqa: E402
    embedding_spec,
)
from pes2ts_core.pesn.offline.training.v541_identity import (  # noqa: E402
    env_state,
    git_state,
    sha256_json,
)

__all__ = [
    "COMMON_SCHEMA",
    "EMBEDDING_SPEC_SCHEMA",
    "FEATURE_RECIPE_SCHEMA",
    "LABEL_SPEC_SCHEMA",
    "SOURCE_MANIFEST_SCHEMA",
    "V541ProvenanceError",
    "artifact_shas",
    "code_sha256",
    "common_protocol_payload",
    "common_protocol_sha256",
    "config_file_sha256",
    "embedding_spec_sha256",
    "environment_sha256",
    "feature_recipe_sha256",
    "git_commit",
    "label_spec_shas",
    "r3_init_sha256",
    "sha256_file",
    "source_manifest",
]

SOURCE_MANIFEST_SCHEMA = "pes2ts_v541_source_manifest_v1"
COMMON_SCHEMA = "pes2ts_v541_common_protocol_v1"
EMBEDDING_SPEC_SCHEMA = "pes2ts_v541_embedding_spec_v1"
FEATURE_RECIPE_SCHEMA = "pes2ts_v541_feature_recipe_v1"
LABEL_SPEC_SCHEMA = "pes2ts_v541_label_spec_v1"

#: V5.3-MP config that still freezes the endpoint-label spec (read-only).
DEFAULT_V53MP_CONFIG = "config/campaigns/pes2ts_v53mp_20261021.yaml"


class V541ProvenanceError(RuntimeError):
    """Typed refusal of the V541 provenance machinery."""

    def __init__(self, code: str, detail: str = "") -> None:
        super().__init__(f"{code}: {detail}" if detail else code)
        self.code = str(code)
        self.detail = str(detail)


def sha256_file(path: Any) -> str:
    """Streaming sha256 of a file (identical digest to the legacy helper)."""
    digest = hashlib.sha256()
    try:
        with Path(path).open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
    except OSError as exc:
        raise V541ProvenanceError(
            "V541_SHA_UNREADABLE", "{}: {}".format(path, exc)) from exc
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# code identity (source-tree fingerprint; git commit is informational)
# ---------------------------------------------------------------------------
def _walk_sources(code_contract: Mapping[str, Any],
                  root: Path) -> Dict[str, str]:
    """``{relative posix path: sha256}`` over the declared source set."""
    includes = [str(x) for x in (code_contract.get("includes") or ())]
    suffixes = tuple(str(x) for x in (code_contract.get("suffixes") or ()))
    excludes = {str(x) for x in (code_contract.get("excludes") or ())}
    if not includes or not suffixes:
        raise V541ProvenanceError(
            "V541_SOURCE_SPEC", "identity_contract.code includes/suffixes empty")

    files: Dict[str, str] = {}

    def _excluded(part: str) -> bool:
        return part in excludes

    def _add(path: Path) -> None:
        if path.suffix not in suffixes:
            return
        rel = path.relative_to(root).as_posix()
        if any(_excluded(part) for part in Path(rel).parts):
            return
        files[rel] = sha256_file(path)

    for entry in includes:
        target = root / entry
        if not target.exists():
            raise V541ProvenanceError(
                "V541_SOURCE_MISSING",
                "identity_contract.code.includes entry missing: {}".format(entry),
            )
        if target.is_file():
            _add(target)
            continue
        for path in sorted(target.rglob("*")):
            if not path.is_file():
                continue
            if any(_excluded(part) for part in path.relative_to(root).parts):
                continue
            _add(path)
    if not files:
        raise V541ProvenanceError(
            "V541_SOURCE_EMPTY", "source fingerprint matched zero files")
    return dict(sorted(files.items()))


def source_manifest(code_contract: Optional[Mapping[str, Any]] = None, *,
                    root: Any = None) -> Dict[str, Any]:
    """Per-file SHA manifest + the code fingerprint (self-excluding digests).

    ``code_sha256`` hashes ONLY the file map (that is the code identity);
    ``source_manifest_sha256`` hashes the whole document (spec + map), so a
    changed include/exclude set is also visible.  Both are recomputed from
    disk on every plan/gate/run — a drift is a typed refusal, never a
    silent pass.
    """
    if code_contract is None:
        code_contract = _default_code_contract()
    base = Path(root) if root is not None else PROJECT_ROOT
    files = _walk_sources(code_contract, base)
    doc: Dict[str, Any] = {
        "schema": SOURCE_MANIFEST_SCHEMA,
        "method": str(code_contract.get("method") or "source_tree_fingerprint"),
        "includes": [str(x) for x in (code_contract.get("includes") or ())],
        "suffixes": [str(x) for x in (code_contract.get("suffixes") or ())],
        "excludes": sorted(str(x) for x in (code_contract.get("excludes") or ())),
        "files": files,
        "n_files": len(files),
    }
    doc["code_sha256"] = sha256_json(files)
    doc["source_manifest_sha256"] = sha256_json(doc)
    return doc


def _default_code_contract() -> Mapping[str, Any]:
    from scripts.agent_ops.v541_config_schema import load_config

    cfg = load_config()
    contract = (cfg.get("identity_contract") or {}).get("code")
    if not isinstance(contract, Mapping):
        raise V541ProvenanceError(
            "V541_SOURCE_SPEC", "config identity_contract.code missing")
    return contract


def code_sha256(code_contract: Optional[Mapping[str, Any]] = None, *,
                root: Any = None) -> str:
    """The authoritative code identity (source-tree fingerprint)."""
    return str(source_manifest(code_contract, root=root)["code_sha256"])


def git_commit() -> Optional[str]:
    """Subprocess-free ``.git/HEAD`` commit (informational, never sole id)."""
    state = git_state()
    commit = state.get("commit")
    return str(commit) if commit else None


def environment_sha256() -> str:
    """Explicit environment digest (Python/Torch/NumPy/CUDA/platform)."""
    return sha256_json(env_state())


# ---------------------------------------------------------------------------
# common protocol (numeric) vs per-candidate recipe SHAs
# ---------------------------------------------------------------------------
def common_protocol_payload(
    cfg: Mapping[str, Any],
    *,
    code_sha: str,
    data_manifest_sha256: Optional[str],
    teacher_manifest_sha256: Optional[str],
    split_sha256: Optional[str],
    label_spec_sha256: Optional[str],
) -> Dict[str, Any]:
    """Build the numeric-protocol payload declared by the config.

    Every key in ``identity_contract.common_protocol.includes`` must be
    resolvable; an unknown include is a typed refusal (the config and the
    code must not drift apart).  Per-candidate / per-run fields named in
    ``excludes`` are never part of the payload.
    """
    identity = cfg.get("identity_contract") or {}
    common = identity.get("common_protocol") or {}
    includes = [str(x) for x in (common.get("includes") or ())]
    excludes = {str(x) for x in (common.get("excludes") or ())}
    tc = cfg.get("training_contract") or {}
    split = cfg.get("split_contract") or {}
    sources: Dict[str, Any] = {
        "schema": cfg.get("schema"),
        "campaign_id": cfg.get("campaign_id"),
        "objective": cfg.get("objective"),
        "schedule": tc.get("schedule"),
        "lambdas": tc.get("lambdas"),
        "batch": tc.get("batch"),
        "sampler_binding": tc.get("sampler_binding"),
        "selection": tc.get("selection"),
        "tau_end": tc.get("tau_end"),
        "gradient_monitoring": tc.get("gradient_monitoring"),
        "inner_val_max_samples": split.get("inner_val_max_samples"),
        "split_contract": split,
        "backbone_init": tc.get("backbone_init"),
        "data_manifest_sha256": data_manifest_sha256,
        "teacher_manifest_sha256": teacher_manifest_sha256,
        "split_sha256": split_sha256,
        "label_spec_sha256": label_spec_sha256,
        "code_sha256": code_sha,
    }
    unknown = sorted(set(includes) - set(sources))
    if unknown:
        raise V541ProvenanceError(
            "V541_COMMON_SPEC",
            "identity_contract.common_protocol.includes names unknown "
            "field(s) {}".format(unknown),
        )
    overlap = excludes & set(includes)
    if overlap:
        raise V541ProvenanceError(
            "V541_COMMON_SPEC",
            "common_protocol includes an excluded field(s) {}".format(
                sorted(overlap)),
        )
    return {
        "schema": COMMON_SCHEMA,
        "protocol": {key: sources[key] for key in includes},
        "excludes": sorted(excludes),
    }


def common_protocol_sha256(
    cfg: Mapping[str, Any],
    *,
    code_sha: str,
    data_manifest_sha256: Optional[str] = None,
    teacher_manifest_sha256: Optional[str] = None,
    split_sha256: Optional[str] = None,
    label_spec_sha256: Optional[str] = None,
) -> str:
    """Numeric-protocol SHA — shared by every candidate on this protocol."""
    payload = common_protocol_payload(
        cfg, code_sha=code_sha,
        data_manifest_sha256=data_manifest_sha256,
        teacher_manifest_sha256=teacher_manifest_sha256,
        split_sha256=split_sha256,
        label_spec_sha256=label_spec_sha256)
    return sha256_json(payload)


def _registry_row(cfg: Mapping[str, Any], embedding_id: str) -> Mapping[str, Any]:
    contract = cfg.get("embedding_contract") or {}
    registry = contract.get("embedding_registry") or {}
    row = registry.get(str(embedding_id))
    if not isinstance(row, Mapping):
        raise V541ProvenanceError(
            "V541_EMBEDDING_UNKNOWN",
            "embedding_registry has no row for {!r}".format(embedding_id),
        )
    return row


def embedding_spec_sha256(cfg: Mapping[str, Any], embedding_id: str) -> str:
    """Per-candidate embedding identity: config row AND the WP2 code spec.

    Both sides are hashed, so a config-only or code-only change moves the
    SHA (plan §6.1 / §17 embedding_spec freeze).  ``implemented_in_code``
    is DYNAMIC (:func:`is_embedding_implemented` imports the BUILT-IN WP3
    module on demand) — never the frozen ``EmbeddingSpec.implemented``
    flag, which is WP2-era metadata and would keep claiming E1–E4 are
    unimplemented after they landed.
    """
    embedding_id = str(embedding_id)
    row = dict(_registry_row(cfg, embedding_id))
    try:
        code_spec = embedding_spec(embedding_id)
        spec_dict = dict(vars(code_spec))
    except Exception as exc:  # noqa: BLE001 - unknown id -> typed refusal
        raise V541ProvenanceError(
            "V541_EMBEDDING_UNKNOWN",
            "{}: {}".format(embedding_id, exc),
        ) from exc
    from pes2ts_core.pesn.models.v541.embedding_registry import (
        V541InterfaceError,
        is_embedding_implemented,
    )

    try:
        implemented = bool(is_embedding_implemented(embedding_id))
        code_status = "implemented" if implemented else "pending"
    except V541InterfaceError:
        # A landed module that fails to import is a hard failure for the
        # MAIN matrix (E0–E4) but the conditional E5 / diagnostic E6 may
        # legitimately still be in flight — plan-init records them as not
        # implemented, and the runner's registry dispatch still refuses to
        # actually LAUNCH them (so a broken module can never train).
        if embedding_id in ("E5", "E6"):
            implemented, code_status = False, "module_import_failed"
        else:
            raise

    return sha256_json({
        "schema": EMBEDDING_SPEC_SCHEMA,
        "embedding_id": embedding_id,
        "config_row": row,
        "code_spec": spec_dict,
        "implemented_in_code": implemented,
        "code_status": code_status,
    })


def feature_recipe_sha256(cfg: Mapping[str, Any], embedding_id: str) -> str:
    """Per-candidate FEATURE identity: recipe + provider constants + view.

    ``E1`` and ``E2`` share the ``chem_union_plus_edit`` view but have
    different families — the family is part of the digest, so two
    candidates can never collapse onto one feature SHA.
    """
    embedding_id = str(embedding_id)
    row = _registry_row(cfg, embedding_id)
    recipe = (cfg.get("embedding_contract") or {}).get("feature_recipe")
    if not isinstance(recipe, Mapping):
        raise V541ProvenanceError(
            "V541_FEATURE_RECIPE", "embedding_contract.feature_recipe missing")
    from pes2ts_core.pesn.models.v541 import feature_providers as prov

    return sha256_json({
        "schema": FEATURE_RECIPE_SCHEMA,
        "embedding_id": embedding_id,
        "family": row.get("family"),
        "internal_view": row.get("internal_view"),
        "output": row.get("output"),
        "feature_recipe": dict(recipe),
        "provider_constants": {
            "chem_edge_dim": int(prov.CHEM_EDGE_DIM),
            "chem_edge_fields": list(prov.CHEM_EDGE_FIELDS),
            "angle_shell_tie_tol": float(prov.ANGLE_SHELL_TIE_TOL),
            "angle_tie_policy": str(prov.ANGLE_TIE_POLICY),
            "angle_max_neighbors": int(prov.ANGLE_MAX_NEIGHBORS),
            "angle_max_pairs": int(prov.ANGLE_MAX_PAIRS),
            "rbf_n": int(prov.RBF_N),
            "rbf_center_end": float(prov.RBF_CENTER_END),
            "rbf_sigma": float(prov.RBF_SIGMA_ANGSTROM),
        },
    })


# ---------------------------------------------------------------------------
# file / artifact digests
# ---------------------------------------------------------------------------
def config_file_sha256(config_path: Any) -> str:
    return sha256_file(config_path)


def r3_init_sha256(cfg: Mapping[str, Any], *, verify_file: bool = True) -> str:
    """R3 init SHA from the config, byte-verified against the checkpoint."""
    init = (cfg.get("training_contract") or {}).get("backbone_init")
    if not isinstance(init, Mapping):
        raise V541ProvenanceError(
            "V541_R3_INIT", "training_contract.backbone_init missing")
    sha = init.get("sha256")
    path = init.get("path")
    if not isinstance(sha, str) or len(sha) != 64:
        raise V541ProvenanceError(
            "V541_R3_INIT", "backbone_init.sha256 malformed: {!r}".format(sha))
    if verify_file:
        target = Path(str(path))
        if not target.is_absolute():
            target = PROJECT_ROOT / target
        if not target.is_file():
            raise V541ProvenanceError(
                "V541_R3_INIT_MISSING", "R3 init checkpoint missing: {}".format(path))
        actual = sha256_file(target)
        if actual != sha:
            raise V541ProvenanceError(
                "V541_R3_INIT_DRIFT",
                "R3 init {} is {} but config carries {}".format(
                    path, actual[:12], sha[:12]),
            )
    return str(sha)


def label_spec_shas(
    cfg: Mapping[str, Any],
    v53mp_cfg: Optional[Mapping[str, Any]] = None,
    *,
    v53mp_config_path: Any = None,
) -> Tuple[str, str]:
    """``(label_spec_sha256, endpoint_label_spec_sha256)``.

    The endpoint-label spec is still frozen in the V5.3-MP campaign yaml;
    its digest is computed by the OLD runner's own helper (read-only
    reuse), then wrapped together with the V5.4.1 teacher/process contract
    so the V541 label identity covers both halves.
    """
    if v53mp_cfg is None:
        import yaml

        path = Path(v53mp_config_path or (PROJECT_ROOT / DEFAULT_V53MP_CONFIG))
        if not path.is_file():
            raise V541ProvenanceError(
                "V541_LABEL_SPEC_MISSING",
                "V5.3-MP config with the frozen endpoint-label spec is "
                "missing: {}".format(path),
            )
        try:
            v53mp_cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            raise V541ProvenanceError(
                "V541_LABEL_SPEC_MISSING",
                "{} unreadable: {}".format(path, exc),
            ) from exc
    from scripts.campaigns.r3_v53mp_run import _label_spec_sha256

    try:
        endpoint_sha = _label_spec_sha256(dict(v53mp_cfg))
    except Exception as exc:  # noqa: BLE001 - legacy typed error -> V541 code
        raise V541ProvenanceError(
            "V541_LABEL_SPEC_MISSING", str(exc)) from exc
    data = cfg.get("data_contract") or {}
    payload = {
        "schema": LABEL_SPEC_SCHEMA,
        "endpoint_label_spec_sha256": endpoint_sha,
        "move_teacher": data.get("move_teacher"),
        "move_teacher_params": data.get("move_teacher_params"),
        "process_statuses": list(data.get("process_statuses") or ()),
        "split_contract_subsets": list(
            (cfg.get("split_contract") or {}).get("subsets") or ()),
    }
    if not payload["move_teacher"] or not payload["process_statuses"]:
        raise V541ProvenanceError(
            "V541_LABEL_SPEC_MISSING",
            "data_contract.move_teacher / process_statuses missing",
        )
    return sha256_json(payload), endpoint_sha


def artifact_shas(
    cfg: Mapping[str, Any],
    *,
    config_path: Any,
    tables_dir: Any,
    split_path: Any,
    code_sha: str,
    v53mp_config_path: Any = None,
) -> Dict[str, str]:
    """The full SHA block every runplan line carries (fail-closed)."""
    tables = Path(tables_dir)
    data_manifest = tables / "data_manifest.json"
    teacher_manifest = tables / "teacher_manifest.json"
    split = Path(split_path)
    for path in (data_manifest, teacher_manifest, split):
        if not path.is_file():
            raise V541ProvenanceError(
                "V541_ARTIFACT_MISSING", "required artifact missing: {}".format(path))
    label_sha, endpoint_label_sha = label_spec_shas(
        cfg, v53mp_config_path=v53mp_config_path)
    return {
        "config_sha256": config_file_sha256(config_path),
        "data_manifest_sha256": sha256_file(data_manifest),
        "teacher_manifest_sha256": sha256_file(teacher_manifest),
        "split_sha256": sha256_file(split),
        "label_spec_sha256": label_sha,
        "endpoint_label_spec_sha256": endpoint_label_sha,
        "r3_init_sha256": r3_init_sha256(cfg),
        "code_sha256": code_sha,
        "environment_sha256": environment_sha256(),
        "git_commit": git_commit() or "",
    }
