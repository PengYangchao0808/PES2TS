"""V5.4.2 run identity — the ONE canonical SHA recipe (gate-0 advisory A3
closed; ADR-0033 §2.6, config ``identity_contract``).

Canonicalization (FROZEN, the single convention for every JSON-shaped
identity field in V5.4.2)::

    canonical_bytes(obj) == json.dumps(
        obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    canonical_sha256(obj) == hashlib.sha256(canonical_bytes(obj)).hexdigest()

No ordering / mtime / locale dependence: dict keys are sorted recursively by
``sort_keys=True``, there is no whitespace, and unicode is emitted as UTF-8
bytes (never ``\\uXXXX``-escaped).  File-shaped fields (config, plan, oracle
design source, source manifest) are hashed over their RAW FILE BYTES — never
re-serialized — so a byte-identical file always yields the same digest on
every machine.  No field may invent a different canonical form.

Every per-field builder returns ``(sha256, canonical_payload)``: the payload
is the exact JSON object that was hashed (or, for file-byte digests, a
reconstruction record naming the path/length), so it can be persisted next
to the digest and the SHA is later RECONSTRUCTIBLE by re-canonicalizing the
persisted payload.

Fields built here (the 14 frozen required run-identity fields of config
``identity_contract.run_identity_required`` plus the optional record-only
``code_commit``):

* ``config_sha256(path)`` — campaign config file bytes;
* ``runplan_sha256(runplan_record)`` — canonical JSON of the frozen runplan
  pin record;
* ``budget_sha256(budget_block)`` — canonical JSON of the budget policy
  record (pre-profile records legitimately contain ``null`` entries — the
  digest binds the null state itself);
* ``common_protocol_sha256(config)`` — the shared numeric protocol only;
  see :data:`COMMON_PROTOCOL_STRUCTURAL_EXCLUDES` for the explicit
  candidate-specific exclusion list;
* ``environment_sha256()`` — python/torch/CUDA/cuDNN + deterministic flags,
  cached so it is stable across calls;
* ``source_tree_fingerprint(root)`` — ``git rev-parse HEAD`` equivalent
  (read from ``.git`` directly — library code here never spawns a child
  process) plus a sorted SHA-256 manifest of the FROZEN config
  ``identity_contract.code`` scope: the include dirs ``pes2ts_core/``,
  ``config/campaigns/v542_templates/``, ``scripts/campaigns/``,
  ``scripts/agent_ops/``, ``scripts/infra/`` plus the exact include files
  ``config/campaigns/pes2ts_v542_backbone_50k.yaml``, ``requirements.txt``,
  ``requirements-pesn-training.txt``, ``requirements-pesn-runtime.txt``,
  filtered by the frozen suffixes ``.py``/``.yaml``/``.json``/``.txt``
  (excluding ``__pycache__``, ``.git``, ``outputs``, ``node_modules`` and
  the file ``docs_consistency_config.json``) plus an exact content-based
  ``dirty`` flag;
* ``init_mapping_sha256(config)`` — ``init_policy.init_mapping`` +
  ``per_candidate_zero_init`` + the frozen seed-recipe blocks;
* ``process_head_anchor_sha256(seed)`` — the anchor algorithm record
  (method/seed/steps/``excludes_w_v``) PLUS the key-sorted tensor digest of
  ``candidates.common.build_process_head(seed)``, so the SHA is
  reconstructible from the seed alone;
* ``statistics_contract_sha256(config)`` — ``evaluation_contract.
  statistics`` + ``route_logic`` + ``final_selection_rule`` + top-level
  ``thresholds``;
* ``backbone_spec_sha256(config, candidate_id)`` — that candidate's registry
  row + its R3 semantic-mapping row + its zero-init row;
* ``source_manifest_sha256(path)`` — source manifest file bytes (the
  manifest FILE must be persisted as ``canonical_bytes(fingerprint_payload)``
  — see :func:`source_tree_fingerprint` — so its file-bytes digest equals
  the fingerprint digest computed before persistence);
* ``plan_sha256()`` / ``oracle_design_source_sha256()`` — the pinned plan
  document and the Oracle design source, the latter VERIFIED against the
  WP0-frozen value ``4f13bfbd…ccccb``.

:func:`assemble_run_identity` wires all of them into the frozen field set
with TYPED refusals (:class:`V542IdentityError`) for missing/None inputs,
digest drift against a pinned value, and — when the caller demands
cleanliness — a dirty source tree.

Precedent/style: the V5.4.1 helper
``pes2ts_core.pesn.offline.training.v541_identity`` (canonical JSON SHA,
``.git``-file HEAD reading without subprocess).  Truth-guard discipline:
nothing in this module spawns a child process, imports the truth reader, or
names quarantined paths.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import struct
from pathlib import Path
from typing import Any, Final, Mapping, Sequence

import torch

from pes2ts_core.pesn.offline.training.v541_identity import git_state

__all__ = [
    "COMMON_PROTOCOL_INCLUDES_FALLBACK",
    "COMMON_PROTOCOL_STRUCTURAL_EXCLUDES",
    "DEFAULT_CAMPAIGN_CONFIG",
    "DEFAULT_ORACLE_DESIGN_SOURCE",
    "DEFAULT_PLAN_PATH",
    "FINGERPRINT_EXCLUDED_DIRS",
    "FINGERPRINT_EXCLUDED_FILES",
    "FINGERPRINT_INCLUDE_DIRS",
    "FINGERPRINT_INCLUDE_FILES",
    "FINGERPRINT_INCLUDES",
    "FINGERPRINT_SUFFIXES",
    "IDENTITY_REQUIRED_FIELDS",
    "PINNED_ORACLE_DESIGN_SOURCE_SHA256",
    "RUN_IDENTITY_SCHEMA",
    "V542IdentityError",
    "assemble_run_identity",
    "backbone_spec_sha256",
    "budget_sha256",
    "canonical_bytes",
    "canonical_sha256",
    "common_protocol_sha256",
    "config_sha256",
    "environment_sha256",
    "init_mapping_sha256",
    "oracle_design_source_sha256",
    "plan_sha256",
    "process_head_anchor_sha256",
    "runplan_sha256",
    "source_manifest_sha256",
    "source_tree_fingerprint",
    "state_dict_sha256",
    "statistics_contract_sha256",
]

# ---------------------------------------------------------------------------
# frozen constants (mirror config identity_contract / governance / plan §6)
# ---------------------------------------------------------------------------
#: Run-identity record schema id.
RUN_IDENTITY_SCHEMA: Final[str] = "pes2ts_v542_run_identity_v1"

#: The 14 frozen required fields, in config ``run_identity_required`` order.
IDENTITY_REQUIRED_FIELDS: Final[tuple[str, ...]] = (
    "source_manifest_sha256",
    "config_sha256",
    "common_protocol_sha256",
    "environment_sha256",
    "backbone_spec_sha256",
    "init_mapping_sha256",
    "process_head_anchor_sha256",
    "statistics_contract_sha256",
    "bootstrap_seed_root",
    "plan_sha256",
    "oracle_design_source_sha256",
    "runplan_sha256",
    "budget_policy_sha256",
    "sampler_spec_sha256",
)

DEFAULT_CAMPAIGN_CONFIG: Final[str] = (
    "config/campaigns/pes2ts_v542_backbone_50k.yaml")
DEFAULT_PLAN_PATH: Final[str] = (
    "docs/plans/PES2TS_V5.4.2_Backbone_50K_详细阶段实施方案_v1.0_20261029.md")
DEFAULT_ORACLE_DESIGN_SOURCE: Final[str] = (
    ".slim/deepwork/v542-backbone-20261029/oracle-architecture.md")
#: WP0-frozen byte identity of the Oracle design source (ADR-0033 §2.6).
PINNED_ORACLE_DESIGN_SOURCE_SHA256: Final[str] = (
    "4f13bfbd26d4acf17572641e30453000d65243599519c269473f07d6678ccccb")

#: Fingerprint manifest scope — the FROZEN config ``identity_contract.
#: code`` block implemented literally (oracle gate-3 B1).  The include
#: list mixes DIRECTORY roots (walked recursively) with exact repo-root
#: FILE paths; the suffix filter applies to walked files, and the frozen
#: exclusions prune directories by name anywhere plus one excluded FILE
#: (``docs_consistency_config.json`` — a tooling config that must never
#: enter the source identity).
FINGERPRINT_INCLUDE_DIRS: Final[tuple[str, ...]] = (
    "config/campaigns/v542_templates",
    "pes2ts_core",
    "scripts/agent_ops",
    "scripts/campaigns",
    "scripts/infra",
)
FINGERPRINT_INCLUDE_FILES: Final[tuple[str, ...]] = (
    "config/campaigns/pes2ts_v542_backbone_50k.yaml",
    "requirements-pesn-runtime.txt",
    "requirements-pesn-training.txt",
    "requirements.txt",
)
#: The complete frozen include list (dirs + files), config order sorted.
FINGERPRINT_INCLUDES: Final[tuple[str, ...]] = tuple(
    sorted(FINGERPRINT_INCLUDE_DIRS + FINGERPRINT_INCLUDE_FILES))
FINGERPRINT_SUFFIXES: Final[tuple[str, ...]] = (".py", ".yaml", ".json", ".txt")
#: Directory names pruned everywhere under the fingerprint include dirs.
FINGERPRINT_EXCLUDED_DIRS: Final[frozenset[str]] = frozenset(
    {"__pycache__", ".git", "outputs", "node_modules"})
#: File names excluded from the fingerprint EVERYWHERE (config
#: ``identity_contract.code.excludes``): editing this tooling config must
#: never move the source identity or flip ``dirty``.
FINGERPRINT_EXCLUDED_FILES: Final[frozenset[str]] = frozenset(
    {"docs_consistency_config.json"})

#: Fallback include list for :func:`common_protocol_sha256` — normally read
#: from config ``identity_contract.common_protocol.includes`` (the config is
#: the authority; this constant is only the documented fallback).
COMMON_PROTOCOL_INCLUDES_FALLBACK: Final[tuple[str, ...]] = (
    "schema", "campaign_id", "objective", "schedule", "lambdas", "batch",
    "sampler_binding", "selection", "tau_end", "inner_val_max_samples",
    "split_contract", "backbone_init", "data_manifest_sha256",
    "teacher_manifest_sha256", "split_sha256", "label_spec_sha256",
    "code_sha256")

#: The EXPLICIT candidate-specific exclusion list of the common-protocol
#: layer (config ``identity_contract.common_protocol.excludes`` plus the
#: structural candidate-specific blocks).  These are hashed by their OWN
#: identity layers (``backbone_spec_sha256`` / ``init_mapping_sha256``) —
#: mutating any of them must NEVER move ``common_protocol_sha256``.
COMMON_PROTOCOL_STRUCTURAL_EXCLUDES: Final[tuple[str, ...]] = (
    "backbone_contract.candidates",  # per-candidate registry rows
    "training_contract.init_policy.init_mapping.r3_semantic_mapping",
    "training_contract.init_policy.per_candidate_zero_init",
    "smoke", "profile", "diagnostic",  # diagnostic-only flag blocks
    "run_matrix.runs",  # candidate ids / seeds in run rows
)

#: The canonical anchor algorithm record (config
#: ``training_contract.init_policy.process_head_anchor``; mirrored by
#: ``candidates.common.build_process_head``).
PROCESS_HEAD_ANCHOR_METHOD: Final[str] = "fork_rng_replay"
PROCESS_HEAD_ANCHOR_STEPS: Final[tuple[str, ...]] = (
    "torch.random.fork_rng(devices=[]) scope",
    "torch.manual_seed(anchor_seed)",
    "StarEGNNV3()  # legacy construction draw order, discarded",
    "ReactionProcessHeadV53MP()  # the anchor head",
    "state_dict digest over sorted keys (dtype+shape+raw bytes)",
)


class V542IdentityError(RuntimeError):
    """Typed refusal of the V5.4.2 run-identity machinery (``code`` + ``detail``)."""

    def __init__(self, code: str, detail: str = "") -> None:
        super().__init__(f"{code}: {detail}" if detail else code)
        self.code = str(code)
        self.detail = str(detail)


# ---------------------------------------------------------------------------
# the ONE canonical recipe
# ---------------------------------------------------------------------------
def canonical_bytes(obj: Any) -> bytes:
    """Canonical JSON bytes: sorted keys, compact, UTF-8, never escaped.

    The single frozen serialization of V5.4.2 identity payloads — every
    JSON-shaped identity field hashes exactly these bytes.  A value that is
    not JSON-serializable is a typed refusal (never a silent ``str()``
    fallback: an identity digest must never depend on ``repr``).
    """
    try:
        return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=False).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise V542IdentityError(
            "V542_IDENTITY_SERIALIZE",
            f"payload is not canonically JSON-serializable: {exc}") from exc


def canonical_sha256(obj: Any) -> str:
    """SHA-256 hex of :func:`canonical_bytes` — the canonical JSON digest."""
    return hashlib.sha256(canonical_bytes(obj)).hexdigest()


def state_dict_sha256(state: Mapping[str, torch.Tensor]) -> str:
    """Key-sorted tensor digest of a state dict (anchor reconstructibility).

    Feeds every entry in sorted key order as ``key\\0 dtype\\0 shape\\0 len\\0
    raw_bytes`` — dtype/shape framing makes the digest collision-safe
    against reshapes, and the raw bytes are the exact little-endian tensor
    payload.  Non-numpy-representable dtypes fall back to a ``uint8`` view
    (same convention as the V541 precedent).
    """
    h = hashlib.sha256()
    for key in sorted(state):
        tensor = state[key].detach().cpu().contiguous()
        h.update(key.encode("utf-8"))
        h.update(b"\0")
        h.update(str(tensor.dtype).encode("utf-8"))
        h.update(b"\0")
        h.update(",".join(str(int(d)) for d in tensor.shape).encode("utf-8"))
        h.update(b"\0")
        try:
            raw = tensor.numpy().tobytes()
        except (TypeError, RuntimeError):  # bfloat16 / float8 / etc.
            raw = tensor.view(torch.uint8).numpy().tobytes()
        h.update(str(len(raw)).encode("utf-8"))
        h.update(b"\0")
        h.update(raw)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# file-byte digests (config / manifest / plan / oracle design source)
# ---------------------------------------------------------------------------
def _display_path(path: Path, root: Path | str | None) -> str:
    """Prefer a repo-relative posix path in reconstruction records."""
    resolved = Path(path)
    if root is not None:
        try:
            return resolved.resolve().relative_to(
                Path(root).resolve()).as_posix()
        except ValueError:
            pass
    return resolved.as_posix()


def _file_bytes_sha256(path: Path | str) -> tuple[str, dict[str, Any], int]:
    p = Path(path)
    if not p.is_file():
        raise V542IdentityError(
            "V542_IDENTITY_FILE_MISSING",
            f"identity input file not found: {p}")
    digest = hashlib.sha256()
    n_bytes = 0
    with p.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
            n_bytes += len(chunk)
    return digest.hexdigest(), {}, n_bytes


def _file_digest_record(path: Path | str, sha: str, n_bytes: int,
                        root: Path | str | None = None) -> dict[str, Any]:
    return {"recipe": "file_bytes_sha256",
            "path": _display_path(path, root),
            "n_bytes": int(n_bytes),
            "sha256": sha}


def config_sha256(path: Path | str, *,
                  root: Path | str | None = None) -> tuple[str, dict[str, Any]]:
    """SHA-256 of the campaign config FILE BYTES (never a re-serialization)."""
    sha, _, n_bytes = _file_bytes_sha256(path)
    return sha, _file_digest_record(path, sha, n_bytes, root)


def source_manifest_sha256(path: Path | str, *,
                           root: Path | str | None = None
                           ) -> tuple[str, dict[str, Any]]:
    """SHA-256 of the frozen source-manifest FILE BYTES.

    The manifest file must be persisted as
    ``canonical_bytes(source_tree_fingerprint payload)`` (compact canonical
    JSON, no trailing newline) — then this file-bytes digest EQUALS the
    fingerprint digest computed before persistence, and
    ``source_manifest_sha256`` is reconstructible both ways.
    """
    sha, _, n_bytes = _file_bytes_sha256(path)
    return sha, _file_digest_record(path, sha, n_bytes, root)


def plan_sha256(path: Path | str | None = None, *,
                root: Path | str | None = None) -> tuple[str, dict[str, Any]]:
    """SHA-256 of the pinned V5.4.2 plan document (file bytes)."""
    target = Path(path) if path is not None else _repo_path(DEFAULT_PLAN_PATH)
    sha, _, n_bytes = _file_bytes_sha256(target)
    return sha, _file_digest_record(target, sha, n_bytes, root)


def oracle_design_source_sha256(path: Path | str | None = None, *,
                                root: Path | str | None = None
                                ) -> tuple[str, dict[str, Any]]:
    """SHA-256 of the Oracle design source, VERIFIED against the WP0 pin.

    The file bytes must hash to
    :data:`PINNED_ORACLE_DESIGN_SOURCE_SHA256`; any drift is the typed
    refusal ``V542_IDENTITY_ORACLE_SHA`` — the config cannot silently
    re-pin the design source.
    """
    target = (Path(path) if path is not None
              else _repo_path(DEFAULT_ORACLE_DESIGN_SOURCE))
    sha, _, n_bytes = _file_bytes_sha256(target)
    if sha != PINNED_ORACLE_DESIGN_SOURCE_SHA256:
        raise V542IdentityError(
            "V542_IDENTITY_ORACLE_SHA",
            f"oracle design source byte drift at {target}: sha256 {sha} != "
            f"pinned {PINNED_ORACLE_DESIGN_SOURCE_SHA256}")
    record = _file_digest_record(target, sha, n_bytes, root)
    record["pinned_sha256"] = PINNED_ORACLE_DESIGN_SOURCE_SHA256
    record["verified_against_pin"] = True
    return sha, record


# ---------------------------------------------------------------------------
# git plumbing — pure file I/O, NEVER a child process (truth-guard rule)
# ---------------------------------------------------------------------------
def _locate_gitdir(root: Path) -> Path:
    """Resolve the checkout's ``.git`` (directory or ``gitdir:`` link file)."""
    dot = root / ".git"
    if dot.is_dir():
        return dot
    if dot.is_file():
        text = dot.read_text(encoding="utf-8", errors="replace")
        if text.startswith("gitdir:"):
            target = Path(text.split(":", 1)[1].strip())
            if not target.is_absolute():
                target = root / target
            return target
    raise V542IdentityError(
        "V542_IDENTITY_GIT",
        f"no .git directory/link found at {root} — the source tree "
        "fingerprint requires a git checkout")


def _git_index_entries(gitdir: Path) -> dict[str, str]:
    """Parse the git index (v2/v3) into ``{relpath: blob_sha1_hex}``.

    The index is the authoritative TRACKED-FILE list (equivalent to
    ``git ls-files``) and carries each entry's git blob SHA-1, which the
    dirty check compares against the worktree content — content-based, no
    mtime dependence.  Unsupported index versions are a typed refusal, never
    a silent misparse.
    """
    index_path = gitdir / "index"
    if not index_path.is_file():
        raise V542IdentityError(
            "V542_IDENTITY_GIT",
            f"git index not found at {index_path} — cannot enumerate "
            "tracked files for the source tree fingerprint")
    data = index_path.read_bytes()
    if len(data) < 12 or data[:4] != b"DIRC":
        raise V542IdentityError(
            "V542_IDENTITY_GIT",
            f"{index_path} is not a git index (bad DIRC header)")
    version, count = struct.unpack(">II", data[4:12])
    if version not in (2, 3):
        raise V542IdentityError(
            "V542_IDENTITY_GIT",
            f"unsupported git index version {version} at {index_path} "
            "(only v2/v3 entries are parsed; convert with "
            "`git update-index --index-version 2`)")
    entries: dict[str, str] = {}
    pos = 12
    for _ in range(count):
        start = pos
        if pos + 62 > len(data):
            raise V542IdentityError(
                "V542_IDENTITY_GIT",
                f"truncated git index entry at offset {pos}")
        blob_sha = data[pos + 40:pos + 60].hex()
        flags = struct.unpack(">H", data[pos + 60:pos + 62])[0]
        pos += 62
        if flags & 0x4000:  # extended flag word
            pos += 2
        name_len = flags & 0x0FFF
        if name_len == 0x0FFF:  # long name: NUL-terminated
            end = data.index(b"\0", pos)
            name = data[pos:end].decode("utf-8", "surrogateescape")
            pos = end + 1
        else:
            name = data[pos:pos + name_len].decode("utf-8", "surrogateescape")
            pos += name_len + 1
        pos = start + ((pos - start + 7) // 8) * 8  # 8-byte entry padding
        entries[name] = blob_sha
    return entries


def _git_blob_sha1(content: bytes) -> str:
    """Git object SHA-1 of a blob (``sha1("blob <len>\\0" + content)``)."""
    h = hashlib.sha1()
    h.update(f"blob {len(content)}\0".encode("ascii"))
    h.update(content)
    return h.hexdigest()


def _repo_path(relative: str, root: Path | str | None = None) -> Path:
    base = Path(root) if root is not None else _default_repo_root()
    return base / relative


def _default_repo_root() -> Path:
    # pes2ts_core/pesn/models/v542/identity.py -> repo root is parents[4]
    return Path(__file__).resolve().parents[4]


# ---------------------------------------------------------------------------
# source tree fingerprint
# ---------------------------------------------------------------------------
def source_tree_fingerprint(root: Path | str | None = None
                            ) -> tuple[str, dict[str, Any]]:
    """Code identity: HEAD + tracked-source manifest + exact dirty flag.

    The manifest scope is the FROZEN config ``identity_contract.code``
    block (oracle gate-3 B1): the include DIRS
    ``config/campaigns/v542_templates``, ``pes2ts_core``,
    ``scripts/agent_ops``, ``scripts/campaigns``, ``scripts/infra`` plus
    the exact include FILES ``config/campaigns/
    pes2ts_v542_backbone_50k.yaml``, ``requirements.txt``,
    ``requirements-pesn-training.txt``, ``requirements-pesn-runtime.txt``,
    filtered by the frozen suffixes ``.py``/``.yaml``/``.json``/``.txt``
    and pruned by the frozen exclusions (directories ``__pycache__``,
    ``.git``, ``outputs``, ``node_modules`` anywhere under the scope, and
    the FILE ``docs_consistency_config.json`` everywhere).

    Payload (deterministic, reconstructible, no ordering/mtime/locale
    dependence)::

        {"recipe": "pes2ts_v542_source_tree_fingerprint_v1",
         "git_head": <40-hex from .git/HEAD>,          # git rev-parse HEAD
         "dirty": true|false,
         "includes": [<sorted frozen include dirs + files>],
         "suffixes": [".json", ".py", ".txt", ".yaml"],
         "excluded_dirs": ["__pycache__", ".git", "node_modules", "outputs"],
         "excluded_files": ["docs_consistency_config.json"],
         "n_files": N,
         "files": [{"path": ..., "sha256": ...}, ...],  # sorted by path
         "modified_tracked": [...], "missing_tracked": [...],
         "untracked": [...]}

    ``files`` lists every TRACKED suffix-matching file under the include
    scope (git-index sourced — equivalent to ``git ls-files``), each
    hashed over its worktree bytes.  ``dirty`` is exact and content-based:
    any tracked in-scope file whose worktree bytes no longer match its
    index blob SHA-1, any missing tracked in-scope file, or any untracked
    suffix-matching file inside the include dirs (an untracked include
    FILE counts too).  Excluded files (``docs_consistency_config.json``)
    never appear in the manifest and never flip ``dirty``.  Full
    ``.gitignore`` emulation is deliberately NOT attempted for untracked
    detection (over-reporting dirty is the fail-safe direction); the
    excluded-directory set mirrors the repo ``.gitignore`` for this scope.

    The returned digest is ``canonical_sha256(payload)``.  Persisting the
    payload as the manifest FILE (canonical bytes, no trailing newline)
    makes :func:`source_manifest_sha256` of that file return the SAME
    digest.
    """
    base = Path(root) if root is not None else _default_repo_root()
    gitdir = _locate_gitdir(base)
    head = git_state(cwd=str(base)).get("commit")
    if not isinstance(head, str) or len(head) != 40:
        raise V542IdentityError(
            "V542_IDENTITY_GIT",
            f"cannot read a 40-hex commit from {gitdir}/HEAD "
            f"(got {head!r})")
    index = _git_index_entries(gitdir)

    in_scope_dirs = tuple(name for name in FINGERPRINT_INCLUDE_DIRS
                          if (base / name).is_dir())
    in_scope_files = tuple(name for name in FINGERPRINT_INCLUDE_FILES
                           if (base / name).is_file())

    def _excluded(rel: str) -> bool:
        parts = rel.split("/")
        return (any(part in FINGERPRINT_EXCLUDED_DIRS for part in parts)
                or parts[-1] in FINGERPRINT_EXCLUDED_FILES)

    def _in_scope(rel: str) -> bool:
        if _excluded(rel):
            return False
        if rel in FINGERPRINT_INCLUDE_FILES:
            return True
        return any(rel.startswith(d + "/") for d in FINGERPRINT_INCLUDE_DIRS)

    files: list[dict[str, str]] = []
    modified: list[str] = []
    missing: list[str] = []
    tracked_paths: set[str] = set()
    for rel in sorted(index):
        if not _in_scope(rel):
            continue
        tracked_paths.add(rel)
        work = base / rel
        if not work.is_file():
            missing.append(rel)
            continue
        content = work.read_bytes()
        if _git_blob_sha1(content) != index[rel]:
            modified.append(rel)
        if rel.endswith(FINGERPRINT_SUFFIXES):
            files.append({
                "path": rel,
                "sha256": hashlib.sha256(content).hexdigest(),
            })
    files.sort(key=lambda entry: entry["path"])

    untracked: list[str] = []
    for scope in in_scope_dirs:
        for dirpath, dirnames, filenames in os.walk(base / scope):
            dirnames[:] = sorted(d for d in dirnames
                                 if d not in FINGERPRINT_EXCLUDED_DIRS)
            for filename in sorted(filenames):
                if not filename.endswith(FINGERPRINT_SUFFIXES):
                    continue
                if filename in FINGERPRINT_EXCLUDED_FILES:
                    continue
                full = Path(dirpath) / filename
                rel = full.relative_to(base).as_posix()
                if rel not in tracked_paths:
                    untracked.append(rel)
    # an untracked include FILE (e.g. a deleted-from-index requirements
    # file that still sits in the worktree) counts as dirty, fail-safe
    for rel in in_scope_files:
        if rel not in tracked_paths and rel.endswith(FINGERPRINT_SUFFIXES):
            if rel not in untracked:
                untracked.append(rel)
    untracked.sort()

    payload: dict[str, Any] = {
        "recipe": "pes2ts_v542_source_tree_fingerprint_v1",
        "git_head": head,
        "dirty": bool(modified or missing or untracked),
        "includes": list(FINGERPRINT_INCLUDES),
        "suffixes": sorted(FINGERPRINT_SUFFIXES),
        "excluded_dirs": sorted(FINGERPRINT_EXCLUDED_DIRS),
        "excluded_files": sorted(FINGERPRINT_EXCLUDED_FILES),
        "n_files": len(files),
        "files": files,
        "modified_tracked": sorted(modified),
        "missing_tracked": sorted(missing),
        "untracked": untracked,
    }
    return canonical_sha256(payload), payload


# ---------------------------------------------------------------------------
# environment
# ---------------------------------------------------------------------------
_ENV_CACHE: dict[str, Any] = {}


def environment_sha256() -> tuple[str, dict[str, Any]]:
    """Interpreter/framework/device identity, cached for cross-call stability.

    Records exactly: python version, torch version, CUDA availability and
    version, device name(s), cuDNN version (when available) and the
    deterministic flags (cuDNN deterministic/benchmark, deterministic
    algorithms mode, TF32 switches).  The payload is computed once per
    process and cached, so two calls in the same environment always agree.
    """
    if not _ENV_CACHE:
        cuda_available = bool(torch.cuda.is_available())
        device_names: list[str] = []
        if cuda_available:
            try:
                device_names = [torch.cuda.get_device_name(i)
                                 for i in range(torch.cuda.device_count())]
            except RuntimeError:  # pragma: no cover - driver edge
                device_names = ["<unavailable>"]
        cudnn_version: int | None
        try:
            cudnn_version = torch.backends.cudnn.version()
        except Exception:  # pragma: no cover - defensive
            cudnn_version = None
        _ENV_CACHE["payload"] = {
            "recipe": "v542_environment_v1",
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "cuda_available": cuda_available,
            "cuda_version": torch.version.cuda,
            "device_names": device_names,
            "cudnn_version": cudnn_version,
            "deterministic": {
                "cudnn_deterministic": bool(
                    torch.backends.cudnn.deterministic),
                "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
                "deterministic_algorithms": bool(
                    torch.are_deterministic_algorithms_enabled()),
                "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
                "cuda_matmul_allow_tf32": bool(
                    torch.backends.cuda.matmul.allow_tf32),
            },
        }
    payload = _ENV_CACHE["payload"]
    return canonical_sha256(payload), dict(payload)


# ---------------------------------------------------------------------------
# config-derived builders
# ---------------------------------------------------------------------------
def _require_mapping(obj: Any, what: str) -> Mapping[str, Any]:
    if not isinstance(obj, Mapping):
        raise V542IdentityError(
            "V542_IDENTITY_CONFIG",
            f"{what} must be a mapping, got {type(obj).__name__}")
    return obj


def _require_block(cfg: Mapping[str, Any], path: Sequence[str]) -> Any:
    node: Any = cfg
    walked: list[str] = []
    for key in path:
        node = _require_mapping(node, "config." + ".".join(walked)
                                if walked else "config")
        if key not in node:
            raise V542IdentityError(
                "V542_IDENTITY_CONFIG",
                f"missing config block config.{'.'.join(walked + [key])}")
        walked.append(key)
        node = node[key]
    return node


def _resolve_common_include(cfg: Mapping[str, Any], key: str) -> Any:
    """Resolve one common-protocol include: top-level, then training/split."""
    if key in cfg:
        return cfg[key]
    for section in ("training_contract", "split_contract"):
        block = cfg.get(section)
        if isinstance(block, Mapping) and key in block:
            return block[key]
    # Runtime-provenance fields (data/teacher/split/label/code manifest
    # SHAs) are runner-recorded identity layers, not config content: they
    # resolve to null HERE and are filled by the runner in its own record.
    return None


def common_protocol_sha256(config: Mapping[str, Any]
                           ) -> tuple[str, dict[str, Any]]:
    """Shared numeric protocol digest — candidate-specific fields EXCLUDED.

    The payload is an ALLOWLIST: exactly the include keys declared by config
    ``identity_contract.common_protocol.includes`` (resolved top-level →
    ``training_contract`` → ``split_contract``), plus the EXPLICIT exclusion
    list so the contract is self-describing inside the hash:

    * config-declared excludes (``backbone_id``, ``backbone_spec_sha256``,
      ``seed``, ``run_id``, ``environment_sha256``) and
    * the structural candidate-specific blocks of
      :data:`COMMON_PROTOCOL_STRUCTURAL_EXCLUDES` — the per-candidate
      registry rows, per-candidate R3 mapping tables, per-candidate
      zero-init rows, diagnostic-only flag blocks and the run-matrix rows
      carrying candidate ids/seeds.

    Because only the allowlist is hashed, mutating ANY excluded
    candidate-specific field cannot move this digest, while mutating a
    shared field (``training_contract.schedule``, ``lambdas``,
    ``sampler_binding``, ``split_contract``, …) must change it.  Same
    protocol ⇒ same digest across every candidate/seed of the campaign
    (config ``common_protocol.note``).
    """
    cfg = _require_mapping(config, "config")
    ic = _require_block(cfg, ("identity_contract", "common_protocol"))
    includes = ic.get("includes")
    if not isinstance(includes, Sequence) or isinstance(includes, str) \
            or not includes:
        includes = list(COMMON_PROTOCOL_INCLUDES_FALLBACK)
    config_excludes = ic.get("excludes") or []
    excludes = sorted({str(x) for x in config_excludes}
                      | set(COMMON_PROTOCOL_STRUCTURAL_EXCLUDES))
    payload: dict[str, Any] = {
        "recipe": "v542_common_protocol_v1",
        "method": ic.get("method"),
        "includes": {str(key): _resolve_common_include(cfg, str(key))
                     for key in includes},
        "excludes": excludes,
    }
    return canonical_sha256(payload), payload


def init_mapping_sha256(config: Mapping[str, Any]
                        ) -> tuple[str, dict[str, Any]]:
    """Initialization identity: mapping + zero-init + seed-recipe blocks.

    Hashes ``training_contract.init_policy`` sub-blocks ``init_mapping``,
    ``per_candidate_zero_init``, ``candidate_fresh_init`` (the frozen seed
    recipe: method/algorithm/overrides) and ``construction_order`` — the
    complete frozen initialization contract of ADR-0033 §2.3.
    """
    cfg = _require_mapping(config, "config")
    init_policy = _require_block(cfg, ("training_contract", "init_policy"))
    fresh = _require_mapping(
        init_policy.get("candidate_fresh_init"),
        "training_contract.init_policy.candidate_fresh_init")
    seed_keys = ("method", "seed_recipe", "seed_recipe_algorithm",
                 "init_source_id_default", "init_source_id_overrides")
    payload: dict[str, Any] = {
        "recipe": "v542_init_mapping_v1",
        "init_mapping": init_policy.get("init_mapping"),
        "per_candidate_zero_init": init_policy.get("per_candidate_zero_init"),
        "seed_recipe": {key: fresh.get(key) for key in seed_keys},
        "construction_order": init_policy.get("construction_order"),
    }
    for key in ("init_mapping", "per_candidate_zero_init",
                "construction_order"):
        if payload[key] is None:
            raise V542IdentityError(
                "V542_IDENTITY_CONFIG",
                f"missing training_contract.init_policy.{key}")
    return canonical_sha256(payload), payload


def statistics_contract_sha256(config: Mapping[str, Any]
                               ) -> tuple[str, dict[str, Any]]:
    """Statistics identity: statistics + route_logic + final_selection_rule
    + thresholds (ADR-0033 §2.5; config freezes all four into this SHA)."""
    cfg = _require_mapping(config, "config")
    evaluation = _require_block(cfg, ("evaluation_contract",))
    for block in ("statistics", "route_logic", "final_selection_rule"):
        if block not in evaluation:
            raise V542IdentityError(
                "V542_IDENTITY_CONFIG",
                f"missing evaluation_contract.{block}")
    if "thresholds" not in cfg:
        raise V542IdentityError(
            "V542_IDENTITY_CONFIG", "missing top-level thresholds block")
    payload: dict[str, Any] = {
        "recipe": "v542_statistics_contract_v1",
        "statistics": evaluation["statistics"],
        "route_logic": evaluation["route_logic"],
        "final_selection_rule": evaluation["final_selection_rule"],
        "thresholds": cfg["thresholds"],
    }
    return canonical_sha256(payload), payload


def backbone_spec_sha256(config: Mapping[str, Any], candidate_id: str
                         ) -> tuple[str, dict[str, Any]]:
    """Per-candidate backbone spec: registry row + mapping row + zero-init.

    Unknown candidates are the typed refusal
    ``V542_IDENTITY_CANDIDATE_UNKNOWN``.
    """
    cfg = _require_mapping(config, "config")
    if not isinstance(candidate_id, str) or not candidate_id:
        raise V542IdentityError(
            "V542_IDENTITY_CANDIDATE_UNKNOWN",
            f"candidate_id must be a non-empty str, got {candidate_id!r}")
    candidates = _require_block(
        cfg, ("backbone_contract", "candidates"))
    if candidate_id not in candidates:
        raise V542IdentityError(
            "V542_IDENTITY_CANDIDATE_UNKNOWN",
            f"candidate {candidate_id!r} is not in "
            "backbone_contract.candidates "
            f"(known: {sorted(str(k) for k in candidates)})")
    init_policy = _require_block(cfg, ("training_contract", "init_policy"))
    mapping = (init_policy.get("init_mapping") or {}).get(
        "r3_semantic_mapping") or {}
    zero_init = init_policy.get("per_candidate_zero_init") or {}
    payload: dict[str, Any] = {
        "recipe": "v542_backbone_spec_v1",
        "candidate_id": candidate_id,
        "registry_row": candidates[candidate_id],
        "r3_semantic_mapping": mapping.get(candidate_id),
        "zero_init": zero_init.get(candidate_id),
    }
    return canonical_sha256(payload), payload


def runplan_sha256(runplan_record: Mapping[str, Any]
                   ) -> tuple[str, dict[str, Any]]:
    """Canonical JSON digest of the frozen runplan pin record."""
    record = _require_mapping(runplan_record, "runplan_record")
    payload = {"recipe": "v542_runplan_record_v1", "record": dict(record)}
    return canonical_sha256(payload), payload


def budget_sha256(budget_block: Mapping[str, Any]
                  ) -> tuple[str, dict[str, Any]]:
    """Canonical JSON digest of the budget policy record.

    A pre-profile record legitimately contains ``null`` budget entries
    (config: budget stays null until the same-env profile) — the digest
    binds that frozen null state; refilling the budget necessarily changes
    ``budget_policy_sha256``.
    """
    block = _require_mapping(budget_block, "budget_block")
    payload = {"recipe": "v542_budget_policy_v1", "record": dict(block)}
    return canonical_sha256(payload), payload


def process_head_anchor_sha256(seed: int) -> tuple[str, dict[str, Any]]:
    """Process-head anchor identity for one seed (reconstructible).

    Payload = the anchor ALGORITHM record (method/seed/construction steps/
    ``excludes_w_v``) PLUS the key-sorted tensor digest of the actual anchor
    head from ``candidates.common.build_process_head(seed)``.  Anyone can
    reconstruct: replay ``build_process_head(seed)``, recompute
    :func:`state_dict_sha256`, re-canonicalize, re-hash.
    """
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise V542IdentityError(
            "V542_IDENTITY_SEED",
            f"anchor seed must be a non-negative int, got {seed!r}")
    from pes2ts_core.pesn.models.v542.candidates.common import (  # lazy: heavy
        build_process_head)
    head = build_process_head(seed)
    payload: dict[str, Any] = {
        "recipe": "v542_process_head_anchor_v1",
        "method": PROCESS_HEAD_ANCHOR_METHOD,
        "seed": int(seed),
        "construction_steps": list(PROCESS_HEAD_ANCHOR_STEPS),
        "excludes_w_v": True,
        "anchor_state_sha256": state_dict_sha256(head.state_dict()),
    }
    return canonical_sha256(payload), payload


# ---------------------------------------------------------------------------
# assembly
# ---------------------------------------------------------------------------
def assemble_run_identity(
    config: Mapping[str, Any],
    *,
    candidate_id: str,
    run_seed: int,
    runplan_record: Mapping[str, Any] | None,
    budget_block: Mapping[str, Any] | None,
    root: Path | str | None = None,
    config_path: Path | str | None = None,
    source_manifest_path: Path | str | None = None,
    plan_path: Path | str | None = None,
    oracle_path: Path | str | None = None,
    require_clean: bool = False,
) -> dict[str, Any]:
    """Assemble the frozen 14-field run identity (+ optional ``code_commit``).

    Returns a dict with EXACTLY the 14 fields of
    :data:`IDENTITY_REQUIRED_FIELDS` plus ``code_commit`` (record-only, per
    ``git_commit_authoritative=false`` — it may be ``None``).  Every
    ``*_sha256`` value is a 64-hex digest; ``bootstrap_seed_root`` is the
    signed integer root.

    Typed refusals (all :class:`V542IdentityError`):

    * ``V542_IDENTITY_FIELD_MISSING`` — a required input (``runplan_record``
      / ``budget_block`` / ``config_path`` / …) is missing or ``None``;
    * ``V542_IDENTITY_CONFIG`` / ``V542_IDENTITY_CANDIDATE_UNKNOWN`` /
      ``V542_IDENTITY_SEED`` — malformed config / unknown candidate /
      bad run seed;
    * ``V542_IDENTITY_FILE_MISSING`` / ``V542_IDENTITY_ORACLE_SHA`` — an
      identity input file is absent, or the Oracle design source drifted
      from its WP0 pin;
    * ``V542_IDENTITY_FIELD_CONFLICT`` — ``bootstrap_seed_root`` disagrees
      across statistics/thresholds/runplan-pin, or the config's pinned
      oracle SHA disagrees with the frozen constant;
    * ``V542_IDENTITY_SAMPLER_DRIFT`` — the config's sampler spec SHA
      disagrees with the live frozen v541 strata spec;
    * ``V542_IDENTITY_SOURCE_DIRTY`` — ``require_clean`` demanded a clean
      source tree and the fingerprint says dirty.

    ``source_manifest_path``: when given, ``source_manifest_sha256`` is the
    file-bytes digest of that frozen manifest file; when omitted it is
    DERIVED from the live :func:`source_tree_fingerprint` (equal to the
    future file-bytes digest — the manifest file is defined to be the
    canonical bytes of the fingerprint payload).
    """
    cfg = _require_mapping(config, "config")
    base = Path(root) if root is not None else _default_repo_root()
    if not isinstance(candidate_id, str) or not candidate_id:
        raise V542IdentityError(
            "V542_IDENTITY_CANDIDATE_UNKNOWN",
            f"candidate_id must be a non-empty str, got {candidate_id!r}")
    if not isinstance(run_seed, int) or isinstance(run_seed, bool) \
            or run_seed < 0:
        raise V542IdentityError(
            "V542_IDENTITY_SEED",
            f"run_seed must be a non-negative int, got {run_seed!r}")
    if runplan_record is None:
        raise V542IdentityError(
            "V542_IDENTITY_FIELD_MISSING",
            "runplan_record is required (frozen runplan pin record)")
    if budget_block is None:
        raise V542IdentityError(
            "V542_IDENTITY_FIELD_MISSING",
            "budget_block is required (budget policy record; null budget "
            "VALUES stay null — the record itself must exist)")
    if config_path is None:
        raise V542IdentityError(
            "V542_IDENTITY_FIELD_MISSING",
            "config_path is required (config_sha256 hashes the file bytes)")

    # --- code identity layer ---------------------------------------------
    fingerprint_sha, fingerprint = source_tree_fingerprint(base)
    if require_clean and fingerprint["dirty"]:
        offenders = (fingerprint["modified_tracked"]
                     + fingerprint["missing_tracked"]
                     + fingerprint["untracked"])
        raise V542IdentityError(
            "V542_IDENTITY_SOURCE_DIRTY",
            "require_clean: the source tree is dirty under the frozen "
            f"identity_contract.code scope {list(FINGERPRINT_INCLUDES)} — "
            f"first offenders: {offenders[:8]}"
            + (" …" if len(offenders) > 8 else ""))
    if source_manifest_path is not None:
        source_manifest_sha, _ = source_manifest_sha256(
            source_manifest_path, root=base)
    else:
        source_manifest_sha = fingerprint_sha

    # --- config / shared protocol / environment ---------------------------
    config_digest, _ = config_sha256(config_path, root=base)
    common_sha, _ = common_protocol_sha256(cfg)
    environment_digest, _ = environment_sha256()
    backbone_sha, _ = backbone_spec_sha256(cfg, candidate_id)
    init_mapping_digest, _ = init_mapping_sha256(cfg)
    anchor_sha, _ = process_head_anchor_sha256(run_seed)
    statistics_sha, _ = statistics_contract_sha256(cfg)

    # --- bootstrap seed root (statistics == thresholds == runplan pin) ----
    statistics = _require_block(
        cfg, ("evaluation_contract", "statistics"))
    root_statistics = statistics.get("bootstrap_seed_root")
    root_thresholds = (cfg.get("thresholds") or {}).get(
        "bootstrap_seed_root")
    if not isinstance(root_statistics, int) or isinstance(root_statistics,
                                                          bool):
        raise V542IdentityError(
            "V542_IDENTITY_FIELD_MISSING",
            "evaluation_contract.statistics.bootstrap_seed_root must be an "
            f"int (got {root_statistics!r})")
    if root_thresholds != root_statistics:
        raise V542IdentityError(
            "V542_IDENTITY_FIELD_CONFLICT",
            f"bootstrap_seed_root disagrees: statistics="
            f"{root_statistics} vs thresholds={root_thresholds}")
    root_runplan = dict(runplan_record).get("bootstrap_seed_root")
    if root_runplan is not None and root_runplan != root_statistics:
        raise V542IdentityError(
            "V542_IDENTITY_FIELD_CONFLICT",
            f"bootstrap_seed_root disagrees: statistics="
            f"{root_statistics} vs runplan pin={root_runplan}")

    # --- pinned design sources -------------------------------------------
    plan_digest, _ = plan_sha256(plan_path, root=base)
    governance = _require_block(cfg, ("governance",))
    config_oracle_sha = governance.get("oracle_design_source_sha256")
    if config_oracle_sha is not None \
            and str(config_oracle_sha) != PINNED_ORACLE_DESIGN_SOURCE_SHA256:
        raise V542IdentityError(
            "V542_IDENTITY_ORACLE_SHA",
            "config governance.oracle_design_source_sha256 "
            f"({config_oracle_sha}) disagrees with the WP0-frozen pin "
            f"{PINNED_ORACLE_DESIGN_SOURCE_SHA256}")
    oracle_sha, _ = oracle_design_source_sha256(oracle_path, root=base)

    # --- runplan / budget / sampler --------------------------------------
    runplan_digest, _ = runplan_sha256(runplan_record)
    budget_digest, _ = budget_sha256(budget_block)
    binding = _require_block(
        cfg, ("training_contract", "sampler_binding"))
    sampler_sha = binding.get("spec_sha256")
    if not isinstance(sampler_sha, str) or len(sampler_sha) != 64:
        raise V542IdentityError(
            "V542_IDENTITY_FIELD_MISSING",
            "training_contract.sampler_binding.spec_sha256 must be a "
            f"64-hex digest (got {sampler_sha!r})")
    from pes2ts_core.pesn.offline.training.v541_strata import (  # lazy
        sampler_spec_sha256 as live_sampler_spec_sha256)
    live_sampler = live_sampler_spec_sha256()
    if sampler_sha != live_sampler:
        raise V542IdentityError(
            "V542_IDENTITY_SAMPLER_DRIFT",
            f"config sampler spec {sampler_sha} != live frozen v541 strata "
            f"spec {live_sampler}")

    identity: dict[str, Any] = {
        "source_manifest_sha256": source_manifest_sha,
        "config_sha256": config_digest,
        "common_protocol_sha256": common_sha,
        "environment_sha256": environment_digest,
        "backbone_spec_sha256": backbone_sha,
        "init_mapping_sha256": init_mapping_digest,
        "process_head_anchor_sha256": anchor_sha,
        "statistics_contract_sha256": statistics_sha,
        "bootstrap_seed_root": int(root_statistics),
        "plan_sha256": plan_digest,
        "oracle_design_source_sha256": oracle_sha,
        "runplan_sha256": runplan_digest,
        "budget_policy_sha256": budget_digest,
        "sampler_spec_sha256": sampler_sha,
        "code_commit": fingerprint["git_head"],  # record-only, may be None
    }
    # completeness invariant: no silent extra/missing fields
    expected = set(IDENTITY_REQUIRED_FIELDS) | {"code_commit"}
    if set(identity) != expected:  # pragma: no cover - construction guard
        raise V542IdentityError(
            "V542_IDENTITY_FIELD_MISSING",
            f"assembled identity field set drifted: "
            f"{sorted(set(identity) ^ expected)}")
    return identity
