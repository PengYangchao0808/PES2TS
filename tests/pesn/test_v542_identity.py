"""V5.4.2 WP3 lane A — run identity / canonical SHA certificate
(gate-0 advisory A3; ADR-0033 §2.6, config ``identity_contract``, plan §6).

Headline guarantees under test:

* the ONE canonical recipe — compact JSON, ``sort_keys=True``,
  ``separators=(",", ":")``, ``ensure_ascii=False``, UTF-8 bytes, SHA-256
  hex — deterministic, key-order independent, unicode-stable, with NO
  ordering/mtime/locale dependence;
* per-field builders return ``(sha256, canonical_payload)`` and are
  STABLE across two calls (config/plan/oracle file-bytes digests included);
* the common-protocol layer EXCLUDES every candidate-specific field:
  mutating a registry row / mapping table / zero-init row / run row /
  diagnostic flag cannot move ``common_protocol_sha256``, while mutating a
  shared field must;
* tamper detection: config / plan / oracle / statistics / init-mapping
  mutations all move their digests (and the oracle's WP0 pin refuses
  drift);
* the process-head anchor digest is RECONSTRUCTIBLE from the seed alone
  (``build_process_head(seed)`` + key-sorted tensor digest);
* ``assemble_run_identity`` returns exactly the 14 frozen required fields
  plus record-only ``code_commit``, and raises TYPED refusals for
  missing/None inputs, unknown candidates, and a dirty tree under
  ``require_clean``;
* the loader/parity ``mapping_table_sha256`` is now CANONICAL (no
  ``LOCAL_PLACEHOLDER:`` prefix) with UNCHANGED digest value (same table
  semantics) and UNCHANGED parity results.

Fixture/config-level only (the frozen campaign config + templates); no
training, no GPU run, no data access, no dev/confirm touch."""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from pes2ts_core.pesn.models.v542.identity import (  # noqa: E402
    COMMON_PROTOCOL_STRUCTURAL_EXCLUDES,
    DEFAULT_CAMPAIGN_CONFIG,
    DEFAULT_ORACLE_DESIGN_SOURCE,
    DEFAULT_PLAN_PATH,
    FINGERPRINT_EXCLUDED_DIRS,
    FINGERPRINT_EXCLUDED_FILES,
    FINGERPRINT_INCLUDE_DIRS,
    FINGERPRINT_INCLUDE_FILES,
    FINGERPRINT_INCLUDES,
    FINGERPRINT_SUFFIXES,
    IDENTITY_REQUIRED_FIELDS,
    PINNED_ORACLE_DESIGN_SOURCE_SHA256,
    RUN_IDENTITY_SCHEMA,
    V542IdentityError,
    assemble_run_identity,
    backbone_spec_sha256,
    budget_sha256,
    canonical_bytes,
    canonical_sha256,
    common_protocol_sha256,
    config_sha256,
    environment_sha256,
    init_mapping_sha256,
    oracle_design_source_sha256,
    plan_sha256,
    process_head_anchor_sha256,
    runplan_sha256,
    source_manifest_sha256,
    source_tree_fingerprint,
    state_dict_sha256,
    statistics_contract_sha256,
)

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("yaml") is None,
    reason="PyYAML required to read the frozen campaign config")

PROJECT_ROOT = Path(__file__).resolve().parents[2]

#: Frozen candidate/seed pair used across the suite (run seeds: 17/29/43).
CANDIDATE = "A1"
RUN_SEED = 17

_RUNPLAN_TEMPLATE = PROJECT_ROOT / (
    "config/campaigns/v542_templates/runplan_pin.template.json")
_BUDGET_TEMPLATE = PROJECT_ROOT / (
    "config/campaigns/v542_templates/budget_policy.template.json")


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def cfg() -> dict:
    import yaml

    path = PROJECT_ROOT / DEFAULT_CAMPAIGN_CONFIG
    if not path.is_file():  # pragma: no cover - config is frozen
        pytest.skip(f"frozen config unavailable: {path}")
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    return loaded


@pytest.fixture(scope="module")
def runplan_record() -> dict:
    if not _RUNPLAN_TEMPLATE.is_file():  # pragma: no cover - frozen
        pytest.skip(f"runplan template unavailable: {_RUNPLAN_TEMPLATE}")
    return json.loads(_RUNPLAN_TEMPLATE.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def budget_block() -> dict:
    if not _BUDGET_TEMPLATE.is_file():  # pragma: no cover - frozen
        pytest.skip(f"budget template unavailable: {_BUDGET_TEMPLATE}")
    return json.loads(_BUDGET_TEMPLATE.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# 1. canonicalization: determinism + key-order independence + unicode
# ---------------------------------------------------------------------------
def test_canonical_bytes_is_the_frozen_recipe() -> None:
    assert canonical_bytes({"b": 1, "a": 2}) == b'{"a":2,"b":1}'
    assert canonical_bytes({}) == b"{}"


def test_canonical_sha256_key_order_independent() -> None:
    left = {"alpha": 1, "beta": {"x": True, "y": [1, 2, 3]}, "gamma": None}
    right = {"gamma": None, "beta": {"y": [1, 2, 3], "x": True}, "alpha": 1}
    assert canonical_sha256(left) == canonical_sha256(right)
    # nested key order also independent
    assert canonical_sha256({"a": {"b": 1, "c": 2}}) == \
        canonical_sha256({"a": {"c": 2, "b": 1}})
    # but content changes DO change the digest
    assert canonical_sha256(left) != canonical_sha256(
        {"alpha": 1, "beta": {"x": True, "y": [1, 2, 4]}, "gamma": None})


def test_canonical_sha256_deterministic_and_utf8() -> None:
    payload = {"note": "统一接口", "z": 64, "edge_msg": 48.5}
    digest = canonical_sha256(payload)
    assert digest == hashlib_sha256_hex(
        json.dumps(payload, sort_keys=True, separators=(",", ":"),
                   ensure_ascii=False).encode("utf-8"))
    # unicode is NOT \u-escaped (ensure_ascii=False) — raw UTF-8 bytes
    assert "统一接口".encode("utf-8") in canonical_bytes(payload)
    assert canonical_sha256(payload) == canonical_sha256(
        copy.deepcopy(payload))


def test_canonical_sha256_typed_refusal_on_unserializable() -> None:
    with pytest.raises(V542IdentityError) as excinfo:
        canonical_sha256({"bad": object()})
    assert excinfo.value.code == "V542_IDENTITY_SERIALIZE"


def hashlib_sha256_hex(data: bytes) -> str:
    import hashlib

    return hashlib.sha256(data).hexdigest()


# ---------------------------------------------------------------------------
# 2. common-protocol exclusion list behaviour
# ---------------------------------------------------------------------------
def test_common_protocol_excludes_candidate_specific_fields(cfg) -> None:
    baseline, payload = common_protocol_sha256(cfg)
    # the exclusion list is documented inside the hashed payload
    for excluded in COMMON_PROTOCOL_STRUCTURAL_EXCLUDES:
        assert excluded in payload["excludes"]

    # mutating a candidate-specific field MUST NOT move the digest
    mutated = copy.deepcopy(cfg)
    mutated["backbone_contract"]["candidates"]["A1"]["hidden"] = 999
    assert common_protocol_sha256(mutated)[0] == baseline

    mutated = copy.deepcopy(cfg)
    mutated["training_contract"]["init_policy"]["init_mapping"][
        "r3_semantic_mapping"]["A1"] = "totally different mapping"
    assert common_protocol_sha256(mutated)[0] == baseline

    mutated = copy.deepcopy(cfg)
    mutated["training_contract"]["init_policy"]["per_candidate_zero_init"][
        "A5"] = "no longer zero"
    assert common_protocol_sha256(mutated)[0] == baseline

    mutated = copy.deepcopy(cfg)
    mutated["run_matrix"]["runs"][0]["backbone"] = "A9"
    assert common_protocol_sha256(mutated)[0] == baseline

    mutated = copy.deepcopy(cfg)
    mutated["diagnostic"]["updates"] = 999999  # diagnostic-only flag block
    assert common_protocol_sha256(mutated)[0] == baseline


def test_common_protocol_mutating_shared_field_changes_digest(cfg) -> None:
    baseline, _ = common_protocol_sha256(cfg)
    for mutate in (
        lambda c: c["training_contract"]["schedule"].__setitem__(
            "total_updates", 12345),
        lambda c: c["training_contract"]["lambdas"].__setitem__(
            "lambda_rank", 0.111),
        lambda c: c["split_contract"].__setitem__("inner_val_max_samples",
                                                  1024),
        lambda c: c["training_contract"]["sampler_binding"].__setitem__(
            "batch_curves", 16),
    ):
        mutated = copy.deepcopy(cfg)
        mutate(mutated)
        assert common_protocol_sha256(mutated)[0] != baseline


def test_common_protocol_same_across_candidates(cfg) -> None:
    # the common layer is candidate-INdependent by construction (it never
    # reads the candidate id at all) — one digest for the whole campaign
    digest, payload = common_protocol_sha256(cfg)
    assert digest == canonical_sha256(payload)


# ---------------------------------------------------------------------------
# 3. every builder stable across two calls (+ unicode config content)
# ---------------------------------------------------------------------------
def test_file_builders_stable_and_detect_tamper(tmp_path, cfg,
                                                runplan_record,
                                                budget_block) -> None:
    config_path = PROJECT_ROOT / DEFAULT_CAMPAIGN_CONFIG
    # stable across two calls
    assert config_sha256(config_path) == config_sha256(config_path)
    assert plan_sha256() == plan_sha256()
    assert oracle_design_source_sha256() == oracle_design_source_sha256()
    assert runplan_sha256(runplan_record) == runplan_sha256(runplan_record)
    assert budget_sha256(budget_block) == budget_sha256(budget_block)
    assert common_protocol_sha256(cfg) == common_protocol_sha256(cfg)
    assert environment_sha256() == environment_sha256()
    assert backbone_spec_sha256(cfg, CANDIDATE) == \
        backbone_spec_sha256(cfg, CANDIDATE)
    assert init_mapping_sha256(cfg) == init_mapping_sha256(cfg)
    assert statistics_contract_sha256(cfg) == \
        statistics_contract_sha256(cfg)

    # config tamper detection (file bytes)
    tampered = tmp_path / "config.yaml"
    tampered.write_bytes(config_path.read_bytes() + b"\n# tampered\n")
    assert config_sha256(tampered)[0] != config_sha256(config_path)[0]

    # unicode: a config with non-ASCII content hashes its UTF-8 bytes
    unicode_cfg = tmp_path / "unicode.yaml"
    unicode_cfg.write_text("note: 统一 Backbone 接口\n", encoding="utf-8")
    same = tmp_path / "unicode_copy.yaml"
    same.write_text("note: 统一 Backbone 接口\n", encoding="utf-8")
    assert config_sha256(unicode_cfg)[0] == config_sha256(same)[0]
    escaped = tmp_path / "unicode_escaped.yaml"
    escaped.write_bytes(json.dumps(
        {"note": "统一 Backbone 接口"}, ensure_ascii=True).encode("utf-8"))
    assert config_sha256(escaped)[0] != config_sha256(unicode_cfg)[0]


def test_config_payload_records_reconstruction_path(tmp_path) -> None:
    config_path = PROJECT_ROOT / DEFAULT_CAMPAIGN_CONFIG
    sha, record = config_sha256(config_path, root=PROJECT_ROOT)
    assert record["recipe"] == "file_bytes_sha256"
    assert record["path"] == DEFAULT_CAMPAIGN_CONFIG
    assert record["sha256"] == sha
    assert record["n_bytes"] == config_path.stat().st_size


def test_statistics_and_init_mapping_tamper_detection(cfg) -> None:
    baseline_stats = statistics_contract_sha256(cfg)[0]
    baseline_init = init_mapping_sha256(cfg)[0]

    mutated = copy.deepcopy(cfg)
    mutated["evaluation_contract"]["statistics"]["holm_family"][
        "move_direction_exit"]["alpha"] = 0.5
    assert statistics_contract_sha256(mutated)[0] != baseline_stats

    mutated = copy.deepcopy(cfg)
    mutated["thresholds"]["h3_process_delta_auprc_abs_min"] = 0.2
    assert statistics_contract_sha256(mutated)[0] != baseline_stats

    mutated = copy.deepcopy(cfg)
    mutated["training_contract"]["init_policy"]["construction_order"][
        0] = "reordered_step"
    assert init_mapping_sha256(mutated)[0] != baseline_init

    mutated = copy.deepcopy(cfg)
    mutated["training_contract"]["init_policy"]["candidate_fresh_init"][
        "seed_recipe"] = "int(sha256('v542|{run_seed}').digest()[:8],'big')"
    assert init_mapping_sha256(mutated)[0] != baseline_init


def test_oracle_design_source_pin_verifies_and_refuses_drift(
        tmp_path) -> None:
    sha, record = oracle_design_source_sha256()
    assert sha == PINNED_ORACLE_DESIGN_SOURCE_SHA256
    assert record["verified_against_pin"] is True

    tampered = tmp_path / "oracle-architecture.md"
    tampered.write_text("# drifted design source\n", encoding="utf-8")
    with pytest.raises(V542IdentityError) as excinfo:
        oracle_design_source_sha256(tampered)
    assert excinfo.value.code == "V542_IDENTITY_ORACLE_SHA"

    missing = tmp_path / "absent.md"
    with pytest.raises(V542IdentityError) as excinfo:
        oracle_design_source_sha256(missing)
    assert excinfo.value.code == "V542_IDENTITY_FILE_MISSING"


def test_plan_sha256_hashes_pinned_plan_document() -> None:
    sha, record = plan_sha256(root=PROJECT_ROOT)
    assert record["path"] == DEFAULT_PLAN_PATH
    assert record["sha256"] == sha
    assert record["recipe"] == "file_bytes_sha256"
    plan_path = PROJECT_ROOT / DEFAULT_PLAN_PATH
    assert record["n_bytes"] == plan_path.stat().st_size


def test_source_manifest_file_equals_fingerprint_derivation(
        tmp_path) -> None:
    """Persisting the fingerprint payload as the manifest FILE makes the
    file-bytes digest equal the derived fingerprint digest."""
    sha, payload = source_tree_fingerprint(PROJECT_ROOT)
    manifest = tmp_path / "source_manifest.json"
    manifest.write_bytes(canonical_bytes(payload))
    assert source_manifest_sha256(manifest)[0] == sha
    # and re-deriving from the SAME tree is stable
    assert source_tree_fingerprint(PROJECT_ROOT)[0] == sha


# ---------------------------------------------------------------------------
# 4. source tree fingerprint: frozen identity_contract.code scope +
#    determinism + dirty accounting
# ---------------------------------------------------------------------------
def _git_blob_sha1(content: bytes) -> str:
    """Git object SHA-1 of a blob (mirrors identity._git_blob_sha1)."""
    import hashlib

    h = hashlib.sha1()
    h.update(f"blob {len(content)}\0".encode("ascii"))
    h.update(content)
    return h.hexdigest()


def _write_git_index(path: Path, entries: "dict[str, bytes]") -> None:
    """Hand-craft a git v2 index (no subprocess) mapping path → content.

    Layout per entry (exactly what ``identity._git_index_entries``
    parses): 40-byte stat block, 20-byte blob SHA-1, 2-byte flags (name
    length in the low 12 bits), NUL-terminated name, padded to an
    8-byte boundary.  The trailing index checksum is not verified by the
    parser and is therefore omitted.
    """
    import struct

    blobs = sorted(entries.items())
    data = bytearray(b"DIRC")
    data += struct.pack(">II", 2, len(blobs))
    for name, content in blobs:
        name_bytes = name.encode("utf-8")
        entry = bytearray(40)                      # stat block: all zero
        entry += bytes.fromhex(_git_blob_sha1(content))
        entry += struct.pack(">H", len(name_bytes) & 0x0FFF)
        entry += name_bytes + b"\0"
        pad = (8 - (len(entry) % 8)) % 8
        entry += b"\0" * pad
        data += entry
    path.write_bytes(bytes(data))


def _make_scope_repo(root: Path) -> "dict[str, bytes]":
    """A hermetic miniature checkout exercising the FULL frozen scope."""
    files: "dict[str, bytes]" = {
        "requirements.txt": b"torch==2.0.0\n",
        "requirements-pesn-training.txt": b"numpy==1.24.0\n",
        "requirements-pesn-runtime.txt": b"pyyaml==6.0\n",
        "config/campaigns/pes2ts_v542_backbone_50k.yaml": b"schema: v542\n",
        "config/campaigns/v542_templates/budget_policy.template.json": b"{}\n",
        "pes2ts_core/mod.py": b"X = 1\n",
        "scripts/campaigns/run.py": b"Y = 2\n",
        "scripts/agent_ops/tool.py": b"Z = 3\n",
        "scripts/agent_ops/docs_consistency_config.json": b'{"old": 1}\n',
        "scripts/infra/infra.py": b"W = 4\n",
    }
    for rel, content in files.items():
        target = root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
    git = root / ".git"
    (git / "refs" / "heads").mkdir(parents=True)
    (git / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
    (git / "refs" / "heads" / "main").write_text("a" * 40 + "\n",
                                                 encoding="utf-8")
    _write_git_index(git / "index", files)
    return files


def test_fingerprint_scope_mirrors_the_frozen_identity_contract_code(cfg):
    """The implementation constants EQUAL the frozen config block — the
    scope is read LITERALLY (oracle gate-3 B1): includes (dirs + files),
    suffixes (incl. ``.txt``) and excludes (dirs + the
    ``docs_consistency_config.json`` file)."""
    import pes2ts_core.pesn.models.v542.identity as identity_mod

    code = cfg["identity_contract"]["code"]
    includes = [str(x) for x in code["includes"]]
    assert sorted(identity_mod.FINGERPRINT_INCLUDES) == sorted(includes)
    assert sorted(identity_mod.FINGERPRINT_INCLUDE_DIRS
                  + identity_mod.FINGERPRINT_INCLUDE_FILES) == \
        sorted(includes)
    assert list(identity_mod.FINGERPRINT_SUFFIXES) == \
        [str(s) for s in code["suffixes"]]
    excludes = {str(x) for x in code["excludes"]}
    assert (set(identity_mod.FINGERPRINT_EXCLUDED_DIRS)
            | set(identity_mod.FINGERPRINT_EXCLUDED_FILES)) == excludes
    # the frozen requirements files and the docs-consistency exclusion
    # are exactly where the config puts them
    assert identity_mod.FINGERPRINT_EXCLUDED_FILES == \
        {"docs_consistency_config.json"}
    for req in ("requirements.txt", "requirements-pesn-training.txt",
                "requirements-pesn-runtime.txt"):
        assert req in identity_mod.FINGERPRINT_INCLUDE_FILES


def test_source_tree_fingerprint_deterministic_and_structured() -> None:
    sha, payload = source_tree_fingerprint(PROJECT_ROOT)
    sha_again, payload_again = source_tree_fingerprint(PROJECT_ROOT)
    assert sha == sha_again
    assert payload == payload_again
    assert payload["recipe"] == "pes2ts_v542_source_tree_fingerprint_v1"
    assert len(payload["git_head"]) == 40
    assert isinstance(payload["dirty"], bool)
    assert payload["includes"] == sorted(FINGERPRINT_INCLUDES)
    assert payload["suffixes"] == [".json", ".py", ".txt", ".yaml"]
    assert payload["excluded_dirs"] == sorted(FINGERPRINT_EXCLUDED_DIRS)
    assert payload["excluded_files"] == ["docs_consistency_config.json"]
    assert payload["n_files"] == len(payload["files"]) > 0
    paths = [entry["path"] for entry in payload["files"]]
    assert paths == sorted(paths)
    for entry in payload["files"]:
        rel = entry["path"]
        # every manifest path is inside the frozen include scope …
        assert (rel in FINGERPRINT_INCLUDE_FILES
                or rel.startswith(tuple(d + "/"
                                        for d in FINGERPRINT_INCLUDE_DIRS)))
        # … and never inside an excluded directory or is an excluded file
        assert not any(part in FINGERPRINT_EXCLUDED_DIRS
                       for part in rel.split("/"))
        assert rel.split("/")[-1] not in FINGERPRINT_EXCLUDED_FILES
        assert len(entry["sha256"]) == 64
    # the three frozen requirements files ARE part of the manifest (B1)
    for req in ("requirements.txt", "requirements-pesn-training.txt",
                "requirements-pesn-runtime.txt"):
        assert req in paths
    # the config file itself is part of the manifest
    assert "config/campaigns/pes2ts_v542_backbone_50k.yaml" in paths
    # the docs-consistency tooling config is EXCLUDED (B1)
    assert "scripts/agent_ops/docs_consistency_config.json" not in paths
    assert "scripts/agent_ops/docs_consistency_config.json" not in \
        payload["modified_tracked"]
    assert "scripts/agent_ops/docs_consistency_config.json" not in \
        payload["untracked"]
    # EVERY frozen suffix is represented in the manifest
    for suffix in FINGERPRINT_SUFFIXES:
        assert any(p.endswith(suffix) for p in paths), suffix
    # a known TRACKED module is part of the manifest
    assert "pes2ts_core/pesn/models/v542/loader.py" in paths
    # the new identity module is untracked until committed: it must show
    # up either in the tracked manifest (post-commit) or in `untracked`
    # (pre-commit) — never silently absent
    assert ("pes2ts_core/pesn/models/v542/identity.py" in paths) or (
        "pes2ts_core/pesn/models/v542/identity.py" in payload["untracked"])
    assert sha == canonical_sha256(payload)


def test_source_tree_fingerprint_no_subprocess_guarantee() -> None:
    """The fingerprint implementation never spawns a child process — the
    git state is read from ``.git`` files directly (truth-guard rule)."""
    import ast

    import pes2ts_core.pesn.models.v542.identity as identity_mod

    tree = ast.parse(Path(identity_mod.__file__).read_text(
        encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            assert all(alias.name.split(".")[0] != "subprocess"
                       for alias in node.names)
        if isinstance(node, ast.ImportFrom):
            assert (node.module or "").split(".")[0] != "subprocess"


def test_dirty_requirements_file_flips_the_fingerprint(tmp_path) -> None:
    """A dirty tracked ``requirements-*.txt`` flips ``dirty``, lands in
    ``modified_tracked`` AND moves the fingerprint digest — the frozen
    dependency files are first-class identity inputs (oracle gate-3 B1);
    an edit to the EXCLUDED ``docs_consistency_config.json`` does NOT."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _make_scope_repo(repo)

    baseline_sha, baseline = source_tree_fingerprint(repo)
    assert baseline["dirty"] is False
    paths = [e["path"] for e in baseline["files"]]
    for req in ("requirements.txt", "requirements-pesn-training.txt",
                "requirements-pesn-runtime.txt"):
        assert req in paths
    # every configured extension is represented in this synthetic repo too
    for suffix in FINGERPRINT_SUFFIXES:
        assert any(p.endswith(suffix) for p in paths), suffix
    # determinism on the synthetic repo
    assert source_tree_fingerprint(repo) == (baseline_sha, baseline)

    # --- dirty a tracked requirements file → fingerprint MUST move ------
    dirty_path = repo / "requirements-pesn-training.txt"
    dirty_path.write_bytes(b"numpy==1.26.0\n")   # dependency bump
    dirty_sha, dirty = source_tree_fingerprint(repo)
    assert dirty["dirty"] is True
    assert "requirements-pesn-training.txt" in dirty["modified_tracked"]
    assert dirty_sha != baseline_sha
    entry = next(e for e in dirty["files"]
                 if e["path"] == "requirements-pesn-training.txt")
    import hashlib as _hashlib
    assert entry["sha256"] == _hashlib.sha256(
        b"numpy==1.26.0\n").hexdigest()

    # … and restoring the exact bytes restores the digest (determinism)
    dirty_path.write_bytes(b"numpy==1.24.0\n")
    assert source_tree_fingerprint(repo)[0] == baseline_sha

    # --- the EXCLUDED docs_consistency_config.json NEVER counts ---------
    doc = repo / "scripts" / "agent_ops" / "docs_consistency_config.json"
    doc.write_bytes(b'{"mutated": true}\n')
    doc_sha, doc_payload = source_tree_fingerprint(repo)
    assert doc_sha == baseline_sha              # digest unchanged
    assert doc_payload["dirty"] is False        # not dirty
    assert "scripts/agent_ops/docs_consistency_config.json" not in \
        [e["path"] for e in doc_payload["files"]]

    # --- an untracked in-scope file with a frozen suffix flips dirty ----
    (repo / "pes2ts_core" / "new_module.py").write_text(
        "NEW = 5\n", encoding="utf-8")
    untracked_sha, untracked = source_tree_fingerprint(repo)
    assert untracked["dirty"] is True
    assert "pes2ts_core/new_module.py" in untracked["untracked"]
    assert untracked_sha != baseline_sha

    # --- a missing tracked in-scope file flips dirty --------------------
    (repo / "pes2ts_core" / "new_module.py").unlink()
    (repo / "scripts" / "infra" / "infra.py").unlink()
    _, missing = source_tree_fingerprint(repo)
    assert missing["dirty"] is True
    assert "scripts/infra/infra.py" in missing["missing_tracked"]


def test_source_tree_fingerprint_dirty_refusal_paths(tmp_path) -> None:
    # a directory that is not a git checkout is a typed refusal
    with pytest.raises(V542IdentityError) as excinfo:
        source_tree_fingerprint(tmp_path)
    assert excinfo.value.code == "V542_IDENTITY_GIT"


# ---------------------------------------------------------------------------
# 5. environment: stable + deterministic flags recorded
# ---------------------------------------------------------------------------
def test_environment_sha256_stable_and_complete() -> None:
    sha, payload = environment_sha256()
    assert sha == environment_sha256()[0]
    assert payload["recipe"] == "v542_environment_v1"
    assert payload["python"] and payload["torch"]
    assert isinstance(payload["cuda_available"], bool)
    assert isinstance(payload["device_names"], list)
    assert set(payload["deterministic"]) == {
        "cudnn_deterministic", "cudnn_benchmark",
        "deterministic_algorithms", "cudnn_allow_tf32",
        "cuda_matmul_allow_tf32"}
    assert sha == canonical_sha256(payload)


# ---------------------------------------------------------------------------
# 6. process-head anchor: reconstructible from the seed alone
# ---------------------------------------------------------------------------
def test_process_head_anchor_reconstructible_from_seed() -> None:
    sha, payload = process_head_anchor_sha256(RUN_SEED)
    assert sha == process_head_anchor_sha256(RUN_SEED)[0]
    assert payload["method"] == "fork_rng_replay"
    assert payload["seed"] == RUN_SEED
    assert payload["excludes_w_v"] is True
    assert payload["construction_steps"]

    # reconstruction: replay build_process_head(seed) and re-derive
    from pes2ts_core.pesn.models.v542.candidates.common import (
        build_process_head,
    )
    head = build_process_head(RUN_SEED)
    assert payload["anchor_state_sha256"] == state_dict_sha256(
        head.state_dict())
    rebuilt = dict(payload)
    assert canonical_sha256(rebuilt) == sha

    # a different seed gives a different anchor
    assert process_head_anchor_sha256(29)[0] != sha

    # typed refusal for a bad seed
    with pytest.raises(V542IdentityError) as excinfo:
        process_head_anchor_sha256(-1)
    assert excinfo.value.code == "V542_IDENTITY_SEED"


def test_state_dict_sha256_key_order_and_content() -> None:
    sd = {"b.weight": torch.zeros(2), "a.weight": torch.ones(3)}
    sha = state_dict_sha256(sd)
    assert sha == state_dict_sha256(dict(reversed(list(sd.items()))))
    mutated = {"b.weight": torch.zeros(2), "a.weight": torch.ones(3) * 2}
    assert state_dict_sha256(mutated) != sha


# ---------------------------------------------------------------------------
# 7. assemble_run_identity: completeness + typed refusals
# ---------------------------------------------------------------------------
def _assemble(cfg, runplan_record, budget_block, **kwargs):
    return assemble_run_identity(
        cfg, candidate_id=CANDIDATE, run_seed=RUN_SEED,
        runplan_record=runplan_record, budget_block=budget_block,
        root=PROJECT_ROOT,
        config_path=PROJECT_ROOT / DEFAULT_CAMPAIGN_CONFIG, **kwargs)


def test_assemble_completeness_and_field_set(cfg, runplan_record,
                                             budget_block) -> None:
    identity = _assemble(cfg, runplan_record, budget_block)
    assert set(identity) == set(IDENTITY_REQUIRED_FIELDS) | {"code_commit"}
    # 14 required fields exactly, in the frozen order when listed
    assert tuple(k for k in identity if k != "code_commit") == \
        IDENTITY_REQUIRED_FIELDS
    for field in IDENTITY_REQUIRED_FIELDS:
        if field == "bootstrap_seed_root":
            assert identity[field] == 20261029
            continue
        value = identity[field]
        assert isinstance(value, str) and len(value) == 64, field
        assert all(c in "0123456789abcdef" for c in value), field
    # code_commit is record-only: a 40-hex HEAD here (git checkout)
    assert isinstance(identity["code_commit"], str)
    assert len(identity["code_commit"]) == 40
    # stability: assembling twice gives the identical record
    assert identity == _assemble(cfg, runplan_record, budget_block)


def test_assemble_schema_and_candidate_specificity(cfg, runplan_record,
                                                   budget_block) -> None:
    identity_a1 = _assemble(cfg, runplan_record, budget_block)
    identity_a2 = assemble_run_identity(
        cfg, candidate_id="A2", run_seed=RUN_SEED,
        runplan_record=runplan_record, budget_block=budget_block,
        root=PROJECT_ROOT,
        config_path=PROJECT_ROOT / DEFAULT_CAMPAIGN_CONFIG)
    # candidate-specific layers differ; shared layers are identical
    assert identity_a1["backbone_spec_sha256"] != \
        identity_a2["backbone_spec_sha256"]
    for shared in ("common_protocol_sha256", "config_sha256",
                   "environment_sha256", "statistics_contract_sha256",
                   "init_mapping_sha256", "plan_sha256",
                   "oracle_design_source_sha256", "runplan_sha256",
                   "budget_policy_sha256", "sampler_spec_sha256",
                   "source_manifest_sha256"):
        assert identity_a1[shared] == identity_a2[shared], shared
    # seed-specific layer differs
    identity_s29 = assemble_run_identity(
        cfg, candidate_id=CANDIDATE, run_seed=29,
        runplan_record=runplan_record, budget_block=budget_block,
        root=PROJECT_ROOT,
        config_path=PROJECT_ROOT / DEFAULT_CAMPAIGN_CONFIG)
    assert identity_s29["process_head_anchor_sha256"] != \
        identity_a1["process_head_anchor_sha256"]


def test_assemble_typed_refusals(cfg, runplan_record, budget_block) -> None:
    # missing runplan record -> typed refusal
    with pytest.raises(V542IdentityError) as excinfo:
        assemble_run_identity(
            cfg, candidate_id=CANDIDATE, run_seed=RUN_SEED,
            runplan_record=None, budget_block=budget_block,
            root=PROJECT_ROOT,
            config_path=PROJECT_ROOT / DEFAULT_CAMPAIGN_CONFIG)
    assert excinfo.value.code == "V542_IDENTITY_FIELD_MISSING"

    # missing budget block -> typed refusal
    with pytest.raises(V542IdentityError) as excinfo:
        assemble_run_identity(
            cfg, candidate_id=CANDIDATE, run_seed=RUN_SEED,
            runplan_record=runplan_record, budget_block=None,
            root=PROJECT_ROOT,
            config_path=PROJECT_ROOT / DEFAULT_CAMPAIGN_CONFIG)
    assert excinfo.value.code == "V542_IDENTITY_FIELD_MISSING"

    # unknown candidate -> typed refusal
    with pytest.raises(V542IdentityError) as excinfo:
        assemble_run_identity(
            cfg, candidate_id="A404", run_seed=RUN_SEED,
            runplan_record=runplan_record, budget_block=budget_block,
            root=PROJECT_ROOT,
            config_path=PROJECT_ROOT / DEFAULT_CAMPAIGN_CONFIG)
    assert excinfo.value.code == "V542_IDENTITY_CANDIDATE_UNKNOWN"

    # bad seed -> typed refusal
    with pytest.raises(V542IdentityError) as excinfo:
        assemble_run_identity(
            cfg, candidate_id=CANDIDATE, run_seed="17",
            runplan_record=runplan_record, budget_block=budget_block,
            root=PROJECT_ROOT)
    assert excinfo.value.code == "V542_IDENTITY_SEED"

    # None config path -> typed refusal
    with pytest.raises(V542IdentityError) as excinfo:
        assemble_run_identity(
            cfg, candidate_id=CANDIDATE, run_seed=RUN_SEED,
            runplan_record=runplan_record, budget_block=budget_block,
            root=PROJECT_ROOT, config_path=None)
    assert excinfo.value.code == "V542_IDENTITY_FIELD_MISSING"


def test_assemble_require_clean_refusal(cfg, runplan_record,
                                        budget_block) -> None:
    """require_clean refuses a dirty source tree; the CLEAN path is proved
    by monkeypatching the fingerprint to a clean synthetic payload."""
    import pes2ts_core.pesn.models.v542.identity as identity_mod

    sha, payload = source_tree_fingerprint(PROJECT_ROOT)
    if payload["dirty"]:
        with pytest.raises(V542IdentityError) as excinfo:
            _assemble(cfg, runplan_record, budget_block, require_clean=True)
        assert excinfo.value.code == "V542_IDENTITY_SOURCE_DIRTY"

    clean = dict(payload)
    clean["dirty"] = False
    clean["modified_tracked"] = []
    clean["missing_tracked"] = []
    clean["untracked"] = []
    original = identity_mod.source_tree_fingerprint
    try:
        identity_mod.source_tree_fingerprint = \
            lambda root=None: (canonical_sha256(clean), clean)
        identity = _assemble(cfg, runplan_record, budget_block,
                             require_clean=True)
        assert set(identity) == set(IDENTITY_REQUIRED_FIELDS) | {
            "code_commit"}
    finally:
        identity_mod.source_tree_fingerprint = original


def test_assemble_bootstrap_root_conflict(cfg, runplan_record,
                                          budget_block) -> None:
    conflicted = copy.deepcopy(runplan_record)
    conflicted["bootstrap_seed_root"] = 1
    with pytest.raises(V542IdentityError) as excinfo:
        assemble_run_identity(
            cfg, candidate_id=CANDIDATE, run_seed=RUN_SEED,
            runplan_record=conflicted, budget_block=budget_block,
            root=PROJECT_ROOT,
            config_path=PROJECT_ROOT / DEFAULT_CAMPAIGN_CONFIG)
    assert excinfo.value.code == "V542_IDENTITY_FIELD_CONFLICT"


def test_cli_identity_record_schema(cfg, runplan_record, budget_block,
                                    tmp_path) -> None:
    """The CLI builder produces the printed record shape (schema +
    identity + payloads) without refusal."""
    import scripts.agent_ops.v542_identity as identity_cli

    record = identity_cli.build_identity_record(
        CANDIDATE, RUN_SEED,
        config_path=PROJECT_ROOT / DEFAULT_CAMPAIGN_CONFIG,
        runplan_path=_RUNPLAN_TEMPLATE,
        budget_path=_BUDGET_TEMPLATE,
        source_manifest_path=None,
        require_clean=False,
        root=PROJECT_ROOT)
    assert record["schema"] == RUN_IDENTITY_SCHEMA
    assert record["candidate_id"] == CANDIDATE
    assert record["run_seed"] == RUN_SEED
    assert set(record["identity"]) == set(IDENTITY_REQUIRED_FIELDS) | {
        "code_commit"}
    assert set(record["payloads"]) >= {
        "source_tree_fingerprint", "config", "common_protocol",
        "environment", "backbone_spec", "init_mapping",
        "process_head_anchor", "statistics_contract", "plan",
        "oracle_design_source", "runplan", "budget_policy"}
    # every assembled digest matches its builder payload record
    assert record["identity"]["config_sha256"] == \
        record["payloads"]["config"]["sha256"]
    assert record["identity"]["oracle_design_source_sha256"] == \
        PINNED_ORACLE_DESIGN_SOURCE_SHA256


# ---------------------------------------------------------------------------
# 8. loader / parity digest: canonical (not placeholder), UNCHANGED value
# ---------------------------------------------------------------------------
def test_loader_mapping_digest_is_canonical_and_unchanged() -> None:
    """The loader's mapping_table_sha256 is now the bare canonical hex —
    the SAME digest value the old LOCAL_PLACEHOLDER recipe produced (same
    table semantics), only the prefix is gone; parity results are
    therefore unchanged."""
    import hashlib
    import json as _json

    from pes2ts_core.pesn.models.key_atom_heads_v3 import (
        KeyAtomV53MPModel,
    )
    from pes2ts_core.pesn.models.v542 import V542Model, load_a0_strict
    from pes2ts_core.pesn.models.v542.loader import (
        _mapping_table_digest,
        a0_mapping_table,
    )

    table = a0_mapping_table()
    digest = _mapping_table_digest(table)
    # canonical: 64 lowercase hex, NO placeholder prefix
    assert len(digest) == 64
    assert all(c in "0123456789abcdef" for c in digest)
    assert not digest.startswith("LOCAL_PLACEHOLDER:")
    # unchanged value: the old recipe over the same table semantics
    legacy_payload = _json.dumps({k: list(v) for k, v in
                                  sorted(table.items())},
                                 sort_keys=True, separators=(",", ":"))
    assert digest == hashlib.sha256(
        legacy_payload.encode("utf-8")).hexdigest()
    # deterministic
    assert _mapping_table_digest(a0_mapping_table()) == digest

    # full strict-load audit surfaces the canonical digest and UNCHANGED
    # results (43/45/dual-write/empty categories)
    torch.manual_seed(RUN_SEED)
    legacy = KeyAtomV53MPModel()
    v542 = V542Model()
    audit = load_a0_strict(v542, {"model": legacy.state_dict()})
    assert audit["ok"] is True and audit["strict"] is True
    assert audit["n_source_keys"] == 43
    assert audit["n_mapped_targets"] == 45
    assert audit["missing"] == [] and audit["unexpected"] == []
    assert audit["mapping_table_sha256"] == digest
    assert not str(audit["mapping_table_sha256"]).startswith(
        "LOCAL_PLACEHOLDER:")


def test_parity_audit_record_surfaces_canonical_digest(tmp_path) -> None:
    """The parity harness audit record passes the canonical digest through
    (no LOCAL_PLACEHOLDER prefix in the record)."""
    from pes2ts_core.pesn.models.key_atom_heads_v3 import KeyAtomV53MPModel
    from pes2ts_core.pesn.models.v542 import V542Model, load_a0_strict
    from scripts.agent_ops.v542_a0_parity import _asset_load_audit_record

    torch.manual_seed(RUN_SEED)
    v542 = V542Model()
    audit = load_a0_strict(
        v542, {"model": KeyAtomV53MPModel().state_dict()})

    doctor = tmp_path / "synthetic.pt"
    doctor.write_bytes(b"synthetic-bytes-for-digest-only")
    record = _asset_load_audit_record(
        {"id": "SYNTHETIC", "path": "synthetic/checkpoint.pt"}, doctor, audit)
    digest = str(record["mapping_table_sha256"])
    assert len(digest) == 64 and not digest.startswith("LOCAL_PLACEHOLDER:")
    assert digest == audit["mapping_table_sha256"]
    assert record["n_source_keys"] == 43
    assert record["n_mapped_targets"] == 45


# ---------------------------------------------------------------------------
# C1 — ONE canonical recipe everywhere (no ``default=str`` divergence)
# ---------------------------------------------------------------------------
def test_assembly_identity_sha256_is_the_strict_canonical_recipe() -> None:
    """``assembly.identity_sha256`` delegates to
    ``identity.canonical_sha256`` — same digest for JSON-native blocks,
    TYPED refusal (never silent ``str()``) for anything else."""
    import hashlib
    import json as _json
    from pathlib import Path as _Path

    from pes2ts_core.pesn.models.v542.assembly import identity_sha256

    block = {"source_manifest_sha256": "a" * 64, "seed": 17}
    # same digest as the frozen canonical recipe …
    assert identity_sha256(block) == canonical_sha256(block)
    # … and byte-identical to the legacy lenient recipe for JSON-native
    # values (no digest migration for well-typed identity blocks)
    legacy = hashlib.sha256(_json.dumps(
        block, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        default=str).encode("utf-8")).hexdigest()
    assert identity_sha256(block) == legacy
    # a non-JSON value is a TYPED refusal — never silently str()-coerced
    with pytest.raises(V542IdentityError) as excinfo:
        identity_sha256({"x": _Path("/tmp")})
    assert excinfo.value.code == "V542_IDENTITY_SERIALIZE"
    with pytest.raises(V542IdentityError) as excinfo:
        identity_sha256({"x": object()})
    assert excinfo.value.code == "V542_IDENTITY_SERIALIZE"


def test_trainer_identity_digests_use_the_strict_canonical_recipe() -> None:
    """The trainer's identity/checkpoint digests come from the SAME
    strict function imported out of ``identity.py`` — one recipe, no
    local reimplementation, no silent coercion."""
    import hashlib
    import json as _json
    from dataclasses import asdict
    from pathlib import Path as _Path

    from pes2ts_core.pesn.offline.training import v542_train

    cfg = _trainer_cfg()
    blob = v542_train.config_identity_dict(cfg)
    expected = hashlib.sha256(_json.dumps(
        blob, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")).hexdigest()
    assert v542_train.config_identity_sha256(cfg) == expected
    assert v542_train.config_full_sha256(cfg) == canonical_sha256(
        asdict(cfg))
    # the identity block digest is the canonical one
    block = v542_train.identity_block_v542(cfg, {"k": "v"},
                                           pool_sha256=None)
    assert canonical_sha256(block) == hashlib.sha256(_json.dumps(
        block, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")).hexdigest()
    # the trainer module binds the ONE canonical function itself
    assert v542_train.canonical_sha256 is canonical_sha256
    # a non-JSON identity value refuses under the canonical recipe
    with pytest.raises(V542IdentityError) as excinfo:
        v542_train.canonical_sha256({"x": _Path("/tmp")})
    assert excinfo.value.code == "V542_IDENTITY_SERIALIZE"


def _trainer_cfg():
    """Minimal frozen-surface trainer config for digest tests."""
    from pes2ts_core.pesn.offline.training.v542_train import (
        V542TrainerConfig)

    return V542TrainerConfig(candidate_id="A1", seed=17, device="cpu")


# ---------------------------------------------------------------------------
# C5 — CLI robustness: frozen-seed guard + typed catch-all
# ---------------------------------------------------------------------------
def test_cli_refuses_out_of_set_seed_with_typed_refusal(capsys) -> None:
    """``--run-seed 99`` (not in {17, 29, 43}) is a typed refusal with
    exit 2 and NO identity on stdout — consistent with assembly/trainer
    (oracle gate-3 C5)."""
    import scripts.agent_ops.v542_identity as identity_cli

    code = identity_cli.main(["--candidate", "A1", "--run-seed", "99"])
    assert code == 2
    captured = capsys.readouterr()
    assert "REFUSED" in captured.err
    assert "V542_IDENTITY_SEED" in captured.err
    assert "17" in captured.err and "29" in captured.err and \
        "43" in captured.err
    assert captured.out == ""               # NO identity JSON printed
    # … and the SAME guard at the builder level
    with pytest.raises(V542IdentityError) as excinfo:
        identity_cli.build_identity_record(
            "A1", 99,
            config_path=PROJECT_ROOT / DEFAULT_CAMPAIGN_CONFIG,
            runplan_path=_RUNPLAN_TEMPLATE,
            budget_path=_BUDGET_TEMPLATE,
            source_manifest_path=None, require_clean=False,
            root=PROJECT_ROOT)
    assert excinfo.value.code == "V542_IDENTITY_SEED"


def test_cli_accepts_the_three_frozen_seeds(capsys) -> None:
    import scripts.agent_ops.v542_identity as identity_cli

    for seed in (17, 29, 43):
        record = identity_cli.build_identity_record(
            "A1", seed,
            config_path=PROJECT_ROOT / DEFAULT_CAMPAIGN_CONFIG,
            runplan_path=_RUNPLAN_TEMPLATE,
            budget_path=_BUDGET_TEMPLATE,
            source_manifest_path=None, require_clean=False,
            root=PROJECT_ROOT)
        assert record["run_seed"] == seed
        assert record["identity"]["process_head_anchor_sha256"] == \
            process_head_anchor_sha256(seed)[0]


def test_cli_unexpected_exception_is_a_typed_refusal(monkeypatch,
                                                     capsys) -> None:
    """An unexpected exception inside the CLI is printed as a typed
    ``V542_IDENTITY_UNEXPECTED`` refusal with exit 2 — never a raw
    traceback (oracle gate-3 C5)."""
    import scripts.agent_ops.v542_identity as identity_cli

    def _boom(*_args, **_kwargs):
        raise RuntimeError("synthetic unexpected failure")

    monkeypatch.setattr(identity_cli, "build_identity_record", _boom)
    code = identity_cli.main(["--candidate", "A1", "--run-seed", "17"])
    assert code == 2
    captured = capsys.readouterr()
    assert "REFUSED" in captured.err
    assert "V542_IDENTITY_UNEXPECTED" in captured.err
    assert "synthetic unexpected failure" in captured.err
    assert "Traceback" not in captured.err
    assert captured.out == ""               # NO identity JSON printed
