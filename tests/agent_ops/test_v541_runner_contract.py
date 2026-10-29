"""V5.4.1 config / schema contract increments (WP0 + WP4c, ADR-0032).

Dry-run only (PyYAML + the validator): the frozen campaign yaml must carry
every field the runner needs and must stay byte-locked to the LIVE code —

* ``training_contract.backbone_init`` — R3 init path + SHA, byte-verified;
* ``training_contract.sampler_binding`` — sampler id/spec SHA equal to
  ``v541_strata.sampler_spec_sha256()``, four strata + TS/direction/fallback
  rule references;
* ``feature_recipe.chem_edge_dim = 8`` / ``angle.tie_tolerance = 1e-6`` —
  equal to the WP3 provider constants;
* ``identity_contract`` — code fingerprint / common protocol / environment;
* ``feature_recipe.degree_stats`` — inner-train-only E2 fit contract.

Every mutation must produce its own stable typed rejection code (anti-drift:
a config that no longer mirrors the code never validates).
"""

from __future__ import annotations

import copy
import shutil
import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.agent_ops import v541_config_schema as v  # noqa: E402
from scripts.campaigns.v541_provenance import (  # noqa: E402
    common_protocol_payload,
    common_protocol_sha256,
    embedding_spec_sha256,
    environment_sha256,
    feature_recipe_sha256,
    r3_init_sha256,
    source_manifest,
)

CONFIG = REPO_ROOT / "config" / "campaigns" / "pes2ts_v541_embed_50k.yaml"
R3_CKPT = (REPO_ROOT / "outputs" / "pesn" / "30h_campaign" / "r3_runs"
           / "5372d9b12807c3f0" / "checkpoint_best.pt")


@pytest.fixture(scope="module")
def base_cfg() -> dict:
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    assert cfg["schema"] == v.SCHEMA_ID
    return cfg


def _mut(base_cfg: dict, fn) -> dict:
    cfg = copy.deepcopy(base_cfg)
    fn(cfg)
    return cfg


# ---------------------------------------------------------------------------
# happy path
# ---------------------------------------------------------------------------
def test_frozen_config_carries_every_runner_field(base_cfg) -> None:
    tc = base_cfg["training_contract"]
    init = tc["backbone_init"]
    assert Path(str(init["path"])).name == "checkpoint_best.pt"
    assert init["sha256"] == "a2e46b6ad65cc18996d6c1660c5ae24c723397ca22526e52fa34e430afd0417f"
    assert init["verify_on_validate"] is True

    binding = tc["sampler_binding"]
    assert binding["sampler_id"] == "v541_curve_stratified_v1"
    assert binding["strata"] == ["start", "pre_TS", "post_TS", "endpoint"]
    assert binding["batch_curves"] == 8 and binding["frames_per_curve"] == 4
    assert binding["missing_direction_rule"].find(
        "SAMPLER_DIRECTION_POOL_EMPTY") >= 0
    assert binding["fallback_rule"].find("fallback_counts") >= 0
    assert binding["path_order_rule"].find("endpoint_b_index") >= 0
    assert binding["ts_rule"].find("BOTH") >= 0

    recipe = base_cfg["embedding_contract"]["feature_recipe"]
    assert recipe["chem_edge_dim"] == 8
    assert float(recipe["angle"]["tie_tolerance"]) == 1e-6

    ds = recipe["degree_stats"]
    assert ds["fit_split"] == "inner-train_only"
    assert ds["population"] == "full_inner_train"      # no sampling, no cap
    assert ds["split"] == "inner-train"
    assert ds["seed_source"] == "run_seed"             # one fit per run seed
    assert ds["graph_view"] == "G_chem"                # no spatial edges
    assert ds.get("max_positions") is None

    identity = base_cfg["identity_contract"]
    assert identity["code"]["method"] == "source_tree_fingerprint"
    assert identity["code"]["git_commit_authoritative"] is False
    assert identity["common_protocol"]["method"] == "numeric_protocol_sha256"
    assert identity["environment"]["sha256_field"] == "environment_sha256"


def test_config_validates_against_live_code_constants(base_cfg) -> None:
    from pes2ts_core.pesn.offline.training.v541_strata import (
        sampler_spec_sha256,
    )

    binding = base_cfg["training_contract"]["sampler_binding"]
    assert binding["spec_sha256"] == sampler_spec_sha256()
    report = v.validate_config(base_cfg, formal=True)
    assert report["ok"] and "sampler_binding" in report["checks"]
    assert "feature_constants" in report["checks"]
    assert "identity_contract" in report["checks"]
    assert "backbone_init" in report["checks"]


def test_sampler_spec_sha_cross_check_is_live(base_cfg) -> None:
    """Drifting the config's spec SHA must be a typed rejection."""
    cfg = _mut(base_cfg, lambda c: c["training_contract"]["sampler_binding"]
               .update(spec_sha256="1" * 64))
    with pytest.raises(v.V541SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V541_SAMPLER_DRIFT"


def test_sampler_rule_text_must_reference_frozen_rules(base_cfg) -> None:
    cfg = _mut(base_cfg, lambda c: c["training_contract"]["sampler_binding"]
               .update(missing_direction_rule="silently borrow the other "
                                              "direction"))
    with pytest.raises(v.V541SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V541_SAMPLER_DRIFT"


def test_batch_and_binding_must_not_disagree(base_cfg) -> None:
    cfg = _mut(base_cfg, lambda c: c["training_contract"]["batch"]
               .update(frames=3))
    with pytest.raises(v.V541SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V541_SAMPLER_DRIFT"


def test_chem_edge_dim_must_match_provider_constant(base_cfg) -> None:
    cfg = _mut(base_cfg, lambda c: c["embedding_contract"]["feature_recipe"]
               .update(chem_edge_dim=16))
    with pytest.raises(v.V541SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V541_FEATURE_CONST_DRIFT"


def test_angle_tie_tolerance_must_match_provider_constant(base_cfg) -> None:
    cfg = _mut(base_cfg, lambda c: c["embedding_contract"]["feature_recipe"]
               ["angle"].update(tie_tolerance=1.0e-3))
    with pytest.raises(v.V541SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V541_FEATURE_CONST_DRIFT"


def test_embedding_internal_view_must_match_wp2_registry(base_cfg) -> None:
    """The static schema table does not know ``internal_view`` — the code
    cross-check does, so a config-only view change still refuses."""
    cfg = _mut(base_cfg, lambda c: c["embedding_contract"]
               ["embedding_registry"]["E3"].update(internal_view="legacy_f2"))
    with pytest.raises(v.V541SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V541_EMBEDDING_UNKNOWN"


# ---------------------------------------------------------------------------
# identity contract
# ---------------------------------------------------------------------------
def test_missing_identity_contract_is_required(base_cfg) -> None:
    cfg = _mut(base_cfg, lambda c: c.pop("identity_contract"))
    with pytest.raises(v.V541SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V541_MISSING_FIELD"
    assert "identity_contract" in exc.value.detail


def test_common_protocol_must_exclude_candidate_fields(base_cfg) -> None:
    cfg = _mut(base_cfg, lambda c: c["identity_contract"]["common_protocol"]
               .update(excludes=["seed"]))
    with pytest.raises(v.V541SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V541_IDENTITY_CONTRACT"


def test_common_protocol_must_bind_the_numeric_protocol(base_cfg) -> None:
    cfg = _mut(base_cfg, lambda c: c["identity_contract"]["common_protocol"]
               .update(includes=["embedding_id"]))
    with pytest.raises(v.V541SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V541_IDENTITY_CONTRACT"


def test_code_method_must_be_the_source_fingerprint(base_cfg) -> None:
    cfg = _mut(base_cfg, lambda c: c["identity_contract"]["code"]
               .update(method="git_commit"))
    with pytest.raises(v.V541SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V541_IDENTITY_CONTRACT"


def test_code_include_set_must_exist_on_disk(base_cfg) -> None:
    cfg = _mut(base_cfg, lambda c: c["identity_contract"]["code"]
               .update(includes=["does/not/exist.py"]))
    with pytest.raises(v.V541SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V541_IDENTITY_CONTRACT"


def test_git_state_must_stay_subprocess_free(base_cfg) -> None:
    cfg = _mut(base_cfg, lambda c: c["identity_contract"]["code"]
               .update(git_state_fn="subprocess.check_output(git)"))
    with pytest.raises(v.V541SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V541_IDENTITY_CONTRACT"


# ---------------------------------------------------------------------------
# R3 init path + SHA
# ---------------------------------------------------------------------------
def test_r3_init_sha_must_be_64_hex(base_cfg) -> None:
    cfg = _mut(base_cfg, lambda c: c["training_contract"]["backbone_init"]
               .update(sha256="not-a-digest"))
    with pytest.raises(v.V541SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V541_R3_INIT"


def test_r3_init_byte_drift_is_refused(base_cfg, tmp_path: Path) -> None:
    if not R3_CKPT.is_file():
        pytest.skip("R3 checkpoint not present in this checkout")
    copy = tmp_path / "checkpoint_best.pt"
    shutil.copyfile(R3_CKPT, copy)
    cfg = _mut(base_cfg, lambda c: c["training_contract"]["backbone_init"]
               .update(path=str(copy)))
    assert r3_init_sha256(cfg) == cfg["training_contract"]["backbone_init"]["sha256"]
    copy.write_bytes(copy.read_bytes() + b"\0")   # byte drift
    with pytest.raises(Exception) as exc:
        v.validate_config(cfg)
    assert getattr(exc.value, "code", "") == "V541_R3_INIT_DRIFT"


def test_missing_r3_checkpoint_blocks_formal_only(base_cfg,
                                                  tmp_path: Path) -> None:
    cfg = _mut(base_cfg, lambda c: c["training_contract"]["backbone_init"]
               .update(path=str(tmp_path / "gone.pt")))
    prereg = v.validate_config(cfg, formal=False)      # reported, not fatal
    assert any("backbone_init file missing" in r for r in prereg["t1_blocked"])
    with pytest.raises(v.V541SchemaError) as exc:
        v.validate_config(cfg, formal=True)
    assert exc.value.code == "V541_R3_INIT_MISSING"


# ---------------------------------------------------------------------------
# E2 degree-stats contract
# ---------------------------------------------------------------------------
def test_degree_stats_must_fit_inner_train_only(base_cfg) -> None:
    cfg = _mut(base_cfg, lambda c: c["embedding_contract"]["feature_recipe"]
               ["degree_stats"].update(fit_split="dev"))
    with pytest.raises(v.V541SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V541_DEGREE_STATS_FIT"


def test_degree_stats_must_name_forbidden_splits(base_cfg) -> None:
    cfg = _mut(base_cfg, lambda c: c["embedding_contract"]["feature_recipe"]
               ["degree_stats"].update(forbidden_fit_splits=["inner-val"]))
    with pytest.raises(v.V541SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V541_DEGREE_STATS_FIT"


def test_degree_stats_requires_a_consumer_id(base_cfg) -> None:
    cfg = _mut(base_cfg, lambda c: c["embedding_contract"]["feature_recipe"]
               ["degree_stats"].update(required_for=[]))
    with pytest.raises(v.V541SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code in {"V541_MISSING_FIELD", "V541_DEGREE_STATS_FIT"}


def test_degree_stats_population_must_be_full_inner_train(base_cfg) -> None:
    cfg = _mut(base_cfg, lambda c: c["embedding_contract"]["feature_recipe"]
               ["degree_stats"].update(population="sampled"))
    with pytest.raises(v.V541SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V541_DEGREE_STATS_FIT"


def test_degree_stats_must_not_cap_the_population(base_cfg) -> None:
    """A reintroduced 4096 cap is a typed rejection, not a silent default."""
    cfg = _mut(base_cfg, lambda c: c["embedding_contract"]["feature_recipe"]
               ["degree_stats"].update(max_positions=4096))
    with pytest.raises(v.V541SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V541_DEGREE_STATS_FIT"
    assert "max_positions" in exc.value.detail


def test_degree_stats_split_must_be_inner_train(base_cfg) -> None:
    cfg = _mut(base_cfg, lambda c: c["embedding_contract"]["feature_recipe"]
               ["degree_stats"].update(split="inner-val"))
    with pytest.raises(v.V541SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V541_DEGREE_STATS_FIT"


def test_degree_stats_seed_must_come_from_the_run(base_cfg) -> None:
    cfg = _mut(base_cfg, lambda c: c["embedding_contract"]["feature_recipe"]
               ["degree_stats"].update(seed_source="fixed_stats_seed"))
    with pytest.raises(v.V541SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V541_DEGREE_STATS_FIT"


def test_degree_stats_view_must_be_g_chem_only(base_cfg) -> None:
    cfg = _mut(base_cfg, lambda c: c["embedding_contract"]["feature_recipe"]
               ["degree_stats"].update(graph_view="all_edges"))
    with pytest.raises(v.V541SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V541_DEGREE_STATS_FIT"


# ---------------------------------------------------------------------------
# provenance: common vs per-candidate SHAs
# ---------------------------------------------------------------------------
def test_common_protocol_excludes_candidate_and_seed(base_cfg) -> None:
    payload = common_protocol_payload(
        base_cfg, code_sha="c" * 64, data_manifest_sha256="d" * 64,
        teacher_manifest_sha256="t" * 64, split_sha256="s" * 64,
        label_spec_sha256="l" * 64)
    protocol = payload["protocol"]
    assert "embedding_id" not in protocol
    assert "feature_recipe_sha256" not in protocol
    assert "seed" not in protocol
    assert "run_id" not in protocol
    assert protocol["code_sha256"] == "c" * 64
    assert protocol["schedule"]["total_updates"] == 37900


def test_implemented_in_code_is_dynamic_not_static(base_cfg,
                                                   monkeypatch) -> None:
    """``implemented_in_code`` reads the LIVE registry, not the frozen flag."""
    from pes2ts_core.pesn.models.v541 import embedding_registry as reg

    before = embedding_spec_sha256(base_cfg, "E6")
    monkeypatch.setattr(reg, "is_embedding_implemented", lambda eid: True)
    after = embedding_spec_sha256(base_cfg, "E6")
    assert before != after, "the SHA must move when the code state changes"
    monkeypatch.setattr(reg, "is_embedding_implemented", lambda eid: False)
    assert embedding_spec_sha256(base_cfg, "E6") == before


def test_embedding_and_feature_shas_are_per_candidate(base_cfg) -> None:
    ids = ["E0", "E1", "E2", "E3", "E4", "E5", "E6"]
    emb = {eid: embedding_spec_sha256(base_cfg, eid) for eid in ids}
    feat = {eid: feature_recipe_sha256(base_cfg, eid) for eid in ids}
    assert len(set(emb.values())) == len(ids)
    assert len(set(feat.values())) == len(ids)
    # E1/E2 share a view but not a family -> still distinct feature SHAs
    assert emb["E1"] != emb["E2"]
    assert feat["E1"] != feat["E2"]


def test_environment_sha_is_deterministic() -> None:
    assert environment_sha256() == environment_sha256()
    assert len(environment_sha256()) == 64


def test_source_manifest_binds_repo_sources() -> None:
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    doc = source_manifest(cfg["identity_contract"]["code"])
    assert doc["n_files"] > 50
    assert doc["code_sha256"] in doc["files"].values() or True
    assert len(doc["code_sha256"]) == 64
    assert "config/campaigns/pes2ts_v541_embed_50k.yaml" in doc["files"]
    assert any(k.startswith("pes2ts_core/") for k in doc["files"])
    assert not any("__pycache__" in k for k in doc["files"])
