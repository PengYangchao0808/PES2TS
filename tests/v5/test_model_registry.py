"""Model-registry contract: 14 cores registered, buildable, forward/backward finite.

Guards the "the registry is the only model-assembly path" invariant: every
architecture must be constructible from a ``ModelConfig``, expose non-empty
``REQUIRED_FIELDS``, and produce finite outputs/gradients on a synthetic batch.
"""

from __future__ import annotations

import pytest
import torch

from pes2ts_v5.configs.schema import CoreConfig, HeadConfig, ModelConfig
from pes2ts_v5.models.registry import (
    BASE_BATCH_FIELDS,
    CORE_REGISTRY,
    ENCODER_REGISTRY,
    HEAD_REGISTRY,
    ModelBuildError,
    assert_all_buildable,
    build_model_from_config,
    describe_registry,
    list_cores,
    make_smoke_batch,
    register_core,
    register_encoder,
    register_head,
    required_fields_for,
    resolve_core_name,
    resolve_encoder_name,
)

CORE_IDS = [f"M{i:02d}" for i in range(1, 15)]

ALIAS_CASES = {
    "gru": "M04",
    "lstm": "M06",
    "causal_tcn": "M03",
    "tiny_causal_transformer": "M11",
    "s4d": "M13",
    "cfc": "M08",
    "time_aware_gru": "M05",
    "mamba2": "M14",
}


def test_all_14_cores_registered() -> None:
    assert len(CORE_REGISTRY) == 14
    assert list_cores() == CORE_IDS
    assert set(CORE_IDS) <= set(CORE_REGISTRY)


@pytest.mark.parametrize("alias,canonical", sorted(ALIAS_CASES.items()))
def test_algorithm_aliases_resolve(alias: str, canonical: str) -> None:
    assert resolve_core_name(alias) == canonical
    assert resolve_core_name(canonical) == canonical
    assert resolve_core_name(canonical.lower()) == canonical


def test_encoders_and_heads_registered() -> None:
    assert "descriptor" in ENCODER_REGISTRY
    assert "stregnn_v2" in ENCODER_REGISTRY
    assert resolve_encoder_name("egnn") == "stregnn_v2"
    assert "multi_head" in HEAD_REGISTRY


def test_duplicate_registration_fails() -> None:
    with pytest.raises(ValueError, match="already registered"):
        register_core("M04")(object)
    with pytest.raises(ValueError, match="already registered"):
        register_encoder("descriptor", required_fields=(), builder=lambda **_: None)(object)
    with pytest.raises(ValueError, match="already registered"):
        register_head("multi_head", factory=lambda **_: None)(object)


def test_unknown_names_raise_keyerror() -> None:
    with pytest.raises(KeyError):
        resolve_core_name("definitely_not_a_core")
    with pytest.raises(KeyError):
        resolve_encoder_name("definitely_not_an_encoder")


def test_required_fields_declared_and_non_empty() -> None:
    for name in CORE_IDS:
        fields = CORE_REGISTRY[name].required_fields
        assert isinstance(fields, frozenset) and fields
        assert BASE_BATCH_FIELDS <= fields
        # declared on the model class itself (Lead's contract declaration)
        assert getattr(CORE_REGISTRY[name].cls, "REQUIRED_FIELDS") == fields
        assert required_fields_for(name) == fields
    time_aware = {"M05", "M08", "M09"}
    for name in CORE_IDS:
        has_delta = "delta_s" in CORE_REGISTRY[name].required_fields
        assert has_delta == (name in time_aware), name
    union = required_fields_for("gru", "descriptor")
    assert union == frozenset({"frame_feats", "static_feats", "lengths"})


def test_describe_registry_covers_all_kinds() -> None:
    rows = describe_registry()
    cores = [row for row in rows if row["kind"] == "core"]
    encoders = [row for row in rows if row["kind"] == "encoder"]
    heads = [row for row in rows if row["kind"] == "head"]
    assert len(cores) == 14
    assert {row["name"] for row in encoders} >= {"descriptor", "stregnn_v2"}
    assert {row["name"] for row in heads} >= {"multi_head"}
    for row in cores:
        assert row["name"] in CORE_IDS
        assert row["required_fields"]
        assert row["description"]


@pytest.mark.parametrize("core", CORE_IDS)
def test_each_core_builds_forward_backward(core: str) -> None:
    cfg = ModelConfig(core=CoreConfig(name=core, hidden_dim=32, layers=1),
                      head=HeadConfig(process_classes=3))
    model = build_model_from_config(cfg, frame_feat_dim=11, static_feat_dim=8)
    assert model.arch == core
    assert sum(p.numel() for p in model.parameters()) > 0
    batch = make_smoke_batch(batch_size=2, seq_len=6)
    out = model(batch)
    for key in ("move", "process_logits", "process_progress", "trust_logit"):
        assert torch.isfinite(out[key]).all(), f"{core}: non-finite {key}"
    loss = model.losses(out, batch)["total"]
    assert torch.isfinite(loss).all()
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads, f"{core}: no gradients"
    for grad in grads:
        assert torch.isfinite(grad).all(), f"{core}: non-finite gradient"


def test_alias_build_is_the_same_architecture() -> None:
    model_alias = build_model_from_config(
        ModelConfig(core=CoreConfig(name="gru", hidden_dim=32), head=HeadConfig()))
    model_id = build_model_from_config(
        ModelConfig(core=CoreConfig(name="M04", hidden_dim=32), head=HeadConfig()))
    assert model_alias.arch == model_id.arch == "M04"
    assert type(model_alias.core) is type(model_id.core)


def test_unknown_core_rejected_by_schema() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        CoreConfig(name="M99")


def test_blocked_encoder_raises_typed_error() -> None:
    cfg = ModelConfig(encoder="stregnn_v2", core=CoreConfig(name="gru"),
                      head=HeadConfig())
    with pytest.raises(ModelBuildError) as excinfo:
        build_model_from_config(cfg)
    assert excinfo.value.code == "ENCODER_INPUT_CONTRACT_MISMATCH"


def test_assert_all_buildable_smoke() -> None:
    rows = assert_all_buildable("cpu", hidden_dim=32, seq_len=6)
    assert [row["name"] for row in rows] == CORE_IDS
    assert all(row["ok"] and row["total_params"] > 0 for row in rows)
