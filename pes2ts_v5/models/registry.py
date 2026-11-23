"""Decorator-based model registry — the **only** model-assembly path.

Every runnable model is ``encoder + core + head``.  This module:

* registers the 14 existing memory cores (M01..M14) **and** stable algorithm
  aliases (``gru``, ``lstm``, ``causal_tcn``, ``s4d``, ...) by importing the
  existing classes — no architecture code is duplicated here;
* registers the existing encoders (``descriptor``, ``stregnn_v2``) and heads
  (``multi_head``);
* declares, per core/encoder, the batch fields it requires
  (``REQUIRED_FIELDS`` / :func:`required_fields_for`);
* builds models from :class:`~pes2ts_v5.configs.schema.ModelConfig` via
  :func:`build_model_from_config`.

Honest-capability note: ``stregnn_v2`` is registered for completeness but its
``forward`` needs padded atom tensors (``coords``/``atom_types``/``atom_mask``)
that the current V5.5-style trajectory batch does not ship.  Requesting it raises
:class:`ModelBuildError` with ``code="ENCODER_INPUT_CONTRACT_MISMATCH"`` rather
than fabricating atom data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Iterable, Mapping

import torch

from pes2ts_v5.models.cores import (
    CORE_META,
    build_core,
    count_params,
    resolve_core_config,
)
from pes2ts_v5.models.encoders import build_encoder
from pes2ts_v5.models.heads import MultiHead
from pes2ts_v5.models.v5_model import V5Model

if TYPE_CHECKING:  # pragma: no cover - typing only (no import cycle at runtime)
    from pes2ts_v5.configs.schema import ModelConfig

#: fields every trajectory batch for this engine provides
BASE_BATCH_FIELDS: frozenset[str] = frozenset({"frame_feats", "static_feats", "lengths"})
#: extra field required by time-aware (irregular-sampling) cores
TIME_AWARE_FIELDS: frozenset[str] = frozenset({"delta_s"})


class ModelBuildError(RuntimeError):
    """Typed model-assembly failure (``code`` is stable for callers/tests)."""

    def __init__(self, code: str, message: str):
        super().__init__(f"[{code}] {message}")
        self.code = str(code)
        self.message = str(message)


# ---------------------------------------------------------------------------
# Specs + registries
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CoreSpec:
    name: str                       #: canonical id, e.g. ``M04``
    cls: type
    paradigm: str
    stateful: bool
    required_fields: frozenset[str]
    description: str
    aliases: tuple[str, ...] = ()
    simplified: bool = False


@dataclass(frozen=True)
class EncoderSpec:
    name: str
    cls: type
    required_fields: frozenset[str]
    description: str
    builder: Callable[..., Any]
    #: ``False`` = cannot consume the current V5 trajectory batch yet
    trainable_on_v5_batch: bool = True
    aliases: tuple[str, ...] = ()


@dataclass(frozen=True)
class HeadSpec:
    name: str
    cls: type
    factory: Callable[..., Any]
    description: str
    aliases: tuple[str, ...] = ()


CORE_REGISTRY: dict[str, CoreSpec] = {}
ENCODER_REGISTRY: dict[str, EncoderSpec] = {}
HEAD_REGISTRY: dict[str, HeadSpec] = {}
_CORE_ALIASES: dict[str, str] = {}
_ENCODER_ALIASES: dict[str, str] = {}
_HEAD_ALIASES: dict[str, str] = {}


def _register_alias(table: dict[str, str], alias: str, canonical: str,
                    registry: Mapping[str, Any], kind: str) -> None:
    key = str(alias).strip().lower()
    if not key:
        raise ValueError(f"{kind} alias must be non-empty")
    if key in table:
        raise ValueError(f"{kind} alias {key!r} already registered -> {table[key]!r}")
    if key in registry:
        raise ValueError(f"{kind} alias {key!r} collides with a canonical name")
    table[key] = canonical


def register_core(
    name: str,
    *,
    aliases: Iterable[str] = (),
    paradigm: str = "unknown",
    stateful: bool = False,
    required_fields: Iterable[str] = (),
    description: str = "",
    simplified: bool = False,
):
    """Register a memory core class under ``name`` (canonical id) + aliases.

    On registration the class gains ``REQUIRED_FIELDS`` (a ``frozenset`` of batch
    keys it needs), which is the per-model declaration consumed by
    :func:`required_fields_for`.
    """
    canonical = str(name).strip().upper()

    def _decorator(cls: type) -> type:
        if canonical in CORE_REGISTRY:
            raise ValueError(
                f"core {canonical!r} already registered as "
                f"{CORE_REGISTRY[canonical].cls.__name__}")
        fields = frozenset(BASE_BATCH_FIELDS | {str(f) for f in required_fields})
        if not fields:
            raise ValueError(f"core {canonical!r} must declare required_fields")
        spec = CoreSpec(
            name=canonical, cls=cls, paradigm=str(paradigm),
            stateful=bool(stateful), required_fields=fields,
            description=str(description or getattr(cls, "note", "")),
            aliases=tuple(str(a).strip().lower() for a in aliases),
            simplified=bool(simplified))
        CORE_REGISTRY[canonical] = spec
        cls.REQUIRED_FIELDS = fields          # declared on the model itself
        cls.CORE_ID = canonical
        for alias in spec.aliases:
            _register_alias(_CORE_ALIASES, alias, canonical, CORE_REGISTRY, "core")
        return cls

    return _decorator


def register_encoder(
    name: str,
    *,
    required_fields: Iterable[str],
    builder: Callable[..., Any],
    description: str = "",
    trainable_on_v5_batch: bool = True,
    aliases: Iterable[str] = (),
):
    """Register an encoder class (``builder`` maps config knobs -> module)."""
    canonical = str(name).strip().lower()

    def _decorator(cls: type) -> type:
        if canonical in ENCODER_REGISTRY:
            raise ValueError(
                f"encoder {canonical!r} already registered as "
                f"{ENCODER_REGISTRY[canonical].cls.__name__}")
        fields = frozenset(str(f) for f in required_fields)
        spec = EncoderSpec(
            name=canonical, cls=cls, required_fields=fields,
            description=str(description), builder=builder,
            trainable_on_v5_batch=bool(trainable_on_v5_batch),
            aliases=tuple(str(a).strip().lower() for a in aliases))
        ENCODER_REGISTRY[canonical] = spec
        cls.REQUIRED_FIELDS = fields
        for alias in spec.aliases:
            _register_alias(_ENCODER_ALIASES, alias, canonical, ENCODER_REGISTRY,
                            "encoder")
        return cls

    return _decorator


def register_head(
    name: str,
    *,
    factory: Callable[..., Any],
    description: str = "",
    aliases: Iterable[str] = (),
):
    """Register a prediction-head factory."""
    canonical = str(name).strip().lower()

    def _decorator(cls: type) -> type:
        if canonical in HEAD_REGISTRY:
            raise ValueError(
                f"head {canonical!r} already registered as "
                f"{HEAD_REGISTRY[canonical].cls.__name__}")
        spec = HeadSpec(name=canonical, cls=cls, factory=factory,
                        description=str(description),
                        aliases=tuple(str(a).strip().lower() for a in aliases))
        HEAD_REGISTRY[canonical] = spec
        for alias in spec.aliases:
            _register_alias(_HEAD_ALIASES, alias, canonical, HEAD_REGISTRY, "head")
        return cls

    return _decorator


# ---------------------------------------------------------------------------
# Registration of the existing implementations (no code duplication)
# ---------------------------------------------------------------------------

def _register_builtin_cores() -> None:
    from pes2ts_v5.models.cores import (
        CORE_REGISTRY as CORE_CLASSES,
    )
    from pes2ts_v5.models.cores import (
        S4D,
        CausalTCN,
        CfCSmall,
        DiagonalSelectiveSSM,
        GRUCore,
        LatentNeuralODE,
        LSTMCore,
        LTCSmall,
        Mamba2Small,
        StatelessMLP,
        TimeAwareGRU,
        TinyCausalTransformer,
        WindowMLP,
        XLSTMSmall,
    )

    # (canonical id, class, aliases, extra batch fields)
    table = (
        ("M01", StatelessMLP, ("stateless_mlp", "mlp"), ()),
        ("M02", WindowMLP, ("window_mlp", "window"), ()),
        ("M03", CausalTCN, ("causal_tcn", "tcn"), ()),
        ("M04", GRUCore, ("gru",), ()),
        ("M05", TimeAwareGRU, ("time_aware_gru", "ta_gru"), ("delta_s",)),
        ("M06", LSTMCore, ("lstm",), ()),
        ("M07", XLSTMSmall, ("xlstm",), ()),
        ("M08", CfCSmall, ("cfc",), ("delta_s",)),
        ("M09", LTCSmall, ("ltc",), ("delta_s",)),
        ("M10", LatentNeuralODE, ("latent_neural_ode", "neural_ode"), ()),
        ("M11", TinyCausalTransformer, ("tiny_causal_transformer", "transformer"), ()),
        ("M12", DiagonalSelectiveSSM, ("diagonal_selective_ssm", "selective_ssm"), ()),
        ("M13", S4D, ("s4d",), ()),
        ("M14", Mamba2Small, ("mamba2_small", "mamba2"), ()),
    )
    for canonical, cls, aliases, extra_fields in table:
        meta = CORE_META.get(canonical, {})
        description = str(meta.get("title", cls.__name__))
        if meta.get("simplification"):
            description += f" (simplified: {meta['simplification']})"
        register_core(
            canonical, aliases=aliases, paradigm=str(meta.get("paradigm", "unknown")),
            stateful=bool(meta.get("stateful", False)),
            required_fields=extra_fields, description=description,
            simplified=bool(meta.get("simplified", False)))(cls)
    # the canonical ids must stay usable as the class-registry keys too
    assert set(CORE_CLASSES) <= set(CORE_REGISTRY), "core id mismatch"


def _register_builtin_encoders() -> None:
    from pes2ts_v5.models.encoders import DescriptorEncoder, StarEGNNv2Encoder

    register_encoder(
        "descriptor", required_fields=("frame_feats", "static_feats"),
        builder=lambda **kw: build_encoder("descriptor", **kw),
        description="low-dimensional reaction descriptor (per-frame + static "
                    "conditioning); the only encoder trainable on the current "
                    "trajectory batch")(DescriptorEncoder)
    register_encoder(
        "stregnn_v2", required_fields=("coords", "atom_types", "atom_mask"),
        builder=lambda hidden_dim, dropout=0.0, **_kw: StarEGNNv2Encoder(
            hidden_dim, dropout=dropout),
        description="StarEGNN v2 (EGNN-lite) stub; needs padded atom tensors the "
                    "V5.5-style trajectory batch does not ship",
        trainable_on_v5_batch=False, aliases=("egnn", "stregnn"))(StarEGNNv2Encoder)


def _register_builtin_heads() -> None:
    register_head(
        "multi_head", factory=lambda **kw: MultiHead(**kw),
        description="Move / Process / Trust multi-head (existing MultiHead)")(MultiHead)


def _bootstrap() -> None:
    if CORE_REGISTRY:
        return
    _register_builtin_cores()
    _register_builtin_encoders()
    _register_builtin_heads()


_bootstrap()


# ---------------------------------------------------------------------------
# Name resolution
# ---------------------------------------------------------------------------

def resolve_core_name(name: str) -> str:
    """Resolve ``M04`` / ``m04`` / ``gru`` to the canonical id; ``KeyError`` else."""
    text = str(name).strip()
    if text.upper() in CORE_REGISTRY:
        return text.upper()
    alias = _CORE_ALIASES.get(text.lower())
    if alias is not None:
        return alias
    raise KeyError(f"unknown core {name!r}; canonical={sorted(CORE_REGISTRY)}; "
                   f"aliases={sorted(_CORE_ALIASES)}")


def resolve_encoder_name(name: str) -> str:
    text = str(name).strip().lower()
    if text in ENCODER_REGISTRY:
        return text
    alias = _ENCODER_ALIASES.get(text)
    if alias is not None:
        return alias
    raise KeyError(f"unknown encoder {name!r}; registered={sorted(ENCODER_REGISTRY)}")


def resolve_head_name(name: str) -> str:
    text = str(name).strip().lower()
    if text in HEAD_REGISTRY:
        return text
    alias = _HEAD_ALIASES.get(text)
    if alias is not None:
        return alias
    raise KeyError(f"unknown head {name!r}; registered={sorted(HEAD_REGISTRY)}")


def list_cores(*, include_aliases: bool = False) -> list[str]:
    """Canonical core ids (optionally followed by their aliases)."""
    names = sorted(CORE_REGISTRY)
    if not include_aliases:
        return names
    return names + sorted(_CORE_ALIASES)


def list_encoders() -> list[str]:
    return sorted(ENCODER_REGISTRY)


def list_heads() -> list[str]:
    return sorted(HEAD_REGISTRY)


def required_fields_for(core_name: str | None = None,
                        encoder_name: str | None = None) -> frozenset[str]:
    """Union of batch fields needed by the core and (optionally) the encoder."""
    if core_name is None and encoder_name is None:
        raise ValueError("required_fields_for needs a core_name and/or encoder_name")
    fields: frozenset[str] = frozenset()
    if core_name is not None:
        fields |= CORE_REGISTRY[resolve_core_name(core_name)].required_fields
    if encoder_name is not None:
        fields |= ENCODER_REGISTRY[resolve_encoder_name(encoder_name)].required_fields
    return fields


def describe_registry() -> list[dict]:
    """JSON-able description of every registered core/encoder/head."""
    rows: list[dict] = []
    for name in sorted(CORE_REGISTRY):
        spec = CORE_REGISTRY[name]
        rows.append({
            "kind": "core", "name": spec.name, "class": spec.cls.__name__,
            "aliases": list(spec.aliases), "paradigm": spec.paradigm,
            "stateful": spec.stateful, "simplified": spec.simplified,
            "required_fields": sorted(spec.required_fields),
            "description": spec.description,
        })
    for name in sorted(ENCODER_REGISTRY):
        spec = ENCODER_REGISTRY[name]
        rows.append({
            "kind": "encoder", "name": spec.name, "class": spec.cls.__name__,
            "aliases": list(spec.aliases),
            "required_fields": sorted(spec.required_fields),
            "trainable_on_v5_batch": spec.trainable_on_v5_batch,
            "description": spec.description,
        })
    for name in sorted(HEAD_REGISTRY):
        spec = HEAD_REGISTRY[name]
        rows.append({
            "kind": "head", "name": spec.name, "class": spec.cls.__name__,
            "aliases": list(spec.aliases), "description": spec.description,
        })
    return rows


# ---------------------------------------------------------------------------
# Model assembly
# ---------------------------------------------------------------------------

def _encoder_contract_error(encoder_name: str, spec: "EncoderSpec") -> ModelBuildError:
    return ModelBuildError(
        "ENCODER_INPUT_CONTRACT_MISMATCH",
        f"encoder {encoder_name!r} requires batch fields "
        f"{sorted(spec.required_fields)} which the current V5.5-style trajectory "
        f"batch does not provide ({sorted(BASE_BATCH_FIELDS)}); no padded-atom-"
        "tensor adapter exists yet, so this config cannot train — it is "
        "registered for wiring/ablation only")


def assert_model_config_runnable(cfg: "ModelConfig | Mapping[str, Any]") -> None:
    """Fail fast (typed) when a config names a component that cannot run yet.

    Called by the CLI *before* data resolution so an encoder-capability blocker is
    reported as ``ENCODER_INPUT_CONTRACT_MISMATCH`` rather than a later data error.
    """
    if not isinstance(cfg, _model_config_type()):
        from pes2ts_v5.configs.schema import ModelConfig as _ModelConfig
        cfg = _ModelConfig.model_validate(dict(cfg))
    try:
        encoder_name = resolve_encoder_name(cfg.encoder)
        resolve_core_name(cfg.core.name)
        resolve_head_name(cfg.head.name)
    except KeyError as exc:
        raise ModelBuildError("UNKNOWN_COMPONENT", str(exc)) from exc
    spec = ENCODER_REGISTRY[encoder_name]
    if not spec.trainable_on_v5_batch:
        raise _encoder_contract_error(encoder_name, spec)


def _resolve_core_hidden(core_cfg: Any, canonical: str) -> tuple[int, dict]:
    """Return ``(hidden_dim, core_info)`` for an explicit or ``auto`` width."""
    hidden = core_cfg.hidden_dim
    if isinstance(hidden, int):
        return int(hidden), {"hidden_dim": int(hidden), "target": None,
                             "rel_error": None, "source": "explicit"}
    info = resolve_core_config(canonical, int(core_cfg.param_budget),
                               layers=int(core_cfg.layers))
    return int(info["hidden_dim"]), {**info, "source": "auto"}


def resolve_core_hidden_dim(core_cfg: Any) -> int:
    """Resolve the effective hidden width for a ``CoreConfig`` (``auto`` -> int)."""
    canonical = resolve_core_name(core_cfg.name)
    return int(_resolve_core_hidden(core_cfg, canonical)[0])


def build_model_from_config(
    cfg: "ModelConfig | Mapping[str, Any]",
    *,
    frame_feat_dim: int = 11,
    static_feat_dim: int = 8,
) -> V5Model:
    """Assemble the ``V5Model`` described by a :class:`ModelConfig`.

    Raises :class:`ModelBuildError` (typed ``code``) when the encoder cannot
    consume the current trajectory batch, or when a name cannot be resolved.
    """
    if not isinstance(cfg, _model_config_type()):
        from pes2ts_v5.configs.schema import ModelConfig as _ModelConfig
        cfg = _ModelConfig.model_validate(dict(cfg))

    try:
        encoder_name = resolve_encoder_name(cfg.encoder)
        core_name = resolve_core_name(cfg.core.name)
        head_name = resolve_head_name(cfg.head.name)
    except KeyError as exc:
        raise ModelBuildError("UNKNOWN_COMPONENT", str(exc)) from exc

    encoder_spec = ENCODER_REGISTRY[encoder_name]
    if not encoder_spec.trainable_on_v5_batch:
        raise _encoder_contract_error(encoder_name, encoder_spec)

    hidden, core_info = _resolve_core_hidden(cfg.core, core_name)
    encoder = encoder_spec.builder(
        frame_feat_dim=int(frame_feat_dim), static_feat_dim=int(static_feat_dim),
        hidden_dim=hidden, dropout=float(cfg.core.dropout))
    core = build_core(core_name, hidden_dim=hidden, layers=int(cfg.core.layers),
                      dropout=float(cfg.core.dropout))
    head_spec = HEAD_REGISTRY[head_name]
    heads = head_spec.factory(hidden_dim=hidden, q_dim=int(cfg.head.q_dim),
                              process_classes=int(cfg.head.process_classes),
                              dropout=float(cfg.head.dropout))
    model = V5Model(core_name, encoder, core, heads, q_dim=int(cfg.head.q_dim),
                    core_info=core_info)
    model.encoder_name = encoder_name
    model.head_name = head_name
    required = required_fields_for(core_name, encoder_name)
    model.REQUIRED_FIELDS = required      # standard batch contract declaration
    model.required_fields = required
    return model


def _model_config_type() -> type:
    from pes2ts_v5.configs.schema import ModelConfig
    return ModelConfig


# ---------------------------------------------------------------------------
# Smoke helper
# ---------------------------------------------------------------------------

def make_smoke_batch(
    *,
    batch_size: int = 2,
    seq_len: int = 8,
    frame_feat_dim: int = 11,
    static_feat_dim: int = 8,
    q_dim: int = 3,
    process_classes: int = 3,
    seed: int = 0,
    device: str | torch.device = "cpu",
) -> dict[str, torch.Tensor]:
    """Deterministic synthetic trajectory batch (all loss heads + labels)."""
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    b, t, q, c = int(batch_size), int(seq_len), int(q_dim), int(process_classes)
    lengths = torch.full((b,), t, dtype=torch.int64)
    batch = {
        "frame_feats": torch.randn(b, t, frame_feat_dim, generator=g),
        "static_feats": torch.randn(b, static_feat_dim, generator=g),
        "delta_s": torch.rand(b, t, generator=g) * 0.5,
        "lengths": lengths,
        "frame_mask": torch.ones(b, t, dtype=torch.bool),
        "move_target": torch.randn(b, t, q, generator=g) * 0.1,
        "move_mask": torch.ones(b, t, dtype=torch.bool),
        "process_label": torch.randint(0, c, (b, t), generator=g),
        "process_mask": torch.ones(b, t, dtype=torch.bool),
        "process_progress": torch.rand(b, t, generator=g),
        "progress_mask": torch.ones(b, t, dtype=torch.bool),
        "trust_label": torch.rand(b, t, generator=g),
        "trust_mask": torch.ones(b, t, dtype=torch.bool),
    }
    if str(device) != "cpu":
        batch = {k: v.to(device) for k, v in batch.items()}
    return batch


def assert_all_buildable(
    device: str = "cpu",
    *,
    hidden_dim: int = 32,
    batch_size: int = 2,
    seq_len: int = 8,
) -> list[dict]:
    """Build every registered core, forward one synthetic batch, verify finiteness.

    Returns one info dict per core; raises :class:`ModelBuildError` listing every
    failure when any core cannot be built/forwarded.
    """
    from pes2ts_v5.configs.schema import CoreConfig, HeadConfig, ModelConfig

    batch = make_smoke_batch(batch_size=batch_size, seq_len=seq_len, device=device)
    results: list[dict] = []
    failures: list[str] = []
    for name in list_cores():
        cfg = ModelConfig(core=CoreConfig(name=name, hidden_dim=hidden_dim),
                          head=HeadConfig())
        try:
            model = build_model_from_config(cfg).to(device)
            model.eval()
            with torch.no_grad():
                out = model(batch)
            finite = all(torch.isfinite(v).all().item()
                         for k, v in out.items()
                         if torch.is_tensor(v) and k != "state")
            if not finite:
                raise ModelBuildError("NON_FINITE_FORWARD",
                                      f"non-finite output for core {name}")
            results.append({
                "name": name, "ok": True, "total_params": count_params(model),
                "required_fields": sorted(required_fields_for(name)),
            })
        except Exception as exc:
            failures.append(f"{name}: {type(exc).__name__}: {exc}")
    if failures:
        raise ModelBuildError(
            "REGISTRY_NOT_BUILDABLE",
            f"{len(failures)}/{len(list_cores())} cores failed: " + " | ".join(failures))
    return results


__all__ = [
    "BASE_BATCH_FIELDS",
    "CORE_REGISTRY",
    "CoreSpec",
    "ENCODER_REGISTRY",
    "EncoderSpec",
    "HEAD_REGISTRY",
    "HeadSpec",
    "ModelBuildError",
    "TIME_AWARE_FIELDS",
    "assert_all_buildable",
    "assert_model_config_runnable",
    "build_model_from_config",
    "describe_registry",
    "list_cores",
    "list_encoders",
    "list_heads",
    "make_smoke_batch",
    "register_core",
    "register_encoder",
    "register_head",
    "required_fields_for",
    "resolve_core_hidden_dim",
    "resolve_core_name",
    "resolve_encoder_name",
    "resolve_head_name",
]


# dataclass field import kept explicitly for callers relying on CoreSpec.aliases order
_ = field
