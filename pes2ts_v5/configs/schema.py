"""The **only** experiment-config contract for the V5.6 training/eval engine.

One YAML document (``config/experiments/*.yaml``) -> :class:`ExperimentConfig`;
everything downstream (model assembly, losses, trainer, runner) consumes this
object.  Unknown keys are rejected, registry-resolved names are validated at load
time, NaN/Inf are rejected, and :func:`config_hash` is a SHA256 over canonical
JSON (sorted keys, compact separators) — stable across processes/dict ordering.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Literal, Mapping

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

SCHEMA_NAME = "pes2ts_v5_experiment_v1"
EXPERIMENT_ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$"
#: split policies meaning "build/read the synthetic fixture"; anything else is a
#: real V5.5 store and must ship a manifest.
SYNTHETIC_SPLIT_POLICIES = ("synthetic_fixture", "synthetic_fixture_v1")
DEFAULT_LOSS_COMPONENTS = (("move_huber", 1.0), ("process_ce", 0.5),
                           ("progress_huber", 0.2), ("trust_bce", 0.2))
DEFAULT_CALLBACKS = ("checkpoint_best", "cost_ledger", "provenance_ledger",
                     "log", "early_stop")
HIDDEN_DIM_MIN, HIDDEN_DIM_MAX = 16, 1024


class ConfigError(ValueError):
    """Raised for any unreadable / invalid experiment config document."""


class _StrictModel(BaseModel):
    """Base: reject unknown keys and NaN/Inf values."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


def _require_nonempty(value: str, field_name: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{field_name} must be a non-empty string")
    return text


class DataConfig(_StrictModel):
    """Data source and split policy (explicit dims are checked, never defaulted)."""

    store_manifest: str
    split_policy: str
    splits: dict[str, str] | None = None
    max_frames: int | None = Field(default=None, ge=1)
    limit: int | None = Field(default=None, ge=1)
    frame_feat_dim: int | None = Field(default=None, ge=1, le=4096)
    static_feat_dim: int | None = Field(default=None, ge=1, le=4096)

    @field_validator("store_manifest", "split_policy")
    @classmethod
    def _nonempty(cls, value: str, info) -> str:
        return _require_nonempty(value, info.field_name)

    @field_validator("splits")
    @classmethod
    def _split_map(cls, value: dict[str, str] | None) -> dict[str, str] | None:
        if value is None:
            return None
        return {_require_nonempty(k, "data.splits key"):
                _require_nonempty(v, "data.splits value") for k, v in value.items()}


class CoreConfig(_StrictModel):
    """Memory-core (M01..M14 or algorithm alias) and its capacity budget."""

    name: str
    hidden_dim: int | Literal["auto"] = "auto"
    layers: int = Field(default=2, ge=1, le=12)
    dropout: float = Field(default=0.0, ge=0.0, le=0.9)
    param_budget: int = Field(default=100_000, ge=16, le=100_000_000)

    @field_validator("name")
    @classmethod
    def _known_core(cls, value: str) -> str:
        from pes2ts_v5.models.registry import CORE_REGISTRY, resolve_core_name

        text = _require_nonempty(value, "model.core.name")
        try:
            resolve_core_name(text)
        except KeyError as exc:
            raise ValueError(
                f"unknown core {text!r}; canonical={sorted(CORE_REGISTRY)}") from exc
        return text

    @field_validator("hidden_dim")
    @classmethod
    def _hidden_dim_range(cls, value: int | str) -> int | str:
        if isinstance(value, str):
            if value != "auto":
                raise ValueError("model.core.hidden_dim must be an int or 'auto'")
            return value
        if not HIDDEN_DIM_MIN <= int(value) <= HIDDEN_DIM_MAX:
            raise ValueError(
                f"model.core.hidden_dim must be in [{HIDDEN_DIM_MIN}, "
                f"{HIDDEN_DIM_MAX}] or 'auto', got {value}")
        return int(value)


class HeadConfig(_StrictModel):
    """Prediction head (Move / Process / Trust)."""

    name: str = "multi_head"
    q_dim: int = Field(default=3, ge=1, le=32)
    process_classes: int = Field(default=3, ge=2, le=64)
    dropout: float = Field(default=0.0, ge=0.0, le=0.9)

    @field_validator("name")
    @classmethod
    def _known_head(cls, value: str) -> str:
        from pes2ts_v5.models.registry import HEAD_REGISTRY

        text = _require_nonempty(value, "model.head.name")
        if text not in HEAD_REGISTRY:
            raise ValueError(f"unknown head {text!r}; registered={sorted(HEAD_REGISTRY)}")
        return text


class ModelConfig(_StrictModel):
    """Encoder + memory core + head (the model registry's assembly contract)."""

    encoder: str = "descriptor"
    core: CoreConfig
    head: HeadConfig = Field(default_factory=HeadConfig)

    @field_validator("encoder")
    @classmethod
    def _known_encoder(cls, value: str) -> str:
        from pes2ts_v5.models.registry import ENCODER_REGISTRY

        text = _require_nonempty(value, "model.encoder").lower()
        if text not in ENCODER_REGISTRY:
            raise ValueError(
                f"unknown encoder {text!r}; registered={sorted(ENCODER_REGISTRY)}")
        return text


class LossComponent(_StrictModel):
    name: str
    weight: float = Field(ge=0.0)

    @field_validator("name")
    @classmethod
    def _known_loss(cls, value: str) -> str:
        from pes2ts_v5.losses.registry import LOSS_REGISTRY

        text = _require_nonempty(value, "loss.components[].name")
        if text not in LOSS_REGISTRY:
            raise ValueError(
                f"unknown loss component {text!r}; registered={sorted(LOSS_REGISTRY)}")
        return text


class LossConfig(_StrictModel):
    components: list[LossComponent] = Field(
        default_factory=lambda: [LossComponent(name=n, weight=w)
                                 for n, w in DEFAULT_LOSS_COMPONENTS])

    @model_validator(mode="after")
    def _unique_names(self) -> "LossConfig":
        names = [c.name for c in self.components]
        if len(names) != len(set(names)):
            raise ValueError(f"duplicate loss component names: {names}")
        if not names:
            raise ValueError("loss.components must not be empty")
        return self


class OptimizerConfig(_StrictModel):
    name: Literal["adamw", "adam", "sgd"] = "adamw"
    lr: float | None = Field(default=None, gt=0.0)
    weight_decay: float | None = Field(default=None, ge=0.0)


class TrainingConfig(_StrictModel):
    """Trainer knobs, including the ordered callback list."""

    epochs: int = Field(default=1, ge=1, le=100_000)
    batch_size: int = Field(default=8, ge=1, le=4096)
    lr: float = Field(default=1e-3, gt=0.0)
    weight_decay: float = Field(default=0.01, ge=0.0)
    grad_clip: float = Field(default=1.0, ge=0.0)
    grad_accum: int = Field(default=1, ge=1, le=4096)
    patience: int = Field(default=3, ge=0)
    amp: bool = False
    ema_decay: float | None = Field(default=None, gt=0.0, lt=1.0)
    optimizer: OptimizerConfig = Field(default_factory=OptimizerConfig)
    callbacks: list[str] = Field(default_factory=lambda: list(DEFAULT_CALLBACKS))
    device: str = "auto"
    cpu_threads: int = Field(default=1, ge=0)

    @field_validator("callbacks")
    @classmethod
    def _known_callbacks(cls, value: list[str]) -> list[str]:
        from pes2ts_v5.training.callbacks import resolve_callback_name

        names = [_require_nonempty(name, "training.callbacks[]") for name in value]
        for name in names:
            try:
                resolve_callback_name(name)
            except KeyError as exc:
                raise ValueError(str(exc)) from exc
        return names

    @field_validator("device")
    @classmethod
    def _device(cls, value: str) -> str:
        import torch

        text = _require_nonempty(value, "training.device").lower()
        if text == "auto":
            return text
        try:
            torch.device(text)
        except (RuntimeError, ValueError) as exc:
            raise ValueError(f"invalid training.device {text!r}: {exc}") from exc
        return text


class EvalConfig(_StrictModel):
    splits: list[str] = Field(default_factory=lambda: ["valid"])
    ood: list[str] = Field(default_factory=list)
    cpu_benchmark: bool = True

    @field_validator("splits", "ood")
    @classmethod
    def _known_splits(cls, value: list[str], info) -> list[str]:
        from pes2ts_v5.data.manifest import SPLIT_NAMES

        names = [_require_nonempty(name, f"evaluation.{info.field_name}[]")
                 for name in value]
        unknown = [name for name in names if name not in SPLIT_NAMES]
        if unknown:
            raise ValueError(
                f"unknown split(s) {unknown} in evaluation.{info.field_name}; "
                f"known={list(SPLIT_NAMES)}")
        if info.field_name == "splits" and not names:
            raise ValueError("evaluation.splits must not be empty")
        return names


class ExperimentConfig(_StrictModel):
    """Complete, validated description of one training/evaluation run."""

    experiment_id: str
    seed: int = Field(default=17, ge=0, le=2 ** 31 - 1)
    data: DataConfig
    model: ModelConfig
    loss: LossConfig = Field(default_factory=LossConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    evaluation: EvalConfig = Field(default_factory=EvalConfig)

    @field_validator("experiment_id")
    @classmethod
    def _experiment_id(cls, value: str) -> str:
        text = _require_nonempty(value, "experiment_id")
        if not re.match(EXPERIMENT_ID_PATTERN, text):
            raise ValueError(
                "experiment_id must match "
                f"{EXPERIMENT_ID_PATTERN} (letters/digits start, then "
                "letters/digits/._-, max 64 chars)")
        return text

    def is_synthetic(self) -> bool:
        """True when ``data.split_policy`` selects the synthetic fixture."""
        return self.data.split_policy.strip().lower() in SYNTHETIC_SPLIT_POLICIES


def _format_validation_error(exc: ValidationError) -> str:
    return "; ".join(
        f"{'.'.join(str(x) for x in err.get('loc', ())) or '<root>'}: "
        f"{err.get('msg')}" for err in exc.errors())


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    """Load + validate one YAML experiment config (raises :class:`ConfigError`)."""
    config_path = Path(path)
    try:
        raw = config_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"cannot read config {config_path}: {exc}") from exc
    try:
        doc = yaml.safe_load(raw)
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid YAML in {config_path}: {exc}") from exc
    if not isinstance(doc, Mapping):
        raise ConfigError(
            f"{config_path} must contain a YAML mapping, got {type(doc).__name__}")
    try:
        return ExperimentConfig.model_validate(dict(doc))
    except ValidationError as exc:
        raise ConfigError(
            f"invalid experiment config {config_path}: "
            f"{_format_validation_error(exc)}") from exc


def config_to_dict(cfg: ExperimentConfig) -> dict[str, Any]:
    """Plain, JSON-native dict of the whole config (``Path``/enum free)."""
    return cfg.model_dump(mode="json")


def config_hash(cfg: ExperimentConfig) -> str:
    """SHA256 over canonical JSON of :class:`ExperimentConfig`."""
    canonical = json.dumps(config_to_dict(cfg), sort_keys=True,
                           separators=(",", ":"), ensure_ascii=False,
                           allow_nan=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
