"""Shared fixtures for the V5.6 contract test suite.

CPU-only, tiny sizes, deterministic: no network, no GPU, no real data store.
Repository root is put on ``sys.path`` the same way ``tests/conftest.py`` does.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

REPO_ROOT = PROJECT_ROOT
CONFIG_DIR = REPO_ROOT / "config" / "experiments"


def pytest_configure(config) -> None:
    torch.set_num_threads(1)


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def synthetic_store(tmp_path_factory) -> Path:
    """Deterministic synthetic V5.5-shaped fixture (24 reactions, 48 trajectories)."""
    from pes2ts_v5.data.trajectories import build_synthetic_dataset

    out = tmp_path_factory.mktemp("v5_store") / "synthetic"
    build_synthetic_dataset(out, seed=17, n_trajectories_per_reaction=2)
    return out


@pytest.fixture(scope="session")
def dataset(synthetic_store: Path):
    from pes2ts_v5.data.trajectories import V5TrajectoryDataset

    return V5TrajectoryDataset(synthetic_store)


@pytest.fixture()
def one_batch(dataset):
    """First training batch (CPU tensors) + its meta rows."""
    batch, meta = next(iter(dataset.iter_batches(
        "train", batch_size=2, shuffle=False, seed=17, max_batches=1)))
    return batch, meta


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

@pytest.fixture()
def smoke_doc() -> dict:
    """The shipped smoke experiment as a plain dict."""
    from pes2ts_v5.configs.schema import config_to_dict, load_experiment_config

    return config_to_dict(load_experiment_config(CONFIG_DIR / "smoke.yaml"))


@pytest.fixture()
def write_yaml(tmp_path):
    """``write_yaml(doc) -> Path`` writer for temporary experiment configs."""
    import yaml

    counter = {"n": 0}

    def _write(doc: dict, name: str | None = None) -> Path:
        counter["n"] += 1
        path = tmp_path / (name or f"experiment_{counter['n']}.yaml")
        path.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
        return path

    return _write


# ---------------------------------------------------------------------------
# Model / trainer factories
# ---------------------------------------------------------------------------

@pytest.fixture()
def build_model():
    """``build_model(core_name, hidden_dim=32) -> V5Model`` via the registry."""
    from pes2ts_v5.configs.schema import CoreConfig, HeadConfig, ModelConfig
    from pes2ts_v5.models.registry import build_model_from_config

    def _build(core: str = "gru", hidden_dim: int = 32):
        cfg = ModelConfig(core=CoreConfig(name=core, hidden_dim=hidden_dim),
                          head=HeadConfig())
        return build_model_from_config(cfg)

    return _build


@pytest.fixture()
def build_trainer(dataset):
    """``build_trainer(core=..., **overrides) -> V5Trainer`` on the fixture dataset."""
    from pes2ts_v5.configs.schema import CoreConfig, HeadConfig, ModelConfig
    from pes2ts_v5.models.registry import build_model_from_config
    from pes2ts_v5.training.trainer import TrainConfig, V5Trainer

    def _build(core: str = "gru", *, out_dir=None, callbacks=None, loss_fn=None,
               experiment_id: str = "test-run", validate_contract: bool = True,
               **overrides):
        defaults = dict(
            batch_size=2, lr=0.01, max_epochs=1, patience=10, seed=17,
            device="cpu", cpu_threads=1, max_train_batches=1, max_val_batches=1,
            grad_accum=1, weight_decay=0.0,
        )
        defaults.update(overrides)
        cfg = TrainConfig(**defaults)
        model = build_model_from_config(
            ModelConfig(core=CoreConfig(name=core, hidden_dim=32),
                        head=HeadConfig()))
        return V5Trainer(model, dataset, cfg, out_dir=out_dir,
                         experiment_id=experiment_id, callbacks=callbacks,
                         loss_fn=loss_fn, config_hash="test-config-hash",
                         validate_contract=validate_contract)

    return _build


# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------

@pytest.fixture()
def json_read():
    """Read a strict-JSON file (fails on NaN/Inf, i.e. asserts valid JSON)."""

    def _read(path: str | Path):
        return json.loads(Path(path).read_text(encoding="utf-8"))

    return _read


def pytest_report_header(config) -> str:  # pragma: no cover - informational
    return f"tests/v5: repo_root={PROJECT_ROOT} torch={torch.__version__}"
