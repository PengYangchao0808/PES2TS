"""Load and merge PES2TS YAML configuration.

The bundled defaults live in ``config/defaults.yaml`` and are resolved relative
to the project root (derived from this file's location), so loading works from
any current working directory.  A user-supplied YAML file, when given, is
deep-merged on top of the defaults.
"""

from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "defaults.yaml"

#: Exit code returned by the CLI when configuration loading fails.
EXIT_CONFIG_ERROR = 2


class ConfigError(Exception):
    """Raised when a configuration file is missing, unreadable, or malformed."""


def _read_yaml_mapping(path: Path) -> dict[str, Any]:
    """Read *path* and return it as a YAML mapping, raising ``ConfigError``."""
    logger = logging.getLogger(__name__)
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
    except OSError as exc:
        message = f"Cannot read config file: {path} ({exc})"
        logger.error(message)
        raise ConfigError(message) from exc
    if data is None:
        return {}
    if not isinstance(data, dict):
        message = f"Config file must contain a YAML mapping: {path}"
        logger.error(message)
        raise ConfigError(message)
    return data


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Return a new mapping with *override* recursively merged onto *base*."""
    merged = copy.deepcopy(base)
    for key, value in override.items():
        current = merged.get(key)
        if isinstance(current, dict) and isinstance(value, dict):
            merged[key] = _deep_merge(current, value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load the PES2TS configuration.

    Parameters
    ----------
    path:
        Optional path to a user YAML file that overrides the bundled defaults.
        When ``None`` (the default), only ``config/defaults.yaml`` is returned.

    Returns
    -------
    dict
        The merged configuration as a plain dictionary.

    Raises
    ------
    ConfigError
        If *path* is given but does not exist, cannot be read, or is not a
        YAML mapping.
    """
    logger = logging.getLogger(__name__)
    defaults = _read_yaml_mapping(DEFAULT_CONFIG_PATH)
    if path is None:
        return defaults

    user_path = Path(path)
    if not user_path.is_absolute():
        user_path = Path.cwd() / user_path
    if not user_path.exists():
        message = f"User config not found: {user_path}"
        logger.error(message)
        raise ConfigError(message)

    override = _read_yaml_mapping(user_path)
    logger.debug("Merged user config %s over defaults %s", user_path, DEFAULT_CONFIG_PATH)
    return _deep_merge(defaults, override)
