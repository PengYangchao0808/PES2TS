#!/usr/bin/env python3
"""ADR-0030 documentation consistency gate (forbidden-token + banner scan).

Single source of truth: ``docs/design/decisions/ADR-0030-V5.2全路径导航与端点停止.md``
and ``docs/contracts/PES2TS_FullPathNavigationContract_v1.md`` (Full-Path
Navigation; Move crosses TS; No-Move = Endpoint Stop with d_end windows
0.10/0.30 Å; TS is a waypoint that never triggers stop).  Deprecated /
retracted concepts (``TSStop``, ``TS Stop``, ``TS-neighborhood``,
``approach-side``, ``TS-directed``, ``No-Move candidate``) must not appear in
ACTIVE docs.

Scope:

* all ``*.md`` under ``docs/`` — excluding the ``docs/archive/`` prefix and
  every path listed in the ``historical`` config list;
* the ``root_scope`` files (repo root markdown), minus historical ones.

Rules:

1. **forbidden-token rule** — a line violates if it contains a forbidden token
   and carries no exemption marker (``弃用``/``Superseded``/...) on the same
   line; reported as ``file:line:token``;
2. **historical rule** — every file in the ``historical`` list must exist and
   contain the banner marker (``ADR-0030``); ``docs/archive/`` is exempt by
   prefix with no banner required;
3. **required-files rule** — each required file must exist and contain every
   required substring.

Modes: the CLI always prints a summary (files scanned, violations grouped by
category).  ``--strict`` exits 1 on ANY violation; advisory mode (default)
prints violations but exits 0.

Importable API: ``run_checks(root: Path, config_path: Path | None = None)``
returns the machine-readable report dict; ``main(argv)`` is the CLI entry.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

__all__ = [
    "DEFAULT_CONFIG",
    "DocsConsistencyConfigError",
    "load_config",
    "main",
    "run_checks",
]

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = SCRIPT_DIR / "docs_consistency_config.json"
ARCHIVE_PREFIX = ("docs", "archive")

CONFIG_KEYS = (
    "forbidden_tokens",
    "exemption_markers",
    "historical",
    "root_scope",
    "required_files",
    "historical_banner_marker",
)

#: optional config key — path prefixes skipped by the forbidden-token scan
#: only (operational logs / handoffs); no banner is required for them.
OPTIONAL_LIST_KEYS = ("token_exempt_prefixes",)


class DocsConsistencyConfigError(ValueError):
    """Config file is missing, unreadable, or missing required keys."""


def load_config(config_path: Path) -> Dict[str, Any]:
    """Load and validate the consistency config JSON (pure stdlib)."""
    if not config_path.is_file():
        raise DocsConsistencyConfigError(f"config not found: {config_path}")
    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DocsConsistencyConfigError(
            f"cannot read config {config_path}: {exc}"
        ) from exc
    if not isinstance(raw, dict):
        raise DocsConsistencyConfigError(f"config root must be an object: {config_path}")
    missing = [key for key in CONFIG_KEYS if key not in raw]
    if missing:
        raise DocsConsistencyConfigError(
            f"config missing key(s) {', '.join(missing)}: {config_path}"
        )
    for key in ("forbidden_tokens", "exemption_markers", "historical", "root_scope"):
        if not isinstance(raw[key], list):
            raise DocsConsistencyConfigError(f"config key {key!r} must be a list")
    if not isinstance(raw["required_files"], dict):
        raise DocsConsistencyConfigError("config key 'required_files' must be an object")
    if not isinstance(raw["historical_banner_marker"], str) or not raw["historical_banner_marker"]:
        raise DocsConsistencyConfigError("config key 'historical_banner_marker' must be a non-empty string")
    return raw


def _is_archive(path: Path, root: Path) -> bool:
    """True when *path* sits under the exempt ``docs/archive/`` prefix."""
    try:
        parts = path.relative_to(root).parts
    except ValueError:
        return False
    return parts[:2] == ARCHIVE_PREFIX


def _scope_files(root: Path, cfg: Dict[str, Any]) -> List[Path]:
    """Active-doc scan scope: docs/**.md (minus archive/historical) + root_scope.

    Files under any ``token_exempt_prefixes`` entry are ALSO excluded — those
    are operational records (e.g. ``docs/agent_runs/``) where deprecated
    tokens may appear historically without an exemption marker and without
    requiring an ADR-0030 banner.
    """
    historical = set(cfg["historical"])
    exempt_prefixes = tuple(cfg.get("token_exempt_prefixes") or ())
    files: List[Path] = []
    docs = root / "docs"
    if docs.is_dir():
        for path in sorted(docs.rglob("*.md")):
            rel = path.relative_to(root).as_posix()
            if _is_archive(path, root) or rel in historical:
                continue
            if rel.startswith(exempt_prefixes):
                continue
            files.append(path)
    for rel in cfg["root_scope"]:
        if rel in historical:
            continue
        path = root / rel
        if path.is_file():
            files.append(path)
    return files


def _check_forbidden_tokens(
    root: Path, cfg: Dict[str, Any]
) -> List[Dict[str, Any]]:
    violations: List[Dict[str, Any]] = []
    tokens: List[str] = cfg["forbidden_tokens"]
    markers: List[str] = cfg["exemption_markers"]
    for path in _scope_files(root, cfg):
        rel = path.relative_to(root).as_posix()
        text = path.read_text(encoding="utf-8", errors="replace")
        for lineno, line in enumerate(text.splitlines(), start=1):
            if not any(token in line for token in tokens):
                continue
            if any(marker in line for marker in markers):
                continue
            for token in tokens:
                if token in line:
                    violations.append(
                        {
                            "category": "forbidden_token",
                            "file": rel,
                            "line": lineno,
                            "token": token,
                        }
                    )
    return violations


def _check_historical(root: Path, cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    violations: List[Dict[str, Any]] = []
    marker: str = cfg["historical_banner_marker"]
    for rel in cfg["historical"]:
        path = root / rel
        if not path.is_file():
            violations.append(
                {"category": "historical_missing_file", "file": rel}
            )
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        if marker not in text:
            violations.append(
                {
                    "category": "historical_missing_banner",
                    "file": rel,
                    "marker": marker,
                }
            )
    return violations


def _check_required_files(root: Path, cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    violations: List[Dict[str, Any]] = []
    required: Dict[str, Any] = cfg["required_files"]
    for rel, substrings in required.items():
        path = root / rel
        if not path.is_file():
            violations.append({"category": "required_missing_file", "file": rel})
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for substring in substrings:
            if substring not in text:
                violations.append(
                    {
                        "category": "required_substring_missing",
                        "file": rel,
                        "substring": substring,
                    }
                )
    return violations


def run_checks(root: Path, config_path: Optional[Path] = None) -> Dict[str, Any]:
    """Run all three rule sets against *root*; return the report dict.

    ``config_path`` defaults to the JSON next to this script; tests inject a
    config via a temporary repo tree.  Report keys: ``root``, ``config``,
    ``n_files_scanned``, ``violations`` (list of category dicts), ``clean``.
    """
    root = Path(root)
    cfg_path = Path(config_path) if config_path is not None else DEFAULT_CONFIG
    cfg = load_config(cfg_path)

    scanned = _scope_files(root, cfg)
    violations: List[Dict[str, Any]] = []
    violations.extend(_check_forbidden_tokens(root, cfg))
    violations.extend(_check_historical(root, cfg))
    violations.extend(_check_required_files(root, cfg))

    return {
        "root": str(root),
        "config": str(cfg_path),
        "n_files_scanned": len(scanned),
        "violations": violations,
        "clean": not violations,
    }


def _format_violation(v: Dict[str, Any]) -> str:
    category = v["category"]
    if category == "forbidden_token":
        return f"{v['file']}:{v['line']}:{v['token']}"
    if category == "historical_missing_file":
        return f"{v['file']}: historical file missing"
    if category == "historical_missing_banner":
        return f"{v['file']}: missing banner marker {v['marker']!r}"
    if category == "required_missing_file":
        return f"{v['file']}: required file missing"
    if category == "required_substring_missing":
        return f"{v['file']}: missing required substring {v['substring']!r}"
    return str(v)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="ADR-0030 documentation consistency gate "
        "(forbidden tokens, historical banners, required substrings)."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=SCRIPT_DIR.parents[1],
        help="repository root to scan (default: repo root of this script)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="config JSON path (default: docs_consistency_config.json next to this script)",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="exit 1 on any violation (default: advisory, exit 0)",
    )
    args = parser.parse_args(argv)

    try:
        report = run_checks(args.root, config_path=args.config)
    except DocsConsistencyConfigError as exc:
        print(f"CONFIG ERROR {exc}", file=sys.stderr)
        return 2

    by_category: Dict[str, int] = {}
    for violation in report["violations"]:
        by_category[violation["category"]] = by_category.get(violation["category"], 0) + 1

    for violation in report["violations"]:
        print(f"VIOLATION {_format_violation(violation)}")

    status = "CLEAN" if report["clean"] else "VIOLATIONS"
    summary = (
        f"docs_consistency: scanned {report['n_files_scanned']} file(s); "
        f"{len(report['violations'])} violation(s)"
    )
    if by_category:
        detail = ", ".join(
            f"{category}={count}" for category, count in sorted(by_category.items())
        )
        summary += f" [{detail}]"
    print(f"{summary}; {status}")

    if args.strict and not report["clean"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
