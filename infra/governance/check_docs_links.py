#!/usr/bin/env python
"""Check relative Markdown links in live docs (frozen-history aware).

Scope: README.md, AGENT.md, and docs/**/*.md except docs/archive/**.
Policy (2026-11-24 three-plate cleanup):

* missing relative file links are **errors**, EXCEPT when the target resolves
  under a frozen-history prefix that was moved out of master to the read-only
  branch ``frozen/v4-v54-history`` -- ``archive/``, ``tests/legacy/``,
  ``pes2ts_core/pesn/``, ``scripts/`` or ``config/campaigns/``.  Those emit a
  WARN tagged ``frozen-history`` instead of failing the gate, because the
  evidence still exists (on the branch/tags) but is intentionally absent from
  the working tree;
* absolute ``E:/...`` and ``outputs/...`` links are warnings (historical local
  evidence; not resolvable on GitHub);
* external http(s) links are not fetched.

Everything else stays strict: a broken link to any non-frozen path is an error.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from urllib.parse import unquote

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LINK = re.compile(r"\[[^\]]*\]\(([^)]+)\)")
SCHEMES = ("http://", "https://", "mailto:", "tel:", "#")

#: Repo-relative prefixes whose history lives on ``frozen/v4-v54-history``.
FROZEN_HISTORY_PREFIXES = (
    "archive/",
    "tests/legacy/",
    "pes2ts_core/pesn/",
    "scripts/",
    "config/campaigns/",
    "pes2ts_data/builders/",
)

#: Warning text shared by every frozen-history hit (branch is the source of truth).
FROZEN_HISTORY_NOTE = "frozen-history: see branch frozen/v4-v54-history"


def _iter_docs(root: Path) -> list[Path]:
    files = [root / "README.md", root / "AGENT.md"]
    docs = root / "docs"
    for path in sorted(docs.rglob("*.md")):
        if "archive" in path.relative_to(docs).parts:
            continue
        files.append(path)
    return [path for path in files if path.is_file()]


def _targets(text: str) -> list[str]:
    targets: list[str] = []
    for raw in LINK.findall(text):
        target = raw.strip()
        if target.startswith("<") and target.endswith(">"):
            target = target[1:-1]
        # drop an optional Markdown title:  path "title"
        if " " in target and not target.startswith("<"):
            target = target.split(" ", 1)[0]
        targets.append(target)
    return targets


def _frozen_history_rel(target: str, doc: Path, root: Path) -> str | None:
    """Return the repo-relative target when it is frozen history, else ``None``.

    Resolution follows the same rule as the existence check: absolute-looking
    links resolve from the repo root, everything else from the linking doc.
    A target is frozen history when the *resolved* repo-relative path starts
    with one of :data:`FROZEN_HISTORY_PREFIXES`.
    """
    resolved = (
        (root / target.lstrip("/"))
        if target.startswith("/")
        else (doc.parent / target)
    )
    try:
        rel = resolved.resolve().relative_to(root.resolve()).as_posix()
    except (OSError, ValueError):
        return None
    if rel.startswith(FROZEN_HISTORY_PREFIXES):
        return rel
    return None


def main(argv: list[str] | None = None) -> int:
    """CLI: check relative links; exit 0 when there are no non-frozen errors."""
    parser = argparse.ArgumentParser(
        description="Check relative Markdown links in live docs (frozen-history "
        "targets warn instead of failing; everything else stays strict)."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=PROJECT_ROOT,
        help="repository root to scan (default: repo root of this script)",
    )
    args = parser.parse_args(argv)
    root = args.root.resolve()

    errors: list[str] = []
    warnings: list[str] = []
    for path in _iter_docs(root):
        text = path.read_text(encoding="utf-8")
        for target in _targets(text):
            if not target or target.startswith(SCHEMES):
                continue
            if target.startswith("E:/") or target.startswith("e:/"):
                warnings.append(f"{path.relative_to(root)}: local E:/ link {target}")
                continue
            cleaned = unquote(target.split("#", 1)[0])
            if not cleaned:
                continue
            parts = Path(cleaned).parts
            if (
                "outputs" in parts
                or ".slim" in parts
                or ".omo" in parts
                or cleaned.endswith(".log")
            ):
                warnings.append(
                    f"{path.relative_to(root)}: local-only link {target}"
                )
                continue
            resolved = (
                (root / cleaned.lstrip("/"))
                if cleaned.startswith("/")
                else (path.parent / cleaned)
            )
            if resolved.exists():
                continue
            frozen = _frozen_history_rel(cleaned, path, root)
            if frozen is not None:
                warnings.append(
                    f"{path.relative_to(root)}: {FROZEN_HISTORY_NOTE} "
                    f"(target {frozen})"
                )
                continue
            errors.append(
                f"{path.relative_to(root)}: missing link target {target}"
            )

    for warning in warnings:
        print(f"WARN {warning}")
    for error in errors:
        print(f"ERROR {error}", file=sys.stderr)
    print(
        f"checked {len(_iter_docs(root))} docs; "
        f"{len(errors)} error(s), {len(warnings)} warning(s) "
        f"[frozen-history tolerated: {sum(1 for w in warnings if FROZEN_HISTORY_NOTE in w)}]"
    )
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
