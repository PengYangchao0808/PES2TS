#!/usr/bin/env python
"""Check relative Markdown links in live docs.

Scope: README.md, AGENT.md, and docs/**/*.md except docs/archive/**.
- missing relative file links are errors;
- absolute ``E:/...`` and ``outputs/...`` links are warnings (historical local
  evidence; not resolvable on GitHub);
- external http(s) links are not fetched.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from urllib.parse import unquote

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LINK = re.compile(r"\[[^\]]*\]\(([^)]+)\)")
SCHEMES = ("http://", "https://", "mailto:", "tel:", "#")


def _iter_docs() -> list[Path]:
    files = [PROJECT_ROOT / "README.md", PROJECT_ROOT / "AGENT.md"]
    docs = PROJECT_ROOT / "docs"
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


def main() -> int:
    errors: list[str] = []
    warnings: list[str] = []
    for path in _iter_docs():
        text = path.read_text(encoding="utf-8")
        for target in _targets(text):
            if not target or target.startswith(SCHEMES):
                continue
            if target.startswith("E:/") or target.startswith("e:/"):
                warnings.append(f"{path.relative_to(PROJECT_ROOT)}: local E:/ link {target}")
                continue
            cleaned = unquote(target.split("#", 1)[0])
            if not cleaned:
                continue
            if (
                cleaned.startswith("outputs/")
                or cleaned.startswith("../outputs/")
                or ".slim/" in cleaned
                or ".omo/" in cleaned
            ):
                warnings.append(
                    f"{path.relative_to(PROJECT_ROOT)}: local-only link {target}"
                )
                continue
            resolved = (
                (PROJECT_ROOT / cleaned.lstrip("/"))
                if cleaned.startswith("/")
                else (path.parent / cleaned)
            )
            if not resolved.exists():
                errors.append(
                    f"{path.relative_to(PROJECT_ROOT)}: missing link target {target}"
                )

    for warning in warnings:
        print(f"WARN {warning}")
    for error in errors:
        print(f"ERROR {error}", file=sys.stderr)
    print(f"checked {len(_iter_docs())} docs; {len(errors)} error(s), {len(warnings)} warning(s)")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
