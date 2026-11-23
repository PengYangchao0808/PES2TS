#!/usr/bin/env python3
"""Architecture boundary guard for the five-zone layout (ADR-0036).

Single source of truth: ``docs/design/decisions/ADR-0036-功能分区与单一训练引擎.md``.
The repository is split into five zones -- ``pes2ts_core/`` (protected; the
legacy ``pes2ts_core/pesn/`` model tree is frozen read-only), ``pes2ts_data/``
(contracts + splits + registry + builders), ``pes2ts_v5/`` (the *only*
training/evaluation engine), ``infra/`` (runtime + governance tooling) and
``archive/`` (frozen historical V4 / V5.x campaigns; never imported by active
code).  This module implements the five hard gates that keep that layout from
decaying back into campaign-versioned script piles:

G1 single training loop
    Only ``pes2ts_v5/training/**`` may contain calls to ``.backward()`` or to
    ``<optimizer>.step()``.  ``pes2ts_core/pesn/`` is frozen legacy and is not
    scanned (it is a read-only record of V4/V5.1-V5.4 training, not active
    code).  AST-only, so comments and docstrings never trigger the gate.

G2 no campaign-versioned code files
    A ``.py`` file in an active zone (``pes2ts_v5``, ``pes2ts_data``,
    ``infra``, and ``pes2ts_core`` minus the frozen ``pesn/`` tree) whose
    *stem* matches ``^v(?:5[4-9]\\d*|[6-9]\\d*)_`` is a violation.  That
    rejects ``v54_trainer.py`` / ``v541_adapter.py`` / ``v563_trainer.py`` /
    ``v6_core.py`` / ``v7_runner.py`` while keeping the legitimate
    ``pes2ts_v5/models/v5_model.py`` (stem ``v5_model`` does not match: the
    rule requires v + 54..59 (optionally + digits) or v6..v99, then ``_``).

G3 one-way imports
    AST ``Import``/``ImportFrom`` only; string literals and comments are not
    imports.

    * ``pes2ts_v5/**`` and ``pes2ts_data/**`` must not import
      ``pes2ts_core.pesn``, ``archive`` or ``tests.legacy``;
    * the protected ``pes2ts_core`` sub-packages (``utils``, ``g0``, ``g1``,
      ``generation``, ``integration``, ``ranking``, ``evaluation``) and
      ``pes2ts_core/cli.py`` must not import ``torch``, ``pes2ts_v5``,
      ``pes2ts_data`` or ``archive``;
    * ``infra/**`` must not import business code (``pes2ts_core``,
      ``pes2ts_v5``, ``pes2ts_data``, ``archive``);
    * ``tests/**`` outside ``tests/legacy/**`` must not import ``archive`` or
      ``tests.legacy``;
    * ``archive/**`` is a frozen *sink*: it may import ``infra.runtime`` (the
      documented exception used by the archived campaign runners) and nothing
      else from the active zones.  Two footnotes: archived scripts importing
      the old ``pes2ts_core`` API are grandfathered -- they are the historical
      callers of that code and are frozen, so the gate only enforces
      ``archive -> (pes2ts_v5 | pes2ts_data | infra-non-runtime)``.

G4 frozen-zone read-only
    ``git diff --name-only HEAD`` must not touch ``archive/**`` or
    ``pes2ts_core/pesn/**``.  Advisory by default (the working tree is
    routinely dirty mid-restructure); the gate is enforced only with
    ``--strict-frozen`` (CI usage on a clean checkout).

G5 active-zone lint
    ``python -m ruff check pes2ts_v5 pes2ts_data infra`` is run via
    subprocess.  Pre-existing findings are tolerated through the documented
    baseline ``infra/governance/arch_guard_baseline.json`` (entries keyed by
    ``(gate, path, code)``); any finding that is *not* in the baseline fails
    the gate, and a baseline entry that no longer matches any finding is
    reported as stale.  ``tests/legacy/`` and ``archive/`` are never linted.

Public API
----------
``Finding(path, line, gate, message, code="")``
    ``gate`` is one of ``G1``..``G5`` (``BASELINE`` marks stale entries).
``scan_g1`` .. ``scan_g5``
    Per-gate scanners returning raw (unbaselined) findings.
``scan(root, ...) -> list[Finding]``
    Convenience wrapper returning the unbaselined hard findings.
``run_guard(root, ...) -> dict``
    Full machine-readable report (findings, tolerances, stale baseline,
    ruff availability, frozen-zone advisory status).
``main(argv)``
    CLI entry (``python infra/governance/arch_guard.py [--root .] [--json]``).
    Exit code 1 on violation; 0 when clean.

The module is stdlib-only (no torch/pandas), so it runs in the torch-free
``tests/kernel`` tier.  Subprocess use (git/ruff) is confined to this
``infra/`` tool on purpose; ADR-0036 documents that exception.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

__all__ = [
    "ACTIVE_ZONES",
    "BASELINE_PATH",
    "Finding",
    "GuardConfigError",
    "RuffUnavailable",
    "frozen_paths",
    "load_baseline",
    "main",
    "run_guard",
    "scan",
    "scan_g1",
    "scan_g2",
    "scan_g3",
    "scan_g4",
    "scan_g5",
]

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_ROOT = SCRIPT_DIR.parents[1]
BASELINE_PATH = SCRIPT_DIR / "arch_guard_baseline.json"
BASELINE_SCHEMA = 1

# --- gate scopes -----------------------------------------------------------

#: Zones that hold active code.  ``pes2ts_core/pesn/`` is carved out below
#: because it is the frozen legacy model tree (read-only record, ADR-0036).
ACTIVE_ZONES = ("pes2ts_v5", "pes2ts_data", "infra", "pes2ts_core")
FROZEN_MODEL_PREFIX = "pes2ts_core/pesn/"
FROZEN_PREFIXES = ("archive/", FROZEN_MODEL_PREFIX)
TRAINING_PREFIX = "pes2ts_v5/training/"

CORE_PROTECTED_PREFIXES = (
    "pes2ts_core/utils/",
    "pes2ts_core/g0/",
    "pes2ts_core/g1/",
    "pes2ts_core/generation/",
    "pes2ts_core/integration/",
    "pes2ts_core/ranking/",
    "pes2ts_core/evaluation/",
)
CORE_PROTECTED_FILES = ("pes2ts_core/cli.py",)

RUFF_ZONES = ("pes2ts_v5", "pes2ts_data", "infra")

#: Campaign-versioned code names: ``v54_..v59_`` / ``v563_`` / ``v541_``
#: (V5.4+ sub-campaigns) and ``v6_..v99_``.  ``v5_model.py`` (stem
#: ``v5_model``) does NOT match: the rule requires v + 54..59 (optionally
#: followed by more digits) or v6..v99, followed by ``_``.
CAMPAIGN_VERSION_RE = re.compile(r"^v(?:5[4-9]\d*|[6-9]\d*)_")

V5_DATA_FORBIDDEN = ("pes2ts_core.pesn", "archive", "tests.legacy")
CORE_FORBIDDEN = ("torch", "pes2ts_v5", "pes2ts_data", "archive")
INFRA_FORBIDDEN = ("pes2ts_core", "pes2ts_v5", "pes2ts_data", "archive")
TESTS_FORBIDDEN = ("archive", "tests.legacy")
ARCHIVE_FORBIDDEN = ("pes2ts_v5", "pes2ts_data")
ARCHIVE_INFRA_ALLOWED = ("infra.runtime",)

#: Marker in the current commit message that waives G4 (owner-approved
#: frozen-zone unlock, e.g. a deliberately logged legacy fix).
UNLOCK_MARKER = "[unlock-legacy]"


class GuardConfigError(ValueError):
    """Baseline/config file is missing, unreadable or malformed."""


class RuffUnavailable(RuntimeError):
    """``python -m ruff`` is not installed or produced unusable output."""


# --- findings --------------------------------------------------------------


@dataclass(frozen=True)
class Finding:
    """One architecture-gate finding.

    ``gate`` is ``G1``..``G5`` (``BASELINE`` for stale tolerance entries).
    ``code`` is a stable rule key used by the baseline (a ruff rule code for
    G5, ``import:<target>`` for G3); it is empty for text-free gates.
    """

    path: str
    line: int
    gate: str
    message: str
    code: str = ""

    def to_dict(self) -> Dict[str, object]:
        return {
            "path": self.path,
            "line": self.line,
            "gate": self.gate,
            "message": self.message,
            "code": self.code,
        }


# --- file / AST helpers ----------------------------------------------------


def _iter_py(root: Path, bases: Iterable[str]) -> Iterator[Tuple[Path, str]]:
    """Yield ``(abs_path, repo-relative posix path)`` for ``*.py`` under bases."""
    for base in bases:
        base_dir = root / base
        if not base_dir.is_dir():
            continue
        for path in sorted(base_dir.rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            yield path, path.relative_to(root).as_posix()


def _iter_active_py(root: Path) -> Iterator[Tuple[Path, str]]:
    """Active-zone ``*.py`` files (frozen ``pes2ts_core/pesn/`` excluded)."""
    for path, rel in _iter_py(root, ACTIVE_ZONES):
        if rel.startswith(FROZEN_MODEL_PREFIX):
            continue
        yield path, rel


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


#: Per-run source cache: ``{repo-relative path: source text}``.  ``run_guard``
#: shares it between G1 and G3 so a 44万行 tree is read once, not four times.
TextCache = Dict[str, str]


def _module_text(path: Path, rel: str, cache: Optional[TextCache] = None) -> str:
    """Read module source once per guard run."""
    if cache is None:
        return _read_text(path)
    if rel not in cache:
        cache[rel] = _read_text(path)
    return cache[rel]


def _parse_text(text: str, path: Path) -> Optional[ast.Module]:
    """Parse module source; unparseable files are skipped (never expected in-tree)."""
    try:
        return ast.parse(text, filename=str(path))
    except SyntaxError:
        return None


def _imports(tree: ast.Module) -> Iterator[Tuple[int, str, Tuple[str, ...]]]:
    """Yield ``(lineno, module, from_names)`` for absolute imports only.

    Relative imports (``from .x import y``) stay inside their own top-level
    package and can never cross a zone boundary, so they are ignored.
    """
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield node.lineno, alias.name, ()
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                continue
            yield node.lineno, node.module or "", tuple(a.name for a in node.names)


def _import_matches(module: str, names: Sequence[str], target: str) -> bool:
    """True when an absolute import reaches dotted *target*."""
    if module == target or module.startswith(target + "."):
        return True
    # ``from pes2ts_core import pesn`` reaches target ``pes2ts_core.pesn``.
    if "." in target:
        parent, child = target.rsplit(".", 1)
        if module == parent and child in names:
            return True
    return False


def _match_imports(
    rel: str,
    imports: Iterable[Tuple[int, str, Tuple[str, ...]]],
    forbidden: Iterable[str],
    scope_label: str,
) -> List[Finding]:
    findings: List[Finding] = []
    targets = tuple(forbidden)
    for lineno, module, names in imports:
        for target in targets:
            if _import_matches(module, names, target):
                findings.append(
                    Finding(
                        rel,
                        lineno,
                        "G3",
                        f"forbidden import of {target!r} via {module or '?'} "
                        f"({scope_label})",
                        code=f"import:{target}",
                    )
                )
                break
    return findings


def _is_core_protected(rel: str) -> bool:
    return rel in CORE_PROTECTED_FILES or rel.startswith(CORE_PROTECTED_PREFIXES)


# --- G1 single training loop ----------------------------------------------


def _is_optimizer_receiver(expr: ast.expr) -> bool:
    try:
        text = ast.unparse(expr).lower().replace(" ", "")
    except Exception:  # pragma: no cover - ast.unparse is stdlib and total
        return False
    return "optimizer" in text or text.startswith("opt") or ".opt" in text


def _is_backward_call(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "backward"
    )


def _is_optimizer_step_call(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "step"
        and _is_optimizer_receiver(node.func.value)
    )


def scan_g1(
    root: Path, *, _cache: Optional[TextCache] = None
) -> List[Finding]:
    """G1: only ``pes2ts_v5/training/**`` may run the backward/step loop.

    The AST is authoritative; the textual prefilter (``.backward``/``.step``)
    only skips files that cannot contain such a call (comments/docstrings are
    never parsed as calls), which keeps the gate fast on the whole tree.
    """
    findings: List[Finding] = []
    for path, rel in _iter_active_py(root):
        if rel.startswith(TRAINING_PREFIX):
            continue
        text = _module_text(path, rel, _cache)
        if ".backward" not in text and ".step" not in text:
            continue
        tree = _parse_text(text, path)
        if tree is None:
            continue
        for node in ast.walk(tree):
            if _is_backward_call(node):
                findings.append(
                    Finding(
                        rel,
                        node.lineno,
                        "G1",
                        "backward() outside pes2ts_v5/training/ "
                        "(single-trainer rule)",
                    )
                )
            elif _is_optimizer_step_call(node):
                findings.append(
                    Finding(
                        rel,
                        node.lineno,
                        "G1",
                        "optimizer.step() outside pes2ts_v5/training/ "
                        "(single-trainer rule)",
                    )
                )
    return findings


# --- G2 no campaign-versioned code files ----------------------------------


def scan_g2(root: Path) -> List[Finding]:
    """G2: active zones may not hold campaign-versioned ``.py`` file names."""
    findings: List[Finding] = []
    for _, rel in _iter_active_py(root):
        stem = Path(rel).stem
        if CAMPAIGN_VERSION_RE.match(stem):
            findings.append(
                Finding(
                    rel,
                    0,
                    "G2",
                    f"campaign-versioned code file {stem!r} in an active zone "
                    "(one engine, config-as-experiment)",
                )
            )
    return findings


# --- G3 one-way imports ----------------------------------------------------

#: Conservative textual prefilters run before AST parsing.  Each token set is a
#: superset of every syntactically possible forbidden import in its scope, so
#: skipping a non-matching file can never hide a finding.
_V5_DATA_TOKENS = ("archive", "pesn", "tests")
_CORE_TOKENS = ("torch", "pes2ts_v5", "pes2ts_data", "archive")
_INFRA_TOKENS = ("pes2ts_core", "pes2ts_v5", "pes2ts_data", "archive")
_TESTS_TOKENS = ("archive", "from tests import", "tests.legacy")
_ARCHIVE_TOKENS = ("infra", "pes2ts_v5", "pes2ts_data")


def _g3_scope(rel: str) -> Optional[Tuple[Tuple[str, ...], Tuple[str, ...], str]]:
    """Return ``(forbidden targets, prefilter tokens, label)`` for a G3 scope."""
    if rel.startswith(("pes2ts_v5/", "pes2ts_data/")):
        return V5_DATA_FORBIDDEN, _V5_DATA_TOKENS, "v5/data zone"
    if _is_core_protected(rel):
        return CORE_FORBIDDEN, _CORE_TOKENS, "pes2ts_core protected"
    if rel.startswith("infra/"):
        return INFRA_FORBIDDEN, _INFRA_TOKENS, "infra zone"
    return None


def scan_g3(
    root: Path, *, _cache: Optional[TextCache] = None
) -> List[Finding]:
    """G3: enforce the one-way dependency matrix described in the docstring."""
    findings: List[Finding] = []

    for path, rel in _iter_active_py(root):
        scope = _g3_scope(rel)
        if scope is None:
            continue
        forbidden, tokens, label = scope
        text = _module_text(path, rel, _cache)
        if not any(token in text for token in tokens):
            continue
        tree = _parse_text(text, path)
        if tree is None:
            continue
        findings.extend(_match_imports(rel, list(_imports(tree)), forbidden, label))

    for path, rel in _iter_py(root, ("tests",)):
        if rel.startswith("tests/legacy/"):
            continue
        text = _module_text(path, rel, _cache)
        if not any(token in text for token in _TESTS_TOKENS):
            continue
        tree = _parse_text(text, path)
        if tree is None:
            continue
        findings.extend(
            _match_imports(rel, list(_imports(tree)), TESTS_FORBIDDEN, "tests zone")
        )

    for path, rel in _iter_py(root, ("archive",)):
        text = _module_text(path, rel, _cache)
        if not any(token in text for token in _ARCHIVE_TOKENS):
            continue
        tree = _parse_text(text, path)
        if tree is None:
            continue
        findings.extend(_scan_archive_imports(rel, list(_imports(tree))))

    return findings


def _scan_archive_imports(
    rel: str, imports: Iterable[Tuple[int, str, Tuple[str, ...]]]
) -> List[Finding]:
    """Frozen sink rule: archive may only reach ``infra.runtime``."""
    findings: List[Finding] = []
    for lineno, module, names in imports:
        top = module.split(".", 1)[0] if module else ""
        if top in ARCHIVE_FORBIDDEN:
            findings.append(
                Finding(
                    rel,
                    lineno,
                    "G3",
                    f"archive is a frozen sink but imports {module!r} "
                    "(new-zone import is forbidden)",
                    code=f"import:{top}",
                )
            )
            continue
        if top != "infra":
            continue
        allowed = module == ARCHIVE_INFRA_ALLOWED[0] or module.startswith(
            ARCHIVE_INFRA_ALLOWED[0] + "."
        )
        if module == "infra" and set(names) == {"runtime"}:
            allowed = True
        if not allowed:
            findings.append(
                Finding(
                    rel,
                    lineno,
                    "G3",
                    f"archive may only import infra.runtime (documented sink "
                    f"exception); got {module!r}",
                    code="import:infra-non-runtime",
                )
            )
    return findings


# --- G4 frozen-zone read-only ----------------------------------------------


def _run_git(root: Path, *args: str) -> Optional[str]:
    """Run git in *root*; return stdout or ``None`` when git is unavailable."""
    try:
        proc = subprocess.run(
            ["git", *args],
            cwd=str(root),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=60,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout


def frozen_paths(
    root: Path, diff_paths: Optional[Sequence[str]] = None
) -> List[str]:
    """Repo-relative paths changed vs ``HEAD`` that live in a frozen zone."""
    if diff_paths is None:
        raw = _run_git(root, "diff", "--name-only", "HEAD")
        paths = [] if raw is None else raw.splitlines()
    else:
        paths = list(diff_paths)
    changed: List[str] = []
    for raw_path in paths:
        rel = raw_path.strip().replace("\\", "/")
        if not rel:
            continue
        if any(rel == prefix.rstrip("/") for prefix in FROZEN_PREFIXES):
            changed.append(rel)
        elif rel.startswith(FROZEN_PREFIXES):
            changed.append(rel)
    return changed


def _commit_message(root: Path, commit_message: Optional[str]) -> Optional[str]:
    if commit_message is not None:
        return commit_message
    return _run_git(root, "log", "-1", "--format=%B")


def scan_g4(
    root: Path,
    *,
    strict_frozen: bool = False,
    diff_paths: Optional[Sequence[str]] = None,
    commit_message: Optional[str] = None,
) -> List[Finding]:
    """G4: advisory by default; enforce only with ``strict_frozen=True``."""
    if not strict_frozen:
        return []
    message = _commit_message(root, commit_message) or ""
    if UNLOCK_MARKER in message:
        return []
    return [
        Finding(
            rel,
            0,
            "G4",
            "frozen zone modified without the [unlock-legacy] commit waiver "
            "(archive/ and pes2ts_core/pesn/ are read-only)",
        )
        for rel in frozen_paths(root, diff_paths)
    ]


# --- G5 active-zone lint ---------------------------------------------------


def _ruff_check(root: Path) -> List[Finding]:
    """Run ``python -m ruff check`` on the active zones; return findings."""
    zones = [zone for zone in RUFF_ZONES if (root / zone).is_dir()]
    if not zones:
        return []
    cmd = [
        sys.executable,
        "-m",
        "ruff",
        "check",
        *zones,
        "--output-format=json",
        "--no-cache",
    ]
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(root),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=180,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:  # pragma: no cover
        raise RuffUnavailable(f"cannot run ruff: {exc}") from exc
    stdout = (proc.stdout or "").strip()
    stderr = (proc.stderr or "").strip()
    if not stdout:
        if "No module named" in stderr or "No module named" in (proc.stdout or ""):
            raise RuffUnavailable("ruff is not installed in this interpreter")
        if proc.returncode == 0:
            return []
        raise RuffUnavailable(f"ruff exited {proc.returncode}: {stderr[:300]}")
    try:
        payload = json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise RuffUnavailable(f"unparseable ruff output: {stdout[:300]}") from exc
    if not isinstance(payload, list):
        raise RuffUnavailable("ruff JSON output is not a list")
    root_resolved = root.resolve()
    findings: List[Finding] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        filename = str(item.get("filename") or "")
        try:
            rel = Path(filename).resolve().relative_to(root_resolved).as_posix()
        except (OSError, ValueError):
            rel = filename.replace("\\", "/")
        location = item.get("location") or {}
        line = location.get("row", 0) if isinstance(location, dict) else 0
        code = str(item.get("code") or "RUFF")
        message = str(item.get("message") or "").strip()
        findings.append(Finding(rel, int(line or 0), "G5", f"{code} {message}", code))
    return findings


def scan_g5(
    root: Path, ruff_runner: Optional[Callable[[Path], List[Finding]]] = None
) -> List[Finding]:
    """G5: ruff findings in the three active zones (raw, unbaselined)."""
    runner = ruff_runner if ruff_runner is not None else _ruff_check
    return list(runner(root))


# --- baseline --------------------------------------------------------------


def load_baseline(baseline_path: Optional[Path] = None) -> List[Dict[str, object]]:
    """Load and validate the tolerated-findings baseline."""
    path = Path(baseline_path) if baseline_path is not None else BASELINE_PATH
    if not path.is_file():
        return []
    try:
        raw = json.loads(_read_text(path))
    except json.JSONDecodeError as exc:
        raise GuardConfigError(f"invalid baseline JSON {path}: {exc}") from exc
    if not isinstance(raw, dict) or raw.get("schema") != BASELINE_SCHEMA:
        raise GuardConfigError(
            f"baseline {path} must be an object with schema={BASELINE_SCHEMA}"
        )
    entries = raw.get("entries")
    if not isinstance(entries, list):
        raise GuardConfigError(f"baseline {path} key 'entries' must be a list")
    normalized: List[Dict[str, object]] = []
    for entry in entries:
        if not isinstance(entry, dict):
            raise GuardConfigError(f"baseline entry is not an object: {entry!r}")
        gate = entry.get("gate")
        rel = entry.get("path")
        code = entry.get("code")
        count = entry.get("count")
        if not isinstance(gate, str) or not isinstance(rel, str):
            raise GuardConfigError(f"baseline entry needs string gate/path: {entry!r}")
        if not isinstance(code, str) or not isinstance(count, int) or count < 1:
            raise GuardConfigError(
                f"baseline entry needs string code and positive int count: {entry!r}"
            )
        normalized.append(
            {
                "gate": gate,
                "path": rel,
                "code": code,
                "count": count,
                "reason": str(entry.get("reason") or ""),
            }
        )
    return normalized


def apply_baseline(
    findings: Sequence[Finding], entries: Sequence[Dict[str, object]]
) -> Tuple[List[Finding], List[Finding], List[Finding]]:
    """Split findings into (unbaselined, tolerated, stale_baseline)."""
    by_key: Dict[Tuple[str, str, str], Dict[str, object]] = {}
    for entry in entries:
        by_key[(str(entry["gate"]), str(entry["path"]), str(entry["code"]))] = entry
    used: Dict[Tuple[str, str, str], int] = {}
    unbaselined: List[Finding] = []
    tolerated: List[Finding] = []
    for finding in findings:
        key = (finding.gate, finding.path, finding.code)
        entry = by_key.get(key)
        if entry is None:
            unbaselined.append(finding)
            continue
        tolerated_count = int(entry["count"])
        seen = used.get(key, 0)
        if seen < tolerated_count:
            used[key] = seen + 1
            tolerated.append(finding)
        else:
            unbaselined.append(
                replace(
                    finding,
                    message=(
                        f"{finding.message} [exceeds baseline count "
                        f"{tolerated_count} for {key[0]} {key[2]}]"
                    ),
                )
            )
    stale: List[Finding] = []
    for key, entry in sorted(by_key.items()):
        if used.get(key, 0) == 0:
            stale.append(
                Finding(
                    str(entry["path"]),
                    0,
                    "BASELINE",
                    f"baseline entry disappeared: gate {entry['gate']} "
                    f"code {entry['code']!r} (baseline count {entry['count']}); "
                    "refresh infra/governance/arch_guard_baseline.json",
                    code=str(entry["code"]),
                )
            )
    return unbaselined, tolerated, stale


SCAN_GATES = ("G1", "G2", "G3", "G4", "G5")


def _gate_counts(findings: Iterable[Finding]) -> Dict[str, int]:
    counts = {gate: 0 for gate in SCAN_GATES + ("BASELINE",)}
    for finding in findings:
        counts[finding.gate] = counts.get(finding.gate, 0) + 1
    return counts


# --- report / scan ---------------------------------------------------------


def run_guard(
    root: Path,
    *,
    strict_frozen: bool = False,
    ruff: bool = True,
    use_baseline: bool = True,
    baseline_path: Optional[Path] = None,
    diff_paths: Optional[Sequence[str]] = None,
    commit_message: Optional[str] = None,
    ruff_runner: Optional[Callable[[Path], List[Finding]]] = None,
) -> Dict[str, object]:
    """Run all five gates and return the machine-readable report dict."""
    root = Path(root)
    warnings: List[str] = []
    raw: List[Finding] = []
    text_cache: TextCache = {}

    raw.extend(scan_g1(root, _cache=text_cache))
    raw.extend(scan_g2(root))
    raw.extend(scan_g3(root, _cache=text_cache))

    raw_g4 = scan_g4(
        root,
        strict_frozen=strict_frozen,
        diff_paths=diff_paths,
        commit_message=commit_message,
    )
    raw.extend(raw_g4)

    ruff_available = False
    ruff_findings = 0
    if ruff:
        try:
            gate5 = scan_g5(root, ruff_runner=ruff_runner)
            ruff_available = True
            ruff_findings = len(gate5)
            raw.extend(gate5)
        except RuffUnavailable as exc:
            warnings.append(f"G5 skipped: {exc}")
    else:
        warnings.append("G5 skipped: ruff disabled by caller")

    entries = load_baseline(baseline_path) if use_baseline else []
    unbaselined, tolerated, stale = apply_baseline(raw, entries)

    changed_frozen = frozen_paths(root, diff_paths)
    message = _commit_message(root, commit_message) or ""

    report: Dict[str, object] = {
        "root": str(root),
        "strict_frozen": strict_frozen,
        "baseline": {
            "enabled": bool(use_baseline),
            "path": str(Path(baseline_path) if baseline_path is not None else BASELINE_PATH),
            "entries": len(entries),
            "tolerated": len(tolerated),
            "stale": len(stale),
        },
        "ruff": {
            "available": ruff_available,
            "findings": ruff_findings,
        },
        "frozen": {
            "enforced": strict_frozen,
            "waiver": UNLOCK_MARKER in message,
            "changed": changed_frozen,
        },
        "counts": {
            "findings": _gate_counts(raw),
            "tolerated": _gate_counts(tolerated),
            "unbaselined": _gate_counts(unbaselined),
        },
        "violations": [finding.to_dict() for finding in unbaselined],
        "stale_baseline": [finding.to_dict() for finding in stale],
        "warnings": warnings,
        "clean": not unbaselined and not stale,
    }
    return report


def scan(
    root: Path,
    *,
    strict_frozen: bool = False,
    ruff: bool = True,
) -> List[Finding]:
    """Return the unbaselined hard findings for *root* (no stale entries)."""
    report = run_guard(root, strict_frozen=strict_frozen, ruff=ruff)
    violations = [
        Finding(
            str(item["path"]),
            int(item["line"]),
            str(item["gate"]),
            str(item["message"]),
            str(item.get("code") or ""),
        )
        for item in report["violations"]  # type: ignore[index]
    ]
    return violations


# --- CLI -------------------------------------------------------------------


def _format_finding(finding: Dict[str, object]) -> str:
    path = finding["path"]
    line = finding["line"]
    gate = finding["gate"]
    message = finding["message"]
    location = f"{path}:{line}" if line else str(path)
    return f"{gate} {location}: {message}"


def _print_human(report: Dict[str, object]) -> None:
    counts = report["counts"]
    findings_by_gate = counts["findings"]  # type: ignore[index]
    tolerated_by_gate = counts["tolerated"]  # type: ignore[index]
    unbaselined_by_gate = counts["unbaselined"]  # type: ignore[index]
    print(f"arch_guard: root={report['root']}")
    for gate in SCAN_GATES:
        if gate == "G4":
            frozen = report["frozen"]  # type: ignore[index]
            if frozen["enforced"]:  # type: ignore[index]
                print(
                    f"  G4 frozen-zone (strict): {findings_by_gate.get('G4', 0)} "
                    "finding(s)"
                )
            else:
                print(
                    "  G4 frozen-zone: advisory (use --strict-frozen to enforce); "
                    f"{len(frozen['changed'])} changed frozen path(s)"  # type: ignore[index]
                )
            continue
        if gate == "G5":
            ruff_info = report["ruff"]  # type: ignore[index]
            if not ruff_info["available"]:  # type: ignore[index]
                print("  G5 active-zone lint: SKIPPED (ruff unavailable)")
                continue
            print(
                "  G5 active-zone lint: "
                f"{findings_by_gate.get('G5', 0)} finding(s), "
                f"{tolerated_by_gate.get('G5', 0)} baselined, "
                f"{unbaselined_by_gate.get('G5', 0)} unbaselined"
            )
            continue
        print(
            f"  {gate}: {findings_by_gate.get(gate, 0)} finding(s) "
            f"({tolerated_by_gate.get(gate, 0)} baselined, "
            f"{unbaselined_by_gate.get(gate, 0)} unbaselined)"
        )
    baseline = report["baseline"]  # type: ignore[index]
    print(
        "  baseline: "
        f"{baseline['entries']} entry(ies), "  # type: ignore[index]
        f"{baseline['tolerated']} tolerated, "  # type: ignore[index]
        f"{baseline['stale']} stale"  # type: ignore[index]
    )
    for warning in report["warnings"]:  # type: ignore[index]
        print(f"  WARN {warning}")
    for finding in report["stale_baseline"]:  # type: ignore[index]
        print(f"  STALE {_format_finding(finding)}")
    for finding in report["violations"]:  # type: ignore[index]
        print(f"  VIOLATION {_format_finding(finding)}")
    status = "CLEAN" if report["clean"] else "VIOLATIONS"
    total = len(report["violations"]) + len(report["stale_baseline"])  # type: ignore[arg-type]
    print(f"arch_guard: {total} problem(s); {status}")


def _build_baseline(
    raw_findings: Sequence[Finding],
    existing: Sequence[Dict[str, object]],
) -> List[Dict[str, object]]:
    existing_reasons = {
        (str(entry["gate"]), str(entry["path"]), str(entry["code"])): str(
            entry.get("reason") or ""
        )
        for entry in existing
    }
    grouped: Dict[Tuple[str, str, str], int] = {}
    for finding in raw_findings:
        key = (finding.gate, finding.path, finding.code)
        grouped[key] = grouped.get(key, 0) + 1
    entries: List[Dict[str, object]] = []
    for (gate, rel, code), count in sorted(grouped.items()):
        entries.append(
            {
                "gate": gate,
                "path": rel,
                "code": code,
                "count": count,
                "reason": existing_reasons.get((gate, rel, code), "auto-tolerated; review"),
            }
        )
    return entries


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Architecture boundary guard (ADR-0036): five hard gates "
        "for the five-zone layout."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("."),
        help="repository root to scan (default: current directory)",
    )
    parser.add_argument(
        "--json", action="store_true", help="print the machine-readable report as JSON"
    )
    parser.add_argument(
        "--strict-frozen",
        action="store_true",
        help="enforce G4 (frozen-zone read-only); advisory without this flag",
    )
    parser.add_argument(
        "--no-ruff", action="store_true", help="skip G5 (ruff) for a fast structural run"
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=None,
        help="baseline JSON path (default: arch_guard_baseline.json next to this file)",
    )
    parser.add_argument(
        "--update-baseline",
        action="store_true",
        help="rewrite the tolerance baseline from current G3/G5 findings "
        "(refuses if G1/G2 hard gates fail)",
    )
    args = parser.parse_args(argv)

    root = args.root.resolve()
    if args.update_baseline:
        report = run_guard(
            root,
            strict_frozen=args.strict_frozen,
            ruff=not args.no_ruff,
            use_baseline=False,
            baseline_path=args.baseline,
        )
        if not report["ruff"]["available"]:  # type: ignore[index]
            print(
                "arch_guard: refusing to rewrite baseline without a working "
                "ruff (G5 entries would be silently dropped)",
                file=sys.stderr,
            )
            return 2
        hard = [
            finding
            for finding in report["violations"]  # type: ignore[index]
            if finding["gate"] in ("G1", "G2", "G4")
        ]
        if hard:
            for finding in hard:
                print(f"VIOLATION {_format_finding(finding)}", file=sys.stderr)
            print(
                "arch_guard: refusing to rewrite baseline while G1/G2/G4 hard "
                "gates fail",
                file=sys.stderr,
            )
            return 2
        tolerated_entries = load_baseline(args.baseline)
        raw_findings = [
            Finding(
                str(item["path"]),
                int(item["line"]),
                str(item["gate"]),
                str(item["message"]),
                str(item.get("code") or ""),
            )
            for item in report["violations"]  # type: ignore[index]
            if item["gate"] in ("G3", "G5")
        ]
        entries = _build_baseline(raw_findings, tolerated_entries)
        payload = {
            "schema": BASELINE_SCHEMA,
            "note": "Documented tolerances for arch_guard G3/G5 (ADR-0036). "
            "Keyed by (gate, path, code). Remove entries as debt is paid down; "
            "the guard reports stale entries that must be pruned.",
            "entries": entries,
        }
        path = args.baseline or BASELINE_PATH
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"arch_guard: wrote {len(entries)} baseline entrie(s) to {path}")
        return 0

    try:
        report = run_guard(
            root,
            strict_frozen=args.strict_frozen,
            ruff=not args.no_ruff,
            baseline_path=args.baseline,
        )
    except GuardConfigError as exc:
        print(f"CONFIG ERROR {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    else:
        _print_human(report)
    return 0 if report["clean"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
