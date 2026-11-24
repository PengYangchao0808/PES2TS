#!/usr/bin/env python3
"""Architecture guard for the post-cleanup three-plate layout.

Single source of truth for the current repository law: ``AGENT.md`` §5.1 plus
the three-plate cleanup (2026-11-24).  The historic five-zone ADR-0036 is
superseded: the V4 / V5.1-V5.4 history (``archive/``, ``tests/legacy/``,
``pes2ts_core/pesn/``) no longer lives on the active tree -- it is preserved on
branch ``frozen/v4-v54-history`` and release tags.  The active repository is:

* ``pes2ts_core/``  -- g0 / g1 / generation / integration / utils + contracts
  (protected: no torch, no engine, no data zone, no history)
* ``pes2ts_data/``  -- contracts / readers / splits / ingest / builders
* ``pes2ts_v5/``    -- the single training + evaluation engine
* ``infra/``        -- governance + WSL ops only (no business imports)
* ``config/{experiments,data,registry}``
* ``tests/{kernel,data,v5}`` -- ``tests/legacy/`` is frozen history

The guard implements six hard gates:

G1 single training loop
    Only ``pes2ts_v5/training/**`` may contain a ``.backward()`` call or an
    ``<optimizer>.step()`` call.  AST-only, so comments/docstrings never fire.

G2 no campaign-versioned code files
    An active-zone ``.py`` file whose stem matches
    ``^v(?:5[5-9]\\d*|[6-9]\\d*)_`` (``v55_*``, ``v563_*``, ``v6_*``,
    ``v7_*`` ...) is a violation.  ``pes2ts_v5/models/v5_model.py`` stays
    allowed (stem ``v5_model`` does not match).

G3 one-way imports
    AST ``Import``/``ImportFrom`` only (string literals are not imports):

    * ``pes2ts_v5/**`` and ``pes2ts_data/**`` must not import
      ``pes2ts_core.pesn`` (impossible after the cleanup) or ``archive``;
    * protected ``pes2ts_core`` code (``utils``, ``g0``, ``g1``,
      ``generation``, ``integration``, ``contracts``) must not import
      ``torch``, ``pes2ts_v5``, ``pes2ts_data``, ``archive`` or
      ``pes2ts_core.pesn``;
    * ``infra/**`` must not import business code (``pes2ts_core``,
      ``pes2ts_v5``, ``pes2ts_data``, ``archive``).  One narrow, documented
      exception (same style as the former archive->infra sink): the governance
      renderer ``infra/governance/render_gv_map.py`` may import
      ``pes2ts_core.gv_registry`` only -- the registry schema/validator is a
      spec contract, not business/runtime logic (``INFRA_G3_EXCEPTIONS``);
    * ``tests/**`` outside ``tests/legacy/**`` must not import frozen modules
      (``archive``, ``tests.legacy``, ``pes2ts_core.pesn``).

G4 frozen history must stay deleted
    Hard-fail if any *tracked* path under ``archive/``, ``tests/legacy/`` or
    ``pes2ts_core/pesn/`` reappears in the git index (``git ls-files``).  The
    only waiver is a ``HEAD`` commit message containing ``[unlock-legacy]``
    (owner-approved one-commit unlock; no long-lived reconnect).  The gate is
    honest about transient deletion windows: while the cleanup commit is still
    being prepared the paths are reported and fail; once the deleted paths are
    committed the gate is green.

G5 active-zone lint -- zero tolerance
    ``python -m ruff check pes2ts_v5 pes2ts_data infra`` (project-pinned
    ``ruff.toml``) must be clean.  The tolerance baseline is **empty**: there
    is no ruff/G3 debt left after the archive/legacy deletion and infra
    slimming.  A missing ruff interpreter degrades the gate to SKIPPED (never a
    violation), so the torch-free CI tier can run the structural gates.

G6 size / anti-fork
    * no active-zone (``pes2ts_data`` + ``pes2ts_v5`` + ``infra``) ``.py`` file
      may exceed 350 lines unless listed in the explicit ``grandfathered``
      baseline with a reason; a grandfathered file that no longer exists or no
      longer exceeds the limit is reported as stale and must be pruned;
    * anti-fork evidence: exactly one ``.backward()`` owner (enforced by G1)
      and no campaign-versioned filenames (enforced by G2); both are reported
      in ``anti_fork``;
    * the combined ``pes2ts_data + pes2ts_v5 + infra`` Python LOC must stay
      under 25,000.

Public API: ``Finding(path, line, gate, message, code="")``, per-gate
``scan_g1 .. scan_g6``, ``scan(root) -> list[Finding]`` (unbaselined hard
findings) and ``run_guard(root) -> dict`` (machine-readable report).  CLI:
``python infra/governance/arch_guard.py [--root .] [--json] [--no-ruff]
[--update-baseline]``; exit code 1 on violation, 0 when clean.

The module is stdlib-only (no torch/pandas) so it runs in the torch-free
``tests/kernel`` tier.  Subprocess use (git / ruff) is confined to this
``infra/`` governance tool on purpose.
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
    "FROZEN_PREFIXES",
    "GuardConfigError",
    "INFRA_G3_EXCEPTIONS",
    "MAX_ACTIVE_LOC",
    "MAX_FILE_LINES",
    "RuffUnavailable",
    "active_stats",
    "load_baseline",
    "main",
    "run_guard",
    "scan",
    "scan_g1",
    "scan_g2",
    "scan_g3",
    "scan_g4",
    "scan_g5",
    "scan_g6",
]

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_ROOT = SCRIPT_DIR.parents[1]
BASELINE_PATH = SCRIPT_DIR / "arch_guard_baseline.json"
BASELINE_SCHEMA = 2

#: Zones scanned for code-level gates (G1/G2/G3).  ``pes2ts_core/pesn/`` is
#: excluded while it still exists: it is frozen history awaiting deletion.
ACTIVE_ZONES = ("pes2ts_v5", "pes2ts_data", "infra", "pes2ts_core")

#: Frozen history that must not reappear on the active tree.
FROZEN_PREFIXES = ("archive/", "tests/legacy/", "pes2ts_core/pesn/")

TRAINING_PREFIX = "pes2ts_v5/training/"

CORE_PROTECTED_PREFIXES = (
    "pes2ts_core/utils/",
    "pes2ts_core/g0/",
    "pes2ts_core/g1/",
    "pes2ts_core/generation/",
    "pes2ts_core/integration/",
    "pes2ts_core/contracts",  # package ``contracts/`` or transitional ``contracts.py``
)

RUFF_ZONES = ("pes2ts_v5", "pes2ts_data", "infra")
SIZE_ZONES = ("pes2ts_data", "pes2ts_v5", "infra")

#: G6 limits (anti-fork / size discipline).
MAX_FILE_LINES = 350
MAX_ACTIVE_LOC = 25_000
MAX_GRANDFATHERED = 40  # keep the grandfather list short and explicit

#: ``v55_*`` .. ``v59_*`` / ``v563_*`` (V5.5+ sub-campaigns) and ``v6_*`` ..
#: ``v99_*``.  ``v5_model.py`` (stem ``v5_model``) does NOT match.
CAMPAIGN_VERSION_RE = re.compile(r"^v(?:5[5-9]\d*|[6-9]\d*)_")

V5_DATA_FORBIDDEN = ("pes2ts_core.pesn", "archive")
CORE_FORBIDDEN = ("torch", "pes2ts_v5", "pes2ts_data", "archive", "pes2ts_core.pesn")
INFRA_FORBIDDEN = ("pes2ts_core", "pes2ts_v5", "pes2ts_data", "archive")
TESTS_FORBIDDEN = ("archive", "tests.legacy", "pes2ts_core.pesn")

#: Narrow per-file G3 allowances for governance tooling.  NOT a blanket
#: baseline: each entry names exactly one forbidden target and one permitted
#: module.  The G-V registry is a schema/validator spec contract used by the
#: doc renderer, not business/runtime logic.
INFRA_G3_EXCEPTIONS: Dict[str, Dict[str, str]] = {
    "infra/governance/render_gv_map.py": {
        "target": "pes2ts_core",
        "allowed_module": "pes2ts_core.gv_registry",
        "reason": "G-V map renderer validates the registry schema "
        "(spec contract, not business/runtime logic)",
    },
}

#: HEAD-message waiver for G4 (one commit, owner-approved).
UNLOCK_MARKER = "[unlock-legacy]"

RUFF_CONFIG_FILES = ("ruff.toml", ".ruff.toml", "pyproject.toml")


class GuardConfigError(ValueError):
    """Baseline/config file is missing, unreadable or malformed."""


class RuffUnavailable(RuntimeError):
    """``python -m ruff`` is not installed or produced unusable output."""


@dataclass(frozen=True)
class Finding:
    """One architecture-gate finding.

    ``gate`` is ``G1``..``G6`` (``BASELINE`` for stale tolerance/grandfathered
    entries).  ``code`` is a stable rule key (ruff rule code, ``size``,
    ``loc``, ``import:<target>`` ...) used by the baseline.
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
    """Active-zone ``*.py`` files, frozen history excluded."""
    for path, rel in _iter_py(root, ACTIVE_ZONES):
        if rel.startswith(FROZEN_PREFIXES):
            continue
        yield path, rel


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


#: Per-run source cache: ``{repo-relative path: source text}``.
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
    """Yield ``(lineno, module, from_names)`` for absolute imports.

    Relative imports (``from .x import y``) stay inside their own top-level
    package and can never cross a plate boundary, so they are ignored.
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
    if "." in target:
        parent, child = target.rsplit(".", 1)
        if module == parent and child in names:
            return True
    return False


def _allowed_module(module: str, names: Sequence[str], allowed: str) -> bool:
    """True when an import is the documented allowed module (exact or parent)."""
    if module == allowed or module.startswith(allowed + "."):
        return True
    if "." in allowed:
        parent, leaf = allowed.rsplit(".", 1)
        if module == parent and leaf in names:
            return True
    return False


def _match_imports(
    rel: str,
    imports: Iterable[Tuple[int, str, Tuple[str, ...]]],
    forbidden: Iterable[str],
    scope_label: str,
    exception: Optional[Dict[str, str]] = None,
) -> List[Finding]:
    findings: List[Finding] = []
    targets = tuple(forbidden)
    for lineno, module, names in imports:
        for target in targets:
            if not _import_matches(module, names, target):
                continue
            if (
                exception is not None
                and target == exception.get("target")
                and _allowed_module(module, names, str(exception.get("allowed_module") or ""))
            ):
                continue
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
    return rel.startswith(CORE_PROTECTED_PREFIXES)


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


def _backward_owners(root: Path, cache: Optional[TextCache] = None) -> List[str]:
    """All active-zone files that own a real ``.backward()`` AST call."""
    owners: List[str] = []
    for path, rel in _iter_active_py(root):
        text = _module_text(path, rel, cache)
        if ".backward" not in text:
            continue
        tree = _parse_text(text, path)
        if tree is None:
            continue
        if any(_is_backward_call(node) for node in ast.walk(tree)):
            owners.append(rel)
    return sorted(owners)


def scan_g1(root: Path, *, _cache: Optional[TextCache] = None) -> List[Finding]:
    """G1: only ``pes2ts_v5/training/**`` may run the backward/step loop."""
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

#: Conservative textual prefilters run before AST parsing; each token set is a
#: superset of every syntactically possible forbidden import in its scope.
_V5_DATA_TOKENS = ("archive", "pesn")
_CORE_TOKENS = ("torch", "pes2ts_v5", "pes2ts_data", "archive", "pesn")
_INFRA_TOKENS = ("pes2ts_core", "pes2ts_v5", "pes2ts_data", "archive")
_TESTS_TOKENS = ("archive", "from tests import", "tests.legacy", "pesn")


def _g3_scope(rel: str) -> Optional[Tuple[Tuple[str, ...], Tuple[str, ...], str]]:
    """Return ``(forbidden targets, prefilter tokens, label)`` for a G3 scope."""
    if rel.startswith(("pes2ts_v5/", "pes2ts_data/")):
        return V5_DATA_FORBIDDEN, _V5_DATA_TOKENS, "v5/data plate"
    if _is_core_protected(rel):
        return CORE_FORBIDDEN, _CORE_TOKENS, "pes2ts_core protected"
    if rel.startswith("infra/"):
        return INFRA_FORBIDDEN, _INFRA_TOKENS, "infra plate"
    return None


def scan_g3(root: Path, *, _cache: Optional[TextCache] = None) -> List[Finding]:
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
        exception = INFRA_G3_EXCEPTIONS.get(rel) if rel.startswith("infra/") else None
        findings.extend(
            _match_imports(
                rel,
                list(_imports(tree)),
                forbidden,
                label,
                exception=exception,
            )
        )

    for path, rel in _iter_py(root, ("tests",)):
        if rel.startswith(FROZEN_PREFIXES):
            continue
        text = _module_text(path, rel, _cache)
        if not any(token in text for token in _TESTS_TOKENS):
            continue
        tree = _parse_text(text, path)
        if tree is None:
            continue
        findings.extend(
            _match_imports(rel, list(_imports(tree)), TESTS_FORBIDDEN, "tests plate")
        )

    return findings


# --- G4 frozen history must stay deleted -----------------------------------


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
            timeout=120,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout


def frozen_tracked_paths(
    root: Path, tracked_paths: Optional[Sequence[str]] = None
) -> List[str]:
    """Tracked git paths that fall under a frozen-history prefix.

    ``tracked_paths`` is injectable for tests (no git repo needed).  When git
    is unavailable (e.g. a synthetic tree), the worktree itself is scanned so
    the gate still catches a reappearing directory.
    """
    if tracked_paths is None:
        raw = _run_git(root, "ls-files", "-z")
        if raw is None:
            paths = [
                rel
                for prefix in FROZEN_PREFIXES
                for _, rel in _iter_py(root, (prefix.rstrip("/"),))
            ]
        else:
            paths = [item for item in raw.split("\0") if item]
    else:
        paths = list(tracked_paths)
    tracked: List[str] = []
    for item in paths:
        rel = item.strip().replace("\\", "/")
        if not rel:
            continue
        if rel in FROZEN_PREFIXES or rel.startswith(FROZEN_PREFIXES):
            tracked.append(rel)
    return sorted(set(tracked))


def _commit_message(root: Path, commit_message: Optional[str]) -> Optional[str]:
    if commit_message is not None:
        return commit_message
    return _run_git(root, "log", "-1", "--format=%B")


def scan_g4(
    root: Path,
    *,
    tracked_paths: Optional[Sequence[str]] = None,
    commit_message: Optional[str] = None,
) -> List[Finding]:
    """G4: hard-fail if frozen history reappears on the active tree."""
    paths = frozen_tracked_paths(root, tracked_paths)
    if not paths:
        return []
    message = _commit_message(root, commit_message) or ""
    if UNLOCK_MARKER in message:
        return []
    return [
        Finding(
            rel,
            0,
            "G4",
            "frozen history path reappeared in the active tree (archive/ and "
            "tests/legacy/ live on branch frozen/v4-v54-history; "
            "pes2ts_core/pesn/ is being deleted); delete it or carry the "
            "[unlock-legacy] waiver",
            code="frozen-reappeared",
        )
        for rel in paths
    ]


# --- G5 active-zone lint (zero tolerance) ----------------------------------


def ruff_config(root: Path) -> Optional[str]:
    """Return the repo-relative pinned ruff config path, if present."""
    for name in RUFF_CONFIG_FILES:
        if (root / name).is_file():
            return name
    return None


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
    """G5: ruff findings in the active zones (raw, unbaselined; zero tolerance)."""
    runner = ruff_runner if ruff_runner is not None else _ruff_check
    return list(runner(root))


# --- G6 size / anti-fork ---------------------------------------------------


def _count_lines(path: Path) -> int:
    return len(_read_text(path).splitlines())


def active_stats(root: Path) -> Dict[str, object]:
    """Per-zone and total Python file/LOC counts for the size-budget plates."""
    per_zone: Dict[str, Dict[str, int]] = {}
    total_files = 0
    total_loc = 0
    for zone in SIZE_ZONES:
        files = 0
        loc = 0
        for path, _ in _iter_py(root, (zone,)):
            files += 1
            loc += _count_lines(path)
        per_zone[zone] = {"files": files, "loc": loc}
        total_files += files
        total_loc += loc
    return {
        "zones": list(SIZE_ZONES),
        "per_zone": per_zone,
        "files": total_files,
        "loc": total_loc,
        "max_file_lines": MAX_FILE_LINES,
        "max_loc": MAX_ACTIVE_LOC,
    }


def scan_g6(
    root: Path,
    *,
    grandfathered_paths: Sequence[str] = (),
    max_file_lines: int = MAX_FILE_LINES,
    max_active_loc: int = MAX_ACTIVE_LOC,
) -> List[Finding]:
    """G6: oversized active-zone files (unless grandfathered) and LOC budget."""
    findings: List[Finding] = []
    grandfathered = set(grandfathered_paths)
    for zone in SIZE_ZONES:
        for path, rel in _iter_py(root, (zone,)):
            if rel in grandfathered:
                continue
            lines = _count_lines(path)
            if lines > max_file_lines:
                findings.append(
                    Finding(
                        rel,
                        lines,
                        "G6",
                        f"file has {lines} lines (> {max_file_lines}) and is not "
                        "in the grandfathered baseline; split it or register a "
                        "reasoned grandfather entry",
                        code="size",
                    )
                )
    stats = active_stats(root)
    loc = int(stats["loc"])
    if loc > max_active_loc:
        findings.append(
            Finding(
                ",".join(SIZE_ZONES),
                0,
                "G6",
                f"active python LOC {loc} exceeds the {max_active_loc} budget",
                code="loc",
            )
        )
    return findings


def stale_grandfathered(
    root: Path,
    grandfathered: Sequence[Dict[str, object]],
    *,
    max_file_lines: int = MAX_FILE_LINES,
) -> List[Finding]:
    """Grandfathered entries that no longer justify themselves (must be pruned)."""
    stale: List[Finding] = []
    for entry in grandfathered:
        rel = str(entry["path"])
        if not rel:
            continue
        path = root / rel
        if not path.is_file():
            stale.append(
                Finding(
                    rel,
                    0,
                    "BASELINE",
                    "grandfathered file no longer exists; remove the entry",
                    code="grandfathered",
                )
            )
            continue
        lines = _count_lines(path)
        if lines <= max_file_lines:
            stale.append(
                Finding(
                    rel,
                    0,
                    "BASELINE",
                    f"grandfathered file is now {lines} lines (<= {max_file_lines}); "
                    "remove the entry",
                    code="grandfathered",
                )
            )
    return stale


# --- baseline --------------------------------------------------------------


def load_baseline(baseline_path: Optional[Path] = None) -> Dict[str, object]:
    """Load and validate the baseline (``schema=2``).

    ``tolerances`` must stay empty in the three-plate reality (G5/G3 are
    zero-tolerance); ``grandfathered`` lists the explicit oversized files.
    """
    path = Path(baseline_path) if baseline_path is not None else BASELINE_PATH
    if not path.is_file():
        return {"tolerances": [], "grandfathered": [], "path": str(path)}
    try:
        raw = json.loads(_read_text(path))
    except json.JSONDecodeError as exc:
        raise GuardConfigError(f"invalid baseline JSON {path}: {exc}") from exc
    if not isinstance(raw, dict) or raw.get("schema") != BASELINE_SCHEMA:
        raise GuardConfigError(
            f"baseline {path} must be an object with schema={BASELINE_SCHEMA}"
        )
    tolerances = raw.get("tolerances", [])
    grandfathered = raw.get("grandfathered", [])
    if not isinstance(tolerances, list) or not isinstance(grandfathered, list):
        raise GuardConfigError(
            f"baseline {path}: 'tolerances' and 'grandfathered' must be lists"
        )
    normalized_tolerances: List[Dict[str, object]] = []
    for entry in tolerances:
        if not isinstance(entry, dict):
            raise GuardConfigError(f"tolerance entry is not an object: {entry!r}")
        gate = entry.get("gate")
        rel = entry.get("path")
        code = entry.get("code")
        count = entry.get("count")
        if not isinstance(gate, str) or not isinstance(rel, str):
            raise GuardConfigError(
                f"tolerance entry needs string gate/path: {entry!r}"
            )
        if not isinstance(code, str) or not isinstance(count, int) or count < 1:
            raise GuardConfigError(
                f"tolerance entry needs string code and positive int count: {entry!r}"
            )
        normalized_tolerances.append(
            {
                "gate": gate,
                "path": rel,
                "code": code,
                "count": count,
                "reason": str(entry.get("reason") or ""),
            }
        )
    normalized_grandfathered: List[Dict[str, object]] = []
    for entry in grandfathered:
        if not isinstance(entry, dict):
            raise GuardConfigError(
                f"grandfathered entry is not an object: {entry!r}"
            )
        rel = entry.get("path")
        reason = entry.get("reason")
        if not isinstance(rel, str) or not rel:
            raise GuardConfigError("grandfathered entry needs a non-empty path")
        if not isinstance(reason, str) or not reason.strip():
            raise GuardConfigError(
                f"grandfathered entry {rel!r} needs a non-empty reason"
            )
        normalized_grandfathered.append(
            {
                "path": rel,
                "lines": int(entry.get("lines") or 0),
                "reason": reason.strip(),
            }
        )
    return {
        "tolerances": normalized_tolerances,
        "grandfathered": normalized_grandfathered,
        "path": str(path),
    }


def apply_baseline(
    findings: Sequence[Finding], entries: Sequence[Dict[str, object]]
) -> Tuple[List[Finding], List[Finding], List[Finding]]:
    """Split findings into (unbaselined, tolerated, stale tolerance entries)."""
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
    for key in sorted(by_key):
        if used.get(key, 0) == 0:
            entry = by_key[key]
            stale.append(
                Finding(
                    str(entry["path"]),
                    0,
                    "BASELINE",
                    f"tolerance entry disappeared: gate {entry['gate']} "
                    f"code {entry['code']!r}; the tolerance baseline must stay "
                    "empty",
                    code=str(entry["code"]),
                )
            )
    return unbaselined, tolerated, stale


SCAN_GATES = ("G1", "G2", "G3", "G4", "G5", "G6")


def _gate_counts(findings: Iterable[Finding]) -> Dict[str, int]:
    counts = {gate: 0 for gate in SCAN_GATES + ("BASELINE",)}
    for finding in findings:
        counts[finding.gate] = counts.get(finding.gate, 0) + 1
    return counts


# --- report / scan ---------------------------------------------------------


def run_guard(
    root: Path,
    *,
    ruff: bool = True,
    use_baseline: bool = True,
    baseline_path: Optional[Path] = None,
    tracked_paths: Optional[Sequence[str]] = None,
    commit_message: Optional[str] = None,
    ruff_runner: Optional[Callable[[Path], List[Finding]]] = None,
    max_file_lines: int = MAX_FILE_LINES,
    max_active_loc: int = MAX_ACTIVE_LOC,
) -> Dict[str, object]:
    """Run all six gates and return the machine-readable report dict."""
    root = Path(root)
    warnings: List[str] = []
    raw: List[Finding] = []
    text_cache: TextCache = {}

    raw.extend(scan_g1(root, _cache=text_cache))
    raw.extend(scan_g2(root))
    raw.extend(scan_g3(root, _cache=text_cache))
    raw.extend(
        scan_g4(
            root,
            tracked_paths=tracked_paths,
            commit_message=commit_message,
        )
    )

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

    baseline = (
        load_baseline(baseline_path)
        if use_baseline
        else {"tolerances": [], "grandfathered": [], "path": "disabled"}
    )
    tolerances = baseline["tolerances"]
    grandfathered = baseline["grandfathered"]
    grandfathered_paths = [str(entry["path"]) for entry in grandfathered]
    raw.extend(
        scan_g6(
            root,
            grandfathered_paths=grandfathered_paths,
            max_file_lines=max_file_lines,
            max_active_loc=max_active_loc,
        )
    )

    unbaselined, tolerated, stale_tol = apply_baseline(raw, tolerances)  # type: ignore[arg-type]
    stale_grand = (
        stale_grandfathered(
            root, grandfathered, max_file_lines=max_file_lines  # type: ignore[arg-type]
        )
        if use_baseline
        else []
    )
    stale = list(stale_tol) + list(stale_grand)

    frozen_paths = frozen_tracked_paths(root, tracked_paths)
    message = _commit_message(root, commit_message) or ""
    stats = active_stats(root)
    anti_fork = {
        "backward_owners": _backward_owners(root, text_cache),
        "campaign_version_files": [f.path for f in scan_g2(root)],
    }

    report: Dict[str, object] = {
        "root": str(root),
        "gates": {
            "findings": _gate_counts(raw),
            "tolerated": _gate_counts(tolerated),
            "unbaselined": _gate_counts(unbaselined),
        },
        "violations": [finding.to_dict() for finding in unbaselined],
        "stale_baseline": [finding.to_dict() for finding in stale],
        "baseline": {
            "path": baseline["path"],
            "tolerances": len(tolerances),
            "grandfathered": len(grandfathered),
            "tolerated": len(tolerated),
            "stale": len(stale),
        },
        "frozen": {
            "prefixes": list(FROZEN_PREFIXES),
            "tracked": frozen_paths,
            "waiver": UNLOCK_MARKER in message,
            "clean": not frozen_paths,
        },
        "ruff": {
            "available": ruff_available,
            "findings": ruff_findings,
            "config": ruff_config(root),
        },
        "active": stats,
        "anti_fork": anti_fork,
        "g3_exceptions": {
            rel: data.get("reason", "") for rel, data in INFRA_G3_EXCEPTIONS.items()
        },
        "warnings": warnings,
        "clean": not unbaselined and not stale,
    }
    return report


def scan(
    root: Path,
    *,
    ruff: bool = True,
) -> List[Finding]:
    """Return the unbaselined hard findings for *root* (no stale entries)."""
    report = run_guard(root, ruff=ruff)
    return [
        Finding(
            str(item["path"]),
            int(item["line"]),
            str(item["gate"]),
            str(item["message"]),
            str(item.get("code") or ""),
        )
        for item in report["violations"]  # type: ignore[index]
    ]


# --- CLI -------------------------------------------------------------------


def _format_finding(finding: Dict[str, object]) -> str:
    path = finding["path"]
    line = finding["line"]
    gate = finding["gate"]
    message = finding["message"]
    location = f"{path}:{line}" if line else str(path)
    return f"{gate} {location}: {message}"


def _print_human(report: Dict[str, object]) -> None:
    gates = report["gates"]
    findings_by_gate = gates["findings"]
    tolerated_by_gate = gates["tolerated"]
    unbaselined_by_gate = gates["unbaselined"]
    print(f"arch_guard: root={report['root']}")
    for gate in SCAN_GATES:
        print(
            f"  {gate}: {findings_by_gate.get(gate, 0)} finding(s) "
            f"({tolerated_by_gate.get(gate, 0)} tolerated, "
            f"{unbaselined_by_gate.get(gate, 0)} unbaselined)"
        )
    frozen = report["frozen"]
    print(
        "  frozen-history: "
        f"{len(frozen['tracked'])} tracked frozen path(s) "
        f"(waiver={frozen['waiver']})"
    )
    ruff_info = report["ruff"]
    if ruff_info["available"]:
        print(
            f"  ruff: {ruff_info['findings']} finding(s); "
            f"config={ruff_info['config']}"
        )
    else:
        print("  ruff: SKIPPED (unavailable)")
    active = report["active"]
    print(
        "  active budget: "
        f"{active['files']} files / {active['loc']} LOC "
        f"(limit {active['max_loc']}); "
        f"max file {active['max_file_lines']} lines"
    )
    baseline = report["baseline"]
    print(
        f"  baseline: tolerances={baseline['tolerances']}, "
        f"grandfathered={baseline['grandfathered']}, stale={baseline['stale']}"
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


def _build_grandfathered(
    oversized: Sequence[Finding],
    existing: Sequence[Dict[str, object]],
) -> List[Dict[str, object]]:
    reasons = {str(entry["path"]): str(entry["reason"]) for entry in existing}
    entries: List[Dict[str, object]] = []
    for finding in sorted(oversized, key=lambda item: item.path):
        entries.append(
            {
                "path": finding.path,
                "lines": finding.line,
                "reason": reasons.get(
                    finding.path,
                    "legacy oversized module shipped by the three-plate cleanup; "
                    "split or archive it in a follow-up",
                ),
            }
        )
    return entries


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Architecture guard for the three-plate layout: six hard "
        "gates (single trainer, no campaign code, one-way imports, frozen "
        "history, zero ruff, size/anti-fork)."
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
        "--no-ruff", action="store_true", help="skip G5 (ruff) for a fast structural run"
    )
    parser.add_argument(
        "--strict-frozen",
        action="store_true",
        help="compatibility alias: G4 (frozen-history reappearance) is always "
        "enforced in the three-plate guard; accepted so the frozen-check CI "
        "step can pass it explicitly",
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
        help="rewrite the grandfathered-size list from current G6 findings "
        "(tolerances stay empty; refuses if any hard gate fails)",
    )
    args = parser.parse_args(argv)

    root = args.root.resolve()

    try:
        if args.update_baseline:
            report = run_guard(
                root,
                ruff=not args.no_ruff,
                use_baseline=False,
                baseline_path=args.baseline,
            )
            if not report["ruff"]["available"]:  # type: ignore[index]
                print(
                    "arch_guard: refusing to rewrite baseline without a working "
                    "ruff (G5 evidence would be missing)",
                    file=sys.stderr,
                )
                return 2
            hard = [
                finding
                for finding in report["violations"]  # type: ignore[index]
                if finding["gate"] in ("G1", "G2", "G3", "G4", "G5")
                or (finding["gate"] == "G6" and finding["code"] != "size")
            ]
            if hard:
                for finding in hard:
                    print(f"VIOLATION {_format_finding(finding)}", file=sys.stderr)
                print(
                    "arch_guard: refusing to rewrite baseline while hard gates fail",
                    file=sys.stderr,
                )
                return 2
            oversized = [
                Finding(
                    str(item["path"]),
                    int(item["line"]),
                    str(item["gate"]),
                    str(item["message"]),
                    str(item.get("code") or ""),
                )
                for item in report["violations"]  # type: ignore[index]
                if item["gate"] == "G6" and item["code"] == "size"
            ]
            existing = load_baseline(args.baseline)["grandfathered"]  # type: ignore[index]
            entries = _build_grandfathered(oversized, existing)  # type: ignore[arg-type]
            if len(entries) > MAX_GRANDFATHERED:
                print(
                    f"arch_guard: refusing to grandfather {len(entries)} files "
                    f"(> {MAX_GRANDFATHERED}); split them instead",
                    file=sys.stderr,
                )
                return 2
            payload = {
                "schema": BASELINE_SCHEMA,
                "note": (
                    "Three-plate architecture baseline (post-cleanup). "
                    "'tolerances' MUST stay empty: G3/G5 are zero-tolerance after "
                    "archive/legacy deletion and infra slimming. 'grandfathered' "
                    "lists explicit oversized active-zone modules with a reason; "
                    "the guard reports stale entries (deleted or shrunken files) "
                    "that must be pruned, and refuses any new file above 350 lines."
                ),
                "tolerances": [],
                "grandfathered": entries,
            }
            path = args.baseline or BASELINE_PATH
            path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            print(
                f"arch_guard: wrote baseline with 0 tolerance(s) and "
                f"{len(entries)} grandfathered file(s) to {path}"
            )
            return 0

        report = run_guard(
            root,
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
