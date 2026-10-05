"""Static guard against unauthorized ground-truth access.

:func:`assert_no_truth_access` AST-parses every ``.py`` module under the
package root (plus the project ``bin/`` scripts) and fails when a module
outside the explicit allowlist references a quarantined path, imports the
truth reader, names the truth-bearing HDF5 artifacts, or uses
dynamic-execution primitives.  The allowlist is intentionally tiny:

* ``pes2ts_core/g0/truth_quarantine.py`` — extracts and relocates the truth;
* ``pes2ts_core/g0/truth/truth_reader.py`` — the only audited accessors;
* ``pes2ts_core/utils/truth_guard.py`` — this scanner;
* ``pes2ts_core/g1/p1_truth.py`` — the single G1 P1 annotation module that
  reads the quarantined TS/IRC truth through the audited accessors (the
  plan's scoped exception); every other ``g1``/``g2`` module is checked
  normally;
* ``pes2ts_core/cli.py`` — only for the ``truth-index`` handler, the one
  deliberate human-facing access point.  Every other part of ``cli.py`` is
  checked normally, and the handler itself must not use subprocess/importlib.

A second, narrower list exempts modules from *only* the
``DYNAMIC_EXEC_RISK`` findings.  It is EMPTY since ADR-0002 X2'-C deleted the
local xTB runner (``pes2ts_core/generation/execution/xtb_path/runner.py``),
the one module that needed it: xTB PATH execution now goes through the ACP
CLI adapters on ``SUBPROCESS_IMPORT_ALLOWLIST`` below, and no shipped module
currently requires a dynamic-execution exemption.

A third, narrowest list permits the reviewed ACP CLI adapters
(``pes2ts_core/integration/acp/cli_backend.py``,
``pes2ts_core/integration/acp/stage_cli.py``, and the XtbPathSearch transport
``pes2ts_core/integration/acp/xtb_path_transport.py``) to import ``subprocess``
for the sanctioned ACP process boundary; every other dynamic-execution
primitive (``importlib``, ``eval``, ``exec``) and every truth check still
apply to them.

Importing ``pes2ts_core.g0.truth_quarantine`` from elsewhere stays allowed: it
returns relocation metadata, never trajectory data.  Importing the
``pes2ts_core.g0.truth`` package or ``truth_reader`` is flagged.
"""

from __future__ import annotations

import ast
import logging
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

logger = logging.getLogger(__name__)

#: A string literal references a quarantined path.
TRUTH_PATH_REF: Final[str] = "TRUTH_PATH_REF"
#: An import references the truth package or the truth reader.
TRUTH_IMPORT: Final[str] = "TRUTH_IMPORT"
#: A string literal names a truth-bearing source file.
TRUTH_FILE_REF: Final[str] = "TRUTH_FILE_REF"
#: subprocess/importlib/eval/exec/__import__ — flagged for manual review.
DYNAMIC_EXEC_RISK: Final[str] = "DYNAMIC_EXEC_RISK"
#: A scanned file could not be parsed; treat as hostile until reviewed.
UNPARSEABLE_MODULE: Final[str] = "UNPARSEABLE_MODULE"

#: Modules whose truth references need no further review (project-relative).
DEFAULT_ALLOWLIST: Final[tuple[str, ...]] = (
    "pes2ts_core/g0/truth_quarantine.py",
    "pes2ts_core/g0/truth/truth_reader.py",
    "pes2ts_core/utils/truth_guard.py",
    # The single G1 annotation module sanctioned by the P1 plan to read the
    # quarantined TS/IPC truth through the audited accessors; every other
    # g1/g2 module stays fully guarded.
    "pes2ts_core/g1/p1_truth.py",
)

#: Modules exempt ONLY from ``DYNAMIC_EXEC_RISK`` findings (project-relative).
#: Empty since ADR-0002 X2'-C deleted the local xTB runner — the one module
#: that spawned the external GFN2-xTB binary via ``subprocess``.  ACP
#: execution goes through the ``SUBPROCESS_IMPORT_ALLOWLIST`` adapters below;
#: every other check (quarantined-path strings, truth-file names, truth
#: imports) still applies to every module in full.
DYNAMIC_EXEC_ALLOWLIST: Final[tuple[str, ...]] = ()

#: Modules permitted to import ``subprocess`` only (project-relative). The
#: reviewed ACP CLI adapters launch the ACP engine as an argv-only child
#: process; every other dynamic-execution finding (``importlib``, ``eval``,
#: ``exec``, ``__import__``) and every truth check still applies to them.
SUBPROCESS_IMPORT_ALLOWLIST: Final[tuple[str, ...]] = (
    "pes2ts_core/integration/acp/cli_backend.py",
    "pes2ts_core/integration/acp/stage_cli.py",
    "pes2ts_core/integration/acp/xtb_path_transport.py",
    "pes2ts_core/integration/acp/orca_gradient_transport.py",
    # G2-AB2 WP-3: origin free-optimization adapter — same argv-only ACP CLI
    # child-process discipline (claim/receipt/log/timeout) as the adapters
    # above; truth checks still apply to it in full.
    "pes2ts_core/integration/acp/origin_backend.py",
)

TRUTH_PATH_MARKERS: Final[tuple[str, ...]] = ("ground_truth", "truth_sources")
TRUTH_FILE_MARKERS: Final[tuple[str, ...]] = ("B3LYPD3_TZVP.h5", "B3LYPD3_TZVP_IRC.h5")
TRUTH_PACKAGE: Final[str] = "pes2ts_core.g0.truth"
TRUTH_READER_NAME: Final[str] = "truth_reader"
DYNAMIC_MODULES: Final[frozenset[str]] = frozenset({"subprocess", "importlib"})
DYNAMIC_CALL_NAMES: Final[frozenset[str]] = frozenset({"eval", "exec", "__import__"})
CLI_FILENAME: Final[str] = "cli.py"
CLI_TRUTH_HANDLER_MARKER: Final[str] = "truth_index"
_PACKAGE_PARENT_FALLBACK: Final[Path] = Path(__file__).resolve().parents[2]


@dataclass(frozen=True, slots=True)
class Finding:
    """One unauthorized reference found by the guard."""

    file: str
    line: int
    kind: str
    detail: str

    def __str__(self) -> str:
        return f"{self.file}:{self.line} {self.kind} {self.detail}"


class TruthAccessViolation(AssertionError):
    """Raised by :func:`assert_no_truth_access` when findings exist."""

    def __init__(self, findings: Sequence[Finding]) -> None:
        self.findings: tuple[Finding, ...] = tuple(findings)
        header = f"Ground-truth guard found {len(self.findings)} unauthorized reference(s):"
        super().__init__("\n".join([header, *(str(finding) for finding in self.findings)]))


def _resolve_scan_root(package_root: str | Path) -> Path:
    """Return the absolute package root, falling back to the project root."""
    candidate = Path(package_root)
    if candidate.is_absolute():
        return candidate
    if candidate.is_dir():
        return candidate.resolve()
    fallback = _PACKAGE_PARENT_FALLBACK / candidate
    if fallback.is_dir():
        return fallback
    raise FileNotFoundError(f"package_root does not exist: {package_root}")


def _iter_modules(root: Path) -> Iterator[Path]:
    """Yield every ``.py`` module under *root* (``__pycache__`` excluded)."""
    for path in sorted(root.rglob("*.py")):
        if "__pycache__" not in path.parts:
            yield path


def _iter_scripts(root: Path) -> Iterator[Path]:
    """Yield extensionless executable scripts directly under *root*."""
    for path in sorted(root.iterdir()):
        if path.is_file() and not path.suffix:
            yield path


def _relative_module_path(path: Path, root: Path) -> str:
    """Return *path*'s posix path relative to the scan root's parent."""
    base = root.parent if root.parent != root else root
    try:
        return path.resolve().relative_to(base.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _string_findings(node: ast.Constant, file: str) -> list[Finding]:
    """Flag quarantined-path and truth-file literals."""
    text = node.value
    if not isinstance(text, str):
        return []
    findings: list[Finding] = []
    for marker in TRUTH_PATH_MARKERS:
        if marker in text:
            findings.append(
                Finding(file, node.lineno, TRUTH_PATH_REF, f"string literal contains {marker!r}")
            )
    for marker in TRUTH_FILE_MARKERS:
        if marker in text:
            findings.append(
                Finding(file, node.lineno, TRUTH_FILE_REF, f"string literal contains {marker!r}")
            )
    return findings


def _imported_names(node: ast.Import | ast.ImportFrom) -> list[str]:
    """Return every module-ish name an import statement references."""
    names = [alias.name for alias in node.names]
    if isinstance(node, ast.ImportFrom) and node.module:
        names.append(node.module)
        names.extend(f"{node.module}.{alias.name}" for alias in node.names)
    return names


def _dynamic_import_findings(
    node: ast.Import | ast.ImportFrom, file: str
) -> list[Finding]:
    """Flag subprocess/importlib imports (used inside narrow exceptions)."""
    findings: list[Finding] = []
    for name in _imported_names(node):
        if name.split(".")[0] in DYNAMIC_MODULES:
            findings.append(
                Finding(file, node.lineno, DYNAMIC_EXEC_RISK, f"imports {name} (manual review)")
            )
    return findings


def _import_findings(node: ast.Import | ast.ImportFrom, file: str) -> list[Finding]:
    """Flag truth-reader imports and subprocess/importlib imports."""
    findings: list[Finding] = []
    for name in _imported_names(node):
        if name == TRUTH_PACKAGE or name.startswith(f"{TRUTH_PACKAGE}."):
            findings.append(
                Finding(file, node.lineno, TRUTH_IMPORT, f"imports the truth package: {name}")
            )
        elif TRUTH_READER_NAME in name.split("."):
            findings.append(
                Finding(file, node.lineno, TRUTH_IMPORT, f"imports the truth reader: {name}")
            )
    return findings + _dynamic_import_findings(node, file)


def _dynamic_findings(node: ast.AST, file: str) -> list[Finding]:
    """Flag eval/exec/__import__ calls and ``.__import__`` attribute access."""
    if isinstance(node, ast.Call):
        func = node.func
        if isinstance(func, ast.Name) and func.id in DYNAMIC_CALL_NAMES:
            return [Finding(file, node.lineno, DYNAMIC_EXEC_RISK, f"calls {func.id}()")]
        if isinstance(func, ast.Attribute) and func.attr in DYNAMIC_CALL_NAMES:
            return [Finding(file, node.lineno, DYNAMIC_EXEC_RISK, f"calls .{func.attr}()")]
    if isinstance(node, ast.Attribute) and node.attr == "__import__":
        return [Finding(file, node.lineno, DYNAMIC_EXEC_RISK, "accesses .__import__")]
    return []


def _check_node(node: ast.AST, file: str) -> list[Finding]:
    """Apply every rule to a single node (no recursion)."""
    if isinstance(node, ast.Constant):
        return _string_findings(node, file)
    if isinstance(node, (ast.Import, ast.ImportFrom)):
        return _import_findings(node, file)
    return _dynamic_findings(node, file)


def _scan_cli_module(module: ast.Module, file: str) -> list[Finding]:
    """Scan ``cli.py`` while exempting the ``truth-index`` handler subtree."""
    findings: list[Finding] = []
    exempt: list[ast.AST] = []

    def visit(node: ast.AST) -> None:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and (
            CLI_TRUTH_HANDLER_MARKER in node.name
        ):
            exempt.append(node)
            return
        findings.extend(_check_node(node, file))
        for child in ast.iter_child_nodes(node):
            visit(child)

    visit(module)
    for handler in exempt:
        for node in ast.walk(handler):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                findings.extend(_dynamic_import_findings(node, file))
            else:
                findings.extend(_dynamic_findings(node, file))
    return findings


def _scan_file(
    path: Path,
    root: Path,
    allowlist: Sequence[str],
    dynamic_exec_allowlist: Sequence[str],
) -> list[Finding]:
    """Scan one module, honoring every allowlist and the ``cli.py`` rule."""
    file = str(path)
    try:
        source = path.read_text(encoding="utf-8")
        module = ast.parse(source, filename=file)
    except (OSError, UnicodeDecodeError) as exc:
        return [Finding(file, 0, UNPARSEABLE_MODULE, f"cannot read module: {exc}")]
    except SyntaxError as exc:
        return [
            Finding(file, exc.lineno or 0, UNPARSEABLE_MODULE, f"syntax error: {exc.msg}")
        ]
    if path.name == CLI_FILENAME:
        return _scan_cli_module(module, file)
    relative = _relative_module_path(path, root)
    if relative in allowlist:
        return []
    findings: list[Finding] = []
    for node in ast.walk(module):
        findings.extend(_check_node(node, file))
    if relative in dynamic_exec_allowlist:
        findings = [finding for finding in findings if finding.kind != DYNAMIC_EXEC_RISK]
    if relative in SUBPROCESS_IMPORT_ALLOWLIST:
        findings = [
            finding
            for finding in findings
            if not (
                finding.kind == DYNAMIC_EXEC_RISK
                and finding.detail == "imports subprocess (manual review)"
            )
        ]
    return findings


def scan_truth_access(
    package_root: str | Path = "pes2ts_core",
    allowlist: Sequence[str] = DEFAULT_ALLOWLIST,
    dynamic_exec_allowlist: Sequence[str] = DYNAMIC_EXEC_ALLOWLIST,
) -> list[Finding]:
    """Return every finding across the package and the project ``bin/`` tree.

    Modules in *allowlist* are skipped entirely; modules in
    *dynamic_exec_allowlist* are scanned in full except that their
    ``DYNAMIC_EXEC_RISK`` findings are suppressed; modules in
    ``SUBPROCESS_IMPORT_ALLOWLIST`` are scanned in full except that their
    subprocess-import finding is suppressed.
    """
    root = _resolve_scan_root(package_root)
    findings: list[Finding] = []
    for module in _iter_modules(root):
        findings.extend(_scan_file(module, root, allowlist, dynamic_exec_allowlist))
    bin_dir = root.parent / "bin"
    if bin_dir.is_dir():
        for module in _iter_modules(bin_dir):
            findings.extend(
                _scan_file(module, root, allowlist, dynamic_exec_allowlist)
            )
        for script in _iter_scripts(bin_dir):
            findings.extend(
                _scan_file(script, root, allowlist, dynamic_exec_allowlist)
            )
    return sorted(set(findings), key=lambda finding: (finding.file, finding.line, finding.kind))


def assert_no_truth_access(
    package_root: str | Path = "pes2ts_core",
    allowlist: Sequence[str] = DEFAULT_ALLOWLIST,
    dynamic_exec_allowlist: Sequence[str] = DYNAMIC_EXEC_ALLOWLIST,
) -> list[Finding]:
    """Raise :class:`TruthAccessViolation` on any finding, else return ``[]``.

    The findings are attached to the raised exception as
    :attr:`TruthAccessViolation.findings` so callers can assert on kinds.
    """
    findings: Iterable[Finding] = scan_truth_access(
        package_root=package_root,
        allowlist=allowlist,
        dynamic_exec_allowlist=dynamic_exec_allowlist,
    )
    collected = list(findings)
    if collected:
        raise TruthAccessViolation(collected)
    return collected


__all__ = [
    "CLI_FILENAME",
    "CLI_TRUTH_HANDLER_MARKER",
    "DEFAULT_ALLOWLIST",
    "DYNAMIC_EXEC_ALLOWLIST",
    "SUBPROCESS_IMPORT_ALLOWLIST",
    "DYNAMIC_EXEC_RISK",
    "TRUTH_FILE_REF",
    "TRUTH_IMPORT",
    "TRUTH_PATH_REF",
    "UNPARSEABLE_MODULE",
    "Finding",
    "TruthAccessViolation",
    "assert_no_truth_access",
    "scan_truth_access",
]
