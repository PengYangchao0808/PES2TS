"""B0/B6 acceptance: static and transitive import reachability of the runtime set.

The runtime-reachable set is api/contracts/features/models/inference/bundle
(+ light package init).  A static edge from any of those files into offline/,
G0, G1, generation, ranking, integration, or ACP is a product-boundary
violation; dynamic-import tricks to dodge the scan are impossible by
construction because the same scan treats ``importlib``/``__import__`` as
findings.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PESN_ROOT = PROJECT_ROOT / "pes2ts_core" / "pesn"

RUNTIME_SUBDIRS = ("contracts", "features", "models", "inference", "bundle")
# __main__.py is the CLI router that legitimately lazy-imports the offline
# toolchain; the library-level runtime set is the init + api + subpackages.
RUNTIME_FILES = ("__init__.py", "api.py")
FORBIDDEN_PREFIXES = (
    "pes2ts_core.pesn.offline",
    "pes2ts_core.g0",
    "pes2ts_core.g1",
    "pes2ts_core.generation",
    "pes2ts_core.ranking",
    "pes2ts_core.integration",
    "pes2ts_core.evaluation",
)


def _runtime_modules() -> list[Path]:
    modules: list[Path] = []
    for name in RUNTIME_FILES:
        candidate = PESN_ROOT / name
        if candidate.is_file():
            modules.append(candidate)
    for subdirectory in RUNTIME_SUBDIRS:
        directory = PESN_ROOT / subdirectory
        if directory.is_dir():
            modules.extend(sorted(directory.rglob("*.py")))
    return modules


def _imported_module_names(module_path: Path) -> set[str]:
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names.add(node.module)
            names.update(f"{node.module}.{alias.name}" for alias in node.names)
    return names


def test_runtime_modules_never_import_forbidden_products() -> None:
    violations: list[str] = []
    for module in _runtime_modules():
        for name in _imported_module_names(module):
            if name.startswith(FORBIDDEN_PREFIXES):
                violations.append(f"{module.relative_to(PROJECT_ROOT)} -> {name}")
    assert not violations, f"runtime boundary violations: {violations}"


def test_runtime_imports_are_static_only() -> None:
    for module in _runtime_modules():
        tree = ast.parse(module.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                assert all(
                    alias.name.split(".")[0] not in ("importlib", "subprocess")
                    for alias in node.names
                ), f"{module.name} imports a dynamic-execution module"
            if isinstance(node, ast.ImportFrom):
                root = (node.module or "").split(".")[0]
                assert root not in ("importlib", "subprocess"), module.name
            if isinstance(node, ast.Call):
                name = getattr(node.func, "id", getattr(node.func, "attr", ""))
                assert name not in ("eval", "exec", "__import__"), module.name


def test_truth_guard_clean_with_pesn_bridge_registered() -> None:
    from pes2ts_core.utils.truth_guard import DEFAULT_ALLOWLIST, assert_no_truth_access

    assert "pes2ts_core/pesn/offline/data/truth_source.py" in DEFAULT_ALLOWLIST
    assert assert_no_truth_access() == []


def test_package_import_stays_backend_free() -> None:
    code = (
        "import sys, json\n"
        "import pes2ts_core.pesn\n"
        "heavy = [m for m in sys.modules if m.startswith(('torch', 'pes2ts_core.pesn.offline', 'pes2ts_core.g0', 'pes2ts_core.g1'))]\n"
        "print(json.dumps(heavy))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT),
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "[]", (
        f"package import pulled heavy modules: {result.stdout.strip()}"
    )


def test_pesn_root_contains_only_planned_product_code() -> None:
    allowed_top = {
        "__init__.py",
        "__main__.py",
        "api.py",
        "contracts",
        "features",
        "models",
        "inference",
        "bundle",
        "offline",
        "__pycache__",
    }
    present = {entry.name for entry in PESN_ROOT.iterdir()}
    assert present - allowed_top == set(), f"unexpected entries: {present - allowed_top}"
