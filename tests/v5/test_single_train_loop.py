"""The "one trainer" invariant: ``.backward()`` / ``optimizer.step()`` live only
inside ``pes2ts_v5/training/``.

This is a static AST scan (no imports, no torch), so it stays fast and catches a
second loss/optimiser loop anywhere in the training engine — including a future
copy-pasted trainer in another subpackage.
"""

from __future__ import annotations

import ast
from pathlib import Path

V5_ROOT = Path(__file__).resolve().parents[2] / "pes2ts_v5"
TRAINING_PREFIX = "training/"


def _iter_sources() -> list[Path]:
    return sorted(path for path in V5_ROOT.rglob("*.py")
                  if "__pycache__" not in path.parts)


def _is_optimizer_expr(node: ast.AST) -> bool:
    names: list[str] = []
    while isinstance(node, ast.Attribute):
        names.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        names.append(node.id)
    return any("optimizer" in name.lower() for name in names)


def _scan() -> tuple[list[tuple[str, int, str]], list[tuple[str, int, str]]]:
    """Return ``(violations_outside_training, hits_inside_training)``."""
    violations: list[tuple[str, int, str]] = []
    hits: list[tuple[str, int, str]] = []
    for path in _iter_sources():
        relative = path.relative_to(V5_ROOT).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue
            target: str | None = None
            if node.func.attr == "backward":
                target = ".backward("
            elif node.func.attr == "step" and _is_optimizer_expr(node.func.value):
                target = "optimizer.step("
            if target is None:
                continue
            entry = (relative, int(node.lineno), target)
            if relative.startswith(TRAINING_PREFIX):
                hits.append(entry)
            else:
                violations.append(entry)
    return violations, hits


def test_only_training_package_backprops() -> None:
    violations, hits = _scan()
    assert not violations, (
        "loss.backward()/optimizer.step() outside pes2ts_v5/training/: "
        + ", ".join(f"{path}:{line} ({what})" for path, line, what in violations))
    assert hits, "expected at least one backward/step call inside training/"


def test_trainer_is_the_single_training_loop() -> None:
    _, hits = _scan()
    files_with_backward = {path for path, _, what in hits if what == ".backward("}
    files_with_step = {path for path, _, what in hits if what == "optimizer.step("}
    assert files_with_backward == {"training/trainer.py"}
    assert files_with_step == {"training/trainer.py"}


def test_no_second_optimizer_instantiation_outside_training() -> None:
    offenders: list[str] = []
    for path in _iter_sources():
        relative = path.relative_to(V5_ROOT).as_posix()
        if relative.startswith(TRAINING_PREFIX):
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr in {"AdamW", "Adam", "SGD", "RMSprop"}
                    and "torch" in ast.unparse(node.func)):
                offenders.append(f"{relative}:{node.lineno}")
    assert not offenders, f"optimizer construction outside training/: {offenders}"
