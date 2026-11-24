"""Repository layout guarantees for the post-cleanup three-plate layout.

Cheap structural assertions (no business imports, no torch): they lock the
directory contract so a future change cannot resurrect the deleted frozen
history (``archive/``, ``tests/legacy/``, ``pes2ts_core/pesn/`` -- preserved on
branch ``frozen/v4-v54-history``), bring back the retired ``scripts/`` pile, or
drop the anchor modules of the three active plates.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

PLATES = ("pes2ts_core", "pes2ts_data", "pes2ts_v5", "infra")

#: Frozen history that must stay absent from the active tree.
FROZEN_ABSENT = ("archive", "tests/legacy", "pes2ts_core/pesn")

#: Anchor modules per plate.  Package ``__init__`` files are the
#: read-this-first anchors; the entry modules are the documented front doors.
ANCHOR_MODULES = (
    "pes2ts_core/__init__.py",
    "pes2ts_core/cli.py",
    "pes2ts_core/contracts",  # relocated contracts package (or module)
    "pes2ts_core/g0/__init__.py",
    "pes2ts_core/g1/__init__.py",
    "pes2ts_core/generation/__init__.py",
    "pes2ts_core/integration/acp/__init__.py",
    "pes2ts_core/utils/__init__.py",
    "pes2ts_data/__init__.py",
    "pes2ts_data/contracts.py",  # StandardBatch
    "pes2ts_data/registry.py",
    "pes2ts_data/readers",
    "pes2ts_data/splits/__init__.py",
    "pes2ts_data/ingest/__init__.py",
    "pes2ts_data/readers/__init__.py",
    "pes2ts_v5/__init__.py",
    "pes2ts_v5/runner.py",  # the only training/eval entry point
    "pes2ts_v5/screening.py",
    "pes2ts_v5/configs/schema.py",
    "pes2ts_v5/models/registry.py",
    "pes2ts_v5/losses/registry.py",
    "pes2ts_v5/training/trainer.py",  # the single training loop
    "infra/__init__.py",
    "infra/governance/arch_guard.py",
    "infra/governance/arch_guard_baseline.json",
    "infra/governance/check_docs_links.py",
    "infra/governance/docs_consistency.py",
    "infra/runtime/sha_sidecar.py",
)

TEST_TIERS = ("kernel", "data", "v5")
TEST_TIER_ANCHORS = (
    "tests/kernel/__init__.py",
    "tests/data/conftest.py",
    "tests/v5/conftest.py",
)

INFRA_MAX_LINES = 2_500


def _zone_py_lines(zone: str) -> int:
    total = 0
    for path in (REPO_ROOT / zone).rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        total += len(path.read_text(encoding="utf-8", errors="replace").splitlines())
    return total


def test_three_plates_and_infra_exist() -> None:
    for plate in PLATES:
        path = REPO_ROOT / plate
        assert path.is_dir(), f"missing plate directory: {plate}/"
        assert (path / "__init__.py").is_file(), (
            f"plate {plate}/ needs a package __init__.py anchor"
        )


def _frozen_leftovers(rel: str) -> list[str]:
    """Files of frozen history still present (``__pycache__`` junk ignored)."""
    path = REPO_ROOT / rel
    if not path.exists():
        return []
    if path.is_file():
        return [rel]
    leftovers: list[str] = []
    for candidate in path.rglob("*"):
        if not candidate.is_file():
            continue
        if "__pycache__" in candidate.parts:
            continue
        leftovers.append(candidate.relative_to(REPO_ROOT).as_posix())
    return sorted(leftovers)


def test_frozen_history_is_absent() -> None:
    leftovers = {rel: _frozen_leftovers(rel) for rel in FROZEN_ABSENT}
    present = {rel: files for rel, files in leftovers.items() if files}
    assert present == {}, (
        "frozen history must stay off the active tree (branch "
        f"frozen/v4-v54-history): {present}"
    )


def test_config_and_test_layout() -> None:
    assert (REPO_ROOT / "config" / "experiments").is_dir()
    experiments = sorted((REPO_ROOT / "config" / "experiments").glob("*.yaml"))
    assert experiments, "config/experiments/ must hold the active experiment YAMLs"
    assert (REPO_ROOT / "config" / "data" / "splits.yaml").is_file()
    assert (REPO_ROOT / "config" / "registry" / "g_v_map.yaml").is_file()
    for tier in TEST_TIERS:
        assert (REPO_ROOT / "tests" / tier).is_dir(), f"missing tests/{tier}/"
    for rel in TEST_TIER_ANCHORS:
        assert (REPO_ROOT / rel).is_file(), f"missing test-tier anchor: {rel}"


def test_anchor_modules_exist() -> None:
    missing = [rel for rel in ANCHOR_MODULES if not (REPO_ROOT / rel).exists()]
    assert missing == [], f"missing plate anchor modules: {missing}"


def test_scripts_has_no_python_files_left() -> None:
    scripts = REPO_ROOT / "scripts"
    if not scripts.exists():
        return
    leftovers = sorted(
        path.relative_to(REPO_ROOT).as_posix() for path in scripts.rglob("*.py")
    )
    assert leftovers == [], (
        "scripts/ was retired by the restructure; code belongs to "
        f"pes2ts_core/ pes2ts_data/ pes2ts_v5/ or infra/: {leftovers}"
    )


def test_standard_batch_contract_lives_in_pes2ts_data() -> None:
    contracts = (REPO_ROOT / "pes2ts_data" / "contracts.py").read_text(encoding="utf-8")
    assert "class StandardBatch" in contracts


def test_single_trainer_module_is_the_engine_loop() -> None:
    trainer = (REPO_ROOT / "pes2ts_v5" / "training" / "trainer.py").read_text(
        encoding="utf-8"
    )
    assert "class V5Trainer" in trainer
    assert "backward(" in trainer


def test_core_contracts_plate_exists() -> None:
    contracts = REPO_ROOT / "pes2ts_core" / "contracts"
    assert contracts.exists(), "pes2ts_core/contracts must hold the versioned contracts"
    if contracts.is_dir():
        assert any(contracts.glob("*.py")), "pes2ts_core/contracts/ must contain modules"


def test_infra_stays_small() -> None:
    lines = _zone_py_lines("infra")
    assert lines <= INFRA_MAX_LINES, (
        f"infra/ is {lines} py lines (> {INFRA_MAX_LINES}); it may only hold "
        "governance and WSL ops"
    )


def test_guard_documents_three_plates_and_frozen_branch() -> None:
    text = (REPO_ROOT / "infra" / "governance" / "arch_guard.py").read_text(
        encoding="utf-8"
    )
    assert "three-plate" in text
    assert "frozen/v4-v54-history" in text
    assert "G6" in text
