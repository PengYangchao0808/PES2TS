"""Repository layout guarantees for the five-zone restructure (ADR-0036).

These are cheap structural assertions (no imports of business code, no torch):
they lock the directory contract so a future change cannot silently drop a
zone, resurrect the retired ``scripts/`` pile, or remove the anchor modules
that make each zone discoverable.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

ZONES = ("pes2ts_core", "pes2ts_data", "pes2ts_v5", "infra", "archive")

#: Anchor modules per zone.  Package ``__init__`` files are the read-this-first
#: anchors; the entry modules are the documented front doors (ADR-0036).
ANCHOR_MODULES = (
    "pes2ts_core/__init__.py",
    "pes2ts_core/cli.py",
    "pes2ts_core/pesn/__init__.py",  # frozen legacy model tree
    "pes2ts_data/__init__.py",
    "pes2ts_data/contracts.py",  # StandardBatch
    "pes2ts_data/registry.py",
    "pes2ts_data/splits/__init__.py",
    "pes2ts_data/ingest/__init__.py",
    "pes2ts_data/builders/__init__.py",
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
    "infra/governance/render_gv_map.py",
    "infra/runtime/sha_sidecar.py",
    "archive/__init__.py",
    "archive/v4/README.md",
    "archive/v5x_scripts/__init__.py",
    "archive/v5x_configs/campaigns",
)

TEST_TIERS = ("kernel", "data", "v5", "legacy")
TEST_TIER_ANCHORS = (
    "tests/kernel/__init__.py",
    "tests/data/conftest.py",
    "tests/v5/conftest.py",
    "tests/legacy/conftest.py",
)


def test_five_zones_exist() -> None:
    for zone in ZONES:
        path = REPO_ROOT / zone
        assert path.is_dir(), f"missing zone directory: {zone}/"
        assert (path / "__init__.py").is_file(), (
            f"zone {zone}/ needs a package __init__.py anchor"
        )


def test_zone_anchor_modules_exist() -> None:
    missing = [rel for rel in ANCHOR_MODULES if not (REPO_ROOT / rel).exists()]
    assert missing == [], f"missing zone anchor modules: {missing}"


def test_active_config_and_test_layout() -> None:
    assert (REPO_ROOT / "config" / "experiments").is_dir()
    experiments = sorted((REPO_ROOT / "config" / "experiments").glob("*.yaml"))
    assert experiments, "config/experiments/ must hold the active experiment YAMLs"
    assert (REPO_ROOT / "config" / "data" / "splits.yaml").is_file()
    assert (REPO_ROOT / "config" / "registry" / "g_v_map.yaml").is_file()
    for tier in TEST_TIERS:
        assert (REPO_ROOT / "tests" / tier).is_dir(), f"missing tests/{tier}/"
    for rel in TEST_TIER_ANCHORS:
        assert (REPO_ROOT / rel).is_file(), f"missing test-tier anchor: {rel}"


def test_scripts_has_no_python_files_left() -> None:
    scripts = REPO_ROOT / "scripts"
    if not scripts.exists():
        return
    leftovers = sorted(path.relative_to(REPO_ROOT).as_posix() for path in scripts.rglob("*.py"))
    assert leftovers == [], (
        "scripts/ was retired by the five-zone restructure; move code to "
        f"pes2ts_core/ pes2ts_data/ pes2ts_v5/ infra/ or archive/: {leftovers}"
    )


def test_standard_batch_contract_lives_in_pes2ts_data() -> None:
    contracts = (REPO_ROOT / "pes2ts_data" / "contracts.py").read_text(encoding="utf-8")
    assert "class StandardBatch" in contracts
    # And it is not duplicated at the repo root or resurrected under scripts/.
    assert not (REPO_ROOT / "contracts.py").exists()


def test_single_trainer_module_is_the_engine_loop() -> None:
    trainer = (REPO_ROOT / "pes2ts_v5" / "training" / "trainer.py").read_text(
        encoding="utf-8"
    )
    assert "class V5Trainer" in trainer
    assert ".backward()" in trainer or "backward(" in trainer


def test_governance_gate_scripts_exist() -> None:
    governance = REPO_ROOT / "infra" / "governance"
    for name in ("arch_guard.py", "check_docs_links.py", "docs_consistency.py"):
        assert (governance / name).is_file(), f"missing governance gate: {name}"
    assert (governance / "docs_consistency_config.json").is_file()


def test_adr_0036_and_agent_zones_are_present() -> None:
    adr = (
        REPO_ROOT
        / "docs"
        / "design"
        / "decisions"
        / "ADR-0036-功能分区与单一训练引擎.md"
    )
    assert adr.is_file(), f"missing ADR-0036: {adr}"
    agent = (REPO_ROOT / "AGENT.md").read_text(encoding="utf-8")
    assert "No-Read / No-Edit Zone" in agent
    assert "ADR-0036" in agent


def test_rgignore_covers_frozen_zones() -> None:
    rgignore = REPO_ROOT / ".rgignore"
    assert rgignore.is_file(), "missing repo-root .rgignore"
    text = rgignore.read_text(encoding="utf-8")
    for pattern in (
        "archive/",
        "tests/legacy/",
        "pes2ts_core/pesn/models/v54*",
        "pes2ts_core/pesn/**/*dual_ts*",
        "data/",
        "outputs/",
        ".slim/",
    ):
        assert pattern in text, f".rgignore must cover {pattern!r}"
