from __future__ import annotations

import json

import pytest

from pes2ts_core.contracts import FORBIDDEN_TRUTH_KEYS
from pes2ts_core.g1.v2_verify import FORBIDDEN_EXPORT_KEYS
from pes2ts_core.generation.planning import capabilities as caps

ORGANIC = ["B", "Br", "C", "Cl", "F", "H", "I", "N", "O", "P", "S", "Si"]


def _receipt(
    probe_id: str = "probe-1",
    *,
    modes: tuple[str, ...] = ("SINGLE_1D",),
    status: str = "pass",
    engine_version: str = "6.1.0",
    adapter_version: str | None = None,
    date: str = "2026-10-01",
    evidence_ref: str = "evidence/task-16",
) -> dict:
    doc = {
        "probe_id": probe_id,
        "date": date,
        "engine_version": engine_version,
        "status": status,
        "evidence_ref": evidence_ref,
        "modes": list(modes),
    }
    if adapter_version is not None:
        doc["adapter_version"] = adapter_version
    return doc


def _entry_doc(
    entry_id: str,
    layer: str,
    engine: str,
    adapter: str,
    *,
    modes: list[str],
    kinds: list[str],
    max_coords: int,
    engine_version: str = "6.1.0",
    adapter_version: str = "acp-pes-scan-v0",
    receipts: list[dict] | None = None,
    coverage: dict | None = None,
    constraints: dict | None = None,
    custom_schedule: bool = False,
    point_limits: tuple[int, int] = (9, 101),
    notes: str = "",
) -> dict:
    return {
        "entry_id": entry_id,
        "layer": layer,
        "engine": engine,
        "adapter": adapter,
        "capability": {
            "engine_version": engine_version,
            "adapter_version": adapter_version,
            "supported_modes": modes,
            "coordinate_kinds": kinds,
            "max_scan_coordinates": max_coords,
            "point_limits": {"baseline": point_limits[0], "max": point_limits[1]},
            "custom_schedule_support": custom_schedule,
            "constraint_support": constraints
            or {
                "native_scan": True,
                "per_point_constraints": False,
                "simul_scan": False,
            },
            "method_element_coverage": coverage if coverage is not None else {},
            "probe_receipts": receipts if receipts is not None else [],
        },
        "notes": notes,
    }


def _registry_doc(entries: list[dict], registry_id: str = "test-registry") -> dict:
    return {
        "schema_version": caps.REGISTRY_SCHEMA_VERSION,
        "registry_id": registry_id,
        "description": "synthetic test registry",
        "entries": entries,
    }


def _load(tmp_path, doc: dict) -> caps.CapabilityRegistry:
    path = tmp_path / "orca_capabilities_v1.json"
    path.write_text(json.dumps(doc), encoding="utf-8")
    return caps.load_capability_registry(path)


def _pass(modes: tuple[str, ...], engine_version: str = "6.1.0") -> list[dict]:
    return [_receipt(modes=modes, engine_version=engine_version)]


def _smoked_stack(
    tmp_path,
    *,
    engine_modes: list[str],
    adapter_modes: list[str],
    deployment_modes: list[str],
    engine_kinds: list[str],
    adapter_kinds: list[str],
    deployment_kinds: list[str],
    engine_max: int,
    adapter_max: int,
    deployment_max: int,
    engine_receipts: list[dict] | None = None,
    adapter_receipts: list[dict] | None = None,
    deployment_receipts: list[dict] | None = None,
    engine_coverage: dict | None = None,
    adapter_coverage: dict | None = None,
    deployment_coverage: dict | None = None,
    engine_constraints: dict | None = None,
    adapter_constraints: dict | None = None,
    deployment_constraints: dict | None = None,
    engine_limits: tuple[int, int] = (9, 101),
    adapter_limits: tuple[int, int] = (9, 101),
    deployment_limits: tuple[int, int] = (9, 101),
    engine_versions: str = "6.1.0",
) -> tuple[caps.CapabilityRegistry, caps.EffectiveCapability]:
    engine = _entry_doc(
        "eng",
        "engine",
        "orca",
        "acp",
        modes=engine_modes,
        kinds=engine_kinds,
        max_coords=engine_max,
        engine_version=engine_versions,
        receipts=engine_receipts if engine_receipts is not None else [],
        coverage=engine_coverage,
        constraints=engine_constraints,
        point_limits=engine_limits,
    )
    adapter = _entry_doc(
        "adp",
        "adapter",
        "orca",
        "acp",
        modes=adapter_modes,
        kinds=adapter_kinds,
        max_coords=adapter_max,
        engine_version=engine_versions,
        receipts=adapter_receipts if adapter_receipts is not None else [],
        coverage=adapter_coverage,
        constraints=adapter_constraints,
        point_limits=adapter_limits,
    )
    deployment = _entry_doc(
        "dep",
        "deployment",
        "orca",
        "acp",
        modes=deployment_modes,
        kinds=deployment_kinds,
        max_coords=deployment_max,
        engine_version=engine_versions,
        receipts=deployment_receipts if deployment_receipts is not None else [],
        coverage=deployment_coverage,
        constraints=deployment_constraints,
        point_limits=deployment_limits,
    )
    registry = _load(tmp_path, _registry_doc([engine, adapter, deployment]))
    effective = caps.effective_capability("eng", "adp", "dep", registry=registry)
    return registry, effective


def test_shipped_registry_loads_with_honest_initial_state():
    registry = caps.load_capability_registry()
    assert [entry.entry_id for entry in registry.entries] == [
        "orca-acp-engine",
        "acp-adapter-v0",
        "orca-xtb-bridge",
        "xtb-native-path",
        "g2-path-adapter",
        "acp-xtb-path-adapter",
        "deployment-baseline",
    ]
    assert all(not entry.probe_receipts for entry in registry.entries)

    orca = registry.entry("orca-acp-engine")
    assert set(orca.supported_modes) == {
        "SINGLE_1D",
        "COUPLED_1D",
        "SCHEDULED_1D",
        "PATH_NEB",
    }
    assert orca.coordinate_kinds == ("B", "A", "D")
    assert orca.max_scan_coordinates == 3
    assert orca.point_limits == caps.PointLimits(baseline=9, max=101)

    adapter = registry.entry("acp-adapter-v0")
    assert adapter.supported_modes == ("SINGLE_1D",)
    assert adapter.coordinate_kinds == ("B",)
    assert adapter.max_scan_coordinates == 1
    assert adapter.custom_schedule_support is False
    assert adapter.constraint_support == caps.ConstraintSupport(
        native_scan=True, per_point_constraints=False, simul_scan=False
    )


def test_shipped_xtb_path_entries_execute_through_acp():
    registry = caps.load_capability_registry()
    xtb_engine = registry.entry("xtb-native-path")
    legacy_adapter = registry.entry("g2-path-adapter")
    acp_adapter = registry.entry("acp-xtb-path-adapter")

    for entry in (xtb_engine, legacy_adapter, acp_adapter):
        assert entry.adapter == "acp"
        assert entry.engine == "xtb"
        assert entry.supported_modes == ("PATH_NEB",)
        assert entry.coordinate_kinds == ()
        assert entry.max_scan_coordinates == 0
        assert entry.constraint_support == caps.ConstraintSupport(
            native_scan=False, per_point_constraints=False, simul_scan=False
        )
        assert entry.probe_receipts == ()
    assert acp_adapter.layer == "adapter"
    assert acp_adapter.method_element_coverage == {}

    effective = caps.effective_capability(
        "xtb-native-path", "acp-xtb-path-adapter", "deployment-baseline",
        registry=registry,
    )
    assert effective.supported_modes == frozenset({"PATH_NEB"})
    assert effective.enabled_modes == frozenset()
    assert any("unprobed" in reason for reason in effective.disabled_reasons)
    with pytest.raises(caps.BackendCapabilityError) as excinfo:
        caps.assert_mode_supported(effective, "PATH_NEB", (), 0)
    assert excinfo.value.code == caps.BACKEND_CAPABILITY_MISSING
    check = caps.capability_check(effective, "PATH_NEB", (), 0)
    assert check["status"] == "unknown"
    assert check["missing"]


def test_shipped_baseline_effective_capability_disables_everything():
    registry = caps.load_capability_registry()
    effective = caps.effective_capability(
        "orca-acp-engine", "acp-adapter-v0", "deployment-baseline", registry=registry
    )
    assert effective.supported_modes == frozenset({"SINGLE_1D"})
    assert effective.enabled_modes == frozenset()
    assert effective.coordinate_kinds == frozenset({"B"})
    assert effective.max_scan_coordinates == 1
    assert effective.custom_schedule_support is False
    assert effective.constraint_support.native_scan is True
    assert effective.constraint_support.simul_scan is False
    assert effective.method_element_coverage == {}
    assert effective.probe_receipts == ()

    with pytest.raises(caps.BackendCapabilityError) as excinfo:
        caps.assert_mode_supported(effective, "SINGLE_1D", ("B",), 1)
    assert excinfo.value.code == caps.BACKEND_CAPABILITY_MISSING
    assert any("not enabled" in reason for reason in excinfo.value.reasons)

    check = caps.capability_check(effective, "SINGLE_1D", ("B",), 1)
    assert check["status"] == "unknown"
    assert check["missing"]


def test_effective_capability_intersects_three_layers(tmp_path):
    _, effective = _smoked_stack(
        tmp_path,
        engine_modes=["SINGLE_1D", "COUPLED_1D", "SCHEDULED_1D"],
        adapter_modes=["SINGLE_1D", "COUPLED_1D"],
        deployment_modes=["SINGLE_1D", "COUPLED_1D"],
        engine_kinds=["B", "A", "D"],
        adapter_kinds=["B"],
        deployment_kinds=["B"],
        engine_max=3,
        adapter_max=1,
        deployment_max=1,
        engine_receipts=_pass(("SINGLE_1D", "COUPLED_1D")),
        adapter_receipts=_pass(("SINGLE_1D", "COUPLED_1D")),
        deployment_receipts=_pass(("SINGLE_1D", "COUPLED_1D")),
    )
    assert effective.supported_modes == frozenset({"SINGLE_1D", "COUPLED_1D"})
    assert effective.enabled_modes == frozenset({"SINGLE_1D", "COUPLED_1D"})
    assert effective.coordinate_kinds == frozenset({"B"})
    assert effective.max_scan_coordinates == 1
    assert effective.point_limits == caps.PointLimits(baseline=9, max=101)
    assert effective.engine_entry_id == "eng"
    assert effective.adapter_entry_id == "adp"
    assert effective.deployment_entry_id == "dep"

    caps.assert_mode_supported(effective, "SINGLE_1D", ("B",), 1)
    assert "SCHEDULED_1D" not in effective.supported_modes

    check = caps.capability_check(effective, "SINGLE_1D", ("B",), 1)
    assert check == {"status": "pass", "missing": []}


def test_mode_not_declared_by_adapter_is_disabled(tmp_path):
    _, effective = _smoked_stack(
        tmp_path,
        engine_modes=["SINGLE_1D", "COUPLED_1D", "SCHEDULED_1D"],
        adapter_modes=["SINGLE_1D"],
        deployment_modes=["SINGLE_1D"],
        engine_kinds=["B", "A", "D"],
        adapter_kinds=["B"],
        deployment_kinds=["B"],
        engine_max=3,
        adapter_max=1,
        deployment_max=1,
        engine_receipts=_pass(("SINGLE_1D", "COUPLED_1D", "SCHEDULED_1D")),
        adapter_receipts=_pass(("SINGLE_1D",)),
        deployment_receipts=_pass(("SINGLE_1D",)),
    )
    assert "COUPLED_1D" not in effective.supported_modes
    assert "COUPLED_1D" not in effective.enabled_modes

    with pytest.raises(caps.BackendCapabilityError) as excinfo:
        caps.assert_mode_supported(effective, "COUPLED_1D", ("B", "B"), 2)
    assert excinfo.value.code == caps.BACKEND_CAPABILITY_MISSING
    assert any("intersection" in reason for reason in excinfo.value.reasons)

    check = caps.capability_check(effective, "COUPLED_1D", ("B", "B"), 2)
    assert check["status"] == "fail"


def test_unsmoked_mode_not_enabled(tmp_path):
    _, effective = _smoked_stack(
        tmp_path,
        engine_modes=["SINGLE_1D", "COUPLED_1D"],
        adapter_modes=["SINGLE_1D", "COUPLED_1D"],
        deployment_modes=["SINGLE_1D", "COUPLED_1D"],
        engine_kinds=["B", "A"],
        adapter_kinds=["B"],
        deployment_kinds=["B"],
        engine_max=3,
        adapter_max=3,
        deployment_max=3,
    )
    assert effective.supported_modes == frozenset({"SINGLE_1D", "COUPLED_1D"})
    assert effective.enabled_modes == frozenset()
    assert any("unprobed" in reason for reason in effective.disabled_reasons)

    with pytest.raises(caps.BackendCapabilityError) as excinfo:
        caps.assert_mode_supported(effective, "SINGLE_1D", ("B",), 1)
    assert excinfo.value.code == caps.BACKEND_CAPABILITY_MISSING

    check = caps.capability_check(effective, "SINGLE_1D", ("B",), 1)
    assert check["status"] == "unknown"


def test_multi_coordinate_request_rejected_when_max_scan_coordinates_is_one(tmp_path):
    _, effective = _smoked_stack(
        tmp_path,
        engine_modes=["SINGLE_1D", "COUPLED_1D"],
        adapter_modes=["SINGLE_1D", "COUPLED_1D"],
        deployment_modes=["SINGLE_1D", "COUPLED_1D"],
        engine_kinds=["B", "A"],
        adapter_kinds=["B"],
        deployment_kinds=["B"],
        engine_max=3,
        adapter_max=1,
        deployment_max=1,
        engine_receipts=_pass(("SINGLE_1D", "COUPLED_1D")),
        adapter_receipts=_pass(("SINGLE_1D", "COUPLED_1D")),
        deployment_receipts=_pass(("SINGLE_1D", "COUPLED_1D")),
    )
    assert "COUPLED_1D" in effective.enabled_modes
    assert effective.max_scan_coordinates == 1

    with pytest.raises(caps.BackendCapabilityError) as excinfo:
        caps.assert_mode_supported(effective, "COUPLED_1D", ("B", "B"), 2)
    assert excinfo.value.code == caps.BACKEND_CAPABILITY_MISSING
    assert any("max_scan_coordinates" in reason for reason in excinfo.value.reasons)


def test_xtb_and_orca_entries_are_separate():
    registry = caps.load_capability_registry()
    xtb = registry.entry("xtb-native-path")
    bridge = registry.entry("orca-xtb-bridge")
    orca = registry.entry("orca-acp-engine")

    assert xtb.entry_id != bridge.entry_id
    assert xtb.engine == "xtb" and bridge.engine == "orca"
    assert xtb.supported_modes == ("PATH_NEB",)
    assert xtb.coordinate_kinds == ()
    assert xtb.max_scan_coordinates == 0
    assert xtb.constraint_support == caps.ConstraintSupport(
        native_scan=False, per_point_constraints=False, simul_scan=False
    )
    assert "PATH_NEB" not in bridge.supported_modes
    assert set(bridge.supported_modes) == {"SINGLE_1D", "COUPLED_1D", "SCHEDULED_1D"}
    assert set(orca.supported_modes) != set(xtb.supported_modes)

    effective = caps.effective_capability(
        "xtb-native-path", "g2-path-adapter", "deployment-baseline", registry=registry
    )
    assert effective.supported_modes == frozenset({"PATH_NEB"})
    assert effective.enabled_modes == frozenset()
    with pytest.raises(caps.BackendCapabilityError) as excinfo:
        caps.assert_mode_supported(effective, "PATH_NEB", (), 0)
    assert excinfo.value.code == caps.BACKEND_CAPABILITY_MISSING


def test_probe_receipt_roundtrip():
    receipt_doc = _receipt(
        modes=["SINGLE_1D", "COUPLED_1D"],
        adapter_version="acp-pes-scan-v0",
    )
    receipt = caps.ProbeReceipt.from_doc(receipt_doc)
    assert receipt.to_doc() == receipt_doc
    assert receipt.modes == ("SINGLE_1D", "COUPLED_1D")

    bare_doc = _receipt(modes=[])
    bare = caps.ProbeReceipt.from_doc(bare_doc)
    assert "modes" not in bare.to_doc()
    assert caps.ProbeReceipt.from_doc(bare.to_doc()) == bare
    assert bare.modes == ()

    registry = caps.load_capability_registry()
    payload = caps.dumps_capability_registry(registry)
    restored = caps.loads_capability_registry(payload)
    assert restored.registry_id == registry.registry_id
    assert [entry.entry_id for entry in restored.entries] == [
        entry.entry_id for entry in registry.entries
    ]
    for original, loaded in zip(registry.entries, restored.entries, strict=True):
        assert original.to_doc() == loaded.to_doc()
        assert original.probe_receipts == loaded.probe_receipts
    assert caps.dumps_capability_registry(restored) == payload


def test_shipped_registry_dumps_deterministically():
    first = caps.load_capability_registry()
    second = caps.load_capability_registry()
    assert caps.dumps_capability_registry(first) == caps.dumps_capability_registry(second)


def test_fail_receipt_blocks_enablement(tmp_path):
    _, effective = _smoked_stack(
        tmp_path,
        engine_modes=["SINGLE_1D"],
        adapter_modes=["SINGLE_1D"],
        deployment_modes=["SINGLE_1D"],
        engine_kinds=["B"],
        adapter_kinds=["B"],
        deployment_kinds=["B"],
        engine_max=1,
        adapter_max=1,
        deployment_max=1,
        engine_receipts=[
            _receipt(probe_id="probe-pass", modes=("SINGLE_1D",), status="pass"),
            _receipt(probe_id="probe-fail", modes=("SINGLE_1D",), status="fail"),
        ],
        adapter_receipts=_pass(("SINGLE_1D",)),
        deployment_receipts=_pass(("SINGLE_1D",)),
    )
    assert "SINGLE_1D" in effective.supported_modes
    assert "SINGLE_1D" not in effective.enabled_modes
    assert any("failed" in reason for reason in effective.disabled_reasons)

    with pytest.raises(caps.BackendCapabilityError):
        caps.assert_mode_supported(effective, "SINGLE_1D", ("B",), 1)


def test_receipt_without_modes_enables_nothing(tmp_path):
    _, effective = _smoked_stack(
        tmp_path,
        engine_modes=["SINGLE_1D"],
        adapter_modes=["SINGLE_1D"],
        deployment_modes=["SINGLE_1D"],
        engine_kinds=["B"],
        adapter_kinds=["B"],
        deployment_kinds=["B"],
        engine_max=1,
        adapter_max=1,
        deployment_max=1,
        engine_receipts=[_receipt(modes=())],
        adapter_receipts=[_receipt(modes=())],
        deployment_receipts=[_receipt(modes=())],
    )
    assert effective.supported_modes == frozenset({"SINGLE_1D"})
    assert effective.enabled_modes == frozenset()


def test_receipt_at_wrong_engine_version_does_not_enable(tmp_path):
    _, effective = _smoked_stack(
        tmp_path,
        engine_modes=["SINGLE_1D"],
        adapter_modes=["SINGLE_1D"],
        deployment_modes=["SINGLE_1D"],
        engine_kinds=["B"],
        adapter_kinds=["B"],
        deployment_kinds=["B"],
        engine_max=1,
        adapter_max=1,
        deployment_max=1,
        engine_receipts=_pass(("SINGLE_1D",)),
        adapter_receipts=_pass(("SINGLE_1D",)),
        deployment_receipts=_pass(("SINGLE_1D",), engine_version="5.0.0"),
    )
    assert effective.enabled_modes == frozenset()
    assert any(
        "dep [unprobed]: no probe receipt covers SINGLE_1D" in reason
        for reason in effective.disabled_reasons
    )


def test_registry_load_rejects_bad_mode(tmp_path):
    engine = _entry_doc(
        "eng", "engine", "orca", "acp",
        modes=["GRID_2D"], kinds=["B"], max_coords=1,
    )
    adapter = _entry_doc(
        "adp", "adapter", "orca", "acp",
        modes=["SINGLE_1D"], kinds=["B"], max_coords=1,
    )
    deployment = _entry_doc(
        "dep", "deployment", "orca", "acp",
        modes=["SINGLE_1D"], kinds=["B"], max_coords=1,
    )
    path = tmp_path / "bad.json"
    path.write_text(
        json.dumps(_registry_doc([engine, adapter, deployment])), encoding="utf-8"
    )
    with pytest.raises(caps.CapabilityRegistryError) as excinfo:
        caps.load_capability_registry(path)
    assert excinfo.value.code == caps.CAPABILITY_REGISTRY_INVALID


def test_registry_load_rejects_duplicate_entry_ids(tmp_path):
    entry = _entry_doc(
        "eng", "engine", "orca", "acp",
        modes=["SINGLE_1D"], kinds=["B"], max_coords=1,
    )
    path = tmp_path / "dup.json"
    path.write_text(json.dumps(_registry_doc([entry, entry])), encoding="utf-8")
    with pytest.raises(caps.CapabilityRegistryError) as excinfo:
        caps.load_capability_registry(path)
    assert excinfo.value.code == caps.CAPABILITY_REGISTRY_INVALID
    assert any("duplicate" in reason for reason in excinfo.value.reasons)


def test_registry_load_rejects_inverted_point_limits(tmp_path):
    entry = _entry_doc(
        "eng", "engine", "orca", "acp",
        modes=["SINGLE_1D"], kinds=["B"], max_coords=1,
        point_limits=(50, 40),
    )
    path = tmp_path / "limits.json"
    path.write_text(json.dumps(_registry_doc([entry])), encoding="utf-8")
    with pytest.raises(caps.CapabilityRegistryError) as excinfo:
        caps.load_capability_registry(path)
    assert excinfo.value.code == caps.CAPABILITY_REGISTRY_INVALID


def test_registry_load_rejects_bad_probe_status(tmp_path):
    entry = _entry_doc(
        "eng", "engine", "orca", "acp",
        modes=["SINGLE_1D"], kinds=["B"], max_coords=1,
        receipts=[_receipt(status="unknown")],
    )
    path = tmp_path / "receipt.json"
    path.write_text(json.dumps(_registry_doc([entry])), encoding="utf-8")
    with pytest.raises(caps.CapabilityRegistryError) as excinfo:
        caps.load_capability_registry(path)
    assert excinfo.value.code == caps.CAPABILITY_REGISTRY_INVALID
    assert any("status" in reason for reason in excinfo.value.reasons)


def test_unknown_entry_id_raises(tmp_path):
    registry, _ = _smoked_stack(
        tmp_path,
        engine_modes=["SINGLE_1D"],
        adapter_modes=["SINGLE_1D"],
        deployment_modes=["SINGLE_1D"],
        engine_kinds=["B"],
        adapter_kinds=["B"],
        deployment_kinds=["B"],
        engine_max=1,
        adapter_max=1,
        deployment_max=1,
    )
    with pytest.raises(caps.CapabilityRegistryError) as excinfo:
        caps.effective_capability("nope", "adp", "dep", registry=registry)
    assert excinfo.value.code == caps.CAPABILITY_REGISTRY_ENTRY_MISSING


def test_layer_mismatch_raises(tmp_path):
    registry, _ = _smoked_stack(
        tmp_path,
        engine_modes=["SINGLE_1D"],
        adapter_modes=["SINGLE_1D"],
        deployment_modes=["SINGLE_1D"],
        engine_kinds=["B"],
        adapter_kinds=["B"],
        deployment_kinds=["B"],
        engine_max=1,
        adapter_max=1,
        deployment_max=1,
    )
    with pytest.raises(caps.CapabilityRegistryError) as excinfo:
        caps.effective_capability("adp", "adp", "dep", registry=registry)
    assert excinfo.value.code == caps.CAPABILITY_LAYER_MISMATCH


def test_assert_rejects_kind_and_arity_violations(tmp_path):
    _, effective = _smoked_stack(
        tmp_path,
        engine_modes=["SINGLE_1D"],
        adapter_modes=["SINGLE_1D"],
        deployment_modes=["SINGLE_1D"],
        engine_kinds=["B"],
        adapter_kinds=["B"],
        deployment_kinds=["B"],
        engine_max=3,
        adapter_max=3,
        deployment_max=3,
        engine_receipts=_pass(("SINGLE_1D",)),
        adapter_receipts=_pass(("SINGLE_1D",)),
        deployment_receipts=_pass(("SINGLE_1D",)),
    )
    caps.assert_mode_supported(effective, "SINGLE_1D", ("B",), 1)

    with pytest.raises(caps.BackendCapabilityError) as excinfo:
        caps.assert_mode_supported(effective, "SINGLE_1D", ("A",), 1)
    assert any("coordinate kinds" in reason for reason in excinfo.value.reasons)

    with pytest.raises(caps.BackendCapabilityError) as excinfo:
        caps.assert_mode_supported(effective, "SINGLE_1D", ("B", "B"), 2)
    assert any("exactly one coordinate" in reason for reason in excinfo.value.reasons)

    with pytest.raises(caps.BackendCapabilityError):
        caps.assert_mode_supported(effective, "GRID_2D", ("B",), 1)


def test_assert_rejects_coupled_with_single_coordinate(tmp_path):
    _, effective = _smoked_stack(
        tmp_path,
        engine_modes=["COUPLED_1D"],
        adapter_modes=["COUPLED_1D"],
        deployment_modes=["COUPLED_1D"],
        engine_kinds=["B", "A"],
        adapter_kinds=["B", "A"],
        deployment_kinds=["B", "A"],
        engine_max=3,
        adapter_max=3,
        deployment_max=3,
        engine_receipts=_pass(("COUPLED_1D",)),
        adapter_receipts=_pass(("COUPLED_1D",)),
        deployment_receipts=_pass(("COUPLED_1D",)),
    )
    caps.assert_mode_supported(effective, "COUPLED_1D", ("B", "A"), 2)

    with pytest.raises(caps.BackendCapabilityError) as excinfo:
        caps.assert_mode_supported(effective, "COUPLED_1D", ("B",), 1)
    assert any("requires 2..3" in reason for reason in excinfo.value.reasons)


def test_assert_rejects_path_mode_with_scan_coordinates(tmp_path):
    engine = _entry_doc(
        "eng", "engine", "xtb", "acp",
        modes=["PATH_NEB"], kinds=[], max_coords=0,
        engine_version="6.7.1", adapter_version="pes2ts_xtb_path_request_v1",
        receipts=[_receipt(modes=("PATH_NEB",), engine_version="6.7.1")],
    )
    adapter = _entry_doc(
        "adp", "adapter", "xtb", "acp",
        modes=["PATH_NEB"], kinds=[], max_coords=0,
        engine_version="6.7.1", adapter_version="pes2ts_xtb_path_request_v1",
        receipts=[_receipt(modes=("PATH_NEB",), engine_version="6.7.1")],
    )
    deployment = _entry_doc(
        "dep", "deployment", "xtb", "acp",
        modes=["PATH_NEB"], kinds=[], max_coords=0,
        engine_version="6.7.1", adapter_version="pes2ts_xtb_path_request_v1",
        receipts=[_receipt(modes=("PATH_NEB",), engine_version="6.7.1")],
    )
    registry = _load(tmp_path, _registry_doc([engine, adapter, deployment]))
    effective = caps.effective_capability("eng", "adp", "dep", registry=registry)
    assert effective.enabled_modes == frozenset({"PATH_NEB"})
    assert effective.max_scan_coordinates == 0

    caps.assert_mode_supported(effective, "PATH_NEB", (), 0)
    with pytest.raises(caps.BackendCapabilityError) as excinfo:
        caps.assert_mode_supported(effective, "PATH_NEB", ("B",), 1)
    assert any("no native scan coordinates" in reason for reason in excinfo.value.reasons)


def test_method_element_coverage_intersects_across_all_layers(tmp_path):
    _, effective = _smoked_stack(
        tmp_path,
        engine_modes=["SINGLE_1D"],
        adapter_modes=["SINGLE_1D"],
        deployment_modes=["SINGLE_1D"],
        engine_kinds=["B"],
        adapter_kinds=["B"],
        deployment_kinds=["B"],
        engine_max=1,
        adapter_max=1,
        deployment_max=1,
        engine_coverage={
            "B3LYP-D3": ORGANIC,
            "GFN2-xTB": ORGANIC,
            "X2C": ["H", "C"],
        },
        adapter_coverage={
            "B3LYP-D3": ["H", "C", "N", "O"],
            "GFN2-xTB": ORGANIC,
        },
        deployment_coverage={
            "B3LYP-D3": ["H", "C", "N", "O", "F"],
            "GFN2-xTB": ORGANIC,
        },
    )
    assert effective.method_element_coverage["B3LYP-D3"] == frozenset(
        {"H", "C", "N", "O"}
    )
    assert effective.method_element_coverage["GFN2-xTB"] == frozenset(ORGANIC)
    assert "X2C" not in effective.method_element_coverage


def test_constraint_support_intersects_per_field(tmp_path):
    _, effective = _smoked_stack(
        tmp_path,
        engine_modes=["SINGLE_1D"],
        adapter_modes=["SINGLE_1D"],
        deployment_modes=["SINGLE_1D"],
        engine_kinds=["B"],
        adapter_kinds=["B"],
        deployment_kinds=["B"],
        engine_max=1,
        adapter_max=1,
        deployment_max=1,
        engine_constraints={
            "native_scan": True,
            "per_point_constraints": False,
            "simul_scan": True,
        },
        adapter_constraints={
            "native_scan": True,
            "per_point_constraints": False,
            "simul_scan": False,
        },
        deployment_constraints={
            "native_scan": True,
            "per_point_constraints": True,
            "simul_scan": True,
        },
    )
    assert effective.constraint_support == caps.ConstraintSupport(
        native_scan=True, per_point_constraints=False, simul_scan=False
    )


def test_point_limits_intersect_across_layers(tmp_path):
    _, effective = _smoked_stack(
        tmp_path,
        engine_modes=["SINGLE_1D"],
        adapter_modes=["SINGLE_1D"],
        deployment_modes=["SINGLE_1D"],
        engine_kinds=["B"],
        adapter_kinds=["B"],
        deployment_kinds=["B"],
        engine_max=1,
        adapter_max=1,
        deployment_max=1,
        engine_limits=(9, 101),
        adapter_limits=(30, 60),
        deployment_limits=(9, 101),
    )
    assert effective.point_limits == caps.PointLimits(baseline=30, max=60)

    _, inverted = _smoked_stack(
        tmp_path,
        engine_modes=["SINGLE_1D"],
        adapter_modes=["SINGLE_1D"],
        deployment_modes=["SINGLE_1D"],
        engine_kinds=["B"],
        adapter_kinds=["B"],
        deployment_kinds=["B"],
        engine_max=1,
        adapter_max=1,
        deployment_max=1,
        engine_limits=(9, 30),
        adapter_limits=(40, 101),
        deployment_limits=(9, 101),
    )
    assert inverted.point_limits == caps.PointLimits(baseline=40, max=30)


def test_effective_capability_accepts_inline_entries():
    engine = caps.CapabilityEntry.from_doc(
        _entry_doc(
            "eng", "engine", "orca", "acp",
            modes=["SINGLE_1D"], kinds=["B"], max_coords=1,
            receipts=_pass(("SINGLE_1D",)),
        )
    )
    adapter = caps.CapabilityEntry.from_doc(
        _entry_doc(
            "adp", "adapter", "orca", "acp",
            modes=["SINGLE_1D"], kinds=["B"], max_coords=1,
            receipts=_pass(("SINGLE_1D",)),
        )
    )
    deployment = caps.CapabilityEntry.from_doc(
        _entry_doc(
            "dep", "deployment", "orca", "acp",
            modes=["SINGLE_1D"], kinds=["B"], max_coords=1,
            receipts=_pass(("SINGLE_1D",)),
        )
    )
    effective = caps.effective_capability(engine, adapter, deployment)
    assert effective.enabled_modes == frozenset({"SINGLE_1D"})
    caps.assert_mode_supported(effective, "SINGLE_1D", ("B",), 1)


def test_capability_docs_carry_no_forbidden_truth_keys():
    forbidden = {key.lower() for key in FORBIDDEN_TRUTH_KEYS | FORBIDDEN_EXPORT_KEYS}

    def walk(obj) -> None:
        if isinstance(obj, dict):
            for key, value in obj.items():
                assert str(key).lower() not in forbidden
                walk(value)
        elif isinstance(obj, list):
            for item in obj:
                walk(item)

    registry = caps.load_capability_registry()
    walk(registry.to_doc())
    effective = caps.effective_capability(
        "orca-acp-engine", "acp-adapter-v0", "deployment-baseline", registry=registry
    )
    walk(effective.to_doc())
