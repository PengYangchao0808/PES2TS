"""Campaign output-root assertion tests (mission: campaign_root, blocker 7).

Covers ``scripts/agent_ops/supervisor/campaign_root.py``: runners must write
only under ``outputs/pesn/30h_campaign/**``; the sealed legacy prefixes
(``outputs/pesn/v5_1``, ``outputs/pesn/v5_1_r1``) and anything else outside
the campaign root refuse fail-closed (``parallel_config.driver_campaign_root_
rule``; integration into drivers lands at E3).
"""

from __future__ import annotations

import json

import pytest

from scripts.agent_ops.supervisor import campaign_root as cr


def test_campaign_root_accepts_campaign_paths():
    assert cr.assert_campaign_root("outputs/pesn/30h_campaign") == "outputs/pesn/30h_campaign"
    assert (
        cr.assert_campaign_root("outputs/pesn/30h_campaign/runs/r1")
        == "outputs/pesn/30h_campaign/runs/r1"
    )
    assert cr.assert_campaign_root("./outputs/pesn/30h_campaign") == "outputs/pesn/30h_campaign"
    # absolute paths anchor at outputs/pesn
    assert (
        cr.assert_campaign_root("/mnt/data/somewhere/outputs/pesn/30h_campaign/x")
        == "outputs/pesn/30h_campaign/x"
    )


@pytest.mark.parametrize(
    "sealed",
    [
        "outputs/pesn/v5_1",
        "outputs/pesn/v5_1/",
        "outputs/pesn/v5_1/runs/old.pt",
        "outputs/pesn/v5_1_r1",
        "outputs/pesn/v5_1_r1/checkpoints/x.pt",
    ],
)
def test_sealed_prefixes_refused(sealed):
    with pytest.raises(cr.CampaignRootError) as excinfo:
        cr.assert_campaign_root(sealed)
    assert "SEALED_ROOT_REFUSED" in str(excinfo.value)


@pytest.mark.parametrize(
    "outside",
    [
        "outputs/pesn/other_campaign",
        "outputs/pesn",
        "outputs",
        "data/tables",
        "pes2ts_core/pesn",
    ],
)
def test_outside_campaign_root_refused(outside):
    with pytest.raises(cr.CampaignRootError) as excinfo:
        cr.assert_campaign_root(outside)
    assert "OUTSIDE_CAMPAIGN_ROOT" in str(excinfo.value)


@pytest.mark.parametrize("garbage", [None, 123, "", "   "])
def test_invalid_roots_refused(garbage):
    with pytest.raises(cr.CampaignRootError) as excinfo:
        cr.assert_campaign_root(garbage)
    assert "INVALID_ROOT" in str(excinfo.value)


def test_parent_traversal_refused_before_normalization():
    """``..`` must refuse outright -- never normalize into a sealed root."""
    with pytest.raises(cr.CampaignRootError) as excinfo:
        cr.assert_campaign_root("outputs/pesn/30h_campaign/../v5_1")
    assert "INVALID_ROOT" in str(excinfo.value)


def test_relative_path_with_campaign_root_base():
    ok = cr.assert_campaign_root("runs/x", campaign_root="outputs/pesn/30h_campaign")
    assert ok == "outputs/pesn/30h_campaign/runs/x"
    # a driver whose default base IS the sealed dir must still refuse
    with pytest.raises(cr.CampaignRootError) as excinfo:
        cr.assert_campaign_root("runs/x", campaign_root="outputs/pesn/v5_1")
    assert "SEALED_ROOT_REFUSED" in str(excinfo.value)


def test_absolute_path_without_outputs_component_refused():
    with pytest.raises(cr.CampaignRootError) as excinfo:
        cr.assert_campaign_root("/tmp/training/output")
    assert "OUTSIDE_CAMPAIGN_ROOT" in str(excinfo.value)


def test_cli_exit_codes(capsys):
    assert cr.main(["outputs/pesn/30h_campaign"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload == {"ok": True, "root": "outputs/pesn/30h_campaign"}

    assert cr.main(["outputs/pesn/v5_1_r1"]) == 1
    assert "SEALED_ROOT_REFUSED" in capsys.readouterr().err

    assert cr.main([]) == 2  # no path -> usage
