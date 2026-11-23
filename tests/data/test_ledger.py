"""Tests for the retired-campaign ledger reader (``pes2ts_data.ingest.ledger``).

All fixtures are synthetic ``tmp_path`` trees; the real archive data is never
touched.
"""

from __future__ import annotations

import hashlib
import json

import pytest

from pes2ts_data.ingest.ledger import (
    CAMPAIGN_INDEX_SCHEMA,
    RETIRED_LEDGER_SCHEMA,
    CampaignEntry,
    LedgerError,
    RetiredEntry,
    VerifyStatus,
    load_campaign_index,
    load_ledger,
    resolve_artifact_path,
    sha256_file,
    summarize,
    verify_entry,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _sha(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture()
def ledger_root(tmp_path):
    """Build a synthetic repo-like root with an index, a ledger and artifacts."""
    root = tmp_path / "repo"
    archives = root / "data" / "archives_summary"
    campaigns = archives / "campaigns"
    raw = root / "data" / "raw"
    campaigns.mkdir(parents=True)
    raw.mkdir(parents=True)

    (campaigns / "camp_a.json").write_text("{}", encoding="utf-8")
    ok_file = raw / "pack_ok.tar.zst"
    ok_file.write_bytes(b"ok-payload")
    bad_file = raw / "pack_bad.tar.zst"
    bad_file.write_bytes(b"bad-payload")
    logs_dir = raw / "logs"
    logs_dir.mkdir()
    (logs_dir / "one.log").write_text("log-line", encoding="utf-8")

    campaign_index = {
        "schema": CAMPAIGN_INDEX_SCHEMA,
        "generated_at": "2026-11-23T08:07:42.392272+00:00",
        "archives_dir": "data\\archives_summary",
        "campaigns": [
            {
                "campaign_id": "camp_a",
                "bytes": 10,
                "files": 1,
                "tombstone": "data\\archives_summary\\campaigns\\camp_a.json",
            },
            {
                "campaign_id": "camp_b",
                "bytes": 20,
                "files": 2,
                "tombstone": "data\\archives_summary\\campaigns\\camp_b.json",
            },
        ],
        "total_bytes": 30,
    }
    retired = {
        "schema": RETIRED_LEDGER_SCHEMA,
        "generated_at": "2026-11-23T09:25:06.863791+00:00",
        "policy": "keep-truth-drop-reproducible",
        "retired": [
            {
                "kind": "xtb_archive",
                "path": "data/raw/pack_ok.tar.zst",
                "bytes": 100,
                "sha256": _sha(ok_file),
                "verified_before_delete": True,
                "deleted_at": "2026-11-23T08:05:53Z",
                "reason": "reproducible-from-zenodo",
            },
            {
                "kind": "xtb_archive",
                "path": "data/raw/pack_bad.tar.zst",
                "bytes": 200,
                "sha256": "0" * 64,
            },
            {
                "kind": "xtb_archive",
                "path": "data/raw/gone.tar.zst",
                "bytes": 300,
                "sha256": "1" * 64,
            },
            {"kind": "xtb_logs", "path": "data/raw/logs", "files": 1, "gib": 1.5},
            {
                "kind": "campaign_dir",
                "campaign_id": "camp_a",
                "bytes": 10,
                "tombstone": "data\\archives_summary\\campaigns\\camp_a.json",
            },
            {"kind": "cost_only", "bytes": 5},
        ],
    }
    (archives / "campaigns_index.json").write_text(
        json.dumps(campaign_index), encoding="utf-8")
    (archives / "retired_raw_packs.json").write_text(
        json.dumps(retired), encoding="utf-8")
    return root


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def test_load_campaign_index(ledger_root):
    index = load_campaign_index(ledger_root)
    assert index.schema == CAMPAIGN_INDEX_SCHEMA
    assert index.total_bytes == 30
    assert [entry.campaign_id for entry in index.campaigns] == ["camp_a", "camp_b"]
    by_id = {entry.campaign_id: entry for entry in index.campaigns}
    assert by_id["camp_a"].tombstone_exists is True
    assert by_id["camp_b"].tombstone_exists is False
    assert by_id["camp_a"].bytes == 10 and by_id["camp_a"].files == 1
    assert index.source_path.name == "campaigns_index.json"


def test_load_campaign_index_accepts_flexible_roots(ledger_root):
    for root in (ledger_root, ledger_root / "data",
                 ledger_root / "data" / "archives_summary"):
        assert len(load_campaign_index(root).campaigns) == 2
        assert len(load_ledger(root).entries) == 6


def test_load_ledger(ledger_root):
    ledger = load_ledger(ledger_root)
    assert ledger.schema == RETIRED_LEDGER_SCHEMA
    assert ledger.policy == "keep-truth-drop-reproducible"
    assert len(ledger.entries) == 6
    first = ledger.entries[0]
    assert first.kind == "xtb_archive"
    assert first.artifact == "data/raw/pack_ok.tar.zst"
    assert first.verified_before_delete is True
    assert ledger.entries[3].gib == 1.5
    assert ledger.entries[4].campaign_id == "camp_a"
    assert ledger.entries[5].artifact is None


# ---------------------------------------------------------------------------
# Verification (never raises on missing artifacts)
# ---------------------------------------------------------------------------

def test_verify_entry_status_vocabulary(ledger_root):
    ledger = load_ledger(ledger_root)
    statuses = [verify_entry(ledger_root, entry).status for entry in ledger.entries]
    assert statuses == [
        VerifyStatus.OK,
        VerifyStatus.MISMATCH,
        VerifyStatus.MISSING,
        VerifyStatus.UNVERIFIABLE,   # directory, no sha in ledger
        VerifyStatus.UNVERIFIABLE,   # tombstone exists but no sha
        VerifyStatus.UNVERIFIABLE,   # cost-only row, no artifact path
    ]


def test_verify_entry_never_raises_on_missing(ledger_root):
    result = verify_entry(ledger_root, RetiredEntry(kind="ghost",
                                                    path="data/raw/nope.bin"))
    assert result.status is VerifyStatus.MISSING
    assert result.target is not None and not result.target.exists()

    campaign = CampaignEntry(campaign_id="ghost",
                             tombstone="data/archives_summary/campaigns/none.json")
    assert verify_entry(ledger_root, campaign).status is VerifyStatus.MISSING


def test_verify_entry_accepts_existing_ok_hash(ledger_root, tmp_path):
    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(b"ledger-payload")
    entry = RetiredEntry(kind="xtb_archive", path=str(artifact),
                         sha256=sha256_file(artifact))
    result = verify_entry(ledger_root, entry)
    assert result.status is VerifyStatus.OK
    assert result.actual_sha256 == result.expected_sha256


def test_verify_entry_rejects_non_entry():
    with pytest.raises(LedgerError) as excinfo:
        verify_entry(".", {"kind": "xtb_archive"})
    assert excinfo.value.code == "LEDGER_BAD_ARGUMENT"


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def test_summarize_totals(ledger_root):
    report = summarize(ledger_root)
    assert report["campaigns"]["count"] == 2
    assert report["campaigns"]["total_bytes"] == 30
    assert report["campaigns"]["tombstones_present"] == 1
    assert report["campaigns"]["tombstones_missing"] == 1
    assert report["retired"]["entries"] == 6
    assert report["retired"]["known_bytes"] == 615
    assert report["retired"]["known_gib"] == 1.5
    assert report["retired"]["by_kind"] == {
        "campaign_dir": 1, "cost_only": 1, "xtb_archive": 3, "xtb_logs": 1}
    assert report["verification"] == {
        "OK": 1, "MISSING": 1, "MISMATCH": 1, "UNVERIFIABLE": 3}
    assert any(problem["status"] == "MISMATCH" for problem in report["problems"])
    assert any(problem["kind"] == "campaign_tombstone"
               for problem in report["problems"])


# ---------------------------------------------------------------------------
# Typed errors
# ---------------------------------------------------------------------------

def test_ledger_typed_errors(ledger_root, tmp_path):
    with pytest.raises(LedgerError) as excinfo:
        load_campaign_index(tmp_path / "no-such-root")
    assert excinfo.value.code == "LEDGER_BAD_ARGUMENT"

    with pytest.raises(LedgerError) as excinfo:
        load_ledger(tmp_path)  # empty dir: no ledger files
    assert excinfo.value.code == "LEDGER_FILE_MISSING"

    broken = tmp_path / "broken" / "data" / "archives_summary"
    broken.mkdir(parents=True)
    (broken / "campaigns_index.json").write_text("{not-json", encoding="utf-8")
    with pytest.raises(LedgerError) as excinfo:
        load_campaign_index(broken)
    assert excinfo.value.code == "LEDGER_PARSE_ERROR"

    (broken / "campaigns_index.json").write_text(
        json.dumps({"schema": "wrong_schema", "campaigns": []}), encoding="utf-8")
    with pytest.raises(LedgerError) as excinfo:
        load_campaign_index(broken)
    assert excinfo.value.code == "LEDGER_SCHEMA_UNSUPPORTED"

    (broken / "campaigns_index.json").write_text(
        json.dumps({"schema": CAMPAIGN_INDEX_SCHEMA}), encoding="utf-8")
    with pytest.raises(LedgerError) as excinfo:
        load_campaign_index(broken)
    assert excinfo.value.code == "LEDGER_MALFORMED"


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------

def test_resolve_artifact_path_handles_windows_and_absolute_paths(tmp_path):
    assert resolve_artifact_path(tmp_path, "data/raw/x.bin") == tmp_path / "data" / "raw" / "x.bin"
    assert resolve_artifact_path(
        tmp_path, "data\\raw\\x.bin") == tmp_path / "data" / "raw" / "x.bin"
    absolute = tmp_path / "abs.bin"
    assert resolve_artifact_path(tmp_path, str(absolute)) == absolute
    assert resolve_artifact_path(tmp_path, None) is None


def test_sha256_file_matches_hashlib(tmp_path):
    artifact = tmp_path / "x.bin"
    artifact.write_bytes(b"pes2ts")
    assert sha256_file(artifact) == hashlib.sha256(b"pes2ts").hexdigest()
