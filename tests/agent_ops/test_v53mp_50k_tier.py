"""V5.3-MP 50k tier support in the data-gate scripts (收尾方案 §4 阶段 C).

CPU-only, tiny synthetic fixtures — no data build, no GPU, no long I/O:

* ``freeze_cohort``: pool-candidate target parameterized (default 26,000
  unchanged); a LARGER target STRICT PREFIX-EXTENDS the ranked list, so
  the nested-prefix tier property (2k ⊂ 5k ⊂ 20k ⊂ 50k) survives across
  builds by construction;
* ``finalize.freeze_pool``: a tier entry is emitted only when its size
  fits the frozen pool — the default (pool_target 20,000) still writes
  exactly {2k, 5k, 20k}, while ``--pool-target 50000`` emits all four
  with the nested-prefix property; pool targets below the 2k floor are a
  typed refusal; the SHARED dev pool is identical for every tier;
* certificates: startup / loader-v2 pool-id selection parameterized by
  ``--tier`` (dict-shape and list-ranked manifest shapes; unknown tiers
  refused); defaults stay "20k".
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.agent_ops.data_gate import r3_v53mp_finalize as fin  # noqa: E402
from scripts.agent_ops.data_gate import (  # noqa: E402
    r3_v53mp_freeze_cohort as fc,
)
from scripts.agent_ops.data_gate import (  # noqa: E402
    r3_v53mp_loader_v2_certificate as l2c,
)
from scripts.agent_ops.data_gate import (  # noqa: E402
    r3_v53mp_startup_certificate as scc,
)

#: injected tier sizes for tiny synthetic pools — SAME nested-prefix
#: layout as the real TIER_SIZES, scaled down (a real default finalize
#: would need a 20k-curve fixture; the emission RULE is what's under test).
TINY_TIERS = {"2k": 4, "5k": 8, "20k": 16, "50k": 32}


# ---------------------------------------------------------------------------
# synthetic S0-A / cohort / curve_quality fixture
# ---------------------------------------------------------------------------


def _write_s0a(s0a_dir: Path, n_fit: int, n_dev: int, n_cal: int) -> None:
    """Minimal eligible_reactions.parquet + qualification json."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    s0a_dir.mkdir(parents=True, exist_ok=True)
    reaction_ids, canonical_groups, subsets = [], [], []
    for subset, n, prefix in (
            ("train_fit", n_fit, "fit"), ("train_dev", n_dev, "dev"),
            ("train_calibration", n_cal, "cal")):
        for i in range(n):
            gid = "{}{:05d}".format(prefix, i)
            reaction_ids.append(gid)
            canonical_groups.append(gid)
            subsets.append(subset)
    table = pa.table({
        "reaction_id": pa.array(reaction_ids, pa.string()),
        "canonical_group": pa.array(canonical_groups, pa.string()),
        "subset": pa.array(subsets, pa.string()),
        "family_labels": pa.array(["f"] * len(reaction_ids), pa.string()),
        "edit_family_id": pa.array([0] * len(reaction_ids), pa.int64()),
    })
    pq.write_table(table, s0a_dir / "eligible_reactions.parquet")
    (s0a_dir / "metadata_qualification.json").write_text(
        "{}\n", encoding="utf-8")


def _write_g3p1_split(repo: Path) -> None:
    split = repo / "outputs/pesn/g3_p1/splits/pesn_irc_split.json"
    split.parent.mkdir(parents=True, exist_ok=True)
    split.write_text("{}\n", encoding="utf-8")


def _freeze_pool_env(tmp_path: Path, *, n_pool: int = 64, n_dev: int = 8,
                     n_cal: int = 4):
    """repo + cohort + data dir where EVERY candidate built ok."""
    repo = tmp_path
    s0a = repo / "s0a"
    _write_s0a(s0a, n_fit=n_pool, n_dev=n_dev, n_cal=n_cal)
    _write_g3p1_split(repo)
    cohort = fc.freeze_cohort(repo, s0a, repo / "cohort",
                              pool_candidate_target=n_pool)
    data_dir = repo / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    all_ids = (cohort["pool_candidates"] + cohort["dev"]
               + cohort["confirm"])
    with open(data_dir / "curve_quality.csv", "w", encoding="utf-8",
              newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["curve_id", "quality_status"])
        for cid in all_ids:
            w.writerow([cid, "ok"])
    return repo, data_dir, repo / "cohort/v53mp_cohort_v1.json", s0a


# ---------------------------------------------------------------------------
# freeze_cohort: target parameterization + prefix extension
# ---------------------------------------------------------------------------


def test_freeze_cohort_default_target_unchanged(tmp_path: Path) -> None:
    s0a = tmp_path / "s0a"
    _write_s0a(s0a, n_fit=300, n_dev=20, n_cal=10)
    _write_g3p1_split(tmp_path)
    cohort = fc.freeze_cohort(tmp_path, s0a, tmp_path / "cohort")
    # default VALUES unchanged (20k build cohort)
    assert fc.POOL_CANDIDATE_TARGET == 26_000
    assert fc.DEV_TARGET == 600
    assert fc.CONFIRM_TARGET == 2_000
    assert cohort["targets"]["pool_candidates"] == 26_000
    # only 300 fit groups exist -> the slice yields all of them
    assert cohort["counts"]["pool_candidates"] == 300
    assert cohort["counts"]["dev"] == 20
    assert cohort["counts"]["confirm"] == 10
    # legacy note verbatim at the default target
    assert cohort["notes"][1] == (
        "pool candidates carry build margin over the 20,000 target; the "
        "frozen 20k pool is fixed AFTER the build (first 20,000 qualified "
        "groups in this rank order)")


def test_freeze_cohort_65k_target_prefix_extends_26k(tmp_path: Path) -> None:
    """A larger candidate target is a strict prefix EXTENSION of the ranked
    list — the nested-prefix property survives across builds."""
    s0a = tmp_path / "s0a"
    _write_s0a(s0a, n_fit=400, n_dev=20, n_cal=10)
    _write_g3p1_split(tmp_path)
    small = fc.freeze_cohort(tmp_path, s0a, tmp_path / "c_small",
                             pool_candidate_target=26_000)
    big = fc.freeze_cohort(tmp_path, s0a, tmp_path / "c_big",
                           pool_candidate_target=65_000)
    # 400 available: small yields all 400 (below its 26k target), big too
    assert small["pool_candidates"] == big["pool_candidates"]
    assert big["targets"]["pool_candidates"] == 65_000
    # generic note at a non-default target
    assert "first 65,000 qualified groups" in big["notes"][1]
    # prefix-extension with a real slice difference
    s2 = fc.freeze_cohort(tmp_path, s0a, tmp_path / "c_s2",
                          pool_candidate_target=100)
    b2 = fc.freeze_cohort(tmp_path, s0a, tmp_path / "c_b2",
                          pool_candidate_target=250)
    assert len(b2["pool_candidates"]) == 250
    assert b2["pool_candidates"][:100] == s2["pool_candidates"]
    # ranking is identical regardless of target (pure per-group function)
    assert b2["canonical_groups"]["pool"][:100] == \
        s2["canonical_groups"]["pool"]


def test_freeze_cohort_rejects_bad_target(tmp_path: Path) -> None:
    s0a = tmp_path / "s0a"
    _write_s0a(s0a, n_fit=5, n_dev=2, n_cal=2)
    _write_g3p1_split(tmp_path)
    with pytest.raises(ValueError):
        fc.freeze_cohort(tmp_path, s0a, tmp_path / "c", pool_candidate_target=0)
    # CLI: argparse refuses before touching anything
    with pytest.raises(SystemExit) as exc:
        fc.main(["--repo-root", str(tmp_path), "--s0a", str(s0a),
                 "--out", str(tmp_path / "c"), "--pool-candidate-target", "0"])
    assert exc.value.code == 2


# ---------------------------------------------------------------------------
# finalize.freeze_pool: tier emission rule + nested prefixes + shared dev
# ---------------------------------------------------------------------------


def test_finalize_tier_sizes_constant_includes_50k() -> None:
    assert fin.POOL_TARGET == 20_000
    assert fin.TIER_SIZES == {"2k": 2_000, "5k": 5_000,
                              "20k": 20_000, "50k": 50_000}
    # the emission rule at the DEFAULT pool target: 50k does not fit a
    # 20,000-curve pool, so the default manifest keeps {2k, 5k, 20k}
    emitted = {name for name, size in fin.TIER_SIZES.items()
               if size <= fin.POOL_TARGET}
    assert emitted == {"2k", "5k", "20k"}


def test_freeze_pool_default_emits_three_tiers(tmp_path: Path) -> None:
    """pool_target == the 20k tier (scaled): exactly {2k, 5k, 20k}; the
    rules chain is the pre-50k string verbatim; no 50k entry."""
    repo, data_dir, cohort_path, s0a = _freeze_pool_env(tmp_path, n_pool=64)
    out = fin.freeze_pool(repo, data_dir, cohort_path, s0a,
                          pool_target=16, tier_sizes=TINY_TIERS)
    doc = out["split_doc"]
    pool = doc["subsets"]["pool"]
    assert set(pool) == {"2k", "5k", "20k"}       # 50k (32) does not fit 16
    assert doc["rules"][0] == (
        "tiers are nested prefixes of the frozen pool in cohort rank "
        "order (2k ⊂ 5k ⊂ 20k)")
    # nested prefixes over the SAME frozen pool
    assert pool["2k"] == pool["5k"][:4] == pool["20k"][:4]
    assert pool["5k"] == pool["20k"][:8]
    assert doc["counts"]["pool_20k"] == 16
    assert doc["counts"]["pool_frozen"] == 16
    assert out["n_pool"] == 16


def test_freeze_pool_50k_target_emits_all_four_tiers(tmp_path: Path) -> None:
    repo, data_dir, cohort_path, s0a = _freeze_pool_env(tmp_path, n_pool=64)
    out = fin.freeze_pool(repo, data_dir, cohort_path, s0a,
                          pool_target=32, tier_sizes=TINY_TIERS)
    doc = out["split_doc"]
    pool = doc["subsets"]["pool"]
    assert set(pool) == {"2k", "5k", "20k", "50k"}
    assert doc["rules"][0] == (
        "tiers are nested prefixes of the frozen pool in cohort rank "
        "order (2k ⊂ 5k ⊂ 20k ⊂ 50k)")
    # nested prefixes: 2k ⊂ 5k ⊂ 20k ⊂ 50k over the same frozen pool
    assert pool["2k"] == pool["50k"][:4]
    assert pool["5k"] == pool["50k"][:8]
    assert pool["20k"] == pool["50k"][:16]
    assert len(pool["50k"]) == 32
    assert doc["counts"]["pool_frozen"] == 32
    # dev subset is SHARED (identical regardless of the pool target)
    out_small = fin.freeze_pool(repo, data_dir, cohort_path, s0a,
                                pool_target=16, tier_sizes=TINY_TIERS)
    assert doc["subsets"]["dev"] == out_small["split_doc"]["subsets"]["dev"]
    assert len(doc["subsets"]["dev"]) == 8
    # confirm ids NEVER enter any tier (list-only, fold 4)
    confirm = set(doc["subsets"]["confirm"])
    assert confirm and not (confirm & set(pool["50k"]))
    assert all(doc["curve_folds"][cid] == 4 for cid in confirm)


def test_freeze_pool_rejects_target_below_2k_floor(tmp_path: Path) -> None:
    repo, data_dir, cohort_path, s0a = _freeze_pool_env(tmp_path, n_pool=64)
    with pytest.raises(ValueError):
        fin.freeze_pool(repo, data_dir, cohort_path, s0a,
                        pool_target=2, tier_sizes=TINY_TIERS)
    # real default sizes: the floor is the 2k tier
    with pytest.raises(ValueError):
        fin.freeze_pool(repo, data_dir, cohort_path, s0a, pool_target=1999)


def test_finalize_cli_pool_target_bad_value_exits_2(tmp_path: Path,
                                                    capsys) -> None:
    repo, data_dir, cohort_path, s0a = _freeze_pool_env(tmp_path, n_pool=64)
    rc = fin.main([
        "--repo-root", str(repo), "--data", str(data_dir),
        "--cohort", str(cohort_path), "--s0a", str(s0a),
        "--pool-target", "1", "--skip-labels", "--skip-resource",
    ])
    assert rc == 2
    assert "FATAL" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# certificates: --tier parameterization (defaults unchanged)
# ---------------------------------------------------------------------------


def test_certificate_tier_constants_match_driver() -> None:
    from scripts.campaigns.r3_v53mp_run import TIER_LIMITS

    assert scc.DEFAULT_TIER == "20k"
    assert l2c.DEFAULT_TIER == "20k"
    assert scc.TIER_LIMITS == TIER_LIMITS
    assert l2c.TIER_LIMITS == TIER_LIMITS
    assert "50k" in scc.TIER_LIMITS and scc.TIER_LIMITS["50k"] == 50_000


def test_startup_cert_select_pool_ids_tier_parameterized() -> None:
    doc = {"subsets": {"pool": {"20k": ["a", "b"], "50k": ["a", "b", "c"]}}}
    assert scc._select_pool_ids(doc) == ["a", "b"]          # default 20k
    assert scc._select_pool_ids(doc, tier="50k") == ["a", "b", "c"]
    # list-ranked shape: the tier limit decides inclusion
    ranked = {"subsets": {"pool": [
        {"curve_id": "a", "rank": 10},
        {"curve_id": "b", "rank": 30000}]}}
    assert scc._select_pool_ids(ranked, tier="50k") == ["a", "b"]
    assert scc._select_pool_ids(ranked, tier="20k") == ["a"]
    # unknown tier: typed refusal
    with pytest.raises(SystemExit) as exc:
        scc._select_pool_ids(doc, tier="60k")
    assert "unknown tier" in str(exc.value)


def test_loader_v2_cert_select_pool_ids_tier_parameterized() -> None:
    doc = {"subsets": {"pool": {"20k": ["a", "b"], "50k": ["a", "b", "c"]}}}
    assert l2c._select_pool_curve_ids(doc) == ["a", "b"]    # default 20k
    assert l2c._select_pool_curve_ids(doc, tier="50k") == ["a", "b", "c"]
    ranked = {"subsets": {"pool": [
        {"curve_id": "a", "rank": 10},
        {"curve_id": "b", "rank": 30000}]}}
    assert l2c._select_pool_curve_ids(ranked, tier="50k") == ["a", "b"]
    assert l2c._select_pool_curve_ids(ranked, tier="20k") == ["a"]
    with pytest.raises(RuntimeError):
        l2c._select_pool_curve_ids(doc, tier="60k")
    # a missing 50k tier in a mapping-shaped manifest still fails closed
    only20k = {"subsets": {"pool": {"20k": ["a"]}}}
    with pytest.raises(RuntimeError):
        l2c._select_pool_curve_ids(only20k, tier="50k")


def test_certificate_cli_rejects_unknown_tier() -> None:
    for main in (scc.main, l2c.main):
        with pytest.raises(SystemExit) as exc:
            main(["--tier", "60k"])
        assert exc.value.code == 2
