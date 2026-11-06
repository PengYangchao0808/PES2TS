"""V5.4.3 WP3 — S0-B probe / S0-C basis coverage CPU certificate.

Plan §7.1 S0-B (:614–630) and S0-C (:632–642), plus §2 F7 / 附录 E #4 (:1659)
("覆盖率" alone inflates → condition number + effective rank + teacher
coefficients + leave-one-out + probe must be reported together) and §7.1 S0
出口 (:644–649).

What is pinned here (fixture-level, CPU-fast, no data/checkpoint access):

* the four probes are the plan's table,字面: B1 ``graph_s`` → ``q``/endpoint,
  B2 ``graph_s+key_s+key_v`` → ``δu``, B3 ``key_s+key_v`` → ``δu``,
  B4 ``graph_s+key_s`` → ``δu`` (input sets, row level and target sets);
* the concatenation order/widths of each probe input, and the typed refusals
  for a missing feature / drifted width;
* the metrics are the ones the plan names (residual MAE vs the M0 ``‖y-u0‖``
  baseline, R²) with hand-computed values on 3-vector rows;
* the fitting machinery actually learns: on a PLANTED signal that lives only
  in ``key_v`` the B2 probe reaches a high R² while B4 (no ``key_v``) cannot —
  i.e. the S0-B comparison would detect basis information if it existed;
* the fitting is deterministic (same init seeds → identical metrics) and the
  epoch is chosen on the probe-val cut, never on inner-val;
* S0-C: rank / singular values / condition number / projection residual
  (with AND without channel 0) / teacher coefficients / leave-one-channel-out
  are hand-computed on a 3-channel basis, including a rank-deficient basis
  where the condition number is ``inf`` and the pseudo-inverse must not blow
  up; zero-residual rows are excluded instead of producing 0/0;
* the plan's thresholds are REPORTED as comparisons (100.0 / 0.5 / 10 %) and
  no function in the module returns a go/no-go verdict.

The real-data leg lives in ``tests/pesn/test_v543_s0_probe_realdata.py``
(``realdata`` marker).
"""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from pes2ts_core.pesn.features.key_atom_spec_v3 import BOND_SCALE  # noqa: E402
from pes2ts_core.pesn.models.v543 import (  # noqa: E402
    GRAPH_S_DIM_V543,
    KEY_S_DIM_V543,
    KEY_V_CHANNELS_V543,
    KEY_V_DIM_V543,
)
from pes2ts_core.pesn.offline.evaluation.v543_s0_probe import (  # noqa: E402
    BOND_SCALE_NOTE,
    INTENT_CLASSES,
    PROBE_ORDER,
    PROBE_SPECS,
    S0_B_GO_RELATIVE_IMPROVEMENT,
    S0_C_CONDITION_THRESHOLD,
    S0_C_PROJECTION_RESIDUAL_THRESHOLD,
    STRATA_ORDER,
    TARGET_DELTA_U,
    TARGET_ENDPOINT,
    TARGET_Q,
    ProbeHead,
    ProbeRows,
    ProbeSpec,
    ProbeTrainConfig,
    Standardizer,
    V543ProbeError,
    baseline_metrics,
    basis_coverage,
    concat_probe_rows,
    evaluate_probe,
    go_comparison,
    grouped_metrics,
    probe_input_matrix,
    rank_rtol,
    regression_metrics,
    stratify_metrics,
    summarize_basis_coverage,
    take_probe_rows,
    train_probe,
)

SEED = 17


# ---------------------------------------------------------------------------
# fixtures: synthetic rows whose signal is planted where the test wants it
# ---------------------------------------------------------------------------
def _rows(n: int, *, kind: str = "key", seed: int = SEED,
          with_graph: bool = True, with_key_s: bool = True,
          with_key_v: bool = True, target_from: str = "key_v",
          strata: tuple[str, ...] | None = None,
          intents: tuple[str, ...] | None = None) -> ProbeRows:
    generator = torch.Generator().manual_seed(int(seed))
    graph_s = torch.randn(n, GRAPH_S_DIM_V543, generator=generator)
    key_s = torch.randn(n, KEY_S_DIM_V543, generator=generator) * 0.5
    key_v = torch.randn(n, KEY_V_CHANNELS_V543, KEY_V_DIM_V543,
                        generator=generator)
    if target_from == "key_v":
        target = 0.3 * key_v[:, 0, :] + 0.1 * key_v[:, 3, :]
    elif target_from == "key_s":
        target = 0.2 * key_s[:, :3]
    elif target_from == "graph_s":
        target = 0.1 * graph_s[:, None, :3].expand(n, 1, 3).reshape(n, 3)
    else:
        raise AssertionError(f"unknown target source {target_from!r}")
    features: dict[str, torch.Tensor] = {}
    if with_graph:
        features["graph_s"] = graph_s
    if with_key_s:
        features["key_s"] = key_s
    if with_key_v:
        features["key_v"] = key_v.reshape(n, -1)
    labels = list(intents) if intents else ["both"] * n
    return ProbeRows(
        kind=kind,
        features=features,
        targets={TARGET_DELTA_U: target},
        frame_index=torch.arange(n, dtype=torch.int64) // 4,
        n_frames=int(math.ceil(n / 4)),
        strata=tuple(strata) if strata else tuple(
            STRATA_ORDER[i % len(STRATA_ORDER)] for i in range(n)),
        intent_classes=tuple(labels),
        eligible=torch.ones(n, dtype=torch.bool),
    ).check()


def _frame_rows(n: int, *, seed: int = SEED) -> ProbeRows:
    generator = torch.Generator().manual_seed(int(seed) + 1)
    graph_s = torch.randn(n, GRAPH_S_DIM_V543, generator=generator)
    q = torch.tanh(0.4 * graph_s[:, 0] + 0.2 * graph_s[:, 3]).reshape(-1, 1)
    endpoint = (0.7 * graph_s[:, 1] - 0.3 * graph_s[:, 5]).reshape(-1, 1)
    return ProbeRows(
        kind="frame",
        features={"graph_s": graph_s},
        targets={TARGET_Q: q, TARGET_ENDPOINT: endpoint},
        frame_index=torch.arange(n, dtype=torch.int64),
        n_frames=n,
        strata=tuple(STRATA_ORDER[i % len(STRATA_ORDER)] for i in range(n)),
        intent_classes=tuple(INTENT_CLASSES[i % len(INTENT_CLASSES)]
                             for i in range(n)),
        eligible=torch.ones(n, dtype=torch.bool),
    ).check()


SMALL = ProbeTrainConfig(hidden=32, lr=1e-2, batch_size=256,
                         max_epochs=80, patience=20, init_seeds=(0,))


# ---------------------------------------------------------------------------
# the plan's table, verbatim
# ---------------------------------------------------------------------------
def test_probe_table_matches_plan_lines() -> None:
    """§7.1 :618–629 — input / target / row level of B1–B4,字面."""
    assert PROBE_ORDER == ("B1", "B2", "B3", "B4")
    b1, b2, b3, b4 = (PROBE_SPECS[k] for k in PROBE_ORDER)
    assert b1.features == ("graph_s",)
    assert set(b1.targets) == {TARGET_Q, TARGET_ENDPOINT}
    assert b1.row_level == "frame"
    assert b2.features == ("graph_s", "key_s", "key_v")
    assert b3.features == ("key_s", "key_v")
    assert b4.features == ("graph_s", "key_s")
    for spec in (b2, b3, b4):
        assert spec.targets == (TARGET_DELTA_U,)
        assert spec.row_level == "key"
    # the plan's purpose texts, verbatim
    assert b1.purpose == "全局标量是否足够 PROCESS"
    assert b2.purpose == "Bridge 是否包含 MOVE 残差信息"
    assert b3.purpose == "局部信息是否足够"
    assert b4.purpose == "向量基底是否有额外贡献"


def test_probe_input_widths_and_order() -> None:
    """The concatenation order is the plan's feature order, channel by channel."""
    rows = _rows(7)
    b2 = probe_input_matrix(PROBE_SPECS["B2"], rows)
    assert list(b2.shape) == [
        7, GRAPH_S_DIM_V543 + KEY_S_DIM_V543 + KEY_V_CHANNELS_V543 * KEY_V_DIM_V543]
    # block 0 = graph_s, block 1 = key_s, block 2 = key_v flattened
    assert torch.equal(b2[:, :GRAPH_S_DIM_V543], rows.features["graph_s"])
    assert torch.equal(
        b2[:, GRAPH_S_DIM_V543:GRAPH_S_DIM_V543 + KEY_S_DIM_V543],
        rows.features["key_s"])
    assert torch.equal(
        b2[:, GRAPH_S_DIM_V543 + KEY_S_DIM_V543:], rows.features["key_v"])
    # B4 drops key_v, B3 drops graph_s — widths follow the table
    assert int(probe_input_matrix(PROBE_SPECS["B4"], rows).shape[1]) == (
        GRAPH_S_DIM_V543 + KEY_S_DIM_V543)
    assert int(probe_input_matrix(PROBE_SPECS["B3"], rows).shape[1]) == (
        KEY_S_DIM_V543 + KEY_V_CHANNELS_V543 * KEY_V_DIM_V543)


def test_missing_feature_is_a_typed_refusal() -> None:
    rows = _rows(4, with_key_v=False)
    with pytest.raises(V543ProbeError) as info:
        probe_input_matrix(PROBE_SPECS["B2"], rows)
    assert info.value.code == "V543_PROBE_FEATURE"
    assert "key_v" in info.value.detail


def test_probe_spec_feature_width_refusal() -> None:
    bad = ProbeSpec(probe_id="BX", features=("nope",), targets=("t",),
                    row_level="key", purpose="", plan_lines="")
    with pytest.raises(V543ProbeError) as info:
        bad.feature_dim()
    assert info.value.code == "V543_PROBE_FEATURE"


# ---------------------------------------------------------------------------
# rows / standardizer / head
# ---------------------------------------------------------------------------
def test_probe_rows_reject_misalignment() -> None:
    rows = _rows(5)
    rows.targets[TARGET_DELTA_U] = rows.targets[TARGET_DELTA_U][:3]
    with pytest.raises(V543ProbeError) as info:
        rows.check()
    assert info.value.code in ("V543_PROBE_SHAPE", "V543_PROBE_ROW_COUNT")


def test_probe_rows_reject_non_finite() -> None:
    rows = _rows(5)
    rows.features["key_s"][0, 0] = float("nan")
    with pytest.raises(V543ProbeError) as info:
        rows.check()
    assert info.value.code == "V543_PROBE_NONFINITE"


def test_concat_reindexes_frames_and_keeps_labels() -> None:
    left = _rows(6, strata=("start",) * 6, intents=("breaking",) * 6)
    right = _rows(6, strata=("post_TS",) * 6, intents=("forming",) * 6,
                  seed=3)
    pooled = concat_probe_rows([left, right])
    assert pooled.n_rows == 12
    assert pooled.n_frames == left.n_frames + right.n_frames
    assert pooled.strata[:6] == ("start",) * 6
    assert pooled.strata[6:] == ("post_TS",) * 6
    # the right part's frame indices are shifted by the left part's count
    assert int(pooled.frame_index[6]) == int(left.frame_index[0]) + left.n_frames
    assert pooled.intent_classes[0] == "breaking"
    assert pooled.intent_classes[-1] == "forming"


def test_take_rows_subset_keeps_stratifiers() -> None:
    rows = _rows(8, strata=("start", "pre_TS") * 4,
                 intents=("breaking", "forming") * 4)
    mask = torch.tensor([True, False] * 4)
    taken = take_probe_rows(rows, mask)
    assert taken.n_rows == 4
    assert taken.strata == ("start",) * 4
    assert taken.intent_classes == ("breaking",) * 4


def test_standardizer_is_fit_on_train_only() -> None:
    train = _rows(64, seed=1)
    probe_val = _rows(64, seed=2)
    std = Standardizer.fit(train.features["key_s"])
    train_z = std.apply(train.features["key_s"])
    val_z = std.apply(probe_val.features["key_s"])
    assert abs(float(train_z.mean())) < 1e-5
    assert abs(float(train_z.std(unbiased=False)) - 1.0) < 1e-4
    # the val rows are standardised with the TRAIN statistics (not refitted)
    assert abs(float(val_z.mean())) > 1e-3
    with pytest.raises(V543ProbeError) as info:
        Standardizer.fit(torch.zeros(0, 3))
    assert info.value.code == "V543_PROBE_EMPTY"


def test_probe_head_parameter_formula() -> None:
    head = ProbeHead(10, 3, 7)
    assert head.param_count() == 10 * 7 + 7 + 7 * 3 + 3
    with pytest.raises(V543ProbeError) as info:
        ProbeHead(0, 3, 7)
    assert info.value.code == "V543_PROBE_SHAPE"


# ---------------------------------------------------------------------------
# metrics: hand-computed
# ---------------------------------------------------------------------------
def test_regression_metrics_hand_computed() -> None:
    pred = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    target = torch.tensor([[0.0, 0.0, 0.0], [3.0, 4.0, 0.0]])
    rows = regression_metrics(pred, target)
    # per-key L2 norms are 1 and 5 -> MAE 3
    assert rows["mae_atom_angstrom"] == pytest.approx(3.0)
    # |diff| elements: 1,0,0,3,4,0 -> component MAE 8/6
    assert rows["mae_component_angstrom"] == pytest.approx(8.0 / 6.0)
    # SS_res = 1 + 25 = 26 over 6 elements
    assert rows["rmse_angstrom"] == pytest.approx(math.sqrt(26.0 / 6.0))
    # SS_tot uses the per-column means of the target
    mean = target.mean(dim=0)
    ss_tot = float(((target - mean) ** 2).sum())
    assert rows["r2"] == pytest.approx(1.0 - 26.0 / ss_tot)
    assert rows["n_rows"] == 2.0


def test_baseline_m0_is_the_plan_definition() -> None:
    """§7.1 :629 — the M0 comparison is ``‖y - u0‖``, i.e. predicting δu = 0."""
    target = torch.tensor([[3.0, 4.0, 0.0], [0.0, 0.0, 0.0]])
    base = baseline_metrics(target)
    assert base["m0_mae_atom_angstrom"] == pytest.approx(2.5)
    assert base["m0_n_rows"] == 2.0


def test_regression_metrics_shape_refusal() -> None:
    with pytest.raises(V543ProbeError) as info:
        regression_metrics(torch.zeros(2, 3), torch.zeros(3, 3))
    assert info.value.code == "V543_PROBE_SHAPE"


def test_grouped_metrics_keeps_empty_buckets() -> None:
    pred = torch.zeros(4, 3)
    target = torch.ones(4, 3)
    labels = ["start", "start", "post_TS", "post_TS"]
    table = grouped_metrics(pred, target, labels, STRATA_ORDER)
    assert table["start"]["n_rows"] == 2.0
    assert table["endpoint"] == {"n_rows": 0.0}      # never silently dropped
    assert table["start"]["mae_atom_angstrom"] == pytest.approx(math.sqrt(3.0))


# ---------------------------------------------------------------------------
# fitting: learns a planted signal, deterministic, epoch from probe-val
# ---------------------------------------------------------------------------
def test_probe_learns_a_planted_key_v_signal() -> None:
    """The S0-B machinery must DETECT basis information when it is there."""
    train = _rows(768, seed=11, target_from="key_v")
    val = _rows(256, seed=12, target_from="key_v")
    test = _rows(256, seed=13, target_from="key_v")
    b2 = train_probe(PROBE_SPECS["B2"], train, val,
                     target=TARGET_DELTA_U, config=SMALL)
    b4 = train_probe(PROBE_SPECS["B4"], train, val,
                     target=TARGET_DELTA_U, config=SMALL)
    m2 = evaluate_probe(b2, test, per_init_seed=False)
    m4 = evaluate_probe(b4, test, per_init_seed=False)
    assert m2["metrics_ensemble"]["r2"] > 0.5
    assert m2["metrics_ensemble"]["mae_atom_angstrom"] < \
        m4["metrics_ensemble"]["mae_atom_angstrom"]
    assert m2["metrics_ensemble"]["mae_atom_angstrom"] < \
        m2["baseline_m0"]["m0_mae_atom_angstrom"]
    assert m2["relative_improvement_vs_m0"] > 0.0
    assert m2["relative_improvement_vs_m0"] > m4["relative_improvement_vs_m0"]


def test_probe_fit_is_deterministic() -> None:
    train = _rows(384, seed=5, target_from="key_v")
    val = _rows(128, seed=6, target_from="key_v")
    test = _rows(128, seed=7, target_from="key_v")
    first = train_probe(PROBE_SPECS["B3"], train, val,
                        target=TARGET_DELTA_U, config=SMALL)
    second = train_probe(PROBE_SPECS["B3"], train, val,
                         target=TARGET_DELTA_U, config=SMALL)
    assert first.selected_epoch == second.selected_epoch
    a = evaluate_probe(first, test, per_init_seed=False)["metrics_ensemble"]
    b = evaluate_probe(second, test, per_init_seed=False)["metrics_ensemble"]
    assert a["mae_atom_angstrom"] == pytest.approx(b["mae_atom_angstrom"])


def test_probe_epoch_is_selected_not_max_epochs() -> None:
    train = _rows(512, seed=21, target_from="key_v")
    val = _rows(128, seed=22, target_from="key_v")
    fitted = train_probe(PROBE_SPECS["B2"], train, val,
                         target=TARGET_DELTA_U,
                         config=ProbeTrainConfig(hidden=32, lr=5e-3,
                                                 batch_size=128,
                                                 max_epochs=30, patience=5,
                                                 init_seeds=(0,)))
    epoch = fitted.selected_epoch[0]
    history = fitted.val_history[0]
    assert 1 <= epoch <= min(30, len(history))
    # the chosen epoch is the argmin of the probe-val curve
    assert history[epoch - 1] == pytest.approx(min(history))
    assert fitted.n_train_rows == 512
    assert fitted.n_val_rows == 128


def test_b1_fits_both_process_scalars() -> None:
    train = _frame_rows(1024, seed=31)
    val = _frame_rows(256, seed=32)
    test = _frame_rows(256, seed=33)
    config = ProbeTrainConfig(hidden=32, lr=5e-3, batch_size=256,
                              max_epochs=80, patience=20, init_seeds=(0,))
    for target in (TARGET_Q, TARGET_ENDPOINT):
        fitted = train_probe(PROBE_SPECS["B1"], train, val, target=target,
                             config=config)
        metrics = evaluate_probe(fitted, test, per_init_seed=False)
        assert metrics["metrics_ensemble"]["r2"] > 0.8
        assert metrics["baseline_m0"]["m0_mae_atom_angstrom"] > 0.0


def test_stratify_metrics_by_stratum_and_intent() -> None:
    train = _rows(512, seed=41, target_from="key_v")
    val = _rows(128, seed=42, target_from="key_v")
    test = _rows(256, seed=43, target_from="key_v",
                 intents=tuple(INTENT_CLASSES[i % 4] for i in range(256)))
    fitted = train_probe(PROBE_SPECS["B2"], train, val,
                         target=TARGET_DELTA_U, config=SMALL)
    by_stratum = stratify_metrics(fitted, test, key="stratum")
    assert sorted(by_stratum) == sorted(STRATA_ORDER)
    by_intent = stratify_metrics(fitted, test, key="intent")
    assert sorted(by_intent) == sorted(INTENT_CLASSES)
    for table in (by_stratum, by_intent):
        for bucket in table.values():
            assert "mae_atom_angstrom" in bucket or bucket == {"n_rows": 0.0}
    with pytest.raises(V543ProbeError) as info:
        stratify_metrics(fitted, test, key="nope")
    assert info.value.code == "V543_PROBE_STRATIFIER"


def test_go_comparison_is_a_measurement_not_a_verdict() -> None:
    """§7.1 :630 — the ≥10 % figure is reported; nothing is ruled."""
    s0_b = {
        "B2": {"metrics_ensemble": {"mae_atom_angstrom": 0.090}},
        "B3": {"metrics_ensemble": {"mae_atom_angstrom": 0.120}},
        "B4": {"metrics_ensemble": {"mae_atom_angstrom": 0.100}},
        "B1": {"delta_u_m0_baseline": {"m0_mae_atom_angstrom": 0.200}},
    }
    out = go_comparison(s0_b)
    assert out["relative_improvement_b2_vs_b4"] == pytest.approx(0.10)
    assert out["go_threshold_met"] is True
    assert out["go_threshold_relative_improvement"] == \
        S0_B_GO_RELATIVE_IMPROVEMENT
    assert out["relative_improvement_b2_vs_m0"] == pytest.approx(0.55)
    assert "thresholds_only_no_verdict" in out
    # below the threshold the flag flips, but again nothing is "failed"
    s0_b["B2"]["metrics_ensemble"]["mae_atom_angstrom"] = 0.098
    out = go_comparison(s0_b)
    assert out["go_threshold_met"] is False
    assert out["relative_improvement_b2_vs_b4"] == pytest.approx(0.02)


# ---------------------------------------------------------------------------
# S0-C: hand-computable bases
# ---------------------------------------------------------------------------
def _orthonormal_basis(n: int = 1) -> torch.Tensor:
    """8 channels cycling through the 3 axis directions.

    Channel ``i`` is ``e_{i mod 3}``, so the channels span exactly R³ (3
    directions) while being maximally redundant in the column sense — the
    "8 个任意 3D 向量几乎必然张成 R3" situation of plan §7.1 :642.  The
    columns of direction 0/1 appear 3 times, direction 2 twice, so the
    singular values are ``√3, √3, √2``.
    """
    frame = torch.tensor([[1.0, 0.0, 0.0],
                          [0.0, 1.0, 0.0],
                          [0.0, 0.0, 1.0]])
    channels = [frame[i % 3] for i in range(KEY_V_CHANNELS_V543)]
    basis = torch.stack(channels, dim=0)              # [8,3]
    return basis.unsqueeze(0).repeat(n, 1, 1)


def test_basis_coverage_full_span() -> None:
    key_v = _orthonormal_basis(1)
    residual = torch.tensor([[0.3, -0.4, 0.5]])
    cov = basis_coverage(key_v, residual)
    assert int(cov.rank[0]) == 3
    assert cov.singular_values[0, 0].item() == pytest.approx(math.sqrt(3.0))
    assert cov.singular_values[0, 1].item() == pytest.approx(math.sqrt(3.0))
    assert cov.singular_values[0, -1].item() == pytest.approx(math.sqrt(2.0))
    assert cov.condition_number[0].item() == pytest.approx(
        math.sqrt(3.0 / 2.0))
    assert cov.coverage[0].item() == pytest.approx(1.0, abs=1e-6)
    assert cov.projection_residual_ratio[0].item() == pytest.approx(0.0,
                                                                  abs=1e-6)
    assert cov.coverage_no_ch0[0].item() == pytest.approx(1.0, abs=1e-6)
    assert cov.residual_norm[0].item() == pytest.approx(
        math.sqrt(0.09 + 0.16 + 0.25))
    # the minimum-norm least-squares solution splits each residual component
    # equally over the duplicate channels of its direction
    coefficients = cov.coefficients[0]
    assert float(coefficients[0] + coefficients[3] + coefficients[6]) == \
        pytest.approx(0.3, abs=1e-5)
    assert float(coefficients[1] + coefficients[4] + coefficients[7]) == \
        pytest.approx(-0.4, abs=1e-5)
    assert float(coefficients[2] + coefficients[5]) == pytest.approx(
        0.5, abs=1e-5)


def test_basis_coverage_rank_deficient_and_infinite_condition() -> None:
    """A basis spanning ONE direction: rank 1, condition number ``inf``."""
    direction = torch.tensor([1.0, 0.0, 0.0])
    key_v = direction.reshape(1, 1, 3).repeat(1, KEY_V_CHANNELS_V543, 1)
    residual = torch.tensor([[0.0, 2.0, 0.0]])       # orthogonal to the basis
    cov = basis_coverage(key_v, residual)
    assert int(cov.rank[0]) == 1
    assert math.isinf(float(cov.condition_number[0]))
    assert cov.coverage[0].item() == pytest.approx(0.0, abs=1e-6)
    assert cov.projection_residual_ratio[0].item() == pytest.approx(1.0)
    assert float(cov.singular_values[0, 0]) == pytest.approx(
        math.sqrt(KEY_V_CHANNELS_V543))
    assert float(cov.effective_rank[0]) == pytest.approx(1.0, abs=1e-5)
    assert float(cov.stable_rank[0]) == pytest.approx(1.0, abs=1e-5)


def test_basis_coverage_without_channel_zero() -> None:
    """Channel 0 (``u0/a_ref``) is the only channel with a y component here."""
    key_v = torch.zeros(1, KEY_V_CHANNELS_V543, 3)
    key_v[0, 0] = torch.tensor([0.0, 1.0, 0.0])      # channel 0 → +y
    key_v[0, 1] = torch.tensor([1.0, 0.0, 0.0])      # channel 1 → +x
    residual = torch.tensor([[0.0, 0.5, 0.0]])
    cov = basis_coverage(key_v, residual)
    assert cov.coverage[0].item() == pytest.approx(1.0)
    assert cov.coverage_no_ch0[0].item() == pytest.approx(0.0, abs=1e-6)
    assert cov.projection_residual_ratio_no_ch0[0].item() == \
        pytest.approx(1.0)


def test_leave_one_channel_out_is_hand_computable() -> None:
    key_v = torch.zeros(1, KEY_V_CHANNELS_V543, 3)
    key_v[0, 0] = torch.tensor([0.0, 1.0, 0.0])
    key_v[0, 1] = torch.tensor([1.0, 0.0, 0.0])
    residual = torch.tensor([[1.0, 1.0, 0.0]])
    cov = basis_coverage(key_v, residual)
    deltas = cov.leave_one_out_delta[0]
    # dropping the +y channel costs half of the projection; the empty
    # channels cost nothing
    assert float(deltas[0]) == pytest.approx(1.0 - 1.0 / math.sqrt(2.0),
                                             abs=1e-5)
    assert float(deltas[1]) == pytest.approx(1.0 - 1.0 / math.sqrt(2.0),
                                             abs=1e-5)
    for channel in range(2, KEY_V_CHANNELS_V543):
        assert float(deltas[channel]) == pytest.approx(0.0, abs=1e-6)


def test_zero_residual_rows_are_excluded_not_0_over_0() -> None:
    key_v = _orthonormal_basis(2)
    residual = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    cov = basis_coverage(key_v, residual)
    assert cov.valid.tolist() == [False, True]
    summary = summarize_basis_coverage(cov)
    assert summary["n_zero_residual_rows"] == 1
    assert summary["n_valid_rows"] == 1
    assert summary["projection_residual_ratio"]["n_rows"] == 1
    assert not math.isnan(summary["projection_residual_ratio"]["median"])


def test_basis_coverage_refusals() -> None:
    with pytest.raises(V543ProbeError) as info:
        basis_coverage(torch.zeros(2, 7, 3), torch.zeros(2, 3))
    assert info.value.code == "V543_PROBE_SHAPE"
    with pytest.raises(V543ProbeError) as info:
        basis_coverage(torch.zeros(2, 8, 3), torch.zeros(3, 3))
    assert info.value.code == "V543_PROBE_SHAPE"
    key_v = _orthonormal_basis(1)
    key_v[0, 0, 0] = float("inf")
    with pytest.raises(V543ProbeError) as info:
        basis_coverage(key_v, torch.zeros(1, 3))
    assert info.value.code == "V543_PROBE_NONFINITE"
    with pytest.raises(V543ProbeError) as info:
        summarize_basis_coverage(basis_coverage(_orthonormal_basis(1),
                                                torch.zeros(1, 3)))
    assert info.value.code == "V543_PROBE_EMPTY"


def test_summarize_reports_thresholds_as_data() -> None:
    """§7.1 :642 — condition > 100 / projection residual median > 0.5 are REPORTED."""
    # a basis that is orthogonal in one row and near-degenerate in another
    key_v = _orthonormal_basis(4).clone()
    key_v[3, 2] = key_v[3, 0]                        # duplicate a channel
    key_v[1, 1] = 1e-4 * key_v[1, 0]                 # near-degenerate
    residual = torch.randn(4, 3, generator=torch.Generator().manual_seed(7))
    cov = basis_coverage(key_v, residual)
    summary = summarize_basis_coverage(cov)
    for key in ("rank", "singular_values", "condition_number", "effective_rank",
                "stable_rank", "coverage", "coverage_no_channel0",
                "projection_residual_ratio",
                "projection_residual_ratio_no_channel0",
                "teacher_coefficients", "leave_one_channel_out",
                "thresholds_no_verdict"):
        assert key in summary, key
    thresholds = summary["thresholds_no_verdict"]
    assert thresholds["condition_number_threshold"] == \
        S0_C_CONDITION_THRESHOLD
    assert thresholds["projection_residual_median_threshold"] == \
        S0_C_PROJECTION_RESIDUAL_THRESHOLD
    assert isinstance(thresholds["measured_condition_number_median"], float)
    assert isinstance(
        thresholds["measured_projection_residual_median"], float)
    assert summary["condition_number"]["share_rows_over_threshold"] == \
        pytest.approx(summary["condition_number"]["n_rows_over_threshold"]
                      / summary["n_valid_rows"])
    # the per-channel leave-one-out table covers all eight channels
    channels = [row["channel"] for row in
                summary["leave_one_channel_out"]["per_channel"]]
    assert channels == list(range(KEY_V_CHANNELS_V543))


def test_summarize_groups_by_stratum_and_intent() -> None:
    n = 8
    key_v = _orthonormal_basis(n)
    residual = torch.randn(n, 3, generator=torch.Generator().manual_seed(3))
    cov = basis_coverage(key_v, residual)
    strata = ("start", "pre_TS") * (n // 2)
    intents = ("breaking", "forming", "both", "unmoved") * (n // 4)
    summary = summarize_basis_coverage(cov, strata=strata,
                                       intent_classes=intents)
    assert sorted(summary["by_stratum"]) == sorted(STRATA_ORDER)
    assert sorted(summary["by_intent_class"]) == sorted(INTENT_CLASSES)
    assert summary["by_stratum"]["post_TS"] == {"n_key_rows": 0}
    assert summary["by_stratum"]["start"]["n_key_rows"] == n // 2
    with pytest.raises(V543ProbeError) as info:
        summarize_basis_coverage(cov, strata=strata[:-1])
    assert info.value.code == "V543_PROBE_ROW_COUNT"


def test_rank_rtol_is_the_numpy_convention() -> None:
    assert rank_rtol((3, 8)) == pytest.approx(8 * torch.finfo(
        torch.float32).eps)
    # a numerically rank-2 basis is counted as 2, not 3
    key_v = torch.zeros(1, KEY_V_CHANNELS_V543, 3)
    key_v[0, 0] = torch.tensor([1.0, 0.0, 0.0])
    key_v[0, 1] = torch.tensor([0.0, 1.0, 0.0])
    key_v[0, 2] = 1e-9 * key_v[0, 0]
    cov = basis_coverage(key_v, torch.tensor([[1.0, 1.0, 1.0]]))
    assert int(cov.rank[0]) == 2


def test_bond_scale_note_records_the_adr_freeze() -> None:
    """ADR-0034 §10 v1.1 — pair_edit = BOND_SCALE × forward ΔB, recorded."""
    assert BOND_SCALE == 3.0
    assert BOND_SCALE_NOTE["bond_scale"] == 3.0
    assert BOND_SCALE_NOTE["relation"] == "pair_edit = BOND_SCALE * forward ΔB"
    assert "NOT pair_edit" in BOND_SCALE_NOTE["s0_b_targets"]
    assert "sign only" in BOND_SCALE_NOTE["s0_c_pair_edit_use"]
