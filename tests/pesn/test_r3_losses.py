"""R3 v3 loss math tests (T1.5): λ=0 bit-identity, buckets, weights-once.

Frozen semantics under test (campaign ``r3_training.loss``, R3 plan §5):

* ``λ_strong=λ_dead=0`` ⇒ the total AND every base component are BIT-identical
  to the B0-F baseline
  (:func:`pes2ts_core.pesn.offline.training.key_atom_bc.key_atom_bc_loss_vectorized`)
  — asserted with ``torch.equal`` over randomized fixtures;
* empty buckets contribute EXACTLY 0 (and leave the total bitwise unchanged);
* strong/dead formulas match the frozen hand-computed values;
* frame weights apply EXACTLY ONCE (curve-level totals split across
  intent/node samples; uniform rescaling is a no-op, re-application is not);
* deterministic reduction: repeated calls are bitwise equal;
* aux arms S2/S10/D2/D10 are single-bucket arms (mixing refused).

Fixture-level results — ``not_admissible_until_gate_pass`` until G0+G1.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from pes2ts_core.pesn.models.key_atom_losses_v3 import (  # noqa: E402
    AUX_ARMS_V3,
    LAMBDA_COS_MAX,
    KeyAtomLossV3Config,
    KeyAtomV3LossError,
    aux_arm_lambdas_v3,
    cos_lambda_r0,
    key_atom_loss_v3,
)
from pes2ts_core.pesn.offline.training.key_atom_bc import (  # noqa: E402
    KeyAtomLossConfig,
    key_atom_bc_loss_vectorized,
)

DTYPE = torch.float32


def _fixture(n_frames: int = 2, keys_per_frame: int = 3, seed: int = 0,
             pairs_per_frame: int = 1):
    """Randomized-but-deterministic loss fixture (global-row pair_index)."""
    g = torch.Generator().manual_seed(seed)
    n_nodes = n_frames * keys_per_frame
    pos = (torch.randn(n_nodes, 3, generator=g) * 1.5).to(DTYPE)
    key_index = torch.arange(n_nodes, dtype=torch.long)
    key_ptr = torch.arange(0, n_nodes + 1, keys_per_frame, dtype=torch.long)
    y = (torch.randn(n_nodes, 3, generator=g) * 0.03).to(DTYPE)
    u = y + (torch.randn(n_nodes, 3, generator=g) * 0.01).to(DTYPE)
    degK = torch.randint(1, 3, (n_nodes,), generator=g).to(DTYPE)
    pairs, edits = [], []
    for f in range(n_frames):
        base = f * keys_per_frame
        for p in range(pairs_per_frame):
            i = base + p
            j = base + (p + 1) % keys_per_frame
            pairs.append((i, j))
            edits.append(1.0 if (p + f) % 2 == 0 else -1.0)
    pair_index = torch.tensor(pairs, dtype=torch.long).t().contiguous()
    pair_edit = torch.tensor(edits, dtype=DTYPE)
    return dict(u=u, y=y, degK=degK, key_ptr=key_ptr, pos=pos,
                key_index=key_index, pair_index=pair_index,
                pair_edit=pair_edit, n_frames=n_frames)


def _baseline(fx, *, lambda_cos: float, frame_mask=None):
    return key_atom_bc_loss_vectorized(
        fx["u"], fx["y"], fx["degK"], fx["key_ptr"], fx["pos"],
        fx["key_index"], fx["pair_index"],
        config=KeyAtomLossConfig(), lambda_cos=lambda_cos,
        frame_mask=frame_mask, pair_edit=fx["pair_edit"])


# ---------------------------------------------------------------------------
# T1.5 — λ=0 bitwise identity with B0-F
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("seed", (0, 7, 123))
@pytest.mark.parametrize("lambda_cos", (0.0, 0.02, 0.008))
def test_lambda_zero_bitwise_identity(seed, lambda_cos) -> None:
    fx = _fixture(seed=seed)
    mine = key_atom_loss_v3(
        fx["u"], fx["y"], fx["degK"], fx["key_ptr"], fx["pos"],
        fx["key_index"], fx["pair_index"], pair_edit=fx["pair_edit"],
        lambda_cos=lambda_cos)
    base = _baseline(fx, lambda_cos=lambda_cos)
    assert torch.equal(mine["loss"], base["loss"])
    assert torch.equal(mine["l_atom"], base["l_atom"])
    assert torch.equal(mine["l_pair"], base["l_pair"])
    assert torch.equal(mine["l_cos"], base["l_cos"])


def test_lambda_zero_identity_with_frame_mask() -> None:
    fx = _fixture(n_frames=4, seed=3)
    mask = torch.tensor([True, False, True, True])
    mine = key_atom_loss_v3(
        fx["u"], fx["y"], fx["degK"], fx["key_ptr"], fx["pos"],
        fx["key_index"], fx["pair_index"], pair_edit=fx["pair_edit"],
        lambda_cos=0.02, frame_mask=mask)
    base = _baseline(fx, lambda_cos=0.02, frame_mask=mask)
    assert torch.equal(mine["loss"], base["loss"])
    assert mine["n_frames"] == base["n_frames"] == 3


def test_lambda_zero_identity_with_uniform_weights() -> None:
    fx = _fixture(seed=11)
    w = torch.full((fx["n_frames"],), 0.5, dtype=DTYPE)
    mine = key_atom_loss_v3(
        fx["u"], fx["y"], fx["degK"], fx["key_ptr"], fx["pos"],
        fx["key_index"], fx["pair_index"], pair_edit=fx["pair_edit"],
        lambda_cos=0.02, frame_weight=w)
    base = _baseline(fx, lambda_cos=0.02)
    assert torch.equal(mine["loss"], base["loss"])


# ---------------------------------------------------------------------------
# λ_cos ramp R0 (official main schedule)
# ---------------------------------------------------------------------------
def test_cos_ramp_r0_frozen_values() -> None:
    assert cos_lambda_r0(1) == 0.0
    assert cos_lambda_r0(250) == 0.0            # starts AFTER 250 updates
    assert abs(cos_lambda_r0(251) - 8e-5) < 1e-15
    assert cos_lambda_r0(375) == pytest.approx(0.01)
    assert cos_lambda_r0(500) == LAMBDA_COS_MAX
    assert cos_lambda_r0(9999) == LAMBDA_COS_MAX


def test_cos_ramp_never_exceeds_max() -> None:
    for update in range(0, 2000, 7):
        assert 0.0 <= cos_lambda_r0(update) <= LAMBDA_COS_MAX


# ---------------------------------------------------------------------------
# Bucket formulas (frozen hand-computed values)
# ---------------------------------------------------------------------------
def _axis_fixture(u=None, y=None):
    """3 keys on the x-axis: pairs (0,1) and (1,2); controlled Δd*."""
    pos = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=DTYPE)
    if y is None:
        y = torch.zeros(3, 3, dtype=DTYPE)
    if u is None:
        u = torch.zeros(3, 3, dtype=DTYPE)
    return dict(
        u=u, y=y, degK=torch.ones(3, dtype=DTYPE),
        key_ptr=torch.tensor([0, 3], dtype=torch.long), pos=pos,
        key_index=torch.arange(3, dtype=torch.long),
        pair_index=torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
        pair_edit=torch.tensor([1.0, 1.0], dtype=DTYPE), n_frames=1)


def test_strong_bucket_formula_exact() -> None:
    """Δd*(01)=0.05 (strong), Δd*(12)=0 (dead): Huber_0.25((0−0.05)/0.05)
    = Huber_0.25(1) = 0.25·(1−0.125) = 0.21875."""
    y = torch.tensor(
        [[0.0, 0.0, 0.0], [0.05, 0.0, 0.0], [0.05, 0.0, 0.0]], dtype=DTYPE)
    fx = _axis_fixture(y=y)
    out = key_atom_loss_v3(
        fx["u"], fx["y"], fx["degK"], fx["key_ptr"], fx["pos"],
        fx["key_index"], fx["pair_index"], pair_edit=fx["pair_edit"],
        lambda_cos=0.0, lambda_strong=0.02, aux_arm="S2")
    assert out["n_strong_pairs"] == 1
    assert out["n_dead_pairs"] == 0   # dead bucket not computed (λ_dead=0)
    assert float(out["l_strong"]) == pytest.approx(0.21875, abs=1e-6)


def test_dead_bucket_formula_exact() -> None:
    """Δd̂(12)=0.01: [max(0,0.01−0.005)/0.005]² = 1.0."""
    y = torch.tensor(
        [[0.0, 0.0, 0.0], [0.05, 0.0, 0.0], [0.05, 0.0, 0.0]], dtype=DTYPE)
    u = torch.tensor(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.01, 0.0, 0.0]], dtype=DTYPE)
    fx = _axis_fixture(u=u, y=y)
    out = key_atom_loss_v3(
        fx["u"], fx["y"], fx["degK"], fx["key_ptr"], fx["pos"],
        fx["key_index"], fx["pair_index"], pair_edit=fx["pair_edit"],
        lambda_cos=0.0, lambda_dead=0.02, aux_arm="D2")
    assert out["n_dead_pairs"] == 1
    # float32 norm rounding: |Δd̂| lands within ~4e-6 of 0.01
    assert float(out["l_dead"]) == pytest.approx(1.0, abs=1e-4)


def test_empty_buckets_contribute_exactly_zero() -> None:
    """Δd*=0.01 Å for both pairs sits BETWEEN the dead (≤0.005) and strong
    (≥0.02) thresholds: both buckets are empty → exactly 0, and with λ>0
    the total is BITWISE the base total."""
    y = torch.tensor(
        [[0.0, 0.0, 0.0], [0.01, 0.0, 0.0], [0.02, 0.0, 0.0]], dtype=DTYPE)
    fx = _axis_fixture(y=y)
    base = key_atom_loss_v3(
        fx["u"], fx["y"], fx["degK"], fx["key_ptr"], fx["pos"],
        fx["key_index"], fx["pair_index"], pair_edit=fx["pair_edit"],
        lambda_cos=0.0)
    strong = key_atom_loss_v3(
        fx["u"], fx["y"], fx["degK"], fx["key_ptr"], fx["pos"],
        fx["key_index"], fx["pair_index"], pair_edit=fx["pair_edit"],
        lambda_cos=0.0, lambda_strong=0.02, aux_arm="S2")
    dead = key_atom_loss_v3(
        fx["u"], fx["y"], fx["degK"], fx["key_ptr"], fx["pos"],
        fx["key_index"], fx["pair_index"], pair_edit=fx["pair_edit"],
        lambda_cos=0.0, lambda_dead=0.02, aux_arm="D2")
    assert float(strong["l_strong"]) == 0.0
    assert float(dead["l_dead"]) == 0.0
    assert strong["n_strong_pairs"] == 0
    assert dead["n_dead_pairs"] == 0
    assert torch.equal(strong["loss"], base["loss"])
    assert torch.equal(dead["loss"], base["loss"])


def test_unchanged_pairs_excluded_from_buckets() -> None:
    """pair_edit == 0 pairs never enter a bucket even at extreme Δd*."""
    y = torch.tensor(
        [[0.0, 0.0, 0.0], [0.05, 0.0, 0.0], [0.05, 0.0, 0.0]], dtype=DTYPE)
    fx = _axis_fixture(y=y)
    fx["pair_edit"] = torch.tensor([0.0, 0.0], dtype=DTYPE)
    out = key_atom_loss_v3(
        fx["u"], fx["y"], fx["degK"], fx["key_ptr"], fx["pos"],
        fx["key_index"], fx["pair_index"], pair_edit=fx["pair_edit"],
        lambda_cos=0.0, lambda_strong=0.02, aux_arm="S2")
    assert out["n_strong_pairs"] == 0
    assert float(out["l_strong"]) == 0.0


# ---------------------------------------------------------------------------
# Weights apply exactly once
# ---------------------------------------------------------------------------
def test_uniform_weight_rescale_is_a_noop() -> None:
    fx = _fixture(n_frames=3, seed=5)
    kwargs = dict(lambda_cos=0.02)
    base = key_atom_loss_v3(
        fx["u"], fx["y"], fx["degK"], fx["key_ptr"], fx["pos"],
        fx["key_index"], fx["pair_index"], pair_edit=fx["pair_edit"], **kwargs)
    for scale in (2.0, 0.5, 8.0):
        w = torch.full((3,), scale, dtype=DTYPE)
        out = key_atom_loss_v3(
            fx["u"], fx["y"], fx["degK"], fx["key_ptr"], fx["pos"],
            fx["key_index"], fx["pair_index"], pair_edit=fx["pair_edit"],
            frame_weight=w, **kwargs)
        assert torch.equal(out["loss"], base["loss"]), scale


def test_curve_level_split_matches_manual_single_application() -> None:
    """Curve A = frames 0+1 (0.5 each, total 1), curve B = frame 2 (1.0):
    the batched loss equals the manual once-weighted frame combination."""
    fx = _fixture(n_frames=3, keys_per_frame=3, seed=9)
    singles = []
    for f in range(3):
        mask = torch.zeros(3, dtype=torch.bool)
        mask[f] = True
        out = key_atom_loss_v3(
            fx["u"], fx["y"], fx["degK"], fx["key_ptr"], fx["pos"],
            fx["key_index"], fx["pair_index"], pair_edit=fx["pair_edit"],
            lambda_cos=0.02, frame_mask=mask)
        singles.append(float(out["loss"]))
    manual = (0.5 * singles[0] + 0.5 * singles[1] + 1.0 * singles[2]) / 2.0
    w = torch.tensor([0.5, 0.5, 1.0], dtype=DTYPE)
    batched = key_atom_loss_v3(
        fx["u"], fx["y"], fx["degK"], fx["key_ptr"], fx["pos"],
        fx["key_index"], fx["pair_index"], pair_edit=fx["pair_edit"],
        lambda_cos=0.02, frame_weight=w)
    assert float(batched["loss"]) == pytest.approx(manual, abs=1e-7)


def test_double_application_is_detectable() -> None:
    """Weights applied twice (w∘w) must NOT reproduce the single application
    for heterogeneous weights — the detector behind 'applied exactly once'."""
    fx = _fixture(n_frames=3, seed=9)
    w = torch.tensor([0.25, 0.5, 1.0], dtype=DTYPE)
    once = key_atom_loss_v3(
        fx["u"], fx["y"], fx["degK"], fx["key_ptr"], fx["pos"],
        fx["key_index"], fx["pair_index"], pair_edit=fx["pair_edit"],
        lambda_cos=0.02, frame_weight=w)
    twice = key_atom_loss_v3(
        fx["u"], fx["y"], fx["degK"], fx["key_ptr"], fx["pos"],
        fx["key_index"], fx["pair_index"], pair_edit=fx["pair_edit"],
        lambda_cos=0.02, frame_weight=w * w)
    assert abs(float(once["loss"]) - float(twice["loss"])) > 1e-9


def test_frame_weight_shape_validated() -> None:
    fx = _fixture(n_frames=2, seed=1)
    with pytest.raises(KeyAtomV3LossError) as exc:
        key_atom_loss_v3(
            fx["u"], fx["y"], fx["degK"], fx["key_ptr"], fx["pos"],
            fx["key_index"], fx["pair_index"], pair_edit=fx["pair_edit"],
            frame_weight=torch.ones(5, dtype=DTYPE))
    assert exc.value.code == "FRAME_WEIGHT_SHAPE"


# ---------------------------------------------------------------------------
# Determinism + aux arms
# ---------------------------------------------------------------------------
def test_deterministic_reduction_bitwise() -> None:
    fx = _fixture(n_frames=3, seed=21)
    outs = [
        key_atom_loss_v3(
            fx["u"], fx["y"], fx["degK"], fx["key_ptr"], fx["pos"],
            fx["key_index"], fx["pair_index"], pair_edit=fx["pair_edit"],
            lambda_cos=0.02, lambda_strong=0.02, aux_arm="S2")
        for _ in range(2)]
    for key in ("loss", "l_atom", "l_pair", "l_cos", "l_strong", "l_dead"):
        assert torch.equal(outs[0][key], outs[1][key]), key


def test_aux_arms_registry_and_separation() -> None:
    assert set(AUX_ARMS_V3) == {"S2", "S10", "D2", "D10"}
    assert AUX_ARMS_V3["S2"][0] == "strong"
    assert AUX_ARMS_V3["D10"] == ("dead", 0.10)
    assert aux_arm_lambdas_v3("S2", 0.02) == (0.02, 0.0)
    assert aux_arm_lambdas_v3("D10", 0.1) == (0.0, 0.1)
    with pytest.raises(KeyAtomV3LossError):
        aux_arm_lambdas_v3("S2", 0.7)        # λ ≤ 0.5
    with pytest.raises(KeyAtomV3LossError):
        aux_arm_lambdas_v3("SD", 0.1)        # unknown / mixed


def test_mixed_arms_refused() -> None:
    fx = _fixture(seed=2)
    with pytest.raises(KeyAtomV3LossError) as exc:
        key_atom_loss_v3(
            fx["u"], fx["y"], fx["degK"], fx["key_ptr"], fx["pos"],
            fx["key_index"], fx["pair_index"], pair_edit=fx["pair_edit"],
            lambda_strong=0.02, lambda_dead=0.02)
    assert exc.value.code == "AUX_ARM_MIXED"
    with pytest.raises(KeyAtomV3LossError):
        key_atom_loss_v3(
            fx["u"], fx["y"], fx["degK"], fx["key_ptr"], fx["pos"],
            fx["key_index"], fx["pair_index"], pair_edit=fx["pair_edit"],
            aux_arm="S2", lambda_strong=0.02, lambda_dead=0.01)


def test_gradients_finite_through_all_terms() -> None:
    fx = _fixture(n_frames=2, seed=33)
    u = fx["u"].clone().requires_grad_(True)
    out = key_atom_loss_v3(
        u, fx["y"], fx["degK"], fx["key_ptr"], fx["pos"],
        fx["key_index"], fx["pair_index"], pair_edit=fx["pair_edit"],
        lambda_cos=0.02, lambda_strong=0.02, aux_arm="S2")
    out["loss"].backward()
    assert u.grad is not None
    assert bool(torch.isfinite(u.grad).all())


def test_pair_mode_frozen() -> None:
    with pytest.raises(KeyAtomV3LossError):
        KeyAtomLossV3Config(pair_mode="exact_trio")
