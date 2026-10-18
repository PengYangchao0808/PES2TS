"""v4.1 WP1 acceptance addendum (finding 2 / F4(b) "energy" half).

Pins the ONLINE ΔE reference wiring end-to-end (``dual_ts_freerun.py``
``run_rollout``): the only online reference is

    ΔE_0 = 0
    ΔE_t = E_t − E_0        (E_0 = first visited energy)

and the node-scalar channel 2 must equal ``ΔE / s_ΔE`` (the online builder
path via ``build_rollout_state_inputs`` → ``build_state_features``) at every
step.  Uses a scripted observer (reusing the free-run test fixture pattern)
so the per-step energies are exactly known — no ACP, no QC.

The offline semantic (per-branch ``E − E_episode_start``) and the online
semantic (branch-A ``E_t − E_0`` with ``E_0`` at that same endpoint) are the
SAME per-branch reference; this test asserts the online wiring literally.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pes2ts_core.pesn.models.dual_ts_policy import (  # noqa: E402
    DEFAULT_A_MAX_ANGSTROM,
    DualTSModelConfig,
    DualTSPolicy,
)
from pes2ts_core.pesn.offline.evaluation.dual_ts_freerun import (  # noqa: E402
    ObservationCall,
    RolloutSettings,
    RolloutStart,
    run_rollout,
)

Z3 = np.asarray([8, 1, 1], dtype=np.int64)          # O + H + H toy
STEP = 0.14
SCALES = {"s_g": 0.0154, "s_delta_e": 0.0789, "spec_digest": "0" * 64}
IDENTITY = {"action_max": DEFAULT_A_MAX_ANGSTROM}


def coords_at(t: int) -> np.ndarray:
    return np.array([[0.0, 0.0, 0.0],
                     [0.96 + t * STEP, 0.0, 0.0],
                     [1.75, 0.0, 0.0]])


def make_start() -> RolloutStart:
    x = coords_at(0)
    edge_pairs = np.asarray([[0, 1], [0, 2]], dtype=np.int64)
    directed = np.concatenate([edge_pairs, edge_pairs[:, ::-1]], axis=0)
    return RolloutStart(
        group_id="RXN_T0", reaction_id="RXN_T0",
        branch_id="RXN_T0#brA", episode_id="RXN_T0#epA0",
        start_state_id="RXN_T0|A|0000",
        atomic_numbers=Z3.copy(),
        coordinates=np.asarray(x, dtype=np.float64),
        atom_map_ids=tuple(f"m{i + 1:04d}" for i in range(Z3.size)),
        charge=0, multiplicity=1, edge_pairs=edge_pairs,
        edge_index=np.ascontiguousarray(directed.T),
        atom_feats_a=np.asarray([[1.0, 2.0, 0.0, 0.0]] * 3),
        atom_feats_b=np.asarray([[1.0, 1.0, 1.0, 0.0]] * 3),
        edge_feats_a=np.asarray([[1.0, 1.0, 0.0, 0.0]] * 4),
        edge_feats_b=np.asarray([[0.0, 0.0, 1.0, 0.0]] * 4),
        bond_source_a="authority", bond_source_b="authority")


def make_policy(seed: int = 0) -> DualTSPolicy:
    torch.manual_seed(seed)
    return DualTSPolicy(
        config=DualTSModelConfig(light=True),
        action_max=DEFAULT_A_MAX_ANGSTROM).to(torch.float32)


class ScriptedObserver:
    """Offline stub with the observer interface (no ACP, no receipts)."""

    def __init__(self, energy_fn=None, gradient_fn=None):
        self.calls: list[np.ndarray] = []
        self.energy_fn = energy_fn or (lambda x, t: -1.0 - 0.01 * t)
        self.gradient_fn = gradient_fn or (
            lambda x, t: np.full((x.shape[0], 3), 0.01))
        self.billed_count = 0

    def observe(self, z, coords, maps, *, charge, multiplicity,
                node_label):
        t = len(self.calls)
        self.calls.append(np.asarray(coords, dtype=float).copy())
        self.billed_count += 1
        return ObservationCall(
            digest=f"{t:064x}", status="computed", billed=True,
            energy=float(self.energy_fn(coords, t)),
            gradient=np.asarray(self.gradient_fn(coords, t),
                                dtype=np.float64),
            backend_seconds=0.001, receipt_dir=f"/stub/{t}",
            provenance="stub", error=None)


def rollout_settings(max_steps: int) -> RolloutSettings:
    return RolloutSettings(max_steps=max_steps, qc_budget=10_000)


def test_freerun_delta_e_wiring_pins_online_reference():
    """ΔE_0 = 0; ΔE_t = E_t − E_0; node_scalars[:, 2] == ΔE / s_ΔE at every
    step, with E_0 = first visited energy (dual_ts_freerun.py:912-916)."""
    energies = [-1.0, -1.05, -1.11, -1.18]

    def energy_fn(coords, t):
        return energies[min(int(t), len(energies) - 1)]

    observer = ScriptedObserver(
        energy_fn=energy_fn,
        gradient_fn=lambda coords, t: np.full((coords.shape[0], 3), 0.01))

    captured: list[tuple[int, ObservationCall, float, dict]] = []

    def probe(t, inputs, features, obs, delta_e, a, x_next):
        captured.append((int(t), obs, float(delta_e), dict(inputs)))

    record = run_rollout(torch=torch, model=make_policy(seed=3),
                         observer=observer, start=make_start(), scales=SCALES,
                         identity=IDENTITY,
                         settings=rollout_settings(max_steps=3),
                         input_probe=probe)

    assert len(captured) == 3
    # reference = first visited energy (step 0's observed E)
    e0 = float(record["steps"][0]["observation"]["energy_hartree"])
    assert e0 == pytest.approx(energies[0])

    for t, obs, delta_e, inputs in captured:
        assert obs.energy == pytest.approx(energies[t])
        expected = 0.0 if t == 0 else energies[t] - e0
        # the wired ΔE equals E_t − E_0 (and ΔE_0 == 0 exactly)
        assert delta_e == pytest.approx(expected)
        assert record["steps"][t]["delta_e_hartree"] == pytest.approx(expected)
        ns = np.asarray(inputs["node_scalars"])
        assert ns.shape[1] == 5
        # channel 2 == ΔE / s_ΔE (the online builder path)
        np.testing.assert_allclose(
            ns[:, 2], delta_e / SCALES["s_delta_e"], atol=1e-12)

    # ΔE_0 == 0, not merely approx
    assert record["steps"][0]["delta_e_hartree"] == 0.0
