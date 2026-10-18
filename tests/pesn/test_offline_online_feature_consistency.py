"""v4.1 WP1 acceptance test (b): offline/online feature consistency.

For >= 3 real reactions and several states each (including a terminal
state's predecessor and a near-TS state), rebuild the shard row OFFLINE via
:func:`build_state_features` and the rollout input ONLINE via
:func:`build_rollout_state_inputs` from the SAME geometry / gradient / ΔE /
topology.  ``node_scalars`` must be identical within an EXPLICIT tolerance:
exact equality in float64, and <= 1e-6 after a float32 cast.
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pes2ts_core.pesn.contracts.dual_ts import (  # noqa: E402
    DualEndpointTask,
    DualTSTeacher,
)
from pes2ts_core.pesn.features.dual_ts_graph import (  # noqa: E402
    build_state_features,
)
from pes2ts_core.pesn.features.dual_ts_spec import (  # noqa: E402
    load_spec_payload,
)
from pes2ts_core.pesn.offline.evaluation.dual_ts_freerun import (  # noqa: E402
    RolloutStart,
    build_rollout_state_inputs,
)

FEATURES_DIR = PROJECT_ROOT / "outputs/pesn/dual_ts_v2_20261015/wp3_features_v41"
OBSERVED_DIR = PROJECT_ROOT / "outputs/pesn/dual_ts_v2_20261015/wp2_fit50_observed"

N_REACTIONS = 3
FLOAT64_TOL = 0.0           # exact equality in float64
FLOAT32_TOL = 1e-6          # explicit tolerance after float32 cast

pytestmark = pytest.mark.skipif(
    not FEATURES_DIR.is_dir() or not OBSERVED_DIR.is_dir(),
    reason="wp3_features_v41 / wp2_fit50_observed artifacts not present")


def _episode_start_energy(teacher: DualTSTeacher) -> dict:
    by_id = {s.state_id: s for s in teacher.states}
    out = {}
    for ep in teacher.episodes:
        st = by_id[ep.start_state_id]
        assert st.observation is not None
        out[ep.episode_id] = float(st.observation.energy_hartree)
    return out


def _rollout_start(task: DualEndpointTask, shard) -> RolloutStart:
    n = int(np.asarray(shard["atomic_numbers"]).size)
    return RolloutStart(
        group_id=str(task.reaction_group_id),
        reaction_id=str(task.reaction_id),
        branch_id="", episode_id="", start_state_id="",
        atomic_numbers=np.asarray(shard["atomic_numbers"], dtype=np.int64),
        coordinates=np.zeros((n, 3), dtype=np.float64),
        atom_map_ids=tuple(),
        charge=int(task.charge),
        multiplicity=int(task.multiplicity),
        edge_pairs=np.asarray(shard["edge_pairs"], dtype=np.int64),
        edge_index=np.asarray(shard["edge_index"], dtype=np.int64),
        atom_feats_a=np.asarray(shard["atom_feats_a"], dtype=np.float64),
        atom_feats_b=np.asarray(shard["atom_feats_b"], dtype=np.float64),
        edge_feats_a=np.asarray(shard["edge_feats_a"], dtype=np.float64),
        edge_feats_b=np.asarray(shard["edge_feats_b"], dtype=np.float64),
        bond_source_a=str(shard["bond_source_a"]),
        bond_source_b=str(shard["bond_source_b"]),
    )


def test_offline_online_node_scalars_consistent():
    spec, _payload = load_spec_payload(FEATURES_DIR / "features_spec.json")
    s_g = float(spec.frozen_scales["s_g"])
    s_delta_e = float(spec.frozen_scales["s_delta_e"])
    scales = {"s_g": s_g, "s_delta_e": s_delta_e}

    shard_paths = sorted(glob.glob(str(FEATURES_DIR / "features_RXN_*.npz")))
    assert len(shard_paths) >= N_REACTIONS

    saw_terminal_predecessor = False
    saw_near_ts = False
    for shard_path in shard_paths[:N_REACTIONS]:
        rid = Path(shard_path).stem.removeprefix("features_")
        shard = np.load(shard_path)
        task = DualEndpointTask.from_payload(json.loads(
            (OBSERVED_DIR / f"{rid}.task.json").read_text(encoding="utf-8")))
        teacher = DualTSTeacher.from_payload(json.loads(
            (OBSERVED_DIR / f"{rid}.teacher.json").read_text(encoding="utf-8")))
        e_start = _episode_start_energy(teacher)
        labels = {l.state_id: l for l in teacher.labels}
        start = _rollout_start(task, shard)
        term_ids = {s.state_id for s in teacher.states
                    if s.boundary_type == "teacher_endpoint"}
        non_term = [s for s in teacher.states
                    if s.boundary_type != "teacher_endpoint"]

        for row, sid in enumerate(shard["state_id"]):
            state = next(s for s in teacher.states if s.state_id == str(sid))
            assert state.observation is not None
            delta_e = float(state.observation.energy_hartree) \
                - e_start[state.episode_id]
            grad = np.asarray(state.observation.gradient_hartree_per_angstrom,
                              dtype=np.float64)
            # mark coverage flags for the acceptance narrative
            if state.next_state_id in term_ids:
                saw_terminal_predecessor = True
            if (state in non_term and labels[str(sid)].remaining_arc_length
                    == min(labels[s.state_id].remaining_arc_length
                           for s in non_term)):
                saw_near_ts = True

            # OFFLINE rebuild
            offline = build_state_features(
                state_id=str(sid),
                atomic_numbers=state.atomic_numbers,
                coordinates=state.coordinates,
                gradient=grad,
                delta_e_hartree=delta_e,
                s_g=s_g, s_delta_e=s_delta_e,
                charge=int(task.charge), multiplicity=int(task.multiplicity))
            # ONLINE rebuild (mimic build_rollout_state_inputs wiring)
            inputs, _ = build_rollout_state_inputs(
                start=start, state_id=str(sid),
                coordinates=state.coordinates,
                gradient_hartree_per_angstrom=grad,
                delta_e_hartree=delta_e, scales=scales)

            shard_ns = np.asarray(shard["node_scalars"][row], dtype=np.float64)
            assert offline.node_scalars.shape == (int(shard["atomic_numbers"].size), 5)
            # exact float64
            np.testing.assert_array_equal(offline.node_scalars, shard_ns)
            np.testing.assert_array_equal(
                np.asarray(inputs["node_scalars"], dtype=np.float64),
                offline.node_scalars)
            # <= 1e-6 after float32 cast (state the tolerance)
            np.testing.assert_allclose(
                offline.node_scalars.astype(np.float32),
                shard_ns.astype(np.float32), atol=FLOAT32_TOL)
            np.testing.assert_allclose(
                np.asarray(inputs["node_scalars"], dtype=np.float64)
                .astype(np.float32),
                shard_ns.astype(np.float32), atol=FLOAT32_TOL)

    assert saw_terminal_predecessor, "sample never included a terminal predecessor"
    assert saw_near_ts, "sample never included a near-TS state"
