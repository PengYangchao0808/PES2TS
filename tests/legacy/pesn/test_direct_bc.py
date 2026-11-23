"""Direct BC trainer smoke on synthetic transitions (CPU, tiny model)."""
from __future__ import annotations

import numpy as np
import pytest

from pes2ts_core.pesn.offline.data.direct_dataset import (
    DirectSample,
    DirectTransitionDataset,
)
from pes2ts_core.pesn.offline.training.direct_bc import (
    DirectBCSettings,
    DirectBCTrainer,
    sample_to_feature_inputs,
)


TASK_GRAPH = {"edits": [{"kind": "broken", "i": 0, "j": 1},
                        {"kind": "formed", "i": 1, "j": 2}]}


def _sample(rid: str, step: int, n: int = 3) -> DirectSample:
    rng = np.random.default_rng(hash((rid, step)) % 2**31)
    coords = np.array([[0.0, 0, 0], [1.1 + 0.01 * step, 0, 0], [2.5, 0, 0]][:n])
    action = np.zeros((n, 3))
    action[1, 0] = 0.05
    # keep the per-atom norm within 0.9 * a_max (a_max = 0.1 in these tests)
    action = action * (0.09 / max(1e-9, np.linalg.norm(action, axis=1).max()))
    return DirectSample(
        reaction_id=rid, group_id=f"G-{rid}", subset="synthetic", step_index=step,
        atomic_numbers=np.array([6, 1, 6][:n]), coords=coords,
        gradient=rng.normal(scale=0.02, size=(n, 3)),
        energy=-10.0 + step * 1e-3, delta_e=-0.05, steps_remaining=100 - step,
        event_state=np.array([0.0, 0.0, -0.3]), action=action,
        reward=-0.01, terminated=step == 4, bootstrap=0.0 if step == 4 else 1.0,
        td_eligible=True, termination_reason="continued", task_graph=TASK_GRAPH,
        charge=0, multiplicity=1,
    )


class _MemDataset(DirectTransitionDataset):
    """Duck-typed minimal dataset (no shards on disk)."""

    def __init__(self, reactions_per_group: int = 3, n_groups: int = 4):  # noqa: D107
        self._data: dict[str, list[DirectSample]] = {}
        for g in range(n_groups):
            rid = f"RXN_SYN{g:04d}"
            self._data[rid] = [_sample(rid, s) for s in range(5)]
        self._subsets = {rid: "synthetic" for rid in self._data}

    @property
    def reaction_ids(self) -> list[str]:
        return sorted(self._data)

    @property
    def subsets(self) -> dict[str, str]:
        return dict(self._subsets)

    def reaction_ids_for_subset(self, subset: str) -> list[str]:
        return self.reaction_ids

    def get_reaction(self, reaction_id: str) -> list[DirectSample]:
        return self._data[reaction_id]


@pytest.mark.filterwarnings("ignore")
def test_bc_trainer_runs_and_loss_finite_on_cpu():
    dataset = _MemDataset()
    settings = DirectBCSettings(
        max_epochs=2, patience=2, seed=7, batch_reactions=2,
        hidden=16, n_layers=1, device="cpu", require_cuda=False,
        cpu_intra_threads=1, cpu_inter_threads=1, sigma_range=(0.05, 1.0))
    trainer = DirectBCTrainer(
        settings=settings, train_dataset=dataset, monitor_dataset=dataset,
        run_dir="/tmp/opencode/bc_smoke_test", config_sha256="cfg-test",
        dataset_manifest_sha256="ds-test")
    result = trainer.train()
    assert all(
        np.isfinite(entry["train_group_nll"]) for entry in result["history"])
    assert result["model_summary"]["n_parameters"] > 0
    trainer.write_run_record()
    import json
    from pathlib import Path
    record = json.loads(Path("/tmp/opencode/bc_smoke_test/run_record.json").read_text())
    assert record["trainer"] == "direct_bc"
    assert Path("/tmp/opencode/bc_smoke_test/checkpoint_last.pt").is_file()


def test_sample_to_feature_inputs_builds():
    sample = _sample("RXN_SYN0000", 0)
    features = sample_to_feature_inputs(sample)
    assert features.n_atoms == 3
    assert features.edge_task_roles.sum() > 0
    assert np.isfinite(features.graph_scalars).all()
