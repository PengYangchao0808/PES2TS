"""V5.4.3 WP6 — PROCESS-candidate training smoke through the WP4 trainer
(plan §7.2 :656–672, §6.3 :538–556, §11.5 :1236 ``test_v543_flow_no_reencode``
neighbours; WP6 task item 8).

What is pinned here: each of the five PROCESS candidates (P1/P2/P3/P4 and the
P4-C control) is *trainable* in the real single-track pipeline — 3 updates of
:class:`~pes2ts_core.pesn.offline.training.v543_train.V543Trainer` on the WP4
synthetic source:

* the run finishes with ``outcome == "done"`` and a FINITE ``best_val_loss``;
* every logged update reports a finite gradient (``grad_finite is True``),
  which is the numerical stability evidence a 36 K decoder needs before S1;
* the candidate's parameters actually MOVE (a candidate that silently produced
  a zero residual because of a detached path would fail this);
* the encoder and the inactive M0 decoder receive NO gradient (§7.2 :713–726);
* the PROCESS loss terms are the ones logged (``l_rank``/``l_q``/``l_endpoint``
  and no ``l_move``), i.e. the track really trains the PROCESS head (§7.2 :706).

The smoke is a SMOKE: ``stop_after=3`` is the trainer's own honest short leg
(``V543TrainerConfig`` refuses a reduced flat dose, §7.2 :659)."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from pes2ts_core.pesn.offline.training.v543_train import (  # noqa: E402
    V543Trainer,
    V543TrainerConfig,
)
from pes2ts_core.pesn.models.v543.process import (  # noqa: E402
    MEASURED_PARAMS_V543,
    PROCESS_CANDIDATE_IDS_V543,
    build_process_candidate,
)
from tests.pesn._v543_fixtures_wp4 import SyntheticTables, SynthSource  # noqa: E402
from tests.pesn._v543_fixtures_wp4 import (  # noqa: E402
    batch_with_status,
    synthetic_tables,
)
from tests.pesn._v543_fixtures_wp6 import process_model  # noqa: E402

#: Three updates: the trainer's own smoke leg length (``stop_after``).
SMOKE_UPDATES = 3


def _source(tables: SyntheticTables) -> SynthSource:
    return SynthSource(tables, label_status=("ok",) * 4, repeat_batch=False)


@pytest.mark.parametrize("decoder_id", PROCESS_CANDIDATE_IDS_V543)
def test_candidate_trains_for_three_updates(tmp_path: Path,
                                           decoder_id: str) -> None:
    """3 updates, finite loss, live gradients, PROCESS-only loss terms."""
    batch = batch_with_status(("ok",) * 4, jitter_seed=17)
    tables = synthetic_tables(batch, endpoint_labels=(1, 0, 1, 0))
    source = _source(tables)
    decoder = build_process_candidate(decoder_id)
    model = process_model(decoder, decoder_id=decoder_id)
    config = V543TrainerConfig(track="process", decoder_id=decoder_id,
                               seed=17, stop_after=SMOKE_UPDATES)
    before = [p.detach().clone() for p in decoder.parameters()]
    trainer = V543Trainer(
        model=model, config=config, source=source,
        labels_v52=tables.labels, endpoint_labels=tables.endpoints,
        out_dir=tmp_path / decoder_id.replace("-", "_"))
    metrics = trainer.run()

    # -- a real, finished run -------------------------------------------------
    assert metrics["outcome"] == "done"
    assert metrics["updates"] == SMOKE_UPDATES
    assert metrics["is_formal_schedule"] is False
    assert metrics["skipped_updates"] == 0
    assert math.isfinite(float(metrics["best_val_loss"]))

    # -- the decoder is really being optimised --------------------------------
    rows = [json.loads(line) for line in
            (tmp_path / decoder_id.replace("-", "_") / "train_log.jsonl")
            .read_text().splitlines()]
    assert len(rows) == SMOKE_UPDATES
    assert all(row["event"] == "train" for row in rows)
    assert all(row["grad_finite"] is True for row in rows)
    for row in rows:
        assert math.isfinite(float(row["loss"]))
        for key in ("l_rank", "l_q", "l_endpoint"):
            assert key in row, key
            assert math.isfinite(float(row[key]))
        assert "l_move" not in row, "the PROCESS track must not call MOVE loss"
    after = [p.detach().clone() for p in decoder.parameters()]
    assert any(not torch.equal(a, b) for a, b in zip(before, after)), \
        f"{decoder_id} parameters did not move in 3 updates"

    # -- §7.2 :713–726 isolation ---------------------------------------------
    for row in rows:
        isolated = row["isolation"]
        assert isolated["encoder_params_with_grad"] == 0
        assert isolated["inactive_params_with_grad"] == 0
        assert isolated["active_params_with_grad"] > 0
    summary = metrics["gradient_isolation"]
    assert summary["encoder_requires_grad"] is False
    assert summary["inactive_requires_grad"] is False
    assert summary["active_trainable_params"] > 0
    for parameter in model.assembly.encoder.parameters():
        assert parameter.grad is None
    for parameter in model.assembly.move.decoder.parameters():
        assert parameter.grad is None

    # -- the registered measured count is still the reported one -------------
    assert metrics["params"]["active"]["n_trainable_params"] == \
        MEASURED_PARAMS_V543[decoder_id]
    assert metrics["params"]["active"]["decoder_id"] == decoder_id
