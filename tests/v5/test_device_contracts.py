"""Device/precision + contract-boundary guarantees for the single trainer.

Covers:
* AMP guard semantics — ``amp.enabled == (amp.requested and torch.cuda.is_available())``,
  CPU disables with a documented reason, and the ``GradScaler`` path is only taken
  on CUDA (CUDA AMP smoke is skipped on CPU-only hosts);
* EMA weight-swap semantics — the EMA shadow updates during training, the applied
  weights differ from the raw snapshot, and the swap is reversible + reproducible;
* every batch crosses the shared contract boundary (``SequenceBatch`` when the
  data package ships contract v2, else the documented ``StandardBatch`` compat
  backend) with per-sample ``provenance_id``.
"""

from __future__ import annotations

import pytest
import torch

from pes2ts_v5.data.contracts import (
    contract_backend,
    contract_version,
    validate_sequence_batch,
)
from pes2ts_v5.training._util import weights_sha256

CUDA = torch.cuda.is_available()


# ---------------------------------------------------------------------------
# AMP guard
# ---------------------------------------------------------------------------

def test_amp_enabled_semantics(build_trainer) -> None:
    """``enabled`` is exactly ``requested and cuda_available`` (CPU stays float32)."""
    for requested in (False, True):
        trainer = build_trainer("gru", max_epochs=1, amp=requested)
        report = trainer.train()
        assert report["amp"]["requested"] is requested
        assert report["amp"]["enabled"] == (requested and CUDA)
        if requested and not CUDA:
            assert "CPU" in report["amp"]["reason"]
            assert trainer._scaler is None  # no GradScaler on CPU
        if not requested:
            assert trainer._scaler is None
            assert report["amp"]["reason"] == ""


def test_contract_backend_is_recorded(build_trainer) -> None:
    """The trainer reports which contract backend validated its batches."""
    trainer = build_trainer("gru", max_epochs=1)
    report = trainer.train()
    backend = contract_backend()
    assert backend in {"SequenceBatch", "StandardBatch"}
    assert report["contract"]["backend"] == backend
    assert report["contract"]["validated"] is True
    if contract_version():
        assert report["contract"]["version"] == contract_version()


@pytest.mark.skipif(not CUDA, reason="CUDA required for the AMP smoke (host is CPU-only)")
def test_cuda_amp_smoke(build_trainer) -> None:  # pragma: no cover - CPU-only host
    """On a CUDA host, requested AMP really enables autocast + GradScaler."""
    trainer = build_trainer("gru", max_epochs=1, device="cuda", amp=True)
    report = trainer.train()
    assert report["amp"]["enabled"] is True
    assert report["amp"]["requested"] is True
    assert trainer._scaler is not None
    assert torch.isfinite(torch.tensor(report["history"][0]["train_loss"]))


# ---------------------------------------------------------------------------
# EMA weight swap
# ---------------------------------------------------------------------------

def test_ema_shadow_updates_and_swap_is_reversible(build_trainer) -> None:
    trainer = build_trainer("gru", max_epochs=1, lr=0.05, ema_decay=0.5)
    report = trainer.train()
    assert report["ema"]["applied"] is True
    assert report["ema"]["raw_snapshot"] is True
    assert report["ema"]["reversible"] is True

    ema_state = trainer.ema_state()
    assert ema_state is not None
    applied_sha = weights_sha256(trainer.model)
    assert trainer.ema_applied is True

    # raw (pre-swap) weights exist, differ from the applied EMA weights
    assert trainer.restore_pre_ema_weights() is True
    assert trainer.ema_applied is False
    raw_sha = weights_sha256(trainer.model)
    assert raw_sha != applied_sha
    assert any(not torch.equal(ema_state[name], tensor)
               for name, tensor in trainer.model.state_dict().items())

    # re-applying the EMA gives exactly the same weights (reproducible)
    assert trainer.apply_ema() is True
    assert weights_sha256(trainer.model) == applied_sha

    # restoring again returns to the same raw weights (reversible)
    assert trainer.restore_pre_ema_weights() is True
    assert weights_sha256(trainer.model) == raw_sha


def test_ema_disabled_is_a_noop(build_trainer) -> None:
    trainer = build_trainer("gru", max_epochs=1)
    report = trainer.train()
    assert report["ema"]["applied"] is False
    assert trainer.ema_state() is None
    assert trainer.apply_ema() is False
    assert trainer.restore_pre_ema_weights() is False


# ---------------------------------------------------------------------------
# Contract boundary
# ---------------------------------------------------------------------------

def test_batches_are_contract_objects_with_provenance(build_trainer) -> None:
    trainer = build_trainer("gru", max_epochs=1)
    trainer.train()
    sequences = trainer.last_sequence_batches
    assert sequences
    assert trainer.last_sequence_batch is sequences[0]
    # contract v2 = one batch object per sample (per-sample identity/provenance);
    # the v1 compat backend carries the whole batch plus a provenance_id list.
    if contract_backend() == "SequenceBatch":
        assert len(sequences) == 2  # fixture batch size
        for item in sequences:
            token = getattr(item, "provenance", None)
            token = getattr(token, "token", token)
            assert isinstance(token, str) and token.startswith("traj:")
            assert getattr(item, "sample_id", None) == token
    else:  # pragma: no cover - only while contract v1 is installed
        assert len(sequences) == 1
        assert sequences[0].provenance_id
    # the batch was validated against the model's REQUIRED_FIELDS
    assert trainer.contract_fields
    report = validate_sequence_batch(sequences[0], trainer.contract_fields)
    assert report.get("provenance") or report.get("provenance_id")


def test_contract_validation_can_be_disabled_explicitly(build_trainer) -> None:
    """Escape hatch for environments without the data package (no silent skip)."""
    trainer = build_trainer("gru", max_epochs=1, validate_contract=False)
    report = trainer.train()
    assert report["contract"]["validated"] is False
    assert trainer.last_sequence_batch is None
