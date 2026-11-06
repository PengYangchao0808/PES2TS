"""WP4 shared synthetic fixtures for the V5.4.3 registry / assembly / cache /
trainer tests (CPU-fast; no data root, no checkpoint).

Everything here exists ONLY to drive the WP4 infrastructure:

* :class:`SynthMoveRefiner` / :class:`SynthProcessRefiner` are throw-away
  decoders whose ``sum(p.numel())`` lands INSIDE the frozen 34,200–37,800 band
  (plan §6.4 :563), so they can stand in for an M1/P1 candidate without
  touching WP5/WP6's real implementations.  ``MOVE = 49·620 + 8·620 = 35,340``
  and ``PROCESS = 64·160+160 + 160·160+160 + 160·2+2 = 36,482``.
* :class:`SynthSource` is a deterministic ``V543DataSource`` over
  ``scripts.agent_ops.feature_gate.v3_fixtures.make_v3_batch`` (the frozen
  collated-batch fixture), so the trainer exercises the REAL frozen loss
  functions on REAL forward-view tensors without a data root.
* :func:`labels_for` / :func:`endpoints_for` build the minimal label tables the
  frozen loss helpers read (``q``/``q_mask``/``direction``/``curve_id`` and
  ``endpoint_label``/``ts_row``/``start_row``)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F

from pes2ts_core.pesn.contracts.key_atom_v3 import forward_view_v3
from pes2ts_core.pesn.models.v543.assembly import assemble_v543
from pes2ts_core.pesn.models.v543.contracts import (
    MoveContextV1,
    ProcessContextV1,
    ProcessResidual,
)
from pes2ts_core.pesn.models.v543.model import V543Model
from pes2ts_core.pesn.models.v543.registry import (
    DecoderSpec,
    DecoderRegistry,
    TRACK_MOVE,
    TRACK_PROCESS,
)
from pes2ts_core.pesn.models.v543 import CoarseEncoder
from pes2ts_core.pesn.offline.training.v543_train import (
    TrainBatch,
    V543DataSource,
    V543_BATCH_FRAMES,
)
from scripts.agent_ops.feature_gate.v3_fixtures import make_v3_batch

# ---------------------------------------------------------------------------
# decoders whose parameter count is inside the frozen band
# ---------------------------------------------------------------------------
class SynthMoveRefiner(torch.nn.Module):
    """M1-shaped stand-in: ``δu = Σ_j c_j key_v_j`` (plan §6.2 :526 readout).

    49·620 + 8·620 = 35,340 trainable parameters — inside 34,200–37,800.
    """

    decoder_id = "M1"
    track = TRACK_MOVE
    model_version = "wp4_test_move_refiner_v1"

    def __init__(self, hidden: int = 620) -> None:
        super().__init__()
        self.trunk = torch.nn.Linear(48, hidden)
        self.coeff = torch.nn.Linear(hidden, 8, bias=False)

    def forward(self, ctx: MoveContextV1) -> torch.Tensor:
        hidden = F.silu(self.trunk(ctx.key_s))
        coeff = self.coeff(hidden)                       # [K,8]
        return (coeff.unsqueeze(-1) * ctx.key_v).sum(dim=1)   # [K,3]


class SynthProcessRefiner(torch.nn.Module):
    """P1-shaped stand-in: ``graph_s → 160 → 160 → 2`` (plan §6.3 :543).

    36,482 trainable parameters — inside 34,200–37,800.
    """

    decoder_id = "P1"
    track = TRACK_PROCESS
    model_version = "wp4_test_process_refiner_v1"

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = torch.nn.Linear(64, 160)
        self.fc2 = torch.nn.Linear(160, 160)
        self.head = torch.nn.Linear(160, 2)

    def forward(self, ctx: ProcessContextV1) -> ProcessResidual:
        hidden = F.silu(self.fc1(ctx.graph_s))
        hidden = F.silu(self.fc2(hidden))
        out = self.head(hidden)
        return ProcessResidual(delta_q_raw=out[:, 0], delta_e=out[:, 1])


def synth_move_spec() -> DecoderSpec:
    return DecoderSpec(
        decoder_id="M1", track=TRACK_MOVE,
        model_version="wp4_test_move_refiner_v1",
        factory=SynthMoveRefiner, equivariance="polar", stochastic=False,
        nfe=1, zero_init=False, status="smoke",
        acceptance="wp4_test_only")


def synth_process_spec() -> DecoderSpec:
    return DecoderSpec(
        decoder_id="P1", track=TRACK_PROCESS,
        model_version="wp4_test_process_refiner_v1",
        factory=SynthProcessRefiner, equivariance="invariant", stochastic=False,
        nfe=1, zero_init=False, status="smoke",
        acceptance="wp4_test_only")


def registry_with_synthetic_candidates() -> DecoderRegistry:
    from pes2ts_core.pesn.models.v543.registry import default_registry

    registry = default_registry()
    registry.register(synth_move_spec())
    registry.register(synth_process_spec())
    return registry


# ---------------------------------------------------------------------------
# batches
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SyntheticTables:
    """The minimal label tables the frozen loss helpers consume."""

    labels: Any
    endpoints: Any
    sample_ids: tuple[str, ...]

    def subset(self, sample_ids: Sequence[str]) -> "SyntheticTables":
        wanted = [str(s) for s in sample_ids]
        return SyntheticTables(
            labels=_LabelTable({sid: self.labels.by_sample[sid]
                                for sid in wanted}),
            endpoints=_EndpointTable({sid: self.endpoints.by_sample[sid]
                                      for sid in wanted}),
            sample_ids=tuple(wanted))


class _V52Label:
    __slots__ = ("sample_id", "curve_id", "direction", "q", "q_mask")

    def __init__(self, sample_id: str, curve_id: str, direction: str,
                 q: float | None, q_mask: bool) -> None:
        self.sample_id = sample_id
        self.curve_id = curve_id
        self.direction = direction
        self.q = q
        self.q_mask = q_mask


class _EndpointLabel:
    __slots__ = ("endpoint_label", "ts_row", "start_row")

    def __init__(self, endpoint_label: int, ts_row: bool,
                 start_row: bool) -> None:
        self.endpoint_label = int(endpoint_label)
        self.ts_row = bool(ts_row)
        self.start_row = bool(start_row)


class _LabelTable:
    def __init__(self, by_sample: Mapping[str, Any]) -> None:
        self.by_sample = dict(by_sample)


class _EndpointTable:
    def __init__(self, by_sample: Mapping[str, Any]) -> None:
        self.by_sample = dict(by_sample)


def labels_for(batch: Mapping[str, Any], *, q_step: float = 0.3,
               q_mask: bool = True) -> _LabelTable:
    sample_ids = [str(s) for s in batch["sample_ids"]]
    by: dict[str, Any] = {}
    for index, sample_id in enumerate(sample_ids):
        by[sample_id] = _V52Label(
            sample_id=sample_id, curve_id=str(batch["curve_ids"][index]),
            direction="forward", q=float(index) * float(q_step),
            q_mask=bool(q_mask))
    return _LabelTable(by)


def endpoints_for(batch: Mapping[str, Any], *,
                  labels: Sequence[int] | None = None) -> _EndpointTable:
    sample_ids = [str(s) for s in batch["sample_ids"]]
    by: dict[str, Any] = {}
    for index, sample_id in enumerate(sample_ids):
        value = (int(labels[index]) if labels is not None
                 else (1 if index % 2 else 0))
        by[sample_id] = _EndpointLabel(value, ts_row=(index == 0),
                                       start_row=(index == 1))
    return _EndpointTable(by)


def synthetic_tables(batch: Mapping[str, Any], *,
                     endpoint_labels: Sequence[int] | None = None
                     ) -> SyntheticTables:
    return SyntheticTables(
        labels=labels_for(batch), endpoints=endpoints_for(
            batch, labels=endpoint_labels),
        sample_ids=tuple(str(s) for s in batch["sample_ids"]))


def batch_with_status(status: Sequence[str], *, jitter_seed: int | None = 0,
                      n_graphs: int = 2) -> dict[str, Any]:
    """A collated fixture batch (4 graphs for ``n_graphs=2``)."""
    return make_v3_batch(n_graphs=int(n_graphs), with_teacher=True,
                         jitter_seed=jitter_seed)


class SynthSource(V543DataSource):
    """Deterministic in-memory data source (no data root, no dataset)."""

    def __init__(
            self,
            tables: SyntheticTables,
            *,
            label_status: Sequence[str] | None = None,
            repeat_batch: bool = False,
            n_graphs: int = 2,
            cache_keys: bool = True,
            n_val_batches: int = 2,
    ) -> None:
        self.tables = tables
        self.label_status = tuple(label_status or ("ok",) * 4)
        self.repeat_batch = bool(repeat_batch)
        self.n_graphs = int(n_graphs)
        self.cache_keys = bool(cache_keys)
        self.n_val_batches = int(n_val_batches)
        self.n_train = 64
        self.n_inner_val = V543_BATCH_FRAMES * int(n_val_batches)
        self._batches: dict[int, TrainBatch] = {}

    # -- helpers ------------------------------------------------------------
    def _make(self, update: int, *, split: str, jitter: int | None) -> TrainBatch:
        batch = make_v3_batch(n_graphs=self.n_graphs, with_teacher=True,
                              jitter_seed=jitter)
        sample_ids = tuple(str(s) for s in batch["sample_ids"])
        keys = (tuple(f"frame|{split}|{update}|{index}"
                      for index in range(len(sample_ids)))
                if self.cache_keys else ())
        return TrainBatch(
            positions=tuple(range(len(sample_ids))), batch=batch,
            sample_ids=sample_ids,
            curve_ids=tuple(str(c) for c in batch["curve_ids"]),
            label_status=tuple(self.label_status[:len(sample_ids)]),
            split=split, frame_keys=keys)

    # -- V543DataSource -----------------------------------------------------
    def draw_train_batch(self, update: int) -> TrainBatch:
        if self.repeat_batch:
            batch = self._batches.get(1)
            if batch is None:
                batch = self._make(0, split="inner_train", jitter=7)
                self._batches[1] = batch
            return batch
        return self._make(int(update), split="inner_train", jitter=int(update))

    def inner_val_batches(self, limit: int) -> list[TrainBatch]:
        keep = max(1, int(limit) // max(V543_BATCH_FRAMES, 1))
        count = min(self.n_val_batches, keep)
        return [self._make(9000 + index, split="inner_val",
                           jitter=9000 + index)
                for index in range(count)]

    def identity(self) -> dict[str, str]:
        return {
            "source_manifest_sha256": "a" * 64,
            "data_manifest_sha256": "b" * 64,
            "split_sha256": "c" * 64,
            "label_spec_sha256": "d" * 64,
        }

    @property
    def provides_cache_keys(self) -> bool:
        """The synthetic source can emit per-frame §5.7 keys (or not)."""
        return bool(self.cache_keys)


# ---------------------------------------------------------------------------
# assemblies / models / contexts
# ---------------------------------------------------------------------------
def make_encoder(state: Mapping[str, Any] | None = None) -> CoarseEncoder:
    """A frozen A0 encoder (57,619 params).

    ``state`` (optional) is a ``state_dict`` to load into it: a re-created or
    resumed run must carry the SAME encoder weights, otherwise the §7.4
    ``encoder_ckpt_sha256`` identity legitimately differs and the resume is
    refused — exactly what :func:`encoder_digest_v543` measures.
    """
    encoder = CoarseEncoder()
    if state is not None:
        encoder.load_state_dict(dict(state))
    return encoder


def make_assembly(*, track: str = TRACK_MOVE, encoder: Any | None = None,
                  decoder: Any | None = None, decoder_id: str | None = None,
                  **kwargs: Any):
    encoder = encoder if encoder is not None else make_encoder()
    if track == TRACK_MOVE:
        return assemble_v543(
            encoder, active_track=TRACK_MOVE, move_refiner=decoder,
            move_decoder_id=decoder_id, **kwargs)
    return assemble_v543(
        encoder, active_track=TRACK_PROCESS, process_refiner=decoder,
        process_decoder_id=decoder_id, **kwargs)


def make_model(*, track: str = TRACK_MOVE, decoder: Any | None = None,
               decoder_id: str | None = None,
               encoder: Any | None = None) -> V543Model:
    return V543Model(make_assembly(track=track, decoder=decoder,
                                   decoder_id=decoder_id, encoder=encoder))


def context_of(batch: Mapping[str, Any], *, track: str = TRACK_MOVE):
    """``(packet, context)`` of one fixture batch through the real encoder."""
    from pes2ts_core.pesn.models.v543 import LatentBridge

    view = forward_view_v3(dict(batch))
    encoder = make_encoder()
    packet = encoder(view)
    bridge = LatentBridge()
    return packet, bridge.build_context(packet, view)


__all__ = [
    "SynthMoveRefiner",
    "SynthProcessRefiner",
    "SynthSource",
    "SyntheticTables",
    "batch_with_status",
    "context_of",
    "endpoints_for",
    "labels_for",
    "make_assembly",
    "make_encoder",
    "make_model",
    "registry_with_synthetic_candidates",
    "synth_move_spec",
    "synth_process_spec",
    "synthetic_tables",
]
