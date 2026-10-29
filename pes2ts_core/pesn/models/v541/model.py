"""V5.4.1 ``V541Model`` — legacy V5.3-MP model + gated residual embedding
(WP2, plan §5 / ADR-0032).

Design (the WP2-frozen mechanics):

* **Submodule layout is the legacy one**: children ``backbone``
  (``StarEGNNV3``) and ``process_head`` (``ReactionProcessHeadV53MP``)
  with their exact legacy state-dict names.  ``V541Model`` SUBCLASSES
  ``KeyAtomV53MPModel``; E0 forward is literally ``super().forward``
  (zero new parameters, zero extra RNG, five outputs bitwise identical —
  plan §5.3 certificate items 1–5).
* **Unified residual** (plan §5.2)::

      h_v541 = h_legacy + alpha * W * f_phi(h_legacy, V_legal),  alpha_0 = 0

  ``W`` = ``w_out``: ``nn.Linear(48, 48, bias=False)`` with a NON-zero
  (orthogonal) init; ``alpha`` is a single 0-dim scalar Parameter zero-
  initialized.  E0 has neither — ``e0_has_no_branch: true``.
* **Controlled ``node_proj`` forward-hook injection** — the three-layer
  core loop of ``StarEGNNV3`` is NOT copied.  For a branch model the
  forward: (1) enforces the forward-view whitelist and mirrors ONLY the
  legacy node-init line to obtain ``h_legacy``; (2) computes the branch
  residual ``alpha * W * f``; (3) registers a temporary forward hook on
  ``backbone.node_proj`` that adds the residual to its output; (4) runs
  the ORIGINAL ``KeyAtomV53MPModel.forward`` (backbone + process head);
  (5) ``finally`` removes the hook — always, including on typed-invalid
  aborts.  With ``alpha == 0`` the added tensor is exactly zeros, so the
  five outputs are bitwise-equal to legacy; ``alpha`` receives a nonzero
  gradient while ``W``/branch grads are zero for the first step (they are
  multiplied by ``alpha``), which is the EXPECTED cold-start behavior —
  after ``alpha`` moves off zero the whole branch receives gradients.
* **Independent init RNG** (plan §5.4): branch + ``W`` are constructed
  inside ``torch.random.fork_rng`` seeded from
  ``embedding_init_seed`` via the CPU default generator, so the legacy
  RNG sequence of a same-seed legacy build is bit-for-bit preserved.
  A branch build WITHOUT an explicit seed is a typed refusal
  (``V541_INIT_SEED_REQUIRED``).
* **Legality**: non-E0 reads go through the frozen v3 forward-view
  whitelist first (supervision/metadata keys are refused with the same
  ``FORWARD_VIEW_LEAK`` typed error as the legacy model, so they can
  never influence an output); the branch is fed only whitelisted
  forward-view tensors.  ``node_vector`` returned by a candidate is
  recorded as detached diagnostics only — never consumed by the core —
  and a candidate returning ``edge_scalar`` is refused (the core always
  receives the legacy 22-channel edge input in legacy order).
* **WP3 surface**: registry (:mod:`...v541.embedding_registry`), typed
  errors + ``NodeDeltaEmbedding`` + diagnostics helpers
  (:mod:`...v541.interfaces`), :meth:`V541Model.diagnostics`,
  :meth:`V541Model.audit_legacy_state_dict` /
  :meth:`V541Model.load_legacy_state_dict` (explicit strict-load audit:
  missing/unexpected/shape/dtype), :meth:`V541Model.legacy_state_dict`.

E1–E5 are NOT implemented here — constructing a branch model with those
ids raises ``V541EmbeddingNotImplementedError`` (typed refusal, no
placeholder math)."""

from __future__ import annotations

from typing import Final, Mapping

import torch
import torch.nn as nn
import torch.utils.hooks

from pes2ts_core.pesn.contracts.key_atom_v3 import (
    EDGE_INDEX_KEY,
    GRAPH_CV_DELTA_KEY,
    NODE_DEGK_KEY,
    NODE_KEY_SET_KEY,
    NODE_POS_KEY,
    NODE_Z_KEY,
    NODE_INTENT_ROLE_KEY,
    NODE_IS_KEY_KEY,
    check_forward_view_keys_v3,
)
from pes2ts_core.pesn.models.key_atom_egnn_v3 import (
    KeyAtomV3InvalidInputError,
    StarEGNNV3,
)
from pes2ts_core.pesn.models.key_atom_heads_v3 import (
    KeyAtomV53MPModel,
)
from pes2ts_core.pesn.models.v541.embedding_registry import (
    build_embedding,
    embedding_spec,
    is_embedding_implemented,
    is_embedding_known,
)
from pes2ts_core.pesn.models.v541.interfaces import (
    EMBEDDING_ID_E0,
    HIDDEN_V541,
    V541_SCHEMA_VERSION,
    NodeDeltaEmbedding,
    V541InitError,
    V541LoadError,
    count_scalar,
    detached_rms,
    detached_scalar,
    normalize_branch_output,
)

#: State-dict key prefixes that belong to the LEGACY (pre-V541) build.
LEGACY_STATE_PREFIXES: Final[tuple[str, ...]] = ("backbone.", "process_head.")

#: Max atomic number accepted by the frozen ``nn.Embedding(119, 16)``.
_Z_MAX: Final[int] = 118


def _extract_state_dict(payload: object) -> Mapping[str, torch.Tensor]:
    """Accept a raw state dict OR a trainer checkpoint payload."""
    if not isinstance(payload, Mapping):
        raise V541LoadError(
            "V541_LOAD_PAYLOAD",
            f"checkpoint payload must be a mapping, got {type(payload).__name__}")
    if "model" in payload and isinstance(payload["model"], Mapping):
        payload = payload["model"]
    for key, value in payload.items():
        if not isinstance(value, torch.Tensor):
            raise V541LoadError(
                "V541_LOAD_PAYLOAD",
                f"state entry {key!r} is {type(value).__name__}, expected a tensor")
    return payload  # type: ignore[return-value]


class V541Model(KeyAtomV53MPModel):
    """Legacy V5.3-MP model + optional gated residual embedding (plan §5).

    Args:
        backbone: existing ``StarEGNNV3`` shared BY REFERENCE (same policy
            as ``KeyAtomV53MPModel``); ``None`` builds a fresh one.
        embedding_id: ``"E0"`` (default — exact legacy behavior, no
            branch) or a registered candidate id.  Reserved candidates
            E1–E6 raise the typed not-implemented refusal until WP3.
        embedding_init_seed: REQUIRED for branch builds; the branch and
            ``W`` are initialized under ``torch.random.fork_rng`` with
            this seed so the legacy RNG stream is untouched (plan §5.4).
    """

    #: Registered model version.
    v541_version: Final[str] = V541_SCHEMA_VERSION

    def __init__(
            self,
            backbone: StarEGNNV3 | None = None,
            *,
            embedding_id: str = EMBEDDING_ID_E0,
            embedding_init_seed: int | None = None,
    ) -> None:
        # Legacy construction FIRST: identical submodule names, identical
        # RNG consumption, identical legacy parameter set (plan §5.4 step 1).
        super().__init__(backbone)
        self.embedding_id = str(embedding_id)
        # Unknown id → typed refusal before any branch work.
        if self.embedding_id != EMBEDDING_ID_E0 and not is_embedding_known(self.embedding_id):
            embedding_spec(self.embedding_id)  # always raises: unknown id
        # Reserved-but-unimplemented candidates (E1–E6 until WP3) refuse
        # HERE — before any seed/RNG work — never a placeholder branch.
        if (self.embedding_id != EMBEDDING_ID_E0
                and not is_embedding_implemented(self.embedding_id)):
            build_embedding(self.embedding_id)  # always raises: not implemented

        self._diagnostics: dict[str, float] = {}
        # Scalar diagnostics synchronize with the host; WP4 disables them
        # except on registered monitor/profiling steps so they are not
        # charged as a per-update candidate-only overhead.
        self.diagnostics_enabled: bool = True
        if self.embedding_id == EMBEDDING_ID_E0:
            # E0: NO branch, NO W, NO alpha — state dict == legacy exactly.
            self.embedding: NodeDeltaEmbedding | None = None
            self.w_out: nn.Linear | None = None
            self.alpha: nn.Parameter | None = None
            return

        if embedding_init_seed is None:
            raise V541InitError(
                "V541_INIT_SEED_REQUIRED",
                f"embedding {self.embedding_id!r} requires an explicit "
                f"embedding_init_seed so branch init runs on an independent, "
                f"reproducible RNG and never consumes the legacy sequence "
                f"(plan §5.4 step 3)")
        seed = int(embedding_init_seed)
        # fork_rng saves/restores the CPU generator; seeding happens on
        # default_generator only (no CUDA state touched — construction is
        # on CPU), so the legacy RNG stream continues bit-for-bit unchanged.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            branch = build_embedding(self.embedding_id)   # typed refusals here
            w_out = nn.Linear(HIDDEN_V541, HIDDEN_V541, bias=False)
            nn.init.orthogonal_(w_out.weight)             # NON-zero init (§5.2)
            assert branch is not None and w_out is not None
            self.embedding = branch
            self.w_out = w_out
            # alpha: single scalar gate, zero init, constructed WITHOUT RNG
            # (torch.zeros does not draw from the global stream).
            self.alpha = nn.Parameter(torch.zeros(()))
        if self.alpha is None or self.w_out is None or self.embedding is None:  # pragma: no cover
            raise V541InitError(
                "V541_INIT_BRANCH", "branch construction left a None part")

    # -- legacy view helpers ------------------------------------------------
    @property
    def has_branch(self) -> bool:
        """True when a residual candidate (non-E0) is attached."""
        return self.embedding is not None

    @property
    def is_e0(self) -> bool:
        return self.embedding_id == EMBEDDING_ID_E0

    def legacy_state_dict(self) -> dict[str, torch.Tensor]:
        """The legacy-only view (``backbone.*`` + ``process_head.*``).

        For E0 this IS the full state dict; WP4 uses it to hash/save the
        legacy core before any branch init (plan §5.4 step 2).
        """
        return {
            k: v for k, v in self.state_dict().items()
            if k.startswith(LEGACY_STATE_PREFIXES)}

    # -- branch math --------------------------------------------------------
    def _compute_h_legacy(self, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """Mirror ONLY the frozen legacy node-init line (plan §5.2).

        ``StarEGNNV3.forward``::

            h = node_proj(cat[elem_emb(z), cat[is_key, degK/4, role]])

        No core-loop code is duplicated: this is the same module calls in
        the same order on the same tensors, so ``h_legacy`` is bitwise the
        value the backbone's own ``node_proj`` will produce.
        """
        backbone = self.backbone
        dtype = backbone.node_proj.weight.dtype
        device = backbone.node_proj.weight.device

        z_atomic = batch[NODE_Z_KEY]
        # Same typed-invalid range checks as StarEGNNV3.forward — raised
        # BEFORE elem_emb so an out-of-range Z can never become an IndexError.
        if int(z_atomic.numel()) and int(z_atomic.max()) > _Z_MAX:
            raise KeyAtomV3InvalidInputError(
                "Z_OUT_OF_RANGE", "z > 118 (padding/unknown Z rejected)")
        if int(z_atomic.numel()) and int(z_atomic.min()) < 1:
            raise KeyAtomV3InvalidInputError(
                "Z_PADDING", "z < 1 (padding rows are not valid input)")

        is_key = batch[NODE_IS_KEY_KEY].to(dtype=dtype)
        degk = batch[NODE_DEGK_KEY].to(dtype=dtype)
        role = batch[NODE_INTENT_ROLE_KEY].to(dtype=dtype)
        degk_norm = degk.clamp(min=0.0, max=4.0) / 4.0
        return backbone.node_proj(torch.cat(
            [backbone.elem_emb(z_atomic.to(device=device)),
             torch.cat([is_key.unsqueeze(-1), degk_norm.unsqueeze(-1), role],
                       dim=-1)], dim=-1))

    def _branch_residual(
            self, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """Whitelist → ``h_legacy`` → ``f`` → ``alpha * W * f`` (+ diagnostics)."""
        # Forward-view whitelist FIRST: supervision/metadata keys are
        # refused exactly like the legacy model (FORWARD_VIEW_LEAK), so
        # they can never influence h_legacy or any output (plan §5.3 item 6).
        check_forward_view_keys_v3(batch)
        h_legacy = self._compute_h_legacy(batch)

        assert self.embedding is not None and self.w_out is not None
        raw = self.embedding(h_legacy, batch)
        delta, node_vector = normalize_branch_output(
            raw, n_nodes=int(h_legacy.shape[0]), hidden=HIDDEN_V541,
            ref_dtype=h_legacy.dtype)

        assert self.alpha is not None
        residual = self.w_out(delta) * self.alpha   # [N,48]; alpha=0 → zeros

        # Detached-scalar diagnostics ONLY — never store tensors/graphs.
        if self.diagnostics_enabled:
            self._diagnostics = {
                "n_nodes": count_scalar(batch[NODE_POS_KEY]),
                "n_edges": float(batch[EDGE_INDEX_KEY].shape[1]),
                "n_graphs": count_scalar(batch[GRAPH_CV_DELTA_KEY]),
                "n_keys": count_scalar(batch[NODE_KEY_SET_KEY]),
                "alpha": detached_scalar(self.alpha),
                "rms_h_legacy": detached_rms(h_legacy),
                "rms_delta_node": detached_rms(delta),
                "rms_alpha_W_f": detached_rms(residual),
                "rms_node_vector": (detached_rms(node_vector)
                                    if node_vector is not None else 0.0),
                # plan §5.2: the fixed core does not consume node_vector.
                "node_vector_consumed": 0.0,
            }
        else:
            self._diagnostics = {}
        return residual

    @staticmethod
    def _install_residual_hook(
            node_proj: nn.Linear, residual: torch.Tensor) -> torch.utils.hooks.RemovableHandle:
        """``node_proj output += residual`` for ONE backbone call."""
        def _hook(module: nn.Module, inputs: tuple, output: torch.Tensor) -> torch.Tensor:
            return output + residual
        return node_proj.register_forward_hook(_hook)

    # -- forward ------------------------------------------------------------
    def forward(self, batch: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Five outputs of contract §3.2 — E0 is ``super().forward`` verbatim."""
        if not self.has_branch:
            return super().forward(batch)
        residual = self._branch_residual(batch)
        assert isinstance(self.backbone.node_proj, nn.Linear)
        handle = self._install_residual_hook(self.backbone.node_proj, residual)
        try:
            # ORIGINAL legacy forward (backbone core loop + process head)
            # runs with the hook installed; head math is untouched.
            return super().forward(batch)
        finally:
            handle.remove()          # hook cleanup, even on typed-invalid abort

    def forward_move(self, batch: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Backbone-only view with the same gate semantics as :meth:`forward`.

        E0: byte-identical to the legacy ``forward_move`` (direct backbone
        call).  Branch: identical hook lifecycle around the backbone call.
        """
        if not self.has_branch:
            return self.backbone(batch)
        residual = self._branch_residual(batch)
        assert isinstance(self.backbone.node_proj, nn.Linear)
        handle = self._install_residual_hook(self.backbone.node_proj, residual)
        try:
            return self.backbone(batch)
        finally:
            handle.remove()

    # -- diagnostics (WP3/WP4 monitoring surface) ---------------------------
    @property
    def diagnostics(self) -> dict[str, float]:
        """Detached scalar diagnostics of the LAST forward (plain floats).

        Empty for E0 (no branch).  Values are Python floats — no tensors,
        no autograd graph is ever retained here.
        """
        return dict(self._diagnostics)

    # -- strict-load certificate (plan §5.3 item 1) -------------------------
    def audit_legacy_state_dict(
            self, payload: object) -> dict[str, object]:
        """Explicit audit of a legacy V5.3-MP checkpoint against this model.

        Returns a JSON-able report::

            {"missing": [...], "unexpected": [...],
             "shape_mismatch": [(key, model_shape, ckpt_shape), ...],
             "dtype_mismatch": [(key, model_dtype, ckpt_dtype), ...],
             "ok": bool}

        ``missing``/``unexpected`` are key-level; shape/dtype are
        tensor-level (the certificate requires ALL FOUR empty for
        ``strict=True``).
        """
        state = _extract_state_dict(payload)
        model_state = self.state_dict()
        missing = sorted(set(model_state) - set(state))
        unexpected = sorted(set(state) - set(model_state))
        shape_mismatch: list[tuple[str, tuple[int, ...], tuple[int, ...]]] = []
        dtype_mismatch: list[tuple[str, str, str]] = []
        for key in sorted(set(model_state) & set(state)):
            ref, got = model_state[key], state[key]
            if tuple(ref.shape) != tuple(got.shape):
                shape_mismatch.append(
                    (key, tuple(ref.shape), tuple(got.shape)))
            if ref.dtype != got.dtype:
                dtype_mismatch.append((key, str(ref.dtype), str(got.dtype)))
        ok = not (missing or unexpected or shape_mismatch or dtype_mismatch)
        return {
            "missing": missing,
            "unexpected": unexpected,
            "shape_mismatch": shape_mismatch,
            "dtype_mismatch": dtype_mismatch,
            "ok": bool(ok),
        }

    def load_legacy_state_dict(
            self, payload: object, *, strict: bool = True) -> dict[str, object]:
        """Load a legacy checkpoint with an EXPLICIT audit before load.

        ``strict=True`` (default; the WP2 certificate path) refuses unless
        missing/unexpected/shape/dtype are ALL clean, then delegates to
        ``nn.Module.load_state_dict(strict=True)``.  For E0 that is the
        plain legacy load (state keys are identical).  A branch model
        loading a legacy checkpoint under ``strict=True`` correctly
        refuses (its branch keys are missing — WP3 initializes those via
        the independent seeded fork_rng path, plan §5.4 step 3).
        """
        state = _extract_state_dict(payload)
        audit = self.audit_legacy_state_dict(state)
        if strict and not audit["ok"]:
            raise V541LoadError(
                "V541_LOAD_AUDIT",
                "strict load refused — missing={missing} unexpected={unexpected} "
                "shape_mismatch={shape} dtype_mismatch={dtype}".format(
                    missing=audit["missing"], unexpected=audit["unexpected"],
                    shape=[m[0] for m in audit["shape_mismatch"]],
                    dtype=[m[0] for m in audit["dtype_mismatch"]]))
        if audit["shape_mismatch"] or audit["dtype_mismatch"]:
            # torch's load_state_dict hard-fails on size/dtype mismatch even
            # with strict=False — surface it as our typed error instead.
            raise V541LoadError(
                "V541_LOAD_SHAPE",
                "shape/dtype mismatch cannot be loaded under any strict mode "
                "— shape={shape} dtype={dtype}".format(
                    shape=[m[0] for m in audit["shape_mismatch"]],
                    dtype=[m[0] for m in audit["dtype_mismatch"]]))
        result = super().load_state_dict(state, strict=strict)
        return {
            "audit": audit,
            "missing_keys": list(result.missing_keys),
            "unexpected_keys": list(result.unexpected_keys),
        }

    # -- accounting ---------------------------------------------------------
    @property
    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def head_parameter_counts(self) -> dict[str, int]:
        """Per-head counts; E0 output is IDENTICAL to the legacy method."""
        counts = super().head_parameter_counts()
        if self.has_branch:
            counts["embedding"] = sum(
                p.numel() for p in self.embedding.parameters())  # type: ignore[union-attr]
            counts["w_out"] = int(self.w_out.weight.numel())     # type: ignore[union-attr]
            counts["alpha"] = int(self.alpha.numel())            # type: ignore[union-attr]
        return counts


def measured_params_v541_e0() -> int:
    """Exact ``sum(p.numel())`` of the default E0 build (= legacy joint)."""
    return V541Model().n_parameters


__all__ = [
    "LEGACY_STATE_PREFIXES",
    "V541Model",
    "measured_params_v541_e0",
]
