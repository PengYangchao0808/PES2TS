"""Tiny self-contained E(n)-style dummy model for the feature-gate harness.

Distances -> MLP -> scalar + vector heads.  Equivariance is guaranteed by
construction:

* translation invariance — the forward pass only ever consumes interatomic
  difference vectors ``pos[j] - pos[i]`` and scalar distances;
* rotation equivariance — the vector head is a linear combination of edge
  difference vectors with invariant scalar coefficients;
* permutation equivariance — pure message passing with sum aggregation, no
  index-dependent parameters, and edge-aligned attributes that keep their
  edge ORDER (the harness only re-labels edge endpoints);
* NaN input raises :class:`~scripts.agent_ops.feature_gate.equivariance_harness.TypedInvalidError`
  (GeometryPolicyContract v1 S4.3) instead of fabricating an action.

The synthetic batch is intent-ON: it carries all five PD-6 intent classes
(``b_src``/``b_tgt``, ``delta_b_intent``, ``intent_role``, ``cv_target_delta``,
``signed_intent``/``direction_encoding``) plus the legal unordered chemical
context (``z``, ``edge_index``, ``graph_index``, ``b_abs``, ``is_changed``),
so :func:`~scripts.agent_ops.feature_gate.equivariance_harness.check_intent_off_mask`
has full 5/5 class coverage to verify.

Deliberate, switchable bug modes prove that the harness actually catches
violations.  Each switch below is engineered to break *exactly one* gate:

* ``use_raw_coords=True``       -> adds ``pos_i`` itself to the vector head:
  breaks **translation invariance** only (still rotates with R).
* ``use_atom_index=True``       -> scales ``u_i`` by ``(1 + 0.01 * i)`` with
  the *storage* index: breaks **permutation equivariance** only.
* ``use_endpoint_names=True``   -> rescales ``u`` by a factor depending on
  which display name is first: breaks **relabel invariance** only.
* ``ignore_intent=True``        -> drops the signed-intent factor so a sign
  flip leaves outputs unchanged: breaks **reverse sensitivity** only.

:func:`make_intent_off_batch` generates PD-6 intent-off variants
(``mode="full"`` is the legal ablation arm; ``"partial"``/``"nuke"`` are
deliberate PD-6 violations used to prove the mask check bites).
"""

from __future__ import annotations

from typing import Any, Mapping

import torch

from .equivariance_harness import TypedInvalidError

__all__ = [
    "DummyEquivariantModel",
    "INTENT_FIELD_KEYS",
    "make_dummy_batch",
    "make_dummy_model",
    "make_intent_off_batch",
]

#: keys carrying PD-6 intent information (must be zeroed when intent is off)
INTENT_FIELD_KEYS: tuple[str, ...] = (
    "b_src",              # ordered source/target (class 1)
    "b_tgt",              # ordered source/target (class 1)
    "delta_b_intent",     # signed graph edits (class 2)
    "intent_role",        # signed role of the edit (class 3)
    "cv_target_delta",    # directed CV descriptor (class 4)
    "signed_intent",      # implicit direction encoding (class 5)
    "direction_encoding", # implicit direction encoding (class 5)
)

#: unordered chemical topology/context PD-6 requires to be PRESERVED
PRESERVED_FIELD_KEYS: tuple[str, ...] = (
    "pos", "z", "edge_index", "graph_index", "b_abs", "is_changed",
)


class DummyEquivariantModel(torch.nn.Module):
    """E(n)-style dummy: invariant edge MLP, equivariant vector head."""

    def __init__(
        self,
        *,
        hidden: int = 16,
        use_raw_coords: bool = False,
        use_atom_index: bool = False,
        use_endpoint_names: bool = False,
        ignore_intent: bool = False,
    ) -> None:
        super().__init__()
        self.use_raw_coords = use_raw_coords
        self.use_atom_index = use_atom_index
        self.use_endpoint_names = use_endpoint_names
        self.ignore_intent = ignore_intent
        self.edge_mlp = torch.nn.Sequential(
            torch.nn.Linear(2 + 1 + 1 + 2, hidden),  # z_i,z_j,d,d^2 + 2 edge attrs
            torch.nn.SiLU(),
            torch.nn.Linear(hidden, hidden),
            torch.nn.SiLU(),
        )
        self.node_mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden, hidden),
            torch.nn.SiLU(),
        )
        self.coef_head = torch.nn.Linear(hidden, 1)    # signed edge coefficient
        self.scalar_head = torch.nn.Linear(hidden, 1)  # invariant scalar out

    def forward(self, batch: Mapping[str, Any]) -> dict[str, torch.Tensor]:
        pos = batch["pos"]
        if not torch.isfinite(pos).all():
            # GeometryPolicyContract v1 S4.3: typed invalid, no fabricated action.
            raise TypedInvalidError(
                "NAN_INPUT", "non-finite coordinates presented to model")
        z = batch["z"].to(dtype=pos.dtype)
        src = batch["edge_index"][0].long()
        dst = batch["edge_index"][1].long()
        dtype = pos.dtype

        rij = pos[dst] - pos[src]                       # [E,3] equivariant
        d = rij.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        # edge-aligned intent attributes are already ordered like edge_index
        # (the harness only re-labels edge ENDPOINTS, never re-orders edges),
        # so they are consumed per-edge -- never gathered by atom index.
        delta_b = batch.get("delta_b_intent")
        role = batch.get("intent_role")
        attr_i = (delta_b.to(dtype) if delta_b is not None
                  else torch.zeros(src.shape[0], dtype=dtype, device=pos.device))
        attr_j = (role.to(dtype) if role is not None
                  else torch.zeros(src.shape[0], dtype=dtype, device=pos.device))
        edge_in = torch.cat(
            [z[src].unsqueeze(-1) / 10.0,
             z[dst].unsqueeze(-1) / 10.0,
             d, d.square(),
             attr_i.unsqueeze(-1), attr_j.unsqueeze(-1)], dim=-1)
        m = self.edge_mlp(edge_in.to(dtype))            # [E,H] invariant

        n = pos.shape[0]
        h = torch.zeros(n, m.shape[-1], dtype=dtype, device=pos.device)
        h = h.index_add_(0, dst, m)
        h = h + self.node_mlp(h)

        coef = torch.tanh(self.coef_head(h[dst] - h[src]))  # [E,1] invariant
        u = torch.zeros_like(pos).index_add_(0, dst, coef * rij)

        key = batch.get("key_index")
        rows = (torch.arange(n, device=pos.device) if key is None
                else key.long())
        u = u[rows]

        # --- switchable bug modes (each breaks exactly one gate) ------------
        if self.use_raw_coords:
            u = u + pos[rows]
        if self.use_atom_index:
            scale = 1.0 + 0.01 * rows.to(dtype)
            u = u * scale.unsqueeze(-1)
        if self.use_endpoint_names:
            names = tuple(batch.get("endpoint_names", ("", "")))
            first = names[0] if names else ""
            u = u * (1.5 if first == "reactant" else 1.0)

        # signed intent (T1.2): a flip must CHANGE the outputs.  All other
        # intent fields are invariant scalars -> equivariance is preserved.
        for name in ("signed_intent", "direction_encoding", "cv_target_delta"):
            if name == "signed_intent" and self.ignore_intent:
                continue  # bug mode: model is blind to the signed intent
            val = batch.get(name)
            if val is not None:
                u = u * val.to(dtype)

        s = self.scalar_head(h).squeeze(-1)             # invariant scalar
        if key is not None:
            s = s[rows]
        return {"u": u, "s": s}


def make_dummy_model(*, seed: int = 0, dtype: torch.dtype = torch.float64,
                     **bug_flags: bool) -> DummyEquivariantModel:
    torch.manual_seed(seed)
    model = DummyEquivariantModel(**bug_flags)
    return model.to(dtype)


def make_dummy_batch(*, seed: int = 0, dtype: torch.dtype = torch.float64,
                     n_graphs: int = 2, atoms_per_graph: int = 6,
                     ) -> dict[str, Any]:
    """Deterministic synthetic intent-ON batch (two small molecules)."""
    gen = torch.Generator().manual_seed(seed)
    pos_list, z_list, edge_src, edge_dst, graph, keys = [], [], [], [], [], []
    db_list, role_list, abs_list, changed_list = [], [], [], []
    offset = 0
    for g in range(n_graphs):
        n = atoms_per_graph
        pos = torch.randn(n, 3, generator=gen, dtype=dtype) * 1.7
        # spread molecules apart so graph separation is meaningful
        pos = pos + torch.tensor([6.0 * g, 0.0, 0.0], dtype=dtype)
        z = torch.randint(1, 10, (n,), generator=gen)
        db, role, absb, chg = [], [], [], []
        for i in range(n):
            for j in range(n):
                if i != j:
                    edge_src.append(offset + i)
                    edge_dst.append(offset + j)
                    # signed intent edit in {-1, 0, +1} and its unsigned twin
                    signed = int(torch.randint(-1, 2, (1,), generator=gen))
                    db.append(float(signed))
                    role.append(float(1.0 if signed != 0 else 0.0))
                    absb.append(float(abs(signed)))
                    chg.append(float(signed != 0))
        pos_list.append(pos)
        z_list.append(z)
        db_list.extend(db)
        role_list.extend(role)
        abs_list.extend(absb)
        changed_list.extend(chg)
        graph.extend([g] * n)
        keys.extend([offset + 0, offset + 1, offset + 2])  # 3 key atoms / graph
        offset += n

    pos_all = torch.cat(pos_list)
    return {
        # --- current geometry / identity / graph (legal context) ------------
        "pos": pos_all,
        "z": torch.cat(z_list).long(),
        "edge_index": torch.stack([torch.tensor(edge_src, dtype=torch.long),
                                   torch.tensor(edge_dst, dtype=torch.long)]),
        "graph_index": torch.tensor(graph, dtype=torch.long),
        "key_index": torch.tensor(keys, dtype=torch.long),
        "b_abs": torch.tensor(abs_list, dtype=dtype),
        "is_changed": torch.tensor(changed_list, dtype=dtype),
        # --- reaction intent (PD-6 five classes, intent-ON) -----------------
        "b_src": torch.ones(len(edge_src), dtype=dtype),  # single-bond baseline
        "b_tgt": torch.tensor(db_list, dtype=dtype),
        "delta_b_intent": torch.tensor(db_list, dtype=dtype),
        "intent_role": torch.tensor(role_list, dtype=dtype),
        "cv_target_delta": torch.tensor(0.5, dtype=dtype),
        "signed_intent": torch.tensor(1.0, dtype=dtype),
        "direction_encoding": torch.tensor(1.0, dtype=dtype),
        # --- harness metadata -----------------------------------------------
        "endpoint_names": ("reactant", "product"),
    }


def _zero(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return torch.zeros_like(value)
    if isinstance(value, (list, tuple)):
        return type(value)(_zero(v) for v in value)
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        return type(value)(0)
    return value


def make_intent_off_batch(
    batch: Mapping[str, Any],
    *,
    mode: str = "full",
) -> dict[str, Any]:
    """Generate a PD-6 intent-off variant of an intent-ON batch.

    modes:

    * ``"full"``   — the legal ablation arm: ALL five intent classes zeroed,
      unordered chemical context preserved bit-identically.
    * ``"partial"``— deliberate PD-6 violation: only classes 1+3 masked, the
      signed edits / directed CV / implicit direction encodings leak.
    * ``"nuke"``   — deliberate PD-6 violation: five classes masked but the
      unordered context (``z``/``edge_index``/``b_abs``/``is_changed``) is
      destroyed too — masking must not nuke legal undirected context.
    """
    if mode not in ("full", "partial", "nuke"):
        raise ValueError(f"unknown intent-off mode {mode!r}")
    out: dict[str, Any] = dict(batch)
    if mode == "partial":
        masked = ("b_src", "b_tgt", "intent_role")
    else:
        masked = INTENT_FIELD_KEYS
    for key in masked:
        if key in out:
            out[key] = _zero(out[key])
    if mode == "nuke":
        for key in PRESERVED_FIELD_KEYS:
            if key in out and key != "pos":  # keep pos so the model still runs
                out[key] = _zero(out[key])
    return out
