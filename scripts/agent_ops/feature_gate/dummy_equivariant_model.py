"""Tiny self-contained E(n)-style dummy model for the feature-gate harness.

Distances -> MLP -> scalar + vector heads.  Equivariance is guaranteed by
construction:

* translation invariance — the forward pass only ever consumes interatomic
  difference vectors ``pos[j] - pos[i]`` and scalar distances;
* rotation equivariance — the vector head is a linear combination of edge
  difference vectors with invariant scalar coefficients;
* permutation equivariance — pure message passing with sum aggregation and
  no index-dependent parameters;
* NaN input raises :class:`~scripts.agent_ops.feature_gate.equivariance_harness.TypedInvalidError`
  (GeometryPolicyContract v1 S4.3) instead of fabricating an action.

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
"""

from __future__ import annotations

from typing import Any, Mapping

import torch

from .equivariance_harness import TypedInvalidError

__all__ = ["DummyEquivariantModel", "make_dummy_batch", "make_dummy_model"]


class DummyEquivariantModel(torch.nn.Module):
    """E(n)-style dummy: invariant edge MLP, equivariant vector head."""

    def __init__(
        self,
        *,
        hidden: int = 16,
        n_z: int = 10,
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
            torch.nn.Linear(2 + 1 + 1, hidden),   # z_i, z_j (scaled), d, d^2
            torch.nn.SiLU(),
            torch.nn.Linear(hidden, hidden),
            torch.nn.SiLU(),
        )
        self.node_mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden, hidden),
            torch.nn.SiLU(),
        )
        self.coef_head = torch.nn.Linear(hidden, 1)   # signed edge coefficient
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
        edge_in = torch.cat(
            [z[src].unsqueeze(-1) / 10.0,
             z[dst].unsqueeze(-1) / 10.0,
             d, d.square()], dim=-1)
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

        # signed intent (T1.2): a flip must CHANGE the outputs
        intent = batch.get("signed_intent")
        if intent is not None and not self.ignore_intent:
            u = u * intent.to(dtype)

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
    """Deterministic synthetic batch: two small molecules, full intra-graph edges."""
    gen = torch.Generator().manual_seed(seed)
    pos_list, z_list, edge_src, edge_dst, graph, keys = [], [], [], [], [], []
    offset = 0
    for g in range(n_graphs):
        n = atoms_per_graph
        pos = torch.randn(n, 3, generator=gen, dtype=dtype) * 1.7
        # spread molecules apart so graph separation is meaningful
        pos = pos + torch.tensor([6.0 * g, 0.0, 0.0], dtype=dtype)
        z = torch.randint(1, 10, (n,), generator=gen)
        for i in range(n):
            for j in range(n):
                if i != j:
                    edge_src.append(offset + i)
                    edge_dst.append(offset + j)
        pos_list.append(pos)
        z_list.append(z)
        graph.extend([g] * n)
        keys.extend([offset + 0, offset + 1, offset + 2])  # 3 key atoms / graph
        offset += n

    pos_all = torch.cat(pos_list)
    return {
        "pos": pos_all,
        "z": torch.cat(z_list).long(),
        "edge_index": torch.stack([torch.tensor(edge_src, dtype=torch.long),
                                   torch.tensor(edge_dst, dtype=torch.long)]),
        "graph_index": torch.tensor(graph, dtype=torch.long),
        "key_index": torch.tensor(keys, dtype=torch.long),
        "endpoint_names": ("reactant", "product"),
        "signed_intent": torch.tensor(1.0, dtype=dtype),
    }
