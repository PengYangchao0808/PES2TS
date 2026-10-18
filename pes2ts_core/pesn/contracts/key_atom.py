"""Key-atom displacement contracts for the V5 sprint (ADR-0026).

Schema: ``pesn_key_atom_task_v1`` / feature-width constants shared by the
teacher builder, feature builder, models and trainer lanes.

Node feature width is 36 = 16 (element embedding, realized model-side via
``nn.Embedding(119, 16)`` over ``z_atomic``) + 20 invariant scalars stored in
feature shards as ``h_node_inv``.  Edge width 29 and task width 19 follow the
sprint plan §2 table verbatim; slot layouts below are normative for the
anchor on/off masking contract.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Final, Mapping, Optional

try:  # torch is optional for the pure-geometry lanes (ADR-0026 Lane A):
    import torch  # data/teacher/split work never touches it; KeyAtomBatch
except ModuleNotFoundError:  # annotations stay lazy (``from __future__``).
    torch = None  # type: ignore[assignment]

SCHEMA_VERSION = "pesn_key_atom_task_v1"
FEATURE_SPEC_SCHEMA = "pesn_v5_feature_spec_v1"
TEACHER_SCHEMA = "pesn_v5_teacher_v1"
SPLIT_SCHEMA = "pesn_v5_split_v1"
FRAME_CONTRACT_SCHEMA = "pesn_v5_frame_contract_v1"
CONFLICT_AUDIT_SCHEMA = "pesn_v5_conflict_audit_v1"

# --- normative slot layouts -------------------------------------------------
NODE_INV_DIM = 20  # + model-side element embedding 16 => 36 total
EDGE_DIM = 29
TASK_DIM = 19
Z_DIM = 64

# h_node_inv indices (final node vector index = 16 + inv index)
NODE_INV_SLOTS = {
    "charge_start": 0,        # formal charge start /4
    "charge_other": 1,        # formal charge other /4
    "deg_start": 2,           # covalent degree start /6
    "deg_other": 3,           # covalent degree other /6
    "aromatic_start": 4,
    "aromatic_other": 5,
    "is_key": 6,
    "degK": 7,                # min(degK,4)/4
    "hop0": 8,                # hop-from-A one-hot 0/1/2/3/>3 -> 8..12
    "hop1": 9,
    "hop2": 10,
    "hop3": 11,
    "dist_cur": 13,           # ||x_i - c_A||/5
    "dist_x0": 14,            # ||x0_i - c0_A||/5          (anchor)
    "disp_anchor": 15,        # ||(x-c_A)-(x0-c0_A)||/1     (anchor)
    "radial_angle": 16,       # current/start radial angle (anchor)
    "anchor_valid": 17,       # (anchor)
    "is_h": 18,
    "isotope_diff": 19,
}
NODE_ANCHOR_SLOTS = (14, 15, 16, 17)  # within h_node_inv

# edge_attr indices
EDGE_SLOTS = {
    "rbf0": 0,                # 16 Gaussian RBF on current r, 16..15
    "r0_scaled": 16,          # r0/4.5                     (anchor)
    "delta_d": 17,            # clip((r-r0)/1, -3, 3)       (anchor)
    "b_start": 18,            # B_start/3
    "b_other": 19,            # B_other/3
    "db": 20,                 # dB/3
    "covalent_start": 21,
    "covalent_other": 22,
    "is_k": 23,
    "is_radius": 24,
    "aromatic_start": 25,
    "aromatic_other": 26,
    "pauling_proxy": 27,
    "proxy_mask": 28,
}
EDGE_ANCHOR_SLOTS = (16, 17)

# task19 indices
TASK_SLOTS = {
    "branch0": 0,
    "branch1": 1,
    "method0": 2,             # 8-slot one-hot 2..9 (UNK -> slot 2)
    "solvent0": 10,           # 4-slot one-hot 10..13 (UNK -> slot 10)
    "charge": 14,             # charge/4
    "multiplicity": 15,       # (mult-1)/4
    "d0": 16,                 # d0/0.08
    "rho_known": 17,
    "solvent_known": 18,
}

# teacher eligibility reason ids (stored as uint8 in shards)
REASON_OK = 0
REASON_NO_FUTURE_STEP = 1
REASON_PAIR_METRIC_UNINFORMATIVE = 2
REASON_GAUGE_UNRESOLVED = 3
REASON_REFERENCE_INVALID = 4
REASON_IDS = {
    REASON_OK: "ok",
    REASON_NO_FUTURE_STEP: "no_future_step",
    REASON_PAIR_METRIC_UNINFORMATIVE: "pair_metric_uninformative",
    REASON_GAUGE_UNRESOLVED: "gauge_unresolved",
    REASON_REFERENCE_INVALID: "reference_invalid",
}


@dataclass
class KeyAtomBatch:
    """Collated batch consumed by V5 models (sprint plan §2 semantics).

    All tensors live on the target device; ``ptr``-style offsets use int64.
    ``y_teacher`` and diagnostic fields are excluded from model.forward
    consumption (training/eval side only).
    """

    z_atomic: torch.Tensor            # [ΣN] int64
    h_node_inv: torch.Tensor          # [ΣN, 20] float32
    pos: torch.Tensor                 # [ΣN, 3] float32 (current, centered)
    pos0: torch.Tensor                # [ΣN, 3] float32 (causal X0, same frame)
    anchor_vec: torch.Tensor          # [ΣN, 3] float32 ((x-c_A)-(x0-c0_A))
    edge_index: torch.Tensor          # [2, ΣE] int64 global rows
    edge_attr: torch.Tensor           # [ΣE, 29] float32
    r_ij: torch.Tensor                # [ΣE, 3] float32
    r0_ij: torch.Tensor               # [ΣE, 3] float32
    task19: torch.Tensor              # [B, 19] float32
    batch: torch.Tensor               # [ΣN] int64
    ptr: torch.Tensor                 # [B+1] int64
    key_index: torch.Tensor           # [Σm] int64 global rows
    key_ptr: torch.Tensor             # [B+1] int64
    pair_index: torch.Tensor          # [2, ΣP] int64 global key rows
    pair_edit: torch.Tensor           # [ΣP] float32 signed ΔB (start->other)
    degK: torch.Tensor                # [Σm] float32 key-pair degree per key atom
    # --- supervision / diagnostics (not model inputs) ---
    y_teacher: Optional[torch.Tensor] = None   # [Σm, 3] float32
    q_weight: Optional[torch.Tensor] = None    # [B] float32
    near_ts: Optional[torch.Tensor] = None     # [B] float32
    terminal: Optional[torch.Tensor] = None    # [B] bool
    capped: Optional[torch.Tensor] = None      # [B] bool
    d0: Optional[torch.Tensor] = None          # [B] float32
    group_ids: tuple = field(default=())
    reaction_ids: tuple = field(default=())
    branch: Optional[torch.Tensor] = None      # [B] int64

    def anchor_off_(self) -> "KeyAtomBatch":
        """Zero the X0-derived slots in place (shapes preserved, plan §5.2)."""
        self.h_node_inv[:, list(NODE_ANCHOR_SLOTS)] = 0.0
        self.edge_attr[:, list(EDGE_ANCHOR_SLOTS)] = 0.0
        self.r0_ij = torch.zeros_like(self.r0_ij)
        self.anchor_vec = torch.zeros_like(self.anchor_vec)
        self.pos0 = torch.zeros_like(self.pos0)
        return self


# ---------------------------------------------------------------------------
# KeyAtomTask (ADR-0026 §2.1; plan §1.1/§1.3) — schema pesn_key_atom_task_v1
# ---------------------------------------------------------------------------

KEY_ATOM_TASK_SCHEMA: Final[str] = SCHEMA_VERSION

#: Top-level fields of a ``pesn_key_atom_task_v1`` document.  The loader is
#: strict: any unknown key is a typed rejection, never a silent ignore.
KEY_ATOM_TASK_FIELDS: Final[frozenset[str]] = frozenset({
    "schema_version",
    "reaction_id",
    "reaction_group_id",
    "atomic_numbers",          # [N] int, ts_irc_index row order
    "atom_map_ids",            # [N] str, per-row map id
    "key_atom_rows",           # [m] int64 row indices, sorted unique
    "endpoints",               # [2] {bonds: [[i, j, order], ...]} row-index pairs
    "endpoint_sides",          # [2] "R" | "P" per endpoint slot
    "endpoint_match_evidence",  # per-endpoint pair-set match counts vs p1 r/p bonds
    "charge",
    "multiplicity",
    "method_id",
    "solvent",
    "group_id",
    "sources",                 # {name: {path, sha256}}
})


class KeyAtomTaskError(ValueError):
    """Typed refusal when building/loading a KeyAtomTask document."""


def _endpoint_pair_set(bonds) -> set[tuple[int, int]]:
    return {(min(int(b[0]), int(b[1])), max(int(b[0]), int(b[1]))) for b in bonds}


def _map_pair_set(bonds_map_space) -> set[tuple[int, int]]:
    return {(min(int(b[0]), int(b[1])), max(int(b[0]), int(b[1]))) for b in bonds_map_space}


def _map_pair_order_set(bonds_map_space) -> set[tuple[int, int, float]]:
    """(min_map, max_map, order) signatures for order-sensitive matching."""
    return {
        (min(int(b[0]), int(b[1])), max(int(b[0]), int(b[1])),
         round(float(b[2]), 3))
        for b in bonds_map_space
    }


def collect_key_maps(p1_mapping: Mapping) -> list[int]:
    """Unique map ids touched by bond_events, sorted (ADR-0026 §2.1).

    Union of broken/formed/order_changed ``atoms`` and hydrogen_migration
    ``from``/``h``/``to`` map ids.
    """
    events = p1_mapping.get("bond_events") or {}
    keys: set[int] = set()
    for kind in ("broken", "formed", "order_changed"):
        for ev in events.get(kind) or ():
            for atom in ev.get("atoms") or ():
                keys.add(int(atom))
    for ev in events.get("hydrogen_migration") or ():
        for field_name in ("from", "h", "to"):
            keys.add(int(ev[field_name]))
    return sorted(keys)


@dataclass(frozen=True)
class KeyAtomTask:
    """One reaction's key-atom task record (ADR-0026 §2.1).

    Atom row order is the IRC ``ts_irc_index`` order recorded in the G1 P1
    mapping; endpoint graphs use those row indices.  Branch ``b`` deploys
    ``endpoints[b]`` as its start graph and ``endpoints[1-b]`` as the other
    graph, so ΔB = B_other − B_start for that branch.
    """

    schema_version: str
    reaction_id: str
    reaction_group_id: str
    atomic_numbers: tuple[int, ...]
    atom_map_ids: tuple[str, ...]
    key_atom_rows: tuple[int, ...]
    endpoints: tuple[Mapping, Mapping]        # each: {"bonds": [[i, j, order], ...]}
    endpoint_sides: tuple[str, str]           # "R"/"P" per endpoint slot
    endpoint_match_evidence: Mapping
    charge: int
    multiplicity: int
    method_id: str
    solvent: str
    group_id: str
    sources: Mapping[str, Mapping]            # {name: {"path": str, "sha256": str}}

    # --- derived views -----------------------------------------------------

    @property
    def n_atoms(self) -> int:
        return len(self.atomic_numbers)

    @property
    def key_map_ids(self) -> list[int]:
        """Map ids of the key atoms (row order)."""
        return [_map_id_of(self.atom_map_ids[r]) for r in self.key_atom_rows]

    def start_bonds(self, branch: int) -> tuple[tuple[int, int, float], ...]:
        """Covalent bonds ``(i, j, order)`` of the branch's start endpoint."""
        return _bonds_of(self.endpoints[branch])

    def other_bonds(self, branch: int) -> tuple[tuple[int, int, float], ...]:
        """Covalent bonds of the other endpoint (branch ``1 - branch``)."""
        return _bonds_of(self.endpoints[1 - int(branch)])

    def union_edges(self) -> tuple[tuple[int, int], ...]:
        """G_R ∪ G_P covalent edges as sorted unique row pairs."""
        pairs: set[tuple[int, int]] = set()
        for endpoint in self.endpoints:
            pairs |= _endpoint_pair_set(_bonds_of(endpoint))
        return tuple(sorted(pairs))

    def endpoint_orders(self, endpoint: int) -> dict[tuple[int, int], float]:
        return {(min(i, j), max(i, j)): float(o)
                for i, j, o in _bonds_of(self.endpoints[endpoint])}

    def K_pairs(self, branch: int) -> tuple[tuple[int, int, float, float, float], ...]:
        """Changed pairs for branch ``b`` with both orders and signed ΔB.

        Returns ``(i, j, B_start, B_other, dB)`` rows sorted by pair, where
        the start endpoint is ``endpoints[b]`` and dB = B_other − B_start
        (ADR-0026 §2.1; plan §0 branch semantics).
        """
        b = int(branch)
        if b not in (0, 1):
            raise KeyAtomTaskError(f"branch must be 0 or 1, got {branch!r}")
        start = self.endpoint_orders(b)
        other = self.endpoint_orders(1 - b)
        out = []
        for pair in sorted(set(start) | set(other)):
            b_start = start.get(pair, 0.0)
            b_other = other.get(pair, 0.0)
            if b_start != b_other:
                out.append((pair[0], pair[1], b_start, b_other, b_other - b_start))
        return tuple(out)

    # --- persistence -------------------------------------------------------

    def to_payload(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "reaction_id": self.reaction_id,
            "reaction_group_id": self.reaction_group_id,
            "atomic_numbers": [int(z) for z in self.atomic_numbers],
            "atom_map_ids": [str(m) for m in self.atom_map_ids],
            "key_atom_rows": [int(r) for r in self.key_atom_rows],
            "endpoints": [
                {"bonds": [[int(i), int(j), float(o)]
                           for i, j, o in _bonds_of(ep)]}
                for ep in self.endpoints
            ],
            "endpoint_sides": [str(s) for s in self.endpoint_sides],
            "endpoint_match_evidence": dict(self.endpoint_match_evidence),
            "charge": int(self.charge),
            "multiplicity": int(self.multiplicity),
            "method_id": str(self.method_id),
            "solvent": str(self.solvent),
            "group_id": str(self.group_id),
            "sources": {str(k): dict(v) for k, v in self.sources.items()},
        }

    @classmethod
    def from_payload(cls, doc: Mapping) -> "KeyAtomTask":
        """Strict loader: schema check + unknown-field rejection."""
        if not isinstance(doc, Mapping):
            raise KeyAtomTaskError("task document must be a mapping")
        got_schema = doc.get("schema_version")
        if got_schema != KEY_ATOM_TASK_SCHEMA:
            raise KeyAtomTaskError(
                f"schema_version {got_schema!r} != {KEY_ATOM_TASK_SCHEMA!r}")
        unknown = set(doc) - KEY_ATOM_TASK_FIELDS
        if unknown:
            raise KeyAtomTaskError(f"unknown task fields: {sorted(unknown)}")
        missing = KEY_ATOM_TASK_FIELDS - set(doc)
        if missing:
            raise KeyAtomTaskError(f"missing task fields: {sorted(missing)}")
        endpoints = doc["endpoints"]
        if not (isinstance(endpoints, (list, tuple)) and len(endpoints) == 2):
            raise KeyAtomTaskError("endpoints must be a pair")
        sides = tuple(str(s) for s in doc["endpoint_sides"])
        if sorted(sides) != ["P", "R"]:
            raise KeyAtomTaskError(
                f"endpoint_sides must be one R and one P, got {sides!r}")
        rows = tuple(int(r) for r in doc["key_atom_rows"])
        if tuple(sorted(set(rows))) != rows:
            raise KeyAtomTaskError("key_atom_rows must be sorted unique")
        n = len(doc["atomic_numbers"])
        if len(doc["atom_map_ids"]) != n:
            raise KeyAtomTaskError("atom_map_ids length mismatch")
        if any(not (0 <= r < n) for r in rows):
            raise KeyAtomTaskError("key_atom_rows out of range")
        return cls(
            schema_version=str(doc["schema_version"]),
            reaction_id=str(doc["reaction_id"]),
            reaction_group_id=str(doc["reaction_group_id"]),
            atomic_numbers=tuple(int(z) for z in doc["atomic_numbers"]),
            atom_map_ids=tuple(str(m) for m in doc["atom_map_ids"]),
            key_atom_rows=rows,
            endpoints=(dict(endpoints[0]), dict(endpoints[1])),
            endpoint_sides=sides,
            endpoint_match_evidence=dict(doc["endpoint_match_evidence"]),
            charge=int(doc["charge"]),
            multiplicity=int(doc["multiplicity"]),
            method_id=str(doc["method_id"]),
            solvent=str(doc["solvent"]),
            group_id=str(doc["group_id"]),
            sources={str(k): dict(v) for k, v in dict(doc["sources"]).items()},
        )

    def save(self, path) -> None:
        from pes2ts_core.utils.jsonio import write_json
        write_json(path, self.to_payload())


def _map_id_of(atom_map_id: str) -> int:
    """Parse ``m0007``-style map ids to int; plain digits pass through."""
    text = str(atom_map_id)
    if text.startswith("m") or text.startswith("M"):
        text = text[1:]
    return int(text)


def _bonds_of(endpoint: Mapping) -> tuple[tuple[int, int, float], ...]:
    return tuple((int(b[0]), int(b[1]), float(b[2]))
                 for b in endpoint.get("bonds") or ())


def _sha256_of_json(doc: Mapping) -> str:
    import json
    payload = json.dumps(doc, sort_keys=True, ensure_ascii=False,
                         separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_key_atom_task(
    wp2_task: Mapping,
    p1_mapping: Mapping,
    *,
    group_id: str | None = None,
    wp2_path: str | None = None,
    p1_path: str | None = None,
    solvent: str = "unknown",
) -> KeyAtomTask:
    """Build a KeyAtomTask from a wp2 task.json + G1 P1 mapping document.

    Row order follows ``mapping.map_to_atoms[*].ts_irc_index``; key rows are
    the sorted unique map ids of ``bond_events`` mapped to rows.  The two wp2
    endpoints are matched to the R/P sides by comparing their bond pair sets
    (map-id space) against ``graph.r_bonds`` / ``graph.p_bonds``; the match
    evidence is recorded in the task document (ADR-0026 §2.1).
    """
    map_entries = p1_mapping.get("mapping", {}).get("map_to_atoms") or []
    if not map_entries:
        raise KeyAtomTaskError("p1 mapping lacks map_to_atoms")
    row_of_map: dict[int, int] = {}
    ordered: list[dict] = [None] * len(map_entries)  # type: ignore[list-item]
    for entry in map_entries:
        row = int(entry["ts_irc_index"])
        if not (0 <= row < len(map_entries)):
            raise KeyAtomTaskError(f"ts_irc_index {row} out of range")
        if ordered[row] is not None:
            raise KeyAtomTaskError("duplicate ts_irc_index in map_to_atoms")
        ordered[row] = entry
        row_of_map[int(entry["map"])] = row
    if any(e is None for e in ordered):
        raise KeyAtomTaskError("map_to_atoms does not cover every ts row")

    atomic_numbers = tuple(_z_of(str(e["element"])) for e in ordered)
    atom_map_ids = tuple(f"m{int(e['map']):04d}" for e in ordered)

    # key rows: sorted unique event map ids -> rows
    key_maps = collect_key_maps(p1_mapping)
    missing = [m for m in key_maps if m not in row_of_map]
    if missing:
        raise KeyAtomTaskError(f"key map ids absent from mapping: {missing}")
    key_rows = tuple(sorted(row_of_map[m] for m in key_maps))

    # endpoints: convert wp2 row space -> this record's row space through
    # the endpoint's own atom_map_ids (wp2 row r <-> map id of that row)
    endpoints = []
    for ep in _expect_endpoints(wp2_task):
        ep_maps = [_map_id_of(m) for m in ep["atom_map_ids"]]
        bonds = []
        for b in ep["bonds"]:
            i = row_of_map[ep_maps[int(b["i"])]]
            j = row_of_map[ep_maps[int(b["j"])]]
            bonds.append([min(i, j), max(i, j), float(b["bond_order"])])
        bonds.sort()
        endpoints.append({"bonds": bonds})

    # side matching: endpoint bond pair sets (map space) vs p1 r/p bonds;
    # ties (bond SETS equal — order-only reactions) fall through to
    # (pair, order) signatures; a full ordered-graph tie (no event can
    # distinguish) resolves deterministically and is recorded as such.
    r_pairs = _map_pair_set(p1_mapping["graph"]["r_bonds"])
    p_pairs = _map_pair_set(p1_mapping["graph"]["p_bonds"])
    r_sig = _map_pair_order_set(p1_mapping["graph"]["r_bonds"])
    p_sig = _map_pair_order_set(p1_mapping["graph"]["p_bonds"])
    sides: list[str] = []
    match_evidence = {}
    for slot, ep in enumerate(_expect_endpoints(wp2_task)):
        ep_bonds = [
            (_map_id_of(ep["atom_map_ids"][int(b["i"])]),
             _map_id_of(ep["atom_map_ids"][int(b["j"])]),
             float(b.get("bond_order", 0.0) or 0.0))
            for b in ep["bonds"]
        ]
        ep_pair_maps = {
            (min(a, c), max(a, c)) for a, c, _o in ep_bonds}
        ep_sig = {
            (min(a, c), max(a, c), round(o, 3)) for a, c, o in ep_bonds}
        r_hit = len(ep_pair_maps & r_pairs)
        p_hit = len(ep_pair_maps & p_pairs)
        tiebreak = "pair_set"
        if r_hit > p_hit:
            side = "R"
        elif p_hit > r_hit:
            side = "P"
        else:
            r_hit = len(ep_sig & r_sig)
            p_hit = len(ep_sig & p_sig)
            tiebreak = "pair_order_signature"
            if r_hit > p_hit:
                side = "R"
            elif p_hit > r_hit:
                side = "P"
            else:
                # ordered graphs indistinguishable at this matching depth;
                # any bond event forces an ordered-set difference, so this
                # is a benign symmetric case — deterministic fallback that
                # never collides with the side already taken by slot 0.
                if slot == 0:
                    side = "R"
                else:
                    side = "P" if sides[0] == "R" else "R"
                tiebreak = "graph_ambiguous_fallback"
        sides.append(side)
        match_evidence[f"endpoint_{slot}"] = {
            "side": side,
            "r_hits": r_hit,
            "p_hits": p_hit,
            "tiebreak": tiebreak,
            "n_endpoint_pairs": len(ep_pair_maps),
            "n_r_pairs": len(r_pairs),
            "n_p_pairs": len(p_pairs),
        }
    if sides[0] == sides[1]:
        raise KeyAtomTaskError(
            f"both endpoints matched the same side {sides[0]!r}")

    def _source_entry(path: str | None, doc: Mapping) -> dict:
        """Prefer the on-disk file digest; fall back to the canonical-JSON
        content digest for in-memory builds (both are sha256 hex)."""
        if path:
            from pathlib import Path
            from pes2ts_core.utils.hashing import sha256_file
            p = Path(path)
            if p.is_file():
                return {"path": str(path), "sha256": sha256_file(p)}
        return {"path": str(path or ""), "sha256": _sha256_of_json(doc)}

    return KeyAtomTask(
        schema_version=KEY_ATOM_TASK_SCHEMA,
        reaction_id=str(wp2_task["reaction_id"]),
        reaction_group_id=str(wp2_task.get("reaction_group_id")
                              or wp2_task["reaction_id"]),
        atomic_numbers=atomic_numbers,
        atom_map_ids=atom_map_ids,
        key_atom_rows=key_rows,
        endpoints=(endpoints[0], endpoints[1]),
        endpoint_sides=(sides[0], sides[1]),
        endpoint_match_evidence=match_evidence,
        charge=int(wp2_task.get("charge", 0)),
        multiplicity=int(wp2_task.get("multiplicity", 1)),
        method_id=str(wp2_task.get("method_id", "unknown")),
        solvent=str(solvent),
        group_id=str(group_id or wp2_task.get("reaction_group_id")
                     or wp2_task["reaction_id"]),
        sources={
            "wp2_task": _source_entry(wp2_path, wp2_task),
            "p1_mapping": _source_entry(p1_path, p1_mapping),
        },
    )


def load_key_atom_task(path) -> KeyAtomTask:
    """Load and strictly validate a ``pesn_key_atom_task_v1`` JSON file."""
    from pes2ts_core.utils.jsonio import read_json
    return KeyAtomTask.from_payload(read_json(path))


#: Minimal element symbol -> atomic number table for the P1 ``elements``
#: field (H..Ar plus common heavier rows; unknown symbols are refused).
_ELEMENT_Z: Final[dict[str, int]] = {
    "H": 1, "He": 2, "Li": 3, "Be": 4, "B": 5, "C": 6, "N": 7, "O": 8,
    "F": 9, "Ne": 10, "Na": 11, "Mg": 12, "Al": 13, "Si": 14, "P": 15,
    "S": 16, "Cl": 17, "Ar": 18, "K": 19, "Ca": 20, "Sc": 21, "Ti": 22,
    "V": 23, "Cr": 24, "Mn": 25, "Fe": 26, "Co": 27, "Ni": 28, "Cu": 29,
    "Zn": 30, "Ga": 31, "Ge": 32, "As": 33, "Se": 34, "Br": 35, "Kr": 36,
    "I": 53, "Xe": 54,
}


def _z_of(symbol: str) -> int:
    z = _ELEMENT_Z.get(symbol.strip().capitalize())
    if z is None:
        raise KeyAtomTaskError(f"unknown element symbol {symbol!r}")
    return z


def _expect_endpoints(wp2_task: Mapping) -> tuple[Mapping, Mapping]:
    endpoints = wp2_task.get("endpoints")
    if not (isinstance(endpoints, (list, tuple)) and len(endpoints) == 2):
        raise KeyAtomTaskError("wp2 task endpoints must be a pair")
    for ep in endpoints:
        if "atom_map_ids" not in ep or "bonds" not in ep:
            raise KeyAtomTaskError("wp2 endpoint lacks atom_map_ids/bonds")
    return endpoints[0], endpoints[1]
