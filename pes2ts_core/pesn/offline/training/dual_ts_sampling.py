"""§I.10 unified sampling + exposure accounting for dual-TS BC and IQL.

This module is the SINGLE sampling surface consumed by BOTH trainers (v4.1
WP2 / P06).  Neither the BC trainer nor the IQL trainer may re-implement
draw logic inline; both call :meth:`DualTSGroupBranchSampler.draw_batch` so
that the same ``sampler_seed`` + the same dataset + the same ``batch_states``
yield the IDENTICAL row sequence (group/branch/arc-position draws are fully
deterministic given the seed).

Frozen draw probabilities (v4.1):

* **group** — each reaction group is drawn exactly once per round, in sorted
  reaction-group-id order (deterministic; no group RNG).  A round = ``n_groups``
  draws.  ``draw_batch(batch_states)`` is a group-round-robin: it fills the
  batch group-by-group, a batch may span several rounds, and the remainder of
  a partially-consumed round carries over to the next batch.
* **branch** — per group draw, one branch is chosen uniformly at random via
  the sampler RNG (``rng.integers(0, n_branches)``).
* **arc position** — within a branch, rows are ordered by ``arc_key``
  ascending (the canonical order is ``remaining_arc_length`` ascending, i.e.
  NEAR-TS FIRST), then partitioned into ``N_ARC_BINS`` = 3 equal thirds:
  **bin 0 = the smallest-``remaining_arc_length`` third (near-TS), bin 1 =
  middle, bin 2 = the largest third (far-from-TS / chain start)**.  The bin
  cursor advances one bin per (branch, draw), round-robin in the order
  0 → 1 → 2 → 0 …  The exact row inside the bin comes from the sampler RNG.
  A bin that is empty for a short branch falls back to a uniform draw over
  the non-empty bins.

  *Per-trainer batch semantics (registered in ``protocol_v41_r1.json``):*
  BC consumes ``batch_states = 64`` rows per update; IQL consumes
  ``rows_per_round = n_groups`` rows per update (``batch_reactions`` is
  unused).  These are a REGISTERED difference in per-update chunk size only —
  the marginal draw law is the same, and after the canonical-order fix
  (finding 3) the same-seed draw SEQUENCE is identical up to the chunking.

The near-TS *definition* used by exposure accounting is the frozen
``remaining_arc_length <= NEAR_TS_ARC_FRACTION * branch_max`` rule (0.2 × the
per-branch maximum), preserved from the WP3/WP4 arc stratification — this is
the accounting stratum, independent of the 3-bin sampling cursor.

``state_dict()`` captures the bit-level PCG64 state plus all cursors so an
interrupted run resumes on the EXACT same draws (checkpoint fixture test).
"""
from __future__ import annotations

from typing import Final, Mapping, Sequence

import numpy as np

GROUP_BRANCH_SAMPLING_RULE: Final[str] = (
    "reaction_group_balanced->branch_balanced->frozen_within_branch_v0")
WITHIN_BRANCH_RULE: Final[str] = "arc_bin_round_robin_n3_rng_within_bin_v0"
LOGPROB_REDUCTION_RULE: Final[str] = (
    "per_step_valid_atom_sum->within_branch_mean->"
    "per_reaction_branch_mean->group_mean_v0")
N_ARC_BINS: Final[int] = 3

#: Frozen near-TS stratum rule (ral <= 0.2 * branch max), reused from the
#: WP3/WP4 arc stratification.  This is the DEFINITION used for exposure
#: accounting; the 3-bin sampling cursor is a separate mechanism.
NEAR_TS_ARC_FRACTION: Final[float] = 0.2


def hierarchical_logprob_reduction(torch, *, per_atom, atom_sample_index,
                                   n_samples: int, sample_branch_index,
                                   n_branches: int, branch_group_index,
                                   n_groups: int):
    """Frozen §I.10 reduction of per-atom log-probs (or weighted log-probs).

    1. per-step (sample) value = SUM over the sample's valid atom dims;
    2. within-branch mean over the samples drawn from that branch;
    3. per-reaction mean over the branches available in the batch;
    4. mean across reaction groups.

    Returns ``(scalar, group_mean_tensor)``.  Pure tensor ops — no data
    allocation outside the caller's device.
    """
    dev = atom_sample_index.device
    dt = per_atom.dtype
    per_sample = torch.zeros(n_samples, device=dev, dtype=dt).index_add_(
        0, atom_sample_index, per_atom)
    ones_s = torch.ones(n_samples, device=dev, dtype=dt)
    branch_sum = torch.zeros(n_branches, device=dev, dtype=dt).index_add_(
        0, sample_branch_index, per_sample)
    branch_cnt = torch.zeros(n_branches, device=dev, dtype=dt).index_add_(
        0, sample_branch_index, ones_s)
    branch_mean = branch_sum / branch_cnt.clamp(min=1.0)
    ones_b = torch.ones(n_branches, device=dev, dtype=dt)
    group_sum = torch.zeros(n_groups, device=dev, dtype=dt).index_add_(
        0, branch_group_index, branch_mean)
    group_cnt = torch.zeros(n_groups, device=dev, dtype=dt).index_add_(
        0, branch_group_index, ones_b)
    group_mean = group_sum / group_cnt.clamp(min=1.0)
    return group_mean.mean(), group_mean


def build_sampler_groups(
    records: Sequence,
    *,
    group_key,
    branch_key,
    arc_key,
) -> dict[str, dict[str, list[int]]]:
    """Build the canonical §I.10 sampler view ``group -> branch -> [row idx]``.

    Deterministic and SHARED by both trainers so the same dataset yields the
    same row-index layout (and therefore the same sampler draws): groups are
    sorted by id, branches are sorted by id, and rows within a branch are
    ordered by ``arc_key`` ascending (arc / chain order; ties broken by the
    original record position).  Row indices are positions into ``records``.

    ``group_key`` / ``branch_key`` / ``arc_key`` are callables: ``group_key(rec)``
    and ``branch_key(rec)`` return the reaction-group id and branch id, and
    ``arc_key(rec, index)`` returns the arc position (a comparable value, e.g.
    remaining-arc-length for BC, or ``(episode_id, chain_position)`` for IQL).
    Both trainers MUST call this (never build the dict inline).
    """
    groups: dict[str, dict[str, list[tuple]]] = {}
    for i, rec in enumerate(records):
        g = str(group_key(rec))
        b = str(branch_key(rec))
        groups.setdefault(g, {}).setdefault(b, []).append((arc_key(rec, i), i))
    out: dict[str, dict[str, list[int]]] = {}
    for g in sorted(groups):
        out[g] = {}
        for b in sorted(groups[g]):
            pairs = sorted(groups[g][b], key=lambda p: (p[0], p[1]))
            out[g][b] = [i for _, i in pairs]
    return out


class DualTSGroupBranchSampler:
    """Group→branch→within-branch row sampler with restorable RNG state.

    ``groups`` maps ``reaction_group_id -> branch_id -> [row indices in
    arc (chain) order]``.  Every round covers EVERY reaction group exactly
    once (group balance), picks branches by RNG (branch balance), and picks
    the row inside the branch by the frozen arc-bin round robin: positions
    are split into ``N_ARC_BINS`` thirds in ``arc_key``-ascending order (bin 0
    = near-TS, the smallest ``remaining_arc_length``; bin 2 = far-from-TS /
    chain start), the bin cursor advances one bin per (branch, draw) round
    robin 0→1→2 so near-TS, middle and far steps are all covered over
    consecutive draws, and the exact row inside the bin comes from the
    sampler RNG.

    ``draw()`` returns one full round (``rows_per_group`` rows per group —
    the legacy per-update unit).  ``draw_batch(batch_states)`` returns exactly
    ``batch_states`` rows via group-round-robin, carrying the partially
    consumed round over to the next call.

    ``state_dict()`` captures the bit-level PCG64 state plus all cursors so an
    interrupted run resumes on the EXACT same draws.
    """

    def __init__(self, *, seed: int,
                 groups: Mapping[str, Mapping[str, Sequence[int]]],
                 rows_per_group: int = 1) -> None:
        self.seed = int(seed)
        self.rows_per_group = int(rows_per_group)
        if self.rows_per_group < 1:
            raise ValueError("rows_per_group must be >= 1")
        # deterministic insertion order: sorted groups, sorted branches
        self._groups: dict[str, dict[str, list[int]]] = {
            str(g): {str(b): [int(i) for i in list(rows)]
                     for b, rows in sorted(branches.items())}
            for g, branches in sorted(groups.items())
        }
        if not self._groups:
            raise ValueError("sampler requires at least one reaction group")
        for g, branches in self._groups.items():
            if not branches:
                raise ValueError(f"reaction group {g!r} has no branches")
            for b, rows in branches.items():
                if not rows:
                    raise ValueError(
                        f"branch {b!r} of group {g!r} has no rows")
        self._group_order: list[str] = sorted(self._groups)
        self._n_groups = len(self._group_order)
        self._rng = np.random.default_rng(self.seed)
        self.draws = 0
        self._round_cursor = 0
        self.rounds_completed = 0
        self._bin_cursor: dict[str, int] = {
            f"{g}|{b}": 0
            for g, branches in self._groups.items()
            for b in branches
        }
        self._bin_rows: dict[str, list[list[int]]] = {}
        for g, branches in self._groups.items():
            for b, rows in branches.items():
                self._bin_rows[f"{g}|{b}"] = [
                    list(part) for part in np.array_split(rows, N_ARC_BINS)]

    # -- properties ----------------------------------------------------------

    @property
    def sampling_rule(self) -> str:
        return GROUP_BRANCH_SAMPLING_RULE

    @property
    def within_branch_rule(self) -> str:
        return WITHIN_BRANCH_RULE

    @property
    def n_groups(self) -> int:
        return self._n_groups

    @property
    def rows_per_round(self) -> int:
        return self._n_groups * self.rows_per_group

    # -- low-level draw --------------------------------------------------------

    def _draw_group_row(self, g: str) -> int:
        branches = self._groups[g]
        names = list(branches.keys())
        b = str(names[int(self._rng.integers(0, len(names)))])
        key = f"{g}|{b}"
        slot = self._bin_cursor[key] % N_ARC_BINS
        self._bin_cursor[key] = self._bin_cursor[key] + 1
        bin_rows = self._bin_rows[key][slot]
        # a bin may be empty when a branch has fewer rows than bins
        if not bin_rows:
            nonempty = [part for part in self._bin_rows[key] if part]
            bin_rows = nonempty[int(self._rng.integers(0, len(nonempty)))]
        return int(bin_rows[int(self._rng.integers(0, len(bin_rows)))])

    # -- public draw -----------------------------------------------------------

    def draw(self) -> list[int]:
        """One full round: ``rows_per_group`` rows per reaction group.

        Legacy per-update unit (every group drawn once per inner round).
        Equivalent in stream to ``draw_batch(self.rows_per_round)``.
        """
        chosen: list[int] = []
        for _ in range(self.rows_per_group):
            for g in self._group_order:
                chosen.append(self._draw_group_row(g))
        self.draws += len(chosen)
        return chosen

    def draw_batch(self, batch_states: int) -> list[int]:
        """Exactly ``batch_states`` rows via group-round-robin.

        Fills the batch group-by-group; a batch may span several rounds and
        the remainder of a partially-consumed round carries over to the next
        call (``_round_cursor``).  Deterministic given the seed.
        """
        if int(batch_states) < 1:
            raise ValueError("batch_states must be >= 1")
        chosen: list[int] = []
        for _ in range(int(batch_states)):
            g = self._group_order[self._round_cursor]
            chosen.append(self._draw_group_row(g))
            self._round_cursor += 1
            if self._round_cursor >= self._n_groups:
                self._round_cursor = 0
                self.rounds_completed += 1
        self.draws += len(chosen)
        return chosen

    # -- resume state -----------------------------------------------------------

    def state_dict(self) -> dict:
        return {
            "seed": self.seed,
            "rows_per_group": self.rows_per_group,
            "draws": int(self.draws),
            "round_cursor": int(self._round_cursor),
            "rounds_completed": int(self.rounds_completed),
            "bin_cursor": {k: int(v) for k, v in self._bin_cursor.items()},
            "bit_generator_state": self._rng.bit_generator.state,
        }

    def load_state_dict(self, state: Mapping) -> None:
        if int(state["seed"]) != self.seed:
            raise ValueError(
                f"sampler seed mismatch: {state['seed']!r} != {self.seed!r}")
        if int(state["rows_per_group"]) != self.rows_per_group:
            raise ValueError("sampler rows_per_group mismatch on restore")
        self.draws = int(state["draws"])
        self._round_cursor = int(state.get("round_cursor", 0))
        self.rounds_completed = int(state.get("rounds_completed", 0))
        self._bin_cursor = {str(k): int(v)
                            for k, v in dict(state["bin_cursor"]).items()}
        self._rng = np.random.default_rng()
        self._rng.bit_generator.state = state["bit_generator_state"]


class ExposureTracker:
    """Per-reaction-group exposure accounting over the sampler stream.

    Tracks, per group and cumulatively: draws, distinct states drawn,
    coverage (``distinct / drawable``), repeat-sampling rate
    (``(draws - distinct) / draws``), and the near-TS draw fraction
    (``remaining_arc_length <= 0.2 * branch_max``).  Full history is kept as
    per-interval snapshots (no first-3+last truncation).

    ``row_meta`` is a list with one entry per drawable row (index-aligned with
    the sampler's row indices), each a mapping with keys ``group_id``,
    ``state_id``, ``branch_id`` and an optional ``remaining_arc_length``
    (``None`` → near-TS unknown → counted as not-near-TS).
    """

    def __init__(self, *, row_meta: Sequence[Mapping]) -> None:
        self.row_meta = [dict(r) for r in row_meta]
        self.n_drawable = len(self.row_meta)
        self.group_ids: list[str] = sorted(
            {str(r["group_id"]) for r in self.row_meta})
        # per-group drawable counts + per-(group, branch) max ral
        self.group_drawable: dict[str, int] = {g: 0 for g in self.group_ids}
        branch_max: dict[tuple, float] = {}
        for r in self.row_meta:
            g = str(r["group_id"])
            self.group_drawable[g] += 1
            ral = r.get("remaining_arc_length")
            if ral is None:
                continue
            key = (g, str(r["branch_id"]))
            branch_max[key] = max(branch_max.get(key, 0.0), float(ral))
        self._near_ts: list[bool] = []
        for r in self.row_meta:
            ral = r.get("remaining_arc_length")
            if ral is None:
                self._near_ts.append(False)
            else:
                key = (str(r["group_id"]), str(r["branch_id"]))
                self._near_ts.append(
                    float(ral) <= NEAR_TS_ARC_FRACTION * branch_max[key])
        self._reset()

    # -- state -----------------------------------------------------------

    def _reset(self) -> None:
        self.total_draws = 0
        self.distinct_all: set = set()
        self.near_ts_total = 0
        self.interval_draws = 0
        self.interval_distinct_all: set = set()
        self.interval_near_ts = 0
        self.per_group: dict[str, dict] = {
            g: {
                "draws": 0,
                "distinct": set(),
                "near_ts": 0,
                "interval_draws": 0,
                "interval_distinct": set(),
                "interval_near_ts": 0,
            }
            for g in self.group_ids
        }
        self.history: list[dict] = []

    # -- recording -----------------------------------------------------------

    def record_batch(self, row_indices: Sequence[int]) -> None:
        for i in row_indices:
            meta = self.row_meta[int(i)]
            g = str(meta["group_id"])
            sid = str(meta["state_id"])
            self.total_draws += 1
            self.interval_draws += 1
            self.distinct_all.add(sid)
            self.interval_distinct_all.add(sid)
            pg = self.per_group[g]
            pg["draws"] += 1
            pg["interval_draws"] += 1
            pg["distinct"].add(sid)
            pg["interval_distinct"].add(sid)
            if self._near_ts[int(i)]:
                self.near_ts_total += 1
                self.interval_near_ts += 1
                pg["near_ts"] += 1
                pg["interval_near_ts"] += 1

    def snapshot(self, *, kind: str, update: int, epoch: int) -> dict:
        """Close the current interval into ``history`` and reset it.

        Entries are kept COMPACT: per-interval scalars + per-group interval
        deltas + a compact cumulative (no per-group expansion).  The full
        per-group cumulative is emitted once by :meth:`payload`.
        """
        entry = {
            "kind": kind,
            "update": int(update),
            "epoch": int(epoch),
            "draws_in_interval": self.interval_draws,
            "distinct_in_interval": len(self.interval_distinct_all),
            "near_ts_in_interval": self.interval_near_ts,
            "per_group_interval": {
                g: {
                    "draws": pg["interval_draws"],
                    "distinct": len(pg["interval_distinct"]),
                    "near_ts": pg["interval_near_ts"],
                }
                for g, pg in self.per_group.items()
            },
            "cumulative": self.cumulative_payload(include_per_group=False),
        }
        self.history.append(entry)
        self.interval_draws = 0
        self.interval_distinct_all = set()
        self.interval_near_ts = 0
        for pg in self.per_group.values():
            pg["interval_draws"] = 0
            pg["interval_distinct"] = set()
            pg["interval_near_ts"] = 0
        return entry

    def cumulative_payload(self, *, include_per_group: bool = True) -> dict:
        payload = {
            "total_draws": self.total_draws,
            "distinct_states": len(self.distinct_all),
            "coverage": self._ratio(len(self.distinct_all), self.n_drawable),
            "repeat_rate": self._ratio(
                self.total_draws - len(self.distinct_all), self.total_draws),
            "near_ts_draws": self.near_ts_total,
            "near_ts_fraction": self._ratio(
                self.near_ts_total, self.total_draws),
        }
        if include_per_group:
            payload["per_group"] = {
                g: self._group_payload(g, self.per_group[g])
                for g in self.group_ids
            }
        return payload

    def _group_payload(self, g: str, pg: dict) -> dict:
        drawable = self.group_drawable[g]
        draws = pg["draws"]
        return {
            "drawable_states": drawable,
            "draws": draws,
            "distinct_states": len(pg["distinct"]),
            "coverage": self._ratio(len(pg["distinct"]), drawable),
            "repeat_rate": self._ratio(draws - len(pg["distinct"]), draws),
            "near_ts_draws": pg["near_ts"],
            "near_ts_fraction": self._ratio(pg["near_ts"], draws),
        }

    @staticmethod
    def _ratio(numerator: int, denominator: int) -> float:
        if denominator <= 0:
            return 0.0
        return float(numerator) / float(denominator)

    def payload(self) -> dict:
        return {
            "drawable_states": self.n_drawable,
            "n_groups": len(self.group_ids),
            "history": self.history,
            **self.cumulative_payload(include_per_group=True),
        }


__all__ = [
    "GROUP_BRANCH_SAMPLING_RULE",
    "LOGPROB_REDUCTION_RULE",
    "N_ARC_BINS",
    "NEAR_TS_ARC_FRACTION",
    "WITHIN_BRANCH_RULE",
    "DualTSGroupBranchSampler",
    "ExposureTracker",
    "build_sampler_groups",
    "hierarchical_logprob_reduction",
]
