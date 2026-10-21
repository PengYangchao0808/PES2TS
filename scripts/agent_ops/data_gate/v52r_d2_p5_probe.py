#!/usr/bin/env python3
"""V5.2-R D2 phase 2: P5 chemical-endpoint basin supervision (plan §4.6).

Completes the D2 evidence pack: writes ``p5_spec.json`` (PRE-REGISTERED before
ANY label computation or xtb call), builds the ``d_basin`` three-state labels
(``pes2ts_core/pesn/offline/data/basin_labels_v1``), runs the bounded xtb
GFN-2 conformer enrichment (the ONE sanctioned subprocess-QC path — ADR-0002
deleted the pes2ts_core local runner and truth_guard scans pes2ts_core/bin
only; cost is logged per call), runs the 6 validity checks, probes P5 with
the EXACT phase-1 P4 feature builder (goal-free, frame guard active), and
appends the phase-2 block to ``outputs/pesn/v52r/d2/evidence_pack.json`` +
rewrites ``D2_PROBE.md`` (§9 filled, gate extended).

Pre-registered semantics (verbatim in p5_spec.json):

* B_target per curve-intent: (a) IRC reference endpoint frame;
  (b) graph-automorphism images via K-perms of the target-endpoint graph
  (forward → g_tgt, reverse → g_src); (c) conformer enrichment on the FIRST
  10 dev curves (sorted) — forward-intent basins only, the budget-fitting
  pre-registered subset (10 ref SP + ≤3 conformers/curve ≤ 40 calls leaves
  the seed-flip rebuild its share of the global ≤50 cap); (d) no independent
  IRC endpoints (skip).
* d_basin = min over Y of sym-min proper-rotation Kabsch RMSD_K.
* label: 1 ⟺ d_basin ≤ 0.10 AND edit complete; 0 ⟺ d_basin > 0.30 OR
  (d_basin ≤ 0.10 AND incomplete); Mask ⟺ 0.10 < d_basin ≤ 0.30.
  Edit complete = ALL CV pairs ≤ 0.25 Å from the target endpoint frame
  (D0-verbatim, fixed correspondence) → the 22 D0.1 T-rows are NEGATIVES.
* HARD CAPS: ≤50 xtb calls, ≤600 s wall (global across both seeds incl. the
  reference SPs and a CLI pre-flight); abort-enrichment keeps the
  already-built B_target.
* Validity: 6 checks (all must pass; each reported separately in the gate).
* Probe: P4 builder (297-d) → P5; HGB primary; 9 units; same threshold rule.
  PRIMARY P5−P0 on the INTERSECTION population (definitive labels under BOTH
  schemes, non-degenerate); MME median paired Δ ≥ +0.10 AND paired
  curve-cluster bootstrap CI lower bound > 0 (1000×, seed 0).

Report-only: ``training_authorized`` stays False — D2-G is the manager's
review.  ``--strict`` exits 1 when the merged gate verdict is FAIL.

Usage::

    python scripts/agent_ops/data_gate/v52r_d2_p5_probe.py [--strict]
        [--tables-dir ...] [--embeddings-dir outputs/pesn/v52r/d1]
        [--out outputs/pesn/v52r/d2] [--bootstrap 1000]
        [--d01-audit outputs/pesn/v52r/d0_1/topology_anomaly_audit.json]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.agent_ops.data_gate import v52r_d1_probe as d1  # noqa: E402
from scripts.agent_ops.data_gate import v52r_d2_probe as d2  # noqa: E402
from pes2ts_core.pesn.offline.data.basin_labels_v1 import (  # noqa: E402
    BASIN_NEGATIVE_WINDOW_ANGSTROM,
    BASIN_POSITIVE_WINDOW_ANGSTROM,
    CONFORMER_SEED_FLIP,
    CONFORMER_SEED_PRIMARY,
    EDIT_TOLERANCE_ANGSTROM,
    MAX_CONFORMERS_ACCEPTED,
    MAX_GENERATION_ATTEMPTS,
    BasinTarget,
    as_endpoint_stop_labels,
    build_basin_labels,
    build_reference_targets,
    changing_bond_atoms,
    check_t_rows,
    d_basin,
    endpoint_graph,
    flip_stats,
    generate_conformer,
    has_clash,
    population_counts,
    restrict_to,
    rotatable_bonds,
    with_conformers,
)
from pes2ts_core.pesn.offline.data.key_atom_dataset_v3 import (  # noqa: E402
    KeyAtomDatasetV3,
)
from pes2ts_core.pesn.offline.data.key_atom_endpoint_labels_v1 import (  # noqa: E402
    DEGENERATE_TS_CURVES,
    build_endpoint_stop_labels,
)

SCHEMA_ID = "pes2ts_v52r_d2_evidence_v1"      # pack schema unchanged (block appended)
P5_SPEC_SCHEMA = "pes2ts_v52r_d2_p5_spec_v1"
PHASE2_SCHEMA = "pes2ts_v52r_d2_phase2_v1"
GATE_ID = "V5.2R-D2-PHASE1-PROBE"             # gate id unchanged (checks extended)

DEFAULT_TABLES_DIR = REPO_ROOT / "outputs/pesn/30h_campaign/data"
DEFAULT_EMBEDDINGS_DIR = REPO_ROOT / "outputs/pesn/v52r/d1"
DEFAULT_OUT_DIR = REPO_ROOT / "outputs/pesn/v52r/d2"
DEFAULT_D01_AUDIT = REPO_ROOT / "outputs/pesn/v52r/d0_1/topology_anomaly_audit.json"
DEFAULT_D1_PACK = d2.DEFAULT_D1_PACK

DEV_FOLDS = d1.DEV_FOLDS
SEEDS = d1.SEEDS
N_ENRICHED_CURVES = 10
ENRICH_DIRECTION = "forward"          # pre-registered budget-fitting subset
MAX_XTB_CALLS = 50
MAX_XTB_WALL_S = 600.0
MIN_CALL_HEADROOM_S = 5.0
ENERGY_WINDOW_HARTREE = 0.00797       # 5 kcal/mol (pinned by manager)
MAX_XTB_PER_CURVE = 3                 # conformer opts per curve-intent/run
FLIP_RATE_LIMIT = 0.05
POSITIVE_RATE_BAND = (0.15, 0.35)
MME_P5 = d2.MME_P5                    # +0.10
XTB_BIN = "/opt/software/xtb/bin/xtb"
XTB_THREADS = "1"
#: frozen-K recipe: xtb $constrain harmonic restraints on the K list (the
#: exact $fix group does NOT project ANC optimizer displacements in xtb 6.7
#: — verified empirically); 50.0 Eh/Å² matches the repo-proven v53 relax
#: recipe (pes2ts_core.pesn.models.rollout_v3.RELAX_RECIPE_V3).
XTB_FREEZE_FORCE_CONSTANT = 50.0

_ELEMENTS = (
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg",
    "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca", "Sc", "Ti", "V", "Cr",
    "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr",
    "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
    "In", "Sn", "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce")

_ENERGY_RE = re.compile(r"TOTAL ENERGY\s+(-?\d+\.\d+(?:[EeDd][+-]?\d+)?)")

Runner = Callable[..., dict]


def _symbols(atomic: Sequence[int]) -> list[str]:
    out = []
    for z in atomic:
        z = int(z)
        if not 1 <= z <= len(_ELEMENTS):
            raise ValueError(f"atomic number {z} out of table range")
        out.append(_ELEMENTS[z - 1])
    return out


# ---------------------------------------------------------------------------
# pre-registration (pure; written BEFORE any label computation / xtb call)
# ---------------------------------------------------------------------------


def build_p5_spec(*, hashes: dict[str, Any], n_bootstrap: int,
                  subset_curves: Sequence[str],
                  n_dev_rows: int, n_all_rows: int) -> dict[str, Any]:
    """P5 phase-2 pre-registration (pure — deterministic for fixed inputs)."""
    return {
        "schema": P5_SPEC_SCHEMA,
        "stage": "V5.2-R D2 phase 2 — P5 chemical-endpoint basin supervision "
                 "(plan §4.6); appended to the frozen phase-1 pack",
        "plan": "docs/plans/PES2TS_V52R_输入充分性与化学终点识别优化方案_v1_20261021.md §4.6/§5",
        "prerequisite": "D0.1 COMPLETE: 22/22 topology anomalies classified T "
                        "(bond-edit incomplete → Mask under d_end); "
                        "d_end_sym flips 0/1781 — clears plan §3.1 gate for P5",
        "hashes": hashes,
        "label_schema": "pes2ts_v52r_basin_labels_v1",
        "b_target_construction_order": [
            "(a) IRC reference endpoint frame = dataset.frame_xyz(curve, "
            "target_endpoint_node) per curve-intent",
            "(b) graph-automorphism images — handled inside min_sym_rmsd via "
            "K-perms of the TARGET-endpoint graph (forward → g_tgt, "
            "reverse → g_src); identity always included",
            "(c) conformer enrichment: FIRST {n} dev curves by sorted "
            "curve_id — pre-registered subset {subset}; xtb GFN-2 constrained "
            "opt per conformer (K atoms frozen via $fix), clash-rejected "
            "candidates never reach xtb; forward-intent basins only "
            "(budget-fitting reading of '3–5 conformers each' under the "
            "≤50-call cap: 10 ref SP + ≤3×10 opts = 40 for the primary seed, "
            "the remainder reserved for the seed-flip rebuild; reverse/"
            "A-endpoint basins stay reference+automorphism)".format(
                n=N_ENRICHED_CURVES, subset=list(subset_curves)),
            "(d) no independent IRC endpoints in this dataset — skipped",
        ],
        "d_basin": "min over Y ∈ B_target of min_sym_rmsd(X_k, Y_k, K-perms) "
                   "(proper-rotation Kabsch, graph_symmetry.min_sym_rmsd)",
        "basin_label_rule": {
            "positive": f"d_basin ≤ {BASIN_POSITIVE_WINDOW_ANGSTROM} Å AND "
                        "bond-edit complete",
            "negative": f"d_basin > {BASIN_NEGATIVE_WINDOW_ANGSTROM} Å OR "
                        f"(d_basin ≤ {BASIN_POSITIVE_WINDOW_ANGSTROM} Å AND "
                        "bond-edit incomplete)",
            "mask": f"{BASIN_POSITIVE_WINDOW_ANGSTROM} < d_basin ≤ "
                    f"{BASIN_NEGATIVE_WINDOW_ANGSTROM} Å",
            "edit_complete": f"ALL intent CV pairs within "
                             f"{EDIT_TOLERANCE_ANGSTROM} Å of the target "
                             "endpoint frame (D0-verbatim topology proxy, "
                             "fixed correspondence, NO automorphism)",
            "t_rows_expectation": "the 22 D0.1 class-T rows receive basin "
                                  "label 0 (asserted exactly; none positive)",
        },
        "conformer_construction": {
            "seed_primary": CONFORMER_SEED_PRIMARY,
            "seed_flip": CONFORMER_SEED_FLIP,
            "torsion_kicks": "on EVERY rotatable single bond not touching "
                             "changing-bond atoms (both endpoints graph "
                             "degree ≥ 2): θ = sign × Uniform(30°,150°), "
                             "rigid rotation of the smaller side (ties → "
                             "second atom's side)",
            "h_jitter": "Gaussian σ=0.05 Å on H atoms only",
            "clash": "reject any candidate with an interatomic distance "
                     "< 0.8 Å (no xtb call consumed)",
            "xtb": f"{XTB_BIN} --gfn 2 --opt with K-atom positions held by "
                   f"$constrain harmonics (force constant="
                   f"{XTB_FREEZE_FORCE_CONSTANT} Eh/Å², 1-based atom list — "
                   "the repo-proven v53 relax recipe; xtb's exact $fix does "
                   "NOT project ANC optimizer displacements, verified "
                   "empirically), OMP_NUM_THREADS="
                   f"{XTB_THREADS}; per-call K drift logged",
            "acceptance": f"exit 0 AND 'OPTIMIZATION CONVERGED' AND "
                          f"|E − E_ref_SP| ≤ {ENERGY_WINDOW_HARTREE} Hartree "
                          "(5 kcal/mol)",
            "caps_per_curve_per_run": {
                "max_accepted": MAX_CONFORMERS_ACCEPTED,
                "max_generation_attempts": MAX_GENERATION_ATTEMPTS,
                "max_xtb_opts": MAX_XTB_PER_CURVE,
            },
            "hard_caps_global": {
                "max_xtb_calls": MAX_XTB_CALLS,
                "max_wall_seconds": MAX_XTB_WALL_S,
                "covers": "CLI pre-flight + 10 reference single points + "
                          "primary-seed conformer opts + seed-flip rebuild "
                          "opts (flip SPs are seed-independent and reused)",
                "on_exceed": "abort enrichment, KEEP the already-built "
                             "B_target, log the reason",
            },
        },
        "validity_checks": [
            {"id": "reference_in_basin",
             "rule": "reference endpoint frame ∈ B_target: d_basin ≈ 0 "
                     "(tol 1e-6) for every curve-intent"},
            {"id": "endpoints_outside",
             "rule": "for ALL 50 curves (QC counting incl. sealed), the "
                     "A-endpoint frame and the TS frame satisfy d_basin > "
                     "0.30 Å under the FORWARD (B) basin"},
            {"id": "t_rows_negative",
             "rule": "all 22 D0.1 class-T rows carry basin label 0"},
            {"id": "seed_flip",
             "rule": "rebuild the conformer enrichment with seed "
                     f"{CONFORMER_SEED_FLIP} (within the remaining global "
                     "budget); three-state basin-label flip rate on dev "
                     f"usable rows < {FLIP_RATE_LIMIT:.0%} "
                     "(denominator: primary-seed dev usable rows)"},
            {"id": "energy_window",
             "rule": f"all 10 reference SPs obtained; every accepted "
                     f"conformer GFN2 energy within {ENERGY_WINDOW_HARTREE} "
                     "Hartree of its reference endpoint SP"},
            {"id": "positive_rate",
             "rule": f"basin positive rate on dev usable population ∈ "
                     f"[{POSITIVE_RATE_BAND[0]:.0%}, "
                     f"{POSITIVE_RATE_BAND[1]:.0%}]"},
        ],
        "probe": {
            "features": "EXACTLY the phase-1 P4 builder (297-dim: 28 K "
                        "distances + 56 angles + 70 dihedrals + 70 signed "
                        "volumes + 70 cis/trans + K-size + direction one-hot) "
                        "— goal-free; the frame guard stays active; no goal "
                        "coordinates, no d_end",
            "models": d2.build_probe_spec(hashes={}, n_bootstrap=0,
                                           d1_pack_path="")["models"],
            "units": "9 (3 dev folds × 3 seeds), same threshold rule "
                     "(inner curve-group split, precision ≥ 0.80, else ∅)",
            "primary_comparison": "P5 − P0 on the INTERSECTION population "
                                  "(rows definitive under BOTH d_end and "
                                  "basin labels, degenerate rows excluded): "
                                  "P0 = 30-dim D1 features × d_end labels, "
                                  "P5 = 297-dim × basin labels — same rows, "
                                  "same folds/seeds, cross-scheme paired",
            "mme": f"median paired unit ΔAUPRC ≥ +{MME_P5:.2f} AND paired "
                   f"curve-cluster bootstrap 95% CI lower bound > 0 "
                   f"({n_bootstrap}×, seed {d1.BOOTSTRAP_SEED}, curve = unit)",
            "exploratory_no_gating": [
                "P5 − P1 on the intersection (basin labels both sides: "
                "297-d vs 114-d goal features)",
                "P5 − P4 on the intersection (identical 297-d X, basin vs "
                "d_end labels: supervision-only effect)",
                "full-population variants (P5 basin-usable vs P0 d_end-usable "
                "— descriptive, populations differ)",
            ],
        },
        "seed_set": list(SEEDS),
        "fold_protocol": "dev folds 0–2 only; sealed folds 3/4 label-QC "
                         "counting only (never encoded, never probed); "
                         "train = other two dev folds, eval = current, "
                         "curve-grouped",
        "bootstrap": {"method": "curve cluster bootstrap", "n_resamples": int(n_bootstrap),
                      "seed": d1.BOOTSTRAP_SEED},
        "population_sizes": {"n_rows_all": int(n_all_rows),
                             "n_rows_dev": int(n_dev_rows)},
        "report_only": "phase 2 stays report-only: training_authorized "
                       "remains False; D2-G is the manager's review",
        "subprocess_note": "direct subprocess from this script is sanctioned "
                           "for this QC only (truth_guard scans "
                           "pes2ts_core/bin; ADR-0002 deleted the "
                           "pes2ts_core local runner); every call's seconds/"
                           "exit code/energy is logged in the pack",
        "phase1_spec_untouched": "outputs/pesn/v52r/d2/probe_spec.json is "
                                 "NOT rewritten by phase 2; its sha256 chains "
                                 "into hashes.phase1_probe_spec_sha256",
    }


def spec_text(spec: dict[str, Any]) -> str:
    return d2.spec_text(spec)          # canonical sort_keys + indent=2 + \n


def spec_sha256(spec: dict[str, Any]) -> str:
    return d2.spec_sha256(spec)


# ---------------------------------------------------------------------------
# xtb driver (script-side subprocess — sanctioned QC only)
# ---------------------------------------------------------------------------


@dataclass
class XtbBudget:
    """Global hard caps on xtb invocations and wall time (both seeds)."""

    max_calls: int = MAX_XTB_CALLS
    max_seconds: float = MAX_XTB_WALL_S
    calls: int = 0
    wall: float = 0.0
    log: list[dict] = field(default_factory=list)

    def can_call(self) -> bool:
        return (self.calls < self.max_calls
                and self.remaining_seconds() > MIN_CALL_HEADROOM_S)

    def remaining_seconds(self) -> float:
        return max(0.0, float(self.max_seconds) - float(self.wall))

    def exhausted_reason(self) -> str | None:
        if self.calls >= self.max_calls:
            return "call_cap"
        if self.remaining_seconds() <= MIN_CALL_HEADROOM_S:
            return "wall_cap"
        return None

    def record(self, tag: Mapping[str, Any], res: Mapping[str, Any]) -> dict:
        entry = {
            "purpose": tag.get("purpose"), "curve_id": tag.get("curve_id"),
            "seed": tag.get("seed"),
            "seconds": round(float(res.get("wall_seconds") or 0.0), 3),
            "exit_code": res.get("exit_code"),
            "energy_ha": (None if res.get("energy_ha") is None
                          else round(float(res["energy_ha"]), 9)),
            "converged": bool(res.get("converged")),
            "ok": bool(res.get("ok")),
            "k_drift_max": (None if res.get("k_drift_max") is None
                            else round(float(res["k_drift_max"]), 6)),
            "error": res.get("error"),
        }
        self.calls += 1
        self.wall += float(res.get("wall_seconds") or 0.0)
        self.log.append(entry)
        return entry


def _write_xyz(path: Path, symbols: Sequence[str],
               coords: np.ndarray) -> None:
    lines = [str(len(symbols)), "pes2ts v52r d2 p5 conformer"]
    for el, row in zip(symbols, np.asarray(coords, dtype=np.float64)):
        lines.append(f"{el:2s} {row[0]: .10f} {row[1]: .10f} {row[2]: .10f}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _parse_energy(stdout: str) -> float | None:
    hits = _ENERGY_RE.findall(stdout or "")
    if not hits:
        return None
    try:
        return float(hits[-1].replace("D", "E").replace("d", "e"))
    except ValueError:
        return None


def xtb_invoke(coords: np.ndarray, symbols: Sequence[str], *,
               freeze_atoms: Sequence[int] = (), mode: str = "sp",
               timeout_s: float = 120.0,
               xtb_bin: str = XTB_BIN) -> dict:
    """ONE xtb call in a private temp workdir (mirrors the v53 relax recipe).

    ``freeze_atoms`` are 0-based rows → 1-based ``$fix atoms:`` list.
    ``mode``: ``sp`` (single point) or ``opt`` (constrained optimization).
    """
    if mode not in ("sp", "opt"):
        raise ValueError(f"mode must be sp|opt, got {mode!r}")
    coords = np.asarray(coords, dtype=np.float64)
    t0 = time.time()
    with tempfile.TemporaryDirectory(prefix="p5xtb_") as td:
        wd = Path(td)
        _write_xyz(wd / "coord.xyz", symbols, coords)
        cmd = [xtb_bin, "coord.xyz", "--gfn", "2"]
        if freeze_atoms and mode == "opt":
            one_based = ",".join(str(int(a) + 1) for a in freeze_atoms)
            (wd / "fix.inp").write_text(
                "$constrain\n"
                f"  force constant={XTB_FREEZE_FORCE_CONSTANT}\n"
                f"  atoms: {one_based}\n$end\n", encoding="utf-8")
            cmd += ["--input", "fix.inp"]
        if mode == "opt":
            cmd.append("--opt")
        env = dict(os.environ)
        env["OMP_NUM_THREADS"] = XTB_THREADS
        try:
            proc = subprocess.Popen(
                cmd, cwd=str(wd), env=env, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, text=True, start_new_session=True)
            try:
                out, err = proc.communicate(timeout=float(timeout_s))
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, 9)
                out, err = proc.communicate()
                wall = time.time() - t0
                return {"ok": False, "converged": False, "energy_ha": None,
                        "wall_seconds": wall, "exit_code": None,
                        "coords": None, "error": "timeout"}
        except FileNotFoundError as exc:
            return {"ok": False, "converged": False, "energy_ha": None,
                    "wall_seconds": time.time() - t0, "exit_code": None,
                    "coords": None, "error": f"launch failed: {exc}"}
        wall = time.time() - t0
        stdout = out or ""
        rc = proc.returncode
        usage_error = (rc == 128 or "Command line argument parsing failed"
                       in stdout)
        energy = _parse_energy(stdout)
        if mode == "sp":
            converged = energy is not None and rc == 0
            final_coords = coords
        else:
            converged = ("OPTIMIZATION CONVERGED" in stdout
                         and rc == 0 and energy is not None)
            final_coords = None
            opt_xyz = wd / "xtbopt.xyz"
            if opt_xyz.is_file():
                rows = opt_xyz.read_text(encoding="utf-8").splitlines()
                try:
                    n = int(rows[0].strip())
                    final_coords = np.array(
                        [[float(x) for x in line.split()[1:4]]
                         for line in rows[2:2 + n]], dtype=np.float64)
                except (ValueError, IndexError):
                    final_coords = None
        ok = converged and (final_coords is not None)
        k_drift = None
        if (final_coords is not None and freeze_atoms and mode == "opt"):
            idx = np.asarray([int(a) for a in freeze_atoms], dtype=np.int64)
            k_drift = float(np.abs(final_coords[idx] - coords[idx]).max())
        error = None
        if usage_error:
            error = "xtb usage error (command line argument parsing failed)"
        elif not ok:
            error = ("no xtbopt.xyz" if mode == "opt" and final_coords is None
                     else f"exit={rc} converged={converged}")
        return {"ok": bool(ok), "converged": bool(converged),
                "energy_ha": energy, "wall_seconds": wall, "exit_code": rc,
                "coords": final_coords, "error": error, "k_drift_max": k_drift}


def make_runner(budget: XtbBudget, xtb_bin: str = XTB_BIN) -> Runner:
    """Budget-checked runner used by SP/enrichment (and mocked in tests)."""

    def runner(coords: np.ndarray, *, symbols: Sequence[str],
               freeze_atoms: Sequence[int] = (), mode: str = "sp",
               tag: Mapping[str, Any]) -> dict:
        if not budget.can_call():
            return {"ok": False, "converged": False, "energy_ha": None,
                    "wall_seconds": 0.0, "exit_code": None, "coords": None,
                    "error": f"budget exhausted: {budget.exhausted_reason()}",
                    "budget_exhausted": budget.exhausted_reason() or "exhausted"}
        timeout = budget.remaining_seconds()
        res = xtb_invoke(coords, symbols, freeze_atoms=freeze_atoms,
                         mode=mode, timeout_s=timeout, xtb_bin=xtb_bin)
        entry = budget.record(tag, res)
        res["log_entry"] = entry
        res["budget_exhausted"] = False
        return res
    return runner


# ---------------------------------------------------------------------------
# enrichment driver (runner injectable → unit-testable without xtb)
# ---------------------------------------------------------------------------


@dataclass
class ConformerJob:
    curve_id: str
    ref_frame: np.ndarray          # [N, 3] reference endpoint (B side)
    atomic: tuple[int, ...]
    k_rows: tuple[int, ...]
    rot_bonds: tuple[tuple[int, int], ...]
    graph: Any
    symbols: tuple[str, ...]


def build_conformer_jobs(dataset: KeyAtomDatasetV3,
                         subset_curves: Sequence[str]) -> list[ConformerJob]:
    """Pre-registered jobs: forward-intent (target B) basins of the subset."""
    forward_intents = {cid: it for (cid, _iid), it in dataset.intents.items()
                       if it.direction == ENRICH_DIRECTION}
    atomic_by_curve: dict[str, tuple[int, ...]] = {}
    for rec in dataset.samples:
        atomic_by_curve.setdefault(rec.curve_id,
                                   tuple(int(z) for z in rec.atomic_numbers))
    jobs: list[ConformerJob] = []
    for cid in subset_curves:
        it = forward_intents.get(cid)
        if it is None:
            raise KeyError(f"curve {cid} has no forward intent")
        graph = endpoint_graph(it, ENRICH_DIRECTION)
        changing = changing_bond_atoms(it)
        rot = rotatable_bonds(graph, changing)
        atomic = atomic_by_curve[cid]
        target_node = _target_node_for(dataset, cid, ENRICH_DIRECTION)
        jobs.append(ConformerJob(
            curve_id=cid,
            ref_frame=np.asarray(dataset.frame_xyz(cid, target_node),
                                 dtype=np.float64),
            atomic=atomic,
            k_rows=tuple(int(k) for k in it.key_rows()),
            rot_bonds=rot, graph=graph, symbols=tuple(_symbols(atomic))))
    return jobs


def _target_node_for(dataset: KeyAtomDatasetV3, curve_id: str,
                     direction: str) -> int:
    a, b = dataset.curve_endpoints(curve_id)
    return int(b) if direction == "forward" else int(a)


def cli_pre_flight(runner: Runner) -> dict:
    """ONE constrained opt on water — validates CLI/$fix/parse BEFORE spend."""
    water = np.array([[0.0, 0.0, 0.119],
                      [0.0, 0.763, -0.477],
                      [0.0, -0.763, -0.477]], dtype=np.float64)
    return runner(water, symbols=("O", "H", "H"), freeze_atoms=(0, 1, 2),
                  mode="opt",
                  tag={"purpose": "cli_pre_flight", "curve_id": None,
                       "seed": None})


def run_reference_sps(jobs: Sequence[ConformerJob], runner: Runner,
                      ) -> tuple[dict[str, float], str | None]:
    """GFN-2 single point per enriched reference endpoint (counts to budget)."""
    energies: dict[str, float] = {}
    for job in jobs:
        res = runner(job.ref_frame, symbols=job.symbols, freeze_atoms=(),
                     mode="sp",
                     tag={"purpose": "reference_sp", "curve_id": job.curve_id,
                          "seed": None})
        if res.get("budget_exhausted"):
            reason = res["budget_exhausted"]
            return energies, (f"budget_exhausted_before_reference_sp"
                              f"({reason})" if isinstance(reason, str)
                              else "budget_exhausted_before_reference_sp")
        if not res.get("ok") or res.get("energy_ha") is None:
            return energies, f"reference_sp_failed:{job.curve_id}"
        energies[job.curve_id] = float(res["energy_ha"])
    return energies, None


def enrich_conformers(jobs: Sequence[ConformerJob], *, seed: int,
                      runner: Runner, sp_energies: Mapping[str, float],
                      ) -> tuple[dict[str, list[dict]], str | None]:
    """Round-robin conformer enrichment under the global budget.

    Per curve-intent: ≤ ``MAX_GENERATION_ATTEMPTS`` candidate generations
    (clash rejects consume an attempt but NO xtb call), ≤
    ``MAX_XTB_PER_CURVE`` xtb opts, ≤ ``MAX_CONFORMERS_ACCEPTED`` accepted
    (converged + within the energy window).  Returns accepted-conformer
    records and an abort reason (None = completed).  Deterministic per seed.
    """
    rng = np.random.RandomState(int(seed))
    accepted: dict[str, list[dict]] = {j.curve_id: [] for j in jobs}
    state = {j.curve_id: {"attempts": 0, "xtb": 0} for j in jobs}
    abort: str | None = None
    progress = True
    while progress and abort is None:
        progress = False
        for job in jobs:
            st = state[job.curve_id]
            if (st["attempts"] >= MAX_GENERATION_ATTEMPTS
                    or st["xtb"] >= MAX_XTB_PER_CURVE
                    or len(accepted[job.curve_id]) >= MAX_CONFORMERS_ACCEPTED):
                continue
            progress = True
            st["attempts"] += 1
            cand = generate_conformer(job.ref_frame, job.rot_bonds,
                                      job.graph, job.atomic, rng)
            if has_clash(cand):
                continue
            res = runner(cand, symbols=job.symbols, freeze_atoms=job.k_rows,
                         mode="opt",
                         tag={"purpose": "conformer_opt", "curve_id": job.curve_id,
                              "seed": int(seed)})
            if res.get("budget_exhausted"):
                reason = res["budget_exhausted"]
                abort = reason if isinstance(reason, str) else "budget_exhausted"
                break
            st["xtb"] += 1
            entry = res.get("log_entry")
            energy = res.get("energy_ha")
            within = (energy is not None and job.curve_id in sp_energies
                      and abs(float(energy) - float(sp_energies[job.curve_id]))
                      <= ENERGY_WINDOW_HARTREE)
            if entry is not None:
                entry["within_window"] = bool(within)
            if (res.get("ok") and within and res.get("coords") is not None):
                if entry is not None:
                    entry["accepted"] = True
                accepted[job.curve_id].append({
                    "coords": np.asarray(res["coords"], dtype=np.float64),
                    "energy_ha": float(energy),
                    "seed": int(seed),
                })
            elif entry is not None:
                entry["accepted"] = False
    return accepted, abort


# ---------------------------------------------------------------------------
# validity checks (pure parts unit-tested)
# ---------------------------------------------------------------------------


def validity_reference_in_basin(
    targets: Mapping[tuple[str, str], BasinTarget],
) -> dict:
    worst = 0.0
    worst_key = None
    n = 0
    for key, tgt in sorted(targets.items()):
        val = d_basin(tgt.ref_frame, tgt)
        n += 1
        if val > worst:
            worst, worst_key = val, key
    ok = worst <= 1e-6
    return {"id": "reference_in_basin",
            "expect": "reference endpoint frame ∈ B_target (d_basin ≈ 0, "
                      "tol 1e-6) for every curve-intent",
            "actual": f"n={n} curve-intents, max d_basin(ref)={worst:.3e}"
                      + (f" at {worst_key}" if worst_key else ""),
            "pass": bool(ok), "max_deviation": worst}


def validity_endpoints_outside(
    dataset: KeyAtomDatasetV3,
    targets_forward: Mapping[str, BasinTarget],
) -> dict:
    """A-endpoint + TS frame must sit OUTSIDE every forward (B) basin."""
    violations: list[dict] = []
    n_checked = 0
    worst = 0.0
    for cid, tgt in sorted(targets_forward.items()):
        a_node, _b = dataset.curve_endpoints(cid)
        ts_node = dataset.curve_ts_index(cid)
        for kind, node in (("A_endpoint", int(a_node)), ("TS", int(ts_node))):
            frame = np.asarray(dataset.frame_xyz(cid, node), dtype=np.float64)
            val = d_basin(frame, tgt)
            n_checked += 1
            worst = max(worst, val)
            if not val > BASIN_NEGATIVE_WINDOW_ANGSTROM:
                violations.append({"curve_id": cid, "frame": kind,
                                   "d_basin": round(val, 6),
                                   "degenerate_curve":
                                       cid in DEGENERATE_TS_CURVES})
    deg_only = bool(violations) and all(v["degenerate_curve"]
                                        for v in violations)
    n_deg_viol = sum(1 for v in violations if v["degenerate_curve"])
    deg_curves = sorted({v["curve_id"] for v in violations
                         if v["degenerate_curve"]})
    n_checked_excl = n_checked - 2 * len(deg_curves)
    return {"id": "endpoints_outside",
            "expect": "all 50 curves (QC incl. sealed): A-endpoint frame and "
                      "TS frame have d_basin > 0.30 Å under the forward (B) "
                      "basin",
            "actual": f"checked {n_checked} frames (50 curves × 2), worst "
                      f"d_basin={worst:.6f}, violations={len(violations)}"
                      + (f" {[(v['curve_id'], v['frame']) for v in violations[:6]]}"
                         if violations else ""),
            "pass": not violations,
            "n_checked": n_checked, "n_violations": len(violations),
            "violations": violations, "violations_all_rule4_degenerate":
                deg_only, "n_rule4_violations": n_deg_viol,
            "rule4_degenerate_curves": deg_curves,
            "n_checked_excluding_rule4": n_checked_excl,
            "pass_excluding_rule4": bool(
                not violations or (deg_only and len(violations) == n_deg_viol)),
            "worst": round(worst, 6)}


def validity_energy_window(sp_energies: Mapping[str, float],
                           accepted: Mapping[str, Sequence[dict]],
                           subset_curves: Sequence[str]) -> dict:
    missing_sp = [c for c in subset_curves if c not in sp_energies]
    devs: list[float] = []
    n_accepted = 0
    n_out = 0
    for cid in subset_curves:
        e_ref = sp_energies.get(cid)
        for rec in accepted.get(cid, ()):
            n_accepted += 1
            if e_ref is None:
                n_out += 1
                continue
            dev = abs(float(rec["energy_ha"]) - float(e_ref))
            devs.append(dev)
            if dev > ENERGY_WINDOW_HARTREE:
                n_out += 1
    ok = not missing_sp and n_out == 0
    max_dev = max(devs) if devs else None
    return {"id": "energy_window",
            "expect": f"10 reference SPs obtained; every accepted conformer "
                      f"|ΔE| ≤ {ENERGY_WINDOW_HARTREE} Hartree (5 kcal/mol) "
                      "vs its reference endpoint SP",
            "actual": f"sp {len(sp_energies)}/10, accepted={n_accepted}, "
                      f"outside_window={n_out}, max|ΔE|="
                      + (f"{max_dev:.6f} Ha" if max_dev is not None else "n/a"),
            "pass": bool(ok), "n_ref_sp": len(sp_energies),
            "n_accepted": n_accepted, "n_outside_window": n_out,
            "max_abs_dev_ha": (round(max_dev, 9) if max_dev is not None
                               else None),
            "missing_sp_curves": missing_sp}


def validity_seed_flip(flip: dict) -> dict:
    rate = flip.get("flip_rate")
    ok = rate is not None and float(rate) < FLIP_RATE_LIMIT
    return {"id": "seed_flip",
            "expect": f"basin-label flip rate (seed {CONFORMER_SEED_PRIMARY} "
                      f"vs {CONFORMER_SEED_FLIP}) on dev usable rows < "
                      f"{FLIP_RATE_LIMIT:.0%}",
            "actual": f"{flip.get('n_flips')}/{flip.get('n_rows')} = {rate} "
                      f"transitions={flip.get('transitions')}",
            "pass": bool(ok)}


def validity_t_rows(t_rows: dict) -> dict:
    return {"id": "t_rows_negative",
            "expect": "all 22 D0.1 class-T rows carry basin label 0 "
                      "(edit incomplete → never positive)",
            "actual": f"{t_rows.get('n_ok')}/{t_rows.get('n_rows')} label 0 "
                      f"(pos={t_rows.get('n_positive')}, "
                      f"mask={t_rows.get('n_mask')}, "
                      f"missing={len(t_rows.get('missing', []))})",
            "pass": bool(t_rows.get("pass"))}


def validity_positive_rate(pop_dev_usable: dict) -> dict:
    rate = pop_dev_usable.get("usable_positive_rate")
    ok = (rate is not None and POSITIVE_RATE_BAND[0] <= float(rate)
          <= POSITIVE_RATE_BAND[1])
    return {"id": "positive_rate",
            "expect": f"basin positive rate on dev usable population ∈ "
                      f"[{POSITIVE_RATE_BAND[0]:.0%}, "
                      f"{POSITIVE_RATE_BAND[1]:.0%}]",
            "actual": f"{pop_dev_usable.get('usable_positive')}/"
                      f"{pop_dev_usable.get('n_rows')} = {rate}",
            "pass": bool(ok)}


# ---------------------------------------------------------------------------
# statistics helpers (unit-tested)
# ---------------------------------------------------------------------------


def unit_deltas(units_a: Sequence[dict], key_a: str,
                units_b: Sequence[dict], key_b: str) -> dict:
    """Paired per-(fold, seed) ΔAUPRC between two 9-unit runs."""
    index = {(u["fold"], u["seed"]): u for u in units_b}
    per: list[dict] = []
    for u in sorted(units_a, key=lambda u: (u["fold"], u["seed"])):
        other = index.get((u["fold"], u["seed"]))
        if other is None:
            continue
        a = u.get("probes", {}).get(key_a, {}).get("auprc")
        b = other.get("probes", {}).get(key_b, {}).get("auprc")
        if a is None or b is None:
            continue
        per.append({"fold": u["fold"], "seed": u["seed"],
                    "delta": d1._r6(float(a) - float(b))})
    vals = [p["delta"] for p in per]
    return {"n_units": len(vals),
            "median": d1._r6(np.median(vals)) if vals else None,
            "min": d1._r6(min(vals)) if vals else None,
            "max": d1._r6(max(vals)) if vals else None,
            "units": per}


def paired_curve_bootstrap_two_pops(
    scores_a: np.ndarray, labels_a: np.ndarray, curves_a: np.ndarray,
    scores_b: np.ndarray, labels_b: np.ndarray, curves_b: np.ndarray,
    *, n_resamples: int = d1.BOOTSTRAP_RESAMPLES,
    seed: int = d1.BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Curve-cluster bootstrap of ΔAUPRC across TWO populations (descriptive).

    Each side keeps its OWN rows/labels (full-population variants); the SAME
    curve resample indexes both sides.  Requires an identical curve set.
    """
    from sklearn.metrics import average_precision_score

    scores_a = np.asarray(scores_a, dtype=np.float64)
    labels_a = np.asarray(labels_a, dtype=np.int64)
    curves_a = np.asarray(curves_a, dtype=object)
    scores_b = np.asarray(scores_b, dtype=np.float64)
    labels_b = np.asarray(labels_b, dtype=np.int64)
    curves_b = np.asarray(curves_b, dtype=object)
    curves = np.array(sorted(set(curves_a.tolist())))
    if set(curves_b.tolist()) != set(curves.tolist()):
        raise ValueError("two-population bootstrap: curve sets differ")
    point_a = float(average_precision_score(labels_a, scores_a)) \
        if labels_a.size else None
    point_b = float(average_precision_score(labels_b, scores_b)) \
        if labels_b.size else None
    rng = np.random.RandomState(int(seed))
    deltas: list[float] = []
    for _ in range(int(n_resamples)):
        picked = rng.choice(curves, size=curves.size, replace=True)
        parts_a = [np.flatnonzero(curves_a == c) for c in picked]
        parts_b = [np.flatnonzero(curves_b == c) for c in picked]
        idx_a = np.concatenate(parts_a) if parts_a else np.array([], dtype=np.int64)
        idx_b = np.concatenate(parts_b) if parts_b else np.array([], dtype=np.int64)
        if idx_a.size == 0 or idx_b.size == 0:
            continue
        ya, yb = labels_a[idx_a], labels_b[idx_b]
        if ya.min() == ya.max() or yb.min() == yb.max():
            continue
        deltas.append(float(average_precision_score(ya, scores_a[idx_a])
                            - average_precision_score(yb, scores_b[idx_b])))
    ci = ({"lower": d1._r6(np.percentile(deltas, 2.5)),
           "upper": d1._r6(np.percentile(deltas, 97.5))} if deltas
          else {"lower": None, "upper": None})
    point = (d1._r6(point_a - point_b)
             if point_a is not None and point_b is not None else None)
    return {"point": point, "point_a": d1._r6(point_a),
            "point_b": d1._r6(point_b), "ci95": ci,
            "prevalence_a": d1._r6(labels_a.mean()) if labels_a.size else None,
            "prevalence_b": d1._r6(labels_b.mean()) if labels_b.size else None,
            "two_populations": True,
            "n_resamples": int(n_resamples),
            "n_valid_resamples": len(deltas), "n_curves": int(curves.size),
            "seed": int(seed)}


def _d_end_counts(table, sample_ids: Iterable[str]) -> dict:
    """population_counts() schema for an EndpointStopLabelTable."""
    ids = [str(s) for s in sample_ids]
    pos = neg = mask = deg = 0
    usable_pos = usable_neg = 0
    for sid in ids:
        lab = table.by_sample.get(sid)
        if lab is None:
            continue
        if lab.degenerate_flag:
            deg += 1
        if lab.endpoint_stop_label == 1:
            pos += 1
        elif lab.endpoint_stop_label == 0:
            neg += 1
        else:
            mask += 1
        if not lab.degenerate_flag and lab.endpoint_stop_label >= 0:
            if lab.endpoint_stop_label == 1:
                usable_pos += 1
            else:
                usable_neg += 1
    usable = usable_pos + usable_neg
    return {"n_rows": len(ids), "positive": pos, "negative": neg,
            "mask": mask, "degenerate_rows": deg, "usable_rows": usable,
            "usable_positive": usable_pos, "usable_negative": usable_neg,
            "usable_positive_rate": (usable_pos / usable) if usable else None}


def _usable_subset(pop: dict) -> dict:
    """The 'usable' slice of a population_counts() dict as its own row."""
    return {"n_rows": pop["usable_rows"], "positive": pop["usable_positive"],
            "negative": pop["usable_negative"], "mask": 0,
            "degenerate_rows": 0, "usable_rows": pop["usable_rows"],
            "usable_positive": pop["usable_positive"],
            "usable_negative": pop["usable_negative"],
            "usable_positive_rate": pop["usable_positive_rate"]}


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def _collect_hashes(tables_dir: Path, d01_audit: Path, p5_script: Path,
                    phase1_spec: Path) -> dict[str, Any]:
    data_manifest = tables_dir / d1.DATA_MANIFEST_FILENAME
    dataset_sha = ""
    if data_manifest.is_file():
        doc = json.loads(data_manifest.read_text(encoding="utf-8"))
        dataset_sha = str(doc.get("dataset_sha256", ""))
    return {
        "code_commit": d1.git_head(),
        "data_manifest_sha256": d1.sha256_file(data_manifest)
        if data_manifest.is_file() else "",
        "dataset_sha256": dataset_sha,
        "graph_symmetry_sha256": d1.sha256_file(
            REPO_ROOT / "pes2ts_core/pesn/offline/data/graph_symmetry.py"),
        "basin_labels_v1_sha256": d1.sha256_file(
            REPO_ROOT / "pes2ts_core/pesn/offline/data/basin_labels_v1.py"),
        "d01_audit_sha256": d1.sha256_file(d01_audit)
        if d01_audit.is_file() else "",
        "phase1_probe_spec_sha256": d1.sha256_file(phase1_spec)
        if phase1_spec.is_file() else "",
        "phase1_d2_probe_script_sha256": d1.sha256_file(
            Path(d2.__file__)),
        "d2_p5_probe_script_sha256": d1.sha256_file(p5_script),
        "label_builder_sha256": d1.sha256_file(
            REPO_ROOT / "pes2ts_core/pesn/offline/data/key_atom_endpoint_labels_v1.py"),
    }


def run(tables_dir: Path, out_dir: Path, embeddings_dir: Path, *,
        n_bootstrap: int = d1.BOOTSTRAP_RESAMPLES,
        d01_audit_path: Path = DEFAULT_D01_AUDIT,
        d1_pack_path: Path = DEFAULT_D1_PACK,
        xtb_bin: str = XTB_BIN) -> dict[str, Any]:
    t_start = time.time()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pack_path = out_dir / "evidence_pack.json"
    spec1_path = out_dir / "probe_spec.json"
    if not pack_path.is_file() or not spec1_path.is_file():
        raise FileNotFoundError(
            "phase 2 appends to the phase-1 pack: run v52r_d2_probe.py first")
    pack = json.loads(pack_path.read_text(encoding="utf-8"))
    probe_spec = json.loads(spec1_path.read_text(encoding="utf-8"))
    if d2.spec_sha256(probe_spec) != pack.get("probe_spec_sha256"):
        raise RuntimeError(
            "probe_spec.json no longer matches the pack sha — phase 2 must "
            "leave the phase-1 spec UNTOUCHED")

    # ---- dataset / embeddings (no label computation yet) -------------------
    dataset = KeyAtomDatasetV3(
        tables_dir, label_statuses=("ok", "terminal", "masked", "conflict"))
    samples = dataset.samples
    dev_positions = [i for i, s in enumerate(samples) if s.fold in DEV_FOLDS]
    sealed_positions = [i for i, s in enumerate(samples)
                        if s.fold in d1.SEALED_FOLDS]
    meta = d1.load_meta(dataset, dev_positions)
    z_by_unit, h_by_unit, _ = d2.load_embeddings(Path(embeddings_dir), meta)
    dev_curves = sorted({str(s.curve_id) for s in samples
                         if s.fold in DEV_FOLDS})
    subset_curves = dev_curves[:N_ENRICHED_CURVES]

    # ---- PRE-REGISTRATION: p5_spec.json BEFORE labels / xtb ----------------
    t0 = time.time()
    hashes = _collect_hashes(Path(tables_dir), Path(d01_audit_path),
                             Path(__file__), spec1_path)
    p5_spec = build_p5_spec(hashes=hashes, n_bootstrap=n_bootstrap,
                            subset_curves=subset_curves,
                            n_dev_rows=len(dev_positions),
                            n_all_rows=len(samples))
    p5_sha = spec_sha256(p5_spec)
    (out_dir / "p5_spec.json").write_text(spec_text(p5_spec),
                                          encoding="utf-8")
    t_spec = time.time() - t0

    # ---- labels: phase-1 d_end rebuild + D0.1 audit rows -------------------
    t0 = time.time()
    v52_labels = dataset.v52_labels()
    d_end = build_endpoint_stop_labels(dataset, v52_labels)
    audit = json.loads(Path(d01_audit_path).read_text(encoding="utf-8"))
    t_rows = audit.get("rows", [])
    if len(t_rows) != 22 or any(r.get("class") != "T" for r in t_rows):
        raise RuntimeError(
            f"D0.1 audit must hold exactly 22 class-T rows, got "
            f"{len(t_rows)} rows / classes "
            f"{sorted({r.get('class') for r in t_rows})}")
    t_ids = [str(r["sample_id"]) for r in t_rows]

    # ---- B_target: reference for all 50 curves, then enrichment ------------
    targets_ref = build_reference_targets(dataset, d_end)
    jobs = build_conformer_jobs(dataset, subset_curves)
    budget = XtbBudget(max_calls=MAX_XTB_CALLS, max_seconds=MAX_XTB_WALL_S)
    runner = make_runner(budget, xtb_bin=xtb_bin)

    enrichment_abort: str | None = None
    sp_energies: dict[str, float] = {}
    accepted_primary: dict[str, list[dict]] = {j.curve_id: [] for j in jobs}
    accepted_flip: dict[str, list[dict]] = {j.curve_id: [] for j in jobs}

    pre = cli_pre_flight(runner)
    if not (pre.get("ok") or pre.get("converged")):
        enrichment_abort = "cli_pre_flight_failed"
    if enrichment_abort is None:
        sp_energies, enrichment_abort = run_reference_sps(jobs, runner)
    if enrichment_abort is None:
        accepted_primary, enrichment_abort = enrich_conformers(
            jobs, seed=CONFORMER_SEED_PRIMARY, runner=runner,
            sp_energies=sp_energies)
    flip_abort: str | None = None
    if sp_energies:
        # seed-flip rebuild with whatever global budget remains (immediately
        # aborts with the budget reason when none is left)
        accepted_flip, flip_abort = enrich_conformers(
            jobs, seed=CONFORMER_SEED_FLIP, runner=runner,
            sp_energies=sp_energies)

    targets_primary = dict(targets_ref)
    targets_flip = dict(targets_ref)
    for job in jobs:
        key = (job.curve_id, ENRICH_DIRECTION)
        if accepted_primary.get(job.curve_id):
            targets_primary[key] = with_conformers(
                targets_ref[key],
                [c["coords"] for c in accepted_primary[job.curve_id]])
        if accepted_flip.get(job.curve_id):
            targets_flip[key] = with_conformers(
                targets_ref[key],
                [c["coords"] for c in accepted_flip[job.curve_id]])

    labels_basin = build_basin_labels(dataset, d_end, targets_primary)
    labels_flip = build_basin_labels(dataset, d_end, targets_flip)

    # ---- validity checks ---------------------------------------------------
    all_sids = [str(s.sample_id) for s in samples]
    dev_sids = [str(meta["sample_ids"][i]) for i in range(len(dev_positions))]
    forward_targets = {cid: targets_primary[(cid, ENRICH_DIRECTION)]
                       for cid in sorted({str(s.curve_id) for s in samples})}

    v1 = validity_reference_in_basin(targets_primary)
    v2 = validity_endpoints_outside(dataset, forward_targets)
    t_rows_check = check_t_rows(t_ids, labels_basin)
    v3 = validity_t_rows(t_rows_check)
    pop_full = population_counts(labels_basin, all_sids)
    pop_dev = population_counts(labels_basin, dev_sids)
    pop_dev_usable = _usable_subset(pop_dev)
    dev_usable_ids = [sid for sid in dev_sids
                      if not labels_basin.by_sample[sid].degenerate_flag
                      and labels_basin.by_sample[sid].basin_label >= 0]
    flip = flip_stats(labels_basin.labels_by_sample(),
                      labels_flip.labels_by_sample(), dev_usable_ids)
    flip = {"seed_a": CONFORMER_SEED_PRIMARY, "seed_b": CONFORMER_SEED_FLIP,
            **flip}
    v4 = validity_seed_flip(flip)
    v5 = validity_energy_window(sp_energies, accepted_primary,
                                subset_curves)
    v6 = validity_positive_rate(pop_dev_usable)
    validity_checks = [v1, v2, v3, v4, v5, v6]
    n_pass = sum(1 for c in validity_checks if c["pass"])
    validity_notes: list[str] = []
    if not v2["pass"]:
        validity_notes.append(
            f"endpoints_outside 违反 {v2['n_violations']} 项；其中 rule-4 "
            f"退化曲线贡献 {v2['n_rule4_violations']} 项 "
            f"(curves {v2['rule4_degenerate_curves']})，全部为 TS 帧：这些行"
            f"在两种标签方案下均被 degenerate_flag 排除出全部探针 train/eval；"
            f"剔除 rule-4 曲线后 {v2['n_checked_excluding_rule4']} 帧全部通过 "
            f"= {v2['pass_excluding_rule4']}")
    if enrichment_abort:
        validity_notes.append(
            f"富集中止: {enrichment_abort}（已构建的 B_target 保留；"
            f"flip 重建: {flip_abort or 'n/a'}）")

    d_end_full_counts = _d_end_counts(d_end, all_sids)
    d_end_dev_counts = _d_end_counts(d_end, dev_sids)
    population = {
        "full_table": pop_full,
        "dev": pop_dev,
        "dev_usable": pop_dev_usable,
        "d_end_full": d_end_full_counts,
        "d_end_dev_usable": _usable_subset(d_end_dev_counts),
    }
    t_labels = time.time() - t0

    # ---- features (frame guard active; P5 = P4 builder) --------------------
    t0 = time.time()
    blocks, feat_info = d2.build_probe_blocks(dataset, dev_positions, d_end)
    feat_info["per_probe"]["P5"] = dict(feat_info["per_probe"]["P4"])
    feat_info["per_probe"]["P5"]["purpose"] = (
        "P5 probe: EXACTLY the P4 feature builder (297-dim, goal-free, "
        "frame-guard active) evaluated on the P5 basin labels")
    x_p5 = blocks["P4"]
    x_p0 = blocks["P0"]
    x_p1 = blocks["P1"]
    x_p4 = blocks["P4"]
    t_feat = time.time() - t0

    # ---- label tables: full + intersection --------------------------------
    tbl_basin_full = as_endpoint_stop_labels(labels_basin)
    basin_def = {sid for sid in dev_sids
                 if not labels_basin.by_sample[sid].degenerate_flag
                 and labels_basin.by_sample[sid].basin_label >= 0}
    dend_def = {sid for sid in dev_sids
                if not d_end.by_sample[sid].degenerate_flag
                and d_end.by_sample[sid].endpoint_stop_label >= 0}
    keep = basin_def & dend_def
    tbl_basin_isect = restrict_to(tbl_basin_full, keep)
    tbl_dend_isect = restrict_to(d_end, keep)
    mask_basin = np.array([tbl_basin_isect.by_sample[sid].endpoint_stop_label
                           for sid in meta["sample_ids"]]) >= 0
    mask_dend = np.array([tbl_dend_isect.by_sample[sid].endpoint_stop_label
                          for sid in meta["sample_ids"]]) >= 0
    if not np.array_equal(mask_basin, mask_dend):
        raise AssertionError("intersection populations diverge between "
                             "the two label schemes")
    pos_basin_isect = sum(1 for sid in keep
                          if tbl_basin_isect.by_sample[sid].endpoint_stop_label == 1)
    pos_dend_isect = sum(1 for sid in keep
                         if tbl_dend_isect.by_sample[sid].endpoint_stop_label == 1)
    agree = sum(1 for sid in keep
                if (tbl_basin_isect.by_sample[sid].endpoint_stop_label
                    == tbl_dend_isect.by_sample[sid].endpoint_stop_label))

    # ---- probe runs --------------------------------------------------------
    t0_probe = time.time()
    mlp_state: dict[str, float] = {"spent": 0.0}
    run_specs: list[tuple[str, Any, dict, tuple, bool]] = [
        ("P5_basin_full", tbl_basin_full, {"P5": x_p5}, ("P5",), True),
        ("P0_dend_full", d_end, {"P0": x_p0}, ("P0",), False),
        ("P5_basin_isect", tbl_basin_isect, {"P5": x_p5}, ("P5",), False),
        ("P0_dend_isect", tbl_dend_isect, {"P0": x_p0}, ("P0",), False),
        ("P4_dend_isect", tbl_dend_isect, {"P4": x_p4}, ("P4",), False),
        ("P1_basin_isect", tbl_basin_isect, {"P1": x_p1}, ("P1",), False),
    ]
    results: dict[str, dict] = {}
    for name, table, rblocks, names, with_enc in run_specs:
        units_r, pooled_r = d2.run_probe_units(
            rblocks, table, meta, z_by_unit, h_by_unit, mlp_state,
            probe_names=names, with_encoder=with_enc)
        results[name] = {"units": units_r, "pooled": pooled_r,
                         "aggregate": d2.aggregate_units(units_r)}

    # merge the FULL-population P5 probe into the phase-1 tables (§3)
    r5_full = results["P5_basin_full"]
    for u5 in r5_full["units"]:
        for target_u in pack["probe_results"]["units"]:
            if (target_u["fold"], target_u["seed"]) == (u5["fold"], u5["seed"]):
                for key, m in u5["probes"].items():
                    if key.startswith("P5/"):
                        target_u["probes"][key] = m
    for key, agg in r5_full["aggregate"].items():
        if key.startswith("P5/"):
            pack["probe_results"]["aggregate"][key] = agg
    enc_aggregate = {k: v for k, v in r5_full["aggregate"].items()
                     if k.startswith("enc/")}
    pack["bootstrap"]["per_probe_pooled"]["P5"] = d1.curve_cluster_bootstrap_auprc(
        *r5_full["pooled"]["P5"], n_resamples=n_bootstrap)
    for cid, row in d2.per_curve_auprc_table(
            {"P5": r5_full["pooled"]["P5"]}).items():
        pack["per_curve_auprc"].setdefault(cid, {})["P5"] = row.get("P5")

    # ---- comparisons -------------------------------------------------------
    s5i, y5i, ci5 = results["P5_basin_isect"]["pooled"]["P5"]
    s0i, y0i, ci0 = results["P0_dend_isect"]["pooled"]["P0"]
    s4i, y4i, ci4 = results["P4_dend_isect"]["pooled"]["P4"]
    s1i, y1i, ci1 = results["P1_basin_isect"]["pooled"]["P1"]
    for other in (ci0, ci4, ci1):
        if not (np.array_equal(ci5, other) and ci5.size == s5i.size):
            raise AssertionError("intersection pooled rows are not aligned "
                                 "across configs")
    comparisons = pack["bootstrap"]["comparisons"]
    deltas = pack["paired_deltas"]

    primary_boot = d2.paired_cluster_bootstrap_auprc(
        s5i, s0i, y5i, ci5, labels_b=y0i, n_resamples=n_bootstrap)
    primary_boot["role"] = "primary"
    primary_boot["probes"] = ["P5", "P0"]
    primary_boot["population"] = "intersection"
    comparisons["P5-P0"] = primary_boot
    deltas["P5-P0"] = unit_deltas(results["P5_basin_isect"]["units"], "P5/hgb",
                                  results["P0_dend_isect"]["units"], "P0/hgb")

    c_p5p4 = d2.paired_cluster_bootstrap_auprc(
        s5i, s4i, y5i, ci5, labels_b=y4i, n_resamples=n_bootstrap)
    c_p5p4.update(role="exploratory", probes=["P5", "P4"],
                  population="intersection", same_x_different_labels=True)
    comparisons["P5-P4"] = c_p5p4
    deltas["P5-P4"] = unit_deltas(results["P5_basin_isect"]["units"], "P5/hgb",
                                  results["P4_dend_isect"]["units"], "P4/hgb")

    c_p5p1 = d2.paired_cluster_bootstrap_auprc(
        s5i, s1i, y5i, ci5, n_resamples=n_bootstrap)
    c_p5p1.update(role="exploratory", probes=["P5", "P1"],
                  population="intersection", same_labels=True)
    comparisons["P5-P1"] = c_p5p1
    deltas["P5-P1"] = unit_deltas(results["P5_basin_isect"]["units"], "P5/hgb",
                                  results["P1_basin_isect"]["units"], "P1/hgb")

    s5f, y5f, cf5 = r5_full["pooled"]["P5"]
    s0f, y0f, cf0 = results["P0_dend_full"]["pooled"]["P0"]
    c_full = paired_curve_bootstrap_two_pops(
        s5f, y5f, cf5, s0f, y0f, cf0, n_resamples=n_bootstrap)
    c_full.update(role="exploratory", probes=["P5", "P0"],
                  population="full (two populations — descriptive)")
    comparisons["P5-P0-full"] = c_full
    deltas["P5-P0-full"] = unit_deltas(
        r5_full["units"], "P5/hgb",
        results["P0_dend_full"]["units"], "P0/hgb")

    # P0 re-run consistency vs phase 1 (same data/labels/protocol)
    p0_rerun = {"status": "SKIPPED", "max_abs_diff": None}
    a1 = pack["probe_results"]["aggregate"].get("P0/hgb", {})
    a2 = results["P0_dend_full"]["aggregate"].get("P0/hgb", {})
    if a1 and a2 and not a2.get("skipped"):
        diffs = []
        for field_name in ("auprc", "auroc"):
            for stat in ("median", "min", "max"):
                v1 = a1[field_name][stat]
                v2 = a2[field_name][stat]
                if v1 is not None and v2 is not None:
                    diffs.append(abs(float(v1) - float(v2)))
        p0_rerun = {"status": "PASS" if diffs and max(diffs) <= 1e-9 else "FAIL",
                    "max_abs_diff": d1._r6(max(diffs) if diffs else None),
                    "note": "P0 re-run on the full d_end population must "
                            "reproduce phase 1 bit-for-bit"}

    t_probe = time.time() - t0_probe

    # ---- gate merge --------------------------------------------------------
    t0 = time.time()
    gate = pack["gate"]
    gate["checks"] = [c for c in gate["checks"]
                      if c["id"] != "mme_p5_minus_p0"]
    for c in validity_checks:
        gate["checks"].append({"id": f"p5_validity_{c['id']}",
                               "expect": c["expect"], "actual": c["actual"],
                               "status": "PASS" if c["pass"] else "FAIL",
                               "pass": bool(c["pass"])})
    med_p5 = deltas["P5-P0"].get("median")
    lb_p5 = primary_boot["ci95"].get("lower")
    mme_ok = (med_p5 is not None and lb_p5 is not None
              and float(med_p5) >= MME_P5 and float(lb_p5) > 0.0)
    mme_status = "PASS" if mme_ok else "PENDING-GATE"
    gate["checks"].append({
        "id": "mme_p5_minus_p0",
        "expect": f"median paired ΔAUPRC ≥ +{MME_P5:.2f} AND paired "
                  "curve-cluster bootstrap 95% CI lower bound > 0 "
                  "(intersection population, primary HGB)",
        "actual": f"median_delta={med_p5} ci95_lb={lb_p5} "
                  f"(pooled Δ={primary_boot['point']}, n_intersection="
                  f"{len(keep)})",
        "status": mme_status, "pass": bool(mme_ok)})
    integrity_ids = {"p0_reproduces_d1", "p2_self_check", "p3_leak_rule"} | {
        f"p5_validity_{c['id']}" for c in validity_checks}
    hard_fail = any(c["status"] in ("FAIL", "LEAK") for c in gate["checks"]
                    if c["id"] in integrity_ids)
    gate["verdict"] = "FAIL" if hard_fail else "PASS"
    gate["training_authorized"] = False
    ms = gate.setdefault("mme_summary", {})
    ms["p5_p0"] = med_p5
    ms["p5_p0_ci95"] = [lb_p5, primary_boot["ci95"].get("upper")]
    ms["p5_status"] = mme_status
    ms["p5_population"] = "intersection"
    t_gate = time.time() - t0

    # ---- pack assembly -----------------------------------------------------
    call_summary: dict[str, dict] = {}
    for e in budget.log:
        s = call_summary.setdefault(str(e["purpose"]),
                                    {"n": 0, "wall_s": 0.0, "note": ""})
        s["n"] += 1
        s["wall_s"] = round(s["wall_s"] + float(e["seconds"]), 3)
    for s in call_summary.values():
        s["wall_s"] = round(s["wall_s"], 3)
    if call_summary.get("conformer_opt"):
        accepted_n = sum(1 for e in budget.log
                         if e["purpose"] == "conformer_opt" and e.get("accepted"))
        drifts = [e["k_drift_max"] for e in budget.log
                  if e["purpose"] == "conformer_opt"
                  and e.get("k_drift_max") is not None]
        call_summary["conformer_opt"]["note"] = (
            f"accepted {accepted_n} (primary "
            f"{sum(len(v) for v in accepted_primary.values())} / flip "
            f"{sum(len(v) for v in accepted_flip.values())}), max K drift "
            f"{max(drifts) if drifts else None} Å")

    conclusion_notes = [
        f"主比较 P5−P0（交集 n={len(keep)}）: median Δ = {med_p5}, "
        f"pooled Δ = {primary_boot['point']}, 95% CI "
        f"[{primary_boot['ci95'].get('lower')}, "
        f"{primary_boot['ci95'].get('upper')}] → MME **{mme_status}**",
        f"探索: P5−P1（同标签、297 vs 114 维）median Δ = "
        f"{deltas['P5-P1'].get('median')}，CI "
        f"[{c_p5p1['ci95'].get('lower')}, {c_p5p1['ci95'].get('upper')}]；"
        f"P5−P4（同 X、标签差异）median Δ = {deltas['P5-P4'].get('median')}，"
        f"CI [{c_p5p4['ci95'].get('lower')}, {c_p5p4['ci95'].get('upper')}]",
        f"全人群变体（描述性，人群不同）: P5−P0-full median Δ = "
        f"{deltas['P5-P0-full'].get('median')}，pooled Δ = "
        f"{c_full['point']}，CI [{c_full['ci95'].get('lower')}, "
        f"{c_full['ci95'].get('upper')}]",
        f"有效性 {n_pass}/6" + ("" if n_pass == 6 else
                                f"（失败: {[c['id'] for c in validity_checks if not c['pass']]}）")
        + "；P1/P4 的 phase-1 MME 结果（见 §4）不受本阶段影响。",
    ]

    phase2 = {
        "schema": PHASE2_SCHEMA,
        "p5_spec_sha256": p5_sha,
        "prerequisite": p5_spec["prerequisite"],
        "enrichment": {
            "subset_rule": "first 10 dev curves by sorted curve_id; "
                           "forward-intent (target B) basins",
            "subset_curves": subset_curves,
            "seeds": {"primary": CONFORMER_SEED_PRIMARY,
                      "flip": CONFORMER_SEED_FLIP},
            "caps": {"max_calls": MAX_XTB_CALLS, "max_wall_s": MAX_XTB_WALL_S,
                     "energy_window_hartree": ENERGY_WINDOW_HARTREE,
                     "max_accepted": MAX_CONFORMERS_ACCEPTED,
                     "max_attempts": MAX_GENERATION_ATTEMPTS,
                     "max_xtb_per_curve_per_run": MAX_XTB_PER_CURVE},
            "n_calls": budget.calls, "wall_s": round(budget.wall, 3),
            "aborted": bool(enrichment_abort),
            "aborted_reason": enrichment_abort,
            "flip_abort_reason": flip_abort,
            "n_pre_flight": 1 if pre.get("log_entry") or pre.get("ok") else 0,
            "n_ref_sp": sum(1 for e in budget.log
                            if e["purpose"] == "reference_sp"),
            "n_conformer_primary": sum(
                1 for e in budget.log
                if e["purpose"] == "conformer_opt"
                and e.get("seed") == CONFORMER_SEED_PRIMARY),
            "n_conformer_flip": sum(
                1 for e in budget.log
                if e["purpose"] == "conformer_opt"
                and e.get("seed") == CONFORMER_SEED_FLIP),
            "accepted_primary": {c: len(v)
                                 for c, v in accepted_primary.items()},
            "accepted_flip": {c: len(v) for c, v in accepted_flip.items()},
            "ref_sp_energy_ha": {c: round(e, 9)
                                 for c, e in sorted(sp_energies.items())},
            "call_summary": call_summary,
            "per_call_log": budget.log,
            "xtb_binary": xtb_bin,
            "subprocess_sanction": p5_spec["subprocess_note"],
        },
        "labels": {
            "population": population,
            "t_rows": {**t_rows_check, "rows": t_rows_check["rows"][:30],
                       "audit_sha256": hashes["d01_audit_sha256"],
                       "all_edit_incomplete": all(
                           not r["edit_complete"] for r in t_rows_check["rows"])},
            "flip": flip,
            "label_table_summary": {
                "full": {"positive": labels_basin.n_positive,
                         "negative": labels_basin.n_negative,
                         "mask": labels_basin.n_mask},
            },
        },
        "validity": {"checks": validity_checks,
                     "summary": f"{n_pass}/6",
                     "notes": validity_notes},
        "probe": {
            "configs": {
                "P5_basin_full": {"label_scheme": "basin", "probe": "P5",
                                  "n_features": 297, "population": "full"},
                "P0_dend_full": {"label_scheme": "d_end", "probe": "P0",
                                 "n_features": 30, "population": "full"},
                "P5_basin_isect": {"label_scheme": "basin", "probe": "P5",
                                   "n_features": 297,
                                   "population": "intersection"},
                "P0_dend_isect": {"label_scheme": "d_end", "probe": "P0",
                                  "n_features": 30,
                                  "population": "intersection"},
                "P4_dend_isect": {"label_scheme": "d_end", "probe": "P4",
                                  "n_features": 297,
                                  "population": "intersection"},
                "P1_basin_isect": {"label_scheme": "basin", "probe": "P1",
                                   "n_features": 114,
                                   "population": "intersection"},
            },
            "enc_aggregate": enc_aggregate,
            "intersection": {
                "n_rows": len(keep),
                "n_basin_definitive": len(basin_def),
                "n_dend_definitive": len(dend_def),
                "pos_basin": pos_basin_isect, "pos_dend": pos_dend_isect,
                "agree": agree, "disagree": len(keep) - agree,
                "config_aggregates": {
                    name: results[name]["aggregate"]
                    for name in ("P5_basin_isect", "P0_dend_isect",
                                 "P4_dend_isect", "P1_basin_isect")},
                "config_units": {
                    name: results[name]["units"]
                    for name in ("P5_basin_isect", "P0_dend_isect",
                                 "P4_dend_isect", "P1_basin_isect")},
            },
        },
        "p0_rerun_check": p0_rerun,
        "conclusion_notes": conclusion_notes,
        "provenance": {
            "p5_spec_sha256": p5_sha,
            "d01_audit_sha256": hashes["d01_audit_sha256"],
            "graph_symmetry_sha256": hashes["graph_symmetry_sha256"],
            "basin_labels_v1_sha256": hashes["basin_labels_v1_sha256"],
            "d2_p5_probe_script_sha256": hashes["d2_p5_probe_script_sha256"],
        },
        "runtime_s": {},
    }
    pack["phase2"] = phase2
    pack["feature_disclosure"] = feat_info
    pack["phase2"]["runtime_s"] = {
        "spec_write": d1._r6(t_spec),
        "labels_and_enrichment": d1._r6(t_labels),
        "features": d1._r6(t_feat),
        "probing": d1._r6(t_probe),
        "stats_and_gate": d1._r6(t_gate),
        "total": d1._r6(time.time() - t_start),
    }

    pack_path.write_text(json.dumps(pack, indent=2, sort_keys=True) + "\n",
                         encoding="utf-8")
    (out_dir / "D2_PROBE.md").write_text(d2.render_markdown(pack),
                                         encoding="utf-8")
    return pack


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tables-dir", type=Path, default=DEFAULT_TABLES_DIR)
    ap.add_argument("--embeddings-dir", type=Path,
                    default=DEFAULT_EMBEDDINGS_DIR)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--bootstrap", type=int, default=d1.BOOTSTRAP_RESAMPLES)
    ap.add_argument("--d01-audit", type=Path, default=DEFAULT_D01_AUDIT)
    ap.add_argument("--d1-pack", type=Path, default=DEFAULT_D1_PACK)
    ap.add_argument("--xtb-bin", default=XTB_BIN)
    ap.add_argument("--strict", action="store_true",
                    help="exit 1 when the merged gate verdict is FAIL "
                         "(report-only: never authorizes training)")
    args = ap.parse_args(argv)
    pack = run(args.tables_dir, args.out, args.embeddings_dir,
               n_bootstrap=args.bootstrap, d01_audit_path=args.d01_audit,
               d1_pack_path=args.d1_pack, xtb_bin=args.xtb_bin)
    p2 = pack["phase2"]
    gate = pack["gate"]
    print(f"[{gate['gate_id']}] merged verdict: {gate['verdict']} "
          f"(training_authorized={gate['training_authorized']})")
    for c in gate["checks"]:
        if c["id"].startswith("p5_") or c["id"] == "mme_p5_minus_p0":
            print(f"  {c['status']:<12} {c['id']}: {c['actual']}")
    print(f"validity: {p2['validity']['summary']}")
    enr = p2["enrichment"]
    print(f"xtb: {enr['n_calls']} calls / {enr['wall_s']} s "
          f"(cap {enr['caps']['max_calls']} / {enr['caps']['max_wall_s']}) "
          f"aborted={enr['aborted']} {enr['aborted_reason'] or ''}")
    print(f"p5 population (dev usable): "
          f"{p2['labels']['population']['dev_usable']}"[:220])
    print(f"intersection n={p2['probe']['intersection']['n_rows']} "
          f"(pos basin {p2['probe']['intersection']['pos_basin']} / d_end "
          f"{p2['probe']['intersection']['pos_dend']})")
    d = pack["paired_deltas"]["P5-P0"]
    b = pack["bootstrap"]["comparisons"]["P5-P0"]
    print(f"P5-P0 (intersection): median Δ {d['median']}, pooled Δ "
          f"{b['point']}, CI [{b['ci95']['lower']}, {b['ci95']['upper']}] → "
          f"{[c['status'] for c in gate['checks'] if c['id'] == 'mme_p5_minus_p0'][0]}")
    print(f"p5_spec sha256: {p2['p5_spec_sha256']}")
    print(f"runtime total: {p2['runtime_s']['total']} s")
    print(f"evidence: {args.out / 'evidence_pack.json'}")
    print(f"report:   {args.out / 'D2_PROBE.md'}")
    if args.strict and gate["verdict"] != "PASS":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
