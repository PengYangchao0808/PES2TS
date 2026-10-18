"""ACP-backed unconstrained in-place E/g observation (WP1, plan v3 §9 row 3).

Wraps the existing ``ORCAGradientBackend`` (ACP ``OrcaGradient`` workflow,
GFN2-xTB method) as a direct-MDP ``ObservationSource``. No new execution
path is created (R3): every evaluation still goes through ACP with its
content-addressed receipts; identical geometry replays the cached receipt
and a binding conflict is a hard error, never a recompute.

Failure mapping (first version; the fit50 review refines the taxonomy):
timeouts / transport faults -> ``failed_infra`` (censoring), deterministic
QC failures -> ``failed_geometry`` (terminal). The raw ``failure_class`` is
always carried in the evidence for later re-classification.
"""
from __future__ import annotations

from pathlib import Path
from typing import Final, Mapping

import numpy as np

from pes2ts_core.generation.planning.direct_mdp import DirectObservation
from pes2ts_core.integration.acp.gradient_backend import ORCAGradientBackend
from pes2ts_core.utils.hashing import sha256_bytes

import re

_PORTABLE_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")

_DIRECT_OBSERVATION_VERSION: Final[str] = "pes2ts_direct_observation_v1"

_Z_TO_SYMBOL: Final[dict[int, str]] = {
    1: "H", 2: "He", 3: "Li", 4: "Be", 5: "B", 6: "C", 7: "N", 8: "O", 9: "F",
    10: "Ne", 11: "Na", 12: "Mg", 13: "Al", 14: "Si", 15: "P", 16: "S", 17: "Cl",
    18: "Ar", 19: "K", 20: "Ca", 26: "Fe", 27: "Co", 28: "Ni", 29: "Cu", 30: "Zn",
    34: "Se", 35: "Br", 44: "Ru", 45: "Rh", 46: "Pd", 47: "Ag", 48: "Cd", 53: "I",
    78: "Pt", 79: "Au",
}

#: failure classes that censor (infrastructure) instead of terminating the
#: MDP; the vocabulary mirrors ``_classify_acp_gradient_failure`` plus ACP
#: transport errors surfaced as PROCESS_FAILED_WITHOUT_NORMAL_TERMINATION
#: only when the ACP receipt itself never produced a manifest (fit50 review
#: refines which of those are truly geometry-related).
INFRA_FAILURE_MARKERS: Final[frozenset[str]] = frozenset({
    "TIMEOUT",
    "ACP_TIMEOUT", "ACP_LAUNCH_FAILED", "ACP_TRANSPORT", "ACP_INTERRUPTED",
})


class DirectObservationError(RuntimeError):
    """Typed misuse of the direct observation backend."""


def symbols_for(atomic_numbers) -> list[str]:
    symbols = []
    for z in np.asarray(atomic_numbers).astype(int):
        symbol = _Z_TO_SYMBOL.get(int(z))
        if symbol is None:
            raise DirectObservationError(f"no symbol registered for Z={z}")
        symbols.append(symbol)
    return symbols


class DirectGradientObservation:
    """One-reaction E/g source through ACP (content-addressed receipts)."""

    version = _DIRECT_OBSERVATION_VERSION

    def __init__(
        self,
        *,
        atomic_numbers,
        receipt_folder: str | Path,
        trial_prefix: str,
        charge: int = 0,
        multiplicity: int = 1,
        method: str = "GFN2-xTB",
        timeout_seconds: float = 120.0,
        nproc: int = 2,
        acp_root=None,
        acp_python=None,
        acp_config=None,
        acp_wiring: Mapping | None = None,
    ) -> None:
        if not _PORTABLE_ID_RE.fullmatch(trial_prefix):
            raise DirectObservationError(
                f"trial_prefix {trial_prefix!r} is not a portable evaluation id part"
            )
        self._trial_prefix = trial_prefix
        self._backend = ORCAGradientBackend(
            charge=charge, multiplicity=multiplicity,
            elements=symbols_for(atomic_numbers),
            folder=Path(receipt_folder), method=method,
            timeout_seconds=timeout_seconds, nproc=nproc,
            acp_root=acp_root, acp_python=acp_python, acp_config=acp_config,
            acp_wiring=dict(acp_wiring) if acp_wiring else None,
        )

    def _evaluation_id(self, coordinates: np.ndarray, evaluation_id: str) -> str:
        """``<trial>/<content>`` receipt id; content binds geometry + node id."""
        payload = str(np.asarray(coordinates, dtype=float).round(8).tolist())
        content = sha256_bytes(
            f"{payload}|{evaluation_id}".encode("utf-8"))[:16]
        return f"{self._trial_prefix}/eval-{content}"

    @property
    def backend(self) -> ORCAGradientBackend:
        return self._backend

    def __call__(
        self,
        coordinates: np.ndarray,
        atomic_numbers: np.ndarray,
        *,
        charge: int,
        multiplicity: int,
        evaluation_id: str,
    ) -> DirectObservation:
        coords = np.asarray(coordinates, dtype=float)
        record = self._backend(coords, self._evaluation_id(coords, evaluation_id))
        if record.get("success"):
            gradient = record.get("gradient_hartree_per_angstrom")
            return DirectObservation(
                status="completed",
                energy_hartree=float(record["energy"]),
                gradient_hartree_per_angstrom=(
                    None if gradient is None else np.asarray(gradient, dtype=float)
                ),
                cache_reused=bool(record.get("acp", {}).get("reused", False)),
                request_sha256=str(record.get("request_sha256")),
                wall_seconds=float(record.get("duration_seconds", 0.0) or 0.0),
            )
        failure = str(record.get("failure_class") or "UNKNOWN")
        status = "failed_infra" if failure in INFRA_FAILURE_MARKERS else "failed_geometry"
        return DirectObservation(
            status=status,
            cache_reused=False,
            request_sha256=str(record.get("request_sha256")),
            wall_seconds=float(record.get("duration_seconds", 0.0) or 0.0),
            error=failure,
        )


__all__ = [
    "DirectGradientObservation",
    "DirectObservationError",
    "INFRA_FAILURE_MARKERS",
    "symbols_for",
]
