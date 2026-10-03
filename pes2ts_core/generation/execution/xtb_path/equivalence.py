"""ADR-0002 X3'-A: the two-part equivalence certificate (pure derivation).

The retired guarantee was a byte-identical local xTB wrap; ADR-0002 X2'-C
deleted that runner and the xTB PATH step now executes through ACP, whose
scheduling and the xTB metadynamics raw output are not byte-reproducible.
The replacement certificate has two independent, deterministic halves:

* **RecipeEquivalence** — the PES2TS-authored ``$path`` recipe rendered from
  a ``path_config`` mapping is byte-stable against a frozen expected block
  (and against a frozen recipe-payload digest).  Rendered by the same
  :func:`pes2ts_core.integration.acp.xtb_path_request.build_path_inp_text`
  the ACP request carries.
* **ReplayParity** — given a FROZEN raw multi-frame trajectory, the derived
  artifacts are byte-stable: the certificate re-runs the EXACT pipeline
  derivation (:func:`xtb_output.parse_path_xyz` ->
  :func:`inspect.frame_metrics` / :func:`inspect.evaluate_validity`) and
  digests the deterministic serialization of the resulting metric rows and
  validity verdict.

This module is pure derivation + comparison.  It never imports the ACP
transport, never spawns a process, and never raises on a digest/text
mismatch — a failed comparison returns ``passed=False`` with a
machine-readable ``mismatch`` string.  ``build_replay_parity_certificate``
additionally never raises when the frozen trajectory cannot be parsed: the
derivation failure is reported as ``passed=False``.

``charge``/``uhf`` are part of the replay recipe context (the ACP request
carried them); the frame-metric/validity derivation is spin-blind, so they
do not enter the digests — they are accepted to keep the certificate call
shape aligned with the request recipe context.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, Final

from pes2ts_core.generation.execution.xtb_path.inspect import (
    Metric,
    Verdict,
    evaluate_validity,
    frame_metrics,
)
from pes2ts_core.generation.execution.xtb_path.xtb_output import (
    Frame,
    XtbOutputError,
    parse_path_xyz,
    parse_start_xyz,
)
from pes2ts_core.integration.acp.xtb_path_request import build_path_inp_text
from pes2ts_core.utils.hashing import sha256_bytes, stable_json_dumps
from pes2ts_core.utils.jsonio import write_json

#: Conventional filename of the frozen certificate artifact.
EQUIVALENCE_REPORT_FILENAME: Final[str] = "equivalence_report.json"
#: Schema identifier stamped into every ``equivalence_report.json``.
SCHEMA_EQUIVALENCE: Final[str] = "pes2ts_equivalence_report_v1"
#: Default ``reaction_id`` stamped into replay rows/verdict when the caller
#: supplies none (the digest then covers this literal; pass a real id to bind
#: a specific reaction).
DEFAULT_REPLAY_REACTION_ID: Final[str] = "replay"
#: Sibling product-endpoint filename probed beside the frozen trajectory when
#: neither ``product_xyz_path`` nor ``product_coords`` is given (the pinned
#: fixture and manual-run layout keep ``start.xyz``/``end.xyz`` beside
#: ``xtbpath.xyz``).
_PRODUCT_FALLBACK_FILENAME: Final[str] = "end.xyz"


def _failed_recipe(detail: str, *, config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "passed": False,
        "path_inp_sha256": None,
        "expected_path_inp_sha256": None,
        "mismatch": detail,
        "config": dict(config),
    }


def build_recipe_equivalence_certificate(
    path_config: Mapping[str, Any],
    *,
    expected_path_inp: str | None = None,
    expected_request_sha256: str | None = None,
) -> dict[str, Any]:
    """Certificate that the PES2TS-authored ``$path`` recipe is byte-stable.

    Regenerates the ``$path`` block via
    :func:`pes2ts_core.integration.acp.xtb_path_request.build_path_inp_text`
    (the same renderer the ACP request carries), compares it byte-for-byte
    with *expected_path_inp* when provided, and digests the regenerated
    text.

    ``expected_request_sha256`` anchors the recipe-payload digest: the sha256
    of the regenerated ``$path`` text (the payload ``recipe.path_inp_text``
    inside a ``pes2ts_xtb_path_request_v1`` request).  The full request
    digest additionally covers reaction/source context and is verified by
    the request builder's own tests; this certificate locks the recipe
    payload itself, which is the only thing ``path_config`` determines.

    Never raises on a mismatch (or a non-mapping ``path_config``): returns
    ``passed=False`` with a machine-readable ``mismatch``.
    """
    try:
        path_inp = build_path_inp_text(path_config)
    except ValueError as error:
        return _failed_recipe(f"path_config rejected: {error}", config=path_config)
    path_inp_sha256 = sha256_bytes(path_inp.encode("utf-8"))
    expected_path_inp_sha256 = (
        None if expected_path_inp is None
        else sha256_bytes(expected_path_inp.encode("utf-8"))
    )
    mismatch: str | None = None
    if expected_path_inp is not None and path_inp != expected_path_inp:
        mismatch = (
            "path_inp byte mismatch: regenerated sha256="
            f"{path_inp_sha256} expected sha256={expected_path_inp_sha256}"
        )
    elif expected_request_sha256 is not None and path_inp_sha256 != expected_request_sha256:
        mismatch = (
            "request_sha256 mismatch: recipe payload sha256="
            f"{path_inp_sha256} expected request_sha256={expected_request_sha256}"
        )
    return {
        "passed": mismatch is None,
        "path_inp_sha256": path_inp_sha256,
        "expected_path_inp_sha256": expected_path_inp_sha256,
        "mismatch": mismatch,
        "config": dict(path_config),
    }


def _keyed(frame: Frame) -> dict[int, tuple[float, float, float]]:
    """Map-keyed endpoint coords: atom ``i`` is map ``i + 1`` (map order)."""
    return {index + 1: xyz for index, xyz in enumerate(frame.coordinates)}


def _elements_of(frame: Frame) -> dict[int, str]:
    return {index + 1: element for index, element in enumerate(frame.elements)}


def _load_product_context(
    *,
    product_xyz_path: str | Path | None,
    product_coords: Mapping[int, Sequence[float]] | None,
    frozen_trajectory_path: str | Path,
) -> Mapping[int, Sequence[float]]:
    """Return the product-side endpoint mapping for the replay derivation."""
    if product_coords is not None:
        return product_coords
    if product_xyz_path is not None:
        return _keyed(parse_start_xyz(Path(product_xyz_path)))
    sibling = Path(frozen_trajectory_path).with_name(_PRODUCT_FALLBACK_FILENAME)
    if sibling.is_file():
        return _keyed(parse_start_xyz(sibling))
    raise ValueError(
        "product endpoint context missing: pass product_xyz_path or "
        f"product_coords (no {_PRODUCT_FALLBACK_FILENAME!r} beside "
        f"{frozen_trajectory_path})"
    )


def _replay_document(
    *, reaction_id: str, rows: Sequence[Metric], verdict: Verdict
) -> dict[str, Any]:
    """Deterministic verdict projection serialized as the replay document."""
    return {
        "reaction_id": str(reaction_id),
        "status": str(verdict.status),
        "failure_code": None if verdict.failure_code is None else verdict.failure_code.value,
        "failure_detail": verdict.detail,
        "n_frames": len(rows),
        "validity": dict(verdict.summary),
    }


def build_replay_parity_certificate(
    *,
    frozen_trajectory_path: str | Path,
    start_xyz_path: str | Path,
    charge: int,
    uhf: int,
    r_pairs: Iterable[Sequence[int]],
    p_pairs: Iterable[Sequence[int]],
    events: Mapping[str, Sequence[Mapping[str, Any]]],
    config: Mapping[str, Any] | None = None,
    expected_frames_sha256: str | None = None,
    expected_document_sha256: str | None = None,
    reaction_id: str = DEFAULT_REPLAY_REACTION_ID,
    product_xyz_path: str | Path | None = None,
    reactant_coords: Mapping[int, Sequence[float]] | None = None,
    product_coords: Mapping[int, Sequence[float]] | None = None,
    elements: Mapping[int, str] | None = None,
) -> dict[str, Any]:
    """Certificate that a frozen raw trajectory derives byte-stable artifacts.

    Parses *frozen_trajectory_path* with the EXISTING
    :func:`xtb_output.parse_path_xyz` (``start_path=start_xyz_path``), then
    computes :func:`inspect.frame_metrics` and :func:`inspect.evaluate_validity`
    exactly as the G2 pipeline does (R\\u2192P frame order, ``xtb_failure=None``,
    ``energy_rel_kcal_raw`` mirrored from the parsed relative energies), and
    digests the deterministic serialization (``stable_json_dumps``) of the
    metric rows (``frames_sha256``) and the verdict projection
    (``document_sha256``).  The digests are compared with the expected values
    when provided.

    Endpoint context defaults (pipeline parity): ``elements`` and
    ``reactant_coords`` derive from *start_xyz_path* (map ``i+1`` = atom
    ``i``); ``product_coords`` derives from *product_xyz_path*, else a
    sibling ``end.xyz`` beside the trajectory, else an explicit mapping is
    required.  ``charge``/``uhf`` are accepted as request-recipe context and
    do not enter the spin-blind derivation digests.

    Never raises on a digest mismatch or an unparseable trajectory: returns
    ``passed=False`` with a machine-readable ``mismatch``.
    """
    del charge, uhf  # request-recipe context; derivation is spin-blind.
    config_block: Mapping[str, Any] = config if isinstance(config, Mapping) else {}
    try:
        frames = parse_path_xyz(Path(frozen_trajectory_path), start_path=Path(start_xyz_path))
        start_frame = parse_start_xyz(Path(start_xyz_path))
        derived_elements = elements if elements is not None else _elements_of(start_frame)
        derived_reactant = (
            reactant_coords if reactant_coords is not None else _keyed(start_frame)
        )
        derived_product = _load_product_context(
            product_xyz_path=product_xyz_path,
            product_coords=product_coords,
            frozen_trajectory_path=frozen_trajectory_path,
        )
        rows = frame_metrics(
            frames,
            reaction_id=str(reaction_id),
            reactant_coords=derived_reactant,
            product_coords=derived_product,
            elements=dict(derived_elements),
            r_pairs=r_pairs,
            p_pairs=p_pairs,
            events=events,
            config=config_block,
        )
        for row in rows:
            energy = row.get("energy_rel_kcal")
            row["energy_rel_kcal_raw"] = None if energy is None else float(energy)
        verdict = evaluate_validity(
            frames,
            reaction_id=str(reaction_id),
            reactant_coords=derived_reactant,
            product_coords=derived_product,
            elements=dict(derived_elements),
            r_pairs=r_pairs,
            p_pairs=p_pairs,
            events=events,
            xtb_failure=None,
            config=config_block,
        )
    except (XtbOutputError, ValueError, OSError) as error:
        return {
            "passed": False,
            "n_frames": 0,
            "frames_sha256": None,
            "expected_frames_sha256": expected_frames_sha256,
            "document_sha256": None,
            "expected_document_sha256": expected_document_sha256,
            "mismatch": f"replay derivation failed: {error}",
        }
    frames_sha256 = sha256_bytes(stable_json_dumps(list(rows)).encode("utf-8"))
    document_sha256 = sha256_bytes(
        stable_json_dumps(_replay_document(reaction_id=str(reaction_id), rows=rows, verdict=verdict)).encode("utf-8")
    )
    mismatch: str | None = None
    if expected_frames_sha256 is not None and frames_sha256 != expected_frames_sha256:
        mismatch = (
            "frames digest mismatch: derived sha256="
            f"{frames_sha256} expected sha256={expected_frames_sha256}"
        )
    elif (
        expected_document_sha256 is not None
        and document_sha256 != expected_document_sha256
    ):
        mismatch = (
            "document digest mismatch: derived sha256="
            f"{document_sha256} expected sha256={expected_document_sha256}"
        )
    return {
        "passed": mismatch is None,
        "n_frames": len(frames),
        "frames_sha256": frames_sha256,
        "expected_frames_sha256": expected_frames_sha256,
        "document_sha256": document_sha256,
        "expected_document_sha256": expected_document_sha256,
        "mismatch": mismatch,
    }


def write_equivalence_report(
    path: str | Path, *, recipe: Mapping[str, Any], replay: Mapping[str, Any]
) -> None:
    """Atomically write the frozen ``equivalence_report.json`` certificate.

    *path* is the report file path (the conventional filename is
    :data:`EQUIVALENCE_REPORT_FILENAME`); when *path* is an existing
    directory, the report is written inside it under that filename.
    Serialization is delegated to :func:`pes2ts_core.utils.jsonio.write_json`
    (canonical JSON, atomic publish), so two writes over identical
    certificates are byte-identical.
    """
    target = Path(path)
    if target.is_dir():
        target = target / EQUIVALENCE_REPORT_FILENAME
    write_json(
        target,
        {
            "schema_version": SCHEMA_EQUIVALENCE,
            "recipe_equivalence": dict(recipe),
            "replay_parity": dict(replay),
        },
    )


__all__ = [
    "DEFAULT_REPLAY_REACTION_ID",
    "EQUIVALENCE_REPORT_FILENAME",
    "SCHEMA_EQUIVALENCE",
    "build_recipe_equivalence_certificate",
    "build_replay_parity_certificate",
    "write_equivalence_report",
]
