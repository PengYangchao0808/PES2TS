"""V5.4.1 embedding registry (WP2; plan §5.1 / ADR-0032; config
``embedding_contract.embedding_registry``).

The registry is the single addressing table for residual-embedding ids:

* ``E0`` — the legacy passthrough baseline: **no branch module exists**
  (plan §5.2: E0 数学上等价于 delta=0，不使用“极小非零 alpha”冒充 parity；
  config ``e0_has_no_branch: true``).  ``build_embedding("E0")`` returns
  ``None``.
* ``E1``–``E5`` — the five preregistered main candidates.  Their SPECS are
  frozen here (family / internal view / output kind, copied from the
  campaign yaml).  The real WP3 modules live under
  ``pes2ts_core/pesn/models/v541/embeddings/`` and are imported LAZILY via
  :data:`BUILTIN_EMBEDDING_MODULES` / :func:`ensure_embedding` — no
  placeholder math is ever registered here, and a landed module becomes
  buildable with no registry edit.  A module that has not landed yet (E5
  until ``dual_stream.py`` ships) is a typed
  ``V541_EMBEDDING_NOT_IMPLEMENTED``; an ImportError raised INSIDE an
  existing module is ``V541_EMBEDDING_IMPORT_FAILED`` and is never
  downgraded to "unimplemented".
* ``E6`` (``painn_scalar``) — diagnostic-only, requires separate approval
  (config ``embedding_registry.E6.diagnostic_only: true``); same typed
  refusal until then.

Test-only embeddings (dummy branches for WP2 parity/gradient/E(3) tests)
may be registered under any NEW id via :func:`register_embedding`; the
snapshot/restore helpers keep test registration from leaking across tests
or into a run.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Final, Mapping, Sequence

from pes2ts_core.pesn.models.v541.interfaces import (
    EMBEDDING_ID_E0,
    NodeDeltaEmbedding,
    V541EmbeddingNotImplementedError,
    V541EmbeddingUnknownError,
    V541InterfaceError,
)

#: Frozen candidate ids (campaign yaml registry rows).
FROZEN_EMBEDDING_IDS: Final[tuple[str, ...]] = ("E0", "E1", "E2", "E3", "E4", "E5", "E6")
#: Main-matrix candidate ids (in_main_matrix: true; E0 is the baseline).
MAIN_MATRIX_EMBEDDING_IDS: Final[tuple[str, ...]] = ("E0", "E1", "E2", "E3", "E4")
#: Reserved candidate ids — WP3 implementations register over these with
#: ``replace=True``; nothing else may squat them.
RESERVED_CANDIDATE_IDS: Final[tuple[str, ...]] = ("E1", "E2", "E3", "E4", "E5", "E6")


@dataclass(frozen=True)
class EmbeddingSpec:
    """Frozen registry row for one embedding id (config yaml mirror)."""

    embedding_id: str
    family: str
    internal_view: str
    output: str
    in_main_matrix: bool = False
    diagnostic_only: bool = False
    conditional_parent: tuple[str, ...] = ()
    spec: str = ""
    note: str = ""
    #: WP3 status: False until a real implementation is registered.
    implemented: bool = field(default=False)

    def __post_init__(self) -> None:
        if self.embedding_id not in FROZEN_EMBEDDING_IDS:
            raise V541InterfaceError(
                "V541_EMBEDDING_FROZEN_ROW",
                f"{self.embedding_id!r} is not a frozen registry id "
                f"{FROZEN_EMBEDDING_IDS}; ad-hoc test ids do not get specs")


#: Frozen spec table — mirrors ``config/campaigns/pes2ts_v541_embed_50k.yaml``
#: ``embedding_contract.embedding_registry`` (WP0-signed rows; do not edit
#: without a config re-signoff).
EMBEDDING_SPECS: Final[Mapping[str, EmbeddingSpec]] = {
    "E0": EmbeddingSpec(
        embedding_id="E0", family="legacy", internal_view="legacy_f2",
        output="legacy", in_main_matrix=True,
        implemented=True,  # baseline: no branch, implemented by V541Model itself
        note="element 16d + 21->48 linear; strict-load/runner parity baseline"),
    "E1": EmbeddingSpec(
        embedding_id="E1", family="gin_e_chemical",
        internal_view="chem_union_plus_edit", output="residual_node_delta",
        in_main_matrix=True,
        spec="1x GINE-chem 48ch; epsilon=0 fixed; MLP 48x48 ReLU; explicit "
             "self term; no graph self-loop"),
    "E2": EmbeddingSpec(
        embedding_id="E2", family="pna_chemical",
        internal_view="chem_union_plus_edit", output="residual_node_delta",
        in_main_matrix=True,
        spec="1x PNA 48ch; aggregators [mean,max,min,std]; degree scalers "
             "[identity,amplification,attenuation]; towers=1; population variance"),
    "E3": EmbeddingSpec(
        embedding_id="E3", family="schnet_spatial",
        internal_view="legacy_spatial_subset_recomputed_rbf12",
        output="residual_node_delta", in_main_matrix=True,
        spec="1x continuous-filter update 48ch; RBF12 recomputed from physical "
             "current distances; cosine envelope pi*r/4.5; F2 mask untouched"),
    "E4": EmbeddingSpec(
        embedding_id="E4", family="angle_mpnn",
        internal_view="prereg_angle_triplets", output="residual_node_delta",
        in_main_matrix=True,
        spec="1x Angle-MPNN 48ch; unordered neighbor pairs; cos basis "
             "[1,c,c^2,c^3]; tie policy reject_boundary_tie_shell; <=12 neighbors/center"),
    "E5": EmbeddingSpec(
        embedding_id="E5", family="dual_stream",
        internal_view="E1_and_E4_shared_inputs", output="residual_node_delta",
        conditional_parent=("E1", "E4"),
        spec="GINE24 + Angle24 -> concat -> fusion 48->48; conditional launch (§9.2)"),
    "E6": EmbeddingSpec(
        embedding_id="E6", family="painn_scalar", internal_view="legacy_spatial",
        output="residual_node_delta", diagnostic_only=True,
        spec="scalar-projected diagnostic only; requires separate E6 approval; "
             "not in core matrix"),
}

#: Registered builders for implemented candidates (WP3 fills this for
#: E1–E5/E6; tests fill it for TEST-only ids).
_BUILDERS: dict[str, Callable[[], NodeDeltaEmbedding]] = {}

#: BUILT-IN candidate modules shipped by WP3 (plan §6): ``id -> module
#: path``.  The registry imports these LAZILY but with plain STATIC
#: ``from <module> import <builder>`` statements inside
#: :func:`ensure_embedding` — the product boundary refuses dynamic import
#: machinery (``importlib`` / ``__import__``) in the runtime set, and a
#: static edge is exactly as auditable as any other models/ import.  E6 is
#: deliberately absent (diagnostic-only, separate approval required).
BUILTIN_EMBEDDING_MODULES: Final[Mapping[str, str]] = {
    "E1": "pes2ts_core.pesn.models.v541.embeddings.gin_e_chemical",
    "E2": "pes2ts_core.pesn.models.v541.embeddings.pna_chemical",
    "E3": "pes2ts_core.pesn.models.v541.embeddings.schnet_spatial",
    "E4": "pes2ts_core.pesn.models.v541.embeddings.angle_mpnn",
    "E5": "pes2ts_core.pesn.models.v541.embeddings.dual_stream",
}


def known_embedding_ids() -> tuple[str, ...]:
    """Every id the model layer will accept (E0–E6 + registered test ids)."""
    return tuple(sorted(set(FROZEN_EMBEDDING_IDS) | set(_BUILDERS)))


def embedding_spec(embedding_id: str) -> EmbeddingSpec:
    """Frozen spec row for a FROZEN id (unknown ids → typed refusal)."""
    spec = EMBEDDING_SPECS.get(str(embedding_id))
    if spec is None:
        raise V541EmbeddingUnknownError(
            "V541_EMBEDDING_UNKNOWN",
            f"unknown embedding id {embedding_id!r}; known {FROZEN_EMBEDDING_IDS}")
    return spec


def is_embedding_known(embedding_id: str) -> bool:
    return str(embedding_id) in known_embedding_ids()


def ensure_embedding(embedding_id: str, *, replace: bool = False) -> bool:
    """Lazily import + register the BUILT-IN module for ``embedding_id``.

    Returns ``True`` when the id resolves to a registered builder after the
    call, ``False`` when the id simply has no built-in module (E0 / E6 /
    test-only ids — the caller decides what that means).

    The imports are PLAIN STATIC ``from ... import ...`` statements (one
    branch per frozen candidate): the product boundary treats any
    ``importlib`` / ``__import__`` edge in the runtime set as a dynamic-
    execution risk, and a static edge is just as lazy — it only runs when
    this function is called.

    Fail-closed import discipline:

    * the CANDIDATE module not existing on disk is the only "not
      implemented yet" case — typed :class:`V541EmbeddingNotImplementedError`
      (today that can only be a candidate whose file has not landed; the
      moment the file appears this starts resolving with NO registry edit);
    * an ``ImportError`` raised from INSIDE an existing module (or a
      missing/misnamed factory) is a typed ``V541_EMBEDDING_IMPORT_FAILED``
      — it is NEVER converted into "not implemented", so a broken real
      implementation cannot masquerade as an unfinished one;
    * an already-registered builder (e.g. a WP3/test fixture installed with
      ``replace=True``) is left untouched unless ``replace=True`` is asked
      for explicitly.
    """
    embedding_id = str(embedding_id)
    target_module = BUILTIN_EMBEDDING_MODULES.get(embedding_id)
    if target_module is None:
        return False
    if embedding_id in _BUILDERS and not replace:
        return True
    try:
        # One static branch per frozen candidate (no dynamic import path).
        if embedding_id == "E1":
            from pes2ts_core.pesn.models.v541.embeddings.gin_e_chemical import (
                build_gin_e_chemical as _builder,
            )
        elif embedding_id == "E2":
            from pes2ts_core.pesn.models.v541.embeddings.pna_chemical import (
                build_pna_chemical as _builder,
            )
        elif embedding_id == "E3":
            from pes2ts_core.pesn.models.v541.embeddings.schnet_spatial import (
                build_schnet_spatial as _builder,
            )
        elif embedding_id == "E4":
            from pes2ts_core.pesn.models.v541.embeddings.angle_mpnn import (
                build_angle_mpnn as _builder,
            )
        elif embedding_id == "E5":
            from pes2ts_core.pesn.models.v541.embeddings.dual_stream import (
                build_dual_stream as _builder,
            )
        else:                                      # pragma: no cover
            return False
    except ModuleNotFoundError as exc:
        missing = str(getattr(exc, "name", "") or "")
        if missing == target_module or target_module.startswith(missing + "."):
            spec = EMBEDDING_SPECS.get(embedding_id)
            family = spec.family if spec is not None else "candidate"
            raise V541EmbeddingNotImplementedError(
                "V541_EMBEDDING_NOT_IMPLEMENTED",
                f"{embedding_id} ({family}) is preregistered but not "
                f"implemented yet (WP3): its module {target_module!r} has "
                f"not landed; refusing rather than substituting a placeholder "
                f"branch") from exc
        raise V541InterfaceError(
            "V541_EMBEDDING_IMPORT_FAILED",
            f"importing {target_module} for {embedding_id} failed on a "
            f"missing dependency {missing!r}") from exc
    except Exception as exc:  # noqa: BLE001 - ImportError/SyntaxError/... at
        # import time is still a BROKEN real module — typed, never downgraded
        # to "not implemented yet" (that would let a broken candidate pass as
        # an unfinished one).
        raise V541InterfaceError(
            "V541_EMBEDDING_IMPORT_FAILED",
            f"importing {target_module} for {embedding_id} raised "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    if not callable(_builder):
        raise V541InterfaceError(
            "V541_EMBEDDING_IMPORT_FAILED",
            f"{target_module} exposes no callable registry builder for "
            f"{embedding_id}")
    _BUILDERS[embedding_id] = _builder
    return True


def is_embedding_implemented(embedding_id: str) -> bool:
    """E0 by construction; candidates via registration or the BUILT-IN module.

    The answer is DYNAMIC: a landed WP3 module counts as implemented the
    moment it imports, without any registry edit (and without any fake
    builder being pre-installed to "make the check pass").  A broken real
    module propagates its typed import failure instead of reporting False.
    """
    embedding_id = str(embedding_id)
    if embedding_id == EMBEDDING_ID_E0:
        return True
    if embedding_id in _BUILDERS:
        return True
    if embedding_id in BUILTIN_EMBEDDING_MODULES:
        try:
            return ensure_embedding(embedding_id)
        except V541EmbeddingNotImplementedError:
            # the candidate MODULE has not landed yet — honest False, and
            # build_embedding() will give the typed refusal for this id.
            return False
        # any other typed failure (internal ImportError / bad factory) is
        # deliberately NOT swallowed — a broken real module is not "unimplemented"
    return False


def implemented_embedding_ids() -> tuple[str, ...]:
    return tuple(sorted(
        eid for eid in known_embedding_ids() if is_embedding_implemented(eid)))


def register_embedding(
        embedding_id: str,
        builder: Callable[[], NodeDeltaEmbedding],
        *,
        replace: bool = False,
) -> None:
    """Register a candidate builder (WP3 real ids or TEST-only dummies).

    * ``E0`` can never be registered — it structurally has no branch;
    * reserved ids ``E1``–``E6`` require ``replace=True`` (the WP3 path);
      until then their default build is the typed not-implemented refusal;
    * re-registering any other id requires ``replace=True``.
    """
    embedding_id = str(embedding_id)
    if embedding_id == EMBEDDING_ID_E0:
        raise V541InterfaceError(
            "V541_EMBEDDING_E0_NO_BRANCH",
            "E0 has no residual branch (e0_has_no_branch); nothing to register")
    if embedding_id in RESERVED_CANDIDATE_IDS and not replace:
        # E1–E6 are reserved campaign rows: only the explicit WP3 path
        # (replace=True) may install their real implementations; nothing
        # may squat them with a test/placeholder builder by accident.
        raise V541InterfaceError(
            "V541_EMBEDDING_FROZEN",
            f"{embedding_id!r} is a reserved candidate id; WP3 installs "
            f"the real implementation with replace=True")
    if not callable(builder):
        raise TypeError(f"builder must be callable, got {type(builder).__name__}")
    if embedding_id in _BUILDERS and not replace:
        raise V541InterfaceError(
            "V541_EMBEDDING_EXISTS",
            f"{embedding_id!r} already registered; pass replace=True to override")
    _BUILDERS[embedding_id] = builder


def build_embedding(embedding_id: str) -> NodeDeltaEmbedding | None:
    """Instantiate the branch for ``embedding_id`` (``None`` for E0).

    * E0 → ``None`` (no branch, zero new parameters);
    * registered id → ``builder()`` (RNG policy is the CALLER's job —
      ``V541Model`` wraps construction in ``torch.random.fork_rng`` with
      the independent embedding-init seed);
    * BUILT-IN but not yet imported → :func:`ensure_embedding` imports the
      real WP3 module first (a not-landed module and a broken module are
      DIFFERENT typed refusals, see there);
    * reserved-but-unimplemented candidate (today: E6) → typed refusal;
    * unknown id → typed unknown refusal.
    """
    embedding_id = str(embedding_id)
    if embedding_id == EMBEDDING_ID_E0:
        return None
    if embedding_id not in _BUILDERS and embedding_id in BUILTIN_EMBEDDING_MODULES:
        ensure_embedding(embedding_id)      # typed refusals propagate here
    builder = _BUILDERS.get(embedding_id)
    if builder is not None:
        module = builder()
        if not isinstance(module, NodeDeltaEmbedding):
            raise V541InterfaceError(
                "V541_EMBEDDING_BUILDER",
                f"builder for {embedding_id!r} returned "
                f"{type(module).__name__}, expected a NodeDeltaEmbedding")
        return module
    if embedding_id in EMBEDDING_SPECS:
        spec = EMBEDDING_SPECS[embedding_id]
        raise V541EmbeddingNotImplementedError(
            "V541_EMBEDDING_NOT_IMPLEMENTED",
            f"{embedding_id} ({spec.family}) is preregistered but not "
            f"implemented yet (WP3); refusing rather than substituting a "
            f"placeholder branch")
    raise V541EmbeddingUnknownError(
        "V541_EMBEDDING_UNKNOWN",
        f"unknown embedding id {embedding_id!r}; known {known_embedding_ids()}")


# ---------------------------------------------------------------------------
# test-isolation helpers (also usable by WP3 to swap implementations)
# ---------------------------------------------------------------------------
def registry_snapshot() -> dict[str, Callable[[], NodeDeltaEmbedding]]:
    """Copy of the registered-builder table (for scoped restore)."""
    return dict(_BUILDERS)


def registry_restore(snapshot: Mapping[str, Callable[[], NodeDeltaEmbedding]]) -> None:
    """Restore the builder table to a previous :func:`registry_snapshot`."""
    _BUILDERS.clear()
    _BUILDERS.update(dict(snapshot))


def registered_test_ids() -> Sequence[str]:
    """Registered ids that are NOT frozen campaign rows (test-only ids)."""
    return tuple(sorted(set(_BUILDERS) - set(FROZEN_EMBEDDING_IDS)))


__all__ = [
    "BUILTIN_EMBEDDING_MODULES",
    "EMBEDDING_SPECS",
    "EmbeddingSpec",
    "FROZEN_EMBEDDING_IDS",
    "MAIN_MATRIX_EMBEDDING_IDS",
    "RESERVED_CANDIDATE_IDS",
    "build_embedding",
    "embedding_spec",
    "ensure_embedding",
    "implemented_embedding_ids",
    "is_embedding_implemented",
    "is_embedding_known",
    "known_embedding_ids",
    "register_embedding",
    "registered_test_ids",
    "registry_restore",
    "registry_snapshot",
]
