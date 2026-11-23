"""R6 split firewall as pure, testable functions (Zone 2).

The firewall has two halves:

1. **Declarative** -- :class:`SplitPolicy` is the validated in-memory form of
   ``config/data/splits.yaml`` (five-layer L0-L4 chain, four evaluation suites,
   one sealed confirm set, ``family_holdout_v1``).
2. **Procedural** -- :func:`assert_no_family_leak`, :func:`assert_sealed` and
   :func:`check_split_frame` are the guards every loader / evaluator must call
   before it touches data.

Constitution mapping: R6 (evaluation integrity / split firewall) and P2 (typed
honest rejection -- every guard raises :class:`SplitViolation` with a stable
``.code`` instead of asserting / silently dropping rows).

All functions are pure: they never read data, never mutate their arguments and
never access truth.  The only I/O is :func:`load_split_policy` reading the YAML
file it is explicitly handed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

__all__ = [
    "BASE_SPLIT_NAMES",
    "CONFIRM_N",
    "CONFIRM_SPLIT",
    "REQUIRED_LAYERS",
    "SPLIT_POLICY_SCHEMA_ID",
    "TEST_SET_NAMES",
    "ConfirmSpec",
    "LayerSpec",
    "RulesSpec",
    "SplitPolicy",
    "SplitPolicyError",
    "SplitViolation",
    "TestSetSpec",
    "assert_no_family_leak",
    "assert_sealed",
    "check_split_frame",
    "default_split_policy_path",
    "family_leaks",
    "load_split_policy",
    "validate_split_policy_doc",
]

#: Schema id carried by ``config/data/splits.yaml``.
SPLIT_POLICY_SCHEMA_ID = "pes2ts_split_policy_v1"

#: The only supported holdout policy.
FAMILY_HOLDOUT_POLICY = "family_holdout_v1"

#: Fixed layer names of the inheritance chain (L0 -> L4).
REQUIRED_LAYERS: tuple[str, ...] = ("L0", "L1", "L2", "L3", "L4")

#: The four evaluation suites; names are frozen by the data design.
TEST_SET_NAMES: tuple[str, ...] = ("id_dev", "scaffold_ood", "family_ood", "size_ood")

#: Name of the sealed confirm set.
CONFIRM_SPLIT = "confirm"

#: Frozen size of the confirm set (2,000 id-only rows).
CONFIRM_N = 2000

#: Base split names of the L0/L1 chain.
BASE_SPLIT_NAMES: tuple[str, ...] = ("train", "valid", "test")


class SplitViolation(Exception):
    """Typed split-firewall rejection (P2).

    Attributes
    ----------
    code:
        Stable machine code, e.g. ``FAMILY_LEAK``, ``SEALED_SPLIT_ACCESS``,
        ``SPLIT_OVERLAP``, ``MISSING_SPLIT_KEY``.
    detail:
        Human-readable explanation.
    split:
        Optional split name involved in the violation.
    families:
        Optional mapping of offending family -> example ids (for ``FAMILY_LEAK``).
    """

    def __init__(
        self,
        code: str,
        detail: str,
        *,
        split: str | None = None,
        families: Mapping[str, SequenceLike] | None = None,
    ) -> None:
        self.code = str(code)
        self.detail = str(detail)
        self.split = split
        self.families: dict[str, list[str]] | None = (
            {str(k): [str(x) for x in v] for k, v in families.items()}
            if families is not None else None)
        prefix = f"{self.code}: {self.detail}"
        if split is not None:
            prefix = f"{self.code} [{split}]: {self.detail}"
        super().__init__(prefix)


class SplitPolicyError(SplitViolation, ValueError):
    """Malformed / unsupported split policy document (config-level error)."""


# ---------------------------------------------------------------------------
# Pydantic v2 schema of config/data/splits.yaml
# ---------------------------------------------------------------------------

class LayerSpec(BaseModel):
    """One layer of the L0-L4 inheritance chain."""

    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)

    name: str = Field(min_length=2)
    artifact: str | None = None
    schema_id: str | None = Field(default=None, alias="schema")
    frozen: bool
    historical: bool = False
    recomputed_per_run: bool = False
    description: str = Field(min_length=1)


class TestSetSpec(BaseModel):
    """One of the four evaluation suites."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    layer: str
    kind: str
    description: str = Field(min_length=1)
    access: str
    sealed: bool = False
    n_hint: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _check_access(self) -> "TestSetSpec":
        if self.access not in ("evaluation_only", "one_shot_dev", "sealed_never_read"):
            raise ValueError(f"unknown access policy {self.access!r}")
        return self


class ConfirmSpec(BaseModel):
    """The permanently sealed confirm set."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    n: int = Field(gt=0)
    sealed: bool
    layer: str
    description: str = Field(min_length=1)
    access: str = "sealed_never_read"


class RulesSpec(BaseModel):
    """Firewall rules carried by the policy document."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    family_holdout_required: bool = True
    family_label_source: str | None = None
    family_column: str | None = None
    sealed_never_read: list[str] = Field(default_factory=lambda: [CONFIRM_SPLIT])
    no_holdout_tuning: bool = True
    reverse_and_neighbor_grouping: bool = True
    eval_read_once: bool = True
    required_split_keys: list[str] = Field(
        default_factory=lambda: list(BASE_SPLIT_NAMES))


class SplitPolicy(BaseModel):
    """Validated R6 split policy (in-memory form of ``config/data/splits.yaml``)."""

    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)

    schema_id: str = Field(alias="schema")
    policy: str
    version: str
    unit: str
    description: str = Field(min_length=1)
    layers: dict[str, LayerSpec]
    test_sets: dict[str, TestSetSpec]
    confirm: ConfirmSpec
    rules: RulesSpec = Field(default_factory=RulesSpec)

    # ------------------------------------------------------------- invariants
    @model_validator(mode="after")
    def _check_invariants(self) -> "SplitPolicy":
        if self.schema_id != SPLIT_POLICY_SCHEMA_ID:
            raise ValueError(
                f"schema must be {SPLIT_POLICY_SCHEMA_ID!r}, got {self.schema_id!r}")
        if self.policy != FAMILY_HOLDOUT_POLICY:
            raise ValueError(
                f"policy must be {FAMILY_HOLDOUT_POLICY!r}, got {self.policy!r}")
        layer_keys = tuple(self.layers.keys())
        if set(layer_keys) != set(REQUIRED_LAYERS):
            raise ValueError(
                f"layers must be exactly {REQUIRED_LAYERS}, got {layer_keys}")
        set_keys = tuple(self.test_sets.keys())
        if set(set_keys) != set(TEST_SET_NAMES):
            raise ValueError(
                f"test_sets must be exactly {TEST_SET_NAMES}, got {set_keys}")
        for set_name, spec in self.test_sets.items():
            if spec.layer not in self.layers:
                raise ValueError(
                    f"test set {set_name!r} references unknown layer {spec.layer!r}")
        if self.confirm.layer not in self.layers:
            raise ValueError(
                f"confirm references unknown layer {self.confirm.layer!r}")
        if self.confirm.n != CONFIRM_N:
            raise ValueError(
                f"confirm.n is frozen at {CONFIRM_N:,}, got {self.confirm.n!r}")
        if not self.confirm.sealed:
            raise ValueError("confirm must be sealed=true (R6)")
        if CONFIRM_SPLIT not in self.rules.sealed_never_read:
            raise ValueError("rules.sealed_never_read must list 'confirm'")
        if not self.rules.family_holdout_required:
            raise ValueError("family_holdout_required must stay true (R6)")
        missing = [key for key in self.rules.required_split_keys
                   if key not in BASE_SPLIT_NAMES]
        if missing:
            raise ValueError(
                f"required_split_keys must be a subset of {BASE_SPLIT_NAMES}, "
                f"got extras {missing}")
        return self

    # ------------------------------------------------------------- properties
    @property
    def test_set_names(self) -> tuple[str, ...]:
        """Names of the four evaluation suites (declared order)."""
        return tuple(self.test_sets.keys())

    @property
    def layer_names(self) -> tuple[str, ...]:
        """Names of the five layers (L0-L4, declared order)."""
        return tuple(self.layers.keys())

    @property
    def sealed_split_names(self) -> frozenset[str]:
        """All split names that must not be read without explicit authorisation."""
        names = {CONFIRM_SPLIT} | {name for name, spec in self.test_sets.items()
                                  if spec.sealed}
        return frozenset(names)

    @property
    def evaluation_split_names(self) -> frozenset[str]:
        """Split names that are legitimately used for evaluation."""
        return frozenset(self.test_set_names) | {CONFIRM_SPLIT}

    def layer_of(self, split: str) -> str:
        """Return the layer that produces ``split``.

        Raises
        ------
        SplitViolation
            ``UNKNOWN_SPLIT`` when ``split`` is not declared by the policy.
        """
        if split in self.test_sets:
            return self.test_sets[split].layer
        if split == CONFIRM_SPLIT:
            return self.confirm.layer
        if split in BASE_SPLIT_NAMES:
            return "L0"
        raise SplitViolation("UNKNOWN_SPLIT", f"split {split!r} is not declared "
                                             "by the split policy", split=split)

    def is_sealed(self, split: str) -> bool:
        """Return ``True`` when ``split`` is sealed (confirm and sealed suites)."""
        return split in self.sealed_split_names


# ``SequenceLike`` is only used in the SplitViolation type hint above.
SequenceLike = Iterable[str]


# ---------------------------------------------------------------------------
# Loading / document validation
# ---------------------------------------------------------------------------

def default_split_policy_path() -> Path:
    """Return the canonical in-repo path of ``config/data/splits.yaml``."""
    return Path(__file__).resolve().parents[2] / "config" / "data" / "splits.yaml"


def _require_mapping(doc: Any, code: str, what: str) -> Mapping[str, Any]:
    if not isinstance(doc, Mapping):
        raise SplitPolicyError(
            code, f"{what} must be a mapping, got {type(doc).__name__!r}")
    return doc


def validate_split_policy_doc(doc: Any) -> dict[str, Any]:
    """Structurally validate a raw policy document and return it as a ``dict``.

    This pre-check produces precise codes (``POLICY_LAYER_MISSING``,
    ``POLICY_CONFIRM_INVALID`` ...) before pydantic takes over for full type
    validation, so callers always receive a typed :class:`SplitPolicyError`.
    """
    doc = _require_mapping(doc, "POLICY_MALFORMED", "split policy")
    if doc.get("schema") != SPLIT_POLICY_SCHEMA_ID:
        raise SplitPolicyError(
            "POLICY_SCHEMA_MISMATCH",
            f"expected schema {SPLIT_POLICY_SCHEMA_ID!r}, got {doc.get('schema')!r}")
    if doc.get("policy") != FAMILY_HOLDOUT_POLICY:
        raise SplitPolicyError(
            "POLICY_NAME_MISMATCH",
            f"expected policy {FAMILY_HOLDOUT_POLICY!r}, got {doc.get('policy')!r}")

    layers = _require_mapping(doc.get("layers"), "POLICY_LAYER_MISSING",
                              "layers")
    missing_layers = [name for name in REQUIRED_LAYERS if name not in layers]
    if missing_layers:
        raise SplitPolicyError(
            "POLICY_LAYER_MISSING",
            f"missing layer(s) {missing_layers}; expected {list(REQUIRED_LAYERS)}")
    unknown_layers = sorted(set(layers) - set(REQUIRED_LAYERS))
    if unknown_layers:
        raise SplitPolicyError(
            "POLICY_LAYER_UNKNOWN",
            f"unknown layer(s) {unknown_layers}; expected {list(REQUIRED_LAYERS)}")

    test_sets = _require_mapping(doc.get("test_sets"), "POLICY_TEST_SET_MISSING",
                                 "test_sets")
    missing_sets = [name for name in TEST_SET_NAMES if name not in test_sets]
    if missing_sets:
        raise SplitPolicyError(
            "POLICY_TEST_SET_MISSING",
            f"missing test set(s) {missing_sets}; expected {list(TEST_SET_NAMES)}")
    unknown_sets = sorted(set(test_sets) - set(TEST_SET_NAMES))
    if unknown_sets:
        raise SplitPolicyError(
            "POLICY_TEST_SET_UNKNOWN",
            f"unknown test set(s) {unknown_sets}; expected {list(TEST_SET_NAMES)}")

    confirm = _require_mapping(doc.get("confirm"), "POLICY_CONFIRM_MISSING",
                               "confirm")
    n = confirm.get("n")
    if not isinstance(n, int) or isinstance(n, bool) or n != CONFIRM_N:
        raise SplitPolicyError(
            "POLICY_CONFIRM_INVALID",
            f"confirm.n must be the frozen integer {CONFIRM_N:,}, got {n!r}")
    if confirm.get("sealed") is not True:
        raise SplitPolicyError(
            "POLICY_CONFIRM_INVALID",
            f"confirm.sealed must be true, got {confirm.get('sealed')!r}")
    return dict(doc)


def load_split_policy(path: str | Path | None = None) -> SplitPolicy:
    """Load and validate ``config/data/splits.yaml`` (or an explicit path).

    Parameters
    ----------
    path:
        Path to the policy YAML.  ``None`` uses :func:`default_split_policy_path`.

    Returns
    -------
    SplitPolicy
        Fully validated, frozen policy object.

    Raises
    ------
    SplitPolicyError
        ``POLICY_FILE_MISSING`` / ``POLICY_PARSE_ERROR`` /
        ``POLICY_MALFORMED`` with a stable ``.code``; never a bare
        ``FileNotFoundError`` / ``yaml.YAMLError`` / ``pydantic.ValidationError``.
    """
    target = Path(path) if path is not None else default_split_policy_path()
    try:
        text = target.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise SplitPolicyError(
            "POLICY_FILE_MISSING", f"split policy file not found: {target}") from exc
    except OSError as exc:
        raise SplitPolicyError(
            "POLICY_PARSE_ERROR", f"cannot read split policy {target}: {exc}") from exc

    try:
        import yaml  # local import: policy module stays importable without PyYAML
    except ImportError as exc:  # pragma: no cover - pyyaml is a repo dependency
        raise SplitPolicyError(
            "POLICY_PARSE_ERROR", "PyYAML is required to load the split policy") from exc
    try:
        doc = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise SplitPolicyError(
            "POLICY_PARSE_ERROR", f"invalid YAML in {target}: {exc}") from exc

    doc = validate_split_policy_doc(doc)
    try:
        return SplitPolicy.model_validate(doc)
    except ValidationError as exc:
        raise SplitPolicyError(
            "POLICY_MALFORMED", f"invalid split policy {target}: {exc}") from exc


# ---------------------------------------------------------------------------
# Runtime guards
# ---------------------------------------------------------------------------

def _normalise_ids(values: Any, label: str) -> list[str]:
    """Normalise an iterable of reaction ids; raises ``BAD_SPLIT_VALUE``."""
    if values is None:
        raise SplitViolation(
            "BAD_SPLIT_VALUE", f"{label} is None; pass an iterable of ids", split=label)
    if isinstance(values, str):
        return [values]
    try:
        ids = list(values)
    except TypeError as exc:
        raise SplitViolation(
            "BAD_SPLIT_VALUE",
            f"{label} must be an iterable of ids, got {type(values).__name__!r}",
            split=label) from exc
    out: list[str] = []
    for item in ids:
        if isinstance(item, str):
            out.append(item)
        elif isinstance(item, (int,)) and not isinstance(item, bool):
            out.append(str(item))
        else:
            raise SplitViolation(
                "BAD_SPLIT_VALUE",
                f"{label} contains non-string id {item!r}", split=label)
    return out


def family_leaks(
    train: Iterable[str],
    test: Iterable[str],
    families: Mapping[str, str],
) -> dict[str, list[str]]:
    """Return ``family -> [ids]`` for families that appear on both sides.

    ``train``/``test`` are iterables of reaction (or canonical group) ids;
    ``families`` maps id -> reaction family.  Ids without a family entry are
    reported under the family key ``"<unlabelled>"`` so the caller can decide
    (P2: never silently drop).
    """
    train_ids = _normalise_ids(train, "train")
    test_ids = _normalise_ids(test, "test")
    family_map = {str(key): str(value) for key, value in dict(families).items()}
    sides: dict[str, dict[str, list[str]]] = {}
    for side, ids in (("train", train_ids), ("test", test_ids)):
        for rid in ids:
            family = family_map.get(rid, "<unlabelled>")
            sides.setdefault(family, {"train": [], "test": []})[side].append(rid)
    leaks: dict[str, list[str]] = {}
    for family, by_side in sides.items():
        if by_side["train"] and by_side["test"]:
            affected = sorted(by_side["train"] + by_side["test"])
            leaks[family] = affected
    return leaks


def assert_no_family_leak(
    train: Iterable[str],
    test: Iterable[str],
    families: Mapping[str, str],
    *,
    require_labels: bool = True,
) -> None:
    """Assert that no reaction family straddles ``train`` and ``test`` (R6).

    Parameters
    ----------
    train, test:
        Iterables of reaction / canonical-group ids.
    families:
        Mapping id -> reaction family label (e.g. ``curve_family.csv`` labels).
    require_labels:
        When ``True`` (default), an id without a family label is a typed
        rejection ``FAMILY_LABEL_MISSING`` instead of being ignored (P2).

    Raises
    ------
    SplitViolation
        ``REACTION_LEAK`` (same id on both sides), ``FAMILY_LABEL_MISSING``
        (unlabelled id) or ``FAMILY_LEAK`` (a family crosses the boundary).
    """
    train_ids = _normalise_ids(train, "train")
    test_ids = _normalise_ids(test, "test")
    family_map = {str(key): str(value) for key, value in dict(families).items()}

    overlap = sorted(set(train_ids) & set(test_ids))
    if overlap:
        raise SplitViolation(
            "REACTION_LEAK",
            f"{len(overlap)} reaction id(s) appear in both train and test, "
            f"e.g. {overlap[:5]}",
            split="train/test")

    if require_labels:
        missing = sorted({rid for rid in (*train_ids, *test_ids)
                          if rid not in family_map})
        if missing:
            raise SplitViolation(
                "FAMILY_LABEL_MISSING",
                f"{len(missing)} id(s) have no reaction-family label, e.g. "
                f"{missing[:5]}; label every id or pass require_labels=False",
                split="train/test")

    leaks = family_leaks(train_ids, test_ids, family_map)
    if leaks:
        sample = dict(list(sorted(leaks.items()))[:5])
        raise SplitViolation(
            "FAMILY_LEAK",
            f"{len(leaks)} reaction family(ies) cross the train/test firewall, "
            f"e.g. {sorted(leaks)[:5]}",
            split="train/test",
            families=sample)


def assert_sealed(
    name: str,
    *,
    allow_confirm: bool = False,
    policy: SplitPolicy | None = None,
) -> None:
    """Refuse evaluation on the sealed confirm set (R6).

    Parameters
    ----------
    name:
        Split / evaluation-set name that the caller intends to read.
    allow_confirm:
        Explicit, auditable escape hatch for the single final confirm run.
        Never pass this for development, diagnostics or model selection.
    policy:
        Optional validated policy; when omitted the default sealed set is
        ``{confirm}`` and the four suite names are known.

    Raises
    ------
    SplitViolation
        ``SEALED_SPLIT_ACCESS`` for a sealed name without ``allow_confirm``;
        ``UNKNOWN_SPLIT`` for a name the firewall does not know.
    """
    split = str(name)
    if policy is not None:
        sealed = set(policy.sealed_split_names)
        known = (set(BASE_SPLIT_NAMES) | set(policy.test_set_names) | sealed)
    else:
        sealed = {CONFIRM_SPLIT}
        known = set(BASE_SPLIT_NAMES) | set(TEST_SET_NAMES) | sealed

    if split in sealed:
        if allow_confirm:
            return
        raise SplitViolation(
            "SEALED_SPLIT_ACCESS",
            f"split {split!r} is sealed_never_read; evaluation requires the "
            "explicit final-run authorization (allow_confirm=True) and an "
            "isolated evaluator (R6)",
            split=split)
    if split not in known:
        raise SplitViolation(
            "UNKNOWN_SPLIT", f"split {split!r} is not declared by the split "
                             "firewall", split=split)


def check_split_frame(split: Mapping[str, Any]) -> dict[str, frozenset[str]]:
    """Validate a train/valid/test split frame and return it as ``frozenset``s.

    Checks
    ------
    * the frame is a mapping carrying ``train``/``valid``/``test``
      (``MALFORMED_SPLIT_FRAME``, ``MISSING_SPLIT_KEY``);
    * every value is an iterable of string ids (``BAD_SPLIT_VALUE``);
    * no required split is empty (``EMPTY_SPLIT``);
    * ``train``/``valid``/``test`` are pairwise disjoint, and a supplied
      ``confirm`` key is disjoint from them too (``SPLIT_OVERLAP``).

    Returns
    -------
    dict[str, frozenset[str]]
        Normalised frame for every provided key.

    """
    if not isinstance(split, Mapping):
        raise SplitViolation(
            "MALFORMED_SPLIT_FRAME",
            f"split frame must be a mapping, got {type(split).__name__!r}")
    frame: dict[str, frozenset[str]] = {}
    for key in (*BASE_SPLIT_NAMES, CONFIRM_SPLIT):
        if key in split:
            frame[key] = frozenset(_normalise_ids(split[key], key))
    missing = [key for key in BASE_SPLIT_NAMES if key not in frame]
    if missing:
        raise SplitViolation(
            "MISSING_SPLIT_KEY",
            f"split frame is missing required key(s) {missing}", split=missing[0])
    empty = [key for key in BASE_SPLIT_NAMES if not frame[key]]
    if empty:
        raise SplitViolation(
            "EMPTY_SPLIT", f"split(s) {empty} are empty", split=empty[0])

    overlap_checked = list(BASE_SPLIT_NAMES) + (
        [CONFIRM_SPLIT] if CONFIRM_SPLIT in frame else [])
    for i, left in enumerate(overlap_checked):
        for right in overlap_checked[i + 1:]:
            shared = frame[left] & frame[right]
            if shared:
                raise SplitViolation(
                    "SPLIT_OVERLAP",
                    f"{len(shared)} id(s) appear in both {left!r} and {right!r}, "
                    f"e.g. {sorted(shared)[:5]}",
                    split=left)
    return frame
