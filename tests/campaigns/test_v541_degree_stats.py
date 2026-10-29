"""V5.4.1 E2 degree statistics — runner fit over the FULL inner-train set +
the trainer's strict TWO-argument interface (plan §6.6 / §6.7 / §17).

What this file pins down:

* **population** — the fit consumes EVERY inner-train position of the run
  seed (no sampling, no 4096 cap, no under-count);
* **view** — it delegates to the frozen
  ``v541_degree_stats.compute_degree_stats`` helper, which derives the
  degrees from the legal ``G_chem`` view only (no spatial edge, no self
  loop, no label weight);
* **seed** — one fit / cache file / SHA per RUN seed;
* **envelope** — the cache file is the trainer-compatible
  ``pes2ts_v541_degree_stats_v1`` document that
  ``load_degree_stats_v541`` reads and hands to
  ``set_degree_stats(mu_log, stats_sha256)`` — the real PNA signature;
* **identity** — the file SHA is part of the resume identity, so a re-fit
  or an edited envelope can never continue an existing run.

Fixture data only; no real 50K pool, no dev/confirm access, no GPU.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from pes2ts_core.pesn.models.v541 import (  # noqa: E402
    HIDDEN_V541,
    NodeDeltaEmbedding,
    build_embedding,
    register_embedding,
    registry_restore,
    registry_snapshot,
)
from pes2ts_core.pesn.offline.training.key_atom_v53mp_train_v3 import (  # noqa: E402
    v53mp_pool_split,
)
from pes2ts_core.pesn.offline.training.v541_degree_stats import (  # noqa: E402
    GRAPH_RECIPE_V541_CHEM,
    compute_degree_stats,
)
from pes2ts_core.pesn.offline.training.v541_train import (  # noqa: E402
    DEGREE_STATS_DIGEST_KEYS,
    V541_TRAIN_SCHEMA,
    V541TrainerConfig,
    V541TrainerError,
    assert_resume_identity,
    degree_stats_payload_digest,
    identity_block_v541,
    load_degree_stats_v541,
    load_model_v541,
)
from scripts.agent_ops.v541_config_schema import load_config  # noqa: E402
from scripts.campaigns.v541_provenance import (  # noqa: E402
    sha256_file,
    sha256_json,
)
from scripts.campaigns.v541_run import (  # noqa: E402
    RunInputs,
    degree_stats_required,
    ensure_degree_stats,
)
from tests.pesn.test_r3_v53mp_trainer import (  # noqa: E402
    DEFAULT_FOLDS,
    _build_env,
)

TEST_ID = "TEST_DEGREE_STATS_MLP"
PLAIN_ID = "TEST_NO_DEGREE_STATS_MLP"
POOL_CURVES = ("cv0", "cv1", "cv2")
DEV_CURVES = ("cv_dev0", "cv_dev1")
PROV = {"tier": "50k", "data_manifest_sha256": "d" * 64,
        "split_manifest_sha256": "s" * 64,
        "teacher_manifest_sha256": "t" * 64,
        "endpoint_label_spec_sha256": "e" * 64}


class _DegreeEmbedding(NodeDeltaEmbedding):
    """Candidate that CONSUMES frozen degree statistics (E2-shaped mock).

    The signature mirrors the REAL
    ``pna_chemical.PNAChemicalEmbedding.set_degree_stats(mu_log,
    stats_sha256)`` exactly — an envelope passed in one argument would
    ``TypeError``, which is the regression this guards against.
    """

    def __init__(self, hidden: int = HIDDEN_V541) -> None:
        super().__init__()
        self.net = torch.nn.Linear(hidden, hidden)
        self.mu_log = None
        self.stats_sha256 = None
        self.set_degree_stats_calls = 0

    def set_degree_stats(self, mu_log, stats_sha256) -> None:
        self.mu_log = float(mu_log)
        self.stats_sha256 = str(stats_sha256)
        self.set_degree_stats_calls += 1

    def forward(self, h_legacy, batch):          # batch deliberately unread
        return self.net(h_legacy)


class _PlainEmbedding(NodeDeltaEmbedding):
    """Candidate with NO degree-stats hook (E1/E3/E4-shaped)."""

    def __init__(self, hidden: int = HIDDEN_V541) -> None:
        super().__init__()
        self.net = torch.nn.Linear(hidden, hidden)

    def forward(self, h_legacy, batch):
        return self.net(h_legacy)


@pytest.fixture(autouse=True)
def _registry():
    snap = registry_snapshot()
    register_embedding(TEST_ID, lambda: _DegreeEmbedding())
    register_embedding(PLAIN_ID, lambda: _PlainEmbedding())
    yield
    registry_restore(snap)


@pytest.fixture()
def env(tmp_path):
    return _build_env(tmp_path, DEFAULT_FOLDS)


def _inputs(env) -> RunInputs:
    return RunInputs(
        dataset=env.ds, labels=env.labels, endpoint=env.endpoint,
        split_doc={"schema": "v53mp_split_v1",
                   "subsets": {"pool": list(POOL_CURVES),
                               "dev": list(DEV_CURVES), "confirm": []}},
        pool_curve_ids=list(POOL_CURVES),
        dev_curve_ids=list(DEV_CURVES),
        confirm_curve_ids=[],
        provenance={"tier": "50k"},
    )


def _inner_train(env, seed: int = 17) -> list:
    """The runner's inner-train position list for one seed (NOT the union)."""
    train_pos, _val = v53mp_pool_split(
        env.ds.samples, pool_curve_ids=list(POOL_CURVES), seed=seed)
    return sorted(int(p) for p in train_pos)


def _dev_positions(env) -> list:
    return [i for i, r in enumerate(env.ds.samples)
            if int(getattr(r, "fold", -1)) == 0]


def _fit(env, tmp_path, **over) -> dict:
    seed = int(over.pop("seed", 17))
    kwargs = dict(
        embedding_id="E2", inputs=_inputs(env),
        train_positions=_inner_train(env, seed), root=tmp_path / "root",
        code_sha="c" * 64,
        dev_positions=_dev_positions(env),
        confirm_curve_ids=[], seed=seed)
    kwargs.update(over)
    return ensure_degree_stats(load_config(), **kwargs)


# ---------------------------------------------------------------------------
# runner side: FULL inner-train population, G_chem only, per-run-seed cache
# ---------------------------------------------------------------------------
def test_e0_never_requires_degree_stats(env, tmp_path) -> None:
    out = _fit(env, tmp_path, embedding_id="E0")
    assert out == {"required": False, "path": None, "sha256": None}
    assert degree_stats_required(load_config(), "E0") is False
    assert degree_stats_required(load_config(), "E2") is True


def test_fit_uses_the_full_inner_train_population(env, tmp_path) -> None:
    inner_train = _inner_train(env)
    assert inner_train, "fixture must expose an inner-train slice"

    out = _fit(env, tmp_path)
    assert out["required"] is True and out["cache_hit"] is False
    doc = json.loads(Path(out["path"]).read_text(encoding="utf-8"))

    # NO cap / NO sampling: the population IS the inner-train position set
    assert doc["inputs"]["population"] == "full_inner_train"
    assert doc["inputs"]["split"] == "inner-train"
    assert doc["n_positions"] == len(set(inner_train)) == len(inner_train)
    assert doc["weighted_graph_total"] == len(inner_train)
    assert sum(doc["degree_histogram"].values()) == doc["weighted_node_total"]
    assert doc["histogram"] == doc["degree_histogram"]
    # the run seed is bound into both the fit and the cache inputs
    assert doc["seed"] == 17 and doc["inputs"]["seed"] == 17

    # ...and it equals a direct call to the frozen helper on the same set
    direct = compute_degree_stats(env.ds.samples, inner_train,
                                  split="inner-train", seed=17)
    assert doc["stats_sha256"] == direct.stats_sha256
    assert doc["mu_log"] == pytest.approx(direct.mu_log)
    assert doc["degree_histogram"] == {
        str(k): v for k, v in sorted(direct.degree_histogram.items())}


def test_fit_is_g_chem_only_never_spatial(env, tmp_path) -> None:
    out = _fit(env, tmp_path)
    doc = json.loads(Path(out["path"]).read_text(encoding="utf-8"))
    assert doc["inputs"]["graph_view"] == "G_chem"
    assert doc["graph_recipe_sha256"] == sha256_json(
        dict(GRAPH_RECIPE_V541_CHEM))
    assert "no spatial" in GRAPH_RECIPE_V541_CHEM["degree"]
    assert "G_chem" in doc["definition"]
    # a bidirectionally expanded chem graph: edges == 2 x pairs exactly
    assert doc["weighted_edge_total"] == 2 * doc["weighted_pair_total"]
    # the legacy full-edge (chem + spatial) view is NOT what was hashed
    assert "edge_index" not in doc and "spatial" not in str(
        doc["graph_recipe_sha256"])


def test_fit_cache_and_sha_are_per_run_seed(env, tmp_path) -> None:
    first = _fit(env, tmp_path, seed=17)
    second = _fit(env, tmp_path, seed=17)
    assert second["cache_hit"] is True
    assert second["sha256"] == first["sha256"]
    assert second["path"] == first["path"]

    other = _fit(env, tmp_path, seed=29)
    assert other["path"] != first["path"], "one cache file per run seed"
    assert other["sha256"] != first["sha256"]
    assert other["seed"] == 29
    doc29 = json.loads(Path(other["path"]).read_text(encoding="utf-8"))
    assert doc29["seed"] == 29 and doc29["inputs"]["seed"] == 29


def test_code_change_invalidates_the_cache(env, tmp_path) -> None:
    first = _fit(env, tmp_path, seed=17)
    changed = _fit(env, tmp_path, seed=17, code_sha="d" * 64)
    assert changed["cache_hit"] is False
    assert changed["path"] != first["path"]
    assert changed["sha256"] != first["sha256"]


def test_envelope_carries_full_provenance_and_cost(env, tmp_path) -> None:
    out = _fit(env, tmp_path)
    doc = json.loads(Path(out["path"]).read_text(encoding="utf-8"))
    for key in ("mu_log", "stats_sha256", "degree_histogram",
                "split_sha256", "positions_sha256", "graph_recipe_sha256",
                "seed_sha256", "n_unique_graphs", "weighted_graph_total",
                "weighted_node_total", "weighted_edge_total",
                "weighted_pair_total", "elapsed_seconds", "max_rss_kb",
                "inputs_sha256"):
        assert key in doc, key
    assert doc["mu_log"] > 0.0
    assert len(doc["stats_sha256"]) == 64
    assert doc["forward_recomputation"] is False
    # the trainer's digest recomputation must accept this envelope
    assert degree_stats_payload_digest(doc) == doc["stats_sha256"]
    assert set(DEGREE_STATS_DIGEST_KEYS) <= set(doc)


def test_fit_refuses_dev_positions(env, tmp_path) -> None:
    poisoned = _inner_train(env)[:1]              # pretend a dev row leaked in
    with pytest.raises(Exception) as exc:
        _fit(env, tmp_path, dev_positions=poisoned)
    assert exc.value.code == "V541_DEGREE_STATS_FIT"


def test_fit_refuses_confirm_curve(env, tmp_path) -> None:
    with pytest.raises(Exception) as exc:
        _fit(env, tmp_path, confirm_curve_ids=["cv0"])
    assert exc.value.code == "V541_DEGREE_STATS_FIT"


def test_fit_refuses_empty_slice(env, tmp_path) -> None:
    with pytest.raises(Exception) as exc:
        _fit(env, tmp_path, train_positions=[])
    assert exc.value.code == "V541_DEGREE_STATS_FIT"


# ---------------------------------------------------------------------------
# trainer config interface
# ---------------------------------------------------------------------------
def test_config_requires_path_and_sha_together() -> None:
    with pytest.raises(V541TrainerError) as exc:
        V541TrainerConfig(seed=17, backbone_init="scratch",
                          degree_stats_path="/tmp/x.json")
    assert exc.value.code == "DEGREE_STATS_INCOMPLETE"


def test_config_rejects_malformed_sha() -> None:
    with pytest.raises(V541TrainerError) as exc:
        V541TrainerConfig(seed=17, backbone_init="scratch",
                          degree_stats_path="/tmp/x.json",
                          degree_stats_sha256="nope")
    assert exc.value.code == "DEGREE_STATS_SHA"


# ---------------------------------------------------------------------------
# strict read (envelope validation incl. the two consumed fields)
# ---------------------------------------------------------------------------
def _digest_keys_doc(**over) -> dict:
    """A digest-complete payload (all DEGREE_STATS_DIGEST_KEYS present)."""
    doc = {
        "schema": "pes2ts_v541_degree_stats_v1",
        "split": "inner-train",
        "split_sha256": "a" * 64,
        "positions_sha256": "b" * 64,
        "graph_recipe_sha256": "c" * 64,
        "seed": 17,
        "seed_sha256": "d" * 64,
        "mu_log": 0.75,
        "n_positions": 42,
        "n_unique_graphs": 3,
        "weighted_graph_total": 42,
        "weighted_node_total": 294,
        "weighted_edge_total": 240,
        "weighted_pair_total": 120,
        "degree_histogram": {"1": 6, "2": 20, "3": 16},
    }
    doc.update(over)
    return doc


def _write_stats(path: Path, *, tamper: bool = False, **over) -> str:
    """Write a digest-valid envelope (``tamper``/``over`` break it on purpose)."""
    doc = _digest_keys_doc()
    digest = "e" * 64 if tamper else degree_stats_payload_digest(doc)
    doc["stats_sha256"] = digest
    doc.setdefault("fit_split", "inner-train_only")
    doc.setdefault("fit_forbidden_splits",
                   ["dev", "inner-val", "confirm", "dev_confirm"])
    doc.setdefault("histogram", dict(doc["degree_histogram"]))
    doc.update(over)                        # overrides land AFTER the digest
    path.write_text(json.dumps(doc), encoding="utf-8")
    return sha256_file(path)


def test_strict_read_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(V541TrainerError) as exc:
        load_degree_stats_v541(tmp_path / "absent.json", "a" * 64)
    assert exc.value.code == "DEGREE_STATS_MISSING"


def test_strict_read_rejects_file_sha_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "stats.json"
    _write_stats(path)
    with pytest.raises(V541TrainerError) as exc:
        load_degree_stats_v541(path, "b" * 64)
    assert exc.value.code == "DEGREE_STATS_SHA_MISMATCH"
    assert "degree_stats_sha256" in exc.value.fields


def test_strict_read_rejects_wrong_schema(tmp_path: Path) -> None:
    path = tmp_path / "stats.json"
    path.write_text(json.dumps({"schema": "something_else",
                                "histogram": {"1": 1}}), encoding="utf-8")
    with pytest.raises(V541TrainerError) as exc:
        load_degree_stats_v541(path, sha256_file(path))
    assert exc.value.code == "DEGREE_STATS_SCHEMA"


def test_strict_read_rejects_dev_fit(tmp_path: Path) -> None:
    path = tmp_path / "stats.json"
    digest = _write_stats(path, fit_split="dev")
    with pytest.raises(V541TrainerError) as exc:
        load_degree_stats_v541(path, digest)
    assert exc.value.code == "DEGREE_STATS_FIT"


@pytest.mark.parametrize("over", [
    {"mu_log": None},
    {"mu_log": 0.0},
    {"mu_log": float("nan")},
    {"stats_sha256": None},
    {"stats_sha256": "not-a-digest"},
])
def test_strict_read_rejects_bad_consumed_fields(tmp_path: Path,
                                                 over) -> None:
    path = tmp_path / "stats.json"
    digest = _write_stats(path, **over)
    with pytest.raises(V541TrainerError) as exc:
        load_degree_stats_v541(path, digest)
    assert exc.value.code == "DEGREE_STATS_SCHEMA"


def test_strict_read_rejects_a_tampered_envelope(tmp_path: Path) -> None:
    """Editing mu_log after the fit breaks the recomputed payload digest."""
    path = tmp_path / "stats.json"
    doc = _digest_keys_doc()
    doc["stats_sha256"] = degree_stats_payload_digest(doc)
    doc["mu_log"] = 9.99                                   # post-fit edit
    doc["fit_split"] = "inner-train_only"
    doc["fit_forbidden_splits"] = ["dev", "inner-val", "confirm"]
    doc["histogram"] = dict(doc["degree_histogram"])
    path.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(V541TrainerError) as exc:
        load_degree_stats_v541(path, sha256_file(path))
    assert exc.value.code == "DEGREE_STATS_SHA_MISMATCH"


# ---------------------------------------------------------------------------
# load_model_v541 dispatch + identity / resume
# ---------------------------------------------------------------------------
def _cfg(path: str, sha: str, embedding_id: str = TEST_ID,
         **over) -> V541TrainerConfig:
    base = dict(seed=17, backbone_init="scratch", embedding_id=embedding_id,
                init_seed=1000020 if embedding_id != "E0" else None,
                degree_stats_path=path, degree_stats_sha256=sha)
    base.update(over)
    return V541TrainerConfig(**base)


def test_load_model_calls_set_degree_stats_with_two_arguments(
        tmp_path: Path) -> None:
    path = tmp_path / "stats.json"
    digest = _write_stats(path)
    model = load_model_v541(_cfg(str(path), digest), device="cpu")
    branch = model.embedding
    assert branch.set_degree_stats_calls == 1
    # the REAL PNA signature is (mu_log, stats_sha256) — NOT the envelope
    assert branch.mu_log == pytest.approx(0.75)
    assert branch.stats_sha256 == degree_stats_payload_digest(
        json.loads(path.read_text(encoding="utf-8")))


def test_load_model_refuses_missing_stats_for_consumer() -> None:
    with pytest.raises(V541TrainerError) as exc:
        load_model_v541(_cfg(None, None), device="cpu")
    assert exc.value.code == "DEGREE_STATS_REQUIRED"


def test_load_model_refuses_stats_without_consumer(tmp_path: Path) -> None:
    path = tmp_path / "stats.json"
    digest = _write_stats(path)
    with pytest.raises(V541TrainerError) as exc:
        load_model_v541(_cfg(str(path), digest, embedding_id=PLAIN_ID),
                        device="cpu")
    assert exc.value.code == "DEGREE_STATS_NOT_SUPPORTED"


def test_load_model_refuses_stats_on_e0(tmp_path: Path) -> None:
    path = tmp_path / "stats.json"
    digest = _write_stats(path)
    with pytest.raises(V541TrainerError) as exc:
        load_model_v541(_cfg(str(path), digest, embedding_id="E0",
                             init_seed=None), device="cpu")
    assert exc.value.code == "DEGREE_STATS_NOT_SUPPORTED"


# ---------------------------------------------------------------------------
# REAL PNA integration (the WP3 E2 module consumes the runner envelope)
# ---------------------------------------------------------------------------
def _e2_available() -> bool:
    from pes2ts_core.pesn.models.v541.embedding_registry import (
        is_embedding_implemented,
    )

    return bool(is_embedding_implemented("E2"))


def test_real_pna_embedding_consumes_the_runner_envelope(env,
                                                         tmp_path) -> None:
    if not _e2_available():                     # pragma: no cover
        pytest.skip("E2 module not landed")
    out = _fit(env, tmp_path, seed=17)
    branch = build_embedding("E2")
    # EXACTLY the two-argument call the trainer makes
    branch.set_degree_stats(out["mu_log"], out["stats_sha256"])
    prov = branch.degree_stats_provenance
    assert prov["state"] == "ready"          # a valid (mu_log, digest) pair
    assert prov["mu_log"] == pytest.approx(out["mu_log"])
    assert prov["stats_sha256"] == out["stats_sha256"]


def test_load_model_v541_with_the_real_e2(env, tmp_path) -> None:
    """End-to-end: runner envelope -> load_model_v541 -> real PNA buffers."""
    if not _e2_available():                     # pragma: no cover
        pytest.skip("E2 module not landed")
    out = _fit(env, tmp_path, seed=17)
    cfg = _cfg(out["path"], out["sha256"], embedding_id="E2")
    model = load_model_v541(cfg, device="cpu")
    prov = model.embedding.degree_stats_provenance
    assert prov["state"] == "ready"
    assert prov["mu_log"] == pytest.approx(out["mu_log"])
    assert prov["stats_sha256"] == out["stats_sha256"]


def test_identity_carries_degree_stats_sha(tmp_path: Path) -> None:
    path = tmp_path / "stats.json"
    digest = _write_stats(path)
    with_stats = identity_block_v541(_cfg(str(path), digest), PROV,
                                     pool_sha256=None)
    without = identity_block_v541(_cfg(None, None), PROV, pool_sha256=None)
    assert with_stats["degree_stats_sha256"] == digest
    assert without["degree_stats_sha256"] is None
    assert with_stats["config_sha256"] != without["config_sha256"]


def test_resume_refuses_degree_stats_sha_drift(tmp_path: Path) -> None:
    path = tmp_path / "stats.json"
    digest = _write_stats(path)
    frozen = identity_block_v541(_cfg(str(path), digest), PROV,
                                 pool_sha256=None)
    state = {"schema": V541_TRAIN_SCHEMA, "identity": frozen}
    assert_resume_identity(state, _cfg(str(path), digest), PROV,
                           pool_sha256=None)
    with pytest.raises(V541TrainerError) as exc:
        assert_resume_identity(state, _cfg(str(path), "e" * 64), PROV,
                               pool_sha256=None)
    assert exc.value.code == "RESUME_IDENTITY_MISMATCH"
    assert "degree_stats_sha256" in exc.value.fields
