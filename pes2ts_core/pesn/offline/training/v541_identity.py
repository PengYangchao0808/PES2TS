"""V5.4.1 WP4b — run identity, RNG/digest and manifest helpers (route E).

Scope (plan §7.3 / §7.4 / §12.2): everything the NEW unified trainer needs
to build an auditable, resumable identity WITHOUT touching any legacy
V5.3-MP / R3 file.  Nothing here reads data, builds features, or touches
dev/confirm — it is pure payload machinery:

* :func:`sha256_json` — canonical JSON SHA (identity fields);
* :func:`canonical_digest` — a deterministic content digest over an
  arbitrary payload (tensors / ndarrays / RNG states included) — the
  ``content_digest_sha256`` required by §7.3;
* :data:`RUN_OUTCOMES_V541` + :func:`run_outcome_for_exception` — the
  INDEPENDENT V541 outcome vocabulary (§7.4).  The legacy run-ledger
  status machine is deliberately NOT modelled here (旧 ledger 不改);
* :func:`rng_state_snapshot` / :func:`rng_state_restore` — python /
  numpy / torch (+ CUDA) RNG capture for bitwise resume;
* :func:`git_state` / :func:`env_state` — provenance of the checkout and
  the interpreter/GPU stack;
* :func:`append_manifest` — APPEND-only ``manifest.jsonl`` (§7.3:
  manifest 追加而非覆盖).

Route E discipline: this module only ADDS V541 files; the legacy trainer
is imported read-only (never patched, never edited).
"""

from __future__ import annotations

import hashlib
import json
import platform
import random
import socket
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np
import torch

__all__ = [
    "RUN_OUTCOMES_V541",
    "V541IdentityError",
    "V541RunBudgetError",
    "append_manifest",
    "canonical_digest",
    "content_digest_of",
    "env_state",
    "git_state",
    "positions_sha256",
    "read_manifest",
    "rng_state_restore",
    "rng_state_snapshot",
    "run_outcome_for_exception",
    "sha256_json",
    "sha256_text",
]

#: §7.4 frozen outcome vocabulary — INDEPENDENT of the legacy ledger
#: status machine (``planned/running/done/failed/killed/skipped/
#: interrupted``), which stays untouched.
RUN_OUTCOMES_V541: tuple = (
    "done",
    "invalid_run",
    "scientific_failure",
    "resource_failure",
    "budget_censored",
    "gate_skipped",
)


class V541IdentityError(RuntimeError):
    """Typed refusal of the V541 identity/digest/manifest machinery."""

    def __init__(self, code: str, detail: str = "") -> None:
        super().__init__(f"{code}: {detail}" if detail else code)
        self.code = str(code)


class V541RunBudgetError(RuntimeError):
    """Runner-raised budget censor (§7.4 ``budget_censored``).

    Carries ``outcome`` so :func:`run_outcome_for_exception` maps it
    without importing the trainer.
    """

    outcome = "budget_censored"


# ---------------------------------------------------------------------------
# canonical SHAs
# ---------------------------------------------------------------------------


def sha256_text(text: str) -> str:
    """sha256 hex of a UTF-8 string."""
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()


def sha256_json(obj: Any) -> str:
    """Canonical JSON SHA: sorted keys, no whitespace, UTF-8.

    Non-JSON scalars fall back to ``str()`` so the function never raises
    on an exotic config value (the fallback is deterministic).
    """
    blob = json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def positions_sha256(positions: Sequence[int]) -> str:
    """sha256 over the canonical little-endian int64 bytes of a position list."""
    arr = np.asarray([int(p) for p in positions], dtype="<i8")
    return hashlib.sha256(arr.tobytes()).hexdigest()


# ---------------------------------------------------------------------------
# content digest (checkpoint §7.3 "checkpoint content digest")
# ---------------------------------------------------------------------------


def _feed_tensor(h: "hashlib._Hash", tensor: torch.Tensor) -> None:
    view = tensor.detach().cpu().contiguous().reshape(-1)
    try:
        raw = view.numpy().tobytes()
    except (TypeError, RuntimeError):        # bfloat16 / float8 / etc.
        raw = view.view(torch.uint8).numpy().tobytes()
    h.update(b"T|")
    h.update(str(tensor.dtype).encode("utf-8"))
    h.update(b"|")
    h.update(",".join(str(int(d)) for d in tensor.shape).encode("utf-8"))
    h.update(b"|")
    h.update(str(len(raw)).encode("utf-8"))
    h.update(b"|")
    h.update(raw)


def _feed(h: "hashlib._Hash", obj: Any) -> None:
    """Recursively fold ``obj`` into the digest (deterministic ordering)."""
    if obj is None:
        h.update(b"null;")
        return
    if isinstance(obj, bool):
        h.update(b"true;" if obj else b"false;")
        return
    if isinstance(obj, (int, np.integer)):
        h.update(b"i:" + str(int(obj)).encode("utf-8") + b";")
        return
    if isinstance(obj, (float, np.floating)):
        h.update(b"f:" + repr(float(obj)).encode("utf-8") + b";")
        return
    if isinstance(obj, str):
        raw = obj.encode("utf-8")
        h.update(b"s:" + str(len(raw)).encode("utf-8") + b":" + raw + b";")
        return
    if isinstance(obj, (bytes, bytearray, memoryview)):
        raw = bytes(obj)
        h.update(b"y:" + str(len(raw)).encode("utf-8") + b":" + raw + b";")
        return
    if isinstance(obj, torch.Tensor):
        _feed_tensor(h, obj)
        h.update(b";")
        return
    if isinstance(obj, np.ndarray):
        arr = np.ascontiguousarray(obj)
        raw = arr.tobytes()
        h.update(b"a:" + str(arr.dtype).encode("utf-8") + b"|"
                 + ",".join(str(int(d)) for d in arr.shape).encode("utf-8")
                 + b"|" + str(len(raw)).encode("utf-8") + b"|" + raw + b";")
        return
    if isinstance(obj, Mapping):
        h.update(b"{")
        for key in sorted(obj.keys(), key=lambda k: (type(k).__name__, repr(k))):
            _feed(h, key)
            h.update(b"=>")
            _feed(h, obj[key])
            h.update(b",")
        h.update(b"}")
        return
    if isinstance(obj, (list, tuple)):
        tag = b"[" if isinstance(obj, list) else b"("
        h.update(tag + str(len(obj)).encode("utf-8") + b"|")
        for item in obj:
            _feed(h, item)
            h.update(b",")
        h.update(b"]" if isinstance(obj, list) else b")")
        return
    if isinstance(obj, (set, frozenset)):
        h.update(b"st:" + str(len(obj)).encode("utf-8") + b"|")
        for item in sorted(obj, key=repr):
            _feed(h, item)
            h.update(b",")
        h.update(b";")
        return
    h.update(b"o:" + type(obj).__name__.encode("utf-8") + b":"
             + repr(obj).encode("utf-8") + b";")


def content_digest_of(payload: Mapping[str, Any]) -> str:
    """Recompute a checkpoint's ``content_digest_sha256`` (self excluded).

    The digest key never participates in its own hash, so verification is
    ``content_digest_of(loaded_payload) == loaded_payload[...]``.
    """
    return canonical_digest(
        {k: v for k, v in payload.items()
         if k != "content_digest_sha256"})


def canonical_digest(obj: Any) -> str:
    """Deterministic sha256 content digest over an arbitrary payload.

    Handles the checkpoint payload shape end-to-end: nested dicts/lists,
    torch tensors (raw bytes + dtype + shape), numpy arrays (RNG states),
    and plain scalars.  Used for ``content_digest_sha256`` (§7.3) — it is
    computed over the payload BEFORE the digest key is inserted, so the
    digest never depends on itself.
    """
    h = hashlib.sha256()
    _feed(h, obj)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# RNG state (§7.3 "所有 RNG 状态")
# ---------------------------------------------------------------------------


def rng_state_snapshot(device: Optional[str] = None) -> Dict[str, Any]:
    """Capture python + numpy + torch (+ CUDA on a CUDA run) RNG state.

    ``device`` is the run device: CUDA state is captured only when the run
    actually lives on CUDA, so a CPU fixture run never initialises a CUDA
    context just to serialize a checkpoint.
    """
    state: Dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    on_cuda = str(device or "").startswith("cuda")
    if on_cuda and torch.cuda.is_available():
        try:
            state["torch_cuda"] = torch.cuda.get_rng_state_all()
        except RuntimeError:                # no device initialised yet
            state["torch_cuda"] = None
    return state


def rng_state_restore(state: Mapping[str, Any]) -> None:
    """Restore a :func:`rng_state_snapshot` (typed refusal on garbage)."""
    if not isinstance(state, Mapping):
        raise V541IdentityError(
            "RNG_STATE_INVALID",
            f"expected a mapping, got {type(state).__name__}")
    missing = [k for k in ("python", "numpy", "torch") if k not in state]
    if missing:
        raise V541IdentityError(
            "RNG_STATE_INVALID", f"missing RNG entries: {missing}")
    try:
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        torch_state = state["torch"]
        if not isinstance(torch_state, torch.Tensor):
            raise V541IdentityError(
                "RNG_STATE_INVALID",
                "torch RNG state must be a ByteTensor")
        torch.set_rng_state(torch_state.cpu().to(torch.uint8))
        cuda_state = state.get("torch_cuda")
        if cuda_state is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(cuda_state)
    except V541IdentityError:
        raise
    except (TypeError, ValueError, RuntimeError) as exc:
        raise V541IdentityError(
            "RNG_STATE_INVALID", f"cannot restore RNG state: {exc}") from exc


# ---------------------------------------------------------------------------
# git / env provenance (§12.2 "git_commit, Python/Torch/CUDA")
# ---------------------------------------------------------------------------


def _find_git_dir(start: Optional[Path] = None) -> Optional[Path]:
    """Locate the checkout's ``.git`` by walking up from ``start``.

    Deliberately pure file I/O — this module must never import
    ``subprocess``/``importlib`` (ground-truth guard DYNAMIC_EXEC_RISK), so
    the commit is read straight from ``.git/HEAD`` and its ref instead of
    by spawning ``git``.
    """
    cur = (start or Path(__file__).resolve().parent).resolve()
    for candidate in (cur, *cur.parents):
        dot = candidate / ".git"
        if dot.is_dir():
            return dot
        if dot.is_file():
            try:
                text = dot.read_text(encoding="utf-8", errors="replace")
            except OSError:
                return None
            if text.startswith("gitdir:"):
                target = Path(text.split(":", 1)[1].strip())
                if not target.is_absolute():
                    target = candidate / target
                return target
    return None


def _read_head_sha(gitdir: Optional[Path]) -> Optional[str]:
    """Commit SHA from ``HEAD`` (symbolic ref → ref file → packed-refs)."""
    if gitdir is None:
        return None
    head = gitdir / "HEAD"
    try:
        text = head.read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        return None
    if not text:
        return None
    ref = text[4:].strip() if text.startswith("ref:") else text
    ref_file = gitdir / ref
    try:
        if ref_file.is_file():
            value = ref_file.read_text(
                encoding="utf-8", errors="replace").strip().split()
            if value:
                return value[0]
        packed = gitdir / "packed-refs"
        if packed.is_file():
            for line in packed.read_text(
                    encoding="utf-8", errors="replace").splitlines():
                if line.startswith("#") or line.startswith("^"):
                    continue
                parts = line.split()
                if len(parts) == 2 and parts[1] == ref:
                    return parts[0]
    except OSError:
        return None
    # detached HEAD stores the sha directly
    if len(ref) == 40 and all(c in "0123456789abcdef" for c in ref):
        return ref
    return None


def git_state(*, cwd: Optional[str] = None) -> Dict[str, Any]:
    """Checkout identity (fail-open: a missing git never breaks a run).

    ``commit`` is read from the ``.git`` files; ``dirty`` is intentionally
    NOT probed (it would need a ``git status`` child process, which the
    ground-truth guard forbids in library code) and is reported as ``None``.
    """
    start = Path(cwd).resolve() if cwd else None
    commit = _read_head_sha(_find_git_dir(start))
    return {
        "commit": commit,
        "dirty": None,
        "source": ".git/HEAD (+ ref / packed-refs); "
                  "dirty not probed (no subprocess in library code)",
    }


def env_state() -> Dict[str, Any]:
    """Interpreter + framework + host stack recorded with every run."""
    cuda_version = None
    cudnn_version = None
    try:
        cuda_version = torch.version.cuda
        if torch.cuda.is_available():
            cudnn_version = torch.backends.cudnn.version()
    except Exception:                        # pragma: no cover - defensive
        pass
    return {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "torch": str(torch.__version__),
        "numpy": str(np.__version__),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_version": cuda_version,
        "cudnn_version": cudnn_version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "hostname": socket.gethostname(),
    }


# ---------------------------------------------------------------------------
# manifest (§7.3: APPEND, never overwrite)
# ---------------------------------------------------------------------------


def append_manifest(path: Any, record: Mapping[str, Any]) -> Dict[str, Any]:
    """Append ONE JSON line to ``manifest.jsonl`` and return it.

    Opened in ``"a"`` mode only — an existing file is never truncated or
    rewritten, so every attempt/leg of a run keeps its own line (§12.2:
    所有 attempt 追加写入总账, 不得覆盖).
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(dict(record), sort_keys=True, default=str)
    with p.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
    return json.loads(line)


def read_manifest(path: Any) -> list:
    """Read every JSON line of ``manifest.jsonl`` (tests / audits)."""
    p = Path(path)
    if not p.is_file():
        return []
    out = []
    for raw in p.read_text(encoding="utf-8").splitlines():
        if raw.strip():
            out.append(json.loads(raw))
    return out


# ---------------------------------------------------------------------------
# run outcome vocabulary (§7.4) — independent of the legacy ledger
# ---------------------------------------------------------------------------


def run_outcome_for_exception(exc: BaseException) -> str:
    """Map an exception to the §7.4 V541 outcome vocabulary.

    * an explicit ``outcome`` attribute (e.g.
      :class:`V541RunBudgetError`) wins when it is in the vocabulary;
    * ``MemoryError`` / ``torch.cuda.OutOfMemoryError`` →
      ``resource_failure``;
    * everything else — protocol/implementation problems, typed refusals,
      plain runtime errors — → ``invalid_run``.

    ``scientific_failure`` is never produced here (it comes from a
    completed run whose H2 guard failed) and ``gate_skipped`` is a
    runner-side scheduling decision.
    """
    tagged = getattr(exc, "outcome", None)
    if isinstance(tagged, str) and tagged in RUN_OUTCOMES_V541:
        return tagged
    if isinstance(exc, MemoryError):
        return "resource_failure"
    if type(exc).__name__ in ("OutOfMemoryError", "CuDNNError"):
        return "resource_failure"
    return "invalid_run"
