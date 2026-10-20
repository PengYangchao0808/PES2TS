"""E0 per-file write allowlist + frozen-file denylist (WP-L2, supervisor tool).

``generate_allowlist(out.json)`` builds the machine-checkable write policy:

* allowlist = union of every workflow ``write_scopes`` entry in
  ``docs/agent_runs/state/parallel_config.json`` + the two supervisor-owned
  ``shared_files`` + the E0 additions (``pes2ts_core/pesn/**/*_v3.py``,
  ``tests/pesn/test_r3_*.py``, ``scripts/campaigns/r3_*.py``,
  ``scripts/campaigns/run_v5_1_r3.py``);
* denylist = the frozen E0 protocol/training files pinned by sha256
  (``key_atom_protocols.py``, ``key_atom_bc.py``, ``key_atom_teacher.py``) --
  hashing a missing/unreadable frozen file refuses (fail-closed); a *later*
  re-check (``check_frozen``) reports any content drift as a
  ``frozen-file-modified`` alarm.

``check_merge(branch, base)`` runs ``git diff --name-only base..branch`` and
requires every changed file to match the allowlist (shared files allowed);
violations are listed and the CLI exits 1.  A missing/corrupt allowlist or
config, an unknown revision or a git error also exit non-zero -- there is no
warn-and-pass path.
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

__all__ = [
    "DEFAULT_ALLOWLIST_PATH",
    "DEFAULT_CONFIG_PATH",
    "E0_ADDITIONS",
    "FROZEN_FILES",
    "AllowlistError",
    "check_frozen",
    "check_merge",
    "generate_allowlist",
    "main",
    "path_allowed",
]

DEFAULT_CONFIG_PATH = "docs/agent_runs/state/parallel_config.json"
DEFAULT_ALLOWLIST_PATH = "docs/agent_runs/state/e0_write_allowlist.json"

#: E0 additions on top of the lane write_scopes (interface-freeze scope)
E0_ADDITIONS = [
    "pes2ts_core/pesn/**/*_v3.py",
    "tests/pesn/test_r3_*.py",
    "scripts/campaigns/r3_*.py",
    "scripts/campaigns/run_v5_1_r3.py",
]

#: frozen files pinned at allowlist generation; any later hash drift = alarm
FROZEN_FILES = [
    "pes2ts_core/pesn/offline/evaluation/key_atom_protocols.py",
    "pes2ts_core/pesn/offline/training/key_atom_bc.py",
    "pes2ts_core/pesn/offline/data/key_atom_teacher.py",
]


class AllowlistError(Exception):
    """Missing/corrupt policy input -- always a refusal, never a warning."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(str(path), "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_config(config_path: Path) -> Dict[str, Any]:
    if not config_path.is_file():
        raise AllowlistError(
            "CONFIG_MISSING: parallel config not found: {}".format(config_path)
        )
    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AllowlistError(
            "CONFIG_CORRUPT: {}: {}".format(config_path, exc)
        ) from exc
    if not isinstance(data, dict) or not isinstance(data.get("workflows"), dict):
        raise AllowlistError(
            "CONFIG_CORRUPT: {} has no workflows object".format(config_path)
        )
    return data


def _load_allowlist(allowlist_path: Path) -> Dict[str, Any]:
    if not allowlist_path.is_file():
        raise AllowlistError(
            "ALLOWLIST_MISSING: generate it first (generate_allowlist): {}".format(
                allowlist_path
            )
        )
    try:
        data = json.loads(allowlist_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AllowlistError(
            "ALLOWLIST_CORRUPT: {}: {}".format(allowlist_path, exc)
        ) from exc
    if (
        not isinstance(data, dict)
        or not isinstance(data.get("write_allowlist"), list)
        or not isinstance(data.get("frozen_files"), dict)
    ):
        raise AllowlistError(
            "ALLOWLIST_CORRUPT: {} lacks write_allowlist/frozen_files".format(
                allowlist_path
            )
        )
    return data


def path_allowed(path: str, patterns: Sequence[str]) -> bool:
    """True when ``path`` matches at least one allowlist pattern.

    ``**`` matches across directory separators (via ``fnmatch`` semantics).
    """
    normalized = path.strip().lstrip("./")
    for pattern in patterns:
        if fnmatch.fnmatch(normalized, pattern.replace("**", "*")):
            return True
    return False


def generate_allowlist(
    out_json: Any,
    config_path: Optional[Any] = None,
    repo_root: Optional[Any] = None,
) -> Dict[str, Any]:
    """Write the E0 write allowlist + frozen-file sha256 denylist to ``out_json``."""
    root = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parents[3]
    config_path = Path(config_path) if config_path is not None else root / DEFAULT_CONFIG_PATH
    config = _load_config(config_path)

    patterns: List[str] = []
    for lane, workflow in sorted(config["workflows"].items()):
        scopes = workflow.get("write_scopes")
        if not isinstance(scopes, list) or not scopes:
            raise AllowlistError(
                "CONFIG_CORRUPT: workflow {!r} has no write_scopes".format(lane)
            )
        for scope in scopes:
            if not isinstance(scope, str) or not scope:
                raise AllowlistError(
                    "CONFIG_CORRUPT: workflow {!r} has an invalid write scope {!r}".format(
                        lane, scope
                    )
                )
            patterns.append(scope)
    shared = config.get("shared_files")
    if not isinstance(shared, dict) or not shared:
        raise AllowlistError("CONFIG_CORRUPT: missing shared_files object")
    shared_files = sorted(shared.keys())

    frozen: Dict[str, Dict[str, str]] = {}
    for rel in FROZEN_FILES:
        path = root / rel
        if not path.is_file():
            raise AllowlistError(
                "FROZEN_FILE_MISSING: cannot pin {}: {}".format(rel, path)
            )
        try:
            digest = _sha256_file(path)
        except OSError as exc:
            raise AllowlistError("FROZEN_FILE_UNREADABLE: {}: {}".format(path, exc)) from exc
        frozen[rel] = {"sha256": digest, "prefix": digest[:16]}

    payload: Dict[str, Any] = {
        "kind": "e0_write_allowlist",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config": str(config_path),
        "write_allowlist": sorted(set(patterns) | set(E0_ADDITIONS)),
        "e0_additions": list(E0_ADDITIONS),
        "shared_files": shared_files,
        "never_write_all_lanes": config.get("never_write_all_lanes", []),
        "frozen_files": frozen,
    }

    out = Path(out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(out.parent), prefix=out.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, str(out))
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return payload


def _git_names(repo_root: Path, revision: str) -> List[str]:
    proc = subprocess.run(
        ["git", "diff", "--name-only", revision],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        timeout=120,
    )
    if proc.returncode != 0:
        raise AllowlistError(
            "GIT_DIFF_FAILED: git diff --name-only {} exited {}: {}".format(
                revision, proc.returncode, proc.stderr.strip()
            )
        )
    return [ln.strip() for ln in proc.stdout.splitlines() if ln.strip()]


def check_merge(
    branch: str,
    base: str,
    allowlist_path: Optional[Any] = None,
    repo_root: Optional[Any] = None,
    allowlist: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Diff ``base..branch``; every changed file must be allowlisted.

    Returns ``{"ok": bool, "violations": [...], "files": [...]}`` -- the caller
    (or the CLI) turns ``ok=False`` into exit code 1.
    """
    root = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parents[3]
    if allowlist is None:
        allowlist_path = (
            Path(allowlist_path) if allowlist_path is not None else root / DEFAULT_ALLOWLIST_PATH
        )
        allowlist = _load_allowlist(allowlist_path)

    patterns: List[str] = list(allowlist.get("write_allowlist", []))
    shared: List[str] = list(allowlist.get("shared_files", []))
    never: List[str] = list(allowlist.get("never_write_all_lanes", []))
    allowed = patterns + shared

    files = _git_names(root, "{}..{}".format(base, branch))
    violations: List[str] = []
    for name in files:
        if never and path_allowed(name, never):
            violations.append("{} (never_write_all_lanes)".format(name))
        elif not path_allowed(name, allowed):
            violations.append(name)
    return {
        "ok": not violations,
        "base": base,
        "branch": branch,
        "files": files,
        "violations": violations,
    }


def check_frozen(
    allowlist_path: Optional[Any] = None,
    repo_root: Optional[Any] = None,
) -> Dict[str, Any]:
    """Re-hash the frozen files; any drift is a frozen-file-modified alarm."""
    root = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parents[3]
    allowlist_path = (
        Path(allowlist_path) if allowlist_path is not None else root / DEFAULT_ALLOWLIST_PATH
    )
    allowlist = _load_allowlist(allowlist_path)
    frozen = allowlist.get("frozen_files")
    if not isinstance(frozen, dict) or not frozen:
        raise AllowlistError(
            "ALLOWLIST_CORRUPT: {} has no frozen_files denylist".format(allowlist_path)
        )
    mismatches: List[Dict[str, Any]] = []
    missing: List[str] = []
    for rel, expected in sorted(frozen.items()):
        path = root / rel
        if not path.is_file():
            missing.append(rel)
            continue
        actual = _sha256_file(path)
        want = expected.get("sha256") if isinstance(expected, dict) else None
        if actual != want:
            mismatches.append({"file": rel, "expected": want, "actual": actual})
    return {
        "ok": not mismatches and not missing,
        "mismatches": mismatches,
        "missing": missing,
        "frozen_files": sorted(frozen.keys()),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="allowlist",
        description="E0 write allowlist generation + merge/frozen-file checks (fail-closed)",
    )
    parser.add_argument(
        "--root",
        default=str(Path(__file__).resolve().parents[3]),
        help="repository / campaign root",
    )
    parser.add_argument("--config", default=None, help="parallel_config.json path")
    parser.add_argument("--allowlist", default=None, help="allowlist json path")
    sub = parser.add_subparsers(dest="cmd", required=True)

    gen = sub.add_parser("generate", help="write the allowlist + frozen denylist json")
    gen.add_argument("--out", required=True, help="output json path")

    cm = sub.add_parser("check-merge", help="verify base..branch diff subset of allowlist")
    cm.add_argument("--base", required=True)
    cm.add_argument("--branch", required=True)

    sub.add_parser("check-frozen", help="re-hash frozen files; alarm on any drift")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    root = Path(args.root)
    config = Path(args.config) if args.config else None
    allowlist_path = Path(args.allowlist) if args.allowlist else None
    try:
        if args.cmd == "generate":
            payload = generate_allowlist(args.out, config_path=config, repo_root=root)
            print(
                json.dumps(
                    {
                        "out": args.out,
                        "patterns": len(payload["write_allowlist"]),
                        "frozen_files": sorted(payload["frozen_files"].keys()),
                    },
                    sort_keys=True,
                )
            )
            return 0
        if args.cmd == "check-merge":
            result = check_merge(
                args.branch,
                args.base,
                allowlist_path=allowlist_path,
                repo_root=root,
            )
            if result["ok"]:
                print(
                    "MERGE OK: {} file(s) in {}..{} are within the allowlist".format(
                        len(result["files"]), args.base, args.branch
                    )
                )
                return 0
            for violation in result["violations"]:
                sys.stderr.write("WRITE_SCOPE_VIOLATION: {}\n".format(violation))
            sys.stderr.write(
                "MERGE REFUSED: {} violation(s) in {}..{}\n".format(
                    len(result["violations"]), args.base, args.branch
                )
            )
            return 1
        if args.cmd == "check-frozen":
            result = check_frozen(allowlist_path=allowlist_path, repo_root=root)
            if result["ok"]:
                print(
                    "FROZEN OK: {} file(s) match their recorded sha256".format(
                        len(result["frozen_files"])
                    )
                )
                return 0
            for item in result["mismatches"]:
                sys.stderr.write(
                    "FROZEN_FILE_MODIFIED: {} expected {} actual {}\n".format(
                        item["file"], item["expected"], item["actual"]
                    )
                )
            for rel in result["missing"]:
                sys.stderr.write("FROZEN_FILE_MISSING: {}\n".format(rel))
            return 1
        return 2
    except AllowlistError as exc:
        sys.stderr.write("error: {}\n".format(exc))
        return 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
