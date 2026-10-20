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
    "E0_LANE_ADDITIONS",
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

#: E0 additions on top of the lane write_scopes (interface-freeze scope).
#: Ownership mapping per PARALLEL_EXECUTION_PLAN.md §5.1 (interface-freeze
#: allowlist line) and the §5 WP table: WP-2/WP-6 deliver pesn features /
#: models / contracts v3 + tests/pesn/test_r3_* to W-FEAT; WP-4 delivers pesn
#: offline/training v3 + scripts/campaigns/{r3_*,run_v5_1_r3} to W-LOSS;
#: W-DATA gets no pes2ts_core E0 addition (WP-1 writes scripts/campaigns via
#: its own scope).  Hardcoded here on purpose -- parallel_config stays
#: lane-scope-only (do NOT move this mapping into parallel_config).
E0_LANE_ADDITIONS = {
    "W-DATA": [],
    "W-FEAT": [
        "pes2ts_core/pesn/features/**/*_v3.py",
        "pes2ts_core/pesn/models/**/*_v3.py",
        "pes2ts_core/pesn/contracts/**/*_v3.py",
        "tests/pesn/test_r3_*.py",
    ],
    "W-LOSS": [
        "pes2ts_core/pesn/offline/training/**/*_v3.py",
        "scripts/campaigns/r3_*.py",
        "scripts/campaigns/run_v5_1_r3.py",
    ],
}
#: global (supervisor) union of every lane's E0 addition
E0_ADDITIONS = sorted({p for pats in E0_LANE_ADDITIONS.values() for p in pats})

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


def _match_components(path_segs: List[str], pat_segs: List[str]) -> bool:
    """Component-wise glob: ``*``/``?`` match ONE segment (never cross ``/``);
    ``**`` matches zero or more path segments (ORA-2 blocker 6)."""
    if not pat_segs:
        return not path_segs
    head, rest = pat_segs[0], pat_segs[1:]
    if head == "**":
        for i in range(len(path_segs) + 1):
            if _match_components(path_segs[i:], rest):
                return True
        return False
    if not path_segs:
        return False
    if fnmatch.fnmatchcase(path_segs[0], head):
        return _match_components(path_segs[1:], rest)
    return False


def path_allowed(path: str, patterns: Sequence[str]) -> bool:
    """True when ``path`` matches at least one allowlist pattern.

    Matching is component-wise (slash-aware): ``tests/pesn/test_r3_*.py``
    does NOT match ``tests/pesn/sub/test_r3_x.py`` because ``*`` stays inside
    one path component; ``**`` is the only wildcard that crosses ``/``.
    """
    normalized = str(path).strip().replace("\\", "/")
    path_segs = [s for s in normalized.split("/") if s and s != "."]
    for pattern in patterns:
        pat_segs = [s for s in str(pattern).split("/") if s and s != "."]
        if pat_segs and _match_components(path_segs, pat_segs):
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
    lanes: Dict[str, Dict[str, Any]] = {}
    for lane, workflow in sorted(config["workflows"].items()):
        scopes = workflow.get("write_scopes")
        if not isinstance(scopes, list) or not scopes:
            raise AllowlistError(
                "CONFIG_CORRUPT: workflow {!r} has no write_scopes".format(lane)
            )
        lane_patterns: List[str] = []
        for scope in scopes:
            if not isinstance(scope, str) or not scope:
                raise AllowlistError(
                    "CONFIG_CORRUPT: workflow {!r} has an invalid write scope {!r}".format(
                        lane, scope
                    )
                )
            patterns.append(scope)
            lane_patterns.append(scope)
        # lane-scoped E0 additions (see E0_LANE_ADDITIONS comment, §5.1)
        lane_patterns.extend(E0_LANE_ADDITIONS.get(lane, []))
        lanes[lane] = {"write_allowlist": sorted(set(lane_patterns))}
    shared = config.get("shared_files")
    if not isinstance(shared, dict) or not shared:
        raise AllowlistError("CONFIG_CORRUPT: missing shared_files object")
    shared_files = sorted(shared.keys())
    for lane in lanes:
        lanes[lane]["shared_files"] = list(shared_files)

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
        # supervisor/global union (lane sets are authoritative per --lane)
        "write_allowlist": sorted(set(patterns) | set(E0_ADDITIONS)),
        "lanes": lanes,
        "e0_additions": list(E0_ADDITIONS),
        "e0_lane_additions": {k: list(v) for k, v in E0_LANE_ADDITIONS.items()},
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
    lane: Optional[str] = None,
) -> Dict[str, Any]:
    """Diff ``base..branch``; every changed file must be allowlisted.

    ``lane`` (e.g. ``"W-LOSS"``) validates against that lane's scoped set
    (write_scopes + its E0 additions + shared files); without a lane the
    global union is used (supervisor/merge integration view).  Returns
    ``{"ok": bool, "violations": [...], "files": [...]}`` -- the caller
    (or the CLI) turns ``ok=False`` into exit code 1.
    """
    root = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parents[3]
    if allowlist is None:
        allowlist_path = (
            Path(allowlist_path) if allowlist_path is not None else root / DEFAULT_ALLOWLIST_PATH
        )
        allowlist = _load_allowlist(allowlist_path)

    if lane is not None:
        lanes = allowlist.get("lanes")
        if not isinstance(lanes, dict) or not lanes:
            raise AllowlistError(
                "ALLOWLIST_CORRUPT: {} has no per-lane sets (regenerate with this "
                "tool version)".format(allowlist_path or "<payload>")
            )
        if lane not in lanes:
            raise AllowlistError(
                "LANE_UNKNOWN: {!r} not in {}".format(lane, ", ".join(sorted(lanes)))
            )
        lane_entry = lanes[lane]
        if not isinstance(lane_entry, dict) or not isinstance(
            lane_entry.get("write_allowlist"), list
        ):
            raise AllowlistError(
                "ALLOWLIST_CORRUPT: lane {!r} has no write_allowlist".format(lane)
            )
        patterns = list(lane_entry["write_allowlist"])
        shared = list(lane_entry.get("shared_files", allowlist.get("shared_files", [])))
    else:
        patterns = list(allowlist.get("write_allowlist", []))
        shared = list(allowlist.get("shared_files", []))
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
        "lane": lane,
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
    parser.add_argument(
        "--lane",
        default=None,
        help="check against this lane's scoped allowlist (e.g. W-LOSS); "
        "omit for the supervisor/global union",
    )
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
                lane=args.lane,
            )
            if result["ok"]:
                print(
                    "MERGE OK: {} file(s) in {}..{}{} within the allowlist".format(
                        len(result["files"]),
                        args.base,
                        args.branch,
                        " for lane {}".format(args.lane) if args.lane else "",
                    )
                )
                return 0
            for violation in result["violations"]:
                sys.stderr.write("WRITE_SCOPE_VIOLATION: {}\n".format(violation))
            sys.stderr.write(
                "MERGE REFUSED: {} violation(s) in {}..{}{}\n".format(
                    len(result["violations"]),
                    args.base,
                    args.branch,
                    " for lane {}".format(args.lane) if args.lane else "",
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
