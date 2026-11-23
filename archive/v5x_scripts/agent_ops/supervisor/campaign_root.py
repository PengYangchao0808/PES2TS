"""Campaign output-root assertion for runners (WP-L2 / ORA-2 blocker 7).

``parallel_config.driver_campaign_root_rule``: reused drivers
(``run_v5_1_sprint`` / ``r1_gradient_probe`` / ...) default their output root
to the SEALED directories (``outputs/pesn/v5_1/**``, ``outputs/pesn/v5_1_r1/**``);
the 30h campaign must explicitly point at ``outputs/pesn/30h_campaign/**`` and
the driver must ASSERT the refusal at startup.

:func:`assert_campaign_root` is that assertion:

* sealed prefixes ``outputs/pesn/v5_1`` and ``outputs/pesn/v5_1_r1`` (and
  anything below them) -> :class:`CampaignRootError` ``SEALED_ROOT_REFUSED``;
* anything outside ``outputs/pesn/30h_campaign`` -> ``OUTSIDE_CAMPAIGN_ROOT``;
* empty/non-path/absolute-escape garbage -> ``INVALID_ROOT``.

Integration into the actual drivers happens at E3 (out of scope for WP-L2);
runners call ``assert_campaign_root(path)`` before opening any output file.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional, Union

__all__ = [
    "CAMPAIGN_ROOT",
    "CampaignRootError",
    "assert_campaign_root",
    "main",
    "normalize_root",
]

CAMPAIGN_ROOT = "outputs/pesn/30h_campaign"
SEALED_PREFIXES = ("outputs/pesn/v5_1", "outputs/pesn/v5_1_r1")


class CampaignRootError(Exception):
    """Refused output root (sealed, outside the campaign, or invalid)."""


def normalize_root(path: Any, campaign_root: Optional[Union[str, Path]] = None) -> str:
    """Normalize ``path`` to a ``outputs/pesn/...``-style relative key.

    Absolute paths are anchored at their ``outputs/pesn`` component (drivers
    usually pass absolute roots); purely relative paths are joined with
    ``campaign_root`` when given (so ``assert_campaign_root("runs/x",
    campaign_root=...)`` works).
    """
    if path is None or not isinstance(path, (str, Path)):
        raise CampaignRootError(
            "INVALID_ROOT: output root must be a path string, got {!r}".format(path)
        )
    text = str(path).strip().replace("\\", "/")
    if not text:
        raise CampaignRootError("INVALID_ROOT: empty output root")
    parts = [p for p in text.split("/") if p and p != "."]
    if any(p == ".." for p in parts):
        raise CampaignRootError(
            "INVALID_ROOT: parent references are not allowed: {!r}".format(path)
        )
    if text.startswith("/") or (len(text) > 1 and text[1] == ":"):
        # absolute: anchor at the outputs/pesn component if present
        anchored = None
        for i in range(len(parts) - 1):
            if parts[i] == "outputs" and parts[i + 1] == "pesn":
                anchored = parts[i:]
                break
        if anchored is None:
            raise CampaignRootError(
                "OUTSIDE_CAMPAIGN_ROOT: absolute path {!r} contains no "
                "outputs/pesn component".format(path)
            )
        parts = anchored
    elif "outputs" in parts:
        parts = parts[parts.index("outputs"):]
    elif campaign_root is not None:
        base = str(campaign_root).strip().replace("\\", "/").rstrip("/")
        parts = [p for p in (base.split("/") + parts) if p and p != "."]
        # re-anchor if the joined path now contains outputs/pesn
        for i in range(len(parts) - 1):
            if parts[i] == "outputs" and parts[i + 1] == "pesn":
                parts = parts[i:]
                break
    if len(parts) < 2 or parts[0] != "outputs" or parts[1] != "pesn":
        raise CampaignRootError(
            "OUTSIDE_CAMPAIGN_ROOT: {!r} is not under outputs/pesn/** "
            "(expected {}/...)".format(path, CAMPAIGN_ROOT)
        )
    return "/".join(parts)


def assert_campaign_root(
    path: Any, campaign_root: Optional[Union[str, Path]] = None
) -> str:
    """Refuse sealed or non-campaign output roots; return the normalized key.

    Call at runner startup BEFORE any output file is opened (fail-closed:
    unknown/invalid input is always a refusal).
    """
    normalized = normalize_root(path, campaign_root=campaign_root)
    rel = normalized[len("outputs/pesn/"):]
    for sealed in SEALED_PREFIXES:
        sealed_rel = sealed[len("outputs/pesn/"):]
        if rel == sealed_rel or rel.startswith(sealed_rel + "/"):
            raise CampaignRootError(
                "SEALED_ROOT_REFUSED: {} is a sealed read-only prefix "
                "({}); the 30h campaign must write only under {}".format(
                    normalized, sealed, CAMPAIGN_ROOT
                )
            )
    if rel != CAMPAIGN_ROOT[len("outputs/pesn/"):] and not rel.startswith(
        CAMPAIGN_ROOT[len("outputs/pesn/"):] + "/"
    ):
        raise CampaignRootError(
            "OUTSIDE_CAMPAIGN_ROOT: {} is outside the campaign root {}".format(
                normalized, CAMPAIGN_ROOT
            )
        )
    return normalized


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="campaign_root",
        description="Assert an output root is inside outputs/pesn/30h_campaign "
        "(sealed v5_1/v5_1_r1 prefixes are refused)",
    )
    parser.add_argument("path", nargs="?", help="output root to assert")
    parser.add_argument(
        "--campaign-root", default=None, help="base for relative paths (optional)"
    )
    return parser


def main(argv: Any = None) -> int:
    args = _build_parser().parse_args(argv)
    if not args.path:
        _build_parser().print_help()
        return 2
    try:
        normalized = assert_campaign_root(args.path, campaign_root=args.campaign_root)
    except CampaignRootError as exc:
        sys.stderr.write("error: {}\n".format(exc))
        return 1
    print(json.dumps({"ok": True, "root": normalized}, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
