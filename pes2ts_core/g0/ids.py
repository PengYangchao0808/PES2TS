"""Reaction-ID normalization for the G0 stage.

All G0 artifacts key reactions by the canonical ``RXN_<10 digits>`` identifier
used by the Reaction-QM dataset. Normalizing at the boundary means every
downstream stage compares identical strings regardless of whether the source
stored ``123``, ``0000000123``, or ``RXN_0000000123``.
"""

from __future__ import annotations

import math
from typing import Final

#: Canonical prefix of a normalized reaction ID (case-sensitive).
REACTION_ID_PREFIX: Final[str] = "RXN_"

#: Number of digits the numeric part is left-padded to.
REACTION_ID_DIGITS: Final[int] = 10


class BadReactionId(ValueError):
    """Raised when a value cannot be normalized into a reaction ID."""


def _canonical(digits: str) -> str:
    """Return ``RXN_`` plus *digits* left-padded to 10 characters.

    Python's :meth:`str.zfill` leaves strings longer than 10 characters
    untouched, which is exactly the contract: never truncate a longer digit
    string.
    """
    return f"{REACTION_ID_PREFIX}{digits.zfill(REACTION_ID_DIGITS)}"


def normalize_reaction_id(value: object) -> str:
    """Normalize *value* to the canonical ``RXN_<10-digit>`` form.

    Accepted inputs are a non-negative :class:`int` (or a finite integral
    :class:`float`) and :class:`str` forms ``"123"``, ``"0000000123"``, and
    ``"RXN_0000000123"`` (the ``RXN_`` prefix is case-sensitive). Surrounding
    whitespace is tolerated. The numeric part is zero-padded on the left to 10
    digits; a numeric part longer than 10 digits is preserved verbatim and
    never truncated.

    Raises
    ------
    BadReactionId
        For empty, negative, non-digit, or non-numeric values, including a
        lowercase ``rxn_`` prefix, ``None``, booleans, and floats with a
        fractional part.
    """
    if isinstance(value, bool):
        msg = f"Invalid reaction id {value!r}: booleans are not reaction ids"
        raise BadReactionId(msg)
    if isinstance(value, int):
        if value < 0:
            msg = f"Invalid reaction id {value!r}: negative values are not allowed"
            raise BadReactionId(msg)
        return _canonical(str(value))
    if isinstance(value, float):
        if not math.isfinite(value):
            msg = f"Invalid reaction id {value!r}: value is not finite"
            raise BadReactionId(msg)
        if not value.is_integer():
            msg = f"Invalid reaction id {value!r}: fractional values are not allowed"
            raise BadReactionId(msg)
        if value < 0:
            msg = f"Invalid reaction id {value!r}: negative values are not allowed"
            raise BadReactionId(msg)
        return _canonical(str(int(value)))
    if not isinstance(value, str):
        msg = f"Invalid reaction id {value!r}: expected an int or a digit string"
        raise BadReactionId(msg)

    text = value.strip()
    if not text:
        msg = f"Invalid reaction id {value!r}: value is empty"
        raise BadReactionId(msg)
    digits = text.removeprefix(REACTION_ID_PREFIX)
    if not digits or not digits.isascii() or not digits.isdecimal():
        msg = (
            f"Invalid reaction id {value!r}: expected digits, optionally "
            f"prefixed with {REACTION_ID_PREFIX!r}"
        )
        raise BadReactionId(msg)
    return _canonical(digits)


__all__ = [
    "REACTION_ID_DIGITS",
    "REACTION_ID_PREFIX",
    "BadReactionId",
    "normalize_reaction_id",
]
