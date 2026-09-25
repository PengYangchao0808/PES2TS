"""Unit tests for reaction-ID normalization."""

from __future__ import annotations

import pytest

from pes2ts_core.g0.ids import (
    REACTION_ID_DIGITS,
    REACTION_ID_PREFIX,
    BadReactionId,
    normalize_reaction_id,
)


def test_bad_reaction_id_is_a_value_error() -> None:
    assert issubclass(BadReactionId, ValueError)


def test_string_forms_normalize_to_the_same_canonical_id() -> None:
    assert normalize_reaction_id("1") == "RXN_0000000001"
    assert normalize_reaction_id("123") == "RXN_0000000123"
    assert (
        normalize_reaction_id("RXN_0000000123")
        == normalize_reaction_id("0000000123")
        == normalize_reaction_id("123")
    )


def test_int_input_normalizes_like_its_digits() -> None:
    assert normalize_reaction_id(123) == "RXN_0000000123"
    assert normalize_reaction_id(1) == "RXN_0000000001"
    assert normalize_reaction_id(0) == "RXN_0000000000"


def test_integral_float_is_accepted_like_the_equivalent_int() -> None:
    assert normalize_reaction_id(123.0) == "RXN_0000000123"


def test_surrounding_whitespace_is_tolerated() -> None:
    assert normalize_reaction_id("  123  ") == "RXN_0000000123"
    assert normalize_reaction_id("\tRXN_0000000123\n") == "RXN_0000000123"


def test_digit_strings_longer_than_ten_digits_are_never_truncated() -> None:
    assert REACTION_ID_DIGITS == 10
    assert REACTION_ID_PREFIX == "RXN_"
    assert normalize_reaction_id("12345678901") == "RXN_12345678901"
    assert normalize_reaction_id("RXN_12345678901234") == "RXN_12345678901234"
    assert normalize_reaction_id(12345678901234) == "RXN_12345678901234"


def test_lowercase_prefix_is_not_accepted() -> None:
    with pytest.raises(BadReactionId):
        normalize_reaction_id("rxn_123")


@pytest.mark.parametrize(
    "bad_value",
    [
        "",
        "   ",
        "\n",
        "abc",
        "RXN_",
        "RXN_abc",
        "rxn_1",
        "RXN_1.5",
        "1.5",
        "12a",
        "1 2",
        "RXN 123",
        "-1",
        "-123",
        "RXN_-1",
        "\u0661\u0662\u0663",  # non-ASCII digits are still not reaction ids
        None,
        True,
        False,
        -1,
        -123,
        1.5,
        float("nan"),
        float("inf"),
        [],
        {},
        object(),
    ],
)
def test_invalid_values_raise_bad_reaction_id(bad_value: object) -> None:
    with pytest.raises(BadReactionId) as excinfo:
        normalize_reaction_id(bad_value)
    assert str(excinfo.value)
