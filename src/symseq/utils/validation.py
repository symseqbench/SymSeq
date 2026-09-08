# SPDX-License-Identifier: MIT
# Copyright (c) 2025-present, symseq Contributors

"""
validation.py

Shared helpers for validating and normalizing scalar arguments.

Two levels are offered:

* Predicates (:func:`is_integer`, :func:`is_real`) answer a type question and
  return a bool. Use them where the caller needs to raise its own exception type
  or phrase its own message.
* Validators (:func:`validate_integer`, :func:`validate_boolean`,
  :func:`validate_unit_fraction`, :func:`validate_length_bounds`) raise
  ``TypeError`` for a wrong type and ``ValueError`` for an out-of-range value,
  and return the normalized Python scalar.

Booleans are rejected wherever a number is expected. ``bool`` subclasses ``int``,
so without an explicit check a flag passed by mistake silently becomes 0 or 1.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "is_integer",
    "is_real",
    "validate_boolean",
    "validate_integer",
    "validate_length_bounds",
    "validate_unit_fraction",
]


def is_integer(value: object) -> bool:
    """
    Check whether a value is a true integer.

    Parameters
    ----------
    value : object
        The value to check.

    Returns
    -------
    bool
        True if `value` is a Python or NumPy integer and not a Boolean.
    """
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def is_real(value: object) -> bool:
    """
    Check whether a value is a real number.

    Integers are accepted, since every integer is a valid real number.

    Parameters
    ----------
    value : object
        The value to check.

    Returns
    -------
    bool
        True if `value` is a Python or NumPy integer or float and not a Boolean.
    """
    return isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, (bool, np.bool_))


def validate_integer(value: object, name: str, *, minimum: int) -> int:
    """
    Validate and normalize an integer argument.

    Parameters
    ----------
    value : object
        The value to validate.
    name : str
        Argument name, used in the error messages.
    minimum : int
        Smallest accepted value.

    Returns
    -------
    int
        The value as a Python ``int``.

    Raises
    ------
    TypeError
        If `value` is not an integer.
    ValueError
        If `value` is smaller than `minimum`.
    """
    if not is_integer(value):
        raise TypeError(f"{name} must be an integer.")
    parsed = int(value)
    if parsed < minimum:
        qualifier = "non-negative" if minimum == 0 else f"at least {minimum}"
        raise ValueError(f"{name} must be {qualifier}.")
    return parsed


def validate_boolean(value: object, name: str) -> bool:
    """
    Validate and normalize a Boolean argument.

    Truthy non-Boolean values are rejected rather than coerced, so that an
    argument passed in the wrong position surfaces immediately.

    Parameters
    ----------
    value : object
        The value to validate.
    name : str
        Argument name, used in the error message.

    Returns
    -------
    bool
        The value as a Python ``bool``.

    Raises
    ------
    TypeError
        If `value` is not a Boolean.
    """
    if not isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a boolean.")
    return bool(value)


def validate_unit_fraction(value: object, name: str) -> float:
    """
    Validate and normalize a finite fraction in the closed unit interval.

    Parameters
    ----------
    value : object
        The value to validate.
    name : str
        Argument name, used in the error messages.

    Returns
    -------
    float
        The value as a Python ``float``.

    Raises
    ------
    TypeError
        If `value` is not a real number.
    ValueError
        If `value` is not finite or lies outside ``[0, 1]``.
    """
    if not is_real(value):
        raise TypeError(f"{name} must be a real number.")
    parsed = float(value)
    if not np.isfinite(parsed) or not 0.0 <= parsed <= 1.0:
        raise ValueError(f"{name} must lie in [0, 1].")
    return parsed


def validate_length_bounds(
    min_length: object,
    max_length: object,
    length_range: object,
) -> tuple[int, int]:
    """
    Validate and resolve effective sequence-length bounds.

    Parameters
    ----------
    min_length : object
        Lower bound, used when `length_range` is None.
    max_length : object
        Upper bound, used when `length_range` is None.
    length_range : object
        Pair ``(min, max)`` overriding `min_length` and `max_length`, or None.

    Returns
    -------
    tuple of int
        The resolved ``(minimum, maximum)`` bounds.

    Raises
    ------
    ValueError
        If `length_range` is not a pair, if either bound is not an integer, if a
        bound is negative, or if the minimum exceeds the maximum.
    """
    if length_range is not None:
        if not isinstance(length_range, (list, tuple)) or len(length_range) != 2:
            raise ValueError("length_range must contain exactly two integers.")
        min_length, max_length = length_range

    try:
        minimum = validate_integer(min_length, "Minimum length", minimum=0)
        maximum = validate_integer(max_length, "Maximum length", minimum=0)
    except TypeError as exc:
        raise ValueError("Length bounds must be integers.") from exc
    if minimum > maximum:
        raise ValueError("Minimum length cannot exceed maximum length.")
    return minimum, maximum
