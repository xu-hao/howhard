"""Shared strict validation. Booleans are never accepted as numbers."""

import math
import re


class HowHardError(ValueError):
    """Invalid input or an unsuccessful operation, suitable for a CLI diagnostic."""


def table(value, where):
    if not isinstance(value, dict):
        raise HowHardError(f"{where}: expected a table/object")
    return value


def keys(value, required, optional=(), where="value"):
    if not isinstance(value, dict):
        raise HowHardError(f"{where}: expected a table/object")
    missing = set(required) - value.keys()
    unknown = value.keys() - set(required) - set(optional)
    if missing or unknown:
        raise HowHardError(f"{where}: missing keys {sorted(missing)}; unknown keys {sorted(unknown)}")


def number(value, where, *, positive=False, nonnegative=False):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise HowHardError(f"{where}: expected a finite number")
    if positive and value <= 0 or nonnegative and value < 0:
        raise HowHardError(f"{where}: expected a {'positive' if positive else 'non-negative'} number")
    return float(value)


def text(value, where, *, label=False):
    if not isinstance(value, str) or not value:
        raise HowHardError(f"{where}: expected a non-empty string")
    if label and not re.fullmatch(r"[A-Za-z0-9_-]+", value):
        raise HowHardError(f"{where}: names must contain only letters, digits, '_' or '-'")
    return value
