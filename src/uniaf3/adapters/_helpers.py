"""Shared helpers for adapter modules."""

from __future__ import annotations

import warnings


def ensure_list(val: str | list[str]) -> list[str]:
    """Normalize id field to a list."""
    return val if isinstance(val, list) else [val]


def warn_lossy_conversion(msg: str):
    """Emit a warning for lossy conversion behavior."""
    warnings.warn(f"Lossy conversion: {msg}", UserWarning, stacklevel=3)


def err_unsupported_feature(strict: bool, msg: str):
    """Help handle unsupported features based on the strict flag."""
    if strict:
        raise ValueError(msg)
    else:
        warnings.warn(f"Skipping unsupported feature: {msg}", UserWarning, stacklevel=3)
