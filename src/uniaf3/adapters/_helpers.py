"""Shared helpers for adapter modules."""

from __future__ import annotations


def ensure_list(val: str | list[str]) -> list[str]:
    """Normalize id field to a list."""
    return val if isinstance(val, list) else [val]


def err_unsupported_feature(strict: bool, msg: str):
    """Help handle unsupported features based on the strict flag."""
    if strict:
        raise ValueError(msg)
    else:
        print(f"[Warning] Skipping unsupported feature: {msg}")
