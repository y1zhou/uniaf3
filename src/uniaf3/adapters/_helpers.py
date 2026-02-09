"""Shared helpers for adapter modules."""

from __future__ import annotations

_KNOWN_ION_CCD_CODES = frozenset(
    {
        "AG",
        "BA",
        "BR",
        "CA",
        "CD",
        "CL",
        "CO",
        "CS",
        "CU",
        "F",
        "FE",
        "HG",
        "I",
        "K",
        "LI",
        "MG",
        "MN",
        "NA",
        "NI",
        "PB",
        "RB",
        "SE",
        "SR",
        "TL",
        "ZN",
    }
)


def _ensure_list(val: str | list[str]) -> list[str]:
    """Normalize id field to a list."""
    return val if isinstance(val, list) else [val]
