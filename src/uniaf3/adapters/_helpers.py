"""Shared helpers for adapter modules."""

from __future__ import annotations

KNOWN_ION_CCD_CODES = frozenset(
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


def ensure_list(val: str | list[str]) -> list[str]:
    """Normalize id field to a list."""
    return val if isinstance(val, list) else [val]


def err_unsupported_feature(strict: bool, msg: str):
    """Help handle unsupported features based on the strict flag."""
    if strict:
        raise ValueError(msg)
    else:
        print(f"[Warning] Skipping unsupported feature: {msg}")


def int_to_letters(n: int) -> str:
    """Convert int to letters.

    Useful for converting chain index to label_asym_id.

    Args:
        n (int): int number

    Returns:
        chain ID, e.g. 1 -> A, 2 -> B, 27 -> AA, 28 -> AB

    """
    result = ""
    while n > 0:
        n, remainder = divmod(n - 1, 26)
        result = chr(65 + remainder) + result
    return result
