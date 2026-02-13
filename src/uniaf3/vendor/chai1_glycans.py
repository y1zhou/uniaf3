"""Vendored from Chai Discovery, Chai-1 codebase.

https://github.com/chaidiscovery/chai-lab/tree/af596cbc075a1fce368cec0ab5f31be1090ca7e2

chai_lab/data/parsing/glycans.py
"""
# Copyright (c) 2024 Chai Discovery, Inc.
# Licensed under the Apache License, Version 2.0.
# See the LICENSE file for details.

import re
from dataclasses import dataclass
from functools import lru_cache


@dataclass(frozen=True)
class GlycosidicBond:
    """Represents a glycosidic bond between two sugars in a glycan."""

    src_sugar_index: int  # 0-indexed
    dst_sugar_index: int  # 0-indexed
    src_atom: int  # 1-indexed
    dst_atom: int  # 1-indexed

    def __post_init__(self):
        """Validate the bond information."""
        if self.src_sugar_index == self.dst_sugar_index:
            raise ValueError(
                f"Invalid glycosidic bond: source and destination sugars cannot be the same (index {self.src_sugar_index})."
            )
        if not (self.src_atom > 0 and self.dst_atom > 0):
            raise ValueError(
                f"Invalid glycosidic bond: atom numbers must be positive (got {self.src_atom=}, {self.dst_atom=})."
            )

    @property
    def src_atom_name(self) -> str:
        """Links between sugars are O-glycosidic bonds; we use src O dst C."""
        return f"O{self.src_atom}"

    @property
    def dst_atom_name(self) -> str:
        """Links between sugars are O-glycosidic bonds; we use src O dst C."""
        return f"C{self.dst_atom}"


@lru_cache(maxsize=32)
def _glycan_string_to_sugars_and_bonds(
    glycan_string: str,
) -> tuple[list[str], list[GlycosidicBond]]:
    """Parse the glycan string to its constituent sugars and bonds."""
    glycan_string = glycan_string.strip()  # Remove leading/trailing spaces
    sugars: list[str] = []  # Tracks all sugars
    parent_sugar_idx: list[int] = []  # Tracks the parent sugar for bond formation
    bonds: list[GlycosidicBond] = []
    open_count, closed_count = 0, 0
    i = 0  # We increment unevenly so manually handle
    while i < len(glycan_string):
        char = glycan_string[i]
        if char == " ":  # Space; skip
            i += 1
            continue
        if char == "(":  # Open bracket
            i += 1
            open_count += 1
            continue
        if char == ")":  # Close bracket
            closed_count += 1
            parent_sugar_idx.pop()  # Remove
            i += 1
            continue
        # Not a bracket or a space - should be either bond info or CCD
        chunk = glycan_string[i : i + 3]
        if re.match(r"[1-6]{1}-[1-6]{1}", chunk):
            s, d = chunk.split("-")
            if not parent_sugar_idx:
                raise ValueError(
                    f"Invalid glycan string: bond {chunk} cannot be formed without a parent sugar"
                )
            bonds.append(
                GlycosidicBond(
                    src_sugar_index=parent_sugar_idx[-1],
                    dst_sugar_index=len(sugars),  # Anticipate next
                    src_atom=int(s),
                    dst_atom=int(d),
                )
            )
            i += 3
        elif re.match(r"[0-9A-Z]{3}", chunk):
            sugars.append(chunk)
            parent_sugar_idx.append(len(sugars) - 1)  # latest sugar
            i += 3
        else:
            raise ValueError(f"Invalid glycan string: {glycan_string}")
    if open_count != closed_count:
        raise ValueError(
            f"Invalid glycan string: unbalanced parentheses in {glycan_string}"
        )
    return sugars, bonds
