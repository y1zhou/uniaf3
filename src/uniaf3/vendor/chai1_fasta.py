"""Simple FASTA parsing utilities."""

# Copyright (c) 2024 Chai Discovery, Inc.
# Licensed under the Apache License, Version 2.0.
# See the LICENSE file for details.
from pathlib import Path
from string import ascii_letters, digits
from typing import NamedTuple


class Fasta(NamedTuple):
    """Simple container for FASTA entries."""

    header: str
    sequence: str


def read_fasta(file_path: str | Path) -> list[Fasta]:
    """Read a FASTA file and return a list of Fasta named tuples."""
    sequences = []
    with open(file_path) as f:
        header, seq_lines = None, []
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if header is not None:
                    sequences.append(Fasta(header=header, sequence="".join(seq_lines)))
                header = line[1:]  # Remove '>'
                seq_lines = []
            else:
                seq_lines.append(line)
        if header is not None:
            sequences.append(Fasta(header=header, sequence="".join(seq_lines)))
    return sequences


def constituents_of_modified_fasta(x: str) -> list[str] | None:
    """Parse a FASTA sequence into a list of its constituents.

    Accepts RNA/DNA inputs: 'agtc', 'AGT(ASP)TG', etc. Does not accept SMILES strings.
    Returns constituents, e.g, [A, G, T, ASP, T, G] or None if string is incorrect.
    Everything in returned list is single character, except for blocks specified in brackets.
    """
    x = x.strip().upper()
    # it is a bit strange that digits are here, but [NH2] was in one protein
    allowed_chars = ascii_letters + "()" + digits
    if not all(letter in allowed_chars for letter in x):
        return None

    current_modified: str | None = None

    constituents = []
    for letter in x:
        if letter == "(":
            if current_modified is not None:
                return None  # double open bracket
            current_modified = ""
        elif letter == ")":
            if current_modified is None:
                return None  # closed without opening
            if len(current_modified) <= 1:
                return None  # empty modification: () or single (K)
            constituents.append(current_modified)
            current_modified = None
        else:
            if current_modified is not None:
                current_modified += letter
            else:
                if letter not in ascii_letters:
                    return None  # strange single-letter residue
                constituents.append(letter)
    if current_modified is not None:
        return None  # did not close bracket
    return constituents
