"""Simple FASTA parsing utilities."""

from pathlib import Path
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
