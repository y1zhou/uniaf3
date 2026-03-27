"""Vendored from the Protenix codebase.

<https://github.com/bytedance/Protenix/blob/main/protenix/data/template/template_parser.py>
"""
# Copyright 2024 ByteDance and/or its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Copyright 2021 AlQuraishi Laboratory

import dataclasses
import functools
import re
from collections.abc import Mapping, Sequence

from uniaf3.vendor.chai1_fasta import read_fasta


@dataclasses.dataclass(frozen=True)
class TemplateHit:
    """Represents a template hit from a search tool (e.g., HHSearch, hmmsearch)."""

    index: int
    name: str
    aligned_cols: int
    sum_probs: float | None
    query: str
    hit_sequence: str
    indices_query: list[int]
    indices_hit: list[int]

    @functools.cached_property
    def query_to_hit_mapping(self) -> Mapping[int, int]:
        """Maps 0-based query indices to 0-based hit indices."""
        mapping = {}
        for q_idx, h_idx in zip(self.indices_query, self.indices_hit, strict=True):
            if (q_idx != -1) and (h_idx != -1):
                mapping[q_idx] = h_idx
        return mapping


@dataclasses.dataclass(frozen=True)
class HitMetadata:
    """Metadata parsed from an hmmsearch A3M description line."""

    pdb_id: str
    chain: str
    start: int
    end: int
    length: int
    text: str


class HHRParser:
    """Class to parse HHR files from HHSearch."""

    @staticmethod
    def parse(hhr_string: str) -> list[TemplateHit]:
        """Parse an entire HHR file content.

        Args:
            hhr_string: The content of the HHR file.

        Returns:
            A list of TemplateHit objects.

        """
        lines = hhr_string.splitlines()
        block_starts = [i for i, line in enumerate(lines) if line.startswith("No ")]
        hits = []
        if block_starts:
            block_starts.append(len(lines))
            for i in range(len(block_starts) - 1):
                hits.append(
                    HHRParser._parse_hit(lines[block_starts[i] : block_starts[i + 1]])
                )
        return hits

    @staticmethod
    def _parse_hit(lines: Sequence[str]) -> TemplateHit:
        """Parse a single hit block from an HHR file."""
        hit_num = int(lines[0].split()[-1])
        hit_name = lines[1][1:].strip()
        summary = lines[2]
        match = re.search(r"Aligned_cols=(\d+).*Sum_probs=([0-9.]+)", summary)
        if not match:
            raise RuntimeError(f"Could not parse HHR summary: {summary}")
        cols, probs = int(match.group(1)), float(match.group(2))

        query, hit_seq = "", ""
        idx_q, idx_h = [], []
        for line in lines[3:]:
            if line.startswith("Q ") and not any(
                line.startswith(x) for x in ("Q ss_", "Q Consensus")
            ):
                match = re.search(r"\s+(\d+)\s+([A-Z-]+)\s+(\d+)", line[17:])
                if match:
                    start = int(match.group(1)) - 1
                    seq = match.group(2)
                    query += seq
                    HHRParser._update_residue_indices(seq, start, idx_q)
            elif line.startswith("T ") and not any(
                line.startswith(x) for x in ("T ss_", "T Consensus")
            ):
                match = re.search(r"\s+(\d+)\s+([A-Z-]+)", line[17:])
                if match:
                    start = int(match.group(1)) - 1
                    seq = match.group(2)
                    hit_seq += seq
                    HHRParser._update_residue_indices(seq, start, idx_h)

        return TemplateHit(hit_num, hit_name, cols, probs, query, hit_seq, idx_q, idx_h)

    @staticmethod
    def _update_residue_indices(seq: str, start: int, indices: list[int]):
        """Update the list of residue indices for a sequence segment."""
        curr = start
        for char in seq:
            if char == "-":
                indices.append(-1)
            else:
                indices.append(curr)
                curr += 1


class HmmsearchA3MParser:
    """Class to parse A3M files from hmmsearch."""

    @staticmethod
    def parse(
        query_seq: str, a3m_str: str, skip_first: bool = True
    ) -> list[TemplateHit]:
        """Parse an A3M string from hmmsearch.

        Args:
            query_seq: The query sequence.
            a3m_str: The content of the A3M file.
            skip_first: Whether to skip the first sequence (usually the query).

        Returns:
            A list of TemplateHit objects.

        """
        from tempfile import NamedTemporaryFile

        with NamedTemporaryFile(suffix=".fa") as temp_fasta:
            temp_fasta.write(a3m_str.encode())
            temp_fasta.seek(0)
            parsed = read_fasta(temp_fasta.name)

        if skip_first:
            parsed = parsed[1:]

        idx_q = HmmsearchA3MParser._get_indices(query_seq, 0)
        hits = []
        for i, seq in enumerate(parsed, start=1):
            h_seq, h_desc = seq.sequence, seq.header
            if "mol:protein" not in h_desc:
                continue
            meta = HmmsearchA3MParser._parse_description(h_desc)
            cols = sum(1 for r in h_seq if r.isupper() and r != "-")
            idx_h = HmmsearchA3MParser._get_indices(h_seq, meta.start - 1)
            hits.append(
                TemplateHit(
                    i,
                    f"{meta.pdb_id}_{meta.chain}",
                    cols,
                    None,
                    query_seq,
                    h_seq.upper(),
                    idx_q,
                    idx_h,
                )
            )
        return hits

    @staticmethod
    def _get_indices(seq: str, start: int) -> list[int]:
        """Calculate residue indices for a sequence with gaps or insertions."""
        indices = []
        curr = start
        for char in seq:
            if char == "-":
                indices.append(-1)
            elif char.islower():
                curr += 1
            else:
                indices.append(curr)
                curr += 1
        return indices

    @staticmethod
    def _parse_description(desc: str) -> HitMetadata:
        """Parse the description line from an hmmsearch hit."""
        pattern = (
            r"^>?([a-z0-9]+)_(\w+)/([0-9]+)-([0-9]+).*protein length:([0-9]+) *(.*)$"
        )
        match = re.match(pattern, desc.strip())
        if not match:
            raise ValueError(f"Could not parse hmmsearch description: {desc}")
        return HitMetadata(
            match[1], match[2], int(match[3]), int(match[4]), int(match[5]), match[6]
        )
