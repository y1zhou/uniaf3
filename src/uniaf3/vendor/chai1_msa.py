"""Vendored from Chai Discovery, Chai-1 codebase.

https://github.com/chaidiscovery/chai-lab/tree/af596cbc075a1fce368cec0ab5f31be1090ca7e2

chai_lab/data/dataset/msas/colabfold.py
chai_lab/data/parsing/fasta.py
chai_lab/data/parsing/msas/aligned_pqt.py
chai_lab/data/parsing/msas/data_source.py
chai_lab/data/parsing/templates/m8.py
"""

# Licensed under the Apache License, Version 2.0.
# See the LICENSE file for details.
# Copyright (c) 2024 Chai Discovery, Inc.
import logging
import tempfile
from enum import Enum
from pathlib import Path

import polars as pl

from uniaf3.msa import parse_m8_file
from uniaf3.utils import hash_sequence
from uniaf3.vendor.chai1_fasta import Fasta, read_fasta
from uniaf3.vendor.colabfold_msa import run_mmseqs2

logger = logging.getLogger(__name__)

TQDM_BAR_FORMAT = (
    "{l_bar}{bar}| {n_fmt}/{total_fmt} [elapsed: {elapsed} remaining: {remaining}]"
)

# from chai_lab import __version__
CHAI_VERSION = "0.6.1"


# from chai_lab.data.parsing.msas.aligned_pqt import expected_basename, hash_sequence
def expected_basename(query_sequence: str) -> str:
    """Get the expected filename based on the uppercased query sequence."""
    seqhash = hash_sequence(query_sequence.upper())
    return f"{seqhash}.aligned.pqt"


# from chai_lab.data.parsing.msas.data_source import MSADataSource
class MSADataSource(Enum):
    """Enum for MSA data sources."""

    # Special value for the query sequence
    QUERY = "query"

    UNIPROT = "uniprot"
    UNIREF90 = "uniref90"
    BFD = "BFD"
    MGNIFY = "mgnify"
    PAIRED = "paired"
    MAIN = "main"
    BFD_UNICLUST = "bfd_uniclust"
    SINGLETON = "singleton"

    # pad value
    NONE = "none"

    # templates
    PDB70 = "pdb70"

    # ran with 3 jackhmmer iterations (-N=3),
    # higher quality but sloow to generate
    UNIPROT_N3 = "uniprot_n3"
    UNIREF90_N3 = "uniref90_n3"
    MGNIFY_N3 = "mgnify_n3"

    @classmethod
    def get_default_sources(cls):
        """Get the default MSA data sources to use for Chai-1."""
        return [
            MSADataSource.BFD_UNICLUST,
            MSADataSource.MGNIFY,
            MSADataSource.UNIREF90,
            MSADataSource.UNIPROT,
        ]


def _is_padding_msa_row(sequence: str) -> bool:
    """Check if the given MSA sequence is a a padding sequence."""
    seq_chars = set(sequence)
    return len(seq_chars) == 1 and seq_chars.pop() == "-"


def a3m_to_aligned_pqt(
    single_a3m_path: str | Path,
    pair_a3m_path: str | Path | None,
    output_dir: str | Path,
    query_sequence: str,
) -> Path:
    """Convert A3M MSA files to Chai-1's aligned Parquet format.

    Reads paired and unpaired A3M files, assigns pairing keys and source
    database annotations, and writes a single ``.aligned.pqt`` Parquet file
    suitable for ``chai_lab.chai1.run_inference``.

    Args:
        single_a3m_path: Path to the unpaired/single MSA A3M file.
        pair_a3m_path: Path to the paired MSA A3M file, or None if no paired
            MSA exists.
        output_dir: Directory to write the .aligned.pqt file.
        query_sequence: The query protein sequence (used for hash-based filename).

    Returns:
        Path to the written .aligned.pqt file.

    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    msa_path = output_dir / expected_basename(query_sequence)
    if msa_path.exists():
        # Homomer dedup: same sequence produces the same file
        return msa_path

    single_a3m_path = Path(single_a3m_path)
    if not single_a3m_path.exists():
        raise FileNotFoundError(f"Single MSA A3M file not found at {single_a3m_path}")

    # Read paired A3M if available
    if pair_a3m_path is not None:
        pair_a3m_path = Path(pair_a3m_path)
        paired_fasta: list[tuple[str, str, str]] = [
            (str(pairkey), record.header, record.sequence)
            for pairkey, record in enumerate(read_fasta(pair_a3m_path))
            if not _is_padding_msa_row(record.sequence)
        ]
        pairing_key, paired_headers, paired_msa_seqs = (
            zip(*paired_fasta, strict=True) if paired_fasta else ((), (), ())
        )
    else:
        pairing_key, paired_headers, paired_msa_seqs = (), (), ()

    unique_paired_msa_seqs = set(paired_msa_seqs)

    # Non-paired MSA sequences that weren't already covered in the paired MSA
    # If there were paired MSAs, then skip the header to avoid duplication
    single_fasta: list[Fasta] = [
        record
        for i, record in enumerate(read_fasta(single_a3m_path))
        if (
            (len(paired_headers) == 0 or i > 0)
            and not _is_padding_msa_row(record.sequence)
            and record.sequence not in unique_paired_msa_seqs
        )
    ]
    single_headers = [record.header for record in single_fasta]
    single_msa_seqs = [record.sequence for record in single_fasta]
    single_null_pair_keys = [""] * len(single_msa_seqs)

    # Best-effort source database synthesis from headers
    source_databases = ["query"] + [
        (
            MSADataSource.UNIREF90.value
            if h.startswith("UniRef")
            else MSADataSource.BFD_UNICLUST.value
        )
        for h in (list(paired_headers) + single_headers)[1:]
    ]

    all_sequences = list(paired_msa_seqs) + single_msa_seqs
    all_pairing_keys = list(pairing_key) + single_null_pair_keys
    if not (len(all_sequences) == len(all_pairing_keys) == len(source_databases)):
        raise ValueError(
            f"Mismatched lengths: {len(all_sequences)=} {len(all_pairing_keys)=} {len(source_databases)=}"
        )

    aligned_df = pl.from_dict(
        data=dict(
            sequence=all_sequences,
            source_database=source_databases,
            pairing_key=all_pairing_keys,
        ),
    ).with_columns(pl.lit("").alias("comment"))
    aligned_df.write_parquet(msa_path)
    return msa_path


def generate_colabfold_msas(
    protein_seqs: list[str],
    msa_dir: Path,
    msa_server_url: str,
    search_templates: bool = False,
    write_a3m_to_msa_dir: bool = False,  # Useful for manual inspection + debugging
) -> dict[str, Path]:
    """Generate MSA for protein sequences.

    Generate MSAs using the ColabFold (https://github.com/sokrypton/ColabFold)
    server. No-op if no protein sequences are given.

    N.B.:
    - the MSAs in our technical report were generated using jackhmmer, not
    ColabFold, so we would expect some difference in results.
    - this implementation relies on ColabFold's chain pairing algorithm
    rather than using Chai-1's own algorithm, which could also lead to
    differences in results.

    Places .aligned.pqt files in msa_dir; does not save intermediate a3m files.
    """
    if not msa_dir.is_dir():
        raise NotADirectoryError("MSA directory must be a dir")
    if any(msa_dir.iterdir()):
        raise FileExistsError("MSA directory must be empty")
    if not protein_seqs:
        logger.warning("No protein sequences for MSA generation; this is a no-op.")
        return {}

    with tempfile.TemporaryDirectory() as tmp_dir_path:
        tmp_dir = Path(tmp_dir_path)

        mmseqs_paired_dir = tmp_dir / "mmseqs_paired"
        mmseqs_paired_dir.mkdir()

        mmseqs_dir = tmp_dir / "mmseqs"
        mmseqs_dir.mkdir()

        a3ms_dir = (tmp_dir if not write_a3m_to_msa_dir else msa_dir) / "a3ms"
        a3ms_dir.mkdir()

        # Generate MSAs for each protein chain
        logger.info(f"Running MSA generation for {len(protein_seqs)} protein sequences")

        # Identify ourselves to the ColabFold server
        user_agent = f"chai-lab/{CHAI_VERSION} feedback@chaidiscovery.com"

        # In paired mode, mmseqs2 returns paired a3ms where all a3ms have the same number of rows
        # and each row is already paired to have the same species. As such, we insert pairing key
        # as the i-th index of the sequence so long as it isn't a padding sequence (all -)
        paired_msas: list[str]
        if len(protein_seqs) > 1:
            paired_msas, _ = run_mmseqs2(
                protein_seqs,
                mmseqs_paired_dir,
                use_pairing=True,
                use_templates=False,  # No templates when running paired search
                host_url=msa_server_url,
                user_agent=user_agent,
            )
        else:
            # If we only have a single protein chain, there are no paired MSAs by definition
            paired_msas = [""] * len(protein_seqs)

        # MSAs without pairing logic attached; may include sequences not contained in the paired MSA
        # Needs a second call as the colabfold server returns either paired or unpaired, not both
        per_chain_msas, template_hits_file = run_mmseqs2(
            protein_seqs,
            mmseqs_dir,
            use_pairing=False,
            use_templates=search_templates,
            host_url=msa_server_url,
            user_agent=user_agent,
        )
        if search_templates:
            if template_hits_file is None or not template_hits_file.is_file():
                raise FileNotFoundError(
                    f"Expected template hits file not found at {template_hits_file}."
                )
            all_templates = parse_m8_file(template_hits_file)
            # query IDs are 101, 102, ... from the server; remap IDs
            query_map = {}
            for orig_query_id, orig_seq in enumerate(protein_seqs, start=101):
                h = hash_sequence(orig_seq)
                query_map[orig_query_id] = h
            all_templates = all_templates.with_columns(
                pl.col("query_id").replace_strict(query_map)
            )

            logger.info(f"Found {len(all_templates)} template hits")
            all_templates.write_csv(
                msa_dir / "all_chain_templates.m8", include_header=False, separator="\t"
            )

        # Process the MSAs into our internal format
        msa_paths: dict[str, Path] = {}  # Map each sequence to path of aligned pqt
        for protein_seq, pair_msa, single_msa in zip(
            protein_seqs, paired_msas, per_chain_msas, strict=True
        ):
            # Write out an A3M file for both
            hkey = hash_sequence(protein_seq.upper())
            pair_a3m_path = a3ms_dir / f"{hkey}.pair.a3m"
            pair_a3m_path.write_text(pair_msa)
            single_a3m_path = a3ms_dir / f"{hkey}.single.a3m"
            single_a3m_path.write_text(single_msa)

            # Convert the A3M files into aligned parquet files
            pair_path = pair_a3m_path if pair_msa else None
            msa_path = a3m_to_aligned_pqt(
                single_a3m_path=single_a3m_path,
                pair_a3m_path=pair_path,
                output_dir=msa_dir,
                query_sequence=protein_seq,
            )
            msa_paths[protein_seq] = msa_path
    return msa_paths
