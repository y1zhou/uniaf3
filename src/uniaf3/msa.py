"""Query MSA and templates for protein sequences."""

import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import gemmi
import polars as pl
from pydantic import BaseModel, ConfigDict, model_validator

from uniaf3.constant import PDB_SERVER_URL
from uniaf3.utils import download_files, hash_sequence, normalize_out_dir
from uniaf3.vendor.colabfold_msa import run_mmseqs2


# TODO: edge case where the MSA search returned no hits
class ColabFoldResponse(BaseModel):
    """Response from ColabFold API."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    query_key: str  # sha256(seq1|seq2|...)
    msa_dir: Path  # directory containing MSA files and template hits
    protein_seqs: list[str]
    seq_hashes: list[str]  # sha256(seq1), sha256(seq2), ...
    query_ids: list[int]  # ColabFold query IDs, starting from 101
    single_msas: list[Path]
    paired_msas: list[Path] | None = None
    templates_m8_file: Path | None = None
    templates_df: pl.DataFrame | None = None

    @model_validator(mode="after")
    def check_list_lengths(self):
        """Validate that the lengths of fields are the same."""
        n = len(self.protein_seqs)
        if len(self.seq_hashes) != n:
            raise ValueError(
                f"Length of seq_hashes ({len(self.seq_hashes)}) does not match length of protein_seqs ({n})."
            )
        if len(self.query_ids) != n:
            raise ValueError(
                f"Length of query_ids ({len(self.query_ids)}) does not match length of protein_seqs ({n})."
            )
        if len(self.single_msas) != n:
            raise ValueError(
                f"Length of single_msas ({len(self.single_msas)}) does not match length of protein_seqs ({n})."
            )
        if n > 1:
            if self.paired_msas is None:
                raise ValueError(
                    "paired_msas cannot be None when there are multiple protein sequences."
                )
            if len(self.paired_msas) != n:
                raise ValueError(
                    f"Length of paired_msas ({len(self.paired_msas)}) does not match length of protein_seqs ({n})."
                )
        return self

    def __getitem__(self, protein_seq: str) -> dict[str, int | str | Path | None]:
        """Get MSA and template paths for a given protein sequence."""
        idx = self.protein_seqs.index(protein_seq)
        return {
            "seq_hash": self.seq_hashes[idx],
            "query_id": self.query_ids[idx],
            "single_msa": self.single_msas[idx],
            "paired_msa": self.paired_msas[idx]
            if self.paired_msas is not None
            else None,
        }


def query_colabfold(
    seqs: Sequence[str],
    msa_cache_dir: str | Path | None = None,
    search_templates: bool = False,
    download_num_templates_per_seq: int = 0,
    template_cache_dir: Path | None = None,
    force: bool = False,
    **kwargs,
) -> ColabFoldResponse:
    """Query ColabFold API for MSAs and templates.

    Note that if ``kwargs`` are passed with filters or pairing strategies
    specified, the cache results could be invalid. In that case, set ``force=True`` to ignore cache and re-query the server.

    Args:
        seqs: List of protein sequences. The order matters, as the concatenated sequences
            will be used as the cache key under ``msa_cache_dir``.
        msa_cache_dir: Directory to cache MSA files. Defaults to
            ``$XDG_CACHE_HOME/uniaf3/colabfold_msas/``.
        search_templates: Whether to search for templates.
        download_num_templates_per_seq: Number of template files to fetch per sequence.
            If greater than 0, asymmetric unit mmCIF files will be fetched from RCSB.
            If smaller than 0, all templates in the API-returned m8 file will be fetched.
            Defaults to 0, i.e. no template files will be fetched.
        template_cache_dir: Directory to cache template hits. Defaults to
            ``$XDG_CACHE_HOME/uniaf3/rcsb/``.
        force: Whether to ignore cache and re-query the server.
        kwargs: Additional args to pass to ``run_mmseqs2``.

    Returns:
        The ColabFoldResponse containing MSA paths. Files are structured as:

        ```
        msa_cache_dir/
          <seq_hash>[:2]/
            <seq_hash>/
              <seq_hash>.single.a3m          (per individual sequence)
          <seqs_hash>[:2]/
            <seqs_hash>/
              a3ms/
                <seq_hash>.pair.a3m          (per query combination)
              pdb70.m8 (if use_templates is True)
        ```

        Single MSAs are stored per-sequence so that different queries sharing
        the same chains can reuse cached results without re-hitting the API.

    """
    # Setup output directories
    if not seqs:
        raise ValueError("No protein sequences for MSA generation; this is a no-op.")

    msa_cache_root = normalize_out_dir(msa_cache_dir, "colabfold_msas")

    seqs_unique = list(dict.fromkeys(x.upper() for x in seqs))  # >=Python 3.7
    query_indices = [101 + seqs_unique.index(seq.upper()) for seq in seqs]

    seqs_hash = hash_sequence("|".join(seqs_unique))
    paired_msa_dir = normalize_out_dir(
        msa_cache_root / "paired" / seqs_hash[:2] / seqs_hash
    )

    # Expected output files to be cached
    seq_hashes = [hash_sequence(s.upper()) for s in seqs_unique]
    num_seqs = len(seqs_unique)

    # Single MSAs go to per-sequence directories for cross-query reuse
    single_a3m_files = []
    for seq_hash in seq_hashes:
        single_dir = normalize_out_dir(
            msa_cache_root / "single" / seq_hash[:2] / seq_hash
        )
        single_a3m_files.append((single_dir / f"{seq_hash}.single.a3m").resolve())

    # Paired MSAs stay in the per-query directory
    if num_seqs > 1:
        a3ms_dir = normalize_out_dir(paired_msa_dir, "a3ms")
    paired_a3m_files = (
        [a3ms_dir / f"{seq_hash}.pair.a3m" for seq_hash in seq_hashes]
        if num_seqs > 1
        else []
    )
    expected_tmpl_m8_file = paired_msa_dir / "pdb70.m8"

    # Query ColabFold API if cached results do not exist
    with tempfile.TemporaryDirectory() as tmp_dir:
        mmseqs_paired_dir = normalize_out_dir(tmp_dir, "mmseqs2_paired")

        mmseqs_dir = normalize_out_dir(tmp_dir, "mmseqs2")

        # Run paired MSA search
        # In paired mode, mmseqs2 returns paired a3ms where all a3ms have the same number of rows
        # and each row is already paired to have the same species.

        if num_seqs > 1:
            if force or not all(p.exists() for p in paired_a3m_files):
                paired_msas, _ = run_mmseqs2(
                    x=seqs_unique,
                    prefix=mmseqs_paired_dir,
                    use_pairing=True,
                    use_templates=False,
                    **kwargs,
                )
                for f, paired_msa in zip(paired_a3m_files, paired_msas, strict=True):
                    if force or not f.exists():
                        f.write_text(paired_msa)
        else:
            # By definition, a single protein chain has no paired MSAs
            paired_msas = [""] * num_seqs

        # Run MSA search without pairing to get more hits for each chain
        if (
            force
            or not all(p.exists() for p in single_a3m_files)
            or (search_templates and not expected_tmpl_m8_file.exists())
        ):
            per_chain_msas, template_hits_m8_file = run_mmseqs2(
                x=seqs_unique,
                prefix=mmseqs_dir,
                use_pairing=False,
                use_templates=search_templates,
                **kwargs,
            )
            for f, single_msa in zip(single_a3m_files, per_chain_msas, strict=True):
                if force or not f.exists():
                    f.write_text(single_msa)
            if template_hits_m8_file is not None:
                if force or not expected_tmpl_m8_file.exists():
                    import shutil

                    shutil.copyfile(template_hits_m8_file, expected_tmpl_m8_file)

    # Cache mmCIF files for templates (some models can use local files)
    # Note that we always go for asymmetric unit files, as biological assemblies can
    # miss chains that are present in the m8 file.
    templates_df = None
    if search_templates and download_num_templates_per_seq != 0:
        template_cache_dir = normalize_out_dir(template_cache_dir, "rcsb")

        if search_templates and not expected_tmpl_m8_file.exists():
            raise FileNotFoundError(
                f"Expected template hits file not found at {expected_tmpl_m8_file}."
            )
        all_templates = parse_m8_file(expected_tmpl_m8_file)

        # Models cannot fit too many templates into GPU memory anyways, so it makes
        # sense to only download a few templates per query sequence
        if download_num_templates_per_seq > 0:
            all_templates = all_templates.group_by(
                "query_id", maintain_order=True
            ).head(download_num_templates_per_seq)

        all_templates = all_templates.with_columns(
            pl
            .col("subject_id")
            .str.split("_")
            .list.first()
            .str.to_uppercase()
            .alias("subject_pdb_id")
        )
        template_pdb_ids = (
            all_templates
            .select("subject_pdb_id")
            .unique()
            .with_columns(
                pl.concat_str(
                    pl.lit(f"{PDB_SERVER_URL}/"),
                    pl.col("subject_pdb_id"),
                    pl.lit(".cif.gz"),  # -assembly1.cif.gz for biological assembly
                ).alias("template_cif_url"),
                pl.concat_str(
                    pl.lit(f"{template_cache_dir}/"),
                    pl.col("subject_pdb_id").str.slice(offset=-3, length=2),
                    pl.lit("/"),
                    pl.col("subject_pdb_id"),
                    pl.lit(".cif.gz"),
                ).alias("template_cif_path"),
            )
        )
        download_files(
            {
                r["template_cif_url"]: Path(r["template_cif_path"])
                for r in template_pdb_ids.iter_rows(named=True)
            },
            force=force,
            num_retries=3,
            progress_bar_desc="Downloading templates from RCSB",
        )

        # Template files are gzipped, but all except Protenix can handle them directly.
        # TODO: If needed, we can add logic to unzip files after the downloads.
        templates_df = all_templates.join(
            template_pdb_ids, on="subject_pdb_id", maintain_order="left"
        )

    return ColabFoldResponse(
        query_key=seqs_hash,
        msa_dir=paired_msa_dir,
        protein_seqs=seqs_unique,
        seq_hashes=seq_hashes,
        query_ids=query_indices,
        single_msas=single_a3m_files,
        paired_msas=paired_a3m_files if num_seqs > 1 else None,
        templates_m8_file=expected_tmpl_m8_file if search_templates else None,
        templates_df=templates_df,
    )


def parse_m8_file(fname: str | Path) -> pl.DataFrame:
    """Parse the m8 alignment format describing template information.

    Inspired by: chai_lab.data.parsing.templates.m8 import parse_m8_file.
    """
    table = (
        pl
        .scan_csv(
            fname,
            separator="\t",
            has_header=False,
            new_columns=[
                "query_id",
                "subject_id",
                "pident",
                "length",
                "mismatch",
                "gapopen",
                "query_start",
                "query_end",
                "subject_start",
                "subject_end",
                "evalue",
                "bitscore",
                "cigar",
            ],
        )
        .sort(
            ["query_id", "bitscore", "evalue", "pident", "subject_id"],
            descending=[False, True, False, True, False],
        )
        .with_columns(
            pl.col(c).cast(pl.Int64)
            for c in ("query_start", "query_end", "subject_start", "subject_end")
        )
        .collect()
    )
    return table


def cigar_to_indices(
    query_start: int,
    subject_start: int,
    cigar: str,
    index_offset: int = -1,
) -> tuple[list[int], list[int]]:
    """Convert CIGAR string to query and subject indices.

    Note that the CIGAR string is a subset of the full spec, and only contains
    M (match), D (deletion, gap in query), or I (insertion, gap in subject)
    operations. Ops like S, H, or X are not expected in ColabFold's m8 output.
    """
    import re

    CIGAR_REGEX = re.compile(r"(\d+)([MID])")

    query_indices = []
    subject_indices = []
    q_pos = query_start + index_offset
    s_pos = subject_start + index_offset
    for match in CIGAR_REGEX.finditer(cigar):
        length_str, op = match.groups()
        length = int(length_str)
        if op == "M":
            query_indices.extend(range(q_pos, q_pos + length))
            subject_indices.extend(range(s_pos, s_pos + length))
            q_pos += length
            s_pos += length
        elif op == "I":
            q_pos += length
        elif op == "D":
            s_pos += length

    if len(query_indices) != len(subject_indices):
        raise ValueError(
            f"CIGAR parsing error: number of query indices ({len(query_indices)}) does not match number of subject indices ({len(subject_indices)})."
        )
    return query_indices, subject_indices


@dataclass
class GemmiAlignmentResult:
    """A wrapper for gemmi.AlignmentResult extended with sequences."""

    raw: gemmi.AlignmentResult
    struct_chain_id: str
    # 1-based indices for sequences (not residue index in the structure!)
    query_idx: list[int]
    struct_idx: list[int]


def align_seq_to_structure(
    seq: str, struct_path: str | Path, chain_id: str | None = None, model_id: int = 0
) -> GemmiAlignmentResult:
    """Align a sequence to a structure and return the aligned sequence with gaps.

    <https://gemmi.readthedocs.io/en/stable/analysis.html#sequence-alignment>

    If ``chain_id`` is not given, the query sequence is aligned to all chains
    in the structure, and the chain with the best alignment is returned.
    """
    st = gemmi.read_structure(str(struct_path), format=gemmi.CoorFormat.Detect)
    st.setup_entities()
    st.assign_label_seq_id()

    query_seq = gemmi.expand_one_letter_sequence(seq, gemmi.ResidueKind.AA)
    model = st[model_id]
    blosum62 = gemmi.AlignmentScoring("b")

    best_score = float("-inf")
    best_aln: gemmi.AlignmentResult | None = None
    best_chain_id: str = ""
    for chain in model:
        if chain_id is not None and chain.name != chain_id:
            continue
        chain_aln = gemmi.align_sequence_to_polymer(
            query_seq, chain.get_polymer(), gemmi.PolymerType.PeptideL, blosum62
        )
        if chain_aln.score > best_score:
            best_score = chain_aln.score
            best_aln = chain_aln
            # best_seq = [
            #     gemmi.find_tabulated_residue(r.name).one_letter_code.upper()
            #     for r in chain
            #     if r.entity_type is gemmi.EntityType.Polymer
            # ]
            best_chain_id = chain.name

            # Find the start and end indices of the aligned region
            query_idx, template_idx = cigar_to_indices(
                query_start=1,
                subject_start=1,
                cigar=best_aln.cigar_str(),
                index_offset=0,
            )

    if best_aln is None:
        raise ValueError("No valid alignment found.")
    return GemmiAlignmentResult(
        raw=best_aln,
        struct_chain_id=best_chain_id,
        query_idx=query_idx,
        struct_idx=template_idx,
    )
