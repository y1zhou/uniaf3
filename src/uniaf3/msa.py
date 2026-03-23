"""Query MSA and templates for protein sequences."""

import asyncio
import tempfile
from pathlib import Path

import polars as pl
from platformdirs import PlatformDirs
from pydantic import BaseModel, model_validator

from uniaf3.utils import download_files, hash_sequence
from uniaf3.vendor.colabfold_msa import run_mmseqs2


class ColabFoldResponse(BaseModel):
    """Response from ColabFold API."""

    query_key: str  # sha256(seq1|seq2|...)
    msa_dir: Path  # directory containing MSA files and template hits
    protein_seqs: list[str]
    seq_hashes: list[str]  # sha256(seq1), sha256(seq2), ...
    query_ids: list[int]  # ColabFold query IDs, starting from 101
    single_msas: list[Path]
    paired_msas: list[Path]
    templates_m8_file: Path | None = None

    @model_validator(mode="after")
    def check_list_lengths(self):
        """Validate that the lengths of fields are the same."""
        n = len(self.protein_seqs)
        if not (
            len(self.seq_hashes)
            == len(self.query_ids)
            == len(self.single_msas)
            == len(self.paired_msas)
            == n
        ):
            raise ValueError(
                "The lengths of protein_seqs, seq_hashes, query_ids, single_msas, and paired_msas must be the same."
            )
        return self

    def __getitem__(self, protein_seq: str) -> dict[str, int | str | Path | None]:
        """Get MSA and template paths for a given protein sequence."""
        idx = self.protein_seqs.index(protein_seq)
        return {
            "seq_hash": self.seq_hashes[idx],
            "query_id": self.query_ids[idx],
            "single_msa": self.single_msas[idx],
            "paired_msa": self.paired_msas[idx],
        }


def query_colabfold(
    seqs: list[str],
    msa_cache_dir: str | Path | None = None,
    search_templates: bool = False,
    download_templates: bool = False,
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
        download_templates: Whether to download template files. If True,
            asymmetric unit mmCIF files will be fetched from RCSB.
        template_cache_dir: Directory to cache template hits. Defaults to
            ``$XDG_CACHE_HOME/uniaf3/rcsb/``.
        force: Whether to ignore cache and re-query the server.
        kwargs: Additional args to pass to ``run_mmseqs2``.

    Returns:
        The MSA directory. The MSA directory has the following structure:

        ```
        msa_cache_dir/
          <seqs_hash>[:2]/
            <seqs_hash>/
                a3ms/
                  <seq_hash>.single.a3m
                  <seq_hash>.pair.a3m
                pdb70.m8 (if use_templates is True)
        ```

    """
    # Setup output directories
    if not seqs:
        raise ValueError("No protein sequences for MSA generation; this is a no-op.")
    if msa_cache_dir is None:
        msa_cache_dir = PlatformDirs("uniaf3").user_cache_path / "colabfold_msas"

    seqs_unique = list(dict.fromkeys(x.upper() for x in seqs))  # >=Python 3.7
    query_indices = [101 + seqs_unique.index(seq.upper()) for seq in seqs]

    seqs_hash = hash_sequence("|".join(seqs_unique))
    msa_dir = Path(msa_cache_dir).expanduser() / seqs_hash[:2] / seqs_hash
    msa_dir.mkdir(parents=True, exist_ok=True)
    msa_dir = msa_dir.resolve()

    a3ms_dir = msa_dir / "a3ms"
    a3ms_dir.mkdir(exist_ok=True)

    seq_hashes = [hash_sequence(s.upper()) for s in seqs_unique]
    expected_tmpl_m8_file = msa_dir / "pdb70.m8"

    # Query ColabFold API if cached results do not exist
    with tempfile.TemporaryDirectory() as tmp_dir_path:
        tmp_dir = Path(tmp_dir_path)
        mmseqs_paired_dir = tmp_dir / "mmseqs2_paired"
        mmseqs_paired_dir.mkdir()

        mmseqs_dir = tmp_dir / "mmseqs2"
        mmseqs_dir.mkdir()

        # Run paired MSA search
        # In paired mode, mmseqs2 returns paired a3ms where all a3ms have the same number of rows
        # and each row is already paired to have the same species.
        num_seqs = len(seqs_unique)
        if num_seqs > 1:
            paired_a3m_files = [
                a3ms_dir / f"{seq_hash}.pair.a3m" for seq_hash in seq_hashes
            ]
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
        single_a3m_files = [
            a3ms_dir / f"{seq_hash}.single.a3m" for seq_hash in seq_hashes
        ]

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
    if download_templates:
        if template_cache_dir is None:
            template_cache_dir = PlatformDirs("uniaf3").user_cache_path / "rcsb"
        template_cache_dir = Path(template_cache_dir).expanduser().resolve()

        if not expected_tmpl_m8_file.exists():
            raise FileNotFoundError(
                f"Expected template hits file not found at {expected_tmpl_m8_file}."
            )
        all_templates = parse_m8_file(expected_tmpl_m8_file)
        template_pdb_ids = (
            all_templates.select(
                pl.col("subject_id")
                .str.split("_")
                .list.first()
                .str.to_uppercase()
                .alias("pdb_id")
            )
            .unique()
            .with_columns(
                pl.concat_str(
                    pl.lit("https://files.rcsb.org/download/"),
                    pl.col("pdb_id"),
                    pl.lit(".cif.gz"),  # -assembly1.cif.gz for biological assembly
                ).alias("cif_url")
            )
        )
        asyncio.run(
            download_files(
                {
                    r["cif_url"]: template_cache_dir
                    / r["pdb_id"][-3:-1]
                    / f"{r['pdb_id']}.cif.gz"
                    for r in template_pdb_ids.iter_rows(named=True)
                },
                force=force,
                max_connections=10,
                num_retries=3,
                progress_bar_desc="Downloading templates from RCSB",
            )
        )

    return ColabFoldResponse(
        query_key=seqs_hash,
        msa_dir=msa_dir,
        protein_seqs=seqs_unique,
        seq_hashes=seq_hashes,
        query_ids=query_indices,
        single_msas=[a3ms_dir / f"{seq_hash}.single.a3m" for seq_hash in seq_hashes],
        paired_msas=[a3ms_dir / f"{seq_hash}.pair.a3m" for seq_hash in seq_hashes],
        templates_m8_file=expected_tmpl_m8_file if search_templates else None,
    )


def parse_m8_file(fname: str | Path) -> pl.DataFrame:
    """Parse the m8 alignment format describing template information.

    Inspired by: chai_lab.data.parsing.templates.m8 import parse_m8_file.
    """
    table = (
        pl.scan_csv(
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
                "comment",
            ],
        )
        .sort(by=["query_id", "evalue"])
        .with_columns(
            pl.col(c).cast(pl.Int64)
            for c in ("query_start", "query_end", "subject_start", "subject_end")
        )
        .collect()
    )
    return table
