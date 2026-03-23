"""Query MSA and templates for protein sequences."""

import tempfile
from pathlib import Path

import polars as pl
import requests
from platformdirs import PlatformDirs
from pydantic import BaseModel
from tqdm.rich import tqdm

from uniaf3.schema import UniAF3Config
from uniaf3.schema.base import ProteinSeq, StructuralTemplate
from uniaf3.utils import hash_sequence
from uniaf3.vendor.chai1_msa import generate_colabfold_msas, parse_m8_file
from uniaf3.vendor.colabfold_msa import run_mmseqs2


class ColabFoldResponse(BaseModel):
    """Response from ColabFold API."""

    query_key: str  # sha256(seq1|seq2|...)
    msa_dir: Path  # directory containing MSA files and template hits
    protein_seqs: list[str]
    seq_hashes: list[str]  # sha256(seq1), sha256(seq2), ...
    single_msas: list[Path]
    paired_msas: list[Path]
    templates_m8_file: Path | None = None


def query_colabfold(
    seqs: list[str],
    msa_cache_dir: str | Path | None = None,
    use_templates: bool = False,
    force: bool = False,
    **kwargs,
) -> ColabFoldResponse:
    """Query ColabFold API for MSAs and templates.

    Note that if ``kwargs`` are passed with filters or pairing strategies
    specified, the cache results could be invalid. In that case, set ``force=True`` to ignore cache and re-query the server.

    Args:
        seqs: List of protein sequences. The order matters, as the concatenated sequences
            will be used as the cache key under ``msa_cache_dir``.
        msa_cache_dir: Directory to cache MSA files. Defaults to $XDG_CACHE_HOME/uniaf3/colabfold_msas/.
        use_templates: Whether to search for templates.
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
                templates/ (TODO: if use_templates is True, currently not implemented)

        ```

    """
    if not seqs:
        raise ValueError("No protein sequences for MSA generation; this is a no-op.")
    if msa_cache_dir is None:
        msa_cache_dir = PlatformDirs("uniaf3").user_cache_path / "colabfold_msas"

    seqs_hash = hash_sequence("|".join(seqs).upper())
    msa_dir = Path(msa_cache_dir).expanduser() / seqs_hash[:2] / seqs_hash
    msa_dir.mkdir(parents=True, exist_ok=True)
    msa_dir = msa_dir.resolve()

    a3ms_dir = msa_dir / "a3ms"
    a3ms_dir.mkdir(exist_ok=True)

    seq_hashes = [hash_sequence(s.upper()) for s in seqs]

    with tempfile.TemporaryDirectory() as tmp_dir_path:
        tmp_dir = Path(tmp_dir_path)
        mmseqs_paired_dir = tmp_dir / "mmseqs2_paired"
        mmseqs_paired_dir.mkdir()

        mmseqs_dir = tmp_dir / "mmseqs2"
        mmseqs_dir.mkdir()

        # Run paired MSA search
        num_seqs = len(seqs)
        if num_seqs > 1:
            # Skip run if a cached tarball already exists, but we reuse
            # the run_mmseqs2 function to load the a3m into memory
            paired_a3m_files = [
                a3ms_dir / f"{seq_hash}.pair.a3m" for seq_hash in seq_hashes
            ]
            if force or not all(p.exists() for p in paired_a3m_files):
                paired_msas, _ = run_mmseqs2(
                    x=seqs,
                    prefix=mmseqs_paired_dir,
                    use_pairing=True,
                    use_templates=False,
                    **kwargs,
                )
                for f, paired_msa in zip(paired_a3m_files, paired_msas, strict=True):
                    if force or not f.exists():
                        f.write_text(paired_msa)
        else:
            paired_msas = [""] * num_seqs

        # Run MSA search without pairing to get more hits for each chain
        single_a3m_files = [
            a3ms_dir / f"{seq_hash}.single.a3m" for seq_hash in seq_hashes
        ]
        expected_tmpl_m8_file = msa_dir / "pdb70.m8"
        if (
            force
            or not all(p.exists() for p in single_a3m_files)
            or (use_templates and not expected_tmpl_m8_file.exists())
        ):
            per_chain_msas, template_hits_m8_file = run_mmseqs2(
                x=seqs,
                prefix=mmseqs_dir,
                use_pairing=False,
                use_templates=use_templates,
                **kwargs,
            )
            for f, single_msa in zip(single_a3m_files, per_chain_msas, strict=True):
                if force or not f.exists():
                    f.write_text(single_msa)
            if template_hits_m8_file is not None:
                if force or not expected_tmpl_m8_file.exists():
                    import shutil

                    shutil.copyfile(template_hits_m8_file, expected_tmpl_m8_file)

            # TODO: fetch templates from RCSB using template_hits_m8_file

    return ColabFoldResponse(
        query_key=seqs_hash,
        msa_dir=msa_dir,
        protein_seqs=seqs,
        seq_hashes=seq_hashes,
        single_msas=[a3ms_dir / f"{seq_hash}.single.a3m" for seq_hash in seq_hashes],
        paired_msas=[a3ms_dir / f"{seq_hash}.pair.a3m" for seq_hash in seq_hashes],
        templates_m8_file=expected_tmpl_m8_file if use_templates else None,
    )


# Adapted from antid.io.struct.RCSBDownloader
class RCSBDownloader:
    """Download structure files from RCSB."""

    def __init__(
        self,
        out_dir: str | Path,
        make_subdir: bool = False,
        req_session: requests.Session | None = None,
        timeout: int = 10,
    ):
        """Initialize the downloader.

        Args:
            out_dir: Directory to save downloaded files.
            make_subdir: If True, always use the two characters in the middle of the PDB
                ID as subdirectory names. This is useful when downloading a large number
                of PDB files to avoid too many files in a single directory.
            req_session: Optional requests.Session object.
            timeout: Timeout for requests in seconds.

        """
        self.out_dir = Path(out_dir).expanduser().resolve()
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.subdir = make_subdir
        self.session = requests.Session() if req_session is None else req_session
        self.timeout = timeout

    def fetch_pdb(
        self, pdb_id: str, file_type: str = "bio", fallback_to_cif: bool = True
    ) -> Path:
        """Download a PDB file from RCSB.

        Args:
            pdb_id: The PDB ID.
            file_type: Bio-assembly1 (bio) or asymmetric unit (asu).
            fallback_to_cif: If True, fall back to downloading the mmCIF file if the PDB
            file is not found. For details, see https://www.rcsb.org/docs/general-help/structures-without-legacy-pdb-format-files

        Returns:
            Path to the downloaded file.

        """
        pdb_id = pdb_id.upper()
        file_type = self._check_file_type(file_type)
        out_dir = (
            Path(self.out_dir / file_type / pdb_id[-3:-1])
            if self.subdir
            else self.out_dir / file_type
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        gz_pdb_file = out_dir / f"{pdb_id}.pdb.gz"
        if gz_pdb_file.exists():
            return gz_pdb_file

        pdb_url = (
            f"https://files.rcsb.org/download/{pdb_id}.pdb1.gz"
            if file_type == "bio"
            else f"https://files.rcsb.org/download/{pdb_id}.pdb.gz"
        )
        r = self.session.get(pdb_url, timeout=self.timeout)
        if r.status_code == 404 and fallback_to_cif:
            # Try mmCIF file if the PDB is nonexistent
            print(f"[Warning] PDB for {pdb_id} not found, falling back to mmCIF.")
            return self.fetch_mmcif(pdb_id)

        r.raise_for_status()
        with open(gz_pdb_file, "wb") as f:
            f.write(r.content)
        return gz_pdb_file

    def fetch_mmcif(self, pdb_id: str, file_type: str = "bio") -> Path:
        """Download a mmCIF file from RCSB.

        Args:
            pdb_id: The PDB ID.
            file_type: Bio-assembly1 (bio) or asymmetric unit (asu).

        Returns:
            Path to the downloaded file.

        """
        pdb_id = pdb_id.upper()
        file_type = self._check_file_type(file_type)
        out_dir = (
            Path(self.out_dir / file_type / pdb_id[-3:-1])
            if self.subdir
            else Path(self.out_dir / file_type)
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        gz_cif_file = out_dir / f"{pdb_id}.cif.gz"
        if gz_cif_file.exists():
            return gz_cif_file

        cif_url = (
            f"https://files.rcsb.org/download/{pdb_id}-assembly1.cif.gz"
            if file_type == "bio"
            else f"https://files.rcsb.org/download/{pdb_id}.cif.gz"
        )
        r = self.session.get(cif_url, timeout=self.timeout)
        r.raise_for_status()
        with open(gz_cif_file, "wb") as f:
            f.write(r.content)
        return gz_cif_file

    @staticmethod
    def _check_file_type(file_type: str) -> str:
        """Check if the file type is valid."""
        match file_type.lower():
            case "bio" | "biological" | "biological_assembly" | "biological-assembly":
                return "bio"
            case "asu" | "asymmetric" | "asymmetric_unit" | "asymmetric-unit":
                return "asu"
            case _:
                raise ValueError(f"Invalid file type: {file_type}")


def add_msa_to_config(
    conf: UniAF3Config,
    out_dir: str | Path,
    chains: set[str] | None = None,
    search_templates: bool = False,
    template_cache_dir: Path | None = None,
) -> UniAF3Config:
    """Add MSA paths to protein sequences in the config.

    Args:
        conf: UniAF3Config object.
        out_dir: Output directory to store MSA files.
        chains: Set of chain IDs to process. If None, process all protein chains.
        search_templates: Whether to search for templates.
        template_cache_dir: Directory to cache fetched templates. Defaults to
            $XDG_CACHE_HOME/uniaf3/rcsb/ if not set.

    """
    # Figure out which protein sequences to process
    if chains is None:
        chains = {
            c
            for seq in conf.sequences
            if isinstance(seq, ProteinSeq)
            for c in (seq.id if isinstance(seq.id, list) else [seq.id])
        }
    protein_seqs = [
        seq.sequence
        for seq in conf.sequences
        if isinstance(seq, ProteinSeq)
        and any(c in chains for c in (seq.id if isinstance(seq.id, list) else [seq.id]))
    ]

    # Generate MSAs using ColabFold API
    out_path = Path(out_dir).expanduser().resolve()
    out_path.mkdir(parents=True, exist_ok=True)
    _ = generate_colabfold_msas(
        protein_seqs=protein_seqs,
        msa_dir=out_path,
        msa_server_url="https://api.colabfold.com",
        search_templates=search_templates,
        write_a3m_to_msa_dir=True,
    )

    # Fetch templates from RCSB if needed
    # TODO: make dummy m8 file when search_templates is False and custom templates are provided
    template_map: dict[str, list[StructuralTemplate]] = {}
    if search_templates:
        template_cache = (
            PlatformDirs("uniaf3").user_cache_path / "rcsb"
            if template_cache_dir is None
            else template_cache_dir
        )
        template_cache = template_cache.expanduser().resolve()
        template_cache.mkdir(parents=True, exist_ok=True)
        dl = RCSBDownloader(template_cache, make_subdir=True)

        templates_path = out_path / "all_chain_templates.m8"
        templates_df = parse_m8_file(templates_path)

        # Remove duplicates
        templates_df = (
            templates_df.with_columns(
                pl.col("subject_id").str.extract(r"^(\w+)_\w$").alias("subject_pdb_id")
            )
            .unique("subject_pdb_id", maintain_order=True, keep="first")
            .drop("subject_pdb_id")
        )
        templates_df.write_csv(templates_path, include_header=False, separator="\t")

        hash_to_chains: dict[str, list[str]] = {
            seq.seq_hash: (seq.id if isinstance(seq.id, list) else [seq.id])
            for seq in conf.sequences
            if isinstance(seq, ProteinSeq)
        }
        # TODO: no need to fetch all as only the top 4/chain are used
        for r in tqdm(
            templates_df.iter_rows(named=True),
            total=templates_df.height,
            desc="Fetching templates",
        ):
            template_pdb_id, template_chain_id = r["subject_id"].split("_")
            template_pdb_id = template_pdb_id.upper()
            template_cif_path = dl.fetch_mmcif(template_pdb_id, file_type="asu")

            if r["query_id"] not in template_map:
                template_map[r["query_id"]] = []
            template_map[r["query_id"]].append(
                StructuralTemplate(
                    path=str(template_cif_path),
                    query_idx=list(range(r["query_start"] - 1, r["query_end"])),
                    template_idx=list(range(r["subject_start"] - 1, r["subject_end"])),
                    query_chains=hash_to_chains[r["query_id"]],
                    template_chains=[template_chain_id],
                )
            )

    # Update config with MSA paths and templates
    (out_path / "templates").mkdir(parents=True, exist_ok=True)
    for seq in conf.sequences:
        if not isinstance(seq, ProteinSeq):
            continue
        if isinstance(seq.id, str) and seq.id not in chains:
            continue
        elif isinstance(seq.id, list) and not any(c in chains for c in seq.id):
            continue

        seq.msa_dir = str(out_path)

        custom_templates = seq.templates or []
        if seq.seq_hash in template_map:
            seq.templates = custom_templates + template_map[seq.seq_hash]

        # Soft link all templates to out-dir/templates
        for t in seq.templates or []:
            template_path = Path(t.path)
            link_path = out_path / "templates" / template_path.name
            if not link_path.exists():
                link_path.symlink_to(template_path)
            t.path = str(link_path)

    return conf
