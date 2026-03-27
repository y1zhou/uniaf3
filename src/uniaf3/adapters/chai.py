"""Adapter for converting between UniAF3Config and Chai-1 config."""

from __future__ import annotations

from pathlib import Path

import polars as pl

from uniaf3.adapters._helpers import (
    ensure_list,
    err_unsupported_feature,
    warn_lossy_conversion,
)
from uniaf3.constant import PDB_SERVER_URL
from uniaf3.msa import align_seq_to_structure, cigar_to_indices, parse_m8_file
from uniaf3.schema.base import (
    Atom,
    AuxiliaryParams,
    ContactRestraint,
    CovalentBond,
    Glycan,
    Ligand,
    PocketRestraint,
    Polymer,
    PolymerType,
    ProteinSeq,
    SequenceModification,
    StructuralTemplate,
    UniAF3Config,
)
from uniaf3.schema.chai import (
    ChaiConfig,
    ChaiEntity,
    ChaiEntityType,
    ChaiRestraint,
    ChaiRestraintType,
)
from uniaf3.utils import (
    download_files,
    hash_sequence,
    int_to_letters,
    normalize_out_dir,
)
from uniaf3.vendor.chai1_fasta import constituents_of_modified_fasta


def _find_or_reconstruct_m8(
    protein_seqs: list[ProteinSeq], output_dir: Path
) -> str | None:
    """Find the original pdb70.m8 template hits file or reconstruct one.

    Strategy:
    1. Try to find the original pdb70.m8 file relative to the MSA paths.
       If found, remap query IDs from integer (101, 102, ...) to sequence
       hashes, as Chai-1 expects.
    2. If not found, reconstruct a minimal m8 file from StructuralTemplate data
       with placeholder scoring fields.

    Returns:
        Path to the remapped/reconstructed m8 file, or None if not possible.

    """
    out_path = output_dir / "all_chain_templates.m8"

    # Strategy 1: Find original m8 file from any protein's MSA path
    m8_path: Path | None = None
    for seq in protein_seqs:
        if seq.paired_msa is not None:
            candidate = Path(seq.paired_msa).parent.parent / "pdb70.m8"
            if candidate.exists():
                m8_path = candidate
                break

    if m8_path is not None:
        all_templates = parse_m8_file(m8_path)
        # Remap integer query IDs (101, 102, ...) to sequence hashes
        unique_seqs = list(dict.fromkeys(seq.sequence for seq in protein_seqs))
        query_map = {
            str(101 + i): hash_sequence(seq) for i, seq in enumerate(unique_seqs)
        }
        all_templates = all_templates.with_columns(
            pl.col("query_id").cast(pl.Utf8).replace_strict(query_map)
        )
        all_templates.write_csv(out_path, include_header=False, separator="\t")
        return str(out_path)

    # Strategy 2: Reconstruct from StructuralTemplate objects
    # Note that this alignment differs from the MMSeqs2 algorithm, so scores
    # will not be exact
    rows: list[dict] = []
    for seq in protein_seqs:
        if not seq.templates:
            continue
        for tmpl in seq.templates:
            # Extract PDB ID and chain from the template path
            tmpl_filename = Path(tmpl.path).name
            # Handle .cif.gz, .pdb.gz, .cif, .pdb
            pdb_id = tmpl_filename.split(".")[0].lower()
            tmpl_chain = (
                tmpl.template_chains[0] if tmpl.template_chains is not None else None
            )

            try:
                tmpl_alignment = align_seq_to_structure(
                    seq.sequence, tmpl.path, tmpl_chain
                )
            except (FileNotFoundError, OSError) as e:
                warn_lossy_conversion(
                    f"Failed to align sequence to template {tmpl.path}; "
                    f"error: {e}. Skipping this template in m8 reconstruction."
                )
                continue
            subject_id = f"{pdb_id}_{tmpl_alignment.struct_chain_id}"
            q_start, q_end = tmpl_alignment.query_idx[0], tmpl_alignment.query_idx[-1]
            rows.append(
                {
                    "query_id": seq.seq_hash,
                    "subject_id": subject_id,
                    "pident": tmpl_alignment.raw.calculate_identity(),
                    "length": q_end - q_start + 1,
                    "mismatch": tmpl_alignment.raw.match_string.count("."),
                    "gapopen": 0,  # not important
                    "query_start": q_start,
                    "query_end": q_end,
                    "subject_start": tmpl_alignment.struct_idx[0],
                    "subject_end": tmpl_alignment.struct_idx[-1],
                    "evalue": 1.0 / tmpl_alignment.raw.score,  # approx.
                    "bitscore": tmpl_alignment.raw.score,  # approx.
                    "comment": tmpl_alignment.raw.cigar_str(),
                }
            )

    if not rows:
        return None

    warn_lossy_conversion(
        "UniAF3 StructuralTemplate objects were reconstructed into Chai m8 "
        "format with placeholder scoring fields (pident, evalue, bitscore); "
        "template ranking may differ from the original search results."
    )
    df = pl.DataFrame(rows)
    df.write_csv(out_path, include_header=False, separator="\t")
    return str(out_path)


def to_chai(
    config: UniAF3Config, msa_dir: str | Path | None = None, strict: bool = False
) -> ChaiConfig:
    """Convert a UniAF3Config to a Chai-1 config.

    Lossy terms:

    - CCD ligand IDs are not supported in Chai FASTA format.
    - Chain IDs are not preserved. Chai uses A, B, ..., Z, AA, AB, ... outputs.
    - MSA A3M files are converted to Chai's .aligned.pqt Parquet format;
      header information is partially lost (source database is heuristic).
    - Templates are passed via an m8 file; if the original pdb70.m8 is not
      available, a minimal file is reconstructed with placeholder scores.
    """
    entities: list[ChaiEntity] = []
    if any(len(ensure_list(seq.id)) > 1 for seq in config.sequences):
        warn_lossy_conversion(
            "UniAF3Config.sequences[*].id with multiple chain IDs is expanded to multiple ChaiEntity rows because Chai FASTA has no count field."
        )
    entity_types: dict[str, ChaiEntityType] = {}
    for seq in config.sequences:
        ids = ensure_list(seq.id)

        if isinstance(seq, Polymer):
            if isinstance(seq, ProteinSeq) or (
                isinstance(seq, Polymer) and seq.polymer_type == PolymerType.Protein
            ):
                etype = ChaiEntityType.Protein
            elif seq.polymer_type == PolymerType.DNA:
                etype = ChaiEntityType.DNA
            elif seq.polymer_type == PolymerType.RNA:
                etype = ChaiEntityType.RNA
            else:
                raise ValueError(
                    f"Unsupported polymer type for Chai conversion: {seq.polymer_type}"
                )

            # Chai-1 inlines modifications using CCD codes in parentheses
            seq_list = list(seq.sequence)
            if seq.modifications:
                for mod in seq.modifications:
                    seq_list[mod.position - 1] = f"({mod.ccd})"
            seq_str = "".join(seq_list)

            for chain_id in ids:
                entity_types[chain_id] = etype
                entities.append(
                    ChaiEntity(
                        entity_type=etype, entity_name=chain_id, sequence=seq_str
                    )
                )
        elif isinstance(seq, Ligand):
            ids = ensure_list(seq.id)
            if seq.smiles is not None:
                lig_seq = seq.smiles

            # Chai does not support CCD ligands, but we can attempt to look up the
            # corresponding SMILES if a single CCD code is provided. If multiple CCD codes are provided, issue warning
            elif seq.ccd is not None and len(seq.ccd) == 1:
                import polars as pl

                from uniaf3.vendor.ccd import CCD_LIB

                lig_ccd = seq.ccd[0]
                lig_smiles = CCD_LIB.filter(
                    pl.col("CCD") == pl.lit(lig_ccd)
                ).get_column("SMILES")
                if lig_smiles.len() == 0:
                    err_unsupported_feature(
                        strict,
                        f"CCD ligand {lig_ccd} not found in CCD library.",
                    )
                    continue
                else:
                    lig_seq = lig_smiles.item()
                    warn_lossy_conversion(
                        f"UniAF3 Ligand.ccd '{lig_ccd}' is converted to ChaiEntity.sequence SMILES; original CCD identity is not preserved in FASTA."
                    )

            else:
                err_unsupported_feature(
                    strict, "Multi-CCD ligands are not supported in Chai."
                )
                continue
            for chain_id in ids:
                entity_types[chain_id] = ChaiEntityType.Ligand
                entities.append(
                    ChaiEntity(
                        entity_type=ChaiEntityType.Ligand,
                        entity_name=chain_id,
                        sequence=lig_seq,
                    )
                )
        elif isinstance(seq, Glycan):
            ids = ensure_list(seq.id)
            for chain_id in ids:
                entity_types[chain_id] = ChaiEntityType.Glycan
                entities.append(
                    ChaiEntity(
                        entity_type=ChaiEntityType.Glycan,
                        entity_name=chain_id,
                        sequence=seq.chai_str,
                    )
                )

    # Map original chain IDs to Chai-ordered chain IDs
    entity_id_map = {
        entity.entity_name: int_to_letters(idx)
        for idx, entity in enumerate(entities, start=1)
    }

    # Restraints → Chai CSV restraints
    restraints: list[ChaiRestraint] = []
    restraint_idx: int = 0
    for r in config.covalent_bonds or []:
        res_idx: list[str] = []
        for atom in [r.atom1, r.atom2]:
            if entity_types[atom.chain_id] in {
                ChaiEntityType.Ligand,
                ChaiEntityType.Glycan,
            }:
                res_idx.append(f"@{atom.atom_name}")
            else:
                if atom.residue_name is None:
                    raise ValueError(
                        f"Missing residue name for covalent bond atom: {atom}"
                    )
                res_idx.append(
                    f"{atom.residue_name}{atom.residue_idx}@{atom.atom_name}"
                )

        restraints.append(
            ChaiRestraint(
                restraint_id=f"restraint{restraint_idx}",
                connection_type=ChaiRestraintType.Covalent,
                chainA=entity_id_map[r.atom1.chain_id],
                res_idxA=res_idx[0],
                chainB=entity_id_map[r.atom2.chain_id],
                res_idxB=res_idx[1],
                max_distance_angstrom=0.0,
                comment=r.description,
            )
        )
        restraint_idx += 1
    for r in config.contact_restraints or []:
        for atom in [r.token1, r.token2]:
            if entity_types[atom.chain_id] not in {
                ChaiEntityType.Protein,
                ChaiEntityType.DNA,
                ChaiEntityType.RNA,
            }:
                raise ValueError(
                    f"Contact restraints are only supported between protein/DNA/RNA entities in Chai conversion: {atom}"
                )
            if atom.residue_name is None:
                raise ValueError(
                    f"Missing residue name for contact restraint token: {atom}"
                )
        restraints.append(
            ChaiRestraint(
                restraint_id=f"restraint{restraint_idx}",
                connection_type=ChaiRestraintType.Contact,
                chainA=entity_id_map[r.token1.chain_id],
                res_idxA=f"{r.token1.residue_name}{r.token1.residue_idx}",
                chainB=entity_id_map[r.token2.chain_id],
                res_idxB=f"{r.token2.residue_name}{r.token2.residue_idx}",
                max_distance_angstrom=r.max_distance,
                min_distance_angstrom=r.min_distance,
                comment=r.description,
            )
        )
        restraint_idx += 1
    for r in config.pocket_restraints or []:
        for t in r.contact_tokens:
            if t.residue_name is None:
                raise ValueError(
                    f"Missing residue name for pocket restraint token: {t}"
                )
            restraints.append(
                ChaiRestraint(
                    restraint_id=f"restraint{restraint_idx}",
                    connection_type=ChaiRestraintType.Pocket,
                    chainA=entity_id_map[r.binder_chain],
                    res_idxA=None,
                    chainB=entity_id_map[t.chain_id],
                    res_idxB=f"{t.residue_name}{t.residue_idx}",
                    max_distance_angstrom=r.max_distance,
                    min_distance_angstrom=r.min_distance,
                    comment=r.description,
                )
            )
            restraint_idx += 1

    # --- MSA handling ---
    chai_msa_directory: str | None = None
    for seq in config.sequences:
        if not (isinstance(seq, ProteinSeq) and seq.unpaired_msa is not None):
            continue

        if msa_dir is not None:
            from uniaf3.vendor.chai1_msa import a3m_to_aligned_pqt

            msa_dir_path = normalize_out_dir(msa_dir)

            a3m_to_aligned_pqt(
                single_a3m_path=seq.unpaired_msa,
                pair_a3m_path=seq.paired_msa,
                output_dir=msa_dir_path,
                query_sequence=seq.sequence,
            )
            chai_msa_directory = str(msa_dir_path)
        else:
            warn_lossy_conversion(
                "ProteinSeq MSA data (msa_dir/unpaired_msa/paired_msa) cannot be "
                "converted to Chai format without an output msa_dir parameter; "
                f"MSA information is dropped: {seq}"
            )

    # --- Template handling ---
    chai_template_hits_path: str | None = None

    protein_seqs_with_templates = [
        seq for seq in config.sequences if isinstance(seq, ProteinSeq) and seq.templates
    ]
    if protein_seqs_with_templates:
        if msa_dir is not None:
            msa_dir_path = normalize_out_dir(msa_dir)

            # Warn about Boltz-specific fields that are not representable
            has_boltz_fields = any(
                tmpl.boltz_enable_force or tmpl.boltz_template_threshold is not None
                for seq in protein_seqs_with_templates
                for tmpl in (seq.templates or [])
            )
            if has_boltz_fields:
                warn_lossy_conversion(
                    "StructuralTemplate.{boltz_enable_force,boltz_template_threshold} "
                    "are not represented in Chai's template format."
                )

            chai_template_hits_path = _find_or_reconstruct_m8(
                protein_seqs_with_templates, msa_dir_path
            )
        else:
            warn_lossy_conversion(
                "ProteinSeq.templates cannot be converted to Chai's m8 format "
                "without an output msa_dir parameter; template information is dropped."
            )

    seeds = config.aux.seeds
    if len(seeds) > 1:
        warn_lossy_conversion(
            "Multiple seeds provided in UniAF3Config.aux.seeds; "
            "ChaiConfig.seed can only take a single value. "
            f"Using the first seed {seeds[0]} for ChaiConfig."
        )

    return ChaiConfig(
        entities=entities,
        restraints=restraints or None,
        num_trunk_recycles=config.aux.num_trunk_recycles,
        num_diffn_timesteps=config.aux.num_diffn_timesteps,
        num_diffn_samples=config.aux.num_diffn_samples,
        num_trunk_samples=config.aux.num_trunk_samples,
        seed=seeds[0],
        msa_directory=chai_msa_directory,
        template_hits_path=chai_template_hits_path,
    )


def _parse_chai_res_idx(chain: str, res_idx: str | None) -> Atom:
    """Parse a Chai-style residue index string into an Atom object.

    The format is ``<residue_name><position>[@atom_name]``
    (e.g. ``A219``, ``D45@CB``).
    """
    atom_name: str | None = None
    residue_name: str | None = None
    residue_idx: int = 0
    if res_idx is None:
        return Atom(
            chain_id=chain,
            residue_idx=residue_idx,
            atom_name=atom_name,
            residue_name=residue_name,
        )
    if "@" in res_idx:
        res_part, atom_name = res_idx.split("@")
    else:
        res_part = res_idx

    if res_part:
        # Extract numeric suffix as residue index
        residue_name, residue_idx = res_part[0], int(res_part[1:])

    return Atom(
        chain_id=chain,
        residue_idx=residue_idx,
        atom_name=atom_name,
        residue_name=residue_name,
    )


def _parse_chai_polymer_modifications(
    seq: str,
) -> tuple[str, list[SequenceModification] | None]:
    """Parse inline modifications from a Chai-style polymer sequence."""
    if "(" not in seq:
        return seq, None

    import gemmi

    tokens = constituents_of_modified_fasta(seq)
    modifications: list[SequenceModification] = []
    canonical_seq: list[str] = []
    for i, token in enumerate(tokens, start=1):
        if len(token) == 1:
            canonical_seq.append(token)
            continue

        modifications.append(SequenceModification(position=i, ccd=token))

        # Try to map the modified residue to a canonical one-letter code using the CCD.
        # If no mapping is found, put down "X"
        ccd_related_token = gemmi.find_tabulated_residue(token)
        if canonical_token := ccd_related_token.one_letter_code.strip():
            canonical_seq.append(canonical_token)
        else:
            warn_lossy_conversion(
                f"Chai inline modification token '{token}' has no canonical one-letter mapping; UniAF3 polymer sequence uses 'X'."
            )
            canonical_seq.append("X")

    return "".join(canonical_seq), modifications or None


def from_chai(config: ChaiConfig, msa_dir: str | Path | None = None) -> UniAF3Config:
    """Convert a Chai-1 config to a UniAF3Config."""
    sequences: list[Polymer | ProteinSeq | Ligand | Glycan] = []
    prot_seq_hashes: dict[str, int] = {}
    for i, entity in enumerate(config.entities, start=1):
        if entity.entity_type == ChaiEntityType.Protein:
            seq, mods = _parse_chai_polymer_modifications(entity.sequence)
            prot_seq = ProteinSeq(
                polymer_type=PolymerType.Protein,
                id=int_to_letters(i),
                description=entity.entity_name,
                sequence=seq,
                modifications=mods,
            )
            sequences.append(prot_seq)
            prot_seq_hashes[prot_seq.seq_hash] = len(sequences) - 1
        elif entity.entity_type == ChaiEntityType.DNA:
            seq, mods = _parse_chai_polymer_modifications(entity.sequence)
            sequences.append(
                Polymer(
                    polymer_type=PolymerType.DNA,
                    id=int_to_letters(i),
                    description=entity.entity_name,
                    sequence=seq,
                    modifications=mods,
                )
            )
        elif entity.entity_type == ChaiEntityType.RNA:
            seq, mods = _parse_chai_polymer_modifications(entity.sequence)
            sequences.append(
                Polymer(
                    polymer_type=PolymerType.RNA,
                    id=int_to_letters(i),
                    description=entity.entity_name,
                    sequence=seq,
                    modifications=mods,
                )
            )
        elif entity.entity_type == ChaiEntityType.Ligand:
            warn_lossy_conversion(
                "ChaiEntityType.Ligand sequence is imported into UniAF3Config.sequences[*].Ligand.smiles; CCD identity cannot be recovered reliably."
            )
            lig = Ligand(
                id=int_to_letters(i),
                description=entity.entity_name,
                smiles=entity.sequence,
            )
            sequences.append(lig)
        elif entity.entity_type == ChaiEntityType.Glycan:
            glycan = Glycan(
                id=int_to_letters(i),
                description=entity.entity_name,
                chai_str=entity.sequence,
            )
            sequences.append(glycan)

    covalent_bonds: list[CovalentBond] = []
    contact_restraints: list[ContactRestraint] = []
    pocket_restraints: dict[str, PocketRestraint] = {}
    if config.restraints:
        for cr in config.restraints:
            atom1 = _parse_chai_res_idx(cr.chainA, cr.res_idxA)
            atom2 = _parse_chai_res_idx(cr.chainB, cr.res_idxB)
            if cr.connection_type == ChaiRestraintType.Covalent:
                covalent_bonds.append(
                    CovalentBond(
                        atom1=atom1,
                        atom2=atom2,
                        description=cr.comment,
                    )
                )
            elif cr.connection_type == ChaiRestraintType.Contact:
                contact_restraints.append(
                    ContactRestraint(
                        token1=atom1,
                        token2=atom2,
                        max_distance=cr.max_distance_angstrom,
                        min_distance=cr.min_distance_angstrom,
                        description=cr.comment,
                    )
                )
            elif cr.connection_type == ChaiRestraintType.Pocket:
                binder_chain = cr.chainA if cr.res_idxA is None else cr.chainB
                contact_token = atom2 if cr.res_idxA is None else atom1
                if binder_chain in pocket_restraints:
                    # If multiple contact tokens map to the same binder chain, combine them into a single restraint
                    pocket_restraints[binder_chain].contact_tokens.append(contact_token)
                else:
                    pocket_restraints[binder_chain] = PocketRestraint(
                        binder_chain=binder_chain,
                        contact_tokens=[contact_token],
                        max_distance=cr.max_distance_angstrom,
                        min_distance=cr.min_distance_angstrom,
                        description=cr.comment,
                    )

            else:
                continue

    if config.msa_directory is not None:
        # Dump Chai parquet MSAs back to ColabFold A3M files
        # Note that the query sequence only appears once at the top of the parquet,
        # so the final #records in the single and paired A3Ms should be #pqt+1.
        msa_path = Path(config.msa_directory)
        if not msa_path.exists():
            raise ValueError(f"Chai MSA directory does not exist: {msa_path}")
        if msa_dir is None:
            raise ValueError(
                "ChaiConfig.msa_directory is provided but no msa_dir specified."
            )
        msa_out_path = normalize_out_dir(msa_dir)
        for prot_seq_hash, seq_idx in prot_seq_hashes.items():
            prot_seq = sequences[seq_idx]
            if not isinstance(prot_seq, ProteinSeq):
                raise ValueError(
                    f"Expected ProteinSeq for hash {prot_seq_hash}, got {type(prot_seq)}"
                )

            seq_msa_file = msa_path / f"{prot_seq_hash}.aligned.pqt"
            if not seq_msa_file.exists():
                raise ValueError(
                    f"Expected MSA for {prot_seq} not found at {seq_msa_file}"
                )

            # Reverse the Chai parquet MSA back to single/paired A3M files
            seq_msa_df = pl.scan_parquet(seq_msa_file)

            query_entry = (
                seq_msa_df.filter(pl.col("source_database") == pl.lit("query"))
                .select("comment", "sequence")
                .collect()
            )
            if query_entry.height > 1:
                raise ValueError(
                    f"Multiple query entries found in {seq_msa_file}: {query_entry}"
                )
            q = query_entry.to_dicts()[0]
            q_str = f">{q['comment']}\n{q['sequence']}\n"

            single_msa_df = (
                seq_msa_df.filter(
                    (pl.col("pairing_key") == pl.lit(""))
                    & (pl.col("source_database") != pl.lit("query"))
                )
                .select("comment", "sequence")
                .collect()
            )
            if single_msa_df.height > 0:
                single_a3m_path = msa_out_path / f"{prot_seq_hash}.single.a3m"
                with single_a3m_path.open("w") as f:
                    f.write(q_str)
                    f.writelines(
                        f">{r['comment']}\n{r['sequence']}\n"
                        for r in single_msa_df.iter_rows(named=True)
                    )
                prot_seq.unpaired_msa = str(single_a3m_path)

            paired_msa_df = (
                seq_msa_df.filter(
                    (pl.col("pairing_key") != pl.lit(""))
                    & (pl.col("source_database") != pl.lit("query"))
                )
                .select("comment", "sequence")
                .collect()
            )
            if paired_msa_df.height > 0:
                paired_a3m_path = msa_out_path / f"{prot_seq_hash}.pair.a3m"
                with paired_a3m_path.open("w") as f:
                    f.write(q_str)
                    f.writelines(
                        f">{r['comment']}\n{r['sequence']}\n"
                        for r in paired_msa_df.iter_rows(named=True)
                    )
                prot_seq.paired_msa = str(paired_a3m_path)

    if config.template_hits_path is not None:
        template_hits_path = Path(config.template_hits_path)
        if not template_hits_path.exists():
            raise ValueError(
                f"Chai template hits file does not exist: {template_hits_path}"
            )
        if template_hits_path.suffix != ".m8":
            raise ValueError(
                f"Chai template hits file must be in .m8 format: {template_hits_path}"
            )

        if msa_dir is None:
            raise ValueError(
                "ChaiConfig.template_hits_path is provided but no msa_dir specified."
            )
        tmpl_files_dir = normalize_out_dir(msa_dir, "templates")

        # Parse the m8 file and populate UniAF3Config StructuralTemplate objects
        # Note that Chai parses the top 4 templates per chain, so there's
        # no need to download more than that
        m8_df = (
            parse_m8_file(template_hits_path)
            .group_by("query_id", maintain_order=True)
            .head(4)
            .with_columns(pl.col("subject_id").str.split("_"))
            .with_columns(
                pl.col("subject_id")
                .list.first()
                .str.to_uppercase()
                .alias("subject_pdb_id"),
                pl.col("subject_id").list.last().alias("subject_chain_id"),
            )
        )
        fetch_template_tasks: dict[str, Path] = {}
        for r in m8_df.iter_rows(named=True):
            query_hash = r["query_id"]
            if query_hash not in prot_seq_hashes:
                raise ValueError(
                    f"Template hit query_id {query_hash} not found among protein sequences."
                )
            seq_idx = prot_seq_hashes[query_hash]
            prot_seq = sequences[seq_idx]
            if not isinstance(prot_seq, ProteinSeq):
                raise ValueError(
                    f"Expected ProteinSeq for template hit with query_id {query_hash}, got {type(prot_seq)}"
                )

            tmpl: list[StructuralTemplate] = prot_seq.templates or []
            q_idx, tmpl_idx = cigar_to_indices(
                r["query_start"], r["subject_start"], r["cigar"]
            )
            tmpl_pdb_id = r["subject_pdb_id"]
            tmpl_path = tmpl_files_dir / f"{tmpl_pdb_id}.cif.gz"
            tmpl.append(
                StructuralTemplate(
                    path=str(tmpl_path),
                    query_idx=q_idx,
                    template_idx=tmpl_idx,
                    query_chains=ensure_list(prot_seq.id),
                    template_chains=[r["subject_chain_id"]],
                )
            )
            prot_seq.templates = tmpl
            fetch_template_tasks[f"{PDB_SERVER_URL}/{tmpl_pdb_id}.cif.gz"] = tmpl_path
        if fetch_template_tasks:
            warn_lossy_conversion(
                "Chai template hits are represented as StructuralTemplate objects "
                "with path pointing to downloaded CIF files; original template metadata "
                "(e.g. alignment scores) is not preserved."
            )
            download_files(
                fetch_template_tasks,
                force=False,
                num_retries=3,
                progress_bar_desc="Template CIFs for Chai",
            )

    aux = AuxiliaryParams(
        seeds=[config.seed] if config.seed is not None else [42],
        num_trunk_recycles=config.num_trunk_recycles,
        num_diffn_timesteps=config.num_diffn_timesteps,
        num_diffn_samples=config.num_diffn_samples,
        num_trunk_samples=config.num_trunk_samples,
    )
    if config.seed is None:
        warn_lossy_conversion(
            "ChaiConfig.seed is missing; UniAF3Config.aux.seeds defaults to [42]."
        )

    return UniAF3Config(
        sequences=sequences,
        covalent_bonds=covalent_bonds or None,
        contact_restraints=contact_restraints or None,
        pocket_restraints=list(pocket_restraints.values()) or None,
        aux=aux,
    )
