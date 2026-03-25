"""Adapter for converting between UniAF3Config and Boltz config."""

from __future__ import annotations

from pathlib import Path

import polars as pl

from uniaf3.adapters._helpers import (
    ensure_list,
    err_unsupported_feature,
    warn_lossy_conversion,
)
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
from uniaf3.schema.boltz import (
    BoltzAffinityProperty,
    BoltzBondConstraint,
    BoltzConfig,
    BoltzConstraintEntry,
    BoltzContactConstraint,
    BoltzDNA,
    BoltzLigand,
    BoltzModification,
    BoltzPocketConstraint,
    BoltzPropertyEntry,
    BoltzProtein,
    BoltzRNA,
    BoltzSequenceEntry,
    BoltzTemplate,
)
from uniaf3.utils import hash_sequence
from uniaf3.vendor.chai1_fasta import read_fasta
from uniaf3.vendor.chai1_glycans import _glycan_string_to_sugars_and_bonds


def merge_colabfold_msa_to_csv(
    unpaired_msa_file: str | Path | None,
    paired_msa_file: str | Path | None,
    msa_id: str,
    out_dir: str | Path,
) -> Path:
    """Merge unpaired and paired MSAs into a single CSV file for Boltz.

    Adapted from Boltz's own MSA processing code: boltz.main.compute_msa

    Args:
        unpaired_msa_file: Path to unpaired MSA file in A3M format.
        paired_msa_file: Path to paired MSA file in A3M format.
        msa_id: Output file name. Boltz uses `{target_id}_{entity_id}.csv`.
        out_dir: Directory to save the output CSV file, and a parquet file mapping the
            original headers to the sequences.

    Returns:
        Path to the output CSV file containing the merged MSA.

    """
    out_dir_path = Path(out_dir).expanduser().resolve()
    out_dir_path.mkdir(parents=True, exist_ok=True)
    out_file = out_dir_path / f"{msa_id}.csv"

    # fast fail if unpaired MSA file does not exist
    if unpaired_msa_file is None:
        raise ValueError("Unpaired MSA file must be provided.")
    unpaired_path = Path(unpaired_msa_file).expanduser().resolve()
    if not unpaired_path.exists():
        raise FileNotFoundError(f"Unpaired MSA file not found: {unpaired_path}")

    if paired_msa_file is not None:
        paired_path = Path(paired_msa_file).expanduser().resolve()
        if not paired_path.exists():
            raise FileNotFoundError(f"Paired MSA file not found: {paired_path}")

        paired_fasta = read_fasta(paired_path)

        # ignore headers
        # Boltz also does subsampling here but we skip that for now
        paired_df = (
            pl.DataFrame(paired_fasta)
            .with_row_index(name="key")
            # filter out padding rows (rows that are all gaps)
            .filter(
                pl.col("sequence").str.count_matches("-", literal=True)
                < pl.col("sequence").str.len_bytes()
            )
        )
    else:
        paired_df = pl.DataFrame([], schema=["key", "header", "sequence"])

    # combine paired-unpaired sequences
    unpaired_fasta = read_fasta(unpaired_path)
    if paired_df.height > 0:
        # ignore query seq
        unpaired_fasta = unpaired_fasta[1:]

    combined_df = pl.concat(
        [
            paired_df,
            pl.DataFrame(unpaired_fasta)
            .with_columns(pl.lit(-1).alias("key"))
            .select("key", "header", "sequence"),
        ],
        how="vertical_relaxed",
    )
    combined_df.select("key", "sequence").write_csv(out_file)
    combined_df.select("header", "sequence").write_parquet(
        out_file.with_suffix(".parquet")
    )
    return out_file


def split_boltz_csv_to_a3m(
    csv_file: str | Path, out_dir: str | Path
) -> tuple[Path, Path | None]:
    """Split a Boltz MSA CSV file into unpaired and paired A3M files for UniAF3.

    Args:
        csv_file: Path to Boltz MSA CSV file.
        msa_id: Output file name.
        out_dir: Directory to save the output MSA files. Files `{hash}.single.a3m` and
        `{hash}.pair.a3m` should be created, where `hash` is the SHA256 hash of the
        query sequence.

    Returns:
        Paths to the unpaired and paired MSA A3M files.

    """
    out_dir_path = Path(out_dir).expanduser().resolve()
    out_dir_path.mkdir(parents=True, exist_ok=True)

    headers_df = pl.read_parquet(Path(csv_file).with_suffix(".parquet"))
    msa_df = (
        pl.read_csv(csv_file)
        .join(headers_df, on="sequence", maintain_order="left")
        .select(
            pl.when(pl.col("key") == pl.lit(-1))
            .then(pl.lit("unpaired"))
            .otherwise(pl.lit("paired"))
            .alias("msa_type"),
            "header",
            "sequence",
            "key",
        )
    )

    # Dump .single.a3m MSA
    unpaired_df = msa_df.filter(pl.col("msa_type") == pl.lit("unpaired"))
    if unpaired_df.height == 0:
        raise ValueError(f"No unpaired sequences found in MSA CSV: {csv_file}")

    try:
        # Get the query sequence, calculate its hash, and add it to the unpaired MSA
        query_seq_entry = msa_df.filter(pl.col("key") == pl.lit(0))
        prefix = hash_sequence(query_seq_entry.item(0, "sequence"))
    except Exception as e:
        raise ValueError(
            f"Failed to extract query sequence from MSA CSV: {csv_file}"
        ) from e

    unpaired_df = pl.concat([query_seq_entry, unpaired_df], how="vertical")
    unpaired_a3m_path = out_dir_path / f"{prefix}.single.a3m"
    with unpaired_a3m_path.open("w") as f:
        for r in unpaired_df.select("header", "sequence").iter_rows(named=True):
            f.write(f">{r['header']}\n{r['sequence']}\n")

    # Dump .pair.a3m MSA
    paired_df = (
        msa_df.filter(pl.col("key") != pl.lit(-1))
        .join(headers_df, on="sequence", maintain_order="left")
        .select("header", "sequence")
    )
    if paired_df.height != 0:
        paired_a3m_path = out_dir_path / f"{prefix}.pair.a3m"
        with paired_a3m_path.open("w") as f:
            for r in paired_df.iter_rows(named=True):
                f.write(f">{r['header']}\n{r['sequence']}\n")
    else:
        paired_a3m_path = None

    return unpaired_a3m_path, paired_a3m_path


def to_boltz(
    config: UniAF3Config,
    msa_dir: str | Path,
    max_num_templates_per_chain: int = 4,
    strict: bool = False,
) -> BoltzConfig:
    """Convert a UniAF3Config to a Boltz config.

    Args:
        config: UniAF3Config pydantic object.
        msa_dir: Directory to save MSA CSV files for Boltz.
        max_num_templates_per_chain: Maximum number of templates to use per chain.
            The default is 4, which matches Chai's limit. A higher number
            may lead to excessive GPU memory usage.
        strict: If True, raise errors when encountering unsupported features.
            If False, skip unsupported features with warnings.

    """
    msa_dir_path = Path(msa_dir).expanduser().resolve()
    msa_dir_path.mkdir(parents=True, exist_ok=True)

    sequences: list[BoltzSequenceEntry] = []
    templates: list[BoltzTemplate] = []
    seq_types: dict[str, str] = {}
    for seq in config.sequences:
        # Note the chain types for later pocket constraint processing
        for chain_id in ensure_list(seq.id):
            seq_types[chain_id] = (
                seq.polymer_type.value if isinstance(seq, Polymer) else "ligand"
            )

        if isinstance(seq, Glycan):
            sugars, bonds = _glycan_string_to_sugars_and_bonds(seq.chai_str)
            if not sugars:
                raise ValueError(
                    f"Failed to parse any sugars from glycan: {seq.chai_str}"
                )
            if bonds:
                # TODO: add constraint to keep glycan as PTM
                err_unsupported_feature(
                    strict, f"Bonded glycans are not directly supported in Boltz: {seq}"
                )
            if len(sugars) > 1:
                err_unsupported_feature(
                    strict,
                    f"Multi-CCD ligands are not supported in Boltz, maybe use SMILES instead: {seq}",
                )
            else:
                glycan = BoltzLigand(id=seq.id, ccd=sugars[0])
                sequences.append(BoltzSequenceEntry(ligand=glycan))

        elif isinstance(seq, ProteinSeq):
            mods = (
                [
                    BoltzModification(position=m.position, ccd=m.ccd)
                    for m in seq.modifications
                ]
                if seq.modifications
                else None
            )
            if seq.unpaired_msa is not None:
                msa_csv_path = merge_colabfold_msa_to_csv(
                    seq.unpaired_msa,
                    seq.paired_msa,
                    msa_id=seq.seq_hash,
                    out_dir=msa_dir_path,
                )
            else:
                msa_csv_path = "empty"
            protein = BoltzProtein(
                id=seq.id,
                sequence=seq.sequence,
                msa=str(msa_csv_path),
                modifications=mods,
                cyclic=seq.boltz_cyclic,
            )
            sequences.append(BoltzSequenceEntry(protein=protein))

            # Protein templates are added as separate entries in Boltz
            if not seq.templates:
                continue
            for i, tmpl in enumerate(seq.templates, start=1):
                if i > max_num_templates_per_chain:
                    warn_lossy_conversion(
                        f"UniAF3Config.sequences[*].templates beyond index {max_num_templates_per_chain} are dropped when mapping to BoltzConfig.templates."
                    )
                    break
                tmpl_path = Path(tmpl.path).expanduser().resolve()
                cif_path, pdb_path = None, None
                match "".join(tmpl_path.suffixes[-2:]).lower():
                    case ".cif" | ".cif.gz":
                        cif_path = str(tmpl_path)
                    case ".pdb" | ".pdb.gz":
                        pdb_path = str(tmpl_path)
                    case _:
                        raise ValueError(
                            f"Unsupported template file format: {tmpl_path}"
                        )
                templates.append(
                    BoltzTemplate(
                        cif=cif_path,
                        pdb=pdb_path,
                        chain_id=tmpl.query_chains,
                        # TODO: Boltz uses gemmi.structure.entities instead of subchains
                        # See boltz.data.parse.mmcif.parse_mmcif
                        # template_id=tmpl.template_chains,
                        force=tmpl.boltz_enable_force,
                        threshold=tmpl.boltz_template_threshold,
                    )
                )
        elif isinstance(seq, Polymer):
            mods = (
                [
                    BoltzModification(position=m.position, ccd=m.ccd)
                    for m in seq.modifications
                ]
                if seq.modifications
                else None
            )
            if seq.polymer_type == PolymerType.DNA:
                dna = BoltzDNA(
                    id=seq.id,
                    sequence=seq.sequence,
                    modifications=mods,
                    cyclic=seq.boltz_cyclic,
                )
                sequences.append(BoltzSequenceEntry(dna=dna))
            elif seq.polymer_type == PolymerType.RNA:
                rna = BoltzRNA(
                    id=seq.id,
                    sequence=seq.sequence,
                    modifications=mods,
                    cyclic=seq.boltz_cyclic,
                )
                sequences.append(BoltzSequenceEntry(rna=rna))
        elif isinstance(seq, Ligand):
            if seq.ccd and len(seq.ccd) == 1:
                lig = BoltzLigand(id=seq.id, ccd=seq.ccd[0])
            elif seq.smiles:
                lig = BoltzLigand(id=seq.id, smiles=seq.smiles)
            else:
                err_unsupported_feature(
                    strict,
                    f"Multi-CCD ligands are not supported in Boltz, maybe use SMILES instead: {seq}",
                )
                continue
            sequences.append(BoltzSequenceEntry(ligand=lig))
        else:
            err_unsupported_feature(strict, f"Unsupported sequence type {type(seq)}")

    # Constraints
    constraints: list[BoltzConstraintEntry] = []
    for b in config.covalent_bonds or []:
        if b.atom1.atom_name is None or b.atom2.atom_name is None:
            err_unsupported_feature(
                strict, f"Atom names must be specified for Boltz covalent bonds: {b}"
            )
            continue
        bond = BoltzBondConstraint(
            atom1=(b.atom1.chain_id, b.atom1.residue_idx, b.atom1.atom_name),
            atom2=(b.atom2.chain_id, b.atom2.residue_idx, b.atom2.atom_name),
        )
        constraints.append(BoltzConstraintEntry(bond=bond))

    for c in config.contact_restraints or []:
        token1_idx = (
            c.token1.atom_name
            if seq_types[c.token1.chain_id] == "ligand"
            else c.token1.residue_idx
        )
        if token1_idx is None:
            raise ValueError(
                f"Atom name must be specified for contact restraints on ligands: {c.token1}"
            )
        token2_idx = (
            c.token2.atom_name
            if seq_types[c.token2.chain_id] == "ligand"
            else c.token2.residue_idx
        )
        if token2_idx is None:
            raise ValueError(
                f"Atom name must be specified for contact restraints on ligands: {c.token2}"
            )
        contact = BoltzContactConstraint(
            token1=(c.token1.chain_id, token1_idx),
            token2=(c.token2.chain_id, token2_idx),
            max_distance=c.max_distance,
            force=c.boltz_enable_force,
        )
        constraints.append(BoltzConstraintEntry(contact=contact))
    for p in config.pocket_restraints or []:
        token_indices: list[tuple[str, str | int]] = []
        for t in p.contact_tokens:
            if seq_types[t.chain_id] == "ligand":
                if t.atom_name is None:
                    raise ValueError(
                        f"Atom name must be specified for pocket restraints on ligands: {t}"
                    )
                token_indices.append((t.chain_id, t.atom_name))
            else:
                if t.residue_idx is None:
                    raise ValueError(
                        f"Residue index must be specified for pocket restraints on polymers: {t}"
                    )
                token_indices.append((t.chain_id, t.residue_idx))

        pocket = BoltzPocketConstraint(
            binder=p.binder_chain,
            contacts=token_indices,
            max_distance=p.max_distance,
            force=p.boltz_enable_force,
        )
        constraints.append(BoltzConstraintEntry(pocket=pocket))

    # Properties
    properties: list[BoltzPropertyEntry] = []
    if config.aux.boltz_affinity_binder_chain is not None:
        properties.append(
            BoltzPropertyEntry(
                affinity=BoltzAffinityProperty(
                    binder=config.aux.boltz_affinity_binder_chain
                )
            )
        )

    return BoltzConfig(
        sequences=sequences,
        constraints=constraints if constraints else None,
        templates=templates if templates else None,
        properties=properties if properties else None,
    )


def from_boltz(config: BoltzConfig, msa_dir: str | Path) -> UniAF3Config:
    """Convert a Boltz config to a UniAF3Config."""
    sequences: list[Polymer | ProteinSeq | Ligand | Glycan] = []
    polymer_chains: set[str] = set()
    msa_dir_path = Path(msa_dir).expanduser().resolve()
    for entry in config.sequences:
        if entry.protein is not None:
            p = entry.protein
            mods = (
                [
                    SequenceModification(ccd=m.ccd, position=m.position)
                    for m in p.modifications
                ]
                if p.modifications
                else None
            )
            if p.msa and p.msa != "empty":
                msa_dir_path.mkdir(parents=True, exist_ok=True)
                input_msa_filetype = Path(p.msa).suffix
                if input_msa_filetype == ".csv":
                    unpaired_path, paired_path = split_boltz_csv_to_a3m(
                        p.msa, msa_dir_path / "a3ms"
                    )
                elif input_msa_filetype == ".a3m":
                    import shutil

                    (msa_dir_path / "a3ms").mkdir(parents=True, exist_ok=True)
                    prefix = hash_sequence(p.sequence)
                    unpaired_path = msa_dir_path / "a3ms" / f"{prefix}.single.a3m"
                    shutil.copyfile(p.msa, unpaired_path)
                    paired_path = None
                else:
                    raise ValueError(
                        f"Unsupported MSA file type in Boltz config: {p.msa}"
                    )
            else:
                unpaired_path = None
                paired_path = None
            seq = ProteinSeq(
                polymer_type=PolymerType.Protein,
                id=p.id,
                sequence=p.sequence,
                modifications=mods,
                boltz_cyclic=p.cyclic,
                unpaired_msa=str(unpaired_path) if unpaired_path else None,
                paired_msa=str(paired_path) if paired_path else None,
            )
            sequences.append(seq)
            polymer_chains.update(ensure_list(p.id))
        elif entry.dna is not None:
            d = entry.dna
            mods = (
                [
                    SequenceModification(ccd=m.ccd, position=m.position)
                    for m in d.modifications
                ]
                if d.modifications
                else None
            )
            seq = Polymer(
                polymer_type=PolymerType.DNA,
                id=d.id,
                sequence=d.sequence,
                modifications=mods,
                boltz_cyclic=d.cyclic,
            )
            sequences.append(seq)
            polymer_chains.update(ensure_list(d.id))
        elif entry.rna is not None:
            r = entry.rna
            mods = (
                [
                    SequenceModification(ccd=m.ccd, position=m.position)
                    for m in r.modifications
                ]
                if r.modifications
                else None
            )
            seq = Polymer(
                polymer_type=PolymerType.RNA,
                id=r.id,
                sequence=r.sequence,
                modifications=mods,
                boltz_cyclic=r.cyclic,
            )
            sequences.append(seq)
            polymer_chains.update(ensure_list(r.id))
        elif entry.ligand is not None:
            lg = entry.ligand
            ccd = [lg.ccd] if lg.ccd else None
            lig = Ligand(id=lg.id, ccd=ccd, smiles=lg.smiles)
            sequences.append(lig)

    # Map Boltz templates to ProteinSeq instances
    if config.templates:
        # Build a mapping from chain_id → ProteinSeq index
        chain_to_seq_idx: dict[str, int] = {}
        for idx, seq in enumerate(sequences):
            if isinstance(seq, ProteinSeq):
                for cid in ensure_list(seq.id):
                    chain_to_seq_idx[cid] = idx

        for tmpl in config.templates:
            # BoltzTemplate validator ensures exactly one of cif/pdb is set
            if tmpl.cif is not None:
                tmpl_path = tmpl.cif
            elif tmpl.pdb is not None:
                tmpl_path = tmpl.pdb
            else:
                raise ValueError(
                    f"BoltzTemplate must have either cif or pdb path specified: {tmpl}"
                )
            tmpl_chain_ids = ensure_list(tmpl.chain_id) if tmpl.chain_id else []
            tmpl_template_ids = (
                ensure_list(tmpl.template_id) if tmpl.template_id else None
            )

            structural_tmpl = StructuralTemplate(
                path=tmpl_path,
                query_chains=tmpl_chain_ids or None,
                template_chains=tmpl_template_ids,
                boltz_enable_force=tmpl.force,
                boltz_template_threshold=tmpl.threshold,
            )

            # Attach template to matching protein(s)
            matched = False
            for cid in tmpl_chain_ids:
                if cid in chain_to_seq_idx:
                    seq_idx = chain_to_seq_idx[cid]
                    prot = sequences[seq_idx]
                    if isinstance(prot, ProteinSeq):
                        if prot.templates is None:
                            prot.templates = []
                        prot.templates.append(structural_tmpl)
                        matched = True
            if not matched and tmpl_chain_ids:
                warn_lossy_conversion(
                    f"BoltzConfig.templates[*].chain_id references unknown UniAF3 protein chain(s) {tmpl_chain_ids}: {tmpl_path}"
                )
            elif not tmpl_chain_ids:
                warn_lossy_conversion(
                    f"BoltzConfig.templates[*].chain_id is missing; template cannot be attached to UniAF3 ProteinSeq.templates and is dropped: {tmpl_path}"
                )

    # Restraints
    covalent_bonds: list[CovalentBond] = []
    pocket_rsts: list[PocketRestraint] = []
    contact_rsts: list[ContactRestraint] = []
    if config.constraints:
        for c in config.constraints:
            if c.bond is not None:
                b = c.bond
                covalent_bonds.append(
                    CovalentBond(
                        atom1=Atom(
                            chain_id=b.atom1[0],
                            residue_idx=b.atom1[1],
                            atom_name=b.atom1[2],
                            residue_name=None,
                        ),
                        atom2=Atom(
                            chain_id=b.atom2[0],
                            residue_idx=b.atom2[1],
                            atom_name=b.atom2[2],
                            residue_name=None,
                        ),
                    )
                )
            elif c.contact is not None:
                ct = c.contact
                chain1, resi_or_atomn1 = ct.token1
                chain2, resi_or_atomn2 = ct.token2
                resi1, atomn1 = (
                    (int(resi_or_atomn1), None)
                    if chain1 in polymer_chains
                    else (0, str(resi_or_atomn1))
                )
                resi2, atomn2 = (
                    (int(resi_or_atomn2), None)
                    if chain2 in polymer_chains
                    else (0, str(resi_or_atomn2))
                )
                contact_rsts.append(
                    ContactRestraint(
                        token1=Atom(
                            chain_id=chain1,
                            residue_idx=resi1,
                            atom_name=atomn1,
                            residue_name=None,
                        ),
                        token2=Atom(
                            chain_id=chain2,
                            residue_idx=resi2,
                            atom_name=atomn2,
                            residue_name=None,
                        ),
                        max_distance=ct.max_distance,
                        boltz_enable_force=ct.force,
                    )
                )
            elif c.pocket is not None:
                pk = c.pocket
                contact_atoms: list[Atom] = []
                for t in pk.contacts:
                    chain_id, resi_or_atomn = t
                    resi, atomn = (
                        (int(resi_or_atomn), None)
                        if chain_id in polymer_chains
                        else (0, str(resi_or_atomn))
                    )
                    contact_atoms.append(
                        Atom(
                            chain_id=chain_id,
                            residue_idx=resi,
                            atom_name=atomn,
                            residue_name=None,
                        )
                    )
                pocket_rsts.append(
                    PocketRestraint(
                        binder_chain=pk.binder,
                        contact_tokens=contact_atoms,
                        max_distance=pk.max_distance,
                        boltz_enable_force=pk.force,
                    )
                )

    aux = AuxiliaryParams()
    if config.properties is not None:
        for prop in config.properties:
            if prop.affinity is not None:
                aux.boltz_affinity_binder_chain = prop.affinity.binder

    warn_lossy_conversion(
        "BoltzConfig has no seed field; UniAF3Config.aux.seeds defaults to [42]."
    )
    aux.seeds = [42]
    return UniAF3Config(
        sequences=sequences,
        covalent_bonds=covalent_bonds or None,
        pocket_restraints=pocket_rsts or None,
        contact_restraints=contact_rsts or None,
        aux=aux,
    )
