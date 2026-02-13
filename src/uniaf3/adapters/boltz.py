"""Adapter for converting between UniAF3Config and Boltz config."""

from __future__ import annotations

from pathlib import Path

import polars as pl

from uniaf3.adapters._helpers import err_unsupported_feature
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
from uniaf3.vendor.chai1_fasta import read_fasta


def merge_chai_msa_to_csv(
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
        out_dir: Directory to save the output CSV file.

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
            .drop("header")
            .with_row_index(name="key")
            # filter out padding rows (rows that are all gaps)
            .filter(
                pl.col("sequence").str.count_matches("-", literal=True)
                < pl.col("sequence").str.len_bytes()
            )
        )
    else:
        paired_df = pl.DataFrame([], schema=["key", "sequence"])

    # combine paired-unpaired sequences
    unpaired_fasta = read_fasta(unpaired_path)
    if paired_df.height > 0:
        # ignore query seq
        unpaired_fasta = unpaired_fasta[1:]

    combined_df = pl.concat(
        [
            paired_df,
            pl.DataFrame(unpaired_fasta)
            .drop("header")
            .with_columns(pl.lit(-1).alias("key")),
        ],
        how="vertical_relaxed",
    )
    combined_df.write_csv(out_file)
    return out_file


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
        if isinstance(seq.id, list):
            for chain_id in seq.id:
                seq_types[chain_id] = (
                    seq.seq_type.value if isinstance(seq, Polymer) else "ligand"
                )
        else:
            seq_types[seq.id] = (
                seq.seq_type.value if isinstance(seq, Polymer) else "ligand"
            )

        if isinstance(seq, Glycan):
            # TODO: use SMILES as a fallback representation.
            # need atom_idx for glycan and add constraint to keep glycan as PTM
            # lig = BoltzLigand(id=seq.id, smiles=seq.chai_str)
            # sequences.append(BoltzSequenceEntry(ligand=lig))
            err_unsupported_feature(
                strict, f"Glycans are not directly supported in Boltz: {seq}"
            )
            continue

        if isinstance(seq, ProteinSeq):
            mods = (
                [
                    BoltzModification(position=m.position, ccd=m.ccd)
                    for m in seq.modifications
                ]
                if seq.modifications
                else None
            )
            if seq.msa_dir is not None:
                msa_csv_path = merge_chai_msa_to_csv(
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
            if seq.seq_type == PolymerType.DNA:
                dna = BoltzDNA(
                    id=seq.id,
                    sequence=seq.sequence,
                    modifications=mods,
                    cyclic=seq.boltz_cyclic,
                )
                sequences.append(BoltzSequenceEntry(dna=dna))
            elif seq.seq_type == PolymerType.RNA:
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
            sequences.append(BoltzSequenceEntry(ligand=lig))
        else:
            err_unsupported_feature(strict, f"Unsupported sequence type {type(seq)}")

    # Constraints
    constraints: list[BoltzConstraintEntry] = []
    for b in config.covalent_bonds or []:
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
        token2_idx = (
            c.token2.atom_name
            if seq_types[c.token2.chain_id] == "ligand"
            else c.token2.residue_idx
        )
        contact = BoltzContactConstraint(
            token1=(c.token1.chain_id, token1_idx),
            token2=(c.token2.chain_id, token2_idx),
            max_distance=c.max_distance,
            force=c.boltz_enable_force,
        )
        constraints.append(BoltzConstraintEntry(contact=contact))
    for p in config.pocket_restraints or []:
        token_indices = [
            (t.chain_id, t.atom_name)
            if seq_types[t.chain_id] == "ligand"
            else (t.chain_id, t.residue_idx)
            for t in p.contact_tokens
        ]
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


def from_boltz(config: BoltzConfig) -> UniAF3Config:
    """Convert a Boltz config to a UniAF3Config."""
    sequences: list[Polymer | ProteinSeq | Ligand | Glycan] = []
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
            # NOTE: Boltz provides a single MSA path; UniAF3 uses msa_dir for
            # directory-based lookup. The direct path is not stored in msa_dir.
            seq = ProteinSeq(
                seq_type=PolymerType.Protein,
                id=p.id,
                sequence=p.sequence,
                modifications=mods,
                boltz_cyclic=p.cyclic,
            )
            sequences.append(seq)
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
                seq_type=PolymerType.DNA,
                id=d.id,
                sequence=d.sequence,
                modifications=mods,
                boltz_cyclic=d.cyclic,
            )
            sequences.append(seq)
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
                seq_type=PolymerType.RNA,
                id=r.id,
                sequence=r.sequence,
                modifications=mods,
                boltz_cyclic=r.cyclic,
            )
            sequences.append(seq)
        elif entry.ligand is not None:
            lg = entry.ligand
            ccd = [lg.ccd] if lg.ccd else None
            lig = Ligand(id=lg.id, ccd=ccd, smiles=lg.smiles)
            sequences.append(lig)

    # Restraints
    covalent_bonds: list[CovalentBond] = []
    pocket_rsts: list[PocketRestraint] = []
    contact_rsts: list[ContactRestraint] = []
    # TODO: helper function to determine if atom is in polymer or ligand
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
                contact_rsts.append(
                    ContactRestraint(
                        token1=Atom(
                            chain_id=ct.token1[0],
                            residue_idx=int(ct.token1[1]),
                            atom_name="",
                            residue_name=None,
                        ),
                        token2=Atom(
                            chain_id=ct.token2[0],
                            residue_idx=int(ct.token2[1]),
                            atom_name="",
                            residue_name=None,
                        ),
                        max_distance=ct.max_distance,
                        boltz_enable_force=ct.force,
                    )
                )
            elif c.pocket is not None:
                pk = c.pocket
                pocket_rsts.append(
                    PocketRestraint(
                        binder_chain=pk.binder,
                        contact_tokens=[
                            Atom(
                                chain_id=t[0],
                                residue_idx=int(t[1]),
                                atom_name=str(t[1]),
                                residue_name=None,
                            )
                            for t in pk.contacts
                        ],
                        max_distance=pk.max_distance,
                        boltz_enable_force=pk.force,
                    )
                )

    aux = AuxiliaryParams()
    if config.properties is not None:
        for prop in config.properties:
            if prop.affinity is not None:
                aux.boltz_affinity_binder_chain = prop.affinity.binder

    return UniAF3Config(
        sequences=sequences,
        covalent_bonds=covalent_bonds or None,
        pocket_restraints=pocket_rsts or None,
        contact_restraints=contact_rsts or None,
        seeds=[42],  # NOTE: Boltz config does not include seeds
        aux=aux,
    )
