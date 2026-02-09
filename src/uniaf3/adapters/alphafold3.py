"""Adapter for converting between UniAF3Config and AlphaFold3 config."""

from __future__ import annotations

from uniaf3.schema import (
    Atom,
    Glycan,
    Ligand,
    Polymer,
    PolymerType,
    ProteinSeq,
    Restraint,
    RestraintType,
    SequenceModification,
    StructuralTemplate,
    UniAF3Config,
)
from uniaf3.schema.alphafold3 import (
    AF3DNA,
    AF3RNA,
    AF3BondedAtom,
    AF3Config,
    AF3Ligand,
    AF3NucleotideModification,
    AF3Protein,
    AF3ProteinModification,
    AF3SequenceEntry,
    AF3ServerConfig,
    AF3ServerDNA,
    AF3ServerIon,
    AF3ServerLigand,
    AF3ServerProtein,
    AF3ServerRNA,
    AF3ServerSequenceEntry,
    AF3Template,
)


def to_alphafold3(config: UniAF3Config, name: str = "uniaf3_job") -> AF3Config:
    """Convert a UniAF3Config to an AlphaFold3 config."""
    sequences: list[AF3SequenceEntry] = []
    for seq in config.sequences:
        if isinstance(seq, ProteinSeq):
            mods = (
                [
                    AF3ProteinModification(ptmType=m.ccd, ptmPosition=m.position)
                    for m in seq.modifications
                ]
                if seq.modifications
                else None
            )
            templates: list[AF3Template] | None = None
            if seq.templates:
                templates = [
                    AF3Template(
                        mmcifPath=t.path,
                        queryIndices=t.query_idx if t.query_idx else [],
                        templateIndices=t.template_idx if t.template_idx else [],
                    )
                    for t in seq.templates
                ]
            protein = AF3Protein(
                id=seq.id,
                sequence=seq.sequence,
                modifications=mods,
                description=seq.description,
                unpairedMsaPath=seq.unpaired_msa,
                pairedMsaPath=seq.paired_msa,
                templates=templates,
            )
            # NOTE: cyclic polymers are not supported in AlphaFold3
            sequences.append(AF3SequenceEntry(protein=protein))
        elif isinstance(seq, Polymer):
            if seq.seq_type == PolymerType.Protein:
                mods = (
                    [
                        AF3ProteinModification(ptmType=m.ccd, ptmPosition=m.position)
                        for m in seq.modifications
                    ]
                    if seq.modifications
                    else None
                )
                protein = AF3Protein(
                    id=seq.id,
                    sequence=seq.sequence,
                    modifications=mods,
                    description=seq.description,
                )
                # NOTE: cyclic polymers are not supported in AlphaFold3
                sequences.append(AF3SequenceEntry(protein=protein))
            elif seq.seq_type == PolymerType.RNA:
                mods = (
                    [
                        AF3NucleotideModification(
                            modificationType=m.ccd, basePosition=m.position
                        )
                        for m in seq.modifications
                    ]
                    if seq.modifications
                    else None
                )
                rna = AF3RNA(
                    id=seq.id,
                    sequence=seq.sequence,
                    modifications=mods,
                    description=seq.description,
                )
                sequences.append(AF3SequenceEntry(rna=rna))
            elif seq.seq_type == PolymerType.DNA:
                mods = (
                    [
                        AF3NucleotideModification(
                            modificationType=m.ccd, basePosition=m.position
                        )
                        for m in seq.modifications
                    ]
                    if seq.modifications
                    else None
                )
                dna = AF3DNA(
                    id=seq.id,
                    sequence=seq.sequence,
                    modifications=mods,
                    description=seq.description,
                )
                sequences.append(AF3SequenceEntry(dna=dna))
        elif isinstance(seq, Ligand):
            lig = AF3Ligand(
                id=seq.id,
                ccdCodes=seq.ccd,
                smiles=seq.smiles,
                description=seq.description,
            )
            sequences.append(AF3SequenceEntry(ligand=lig))
        elif isinstance(seq, Glycan):
            # NOTE: Glycans must be specified as multi-CCD ligands with bonds
            # in AlphaFold3. This is a lossy conversion.
            lig = AF3Ligand(
                id=seq.id,
                smiles=seq.chai_str,
                description=seq.description,
            )
            sequences.append(AF3SequenceEntry(ligand=lig))

    # Bonded atom pairs (only covalent restraints)
    bonded: list[tuple[AF3BondedAtom, AF3BondedAtom]] | None = None
    if config.restraints:
        bond_pairs: list[tuple[AF3BondedAtom, AF3BondedAtom]] = []
        for r in config.restraints:
            if r.restraint_type == RestraintType.Covalent:
                a1: AF3BondedAtom = (
                    r.atom1.chain_id,
                    r.atom1.residue_idx,
                    r.atom1.atom_name,
                )
                a2: AF3BondedAtom = (
                    r.atom2.chain_id,
                    r.atom2.residue_idx,
                    r.atom2.atom_name,
                )
                bond_pairs.append((a1, a2))
            # NOTE: AF3 only supports bonded restraints; pocket and contact
            # restraints are ignored.
        bonded = bond_pairs if bond_pairs else None

    return AF3Config(
        name=name,
        modelSeeds=config.seeds,
        sequences=sequences,
        bondedAtomPairs=bonded,
    )


def from_alphafold3(config: AF3Config) -> UniAF3Config:
    """Convert an AlphaFold3 config to a UniAF3Config."""
    sequences: list[Polymer | ProteinSeq | Ligand | Glycan] = []
    for entry in config.sequences:
        if entry.protein is not None:
            p = entry.protein
            mods = (
                [
                    SequenceModification(ccd=m.ptmType, position=m.ptmPosition)
                    for m in p.modifications
                ]
                if p.modifications
                else None
            )
            templates = None
            if p.templates:
                templates = [
                    StructuralTemplate(
                        path=t.mmcifPath or "",
                        query_idx=t.queryIndices,
                        template_idx=t.templateIndices,
                    )
                    for t in p.templates
                ]
            # NOTE: AF3 provides MSA inline or via path; we store the path.
            # The msa_dir concept does not directly map.
            seq = ProteinSeq(
                seq_type=PolymerType.Protein,
                id=p.id,
                sequence=p.sequence,
                modifications=mods,
                description=p.description,
                templates=templates,
            )
            sequences.append(seq)
        elif entry.rna is not None:
            r = entry.rna
            mods = (
                [
                    SequenceModification(
                        ccd=m.modificationType, position=m.basePosition
                    )
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
                description=r.description,
            )
            sequences.append(seq)
        elif entry.dna is not None:
            d = entry.dna
            mods = (
                [
                    SequenceModification(
                        ccd=m.modificationType, position=m.basePosition
                    )
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
                description=d.description,
            )
            sequences.append(seq)
        elif entry.ligand is not None:
            lg = entry.ligand
            lig = Ligand(
                id=lg.id,
                ccd=lg.ccdCodes,
                smiles=lg.smiles,
                description=lg.description,
            )
            sequences.append(lig)

    # Restraints from bonded atom pairs
    restraints: list[Restraint] | None = None
    if config.bondedAtomPairs:
        restraints = []
        for a1, a2 in config.bondedAtomPairs:
            restraints.append(
                Restraint(
                    restraint_type=RestraintType.Covalent,
                    atom1=Atom(
                        chain_id=a1[0],
                        residue_idx=a1[1],
                        atom_name=a1[2],
                        residue_name=None,
                    ),
                    atom2=Atom(
                        chain_id=a2[0],
                        residue_idx=a2[1],
                        atom_name=a2[2],
                        residue_name=None,
                    ),
                    max_distance=0.0,  # ignored for covalent bonds
                )
            )

    return UniAF3Config(
        sequences=sequences,
        restraints=restraints,
        seeds=config.modelSeeds,
    )


def to_alphafold3_server(
    config: UniAF3Config, name: str = "uniaf3_job"
) -> AF3ServerConfig:
    """Convert a UniAF3Config to an AlphaFold3 Server config.

    The server config is simpler – no seeds, no MSA, no templates.
    Ions (detected from known CCD codes) get their own entity type.
    """
    from uniaf3.adapters._helpers import _KNOWN_ION_CCD_CODES

    sequences: list[AF3ServerSequenceEntry] = []
    for seq in config.sequences:
        if isinstance(seq, ProteinSeq) or (
            isinstance(seq, Polymer) and seq.seq_type == PolymerType.Protein
        ):
            mods = (
                [
                    AF3ProteinModification(ptmType=m.ccd, ptmPosition=m.position)
                    for m in seq.modifications
                ]
                if seq.modifications
                else None
            )
            sequences.append(
                AF3ServerSequenceEntry(
                    protein=AF3ServerProtein(
                        id=seq.id,
                        sequence=seq.sequence,
                        modifications=mods,
                        description=seq.description,
                    )
                )
            )
        elif isinstance(seq, Polymer):
            if seq.seq_type == PolymerType.DNA:
                mods = (
                    [
                        AF3NucleotideModification(
                            modificationType=m.ccd, basePosition=m.position
                        )
                        for m in seq.modifications
                    ]
                    if seq.modifications
                    else None
                )
                sequences.append(
                    AF3ServerSequenceEntry(
                        dna=AF3ServerDNA(
                            id=seq.id,
                            sequence=seq.sequence,
                            modifications=mods,
                            description=seq.description,
                        )
                    )
                )
            elif seq.seq_type == PolymerType.RNA:
                mods = (
                    [
                        AF3NucleotideModification(
                            modificationType=m.ccd, basePosition=m.position
                        )
                        for m in seq.modifications
                    ]
                    if seq.modifications
                    else None
                )
                sequences.append(
                    AF3ServerSequenceEntry(
                        rna=AF3ServerRNA(
                            id=seq.id,
                            sequence=seq.sequence,
                            modifications=mods,
                            description=seq.description,
                        )
                    )
                )
        elif isinstance(seq, Ligand):
            if seq.ccd and len(seq.ccd) == 1 and seq.ccd[0] in _KNOWN_ION_CCD_CODES:
                sequences.append(
                    AF3ServerSequenceEntry(ion=AF3ServerIon(id=seq.id, ion=seq.ccd[0]))
                )
            else:
                sequences.append(
                    AF3ServerSequenceEntry(
                        ligand=AF3ServerLigand(
                            id=seq.id,
                            ccdCodes=seq.ccd,
                            smiles=seq.smiles,
                            description=seq.description,
                        )
                    )
                )
        elif isinstance(seq, Glycan):
            sequences.append(
                AF3ServerSequenceEntry(
                    ligand=AF3ServerLigand(
                        id=seq.id,
                        smiles=seq.chai_str,
                        description=seq.description,
                    )
                )
            )

    bonded: list[tuple[AF3BondedAtom, AF3BondedAtom]] | None = None
    if config.restraints:
        bond_pairs: list[tuple[AF3BondedAtom, AF3BondedAtom]] = []
        for r in config.restraints:
            if r.restraint_type == RestraintType.Covalent:
                a1: AF3BondedAtom = (
                    r.atom1.chain_id,
                    r.atom1.residue_idx,
                    r.atom1.atom_name,
                )
                a2: AF3BondedAtom = (
                    r.atom2.chain_id,
                    r.atom2.residue_idx,
                    r.atom2.atom_name,
                )
                bond_pairs.append((a1, a2))
        bonded = bond_pairs if bond_pairs else None

    return AF3ServerConfig(name=name, sequences=sequences, bondedAtomPairs=bonded)


def from_alphafold3_server(config: AF3ServerConfig) -> UniAF3Config:
    """Convert an AlphaFold3 Server config to a UniAF3Config."""
    sequences: list[Polymer | ProteinSeq | Ligand | Glycan] = []
    for entry in config.sequences:
        if entry.protein is not None:
            p = entry.protein
            mods = (
                [
                    SequenceModification(ccd=m.ptmType, position=m.ptmPosition)
                    for m in p.modifications
                ]
                if p.modifications
                else None
            )
            sequences.append(
                ProteinSeq(
                    seq_type=PolymerType.Protein,
                    id=p.id,
                    sequence=p.sequence,
                    modifications=mods,
                    description=p.description,
                )
            )
        elif entry.dna is not None:
            d = entry.dna
            mods = (
                [
                    SequenceModification(
                        ccd=m.modificationType, position=m.basePosition
                    )
                    for m in d.modifications
                ]
                if d.modifications
                else None
            )
            sequences.append(
                Polymer(
                    seq_type=PolymerType.DNA,
                    id=d.id,
                    sequence=d.sequence,
                    modifications=mods,
                    description=d.description,
                )
            )
        elif entry.rna is not None:
            r = entry.rna
            mods = (
                [
                    SequenceModification(
                        ccd=m.modificationType, position=m.basePosition
                    )
                    for m in r.modifications
                ]
                if r.modifications
                else None
            )
            sequences.append(
                Polymer(
                    seq_type=PolymerType.RNA,
                    id=r.id,
                    sequence=r.sequence,
                    modifications=mods,
                    description=r.description,
                )
            )
        elif entry.ligand is not None:
            lg = entry.ligand
            sequences.append(
                Ligand(
                    id=lg.id,
                    ccd=lg.ccdCodes,
                    smiles=lg.smiles,
                    description=lg.description,
                )
            )
        elif entry.ion is not None:
            io = entry.ion
            sequences.append(Ligand(id=io.id, ccd=[io.ion]))

    restraints: list[Restraint] | None = None
    if config.bondedAtomPairs:
        restraints = []
        for a1, a2 in config.bondedAtomPairs:
            restraints.append(
                Restraint(
                    restraint_type=RestraintType.Covalent,
                    atom1=Atom(
                        chain_id=a1[0],
                        residue_idx=a1[1],
                        atom_name=a1[2],
                        residue_name=None,
                    ),
                    atom2=Atom(
                        chain_id=a2[0],
                        residue_idx=a2[1],
                        atom_name=a2[2],
                        residue_name=None,
                    ),
                    max_distance=0.0,
                )
            )

    # NOTE: Server config has no seeds; default to [42]
    return UniAF3Config(
        sequences=sequences,
        restraints=restraints,
        seeds=[42],
    )
