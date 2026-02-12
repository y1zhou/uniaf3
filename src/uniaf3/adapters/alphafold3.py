"""Adapter for converting between UniAF3Config and AlphaFold3 config."""

from __future__ import annotations

from pathlib import Path

from uniaf3.adapters._helpers import (
    err_unsupported_feature,
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
    AF3Template,
)
from uniaf3.schema.base import (
    Atom,
    CovalentBond,
    Glycan,
    Ligand,
    Polymer,
    PolymerType,
    ProteinSeq,
    SequenceModification,
    StructuralTemplate,
    UniAF3Config,
)


def to_alphafold3(
    config: UniAF3Config,
    name: str = "uniaf3_job",
    strict: bool = True,
) -> AF3Config:
    """Convert a UniAF3Config to an AlphaFold3 config.

    Args:
        config: UniAF3Config pydantic object.
        name: Job name for the AF3 config.
        strict: If True, raise errors when encountering unsupported features.
            If False, skip unsupported features with warnings.

    """
    sequences: list[AF3SequenceEntry] = []
    for seq in config.sequences:
        if isinstance(seq, Glycan):
            # NOTE: AF3 does not have a native glycan type. Glycans must be
            # represented as multi-CCD ligands. This requires knowing the
            # component CCD codes, which the chai_str notation may not directly
            # map to.
            err_unsupported_feature(
                strict,
                f"Glycans are not directly supported in AF3: {seq}",
            )
            continue

        if isinstance(seq, ProteinSeq):
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
                unpairedMsaPath=seq.unpaired_msa,
                pairedMsaPath=seq.paired_msa,
            )
            # Templates
            if seq.templates:
                af3_templates = []
                for tmpl in seq.templates:
                    af3_templates.append(
                        AF3Template(
                            mmcifPath=tmpl.path,
                            queryIndices=tmpl.query_idx or [],
                            templateIndices=tmpl.template_idx or [],
                        )
                    )
                protein.templates = af3_templates
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
                sequences.append(AF3SequenceEntry(protein=protein))
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

        elif isinstance(seq, Ligand):
            if seq.ccd:
                lig = AF3Ligand(
                    id=seq.id,
                    ccdCodes=seq.ccd,
                    description=seq.description,
                )
            elif seq.smiles:
                lig = AF3Ligand(
                    id=seq.id,
                    smiles=seq.smiles,
                    description=seq.description,
                )
            else:
                continue
            sequences.append(AF3SequenceEntry(ligand=lig))

    # Bonded atom pairs (only covalent bonds)
    bonded_atom_pairs: list[tuple[AF3BondedAtom, AF3BondedAtom]] = []
    if config.covalent_bonds:
        for r in config.covalent_bonds:
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
            bonded_atom_pairs.append((a1, a2))

    if config.contact_restraints or config.pocket_restraints:
        err_unsupported_feature(
            strict,
            "AF3 does not support contact or pocket restraints.",
        )

    return AF3Config(
        name=name,
        modelSeeds=config.seeds,
        sequences=sequences,
        bondedAtomPairs=bonded_atom_pairs or None,
    )


def from_alphafold3(config: AF3Config) -> UniAF3Config:
    """Convert an AlphaFold3 config to a UniAF3Config.

    Args:
        config: AF3Config pydantic object.

    Returns:
        A UniAF3Config.

    """
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
            # Determine MSA dir from MSA paths
            msa_dir = None
            if p.unpairedMsaPath:
                # NOTE: AF3 uses direct file paths; UniAF3 uses directory-based
                # lookup. We set msa_dir to the parent of the unpaired MSA path.
                msa_dir = str(Path(p.unpairedMsaPath).parent)

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

            seq = ProteinSeq(
                seq_type=PolymerType.Protein,
                id=p.id,
                sequence=p.sequence,
                modifications=mods,
                description=p.description,
                msa_dir=msa_dir,
                templates=templates,
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

        elif entry.ligand is not None:
            lg = entry.ligand
            lig = Ligand(
                id=lg.id,
                ccd=lg.ccdCodes,
                smiles=lg.smiles,
                description=lg.description,
            )
            sequences.append(lig)

    # Bonded atom pairs → covalent restraints
    covalent_bonds: list[CovalentBond] = []
    if config.bondedAtomPairs:
        for a1, a2 in config.bondedAtomPairs:
            covalent_bonds.append(
                CovalentBond(
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
                )
            )

    return UniAF3Config(
        sequences=sequences,
        covalent_bonds=covalent_bonds or None,
        seeds=config.modelSeeds,
    )
