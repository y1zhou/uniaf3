"""Adapter for converting between UniAF3Config and AlphaFold3 config."""

from __future__ import annotations

from pathlib import Path

from uniaf3.adapters._helpers import (
    err_unsupported_feature,
    warn_lossy_conversion,
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
    hash_sequence,
)


def to_alphafold3(
    config: UniAF3Config,
    msa_dir: str | Path,
    name: str = "uniaf3_job",
    strict: bool = False,
) -> AF3Config:
    """Convert a UniAF3Config to an AlphaFold3 config.

    Args:
        config: UniAF3Config pydantic object.
        msa_dir: Directory to save MSA files.
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
                    if tmpl.query_chains or tmpl.template_chains:
                        warn_lossy_conversion(
                            "UniAF3Config.sequences[*].templates.{query_chains,template_chains} are not represented by AF3Config.sequences[*].protein.templates."
                        )
                    if (
                        tmpl.boltz_enable_force
                        or tmpl.boltz_template_threshold is not None
                    ):
                        warn_lossy_conversion(
                            "UniAF3Config.sequences[*].templates.{boltz_enable_force,boltz_template_threshold} are not represented by AF3Config.sequences[*].protein.templates."
                        )
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


def from_alphafold3(config: AF3Config, msa_dir: str | Path) -> UniAF3Config:
    """Convert an AlphaFold3 config to a UniAF3Config.

    Args:
        config: AF3Config pydantic object.
        msa_dir: Directory to save MSA files.

    Returns:
        A UniAF3Config.

    """
    sequences: list[Polymer | ProteinSeq | Ligand | Glycan] = []
    msa_dir_path = Path(msa_dir) / "a3ms"
    if config.name:
        warn_lossy_conversion(
            f"AF3Config.name ('{config.name}') is not represented in UniAF3Config."
        )
    if config.userCCD is not None or config.userCCDPath is not None:
        warn_lossy_conversion(
            "AF3Config.{userCCD,userCCDPath} are not represented in UniAF3Config."
        )

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
            for i, f in enumerate((p.unpairedMsaPath, p.pairedMsaPath)):
                if f is not None:
                    msa_dir_path.mkdir(parents=True, exist_ok=True)
                    seq_hash = hash_sequence(p.sequence)
                    msa_type = "single" if i == 0 else "pair"

                    msa_path = msa_dir_path / f"{seq_hash}.{msa_type}.a3m"
                    if not msa_path.exists():
                        import shutil

                        shutil.copyfile(f, msa_path)
            if p.unpairedMsa is not None or p.pairedMsa is not None:
                warn_lossy_conversion(
                    "AF3Config.sequences[*].protein.{unpairedMsa,pairedMsa} are not imported; UniAF3 maps only file-based MSA paths."
                )

            templates = None
            if p.templates:
                if any(
                    t.mmcif is not None and t.mmcifPath is None for t in p.templates
                ):
                    warn_lossy_conversion(
                        "AF3Config.sequences[*].protein.templates[*].mmcif is not preserved; only mmcifPath maps to UniAF3 templates.path."
                    )
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
                msa_dir=str(msa_dir_path.parent),
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
