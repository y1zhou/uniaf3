"""Adapter for converting between UniAF3Config and Boltz config."""

from __future__ import annotations

from uniaf3.adapters._helpers import _ensure_list
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


def to_boltz(config: UniAF3Config) -> BoltzConfig:
    """Convert a UniAF3Config to a Boltz config."""
    sequences: list[BoltzSequenceEntry] = []
    for seq in config.sequences:
        if isinstance(seq, ProteinSeq):
            mods = (
                [
                    BoltzModification(position=m.position, ccd=m.ccd)
                    for m in seq.modifications
                ]
                if seq.modifications
                else None
            )
            msa_path: str | None = seq.unpaired_msa
            protein = BoltzProtein(
                id=seq.id,
                sequence=seq.sequence,
                msa=msa_path,
                modifications=mods,
                cyclic=seq.cyclic,
            )
            sequences.append(BoltzSequenceEntry(protein=protein))
        elif isinstance(seq, Polymer):
            mods = (
                [
                    BoltzModification(position=m.position, ccd=m.ccd)
                    for m in seq.modifications
                ]
                if seq.modifications
                else None
            )
            if seq.seq_type == PolymerType.Protein:
                protein = BoltzProtein(
                    id=seq.id,
                    sequence=seq.sequence,
                    modifications=mods,
                    cyclic=seq.cyclic,
                )
                sequences.append(BoltzSequenceEntry(protein=protein))
            elif seq.seq_type == PolymerType.DNA:
                dna = BoltzDNA(
                    id=seq.id,
                    sequence=seq.sequence,
                    modifications=mods,
                    cyclic=seq.cyclic,
                )
                sequences.append(BoltzSequenceEntry(dna=dna))
            elif seq.seq_type == PolymerType.RNA:
                rna = BoltzRNA(
                    id=seq.id,
                    sequence=seq.sequence,
                    modifications=mods,
                    cyclic=seq.cyclic,
                )
                sequences.append(BoltzSequenceEntry(rna=rna))
        elif isinstance(seq, Ligand):
            if seq.ccd and len(seq.ccd) == 1:
                lig = BoltzLigand(id=seq.id, ccd=seq.ccd[0])
            elif seq.smiles:
                lig = BoltzLigand(id=seq.id, smiles=seq.smiles)
            else:
                # NOTE: Boltz ligands only accept a single CCD code.
                # Multi-CCD ligands (e.g. glycans) are not natively supported.
                lig = BoltzLigand(id=seq.id, ccd=seq.ccd[0] if seq.ccd else None)
            sequences.append(BoltzSequenceEntry(ligand=lig))
        elif isinstance(seq, Glycan):
            # NOTE: Glycans are not directly supported in Boltz; using SMILES
            # as a fallback representation.
            lig = BoltzLigand(id=seq.id, smiles=seq.chai_str)
            sequences.append(BoltzSequenceEntry(ligand=lig))

    # Constraints
    constraints: list[BoltzConstraintEntry] | None = None
    if config.restraints:
        constraint_list: list[BoltzConstraintEntry] = []
        for r in config.restraints:
            if r.restraint_type == RestraintType.Covalent:
                bond = BoltzBondConstraint(
                    atom1=(r.atom1.chain_id, r.atom1.residue_idx, r.atom1.atom_name),
                    atom2=(r.atom2.chain_id, r.atom2.residue_idx, r.atom2.atom_name),
                )
                constraint_list.append(BoltzConstraintEntry(bond=bond))
            elif r.restraint_type == RestraintType.Contact:
                contact = BoltzContactConstraint(
                    token1=(r.atom1.chain_id, r.atom1.residue_idx),
                    token2=(r.atom2.chain_id, r.atom2.residue_idx),
                    max_distance=r.max_distance,
                    force=r.enable_boltz_force,
                )
                constraint_list.append(BoltzConstraintEntry(contact=contact))
            elif r.restraint_type == RestraintType.Pocket:
                if r.boltz_binder_chain is None:
                    # NOTE: Pocket restraints require boltz_binder_chain to be
                    # set. Skipping this restraint.
                    continue
                pocket = BoltzPocketConstraint(
                    binder=r.boltz_binder_chain,
                    contacts=[
                        (r.atom1.chain_id, r.atom1.residue_idx),
                        (r.atom2.chain_id, r.atom2.residue_idx),
                    ],
                    max_distance=r.max_distance,
                    force=r.enable_boltz_force,
                )
                constraint_list.append(BoltzConstraintEntry(pocket=pocket))
        constraints = constraint_list if constraint_list else None

    # Templates from protein sequences
    templates: list[BoltzTemplate] | None = None
    template_list: list[BoltzTemplate] = []
    for seq in config.sequences:
        if isinstance(seq, ProteinSeq) and seq.templates:
            for t in seq.templates:
                tmpl = BoltzTemplate(
                    cif=t.path,
                    chain_id=seq.id if isinstance(seq.id, str) else seq.id[0],
                    force=t.enable_boltz_force,
                    threshold=t.boltz_template_threshold,
                )
                template_list.append(tmpl)
    templates = template_list if template_list else None

    # Properties
    properties: list[BoltzPropertyEntry] | None = None
    if config.boltz_affinity_binder_chain:
        properties = [
            BoltzPropertyEntry(
                affinity=BoltzAffinityProperty(
                    binder=config.boltz_affinity_binder_chain
                )
            )
        ]

    return BoltzConfig(
        sequences=sequences,
        constraints=constraints,
        templates=templates,
        properties=properties,
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
                cyclic=p.cyclic,
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
                cyclic=d.cyclic,
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
                cyclic=r.cyclic,
            )
            sequences.append(seq)
        elif entry.ligand is not None:
            lg = entry.ligand
            ccd = [lg.ccd] if lg.ccd else None
            lig = Ligand(id=lg.id, ccd=ccd, smiles=lg.smiles)
            sequences.append(lig)

    # Restraints
    restraints: list[Restraint] | None = None
    if config.constraints:
        restraint_list: list[Restraint] = []
        for c in config.constraints:
            if c.bond is not None:
                b = c.bond
                restraint_list.append(
                    Restraint(
                        restraint_type=RestraintType.Covalent,
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
                        max_distance=0.0,
                    )
                )
            elif c.contact is not None:
                ct = c.contact
                restraint_list.append(
                    Restraint(
                        restraint_type=RestraintType.Contact,
                        atom1=Atom(
                            chain_id=ct.token1[0],
                            residue_idx=int(ct.token1[1]),
                            atom_name="",
                            residue_name=None,
                        ),
                        atom2=Atom(
                            chain_id=ct.token2[0],
                            residue_idx=int(ct.token2[1]),
                            atom_name="",
                            residue_name=None,
                        ),
                        max_distance=ct.max_distance,
                        enable_boltz_force=ct.force,
                    )
                )
            elif c.pocket is not None:
                pk = c.pocket
                # NOTE: Pocket constraints in Boltz map contacts as a list of
                # (chain, residue) tuples. We convert the first contact pair
                # into atom1/atom2 representation. This is a lossy conversion.
                if len(pk.contacts) >= 1:
                    first_contact = pk.contacts[0]
                    restraint_list.append(
                        Restraint(
                            restraint_type=RestraintType.Pocket,
                            atom1=Atom(
                                chain_id=first_contact[0],
                                residue_idx=int(first_contact[1]),
                                atom_name="",
                                residue_name=None,
                            ),
                            atom2=Atom(
                                chain_id=pk.binder,
                                residue_idx=1,
                                atom_name="",
                                residue_name=None,
                            ),
                            max_distance=pk.max_distance,
                            enable_boltz_force=pk.force,
                            boltz_binder_chain=pk.binder,
                        )
                    )
        restraints = restraint_list if restraint_list else None

    # NOTE: Boltz inference parameters (recycling_steps, sampling_steps,
    # diffusion_samples) are CLI options, not part of the YAML config.
    # We use default values here.
    return UniAF3Config(
        sequences=sequences,
        restraints=restraints,
        seeds=[42],  # NOTE: Boltz config does not include seeds
    )
