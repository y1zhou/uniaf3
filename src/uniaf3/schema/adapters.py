"""Adapters to convert between UniAF3Config and model-specific configs.

Each model has a ``to_*`` and ``from_*`` function pair. Items that cannot be
mapped are annotated with ``# NOTE: `` comments for future attention.
"""

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
    AF3Template,
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
from uniaf3.schema.chai import (
    ChaiConfig,
    ChaiEntity,
    ChaiEntityType,
    ChaiRestraint,
    ChaiRestraintType,
)
from uniaf3.schema.protenix import (
    ProtenixConfig,
    ProtenixConstraint,
    ProtenixContactConstraint,
    ProtenixCovalentBond,
    ProtenixDNASequence,
    ProtenixIon,
    ProtenixJob,
    ProtenixLigand,
    ProtenixNucleotideModification,
    ProtenixPocketBinderChain,
    ProtenixPocketConstraint,
    ProtenixPocketContactResidue,
    ProtenixProteinChain,
    ProtenixProteinModification,
    ProtenixRNASequence,
    ProtenixSequenceEntry,
)

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

_KNOWN_ION_CCD_CODES = frozenset(
    {
        "MG",
        "ZN",
        "CA",
        "FE",
        "NA",
        "K",
        "CL",
        "MN",
        "CU",
        "CO",
        "NI",
        "CD",
        "BR",
        "I",
        "SE",
        "LI",
        "BA",
        "CS",
        "SR",
        "RB",
        "PB",
        "AG",
        "HG",
        "TL",
        "F",
    }
)


def _ensure_list(val: str | list[str]) -> list[str]:
    """Normalize id field to a list."""
    return val if isinstance(val, list) else [val]


def _first_or_str(val: list[str]) -> str | list[str]:
    """Return a single string when the list has one element."""
    return val[0] if len(val) == 1 else val


# ===========================================================================
# AlphaFold3
# ===========================================================================


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
            # Determine msa_dir from paths if available
            msa_dir: str | None = None
            # NOTE: AF3 provides MSA inline or via path; we store the path.
            # The msa_dir concept does not directly map; store unpairedMsaPath.
            seq = ProteinSeq(
                seq_type=PolymerType.Protein,
                id=p.id,
                sequence=p.sequence,
                modifications=mods,
                description=p.description,
                templates=templates,
                msa_dir=msa_dir,
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


# ===========================================================================
# Boltz
# ===========================================================================


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


# ===========================================================================
# Chai-1
# ===========================================================================


def to_chai(config: UniAF3Config) -> ChaiConfig:
    """Convert a UniAF3Config to a Chai-1 config."""
    entities: list[ChaiEntity] = []
    for seq in config.sequences:
        if isinstance(seq, ProteinSeq):
            ids = _ensure_list(seq.id)
            for chain_id in ids:
                entities.append(
                    ChaiEntity(
                        entity_type=ChaiEntityType.Protein,
                        entity_name=chain_id,
                        sequence=seq.sequence,
                    )
                )
                # NOTE: Chai-1 does not support polymer modifications in its
                # FASTA input format.
        elif isinstance(seq, Polymer):
            ids = _ensure_list(seq.id)
            if seq.seq_type == PolymerType.Protein:
                etype = ChaiEntityType.Protein
            elif seq.seq_type == PolymerType.DNA:
                etype = ChaiEntityType.DNA
            elif seq.seq_type == PolymerType.RNA:
                etype = ChaiEntityType.RNA
            else:
                continue
            for chain_id in ids:
                entities.append(
                    ChaiEntity(
                        entity_type=etype,
                        entity_name=chain_id,
                        sequence=seq.sequence,
                    )
                )
                # NOTE: Chai-1 does not support polymer modifications in its
                # FASTA input format.
        elif isinstance(seq, Ligand):
            ids = _ensure_list(seq.id)
            smiles_or_ccd = seq.smiles or (seq.ccd[0] if seq.ccd else "")
            for chain_id in ids:
                entities.append(
                    ChaiEntity(
                        entity_type=ChaiEntityType.Ligand,
                        entity_name=chain_id,
                        sequence=smiles_or_ccd,
                    )
                )
        elif isinstance(seq, Glycan):
            ids = _ensure_list(seq.id)
            for chain_id in ids:
                entities.append(
                    ChaiEntity(
                        entity_type=ChaiEntityType.Glycan,
                        entity_name=chain_id,
                        sequence=seq.chai_str,
                    )
                )

    # Restraints → Chai CSV restraints
    restraints: list[ChaiRestraint] | None = None
    if config.restraints:
        restraint_list: list[ChaiRestraint] = []
        for i, r in enumerate(config.restraints):
            if r.restraint_type == RestraintType.Covalent:
                # Format: residueName+position@atomName
                res_a = (
                    f"{r.atom1.residue_name or ''}{r.atom1.residue_idx}"
                    f"@{r.atom1.atom_name}"
                )
                res_b = (
                    f"{r.atom2.residue_name or ''}{r.atom2.residue_idx}"
                    f"@{r.atom2.atom_name}"
                )
                restraint_list.append(
                    ChaiRestraint(
                        restraint_id=f"restraint_{i}",
                        chainA=r.atom1.chain_id,
                        res_idxA=res_a,
                        chainB=r.atom2.chain_id,
                        res_idxB=res_b,
                        max_distance_angstrom=r.max_distance,
                        connection_type=ChaiRestraintType.Covalent,
                    )
                )
            elif r.restraint_type == RestraintType.Contact:
                res_a = f"{r.atom1.residue_name or ''}{r.atom1.residue_idx}"
                res_b = f"{r.atom2.residue_name or ''}{r.atom2.residue_idx}"
                restraint_list.append(
                    ChaiRestraint(
                        restraint_id=f"restraint_{i}",
                        chainA=r.atom1.chain_id,
                        res_idxA=res_a,
                        chainB=r.atom2.chain_id,
                        res_idxB=res_b,
                        max_distance_angstrom=r.max_distance,
                        connection_type=ChaiRestraintType.Contact,
                    )
                )
            elif r.restraint_type == RestraintType.Pocket:
                # NOTE: Chai pocket restraints leave the binder residue index
                # empty. We use atom1 for the pocket residue and atom2's chain
                # for the binder.
                res_a = f"{r.atom1.residue_name or ''}{r.atom1.residue_idx}"
                restraint_list.append(
                    ChaiRestraint(
                        restraint_id=f"restraint_{i}",
                        chainA=r.atom2.chain_id,
                        res_idxA="",
                        chainB=r.atom1.chain_id,
                        res_idxB=res_a,
                        max_distance_angstrom=r.max_distance,
                        connection_type=ChaiRestraintType.Pocket,
                    )
                )
        restraints = restraint_list if restraint_list else None

    return ChaiConfig(
        entities=entities,
        restraints=restraints,
        num_trunk_recycles=config.num_trunk_recycles,
        num_diffn_timesteps=config.num_diffn_timesteps,
        num_diffn_samples=config.num_diffn_samples,
        num_trunk_samples=config.num_trunk_samples,
        seed=config.seeds[0] if config.seeds else None,
    )


def from_chai(config: ChaiConfig) -> UniAF3Config:
    """Convert a Chai-1 config to a UniAF3Config."""
    sequences: list[Polymer | ProteinSeq | Ligand | Glycan] = []
    for entity in config.entities:
        if entity.entity_type == ChaiEntityType.Protein:
            seq = ProteinSeq(
                seq_type=PolymerType.Protein,
                id=entity.entity_name,
                sequence=entity.sequence,
            )
            sequences.append(seq)
        elif entity.entity_type == ChaiEntityType.DNA:
            seq = Polymer(
                seq_type=PolymerType.DNA,
                id=entity.entity_name,
                sequence=entity.sequence,
            )
            sequences.append(seq)
        elif entity.entity_type == ChaiEntityType.RNA:
            seq = Polymer(
                seq_type=PolymerType.RNA,
                id=entity.entity_name,
                sequence=entity.sequence,
            )
            sequences.append(seq)
        elif entity.entity_type == ChaiEntityType.Ligand:
            # NOTE: Cannot distinguish CCD vs SMILES from Chai entity alone.
            # Assume SMILES if not a simple CCD code pattern.
            lig = Ligand(
                id=entity.entity_name,
                smiles=entity.sequence,
            )
            sequences.append(lig)
        elif entity.entity_type == ChaiEntityType.Glycan:
            glycan = Glycan(
                id=entity.entity_name,
                chai_str=entity.sequence,
            )
            sequences.append(glycan)

    # NOTE: Converting Chai restraints back to UniAF3 restraints requires
    # parsing the residue index format (e.g. "D45@CB") which is complex.
    # For now we store the raw data.
    restraints: list[Restraint] | None = None
    if config.restraints:
        restraint_list: list[Restraint] = []
        for cr in config.restraints:
            if cr.connection_type == ChaiRestraintType.Covalent:
                rtype = RestraintType.Covalent
            elif cr.connection_type == ChaiRestraintType.Contact:
                rtype = RestraintType.Contact
            elif cr.connection_type == ChaiRestraintType.Pocket:
                rtype = RestraintType.Pocket
            else:
                continue

            # NOTE: Parsing Chai residue index format back to Atom fields is
            # approximate; atom_name extracted from "@..." suffix if present.
            def _parse_chai_res_idx(chain: str, res_idx: str) -> Atom:
                atom_name = ""
                residue_name = None
                residue_idx = 1
                if "@" in res_idx:
                    parts = res_idx.split("@")
                    res_part = parts[0]
                    atom_name = parts[1]
                else:
                    res_part = res_idx

                if res_part:
                    # Extract numeric suffix as residue index
                    num_str = ""
                    name_str = ""
                    for ch in res_part:
                        if ch.isdigit():
                            num_str += ch
                        else:
                            name_str += ch
                    if num_str:
                        residue_idx = int(num_str)
                    if name_str:
                        residue_name = name_str

                return Atom(
                    chain_id=chain,
                    residue_idx=residue_idx,
                    atom_name=atom_name,
                    residue_name=residue_name,
                )

            atom1 = _parse_chai_res_idx(cr.chainA, cr.res_idxA)
            atom2 = _parse_chai_res_idx(cr.chainB, cr.res_idxB)
            restraint_list.append(
                Restraint(
                    restraint_type=rtype,
                    atom1=atom1,
                    atom2=atom2,
                    max_distance=cr.max_distance_angstrom,
                )
            )
        restraints = restraint_list if restraint_list else None

    seeds = [config.seed] if config.seed is not None else [42]

    return UniAF3Config(
        sequences=sequences,
        restraints=restraints,
        seeds=seeds,
        num_trunk_recycles=config.num_trunk_recycles,
        num_diffn_timesteps=config.num_diffn_timesteps,
        num_diffn_samples=config.num_diffn_samples,
        num_trunk_samples=config.num_trunk_samples,
    )


# ===========================================================================
# Protenix
# ===========================================================================


def to_protenix(config: UniAF3Config, name: str = "uniaf3_job") -> ProtenixConfig:
    """Convert a UniAF3Config to a Protenix config."""
    sequences: list[ProtenixSequenceEntry] = []

    # Build a chain-id → entity-index mapping for covalent bonds
    chain_to_entity: dict[str, int] = {}
    entity_idx = 1
    for seq in config.sequences:
        if isinstance(seq, (Polymer, ProteinSeq)):
            ids = _ensure_list(seq.id)
            # NOTE: Protenix does not support assigning chain IDs to input
            # entities. The entity number is determined by the order in the
            # sequences list, and copies are controlled by the count field.
            chain_to_entity.update({cid: entity_idx for cid in ids})
            count = len(ids)

            if isinstance(seq, ProteinSeq) or (
                isinstance(seq, Polymer) and seq.seq_type == PolymerType.Protein
            ):
                mods = None
                if seq.modifications:
                    mods = [
                        ProtenixProteinModification(
                            ptmType=f"CCD_{m.ccd}", ptmPosition=m.position
                        )
                        for m in seq.modifications
                    ]
                pc = ProtenixProteinChain(
                    sequence=seq.sequence,
                    count=count,
                    modifications=mods,
                )
                if isinstance(seq, ProteinSeq):
                    pc.unpairedMsaPath = seq.unpaired_msa
                    pc.pairedMsaPath = seq.paired_msa
                    if seq.templates:
                        # NOTE: Protenix uses a single templatesPath for
                        # template .a3m/.hhr files; we use the first template.
                        pc.templatesPath = seq.templates[0].path
                sequences.append(ProtenixSequenceEntry(proteinChain=pc))
            elif seq.seq_type == PolymerType.DNA:
                mods = None
                if seq.modifications:
                    mods = [
                        ProtenixNucleotideModification(
                            modificationType=f"CCD_{m.ccd}", basePosition=m.position
                        )
                        for m in seq.modifications
                    ]
                sequences.append(
                    ProtenixSequenceEntry(
                        dnaSequence=ProtenixDNASequence(
                            sequence=seq.sequence, count=count, modifications=mods
                        )
                    )
                )
            elif seq.seq_type == PolymerType.RNA:
                mods = None
                if seq.modifications:
                    mods = [
                        ProtenixNucleotideModification(
                            modificationType=f"CCD_{m.ccd}", basePosition=m.position
                        )
                        for m in seq.modifications
                    ]
                sequences.append(
                    ProtenixSequenceEntry(
                        rnaSequence=ProtenixRNASequence(
                            sequence=seq.sequence, count=count, modifications=mods
                        )
                    )
                )
            entity_idx += 1
        elif isinstance(seq, Ligand):
            ids = _ensure_list(seq.id)
            chain_to_entity.update({cid: entity_idx for cid in ids})
            count = len(ids)
            if seq.ccd:
                for ccd_code in seq.ccd:
                    if ccd_code in _KNOWN_ION_CCD_CODES:
                        sequences.append(
                            ProtenixSequenceEntry(
                                ion=ProtenixIon(ion=ccd_code, count=count)
                            )
                        )
                    else:
                        sequences.append(
                            ProtenixSequenceEntry(
                                ligand=ProtenixLigand(
                                    ligand=f"CCD_{ccd_code}", count=count
                                )
                            )
                        )
            elif seq.smiles:
                sequences.append(
                    ProtenixSequenceEntry(
                        ligand=ProtenixLigand(ligand=seq.smiles, count=count)
                    )
                )
            entity_idx += 1
        elif isinstance(seq, Glycan):
            ids = _ensure_list(seq.id)
            chain_to_entity.update({cid: entity_idx for cid in ids})
            count = len(ids)
            # NOTE: Glycans in Protenix are represented as multi-CCD ligands
            # or SMILES. Using the Chai notation string as SMILES is a lossy
            # conversion.
            sequences.append(
                ProtenixSequenceEntry(
                    ligand=ProtenixLigand(ligand=seq.chai_str, count=count)
                )
            )
            entity_idx += 1

    # Covalent bonds
    covalent_bonds: list[ProtenixCovalentBond] | None = None
    # Constraints
    constraint: ProtenixConstraint | None = None

    if config.restraints:
        bond_list: list[ProtenixCovalentBond] = []
        contact_list: list[ProtenixContactConstraint] = []
        pocket: ProtenixPocketConstraint | None = None

        for r in config.restraints:
            eidx1 = chain_to_entity.get(r.atom1.chain_id, 0)
            eidx2 = chain_to_entity.get(r.atom2.chain_id, 0)

            if r.restraint_type == RestraintType.Covalent:
                bond_list.append(
                    ProtenixCovalentBond(
                        entity1=str(eidx1),
                        position1=str(r.atom1.residue_idx),
                        atom1=r.atom1.atom_name,
                        entity2=str(eidx2),
                        position2=str(r.atom2.residue_idx),
                        atom2=r.atom2.atom_name,
                    )
                )
            elif r.restraint_type == RestraintType.Contact:
                contact_list.append(
                    ProtenixContactConstraint(
                        entity1=eidx1,
                        copy1=1,
                        position1=r.atom1.residue_idx,
                        atom1=r.atom1.atom_name if r.atom1.atom_name else None,
                        entity2=eidx2,
                        copy2=1,
                        position2=r.atom2.residue_idx,
                        atom2=r.atom2.atom_name if r.atom2.atom_name else None,
                        max_distance=r.max_distance,
                    )
                )
            elif r.restraint_type == RestraintType.Pocket:
                # NOTE: Protenix supports only a single pocket constraint per
                # job. The last pocket restraint wins.
                binder_entity = eidx2
                if r.boltz_binder_chain:
                    binder_entity = chain_to_entity.get(r.boltz_binder_chain, eidx2)
                pocket = ProtenixPocketConstraint(
                    binder_chain=ProtenixPocketBinderChain(
                        entity=binder_entity, copy_idx=1
                    ),
                    contact_residues=[
                        ProtenixPocketContactResidue(
                            entity=eidx1,
                            copy_idx=1,
                            position=r.atom1.residue_idx,
                        )
                    ],
                    max_distance=r.max_distance,
                )

        covalent_bonds = bond_list if bond_list else None
        if contact_list or pocket:
            constraint = ProtenixConstraint(
                contact=contact_list if contact_list else None,
                pocket=pocket,
            )

    job = ProtenixJob(
        name=name,
        sequences=sequences,
        covalent_bonds=covalent_bonds,
        constraint=constraint,
    )
    return ProtenixConfig(jobs=[job])


def from_protenix(config: ProtenixConfig) -> UniAF3Config:
    """Convert a Protenix config to a UniAF3Config.

    Only the first job is converted when multiple jobs are present.
    """
    if not config.jobs:
        raise ValueError("ProtenixConfig must have at least one job.")

    job = config.jobs[0]
    sequences: list[Polymer | ProteinSeq | Ligand | Glycan] = []

    # NOTE: Protenix does not support assigning chain IDs to input entities.
    # We generate chain IDs based on entity order (A, B, C, ...).
    chain_counter = 0

    def _next_chain_ids(count: int) -> str | list[str]:
        nonlocal chain_counter
        ids = []
        for _ in range(count):
            # Generate spreadsheet-style IDs: A-Z, AA-ZA, AB-ZB, ...
            n = chain_counter
            if n < 26:
                ids.append(chr(65 + n))
            else:
                first = chr(65 + (n - 26) % 26)
                second = chr(65 + (n - 26) // 26)
                ids.append(f"{first}{second}")
            chain_counter += 1
        return ids[0] if len(ids) == 1 else ids

    # Map entity index (1-based) → chain IDs for bond conversion
    entity_to_chains: dict[int, list[str]] = {}
    entity_idx = 1

    for entry in job.sequences:
        if entry.proteinChain is not None:
            pc = entry.proteinChain
            chain_ids = _next_chain_ids(pc.count)
            entity_to_chains[entity_idx] = _ensure_list(chain_ids)
            mods = None
            if pc.modifications:
                mods = [
                    SequenceModification(
                        ccd=m.ptmType.removeprefix("CCD_"), position=m.ptmPosition
                    )
                    for m in pc.modifications
                ]
            seq = ProteinSeq(
                seq_type=PolymerType.Protein,
                id=chain_ids,
                sequence=pc.sequence,
                modifications=mods,
            )
            sequences.append(seq)
        elif entry.dnaSequence is not None:
            ds = entry.dnaSequence
            chain_ids = _next_chain_ids(ds.count)
            entity_to_chains[entity_idx] = _ensure_list(chain_ids)
            mods = None
            if ds.modifications:
                mods = [
                    SequenceModification(
                        ccd=m.modificationType.removeprefix("CCD_"),
                        position=m.basePosition,
                    )
                    for m in ds.modifications
                ]
            seq = Polymer(
                seq_type=PolymerType.DNA,
                id=chain_ids,
                sequence=ds.sequence,
                modifications=mods,
            )
            sequences.append(seq)
        elif entry.rnaSequence is not None:
            rs = entry.rnaSequence
            chain_ids = _next_chain_ids(rs.count)
            entity_to_chains[entity_idx] = _ensure_list(chain_ids)
            mods = None
            if rs.modifications:
                mods = [
                    SequenceModification(
                        ccd=m.modificationType.removeprefix("CCD_"),
                        position=m.basePosition,
                    )
                    for m in rs.modifications
                ]
            seq = Polymer(
                seq_type=PolymerType.RNA,
                id=chain_ids,
                sequence=rs.sequence,
                modifications=mods,
            )
            sequences.append(seq)
        elif entry.ligand is not None:
            lg = entry.ligand
            chain_ids = _next_chain_ids(lg.count)
            entity_to_chains[entity_idx] = _ensure_list(chain_ids)
            ligand_str = lg.ligand
            if ligand_str.startswith("CCD_"):
                # CCD ligand (may be multi-CCD like "CCD_NAG_BMA_BGC")
                ccd_codes = ligand_str.removeprefix("CCD_").split("_")
                lig = Ligand(id=chain_ids, ccd=ccd_codes)
            else:
                # NOTE: Cannot distinguish SMILES from FILE_ path; assume
                # SMILES if it does not start with FILE_.
                smiles = ligand_str.removeprefix("FILE_")
                lig = Ligand(id=chain_ids, smiles=smiles)
            sequences.append(lig)
        elif entry.ion is not None:
            io = entry.ion
            chain_ids = _next_chain_ids(io.count)
            entity_to_chains[entity_idx] = _ensure_list(chain_ids)
            lig = Ligand(id=chain_ids, ccd=[io.ion])
            sequences.append(lig)
        entity_idx += 1

    # Covalent bonds → restraints
    restraints: list[Restraint] | None = None
    if job.covalent_bonds:
        restraint_list: list[Restraint] = []
        for bond in job.covalent_bonds:
            e1_chains = entity_to_chains.get(int(bond.entity1), ["?"])
            e2_chains = entity_to_chains.get(int(bond.entity2), ["?"])
            restraint_list.append(
                Restraint(
                    restraint_type=RestraintType.Covalent,
                    atom1=Atom(
                        chain_id=e1_chains[0],
                        residue_idx=int(bond.position1),
                        atom_name=bond.atom1,
                        residue_name=None,
                    ),
                    atom2=Atom(
                        chain_id=e2_chains[0],
                        residue_idx=int(bond.position2),
                        atom_name=bond.atom2,
                        residue_name=None,
                    ),
                    max_distance=0.0,
                )
            )
        restraints = restraint_list if restraint_list else None

    # Contact and pocket constraints → restraints
    if job.constraint:
        if restraints is None:
            restraints = []
        if job.constraint.contact:
            for ct in job.constraint.contact:
                e1_chains = entity_to_chains.get(ct.entity1, ["?"])
                e2_chains = entity_to_chains.get(ct.entity2, ["?"])
                restraints.append(
                    Restraint(
                        restraint_type=RestraintType.Contact,
                        atom1=Atom(
                            chain_id=e1_chains[0],
                            residue_idx=ct.position1,
                            atom_name=ct.atom1 or "",
                            residue_name=None,
                        ),
                        atom2=Atom(
                            chain_id=e2_chains[0],
                            residue_idx=ct.position2,
                            atom_name=ct.atom2 or "",
                            residue_name=None,
                        ),
                        max_distance=ct.max_distance,
                    )
                )
        if job.constraint.pocket:
            pk = job.constraint.pocket
            binder_chains = entity_to_chains.get(pk.binder_chain.entity, ["?"])
            for cr in pk.contact_residues:
                cr_chains = entity_to_chains.get(cr.entity, ["?"])
                restraints.append(
                    Restraint(
                        restraint_type=RestraintType.Pocket,
                        atom1=Atom(
                            chain_id=cr_chains[0],
                            residue_idx=cr.position,
                            atom_name="",
                            residue_name=None,
                        ),
                        atom2=Atom(
                            chain_id=binder_chains[0],
                            residue_idx=1,
                            atom_name="",
                            residue_name=None,
                        ),
                        max_distance=pk.max_distance,
                        boltz_binder_chain=binder_chains[0],
                    )
                )
        if not restraints:
            restraints = None

    # NOTE: Protenix seeds are passed as CLI arguments, not in the JSON config.
    return UniAF3Config(
        sequences=sequences,
        restraints=restraints,
        seeds=[42],  # NOTE: Protenix config does not include seeds
    )
