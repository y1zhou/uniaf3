"""Adapter for converting between UniAF3Config and Protenix config."""

from __future__ import annotations

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
from uniaf3.schema.protenix import (
    ProtenixConfig,
    ProtenixConstraint,
    ProtenixContactConstraint,
    ProtenixCovalentBond,
    ProtenixDNASequence,
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
from uniaf3.vendor.chai1_glycans import _glycan_string_to_sugars_and_bonds


def _to_protenix(
    config: UniAF3Config, name: str = "uniaf3_job", strict: bool = True
) -> ProtenixJob:
    """Convert a UniAF3Config to a Protenix job."""
    warn_lossy_conversion(
        "UniAF3Config.sequences[*].id are converted to Protenix entity/copy indices; explicit chain IDs are not preserved."
    )
    sequences: list[ProtenixSequenceEntry] = []

    # Build a chain-id -> (entity, copy) mapping
    chain_to_entity: dict[str, tuple[int, int]] = {}
    for entity_idx, seq in enumerate(config.sequences, start=1):
        # NOTE: Protenix does not support assigning chain IDs to input
        # entities. The entity number is determined by the order in the
        # sequences list, and copies are controlled by the count field.
        ids = ensure_list(seq.id)
        for copy_idx, chain_id in enumerate(ids, start=1):
            chain_to_entity[chain_id] = (entity_idx, copy_idx)
        count = len(ids)

        if isinstance(seq, (Polymer, ProteinSeq)):
            if isinstance(seq, ProteinSeq) or (
                isinstance(seq, Polymer) and seq.polymer_type == PolymerType.Protein
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
                        if len(seq.templates) > 1:
                            warn_lossy_conversion(
                                "ProtenixProteinChain.templatesPath accepts one path; only the first UniAF3 ProteinSeq.templates entry is kept."
                            )
                        # NOTE: Protenix uses a single templatesPath for
                        # template .a3m/.hhr files; we use the first template.
                        tmpl = seq.templates[0]
                        pc.templatesPath = tmpl.path
                        if (
                            tmpl.boltz_enable_force
                            or tmpl.boltz_template_threshold is not None
                        ):
                            warn_lossy_conversion(
                                "UniAF3Config.sequences[*].templates.{boltz_enable_force,boltz_template_threshold} are not represented by ProtenixProteinChain.templatesPath."
                            )
                sequences.append(ProtenixSequenceEntry(proteinChain=pc))
            elif seq.polymer_type == PolymerType.DNA:
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
            elif seq.polymer_type == PolymerType.RNA:
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
        elif isinstance(seq, Ligand):
            if seq.ccd is not None:
                # We never use the ion field in Protenix
                sequences.append(
                    ProtenixSequenceEntry(
                        ligand=ProtenixLigand(
                            ligand=f"CCD_{'_'.join(seq.ccd)}", count=count
                        )
                    )
                )
            elif seq.smiles:
                sequences.append(
                    ProtenixSequenceEntry(
                        ligand=ProtenixLigand(ligand=seq.smiles, count=count)
                    )
                )
        elif isinstance(seq, Glycan):
            # Protenix glycans are represented as multi-CCD ligands or SMILES.
            # Converting from the Chai notation string is lossy since we cannot capture
            # bonds within glycans.
            glycans, glycan_bonds = _glycan_string_to_sugars_and_bonds(seq.chai_str)
            if glycan_bonds:
                err_unsupported_feature(
                    strict,
                    f"Glycan with bonds not supported in Protenix: {seq.chai_str}",
                )
            sequences.append(
                ProtenixSequenceEntry(
                    ligand=ProtenixLigand(
                        ligand=f"CCD_{'_'.join(glycans)}", count=count
                    )
                )
            )

    # Covalent bonds
    covalent_bonds: list[ProtenixCovalentBond] = []
    for r in config.covalent_bonds or []:
        try:
            entity1, copy1 = chain_to_entity[r.atom1.chain_id]
            entity2, copy2 = chain_to_entity[r.atom2.chain_id]
        except KeyError as e:
            raise KeyError(
                f"Chain ID corresponding to entity not found for covalent bond: {r}"
            ) from e

        covalent_bonds.append(
            ProtenixCovalentBond(
                entity1=entity1,
                copy1=copy1,
                position1=r.atom1.residue_idx,
                atom1=r.atom1.atom_name,
                entity2=entity2,
                copy2=copy2,
                position2=r.atom2.residue_idx,
                atom2=r.atom2.atom_name,
            )
        )

    # Constraints
    contacts: list[ProtenixContactConstraint] = []
    for r in config.contact_restraints or []:
        try:
            entity1, copy1 = chain_to_entity[r.token1.chain_id]
            entity2, copy2 = chain_to_entity[r.token2.chain_id]
        except KeyError as e:
            raise KeyError(
                f"Chain ID corresponding to entity not found for covalent bond: {r}"
            ) from e

        contacts.append(
            ProtenixContactConstraint(
                entity1=entity1,
                copy1=copy1,
                position1=r.token1.residue_idx,
                atom1=r.token1.atom_name,
                entity2=entity2,
                copy2=copy2,
                position2=r.token2.residue_idx,
                atom2=r.token2.atom_name,
                max_distance=r.max_distance,
            )
        )
    pocket: ProtenixPocketConstraint | None = None
    if config.pocket_restraints is not None:
        # Protenix supports only a single pocket constraint per
        # job. The first pocket restraint wins.
        if (
            num_pockets := len(set(x.binder_chain for x in config.pocket_restraints))
        ) > 1:
            err_unsupported_feature(
                strict,
                f"Protenix only supports a single pocket constraint, got {num_pockets}",
            )
        r = config.pocket_restraints[0]
        try:
            contact_entities = [
                (*chain_to_entity[t.chain_id], t.residue_idx) for t in r.contact_tokens
            ]
        except KeyError as e:
            raise KeyError(
                f"Chain ID corresponding to entity not found for covalent bond: {r}"
            ) from e

        binder_entity, binder_copy = chain_to_entity[r.binder_chain]
        pocket = ProtenixPocketConstraint(
            binder_chain=ProtenixPocketBinderChain(
                entity=binder_entity, copy=binder_copy
            ),
            contact_residues=[
                ProtenixPocketContactResidue(entity=x[0], copy=x[1], position=x[2])
                for x in contact_entities
            ],
            max_distance=r.max_distance,
        )

    constraint: ProtenixConstraint | None = None
    if contacts or pocket:
        constraint = ProtenixConstraint(contact=contacts or None, pocket=pocket or None)

    return ProtenixJob(
        name=name,
        sequences=sequences,
        covalent_bonds=covalent_bonds,
        constraint=constraint,
    )


def to_protenix(
    config: UniAF3Config | list[UniAF3Config],
    name: str = "uniaf3_job",
    strict: bool = False,
) -> ProtenixConfig:
    """Convert a list of UniAF3Config to a Protenix config."""
    if isinstance(config, UniAF3Config):
        # Allow passing a single UniAF3Config for convenience
        config = [config]

    if len(config) == 1:
        names = [name]
    else:
        names = [f"{name}_{i}" for i in range(1, len(config) + 1)]
    return ProtenixConfig(
        [_to_protenix(c, name=names[i], strict=strict) for i, c in enumerate(config)]
    )


def _from_protenix(job: ProtenixJob) -> UniAF3Config:
    """Convert a Protenix job to a UniAF3Config."""
    from uniaf3.utils import int_to_letters

    # Protenix does not support assigning chain IDs to input entities.
    # We generate chain IDs based on entity order (A, B, C, ...).
    sequences: list[Polymer | ProteinSeq | Ligand | Glycan] = []

    # Map entity index (1-based) → chain IDs for bond conversion
    seq_count: int = 1
    entity_to_chains: dict[int, list[str]] = {}
    for entity_id, entry in enumerate(job.sequences, start=1):
        if entry.proteinChain is not None:
            pc = entry.proteinChain
            chain_ids = [int_to_letters(seq_count + i) for i in range(pc.count)]
            entity_to_chains[entity_id] = chain_ids
            seq_count += pc.count
            mods = None
            if pc.modifications:
                mods = [
                    SequenceModification(
                        ccd=m.ptmType.removeprefix("CCD_"), position=m.ptmPosition
                    )
                    for m in pc.modifications
                ]
            seq = ProteinSeq(
                polymer_type=PolymerType.Protein,
                id=chain_ids,
                sequence=pc.sequence,
                modifications=mods,
                unpaired_msa=pc.unpairedMsaPath,
                paired_msa=pc.pairedMsaPath,
            )
            if pc.templatesPath:
                # TODO: Protenix templates use a3m/hhr format files, which need
                # conversion to m8 to be correctly parsed by other models.
                # Implement this format conversion in a future change.
                seq.templates = [StructuralTemplate(path=pc.templatesPath)]
            sequences.append(seq)
        elif entry.dnaSequence is not None:
            ds = entry.dnaSequence
            chain_ids = [int_to_letters(seq_count + i) for i in range(ds.count)]
            entity_to_chains[entity_id] = chain_ids
            seq_count += ds.count
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
                polymer_type=PolymerType.DNA,
                id=chain_ids,
                sequence=ds.sequence,
                modifications=mods,
            )
            sequences.append(seq)
        elif entry.rnaSequence is not None:
            rs = entry.rnaSequence
            chain_ids = [int_to_letters(seq_count + i) for i in range(rs.count)]
            entity_to_chains[entity_id] = chain_ids
            seq_count += rs.count
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
                polymer_type=PolymerType.RNA,
                id=chain_ids,
                sequence=rs.sequence,
                modifications=mods,
            )
            sequences.append(seq)
        elif entry.ligand is not None:
            lg = entry.ligand
            chain_ids = [int_to_letters(seq_count + i) for i in range(lg.count)]
            entity_to_chains[entity_id] = chain_ids
            seq_count += lg.count
            ligand_str = lg.ligand
            ligand_type = lg.ligand_type

            if ligand_type == "CCD":
                # CCD ligand (may be multi-CCD like "CCD_NAG_BMA_BGC")
                ccd_codes = ligand_str.removeprefix("CCD_").split("_")
                lig = Ligand(id=chain_ids, ccd=ccd_codes)

            elif ligand_type == "SMILES":
                lig = Ligand(id=chain_ids, smiles=ligand_str)
            else:
                err_unsupported_feature(
                    False, f"Unsupported FILE ligand type in Protenix entry: {lg}"
                )
                continue
            sequences.append(lig)
        elif entry.ion is not None:
            io = entry.ion
            chain_ids = [int_to_letters(seq_count + i) for i in range(io.count)]
            entity_to_chains[entity_id] = chain_ids
            seq_count += io.count
            lig = Ligand(id=chain_ids, ccd=[io.ion])
            sequences.append(lig)

    # Covalent bonds
    covalent_bonds: list[CovalentBond] = []
    for bond in job.covalent_bonds or []:
        try:
            e1_chains = entity_to_chains[bond.entity1]
            e2_chains = entity_to_chains[bond.entity2]

            if bond.copy1 is not None and bond.copy2 is not None:
                e1_chains = [e1_chains[bond.copy1 - 1]]
                e2_chains = [e2_chains[bond.copy2 - 1]]
        except KeyError as e:
            raise KeyError(
                f"Chain ID corresponding to entity not found for covalent bond: {bond}"
            ) from e
        except IndexError as e:
            raise IndexError(
                f"Copy index out of range for covalent bond: {bond}"
            ) from e

        for e1_chain, e2_chain in zip(e1_chains, e2_chains, strict=True):
            covalent_bonds.append(
                CovalentBond(
                    atom1=Atom(
                        chain_id=e1_chain,
                        residue_idx=bond.position1,
                        atom_name=bond.atom1,
                        residue_name=None,
                    ),
                    atom2=Atom(
                        chain_id=e2_chain,
                        residue_idx=bond.position2,
                        atom_name=bond.atom2,
                        residue_name=None,
                    ),
                )
            )

    # Contact and pocket constraints → restraints
    contact_rsts: list[ContactRestraint] = []
    pocket_rsts: list[PocketRestraint] = []
    if job.constraint is not None:
        for ct in job.constraint.contact or []:
            try:
                e1_chains = entity_to_chains[ct.entity1]
                e2_chains = entity_to_chains[ct.entity2]

                e1_chain = e1_chains[ct.copy1 - 1]
                e2_chain = e2_chains[ct.copy2 - 1]
            except KeyError as e:
                raise KeyError(
                    f"Chain ID corresponding to entity not found for contact constraint: {ct}"
                ) from e
            except IndexError as e:
                raise IndexError(
                    f"Copy index out of range for contact constraint: {ct}"
                ) from e

            contact_rsts.append(
                # Protenix can omit the atom_name field
                ContactRestraint(
                    token1=Atom(
                        chain_id=e1_chain,
                        residue_idx=ct.position1,
                        atom_name=ct.atom1,
                        residue_name=None,
                    ),
                    token2=Atom(
                        chain_id=e2_chain,
                        residue_idx=ct.position2,
                        atom_name=ct.atom2,
                        residue_name=None,
                    ),
                    max_distance=ct.max_distance,
                    min_distance=ct.min_distance,
                )
            )
        if (pct := job.constraint.pocket) is not None:
            try:
                binder_chain = entity_to_chains[pct.binder_chain.entity][
                    pct.binder_chain.copy_idx - 1
                ]
                contact_residues = [
                    (entity_to_chains[cr.entity][cr.copy_idx - 1], cr.position)
                    for cr in pct.contact_residues
                ]
            except KeyError as e:
                raise KeyError(
                    f"Chain ID corresponding to entity not found for pocket: {pct}"
                ) from e
            pocket_rsts.append(
                PocketRestraint(
                    binder_chain=binder_chain,
                    contact_tokens=[
                        Atom(
                            chain_id=cr_chain,
                            residue_idx=cr_pos,
                            atom_name=None,
                            residue_name=None,
                        )
                        for cr_chain, cr_pos in contact_residues
                    ],
                    max_distance=pct.max_distance,
                )
            )

    warn_lossy_conversion(
        "ProtenixConfig has no seed field; UniAF3Config.aux.seeds defaults to [42]."
    )
    return UniAF3Config(
        sequences=sequences,
        covalent_bonds=covalent_bonds or None,
        contact_restraints=contact_rsts or None,
        pocket_restraints=pocket_rsts or None,
        aux=AuxiliaryParams(seeds=[42]),
    )


def from_protenix(config: ProtenixConfig) -> list[UniAF3Config]:
    """Convert a Protenix config to a list of UniAF3Config."""
    if len(config) == 0:
        raise ValueError("ProtenixConfig must have at least one job.")

    return [_from_protenix(job) for job in config]  # ty:ignore[not-iterable]
