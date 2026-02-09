"""Adapter for converting between UniAF3Config and Protenix config."""

from __future__ import annotations

from uniaf3.adapters._helpers import _KNOWN_ION_CCD_CODES, _ensure_list
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
            # Reverse spreadsheet-style IDs: A-Z, AA-ZA, AB-ZB, ...
            # This matches the convention in UniAF3Config and AlphaFold3.
            n = chain_counter
            if n < 26:
                ids.append(chr(65 + n))
            else:
                left_char = chr(65 + (n - 26) % 26)
                right_char = chr(65 + (n - 26) // 26)
                ids.append(f"{left_char}{right_char}")
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
