"""Adapter for converting between UniAF3Config and Chai-1 config."""

from __future__ import annotations

from uniaf3.adapters._helpers import ensure_list
from uniaf3.schema.base import (
    Atom,
    AuxiliaryParams,
    Glycan,
    Ligand,
    Polymer,
    PolymerType,
    ProteinSeq,
    UniAF3Config,
)
from uniaf3.schema.chai import (
    ChaiConfig,
    ChaiEntity,
    ChaiEntityType,
    ChaiRestraint,
    ChaiRestraintType,
)


def to_chai(config: UniAF3Config) -> ChaiConfig:
    """Convert a UniAF3Config to a Chai-1 config."""
    entities: list[ChaiEntity] = []
    for seq in config.sequences:
        ids = ensure_list(seq.id)

        if isinstance(seq, Polymer):
            if isinstance(seq, ProteinSeq) or (
                isinstance(seq, Polymer) and seq.seq_type == PolymerType.Protein
            ):
                etype = ChaiEntityType.Protein
            elif seq.seq_type == PolymerType.DNA:
                etype = ChaiEntityType.DNA
            elif seq.seq_type == PolymerType.RNA:
                etype = ChaiEntityType.RNA
            else:
                raise ValueError(
                    f"Unsupported polymer type for Chai conversion: {seq.seq_type}"
                )
            seq_list = list(seq.sequence)
            if seq.modifications:
                # Chai-1 inlines modifications using CCD codes in parentheses
                pass
            for chain_id in ids:
                entities.append(
                    ChaiEntity(
                        entity_type=etype, entity_name=chain_id, sequence=seq.sequence
                    )
                )
                # NOTE: Chai-1 does not support polymer modifications in its
                # FASTA input format.
        elif isinstance(seq, Ligand):
            ids = ensure_list(seq.id)
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
            ids = ensure_list(seq.id)
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
        num_trunk_recycles=config.aux.num_trunk_recycles,
        num_diffn_timesteps=config.aux.num_diffn_timesteps,
        num_diffn_samples=config.aux.num_diffn_samples,
        num_trunk_samples=config.aux.num_trunk_samples,
        seed=config.seeds[0] if config.seeds else None,
    )


def _parse_chai_res_idx(chain: str, res_idx: str) -> Atom:
    """Parse a Chai-style residue index string into an Atom object.

    The format is ``<residue_name><position>[@atom_name]``
    (e.g. ``A219``, ``D45@CB``).
    """
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
        aux=AuxiliaryParams(
            num_trunk_recycles=config.num_trunk_recycles,
            num_diffn_timesteps=config.num_diffn_timesteps,
            num_diffn_samples=config.num_diffn_samples,
            num_trunk_samples=config.num_trunk_samples,
        ),
    )
