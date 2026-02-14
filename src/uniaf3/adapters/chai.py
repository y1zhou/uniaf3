"""Adapter for converting between UniAF3Config and Chai-1 config."""

from __future__ import annotations

from pathlib import Path

from uniaf3.adapters._helpers import ensure_list, err_unsupported_feature
from uniaf3.constant import int_to_letters
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
    UniAF3Config,
)
from uniaf3.schema.chai import (
    ChaiConfig,
    ChaiEntity,
    ChaiEntityType,
    ChaiRestraint,
    ChaiRestraintType,
)


def to_chai(config: UniAF3Config, strict: bool = False) -> ChaiConfig:
    """Convert a UniAF3Config to a Chai-1 config.

    Lossy terms:

    - CCD ligand IDs are not supported in Chai FASTA format.
    - Chain IDs are not preserved. Chai uses A, B, ..., Z, AA, AB, ... outputs.
    """
    entities: list[ChaiEntity] = []
    entity_types: dict[str, ChaiEntityType] = {}
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

            # Chai-1 inlines modifications using CCD codes in parentheses
            seq_list = list(seq.sequence)
            if seq.modifications:
                for mod in seq.modifications:
                    seq_list[mod.position - 1] = f"({mod.ccd})"
            seq_str = "".join(seq_list)

            for chain_id in ids:
                entity_types[chain_id] = etype
                entities.append(
                    ChaiEntity(
                        entity_type=etype, entity_name=chain_id, sequence=seq_str
                    )
                )
        elif isinstance(seq, Ligand):
            ids = ensure_list(seq.id)
            if seq.smiles is not None:
                lig_seq = seq.smiles

            # Chai does not support CCD ligands, but we can attempt to look up the
            # corresponding SMILES if a single CCD code is provided. If multiple CCD codes are provided, issue warning
            elif seq.ccd is not None and len(seq.ccd) == 1:
                import polars as pl

                from uniaf3.vendor.ccd import CCD_LIB

                lig_ccd = seq.ccd[0]
                lig_smiles = CCD_LIB.filter(
                    pl.col("CCD") == pl.lit(lig_ccd)
                ).get_column("SMILES")
                if lig_smiles.len() == 0:
                    err_unsupported_feature(
                        strict,
                        f"CCD ligand {lig_ccd} not found in CCD library.",
                    )
                else:
                    lig_seq = lig_smiles.item()
                    print(
                        f"Converting CCD ligand {lig_ccd} to SMILES for Chai: {lig_seq}"
                    )

            else:
                err_unsupported_feature(
                    strict, "Multi-CCD ligands are not supported in Chai."
                )
                continue
            for chain_id in ids:
                entity_types[chain_id] = ChaiEntityType.Ligand
                entities.append(
                    ChaiEntity(
                        entity_type=ChaiEntityType.Ligand,
                        entity_name=chain_id,
                        sequence=lig_seq,
                    )
                )
        elif isinstance(seq, Glycan):
            ids = ensure_list(seq.id)
            for chain_id in ids:
                entity_types[chain_id] = ChaiEntityType.Glycan
                entities.append(
                    ChaiEntity(
                        entity_type=ChaiEntityType.Glycan,
                        entity_name=chain_id,
                        sequence=seq.chai_str,
                    )
                )

    # Map original chain IDs to Chai-ordered chain IDs
    entity_id_map = {
        entity.entity_name: int_to_letters(idx)
        for idx, entity in enumerate(entities, start=1)
    }

    # Restraints → Chai CSV restraints
    restraints: list[ChaiRestraint] = []
    restraint_idx: int = 0
    for r in config.covalent_bonds or []:
        res_idx: list[str] = []
        for atom in [r.atom1, r.atom2]:
            if entity_types[atom.chain_id] in {
                ChaiEntityType.Ligand,
                ChaiEntityType.Glycan,
            }:
                res_idx.append(f"@{atom.atom_name}")
            else:
                if atom.residue_name is None:
                    raise ValueError(
                        f"Missing residue name for covalent bond atom: {r.atom1}"
                    )
                res_idx.append(
                    f"{r.atom1.residue_name}{r.atom1.residue_idx}@{r.atom1.atom_name}"
                )

        restraints.append(
            ChaiRestraint(
                restraint_id=f"restraint{restraint_idx}",
                connection_type=ChaiRestraintType.Covalent,
                chainA=entity_id_map[r.atom1.chain_id],
                res_idxA=res_idx[0],
                chainB=entity_id_map[r.atom2.chain_id],
                res_idxB=res_idx[1],
                max_distance_angstrom=0.0,
                comment=r.description,
            )
        )
        restraint_idx += 1
    for r in config.contact_restraints or []:
        for atom in [r.token1, r.token2]:
            if entity_types[atom.chain_id] not in {
                ChaiEntityType.Protein,
                ChaiEntityType.DNA,
                ChaiEntityType.RNA,
            }:
                raise ValueError(
                    f"Contact restraints are only supported between protein/DNA/RNA entities in Chai conversion: {atom}"
                )
            if atom.residue_name is None:
                raise ValueError(
                    f"Missing residue name for contact restraint token: {atom}"
                )
        restraints.append(
            ChaiRestraint(
                restraint_id=f"restraint{restraint_idx}",
                connection_type=ChaiRestraintType.Contact,
                chainA=entity_id_map[r.token1.chain_id],
                res_idxA=f"{r.token1.residue_name}{r.token1.residue_idx}",
                chainB=entity_id_map[r.token2.chain_id],
                res_idxB=f"{r.token2.residue_name}{r.token2.residue_idx}",
                max_distance_angstrom=r.max_distance,
                min_distance_angstrom=r.min_distance,
                comment=r.description,
            )
        )
        restraint_idx += 1
    for r in config.pocket_restraints or []:
        for t in r.contact_tokens:
            if t.residue_name is None:
                raise ValueError(
                    f"Missing residue name for pocket restraint token: {t}"
                )
            restraints.append(
                ChaiRestraint(
                    restraint_id=f"restraint{restraint_idx}",
                    connection_type=ChaiRestraintType.Pocket,
                    chainA=entity_id_map[r.binder_chain],
                    res_idxA=None,
                    chainB=entity_id_map[t.chain_id],
                    res_idxB=f"{t.residue_name}{t.residue_idx}",
                    max_distance_angstrom=r.max_distance,
                    min_distance_angstrom=r.min_distance,
                    comment=r.description,
                )
            )
            restraint_idx += 1

    return ChaiConfig(
        entities=entities,
        restraints=restraints or None,
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
    msa_dir = config.msa_directory
    if msa_dir is not None:
        msa_dir_path = Path(msa_dir)
        # Use subdirectory following MMSeqs2 output file structure
        if (msa_dir_path / "a3ms").exists():
            msa_dir_path = msa_dir_path / "a3ms"
    else:
        msa_dir_path = None
    for i, entity in enumerate(config.entities, start=1):
        if entity.entity_type == ChaiEntityType.Protein:
            # TODO: extract inline modifications from polymer sequences
            seq = ProteinSeq(
                seq_type=PolymerType.Protein,
                id=int_to_letters(i),
                description=entity.entity_name,
                sequence=entity.sequence,
                modifications=None,
                msa_dir=str(msa_dir_path) if msa_dir_path else None,
                templates=None,
            )
            sequences.append(seq)
        elif entity.entity_type == ChaiEntityType.DNA:
            seq = Polymer(
                seq_type=PolymerType.DNA,
                id=int_to_letters(i),
                description=entity.entity_name,
                sequence=entity.sequence,
                modifications=None,
            )
            sequences.append(seq)
        elif entity.entity_type == ChaiEntityType.RNA:
            seq = Polymer(
                seq_type=PolymerType.RNA,
                id=int_to_letters(i),
                description=entity.entity_name,
                sequence=entity.sequence,
                modifications=None,
            )
            sequences.append(seq)
        elif entity.entity_type == ChaiEntityType.Ligand:
            lig = Ligand(
                id=int_to_letters(i),
                description=entity.entity_name,
                smiles=entity.sequence,
            )
            sequences.append(lig)
        elif entity.entity_type == ChaiEntityType.Glycan:
            glycan = Glycan(
                id=int_to_letters(i),
                description=entity.entity_name,
                chai_str=entity.sequence,
            )
            sequences.append(glycan)

    # NOTE: Converting Chai restraints back to UniAF3 restraints requires
    # parsing the residue index format (e.g. "D45@CB") which is complex.
    covalent_bonds: list[CovalentBond] = []
    contact_restraints: list[ContactRestraint] = []
    pocket_restraints: list[PocketRestraint] = []
    if config.restraints:
        for cr in config.restraints:
            atom1 = _parse_chai_res_idx(cr.chainA, cr.res_idxA)
            atom2 = _parse_chai_res_idx(cr.chainB, cr.res_idxB)
            if cr.connection_type == ChaiRestraintType.Covalent:
                pass
            elif cr.connection_type == ChaiRestraintType.Contact:
                pass
            elif cr.connection_type == ChaiRestraintType.Pocket:
                pass
            else:
                continue

    return UniAF3Config(
        sequences=sequences,
        covalent_bonds=covalent_bonds or None,
        contact_restraints=contact_restraints or None,
        pocket_restraints=pocket_restraints or None,
        seeds=[config.seed] if config.seed is not None else [42],
        aux=AuxiliaryParams(
            num_trunk_recycles=config.num_trunk_recycles,
            num_diffn_timesteps=config.num_diffn_timesteps,
            num_diffn_samples=config.num_diffn_samples,
            num_trunk_samples=config.num_trunk_samples,
        ),
    )
