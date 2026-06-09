"""Build UniAF3 configs from coordinate structure files."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from warnings import warn

import gemmi

from uniaf3.schema.base import (
    Atom,
    AuxiliaryParams,
    ContactRestraint,
    CovalentBond,
    Ligand,
    PocketRestraint,
    Polymer,
    PolymerType,
    ProteinSeq,
    SequenceModification,
    UniAF3Config,
)

SeqSource = Literal["full", "observed"]
NonCovalentConnections = Literal["ignore", "contacts", "pockets"]


@dataclass(frozen=True)
class _ImportedAtom:
    chain_id: str
    residue_idx: int
    atom_name: str
    residue_name: str | None
    kind: Literal["polymer", "ligand"]

    def to_atom(self) -> Atom:
        return Atom(
            chain_id=self.chain_id,
            residue_idx=self.residue_idx,
            atom_name=self.atom_name,
            residue_name=self.residue_name,
        )


@dataclass
class _PolymerCopy:
    entity_key: str
    entity_chain_id: str
    polymer_type: PolymerType
    sequence: str
    modifications: tuple[SequenceModification, ...]
    description: str | None
    used_full_sequence: bool


@dataclass
class _LigandCopy:
    entity_key: str
    entity_chain_id: str
    ccd: tuple[str, ...]
    description: str | None


def from_structure_file(
    struct_file: str | Path,
    *,
    seq_source: SeqSource = "full",
    chains: Sequence[str] | None = None,
    include_ligands: bool | None = None,
    include_waters: bool = False,
    model_index: int = 0,
    strict: bool = False,
    non_covalent_connections: NonCovalentConnections = "ignore",
) -> UniAF3Config:
    """Generate a UniAF3 config from a PDB or mmCIF structure file."""
    if seq_source not in {"full", "observed"}:
        raise ValueError("seq_source must be one of: full, observed")
    if non_covalent_connections not in {"ignore", "contacts", "pockets"}:
        raise ValueError(
            "non_covalent_connections must be one of: ignore, contacts, pockets"
        )

    struct_path = Path(struct_file).expanduser().resolve()
    if not struct_path.exists():
        raise FileNotFoundError(f"Structure file not found: {struct_path}")

    st = gemmi.read_structure(str(struct_path), format=gemmi.CoorFormat.Detect)
    st.setup_entities()
    st.assign_label_seq_id()

    if not 0 <= model_index < len(st):
        raise IndexError(
            f"model_index {model_index} out of range for structure with {len(st)} models"
        )

    selected_author_chains = set(chains) if chains is not None else None
    include_ligands_effective = (
        selected_author_chains is None if include_ligands is None else include_ligands
    )

    model = st[model_index]
    entity_by_subchain = _entity_by_subchain(st)
    residue_atoms: dict[tuple[str, str, str], _ImportedAtom] = {}

    polymer_copies = _collect_polymer_copies(
        st,
        model,
        entity_by_subchain,
        seq_source,
        selected_author_chains,
        strict,
        residue_atoms,
    )
    ligand_copies = _collect_ligand_copies(
        model,
        entity_by_subchain,
        selected_author_chains,
        include_ligands_effective,
        include_waters,
        residue_atoms,
    )

    sequences: list[Polymer | ProteinSeq | Ligand] = []
    sequences.extend(_build_polymer_entries(polymer_copies))
    sequences.extend(_build_ligand_entries(ligand_copies))

    covalent_bonds, contact_restraints, pocket_restraints = _collect_connections(
        st,
        residue_atoms,
        strict,
        non_covalent_connections,
    )

    return UniAF3Config(
        sequences=sequences,
        covalent_bonds=covalent_bonds or None,
        contact_restraints=contact_restraints or None,
        pocket_restraints=pocket_restraints or None,
        aux=AuxiliaryParams(name=st.name or struct_path.stem),
    )


def _entity_by_subchain(st: gemmi.Structure) -> dict[str, gemmi.Entity]:
    result: dict[str, gemmi.Entity] = {}
    for entity in st.entities:
        for subchain in entity.subchains:
            result[subchain] = entity
    return result


def _collect_polymer_copies(
    st: gemmi.Structure,
    model: gemmi.Model,
    entity_by_subchain: dict[str, gemmi.Entity],
    seq_source: SeqSource,
    selected_author_chains: set[str] | None,
    strict: bool,
    residue_atoms: dict[tuple[str, str, str], _ImportedAtom],
) -> list[_PolymerCopy]:
    copies: list[_PolymerCopy] = []
    for chain in model:
        if (
            selected_author_chains is not None
            and chain.name not in selected_author_chains
        ):
            continue

        polymer = chain.get_polymer()
        if not polymer:
            continue

        gemmi_polymer_type = polymer.check_polymer_type()
        polymer_type = _to_uniaf3_polymer_type(gemmi_polymer_type, strict)
        if polymer_type is None:
            continue

        residues = list(polymer.first_conformer())
        if not residues:
            continue

        entity = st.get_entity_of(polymer)
        entity_chain_id = residues[0].subchain or chain.name
        entity = entity or entity_by_subchain.get(entity_chain_id)
        full_sequence_names = list(entity.full_sequence) if entity is not None else []
        use_full_sequence = seq_source == "full" and bool(full_sequence_names)

        if use_full_sequence:
            sequence = _residue_names_to_sequence(
                full_sequence_names,
                polymer_type,
                strict,
            )
        else:
            sequence = _residue_names_to_sequence(
                [res.name for res in residues],
                polymer_type,
                strict,
            )

        modifications = _polymer_modifications(
            residues,
            sequence,
            full_sequence_names if use_full_sequence else None,
            strict,
        )

        for observed_idx, res in enumerate(residues, start=1):
            residue_idx = _uniaf3_residue_idx(res, observed_idx, use_full_sequence)
            if residue_idx is None or residue_idx > len(sequence):
                continue
            residue_atoms[_residue_key(chain.name, res)] = _ImportedAtom(
                chain_id=entity_chain_id,
                residue_idx=residue_idx,
                atom_name="",
                residue_name=sequence[residue_idx - 1],
                kind="polymer",
            )

        entity_key = entity.name if entity is not None else entity_chain_id
        if not use_full_sequence:
            entity_key = f"{entity_key}:{sequence}"

        copies.append(
            _PolymerCopy(
                entity_key=entity_key,
                entity_chain_id=entity_chain_id,
                polymer_type=polymer_type,
                sequence=sequence,
                modifications=tuple(modifications),
                description=_entity_description(entity, st.name, entity_chain_id),
                used_full_sequence=use_full_sequence,
            )
        )

    return copies


def _collect_ligand_copies(
    model: gemmi.Model,
    entity_by_subchain: dict[str, gemmi.Entity],
    selected_author_chains: set[str] | None,
    include_ligands: bool,
    include_waters: bool,
    residue_atoms: dict[tuple[str, str, str], _ImportedAtom],
) -> list[_LigandCopy]:
    residues_by_subchain: dict[str, list[tuple[str, gemmi.Residue]]] = defaultdict(list)

    for chain in model:
        author_selected = (
            selected_author_chains is None or chain.name in selected_author_chains
        )
        if not author_selected:
            continue

        if include_ligands:
            for res in chain.get_ligands():
                subchain = res.subchain or _fallback_subchain(chain.name, res)
                residues_by_subchain[subchain].append((chain.name, res))

        if include_waters:
            for res in chain.get_waters():
                subchain = res.subchain or _fallback_subchain(chain.name, res)
                residues_by_subchain[subchain].append((chain.name, res))

    copies: list[_LigandCopy] = []
    for subchain, chain_residues in residues_by_subchain.items():
        entity = entity_by_subchain.get(subchain)
        ccd = tuple(res.name for _, res in chain_residues)
        for residue_idx, (author_chain, res) in enumerate(chain_residues, start=1):
            residue_atoms[_residue_key(author_chain, res)] = _ImportedAtom(
                chain_id=subchain,
                residue_idx=residue_idx,
                atom_name="",
                residue_name=res.name,
                kind="ligand",
            )

        copies.append(
            _LigandCopy(
                entity_key=entity.name if entity is not None else f"{subchain}:{ccd}",
                entity_chain_id=subchain,
                ccd=ccd,
                description=_entity_description(entity, "", subchain),
            )
        )

    return copies


def _build_polymer_entries(
    copies: list[_PolymerCopy],
) -> list[Polymer | ProteinSeq]:
    grouped: dict[
        tuple[str, PolymerType, str, tuple[tuple[str, int], ...]],
        list[_PolymerCopy],
    ] = defaultdict(list)
    for copy in copies:
        mods_key = tuple((mod.ccd, mod.position) for mod in copy.modifications)
        grouped[(copy.entity_key, copy.polymer_type, copy.sequence, mods_key)].append(
            copy
        )

    entries: list[Polymer | ProteinSeq] = []
    for (_, polymer_type, sequence, _), group in grouped.items():
        ids = _compact_ids([copy.entity_chain_id for copy in group])
        modifications = list(group[0].modifications) or None
        kwargs = {
            "id": ids,
            "polymer_type": polymer_type,
            "sequence": sequence,
            "modifications": modifications,
            "description": group[0].description,
        }
        if polymer_type == PolymerType.Protein:
            entries.append(ProteinSeq(**kwargs))  # type: ignore[ty:invalid-argument-type]
        else:
            entries.append(Polymer(**kwargs))  # type: ignore[ty:invalid-argument-type]
    return entries


def _build_ligand_entries(copies: list[_LigandCopy]) -> list[Ligand]:
    grouped: dict[tuple[str, tuple[str, ...]], list[_LigandCopy]] = defaultdict(list)
    for copy in copies:
        grouped[(copy.entity_key, copy.ccd)].append(copy)

    entries: list[Ligand] = []
    for (_, ccd), group in grouped.items():
        entries.append(
            Ligand(
                id=_compact_ids([copy.entity_chain_id for copy in group]),
                ccd=list(ccd),
                description=group[0].description,
            )
        )
    return entries


def _collect_connections(
    st: gemmi.Structure,
    residue_atoms: dict[tuple[str, str, str], _ImportedAtom],
    strict: bool,
    non_covalent_connections: NonCovalentConnections,
) -> tuple[list[CovalentBond], list[ContactRestraint], list[PocketRestraint]]:
    covalent_bonds: list[CovalentBond] = []
    contact_restraints: list[ContactRestraint] = []
    pocket_tokens: dict[str, list[Atom]] = defaultdict(list)

    for conn in st.connections:
        if conn.type == gemmi.ConnectionType.Disulf:
            continue
        if conn.type != gemmi.ConnectionType.Covale and (
            non_covalent_connections == "ignore"
            or conn.type
            not in {
                gemmi.ConnectionType.Hydrog,
                gemmi.ConnectionType.MetalC,
                gemmi.ConnectionType.Unknown,
            }
        ):
            continue

        atom1 = _connection_atom(conn.partner1, residue_atoms)
        atom2 = _connection_atom(conn.partner2, residue_atoms)
        if atom1 is None or atom2 is None:
            continue

        atom1 = _set_atom_name(atom1, conn.partner1.atom_name)
        atom2 = _set_atom_name(atom2, conn.partner2.atom_name)

        if conn.type == gemmi.ConnectionType.Covale:
            covalent_bonds.append(
                CovalentBond(atom1=atom1.to_atom(), atom2=atom2.to_atom())
            )
            continue

        if non_covalent_connections == "contacts":
            contact_restraints.append(
                ContactRestraint(token1=atom1.to_atom(), token2=atom2.to_atom())
            )
        elif non_covalent_connections == "pockets":
            ligand, polymer = _ligand_polymer_pair(atom1, atom2)
            if ligand is None or polymer is None:
                _handle_problem(
                    "Only ligand-polymer non-covalent connections can be imported as pocket restraints.",
                    strict,
                )
                continue
            pocket_tokens[ligand.chain_id].append(polymer.to_atom())

    pocket_restraints = [
        PocketRestraint(binder_chain=binder_chain, contact_tokens=tokens)
        for binder_chain, tokens in pocket_tokens.items()
    ]
    return covalent_bonds, contact_restraints, pocket_restraints


def _connection_atom(
    partner: gemmi.AtomAddress,
    residue_atoms: dict[tuple[str, str, str], _ImportedAtom],
) -> _ImportedAtom | None:
    key = (partner.chain_name, str(partner.res_id.seqid), partner.res_id.name)
    return residue_atoms.get(key)


def _set_atom_name(atom: _ImportedAtom, atom_name: str) -> _ImportedAtom:
    return _ImportedAtom(
        chain_id=atom.chain_id,
        residue_idx=atom.residue_idx,
        atom_name=atom_name,
        residue_name=atom.residue_name,
        kind=atom.kind,
    )


def _ligand_polymer_pair(
    atom1: _ImportedAtom, atom2: _ImportedAtom
) -> tuple[_ImportedAtom | None, _ImportedAtom | None]:
    if atom1.kind == "ligand" and atom2.kind == "polymer":
        return atom1, atom2
    if atom2.kind == "ligand" and atom1.kind == "polymer":
        return atom2, atom1
    return None, None


def _polymer_modifications(
    residues: list[gemmi.Residue],
    sequence: str,
    full_sequence_names: list[str] | None,
    strict: bool,
) -> list[SequenceModification]:
    modifications: list[SequenceModification] = []
    for observed_idx, res in enumerate(residues, start=1):
        residue_idx = _uniaf3_residue_idx(
            res,
            observed_idx,
            use_full_sequence=full_sequence_names is not None,
        )
        if residue_idx is None or residue_idx > len(sequence):
            continue

        raw_code = _raw_one_letter_code(res.name)
        if not raw_code:
            _handle_problem(
                f"Polymer residue {res.name} has no one-letter mapping.", strict
            )
            continue

        canonical_code = raw_code.upper()
        expected_code = sequence[residue_idx - 1]
        full_residue_name = (
            full_sequence_names[residue_idx - 1]
            if full_sequence_names is not None
            and residue_idx <= len(full_sequence_names)
            else None
        )
        is_modified_code = raw_code.islower()
        differs_from_full = (
            full_residue_name is not None
            and full_residue_name != res.name
            and canonical_code == expected_code
        )
        if canonical_code == expected_code and (is_modified_code or differs_from_full):
            modifications.append(
                SequenceModification(ccd=res.name, position=residue_idx)
            )
    return modifications


def _residue_names_to_sequence(
    residue_names: list[str],
    polymer_type: PolymerType,
    strict: bool,
) -> str:
    unknown = "X" if polymer_type == PolymerType.Protein else "N"
    sequence: list[str] = []
    for residue_name in residue_names:
        code = _raw_one_letter_code(residue_name)
        if not code:
            _handle_problem(
                f"Residue {residue_name} has no one-letter mapping; using {unknown}.",
                strict,
            )
            code = unknown
        sequence.append(code.upper())
    return "".join(sequence)


def _raw_one_letter_code(residue_name: str) -> str:
    return gemmi.find_tabulated_residue(residue_name).one_letter_code.strip()


def _to_uniaf3_polymer_type(
    polymer_type: gemmi.PolymerType, strict: bool
) -> PolymerType | None:
    if polymer_type == gemmi.PolymerType.PeptideL:
        return PolymerType.Protein
    if polymer_type == gemmi.PolymerType.Dna:
        return PolymerType.DNA
    if polymer_type == gemmi.PolymerType.Rna:
        return PolymerType.RNA

    _handle_problem(f"Unsupported polymer type: {polymer_type}", strict)
    return None


def _uniaf3_residue_idx(
    residue: gemmi.Residue,
    observed_idx: int,
    use_full_sequence: bool,
) -> int | None:
    if use_full_sequence:
        return residue.label_seq
    return observed_idx


def _residue_key(author_chain: str, residue: gemmi.Residue) -> tuple[str, str, str]:
    return (author_chain, str(residue.seqid), residue.name)


def _fallback_subchain(author_chain: str, residue: gemmi.Residue) -> str:
    return f"{author_chain}_{residue.seqid}_{residue.name}"


def _compact_ids(ids: list[str]) -> str | list[str]:
    if len(ids) == 1:
        return ids[0]
    return ids


def _entity_description(
    entity: gemmi.Entity | None,
    structure_name: str,
    entity_chain_id: str,
) -> str | None:
    if entity is not None and entity.name and not entity.name.isdigit():
        return entity.name
    if structure_name:
        return f"{structure_name} chain {entity_chain_id}"
    return None


def _handle_problem(message: str, strict: bool) -> None:
    if strict:
        raise ValueError(message)
    warn(message, stacklevel=2)
