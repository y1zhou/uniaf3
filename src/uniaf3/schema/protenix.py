"""Pydantic schemas for Protenix (v1) input JSON config.

Reference:
    https://github.com/bytedance/Protenix/blob/main/docs/infer_json_format.md
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeFloat,
    PositiveInt,
    RootModel,
    computed_field,
    model_validator,
)

from uniaf3.schema.base import UniAF3BaseConfig


##########################################
# Sequence entity types
##########################################
class ProtenixProteinModification(BaseModel):
    """Post-translational modification for a protein residue."""

    ptmType: str  # CCD code (e.g. "CCD_HY3")
    ptmPosition: PositiveInt  # 1-based position


class ProtenixProteinChain(BaseModel):
    """Protenix protein chain specification."""

    sequence: str  # May contain 20 standard and X (UNK) for unknown residues
    count: PositiveInt = 1
    modifications: list[ProtenixProteinModification] | None = None
    unpairedMsaPath: str | None = None
    pairedMsaPath: str | None = None
    templatesPath: str | None = None  # .a3m and .hhr supported

    @model_validator(mode="after")
    def check_modifications_in_range(self):
        """Ensure all modification positions are within the sequence length."""
        seq_len = len(self.sequence)
        for mod in self.modifications or []:
            if not (1 <= mod.ptmPosition <= seq_len):
                raise ValueError(
                    f"Modification position {mod.ptmPosition} out of range for sequence of length {seq_len}."
                )
        return self


class ProtenixNucleotideModification(BaseModel):
    """Chemical modification for DNA/RNA bases."""

    modificationType: str  # CCD code (e.g. "CCD_6OG")
    basePosition: PositiveInt  # 1-based position


class ProtenixDNASequence(BaseModel):
    """Protenix DNA single-strand specification."""

    sequence: str
    count: PositiveInt = 1
    modifications: list[ProtenixNucleotideModification] | None = None

    @model_validator(mode="after")
    def check_modifications_in_range(self):
        """Ensure all modification positions are within the sequence length."""
        seq_len = len(self.sequence)
        for mod in self.modifications or []:
            if not (1 <= mod.basePosition <= seq_len):
                raise ValueError(
                    f"Modification position {mod.basePosition} out of range for sequence of length {seq_len}."
                )
        return self


class ProtenixRNASequence(BaseModel):
    """Protenix RNA single-strand specification."""

    sequence: str
    count: PositiveInt = 1
    modifications: list[ProtenixNucleotideModification] | None = None
    unpairedMsaPath: str | None = None

    @model_validator(mode="after")
    def check_modifications_in_range(self):
        """Ensure all modification positions are within the sequence length."""
        seq_len = len(self.sequence)
        for mod in self.modifications or []:
            if not (1 <= mod.basePosition <= seq_len):
                raise ValueError(
                    f"Modification position {mod.basePosition} out of range for sequence of length {seq_len}."
                )
        return self


class ProtenixLigand(BaseModel):
    """Protenix ligand specification.

    The ``ligand`` field can be:
      - A CCD code prefixed with ``CCD_`` (e.g. ``CCD_ATP``)
      - A SMILES string
      - A file path prefixed with ``FILE_`` (e.g. ``FILE_/path/to/atp.sdf``)
    """

    ligand: str
    count: PositiveInt = 1

    @computed_field
    @property
    def ligand_type(self) -> str:
        """Determine the ligand type based on the ligand string."""
        if self.ligand.startswith("CCD_"):
            return "CCD"
        elif self.ligand.startswith("FILE_"):
            return "FILE"
        else:
            return "SMILES"


class ProtenixIon(BaseModel):
    """Protenix ion specification.

    The ``ion`` field is a CCD code **without** a prefix (e.g. ``MG``).
    """

    ion: str
    count: PositiveInt = 1


class ProtenixSequenceEntry(BaseModel):
    """A single entry in the sequences list.

    Exactly one of the entity types must be set.
    """

    proteinChain: ProtenixProteinChain | None = None
    dnaSequence: ProtenixDNASequence | None = None
    rnaSequence: ProtenixRNASequence | None = None
    ligand: ProtenixLigand | None = None
    ion: ProtenixIon | None = None

    @model_validator(mode="after")
    def check_exactly_one(self):
        """Ensure exactly one entity type is set."""
        fields = [
            self.proteinChain,
            self.dnaSequence,
            self.rnaSequence,
            self.ligand,
            self.ion,
        ]
        if sum(f is not None for f in fields) != 1:
            raise ValueError(
                "Exactly one of proteinChain, dnaSequence, rnaSequence, "
                "ligand, or ion must be provided."
            )
        return self


##########################################
# Covalent bonds
##########################################
class ProtenixCovalentBond(BaseModel):
    """Covalent bond between two atoms from different entities.

    The entity number corresponds to the order in which the entity appears in the
    sequences list.

    The copy index must be both specified or both None. When both are empty, bonds will
    be created between all pairs of copies for the two entities, e.g. for two entities
    with two coplies, two bonds will be created between copy1=1 and copy2=1, and between
    copy1=2 and copy2=2. In this case the two entities must have the same number of
    copies.

    The position value varies based on the entity type: for polymers it is the residue
    position, for ligands composed of multiple CCD codes, it is the serial number of the
    ligand part. For ligands with a single CCD code, it should always be 1.

    The atom value should be the atom name for polymers and CCD ligands, and the
    atom index for SMILES ligands.
    """

    entity1: PositiveInt  # 1-based entity number
    copy1: PositiveInt | None = None  # optional copy index (1-based)
    position1: PositiveInt  # 1-based residue/ligand-part position
    atom1: str  # atom name or atom index
    entity2: PositiveInt
    copy2: PositiveInt | None = None
    position2: PositiveInt
    atom2: str

    @model_validator(mode="after")
    def check_copy_indices(self):
        """Ensure copy indices are both specified or both None."""
        if (self.copy1 is None) != (self.copy2 is None):
            raise ValueError("copy1 and copy2 must be both specified or both None.")
        return self


##########################################
# Constraints
##########################################
class ProtenixContactConstraint(BaseModel):
    """Contact constraint between two residues or atoms.

    Unlike covalent bonds, the copy indices must be specified, and the atom fields are
    optional. If atoms are omitted, the distance constraint is applied at the token
    granularity by default, specifically the central atom of the token.
    """

    entity1: PositiveInt  # 1-based entity number
    copy1: PositiveInt  # 1-based copy index
    position1: PositiveInt  # 1-based residue/ligand-part position
    atom1: str | None = None
    entity2: PositiveInt
    copy2: PositiveInt
    position2: PositiveInt
    atom2: str | None = None
    max_distance: NonNegativeFloat = 6.0
    min_distance: NonNegativeFloat = 0.0


class ProtenixPocketBinderChain(BaseModel):
    """Binder chain for pocket constraint."""

    model_config = ConfigDict(populate_by_name=True)

    entity: PositiveInt
    copy_idx: PositiveInt = Field(alias="copy")  # 1-based copy index


class ProtenixPocketContactResidue(BaseModel):
    """Contact residue for pocket constraint."""

    model_config = ConfigDict(populate_by_name=True)

    entity: PositiveInt
    copy_idx: PositiveInt = Field(alias="copy")  # 1-based copy index
    position: PositiveInt


class ProtenixPocketConstraint(BaseModel):
    """Pocket constraint for binding interface guidance."""

    binder_chain: ProtenixPocketBinderChain
    contact_residues: list[ProtenixPocketContactResidue]
    max_distance: NonNegativeFloat = 6.0

    @model_validator(mode="after")
    def check_contact_residues(self):
        """Ensure all contact residues are not on the binder chain."""
        for contact in self.contact_residues:
            if (
                contact.entity == self.binder_chain.entity
                and contact.copy_idx == self.binder_chain.copy_idx
            ):
                raise ValueError(
                    f"Contact residue {contact} cannot be the same as binder chain."
                )
        return self


class ProtenixConstraint(BaseModel):
    """Constraint section for a Protenix job."""

    contact: list[ProtenixContactConstraint] | None = None
    pocket: ProtenixPocketConstraint | None = None


##########################################
# Top-level config
##########################################
class ProtenixJob(BaseModel):
    """A single Protenix inference job."""

    name: str
    sequences: list[ProtenixSequenceEntry]
    covalent_bonds: list[ProtenixCovalentBond] | None = None
    constraint: ProtenixConstraint | None = None

    @model_validator(mode="after")
    def check_bonds_in_range(self):
        """Ensure all covalent bond positions are within the sequence length."""
        entity_lengths: dict[tuple[int, int], int] = {}
        for entity_id, entry in enumerate(self.sequences, start=1):
            if entry.proteinChain is not None:
                for copy_idx in range(1, entry.proteinChain.count + 1):
                    entity_lengths[(entity_id, copy_idx)] = len(
                        entry.proteinChain.sequence
                    )
            elif entry.dnaSequence is not None:
                for copy_idx in range(1, entry.dnaSequence.count + 1):
                    entity_lengths[(entity_id, copy_idx)] = len(
                        entry.dnaSequence.sequence
                    )
            elif entry.rnaSequence is not None:
                for copy_idx in range(1, entry.rnaSequence.count + 1):
                    entity_lengths[(entity_id, copy_idx)] = len(
                        entry.rnaSequence.sequence
                    )
            elif entry.ligand is not None:
                if entry.ligand.ligand_type == "CCD":
                    # SMILES and FILE ligands always have position=1
                    lig_len = len(entry.ligand.ligand.removeprefix("CCD_").split("_"))
                else:
                    lig_len = 1
                for copy_idx in range(1, entry.ligand.count + 1):
                    entity_lengths[(entity_id, copy_idx)] = lig_len
            elif entry.ion is not None:
                for copy_idx in range(1, entry.ion.count + 1):
                    entity_lengths[(entity_id, copy_idx)] = -1

        if (bonds := self.covalent_bonds) is not None:
            for bond in bonds:
                seq1_len = entity_lengths.get((bond.entity1, bond.copy1 or 1), 0)
                if not (1 <= bond.position1 <= seq1_len):
                    raise ValueError(
                        f"Bond position1 {bond.position1} out of range for entity {bond.entity1} copy {bond.copy1}"
                    )
                seq2_len = entity_lengths.get((bond.entity2, bond.copy2 or 1), 0)
                if not (1 <= bond.position2 <= seq2_len):
                    raise ValueError(
                        f"Bond position2 {bond.position2} out of range for entity {bond.entity2} copy {bond.copy2}"
                    )

        if self.constraint is None:
            return self
        if (contacts := self.constraint.contact) is not None:
            for contact in contacts:
                seq1_len = entity_lengths.get((contact.entity1, contact.copy1), 0)
                if not (1 <= contact.position1 <= seq1_len):
                    raise ValueError(
                        f"Contact constraint position1 {contact.position1} out of range for entity {contact.entity1} copy {contact.copy1}"
                    )
                seq2_len = entity_lengths.get((contact.entity2, contact.copy2), 0)
                if not (1 <= contact.position2 <= seq2_len):
                    raise ValueError(
                        f"Contact constraint position2 {contact.position2} out of range for entity {contact.entity2} copy {contact.copy2}"
                    )
        if (pocket := self.constraint.pocket) is not None:
            for contact in pocket.contact_residues:
                seq_len = entity_lengths.get((contact.entity, contact.copy_idx), 0)
                if not (1 <= contact.position <= seq_len):
                    raise ValueError(
                        f"Pocket constraint contact residue position {contact.position} out of range for entity {contact.entity} copy {contact.copy_idx}"
                    )

        return self


class ProtenixConfig(RootModel, UniAF3BaseConfig):
    """Top-level Protenix input config.

    The Protenix JSON is always a list of jobs, even for a single job.
    """

    root: list[ProtenixJob]

    def __iter__(self) -> Iterable[ProtenixJob]:  # ty:ignore[invalid-method-override]
        """Iterate over jobs in the config."""
        return iter(self.root)

    def __getitem__(self, idx) -> ProtenixJob:
        """Get job by index."""
        return self.root[idx]

    def __len__(self) -> int:
        """Get number of jobs in the config."""
        return len(self.root)

    def to_str(self, **kwargs) -> str:
        """Get JSON string representation of the config."""
        return self.to_json(**kwargs)

    def to_files(self, output_dir: str | Path, prefix: str, **kwargs):
        """Dump the config to a JSON file in the specified output directory."""
        output_path = Path(output_dir).expanduser().resolve() / f"{prefix}.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write(self.to_str(**kwargs))
