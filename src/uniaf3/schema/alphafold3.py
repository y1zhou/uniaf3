"""Pydantic schemas for AlphaFold3 input JSON config.

Reference:
    https://github.com/google-deepmind/alphafold3/blob/main/docs/input.md
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, Field, NonNegativeInt, PositiveInt, model_validator

from uniaf3.schema.base import UniAF3BaseConfig
from uniaf3.utils import normalize_out_dir


class AF3ProteinModification(BaseModel):
    """Post-translational modification for a protein residue."""

    ptmType: str  # CCD code
    ptmPosition: PositiveInt  # 1-based residue position


class AF3NucleotideModification(BaseModel):
    """Chemical modification for DNA/RNA bases."""

    modificationType: str  # CCD code
    basePosition: PositiveInt  # 1-based base position


class AF3Template(BaseModel):
    """Structural template for protein chains."""

    mmcif: str | None = None  # inline mmCIF string (mutually exclusive with mmcifPath)
    mmcifPath: str | None = None  # path to mmCIF file
    queryIndices: list[NonNegativeInt]  # 0-based query residue indices
    templateIndices: list[NonNegativeInt]  # 0-based template residue indices

    @model_validator(mode="after")
    def check_mmcif_fields(self):
        """Ensure exactly one of mmcif or mmcifPath is provided."""
        if (self.mmcif is None) == (self.mmcifPath is None):
            raise ValueError("Exactly one of mmcif or mmcifPath must be provided.")
        return self

    @model_validator(mode="after")
    def check_indices_length(self):
        """Ensure queryIndices and templateIndices have the same length."""
        if len(self.queryIndices) != len(self.templateIndices):
            raise ValueError(
                "queryIndices and templateIndices must have the same length."
            )
        return self


class AF3Protein(BaseModel):
    """AlphaFold3 protein chain specification."""

    id: str | list[str]
    sequence: str
    modifications: list[AF3ProteinModification] | None = None
    description: str | None = None
    unpairedMsa: str | None = None  # inline A3M (mutually exclusive with Path)
    unpairedMsaPath: str | None = None
    pairedMsa: str | None = None  # inline A3M (mutually exclusive with Path)
    pairedMsaPath: str | None = None
    templates: list[AF3Template] | None = None

    @model_validator(mode="after")
    def check_modifications_in_range(self):
        """Ensure all modification positions are valid for the sequence length."""
        if self.modifications:
            seq_length = len(self.sequence)
            for mod in self.modifications:
                if not (1 <= mod.ptmPosition <= seq_length):
                    raise ValueError(
                        f"Modification position {mod.ptmPosition} out of range for sequence length {seq_length}."
                    )
        return self


class AF3RNA(BaseModel):
    """AlphaFold3 RNA chain specification."""

    id: str | list[str]
    sequence: str
    modifications: list[AF3NucleotideModification] | None = None
    description: str | None = None
    unpairedMsa: str | None = None
    unpairedMsaPath: str | None = None

    @model_validator(mode="after")
    def check_modifications_in_range(self):
        """Ensure all modification positions are valid for the sequence length."""
        if self.modifications:
            seq_length = len(self.sequence)
            for mod in self.modifications:
                if not (1 <= mod.basePosition <= seq_length):
                    raise ValueError(
                        f"Modification position {mod.basePosition} out of range for sequence length {seq_length}."
                    )
        return self


class AF3DNA(BaseModel):
    """AlphaFold3 DNA chain specification."""

    id: str | list[str]
    sequence: str
    modifications: list[AF3NucleotideModification] | None = None
    description: str | None = None

    @model_validator(mode="after")
    def check_modifications_in_range(self):
        """Ensure all modification positions are valid for the sequence length."""
        if self.modifications:
            seq_length = len(self.sequence)
            for mod in self.modifications:
                if not (1 <= mod.basePosition <= seq_length):
                    raise ValueError(
                        f"Modification position {mod.basePosition} out of range for sequence length {seq_length}."
                    )
        return self


class AF3Ligand(BaseModel):
    """AlphaFold3 ligand specification.

    Each ligand uses either CCD codes or a SMILES string, not both.
    """

    id: str | list[str]
    ccdCodes: list[str] | None = None
    smiles: str | None = None
    description: str | None = None

    @model_validator(mode="after")
    def check_ccd_smiles_fields(self):
        """Ensure exactly one of ccdCodes or smiles is provided."""
        if (self.ccdCodes is None) == (self.smiles is None):
            raise ValueError("Exactly one of ccdCodes or smiles must be provided.")
        return self


# A bonded atom is [entity_id, residue_id (1-based), atom_name]
AF3BondedAtom = Annotated[
    tuple[str, PositiveInt, str],
    Field(description="(entity_id, 1-based residue index, atom name)"),
]


class AF3SequenceEntry(BaseModel):
    """A single entry in the sequences list.

    Exactly one of protein, rna, dna, or ligand must be set.
    """

    protein: AF3Protein | None = None
    rna: AF3RNA | None = None
    dna: AF3DNA | None = None
    ligand: AF3Ligand | None = None

    @model_validator(mode="after")
    def check_exactly_one(self):
        """Ensure exactly one entity type is set."""
        fields = [self.protein, self.rna, self.dna, self.ligand]
        if sum(f is not None for f in fields) != 1:
            raise ValueError(
                "Exactly one of protein, rna, dna, or ligand must be provided."
            )
        return self


class AF3Config(UniAF3BaseConfig):
    """Top-level AlphaFold3 input JSON config."""

    name: str
    modelSeeds: list[int]  # at least one seed required
    sequences: list[AF3SequenceEntry]
    bondedAtomPairs: list[tuple[AF3BondedAtom, AF3BondedAtom]] | None = None
    userCCD: str | None = None  # string in CCD mmCIF format
    userCCDPath: str | None = None  # mutually exclusive with userCCD
    dialect: Literal["alphafold3"] = "alphafold3"
    version: Literal[1, 2, 3, 4] = 4

    @classmethod
    def from_file(cls, conf_file: str | Path) -> AF3Config:
        """Load UniAF3 config from a file."""
        conf = super().from_file(conf_file)

        for seq in conf.sequences:
            if seq.protein is not None:
                if seq.protein.pairedMsaPath is not None:
                    seq.protein.pairedMsaPath = str(
                        (Path(conf_file).parent / seq.protein.pairedMsaPath).resolve()
                    )
                if seq.protein.unpairedMsaPath is not None:
                    seq.protein.unpairedMsaPath = str(
                        (Path(conf_file).parent / seq.protein.unpairedMsaPath).resolve()
                    )

        return conf

    def to_str(self, **kwargs) -> str:
        """Get JSON string representation of the config."""
        return self.to_json(**kwargs)

    def to_files(self, output_dir: str | Path, prefix: str, **kwargs):
        """Dump the config to a JSON file in the specified output directory."""
        output_dir = normalize_out_dir(output_dir)
        output_path = output_dir / f"{prefix}.json"
        with open(output_path, "w") as f:
            f.write(self.to_json(**kwargs))

    @model_validator(mode="after")
    def check_bonds_in_range(self):
        """Ensure all bonded atom positions are valid for the sequence lengths."""
        if self.bondedAtomPairs is None:
            return self

        # Build a mapping from entity_id to sequence length
        entity_lengths = {}
        for entry in self.sequences:
            if entry.protein:
                for eid in (
                    entry.protein.id
                    if isinstance(entry.protein.id, list)
                    else [entry.protein.id]
                ):
                    entity_lengths[eid] = len(entry.protein.sequence)
            elif entry.rna:
                for eid in (
                    entry.rna.id if isinstance(entry.rna.id, list) else [entry.rna.id]
                ):
                    entity_lengths[eid] = len(entry.rna.sequence)
            elif entry.dna:
                for eid in (
                    entry.dna.id if isinstance(entry.dna.id, list) else [entry.dna.id]
                ):
                    entity_lengths[eid] = len(entry.dna.sequence)
            elif entry.ligand:
                if entry.ligand.smiles is not None:
                    ligand_len = -1
                else:
                    ligand_len = (
                        len(entry.ligand.ccdCodes)
                        if entry.ligand.ccdCodes is not None
                        else -1  # should never happen because SMILES | CCD must exist
                    )
                for eid in (
                    entry.ligand.id
                    if isinstance(entry.ligand.id, list)
                    else [entry.ligand.id]
                ):
                    entity_lengths[eid] = ligand_len

        # Check each bonded atom pair
        for atom1, atom2 in self.bondedAtomPairs:
            for atom in (atom1, atom2):
                entity_id, residue_id, _ = atom
                if entity_id not in entity_lengths:
                    raise ValueError(
                        f"Entity ID {entity_id} not found for bonded atom."
                    )
                if not (1 <= residue_id <= entity_lengths[entity_id]):
                    if entity_lengths[entity_id] == -1:
                        raise ValueError(
                            f"Cannot specify covalent bond {atom} to SMILES ligand chain {entity_id}"
                        )
                    raise ValueError(
                        f"Residue ID {residue_id} out of range for entity {entity_id} with length {entity_lengths[entity_id]}."
                    )
        return self
