"""Pydantic schemas for AlphaFold3 input JSON config.

Reference:
    https://github.com/google-deepmind/alphafold3/blob/main/docs/input.md
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field, NonNegativeInt, PositiveInt, model_validator


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


class AF3RNA(BaseModel):
    """AlphaFold3 RNA chain specification."""

    id: str | list[str]
    sequence: str
    modifications: list[AF3NucleotideModification] | None = None
    description: str | None = None
    unpairedMsa: str | None = None
    unpairedMsaPath: str | None = None


class AF3DNA(BaseModel):
    """AlphaFold3 DNA chain specification."""

    id: str | list[str]
    sequence: str
    modifications: list[AF3NucleotideModification] | None = None
    description: str | None = None


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


class AF3Config(BaseModel):
    """Top-level AlphaFold3 input JSON config."""

    name: str
    modelSeeds: list[int]  # at least one seed required
    sequences: list[AF3SequenceEntry]
    bondedAtomPairs: list[tuple[AF3BondedAtom, AF3BondedAtom]] | None = None
    userCCD: str | None = None  # mutually exclusive with userCCDPath
    userCCDPath: str | None = None
    dialect: Literal["alphafold3"] = "alphafold3"
    version: Literal[1, 2, 3, 4] = 4
