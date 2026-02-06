"""Pydantic schemas for Boltz input YAML config.

Reference:
    https://github.com/jwohlwend/boltz/blob/main/docs/prediction.md
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, PositiveInt, model_validator


class BoltzModification(BaseModel):
    """Modification for a polymer residue (protein, DNA, or RNA)."""

    position: PositiveInt  # 1-based index
    ccd: str  # CCD code of the modified residue


class BoltzProtein(BaseModel):
    """Boltz protein chain specification."""

    id: str | list[str]
    sequence: str
    msa: str | None = None  # path to .a3m file, or "empty" for single-sequence
    modifications: list[BoltzModification] | None = None
    cyclic: bool = False


class BoltzDNA(BaseModel):
    """Boltz DNA chain specification."""

    id: str | list[str]
    sequence: str
    modifications: list[BoltzModification] | None = None
    cyclic: bool = False


class BoltzRNA(BaseModel):
    """Boltz RNA chain specification."""

    id: str | list[str]
    sequence: str
    modifications: list[BoltzModification] | None = None
    cyclic: bool = False


class BoltzLigand(BaseModel):
    """Boltz ligand specification.

    Each ligand uses either a CCD code or a SMILES string, not both.
    """

    id: str | list[str]
    smiles: str | None = None
    ccd: str | None = None

    @model_validator(mode="after")
    def check_ccd_smiles_fields(self):
        """Ensure exactly one of ccd or smiles is provided."""
        if (self.ccd is None) == (self.smiles is None):
            raise ValueError("Exactly one of ccd or smiles must be provided.")
        return self


class BoltzSequenceEntry(BaseModel):
    """A single entry in the sequences list.

    Exactly one of protein, dna, rna, or ligand must be set.
    """

    protein: BoltzProtein | None = None
    dna: BoltzDNA | None = None
    rna: BoltzRNA | None = None
    ligand: BoltzLigand | None = None

    @model_validator(mode="after")
    def check_exactly_one(self):
        """Ensure exactly one entity type is set."""
        fields = [self.protein, self.dna, self.rna, self.ligand]
        if sum(f is not None for f in fields) != 1:
            raise ValueError(
                "Exactly one of protein, dna, rna, or ligand must be provided."
            )
        return self


# Constraints


class BoltzBondConstraint(BaseModel):
    """Covalent bond constraint between two atoms."""

    atom1: tuple[str, int, str]  # (chain_id, 1-based residue index, atom name)
    atom2: tuple[str, int, str]


class BoltzPocketConstraint(BaseModel):
    """Pocket constraint specifying binding residues."""

    binder: str  # chain ID of the binder
    contacts: list[tuple[str, int | str]]  # (chain_id, residue index or atom name)
    max_distance: float = 6.0
    force: bool = False


class BoltzContactConstraint(BaseModel):
    """Contact constraint between two residues/atoms."""

    token1: tuple[str, int | str]  # (chain_id, residue index or atom name)
    token2: tuple[str, int | str]
    max_distance: float = 6.0
    force: bool = False


class BoltzConstraintEntry(BaseModel):
    """A single constraint entry.

    Exactly one of bond, pocket, or contact must be set.
    """

    bond: BoltzBondConstraint | None = None
    pocket: BoltzPocketConstraint | None = None
    contact: BoltzContactConstraint | None = None

    @model_validator(mode="after")
    def check_exactly_one(self):
        """Ensure exactly one constraint type is set."""
        fields = [self.bond, self.pocket, self.contact]
        if sum(f is not None for f in fields) != 1:
            raise ValueError(
                "Exactly one of bond, pocket, or contact must be provided."
            )
        return self


# Templates


class BoltzTemplate(BaseModel):
    """Structural template specification."""

    cif: str | None = None  # path to CIF file (mutually exclusive with pdb)
    pdb: str | None = None  # path to PDB file
    chain_id: str | list[str] | None = None  # which chains to template
    template_id: str | list[str] | None = None  # explicit template chain mapping
    force: bool = False  # use potential to enforce template
    threshold: float | None = None  # distance threshold for force (Angstroms)

    @model_validator(mode="after")
    def check_cif_pdb_fields(self):
        """Ensure exactly one of cif or pdb is provided."""
        if (self.cif is None) == (self.pdb is None):
            raise ValueError("Exactly one of cif or pdb must be provided.")
        return self


# Properties


class BoltzAffinityProperty(BaseModel):
    """Affinity prediction property."""

    binder: str  # chain ID of the ligand


class BoltzPropertyEntry(BaseModel):
    """A single property entry."""

    affinity: BoltzAffinityProperty | None = None


# Top-level config


class BoltzConfig(BaseModel):
    """Top-level Boltz input YAML config."""

    version: Literal[1] = 1
    sequences: list[BoltzSequenceEntry]
    constraints: list[BoltzConstraintEntry] | None = None
    templates: list[BoltzTemplate] | None = None
    properties: list[BoltzPropertyEntry] | None = None
