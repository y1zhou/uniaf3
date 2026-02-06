"""Pydantic schemas for Protenix (v1) input JSON config.

Reference:
    https://github.com/bytedance/Protenix/blob/main/docs/infer_json_format.md
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, PositiveInt, model_validator

# Sequence entity types


class ProtenixProteinModification(BaseModel):
    """Post-translational modification for a protein residue."""

    ptmType: str  # CCD code (e.g. "CCD_HY3")
    ptmPosition: PositiveInt  # 1-based position


class ProtenixProteinChain(BaseModel):
    """Protenix protein chain specification."""

    sequence: str
    count: PositiveInt = 1
    modifications: list[ProtenixProteinModification] | None = None
    pairedMsaPath: str | None = None
    unpairedMsaPath: str | None = None
    templatesPath: str | None = None


class ProtenixNucleotideModification(BaseModel):
    """Chemical modification for DNA/RNA bases."""

    modificationType: str  # CCD code (e.g. "CCD_6OG")
    basePosition: PositiveInt  # 1-based position


class ProtenixDNASequence(BaseModel):
    """Protenix DNA single-strand specification."""

    sequence: str
    count: PositiveInt = 1
    modifications: list[ProtenixNucleotideModification] | None = None


class ProtenixRNASequence(BaseModel):
    """Protenix RNA single-strand specification."""

    sequence: str
    count: PositiveInt = 1
    modifications: list[ProtenixNucleotideModification] | None = None
    unpairedMsaPath: str | None = None


class ProtenixLigand(BaseModel):
    """Protenix ligand specification.

    The ``ligand`` field can be:
      - A CCD code prefixed with ``CCD_`` (e.g. ``CCD_ATP``)
      - A SMILES string
      - A file path prefixed with ``FILE_`` (e.g. ``FILE_/path/to/atp.sdf``)
    """

    ligand: str
    count: PositiveInt = 1


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


# Covalent bonds


class ProtenixCovalentBond(BaseModel):
    """Covalent bond between two atoms from different entities."""

    entity1: str  # 1-based entity number (as string)
    position1: str  # 1-based residue/ligand-part position (as string)
    atom1: str  # atom name or atom index
    entity2: str
    position2: str
    atom2: str
    copy1: PositiveInt | None = None  # optional copy index (1-based)
    copy2: PositiveInt | None = None


# Constraints


class ProtenixContactConstraint(BaseModel):
    """Contact constraint between two residues or atoms."""

    entity1: int
    copy1: int
    position1: int
    atom1: str | None = None
    entity2: int
    copy2: int
    position2: int
    atom2: str | None = None
    max_distance: float
    min_distance: float = 0.0


class ProtenixPocketBinderChain(BaseModel):
    """Binder chain for pocket constraint."""

    model_config = ConfigDict(populate_by_name=True)

    entity: int
    copy_idx: int = Field(alias="copy")  # 1-based copy index


class ProtenixPocketContactResidue(BaseModel):
    """Contact residue for pocket constraint."""

    model_config = ConfigDict(populate_by_name=True)

    entity: int
    copy_idx: int = Field(alias="copy")  # 1-based copy index
    position: int


class ProtenixPocketConstraint(BaseModel):
    """Pocket constraint for binding interface guidance."""

    binder_chain: ProtenixPocketBinderChain
    contact_residues: list[ProtenixPocketContactResidue]
    max_distance: float


class ProtenixConstraint(BaseModel):
    """Constraint section for a Protenix job."""

    contact: list[ProtenixContactConstraint] | None = None
    pocket: ProtenixPocketConstraint | None = None


# Top-level config


class ProtenixJob(BaseModel):
    """A single Protenix inference job."""

    name: str
    sequences: list[ProtenixSequenceEntry]
    covalent_bonds: list[ProtenixCovalentBond] | None = None
    constraint: ProtenixConstraint | None = None


class ProtenixConfig(BaseModel):
    """Top-level Protenix input config.

    The Protenix JSON is always a list of jobs, even for a single job.
    """

    jobs: list[ProtenixJob]
