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
