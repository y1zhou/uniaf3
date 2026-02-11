"""Pydantic schemas for AlphaFold3 server input JSON config.

Reference:
    https://github.com/google-deepmind/alphafold/blob/main/server/README.md
    https://github.com/google-deepmind/alphafold3/blob/main/docs/input.md#alphafold-server-json-compatibility
"""

from collections.abc import Iterable
from datetime import date
from typing import Literal

from pydantic import BaseModel, PositiveInt, RootModel, model_validator

from uniaf3.schema.base import UniAF3BaseConfig


class AF3ServerGlycan(BaseModel):
    """AlphaFold3 Server protein glycosylation specification.

    Supported glycan residues include:
    alpha/beta-D-glucose, alpha/beta-D-mannose, alpha-L-fucose,
    beta-D-galactose, and N-acetyl-beta-D-glucosamine.

    See the CCD library for their corresponding codes:
    <https://www.ebi.ac.uk/pdbe-srv/pdbechem/>.
    """

    residues: str  # 3-letter codes of the glycan residues, e.g. NAG(NAG)(BMA)
    position: PositiveInt  # position of the modified amino acid


class AF3ServerProteinModification(BaseModel):
    """AlphaFold3 Server PTM for a protein residue."""

    ptmType: str  # CCD code
    ptmPosition: PositiveInt  # 1-based residue position

    @model_validator(mode="after")
    def check_ptm_type(self):
        """Ensure ptmType is a supported CCD code."""
        supported_ptms = {
            "CCD_SEP",
            "CCD_TPO",
            "CCD_PTR",
            "CCD_NEP",
            "CCD_HIP",
            "CCD_ALY",
            "CCD_MLY",
            "CCD_M3L",
            "CCD_MLZ",
            "CCD_2MR",
            "CCD_AGM",
            "CCD_MCS",
            "CCD_HYP",
            "CCD_HY3",
            "CCD_LYZ",
            "CCD_AHB",
            "CCD_P1L",
            "CCD_SNN",
            "CCD_SNC",
            "CCD_TRF",
            "CCD_KCR",
            "CCD_CIR",
            "CCD_YHA",
        }
        if self.ptmType not in supported_ptms:
            raise ValueError(f"Unsupported ptmType: {self.ptmType}")
        return self


class AF3ServerProtein(BaseModel):
    """AlphaFold3 Server protein chain specification."""

    sequence: str  # only 20 standard amino acids supported
    count: PositiveInt = 1  # number of copies of this protein chain
    glycans: list[AF3ServerGlycan] | None = None
    modifications: list[AF3ServerProteinModification] | None = None
    useStructureTemplate: bool = True  # whether the model should use PDB templates
    maxTemplateDate: date | None = (
        None  # upper date limit for considering PDB templates
    )

    @model_validator(mode="after")
    def check_template_cutoff(self):
        """Ensure maxTemplateDate is not in the future."""
        dt = self.maxTemplateDate
        if dt is not None:
            if dt < date(1976, 1, 1):
                raise ValueError("maxTemplateDate cannot be before 1976-01-01.")
            if dt > date(2025, 2, 3):
                raise ValueError("maxTemplateDate cannot be later than 2025-02-03.")
        return self


class AF3ServerRNAModification(BaseModel):
    """AlphaFold3 Server RNA chemical modifications."""

    modificationType: str  # CCD code
    basePosition: PositiveInt  # 1-based residue position

    @model_validator(mode="after")
    def check_modification_type(self):
        """Ensure modificationType is a supported CCD code."""
        supported_mods = {
            "CCD_PSU",
            "CCD_5MC",
            "CCD_OMC",
            "CCD_4OC",
            "CCD_5MU",
            "CCD_OMU",
            "CCD_UR3",
            "CCD_A2M",
            "CCD_MA6",
            "CCD_6MZ",
            "CCD_2MG",
            "CCD_OMG",
            "CCD_7MG",
            "CCD_RSQ",
        }
        if self.modificationType not in supported_mods:
            raise ValueError(
                f"Unsupported DNA modificationType: {self.modificationType}"
            )
        return self


class AF3ServerRNA(BaseModel):
    """AlphaFold3 Server RNA chain specification."""

    sequence: str  # only A,U,G,C allowed
    modifications: list[AF3ServerRNAModification] | None = None
    count: PositiveInt = 1


class AF3ServerDNAModification(BaseModel):
    """AlphaFold3 Server DNA chemical modifications."""

    modificationType: str  # CCD code
    basePosition: PositiveInt  # 1-based residue position

    @model_validator(mode="after")
    def check_modification_type(self):
        """Ensure modificationType is a supported CCD code."""
        supported_mods = {
            "CCD_5CM",
            "CCD_C34",
            "CCD_5HC",
            "CCD_6OG",
            "CCD_6MA",
            "CCD_1CC",
            "CCD_8OG",
            "CCD_5FC",
            "CCD_3DR",
        }
        if self.modificationType not in supported_mods:
            raise ValueError(
                f"Unsupported DNA modificationType: {self.modificationType}"
            )
        return self


class AF3ServerDNA(BaseModel):
    """AlphaFold3 Server single-stranded DNA chain specification."""

    sequence: str  # Only A,T,G,C allowed
    modifications: list[AF3ServerDNAModification] | None = None
    count: PositiveInt = 1


class AF3ServerLigand(BaseModel):
    """AlphaFold3 Server ligand specification."""

    ligand: str  # CCD code of the ligand (e.g., "HEM", "NAD")
    count: PositiveInt = 1

    @model_validator(mode="after")
    def check_ccd_code(self):
        """Ensure the ligand is supported by the server."""
        supported_ligands = {
            "CCD_ADP",
            "CCD_ATP",
            "CCD_AMP",
            "CCD_GTP",
            "CCD_GDP",
            "CCD_FAD",
            "CCD_NAD",
            "CCD_NAP",
            "CCD_NDP",
            "CCD_HEM",
            "CCD_HEC",
            "CCD_PLM",
            "CCD_OLA",
            "CCD_MYR",
            "CCD_CIT",
            "CCD_CLA",
            "CCD_CHL",
            "CCD_BCL",
            "CCD_BCB",
        }
        if self.ligand not in supported_ligands:
            raise ValueError(f"Unsupported ligand CCD code: {self.ligand}")
        return self


class AF3ServerIon(BaseModel):
    """AlphaFold3 Server ion specification.

    The server has a separate ion type with a single CCD code.
    """

    ion: str  # CCD code of the ion (e.g., "MG", "ZN")
    count: PositiveInt = 1

    @model_validator(mode="after")
    def check_ccd_code(self):
        """Ensure the ion is supported by the server."""
        supported_ions = {"MG", "ZN", "CL", "CA", "NA", "MN", "K", "FE", "CU", "CO"}
        if self.ion not in supported_ions:
            raise ValueError(f"Unsupported ion CCD code: {self.ion}")
        return self


class AF3ServerSequenceEntry(BaseModel):
    """Server sequence entry. Exactly one type must be set."""

    proteinChain: AF3ServerProtein | None = None
    dnaSequence: AF3ServerDNA | None = None
    rnaSequence: AF3ServerRNA | None = None
    ligand: AF3ServerLigand | None = None
    ion: AF3ServerIon | None = None

    @model_validator(mode="after")
    def check_exactly_one(self):
        """Ensure exactly one entity type is provided."""
        fields = [
            self.proteinChain,
            self.dnaSequence,
            self.rnaSequence,
            self.ligand,
            self.ion,
        ]
        if sum(f is not None for f in fields) != 1:
            raise ValueError("Exactly one entity type must be provided.")
        return self


class AF3ServerJob(BaseModel):
    """AlphaFold3 Server input JSON config.

    The server variant is simpler: no seeds, no MSA, no templates, no userCCD.
    It also supports an explicit ion type.
    """

    name: str
    modelSeeds: list[int]  # can be empty, in which case a single seed will be generated
    sequences: list[AF3ServerSequenceEntry]
    dialect: Literal["alphafoldserver"] = "alphafoldserver"
    version: Literal[1] = 1


class AF3ServerConfig(RootModel, UniAF3BaseConfig):
    """AlphaFold3 Server input JSON config.

    The server variant is simpler: no seeds, no MSA, no templates, no userCCD.
    It also supports an explicit ion type.
    """

    root: list[AF3ServerJob]

    def __iter__(self) -> Iterable[AF3ServerJob]:
        """Iterate over jobs in the config."""
        return iter(self.root)

    def __getitem__(self, idx) -> AF3ServerJob:
        """Get job by index."""
        return self.root[idx]

    def __len__(self) -> int:
        """Get number of jobs in the config."""
        return len(self.root)

    def to_str(self, **kwargs) -> str:
        """Get JSON string representation."""
        return self.to_json(**kwargs)
