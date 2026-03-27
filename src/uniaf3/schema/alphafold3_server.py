"""Pydantic schemas for AlphaFold3 server input JSON config.

Reference:
    https://github.com/google-deepmind/alphafold/blob/main/server/README.md
    https://github.com/google-deepmind/alphafold3/blob/main/docs/input.md#alphafold-server-json-compatibility
"""

from collections.abc import Iterable
from datetime import date
from pathlib import Path
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
        from uniaf3.constant import KNOWN_PTM_CCD_CODES

        if self.ptmType not in KNOWN_PTM_CCD_CODES:
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

    @model_validator(mode="after")
    def check_modification_in_range(self):
        """Ensure all modifications are within the sequence length."""
        if self.modifications is not None:
            seq_len = len(self.sequence)
            for mod in self.modifications:
                if mod.ptmPosition > seq_len:
                    raise ValueError(
                        f"Modification position {mod.ptmPosition} exceeds sequence length {seq_len}."
                    )
        return self


class AF3ServerRNAModification(BaseModel):
    """AlphaFold3 Server RNA chemical modifications."""

    modificationType: str  # CCD code
    basePosition: PositiveInt  # 1-based residue position

    @model_validator(mode="after")
    def check_modification_type(self):
        """Ensure modificationType is a supported CCD code."""
        from uniaf3.constant import KNOWN_RNA_MODIFICATION_CCD_CODES

        if self.modificationType not in KNOWN_RNA_MODIFICATION_CCD_CODES:
            raise ValueError(
                f"Unsupported RNA modificationType: {self.modificationType}"
            )
        return self


class AF3ServerRNA(BaseModel):
    """AlphaFold3 Server RNA chain specification."""

    sequence: str  # only A,U,G,C allowed
    modifications: list[AF3ServerRNAModification] | None = None
    count: PositiveInt = 1

    @model_validator(mode="after")
    def check_modification_in_range(self):
        """Ensure all modifications are within the sequence length."""
        if self.modifications is not None:
            seq_len = len(self.sequence)
            for mod in self.modifications:
                if mod.basePosition > seq_len:
                    raise ValueError(
                        f"Modification position {mod.basePosition} exceeds sequence length {seq_len}."
                    )
        return self


class AF3ServerDNAModification(BaseModel):
    """AlphaFold3 Server DNA chemical modifications."""

    modificationType: str  # CCD code
    basePosition: PositiveInt  # 1-based residue position

    @model_validator(mode="after")
    def check_modification_type(self):
        """Ensure modificationType is a supported CCD code."""
        from uniaf3.constant import KNOWN_DNA_MODIFICATION_CCD_CODES

        if self.modificationType not in KNOWN_DNA_MODIFICATION_CCD_CODES:
            raise ValueError(
                f"Unsupported DNA modificationType: {self.modificationType}"
            )
        return self


class AF3ServerDNA(BaseModel):
    """AlphaFold3 Server single-stranded DNA chain specification."""

    sequence: str  # Only A,T,G,C allowed
    modifications: list[AF3ServerDNAModification] | None = None
    count: PositiveInt = 1

    @model_validator(mode="after")
    def check_modification_in_range(self):
        """Ensure all modifications are within the sequence length."""
        if self.modifications is not None:
            seq_len = len(self.sequence)
            for mod in self.modifications:
                if mod.basePosition > seq_len:
                    raise ValueError(
                        f"Modification position {mod.basePosition} exceeds sequence length {seq_len}."
                    )
        return self


class AF3ServerLigand(BaseModel):
    """AlphaFold3 Server ligand specification."""

    ligand: str  # CCD code of the ligand (e.g., "HEM", "NAD")
    count: PositiveInt = 1

    @model_validator(mode="after")
    def check_ccd_code(self):
        """Ensure the ligand is supported by the server."""
        from uniaf3.constant import KNOWN_LIGAND_CCD_CODES

        if self.ligand not in KNOWN_LIGAND_CCD_CODES:
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
        from uniaf3.constant import KNOWN_ION_CCD_CODES

        if self.ion not in KNOWN_ION_CCD_CODES:
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

    The server variant is simpler: no MSA, no templates, no userCCD.
    It also supports an explicit ion type.
    """

    name: str
    modelSeeds: list[int]  # can be empty, in which case a single seed will be generated
    sequences: list[AF3ServerSequenceEntry]
    dialect: Literal["alphafoldserver"] = "alphafoldserver"
    version: Literal[1] = 1


class AF3ServerConfig(RootModel, UniAF3BaseConfig):
    """AlphaFold3 Server input JSON config (list of jobs).

    The server variant is simpler: no MSA, no templates, no userCCD.
    It also supports an explicit ion type.
    """

    root: list[AF3ServerJob]

    def __iter__(self) -> Iterable[AF3ServerJob]:  # ty:ignore[invalid-method-override]
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

    def to_files(self, output_dir: str | Path, prefix: str, **kwargs):
        """Dump the config to a JSON file in the specified output directory."""
        output_path = Path(output_dir).expanduser().resolve() / f"{prefix}.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write(self.to_json(**kwargs))
