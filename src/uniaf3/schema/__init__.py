"""Schemas for UniAF3 input configs."""

import hashlib
from enum import Enum, StrEnum
from functools import cached_property
from pathlib import Path
from typing import Type, TypeVar

import yaml
from pydantic import (
    BaseModel,
    NonNegativeInt,
    PositiveInt,
    computed_field,
    model_validator,
)
from yaml import representer


def hash_sequence(seq: str | bytes) -> str:
    """Compute the Chai-style sequence hash.

    Source: chai_lab.data.parsing.msas.aligned_pqt.hash_sequence
    """
    return hashlib.sha256(seq.encode() if isinstance(seq, str) else seq).hexdigest()


def type_key_serializer(seq: BaseModel, type_keys: dict[type[BaseModel], str]):
    """Serialize sequences as dict with type key."""
    if type(seq) not in type_keys:
        raise TypeError(f"Unsupported sequence type: {type(seq)}")

    return {type_keys[type(seq)]: seq}


def type_key_validator(key: str, val: BaseModel, type_keys: dict[str, BaseModel]):
    """Validate data based on its type key."""
    if key not in type_keys:
        raise TypeError(f"Unsupported value type: {key}")

    return type_keys[key].model_validate(val)


T = TypeVar("T", bound="UniAF3BaseConfig")


class UniAF3BaseConfig(BaseModel):
    """Base class for all UniAF3 config models.

    Provides shared serialization, hashing, and file-loading helpers.
    """

    def to_yaml(self, **kwargs) -> str:
        """Get YAML string representation of the config.

        Args:
            **kwargs: Extra keyword arguments forwarded to ``BaseModel.model_dump``.

        """
        yaml.SafeDumper.add_multi_representer(
            Enum, representer.SafeRepresenter.represent_str
        )

        for default_arg in ("exclude_unset", "exclude_none", "exclude_computed_fields"):
            if default_arg not in kwargs:
                kwargs[default_arg] = True

        return yaml.safe_dump(
            self.model_dump(**kwargs), sort_keys=False, default_flow_style=None
        )

    def to_json(self, **kwargs) -> str:
        """Get JSON string representation of the config.

        Args:
            **kwargs: Extra keyword arguments forwarded to ``BaseModel.model_dump_json``.

        """
        for default_arg in ("exclude_unset", "exclude_none", "exclude_computed_fields"):
            if default_arg not in kwargs:
                kwargs[default_arg] = True

        return self.model_dump_json(indent=2, **kwargs)

    @computed_field
    @property
    def hash(self) -> str:
        """Get SHA256 hash of the YAML representation."""
        return hash_sequence(self.to_yaml())

    @classmethod
    def from_file(cls: Type[T], conf_file: str | Path) -> T:
        """Load config from a YAML or JSON file.

        Args:
            conf_file: Path to the config file.

        Returns:
            A validated config instance.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file extension is unsupported.

        """
        conf_path = Path(conf_file).expanduser().resolve()
        if not conf_path.exists():
            raise FileNotFoundError(f"Config file not found: {conf_path}")
        if conf_path.suffix in {".yml", ".yaml"}:
            with open(conf_path) as f:
                return cls.model_validate(yaml.safe_load(f))
        elif conf_path.suffix == ".json":
            return cls.model_validate_json(conf_path.read_bytes())
        else:
            raise ValueError(
                "Unsupported config file format. Use .yaml, .yml, or .json"
            )


class Atom(BaseModel):
    """Schema for an atom for specifying bonds."""

    chain_id: str  # corresponding to the `id` field for the entity
    residue_idx: PositiveInt  # 1-based residue index within the chain
    atom_name: str  # e.g., "CA", "N", "C", etc. Follow rdkit for ligands
    residue_name: str | None  # Chai requires this for restraints on proteins


class SequenceModification(BaseModel):
    """Schema for polymer modifications."""

    ccd: str  # CCD code of the PTM
    position: PositiveInt  # 1-based index


class StructuralTemplate(BaseModel):
    """Base schema for structural templates."""

    path: str  # path to the template structure file (mmCIF or PDB)
    # 0-based indices
    query_idx: list[NonNegativeInt] | None = None
    template_idx: list[NonNegativeInt] | None = None
    # IDs in multi-chain templates (not supported in AF3)
    query_chains: list[str] | None = None
    template_chains: list[str] | None = None
    # Boltz-specific fields
    enable_boltz_force: bool = False  # use a potential to enforce the template
    boltz_template_threshold: float | None = (
        None  # distance (Angstroms) that the prediction can deviate from the template
    )


class PolymerType(StrEnum):
    """Enum for polymer types."""

    Protein = "protein"
    DNA = "dna"
    RNA = "rna"

    def __repr__(self):
        """Return string representation when being serialized."""
        return self.value


class Polymer(BaseModel):
    """Base schema for polymers (protein, DNA, and RNA)."""

    seq_type: PolymerType
    id: str | list[str]  # A, B, ..., Z, AA, BA, CA, ..., ZA, AB, BB, CB, ..., ZB, ...
    sequence: str
    modifications: list[SequenceModification] | None = None
    description: str | None = None  # comment describing the chain
    cyclic: bool = False  # Boltz only

    @computed_field
    @cached_property
    def seq_hash(self) -> str:
        """Compute the Chai-style sequence hash."""
        return hash_sequence(self.sequence)


class ProteinSeq(Polymer):
    """Schema for individual protein sequences."""

    msa_dir: str | None = None
    templates: list[StructuralTemplate] | None = None

    @computed_field
    @property
    def unpaired_msa(self) -> str | None:
        """Get path to unpaired MSA file."""
        if self.msa_dir is None:
            return None

        return str(
            Path(self.msa_dir).expanduser().resolve()
            / "a3ms"
            / f"{self.seq_hash}.single.a3m"
        )

    @computed_field
    @property
    def paired_msa(self) -> str | None:
        """Get path to paired MSA file."""
        if self.msa_dir is None:
            return None

        return str(
            Path(self.msa_dir).expanduser().resolve()
            / "a3ms"
            / f"{self.seq_hash}.pair.a3m"
        )


class Ligand(BaseModel):
    """Schema for individual ligands.

    Ligands can be specified using two formats:

    1. a list of standard CCD codes (e.g., ["HEM", "ZN2"])
    2. a SMILES string that is not in the standard CCD library

    Note that with the SMILES option, you cannot specify covalent bonds to other
    entities as they rely on specific atom names.
    """

    id: str | list[str]  # chain ID(s)
    smiles: str | None = None  # optional SMILES string defining the ligand
    ccd: list[str] | None = None  # list of standard CCD codes
    description: str | None = None  # comment describing the ligand

    @model_validator(mode="after")
    def check_ccd_smiles_fields(self):
        """Ensure that exactly one of ccd or smiles is provided."""
        if (self.ccd is None) == (self.smiles is None):
            raise ValueError("Exactly one of ccd or smiles must be provided.")
        return self


class Glycan(BaseModel):
    """Schema for individual glycans."""

    id: str | list[str]  # chain ID(s)
    chai_str: str  # glycan string in Chai notation (modified CCD codes)
    description: str | None = None  # comment describing the glycan


class RestraintType(StrEnum):
    """Enum for restraint types."""

    Covalent = "bond"
    Pocket = "pocket"
    Contact = "contact"

    def __repr__(self):
        """Return string representation when being serialized."""
        return self.value


class Restraint(BaseModel):
    """Schema for distance restraints.

    Note that AF3 only supports bonded restraints.

    In Boltz, the `boltz_binder_chain` should be set to the ligand chain ID that binds
    to the pocket.
    In Chai, the atom that does not belong to `boltz_binder_chain` would be used for
    specifying the pocket, and for the binder chain only the chain ID is needed.
    The atom and residue index information would be ignored.
    Chai also expects the pocket chain to be in Chain B.
    """

    restraint_type: RestraintType
    atom1: Atom
    atom2: Atom
    max_distance: float  # maximum distance (Angstroms); ignored for covalent bonds
    description: str | None = None  # comment describing the restraint

    # Boltz specific fields
    enable_boltz_force: bool = False  # use a potential to enforce the restraint
    boltz_binder_chain: str | None = None  # only used for pocket restraints


class UniAF3Config(UniAF3BaseConfig):
    """Config schema for UniAF3."""

    # General settings
    sequences: list[Polymer | ProteinSeq | Ligand | Glycan]
    restraints: list[Restraint] | None = None
    seeds: list[int]

    # Inference parameters
    num_trunk_recycles: int = 3  # Boltz: recycling_steps
    num_diffn_timesteps: int = 200  # Boltz: sampling_steps
    num_diffn_samples: int = 5  # Boltz: diffusion_samples
    num_trunk_samples: int = 1  # >1 will add to seed and run multiple times in Chai-1

    # Model-specific settings
    boltz_affinity_binder_chain: str | None = None
    boltz_additional_cli_args: list[str] | None = [
        "--override",
        "--write_full_pae",
        "--write_full_pde",
        # "--use_potentials",
    ]

    @computed_field
    @property
    def yaml_str(self) -> str:
        """Get YAML string representation of the config."""
        return self.to_yaml(include={"sequences", "restraints"})

    @computed_field
    @property
    def hash(self) -> str:
        """Get hash of the config in its YAML representation.

        Note that the hash computation only considers `sequences` and `restraints`.
        """
        return hash_sequence(self.yaml_str)

    @classmethod
    def from_file(cls, conf_file: str | Path) -> "UniAF3Config":
        """Load UniAF3 config from a file."""
        conf = super().from_file(conf_file)

        for i, seq in enumerate(conf.sequences):
            if isinstance(seq, Polymer) and seq.seq_type == PolymerType.Protein:
                conf.sequences[i] = ProteinSeq(**seq.model_dump())

        return conf


def dump_config(conf: "UniAF3BaseConfig", fmt: str = "yaml", **kwargs) -> str:
    """Serialize a config model to a string.

    Args:
        conf: The config model to serialize.
        fmt: Output format (``"yaml"`` or ``"json"``).
        **kwargs: Extra keyword arguments forwarded to ``model_dump`` /
            ``model_dump_json``.

    Returns:
        The serialized config string.

    Raises:
        ValueError: If *fmt* is not ``"yaml"`` or ``"json"``.

    """
    if fmt == "yaml":
        return conf.to_yaml(**kwargs)
    elif fmt == "json":
        return conf.to_json(**kwargs)
    else:
        raise ValueError(f"Unsupported format: {fmt!r}. Use 'yaml' or 'json'.")


def write_config(conf: "UniAF3BaseConfig", out_file: str | Path, **kwargs) -> None:
    """Write a config model to a file.

    The output format is inferred from the file extension.

    Args:
        conf: The config model to write.
        out_file: Output file path (``.yaml``, ``.yml``, or ``.json``).
        **kwargs: Extra keyword arguments forwarded to :func:`dump_config`.

    Raises:
        ValueError: If the file extension is unsupported.

    """
    suffix = Path(out_file).suffix.lower()
    if suffix in {".yml", ".yaml"}:
        fmt = "yaml"
    elif suffix == ".json":
        fmt = "json"
    else:
        raise ValueError("Unsupported config file format. Use .yaml, .yml, or .json")

    text = dump_config(conf, fmt=fmt, **kwargs)
    out_path = Path(out_file).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write(text)
