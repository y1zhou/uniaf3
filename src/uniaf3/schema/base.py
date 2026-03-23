"""Schemas for UniAF3 input configs."""

from enum import Enum, StrEnum
from functools import cached_property
from pathlib import Path
from typing import TypeVar

import orjson
import yaml
from platformdirs import PlatformDirs
from polars.dataframe.frame import DataFrame
from pydantic import (
    BaseModel,
    Field,
    NonNegativeFloat,
    NonNegativeInt,
    PositiveInt,
    computed_field,
    model_validator,
)
from yaml import representer

from uniaf3.msa import query_colabfold
from uniaf3.utils import hash_sequence

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
            self.model_dump(by_alias=True, **kwargs),
            sort_keys=False,
            default_flow_style=None,
            width=80,
        )

    def to_json(self, **kwargs) -> str:
        """Get JSON string representation of the config.

        Args:
            **kwargs: Extra keyword arguments forwarded to ``BaseModel.model_dump_json``.

        """
        for default_arg in ("exclude_unset", "exclude_none", "exclude_computed_fields"):
            if default_arg not in kwargs:
                kwargs[default_arg] = True

        return orjson.dumps(
            self.model_dump(by_alias=True, **kwargs), option=orjson.OPT_INDENT_2
        ).decode("utf-8")

    def to_str(self, **kwargs) -> str:
        """Get string representation of the config.

        Different models expect different input formats, so each should implement its
        own version of this method.
        """
        raise NotImplementedError("to_str method must be implemented by subclasses.")

    def to_files(self, output_dir: str | Path, prefix: str, **kwargs):
        """Dump the config to a series of files in the specified output directory.

        Different models expect different input formats, so each should implement its
        own version of this method.

        Args:
            output_dir: Directory to write the config file(s) to.
            prefix: Prefix for the output filename(s).
            **kwargs: Extra keyword arguments forwarded to the model-specific file
                writing method.

        """
        raise NotImplementedError("to_file method must be implemented by subclasses.")

    @classmethod
    def from_file(cls: type[T], conf_file: str | Path) -> T:
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
    """Schema for an atom for specifying bonds.

    A `residue_idx` of 0 is reserved for cases where the atom is part of a
    ligand, in which case the `atom_name` and `residue_name` fields are used
    to specify the atom within the ligand.
    """

    chain_id: str  # corresponding to the `id` field for the entity
    residue_idx: NonNegativeInt  # 1-based residue index within the chain
    atom_name: str | None  # e.g., "CA", "N", "C", etc. Follow rdkit for ligands
    residue_name: str | None  # Required by Chai when specifying restraints


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
    boltz_enable_force: bool = False  # use a potential to enforce the template
    boltz_template_threshold: NonNegativeFloat | None = (
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

    id: str | list[str]  # A, B, ..., Z, AA, BA, CA, ..., ZA, AB, BB, CB, ..., ZB, ...
    polymer_type: PolymerType
    sequence: str
    modifications: list[SequenceModification] | None = None
    description: str | None = None  # comment describing the chain
    boltz_cyclic: bool = False  # Boltz only

    @computed_field
    @cached_property
    def seq_hash(self) -> str:
        """Compute the Chai-style sequence hash."""
        return hash_sequence(self.sequence)

    @model_validator(mode="after")
    def check_modification_in_range(self):
        """Ensure that modification positions are within the sequence length."""
        if self.modifications is not None:
            seq_len = len(self.sequence)
            for mod in self.modifications:
                if mod.position > seq_len:
                    raise ValueError(
                        f"Modification position {mod.position} is out of range for sequence of length {seq_len}."
                    )
        return self


class ProteinSeq(Polymer):
    """Schema for individual protein sequences."""

    msa_dir: str | None = None
    templates: list[StructuralTemplate] | None = None

    @computed_field
    @property
    def unpaired_msa(self) -> str | None:
        """Get path to unpaired MSA file.

        The filename assumption comes from Chai-1 MSA search methods.
        """
        if self.msa_dir is None:
            return None

        a3m_path = (
            Path(self.msa_dir).expanduser().resolve()
            / "a3ms"
            / f"{self.seq_hash}.single.a3m"
        )
        if a3m_path.exists():
            return str(a3m_path)

        return None

    @computed_field
    @property
    def paired_msa(self) -> str | None:
        """Get path to paired MSA file.

        The filename assumption comes from Chai-1 MSA search methods.
        """
        if self.msa_dir is None:
            return None

        a3m_path = (
            Path(self.msa_dir).expanduser().resolve()
            / "a3ms"
            / f"{self.seq_hash}.pair.a3m"
        )
        if a3m_path.exists():
            return str(a3m_path)

        return None


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
    """Schema for individual glycans.

    <https://github.com/chaidiscovery/chai-lab/blob/main/examples/covalent_bonds/README.md>
    """

    id: str | list[str]  # chain ID(s)
    chai_str: str  # glycan string in Chai notation (modified CCD codes)
    description: str | None = None  # comment describing the glycan


class CovalentBond(BaseModel):
    """Schema for covalent bonds between two atoms from different entities.

    AF3: [chain, res_idx, atom_name]
    Boltz: [chain, res_idx, atom_name]
    Chai-1: chain, <{res_name}{res_idx}>@{atom_name} where the residue part can be
        omitted for ligands. The chain ID is assigned sequentially in A-Z order
        based on the order in the input FASTA file.
    Protenix: [entity, copy, position, atom_name]
    """

    atom1: Atom
    atom2: Atom
    description: str | None = None  # comment describing the restraint

    @model_validator(mode="after")
    def check_atom_names(self):
        """Ensure that atom names are provided."""
        for atom in (self.atom1, self.atom2):
            if atom.atom_name is None:
                raise ValueError("Atom name must be provided for atom1.")
            # TODO: only Chai does not need residue information
            # when the atom is part of a ligand
            # if atom.residue_idx == 0:
            #     raise ValueError("Residue index must be >0 for covalent bonds.")
        return self


class ContactRestraint(BaseModel):
    """Schema for distance restraints between two atoms from different entities.

    Note that AF3 and AF3-server do not support non-covalent restraints.
    The `min_distance` field is only used by Protenix.

    Boltz: [chain, res_idx/atom_name] res_idx for polymers, atom_name for ligands.
    Chai-1: chain, {res_name}{res_idx}; only polymers supported. The chain ID is
        assigned sequentially in A-Z order based on the order in the input FASTA file.
    Protenix: [entity, copy, position, atom_name] where atom_name is optional.
    """

    token1: Atom
    token2: Atom
    max_distance: NonNegativeFloat = 6.0  # maximum distance (Angstroms)
    min_distance: NonNegativeFloat = 0.0  # minimum distance (Angstroms)
    description: str | None = None  # comment describing the restraint

    boltz_enable_force: bool = False  # use a potential to enforce the restraint

    @model_validator(mode="after")
    def check_token_atoms(self):
        """Ensure that token atoms have valid atom names and residue indices."""
        for token in (self.token1, self.token2):
            if token.atom_name is None and token.residue_idx == 0:
                raise ValueError(
                    f"atom_name/residue_idx must be provided for contact restraint tokens: {token}."
                )
        return self

    @model_validator(mode="after")
    def check_distance_range(self):
        """Ensure that max_distance is greater than min_distance."""
        if self.max_distance <= self.min_distance:
            raise ValueError("max_distance must be greater than min_distance.")
        if not 4.0 <= self.max_distance <= 20.0:
            # This is required by Boltz but makes sense anyways
            raise ValueError("max_distance should be between 4 and 20 Angstroms.")
        return self


class PocketRestraint(BaseModel):
    """Schema for distance restraints.

    Note that AF3 and AF3-server do not support non-covalent restraints.
    The `min_distance` field is only used by Protenix.

    Boltz: [[chain, res_idx/atom_name]] res_idx for polymers, atom_name for ligands.
    Chai-1: [chain, {res_name}{res_idx}]; only polymers supported. The chain ID is
        assigned sequentially in A-Z order based on the order in the input FASTA file.
    Protenix: [[entity, copy, position]]. There can only be one pocket for Protenix,
        meaning that all contact tokens need to have the same `binder_chain`.
    """

    binder_chain: str  # ID of the chain binding to the pocket
    contact_tokens: list[Atom]
    max_distance: NonNegativeFloat = 6.0  # maximum distance (Angstroms)
    min_distance: NonNegativeFloat = 0.0  # minimum distance (Angstroms)
    description: str | None = None  # comment describing the restraint

    # Boltz specific fields
    boltz_enable_force: bool = False  # use a potential to enforce the restraint

    @model_validator(mode="after")
    def check_token_atoms(self):
        """Ensure that token atoms have valid atom names and residue indices."""
        for token in self.contact_tokens:
            if token.atom_name is None and token.residue_idx == 0:
                raise ValueError(
                    f"atom_name/residue_idx must be provided for contact restraint tokens: {token}."
                )
        return self

    @model_validator(mode="after")
    def check_distance_range(self):
        """Ensure that max_distance is greater than min_distance."""
        if self.max_distance <= self.min_distance:
            raise ValueError("max_distance must be greater than min_distance.")
        if not 4.0 <= self.max_distance <= 20.0:
            # This is required by Boltz but makes sense anyways
            raise ValueError("max_distance should be between 4 and 20 Angstroms.")
        return self

    @model_validator(mode="after")
    def check_contact_tokens_same_binder_chain(self):
        """Ensure that all contact tokens have the same binder chain ID."""
        for token in self.contact_tokens:
            if token.chain_id == self.binder_chain:
                raise ValueError(
                    f"Contact token chain ID cannot be the same as the binder chain ID {self.binder_chain}: {token}."
                )
        return self


class AuxiliaryParams(BaseModel):
    """Parameters used for inference or model-specific settings."""

    num_trunk_recycles: int = 3  # Boltz: recycling_steps
    num_diffn_timesteps: int = 200  # Boltz: sampling_steps
    num_diffn_samples: int = 5  # Boltz: diffusion_samples
    num_trunk_samples: int = 1  # >1 will add to seed and run multiple times in Chai-1

    # Model-specific settings
    name: str | None = None  # optional name for the config, used in AF3 server
    boltz_affinity_binder_chain: str | None = None
    seeds: list[int] = Field(default_factory=lambda: [42])


class UniAF3Config(UniAF3BaseConfig):
    """Config schema for UniAF3."""

    # General settings
    sequences: list[Polymer | ProteinSeq | Ligand | Glycan]
    covalent_bonds: list[CovalentBond] | None = None
    contact_restraints: list[ContactRestraint] | None = None
    pocket_restraints: list[PocketRestraint] | None = None

    # Inference parameters and model-specific settings
    aux: AuxiliaryParams = AuxiliaryParams()

    def to_str(self, **kwargs) -> str:
        """Get YAML string representation of the config."""
        return self.to_yaml(include={"sequences", "restraints"})

    def to_files(self, output_dir: str | Path, prefix: str, **kwargs):
        """Dump the config to a YAML file in the specified output directory."""
        output_path = Path(output_dir).expanduser().resolve() / f"{prefix}.yaml"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write(self.to_yaml(**kwargs))

    @computed_field
    @property
    def hash(self) -> str:
        """Get SHA256 hash of the string representation."""
        return hash_sequence(self.to_str())

    @classmethod
    def from_file(cls, conf_file: str | Path) -> "UniAF3Config":
        """Load UniAF3 config from a file."""
        conf = super().from_file(conf_file)

        for i, seq in enumerate(conf.sequences):
            if isinstance(seq, Polymer) and seq.polymer_type == PolymerType.Protein:
                conf.sequences[i] = ProteinSeq(**seq.model_dump())
                if (msa_dir := conf.sequences[i].msa_dir) is not None:
                    conf.sequences[i].msa_dir = (
                        Path(conf_file).parent / msa_dir
                    ).resolve()

        return conf

    @model_validator(mode="after")
    def check_has_sequences(self):
        """Ensure that at least one sequence is provided."""
        if not self.sequences:
            raise ValueError("At least one sequence must be provided.")
        return self

    @model_validator(mode="after")
    def check_restraints_in_range(self):
        """Ensure that restraint atom indices are within the corresponding sequence lengths."""
        restraint_atoms: list[Atom] = []
        if self.covalent_bonds is not None:
            for bond in self.covalent_bonds:
                restraint_atoms.extend([bond.atom1, bond.atom2])
        if self.contact_restraints is not None:
            for restraint in self.contact_restraints:
                restraint_atoms.extend([restraint.token1, restraint.token2])
        if self.pocket_restraints is not None:
            restraint_atoms.extend(
                token
                for restraint in self.pocket_restraints
                for token in restraint.contact_tokens
            )

        if restraint_atoms:
            # Build chain_id → sequence mapping, handling list[str] ids
            seq_dict: dict[str, Polymer | ProteinSeq | Ligand | Glycan] = {}
            for seq in self.sequences:
                ids = seq.id if isinstance(seq.id, list) else [seq.id]
                for cid in ids:
                    seq_dict[cid] = seq

            for atom in restraint_atoms:
                if atom.chain_id not in seq_dict:
                    raise ValueError(
                        f"Atom chain ID {atom.chain_id} not found in sequences."
                    )
                seq = seq_dict[atom.chain_id]
                if isinstance(seq, Ligand) or isinstance(seq, Glycan):
                    continue  # skip ligand chains since they don't have residue indices

                seq_len = len(seq.sequence)
                if atom.residue_idx < 1 or atom.residue_idx > seq_len:
                    raise ValueError(
                        f"Atom index out of range for sequence of length {seq_len}: {atom}."
                    )
                if atom.residue_name is not None:
                    res_name = seq.sequence[atom.residue_idx - 1]
                    if atom.residue_name != res_name:
                        raise ValueError(
                            f"Atom residue name {atom.residue_name} does not match sequence residue name {res_name} at index {atom.residue_idx} for chain {atom.chain_id}."
                        )
        return self

    def add_msa_for_protein_seqs(
        self,
        msa_cache_dir: str | Path | None = None,
        chains: set[str] | None = None,
        search_templates: bool = False,
        num_templates_per_seq: int = 5,
        template_cache_dir: Path | None = None,
        force: bool = False,
    ):
        """Add MSA paths to protein sequences in the config.

        Args:
            msa_cache_dir: Output directory to store MSA files. Defaults to
                $XDG_CACHE_HOME/uniaf3/colabfold_msas/ if not set.
            chains: Set of chain IDs to process. If None, process all protein chains.
            search_templates: Whether to search for templates.
            num_templates_per_seq: Number of templates to fetch per sequence.
            template_cache_dir: Directory to cache fetched templates. Defaults to
                $XDG_CACHE_HOME/uniaf3/rcsb/ if not set.
            force: Whether to overwrite existing files.

        """
        # Figure out which protein sequences to process
        if chains is None:
            chains = {
                c
                for seq in self.sequences
                if isinstance(seq, ProteinSeq)
                for c in (seq.id if isinstance(seq.id, list) else [seq.id])
            }
        protein_seqs = [
            seq.sequence
            for seq in self.sequences
            if isinstance(seq, ProteinSeq)
            and any(
                c in chains for c in (seq.id if isinstance(seq.id, list) else [seq.id])
            )
        ]

        # Generate MSAs using ColabFold API
        if msa_cache_dir is None:
            msa_cache_dir = PlatformDirs("uniaf3").user_cache_path / "colabfold_msas"

        out_path = Path(msa_cache_dir).expanduser().resolve()
        out_path.mkdir(parents=True, exist_ok=True)
        msa_data = query_colabfold(
            protein_seqs,
            msa_cache_dir=out_path,
            search_templates=search_templates,
            download_num_templates_per_seq=num_templates_per_seq,
            template_cache_dir=template_cache_dir,
            force=force,
        )

        # TODO: make dummy m8 file when search_templates is False and custom templates are provided
        # As of 2026-03, AF3, Boltz, and Chai-1 support custom local templates.
        # AF3-server searches MSA and templates online.
        # Protenix only supports automatically searched templates via the a3m/hhr files.
        template_map: dict[str, list[StructuralTemplate]] = {}
        templates_df: DataFrame | None = msa_data.templates_df
        if templates_df is not None:
            hash_to_chains: dict[str, list[str]] = {
                seq.seq_hash: (seq.id if isinstance(seq.id, list) else [seq.id])
                for seq in self.sequences
                if isinstance(seq, ProteinSeq)
            }
            for r in templates_df.iter_rows(named=True):
                _, template_chain_id = r["subject_id"].split("_")

                if r["query_id"] not in template_map:
                    template_map[r["query_id"]] = []
                template_map[r["query_id"]].append(
                    StructuralTemplate(
                        path=r["local_cif_path"],
                        query_idx=list(range(r["query_start"] - 1, r["query_end"])),
                        template_idx=list(
                            range(r["subject_start"] - 1, r["subject_end"])
                        ),
                        query_chains=hash_to_chains[r["query_id"]],
                        template_chains=[template_chain_id],
                    )
                )

        # Update config with MSA paths and templates
        for seq in self.sequences:
            if not isinstance(seq, ProteinSeq):
                continue
            if isinstance(seq.id, str) and seq.id not in chains:
                continue
            elif isinstance(seq.id, list) and not any(c in chains for c in seq.id):
                continue

            seq.msa_dir = str(out_path)

            custom_templates = seq.templates or []
            if seq.seq_hash in template_map:
                seq.templates = custom_templates + template_map[seq.seq_hash]

            # TODO: copy template files to expected locations?
