"""Pydantic schemas for Chai-1 input config.

Chai-1 uses a FASTA file for sequences and an optional CSV for restraints.
This module defines Pydantic schemas to represent the structured config that
feeds into ``chai_lab.chai1.run_inference``.

Reference:
    https://github.com/chaidiscovery/chai-lab
    https://github.com/chaidiscovery/chai-lab/blob/main/chai_lab/chai1.py
    https://github.com/chaidiscovery/chai-lab/tree/main/examples/msas/README.md
    https://github.com/chaidiscovery/chai-lab/tree/main/examples/restraints/README.md
    https://github.com/chaidiscovery/chai-lab/tree/main/examples/covalent_bonds/README.md

"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path

import polars as pl
from pydantic import BaseModel, NonNegativeFloat, model_validator

from uniaf3.constant import int_to_letters
from uniaf3.schema.base import UniAF3BaseConfig


class ChaiEntityType(StrEnum):
    """Supported entity types in Chai-1 FASTA input."""

    Protein = "protein"
    DNA = "dna"
    RNA = "rna"
    Ligand = "ligand"
    Glycan = "glycan"


class ChaiEntity(BaseModel):
    """A single entity in the Chai-1 input FASTA.

    Chai-1 expects a multi-entity FASTA with structured headers:
      ``>entity_type|entity_name``

    The sequence field holds the amino acid / nucleotide sequence or a SMILES
    string (for ligands) / Chai glycan notation (for glycans).
    """

    entity_type: ChaiEntityType
    entity_name: str  # unique name used as chain description
    sequence: str  # AA/nucleotide sequence, SMILES, or glycan notation

    # TODO: UniAF3 MSAs are not compatible with the FASTA format


class ChaiRestraintType(StrEnum):
    """Connection types supported in the Chai-1 restraints CSV."""

    Contact = "contact"
    Pocket = "pocket"
    Covalent = "covalent"


class ChaiRestraint(BaseModel):
    """A single row in the Chai-1 restraints CSV.

    CSV columns:
      restraint_id, chainA, res_idxA, chainB, res_idxB,
      max_distance_angstrom, min_distance_angstrom, connection_type,
      confidence, comment

    Residue index format: ``<residue_name><position>[@atom_name]``
    (e.g. ``A219``, ``D45@CB``).
    For pocket restraints the binder residue index can be empty.

    Columns confidence, comment, and min_distance_angstrom are currently not used by
    the model, but are included in the schema for completeness and potential future use.
    """

    restraint_id: str
    chainA: str
    res_idxA: str | None  # can be empty for pocket restraints
    chainB: str
    res_idxB: str
    connection_type: ChaiRestraintType
    confidence: float = 1.0
    max_distance_angstrom: NonNegativeFloat
    min_distance_angstrom: NonNegativeFloat = 0.0
    comment: str | None = None


class ChaiConfig(UniAF3BaseConfig):
    """Structured representation of a Chai-1 inference job.

    This schema mirrors the keyword arguments of
    ``chai_lab.chai1.run_inference``.
    """

    entities: list[ChaiEntity]
    restraints: list[ChaiRestraint] | None = None

    # Inference parameters
    num_trunk_recycles: int = 3
    num_diffn_timesteps: int = 200
    num_diffn_samples: int = 5
    num_trunk_samples: int = 1
    seed: int | None = None

    # Optional input paths
    msa_directory: str | None = None
    constraint_path: str | None = None  # path to restraints CSV
    template_hits_path: str | None = None  # path to templates .m8 file

    # Feature flags
    use_esm_embeddings: bool = True
    use_msa_server: bool = False
    msa_server_url: str = "https://api.colabfold.com"
    use_templates_server: bool = False

    @model_validator(mode="after")
    def check_entity_names_unique(self):
        """Chai-1 requires each entity to have a unique name."""
        names = [e.entity_name for e in self.entities]
        if len(names) != len(set(names)):
            raise ValueError("All entity names must be unique.")
        return self

    @model_validator(mode="after")
    def check_restraints(self):
        """Ensure restraints refer to valid entities."""
        from uniaf3.vendor.chai1_fasta import constituents_of_modified_fasta

        if self.restraints is None:
            return self

        entity_map: dict[str, ChaiEntity] = {
            int_to_letters(i): e for i, e in enumerate(self.entities, start=1)
        }
        for r in self.restraints:
            seq_a = entity_map[r.chainA].sequence
            if entity_map[r.chainA].entity_type in {
                ChaiEntityType.Protein,
                ChaiEntityType.DNA,
                ChaiEntityType.RNA,
            }:
                seq_a = constituents_of_modified_fasta(seq_a)
            _ensure_valid_restraint(
                r.connection_type, entity_map[r.chainA].entity_type, r.res_idxA, seq_a
            )
            seq_b = entity_map[r.chainB].sequence
            if entity_map[r.chainB].entity_type in {
                ChaiEntityType.Protein,
                ChaiEntityType.DNA,
                ChaiEntityType.RNA,
            }:
                seq_b = constituents_of_modified_fasta(seq_b)
            _ensure_valid_restraint(
                r.connection_type, entity_map[r.chainB].entity_type, r.res_idxB, seq_b
            )
        return self

    def entities_to_fasta(self) -> str:
        """Convert the entities list to a multi-FASTA string."""
        lines = []
        for e in self.entities:
            header = f">{e.entity_type.value}|{e.entity_name}"
            lines.append(header)
            lines.append(e.sequence)
        return "\n".join(lines)

    def restraints_to_df(self) -> pl.DataFrame | None:
        """Convert the restraints list to a CSV string."""
        if self.restraints is None:
            return None

        return pl.DataFrame(self.restraints)

    @classmethod
    def from_yaml(cls, conf_file: str | Path) -> ChaiConfig:
        """Load a ChaiConfig from a YAML file."""
        return super().from_file(conf_file)

    @classmethod
    def from_file(
        cls, fasta_file: str | Path, restraints_file: str | Path | None = None, **kwargs
    ) -> ChaiConfig:
        """Load a ChaiConfig from a FASTA file and optional restraints CSV."""
        from uniaf3.vendor.chai1_fasta import read_fasta

        entries = read_fasta(fasta_file)
        entities: list[ChaiEntity] = []
        for entry in entries:
            entity_type, entity_name = entry.header.split("|", maxsplit=1)
            entities.append(
                ChaiEntity(
                    entity_type=ChaiEntityType(entity_type),
                    entity_name=entity_name,
                    sequence=entry.sequence,
                )
            )
        restraints: list[ChaiRestraint] | None = None
        if restraints_file is not None:
            restraints_df = pl.read_csv(restraints_file)
            restraints = [
                ChaiRestraint.model_validate(r)
                for r in restraints_df.iter_rows(named=True)
            ]

        return cls(entities=entities, restraints=restraints, **kwargs)

    def to_files(self, output_dir: str | Path, prefix: str, **kwargs):
        """Dump the config to a YAML file in the specified output directory."""
        output_path = Path(output_dir).expanduser().resolve() / f"{prefix}.yaml"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write(self.to_yaml(**kwargs))

        # Also dump FASTA and restraint CSV for direct use by Chai-1
        fasta_path = output_path.with_suffix(".fasta")
        with open(fasta_path, "w") as f:
            f.write(self.entities_to_fasta())
        if self.restraints is not None:
            restraints_path = output_path.with_suffix(".csv")
            self.restraints_to_df().write_csv(restraints_path)


def _ensure_valid_restraint(
    connection: ChaiRestraintType,
    entity_type: ChaiEntityType,
    res_idx: str,
    seq: list[str],
):
    """Validate that covalent bonds refer to valid entities and atoms."""
    polymer_type = {ChaiEntityType.Protein, ChaiEntityType.DNA, ChaiEntityType.RNA}
    if connection == ChaiRestraintType.Covalent:
        # N436@N for residues, @C1 for ligands and glycans
        try:
            idx, atom = res_idx.split("@")
        except ValueError as e:
            raise ValueError(
                f"Invalid residue index format for covalent restraint: {res_idx}"
            ) from e
        if entity_type in polymer_type:
            try:
                res_name, res_pos = idx[0], int(idx[1:])
            except Exception as e:
                raise ValueError(f"Failed to parse residue index: {res_idx}") from e

            if res_name != seq[res_pos - 1]:
                raise ValueError(
                    f"Residue name in index does not match sequence: {res_idx} vs {seq}"
                )

        # TODO: check atom names follow rdkit
        if not atom:
            raise ValueError(
                f"Atom name must be specified for covalent restraints: {res_idx}"
            )
    elif connection == ChaiRestraintType.Contact:
        # R84 for residues; ligands and glycans not supported
        if entity_type not in polymer_type:
            raise ValueError(
                f"Contact restraints currently only supported for protein/DNA/RNA entities, got {entity_type}"
            )
        try:
            res_name, res_pos = res_idx[0], int(res_idx[1:])
        except Exception as e:
            raise ValueError(f"Failed to parse residue index: {res_idx}") from e

        if res_name != seq[res_pos - 1]:
            raise ValueError(
                f"Residue name in index does not match sequence: {res_idx} vs {seq}"
            )
    elif connection == ChaiRestraintType.Pocket:
        # R84 for residues; empty for non-binder chain.
        if res_idx:
            try:
                res_name, res_pos = res_idx[0], int(res_idx[1:])
            except Exception as e:
                raise ValueError(f"Failed to parse residue index: {res_idx}") from e

            if res_name != seq[res_pos - 1]:
                raise ValueError(
                    f"Residue name in index does not match sequence: {res_idx} vs {seq}"
                )
    else:
        raise ValueError(f"Unsupported restraint connection type: {connection}")
