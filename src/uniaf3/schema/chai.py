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

from pydantic import BaseModel, model_validator

from uniaf3.schema import UniAF3BaseConfig


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
      ``>entity_type|name=entity_name``

    The sequence field holds the amino acid / nucleotide sequence or a SMILES
    string (for ligands) / Chai glycan notation (for glycans).
    """

    entity_type: ChaiEntityType
    entity_name: str  # unique name used as chain label
    sequence: str  # AA/nucleotide sequence, SMILES, or glycan notation


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
    res_idxA: str  # can be empty for pocket restraints
    chainB: str
    res_idxB: str
    connection_type: ChaiRestraintType
    confidence: float = 1.0
    max_distance_angstrom: float
    min_distance_angstrom: float = 0.0
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

    def entities_to_fasta(self) -> str:
        """Convert the entities list to a multi-FASTA string."""
        lines = []
        for e in self.entities:
            header = f">{e.entity_type.value}|{e.entity_name}"
            lines.append(header)
            lines.append(e.sequence)
        return "\n".join(lines)

    def to_str(self, **kwargs) -> str:
        """Get string representation with FASTA entities and CSV restraints."""
        parts = [self.entities_to_fasta()]
        csv = self.restraints_to_csv()
        if csv is not None:
            parts.append(csv)
        return "\n---\n".join(parts)

    def restraints_to_csv(self) -> str | None:
        """Convert the restraints list to a CSV string."""
        if self.restraints is None:
            return None
        lines = [
            "restraint_id,chainA,res_idxA,chainB,res_idxB,connection_type,"
            "confidence,max_distance_angstrom,min_distance_angstrom,comment"
        ]
        for r in self.restraints:
            line = (
                f"{r.restraint_id},{r.chainA},{r.res_idxA},{r.chainB},"
                f"{r.res_idxB},{r.connection_type.value},{r.confidence},"
                f"{r.max_distance_angstrom},{r.min_distance_angstrom},"
                f"'{r.comment or ''}'"
            )
            lines.append(line)
        return "\n".join(lines)
