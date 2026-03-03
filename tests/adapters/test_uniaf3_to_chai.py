"""Tests for UniAF3Config -> ChaiConfig adapter."""

import pytest

from uniaf3.schema import ChaiConfig, UniAF3Config
from uniaf3.schema.base import Glycan, Ligand, Polymer, ProteinSeq
from uniaf3.schema.chai import ChaiEntityType


@pytest.fixture(scope="module")
def chai(uniaf3_conf: UniAF3Config):
    """Convert UniAF3 to Chai config."""
    from uniaf3.adapters import to_chai

    with pytest.warns(UserWarning):
        return to_chai(uniaf3_conf, strict=False)


# ruff: noqa: S101
def test_entity_count(uniaf3_conf: UniAF3Config, chai: ChaiConfig):
    # protein id=["A","B"] expands to 2 entities; dna, 2 ligands, 1 glycan = 6
    assert len(chai.entities) == 6


def test_warns_on_ccd_to_smiles_conversion(uniaf3_conf: UniAF3Config):
    from uniaf3.adapters import to_chai

    with pytest.warns(UserWarning) as records:
        _ = to_chai(uniaf3_conf, strict=False)
    assert any("Ligand.ccd 'ATP' is converted" in str(w.message) for w in records)


def test_protein_entity_type(uniaf3_conf: UniAF3Config, chai: ChaiConfig):
    prot = chai.entities[0]
    assert prot.entity_type == ChaiEntityType.Protein
    assert prot.entity_name == "A"


def test_protein_modification_inlined(uniaf3_conf: UniAF3Config, chai: ChaiConfig):
    """UniAF3 modifications should be inlined as (CCD) tokens in Chai sequence."""
    prot = chai.entities[0]
    src = uniaf3_conf.sequences[0]
    assert isinstance(src, ProteinSeq)
    assert src.modifications is not None
    # First modification at position 1: HY3
    assert prot.sequence.startswith("(HY3)")


def test_dna_entity(uniaf3_conf: UniAF3Config, chai: ChaiConfig):
    dna = chai.entities[2]
    src = uniaf3_conf.sequences[1]
    assert isinstance(src, Polymer)
    assert dna.entity_type == ChaiEntityType.DNA
    assert dna.sequence == src.sequence
    assert dna.entity_name == "C"


def test_ligand_ccd_to_smiles(uniaf3_conf: UniAF3Config, chai: ChaiConfig):
    """CCD ligands should be converted to SMILES for Chai."""
    lig = chai.entities[3]
    src = uniaf3_conf.sequences[2]
    assert isinstance(src, Ligand)
    assert lig.entity_type == ChaiEntityType.Ligand
    # CCD ATP should be resolved to a SMILES string
    assert lig.sequence != "ATP"
    assert len(lig.sequence) > 0


def test_ligand_smiles_preserved(uniaf3_conf: UniAF3Config, chai: ChaiConfig):
    lig = chai.entities[4]
    src = uniaf3_conf.sequences[3]
    assert isinstance(src, Ligand)
    assert lig.entity_type == ChaiEntityType.Ligand
    assert lig.sequence == src.smiles


def test_glycan_entity(uniaf3_conf: UniAF3Config, chai: ChaiConfig):
    glycan = chai.entities[5]
    src = uniaf3_conf.sequences[4]
    assert isinstance(src, Glycan)
    assert glycan.entity_type == ChaiEntityType.Glycan
    assert glycan.sequence == src.chai_str


def test_covalent_restraint(uniaf3_conf: UniAF3Config, chai: ChaiConfig):
    assert chai.restraints is not None
    assert uniaf3_conf.covalent_bonds is not None
    cov = [r for r in chai.restraints if r.connection_type == "covalent"]
    assert len(cov) == len(uniaf3_conf.covalent_bonds) == 1
    r = cov[0]
    # Chain IDs are remapped to Chai A-Z ordering
    assert r.chainA == "B"  # B is 2nd entity
    assert r.chainB == "D"  # D is 4th entity (CCD ligand)
    assert r.res_idxA is not None and r.res_idxB is not None
    assert "@" in r.res_idxA  # atom name for polymer
    assert "@" in r.res_idxB  # atom name for ligand

    src = uniaf3_conf.covalent_bonds[0]
    assert src.atom2.atom_name is not None
    assert src.atom2.atom_name in r.res_idxB


def test_contact_restraint(uniaf3_conf: UniAF3Config, chai: ChaiConfig):
    assert chai.restraints is not None
    assert uniaf3_conf.contact_restraints is not None
    ct = [r for r in chai.restraints if r.connection_type == "contact"]
    assert len(ct) == len(uniaf3_conf.contact_restraints)
    r = ct[0]
    src = uniaf3_conf.contact_restraints[0]
    assert r.max_distance_angstrom == src.max_distance


def test_pocket_restraints(uniaf3_conf: UniAF3Config, chai: ChaiConfig):
    assert chai.restraints is not None
    assert uniaf3_conf.pocket_restraints is not None
    pk = [r for r in chai.restraints if r.connection_type == "pocket"]
    # Each contact token generates a separate pocket restraint row
    src = uniaf3_conf.pocket_restraints[0]
    assert len(pk) == len(src.contact_tokens)


def test_inference_params(uniaf3_conf: UniAF3Config, chai: ChaiConfig):
    assert chai.num_trunk_recycles == uniaf3_conf.aux.num_trunk_recycles
    assert chai.num_diffn_timesteps == uniaf3_conf.aux.num_diffn_timesteps
    assert chai.num_diffn_samples == uniaf3_conf.aux.num_diffn_samples
    assert chai.num_trunk_samples == uniaf3_conf.aux.num_trunk_samples


def test_seed(uniaf3_conf: UniAF3Config, chai: ChaiConfig):
    # Only first seed is taken
    assert chai.seed == uniaf3_conf.aux.seeds[0]
