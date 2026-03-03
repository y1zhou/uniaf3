"""Tests for ChaiConfig -> UniAF3Config -> ChaiConfig adapter."""

import pytest

from uniaf3.schema import ChaiConfig, UniAF3Config
from uniaf3.schema.base import Glycan, Ligand, Polymer, PolymerType, ProteinSeq
from uniaf3.schema.chai import ChaiEntityType


@pytest.fixture(scope="module")
def chai_uni(chai_conf: ChaiConfig):
    """Convert ChaiConfig to UniAF3Config."""
    from uniaf3.adapters import from_chai

    with pytest.warns(UserWarning):
        return from_chai(chai_conf)


@pytest.fixture(scope="module")
def chai_rt(chai_uni: UniAF3Config):
    """Convert UniAF3Config back to ChaiConfig, i.e. roundtrip."""
    from uniaf3.adapters import to_chai

    return to_chai(chai_uni, strict=False)


# ruff: noqa: S101
##########################################
# ChaiConfig -> UniAF3Config
##########################################
def test_sequence_count(chai_uni: UniAF3Config, chai_conf: ChaiConfig):
    assert len(chai_uni.sequences) == len(chai_conf.entities)


def test_protein_fields(chai_uni: UniAF3Config, chai_conf: ChaiConfig):
    prot = chai_uni.sequences[0]
    src = chai_conf.entities[0]
    assert isinstance(prot, ProteinSeq)
    assert src.entity_type == ChaiEntityType.Protein
    assert prot.polymer_type == PolymerType.Protein
    assert prot.description == src.entity_name


def test_protein_with_modification(chai_uni: UniAF3Config, chai_conf: ChaiConfig):
    """Second entity has inline modification (HY3) at position 1."""
    prot = chai_uni.sequences[1]
    src = chai_conf.entities[1]
    assert isinstance(prot, ProteinSeq)
    assert src.entity_type == ChaiEntityType.Protein
    assert prot.modifications is not None
    assert len(prot.modifications) == 1
    assert prot.modifications[0].ccd == "HY3"
    assert prot.modifications[0].position == 1


def test_dna_fields(chai_uni: UniAF3Config, chai_conf: ChaiConfig):
    dna = chai_uni.sequences[2]
    src = chai_conf.entities[2]
    assert isinstance(dna, Polymer)
    assert dna.polymer_type == PolymerType.DNA
    assert dna.sequence == src.sequence
    assert dna.description == src.entity_name


def test_ligand_smiles(chai_uni: UniAF3Config, chai_conf: ChaiConfig):
    """Both CCD and SMILES ligands from Chai are stored as SMILES in UniAF3."""
    lig_ccd = chai_uni.sequences[3]
    assert isinstance(lig_ccd, Ligand)
    # NOTE: Chai ligands that look like short CCD codes are stored as SMILES
    # since we cannot reliably distinguish between CCD codes and SMILES
    assert lig_ccd.smiles is not None

    lig_smiles = chai_uni.sequences[4]
    assert isinstance(lig_smiles, Ligand)
    assert lig_smiles.smiles == chai_conf.entities[4].sequence


def test_glycan_fields(chai_uni: UniAF3Config, chai_conf: ChaiConfig):
    glycan = chai_uni.sequences[5]
    src = chai_conf.entities[5]
    assert isinstance(glycan, Glycan)
    assert glycan.chai_str == src.sequence
    assert glycan.description == src.entity_name


def test_covalent_bond(chai_uni: UniAF3Config, chai_conf: ChaiConfig):
    assert chai_uni.covalent_bonds is not None
    assert chai_conf.restraints is not None
    cov_src = [r for r in chai_conf.restraints if r.connection_type == "covalent"]
    assert len(chai_uni.covalent_bonds) == len(cov_src)
    bond = chai_uni.covalent_bonds[0]
    assert bond.atom1.atom_name == "CG"
    assert bond.atom2.atom_name == "C1"


def test_contact_restraint(chai_uni: UniAF3Config, chai_conf: ChaiConfig):
    assert chai_uni.contact_restraints is not None
    assert chai_conf.restraints is not None
    ct_src = [r for r in chai_conf.restraints if r.connection_type == "contact"]
    assert len(chai_uni.contact_restraints) == len(ct_src)
    ct = chai_uni.contact_restraints[0]
    assert ct.max_distance == 6.0


def test_pocket_restraint(chai_uni: UniAF3Config, chai_conf: ChaiConfig):
    assert chai_uni.pocket_restraints is not None
    assert chai_conf.restraints is not None
    pk_src = [r for r in chai_conf.restraints if r.connection_type == "pocket"]
    assert len(chai_uni.pocket_restraints) == 1
    pk = chai_uni.pocket_restraints[0]
    assert len(pk.contact_tokens) == len(pk_src)
    assert pk.max_distance == 8.0


def test_seeds_default(chai_uni: UniAF3Config):
    # NOTE: Chai seed=None → default [42]
    assert chai_uni.seeds == [42]


def test_no_pocket_restraints_returns_none():
    """from_chai should return None for pocket_restraints when no pockets exist."""
    from uniaf3.adapters import from_chai
    from uniaf3.schema.chai import (
        ChaiEntity,
        ChaiEntityType,
        ChaiRestraint,
        ChaiRestraintType,
    )

    conf = ChaiConfig(
        entities=[
            ChaiEntity(
                entity_type=ChaiEntityType.Protein,
                entity_name="A",
                sequence="MVLSPADKTNVK",
            ),
            ChaiEntity(
                entity_type=ChaiEntityType.Protein,
                entity_name="B",
                sequence="GKVGAHAG",
            ),
        ],
        restraints=[
            ChaiRestraint(
                restraint_id="r0",
                chainA="A",
                res_idxA="V2",
                chainB="B",
                res_idxB="K2",
                connection_type=ChaiRestraintType.Contact,
                max_distance_angstrom=8.0,
            ),
        ],
    )

    result = from_chai(conf)
    assert result.pocket_restraints is None


def test_warns_on_ligand_identity_loss(chai_conf: ChaiConfig):
    from uniaf3.adapters import from_chai

    with pytest.warns(UserWarning) as records:
        _ = from_chai(chai_conf)
    assert any(
        "ChaiEntityType.Ligand sequence is imported" in str(w.message) for w in records
    )


##########################################
# ChaiConfig -> UniAF3Config -> ChaiConfig
##########################################
def test_roundtrip_entity_count(chai_rt: ChaiConfig, chai_conf: ChaiConfig):
    assert len(chai_rt.entities) == len(chai_conf.entities)


def test_roundtrip_entity_types(chai_rt: ChaiConfig, chai_conf: ChaiConfig):
    for src, rt in zip(chai_conf.entities, chai_rt.entities, strict=True):
        assert src.entity_type == rt.entity_type


def test_roundtrip_protein_sequence(chai_rt: ChaiConfig, chai_conf: ChaiConfig):
    for src, rt in zip(chai_conf.entities, chai_rt.entities, strict=True):
        if src.entity_type == ChaiEntityType.Protein:
            # Sequences should be identical (including inline modifications)
            assert src.sequence == rt.sequence


def test_roundtrip_dna_sequence(chai_rt: ChaiConfig, chai_conf: ChaiConfig):
    for src, rt in zip(chai_conf.entities, chai_rt.entities, strict=True):
        if src.entity_type == ChaiEntityType.DNA:
            assert src.sequence == rt.sequence


def test_roundtrip_glycan_sequence(chai_rt: ChaiConfig, chai_conf: ChaiConfig):
    for src, rt in zip(chai_conf.entities, chai_rt.entities, strict=True):
        if src.entity_type == ChaiEntityType.Glycan:
            assert src.sequence == rt.sequence


def test_roundtrip_restraint_count(chai_rt: ChaiConfig, chai_conf: ChaiConfig):
    if chai_conf.restraints is None:
        assert chai_rt.restraints is None
        return
    assert chai_rt.restraints is not None
    assert len(chai_rt.restraints) == len(chai_conf.restraints)


def test_roundtrip_restraint_types(chai_rt: ChaiConfig, chai_conf: ChaiConfig):
    if chai_conf.restraints is None:
        return
    assert chai_rt.restraints is not None
    for src, rt in zip(chai_conf.restraints, chai_rt.restraints, strict=True):
        assert src.connection_type == rt.connection_type


def test_roundtrip_contact_max_distance(chai_rt: ChaiConfig, chai_conf: ChaiConfig):
    if chai_conf.restraints is None:
        return
    assert chai_rt.restraints is not None
    for src, rt in zip(chai_conf.restraints, chai_rt.restraints, strict=True):
        if src.connection_type == "contact":
            assert src.max_distance_angstrom == rt.max_distance_angstrom
