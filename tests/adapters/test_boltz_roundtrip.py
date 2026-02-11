"""Tests for BoltzConfig -> UniAF3Config -> BoltzConfig adapter."""

import pytest

from uniaf3.schema import BoltzConfig, UniAF3Config
from uniaf3.schema.base import Ligand, PolymerType, ProteinSeq, RestraintType


@pytest.fixture(scope="module")
def boltz_uni(boltz_conf: BoltzConfig, tmp_path_factory: pytest.TempPathFactory):
    """Convert BoltzConfig to UniAF3Config."""
    from uniaf3.adapters import from_boltz

    return from_boltz(boltz_conf)


@pytest.fixture(scope="module")
def boltz_rt(boltz_uni: UniAF3Config, tmp_path_factory: pytest.TempPathFactory):
    """Convert UniAF3Config back to BoltzConfig, i.e. roundtrip."""
    from uniaf3.adapters import to_boltz

    return to_boltz(boltz_uni, msa_dir=tmp_path_factory.mktemp("msa"), strict=False)


# ruff: noqa: S101
##########################################
# BoltzConfig -> UniAF3Config
##########################################
def test_sequence_count(boltz_uni: UniAF3Config, boltz_conf: BoltzConfig):
    assert len(boltz_uni.sequences) == len(boltz_conf.sequences)


def test_protein_fields(boltz_uni: UniAF3Config, boltz_conf: BoltzConfig):
    prot = boltz_uni.sequences[0]
    src = boltz_conf.sequences[0].protein
    assert isinstance(prot, ProteinSeq)
    assert src is not None
    assert prot.id == src.id
    assert prot.sequence == src.sequence
    assert prot.seq_type == PolymerType.Protein
    assert prot.cyclic == src.cyclic


def test_protein_modifications(boltz_uni: UniAF3Config, boltz_conf: BoltzConfig):
    prot = boltz_uni.sequences[0]
    src = boltz_conf.sequences[0].protein
    assert isinstance(prot, ProteinSeq)
    assert src is not None
    assert prot.modifications is not None
    assert src.modifications is not None
    assert len(prot.modifications) == len(src.modifications)
    assert prot.modifications[0].ccd == src.modifications[0].ccd
    assert prot.modifications[0].position == src.modifications[0].position


def test_protein_msa_not_mapped(boltz_uni: UniAF3Config, boltz_conf: BoltzConfig):
    """Boltz MSA path cannot be mapped to UniAF3 msa_dir."""
    prot = boltz_uni.sequences[0]
    assert isinstance(prot, ProteinSeq)
    # NOTE: Boltz provides a single MSA path; UniAF3 uses msa_dir
    assert prot.msa_dir is None


def test_ligand_ccd(boltz_uni: UniAF3Config, boltz_conf: BoltzConfig):
    lig = boltz_uni.sequences[1]
    src = boltz_conf.sequences[1].ligand
    assert isinstance(lig, Ligand)
    assert src is not None
    # Boltz single CCD → UniAF3 list
    assert lig.ccd == [src.ccd]
    assert lig.id == src.id


def test_ligand_smiles(boltz_uni: UniAF3Config, boltz_conf: BoltzConfig):
    lig = boltz_uni.sequences[2]
    src = boltz_conf.sequences[2].ligand
    assert isinstance(lig, Ligand)
    assert src is not None
    assert lig.smiles == src.smiles
    assert lig.id == src.id


def test_bond_restraint(boltz_uni: UniAF3Config, boltz_conf: BoltzConfig):
    assert boltz_uni.restraints is not None
    assert boltz_conf.constraints is not None
    bond = boltz_uni.restraints[0]
    src = boltz_conf.constraints[0].bond
    assert src is not None
    assert bond.restraint_type == RestraintType.Covalent
    assert bond.atom1.chain_id == src.atom1[0]
    assert bond.atom1.residue_idx == src.atom1[1]
    assert bond.atom1.atom_name == src.atom1[2]
    assert bond.atom2.chain_id == src.atom2[0]
    assert bond.atom2.residue_idx == src.atom2[1]
    assert bond.atom2.atom_name == src.atom2[2]


def test_contact_restraint(boltz_uni: UniAF3Config, boltz_conf: BoltzConfig):
    assert boltz_uni.restraints is not None
    assert boltz_conf.constraints is not None
    ct = boltz_uni.restraints[1]
    src = boltz_conf.constraints[1].contact
    assert src is not None
    assert ct.restraint_type == RestraintType.Contact
    assert ct.atom1.chain_id == src.token1[0]
    assert ct.atom1.residue_idx == int(src.token1[1])
    assert ct.max_distance == src.max_distance
    assert ct.boltz_enable_force == src.force


def test_pocket_restraint(boltz_uni: UniAF3Config, boltz_conf: BoltzConfig):
    assert boltz_uni.restraints is not None
    assert boltz_conf.constraints is not None
    pk = boltz_uni.restraints[2]
    src = boltz_conf.constraints[2].pocket
    assert src is not None
    assert pk.restraint_type == RestraintType.Pocket
    assert pk.boltz_binder_chain == src.binder
    assert pk.max_distance == src.max_distance
    # atom2 is the binder chain with a placeholder residue_idx
    assert pk.atom2.chain_id == src.binder


def test_seeds_default(boltz_uni: UniAF3Config, boltz_conf: BoltzConfig):
    # NOTE: Boltz config does not include seeds
    assert boltz_uni.seeds == [42]


##########################################
# BoltzConfig -> UniAF3Config -> BoltzConfig
##########################################
def test_roundtrip_sequences(boltz_rt: BoltzConfig, boltz_conf: BoltzConfig):
    # Both have glycan chain dropped
    assert len(boltz_rt.sequences) == len(boltz_conf.sequences)


def test_roundtrip_protein_sequence(boltz_rt: BoltzConfig, boltz_conf: BoltzConfig):
    src = boltz_conf.sequences[0].protein
    assert src is not None
    prot = boltz_rt.sequences[0].protein
    assert prot is not None
    assert prot.sequence == src.sequence
    assert prot.id == src.id

    assert prot.modifications is not None
    assert src.modifications is not None

    for mod, src_mod in zip(prot.modifications, src.modifications, strict=True):
        assert mod.ccd == src_mod.ccd
        assert mod.position == src_mod.position


def test_roundtrip_protein_templates(boltz_rt: BoltzConfig, boltz_conf: BoltzConfig):
    src = boltz_conf.templates
    assert src is not None
    prot = boltz_rt.templates
    assert prot is not None

    assert len(prot) == len(src) == 1
    assert prot[0] == src[0]


def test_roundtrip_ligand(boltz_rt: BoltzConfig, boltz_conf: BoltzConfig):
    src = boltz_conf.sequences[1].ligand
    ligand = boltz_rt.sequences[1].ligand

    assert src is not None
    assert ligand is not None

    assert ligand.smiles == src.smiles
    assert ligand.id == src.id
    assert ligand.ccd == src.ccd


def test_roundtrip_restraints(boltz_rt: BoltzConfig, boltz_conf: BoltzConfig):
    assert boltz_conf.constraints is not None
    assert boltz_rt.constraints is not None

    for rt_cst, boltz_cst in zip(
        boltz_rt.constraints, boltz_conf.constraints, strict=True
    ):
        if boltz_cst.bond is not None:
            assert rt_cst.bond.atom1 == boltz_cst.bond.atom1
            assert rt_cst.bond.atom2 == boltz_cst.bond.atom2
        elif boltz_cst.contact is not None:
            assert rt_cst.contact.token1 == boltz_cst.contact.token1
            assert rt_cst.contact.token2 == boltz_cst.contact.token2
            assert rt_cst.contact.max_distance == boltz_cst.contact.max_distance
            assert rt_cst.contact.force == boltz_cst.contact.force
        elif boltz_cst.pocket is not None:
            assert rt_cst.pocket.binder == boltz_cst.pocket.binder
            assert rt_cst.pocket.contacts == boltz_cst.pocket.contacts
            assert rt_cst.pocket.max_distance == boltz_cst.pocket.max_distance
            assert rt_cst.pocket.force == boltz_cst.pocket.force
