"""Tests for UniAF3Config -> BoltzConfig adapter."""

import pytest

from uniaf3.schema import BoltzConfig, UniAF3Config
from uniaf3.schema.base import Ligand, Polymer, ProteinSeq


@pytest.fixture(scope="module")
def boltz(uniaf3_conf: UniAF3Config, tmp_path_factory: pytest.TempPathFactory):
    """Convert UniAF3 to Boltz config."""
    from uniaf3.adapters import to_boltz

    return to_boltz(uniaf3_conf, msa_dir=tmp_path_factory.mktemp("msa"), strict=False)


# ruff: noqa: S101
def test_unsupported_glycan_strict(uniaf3_conf: UniAF3Config, tmp_path):
    from uniaf3.adapters import to_boltz

    with pytest.raises(ValueError, match="Glycans are not directly supported in Boltz"):
        to_boltz(uniaf3_conf, msa_dir=tmp_path, strict=True)


def test_version(uniaf3_conf: UniAF3Config, boltz: BoltzConfig):
    assert boltz.version == 1


def test_sequence_count_drops_glycan(uniaf3_conf: UniAF3Config, boltz: BoltzConfig):
    # protein + dna + 2 ligands = 4; 1 glycan dropped
    assert len(boltz.sequences) == len(uniaf3_conf.sequences) - 1 == 4


def test_protein_fields(uniaf3_conf: UniAF3Config, boltz: BoltzConfig):
    prot = boltz.sequences[0].protein
    src = uniaf3_conf.sequences[0]
    assert isinstance(src, ProteinSeq)
    assert prot is not None
    assert prot.id == src.id
    assert prot.sequence == src.sequence
    assert prot.cyclic == src.cyclic
    # MSA: "empty" because src.msa_dir is None
    assert prot.msa == "empty"


def test_protein_modifications(uniaf3_conf: UniAF3Config, boltz: BoltzConfig):
    prot = boltz.sequences[0].protein
    assert prot is not None

    src = uniaf3_conf.sequences[0]
    assert isinstance(src, ProteinSeq)

    assert prot.modifications is not None
    assert src.modifications is not None
    assert len(prot.modifications) == len(src.modifications)
    assert prot.modifications[0].ccd == src.modifications[0].ccd
    assert prot.modifications[0].position == src.modifications[0].position


def test_dna_fields(uniaf3_conf: UniAF3Config, boltz: BoltzConfig):
    dna = boltz.sequences[1].dna
    src = uniaf3_conf.sequences[1]
    assert isinstance(src, Polymer)
    assert dna is not None
    assert dna.id == src.id
    assert dna.sequence == src.sequence
    assert dna.cyclic == src.cyclic


def test_ligand_ccd(uniaf3_conf: UniAF3Config, boltz: BoltzConfig):
    lig = boltz.sequences[2].ligand
    src = uniaf3_conf.sequences[2]
    assert isinstance(src, Ligand)
    assert lig is not None
    assert lig.id == src.id
    # UniAF3 uses list of CCD codes; Boltz uses single CCD string
    assert src.ccd is not None
    assert lig.ccd == src.ccd[0]


def test_ligand_smiles(uniaf3_conf: UniAF3Config, boltz: BoltzConfig):
    lig = boltz.sequences[3].ligand
    src = uniaf3_conf.sequences[3]
    assert isinstance(src, Ligand)
    assert lig is not None
    assert lig.id == src.id
    assert lig.smiles == src.smiles


def test_bond_constraint(uniaf3_conf: UniAF3Config, boltz: BoltzConfig):
    assert boltz.constraints is not None
    bond = boltz.constraints[0].bond
    assert uniaf3_conf.restraints is not None
    src = uniaf3_conf.restraints[0]
    assert bond is not None
    assert bond.atom1 == (
        src.atom1.chain_id,
        src.atom1.residue_idx,
        src.atom1.atom_name,
    )
    assert bond.atom2 == (
        src.atom2.chain_id,
        src.atom2.residue_idx,
        src.atom2.atom_name,
    )


def test_contact_constraint(uniaf3_conf: UniAF3Config, boltz: BoltzConfig):
    assert boltz.constraints is not None
    ct = boltz.constraints[1].contact
    assert uniaf3_conf.restraints is not None
    src = uniaf3_conf.restraints[1]
    assert ct is not None
    assert ct.token1 == (src.atom1.chain_id, src.atom1.residue_idx)
    assert ct.token2 == (src.atom2.chain_id, src.atom2.residue_idx)
    assert ct.max_distance == src.max_distance
    assert ct.force == src.boltz_enable_force


def test_pocket_constraint(uniaf3_conf: UniAF3Config, boltz: BoltzConfig):
    assert boltz.constraints is not None
    pk = boltz.constraints[2].pocket
    assert uniaf3_conf.restraints is not None
    src = uniaf3_conf.restraints[2]
    assert pk is not None
    assert pk.binder == src.boltz_binder_chain
    assert pk.max_distance == src.max_distance


def test_affinity_property(uniaf3_conf: UniAF3Config, boltz: BoltzConfig):
    assert boltz.properties is not None
    assert len(boltz.properties) == 1
    assert boltz.properties[0].affinity is not None
    assert (
        boltz.properties[0].affinity.binder == uniaf3_conf.boltz_affinity_binder_chain
    )
