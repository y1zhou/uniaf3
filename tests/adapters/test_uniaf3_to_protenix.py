"""Tests for UniAF3Config -> ProtenixConfig adapter."""

import pytest

from uniaf3.schema import ProtenixConfig, UniAF3Config
from uniaf3.schema.base import Ligand, Polymer, ProteinSeq


@pytest.fixture(scope="module")
def ptx(uniaf3_conf: UniAF3Config):
    """Convert UniAF3 to Protenix config."""
    from uniaf3.adapters import to_protenix

    return to_protenix([uniaf3_conf], name="test")


# ruff: noqa: S101
def test_job_count(uniaf3_conf: UniAF3Config, ptx: ProtenixConfig):
    assert len(ptx) == 1
    assert ptx[0].name == "test"


def test_protein_fields(uniaf3_conf: UniAF3Config, ptx: ProtenixConfig):
    prot = ptx[0].sequences[0].proteinChain
    src = uniaf3_conf.sequences[0]
    assert isinstance(src, ProteinSeq)
    assert prot is not None
    assert prot.sequence == src.sequence
    # id ["A", "B"] → count=2
    assert isinstance(src.id, list)
    assert len(src.id) == prot.count == 2

    assert prot.unpairedMsaPath == src.unpaired_msa
    assert prot.pairedMsaPath == src.paired_msa

    # TODO: mapping of templates

    assert prot.modifications is not None
    assert src.modifications is not None
    for mod_ptx, mod_uni in zip(prot.modifications, src.modifications, strict=True):
        # CCD_ prefix stripped
        assert mod_ptx.ptmType == f"CCD_{mod_uni.ccd}"
        assert mod_ptx.ptmPosition == mod_uni.position


def test_dna_fields(uniaf3_conf: UniAF3Config, ptx: ProtenixConfig):
    dna = ptx[0].sequences[1].dnaSequence
    src = uniaf3_conf.sequences[1]
    assert isinstance(src, Polymer)
    assert dna is not None
    assert dna.sequence == src.sequence


def test_ccd_ligand_fields(uniaf3_conf: UniAF3Config, ptx: ProtenixConfig):
    lig = ptx[0].sequences[2].ligand
    src = uniaf3_conf.sequences[2]
    assert isinstance(src, Ligand)
    assert lig is not None
    assert lig.ligand == f"CCD_{src.ccd[0]}"  # TODO: support multiple CCD codes


# TODO: test SMILES ligands


def test_covalent_bond(uniaf3_conf: UniAF3Config, ptx: ProtenixConfig):
    job = ptx[0]
    assert job.covalent_bonds is not None
    assert len(job.covalent_bonds) == 1
    assert uniaf3_conf.covalent_bonds is not None
    src = uniaf3_conf.covalent_bonds[0]
    bond = job.covalent_bonds[0]
    assert bond.atom1 == src.atom1.atom_name
    assert bond.atom2 == src.atom2.atom_name
    assert bond.position1 == src.atom1.residue_idx
    assert bond.position2 == src.atom2.residue_idx

    assert bond.entity1 == 1
    assert bond.entity2 == 3
    assert bond.copy1 == 2
    assert bond.copy2 == 1


def test_contact_constraint(uniaf3_conf: UniAF3Config, ptx: ProtenixConfig):
    job = ptx[0]
    assert job.constraint is not None
    assert job.constraint.contact is not None
    assert len(job.constraint.contact) == 1
    assert uniaf3_conf.contact_restraints is not None
    ct = job.constraint.contact[0]
    src = uniaf3_conf.contact_restraints[0]

    assert ct.atom1 == src.token1.atom_name
    assert ct.atom2 == src.token2.atom_name
    assert ct.position1 == src.token1.residue_idx
    assert ct.position2 == src.token2.residue_idx
    assert ct.max_distance == src.max_distance

    assert ct.entity1 == 1
    assert ct.entity2 == 1
    assert ct.copy1 == 1
    assert ct.copy2 == 2


def test_pocket_constraint(uniaf3_conf: UniAF3Config, ptx: ProtenixConfig):
    job = ptx[0]
    assert job.constraint is not None
    assert job.constraint.pocket is not None
    assert uniaf3_conf.pocket_restraints is not None
    src = uniaf3_conf.pocket_restraints[0]
    assert job.constraint.pocket.max_distance == src.max_distance

    pocket = job.constraint.pocket
    assert pocket.max_distance == src.max_distance
    assert pocket.contact_residues[0].entity == 1
    assert pocket.contact_residues[0].copy_idx == 1
    assert pocket.contact_residues[0].position == src.contact_tokens[0].residue_idx

    assert pocket.binder_chain.entity == 3
    assert pocket.binder_chain.copy_idx == 1
