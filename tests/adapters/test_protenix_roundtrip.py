"""Tests for ProtenixConfig -> UniAF3Config -> ProtenixConfig adapter."""

import pytest

from uniaf3.schema import ProtenixConfig, UniAF3Config
from uniaf3.schema.base import Ligand, Polymer, PolymerType, ProteinSeq, RestraintType


@pytest.fixture(scope="module")
def ptx_uni(protenix_confs):
    """Convert ProtenixConfig to UniAF3Config."""
    from uniaf3.adapters import from_protenix

    return from_protenix(protenix_confs)


@pytest.fixture(scope="module")
def ptx_rt(ptx_uni: list[UniAF3Config]):
    """Convert UniAF3Config → ProtenixConfig → UniAF3Config, i.e. roundtrip."""
    from uniaf3.adapters import to_protenix

    return to_protenix(ptx_uni)


# ruff: noqa: S101
##########################################
# ProtenixConfig -> UniAF3Config
##########################################
def test_sequence_count(ptx_uni: list[UniAF3Config], protenix_confs: ProtenixConfig):
    assert len(ptx_uni) == len(protenix_confs) == 1
    assert len(ptx_uni[0].sequences) == len(protenix_confs[0].sequences) == 6


def test_protein_fields(ptx_uni: list[UniAF3Config], protenix_confs: ProtenixConfig):
    prot = ptx_uni[0].sequences[0]
    src = protenix_confs[0].sequences[0].proteinChain
    assert isinstance(prot, ProteinSeq)
    assert src is not None
    assert prot.sequence == src.sequence
    # count=2 → list of 2 chain IDs
    assert isinstance(prot.id, list)
    assert len(prot.id) == src.count

    assert prot.modifications is not None
    assert src.modifications is not None
    for mod_uni, mod_src in zip(prot.modifications, src.modifications, strict=True):
        # CCD_ prefix stripped
        assert f"CCD_{mod_uni.ccd}" == mod_src.ptmType
        assert mod_uni.position == mod_src.ptmPosition


def test_dna_fields(ptx_uni: list[UniAF3Config], protenix_confs: ProtenixConfig):
    dna = ptx_uni[0].sequences[1]
    src = protenix_confs[0].sequences[1].dnaSequence
    assert isinstance(dna, Polymer)
    assert src is not None
    assert dna.seq_type == PolymerType.DNA
    assert dna.sequence == src.sequence

    assert dna.modifications is not None
    assert src.modifications is not None
    for mod_uni, mod_src in zip(dna.modifications, src.modifications, strict=True):
        assert f"CCD_{mod_uni.ccd}" == mod_src.modificationType
        assert mod_uni.position == mod_src.basePosition


def test_rna_fields(ptx_uni: list[UniAF3Config], protenix_confs: ProtenixConfig):
    rna = ptx_uni[0].sequences[2]
    src = protenix_confs[0].sequences[2].rnaSequence
    assert isinstance(rna, Polymer)
    assert src is not None
    assert rna.seq_type == PolymerType.RNA
    assert rna.sequence == src.sequence


def test_ligand_ccd(ptx_uni: list[UniAF3Config], protenix_confs: ProtenixConfig):
    lig = ptx_uni[0].sequences[3]
    src = protenix_confs[0].sequences[3].ligand
    assert isinstance(lig, Ligand)
    assert src is not None
    assert lig.ccd == [src.ligand.removeprefix("CCD_")]

    # TODO: multiple CCD codes per ligand


def test_ion_as_ligand(ptx_uni: list[UniAF3Config], protenix_confs: ProtenixConfig):
    ion = ptx_uni[0].sequences[4]
    src = protenix_confs[0].sequences[4].ion
    assert isinstance(ion, Ligand)
    assert src is not None
    assert ion.ccd == [src.ion]
    assert isinstance(ion.id, list)
    assert len(ion.id) == src.count


def test_smiles_ligand(ptx_uni: list[UniAF3Config], protenix_confs: ProtenixConfig):
    lig = ptx_uni[0].sequences[5]
    src = protenix_confs[0].sequences[5].ligand
    assert isinstance(lig, Ligand)
    assert src is not None
    # SMILES string not prefixed with CCD_
    assert lig.smiles == src.ligand  # TODO: convert SMILES to CCD code if possible


def test_covalent_bond_restraint(
    ptx_uni: list[UniAF3Config], protenix_confs: ProtenixConfig
):
    assert ptx_uni[0].restraints is not None
    bonds = [
        r for r in ptx_uni[0].restraints if r.restraint_type == RestraintType.Covalent
    ]
    assert len(bonds) == 1
    src = protenix_confs[0].covalent_bonds[0]
    bond = bonds[0]
    assert bond.atom1.atom_name == src.atom1
    assert bond.atom2.atom_name == src.atom2

    # TODO: fix entity and copy mapping


def test_contact_restraint(ptx_uni: list[UniAF3Config], protenix_confs: ProtenixConfig):
    assert ptx_uni[0].restraints is not None
    contacts = [
        r for r in ptx_uni[0].restraints if r.restraint_type == RestraintType.Contact
    ]
    assert len(contacts) == 1
    src = protenix_confs[0].constraint.contact[0]
    ct = contacts[0]
    assert ct.max_distance == src.max_distance


def test_pocket_restraint(ptx_uni: list[UniAF3Config], protenix_confs: ProtenixConfig):
    assert ptx_uni[0].restraints is not None
    pockets = [
        r for r in ptx_uni[0].restraints if r.restraint_type == RestraintType.Pocket
    ]
    assert len(pockets) == 1
    src = protenix_confs[0].constraint.pocket
    pk = pockets[0]
    assert pk.max_distance == src.max_distance
    assert pk.binder_chain is not None


def test_seeds_default(ptx_uni: list[UniAF3Config]):
    # NOTE: Protenix config does not include seeds
    assert ptx_uni[0].seeds == [42]


##########################################
# ProtenixConfig -> UniAF3Config -> ProtenixConfig
##########################################
def test_roundtrip_sequence_count(
    ptx_rt: ProtenixConfig, protenix_confs: ProtenixConfig
):
    assert len(ptx_rt) == len(protenix_confs) == 1
    assert len(ptx_rt[0].sequences) == len(protenix_confs[0].sequences) == 6


def test_roundtrip_protein_sequence(
    ptx_rt: ProtenixConfig, protenix_confs: ProtenixConfig
):
    src = protenix_confs[0].sequences[0].proteinChain
    assert src is not None
    prot = ptx_rt[0].sequences[0].proteinChain
    assert prot is not None
    assert prot == src


def test_roundtrip_ligand_ccd(ptx_rt: ProtenixConfig, protenix_confs: ProtenixConfig):
    src = protenix_confs[0].sequences[3].ligand
    assert src is not None
    lig = ptx_rt[0].sequences[3].ligand
    assert lig is not None
    assert lig == src


def test_roundtrip_covalent_bond(
    ptx_rt: ProtenixConfig, protenix_confs: ProtenixConfig
):
    src_bonds = protenix_confs[0].covalent_bonds
    assert src_bonds is not None
    rt_bonds = ptx_rt[0].covalent_bonds
    assert rt_bonds is not None

    assert rt_bonds == src_bonds


def test_roundtrip_contact_constraint(
    ptx_rt: ProtenixConfig, protenix_confs: ProtenixConfig
):
    src = protenix_confs[0].constraint.contact
    assert src is not None
    ct = ptx_rt[0].constraint.contact
    assert ct is not None
    assert ct == src


def test_roundtrip_pocket_restraint(
    ptx_rt: ProtenixConfig, protenix_confs: ProtenixConfig
):
    src = protenix_confs[0].constraint.pocket
    assert src is not None
    pk = ptx_rt[0].constraint.pocket
    assert pk is not None
    assert pk == src
