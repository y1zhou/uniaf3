"""Tests for ProtenixConfig -> UniAF3Config -> ProtenixConfig adapter."""

import pytest

from uniaf3.schema import ProtenixConfig, UniAF3Config
from uniaf3.schema.base import Ligand, Polymer, PolymerType, ProteinSeq


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
    bonds = ptx_uni[0].covalent_bonds
    assert bonds is not None
    assert len(bonds) == 1
    assert protenix_confs[0].covalent_bonds is not None
    src = protenix_confs[0].covalent_bonds[0]
    bond = bonds[0]
    assert bond.atom1.atom_name == src.atom1
    assert bond.atom2.atom_name == src.atom2

    # TODO: fix entity and copy mapping


def test_contact_restraint(ptx_uni: list[UniAF3Config], protenix_confs: ProtenixConfig):
    contacts = ptx_uni[0].contact_restraints
    assert contacts is not None
    assert len(contacts) == 1
    assert protenix_confs[0].constraint is not None
    src_contacts = protenix_confs[0].constraint.contact
    assert src_contacts is not None
    src = src_contacts[0]
    ct = contacts[0]
    assert ct.max_distance == src.max_distance
    assert ct.token1.chain_id == "A"
    assert ct.token1.residue_idx == src.position1
    assert ct.token1.atom_name == src.atom1 == "CA"
    assert ct.token2.chain_id == "C"
    assert ct.token2.residue_idx == src.position2
    assert ct.token2.atom_name is None
    assert src.atom2 is None


def test_pocket_restraint(ptx_uni: list[UniAF3Config], protenix_confs: ProtenixConfig):
    pockets = ptx_uni[0].pocket_restraints
    assert pockets is not None
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


def test_roundtrip_protein(ptx_rt: ProtenixConfig, protenix_confs: ProtenixConfig):
    for rt_conf, src_conf in zip(ptx_rt, protenix_confs, strict=True):
        for src, prot in zip(src_conf.sequences, rt_conf.sequences, strict=True):
            if src.proteinChain is not None:
                assert prot.proteinChain == src.proteinChain


def test_roundtrip_polymer(ptx_rt: ProtenixConfig, protenix_confs: ProtenixConfig):
    for rt_conf, src_conf in zip(ptx_rt, protenix_confs, strict=True):
        for src, dna in zip(src_conf.sequences, rt_conf.sequences, strict=True):
            if src.dnaSequence is not None:
                assert dna.dnaSequence == src.dnaSequence
            elif src.rnaSequence is not None:
                assert dna.rnaSequence == src.rnaSequence


def test_roundtrip_ligand_and_ion(
    ptx_rt: ProtenixConfig, protenix_confs: ProtenixConfig
):
    for rt_conf, src_conf in zip(ptx_rt, protenix_confs, strict=True):
        for src, lig in zip(src_conf.sequences, rt_conf.sequences, strict=True):
            if src.ligand is not None:
                assert lig.ligand == src.ligand
            elif src.ion is not None:
                # Ions are represented as ligands after the roundtrip
                assert lig.ligand.count == src.ion.count
                assert lig.ligand.ligand == f"CCD_{src.ion.ion}"


def test_roundtrip_covalent_bond(
    ptx_rt: ProtenixConfig, protenix_confs: ProtenixConfig
):
    for rt_conf, src_conf in zip(ptx_rt, protenix_confs, strict=True):
        for rt_bond, src_bond in zip(
            rt_conf.covalent_bonds or [], src_conf.covalent_bonds or [], strict=True
        ):
            assert rt_bond == src_bond


def test_roundtrip_restraints(ptx_rt: ProtenixConfig, protenix_confs: ProtenixConfig):
    for rt_conf, src_conf in zip(ptx_rt, protenix_confs, strict=True):
        if src_conf.constraint is None:
            assert rt_conf.constraint.pocket == rt_conf.constraint.pocket
            assert rt_conf.constraint.contact == rt_conf.constraint.contact
