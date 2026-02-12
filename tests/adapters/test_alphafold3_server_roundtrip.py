"""Tests for AF3ServerConfig -> UniAF3Config -> AF3ServerConfig adapter."""

import pytest

from uniaf3.schema import AF3ServerConfig, UniAF3Config
from uniaf3.schema.base import Glycan, Ligand, Polymer, ProteinSeq


@pytest.fixture(scope="module")
def af3s_uni(af3_server_confs: AF3ServerConfig):
    """Convert AF3ServerConfig to UniAF3Config."""
    from uniaf3.adapters import from_alphafold3_server

    return from_alphafold3_server(af3_server_confs)


@pytest.fixture(scope="module")
def af3s_rt(af3s_uni: list[UniAF3Config]):
    """Convert list[UniAF3Config] → AF3ServerConfig → list[UniAF3Config], i.e. roundtrip."""
    from uniaf3.adapters import to_alphafold3_server

    return to_alphafold3_server(af3s_uni, name="test-roundtrip", strict=False)


# ruff: noqa: S101
##########################################
# AF3ServerConfig -> UniAF3Config
##########################################
def test_num_jobs_seqs_match(
    af3s_uni: list[UniAF3Config], af3_server_confs: AF3ServerConfig
):
    assert len(af3s_uni) == len(af3_server_confs) == 2
    assert len(af3s_uni[0].sequences) == len(af3_server_confs[0].sequences) == 9
    assert len(af3s_uni[1].sequences) == len(af3_server_confs[1].sequences) == 2


def test_protein_fields(
    af3s_uni: list[UniAF3Config], af3_server_confs: AF3ServerConfig
):
    prot = af3s_uni[0].sequences[0]
    src = af3_server_confs[0].sequences[0].proteinChain
    assert isinstance(prot, ProteinSeq)
    assert src is not None
    assert prot.sequence == src.sequence

    assert src.modifications is not None
    assert prot.modifications is not None
    for mod_uni, mod_src in zip(prot.modifications, src.modifications, strict=True):
        assert f"CCD_{mod_uni.ccd}" == mod_src.ptmType
        assert mod_uni.position == mod_src.ptmPosition


def test_transferred_glycans(
    af3s_uni: list[UniAF3Config], af3_server_confs: AF3ServerConfig
):
    src = af3_server_confs[0].sequences[0].proteinChain
    assert src is not None
    src_glycans = src.glycans
    assert src_glycans is not None

    for src_glycan in src_glycans:
        assert src_glycan.position

    uni_glycans = [x for x in af3s_uni[0].sequences if isinstance(x, Glycan)]

    # TODO: in AF3 the glycan is associated with the protein chain, but
    # in UniAF3 it's a separate sequence with corresponding constraints.


def test_protein_chain_id_generated(af3s_uni: list[UniAF3Config]):
    prot = af3s_uni[0].sequences[0]
    assert isinstance(prot, ProteinSeq)
    # count=1 → single chain id "A"
    assert prot.id == "A"


def test_dna_modifications(
    af3s_uni: list[UniAF3Config], af3_server_confs: AF3ServerConfig
):
    dna = af3s_uni[0].sequences[2]
    src = af3_server_confs[0].sequences[2].dnaSequence
    assert isinstance(dna, Polymer)
    assert src is not None
    assert dna.modifications is not None
    assert src.modifications is not None

    for mod_uni, mod_src in zip(dna.modifications, src.modifications, strict=True):
        assert f"CCD_{mod_uni.ccd}" == mod_src.modificationType
        assert mod_uni.position == mod_src.basePosition


def test_rna_modifications(
    af3s_uni: list[UniAF3Config], af3_server_confs: AF3ServerConfig
):
    rna = af3s_uni[0].sequences[4]
    src = af3_server_confs[0].sequences[4].rnaSequence
    assert isinstance(rna, Polymer)
    assert src is not None
    assert rna.modifications is not None
    assert src.modifications is not None
    for mod_uni, mod_src in zip(rna.modifications, src.modifications, strict=True):
        assert f"CCD_{mod_uni.ccd}" == mod_src.modificationType
        assert mod_uni.position == mod_src.basePosition


def test_ligand_ccd(af3s_uni: list[UniAF3Config], af3_server_confs: AF3ServerConfig):
    lig = af3s_uni[0].sequences[5]
    src = af3_server_confs[0].sequences[5].ligand
    assert isinstance(lig, Ligand)
    assert src is not None
    assert lig.ccd == [src.ligand.removeprefix("CCD_")]


def test_ligand_count_as_chain_ids(
    af3s_uni: list[UniAF3Config], af3_server_confs: AF3ServerConfig
):
    """count=2 for HEM → 2 chain IDs."""
    lig = af3s_uni[0].sequences[6]
    src = af3_server_confs[0].sequences[6].ligand
    assert isinstance(lig, Ligand)
    assert src is not None
    assert isinstance(lig.id, list)
    assert len(lig.id) == src.count
    assert lig.id == ["G", "H"]


def test_ion_as_ligand(af3s_uni: list[UniAF3Config], af3_server_confs: AF3ServerConfig):
    ion = af3s_uni[0].sequences[7]
    src = af3_server_confs[0].sequences[7].ion
    assert isinstance(ion, Ligand)
    assert src is not None
    assert ion.ccd == [src.ion]
    assert isinstance(ion.id, list)
    assert len(ion.id) == src.count


def test_seeds_from_job(af3s_uni: list[UniAF3Config]):
    # AF3 server fixture has empty modelSeeds → default [42]
    assert af3s_uni[0].seeds == [42]


##########################################
# AF3ServerConfig -> UniAF3Config -> AF3ServerConfig
##########################################
def test_roundtrip_nums(af3s_rt: AF3ServerConfig, af3_server_confs: AF3ServerConfig):
    assert len(af3s_rt) == len(af3_server_confs) == 2
    assert len(af3s_rt[0].sequences) == len(af3_server_confs[0].sequences) == 9


def test_roundtrip_protein_sequence(
    af3s_rt: AF3ServerConfig, af3_server_confs: AF3ServerConfig
):
    src = af3_server_confs[0].sequences[0].proteinChain
    assert src is not None
    prot = af3s_rt[0].sequences[0].proteinChain
    assert prot is not None
    assert prot == src

    assert src.modifications is not None
    assert prot.modifications is not None
    for mod_rt, mod_src in zip(prot.modifications, src.modifications, strict=True):
        assert mod_rt.ptmType == mod_src.ptmType
        assert mod_rt.ptmPosition == mod_src.ptmPosition
