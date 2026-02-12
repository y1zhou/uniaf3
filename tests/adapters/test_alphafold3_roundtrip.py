"""Tests for AF3Config -> UniAF3Config -> BoltzConfig adapter."""

from pathlib import Path

import pytest

from uniaf3.schema import AF3Config, UniAF3Config
from uniaf3.schema.base import Ligand, Polymer, PolymerType, ProteinSeq


@pytest.fixture(scope="module")
def af3_uni(af3_conf: AF3Config):
    """Convert AF3Config to UniAF3Config."""
    from uniaf3.adapters import from_alphafold3

    return from_alphafold3(af3_conf)


@pytest.fixture(scope="module")
def af3_rt(af3_uni: UniAF3Config):
    """Convert UniAF3Config back to AF3Config, i.e. roundtrip."""
    from uniaf3.adapters import to_alphafold3

    return to_alphafold3(af3_uni, name="test-roundtrip", strict=False)


# ruff: noqa: S101
##########################################
# AF3Config -> UniAF3Config
##########################################
def test_sequence_count(af3_uni: UniAF3Config, af3_conf: AF3Config):
    assert len(af3_uni.sequences) == len(af3_conf.sequences)


def test_seeds(af3_uni: UniAF3Config, af3_conf: AF3Config):
    assert af3_uni.seeds == af3_conf.modelSeeds == [10, 42]


def test_protein_fields(af3_uni: UniAF3Config, af3_conf: AF3Config):
    prot = af3_uni.sequences[0]
    src = af3_conf.sequences[0].protein
    assert isinstance(prot, ProteinSeq)
    assert src is not None
    assert prot.id == src.id
    assert prot.sequence == src.sequence
    assert prot.description == src.description
    assert prot.seq_type == PolymerType.Protein


def test_protein_modifications(af3_uni: UniAF3Config, af3_conf: AF3Config):
    prot = af3_uni.sequences[0]
    src = af3_conf.sequences[0].protein
    assert isinstance(prot, ProteinSeq)
    assert src is not None
    assert prot.modifications is not None
    assert src.modifications is not None
    assert len(prot.modifications) == len(src.modifications)
    assert prot.modifications[0].ccd == src.modifications[0].ptmType
    assert prot.modifications[0].position == src.modifications[0].ptmPosition


def test_protein_msa_dir_derived(af3_uni: UniAF3Config, af3_conf: AF3Config):
    """AF3 MSA paths → msa_dir is parent dir."""
    prot = af3_uni.sequences[0]
    src = af3_conf.sequences[0].protein
    assert isinstance(prot, ProteinSeq)
    assert src is not None
    if src.unpairedMsaPath:
        assert prot.msa_dir is not None
        assert prot.msa_dir == str(Path(src.unpairedMsaPath).parent)
    else:
        assert prot.msa_dir is None


def test_protein_templates(af3_uni: UniAF3Config, af3_conf: AF3Config):
    prot = af3_uni.sequences[1]
    src = af3_conf.sequences[1].protein
    assert isinstance(prot, ProteinSeq)
    assert src is not None
    assert src.templates is not None
    assert prot.templates is not None
    assert len(prot.templates) == len(src.templates)
    assert prot.templates[0].path == src.templates[0].mmcifPath
    assert prot.templates[0].query_idx == src.templates[0].queryIndices
    assert prot.templates[0].template_idx == src.templates[0].templateIndices


def test_dna_fields(af3_uni: UniAF3Config, af3_conf: AF3Config):
    dna = af3_uni.sequences[2]
    src = af3_conf.sequences[2].dna
    assert isinstance(dna, Polymer)
    assert src is not None
    assert dna.seq_type == PolymerType.DNA
    assert dna.sequence == src.sequence


def test_dna_modifications(af3_uni: UniAF3Config, af3_conf: AF3Config):
    dna = af3_uni.sequences[2]
    src = af3_conf.sequences[2].dna
    assert isinstance(dna, Polymer)
    assert src is not None
    assert dna.modifications is not None
    assert src.modifications is not None
    assert len(dna.modifications) == len(src.modifications)
    assert dna.modifications[0].ccd == src.modifications[0].modificationType


def test_rna_fields(af3_uni: UniAF3Config, af3_conf: AF3Config):
    rna = af3_uni.sequences[3]
    src = af3_conf.sequences[3].rna
    assert isinstance(rna, Polymer)
    assert src is not None
    assert rna.seq_type == PolymerType.RNA
    assert rna.sequence == src.sequence
    assert rna.description == src.description


def test_rna_modifications(af3_uni: UniAF3Config, af3_conf: AF3Config):
    rna = af3_uni.sequences[3]
    src = af3_conf.sequences[3].rna
    assert isinstance(rna, Polymer)
    assert src is not None
    assert rna.modifications is not None
    assert src.modifications is not None
    assert rna.modifications[0].ccd == src.modifications[0].modificationType


def test_ligand_ccd(af3_uni: UniAF3Config, af3_conf: AF3Config):
    lig = af3_uni.sequences[4]
    src = af3_conf.sequences[4].ligand
    assert isinstance(lig, Ligand)
    assert src is not None
    assert lig.ccd == src.ccdCodes
    assert lig.id == src.id


def test_ligand_multi_ccd(af3_uni: UniAF3Config, af3_conf: AF3Config):
    lig = af3_uni.sequences[5]
    src = af3_conf.sequences[5].ligand
    assert isinstance(lig, Ligand)
    assert src is not None
    assert lig.ccd == src.ccdCodes
    assert lig.ccd is not None
    assert len(lig.ccd) == 2


def test_ligand_smiles(af3_uni: UniAF3Config, af3_conf: AF3Config):
    lig = af3_uni.sequences[6]
    src = af3_conf.sequences[6].ligand
    assert isinstance(lig, Ligand)
    assert src is not None
    assert lig.smiles == src.smiles


def test_covalent_restraints(af3_uni: UniAF3Config, af3_conf: AF3Config):
    assert af3_uni.covalent_bonds is not None
    assert af3_conf.bondedAtomPairs is not None
    assert len(af3_uni.covalent_bonds) == len(af3_conf.bondedAtomPairs)
    for restraint, (a1, a2) in zip(
        af3_uni.covalent_bonds, af3_conf.bondedAtomPairs, strict=True
    ):
        assert restraint.atom1.chain_id == a1[0]
        assert restraint.atom1.residue_idx == a1[1]
        assert restraint.atom1.atom_name == a1[2]
        assert restraint.atom2.chain_id == a2[0]


##########################################
# AF3Config -> UniAF3Config -> AF3Config
##########################################
def test_roundtrip_sequences(af3_rt: AF3Config, af3_conf: AF3Config):
    # Glycan dropped
    assert len(af3_rt.sequences) == len(af3_conf.sequences) == 7


def test_roundtrip_seeds(af3_rt: AF3Config, af3_conf: AF3Config):
    assert af3_rt.modelSeeds == af3_conf.modelSeeds


def test_roundtrip_protein_sequence(af3_rt: AF3Config, af3_conf: AF3Config):
    for src, prot in zip(af3_conf.sequences, af3_rt.sequences, strict=True):
        if src.protein is not None:
            assert src.protein == prot.protein


def test_roundtrip_polymer(af3_rt: AF3Config, af3_conf: AF3Config):
    for src, dna in zip(af3_conf.sequences, af3_rt.sequences, strict=True):
        if src.dna is not None:
            assert src.dna == dna.dna
        elif src.rna is not None:
            assert src.rna == dna.rna


def test_roundtrip_ligand(af3_rt: AF3Config, af3_conf: AF3Config):
    for src, lig in zip(af3_conf.sequences, af3_rt.sequences, strict=True):
        if src.ligand is not None:
            assert src.ligand == lig.ligand


def test_roundtrip_covalent_bond(af3_rt: AF3Config, af3_conf: AF3Config):
    assert af3_conf.bondedAtomPairs is not None
    assert af3_rt.bondedAtomPairs is not None

    for (a1_rt, a2_rt), (a1_src, a2_src) in zip(
        af3_rt.bondedAtomPairs, af3_conf.bondedAtomPairs, strict=True
    ):
        assert a1_rt == a1_src
        assert a2_rt == a2_src
