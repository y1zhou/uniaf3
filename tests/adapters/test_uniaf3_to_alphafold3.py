"""Tests for UniAF3Config -> AF3Config adapter."""

import pytest

from uniaf3.schema import AF3Config, UniAF3Config
from uniaf3.schema.base import (
    Ligand,
    Polymer,
    ProteinSeq,
)


@pytest.fixture(scope="module")
def af3(uniaf3_conf: UniAF3Config, tmp_path_factory: pytest.TempPathFactory):
    """Convert UniAF3 to AlphaFold3 config."""
    from uniaf3.adapters import to_alphafold3

    return to_alphafold3(
        uniaf3_conf,
        msa_dir=tmp_path_factory.mktemp("msa"),
        name="test-af3-adapter",
        strict=False,
    )


# ruff: noqa: S101
def test_unsupported_glycan_strict(
    uniaf3_conf: UniAF3Config, tmp_path_factory: pytest.TempPathFactory
):
    from uniaf3.adapters import to_alphafold3

    with pytest.raises(ValueError, match="Glycans are not directly supported in AF3"):
        to_alphafold3(
            uniaf3_conf,
            msa_dir=tmp_path_factory.mktemp("msa"),
            name="test-af3-adapter",
            strict=True,
        )


def test_name_and_seeds(uniaf3_conf: UniAF3Config, af3: AF3Config):
    assert af3.name == "test-af3-adapter"
    assert af3.modelSeeds == uniaf3_conf.seeds
    assert af3.dialect == "alphafold3"


def test_sequence_count_drops_glycan(uniaf3_conf: UniAF3Config, af3: AF3Config):
    # protein + dna + 2 ligands = 4; 1 glycan dropped
    assert len(af3.sequences) == len(uniaf3_conf.sequences) - 1 == 4


def test_protein_fields(uniaf3_conf: UniAF3Config, af3: AF3Config):
    prot = af3.sequences[0].protein
    src = uniaf3_conf.sequences[0]
    assert isinstance(src, ProteinSeq)
    assert prot is not None
    assert prot.id == src.id
    assert prot.sequence == src.sequence
    assert prot.description == src.description

    assert prot.modifications is not None
    assert src.modifications is not None
    assert len(prot.modifications) == len(src.modifications)
    for p_mod, s_mod in zip(prot.modifications, src.modifications, strict=True):
        assert p_mod.ptmType == s_mod.ccd
        assert p_mod.ptmPosition == s_mod.position


def test_dna_fields(uniaf3_conf: UniAF3Config, af3: AF3Config):
    dna = af3.sequences[1].dna
    src = uniaf3_conf.sequences[1]
    assert isinstance(src, Polymer)
    assert dna is not None
    assert dna.id == src.id
    assert dna.sequence == src.sequence


def test_ligand_ccd(uniaf3_conf: UniAF3Config, af3: AF3Config):
    lig = af3.sequences[2].ligand
    src = uniaf3_conf.sequences[2]
    assert isinstance(src, Ligand)
    assert lig is not None
    assert lig.ccdCodes == src.ccd


def test_ligand_smiles(uniaf3_conf: UniAF3Config, af3: AF3Config):
    lig = af3.sequences[3].ligand
    src = uniaf3_conf.sequences[3]
    assert isinstance(src, Ligand)
    assert lig is not None
    assert lig.smiles == src.smiles


def test_only_covalent_bonds(uniaf3_conf: UniAF3Config, af3: AF3Config):
    # Only covalent bonds preserved in AF3
    assert af3.bondedAtomPairs is not None
    assert len(af3.bondedAtomPairs) == 1
    assert uniaf3_conf.covalent_bonds is not None
    a1, a2 = af3.bondedAtomPairs[0]
    src = uniaf3_conf.covalent_bonds[0]
    assert a1 == (src.atom1.chain_id, src.atom1.residue_idx, src.atom1.atom_name)
    assert a2 == (src.atom2.chain_id, src.atom2.residue_idx, src.atom2.atom_name)
