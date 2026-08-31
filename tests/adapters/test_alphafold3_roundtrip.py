"""Tests for AF3Config -> UniAF3Config -> BoltzConfig adapter."""

import pytest

from uniaf3.schema import AF3Config, UniAF3Config
from uniaf3.schema.base import Ligand, Polymer, PolymerType, ProteinSeq


@pytest.fixture(scope="module")
def af3_uni(
    af3_conf: AF3Config, tmp_path_factory: pytest.TempPathFactory
) -> UniAF3Config:
    """Convert AF3Config to UniAF3Config."""
    from uniaf3.adapters import from_alphafold3

    with pytest.warns(UserWarning):
        return from_alphafold3(af3_conf)


@pytest.fixture(scope="module")
def af3_rt(af3_uni: UniAF3Config, tmp_path_factory: pytest.TempPathFactory):
    """Convert UniAF3Config back to AF3Config, i.e. roundtrip."""
    from uniaf3.adapters import to_alphafold3

    return to_alphafold3(af3_uni, name="test-roundtrip", strict=False)


# ruff: noqa: S101
##########################################
# AF3Config -> UniAF3Config
##########################################
def test_sequence_count(af3_uni: UniAF3Config, af3_conf: AF3Config):
    assert len(af3_uni.sequences) == len(af3_conf.sequences)


def test_to_uniaf3_af3_preserves_sequences(
    af3_conf: AF3Config, tmp_path_factory: pytest.TempPathFactory
):
    """to_uniaf3() for AF3 should produce the same number of sequences."""
    from uniaf3.adapters import to_uniaf3

    with pytest.warns(UserWarning):
        result = to_uniaf3(af3_conf)
    assert isinstance(result, UniAF3Config)
    assert len(result.sequences) == len(af3_conf.sequences)


def test_from_af3_warns_on_lossy_metadata(
    af3_conf: AF3Config, tmp_path_factory: pytest.TempPathFactory
):
    from uniaf3.adapters import from_alphafold3

    conf = af3_conf.model_copy(deep=True)
    conf.userCCD = "data_MY_LIGAND"

    with pytest.warns(UserWarning) as records:
        _ = from_alphafold3(conf)
    assert any("AF3Config.{userCCD,userCCDPath}" in str(w.message) for w in records)


def test_seeds(af3_uni: UniAF3Config, af3_conf: AF3Config):
    assert af3_uni.aux.seeds == af3_conf.modelSeeds == [10, 42]


def test_protein_fields(af3_uni: UniAF3Config, af3_conf: AF3Config):
    prot = af3_uni.sequences[0]
    src = af3_conf.sequences[0].protein
    assert isinstance(prot, ProteinSeq)
    assert src is not None
    assert prot.id == src.id
    assert prot.sequence == src.sequence
    assert prot.description == src.description
    assert prot.polymer_type == PolymerType.Protein


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


def test_protein_msa_paths_preserved(af3_uni: UniAF3Config, af3_conf: AF3Config):
    """AF3 MSA paths are preserved directly in ProteinSeq."""
    prot = af3_uni.sequences[0]
    src = af3_conf.sequences[0].protein
    assert isinstance(prot, ProteinSeq)
    assert src is not None
    if src.unpairedMsaPath:
        assert prot.unpaired_msa == src.unpairedMsaPath
    else:
        assert prot.unpaired_msa is None
    if src.pairedMsaPath:
        assert prot.paired_msa == src.pairedMsaPath
    else:
        assert prot.paired_msa is None


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
    assert dna.polymer_type == PolymerType.DNA
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
    assert rna.polymer_type == PolymerType.RNA
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
            assert prot.protein is not None
            assert src.protein.id == prot.protein.id
            assert src.protein.sequence == prot.protein.sequence
            assert src.protein.modifications == prot.protein.modifications
            assert src.protein.description == prot.protein.description
            assert src.protein.templates == prot.protein.templates
            # NOTE: MSA paths may differ after roundtrip because UniAF3
            # uses hash-based directory lookup while AF3 uses direct paths.


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


@pytest.mark.parametrize("templates", [None, []], ids=["null", "empty"])
def test_template_presence_preserved_from_af3(templates):
    from uniaf3.adapters import from_alphafold3
    from uniaf3.schema.alphafold3 import AF3Protein, AF3SequenceEntry

    config = AF3Config(
        name="template-presence",
        modelSeeds=[1],
        sequences=[
            AF3SequenceEntry(
                protein=AF3Protein(id="A", sequence="M", templates=templates)
            )
        ],
    )

    with pytest.warns(UserWarning, match="AF3Config.name"):
        converted = from_alphafold3(config)
    protein = converted.sequences[0]
    assert isinstance(protein, ProteinSeq)
    assert protein.templates == templates
