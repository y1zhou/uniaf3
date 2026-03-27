"""Tests for ChaiConfig -> UniAF3Config -> ChaiConfig adapter."""

import pytest

from uniaf3.schema import ChaiConfig, UniAF3Config
from uniaf3.schema.base import Glycan, Ligand, Polymer, PolymerType, ProteinSeq
from uniaf3.schema.chai import ChaiEntityType
from uniaf3.utils import normalize_out_dir


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
    assert chai_uni.aux.seeds == [42]


def test_warns_on_missing_seed(chai_conf: ChaiConfig):
    from uniaf3.adapters import from_chai

    with pytest.warns(UserWarning) as records:
        _ = from_chai(chai_conf)
    assert any(
        "ChaiConfig.seed is missing; UniAF3Config.aux.seeds defaults to [42]."
        in str(w.message)
        for w in records
    )


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
        seed=1,
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


def test_from_chai_rna_entity():
    """from_chai should convert RNA entities to Polymer with RNA type."""
    from uniaf3.adapters import from_chai
    from uniaf3.schema.chai import ChaiEntity, ChaiEntityType

    conf = ChaiConfig(
        entities=[
            ChaiEntity(
                entity_type=ChaiEntityType.RNA,
                entity_name="R",
                sequence="ACGU",
            )
        ],
        seed=1,
    )
    result = from_chai(conf)
    assert len(result.sequences) == 1
    rna = result.sequences[0]
    assert isinstance(rna, Polymer)
    assert rna.polymer_type == PolymerType.RNA
    assert rna.sequence == "ACGU"


def test_from_chai_msa_directory_no_msa_dir_raises(tmp_path):
    """from_chai with msa_directory but no msa_dir param should raise."""
    from uniaf3.adapters import from_chai
    from uniaf3.schema.chai import ChaiEntity, ChaiEntityType

    # Create a fake MSA directory
    msa_dir = normalize_out_dir(tmp_path, "fake_msa")

    conf = ChaiConfig(
        entities=[
            ChaiEntity(
                entity_type=ChaiEntityType.Protein,
                entity_name="A",
                sequence="MVLSPADKTNVK",
            )
        ],
        seed=1,
        msa_directory=str(msa_dir),
    )
    with pytest.raises(ValueError, match="no msa_dir specified"):
        from_chai(conf, msa_dir=None)


def test_from_chai_msa_directory_nonexistent_raises(tmp_path):
    """from_chai with nonexistent msa_directory should raise ValueError."""
    from uniaf3.adapters import from_chai
    from uniaf3.schema.chai import ChaiEntity, ChaiEntityType

    conf = ChaiConfig(
        entities=[
            ChaiEntity(
                entity_type=ChaiEntityType.Protein,
                entity_name="A",
                sequence="MVLSPADKTNVK",
            )
        ],
        seed=1,
        msa_directory="/nonexistent/path",
    )
    with pytest.raises(ValueError, match="MSA directory does not exist"):
        from_chai(conf, msa_dir=tmp_path)


def test_from_chai_template_hits_path_no_msa_dir_raises(tmp_path):
    """from_chai with template_hits_path but no msa_dir param should raise."""
    from uniaf3.adapters import from_chai
    from uniaf3.schema.chai import ChaiEntity, ChaiEntityType

    # Create a fake m8 file
    m8_file = tmp_path / "templates.m8"
    m8_file.write_text(
        "hash\t1abc_A\t100.0\t10\t0\t0\t1\t10\t1\t10\t1e-10\t100.0\t1M\n"
    )

    conf = ChaiConfig(
        entities=[
            ChaiEntity(
                entity_type=ChaiEntityType.Protein,
                entity_name="A",
                sequence="MVLSPADKTNVK",
            )
        ],
        seed=1,
        template_hits_path=str(m8_file),
    )
    with pytest.raises(ValueError, match="no msa_dir specified"):
        from_chai(conf, msa_dir=None)


def test_from_chai_inline_modification_unknown_ccd_uses_x():
    """from_chai modification token with no canonical mapping should use 'X'."""
    from uniaf3.adapters import from_chai
    from uniaf3.schema.chai import ChaiEntity, ChaiEntityType

    # Use an unknown CCD that won't have a canonical one-letter code
    conf = ChaiConfig(
        entities=[
            ChaiEntity(
                entity_type=ChaiEntityType.Protein,
                entity_name="A",
                sequence="(XYZ)VLSPADKTNVK",  # XYZ is an unknown CCD
            )
        ],
        seed=1,
    )
    with pytest.warns(UserWarning, match="no canonical one-letter mapping"):
        result = from_chai(conf)

    prot = result.sequences[0]
    assert isinstance(prot, ProteinSeq)
    # The unknown CCD position should be represented as 'X'
    assert "X" in prot.sequence


def test_from_chai_with_msa_parquet(tmp_path):
    """from_chai with msa_directory should reconstruct A3M files from parquet."""
    from uniaf3.adapters import from_chai, to_chai
    from uniaf3.utils import hash_sequence

    seq_str = "MVLSPADKTNVK"
    seq_hash = hash_sequence(seq_str)
    a3ms_dir = normalize_out_dir(tmp_path, "a3ms")

    single_path = a3ms_dir / f"{seq_hash}.single.a3m"
    single_path.write_text(f">query\n{seq_str}\n>hit1\n{seq_str[:-1]}-\n")

    paired_path = a3ms_dir / f"{seq_hash}.pair.a3m"
    paired_path.write_text(f">query\n{seq_str}\n>pair1\n{seq_str}\n")

    uniaf3_config = UniAF3Config(
        sequences=[
            ProteinSeq(
                polymer_type=PolymerType.Protein,
                id="A",
                sequence=seq_str,
                unpaired_msa=str(single_path),
                paired_msa=str(paired_path),
            )
        ]
    )

    msa_out = tmp_path / "chai_msa"
    chai_config = to_chai(uniaf3_config, msa_dir=msa_out)

    assert chai_config.msa_directory is not None

    # Now go from_chai back to UniAF3, which should reconstruct A3M files
    result_dir = tmp_path / "result_a3ms"
    result = from_chai(chai_config, msa_dir=result_dir)

    prot = result.sequences[0]
    assert isinstance(prot, ProteinSeq)
    assert prot.unpaired_msa is not None
    from pathlib import Path as _Path

    assert _Path(prot.unpaired_msa).exists()


def test_from_chai_msa_missing_parquet_raises(tmp_path):
    """from_chai should raise if expected parquet MSA file is missing."""
    from uniaf3.adapters import from_chai
    from uniaf3.schema.chai import ChaiEntity, ChaiEntityType

    # Create a fake MSA directory without the expected parquet
    msa_dir = normalize_out_dir(tmp_path, "msa")

    conf = ChaiConfig(
        entities=[
            ChaiEntity(
                entity_type=ChaiEntityType.Protein,
                entity_name="A",
                sequence="MVLSPADKTNVK",
            )
        ],
        seed=1,
        msa_directory=str(msa_dir),
    )
    with pytest.raises(ValueError, match="Expected MSA for"):
        from_chai(conf, msa_dir=tmp_path / "out")


def test_from_chai_template_nonexistent_m8_raises(tmp_path):
    """from_chai with nonexistent template_hits_path should raise ValueError."""
    from uniaf3.adapters import from_chai
    from uniaf3.schema.chai import ChaiEntity, ChaiEntityType

    conf = ChaiConfig(
        entities=[
            ChaiEntity(
                entity_type=ChaiEntityType.Protein,
                entity_name="A",
                sequence="MVLSPADKTNVK",
            )
        ],
        seed=1,
        template_hits_path="/nonexistent/templates.m8",
    )
    with pytest.raises(ValueError, match="template hits file does not exist"):
        from_chai(conf, msa_dir=tmp_path)


def test_from_chai_template_wrong_extension_raises(tmp_path):
    """from_chai with template_hits_path that's not .m8 should raise ValueError."""
    from uniaf3.adapters import from_chai
    from uniaf3.schema.chai import ChaiEntity, ChaiEntityType

    bad_tmpl = tmp_path / "templates.txt"
    bad_tmpl.write_text("content")

    conf = ChaiConfig(
        entities=[
            ChaiEntity(
                entity_type=ChaiEntityType.Protein,
                entity_name="A",
                sequence="MVLSPADKTNVK",
            )
        ],
        seed=1,
        template_hits_path=str(bad_tmpl),
    )
    with pytest.raises(ValueError, match="must be in .m8 format"):
        from_chai(conf, msa_dir=tmp_path)


def test_from_chai_template_with_valid_m8(tmp_path):
    """from_chai with valid m8 file should create StructuralTemplate entries."""
    from unittest.mock import patch

    from uniaf3.adapters import from_chai
    from uniaf3.schema.chai import ChaiEntity, ChaiEntityType
    from uniaf3.utils import hash_sequence

    seq_str = "MVLSPADKTNVK"
    seq_hash = hash_sequence(seq_str)

    # Create a valid m8 file with the correct query hash
    m8_file = tmp_path / "templates.m8"
    m8_file.write_text(
        f"{seq_hash}\t1abc_A\t95.0\t12\t0\t0\t1\t12\t1\t12\t1e-5\t50.0\t12M\n"
    )

    conf = ChaiConfig(
        entities=[
            ChaiEntity(
                entity_type=ChaiEntityType.Protein,
                entity_name="A",
                sequence=seq_str,
            )
        ],
        seed=1,
        template_hits_path=str(m8_file),
    )

    # Mock download_files to avoid actual network calls
    with patch("uniaf3.adapters.chai.download_files") as mock_dl:
        # Create the CIF file that would be downloaded
        tmpl_dir = normalize_out_dir(tmp_path / "out" / "templates")
        (tmpl_dir / "1ABC.cif.gz").write_bytes(b"")

        with pytest.warns(UserWarning) as records:
            result = from_chai(conf, msa_dir=tmp_path / "out")

    mock_dl.assert_called_once()
    assert any("StructuralTemplate" in str(w.message) for w in records)

    prot = result.sequences[0]
    assert isinstance(prot, ProteinSeq)
    assert prot.templates is not None
    assert len(prot.templates) == 1
    assert "1ABC" in prot.templates[0].path


def test_from_chai_template_unknown_query_hash_raises(tmp_path):
    """from_chai should raise if template hit has an unknown query hash."""

    from uniaf3.adapters import from_chai
    from uniaf3.schema.chai import ChaiEntity, ChaiEntityType

    seq_str = "MVLSPADKTNVK"

    # Create an m8 file with an unknown query hash
    m8_file = tmp_path / "templates.m8"
    m8_file.write_text(
        "unknownhash\t1abc_A\t95.0\t12\t0\t0\t1\t12\t1\t12\t1e-5\t50.0\t12M\n"
    )

    conf = ChaiConfig(
        entities=[
            ChaiEntity(
                entity_type=ChaiEntityType.Protein,
                entity_name="A",
                sequence=seq_str,
            )
        ],
        seed=1,
        template_hits_path=str(m8_file),
    )

    with pytest.raises(ValueError, match="query_id.*not found among protein sequences"):
        from_chai(conf, msa_dir=tmp_path / "out")


def test_from_chai_msa_no_msa_dir_raises(tmp_path):
    """from_chai with msa_directory but no msa_dir parameter should raise."""
    from uniaf3.adapters import from_chai
    from uniaf3.schema.chai import ChaiEntity, ChaiEntityType

    msa_dir = normalize_out_dir(tmp_path, "msa")

    conf = ChaiConfig(
        entities=[
            ChaiEntity(
                entity_type=ChaiEntityType.Protein,
                entity_name="A",
                sequence="MVLSPADKTNVK",
            )
        ],
        seed=1,
        msa_directory=str(msa_dir),
    )
    with pytest.raises(ValueError, match="no msa_dir specified"):
        from_chai(conf, msa_dir=None)
