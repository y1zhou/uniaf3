"""Tests for UniAF3Config -> ChaiConfig adapter."""

import pytest

from uniaf3.schema import ChaiConfig, UniAF3Config
from uniaf3.schema.base import (
    Glycan,
    Ligand,
    Polymer,
    PolymerType,
    ProteinSeq,
    StructuralTemplate,
)
from uniaf3.schema.chai import ChaiEntityType


@pytest.fixture(scope="module")
def chai(uniaf3_conf: UniAF3Config):
    """Convert UniAF3 to Chai config."""
    from uniaf3.adapters import to_chai

    with pytest.warns(UserWarning):
        return to_chai(uniaf3_conf, strict=False)


# ruff: noqa: S101
def test_entity_count(uniaf3_conf: UniAF3Config, chai: ChaiConfig):
    # protein id=["A","B"] expands to 2 entities; dna, 2 ligands, 1 glycan = 6
    assert len(chai.entities) == 6


def test_warns_on_ccd_to_smiles_conversion(uniaf3_conf: UniAF3Config):
    from uniaf3.adapters import to_chai

    with pytest.warns(UserWarning) as records:
        _ = to_chai(uniaf3_conf, strict=False)
    assert any("Ligand.ccd 'ATP' is converted" in str(w.message) for w in records)


def test_protein_entity_type(uniaf3_conf: UniAF3Config, chai: ChaiConfig):
    prot = chai.entities[0]
    assert prot.entity_type == ChaiEntityType.Protein
    assert prot.entity_name == "A"


def test_protein_modification_inlined(uniaf3_conf: UniAF3Config, chai: ChaiConfig):
    """UniAF3 modifications should be inlined as (CCD) tokens in Chai sequence."""
    prot = chai.entities[0]
    src = uniaf3_conf.sequences[0]
    assert isinstance(src, ProteinSeq)
    assert src.modifications is not None
    # First modification at position 1: HY3
    assert prot.sequence.startswith("(HY3)")


def test_dna_entity(uniaf3_conf: UniAF3Config, chai: ChaiConfig):
    dna = chai.entities[2]
    src = uniaf3_conf.sequences[1]
    assert isinstance(src, Polymer)
    assert dna.entity_type == ChaiEntityType.DNA
    assert dna.sequence == src.sequence
    assert dna.entity_name == "C"


def test_ligand_ccd_to_smiles(uniaf3_conf: UniAF3Config, chai: ChaiConfig):
    """CCD ligands should be converted to SMILES for Chai."""
    lig = chai.entities[3]
    src = uniaf3_conf.sequences[2]
    assert isinstance(src, Ligand)
    assert lig.entity_type == ChaiEntityType.Ligand
    # CCD ATP should be resolved to a SMILES string
    assert lig.sequence != "ATP"
    assert len(lig.sequence) > 0


def test_ligand_smiles_preserved(uniaf3_conf: UniAF3Config, chai: ChaiConfig):
    lig = chai.entities[4]
    src = uniaf3_conf.sequences[3]
    assert isinstance(src, Ligand)
    assert lig.entity_type == ChaiEntityType.Ligand
    assert lig.sequence == src.smiles


def test_glycan_entity(uniaf3_conf: UniAF3Config, chai: ChaiConfig):
    glycan = chai.entities[5]
    src = uniaf3_conf.sequences[4]
    assert isinstance(src, Glycan)
    assert glycan.entity_type == ChaiEntityType.Glycan
    assert glycan.sequence == src.chai_str


def test_covalent_restraint(uniaf3_conf: UniAF3Config, chai: ChaiConfig):
    assert chai.restraints is not None
    assert uniaf3_conf.covalent_bonds is not None
    cov = [r for r in chai.restraints if r.connection_type == "covalent"]
    assert len(cov) == len(uniaf3_conf.covalent_bonds) == 1
    r = cov[0]
    # Chain IDs are remapped to Chai A-Z ordering
    assert r.chainA == "B"  # B is 2nd entity
    assert r.chainB == "D"  # D is 4th entity (CCD ligand)
    assert r.res_idxA is not None and r.res_idxB is not None
    assert "@" in r.res_idxA  # atom name for polymer
    assert "@" in r.res_idxB  # atom name for ligand

    src = uniaf3_conf.covalent_bonds[0]
    assert src.atom2.atom_name is not None
    assert src.atom2.atom_name in r.res_idxB


def test_contact_restraint(uniaf3_conf: UniAF3Config, chai: ChaiConfig):
    assert chai.restraints is not None
    assert uniaf3_conf.contact_restraints is not None
    ct = [r for r in chai.restraints if r.connection_type == "contact"]
    assert len(ct) == len(uniaf3_conf.contact_restraints)
    r = ct[0]
    src = uniaf3_conf.contact_restraints[0]
    assert r.max_distance_angstrom == src.max_distance


def test_pocket_restraints(uniaf3_conf: UniAF3Config, chai: ChaiConfig):
    assert chai.restraints is not None
    assert uniaf3_conf.pocket_restraints is not None
    pk = [r for r in chai.restraints if r.connection_type == "pocket"]
    # Each contact token generates a separate pocket restraint row
    src = uniaf3_conf.pocket_restraints[0]
    assert len(pk) == len(src.contact_tokens)


def test_inference_params(uniaf3_conf: UniAF3Config, chai: ChaiConfig):
    assert chai.num_trunk_recycles == uniaf3_conf.aux.num_trunk_recycles
    assert chai.num_diffn_timesteps == uniaf3_conf.aux.num_diffn_timesteps
    assert chai.num_diffn_samples == uniaf3_conf.aux.num_diffn_samples
    assert chai.num_trunk_samples == uniaf3_conf.aux.num_trunk_samples


def test_seed(uniaf3_conf: UniAF3Config, chai: ChaiConfig):
    # Only first seed is taken
    assert chai.seed == uniaf3_conf.aux.seeds[0]


# --- MSA and template conversion tests ---


def _make_a3m_content(query_seq: str, num_hits: int = 3) -> str:
    """Create a minimal A3M file content."""
    lines = [f">query\n{query_seq}"]
    for i in range(num_hits):
        gap_seq = query_seq[: max(1, len(query_seq) - i)] + "-" * i
        lines.append(f">UniRef100_hit{i}\n{gap_seq}")
    return "\n".join(lines) + "\n"


def _make_paired_a3m_content(query_seq: str, num_hits: int = 2) -> str:
    """Create a minimal paired A3M file content."""
    lines = [f">query\n{query_seq}"]
    for i in range(num_hits):
        lines.append(f">paired_hit{i}\n{query_seq}")
    return "\n".join(lines) + "\n"


@pytest.fixture
def msa_config_with_files(tmp_path):
    """Create a UniAF3Config with ProteinSeq that has actual MSA files."""
    from uniaf3.utils import hash_sequence

    seq_str = "MVLSPADKTNVK"
    seq_hash = hash_sequence(seq_str)

    # Create MSA directory structure
    a3ms_dir = tmp_path / "msas" / "a3ms"
    a3ms_dir.mkdir(parents=True)

    single_path = a3ms_dir / f"{seq_hash}.single.a3m"
    single_path.write_text(_make_a3m_content(seq_str))

    paired_path = a3ms_dir / f"{seq_hash}.pair.a3m"
    paired_path.write_text(_make_paired_a3m_content(seq_str))

    config = UniAF3Config(
        sequences=[
            ProteinSeq(
                polymer_type=PolymerType.Protein,
                id="A",
                sequence=seq_str,
                unpaired_msa_path=str(single_path),
                paired_msa_path=str(paired_path),
                msa_dir=str(tmp_path / "msas"),
            )
        ]
    )
    return config


def test_msa_directory_set_when_msa_data_present(msa_config_with_files, tmp_path):
    """When ProteinSeq has MSA data and msa_dir is provided, chai.msa_directory is set."""
    from uniaf3.adapters import to_chai

    out_dir = tmp_path / "chai_msa_out"
    chai = to_chai(msa_config_with_files, msa_dir=out_dir)

    assert chai.msa_directory is not None
    assert chai.msa_directory == str(out_dir.resolve())

    # Check that .aligned.pqt file was created
    import polars as pl

    from uniaf3.utils import hash_sequence

    seq_hash = hash_sequence(msa_config_with_files.sequences[0].sequence.upper())
    pqt_path = out_dir.resolve() / f"{seq_hash}.aligned.pqt"
    assert pqt_path.exists()

    # Verify parquet has expected columns
    df = pl.read_parquet(pqt_path)
    assert set(df.columns) == {"sequence", "source_database", "pairing_key", "comment"}
    assert df.height > 0
    assert df.item(0, "source_database") == "query"


def test_msa_directory_none_when_no_msa_data(tmp_path):
    """When ProteinSeq has no MSA data, chai.msa_directory stays None."""
    from uniaf3.adapters import to_chai

    config = UniAF3Config(
        sequences=[
            ProteinSeq(
                polymer_type=PolymerType.Protein, id="A", sequence="MVLSPADKTNVK"
            )
        ]
    )
    chai = to_chai(config, msa_dir=tmp_path / "out")
    assert chai.msa_directory is None


def test_warns_when_msa_present_but_no_msa_dir(msa_config_with_files):
    """When MSA data exists but no msa_dir param, a lossy warning is emitted."""
    from uniaf3.adapters import to_chai

    with pytest.warns(UserWarning, match="MSA information is dropped"):
        chai = to_chai(msa_config_with_files)

    assert chai.msa_directory is None


def test_template_reconstruction_without_m8(tmp_path):
    """StructuralTemplate objects should be reconstructed into an m8 file."""
    from uniaf3.adapters import to_chai

    config = UniAF3Config(
        sequences=[
            ProteinSeq(
                polymer_type=PolymerType.Protein,
                id="A",
                sequence="MVLSPADKTNVK",
                templates=[
                    StructuralTemplate(
                        path="/some/path/1abc.cif.gz",
                        query_idx=[0, 1, 2, 3, 4],
                        template_idx=[10, 11, 12, 13, 14],
                        template_chains=["B"],
                    )
                ],
            )
        ]
    )

    out_dir = tmp_path / "chai_out"
    with pytest.warns(UserWarning, match="placeholder scoring"):
        chai = to_chai(config, msa_dir=out_dir)

    assert chai.template_hits_path is not None
    from pathlib import Path

    m8_path = Path(chai.template_hits_path)
    assert m8_path.exists()

    content = m8_path.read_text()
    assert "1abc_B" in content
    assert "reconstructed_by_uniaf3" in content


def test_template_warns_on_boltz_fields(tmp_path):
    """Boltz-specific template fields should emit a lossy warning."""
    from uniaf3.adapters import to_chai

    config = UniAF3Config(
        sequences=[
            ProteinSeq(
                polymer_type=PolymerType.Protein,
                id="A",
                sequence="MVLSPADKTNVK",
                templates=[
                    StructuralTemplate(
                        path="/some/path/1abc.cif",
                        boltz_enable_force=True,
                        boltz_template_threshold=2.0,
                    )
                ],
            )
        ]
    )

    out_dir = tmp_path / "chai_out"
    with pytest.warns(UserWarning) as records:
        chai = to_chai(config, msa_dir=out_dir)

    assert any("boltz_enable_force" in str(w.message) for w in records)


def test_template_warns_when_no_msa_dir(tmp_path):
    """Templates without msa_dir should emit a lossy warning."""
    from uniaf3.adapters import to_chai

    config = UniAF3Config(
        sequences=[
            ProteinSeq(
                polymer_type=PolymerType.Protein,
                id="A",
                sequence="MVLSPADKTNVK",
                templates=[StructuralTemplate(path="/some/path/1abc.cif")],
            )
        ]
    )

    with pytest.warns(UserWarning, match="template information is dropped"):
        chai = to_chai(config)

    assert chai.template_hits_path is None
