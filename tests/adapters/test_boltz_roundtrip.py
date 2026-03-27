"""Tests for BoltzConfig -> UniAF3Config -> BoltzConfig adapter."""

from pathlib import Path

import pytest

from uniaf3.schema import BoltzConfig, UniAF3Config
from uniaf3.schema.base import Ligand, PolymerType, ProteinSeq


@pytest.fixture(scope="module")
def boltz_uni(boltz_conf: BoltzConfig, tmp_path_factory: pytest.TempPathFactory):
    """Convert BoltzConfig to UniAF3Config."""
    from uniaf3.adapters import from_boltz

    with pytest.warns(UserWarning):
        return from_boltz(boltz_conf, tmp_path_factory.mktemp("msa"))


@pytest.fixture(scope="module")
def boltz_rt(boltz_uni: UniAF3Config, tmp_path_factory: pytest.TempPathFactory):
    """Convert UniAF3Config back to BoltzConfig, i.e. roundtrip."""
    from uniaf3.adapters import to_boltz

    return to_boltz(boltz_uni, msa_dir=tmp_path_factory.mktemp("msa"), strict=False)


# ruff: noqa: S101
##########################################
# BoltzConfig -> UniAF3Config
##########################################
def test_sequence_count(boltz_uni: UniAF3Config, boltz_conf: BoltzConfig):
    assert len(boltz_uni.sequences) == len(boltz_conf.sequences)


def test_protein_fields(boltz_uni: UniAF3Config, boltz_conf: BoltzConfig):
    prot = boltz_uni.sequences[0]
    src = boltz_conf.sequences[0].protein
    assert isinstance(prot, ProteinSeq)
    assert src is not None
    assert prot.id == src.id
    assert prot.sequence == src.sequence
    assert prot.polymer_type == PolymerType.Protein
    assert prot.boltz_cyclic == src.cyclic


def test_protein_modifications(boltz_uni: UniAF3Config, boltz_conf: BoltzConfig):
    prot = boltz_uni.sequences[0]
    src = boltz_conf.sequences[0].protein
    assert isinstance(prot, ProteinSeq)
    assert src is not None
    assert prot.modifications is not None
    assert src.modifications is not None
    assert len(prot.modifications) == len(src.modifications)
    assert prot.modifications[0].ccd == src.modifications[0].ccd
    assert prot.modifications[0].position == src.modifications[0].position


def test_ligand_ccd(boltz_uni: UniAF3Config, boltz_conf: BoltzConfig):
    lig = boltz_uni.sequences[1]
    src = boltz_conf.sequences[1].ligand
    assert isinstance(lig, Ligand)
    assert src is not None
    # Boltz single CCD → UniAF3 list
    assert lig.ccd == [src.ccd]
    assert lig.id == src.id


def test_ligand_smiles(boltz_uni: UniAF3Config, boltz_conf: BoltzConfig):
    lig = boltz_uni.sequences[2]
    src = boltz_conf.sequences[2].ligand
    assert isinstance(lig, Ligand)
    assert src is not None
    assert lig.smiles == src.smiles
    assert lig.id == src.id


def test_bond_restraint(boltz_uni: UniAF3Config, boltz_conf: BoltzConfig):
    assert boltz_uni.covalent_bonds is not None
    assert boltz_conf.constraints is not None
    bond = boltz_uni.covalent_bonds[0]
    src = boltz_conf.constraints[0].bond
    assert src is not None
    assert bond.atom1.chain_id == src.atom1[0]
    assert bond.atom1.residue_idx == src.atom1[1]
    assert bond.atom1.atom_name == src.atom1[2]
    assert bond.atom2.chain_id == src.atom2[0]
    assert bond.atom2.residue_idx == src.atom2[1]
    assert bond.atom2.atom_name == src.atom2[2]


def test_contact_restraint(boltz_uni: UniAF3Config, boltz_conf: BoltzConfig):
    assert boltz_uni.contact_restraints is not None
    assert boltz_conf.constraints is not None
    ct = boltz_uni.contact_restraints[0]
    src = boltz_conf.constraints[1].contact
    assert src is not None
    assert ct.token1.chain_id == src.token1[0]
    assert ct.token1.residue_idx == int(src.token1[1])
    assert ct.max_distance == src.max_distance
    assert ct.boltz_enable_force == src.force


def test_pocket_restraint(boltz_uni: UniAF3Config, boltz_conf: BoltzConfig):
    assert boltz_uni.pocket_restraints is not None
    assert boltz_conf.constraints is not None
    pk = boltz_uni.pocket_restraints[0]
    src = boltz_conf.constraints[2].pocket
    assert src is not None
    assert pk.binder_chain == src.binder
    assert pk.max_distance == src.max_distance

    assert [(t.chain_id, t.residue_idx) for t in pk.contact_tokens] == src.contacts


def test_seeds_default(boltz_uni: UniAF3Config, boltz_conf: BoltzConfig):
    # NOTE: Boltz config does not include seeds
    assert boltz_uni.aux.seeds == [42]


def test_a3m_msa_writes_to_a3ms_subdir(
    boltz_conf: BoltzConfig, tmp_path_factory: pytest.TempPathFactory
):
    """A3M MSA files should be written to msa_dir/a3ms/ subdirectory."""
    from uniaf3.adapters import from_boltz

    msa_dir = tmp_path_factory.mktemp("msa_bug5")
    with pytest.warns(UserWarning):
        result = from_boltz(boltz_conf, msa_dir=msa_dir)

    prot = result.sequences[0]
    if isinstance(prot, ProteinSeq) and prot.unpaired_msa is not None:
        assert "/a3ms/" in prot.unpaired_msa


def test_warns_on_default_seed_from_boltz(
    boltz_conf: BoltzConfig, tmp_path_factory: pytest.TempPathFactory
):
    from uniaf3.adapters import from_boltz

    with pytest.warns(
        UserWarning, match=r"UniAF3Config\.aux\.seeds defaults to \[42\]"
    ):
        _ = from_boltz(boltz_conf, msa_dir=tmp_path_factory.mktemp("msa_warn_boltz"))


##########################################
# BoltzConfig -> UniAF3Config -> BoltzConfig
##########################################
def test_roundtrip_sequences(boltz_rt: BoltzConfig, boltz_conf: BoltzConfig):
    # Both have glycan chain dropped
    assert len(boltz_rt.sequences) == len(boltz_conf.sequences)


def test_roundtrip_protein_sequence(boltz_rt: BoltzConfig, boltz_conf: BoltzConfig):
    for src, prot in zip(boltz_conf.sequences, boltz_rt.sequences, strict=True):
        if src.protein is not None:
            assert src.protein == prot.protein


def test_roundtrip_polymer(boltz_rt: BoltzConfig, boltz_conf: BoltzConfig):
    for src, dna in zip(boltz_conf.sequences, boltz_rt.sequences, strict=True):
        if src.dna is not None:
            assert src.dna == dna.dna
        elif src.rna is not None:
            assert src.rna == dna.rna


def test_roundtrip_ligand(boltz_rt: BoltzConfig, boltz_conf: BoltzConfig):
    for src, lig in zip(boltz_conf.sequences, boltz_rt.sequences, strict=True):
        if src.ligand is not None:
            assert src.ligand == lig.ligand


def test_roundtrip_restraints(boltz_rt: BoltzConfig, boltz_conf: BoltzConfig):
    assert boltz_conf.constraints is not None
    assert boltz_rt.constraints is not None

    for rt_cst, boltz_cst in zip(
        boltz_rt.constraints, boltz_conf.constraints, strict=True
    ):
        if boltz_cst.bond is not None:
            assert rt_cst.bond == boltz_cst.bond
        elif boltz_cst.contact is not None:
            assert rt_cst.contact == boltz_cst.contact
        elif boltz_cst.pocket is not None:
            assert rt_cst.pocket == boltz_cst.pocket


def test_roundtrip_protein_templates(boltz_rt: BoltzConfig, boltz_conf: BoltzConfig):
    src = boltz_conf.templates
    assert src is not None
    prot = boltz_rt.templates
    assert prot is not None

    assert len(prot) == len(src) == 1
    # Template paths are resolved to absolute during roundtrip
    assert prot[0].cif is not None and src[0].cif is not None
    assert Path(prot[0].cif).name == Path(src[0].cif).name
    assert prot[0].force == src[0].force
    assert prot[0].threshold == src[0].threshold


def test_from_boltz_a3m_msa(tmp_path_factory):
    """from_boltz should handle .a3m MSA files by copying them to output dir."""
    from uniaf3.adapters import from_boltz
    from uniaf3.schema.boltz import BoltzConfig, BoltzProtein, BoltzSequenceEntry

    msa_source = tmp_path_factory.mktemp("msa_src")
    seq_str = "MVLSPADKTNVK"
    a3m_path = msa_source / "test.a3m"
    a3m_path.write_text(f">query\n{seq_str}\n>hit1\n{seq_str[:-1]}-\n")

    config = BoltzConfig(
        sequences=[
            BoltzSequenceEntry(
                protein=BoltzProtein(
                    id="A",
                    sequence=seq_str,
                    msa=str(a3m_path),
                )
            )
        ]
    )
    msa_out = tmp_path_factory.mktemp("msa_out_a3m")
    with pytest.warns(UserWarning):
        result = from_boltz(config, msa_dir=msa_out)

    prot = result.sequences[0]
    assert isinstance(prot, ProteinSeq)
    assert prot.unpaired_msa is not None
    assert prot.unpaired_msa.endswith(".single.a3m")
    assert Path(prot.unpaired_msa).exists()


def test_from_boltz_unsupported_msa_type_raises(tmp_path_factory):
    """from_boltz should raise on unsupported MSA file types."""
    from uniaf3.adapters import from_boltz
    from uniaf3.schema.boltz import BoltzConfig, BoltzProtein, BoltzSequenceEntry

    msa_source = tmp_path_factory.mktemp("msa_bad")
    seq_str = "MVLSPADKTNVK"
    fake_msa = msa_source / "test.fasta"
    fake_msa.write_text(f">query\n{seq_str}\n")

    config = BoltzConfig(
        sequences=[
            BoltzSequenceEntry(
                protein=BoltzProtein(
                    id="A",
                    sequence=seq_str,
                    msa=str(fake_msa),
                )
            )
        ]
    )
    msa_out = tmp_path_factory.mktemp("msa_out_bad")
    with pytest.raises(ValueError, match="Unsupported MSA file type"):
        from_boltz(config, msa_dir=msa_out)


def test_from_boltz_msa_no_dir_raises(tmp_path_factory):
    """from_boltz should raise if MSA is present but no msa_dir is given."""
    from uniaf3.adapters import from_boltz
    from uniaf3.schema.boltz import BoltzConfig, BoltzProtein, BoltzSequenceEntry

    config = BoltzConfig(
        sequences=[
            BoltzSequenceEntry(
                protein=BoltzProtein(
                    id="A",
                    sequence="MVLSPADKTNVK",
                    msa="/some/fake/path.csv",
                )
            )
        ]
    )
    with pytest.raises(ValueError, match="no msa_dir was provided"):
        from_boltz(config, msa_dir=None)


def test_from_boltz_template_unknown_chain_warns(tmp_path_factory):
    """Templates with unknown chain_id should emit a warning."""
    from uniaf3.adapters import from_boltz
    from uniaf3.schema.boltz import (
        BoltzConfig,
        BoltzProtein,
        BoltzSequenceEntry,
        BoltzTemplate,
    )

    config = BoltzConfig(
        sequences=[
            BoltzSequenceEntry(protein=BoltzProtein(id="A", sequence="MVLSPADKTNVK"))
        ],
        templates=[BoltzTemplate(cif="/some/path/1abc.cif.gz", chain_id=["Z"])],
    )
    with pytest.warns(UserWarning) as records:
        result = from_boltz(config)

    assert any(
        "references unknown UniAF3 protein chain" in str(w.message) for w in records
    )
    # Template should not be attached since chain Z doesn't exist
    prot = result.sequences[0]
    assert isinstance(prot, ProteinSeq)
    assert prot.templates is None


def test_from_boltz_template_no_chain_id_warns(tmp_path_factory):
    """Templates with no chain_id should emit a warning about being dropped."""
    from uniaf3.adapters import from_boltz
    from uniaf3.schema.boltz import (
        BoltzConfig,
        BoltzProtein,
        BoltzSequenceEntry,
        BoltzTemplate,
    )

    config = BoltzConfig(
        sequences=[
            BoltzSequenceEntry(protein=BoltzProtein(id="A", sequence="MVLSPADKTNVK"))
        ],
        templates=[BoltzTemplate(cif="/some/path/1abc.cif.gz")],
    )
    with pytest.warns(UserWarning) as records:
        result = from_boltz(config)

    assert any("chain_id is missing" in str(w.message) for w in records)
    prot = result.sequences[0]
    assert isinstance(prot, ProteinSeq)
    assert prot.templates is None


def test_from_boltz_affinity_property(tmp_path_factory):
    """from_boltz should convert affinity property to UniAF3Config.aux."""
    from uniaf3.adapters import from_boltz
    from uniaf3.schema.boltz import (
        BoltzAffinityProperty,
        BoltzConfig,
        BoltzPropertyEntry,
        BoltzProtein,
        BoltzSequenceEntry,
    )

    config = BoltzConfig(
        sequences=[
            BoltzSequenceEntry(protein=BoltzProtein(id="A", sequence="MVLSPADKTNVK"))
        ],
        properties=[BoltzPropertyEntry(affinity=BoltzAffinityProperty(binder="A"))],
    )
    with pytest.warns(UserWarning):
        result = from_boltz(config)

    assert result.aux.boltz_affinity_binder_chain == "A"


##########################################
# Direct function tests for boltz helpers
##########################################


def test_merge_colabfold_msa_no_unpaired_raises(tmp_path):
    """merge_colabfold_msa_to_csv should raise if unpaired MSA is None."""
    from uniaf3.adapters.boltz import merge_colabfold_msa_to_csv

    with pytest.raises(ValueError, match="Unpaired MSA file must be provided"):
        merge_colabfold_msa_to_csv(
            unpaired_msa_file=None,
            paired_msa_file=None,
            msa_id="test",
            out_dir=tmp_path,
        )


def test_merge_colabfold_msa_missing_unpaired_raises(tmp_path):
    """merge_colabfold_msa_to_csv should raise if unpaired MSA file doesn't exist."""
    from uniaf3.adapters.boltz import merge_colabfold_msa_to_csv

    with pytest.raises(FileNotFoundError, match="Unpaired MSA file not found"):
        merge_colabfold_msa_to_csv(
            unpaired_msa_file="/nonexistent/path/test.a3m",
            paired_msa_file=None,
            msa_id="test",
            out_dir=tmp_path,
        )


def test_merge_colabfold_msa_missing_paired_raises(tmp_path):
    """merge_colabfold_msa_to_csv should raise if paired MSA file doesn't exist."""
    from uniaf3.adapters.boltz import merge_colabfold_msa_to_csv

    seq_str = "MVLSPADKTNVK"
    unpaired = tmp_path / "test.single.a3m"
    unpaired.write_text(f">query\n{seq_str}\n")

    with pytest.raises(FileNotFoundError, match="Paired MSA file not found"):
        merge_colabfold_msa_to_csv(
            unpaired_msa_file=str(unpaired),
            paired_msa_file="/nonexistent/paired.a3m",
            msa_id="test",
            out_dir=tmp_path,
        )


def test_merge_colabfold_msa_with_paired(tmp_path):
    """merge_colabfold_msa_to_csv with both unpaired and paired MSAs."""
    from uniaf3.adapters.boltz import merge_colabfold_msa_to_csv

    seq_str = "MVLSPADKTNVK"
    unpaired = tmp_path / "test.single.a3m"
    unpaired.write_text(f">query\n{seq_str}\n>hit1\n{seq_str[:-1]}-\n")

    paired = tmp_path / "test.paired.a3m"
    paired.write_text(f">query\n{seq_str}\n>paired1\n{seq_str}\n")

    out_file = merge_colabfold_msa_to_csv(
        unpaired_msa_file=str(unpaired),
        paired_msa_file=str(paired),
        msa_id="test_merge",
        out_dir=tmp_path / "out",
    )
    assert out_file.exists()


def test_split_boltz_csv_to_a3m_with_paired(tmp_path):
    """split_boltz_csv_to_a3m should produce both single and paired A3M files."""
    from uniaf3.adapters.boltz import merge_colabfold_msa_to_csv, split_boltz_csv_to_a3m

    seq_str = "MVLSPADKTNVK"
    unpaired = tmp_path / "test.single.a3m"
    unpaired.write_text(f">query\n{seq_str}\n>hit1\n{seq_str[:-1]}-\n")

    paired = tmp_path / "test.paired.a3m"
    paired.write_text(f">query\n{seq_str}\n>paired1\n{seq_str}\n")

    csv_file = merge_colabfold_msa_to_csv(
        unpaired_msa_file=str(unpaired),
        paired_msa_file=str(paired),
        msa_id="test_split",
        out_dir=tmp_path / "csv_out",
    )

    out_dir = tmp_path / "a3m_out"
    unpaired_path, paired_path = split_boltz_csv_to_a3m(csv_file, out_dir)
    assert unpaired_path.exists()
    assert paired_path is not None
    assert paired_path.exists()


def test_split_boltz_csv_to_a3m_no_paired(tmp_path):
    """split_boltz_csv_to_a3m should produce only unpaired A3M when no paired sequences exist."""
    from uniaf3.adapters.boltz import merge_colabfold_msa_to_csv, split_boltz_csv_to_a3m

    seq_str = "MVLSPADKTNVK"
    # Use unpaired MSA only (no paired MSA file)
    unpaired = tmp_path / "test.single.a3m"
    # query + one hit, then a minimal paired MSA with just query to give key=0
    unpaired.write_text(f">query\n{seq_str}\n>hit1\n{seq_str[:-1]}-\n")

    paired = tmp_path / "test.paired.a3m"
    # Paired with query + one actual paired hit
    paired.write_text(f">query\n{seq_str}\n>paired1\n{seq_str}\n")

    csv_file = merge_colabfold_msa_to_csv(
        unpaired_msa_file=str(unpaired),
        paired_msa_file=str(paired),
        msa_id="test_split_nopair",
        out_dir=tmp_path / "csv_out",
    )

    out_dir = tmp_path / "a3m_out"
    unpaired_path, paired_path = split_boltz_csv_to_a3m(csv_file, out_dir)
    assert unpaired_path.exists()
    # With paired sequences present, paired A3M should also be created
    assert paired_path is not None and paired_path.exists()


def test_from_boltz_dna_rna(tmp_path_factory):
    """from_boltz should handle DNA and RNA sequence entries."""
    from uniaf3.adapters import from_boltz
    from uniaf3.schema.boltz import (
        BoltzConfig,
        BoltzDNA,
        BoltzRNA,
        BoltzSequenceEntry,
    )

    config = BoltzConfig(
        sequences=[
            BoltzSequenceEntry(dna=BoltzDNA(id="A", sequence="ACGT")),
            BoltzSequenceEntry(rna=BoltzRNA(id="B", sequence="ACGU")),
        ]
    )
    with pytest.warns(UserWarning):
        result = from_boltz(config)

    assert len(result.sequences) == 2
    from uniaf3.schema.base import Polymer, PolymerType

    dna = result.sequences[0]
    assert isinstance(dna, Polymer)
    assert dna.polymer_type == PolymerType.DNA
    assert dna.sequence == "ACGT"

    rna = result.sequences[1]
    assert isinstance(rna, Polymer)
    assert rna.polymer_type == PolymerType.RNA
    assert rna.sequence == "ACGU"


def test_from_boltz_template_with_pdb_path(tmp_path_factory):
    """from_boltz should handle BoltzTemplate with pdb (not cif) path."""
    from uniaf3.adapters import from_boltz
    from uniaf3.schema.boltz import (
        BoltzConfig,
        BoltzProtein,
        BoltzSequenceEntry,
        BoltzTemplate,
    )

    config = BoltzConfig(
        sequences=[
            BoltzSequenceEntry(protein=BoltzProtein(id="A", sequence="MVLSPADKTNVK"))
        ],
        templates=[BoltzTemplate(pdb="/some/path/template.pdb", chain_id=["A"])],
    )
    with pytest.warns(UserWarning):
        result = from_boltz(config)

    prot = result.sequences[0]
    assert isinstance(prot, ProteinSeq)
    assert prot.templates is not None
    assert prot.templates[0].path == "/some/path/template.pdb"


def test_from_boltz_csv_msa(tmp_path_factory):
    """from_boltz should split a CSV MSA back into A3M files."""
    from uniaf3.adapters import from_boltz
    from uniaf3.adapters.boltz import merge_colabfold_msa_to_csv
    from uniaf3.schema.boltz import BoltzConfig, BoltzProtein, BoltzSequenceEntry
    from uniaf3.utils import hash_sequence

    seq_str = "MVLSPADKTNVK"
    seq_hash = hash_sequence(seq_str)

    # Create the source A3M files
    msa_src = tmp_path_factory.mktemp("msa_csv_src")
    single = msa_src / f"{seq_hash}.single.a3m"
    single.write_text(f">query\n{seq_str}\n>hit1\n{seq_str[:-1]}-\n")
    paired = msa_src / f"{seq_hash}.pair.a3m"
    paired.write_text(f">query\n{seq_str}\n>pair1\n{seq_str}\n")

    # Merge into CSV
    csv_out_dir = tmp_path_factory.mktemp("msa_csv_out")
    csv_file = merge_colabfold_msa_to_csv(
        unpaired_msa_file=str(single),
        paired_msa_file=str(paired),
        msa_id=seq_hash,
        out_dir=csv_out_dir,
    )

    config = BoltzConfig(
        sequences=[
            BoltzSequenceEntry(
                protein=BoltzProtein(
                    id="A",
                    sequence=seq_str,
                    msa=str(csv_file),
                )
            )
        ]
    )
    msa_result = tmp_path_factory.mktemp("msa_csv_result")
    with pytest.warns(UserWarning):
        result = from_boltz(config, msa_dir=msa_result)

    prot = result.sequences[0]
    assert isinstance(prot, ProteinSeq)
    assert prot.unpaired_msa is not None
    from pathlib import Path as _Path

    assert _Path(prot.unpaired_msa).exists()
    assert prot.paired_msa is not None
    assert _Path(prot.paired_msa).exists()
