from pathlib import Path

from typer.testing import CliRunner

from uniaf3.cli import app
from uniaf3.schema import UniAF3Config
from uniaf3.schema.base import ContactRestraint, Ligand, PocketRestraint, ProteinSeq

FIXTURES = Path(__file__).parent / "fixtures"


def _proteins(config: UniAF3Config) -> list[ProteinSeq]:
    return [seq for seq in config.sequences if isinstance(seq, ProteinSeq)]


def _ligands(config: UniAF3Config) -> list[Ligand]:
    return [seq for seq in config.sequences if isinstance(seq, Ligand)]


def test_from_structure_file_seq_source_controls_full_vs_observed(tmp_path):
    pdb_file = tmp_path / "missing_residues.pdb"
    pdb_file.write_text(
        "\n".join(
            [
                "HEADER    TEST",
                "SEQRES   1 A    4  MET GLY ALA SER",
                "ATOM      1  N   GLY A   2       0.000   0.000   0.000  1.00 20.00           N",
                "ATOM      2  CA  GLY A   2       1.000   0.000   0.000  1.00 20.00           C",
                "ATOM      3  C   GLY A   2       1.000   1.000   0.000  1.00 20.00           C",
                "ATOM      4  N   ALA A   3       2.000   1.000   0.000  1.00 20.00           N",
                "ATOM      5  CA  ALA A   3       2.000   2.000   0.000  1.00 20.00           C",
                "TER",
                "END",
                "",
            ]
        )
    )

    full = UniAF3Config.from_structure_file(pdb_file)
    observed = UniAF3Config.from_structure_file(pdb_file, seq_source="observed")

    assert _proteins(full)[0].sequence == "MGAS"
    assert _proteins(observed)[0].sequence == "GA"


def test_from_structure_file_detects_modification_against_full_sequence(tmp_path):
    pdb_file = tmp_path / "mse_modified.pdb"
    pdb_file.write_text(
        "\n".join(
            [
                "HEADER    TEST",
                "SEQRES   1 A    2  MET GLY",
                "HETATM    1  N   MSE A   1       0.000   0.000   0.000  1.00 20.00           N",
                "HETATM    2  CA  MSE A   1       1.000   0.000   0.000  1.00 20.00           C",
                "HETATM    3  C   MSE A   1       1.000   1.000   0.000  1.00 20.00           C",
                "ATOM      4  N   GLY A   2       2.000   1.000   0.000  1.00 20.00           N",
                "ATOM      5  CA  GLY A   2       2.000   2.000   0.000  1.00 20.00           C",
                "TER",
                "END",
                "",
            ]
        )
    )

    config = UniAF3Config.from_structure_file(pdb_file)
    protein = _proteins(config)[0]

    assert protein.sequence == "MG"
    assert protein.modifications is not None
    assert [(mod.ccd, mod.position) for mod in protein.modifications] == [("MSE", 1)]


def test_from_structure_file_uses_full_sequences_and_groups_entities():
    config = UniAF3Config.from_structure_file(FIXTURES / "1BZ1.cif.gz")

    assert config.aux.name == "1BZ1"

    proteins = _proteins(config)
    assert len(proteins) == 2
    assert proteins[0].id == ["A", "C"]
    assert len(proteins[0].sequence) == 142
    assert proteins[0].sequence.startswith("MVLSPADKTNVK")
    assert proteins[1].id == ["B", "D"]
    assert len(proteins[1].sequence) == 146
    assert proteins[1].sequence.startswith("VHLTPEEKSAVT")

    ligands = _ligands(config)
    assert len(ligands) == 1
    assert ligands[0].id == ["E", "F", "G", "H"]
    assert ligands[0].ccd == ["HEM"]


def test_from_structure_file_chain_filter_requires_ligand_inclusion():
    config = UniAF3Config.from_structure_file(
        FIXTURES / "1BZ1.cif.gz",
        chains={"A"},
    )

    proteins = _proteins(config)
    assert len(proteins) == 1
    assert proteins[0].id == "A"
    assert not _ligands(config)

    with_ligands = UniAF3Config.from_structure_file(
        FIXTURES / "1BZ1.cif.gz",
        chains={"A"},
        include_ligands=True,
    )

    ligands = _ligands(with_ligands)
    assert len(ligands) == 1
    assert ligands[0].id == "E"
    assert ligands[0].ccd == ["HEM"]


def test_from_structure_file_chain_filter_drops_connections_without_warnings(recwarn):
    UniAF3Config.from_structure_file(
        FIXTURES / "1BZ1.cif.gz",
        chains={"A"},
    )

    assert not recwarn


def test_from_structure_file_imports_non_covalent_connections_as_contacts():
    config = UniAF3Config.from_structure_file(
        FIXTURES / "1BZ1.cif.gz",
        non_covalent_connections="contacts",
    )

    assert config.contact_restraints is not None
    assert len(config.contact_restraints) == 4
    assert all(isinstance(r, ContactRestraint) for r in config.contact_restraints)

    first = config.contact_restraints[0]
    assert first.token1.chain_id == "A"
    assert first.token1.residue_idx == 88
    assert first.token1.atom_name == "NE2"
    assert first.token1.residue_name == "H"
    assert first.token2.chain_id == "E"
    assert first.token2.residue_idx == 1
    assert first.token2.atom_name == "FE"
    assert first.token2.residue_name == "HEM"


def test_from_structure_file_imports_non_covalent_connections_as_pockets():
    config = UniAF3Config.from_structure_file(
        FIXTURES / "1BZ1.cif.gz",
        non_covalent_connections="pockets",
    )

    assert config.pocket_restraints is not None
    assert len(config.pocket_restraints) == 4
    assert all(isinstance(r, PocketRestraint) for r in config.pocket_restraints)
    assert [r.binder_chain for r in config.pocket_restraints] == ["E", "F", "G", "H"]
    assert config.pocket_restraints[0].contact_tokens[0].chain_id == "A"
    assert config.pocket_restraints[0].contact_tokens[0].residue_idx == 88


def test_structure_cli_writes_uniaf3_config(tmp_path):
    result = CliRunner().invoke(
        app,
        [
            "structure",
            str(FIXTURES / "1BZ1.cif.gz"),
            str(tmp_path),
            "hemoglobin_a",
            "--chains",
            "A",
            "--include-ligands",
            "--seq-source",
            "observed",
        ],
    )

    assert result.exit_code == 0, result.output
    config = UniAF3Config.from_file(tmp_path / "hemoglobin_a.yaml")
    assert [seq.id for seq in config.sequences] == ["A", "E"]
