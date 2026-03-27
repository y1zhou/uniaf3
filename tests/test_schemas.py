"""Tests for model-specific schema validation."""

from pathlib import Path

import orjson
import pytest
import yaml

from uniaf3.schema import (
    AF3Config,
    AF3ServerConfig,
    BoltzConfig,
    ChaiConfig,
    ProtenixConfig,
    UniAF3Config,
)
from uniaf3.schema.chai import ChaiEntityType


# ruff: noqa: S101
# ============================================================
# AlphaFold3 schema
# ============================================================
class TestAF3Schema:
    """Validate AF3Config against example input."""

    def test_load_example(self, af3_conf: AF3Config):
        assert af3_conf.name == "Hello fold"
        assert af3_conf.modelSeeds == [10, 42]
        assert len(af3_conf.sequences) == 7
        assert af3_conf.dialect == "alphafold3"
        assert af3_conf.version == 4

    def test_protein_entry(self, af3_conf: AF3Config):
        prot = af3_conf.sequences[0].protein
        assert prot is not None
        assert prot.id == "A"
        assert prot.sequence == "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLS"
        assert prot.modifications is not None
        assert len(prot.modifications) == 2
        assert prot.modifications[0].ptmType == "HY3"
        assert prot.modifications[0].ptmPosition == 1

    def test_dna_entry(self, af3_conf: AF3Config):
        dna = af3_conf.sequences[2].dna
        assert dna is not None
        assert dna.sequence == "GACCTCT"
        assert dna.modifications is not None
        assert len(dna.modifications) == 2

    def test_rna_entry(self, af3_conf: AF3Config):
        rna = af3_conf.sequences[3].rna
        assert rna is not None
        assert rna.sequence == "AGCU"

    def test_ligand_with_ccd(self, af3_conf: AF3Config):
        lig = af3_conf.sequences[4].ligand
        assert lig is not None
        assert lig.id == ["F", "G", "H"]
        assert lig.ccdCodes == ["ATP"]

    def test_ligand_with_smiles(self, af3_conf: AF3Config):
        lig = af3_conf.sequences[6].ligand
        assert lig is not None
        assert lig.smiles == "CC(=O)OC1C[NH+]2CCC1CC2"

    def test_bonded_atom_pairs(self, af3_conf: AF3Config):
        assert af3_conf.bondedAtomPairs is not None
        assert len(af3_conf.bondedAtomPairs) == 2
        a1, a2 = af3_conf.bondedAtomPairs[0]
        assert a1 == ("A", 1, "CA")
        assert a2 == ("F", 1, "CHA")

    def test_json_str_property(self, af3_conf: AF3Config):
        j = af3_conf.to_str()
        assert isinstance(j, str)
        parsed = orjson.loads(j)
        assert parsed["name"] == "Hello fold"

    def test_invalid_ligand_both_ccd_smiles(self):
        from uniaf3.schema.alphafold3 import AF3Ligand

        with pytest.raises(ValueError):
            AF3Ligand(id="X", ccdCodes=["ATP"], smiles="CC(=O)OC1C[NH+]2CCC1CC2")

    def test_invalid_ligand_neither_ccd_smiles(self):
        from uniaf3.schema.alphafold3 import AF3Ligand

        with pytest.raises(ValueError):
            AF3Ligand(id="X")

    def test_sequence_entry_exactly_one(self):
        from uniaf3.schema.alphafold3 import (
            AF3DNA,
            AF3Protein,
            AF3SequenceEntry,
        )

        # Zero entities should fail
        with pytest.raises(ValueError):
            AF3SequenceEntry()

        # Two entities set should fail
        with pytest.raises(ValueError):
            AF3SequenceEntry(
                protein=AF3Protein(id="A", sequence="M"),
                dna=AF3DNA(id="B", sequence="G"),
            )


# ============================================================
# AlphaFold3 Server schema
# ============================================================
class TestAF3ServerSchema:
    """Validate AF3ServerConfig against example input."""

    def test_load_data(self, af3_server_confs: AF3ServerConfig):
        assert len(af3_server_confs) == 2
        conf = af3_server_confs[0]
        assert conf.name == "Test Fold Job"
        assert len(conf.sequences) == 9
        assert conf.version == 1

    def test_protein_entry(self, af3_server_confs: AF3ServerConfig):
        prot = af3_server_confs[0].sequences[0].proteinChain
        assert prot is not None
        assert prot.sequence == "PREACHINGS"

    def test_ion_entry(self, af3_server_confs: AF3ServerConfig):
        ion = af3_server_confs[0].sequences[7].ion
        assert ion is not None
        assert ion.ion == "MG"

    def test_to_str(self, af3_server_confs: AF3ServerConfig):
        s = af3_server_confs.to_str()
        assert isinstance(s, str)
        parsed = orjson.loads(s)
        assert parsed[0]["name"] == "Test Fold Job"
        assert parsed[1]["name"] == "Test Fold Job Number Two"


# ============================================================
# Boltz schema
# ============================================================
class TestBoltzSchema:
    """Validate BoltzConfig against example input."""

    def test_load_data(self, boltz_conf: BoltzConfig):
        assert boltz_conf.version == 1
        assert len(boltz_conf.sequences) == 3

    def test_protein_entry(self, boltz_conf: BoltzConfig):
        prot = boltz_conf.sequences[0].protein
        assert prot is not None
        assert prot.id == ["A", "B"]
        assert prot.msa == str(
            Path(__file__).parent / "fixtures" / "dummy_msa" / "a3ms" / "boltz_A.a3m"
        )

    def test_ligand_ccd(self, boltz_conf: BoltzConfig):
        lig = boltz_conf.sequences[1].ligand
        assert lig is not None
        assert lig.ccd == "SAH"

    def test_ligand_smiles(self, boltz_conf: BoltzConfig):
        lig = boltz_conf.sequences[2].ligand
        assert lig is not None
        assert lig.smiles is not None

    def test_bond_constraint(self, boltz_conf: BoltzConfig):
        assert boltz_conf.constraints is not None
        assert len(boltz_conf.constraints) == 3
        b = boltz_conf.constraints[0].bond
        assert b is not None
        assert b.atom1 == ("A", 111, "SG")

    def test_yaml_str_property(self, boltz_conf: BoltzConfig):
        y = boltz_conf.to_str()
        assert isinstance(y, str)
        parsed = yaml.safe_load(y)
        assert parsed["version"] == 1


# ============================================================
# Chai schema
# ============================================================
class TestChaiSchema:
    """Validate ChaiConfig against example input."""

    def test_load_data(self, chai_conf: ChaiConfig):
        assert len(chai_conf.entities) == 6
        assert chai_conf.restraints is not None
        assert len(chai_conf.restraints) == 3

    def test_protein_entry(self, chai_conf: ChaiConfig):
        prot = chai_conf.entities[0]
        assert prot.entity_type == "protein"
        assert prot.entity_name == "Hemoglobin subunit"
        assert prot.sequence == "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLS"

    def test_protein_with_modification(self, chai_conf: ChaiConfig):
        prot = chai_conf.entities[1]
        assert prot.entity_type == "protein"
        assert prot.sequence.startswith("(HY3)")

    def test_dna_entry(self, chai_conf: ChaiConfig):
        dna = chai_conf.entities[2]
        assert dna.entity_type == "dna"
        assert dna.sequence == "GATTACA"

    def test_ligand_ccd(self, chai_conf: ChaiConfig):
        lig = chai_conf.entities[3]
        assert lig.entity_type == "ligand"
        assert lig.entity_name == "CCD example"
        assert lig.sequence == "ATP"

    def test_ligand_smiles(self, chai_conf: ChaiConfig):
        lig = chai_conf.entities[4]
        assert lig.entity_type == "ligand"
        assert lig.sequence == "CC(=O)OC1C[NH+]2CCC1CC2"

    def test_glycan_entry(self, chai_conf: ChaiConfig):
        glycan = chai_conf.entities[5]
        assert glycan.entity_type == "glycan"
        assert glycan.sequence == "NAG(NAG)(BMA)"

    def test_covalent_restraint(self, chai_conf: ChaiConfig):
        assert chai_conf.restraints is not None
        cov = [r for r in chai_conf.restraints if r.connection_type == "covalent"]
        assert len(cov) == 1
        r = cov[0]
        assert r.chainA == "A"
        assert r.res_idxA == "P5@CG"
        assert r.chainB == "F"
        assert r.res_idxB == "@C1"

    def test_contact_restraint(self, chai_conf: ChaiConfig):
        assert chai_conf.restraints is not None
        ct = [r for r in chai_conf.restraints if r.connection_type == "contact"]
        assert len(ct) == 1
        r = ct[0]
        assert r.chainA == "A"
        assert r.res_idxA == "V11"
        assert r.chainB == "B"
        assert r.res_idxB == "L35"
        assert r.max_distance_angstrom == 6.0

    def test_pocket_restraint(self, chai_conf: ChaiConfig):
        assert chai_conf.restraints is not None
        pk = [r for r in chai_conf.restraints if r.connection_type == "pocket"]
        assert len(pk) == 1
        r = pk[0]
        assert r.chainA == "B"
        assert r.res_idxA is None  # binder side has no residue
        assert r.chainB == "A"
        assert r.res_idxB == "A14"
        assert r.max_distance_angstrom == 8.0

    def test_fasta_output(self, chai_conf: ChaiConfig):
        fasta = chai_conf.entities_to_fasta()
        assert isinstance(fasta, str)
        lines = fasta.strip().split("\n")
        # 6 entities → 12 lines (header + sequence each)
        assert len(lines) == 12
        assert lines[0] == ">protein|Hemoglobin subunit"

    def test_restraints_to_df(self, chai_conf: ChaiConfig):
        import polars as pl

        df = pl.DataFrame(chai_conf.restraints)
        assert df is not None
        assert len(df) == 3
        assert "connection_type" in df.columns

    def test_entity_names_must_be_unique(self):
        from uniaf3.schema.chai import ChaiEntity

        with pytest.raises(ValueError, match="unique"):
            ChaiConfig(
                entities=[
                    ChaiEntity(
                        entity_type=ChaiEntityType.Protein,
                        entity_name="dup",
                        sequence="MVLS",
                    ),
                    ChaiEntity(
                        entity_type=ChaiEntityType.Protein,
                        entity_name="dup",
                        sequence="MVLS",
                    ),
                ]
            )


# ============================================================
# Protenix schema
# ============================================================
class TestProtenixSchema:
    """Validate ProtenixConfig against example input."""

    def test_load_data(self, protenix_confs: ProtenixConfig):
        assert len(protenix_confs) == 1
        conf = protenix_confs[0]
        assert conf.name == "Test Fold Job"
        assert len(conf.sequences) == 6

    def test_protein_chain(self, protenix_confs: ProtenixConfig):
        pc = protenix_confs[0].sequences[0].proteinChain
        assert pc is not None
        assert pc.sequence == "PREACHINGS"
        assert pc.count == 2
        assert pc.modifications is not None
        assert len(pc.modifications) == 2
        assert pc.modifications[0].ptmType == "CCD_HY3"

    def test_dna_sequence(self, protenix_confs: ProtenixConfig):
        ds = protenix_confs[0].sequences[1].dnaSequence
        assert ds is not None
        assert ds.sequence == "GATTACA"

    def test_ligand(self, protenix_confs: ProtenixConfig):
        lig = protenix_confs[0].sequences[3].ligand
        assert lig is not None
        assert lig.ligand == "CCD_ATP"

    def test_ion(self, protenix_confs: ProtenixConfig):
        ion = protenix_confs[0].sequences[4].ion
        assert ion is not None
        assert ion.ion == "MG"
        assert ion.count == 2

    def test_covalent_bonds(self, protenix_confs: ProtenixConfig):
        bonds = protenix_confs[0].covalent_bonds
        assert bonds is not None
        assert len(bonds) == 1
        assert bonds[0].entity1 == 1
        assert bonds[0].atom1 == "N6"

    def test_json_str_property(self, protenix_confs: ProtenixConfig):
        j = protenix_confs.to_str()
        assert isinstance(j, str)
        parsed = orjson.loads(j)
        assert len(parsed) == 1
        assert parsed[0]["name"] == "Test Fold Job"


# ============================================================
# UniAF3 schema
# ============================================================
class TestUniAF3Schema:
    """Validate UniAF3Config against example input."""

    def test_load_data(self, uniaf3_conf: UniAF3Config):
        assert len(uniaf3_conf.sequences) == 5
        assert uniaf3_conf.aux.seeds == [42, 123]

    def test_yaml_str(self, uniaf3_conf: UniAF3Config):
        y = uniaf3_conf.to_yaml()
        assert isinstance(y, str)
        parsed = yaml.safe_load(y)
        assert "sequences" in parsed

    def test_json_str(self, uniaf3_conf: UniAF3Config):
        j = uniaf3_conf.to_json()
        assert isinstance(j, str)

    def test_hash(self, uniaf3_conf: UniAF3Config):
        h = uniaf3_conf.hash
        assert isinstance(h, str)
        assert len(h) == 64  # sha256 hex digest


# ============================================================
# ChaiConfig schema - to_files, from_file, from_yaml, validations
# ============================================================
class TestChaiSchemaIO:
    """Test ChaiConfig file I/O methods."""

    def test_from_yaml(self, tmp_path):
        """ChaiConfig.from_yaml should load from a YAML file."""
        from uniaf3.schema.chai import ChaiEntity, ChaiEntityType

        config = ChaiConfig(
            entities=[
                ChaiEntity(
                    entity_type=ChaiEntityType.Protein,
                    entity_name="A",
                    sequence="MVLSPADKTNVK",
                )
            ],
            seed=42,
        )
        yaml_file = tmp_path / "chai.yaml"
        yaml_file.write_text(config.to_yaml())
        loaded = ChaiConfig.from_yaml(yaml_file)
        assert len(loaded.entities) == 1
        assert loaded.seed == 42

    def test_to_files(self, chai_conf: ChaiConfig, tmp_path):
        """ChaiConfig.to_files should write YAML and FASTA files."""
        chai_conf.to_files(tmp_path, "test_chai")
        yaml_path = tmp_path / "test_chai.yaml"
        fasta_path = tmp_path / "test_chai.fasta"
        restraints_path = tmp_path / "test_chai.restraints"
        assert yaml_path.exists()
        assert fasta_path.exists()
        assert restraints_path.exists()

    def test_from_file_yaml(self, tmp_path):
        """ChaiConfig.from_file should load from YAML."""
        from uniaf3.schema.chai import ChaiEntity, ChaiEntityType

        config = ChaiConfig(
            entities=[
                ChaiEntity(
                    entity_type=ChaiEntityType.Protein,
                    entity_name="A",
                    sequence="MVLSPADKTNVK",
                )
            ],
            seed=1,
        )
        yaml_file = tmp_path / "chai.yaml"
        yaml_file.write_text(config.to_yaml())
        loaded = ChaiConfig.from_file(yaml_file)
        assert len(loaded.entities) == 1

    def test_from_file_fasta(self, tmp_path):
        """ChaiConfig.from_file should load from FASTA file."""
        fasta_content = ">protein|A\nMVLSPADKTNVK\n>ligand|B\nCCO\n"
        fasta_file = tmp_path / "test.fasta"
        fasta_file.write_text(fasta_content)
        loaded = ChaiConfig.from_file(fasta_file)
        assert len(loaded.entities) == 2

    def test_from_file_fasta_with_msa_dir(self, tmp_path):
        """ChaiConfig.from_file should discover msa directory when present."""
        fasta_content = ">protein|A\nMVLSPADKTNVK\n"
        fasta_file = tmp_path / "test.fasta"
        fasta_file.write_text(fasta_content)
        msa_dir = tmp_path / "msa"
        msa_dir.mkdir()
        loaded = ChaiConfig.from_file(fasta_file)
        assert loaded.msa_directory == str(msa_dir)

    def test_from_file_fasta_with_template(self, tmp_path):
        """ChaiConfig.from_file should discover template m8 file when present."""
        fasta_content = ">protein|A\nMVLSPADKTNVK\n"
        fasta_file = tmp_path / "test.fasta"
        fasta_file.write_text(fasta_content)
        msa_dir = tmp_path / "msa"
        msa_dir.mkdir()
        m8_file = msa_dir / "all_chain_templates.m8"
        m8_file.write_text(
            "hash\t1abc_A\t95.0\t12\t0\t0\t1\t12\t1\t12\t1e-5\t50.0\t12M\n"
        )
        loaded = ChaiConfig.from_file(fasta_file)
        assert loaded.template_hits_path == str(m8_file)

    def test_from_file_unsupported_extension_raises(self, tmp_path):
        """ChaiConfig.from_file should raise for unsupported extensions."""
        bad_file = tmp_path / "test.json"
        bad_file.write_text("{}")
        with pytest.raises(ValueError, match="Unsupported config file format"):
            ChaiConfig.from_file(bad_file)

    def test_from_chai_files_with_msa_dir(self, tmp_path):
        """ChaiConfig.from_chai_files should accept msa_directory parameter."""
        fasta_content = ">protein|A\nMVLSPADKTNVK\n"
        fasta_file = tmp_path / "test.fasta"
        fasta_file.write_text(fasta_content)
        msa_dir = tmp_path / "msa"
        msa_dir.mkdir()
        loaded = ChaiConfig.from_chai_files(fasta_file, msa_directory=msa_dir)
        assert loaded.msa_directory == str(msa_dir)

    def test_from_chai_files_template_wrong_extension_raises(self, tmp_path):
        """from_chai_files should raise if template_hits_path has wrong extension."""
        fasta_content = ">protein|A\nMVLSPADKTNVK\n"
        fasta_file = tmp_path / "test.fasta"
        fasta_file.write_text(fasta_content)
        with pytest.raises(ValueError, match="must be in .m8 format"):
            ChaiConfig.from_chai_files(
                fasta_file, template_hits_path=tmp_path / "templates.txt"
            )


# ============================================================
# ChaiConfig restraint validation
# ============================================================
class TestChaiRestraintValidation:
    """Test ChaiConfig restraint validation errors."""

    def _make_protein_config(self, restraints):
        from uniaf3.schema.chai import ChaiEntity, ChaiEntityType

        return ChaiConfig(
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
            restraints=restraints,
        )

    def test_covalent_missing_res_idx(self):
        """Covalent restraint with None res_idxA should raise."""
        from uniaf3.schema.chai import ChaiRestraint, ChaiRestraintType

        with pytest.raises(Exception, match="res_idx cannot be empty"):
            self._make_protein_config(
                [
                    ChaiRestraint(
                        restraint_id="r0",
                        chainA="A",
                        res_idxA=None,
                        chainB="B",
                        res_idxB="G1@CA",
                        connection_type=ChaiRestraintType.Covalent,
                        max_distance_angstrom=0.0,
                    )
                ]
            )

    def test_covalent_invalid_format(self):
        """Covalent restraint with no '@' in res_idx should raise."""
        from uniaf3.schema.chai import ChaiRestraint, ChaiRestraintType

        with pytest.raises(Exception, match="Invalid residue index format"):
            self._make_protein_config(
                [
                    ChaiRestraint(
                        restraint_id="r0",
                        chainA="A",
                        res_idxA="NOATSIGN",  # Missing '@'
                        chainB="B",
                        res_idxB="G1@CA",
                        connection_type=ChaiRestraintType.Covalent,
                        max_distance_angstrom=0.0,
                    )
                ]
            )

    def test_covalent_residue_name_mismatch(self):
        """Covalent restraint with wrong residue name should raise."""
        from uniaf3.schema.chai import ChaiRestraint, ChaiRestraintType

        with pytest.raises(
            Exception, match="Residue name in index does not match sequence"
        ):
            self._make_protein_config(
                [
                    ChaiRestraint(
                        restraint_id="r0",
                        chainA="A",
                        res_idxA="X1@CA",  # X != M at position 1
                        chainB="B",
                        res_idxB="G1@CA",
                        connection_type=ChaiRestraintType.Covalent,
                        max_distance_angstrom=0.0,
                    )
                ]
            )

    def test_covalent_empty_atom_name(self):
        """Covalent restraint with empty atom name should raise."""
        from uniaf3.schema.chai import ChaiRestraint, ChaiRestraintType

        with pytest.raises(Exception, match="Atom name must be specified"):
            self._make_protein_config(
                [
                    ChaiRestraint(
                        restraint_id="r0",
                        chainA="A",
                        res_idxA="M1@",  # Empty atom name
                        chainB="B",
                        res_idxB="G1@CA",
                        connection_type=ChaiRestraintType.Covalent,
                        max_distance_angstrom=0.0,
                    )
                ]
            )

    def test_contact_on_ligand_raises(self):
        """Contact restraint on a ligand entity should raise."""
        from uniaf3.schema.chai import (
            ChaiEntity,
            ChaiEntityType,
            ChaiRestraint,
            ChaiRestraintType,
        )

        with pytest.raises(Exception, match="only supported for protein/DNA/RNA"):
            ChaiConfig(
                entities=[
                    ChaiEntity(
                        entity_type=ChaiEntityType.Protein,
                        entity_name="A",
                        sequence="MVLSPADKTNVK",
                    ),
                    ChaiEntity(
                        entity_type=ChaiEntityType.Ligand,
                        entity_name="B",
                        sequence="CCO",
                    ),
                ],
                seed=1,
                restraints=[
                    ChaiRestraint(
                        restraint_id="r0",
                        chainA="A",
                        res_idxA="M1",
                        chainB="B",
                        res_idxB="C1",  # ligand - not valid for contact
                        connection_type=ChaiRestraintType.Contact,
                        max_distance_angstrom=6.0,
                    )
                ],
            )

    def test_contact_residue_name_mismatch(self):
        """Contact restraint with wrong residue name should raise."""
        from uniaf3.schema.chai import ChaiRestraint, ChaiRestraintType

        with pytest.raises(
            Exception, match="Residue name in index does not match sequence"
        ):
            self._make_protein_config(
                [
                    ChaiRestraint(
                        restraint_id="r0",
                        chainA="A",
                        res_idxA="X1",  # X != M at position 1
                        chainB="B",
                        res_idxB="G1",
                        connection_type=ChaiRestraintType.Contact,
                        max_distance_angstrom=6.0,
                    )
                ]
            )

    def test_pocket_residue_name_mismatch(self):
        """Pocket restraint with wrong residue name in contact token should raise."""
        from uniaf3.schema.chai import ChaiRestraint, ChaiRestraintType

        with pytest.raises(
            Exception, match="Residue name in index does not match sequence"
        ):
            self._make_protein_config(
                [
                    ChaiRestraint(
                        restraint_id="r0",
                        chainA="A",
                        res_idxA=None,  # binder side is empty
                        chainB="B",
                        res_idxB="X1",  # X != G at position 1
                        connection_type=ChaiRestraintType.Pocket,
                        max_distance_angstrom=8.0,
                    )
                ]
            )

    def test_from_file_fasta_with_restraints_csv(self, tmp_path):
        """from_file should discover .restraints file."""
        fasta_content = ">protein|A\nMVLSPADKTNVK\n>protein|B\nGKVGAHAG\n"
        fasta_file = tmp_path / "test.fasta"
        fasta_file.write_text(fasta_content)

        restraints_content = "restraint_id,chainA,res_idxA,chainB,res_idxB,connection_type,max_distance_angstrom,min_distance_angstrom,confidence,comment\nr0,A,M1,B,G1,contact,6.0,0.0,1.0,\n"
        restraints_file = tmp_path / "test.restraints"
        restraints_file.write_text(restraints_content)

        loaded = ChaiConfig.from_file(fasta_file)
        assert loaded.restraints is not None
        assert len(loaded.restraints) == 1


# ============================================================
# base.py schema validators
# ============================================================
class TestBaseSchemaValidators:
    """Test schema validators in base.py."""

    def test_uniaf3_base_config_to_str_raises(self):
        """UniAF3BaseConfig.to_str should raise NotImplementedError."""
        from uniaf3.schema.base import UniAF3BaseConfig

        class ConcreteConfig(UniAF3BaseConfig):
            pass

        config = ConcreteConfig()
        with pytest.raises(NotImplementedError, match="to_str method"):
            config.to_str()

    def test_uniaf3_base_config_to_files_raises(self, tmp_path):
        """UniAF3BaseConfig.to_files should raise NotImplementedError."""
        from uniaf3.schema.base import UniAF3BaseConfig

        class ConcreteConfig(UniAF3BaseConfig):
            pass

        config = ConcreteConfig()
        with pytest.raises(NotImplementedError, match="to_files method"):
            config.to_files(tmp_path, "test")

    def test_from_file_not_found_raises(self, tmp_path):
        """UniAF3Config.from_file should raise FileNotFoundError for missing files."""
        with pytest.raises(FileNotFoundError):
            UniAF3Config.from_file(tmp_path / "nonexistent.yaml")

    def test_from_file_unsupported_extension_raises(self, tmp_path):
        """UniAF3Config.from_file should raise ValueError for unsupported extensions."""
        bad_file = tmp_path / "config.toml"
        bad_file.write_text("[config]")
        with pytest.raises(ValueError, match="Unsupported config file format"):
            UniAF3Config.from_file(bad_file)

    def test_modification_out_of_range_raises(self):
        """Polymer modifications out of sequence range should raise."""
        from uniaf3.schema.base import PolymerType, SequenceModification

        with pytest.raises(Exception, match="out of range"):
            from uniaf3.schema.base import Polymer

            Polymer(
                polymer_type=PolymerType.Protein,
                id="A",
                sequence="MVLSP",
                modifications=[SequenceModification(ccd="HY3", position=10)],  # > len
            )

    def test_contact_restraint_check_distance_range(self):
        """ContactRestraint should raise when max_distance <= min_distance."""
        from uniaf3.schema.base import Atom, ContactRestraint

        with pytest.raises(Exception, match="must be greater than min_distance"):
            ContactRestraint(
                token1=Atom(
                    chain_id="A", residue_idx=1, atom_name=None, residue_name="M"
                ),
                token2=Atom(
                    chain_id="B", residue_idx=1, atom_name=None, residue_name="G"
                ),
                max_distance=3.0,
                min_distance=5.0,  # > max_distance
            )

    def test_pocket_restraint_check_distance_range(self):
        """PocketRestraint should raise when max_distance <= min_distance."""
        from uniaf3.schema.base import Atom, PocketRestraint

        with pytest.raises(Exception, match="must be greater than min_distance"):
            PocketRestraint(
                binder_chain="A",
                contact_tokens=[
                    Atom(chain_id="B", residue_idx=1, atom_name=None, residue_name="G")
                ],
                max_distance=3.0,
                min_distance=5.0,  # > max_distance
            )

    def test_structural_template_mismatched_indices_raises(self):
        """StructuralTemplate should raise when query_idx and template_idx differ in length."""
        from uniaf3.schema.base import StructuralTemplate

        with pytest.raises(Exception, match="same length"):
            StructuralTemplate(
                path="/some/path.cif",
                query_idx=[1, 2, 3],
                template_idx=[1, 2],  # shorter
            )

    def test_uniaf3_config_empty_sequences_raises(self):
        """UniAF3Config should raise if sequences is empty."""
        with pytest.raises(Exception, match="At least one sequence must be provided"):
            UniAF3Config(sequences=[])

    def test_uniaf3_config_atom_out_of_range_raises(self):
        """UniAF3Config should raise if restraint atom index is out of sequence range."""
        from uniaf3.schema.base import Atom, ContactRestraint, PolymerType, ProteinSeq

        with pytest.raises(Exception, match="out of range"):
            UniAF3Config(
                sequences=[
                    ProteinSeq(
                        polymer_type=PolymerType.Protein,
                        id="A",
                        sequence="MVLSP",  # 5 residues
                    )
                ],
                contact_restraints=[
                    ContactRestraint(
                        token1=Atom(
                            chain_id="A",
                            residue_idx=10,
                            atom_name=None,
                            residue_name=None,
                        ),  # > 5
                        token2=Atom(
                            chain_id="A",
                            residue_idx=1,
                            atom_name=None,
                            residue_name=None,
                        ),
                        max_distance=6.0,
                    )
                ],
            )

    def test_uniaf3_config_atom_unknown_chain_raises(self):
        """UniAF3Config should raise if restraint atom chain is not in sequences."""
        from uniaf3.schema.base import Atom, ContactRestraint, PolymerType, ProteinSeq

        with pytest.raises(Exception, match="not found in sequences"):
            UniAF3Config(
                sequences=[
                    ProteinSeq(
                        polymer_type=PolymerType.Protein,
                        id="A",
                        sequence="MVLSP",
                    )
                ],
                contact_restraints=[
                    ContactRestraint(
                        token1=Atom(
                            chain_id="Z",
                            residue_idx=1,
                            atom_name=None,
                            residue_name=None,
                        ),  # Z not in sequences
                        token2=Atom(
                            chain_id="A",
                            residue_idx=1,
                            atom_name=None,
                            residue_name=None,
                        ),
                        max_distance=6.0,
                    )
                ],
            )


# ============================================================
# BoltzConfig schema validators
# ============================================================
class TestBoltzSchemaValidators:
    """Test BoltzConfig constraint validators."""

    def _protein_lig_config(self, constraints):
        from uniaf3.schema.boltz import (
            BoltzConfig,
            BoltzLigand,
            BoltzProtein,
            BoltzSequenceEntry,
        )

        return BoltzConfig(
            sequences=[
                BoltzSequenceEntry(
                    protein=BoltzProtein(id="A", sequence="MVLSPADKTNVK")
                ),
                BoltzSequenceEntry(ligand=BoltzLigand(id="B", ccd="SAH")),
            ],
            constraints=constraints,
        )

    def test_bond_chain_not_found_raises(self):
        """Bond constraint with unknown chain should raise."""
        from uniaf3.schema.boltz import BoltzBondConstraint, BoltzConstraintEntry

        with pytest.raises(Exception, match="not found in sequences"):
            self._protein_lig_config(
                [
                    BoltzConstraintEntry(
                        bond=BoltzBondConstraint(
                            atom1=("Z", 1, "CA"),  # Z not in sequences
                            atom2=("A", 1, "CA"),
                        )
                    )
                ]
            )

    def test_bond_residue_out_of_range_raises(self):
        """Bond constraint with residue out of range should raise."""
        from uniaf3.schema.boltz import BoltzBondConstraint, BoltzConstraintEntry

        with pytest.raises(Exception, match="out of range"):
            self._protein_lig_config(
                [
                    BoltzConstraintEntry(
                        bond=BoltzBondConstraint(
                            atom1=("A", 100, "CA"),  # > len(MVLSPADKTNVK)=12
                            atom2=("A", 1, "CA"),
                        )
                    )
                ]
            )

    def test_bond_ligand_wrong_residue_index_raises(self):
        """Bond constraint on ligand with residue_idx != 1 should raise."""
        from uniaf3.schema.boltz import BoltzBondConstraint, BoltzConstraintEntry

        with pytest.raises(
            Exception, match="Residue index for ligand bond constraint must be 1"
        ):
            self._protein_lig_config(
                [
                    BoltzConstraintEntry(
                        bond=BoltzBondConstraint(
                            atom1=("A", 1, "CA"),
                            atom2=("B", 2, "SD"),  # B is ligand, residue_idx != 1
                        )
                    )
                ]
            )

    def test_bond_invalid_atom_name_raises(self):
        """Bond constraint with invalid atom name should raise."""
        from uniaf3.schema.boltz import BoltzBondConstraint, BoltzConstraintEntry

        with pytest.raises(Exception, match="Invalid atom name"):
            self._protein_lig_config(
                [
                    BoltzConstraintEntry(
                        bond=BoltzBondConstraint(
                            atom1=("A", 1, "INVALID_ATOM"),  # not a valid atom for Met
                            atom2=("A", 2, "CA"),
                        )
                    )
                ]
            )

    def test_pocket_binder_not_found_raises(self):
        """Pocket constraint with unknown binder should raise."""
        from uniaf3.schema.boltz import BoltzConstraintEntry, BoltzPocketConstraint

        with pytest.raises(Exception, match="Binder chain ID .* not found"):
            self._protein_lig_config(
                [
                    BoltzConstraintEntry(
                        pocket=BoltzPocketConstraint(
                            binder="Z",  # Z not in sequences
                            contacts=[("A", 1)],
                            max_distance=6.0,
                        )
                    )
                ]
            )

    def test_rna_entity_in_constraints(self):
        """RNA entity should be handled in check_constraints."""
        from uniaf3.schema.boltz import (
            BoltzConfig,
            BoltzConstraintEntry,
            BoltzContactConstraint,
            BoltzRNA,
            BoltzSequenceEntry,
        )

        # Just verifying RNA sequence in constraints works
        config = BoltzConfig(
            sequences=[
                BoltzSequenceEntry(rna=BoltzRNA(id="A", sequence="ACGU")),
            ],
            constraints=[
                BoltzConstraintEntry(
                    contact=BoltzContactConstraint(
                        token1=("A", 1),
                        token2=("A", 2),
                        max_distance=6.0,
                    )
                )
            ],
        )
        assert config is not None

    def test_protein_eq(self):
        """BoltzProtein __eq__ should work correctly."""
        from uniaf3.schema.boltz import BoltzProtein

        p1 = BoltzProtein(id="A", sequence="MVLS")
        p2 = BoltzProtein(id="A", sequence="MVLS")
        p3 = BoltzProtein(id="B", sequence="MVLS")
        assert p1 == p2
        assert p1 != p3
        assert p1 != "not_a_protein"


class TestChaiRestraintValidationMore:
    """Additional chai restraint validation error paths."""

    def _make_protein_config(self, restraints):
        from uniaf3.schema.chai import ChaiEntity, ChaiEntityType

        return ChaiConfig(
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
            restraints=restraints,
        )

    def test_covalent_parse_residue_failure(self):
        """Covalent restraint with unparseable residue position should raise."""
        from uniaf3.schema.chai import ChaiRestraint, ChaiRestraintType

        with pytest.raises(Exception, match="Failed to parse residue index"):
            self._make_protein_config(
                [
                    ChaiRestraint(
                        restraint_id="r0",
                        chainA="A",
                        res_idxA="MXYZ@CA",  # 'XYZ' can't be parsed as int for position
                        chainB="B",
                        res_idxB="G1@CA",
                        connection_type=ChaiRestraintType.Covalent,
                        max_distance_angstrom=0.0,
                    )
                ]
            )

    def test_contact_missing_res_idx(self):
        """Contact restraint with None res_idx should raise."""
        from uniaf3.schema.chai import ChaiRestraint, ChaiRestraintType

        with pytest.raises(Exception, match="res_idx cannot be empty"):
            self._make_protein_config(
                [
                    ChaiRestraint(
                        restraint_id="r0",
                        chainA="A",
                        res_idxA=None,  # None for contact is not allowed
                        chainB="B",
                        res_idxB="G1",
                        connection_type=ChaiRestraintType.Contact,
                        max_distance_angstrom=6.0,
                    )
                ]
            )

    def test_contact_parse_residue_failure(self):
        """Contact restraint with unparseable residue should raise."""
        from uniaf3.schema.chai import ChaiRestraint, ChaiRestraintType

        with pytest.raises(Exception, match="Failed to parse residue index"):
            self._make_protein_config(
                [
                    ChaiRestraint(
                        restraint_id="r0",
                        chainA="A",
                        res_idxA="MXYZ",  # can't parse 'XYZ' as int
                        chainB="B",
                        res_idxB="G1",
                        connection_type=ChaiRestraintType.Contact,
                        max_distance_angstrom=6.0,
                    )
                ]
            )

    def test_pocket_parse_residue_failure(self):
        """Pocket restraint with unparseable residue should raise."""
        from uniaf3.schema.chai import ChaiRestraint, ChaiRestraintType

        with pytest.raises(Exception, match="Failed to parse residue index"):
            self._make_protein_config(
                [
                    ChaiRestraint(
                        restraint_id="r0",
                        chainA="A",
                        res_idxA=None,  # binder side is empty
                        chainB="B",
                        res_idxB="GXYZ",  # can't parse 'XYZ' as int
                        connection_type=ChaiRestraintType.Pocket,
                        max_distance_angstrom=8.0,
                    )
                ]
            )


class TestBoltzSchemaValidatorsMore:
    """Additional boltz schema validators tests."""

    def test_load_msa_file_not_found(self, tmp_path):
        """_load_msa_seqs should raise FileNotFoundError for missing file."""
        from uniaf3.schema.boltz import _load_msa_seqs

        with pytest.raises(FileNotFoundError):
            _load_msa_seqs("/nonexistent/path/file.a3m")

    def test_load_msa_unsupported_type_raises(self, tmp_path):
        """_load_msa_seqs should raise for unsupported file types."""
        from uniaf3.schema.boltz import _load_msa_seqs

        bad_file = tmp_path / "msa.txt"
        bad_file.write_text("content")
        with pytest.raises(ValueError, match="Unsupported MSA file type"):
            _load_msa_seqs(bad_file)

    def test_dna_modification_out_of_range(self):
        """BoltzDNA modification out of range should raise."""
        from uniaf3.schema.boltz import BoltzDNA, BoltzModification

        with pytest.raises(Exception, match="out of range"):
            BoltzDNA(
                id="A",
                sequence="ACGT",
                modifications=[BoltzModification(position=10, ccd="HY3")],
            )

    def test_rna_modification_out_of_range(self):
        """BoltzRNA modification out of range should raise."""
        from uniaf3.schema.boltz import BoltzModification, BoltzRNA

        with pytest.raises(Exception, match="out of range"):
            BoltzRNA(
                id="A",
                sequence="ACGU",
                modifications=[BoltzModification(position=10, ccd="HY3")],
            )

    def test_ligand_both_ccd_smiles_raises(self):
        """BoltzLigand with both ccd and smiles should raise."""
        from uniaf3.schema.boltz import BoltzLigand

        with pytest.raises(Exception, match="Exactly one of ccd or smiles"):
            BoltzLigand(id="A", ccd="SAH", smiles="CCO")

    def test_ligand_neither_ccd_smiles_raises(self):
        """BoltzLigand with neither ccd nor smiles should raise."""
        from uniaf3.schema.boltz import BoltzLigand

        with pytest.raises(Exception, match="Exactly one of ccd or smiles"):
            BoltzLigand(id="A")

    def test_sequence_entry_multiple_types_raises(self):
        """BoltzSequenceEntry with multiple types should raise."""
        from uniaf3.schema.boltz import BoltzLigand, BoltzProtein, BoltzSequenceEntry

        with pytest.raises(
            Exception, match="Exactly one of protein, dna, rna, or ligand"
        ):
            BoltzSequenceEntry(
                protein=BoltzProtein(id="A", sequence="MVLS"),
                ligand=BoltzLigand(id="B", ccd="SAH"),
            )

    def test_constraint_entry_multiple_types_raises(self):
        """BoltzConstraintEntry with multiple types should raise."""
        from uniaf3.schema.boltz import (
            BoltzBondConstraint,
            BoltzConstraintEntry,
            BoltzPocketConstraint,
        )

        with pytest.raises(Exception, match="Exactly one of bond, pocket, or contact"):
            BoltzConstraintEntry(
                bond=BoltzBondConstraint(atom1=("A", 1, "CA"), atom2=("A", 2, "CA")),
                pocket=BoltzPocketConstraint(
                    binder="A", contacts=[("A", 1)], max_distance=6.0
                ),
            )

    def test_template_both_cif_pdb_raises(self):
        """BoltzTemplate with both cif and pdb should raise."""
        from uniaf3.schema.boltz import BoltzTemplate

        with pytest.raises(Exception, match="Exactly one of cif or pdb"):
            BoltzTemplate(cif="/path/to/file.cif", pdb="/path/to/file.pdb")

    def test_template_neither_cif_pdb_raises(self):
        """BoltzTemplate with neither cif nor pdb should raise."""
        from uniaf3.schema.boltz import BoltzTemplate

        with pytest.raises(Exception, match="Exactly one of cif or pdb"):
            BoltzTemplate()

    def test_boltz_config_to_files(self, tmp_path):
        """BoltzConfig.to_files should write a YAML file."""
        from uniaf3.schema.boltz import BoltzConfig, BoltzProtein, BoltzSequenceEntry

        config = BoltzConfig(
            sequences=[
                BoltzSequenceEntry(protein=BoltzProtein(id="A", sequence="MVLS"))
            ]
        )
        config.to_files(tmp_path, "test_boltz")
        assert (tmp_path / "test_boltz.yaml").exists()


class TestBoltzSchemaConstraintValidators:
    """Test BoltzConfig constraint validator edge cases."""

    def _make_config(self, constraints):
        from uniaf3.schema.boltz import (
            BoltzConfig,
            BoltzLigand,
            BoltzProtein,
            BoltzSequenceEntry,
        )

        return BoltzConfig(
            sequences=[
                BoltzSequenceEntry(protein=BoltzProtein(id="A", sequence="MVLSP")),
                BoltzSequenceEntry(ligand=BoltzLigand(id="B", ccd="SAH")),
            ],
            constraints=constraints,
        )

    def test_protein_modification_out_of_range(self):
        """BoltzProtein modification out of range should raise."""
        from uniaf3.schema.boltz import BoltzModification, BoltzProtein

        with pytest.raises(Exception, match="out of range"):
            BoltzProtein(
                id="A",
                sequence="MVLSP",
                modifications=[BoltzModification(position=10, ccd="HY3")],
            )

    def test_pocket_polymer_contact_wrong_type_raises(self):
        """Pocket constraint with string (not int) for polymer chain should raise."""
        from uniaf3.schema.boltz import BoltzConstraintEntry, BoltzPocketConstraint

        with pytest.raises(Exception, match="should be specified with residue index"):
            self._make_config(
                [
                    BoltzConstraintEntry(
                        pocket=BoltzPocketConstraint(
                            binder="B",
                            contacts=[
                                ("A", "SG")
                            ],  # string for polymer chain - should be int
                            max_distance=6.0,
                        )
                    )
                ]
            )

    def test_pocket_polymer_residue_out_of_range_raises(self):
        """Pocket constraint with polymer residue out of range should raise."""
        from uniaf3.schema.boltz import BoltzConstraintEntry, BoltzPocketConstraint

        with pytest.raises(Exception, match="out of range"):
            self._make_config(
                [
                    BoltzConstraintEntry(
                        pocket=BoltzPocketConstraint(
                            binder="B",
                            contacts=[("A", 100)],  # > len("MVLSP")=5
                            max_distance=6.0,
                        )
                    )
                ]
            )

    def test_pocket_ligand_contact_wrong_type_raises(self):
        """Pocket constraint with int (not string) for ligand chain should raise."""
        from uniaf3.schema.boltz import BoltzConstraintEntry, BoltzPocketConstraint

        with pytest.raises(Exception, match="should be specified with atom name"):
            self._make_config(
                [
                    BoltzConstraintEntry(
                        pocket=BoltzPocketConstraint(
                            binder="A",
                            contacts=[("B", 1)],  # int for ligand - should be str
                            max_distance=6.0,
                        )
                    )
                ]
            )

    def test_contact_polymer_wrong_type_raises(self):
        """Contact constraint with string (not int) for polymer should raise."""
        from uniaf3.schema.boltz import BoltzConstraintEntry, BoltzContactConstraint

        with pytest.raises(Exception, match="should be specified with residue index"):
            self._make_config(
                [
                    BoltzConstraintEntry(
                        contact=BoltzContactConstraint(
                            token1=("A", "SG"),  # string for polymer - should be int
                            token2=("A", 1),
                            max_distance=6.0,
                        )
                    )
                ]
            )

    def test_contact_polymer_residue_out_of_range_raises(self):
        """Contact constraint with polymer residue out of range should raise."""
        from uniaf3.schema.boltz import BoltzConstraintEntry, BoltzContactConstraint

        with pytest.raises(Exception, match="out of range"):
            self._make_config(
                [
                    BoltzConstraintEntry(
                        contact=BoltzContactConstraint(
                            token1=("A", 100),  # > len("MVLSP")=5
                            token2=("A", 1),
                            max_distance=6.0,
                        )
                    )
                ]
            )

    def test_contact_ligand_wrong_type_raises(self):
        """Contact constraint with int (not string) for ligand should raise."""
        from uniaf3.schema.boltz import BoltzConstraintEntry, BoltzContactConstraint

        with pytest.raises(Exception, match="should be specified with atom name"):
            self._make_config(
                [
                    BoltzConstraintEntry(
                        contact=BoltzContactConstraint(
                            token1=("A", 1),
                            token2=("B", 1),  # int for ligand - should be str
                            max_distance=6.0,
                        )
                    )
                ]
            )
