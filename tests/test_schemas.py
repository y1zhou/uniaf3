"""Tests for model-specific schema validation."""

from pathlib import Path

import orjson
import pytest
import yaml

FIXTURES = Path(__file__).parent / "fixtures"


# ============================================================
# AlphaFold3 schema
# ============================================================
class TestAF3Schema:
    """Validate AF3Config against example input."""

    @pytest.fixture(autouse=True)
    def load_json_data(self):
        from uniaf3.schema.alphafold3 import AF3Config

        self.data = orjson.loads((FIXTURES / "alphafold3_example.json").read_bytes())
        self.conf = AF3Config.model_validate(self.data)

    def test_load_example(self):
        assert self.conf.name == "Hello fold"
        assert self.conf.modelSeeds == [10, 42]
        assert len(self.conf.sequences) == 7
        assert self.conf.dialect == "alphafold3"
        assert self.conf.version == 4

    def test_protein_entry(self):
        prot = self.conf.sequences[0].protein
        assert prot is not None
        assert prot.id == "A"
        assert prot.sequence == "PVLSCGEWQL"
        assert prot.modifications is not None
        assert len(prot.modifications) == 2
        assert prot.modifications[0].ptmType == "HY3"
        assert prot.modifications[0].ptmPosition == 1

    def test_dna_entry(self):
        dna = self.conf.sequences[2].dna
        assert dna is not None
        assert dna.sequence == "GACCTCT"
        assert dna.modifications is not None
        assert len(dna.modifications) == 2

    def test_rna_entry(self):
        rna = self.conf.sequences[3].rna
        assert rna is not None
        assert rna.sequence == "AGCU"

    def test_ligand_with_ccd(self):
        lig = self.conf.sequences[4].ligand
        assert lig is not None
        assert lig.id == ["F", "G", "H"]
        assert lig.ccdCodes == ["ATP"]

    def test_ligand_with_smiles(self):
        lig = self.conf.sequences[6].ligand
        assert lig is not None
        assert lig.smiles == "CC(=O)OC1C[NH+]2CCC1CC2"

    def test_bonded_atom_pairs(self):
        assert self.conf.bondedAtomPairs is not None
        assert len(self.conf.bondedAtomPairs) == 2
        a1, a2 = self.conf.bondedAtomPairs[0]
        assert a1 == ("A", 1, "CA")
        assert a2 == ("F", 1, "CHA")

    def test_json_str_property(self):
        j = self.conf.to_str()
        assert isinstance(j, str)
        parsed = orjson.loads(j)
        assert parsed["name"] == "Hello fold"

    def test_from_file(self):
        from uniaf3.schema.alphafold3 import AF3Config

        conf = AF3Config.from_file(FIXTURES / "alphafold3_example.json")
        assert conf.name == "Hello fold"
        assert len(conf.sequences) == 7

    def test_invalid_ligand_both_ccd_smiles(self):
        from uniaf3.schema.alphafold3 import AF3Ligand

        with pytest.raises(Exception):
            AF3Ligand(id="X", ccdCodes=["ATP"], smiles="CC(=O)OC1C[NH+]2CCC1CC2")

    def test_invalid_ligand_neither_ccd_smiles(self):
        from uniaf3.schema.alphafold3 import AF3Ligand

        with pytest.raises(Exception):
            AF3Ligand(id="X")

    def test_sequence_entry_exactly_one(self):
        from uniaf3.schema.alphafold3 import (
            AF3DNA,
            AF3Protein,
            AF3SequenceEntry,
        )

        # Zero entities should fail
        with pytest.raises(Exception):
            AF3SequenceEntry()

        # Two entities set should fail
        with pytest.raises(Exception):
            AF3SequenceEntry(
                protein=AF3Protein(id="A", sequence="M"),
                dna=AF3DNA(id="B", sequence="G"),
            )


# ============================================================
# Boltz schema
# ============================================================
class TestBoltzSchema:
    """Validate BoltzConfig against example input."""

    def test_load_example(self):
        from uniaf3.schema.boltz import BoltzConfig

        with open(FIXTURES / "boltz_example.yaml") as f:
            data = yaml.safe_load(f)
        conf = BoltzConfig.model_validate(data)
        assert conf.version == 1
        assert len(conf.sequences) == 3

    def test_protein_entry(self):
        from uniaf3.schema.boltz import BoltzConfig

        with open(FIXTURES / "boltz_example.yaml") as f:
            data = yaml.safe_load(f)
        conf = BoltzConfig.model_validate(data)
        prot = conf.sequences[0].protein
        assert prot is not None
        assert prot.id == ["A", "B"]
        assert prot.msa == "./examples/msa/seq1.a3m"

    def test_ligand_ccd(self):
        from uniaf3.schema.boltz import BoltzConfig

        with open(FIXTURES / "boltz_example.yaml") as f:
            data = yaml.safe_load(f)
        conf = BoltzConfig.model_validate(data)
        lig = conf.sequences[1].ligand
        assert lig is not None
        assert lig.ccd == "SAH"

    def test_ligand_smiles(self):
        from uniaf3.schema.boltz import BoltzConfig

        with open(FIXTURES / "boltz_example.yaml") as f:
            data = yaml.safe_load(f)
        conf = BoltzConfig.model_validate(data)
        lig = conf.sequences[2].ligand
        assert lig is not None
        assert lig.smiles is not None

    def test_bond_constraint(self):
        from uniaf3.schema.boltz import BoltzConfig

        with open(FIXTURES / "boltz_example.yaml") as f:
            data = yaml.safe_load(f)
        conf = BoltzConfig.model_validate(data)
        assert len(conf.constraints) == 3
        b = conf.constraints[0].bond
        assert b is not None
        assert b.atom1 == ("A", 145, "SG")

    def test_yaml_str_property(self):
        from uniaf3.schema.boltz import BoltzConfig

        with open(FIXTURES / "boltz_example.yaml") as f:
            data = yaml.safe_load(f)
        conf = BoltzConfig.model_validate(data)
        y = conf.to_str()
        assert isinstance(y, str)
        parsed = yaml.safe_load(y)
        assert parsed["version"] == 1

    def test_from_file(self):
        from uniaf3.schema.boltz import BoltzConfig

        conf = BoltzConfig.from_file(FIXTURES / "boltz_example.yaml")
        assert conf.version == 1
        assert len(conf.sequences) == 3


# ============================================================
# Chai schema
# ============================================================
class TestChaiSchema:
    """Validate ChaiConfig against example input."""

    def test_load_example(self):
        from uniaf3.schema.chai import ChaiConfig

        with open(FIXTURES / "chai_example.yaml") as f:
            data = yaml.safe_load(f)
        conf = ChaiConfig.model_validate(data)
        assert len(conf.entities) == 6
        assert conf.seed == 42

    def test_entities(self):
        from uniaf3.schema.chai import ChaiConfig

        with open(FIXTURES / "chai_example.yaml") as f:
            data = yaml.safe_load(f)
        conf = ChaiConfig.model_validate(data)
        assert conf.entities[0].entity_type == "protein"
        assert conf.entities[2].entity_type == "dna"
        assert conf.entities[3].entity_type == "ligand"

    def test_restraints(self):
        from uniaf3.schema.chai import ChaiConfig

        with open(FIXTURES / "chai_example.yaml") as f:
            data = yaml.safe_load(f)
        conf = ChaiConfig.model_validate(data)
        assert len(conf.restraints) == 3
        r = conf.restraints[0]
        assert r.connection_type == "covalent"
        assert r.res_idxA == "A219@CA"

    def test_unique_entity_names(self):
        from uniaf3.schema.chai import ChaiConfig, ChaiEntity, ChaiEntityType

        with pytest.raises(Exception):
            ChaiConfig(
                entities=[
                    ChaiEntity(
                        entity_type=ChaiEntityType.Protein,
                        entity_name="X",
                        sequence="M",
                    ),
                    ChaiEntity(
                        entity_type=ChaiEntityType.Protein,
                        entity_name="X",
                        sequence="M",
                    ),
                ]
            )

    def test_output_strs_property(self):
        raise NotImplementedError("output_strs property not implemented yet")

    def test_from_file(self):
        from uniaf3.schema.chai import ChaiConfig

        conf = ChaiConfig.from_file(FIXTURES / "chai_example.yaml")
        assert len(conf.entities) == 6
        assert conf.seed == 42


# ============================================================
# Protenix schema
# ============================================================
class TestProtenixSchema:
    """Validate ProtenixConfig against example input."""

    def test_load_example(self):
        from uniaf3.schema.protenix import ProtenixConfig

        data = orjson.loads((FIXTURES / "protenix_example.json").read_bytes())
        # Protenix top-level is a list
        conf = ProtenixConfig.model_validate({"jobs": data})
        assert len(conf.jobs) == 1
        job = conf.jobs[0]
        assert job.name == "Test Fold Job"
        assert len(job.sequences) == 6

    def test_protein_chain(self):
        from uniaf3.schema.protenix import ProtenixConfig

        data = orjson.loads((FIXTURES / "protenix_example.json").read_bytes())
        conf = ProtenixConfig.model_validate({"jobs": data})
        pc = conf.jobs[0].sequences[0].proteinChain
        assert pc is not None
        assert pc.sequence == "PREACHINGS"
        assert pc.count == 2
        assert len(pc.modifications) == 2
        assert pc.modifications[0].ptmType == "CCD_HY3"

    def test_dna_sequence(self):
        from uniaf3.schema.protenix import ProtenixConfig

        data = orjson.loads((FIXTURES / "protenix_example.json").read_text())
        conf = ProtenixConfig.model_validate({"jobs": data})
        ds = conf.jobs[0].sequences[1].dnaSequence
        assert ds is not None
        assert ds.sequence == "GATTACA"

    def test_ligand(self):
        from uniaf3.schema.protenix import ProtenixConfig

        data = orjson.loads((FIXTURES / "protenix_example.json").read_text())
        conf = ProtenixConfig.model_validate({"jobs": data})
        lig = conf.jobs[0].sequences[3].ligand
        assert lig is not None
        assert lig.ligand == "CCD_ATP"

    def test_ion(self):
        from uniaf3.schema.protenix import ProtenixConfig

        data = orjson.loads((FIXTURES / "protenix_example.json").read_text())
        conf = ProtenixConfig.model_validate({"jobs": data})
        ion = conf.jobs[0].sequences[4].ion
        assert ion is not None
        assert ion.ion == "MG"
        assert ion.count == 2

    def test_covalent_bonds(self):
        from uniaf3.schema.protenix import ProtenixConfig

        data = orjson.loads((FIXTURES / "protenix_example.json").read_text())
        conf = ProtenixConfig.model_validate({"jobs": data})
        bonds = conf.jobs[0].covalent_bonds
        assert len(bonds) == 1
        assert bonds[0].entity1 == "1"
        assert bonds[0].atom1 == "N6"

    def test_json_str_property(self):
        from uniaf3.schema.protenix import ProtenixConfig

        data = orjson.loads((FIXTURES / "protenix_example.json").read_text())
        conf = ProtenixConfig.model_validate({"jobs": data})
        j = conf.to_str()
        assert isinstance(j, str)
        parsed = orjson.loads(j)
        assert "jobs" in parsed

    def test_from_file(self):
        from uniaf3.schema.protenix import ProtenixConfig

        conf = ProtenixConfig.from_file(FIXTURES / "protenix_example.json")
        assert len(conf.jobs) == 1
        assert conf.jobs[0].name == "Test Fold Job"


# ============================================================
# UniAF3 schema
# ============================================================
class TestUniAF3Schema:
    """Validate UniAF3Config against example input."""

    def test_load_example(self):
        from uniaf3.schema import UniAF3Config

        conf = UniAF3Config.from_file(FIXTURES / "uniaf3_example.yaml")
        assert len(conf.sequences) == 5
        assert conf.seeds == [42, 123]

    def test_yaml_str(self):
        from uniaf3.schema import UniAF3Config

        conf = UniAF3Config.from_file(FIXTURES / "uniaf3_example.yaml")
        y = conf.to_yaml()
        assert isinstance(y, str)
        parsed = yaml.safe_load(y)
        assert "sequences" in parsed

    def test_json_str(self):
        from uniaf3.schema import UniAF3Config

        conf = UniAF3Config.from_file(FIXTURES / "uniaf3_example.yaml")
        j = conf.to_json()
        assert isinstance(j, str)

    def test_hash(self):
        from uniaf3.schema import UniAF3Config

        conf = UniAF3Config.from_file(FIXTURES / "uniaf3_example.yaml")
        h = conf.hash
        assert isinstance(h, str)
        assert len(h) == 64  # sha256 hex digest
