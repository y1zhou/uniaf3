"""Tests for model-specific schema validation."""

import orjson
import pytest
import yaml

from uniaf3.schema import (
    AF3Config,
    AF3ServerConfig,
    BoltzConfig,
    ProtenixConfig,
    UniAF3Config,
)


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
        assert prot.sequence == "PVLSCGEWQL"
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
        assert prot.msa == "./examples/msa/seq1.a3m"

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
        assert b.atom1 == ("A", 145, "SG")

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

    # TODO: Chai should output two files (FASTA and restraints CSV)
    ...


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
        assert bonds[0].entity1 == "1"
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
        assert uniaf3_conf.seeds == [42, 123]

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
