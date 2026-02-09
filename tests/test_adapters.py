"""Tests for adapter conversions between UniAF3 and model configs.

All conversions go through UniAF3Config as an intermediate layer.
"""

import json
from pathlib import Path

import yaml

FIXTURES = Path(__file__).parent / "fixtures"


# ============================================================
# UniAF3 → AlphaFold3 → UniAF3
# ============================================================
class TestAF3Adapter:
    """Test round-trip conversion through AlphaFold3."""

    def test_uniaf3_to_af3(self):
        from uniaf3.adapters import to_alphafold3
        from uniaf3.schema import UniAF3Config

        conf = UniAF3Config.from_file(FIXTURES / "uniaf3_example.yaml")
        af3 = to_alphafold3(conf, name="test_job")
        assert af3.name == "test_job"
        assert af3.modelSeeds == [42, 123]
        assert len(af3.sequences) == 5  # protein, dna, 2 ligands, glycan

    def test_uniaf3_to_af3_protein(self):
        from uniaf3.adapters import to_alphafold3
        from uniaf3.schema import UniAF3Config

        conf = UniAF3Config.from_file(FIXTURES / "uniaf3_example.yaml")
        af3 = to_alphafold3(conf, name="test_job")
        prot = af3.sequences[0].protein
        assert prot is not None
        assert prot.sequence == "MVLSPADKTNVKAAW"
        assert prot.id == ["A", "B"]
        assert len(prot.modifications) == 1

    def test_uniaf3_to_af3_bonds(self):
        from uniaf3.adapters import to_alphafold3
        from uniaf3.schema import UniAF3Config

        conf = UniAF3Config.from_file(FIXTURES / "uniaf3_example.yaml")
        af3 = to_alphafold3(conf)
        assert af3.bondedAtomPairs is not None
        assert len(af3.bondedAtomPairs) == 1

    def test_af3_to_uniaf3(self):
        from uniaf3.adapters import from_alphafold3
        from uniaf3.schema.alphafold3 import AF3Config

        data = json.loads((FIXTURES / "alphafold3_example.json").read_text())
        af3 = AF3Config.model_validate(data)
        uni = from_alphafold3(af3)
        assert len(uni.sequences) == 7
        assert uni.seeds == [10, 42]

    def test_af3_to_uniaf3_restraints(self):
        from uniaf3.adapters import from_alphafold3
        from uniaf3.schema.alphafold3 import AF3Config

        data = json.loads((FIXTURES / "alphafold3_example.json").read_text())
        af3 = AF3Config.model_validate(data)
        uni = from_alphafold3(af3)
        assert uni.restraints is not None
        assert len(uni.restraints) == 2
        assert uni.restraints[0].restraint_type == "bond"

    def test_roundtrip(self):
        from uniaf3.adapters import from_alphafold3, to_alphafold3
        from uniaf3.schema import UniAF3Config

        conf = UniAF3Config.from_file(FIXTURES / "uniaf3_example.yaml")
        af3 = to_alphafold3(conf, name="rt_test")
        back = from_alphafold3(af3)
        # Verify key fields survived the round-trip
        assert len(back.sequences) == len(conf.sequences)
        assert back.seeds == conf.seeds


# ============================================================
# UniAF3 → Boltz → UniAF3
# ============================================================
class TestBoltzAdapter:
    """Test round-trip conversion through Boltz."""

    def test_uniaf3_to_boltz(self):
        from uniaf3.adapters import to_boltz
        from uniaf3.schema import UniAF3Config

        conf = UniAF3Config.from_file(FIXTURES / "uniaf3_example.yaml")
        boltz = to_boltz(conf)
        assert boltz.version == 1
        assert len(boltz.sequences) == 5

    def test_uniaf3_to_boltz_constraints(self):
        from uniaf3.adapters import to_boltz
        from uniaf3.schema import UniAF3Config

        conf = UniAF3Config.from_file(FIXTURES / "uniaf3_example.yaml")
        boltz = to_boltz(conf)
        assert boltz.constraints is not None
        assert len(boltz.constraints) == 3

    def test_boltz_to_uniaf3(self):
        from uniaf3.adapters import from_boltz
        from uniaf3.schema.boltz import BoltzConfig

        with open(FIXTURES / "boltz_example.yaml") as f:
            data = yaml.safe_load(f)
        boltz = BoltzConfig.model_validate(data)
        uni = from_boltz(boltz)
        assert len(uni.sequences) == 3

    def test_boltz_to_uniaf3_restraints(self):
        from uniaf3.adapters import from_boltz
        from uniaf3.schema.boltz import BoltzConfig

        with open(FIXTURES / "boltz_example.yaml") as f:
            data = yaml.safe_load(f)
        boltz = BoltzConfig.model_validate(data)
        uni = from_boltz(boltz)
        assert uni.restraints is not None
        assert len(uni.restraints) == 3
        assert uni.restraints[0].restraint_type == "bond"

    def test_roundtrip(self):
        from uniaf3.adapters import from_boltz, to_boltz
        from uniaf3.schema import UniAF3Config

        conf = UniAF3Config.from_file(FIXTURES / "uniaf3_example.yaml")
        boltz = to_boltz(conf)
        back = from_boltz(boltz)
        assert len(back.sequences) == len(conf.sequences)


# ============================================================
# UniAF3 → Chai → UniAF3
# ============================================================
class TestChaiAdapter:
    """Test round-trip conversion through Chai-1."""

    def test_uniaf3_to_chai(self):
        from uniaf3.adapters import to_chai
        from uniaf3.schema import UniAF3Config

        conf = UniAF3Config.from_file(FIXTURES / "uniaf3_example.yaml")
        chai = to_chai(conf)
        # Protein with ids A,B expands to 2 entities
        # DNA C, ligand D, ligand E, glycan F = 4 more
        assert len(chai.entities) == 6

    def test_uniaf3_to_chai_restraints(self):
        from uniaf3.adapters import to_chai
        from uniaf3.schema import UniAF3Config

        conf = UniAF3Config.from_file(FIXTURES / "uniaf3_example.yaml")
        chai = to_chai(conf)
        assert chai.restraints is not None
        assert len(chai.restraints) == 3
        assert chai.restraints[0].connection_type == "covalent"

    def test_chai_to_uniaf3(self):
        from uniaf3.adapters import from_chai
        from uniaf3.schema.chai import ChaiConfig

        with open(FIXTURES / "chai_example.yaml") as f:
            data = yaml.safe_load(f)
        chai = ChaiConfig.model_validate(data)
        uni = from_chai(chai)
        assert len(uni.sequences) == 6

    def test_chai_to_uniaf3_restraints(self):
        from uniaf3.adapters import from_chai
        from uniaf3.schema.chai import ChaiConfig

        with open(FIXTURES / "chai_example.yaml") as f:
            data = yaml.safe_load(f)
        chai = ChaiConfig.model_validate(data)
        uni = from_chai(chai)
        assert uni.restraints is not None
        assert len(uni.restraints) == 3
        assert uni.restraints[0].restraint_type == "bond"

    def test_roundtrip(self):
        from uniaf3.adapters import from_chai, to_chai
        from uniaf3.schema import UniAF3Config

        conf = UniAF3Config.from_file(FIXTURES / "uniaf3_example.yaml")
        chai = to_chai(conf)
        back = from_chai(chai)
        # Protein id=[A,B] → 2 Chai entities → 2 ProteinSeqs on roundtrip
        assert len(back.sequences) >= len(conf.sequences)


# ============================================================
# UniAF3 → Protenix → UniAF3
# ============================================================
class TestProtenixAdapter:
    """Test round-trip conversion through Protenix."""

    def test_uniaf3_to_protenix(self):
        from uniaf3.adapters import to_protenix
        from uniaf3.schema import UniAF3Config

        conf = UniAF3Config.from_file(FIXTURES / "uniaf3_example.yaml")
        ptx = to_protenix(conf, name="test_job")
        assert len(ptx.jobs) == 1
        job = ptx.jobs[0]
        assert job.name == "test_job"
        # protein(count=2), dna, ATP ligand, SMILES ligand, glycan
        assert len(job.sequences) == 5

    def test_uniaf3_to_protenix_bonds(self):
        from uniaf3.adapters import to_protenix
        from uniaf3.schema import UniAF3Config

        conf = UniAF3Config.from_file(FIXTURES / "uniaf3_example.yaml")
        ptx = to_protenix(conf)
        job = ptx.jobs[0]
        assert job.covalent_bonds is not None
        assert len(job.covalent_bonds) == 1

    def test_protenix_to_uniaf3(self):
        from uniaf3.adapters import from_protenix
        from uniaf3.schema.protenix import ProtenixConfig

        data = json.loads((FIXTURES / "protenix_example.json").read_text())
        ptx = ProtenixConfig.model_validate({"jobs": data})
        uni = from_protenix(ptx)
        assert len(uni.sequences) == 6  # protein, dna, rna, ligand, ion, smiles_ligand

    def test_protenix_to_uniaf3_restraints(self):
        from uniaf3.adapters import from_protenix
        from uniaf3.schema.protenix import ProtenixConfig

        data = json.loads((FIXTURES / "protenix_example.json").read_text())
        ptx = ProtenixConfig.model_validate({"jobs": data})
        uni = from_protenix(ptx)
        assert uni.restraints is not None
        assert len(uni.restraints) == 3
        assert uni.restraints[0].restraint_type == "bond"

    def test_roundtrip(self):
        from uniaf3.adapters import from_protenix, to_protenix
        from uniaf3.schema import UniAF3Config

        conf = UniAF3Config.from_file(FIXTURES / "uniaf3_example.yaml")
        ptx = to_protenix(conf)
        back = from_protenix(ptx)
        assert len(back.sequences) == len(conf.sequences)


# ============================================================
# Cross-model conversion via UniAF3
# ============================================================
class TestCrossModelConversion:
    """Test conversion chains: modelA → UniAF3 → modelB."""

    def test_af3_to_boltz_via_uniaf3(self):
        from uniaf3.adapters import from_alphafold3, to_boltz
        from uniaf3.schema.alphafold3 import AF3Config

        data = json.loads((FIXTURES / "alphafold3_example.json").read_text())
        af3 = AF3Config.model_validate(data)
        uni = from_alphafold3(af3)
        boltz = to_boltz(uni)
        assert boltz.version == 1
        assert len(boltz.sequences) == 7

    def test_boltz_to_chai_via_uniaf3(self):
        from uniaf3.adapters import from_boltz, to_chai
        from uniaf3.schema.boltz import BoltzConfig

        with open(FIXTURES / "boltz_example.yaml") as f:
            data = yaml.safe_load(f)
        boltz = BoltzConfig.model_validate(data)
        uni = from_boltz(boltz)
        chai = to_chai(uni)
        assert len(chai.entities) >= 3

    def test_protenix_to_af3_via_uniaf3(self):
        from uniaf3.adapters import from_protenix, to_alphafold3
        from uniaf3.schema.protenix import ProtenixConfig

        data = json.loads((FIXTURES / "protenix_example.json").read_text())
        ptx = ProtenixConfig.model_validate({"jobs": data})
        uni = from_protenix(ptx)
        af3 = to_alphafold3(uni, name="ptx_to_af3")
        assert af3.name == "ptx_to_af3"
        assert len(af3.sequences) == 6

    def test_chai_to_protenix_via_uniaf3(self):
        from uniaf3.adapters import from_chai, to_protenix
        from uniaf3.schema.chai import ChaiConfig

        with open(FIXTURES / "chai_example.yaml") as f:
            data = yaml.safe_load(f)
        chai = ChaiConfig.model_validate(data)
        uni = from_chai(chai)
        ptx = to_protenix(uni, name="chai_to_ptx")
        assert len(ptx.jobs) == 1
        assert ptx.jobs[0].name == "chai_to_ptx"
