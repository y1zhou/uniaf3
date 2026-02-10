"""Comprehensive tests for adapter conversions between UniAF3 and model configs.

All conversions go through UniAF3Config as an intermediate layer.
"""

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
from uniaf3.schema.base import (
    InferenceParams,
    Polymer,
    PolymerType,
    ProteinSeq,
)

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="module")
def uniaf3_conf():
    """Load the UniAF3 example fixture."""
    return UniAF3Config.from_file(FIXTURES / "uniaf3_example.yaml")


# ============================================================
# UniAF3 → Boltz → UniAF3
# ============================================================
class TestBoltzAdapter:
    """Test round-trip conversion through Boltz."""

    def test_unsupported_glycan(self, uniaf3_conf, tmp_path):
        from uniaf3.adapters import to_boltz

        with pytest.raises(
            ValueError, match="Glycans are not directly supported in Boltz"
        ):
            to_boltz(uniaf3_conf, msa_dir=tmp_path, strict=True)

    def test_uniaf3_to_boltz(self, uniaf3_conf, tmp_path):
        from uniaf3.adapters import to_boltz

        boltz = to_boltz(uniaf3_conf, msa_dir=tmp_path, strict=False)
        assert boltz.version == 1
        # protein + dna + 2 ligands; 1 glycan dropped in non-strict mode
        assert len(boltz.sequences) == len(uniaf3_conf.sequences) - 1

    def test_boltz_constraints(self, uniaf3_conf, tmp_path):
        from uniaf3.adapters import to_boltz

        boltz = to_boltz(uniaf3_conf, msa_dir=tmp_path, strict=False)
        assert boltz.constraints is not None
        assert uniaf3_conf.restraints is not None
        assert len(boltz.constraints) == len(uniaf3_conf.restraints)

    def test_boltz_properties(self, uniaf3_conf, tmp_path):
        from uniaf3.adapters import to_boltz

        boltz = to_boltz(uniaf3_conf, msa_dir=tmp_path, strict=False)
        assert boltz.properties is not None

    def test_boltz_to_uniaf3(self, tmp_path):
        """Load Boltz fixture and convert to UniAF3."""
        from uniaf3.adapters import from_boltz

        boltz = BoltzConfig.from_file(FIXTURES / "boltz_example.yaml")
        uni = from_boltz(boltz)
        assert len(uni.sequences) == len(boltz.sequences)
        # Check protein preserved
        prot = uni.sequences[0]
        assert isinstance(prot, ProteinSeq)
        assert prot.sequence == boltz.sequences[0].protein.sequence

    def test_boltz_to_uniaf3_constraints(self):
        """Verify Boltz constraints convert to UniAF3 restraints."""
        from uniaf3.adapters import from_boltz

        boltz = BoltzConfig.from_file(FIXTURES / "boltz_example.yaml")
        uni = from_boltz(boltz)
        assert uni.restraints is not None
        assert boltz.constraints is not None
        assert len(uni.restraints) == len(boltz.constraints)

    def test_boltz_roundtrip(self, uniaf3_conf, tmp_path):
        """UniAF3 → Boltz → UniAF3 preserves key data."""
        from uniaf3.adapters import from_boltz, to_boltz

        boltz = to_boltz(uniaf3_conf, msa_dir=tmp_path, strict=False)
        roundtrip = from_boltz(boltz)
        # Glycan is dropped, so we have one less sequence
        assert len(roundtrip.sequences) == len(uniaf3_conf.sequences) - 1
        # Check seeds default since Boltz config doesn't include them
        assert roundtrip.seeds == [42]


# ============================================================
# UniAF3 → AF3 → UniAF3
# ============================================================
class TestAF3Adapter:
    """Test round-trip conversion through AlphaFold3."""

    def test_uniaf3_to_af3(self, uniaf3_conf):
        from uniaf3.adapters import to_alphafold3

        af3 = to_alphafold3(uniaf3_conf, name="test", strict=False)
        assert af3.name == "test"
        assert af3.modelSeeds == uniaf3_conf.seeds
        # Glycan is dropped in non-strict mode
        assert len(af3.sequences) == len(uniaf3_conf.sequences) - 1
        assert af3.dialect == "alphafold3"

    def test_af3_protein_preserved(self, uniaf3_conf):
        from uniaf3.adapters import to_alphafold3

        af3 = to_alphafold3(uniaf3_conf, name="test", strict=False)
        prot = af3.sequences[0].protein
        assert prot is not None
        assert prot.sequence == "MVLSPADKTNVKAAW"
        assert prot.modifications is not None
        assert prot.modifications[0].ptmType == "HY3"

    def test_af3_bonds(self, uniaf3_conf):
        from uniaf3.adapters import to_alphafold3

        af3 = to_alphafold3(uniaf3_conf, name="test", strict=False)
        # Only covalent bonds preserved in AF3
        assert af3.bondedAtomPairs is not None
        assert len(af3.bondedAtomPairs) == 1

    def test_af3_non_bond_restraints_dropped(self, uniaf3_conf):
        """AF3 only supports covalent bonds; glycan and contact/pocket raise in strict mode."""
        from uniaf3.adapters import to_alphafold3

        # In strict mode, glycan is the first unsupported feature hit
        with pytest.raises(ValueError, match="not directly supported in AF3"):
            to_alphafold3(uniaf3_conf, name="test", strict=True)

    def test_af3_to_uniaf3(self):
        """Load AF3 fixture and convert to UniAF3."""
        from uniaf3.adapters import from_alphafold3

        af3 = AF3Config.from_file(FIXTURES / "alphafold3_example.json")
        uni = from_alphafold3(af3)
        assert len(uni.sequences) == len(af3.sequences)
        assert uni.seeds == af3.modelSeeds
        # Verify protein is preserved as ProteinSeq
        prot = uni.sequences[0]
        assert isinstance(prot, ProteinSeq)
        assert prot.sequence == "PVLSCGEWQL"
        assert prot.modifications is not None
        assert prot.modifications[0].ccd == "HY3"

    def test_af3_to_uniaf3_bonds(self):
        """AF3 bonded pairs convert to covalent restraints."""
        from uniaf3.adapters import from_alphafold3

        af3 = AF3Config.from_file(FIXTURES / "alphafold3_example.json")
        uni = from_alphafold3(af3)
        assert uni.restraints is not None
        assert len(uni.restraints) == 2
        assert uni.restraints[0].restraint_type.value == "bond"

    def test_af3_to_uniaf3_dna_mods(self):
        """AF3 DNA modifications preserved."""
        from uniaf3.adapters import from_alphafold3

        af3 = AF3Config.from_file(FIXTURES / "alphafold3_example.json")
        uni = from_alphafold3(af3)
        dna = uni.sequences[2]
        assert isinstance(dna, Polymer)
        assert dna.seq_type == PolymerType.DNA
        assert dna.modifications is not None
        assert dna.modifications[0].ccd == "6OG"

    def test_af3_roundtrip(self, uniaf3_conf):
        """UniAF3 → AF3 → UniAF3 preserves key data."""
        from uniaf3.adapters import from_alphafold3, to_alphafold3

        af3 = to_alphafold3(uniaf3_conf, name="test", strict=False)
        roundtrip = from_alphafold3(af3)
        # Glycan dropped, so one less
        assert len(roundtrip.sequences) == len(uniaf3_conf.sequences) - 1
        assert roundtrip.seeds == uniaf3_conf.seeds


# ============================================================
# UniAF3 → AF3 Server → UniAF3
# ============================================================
class TestAF3ServerAdapter:
    """Test round-trip conversion through AlphaFold3 Server."""

    def test_uniaf3_to_af3_server(self, uniaf3_conf):
        from uniaf3.adapters import to_alphafold3_server

        af3s = to_alphafold3_server(uniaf3_conf, name="test", strict=False)
        assert len(af3s) == 1
        job = af3s[0]
        assert job.name == "test"
        assert job.dialect == "alphafoldserver"

    def test_af3_server_to_uniaf3(self):
        """Load AF3 Server fixture and convert to UniAF3."""
        from uniaf3.adapters import from_alphafold3_server

        af3s = AF3ServerConfig.from_file(FIXTURES / "alphafold3_server_example.json")
        uni = from_alphafold3_server(af3s)
        # First job has 9 sequence entries
        assert len(uni.sequences) > 0
        prot = uni.sequences[0]
        assert isinstance(prot, ProteinSeq)
        assert prot.sequence == "PREACHINGS"

    def test_af3_server_modifications_preserved(self):
        """AF3 Server modifications convert correctly."""
        from uniaf3.adapters import from_alphafold3_server

        af3s = AF3ServerConfig.from_file(FIXTURES / "alphafold3_server_example.json")
        uni = from_alphafold3_server(af3s)
        prot = uni.sequences[0]
        assert isinstance(prot, ProteinSeq)
        assert prot.modifications is not None
        assert prot.modifications[0].ccd == "HY3"

    def test_af3_server_ions(self):
        """AF3 Server ions convert to Ligand with CCD."""
        from uniaf3.adapters import from_alphafold3_server

        af3s = AF3ServerConfig.from_file(FIXTURES / "alphafold3_server_example.json")
        uni = from_alphafold3_server(af3s)
        # Find the ion entry (should be MG with count 2)
        from uniaf3.schema.base import Ligand

        ions = [s for s in uni.sequences if isinstance(s, Ligand) and s.ccd == ["MG"]]
        assert len(ions) == 1
        # count=2 means 2 chain IDs
        assert isinstance(ions[0].id, list)
        assert len(ions[0].id) == 2


# ============================================================
# UniAF3 → Chai → UniAF3
# ============================================================
class TestChaiAdapter:
    """Test round-trip conversion through Chai-1."""

    def test_uniaf3_to_chai(self, uniaf3_conf):
        from uniaf3.adapters import to_chai

        chai = to_chai(uniaf3_conf)
        # Chai expands multi-ID entities: protein A,B → 2 entities + dna + 2 ligands + glycan
        assert len(chai.entities) > 0
        # First entity should be protein
        assert chai.entities[0].entity_type.value == "protein"
        assert chai.entities[0].entity_name == "A"

    def test_chai_inference_params(self, uniaf3_conf):
        from uniaf3.adapters import to_chai

        chai = to_chai(uniaf3_conf)
        assert (
            chai.num_trunk_recycles == uniaf3_conf.inference_params.num_trunk_recycles
        )
        assert (
            chai.num_diffn_timesteps == uniaf3_conf.inference_params.num_diffn_timesteps
        )
        assert chai.seed == uniaf3_conf.seeds[0]

    def test_chai_restraints(self, uniaf3_conf):
        from uniaf3.adapters import to_chai

        chai = to_chai(uniaf3_conf)
        assert chai.restraints is not None
        assert len(chai.restraints) == len(uniaf3_conf.restraints)

    def test_chai_to_uniaf3(self):
        """Load Chai fixture and convert to UniAF3."""
        from uniaf3.adapters import from_chai

        chai = ChaiConfig.from_file(FIXTURES / "chai_example.yaml")
        uni = from_chai(chai)
        assert len(uni.sequences) == len(chai.entities)
        # First is protein
        assert isinstance(uni.sequences[0], ProteinSeq)
        assert uni.sequences[0].sequence == chai.entities[0].sequence

    def test_chai_to_uniaf3_restraints(self):
        """Chai restraints convert back to UniAF3."""
        from uniaf3.adapters import from_chai

        chai = ChaiConfig.from_file(FIXTURES / "chai_example.yaml")
        uni = from_chai(chai)
        assert uni.restraints is not None
        assert len(uni.restraints) == len(chai.restraints)

    def test_chai_to_uniaf3_inference_params(self):
        """Chai inference params preserved in UniAF3."""
        from uniaf3.adapters import from_chai

        chai = ChaiConfig.from_file(FIXTURES / "chai_example.yaml")
        uni = from_chai(chai)
        assert uni.inference_params.num_trunk_recycles == chai.num_trunk_recycles
        assert uni.inference_params.num_diffn_timesteps == chai.num_diffn_timesteps

    def test_chai_roundtrip(self, uniaf3_conf):
        """UniAF3 → Chai → UniAF3 preserves key data."""
        from uniaf3.adapters import from_chai, to_chai

        chai = to_chai(uniaf3_conf)
        roundtrip = from_chai(chai)
        # Chai expands multi-ID into separate entities, so count differs
        # but total chain count should match (A,B protein + C dna + D lig + E lig + F glycan)
        assert len(roundtrip.sequences) > 0
        assert roundtrip.seeds == [uniaf3_conf.seeds[0]]


# ============================================================
# UniAF3 → Protenix → UniAF3
# ============================================================
class TestProtenixAdapter:
    """Test round-trip conversion through Protenix."""

    def test_uniaf3_to_protenix(self, uniaf3_conf):
        from uniaf3.adapters import to_protenix

        ptx = to_protenix(uniaf3_conf, name="test")
        assert len(ptx) == 1
        job = ptx[0]
        assert job.name == "test"
        assert len(job.sequences) > 0

    def test_protenix_covalent_bonds(self, uniaf3_conf):
        from uniaf3.adapters import to_protenix

        ptx = to_protenix(uniaf3_conf, name="test")
        job = ptx[0]
        assert job.covalent_bonds is not None
        assert len(job.covalent_bonds) == 1

    def test_protenix_constraints(self, uniaf3_conf):
        from uniaf3.adapters import to_protenix

        ptx = to_protenix(uniaf3_conf, name="test")
        job = ptx[0]
        assert job.constraint is not None
        assert job.constraint.contact is not None
        assert job.constraint.pocket is not None

    def test_protenix_to_uniaf3(self):
        """Load Protenix fixture and convert to UniAF3."""
        from uniaf3.adapters import from_protenix

        ptx = ProtenixConfig.from_file(FIXTURES / "protenix_example.json")
        uni = from_protenix(ptx)
        assert len(uni.sequences) == len(ptx[0].sequences)
        # First is protein with 2 copies
        prot = uni.sequences[0]
        assert isinstance(prot, ProteinSeq)
        assert prot.sequence == "PREACHINGS"
        assert isinstance(prot.id, list)
        assert len(prot.id) == 2  # count=2

    def test_protenix_to_uniaf3_modifications(self):
        """Protenix CCD_ prefixed modifications preserved."""
        from uniaf3.adapters import from_protenix

        ptx = ProtenixConfig.from_file(FIXTURES / "protenix_example.json")
        uni = from_protenix(ptx)
        prot = uni.sequences[0]
        assert isinstance(prot, ProteinSeq)
        assert prot.modifications is not None
        assert prot.modifications[0].ccd == "HY3"  # CCD_HY3 → HY3

    def test_protenix_to_uniaf3_bonds(self):
        """Protenix covalent bonds convert to restraints."""
        from uniaf3.adapters import from_protenix

        ptx = ProtenixConfig.from_file(FIXTURES / "protenix_example.json")
        uni = from_protenix(ptx)
        assert uni.restraints is not None
        # 1 covalent bond + 1 contact + 1 pocket = 3
        bond_restraints = [
            r for r in uni.restraints if r.restraint_type.value == "bond"
        ]
        assert len(bond_restraints) == 1

    def test_protenix_to_uniaf3_constraints(self):
        """Protenix contact/pocket constraints convert to restraints."""
        from uniaf3.adapters import from_protenix

        ptx = ProtenixConfig.from_file(FIXTURES / "protenix_example.json")
        uni = from_protenix(ptx)
        assert uni.restraints is not None
        contact_restraints = [
            r for r in uni.restraints if r.restraint_type.value == "contact"
        ]
        pocket_restraints = [
            r for r in uni.restraints if r.restraint_type.value == "pocket"
        ]
        assert len(contact_restraints) == 1
        assert len(pocket_restraints) == 1

    def test_protenix_roundtrip(self, uniaf3_conf):
        """UniAF3 → Protenix → UniAF3 preserves key data."""
        from uniaf3.adapters import from_protenix, to_protenix

        ptx = to_protenix(uniaf3_conf, name="test")
        roundtrip = from_protenix(ptx)
        # The number of UniAF3 sequences should match Protenix job sequences
        assert len(roundtrip.sequences) == len(ptx[0].sequences)


# ============================================================
# Cross-model conversions via UniAF3
# ============================================================
class TestCrossModelConversion:
    """Test conversions between different models via UniAF3 intermediate."""

    def test_af3_to_boltz_via_uniaf3(self, tmp_path):
        """AF3 → UniAF3 → Boltz."""
        from uniaf3.adapters import from_alphafold3, to_boltz

        af3 = AF3Config.from_file(FIXTURES / "alphafold3_example.json")
        uni = from_alphafold3(af3)
        # Clear MSA dirs since the files don't exist in test fixtures
        for seq in uni.sequences:
            if isinstance(seq, ProteinSeq):
                seq.msa_dir = None
        boltz = to_boltz(uni, msa_dir=tmp_path, strict=False)
        assert len(boltz.sequences) > 0

    def test_boltz_to_chai_via_uniaf3(self):
        """Boltz → UniAF3 → Chai."""
        from uniaf3.adapters import from_boltz, to_chai

        boltz = BoltzConfig.from_file(FIXTURES / "boltz_example.yaml")
        uni = from_boltz(boltz)
        chai = to_chai(uni)
        assert len(chai.entities) > 0

    def test_protenix_to_af3_via_uniaf3(self):
        """Protenix → UniAF3 → AF3."""
        from uniaf3.adapters import from_protenix, to_alphafold3

        ptx = ProtenixConfig.from_file(FIXTURES / "protenix_example.json")
        uni = from_protenix(ptx)
        af3 = to_alphafold3(uni, name="cross_test", strict=False)
        assert af3.name == "cross_test"
        assert len(af3.sequences) > 0

    def test_chai_to_protenix_via_uniaf3(self):
        """Chai → UniAF3 → Protenix."""
        from uniaf3.adapters import from_chai, to_protenix

        chai = ChaiConfig.from_file(FIXTURES / "chai_example.yaml")
        uni = from_chai(chai)
        ptx = to_protenix(uni, name="cross_test")
        assert len(ptx) == 1
        assert len(ptx[0].sequences) > 0

    def test_to_uniaf3_dispatch(self):
        """Test the to_uniaf3 dispatcher function."""
        from uniaf3.adapters import to_uniaf3

        af3 = AF3Config.from_file(FIXTURES / "alphafold3_example.json")
        uni = to_uniaf3(af3)
        assert isinstance(uni, UniAF3Config)

        boltz = BoltzConfig.from_file(FIXTURES / "boltz_example.yaml")
        uni2 = to_uniaf3(boltz)
        assert isinstance(uni2, UniAF3Config)

    def test_from_uniaf3_dispatch(self, uniaf3_conf, tmp_path):
        """Test the from_uniaf3 dispatcher function."""
        from uniaf3.adapters import from_uniaf3

        af3 = from_uniaf3(uniaf3_conf, AF3Config, name="dispatch_test", strict=False)
        assert isinstance(af3, AF3Config)

        chai = from_uniaf3(uniaf3_conf, ChaiConfig)
        assert isinstance(chai, ChaiConfig)

        ptx = from_uniaf3(uniaf3_conf, ProtenixConfig, name="dispatch_test")
        assert isinstance(ptx, ProtenixConfig)
