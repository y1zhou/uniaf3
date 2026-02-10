"""Tests for adapter conversions between UniAF3 and model configs.

All conversions go through UniAF3Config as an intermediate layer.
"""

from pathlib import Path

import pytest
import yaml

from uniaf3.schema import AF3Config, UniAF3Config
from uniaf3.schema.base import Polymer

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="module", autouse=True)
def uniaf3_conf():
    with open(FIXTURES / "uniaf3_example.yaml") as f:
        data = yaml.safe_load(f)
    conf = UniAF3Config.model_validate(data)
    return conf


# ============================================================
# UniAF3 → AlphaFold3 → UniAF3
# ============================================================
@pytest.fixture(scope="class", autouse=True)
def af3_conf(uniaf3_conf: UniAF3Config):
    from uniaf3.adapters import to_alphafold3

    af3 = to_alphafold3(uniaf3_conf, name="test_job")
    return af3


class TestAF3Adapter:
    """Test round-trip conversion through AlphaFold3."""

    def test_uniaf3_to_af3(self, af3_conf: AF3Config, uniaf3_conf: UniAF3Config):

        assert af3_conf.name == "test_job"
        assert af3_conf.modelSeeds == uniaf3_conf.seeds
        assert len(af3_conf.sequences) == len(
            uniaf3_conf.sequences
        )  # protein, dna, 2 ligands, glycan

    def test_uniaf3_to_af3_seqs(self, af3_conf: AF3Config, uniaf3_conf: UniAF3Config):
        for af3_seq, uniaf3_seq in zip(
            af3_conf.sequences, uniaf3_conf.sequences, strict=True
        ):
            # Get the non-None field
            for s in (af3_seq.protein, af3_seq.dna, af3_seq.rna, af3_seq.ligand):
                if s is not None:
                    break

            assert s.id == uniaf3_seq.id
            if isinstance(uniaf3_seq, Polymer):
                assert s.sequence == uniaf3_seq.sequence
                if s.modifications is not None:
                    for af3_mod, uni_mod in zip(
                        s.modifications, uniaf3_seq.modifications, strict=True
                    ):
                        assert af3_mod.ptmType == uni_mod.ccd
                        assert af3_mod.ptmPosition == uni_mod.position

    def test_af3_to_uniaf3(self, af3_conf: AF3Config, uniaf3_conf: UniAF3Config):
        from uniaf3.adapters import from_alphafold3

        uni = from_alphafold3(af3_conf)
        assert uni.hash == uniaf3_conf.hash
