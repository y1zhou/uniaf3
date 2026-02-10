"""Tests for adapter conversions between UniAF3 and model configs.

All conversions go through UniAF3Config as an intermediate layer.
"""

from pathlib import Path

import pytest

from uniaf3.schema import BoltzConfig, UniAF3Config

# ruff: noqa: S101

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="module", autouse=True)
def uniaf3_conf():
    conf = UniAF3Config.from_file(FIXTURES / "uniaf3_example.yaml")
    return conf


# ============================================================
# UniAF3 → Boltz → UniAF3
# ============================================================
@pytest.fixture(scope="class", autouse=True)
def boltz_conf(uniaf3_conf: UniAF3Config):
    from uniaf3.adapters import to_boltz

    boltz = to_boltz(uniaf3_conf, msa_dir=Path.cwd(), strict=False)
    return boltz


class TestBoltzAdapter:
    """Test round-trip conversion through Boltz."""

    def test_unsupported_glycan(self, uniaf3_conf):
        from uniaf3.adapters import to_boltz

        with pytest.raises(
            ValueError, match="Glycans are not directly supported in Boltz"
        ):
            _ = to_boltz(uniaf3_conf, msa_dir=Path.cwd(), strict=True)

    def test_uniaf3_to_boltz(self, boltz_conf: BoltzConfig, uniaf3_conf: UniAF3Config):

        assert boltz_conf.version == 1
        assert (
            len(boltz_conf.sequences) == len(uniaf3_conf.sequences) - 1
        )  # protein, dna, 2 ligands; 1 glycan dropped

        assert boltz_conf.constraints is not None
        assert uniaf3_conf.restraints is not None
        assert len(boltz_conf.constraints) == len(uniaf3_conf.restraints)

        assert boltz_conf.templates is None

        assert boltz_conf.properties is not None
