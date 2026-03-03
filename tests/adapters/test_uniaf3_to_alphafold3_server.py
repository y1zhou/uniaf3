"""Tests for UniAF3Config -> AF3ServerConfig adapter."""

import pytest

from uniaf3.schema import AF3ServerConfig, UniAF3Config
from uniaf3.schema.base import Polymer, ProteinSeq


@pytest.fixture(scope="module")
def af3s(uniaf3_conf: UniAF3Config):
    """Convert UniAF3 to AF3 Server config."""
    from uniaf3.adapters import to_alphafold3_server

    with pytest.warns(UserWarning):
        return to_alphafold3_server([uniaf3_conf], name="test", strict=False)


# ruff: noqa: S101
def test_job_metadata(uniaf3_conf: UniAF3Config, af3s: AF3ServerConfig):
    assert len(af3s) == 1
    job = af3s[0]
    assert job.name == "test"
    assert job.dialect == "alphafoldserver"
    assert job.modelSeeds == uniaf3_conf.aux.seeds


def test_warns_on_chain_id_loss(uniaf3_conf: UniAF3Config):
    from uniaf3.adapters import to_alphafold3_server

    with pytest.warns(UserWarning) as records:
        _ = to_alphafold3_server([uniaf3_conf], strict=False)
    assert any("UniAF3Config.sequences[*].id" in str(w.message) for w in records)


def test_protein_count(uniaf3_conf: UniAF3Config, af3s: AF3ServerConfig):
    prot = af3s[0].sequences[0].proteinChain
    src = uniaf3_conf.sequences[0]
    assert isinstance(src, ProteinSeq)
    assert prot is not None
    assert prot.sequence == src.sequence
    # id ["A", "B"] → count=2
    assert prot.count == 2


def test_protein_modifications_prefixed(
    uniaf3_conf: UniAF3Config, af3s: AF3ServerConfig
):
    prot = af3s[0].sequences[0].proteinChain
    src = uniaf3_conf.sequences[0]
    assert isinstance(src, ProteinSeq)
    assert prot is not None
    assert prot.modifications is not None
    assert src.modifications is not None
    # Server uses CCD_ prefix
    assert prot.modifications[0].ptmType == f"CCD_{src.modifications[0].ccd}"


def test_dna_fields(uniaf3_conf: UniAF3Config, af3s: AF3ServerConfig):
    dna = af3s[0].sequences[1].dnaSequence
    src = uniaf3_conf.sequences[1]
    assert isinstance(src, Polymer)
    assert dna is not None
    assert dna.sequence == src.sequence
