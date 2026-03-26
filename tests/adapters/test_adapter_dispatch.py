"""Tests for to_uniaf3 and from_uniaf3 dispatch functions."""

import pytest

from uniaf3.schema import (
    AF3Config,
    AF3ServerConfig,
    BoltzConfig,
    ChaiConfig,
    ProtenixConfig,
    UniAF3Config,
)
from uniaf3.schema.base import PolymerType, ProteinSeq


# ruff: noqa: S101


@pytest.fixture
def simple_uniaf3():
    return UniAF3Config(
        sequences=[
            ProteinSeq(
                polymer_type=PolymerType.Protein,
                id="A",
                sequence="MVLSPADKTNVK",
            )
        ]
    )


def test_to_uniaf3_returns_same_config(simple_uniaf3):
    """to_uniaf3 with a UniAF3Config should return the same object."""
    from uniaf3.adapters import to_uniaf3

    result = to_uniaf3(simple_uniaf3)
    assert result is simple_uniaf3


def test_to_uniaf3_from_af3(af3_conf):
    """to_uniaf3 should dispatch AF3Config to from_alphafold3."""
    from uniaf3.adapters import to_uniaf3

    result = to_uniaf3(af3_conf)
    assert isinstance(result, UniAF3Config)


def test_to_uniaf3_from_af3_server(af3_server_confs):
    """to_uniaf3 should dispatch AF3ServerConfig to from_alphafold3_server."""
    from uniaf3.adapters import to_uniaf3

    result = to_uniaf3(af3_server_confs)
    assert isinstance(result, (UniAF3Config, list))


def test_to_uniaf3_from_protenix(protenix_confs):
    """to_uniaf3 should dispatch ProtenixConfig to from_protenix."""
    from uniaf3.adapters import to_uniaf3

    result = to_uniaf3(protenix_confs)
    assert isinstance(result, (UniAF3Config, list))


def test_to_uniaf3_from_boltz(boltz_conf, tmp_path):
    """to_uniaf3 should dispatch BoltzConfig to from_boltz."""
    from uniaf3.adapters import to_uniaf3

    with pytest.warns(UserWarning):
        result = to_uniaf3(boltz_conf, msa_dir=tmp_path)
    assert isinstance(result, UniAF3Config)


def test_to_uniaf3_from_chai(chai_conf):
    """to_uniaf3 should dispatch ChaiConfig to from_chai."""
    from uniaf3.adapters import to_uniaf3

    with pytest.warns(UserWarning):
        result = to_uniaf3(chai_conf)
    assert isinstance(result, UniAF3Config)


def test_to_uniaf3_unsupported_type_raises():
    """to_uniaf3 with unsupported config type should raise TypeError."""
    from uniaf3.adapters import to_uniaf3

    with pytest.raises(TypeError, match="Unsupported config type"):
        to_uniaf3("not a config")  # type: ignore[arg-type]


def test_from_uniaf3_returns_same_config(simple_uniaf3):
    """from_uniaf3 with UniAF3Config target should return the same object."""
    from uniaf3.adapters import from_uniaf3

    result = from_uniaf3(simple_uniaf3, UniAF3Config)
    assert result is simple_uniaf3


def test_from_uniaf3_list_returns_same_list(simple_uniaf3):
    """from_uniaf3 with UniAF3Config target and list input should return the list."""
    from uniaf3.adapters import from_uniaf3

    configs = [simple_uniaf3]
    result = from_uniaf3(configs, UniAF3Config)
    assert result is configs


def test_from_uniaf3_list_to_af3(simple_uniaf3):
    """from_uniaf3 with list input and AF3Config target should convert each item."""
    from uniaf3.adapters import from_uniaf3

    result = from_uniaf3([simple_uniaf3], AF3Config)
    assert isinstance(result, list)
    assert len(result) == 1
    assert isinstance(result[0], AF3Config)


def test_from_uniaf3_list_to_boltz(simple_uniaf3, tmp_path):
    """from_uniaf3 with list input and BoltzConfig target should convert each item."""
    from uniaf3.adapters import from_uniaf3

    result = from_uniaf3([simple_uniaf3], BoltzConfig, msa_dir=tmp_path)
    assert isinstance(result, list)
    assert len(result) == 1
    assert isinstance(result[0], BoltzConfig)


def test_from_uniaf3_list_to_chai(simple_uniaf3, tmp_path):
    """from_uniaf3 with list input and ChaiConfig target should convert each item."""
    from uniaf3.adapters import from_uniaf3

    result = from_uniaf3([simple_uniaf3], ChaiConfig, msa_dir=tmp_path)
    assert isinstance(result, list)
    assert len(result) == 1
    assert isinstance(result[0], ChaiConfig)


def test_from_uniaf3_to_af3server(simple_uniaf3):
    """from_uniaf3 with AF3ServerConfig target should use to_alphafold3_server."""
    from uniaf3.adapters import from_uniaf3

    result = from_uniaf3(simple_uniaf3, AF3ServerConfig)
    assert isinstance(result, AF3ServerConfig)


def test_from_uniaf3_to_protenix(simple_uniaf3):
    """from_uniaf3 with ProtenixConfig target should use to_protenix."""
    from uniaf3.adapters import from_uniaf3

    result = from_uniaf3(simple_uniaf3, ProtenixConfig)
    assert isinstance(result, ProtenixConfig)


def test_from_uniaf3_unsupported_target_raises(simple_uniaf3):
    """from_uniaf3 with unsupported target type should raise TypeError."""
    from uniaf3.adapters import from_uniaf3

    with pytest.raises(TypeError, match="Unsupported target type"):
        from_uniaf3(simple_uniaf3, str)  # type: ignore[arg-type]


def test_from_uniaf3_single_to_af3(simple_uniaf3):
    """from_uniaf3 with single config and AF3Config target."""
    from uniaf3.adapters import from_uniaf3

    result = from_uniaf3(simple_uniaf3, AF3Config)
    assert isinstance(result, AF3Config)


def test_from_uniaf3_single_to_boltz(simple_uniaf3, tmp_path):
    """from_uniaf3 with single config and BoltzConfig target."""
    from uniaf3.adapters import from_uniaf3

    result = from_uniaf3(simple_uniaf3, BoltzConfig, msa_dir=tmp_path)
    assert isinstance(result, BoltzConfig)


def test_from_uniaf3_single_to_chai(simple_uniaf3, tmp_path):
    """from_uniaf3 with single config and ChaiConfig target."""
    from uniaf3.adapters import from_uniaf3

    result = from_uniaf3(simple_uniaf3, ChaiConfig, msa_dir=tmp_path)
    assert isinstance(result, ChaiConfig)
