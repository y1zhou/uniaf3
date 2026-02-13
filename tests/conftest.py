"""Fixtures shared across pytest sessions."""

from pathlib import Path

import pytest

from uniaf3.schema import (
    AF3Config,
    AF3ServerConfig,
    BoltzConfig,
    ChaiConfig,
    ProtenixConfig,
    UniAF3Config,
)

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def uniaf3_conf():
    """Load the UniAF3 example."""
    return UniAF3Config.from_file(FIXTURES / "uniaf3_example.yaml")


@pytest.fixture(scope="session")
def af3_conf():
    """Load the AlphaFold3 example."""
    return AF3Config.from_file(FIXTURES / "alphafold3_example.json")


@pytest.fixture(scope="session")
def af3_server_confs():
    """Load the AlphaFold3 Server example."""
    return AF3ServerConfig.from_file(FIXTURES / "alphafold3_server_example.json")


@pytest.fixture(scope="session")
def protenix_confs():
    """Load the Protenix example."""
    return ProtenixConfig.from_file(FIXTURES / "protenix_example.json")


@pytest.fixture(scope="session")
def chai_conf():
    """Load the Chai example."""
    return ChaiConfig.from_file(FIXTURES / "chai_example.yaml")


@pytest.fixture(scope="session")
def boltz_conf():
    """Load the Boltz example."""
    return BoltzConfig.from_file(FIXTURES / "boltz_example.yaml")
