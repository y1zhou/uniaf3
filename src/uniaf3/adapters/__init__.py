"""Adapters to convert between UniAF3Config and model-specific configs.

Each model has a ``to_*`` and ``from_*`` function pair in its own module under
``uniaf3.adapters``.  Items that cannot be mapped are annotated with
``# NOTE: `` comments for future attention.
"""

from __future__ import annotations

from uniaf3.adapters.alphafold3 import (
    from_alphafold3,
    from_alphafold3_server,
    to_alphafold3,
    to_alphafold3_server,
)
from uniaf3.adapters.boltz import from_boltz, to_boltz
from uniaf3.adapters.chai import from_chai, to_chai
from uniaf3.adapters.protenix import from_protenix, to_protenix
from uniaf3.schema import UniAF3Config
from uniaf3.schema.alphafold3 import AF3Config, AF3ServerConfig
from uniaf3.schema.boltz import BoltzConfig
from uniaf3.schema.chai import ChaiConfig
from uniaf3.schema.protenix import ProtenixConfig

AnyConfig = (
    UniAF3Config
    | AF3Config
    | AF3ServerConfig
    | BoltzConfig
    | ChaiConfig
    | ProtenixConfig
)

__all__ = [
    "from_alphafold3",
    "from_alphafold3_server",
    "from_boltz",
    "from_chai",
    "from_protenix",
    "from_uniaf3",
    "to_alphafold3",
    "to_alphafold3_server",
    "to_boltz",
    "to_chai",
    "to_protenix",
    "to_uniaf3",
]


def to_uniaf3(conf: AnyConfig) -> UniAF3Config:
    """Convert any supported model config to UniAF3Config.

    Args:
        conf: A config object from any supported model format.

    Returns:
        The equivalent UniAF3Config.

    Raises:
        TypeError: If the config type is not recognized.

    """
    if isinstance(conf, UniAF3Config):
        return conf
    if isinstance(conf, AF3ServerConfig):
        return from_alphafold3_server(conf)
    if isinstance(conf, AF3Config):
        return from_alphafold3(conf)
    if isinstance(conf, BoltzConfig):
        return from_boltz(conf)
    if isinstance(conf, ChaiConfig):
        return from_chai(conf)
    if isinstance(conf, ProtenixConfig):
        return from_protenix(conf)
    raise TypeError(f"Unsupported config type: {type(conf)}")


def from_uniaf3(
    conf: UniAF3Config,
    target: type[AnyConfig],
    *,
    name: str = "uniaf3_job",
) -> AnyConfig:
    """Convert a UniAF3Config to a specific model config.

    Args:
        conf: The UniAF3Config to convert.
        target: The target config class.
        name: Job name for models that require one (AF3, Protenix).

    Returns:
        The target model config.

    Raises:
        TypeError: If the target type is not recognized.

    """
    if target is UniAF3Config:
        return conf
    if target is AF3Config:
        return to_alphafold3(conf, name=name)
    if target is AF3ServerConfig:
        return to_alphafold3_server(conf, name=name)
    if target is BoltzConfig:
        return to_boltz(conf)
    if target is ChaiConfig:
        return to_chai(conf)
    if target is ProtenixConfig:
        return to_protenix(conf, name=name)
    raise TypeError(f"Unsupported target type: {target}")
