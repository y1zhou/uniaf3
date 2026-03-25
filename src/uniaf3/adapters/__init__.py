"""Adapters to convert between UniAF3Config and model-specific configs.

Each model has a ``to_*`` and ``from_*`` function pair in its own module under
``uniaf3.adapters``.
"""

from __future__ import annotations

from pathlib import Path

from uniaf3.adapters.alphafold3 import from_alphafold3, to_alphafold3
from uniaf3.adapters.alphafold3_server import (
    from_alphafold3_server,
    to_alphafold3_server,
)
from uniaf3.adapters.boltz import from_boltz, to_boltz
from uniaf3.adapters.chai import from_chai, to_chai
from uniaf3.adapters.protenix import from_protenix, to_protenix
from uniaf3.schema import (
    AF3Config,
    AF3ServerConfig,
    AnyConfig,
    AnyConfigList,
    BoltzConfig,
    ChaiConfig,
    ProtenixConfig,
    UniAF3Config,
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


def to_uniaf3(
    conf: AnyConfig, *, msa_dir: str | Path = "."
) -> UniAF3Config | list[UniAF3Config]:
    """Convert any supported model config to UniAF3Config.

    Args:
        conf: A config object from any supported model format.
        msa_dir: Directory to save MSA files (used by Boltz).

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
        return from_boltz(conf, msa_dir=msa_dir)
    if isinstance(conf, ChaiConfig):
        return from_chai(conf, msa_dir=msa_dir)
    if isinstance(conf, ProtenixConfig):
        return from_protenix(conf)
    raise TypeError(f"Unsupported config type: {type(conf)}")


def from_uniaf3(
    conf: UniAF3Config | list[UniAF3Config],
    target: type[AnyConfig],
    *,
    name: str = "uniaf3_job",
    msa_dir: str | Path = ".",
    strict: bool = False,
) -> AnyConfig | AnyConfigList:
    """Convert a UniAF3Config to a specific model config.

    Args:
        conf: The UniAF3Config to convert.
        target: The target config class.
        name: Job name for models that require one (AF3, Protenix).
        msa_dir: Directory to save MSA CSV files (used by Boltz and Chai).
        strict: If True, raise errors for unsupported features.

    Returns:
        The target model config.

    Raises:
        TypeError: If the target type is not recognized.

    """
    if target is UniAF3Config:
        return conf
    if target is AF3Config:
        if isinstance(conf, list):
            return [to_alphafold3(c, name=name, strict=strict) for c in conf]
        return to_alphafold3(conf, name=name, strict=strict)
    if target is AF3ServerConfig:
        return to_alphafold3_server(conf, name=name, strict=strict)
    if target is BoltzConfig:
        if isinstance(conf, list):
            return [to_boltz(c, msa_dir=msa_dir, strict=strict) for c in conf]
        return to_boltz(conf, msa_dir=msa_dir, strict=strict)
    if target is ChaiConfig:
        if isinstance(conf, list):
            return [to_chai(c, msa_dir=msa_dir, strict=strict) for c in conf]
        return to_chai(conf, msa_dir=msa_dir, strict=strict)
    if target is ProtenixConfig:
        return to_protenix(conf, name=name, strict=strict)
    raise TypeError(f"Unsupported target type: {target}")
