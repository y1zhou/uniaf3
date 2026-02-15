"""Schemas for UniAF3 input configs."""

from uniaf3.schema.alphafold3 import AF3Config
from uniaf3.schema.alphafold3_server import AF3ServerConfig
from uniaf3.schema.base import UniAF3BaseConfig, UniAF3Config
from uniaf3.schema.boltz import BoltzConfig
from uniaf3.schema.chai import ChaiConfig
from uniaf3.schema.protenix import ProtenixConfig

__all__ = [
    "UniAF3BaseConfig",
    "UniAF3Config",
    "AF3Config",
    "AF3ServerConfig",
    "BoltzConfig",
    "ChaiConfig",
    "ProtenixConfig",
    "dump_config",
    "write_config",
    "AnyConfig",
]

AnyConfig = (
    UniAF3Config
    | AF3Config
    | AF3ServerConfig
    | BoltzConfig
    | ChaiConfig
    | ProtenixConfig
)
