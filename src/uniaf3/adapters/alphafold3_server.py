"""Adapter for converting between UniAF3Config and AlphaFold3 config."""

from __future__ import annotations

from uniaf3.schema.alphafold3_server import (
    AF3ServerConfig,
)
from uniaf3.schema.base import (
    UniAF3Config,
)


def to_alphafold3_server(
    config: UniAF3Config, name: str = "uniaf3_job"
) -> AF3ServerConfig:
    """Convert a UniAF3Config to an AlphaFold3 Server config.

    The server config is simpler – no seeds, no MSA, no templates.
    Ions (detected from known CCD codes) get their own entity type.
    """
    raise NotImplementedError("to_alphafold3_server is not implemented yet")


def from_alphafold3_server(config: AF3ServerConfig) -> UniAF3Config:
    """Convert an AlphaFold3 Server config to a UniAF3Config."""
    raise NotImplementedError("from_alphafold3_server is not implemented yet")
