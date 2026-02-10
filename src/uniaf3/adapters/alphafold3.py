"""Adapter for converting between UniAF3Config and AlphaFold3 config."""

from __future__ import annotations

from uniaf3.schema.alphafold3 import (
    AF3Config,
)
from uniaf3.schema.base import (
    UniAF3Config,
)


def to_alphafold3(config: UniAF3Config, name: str = "uniaf3_job") -> AF3Config:
    """Convert a UniAF3Config to an AlphaFold3 config."""
    raise NotImplementedError


def from_alphafold3(config: AF3Config) -> UniAF3Config:
    """Convert an AlphaFold3 config to a UniAF3Config."""
    raise NotImplementedError
