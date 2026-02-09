"""Adapters to convert between UniAF3Config and model-specific configs.

Each model has a ``to_*`` and ``from_*`` function pair in its own module under
``uniaf3.adapters``.  Items that cannot be mapped are annotated with
``# NOTE: `` comments for future attention.
"""

from uniaf3.adapters.alphafold3 import from_alphafold3, to_alphafold3
from uniaf3.adapters.boltz import from_boltz, to_boltz
from uniaf3.adapters.chai import from_chai, to_chai
from uniaf3.adapters.protenix import from_protenix, to_protenix

__all__ = [
    "from_alphafold3",
    "from_boltz",
    "from_chai",
    "from_protenix",
    "to_alphafold3",
    "to_boltz",
    "to_chai",
    "to_protenix",
]
