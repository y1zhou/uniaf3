"""Backward-compatible re-exports from the new ``uniaf3.adapters`` package.

.. deprecated:: 0.2.0
    Import from ``uniaf3.adapters`` instead.
"""

from uniaf3.adapters import (
    from_alphafold3,
    from_boltz,
    from_chai,
    from_protenix,
    to_alphafold3,
    to_boltz,
    to_chai,
    to_protenix,
)

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
