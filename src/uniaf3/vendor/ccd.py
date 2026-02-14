"""Load CCD and SMILES mapping downloaded from wwPDB.

Refs:

<https://www.wwpdb.org/data/ccd>
<https://files.wwpdb.org/pub/pdb/data/monomers/Components-smiles-stereo-oe.smi>
"""

from pathlib import Path

import polars as pl

CCD_LIB = pl.read_csv(
    Path(__file__).parent / "Components-smiles-stereo-oe.smi",
    separator="\t",
    has_header=False,
    new_columns=["SMILES", "CCD", "name"],
    quote_char=None,
)
