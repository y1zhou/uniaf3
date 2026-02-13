"""Frozen constants used across the codebase."""

# Support CCD codes by the AlphaFold3 Server
KNOWN_ION_CCD_CODES = frozenset(
    {"MG", "ZN", "CL", "CA", "NA", "MN", "K", "FE", "CU", "CO"}
)
KNOWN_LIGAND_CCD_CODES = frozenset(
    {
        "CCD_ADP",
        "CCD_ATP",
        "CCD_AMP",
        "CCD_GTP",
        "CCD_GDP",
        "CCD_FAD",
        "CCD_NAD",
        "CCD_NAP",
        "CCD_NDP",
        "CCD_HEM",
        "CCD_HEC",
        "CCD_PLM",
        "CCD_OLA",
        "CCD_MYR",
        "CCD_CIT",
        "CCD_CLA",
        "CCD_CHL",
        "CCD_BCL",
        "CCD_BCB",
    }
)
KNOWN_PTM_CCD_CODES = frozenset(
    {
        "CCD_SEP",
        "CCD_TPO",
        "CCD_PTR",
        "CCD_NEP",
        "CCD_HIP",
        "CCD_ALY",
        "CCD_MLY",
        "CCD_M3L",
        "CCD_MLZ",
        "CCD_2MR",
        "CCD_AGM",
        "CCD_MCS",
        "CCD_HYP",
        "CCD_HY3",
        "CCD_LYZ",
        "CCD_AHB",
        "CCD_P1L",
        "CCD_SNN",
        "CCD_SNC",
        "CCD_TRF",
        "CCD_KCR",
        "CCD_CIR",
        "CCD_YHA",
    }
)
KNOWN_RNA_MODIFICATION_CCD_CODES = frozenset(
    {
        "CCD_PSU",
        "CCD_5MC",
        "CCD_OMC",
        "CCD_4OC",
        "CCD_5MU",
        "CCD_OMU",
        "CCD_UR3",
        "CCD_A2M",
        "CCD_MA6",
        "CCD_6MZ",
        "CCD_2MG",
        "CCD_OMG",
        "CCD_7MG",
        "CCD_RSQ",
    }
)
KNOWN_DNA_MODIFICATION_CCD_CODES = frozenset(
    {
        "CCD_5CM",
        "CCD_C34",
        "CCD_5HC",
        "CCD_6OG",
        "CCD_6MA",
        "CCD_1CC",
        "CCD_8OG",
        "CCD_5FC",
        "CCD_3DR",
    }
)
BACKBONE_ATOMS: frozenset[str] = frozenset({"N", "CA", "C", "O", "OXT"})
RESIDUE_ATOMS: dict[str, frozenset[str]] = {
    "A": frozenset({*BACKBONE_ATOMS, "CB"}),
    "R": frozenset({*BACKBONE_ATOMS, "CB", "CG", "CD", "NE", "CZ", "NH1", "NH2"}),
    "N": frozenset({*BACKBONE_ATOMS, "CB", "CG", "OD1", "ND2"}),
    "D": frozenset({*BACKBONE_ATOMS, "CB", "CG", "OD1", "OD2"}),
    "C": frozenset({*BACKBONE_ATOMS, "CB", "SG"}),
    "E": frozenset({*BACKBONE_ATOMS, "CB", "CG", "CD", "OE1", "OE2"}),
    "Q": frozenset({*BACKBONE_ATOMS, "CB", "CG", "CD", "OE1", "NE2"}),
    "G": BACKBONE_ATOMS,
    "H": frozenset({*BACKBONE_ATOMS, "CB", "CG", "ND1", "CD2", "CE1", "NE2"}),
    "I": frozenset({*BACKBONE_ATOMS, "CB", "CG1", "CG2", "CD1"}),
    "L": frozenset({*BACKBONE_ATOMS, "CB", "CG", "CD1", "CD2"}),
    "K": frozenset({*BACKBONE_ATOMS, "CB", "CG", "CD", "CE", "NZ"}),
    "M": frozenset({*BACKBONE_ATOMS, "CB", "CG", "SD", "CE"}),
    "F": frozenset({*BACKBONE_ATOMS, "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ"}),
    "P": frozenset({*BACKBONE_ATOMS, "CB", "CG", "CD"}),
    "S": frozenset({*BACKBONE_ATOMS, "CB", "OG"}),
    "T": frozenset({*BACKBONE_ATOMS, "CB", "OG1", "CG2"}),
    "W": frozenset(
        {
            *BACKBONE_ATOMS,
            "CB",
            "CG",
            "CD1",
            "CD2",
            "NE1",
            "CE2",
            "CE3",
            "CZ2",
            "CZ3",
            "CH2",
        }
    ),
    "Y": frozenset(
        {*BACKBONE_ATOMS, "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ", "OH"}
    ),
    "V": frozenset({*BACKBONE_ATOMS, "CB", "CG1", "CG2"}),
}
