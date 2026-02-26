# AlphaFold3 Server Input Format

## Overview

The AlphaFold Server ([alphafoldserver.com](https://alphafoldserver.com/)) uses a simpler JSON format compared to the open-source AlphaFold3 format. Key differences:

- No custom MSA or templates.
- No user-provided CCD or SMILES ligands.
- Ions have a dedicated entity type (separate from ligands).
- Glycans are specified as modifications on protein chains, not standalone entities.
- Uses `count` instead of explicit chain IDs.
- Supports multiple jobs in a single JSON (top-level is a list).
- Limited set of supported CCD codes for ligands, ions, PTMs, and nucleotide modifications.

**Reference:** <https://github.com/google-deepmind/alphafold/blob/main/server/README.md>

## Top-level Structure

The top-level is always a JSON array of job objects:

```json
[
  {
    "name": "Job name",
    "modelSeeds": [],
    "sequences": [...],
    "dialect": "alphafoldserver",
    "version": 1
  }
]
```

### Fields

| Field | Type | Required | Description |
|---|---|---|---|
| `name` | `str` | Yes | Job name. |
| `modelSeeds` | `list[int]` | Yes | Random seeds. Can be empty (a seed will be generated). |
| `sequences` | `list[SequenceEntry]` | Yes | List of entities. |
| `dialect` | `"alphafoldserver"` | Yes | Must be `"alphafoldserver"`. |
| `version` | `1` | Yes | Must be `1`. |

## Sequences

Each entry must contain exactly one of: `proteinChain`, `dnaSequence`, `rnaSequence`, `ligand`, or `ion`.

### Protein Chain

```json
{
  "proteinChain": {
    "sequence": "PREACHINGS",
    "count": 2,
    "glycans": [
      {"residues": "NAG(NAG)(BMA)", "position": 8}
    ],
    "modifications": [
      {"ptmType": "CCD_HY3", "ptmPosition": 1}
    ],
    "useStructureTemplate": true,
    "maxTemplateDate": "2024-05-08"
  }
}
```

| Field | Type | Description |
|---|---|---|
| `sequence` | `str` | Amino acid sequence (20 standard AA codes only). |
| `count` | `int` | Number of homomeric copies (default: 1). |
| `glycans` | `list[{residues, position}]` | Glycan attachments using Chai notation + 1-based position. |
| `modifications` | `list[{ptmType, ptmPosition}]` | PTMs with `CCD_`-prefixed CCD codes. Limited to a known set. |
| `useStructureTemplate` | `bool` | Whether to use PDB templates (default: true). |
| `maxTemplateDate` | `date` | Upper date limit for PDB templates (1976-01-01 to 2025-02-03). |

**Supported PTM CCD Codes:**
`CCD_SEP`, `CCD_TPO`, `CCD_PTR`, `CCD_NEP`, `CCD_HIP`, `CCD_ALY`, `CCD_MLY`, `CCD_M3L`, `CCD_MLZ`, `CCD_2MR`, `CCD_AGM`, `CCD_MCS`, `CCD_HYP`, `CCD_HY3`, `CCD_LYZ`, `CCD_AHB`, `CCD_P1L`, `CCD_SNN`, `CCD_SNC`, `CCD_TRF`, `CCD_KCR`, `CCD_CIR`, `CCD_YHA`

### DNA Sequence

```json
{
  "dnaSequence": {
    "sequence": "GATTACA",
    "count": 1,
    "modifications": [
      {"modificationType": "CCD_6OG", "basePosition": 1}
    ]
  }
}
```

**Supported DNA Modification CCD Codes:**
`CCD_5CM`, `CCD_C34`, `CCD_5HC`, `CCD_6OG`, `CCD_6MA`, `CCD_1CC`, `CCD_8OG`, `CCD_5FC`, `CCD_3DR`

### RNA Sequence

```json
{
  "rnaSequence": {
    "sequence": "GUAC",
    "count": 1,
    "modifications": [
      {"modificationType": "CCD_2MG", "basePosition": 1}
    ]
  }
}
```

**Supported RNA Modification CCD Codes:**
`CCD_PSU`, `CCD_5MC`, `CCD_OMC`, `CCD_4OC`, `CCD_5MU`, `CCD_OMU`, `CCD_UR3`, `CCD_A2M`, `CCD_MA6`, `CCD_6MZ`, `CCD_2MG`, `CCD_OMG`, `CCD_7MG`, `CCD_RSQ`

### Ligand

```json
{
  "ligand": {
    "ligand": "CCD_ATP",
    "count": 1
  }
}
```

**Supported Ligand CCD Codes:**
`CCD_ADP`, `CCD_ATP`, `CCD_AMP`, `CCD_GTP`, `CCD_GDP`, `CCD_FAD`, `CCD_NAD`, `CCD_NAP`, `CCD_NDP`, `CCD_HEM`, `CCD_HEC`, `CCD_PLM`, `CCD_OLA`, `CCD_MYR`, `CCD_CIT`, `CCD_CLA`, `CCD_CHL`, `CCD_BCL`, `CCD_BCB`

### Ion

```json
{
  "ion": {
    "ion": "MG",
    "count": 2
  }
}
```

**Supported Ion CCD Codes:**
`MG`, `ZN`, `CL`, `CA`, `NA`, `MN`, `K`, `FE`, `CU`, `CO`

Note: Ion CCD codes do NOT have the `CCD_` prefix.

## Limitations

- **No covalent bonds**: The server format does not support specifying covalent bonds.
- **No restraints**: No contact or pocket restraints.
- **No custom MSA or templates**: The server handles these automatically.
- **No SMILES ligands**: Only predefined CCD ligands are supported.
- **No multi-CCD ligands**: Each ligand entry has a single CCD code.
- **No chain IDs**: Chain IDs are auto-assigned by the server.

## UniAF3 Adapter Mapping

### UniAF3 → AF3 Server

| UniAF3 Feature | AF3 Server Mapping | Notes |
|---|---|---|
| `ProteinSeq` / `Polymer(protein)` | `proteinChain` | `count` derived from `len(ids)`. MSA/templates dropped. |
| `Polymer(dna)` | `dnaSequence` | Modifications prefixed with `CCD_`. |
| `Polymer(rna)` | `rnaSequence` | Modifications prefixed with `CCD_`. |
| `Ligand` (known ion CCD) | `ion` entry | Auto-detected from `KNOWN_ION_CCD_CODES`. |
| `Ligand` (known ligand CCD) | `ligand` entry | Only supported CCD codes accepted. |
| `Ligand` (SMILES) | **Dropped** | Not supported. |
| `Glycan` | **Dropped** | Standalone glycans not supported. |
| All restraints | **Dropped** | Not supported. |

### AF3 Server → UniAF3

Chain IDs are auto-generated sequentially (A, B, ..., Z, AA, BA, ...) based on entity order and copy counts.
