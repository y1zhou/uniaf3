# Protenix Input Format

## Overview

Protenix ([bytedance/Protenix](https://github.com/bytedance/Protenix)) uses a JSON input format closely resembling the AlphaFold Server format, with important extensions for covalent bonds, custom ligands, and structural constraints.

**Reference:** <https://github.com/bytedance/Protenix/blob/main/docs/infer_json_format.md>

## Top-level Structure

The JSON is always a list of job objects, even for a single job:

```json
[
  {
    "name": "Test Fold Job",
    "sequences": [...],
    "covalent_bonds": [...],
    "constraint": {...}
  }
]
```

### Fields

| Field            | Type                  | Required | Description                                 |
| ---------------- | --------------------- | -------- | ------------------------------------------- |
| `name`           | `str`                 | Yes      | Job name. Used for output directory naming. |
| `sequences`      | `list[SequenceEntry]` | Yes      | List of molecular entities.                 |
| `covalent_bonds` | `list[CovalentBond]`  | No       | Inter-entity covalent bonds.                |
| `constraint`     | `{contact, pocket}`   | No       | Structural constraints.                     |

## Sequences

### Protein Chain

```json
{
  "proteinChain": {
    "sequence": "PREACHINGS",
    "count": 2,
    "modifications": [
      {"ptmType": "CCD_HY3", "ptmPosition": 1}
    ],
    "unpairedMsaPath": "/path/to/non_pairing.a3m",
    "pairedMsaPath": "/path/to/pairing.a3m",
    "templatesPath": "/path/to/hmmsearch.a3m"
  }
}
```

| Field             | Type                           | Description                                              |
| ----------------- | ------------------------------ | -------------------------------------------------------- |
| `sequence`        | `str`                          | Amino acid sequence. Standard 20 AA + X (UNK).           |
| `count`           | `int`                          | Number of copies (default: 1).                           |
| `modifications`   | `list[{ptmType, ptmPosition}]` | PTMs. CCD codes are `CCD_`-prefixed.                     |
| `unpairedMsaPath` | `str`                          | Path to unpaired MSA (.a3m). Absolute paths recommended. |
| `pairedMsaPath`   | `str`                          | Path to paired MSA (.a3m).                               |
| `templatesPath`   | `str`                          | Path to template file (.a3m or .hhr).                    |

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

### RNA Sequence

```json
{
  "rnaSequence": {
    "sequence": "GUAC",
    "count": 1,
    "modifications": [
      {"modificationType": "CCD_2MG", "basePosition": 1}
    ],
    "unpairedMsaPath": "/path/to/rna_msa.a3m"
  }
}
```

### Ligand

```json
{"ligand": {"ligand": "CCD_ATP", "count": 1}}
{"ligand": {"ligand": "CCD_NAG_BMA_BGC", "count": 1}}
{"ligand": {"ligand": "CC(=O)OC1C[NH+]2CCC1CC2", "count": 1}}
{"ligand": {"ligand": "FILE_/path/to/molecule.sdf", "count": 1}}
```

| Prefix  | Type      | Description                                                                   |
| ------- | --------- | ----------------------------------------------------------------------------- |
| `CCD_`  | CCD code  | Standard or multi-CCD ligand. Multi-CCD uses `_` separator.                   |
| `FILE_` | File path | Molecular structure file (PDB, SDF, MOL, MOL2). Must include 3D conformation. |
| (none)  | SMILES    | SMILES string.                                                                |

### Ion

```json
{
  "ion": {
    "ion": "MG",
    "count": 2
  }
}
```

Ion CCD codes do NOT have the `CCD_` prefix.

## Covalent Bonds

```json
"covalent_bonds": [
  {
    "entity1": 1,
    "copy1": 1,
    "position1": 2,
    "atom1": "N6",
    "entity2": 3,
    "copy2": 1,
    "position2": 1,
    "atom2": "C1"
  }
]
```

| Field                     | Type          | Description                                                                                                                         |
| ------------------------- | ------------- | ----------------------------------------------------------------------------------------------------------------------------------- |
| `entity1` / `entity2`     | `int`         | 1-based entity number (order in `sequences` list).                                                                                  |
| `copy1` / `copy2`         | `int \| null` | 1-based copy index. Both must be set or both null. If null, bonds created between all copy pairs.                                   |
| `position1` / `position2` | `int`         | 1-based position. For polymers: residue index. For multi-CCD ligands: CCD part index. For single CCD/SMILES/FILE ligands: always 1. |
| `atom1` / `atom2`         | `str`         | Atom name (CCD naming) or atom index (for SMILES/FILE ligands).                                                                     |

**Key difference from other formats:** Protenix uses entity/copy indexing instead of chain IDs.

## Constraints

### Contact Constraint

```json
"constraint": {
  "contact": [
    {
      "entity1": 1, "copy1": 1, "position1": 169,
      "atom1": "CA",
      "entity2": 2, "copy2": 1, "position2": 1,
      "atom2": "C5",
      "max_distance": 6.0,
      "min_distance": 0.0
    }
  ]
}
```

| Field                                 | Type          | Description                                                                    |
| ------------------------------------- | ------------- | ------------------------------------------------------------------------------ |
| `entity1/2`, `copy1/2`, `position1/2` | `int`         | Same indexing as covalent bonds. Copy indices are **required** (not optional). |
| `atom1/2`                             | `str \| null` | Optional. If omitted, constraint applies at token level (central atom).        |
| `max_distance`                        | `float`       | Maximum expected distance in Å.                                                |
| `min_distance`                        | `float`       | Minimum expected distance in Å (default: 0).                                   |

Constraints are **soft**: the model is encouraged but not strictly required to satisfy them.

### Pocket Constraint

```json
"constraint": {
  "pocket": {
    "binder_chain": {"entity": 2, "copy": 1},
    "contact_residues": [
      {"entity": 1, "copy": 1, "position": 126}
    ],
    "max_distance": 6.0
  }
}
```

| Field              | Type                             | Description                                        |
| ------------------ | -------------------------------- | -------------------------------------------------- |
| `binder_chain`     | `{entity, copy}`                 | The binding chain.                                 |
| `contact_residues` | `list[{entity, copy, position}]` | Residues forming the binding pocket.               |
| `max_distance`     | `float`                          | Maximum distance between binder and contacts in Å. |

**Note:** Only one pocket constraint is supported per job.

## Inference Parameters

Protenix inference parameters are specified as CLI arguments:

```bash
python runner/inference.py \
  --input_json_path input.json \
  --dump_dir ./output \
  --seeds 42 \
  --sample_diffusion.N_sample 5
```

Seeds and sampling parameters are not stored in the JSON config.

## UniAF3 Adapter Notes

### UniAF3 → Protenix

- Chain IDs are converted to entity/copy indices based on sequence order.
- Modifications are prefixed with `CCD_`.
- CCD ligands use `CCD_` prefix with `_`-joined multi-CCD codes.
- Glycans are converted to multi-CCD ligands (bond information may be lost).
- Only the first template path is used (Protenix uses `templatesPath` for template discovery).
- Only one pocket constraint is preserved (first one wins).
- Seeds are not stored in Protenix config.

### Protenix → UniAF3

- Entity/copy indexing is converted to sequential chain IDs (A, B, C, ...).
- `CCD_` prefix is stripped from modification and ligand CCD codes.
- FILE-type ligands are not supported and produce a warning.
- MSA paths are not directly preserved (Protenix uses absolute paths; UniAF3 uses hash-based directories).
- Default seed `[42]` is used.
