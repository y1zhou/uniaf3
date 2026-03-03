# Boltz Input Format

## Overview

Boltz ([jwohlwend/boltz](https://github.com/jwohlwend/boltz)) uses a YAML-based input format for structure prediction. It supports proteins, DNA, RNA, ligands, covalent bonds, contact constraints, pocket constraints, structural templates, and affinity prediction.

**Reference:** <https://github.com/jwohlwend/boltz/blob/main/docs/prediction.md>

## Top-level Structure

```yaml
version: 1
sequences:
  - protein:
      id: [A, B]
      sequence: MVTPEGNVSLV...
      msa: path/to/msa.csv
      modifications:
        - position: 5
          ccd: HY3
      cyclic: false
  - ligand:
      id: C
      ccd: SAH
constraints:
  - bond:
      atom1: [A, 111, SG]
      atom2: [C, 1, SD]
  - contact:
      token1: [A, 10]
      token2: [C, C4]
      max_distance: 8.0
      force: true
  - pocket:
      binder: C
      contacts:
        - [A, 140]
        - [A, 145]
      max_distance: 6.0
      force: false
templates:
  - cif: path/to/template.cif
    chain_id: A
    force: true
    threshold: 2.0
properties:
  - affinity:
      binder: C
```

## Sequences

### Protein

| Field           | Type                    | Required    | Description                                                                                                                      |
| --------------- | ----------------------- | ----------- | -------------------------------------------------------------------------------------------------------------------------------- |
| `id`            | `str \| list[str]`      | Yes         | Unique chain ID(s). List for homomeric copies.                                                                                   |
| `sequence`      | `str`                   | Yes         | Amino acid sequence.                                                                                                             |
| `msa`           | `str`                   | Conditional | Path to MSA file (`.a3m` or `.csv`). Set to `"empty"` for single-sequence mode. Required unless `--use_msa_server` flag is used. |
| `modifications` | `list[{position, ccd}]` | No          | Modified residues with 1-based position and CCD code.                                                                            |
| `cyclic`        | `bool`                  | No          | Whether the polymer is cyclic (default: false).                                                                                  |

#### MSA Format

Boltz supports two MSA file formats:

1. **A3M format** (`.a3m`): Standard MSA format (FASTA with gap characters).
2. **CSV format** (`.csv`): Two columns - `sequence` and `key`. Sequences with the same `key` are mutually aligned (used for multi-chain pairing).

For multi-chain inputs with multiple protein chains, use CSV format to enable cross-chain MSA pairing.

### DNA

| Field           | Type                    | Required | Description                |
| --------------- | ----------------------- | -------- | -------------------------- |
| `id`            | `str \| list[str]`      | Yes      | Chain ID(s).               |
| `sequence`      | `str`                   | Yes      | DNA sequence (A, T, G, C). |
| `modifications` | `list[{position, ccd}]` | No       | Modified bases.            |
| `cyclic`        | `bool`                  | No       | Cyclic polymer flag.       |

### RNA

Same structure as DNA but uses RNA bases (A, U, G, C).

### Ligand

| Field    | Type               | Required    | Description                                        |
| -------- | ------------------ | ----------- | -------------------------------------------------- |
| `id`     | `str \| list[str]` | Yes         | Chain ID(s).                                       |
| `ccd`    | `str`              | Conditional | Single CCD code. Mutually exclusive with `smiles`. |
| `smiles` | `str`              | Conditional | SMILES string.                                     |

**Note:** Boltz supports only single-CCD ligands (no multi-CCD). For multi-component ligands, use SMILES instead.

## Constraints

### Bond Constraint

Covalent bonds between two atoms:

```yaml
- bond:
    atom1: [CHAIN_ID, RES_IDX, ATOM_NAME]
    atom2: [CHAIN_ID, RES_IDX, ATOM_NAME]
```

- `RES_IDX` is 1-based. For ligands, use `1`.
- `ATOM_NAME` follows CCD naming for standard residues and CCD ligands.

### Contact Constraint

Distance constraint between two residues/atoms:

```yaml
- contact:
    token1: [CHAIN_ID, RES_IDX_OR_ATOM_NAME]
    token2: [CHAIN_ID, RES_IDX_OR_ATOM_NAME]
    max_distance: 6.0
    force: false
```

- For polymers: use residue index (integer, 1-based).
- For ligands: use atom name (string).
- `max_distance`: 4-20 Å (default: 6.0).
- `force`: If true, uses a potential to enforce the constraint at inference time.

### Pocket Constraint

```yaml
- pocket:
    binder: CHAIN_ID
    contacts: [[CHAIN_ID, RES_IDX_OR_ATOM_NAME], ...]
    max_distance: 6.0
    force: false
```

- `binder`: Chain ID of the binding molecule.
- `contacts`: List of `[chain_id, residue_index_or_atom_name]` tuples forming the pocket.
- Same index/atom rules as contact constraints.

## Templates

```yaml
templates:
  - cif: path/to/template.cif      # or pdb: path/to/template.pdb
    chain_id: A                      # Optional: which chain(s) to template
    template_id: A                   # Optional: explicit template chain mapping
    force: true                      # Optional: enforce template with potential
    threshold: 2.0                   # Optional: max deviation in Å (required if force=true)
```

- Exactly one of `cif` or `pdb` must be provided.
- If `chain_id` is not specified, Boltz finds the best matching chains.
- PDB templates use incremental chain IDs (A1, A2, B1, etc.).

## Properties

### Affinity

```yaml
properties:
  - affinity:
      binder: C    # Chain ID of the small molecule
```

- Only one ligand chain can be specified for affinity computation.
- Must be a ligand chain (not protein/DNA/RNA).
- Best for ligands ≤ 56 heavy atoms (training limit), max 128 atoms.
- Only reliable for protein targets.

## Inference Parameters

Boltz inference parameters are specified as CLI arguments, not in the YAML config:

| Parameter             | Default | Description                                                |
| --------------------- | ------- | ---------------------------------------------------------- |
| `--recycling_steps`   | 3       | Number of recycling steps.                                 |
| `--sampling_steps`    | 200     | Number of diffusion sampling steps.                        |
| `--diffusion_samples` | 1       | Number of diffusion samples.                               |
| `--step_scale`        | 1.638   | Diffusion step size (temperature). Lower = more diversity. |

## UniAF3 Adapter Notes

- Boltz config has no seeds; default `[42]` is used on import to UniAF3.
- Multi-CCD UniAF3 ligands produce a warning (only single-CCD supported in Boltz).
- Glycans are converted to single-CCD ligands if possible, or skipped with a warning.
- UniAF3 MSA files (Chai-style A3M) are merged into Boltz CSV format during conversion.
- Templates are separate top-level entries in Boltz (not nested under proteins).
