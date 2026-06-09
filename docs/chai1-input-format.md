# Chai-1 Input Format

## Overview

Chai-1 ([chaidiscovery/chai-lab](https://github.com/chaidiscovery/chai-lab)) uses a multi-entity FASTA file for sequences and an optional CSV file for restraints. This makes it the most different input format among all supported models.

**References:**

- <https://github.com/chaidiscovery/chai-lab/blob/main/chai_lab/chai1.py>
- <https://github.com/chaidiscovery/chai-lab/tree/main/examples/restraints/README.md>
- <https://github.com/chaidiscovery/chai-lab/tree/main/examples/covalent_bonds/README.md>

## FASTA Input

The input FASTA file uses structured headers in the format `>entity_type|entity_name`:

```fasta
>protein|Hemoglobin subunit
MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLS
>protein|Modified protein
(HY3)VLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLS
>dna|example DNA
GATTACA
>ligand|CCD example
ATP
>ligand|SMILES example
CC(=O)OC1C[NH+]2CCC1CC2
>glycan|multi-ring glycan
NAG(NAG)(BMA)
```

### Entity Types

| Type      | Sequence Content                                       | Description                                            |
| --------- | ------------------------------------------------------ | ------------------------------------------------------ |
| `protein` | Amino acid sequence with optional inline modifications | Standard 1-letter codes. Modifications in parentheses. |
| `dna`     | DNA nucleotide sequence                                | A, T, G, C.                                            |
| `rna`     | RNA nucleotide sequence                                | A, U, G, C.                                            |
| `ligand`  | CCD code or SMILES string                              | Single CCD code (3 letters) or full SMILES.            |
| `glycan`  | Chai glycan notation                                   | CCD codes with bond syntax.                            |

### Inline Modifications

Modifications are specified inline within the sequence using parenthesized CCD codes:

```
MVLS(HY3)ADKTNVK
```

This indicates that position 5 is modified from the canonical residue to `HY3`. The CCD code must be more than 1 character long.

### Chain ID Assignment

Chai-1 assigns chain IDs alphabetically (A, B, C, ..., Z, AA, AB, ...) based on the order entities appear in the FASTA file. Each entity gets its own chain ID—there is **no concept of homomeric copies** in Chai-1's input format.

## Restraints CSV

Restraints are provided in a separate CSV file with the following columns:

| Column                  | Type           | Description                                                                                              |
| ----------------------- | -------------- | -------------------------------------------------------------------------------------------------------- |
| `restraint_id`          | `str`          | Unique identifier for the restraint.                                                                     |
| `chainA`                | `str`          | Chain ID of the first entity (auto-assigned A-Z order).                                                  |
| `res_idxA`              | `str \| empty` | Residue index for chain A. Format: `{residue_name}{position}[@atom_name]`. Empty for pocket binder side. |
| `chainB`                | `str`          | Chain ID of the second entity.                                                                           |
| `res_idxB`              | `str`          | Residue index for chain B. Same format as `res_idxA`.                                                    |
| `connection_type`       | `str`          | One of: `contact`, `pocket`, `covalent`.                                                                 |
| `confidence`            | `float`        | Confidence score (currently unused by the model).                                                        |
| `min_distance_angstrom` | `float`        | Minimum distance in Å (currently unused by the model).                                                   |
| `max_distance_angstrom` | `float`        | Maximum distance in Å.                                                                                   |
| `comment`               | `str`          | Optional comment (not used by the model).                                                                |

### Residue Index Format

The `res_idx` fields use a specific format depending on the restraint type:

- **Contact restraints:** `{residue_name}{position}` (e.g., `R84`, `A219`)
  - Only polymers (protein/DNA/RNA) are supported.
- **Covalent bonds:** `{residue_name}{position}@{atom_name}` for polymers (e.g., `N436@N`)
  - For ligands/glycans: `@{atom_name}` (no residue prefix, e.g., `@C1`)
- **Pocket restraints:** Same as contact for the pocket side; empty for the binder side.

### Example CSV

```csv
restraint_id,chainA,res_idxA,chainB,res_idxB,connection_type,confidence,min_distance_angstrom,max_distance_angstrom,comment
restraint0,A,P5@CG,F,@C1,covalent,1.0,0.0,6.0,protein-glycan bond
restraint1,A,V11,B,L35,contact,1.0,0.0,6.0,interface contact
restraint2,B,,A,A14,pocket,1.0,0.0,8.0,binding pocket
```

## Glycan Notation

Chai-1 uses a specialized notation for glycans:

### Single-ring

```
NAG
```

### Multi-ring with bonds

```
NAG(4-1 NAG)
```

This means: root `NAG` connected to a second `NAG` via a bond between O4 of the first sugar and C1 of the second.

### Branched glycans

```
NAG(4-1 NAG(4-1 BMA(3-1 MAN)(6-1 MAN)))
```

The bond notation `X-Y` creates an O-glycosidic bond: O{X} on the source sugar → C{Y} on the destination sugar.

### Glycan-protein attachment

Glycan-protein bonds are specified in the restraints CSV as covalent bonds, not in the glycan notation.

## MSA Support

Chai-1 uses the ColabFold server or precomputed MSA files:

```python
run_inference(
    fasta_file="input.fasta",
    msa_directory="path/to/msa/",
    use_msa_server=True,
    msa_server_url="https://api.colabfold.com",
)
```

The MSA directory should contain A3M files named by sequence hash:

```
msa_dir/
  a3ms/
    {sha256_hash}.single.a3m
    {sha256_hash}.pair.a3m
```

## Inference Parameters

Chai-1 accepts inference parameters as function arguments to `run_inference()`:

| Parameter             | Default | Description                              |
| --------------------- | ------- | ---------------------------------------- |
| `num_trunk_recycles`  | 3       | Number of recycling steps.               |
| `num_diffn_timesteps` | 200     | Number of diffusion timesteps.           |
| `num_diffn_samples`   | 5       | Number of diffusion samples.             |
| `num_trunk_samples`   | 1       | >1 adds to seed for multiple trunk runs. |
| `seed`                | None    | Random seed.                             |
| `use_esm_embeddings`  | True    | Whether to use ESM embeddings.           |

## UniAF3 Adapter Notes

### UniAF3 → Chai-1

- Each chain ID becomes a separate entity (homomeric copies are expanded).
- Modifications are inlined as `(CCD)` tokens in the sequence.
- CCD ligands are converted to SMILES via the CCD library lookup.
- Multi-CCD ligands are not supported and produce a warning.
- Chain IDs are remapped to sequential A, B, C, ... order.
- Restraint chain IDs are updated to match the new mapping.
- `residue_name` is required for covalent bond and contact restraint atoms.

### Chai-1 → UniAF3

- Inline modifications are parsed out from sequences.
- Modified residues are mapped to canonical 1-letter codes using gemmi's CCD lookup.
- If no canonical mapping exists, `X` (unknown) is used.
- Ligands are stored with SMILES (CCD codes lost in FASTA format).
- Pocket restraints with the same binder chain are merged into a single `PocketRestraint`.
- MSA directory is preserved if provided.
