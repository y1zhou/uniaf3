# UniAF3 Input Format

## Overview

UniAF3 provides a unified YAML-based input format that serves as a common intermediate representation for converting between different AlphaFold3-family structure prediction models. The format supports specifying molecular sequences, restraints, and inference parameters in a single configuration file.

### Supported Models

The following table summarizes feature support across all models:

| Feature | UniAF3 | AlphaFold3 | AF3 Server | Boltz | Chai-1 | Protenix |
|---|---|---|---|---|---|---|
| **Sequences** | | | | | | |
| Protein chains | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| DNA chains | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| RNA chains | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Ligands (CCD) | ✅ | ✅ | ✅ (limited set) | ✅ (single CCD only) | ⚠️ (converted to SMILES) | ✅ (multi-CCD supported) |
| Ligands (SMILES) | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ |
| Ligands (file path) | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |
| Multi-CCD ligands | ✅ | ✅ | ❌ | ❌ | ❌ | ✅ |
| Glycans | ✅ (Chai notation) | ❌ | ❌ | ⚠️ (single sugar only) | ✅ | ⚠️ (as multi-CCD ligand) |
| Ions | ✅ (as CCD ligand) | ✅ (as CCD ligand) | ✅ (dedicated type) | ✅ (as CCD ligand) | ❌ | ✅ (dedicated type) |
| Homomeric copies | ✅ (via id list) | ✅ (via id list) | ✅ (via count) | ✅ (via id list) | ❌ (separate entities) | ✅ (via count) |
| **Modifications** | | | | | | |
| Protein PTMs | ✅ | ✅ | ✅ (limited CCD set) | ✅ | ✅ (inline CCD) | ✅ |
| DNA modifications | ✅ | ✅ | ✅ (limited CCD set) | ✅ | ✅ (inline CCD) | ✅ |
| RNA modifications | ✅ | ✅ | ✅ (limited CCD set) | ✅ | ✅ (inline CCD) | ✅ |
| Cyclic polymers | ✅ (Boltz-specific) | ❌ | ❌ | ✅ | ❌ | ❌ |
| **MSA & Templates** | | | | | | |
| Custom MSA | ✅ (via msa_dir) | ✅ (inline or path) | ❌ | ✅ (CSV or A3M) | ✅ (via msa_directory) | ✅ (path) |
| Paired MSA | ✅ | ✅ | ❌ | ✅ (CSV key column) | ✅ | ✅ |
| Structural templates | ✅ | ✅ (mmCIF) | ❌ | ✅ (CIF/PDB) | ✅ (via server) | ✅ (A3M/HHR) |
| **Restraints** | | | | | | |
| Covalent bonds | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ |
| Contact restraints | ✅ | ❌ | ❌ | ✅ | ✅ | ✅ |
| Pocket restraints | ✅ | ❌ | ❌ | ✅ | ✅ | ✅ |
| **Inference Parameters** | | | | | | |
| Random seeds | ✅ | ✅ | ✅ (can be empty) | ❌ (CLI arg) | ✅ (single seed) | ❌ (CLI arg) |
| Recycling steps | ✅ | ❌ (CLI arg) | ❌ | ❌ (CLI arg) | ✅ | ❌ (CLI arg) |
| Diffusion steps | ✅ | ❌ (CLI arg) | ❌ | ❌ (CLI arg) | ✅ | ❌ (CLI arg) |
| Diffusion samples | ✅ | ❌ (CLI arg) | ❌ | ❌ (CLI arg) | ✅ | ❌ (CLI arg) |
| Affinity prediction | ✅ (Boltz-specific) | ❌ | ❌ | ✅ | ❌ | ❌ |

Legend: ✅ = fully supported, ⚠️ = partially supported / lossy conversion, ❌ = not supported

## File Format

UniAF3 configs are written in YAML. The top-level structure is:

```yaml
sequences:
  - # Polymer, Ligand, or Glycan entries
covalent_bonds:   # Optional
  - # CovalentBond entries
contact_restraints:   # Optional
  - # ContactRestraint entries
pocket_restraints:   # Optional
  - # PocketRestraint entries
seeds:
  - 42
aux:   # Optional, inference parameters
  num_trunk_recycles: 3
  num_diffn_timesteps: 200
  num_diffn_samples: 5
  num_trunk_samples: 1
```

## Sequences

Each entry in the `sequences` list must be one of four types:

### Protein

Proteins use the `ProteinSeq` schema (which extends `Polymer`) and support MSA directories and structural templates.

```yaml
- seq_type: protein
  id: A                         # or [A, B] for homomeric copies
  sequence: MVLSPADKTNVK       # Standard 1-letter amino acid codes
  description: "My protein"     # Optional description
  modifications:                # Optional PTMs
    - ccd: HY3                  # CCD code of modification
      position: 1               # 1-based residue index
  msa_dir: path/to/msa/         # Optional, directory containing MSA files
  templates:                    # Optional structural templates
    - path: template.cif        # Path to mmCIF or PDB file
      query_idx: [0, 1, 2]      # 0-based query residue indices
      template_idx: [0, 1, 2]   # 0-based template residue indices
      query_chains: [A]         # Optional, chain IDs in query
      template_chains: [A]      # Optional, chain IDs in template
      boltz_enable_force: false  # Boltz-specific: enforce template
      boltz_template_threshold: null  # Boltz-specific: deviation threshold (Å)
  boltz_cyclic: false           # Boltz-specific: cyclic polymer flag
```

**MSA Directory Structure:**

The `msa_dir` field points to a directory with the following expected structure:

```
msa_dir/
  a3ms/
    {seq_hash}.single.a3m    # Unpaired MSA
    {seq_hash}.pair.a3m      # Paired MSA (optional)
```

Where `{seq_hash}` is the SHA-256 hex digest of the protein sequence. This follows the Chai-1 MSA search output convention.

### DNA

```yaml
- seq_type: dna
  id: C
  sequence: GATTACA        # Only A, T, G, C allowed
  modifications:           # Optional
    - ccd: 6OG
      position: 1
```

### RNA

```yaml
- seq_type: rna
  id: D
  sequence: AGCU           # Only A, U, G, C allowed
  modifications:           # Optional
    - ccd: 2MG
      position: 1
```

### Ligand

Ligands must specify exactly one of `ccd` (a list of CCD codes) or `smiles`:

```yaml
# CCD ligand (single or multi-CCD)
- id: E
  ccd:
    - ATP

# Multi-CCD ligand (e.g., glycan as ligand)
- id: F
  ccd:
    - NAG
    - BMA

# SMILES ligand
- id: G
  smiles: "CC(=O)OC1C[NH+]2CCC1CC2"
```

### Glycan

Glycans use Chai-1's glycan notation (modified CCD codes with bond information):

```yaml
- id: H
  chai_str: "NAG(4-1 NAG(4-1 BMA(3-1 MAN)(6-1 MAN)))"
  description: "Branched glycan"
```

For single sugars without bonds: `chai_str: NAG`

## Chain IDs

Chain IDs (`id` field) serve as unique identifiers for each entity. They can be:
- A single string: `id: A`
- A list of strings for homomeric copies: `id: [A, B, C]`

Chain IDs are used to reference entities in restraints. When converting to models that use count-based copies (AF3 Server, Protenix), the number of IDs in the list determines the copy count.

The chain ID naming convention follows the "reverse spreadsheet style" used by AlphaFold3:
`A, B, ..., Z, AA, BA, CA, ..., ZA, AB, BB, ...`

This is generated by the `int_to_letters()` function (1-indexed): `int_to_letters(1)` → `A`, `int_to_letters(27)` → `AA`, `int_to_letters(28)` → `AB`.

## Restraints

### Covalent Bonds

Specify covalent bonds between atoms from different entities:

```yaml
covalent_bonds:
  - atom1:
      chain_id: A           # Entity ID
      residue_idx: 5        # 1-based residue index (0 for ligands)
      atom_name: CG         # Atom name (e.g., CA, N, SG)
      residue_name: P       # Optional, for validation
    atom2:
      chain_id: E           # Entity ID
      residue_idx: 1        # 1-based position within ligand
      atom_name: C04        # Atom name in the ligand
      residue_name: null    # Not required for ligands
    description: "Optional description"
```

**Notes:**
- `atom_name` is required for both atoms.
- `residue_name` is used by Chai-1 for validation and restraint formatting.
- For ligands, `residue_idx` is typically 1 for single-CCD or SMILES ligands.
- Ligand atom names follow RDKit naming conventions.

### Contact Restraints

Distance restraints between two atoms/residues:

```yaml
contact_restraints:
  - token1:
      chain_id: A
      residue_idx: 10       # 1-based, or 0 if atom_name is used for ligands
      atom_name: null        # Optional for polymers, required for ligands
      residue_name: K        # Optional, for validation
    token2:
      chain_id: C
      residue_idx: 5
      atom_name: null
      residue_name: null
    max_distance: 8.0        # Maximum distance in Å (must be 4-20 Å)
    min_distance: 0.0        # Minimum distance in Å (Protenix only)
    boltz_enable_force: true  # Boltz-specific: enforce with potential
```

**Notes:**
- `max_distance` must be between 4.0 and 20.0 Å (Boltz requirement, applied universally).
- `min_distance` is only used by Protenix.
- AF3 and AF3 Server do **not** support contact restraints.

### Pocket Restraints

Specify a binding pocket where a binder chain interacts with specific contact residues:

```yaml
pocket_restraints:
  - binder_chain: E          # ID of the chain binding to the pocket
    contact_tokens:           # List of residues forming the pocket
      - chain_id: A
        residue_idx: 10
        atom_name: null       # For polymers; use atom_name for ligands
        residue_name: K
      - chain_id: A
        residue_idx: 15
        atom_name: null
        residue_name: G
    max_distance: 6.0         # Maximum distance in Å (4-20 Å)
    min_distance: 0.0         # Protenix only
    boltz_enable_force: false  # Boltz-specific: enforce with potential
```

**Notes:**
- Contact tokens must NOT be on the same chain as `binder_chain`.
- Protenix supports only a single pocket constraint per job.
- AF3 and AF3 Server do **not** support pocket restraints.

## Inference Parameters

The `aux` field contains optional inference parameters:

```yaml
aux:
  num_trunk_recycles: 3         # Default: 3
  num_diffn_timesteps: 200      # Default: 200
  num_diffn_samples: 5          # Default: 5
  num_trunk_samples: 1          # Default: 1
  name: "job_name"              # Optional, used in AF3 Server
  boltz_affinity_binder_chain: D  # Boltz-specific: affinity binder chain ID
```

## Seeds

The `seeds` field is a required list of integer random seeds:

```yaml
seeds:
  - 42
  - 123
```

- AF3 uses all seeds directly.
- Chai-1 uses only the first seed; additional seeds are applied via `num_trunk_samples`.
- Boltz and Protenix do not store seeds in their config format; default `[42]` is used on import.

## Complete Example

```yaml
sequences:
  - seq_type: protein
    id: [A, B]
    sequence: MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLS
    msa_dir: dummy_msa/
    modifications:
      - ccd: HY3
        position: 1
    description: Hemoglobin subunit
  - seq_type: dna
    id: C
    sequence: GATTACA
  - id: D
    ccd:
      - ATP
  - id: E
    smiles: "CC(=O)OC1C[NH+]2CCC1CC2"
  - id: F
    chai_str: NAG
    description: Example glycan

covalent_bonds:
  - atom1:
      chain_id: B
      residue_idx: 2
      atom_name: CA
      residue_name: V
    atom2:
      chain_id: D
      residue_idx: 1
      atom_name: C04
      residue_name: null

contact_restraints:
  - token1:
      chain_id: A
      residue_idx: 5
      atom_name: CG
      residue_name: P
    token2:
      chain_id: B
      residue_idx: 5
      atom_name: null
      residue_name: P
    max_distance: 8.0
    boltz_enable_force: true

pocket_restraints:
  - binder_chain: D
    max_distance: 6.0
    contact_tokens:
      - chain_id: A
        residue_idx: 10
        atom_name: null
        residue_name: N
      - chain_id: B
        residue_idx: 3
        atom_name: null
        residue_name: L

seeds:
  - 42
  - 123

aux:
  num_trunk_recycles: 3
  num_diffn_timesteps: 200
  num_diffn_samples: 5
  num_trunk_samples: 1
  boltz_affinity_binder_chain: D
```

## CLI Usage

### Validate a config

```bash
uniaf3 validate input.yaml --format uniaf3
```

### Convert between formats

```bash
# UniAF3 → AlphaFold3
uniaf3 convert input.yaml output_dir/ --from-format uniaf3 --to-format alphafold3

# Boltz → Chai-1
uniaf3 convert boltz_input.yaml output_dir/ --from-format boltz --to-format chai

# AF3 → Protenix
uniaf3 convert af3_input.json output_dir/ --from-format alphafold3 --to-format protenix
```

Supported format names: `uniaf3`, `alphafold3`, `alphafold3server`, `boltz`, `chai`, `protenix`

## Validation Rules

The UniAF3 schema enforces these validation rules:

1. **At least one sequence** must be provided.
2. **Modification positions** must be within the sequence length.
3. **Ligands** must specify exactly one of `ccd` or `smiles`.
4. **Covalent bond atoms** must have non-null `atom_name`.
5. **Contact restraints** require `max_distance` between 4.0 and 20.0 Å, and `max_distance > min_distance`.
6. **Pocket restraint** contact tokens must not be on the same chain as `binder_chain`.
7. **Restraint atoms** must reference valid chain IDs, and residue indices must be within the sequence length.
8. **Residue names** in restraints (when provided) are validated against the sequence.
