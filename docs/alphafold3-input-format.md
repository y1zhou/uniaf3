# AlphaFold3 Input Format

## Overview

AlphaFold3 uses a custom JSON input format. The open-source release
([google-deepmind/alphafold3](https://github.com/google-deepmind/alphafold3))
uses a different format from the AlphaFold Server (see
[AlphaFold3 Server Input Format](alphafold3-server-input-format.md)).

**Reference:** <https://github.com/google-deepmind/alphafold3/blob/main/docs/input.md>

## Top-level Structure

```json
{
  "name": "Job name",
  "modelSeeds": [1, 2],
  "sequences": [
    {"protein": {...}},
    {"rna": {...}},
    {"dna": {...}},
    {"ligand": {...}}
  ],
  "bondedAtomPairs": [...],
  "userCCD": "...",
  "userCCDPath": "...",
  "dialect": "alphafold3",
  "version": 4
}
```

### Fields

| Field             | Type                   | Required | Description                                                                        |
| ----------------- | ---------------------- | -------- | ---------------------------------------------------------------------------------- |
| `name`            | `str`                  | Yes      | Job name. A sanitized version is used for output filenames.                        |
| `modelSeeds`      | `list[int]`            | Yes      | Random seeds. At least one required. Each seed produces one prediction.            |
| `sequences`       | `list[SequenceEntry]`  | Yes      | List of molecular entities.                                                        |
| `bondedAtomPairs` | `list[BondedAtomPair]` | No       | Covalent bonds between atoms.                                                      |
| `userCCD`         | `str`                  | No       | Inline CCD mmCIF string for custom ligands. Mutually exclusive with `userCCDPath`. |
| `userCCDPath`     | `str`                  | No       | Path to CCD mmCIF file. Mutually exclusive with `userCCD`.                         |
| `dialect`         | `"alphafold3"`         | Yes      | Must be `"alphafold3"`.                                                            |
| `version`         | `1 \| 2 \| 3 \| 4`     | Yes      | Input format version.                                                              |

### Versions

- **v1**: Initial format.
- **v2**: Added `unpairedMsaPath`, `pairedMsaPath`, `mmcifPath` for external files.
- **v3**: Added `userCCDPath` for external CCD files.
- **v4**: Added `description` field for all entity types.

## Sequences

Each entry in `sequences` must contain exactly one of `protein`, `rna`, `dna`, or `ligand`.

### Protein

```json
{
  "protein": {
    "id": "A",
    "sequence": "PVLSCGEWQL",
    "modifications": [
      {"ptmType": "HY3", "ptmPosition": 1}
    ],
    "description": "My protein chain",
    "unpairedMsa": ">query\\nPVLSCGEWQL",
    "pairedMsa": "",
    "templates": [
      {
        "mmcif": "...",
        "mmcifPath": "path/to/template.cif",
        "queryIndices": [0, 1, 2],
        "templateIndices": [0, 1, 2]
      }
    ]
  }
}
```

| Field             | Type                           | Required | Description                                                     |
| ----------------- | ------------------------------ | -------- | --------------------------------------------------------------- |
| `id`              | `str \| list[str]`             | Yes      | Unique chain ID(s). List implies homomeric copies.              |
| `sequence`        | `str`                          | Yes      | Amino acid sequence (1-letter standard codes).                  |
| `modifications`   | `list[{ptmType, ptmPosition}]` | No       | Post-translational modifications. CCD code + 1-based position.  |
| `description`     | `str`                          | No       | Textual description (v4+).                                      |
| `unpairedMsa`     | `str`                          | No       | Inline unpaired MSA (A3M format). Mutually exclusive with path. |
| `unpairedMsaPath` | `str`                          | No       | Path to unpaired MSA file (v2+).                                |
| `pairedMsa`       | `str`                          | No       | Inline paired MSA (A3M format). Not recommended by DeepMind.    |
| `pairedMsaPath`   | `str`                          | No       | Path to paired MSA file (v2+).                                  |
| `templates`       | `list[Template] \| null`       | No       | Structural templates.                                           |

#### MSA and Template Search Rules

AlphaFold3 distinguishes missing evidence from evidence that is explicitly empty. A
`null` or omitted field allows the data pipeline to search for that evidence. An
empty string or list is present and explicitly disables that evidence source; empty
values are not missing values.

The unpaired and paired MSA sides must be supplied together. Each side may use either
its inline field or its path field, but not both. An empty inline string counts as a
supplied side.

| Protein MSA state                            | Template state | Behavior                                                 |
| -------------------------------------------- | -------------- | -------------------------------------------------------- |
| Both sides `null` or omitted                 | `null`/omitted | Search MSAs and templates                                |
| Both sides `null` or omitted                 | `[]`           | Search MSAs only; template search is explicitly disabled |
| Both sides `null` or omitted                 | Populated      | Invalid partial custom evidence                          |
| Both sides supplied, including empty strings | `null`/omitted | Use supplied MSA state and search templates only         |
| Both sides supplied, including empty strings | `[]`           | Use supplied MSA state and do not search templates       |
| Both sides supplied, including empty strings | Populated      | Use all supplied MSA and template evidence               |
| Exactly one MSA side `null` or omitted       | Any            | Invalid partial MSA evidence                             |

- Setting `unpairedMsa` to a non-empty A3M and `pairedMsa` to `""` is the recommended custom MSA approach.
- Setting both MSA strings to `""` runs MSA-free (single-sequence mode).
- Populated templates without supplied paired and unpaired MSA state are invalid.
- The first sequence in a populated MSA must exactly match the query sequence.

#### Templates

| Field             | Type        | Required    | Description                                                            |
| ----------------- | ----------- | ----------- | ---------------------------------------------------------------------- |
| `mmcif`           | `str`       | Conditional | Inline mmCIF string. Mutually exclusive with `mmcifPath`.              |
| `mmcifPath`       | `str`       | Conditional | Path to mmCIF file (v2+).                                              |
| `queryIndices`    | `list[int]` | Yes         | 0-based query residue indices for alignment.                           |
| `templateIndices` | `list[int]` | Yes         | 0-based template residue indices. Must match length of `queryIndices`. |

### RNA

```json
{
  "rna": {
    "id": "A",
    "sequence": "AGCU",
    "modifications": [
      {"modificationType": "2MG", "basePosition": 1}
    ],
    "description": "Short RNA",
    "unpairedMsa": "...",
    "unpairedMsaPath": "..."
  }
}
```

| Field                             | Type                                     | Description                           |
| --------------------------------- | ---------------------------------------- | ------------------------------------- |
| `id`                              | `str \| list[str]`                       | Unique chain ID(s).                   |
| `sequence`                        | `str`                                    | RNA sequence (A, C, G, U only).       |
| `modifications`                   | `list[{modificationType, basePosition}]` | CCD code + 1-based position.          |
| `unpairedMsa` / `unpairedMsaPath` | `str`                                    | Optional MSA (no paired MSA for RNA). |

### DNA

```json
{
  "dna": {
    "id": "A",
    "sequence": "GACCTCT",
    "modifications": [
      {"modificationType": "6OG", "basePosition": 1}
    ],
    "description": "DNA strand"
  }
}
```

| Field           | Type                                     | Description                     |
| --------------- | ---------------------------------------- | ------------------------------- |
| `id`            | `str \| list[str]`                       | Unique chain ID(s).             |
| `sequence`      | `str`                                    | DNA sequence (A, C, G, T only). |
| `modifications` | `list[{modificationType, basePosition}]` | CCD code + 1-based position.    |

**Note:** DNA has no MSA support.

### Ligand

```json
{
  "ligand": {
    "id": ["G", "H"],
    "ccdCodes": ["ATP"],
    "description": "ATP molecules"
  }
}
```

| Field      | Type               | Description                                                                           |
| ---------- | ------------------ | ------------------------------------------------------------------------------------- |
| `id`       | `str \| list[str]` | Unique chain ID(s).                                                                   |
| `ccdCodes` | `list[str]`        | CCD codes. Mutually exclusive with `smiles`. Supports multi-CCD for branched ligands. |
| `smiles`   | `str`              | SMILES string. Cannot specify covalent bonds with SMILES ligands.                     |

**Important:** SMILES strings must be JSON-escaped (backslashes doubled).

#### Ions

Ions are treated as ligands: `{"ligand": {"id": "X", "ccdCodes": ["MG"]}}`.

## Bonded Atom Pairs

Covalent bonds between atoms. Each bond is a pair of `[entity_id, residue_id, atom_name]`:

```json
"bondedAtomPairs": [
  [["A", 1, "CA"], ["F", 1, "CHA"]],
  [["I", 1, "O6"], ["I", 2, "C1"]]
]
```

| Component  | Type  | Description                                                                 |
| ---------- | ----- | --------------------------------------------------------------------------- |
| Entity ID  | `str` | The `id` of the entity.                                                     |
| Residue ID | `int` | 1-based residue index. For multi-CCD ligands, refers to the CCD part index. |
| Atom Name  | `str` | Atom name as defined in CCD.                                                |

**Limitations:**

- Cannot specify bonds to SMILES ligands (no stable atom names).
- Use `userCCD` to define custom ligands with named atoms for bonding.

## User-provided CCD

For custom ligands that need specific atom names (required for bonding) or when RDKit fails to generate a conformer, provide a CCD definition in mmCIF format:

```json
{
  "userCCD": "data_MY_LIGAND\n_chem_comp.id MY_LIGAND\n...",
  "sequences": [
    {"ligand": {"id": "A", "ccdCodes": ["MY_LIGAND"]}}
  ]
}
```

## UniAF3 Adapter Mapping

### UniAF3 → AF3 Conversion

| UniAF3 Feature     | AF3 Mapping              | Notes                                        |
| ------------------ | ------------------------ | -------------------------------------------- |
| `ProteinSeq`       | `protein` entry          | MSA paths preserved if available             |
| `Polymer(dna)`     | `dna` entry              |                                              |
| `Polymer(rna)`     | `rna` entry              |                                              |
| `Ligand(ccd)`      | `ligand` with `ccdCodes` |                                              |
| `Ligand(smiles)`   | `ligand` with `smiles`   |                                              |
| `Glycan`           | **Dropped**              | AF3 has no native glycan type                |
| `CovalentBond`     | `bondedAtomPairs`        | Direct mapping                               |
| `ContactRestraint` | **Dropped**              | AF3 does not support non-covalent restraints |
| `PocketRestraint`  | **Dropped**              | AF3 does not support pocket restraints       |
| `aux.seeds`        | `modelSeeds`             |                                              |

### AF3 → UniAF3 Conversion

| AF3 Feature       | UniAF3 Mapping    | Notes                                    |
| ----------------- | ----------------- | ---------------------------------------- |
| `protein`         | `ProteinSeq`      | MSA files copied to hash-based directory |
| `dna`             | `Polymer(dna)`    |                                          |
| `rna`             | `Polymer(rna)`    |                                          |
| `ligand` (CCD)    | `Ligand(ccd)`     |                                          |
| `ligand` (SMILES) | `Ligand(smiles)`  |                                          |
| `bondedAtomPairs` | `CovalentBond`    |                                          |
| `userCCD`         | **Not preserved** |                                          |
| `name`            | **Not preserved** | Stored in `aux.name` on conversion       |
