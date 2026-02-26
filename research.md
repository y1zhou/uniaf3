# UniAF3 Bug Research Report

## Methodology

This report documents bugs found through systematic analysis of the UniAF3
codebase, including all schema definitions (`src/uniaf3/schema/`), adapter
conversion logic (`src/uniaf3/adapters/`), CLI entry points, and vendor code.
Each bug was identified by tracing data flow through the conversion pipeline and
cross-referencing with the source repository documentation for AlphaFold3,
Boltz, Chai-1, and Protenix.

---

## Bug 1: `to_uniaf3` missing `msa_dir` argument for AlphaFold3 conversion

**File:** `src/uniaf3/adapters/__init__.py`, line 67

**Severity:** Critical (runtime crash)

**Description:**
The `to_uniaf3()` dispatcher function calls `from_alphafold3(conf)` without
passing the required `msa_dir` parameter. The function signature of
`from_alphafold3` is:

```python
def from_alphafold3(config: AF3Config, msa_dir: str | Path) -> UniAF3Config:
```

The `msa_dir` parameter has no default value and is required. Calling
`from_alphafold3(conf)` without it raises:

```
TypeError: from_alphafold3() missing 1 required positional argument: 'msa_dir'
```

This bug is triggered whenever a user converts an AlphaFold3 config to UniAF3
via the `to_uniaf3()` function or the CLI `convert` command.

**Root cause:** The `msa_dir` keyword argument is available in `to_uniaf3()`
but was not forwarded to `from_alphafold3()`. Other adapters (e.g., Boltz)
correctly pass `msa_dir=msa_dir`.

**Fix:**

```python
# Before (broken):
return from_alphafold3(conf)

# After (fixed):
return from_alphafold3(conf, msa_dir=msa_dir)
```

---

## Bug 2: `to_chai` covalent bond adapter uses `r.atom1` for both atoms

**File:** `src/uniaf3/adapters/chai.py`, lines 146–151

**Severity:** High (incorrect output data)

**Description:**
In the `to_chai()` function, when converting covalent bonds to Chai restraints,
the code iterates over `[r.atom1, r.atom2]` with a loop variable `atom`. In the
`else` branch (polymer atoms), the code incorrectly references `r.atom1`
instead of `atom`:

```python
for atom in [r.atom1, r.atom2]:
    if entity_types[atom.chain_id] in {ChaiEntityType.Ligand, ChaiEntityType.Glycan}:
        res_idx.append(f"@{atom.atom_name}")
    else:
        if atom.residue_name is None:
            raise ValueError(
                f"Missing residue name for covalent bond atom: {r.atom1}"  # BUG: should be {atom}
            )
        res_idx.append(
            f"{r.atom1.residue_name}{r.atom1.residue_idx}@{r.atom1.atom_name}"
            # BUG: always uses atom1's data, even when processing atom2
        )
```

When processing `atom2`, this produces `atom1`'s residue name, index, and atom
name instead of `atom2`'s. The resulting Chai restraint CSV will have
duplicated atom1 information for both sides of the covalent bond.

**Fix:**

```python
if atom.residue_name is None:
    raise ValueError(
        f"Missing residue name for covalent bond atom: {atom}"
    )
res_idx.append(
    f"{atom.residue_name}{atom.residue_idx}@{atom.atom_name}"
)
```

---

## Bug 3: `_from_protenix` pocket constraint `copy_idx` off-by-one error

**File:** `src/uniaf3/adapters/protenix.py`, line 450

**Severity:** High (incorrect chain assignment)

**Description:**
When converting Protenix pocket constraints to UniAF3, the contact residue
`copy_idx` is used directly as a list index without subtracting 1. Since
`copy_idx` is 1-based (defined as `PositiveInt`), this accesses the wrong
element:

```python
contact_residues = [
    (entity_to_chains[cr.entity][cr.copy_idx], cr.position)  # BUG: should be cr.copy_idx - 1
    for cr in pct.contact_residues
]
```

The binder chain on line 446–448 correctly uses `copy_idx - 1`:

```python
binder_chain = entity_to_chains[pct.binder_chain.entity][
    pct.binder_chain.copy_idx - 1  # correct
]
```

For an entity with `count=2` and `copy=1`, the chains would be `["A", "B"]`.
Using `[1]` gives "B" instead of the correct "A".

**Fix:**

```python
contact_residues = [
    (entity_to_chains[cr.entity][cr.copy_idx - 1], cr.position)
    for cr in pct.contact_residues
]
```

---

## Bug 4: `_next_chain_ids` generates inconsistent chain IDs for index >= 26

**File:** `src/uniaf3/adapters/alphafold3_server.py`, lines 196–208

**Severity:** Medium (inconsistent behavior, affects edge cases with >26 chains)

**Description:**
The `_from_alphafold3_server()` function uses a custom `_next_chain_ids()`
function to generate chain IDs. For chain indices >= 26, the generated IDs are
in a different order than the `int_to_letters()` function used by all other
adapters:

```python
# _next_chain_ids formula (for n >= 26):
left_char = chr(65 + (n - 26) % 26)   # low-order character
right_char = chr(65 + (n - 26) // 26)  # high-order character
ids.append(f"{left_char}{right_char}")
```

This produces: n=26 → "AA", n=27 → "BA", n=28 → "CA", ...

But `int_to_letters()` (used by Protenix and Chai adapters) produces:
n=27 → "AA", n=28 → "AB", n=29 → "AC", ...

The AF3 documentation also uses "reverse spreadsheet style":
`A, B, ..., Z, AA, BA, CA, ..., ZA, AB, BB, ...`

So `_next_chain_ids` follows the AF3 convention but is inconsistent with
`int_to_letters()`. This means roundtrip conversions through
AF3Server → UniAF3 → Protenix would produce mismatched chain IDs for systems
with more than 26 chains.

**Fix:** Replace the custom logic with `int_to_letters(chain_counter + 1)`:

```python
def _next_chain_ids(count: int) -> str | list[str]:
    nonlocal chain_counter
    ids = []
    for _ in range(count):
        ids.append(int_to_letters(chain_counter + 1))
        chain_counter += 1
    return ids[0] if len(ids) == 1 else ids
```

Note: This changes the ordering convention but makes all adapters consistent.
The AF3 documentation uses the "reverse spreadsheet" order (AA, BA, CA...) while
`int_to_letters` uses standard spreadsheet order (AA, AB, AC...). Within UniAF3,
internal consistency is more important.

---

## Bug 5: `from_boltz` MSA directory path mismatch for CSV format

**File:** `src/uniaf3/adapters/boltz.py`, line 427

**Severity:** Medium (MSA files not found after conversion)

**Description:**
When converting Boltz CSV-format MSA files to UniAF3's A3M format, the
`split_boltz_csv_to_a3m()` function writes files directly to the specified
output directory. However, `ProteinSeq.unpaired_msa` expects files under an
`a3ms/` subdirectory:

```python
# In from_boltz():
_ = split_boltz_csv_to_a3m(p.msa, msa_dir)  # writes to msa_dir/

# But ProteinSeq.unpaired_msa looks for:
# Path(self.msa_dir) / "a3ms" / f"{self.seq_hash}.single.a3m"
```

For `.a3m` input files, the code correctly creates the `a3ms/` subdirectory
(line 431), but for `.csv` files it does not.

**Fix:**

```python
_ = split_boltz_csv_to_a3m(p.msa, msa_dir_path / "a3ms")
```

---

## Bug 6: `from_chai` pocket restraints dict check always True

**File:** `src/uniaf3/adapters/chai.py`, lines 398–400

**Severity:** Low (produces empty list instead of None)

**Description:**
The `pocket_restraints` variable is initialized as an empty dict `{}`, never
`None`. The conditional check `pocket_restraints is not None` always evaluates
to True:

```python
pocket_restraints: dict[str, PocketRestraint] = {}
# ...
pocket_restraints=list(pocket_restraints.values())
    if pocket_restraints is not None  # always True!
    else None,
```

When no pocket restraints exist, this returns `[]` instead of `None`. While an
empty list is valid for the `UniAF3Config` schema, all other adapters use the
`or None` pattern to convert empty collections to `None` for consistency.

**Fix:**

```python
pocket_restraints=list(pocket_restraints.values()) or None,
```

---

## Summary

| Bug | Severity | File | Type |
|-----|----------|------|------|
| 1. Missing `msa_dir` in `to_uniaf3` for AF3 | Critical | `adapters/__init__.py` | Runtime crash |
| 2. `to_chai` covalent bond uses `r.atom1` for both atoms | High | `adapters/chai.py` | Incorrect output |
| 3. `_from_protenix` pocket `copy_idx` off-by-one | High | `adapters/protenix.py` | Wrong chain assignment |
| 4. `_next_chain_ids` inconsistent with `int_to_letters` | Medium | `adapters/alphafold3_server.py` | Inconsistency |
| 5. `from_boltz` MSA path mismatch for CSV | Medium | `adapters/boltz.py` | Files not found |
| 6. `from_chai` pocket dict check always True | Low | `adapters/chai.py` | Empty list vs None |

---

## Recommendations

1. **Fix all bugs** listed above, starting with the critical and high-severity
   ones.
2. **Add test coverage** for the `to_uniaf3()` dispatcher function to catch
   missing parameter bugs.
3. **Add chain ID verification** in the Protenix roundtrip tests for pocket
   constraint contact residues.
4. **Standardize chain ID generation** across all adapters to use
   `int_to_letters()`.
5. **Add integration tests** for CSV-format Boltz MSA conversion.
