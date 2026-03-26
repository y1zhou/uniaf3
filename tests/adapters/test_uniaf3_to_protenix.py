"""Tests for UniAF3Config -> ProtenixConfig adapter."""

import pytest

from uniaf3.schema import ProtenixConfig, UniAF3Config
from uniaf3.schema.base import Ligand, Polymer, ProteinSeq


@pytest.fixture(scope="module")
def ptx(uniaf3_conf: UniAF3Config):
    """Convert UniAF3 to Protenix config."""
    from uniaf3.adapters import to_protenix

    with pytest.warns(UserWarning):
        return to_protenix([uniaf3_conf], name="test")


# ruff: noqa: S101
def test_job_count(uniaf3_conf: UniAF3Config, ptx: ProtenixConfig):
    assert len(ptx) == 1
    assert ptx[0].name == "test"


def test_warns_on_chain_id_loss(uniaf3_conf: UniAF3Config):
    from uniaf3.adapters import to_protenix

    with pytest.warns(UserWarning, match="entity/copy indices"):
        _ = to_protenix([uniaf3_conf], strict=False)


def test_protein_fields(uniaf3_conf: UniAF3Config, ptx: ProtenixConfig):
    prot = ptx[0].sequences[0].proteinChain
    src = uniaf3_conf.sequences[0]
    assert isinstance(src, ProteinSeq)
    assert prot is not None
    assert prot.sequence == src.sequence
    # id ["A", "B"] → count=2
    assert isinstance(src.id, list)
    assert len(src.id) == prot.count == 2

    assert prot.unpairedMsaPath == src.unpaired_msa
    assert prot.pairedMsaPath == src.paired_msa

    # NOTE: UniAF3 example has no templates; template mapping tested via
    # Protenix roundtrip where the fixture includes templatesPath.

    assert prot.modifications is not None
    assert src.modifications is not None
    for mod_ptx, mod_uni in zip(prot.modifications, src.modifications, strict=True):
        # CCD_ prefix stripped
        assert mod_ptx.ptmType == f"CCD_{mod_uni.ccd}"
        assert mod_ptx.ptmPosition == mod_uni.position


def test_dna_fields(uniaf3_conf: UniAF3Config, ptx: ProtenixConfig):
    dna = ptx[0].sequences[1].dnaSequence
    src = uniaf3_conf.sequences[1]
    assert isinstance(src, Polymer)
    assert dna is not None
    assert dna.sequence == src.sequence


def test_ccd_ligand_fields(uniaf3_conf: UniAF3Config, ptx: ProtenixConfig):
    lig = ptx[0].sequences[2].ligand
    src = uniaf3_conf.sequences[2]
    assert isinstance(src, Ligand)
    assert lig is not None
    # NOTE: Protenix does not support multiple CCD codes per ligand
    assert src.ccd is not None
    assert lig.ligand == f"CCD_{src.ccd[0]}"


def test_smiles_ligand_fields(uniaf3_conf: UniAF3Config, ptx: ProtenixConfig):
    lig = ptx[0].sequences[3].ligand
    src = uniaf3_conf.sequences[3]
    assert isinstance(src, Ligand)
    assert lig is not None
    assert lig.ligand == src.smiles


def test_covalent_bond(uniaf3_conf: UniAF3Config, ptx: ProtenixConfig):
    job = ptx[0]
    assert job.covalent_bonds is not None
    assert len(job.covalent_bonds) == 1
    assert uniaf3_conf.covalent_bonds is not None
    src = uniaf3_conf.covalent_bonds[0]
    bond = job.covalent_bonds[0]
    assert bond.atom1 == src.atom1.atom_name
    assert bond.atom2 == src.atom2.atom_name
    assert bond.position1 == src.atom1.residue_idx
    assert bond.position2 == src.atom2.residue_idx

    assert bond.entity1 == 1
    assert bond.entity2 == 3
    assert bond.copy1 == 2
    assert bond.copy2 == 1


def test_contact_constraint(uniaf3_conf: UniAF3Config, ptx: ProtenixConfig):
    job = ptx[0]
    assert job.constraint is not None
    assert job.constraint.contact is not None
    assert len(job.constraint.contact) == 1
    assert uniaf3_conf.contact_restraints is not None
    ct = job.constraint.contact[0]
    src = uniaf3_conf.contact_restraints[0]

    assert ct.atom1 == src.token1.atom_name
    assert ct.atom2 == src.token2.atom_name
    assert ct.position1 == src.token1.residue_idx
    assert ct.position2 == src.token2.residue_idx
    assert ct.max_distance == src.max_distance

    assert ct.entity1 == 1
    assert ct.entity2 == 1
    assert ct.copy1 == 1
    assert ct.copy2 == 2


def test_pocket_constraint(uniaf3_conf: UniAF3Config, ptx: ProtenixConfig):
    job = ptx[0]
    assert job.constraint is not None
    assert job.constraint.pocket is not None
    assert uniaf3_conf.pocket_restraints is not None
    src = uniaf3_conf.pocket_restraints[0]
    assert job.constraint.pocket.max_distance == src.max_distance

    pocket = job.constraint.pocket
    assert pocket.max_distance == src.max_distance
    assert pocket.contact_residues[0].entity == 1
    assert pocket.contact_residues[0].copy_idx == 1
    assert pocket.contact_residues[0].position == src.contact_tokens[0].residue_idx

    assert pocket.binder_chain.entity == 3
    assert pocket.binder_chain.copy_idx == 1


def test_multiple_templates_warns():
    """Multiple templates should warn about lossy conversion."""
    from uniaf3.adapters import to_protenix
    from uniaf3.schema.base import PolymerType, ProteinSeq, StructuralTemplate

    config = UniAF3Config(
        sequences=[
            ProteinSeq(
                polymer_type=PolymerType.Protein,
                id="A",
                sequence="MVLSPADKTNVK",
                templates=[
                    StructuralTemplate(path="/some/path/1abc.cif"),
                    StructuralTemplate(path="/some/path/2xyz.cif"),
                ],
            )
        ]
    )
    with pytest.warns(UserWarning) as records:
        result = to_protenix([config], strict=False)

    assert any("only the first" in str(w.message) for w in records)
    assert result[0].sequences[0].proteinChain is not None
    assert result[0].sequences[0].proteinChain.templatesPath == "/some/path/1abc.cif"


def test_template_with_boltz_fields_warns():
    """Template with boltz-specific fields should emit warning."""
    from uniaf3.adapters import to_protenix
    from uniaf3.schema.base import PolymerType, ProteinSeq, StructuralTemplate

    config = UniAF3Config(
        sequences=[
            ProteinSeq(
                polymer_type=PolymerType.Protein,
                id="A",
                sequence="MVLSPADKTNVK",
                templates=[
                    StructuralTemplate(
                        path="/some/path/1abc.cif",
                        boltz_enable_force=True,
                        boltz_template_threshold=0.5,
                    )
                ],
            )
        ]
    )
    with pytest.warns(UserWarning) as records:
        result = to_protenix([config], strict=False)

    assert any("boltz_enable_force" in str(w.message) for w in records)


def test_glycan_with_bonds_warns():
    """Glycan with bonds should emit warning in non-strict mode."""
    from uniaf3.adapters import to_protenix
    from uniaf3.schema.base import Glycan, PolymerType, ProteinSeq

    config = UniAF3Config(
        sequences=[
            ProteinSeq(
                polymer_type=PolymerType.Protein,
                id="A",
                sequence="MVLSPADKTNVK",
            ),
            Glycan(id="B", chai_str="NAG(1-4 NAG)"),
        ]
    )
    with pytest.warns(UserWarning) as records:
        result = to_protenix([config], strict=False)

    assert any("Glycan with bonds not supported" in str(w.message) for w in records)


def test_multiple_pocket_constraints_warns():
    """Multiple pocket constraints should warn about single pocket support."""
    from uniaf3.adapters import to_protenix
    from uniaf3.schema.base import Atom, PolymerType, ProteinSeq, PocketRestraint

    config = UniAF3Config(
        sequences=[
            ProteinSeq(
                polymer_type=PolymerType.Protein,
                id="A",
                sequence="MVLSPADKTNVK",
            ),
            ProteinSeq(
                polymer_type=PolymerType.Protein,
                id="B",
                sequence="GKVGAHAG",
            ),
        ],
        pocket_restraints=[
            PocketRestraint(
                binder_chain="A",
                contact_tokens=[Atom(chain_id="B", residue_idx=1, atom_name=None, residue_name="G")],
                max_distance=8.0,
            ),
            PocketRestraint(
                binder_chain="B",
                contact_tokens=[Atom(chain_id="A", residue_idx=1, atom_name=None, residue_name="M")],
                max_distance=8.0,
            ),
        ],
    )
    with pytest.warns(UserWarning) as records:
        result = to_protenix([config], strict=False)

    assert any("single pocket constraint" in str(w.message) for w in records)


def test_rna_with_modifications():
    """RNA sequence with modifications should be handled in to_protenix."""
    from uniaf3.adapters import to_protenix
    from uniaf3.schema.base import PolymerType, SequenceModification

    config = UniAF3Config(
        sequences=[
            Polymer(
                polymer_type=PolymerType.RNA,
                id="A",
                sequence="ACGU",
                modifications=[SequenceModification(ccd="HY3", position=2)],
            )
        ]
    )
    with pytest.warns(UserWarning):
        result = to_protenix([config], strict=False)

    assert result[0].sequences[0].rnaSequence is not None
    rna_entry = result[0].sequences[0].rnaSequence
    assert rna_entry.modifications is not None
    assert len(rna_entry.modifications) == 1


def test_from_protenix_rna_sequence():
    """from_protenix should handle RNA sequences."""
    from uniaf3.adapters import from_protenix
    from uniaf3.schema.protenix import ProtenixJob, ProtenixRNASequence, ProtenixSequenceEntry

    job = ProtenixJob(
        name="test",
        sequences=[
            ProtenixSequenceEntry(
                rnaSequence=ProtenixRNASequence(sequence="ACGU", count=1)
            )
        ],
    )
    with pytest.warns(UserWarning):
        result = from_protenix([job])
    assert len(result) == 1
    assert isinstance(result[0].sequences[0], Polymer)
    from uniaf3.schema.base import PolymerType
    assert result[0].sequences[0].polymer_type == PolymerType.RNA
