"""Tests for UniAF3Config -> BoltzConfig adapter."""

import pytest

from uniaf3.schema import BoltzConfig, UniAF3Config
from uniaf3.schema.base import Glycan, Ligand, Polymer, ProteinSeq


@pytest.fixture(scope="module")
def boltz(uniaf3_conf: UniAF3Config, tmp_path_factory: pytest.TempPathFactory):
    """Convert UniAF3 to Boltz config."""
    from uniaf3.adapters import to_boltz

    return to_boltz(uniaf3_conf, msa_dir=tmp_path_factory.mktemp("msa"), strict=False)


# ruff: noqa: S101
def test_unsupported_glycan_strict(uniaf3_conf: UniAF3Config, tmp_path):
    from uniaf3.adapters import to_boltz

    uniaf3_conf_cp = uniaf3_conf.model_copy(deep=True)
    uniaf3_conf_cp.sequences.append(Glycan(id="Z", chai_str="NAG(1-4 NAG)"))

    with pytest.raises(
        ValueError, match="Bonded glycans are not directly supported in Boltz"
    ):
        to_boltz(uniaf3_conf_cp, msa_dir=tmp_path, strict=True)


def test_warns_on_multi_ccd_ligand(uniaf3_conf: UniAF3Config, tmp_path):
    from uniaf3.adapters import to_boltz

    conf = uniaf3_conf.model_copy(deep=True)
    conf.sequences.append(Ligand(id="Z", ccd=["ATP", "HEM"]))

    with pytest.warns(
        UserWarning, match="Multi-CCD ligands are not supported in Boltz"
    ):
        _ = to_boltz(conf, msa_dir=tmp_path / "msa", strict=False)


def test_version(uniaf3_conf: UniAF3Config, boltz: BoltzConfig):
    assert boltz.version == 1


def test_sequence_count(uniaf3_conf: UniAF3Config, boltz: BoltzConfig):
    # protein + dna + 2 ligands = 4; 1 single CCD glycan kept
    assert len(boltz.sequences) == len(uniaf3_conf.sequences) == 5


def test_protein_fields(uniaf3_conf: UniAF3Config, boltz: BoltzConfig):
    prot = boltz.sequences[0].protein
    src = uniaf3_conf.sequences[0]
    assert isinstance(src, ProteinSeq)
    assert prot is not None
    assert prot.id == src.id
    assert prot.sequence == src.sequence
    assert prot.cyclic == src.boltz_cyclic
    # MSA: "empty" because src.unpaired_msa is None
    assert prot.msa == "empty"


def test_protein_modifications(uniaf3_conf: UniAF3Config, boltz: BoltzConfig):
    prot = boltz.sequences[0].protein
    assert prot is not None

    src = uniaf3_conf.sequences[0]
    assert isinstance(src, ProteinSeq)

    assert prot.modifications is not None
    assert src.modifications is not None
    assert len(prot.modifications) == len(src.modifications)
    assert prot.modifications[0].ccd == src.modifications[0].ccd
    assert prot.modifications[0].position == src.modifications[0].position


def test_dna_fields(uniaf3_conf: UniAF3Config, boltz: BoltzConfig):
    dna = boltz.sequences[1].dna
    src = uniaf3_conf.sequences[1]
    assert isinstance(src, Polymer)
    assert dna is not None
    assert dna.id == src.id
    assert dna.sequence == src.sequence
    assert dna.cyclic == src.boltz_cyclic


def test_ligand_ccd(uniaf3_conf: UniAF3Config, boltz: BoltzConfig):
    lig = boltz.sequences[2].ligand
    src = uniaf3_conf.sequences[2]
    assert isinstance(src, Ligand)
    assert lig is not None
    assert lig.id == src.id
    # UniAF3 uses list of CCD codes; Boltz uses single CCD string
    assert src.ccd is not None
    assert lig.ccd == src.ccd[0]


def test_ligand_smiles(uniaf3_conf: UniAF3Config, boltz: BoltzConfig):
    lig = boltz.sequences[3].ligand
    src = uniaf3_conf.sequences[3]
    assert isinstance(src, Ligand)
    assert lig is not None
    assert lig.id == src.id
    assert lig.smiles == src.smiles


def test_bond_constraint(uniaf3_conf: UniAF3Config, boltz: BoltzConfig):
    assert boltz.constraints is not None
    bond = boltz.constraints[0].bond
    assert uniaf3_conf.covalent_bonds is not None
    src = uniaf3_conf.covalent_bonds[0]
    assert bond is not None
    assert bond.atom1 == (
        src.atom1.chain_id,
        src.atom1.residue_idx,
        src.atom1.atom_name,
    )
    assert bond.atom2 == (
        src.atom2.chain_id,
        src.atom2.residue_idx,
        src.atom2.atom_name,
    )


def test_contact_constraint(uniaf3_conf: UniAF3Config, boltz: BoltzConfig):
    assert boltz.constraints is not None
    ct = boltz.constraints[1].contact
    assert uniaf3_conf.contact_restraints is not None
    src = uniaf3_conf.contact_restraints[0]
    assert ct is not None
    assert ct.token1 == (src.token1.chain_id, src.token1.residue_idx)
    assert ct.token2 == (src.token2.chain_id, src.token2.residue_idx)
    assert ct.max_distance == src.max_distance
    assert ct.force == src.boltz_enable_force


def test_pocket_constraint(uniaf3_conf: UniAF3Config, boltz: BoltzConfig):
    assert boltz.constraints is not None
    pk = boltz.constraints[2].pocket
    assert uniaf3_conf.pocket_restraints is not None
    src = uniaf3_conf.pocket_restraints[0]
    assert pk is not None
    assert pk.binder == src.binder_chain
    assert pk.max_distance == src.max_distance


def test_affinity_property(uniaf3_conf: UniAF3Config, boltz: BoltzConfig):
    assert boltz.properties is not None
    assert len(boltz.properties) == 1
    assert boltz.properties[0].affinity is not None
    assert (
        boltz.properties[0].affinity.binder
        == uniaf3_conf.aux.boltz_affinity_binder_chain
    )
