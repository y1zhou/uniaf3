"""Tests for Protenix adapter template handling."""

from pathlib import Path
from unittest.mock import patch

import pytest

from uniaf3.adapters.protenix import (
    _build_a3m_gapped_seq,
    _from_protenix,
    _template_hits_to_structural_templates,
    _to_protenix,
)
from uniaf3.constant import PDB_SERVER_URL
from uniaf3.schema.base import (
    PolymerType,
    ProteinSeq,
    StructuralTemplate,
    UniAF3Config,
)
from uniaf3.schema.protenix import (
    ProtenixJob,
    ProtenixProteinChain,
    ProtenixSequenceEntry,
)
from uniaf3.vendor.protenix_template import HmmsearchA3MParser, TemplateHit

# ruff: noqa: S101

# ─── Fixtures ──────────────────────────────────────────────────────────

QUERY_SEQ = "ACDEFGHIKLMNPQRST"  # 17 residues

# Template has gaps at query positions 2,3 and 9,10
# Full template chain has 17 residues; aligned region covers positions 0-12
TEMPLATE_A3M = """\
>query
ACDEFGHIKLMNPQRST
>1abc_A/1-13 [subseq from] mol:protein length:17  test template
AC--FGHIK--NPQRST
"""

# Same alignment but template has an insertion (lowercase 'x') between positions 4 and 6
TEMPLATE_A3M_WITH_INSERTION = """\
>query
ACDEFGHIKLMNPQRST
>2def_B/1-14 [subseq from] mol:protein length:18  insertion test
AC--FGHxIK--NPQRST
"""

EXPECTED_QUERY_IDX = [0, 1, 4, 5, 6, 7, 8, 11, 12, 13, 14, 15, 16]
EXPECTED_TEMPLATE_IDX = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

# HHR format — must ensure line[17:] starts with whitespace
TEMPLATE_HHR = """\
No 1
>1abc_A some template description
Probab=97.80  E-value=1.2e-10  Score=80.50  Aligned_cols=13  Identities=76%  Similarity=1.200  Sum_probs=12.3  Template_Neff=8.100

Q query             1 ACDEFGHIKLMNPQRST   17 (17)
Q Consensus         1 acdefghiklmnpqrst   17 (17)
                      ||  |||||  ||||||
T 1abc_A            1 AC--FGHIK--NPQRST   13 (17)
T Consensus         1 ac--fghik--npqrst   13 (17)
"""


# ─── _build_a3m_line tests ─────────────────────────────────────────────


def test_build_a3m_line_simple():
    """Simple alignment without insertions."""
    # template_seq covers positions 0-12 (13 residues), no extra residues
    template_seq = "ACFGHIKNPQRST"  # 13 residues
    result = _build_a3m_gapped_seq(
        query_seq_len=17,
        template_seq=template_seq,
        query_idx=EXPECTED_QUERY_IDX,
        template_idx=EXPECTED_TEMPLATE_IDX,
    )
    assert result == "AC--FGHIK--NPQRST"


def test_build_a3m_line_with_insertions():
    """Alignment with lowercase insertions for unmapped template positions."""
    # Template seq: positions 0-13 (14 residues)
    # Positions 0,1,2,3,4,6,7,8,9,10,11,12 are mapped; position 5 is insertion
    template_seq = "ACFGHXIKNPQRSTXXXX"  # 18 residues, X at position 5
    template_idx_ins = [0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13]  # skip pos 5
    result = _build_a3m_gapped_seq(
        query_seq_len=17,
        template_seq=template_seq,
        query_idx=EXPECTED_QUERY_IDX,
        template_idx=template_idx_ins,
    )
    # Position 5 (X) should appear as lowercase 'x' between H and I
    assert result == "AC--FGHxIK--NPQRST"


def test_build_a3m_line_empty_alignment():
    """Empty alignment (no mapped positions)."""
    result = _build_a3m_gapped_seq(
        query_seq_len=5,
        template_seq="ABCDE",
        query_idx=[],
        template_idx=[],
    )
    assert result == "-----"


# ─── _template_hits_to_structural_templates tests ──────────────────────


def test_hits_to_templates_basic():
    """Convert a single TemplateHit to StructuralTemplate."""
    from uniaf3.utils import normalize_out_dir

    hit = TemplateHit(
        index=1,
        name="1abc_A",
        aligned_cols=13,
        sum_probs=None,
        query=QUERY_SEQ,
        hit_sequence="AC--FGHIK--NPQRST",
        indices_query=list(range(17)),
        indices_hit=[0, 1, -1, -1, 2, 3, 4, 5, 6, -1, -1, 7, 8, 9, 10, 11, 12],
    )
    templates, tasks = _template_hits_to_structural_templates(
        [hit], chain_ids=["A"], output_dir=None
    )
    assert len(templates) == 1
    t = templates[0]
    assert t.query_idx == EXPECTED_QUERY_IDX
    assert t.template_idx == EXPECTED_TEMPLATE_IDX
    assert t.template_chains == ["A"]
    assert t.query_chains == ["A"]
    # When output_dir is None, uses the default cache directory
    expected_path = normalize_out_dir(None, "rcsb") / "AB" / "1ABC.cif.gz"
    assert t.path == str(expected_path)
    # Download task is still generated to populate the cache
    assert f"{PDB_SERVER_URL}/1ABC.cif.gz" in tasks


def test_hits_to_templates_with_output_dir(tmp_path):
    """With output_dir, generate download tasks."""
    hit = TemplateHit(
        index=1,
        name="1abc_A",
        aligned_cols=13,
        sum_probs=None,
        query=QUERY_SEQ,
        hit_sequence="AC--FGHIK--NPQRST",
        indices_query=list(range(17)),
        indices_hit=[0, 1, -1, -1, 2, 3, 4, 5, 6, -1, -1, 7, 8, 9, 10, 11, 12],
    )
    templates, tasks = _template_hits_to_structural_templates(
        [hit], chain_ids=["A", "B"], output_dir=tmp_path
    )
    assert len(templates) == 1
    assert len(tasks) == 1
    assert f"{PDB_SERVER_URL}/1ABC.cif.gz" in tasks
    assert templates[0].path == str(tmp_path / "rcsb" / "AB" / "1ABC.cif.gz")
    assert templates[0].query_chains == ["A", "B"]


def test_hits_to_templates_hhr_name_with_description():
    """HHR hit names include descriptions; only the identifier is used."""
    from uniaf3.utils import normalize_out_dir

    hit = TemplateHit(
        index=1,
        name="4v5d_BG Outer membrane protein",
        aligned_cols=10,
        sum_probs=12.3,
        query="ACDEFGHIKL",
        hit_sequence="ACDEFGHIKL",
        indices_query=list(range(10)),
        indices_hit=list(range(10)),
    )
    templates, _ = _template_hits_to_structural_templates(
        [hit], chain_ids=["X"], output_dir=None
    )
    assert len(templates) == 1
    assert templates[0].template_chains == ["BG"]
    expected_path = normalize_out_dir(None, "rcsb") / "V5" / "4V5D.cif.gz"
    assert templates[0].path == str(expected_path)


def test_hits_to_templates_skips_empty_mapping():
    """Hits with empty mappings are skipped."""
    hit = TemplateHit(
        index=1,
        name="1abc_A",
        aligned_cols=0,
        sum_probs=None,
        query="ACDEF",
        hit_sequence="-----",
        indices_query=[-1, -1, -1, -1, -1],
        indices_hit=[-1, -1, -1, -1, -1],
    )
    templates, _ = _template_hits_to_structural_templates(
        [hit], chain_ids=["A"], output_dir=None
    )
    assert len(templates) == 0


# ─── _from_protenix template parsing tests ─────────────────────────────


def test_from_protenix_a3m_parsing(tmp_path):
    """Parse an A3M template file into StructuralTemplates."""
    a3m_path = tmp_path / "templates.a3m"
    a3m_path.write_text(TEMPLATE_A3M)

    job = ProtenixJob(
        name="test",
        sequences=[
            ProtenixSequenceEntry(
                proteinChain=ProtenixProteinChain(
                    sequence=QUERY_SEQ,
                    count=1,
                    templatesPath=str(a3m_path),
                )
            )
        ],
    )

    with pytest.warns(UserWarning), patch("uniaf3.utils.download_files") as mock_dl:
        result = _from_protenix(job, output_dir=tmp_path)

    prot = result.sequences[0]
    assert isinstance(prot, ProteinSeq)
    assert prot.templates is not None
    assert len(prot.templates) == 1

    tmpl = prot.templates[0]
    assert tmpl.query_idx == EXPECTED_QUERY_IDX
    assert tmpl.template_idx == EXPECTED_TEMPLATE_IDX
    assert tmpl.template_chains == ["A"]
    assert "1ABC.cif.gz" in tmpl.path

    # Should have attempted to download the CIF
    mock_dl.assert_called_once()
    dl_urls = mock_dl.call_args[0][0]  # first positional arg is the dict
    assert f"{PDB_SERVER_URL}/1ABC.cif.gz" in dl_urls


def test_from_protenix_a3m_with_insertion(tmp_path):
    """Parse A3M with lowercase insertions."""
    a3m_path = tmp_path / "templates.a3m"
    a3m_path.write_text(TEMPLATE_A3M_WITH_INSERTION)

    job = ProtenixJob(
        name="test",
        sequences=[
            ProtenixSequenceEntry(
                proteinChain=ProtenixProteinChain(
                    sequence=QUERY_SEQ,
                    count=1,
                    templatesPath=str(a3m_path),
                )
            )
        ],
    )

    with (
        pytest.warns(UserWarning),
        patch("uniaf3.utils.download_files"),
    ):
        result = _from_protenix(job, output_dir=tmp_path)

    prot = result.sequences[0]
    assert isinstance(prot, ProteinSeq)
    assert prot.templates is not None
    assert len(prot.templates) == 1

    tmpl = prot.templates[0]
    assert tmpl.template_chains == ["B"]
    # Positions 0-4 in template, then insertion at 5 (skipped), then 6-13
    # {0:0, 1:1, 4:2, 5:3, 6:4, 7:6, 8:7, 11:8, 12:9, 13:10, 14:11, 15:12, 16:13}
    expected_q = [0, 1, 4, 5, 6, 7, 8, 11, 12, 13, 14, 15, 16]
    expected_t = [0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13]
    assert tmpl.query_idx == expected_q
    assert tmpl.template_idx == expected_t


def test_from_protenix_hhr_parsing(tmp_path):
    """Parse an HHR template file into StructuralTemplates."""
    hhr_path = tmp_path / "templates.hhr"
    hhr_path.write_text(TEMPLATE_HHR)

    job = ProtenixJob(
        name="test",
        sequences=[
            ProtenixSequenceEntry(
                proteinChain=ProtenixProteinChain(
                    sequence=QUERY_SEQ,
                    count=1,
                    templatesPath=str(hhr_path),
                )
            )
        ],
    )

    with (
        pytest.warns(UserWarning),
        patch("uniaf3.utils.download_files"),
    ):
        result = _from_protenix(job, output_dir=tmp_path)

    prot = result.sequences[0]
    assert isinstance(prot, ProteinSeq)
    assert prot.templates is not None
    assert len(prot.templates) == 1

    tmpl = prot.templates[0]
    assert tmpl.query_idx == EXPECTED_QUERY_IDX
    assert tmpl.template_idx == EXPECTED_TEMPLATE_IDX
    assert tmpl.template_chains == ["A"]


def test_from_protenix_nonexistent_a3m():
    """Non-existent A3M file falls back to path-only template."""
    job = ProtenixJob(
        name="test",
        sequences=[
            ProtenixSequenceEntry(
                proteinChain=ProtenixProteinChain(
                    sequence=QUERY_SEQ,
                    count=1,
                    templatesPath="/nonexistent/path.a3m",
                )
            )
        ],
    )

    with pytest.warns(UserWarning):
        result = _from_protenix(job)

    prot = result.sequences[0]
    assert isinstance(prot, ProteinSeq)
    assert prot.templates is not None
    assert len(prot.templates) == 1
    assert prot.templates[0].path == "/nonexistent/path.a3m"
    assert prot.templates[0].query_idx is None


def test_from_protenix_unsupported_extension():
    """Unsupported template extension emits warning and falls back."""
    job = ProtenixJob(
        name="test",
        sequences=[
            ProtenixSequenceEntry(
                proteinChain=ProtenixProteinChain(
                    sequence=QUERY_SEQ,
                    count=1,
                    templatesPath="/some/path/templates.pdb",
                )
            )
        ],
    )

    with pytest.warns(UserWarning) as records:
        result = _from_protenix(job)
    assert any("a3m or hhr" in str(w.message) for w in records)

    prot = result.sequences[0]
    assert isinstance(prot, ProteinSeq)
    assert prot.templates is not None
    assert prot.templates[0].path == "/some/path/templates.pdb"


# ─── _to_protenix template generation tests ────────────────────────────


def test_to_protenix_with_output_dir(tmp_path):
    """Generate A3M file from StructuralTemplates when output_dir is provided."""
    # Template chain sequence: 17 residues, positions 0-12 are mapped
    template_chain_seq = "ACFGHIKNPQRSTXXXX"  # 17 residues

    config = UniAF3Config(
        sequences=[
            ProteinSeq(
                polymer_type=PolymerType.Protein,
                id="A",
                sequence=QUERY_SEQ,
                templates=[
                    StructuralTemplate(
                        path="/fake/1abc.cif",
                        query_idx=EXPECTED_QUERY_IDX,
                        template_idx=EXPECTED_TEMPLATE_IDX,
                        template_chains=["A"],
                    )
                ],
            )
        ]
    )

    with (
        pytest.warns(UserWarning),
        patch(
            "uniaf3.adapters.protenix._read_chain_sequence",
            return_value=(template_chain_seq, 17),
        ),
    ):
        result = _to_protenix(config, name="test", strict=False, output_dir=tmp_path)

    pc = result.sequences[0].proteinChain
    assert pc is not None
    assert pc.templatesPath is not None

    a3m_path = Path(pc.templatesPath)
    assert a3m_path.exists()

    content = a3m_path.read_text()
    lines = content.strip().split("\n")

    # First entry is the query
    assert lines[0] == ">query"
    assert lines[1] == QUERY_SEQ

    # Second entry is the template
    assert "1abc_A/" in lines[2]
    assert "mol:protein" in lines[2]
    assert "length:17" in lines[2]
    # The aligned sequence should reconstruct properly
    assert lines[3] == "AC--FGHIK--NPQRST"


def test_to_protenix_with_output_dir_insertion(tmp_path):
    """A3M generation correctly inserts lowercase for unmapped template positions."""
    template_chain_seq = "ACFGHXIKNPQRSTXXXX"  # 18 residues, X at position 5

    config = UniAF3Config(
        sequences=[
            ProteinSeq(
                polymer_type=PolymerType.Protein,
                id="A",
                sequence=QUERY_SEQ,
                templates=[
                    StructuralTemplate(
                        path="/fake/2def.cif",
                        query_idx=EXPECTED_QUERY_IDX,
                        template_idx=[0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13],
                        template_chains=["B"],
                    )
                ],
            )
        ]
    )

    with (
        pytest.warns(UserWarning),
        patch(
            "uniaf3.adapters.protenix._read_chain_sequence",
            return_value=(template_chain_seq, 18),
        ),
    ):
        result = _to_protenix(config, name="test", strict=False, output_dir=tmp_path)

    pc = result.sequences[0].proteinChain
    assert pc is not None
    content = Path(pc.templatesPath).read_text()
    lines = content.strip().split("\n")
    assert lines[3] == "AC--FGHxIK--NPQRST"


def test_to_protenix_multiple_templates(tmp_path):
    """Multiple templates are combined into a single A3M file."""
    config = UniAF3Config(
        sequences=[
            ProteinSeq(
                polymer_type=PolymerType.Protein,
                id="A",
                sequence=QUERY_SEQ,
                templates=[
                    StructuralTemplate(
                        path="/fake/1abc.cif",
                        query_idx=[0, 1, 2],
                        template_idx=[0, 1, 2],
                        template_chains=["A"],
                    ),
                    StructuralTemplate(
                        path="/fake/2def.cif",
                        query_idx=[0, 1, 2, 3],
                        template_idx=[0, 1, 2, 3],
                        template_chains=["B"],
                    ),
                ],
            )
        ]
    )

    with (
        pytest.warns(UserWarning),
        patch(
            "uniaf3.adapters.protenix._read_chain_sequence",
            side_effect=[("ACF", 3), ("ACFG", 4)],
        ),
    ):
        result = _to_protenix(config, name="test", strict=False, output_dir=tmp_path)

    pc = result.sequences[0].proteinChain
    assert pc is not None
    content = Path(pc.templatesPath).read_text()
    # Should have query + 2 template entries = 6 lines (3 pairs of desc+seq)
    lines = content.strip().split("\n")
    assert lines[0] == ">query"
    assert "1abc_A/" in lines[2]
    assert "2def_B/" in lines[4]


def test_to_protenix_unreadable_template_skipped(tmp_path):
    """Templates with unreadable CIF files are skipped with a warning."""
    config = UniAF3Config(
        sequences=[
            ProteinSeq(
                polymer_type=PolymerType.Protein,
                id="A",
                sequence=QUERY_SEQ,
                templates=[
                    StructuralTemplate(
                        path="/fake/bad.cif",
                        query_idx=[0, 1],
                        template_idx=[0, 1],
                        template_chains=["A"],
                    )
                ],
            )
        ]
    )

    with (
        pytest.warns(UserWarning) as records,
        patch(
            "uniaf3.adapters.protenix._read_chain_sequence",
            side_effect=FileNotFoundError("not found"),
        ),
    ):
        result = _to_protenix(config, name="test", strict=False, output_dir=tmp_path)

    assert any("Cannot read template" in str(w.message) for w in records)
    pc = result.sequences[0].proteinChain
    assert pc is not None
    # No A3M written since the only template was skipped
    assert pc.templatesPath is None


def test_to_protenix_without_output_dir_fallback():
    """Without output_dir, templates cannot be converted to A3M; a warning is emitted and templatesPath is left unset."""
    config = UniAF3Config(
        sequences=[
            ProteinSeq(
                polymer_type=PolymerType.Protein,
                id="A",
                sequence=QUERY_SEQ,
                templates=[
                    StructuralTemplate(path="/some/1abc.cif"),
                ],
            )
        ]
    )

    with pytest.warns(UserWarning):
        result = _to_protenix(config, name="test", strict=False)

    pc = result.sequences[0].proteinChain
    assert pc is not None
    assert pc.templatesPath is None


# ─── Roundtrip test ────────────────────────────────────────────────────


def test_roundtrip_a3m(tmp_path):
    """A3M → parse → StructuralTemplates → A3M → parse → same indices."""
    # Step 1: Parse A3M into StructuralTemplates
    a3m_path = tmp_path / "input.a3m"
    a3m_path.write_text(TEMPLATE_A3M)

    job = ProtenixJob(
        name="test",
        sequences=[
            ProtenixSequenceEntry(
                proteinChain=ProtenixProteinChain(
                    sequence=QUERY_SEQ,
                    count=1,
                    templatesPath=str(a3m_path),
                )
            )
        ],
    )

    with (
        pytest.warns(UserWarning),
        patch("uniaf3.utils.download_files"),
    ):
        uni_config = _from_protenix(job, output_dir=tmp_path)

    prot = uni_config.sequences[0]
    assert isinstance(prot, ProteinSeq)
    assert prot.templates is not None
    original_q_idx = prot.templates[0].query_idx
    original_t_idx = prot.templates[0].template_idx

    # Step 2: Convert back to Protenix A3M
    # The template chain seq: 17 residues (positions in CIF)
    # Positions 0-12 are the aligned residues: A,C,F,G,H,I,K,N,P,Q,R,S,T
    # Positions 13-16 are extra chain residues: X,X,X,X
    template_chain_seq = "ACFGHIKNPQRSTXXXX"

    output_dir = tmp_path / "output"
    with (
        pytest.warns(UserWarning),
        patch(
            "uniaf3.adapters.protenix._read_chain_sequence",
            return_value=(template_chain_seq, 17),
        ),
    ):
        ptx_job = _to_protenix(
            uni_config, name="test", strict=False, output_dir=output_dir
        )

    pc = ptx_job.sequences[0].proteinChain
    assert pc is not None
    assert pc.templatesPath is not None

    # Step 3: Parse the generated A3M back
    generated_a3m = Path(pc.templatesPath).read_text()
    hits = HmmsearchA3MParser.parse(QUERY_SEQ, generated_a3m)
    assert len(hits) == 1

    mapping = hits[0].query_to_hit_mapping
    roundtrip_q_idx = sorted(mapping.keys())
    roundtrip_t_idx = [mapping[q] for q in roundtrip_q_idx]

    assert roundtrip_q_idx == original_q_idx
    assert roundtrip_t_idx == original_t_idx
