# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for MCP heavy data detector and adapter.

Tests cover:
- RegexDetector: input schema detection, output description detection, content sniffing
- AdaptedMCPToolWrapper: schema rewriting, execute flow with mock storage
- adapt_mcp_tools_async: end-to-end integration
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from api.mcp.detector import (
    DetectionResult,
    HeavyDataDetector,
    RegexDetector,
    extract_smiles_from_content,
)
from api.mcp.adapter import (
    AdaptedMCPToolWrapper,
    _ref_param_name,
    adapt_mcp_tools_async,
)
from api.mcp.wrapper import MCPToolWrapper
from api.mcp.client import MCPToolSchema
from api.tools.base import DataIOCategory, ToolResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mcp_tool(
    name: str,
    description: str,
    input_schema: dict,
    server_id: str = "srv123",
    server_name: str = "TestServer",
) -> MCPToolWrapper:
    """Create an MCPToolWrapper with a mock client and given schema."""
    mock_client = MagicMock()
    mock_client.call_tool = AsyncMock(
        return_value={
            "content": "mock result",
            "raw": {},
            "isError": False,
        }
    )

    schema = MCPToolSchema(
        name=name,
        description=description,
        input_schema=input_schema,
    )
    return MCPToolWrapper(
        mcp_client=mock_client,
        schema=schema,
        server_id=server_id,
        server_name=server_name,
    )


# ---------------------------------------------------------------------------
# _ref_param_name tests
# ---------------------------------------------------------------------------


class TestRefParamName:
    def test_smiles_becomes_smiles_ref(self):
        assert _ref_param_name("smiles", "smiles") == "smiles_ref"

    def test_ligand_smiles_becomes_ligand_smiles_ref(self):
        assert _ref_param_name("ligand_smiles", "smiles") == "ligand_smiles_ref"

    def test_molecule_becomes_molecule_smiles_ref(self):
        assert _ref_param_name("molecule", "smiles") == "molecule_smiles_ref"

    def test_pdb_becomes_pdb_ref(self):
        assert _ref_param_name("pdb", "pdb") == "pdb_ref"

    def test_input_pdb_becomes_input_pdb_ref(self):
        assert _ref_param_name("input_pdb", "pdb") == "input_pdb_ref"

    def test_sequence_becomes_sequence_ref(self):
        assert _ref_param_name("sequence", "sequence") == "sequence_ref"

    def test_protein_becomes_protein_sequence_ref(self):
        assert _ref_param_name("protein", "sequence") == "protein_sequence_ref"

    def test_a_smiles_param_carrying_safe_becomes_smiles_safe_ref(self):
        # GenMol's field is named `smiles` but takes SAFE. Splicing the detected
        # type in is what tells the model which of the two it must supply.
        assert _ref_param_name("smiles", "safe") == "smiles_safe_ref"

    def test_safe_becomes_safe_ref(self):
        assert _ref_param_name("safe", "safe") == "safe_ref"


# ---------------------------------------------------------------------------
# RegexDetector input detection tests
# ---------------------------------------------------------------------------


class TestRegexDetectorInputs:
    """Test input parameter detection from JSON Schema."""

    @pytest.fixture
    def detector(self):
        return RegexDetector()

    @pytest.mark.asyncio
    async def test_detects_smiles_by_name(self, detector):
        schema = {
            "type": "object",
            "properties": {
                "smiles": {"type": "string", "description": "input molecule"},
            },
        }
        results = await detector.detect_inputs(schema)
        assert len(results) == 1
        assert results[0].param_name == "smiles"
        assert results[0].detected_type == "smiles"
        assert results[0].confidence >= 0.8

    @pytest.mark.asyncio
    async def test_detects_smiles_by_description(self, detector):
        schema = {
            "type": "object",
            "properties": {
                "molecule": {
                    "type": "string",
                    "description": "The SMILES notation for the compound",
                },
            },
        }
        results = await detector.detect_inputs(schema)
        assert len(results) == 1
        assert results[0].detected_type == "smiles"
        assert results[0].confidence >= 0.5

    @pytest.mark.asyncio
    async def test_detects_canonical_smiles(self, detector):
        schema = {
            "type": "object",
            "properties": {
                "canonical_smiles": {
                    "type": "string",
                    "description": "Canonical SMILES representation",
                },
            },
        }
        results = await detector.detect_inputs(schema)
        assert len(results) == 1
        assert results[0].detected_type == "smiles"
        assert results[0].confidence == 1.0  # name + description match

    @pytest.mark.asyncio
    async def test_detects_pdb_by_name(self, detector):
        schema = {
            "type": "object",
            "properties": {
                "pdb_file": {
                    "type": "string",
                    "description": "PDB file contents",
                },
            },
        }
        results = await detector.detect_inputs(schema)
        assert len(results) == 1
        assert results[0].detected_type == "pdb"

    @pytest.mark.asyncio
    async def test_detects_sequence_by_name(self, detector):
        schema = {
            "type": "object",
            "properties": {
                "protein_sequence": {
                    "type": "string",
                    "description": "The amino acid sequence",
                },
            },
        }
        results = await detector.detect_inputs(schema)
        assert len(results) == 1
        assert results[0].detected_type == "sequence"
        assert results[0].confidence == 1.0

    @pytest.mark.asyncio
    async def test_detects_sdf_by_description(self, detector):
        schema = {
            "type": "object",
            "properties": {
                "structure": {
                    "type": "string",
                    "description": "Molecule in SDF format with 3D coordinates",
                },
            },
        }
        results = await detector.detect_inputs(schema)
        assert len(results) == 1
        assert results[0].detected_type == "sdf"

    @pytest.mark.asyncio
    async def test_detects_mol2(self, detector):
        schema = {
            "type": "object",
            "properties": {
                "mol2_file": {
                    "type": "string",
                    "description": "Tripos MOL2 format data",
                },
            },
        }
        results = await detector.detect_inputs(schema)
        assert len(results) == 1
        assert results[0].detected_type == "mol2"

    @pytest.mark.asyncio
    async def test_detects_pdbqt(self, detector):
        schema = {
            "type": "object",
            "properties": {
                "pdbqt": {
                    "type": "string",
                    "description": "PDBQT file for AutoDock",
                },
            },
        }
        results = await detector.detect_inputs(schema)
        assert len(results) == 1
        assert results[0].detected_type == "pdbqt"

    @pytest.mark.asyncio
    async def test_ignores_non_string_params(self, detector):
        schema = {
            "type": "object",
            "properties": {
                "smiles": {"type": "integer", "description": "not actually a SMILES"},
            },
        }
        results = await detector.detect_inputs(schema)
        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_ignores_non_heavy_params(self, detector):
        schema = {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "limit": {"type": "integer", "description": "Max results"},
            },
        }
        results = await detector.detect_inputs(schema)
        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_detects_multiple_heavy_params(self, detector):
        schema = {
            "type": "object",
            "properties": {
                "smiles": {"type": "string", "description": "SMILES string"},
                "pdb_data": {"type": "string", "description": "PDB structure file"},
                "name": {"type": "string", "description": "Compound name"},
            },
        }
        results = await detector.detect_inputs(schema)
        assert len(results) == 2
        types = {r.detected_type for r in results}
        assert types == {"smiles", "pdb"}

    @pytest.mark.asyncio
    async def test_pdb_id_not_matched(self, detector):
        """PDB ID/code params should NOT be detected as heavy data."""
        schema = {
            "type": "object",
            "properties": {
                "pdb_id": {
                    "type": "string",
                    "description": "The PDB ID or accession code",
                },
            },
        }
        results = await detector.detect_inputs(schema)
        # "pdb_id" doesn't match any name patterns (they match pdb, pdb_data, pdb_file, etc.)
        assert len(results) == 0


# ---------------------------------------------------------------------------
# Coverage of every first-class data type, and pattern ordering
# ---------------------------------------------------------------------------


class TestFirstClassTypeCoverage:
    """Each Aevom first-class type must be detectable, not just the original set.

    A type the detector cannot see is a type that reaches the model as raw text,
    which is the failure this whole layer exists to prevent.
    """

    @pytest.fixture
    def detector(self):
        return RegexDetector()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "param_name,expected",
        [
            ("smiles", "smiles"),
            ("sdf_data", "sdf"),
            ("pdb_content", "pdb"),
            ("mmcif", "mmcif"),
            ("cif_file", "mmcif"),
            ("sequence", "sequence"),
            ("dna_sequence", "dna"),
            ("rna_sequence", "rna"),
            ("msa", "msa"),
            ("a3m", "msa"),
            ("alignment", "msa"),
        ],
    )
    async def test_detects_by_name(self, detector, param_name, expected):
        schema = {
            "type": "object",
            "properties": {param_name: {"type": "string"}},
        }
        results = await detector.detect_inputs(schema)
        assert [r.detected_type for r in results] == [expected]

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "description,expected",
        [
            ("An mmCIF structure block", "mmcif"),
            ("A PDBx/mmCIF formatted structure", "mmcif"),
            ("A DNA sequence of the target gene", "dna"),
            ("The nucleotide sequence to search", "dna"),
            ("An mRNA transcript", "rna"),
            ("Multiple sequence alignment in A3M format", "msa"),
            ("Stockholm formatted aligned sequences", "msa"),
        ],
    )
    async def test_detects_by_description(self, detector, description, expected):
        schema = {
            "type": "object",
            "properties": {
                "payload": {"type": "string", "description": description}
            },
        }
        results = await detector.detect_inputs(schema)
        assert [r.detected_type for r in results] == [expected]

    @pytest.mark.asyncio
    async def test_dna_description_wins_over_generic_sequence(self, detector):
        # `sequence` matches the protein pattern by name, but a DNA description
        # is the stronger signal and dna is declared first.
        schema = {
            "type": "object",
            "properties": {
                "sequence": {
                    "type": "string",
                    "description": "The DNA sequence to align",
                }
            },
        }
        results = await detector.detect_inputs(schema)
        assert [r.detected_type for r in results] == ["dna"]

    @pytest.mark.asyncio
    async def test_mmcif_description_mentioning_the_pdb_stays_mmcif(self, detector):
        # mmCIF docs routinely say "mmCIF file from the PDB"; declaring mmcif
        # before pdb keeps that from being classified as a PDB file.
        schema = {
            "type": "object",
            "properties": {
                "structure": {
                    "type": "string",
                    "description": "An mmCIF file downloaded from the PDB",
                }
            },
        }
        results = await detector.detect_inputs(schema)
        assert [r.detected_type for r in results] == ["mmcif"]


class TestSafeVersusSmiles:
    """SAFE and SMILES are the hardest pair the detector has to separate.

    A SAFE string *is* a valid SMILES, and GenMol names its request field
    `smiles` while requiring SAFE in it. Nothing about the value or the parameter
    name distinguishes them, so the description is the only signal, and pattern
    ordering is what lets a weaker SAFE match beat a stronger SMILES one.
    Getting this wrong is silent: the wrong notation is accepted and simply
    generates worse molecules.
    """

    @pytest.fixture
    def detector(self):
        return RegexDetector()

    @pytest.mark.asyncio
    async def test_a_smiles_named_param_documented_as_safe_is_safe(self, detector):
        # The GenMol case verbatim. `smiles` scores 1.0 on the smiles pattern
        # (name and description) against 0.5 here, so this passes only because
        # safe is declared first and matching stops at the first pattern over
        # threshold.
        schema = {
            "type": "object",
            "properties": {
                "smiles": {
                    "type": "string",
                    "description": "Input molecule in SAFE notation",
                }
            },
        }
        results = await detector.detect_inputs(schema)
        assert [r.detected_type for r in results] == ["safe"]

    @pytest.mark.asyncio
    async def test_a_plain_smiles_param_stays_smiles(self, detector):
        # The other half of the bargain: putting safe first must not capture
        # ordinary SMILES parameters.
        schema = {
            "type": "object",
            "properties": {
                "smiles": {
                    "type": "string",
                    "description": "SMILES string of the ligand to dock",
                }
            },
        }
        results = await detector.detect_inputs(schema)
        assert [r.detected_type for r in results] == ["smiles"]

    @pytest.mark.asyncio
    async def test_the_word_safe_in_ordinary_prose_does_not_match(self, detector):
        # `\bSAFE\b` is compiled case-sensitively for exactly this reason: under
        # the module-wide re.IGNORECASE default it would claim every parameter
        # whose docs happen to reassure the caller.
        schema = {
            "type": "object",
            "properties": {
                "smiles": {
                    "type": "string",
                    "description": "The ligand. This call is safe to retry.",
                }
            },
        }
        results = await detector.detect_inputs(schema)
        assert [r.detected_type for r in results] == ["smiles"]

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "param_name",
        ["safe", "safe_smiles", "safe_string", "safe_notation", "safe_input"],
    )
    async def test_detects_safe_by_name(self, detector, param_name):
        schema = {"type": "object", "properties": {param_name: {"type": "string"}}}
        results = await detector.detect_inputs(schema)
        assert [r.detected_type for r in results] == ["safe"]

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "description",
        [
            "Molecule in SAFE-SMILES form",
            "A SAFE encoding of the scaffold",
            "Sequential Attachment-based Fragment Embedding of the input",
        ],
    )
    async def test_detects_safe_by_description(self, detector, description):
        schema = {
            "type": "object",
            "properties": {"payload": {"type": "string", "description": description}},
        }
        results = await detector.detect_inputs(schema)
        assert [r.detected_type for r in results] == ["safe"]

    def test_a_generation_mask_in_content_sniffs_as_safe(self, detector):
        # `[*{20-30}]` is the one SAFE construct that is not valid SMILES, so it
        # is the only content signal worth having. detect_content returns every
        # match and callers take the highest-confidence one; safe and smiles tie
        # at 0.9, and max() keeps the first, which is safe.
        results = detector.detect_content(
            "Suggested input: C1CC(=O)NC1.[*{20-30}] for generation"
        )
        assert results[0].detected_type == "safe"

    def test_a_plain_smiles_body_is_not_sniffed_as_safe(self, detector):
        types = {
            r.detected_type
            for r in detector.detect_content(
                "CC(=O)Oc1ccccc1C(=O)O\nCn1cnc2c1c(=O)n(C)c(=O)n2C"
            )
        }
        assert types == {"smiles"}


class TestContentSniffingNewTypes:
    @pytest.fixture
    def detector(self):
        return RegexDetector()

    def test_sniffs_an_mmcif_block(self, detector):
        content = "data_1ABC\n_entry.id 1ABC\n_atom_site.group_PDB\nATOM\n"
        types = {r.detected_type for r in detector.detect_content(content)}
        assert "mmcif" in types

    def test_sniffs_a_long_nucleotide_run_as_dna(self, detector):
        types = {r.detected_type for r in detector.detect_content("ACGT" * 20)}
        assert "dna" in types

    def test_sniffs_a_stockholm_alignment(self, detector):
        content = "# STOCKHOLM 1.0\nseq1 MKV--LA\nseq2 MKVAALA\n//\n"
        types = {r.detected_type for r in detector.detect_content(content)}
        assert "msa" in types

    def test_a_short_protein_sequence_is_not_dna(self, detector):
        # A/C/G/T are valid amino acid letters, so only a long pure-nucleotide
        # run may be called DNA.
        types = {r.detected_type for r in detector.detect_content("ACGTA")}
        assert "dna" not in types


class TestPatternVocabulary:
    def test_every_pattern_type_is_known_to_storage(self):
        # Detection and storage must share one vocabulary, or a detected
        # artifact lands with no MIME type and no text/binary classification.
        from api.mcp.detector import unknown_pattern_types

        assert unknown_pattern_types() == []


# ---------------------------------------------------------------------------
# RegexDetector output detection tests
# ---------------------------------------------------------------------------


class TestRegexDetectorOutputs:
    """Test output detection from tool description."""

    @pytest.fixture
    def detector(self):
        return RegexDetector()

    @pytest.mark.asyncio
    async def test_detects_sdf_output(self, detector):
        results = await detector.detect_outputs(
            "Converts a SMILES string to 3D coordinates and returns SDF format data."
        )
        assert len(results) == 1
        assert results[0].detected_type == "sdf"
        assert results[0].direction == "output"

    @pytest.mark.asyncio
    async def test_detects_pdb_output(self, detector):
        results = await detector.detect_outputs(
            "This tool predicts protein structure and outputs PDB format data."
        )
        assert len(results) == 1
        assert results[0].detected_type == "pdb"

    @pytest.mark.asyncio
    async def test_detects_smiles_output(self, detector):
        results = await detector.detect_outputs(
            "Generates candidate SMILES strings for the given target."
        )
        assert len(results) == 1
        assert results[0].detected_type == "smiles"

    @pytest.mark.asyncio
    async def test_detects_sequence_output(self, detector):
        results = await detector.detect_outputs(
            "Returns the amino acid sequence for the specified gene."
        )
        assert len(results) == 1
        assert results[0].detected_type == "sequence"

    @pytest.mark.asyncio
    async def test_no_output_detected(self, detector):
        results = await detector.detect_outputs(
            "Returns a similarity score between two molecules."
        )
        assert len(results) == 0


# ---------------------------------------------------------------------------
# RegexDetector content sniffing tests
# ---------------------------------------------------------------------------


class TestRegexDetectorContent:
    """Test runtime content sniffing for heavy data."""

    @pytest.fixture
    def detector(self):
        return RegexDetector()

    def test_detects_sdf_content(self, detector):
        sdf_content = """
     RDKit          3D

  6  6  0  0  0  0  0  0  0  0999 V2000
    1.2124    0.7000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
M  END
$$$$
"""
        results = detector.detect_content(sdf_content)
        types = {r.detected_type for r in results}
        assert "sdf" in types

    def test_detects_pdb_content(self, detector):
        pdb_content = """HEADER    TRANSFERASE                             01-JAN-00   1ABC
REMARK   1 STRUCTURE DETERMINATION
ATOM      1  N   ALA A   1      11.104   6.134  -6.504  1.00  0.00           N
ATOM      2  CA  ALA A   1      11.639   6.071  -5.147  1.00  0.00           C
HETATM    3  O   HOH A   1      12.000   7.000  -4.000  1.00  0.00           O
"""
        results = detector.detect_content(pdb_content)
        types = {r.detected_type for r in results}
        assert "pdb" in types

    def test_detects_fasta_content(self, detector):
        fasta_content = """>sp|P12345|PROT_HUMAN Some protein
MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSH
GSAQVKGHGKKVADALTNAVAHVDDMPNALSALSDLHAHKLRVDPVNFKLL
"""
        results = detector.detect_content(fasta_content)
        types = {r.detected_type for r in results}
        assert "sequence" in types

    def test_detects_mol2_content(self, detector):
        mol2_content = """@<TRIPOS>MOLECULE
benzene
6 6 0 0 0
SMALL
@<TRIPOS>ATOM
      1 C1         1.2124    0.7000    0.0000 C.ar       1 LIG         0.0000
"""
        results = detector.detect_content(mol2_content)
        types = {r.detected_type for r in results}
        assert "mol2" in types

    def test_detects_single_smiles_line(self, detector):
        results = detector.detect_content("CC(=O)Oc1ccccc1C(=O)O")
        types = {r.detected_type for r in results}
        assert "smiles" in types

    def test_detects_multi_smiles_lines(self, detector):
        content = "CC(=O)Oc1ccccc1C(=O)O\nCCO\nc1ccccc1\nCC(C)Cc1ccc(cc1)C(C)C(=O)O"
        results = detector.detect_content(content)
        types = {r.detected_type for r in results}
        assert "smiles" in types

    def test_detects_smiles_in_json_array(self, detector):
        content = '["CC(=O)Oc1ccccc1C(=O)O", "CCO", "c1ccccc1"]'
        results = detector.detect_content(content)
        types = {r.detected_type for r in results}
        assert "smiles" in types

    def test_no_heavy_data_in_plain_text(self, detector):
        results = detector.detect_content("The binding affinity is -9.5 kcal/mol.")
        assert len(results) == 0


# ---------------------------------------------------------------------------
# extract_smiles_from_content tests
# ---------------------------------------------------------------------------


class TestExtractSmilesFromContent:
    """Test SMILES extraction from various output formats."""

    def test_single_smiles_line(self):
        smiles = extract_smiles_from_content("CC(=O)Oc1ccccc1C(=O)O")
        assert smiles == ["CC(=O)Oc1ccccc1C(=O)O"]

    def test_multiple_smiles_one_per_line(self):
        content = "CC(=O)Oc1ccccc1C(=O)O\nCC(C)Cc1ccc(cc1)C(C)C(=O)O\nc1ccc2c(c1)cc1ccc3cccc4ccc2c1c34"
        smiles = extract_smiles_from_content(content)
        assert len(smiles) == 3
        assert smiles[0] == "CC(=O)Oc1ccccc1C(=O)O"

    def test_json_array_of_smiles(self):
        content = '["CC(=O)Oc1ccccc1C(=O)O", "CC(C)Cc1ccc(cc1)C(C)C(=O)O", "c1ccc2c(c1)cc1ccc3cccc4ccc2c1c34"]'
        smiles = extract_smiles_from_content(content)
        assert len(smiles) == 3
        assert "CC(=O)Oc1ccccc1C(=O)O" in smiles

    def test_json_object_with_results_array(self):
        content = '{"results": ["CC(=O)Oc1ccccc1C(=O)O", "c1ccccc1"], "count": 2}'
        smiles = extract_smiles_from_content(content)
        assert len(smiles) == 2

    def test_labeled_lines(self):
        content = """aspirin: CC(=O)Oc1ccccc1C(=O)O
ibuprofen: CC(C)Cc1ccc(cc1)C(C)C(=O)O
caffeine: Cn1c(=O)c2c(ncn2C)n(c1=O)C"""
        smiles = extract_smiles_from_content(content)
        assert len(smiles) == 3
        assert "CC(=O)Oc1ccccc1C(=O)O" in smiles

    def test_mixed_content_with_prose(self):
        content = """Generated 3 candidate molecules:
CC(=O)Oc1ccccc1C(=O)O
CC(C)Cc1ccc(cc1)C(C)C(=O)O
c1ccc2c(c1)cc1ccc3cccc4ccc2c1c34
All molecules pass Lipinski's rule of five."""
        smiles = extract_smiles_from_content(content)
        assert len(smiles) == 3

    def test_deduplicates_smiles(self):
        content = "CC(=O)Oc1ccccc1C(=O)O\nCC(=O)Oc1ccccc1C(=O)O\nCC(=O)Oc1ccccc1C(=O)O"
        smiles = extract_smiles_from_content(content)
        assert len(smiles) == 1

    def test_empty_content(self):
        assert extract_smiles_from_content("") == []

    def test_no_smiles_in_prose(self):
        content = "The binding affinity is -9.5 kcal/mol. No molecules found."
        smiles = extract_smiles_from_content(content)
        assert len(smiles) == 0

    def test_short_tokens_rejected(self):
        """Very short tokens like 'CCO' (3 chars) should be rejected (< 5 char min)."""
        smiles = extract_smiles_from_content("CCO")
        assert len(smiles) == 0

    def test_plain_words_rejected(self):
        """Plain English words shouldn't match even if they contain atom letters."""
        content = "Concentration of sample in the container was measured."
        smiles = extract_smiles_from_content(content)
        assert len(smiles) == 0

    def test_tsv_optimization_output(self):
        """TSV output with scores -- only SMILES column should be extracted.

        This is the key scenario from the production bug where an optimization
        tool returned tab-separated SMILES + scores and the extractor
        incorrectly stored scores as SMILES.
        """
        content = (
            "Step\tSMILES\tScore\tValidity\n"
            "1\tCn1c(=O)c2c(ncn2CCCCC(C)(C)C)n(C)c1=O\t-36.545\t1\n"
            "2\tCC(=O)Oc1ccccc1C(=O)O\t-26.520\t1\n"
            "3\tCC(C)CC(=O)O\t-29.010\t1\n"
            "4\tO=C(O)c1cc(O)c(O)c(O)c1\t-22.100\t1\n"
        )
        smiles = extract_smiles_from_content(content)
        # Should extract only SMILES, not scores or step numbers
        assert "Cn1c(=O)c2c(ncn2CCCCC(C)(C)C)n(C)c1=O" in smiles
        assert "CC(=O)Oc1ccccc1C(=O)O" in smiles
        assert "CC(C)CC(=O)O" in smiles
        assert "O=C(O)c1cc(O)c(O)c(O)c1" in smiles
        # No garbage
        for s in smiles:
            assert not s.startswith("-")
            assert not s.startswith("36")
            assert "\t" not in s

    def test_scores_rejected(self):
        """Numeric scores like -36.545 should never be classified as SMILES."""
        content = "-36.545\n-26.520\n-29.010\n1.23e-4\n417"
        smiles = extract_smiles_from_content(content)
        assert len(smiles) == 0

    def test_csv_optimization_output(self):
        """CSV lines should not be extracted as single tokens."""
        content = (
            "step,smiles,score\n"
            "1,Cn1c(=O)c2c(ncn2CCCCC(C)(C)C)n(C)c1=O,-36.545\n"
            "2,CC(=O)Oc1ccccc1C(=O)O,-26.520\n"
        )
        smiles = extract_smiles_from_content(content)
        # SMILES are extractable via the token regex from CSV lines
        assert "Cn1c(=O)c2c(ncn2CCCCC(C)(C)C)n(C)c1=O" in smiles
        assert "CC(=O)Oc1ccccc1C(=O)O" in smiles
        # Whole CSV lines should NOT be in the result
        for s in smiles:
            assert "," not in s
            assert "36.545" not in s

    def test_unbalanced_parens_rejected(self):
        """Tokens with unbalanced parentheses are not valid SMILES."""
        content = "CC(=O\nscore(123)\n(result"
        smiles = extract_smiles_from_content(content)
        assert len(smiles) == 0

    def test_mixed_format_optimization_output(self):
        """Complex output mixing SMILES, scores, and prose text."""
        content = """Optimization completed after 20 steps.
Best score: -36.545 kcal/mol
Top candidates:
  1. Cn1c(=O)c2c(ncn2CCCCC(C)(C)C)n(C)c1=O (score: -36.545)
  2. CC(=O)Oc1ccccc1C(=O)O (score: -26.520)
  3. O=C(O)c1cc(O)c(O)c(O)c1 (score: -22.100)
Convergence reached at step 15.
Total energy evaluations: 417"""
        smiles = extract_smiles_from_content(content)
        assert "Cn1c(=O)c2c(ncn2CCCCC(C)(C)C)n(C)c1=O" in smiles
        assert "CC(=O)Oc1ccccc1C(=O)O" in smiles
        assert "O=C(O)c1cc(O)c(O)c(O)c1" in smiles
        # No scores, step numbers, or prose
        for s in smiles:
            assert "kcal" not in s
            assert "score" not in s
            assert "step" not in s

    def test_hash_tokens_rejected(self):
        """Hash/comment tokens should not match as SMILES."""
        content = "###\n#####\n# comment\nresult: CC(=O)Oc1ccccc1C(=O)O"
        smiles = extract_smiles_from_content(content)
        assert smiles == ["CC(=O)Oc1ccccc1C(=O)O"]

    def test_bracket_atoms_valid(self):
        """SMILES with bracket atoms like [Fe], [NH4+] should work."""
        content = "[NH4+]\nCC(=O)[O-]\n[Cu](Cl)Cl"
        smiles = extract_smiles_from_content(content)
        assert "[NH4+]" in smiles
        assert "CC(=O)[O-]" in smiles


# ---------------------------------------------------------------------------
# HeavyDataDetector facade tests
# ---------------------------------------------------------------------------


class TestHeavyDataDetector:
    """Test the high-level HeavyDataDetector facade."""

    @pytest.mark.asyncio
    async def test_analyze_tool_with_heavy_inputs(self):
        detector = HeavyDataDetector()
        input_dets, output_dets = await detector.analyze_tool(
            tool_name="convert_smiles",
            tool_description="Converts a molecule to 3D coordinates.",
            input_schema={
                "type": "object",
                "properties": {
                    "smiles": {"type": "string", "description": "SMILES notation"},
                },
            },
        )
        assert len(input_dets) == 1
        assert input_dets[0].detected_type == "smiles"
        assert len(output_dets) == 0

    @pytest.mark.asyncio
    async def test_analyze_tool_with_heavy_outputs(self):
        detector = HeavyDataDetector()
        input_dets, output_dets = await detector.analyze_tool(
            tool_name="predict_structure",
            tool_description="Predicts structure and returns PDB format.",
            input_schema={
                "type": "object",
                "properties": {
                    "gene_name": {"type": "string", "description": "Gene name"},
                },
            },
        )
        assert len(input_dets) == 0
        assert len(output_dets) == 1
        assert output_dets[0].detected_type == "pdb"

    @pytest.mark.asyncio
    async def test_analyze_tool_no_heavy_data(self):
        detector = HeavyDataDetector()
        input_dets, output_dets = await detector.analyze_tool(
            tool_name="search",
            tool_description="Search for compounds by name.",
            input_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                },
            },
        )
        assert len(input_dets) == 0
        assert len(output_dets) == 0


# ---------------------------------------------------------------------------
# AdaptedMCPToolWrapper tests
# ---------------------------------------------------------------------------


class TestAdaptedMCPToolWrapper:
    """Test the adapted wrapper's schema rewriting and execution."""

    def _make_adapted_tool(
        self,
        input_mappings: list[DetectionResult] | None = None,
        output_mappings: list[DetectionResult] | None = None,
        tool_name: str = "convert_molecule",
        description: str = "Converts molecule formats",
        input_schema: dict | None = None,
    ) -> AdaptedMCPToolWrapper:
        """Helper to create an adapted tool with common defaults."""
        if input_schema is None:
            input_schema = {
                "type": "object",
                "properties": {
                    "smiles": {
                        "type": "string",
                        "description": "SMILES string for the molecule",
                    },
                    "format": {
                        "type": "string",
                        "description": "Output format",
                    },
                },
                "required": ["smiles"],
            }
        if input_mappings is None:
            input_mappings = [
                DetectionResult("smiles", "smiles", 1.0, "input"),
            ]
        if output_mappings is None:
            output_mappings = []

        inner = _make_mcp_tool(tool_name, description, input_schema)
        return AdaptedMCPToolWrapper(inner, input_mappings, output_mappings)

    def test_name_delegates_to_inner(self):
        tool = self._make_adapted_tool()
        assert tool.name == "mcp_srv123_convert_molecule"

    def test_data_io_category_input_only(self):
        tool = self._make_adapted_tool(
            input_mappings=[DetectionResult("smiles", "smiles", 1.0, "input")],
            output_mappings=[],
        )
        assert tool.data_io_category == DataIOCategory.INPUT_RETRIEVES

    def test_data_io_category_output_only(self):
        tool = self._make_adapted_tool(
            input_mappings=[],
            output_mappings=[DetectionResult("output", "sdf", 0.7, "output")],
        )
        assert tool.data_io_category == DataIOCategory.OUTPUT_STORES

    def test_data_io_category_both(self):
        tool = self._make_adapted_tool(
            input_mappings=[DetectionResult("smiles", "smiles", 1.0, "input")],
            output_mappings=[DetectionResult("output", "sdf", 0.7, "output")],
        )
        assert tool.data_io_category == DataIOCategory.BOTH

    def test_schema_rewrites_heavy_param(self):
        tool = self._make_adapted_tool()
        schema = tool.get_schema()

        props = schema["function"]["parameters"]["properties"]
        # Original "smiles" param should be gone
        assert "smiles" not in props
        # "smiles_ref" should be present
        assert "smiles_ref" in props
        # Should have x-aevom-reference
        assert props["smiles_ref"]["x-aevom-reference"] == {"data_type": "smiles"}

    def test_schema_preserves_non_heavy_params(self):
        tool = self._make_adapted_tool()
        schema = tool.get_schema()

        props = schema["function"]["parameters"]["properties"]
        assert "format" in props
        assert "x-aevom-reference" not in props["format"]

    def test_schema_updates_required(self):
        tool = self._make_adapted_tool()
        schema = tool.get_schema()

        required = schema["function"]["parameters"]["required"]
        assert "smiles" not in required
        assert "smiles_ref" in required

    def test_adapted_flag(self):
        tool = self._make_adapted_tool()
        assert tool.adapted is True

    def test_adaptation_info(self):
        tool = self._make_adapted_tool()
        info = tool.adaptation_info
        assert "smiles" in info["input_mappings"]
        assert info["input_mappings"]["smiles"]["detected_type"] == "smiles"
        assert info["input_mappings"]["smiles"]["ref_param"] == "smiles_ref"

    @pytest.mark.asyncio
    async def test_execute_unpacks_reference(self):
        """Test that execute retrieves data from storage and passes raw data to inner tool."""
        tool = self._make_adapted_tool()

        # Mock storage
        mock_storage = MagicMock()
        mock_storage.retrieve = AsyncMock(return_value=b"CCO")
        tool.set_storage(mock_storage)

        # Execute with reference argument
        result = await tool.execute({"smiles_ref": "smiles_abc123", "format": "sdf"})

        # Verify storage.retrieve was called with the reference
        mock_storage.retrieve.assert_called_once_with("smiles_abc123")

        # Verify inner tool was called with the raw SMILES
        inner_client = tool._inner._mcp_client
        inner_client.call_tool.assert_called_once()
        call_args = inner_client.call_tool.call_args
        # The arguments should have "smiles" (original name) not "smiles_ref"
        assert call_args[0][1]["smiles"] == "CCO"
        assert call_args[0][1]["format"] == "sdf"
        assert "smiles_ref" not in call_args[0][1]

    @pytest.mark.asyncio
    async def test_execute_returns_error_when_ref_not_found(self):
        """Test error handling when referenced data doesn't exist."""
        from api.storage import ArtifactNotFoundError

        tool = self._make_adapted_tool()

        mock_storage = MagicMock()
        mock_storage.retrieve = AsyncMock(
            side_effect=ArtifactNotFoundError("not found")
        )
        tool.set_storage(mock_storage)

        result = await tool.execute({"smiles_ref": "smiles_nonexistent"})

        assert result.content_type.value == "error"
        assert "not found" in result.data.lower() or "smiles_nonexistent" in result.data

    @pytest.mark.asyncio
    async def test_execute_returns_error_when_no_storage(self):
        """Test error when storage service is not available."""
        tool = self._make_adapted_tool()
        # Don't set storage

        result = await tool.execute({"smiles_ref": "smiles_abc123"})

        assert result.content_type.value == "error"
        assert "storage" in result.data.lower()

    @pytest.mark.asyncio
    async def test_execute_stores_heavy_output(self):
        """Test that heavy output is detected and stored."""
        tool = self._make_adapted_tool(
            input_mappings=[],
            output_mappings=[DetectionResult("output", "sdf", 0.7, "output")],
            input_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "search query"},
                },
            },
        )

        # Mock inner tool to return SDF content
        sdf_content = """
     RDKit          3D

  6  6  0  0  0  0  0  0  0  0999 V2000
    1.2124    0.7000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
M  END
$$$$"""
        tool._inner._mcp_client.call_tool = AsyncMock(
            return_value={
                "content": sdf_content,
                "raw": {},
                "isError": False,
            }
        )

        mock_storage = MagicMock()
        mock_storage.store = AsyncMock(return_value="sdf_out12345")
        tool.set_storage(mock_storage)

        result = await tool.execute({"query": "benzene"})

        # Should have stored the output
        mock_storage.store.assert_called_once()
        store_call = mock_storage.store.call_args
        assert store_call.kwargs["data_type"] == "sdf"

        # Result should be a data reference
        assert result.content_type.value == "data_reference"
        assert result.data == "sdf_out12345"

    @pytest.mark.asyncio
    async def test_execute_stores_multi_smiles_output(self):
        """Test that multiple SMILES in output are stored individually."""
        tool = self._make_adapted_tool(
            input_mappings=[],
            output_mappings=[DetectionResult("output", "smiles", 0.7, "output")],
            input_schema={
                "type": "object",
                "properties": {
                    "target": {"type": "string", "description": "target protein"},
                },
            },
        )

        # Tool returns multiple SMILES, one per line
        smiles_output = (
            "CC(=O)Oc1ccccc1C(=O)O\n"
            "CC(C)Cc1ccc(cc1)C(C)C(=O)O\n"
            "c1ccc2c(c1)cc1ccc3cccc4ccc2c1c34"
        )
        tool._inner._mcp_client.call_tool = AsyncMock(
            return_value={
                "content": smiles_output,
                "raw": {},
                "isError": False,
            }
        )

        # Storage returns incrementing refs
        call_count = 0

        async def mock_store(**kwargs):
            nonlocal call_count
            call_count += 1
            return f"smiles_out{call_count:04d}"

        mock_storage = MagicMock()
        mock_storage.store = AsyncMock(side_effect=mock_store)
        tool.set_storage(mock_storage)

        result = await tool.execute({"target": "COX-2"})

        # Should have stored 3 individual SMILES
        assert mock_storage.store.call_count == 3

        # Each store call should be for data_type="smiles"
        for call in mock_storage.store.call_args_list:
            assert call.kwargs["data_type"] == "smiles"

        # Result should be a data reference with all refs in metadata
        assert result.content_type.value == "data_reference"
        assert result.metadata["count"] == 3
        assert len(result.metadata["all_references"]) == 3
        assert result.data == "smiles_out0001"  # first reference is primary

    @pytest.mark.asyncio
    async def test_execute_stores_json_array_smiles_output(self):
        """Test that a JSON array of SMILES is parsed and stored individually."""
        tool = self._make_adapted_tool(
            input_mappings=[],
            output_mappings=[DetectionResult("output", "smiles", 0.7, "output")],
            input_schema={
                "type": "object",
                "properties": {
                    "target": {"type": "string", "description": "target protein"},
                },
            },
        )

        # Tool returns JSON array of SMILES
        json_output = '["CC(=O)Oc1ccccc1C(=O)O", "CC(C)Cc1ccc(cc1)C(C)C(=O)O"]'
        tool._inner._mcp_client.call_tool = AsyncMock(
            return_value={
                "content": json_output,
                "raw": {},
                "isError": False,
            }
        )

        call_count = 0

        async def mock_store(**kwargs):
            nonlocal call_count
            call_count += 1
            return f"smiles_j{call_count:04d}"

        mock_storage = MagicMock()
        mock_storage.store = AsyncMock(side_effect=mock_store)
        tool.set_storage(mock_storage)

        result = await tool.execute({"target": "EGFR"})

        assert mock_storage.store.call_count == 2
        assert result.content_type.value == "data_reference"
        assert result.metadata["count"] == 2

    @pytest.mark.asyncio
    async def test_execute_passes_through_non_heavy_output(self):
        """Test that non-heavy output passes through unchanged."""
        tool = self._make_adapted_tool(
            input_mappings=[],
            output_mappings=[],
            input_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "search query"},
                },
            },
        )

        tool._inner._mcp_client.call_tool = AsyncMock(
            return_value={
                "content": "Found 3 results with binding affinity > -8.0 kcal/mol",
                "raw": {},
                "isError": False,
            }
        )

        result = await tool.execute({"query": "aspirin"})

        assert result.content_type.value == "text"


# ---------------------------------------------------------------------------
# adapt_mcp_tools_async integration tests
# ---------------------------------------------------------------------------


class TestAdaptMcpToolsAsync:
    """Test the end-to-end adaptation pipeline."""

    @pytest.mark.asyncio
    async def test_adapts_tool_with_smiles_param(self):
        tool = _make_mcp_tool(
            name="convert",
            description="Convert molecule format",
            input_schema={
                "type": "object",
                "properties": {
                    "smiles": {"type": "string", "description": "SMILES notation"},
                },
            },
        )

        result = await adapt_mcp_tools_async([tool])

        assert len(result) == 1
        assert isinstance(result[0], AdaptedMCPToolWrapper)

    @pytest.mark.asyncio
    async def test_passes_through_text_only_tool(self):
        tool = _make_mcp_tool(
            name="search",
            description="Search for compounds",
            input_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search term"},
                    "limit": {"type": "integer", "description": "Max results"},
                },
            },
        )

        result = await adapt_mcp_tools_async([tool])

        assert len(result) == 1
        assert isinstance(result[0], MCPToolWrapper)

    @pytest.mark.asyncio
    async def test_mixed_tools(self):
        """Test a mix of heavy and non-heavy tools."""
        heavy_tool = _make_mcp_tool(
            name="dock",
            description="Dock ligand and returns PDB complex",
            input_schema={
                "type": "object",
                "properties": {
                    "smiles": {"type": "string", "description": "Ligand SMILES"},
                    "pdb_data": {
                        "type": "string",
                        "description": "Receptor PDB file data",
                    },
                },
            },
        )
        text_tool = _make_mcp_tool(
            name="info",
            description="Get molecule info",
            input_schema={
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Molecule name"},
                },
            },
        )

        result = await adapt_mcp_tools_async([heavy_tool, text_tool])

        assert len(result) == 2
        assert isinstance(result[0], AdaptedMCPToolWrapper)
        assert isinstance(result[1], MCPToolWrapper)

    @pytest.mark.asyncio
    async def test_adapted_tool_schema_has_refs(self):
        """End-to-end: adapted tool schema uses reference parameters."""
        tool = _make_mcp_tool(
            name="predict",
            description="Predict protein structure, returns PDB",
            input_schema={
                "type": "object",
                "properties": {
                    "sequence": {
                        "type": "string",
                        "description": "Amino acid sequence in FASTA format",
                    },
                },
                "required": ["sequence"],
            },
        )

        result = await adapt_mcp_tools_async([tool])
        adapted = result[0]
        assert isinstance(adapted, AdaptedMCPToolWrapper)

        schema = adapted.get_schema()
        props = schema["function"]["parameters"]["properties"]
        required = schema["function"]["parameters"]["required"]

        # Original param replaced
        assert "sequence" not in props
        assert "sequence_ref" in props
        assert props["sequence_ref"]["x-aevom-reference"] == {"data_type": "sequence"}

        # Required updated
        assert "sequence" not in required
        assert "sequence_ref" in required
