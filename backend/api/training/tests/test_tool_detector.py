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

"""Unit tests for tool detector and trajectory analysis."""

import pytest
from api.training.tool_detector import (
    TrajectoryAnalyzer,
    TrajectoryAnalysis,
    PythonCodeAnalysis,
    DockingAnalysis,
    analyze_tool_calls,
    extract_docking_results,
    extract_computed_metrics,
)


class TestPythonCodeAnalysis:
    """Tests for Python code analysis."""
    
    @pytest.fixture
    def analyzer(self):
        """Create a trajectory analyzer."""
        return TrajectoryAnalyzer()
    
    def test_analyze_simple_code(self, analyzer):
        """Test analysis of simple Python code."""
        call = {
            "name": "python_exec",
            "arguments": {"code": "x = 1 + 2\nprint(x)"},
            "output": "3",
            "success": True,
        }
        
        result = analyzer._analyze_python_code(call)
        
        assert result.line_count == 2
        assert result.prints_results is True
        assert result.uses_rdkit is False
    
    def test_analyze_rdkit_imports(self, analyzer):
        """Test detection of RDKit imports."""
        call = {
            "name": "python_exec",
            "arguments": {"code": "from rdkit import Chem\nfrom rdkit.Chem import Descriptors"},
            "output": "",
            "success": True,
        }
        
        result = analyzer._analyze_python_code(call)
        
        assert result.has_imports is True
        assert result.uses_rdkit is True
        assert len(result.rdkit_imports) >= 1
    
    def test_analyze_rdkit_function_calls(self, analyzer):
        """Test detection of RDKit function calls."""
        call = {
            "name": "python_exec",
            "arguments": {"code": "mol = Chem.MolFromSmiles('CCO')\nlogp = Descriptors.MolLogP(mol)"},
            "output": "",
            "success": True,
        }
        
        result = analyzer._analyze_python_code(call)
        
        assert result.uses_rdkit is True
        assert "MolFromSmiles" in result.rdkit_calls
        assert "MolLogP" in result.rdkit_calls
    
    def test_analyze_molecule_creation(self, analyzer):
        """Test detection of molecule creation."""
        call = {
            "name": "python_exec",
            "arguments": {"code": "mol = Chem.MolFromSmiles('CCO')"},
            "output": "",
            "success": True,
        }
        
        result = analyzer._analyze_python_code(call)
        
        assert result.creates_molecule is True
    
    def test_analyze_descriptor_computation(self, analyzer):
        """Test detection of descriptor computation."""
        call = {
            "name": "python_exec",
            "arguments": {"code": "logp = MolLogP(mol)\ntpsa = TPSA(mol)\nqed = QED.qed(mol)"},
            "output": "",
            "success": True,
        }
        
        result = analyzer._analyze_python_code(call)
        
        assert result.computes_descriptors is True
        assert len(result.descriptor_names) >= 2


class TestDockingAnalysis:
    """Tests for docking analysis."""
    
    @pytest.fixture
    def analyzer(self):
        """Create a trajectory analyzer."""
        return TrajectoryAnalyzer()
    
    def test_analyze_successful_docking(self, analyzer):
        """Test analysis of successful docking."""
        call = {
            "name": "vina_dock",
            "arguments": {"receptor": "receptor_001", "ligand": "sdf_001"},
            "output": "Docking complete. Affinity: -8.5 kcal/mol",
            "success": True,
        }
        
        result = analyzer._analyze_docking(call)
        
        assert result.success is True
        assert result.receptor_ref == "receptor_001"
        assert result.ligand_ref == "sdf_001"
        assert result.affinity == pytest.approx(-8.5, rel=0.01)
    
    def test_analyze_docking_with_score(self, analyzer):
        """Test affinity extraction from 'score' output."""
        call = {
            "name": "vina_dock",
            "arguments": {},
            "output": "Binding score: -7.2",
            "success": True,
        }
        
        result = analyzer._analyze_docking(call)
        
        assert result.affinity == pytest.approx(-7.2, rel=0.01)
    
    def test_analyze_failed_docking(self, analyzer):
        """Test analysis of failed docking."""
        call = {
            "name": "vina_dock",
            "arguments": {},
            "output": "Error: receptor file not found",
            "success": False,
        }
        
        result = analyzer._analyze_docking(call)
        
        assert result.success is False
        assert result.error is not None


class TestTrajectoryAnalysis:
    """Tests for complete trajectory analysis."""
    
    @pytest.fixture
    def analyzer(self):
        """Create a trajectory analyzer."""
        return TrajectoryAnalyzer()
    
    def test_analyze_empty_trajectory(self, analyzer):
        """Test analysis of empty trajectory."""
        result = analyzer.analyze_trajectory([])
        
        assert result.total_tool_calls == 0
        assert result.used_docking is False
        assert result.used_python is False
    
    def test_analyze_mixed_trajectory(self, analyzer):
        """Test analysis of trajectory with multiple tool types."""
        tool_calls = [
            {"name": "vina_dock", "output": "Affinity: -8.5", "success": True},
            {"name": "python_exec", "arguments": {"code": "from rdkit import Chem"}, "output": "", "success": True},
            {"name": "load_molecule", "output": "Loaded", "success": True},
        ]
        
        result = analyzer.analyze_trajectory(tool_calls)
        
        assert result.total_tool_calls == 3
        assert result.used_docking is True
        assert result.used_python is True
        assert result.used_rdkit is True
        assert result.tool_call_counts["vina_dock"] == 1
        assert result.tool_call_counts["python_exec"] == 1
    
    def test_analyze_counts_successes(self, analyzer):
        """Test that successes and failures are counted."""
        tool_calls = [
            {"name": "tool1", "output": "", "success": True},
            {"name": "tool2", "output": "", "success": True},
            {"name": "tool3", "output": "", "success": False},
        ]
        
        result = analyzer.analyze_trajectory(tool_calls)
        
        assert result.success_count == 2
        assert result.error_count == 1
    
    def test_get_summary_metrics(self, analyzer):
        """Test summary metrics extraction."""
        tool_calls = [
            {"name": "vina_dock", "output": "Affinity: -8.5", "success": True},
            {"name": "vina_dock", "output": "Affinity: -6.1", "success": True},
            {"name": "python_exec", "arguments": {"code": "print(1)"}, "output": "1", "success": True},
        ]
        
        analysis = analyzer.analyze_trajectory(tool_calls)
        metrics = analyzer.get_summary_metrics(analysis)
        
        assert metrics["total_tool_calls"] == 3
        assert metrics["success_rate"] == 1.0
        assert metrics["docking_count"] == 2
        assert metrics["docking_success_rate"] == 1.0


class TestConvenienceFunctions:
    """Tests for convenience functions."""
    
    def test_analyze_tool_calls(self):
        """Test the analyze_tool_calls convenience function."""
        tool_calls = [
            {"name": "vina_dock", "output": "Affinity: -8.5", "success": True},
        ]
        
        result = analyze_tool_calls(tool_calls)
        
        assert "total_tool_calls" in result
        assert result["total_tool_calls"] == 1
    
    def test_extract_docking_results(self):
        """Test the extract_docking_results convenience function."""
        tool_calls = [
            {"name": "vina_dock", "arguments": {"receptor": "r1", "ligand": "l1"}, "output": "Affinity: -8.5", "success": True},
            {"name": "python_exec", "output": "", "success": True},
            {"name": "vina_dock", "arguments": {"receptor": "r2", "ligand": "l2"}, "output": "Affinity: -6.1", "success": True},
        ]
        
        results = extract_docking_results(tool_calls)
        
        assert len(results) == 2
        assert results[0]["affinity"] == pytest.approx(-8.5, rel=0.01)
        assert results[1]["affinity"] == pytest.approx(-6.1, rel=0.01)
    
    def test_extract_computed_metrics(self):
        """Test the extract_computed_metrics convenience function."""
        tool_calls = [
            {
                "name": "python_exec",
                "arguments": {"code": "logp = MolLogP(mol)"},
                "output": "LogP: 2.5",
                "success": True,
            },
        ]
        
        metrics = extract_computed_metrics(tool_calls)
        
        # Should detect LogP/MolLogP
        assert len(metrics) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
