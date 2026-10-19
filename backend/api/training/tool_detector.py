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

"""Tool usage detection and analysis for training.

This module provides utilities for analyzing tool usage patterns
in trajectories for reward computation and debugging.
"""

import re
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class PythonCodeAnalysis:
    """Analysis results for Python code execution."""
    
    # Code metrics
    line_count: int = 0
    has_imports: bool = False
    imports: list[str] = field(default_factory=list)
    
    # RDKit specific
    uses_rdkit: bool = False
    rdkit_imports: list[str] = field(default_factory=list)
    rdkit_calls: list[str] = field(default_factory=list)
    
    # Molecular operations
    creates_molecule: bool = False
    computes_descriptors: bool = False
    descriptor_names: list[str] = field(default_factory=list)
    
    # Output handling
    prints_results: bool = False
    returns_value: bool = False


@dataclass
class DockingAnalysis:
    """Analysis results for docking operations."""
    
    receptor_ref: Optional[str] = None
    ligand_ref: Optional[str] = None
    success: bool = False
    affinity: Optional[float] = None
    error: Optional[str] = None


@dataclass
class TrajectoryAnalysis:
    """Complete analysis of a training trajectory."""
    
    # Summary
    total_steps: int = 0
    total_tool_calls: int = 0
    
    # Tool counts
    tool_call_counts: dict[str, int] = field(default_factory=dict)
    
    # Success rates
    success_count: int = 0
    error_count: int = 0
    
    # Specific analyses
    python_analyses: list[PythonCodeAnalysis] = field(default_factory=list)
    docking_analyses: list[DockingAnalysis] = field(default_factory=list)
    
    # Derived metrics
    used_docking: bool = False
    used_python: bool = False
    used_rdkit: bool = False
    computed_metrics: bool = False


class TrajectoryAnalyzer:
    """Analyzes training trajectories for reward computation.
    
    This analyzer extracts detailed information about tool usage,
    including RDKit operations, docking results, and metric computation.
    """
    
    # RDKit import patterns
    RDKIT_IMPORT_PATTERNS = [
        r"from\s+rdkit\s+import\s+(\w+)",
        r"from\s+rdkit\.(\w+)\s+import",
        r"import\s+rdkit\.(\w+)",
        r"from\s+rdkit\.Chem\s+import\s+(\w+)",
    ]
    
    # RDKit function call patterns
    RDKIT_CALL_PATTERNS = [
        r"Chem\.(\w+)\(",
        r"AllChem\.(\w+)\(",
        r"Descriptors\.(\w+)\(",
        r"Draw\.(\w+)\(",
        r"DataStructs\.(\w+)\(",
    ]
    
    # Descriptor patterns
    DESCRIPTOR_PATTERNS = [
        r"MolLogP|LogP|logP",
        r"MolWt|MolecularWeight",
        r"TPSA",
        r"NumHDonors|HBD",
        r"NumHAcceptors|HBA",
        r"NumRotatableBonds",
        r"RingCount|NumRings",
        r"FractionCSP3",
        r"QED|qed",
    ]
    
    def analyze_trajectory(self, tool_calls: list[dict]) -> TrajectoryAnalysis:
        """Analyze a complete trajectory.
        
        Args:
            tool_calls: List of tool call records.
            
        Returns:
            TrajectoryAnalysis with detailed results.
        """
        analysis = TrajectoryAnalysis()
        analysis.total_tool_calls = len(tool_calls)
        
        for call in tool_calls:
            name = call.get("name", "unknown")
            success = call.get("success", True)
            
            # Update counts
            analysis.tool_call_counts[name] = (
                analysis.tool_call_counts.get(name, 0) + 1
            )
            
            if success:
                analysis.success_count += 1
            else:
                analysis.error_count += 1
            
            # Analyze specific tool types
            if name in ("python_exec", "python_execute", "execute_python"):
                py_analysis = self._analyze_python_code(call)
                analysis.python_analyses.append(py_analysis)
                analysis.used_python = True
                
                if py_analysis.uses_rdkit:
                    analysis.used_rdkit = True
                if py_analysis.computes_descriptors:
                    analysis.computed_metrics = True
            
            elif name in ("vina_dock", "dock_ligand", "docking"):
                dock_analysis = self._analyze_docking(call)
                analysis.docking_analyses.append(dock_analysis)
                analysis.used_docking = True
        
        return analysis
    
    def _analyze_python_code(self, call: dict) -> PythonCodeAnalysis:
        """Analyze a Python execution call.
        
        Args:
            call: Tool call record.
            
        Returns:
            PythonCodeAnalysis with results.
        """
        analysis = PythonCodeAnalysis()
        
        args = call.get("arguments", {})
        code = args.get("code", "")
        
        # Basic metrics
        lines = code.strip().split("\n")
        analysis.line_count = len(lines)
        
        # Check for imports
        import_pattern = r"^(?:from|import)\s+"
        for line in lines:
            if re.match(import_pattern, line.strip()):
                analysis.has_imports = True
                analysis.imports.append(line.strip())
        
        # Check for RDKit imports
        for pattern in self.RDKIT_IMPORT_PATTERNS:
            matches = re.findall(pattern, code)
            if matches:
                analysis.uses_rdkit = True
                analysis.rdkit_imports.extend(matches)
        
        # Check for RDKit function calls
        for pattern in self.RDKIT_CALL_PATTERNS:
            matches = re.findall(pattern, code)
            if matches:
                analysis.uses_rdkit = True
                analysis.rdkit_calls.extend(matches)
        
        # Check for molecule creation
        mol_creation_patterns = [
            r"MolFromSmiles",
            r"MolFromMol2",
            r"MolFromPDBBlock",
            r"MolFromMolBlock",
            r"MolFromSDF",
        ]
        for pattern in mol_creation_patterns:
            if re.search(pattern, code):
                analysis.creates_molecule = True
                break
        
        # Check for descriptor computation
        for pattern in self.DESCRIPTOR_PATTERNS:
            match = re.search(pattern, code, re.IGNORECASE)
            if match:
                analysis.computes_descriptors = True
                analysis.descriptor_names.append(match.group(0))
        
        # Check for output
        if re.search(r"\bprint\s*\(", code):
            analysis.prints_results = True
        if re.search(r"\breturn\s+", code):
            analysis.returns_value = True
        
        return analysis
    
    def _analyze_docking(self, call: dict) -> DockingAnalysis:
        """Analyze a docking call.
        
        Args:
            call: Tool call record.
            
        Returns:
            DockingAnalysis with results.
        """
        analysis = DockingAnalysis()
        
        args = call.get("arguments", {})
        output = call.get("output", "")
        success = call.get("success", True)
        
        analysis.receptor_ref = args.get("receptor_ref") or args.get("receptor")
        analysis.ligand_ref = args.get("ligand_ref") or args.get("ligand")
        analysis.success = success
        
        if not success:
            analysis.error = output
        else:
            # Try to extract affinity from output
            affinity_patterns = [
                r"affinity[:\s]+(-?\d+\.?\d*)",
                r"score[:\s]+(-?\d+\.?\d*)",
                r"(-?\d+\.?\d*)\s*kcal",
            ]
            for pattern in affinity_patterns:
                match = re.search(pattern, output, re.IGNORECASE)
                if match:
                    try:
                        analysis.affinity = float(match.group(1))
                        break
                    except ValueError:
                        pass
        
        return analysis
    
    def get_summary_metrics(self, analysis: TrajectoryAnalysis) -> dict:
        """Get summary metrics from trajectory analysis.
        
        Args:
            analysis: TrajectoryAnalysis to summarize.
            
        Returns:
            Dictionary of summary metrics.
        """
        return {
            "total_tool_calls": analysis.total_tool_calls,
            "success_rate": (
                analysis.success_count / analysis.total_tool_calls
                if analysis.total_tool_calls > 0 else 0.0
            ),
            "used_docking": analysis.used_docking,
            "docking_count": len(analysis.docking_analyses),
            "docking_success_rate": (
                sum(1 for d in analysis.docking_analyses if d.success) /
                len(analysis.docking_analyses)
                if analysis.docking_analyses else 0.0
            ),
            "used_python": analysis.used_python,
            "python_count": len(analysis.python_analyses),
            "used_rdkit": analysis.used_rdkit,
            "computed_metrics": analysis.computed_metrics,
            "unique_tools": len(analysis.tool_call_counts),
            "tool_counts": dict(analysis.tool_call_counts),
        }


def analyze_tool_calls(tool_calls: list[dict]) -> dict:
    """Convenience function to analyze tool calls.
    
    Args:
        tool_calls: List of tool call records.
        
    Returns:
        Dictionary with analysis results.
    """
    analyzer = TrajectoryAnalyzer()
    analysis = analyzer.analyze_trajectory(tool_calls)
    return analyzer.get_summary_metrics(analysis)


def extract_docking_results(tool_calls: list[dict]) -> list[dict]:
    """Extract docking results from tool calls.
    
    Args:
        tool_calls: List of tool call records.
        
    Returns:
        List of docking result dictionaries.
    """
    results = []
    analyzer = TrajectoryAnalyzer()
    
    for call in tool_calls:
        if call.get("name") in ("vina_dock", "dock_ligand", "docking"):
            analysis = analyzer._analyze_docking(call)
            results.append({
                "receptor": analysis.receptor_ref,
                "ligand": analysis.ligand_ref,
                "success": analysis.success,
                "affinity": analysis.affinity,
                "error": analysis.error,
            })
    
    return results


def extract_computed_metrics(tool_calls: list[dict]) -> dict[str, list]:
    """Extract computed metrics from Python execution calls.
    
    Args:
        tool_calls: List of tool call records.
        
    Returns:
        Dictionary mapping metric names to detected values.
    """
    metrics: dict[str, list] = {}
    analyzer = TrajectoryAnalyzer()
    
    for call in tool_calls:
        if call.get("name") in ("python_exec", "python_execute"):
            analysis = analyzer._analyze_python_code(call)
            for descriptor in analysis.descriptor_names:
                if descriptor not in metrics:
                    metrics[descriptor] = []
                # Try to extract value from output
                output = call.get("output", "")
                # Simple pattern for numeric output after descriptor name
                pattern = rf"{descriptor}[:\s=]+(-?\d+\.?\d*)"
                match = re.search(pattern, output, re.IGNORECASE)
                if match:
                    try:
                        metrics[descriptor].append(float(match.group(1)))
                    except ValueError:
                        pass
    
    return metrics
