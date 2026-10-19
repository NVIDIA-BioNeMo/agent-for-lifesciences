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

"""Tests for the analysis libraries the sandbox exposes.

Covers scikit-learn, OpenBabel and biotite, and the RDKit corners the
medicinal-chemistry skills reach for -- scaffolds, MCS, structural alerts,
pharmacophore features. The RDKit cases pass by virtue of the whitelist matching
on module prefixes rather than exact names, which is easy to break by tightening
the matcher, so they are pinned here.

Every library here has to be driven from strings, because the sandbox has no
filesystem. That is the part most likely to regress silently: an import smoke
test passes while the only usable entry point is gone. So each library is
exercised through the in-memory API the skills tell the model to use.

Three submodules are the exception -- sklearn.datasets, biotite.database and
biotite.application -- because they download over HTTP, read arbitrary paths or
launch external programs, the same surface `open(` and `socket.` are blocked for.
"""

import pytest

from app.sandbox import execute_code

pytest.importorskip("sklearn", reason="scikit-learn lives in the RDKit base image")

PHENETHYLAMINE = "c1ccccc1CCN"

# Two residues of a real backbone. Enough for a parser to find a chain, a
# residue and named atoms; small enough to read.
DIPEPTIDE_PDB = """\
ATOM      1  N   ALA A   1      11.104   6.134  -6.504  1.00  0.00           N
ATOM      2  CA  ALA A   1      11.639   6.071  -5.147  1.00  0.00           C
ATOM      3  C   ALA A   1      13.150   6.310  -5.169  1.00  0.00           C
ATOM      4  O   ALA A   1      13.831   5.933  -6.124  1.00  0.00           O
ATOM      5  CB  ALA A   1      10.933   7.093  -4.281  1.00  0.00           C
ATOM      6  N   GLY A   2      13.670   6.949  -4.128  1.00  0.00           N
ATOM      7  CA  GLY A   2      15.091   7.234  -4.020  1.00  0.00           C
ATOM      8  C   GLY A   2      15.865   6.017  -3.545  1.00  0.00           C
ATOM      9  O   GLY A   2      15.310   5.121  -2.910  1.00  0.00           O
END
"""


class TestScikitLearnIsUsable:
    def test_importing_a_submodule_is_allowed(self):
        result = execute_code("from sklearn.cluster import KMeans\nresult = True")
        assert result["success"] is True, result["error"]
        assert result["result"] is True

    def test_importing_the_package_is_allowed(self):
        # Note the sandbox forbids dunder attributes, so this cannot check
        # __version__ -- reaching a real submodule is the meaningful assertion.
        result = execute_code("import sklearn.cluster\nresult = True")
        assert result["success"] is True, result["error"]
        assert result["result"] is True

    def test_clustering_actually_runs(self):
        # Clustering points in space is the reason this is here -- grouping
        # hydrophobic pocket atoms into hotspots.
        result = execute_code(
            "import numpy as np\n"
            "from sklearn.cluster import KMeans\n"
            "points = np.array([[0.0, 0, 0], [0.2, 0, 0], [9.0, 9, 9], [9.2, 9, 9]])\n"
            "labels = KMeans(n_clusters=2, n_init=10, random_state=0).fit_predict(points)\n"
            "result = int(labels[0] == labels[1]) + int(labels[2] == labels[3])"
        )
        assert result["success"] is True, result["error"]
        assert result["result"] == 2

    def test_regression_over_descriptors_runs(self):
        result = execute_code(
            "import numpy as np\n"
            "from sklearn.linear_model import LinearRegression\n"
            "x = np.array([[1.0], [2.0], [3.0], [4.0]])\n"
            "model = LinearRegression().fit(x, np.array([2.0, 4.0, 6.0, 8.0]))\n"
            "result = round(float(model.coef_[0]), 6)"
        )
        assert result["success"] is True, result["error"]
        assert result["result"] == 2.0


class TestScikitLearnDatasetsIsNotReachable:
    def test_importing_datasets_is_blocked(self):
        result = execute_code("from sklearn.datasets import fetch_openml\nresult = 1")
        assert result["success"] is False
        assert "not allowed" in result["error"]

    def test_a_deeper_module_under_datasets_is_blocked(self):
        # The block has to cover children too, or the prefix match on "sklearn"
        # would let this straight through.
        result = execute_code("import sklearn.datasets._base\nresult = 1")
        assert result["success"] is False
        assert "not allowed" in result["error"]

    def test_importing_externals_is_blocked(self):
        result = execute_code("import sklearn.externals\nresult = 1")
        assert result["success"] is False
        assert "not allowed" in result["error"]

    def test_the_error_explains_what_is_still_allowed(self):
        # A dead end the model cannot recover from wastes a turn; say that the
        # rest of sklearn is fine and that data comes from the aevom client.
        result = execute_code("from sklearn.datasets import load_iris\nresult = 1")
        assert "aevom" in result["error"]
        assert "sklearn" in result["error"]

    def test_importing_datasets_from_the_package_is_blocked(self):
        # `from sklearn import datasets` imports "sklearn" and takes the submodule
        # from the fromlist, so the module-name check alone never sees it.
        result = execute_code("from sklearn import datasets\nresult = 1")
        assert result["success"] is False
        assert "not allowed" in result["error"]

    def test_traversing_to_datasets_through_the_package_is_blocked(self):
        # sklearn resolves submodules lazily on attribute access, so this reaches
        # fetch_openml() without an import statement. RestrictedPython does not
        # guard IMPORT_FROM, so both halves have to be closed separately.
        result = execute_code("import sklearn\nresult = sklearn.datasets.load_iris")
        assert result["success"] is False
        assert "not allowed" in result["error"]

    def test_an_unrelated_datasets_attribute_still_works(self):
        # The block is scoped to the sklearn package; shadowing every attribute
        # named `datasets` would break ordinary code for no security gain.
        result = execute_code(
            "class Holder:\n    datasets = 42\nresult = Holder().datasets"
        )
        assert result["success"] is True, result["error"]
        assert result["result"] == 42


class TestRdkitCornersStayReachable:
    def test_murcko_scaffold(self):
        result = execute_code(
            "from rdkit import Chem\n"
            "from rdkit.Chem.Scaffolds import MurckoScaffold\n"
            f'mol = Chem.MolFromSmiles("{PHENETHYLAMINE}")\n'
            "result = Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(mol))"
        )
        assert result["success"] is True, result["error"]
        assert result["result"] == "c1ccccc1"

    def test_maximum_common_substructure(self):
        result = execute_code(
            "from rdkit import Chem\n"
            "from rdkit.Chem import rdFMCS\n"
            f'a = Chem.MolFromSmiles("{PHENETHYLAMINE}")\n'
            'b = Chem.MolFromSmiles("c1ccccc1CCO")\n'
            "result = rdFMCS.FindMCS([a, b]).numAtoms"
        )
        assert result["success"] is True, result["error"]
        assert result["result"] == 8

    def test_structural_alert_catalog(self):
        result = execute_code(
            "from rdkit import Chem\n"
            "from rdkit.Chem import FilterCatalog\n"
            "params = FilterCatalog.FilterCatalogParams()\n"
            "params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS)\n"
            "catalog = FilterCatalog.FilterCatalog(params)\n"
            f'result = catalog.HasMatch(Chem.MolFromSmiles("{PHENETHYLAMINE}"))'
        )
        assert result["success"] is True, result["error"]
        assert result["result"] is False

    def test_pharmacophore_feature_factory(self):
        # BuildFeatureFactory reads a definition file from the RDKit data
        # directory. That happens in C++, so the sandbox's `open(` guard does not
        # block it -- but it does need RDConfig to locate the file.
        result = execute_code(
            "from rdkit import Chem, RDConfig\n"
            "from rdkit.Chem import ChemicalFeatures\n"
            'factory = ChemicalFeatures.BuildFeatureFactory(\n'
            '    RDConfig.RDDataDir + "/BaseFeatures.fdef"\n'
            ")\n"
            f'mol = Chem.MolFromSmiles("{PHENETHYLAMINE}")\n'
            "result = sorted({f.GetFamily() for f in factory.GetFeaturesForMol(mol)})"
        )
        assert result["success"] is True, result["error"]
        assert "Donor" in result["result"]
        assert "Aromatic" in result["result"]

    def test_brics_fragments(self):
        result = execute_code(
            "from rdkit import Chem\n"
            "from rdkit.Chem import BRICS\n"
            f'mol = Chem.MolFromSmiles("{PHENETHYLAMINE}")\n'
            "result = len(BRICS.BRICSDecompose(mol))"
        )
        assert result["success"] is True, result["error"]
        assert result["result"] > 1


class TestOpenBabelIsUsable:
    """Format conversion and protonation, the two jobs RDKit does not cover.

    OpenBabel is here for parity with the environment SMDD gives its reference
    agent, and because PDB-to-PDBQT is the step in front of anything Vina-shaped.
    """

    def test_importing_openbabel_is_allowed(self):
        result = execute_code("from openbabel import openbabel\nresult = True")
        assert result["success"] is True, result["error"]
        assert result["result"] is True

    def test_pybel_is_reachable_as_a_submodule(self):
        # The friendlier wrapper, and the one most examples online use. It is
        # admitted by the prefix match on "openbabel" rather than by name.
        result = execute_code("from openbabel import pybel\nresult = True")
        assert result["success"] is True, result["error"]
        assert result["result"] is True

    def test_conversion_runs_entirely_through_strings(self):
        # ReadString/WriteString is the whole story in the sandbox: the
        # path-based calls have nothing to point at.
        result = execute_code(
            "from openbabel import openbabel as ob\n"
            "conv = ob.OBConversion()\n"
            'conv.SetInAndOutFormats("smi", "can")\n'
            "mol = ob.OBMol()\n"
            f'conv.ReadString(mol, "{PHENETHYLAMINE}")\n'
            "result = conv.WriteString(mol).strip()"
        )
        assert result["success"] is True, result["error"]
        assert "N" in result["result"]

    def test_a_pdb_string_converts_to_pdbqt(self):
        # The receptor-preparation path: parse PDB, add hydrogens for pH 7.4,
        # assign Gasteiger charges, write the format docking tools expect.
        result = execute_code(
            "from openbabel import openbabel as ob\n"
            "conv = ob.OBConversion()\n"
            'conv.SetInAndOutFormats("pdb", "pdbqt")\n'
            "mol = ob.OBMol()\n"
            "conv.ReadString(mol, pdb_text)\n"
            "mol.AddHydrogens(False, True, 7.4)\n"
            "result = conv.WriteString(mol)",
            context={"pdb_text": DIPEPTIDE_PDB},
        )
        assert result["success"] is True, result["error"]
        assert "ATOM" in result["result"]

    def test_adding_hydrogens_changes_the_atom_count(self):
        # Mutating a molecule in place is allowed -- the write guard stops
        # `obj.attr = x`, not methods that mutate internally.
        result = execute_code(
            "from openbabel import openbabel as ob\n"
            "conv = ob.OBConversion()\n"
            'conv.SetInFormat("smi")\n'
            "mol = ob.OBMol()\n"
            f'conv.ReadString(mol, "{PHENETHYLAMINE}")\n'
            "before = mol.NumAtoms()\n"
            "mol.AddHydrogens()\n"
            "result = mol.NumAtoms() - before"
        )
        assert result["success"] is True, result["error"]
        assert result["result"] == 11


class TestBiotiteIsUsable:
    """Structure parsing over AtomArrays, read from a string like everything else."""

    def test_importing_a_submodule_is_allowed(self):
        result = execute_code("import biotite.structure\nresult = True")
        assert result["success"] is True, result["error"]
        assert result["result"] is True

    def test_a_pdb_string_parses_through_stringio(self):
        result = execute_code(
            "import io\n"
            "import biotite.structure.io.pdb as pdb\n"
            "structure = pdb.PDBFile.read(io.StringIO(pdb_text))\n"
            "atoms = structure.get_structure(model=1)\n"
            "result = atoms.array_length()",
            context={"pdb_text": DIPEPTIDE_PDB},
        )
        assert result["success"] is True, result["error"]
        assert result["result"] == 9

    def test_filtering_and_measuring_work_on_the_parsed_array(self):
        # Boolean masks over annotation arrays are biotite's whole idiom, and
        # they read rather than write, so the sandbox leaves them alone.
        result = execute_code(
            "import io\n"
            "import biotite.structure as struc\n"
            "import biotite.structure.io.pdb as pdb\n"
            "structure = pdb.PDBFile.read(io.StringIO(pdb_text))\n"
            "atoms = structure.get_structure(model=1)\n"
            'alphas = atoms[atoms.atom_name == "CA"]\n'
            "result = round(float(struc.distance(alphas[0], alphas[1])), 2)",
            context={"pdb_text": DIPEPTIDE_PDB},
        )
        assert result["success"] is True, result["error"]
        assert 3.0 < result["result"] < 4.5

    def test_coordinates_come_back_as_a_numpy_array(self):
        # The handoff to numpy and sklearn, which is why biotite is worth having
        # alongside BioPython.
        result = execute_code(
            "import io\n"
            "import numpy as np\n"
            "import biotite.structure.io.pdb as pdb\n"
            "structure = pdb.PDBFile.read(io.StringIO(pdb_text))\n"
            "atoms = structure.get_structure(model=1)\n"
            "result = [round(float(v), 2) for v in np.mean(atoms.coord, axis=0)]",
            context={"pdb_text": DIPEPTIDE_PDB},
        )
        assert result["success"] is True, result["error"]
        assert len(result["result"]) == 3


class TestBiotiteNetworkAndSubprocessAreNotReachable:
    def test_importing_the_database_module_is_blocked(self):
        result = execute_code("import biotite.database.rcsb\nresult = 1")
        assert result["success"] is False
        assert "not allowed" in result["error"]

    def test_importing_database_from_the_package_is_blocked(self):
        result = execute_code("from biotite import database\nresult = 1")
        assert result["success"] is False
        assert "not allowed" in result["error"]

    def test_traversing_to_database_through_the_package_is_blocked(self):
        result = execute_code("import biotite\nresult = biotite.database.rcsb")
        assert result["success"] is False
        assert "not allowed" in result["error"]

    def test_importing_the_application_module_is_blocked(self):
        # Wrappers that shell out to DSSP, MUSCLE and BLAST.
        result = execute_code("import biotite.application.dssp\nresult = 1")
        assert result["success"] is False
        assert "not allowed" in result["error"]

    def test_the_error_says_where_structures_come_from_instead(self):
        result = execute_code("from biotite.database import rcsb\nresult = 1")
        assert "aevom" in result["error"]
