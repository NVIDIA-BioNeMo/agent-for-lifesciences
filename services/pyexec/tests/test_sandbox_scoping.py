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

"""Top-level names must be visible from nested scopes.

Executing the module body with separate ``globals`` and ``locals`` mappings
gives it class-body semantics: comprehensions, generator expressions, lambdas
and ``def`` bodies open a scope that skips the enclosing locals, so ordinary
Python raises ``NameError`` for a name assigned two lines earlier. These are
the exact shapes an agent writes when mining SMARTS patterns over a molecule
set, and every one of them failed before the sandbox switched to a single
namespace.
"""

from app.sandbox import execute_code


class TestComprehensionsSeeTopLevelNames:
    def test_list_comprehension_reads_a_top_level_binding(self):
        result = execute_code(
            "threshold = 2\n"
            "result = [n for n in [1, 2, 3, 4] if n > threshold]"
        )
        assert result["error"] is None
        assert result["result"] == [3, 4]

    def test_comprehension_calls_a_top_level_function(self):
        result = execute_code(
            "def double(n):\n"
            "    return n * 2\n"
            "result = [double(n) for n in [1, 2, 3]]"
        )
        assert result["error"] is None
        assert result["result"] == [2, 4, 6]

    def test_nested_comprehension_reads_the_outer_binding(self):
        result = execute_code(
            "rows = [[1, 2], [3, 4]]\n"
            "offset = 10\n"
            "result = [value + offset for row in rows for value in row]"
        )
        assert result["error"] is None
        assert result["result"] == [11, 12, 13, 14]

    def test_dict_and_set_comprehensions(self):
        result = execute_code(
            "keys = ['a', 'b']\n"
            "bump = 1\n"
            "mapping = {k: i + bump for i, k in enumerate(keys)}\n"
            "uniques = {i + bump for i in [0, 0, 1]}\n"
            "result = [mapping, sorted(uniques)]"
        )
        assert result["error"] is None
        assert result["result"] == [{"a": 1, "b": 2}, [1, 2]]

    def test_generator_expression_inside_sum(self):
        result = execute_code(
            "weights = [1, 2, 3]\n"
            "scale = 10\n"
            "result = sum(w * scale for w in weights)"
        )
        assert result["error"] is None
        assert result["result"] == 60


class TestFunctionsSeeTopLevelNames:
    def test_function_body_reads_a_module_global(self):
        result = execute_code(
            "factor = 3\n"
            "def scale(n):\n"
            "    return n * factor\n"
            "result = scale(5)"
        )
        assert result["error"] is None
        assert result["result"] == 15

    def test_lambda_reads_a_module_global(self):
        result = execute_code(
            "offset = 4\n"
            "shift = lambda n: n + offset\n"
            "result = shift(1)"
        )
        assert result["error"] is None
        assert result["result"] == 5

    def test_function_calls_another_function_defined_above_it(self):
        result = execute_code(
            "def inner(n):\n"
            "    return n + 1\n"
            "def outer(n):\n"
            "    return inner(n) * 2\n"
            "result = outer(3)"
        )
        assert result["error"] is None
        assert result["result"] == 8

    def test_class_method_reads_a_module_global(self):
        result = execute_code(
            "base = 100\n"
            "class Counter:\n"
            "    def value(self, n):\n"
            "        return base + n\n"
            "result = Counter().value(1)"
        )
        assert result["error"] is None
        assert result["result"] == 101


class TestSubstructureSearchPattern:
    """The failing shape observed on SMDD-Bench type 1, end to end."""

    def test_smarts_match_over_a_comprehension(self):
        result = execute_code(
            "from rdkit import Chem\n"
            "smiles = ['c1ccccc1', 'CCO', 'c1ccc(O)cc1']\n"
            "mols = [Chem.MolFromSmiles(s) for s in smiles]\n"
            "pat = Chem.MolFromSmarts('c1ccccc1')\n"
            "result = sum(1 for m in mols if m.HasSubstructMatch(pat))"
        )
        assert result["error"] is None
        assert result["result"] == 2
