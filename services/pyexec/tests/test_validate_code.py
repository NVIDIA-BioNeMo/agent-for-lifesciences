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

"""Tests for validate_code's token-aware dangerous-pattern pre-check.

The check exists to reject obvious escape attempts with a friendly message
before RestrictedPython (the real sandbox) would fail them anyway. The old
implementation did a case-folded *substring* scan, which produced false
positives that blocked legitimate cheminformatics code -- most visibly
``file(`` matching identifiers like ``structure_file(`` and ``compile(``
matching ``re.compile(``. These tests pin both halves: the false positives are
gone, and the genuinely dangerous bare calls / module names are still blocked.
"""

import pytest

from app.sandbox import validate_code


def ok(code: str) -> bool:
    return validate_code(code)[0]


class TestNoLongerFalsePositive:
    """Legitimate code the substring scan used to reject."""

    @pytest.mark.parametrize("code", [
        "import re\np = re.compile('c1ccccc1')",          # re.compile, not builtin
        "def to_file(x):\n    return x\nto_file(1)",       # *_file( identifier
        "structure_file(receptor)",                        # identifier ending in file
        'note = "remember to open( the pocket"',           # inside a string
        "# open( a file here\nresult = 1",                 # inside a comment
        "import numpy as np\narr = np.load  # attribute",  # np.load is fine
        "from rdkit import Chem\n"
        "mol = Chem.MolFromSmiles('CCO')\n"
        "result = mol.GetNumAtoms()",                      # ordinary RDKit code
        "os.path.join('a', 'b')",                          # os.path is not os.system
    ])
    def test_allowed(self, code):
        assert ok(code) is True, f"should be allowed: {code!r}"


class TestStillBlocked:
    """The dangerous constructs the check is actually for."""

    @pytest.mark.parametrize("code", [
        "open('/etc/passwd')",
        "exec('1')",
        "eval('1')",
        "compile('1', 'x', 'eval')",
        "__import__('os')",
        "import socket",
        "socket.socket()",
        "import pickle\npickle.loads(b'')",
        "os.system('ls')",
        "os.popen('ls')",
        "globals()",
        "locals()",
        "vars(obj)",
        "setattr(obj, 'a', 1)",
        "delattr(obj, 'a')",
    ])
    def test_blocked(self, code):
        assert ok(code) is False, f"should be blocked: {code!r}"


class TestAttributeAccessIsNotAModuleUse:
    def test_attribute_named_like_a_blocked_module_is_allowed(self):
        # `.socket` as an attribute is not the socket module.
        assert ok("conn.socket") is True

    def test_syntax_error_is_still_reported(self):
        valid, msg = validate_code("def f(:\n    pass")
        assert valid is False
        assert "Syntax error" in msg
