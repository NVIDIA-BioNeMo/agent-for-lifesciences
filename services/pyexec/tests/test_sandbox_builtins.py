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

"""Tests that all expected builtins are accessible in the pyexec sandbox."""

import pytest

from app.sandbox import execute_code


class TestTier1Builtins:
    """Safe, commonly used in scientific code."""

    def test_next(self):
        result = execute_code("result = next(iter([10, 20, 30]))")
        assert result["success"] is True
        assert result["result"] == 10

    def test_iter(self):
        result = execute_code(
            "it = iter([1, 2, 3])\n"
            "result = [next(it), next(it)]"
        )
        assert result["success"] is True
        assert result["result"] == [1, 2]

    def test_frozenset(self):
        result = execute_code("result = len(frozenset([1, 2, 2, 3]))")
        assert result["success"] is True
        assert result["result"] == 3

    def test_bin(self):
        result = execute_code("result = bin(42)")
        assert result["success"] is True
        assert result["result"] == "0b101010"

    def test_ascii(self):
        result = execute_code("result = ascii('hello')")
        assert result["success"] is True
        assert result["result"] == "'hello'"


class TestTier2Builtins:
    """Safe, useful for OOP patterns in analysis code."""

    def test_object(self):
        result = execute_code(
            "class Foo(object):\n"
            "    pass\n"
            "result = isinstance(Foo(), object)"
        )
        assert result["success"] is True
        assert result["result"] is True

    def test_super(self):
        result = execute_code(
            "class Base(object):\n"
            "    def value(self):\n"
            "        return 42\n"
            "class Child(Base):\n"
            "    def value(self):\n"
            "        return super().value() + 1\n"
            "result = Child().value()"
        )
        assert result["success"] is True
        assert result["result"] == 43

    def test_classmethod(self):
        result = execute_code(
            "class C(object):\n"
            "    @classmethod\n"
            "    def make(cls):\n"
            "        return 99\n"
            "result = C.make()"
        )
        assert result["success"] is True
        assert result["result"] == 99

    def test_staticmethod(self):
        result = execute_code(
            "class C(object):\n"
            "    @staticmethod\n"
            "    def add(a, b):\n"
            "        return a + b\n"
            "result = C.add(3, 4)"
        )
        assert result["success"] is True
        assert result["result"] == 7

    def test_property(self):
        result = execute_code(
            "class C(object):\n"
            "    @property\n"
            "    def val(self):\n"
            "        return 7\n"
            "result = C().val"
        )
        assert result["success"] is True
        assert result["result"] == 7

    def test_bytearray(self):
        result = execute_code("result = list(bytearray(b'abc'))")
        assert result["success"] is True
        assert result["result"] == [97, 98, 99]

    def test_memoryview(self):
        result = execute_code("result = len(memoryview(b'hello'))")
        assert result["success"] is True
        assert result["result"] == 5


class TestTier3Skipped:
    """Async builtins should NOT be available in the sync sandbox."""

    def test_aiter_not_available(self):
        result = execute_code("result = aiter")
        assert result["success"] is False

    def test_anext_not_available(self):
        result = execute_code("result = anext")
        assert result["success"] is False
