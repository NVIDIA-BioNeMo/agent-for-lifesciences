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

"""Augmented assignment on a subscript is rejected -- with a teaching message.

``counts[k] += 1`` is the single most common thing models write that the
sandbox rejects (RestrictedPython forbids augmented assignment on items and
slices). A bare rejection just gets retried the same way, so the error names
the working alternatives (Counter / dict.get). Plain item assignment is fine.
"""

from app.sandbox import execute_code


class TestAugmentedSubscriptAssignment:
    def test_dict_item_augmented_assignment_is_rejected_with_guidance(self):
        result = execute_code("d = {'a': 0}\nd['a'] += 1\nresult = d")
        assert result["success"] is False
        err = result["error"]
        assert "Counter" in err
        assert "get(" in err

    def test_list_item_augmented_assignment_is_rejected(self):
        result = execute_code("l = [0]\nl[0] += 1\nresult = l")
        assert result["success"] is False

    def test_plain_dict_item_assignment_is_allowed(self):
        result = execute_code("d = {}\nd['a'] = 1\nresult = d")
        assert result["success"] is True
        assert result["result"] == {"a": 1}

    def test_counter_is_the_working_pattern(self):
        result = execute_code(
            "from collections import Counter\n"
            "c = Counter()\n"
            "for x in ['a', 'a', 'b']:\n"
            "    c[x] += 1\n"  # Counter.__iadd__? no -- c[x] += 1 is still subscript
            "result = dict(c)"
        )
        # Even on a Counter, c[x] += 1 is a subscript aug-assign and is rejected;
        # the supported pattern is Counter(iterable).
        assert result["success"] is False

    def test_counter_from_iterable_is_allowed(self):
        result = execute_code(
            "from collections import Counter\n"
            "result = dict(Counter(['a', 'a', 'b']))"
        )
        assert result["success"] is True
        assert result["result"] == {"a": 2, "b": 1}
