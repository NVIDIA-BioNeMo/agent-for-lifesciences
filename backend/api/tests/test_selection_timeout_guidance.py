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

"""The viewer-timeout message and the structure-selection skill.

A structure_query timeout only happens with no viewer attached -- not the
production or eval configuration, both of which answer selections through
Molstar. So the message stays minimal: don't hammer a viewer that isn't
answering, and visualize the structure first. The reusable guidance for
selection analysis lives in the 'structure-selection-analysis' skill, which
these tests confirm is discoverable.
"""

from api.selections.format import format_selection_timeout


class TestTimeoutMessage:
    def test_tells_the_model_to_visualize_first(self):
        assert "visualize_pdb" in format_selection_timeout()

    def test_discourages_pointless_retries(self):
        # The viewer-less baseline fired the same query several times in a row.
        assert "retry" in format_selection_timeout().lower()

    def test_stays_short_no_inlined_code_block(self):
        # The inlined python fallback was trimmed: it only ever fired in a
        # viewer-less stack, which production/eval never are.
        msg = format_selection_timeout()
        assert "aevom.load_pdb_structure" not in msg
        assert "cdist" not in msg


class TestSkillRegistered:
    def test_structure_selection_skill_is_discoverable(self):
        from api.skills.registry import SkillRegistry

        reg = SkillRegistry()
        reg.load_builtin_skills()
        assert reg.get("structure-selection-analysis") is not None
