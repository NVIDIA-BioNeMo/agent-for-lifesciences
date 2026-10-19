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

"""Training's ladder must match production's, on the other message shape."""

from api.context.assembler import (
    LEVEL_COLLAPSED,
    LEVEL_PREVIEW,
    LEVEL_VERBATIM,
    _one_line,
    _preview,
)
from api.context.responses import degrade_responses_input


def _call(index: int) -> dict:
    return {
        "type": "function_call",
        "call_id": f"c{index}",
        "name": "vina_dock",
        "arguments": '{"ligand": "sdf_aaaaaaaa"}',
    }


def _output(index: int, payload_chars: int) -> dict:
    return {
        "type": "function_call_output",
        "call_id": f"c{index}",
        "output": (
            f"Docking complete. Best affinity -9.{index} kcal/mol\n"
            + ("x" * payload_chars)
            + f"\nPose stored as sdf_{index:08d}"
        ),
    }


def _rollout(turns: int, payload_chars: int) -> list[dict]:
    items: list[dict] = [{"role": "system", "content": "You are Aevom."}]
    items.append({"role": "user", "content": "Dock these ligands."})
    for index in range(turns):
        items.append(_call(index))
        items.append(_output(index, payload_chars))
    return items


def test_a_rollout_within_budget_is_untouched():
    items = _rollout(3, 100)
    before = [item.get("output") for item in items]

    report = degrade_responses_input(items, available_tokens=100_000)

    assert report.level == LEVEL_VERBATIM
    assert [item.get("output") for item in items] == before


def test_no_declared_window_means_no_degradation():
    items = _rollout(40, 20_000)
    before = [item.get("output") for item in items]

    report = degrade_responses_input(items, available_tokens=None)

    assert report.degraded_messages == 0
    assert [item.get("output") for item in items] == before


def test_pressure_previews_oldest_outputs_first():
    items = _rollout(20, 4_000)

    report = degrade_responses_input(items, available_tokens=6_000)

    outputs = [item["output"] for item in items if item.get("type") == "function_call_output"]
    assert report.level >= LEVEL_PREVIEW
    assert "truncated" in outputs[0]
    assert outputs[-1].count("x") > len(outputs[0])


def test_degradation_preserves_the_measured_number_and_the_ref():
    items = _rollout(20, 4_000)

    degrade_responses_input(items, available_tokens=6_000)

    first = next(i["output"] for i in items if i.get("type") == "function_call_output")
    assert "-9.0 kcal/mol" in first
    assert "sdf_00000000" in first


def test_no_item_is_ever_removed_so_no_call_loses_its_result():
    items = _rollout(40, 20_000)
    calls = {i["call_id"] for i in items if i.get("type") == "function_call"}

    degrade_responses_input(items, available_tokens=500)

    outputs = {i["call_id"] for i in items if i.get("type") == "function_call_output"}
    assert calls == outputs


def test_an_unshrinkable_prompt_is_reported_rather_than_hidden():
    items = _rollout(40, 20_000)

    report = degrade_responses_input(items, available_tokens=100)

    assert report.level == LEVEL_COLLAPSED
    assert report.still_over_budget is True


def test_pydantic_style_items_are_handled_alongside_dicts():
    class Item:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    items = [
        Item(role="user", content="dock it"),
        Item(type="function_call", call_id="c0", name="vina_dock", arguments="{}"),
        Item(type="function_call_output", call_id="c0", output="y" * 20_000),
    ]

    degrade_responses_input(items, available_tokens=200)

    assert len(items[2].output) < 20_000


def test_training_and_production_use_the_same_transforms():
    """Parity is not a claim about intent; it is these two functions."""
    text = "Best affinity -9.5 kcal/mol\n" + "z" * 5_000 + "\nsdf_12345678"

    assert "-9.5 kcal/mol" in _preview(text)
    assert "sdf_12345678" in _preview(text)
    assert "-9.5 kcal/mol" in _one_line(text)
    assert "sdf_12345678" in _one_line(text)
    assert len(_one_line(text)) < len(_preview(text)) < len(text)
