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

"""Tests for per-session tool call budgets.

The point of a budget is not the refusal -- it is that the agent knows what it has
left while it can still act on the knowledge. So these tests care about three
things in roughly that order: that remaining spend is visible in every response and
in the prompt envelope, that exhaustion refuses without running the tool, and that
a budget nobody configured stays completely invisible.
"""

import json
import sys
import types

import pytest

from api.storage import InvalidReferenceError
from api.storage.memory import InMemoryDataArtifactRepository
from api.storage.service import DataStorageService
from api.tools import budget
from api.tools.base import BaseTool, ContentType, ToolResult, ToolType
from api.tools.registry import ToolRegistry


class CountingTool(BaseTool):
    """Records how many times it actually ran."""

    def __init__(
        self,
        name: str = "expensive",
        raises: Exception | None = None,
        returns: ToolResult | None = None,
    ):
        self._name = name
        self._raises = raises
        self._returns = returns
        self.calls = 0

    @property
    def name(self) -> str:
        return self._name

    @property
    def tool_type(self) -> ToolType:
        return ToolType.LOCAL

    def get_schema(self) -> dict:
        return {
            "type": "function",
            "function": {"name": self._name, "description": "", "parameters": {}},
        }

    async def execute(self, arguments: dict) -> ToolResult:
        self.calls += 1
        if self._raises is not None:
            raise self._raises
        return self._returns or ToolResult.text("done")


class RefusingTool(BaseTool):
    """Refuses its arguments without doing any work, as boltz2 does."""

    def __init__(self, name: str, message: str):
        self._name = name
        self._message = message

    @property
    def name(self) -> str:
        return self._name

    @property
    def tool_type(self) -> ToolType:
        return ToolType.LOCAL

    def get_schema(self) -> dict:
        return {
            "type": "function",
            "function": {"name": self._name, "description": "", "parameters": {}},
        }

    async def execute(self, arguments: dict) -> ToolResult:
        return ToolResult.invalid_arguments(self._message)


class FakeRedis:
    """The subset of the async client the budget module uses."""

    def __init__(self):
        self.store: dict[str, str] = {}
        self.expiries: dict[str, int] = {}

    def __await__(self):
        async def _self():
            return self

        return _self().__await__()

    async def get(self, key):
        return self.store.get(key)

    async def set(self, key, value, ex=None):
        self.store[key] = value
        if ex is not None:
            self.expiries[key] = ex

    async def delete(self, key):
        self.store.pop(key, None)


@pytest.fixture(autouse=True)
def no_tool_spans():
    from api.telemetry import set_skip_tool_spans

    set_skip_tool_spans(True)
    yield
    set_skip_tool_spans(False)


@pytest.fixture(autouse=True)
def clean_budgets():
    """Budgets are process-global, so a test that sets them must put them back."""
    original = budget.active_budgets()
    yield
    budget.set_budgets(original)


def _install_redis_module(monkeypatch, get_redis) -> None:
    """Stand in for api.redis, which needs a client library the suite lacks.

    The budget module imports it lazily inside each function, so replacing the
    module wholesale is enough and avoids importing the real one.
    """
    module = types.ModuleType("api.redis")
    module.get_redis = get_redis
    monkeypatch.setitem(sys.modules, "api.redis", module)


@pytest.fixture
def redis(monkeypatch):
    client = FakeRedis()
    _install_redis_module(monkeypatch, lambda: client)
    return client


@pytest.fixture
def broken_redis(monkeypatch):
    def _raise():
        raise RuntimeError("REDIS_URL is not set")

    _install_redis_module(monkeypatch, _raise)


@pytest.fixture
def storage():
    return DataStorageService(repository=InMemoryDataArtifactRepository())


def registry_with(tool: BaseTool) -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(tool)
    return registry


class TestParsing:
    def test_reads_a_multi_tool_specification(self):
        assert budget.parse_budgets("boltz2=8,admet_predict=15") == {
            "boltz2": 8,
            "admet_predict": 15,
        }

    def test_tolerates_whitespace_and_trailing_commas(self):
        assert budget.parse_budgets(" boltz2 = 8 , ") == {"boltz2": 8}

    def test_an_empty_specification_means_no_budgets(self):
        assert budget.parse_budgets("") == {}
        assert budget.parse_budgets("   ") == {}

    def test_a_malformed_entry_is_skipped_rather_than_fatal(self):
        # A typo in a deployment variable should not stop the service booting.
        assert budget.parse_budgets("boltz2=8,nonsense,vina_dock=x") == {"boltz2": 8}

    def test_a_negative_limit_is_rejected(self):
        assert budget.parse_budgets("boltz2=-3") == {}

    def test_zero_is_a_real_limit_meaning_never(self):
        assert budget.parse_budgets("boltz2=0") == {"boltz2": 0}

    def test_the_environment_variable_is_read(self, monkeypatch):
        monkeypatch.setenv(budget.BUDGET_ENV_VAR, "vina_dock=4")
        assert budget.configured_budgets() == {"vina_dock": 4}


@pytest.mark.asyncio
class TestRefundingWhenThereIsNoBudget:
    """A tool that refuses its own arguments must still say why.

    The refund path re-reads the budget so the model is told the count after
    the refund rather than before. current_state returns None when the tool has
    no budget -- which, under a profile that runs without budgets, is every
    tool -- and two readers dereferenced it unguarded. The result was that an
    unrestricted run turned every argument rejection into
    "'NoneType' object has no attribute 'state'", stripping exactly the
    guidance the tool had written to help the model fix its call.
    """

    async def test_an_argument_rejection_survives_having_no_budget(self, redis, storage):
        budget.set_budgets({})
        registry = registry_with(
            RefusingTool("picky", "Constraint 1: constraint_type must be 'pocket' or 'bond'.")
        )

        result = await registry.execute("picky", {}, storage=storage, session_id="s1")

        assert "constraint_type" in result.to_tool_response()
        assert "NoneType" not in result.to_tool_response()

    async def test_the_same_rejection_still_reports_a_budget_when_there_is_one(
        self, redis, storage
    ):
        budget.set_budgets({"picky": 3})
        registry = registry_with(RefusingTool("picky", "bad arguments"))

        result = await registry.execute("picky", {}, storage=storage, session_id="s1")

        response = result.to_tool_response()
        assert "bad arguments" in response
        # Refunded, so the count is untouched and the note says so.
        assert "3" in response


class TestDescribingTheConfiguration:
    """A benchmark harness proves a run was unbudgeted by reading this line.

    So "no budgets" has to say so out loud. An absent line is equally consistent
    with logging being misconfigured, and a run whose whole claim is "the limits
    were off" cannot rest on an absence.
    """

    def test_budgets_are_named_with_their_limits(self):
        assert budget.describe_budgets({"boltz2": 8, "admet_predict": 15}) == (
            "Tool call budgets active: admet_predict=15, boltz2=8"
        )

    def test_no_budgets_still_states_the_case(self):
        described = budget.describe_budgets({})

        assert "none configured" in described
        assert "without limit" in described

    def test_the_two_states_cannot_be_confused_by_a_grep(self):
        """"inactive" would contain "active", which is how this goes wrong."""
        off = budget.describe_budgets({})
        on = budget.describe_budgets({"boltz2": 8})

        assert "active" not in off
        assert "budgets active" in on


@pytest.mark.asyncio
class TestSpending:
    async def test_an_unbudgeted_tool_is_never_charged(self, redis):
        budget.set_budgets({"boltz2": 2})

        verdict = await budget.spend("s1", "molecule_gates")

        assert verdict.allowed
        assert verdict.unbudgeted
        assert redis.store == {}

    async def test_each_call_decrements_what_is_left(self, redis):
        budget.set_budgets({"boltz2": 3})

        first = await budget.spend("s1", "boltz2")
        second = await budget.spend("s1", "boltz2")

        assert (first.state.used, first.state.remaining) == (1, 2)
        assert (second.state.used, second.state.remaining) == (2, 1)

    async def test_the_call_after_the_last_one_is_refused(self, redis):
        budget.set_budgets({"boltz2": 2})

        await budget.spend("s1", "boltz2")
        await budget.spend("s1", "boltz2")
        third = await budget.spend("s1", "boltz2")

        assert not third.allowed
        assert third.state.remaining == 0

    async def test_a_refusal_does_not_charge_again(self, redis):
        # Otherwise repeated attempts inflate `used` past the limit and the
        # reported numbers stop making sense.
        budget.set_budgets({"boltz2": 1})
        await budget.spend("s1", "boltz2")

        for _ in range(5):
            refused = await budget.spend("s1", "boltz2")

        assert refused.state.used == 1

    async def test_a_zero_budget_refuses_the_first_call(self, redis):
        budget.set_budgets({"boltz2": 0})

        assert not (await budget.spend("s1", "boltz2")).allowed

    async def test_sessions_do_not_share_a_counter(self, redis):
        budget.set_budgets({"boltz2": 1})

        await budget.spend("s1", "boltz2")
        other = await budget.spend("s2", "boltz2")

        assert other.allowed

    async def test_no_session_means_nothing_to_charge(self, redis):
        budget.set_budgets({"boltz2": 0})

        assert (await budget.spend(None, "boltz2")).allowed

    async def test_counters_are_given_an_expiry(self, redis):
        budget.set_budgets({"boltz2": 2})

        await budget.spend("s1", "boltz2")

        assert redis.expiries["budget:s1"] == budget.BUDGET_TTL

    async def test_a_refund_returns_one_call(self, redis):
        budget.set_budgets({"boltz2": 1})
        await budget.spend("s1", "boltz2")

        await budget.refund("s1", "boltz2")

        assert (await budget.spend("s1", "boltz2")).allowed

    async def test_a_refund_cannot_go_below_zero(self, redis):
        budget.set_budgets({"boltz2": 2})

        await budget.refund("s1", "boltz2")

        states = await budget.get_budget_states("s1")
        assert states[0].used == 0

    async def test_unreadable_counters_are_discarded_not_fatal(self, redis):
        budget.set_budgets({"boltz2": 2})
        redis.store["budget:s1"] = "{not json"

        assert (await budget.spend("s1", "boltz2")).allowed


@pytest.mark.asyncio
class TestRedisFailure:
    async def test_a_redis_outage_opens_the_budget(self, broken_redis):
        """Enforcement is cost control, not safety.

        Turning every budgeted tool into an error because a counter could not be
        read would be a worse outage than allowing an extra call.
        """
        budget.set_budgets({"boltz2": 1})

        verdict = await budget.spend("s1", "boltz2")

        assert verdict.allowed
        assert verdict.state is None

    async def test_the_prompt_block_survives_an_outage(self, broken_redis):
        budget.set_budgets({"boltz2": 4})

        states = await budget.get_budget_states("s1")

        # The limit is known even when spend is not, so report the limit rather
        # than dropping the block entirely.
        assert [(s.tool_name, s.limit, s.used) for s in states] == [("boltz2", 4, 0)]


@pytest.mark.asyncio
class TestBudgetStates:
    async def test_nothing_configured_means_nothing_reported(self, redis):
        budget.set_budgets({})

        assert await budget.get_budget_states("s1") == []

    async def test_the_most_depleted_budget_reads_first(self, redis):
        budget.set_budgets({"boltz2": 5, "vina_dock": 5})
        for _ in range(4):
            await budget.spend("s1", "boltz2")

        states = await budget.get_budget_states("s1")

        assert [state.tool_name for state in states] == ["boltz2", "vina_dock"]

    async def test_reset_clears_the_session(self, redis):
        budget.set_budgets({"boltz2": 2})
        await budget.spend("s1", "boltz2")

        await budget.reset("s1")

        states = await budget.get_budget_states("s1")
        assert states[0].used == 0


@pytest.mark.asyncio
class TestRegistryEnforcement:
    async def test_a_budgeted_tool_runs_while_it_has_calls(self, redis, storage):
        budget.set_budgets({"expensive": 2})
        tool = CountingTool()
        registry = registry_with(tool)

        result = await registry.execute(
            "expensive", {}, storage=storage, session_id="s1"
        )

        assert tool.calls == 1
        assert result.content_type != ContentType.ERROR

    async def test_an_exhausted_tool_is_not_run_at_all(self, redis, storage):
        # The whole point is to not spend the resource, so the refusal has to
        # happen before the call, not be reported after it.
        budget.set_budgets({"expensive": 1})
        tool = CountingTool()
        registry = registry_with(tool)

        await registry.execute("expensive", {}, storage=storage, session_id="s1")
        result = await registry.execute(
            "expensive", {}, storage=storage, session_id="s1"
        )

        assert tool.calls == 1
        assert result.content_type == ContentType.ERROR

    async def test_the_refusal_says_what_to_do_instead(self, redis, storage):
        budget.set_budgets({"expensive": 0})
        registry = registry_with(CountingTool())

        result = await registry.execute(
            "expensive", {}, storage=storage, session_id="s1"
        )

        text = result.to_tool_response()
        assert "0 of 0 calls used" in text
        # Retrying is the obvious wrong move, so rule it out explicitly.
        assert "keep returning this message" in text

    async def test_every_response_reports_what_is_left(self, redis, storage):
        budget.set_budgets({"expensive": 3})
        registry = registry_with(CountingTool())

        result = await registry.execute(
            "expensive", {}, storage=storage, session_id="s1"
        )

        assert "expensive 1/3 used, 2 calls left" in result.to_tool_response()

    async def test_the_last_call_says_it_was_the_last(self, redis, storage):
        budget.set_budgets({"expensive": 1})
        registry = registry_with(CountingTool())

        result = await registry.execute(
            "expensive", {}, storage=storage, session_id="s1"
        )

        assert "That was the last one" in result.to_tool_response()

    async def test_one_call_left_is_singular(self, redis, storage):
        budget.set_budgets({"expensive": 3})
        registry = registry_with(CountingTool())

        await registry.execute("expensive", {}, storage=storage, session_id="s1")
        result = await registry.execute(
            "expensive", {}, storage=storage, session_id="s1"
        )

        assert "1 call left" in result.to_tool_response()

    async def test_an_unbudgeted_tool_gains_no_annotation(self, redis, storage):
        budget.set_budgets({"other": 3})
        registry = registry_with(CountingTool())

        result = await registry.execute(
            "expensive", {}, storage=storage, session_id="s1"
        )

        assert result.to_tool_response() == "done"

    async def test_no_budgets_configured_changes_nothing(self, redis, storage):
        budget.set_budgets({})
        registry = registry_with(CountingTool())

        result = await registry.execute(
            "expensive", {}, storage=storage, session_id="s1"
        )

        assert result.to_tool_response() == "done"
        assert redis.store == {}

    async def test_a_failing_tool_still_costs_a_call(self, redis, storage):
        """Refunding failures would make an unlimited retry loop free.

        A tool that raises has usually already reached the expensive service, so
        the spend the budget exists to bound has happened either way.
        """
        budget.set_budgets({"expensive": 2})
        registry = registry_with(CountingTool(raises=RuntimeError("upstream 500")))

        await registry.execute("expensive", {}, storage=storage, session_id="s1")

        states = await budget.get_budget_states("s1")
        assert states[0].used == 1

    async def test_an_unresolvable_reference_is_refunded(self, redis, storage):
        """Nothing was spent: the arguments never resolved to data.

        Charging a docking call for a mistyped reference would punish the one
        failure the agent can fix for free.
        """
        budget.set_budgets({"expensive": 2})
        registry = registry_with(
            CountingTool(raises=InvalidReferenceError("no such ref pdb_zzz"))
        )

        result = await registry.execute(
            "expensive", {}, storage=storage, session_id="s1"
        )

        assert result.content_type == ContentType.ERROR
        states = await budget.get_budget_states("s1")
        assert states[0].used == 0

    async def test_the_annotation_survives_response_capping(self, redis, storage):
        # A capped response is exactly when the model most needs to know its
        # remaining spend, so the note is appended after truncation.
        budget.set_budgets({"chatty": 3})

        class ChattyTool(CountingTool):
            @property
            def max_response_tokens(self) -> int:
                return 10

            async def execute(self, arguments: dict) -> ToolResult:
                return ToolResult.text("z" * 5000)

        registry = registry_with(ChattyTool(name="chatty"))

        result = await registry.execute(
            "chatty", {}, storage=storage, session_id="s1"
        )

        assert "chatty 1/3 used" in result.to_tool_response()

    async def test_the_counter_is_stored_under_the_session(self, redis, storage):
        budget.set_budgets({"expensive": 2})
        registry = registry_with(CountingTool())

        await registry.execute("expensive", {}, storage=storage, session_id="abc")

        assert json.loads(redis.store["budget:abc"]) == {"expensive": 1}


@pytest.mark.asyncio
class TestRefundOnRejectedArguments:
    """Arguments the tool never accepted cost the resource nothing.

    Measured on two 100-task benchmark runs: 111 of 616 boltz2 calls were
    rejected on their arguments, each charged against a budget of eight per
    task. A misremembered field name should not cost as much as a co-fold.
    """

    async def test_a_rejected_call_is_given_back(self, redis, storage):
        budget.set_budgets({"expensive": 1})
        registry = registry_with(
            CountingTool(returns=ToolResult.invalid_arguments("binder must be a chain id"))
        )

        first = await registry.execute("expensive", {}, storage=storage, session_id="s1")

        assert first.content_type == ContentType.ERROR
        states = await budget.get_budget_states("s1")
        assert states[0].used == 0

    async def test_the_model_is_told_the_call_it_still_has(self, redis, storage):
        # Reusing the spend verdict here would say one fewer than it really has,
        # and an agent that thinks it is out of calls stops trying.
        budget.set_budgets({"expensive": 3})
        registry = registry_with(
            CountingTool(returns=ToolResult.invalid_arguments("constraint_type must be set"))
        )

        result = await registry.execute("expensive", {}, storage=storage, session_id="s1")

        assert "expensive 0/3 used, 3 calls left" in result.to_tool_response()

    async def test_a_tool_that_ran_and_failed_still_pays(self, redis, storage):
        # The guard that matters: a persistently failing endpoint must not be
        # callable without limit.
        budget.set_budgets({"expensive": 2})
        registry = registry_with(CountingTool(returns=ToolResult.error("the service is down")))

        await registry.execute("expensive", {}, storage=storage, session_id="s1")

        states = await budget.get_budget_states("s1")
        assert states[0].used == 1


class TestPromptBlock:
    def test_no_budgets_means_no_block(self):
        from api.state.format import format_tool_budgets, inject_state_into_prompt

        assert format_tool_budgets([]) == ""
        prompt = inject_state_into_prompt(
            "BASE", {"is_empty": True}, {"is_empty": True}, []
        )
        assert "TOOL BUDGET" not in prompt

    def test_the_block_leads_with_what_is_left(self):
        from api.state.format import format_tool_budgets

        block = format_tool_budgets(
            [{"tool_name": "boltz2", "limit": 8, "used": 6}]
        )

        assert "boltz2 2 of 8 left" in block

    def test_an_exhausted_tool_is_named_as_exhausted(self):
        # "0 left" is a number the model has to interpret; "exhausted" is not.
        from api.state.format import format_tool_budgets

        block = format_tool_budgets(
            [{"tool_name": "boltz2", "limit": 8, "used": 8}]
        )

        assert "boltz2 exhausted (8/8)" in block

    def test_a_batching_tool_says_a_call_is_the_unit(self):
        """The arithmetic the model reliably gets wrong.

        Measured over two 100-task benchmark runs: 4.27 molecules per
        admet_predict call against a budget of fifteen calls. It had the
        batching and spent the budget as though each call covered one molecule.
        """
        from api.state.format import format_tool_budgets

        block = format_tool_budgets(
            [{"tool_name": "admet_predict", "limit": 15, "used": 2, "batch_argument": "smiles"}]
        )

        assert "A call is the unit, not an item" in block
        assert "admet_predict takes a list in `smiles`" in block

    def test_a_tool_that_covers_one_thing_per_call_says_nothing_extra(self):
        # boltz2 predicts affinity for one ligand per request, so telling the
        # model to batch it would be advice it cannot follow.
        from api.state.format import format_tool_budgets

        block = format_tool_budgets([{"tool_name": "boltz2", "limit": 8, "used": 1}])

        assert "A call is the unit" not in block

    def test_an_exhausted_batching_tool_does_not_advertise_batching(self):
        from api.state.format import format_tool_budgets

        block = format_tool_budgets(
            [{"tool_name": "admet_predict", "limit": 15, "used": 15, "batch_argument": "smiles"}]
        )

        assert "A call is the unit" not in block

    def test_admet_predict_declares_its_batch_argument(self):
        from api.tools.admet import AdmetPredictTool

        assert AdmetPredictTool().batch_argument == "smiles"

    def test_a_tool_that_does_not_batch_declares_nothing(self):
        from api.tools.boltz2 import Boltz2Tool

        assert Boltz2Tool().batch_argument is None

    def test_overspend_does_not_report_a_negative(self):
        from api.state.format import format_tool_budgets

        block = format_tool_budgets(
            [{"tool_name": "boltz2", "limit": 2, "used": 5}]
        )

        assert "-" not in block.split("--")[0]

    def test_the_block_lands_in_the_envelope(self):
        from api.state.format import inject_state_into_prompt

        prompt = inject_state_into_prompt(
            "BASE",
            {"is_empty": True},
            {"is_empty": True},
            [],
            tool_budgets=[{"tool_name": "boltz2", "limit": 8, "used": 1}],
        )

        assert "[TOOL BUDGET:" in prompt

    def test_the_envelope_still_begins_at_the_sentinel(self):
        """The training agent strips from the sentinel before re-injecting.

        A block ahead of it would survive stripping and stack a stale copy every
        turn, so the budget block has to come after.
        """
        from api.state.format import INJECTION_SENTINEL, inject_state_into_prompt

        prompt = inject_state_into_prompt(
            "BASE",
            {"is_empty": True},
            {"is_empty": True},
            [],
            tool_budgets=[{"tool_name": "boltz2", "limit": 8, "used": 1}],
        )

        assert prompt.split(INJECTION_SENTINEL)[0] == "BASE"

    def test_the_block_is_in_the_never_degraded_floor(self):
        """Budget state must not be summarized away under context pressure.

        It lives in the system message, which the assembler never degrades --
        only trajectory turns are.
        """
        from langchain_core.messages import HumanMessage

        from api.context.assembler import ContextAssembler

        assembler = ContextAssembler("BASE")
        messages = assembler.assemble(
            {
                "messages": [HumanMessage(content="hi")],
                "tool_budgets": [{"tool_name": "boltz2", "limit": 8, "used": 7}],
            }
        )

        assert "[TOOL BUDGET:" in messages[0].content
