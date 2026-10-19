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

"""Tests for LLM-assisted skill contract extraction.

Extraction reads a skill's prose and proposes a contract. Two properties matter
more than accuracy: it must never raise (installing a skill cannot depend on a
model being reachable), and it must never return something confirmed, because
confirmation is what authorizes outbound calls to the host it names.
"""

import asyncio
import json

import pytest

from api.storage.datatypes import heavy_type_values

from api.skills import extract
from api.skills.contract import SkillContractError
from api.skills.extract import (
    build_extraction_prose,
    build_messages,
    extract_contract,
    parse_extraction_response,
    prose_mentions_endpoint,
)
from api.skills.parser import ParsedResource

BOLTZ_PROSE = """
# Boltz-2 Structure Prediction

POST https://health.api.nvidia.com/v1/biology/mit/boltz2/predict

Send a protein sequence and an optional ligand SMILES.
"""

VALID_PAYLOAD = {
    "endpoints": [
        {
            "tool_suffix": "predict",
            "summary": "Predict a protein-ligand complex.",
            "url": "https://health.api.nvidia.com/v1/biology/mit/boltz2/predict",
            "method": "POST",
            "secret_name": "NVIDIA_API_KEY",
            "inputs": [
                {
                    "name": "sequence",
                    "data_type": "sequence",
                    "json_path": "$.polymers[0].sequence",
                }
            ],
            "outputs": [
                {
                    "name": "structure",
                    "json_path": "$.structures[0].structure",
                    "data_type": "mmcif",
                }
            ],
        }
    ]
}


class TestProseGate:
    @pytest.mark.parametrize(
        "prose",
        [
            "Call https://api.example.com/predict",
            "see http://localhost:8000/v1/thing",
            BOLTZ_PROSE,
        ],
    )
    def test_detects_a_url(self, prose):
        assert prose_mentions_endpoint(prose)

    @pytest.mark.parametrize("prose", ["", None, "Use the search_pubchem tool."])
    def test_no_url_means_no_model_call(self, prose):
        # Purely instructional skills must not pay for a completion at install.
        assert not prose_mentions_endpoint(prose)


class TestBuildMessages:
    def test_includes_the_skill_and_its_prose(self):
        messages = build_messages("Boltz-2", "Predict structures.", BOLTZ_PROSE)
        user = messages[1]["content"]
        assert "Boltz-2" in user
        assert "Predict structures." in user
        assert "boltz2/predict" in user

    def test_system_prompt_enumerates_the_data_types(self):
        system = build_messages("S", "d", BOLTZ_PROSE)[0]["content"]
        for expected in ("smiles", "sdf", "pdb", "mmcif", "sequence", "msa", "array"):
            assert expected in system

    def test_system_prompt_offers_every_heavy_type(self):
        # A hand-written subset here omitted mol2 and pdbqt, leaving a documented
        # mol2 ligand with no valid classification for the model to choose.
        system = build_messages("S", "d", BOLTZ_PROSE)[0]["content"]
        for data_type in heavy_type_values():
            assert data_type in system

    def test_system_prompt_documents_list_outputs(self):
        system = build_messages("S", "d", BOLTZ_PROSE)[0]["content"]
        assert '"each"' in system

    def test_system_prompt_distinguishes_safe_from_smiles(self):
        # `safe` appearing in the list is not enough. A SAFE string is valid
        # SMILES, so the model can only choose between them from the friendly
        # text -- which is why that text names GenMol rather than just saying
        # "fragment notation".
        system = build_messages("S", "d", BOLTZ_PROSE)[0]["content"]
        safe_line = next(
            line for line in system.splitlines() if line.startswith("- safe:")
        )
        assert "GenMol" in safe_line


class TestBuildExtractionProse:
    def test_includes_the_body(self):
        prose = build_extraction_prose(BOLTZ_PROSE)
        assert "boltz2/predict" in prose
        assert "--- SKILL.md ---" in prose

    def test_appends_reference_files(self):
        # The request/response schema usually lives in a reference file, not the
        # body, so extraction has to see it.
        prose = build_extraction_prose(
            BOLTZ_PROSE,
            [ParsedResource(path="references/api.md", content=b"| protein | string |")],
        )
        assert "--- references/api.md ---" in prose
        assert "| protein | string |" in prose

    def test_schema_files_come_before_the_rest(self):
        prose = build_extraction_prose(
            BOLTZ_PROSE,
            [
                ParsedResource(path="references/zzz_science.md", content=b"SCIENCE"),
                ParsedResource(path="references/api.md", content=b"SCHEMA"),
            ],
        )
        assert prose.index("SCHEMA") < prose.index("SCIENCE")

    def test_total_size_is_bounded(self):
        prose = build_extraction_prose(
            "x" * (extract.MAX_SKILL_MD_CHARS + 5000),
            [
                ParsedResource(path=f"r{i}.md", content=b"y" * 20_000)
                for i in range(10)
            ],
        )
        assert len(prose) <= extract.MAX_PROSE_CHARS

    def test_skips_non_utf8_resources(self):
        prose = build_extraction_prose(
            BOLTZ_PROSE, [ParsedResource(path="bad.md", content=b"\xff\xfe binary")]
        )
        assert "bad.md" not in prose

    def test_system_prompt_survived_brace_escaping(self):
        # The schema block is full of JSON braces; building it with str.format
        # would raise a KeyError on the first one.
        system = build_messages("S", "d", BOLTZ_PROSE)[0]["content"]
        assert '"endpoints"' in system
        assert '"tool_suffix"' in system


class TestParseExtractionResponse:
    def test_parses_a_bare_object(self):
        assert parse_extraction_response('{"endpoints": []}') == {"endpoints": []}

    def test_strips_a_json_fence(self):
        raw = '```json\n{"endpoints": []}\n```'
        assert parse_extraction_response(raw) == {"endpoints": []}

    def test_strips_surrounding_chatter(self):
        raw = 'Sure! Here is the contract:\n{"endpoints": []}\nHope that helps.'
        assert parse_extraction_response(raw) == {"endpoints": []}

    def test_drops_a_reasoning_block(self):
        raw = '<think>The docs mention one POST.</think>\n{"endpoints": []}'
        assert parse_extraction_response(raw) == {"endpoints": []}

    @pytest.mark.parametrize(
        "raw",
        [
            "",
            "   ",
            "I could not find any endpoints.",
            "{not json at all}",
            '["endpoints"]',
        ],
    )
    def test_unrecoverable_replies_raise(self, raw):
        with pytest.raises(SkillContractError):
            parse_extraction_response(raw)


@pytest.fixture
def stub_completion(monkeypatch):
    """Replace the streaming model call with a canned reply or an exception.

    ``replies`` may be a list of ``(content, finish_reason)`` pairs, one per
    attempt, so a test can make the first call come back truncated and the retry
    succeed. ``reasoning`` is streamed before the content, which is what the
    progress messages are derived from.
    """

    def _install(
        reply=None,
        *,
        replies=None,
        error=None,
        delay=0.0,
        finish="stop",
        reasoning="",
    ):
        calls: list[tuple[list[dict], int]] = []
        queued = list(replies) if replies is not None else None

        async def _stream_completion(messages, max_tokens):
            calls.append((messages, max_tokens))
            if delay:
                await asyncio.sleep(delay)
            if error is not None:
                raise error
            content, finish_reason = (
                queued.pop(0) if queued is not None else (reply, finish)
            )
            # Delivered in pieces, like the real stream: the progress counter
            # only advances on chunk boundaries.
            for start in range(0, len(reasoning), 100):
                yield "reasoning", reasoning[start : start + 100]
            if content:
                yield "content", content
            if finish_reason:
                yield "finish", finish_reason

        monkeypatch.setattr(extract, "_stream_completion", _stream_completion)
        return calls

    return _install


@pytest.mark.asyncio
class TestExtractContract:
    async def test_returns_an_unconfirmed_llm_contract(self, stub_completion):
        stub_completion(json.dumps(VALID_PAYLOAD))
        contract = (await extract_contract("Boltz-2", "Predict.", BOLTZ_PROSE)).contract

        assert contract is not None
        assert contract.source == "llm"
        # The whole point of the review step: no tool exists yet.
        assert contract.confirmed is False
        assert contract.is_actionable is False
        assert contract.hosts == ["health.api.nvidia.com"]

    async def test_a_confirmed_flag_from_the_model_is_ignored(self, stub_completion):
        payload = dict(VALID_PAYLOAD, confirmed=True, source="declared")
        stub_completion(json.dumps(payload))
        contract = (await extract_contract("Boltz-2", "Predict.", BOLTZ_PROSE)).contract
        assert contract.confirmed is False
        assert contract.source == "llm"

    async def test_skips_the_model_when_there_is_no_url(self, stub_completion):
        calls = stub_completion(json.dumps(VALID_PAYLOAD))
        outcome = await extract_contract("Notes", "d", "Just instructions.")
        assert outcome.contract is None
        # A skill with no API is not a failure, so there is nothing to report.
        assert outcome.error is None
        assert calls == []

    async def test_empty_endpoint_list_yields_no_contract(self, stub_completion):
        stub_completion('{"endpoints": []}')
        outcome = await extract_contract("S", "d", BOLTZ_PROSE)
        assert outcome.contract is None
        assert outcome.error is None

    @pytest.mark.parametrize(
        "reply",
        [
            json.dumps(
                {"endpoints": [{"tool_suffix": "go", "summary": "s", "url": "nope"}]}
            ),
            "I'm not sure what you mean.",
        ],
    )
    async def test_an_unusable_reply_is_reported(self, stub_completion, reply):
        stub_completion(reply)
        outcome = await extract_contract("S", "d", BOLTZ_PROSE)
        assert outcome.contract is None
        assert outcome.error

    async def test_hallucinated_data_type_is_reported(self, stub_completion):
        payload = json.loads(json.dumps(VALID_PAYLOAD))
        payload["endpoints"][0]["inputs"][0]["data_type"] = "protein_blob"
        stub_completion(json.dumps(payload))
        outcome = await extract_contract("S", "d", BOLTZ_PROSE)
        assert outcome.contract is None
        assert outcome.error

    async def test_model_error_is_reported(self, stub_completion):
        stub_completion(error=RuntimeError("no model configured"))
        outcome = await extract_contract("S", "d", BOLTZ_PROSE)
        assert outcome.contract is None
        assert "no model configured" in outcome.error

    async def test_slow_model_times_out_without_raising(self, stub_completion):
        stub_completion(json.dumps(VALID_PAYLOAD), delay=0.2)
        outcome = await extract_contract("S", "d", BOLTZ_PROSE, timeout=0.01)
        assert outcome.contract is None
        assert "timed out" in outcome.error


@pytest.mark.asyncio
class TestTruncatedReply:
    """A reasoning model can spend the whole output budget before answering.

    This is what happened to the DiffDock skill: 2048 output tokens all went to
    chain-of-thought, `content` came back empty, and the install silently
    produced no contract and therefore no tool.
    """

    async def test_retries_with_a_larger_budget(self, stub_completion):
        calls = stub_completion(
            replies=[("", "length"), (json.dumps(VALID_PAYLOAD), "stop")]
        )
        outcome = await extract_contract("Boltz-2", "Predict.", BOLTZ_PROSE)

        assert outcome.contract is not None
        assert [max_tokens for _, max_tokens in calls] == [
            extract.MAX_OUTPUT_TOKENS,
            extract.RETRY_OUTPUT_TOKENS,
        ]

    async def test_a_second_truncation_reports_the_budget(self, stub_completion):
        stub_completion(replies=[("", "length"), ("", "length")])
        outcome = await extract_contract("Boltz-2", "Predict.", BOLTZ_PROSE)

        assert outcome.contract is None
        assert str(extract.RETRY_OUTPUT_TOKENS) in outcome.error
        assert "reasoning" in outcome.error

    async def test_a_successful_first_call_does_not_retry(self, stub_completion):
        calls = stub_completion(json.dumps(VALID_PAYLOAD))
        assert (await extract_contract("S", "d", BOLTZ_PROSE)).contract is not None
        assert len(calls) == 1

    async def test_an_empty_reply_that_was_not_truncated_is_reported(
        self, stub_completion
    ):
        stub_completion(replies=[("", "stop"), ("", "stop")])
        outcome = await extract_contract("S", "d", BOLTZ_PROSE)
        assert "empty response" in outcome.error


@pytest.mark.asyncio
class TestProgressReporting:
    """Extraction narrates itself while the model is working.

    On a real skill this call takes over a minute. Reporting only at the start
    left one message on screen for the whole wait, which looks identical to a
    hung request, so progress is tied to the model actually producing output.
    """

    async def _run(self, **kwargs):
        """Collect the progress messages and the single final outcome."""
        messages: list[str] = []
        outcome = None
        async for event in extract.extract_contract_events(
            "Boltz-2", "Predict.", BOLTZ_PROSE, **kwargs
        ):
            if isinstance(event, extract.ExtractionProgress):
                messages.append(event.message)
            else:
                outcome = event
        return messages, outcome

    async def test_reasoning_advances_the_message(self, stub_completion):
        # Well past PROGRESS_REPORT_CHARS, so several updates are due.
        stub_completion(
            json.dumps(VALID_PAYLOAD),
            reasoning="thinking " * 500,
        )
        messages, outcome = await self._run()

        assert outcome.contract is not None
        thinking = [m for m in messages if "Working through the documentation" in m]
        assert len(thinking) > 1, messages

    async def test_updates_stop_short_of_one_per_chunk(self, stub_completion):
        """Reporting every chunk would be hundreds of events for one install."""
        reasoning = "x" * (extract.PROGRESS_REPORT_CHARS * 4)
        stub_completion(json.dumps(VALID_PAYLOAD), reasoning=reasoning)
        messages, _ = await self._run()

        chunks = len(reasoning) / 100
        assert len(messages) < chunks

    async def test_writing_the_answer_is_its_own_message(self, stub_completion):
        stub_completion(json.dumps(VALID_PAYLOAD), reasoning="thinking " * 200)
        messages, _ = await self._run()
        assert any("Writing out the contract" in m for m in messages)

    async def test_validation_is_announced_before_it_can_fail(self, stub_completion):
        stub_completion("I'm not sure what you mean.")
        messages, outcome = await self._run()
        assert any("Checking the proposed contract" in m for m in messages)
        assert outcome.error

    async def test_the_retry_explains_itself(self, stub_completion):
        stub_completion(replies=[("", "length"), (json.dumps(VALID_PAYLOAD), "stop")])
        messages, outcome = await self._run()

        assert outcome.contract is not None
        assert any("ran out of room" in m for m in messages)
        assert any(str(extract.MAX_OUTPUT_TOKENS) in m for m in messages)

    async def test_a_skill_with_no_api_reports_nothing(self, stub_completion):
        stub_completion(json.dumps(VALID_PAYLOAD))
        messages = []
        async for event in extract.extract_contract_events("Notes", "d", "No API here."):
            if isinstance(event, extract.ExtractionProgress):
                messages.append(event.message)
        assert messages == []

    async def test_exactly_one_outcome_is_emitted(self, stub_completion):
        stub_completion(json.dumps(VALID_PAYLOAD), reasoning="thinking " * 300)
        outcomes = [
            event
            async for event in extract.extract_contract_events(
                "Boltz-2", "Predict.", BOLTZ_PROSE
            )
            if isinstance(event, extract.ExtractionOutcome)
        ]
        assert len(outcomes) == 1
