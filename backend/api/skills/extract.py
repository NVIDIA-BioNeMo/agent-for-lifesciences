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

"""Infer a skill's endpoint contract from its prose at install time.

Most skills in the wild have no ``endpoints:`` frontmatter -- they are markdown
written for a human, describing an HTTP API in prose and examples. To synthesize
a typed tool for such a skill we first have to recover the contract the prose
implies: the URL, which request fields carry molecular data, and where the
interesting values sit in the response.

That is a reading-comprehension task, so we ask a model, but the result is
treated as a *proposal*, never as truth:

- it is validated through the same Pydantic models as a declared contract, so a
  hallucinated field or data type is rejected rather than executed;
- it comes back ``confirmed=False``, which means no tool is synthesized yet;
- the install route hands it to the user, who confirms it -- which is also the
  moment they authorize outbound requests to the host it names.

Extraction is best-effort. A model that is unavailable, slow, or incoherent must
not block installing a skill: the skill still works the old way, through
``http_request`` and its prose.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass
from typing import Any, AsyncIterator, Iterable, Optional, TYPE_CHECKING

from api.storage.datatypes import DataType, friendly_name, heavy_type_values

from .contract import (
    SkillContract,
    SkillContractError,
    parse_contract,
)

if TYPE_CHECKING:
    from .parser import ParsedResource

logger = logging.getLogger(__name__)

# Only this much prose is sent in total, and only this much of it from SKILL.md,
# so a long body cannot crowd out the reference files. An unbounded bundle would
# make install latency and token cost a function of whatever the user dropped in.
MAX_PROSE_CHARS = 24_000
MAX_SKILL_MD_CHARS = 12_000

# Reference files whose names suggest a request/response schema are sent first.
# SKILL.md prose tends to describe an API narratively ("use txt for SMILES"),
# while references/api.md tends to hold the table that settles it.
_SCHEMA_FILE_RE = re.compile(
    r"(api|endpoint|schema|reference|spec|openapi|swagger|request|response)",
    re.IGNORECASE,
)

# Hard ceiling on the whole attempt, including the retry and the retries inside
# the client. Install is an interactive action; past this the user is better
# served by the UI opening an empty contract form.
EXTRACTION_TIMEOUT_SECONDS = 90.0

# Reasoning models spend output tokens on chain-of-thought before answering, and
# that thinking is *not* separately budgeted -- it comes out of this allowance.
# At 2048 a Nemotron model burned the entire budget mid-thought and returned
# empty content, which silently produced no contract and no tool.
MAX_OUTPUT_TOKENS = 8192

# One retry gets a larger allowance still. A truncated reply proves the model
# wanted more room, so repeating the same budget would just fail again.
RETRY_OUTPUT_TOKENS = 16384

# How much model output earns a fresh progress message. Extraction takes over a
# minute on a real skill, and reporting per chunk would be hundreds of events;
# reporting per this many characters keeps the message moving without flooding.
PROGRESS_REPORT_CHARS = 600

# Skip the model entirely unless the prose looks like it documents an HTTP API.
_URL_RE = re.compile(r"https?://[^\s\"'`)>]+", re.IGNORECASE)

# Recover the JSON object from a reply wrapped in prose or a ```json fence.
_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)

# Built by concatenation rather than str.format: the schema block is full of JSON
# braces, which format() would try to interpret as fields.
_SYSTEM_PROMPT = """You extract machine-readable API contracts from documentation.

You are given the text of a SKILL.md file that documents one or more HTTP
endpoints. Return ONLY a JSON object describing those endpoints. No prose, no
markdown fence.

Schema:
{
  "endpoints": [
    {
      "tool_suffix": "short_snake_case_verb",
      "summary": "one sentence describing what the endpoint does",
      "url": "https://full/endpoint/url",
      "method": "POST",
      "secret_name": "NAME_OF_API_KEY_ENV_VAR or null",
      "body_template": { ... static request fields with their literal values ... },
      "inputs": [
        {"name": "sequence", "data_type": "sequence", "json_path": "$.polymers[0].sequence"}
      ],
      "scalar_params": [
        {"name": "recycling_steps", "type": "integer", "json_path": "$.recycling_steps"}
      ],
      "outputs": [
        {"name": "structure", "json_path": "$.structures[0].structure", "data_type": "mmcif"},
        {"name": "pose", "json_path": "$.poses", "data_type": "sdf", "each": true}
      ]
    }
  ]
}

Rules:
- "inputs" is ONLY for fields carrying one of the heavy data types listed at the
  end of this message. Every other request field belongs in "scalar_params" (if
  the caller chooses it) or "body_template" (if it is a fixed value).
- "json_path" locates the field in the request body, e.g. "$.polymers[0].sequence".
  Use dot and [index] notation only. No wildcards or filters.
- "outputs" is for response fields worth keeping. Use "array" for numeric
  matrices or per-residue score lists (PAE, pLDDT, contact maps), and "json" for
  structured objects. Include auxiliary scores; they are not discarded.
- Set "each": true on an output whose path resolves to a LIST of same-typed
  values, such as ranked docking poses. Point json_path at the list itself
  ("$.poses"), not at one element ("$.poses[0]"), so every element is kept.
- If the documentation describes several ways to call one endpoint (a hosted URL
  and a local one, or several input formats), pick the hosted URL and the format
  the examples use most. Emit ONE endpoint per distinct URL, and do not deliberate
  about the alternatives -- answer with the JSON.
- Copy URLs, field names and paths EXACTLY as the documentation writes them.
  Never invent a field, a path, or an endpoint that the text does not mention.
- If the text documents no HTTP endpoint, return {"endpoints": []}.
"""

# Heavy types belong in "inputs"; array/json are output-only conveniences.
#
# Derived from `datatypes.py` rather than hand-listed. A hand-written subset here
# drifted from the vocabulary and omitted mol2, pdbqt, mol, cif and fasta -- so a
# skill documenting a mol2 ligand offered the model no valid classification, and
# it spent its output budget reasoning about the contradiction instead of
# answering. Enum order keeps the prompt stable between runs.
_HEAVY_TYPES = tuple(dt for dt in DataType if dt.value in heavy_type_values())
_OUTPUT_ONLY_TYPES = (DataType.ARRAY, DataType.JSON)


def _data_type_help() -> str:
    heavy = "\n".join(f"- {dt.value}: {friendly_name(dt)}" for dt in _HEAVY_TYPES)
    other = "\n".join(f"- {dt.value}: {friendly_name(dt)}" for dt in _OUTPUT_ONLY_TYPES)
    return (
        "Heavy data types, valid for both inputs and outputs:\n"
        f"{heavy}\n\n"
        "Output-only data types:\n"
        f"{other}"
    )


def prose_mentions_endpoint(text: str) -> bool:
    """Whether the prose looks like it documents an HTTP endpoint.

    A cheap gate so purely instructional skills (no API) never pay for a model
    call at install time.
    """
    return bool(_URL_RE.search(text or ""))


def build_extraction_prose(
    body: str,
    resources: Optional[Iterable["ParsedResource"]] = None,
) -> str:
    """Assemble the documentation the extractor reads.

    SKILL.md alone is often the *least* precise description of a skill's API: it
    is written to be read narratively and defers the exact request and response
    schema to a bundled reference file. Sending only the body leaves the model
    inferring field types from examples, which is where it burns its budget.

    Schema-looking files come first, then the rest, until the budget runs out.
    """
    sections = [
        "--- SKILL.md ---",
        (body or "")[:MAX_SKILL_MD_CHARS],
        "--- end SKILL.md ---",
    ]
    remaining = MAX_PROSE_CHARS - len("\n".join(sections))

    files = list(resources or [])
    files.sort(key=lambda r: (0 if _SCHEMA_FILE_RE.search(r.path) else 1, r.path))

    for resource in files:
        header = f"--- {resource.path} ---"
        footer = f"--- end {resource.path} ---"
        # Three sections appended means three more newlines from the join.
        available = remaining - len(header) - len(footer) - 3
        if available <= 0:
            break
        try:
            text = resource.content.decode("utf-8")[:available]
        except UnicodeDecodeError:
            # Resources are declared .md/.txt; a non-UTF-8 one is not documentation.
            continue
        sections += [header, text, footer]
        remaining = available - len(text)

    return "\n".join(sections)


def build_messages(name: str, description: str, prose: str) -> list[dict[str, str]]:
    """Build the extraction chat messages."""
    system = f"{_SYSTEM_PROMPT}\n{_data_type_help()}\n"
    user = (
        "Extract the endpoint contract from this skill's documentation.\n\n"
        f"Skill name: {name or '(unnamed)'}\n"
        f"Skill description: {description or '(none)'}\n\n"
        f"{prose or ''}\n\n"
        "Return only the JSON object."
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def parse_extraction_response(content: str) -> dict[str, Any]:
    """Pull the JSON object out of a model reply.

    Tolerates a ```json fence or surrounding chatter, since we cannot rely on
    every configured model honoring "return only JSON".

    Raises:
        SkillContractError: If no JSON object can be recovered.
    """
    text = (content or "").strip()
    if not text:
        raise SkillContractError("Model returned an empty response.")

    # Thinking models emit a reasoning block before the answer.
    for tag in ("</think>", "</thinking>"):
        if tag in text:
            text = text.split(tag)[-1].strip()

    fence = _FENCE_RE.search(text)
    if fence:
        text = fence.group(1).strip()

    if not text.startswith("{"):
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end <= start:
            raise SkillContractError("Model response contained no JSON object.")
        text = text[start : end + 1]

    try:
        payload = json.loads(text)
    except ValueError as exc:
        raise SkillContractError(f"Model response was not valid JSON: {exc}") from exc

    if not isinstance(payload, dict):
        raise SkillContractError("Model response was not a JSON object.")
    return payload


@dataclass
class ExtractionOutcome:
    """What extraction produced, and if nothing, why.

    The reason matters as much as the contract. Extraction failing used to be an
    INFO log and a ``None``, so the install UI showed a plain "Skill installed!"
    and the user had no way to know that a tool was supposed to be synthesized
    and was not. A skill that quietly keeps working through ``http_request``
    looks identical to one that gained a tool, right up until the model spends
    twenty iterations failing to pass a PDB file through a tool argument.
    """

    contract: Optional[SkillContract] = None
    error: Optional[str] = None


@dataclass
class ExtractionProgress:
    """A note about what extraction is doing, for the install UI to display.

    Emitted when the model produces output rather than on a timer: a single
    "working out the contract" message sat on screen for a whole minute, which
    is indistinguishable from a hung request.
    """

    message: str


async def extract_contract(
    name: str,
    description: str,
    prose: str,
    *,
    timeout: float = EXTRACTION_TIMEOUT_SECONDS,
) -> ExtractionOutcome:
    """Propose a contract for a skill that did not declare one.

    Returns an :class:`ExtractionOutcome` holding an unconfirmed
    :class:`SkillContract` for the user to review, or an ``error`` explaining why
    none was proposed. Never raises: installing a skill must not depend on this
    succeeding.
    """
    outcome = ExtractionOutcome()
    async for event in extract_contract_events(
        name, description, prose, timeout=timeout
    ):
        if isinstance(event, ExtractionOutcome):
            outcome = event
    return outcome


async def extract_contract_events(
    name: str,
    description: str,
    prose: str,
    *,
    timeout: float = EXTRACTION_TIMEOUT_SECONDS,
) -> AsyncIterator[ExtractionProgress | ExtractionOutcome]:
    """Run extraction, reporting progress as the model produces output.

    Yields :class:`ExtractionProgress` while the model works and exactly one
    :class:`ExtractionOutcome` last. :func:`extract_contract` is the same thing
    without the narration.
    """
    if not prose_mentions_endpoint(prose):
        logger.info("Skill '%s': no URL in prose, skipping contract extraction", name)
        yield ExtractionOutcome()
        return

    messages = build_messages(name, description, prose)
    content = ""
    finish_reason: Optional[str] = None
    last_error: Optional[str] = None

    # A reasoning model spends output tokens thinking before it answers, so a
    # truncated reply means "needed more room", not "had nothing to say".
    for attempt, max_tokens in enumerate((MAX_OUTPUT_TOKENS, RETRY_OUTPUT_TOKENS)):
        if attempt:
            yield ExtractionProgress(
                message=(
                    f"The first attempt ran out of room at {MAX_OUTPUT_TOKENS} "
                    "tokens. Trying again with more space..."
                )
            )

        content = ""
        reasoning_chars = 0
        next_report = PROGRESS_REPORT_CHARS
        announced_writing = False

        try:
            async with asyncio.timeout(timeout):
                async for kind, text in _stream_completion(messages, max_tokens):
                    if kind == "finish":
                        finish_reason = text
                        continue
                    if kind == "content":
                        content += text
                        if not announced_writing:
                            announced_writing = True
                            next_report = len(content) + PROGRESS_REPORT_CHARS
                            yield ExtractionProgress(
                                message="Writing out the contract..."
                            )
                        elif len(content) >= next_report:
                            next_report = len(content) + PROGRESS_REPORT_CHARS
                            yield ExtractionProgress(
                                message=(
                                    "Writing out the contract... "
                                    f"({_word_estimate(len(content))} words)"
                                )
                            )
                        continue

                    reasoning_chars += len(text)
                    if reasoning_chars >= next_report:
                        next_report = reasoning_chars + PROGRESS_REPORT_CHARS
                        yield ExtractionProgress(
                            message=(
                                "Working through the documentation... "
                                f"({_word_estimate(reasoning_chars)} words of reasoning)"
                            )
                        )
        except TimeoutError:
            logger.warning(
                "Skill '%s': contract extraction timed out after %.0fs", name, timeout
            )
            yield ExtractionOutcome(
                error=f"Reading the skill's documentation timed out after {timeout:.0f}s."
            )
            return
        except Exception as exc:  # noqa: BLE001 - extraction is always optional
            logger.warning("Skill '%s': contract extraction call failed: %s", name, exc)
            yield ExtractionOutcome(
                error=f"Could not reach the model to read this skill: {exc}"
            )
            return

        if content.strip():
            break

        last_error = (
            f"the model used its entire {max_tokens}-token output budget on "
            "reasoning and returned no answer"
            if finish_reason == "length"
            else "the model returned an empty response"
        )
        logger.warning(
            "Skill '%s': extraction attempt at %d output tokens produced no content "
            "(finish_reason=%s)",
            name,
            max_tokens,
            finish_reason,
        )

    if not content.strip():
        yield ExtractionOutcome(
            error=(
                f"Aevom could not read this skill's API description: {last_error}. "
                "Add an 'endpoints:' block to SKILL.md to declare the contract "
                "explicitly, or the skill will work through http_request only."
            )
        )
        return

    yield ExtractionProgress(message="Checking the proposed contract...")

    try:
        payload = parse_extraction_response(content)
        contract = parse_contract(payload, source="llm")
    except SkillContractError as exc:
        # Common and expected: the model invented a field or mangled a path.
        # The user can still fill the contract in by hand in the review step.
        logger.info("Skill '%s': proposed contract was invalid: %s", name, exc)
        yield ExtractionOutcome(
            error=f"Aevom proposed an invalid contract for this skill: {exc}"
        )
        return

    if not contract.endpoints:
        logger.info("Skill '%s': model found no endpoints", name)
        yield ExtractionOutcome()
        return

    # Belt and braces: never hand back something that would synthesize a tool
    # without the user having seen it.
    contract = contract.model_copy(update={"confirmed": False, "source": "llm"})
    logger.info(
        "Skill '%s': proposed %d endpoint(s) on host(s) %s (awaiting confirmation)",
        name,
        len(contract.endpoints),
        contract.hosts,
    )
    yield ExtractionOutcome(contract=contract)


def _word_estimate(chars: int) -> int:
    """Roughly how many words some number of characters is.

    Only used to give a progress counter a unit people read easily; five
    characters plus a space is the usual English average.
    """
    return max(1, chars // 6)


async def _stream_completion(
    messages: list[dict[str, str]], max_tokens: int
) -> AsyncIterator[tuple[str, str]]:
    """Yield ``("reasoning" | "content" | "finish", text)`` from one completion.

    ``LLMClient.stream`` returns a *synchronous* iterator wrapping the OpenAI
    client, so it is pumped on a worker thread and the pieces handed back
    through a queue. Iterating it here directly would block the event loop for
    the length of the call, stalling every other request.
    """
    from api.llm import get_client

    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[Optional[tuple[str, str]]] = asyncio.Queue()

    def _pump() -> None:
        try:
            for chunk in get_client().stream(
                messages=messages, temperature=0.0, max_tokens=max_tokens
            ):
                if chunk.reasoning:
                    loop.call_soon_threadsafe(
                        queue.put_nowait, ("reasoning", chunk.reasoning)
                    )
                if chunk.content:
                    loop.call_soon_threadsafe(
                        queue.put_nowait, ("content", chunk.content)
                    )
                if chunk.finish_reason:
                    loop.call_soon_threadsafe(
                        queue.put_nowait, ("finish", chunk.finish_reason)
                    )
        finally:
            # Sentinel, so a provider error still terminates the consumer. The
            # exception itself surfaces when the task is awaited below.
            loop.call_soon_threadsafe(queue.put_nowait, None)

    pump = asyncio.create_task(asyncio.to_thread(_pump))
    try:
        while True:
            item = await queue.get()
            if item is None:
                break
            yield item
        # Re-raises whatever the provider threw, which the caller turns into an
        # outcome the install UI can show.
        await pump
    finally:
        pump.cancel()


