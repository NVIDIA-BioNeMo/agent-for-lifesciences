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

"""Training telemetry -- wraps production OTel instrumentation for RL training.

Reuses the exact same OTel infrastructure as production (api.telemetry), adding:
- Configurable per-episode sampling (separate rates for train vs eval)
- EpisodeSpan: long-lived root span that covers seed_session -> verify
- Training-specific span attributes (rewards, wandb run ID, episode ID)
- W3C traceparent propagation helpers (the simple_agent runs in a separate
  process from the resources server; the agent loop's live spans must
  parent under the EpisodeSpan's chat_request via this propagation)
- finalize_trace(): root-attribute aggregation + outcome computation at
  verify time. Live agent-loop spans (agent.turn.{N}, policy_llm.responses,
  tool.{name}) are emitted in real time by the simple_agent overlay +
  ToolRegistry; the synthetic retroactive children that pre-2026-05-27
  build_trace() produced are no longer created here.

Usage in the NeMo-Gym resource server (app.py):
    from api.training.telemetry import (
        init_training_telemetry, TrainingTelemetryConfig, EpisodeSpan,
    )

    # At startup (sample_rate controls OTel-level trace sampling)
    init_training_telemetry(sample_rate=1.0)

    # At seed_session -- always create, sampling handled by OTel sampler
    session.episode_span = EpisodeSpan(task_prompt, episode_id, ...)
    traceparent = session.episode_span.traceparent()  # serialise W3C header
    # ... return traceparent in seed_session response body so the
    # simple_agent can attach its live spans under this trace.

    # At execute_tool -- live tool.{name} spans now emit by default
    # (set_skip_tool_spans(False) is the new training default).

    # At verify
    if session.episode_span:
        session.episode_span.finalize_trace(body.response, session.tool_calls, final_response=...)
        session.episode_span.set_rewards(total, breakdown)
        session.episode_span.end()
"""

import json
import logging
import os
import random
import time
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Optional

from opentelemetry import context as otel_context
from opentelemetry import trace
from opentelemetry.trace import Span, Status, StatusCode
from opentelemetry.trace.propagation.tracecontext import (
    TraceContextTextMapPropagator,
)

from api.telemetry import (
    init_telemetry,
    get_tracer,
    set_current_chat_span,
    set_io,
    set_llm_tools,
    set_span_kind,
)

logger = logging.getLogger(__name__)


class TrainingTelemetryConfig:
    """Sampling rates for training telemetry, read from resource server config."""

    def __init__(
        self,
        sample_rate_train: float = 0.5,
        sample_rate_eval: float = 1.0,
    ):
        self.sample_rate_train = max(0.0, min(1.0, sample_rate_train))
        self.sample_rate_eval = max(0.0, min(1.0, sample_rate_eval))


def init_training_telemetry(sample_rate: float = 1.0) -> None:
    """Initialize OTel for training with optional trace-level sampling.

    When ``sample_rate`` < 1.0, a ``TraceIdRatioBased`` sampler is
    installed so entire traces (parent + all children) are dropped
    atomically -- no more orphan tool spans.

    Reads PHOENIX_COLLECTOR_ENDPOINT and PHOENIX_PROJECT_NAME from env
    (same as production).  Logs the Phoenix UI URL so the developer can
    find it in the SLURM output.
    """
    import api.telemetry as _tel

    # Silence OTel's ``Failed to detach context`` log spam.
    #
    # OTel's ``opentelemetry.context.detach()`` catches the ValueError
    # raised by ``ContextVar.reset(token)`` when the token's contextvars
    # context differs from the current one, and unconditionally calls
    # ``logger.exception("Failed to detach context")``. This is a normal
    # / expected outcome in any FastAPI deployment where attach happens
    # in one request handler and detach in another (EpisodeSpan), or
    # where starlette's BaseHTTPMiddleware spawns sub-tasks that fork
    # the contextvars context (every /tool_name request). The detach
    # is functionally a no-op in those cases (the contextvars value
    # dies with the task anyway), but the spam is real: v0527c-tool-
    # headers-fix had 3109 of these stack traces in 3.7h, accreting
    # measurable IO overhead and drowning out actual errors. Set the
    # OTel context logger to CRITICAL so genuine failures still surface
    # (we have no other paths into that logger) but the cross-task
    # detach noise stays out of the driver log.
    logging.getLogger("opentelemetry.context").setLevel(logging.CRITICAL)

    endpoint = os.environ.get("PHOENIX_COLLECTOR_ENDPOINT", "")
    project = os.environ.get("PHOENIX_PROJECT_NAME", "aevom")

    if not endpoint:
        logger.warning(
            "PHOENIX_COLLECTOR_ENDPOINT not set -- training telemetry disabled"
        )
        return

    sample_rate = max(0.0, min(1.0, sample_rate))
    init_succeeded = False
    try:
        from phoenix.otel import register
        from opentelemetry.sdk.trace.sampling import TraceIdRatioBased, ALWAYS_ON

        sampler = ALWAYS_ON if sample_rate >= 1.0 else TraceIdRatioBased(sample_rate)

        tracer_provider = register(
            project_name=project,
            endpoint=endpoint,
            sampler=sampler,
        )

        _tel._tracer_provider = tracer_provider
        _tel._tracer = trace.get_tracer("aevom", tracer_provider=tracer_provider)
        _tel._initialized = True
        init_succeeded = True

    except Exception:
        logger.exception("Failed to init training telemetry with sampler, falling back")
        init_telemetry()
        # init_telemetry() may itself have hit the same ImportError and
        # left _tel._tracer as a no-op. Reflect that honestly so we don't
        # print a misleading "telemetry enabled" message; consumers can
        # check `_tel._initialized` AND `_tel._tracer_provider` to detect
        # the silent-failure mode.
        init_succeeded = bool(getattr(_tel, "_tracer_provider", None))

    ui_url = endpoint.replace(":4317", ":6006")
    if init_succeeded:
        logger.info(
            f"[AEVOM-TELEMETRY] Training telemetry enabled\n"
            f"  Phoenix endpoint : {endpoint}\n"
            f"  Phoenix UI       : {ui_url}\n"
            f"  Project          : {project}\n"
            f"  Sample rate      : {sample_rate}"
        )
        print(
            f"[AEVOM-TELEMETRY] Phoenix traces -> {endpoint}  "
            f"(UI: {ui_url}, project: {project}, sample_rate: {sample_rate})",
            flush=True,
        )
    else:
        # Make the silent-failure mode loud and grep-able. Caught the
        # hard way by v0527a-profile-3n where every agent-loop span
        # vanished because phoenix.otel was not in the agent venv.
        msg = (
            f"[AEVOM-TELEMETRY-DISABLED] init_training_telemetry could not "
            f"register a Phoenix exporter (likely `arize-phoenix-otel` is "
            f"not installed in this process's venv). Endpoint was {endpoint}; "
            f"all spans from this process will be no-ops."
        )
        logger.error(msg)
        print(msg, flush=True)


def should_sample_episode(config: TrainingTelemetryConfig, is_eval: bool) -> bool:
    """Decide whether to trace this episode based on sampling config."""
    rate = config.sample_rate_eval if is_eval else config.sample_rate_train
    return random.random() < rate


class EpisodeSpan:
    """Manages a root ``chat_request`` OTel span for one training episode.

    Lifecycle (post-2026-05-27 refactor):
        1. Created at ``seed_session`` time. ``activate()`` is invoked
           immediately so any same-process child spans (e.g. judge spans
           emitted later in the same /verify request) parent correctly,
           and ``traceparent()`` yields a W3C header that the simple_agent
           extracts to make its live agent-loop spans children of this
           ``chat_request`` across the process boundary.
        2. ``execute_tool`` lets ToolRegistry emit live ``tool.{name}``
           spans (``set_skip_tool_spans(False)`` is the training default).
        3. ``finalize_trace()`` called at ``verify`` time -- sets root
           aggregate attributes (`chat.total_iterations`,
           `chat.total_tool_calls`, `chat.episode_outcome`,
           `chat.final_response`) and captures `_tool_span_ids` from the
           live tool spans for reward annotation targeting. The
           synthetic retroactive children that pre-2026-05-27
           ``build_trace()`` created are no longer emitted -- the live
           tree is the source of truth.
        4. ``set_rewards()`` attaches reward signals.
        5. ``end()`` finalises and exports the span.
    """

    def __init__(
        self,
        task_prompt: str,
        episode_id: str,
        wandb_run_id: Optional[str] = None,
        is_eval: bool = False,
        system_prompt: str = "",
        tool_schemas: Optional[List[Dict[str, Any]]] = None,
    ):
        tracer = get_tracer()
        # Root regardless of the active context: seed_session runs inside the
        # resource server's HTTP request, whose span would otherwise parent it.
        self.span: Span = tracer.start_span(
            "chat_request", context=otel_context.Context()
        )

        self.span.set_attribute("chat.user_message", task_prompt)
        self.span.set_attribute("training.episode_id", episode_id)
        self.span.set_attribute("training.is_eval", is_eval)

        # OpenInference: render the episode root as an AGENT span and populate
        # the trace-list Input column from input.value. Additive -- all chat.*
        # / training.* attributes below are preserved for profile_run.py.
        set_span_kind(self.span, "AGENT")
        set_io(self.span, input_value=task_prompt)

        if system_prompt:
            self.span.set_attribute("chat.system_prompt", system_prompt)

        if tool_schemas:
            self.span.set_attribute(
                "chat.tool_schemas", json.dumps(tool_schemas, default=str)
            )
            self.span.set_attribute("chat.tools_available", len(tool_schemas))
            set_llm_tools(self.span, tool_schemas)

        if wandb_run_id:
            self.span.set_attribute("training.wandb_run_id", wandb_run_id)

        slurm_job_id = os.environ.get("SLURM_JOB_ID", "")
        if slurm_job_id:
            self.span.set_attribute("training.slurm_job_id", slurm_job_id)

        project = os.environ.get("PHOENIX_PROJECT_NAME", "")
        if project:
            self.span.set_attribute("training.project", project)

        self._ended = False

        # Populated by finalize_trace() for reward annotation targeting
        self._root_span_id: str = format(self.span.get_span_context().span_id, "016x")
        self._tool_span_ids: List[tuple] = []  # (span_id_hex, tool_name, tc_index)

        # OTel context detach token (returned by context.attach() so the
        # caller can detach() symmetrically when the span ends).
        self._otel_ctx_token: Any = None

    def activate(self) -> None:
        """Attach this span to the current OTel context.

        Two effects:
            (1) Sets the chat span ContextVar (so legacy helpers like
                ``trace_tool_execution`` parent under it).
            (2) Attaches an OTel context whose current span IS this
                root ``chat_request`` -- any in-process span started
                via ``tracer.start_as_current_span(...)`` after this
                point becomes a child of ``chat_request``.

        The matching ``deactivate()`` (called from ``end()``) attempts
        to detach the context using the token returned here. In FastAPI
        deployments ``activate()`` runs in seed_session's asyncio task
        and ``deactivate()`` runs in verify's task -- a different
        ``contextvars`` context -- so ``detach(token)`` raises
        ``ValueError: Token was created in a different Context``.
        OTel's own ``detach()`` swallows that and logs ``Failed to
        detach context``. The fix lives in ``init_training_telemetry``,
        which silences the ``opentelemetry.context`` logger to keep the
        driver log clean (v0527c-tool-headers-fix spammed 3109 stack
        traces from this path). The attach itself works fine.
        """
        set_current_chat_span(self.span)
        ctx = trace.set_span_in_context(self.span)
        self._otel_ctx_token = otel_context.attach(ctx)

    def deactivate(self) -> None:
        """Detach the OTel context and clear the chat-span ContextVar.

        See ``activate()``'s docstring for why ``otel_context.detach``
        often raises ``ValueError`` in cross-task FastAPI deployments
        (and how we keep the noise out of the driver log).
        """
        set_current_chat_span(None)
        if self._otel_ctx_token is not None:
            try:
                otel_context.detach(self._otel_ctx_token)
            except Exception:
                logger.debug("EpisodeSpan: otel_context.detach failed", exc_info=True)
            self._otel_ctx_token = None

    def traceparent(self) -> Optional[str]:
        """Serialise this span's context as a W3C ``traceparent`` header.

        The simple_agent overlay extracts this header from the
        seed_session response body, attaches the resulting OTel context,
        and emits its live agent-loop spans under it. Without this
        propagation, the agent.turn / policy_llm.responses / tool.*
        spans would be orphans (different process from the EpisodeSpan).

        Returns a single string of the form
        ``00-{trace_id_hex}-{span_id_hex}-{flags_hex}`` or ``None`` if
        the span's context is not valid.
        """
        sc = self.span.get_span_context()
        if sc is None or not sc.is_valid:
            return None
        carrier: Dict[str, str] = {}
        ctx = trace.set_span_in_context(self.span)
        TraceContextTextMapPropagator().inject(carrier, context=ctx)
        return carrier.get("traceparent")

    def start_child(
        self,
        name: str,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> Optional[Span]:
        """Start a span explicitly parented under this episode's root span.

        Unlike ``tracer.start_as_current_span``, this does NOT rely on the
        ambient OTel context being the ``chat_request`` span. ``activate()``
        runs in ``seed_session``'s asyncio task, but in-process orchestration
        work we want to instrument (the ``verify`` reward phases, the seed
        preload) runs in a *different* request task whose contextvars context
        no longer carries the root span. So we set the parent explicitly via
        ``trace.set_span_in_context(self.span)`` -- the same mechanism the
        judge already uses with ``parent_span=session.episode_span.span``.

        Returns the live child ``Span`` (or ``None`` when telemetry is a
        no-op). The caller owns ``.end()``. Use this when wrapping a region
        in a ``with`` block would force a large re-indent; otherwise prefer
        ``child_span`` (the context-manager form).
        """
        tracer = get_tracer()
        if tracer is None:
            return None
        parent_ctx = trace.set_span_in_context(self.span)
        child = tracer.start_span(name, context=parent_ctx)
        if attributes:
            for k, v in attributes.items():
                try:
                    child.set_attribute(k, v)
                except Exception:
                    pass
        return child

    @contextmanager
    def child_span(
        self,
        name: str,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> Iterator[Optional[Span]]:
        """Context-manager form of ``start_child``.

        Yields the live child ``Span`` (or ``None`` when telemetry is a
        no-op) so callers can set extra attributes. Telemetry must never
        break the pipeline: any exception inside the ``with`` body is
        recorded on the span and re-raised; the span is always ended.
        """
        child = self.start_child(name, attributes)
        if child is None:
            yield None
            return
        try:
            yield child
        except Exception as exc:
            try:
                child.set_status(Status(StatusCode.ERROR, str(exc)))
            except Exception:
                pass
            raise
        finally:
            child.end()

    # ------------------------------------------------------------------
    # Trace finalisation
    # ------------------------------------------------------------------

    def finalize_trace(
        self,
        response: Any,
        tool_calls: List[Dict[str, Any]],
        final_response: str = "",
    ) -> None:
        """Set root aggregate attributes and ``chat.episode_outcome``.

        The live agent-loop spans (``agent.turn.{N}``,
        ``policy_llm.responses``, ``tool.{name}``) are already in the
        Phoenix DB by the time this runs -- the simple_agent overlay
        and ToolRegistry emit them in real time, parented under this
        ``chat_request`` via W3C traceparent propagation. This method
        therefore only:

            * Counts iterations (from response.output's reasoning items).
            * Counts tool calls (from the session's tool_calls list).
            * Computes ``chat.episode_outcome``.
            * Populates ``chat.final_response`` (falling back to the
              last reasoning block if no message was emitted).
            * Walks ``tool_calls`` to surface ``_tool_span_ids`` for
              reward annotation. The span_id strings are filled in by
              the simple_agent's live tool spans (we re-read them out
              of the Phoenix DB at annotate time); here we just record
              the (tool_name, idx) tuples in order so the annotator can
              match.
        """
        output_items = _get_output_items(response)

        # Count iterations the cheap way: each "reasoning" item marks the
        # start of an iteration, plus one final iteration if a message
        # item ends the loop without a reasoning marker. This is a
        # heuristic for the chat.total_iterations attribute only -- the
        # authoritative per-turn timing data is in the live
        # agent.turn.{N} spans.
        iter_count = 0
        for item in output_items:
            if _get_item_type(item) == "reasoning":
                iter_count += 1
        if iter_count == 0 and output_items:
            iter_count = 1

        total_tool_calls = len(tool_calls)

        self._tool_span_ids = []
        for idx, tc in enumerate(tool_calls):
            tc_name = tc.get("tool_name", "unknown")
            # span_id is forwarded by the resource server's execute_tool
            # (read from api.telemetry._last_tool_span_id after
            # ToolRegistry.execute opens the live tool.{name} span). When
            # present, build_reward_annotations targets that span; when
            # missing (no-op tracer, cache hit, registry error before span
            # creation), it falls back to the root chat_request span via
            # ``first_match_id or root_span_id`` in rewards.py.
            tc_span_id = tc.get("span_id") or ""
            self._tool_span_ids.append((tc_span_id, tc_name, idx))

        self.span.set_attribute("chat.total_iterations", iter_count)
        self.span.set_attribute("chat.total_tool_calls", total_tool_calls)
        self.span.set_attribute("chat.hit_iteration_limit", False)

        has_tools = total_tool_calls > 0
        has_response = bool(final_response)

        if not has_response:
            # Fall back to the last reasoning block when no assistant
            # message was emitted (truncation, length limit, etc.).
            last_reasoning = ""
            for item in output_items:
                if _get_item_type(item) == "reasoning":
                    text = _get_reasoning_text(item)
                    if text:
                        last_reasoning = text
            if last_reasoning:
                final_response = last_reasoning

        self.span.set_attribute("chat.final_response", final_response)
        # OpenInference: populate the trace-list Output column from the root
        # span's output.value (additive; chat.final_response is preserved).
        set_io(self.span, output_value=final_response)

        if has_tools and has_response:
            outcome = "complete"
        elif has_tools:
            outcome = "no_response"
        elif has_response:
            outcome = "no_tools"
        else:
            outcome = "empty"
        self.span.set_attribute("chat.episode_outcome", outcome)

    # Backward-compat alias: pre-2026-05-27 callers used build_trace().
    # Keep the symbol so older deployments (or downstream consumers in
    # other repos) don't crash if they call it during the rollout.
    build_trace = finalize_trace

    # ------------------------------------------------------------------
    # Rewards & attributes
    # ------------------------------------------------------------------

    def set_rewards(
        self,
        total_reward: float,
        reward_breakdown: Dict[str, float],
    ) -> None:
        """Attach reward signals to the span (called at verify time)."""
        self.span.set_attribute("training.reward", total_reward)
        for key, value in reward_breakdown.items():
            self.span.set_attribute(f"training.reward.{key}", value)

    def annotate_rewards(
        self,
        reward_annotations: List[Dict[str, Any]],
        phoenix_url: Optional[str] = None,
    ) -> None:
        """Add reward annotations to specific spans via the Phoenix client.

        Each annotation is submitted individually with its own retry loop.
        This prevents a failure on one span (e.g. the root span that takes
        longer to ingest) from blocking annotations on other spans (e.g.
        tool spans that are already available).

        Root-span annotations (``total_effective_reward``) are submitted
        LAST since the root span is ended last and takes longer to appear
        in Phoenix's SQLite store.
        """
        if not reward_annotations:
            return

        try:
            from phoenix.client import Client
        except ImportError:
            logger.warning(
                "phoenix.client not available -- skipping reward annotations"
            )
            return

        url = phoenix_url or os.environ.get(
            "PHOENIX_CLIENT_URL",
            os.environ.get("PHOENIX_COLLECTOR_ENDPOINT", "").replace(":4317", ":6006")
            or "http://phoenix:6006",
        )

        client = Client(base_url=url)

        # Submit child-span annotations first (tool spans are ingested
        # sooner), then root-span annotations (total_effective_reward).
        child_anns = [
            a for a in reward_annotations if a.get("label") != "total_effective_reward"
        ]
        root_anns = [
            a for a in reward_annotations if a.get("label") == "total_effective_reward"
        ]
        ordered = child_anns + root_anns

        succeeded = 0
        failed = 0
        for ann in ordered:
            if self._submit_one_annotation(client, ann):
                succeeded += 1
            else:
                failed += 1

        if succeeded:
            logger.info(f"Submitted {succeeded} reward annotation(s) to Phoenix")
        if failed:
            logger.warning(f"Failed to submit {failed} reward annotation(s) to Phoenix")

    @staticmethod
    def _submit_one_annotation(
        client: Any,
        ann: Dict[str, Any],
        max_attempts: int = 6,
    ) -> bool:
        """Submit a single annotation with retries and exponential backoff."""
        backoff = [2, 4, 6, 10, 15]
        # Phoenix dedups annotations on (span_id, annotation_name, identifier).
        # Every reward shares annotation_name="training_reward", so without a
        # distinct identifier all rewards on the SAME span (e.g. any_tool_usage
        # + successful_dock on one tool span) overwrite each other and only the
        # last survives. Key the identifier off the reward label (callers may
        # override with a unique per-call identifier, e.g. "any_tool_usage:3")
        # so each reward type coexists on the span.
        identifier = ann.get("identifier") or ann["label"]
        for attempt in range(1, max_attempts + 1):
            try:
                client.spans.add_span_annotation(
                    span_id=ann["span_id"],
                    annotation_name="training_reward",
                    annotator_kind="CODE",
                    label=ann["label"],
                    score=float(ann["score"]),
                    metadata=ann.get("metadata", {}),
                    identifier=identifier,
                    sync=True,
                )
                return True
            except Exception as exc:
                if attempt < max_attempts:
                    delay = backoff[attempt - 1] if attempt - 1 < len(backoff) else 15
                    time.sleep(delay)
                else:
                    logger.warning(
                        f"Annotation '{ann['label']}' on span {ann['span_id'][:8]}... "
                        f"failed after {max_attempts} attempts: {exc}"
                    )
        return False

    def set_attribute(self, key: str, value: Any) -> None:
        """Forward arbitrary attributes to the underlying span."""
        self.span.set_attribute(key, value)

    def end(self, error: Optional[str] = None) -> None:
        """End the span.  Safe to call multiple times."""
        if self._ended:
            return
        self._ended = True
        if error:
            self.span.set_status(Status(StatusCode.ERROR, error))
        else:
            self.span.set_status(Status(StatusCode.OK))
        self.span.end()
        self.deactivate()


# =====================================================================
# Helpers for finalize_trace()
# =====================================================================


def _get_item_type(item: Any) -> str:
    """Get the type string from a response output item (dict or Pydantic model)."""
    if isinstance(item, dict):
        return item.get("type", "")
    return getattr(item, "type", "")


def _get_reasoning_text(item: Any) -> str:
    """Extract reasoning text from a reasoning output item."""
    texts: list[str] = []

    summary = (
        item.get("summary", [])
        if isinstance(item, dict)
        else getattr(item, "summary", [])
    )
    if summary:
        for s in summary:
            text = s.get("text", "") if isinstance(s, dict) else getattr(s, "text", "")
            if text:
                texts.append(text)
    if texts:
        return "\n".join(texts)

    content = (
        item.get("content", [])
        if isinstance(item, dict)
        else getattr(item, "content", None)
    )
    if content and isinstance(content, list):
        for c in content:
            text = c.get("text", "") if isinstance(c, dict) else getattr(c, "text", "")
            if text:
                texts.append(text)

    return "\n".join(texts)


def _get_output_items(response: Any) -> list:
    """Safely extract the output list from a NeMo response."""
    if response is None:
        return []
    if isinstance(response, dict):
        return response.get("output", [])
    out = getattr(response, "output", None)
    return list(out) if out else []


