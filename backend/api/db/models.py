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

"""SQLAlchemy models for Aevom database."""

import enum
import uuid
from datetime import datetime
from typing import Any, Optional

from sqlalchemy import (
    DateTime,
    Enum,
    ForeignKey,
    Integer,
    LargeBinary,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """Base class for all SQLAlchemy models.

    ``eager_defaults=True`` is load-bearing, not tuning. Columns declared with
    a SQL-side ``onupdate`` (``onupdate=func.now()`` on ``Job``,
    ``Conversation`` and ``Candidate``) hold a value only the database
    computes, so after an UPDATE flush SQLAlchemy expires the attribute. Any
    subsequent read -- ``to_dict()`` reaching for ``updated_at`` is the one
    that bit us -- then triggers a lazy refresh, which under an async session
    is IO from a sync context and raises ``MissingGreenlet``.

    SQLAlchemy 2.0 defaults to ``"auto"``, which fetches via RETURNING on
    INSERT only. That is exactly why the bug was invisible: the *first* write
    of a row succeeded and the *second* failed. Setting ``True`` extends
    RETURNING to UPDATE.

    Declared here rather than on the three affected models so a model added
    later cannot reintroduce it. It costs nothing for models without
    server-generated columns: with nothing to fetch, no RETURNING is emitted.
    """

    __mapper_args__ = {"eager_defaults": True}


class JobStatus(enum.Enum):
    """Status of an async tool execution job."""
    PENDING = "pending"
    RUNNING = "running"
    # The upstream provider (e.g. an NVCF-hosted NIM) queued the work instead
    # of returning a result within the poll window. Added for the P0 durable
    # job semantics work (ADR 0001) -- see `NvidiaApiTool._call_nvidia_api`.
    # Aevom does not yet resume these jobs; this status makes the queued case
    # observable and recorded instead of silently surfacing only as an error.
    QUEUED_UPSTREAM = "queued_upstream"
    COMPLETED = "completed"
    FAILED = "failed"


class Job(Base):
    """Model for tracking async tool execution jobs.

    Attributes:
        id: Unique job identifier (UUID).
        tool_name: Name of the tool being executed.
        status: Current job status.
        arguments: JSON arguments passed to the tool.
        result: JSON result from tool execution (null until completed).
        error: Error message if job failed.
        created_at: Timestamp when job was created.
        updated_at: Timestamp of last status update.
        session_id: Session that started the job, when known. Optional.
        request_id: Request that started the job, when known. Optional.
        upstream_job_id: The provider-side job/request id, when the job was
            queued upstream (e.g. an NVCF 202 response). Optional.
        poll_after: Earliest time it is worth checking the upstream job again,
            when known. Optional; not yet consumed by a poller (P0 scope).
    """
    __tablename__ = "jobs"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    tool_name: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    status: Mapped[JobStatus] = mapped_column(
        Enum(JobStatus),
        nullable=False,
        default=JobStatus.PENDING,
        index=True,
    )
    arguments: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)
    result: Mapped[Optional[dict[str, Any]]] = mapped_column(JSONB, nullable=True)
    error: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )
    session_id: Mapped[Optional[str]] = mapped_column(
        String(64), nullable=True, index=True,
    )
    request_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    upstream_job_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    poll_after: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True,
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert job to dictionary for JSON serialization."""
        return {
            "id": str(self.id),
            "tool_name": self.tool_name,
            "status": self.status.value,
            "arguments": self.arguments,
            "result": self.result,
            "error": self.error,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "session_id": self.session_id,
            "request_id": self.request_id,
            "upstream_job_id": self.upstream_job_id,
            "poll_after": self.poll_after.isoformat() if self.poll_after else None,
        }


class DataArtifact(Base):
    """Model for storing tool data artifacts.
    
    This enables tools to store large data (e.g., SDF files, PDB structures)
    and pass references to other tools instead of raw data through the LLM.
    
    Attributes:
        id: Unique reference identifier (e.g., "sdf_abc123", "pdb_xyz789").
        data_type: Type of data stored (e.g., "sdf", "pdb", "mol2").
        content: Binary content of the artifact.
        description: Short LLM-generated description of the artifact.
        metadata_json: Optional JSON metadata about the artifact.
        created_at: Timestamp when artifact was created.
        expires_at: Optional expiration timestamp for TTL.
        provenance_json: Structured lineage -- producing tool/model/version,
            input artifact refs, and parameters. Optional; see ADR 0001 and
            `api.storage.envelope.ArtifactProvenance`. Populated (if at all)
            from `metadata["provenance"]` at store time.
        units: Structured unit/convention metadata for the artifact's values
            (e.g. binding score units, coordinate frame). Optional.
        uncertainty: Structured confidence/uncertainty metadata (e.g. model
            confidence, error bars). Optional.
        validation_json: Structured validation status (e.g. which gates or
            checks passed) for the artifact. Optional.
    """
    __tablename__ = "data_artifacts"

    id: Mapped[str] = mapped_column(
        String(64),
        primary_key=True,
    )
    data_type: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        index=True,
    )
    content: Mapped[bytes] = mapped_column(
        LargeBinary,
        nullable=False,
    )
    description: Mapped[Optional[str]] = mapped_column(
        String(256),
        nullable=True,
    )
    session_id: Mapped[Optional[str]] = mapped_column(
        String(64),
        nullable=True,
        index=True,
    )
    metadata_json: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
    expires_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    provenance_json: Mapped[Optional[dict[str, Any]]] = mapped_column(
        JSONB,
        nullable=True,
    )
    units: Mapped[Optional[dict[str, Any]]] = mapped_column(
        JSONB,
        nullable=True,
    )
    uncertainty: Mapped[Optional[dict[str, Any]]] = mapped_column(
        JSONB,
        nullable=True,
    )
    validation_json: Mapped[Optional[dict[str, Any]]] = mapped_column(
        JSONB,
        nullable=True,
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert artifact to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "data_type": self.data_type,
            "description": self.description,
            "metadata": self.metadata_json,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "provenance": self.provenance_json,
            "units": self.units,
            "uncertainty": self.uncertainty,
            "validation": self.validation_json,
        }


class Conversation(Base):
    """A transcript owned by the backend rather than by the browser.

    Before this table the conversation record and the model's context were the
    same object, held by the frontend, which meant the record could only ever
    be as long as the context and any attempt to bound one destroyed the other.
    Splitting them is what makes budgeted context assembly possible.

    A session may own several conversations. Switching models starts a new one
    against the same ``session_id`` so that artifacts and findings -- which are
    chemistry, and model-independent -- carry across while the transcript,
    whose templates and tokenization are not, starts fresh.

    Attributes:
        id: Conversation identifier.
        session_id: Client session this conversation belongs to.
        model_id: Model the conversation was held with.
        created_at: When the conversation started.
        updated_at: When a message was last appended.
    """

    __tablename__ = "conversations"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    session_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    model_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert conversation to dictionary for JSON serialization."""
        return {
            "id": str(self.id),
            "session_id": self.session_id,
            "model_id": self.model_id,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


class Turn(Base):
    """One user message and everything the agent did in response.

    The unit of compaction and of the session log's ``T{turn}.{iteration}``
    numbering. A turn spans many messages: the user's, then an assistant /
    tool pair per agent iteration, then the final assistant answer.

    Attributes:
        id: Turn identifier.
        conversation_id: Conversation this turn belongs to.
        seq: 1-based position within the conversation.
        created_at: When the turn opened.
    """

    __tablename__ = "turns"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    conversation_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("conversations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    seq: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert turn to dictionary for JSON serialization."""
        return {
            "id": str(self.id),
            "conversation_id": str(self.conversation_id),
            "seq": self.seq,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class Message(Base):
    """One message in a turn, stored with everything the model produced.

    The frontend used to send back only ``role`` and ``content``, so tool
    calls, tool results, and reasoning were discarded between user turns and
    the agent restarted each turn blind to its own trajectory. Every one of
    those fields has a column here.

    Attributes:
        id: Message identifier.
        turn_id: Turn this message belongs to.
        seq: 0-based position within the turn.
        role: One of user, assistant, tool, system.
        content: Message text.
        reasoning: The model's own chain of thought, when it emits one.
        tool_calls: OpenAI-format tool calls requested by an assistant message.
        tool_call_id: The call a tool message answers.
        name: Tool name, for tool messages.
        extra: Provider-specific fields, e.g. Gemini 3 thought signatures.
        created_at: When the message was recorded.
    """

    __tablename__ = "messages"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    turn_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("turns.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    seq: Mapped[int] = mapped_column(Integer, nullable=False)
    role: Mapped[str] = mapped_column(String(16), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False, default="")
    reasoning: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    tool_calls: Mapped[Optional[list[Any]]] = mapped_column(JSONB, nullable=True)
    tool_call_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    name: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    extra: Mapped[Optional[dict[str, Any]]] = mapped_column(JSONB, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert message to dictionary for JSON serialization."""
        return {
            "id": str(self.id),
            "seq": self.seq,
            "role": self.role,
            "content": self.content,
            "reasoning": self.reasoning,
            "tool_calls": self.tool_calls,
            "tool_call_id": self.tool_call_id,
            "name": self.name,
            "extra": self.extra,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class Finding(Base):
    """One thing the agent learned by running a tool.

    The centerpiece of the memory design. Most memory systems must summarize
    prose and accept the loss; this domain does not have to. A dock produces a
    number, a fold produces pLDDT and ipTM, and both produce references into a
    store that already exists. Those are facts, so they can be recorded once
    and re-rendered deterministically with no model call, no hallucination,
    and no per-request cost.

    ``scalars`` is the load-bearing column: ``affinity_kcal_mol``, ``plddt``,
    ``iptm``, ``logp``, ``tpsa`` and friends, extracted from metadata the Vina
    and OpenFold3 tools already produce. It is what makes "re-dock to recover
    the -9.5 we measured twenty iterations ago" unnecessary.

    Attributes:
        id: Finding identifier.
        session_id: Session the finding belongs to. Indexed because findings
            outlive any single conversation -- chemistry survives a model switch.
        conversation_id: Conversation during which the tool ran, if known.
        turn_seq: 1-based user turn, for T{turn}.{iteration} rendering.
        iteration: Agent iteration within that turn.
        tool_name: Tool that produced the finding.
        arguments: Arguments the tool was called with.
        input_refs: Artifact references consumed.
        output_refs: Artifact references produced.
        scalars: Extracted numeric results, the part that must never be lost.
        summary: One-line human/LLM-readable rendering.
        success: Whether the tool call succeeded.
        created_at: When the tool ran.
    """

    __tablename__ = "findings"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    session_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    conversation_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("conversations.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    turn_seq: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    iteration: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    tool_name: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    arguments: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)
    input_refs: Mapped[list[str]] = mapped_column(JSONB, nullable=False, default=list)
    output_refs: Mapped[list[str]] = mapped_column(JSONB, nullable=False, default=list)
    scalars: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)
    summary: Mapped[str] = mapped_column(Text, nullable=False, default="")
    success: Mapped[bool] = mapped_column(nullable=False, default=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert finding to dictionary for JSON serialization."""
        return {
            "id": str(self.id),
            "session_id": self.session_id,
            "turn_seq": self.turn_seq,
            "iteration": self.iteration,
            "tool_name": self.tool_name,
            "arguments": self.arguments,
            "input_refs": self.input_refs,
            "output_refs": self.output_refs,
            "scalars": self.scalars,
            "summary": self.summary,
            "success": self.success,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class Candidate(Base):
    """One molecule the session has considered, and everything known about it.

    A sibling of :class:`Finding` rather than a special case of it, because the
    grain differs: a finding is one tool execution, while a candidate accumulates
    across many. A molecule gets its descriptors from ``molecule_gates``, its
    ADMET panel later, an affinity later still, and all three describe the same
    row. Appending a finding per measurement and merging on read would work, but
    every reader would then have to redo the merge, and the merge is the part
    that matters.

    The design problem this exists to solve is documented in SMDD-Bench's
    appendix: agents routinely enumerate a molecule that satisfies every
    constraint, keep working, and then submit a worse one -- not because they
    misjudged, but because the good one scrolled out of the context window.
    Keeping the comparison outside the transcript is what makes it survivable.

    Attributes:
        id: Candidate identifier.
        session_id: Session the candidate belongs to.
        smiles: Canonical SMILES, the identity of the row. Canonicalized before
            writing so the same molecule written two ways is one candidate.
        origin: Where the molecule came from -- a tool name, a generator, the
            user. Provenance is what lets a later reader trust the numbers.
        properties: Computed descriptors, keyed as ``molecule_gates`` reports
            them (mw, logp, tpsa, sa_score, qed, ...).
        scores: Predicted or measured values worth ranking on -- docking
            affinity, pIC50, ADMET endpoints.
        gates: Per-gate pass/fail with the bound each was judged against.
        failures: Human-readable constraint violations, empty when passing.
        passed: Whether the molecule satisfied every gate it was judged against.
            Null means it has not been judged yet, which is different from
            failing.
        notes: Anything the agent wants to remember that is not a number.
        created_at: When the molecule was first seen.
        updated_at: When anything about it last changed.
    """

    __tablename__ = "candidates"
    __table_args__ = (
        UniqueConstraint("session_id", "smiles", name="uq_candidates_session_smiles"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    session_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    smiles: Mapped[str] = mapped_column(Text, nullable=False)
    origin: Mapped[str] = mapped_column(String(128), nullable=False, default="")
    properties: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, default=dict
    )
    scores: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)
    gates: Mapped[list[dict[str, Any]]] = mapped_column(
        JSONB, nullable=False, default=list
    )
    failures: Mapped[list[str]] = mapped_column(JSONB, nullable=False, default=list)
    passed: Mapped[Optional[bool]] = mapped_column(nullable=True)
    notes: Mapped[str] = mapped_column(Text, nullable=False, default="")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert candidate to dictionary for JSON serialization."""
        return {
            "id": str(self.id),
            "session_id": self.session_id,
            "smiles": self.smiles,
            "origin": self.origin,
            "properties": self.properties,
            "scores": self.scores,
            "gates": self.gates,
            "failures": self.failures,
            "passed": self.passed,
            "notes": self.notes,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


class TrainingQuery(Base):
    """Model for storing test queries for DPO training data collection.
    
    Each query can have multiple responses across different prompt modes.
    The pair collection status is computed dynamically from Phoenix telemetry.
    
    Attributes:
        id: Unique query identifier (e.g., "q_001").
        query_text: The actual user query text.
        expected_tool: Tool(s) the model should use (for oracle mode).
        category: Optional category for organizing queries.
        oracle_prompt: Per-query oracle prompt for training (overrides default).
        oracle_prompt_updated_at: Timestamp when oracle prompt was last updated.
        created_at: Timestamp when query was created.
        is_active: Whether the query is active for collection.
    """
    __tablename__ = "training_queries"
    
    id: Mapped[str] = mapped_column(
        String(64),
        primary_key=True,
    )
    query_text: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )
    expected_tool: Mapped[Optional[str]] = mapped_column(
        String(256),
        nullable=True,
    )
    category: Mapped[Optional[str]] = mapped_column(
        String(64),
        nullable=True,
        index=True,
    )
    oracle_prompt: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )
    oracle_prompt_updated_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
    is_active: Mapped[bool] = mapped_column(
        default=True,
        nullable=False,
    )
    
    def to_dict(self) -> dict[str, Any]:
        """Convert query to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "query_text": self.query_text,
            "expected_tool": self.expected_tool,
            "category": self.category,
            "oracle_prompt": self.oracle_prompt,
            "oracle_prompt_updated_at": self.oracle_prompt_updated_at.isoformat() if self.oracle_prompt_updated_at else None,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "is_active": self.is_active,
        }
