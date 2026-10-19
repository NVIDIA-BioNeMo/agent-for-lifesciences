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

"""Job manager for async tool execution with database-backed tracking.

This module handles the lifecycle of async tool jobs:
1. Create job record when tool is invoked
2. Execute tool (potentially in background)
3. Update status as execution progresses
4. Store results for retrieval via polling

P0 architecture-contracts update (see internal_docs/docs/0001-architecture-boundaries.md):
tools opted into job tracking (``tool.is_async == True``) can now report
upstream-queued state (an NVCF 202 response) back into the ``Job`` row via a
``JobProgressHandle`` made available on the tool for the duration of
``start_job``'s call to ``tool.execute()``. This does not implement full
poll-and-resume -- it makes the queued case observable and recorded instead
of only surfacing as an error message.
"""

import uuid
from datetime import datetime
from typing import Any, Optional

from sqlalchemy.ext.asyncio import AsyncSession

from api.db.models import Job, JobStatus
from api.db.repository import JobRepository
from api.tools.base import BaseTool, ToolResult, ContentType, _job_handle_ctx


class JobProgressHandle:
    """Lets a tool report progress into its tracked ``Job`` while it runs.

    Bound to ``api.tools.base._job_handle_ctx`` for the duration of
    ``JobManager.start_job``'s call to ``tool.execute()``, so a tool such as
    ``NvidiaApiTool`` can call ``self._job_handle.record_upstream_pending(...)``
    without importing ``JobManager`` directly (avoids a base/jobs import
    cycle) and without every tool needing to know a ``JobManager`` exists.
    """

    def __init__(self, manager: "JobManager", job_id: uuid.UUID):
        self._manager = manager
        self._job_id = job_id

    async def record_upstream_pending(
        self, upstream_job_id: str, poll_after: Optional[datetime] = None
    ) -> None:
        """Record that the upstream provider queued this job instead of
        returning a result within the poll window."""
        await self._manager.record_upstream_pending(
            self._job_id, upstream_job_id, poll_after=poll_after
        )


class JobManager:
    """Manages async tool job lifecycle.

    The JobManager coordinates between the tool registry and database
    to track async tool executions.

    Example:
        async with get_async_session() as session:
            manager = JobManager(session)

            # Start async job
            job = await manager.start_job(rdkit_tool, {"smiles": "CCO"})

            # Later, check status
            status = await manager.get_job_status(job.id)

            # When complete, get result
            result = await manager.get_job_result(job.id)
    """

    def __init__(self, session: AsyncSession):
        """Initialize with database session.

        Args:
            session: SQLAlchemy async session.
        """
        self._session = session
        self._repo = JobRepository(session)

    async def start_job(
        self,
        tool: BaseTool,
        arguments: dict[str, Any],
        session_id: Optional[str] = None,
        request_id: Optional[str] = None,
    ) -> Job:
        """Start an async tool job.

        Creates a job record and begins execution.
        For truly long-running tools, this would spawn a background task.
        For now, it executes synchronously but tracks via the database.

        Args:
            tool: Tool to execute.
            arguments: Arguments for the tool.
            session_id: Session that started the job, when known.
            request_id: Request that started the job, when known.

        Returns:
            Created Job instance.
        """
        # Create pending job
        job = await self._repo.create(
            tool_name=tool.name,
            arguments=arguments,
            session_id=session_id,
            request_id=request_id,
        )

        # Update to running
        await self._repo.update_status(job.id, JobStatus.RUNNING)

        # Make a progress handle available to the tool for the duration of
        # execute(), so it can report upstream-queued state (or, in future,
        # other progress) back into this job row without a direct dependency
        # on JobManager. Token-based reset mirrors the storage/session_id/
        # request_id ContextVar pattern in api.tools.base.
        handle_token = _job_handle_ctx.set(JobProgressHandle(self, job.id))
        try:
            # Execute the tool
            result = await tool.execute(arguments)

            # Convert result to storable format
            result_data = self._result_to_dict(result)

            # Update to completed with result
            job = await self._repo.update_status(
                job.id,
                JobStatus.COMPLETED,
                result=result_data,
            )

        except Exception as e:
            # Update to failed with error
            job = await self._repo.update_status(
                job.id,
                JobStatus.FAILED,
                error=str(e),
            )
        finally:
            _job_handle_ctx.reset(handle_token)

        return job

    async def record_upstream_pending(
        self,
        job_id: uuid.UUID,
        upstream_job_id: str,
        poll_after: Optional[datetime] = None,
    ) -> Optional[Job]:
        """Mark a job as queued upstream (e.g. an NVCF 202 response).

        Does not attempt to resume the job -- see module docstring. This only
        makes the queued case observable in ``jobs.status``/``upstream_job_id``
        instead of only surfacing as an error message to the model.
        """
        return await self._repo.update_status(
            job_id,
            JobStatus.QUEUED_UPSTREAM,
            upstream_job_id=upstream_job_id,
            poll_after=poll_after,
        )

    async def execute_tool_with_tracking(
        self,
        tool: BaseTool,
        arguments: dict[str, Any],
        session_id: Optional[str] = None,
        request_id: Optional[str] = None,
    ) -> tuple[ToolResult, Optional[str], Optional[str]]:
        """Execute a tool with optional job tracking.

        If the tool is async (tool.is_async == True), creates a job
        and returns immediately with the job ID.
        Otherwise, executes synchronously.

        Args:
            tool: Tool to execute.
            arguments: Arguments for the tool.
            session_id: Session that started the job, when known.
            request_id: Request that started the job, when known.

        Returns:
            Tuple of (ToolResult, job_id or None, job_status or None). The
            status accompanies the id because an id alone only says a job
            existed -- what an eval or a training reward asserts on is the
            state it reached (ADR 0001).
        """
        if tool.is_async:
            # Create job and execute with tracking
            job = await self.start_job(
                tool, arguments, session_id=session_id, request_id=request_id
            )
            job_id, job_status = str(job.id), job.status.value

            if job.status == JobStatus.COMPLETED:
                # Job completed quickly, return result directly
                result = self._dict_to_result(job.result)
                return result, job_id, job_status
            elif job.status == JobStatus.FAILED:
                return ToolResult.error(job.error or "Job failed"), job_id, job_status
            elif job.status == JobStatus.QUEUED_UPSTREAM:
                return ToolResult.error(
                    job.error
                    or f"Job {job.id} was queued by the upstream provider "
                    f"(upstream id: {job.upstream_job_id}) and cannot yet be "
                    "resumed. Retry with a smaller or simpler request."
                ), job_id, job_status
            else:
                # Job still running (for future async execution)
                return ToolResult.text(
                    f"Job started with ID: {job.id}. Poll /api/jobs/{job.id} for status."
                ), job_id, job_status
        else:
            # Synchronous execution without job tracking
            result = await tool.execute(arguments)
            return result, None, None

    async def get_job(self, job_id: uuid.UUID) -> Optional[Job]:
        """Get a job by ID.

        Args:
            job_id: UUID of the job.

        Returns:
            Job instance or None.
        """
        return await self._repo.get(job_id)

    async def get_job_status(
        self, job_id: uuid.UUID, *, session_id: Optional[str] = None
    ) -> Optional[dict[str, Any]]:
        """Get job status and result if complete.

        Args:
            job_id: UUID of the job.
            session_id: Only return the job if this session started it. A job
                another session owns is reported as missing.

        Returns:
            Dict with status info or None if job not found.
        """
        job = await self._repo.get(job_id)
        if job is None:
            return None
        if session_id is not None and job.session_id != session_id:
            return None

        response = {
            "id": str(job.id),
            "status": job.status.value,
            "tool_name": job.tool_name,
        }

        if job.status == JobStatus.COMPLETED and job.result:
            response["result"] = job.result
        elif job.status == JobStatus.FAILED and job.error:
            response["error"] = job.error
        elif job.status == JobStatus.QUEUED_UPSTREAM:
            response["upstream_job_id"] = job.upstream_job_id

        return response

    async def get_job_result(self, job_id: uuid.UUID) -> Optional[ToolResult]:
        """Get the result of a completed job.

        Args:
            job_id: UUID of the job.

        Returns:
            ToolResult or None if job not found or not complete.
        """
        job = await self._repo.get(job_id)
        if job is None:
            return None

        if job.status == JobStatus.COMPLETED and job.result:
            return self._dict_to_result(job.result)
        elif job.status == JobStatus.FAILED:
            return ToolResult.error(job.error or "Job failed")
        else:
            return None

    def _result_to_dict(self, result: ToolResult) -> dict[str, Any]:
        """Convert ToolResult to dictionary for storage."""
        return {
            "content_type": result.content_type.value,
            "data": result.data,
            "metadata": result.metadata,
        }

    def _dict_to_result(self, data: dict[str, Any]) -> ToolResult:
        """Convert stored dictionary back to ToolResult."""
        return ToolResult(
            content_type=ContentType(data["content_type"]),
            data=data["data"],
            metadata=data.get("metadata", {}),
        )
