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

"""Repository pattern for database access abstraction.

This module provides an abstract interface for database operations,
making it easy to swap implementations for different environments
(local PostgreSQL, cloud providers, etc.).
"""

import uuid
from abc import ABC, abstractmethod
from typing import Any, Optional

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from .models import Job, JobStatus


class AbstractJobRepository(ABC):
    """Abstract interface for job repository operations.
    
    Implement this interface for different database backends.
    """
    
    @abstractmethod
    async def create(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        session_id: Optional[str] = None,
        request_id: Optional[str] = None,
    ) -> Job:
        """Create a new job record."""
        pass

    @abstractmethod
    async def get(self, job_id: uuid.UUID) -> Optional[Job]:
        """Get a job by ID."""
        pass

    @abstractmethod
    async def update_status(
        self,
        job_id: uuid.UUID,
        status: JobStatus,
        result: Optional[dict[str, Any]] = None,
        error: Optional[str] = None,
        upstream_job_id: Optional[str] = None,
        poll_after: Optional[Any] = None,
    ) -> Optional[Job]:
        """Update job status and optionally set result, error, or upstream tracking fields."""
        pass
    
    @abstractmethod
    async def list_by_status(
        self,
        status: JobStatus,
        limit: int = 100,
    ) -> list[Job]:
        """List jobs by status."""
        pass


class JobRepository(AbstractJobRepository):
    """SQLAlchemy implementation of job repository.
    
    This implementation uses PostgreSQL via SQLAlchemy async.
    For cloud deployment, you can create a new implementation
    that uses the cloud provider's SDK while maintaining
    the same interface.
    """
    
    def __init__(self, session: AsyncSession):
        """Initialize with database session.
        
        Args:
            session: SQLAlchemy async session.
        """
        self._session = session
    
    async def create(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        session_id: Optional[str] = None,
        request_id: Optional[str] = None,
    ) -> Job:
        """Create a new job record.

        Args:
            tool_name: Name of the tool to execute.
            arguments: Arguments to pass to the tool.
            session_id: Session that started the job, when known.
            request_id: Request that started the job, when known.

        Returns:
            Created Job instance.
        """
        job = Job(
            tool_name=tool_name,
            arguments=arguments,
            status=JobStatus.PENDING,
            session_id=session_id,
            request_id=request_id,
        )
        self._session.add(job)
        await self._session.flush()
        await self._session.refresh(job)
        return job
    
    async def get(self, job_id: uuid.UUID) -> Optional[Job]:
        """Get a job by ID.
        
        Args:
            job_id: UUID of the job.
            
        Returns:
            Job instance or None if not found.
        """
        result = await self._session.execute(
            select(Job).where(Job.id == job_id)
        )
        return result.scalar_one_or_none()
    
    async def update_status(
        self,
        job_id: uuid.UUID,
        status: JobStatus,
        result: Optional[dict[str, Any]] = None,
        error: Optional[str] = None,
        upstream_job_id: Optional[str] = None,
        poll_after: Optional[Any] = None,
    ) -> Optional[Job]:
        """Update job status and optionally set result, error, or upstream tracking fields.

        Args:
            job_id: UUID of the job.
            status: New status.
            result: Result data (for completed jobs).
            error: Error message (for failed jobs).
            upstream_job_id: Provider-side job id (for queued-upstream jobs).
            poll_after: Earliest time it's worth re-checking the upstream job.

        Returns:
            Updated Job instance or None if not found.
        """
        values: dict[str, Any] = {"status": status}
        if result is not None:
            values["result"] = result
        if error is not None:
            values["error"] = error
        if upstream_job_id is not None:
            values["upstream_job_id"] = upstream_job_id
        if poll_after is not None:
            values["poll_after"] = poll_after

        await self._session.execute(
            update(Job).where(Job.id == job_id).values(**values)
        )
        await self._session.flush()
        return await self.get(job_id)
    
    async def list_by_status(
        self,
        status: JobStatus,
        limit: int = 100,
    ) -> list[Job]:
        """List jobs by status.
        
        Args:
            status: Status to filter by.
            limit: Maximum number of jobs to return.
            
        Returns:
            List of Job instances.
        """
        result = await self._session.execute(
            select(Job)
            .where(Job.status == status)
            .order_by(Job.created_at.desc())
            .limit(limit)
        )
        return list(result.scalars().all())



