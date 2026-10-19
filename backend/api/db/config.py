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

"""Database configuration with environment-based connection settings.

This module provides an abstracted database interface that can be easily
swapped between local development (PostgreSQL container) and cloud providers.
"""

import os
from functools import lru_cache
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)


def get_database_url() -> str:
    """Get database URL from environment.
    
    For local development: postgresql+asyncpg://aevom:password@postgres:5432/aevom
    For cloud deployment: Set DATABASE_URL environment variable to cloud provider's URL
    
    Returns:
        Database connection URL for SQLAlchemy async engine.
    """
    return os.environ.get(
        "DATABASE_URL",
        "postgresql+asyncpg://aevom:aevom_dev_password@postgres:5432/aevom"
    )


@lru_cache()
def get_async_engine() -> AsyncEngine:
    """Create and cache the async database engine.
    
    Uses LRU cache to ensure single engine instance across the application.
    """
    return create_async_engine(
        get_database_url(),
        echo=os.environ.get("DB_ECHO", "false").lower() == "true",
        pool_pre_ping=True,  # Verify connections before use
        pool_size=5,
        max_overflow=10,
    )


def get_session_factory() -> async_sessionmaker[AsyncSession]:
    """Get async session factory."""
    return async_sessionmaker(
        bind=get_async_engine(),
        class_=AsyncSession,
        expire_on_commit=False,
    )


async def get_async_session() -> AsyncGenerator[AsyncSession, None]:
    """Dependency for FastAPI to get database sessions.
    
    Usage:
        @app.get("/example")
        async def example(session: AsyncSession = Depends(get_async_session)):
            ...
    """
    session_factory = get_session_factory()
    async with session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def init_db() -> None:
    """Initialize database tables.
    
    Creates any missing tables and clears ephemeral data (jobs, data_artifacts)
    while preserving persistent data (training_queries).
    
    Note: For schema changes, use Alembic migrations.
    """
    import logging
    from sqlalchemy import text
    from .models import Base
    
    logger = logging.getLogger(__name__)
    
    engine = get_async_engine()
    async with engine.begin() as conn:
        # Drop ephemeral tables so create_all rebuilds them with the
        # latest column definitions.  This is safe because their data is
        # discarded on every restart anyway.
        #
        # Conversations, turns, messages and findings share the artifacts'
        # lifecycle by necessity, not convenience: they name artifact
        # references, so keeping a transcript whose refs have been dropped
        # would leave the model reading about data it can no longer load.
        # Child tables first -- foreign keys point the other way.
        from .models import Conversation, DataArtifact, Finding, Job, Message, Turn
        logger.info("Dropping ephemeral tables...")
        for table in (
            Job.__table__,
            DataArtifact.__table__,
            Message.__table__,
            Turn.__table__,
            Finding.__table__,
            Conversation.__table__,
        ):
            await conn.run_sync(lambda sync_conn, t=table: t.drop(sync_conn, checkfirst=True))

        logger.info("Creating tables (if not exist)...")
        await conn.run_sync(Base.metadata.create_all)

        # Fuzzy session-history search. Also carried by migration 0006, but
        # this path recreates the findings table on every restart, so the
        # index has to be recreated with it. Optional by design: without the
        # extension, search matches substrings instead of ranking by
        # similarity.
        if conn.dialect.name == "postgresql":
            try:
                await conn.execute(text("CREATE EXTENSION IF NOT EXISTS pg_trgm"))
                await conn.execute(
                    text(
                        "CREATE INDEX IF NOT EXISTS ix_findings_summary_trgm "
                        "ON findings USING gin (summary gin_trgm_ops)"
                    )
                )
            except Exception as exc:
                logger.warning(
                    f"Could not enable pg_trgm ({exc}); session-history search "
                    f"will match substrings only."
                )

    logger.info("Database initialization complete")


async def close_db() -> None:
    """Close database connections.
    
    Call this on application shutdown.
    """
    engine = get_async_engine()
    await engine.dispose()
    get_async_engine.cache_clear()



