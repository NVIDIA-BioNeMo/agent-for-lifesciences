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

"""Redis connection pool for externalized state.

This module provides a shared async Redis connection pool used by all
state modules (viewer_state, pending_ops, selections, sandbox sessions,
MCP sessions) to store mutable state outside of process memory.

The connection pool is a stateless, thread-safe resource -- it is safe
to hold as a module-level singleton because it contains no application
state, only network connections.

Usage:
    from api.redis import get_redis

    redis = get_redis()
    await redis.set("key", "value", ex=300)
    value = await redis.get("key")
"""

import logging
import os
from typing import Optional

import redis.asyncio as aioredis

logger = logging.getLogger(__name__)

# Module-level connection pool (stateless -- safe as singleton)
_redis_pool: Optional[aioredis.Redis] = None


def get_redis() -> aioredis.Redis:
    """Get the shared async Redis client.

    Creates the connection pool lazily on first call. The pool is
    reused across all callers within the process.

    Returns:
        Async Redis client backed by a connection pool.

    Raises:
        RuntimeError: If REDIS_URL is not configured.
    """
    global _redis_pool

    if _redis_pool is not None:
        return _redis_pool

    redis_url = os.environ.get("REDIS_URL")
    if not redis_url:
        raise RuntimeError(
            "REDIS_URL environment variable is not set. "
            "Redis is required for state management. "
            "Set REDIS_URL=redis://redis:6379/0 in your environment."
        )

    _redis_pool = aioredis.from_url(
        redis_url,
        decode_responses=True,
        max_connections=20,
    )

    logger.info(f"Redis connection pool created: {redis_url}")
    return _redis_pool


async def close_redis() -> None:
    """Close the Redis connection pool.

    Should be called during application shutdown.
    """
    global _redis_pool

    if _redis_pool is not None:
        await _redis_pool.aclose()
        _redis_pool = None
        logger.info("Redis connection pool closed")


async def redis_health_check() -> bool:
    """Check if Redis is reachable.

    Returns:
        True if Redis responds to PING, False otherwise.
    """
    try:
        r = get_redis()
        return await r.ping()
    except Exception as e:
        logger.warning(f"Redis health check failed: {e}")
        return False
