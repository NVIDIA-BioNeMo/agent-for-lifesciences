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

"""Pending operations registry using Redis pub/sub.

This module provides a registry for tracking operations that require
acknowledgment from the frontend before proceeding. This is used to
synchronize state between tool calls that depend on frontend processing.

Primary use case: structure_query emits a tool_event via SSE, and the
frontend must POST selection data back before python_execute can use it.

State is stored in Redis, and acknowledgment is communicated via Redis
pub/sub channels. This allows the SSE stream (on any pod) to wait for
a POST /api/selection (which may arrive on a different pod).
"""

import asyncio
import logging
import time
from typing import Optional

logger = logging.getLogger(__name__)

# Redis key TTL for pending ops markers (5 minutes)
PENDING_OP_TTL = 300


class RedisPendingOps:
    """Per-request pending operations tracker backed by Redis pub/sub.
    
    Each chat request creates its own instance. Pending operations are
    registered with a query_id, and acknowledgments arrive via Redis
    pub/sub channels, allowing cross-pod communication.
    
    Example:
        from api.redis import get_redis
        
        redis = get_redis()
        ops = RedisPendingOps(redis, query_id="req_abc123")
        
        # When emitting structure_query
        await ops.register("call_123", "selection")
        yield json.dumps({"tool_event": ...})
        
        # After all tools in iteration complete
        unacked = await ops.await_all(timeout=5.0)
        
        # In /api/selection endpoint (possibly on another pod)
        await acknowledge_pending_op("call_123")
    """
    
    def __init__(self, redis_client):
        """Initialize with a Redis client.
        
        Args:
            redis_client: Async Redis client instance.
        """
        self._redis = redis_client
        self._pending: dict[str, float] = {}  # query_id -> created_at
        self._pubsub = None
        self._subscribed_channels: list[str] = []
    
    def _key(self, query_id: str) -> str:
        """Redis key for a pending operation marker."""
        return f"pending:{query_id}"
    
    def _channel(self, query_id: str) -> str:
        """Redis pub/sub channel for acknowledgment."""
        return f"pending_ack:{query_id}"
    
    async def register(self, query_id: str, operation_type: str) -> None:
        """Register a new pending operation.
        
        Sets a marker in Redis and subscribes to the acknowledgment channel.
        The subscribe MUST happen before the SSE event is yielded to prevent
        race conditions.
        
        Args:
            query_id: Unique identifier for correlation (usually tool_call_id).
            operation_type: Type of operation ("selection", "visualization").
        """
        # Set marker in Redis
        await self._redis.set(
            self._key(query_id),
            operation_type,
            ex=PENDING_OP_TTL,
        )
        
        # Track locally
        self._pending[query_id] = time.time()
        
        # Subscribe to ack channel
        channel = self._channel(query_id)
        if self._pubsub is None:
            self._pubsub = self._redis.pubsub()
        await self._pubsub.subscribe(channel)
        self._subscribed_channels.append(channel)
        
        logger.info(f"Registered pending operation: {query_id} ({operation_type})")
    
    async def await_all(self, timeout: float = 5.0) -> list[str]:
        """Wait for all pending operations to be acknowledged.
        
        Listens on subscribed pub/sub channels for acknowledgment messages.
        
        Args:
            timeout: Maximum time to wait in seconds.
            
        Returns:
            List of query_ids that were NOT acknowledged (timed out).
        """
        if not self._pending:
            return []
        
        query_ids = set(self._pending.keys())
        logger.info(f"Waiting for {len(query_ids)} pending operations: {list(query_ids)}")
        
        acknowledged = set()
        
        if self._pubsub is None:
            return list(query_ids)
        
        deadline = time.time() + timeout
        
        try:
            while acknowledged != query_ids and time.time() < deadline:
                remaining = deadline - time.time()
                if remaining <= 0:
                    break
                
                # Listen for messages with timeout
                try:
                    message = await asyncio.wait_for(
                        self._pubsub.get_message(
                            ignore_subscribe_messages=True,
                            timeout=min(remaining, 0.5),
                        ),
                        timeout=min(remaining, 1.0),
                    )
                except asyncio.TimeoutError:
                    continue
                
                if message and message.get("type") == "message":
                    # Extract query_id from channel name
                    channel = message.get("channel", "")
                    if isinstance(channel, bytes):
                        channel = channel.decode()
                    # Channel format: pending_ack:{query_id}
                    if channel.startswith("pending_ack:"):
                        acked_id = channel[len("pending_ack:"):]
                        if acked_id in query_ids:
                            acknowledged.add(acked_id)
                            logger.info(f"Acknowledged pending operation: {acked_id}")
        finally:
            # Cleanup subscriptions
            await self._cleanup()
        
        unacknowledged = list(query_ids - acknowledged)
        
        if unacknowledged:
            logger.warning(f"Unacknowledged operations after timeout: {unacknowledged}")
        else:
            logger.info("All pending operations acknowledged")
        
        return unacknowledged
    
    async def _cleanup(self) -> None:
        """Unsubscribe from all channels and clean up."""
        if self._pubsub and self._subscribed_channels:
            try:
                await self._pubsub.unsubscribe(*self._subscribed_channels)
                await self._pubsub.aclose()
            except Exception as e:
                logger.debug(f"Error cleaning up pubsub: {e}")
        
        self._pubsub = None
        self._subscribed_channels.clear()
        
        # Clean up Redis markers
        for query_id in self._pending:
            try:
                await self._redis.delete(self._key(query_id))
            except Exception:
                pass
        
        self._pending.clear()
    
    @property
    def count(self) -> int:
        """Get the number of pending operations."""
        return len(self._pending)


async def acknowledge_pending_op(redis_client, query_id: str) -> bool:
    """Acknowledge a pending operation via Redis pub/sub.
    
    This is called from the /api/selection endpoint (or similar) and may
    run on a different pod than the one waiting for the acknowledgment.
    
    Args:
        redis_client: Redis client instance.
        query_id: The query_id to acknowledge.
        
    Returns:
        True if a pending marker was found, False otherwise.
    """
    key = f"pending:{query_id}"
    channel = f"pending_ack:{query_id}"
    
    # Check if marker exists
    exists = await redis_client.exists(key)
    
    # Publish acknowledgment (even if marker gone, the subscriber may still be listening)
    await redis_client.publish(channel, "ack")
    
    # Remove marker
    if exists:
        await redis_client.delete(key)
        logger.info(f"Acknowledged pending operation: {query_id}")
        return True
    else:
        logger.debug(f"No pending operation marker found for query_id: {query_id}")
        # Still return True since we published - the subscriber may have received it
        return True
