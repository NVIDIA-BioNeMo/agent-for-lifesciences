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

"""Event broadcaster for real-time training query synchronization.

This module provides a simple in-memory pub/sub system for broadcasting
training query events to all connected SSE clients.
"""

import asyncio
import json
import logging
from typing import Any, AsyncGenerator

logger = logging.getLogger(__name__)


class QueryEventBroadcaster:
    """Broadcasts training query events to all subscribed clients.
    
    Uses asyncio queues to fan out events to multiple subscribers.
    Each subscriber gets their own queue, ensuring isolation.
    """

    def __init__(self):
        self._subscribers: set[asyncio.Queue] = set()

    @property
    def subscriber_count(self) -> int:
        """Number of active subscribers."""
        return len(self._subscribers)

    async def subscribe(self) -> AsyncGenerator[str, None]:
        """Subscribe to query events.
        
        Yields:
            JSON strings (EventSourceResponse wraps with 'data: ' prefix).
        """
        queue: asyncio.Queue = asyncio.Queue()
        self._subscribers.add(queue)
        logger.info(f"New SSE subscriber connected. Total: {self.subscriber_count}")
        
        try:
            while True:
                event = await queue.get()
                yield json.dumps(event)
        except asyncio.CancelledError:
            logger.debug("SSE subscriber cancelled")
            raise
        finally:
            self._subscribers.discard(queue)
            logger.info(f"SSE subscriber disconnected. Total: {self.subscriber_count}")

    async def publish(self, event_type: str, data: dict[str, Any]):
        """Publish an event to all subscribers.
        
        Args:
            event_type: The type of event (query_created, query_deleted, etc.)
            data: Event payload data.
        """
        event = {"type": event_type, **data}
        
        if not self._subscribers:
            logger.debug(f"No subscribers for event: {event_type}")
            return
        
        logger.debug(f"Publishing {event_type} to {self.subscriber_count} subscribers")
        
        # Fan out to all subscribers
        for queue in self._subscribers:
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:
                logger.warning("Subscriber queue full, dropping event")


# Global broadcaster instance
broadcaster = QueryEventBroadcaster()
