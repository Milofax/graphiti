"""Abstract queue backend interface for DB-neutral episode processing.

This module defines the interface that queue backends must implement,
allowing the MCP server to work with different backend implementations:
- Redis Streams (persistent, crash-resilient, multi-instance)
- In-Memory asyncio.Queue (simple, no external dependencies)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class EpisodeData:
    """Episode data for queue processing (backend-agnostic)."""

    group_id: str
    name: str
    content: str
    source_description: str
    episode_type: str
    uuid: str | None
    retry_count: int = 0


class QueueBackend(ABC):
    """Abstract interface for queue backends.

    Implementations must provide:
    - Persistent or in-memory episode queuing
    - Background worker management per group_id
    - Graceful shutdown with timeout
    """

    @abstractmethod
    async def initialize(
        self,
        graphiti_client: Any,
        entity_types: Any = None,
        **kwargs,
    ) -> None:
        """Initialize the queue backend with graphiti client.

        Args:
            graphiti_client: The Graphiti client for processing episodes
            entity_types: Entity types dict for extraction
            **kwargs: Backend-specific parameters (e.g., redis_client)
        """
        ...

    @abstractmethod
    async def add_episode(
        self,
        group_id: str,
        name: str,
        content: str,
        source_description: str,
        episode_type: Any,
        entity_types: Any,
        uuid: str | None,
    ) -> str:
        """Add episode to queue for background processing.

        Args:
            group_id: The group ID for the episode
            name: Name of the episode
            content: Episode content
            source_description: Description of the episode source
            episode_type: Type of the episode (EpisodeType enum or string)
            entity_types: Entity types for extraction
            uuid: Episode UUID

        Returns:
            Message/task identifier
        """
        ...

    @abstractmethod
    async def shutdown(self, timeout: float | None = None) -> None:
        """Graceful shutdown - wait for in-flight processing to complete.

        Args:
            timeout: Max seconds to wait for graceful shutdown
        """
        ...

    @abstractmethod
    def get_queue_size(self) -> int:
        """Get total pending message count.

        Note: May be approximate depending on backend.
        """
        ...

    @abstractmethod
    def is_worker_running(self) -> bool:
        """Check if the worker is running."""
        ...

    @abstractmethod
    async def get_status(self) -> tuple[int, int, list[dict]]:
        """Get queue status: (total_pending, currently_processing, groups_info)."""
        ...
