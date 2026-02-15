"""In-Memory queue backend using asyncio.Queue.

Simple backend for development/testing or when no Redis is available.
Based on upstream's original implementation.

Features:
- No external dependencies
- Simple asyncio.Queue per group_id
- Non-persistent (data lost on restart)
"""

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from graphiti_core.nodes import EpisodeType

from .queue_backend import EpisodeData, QueueBackend

logger = logging.getLogger(__name__)


@dataclass
class InMemoryQueueConfig:
    """Configuration for the In-Memory backend."""

    shutdown_timeout: float = 30.0


class InMemoryBackend(QueueBackend):
    """In-memory queue backend using asyncio.Queue.

    This backend manages episode processing queues per group_id using asyncio.Queue.
    Each group_id gets its own queue and worker task.

    Features:
    - Simple: No external dependencies
    - Fast: Direct async processing
    - Non-persistent: Data lost on restart (suitable for dev/testing)
    """

    def __init__(self, config: InMemoryQueueConfig | None = None):
        self._config = config or InMemoryQueueConfig()
        self._graphiti_client: Any = None
        self._entity_types: Any = None

        # Per-group queues and workers
        self._queues: dict[str, asyncio.Queue[EpisodeData]] = {}
        self._worker_tasks: dict[str, asyncio.Task] = {}
        self._worker_running: dict[str, bool] = {}
        self._shutting_down: bool = False

    async def initialize(
        self,
        graphiti_client: Any,
        entity_types: Any = None,
        **kwargs,
    ) -> None:
        """Initialize with graphiti client.

        Args:
            graphiti_client: The Graphiti client for processing episodes
            entity_types: Entity types dict for extraction
            **kwargs: Ignored (for compatibility with Redis backend signature)
        """
        self._graphiti_client = graphiti_client
        self._entity_types = entity_types
        logger.info('In-Memory queue backend initialized')

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
        """Add episode to in-memory queue for processing."""
        if self._graphiti_client is None:
            raise RuntimeError('Queue backend not initialized. Call initialize() first.')

        # Ensure queue exists for this group
        if group_id not in self._queues:
            self._queues[group_id] = asyncio.Queue()

        episode = EpisodeData(
            group_id=group_id,
            name=name,
            content=content,
            source_description=source_description,
            episode_type=str(episode_type.value if hasattr(episode_type, 'value') else episode_type),
            uuid=uuid,
        )

        await self._queues[group_id].put(episode)

        # Generate a simple task ID
        task_id = f'{group_id}:{uuid or "no-uuid"}:{self._queues[group_id].qsize()}'
        logger.info(f'Queued episode {uuid} for group {group_id}: {task_id}')

        await self._ensure_worker_running(group_id)

        return task_id

    async def _ensure_worker_running(self, group_id: str) -> None:
        """Start worker for group_id if not already running."""
        if self._shutting_down:
            return

        if self._worker_running.get(group_id, False):
            return

        # Set flag before creating task to prevent duplicate workers (TOCTOU guard)
        self._worker_running[group_id] = True

        task = asyncio.create_task(self._process_queue(group_id))
        self._worker_tasks[group_id] = task

        def on_done(t: asyncio.Task):
            self._worker_tasks.pop(group_id, None)
            self._worker_running[group_id] = False
            if t.exception() and not self._shutting_down:
                logger.error(f'Worker for {group_id} crashed: {t.exception()}')

        task.add_done_callback(on_done)

    async def _process_queue(self, group_id: str) -> None:
        """Process messages from asyncio.Queue for a group_id."""
        queue = self._queues.get(group_id)
        if queue is None:
            return

        logger.info(f'Starting in-memory worker for {group_id}')

        try:
            while not self._shutting_down:
                try:
                    # Wait for item with timeout to allow shutdown check
                    episode = await asyncio.wait_for(queue.get(), timeout=5.0)
                except asyncio.TimeoutError:
                    # Check if queue is empty and we should exit
                    if queue.empty():
                        logger.info(f'Queue empty for {group_id}, worker exiting')
                        break
                    continue

                await self._process_episode(group_id, episode)
                queue.task_done()

        except asyncio.CancelledError:
            logger.info(f'Worker for {group_id} cancelled')
        except Exception as e:
            logger.error(f'Worker error for {group_id}: {e}')
            raise
        finally:
            self._worker_running[group_id] = False
            logger.info(f'Stopped worker for {group_id}')

    async def _process_episode(self, group_id: str, episode: EpisodeData) -> None:
        """Process a single episode from the queue."""
        try:
            logger.info(f'Processing episode {episode.uuid} for {group_id}')

            await self._graphiti_client.add_episode(
                name=episode.name,
                episode_body=episode.content,
                source_description=episode.source_description,
                source=EpisodeType.from_str(episode.episode_type),
                group_id=group_id,
                reference_time=datetime.now(timezone.utc),
                uuid=episode.uuid,
                entity_types=self._entity_types,
            )

            logger.info(f'Successfully processed episode {episode.uuid}')

        except Exception as e:
            logger.error(f'Failed to process episode {episode.uuid}: {e}')
            # In-memory backend: no retry mechanism, just log the error

    async def shutdown(self, timeout: float | None = None) -> None:
        """Graceful shutdown - wait for in-flight processing to complete."""
        timeout = timeout or self._config.shutdown_timeout
        self._shutting_down = True

        if not self._worker_tasks:
            logger.info('No workers to shut down')
            return

        logger.info(f'Shutting down {len(self._worker_tasks)} workers...')

        # Cancel all workers
        for group_id, task in self._worker_tasks.items():
            task.cancel()
            logger.debug(f'Cancelled worker for {group_id}')

        try:
            await asyncio.wait_for(
                asyncio.gather(*self._worker_tasks.values(), return_exceptions=True),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            logger.warning(f'Shutdown timeout ({timeout}s) reached, forcing termination')

        # Clear queues
        self._queues.clear()

        logger.info('In-Memory backend shutdown complete')

    def get_queue_size(self, group_id: str) -> int:
        """Get pending message count for a group_id."""
        queue = self._queues.get(group_id)
        return queue.qsize() if queue else 0

    async def get_all_pending_count_async(self) -> tuple[int, int, list[dict]]:
        """Get total pending count, active workers, and per-group breakdown."""
        total_pending = 0
        currently_processing = 0
        groups_info = []

        for group_id, queue in self._queues.items():
            pending = queue.qsize()
            total_pending += pending
            is_running = self._worker_running.get(group_id, False)
            if is_running:
                currently_processing += 1
            if pending > 0 or is_running:
                groups_info.append({
                    'group_id': group_id,
                    'pending': pending,
                    'processing': is_running,
                })

        return total_pending, currently_processing, groups_info

    def is_worker_running(self, group_id: str) -> bool:
        """Check if a worker is running for a group_id."""
        return self._worker_running.get(group_id, False)

    # Alias for ABC compatibility (local/combined uses get_status)
    get_status = get_all_pending_count_async
