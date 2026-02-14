"""Redis Streams backend for persistent episode processing.

Features:
- Messages persist in Redis (survives service restarts)
- Consumer Groups with XACK for guaranteed delivery
- XAUTOCLAIM for recovering abandoned messages after crashes
- Task references stored to prevent Python GC from collecting workers
"""

import asyncio
import logging
import os
import socket
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import redis.asyncio as redis

from graphiti_core.nodes import EpisodeType

from .queue_backend import EpisodeData, QueueBackend

logger = logging.getLogger(__name__)


@dataclass
class RedisQueueConfig:
    """Configuration for the Redis Streams backend."""

    redis_url: str = 'redis://localhost:6379'
    consumer_group: str = 'graphiti_workers'
    block_ms: int = 5000  # 5 seconds blocking read
    claim_min_idle_ms: int = 60000  # Reclaim messages idle for 60 seconds
    max_retries: int = 3
    shutdown_timeout: float = 30.0


@dataclass
class EpisodeMessage(EpisodeData):
    """Episode data with Redis Stream serialization."""

    message_id: str = ''

    def to_dict(self) -> dict[str, str]:
        """Serialize for Redis Stream."""
        return {
            'group_id': self.group_id,
            'name': self.name,
            'content': self.content,
            'source_description': self.source_description,
            'episode_type': self.episode_type,
            'uuid': self.uuid or '',
            'retry_count': str(self.retry_count),
        }

    @classmethod
    def from_stream_data(cls, message_id: str, data: dict) -> 'EpisodeMessage':
        """Deserialize from Redis Stream."""
        return cls(
            message_id=message_id,
            group_id=data['group_id'],
            name=data['name'],
            content=data['content'],
            source_description=data['source_description'],
            episode_type=data['episode_type'],
            uuid=data['uuid'] or None,
            retry_count=int(data.get('retry_count', 0)),
        )


class RedisStreamsBackend(QueueBackend):
    """Persistent queue backend using Redis Streams.

    This backend manages episode processing queues per group_id using Redis Streams.
    Each group_id gets its own stream (graphiti:queue:{group_id}).

    Features:
    - Persistent: Messages survive service restarts
    - Guaranteed delivery: Consumer Groups with acknowledgment
    - Crash recovery: XAUTOCLAIM recovers abandoned messages
    - No GC issues: Worker task references are stored
    """

    def __init__(self, config: RedisQueueConfig | None = None):
        self._config = config or RedisQueueConfig()
        self._redis: redis.Redis | None = None
        self._graphiti_client: Any = None
        self._entity_types: Any = None

        # Task references stored to prevent GC collection
        self._worker_tasks: dict[str, asyncio.Task] = {}
        self._worker_running: dict[str, bool] = {}
        self._shutting_down: bool = False

        # Unique consumer name per instance
        self._consumer_name = f'worker_{socket.gethostname()}_{os.getpid()}'

    def _stream_key(self, group_id: str) -> str:
        """Get Redis Stream key for a group_id."""
        return f'graphiti:queue:{group_id}'

    async def initialize(
        self,
        graphiti_client: Any,
        redis_client: redis.Redis | None = None,
        entity_types: Any = None,
        **kwargs,
    ) -> None:
        """Initialize with graphiti client and Redis connection.

        Args:
            graphiti_client: The Graphiti client instance for processing episodes
            redis_client: Optional Redis client (creates new if not provided)
            entity_types: Entity types dict for extraction
        """
        self._graphiti_client = graphiti_client
        self._entity_types = entity_types

        if redis_client is not None:
            self._redis = redis_client
        else:
            self._redis = redis.from_url(self._config.redis_url, decode_responses=True)

        logger.info(f'Redis Streams backend initialized with consumer: {self._consumer_name}')

        # Eager startup: resume workers for streams that have unprocessed messages
        await self._resume_pending_streams()

    async def _resume_pending_streams(self) -> None:
        """Resume workers for any streams that have unprocessed messages.

        This runs at startup to ensure no episodes are lost after a crash or restart.
        """
        if self._redis is None:
            return

        try:
            keys: list[str] = []
            async for key in self._redis.scan_iter(match='graphiti:queue:*', count=100):
                if ':dlq' not in key:
                    keys.append(key)

            if not keys:
                return

            resumed = 0
            for stream_key in keys:
                group_id = stream_key.removeprefix('graphiti:queue:')

                try:
                    groups = await self._redis.xinfo_groups(stream_key)
                    for group in groups:
                        if group.get('name') != self._config.consumer_group:
                            continue
                        lag = group.get('lag', 0) or 0
                        pending = group.get('pending', 0) or 0
                        if lag > 0 or pending > 0:
                            logger.info(
                                f'Resuming worker for {group_id}: lag={lag}, pending={pending}'
                            )
                            await self._ensure_worker_running(group_id)
                            resumed += 1
                except redis.ResponseError as e:
                    logger.debug(f'Could not inspect stream {stream_key}: {e}')

            if resumed > 0:
                logger.info(f'Resumed {resumed} workers for streams with pending messages')
            else:
                logger.info('No pending streams found at startup')

        except redis.ConnectionError as e:
            logger.error(f'Redis connection error during startup scan: {e}')
        except Exception as e:
            logger.warning(f'Error scanning pending streams: {e}')

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
        """Add episode to Redis Stream for processing."""
        if self._redis is None:
            raise RuntimeError('Queue backend not initialized. Call initialize() first.')

        stream_key = self._stream_key(group_id)

        message = EpisodeMessage(
            message_id='',
            group_id=group_id,
            name=name,
            content=content,
            source_description=source_description,
            episode_type=str(episode_type.value if hasattr(episode_type, 'value') else episode_type),
            uuid=uuid,
        )

        message_id: str = await self._redis.xadd(stream_key, message.to_dict())  # type: ignore[arg-type]

        logger.info(f'Queued episode {uuid} for group {group_id}: {message_id}')

        await self._ensure_worker_running(group_id)

        return message_id

    async def _ensure_worker_running(self, group_id: str) -> None:
        """Start worker for group_id if not already running."""
        if self._shutting_down:
            return

        if self._worker_running.get(group_id, False):
            return

        if self._redis is None:
            return

        stream_key = self._stream_key(group_id)

        try:
            await self._redis.xgroup_create(
                stream_key, self._config.consumer_group, id='0', mkstream=True
            )
            logger.info(f'Created consumer group for {group_id}')
        except redis.ResponseError as e:
            if 'BUSYGROUP' not in str(e):
                raise

        # Set flag before creating task to prevent duplicate workers (TOCTOU guard)
        self._worker_running[group_id] = True

        task = asyncio.create_task(self._process_stream(group_id))
        self._worker_tasks[group_id] = task

        def on_done(t: asyncio.Task):
            self._worker_tasks.pop(group_id, None)
            self._worker_running[group_id] = False
            if t.exception() and not self._shutting_down:
                logger.error(f'Worker for {group_id} crashed: {t.exception()}')

        task.add_done_callback(on_done)

    async def _process_stream(self, group_id: str) -> None:
        """Process messages from Redis Stream for a group_id."""
        if self._redis is None:
            raise RuntimeError('Queue backend not initialized')

        stream_key = self._stream_key(group_id)
        logger.info(f'Starting stream worker for {group_id}')

        try:
            await self._claim_abandoned(group_id)

            reclaim_counter = 0
            RECLAIM_EVERY = 12  # ~60s at 5s block_ms

            while not self._shutting_down:
                reclaim_counter += 1
                if reclaim_counter >= RECLAIM_EVERY:
                    reclaim_counter = 0
                    await self._claim_abandoned(group_id)

                try:
                    messages = await self._redis.xreadgroup(
                        groupname=self._config.consumer_group,
                        consumername=self._consumer_name,
                        streams={stream_key: '>'},
                        count=1,
                        block=self._config.block_ms,
                    )
                except redis.ConnectionError as e:
                    logger.error(f'Redis connection error for {group_id}: {e}')
                    await asyncio.sleep(5)
                    continue

                if not messages:
                    continue

                for _stream_name, stream_messages in messages:
                    for message_id, data in stream_messages:
                        await self._process_message(group_id, message_id, data)

        except asyncio.CancelledError:
            logger.info(f'Worker for {group_id} cancelled')
        except Exception as e:
            logger.error(f'Worker error for {group_id}: {e}')
            raise
        finally:
            self._worker_running[group_id] = False
            logger.info(f'Stopped worker for {group_id}')

    async def _process_message(self, group_id: str, message_id: str, data: dict) -> None:
        """Process a single message from the stream."""
        if self._redis is None:
            raise RuntimeError('Queue backend not initialized')

        stream_key = self._stream_key(group_id)
        episode = EpisodeMessage.from_stream_data(message_id, data)

        # Check retry limit before processing
        if episode.retry_count >= self._config.max_retries:
            logger.warning(
                f'Max retries ({self._config.max_retries}) exceeded for {message_id} '
                f'(uuid={episode.uuid}), giving up'
            )
            await self._redis.xack(stream_key, self._config.consumer_group, message_id)
            return

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

            await self._redis.xack(stream_key, self._config.consumer_group, message_id)
            logger.info(f'Successfully processed episode {episode.uuid}')

        except Exception as e:
            logger.error(f'Failed to process {message_id} (uuid={episode.uuid}): {e}')

            # Re-queue with incremented retry count, then ACK the failed message
            try:
                retry_data = episode.to_dict()
                retry_data['retry_count'] = str(episode.retry_count + 1)
                await self._redis.xadd(stream_key, retry_data)
                await self._redis.xack(stream_key, self._config.consumer_group, message_id)
            except Exception as retry_err:
                logger.error(f'Failed to re-queue {message_id}: {retry_err}')

    async def _claim_abandoned(self, group_id: str) -> None:
        """Claim and reprocess abandoned messages from previous crashes."""
        if self._redis is None:
            return

        stream_key = self._stream_key(group_id)
        total_claimed = 0
        cursor = '0'

        try:
            while True:
                result = await self._redis.xautoclaim(
                    stream_key,
                    self._config.consumer_group,
                    self._consumer_name,
                    min_idle_time=self._config.claim_min_idle_ms,
                    start_id=cursor,
                    count=50,
                )

                if not result or len(result) < 2 or not result[1]:
                    break

                next_cursor = result[0]
                claimed_messages = result[1]

                for message_id, data in claimed_messages:
                    await self._process_message(group_id, message_id, data)
                    total_claimed += 1

                if next_cursor == '0-0' or next_cursor == '0':
                    break
                cursor = next_cursor

            if total_claimed > 0:
                logger.info(f'Claimed and processed {total_claimed} abandoned messages for {group_id}')

        except redis.ResponseError as e:
            if 'NOGROUP' not in str(e):
                logger.warning(f'Error claiming abandoned messages for {group_id}: {e}')

    async def shutdown(self, timeout: float | None = None) -> None:
        """Graceful shutdown - wait for in-flight processing to complete."""
        timeout = timeout or self._config.shutdown_timeout
        self._shutting_down = True

        if not self._worker_tasks:
            logger.info('No workers to shut down')
            return

        logger.info(f'Shutting down {len(self._worker_tasks)} workers...')

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

        if self._redis is not None:
            await self._redis.close()

        logger.info('Redis Streams backend shutdown complete')

    def get_queue_size(self, group_id: str) -> int:
        """Get pending message count (approximate - requires async call for accuracy)."""
        return 0

    async def get_pending_count_async(self, group_id: str) -> int:
        """Get total unprocessed message count (pending + lag).

        Returns the sum of:
        - pending: messages delivered to a consumer but not yet acknowledged
        - lag: messages in the stream not yet delivered to any consumer
        """
        if self._redis is None:
            return 0

        stream_key = self._stream_key(group_id)
        try:
            groups = await self._redis.xinfo_groups(stream_key)
            for group in groups:
                if group.get('name') != self._config.consumer_group:
                    continue
                pending = group.get('pending', 0) or 0
                lag = group.get('lag', 0) or 0
                return pending + lag
            return 0
        except Exception:
            return 0

    async def get_status(self) -> tuple[int, int, list[dict]]:
        """Get total pending count, active workers, and per-group breakdown."""
        if self._redis is None:
            return 0, 0, []

        total_pending = 0
        currently_processing = 0
        groups_info = []

        try:
            async for key in self._redis.scan_iter(match='graphiti:queue:*', count=100):
                if isinstance(key, bytes):
                    key = key.decode('utf-8')
                if ':dlq' in key:
                    continue

                group_id = key.removeprefix('graphiti:queue:')
                stream_key = self._stream_key(group_id)

                pending_count = 0
                lag_count = 0
                try:
                    groups = await self._redis.xinfo_groups(stream_key)
                    for group in groups:
                        if group.get('name') != self._config.consumer_group:
                            continue
                        pending_count = group.get('pending', 0) or 0
                        lag_count = group.get('lag', 0) or 0
                        break
                except Exception:
                    pass

                total_count = pending_count + lag_count
                total_pending += total_count

                is_processing = self._worker_running.get(group_id, False)
                if is_processing:
                    currently_processing += 1

                if total_count > 0 or is_processing:
                    groups_info.append({
                        'group_id': group_id,
                        'pending': pending_count,
                        'queued': lag_count,
                        'total': total_count,
                        'processing': is_processing,
                    })
        except Exception as e:
            logger.error(f'Error getting all pending: {e}')

        return total_pending, currently_processing, groups_info

    # Backward compatibility alias
    get_all_pending_async = get_status

    def is_worker_running(self, group_id: str) -> bool:
        """Check if a worker is running for a group_id."""
        return self._worker_running.get(group_id, False)
