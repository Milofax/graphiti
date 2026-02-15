"""Redis Streams backend for persistent episode processing.

Uses a single stream (graphiti:queue) for all group_ids. Messages carry
their group_id as a field. Failed messages are moved to a Dead Letter Queue
(graphiti:queue:dlq) after max_retries instead of being discarded.

Features:
- Single stream: no orphaned queues, one consumer group, one worker
- DLQ: failed messages preserved for inspection instead of silently deleted
- Migration: legacy per-group streams are migrated on first startup
- Crash recovery: XAUTOCLAIM recovers abandoned messages
- Exponential backoff on transient errors
"""

import asyncio
import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any
from zoneinfo import ZoneInfo

import redis.asyncio as redis

from graphiti_core.nodes import EpisodeType

from .queue_backend import EpisodeData, QueueBackend

logger = logging.getLogger(__name__)

MAX_BACKOFF_SECONDS = 60
STREAM_KEY = 'graphiti:queue'
DLQ_KEY = 'graphiti:queue:dlq'
LEGACY_KEY = 'graphiti:queue:legacy'
XTRIM_MAXLEN = 5000

# Legacy throttle time window (CET/CEST)
THROTTLE_TZ = ZoneInfo('Europe/Berlin')
THROTTLE_HOUR_START = 9   # 09:00
THROTTLE_HOUR_END = 22    # 22:00


@dataclass
class RedisQueueConfig:
    """Configuration for the Redis Streams backend."""

    redis_url: str = 'redis://localhost:6379'
    consumer_group: str = 'graphiti_workers'
    block_ms: int = 5000  # 5 seconds blocking read
    claim_min_idle_ms: int = 60000  # Reclaim messages idle for 60 seconds
    max_retries: int = 3
    shutdown_timeout: float = 30.0
    throttle_seconds: float = 0.0  # Pause between legacy messages (0 = no drainer)


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
    """Persistent queue backend using a single Redis Stream.

    All group_ids share one stream (graphiti:queue). The group_id travels
    as a message field. One worker processes messages sequentially.

    Features:
    - Persistent: Messages survive service restarts
    - Guaranteed delivery: Consumer Group with acknowledgment
    - DLQ: Failed messages moved to graphiti:queue:dlq after max_retries
    - Crash recovery: XAUTOCLAIM recovers abandoned messages
    - Migration: Legacy per-group streams migrated on first startup
    """

    def __init__(self, config: RedisQueueConfig | None = None):
        self._config = config or RedisQueueConfig()
        self._redis: redis.Redis | None = None
        self._graphiti_client: Any = None
        self._entity_types: Any = None

        # Single worker state
        self._worker_task: asyncio.Task | None = None
        self._drainer_task: asyncio.Task | None = None
        self._worker_running: bool = False
        self._actively_processing: bool = False
        self._shutting_down: bool = False

        # Stable consumer name — survives container restarts (same PEL ownership)
        self._consumer_name = os.environ.get('GRAPHITI_CONSUMER_NAME', 'worker_0')

    async def initialize(
        self,
        graphiti_client: Any,
        redis_client: redis.Redis | None = None,
        entity_types: Any = None,
        **kwargs,
    ) -> None:
        """Initialize with graphiti client and Redis connection."""
        self._graphiti_client = graphiti_client
        self._entity_types = entity_types

        if redis_client is not None:
            self._redis = redis_client
        else:
            self._redis = redis.from_url(self._config.redis_url, decode_responses=True)

        logger.info(f'Redis Streams backend initialized with consumer: {self._consumer_name}')

        # Create consumer group on the single stream (idempotent)
        try:
            await self._redis.xgroup_create(
                STREAM_KEY, self._config.consumer_group, id='0', mkstream=True
            )
            logger.info('Created consumer group on single stream')
        except redis.ResponseError as e:
            if 'BUSYGROUP' not in str(e):
                raise

        # Migrate legacy per-group streams
        await self._migrate_legacy_streams()

        # Start worker if there are unprocessed messages
        await self._start_worker_if_needed()

        # Start legacy drainer if configured
        if self._config.throttle_seconds > 0:
            await self._start_legacy_drainer()

    async def _migrate_legacy_streams(self) -> None:
        """Migrate messages from legacy per-group streams to the single stream.

        Scans for graphiti:queue:* keys (excluding the main stream and DLQ),
        copies unprocessed messages to STREAM_KEY, then deletes the old stream.
        """
        if self._redis is None:
            return

        try:
            legacy_keys: list[str] = []
            async for key in self._redis.scan_iter(match='graphiti:queue:*', count=100):
                if isinstance(key, bytes):
                    key = key.decode('utf-8')
                # Skip the single stream itself and DLQ
                if key == STREAM_KEY or key == DLQ_KEY or ':dlq' in key:
                    continue
                legacy_keys.append(key)

            if not legacy_keys:
                logger.info('No legacy streams to migrate')
                return

            total_migrated = 0
            total_skipped = 0

            for legacy_key in legacy_keys:
                group_id = legacy_key.removeprefix('graphiti:queue:')

                try:
                    # Get all messages from the legacy stream
                    messages = await self._redis.xrange(legacy_key)
                    if not messages:
                        await self._redis.delete(legacy_key)
                        logger.info(f'Deleted empty legacy stream: {legacy_key}')
                        continue

                    # Determine which messages are already processed (ACK'd)
                    acked_ids: set[str] = set()
                    try:
                        groups = await self._redis.xinfo_groups(legacy_key)
                        for group in groups:
                            group_name = group.get('name', '')
                            # Get pending messages for this consumer group
                            try:
                                pending = await self._redis.xpending_range(
                                    legacy_key, group_name,
                                    min='-', max='+', count=10000,
                                )
                                pending_ids = {p['message_id'] for p in pending}
                            except Exception:
                                pending_ids = set()

                            # Messages in stream but NOT in PEL are already processed
                            all_ids = {msg_id for msg_id, _ in messages}
                            acked_ids.update(all_ids - pending_ids)
                    except redis.ResponseError:
                        # No consumer group — all messages are unprocessed
                        pass

                    # Migrate unprocessed messages
                    migrated = 0
                    skipped = 0
                    # Choose target: legacy list (if throttled) or main stream
                    use_legacy = self._config.throttle_seconds > 0
                    for msg_id, data in messages:
                        if msg_id in acked_ids:
                            skipped += 1
                            continue
                        # Ensure group_id field is set
                        if 'group_id' not in data:
                            data['group_id'] = group_id
                        if use_legacy:
                            await self._redis.rpush(LEGACY_KEY, json.dumps(data))
                        else:
                            await self._redis.xadd(STREAM_KEY, data)
                        migrated += 1

                    total_migrated += migrated
                    total_skipped += skipped

                    # Delete the legacy stream
                    await self._redis.delete(legacy_key)
                    logger.info(
                        f'Migrated {migrated} messages from {legacy_key} '
                        f'(skipped {skipped} already processed)'
                    )

                except Exception as e:
                    logger.warning(f'Error migrating {legacy_key}: {e}')

            logger.info(
                f'Migration complete: {total_migrated} messages migrated, '
                f'{total_skipped} skipped from {len(legacy_keys)} legacy streams'
            )

        except redis.ConnectionError as e:
            logger.error(f'Redis connection error during migration: {e}')
        except Exception as e:
            logger.warning(f'Error during legacy stream migration: {e}')

    async def _start_legacy_drainer(self) -> None:
        """Start the legacy message drainer if there are messages to process."""
        if self._redis is None:
            return
        count = await self._redis.llen(LEGACY_KEY)
        if count == 0:
            logger.info('No legacy messages to drain')
            return
        logger.info(f'Starting legacy drainer for {count} messages')
        self._drainer_task = asyncio.create_task(self._drain_legacy())

    async def _drain_legacy(self) -> None:
        """Drain legacy messages one at a time with throttle + time window."""
        if self._redis is None:
            return

        try:
            while not self._shutting_down:
                # Time window check: only process during 9:00-22:00 CET
                now_cet = datetime.now(THROTTLE_TZ)
                if now_cet.hour < THROTTLE_HOUR_START or now_cet.hour >= THROTTLE_HOUR_END:
                    next_start = now_cet.replace(
                        hour=THROTTLE_HOUR_START, minute=0, second=0, microsecond=0
                    )
                    if now_cet.hour >= THROTTLE_HOUR_END:
                        next_start += timedelta(days=1)
                    sleep_secs = (next_start - now_cet).total_seconds()
                    logger.info(f'Legacy drainer paused until 09:00 CET ({sleep_secs:.0f}s)')
                    await asyncio.sleep(sleep_secs)
                    continue

                # Pop one message from the legacy list
                raw = await self._redis.lpop(LEGACY_KEY)
                if raw is None:
                    logger.info('Legacy drainer complete — all messages processed')
                    return

                data = json.loads(raw)
                group_id = data.get('group_id', 'unknown')
                remaining = await self._redis.llen(LEGACY_KEY)
                logger.info(
                    f'Legacy drainer: processing for {group_id} '
                    f'({remaining} remaining)'
                )

                # Use a synthetic message_id for logging
                msg_id = f'legacy-{datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")}'
                success = await self._process_message(group_id, msg_id, data)

                if not success:
                    # Put back at front of list for retry
                    await self._redis.lpush(LEGACY_KEY, json.dumps(data))
                    logger.warning('Legacy message failed, re-queued for retry')

                # Throttle: sleep between messages
                logger.debug(f'Legacy drainer: sleeping {self._config.throttle_seconds}s')
                await asyncio.sleep(self._config.throttle_seconds)

        except asyncio.CancelledError:
            logger.info('Legacy drainer cancelled')
        except Exception as e:
            logger.error(f'Legacy drainer error: {e}')

    async def _start_worker_if_needed(self) -> None:
        """Start the worker if there are unprocessed messages."""
        if self._redis is None or self._shutting_down:
            return

        try:
            groups = await self._redis.xinfo_groups(STREAM_KEY)
            for group in groups:
                if group.get('name') != self._config.consumer_group:
                    continue
                lag = group.get('lag', 0) or 0
                pending = group.get('pending', 0) or 0
                if lag > 0 or pending > 0:
                    logger.info(f'Starting worker: lag={lag}, pending={pending}')
                    self._start_worker()
                    return
        except redis.ResponseError:
            # Stream might not exist yet
            pass
        except Exception as e:
            logger.warning(f'Error checking stream at startup: {e}')

    def _start_worker(self) -> None:
        """Start the single worker task if not already running."""
        if self._shutting_down or self._worker_running:
            return

        self._worker_running = True
        task = asyncio.create_task(self._process_stream())
        self._worker_task = task

        def on_done(t: asyncio.Task):
            self._worker_task = None
            self._worker_running = False
            self._actively_processing = False
            if t.exception() and not self._shutting_down:
                logger.error(f'Worker crashed: {t.exception()}')

        task.add_done_callback(on_done)

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
        """Add episode to the single Redis Stream for processing."""
        if self._redis is None:
            raise RuntimeError('Queue backend not initialized. Call initialize() first.')

        message = EpisodeMessage(
            message_id='',
            group_id=group_id,
            name=name,
            content=content,
            source_description=source_description,
            episode_type=str(episode_type.value if hasattr(episode_type, 'value') else episode_type),
            uuid=uuid,
        )

        message_id: str = await self._redis.xadd(STREAM_KEY, message.to_dict())  # type: ignore[arg-type]

        logger.info(f'Queued episode {uuid} for group {group_id}: {message_id}')

        self._start_worker()

        return message_id

    async def _process_stream(self) -> None:
        """Process messages from the single Redis Stream.

        Uses a two-phase read: first check for pending (previously failed) messages,
        then read new messages. Exponential backoff on consecutive errors prevents
        rapid retry exhaustion when the LLM is temporarily unavailable.
        """
        if self._redis is None:
            raise RuntimeError('Queue backend not initialized')

        logger.info('Starting stream worker')
        consecutive_errors = 0

        try:
            await self._claim_abandoned()

            reclaim_counter = 0
            RECLAIM_EVERY = 12  # ~60s at 5s block_ms

            while not self._shutting_down:
                reclaim_counter += 1
                if reclaim_counter >= RECLAIM_EVERY:
                    reclaim_counter = 0
                    await self._claim_abandoned()
                    try:
                        await self._redis.xtrim(
                            STREAM_KEY, maxlen=XTRIM_MAXLEN, approximate=True
                        )
                    except Exception:
                        pass

                # Phase 1: Check for pending (unacked) messages from previous failures
                message_to_process = None
                try:
                    pending = await self._redis.xreadgroup(
                        groupname=self._config.consumer_group,
                        consumername=self._consumer_name,
                        streams={STREAM_KEY: '0'},
                        count=1,
                    )
                    if pending:
                        for _stream_name, stream_messages in pending:
                            if stream_messages:
                                message_to_process = stream_messages[0]
                                break
                except redis.ConnectionError as e:
                    logger.error(f'Redis connection error: {e}')
                    await asyncio.sleep(5)
                    continue

                # Phase 2: No pending messages, block-read for new ones
                if message_to_process is None:
                    try:
                        messages = await self._redis.xreadgroup(
                            groupname=self._config.consumer_group,
                            consumername=self._consumer_name,
                            streams={STREAM_KEY: '>'},
                            count=1,
                            block=self._config.block_ms,
                        )
                        if messages:
                            for _stream_name, stream_messages in messages:
                                if stream_messages:
                                    message_to_process = stream_messages[0]
                                    break
                    except redis.ConnectionError as e:
                        logger.error(f'Redis connection error: {e}')
                        await asyncio.sleep(5)
                        continue

                if message_to_process is None:
                    continue

                # Enforce max_retries via Redis delivery count (PEL)
                msg_id_check = message_to_process[0]
                try:
                    pending_info = await self._redis.xpending_range(
                        STREAM_KEY, self._config.consumer_group,
                        min=msg_id_check, max=msg_id_check, count=1,
                    )
                    if (
                        pending_info
                        and pending_info[0].get('times_delivered', 0) > self._config.max_retries
                    ):
                        times = pending_info[0]['times_delivered']
                        msg_data = message_to_process[1]
                        logger.warning(
                            f'Moving {msg_id_check} to DLQ after {times} deliveries: '
                            f'{msg_data.get("name", "?")}'
                        )
                        # Preserve in DLQ with original data + metadata
                        dlq_data = dict(msg_data)
                        dlq_data['original_stream_id'] = msg_id_check
                        dlq_data['times_delivered'] = str(times)
                        dlq_data['moved_to_dlq_at'] = datetime.now(timezone.utc).isoformat()
                        await self._redis.xadd(DLQ_KEY, dlq_data)
                        await self._redis.xack(
                            STREAM_KEY, self._config.consumer_group, msg_id_check
                        )
                        try:
                            await self._redis.xdel(STREAM_KEY, msg_id_check)
                        except Exception:
                            pass
                        continue
                except Exception as e:
                    logger.debug(f'Could not check delivery count for {msg_id_check}: {e}')

                message_id, data = message_to_process
                group_id = data.get('group_id', 'unknown')
                self._actively_processing = True
                try:
                    success = await self._process_message(group_id, message_id, data)
                finally:
                    self._actively_processing = False

                if success:
                    consecutive_errors = 0
                else:
                    consecutive_errors += 1
                    delay = min(2 ** consecutive_errors, MAX_BACKOFF_SECONDS)
                    logger.warning(
                        f'Backing off {delay}s ({consecutive_errors} consecutive error(s))'
                    )
                    await asyncio.sleep(delay)

        except asyncio.CancelledError:
            logger.info('Worker cancelled')
        except Exception as e:
            logger.error(f'Worker error: {e}')
            raise
        finally:
            self._worker_running = False
            self._actively_processing = False
            logger.info('Stopped stream worker')

    async def _process_message(self, group_id: str, message_id: str, data: dict) -> bool:
        """Process a single message from the stream.

        Returns True on success. On failure, the message is NOT acknowledged
        and stays pending for retry with backoff.
        """
        if self._redis is None:
            raise RuntimeError('Queue backend not initialized')

        episode = EpisodeMessage.from_stream_data(message_id, data)

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

            await self._redis.xack(STREAM_KEY, self._config.consumer_group, message_id)
            try:
                await self._redis.xdel(STREAM_KEY, message_id)
            except Exception:
                pass  # Periodic XTRIM will clean up
            logger.info(f'Successfully processed episode {episode.uuid}')
            return True

        except Exception as e:
            logger.error(f'Failed to process {message_id} (uuid={episode.uuid}): {e}')
            # Don't ACK, don't re-queue — message stays pending for retry with backoff
            return False

    async def _claim_abandoned(self) -> None:
        """Claim abandoned messages from dead consumers."""
        if self._redis is None:
            return

        total_claimed = 0
        cursor = '0'

        try:
            while True:
                result = await self._redis.xautoclaim(
                    STREAM_KEY,
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
                total_claimed += len(claimed_messages)

                if next_cursor == '0-0' or next_cursor == '0':
                    break
                cursor = next_cursor

            if total_claimed > 0:
                logger.info(f'Claimed {total_claimed} abandoned messages')

        except redis.ResponseError as e:
            if 'NOGROUP' not in str(e):
                logger.warning(f'Error claiming abandoned messages: {e}')

    async def shutdown(self, timeout: float | None = None) -> None:
        """Graceful shutdown - wait for in-flight processing to complete."""
        timeout = timeout or self._config.shutdown_timeout
        self._shutting_down = True

        # Cancel legacy drainer first
        if self._drainer_task and not self._drainer_task.done():
            self._drainer_task.cancel()
            try:
                await asyncio.wait_for(self._drainer_task, timeout=5)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass

        if self._worker_task is None:
            logger.info('No worker to shut down')
        else:
            logger.info('Shutting down worker...')
            self._worker_task.cancel()

            try:
                await asyncio.wait_for(
                    asyncio.gather(self._worker_task, return_exceptions=True),
                    timeout=timeout,
                )
            except asyncio.TimeoutError:
                logger.warning(f'Shutdown timeout ({timeout}s) reached, forcing termination')

        if self._redis is not None:
            await self._redis.close()

        logger.info('Redis Streams backend shutdown complete')

    def get_queue_size(self) -> int:
        """Get pending message count (approximate - requires async call for accuracy)."""
        return 0

    async def get_status(self) -> tuple[int, int, list[dict]]:
        """Get queue status: total pending, processing flag, and DLQ count."""
        if self._redis is None:
            return 0, 0, []

        total_pending = 0
        currently_processing = 1 if self._actively_processing else 0
        groups_info: list[dict] = []

        try:
            pending_count = 0
            lag_count = 0
            try:
                groups = await self._redis.xinfo_groups(STREAM_KEY)
                for group in groups:
                    if group.get('name') != self._config.consumer_group:
                        continue
                    pending_count = group.get('pending', 0) or 0
                    lag_count = group.get('lag', 0) or 0
                    break
            except redis.ResponseError:
                # Stream might not exist yet — try XLEN as fallback
                try:
                    total_pending = await self._redis.xlen(STREAM_KEY)
                except Exception:
                    pass
                return total_pending, currently_processing, groups_info

            total_pending = pending_count + lag_count

            # DLQ count
            dlq_count = 0
            try:
                dlq_count = await self._redis.xlen(DLQ_KEY)
            except Exception:
                pass

            groups_info.append({
                'stream': STREAM_KEY,
                'pending': pending_count,
                'queued': lag_count,
                'total': total_pending,
                'dlq_count': dlq_count,
                'processing': self._actively_processing,
                'worker_running': self._worker_running,
            })

        except Exception as e:
            logger.error(f'Error getting queue status: {e}')

        return total_pending, currently_processing, groups_info

    def is_worker_running(self) -> bool:
        """Check if the worker is running."""
        return self._worker_running
