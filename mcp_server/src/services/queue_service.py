"""Queue service factory for DB-neutral episode processing.

This module provides the factory function to create the appropriate queue backend
based on configuration.

Backend selection logic:
1. If queue.backend == "redis" → Redis Streams backend
2. If queue.backend == "memory" → In-Memory backend
3. If queue.backend == "auto" (default):
   - If queue.redis_url is set → Redis Streams
   - If database.provider == "falkordb" → Redis Streams (uses FalkorDB's Redis)
   - Otherwise → In-Memory
"""

import logging
from typing import TYPE_CHECKING
from urllib.parse import urlparse

from .queue_backend import QueueBackend
from .queue_memory import InMemoryBackend, InMemoryQueueConfig
from .queue_redis import RedisQueueConfig, RedisStreamsBackend

if TYPE_CHECKING:
    from config.schema import GraphitiConfig

logger = logging.getLogger(__name__)


def create_queue_backend(config: 'GraphitiConfig') -> QueueBackend:
    """Create the appropriate queue backend based on configuration.

    Args:
        config: Full GraphitiConfig (uses config.queue for backend settings)

    Returns:
        QueueBackend instance (Redis Streams or In-Memory)
    """
    q = config.queue
    backend_type = q.backend
    redis_url: str | None = q.redis_url

    # Auto-detection logic
    if backend_type == 'auto':
        if redis_url:
            backend_type = 'redis'
            logger.info('Auto-detected Redis backend from queue.redis_url')
        elif config.database.provider == 'falkordb' and config.database.providers.falkordb:
            # FalkorDB is Redis-based, reuse it for queue
            redis_url = _build_redis_url_from_falkordb(config)
            backend_type = 'redis'
            logger.info('Auto-detected Redis backend from FalkorDB database')
        else:
            backend_type = 'memory'
            logger.info('Auto-detected In-Memory backend (no Redis available)')

    # Create the appropriate backend
    if backend_type == 'redis':
        if not redis_url:
            raise ValueError('Redis backend requires redis_url configuration')
        redis_cfg = RedisQueueConfig(
            redis_url=redis_url,
            consumer_group=q.consumer_group,
            block_ms=q.block_ms,
            claim_min_idle_ms=q.claim_min_idle_ms,
            max_retries=q.max_retries,
            shutdown_timeout=q.shutdown_timeout,
            throttle_seconds=q.throttle_seconds,
        )
        logger.info('Using Redis Streams backend')
        return RedisStreamsBackend(config=redis_cfg)

    # Default: In-Memory
    memory_cfg = InMemoryQueueConfig(
        shutdown_timeout=q.shutdown_timeout,
    )
    logger.info('Using In-Memory backend')
    return InMemoryBackend(config=memory_cfg)


def _build_redis_url_from_falkordb(config: 'GraphitiConfig') -> str:
    """Build Redis URL from FalkorDB configuration.

    FalkorDB uses Redis as its backend, so we can reuse the connection for queue.
    Handles password authentication properly (Redis uses :password@ format).
    """
    falkor_cfg = config.database.providers.falkordb
    if falkor_cfg is None:
        return 'redis://localhost:6379'

    parsed = urlparse(falkor_cfg.uri)

    # Only add password if not already in URI and password is configured
    if falkor_cfg.password and not parsed.password:
        host = parsed.hostname or 'localhost'
        port = parsed.port or 6379
        return f'redis://:{falkor_cfg.password}@{host}:{port}'

    return falkor_cfg.uri


# Export all public symbols
__all__ = [
    'QueueBackend',
    'RedisStreamsBackend',
    'InMemoryBackend',
    'create_queue_backend',
]
