"""Tests for Redis Streams queue robustness fixes.

TDD — these tests are written BEFORE the implementation.
They verify:
1. Stable consumer name (env var instead of hostname+pid)
2. XDEL after XACK (stream cleanup per message)
3. Periodic XTRIM (safety net for stream growth)
4. Max-retries via delivery count (discard zombie messages)
"""

import asyncio
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from services.queue_redis import RedisQueueConfig, RedisStreamsBackend


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def config():
    return RedisQueueConfig(max_retries=3)


@pytest.fixture
def mock_redis():
    """Mock Redis client with all methods used by RedisStreamsBackend."""
    r = MagicMock()
    r.xadd = AsyncMock(return_value='1000-0')
    r.xreadgroup = AsyncMock(return_value=[])
    r.xack = AsyncMock(return_value=1)
    r.xdel = AsyncMock(return_value=1)
    r.xtrim = AsyncMock(return_value=0)
    r.xgroup_create = AsyncMock()
    r.xautoclaim = AsyncMock(return_value=('0-0', []))
    r.xpending_range = AsyncMock(return_value=[])
    r.xinfo_groups = AsyncMock(return_value=[])
    r.close = AsyncMock()
    r.scan_iter = MagicMock(return_value=AsyncIterEmpty())
    return r


@pytest.fixture
def mock_graphiti():
    client = MagicMock()
    client.add_episode = AsyncMock()
    return client


class AsyncIterEmpty:
    """Async iterator that yields nothing."""
    def __aiter__(self):
        return self
    async def __anext__(self):
        raise StopAsyncIteration


# ===========================================================================
# 1. Stable consumer name
# ===========================================================================

class TestStableConsumerName:
    """Consumer name must be stable across container restarts."""

    def test_default_consumer_name_is_worker_0(self, config):
        """Without env var, consumer name defaults to 'worker_0'."""
        with patch.dict(os.environ, {}, clear=False):
            # Remove env var if present
            os.environ.pop('GRAPHITI_CONSUMER_NAME', None)
            backend = RedisStreamsBackend(config=config)
            assert backend._consumer_name == 'worker_0'

    def test_consumer_name_from_env_var(self, config):
        """GRAPHITI_CONSUMER_NAME env var overrides default."""
        with patch.dict(os.environ, {'GRAPHITI_CONSUMER_NAME': 'custom_worker'}):
            backend = RedisStreamsBackend(config=config)
            assert backend._consumer_name == 'custom_worker'

    def test_consumer_name_does_not_contain_hostname(self, config):
        """Consumer name must NOT contain hostname (changes per container)."""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop('GRAPHITI_CONSUMER_NAME', None)
            backend = RedisStreamsBackend(config=config)
            import socket
            assert socket.gethostname() not in backend._consumer_name

    def test_consumer_name_does_not_contain_pid(self, config):
        """Consumer name must NOT contain PID (always 1 in Docker, but still volatile)."""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop('GRAPHITI_CONSUMER_NAME', None)
            backend = RedisStreamsBackend(config=config)
            assert str(os.getpid()) not in backend._consumer_name


# ===========================================================================
# 2. XDEL after XACK
# ===========================================================================

class TestXdelAfterXack:
    """Processed messages must be deleted from stream after acknowledgment."""

    @pytest.mark.asyncio
    async def test_xdel_called_after_successful_processing(
        self, config, mock_redis, mock_graphiti
    ):
        """After XACK, XDEL must be called to remove the message from the stream."""
        backend = RedisStreamsBackend(config=config)
        backend._redis = mock_redis
        backend._graphiti_client = mock_graphiti

        success = await backend._process_message('main', '1000-0', {
            'group_id': 'main',
            'name': 'Test',
            'content': 'Test content',
            'source_description': 'test',
            'episode_type': 'text',
            'uuid': 'uuid-1',
            'retry_count': '0',
        })

        assert success is True
        mock_redis.xack.assert_called_once()
        mock_redis.xdel.assert_called_once_with('graphiti:queue:main', '1000-0')

    @pytest.mark.asyncio
    async def test_xdel_not_called_on_failure(
        self, config, mock_redis, mock_graphiti
    ):
        """On processing failure, neither XACK nor XDEL should be called."""
        mock_graphiti.add_episode = AsyncMock(side_effect=RuntimeError('LLM down'))

        backend = RedisStreamsBackend(config=config)
        backend._redis = mock_redis
        backend._graphiti_client = mock_graphiti

        success = await backend._process_message('main', '1000-0', {
            'group_id': 'main',
            'name': 'Test',
            'content': 'Test content',
            'source_description': 'test',
            'episode_type': 'text',
            'uuid': 'uuid-1',
            'retry_count': '0',
        })

        assert success is False
        mock_redis.xack.assert_not_called()
        mock_redis.xdel.assert_not_called()

    @pytest.mark.asyncio
    async def test_xdel_failure_does_not_break_processing(
        self, config, mock_redis, mock_graphiti
    ):
        """If XDEL fails, processing should still return True (XTRIM will clean up)."""
        mock_redis.xdel = AsyncMock(side_effect=Exception('Redis error'))

        backend = RedisStreamsBackend(config=config)
        backend._redis = mock_redis
        backend._graphiti_client = mock_graphiti

        success = await backend._process_message('main', '1000-0', {
            'group_id': 'main',
            'name': 'Test',
            'content': 'Test content',
            'source_description': 'test',
            'episode_type': 'text',
            'uuid': 'uuid-1',
            'retry_count': '0',
        })

        assert success is True
        mock_redis.xack.assert_called_once()


# ===========================================================================
# 3. Periodic XTRIM
# ===========================================================================

class TestPeriodicXtrim:
    """Maintenance cycle must XTRIM streams as safety net."""

    @pytest.mark.asyncio
    async def test_xtrim_called_during_maintenance(self, config, mock_redis, mock_graphiti):
        """XTRIM should be called alongside _claim_abandoned in maintenance cycle."""
        backend = RedisStreamsBackend(config=config)
        backend._redis = mock_redis
        backend._graphiti_client = mock_graphiti

        # Let the worker run for enough iterations to trigger maintenance.
        # Each loop iteration does 2 xreadgroup calls (Phase 1 pending + Phase 2 new),
        # but reclaim_counter increments once per loop iteration.
        iteration = 0
        RECLAIM_EVERY = 12  # Must match constant in code

        async def fake_xreadgroup(*args, **kwargs):
            nonlocal iteration
            iteration += 1
            # Need RECLAIM_EVERY * 2 calls (2 per loop) + margin
            if iteration > RECLAIM_EVERY * 2 + 4:
                backend._shutting_down = True
            return []

        mock_redis.xreadgroup = fake_xreadgroup

        await backend._ensure_worker_running('main')
        # Wait for worker to run through maintenance cycle
        for _ in range(50):
            if backend._shutting_down:
                break
            await asyncio.sleep(0.05)
        await asyncio.sleep(0.1)

        mock_redis.xtrim.assert_called()
        # Verify approximate trimming to ~100
        call_args = mock_redis.xtrim.call_args
        assert call_args is not None
        # xtrim(stream_key, maxlen=100, approximate=True)
        assert call_args.kwargs.get('maxlen', call_args[1].get('maxlen', None)) == 100


# ===========================================================================
# 4. Max-retries via delivery count
# ===========================================================================

class TestMaxRetriesEnforcement:
    """Messages exceeding max_retries must be discarded, not retried forever."""

    @pytest.mark.asyncio
    async def test_message_discarded_after_max_retries(
        self, config, mock_redis, mock_graphiti
    ):
        """Message with delivery count > max_retries is ACK'd + DEL'd without processing."""
        backend = RedisStreamsBackend(config=config)
        backend._redis = mock_redis
        backend._graphiti_client = mock_graphiti

        msg_id = '1000-0'
        msg_data = {
            'group_id': 'main',
            'name': 'Zombie Episode',
            'content': 'This will never process',
            'source_description': 'test',
            'episode_type': 'text',
            'uuid': 'zombie-uuid',
            'retry_count': '0',
        }

        # Phase 1 returns pending message with high delivery count
        phase1_called = False
        async def fake_xreadgroup(*args, **kwargs):
            nonlocal phase1_called
            streams = kwargs.get('streams', {})
            stream_key = list(streams.keys())[0]
            stream_id = streams[stream_key]

            if stream_id == '0' and not phase1_called:
                # Phase 1: return pending message
                phase1_called = True
                return [(stream_key, [(msg_id, msg_data)])]
            # After discard, shutdown
            backend._shutting_down = True
            return []

        mock_redis.xreadgroup = fake_xreadgroup

        # xpending_range returns delivery count > max_retries (3)
        mock_redis.xpending_range = AsyncMock(return_value=[{
            'message_id': msg_id,
            'consumer': 'worker_0',
            'time_since_delivered': 120000,
            'times_delivered': 5,  # > max_retries=3
        }])

        await backend._ensure_worker_running('main')
        for _ in range(50):
            if backend._shutting_down:
                break
            await asyncio.sleep(0.05)
        await asyncio.sleep(0.1)

        # Message should be ACK'd and DEL'd
        mock_redis.xack.assert_called()
        mock_redis.xdel.assert_called()
        # But NOT processed through graphiti
        mock_graphiti.add_episode.assert_not_called()

    @pytest.mark.asyncio
    async def test_message_within_retry_limit_is_processed(
        self, config, mock_redis, mock_graphiti
    ):
        """Message with delivery count <= max_retries proceeds to processing."""
        backend = RedisStreamsBackend(config=config)
        backend._redis = mock_redis
        backend._graphiti_client = mock_graphiti

        msg_id = '1000-0'
        msg_data = {
            'group_id': 'main',
            'name': 'Retryable Episode',
            'content': 'This should be processed',
            'source_description': 'test',
            'episode_type': 'text',
            'uuid': 'retry-uuid',
            'retry_count': '0',
        }

        phase1_called = False
        async def fake_xreadgroup(*args, **kwargs):
            nonlocal phase1_called
            streams = kwargs.get('streams', {})
            stream_key = list(streams.keys())[0]
            stream_id = streams[stream_key]

            if stream_id == '0' and not phase1_called:
                phase1_called = True
                return [(stream_key, [(msg_id, msg_data)])]
            backend._shutting_down = True
            return []

        mock_redis.xreadgroup = fake_xreadgroup

        # delivery count within limit
        mock_redis.xpending_range = AsyncMock(return_value=[{
            'message_id': msg_id,
            'consumer': 'worker_0',
            'time_since_delivered': 5000,
            'times_delivered': 2,  # <= max_retries=3
        }])

        await backend._ensure_worker_running('main')
        for _ in range(50):
            if backend._shutting_down:
                break
            await asyncio.sleep(0.05)
        await asyncio.sleep(0.1)

        # Message SHOULD be processed
        mock_graphiti.add_episode.assert_called_once()

    @pytest.mark.asyncio
    async def test_xpending_range_failure_does_not_block(
        self, config, mock_redis, mock_graphiti
    ):
        """If xpending_range fails, message proceeds to normal processing."""
        backend = RedisStreamsBackend(config=config)
        backend._redis = mock_redis
        backend._graphiti_client = mock_graphiti

        msg_id = '1000-0'
        msg_data = {
            'group_id': 'main',
            'name': 'Fallback Episode',
            'content': 'Process despite xpending error',
            'source_description': 'test',
            'episode_type': 'text',
            'uuid': 'fallback-uuid',
            'retry_count': '0',
        }

        phase1_called = False
        async def fake_xreadgroup(*args, **kwargs):
            nonlocal phase1_called
            streams = kwargs.get('streams', {})
            stream_key = list(streams.keys())[0]
            stream_id = streams[stream_key]

            if stream_id == '0' and not phase1_called:
                phase1_called = True
                return [(stream_key, [(msg_id, msg_data)])]
            backend._shutting_down = True
            return []

        mock_redis.xreadgroup = fake_xreadgroup
        mock_redis.xpending_range = AsyncMock(side_effect=Exception('Redis error'))

        await backend._ensure_worker_running('main')
        for _ in range(50):
            if backend._shutting_down:
                break
            await asyncio.sleep(0.05)
        await asyncio.sleep(0.1)

        # Message should still be processed (graceful degradation)
        mock_graphiti.add_episode.assert_called_once()
