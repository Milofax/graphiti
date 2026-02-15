"""Tests for Redis Streams queue robustness.

Verifies:
1. Stable consumer name (env var instead of hostname+pid)
2. XDEL after XACK (stream cleanup per message)
3. Periodic XTRIM with maxlen=5000
4. DLQ: failed messages moved to DLQ instead of discarded
5. Single stream architecture (no per-group streams)
6. Legacy drainer: throttled processing of migrated messages
"""

import asyncio
import json
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import redis.asyncio as aioredis

from services.queue_redis import (
    DLQ_KEY,
    LEGACY_KEY,
    STREAM_KEY,
    THROTTLE_HOUR_END,
    THROTTLE_HOUR_START,
    XTRIM_MAXLEN,
    RedisQueueConfig,
    RedisStreamsBackend,
)


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
    r.xlen = AsyncMock(return_value=0)
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
        """Consumer name must NOT contain PID."""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop('GRAPHITI_CONSUMER_NAME', None)
            backend = RedisStreamsBackend(config=config)
            assert str(os.getpid()) not in backend._consumer_name


# ===========================================================================
# 2. XDEL after XACK (single stream)
# ===========================================================================

class TestXdelAfterXack:
    """Processed messages must be deleted from stream after acknowledgment."""

    @pytest.mark.asyncio
    async def test_xdel_called_after_successful_processing(
        self, config, mock_redis, mock_graphiti
    ):
        """After XACK, XDEL must be called on the single stream."""
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
        mock_redis.xack.assert_called_once_with(STREAM_KEY, config.consumer_group, '1000-0')
        mock_redis.xdel.assert_called_once_with(STREAM_KEY, '1000-0')

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
        """If XDEL fails, processing should still return True."""
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
# 3. Periodic XTRIM with maxlen=5000
# ===========================================================================

class TestPeriodicXtrim:
    """Maintenance cycle must XTRIM the single stream."""

    @pytest.mark.asyncio
    async def test_xtrim_called_during_maintenance(self, config, mock_redis, mock_graphiti):
        """XTRIM should be called with maxlen=5000 during maintenance."""
        backend = RedisStreamsBackend(config=config)
        backend._redis = mock_redis
        backend._graphiti_client = mock_graphiti

        iteration = 0
        RECLAIM_EVERY = 12

        async def fake_xreadgroup(*args, **kwargs):
            nonlocal iteration
            iteration += 1
            if iteration > RECLAIM_EVERY * 2 + 4:
                backend._shutting_down = True
            return []

        mock_redis.xreadgroup = fake_xreadgroup

        backend._start_worker()
        for _ in range(50):
            if backend._shutting_down:
                break
            await asyncio.sleep(0.05)
        await asyncio.sleep(0.1)

        mock_redis.xtrim.assert_called()
        call_args = mock_redis.xtrim.call_args
        assert call_args is not None
        # Verify stream key is the single stream
        assert call_args[0][0] == STREAM_KEY
        # Verify maxlen=5000
        assert call_args.kwargs.get('maxlen', call_args[1].get('maxlen', None)) == XTRIM_MAXLEN


# ===========================================================================
# 4. DLQ instead of discard
# ===========================================================================

class TestDLQEnforcement:
    """Messages exceeding max_retries must be moved to DLQ, not discarded."""

    @pytest.mark.asyncio
    async def test_message_moved_to_dlq_after_max_retries(
        self, config, mock_redis, mock_graphiti
    ):
        """Message with delivery count > max_retries is moved to DLQ."""
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

        mock_redis.xpending_range = AsyncMock(return_value=[{
            'message_id': msg_id,
            'consumer': 'worker_0',
            'time_since_delivered': 120000,
            'times_delivered': 5,  # > max_retries=3
        }])

        backend._start_worker()
        for _ in range(50):
            if backend._shutting_down:
                break
            await asyncio.sleep(0.05)
        await asyncio.sleep(0.1)

        # Message should be added to DLQ
        dlq_calls = [c for c in mock_redis.xadd.call_args_list if c[0][0] == DLQ_KEY]
        assert len(dlq_calls) == 1
        dlq_data = dlq_calls[0][0][1]
        assert dlq_data['group_id'] == 'main'
        assert dlq_data['name'] == 'Zombie Episode'
        assert dlq_data['original_stream_id'] == msg_id
        assert dlq_data['times_delivered'] == '5'
        assert 'moved_to_dlq_at' in dlq_data

        # Message should be ACK'd from main stream
        mock_redis.xack.assert_called()
        # NOT processed through graphiti
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

        mock_redis.xpending_range = AsyncMock(return_value=[{
            'message_id': msg_id,
            'consumer': 'worker_0',
            'time_since_delivered': 5000,
            'times_delivered': 2,  # <= max_retries=3
        }])

        backend._start_worker()
        for _ in range(50):
            if backend._shutting_down:
                break
            await asyncio.sleep(0.05)
        await asyncio.sleep(0.1)

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

        backend._start_worker()
        for _ in range(50):
            if backend._shutting_down:
                break
            await asyncio.sleep(0.05)
        await asyncio.sleep(0.1)

        mock_graphiti.add_episode.assert_called_once()


# ===========================================================================
# 5. Single stream architecture
# ===========================================================================

class TestSingleStreamArchitecture:
    """All operations use the single stream, not per-group streams."""

    @pytest.mark.asyncio
    async def test_add_episode_uses_single_stream(self, config, mock_redis, mock_graphiti):
        """add_episode always writes to STREAM_KEY regardless of group_id."""
        backend = RedisStreamsBackend(config=config)
        backend._redis = mock_redis
        backend._graphiti_client = mock_graphiti
        backend._shutting_down = True

        await backend.add_episode(
            group_id='some-random-group',
            name='Test',
            content='Content',
            source_description='test',
            episode_type='text',
            entity_types=None,
            uuid='uuid-1',
        )

        call_args = mock_redis.xadd.call_args
        assert call_args[0][0] == STREAM_KEY

    @pytest.mark.asyncio
    async def test_process_message_acks_on_single_stream(
        self, config, mock_redis, mock_graphiti
    ):
        """_process_message ACKs on the single stream."""
        backend = RedisStreamsBackend(config=config)
        backend._redis = mock_redis
        backend._graphiti_client = mock_graphiti

        await backend._process_message('any-group', '1000-0', {
            'group_id': 'any-group',
            'name': 'Test',
            'content': 'Content',
            'source_description': 'test',
            'episode_type': 'text',
            'uuid': 'uuid-1',
            'retry_count': '0',
        })

        mock_redis.xack.assert_called_once_with(
            STREAM_KEY, config.consumer_group, '1000-0'
        )

    def test_is_worker_running_returns_bool(self, config):
        """is_worker_running returns a simple bool, not per-group dict."""
        backend = RedisStreamsBackend(config=config)
        assert backend.is_worker_running() is False
        backend._worker_running = True
        assert backend.is_worker_running() is True

    def test_get_queue_size_returns_zero(self, config):
        """get_queue_size returns 0 (approximate, async needed for accuracy)."""
        backend = RedisStreamsBackend(config=config)
        assert backend.get_queue_size() == 0


# ===========================================================================
# 6. Legacy drainer
# ===========================================================================

class TestLegacyDrainer:
    """Legacy messages are migrated to a separate list and drained with throttle."""

    def test_default_throttle_is_zero(self):
        """Default throttle_seconds must be 0.0 (no drainer)."""
        config = RedisQueueConfig()
        assert config.throttle_seconds == 0.0

    def test_throttle_config_stored(self):
        """throttle_seconds value is stored in config."""
        config = RedisQueueConfig(throttle_seconds=420.0)
        assert config.throttle_seconds == 420.0

    def test_time_window_constants(self):
        """Time window constants must be 9:00-22:00."""
        assert THROTTLE_HOUR_START == 9
        assert THROTTLE_HOUR_END == 22

    def test_legacy_key_constant(self):
        """LEGACY_KEY must be 'graphiti:queue:legacy'."""
        assert LEGACY_KEY == 'graphiti:queue:legacy'

    @pytest.mark.asyncio
    async def test_migration_uses_rpush(self, mock_redis, mock_graphiti):
        """Migration must write legacy messages to LEGACY_KEY via RPUSH, not XADD to main stream."""
        config = RedisQueueConfig(throttle_seconds=420.0)
        backend = RedisStreamsBackend(config=config)
        backend._redis = mock_redis
        backend._graphiti_client = mock_graphiti

        # Set up mock to return one legacy stream with one unprocessed message
        legacy_key = 'graphiti:queue:test-group'
        msg_data = {
            'group_id': 'test-group',
            'name': 'Legacy Episode',
            'content': 'Legacy content',
            'source_description': 'test',
            'episode_type': 'text',
            'uuid': 'legacy-uuid',
            'retry_count': '0',
        }

        class AsyncIterLegacy:
            def __init__(self):
                self._items = [legacy_key]
                self._idx = 0
            def __aiter__(self):
                return self
            async def __anext__(self):
                if self._idx >= len(self._items):
                    raise StopAsyncIteration
                val = self._items[self._idx]
                self._idx += 1
                return val

        mock_redis.scan_iter = MagicMock(return_value=AsyncIterLegacy())
        mock_redis.xrange = AsyncMock(return_value=[('1000-0', msg_data)])
        mock_redis.xinfo_groups = AsyncMock(side_effect=aioredis.ResponseError('NOGROUP'))
        mock_redis.delete = AsyncMock()
        mock_redis.rpush = AsyncMock()
        mock_redis.llen = AsyncMock(return_value=0)

        await backend._migrate_legacy_streams()

        # RPUSH to legacy list — NOT xadd to main stream
        mock_redis.rpush.assert_called_once()
        rpush_args = mock_redis.rpush.call_args[0]
        assert rpush_args[0] == LEGACY_KEY
        pushed_data = json.loads(rpush_args[1])
        assert pushed_data['group_id'] == 'test-group'
        assert pushed_data['name'] == 'Legacy Episode'

        # xadd should NOT be called for migrated messages
        # (it may be called for DLQ or other purposes, but not for migration)
        xadd_to_main = [
            c for c in mock_redis.xadd.call_args_list
            if c[0][0] == STREAM_KEY
        ]
        assert len(xadd_to_main) == 0, 'Migration should not XADD to main stream'
