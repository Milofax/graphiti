"""Tests for the queue service and Redis Streams backend.

These tests verify the persistent queue implementation using Redis Streams
with a single stream architecture.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from services.queue_redis import (
    DLQ_KEY,
    STREAM_KEY,
    XTRIM_MAXLEN,
    EpisodeMessage,
    RedisQueueConfig,
    RedisStreamsBackend,
)


class AsyncIterEmpty:
    """Async iterator that yields nothing."""
    def __aiter__(self):
        return self
    async def __anext__(self):
        raise StopAsyncIteration


class TestQueueServiceUnit:
    """Unit tests for RedisStreamsBackend using mocked Redis."""

    @pytest.fixture
    def mock_redis(self):
        """Mock Redis client for unit tests."""
        r = MagicMock()
        r.xadd = AsyncMock(return_value='1234567890-0')
        r.xreadgroup = AsyncMock(return_value=[])
        r.xack = AsyncMock(return_value=1)
        r.xdel = AsyncMock(return_value=1)
        r.xgroup_create = AsyncMock()
        r.xpending = AsyncMock(return_value={'pending': 0})
        r.xpending_range = AsyncMock(return_value=[])
        r.xautoclaim = AsyncMock(return_value=('0-0', []))
        r.xinfo_groups = AsyncMock(return_value=[])
        r.close = AsyncMock()
        r.scan_iter = MagicMock(return_value=AsyncIterEmpty())
        return r

    @pytest.fixture
    def mock_graphiti_client(self):
        """Mock Graphiti client for unit tests."""
        client = MagicMock()
        client.add_episode = AsyncMock()
        return client

    @pytest.fixture
    def backend(self, mock_redis, mock_graphiti_client):
        """Create RedisStreamsBackend instance with mocks."""
        b = RedisStreamsBackend()
        b._redis = mock_redis
        b._graphiti_client = mock_graphiti_client
        return b

    @pytest.mark.asyncio
    async def test_add_episode_calls_xadd_on_single_stream(self, backend, mock_redis):
        """Episode is added to the single stream with XADD."""
        backend._shutting_down = True

        await backend.add_episode(
            group_id='main',
            name='Test Episode',
            content='Test content',
            source_description='Test source',
            episode_type='text',
            entity_types=None,
            uuid='test-uuid-123',
        )

        mock_redis.xadd.assert_called_once()
        call_args = mock_redis.xadd.call_args
        stream_key = call_args[0][0]
        assert stream_key == STREAM_KEY

    @pytest.mark.asyncio
    async def test_add_episode_includes_group_id_in_message(self, backend, mock_redis):
        """group_id is stored as a field in the message, not in the stream key."""
        backend._shutting_down = True

        await backend.add_episode(
            group_id='Milofax-infrastructure',
            name='Test Episode',
            content='Test content',
            source_description='Test source',
            episode_type='text',
            entity_types=None,
            uuid='test-uuid-123',
        )

        call_args = mock_redis.xadd.call_args
        stream_key = call_args[0][0]
        message_data = call_args[0][1]
        assert stream_key == STREAM_KEY
        assert message_data['group_id'] == 'Milofax-infrastructure'

    @pytest.mark.asyncio
    async def test_add_episode_returns_message_id(self, backend, mock_redis):
        """XADD returns message ID."""
        backend._shutting_down = True

        result = await backend.add_episode(
            group_id='main',
            name='Test Episode',
            content='Test content',
            source_description='Test source',
            episode_type='text',
            entity_types=None,
            uuid='test-uuid-123',
        )

        assert result == '1234567890-0'

    @pytest.mark.asyncio
    async def test_add_episode_without_initialize_raises(self):
        """add_episode raises RuntimeError if not initialized."""
        backend = RedisStreamsBackend()

        with pytest.raises(RuntimeError, match='not initialized'):
            await backend.add_episode(
                group_id='main',
                name='Test',
                content='Test',
                source_description='Test',
                episode_type='text',
                entity_types=None,
                uuid='test',
            )

    def test_single_stream_key_constant(self):
        """Verify the single stream key constant."""
        assert STREAM_KEY == 'graphiti:queue'

    def test_dlq_key_constant(self):
        """Verify the DLQ key constant."""
        assert DLQ_KEY == 'graphiti:queue:dlq'

    def test_xtrim_maxlen_constant(self):
        """Verify XTRIM maxlen is 5000 for single stream."""
        assert XTRIM_MAXLEN == 5000


class TestWorkerLifecycle:
    """Tests for single worker task lifecycle management."""

    @pytest.fixture
    def mock_redis(self):
        """Mock Redis client."""
        r = MagicMock()
        r.xadd = AsyncMock(return_value='1234567890-0')
        r.xreadgroup = AsyncMock(return_value=[])
        r.xack = AsyncMock(return_value=1)
        r.xdel = AsyncMock(return_value=1)
        r.xgroup_create = AsyncMock()
        r.xautoclaim = AsyncMock(return_value=('0-0', []))
        r.xinfo_groups = AsyncMock(return_value=[])
        r.close = AsyncMock()
        r.scan_iter = MagicMock(return_value=AsyncIterEmpty())
        return r

    @pytest.fixture
    def mock_graphiti_client(self):
        """Mock Graphiti client."""
        client = MagicMock()
        client.add_episode = AsyncMock()
        return client

    @pytest.mark.asyncio
    async def test_worker_task_reference_is_stored(self, mock_redis, mock_graphiti_client):
        """Task reference is stored in _worker_task to prevent GC."""
        backend = RedisStreamsBackend()
        backend._redis = mock_redis
        backend._graphiti_client = mock_graphiti_client

        call_count = 0
        async def mock_xreadgroup(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count >= 2:
                backend._shutting_down = True
            return []

        mock_redis.xreadgroup = mock_xreadgroup
        mock_redis.xautoclaim = AsyncMock(return_value=('0-0', []))

        backend._start_worker()

        assert backend._worker_task is not None
        assert isinstance(backend._worker_task, asyncio.Task)

        # Cleanup
        backend._shutting_down = True
        await asyncio.sleep(0.1)

    @pytest.mark.asyncio
    async def test_single_worker_not_duplicated(self, mock_redis, mock_graphiti_client):
        """Calling _start_worker twice doesn't create duplicate workers."""
        backend = RedisStreamsBackend()
        backend._redis = mock_redis
        backend._graphiti_client = mock_graphiti_client
        backend._shutting_down = True  # Prevent actual processing

        backend._worker_running = True  # Simulate already running

        backend._start_worker()

        # Should not have created a task since worker is already running
        assert backend._worker_task is None

    @pytest.mark.asyncio
    async def test_shutdown_sets_flag(self, mock_redis, mock_graphiti_client):
        """Shutdown sets _shutting_down flag."""
        backend = RedisStreamsBackend()
        backend._redis = mock_redis
        backend._graphiti_client = mock_graphiti_client

        await backend.shutdown(timeout=1.0)

        assert backend._shutting_down is True


class TestEpisodeMessage:
    """Tests for EpisodeMessage dataclass."""

    def test_to_dict_serialization(self):
        msg = EpisodeMessage(
            message_id='123-0',
            group_id='main',
            name='Test Episode',
            content='Test content',
            source_description='Test source',
            episode_type='text',
            uuid='test-uuid',
            retry_count=2,
        )

        result = msg.to_dict()

        assert result['group_id'] == 'main'
        assert result['name'] == 'Test Episode'
        assert result['content'] == 'Test content'
        assert result['source_description'] == 'Test source'
        assert result['episode_type'] == 'text'
        assert result['uuid'] == 'test-uuid'
        assert result['retry_count'] == '2'

    def test_from_stream_data_deserialization(self):
        data = {
            'group_id': 'main',
            'name': 'Test Episode',
            'content': 'Test content',
            'source_description': 'Test source',
            'episode_type': 'text',
            'uuid': 'test-uuid',
            'retry_count': '3',
        }

        msg = EpisodeMessage.from_stream_data('123-0', data)

        assert msg.message_id == '123-0'
        assert msg.group_id == 'main'
        assert msg.name == 'Test Episode'
        assert msg.uuid == 'test-uuid'
        assert msg.retry_count == 3

    def test_from_stream_data_handles_empty_uuid(self):
        data = {
            'group_id': 'main',
            'name': 'Test',
            'content': 'Test',
            'source_description': 'Test',
            'episode_type': 'text',
            'uuid': '',
            'retry_count': '0',
        }

        msg = EpisodeMessage.from_stream_data('123-0', data)

        assert msg.uuid is None


class TestQueueConfig:
    """Tests for RedisQueueConfig dataclass."""

    def test_default_values(self):
        config = RedisQueueConfig()

        assert config.consumer_group == 'graphiti_workers'
        assert config.block_ms == 5000
        assert config.claim_min_idle_ms == 60000
        assert config.max_retries == 3
        assert config.shutdown_timeout == 30.0

    def test_custom_values(self):
        config = RedisQueueConfig(
            redis_url='redis://custom:6380',
            consumer_group='custom_workers',
            block_ms=10000,
        )

        assert config.redis_url == 'redis://custom:6380'
        assert config.consumer_group == 'custom_workers'
        assert config.block_ms == 10000


@pytest.mark.integration
class TestFalkorDBEscaping:
    """Tests for FalkorDB RediSearch escaping fix.

    These tests require FalkorDB/Redis connection.
    Run with: pytest -m integration
    """

    def test_build_fulltext_query_escapes_reserved_words(self):
        """Group IDs like 'main' are quoted to prevent syntax errors."""
        from graphiti_core.driver.falkordb_driver import FalkorDriver

        driver = FalkorDriver(host='localhost', port=6379)
        query = driver.build_fulltext_query('search term', ['main'])

        assert '(@group_id:"main")' in query

    def test_build_fulltext_query_escapes_hyphens(self):
        """Group IDs with hyphens are quoted."""
        from graphiti_core.driver.falkordb_driver import FalkorDriver

        driver = FalkorDriver(host='localhost', port=6379)
        query = driver.build_fulltext_query('search term', ['Milofax-infrastructure'])

        assert '(@group_id:"Milofax-infrastructure")' in query

    def test_build_fulltext_query_multiple_groups(self):
        """Multiple group IDs are all quoted and joined with pipe."""
        from graphiti_core.driver.falkordb_driver import FalkorDriver

        driver = FalkorDriver(host='localhost', port=6379)
        query = driver.build_fulltext_query('test', ['main', 'Milofax-prp'])

        assert '(@group_id:"main"|"Milofax-prp")' in query

    def test_build_fulltext_query_no_groups(self):
        """Empty group_ids results in no group filter."""
        from graphiti_core.driver.falkordb_driver import FalkorDriver

        driver = FalkorDriver(host='localhost', port=6379)
        query = driver.build_fulltext_query('test', [])

        assert '@group_id' not in query
