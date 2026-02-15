"""Smoke tests for MCP tools.

These tests verify:
1. All imports work (catches missing/renamed functions)
2. All tools are callable with mocked dependencies (catches attribute errors)
3. All tools return ErrorResponse when service is None
4. group_id routing calls driver.clone() for FalkorDB
"""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from graphiti_core.driver.driver import GraphProvider
from graphiti_core.edges import EntityEdge
from graphiti_core.nodes import EntityNode, EpisodicNode


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_driver():
    driver = MagicMock()
    driver.provider = GraphProvider.NEO4J
    driver.session = MagicMock()
    driver.execute_query = AsyncMock()
    driver.clone = MagicMock(return_value=MagicMock())
    return driver


@pytest.fixture
def mock_falkor_driver():
    driver = MagicMock()
    driver.provider = GraphProvider.FALKORDB
    driver.session = MagicMock()
    driver.execute_query = AsyncMock()
    cloned = MagicMock()
    cloned.provider = GraphProvider.FALKORDB
    cloned.session = MagicMock()
    cloned.execute_query = AsyncMock()
    driver.clone = MagicMock(return_value=cloned)
    return driver


@pytest.fixture
def mock_embedder():
    embedder = AsyncMock()
    embedder.create = AsyncMock(return_value=[0.0] * 1024)
    return embedder


@pytest.fixture
def mock_entity_node():
    node = MagicMock(spec=EntityNode)
    node.uuid = 'test-uuid-node'
    node.name = 'TestEntity'
    node.summary = 'A test entity'
    node.group_id = 'test-group'
    node.labels = ['Entity']
    node.attributes = {}
    node.created_at = datetime(2025, 1, 1, tzinfo=timezone.utc)
    node.name_embedding = [0.0] * 1024
    node.model_dump = MagicMock(
        return_value={
            'uuid': 'test-uuid-node',
            'name': 'TestEntity',
            'summary': 'A test entity',
            'group_id': 'test-group',
            'labels': ['Entity'],
            'attributes': {},
            'created_at': '2025-01-01T00:00:00+00:00',
        }
    )
    node.save = AsyncMock()
    node.delete = AsyncMock()
    node.generate_name_embedding = AsyncMock()
    node.generate_summary_embedding = AsyncMock()
    return node


@pytest.fixture
def mock_entity_edge():
    edge = MagicMock(spec=EntityEdge)
    edge.uuid = 'test-uuid-edge'
    edge.name = 'RELATES_TO'
    edge.fact = 'Test relates to something'
    edge.group_id = 'test-group'
    edge.source_node_uuid = 'source-uuid'
    edge.target_node_uuid = 'target-uuid'
    edge.created_at = datetime(2025, 1, 1, tzinfo=timezone.utc)
    edge.fact_embedding = [0.0] * 1024
    edge.attributes = {}
    edge.model_dump = MagicMock(
        return_value={
            'uuid': 'test-uuid-edge',
            'name': 'RELATES_TO',
            'fact': 'Test relates to something',
            'group_id': 'test-group',
            'source_node_uuid': 'source-uuid',
            'target_node_uuid': 'target-uuid',
            'created_at': '2025-01-01T00:00:00+00:00',
            'attributes': {},
        }
    )
    edge.save = AsyncMock()
    edge.delete = AsyncMock()
    edge.generate_embedding = AsyncMock()
    return edge


@pytest.fixture
def mock_episodic_node():
    node = MagicMock(spec=EpisodicNode)
    node.uuid = 'test-uuid-episode'
    node.name = 'TestEpisode'
    node.content = 'Some episode content'
    node.group_id = 'test-group'
    node.created_at = datetime(2025, 1, 1, tzinfo=timezone.utc)
    node.source = MagicMock(value='text')
    node.source_description = 'test source'
    node.save = AsyncMock()
    node.delete = AsyncMock()
    return node


@pytest.fixture
def mock_graphiti_client(mock_driver, mock_embedder, mock_entity_node, mock_entity_edge):
    client = AsyncMock()
    client.driver = mock_driver
    client.embedder = mock_embedder
    client.get_entity = AsyncMock(return_value=mock_entity_node)
    client.get_edge = AsyncMock(return_value=mock_entity_edge)
    client.remove_entity = AsyncMock()
    client.remove_edge = AsyncMock()
    client.get_episode = AsyncMock()
    client.search_ = AsyncMock()
    client.search = AsyncMock(return_value=[])
    client.create_entity = AsyncMock(return_value=mock_entity_node)
    client.create_edge = AsyncMock(return_value=mock_entity_edge)
    client.update_entity = AsyncMock(return_value=mock_entity_node)
    client.update_edge = AsyncMock(return_value=mock_entity_edge)
    client.get_entities_by_group_id = AsyncMock(return_value=[])
    client.get_edges_by_group_id = AsyncMock(return_value=[])
    return client


@pytest.fixture
def mock_graphiti_service(mock_graphiti_client):
    service = MagicMock()
    service.get_client = AsyncMock(return_value=mock_graphiti_client)
    service.entity_types = None
    service.config = MagicMock()
    service.config.database.provider = 'neo4j'
    return service


@pytest.fixture
def mock_queue_service():
    qs = AsyncMock()
    qs.add_episode = AsyncMock()
    return qs


@pytest.fixture
def mock_entity_type_service():
    ets = AsyncMock()
    ets.get_all = AsyncMock(return_value=[])
    return ets


@pytest.fixture
def mock_config():
    cfg = MagicMock()
    cfg.graphiti.group_id = 'test-group'
    return cfg


# ---------------------------------------------------------------------------
# Import test
# ---------------------------------------------------------------------------


class TestImports:
    """Verify the module imports without errors."""

    def test_module_imports(self):
        import graphiti_mcp_server  # noqa: F401

    def test_helper_importable(self):
        from graphiti_mcp_server import _get_driver  # noqa: F401

    def test_tool_functions_exist(self):
        import graphiti_mcp_server as m

        tools = [
            'add_memory',
            'create_entity_node',
            'create_entity_edge',
            'search_nodes',
            'search_memory_facts',
            'get_episodes',
            'get_entity_node',
            'get_entity_edge',
            'get_entity_edges_by_node',
            'get_entity_types',
            'get_status',
            'list_nodes',
            'list_edges',
            'update_entity_node',
            'update_entity_edge',
            'delete_entity_node',
            'delete_entity_edge',
            'delete_episode',
            'clear_graph',
        ]
        for tool_name in tools:
            assert hasattr(m, tool_name), f'Missing tool function: {tool_name}'


# ---------------------------------------------------------------------------
# _get_driver unit tests
# ---------------------------------------------------------------------------


class TestGetDriver:
    def test_neo4j_returns_original(self, mock_driver):
        from graphiti_mcp_server import _get_driver

        client = MagicMock()
        client.driver = mock_driver
        result = _get_driver(client, group_id='mygroup')
        assert result is mock_driver
        mock_driver.clone.assert_not_called()

    def test_falkordb_clones_with_group_id(self, mock_falkor_driver):
        from graphiti_mcp_server import _get_driver

        client = MagicMock()
        client.driver = mock_falkor_driver
        result = _get_driver(client, group_id='TankWars')
        assert result is mock_falkor_driver.clone.return_value
        mock_falkor_driver.clone.assert_called_once_with(database='TankWars')

    def test_falkordb_no_group_id_returns_original(self, mock_falkor_driver):
        from graphiti_mcp_server import _get_driver

        client = MagicMock()
        client.driver = mock_falkor_driver
        result = _get_driver(client, group_id=None)
        assert result is mock_falkor_driver
        mock_falkor_driver.clone.assert_not_called()

    def test_no_group_id_returns_original(self, mock_driver):
        from graphiti_mcp_server import _get_driver

        client = MagicMock()
        client.driver = mock_driver
        result = _get_driver(client)
        assert result is mock_driver
        mock_driver.clone.assert_not_called()


# ---------------------------------------------------------------------------
# Error-path tests: service=None -> ErrorResponse
# ---------------------------------------------------------------------------


class TestServiceNoneErrors:
    """Every tool must return ErrorResponse when graphiti_service is None."""

    @pytest.fixture(autouse=True)
    def _clear_globals(self):
        import graphiti_mcp_server as m

        orig_service = m.graphiti_service
        orig_queue = m.queue_service
        orig_ets = m.entity_type_service
        orig_config = getattr(m, 'config', None)
        m.graphiti_service = None
        m.queue_service = None
        m.entity_type_service = None
        yield
        m.graphiti_service = orig_service
        m.queue_service = orig_queue
        m.entity_type_service = orig_ets
        if orig_config is not None:
            m.config = orig_config

    async def test_add_memory_error(self):
        from graphiti_mcp_server import add_memory

        result = await add_memory(name='test', episode_body='test')
        assert 'error' in result

    async def test_create_entity_node_error(self):
        from graphiti_mcp_server import create_entity_node

        result = await create_entity_node(name='TestNode')
        assert 'error' in result

    async def test_create_entity_edge_error(self):
        from graphiti_mcp_server import create_entity_edge

        result = await create_entity_edge(
            source_node_uuid='src', target_node_uuid='tgt', name='REL', fact='test'
        )
        assert 'error' in result

    async def test_search_nodes_error(self):
        from graphiti_mcp_server import search_nodes

        result = await search_nodes(query='test')
        assert 'error' in result

    async def test_search_memory_facts_error(self):
        from graphiti_mcp_server import search_memory_facts

        result = await search_memory_facts(query='test')
        assert 'error' in result

    async def test_get_episodes_error(self):
        from graphiti_mcp_server import get_episodes

        result = await get_episodes()
        assert 'error' in result

    async def test_get_entity_node_error(self):
        from graphiti_mcp_server import get_entity_node

        result = await get_entity_node(uuid='some-uuid')
        assert 'error' in result

    async def test_get_entity_edge_error(self):
        from graphiti_mcp_server import get_entity_edge

        result = await get_entity_edge(uuid='some-uuid')
        assert 'error' in result

    async def test_get_entity_edges_by_node_error(self):
        from graphiti_mcp_server import get_entity_edges_by_node

        result = await get_entity_edges_by_node(node_uuid='some-uuid')
        assert 'error' in result

    async def test_get_entity_types_error(self):
        from graphiti_mcp_server import get_entity_types

        result = await get_entity_types()
        assert 'error' in result

    async def test_get_status_error(self):
        from graphiti_mcp_server import get_status

        result = await get_status()
        assert result['status'] == 'error'

    async def test_list_nodes_error(self):
        from graphiti_mcp_server import list_nodes

        result = await list_nodes()
        assert 'error' in result

    async def test_list_edges_error(self):
        from graphiti_mcp_server import list_edges

        result = await list_edges()
        assert 'error' in result

    async def test_update_entity_node_error(self):
        from graphiti_mcp_server import update_entity_node

        result = await update_entity_node(uuid='some-uuid', name='New')
        assert 'error' in result

    async def test_update_entity_edge_error(self):
        from graphiti_mcp_server import update_entity_edge

        result = await update_entity_edge(uuid='some-uuid', fact='New fact')
        assert 'error' in result

    async def test_delete_entity_node_error(self):
        from graphiti_mcp_server import delete_entity_node

        result = await delete_entity_node(uuid='some-uuid')
        assert 'error' in result

    async def test_delete_entity_edge_error(self):
        from graphiti_mcp_server import delete_entity_edge

        result = await delete_entity_edge(uuid='some-uuid')
        assert 'error' in result

    async def test_delete_episode_error(self):
        from graphiti_mcp_server import delete_episode

        result = await delete_episode(uuid='some-uuid')
        assert 'error' in result

    async def test_clear_graph_error(self):
        from graphiti_mcp_server import clear_graph

        result = await clear_graph()
        assert 'error' in result


# ---------------------------------------------------------------------------
# Happy-path tests with mocked services
# ---------------------------------------------------------------------------


class TestHappyPath:
    """Test each tool returns a non-error response with mocked dependencies."""

    @pytest.fixture(autouse=True)
    def _setup_globals(
        self,
        mock_graphiti_service,
        mock_queue_service,
        mock_entity_type_service,
        mock_config,
    ):
        import graphiti_mcp_server as m

        orig_service = m.graphiti_service
        orig_queue = m.queue_service
        orig_ets = m.entity_type_service
        orig_config = getattr(m, 'config', None)
        m.graphiti_service = mock_graphiti_service
        m.queue_service = mock_queue_service
        m.entity_type_service = mock_entity_type_service
        m.config = mock_config
        yield
        m.graphiti_service = orig_service
        m.queue_service = orig_queue
        m.entity_type_service = orig_ets
        if orig_config is not None:
            m.config = orig_config

    async def test_add_memory(self):
        from graphiti_mcp_server import add_memory

        result = await add_memory(name='test', episode_body='test body')
        assert 'error' not in result
        assert 'message' in result

    async def test_create_entity_node(self, mock_graphiti_client):
        from graphiti_mcp_server import create_entity_node

        result = await create_entity_node(name='TestNode', entity_type='Weapon', summary='A sword')
        assert 'error' not in result
        assert result['name'] == 'TestEntity'
        mock_graphiti_client.create_entity.assert_awaited_once()

    async def test_create_entity_edge(self, mock_graphiti_client):
        from graphiti_mcp_server import create_entity_edge

        result = await create_entity_edge(
            source_node_uuid='src-uuid',
            target_node_uuid='tgt-uuid',
            name='HAS_WEAPON',
            fact='Player has a sword',
        )
        assert 'error' not in result
        assert result['name'] == 'RELATES_TO'
        mock_graphiti_client.create_edge.assert_awaited_once()

    async def test_create_entity_node_default_group_id(self, mock_graphiti_client):
        from graphiti_mcp_server import create_entity_node

        await create_entity_node(name='TestNode')
        mock_graphiti_client.create_entity.assert_awaited_once()
        call_kwargs = mock_graphiti_client.create_entity.call_args.kwargs
        assert call_kwargs['group_id'] == 'test-group'

    async def test_create_entity_edge_default_group_id(self, mock_graphiti_client):
        from graphiti_mcp_server import create_entity_edge

        await create_entity_edge(
            source_node_uuid='src', target_node_uuid='tgt', name='REL', fact='test'
        )
        mock_graphiti_client.create_edge.assert_awaited_once()
        call_kwargs = mock_graphiti_client.create_edge.call_args.kwargs
        assert call_kwargs['group_id'] == 'test-group'

    async def test_search_nodes(self, mock_graphiti_client):
        from graphiti_mcp_server import search_nodes

        mock_results = MagicMock()
        mock_results.nodes = []
        mock_graphiti_client.search_ = AsyncMock(return_value=mock_results)

        result = await search_nodes(query='test')
        assert 'error' not in result

    async def test_search_memory_facts(self):
        from graphiti_mcp_server import search_memory_facts

        result = await search_memory_facts(query='test')
        assert 'error' not in result

    async def test_get_episodes(self, mock_episodic_node, mock_graphiti_client):
        from graphiti_mcp_server import get_episodes

        with patch.object(
            EpisodicNode,
            'get_by_group_ids',
            new_callable=AsyncMock,
            return_value=[mock_episodic_node],
        ):
            result = await get_episodes()
        assert 'error' not in result

    @patch.object(EntityNode, 'get_by_uuid', new_callable=AsyncMock)
    async def test_get_entity_node(self, mock_get, mock_entity_node):
        from graphiti_mcp_server import get_entity_node

        mock_get.return_value = mock_entity_node
        result = await get_entity_node(uuid='test-uuid-node')
        assert 'error' not in result

    @patch.object(EntityEdge, 'get_by_uuid', new_callable=AsyncMock)
    async def test_get_entity_edge(self, mock_get, mock_entity_edge):
        from graphiti_mcp_server import get_entity_edge

        mock_get.return_value = mock_entity_edge
        result = await get_entity_edge(uuid='test-uuid-edge')
        assert 'error' not in result

    @patch.object(EntityEdge, 'get_by_node_uuid', new_callable=AsyncMock)
    async def test_get_entity_edges_by_node(self, mock_get, mock_entity_edge):
        from graphiti_mcp_server import get_entity_edges_by_node

        mock_get.return_value = [mock_entity_edge]
        result = await get_entity_edges_by_node(node_uuid='test-uuid-node')
        assert 'error' not in result

    async def test_get_entity_types(self):
        from graphiti_mcp_server import get_entity_types

        result = await get_entity_types()
        assert 'error' not in result

    async def test_get_status(self, mock_graphiti_client):
        from graphiti_mcp_server import get_status

        # Mock the session context manager with proper async iteration
        mock_session = AsyncMock()
        mock_result = MagicMock()

        # Create a proper async iterator
        async def _empty_aiter():
            return
            yield  # pragma: no cover

        mock_result.__aiter__ = lambda self: _empty_aiter()
        mock_session.run = AsyncMock(return_value=mock_result)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)
        mock_graphiti_client.driver.session.return_value = mock_session

        result = await get_status()
        assert result['status'] == 'ok'

    async def test_list_nodes(self, mock_graphiti_client):
        from graphiti_mcp_server import list_nodes

        mock_graphiti_client.get_entities_by_group_id = AsyncMock(return_value=[])
        result = await list_nodes()
        assert 'error' not in result

    async def test_list_edges(self, mock_graphiti_client):
        from graphiti_mcp_server import list_edges

        mock_graphiti_client.get_edges_by_group_id = AsyncMock(return_value=[])
        result = await list_edges()
        assert 'error' not in result

    async def test_update_entity_node(self, mock_graphiti_client):
        from graphiti_mcp_server import update_entity_node

        result = await update_entity_node(uuid='test-uuid-node', name='NewName')
        assert 'error' not in result
        mock_graphiti_client.update_entity.assert_awaited_once()

    async def test_update_entity_edge(self, mock_graphiti_client):
        from graphiti_mcp_server import update_entity_edge

        result = await update_entity_edge(uuid='test-uuid-edge', fact='Updated fact')
        assert 'error' not in result
        mock_graphiti_client.update_edge.assert_awaited_once()

    @patch.object(EntityEdge, 'get_by_node_uuid', new_callable=AsyncMock)
    @patch.object(EntityNode, 'get_by_uuid', new_callable=AsyncMock)
    async def test_delete_entity_node(self, mock_get_node, mock_get_edges, mock_entity_node):
        from graphiti_mcp_server import delete_entity_node

        mock_get_node.return_value = mock_entity_node
        mock_get_edges.return_value = []
        result = await delete_entity_node(uuid='test-uuid-node')
        assert 'error' not in result
        assert 'message' in result

    @patch.object(EntityEdge, 'get_by_uuid', new_callable=AsyncMock)
    async def test_delete_entity_edge(self, mock_get, mock_entity_edge):
        from graphiti_mcp_server import delete_entity_edge

        mock_get.return_value = mock_entity_edge
        result = await delete_entity_edge(uuid='test-uuid-edge')
        assert 'error' not in result
        assert 'message' in result

    @patch.object(EpisodicNode, 'get_by_uuid', new_callable=AsyncMock)
    async def test_delete_episode(self, mock_get, mock_episodic_node):
        from graphiti_mcp_server import delete_episode

        mock_get.return_value = mock_episodic_node
        result = await delete_episode(uuid='test-uuid-episode')
        assert 'error' not in result
        assert 'message' in result

    @patch('graphiti_mcp_server.clear_data', new_callable=AsyncMock)
    async def test_clear_graph(self, mock_clear):
        from graphiti_mcp_server import clear_graph

        result = await clear_graph(group_ids=['test-group'])
        assert 'error' not in result
        assert 'message' in result


# ---------------------------------------------------------------------------
# FalkorDB group_id routing tests
# ---------------------------------------------------------------------------


class TestFalkorDBGroupIdRouting:
    """Verify that CRUD tools clone the driver when group_id is given and provider=FALKORDB."""

    @pytest.fixture(autouse=True)
    def _setup_globals(
        self,
        mock_graphiti_service,
        mock_queue_service,
        mock_entity_type_service,
        mock_config,
        mock_falkor_driver,
        mock_graphiti_client,
    ):
        import graphiti_mcp_server as m

        # Switch to FalkorDB driver
        mock_graphiti_client.driver = mock_falkor_driver

        orig_service = m.graphiti_service
        orig_queue = m.queue_service
        orig_ets = m.entity_type_service
        orig_config = getattr(m, 'config', None)
        m.graphiti_service = mock_graphiti_service
        m.queue_service = mock_queue_service
        m.entity_type_service = mock_entity_type_service
        m.config = mock_config
        yield
        m.graphiti_service = orig_service
        m.queue_service = orig_queue
        m.entity_type_service = orig_ets
        if orig_config is not None:
            m.config = orig_config

    @patch.object(EntityNode, 'get_by_uuid', new_callable=AsyncMock)
    async def test_get_entity_node_clones(
        self, mock_get, mock_entity_node, mock_falkor_driver
    ):
        from graphiti_mcp_server import get_entity_node

        mock_get.return_value = mock_entity_node
        await get_entity_node(uuid='test-uuid', group_id='TankWars')
        mock_falkor_driver.clone.assert_called_once_with(database='TankWars')

    @patch.object(EntityEdge, 'get_by_uuid', new_callable=AsyncMock)
    async def test_get_entity_edge_clones(
        self, mock_get, mock_entity_edge, mock_falkor_driver
    ):
        from graphiti_mcp_server import get_entity_edge

        mock_get.return_value = mock_entity_edge
        await get_entity_edge(uuid='test-uuid', group_id='TankWars')
        mock_falkor_driver.clone.assert_called_once_with(database='TankWars')

    @patch.object(EntityEdge, 'get_by_node_uuid', new_callable=AsyncMock)
    async def test_get_entity_edges_by_node_clones(
        self, mock_get, mock_entity_edge, mock_falkor_driver
    ):
        from graphiti_mcp_server import get_entity_edges_by_node

        mock_get.return_value = [mock_entity_edge]
        await get_entity_edges_by_node(node_uuid='test-uuid', group_id='TankWars')
        mock_falkor_driver.clone.assert_called_once_with(database='TankWars')

    async def test_update_entity_node_passes_group_id(self, mock_graphiti_client):
        from graphiti_mcp_server import update_entity_node

        await update_entity_node(uuid='test-uuid', name='NewName', group_id='TankWars')
        call_kwargs = mock_graphiti_client.update_entity.call_args.kwargs
        assert call_kwargs['group_id'] == 'TankWars'

    async def test_update_entity_edge_passes_group_id(self, mock_graphiti_client):
        from graphiti_mcp_server import update_entity_edge

        await update_entity_edge(uuid='test-uuid', fact='New fact', group_id='TankWars')
        call_kwargs = mock_graphiti_client.update_edge.call_args.kwargs
        assert call_kwargs['group_id'] == 'TankWars'

    @patch.object(EntityEdge, 'get_by_node_uuid', new_callable=AsyncMock)
    @patch.object(EntityNode, 'get_by_uuid', new_callable=AsyncMock)
    async def test_delete_entity_node_clones(
        self, mock_get_node, mock_get_edges, mock_entity_node, mock_falkor_driver
    ):
        from graphiti_mcp_server import delete_entity_node

        mock_get_node.return_value = mock_entity_node
        mock_get_edges.return_value = []
        await delete_entity_node(uuid='test-uuid', group_id='TankWars')
        mock_falkor_driver.clone.assert_called_once_with(database='TankWars')

    @patch.object(EntityEdge, 'get_by_uuid', new_callable=AsyncMock)
    async def test_delete_entity_edge_clones(
        self, mock_get, mock_entity_edge, mock_falkor_driver
    ):
        from graphiti_mcp_server import delete_entity_edge

        mock_get.return_value = mock_entity_edge
        await delete_entity_edge(uuid='test-uuid', group_id='TankWars')
        mock_falkor_driver.clone.assert_called_once_with(database='TankWars')

    @patch.object(EpisodicNode, 'get_by_uuid', new_callable=AsyncMock)
    async def test_delete_episode_clones(
        self, mock_get, mock_episodic_node, mock_falkor_driver
    ):
        from graphiti_mcp_server import delete_episode

        mock_get.return_value = mock_episodic_node
        await delete_episode(uuid='test-uuid', group_id='TankWars')
        mock_falkor_driver.clone.assert_called_once_with(database='TankWars')

    @patch.object(EntityNode, 'get_by_uuid', new_callable=AsyncMock)
    async def test_no_clone_without_group_id(
        self, mock_get, mock_entity_node, mock_falkor_driver
    ):
        """FalkorDB driver should NOT be cloned when group_id is not provided."""
        from graphiti_mcp_server import get_entity_node

        mock_get.return_value = mock_entity_node
        await get_entity_node(uuid='test-uuid')
        mock_falkor_driver.clone.assert_not_called()
