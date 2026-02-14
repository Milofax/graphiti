"""
Copyright 2024, Zep Software, Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Integration tests for Graphiti CRUD operations.

Tests the new CRUD methods on the Graphiti class:
- Entity: create, get, update, remove
- Edge: create, get, update, remove
- Episode: get, get_episodes_by_group_id
- Group: get_groups, rename_group, remove_group, get_graph_stats
"""

import pytest

from graphiti_core import Graphiti
from graphiti_core.errors import EdgeNotFoundError, NodeNotFoundError
from tests.helpers_test import group_id, group_id_2

pytestmark = pytest.mark.integration
pytest_plugins = ('pytest_asyncio',)


@pytest.mark.asyncio
async def test_create_entity(graph_driver, mock_embedder, mock_llm_client, mock_cross_encoder):
    """Test creating an entity."""
    graphiti = Graphiti(graph_driver=graph_driver, embedder=mock_embedder, llm_client=mock_llm_client, cross_encoder=mock_cross_encoder)

    entity = await graphiti.create_entity(
        name='CRUD Test Entity',
        group_id=group_id,
        entity_type='Person',
        summary='CRUD Test Entity summary',
        attributes={'role': 'tester'},
    )

    assert entity is not None
    assert entity.uuid is not None
    assert entity.name == 'CRUD Test Entity'
    assert entity.summary == 'CRUD Test Entity summary'
    assert entity.group_id == group_id
    assert 'Person' in entity.labels
    assert entity.attributes.get('role') == 'tester'

    # Cleanup
    await entity.delete(graph_driver)


@pytest.mark.asyncio
async def test_create_entity_invalid_type(graph_driver, mock_embedder, mock_llm_client, mock_cross_encoder):
    """Test that invalid entity_type raises ValueError."""
    graphiti = Graphiti(graph_driver=graph_driver, embedder=mock_embedder, llm_client=mock_llm_client, cross_encoder=mock_cross_encoder)

    with pytest.raises(ValueError, match='Invalid entity_type'):
        await graphiti.create_entity(
            name='CRUD Test Entity',
            group_id=group_id,
            entity_type='Invalid:Type',
        )

    with pytest.raises(ValueError, match='Invalid entity_type'):
        await graphiti.create_entity(
            name='CRUD Test Entity',
            group_id=group_id,
            entity_type='123StartWithNumber',
        )


@pytest.mark.asyncio
async def test_get_entity(graph_driver, mock_embedder, mock_llm_client, mock_cross_encoder):
    """Test retrieving an entity by UUID."""
    graphiti = Graphiti(graph_driver=graph_driver, embedder=mock_embedder, llm_client=mock_llm_client, cross_encoder=mock_cross_encoder)

    # Create
    entity = await graphiti.create_entity(
        name='CRUD Test Entity',
        group_id=group_id,
        summary='CRUD Test Entity summary',
    )

    # Get
    retrieved = await graphiti.get_entity(entity.uuid, group_id=group_id)

    assert retrieved is not None
    assert retrieved.uuid == entity.uuid
    assert retrieved.name == 'CRUD Test Entity'

    # Cleanup
    await entity.delete(graph_driver)


@pytest.mark.asyncio
async def test_update_entity_name(graph_driver, mock_embedder, mock_llm_client, mock_cross_encoder):
    """Test updating entity name."""
    graphiti = Graphiti(graph_driver=graph_driver, embedder=mock_embedder, llm_client=mock_llm_client, cross_encoder=mock_cross_encoder)

    # Create
    entity = await graphiti.create_entity(
        name='CRUD Test Entity',
        group_id=group_id,
        summary='CRUD Test Entity summary',
    )

    # Update name
    updated = await graphiti.update_entity(
        uuid=entity.uuid,
        name='Updated Entity Name',
        group_id=group_id,
    )

    assert updated.name == 'Updated Entity Name'

    # Cleanup
    await entity.delete(graph_driver)


@pytest.mark.asyncio
async def test_update_entity_summary(graph_driver, mock_embedder, mock_llm_client, mock_cross_encoder):
    """Test updating entity summary."""
    graphiti = Graphiti(graph_driver=graph_driver, embedder=mock_embedder, llm_client=mock_llm_client, cross_encoder=mock_cross_encoder)

    # Create
    entity = await graphiti.create_entity(
        name='CRUD Test Entity',
        group_id=group_id,
        summary='CRUD Test Entity summary',
    )

    # Update summary
    updated = await graphiti.update_entity(
        uuid=entity.uuid,
        summary='Updated summary text',
        group_id=group_id,
    )

    assert updated.summary == 'Updated summary text'

    # Cleanup
    await entity.delete(graph_driver)


@pytest.mark.asyncio
async def test_update_entity_type(graph_driver, mock_embedder, mock_llm_client, mock_cross_encoder):
    """Test updating entity type changes labels."""
    graphiti = Graphiti(graph_driver=graph_driver, embedder=mock_embedder, llm_client=mock_llm_client, cross_encoder=mock_cross_encoder)

    # Create with Person type
    entity = await graphiti.create_entity(
        name='CRUD Test Entity',
        group_id=group_id,
        entity_type='Person',
        summary='CRUD Test Entity summary',
    )
    assert 'Person' in entity.labels

    # Change to Organization
    updated = await graphiti.update_entity(
        uuid=entity.uuid,
        entity_type='Organization',
        group_id=group_id,
    )

    assert 'Organization' in updated.labels
    assert 'Person' not in updated.labels

    # Verify in database
    retrieved = await graphiti.get_entity(entity.uuid, group_id=group_id)
    assert 'Organization' in retrieved.labels

    # Cleanup
    await entity.delete(graph_driver)


@pytest.mark.asyncio
async def test_update_entity_invalid_type(graph_driver, mock_embedder, mock_llm_client, mock_cross_encoder):
    """Test that invalid entity_type in update raises ValueError."""
    graphiti = Graphiti(graph_driver=graph_driver, embedder=mock_embedder, llm_client=mock_llm_client, cross_encoder=mock_cross_encoder)

    # Create
    entity = await graphiti.create_entity(
        name='CRUD Test Entity',
        group_id=group_id,
        summary='CRUD Test Entity summary',
    )

    # Try invalid update
    with pytest.raises(ValueError, match='Invalid entity_type'):
        await graphiti.update_entity(
            uuid=entity.uuid,
            entity_type='Invalid-Type',
            group_id=group_id,
        )

    # Cleanup
    await entity.delete(graph_driver)


@pytest.mark.asyncio
async def test_remove_entity(graph_driver, mock_embedder, mock_llm_client, mock_cross_encoder):
    """Test removing an entity."""
    graphiti = Graphiti(graph_driver=graph_driver, embedder=mock_embedder, llm_client=mock_llm_client, cross_encoder=mock_cross_encoder)

    # Create
    entity = await graphiti.create_entity(
        name='CRUD Test Entity',
        group_id=group_id,
        summary='CRUD Test Entity summary',
    )

    # Remove
    await graphiti.remove_entity(entity.uuid, group_id=group_id)

    # Verify it's gone
    with pytest.raises(NodeNotFoundError):
        await graphiti.get_entity(entity.uuid, group_id=group_id)


@pytest.mark.asyncio
async def test_get_entities_by_group_id(graph_driver, mock_embedder, mock_llm_client, mock_cross_encoder):
    """Test listing entities by group ID."""
    graphiti = Graphiti(graph_driver=graph_driver, embedder=mock_embedder, llm_client=mock_llm_client, cross_encoder=mock_cross_encoder)

    # Create two entities
    entity1 = await graphiti.create_entity(
        name='CRUD Test Entity',
        group_id=group_id,
        summary='CRUD Test Entity summary',
    )
    entity2 = await graphiti.create_entity(
        name='CRUD Test Target',
        group_id=group_id,
        summary='CRUD Test Target summary',
    )

    # Get by group
    entities = await graphiti.get_entities_by_group_id(group_id)

    assert len(entities) >= 2
    uuids = [e.uuid for e in entities]
    assert entity1.uuid in uuids
    assert entity2.uuid in uuids

    # Cleanup
    await entity1.delete(graph_driver)
    await entity2.delete(graph_driver)


@pytest.mark.asyncio
async def test_create_edge(graph_driver, mock_embedder, mock_llm_client, mock_cross_encoder):
    """Test creating an edge with auto-created episode."""
    graphiti = Graphiti(graph_driver=graph_driver, embedder=mock_embedder, llm_client=mock_llm_client, cross_encoder=mock_cross_encoder)

    # Create two entities
    entity1 = await graphiti.create_entity(
        name='CRUD Test Entity',
        group_id=group_id,
        summary='CRUD Test Entity summary',
    )
    entity2 = await graphiti.create_entity(
        name='CRUD Test Target',
        group_id=group_id,
        summary='CRUD Test Target summary',
    )

    # Create edge
    edge = await graphiti.create_edge(
        source_node_uuid=entity1.uuid,
        target_node_uuid=entity2.uuid,
        name='WORKS_ON',
        fact='CRUD Test Entity works on CRUD Test Target',
        group_id=group_id,
    )

    assert edge is not None
    assert edge.uuid is not None
    assert edge.name == 'WORKS_ON'
    assert edge.source_node_uuid == entity1.uuid
    assert edge.target_node_uuid == entity2.uuid

    # Edge should have an episode (auto-created)
    assert edge.episodes is not None
    assert len(edge.episodes) == 1

    # Cleanup
    await edge.delete(graph_driver)
    await entity1.delete(graph_driver)
    await entity2.delete(graph_driver)


@pytest.mark.asyncio
async def test_get_edge(graph_driver, mock_embedder, mock_llm_client, mock_cross_encoder):
    """Test retrieving an edge by UUID."""
    graphiti = Graphiti(graph_driver=graph_driver, embedder=mock_embedder, llm_client=mock_llm_client, cross_encoder=mock_cross_encoder)

    # Create entities and edge
    entity1 = await graphiti.create_entity(
        name='CRUD Test Entity',
        group_id=group_id,
        summary='CRUD Test Entity summary',
    )
    entity2 = await graphiti.create_entity(
        name='CRUD Test Target',
        group_id=group_id,
        summary='CRUD Test Target summary',
    )
    edge = await graphiti.create_edge(
        source_node_uuid=entity1.uuid,
        target_node_uuid=entity2.uuid,
        name='WORKS_ON',
        fact='CRUD Test Entity works on CRUD Test Target',
        group_id=group_id,
    )

    # Get edge
    retrieved = await graphiti.get_edge(edge.uuid, group_id=group_id)

    assert retrieved is not None
    assert retrieved.uuid == edge.uuid
    assert retrieved.name == 'WORKS_ON'

    # Cleanup
    await edge.delete(graph_driver)
    await entity1.delete(graph_driver)
    await entity2.delete(graph_driver)


@pytest.mark.asyncio
async def test_update_edge(graph_driver, mock_embedder, mock_llm_client, mock_cross_encoder):
    """Test updating an edge."""
    graphiti = Graphiti(graph_driver=graph_driver, embedder=mock_embedder, llm_client=mock_llm_client, cross_encoder=mock_cross_encoder)

    # Create entities and edge
    entity1 = await graphiti.create_entity(
        name='CRUD Test Entity',
        group_id=group_id,
        summary='CRUD Test Entity summary',
    )
    entity2 = await graphiti.create_entity(
        name='CRUD Test Target',
        group_id=group_id,
        summary='CRUD Test Target summary',
    )
    edge = await graphiti.create_edge(
        source_node_uuid=entity1.uuid,
        target_node_uuid=entity2.uuid,
        name='WORKS_ON',
        fact='CRUD Test Entity works on CRUD Test Target',
        group_id=group_id,
    )

    # Update
    updated = await graphiti.update_edge(
        uuid=edge.uuid,
        name='CONTRIBUTES_TO',
        fact='Updated fact text',
        group_id=group_id,
    )

    assert updated.name == 'CONTRIBUTES_TO'
    assert updated.fact == 'Updated fact text'

    # Cleanup
    await edge.delete(graph_driver)
    await entity1.delete(graph_driver)
    await entity2.delete(graph_driver)


@pytest.mark.asyncio
async def test_remove_edge(graph_driver, mock_embedder, mock_llm_client, mock_cross_encoder):
    """Test removing an edge."""
    graphiti = Graphiti(graph_driver=graph_driver, embedder=mock_embedder, llm_client=mock_llm_client, cross_encoder=mock_cross_encoder)

    # Create entities and edge
    entity1 = await graphiti.create_entity(
        name='CRUD Test Entity',
        group_id=group_id,
        summary='CRUD Test Entity summary',
    )
    entity2 = await graphiti.create_entity(
        name='CRUD Test Target',
        group_id=group_id,
        summary='CRUD Test Target summary',
    )
    edge = await graphiti.create_edge(
        source_node_uuid=entity1.uuid,
        target_node_uuid=entity2.uuid,
        name='WORKS_ON',
        fact='CRUD Test Entity works on CRUD Test Target',
        group_id=group_id,
    )

    # Remove edge
    await graphiti.remove_edge(edge.uuid, group_id=group_id)

    # Verify it's gone
    with pytest.raises(EdgeNotFoundError):
        await graphiti.get_edge(edge.uuid, group_id=group_id)

    # Cleanup entities
    await entity1.delete(graph_driver)
    await entity2.delete(graph_driver)


@pytest.mark.asyncio
async def test_get_edges_by_group_id(graph_driver, mock_embedder, mock_llm_client, mock_cross_encoder):
    """Test listing edges by group ID."""
    graphiti = Graphiti(graph_driver=graph_driver, embedder=mock_embedder, llm_client=mock_llm_client, cross_encoder=mock_cross_encoder)

    # Create entities and edge
    entity1 = await graphiti.create_entity(
        name='CRUD Test Entity',
        group_id=group_id,
        summary='CRUD Test Entity summary',
    )
    entity2 = await graphiti.create_entity(
        name='CRUD Test Target',
        group_id=group_id,
        summary='CRUD Test Target summary',
    )
    edge = await graphiti.create_edge(
        source_node_uuid=entity1.uuid,
        target_node_uuid=entity2.uuid,
        name='WORKS_ON',
        fact='CRUD Test Entity works on CRUD Test Target',
        group_id=group_id,
    )

    # Get by group
    edges = await graphiti.get_edges_by_group_id(group_id)

    assert len(edges) >= 1
    assert any(e.uuid == edge.uuid for e in edges)

    # Cleanup
    await edge.delete(graph_driver)
    await entity1.delete(graph_driver)
    await entity2.delete(graph_driver)


@pytest.mark.asyncio
async def test_get_episode(graph_driver, mock_embedder, mock_llm_client, mock_cross_encoder):
    """Test retrieving an episode created with an edge."""
    graphiti = Graphiti(graph_driver=graph_driver, embedder=mock_embedder, llm_client=mock_llm_client, cross_encoder=mock_cross_encoder)

    # Create entities and edge (which creates episode)
    entity1 = await graphiti.create_entity(
        name='CRUD Test Entity',
        group_id=group_id,
        summary='CRUD Test Entity summary',
    )
    entity2 = await graphiti.create_entity(
        name='CRUD Test Target',
        group_id=group_id,
        summary='CRUD Test Target summary',
    )
    edge = await graphiti.create_edge(
        source_node_uuid=entity1.uuid,
        target_node_uuid=entity2.uuid,
        name='WORKS_ON',
        fact='CRUD Test Entity works on CRUD Test Target',
        group_id=group_id,
    )

    # Get episode
    episode = await graphiti.get_episode(edge.episodes[0], group_id=group_id)

    assert episode is not None
    assert episode.content == 'CRUD Test Entity works on CRUD Test Target'
    assert edge.uuid in episode.entity_edges

    # Cleanup
    await edge.delete(graph_driver)
    await entity1.delete(graph_driver)
    await entity2.delete(graph_driver)


@pytest.mark.asyncio
async def test_get_episodes_by_group_id(graph_driver, mock_embedder, mock_llm_client, mock_cross_encoder):
    """Test listing episodes by group ID."""
    graphiti = Graphiti(graph_driver=graph_driver, embedder=mock_embedder, llm_client=mock_llm_client, cross_encoder=mock_cross_encoder)

    # Create entities and edge (which creates episode)
    entity1 = await graphiti.create_entity(
        name='CRUD Test Entity',
        group_id=group_id,
        summary='CRUD Test Entity summary',
    )
    entity2 = await graphiti.create_entity(
        name='CRUD Test Target',
        group_id=group_id,
        summary='CRUD Test Target summary',
    )
    edge = await graphiti.create_edge(
        source_node_uuid=entity1.uuid,
        target_node_uuid=entity2.uuid,
        name='WORKS_ON',
        fact='CRUD Test Entity works on CRUD Test Target',
        group_id=group_id,
    )

    # Get episodes by group
    episodes = await graphiti.get_episodes_by_group_id(group_id)

    assert len(episodes) >= 1
    assert any(e.uuid == edge.episodes[0] for e in episodes)

    # Cleanup
    await edge.delete(graph_driver)
    await entity1.delete(graph_driver)
    await entity2.delete(graph_driver)


@pytest.mark.asyncio
async def test_get_groups(graph_driver, mock_embedder, mock_llm_client, mock_cross_encoder):
    """Test getting all group IDs."""
    graphiti = Graphiti(graph_driver=graph_driver, embedder=mock_embedder, llm_client=mock_llm_client, cross_encoder=mock_cross_encoder)

    # Create entity to ensure group exists
    entity = await graphiti.create_entity(
        name='CRUD Test Entity',
        group_id=group_id,
        summary='CRUD Test Entity summary',
    )

    # Get groups
    groups = await graphiti.get_groups()

    assert isinstance(groups, list)
    assert group_id in groups

    # Cleanup
    await entity.delete(graph_driver)


@pytest.mark.asyncio
async def test_get_graph_stats(graph_driver, mock_embedder, mock_llm_client, mock_cross_encoder):
    """Test getting graph statistics."""
    graphiti = Graphiti(graph_driver=graph_driver, embedder=mock_embedder, llm_client=mock_llm_client, cross_encoder=mock_cross_encoder)

    # Create entities and edge
    entity1 = await graphiti.create_entity(
        name='CRUD Test Entity',
        group_id=group_id,
        summary='CRUD Test Entity summary',
    )
    entity2 = await graphiti.create_entity(
        name='CRUD Test Target',
        group_id=group_id,
        summary='CRUD Test Target summary',
    )
    edge = await graphiti.create_edge(
        source_node_uuid=entity1.uuid,
        target_node_uuid=entity2.uuid,
        name='WORKS_ON',
        fact='CRUD Test Entity works on CRUD Test Target',
        group_id=group_id,
    )

    # Get stats
    stats = await graphiti.get_graph_stats(group_id=group_id)

    assert 'node_count' in stats
    assert 'edge_count' in stats
    assert 'episode_count' in stats
    assert stats['node_count'] >= 2
    assert stats['edge_count'] >= 1
    assert stats['episode_count'] >= 1

    # Cleanup
    await edge.delete(graph_driver)
    await entity1.delete(graph_driver)
    await entity2.delete(graph_driver)


@pytest.mark.asyncio
async def test_rename_group(graph_driver, mock_embedder, mock_llm_client, mock_cross_encoder):
    """Test renaming a group."""
    graphiti = Graphiti(graph_driver=graph_driver, embedder=mock_embedder, llm_client=mock_llm_client, cross_encoder=mock_cross_encoder)

    # Create entity in group_id_2
    entity = await graphiti.create_entity(
        name='Temp Entity',
        group_id=group_id_2,
        summary='CRUD Test Entity summary',
    )

    # Verify group exists
    groups_before = await graphiti.get_groups()
    assert group_id_2 in groups_before

    # Rename
    new_name = f'{group_id_2}_renamed'
    await graphiti.rename_group(group_id_2, new_name)

    # Verify rename worked
    groups_after = await graphiti.get_groups()
    assert group_id_2 not in groups_after
    assert new_name in groups_after

    # Cleanup
    await graphiti.remove_group(new_name)


@pytest.mark.asyncio
async def test_rename_group_same_name_error(graph_driver, mock_embedder, mock_llm_client, mock_cross_encoder):
    """Test that renaming to same name raises ValueError."""
    graphiti = Graphiti(graph_driver=graph_driver, embedder=mock_embedder, llm_client=mock_llm_client, cross_encoder=mock_cross_encoder)

    with pytest.raises(ValueError, match='must be different'):
        await graphiti.rename_group(group_id, group_id)


@pytest.mark.asyncio
async def test_remove_group(graph_driver, mock_embedder, mock_llm_client, mock_cross_encoder):
    """Test removing an entire group."""
    graphiti = Graphiti(graph_driver=graph_driver, embedder=mock_embedder, llm_client=mock_llm_client, cross_encoder=mock_cross_encoder)

    # Create entity in group_id_2
    await graphiti.create_entity(
        name='Entity to delete',
        group_id=group_id_2,
        summary='CRUD Test Entity summary',
    )

    # Verify group exists
    groups = await graphiti.get_groups()
    assert group_id_2 in groups

    # Remove group
    await graphiti.remove_group(group_id_2)

    # Verify it's gone
    groups = await graphiti.get_groups()
    assert group_id_2 not in groups


@pytest.mark.asyncio
async def test_execute_query(graph_driver, mock_embedder, mock_llm_client, mock_cross_encoder):
    """Test executing a raw Cypher query."""
    graphiti = Graphiti(graph_driver=graph_driver, embedder=mock_embedder, llm_client=mock_llm_client, cross_encoder=mock_cross_encoder)

    # Create entity
    entity = await graphiti.create_entity(
        name='CRUD Test Entity',
        group_id=group_id,
        summary='CRUD Test Entity summary',
    )

    # Execute query
    result, _, _ = await graphiti.execute_query(
        'MATCH (n:Entity {group_id: $group_id}) RETURN count(n) AS count',
        group_id=group_id,
    )

    assert len(result) == 1
    assert result[0]['count'] >= 1

    # Cleanup
    await entity.delete(graph_driver)
