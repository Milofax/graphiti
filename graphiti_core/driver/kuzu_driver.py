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
"""

import logging
from typing import Any

import kuzu

from graphiti_core.driver.driver import GraphDriver, GraphDriverSession, GraphProvider

logger = logging.getLogger(__name__)

# Kuzu requires an explicit schema.
# As Kuzu currently does not support creating full text indexes on edge properties,
# we work around this by representing (n:Entity)-[:RELATES_TO]->(m:Entity) as
# (n)-[:RELATES_TO]->(e:RelatesToNode_)-[:RELATES_TO]->(m).
SCHEMA_QUERIES = """
    CREATE NODE TABLE IF NOT EXISTS Episodic (
        uuid STRING PRIMARY KEY,
        name STRING,
        group_id STRING,
        created_at TIMESTAMP,
        source STRING,
        source_description STRING,
        content STRING,
        valid_at TIMESTAMP,
        entity_edges STRING[]
    );
    CREATE NODE TABLE IF NOT EXISTS Entity (
        uuid STRING PRIMARY KEY,
        name STRING,
        group_id STRING,
        labels STRING[],
        created_at TIMESTAMP,
        name_embedding FLOAT[],
        summary STRING,
        attributes STRING
    );
    CREATE NODE TABLE IF NOT EXISTS Community (
        uuid STRING PRIMARY KEY,
        name STRING,
        group_id STRING,
        created_at TIMESTAMP,
        name_embedding FLOAT[],
        summary STRING
    );
    CREATE NODE TABLE IF NOT EXISTS RelatesToNode_ (
        uuid STRING PRIMARY KEY,
        group_id STRING,
        created_at TIMESTAMP,
        name STRING,
        fact STRING,
        fact_embedding FLOAT[],
        episodes STRING[],
        expired_at TIMESTAMP,
        valid_at TIMESTAMP,
        invalid_at TIMESTAMP,
        attributes STRING
    );
    CREATE REL TABLE IF NOT EXISTS RELATES_TO(
        FROM Entity TO RelatesToNode_,
        FROM RelatesToNode_ TO Entity
    );
    CREATE REL TABLE IF NOT EXISTS MENTIONS(
        FROM Episodic TO Entity,
        uuid STRING PRIMARY KEY,
        group_id STRING,
        created_at TIMESTAMP
    );
    CREATE REL TABLE IF NOT EXISTS HAS_MEMBER(
        FROM Community TO Entity,
        FROM Community TO Community,
        uuid STRING,
        group_id STRING,
        created_at TIMESTAMP
    );
"""


class KuzuDriver(GraphDriver):
    provider: GraphProvider = GraphProvider.KUZU
    aoss_client: None = None

    def __init__(
        self,
        db: str = ':memory:',
        max_concurrent_queries: int = 1,
    ):
        super().__init__()
        self.db = kuzu.Database(db)

        self.setup_schema()

        self.client = kuzu.AsyncConnection(self.db, max_concurrent_queries=max_concurrent_queries)

    async def execute_query(
        self, cypher_query_: str, **kwargs: Any
    ) -> tuple[list[dict[str, Any]] | list[list[dict[str, Any]]], None, None]:
        params = {k: v for k, v in kwargs.items() if v is not None}
        # Kuzu does not support these parameters.
        params.pop('database_', None)
        params.pop('routing_', None)

        try:
            results = await self.client.execute(cypher_query_, parameters=params)
        except Exception as e:
            params = {k: (v[:5] if isinstance(v, list) else v) for k, v in params.items()}
            logger.error(f'Error executing Kuzu query: {e}\n{cypher_query_}\n{params}')
            raise

        if not results:
            return [], None, None

        if isinstance(results, list):
            dict_results = [list(result.rows_as_dict()) for result in results]
        else:
            dict_results = list(results.rows_as_dict())
        return dict_results, None, None  # type: ignore

    def session(self, _database: str | None = None) -> GraphDriverSession:
        return KuzuDriverSession(self)

    async def close(self):
        # Do not explicitly close the connection, instead rely on GC.
        pass

    def delete_all_indexes(self, database_: str):
        pass

    async def build_indices_and_constraints(self, delete_existing: bool = False):
        # Kuzu doesn't support dynamic index creation like Neo4j or FalkorDB
        # Schema and indices are created during setup_schema()
        # This method is required by the abstract base class but is a no-op for Kuzu
        pass

    def setup_schema(self):
        conn = kuzu.Connection(self.db)
        conn.execute(SCHEMA_QUERIES)
        conn.close()

    async def copy_group(self, source_group_id: str, target_group_id: str) -> None:
        """
        Copy all nodes from one group to another using Cypher.

        In Kuzu, group_id is a property on nodes. Creates new nodes
        with new UUIDs and the target group_id.

        Note: Kuzu has an explicit schema, so we copy each node type separately.
        Relationships are not copied as they reference specific node UUIDs.
        """
        import uuid as uuid_mod

        if source_group_id == target_group_id:
            raise ValueError('Source and target group IDs must be different')

        # Copy Entity nodes (one by one to generate unique UUIDs)
        records, _, _ = await self.execute_query(
            'MATCH (n:Entity) WHERE n.group_id = $source_group_id RETURN n',
            source_group_id=source_group_id,
        )
        for record in records:
            node = record.get('n', {})
            await self.execute_query(
                """
                CREATE (copy:Entity {
                    uuid: $uuid,
                    name: $name,
                    group_id: $group_id,
                    labels: $labels,
                    created_at: $created_at,
                    name_embedding: $name_embedding,
                    summary: $summary,
                    attributes: $attributes
                })
                """,
                uuid=str(uuid_mod.uuid4()),
                name=node.get('name'),
                group_id=target_group_id,
                labels=node.get('labels'),
                created_at=node.get('created_at'),
                name_embedding=node.get('name_embedding'),
                summary=node.get('summary'),
                attributes=node.get('attributes'),
            )

        # Copy Episodic nodes
        records, _, _ = await self.execute_query(
            'MATCH (n:Episodic) WHERE n.group_id = $source_group_id RETURN n',
            source_group_id=source_group_id,
        )
        for record in records:
            node = record.get('n', {})
            await self.execute_query(
                """
                CREATE (copy:Episodic {
                    uuid: $uuid,
                    name: $name,
                    group_id: $group_id,
                    created_at: $created_at,
                    source: $source,
                    source_description: $source_description,
                    content: $content,
                    valid_at: $valid_at,
                    entity_edges: $entity_edges
                })
                """,
                uuid=str(uuid_mod.uuid4()),
                name=node.get('name'),
                group_id=target_group_id,
                created_at=node.get('created_at'),
                source=node.get('source'),
                source_description=node.get('source_description'),
                content=node.get('content'),
                valid_at=node.get('valid_at'),
                entity_edges=node.get('entity_edges'),
            )

        logger.info(f'Copied group {source_group_id} to {target_group_id}')

    async def rename_group(self, old_group_id: str, new_group_id: str) -> None:
        """
        Rename a group by updating the group_id property on all nodes.

        In Kuzu, this updates Entity, Episodic, RelatesToNode_, and Community nodes.
        """
        if old_group_id == new_group_id:
            raise ValueError('Old and new group IDs must be different')

        # Update Entity nodes
        await self.execute_query(
            """
            MATCH (n:Entity)
            WHERE n.group_id = $old_group_id
            SET n.group_id = $new_group_id
            """,
            old_group_id=old_group_id,
            new_group_id=new_group_id,
        )

        # Update Episodic nodes
        await self.execute_query(
            """
            MATCH (n:Episodic)
            WHERE n.group_id = $old_group_id
            SET n.group_id = $new_group_id
            """,
            old_group_id=old_group_id,
            new_group_id=new_group_id,
        )

        # Update RelatesToNode_ (edge nodes in Kuzu's schema)
        await self.execute_query(
            """
            MATCH (n:RelatesToNode_)
            WHERE n.group_id = $old_group_id
            SET n.group_id = $new_group_id
            """,
            old_group_id=old_group_id,
            new_group_id=new_group_id,
        )

        # Update Community nodes
        await self.execute_query(
            """
            MATCH (n:Community)
            WHERE n.group_id = $old_group_id
            SET n.group_id = $new_group_id
            """,
            old_group_id=old_group_id,
            new_group_id=new_group_id,
        )

        logger.info(f'Renamed group {old_group_id} to {new_group_id}')

    async def list_groups(self) -> list[str]:
        """
        List all groups (distinct group_ids) in Kuzu.

        In Kuzu, all groups are in one database, distinguished by group_id property.
        Queries all node types to avoid missing groups with only non-Entity nodes.
        """
        records, _, _ = await self.execute_query(
            """
            MATCH (n:Entity)
            WHERE n.group_id IS NOT NULL
            RETURN DISTINCT n.group_id AS group_id
            UNION
            MATCH (n:Episodic)
            WHERE n.group_id IS NOT NULL
            RETURN DISTINCT n.group_id AS group_id
            UNION
            MATCH (n:RelatesToNode_)
            WHERE n.group_id IS NOT NULL
            RETURN DISTINCT n.group_id AS group_id
            UNION
            MATCH (n:Community)
            WHERE n.group_id IS NOT NULL
            RETURN DISTINCT n.group_id AS group_id
            """,
        )
        return sorted({record['group_id'] for record in records})

    async def delete_group(self, group_id: str) -> None:
        """
        Delete all nodes belonging to a group.

        In Kuzu, group_id is a property on nodes. Deletes RelatesToNode_ first
        (edge-nodes between Entity nodes), then remaining nodes with DETACH DELETE
        to handle attached relationships (MENTIONS, HAS_MEMBER, RELATES_TO).
        """
        # Delete RelatesToNode_ first (edge-nodes that sit between Entity nodes)
        await self.execute_query(
            """
            MATCH (n:RelatesToNode_)
            WHERE n.group_id = $group_id
            DETACH DELETE n
            """,
            group_id=group_id,
        )

        # Delete Entity nodes (may have MENTIONS relationships from Episodic)
        await self.execute_query(
            """
            MATCH (n:Entity)
            WHERE n.group_id = $group_id
            DETACH DELETE n
            """,
            group_id=group_id,
        )

        # Delete Episodic nodes
        await self.execute_query(
            """
            MATCH (n:Episodic)
            WHERE n.group_id = $group_id
            DETACH DELETE n
            """,
            group_id=group_id,
        )

        # Delete Community nodes
        await self.execute_query(
            """
            MATCH (n:Community)
            WHERE n.group_id = $group_id
            DETACH DELETE n
            """,
            group_id=group_id,
        )

        logger.info(f'Deleted group {group_id}')


class KuzuDriverSession(GraphDriverSession):
    provider = GraphProvider.KUZU

    def __init__(self, driver: KuzuDriver):
        self.driver = driver

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        # No cleanup needed for Kuzu, but method must exist.
        pass

    async def close(self):
        # Do not close the session here, as we're reusing the driver connection.
        pass

    async def execute_write(self, func, *args, **kwargs):
        # Directly await the provided async function with `self` as the transaction/session
        return await func(self, *args, **kwargs)

    async def run(self, query: str | list, **kwargs: Any) -> Any:
        if isinstance(query, list):
            for cypher, params in query:
                await self.driver.execute_query(cypher, **params)
        else:
            await self.driver.execute_query(query, **kwargs)
        return None
