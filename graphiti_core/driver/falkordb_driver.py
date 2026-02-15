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

import asyncio
import datetime
import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from falkordb import Graph as FalkorGraph
    from falkordb.asyncio import FalkorDB
else:
    try:
        from falkordb import Graph as FalkorGraph
        from falkordb.asyncio import FalkorDB
    except ImportError:
        # If falkordb is not installed, raise an ImportError
        raise ImportError(
            'falkordb is required for FalkorDriver. '
            'Install it with: pip install graphiti-core[falkordb]'
        ) from None

from graphiti_core.driver.driver import GraphDriver, GraphDriverSession, GraphProvider
from graphiti_core.graph_queries import get_fulltext_indices, get_range_indices
from graphiti_core.utils.datetime_utils import convert_datetimes_to_strings

logger = logging.getLogger(__name__)

STOPWORDS = [
    # === ENGLISH ===
    # Articles & determiners
    'a', 'an', 'the', 'this', 'that', 'these', 'those', 'some', 'any', 'each', 'every',
    # Pronouns
    'i', 'me', 'my', 'mine', 'myself',
    'you', 'your', 'yours', 'yourself',
    'he', 'him', 'his', 'himself',
    'she', 'her', 'hers', 'herself',
    'it', 'its', 'itself',
    'we', 'us', 'our', 'ours', 'ourselves',
    'they', 'them', 'their', 'theirs', 'themselves',
    # Be/have/do verbs
    'is', 'am', 'are', 'was', 'were', 'be', 'been', 'being',
    'has', 'have', 'had', 'having',
    'do', 'does', 'did', 'doing', 'done',
    # Modal verbs
    'can', 'could', 'will', 'would', 'shall', 'should', 'may', 'might', 'must',
    # Common verbs
    'get', 'gets', 'got', 'getting',
    'go', 'goes', 'went', 'going', 'gone',
    'make', 'makes', 'made', 'making',
    'come', 'comes', 'came', 'coming',
    'take', 'takes', 'took', 'taking', 'taken',
    'give', 'gives', 'gave', 'given',
    'say', 'says', 'said',
    'know', 'knows', 'knew', 'known',
    'see', 'sees', 'saw', 'seen',
    'use', 'uses', 'used', 'using',
    'find', 'finds', 'found',
    'want', 'need', 'keep', 'let', 'put', 'set', 'run', 'show',
    # Prepositions & conjunctions
    'at', 'by', 'for', 'from', 'in', 'into', 'of', 'on', 'to', 'with',
    'about', 'after', 'before', 'between', 'through', 'during', 'without',
    'above', 'below', 'under', 'over', 'up', 'down', 'out', 'off',
    'and', 'but', 'or', 'nor', 'so', 'yet', 'both', 'either', 'neither',
    'if', 'then', 'than', 'when', 'where', 'while', 'because', 'since', 'until', 'although',
    # Adverbs
    'not', 'no', 'also', 'very', 'just', 'only', 'even', 'still', 'already',
    'always', 'never', 'often', 'usually', 'generally', 'sometimes',
    'here', 'there', 'now', 'again', 'once',
    'how', 'what', 'which', 'who', 'whom', 'whose', 'why',
    # Other common
    'own', 'other', 'another', 'such', 'much', 'many', 'more', 'most',
    'all', 'few', 'less', 'same', 'well', 'back',
    'new', 'old', 'first', 'last', 'long', 'great', 'little', 'right',
    'big', 'high', 'different', 'small', 'large', 'next', 'early',
    'able', 'like', 'however', 'way', 'thing', 'things',
    'as',
    # === DEUTSCH ===
    # Artikel & Determinanten
    'der', 'die', 'das', 'den', 'dem', 'des',
    'ein', 'eine', 'einer', 'einem', 'einen', 'eines',
    'kein', 'keine', 'keiner', 'keinem', 'keinen', 'keines',
    # Pronomen
    'ich', 'du', 'er', 'sie', 'es', 'wir', 'ihr',
    'mich', 'dich', 'sich', 'uns', 'euch',
    'mir', 'dir', 'ihm', 'ihnen',
    'mein', 'dein', 'sein', 'unser', 'euer',
    'dieser', 'diese', 'dieses', 'jener', 'jene', 'jenes',
    'man', 'was', 'wer', 'welche', 'welcher', 'welches',
    # Verben (sein/haben/werden)
    'ist', 'bin', 'bist', 'sind', 'seid', 'war', 'waren', 'gewesen',
    'hat', 'habe', 'hast', 'haben', 'habt', 'hatte', 'hatten',
    'wird', 'werde', 'wirst', 'werden', 'werdet', 'wurde', 'wurden',
    # Modalverben
    'kann', 'kannst', 'muss', 'soll', 'darf', 'mag', 'will',
    # Präpositionen & Konjunktionen
    'auf', 'aus', 'bei', 'bis', 'durch', 'nach', 'ohne', 'um', 'unter',
    'vor', 'zwischen', 'gegen', 'seit', 'von', 'zu', 'zum', 'zur',
    'mit', 'als', 'wie', 'ob', 'dass', 'weil', 'wenn', 'aber', 'oder',
    'und', 'denn', 'sondern', 'nicht', 'noch', 'schon', 'auch', 'nur',
    'so', 'da', 'dann', 'doch', 'sehr', 'immer', 'hier', 'dort',
    # Andere häufige
    'andere', 'anderer', 'anderes', 'anderen',
    'alle', 'alles', 'allem', 'allen', 'aller',
    'viel', 'viele', 'mehr', 'ganz', 'etwa', 'dabei',
]

# Safety net: cap OR-terms to prevent fulltext explosion on long queries
MAX_FULLTEXT_TERMS = 10


class FalkorDriverSession(GraphDriverSession):
    provider = GraphProvider.FALKORDB

    def __init__(self, graph: FalkorGraph):
        self.graph = graph

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        # No cleanup needed for Falkor, but method must exist
        pass

    async def close(self):
        # No explicit close needed for FalkorDB, but method must exist
        pass

    async def execute_write(self, func, *args, **kwargs):
        # Directly await the provided async function with `self` as the transaction/session
        return await func(self, *args, **kwargs)

    async def run(self, query: str | list, **kwargs: Any) -> Any:
        # FalkorDB does not support argument for Label Set, so it's converted into an array of queries
        if isinstance(query, list):
            for cypher, params in query:
                params = convert_datetimes_to_strings(params)
                await self.graph.query(str(cypher), params)  # type: ignore[reportUnknownArgumentType]
        else:
            params = dict(kwargs)
            params = convert_datetimes_to_strings(params)
            await self.graph.query(str(query), params)  # type: ignore[reportUnknownArgumentType]
        # Assuming `graph.query` is async (ideal); otherwise, wrap in executor
        return None


class FalkorDriver(GraphDriver):
    provider = GraphProvider.FALKORDB
    default_group_id: str = '\\_'
    fulltext_syntax: str = '@'  # FalkorDB uses a redisearch-like syntax for fulltext queries
    aoss_client: None = None

    def __init__(
        self,
        host: str = 'localhost',
        port: int = 6379,
        username: str | None = None,
        password: str | None = None,
        falkor_db: FalkorDB | None = None,
        database: str = 'default_db',
        _skip_index_init: bool = False,
    ):
        """
        Initialize the FalkorDB driver.

        FalkorDB is a multi-tenant graph database.
        To connect, provide the host and port.
        The default parameters assume a local (on-premises) FalkorDB instance.

        Args:
        host (str): The host where FalkorDB is running.
        port (int): The port on which FalkorDB is listening.
        username (str | None): The username for authentication (if required).
        password (str | None): The password for authentication (if required).
        falkor_db (FalkorDB | None): An existing FalkorDB instance to use instead of creating a new one.
        database (str): The name of the database to connect to. Defaults to 'default_db'.
        _skip_index_init (bool): Internal flag to skip index initialization (used by clone()).
        """
        super().__init__()
        self._database = database
        if falkor_db is not None:
            # If a FalkorDB instance is provided, use it directly
            self.client = falkor_db
        else:
            self.client = FalkorDB(host=host, port=port, username=username, password=password)

        # Schedule the indices and constraints to be built (unless skipped for cloned drivers)
        if not _skip_index_init:
            try:
                # Try to get the current event loop
                loop = asyncio.get_running_loop()
                # Schedule the build_indices_and_constraints to run
                loop.create_task(self.build_indices_and_constraints())
            except RuntimeError:
                # No event loop running, this will be handled later
                pass

    def _get_graph(self, graph_name: str | None) -> FalkorGraph:
        # FalkorDB requires a non-None database name for multi-tenant graphs; the default is "default_db"
        if graph_name is None:
            graph_name = self._database
        return self.client.select_graph(graph_name)

    async def execute_query(self, cypher_query_, **kwargs: Any):
        graph = self._get_graph(self._database)

        # Convert datetime objects to ISO strings (FalkorDB does not support datetime objects directly)
        params = convert_datetimes_to_strings(dict(kwargs))

        try:
            result = await graph.query(cypher_query_, params)  # type: ignore[reportUnknownArgumentType]
        except Exception as e:
            if 'already indexed' in str(e):
                # check if index already exists
                logger.info(f'Index already exists: {e}')
                return None
            logger.error(f'Error executing FalkorDB query: {e}\n{cypher_query_}\n{params}')
            raise

        # Convert the result header to a list of strings
        header = [h[1] for h in result.header]

        # Convert FalkorDB's result format (list of lists) to the format expected by Graphiti (list of dicts)
        records = []
        for row in result.result_set:
            record = {}
            for i, field_name in enumerate(header):
                if i < len(row):
                    record[field_name] = row[i]
                else:
                    # If there are more fields in header than values in row, set to None
                    record[field_name] = None
            records.append(record)

        return records, header, None

    def session(self, database: str | None = None) -> GraphDriverSession:
        return FalkorDriverSession(self._get_graph(database))

    async def close(self) -> None:
        """Close the driver connection."""
        if hasattr(self.client, 'aclose'):
            await self.client.aclose()  # type: ignore[reportUnknownMemberType]
        elif hasattr(self.client.connection, 'aclose'):
            await self.client.connection.aclose()
        elif hasattr(self.client.connection, 'close'):
            await self.client.connection.close()

    async def delete_all_indexes(self) -> None:
        result = await self.execute_query('CALL db.indexes()')
        if not result:
            return

        records, _, _ = result
        drop_tasks = []

        for record in records:
            label = record['label']
            entity_type = record['entitytype']

            for field_name, index_type in record['types'].items():
                if 'RANGE' in index_type:
                    drop_tasks.append(self.execute_query(f'DROP INDEX ON :{label}({field_name})'))
                elif 'FULLTEXT' in index_type:
                    if entity_type == 'NODE':
                        drop_tasks.append(
                            self.execute_query(
                                f'DROP FULLTEXT INDEX FOR (n:{label}) ON (n.{field_name})'
                            )
                        )
                    elif entity_type == 'RELATIONSHIP':
                        drop_tasks.append(
                            self.execute_query(
                                f'DROP FULLTEXT INDEX FOR ()-[e:{label}]-() ON (e.{field_name})'
                            )
                        )

        if drop_tasks:
            await asyncio.gather(*drop_tasks)

    async def build_indices_and_constraints(self, delete_existing=False):
        if delete_existing:
            await self.delete_all_indexes()
        index_queries = get_range_indices(self.provider) + get_fulltext_indices(self.provider)
        for query in index_queries:
            await self.execute_query(query)

    def clone(self, database: str) -> 'GraphDriver':
        """
        Returns a shallow copy of this driver with a different default database.
        Reuses the same connection (e.g. FalkorDB, Neo4j).

        NOTE: _skip_index_init=True prevents auto-creation of graphs when cloning.
        FalkorDB's select_graph() auto-creates non-existent graphs, which would
        recreate a graph we're about to delete.
        """
        if database == self._database:
            cloned = self
        elif database == self.default_group_id:
            cloned = FalkorDriver(falkor_db=self.client, _skip_index_init=True)
        else:
            # Create a new instance of FalkorDriver with the same connection but a different database
            cloned = FalkorDriver(falkor_db=self.client, database=database, _skip_index_init=True)

        return cloned

    async def health_check(self) -> None:
        """Check FalkorDB connectivity by running a simple query."""
        try:
            await self.execute_query('MATCH (n) RETURN 1 LIMIT 1')
            return None
        except Exception as e:
            print(f'FalkorDB health check failed: {e}')
            raise

    @staticmethod
    def convert_datetimes_to_strings(obj):
        if isinstance(obj, dict):
            return {k: FalkorDriver.convert_datetimes_to_strings(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [FalkorDriver.convert_datetimes_to_strings(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(FalkorDriver.convert_datetimes_to_strings(item) for item in obj)
        elif isinstance(obj, datetime):
            return obj.isoformat()
        else:
            return obj

    def sanitize(self, query: str) -> str:
        """
        Replace FalkorDB special characters with whitespace.
        Based on FalkorDB tokenization rules: ,.<>{}[]"':;!@#$%^&*()-+=~
        """
        # FalkorDB separator characters that break text into tokens
        separator_map = str.maketrans(
            {
                ',': ' ',
                '.': ' ',
                '<': ' ',
                '>': ' ',
                '{': ' ',
                '}': ' ',
                '[': ' ',
                ']': ' ',
                '"': ' ',
                "'": ' ',
                ':': ' ',
                ';': ' ',
                '!': ' ',
                '@': ' ',
                '#': ' ',
                '$': ' ',
                '%': ' ',
                '^': ' ',
                '&': ' ',
                '*': ' ',
                '(': ' ',
                ')': ' ',
                '-': ' ',
                '+': ' ',
                '=': ' ',
                '~': ' ',
                '?': ' ',
                '|': ' ',
                '/': ' ',
                '\\': ' ',
            }
        )
        sanitized = query.translate(separator_map)
        # Clean up multiple spaces
        sanitized = ' '.join(sanitized.split())
        return sanitized

    def build_fulltext_query(
        self, query: str, group_ids: list[str] | None = None, max_query_length: int = 128
    ) -> str:
        """
        Build a fulltext query string for FalkorDB using RedisSearch syntax.
        FalkorDB uses RedisSearch-like syntax where:
        - Field queries use @ prefix: @field:value
        - Multiple values for same field: (@field:value1|value2)
        - Text search doesn't need @ prefix for content fields
        - AND is implicit with space: (@group_id:value) (text)
        - OR uses pipe within parentheses: (@group_id:value1|value2)
        """
        # FalkorDB uses separate Redis graphs per group_id.
        # The @handle_multiple_group_ids decorator already clones the driver
        # to the correct graph, so @group_id filtering in the fulltext query
        # is unnecessary and fails (group_id is not in the fulltext index).
        group_filter = ''

        sanitized_query = self.sanitize(query)

        # Remove stopwords and empty tokens from the sanitized query
        query_words = sanitized_query.split()
        filtered_words = [word for word in query_words if word and word.lower() not in STOPWORDS]

        # Cap OR-terms to prevent fulltext explosion on long queries
        if len(filtered_words) > MAX_FULLTEXT_TERMS:
            filtered_words = filtered_words[:MAX_FULLTEXT_TERMS]

        sanitized_query = ' | '.join(filtered_words)

        # If the query is too long return no query
        if len(sanitized_query.split(' ')) + len(group_ids or '') >= max_query_length:
            return ''

        full_query = group_filter + ' (' + sanitized_query + ')'

        return full_query

    async def copy_group(self, source_group_id: str, target_group_id: str) -> None:
        """
        Copy a FalkorDB graph to a new name using GRAPH.COPY Redis command.

        In FalkorDB, each group_id is a separate Redis graph key.
        """
        if source_group_id == target_group_id:
            raise ValueError('Source and target group IDs must be different')

        redis_conn = self.client.connection
        await redis_conn.execute_command('GRAPH.COPY', source_group_id, target_group_id)
        logger.info(f'Copied graph {source_group_id} to {target_group_id}')

    async def rename_group(self, old_group_id: str, new_group_id: str) -> None:
        """
        Rename a FalkorDB graph by copying to new name and deleting the old one.

        In FalkorDB, each group_id is a separate Redis graph key.
        Uses GRAPH.COPY + GRAPH.DELETE since there's no GRAPH.RENAME command.
        """
        if old_group_id == new_group_id:
            raise ValueError('Old and new group IDs must be different')

        redis_conn = self.client.connection

        # Copy to new name
        await redis_conn.execute_command('GRAPH.COPY', old_group_id, new_group_id)
        logger.info(f'Copied graph {old_group_id} to {new_group_id}')

        # Delete old graph
        try:
            await redis_conn.execute_command('GRAPH.DELETE', old_group_id)
            logger.info(f'Deleted old graph {old_group_id}')
        except Exception as e:
            logger.error(
                f'Failed to delete old graph {old_group_id} after copying to '
                f'{new_group_id}. Data exists in both. Manual cleanup required: {e}'
            )
            raise

    async def list_groups(self) -> list[str]:
        """
        List all FalkorDB graphs (groups).

        In FalkorDB, each group is stored as a separate Redis key of type 'graphdata'.
        Returns all graph names excluding system/internal keys.

        NOTE: Don't exclude self._database because it can change when processing
        episodes for different groups (Graphiti.add_episode clones the driver).
        """
        # Excluded system/internal graphs (hardcoded, not based on current _database)
        excluded_graphs = {'graphiti', 'default_db'}
        logger.debug(f'list_groups: excluded={excluded_graphs}')

        redis_conn = self.client.connection
        group_ids = []

        # Scan keys (non-blocking, cursor-based)
        keys = []
        async for key in redis_conn.scan_iter(match='*', count=100):
            keys.append(key)
        logger.debug(f'list_groups: found {len(keys)} keys')

        for key in keys:
            # Decode if bytes
            if isinstance(key, bytes):
                key = key.decode('utf-8')

            # Skip internal/system keys
            if key.startswith('_') or key.startswith('graphiti:') or key.startswith('telemetry{'):
                continue

            # Skip known system graphs
            if key.lower() in excluded_graphs:
                continue

            # Check if it's a FalkorDB graph
            key_type = await redis_conn.type(key)
            if isinstance(key_type, bytes):
                key_type = key_type.decode('utf-8')

            if key_type == 'graphdata':
                group_ids.append(key)
                logger.debug(f'list_groups: found graph {key}')

        logger.debug(f'list_groups: returning {len(group_ids)} groups')
        return sorted(group_ids)

    async def delete_group(self, group_id: str) -> None:
        """
        Delete a FalkorDB graph completely.

        In FalkorDB, each group_id is a separate Redis graph key.
        Uses GRAPH.DELETE to remove the entire graph.
        """
        redis_conn = self.client.connection
        await redis_conn.execute_command('GRAPH.DELETE', group_id)
        logger.info(f'Deleted graph {group_id}')
