#!/usr/bin/env python3
"""
Graphiti MCP Server - Exposes Graphiti functionality through the Model Context Protocol (MCP)
"""

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv
from graphiti_core import Graphiti
from graphiti_core.driver.driver import GraphProvider
from graphiti_core.edges import EntityEdge
from graphiti_core.nodes import EntityNode, EpisodeType, EpisodicNode
from graphiti_core.search.search_filters import SearchFilters
from graphiti_core.utils.maintenance.graph_data_operations import clear_data
from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings
from pydantic import BaseModel, Field, create_model
from starlette.responses import JSONResponse

from config.schema import GraphitiConfig, ServerConfig
from models.response_types import (
    EdgeListResponse,
    EntityTypesResponse,
    EpisodeSearchResponse,
    ErrorResponse,
    FactSearchResponse,
    NodeListResponse,
    NodeResult,
    NodeSearchResponse,
    StatusResponse,
    SuccessResponse,
)
from services.factories import DatabaseDriverFactory, EmbedderFactory, LLMClientFactory
from services.entity_type_service import EntityTypeService
from services.queue_service import QueueBackend, create_queue_backend
from utils.formatting import format_fact_result, format_node_result

# Load .env file from mcp_server directory
mcp_server_dir = Path(__file__).parent.parent
env_file = mcp_server_dir / '.env'
if env_file.exists():
    load_dotenv(env_file)
else:
    # Try current working directory as fallback
    load_dotenv()


# Semaphore limit for concurrent Graphiti operations.
#
# This controls how many episodes can be processed simultaneously. Each episode
# processing involves multiple LLM calls (entity extraction, deduplication, etc.),
# so the actual number of concurrent LLM requests will be higher.
#
# TUNING GUIDELINES:
#
# LLM Provider Rate Limits (requests per minute):
# - OpenAI Tier 1 (free):     3 RPM   -> SEMAPHORE_LIMIT=1-2
# - OpenAI Tier 2:            60 RPM   -> SEMAPHORE_LIMIT=5-8
# - OpenAI Tier 3:           500 RPM   -> SEMAPHORE_LIMIT=10-15
# - OpenAI Tier 4:         5,000 RPM   -> SEMAPHORE_LIMIT=20-50
# - Anthropic (default):     50 RPM   -> SEMAPHORE_LIMIT=5-8
# - Anthropic (high tier): 1,000 RPM   -> SEMAPHORE_LIMIT=15-30
# - Azure OpenAI (varies):  Consult your quota -> adjust accordingly
#
# SYMPTOMS:
# - Too high: 429 rate limit errors, increased costs from parallel processing
# - Too low: Slow throughput, underutilized API quota
#
# MONITORING:
# - Watch logs for rate limit errors (429)
# - Monitor episode processing times
# - Check LLM provider dashboard for actual request rates
#
# DEFAULT: 10 (suitable for OpenAI Tier 3, mid-tier Anthropic)
SEMAPHORE_LIMIT = int(os.getenv('SEMAPHORE_LIMIT', 10))


# Configure structured logging with timestamps
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
DATE_FORMAT = '%Y-%m-%d %H:%M:%S'

logging.basicConfig(
    level=logging.INFO,
    format=LOG_FORMAT,
    datefmt=DATE_FORMAT,
    stream=sys.stderr,
)

# Configure specific loggers
logging.getLogger('uvicorn').setLevel(logging.INFO)
logging.getLogger('uvicorn.access').setLevel(logging.WARNING)  # Reduce access log noise
logging.getLogger('mcp.server.streamable_http_manager').setLevel(
    logging.WARNING
)  # Reduce MCP noise


# Patch uvicorn's logging config to use our format
def configure_uvicorn_logging():
    """Configure uvicorn loggers to match our format after they're created."""
    for logger_name in ['uvicorn', 'uvicorn.error', 'uvicorn.access']:
        uvicorn_logger = logging.getLogger(logger_name)
        # Remove existing handlers and add our own with proper formatting
        uvicorn_logger.handlers.clear()
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT))
        uvicorn_logger.addHandler(handler)
        uvicorn_logger.propagate = False


logger = logging.getLogger(__name__)


def build_transport_security(server_config: ServerConfig) -> TransportSecuritySettings | None:
    """Build TransportSecuritySettings from server configuration.

    Returns None if no custom settings are needed (uses SDK defaults).
    """
    # If no custom hosts/origins configured, return None to use SDK defaults
    if server_config.allowed_hosts is None and server_config.allowed_origins is None:
        return None

    # TransportSecuritySettings requires lists, not None values
    # Use empty list as default when not specified
    return TransportSecuritySettings(
        enable_dns_rebinding_protection=server_config.enable_dns_rebinding_protection,
        allowed_hosts=server_config.allowed_hosts or [],
        allowed_origins=server_config.allowed_origins or [],
    )


# Create global config instance - will be properly initialized later
config: GraphitiConfig

# MCP server instructions
GRAPHITI_MCP_INSTRUCTIONS = """
Graphiti is a memory service for AI agents built on a knowledge graph. Graphiti performs well
with dynamic data such as user interactions, changing enterprise data, and external information.

Graphiti transforms information into a richly connected knowledge network, allowing you to 
capture relationships between concepts, entities, and information. The system organizes data as episodes 
(content snippets), nodes (entities), and facts (relationships between entities), creating a dynamic, 
queryable memory store that evolves with new information. Graphiti supports multiple data formats, including 
structured JSON data, enabling seamless integration with existing data pipelines and systems.

Facts contain temporal metadata, allowing you to track the time of creation and whether a fact is invalid 
(superseded by new information).

Key capabilities:
1. Add episodes (text, messages, or JSON) to the knowledge graph with the add_memory tool
2. Search for nodes (entities) in the graph using natural language queries with search_nodes
3. Find relevant facts (relationships between entities) with search_facts
4. Retrieve specific entity edges or episodes by UUID
5. Manage the knowledge graph with tools like delete_episode, delete_entity_edge, and clear_graph

The server connects to a database for persistent storage and uses language models for certain operations. 
Each piece of information is organized by group_id, allowing you to maintain separate knowledge domains.

When adding information, provide descriptive names and detailed content to improve search quality. 
When searching, use specific queries and consider filtering by group_id for more relevant results.

For optimal performance, ensure the database is properly configured and accessible, and valid 
API keys are provided for any language model operations.
"""

# MCP server instance
mcp = FastMCP(
    'Graphiti Agent Memory',
    instructions=GRAPHITI_MCP_INSTRUCTIONS,
)

# Global services
graphiti_service: Optional['GraphitiService'] = None
queue_service: QueueBackend | None = None
entity_type_service: EntityTypeService | None = None

# Global client for backward compatibility
graphiti_client: Graphiti | None = None
semaphore: asyncio.Semaphore


class GraphitiService:
    """Graphiti service using the unified configuration system."""

    def __init__(
        self,
        config: GraphitiConfig,
        semaphore_limit: int = 10,
        entity_type_svc: EntityTypeService | None = None,
    ):
        self.config = config
        self.semaphore_limit = semaphore_limit
        self.semaphore = asyncio.Semaphore(semaphore_limit)
        self.client: Graphiti | None = None
        self.entity_types = None
        self._entity_type_service = entity_type_svc

    async def initialize(self) -> None:
        """Initialize the Graphiti client with factory-created components."""
        try:
            # Create clients using factories
            llm_client = None
            embedder_client = None

            # Create LLM client based on configured provider
            try:
                llm_client = LLMClientFactory.create(self.config.llm)
            except Exception as e:
                logger.warning(f'Failed to create LLM client: {e}')

            # Create embedder client based on configured provider
            try:
                embedder_client = EmbedderFactory.create(self.config.embedder)
            except Exception as e:
                logger.warning(f'Failed to create embedder client: {e}')

            # Create cross-encoder/reranker client
            cross_encoder = None
            try:
                reranker_cfg = self.config.reranker
                if reranker_cfg.provider.lower() == 'openai' and reranker_cfg.providers.openai:
                    from graphiti_core.cross_encoder.openai_reranker_client import (
                        OpenAIRerankerClient,
                    )
                    from graphiti_core.llm_client import LLMConfig as CoreLLMConfig

                    openai_cfg = reranker_cfg.providers.openai
                    reranker_llm_config = CoreLLMConfig(
                        api_key=openai_cfg.api_key,
                        base_url=openai_cfg.api_url,
                        model=reranker_cfg.model,
                    )
                    cross_encoder = OpenAIRerankerClient(config=reranker_llm_config)
                    logger.info(
                        f'Created reranker client: {reranker_cfg.provider} / {reranker_cfg.model or "default"}'
                    )
            except Exception as e:
                logger.warning(f'Failed to create reranker client, using default: {e}')

            # Get database configuration
            db_config = DatabaseDriverFactory.create_config(self.config.database)

            # Get entity types from EntityTypeService (DB) or fall back to config
            custom_types = None
            if self._entity_type_service:
                # Use entity types from database
                custom_types = await self._entity_type_service.get_as_pydantic_models()
                if custom_types:
                    logger.info(
                        f'Loaded {len(custom_types)} entity types from database: '
                        f'{list(custom_types.keys())}'
                    )
            elif self.config.graphiti.entity_types:
                # Fallback: Build entity types from configuration (legacy mode)
                custom_types = {}
                for entity_type in self.config.graphiti.entity_types:
                    # Build field definitions if entity type has fields
                    field_definitions: dict[str, Any] = {}
                    if entity_type.fields:
                        type_mapping = {
                            'str': str,
                            'int': int,
                            'float': float,
                            'bool': bool,
                        }
                        for field_config in entity_type.fields:
                            python_type = type_mapping.get(field_config.type, str)
                            if field_config.required:
                                field_definitions[field_config.name] = (
                                    python_type,
                                    Field(..., description=field_config.description),
                                )
                            else:
                                field_definitions[field_config.name] = (
                                    python_type | None,
                                    Field(default=None, description=field_config.description),
                                )

                    # Create dynamic Pydantic model with optional fields
                    entity_model = create_model(
                        entity_type.name,
                        __doc__=entity_type.description,
                        **field_definitions,
                    )
                    custom_types[entity_type.name] = entity_model

                    if field_definitions:
                        logger.info(
                            f'Entity type "{entity_type.name}" with fields: '
                            f'{list(field_definitions.keys())}'
                        )
                logger.warning('Using legacy config-based entity types (EntityTypeService not available)')

            # Store entity types for later use
            self.entity_types = custom_types

            # Initialize Graphiti client with appropriate driver
            try:
                if self.config.database.provider.lower() == 'falkordb':
                    # For FalkorDB, create a FalkorDriver instance directly
                    from graphiti_core.driver.falkordb_driver import FalkorDriver

                    falkor_driver = FalkorDriver(
                        host=db_config['host'],
                        port=db_config['port'],
                        password=db_config['password'],
                        database=db_config['database'],
                    )

                    self.client = Graphiti(
                        graph_driver=falkor_driver,
                        llm_client=llm_client,
                        embedder=embedder_client,
                        cross_encoder=cross_encoder,
                        max_coroutines=self.semaphore_limit,
                    )
                else:
                    # For Neo4j (default), use the original approach
                    self.client = Graphiti(
                        uri=db_config['uri'],
                        user=db_config['user'],
                        password=db_config['password'],
                        llm_client=llm_client,
                        embedder=embedder_client,
                        cross_encoder=cross_encoder,
                        max_coroutines=self.semaphore_limit,
                    )
            except Exception as db_error:
                # Check for connection errors
                error_msg = str(db_error).lower()
                if 'connection refused' in error_msg or 'could not connect' in error_msg:
                    db_provider = self.config.database.provider
                    if db_provider.lower() == 'falkordb':
                        raise RuntimeError(
                            f'\n{"=" * 70}\n'
                            f'Database Connection Error: FalkorDB is not running\n'
                            f'{"=" * 70}\n\n'
                            f'FalkorDB at {db_config["host"]}:{db_config["port"]} is not accessible.\n\n'
                            f'To start FalkorDB:\n'
                            f'  - Using Docker Compose: cd mcp_server && docker compose up\n'
                            f'  - Or run FalkorDB manually: docker run -p 6379:6379 falkordb/falkordb\n\n'
                            f'{"=" * 70}\n'
                        ) from db_error
                    elif db_provider.lower() == 'neo4j':
                        raise RuntimeError(
                            f'\n{"=" * 70}\n'
                            f'Database Connection Error: Neo4j is not running\n'
                            f'{"=" * 70}\n\n'
                            f'Neo4j at {db_config.get("uri", "unknown")} is not accessible.\n\n'
                            f'To start Neo4j:\n'
                            f'  - Using Docker Compose: cd mcp_server && docker compose -f docker/docker-compose-neo4j.yml up\n'
                            f'  - Or install Neo4j Desktop from: https://neo4j.com/download/\n'
                            f'  - Or run Neo4j manually: docker run -p 7474:7474 -p 7687:7687 neo4j:latest\n\n'
                            f'{"=" * 70}\n'
                        ) from db_error
                    else:
                        raise RuntimeError(
                            f'\n{"=" * 70}\n'
                            f'Database Connection Error: {db_provider} is not running\n'
                            f'{"=" * 70}\n\n'
                            f'{db_provider} at {db_config.get("uri", "unknown")} is not accessible.\n\n'
                            f'Please ensure {db_provider} is running and accessible.\n\n'
                            f'{"=" * 70}\n'
                        ) from db_error
                # Re-raise other errors
                raise

            # Build indices
            await self.client.build_indices_and_constraints()

            logger.info('Successfully initialized Graphiti client')

            # Log configuration details
            if llm_client:
                logger.info(
                    f'Using LLM provider: {self.config.llm.provider} / {self.config.llm.model}'
                )
            else:
                logger.info('No LLM client configured - entity extraction will be limited')

            if embedder_client:
                logger.info(f'Using Embedder provider: {self.config.embedder.provider}')
            else:
                logger.info('No Embedder client configured - search will be limited')

            if self.entity_types:
                entity_type_names = list(self.entity_types.keys())
                logger.info(f'Using custom entity types: {", ".join(entity_type_names)}')
            else:
                logger.info('Using default entity types')

            logger.info(f'Using database: {self.config.database.provider}')
            logger.info(f'Using group_id: {self.config.graphiti.group_id}')

        except Exception as e:
            logger.error(f'Failed to initialize Graphiti client: {e}')
            raise

    async def get_client(self) -> Graphiti:
        """Get the Graphiti client, initializing if necessary."""
        if self.client is None:
            await self.initialize()
        if self.client is None:
            raise RuntimeError('Failed to initialize Graphiti client')
        return self.client


def _get_driver(client: Graphiti, group_id: str | None = None):
    """Get driver, routing to the correct graph for the given group_id.

    DB-neutral: Only clones the driver for FalkorDB (separate graphs per group_id).
    For Neo4j and other providers, returns the original driver unchanged.
    This matches the pattern used throughout graphiti_core/graphiti.py.
    """
    driver = client.driver
    if group_id and driver.provider == GraphProvider.FALKORDB:
        driver = driver.clone(database=group_id)
    return driver


@mcp.tool()
async def add_memory(
    name: str,
    episode_body: str,
    group_id: str | None = None,
    source: str = 'text',
    source_description: str = '',
    uuid: str | None = None,
) -> SuccessResponse | ErrorResponse:
    """Add an episode to memory. This is the primary way to add information to the graph.

    This function returns immediately and processes the episode addition in the background.
    Episodes for the same group_id are processed sequentially to avoid race conditions.

    IMPORTANT - Entity Resolution Best Practices:
    - Use natural language (source='text') for best entity matching with existing nodes
    - Be consistent with entity names across episodes (e.g., always "Tank" not sometimes "Panzer")
    - Use the same language consistently within a graph to improve entity resolution
    - JSON source may create entities that don't match well with text-based entities

    Args:
        name (str): Name of the episode
        episode_body (str): The content of the episode to persist to memory.
                           RECOMMENDED: Use natural language sentences for better entity resolution.
                           When source='json', this must be a properly escaped JSON string.
        group_id (str, optional): A unique ID for this graph. If not provided, uses the default group_id from CLI
                                 or a generated one.
        source (str, optional): Source type, must be one of:
                               - 'text': For plain text content (default, RECOMMENDED for entity resolution)
                               - 'json': For structured data (may create entities that don't merge well)
                               - 'message': For conversation-style content
        source_description (str, optional): Description of the source
        uuid (str, optional): Optional UUID for the episode

    Examples:
        # RECOMMENDED: Natural language for better entity resolution
        add_memory(
            name="Game Stats",
            episode_body="The Tank entity has 100 hit points and deals 25 damage per shot.",
            source="text",
            source_description="game documentation"
        )

        # JSON (use only when structure is essential, entity matching may be worse)
        add_memory(
            name="Customer Profile",
            episode_body='{"company": {"name": "Acme Technologies"}, "products": [{"id": "P001", "name": "CloudSync"}]}',
            source="json",
            source_description="CRM data"
        )
    """
    global graphiti_service, queue_service

    if graphiti_service is None or queue_service is None:
        return ErrorResponse(error='Services not initialized')

    try:
        # Use the provided group_id or fall back to the default from config
        effective_group_id = group_id or config.graphiti.group_id

        # Try to parse the source as an EpisodeType enum, with fallback to text
        episode_type = EpisodeType.text  # Default
        if source:
            try:
                episode_type = EpisodeType[source.lower()]
            except (KeyError, AttributeError):
                # If the source doesn't match any enum value, use text as default
                logger.warning(f"Unknown source type '{source}', using 'text' as default")
                episode_type = EpisodeType.text

        # Submit to queue service for async processing
        await queue_service.add_episode(
            group_id=effective_group_id,
            name=name,
            content=episode_body,
            source_description=source_description,
            episode_type=episode_type,
            entity_types=graphiti_service.entity_types,
            uuid=uuid or None,  # Ensure None is passed if uuid is None
        )

        return SuccessResponse(
            message=f"Episode '{name}' queued for processing in group '{effective_group_id}'"
        )
    except Exception as e:
        error_msg = str(e)
        logger.error(f'Error queuing episode: {error_msg}')
        return ErrorResponse(error=f'Error queuing episode: {error_msg}')


@mcp.tool()
async def create_entity_node(
    name: str,
    group_id: str | None = None,
    entity_type: str = 'Entity',
    summary: str = '',
    attributes: dict[str, Any] | None = None,
) -> dict[str, Any] | ErrorResponse:
    """Create a new entity node directly in the knowledge graph (without LLM extraction).

    Use this to manually add a specific entity node. For bulk/natural-language ingestion,
    use add_memory instead.

    Args:
        name: Name of the entity
        group_id: Optional group/graph ID. Falls back to the configured default.
        entity_type: Label/type of the entity, e.g. 'Weapon', 'Player' (default: 'Entity')
        summary: Optional description of the entity
        attributes: Optional additional attributes as a dict
    """
    global graphiti_service

    if graphiti_service is None:
        return ErrorResponse(error='Graphiti service not initialized')

    try:
        client = await graphiti_service.get_client()
        effective_group_id = group_id or config.graphiti.group_id

        node = await client.create_entity(
            name=name,
            group_id=effective_group_id,
            entity_type=entity_type,
            summary=summary,
            attributes=attributes,
        )

        return format_node_result(node)
    except Exception as e:
        error_msg = str(e)
        logger.error(f'Error creating entity node: {error_msg}')
        return ErrorResponse(error=f'Error creating entity node: {error_msg}')


@mcp.tool()
async def create_entity_edge(
    source_node_uuid: str,
    target_node_uuid: str,
    name: str,
    fact: str,
    group_id: str | None = None,
) -> dict[str, Any] | ErrorResponse:
    """Create a new edge (relationship) between two entity nodes directly in the knowledge graph.

    Use this to manually add a specific relationship. For bulk/natural-language ingestion,
    use add_memory instead.

    Args:
        source_node_uuid: UUID of the source entity node
        target_node_uuid: UUID of the target entity node
        name: Relationship type in UPPER_SNAKE_CASE, e.g. 'HAS_WEAPON'
        fact: Fact text describing the relationship
        group_id: Optional group/graph ID. Falls back to the configured default.
    """
    global graphiti_service

    if graphiti_service is None:
        return ErrorResponse(error='Graphiti service not initialized')

    try:
        client = await graphiti_service.get_client()
        effective_group_id = group_id or config.graphiti.group_id

        edge = await client.create_edge(
            source_node_uuid=source_node_uuid,
            target_node_uuid=target_node_uuid,
            name=name,
            fact=fact,
            group_id=effective_group_id,
        )

        return format_fact_result(edge)
    except Exception as e:
        error_msg = str(e)
        logger.error(f'Error creating entity edge: {error_msg}')
        return ErrorResponse(error=f'Error creating entity edge: {error_msg}')


@mcp.tool()
async def search_nodes(
    query: str,
    group_ids: list[str] | None = None,
    max_nodes: int = 10,
    entity_types: list[str] | None = None,
) -> NodeSearchResponse | ErrorResponse:
    """Search for nodes in the graph memory.

    Args:
        query: The search query
        group_ids: Optional list of group IDs to filter results
        max_nodes: Maximum number of nodes to return (default: 10)
        entity_types: Optional list of entity type names to filter by
    """
    global graphiti_service

    if graphiti_service is None:
        return ErrorResponse(error='Graphiti service not initialized')

    try:
        client = await graphiti_service.get_client()

        # Use the provided group_ids or fall back to the default from config if none provided
        effective_group_ids = (
            group_ids
            if group_ids is not None
            else [config.graphiti.group_id]
            if config.graphiti.group_id
            else []
        )

        # Create search filters
        search_filters = SearchFilters(
            node_labels=entity_types,
        )

        # Use the search_ method with node search config
        from graphiti_core.search.search_config_recipes import NODE_HYBRID_SEARCH_RRF

        results = await client.search_(
            query=query,
            config=NODE_HYBRID_SEARCH_RRF,
            group_ids=effective_group_ids,
            search_filter=search_filters,
        )

        # Extract nodes from results
        nodes = results.nodes[:max_nodes] if results.nodes else []

        if not nodes:
            return NodeSearchResponse(message='No relevant nodes found', nodes=[])

        # Format the results
        node_results = []
        for node in nodes:
            # Get attributes and ensure no embeddings are included
            attrs = node.attributes if hasattr(node, 'attributes') else {}
            # Remove any embedding keys that might be in attributes
            attrs = {k: v for k, v in attrs.items() if 'embedding' not in k.lower()}

            node_results.append(
                NodeResult(
                    uuid=node.uuid,
                    name=node.name,
                    labels=node.labels if node.labels else [],
                    created_at=node.created_at.isoformat() if node.created_at else None,
                    summary=node.summary,
                    group_id=node.group_id,
                    attributes=attrs,
                )
            )

        return NodeSearchResponse(message='Nodes retrieved successfully', nodes=node_results)
    except Exception as e:
        error_msg = str(e)
        logger.error(f'Error searching nodes: {error_msg}')
        return ErrorResponse(error=f'Error searching nodes: {error_msg}')


@mcp.tool()
async def search_memory_facts(
    query: str,
    group_ids: list[str] | None = None,
    max_facts: int = 10,
    center_node_uuid: str | None = None,
) -> FactSearchResponse | ErrorResponse:
    """Search the graph memory for relevant facts.

    Args:
        query: The search query
        group_ids: Optional list of group IDs to filter results
        max_facts: Maximum number of facts to return (default: 10)
        center_node_uuid: Optional UUID of a node to center the search around
    """
    global graphiti_service

    if graphiti_service is None:
        return ErrorResponse(error='Graphiti service not initialized')

    try:
        # Validate max_facts parameter
        if max_facts <= 0:
            return ErrorResponse(error='max_facts must be a positive integer')

        client = await graphiti_service.get_client()

        # Use the provided group_ids or fall back to the default from config if none provided
        effective_group_ids = (
            group_ids
            if group_ids is not None
            else [config.graphiti.group_id]
            if config.graphiti.group_id
            else []
        )

        relevant_edges = await client.search(
            group_ids=effective_group_ids,
            query=query,
            num_results=max_facts,
            center_node_uuid=center_node_uuid,
        )

        if not relevant_edges:
            return FactSearchResponse(message='No relevant facts found', facts=[])

        facts = [format_fact_result(edge) for edge in relevant_edges]
        return FactSearchResponse(message='Facts retrieved successfully', facts=facts)
    except Exception as e:
        error_msg = str(e)
        logger.error(f'Error searching facts: {error_msg}')
        return ErrorResponse(error=f'Error searching facts: {error_msg}')


@mcp.tool()
async def delete_entity_edge(
    uuid: str, group_id: str | None = None
) -> SuccessResponse | ErrorResponse:
    """Delete an entity edge from the graph memory.

    Args:
        uuid: UUID of the entity edge to delete
        group_id: The group/graph this entity belongs to. Always provide this when you know the group. If you get a 'not found' error, retry with the correct group_id.
    """
    global graphiti_service

    if graphiti_service is None:
        return ErrorResponse(error='Graphiti service not initialized')

    try:
        client = await graphiti_service.get_client()
        driver = _get_driver(client, group_id)

        # Get the entity edge by UUID
        entity_edge = await EntityEdge.get_by_uuid(driver, uuid)
        # Delete the edge using its delete method
        await entity_edge.delete(driver)
        return SuccessResponse(message=f'Entity edge with UUID {uuid} deleted successfully')
    except Exception as e:
        error_msg = str(e)
        logger.error(f'Error deleting entity edge: {error_msg}')
        return ErrorResponse(error=f'Error deleting entity edge: {error_msg}')


@mcp.tool()
async def delete_episode(
    uuid: str, group_id: str | None = None
) -> SuccessResponse | ErrorResponse:
    """Delete an episode from the graph memory.

    Args:
        uuid: UUID of the episode to delete
        group_id: The group/graph this entity belongs to. Always provide this when you know the group. If you get a 'not found' error, retry with the correct group_id.
    """
    global graphiti_service

    if graphiti_service is None:
        return ErrorResponse(error='Graphiti service not initialized')

    try:
        client = await graphiti_service.get_client()
        driver = _get_driver(client, group_id)

        # Get the episodic node by UUID
        episodic_node = await EpisodicNode.get_by_uuid(driver, uuid)
        # Delete the node using its delete method
        await episodic_node.delete(driver)
        return SuccessResponse(message=f'Episode with UUID {uuid} deleted successfully')
    except Exception as e:
        error_msg = str(e)
        logger.error(f'Error deleting episode: {error_msg}')
        return ErrorResponse(error=f'Error deleting episode: {error_msg}')


@mcp.tool()
async def get_entity_edge(
    uuid: str, group_id: str | None = None
) -> dict[str, Any] | ErrorResponse:
    """Get an entity edge from the graph memory by its UUID.

    Args:
        uuid: UUID of the entity edge to retrieve
        group_id: The group/graph this entity belongs to. Always provide this when you know the group. If you get a 'not found' error, retry with the correct group_id.
    """
    global graphiti_service

    if graphiti_service is None:
        return ErrorResponse(error='Graphiti service not initialized')

    try:
        client = await graphiti_service.get_client()
        driver = _get_driver(client, group_id)

        # Get the entity edge directly using the EntityEdge class method
        entity_edge = await EntityEdge.get_by_uuid(driver, uuid)

        # Use the format_fact_result function to serialize the edge
        # Return the Python dict directly - MCP will handle serialization
        return format_fact_result(entity_edge)
    except Exception as e:
        error_msg = str(e)
        logger.error(f'Error getting entity edge: {error_msg}')
        return ErrorResponse(error=f'Error getting entity edge: {error_msg}')


@mcp.tool()
async def get_episodes(
    group_ids: list[str] | None = None,
    max_episodes: int = 10,
) -> EpisodeSearchResponse | ErrorResponse:
    """Get episodes from the graph memory.

    Args:
        group_ids: Optional list of group IDs to filter results
        max_episodes: Maximum number of episodes to return (default: 10)
    """
    global graphiti_service

    if graphiti_service is None:
        return ErrorResponse(error='Graphiti service not initialized')

    try:
        client = await graphiti_service.get_client()

        # Use the provided group_ids or fall back to the default from config if none provided
        effective_group_ids = (
            group_ids
            if group_ids is not None
            else [config.graphiti.group_id]
            if config.graphiti.group_id
            else []
        )

        # Get episodes from the driver directly
        from graphiti_core.nodes import EpisodicNode

        if effective_group_ids:
            episodes = await EpisodicNode.get_by_group_ids(
                client.driver, effective_group_ids, limit=max_episodes
            )
        else:
            # If no group IDs, we need to use a different approach
            # For now, return empty list when no group IDs specified
            episodes = []

        if not episodes:
            return EpisodeSearchResponse(message='No episodes found', episodes=[])

        # Format the results
        episode_results = []
        for episode in episodes:
            episode_dict = {
                'uuid': episode.uuid,
                'name': episode.name,
                'content': episode.content,
                'created_at': episode.created_at.isoformat() if episode.created_at else None,
                'source': episode.source.value
                if hasattr(episode.source, 'value')
                else str(episode.source),
                'source_description': episode.source_description,
                'group_id': episode.group_id,
            }
            episode_results.append(episode_dict)

        return EpisodeSearchResponse(
            message='Episodes retrieved successfully', episodes=episode_results
        )
    except Exception as e:
        error_msg = str(e)
        logger.error(f'Error getting episodes: {error_msg}')
        return ErrorResponse(error=f'Error getting episodes: {error_msg}')


@mcp.tool()
async def clear_graph(group_ids: list[str] | None = None) -> SuccessResponse | ErrorResponse:
    """Clear all data from the graph for specified group IDs.

    Args:
        group_ids: Optional list of group IDs to clear. If not provided, clears the default group.
    """
    global graphiti_service

    if graphiti_service is None:
        return ErrorResponse(error='Graphiti service not initialized')

    try:
        client = await graphiti_service.get_client()

        # Use the provided group_ids or fall back to the default from config if none provided
        effective_group_ids = (
            group_ids or [config.graphiti.group_id] if config.graphiti.group_id else []
        )

        if not effective_group_ids:
            return ErrorResponse(error='No group IDs specified for clearing')

        # Clear data for the specified group IDs
        await clear_data(client.driver, group_ids=effective_group_ids)

        return SuccessResponse(
            message=f'Graph data cleared successfully for group IDs: {", ".join(effective_group_ids)}'
        )
    except Exception as e:
        error_msg = str(e)
        logger.error(f'Error clearing graph: {error_msg}')
        return ErrorResponse(error=f'Error clearing graph: {error_msg}')


@mcp.tool()
async def get_status() -> StatusResponse:
    """Get the status of the Graphiti MCP server and database connection."""
    global graphiti_service

    if graphiti_service is None:
        return StatusResponse(status='error', message='Graphiti service not initialized')

    try:
        client = await graphiti_service.get_client()

        # Test database connection with a simple query
        async with client.driver.session() as session:
            result = await session.run('MATCH (n) RETURN count(n) as count')
            # Consume the result to verify query execution
            if result:
                _ = [record async for record in result]

        # Use the provider from the service's config, not the global
        provider_name = graphiti_service.config.database.provider
        return StatusResponse(
            status='ok',
            message=f'Graphiti MCP server is running and connected to {provider_name} database',
        )
    except Exception as e:
        error_msg = str(e)
        logger.error(f'Error checking database connection: {error_msg}')
        return StatusResponse(
            status='error',
            message=f'Graphiti MCP server is running but database connection failed: {error_msg}',
        )


@mcp.tool()
async def get_entity_types() -> EntityTypesResponse | ErrorResponse:
    """Get all configured entity types with their fields.

    Returns the list of entity types that are used for knowledge extraction.
    Each entity type has a name, description (used as LLM prompt), and optional
    structured fields for attribute extraction.

    Use this to understand what types of entities and attributes will be
    extracted when adding memories to the graph.
    """
    global entity_type_service

    if entity_type_service is None:
        return ErrorResponse(error='EntityTypeService not initialized')

    try:
        # Get entity types from database
        entity_types = await entity_type_service.get_all()
        entity_types_list = []

        for et in entity_types:
            entity_types_list.append({
                'name': et.name,
                'description': et.description,
                'fields': et.fields,
                'source': et.source,
                'created_at': et.created_at,
                'modified_at': et.modified_at,
            })

        return EntityTypesResponse(
            message=f'Found {len(entity_types_list)} entity types',
            entity_types=entity_types_list,
        )
    except Exception as e:
        error_msg = str(e)
        logger.error(f'Error getting entity types: {error_msg}')
        return ErrorResponse(error=f'Error getting entity types: {error_msg}')


@mcp.tool()
async def get_entity_node(
    uuid: str, group_id: str | None = None
) -> dict[str, Any] | ErrorResponse:
    """Get an entity node from the graph memory by its UUID.

    Args:
        uuid: UUID of the entity node to retrieve
        group_id: The group/graph this entity belongs to. Always provide this when you know the group. If you get a 'not found' error, retry with the correct group_id.
    """
    global graphiti_service

    if graphiti_service is None:
        return ErrorResponse(error='Graphiti service not initialized')

    try:
        client = await graphiti_service.get_client()
        driver = _get_driver(client, group_id)
        entity_node = await EntityNode.get_by_uuid(driver, uuid)
        return format_node_result(entity_node)
    except Exception as e:
        error_msg = str(e)
        logger.error(f'Error getting entity node: {error_msg}')
        return ErrorResponse(error=f'Error getting entity node: {error_msg}')


@mcp.tool()
async def get_entity_edges_by_node(
    node_uuid: str,
    group_id: str | None = None,
) -> FactSearchResponse | ErrorResponse:
    """Get all entity edges (facts/relationships) connected to a specific node.

    This is useful for:
    - Seeing all relationships of an entity before modifying or deleting it
    - Finding edges that need to be moved when merging duplicate nodes
    - Understanding the context of a specific entity

    Args:
        node_uuid: UUID of the entity node to get edges for
        group_id: The group/graph this entity belongs to. Always provide this when you know the group. If you get a 'not found' error, retry with the correct group_id.

    Returns:
        List of all edges where this node is either source or target
    """
    global graphiti_service

    if graphiti_service is None:
        return ErrorResponse(error='Graphiti service not initialized')

    try:
        client = await graphiti_service.get_client()
        driver = _get_driver(client, group_id)

        # Get all edges connected to this node
        edges = await EntityEdge.get_by_node_uuid(driver, node_uuid)

        facts = [format_fact_result(edge) for edge in edges]

        return FactSearchResponse(
            message=f'Found {len(facts)} edges connected to node {node_uuid}',
            facts=facts,
        )
    except Exception as e:
        error_msg = str(e)
        logger.error(f'Error getting edges by node: {error_msg}')
        return ErrorResponse(error=f'Error getting edges by node: {error_msg}')


@mcp.tool()
async def list_nodes(
    group_id: str | None = None,
    limit: int = 100,
    cursor: str | None = None,
) -> NodeListResponse | ErrorResponse:
    """List all entity nodes in a graph with pagination support.

    Use this tool to:
    - Get an overview of all entities in a graph
    - Find duplicate nodes (same or similar names)
    - Identify orphan nodes or data quality issues
    - Browse the graph structure without semantic search

    Args:
        group_id: Graph to list nodes from. Uses default if not provided.
        limit: Maximum number of nodes to return (default: 100, max: 500)
        cursor: UUID cursor for pagination. Pass the next_cursor from previous response.

    Returns:
        List of nodes with pagination info. Use next_cursor for subsequent pages.
    """
    global graphiti_service

    if graphiti_service is None:
        return ErrorResponse(error='Graphiti service not initialized')

    try:
        client = await graphiti_service.get_client()

        # Use the provided group_id or fall back to the default from config
        effective_group_id = group_id or config.graphiti.group_id

        if not effective_group_id:
            return ErrorResponse(error='No group_id provided and no default configured')

        # Clamp limit to reasonable bounds
        effective_limit = min(max(1, limit), 500)

        # Get nodes with pagination (request one extra to check if there are more)
        # Use lightweight=True to exclude embedding vectors for better performance
        nodes = await client.get_entities_by_group_id(
            group_id=effective_group_id,
            limit=effective_limit + 1,
            uuid_cursor=cursor,
            lightweight=True,
        )

        # Check if there are more results
        has_more = len(nodes) > effective_limit
        if has_more:
            nodes = nodes[:effective_limit]

        # Format results
        node_results = [format_node_result(node) for node in nodes]

        # Determine next cursor
        next_cursor = nodes[-1].uuid if has_more and nodes else None

        return NodeListResponse(
            message=f'Found {len(node_results)} nodes in group {effective_group_id}',
            nodes=node_results,
            total=len(node_results),
            has_more=has_more,
            next_cursor=next_cursor,
        )
    except Exception as e:
        error_msg = str(e)
        logger.error(f'Error listing nodes: {error_msg}')
        return ErrorResponse(error=f'Error listing nodes: {error_msg}')


@mcp.tool()
async def list_edges(
    group_id: str | None = None,
    limit: int = 100,
    cursor: str | None = None,
) -> EdgeListResponse | ErrorResponse:
    """List all entity edges (facts/relationships) in a graph with pagination support.

    Use this tool to:
    - Get an overview of all relationships in a graph
    - Find redundant or duplicate edges
    - Analyze graph connectivity
    - Browse facts without semantic search

    Args:
        group_id: Graph to list edges from. Uses default if not provided.
        limit: Maximum number of edges to return (default: 100, max: 500)
        cursor: UUID cursor for pagination. Pass the next_cursor from previous response.

    Returns:
        List of edges with pagination info. Use next_cursor for subsequent pages.
    """
    global graphiti_service

    if graphiti_service is None:
        return ErrorResponse(error='Graphiti service not initialized')

    try:
        client = await graphiti_service.get_client()

        # Use the provided group_id or fall back to the default from config
        effective_group_id = group_id or config.graphiti.group_id

        if not effective_group_id:
            return ErrorResponse(error='No group_id provided and no default configured')

        # Clamp limit to reasonable bounds
        effective_limit = min(max(1, limit), 500)

        # Get edges with pagination (request one extra to check if there are more)
        # Use lightweight=True to exclude embedding vectors for better performance
        edges = await client.get_edges_by_group_id(
            group_id=effective_group_id,
            limit=effective_limit + 1,
            uuid_cursor=cursor,
            lightweight=True,
        )

        # Check if there are more results
        has_more = len(edges) > effective_limit
        if has_more:
            edges = edges[:effective_limit]

        # Format results
        edge_results = [format_fact_result(edge) for edge in edges]

        # Determine next cursor
        next_cursor = edges[-1].uuid if has_more and edges else None

        return EdgeListResponse(
            message=f'Found {len(edge_results)} edges in group {effective_group_id}',
            edges=edge_results,
            total=len(edge_results),
            has_more=has_more,
            next_cursor=next_cursor,
        )
    except Exception as e:
        error_msg = str(e)
        logger.error(f'Error listing edges: {error_msg}')
        return ErrorResponse(error=f'Error listing edges: {error_msg}')


@mcp.tool()
async def update_entity_node(
    uuid: str,
    name: str | None = None,
    summary: str | None = None,
    entity_type: str | None = None,
    attributes: dict[str, Any] | None = None,
    group_id: str | None = None,
) -> dict[str, Any] | ErrorResponse:
    """Update an existing entity node in the graph memory.

    Delegates to Graphiti.update_entity() which handles embedding regeneration
    and DB routing internally.

    Use this tool to modify an existing entity when you know its UUID.
    This is preferred over add_memory when correcting facts about existing entities,
    as it ensures the correction is applied to the correct node without creating duplicates.

    Args:
        uuid: UUID of the entity node to update (required)
        name: New name for the entity (optional)
        summary: New summary for the entity (optional)
        entity_type: New entity type/label, e.g. 'Weapon', 'Player' (optional)
        attributes: Attributes to merge into existing attributes (optional)
        group_id: The group/graph this entity belongs to. Always provide this when you know the group. If you get a 'not found' error, retry with the correct group_id.

    Returns:
        The updated entity node data, or an error response
    """
    global graphiti_service

    if graphiti_service is None:
        return ErrorResponse(error='Graphiti service not initialized')

    try:
        client = await graphiti_service.get_client()
        effective_group_id = group_id or config.graphiti.group_id

        node = await client.update_entity(
            uuid=uuid,
            name=name,
            summary=summary,
            entity_type=entity_type,
            attributes=attributes,
            group_id=effective_group_id,
        )

        return format_node_result(node)
    except Exception as e:
        error_msg = str(e)
        logger.error(f'Error updating entity: {error_msg}')
        return ErrorResponse(error=f'Error updating entity: {error_msg}')


@mcp.tool()
async def update_entity_edge(
    uuid: str,
    fact: str | None = None,
    name: str | None = None,
    group_id: str | None = None,
) -> dict[str, Any] | ErrorResponse:
    """Update an existing entity edge (fact/relationship) in the graph memory.

    Delegates to Graphiti.update_edge() which handles embedding regeneration
    and DB routing internally.

    Use this tool to modify an existing relationship when you know its UUID.
    This is particularly useful for correcting the fact text or changing
    the relationship type (name).

    Note: Endpoint changes (source/target node) are not supported. To move an edge,
    delete it and create a new one with create_entity_edge.

    Args:
        uuid: UUID of the entity edge to update (required)
        fact: New fact text describing the relationship (optional)
        name: New relationship type name in UPPER_SNAKE_CASE (optional)
        group_id: The group/graph this entity belongs to. Always provide this when you know the group. If you get a 'not found' error, retry with the correct group_id.

    Returns:
        The updated entity edge data, or an error response
    """
    global graphiti_service

    if graphiti_service is None:
        return ErrorResponse(error='Graphiti service not initialized')

    try:
        client = await graphiti_service.get_client()
        effective_group_id = group_id or config.graphiti.group_id

        edge = await client.update_edge(
            uuid=uuid,
            name=name,
            fact=fact,
            group_id=effective_group_id,
        )

        return format_fact_result(edge)
    except Exception as e:
        error_msg = str(e)
        logger.error(f'Error updating entity edge: {error_msg}')
        return ErrorResponse(error=f'Error updating entity edge: {error_msg}')


@mcp.tool()
async def delete_entity_node(
    uuid: str, group_id: str | None = None
) -> SuccessResponse | ErrorResponse:
    """Delete an entity node from the graph memory.

    IMPORTANT: This will permanently delete all edges connected to this node!

    When merging duplicate nodes, use this workflow instead:
    1. Use get_entity_edges_by_node() to list all edges of the duplicate node
    2. Use update_entity_edge() to move each edge to the correct node
    3. Only then use delete_entity_node() to remove the orphaned duplicate

    Args:
        uuid: UUID of the entity node to delete
        group_id: The group/graph this entity belongs to. Always provide this when you know the group. If you get a 'not found' error, retry with the correct group_id.
    """
    global graphiti_service

    if graphiti_service is None:
        return ErrorResponse(error='Graphiti service not initialized')

    try:
        client = await graphiti_service.get_client()
        driver = _get_driver(client, group_id)

        # First check how many edges are connected
        edges = await EntityEdge.get_by_node_uuid(driver, uuid)

        # Get the node to verify it exists and get its name
        entity_node = await EntityNode.get_by_uuid(driver, uuid)

        if edges:
            logger.warning(
                f'Deleting entity node {uuid} ({entity_node.name}) '
                f'with {len(edges)} connected edges'
            )

        # Delete all connected edges first
        for edge in edges:
            await edge.delete(driver)

        # Delete the node
        await entity_node.delete(driver)

        return SuccessResponse(
            message=f'Deleted entity node "{entity_node.name}" ({uuid}) '
            f'and {len(edges)} connected edges'
        )
    except Exception as e:
        error_msg = str(e)
        logger.error(f'Error deleting entity node: {error_msg}')
        return ErrorResponse(error=f'Error deleting entity node: {error_msg}')


@mcp.custom_route('/health', methods=['GET'])
async def health_check(request) -> JSONResponse:
    """Health check endpoint for Docker and load balancers."""
    return JSONResponse({'status': 'healthy', 'service': 'graphiti-mcp'})


# =============================================================================
# HTTP Endpoints for Entity Types (DB-neutral, file-based)
# =============================================================================


@mcp.custom_route('/entity-types', methods=['GET'])
async def http_get_entity_types(request) -> JSONResponse:
    """List all entity types."""
    global entity_type_service

    if entity_type_service is None:
        return JSONResponse({'error': 'EntityTypeService not initialized'}, status_code=500)

    try:
        entity_types = await entity_type_service.get_all()
        return JSONResponse([et.to_dict() for et in entity_types])
    except Exception as e:
        logger.error(f'Error getting entity types: {e}')
        return JSONResponse({'error': 'Internal error retrieving entity types'}, status_code=500)


@mcp.custom_route('/entity-types', methods=['POST'])
async def http_create_entity_type(request) -> JSONResponse:
    """Create a new entity type."""
    global entity_type_service

    if entity_type_service is None:
        return JSONResponse({'error': 'EntityTypeService not initialized'}, status_code=500)

    try:
        data = await request.json()
        name = data.get('name')
        description = data.get('description')
        fields = data.get('fields', [])

        if not name or not description:
            return JSONResponse(
                {'error': 'name and description are required'},
                status_code=400,
            )

        entity_type = await entity_type_service.create(
            name=name,
            description=description,
            fields=fields,
        )
        return JSONResponse(entity_type.to_dict(), status_code=201)
    except ValueError as e:
        return JSONResponse({'error': str(e)}, status_code=409)
    except Exception as e:
        logger.error(f'Error creating entity type: {e}')
        return JSONResponse({'error': 'Internal error creating entity type'}, status_code=500)


@mcp.custom_route('/entity-types/reset', methods=['POST'])
async def http_reset_entity_types(request) -> JSONResponse:
    """Reset entity types to config defaults."""
    global entity_type_service

    if entity_type_service is None:
        return JSONResponse({'error': 'EntityTypeService not initialized'}, status_code=500)

    try:
        count = await entity_type_service.reset_to_defaults()
        return JSONResponse({'message': f'Reset {count} entity types', 'count': count})
    except RuntimeError as e:
        return JSONResponse({'error': str(e)}, status_code=500)
    except Exception as e:
        logger.error(f'Error resetting entity types: {e}')
        return JSONResponse({'error': 'Internal error resetting entity types'}, status_code=500)


@mcp.custom_route('/entity-types/{name}', methods=['GET'])
async def http_get_entity_type(request) -> JSONResponse:
    """Get a specific entity type by name."""
    global entity_type_service

    if entity_type_service is None:
        return JSONResponse({'error': 'EntityTypeService not initialized'}, status_code=500)

    try:
        name = request.path_params['name']
        entity_type = await entity_type_service.get_by_name(name)
        if not entity_type:
            return JSONResponse({'error': f'Entity type "{name}" not found'}, status_code=404)
        return JSONResponse(entity_type.to_dict())
    except Exception as e:
        logger.error(f'Error getting entity type: {e}')
        return JSONResponse({'error': 'Internal error retrieving entity type'}, status_code=500)


@mcp.custom_route('/entity-types/{name}', methods=['PUT'])
async def http_update_entity_type(request) -> JSONResponse:
    """Update an entity type."""
    global entity_type_service

    if entity_type_service is None:
        return JSONResponse({'error': 'EntityTypeService not initialized'}, status_code=500)

    try:
        name = request.path_params['name']
        data = await request.json()

        entity_type = await entity_type_service.update(
            name=name,
            description=data.get('description'),
            fields=data.get('fields'),
        )
        return JSONResponse(entity_type.to_dict())
    except ValueError as e:
        return JSONResponse({'error': str(e)}, status_code=404)
    except Exception as e:
        logger.error(f'Error updating entity type: {e}')
        return JSONResponse({'error': 'Internal error updating entity type'}, status_code=500)


@mcp.custom_route('/entity-types/{name}', methods=['DELETE'])
async def http_delete_entity_type(request) -> JSONResponse:
    """Delete an entity type."""
    global entity_type_service

    if entity_type_service is None:
        return JSONResponse({'error': 'EntityTypeService not initialized'}, status_code=500)

    try:
        name = request.path_params['name']
        success = await entity_type_service.delete(name)
        if not success:
            return JSONResponse({'error': f'Entity type "{name}" not found'}, status_code=404)
        return JSONResponse({'message': f'Deleted {name}'})
    except Exception as e:
        logger.error(f'Error deleting entity type: {e}')
        return JSONResponse({'error': 'Internal error deleting entity type'}, status_code=500)


@mcp.custom_route('/queue/status', methods=['GET'])
async def queue_status(request) -> JSONResponse:
    """Get queue processing status for UI polling.

    Returns:
        - total_pending: Total messages waiting across all groups
        - currently_processing: Number of active workers
        - groups: Per-group breakdown (optional, if group_id param provided)
    """
    global queue_service

    if queue_service is None:
        return JSONResponse({
            'total_pending': 0,
            'currently_processing': 0,
            'error': 'Queue service not initialized',
        })

    try:
        total_pending, currently_processing, groups = await queue_service.get_status()
        result = {
            'total_pending': total_pending,
            'currently_processing': currently_processing,
        }
        if groups:
            result['groups'] = groups
        return JSONResponse(result)
    except Exception as e:
        logger.error(f'Error getting queue status: {e}')
        return JSONResponse({
            'total_pending': 0,
            'currently_processing': 0,
            'error': 'Internal server error',
        })


# =============================================================================
# HTTP Endpoints for Groups and Stats (DB-neutral)
# =============================================================================


@mcp.custom_route('/groups', methods=['GET'])
async def http_get_groups(request) -> JSONResponse:
    """Get all available group IDs from the database.

    Uses the DB-neutral Graphiti.get_groups() method which delegates to
    the driver's list_groups() implementation (works for all 4 DB providers).
    """
    global graphiti_service

    if graphiti_service is None:
        return JSONResponse({'error': 'Graphiti service not initialized'}, status_code=500)

    try:
        client = await graphiti_service.get_client()
        groups = await client.get_groups()
        return JSONResponse({'groups': groups})

    except Exception as e:
        logger.error(f'Error getting groups: {e}')
        return JSONResponse({'error': 'Internal server error'}, status_code=500)


@mcp.custom_route('/stats', methods=['GET'])
async def http_get_stats(request) -> JSONResponse:
    """Get graph statistics.

    Query parameters:
        - group_id: Optional group ID to filter stats
    """
    global graphiti_service

    if graphiti_service is None:
        return JSONResponse({'error': 'Graphiti service not initialized'}, status_code=500)

    try:
        client = await graphiti_service.get_client()
        group_id = request.query_params.get('group_id')

        # Use Graphiti's built-in stats method
        stats = await client.get_graph_stats(group_id=group_id)

        return JSONResponse({
            'nodes': stats.get('node_count', 0),
            'edges': stats.get('edge_count', 0),
            'episodes': stats.get('episode_count', 0),
        })

    except Exception as e:
        logger.error(f'Error getting stats: {e}')
        return JSONResponse({'error': 'Internal server error'}, status_code=500)

async def initialize_server() -> ServerConfig:
    """Parse CLI arguments and initialize the Graphiti server configuration."""
    global config, graphiti_service, queue_service, entity_type_service, graphiti_client, semaphore

    parser = argparse.ArgumentParser(
        description='Run the Graphiti MCP server with YAML configuration support'
    )

    # Configuration file argument
    # Default to config/config.yaml relative to the mcp_server directory
    default_config = Path(__file__).parent.parent / 'config' / 'config.yaml'
    parser.add_argument(
        '--config',
        type=Path,
        default=default_config,
        help='Path to YAML configuration file (default: config/config.yaml)',
    )

    # Transport arguments
    parser.add_argument(
        '--transport',
        choices=['sse', 'stdio', 'http'],
        help='Transport to use: http (recommended, default), stdio (standard I/O), or sse (deprecated)',
    )
    parser.add_argument(
        '--host',
        help='Host to bind the MCP server to',
    )
    parser.add_argument(
        '--port',
        type=int,
        help='Port to bind the MCP server to',
    )

    # Provider selection arguments
    parser.add_argument(
        '--llm-provider',
        choices=['openai', 'azure_openai', 'anthropic', 'gemini', 'groq'],
        help='LLM provider to use',
    )
    parser.add_argument(
        '--embedder-provider',
        choices=['openai', 'azure_openai', 'gemini', 'voyage'],
        help='Embedder provider to use',
    )
    parser.add_argument(
        '--database-provider',
        choices=['neo4j', 'falkordb'],
        help='Database provider to use',
    )

    # LLM configuration arguments
    parser.add_argument('--model', help='Model name to use with the LLM client')
    parser.add_argument('--small-model', help='Small model name to use with the LLM client')
    parser.add_argument(
        '--temperature', type=float, help='Temperature setting for the LLM (0.0-2.0)'
    )

    # Embedder configuration arguments
    parser.add_argument('--embedder-model', help='Model name to use with the embedder')

    # Graphiti-specific arguments
    parser.add_argument(
        '--group-id',
        help='Namespace for the graph. If not provided, uses config file or generates random UUID.',
    )
    parser.add_argument(
        '--user-id',
        help='User ID for tracking operations',
    )
    parser.add_argument(
        '--destroy-graph',
        action='store_true',
        help='Destroy all Graphiti graphs on startup',
    )

    args = parser.parse_args()

    # Set config path in environment for the settings to pick up
    if args.config:
        os.environ['CONFIG_PATH'] = str(args.config)

    # Load configuration with environment variables and YAML
    config = GraphitiConfig()

    # Apply CLI overrides
    config.apply_cli_overrides(args)

    # Also apply legacy CLI args for backward compatibility
    if hasattr(args, 'destroy_graph'):
        config.destroy_graph = args.destroy_graph

    # Log configuration details
    logger.info('Using configuration:')
    logger.info(f'  - LLM: {config.llm.provider} / {config.llm.model}')
    logger.info(f'  - Embedder: {config.embedder.provider} / {config.embedder.model}')
    logger.info(
        f'  - Reranker: {config.reranker.provider} / {config.reranker.model or "default"}'
    )
    logger.info(f'  - Database: {config.database.provider}')
    logger.info(f'  - Group ID: {config.graphiti.group_id}')
    logger.info(f'  - Transport: {config.server.transport}')

    # Log graphiti-core version
    try:
        import graphiti_core

        graphiti_version = getattr(graphiti_core, '__version__', 'unknown')
        logger.info(f'  - Graphiti Core: {graphiti_version}')
    except Exception:
        # Check for Docker-stored version file
        version_file = Path('/app/.graphiti-core-version')
        if version_file.exists():
            graphiti_version = version_file.read_text().strip()
            logger.info(f'  - Graphiti Core: {graphiti_version}')
        else:
            logger.info('  - Graphiti Core: version unavailable')

    # Handle graph destruction if requested
    if hasattr(config, 'destroy_graph') and config.destroy_graph:
        logger.warning('Destroying all Graphiti graphs as requested...')
        temp_service = GraphitiService(config, SEMAPHORE_LIMIT)
        await temp_service.initialize()
        client = await temp_service.get_client()
        await clear_data(client.driver)
        logger.info('All graphs destroyed')

    # Initialize EntityTypeService first (file-based storage)
    entity_type_service = EntityTypeService(data_dir=Path(config.server.data_dir))
    await entity_type_service.initialize(config=config)

    # Initialize services
    graphiti_service = GraphitiService(config, SEMAPHORE_LIMIT, entity_type_service)

    # Create queue backend using factory (auto-detects Redis from FalkorDB or uses In-Memory)
    queue_service = create_queue_backend(config)
    await graphiti_service.initialize()

    # Set global client for backward compatibility
    graphiti_client = await graphiti_service.get_client()
    semaphore = graphiti_service.semaphore

    # Initialize queue service with the client and entity types
    await queue_service.initialize(
        graphiti_client,
        entity_types=graphiti_service.entity_types,
    )

    # Set MCP server settings
    if config.server.host:
        mcp.settings.host = config.server.host
    if config.server.port:
        mcp.settings.port = config.server.port

    # Configure transport security if custom hosts/origins are specified
    transport_security = build_transport_security(config.server)
    if transport_security:
        mcp.settings.transport_security = transport_security
        logger.info('Transport security configured:')
        if config.server.allowed_hosts:
            logger.info(f'  - Allowed hosts: {", ".join(config.server.allowed_hosts)}')
        if config.server.allowed_origins:
            logger.info(f'  - Allowed origins: {", ".join(config.server.allowed_origins)}')
        logger.info(
            f'  - DNS rebinding protection: {config.server.enable_dns_rebinding_protection}'
        )
    else:
        # Log default security settings
        logger.info(
            f'Transport security: using SDK defaults '
            f'(allowed_hosts: {mcp.settings.transport_security.allowed_hosts})'
        )

    # Return MCP configuration for transport
    return config.server


async def run_mcp_server():
    """Run the MCP server in the current event loop."""
    # Initialize the server
    mcp_config = await initialize_server()

    # Run the server with configured transport
    logger.info(f'Starting MCP server with transport: {mcp_config.transport}')
    if mcp_config.transport == 'stdio':
        await mcp.run_stdio_async()
    elif mcp_config.transport == 'sse':
        logger.info(
            f'Running MCP server with SSE transport on {mcp.settings.host}:{mcp.settings.port}'
        )
        logger.info(f'Access the server at: http://{mcp.settings.host}:{mcp.settings.port}/sse')
        await mcp.run_sse_async()
    elif mcp_config.transport == 'http':
        # Use localhost for display if binding to 0.0.0.0
        display_host = 'localhost' if mcp.settings.host == '0.0.0.0' else mcp.settings.host
        logger.info(
            f'Running MCP server with streamable HTTP transport on {mcp.settings.host}:{mcp.settings.port}'
        )
        logger.info('=' * 60)
        logger.info('MCP Server Access Information:')
        logger.info(f'  Base URL: http://{display_host}:{mcp.settings.port}/')
        logger.info(f'  MCP Endpoint: http://{display_host}:{mcp.settings.port}/mcp/')
        logger.info('  Transport: HTTP (streamable)')

        # Show FalkorDB Browser UI access if enabled
        if os.environ.get('BROWSER', '1') == '1':
            logger.info(f'  FalkorDB Browser UI: http://{display_host}:3000/')

        logger.info('=' * 60)
        logger.info('For MCP clients, connect to the /mcp/ endpoint above')

        # Configure uvicorn logging to match our format
        configure_uvicorn_logging()

        await mcp.run_streamable_http_async()
    else:
        raise ValueError(
            f'Unsupported transport: {mcp_config.transport}. Use "sse", "stdio", or "http"'
        )


def main():
    """Main function to run the Graphiti MCP server."""
    try:
        # Run everything in a single event loop
        asyncio.run(run_mcp_server())
    except KeyboardInterrupt:
        logger.info('Server shutting down...')
    except Exception as e:
        logger.error(f'Error initializing Graphiti MCP server: {str(e)}')
        raise


if __name__ == '__main__':
    main()
