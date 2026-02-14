"""Entity Type Service for managing entity types with file-based storage."""

import asyncio
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from config.schema import EntityTypeConfig, GraphitiConfig

logger = logging.getLogger(__name__)

# Protected field names that conflict with Graphiti's internal EntityNode attributes
PROTECTED_FIELD_NAMES = frozenset({
    'name', 'summary', 'uuid', 'created_at', 'group_id',
    'labels', 'attributes', 'name_embedding', 'summary_embedding',
})


class EntityTypeData:
    """Data class for entity type stored in JSON file."""

    def __init__(
        self,
        name: str,
        description: str,
        fields: list[dict[str, Any]] | None = None,
        uuid: str | None = None,
        source: str = 'user',
        created_at: str | None = None,
        modified_at: str | None = None,
    ):
        self.uuid = uuid or str(uuid4())
        self.name = name
        self.description = description
        self.fields = fields or []
        self.source = source  # 'config' or 'user'
        self.created_at = created_at or datetime.now(timezone.utc).isoformat()
        self.modified_at = modified_at

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'uuid': self.uuid,
            'name': self.name,
            'description': self.description,
            'fields': self.fields,
            'source': self.source,
            'created_at': self.created_at,
            'modified_at': self.modified_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'EntityTypeData':
        """Create from dictionary."""
        return cls(
            uuid=data.get('uuid'),
            name=data['name'],
            description=data['description'],
            fields=data.get('fields', []),
            source=data.get('source', 'user'),
            created_at=data.get('created_at'),
            modified_at=data.get('modified_at'),
        )

    @classmethod
    def from_config(cls, config: EntityTypeConfig) -> 'EntityTypeData':
        """Create from config schema."""
        fields = []
        if config.fields:
            for field in config.fields:
                fields.append({
                    'name': field.name,
                    'type': field.type,
                    'required': field.required,
                    'description': field.description,
                })
        return cls(
            name=config.name,
            description=config.description,
            fields=fields,
            source='config',
        )


class EntityTypeService:
    """Service for managing entity types with file-based storage.

    Uses a JSON file for persistence instead of Redis, making the service
    database-neutral. The file is stored in data_dir/entity_types.json.

    Trade-off: Every mutation rewrites the entire file (unlike upstream's
    per-entity Redis SET). Acceptable for the expected scale (<50 types).
    """

    def __init__(self, data_dir: Path | None = None):
        """Initialize the entity type service.

        Args:
            data_dir: Directory for storing entity_types.json. Defaults to /app/data.
        """
        self._data_dir = data_dir or Path('/app/data')
        self._config: GraphitiConfig | None = None
        self._cache: list[EntityTypeData] | None = None  # In-memory cache
        self._lock = asyncio.Lock()

    @property
    def _file_path(self) -> Path:
        """Path to the entity types JSON file."""
        return self._data_dir / 'entity_types.json'

    def _validate_fields(self, fields: list[dict[str, Any]] | None) -> None:
        """Validate field names don't conflict with protected attributes.

        Raises:
            ValueError: If any field name is protected
        """
        if not fields:
            return
        for field in fields:
            field_name = field.get('name', '')
            if field_name.lower() in PROTECTED_FIELD_NAMES:
                raise ValueError(
                    f'Field name "{field_name}" is reserved by Graphiti. '
                    f'Protected names: {", ".join(sorted(PROTECTED_FIELD_NAMES))}'
                )

    async def initialize(self, config: GraphitiConfig | None = None) -> None:
        """Initialize the service and seed from config if needed.

        Args:
            config: GraphitiConfig for seeding entity types
        """
        self._config = config

        # Ensure data directory exists
        self._data_dir.mkdir(parents=True, exist_ok=True)

        # Load existing data or initialize empty
        self._cache = self._load()

        logger.info(f'EntityTypeService initialized with data_dir: {self._data_dir}')

        # Seed from config if entity types exist and file is empty
        if config and config.graphiti.entity_types:
            await self._seed_from_config(config.graphiti.entity_types)

    def _load(self) -> list[EntityTypeData]:
        """Load entity types from JSON file."""
        if not self._file_path.exists():
            return []

        try:
            with open(self._file_path) as f:
                data = json.load(f)
            return [EntityTypeData.from_dict(et) for et in data]
        except (json.JSONDecodeError, KeyError, OSError) as e:
            logger.error(f'Error loading entity types from {self._file_path}: {e}')
            return []

    def _save(self) -> None:
        """Save entity types to JSON file (atomic via temp + rename)."""
        if self._cache is None:
            return

        data = [et.to_dict() for et in self._cache]
        tmp_path = self._file_path.with_suffix('.tmp')
        with open(tmp_path, 'w') as f:
            json.dump(data, f, indent=2)
        tmp_path.rename(self._file_path)

    async def _seed_from_config(self, config_types: list[EntityTypeConfig]) -> None:
        """Seed entity types from config, adding only new ones.

        Args:
            config_types: List of entity types from config
        """
        async with self._lock:
            if self._cache is None:
                self._cache = []

            existing_names = {et.name for et in self._cache}

            added_count = 0
            for config_type in config_types:
                if config_type.name not in existing_names:
                    entity_type = EntityTypeData.from_config(config_type)
                    self._cache.append(entity_type)
                    added_count += 1
                    logger.info(f'Seeded entity type from config: {config_type.name}')

            if added_count > 0:
                self._save()
                logger.info(f'Seeded {added_count} new entity types from config')
            else:
                logger.info('No new entity types to seed from config')

    async def get_all(self) -> list[EntityTypeData]:
        """Get all entity types.

        Returns:
            List of all entity types
        """
        if self._cache is None:
            self._cache = self._load()
        return list(self._cache)

    async def get_by_name(self, name: str) -> EntityTypeData | None:
        """Get an entity type by name.

        Args:
            name: Entity type name

        Returns:
            EntityTypeData if found, None otherwise
        """
        all_types = await self.get_all()
        for et in all_types:
            if et.name == name:
                return et
        return None

    async def create(
        self,
        name: str,
        description: str,
        fields: list[dict[str, Any]] | None = None,
    ) -> EntityTypeData:
        """Create a new entity type.

        Args:
            name: Entity type name
            description: Entity type description
            fields: Optional list of field definitions

        Returns:
            The created entity type

        Raises:
            ValueError: If entity type with this name already exists or field names are protected
        """
        async with self._lock:
            existing = await self.get_by_name(name)
            if existing:
                raise ValueError(f'Entity type "{name}" already exists')

            self._validate_fields(fields)

            entity_type = EntityTypeData(
                name=name,
                description=description,
                fields=fields or [],
                source='user',
            )

            if self._cache is None:
                self._cache = []
            self._cache.append(entity_type)
            self._save()

            logger.info(f'Created entity type: {name}')
            return entity_type

    async def update(
        self,
        name: str,
        description: str | None = None,
        fields: list[dict[str, Any]] | None = None,
    ) -> EntityTypeData:
        """Update an existing entity type.

        Args:
            name: Entity type name
            description: New description (optional)
            fields: New fields (optional)

        Returns:
            The updated entity type

        Raises:
            ValueError: If entity type not found or field names are protected
        """
        async with self._lock:
            if self._cache is None:
                self._cache = self._load()

            entity_type = await self.get_by_name(name)
            if not entity_type:
                raise ValueError(f'Entity type "{name}" not found')

            if fields is not None:
                self._validate_fields(fields)

            if description is not None:
                entity_type.description = description
            if fields is not None:
                entity_type.fields = fields

            # Mark config-sourced entities as modified
            if entity_type.source == 'config':
                entity_type.source = 'config_modified'

            entity_type.modified_at = datetime.now(timezone.utc).isoformat()

            self._save()
            logger.info(f'Updated entity type: {name}')
            return entity_type

    async def delete(self, name: str) -> bool:
        """Delete an entity type.

        Args:
            name: Entity type name

        Returns:
            True if deleted, False if not found
        """
        async with self._lock:
            if self._cache is None:
                self._cache = self._load()

            original_count = len(self._cache)
            self._cache = [et for et in self._cache if et.name != name]

            if len(self._cache) == original_count:
                return False

            self._save()
            logger.info(f'Deleted entity type: {name}')
            return True

    async def get_as_pydantic_models(self) -> dict[str, type] | None:
        """Get entity types as Pydantic models for Graphiti.

        Returns:
            Dict mapping type names to Pydantic model classes, or None if no types
        """
        from pydantic import Field, create_model

        all_types = await self.get_all()
        if not all_types:
            return None

        custom_types = {}
        for et in all_types:
            field_definitions: dict[str, Any] = {}

            for field in et.fields:
                field_name = field.get('name', '')
                field_type = field.get('type', 'str')
                field_required = field.get('required', False)
                field_desc = field.get('description', '')

                # Map type string to Python type
                python_type = str  # default
                if field_type == 'int':
                    python_type = int
                elif field_type == 'float':
                    python_type = float
                elif field_type == 'bool':
                    python_type = bool

                if field_required:
                    field_definitions[field_name] = (
                        python_type,
                        Field(description=field_desc),
                    )
                else:
                    field_definitions[field_name] = (
                        python_type | None,
                        Field(default=None, description=field_desc),
                    )

            # Create dynamic Pydantic model
            model = create_model(
                et.name,
                __doc__=et.description,
                **field_definitions,
            )
            custom_types[et.name] = model

        return custom_types if custom_types else None

    async def reset_to_defaults(self) -> int:
        """Reset entity types to config defaults.

        Deletes all entity types and re-seeds from config.

        Returns:
            Number of entity types seeded from config

        Raises:
            RuntimeError: If service not initialized or no config available
        """
        if not self._config or not self._config.graphiti.entity_types:
            raise RuntimeError('No entity types configured in config.yaml')

        async with self._lock:
            # Clear cache
            self._cache = []

            # Re-seed from config
            config_types = self._config.graphiti.entity_types
            for config_type in config_types:
                entity_type = EntityTypeData.from_config(config_type)
                self._cache.append(entity_type)

            self._save()

            count = len(config_types)
            logger.info(f'Reset entity types to {count} defaults from config')
            return count

    async def close(self) -> None:
        """Close the service (no-op for file-based storage)."""
        # Ensure any pending changes are saved
        if self._cache is not None:
            self._save()
        logger.info('EntityTypeService closed')
