"""MongoDB connection boundary used by the context store."""

from collections.abc import Callable
from typing import Any

from pymongo import MongoClient
from pymongo.database import Database

from .settings import Settings, get_settings

MongoClientFactory = Callable[[str], MongoClient]


class MongoConnection:
    """Lazily create and close a MongoDB client for application lifespan use."""

    def __init__(
        self,
        settings: Settings | None = None,
        client_factory: MongoClientFactory = MongoClient,
    ) -> None:
        self._settings = settings or get_settings()
        self._client_factory = client_factory
        self._client: MongoClient | None = None

    @property
    def database(self) -> Database[Any]:
        """Return the configured database, creating the client on first use."""

        if self._client is None:
            self._client = self._client_factory(self._settings.mongo_uri)
        return self._client[self._settings.mongo_database]

    def close(self) -> None:
        """Close the client if it has been created."""

        if self._client is not None:
            self._client.close()
            self._client = None
