from orion.database import MongoConnection
from orion.settings import Settings


class FakeClient:
    def __init__(self, uri: str) -> None:
        self.uri = uri
        self.closed = False
        self.databases: list[str] = []

    def __getitem__(self, name: str) -> str:
        self.databases.append(name)
        return name

    def close(self) -> None:
        self.closed = True


def test_mongo_connection_is_lazy_and_uses_configured_database() -> None:
    clients: list[FakeClient] = []

    def factory(uri: str) -> FakeClient:
        client = FakeClient(uri)
        clients.append(client)
        return client

    connection = MongoConnection(
        Settings(mongo_uri="mongodb://test", mongo_database="orion_test"),
        client_factory=factory,
    )

    assert clients == []
    assert connection.database == "orion_test"
    assert len(clients) == 1
    assert connection.database == "orion_test"
    assert len(clients) == 1

    connection.close()
    assert clients[0].closed is True
