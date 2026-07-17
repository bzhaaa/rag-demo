from typing import Any

from app.rag.contracts.protocols import RAGModelGateway


class LegacyGatewayAdapter:
    """Preserves the old all-in-one gateway injection contract."""

    def __init__(self, gateway: Any) -> None:
        self.gateway = gateway

    def __getattr__(self, name: str) -> Any:
        return getattr(self.gateway, name)


def adapt_gateway(gateway: Any) -> RAGModelGateway:
    return LegacyGatewayAdapter(gateway)
