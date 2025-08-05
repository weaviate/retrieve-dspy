from typing import Optional

from retrieve_dspy.retrievers.base_rag import BaseRAG
from retrieve_dspy.models import DSPyAgentRAGResponse

class LoopingQueryWriter(BaseRAG):
    def __init__(
        self,
        collection_name: str,
        target_property_name: Optional[str] = "content",
        verbose: Optional[bool] = False,
        search_only: Optional[bool] = True,
        retrieved_k: Optional[int] = 20
    ):
        super().__init__(collection_name, target_property_name, search_only=search_only, verbose=verbose, retrieved_k=retrieved_k)

    def forward(self, question: str) -> DSPyAgentRAGResponse:
        pass

    async def aforward(self, question: str) -> DSPyAgentRAGResponse:
        pass