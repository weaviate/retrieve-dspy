from retrieve_dspy.retrievers.base_rag import BaseRAG
from retrieve_dspy.models import DSPyAgentRAGResponse

class LoopingQueryWriter(BaseRAG):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, question: str) -> DSPyAgentRAGResponse:
        pass

    async def aforward(self, question: str) -> DSPyAgentRAGResponse:
        pass