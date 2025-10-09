import asyncio
from typing import Optional

import dspy

from retrieve_dspy.database.weaviate_database import (
    weaviate_search_tool,
    async_weaviate_search_tool
)

from retrieve_dspy.retrievers.base_rag import BaseRAG

from retrieve_dspy.models import DSPyAgentRAGResponse
from retrieve_dspy.signatures import HyDE, VerboseHyDE, LameR

class HyDE(BaseRAG):