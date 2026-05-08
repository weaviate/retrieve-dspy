"""
Test script: run a retrieve-dspy retriever through
query_agent_benchmarking.run_search_eval() using the SearchAgent protocol.
"""

import os
from typing import Optional

import weaviate

from query_agent_benchmarking import ObjectID, run_search_eval
from retrieve_dspy import BaseRetriever
from retrieve_dspy.retrievers.embeddings_registry import get_embedding_headers


EMBEDDING_MODEL = "weaviate/Snowflake/snowflake-arctic-embed-l-v2.0"


class RetrieveDSPySearchAgent:
    """Wraps any retrieve-dspy retriever to satisfy the SearchAgent protocol."""

    def __init__(self, retriever, embedding_model: str = EMBEDDING_MODEL):
        self.retriever = retriever
        self.embedding_model = embedding_model
        self._async_client: Optional[weaviate.WeaviateAsyncClient] = None

    def run(self, query: str, tenant: Optional[str] = None) -> list[ObjectID]:
        response = self.retriever.forward(query)
        return [ObjectID(object_id=s.object_id) for s in response.sources]

    async def run_async(self, query: str, tenant: Optional[str] = None) -> list[ObjectID]:
        response = await self.retriever.aforward(
            query, weaviate_async_client=self._async_client
        )
        return [ObjectID(object_id=s.object_id) for s in response.sources]

    async def initialize_async(self) -> None:
        headers = get_embedding_headers(self.embedding_model)
        self._async_client = weaviate.use_async_with_weaviate_cloud(
            cluster_url=os.environ["WEAVIATE_URL"],
            auth_credentials=weaviate.auth.AuthApiKey(os.environ["WEAVIATE_API_KEY"]),
            headers=headers,
        )
        await self._async_client.connect()

    async def close_async(self) -> None:
        if self._async_client is not None:
            await self._async_client.close()


def main():
    retriever = BaseRetriever(
        collection_name="BrightBiology_Default",
        target_property_name="content",
        retrieved_k=100,
        search_type="bm25",
        verbose=True,
    )

    agent = RetrieveDSPySearchAgent(retriever)

    run_search_eval(
        search_dataset="bright/biology",
        search_agent=agent,
        agent_name="BM25",
        use_async=True,
        num_trials=1,
    )


if __name__ == "__main__":
    main()
