import asyncio
import os
from typing import Optional

import dspy
import weaviate

from retrieve_dspy.retrievers.base_rag import BaseRAG
from retrieve_dspy.database.weaviate_database import weaviate_search_tool
from retrieve_dspy.signatures import WriteFollowUpQuery
from retrieve_dspy.models import ObjectFromDB

from retrieve_dspy.retrievers.common.deduplicate import deduplicate

class SimplifiedBaleen(BaseRAG):
    def __init__(
        self,
        weaviate_client: weaviate.WeaviateClient,
        collection_name: str,
        target_property_name: str,
        verbose: bool = False,
        verbose_signature: bool = True,
        search_only: bool = True,
        retrieved_k: int = 5,
        max_hops: int = 2,
    ):
        super().__init__(
            weaviate_client=weaviate_client,
            collection_name=collection_name,
            target_property_name=target_property_name,
            verbose=verbose,
            verbose_signature=verbose_signature,
            search_only=search_only,
            retrieved_k=retrieved_k,
        )

        self.max_hops = max_hops
        self.query_writer = dspy.Predict(WriteFollowUpQuery)

    def forward(self, question: str) -> list[ObjectFromDB]:
        results: list[ObjectFromDB] = []

        for hop in range(self.max_hops):
            query_writer_pred = self.query_writer(question=question, results_found_so_far=results)
            if query_writer_pred.follow_up_query_needed:
                passages = weaviate_search_tool(
                    query=query_writer_pred.follow_up_query, 
                    collection_name=self.collection_name, 
                    target_property_name=self.target_property_name, 
                    retrieved_k=self.retrieved_k,
                    weaviate_client=self.weaviate_client
                )
                results = deduplicate(results, passages)
                if self.verbose:
                    print(f"\033[92mHop {hop + 1}:\nQuery '{query_writer_pred.follow_up_query}'\nreturned {len(passages)} sources\033[0m")

        return results

async def main():
    weaviate_client = weaviate.connect_to_weaviate_cloud(
        cluster_url=os.getenv("WEAVIATE_URL"),
        auth_credentials=weaviate.auth.AuthApiKey(os.getenv("WEAVIATE_API_KEY"))
    )
    retriever = SimplifiedBaleen(
        weaviate_client=weaviate_client,
        collection_name="EnronEmails",
        target_property_name="email_body",
        retrieved_k=5,
        max_hops=2,
        verbose=True,
        verbose_signature=True,
    )
    results = retriever.forward(question="What are the implications of SBX12?")
    print(results)

if __name__ == "__main__":
    asyncio.run(main())