# ==============================
# WIP!
# ==============================

import asyncio
import os
from typing import Optional, Dict, Literal

import dspy
import voyageai
import weaviate

from retrieve_dspy.retrievers.base_rag import BaseRAG
from retrieve_dspy.database.weaviate_database import weaviate_search_tool
from retrieve_dspy.signatures import (
    WriteFollowUpQuery,
    VerboseSummarizeSearchResults,
    SummarizeSearchResults,
)
from retrieve_dspy.models import ObjectFromDB, RerankerClient, DSPyAgentRAGResponse

from retrieve_dspy.retrievers.common.deduplicate import deduplicate_and_join
from retrieve_dspy.retrievers.common.call_ce_ranker import (
    RerankItem,
    ce_rank,
    reorder,
)

RerankProvider = Literal["cohere", "voyage", "hybrid"]

class SimplifiedBaleen(BaseRAG):
    def __init__(
        self,
        weaviate_client: weaviate.WeaviateClient,
        reranker_clients: list[RerankerClient],
        collection_name: str,
        target_property_name: str,
        verbose: bool = False,
        verbose_signature: bool = True,
        search_only: bool = True,
        retrieved_k: int = 5,
        reranked_N: int = 20,
        max_hops: int = 2,
        reranker_provider: Optional[RerankProvider] = None,
        cohere_model: Optional[str] = "rerank-v3.5",
        voyage_model: Optional[str] = "rerank-2.5",
        rrf_k: Optional[int] = 60,
        hybrid_weights: Optional[Dict[str, float]] = None,
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

        self.reranker_clients = reranker_clients
        self.max_hops = max_hops
        self.reranked_N = reranked_N
        self.reranker_provider = reranker_provider
        self.cohere_model = cohere_model
        self.voyage_model = voyage_model
        self.rrf_k = rrf_k
        self.hybrid_weights = hybrid_weights
        self.verbose = verbose
        if self.verbose_signature:
            self.query_writer = dspy.ChainOfThought(WriteFollowUpQuery)
            self.search_results_summarizer = dspy.ChainOfThought(VerboseSummarizeSearchResults)
        else:
            self.query_writer = dspy.Predict(WriteFollowUpQuery)
            self.search_results_summarizer = dspy.Predict(SummarizeSearchResults)

    def forward(self, question: str) -> list[ObjectFromDB]:
        results: list[ObjectFromDB] = []
        # init by searching with the original query
        # summary_of_results_found_so_far = ""

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
                results = deduplicate_and_join(results, passages)
                '''
                summary_of_results_found_so_far = self.search_results_summarizer(
                    question=question, 
                    search_results=results
                ).summary_of_search_results
                '''
                if self.verbose:
                    print(f"\033[92mHop {hop + 1}:\nQuery '{query_writer_pred.follow_up_query}'\nreturned {len(passages)} sources\033[0m")

        # Add Cross Encoder Ranking to the end (or RRF)

        documents = [s.content for s in results]
        reranked_results: list[RerankItem] = ce_rank(
            query=question,
            documents=documents,
            top_k=self.reranked_N,
            clients=self.reranker_clients,
            provider=self.reranker_provider,
            cohere_model=self.cohere_model,
            voyage_model=self.voyage_model,
            rrf_k=self.rrf_k,
            verbose=self.verbose,
        )
        results: list[ObjectFromDB] = reorder(reranked_results, results)
        if self.verbose:
            print(f"\033[93mReranked: Returning {len(results)} documents\033[0m")
        return DSPyAgentRAGResponse(
            final_answer="",
            sources=results,
            searches=[question],
            aggregations=None,
            usage={},
        )

async def main():
    weaviate_client = weaviate.connect_to_weaviate_cloud(
        cluster_url=os.getenv("WEAVIATE_URL"),
        auth_credentials=weaviate.auth.AuthApiKey(os.getenv("WEAVIATE_API_KEY"))
    )
    voyage_client = voyageai.Client(api_key=os.getenv("VOYAGE_API_KEY"))
    retriever = SimplifiedBaleen(
        weaviate_client=weaviate_client,
        collection_name="EnronEmails",
        target_property_name="email_body",
        retrieved_k=5,
        max_hops=2,
        verbose=True,
        verbose_signature=True,
        reranker_clients=[RerankerClient(name="voyage", client=voyage_client)],
        reranked_N=20,
        reranker_provider="voyage",
        voyage_model="rerank-2.5",
    )
    results = retriever.forward(question="What are the implications of SBX12?")
    print(results)

if __name__ == "__main__":
    asyncio.run(main())