import asyncio
import os
import re
from typing import Optional, Any, List, Literal

import dspy
import voyageai
import weaviate

from retrieve_dspy.database.weaviate_database import (
    weaviate_search_tool
)
from retrieve_dspy.retrievers.base_rag import BaseRAG
from retrieve_dspy.models import DSPyAgentRAGResponse, ObjectFromDB, RerankerClient
from retrieve_dspy.signatures import RelevanceRanker, VerboseBestMatchRanker, SummarizeSearchRelevance
from retrieve_dspy.retrievers.common.call_ce_ranker import (
    RerankItem,
    ce_rank,
    async_ce_rank,
    reorder,
)

RerankProvider = Literal["voyage", "hybrid"]
ListwiseRerankerStrategy = Literal["BestMatch", "Relevance"]

class LayeredReranker(BaseRAG):
    def __init__(
        self, 
        weaviate_client: weaviate.WeaviateClient,
        reranker_clients: List[RerankerClient],
        collection_name: str, 
        target_property_name: str, 
        return_property_name: str,
        verbose: bool = False,
        search_only: bool = True,
        retrieved_k: int = 50,
        reranked_N: int = 20,
        reranked_M: int = 5,
        reranker_provider: Optional[RerankProvider] = None,
        cohere_model: Optional[str] = "rerank-v3.5",
        voyage_model: str = "rerank-2.5",
        rrf_k: Optional[int] = 60,
        listwise_reranker_strategy: Optional[ListwiseRerankerStrategy] = "BestMatch",
    ):
        super().__init__(
            weaviate_client=weaviate_client,
            collection_name=collection_name,
            target_property_name=target_property_name,
            verbose=verbose,
            search_only=search_only,
            retrieved_k=retrieved_k
        )
        self.return_property_name = return_property_name
        self.reranker_clients = reranker_clients
        self.reranked_N = reranked_N
        self.reranked_M = reranked_M
        self.voyage_model = voyage_model
        self.reranker_provider = reranker_provider
        self.cohere_model = cohere_model
        self.rrf_k = rrf_k
        self.verbose = verbose
        self.listwise_reranker_strategy = listwise_reranker_strategy

        # Initialize Listwise Reranker
        if self.listwise_reranker_strategy == "BestMatch":
            self.listwise_reranker = dspy.Predict(VerboseBestMatchRanker)
        else:
            self.listwise_reranker = dspy.Predict(RelevanceRanker)
        
        self.summarizer = dspy.Predict(SummarizeSearchRelevance)

    def forward(self, question: str) -> DSPyAgentRAGResponse:
        # first search with the original query
        sources = weaviate_search_tool(
            weaviate_client=self.weaviate_client,
            query=question,
            collection_name=self.collection_name,
            target_property_name=self.target_property_name,
            return_property_name=self.return_property_name,
            retrieved_k=self.retrieved_k,
        )
        
        if self.verbose:
            print(f"\033[96mInitial retrieval: {len(sources)} documents\033[0m")
        
        # Extract document content for reranking
        documents = [s.content for s in sources]
        
        # then apply the cross encoder reranker to truncate the results to N
        reranked_results: List[RerankItem] = ce_rank(
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
        
        # Reorder sources based on Cohere's reranking
        reranked_results: list[ObjectFromDB] = reorder(reranked_results, sources)
        
        if self.verbose:
            print(f"\033[93mCross encoder reranking: {len(reranked_results)} documents\033[0m")
        
        objects_with_summarized_content: List[ObjectFromDB] = []

        for result in reranked_results[:self.reranked_M]:
            summary = self.summarizer(
                query=question,
                passage=result.content,
            ).relevance_summary
            objects_with_summarized_content.append(ObjectFromDB(
                object_id=result.object_id,
                relevance_rank=result.relevance_rank,
                content=summary
            ))

        if self.verbose:
            print("\033[93mSummarized objects...\033[0m")
            print(f"Here is a sample:")
            print(f"{objects_with_summarized_content[0].content[:100]}...")
            print(f"{objects_with_summarized_content[0].object_id}")

        valid_object_ids = [obj.object_id for obj in objects_with_summarized_content]
        
        listwise_reranked_result = self.listwise_reranker(
            query=question,
            search_results=objects_with_summarized_content,
            top_k=self.reranked_M,
            valid_object_ids=valid_object_ids
        ).best_match_object_id

        print(f"\033[96mListwise reranked result: {listwise_reranked_result}\033[0m")
        
        chosen = None
        if self.listwise_reranker_strategy == "BestMatch":
            for idx, obj in enumerate(reranked_results):
                print(f"\033[38;5;208mChecking object {obj.object_id} against {listwise_reranked_result}\033[0m")
                if str(obj.object_id) == str(listwise_reranked_result):
                    chosen = reranked_results.pop(idx)
                    break
            reranked_results.insert(0, chosen)
        elif self.listwise_reranker_strategy == "Relevance":
            pass

        if self.verbose:
            print(f"\033[92mListwise reranking: Returning {self.reranked_M} documents\033[0m")
        
        return DSPyAgentRAGResponse(
            final_answer="",
            sources=reranked_results[:self.reranked_N],
            searches=[question],
            aggregations=None,
            usage={},
        )

    async def aforward(self, question: str) -> DSPyAgentRAGResponse:
        pass

async def main():
    from retrieve_dspy.clients import get_weaviate_client, get_voyage_client
    rag_pipeline = LayeredReranker(
        weaviate_client=get_weaviate_client(),
        reranker_clients=[get_voyage_client()],
        collection_name="EnronEmails",
        target_property_name="email_body",
        return_property_name="email_body",
        retrieved_k=50,
        reranked_N=20,
        reranked_M=5,
        listwise_reranker_strategy="BestMatch",
        voyage_model="rerank-2.5",
        reranker_provider="voyage",
        verbose=True
    )
    print("Testing sync forward")
    test_query = "Where will Governor Gray Davis host a party for the delegates, according to the article “Davis faces dire political consequences if power woes linger?"
    response = rag_pipeline.forward(test_query)
    #print("Testing async forward")
    #response = await rag_pipeline.aforward("What is the best way to learn Angular?")
    #print(response)

if __name__ == "__main__":
    asyncio.run(main())