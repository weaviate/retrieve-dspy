import asyncio
import os
from typing import Optional

import dspy
import weaviate

from retrieve_dspy.database.weaviate_database import (
    weaviate_search_tool,
    async_weaviate_search_tool
)
from retrieve_dspy.retrievers.base_rag import BaseRAG
from retrieve_dspy.models import DSPyAgentRAGResponse
from retrieve_dspy.signatures import ThinkQE, VerboseThinkQE

class ThinkQE_QueryExpander(BaseRAG):
    def __init__(
        self,
        collection_name: str,
        target_property_name: str,
        weaviate_client: Optional[weaviate.WeaviateClient | weaviate.WeaviateAsyncClient] = None,
        verbose: Optional[bool] = False,
        search_only: Optional[bool] = True,
        retrieved_k: Optional[int] = 5,
        num_rounds: Optional[int] = 3,
        num_samples: Optional[int] = 2,
        repetition_lambda: Optional[float] = 3.0,
    ):
        super().__init__(
            collection_name=collection_name,
            target_property_name=target_property_name,
            search_only=search_only,
            verbose=verbose,
            retrieved_k=retrieved_k
        )
        self.num_rounds = num_rounds
        self.num_samples = num_samples
        self.repetition_lambda = repetition_lambda
        self.weaviate_client = weaviate_client
        if self.verbose:
            self.ThinkQE = dspy.ChainOfThought(VerboseThinkQE)
        else:
            self.ThinkQE = dspy.ChainOfThought(ThinkQE)
        
        # Blacklist for redundancy filtering
        self.blacklist = set()
        self.previous_top_k = set()

    def _calculate_query_repetitions(self, original_query: str, expansions: list[str]) -> int:
        """Calculate how many times to repeat the original query based on expansion length."""
        expansion_length = sum(len(exp.split()) for exp in expansions)
        query_length = len(original_query.split())
        
        if query_length == 0:
            return 1
            
        n = int(expansion_length / (query_length * self.repetition_lambda))
        return max(1, n)

    def _filter_documents(self, search_results: list, previous_results: set) -> list:
        """Filter out documents that appear in blacklist or previous top-K."""
        filtered = []
        for doc in search_results:
            doc_id = getattr(doc, 'id', str(doc))
            if doc_id not in self.blacklist and doc_id not in previous_results:
                filtered.append(doc)
        return filtered

    def _update_blacklist(self, all_results: list, filtered_results: list):
        """Add filtered-out documents to blacklist."""
        filtered_ids = {getattr(doc, 'id', str(doc)) for doc in filtered_results}
        for doc in all_results:
            doc_id = getattr(doc, 'id', str(doc))
            if doc_id not in filtered_ids:
                self.blacklist.add(doc_id)

    def forward(
        self, 
        question: str, 
        weaviate_client: Optional[weaviate.WeaviateClient] = None
    ) -> DSPyAgentRAGResponse:
        if weaviate_client is None:
            if isinstance(self.weaviate_client, weaviate.WeaviateClient):
                weaviate_client = self.weaviate_client

        # Reset state for new query
        self.blacklist = set()
        self.previous_top_k = set()
        
        # Initial retrieval
        initial_results = weaviate_search_tool(
            query=question,
            collection_name=self.collection_name,
            target_property_name=self.target_property_name,
            retrieved_k=10,
            weaviate_client=weaviate_client,
        )
        
        current_query = question
        all_expansions = []
        all_searches = [question]
        
        # Iterative expansion rounds
        for round_idx in range(self.num_rounds):
            # Get documents for this round (initial or from previous query)
            if round_idx == 0:
                round_results = initial_results
            else:
                all_round_results = weaviate_search_tool(
                    query=current_query,
                    collection_name=self.collection_name,
                    target_property_name=self.target_property_name,
                    retrieved_k=10,  # Retrieve more for filtering
                    weaviate_client=weaviate_client,
                )
                
                # Apply redundancy filtering
                round_results = self._filter_documents(
                    all_round_results, 
                    self.previous_top_k
                )[:self.retrieved_k]
                
                # Update blacklist and previous top-K
                self._update_blacklist(all_round_results, round_results)
                self.previous_top_k = {getattr(doc, 'id', str(doc)) for doc in round_results}
            
            # Generate multiple expansion samples for diversity
            round_expansions = []
            for _ in range(self.num_samples):
                expansion_response = self.ThinkQE(
                    question=question,
                    possible_answering_passages=round_results
                )
                expansion = expansion_response.correct_answering_passage
                round_expansions.append(expansion)
                
                if self.verbose:
                    print(f"\033[95mRound {round_idx + 1} Expansion:\n{expansion}\033[0m\n")
            
            # Accumulate expansions
            all_expansions.extend(round_expansions)
            
            # Update query by concatenating expansions
            current_query = question + " " + " ".join(all_expansions)
            all_searches.append(current_query)
        
        # Add query repetition to reinforce original intent
        num_repetitions = self._calculate_query_repetitions(question, all_expansions)
        final_query = " ".join([question] * num_repetitions) + " " + " ".join(all_expansions)
        
        if self.verbose:
            print(f"\033[94mFinal query with {num_repetitions} repetitions\033[0m")
        
        # Final retrieval with accumulated query
        final_sources = weaviate_search_tool(
            query=final_query,
            collection_name=self.collection_name,
            target_property_name=self.target_property_name,
            retrieved_k=self.retrieved_k,
            weaviate_client=weaviate_client,
        )
        
        return DSPyAgentRAGResponse(
            final_answer="",
            sources=final_sources,
            searches=all_searches,
            aggregations=None,
            usage={},
        )
    
    async def aforward(
        self, 
        question: str, 
        weaviate_async_client: Optional[weaviate.WeaviateAsyncClient] = None
    ) -> DSPyAgentRAGResponse:
        if weaviate_async_client is None:
            if isinstance(self.weaviate_async_client, weaviate.WeaviateAsyncClient):
                weaviate_async_client = self.weaviate_async_client

        # Reset state for new query
        self.blacklist = set()
        self.previous_top_k = set()
        
        # Initial retrieval
        initial_results = await async_weaviate_search_tool(
            query=question,
            collection_name=self.collection_name,
            target_property_name=self.target_property_name,
            retrieved_k=self.retrieved_k,
            weaviate_async_client=weaviate_async_client,
        )
        
        current_query = question
        all_expansions = []
        all_searches = [question]
        
        # Iterative expansion rounds
        for round_idx in range(self.num_rounds):
            # Get documents for this round
            if round_idx == 0:
                round_results = initial_results
            else:
                all_round_results = await async_weaviate_search_tool(
                    query=current_query,
                    collection_name=self.collection_name,
                    target_property_name=self.target_property_name,
                    retrieved_k=self.retrieved_k * 3,
                    weaviate_async_client=weaviate_async_client,
                )
                
                # Apply redundancy filtering
                round_results = self._filter_documents(
                    all_round_results, 
                    self.previous_top_k
                )[:self.retrieved_k]
                
                # Update blacklist and previous top-K
                self._update_blacklist(all_round_results, round_results)
                self.previous_top_k = {getattr(doc, 'id', str(doc)) for doc in round_results}
            
            # Generate multiple expansion samples for diversity
            round_expansions = []
            for _ in range(self.num_samples):
                expansion_response = await self.ThinkQE.acall(
                    question=question,
                    possible_answering_passages=round_results,
                )
                expansion = expansion_response.correct_answering_passage
                round_expansions.append(expansion)
                
                if self.verbose:
                    print(f"\033[95mRound {round_idx + 1} Expansion:\n{expansion}\033[0m\n")
            
            # Accumulate expansions
            all_expansions.extend(round_expansions)
            
            # Update query
            current_query = question + " " + " ".join(all_expansions)
            all_searches.append(current_query)
        
        # Add query repetition
        num_repetitions = self._calculate_query_repetitions(question, all_expansions)
        final_query = " ".join([question] * num_repetitions) + " " + " ".join(all_expansions)
        
        if self.verbose:
            print(f"\033[94mFinal query with {num_repetitions} repetitions\033[0m")
        
        # Final retrieval
        final_sources = await async_weaviate_search_tool(
            query=final_query,
            collection_name=self.collection_name,
            target_property_name=self.target_property_name,
            retrieved_k=self.retrieved_k,
            weaviate_async_client=weaviate_async_client,
        )
        
        return DSPyAgentRAGResponse(
            final_answer="",
            sources=final_sources,
            searches=all_searches,
            aggregations=None,
            usage={},
        )


async def main():
    test_pipeline = ThinkQE_QueryExpander(
        collection_name="BrightBiology",
        target_property_name="content",
        verbose=True,
        retrieved_k=5,
        num_rounds=3,
        num_samples=2,
    )
    
    test_q = "How many cells are in the human body?"
    
    weaviate_client = weaviate.connect_to_weaviate_cloud(
        cluster_url=os.getenv("WEAVIATE_URL"),
        auth_credentials=weaviate.auth.AuthApiKey(os.getenv("WEAVIATE_API_KEY")),
    )
    
    weaviate_async_client = weaviate.use_async_with_weaviate_cloud(
        cluster_url=os.getenv("WEAVIATE_URL"),
        auth_credentials=weaviate.auth.AuthApiKey(os.getenv("WEAVIATE_API_KEY")),
    )
    
    await weaviate_async_client.connect()
    
    print("=== Testing Sync Forward ===")
    test_sync_response = test_pipeline.forward(test_q, weaviate_client=weaviate_client)
    print(test_sync_response)
    
    print("\n=== Testing Async Forward ===")
    test_async_response = await test_pipeline.aforward(test_q, weaviate_async_client=weaviate_async_client)
    print(test_async_response)
    
    weaviate_client.close()
    await weaviate_async_client.close()


if __name__ == "__main__":
    asyncio.run(main())