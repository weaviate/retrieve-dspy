import asyncio
import os
from typing import Optional, List, Set

import cohere
import dspy
from cohere import RerankResponseResultsItem

from retrieve_dspy.tools.weaviate_database import (
    weaviate_search_tool,
    async_weaviate_search_tool
)

from retrieve_dspy.retrievers.base_rag import BaseRAG
from retrieve_dspy.models import DSPyAgentRAGResponse, Source
from retrieve_dspy.signatures import WriteSearchQueries


class MultiQueryWriterWithCrossEncoderReranker(BaseRAG):    
    def __init__(
        self, 
        collection_name: str, 
        target_property_name: Optional[str] = "content",
        verbose: Optional[bool] = False,
        search_only: Optional[bool] = True, 
        retrieved_k: Optional[int] = 10,
        reranked_k: Optional[int] = 20,
        search_with_queries_concatenated: Optional[bool] = False,
        two_stage_reranking: Optional[bool] = False,
        per_query_top_k: Optional[int] = 5,
        cohere_model: Optional[str] = "rerank-v3.5"
    ):
        """
        Initialize the MultiQuery Writer with Cross Encoder Reranker.
        
        Args:
            collection_name: Weaviate collection name
            target_property_name: Property to search in Weaviate
            verbose: Whether to print debug information
            search_only: Whether to only search without generating answers
            retrieved_k: Number of documents to retrieve per query
            reranked_k: Number of documents to keep after final reranking
            search_with_queries_concatenated: Whether to concatenate queries or search separately
            two_stage_reranking: Whether to use two-stage reranking (rerank per query first)
            per_query_top_k: Number of top documents to keep from each query in two-stage mode
            cohere_model: Cohere reranking model to use
        """
        super().__init__(
            collection_name=collection_name, 
            target_property_name=target_property_name, 
            search_only=search_only, 
            retrieved_k=retrieved_k,
            verbose=verbose
        )
        self.reranked_k = reranked_k
        self.search_with_queries_concatenated = search_with_queries_concatenated
        self.two_stage_reranking = two_stage_reranking
        self.per_query_top_k = per_query_top_k
        self.cohere_model = cohere_model
        
        self.query_writer = dspy.ChainOfThought(WriteSearchQueries)
        
        api_key = os.getenv("COHERE_API_KEY")
        if not api_key:
            raise ValueError("COHERE_API_KEY must be provided or set as environment variable")
        
        self.co = cohere.ClientV2(api_key)
    
    def _deduplicate_sources(self, sources: List[Source]) -> tuple[List[Source], List[str]]:
        """
        Remove duplicate sources based on object_id and return unique content.
        
        Returns:
            Tuple of (unique sources, corresponding document texts)
        """
        seen_ids: Set[str] = set()
        unique_sources: List[Source] = []
        unique_docs: List[str] = []
        
        for source in sources:
            if source.object_id not in seen_ids:
                seen_ids.add(source.object_id)
                unique_sources.append(source)
        
        return unique_sources, unique_docs
    
    def _rerank_with_cohere(
        self, 
        query: str, 
        documents: List[str]
    ) -> List[RerankResponseResultsItem]:
        """
        Rerank documents using Cohere's Cross Encoder.
        """
        try:
            response = self.co.rerank(
                model=self.cohere_model,
                query=query,
                documents=documents,
                top_n=min(self.reranked_k, len(documents))
            )
            return response.results
        except Exception as e:
            if self.verbose:
                print(f"\033[91mError during Cohere reranking: {e}\033[0m")
            raise
    
    async def _async_rerank_with_cohere(
        self, 
        query: str, 
        documents: List[str]
    ) -> List[RerankResponseResultsItem]:
        """
        Asynchronously rerank documents using Cohere's Cross Encoder.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, 
            self._rerank_with_cohere, 
            query, 
            documents
        )
    
    def forward(self, question: str) -> DSPyAgentRAGResponse:
        """
        Execute the multi-query retrieval and reranking pipeline.
        """
        qw_pred = self.query_writer(question=question)
        queries: list[str] = qw_pred.search_queries
        
        if self.verbose:
            print(f"\033[95mGenerated {len(queries)} search queries:\033[0m")
            for i, q in enumerate(queries, 1):
                print(f"  Query {i}: {q}")
        
        usage_buckets = [qw_pred.get_lm_usage() or {}]
        
        if self.two_stage_reranking and not self.search_with_queries_concatenated:
            all_sources: list[Source] = []
            all_documents: list[str] = []
            
            for i, query in enumerate(queries, 1):
                search_results, sources = weaviate_search_tool(
                    query=query,
                    collection_name=self.collection_name,
                    target_property_name=self.target_property_name,
                    retrieved_k=self.retrieved_k,
                    return_format="rerank"
                )
                
                if self.verbose:
                    print(f"\n\033[96mQuery {i} retrieved {len(sources)} documents\033[0m")
                
                query_documents = [result.content for result in search_results]
                
                if len(query_documents) > 0:
                    reranked_for_query = self._rerank_with_cohere(query, query_documents)
                    
                    if self.verbose:
                        print(f"  Reranked for query '{query[:50]}...'")
                        if reranked_for_query:
                            print(f"  Top score: {reranked_for_query[0].relevance_score:.4f}")
                    
                    for j, result in enumerate(reranked_for_query[:self.per_query_top_k]):
                        if 0 <= result.index < len(sources):
                            all_sources.append(sources[result.index])
                            all_documents.append(query_documents[result.index])
                            
                            if self.verbose and j < 2:
                                print(f"    Keeping rank {j+1}: score {result.relevance_score:.4f}")
            
            seen_ids = set()
            unique_sources = []
            unique_documents = []
            
            for source, doc in zip(all_sources, all_documents):
                if source.object_id not in seen_ids:
                    seen_ids.add(source.object_id)
                    unique_sources.append(source)
                    unique_documents.append(doc)
            
            if self.verbose:
                print(f"\n\033[93mCollected {len(unique_documents)} unique documents "
                      f"from top-{self.per_query_top_k} of each query\033[0m")
                print(f"Now reranking against original question: '{question}'\033[0m")
            
            final_reranked = self._rerank_with_cohere(question, unique_documents)
            
        else:
            # Original single-stage approach
            all_sources: list[Source] = []
            all_search_results = []
            
            if self.search_with_queries_concatenated:
                concatenated_query = " ".join(queries)
                search_results, sources = weaviate_search_tool(
                    query=concatenated_query,
                    collection_name=self.collection_name,
                    target_property_name=self.target_property_name,
                    retrieved_k=self.retrieved_k,
                    return_format="rerank"
                )
                all_sources.extend(sources)
                all_search_results.extend(search_results)
            else:
                for q in queries:
                    search_results, sources = weaviate_search_tool(
                        query=q,
                        collection_name=self.collection_name,
                        target_property_name=self.target_property_name,
                        retrieved_k=self.retrieved_k,
                        return_format="rerank"
                    )
                    all_sources.extend(sources)
                    all_search_results.extend(search_results)
            
            if self.verbose:
                print(f"\033[96mRetrieved {len(all_sources)} total documents "
                      f"({len(set(s.object_id for s in all_sources))} unique)\033[0m")
            
            # Deduplicate
            unique_sources, _ = self._deduplicate_sources(all_sources)
            
            # Extract content
            seen_ids = set()
            unique_search_results = []
            for result, source in zip(all_search_results, all_sources):
                if source.object_id not in seen_ids:
                    seen_ids.add(source.object_id)
                    unique_search_results.append(result)
            
            unique_documents = [result.content for result in unique_search_results]
            
            if self.verbose:
                print(f"\n\033[93mReranking {len(unique_documents)} unique documents...\033[0m")
            
            # Single rerank against original question
            final_reranked = self._rerank_with_cohere(question, unique_documents)
        
        # Process final reranking results
        if self.verbose:
            print(f"\n\033[93mFinal reranking complete. Top scores:\033[0m")
        
        reranked_sources = []
        for i, result in enumerate(final_reranked):
            if 0 <= result.index < len(unique_sources):
                reranked_sources.append(unique_sources[result.index])
                
                if self.verbose and i < 5:
                    print(f"Final Rank {i + 1}: "
                          f"Document {result.index + 1} "
                          f"(relevance: {result.relevance_score:.4f})")
        
        if self.verbose:
            print(f"\n\033[96mReturning {len(reranked_sources)} reranked documents\033[0m")
            if self.two_stage_reranking:
                print(f"(Used two-stage reranking: top-{self.per_query_top_k} per query → final rerank)")
        
        return DSPyAgentRAGResponse(
            final_answer="",
            sources=reranked_sources,
            searches=queries,
            aggregations=None,
            usage=self._merge_usage(*usage_buckets),
        )
    
    async def aforward(self, question: str) -> DSPyAgentRAGResponse:
        """
        Asynchronously execute the multi-query retrieval and reranking pipeline.
        """
        qw_pred = await self.query_writer.acall(question=question)
        queries: list[str] = qw_pred.search_queries
        
        if self.verbose:
            print(f"\033[95mGenerated {len(queries)} search queries:\033[0m")
            for i, q in enumerate(queries, 1):
                print(f"  Query {i}: {q}")
        
        usage_buckets = [qw_pred.get_lm_usage() or {}]
        
        if self.two_stage_reranking and not self.search_with_queries_concatenated:
            all_sources: list[Source] = []
            all_documents: list[str] = []
            
            # Execute all searches concurrently
            search_tasks = [
                async_weaviate_search_tool(
                    query=q,
                    collection_name=self.collection_name,
                    target_property_name=self.target_property_name,
                    retrieved_k=self.retrieved_k,
                    return_format="rerank"
                )
                for q in queries
            ]
            
            search_results_list = await asyncio.gather(*search_tasks)
            
            for i, (query, (search_results, sources)) in enumerate(zip(queries, search_results_list), 1):
                if self.verbose:
                    print(f"\n\033[96mQuery {i} retrieved {len(sources)} documents\033[0m")
                
                query_documents = [result.content for result in search_results]
                
                if len(query_documents) > 0:
                    reranked_for_query = await self._async_rerank_with_cohere(query, query_documents)
                    
                    if self.verbose:
                        print(f"  Reranked for query '{query[:50]}...'")
                        if reranked_for_query:
                            print(f"  Top score: {reranked_for_query[0].relevance_score:.4f}")
                    
                    for j, result in enumerate(reranked_for_query[:self.per_query_top_k]):
                        if 0 <= result.index < len(sources):
                            all_sources.append(sources[result.index])
                            all_documents.append(query_documents[result.index])
                            
                            if self.verbose and j < 2:  # Show top 2 per query
                                print(f"    Keeping rank {j+1}: score {result.relevance_score:.4f}")
            
            seen_ids = set()
            unique_sources = []
            unique_documents = []
            
            for source, doc in zip(all_sources, all_documents):
                if source.object_id not in seen_ids:
                    seen_ids.add(source.object_id)
                    unique_sources.append(source)
                    unique_documents.append(doc)
            
            if self.verbose:
                print(f"\n\033[93mCollected {len(unique_documents)} unique documents "
                      f"from top-{self.per_query_top_k} of each query\033[0m")
                print(f"Now reranking against original question: '{question}'\033[0m")
            
            final_reranked = await self._async_rerank_with_cohere(question, unique_documents)
            
        else:
            # Original single-stage approach
            all_sources: list[Source] = []
            all_search_results = []
            
            if self.search_with_queries_concatenated:
                concatenated_query = " ".join(queries)
                search_results, sources = await async_weaviate_search_tool(
                    query=concatenated_query,
                    collection_name=self.collection_name,
                    target_property_name=self.target_property_name,
                    retrieved_k=self.retrieved_k,
                    return_format="rerank"
                )
                all_sources.extend(sources)
                all_search_results.extend(search_results)
            else:
                search_tasks = [
                    async_weaviate_search_tool(
                        query=q,
                        collection_name=self.collection_name,
                        target_property_name=self.target_property_name,
                        retrieved_k=self.retrieved_k,
                        return_format="rerank"
                    )
                    for q in queries
                ]
                
                results = await asyncio.gather(*search_tasks)
                
                for search_results, sources in results:
                    all_sources.extend(sources)
                    all_search_results.extend(search_results)
            
            if self.verbose:
                print(f"\033[96mRetrieved {len(all_sources)} total documents "
                      f"({len(set(s.object_id for s in all_sources))} unique)\033[0m")
            
            unique_sources, _ = self._deduplicate_sources(all_sources)
            
            seen_ids = set()
            unique_search_results = []
            for result, source in zip(all_search_results, all_sources):
                if source.object_id not in seen_ids:
                    seen_ids.add(source.object_id)
                    unique_search_results.append(result)
            
            unique_documents = [result.content for result in unique_search_results]
            
            if self.verbose:
                print(f"\n\033[93mReranking {len(unique_documents)} unique documents...\033[0m")
            
            final_reranked = await self._async_rerank_with_cohere(question, unique_documents)
        
        if self.verbose:
            print(f"\n\033[93mFinal reranking complete. Top scores:\033[0m")
        
        reranked_sources = []
        for i, result in enumerate(final_reranked):
            if 0 <= result.index < len(unique_sources):
                reranked_sources.append(unique_sources[result.index])
                
                if self.verbose and i < 5:
                    print(f"Final Rank {i + 1}: "
                          f"Document {result.index + 1} "
                          f"(relevance: {result.relevance_score:.4f})")
        
        if self.verbose:
            print(f"\n\033[96mReturning {len(reranked_sources)} reranked documents\033[0m")
            if self.two_stage_reranking:
                print(f"(Used two-stage reranking: top-{self.per_query_top_k} per query → final rerank)")
        
        return DSPyAgentRAGResponse(
            final_answer="",
            sources=reranked_sources,
            searches=queries,
            aggregations=None,
            usage=self._merge_usage(*usage_buckets),
        )


async def main():
    """Test the MultiQuery Writer with Cross Encoder Reranker"""
    import dspy
    
    openai_api_key = os.getenv("OPENAI_API_KEY")
    cohere_api_key = os.getenv("COHERE_API_KEY")
    
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY environment variable is required")
    if not cohere_api_key:
        raise ValueError("COHERE_API_KEY environment variable is required")
    
    lm = dspy.LM("openai/gpt-4o-mini", api_key=openai_api_key)
    dspy.configure(lm=lm, track_usage=True)
    print(f"DSPy configured with: {lm}")
    
    test_query = "How do I integrate Weaviate and Langchain?"
    
    print("\n" + "="*80)
    print("TESTING SINGLE-STAGE RERANKING")
    print("="*80)
    
    single_stage_pipeline = MultiQueryWriterWithCrossEncoderReranker(
        collection_name="FreshstackLangchain",
        target_property_name="docs_text",
        retrieved_k=10,
        reranked_k=15,
        search_with_queries_concatenated=False,
        two_stage_reranking=False,  # Single-stage mode
        verbose=True
    )
    
    print("\n--- Single-Stage Reranking ---")
    response = single_stage_pipeline.forward(test_query)
    print(f"\nFinal result: {len(response.sources)} documents")
    print(f"Searches performed: {response.searches}")
    
    print("\n" + "="*80)
    print("TESTING TWO-STAGE RERANKING")
    print("="*80)

    two_stage_pipeline = MultiQueryWriterWithCrossEncoderReranker(
        collection_name="FreshstackLangchain",
        target_property_name="docs_text",
        retrieved_k=10,
        reranked_k=15,
        search_with_queries_concatenated=False,
        two_stage_reranking=True,
        per_query_top_k=5,
        verbose=True
    )
    
    print("\n--- Two-Stage Reranking (Async) ---")
    async_response = await two_stage_pipeline.aforward(test_query)
    print(f"\nFinal result: {len(async_response.sources)} documents")
    print(f"Searches performed: {async_response.searches}")


if __name__ == "__main__":
    asyncio.run(main())