import asyncio
from typing import Optional, List, Literal

import dspy
import weaviate

from retrieve_dspy.database.weaviate_database import (
    weaviate_search_tool
)
from retrieve_dspy.retrievers.base_rag import BaseRAG
from retrieve_dspy.models import DSPyAgentRAGResponse, ObjectFromDB, RerankerClient, MultiLMConfig
from retrieve_dspy.signatures import (
    VerboseRelevanceRanker,
    RelevanceRanker,
    VerboseSummarizeSearchRelevance,
    SummarizeSearchRelevance,
)
from retrieve_dspy.retrievers.common.call_ce_ranker import (
    RerankItem,
    ce_rank,
    reorder,
)

RerankProvider = Literal["voyage", "hybrid"]

class LayeredListwiseReranker(BaseRAG):
    def __init__(
        self, 
        weaviate_client: weaviate.WeaviateClient,
        reranker_clients: List[RerankerClient],
        collection_name: str, 
        target_property_name: str, 
        return_property_name: str,
        verbose: bool = False,
        verbose_signature: bool = True,
        search_only: bool = True,
        retrieved_k: int = 50,
        reranked_N: int = 20,
        reranked_M: int = 5,
        reranker_provider: Optional[RerankProvider] = None,
        cohere_model: Optional[str] = "rerank-v3.5",
        voyage_model: str = "rerank-2.5",
        multi_lm_configs: Optional[List[MultiLMConfig]] = None,
    ):
        super().__init__(
            weaviate_client=weaviate_client,
            collection_name=collection_name,
            target_property_name=target_property_name,
            verbose=verbose,
            search_only=search_only,
            retrieved_k=retrieved_k,
            verbose_signature=verbose_signature,
            multi_lm_configs=multi_lm_configs,
        )
        self.return_property_name = return_property_name
        self.reranker_clients = reranker_clients
        self.reranked_N = reranked_N
        self.reranked_M = reranked_M
        self.voyage_model = voyage_model
        self.reranker_provider = reranker_provider
        self.cohere_model = cohere_model
        # Initialize Listwise Reranker
        if self.verbose_signature:
            self.listwise_reranker = dspy.ChainOfThought(VerboseRelevanceRanker)
        else:
            self.listwise_reranker = dspy.Predict(RelevanceRanker)
        
        if self.verbose_signature:
            self.summarizer = dspy.ChainOfThought(VerboseSummarizeSearchRelevance)
        else:
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
            
        # Debug the initial search results
        print("DEBUG: First 10 source object_ids from initial search:")
        for i, source in enumerate(sources[:10]):
            print(f"  Source {i}: {source.object_id}")
            
        print("DEBUG: Unique object_ids in sources:")
        unique_ids = set(s.object_id for s in sources)
        print(f"  Total sources: {len(sources)}, Unique IDs: {len(unique_ids)}")
        print(f"  Unique IDs: {list(unique_ids)[:10]}") 
        
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
            verbose=self.verbose,
        )

        print("DEBUG: ce_rank results:")
        for i, item in enumerate(reranked_results[:5]):
            print(f"  Item {i}: index={item.index}, score={getattr(item, 'score', 'N/A')}")
            
        print("DEBUG: Original sources object_ids:")
        for i, source in enumerate(sources[:10]):
            print(f"  Source {i}: {source.object_id}")
        
        # Reorder sources based on Cohere's reranking
        reordered_results: list[ObjectFromDB] = reorder(reranked_results, sources)

        if self.verbose:
            print(f"\033[93mCross encoder reranking: {len(reordered_results)} documents\033[0m")
            print("DEBUG: First 5 reranked object_ids:", [r.object_id for r in reordered_results[:5]])
            print("DEBUG: First 5 reranked relevance_ranks:", [r.relevance_rank for r in reordered_results[:5]])
        
        objects_with_summarized_content: List[ObjectFromDB] = []

        for result in reordered_results[:self.reranked_M]:
            if self.multi_lm_configs:
                with dspy.context(lm=self.multi_lm_configs_dict["summarizer"]):
                    summary = self.summarizer(
                            query=question,
                            passage=result.content,
                        ).relevance_summary
            else:
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
            print("Here is a sample:")
            print(f"{objects_with_summarized_content[0].content[:100]}...")
            print(f"{objects_with_summarized_content[0].object_id}")

        valid_object_ids = [obj.object_id for obj in objects_with_summarized_content]
        print("DEBUG: objects_with_summarized_content object_ids:", [obj.object_id for obj in objects_with_summarized_content])
        print("DEBUG: valid_object_ids:", valid_object_ids)

        if self.multi_lm_configs:
            with dspy.context(lm=self.multi_lm_configs_dict["listwise_reranker"]):
                listwise_reranked_pred = self.listwise_reranker(
                    query=question,
                    search_results=objects_with_summarized_content,
                    top_k=self.reranked_M,
                    valid_object_ids=valid_object_ids
                )
        else:
            listwise_reranked_pred = self.listwise_reranker(
                query=question,
                search_results=objects_with_summarized_content,
                top_k=self.reranked_M,
                valid_object_ids=valid_object_ids
            )

        listwise_reranked_result = listwise_reranked_pred.reranked_ids
        # listwise_reranked_result is now a list of IDs in ranked order
        if self.verbose:
            print(f"\033[96mListwise reranked result: {listwise_reranked_result}\033[0m")
            if self.verbose_signature:
                rationale = listwise_reranked_pred.reasoning
                print(f"\033[96mListwise reranked result rationale: {rationale}\033[0m")

        
        # Reorder reranked_results based on the listwise ranking
        # Create a mapping from object_id to the original object
        id_to_obj = {obj.object_id: obj for obj in reordered_results}
        
        # Reorder according to listwise_reranked_result
        reordered_results = []
        for ranked_id in listwise_reranked_result:
            if ranked_id in id_to_obj:
                reordered_results.append(id_to_obj[ranked_id])
                if self.verbose:
                    print(f"\033[38;5;208mAdding object {ranked_id} to reordered results\033[0m")
        
        # Add any remaining objects that weren't in the listwise ranking
        remaining_ids = set(id_to_obj.keys()) - set(listwise_reranked_result)
        for remaining_id in remaining_ids:
            reordered_results.append(id_to_obj[remaining_id])
            if self.verbose:
                print(f"\033[38;5;208mAdding remaining object {remaining_id} to reordered results\033[0m")
        
        reranked_results = reordered_results

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
    from retrieve_dspy.utils import get_lm

    gpt5 = get_lm("openai/gpt-5", max_tokens=32000)
    gpt4_1_mini = get_lm("openai/gpt-4.1-mini", max_tokens=32000)

    rag_pipeline = LayeredListwiseReranker(
        weaviate_client=get_weaviate_client(),
        reranker_clients=[get_voyage_client()],
        collection_name="BeirNq",
        target_property_name="content",
        return_property_name="content",
        retrieved_k=50,
        reranked_N=20,
        reranked_M=5,
        voyage_model="rerank-2.5",
        reranker_provider="voyage",
        verbose=True,
        verbose_signature=True,
        multi_lm_configs=[MultiLMConfig(signature_name="listwise_reranker", lm=gpt5), MultiLMConfig(signature_name="summarizer", lm=gpt4_1_mini)]
    )
    print("Testing sync with Listwise strategy forward")
    test_query = "How many types of MIDI messages are there?"
    response = rag_pipeline.forward(test_query)    
    print(response)
    #print("Testing async forward")
    #response = await rag_pipeline.aforward("What is the best way to learn Angular?")
    #print(response)

if __name__ == "__main__":
    asyncio.run(main())