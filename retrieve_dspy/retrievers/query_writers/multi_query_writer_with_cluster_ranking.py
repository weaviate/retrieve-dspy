################################################################################
# WIP!
################################################################################

import asyncio
from typing import Optional, List, Tuple

import dspy
import numpy as np
import hdbscan
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

from retrieve_dspy.tools.weaviate_database import (
    weaviate_search_tool,
    async_weaviate_search_tool
)
from retrieve_dspy.retrievers.base_rag import BaseRAG
from retrieve_dspy.models import (
    Cluster,
    DSPyAgentRAGResponse,
    Source,
    SourceWithContentAndVector
)
from retrieve_dspy.signatures import WriteSearchQueries

class MultiQueryWriterWithClusterRanking(BaseRAG):
    def __init__(
        self,
        collection_name: str,
        target_property_name: Optional[str] = "content",
        verbose: Optional[bool] = True,
        search_only: Optional[bool] = True,
        retrieved_k: Optional[int] = 10,
        search_with_queries_concatenated: Optional[bool] = False
    ):
        super().__init__(
            collection_name=collection_name,
            target_property_name=target_property_name,
            search_only=search_only,
            retrieved_k=retrieved_k,
            verbose=verbose
        )
        self.search_with_queries_concatenated = search_with_queries_concatenated
        self.query_writer = dspy.Predict(WriteSearchQueries)

    def _plot_and_save_clusters(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        filename: str = "cluster_plot.png"
    ):
        """
        Generates and saves a 2D scatter plot of document clusters.
        """
        unique_labels = set(labels)
        cluster_labels = [label for label in unique_labels if label != -1]
        
        # Guard against case with no actual clusters
        if not cluster_labels:
            colors = []
        else:
            colors = plt.cm.viridis(np.linspace(0, 1, len(cluster_labels)))

        color_map = {label: color for label, color in zip(cluster_labels, colors)}
        color_map[-1] = 'lightgray'

        plt.figure(figsize=(14, 10))

        for label in unique_labels:
            mask = (labels == label)
            color = color_map[label]
            
            if label == -1:
                plt.scatter(
                    embeddings[mask, 0], embeddings[mask, 1],
                    c=[color], s=30, alpha=0.6, label='Noise'
                )
            else:
                plt.scatter(
                    embeddings[mask, 0], embeddings[mask, 1],
                    c=[color], s=80, alpha=0.9, label=f'Cluster {label}',
                    edgecolors='k', linewidth=0.5
                )

        plt.title('2D Visualization of Document Clusters (t-SNE)', fontsize=16)
        plt.xlabel('t-SNE Dimension 1', fontsize=12)
        plt.ylabel('t-SNE Dimension 2', fontsize=12)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)

        plt.savefig(filename)
        plt.close()
        if self.verbose:
            print(f"\033[92m✅ Cluster plot saved to '{filename}'\033[0m")

    def _cluster_results(self, results: List[SourceWithContentAndVector]) -> Tuple[List[Cluster], np.ndarray, np.ndarray]:
        """
        Clusters results and populates the Cluster model with doc_ids and vectors.
        """
        if not results:
            return [], np.array([]), np.array([])

        vectors = np.array([result.vector for result in results])

        if len(vectors) <= 2:
            if self.verbose:
                print("\033[93mNot enough documents to perform clustering.\033[0m")
            return [], np.array([]), np.array([])

        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(vectors) - 1))
        embeddings = tsne.fit_transform(vectors)

        clusterer = hdbscan.HDBSCAN(min_cluster_size=5, metric='euclidean')
        clusterer.fit(embeddings)
        labels = clusterer.labels_

        cluster_map: dict[int, list[tuple[str, list[float]]]] = {}
        for i, label in enumerate(labels):
            if label not in cluster_map:
                cluster_map[label] = []
            cluster_map[label].append((results[i].object_id, results[i].vector))

        clusters = [
            Cluster(
                cluster_name=f"Cluster {label}",
                doc_ids=[item[0] for item in items],
                vectors=[item[1] for item in items]
            )
            for label, items in cluster_map.items() if label != -1
        ]

        if self.verbose:
            print(f"\033[94mFound {len(clusters)} clusters from {len(results)} documents.\033[0m")

        return clusters, embeddings, labels

    def _rank_and_sample_from_clusters(
        self,
        clusters: List[Cluster],
        all_sources: List[SourceWithContentAndVector],
        k: int
    ) -> List[Source]:
        """
        Performs diversity re-ranking using a round-robin approach on clusters.
        """
        if not clusters:
            return all_sources[:k]

        clusters.sort(key=lambda c: len(c.doc_ids), reverse=True)
        source_map = {src.object_id: src for src in all_sources}

        # Create a copy of doc_ids to avoid modifying original cluster objects
        doc_ids_per_cluster = [list(c.doc_ids) for c in clusters]
        
        ranked_source_ids = []
        seen_ids = set()

        while len(ranked_source_ids) < k and any(doc_ids_per_cluster):
            for id_list in doc_ids_per_cluster:
                if len(ranked_source_ids) >= k:
                    break
                if id_list:
                    doc_id = id_list.pop(0)
                    if doc_id not in seen_ids:
                        ranked_source_ids.append(doc_id)
                        seen_ids.add(doc_id)

        final_sources = [source_map[id] for id in ranked_source_ids]
        return final_sources

    def forward(self, question: str) -> DSPyAgentRAGResponse:
        qw_pred = self.query_writer(question=question)
        queries: List[str] = qw_pred.search_queries

        if self.verbose:
            print(f"\033[92mOriginal question:\n{question}\033[0m")
            print(f"\033[95mWrote {len(queries)} queries:\n{queries}\033[0m")

        sources_with_vectors: List[SourceWithContentAndVector] = []
        if self.search_with_queries_concatenated:
            pass 
        else:
            deduplicated_sources: dict[str, SourceWithContentAndVector] = {}
            for q in queries:
                retrieved_docs, _ = weaviate_search_tool(
                    query=q,
                    collection_name=self.collection_name,
                    target_property_name=self.target_property_name,
                    retrieved_k=self.retrieved_k,
                    return_vector=True,
                    return_format="vectors"
                )
                for doc in retrieved_docs:
                    deduplicated_sources[doc.object_id] = doc
            sources_with_vectors = list(deduplicated_sources.values())
        
        clusters, embeddings, labels = self._cluster_results(sources_with_vectors)

        if embeddings.size > 0:
            self._plot_and_save_clusters(embeddings, labels, filename="sync_cluster_plot.png")

        final_sources = self._rank_and_sample_from_clusters(
            clusters=clusters,
            all_sources=sources_with_vectors,
            k=self.retrieved_k
        )

        if self.verbose:
            print(f"\033[96m Returning {len(final_sources)} diverse sources!\033[0m")

        return DSPyAgentRAGResponse(
            final_answer="",
            sources=final_sources,
            searches=queries,
            aggregations=None,
            usage=self._merge_usage(qw_pred.get_lm_usage() or {}),
        )

    async def aforward(self, question: str) -> DSPyAgentRAGResponse:
        qw_pred = await self.query_writer.acall(question=question)
        queries: List[str] = qw_pred.search_queries

        if self.verbose:
            print(f"\033[92mOriginal question:\n{question}\033[0m")
            print(f"\033[95mWrote {len(queries)} queries:\n{queries}\033[0m")

        sources_with_vectors: List[SourceWithContentAndVector] = []
        if self.search_with_queries_concatenated:
            # Not implementing the full logic here as it's not the primary path
            pass
        else:
            search_tasks = [
                async_weaviate_search_tool(
                    query=q,
                    collection_name=self.collection_name,
                    target_property_name=self.target_property_name,
                    retrieved_k=self.retrieved_k,
                    return_vector=True,
                    return_format="vectors"
                )
                for q in queries
            ]
            search_results_tuples = await asyncio.gather(*search_tasks)

            deduplicated_sources: dict[str, SourceWithContentAndVector] = {}
            for retrieved_docs, _ in search_results_tuples:
                for doc in retrieved_docs:
                    deduplicated_sources[doc.object_id] = doc
            sources_with_vectors = list(deduplicated_sources.values())

        clusters, embeddings, labels = self._cluster_results(sources_with_vectors)
        
        if embeddings.size > 0:
            self._plot_and_save_clusters(embeddings, labels, filename="async_cluster_plot.png")

        final_sources = self._rank_and_sample_from_clusters(
            clusters=clusters,
            all_sources=sources_with_vectors,
            k=self.retrieved_k
        )

        if self.verbose:
            print(f"\033[96m Returning {len(final_sources)} diverse sources!\033[0m")

        return DSPyAgentRAGResponse(
            final_answer="",
            sources=final_sources,
            searches=queries,
            aggregations=None,
            usage=self._merge_usage(qw_pred.get_lm_usage() or {}),
        )

async def main():
    test_pipeline = MultiQueryWriterWithClusterRanking(
        collection_name="FreshstackLangchain",
        target_property_name="docs_text",
        retrieved_k=50,
        verbose=True
    )
    test_question = """
from langchain.schema import BaseMemory
class ChatMemory(BaseMemory):
   def __init__(self, user_id: UUID, type: str):
    self.user_id = user_id
    self.type = type
   # implemented abstract methods
class AnotherMem(ChatMemory):
    def __init__(self, user_id: UUID, type: str):
        super().__init__(user_id, type)
This seems simple enough - but I get an error: ValueError: "AnotherMem" object has no field "user_id". What am I doing wrong?
Note that BaseMemory is an interface.
    """
    print("--- Testing sync forward() ---")
    response = test_pipeline.forward(test_question)
    print(response)
    #print("\n--- Testing async aforward() ---")
    #async_response = await test_pipeline.aforward(test_question)
    #print(async_response)

if __name__ == "__main__":
    asyncio.run(main())