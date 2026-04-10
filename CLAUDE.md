# Exploring Hybrid Search with LLM Query Writers

Hypothesis: BM25 and Vector Search benefit from different query formulations. BM25 rewards lexical overlap and specific term matching, while dense retrieval rewards semantic similarity. Optimal query design is therefore likely to diverge.

Evaluation Methodology: BRIGHT subsets (Biology, Earth Science, Economics, Psychology, Robotics). These have 103, 116, 103, 101, and 101 queries in them, respectively. ReasonIR subsets are used for training / optimization experiments. Metrics: Recall @ 1, 5, and 20, nDCG @ 10.

ReasonIR data cleaning: Currently using a subset of queries where BM25 can solve them with k = 30. Note: this is to be explored, we are going to start by collecting the 0-shot / unoptimized baselines before continuing with this.

Solution parameters: Embedding Model: Snowflake Arctic 2.0 Dense Single-Vector Text Embeddings. RSF alpha: 0.7, not sweeping. Reranker: Voyage `rerank-2.5`.

### Stage 0: Retrieval `k` Exploration

How does retrieval depth affect recall and downstream ranking quality? This informs the choice of k for all subsequent experiments. Note: with multi-query conditions in later stages, more queries x k compoudns the candidate pool, so the choice of k interacts with the single- vs. multi-query question.

**Retrieval Only (no reranker)**
 
| Retrieval Method | k | R@1 | R@5 | R@20 | nDCG@10 |
|------------------|---|-----|-----|------|---------|
| BM25 | 20 | | | | |
| BM25 | 50 | | | | |
| BM25 | 100 | | | | |
| Vector | 20 | | | | |
| Vector | 50 | | | | |
| Vector | 100 | | | | |
| Hybrid RSF | 20 | | | | |
| Hybrid RSF | 50 | | | | |
| Hybrid RSF | 100 | | | | |

Baseline: Vanilla Hybrid Search with Relative Score Fusion (RSF), with a single query, reranked with a Cross Encoder.

**Retrieval + Cross Encoder Reranker**
 
| Retrieval Method | k | R@1 | R@5 | R@20 | nDCG@10 |
|------------------|---|-----|-----|------|---------|
| BM25 | 20 | | | | |
| BM25 | 50 | | | | |
| BM25 | 100 | | | | |
| Vector | 20 | | | | |
| Vector | 50 | | | | |
| Vector | 100 | | | | |
| Hybrid RSF | 20 | | | | |
| Hybrid RSF | 50 | | | | |
| Hybrid RSF | 100 | | | | |

Anoter Baseline: An LLM query writer outputs a single query for Hybrid Search, which is then reranked with a Cross Encoder.

If we let each retrieval pathway have its own optimized input, does the fusion result improve?

0-Shot Query for BM25 and Vector Search: Write two separate queries with a single inference (with two outputs, `bm25_query`, `vector_search_query`). Then pool the results and rerank them with the reranker.

Optimized Query Writers:

A: Optimize a single hybrid search query
B: Optimize a single inference that produces two queries
C: Optimize two separate inferences that produces two queries

What if this is a list of queries, e.g. `bm25_queries`, `vector_search_queries`?

0-Shot Baselines:

A: The Hybrid Search query writer writes a list of hybrid search queries.
B: The single inference produces two lists of queries
C: Two separate inferences produce two lists of queries

--

Optimized Query Writers:

A: Optimize the Hybrid Search query writer writing a list of hybrid search queries
B: Optimize the single inference that producs two lists of queries
C: Optimize the two separate inferences producing two lists of queries