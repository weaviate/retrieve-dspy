import dspy

from retrieve_dspy.models import SearchResult, SearchQueryWithFilter

# Rerankers

class RerankResults(dspy.Signature):
    """Rerank passages based on their relevance to the query using listwise comparison.
    
    Your task is to analyze ALL passages simultaneously and produce a single ranked list 
    of the most relevant passages for answering the query.
    
    Instructions:
    1. Read the query carefully and understand the information need
    2. Evaluate each passage for:
       - Direct relevance to answering the query
       - Factual accuracy and completeness
       - Information quality and clarity
    3. Compare passages against each other (not just individually)
    4. Return EXACTLY `top_k` passage IDs in descending order of relevance
    
    CRITICAL: You must return exactly `top_k` IDs - no more, no less.
    """
    
    query: str = dspy.InputField(
        desc="The user's question or information need"
    )
    search_results: list[SearchResult] = dspy.InputField(
        desc="List of passages to rerank. Each contains: id, text, initial_rank, and hybrid_score"
    )
    top_k: int = dspy.InputField(
        desc="Exact number of passage IDs to return (strict requirement)"
    )
    reranked_ids: list[int] = dspy.OutputField(
        desc="List of exactly `top_k` passage IDs ordered by relevance (most relevant first). Must match IDs from search_results."
    )

class DiversityRanker(dspy.Signature):
    """Select a diverse set of relevant passages that cover different aspects of the query.
    
    Your task is to analyze ALL passages simultaneously and select a subset that:
    1. Covers different relevant topics/aspects related to the query
    2. Avoids redundant/duplicate information about the same topic
    3. Excludes passages about irrelevant topics
    
    Instructions:
    1. Read the query carefully and understand the key topics/aspects needed
    2. Group passages by the topics they cover
    3. For each relevant topic:
       - Keep the highest quality passage
       - Remove redundant passages about that same topic
    4. Exclude passages about topics not relevant to the query
    5. Return EXACTLY `top_k` passage IDs representing diverse relevant topics
    
    CRITICAL: You must return exactly `top_k` IDs - no more, no less.
    """
    
    query: str = dspy.InputField(
        desc="The user's question or information need"
    )
    search_results: list[SearchResult] = dspy.InputField(
        desc="List of passages to analyze. Each contains: id, text, initial_rank, and hybrid_score"
    )
    top_k: int = dspy.InputField(
        desc="Exact number of passage IDs to return (strict requirement)"
    )
    reranked_ids: list[int] = dspy.OutputField(
        desc="List of exactly `top_k` passage IDs representing diverse relevant topics. Must match IDs from search_results."
    )

class RerankWithSummaries(dspy.Signature):
    """Rerank passages based on their relevance summaries.
    
    You are provided with relevance summaries and scores for each passage.
    Use these summaries to make a final ranking decision.
    
    IMPORTANT: You must return ONLY THE `top_k` MOST RELEVANT passage IDs.
    
    Consider:
    - The quality and directness of information in each summary
    - The relevance scores as initial guidance
    - How well each passage would satisfy the user's query
    - Prioritize passages that provide complete, actionable answers
    
    Remember: Return EXACTLY `top_k` passage IDs, ranked from most to least relevant.
    """
    
    query: str = dspy.InputField()
    passage_summaries: list[dict] = dspy.InputField(
        desc="List of dicts with keys: passage_id, relevance_summary, relevance_score"
    )
    top_k: int = dspy.InputField(
        desc="Number of passages to return in the reranked list"
    )
    reranked_ids: list[int] = dspy.OutputField(
        desc="EXACTLY `top_k` passage IDs ordered from most to least relevant"
    )

# Query Writers

class ExpandQuery(dspy.Signature):
    """Expand a query to gather information from a search engine that will help answer the question."""

    question: str = dspy.InputField()
    expanded_query: str = dspy.OutputField()

class ExpandQueryWithHint(dspy.Signature):
    """Expand a query to gather information from a search engine that will help answer the question.
    
    Use the initial search results as hints to guide your query expansion. Analyze what information 
    was retrieved and identify gaps or areas that need deeper exploration. Expand the original 
    question to capture missing aspects, alternative terminology, or related concepts that would 
    help retrieve more comprehensive and relevant information.
    
    Consider:
    - What key information might be missing from the initial results
    - Alternative ways to phrase the query that could retrieve different perspectives
    - Related technical terms or concepts that weren't captured initially
    - More specific or broader formulations that could improve retrieval quality
    """

    question: str = dspy.InputField()
    initial_search_results: str = dspy.InputField()
    expanded_query: str = dspy.OutputField()

class WriteSearchQueries(dspy.Signature):
    """Write search queries to gather information from a search engine that will help answer the question.
Consider both exploration and result diversity to capture multiple interpretations and facets of a query.

IMPORTANT!! MAKE SURE EACH QUERY IS VERY DETAILED! LONGER, MORE DETAILED QUERIES TEND TO RETURN BETTER SEARCH RESULTS!"""

    question: str = dspy.InputField()
    search_queries: list[str] = dspy.OutputField()

class DecomposeQueryWithHint(dspy.Signature):
    """Your task is to decompose a complex technical problem into atomic sub-queries that collectively cover all essential aspects needed to answer the question.
    
    You are given the initial search results from the user's original query. Analyze what information is missing or insufficiently covered, then generate sub-queries that will:
    
    1. Break down the main problem into its constituent parts (error messages, specific functions, configuration issues)
    2. Target different aspects of the solution (root causes, prerequisites, implementation steps, troubleshooting)
    3. Use varied terminology and perspectives to maximize document diversity
    4. Include both specific technical terms AND broader conceptual queries
    5. Cover edge cases, common pitfalls, and alternative approaches
    
    Each sub-query should be:
    - Concise (1-6 words when possible, per FreshStack's findings)
    - Unique and non-redundant with other sub-queries
    - Targeted at retrieving documents that would contain different "nuggets" of information
    
    Example decomposition pattern:
    - Original: "Chromadb from_documents function giving error"
    - Sub-queries: ["Chromadb from_documents signature", "EmbeddingFunction interface changes", "HuggingFaceEmbeddings alternative", "sentence-transformers compatibility", "Chroma migration 0.4.16"]
    """

    user_question: str = dspy.InputField(desc="The original technical question or problem statement")
    initial_search_results: str = dspy.InputField(desc="Initial retrieval results to identify coverage gaps")
    sub_queries: list[str] = dspy.OutputField(desc="List of 3-8 atomic sub-queries that maximize nugget coverage")

class WriteSearchQueriesWithFilters(dspy.Signature):
    """Write search queries with optional filters to gather information from a search engine that will help answer the question."""

    question: str = dspy.InputField()
    filters_available: str = dspy.InputField()
    search_queries_with_filters: list[SearchQueryWithFilter] = dspy.OutputField()

class WriteFollowUpQueries(dspy.Signature):
    """Given a user question and contexts retrieved so far from search, assess if additional search queries are needed to fully answer the question.
    
    You are part of a retrieval system that has already performed an initial search and retrieved some contexts. Your job is to:
    1. Analyze whether the current contexts provide sufficient information to answer the user's question
    2. If not, determine what specific information is still missing
    3. Generate targeted search queries that would retrieve the missing information from a search engine
    
    The follow-up queries should be optimized for search engines and designed to fill gaps in the current knowledge base."""

    question: str = dspy.InputField()
    contexts: str = dspy.InputField()
    follow_up_queries_needed: bool = dspy.OutputField()
    follow_up_queries: list[str] = dspy.OutputField()

# Summarizers

class FilterIrrelevantSearchResults(dspy.Signature):
    """Filter out search results that are not relevant to answering the question."""
    
    question: str = dspy.InputField()
    search_results: dict[int, str] = dspy.InputField(desc="The search results keyed by their id.")
    filtered_results: list[int] = dspy.OutputField(desc="The ids of relevant results.")

class SummarizeSearchResults(dspy.Signature):
    """Summarize search results to extract the most important information related to the question."""
    
    question: str = dspy.InputField()
    search_results: dict[int, str] = dspy.InputField()
    summary: str = dspy.OutputField() # add citations to the ids in the summary

class SummarizeSearchRelevance(dspy.Signature):
    """Analyze and summarize how a search result addresses the given query.
    
    Evaluate the passage's relevance by considering:
    - How directly it answers or addresses the query
    - The completeness of information provided
    - The specificity and quality of content
    - Whether it contains actionable information
    
    Provide a concise summary (2-3 sentences) explaining:
    1. What relevant information the passage contains
    2. How well it addresses the query's intent
    3. Any limitations or gaps in the information
    """
    
    query: str = dspy.InputField()
    passage: str = dspy.InputField()
    passage_id: int = dspy.InputField(desc="The ID of this passage for reference")
    
    relevance_summary: str = dspy.OutputField(
        desc="A 2-3 sentence summary of how this passage relates to the query and its relevance"
    )
    relevance_score: float = dspy.OutputField(
        desc="A relevance score from 0.0 to 1.0, where 1.0 is perfectly relevant"
    )

class QuerySummarizer(dspy.Signature):
    """Summarize a technical question into one or two sentences."""

    question: str = dspy.InputField()
    summary: str = dspy.OutputField()