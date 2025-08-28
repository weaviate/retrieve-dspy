import dspy

from retrieve_dspy.models import SearchResult, SearchQueryWithFilter

# Rerankers

class VerboseRelevanceRanker(dspy.Signature):
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

class RelevanceRanker(dspy.Signature):
    """Rerank passages based on their relevance to the query."""
    
    query: str = dspy.InputField(desc="The user's question or information need")
    search_results: list[SearchResult] = dspy.InputField(desc="List of passages to rerank.")
    top_k: int = dspy.InputField(desc="Exact number of passage IDs to return.")
    reranked_ids: list[int] = dspy.OutputField(desc="List of top_k passage IDs ordered by relevance.")


class VerboseBestMatchRanker(dspy.Signature):
    """Identify the single most relevant passage to the query.
    
    Your task is to analyze ALL passages simultaneously and identify the one passage 
    that is most relevant for answering the query.
    
    Instructions:
    1. Read the query carefully and understand the information need
    2. Evaluate each passage for:
       - Direct relevance to answering the query
       - Factual accuracy and completeness
       - Information quality and clarity
    3. Compare passages against each other (not just individually)
    4. Return the ID of the single most relevant passage
    
    CRITICAL: You must return exactly 1 passage ID - the best match.
    """
    
    query: str = dspy.InputField(
        desc="The user's question or information need"
    )
    search_results: list[SearchResult] = dspy.InputField(
        desc="List of passages to analyze. Each contains: id, text, initial_rank, and hybrid_score"
    )
    best_match_id: int = dspy.OutputField(
        desc="The ID of the single most relevant passage. Must match an ID from search_results."
    )

class BestMatchRanker(dspy.Signature):
    """Identify the single most relevant passage to the query."""

    query: str = dspy.InputField(desc="The user's question or information need")
    search_results: list[SearchResult] = dspy.InputField(desc="List of passages to analyze.")
    best_match_id: int = dspy.OutputField(desc="The ID of the single most relevant passage.")

class IdentifyMostRelevantPassage(dspy.Signature):
    """Identify the passage that contains the answer to the query from a list of passages."""
    
    query: str = dspy.InputField(
        desc="The user's question or information need"
    )
    search_results: list[SearchResult] = dspy.InputField(
        desc="List of passages to analyze. Each contains: id, text, and inital_rank"
    )
    most_relevant_passage: int = dspy.OutputField(
        desc="The id of the passage that contains the answer to the query"
    )

class VerboseDiversityRanker(dspy.Signature):
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

class DiversityRanker(dspy.Signature):
    """Select a diverse set of relevant passages that cover different aspects of the query."""
    
    query: str = dspy.InputField(desc="The user's question or information need")
    search_results: list[SearchResult] = dspy.InputField(desc="List of passages to analyze.")
    top_k: int = dspy.InputField(desc="Exact number of passage IDs to return.")
    reranked_ids: list[int] = dspy.OutputField(desc="List of top_k passage IDs representing diverse relevant topics.")

# Query Writers

class VerboseExpandQuery(dspy.Signature):
    """Expand a query to gather more comprehensive information from a search engine.

    Your task is to rewrite the user's question into an expanded query that is optimized for a search engine. The goal is to retrieve documents that will help answer the original question.

    Instructions:
    1.  Analyze the user's question to understand the core intent.
    2.  Identify key concepts, entities, and technical terms.
    3.  Add context, clarify ambiguities, and include synonyms or related terms.
    4.  Formulate a query that is specific and detailed enough to narrow down results, but broad enough to capture relevant variations.
    5.  The expanded query should be a single, coherent question or search phrase.
    
    Example:
    - Question: "dspy signature error"
    - Expanded Query: "How to fix dspy.Signature field validation error for InputField and OutputField in DSPy framework"
    """

    question: str = dspy.InputField(desc="The original user question.")
    expanded_query: str = dspy.OutputField(desc="A detailed, expanded query for a search engine.")

class ExpandQuery(dspy.Signature):
    """Expand a query to gather information from a search engine that will help answer the question."""

    question: str = dspy.InputField()
    expanded_query: str = dspy.OutputField()

class VerboseExpandQueryWithHint(dspy.Signature):
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

class ExpandQueryWithHint(dspy.Signature):
    """Expand a query using initial search results as a hint to improve information retrieval."""

    question: str = dspy.InputField()
    initial_search_results: str = dspy.InputField()
    expanded_query: str = dspy.OutputField()

class VerboseWriteSearchQueries(dspy.Signature):
    """Write search queries to gather information from a search engine that will help answer the question.
Consider both exploration and result diversity to capture multiple interpretations and facets of a query.

IMPORTANT!! MAKE SURE EACH QUERY IS VERY DETAILED! LONGER, MORE DETAILED QUERIES TEND TO RETURN BETTER SEARCH RESULTS!"""

    question: str = dspy.InputField()
    search_queries: list[str] = dspy.OutputField()

class WriteSearchQueries(dspy.Signature):
    """Write search queries to gather information from a search engine that will help answer the question."""

    question: str = dspy.InputField()
    search_queries: list[str] = dspy.OutputField()

class VerboseDecomposeQueryWithHint(dspy.Signature):
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

class DecomposeQueryWithHint(dspy.Signature):
    """Decompose a complex query into atomic sub-queries based on initial search results."""

    user_question: str = dspy.InputField(desc="The original technical question or problem statement")
    initial_search_results: str = dspy.InputField(desc="Initial retrieval results to identify coverage gaps")
    sub_queries: list[str] = dspy.OutputField(desc="List of atomic sub-queries")

class VerboseWriteSearchQueriesWithFilters(dspy.Signature):
    """Write search queries with optional filters to gather targeted information from a search engine that will help answer the question.
    
    Your task is to generate a list of search queries based on the user's question. For each query, you can optionally apply filters to narrow down the search space.
    
    Instructions:
    1.  Analyze the question to identify key topics and entities.
    2.  Formulate several distinct search queries that explore different facets of the question.
    3.  For each query, examine the available filters and decide if any can be applied to improve the results.
    4.  Apply filters only when they directly correspond to the information in the query. For example, if the question is about "react components", a filter `library: "react"` would be appropriate.
    5.  Construct the output as a list of `SearchQueryWithFilter` objects, where each object contains the `query` string and an optional `filter` dictionary.
    
    CRITICAL: The filter keys must be from the `filters_available` list. Do not invent new filter keys.
    """

    question: str = dspy.InputField(desc="The user's question.")
    filters_available: str = dspy.InputField(desc="A string showing available filter keys, e.g., 'library, source, author'.")
    search_queries_with_filters: list[SearchQueryWithFilter] = dspy.OutputField(desc="A list of search queries, each with an optional filter applied.")

class WriteSearchQueriesWithFilters(dspy.Signature):
    """Write search queries with optional filters to gather information from a search engine that will help answer the question."""

    question: str = dspy.InputField()
    filters_available: str = dspy.InputField()
    search_queries_with_filters: list[SearchQueryWithFilter] = dspy.OutputField()

class VerboseWriteFollowUpQueries(dspy.Signature):
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

class WriteFollowUpQueries(dspy.Signature):
    """Assess if more information is needed to answer a question and generate follow-up search queries if necessary."""

    question: str = dspy.InputField()
    contexts: str = dspy.InputField()
    follow_up_queries_needed: bool = dspy.OutputField()
    follow_up_queries: list[str] = dspy.OutputField()

# Summarizers

class VerboseFilterIrrelevantSearchResults(dspy.Signature):
    """Filter out search results that are not relevant to answering the question.
    
    Your task is to act as a strict relevance filter. For each search result, you must decide if it contains information that is directly useful for answering the user's question.
    
    Instructions:
    1.  Carefully read the user's question to understand the specific information needed.
    2.  For each search result, evaluate its content against the question.
    3.  A result is RELEVANT if it:
        - Directly answers a part of the question.
        - Provides essential context or background information.
        - Discusses the key entities or concepts mentioned in the question.
    4.  A result is IRRELEVANT if it:
        - Is on a completely different topic.
        - Only mentions keywords without providing substantive information.
        - Is an ad, a forum index, or other non-informative content.
    5.  Return a list containing ONLY the IDs of the relevant search results.
    """
    
    question: str = dspy.InputField(desc="The user's question.")
    search_results: dict[int, str] = dspy.InputField(desc="The search results keyed by their id.")
    filtered_results: list[int] = dspy.OutputField(desc="A list of the IDs of the relevant results only.")

class FilterIrrelevantSearchResults(dspy.Signature):
    """Filter out search results that are not relevant to answering the question."""
    
    question: str = dspy.InputField()
    search_results: dict[int, str] = dspy.InputField(desc="The search results keyed by their id.")
    filtered_results: list[int] = dspy.OutputField(desc="The ids of relevant results.")

class VerboseSummarizeSearchResults(dspy.Signature):
    """Summarize search results to extract and synthesize the most important information related to the question.
    
    Your task is to read all the provided search results and create a single, coherent summary that directly answers the user's question.
    
    Instructions:
    1.  Thoroughly understand the user's question.
    2.  Read through all the search results to identify key pieces of information, facts, and explanations.
    3.  Synthesize the information from multiple sources. Do not just summarize each document individually.
    4.  Construct a comprehensive answer that flows logically.
    5.  CRITICAL: Wherever you use information from a search result, you MUST cite its ID using the format `[id]`. For example: "The sky is blue because of Rayleigh scattering [1]." A single sentence can have multiple citations, like `[2, 3]`.
    6.  The final summary should be a self-contained answer to the question, based only on the provided search results.
    """
    
    question: str = dspy.InputField(desc="The user's question.")
    search_results: dict[int, str] = dspy.InputField(desc="A dictionary of search results, with ID as key and text as value.")
    summary: str = dspy.OutputField(desc="A comprehensive summary of the search results with citations to the result IDs, e.g., '...information [1, 2].'")

class SummarizeSearchResults(dspy.Signature):
    """Summarize search results to extract the most important information related to the question."""
    
    question: str = dspy.InputField()
    search_results: dict[int, str] = dspy.InputField()
    summary: str = dspy.OutputField() # add citations to the ids in the summary

class VerboseSummarizeSearchRelevance(dspy.Signature):
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
    relevance_summary: str = dspy.OutputField(
        desc="A 2-3 sentence summary of how this passage relates to the query and its relevance"
    )

class SummarizeSearchRelevance(dspy.Signature):
    """Analyze and summarize how a search result addresses the given query."""
    
    query: str = dspy.InputField()
    passage: str = dspy.InputField() 
    relevance_summary: str = dspy.OutputField(
        desc="A summary of how this passage relates to the query and its relevance"
    )

class VerboseQuerySummarizer(dspy.Signature):
    """Summarize a technical question into one or two sentences, capturing the core problem and context.
    
    Your task is to distill a potentially long and detailed technical question into a very short summary. This summary should retain the essential information needed to understand the user's goal.
    
    Instructions:
    1.  Identify the main technology or library involved (e.g., Python, React, Chromadb).
    2.  Pinpoint the specific function, component, or concept that is causing the issue.
    3.  Determine the user's goal or what they are trying to achieve.
    4.  Combine these elements into a concise, one- or two-sentence summary.
    
    Example:
    - Question: "I'm trying to use the from_documents function in chromadb with my own list of documents, but I keep getting a 'NoneType' object has no attribute 'embed_documents' error. I'm using sentence-transformers for embeddings. How do I fix this?"
    - Summary: "The user is encountering a 'NoneType' error when using chromadb's `from_documents` function with sentence-transformers embeddings and needs to resolve it."
    """

    question: str = dspy.InputField(desc="The user's technical question.")
    summary: str = dspy.OutputField(desc="A one- or two-sentence summary of the question.")

class QuerySummarizer(dspy.Signature):
    """Summarize a technical question into one or two sentences."""

    question: str = dspy.InputField()
    summary: str = dspy.OutputField()