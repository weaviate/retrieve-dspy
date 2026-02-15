import os
from typing import Optional
from functools import lru_cache

import dspy

from retrieve_dspy.optimize.toy_dataset import load_nith_bright_biology
from retrieve_dspy.optimize.utils import get_content_from_dataset_id


@lru_cache(maxsize=128)
def _lookup_gold_content(gold_id: str) -> str:
    """Cached lookup of gold document content from Weaviate."""
    return get_content_from_dataset_id(gold_id)


def _retrieval_metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
    """Metric that checks if the gold document was retrieved.

    Compares retrieved source contents against the gold document content
    looked up from Weaviate via gold_id. Returns ScoreWithFeedback so
    GEPA's reflection LM gets actionable signal about what was retrieved
    vs what should have been.
    """
    from dspy.teleprompt.gepa.gepa_utils import ScoreWithFeedback

    gold_id = gold.gold_id
    gold_content = _lookup_gold_content(gold_id)

    if not gold_content:
        return ScoreWithFeedback(
            score=0.0,
            feedback=f"Could not look up gold document '{gold_id}' from Weaviate.",
        )

    # Get retrieved sources from the prediction
    sources = getattr(pred, "sources", None) or []
    searches = getattr(pred, "searches", None) or []
    search_query = ", ".join(searches) if searches else "(no query)"

    # Check for match — compare retrieved content against gold content
    gold_lower = gold_content.lower().strip()
    # Use a meaningful prefix of gold content for matching
    gold_snippet = gold_content[:200].strip()

    best_overlap = 0.0
    best_source_snippet = ""
    found_at_rank = None

    for i, source in enumerate(sources):
        source_content = getattr(source, "content", "") or ""
        source_lower = source_content.lower().strip()

        # Exact or near-exact match (gold content is contained in source or vice versa)
        if gold_lower in source_lower or source_lower in gold_lower:
            found_at_rank = i + 1
            best_overlap = 1.0
            best_source_snippet = source_content[:200]
            break

        # Token overlap as a softer signal
        gold_tokens = set(gold_lower.split())
        source_tokens = set(source_lower.split())
        if gold_tokens:
            overlap = len(gold_tokens & source_tokens) / len(gold_tokens)
            if overlap > best_overlap:
                best_overlap = overlap
                best_source_snippet = source_content[:200]

    # Score: 1.0 for exact hit, scaled by rank; partial overlap otherwise
    if found_at_rank is not None:
        # Reward finding it, with a small bonus for higher rank
        score = 1.0
        feedback = (
            f"Gold document found at rank {found_at_rank}/{len(sources)}. "
            f"Search query: '{search_query}'. "
            f"The query successfully retrieved the target document."
        )
    elif best_overlap >= 0.3:
        score = round(best_overlap * 0.5, 3)  # partial credit, max 0.5
        feedback = (
            f"Gold document NOT found in {len(sources)} results. "
            f"Search query: '{search_query}'. "
            f"Best partial overlap: {best_overlap:.0%}. "
            f"Closest retrieved content: '{best_source_snippet}...'. "
            f"Target content starts with: '{gold_snippet}...'. "
            f"The search query should use more specific terminology from the "
            f"target domain to retrieve the correct document."
        )
    else:
        score = 0.0
        feedback = (
            f"Gold document NOT found in {len(sources)} results. "
            f"Search query: '{search_query}'. "
            f"No meaningful overlap with target content. "
            f"Target content starts with: '{gold_snippet}...'. "
            f"The question was: '{gold.question}'. "
            f"The search query needs to be rewritten to better capture the key "
            f"concepts and domain-specific terms that would match the target document."
        )

    return ScoreWithFeedback(score=score, feedback=feedback)


def run_gepa(
    retriever: dspy.Module,
    dataset: Optional[tuple[list[dspy.Example], list[dspy.Example]]] = None,
    metric=None,
    reflection_lm=None,
    auto: str = "light",
    num_threads: int = 4,
    track_stats: bool = True,
    seed: int = 42,
):
    """Run GEPA optimization on a retrieve-dspy retriever.

    Args:
        retriever: An instantiated dspy.Module (e.g. SearchQueryWriter).
        dataset: A (trainset, valset) tuple of dspy.Example lists.
            If None, loads the NITH Bright Biology dataset.
        metric: A metric function ``(gold, pred, trace?, pred_name?, pred_trace?) -> score``.
            If None, uses the default retrieval metric that checks if the
            gold document was retrieved by comparing content from Weaviate.
        reflection_lm: The LM used by GEPA for reflection. Defaults to openai/gpt-4o.
        auto: GEPA budget preset — 'light', 'medium', or 'heavy'.
        num_threads: Number of parallel threads for evaluation.
        track_stats: Whether to track detailed optimization results.
        seed: Random seed for reproducibility.

    Returns:
        The optimized retriever module.

    Example::

        from retrieve_dspy import SearchQueryWriter
        from retrieve_dspy.optimize import run_gepa

        retriever = SearchQueryWriter(
            collection_name="BrightBiology_Default",
            retrieved_k=10,
        )
        optimized = run_gepa(retriever=retriever)
    """
    # --- dataset ---
    if dataset is None:
        trainset, valset = load_nith_bright_biology()
    else:
        trainset, valset = dataset

    # --- metric ---
    if metric is None:
        metric = _retrieval_metric

    # --- reflection LM ---
    if reflection_lm is None:
        reflection_lm = dspy.LM(
            "openai/gpt-4o",
            temperature=1.0,
            max_tokens=8000,
            api_key=os.getenv("OPENAI_API_KEY"),
        )

    # --- run GEPA ---
    optimizer = dspy.GEPA(
        metric=metric,
        auto=auto,
        reflection_lm=reflection_lm,
        num_threads=num_threads,
        track_stats=track_stats,
        seed=seed,
    )

    optimized_retriever = optimizer.compile(
        student=retriever,
        trainset=trainset,
        valset=valset,
    )

    # --- report results ---
    if track_stats and hasattr(optimized_retriever, "detailed_results"):
        details = optimized_retriever.detailed_results
        print(f"\n\033[92m=== GEPA Optimization Complete ===\033[0m")
        print(f"  Total metric calls: {details.total_metric_calls}")
        print(f"  Best candidate index: {details.best_idx}")
        print(f"  Validation scores: {details.val_aggregate_scores}")

    return optimized_retriever
