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
    from dspy.teleprompt.gepa.gepa_utils import ScoreWithFeedback

    gold_id = str(gold.gold_id)
    gold_content = _lookup_gold_content(gold_id)

    if not gold_content:
        return ScoreWithFeedback(
            score=0.0,
            feedback="YOU WROTE THIS QUERY: (no query)\n"
                     "AND YOU RETURNED THIS: (no results)\n"
                     "THE TARGET DOCUMENT YOU SHOULD HAVE RETURNED IS THIS: (gold lookup failed)",
        )

    sources = getattr(pred, "sources", None) or []
    searches = getattr(pred, "searches", None) or []
    search_query = ", ".join(searches) if searches else "(no query)"

    def snip(text: str, n: int = 1200) -> str:
        text = (text or "").strip()
        text = " ".join(text.split())  # collapse whitespace/newlines
        return text[:n] + ("..." if len(text) > n else "")

    # Combine retrieved content into a single block (no ranks, no ids)
    retrieved_texts = []
    for src in sources[:3]:
        retrieved_texts.append(snip(getattr(src, "content", "") or "", 900))
    retrieved_content = "\n\n".join(retrieved_texts) if retrieved_texts else "(no results)"

    # Internal hit test (kept private; not shown in feedback)
    # Use a stable snippet of gold content to reduce brittleness vs full-doc containment
    gold_probe = snip(gold_content, 600).lower()
    found = False
    if gold_probe:
        for src in sources:
            src_text = snip(getattr(src, "content", "") or "", 1600).lower()
            if gold_probe in src_text:
                found = True
                break

    score = 1.0 if found else 0.0

    feedback = (
        f"YOU WROTE THIS QUERY: {search_query}\n"
        f"AND YOU RETURNED THIS: {retrieved_content}\n"
        f"THE TARGET DOCUMENT YOU SHOULD HAVE RETURNED IS THIS: {snip(gold_content, 1200)}"
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
