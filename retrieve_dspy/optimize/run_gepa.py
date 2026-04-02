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


def _retrieval_metric(gold, pred, trace=None, pred_name=None, pred_trace=None, k=5):
    from dspy.teleprompt.gepa.gepa_utils import ScoreWithFeedback

    gold_id = str(gold.gold_id)
    gold_content = _lookup_gold_content(gold_id)

    if not gold_content:
        return ScoreWithFeedback(
            score=0.0,
            feedback="Gold document lookup failed — cannot evaluate this example.",
        )

    sources = getattr(pred, "sources", None) or []
    searches = getattr(pred, "searches", None) or []
    search_query = ", ".join(searches) if searches else "(no query)"

    def snip(text: str, n: int = 1200) -> str:
        text = (text or "").strip()
        text = " ".join(text.split())
        return text[:n] + ("..." if len(text) > n else "")

    # --- Recall@K hit test with graded scoring ---
    gold_probe = snip(gold_content, 600).lower()
    found_at_rank = None
    if gold_probe:
        for rank, src in enumerate(sources[:k], start=1):
            src_text = snip(getattr(src, "content", "") or "", 1600).lower()
            if gold_probe in src_text:
                found_at_rank = rank
                break

    # Linear decay: rank 1 = 1.0, rank 2 = 0.8, rank 3 = 0.6, rank 4 = 0.4, rank 5 = 0.2
    if found_at_rank is not None:
        score = (k - found_at_rank + 1) / k
    else:
        score = 0.0

    # --- Build actionable feedback ---
    question = str(gold.question)
    gold_snippet = snip(gold_content, 800)

    if found_at_rank is not None:
        feedback = (
            f"{'PERFECT' if found_at_rank == 1 else 'PARTIAL'}: "
            f"The target document was retrieved at rank {found_at_rank} of {k} "
            f"(score: {score:.1f}).\n"
            f"QUERY USED: {search_query}\n"
            f"TARGET DOCUMENT: {gold_snippet}"
        )
        if found_at_rank > 1:
            # Show what outranked the target so reflection can diagnose
            outranking = []
            for r, src in enumerate(sources[:found_at_rank - 1], start=1):
                content = snip(getattr(src, "content", "") or "", 400)
                outranking.append(f"  Rank {r}: {content}")
            outranking_block = "\n".join(outranking)
            feedback += (
                f"\nDOCUMENTS THAT OUTRANKED THE TARGET:\n{outranking_block}\n"
                f"DIAGNOSIS: The target was retrieved but not at rank 1. "
                f"Your query may be partially matching irrelevant documents. "
                f"Try adding more specific terms from the target document or "
                f"removing broad terms that attract the outranking documents."
            )
    else:
        retrieved_summaries = []
        for rank, src in enumerate(sources[:k], start=1):
            content = snip(getattr(src, "content", "") or "", 400)
            retrieved_summaries.append(f"  Rank {rank}: {content}")
        retrieved_block = "\n".join(retrieved_summaries) if retrieved_summaries else "  (no results returned)"

        feedback = (
            f"FAILURE: The target document was NOT in the top {k} results "
            f"(score: 0.0).\n"
            f"QUESTION: {question}\n"
            f"QUERY USED: {search_query}\n"
            f"WHAT YOU RETRIEVED (top {k}):\n{retrieved_block}\n"
            f"TARGET DOCUMENT (what you should have retrieved): {gold_snippet}\n"
            f"DIAGNOSIS: Compare the target document against your query. "
            f"Your query may be missing key terms from the target, using synonyms "
            f"that BM25 cannot match, or focusing on the wrong aspect of the question. "
            f"BM25 matches exact keywords — your query must contain terms that appear "
            f"verbatim in the target document."
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
    use_wandb: bool = False,
    wandb_init_kwargs: Optional[dict] = None,
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
        use_wandb=use_wandb,
        wandb_init_kwargs=wandb_init_kwargs,
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
