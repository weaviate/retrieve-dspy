import os
import math
from typing import Optional
from functools import lru_cache

import dspy

from retrieve_dspy.optimize.toy_dataset import load_search_dataset  # noqa: F401
from retrieve_dspy.optimize.utils import get_content_from_dataset_id


@lru_cache(maxsize=128)
def _lookup_gold_content(gold_id: str) -> str:
    """Cached lookup of gold document content from Weaviate."""
    return get_content_from_dataset_id(gold_id)


def _tokenize(text: str) -> set[str]:
    """Extract lowercased word tokens (3+ chars) for BM25 term overlap analysis."""
    return {w for w in text.lower().split() if len(w) >= 3}


def _term_overlap_report(query_terms: set[str], target_terms: set[str],
                         other_terms: set[str], other_label: str) -> str:
    """Build a concise BM25 term overlap diagnostic."""
    query_target = sorted(query_terms & target_terms)
    query_other = sorted(query_terms & other_terms - target_terms)
    target_missed = sorted(target_terms - query_terms)
    lines = []
    lines.append(f"  Query terms matching target: {query_target[:15] or '(none)'}")
    if query_other:
        lines.append(f"  Query terms matching {other_label} but NOT target: {query_other[:15]}")
    if target_missed:
        lines.append(f"  Key target terms your query missed: {target_missed[:20]}")
    return "\n".join(lines)


_FEEDBACK_TOP_K = 5  # max retrieved docs to show in feedback (controls token budget)


def _retrieval_metric(gold, pred, trace=None, pred_name=None, pred_trace=None, k=10):
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

    # --- nDCG@K scoring (single relevant document) ---
    # Primary: match on document ID (robust to whitespace/encoding differences).
    # Fallback: substring match on content (for sources without dataset_id).
    found_at_rank = None
    for rank, src in enumerate(sources[:k], start=1):
        src_id = str(getattr(src, "object_id", "") or "")
        if src_id and src_id == gold_id:
            found_at_rank = rank
            break

    if found_at_rank is None:
        gold_probe = snip(gold_content, 600).lower()
        if gold_probe:
            for rank, src in enumerate(sources[:k], start=1):
                src_text = snip(getattr(src, "content", "") or "", 1600).lower()
                if gold_probe in src_text:
                    found_at_rank = rank
                    break

    # nDCG with a single relevant document:
    #   DCG  = 1 / log2(1 + found_at_rank)
    #   IDCG = 1 / log2(1 + 1) = 1.0   (ideal is rank 1)
    #   nDCG = DCG / IDCG = 1 / log2(1 + found_at_rank)
    if found_at_rank is not None:
        score = 1.0 / math.log2(1 + found_at_rank)
    else:
        score = 0.0

    # --- Build actionable feedback ---
    question = str(gold.question)
    gold_snippet = snip(gold_content, 800)
    query_terms = _tokenize(search_query)
    target_terms = _tokenize(gold_content)

    if found_at_rank is not None:
        feedback = (
            f"{'PERFECT' if found_at_rank == 1 else 'PARTIAL'}: "
            f"The target document was retrieved at rank {found_at_rank} of {k} "
            f"(nDCG@{k}: {score:.3f}).\n"
            f"QUERY USED: {search_query}\n"
            f"TARGET DOCUMENT: {gold_snippet}"
        )
        if found_at_rank > 1:
            outranking = []
            for r, src in enumerate(sources[:found_at_rank - 1], start=1):
                content = snip(getattr(src, "content", "") or "", 400)
                outranking.append(f"  Rank {r}: {content}")
            outranking_block = "\n".join(outranking)
            # Combine all outranking doc terms for overlap analysis
            outranking_terms = set()
            for src in sources[:found_at_rank - 1]:
                outranking_terms |= _tokenize(getattr(src, "content", "") or "")
            overlap = _term_overlap_report(
                query_terms, target_terms, outranking_terms, "outranking docs"
            )
            feedback += (
                f"\nDOCUMENTS THAT OUTRANKED THE TARGET:\n{outranking_block}\n"
                f"BM25 TERM ANALYSIS:\n{overlap}\n"
                f"DIAGNOSIS: The target was retrieved but not at rank 1. "
                f"Your query shares terms with the outranking documents that are "
                f"absent from the target. Remove or replace those terms and add "
                f"verbatim terms from the target document to push it to rank 1."
            )
    else:
        retrieved_summaries = []
        top_retrieved_terms = set()
        show_k = min(_FEEDBACK_TOP_K, len(sources))
        for rank, src in enumerate(sources[:show_k], start=1):
            content = snip(getattr(src, "content", "") or "", 400)
            retrieved_summaries.append(f"  Rank {rank}: {content}")
            top_retrieved_terms |= _tokenize(getattr(src, "content", "") or "")
        retrieved_block = "\n".join(retrieved_summaries) if retrieved_summaries else "  (no results returned)"
        overlap = _term_overlap_report(
            query_terms, target_terms, top_retrieved_terms, "retrieved docs"
        )

        feedback = (
            f"FAILURE: The target document was NOT in the top {k} results "
            f"(nDCG@{k}: 0.0).\n"
            f"QUESTION: {question}\n"
            f"QUERY USED: {search_query}\n"
            f"WHAT YOU RETRIEVED (top {show_k}):\n{retrieved_block}\n"
            f"TARGET DOCUMENT (what you should have retrieved): {gold_snippet}\n"
            f"BM25 TERM ANALYSIS:\n{overlap}\n"
            f"DIAGNOSIS: BM25 matches exact keywords. Your query must contain "
            f"terms that appear verbatim in the target document. See the term "
            f"analysis above — add missed target terms and remove terms that "
            f"only match irrelevant documents."
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
    train_size: Optional[int] = None,
    val_size: Optional[int] = None,
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
        train_size: If set, truncate the trainset to this many examples.
        val_size: If set, truncate the valset to this many examples.

    Returns:
        The optimized retriever module.

    Example::

        from retrieve_dspy import SearchQueryWriter
        from retrieve_dspy.optimize import run_gepa

        retriever = SearchQueryWriter(
            collection_name="BrightBiology_Default",
            retrieved_k=10,
        )
        optimized = run_gepa(retriever=retriever, train_size=20, val_size=10)
    """
    # --- dataset ---
    if dataset is None:
        raise ValueError("dataset is required — pass a (trainset, valset) tuple from load_search_dataset().")
    else:
        trainset, valset = dataset

    if train_size is not None:
        trainset = trainset[:train_size]
    if val_size is not None:
        valset = valset[:val_size]

    # --- metric ---
    if metric is None:
        metric = _retrieval_metric

    # --- reflection LM ---
    if reflection_lm is None:
        reflection_lm = dspy.LM(
            "openai/gpt-5.4",
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
        print("\n\033[92m=== GEPA Optimization Complete ===\033[0m")
        print(f"  Total metric calls: {details.total_metric_calls}")
        print(f"  Best candidate index: {details.best_idx}")
        print(f"  Validation scores: {details.val_aggregate_scores}")

    return optimized_retriever
