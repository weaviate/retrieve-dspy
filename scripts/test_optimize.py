"""Test script for GEPA optimization of SearchQueryWriter.

Uses the real SearchQueryWriter (with Weaviate) and the NITH Bright Biology
dataset. The metric checks whether the gold document was actually retrieved.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from retrieve_dspy import SearchQueryWriter
from retrieve_dspy.optimize.toy_dataset import load_nith_bright_biology
from retrieve_dspy.optimize.run_gepa import _retrieval_metric, _lookup_gold_content, run_gepa


# ---------------------------------------------------------------------------
# 1. Load dataset
# ---------------------------------------------------------------------------
print("=" * 60)
print("Step 1: Loading NITH Bright Biology dataset")
print("=" * 60)

trainset, valset = load_nith_bright_biology()
print(f"  trainset size: {len(trainset)}")
print(f"  valset   size: {len(valset)}")

ex = trainset[0]
print(f"  example inputs: {ex.inputs().keys()}")
print(f"  example labels: {ex.labels().keys()}")
print(f"  question: {ex.question}")
print(f"  gold_id:  {ex.gold_id}")

gold_content = _lookup_gold_content(ex.gold_id)
print(f"  gold content (first 200 chars): {gold_content[:200]}...")
print()


# ---------------------------------------------------------------------------
# 2. Baseline: run SearchQueryWriter on one example
# ---------------------------------------------------------------------------
print("=" * 60)
print("Step 2: Baseline SearchQueryWriter forward pass")
print("=" * 60)

retriever = SearchQueryWriter(
    collection_name="BrightBiology_Default",
    target_property_name="content",
    retrieved_k=1,
    verbose=False,
)

pred = retriever(question=ex.question)
print(f"  Search query: {pred.searches}")
print(f"  Sources returned: {len(pred.sources)}")

score = _retrieval_metric(ex, pred)
print(f"  Metric score: {score.score}")
print(f"  Metric feedback: {score.feedback}")
print()

# ---------------------------------------------------------------------------
# 3. Run GEPA optimization
# ---------------------------------------------------------------------------
print("=" * 60)
print("Step 4: Running GEPA optimization (auto='light')")
print("=" * 60)

optimized = run_gepa(
    retriever=SearchQueryWriter(
        collection_name="BrightBiology_Default",
        target_property_name="content",
        retrieved_k=1,
        verbose=False,
    ),
    dataset=(trainset, valset),
    auto="light",
    num_threads=2,
)


# ---------------------------------------------------------------------------
# 4. Compare before vs after
# ---------------------------------------------------------------------------
print()
print("=" * 60)
print("Step 5: Optimized scores on all examples")
print("=" * 60)

optimized_scores = []
for example in valset:
    p = optimized(question=example.question)
    s = _retrieval_metric(example, p)
    optimized_scores.append(s.score)
    hit = "HIT" if s.score == 1.0 else "MISS"
    print(f"  [{hit}] score={s.score:.2f} | q='{example.question[:60]}...'")

avg_optimized = sum(optimized_scores) / len(optimized_scores)
print(f"\n  Average optimized score: {avg_optimized:.3f}")
print(f"  Hits: {sum(1 for s in optimized_scores if s == 1.0)}/{len(optimized_scores)}")

print("\nDone!")
