"""
NITH Query Generator — Needle-in-the-Haystack dataset builder.

Iterates through a Weaviate collection using the cursor API, generates
NITH questions for eligible documents (>= MIN_TOKENS tokens), grades each
question by difficulty (easy / medium / hard / extreme), stores gold document
content and baseline retrieval rank, then saves a balanced dataset.

Usage:
    python scripts/NITH_query_generator.py
"""

import json
import os
import random

import dspy
import tiktoken
import weaviate
from weaviate.classes.init import Auth

from retrieve_dspy.retrievers.base_retriever import BaseRetriever

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
COLLECTION_NAME = "BrightBiology_Default"
TARGET_PROPERTY = "content"
MIN_TOKENS = 100
QUESTIONS_PER_DOC = 5
MAX_K_FOR_GRADING = 100
MAX_QUESTIONS = 200
OUTPUT_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "retrieve_dspy",
    "optimize",
    "NITH_queries_bright_biology.json",
)

QUESTION_STYLES = [
    "Ask about a practical application or implication without using any terminology from the document.",
    "Describe the phenomenon the document explains as if you vaguely remember it but forgot the technical terms.",
    "Ask a 'why' or 'how' question that requires the document's content to answer, using everyday language.",
    "Frame the question from an adjacent field or analogy — someone approaching this topic from a different discipline.",
    "Ask a question that a curious beginner might pose after encountering this topic for the first time.",
]

# ---------------------------------------------------------------------------
# Token counter
# ---------------------------------------------------------------------------
_enc = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(_enc.encode(text))


# ---------------------------------------------------------------------------
# DSPy Signature & Module
# ---------------------------------------------------------------------------
class GenerateNITHQuestion(dspy.Signature):
    """You are given a document from a biology knowledge base and a specific
    question style to follow. Your task is to generate ONE search question
    that SHOULD retrieve this document but is HARD for a typical
    vector-similarity search engine to match.

    The question should:
    - Be answerable by this document, but phrased indirectly or abstractly.
    - Follow the requested style closely.
    - Use DIFFERENT vocabulary than the document — paraphrase, use synonyms,
      describe concepts at a higher or lower level of abstraction, or frame
      the question from a different angle.
    - Sound like a real user who vaguely remembers the topic but doesn't
      recall the exact terms.
    - AVOID copying distinctive keywords, named entities, or technical jargon
      that appear verbatim in the document.
    """

    document: str = dspy.InputField(desc="The full text of the target gold document.")
    dataset_id: str = dspy.InputField(desc="The dataset_id of the document (for context, do not include in the question).")
    style: str = dspy.InputField(desc="The specific question style to follow.")
    question: str = dspy.OutputField(desc="A single NITH question targeting this document in the requested style.")


generate_question = dspy.Predict(GenerateNITHQuestion)


# ---------------------------------------------------------------------------
# Cursor-based iteration over Weaviate collection
# ---------------------------------------------------------------------------
def iterate_collection(client: weaviate.WeaviateClient, collection_name: str):
    """Yield (dataset_id, content) for every object using the cursor API."""
    collection = client.collections.get(collection_name)
    for item in collection.iterator(include_vector=False):
        dataset_id = str(item.properties.get("dataset_id") or item.uuid)
        content = str(item.properties.get(TARGET_PROPERTY, ""))
        yield dataset_id, content


# ---------------------------------------------------------------------------
# Retrieval ranking — find the rank of the gold doc
# ---------------------------------------------------------------------------
def get_gold_rank(
    retriever: BaseRetriever,
    question: str,
    gold_id: str,
    weaviate_client: weaviate.WeaviateClient,
    max_k: int = MAX_K_FOR_GRADING,
) -> int | None:
    """Return the 1-based rank of the gold doc, or None if not found in top max_k."""
    retriever.retrieved_k = max_k
    response = retriever.forward(question=question, weaviate_client=weaviate_client)
    for i, source in enumerate(response.sources):
        if source.object_id == gold_id:
            return i + 1
    return None


def rank_to_difficulty(rank: int | None) -> str:
    """Map a retrieval rank to a difficulty label."""
    if rank is not None and rank == 1:
        return "easy"
    if rank is not None and rank <= 10:
        return "medium"
    if rank is not None and rank <= 20:
        return "hard"
    return "extreme"


# ---------------------------------------------------------------------------
# Balance the dataset by difficulty
# ---------------------------------------------------------------------------
def balance_dataset(all_questions: list[dict], max_questions: int) -> list[dict]:
    """Sample a balanced dataset across difficulty buckets."""
    by_difficulty: dict[str, list[dict]] = {"easy": [], "medium": [], "hard": [], "extreme": []}
    for q in all_questions:
        by_difficulty[q["difficulty"]].append(q)

    target_per_bucket = max_questions // 4
    balanced = []
    for diff, qs in by_difficulty.items():
        balanced.extend(random.sample(qs, min(target_per_bucket, len(qs))))

    random.shuffle(balanced)
    return balanced


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    # --- Weaviate client ---
    client = weaviate.connect_to_weaviate_cloud(
        cluster_url=os.getenv("WEAVIATE_URL"),
        auth_credentials=Auth.api_key(os.getenv("WEAVIATE_API_KEY")),
    )

    # --- DSPy LM for question generation ---
    lm = dspy.LM(
        "openai/gpt-4o",
        temperature=1.0,
        max_tokens=4000,
        api_key=os.getenv("OPENAI_API_KEY"),
    )
    dspy.configure(lm=lm, track_usage=True)

    # --- BaseRetriever for grading ---
    retriever = BaseRetriever(
        collection_name=COLLECTION_NAME,
        weaviate_client=client,
        target_property_name=TARGET_PROPERTY,
        verbose=False,
        retrieved_k=MAX_K_FOR_GRADING,
    )

    # --- Stats ---
    total_docs = 0
    skipped_short = 0
    total_questions_generated = 0
    all_questions: list[dict] = []
    difficulty_counts = {"easy": 0, "medium": 0, "hard": 0, "extreme": 0}
    style_difficulty_counts: dict[str, dict[str, int]] = {
        style: {"easy": 0, "medium": 0, "hard": 0, "extreme": 0}
        for style in QUESTION_STYLES
    }

    print(f"\033[95m=== NITH Query Generator ===\033[0m")
    print(f"Collection: {COLLECTION_NAME}")
    print(f"Min tokens: {MIN_TOKENS}")
    print(f"Questions per doc: {QUESTIONS_PER_DOC}")
    print(f"Max K for grading: {MAX_K_FOR_GRADING}")
    print(f"Question styles: {len(QUESTION_STYLES)}")
    print()

    try:
        for dataset_id, content in iterate_collection(client, COLLECTION_NAME):
            total_docs += 1

            # --- Token gate ---
            token_count = count_tokens(content)
            if token_count < MIN_TOKENS:
                skipped_short += 1
                if total_docs % 50 == 0:
                    print(f"  [{total_docs}] Skipped (too short): {dataset_id} ({token_count} tokens)")
                continue

            print(f"\033[96m[{total_docs}] Processing: {dataset_id} ({token_count} tokens)\033[0m")

            # --- Generate NITH questions (one per style) ---
            styles_to_use = QUESTION_STYLES[:QUESTIONS_PER_DOC]
            question_style_pairs: list[tuple[str, str]] = []
            for style in styles_to_use:
                try:
                    prediction = generate_question(
                        document=content,
                        dataset_id=dataset_id,
                        style=style,
                    )
                    question_style_pairs.append((prediction.question, style))
                except Exception as e:
                    print(f"  \033[91mError generating question (style: {style[:40]}...): {e}\033[0m")

            if not question_style_pairs:
                print(f"  No questions generated, skipping.")
                continue

            total_questions_generated += len(question_style_pairs)

            # --- Grade each question by difficulty ---
            for q, style in question_style_pairs:
                rank = get_gold_rank(
                    retriever=retriever,
                    question=q,
                    gold_id=dataset_id,
                    weaviate_client=client,
                )
                difficulty = rank_to_difficulty(rank)
                difficulty_counts[difficulty] += 1
                style_difficulty_counts[style][difficulty] += 1

                all_questions.append({
                    "question": q,
                    "gold_id": dataset_id,
                    "gold_content": content,
                    "gold_token_count": token_count,
                    "difficulty": difficulty,
                    "found_at_k": rank,
                    "baseline_rank": rank,
                    "question_style": style,
                })

                rank_str = str(rank) if rank is not None else ">100"
                color = {"easy": "32", "medium": "33", "hard": "38;5;208", "extreme": "31"}[difficulty]
                print(f"  \033[{color}m[{difficulty.upper()} rank={rank_str}] \033[0m{q[:80]}...")

            # --- Check if we have enough raw questions ---
            if len(all_questions) >= MAX_QUESTIONS * 2:
                print(f"\n\033[92mCollected {len(all_questions)} questions — stopping collection.\033[0m")
                break

            # --- Progress summary every 10 eligible docs ---
            eligible_so_far = total_docs - skipped_short
            if eligible_so_far % 10 == 0:
                print(
                    f"\n  --- Progress: {total_docs} docs scanned | "
                    f"{skipped_short} skipped (short) | "
                    f"{total_questions_generated} generated | "
                    f"easy={difficulty_counts['easy']} medium={difficulty_counts['medium']} "
                    f"hard={difficulty_counts['hard']} extreme={difficulty_counts['extreme']} ---\n"
                )

    except KeyboardInterrupt:
        print("\n\033[93mInterrupted! Saving progress so far...\033[0m")
    finally:
        client.close()

    # --- Save full unbalanced dataset ---
    output_path = os.path.normpath(OUTPUT_PATH)
    full_output_path = output_path.replace(".json", "_full.json")
    with open(full_output_path, "w") as f:
        json.dump(all_questions, f, indent=2)

    # --- Save multiple difficulty mixes ---
    # Each mix is a tuple of (easy%, medium%, hard%, extreme%) that sums to 100.
    # The label encodes the percentages: e.g. "25_25_25_25" means equal split.
    DIFFICULTY_MIXES: list[tuple[int, int, int, int]] = [
        (25, 25, 25, 25),   # balanced
        (100, 0, 0, 0),     # easy only
        (0, 100, 0, 0),     # medium only
        (0, 0, 50, 50),     # hard + extreme only
        (50, 30, 15, 5),    # easy-heavy
        (5, 15, 30, 50),    # hard-heavy
        (0, 0, 0, 100),     # extreme only
        (40, 40, 20, 0),    # no extreme
    ]
    DIFFS = ["easy", "medium", "hard", "extreme"]

    by_difficulty: dict[str, list[dict]] = {d: [] for d in DIFFS}
    for q in all_questions:
        by_difficulty[q["difficulty"]].append(q)

    saved_paths: list[tuple[str, str, int, dict[str, int]]] = []

    for pcts in DIFFICULTY_MIXES:
        mix_name = f"{pcts[0]}_{pcts[1]}_{pcts[2]}_{pcts[3]}"
        mix_questions: list[dict] = []
        mix_counts: dict[str, int] = {}
        for diff, pct in zip(DIFFS, pcts):
            n_target = int(round(MAX_QUESTIONS * pct / 100))
            sampled = random.sample(by_difficulty[diff], min(n_target, len(by_difficulty[diff])))
            mix_questions.extend(sampled)
            mix_counts[diff] = len(sampled)

        random.shuffle(mix_questions)
        mix_path = output_path.replace(".json", f"_{mix_name}.json")

        with open(mix_path, "w") as f:
            json.dump(mix_questions, f, indent=2)

        saved_paths.append((mix_name, mix_path, len(mix_questions), mix_counts))

    # --- Final report ---
    print(f"\n\033[95m=== Generation Complete ===\033[0m")
    print(f"  Total documents scanned:    {total_docs}")
    print(f"  Skipped (< {MIN_TOKENS} tokens):      {skipped_short}")
    print(f"  Eligible documents:         {total_docs - skipped_short}")
    print(f"  Total questions generated:  {total_questions_generated}")
    print(f"\n  Difficulty distribution (all {len(all_questions)} questions):")
    for diff in ["easy", "medium", "hard", "extreme"]:
        print(f"    {diff:>8}: {difficulty_counts[diff]}")
    if total_questions_generated > 0:
        easy_pct = difficulty_counts["easy"] / total_questions_generated * 100
        print(f"\n  Baseline retrieval rate (top-5): {easy_pct:.1f}%")

    print(f"\n  \033[95mQuestion style → difficulty breakdown:\033[0m")
    for i, style in enumerate(QUESTION_STYLES):
        style_label = style[:70]
        counts = style_difficulty_counts[style]
        style_total = sum(counts.values())
        if style_total == 0:
            continue
        dist = " | ".join(
            f"{d}={counts[d]}({counts[d]/style_total*100:.0f}%)"
            for d in ["easy", "medium", "hard", "extreme"]
        )
        print(f"    Style {i+1}: {style_label}...")
        print(f"             {dist}  (n={style_total})")

    print(f"\n  \033[95mSaved datasets (easy_medium_hard_extreme %):\033[0m")
    print(f"    {'full':.<20} {full_output_path}  ({len(all_questions)} questions)")
    for mix_name, mix_path, count, mix_counts in saved_paths:
        dist = " | ".join(f"{d}={mix_counts[d]}" for d in DIFFS)
        print(f"    {mix_name:.<20} {mix_path}  ({count} questions: {dist})")


if __name__ == "__main__":
    main()
