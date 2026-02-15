"""
NITH Query Generator — Needle-in-the-Haystack dataset builder.

Iterates through a Weaviate collection using the cursor API, generates
NITH questions for eligible documents (>= 200 tokens), then filters out
questions the BaseRetriever can already answer. Only "hard" questions
(where the search engine fails to retrieve the gold document) are saved.

Usage:
    python scripts/NITH_query_generator.py
"""

import json
import os

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
MIN_TOKENS = 200
QUESTIONS_PER_DOC = 3
RETRIEVED_K = 20
MAX_QUESTIONS = 100
OUTPUT_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "retrieve_dspy",
    "optimize",
    "NITH_queries_bright_biology.json",
)

# ---------------------------------------------------------------------------
# Token counter
# ---------------------------------------------------------------------------
_enc = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(_enc.encode(text))


# ---------------------------------------------------------------------------
# DSPy Signature & Module
# ---------------------------------------------------------------------------
class GenerateNITHQuestions(dspy.Signature):
    """You are given a document from a biology knowledge base. Your task is to
    generate search questions that SHOULD retrieve this document but are HARD
    for a typical vector-similarity search engine to match.

    Each question should:
    - Be answerable by this document, but phrased indirectly or abstractly.
    - Use DIFFERENT vocabulary than the document — paraphrase, use synonyms,
      describe concepts at a higher or lower level of abstraction, or frame
      the question from a different angle (e.g. ask about the *implication*
      or *application* of what the document describes, rather than restating
      its content).
    - Sound like a real user who vaguely remembers the topic but doesn't
      recall the exact terms — ambiguous, exploratory, or roundabout phrasing.
    - AVOID copying distinctive keywords, named entities, or technical jargon
      that appear verbatim in the document. The more lexical overlap with the
      document, the worse the question is.
    - Vary in style: mix "why" questions, analogy-based questions, application
      questions, and questions that describe the concept without naming it.
    """

    document: str = dspy.InputField(desc="The full text of the target gold document.")
    dataset_id: str = dspy.InputField(desc="The dataset_id of the document (for context, do not include in questions).")
    questions: list[str] = dspy.OutputField(
        desc=f"A list of exactly {QUESTIONS_PER_DOC} distinct NITH questions targeting this document."
    )


generate_questions = dspy.Predict(GenerateNITHQuestions)


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
# Check retrieval — does BaseRetriever already find the gold doc?
# ---------------------------------------------------------------------------
def retriever_finds_gold(
    retriever: BaseRetriever,
    question: str,
    gold_id: str,
    weaviate_client: weaviate.WeaviateClient,
) -> bool:
    """Return True if the gold document appears in the retriever's results."""
    response = retriever.forward(question=question, weaviate_client=weaviate_client)
    for source in response.sources:
        if source.object_id == gold_id:
            return True
    return False


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

    # --- BaseRetriever for filtering ---
    retriever = BaseRetriever(
        collection_name=COLLECTION_NAME,
        weaviate_client=client,
        target_property_name=TARGET_PROPERTY,
        verbose=False,
        retrieved_k=RETRIEVED_K,
    )

    # --- Stats ---
    total_docs = 0
    skipped_short = 0
    total_questions_generated = 0
    already_retrievable = 0
    hard_questions: list[dict] = []

    print(f"\033[95m=== NITH Query Generator ===\033[0m")
    print(f"Collection: {COLLECTION_NAME}")
    print(f"Min tokens: {MIN_TOKENS}")
    print(f"Questions per doc: {QUESTIONS_PER_DOC}")
    print(f"Retrieved K: {RETRIEVED_K}")
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

            # --- Generate NITH questions ---
            try:
                prediction = generate_questions(
                    document=content,
                    dataset_id=dataset_id,
                )
                questions = prediction.questions
            except Exception as e:
                print(f"  \033[91mError generating questions: {e}\033[0m")
                continue

            if not questions:
                print(f"  No questions generated, skipping.")
                continue

            total_questions_generated += len(questions)

            # --- Filter: only keep questions the retriever CANNOT find ---
            for q in questions:
                found = retriever_finds_gold(
                    retriever=retriever,
                    question=q,
                    gold_id=dataset_id,
                    weaviate_client=client,
                )
                if found:
                    already_retrievable += 1
                    print(f"  \033[93m[EASY] \033[0m{q[:80]}...")
                else:
                    hard_questions.append({
                        "question": q,
                        "gold_id": dataset_id,
                    })
                    print(f"  \033[92m[HARD] \033[0m{q[:80]}...")

            # --- Check if we've hit the question limit ---
            if len(hard_questions) >= MAX_QUESTIONS:
                print(f"\n\033[92mReached {MAX_QUESTIONS} hard questions — stopping early.\033[0m")
                break

            # --- Progress summary every 10 eligible docs ---
            eligible_so_far = total_docs - skipped_short
            if eligible_so_far % 10 == 0:
                print(
                    f"\n  --- Progress: {total_docs} docs scanned | "
                    f"{skipped_short} skipped (short) | "
                    f"{total_questions_generated} questions generated | "
                    f"{already_retrievable} easy | {len(hard_questions)} hard ---\n"
                )

    except KeyboardInterrupt:
        print("\n\033[93mInterrupted! Saving progress so far...\033[0m")
    finally:
        client.close()

    # --- Save results ---
    output_path = os.path.normpath(OUTPUT_PATH)
    with open(output_path, "w") as f:
        json.dump(hard_questions, f, indent=2)

    # --- Final report ---
    print(f"\n\033[95m=== Generation Complete ===\033[0m")
    print(f"  Total documents scanned:    {total_docs}")
    print(f"  Skipped (< {MIN_TOKENS} tokens):     {skipped_short}")
    print(f"  Eligible documents:         {total_docs - skipped_short}")
    print(f"  Total questions generated:  {total_questions_generated}")
    print(f"  Already retrievable (easy): {already_retrievable}")
    print(f"  Hard questions saved:       {len(hard_questions)}")
    if total_questions_generated > 0:
        easy_pct = already_retrievable / total_questions_generated * 100
        print(f"  Baseline retrieval rate:    {easy_pct:.1f}%")
    print(f"\n  Output saved to: {output_path}")


if __name__ == "__main__":
    main()
