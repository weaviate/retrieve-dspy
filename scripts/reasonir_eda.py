import json
import os
from collections import Counter
import random

import weaviate
from datasets import load_dataset

from retrieve_dspy.database.weaviate_database import weaviate_search_tool

vl_dataset = load_dataset("reasonir/reasonir-data", "vl")

def get_doc_and_ids(doc_pairs):
    doc_ids = []
    documents = []
    for dp in doc_pairs:
        doc_ids.append(str(dp['id']))
        documents.append(dp['content'])
    return documents, doc_ids

def process_pos_id2doc(entry, id2doc):
    pos_docs = entry["pos"]
    res = []
    for pos in pos_docs:
        instruction, doc_id = pos[0], pos[1]
        doc = id2doc[doc_id]
        res.append([instruction, doc])
    entry["pos"] = res
    return entry

hq_dataset = load_dataset("reasonir/reasonir-data", "hq")
bright_docs = load_dataset("xlangai/BRIGHT", "documents")

print(bright_docs.keys())

all_docs = []
all_ids = []
id2task = {}
for task in bright_docs.keys():
    docs, ids = get_doc_and_ids(bright_docs[task])
    all_docs.extend(docs)
    all_ids.extend(ids)
    for doc_id in ids:
        id2task[doc_id] = task

id2doc = {}
for i in range(len(all_docs)):
    id2doc[all_ids[i]] = all_docs[i]

# Count which BRIGHT tasks the ReasonIR questions map to (before resolving IDs)
task_counts = Counter()
for entry in hq_dataset["train"]:
    entry_tasks = set()
    for pos in entry["pos"]:
        doc_id = str(pos[1])
        if doc_id in id2task:
            entry_tasks.add(id2task[doc_id])
    for t in entry_tasks:
        task_counts[t] += 1

print("\nReasonIR hq questions distributed across BRIGHT tasks:")
for task, count in task_counts.most_common():
    print(f"  {task}: {count}")

# Extract biology-only subset with question + gold_id
biology_examples = []
for entry in hq_dataset["train"]:
    for pos in entry["pos"]:
        doc_id = str(pos[1])
        if id2task.get(doc_id) == "biology":
            query = entry["query"]
            question = query[1] if isinstance(query, list) else str(query)
            biology_examples.append({
                "question": question,
                "gold_id": doc_id,
            })

print(f"\nBiology subset (all): {len(biology_examples)} (question, gold_id) pairs")

# Only process the first N examples to keep runtime reasonable
MAX_EXAMPLES = 500
biology_examples = biology_examples[:MAX_EXAMPLES]
print(f"Biology subset (capped): {len(biology_examples)} examples")

# --------------------------------------------------------------------------- #
# BM25 baseline filter: drop examples where BM25@30 can't retrieve the gold doc
# --------------------------------------------------------------------------- #
BM25_K = 30
COLLECTION_NAME = "BrightBiology_Default"

weaviate_client = weaviate.connect_to_weaviate_cloud(
    cluster_url=os.getenv("WEAVIATE_URL"),
    auth_credentials=weaviate.auth.AuthApiKey(os.getenv("WEAVIATE_API_KEY")),
)

bm25_retrievable = []
bm25_unretrievable = []

for i, ex in enumerate(biology_examples):
    results = weaviate_search_tool(
        query=ex["question"],
        collection_name=COLLECTION_NAME,
        target_property_name="content",
        weaviate_client=weaviate_client,
        retrieved_k=BM25_K,
        search_type="bm25",
    )
    retrieved_ids = {r.object_id for r in results}
    if ex["gold_id"] in retrieved_ids:
        bm25_retrievable.append(ex)
    else:
        bm25_unretrievable.append(ex)
    if (i + 1) % 25 == 0 or (i + 1) == len(biology_examples):
        print(f"  BM25 filter progress: {i + 1}/{len(biology_examples)} "
              f"(retrievable: {len(bm25_retrievable)}, dropped: {len(bm25_unretrievable)})")

weaviate_client.close()

print(f"\nBM25@{BM25_K} filter results:")
print(f"  Retrievable: {len(bm25_retrievable)}")
print(f"  Unretrievable (dropped): {len(bm25_unretrievable)}")

# --------------------------------------------------------------------------- #
# Split into non-overlapping train / test sets (50/50)
# --------------------------------------------------------------------------- #
random.seed(42)
random.shuffle(bm25_retrievable)

split_idx = len(bm25_retrievable) // 2
train_examples = bm25_retrievable[:split_idx]
test_examples = bm25_retrievable[split_idx:]

print(f"\nTrain set: {len(train_examples)} examples")
print(f"Test set:  {len(test_examples)} examples")

train_path = "scripts/reasonir_bright_biology_train.json"
test_path = "scripts/reasonir_bright_biology_test.json"

with open(train_path, "w") as f:
    json.dump(train_examples, f, indent=2)
with open(test_path, "w") as f:
    json.dump(test_examples, f, indent=2)

print(f"Wrote {train_path}")
print(f"Wrote {test_path}")
