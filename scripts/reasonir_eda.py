from collections import Counter
from datasets import load_dataset

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
import json

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
biology_examples = biology_examples[:50]
print(f"Biology subset (truncated): {len(biology_examples)} (question, gold_id) pairs")

output_path = "scripts/reasonir_bright_biology.json"
with open(output_path, "w") as f:
    json.dump(biology_examples, f, indent=2)
print(f"Wrote to {output_path}")
