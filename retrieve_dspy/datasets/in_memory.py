import json
import random
from typing import List, Dict, Optional, Set, Tuple

from datasets import load_dataset
from dspy import Example
import ir_datasets

def in_memory_dataset_loader(dataset_name: str):
    if dataset_name == "enron":
        return _in_memory_dataset_loader_enron()
    elif dataset_name == "wixqa":
        return _in_memory_dataset_loader_wixqa()
    elif dataset_name.startswith("beir/"):
        return _in_memory_dataset_loader_beir(dataset_name)
    elif dataset_name.startswith("bright/"):
        return _in_memory_dataset_loader_bright(dataset_name)
    elif dataset_name.startswith("lotte/"):
        return _in_memory_dataset_loader_lotte(dataset_name)
    elif dataset_name == "freshstack-angular":
        return _in_memory_dataset_loader_freshstack(subset="angular")
    elif dataset_name == "freshstack-godot":
        return _in_memory_dataset_loader_freshstack(subset="godot")
    elif dataset_name == "freshstack-langchain":
        return _in_memory_dataset_loader_freshstack(subset="langchain")
    elif dataset_name == "freshstack-laravel":
        return _in_memory_dataset_loader_freshstack(subset="laravel")
    elif dataset_name == "freshstack-yolo":
        return _in_memory_dataset_loader_freshstack(subset="yolo")
    else:
        return None
    
def _in_memory_dataset_loader_beir(dataset_name: str):
    dataset = ir_datasets.load(f"{dataset_name}")
    print(f"Loading BEIR dataset: {dataset_name}")
    docs, questions = [], []
    for doc in dataset.docs_iter():
        docs.append({
        "title": getattr(doc, "title", ""),
        "content": getattr(doc, "text", ""),
        "doc_id": getattr(doc, "doc_id", None)
    })
    qrels = {}
    for qrel in dataset.qrels_iter():
        query_id = qrel.query_id
        if query_id not in qrels:
            qrels[query_id] = []
        qrels[query_id].append(qrel.doc_id)
    for question in dataset.queries_iter():
        questions.append({
            "query_id": question.query_id,
            "question": question.text,
            "dataset_ids": qrels[question.query_id]
        })
    return docs, questions

def _in_memory_dataset_loader_bright(dataset_name: str):
    all_docs = load_dataset("xlangai/BRIGHT", "documents")
    split = dataset_name.split("/")[1]
    print(f"Loading BRIGHT dataset: {dataset_name}")
    docs, questions = [], []
    for doc in all_docs[split]:
        docs.append({
            "content": doc["content"],
            "dataset_id": doc["id"]
        })
    all_questions = load_dataset("xlangai/BRIGHT", "examples")
    for question in all_questions[split]:
        questions.append({
            "query_id": question["id"],
            "question": question["query"],
            "dataset_ids": question["gold_ids"]
        })
    return docs, questions

def _in_memory_dataset_loader_lotte(dataset_name: str):
    dataset = ir_datasets.load(f"{dataset_name}")
    print(f"Loading LOTTE dataset: {dataset_name}")
    docs, questions = [], []
    for doc in dataset.docs_iter():
        docs.append({
        "text": getattr(doc, "text", ""),
        "doc_id": getattr(doc, "doc_id", None)
    })
    qrels = {}
    for qrel in dataset.qrels_iter():
        query_id = qrel.query_id
        if query_id not in qrels:
            qrels[query_id] = []
        qrels[query_id].append(qrel.doc_id)
    for question in dataset.queries_iter():
        questions.append({
            "query_id": question.query_id,
            "question": question.text,
            "dataset_ids": qrels[question.query_id]
        })
    return docs, questions

def _in_memory_dataset_loader_enron():
    emails = _load_dataset_from_hf_hub("weaviate/enron-qa-emails-dasovich-j")
    questions = _load_dataset_from_hf_hub("weaviate/enron-qa-questions-dasovich-j")
    for question in questions:
        dataset_id = question.pop('dataset_id')
        question['dataset_ids'] = [dataset_id] if not isinstance(dataset_id, list) else dataset_id
    return emails, questions

def _in_memory_dataset_loader_wixqa():
    documents = _load_dataset_from_hf_hub(filepath="Wix/WixQA",subset="wix_kb_corpus")
    questions = _load_dataset_from_hf_hub(filepath="Wix/WixQA",subset="wixqa_expertwritten")
    for question in questions:
        article_ids = question.pop('article_ids')
        question['dataset_ids'] = [article_ids] if not isinstance(article_ids, list) else article_ids
    return documents, questions

def _in_memory_dataset_loader_freshstack(subset: str):
    docs = _load_dataset_from_hf_hub(filepath="freshstack/corpus-oct-2024", subset=subset)
    for doc in docs:
        doc['dataset_id'] = doc.pop('_id')
    questions = _load_dataset_from_hf_hub(
        filepath="freshstack/queries-oct-2024", 
        subset=subset, 
        train=False
    )

    for question in questions:
        all_relevant_ids = []
        nugget_data = []
        ids_per_nugget = {}
        
        for i, nugget in enumerate(question.get('nuggets', [])):
            nugget_id = f"{question['query_id']}_nugget_{i}"
            nugget_text = nugget['text']
            relevant_corpus_ids = nugget['relevant_corpus_ids']
            
            nugget_info = {
                'nugget_id': nugget_id,
                'text': nugget_text,
                'relevant_corpus_ids': relevant_corpus_ids
            }
            nugget_data.append(nugget_info)
            all_relevant_ids.extend(relevant_corpus_ids)
            
            ids_per_nugget[nugget_text] = relevant_corpus_ids
        
        unique_relevant_ids = list(dict.fromkeys(all_relevant_ids))
        
        question['dataset_ids'] = unique_relevant_ids
        question['ids_per_nugget'] = ids_per_nugget
        question['nugget_data'] = nugget_data
        question['num_nuggets'] = len(nugget_data)
        question["question"] = question["query_text"]
    
    return docs, questions

def _load_dataset_from_hf_hub(filepath, subset=None, train=True):
    ds = load_dataset(filepath, subset)
    if train:
        train_dataset = ds["train"]
    else:
        train_dataset = ds["test"]
    
    dataset_dicts = []
    for item in train_dataset:
        dataset_dicts.append(dict(item))
    
    return dataset_dicts

def _load_dataset_from_json(filepath):
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data

def split_dataset(dataset, train_ratio=0.8, shuffle=True):
    if shuffle:
        dataset = dataset.copy()
        random.shuffle(dataset)
        
    split_idx = int(len(dataset) * train_ratio)
    train_data = dataset[:split_idx]
    test_data = dataset[split_idx:]
    
    return train_data, test_data

def prepare_random_subset(
    queries: List[Dict],
    num_samples: int,
    samples_used_in_training: Optional[Set[str]] = None,
    seed: Optional[int] = 42,
) -> Tuple[List[Example], List[Example]]:
    random.seed(seed)

    examples = []
    for query in queries:
        q = query["question"]
        ex = Example().with_inputs("question")
        ex["question"] = q

        if "dataset_ids" in query:
            ex.dataset_ids = query["dataset_ids"]

        if "nugget_data" in query:
            ex.nugget_data = query["nugget_data"]

        examples.append(ex)

    # Filter any samples_used_in_training from the examples
    if samples_used_in_training:
        examples = [ex for ex in examples if ex["question"] not in samples_used_in_training]
    
    # Sample the desired number of examples
    random.shuffle(examples)
    examples = examples[:num_samples]

    return examples