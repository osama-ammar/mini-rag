"""Build a local evaluation dataset for retrieval benchmarking.

This script loads questions from the RAGCare-QA dataset, uploads each context
into the local mini-RAG service, processes and indexes it, then records the
chunk IDs returned for each context so they can be used as expected retrieval
results for later evaluation.
"""

import json
import requests
from datasets import load_dataset

from eval_config import (
    BASE_URL,
    PROJECT_ID,
    CHUNKS_PER_QUESTION,
    PROCESS_REQ_BODY,
    PUSH_REQ_BODY,
    MAX_CONTEXT_CHARS,
)


# prepare_eval_dataset.py
#
# PURPOSE:
# This script bridges the gap between the RAGCare-QA benchmark dataset
# and our RAG system. Since chunk IDs are specific to our database instance,
# we can't use external benchmarks directly. Instead we:
#   1. Upload each question's context as a document into our system
#   2. Index it so PgVector assigns real chunk IDs
#   3. Search with the same context to discover which chunk IDs were created
#   4. Save (question, expected_chunk_ids, expected_answer) as our eval dataset
#
# This gives us a ground-truth eval set that is tied to our actual database,
# making Precision@K and Recall@K meaningful against real retrieval behavior.
# User Question
#      │
#      ▼
# Search Endpoint ──► Retrieved Chunk IDs
#                            │
#                     Compare against
#                            │
#                     Expected Chunk IDs  ◄── eval_data.json
#                            │
#                     ┌──────┴──────┐
#                     │             │
#               Precision@K    Recall@K
#           "how clean is    "how complete
#            the retrieval"   is the retrieval"






def load_ragcare_qa():
    dataset = load_dataset("ChatMED-Project/RAGCare-QA")
    # filter only Basic RAG questions - matches your system type
    print(dataset["train"].column_names)
    # print(dataset["train"][0])
    items = [item for item in dataset["train"] if item["RAG Pipeline"] == "Basic RAG"]
    return items[:50]  # start with 50 questions


def upload_context_as_document(context_text: str) -> bool:
    # truncate to avoid exceeding embedding model context length
    context_text = context_text[:MAX_CONTEXT_CHARS]
    url = f"{BASE_URL}/api/v1/data/upload/{PROJECT_ID}"
    files = {"file": ("context.txt", context_text.encode(), "text/plain")}
    response = requests.post(url, files=files)
    return response.status_code == 200

def process_uploaded_documents(req_body) -> bool:
    url = f"{BASE_URL}/api/v1/data/process/{PROJECT_ID}"
    response = requests.post(url,json=req_body)
    return response.status_code == 200

def push_to_index(req_body) -> bool:
    url = f"{BASE_URL}/api/v1/nlp/index/push/{PROJECT_ID}"
    response = requests.post(url, json=req_body)
    print(f"Push response: {response.status_code} - {response.text}")
    return response.status_code == 200

def get_chunk_ids_for_context(context_text: str, k: int = CHUNKS_PER_QUESTION) -> list:
    url = f"{BASE_URL}/api/v1/nlp/index/search/{PROJECT_ID}"
    response = requests.post(url, json={"text": context_text, "limit": k})
    if response.status_code != 200 or not response.text:
        print(f"  ⚠ search failed (status {response.status_code}), skipping...")
        return []
    
    data = response.json()
    # print(data)
    return [result["chunk_id"] for result in data["results"]]

def build_eval_dataset(items: list) -> list:
    eval_data = []
   
    for item in items:
        context   = item["Context"]
        # upload_context_as_document(context)
        upload_context_as_document(item["Context"])
        
   


    print("Process result:", process_uploaded_documents(PROCESS_REQ_BODY))
    print("Push result:", push_to_index(PUSH_REQ_BODY))
        
        
    for item in items:
        context   = item["Context"]
        question  = item["Question"]
        answer    = item["Text Answer"]

        # search with context to find which chunk_ids were created
        expected_chunk_ids = get_chunk_ids_for_context(context)

        eval_data.append({
            "question": question,
            "expected_chunk_ids": expected_chunk_ids,
            "expected_answer": answer,
            "specialty": item["Type"],
            "complexity": item["Complexity"]
        })
        print(f"✓ processed: {question[:60]}...")

    return eval_data

def save_eval_dataset(eval_data: list, path: str = "eval_data.json"):
    with open(path, "w") as f:
        json.dump(eval_data, f, indent=2)
    print(f"\nSaved {len(eval_data)} questions to {path}")

if __name__ == "__main__":
    print("Loading RAGCare-QA...")
    items = load_ragcare_qa()
    
    print(f"Building eval dataset from {len(items)} questions...")
    eval_data = build_eval_dataset(items)
    
    save_eval_dataset(eval_data)