import json
import requests
from datasets import load_dataset

from eval_config import BASE_URL, PROJECT_ID, CHUNKS_PER_QUESTION, OLLAMA_HOST

import re

def call_ollama_judge(prompt: str) -> dict:
    url = f"{OLLAMA_HOST}/v1/chat/completions"
    response = requests.post(url, json={
        "model": "qwen2:7b",
        "messages": [{"role": "user", "content": prompt}],
        "stream": False
    })
    result = response.json()
    text = result["choices"][0]["message"]["content"]
    
    if "<think>" in text:
        text = text.split("</think>")[-1].strip()
    
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # fallback: extract score from prose response
        match = re.search(r'(\d+\.?\d*)', text)
        score = float(match.group(1)) if match else 0.0
        # normalize if score > 1 (e.g. "0.5 out of 1" vs "50 out of 100")
        if score > 1:
            score = score / 100
        return {"faithfulness_score": score, "reason": "extracted from prose"}
def load_eval_dataset(path: str) -> list:
    with open(path, 'r') as file:
        return json.load(file)

def build_faithfulness_prompt(context: str, answer: str) -> str:
    return f"""You are an evaluation judge for a medical RAG system.

        Given:
        - Retrieved context: {context}
        - Generated answer: {answer}

        Task: Score how much the generated answer is supported by the retrieved context, from 0.0 to 1.0.

        Rules:
        - Be strict
        - 1.0 = answer is fully supported by context
        - 0.5 = answer is partially supported, contains extra info not in context
        - 0.0 = answer is unrelated to or contradicts the context
        - Indicate in reason if the answer is: unrelated, adds extra info, or partially supported

        Return ONLY a JSON object, no other text:
        {{"faithfulness_score": <0.0 to 1.0>, "reason": "<one sentence>"}}"""


def get_context(context_text: str, k: int = CHUNKS_PER_QUESTION) -> list:
    url = f"{BASE_URL}/api/v1/nlp/index/search/{PROJECT_ID}"
    response = requests.post(url, json={"text": context_text, "limit": k})
    if response.status_code != 200 or not response.text:
        print(f"  ⚠ search failed (status {response.status_code}), skipping...")
        return []
    
    data = response.json()
    # print(data)
    return [result["text"] for result in data["results"]]

import time

def get_answer(context_text: str, k: int = CHUNKS_PER_QUESTION, retries: int = 3) -> str:
    url = f"{BASE_URL}/api/v1/nlp/index/answer/{PROJECT_ID}"
    for attempt in range(retries):
        response = requests.post(url, json={"text": context_text, "limit": k}, timeout=60)
        if response.status_code == 200 and response.text:
            return response.text
        print(f"  ⚠ attempt {attempt+1} failed ({response.status_code}), retrying...")
        time.sleep(3)
    return ""


def get_faithfullness_scores(eval_data_path="eval_data.json"):
    data = load_eval_dataset(eval_data_path)
    scores=[]
    for item in data:
        retrieved_context = get_context(item["question"])
        if not retrieved_context:
            continue
        
        pipeline_answer = get_answer(item["question"])
        if not pipeline_answer:
            continue
            
        prompt = build_faithfulness_prompt(retrieved_context[0], pipeline_answer)
        score_dict = call_ollama_judge(prompt)
        score = float(score_dict["faithfulness_score"])
        scores.append(score)
    final_score = sum(scores)/len(scores)
    print(f"AVG Failthfullness score is {final_score} ")

get_faithfullness_scores()