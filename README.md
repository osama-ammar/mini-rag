# 🔍 Production-Ready RAG Application

A fully async, production-grade **Retrieval-Augmented Generation (RAG)** system built with FastAPI, PostgreSQL + PgVector, Celery, and Docker. Supports both OpenAI and local LLMs via Ollama.

> Built to explore and document the full lifecycle of a RAG system — from document ingestion and vector indexing to async processing, deployment, and evaluation.

---

## ✨ Features

- **Async document ingestion** — upload PDFs/text files, processed in the background via Celery workers
- **Vector search** — semantic similarity search using PgVector (PostgreSQL extension)
- **LLM flexibility** — swap between OpenAI GPT models and local Ollama models via a factory pattern
- **Async task queue** — Celery + Redis for non-blocking file processing and indexing pipelines
- **Monitoring** — Flower dashboard for Celery tasks, Grafana + Prometheus for system metrics
- **Database migrations** — Alembic for schema versioning (migrated from MongoDB → PostgreSQL)
- **Containerized** — full Docker Compose setup for all services

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Client / API                         │
│                      FastAPI (REST)                         │
└───────────────┬─────────────────────────┬───────────────────┘
                │                         │
       ┌────────▼────────┐      ┌─────────▼──────────┐
       │  DataController  │      │  ProcessController  │
       │  (CRUD / DB)     │      │  (RAG pipeline)     │
       └────────┬────────┘      └─────────┬──────────┘
                │                         │
       ┌────────▼──────────────────────────▼──────────┐
       │              PostgreSQL + PgVector             │
       │    (document store + vector embeddings)        │
       └───────────────────────────────────────────────┘
                         │
              ┌──────────▼──────────┐
              │    Celery Workers    │
              │  (file_processing,   │
              │   data_indexing)     │
              └──────────┬──────────┘
                         │
              ┌──────────▼──────────┐
              │    LLM Factory       │
              │  OpenAI │ Ollama     │
              └─────────────────────┘
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| API | FastAPI (async) |
| Database | PostgreSQL + PgVector |
| ORM / Migrations | SQLAlchemy + Alembic |
| Task Queue | Celery + Redis |
| LLM | OpenAI API / Ollama (local) |
| Embeddings | OpenAI `text-embedding-ada-002` / local models |
| Monitoring | Flower, Grafana, Prometheus |
| Containerization | Docker Compose |

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- Docker & Docker Compose
- (Optional) Ollama for local LLM inference

### 1. Clone & Configure

```bash
git clone https://github.com/osama-ammar/mini-rag.git
cd mini-rag
cp .env.example .env
# Edit .env and set your OPENAI_API_KEY and other credentials
```

### 2. Start Infrastructure

```bash
cd docker
cp .env.example .env
docker compose up -d
```

This starts PostgreSQL, Redis, Flower, Grafana, and Prometheus.

### 3. Install Python Dependencies

```bash
# Install system libs first
sudo apt update && sudo apt install libpq-dev gcc python3-dev

# Create and activate environment
conda create -n mini-rag python=3.10
conda activate mini-rag

pip install -r requirements.txt
```

### 4. Run Database Migrations

```bash
alembic upgrade head
```

### 5. Start the API Server

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### 6. Start Celery Workers (separate terminal)

```bash
# Worker
python -m celery -A celery_app worker \
  --queues=default,file_processing,data_indexing \
  --loglevel=info

# Beat scheduler (separate terminal)
python -m celery -A celery_app beat --loglevel=info
```

---

## 🌐 Services

| Service | URL | Notes |
|---|---|---|
| FastAPI docs | http://localhost:8000/docs | Swagger UI |
| Flower (Celery) | http://localhost:5555 | Task monitoring |
| Grafana | http://localhost:3000 | System metrics |
| Prometheus | http://localhost:9090 | Metrics scraper |

---

## 📡 API Overview

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/v1/projects` | Create a project (document namespace) |
| `POST` | `/api/v1/projects/{id}/upload` | Upload a document for processing |
| `POST` | `/api/v1/projects/{id}/index` | Trigger vector indexing |
| `POST` | `/api/v1/projects/{id}/search` | Semantic search |
| `POST` | `/api/v1/projects/{id}/answer` | Ask a question (RAG query) |

Download the full Postman collection from [`assets/mini-rag-app.postman_collection.json`](assets/mini-rag-app.postman_collection.json).

---

## 🧪 RAG Evaluation Pipeline

A standalone evaluation module is included to measure the quality of retrieval and generation against a local mini-RAG deployment.

### What it measures

| Metric | What it tests |
|---|---|
| **Precision@K** | Are the top-K retrieved chunks actually relevant? |
| **Recall@K** | Does retrieval capture all relevant chunks? |
| **Answer Faithfulness** | Is the generated answer grounded in the retrieved context? |
| **Answer Correctness** | Does the answer match the expected answer for the question? |

### Evaluation scripts

The main scripts live in [src/eval](src/eval) and share settings from [src/eval/eval_config.py](src/eval/eval_config.py):

- [src/eval/build_eval_dataset.py](src/eval/build_eval_dataset.py) builds an evaluation dataset by uploading context, processing it, and recording the chunk IDs returned by retrieval.
- [src/eval/eval_retrieval.py](src/eval/eval_retrieval.py) scores retrieval quality with Precision@K and Recall@K using the saved evaluation dataset.
- [src/eval/faithfullness.py](src/eval/faithfullness.py) asks an Ollama-based judge to score whether the generated answer is supported by the retrieved context.
- [src/eval/correctness.py](src/eval/correctness.py) asks the same judge to score whether the generated answer matches the expected answer.

### Running evaluations

From the repository root:

```bash
python src/eval/build_eval_dataset.py
python src/eval/eval_retrieval.py
python src/eval/faithfullness.py
python src/eval/correctness.py
```

The evaluation data is stored in [src/eval/eval_data.json](src/eval/eval_data.json). The pipeline retrieves, generates, and scores automatically using the shared config values in [src/eval/eval_config.py](src/eval/eval_config.py).

### Sample Output

```
┌─────────────────────────────────────────┐
│         RAG Evaluation Report           │
├─────────────────┬───────────────────────┤
│ Precision@5     │ 0.82                  │
│ Recall@5        │ 0.76                  │
│ Faithfulness    │ 0.91                  │
│ Answer Relevance│ 0.88                  │
└─────────────────┴───────────────────────┘
```

---

## 🦙 Using Local LLMs (Ollama)

You can run inference fully locally without an OpenAI API key.

1. Install [Ollama](https://ollama.ai) and pull a model:
   ```bash
   ollama pull llama3
   ```
2. Set in `.env`:
   ```
   LLM_PROVIDER=ollama
   OLLAMA_BASE_URL=http://localhost:11434
   OLLAMA_MODEL=llama3
   ```

Alternatively, use the [Colab + Ngrok notebook](https://colab.research.google.com/drive/1KNi3-9KtP-k-93T3wRcmRe37mRmGhL9p) to run Ollama remotely.

---

## 📂 Project Structure

```
mini-rag/
├── src/
│   ├── routes/         # FastAPI route handlers
│   ├── models/         # SQLAlchemy models
│   ├── controllers/    # DataController, ProcessController
│   ├── factories/      # LLM factory, VectorDB factory
│   └── celery_app/     # Celery task definitions
├── eval/               # RAG evaluation pipeline
├── docker/             # Docker Compose + service configs
├── alembic/            # Database migrations
├── assets/             # Postman collection
└── .github/.workflows/ # CI pipeline
```

---

## 🗺️ What I Learned Building This

This project was built step-by-step following a structured curriculum. Key engineering decisions and lessons:

- **Why migrate from MongoDB to PostgreSQL + PgVector?** — consolidating metadata and vector storage in one ACID-compliant store simplifies ops and removes synchronization overhead between two DBs.
- **Why Celery for indexing?** — document parsing and embedding generation are CPU/IO-bound and can take seconds per file. Offloading to workers keeps the API responsive and allows retries on failure.
- **Chunking strategy matters** — fixed-size chunking (used here) is simple but loses semantic boundaries. Future work: sentence-window and semantic chunking.
- **LLM factory pattern** — abstracting the LLM client behind a factory makes it trivial to swap OpenAI for Ollama or any other provider without touching business logic.

---

## 📚 Course Reference

This project was built alongside an Arabic YouTube series by [Eng. Abu Bakr Soliman](https://www.youtube.com/watch?v=Vv6e2Rb1Q6w&list=PLvLvlVqNQGHCUR2p0b8a0QpVjDUg50wQj). Each tutorial branch is tagged in the repo history (`tut-001` → `tut-017`).

---

## 📄 License

Apache 2.0 — see [LICENSE](LICENSE).