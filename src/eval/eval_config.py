BASE_URL = "http://localhost:8000"
PROJECT_ID = "1"
OLLAMA_HOST = "http://172.28.160.1:11434"
CHUNKS_PER_QUESTION = 2
RETRIEVE_K = 5
CHUNK_SIZE = 500
OVERLAP_SIZE = 50
MAX_CONTEXT_CHARS = 2000

PROCESS_REQ_BODY = {
    "chunk_size": CHUNK_SIZE,
    "overlap_size": OVERLAP_SIZE,
    "do_reset": 1,
}

PUSH_REQ_BODY = {
    "do_reset": 0,
}
