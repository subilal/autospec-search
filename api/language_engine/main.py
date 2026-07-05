import os
os.environ["HF_HUB_OFFLINE"] = "1"   # must be set before HF imports

import uuid
import requests
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from qdrant_client import QdrantClient
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore

try:
    import tomllib  # stdlib on Python 3.11+
except ModuleNotFoundError:
    import tomli as tomllib  # backport for Python 3.10

CONFIG_PATH = os.environ.get("CONFIG_PATH", "configs/config.toml")


with open(CONFIG_PATH, "rb") as f:
    _config = tomllib.load(f)

AUDI_QDRANT_VECTOR_STORE_PATH = os.environ.get("VECTOR_STORE_PATH", _config["paths"]["vector_store_path"])
OLLAMA_URL = os.environ.get("OLLAMA_URL", _config["llm"]["ollama_url"])
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", _config["llm"]["ollama_model"])
OLLAMA_TIMEOUT_SECONDS = int(os.environ.get("OLLAMA_TIMEOUT_SECONDS", _config["llm"]["request_timeout_seconds"]))
DEFAULT_TOP_K = int(os.environ.get("DEFAULT_TOP_K", _config["search"]["default_top_k"]))
MAX_TURNS_KEPT = int(os.environ.get("MAX_TURNS_KEPT", _config["conversation"]["max_turns_kept"]))



# Config
AUDI_QDRANT_VECTOR_STORE_PATH = "qdrant_audi_vector_store2"
AUDI_QDRANT_VECTOR_STORE_COLLECTION = "audi_spec_docs2"
AUTOSPEC_EMBEDDING_MODEL = "all-MiniLM-L6-v2"


# App setup
app = FastAPI(title="AutoSpec Search API")

# Allow the frontend (served from a different origin/port) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # fine for local dev; tighten for production
    allow_methods=["*"],
    allow_headers=["*"],
)


class SearchRequest(BaseModel):
    query: str
    top_k: int = 4
    session_id: str | None = None


class SourceChunk(BaseModel):
    text: str
    source: str
    page: int | None = None
    score: float


class SearchResponse(BaseModel):
    answer: str
    sources: list[SourceChunk]
    session_id: str   # always returned so the frontend can choose to reuse it


# Vector store (loaded once at startup, reused across requests)
def get_vector_store() -> QdrantVectorStore:
    client = QdrantClient(path=AUDI_QDRANT_VECTOR_STORE_PATH)
    embeddings = HuggingFaceEmbeddings(
        model_name=AUTOSPEC_EMBEDDING_MODEL,
        model_kwargs={"local_files_only": True},
    )
    return QdrantVectorStore(
        client=client,
        collection_name=AUDI_QDRANT_VECTOR_STORE_COLLECTION,
        embedding=embeddings,
    )


vector_store = get_vector_store()


# Conversation memory
# Simple in-memory store: { session_id: [ {"query": ..., "answer": ...}, ... ] }
# Fine for a local single-user tool. Resets when the server restarts.
conversation_store: dict[str, list[dict]] = {}

MAX_TURNS_KEPT = 5  # how many previous Q&A turns to carry forward as context


# LLM call (Ollama / Llama 3.1)
def call_llama(prompt: str) -> str:
    try:
        response = requests.post(
            OLLAMA_URL,
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
            timeout=300,
        )
        response.raise_for_status()
        return response.json()["response"]
    except requests.exceptions.ConnectionError:
        return "ERROR: Could not connect to Ollama. Is it running? (ollama serve)"
    except Exception as e:
        return f"ERROR calling Llama: {e}"


# Routes
@app.get("/", response_class=FileResponse)
def home():
    return FileResponse("templates/index.html")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/search", response_model=SearchResponse)
def search(request: SearchRequest):
    # Resolve session: reuse if given and known, otherwise start a fresh one
    if request.session_id and request.session_id in conversation_store:
        session_id = request.session_id
    else:
        session_id = str(uuid.uuid4())
        conversation_store[session_id] = []

    history = conversation_store[session_id]

    results = vector_store.similarity_search_with_score(request.query, k=request.top_k)

    if not results:
        answer = "No relevant documents found."
        history.append({"query": request.query, "answer": answer})
        return SearchResponse(answer=answer, sources=[], session_id=session_id)

    retrieved_context = "\n\n".join(
        f"[Source: {doc.metadata.get('source', 'unknown')}, page {doc.metadata.get('page', '?')}]\n{doc.page_content}"
        for doc, score in results
    )

    # Build prior-turns context, most recent last, capped to MAX_TURNS_KEPT
    history_block = ""
    if history:
        recent_turns = history[-MAX_TURNS_KEPT:]
        history_block = "\n\n".join(
            f"Previous question: {turn['query']}\nPrevious answer: {turn['answer']}"
            for turn in recent_turns
        )
        history_block = f"Conversation so far:\n{history_block}\n\n"

    prompt = f"""You are a helpful assistant. Use ONLY the context below to answer.
Always cite the source filename and page number for any fact you use.
{history_block}If the new question refers back to the conversation so far (e.g. "what about...", "and the other one?"), use that context to understand what is being asked.

Context for the current question:
{retrieved_context}

Question:
{request.query}

Answer (include citations like [source, page X]):
"""

    answer = call_llama(prompt)

    history.append({"query": request.query, "answer": answer})

    sources = [
        SourceChunk(
            text=doc.page_content,
            source=doc.metadata.get("source", "unknown"),
            page=doc.metadata.get("page"),
            score=float(score),
        )
        for doc, score in results
    ]

    return SearchResponse(answer=answer, sources=sources, session_id=session_id)


@app.post("/new_chat")
def new_chat(session_id: str | None = None):
    """Explicitly clear a session's history (optional helper endpoint)."""
    if session_id and session_id in conversation_store:
        del conversation_store[session_id]
    return {"status": "cleared"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)