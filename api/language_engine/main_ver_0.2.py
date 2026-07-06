import os
import requests

os.environ["HF_HUB_OFFLINE"] = "1"  # must be set before HF imports

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from qdrant_client import QdrantClient
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore


# Config
AUDI_QDRANT_VECTOR_STORE_PATH = "../models/qdrant_audi_vector_store2"
AUDI_QDRANT_VECTOR_STORE_COLLECTION = "audi_spec_docs2"
AUTOSPEC_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.1"


# App setup
app = FastAPI(title="AutoSpec Search API")

# Allow the frontend (served from a different origin/port) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # fine for local dev; tighten for production
    allow_methods=["*"],
    allow_headers=["*"],
)


class SearchRequest(BaseModel):
    query: str
    top_k: int = 4


class SourceChunk(BaseModel):
    text: str
    source: str
    page: int | None = None
    score: float


class SearchResponse(BaseModel):
    answer: str
    sources: list[SourceChunk]


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


# LLM call (Ollama / Llama 3.1)
def call_llama(prompt: str) -> str:
    try:
        response = requests.post(
            OLLAMA_URL,
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
            timeout=120,
        )
        response.raise_for_status()
        return response.json()["response"]
    except requests.exceptions.ConnectionError:
        return "ERROR: Could not connect to Ollama. Is it running? (ollama serve)"
    except Exception as e:
        return f"ERROR calling Llama: {e}"


# Routes
@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/search", response_model=SearchResponse)
def search(request: SearchRequest):
    results = vector_store.similarity_search_with_score(request.query, k=request.top_k)

    if not results:
        return SearchResponse(answer="No relevant documents found.", sources=[])

    context = "\n\n".join(
        f"[Source: {doc.metadata.get('source', 'unknown')}, page {doc.metadata.get('page', '?')}]\n{doc.page_content}"
        for doc, score in results
    )

    prompt = f"""You are a helpful assistant. Use ONLY the context below to answer.
Always cite the source filename and page number for any fact you use.

Context:
{context}

Question:
{request.query}

Answer (include citations like [source, page X]):
"""

    answer = call_llama(prompt)

    sources = [
        SourceChunk(
            text=doc.page_content,
            source=doc.metadata.get("source", "unknown"),
            page=doc.metadata.get("page"),
            score=float(score),
        )
        for doc, score in results
    ]

    return SearchResponse(answer=answer, sources=sources)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
