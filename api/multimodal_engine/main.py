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
from langchain_core.embeddings import Embeddings
from PIL import Image
from sentence_transformers import SentenceTransformer


# Config
AUDI_QDRANT_VECTOR_STORE_PATH = "qdrant_audi_vector_store2"
AUDI_QDRANT_TEXT_COLLECTION = "audi_spec_docs2"                       # chunks + tables + image captions
AUDI_QDRANT_IMAGE_COLLECTION = f"{AUDI_QDRANT_TEXT_COLLECTION}_images"  # raw images (CLIP)
AUTOSPEC_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CLIP_EMBEDDING_MODEL = "clip-ViT-B-32"
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://ollama:11434/api/generate")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.1")


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
    content_type: str = "text"        # "text" | "table" | "image"
    image_path: str | None = None     # set only for content_type == "image"


class SearchResponse(BaseModel):
    answer: str
    sources: list[SourceChunk]
    session_id: str   # always returned so the frontend can choose to reuse it


# CLIP embedding adapter for the image collection (mirrors ingestion script).
# embed_documents: treats each input as an image path (used at ingest time).
# embed_query: encodes plain text - this is what lets a text query match images.
class ClipImageEmbeddings(Embeddings):
    def __init__(self, model_name: str = CLIP_EMBEDDING_MODEL):
        self.model = SentenceTransformer(model_name, local_files_only=True)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        images = [Image.open(t).convert("RGB") if os.path.isfile(t) else t for t in texts]
        return self.model.encode(images, convert_to_numpy=True).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.model.encode(text, convert_to_numpy=True).tolist()


# Single shared Qdrant client (one file lock per path), two collections on top of it.
client = QdrantClient(path=AUDI_QDRANT_VECTOR_STORE_PATH)

text_store = QdrantVectorStore(
    client=client,
    collection_name=AUDI_QDRANT_TEXT_COLLECTION,
    embedding=HuggingFaceEmbeddings(model_name=AUTOSPEC_EMBEDDING_MODEL, model_kwargs={"local_files_only": True}),
)

image_store = None
if client.collection_exists(AUDI_QDRANT_IMAGE_COLLECTION):
    image_store = QdrantVectorStore(
        client=client,
        collection_name=AUDI_QDRANT_IMAGE_COLLECTION,
        embedding=ClipImageEmbeddings(),
    )


def search_all(query: str, k: int = 4):
    """Search both collections, tag each hit's modality, return the top-k overall by score."""
    hits = [(doc, score, "text") for doc, score in text_store.similarity_search_with_score(query, k=k)]
    if image_store is not None:
        hits += [(doc, score, "image") for doc, score in image_store.similarity_search_with_score(query, k=k)]
    hits.sort(key=lambda h: h[1], reverse=True)
    return hits[:k]


def content_text(doc, content_type: str) -> str:
    # Image docs store a file path as page_content (needed for CLIP embedding),
    # so use the caption instead - the only part a text-only LLM can use.
    if content_type == "image":
        return doc.metadata.get("caption") or "(image with no caption)"
    return doc.page_content


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

    results = search_all(request.query, k=request.top_k)

    if not results:
        answer = "No relevant documents found."
        history.append({"query": request.query, "answer": answer})
        return SearchResponse(answer=answer, sources=[], session_id=session_id)

    retrieved_context = "\n\n".join(
        f"[Source: {doc.metadata.get('source', 'unknown')}, page {doc.metadata.get('page', '?')}"
        f"{', image' if content_type == 'image' else ''}]\n{content_text(doc, content_type)}"
        for doc, score, content_type in results
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
Entries marked "image" are captions describing a diagram or photo, not manual prose.
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
            text=content_text(doc, content_type),
            source=doc.metadata.get("source", "unknown"),
            page=doc.metadata.get("page"),
            score=float(score),
            content_type=content_type,
            image_path=doc.metadata.get("image_path") if content_type == "image" else None,
        )
        for doc, score, content_type in results
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