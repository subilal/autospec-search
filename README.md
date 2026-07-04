# AutoSpec Search

A fully local, privacy-preserving search tool for engineering teams that need to
find details buried inside long product specification PDFs like text, tables,
*and* diagrams/images, without manually scrolling through hundreds of pages
per product.

Nothing leaves the machine it runs on. Document parsing, embeddings, image
captioning, and the language model that answers your question all run
locally,  there is no external API call, no cloud upload, and no telemetry.
This makes it suitable for internal specs, proprietary designs, or any
documentation that can't be sent to a third-party service.

## Why this exists

Engineers often need to answer a specific question  "what's the torque
spec for this component," "what does this connector diagram look like"
that's buried somewhere inside a large PDF manual. Ctrl+F only works if you
already know the exact wording, and it can't search diagrams at all. This
project builds a searchable index over your spec documents (text, tables,
and images) and lets you ask questions in plain language, with the answer
grounded in and cited to  the actual source pages.

## How it works

**1. Ingestion** (`models/Language Engine/RAG_VECTOR_DATABASE_Sentence_transformer.py`) run this once per document
set, or whenever specs are added/updated:
- Loads all PDFs in a folder and splits the text into chunks.
- Extracts embedded images and tables directly from each PDF page (tables
  are converted to Markdown so their row/column structure survives).
- Captions every image locally with BLIP, so images are findable by
  plain-text search even before anyone looks at the actual picture.
- Embeds text/tables/captions with a local sentence-transformer model, and
  embeds the raw images separately with CLIP.
- Stores everything in a local Qdrant vector database (on disk, no server
  required) as two collections: one for text/tables/captions, one for images.

**2. Search API** (`api/Language Engine/main.py`) a small FastAPI service that:
- Takes a question, searches both collections, and merges the results by
  relevance (not a fixed split between text and images).
- Builds a prompt from the retrieved context (image hits contribute their
  caption, since the LLM here is text-only) and sends it to a locally running
  LLM via [Ollama](https://ollama.com).
- Returns an answer with citations, plus the raw source chunks it used, and
  keeps short-term conversation history per session so follow-up questions
  ("what about the other one?") work.

## Requirements

- Python 3.10+
- [Ollama](https://ollama.com) installed and running locally, with a model
  pulled (default expected: `llama3.1` run `ollama pull llama3.1`)
- Enough disk space for the local embedding/captioning models (a few GB,
  downloaded once and cached locally)

Install Python dependencies:

```bash
pip install -r requirements.txt
```

## Setup

1. **Put your spec PDFs in a folder** and set `audiSpecDocsFolder` at the top
   of `RAG_VECTOR_DATABASE_Sentence_transformer.py` or `RAG_VECTOR_DATABASE_Qdrant_Multimodal.py` to that path (also set
   `audi_qdrant_vector_store_path`, `audi_qdrant_vector_store_collection`,
   and `autospec_embedding_model` if you want non-default values).

2. **Run ingestion** (only needed once, or again when documents change):
   ```bash
   python RAG_VECTOR_DATABASE_Sentence_transformer.py
   ## If Multimodal Run
   python RAG_VECTOR_DATABASE_Qdrant_Multimodal.py
   ```
   This builds the local Qdrant store on disk and prints a summary of how
   many text chunks, tables, and images were indexed, plus a quick
   similarity-search sanity check at the end.

3. **Start Ollama** (if it isn't already running) and make sure the model
   referenced by `OLLAMA_MODEL` is pulled:
   ```bash
   ollama serve
   ollama pull llama3.1
   ```

4. **Run the search API**:
   ```bash
   python api/Language Engine/main.py
   ## If Multimodal Run
   python api/MultiModal Engine/main.py
   
   ```
   This starts the FastAPI service on `http://localhost:8000`.

## Using it

**Health check**
```bash
curl http://localhost:8000/health
```

**Ask a question**
```bash
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the torque spec for the rear suspension bolt?", "top_k": 4}'
```

Response includes the answer, the `session_id` (reuse it on your next call to
continue the conversation), and the source chunks used (each tagged
`content_type: "text" | "table" | "image"`, with an `image_path` when
relevant).

**Clear a conversation**
```bash
curl -X POST "http://localhost:8000/new_chat?session_id=<id>"
```

## Configuration reference

| Variable (in `main.py` / ingestion script) | Purpose |
|---|---|
| `audiSpecDocsFolder` | Folder containing the source PDFs |
| `audi_qdrant_vector_store_path` | Where the local Qdrant database is stored on disk |
| `audi_qdrant_vector_store_collection` | Name of the text/table/caption collection (the image collection is this name + `_images`) |
| `autospec_embedding_model` | Local text embedding model (default: `all-MiniLM-L6-v2`) |
| `OLLAMA_URL` / `OLLAMA_MODEL` | Where Ollama is running and which local model to use |

## Privacy notes

- `HF_HUB_OFFLINE=1` is set before any Hugging Face imports, so the app
  refuses to reach out to the internet for models at runtime everything
  must already be cached locally (this happens naturally the first time you
  run ingestion, since that step downloads each model once).
- The vector store is a local, embedded Qdrant instance (a folder on disk),
  not a hosted database.
- The LLM call goes to a local Ollama server, not a cloud API no document
  content is ever transmitted off the machine.



