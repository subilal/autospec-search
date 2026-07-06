import os
import requests
from qdrant_client import QdrantClient

os.environ["HF_HUB_OFFLINE"] = "1"
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore


AUDI_QDRANT_VECTOR_STORE_PATH = "./qdrant_audi_vector_store2"
AUDI_QDRANT_VECTOR_STORE_COLLECTION = "audi_spec_docs2"
AUTOSPEC_EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def call_llama(prompt: str) -> str:
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": "llama3.1", "prompt": prompt, "stream": False},
            timeout=120,
        )
        response.raise_for_status()
        return response.json()["response"]
    except requests.exceptions.ConnectionError:
        return "ERROR: Could not connect to Ollama. Is it running? (ollama serve)"
    except Exception as e:
        return f"ERROR calling Llama: {e}"


def get_vector_store():
    client = QdrantClient(path=AUDI_QDRANT_VECTOR_STORE_PATH)
    embeddings = HuggingFaceEmbeddings(model_name=AUTOSPEC_EMBEDDING_MODEL, model_kwargs={"local_files_only": True})

    return QdrantVectorStore(
        client=client,
        collection_name=AUDI_QDRANT_VECTOR_STORE_COLLECTION,
        embedding=embeddings,
    )


vector_store = get_vector_store()
# llm = ChatOpenAI(model="gpt-4o-mini")
query = input(" Type The Question : ")

results = vector_store.similarity_search_with_score(query, k=4)
print("Got Docs")

context = "\n\n".join(
    f"[Source: {doc.metadata.get('source', 'unknown')}, page {doc.metadata.get('page', '?')}]\n{doc.page_content}"
    for doc, score in results
)

prompt = f"""
You are a helpful assistant. Use ONLY the context below to answer.
Always cite the source filename and page number for any fact you use.

Context:
{context}

Question:
{query}

Answer (include citations like [source, page X]):
"""
print("Call starting")
answer = call_llama(prompt)
print("Call ending")
print(answer)


for doc, score in results:
    print("Text:")
    print(doc.page_content)
    print("\nMetadata:")
    print(doc.metadata)
    print("Score:", score)
    print("-" * 80)
