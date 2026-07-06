from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
from langchain_huggingface import HuggingFaceEmbeddings

dotenv_path = "../.env"

audi_spec_docs_folder  = "../docs/Audi/"

audi_qdrant_vector_store_path = "./qdrant_audi_vector_store2"
audi_qdrant_vector_store_collection = "audi_spec_docs2"
autospec_embedding_model = "all-MiniLM-L6-v2"

###
### Loc
###
loader = DirectoryLoader(
    audi_spec_docs_folder ,
    glob = "**/*.pdf",   # loads all files recursively
    loader_cls = PyPDFLoader
)

audi_spec_docs = loader.load()

###
### Chunking document
###

text_splitter = RecursiveCharacterTextSplitter(
                            chunk_size = 500,
                            chunk_overlap=100
)

audi_spec_chunks = text_splitter.split_documents(audi_spec_docs)

for i, chunk in enumerate(audi_spec_chunks):
    chunk.metadata["chunk_id"] = i

print(f"Loaded {len(audi_spec_docs)} documents")
print(f"Created {len(audi_spec_chunks)} chunks")

###
### Embeddings
###

embeddings = HuggingFaceEmbeddings(model_name=autospec_embedding_model)

qdrant_audi_vector_store =  QdrantVectorStore.from_documents(
    documents=audi_spec_chunks,
    embedding=embeddings,
    path=audi_qdrant_vector_store_path,
    collection_name=audi_qdrant_vector_store_collection,
)
###
### Test a query for vector score
###

query = "What Python version is required?"

print(query)

results = qdrant_audi_vector_store.similarity_search(
    query,
    k=3
)

for doc in results:
    print(doc.page_content)


print("the end")



