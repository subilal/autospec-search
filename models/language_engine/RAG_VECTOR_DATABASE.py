import os
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
dotenv_path = "../.env"

audiSpecDocsFolder = "../docs/Audi/"

audi_qdrant_vector_store_path = "./qdrant_audi_vector_store"
audi_qdrant_vector_store_collection = "audi_spec_docs"

###
### Load access key
###


load_dotenv(dotenv_path = dotenv_path)

###
### Loc
###
loader = DirectoryLoader(
    audiSpecDocsFolder,
    glob="**/*.pdf",   # loads all files recursively
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

print(f"Loaded {len(audi_spec_docs)} documents")
print(f"Created {len(audi_spec_chunks)} chunks")

###
### Embeddings
###


#client = QdrantClient(path="./qdrant_audi_vector_store2")
qdrant_audi_vector_store =  QdrantVectorStore.from_documents(
    documents=audi_spec_chunks,
    embedding=OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY")),
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


