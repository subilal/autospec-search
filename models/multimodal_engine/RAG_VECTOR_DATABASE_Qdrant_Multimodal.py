import io
import os
import uuid
import fitz
from PIL import Image
from typing import Any
from typing import List
from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import BlipProcessor, BlipForConditionalGeneration
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

dotenv_path = "../.env"
audiSpecDocsFolder = "../../docs/Audi/"
audi_qdrant_vector_store_path = "./qdrant_audi_multi_vector_store2"
audi_qdrant_vector_store_collection = "audi_multi_spec_docs2"
autospec_embedding_model = "all-MiniLM-L6-v2"


### Load the pdf and extract the pages
loader = DirectoryLoader(
    audiSpecDocsFolder,
    glob="**/*.pdf",
    loader_cls=PyPDFLoader
)

audi_spec_docs = loader.load()

### Split the texts from documents inot chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

audi_spec_chunks = text_splitter.split_documents(audi_spec_docs)

for i, chunk in enumerate(audi_spec_chunks):
    chunk.metadata["chunk_id"] = i
    chunk.metadata["content_type"] = "text"

print(f"Loaded {len(audi_spec_docs)} documents")
print(f"Created {len(audi_spec_chunks)} chunks")


# Creating a FOlder to save the extracted image( Used later for captioning)
audi_extracted_images_folder = os.path.join(audiSpecDocsFolder, "_extracted_images")
os.makedirs(audi_extracted_images_folder, exist_ok=True)

audi_spec_image_records: list[dict[str, Any]] = []

### Opening the PDF files and extracting images from them
pdf_paths = sorted(set(doc.metadata["source"] for doc in audi_spec_docs))

for pdf_path in pdf_paths:
    try:
        pdf_doc = fitz.open(pdf_path)
    except Exception as e:
        print(f"Skipping unreadable PDF {pdf_path}: {e}")
        continue

    for page_index in range(len(pdf_doc)):
        try:
            page = pdf_doc[page_index]
        except Exception as e:
            print(f"Skipping unreadable page {page_index} in {pdf_path}: {e}")
            continue
        try:
            image_list = page.get_images(full=True)
        except Exception as e:
            print(f"Skipping image scan on {pdf_path} page {page_index}: {e}")
            image_list = []

        if not image_list:
            continue

        for img_index, img in enumerate(image_list):
            xref = img[0]
            base_image = pdf_doc.extract_image(xref)
            image_bytes = base_image["image"]

            try:
                pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            except Exception as e:
                print(f"Skipping unreadable image on {pdf_path} page {page_index}: {e}")
                continue

            # Skip tiny images like logos and  icons
            if pil_image.width < 100 or pil_image.height < 100:
                continue

            #Assign Unique filenames and save in Folder
            image_id = f"{uuid.uuid4()}"
            image_path = os.path.join(audi_extracted_images_folder, f"{image_id}.png")
            pil_image.save(image_path)

            # Creating metadata for the image
            image_metadata = {
                "source": pdf_path,
                "page": page_index,
                "image_index": img_index,
                "content_type": "image",
                "image_id": image_id,
                "image_path": image_path,
            }

            # Register image for embedding generation
            audi_spec_image_records.append({
                "image": pil_image,
                "image_path": image_path,
                "metadata": image_metadata,
            })

    pdf_doc.close()

print(f"Extracted {len(audi_spec_image_records)} images")


### Table extraction
audi_spec_table_records: list[dict[str, Any]] = []

# open PDF files again and extract tables
for pdf_path in pdf_paths:
    pdf_doc = fitz.open(pdf_path)

    for page_index in range(len(pdf_doc)):
        page = pdf_doc[page_index]
        found_tables = page.find_tables()

        for table_index, table in enumerate(found_tables.tables):
            try:
                markdown_table = table.to_markdown()
            except Exception as e:
                print(f"Skipping unreadable table on {pdf_path} page {page_index}: {e}")
                continue

            if not markdown_table.strip():
                continue

            # Creating metadata for the table
            table_id = f"{uuid.uuid4()}"
            table_metadata = {
                "source": pdf_path,
                "page": page_index,
                "table_index": table_index,
                "content_type": "table",
                "table_id": table_id,
            }

            # Register table for embedding generation
            audi_spec_table_records.append({
                "markdown": markdown_table,
                "metadata": table_metadata,
            })

    pdf_doc.close()

print(f"Extracted {len(audi_spec_table_records)} tables")



### Image captioning
# captioning helps readability and finding images through the regular text collection.

caption_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
caption_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

for record in audi_spec_image_records:
    inputs = caption_processor(record["image"], return_tensors="pt")
    output_ids = caption_model.generate(**inputs, max_new_tokens=40)
    caption = caption_processor.decode(output_ids[0], skip_special_tokens=True)
    record["metadata"]["caption"] = caption

print("Captioned all extracted images")

### Embeddings
embeddings = HuggingFaceEmbeddings(model_name=autospec_embedding_model)

class ClipImageEmbeddings(Embeddings):
    """
    """

    def __init__(self, model_name: str = "clip-ViT-B-32"):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        inputs = []
        for text in texts:
            image = None
            if os.path.isfile(text):
                try:
                    image = Image.open(text).convert("RGB")
                except Exception:
                    image = None
            inputs.append(image if image is not None else text)

        vectors = self.model.encode(inputs, convert_to_numpy=True, show_progress_bar=True)
        return vectors.tolist()

    def embed_query(self, text: str) -> List[float]:
        vector = self.model.encode(text, convert_to_numpy=True)
        return vector.tolist()


image_embeddings = ClipImageEmbeddings()

### Vector store

### Create the location to save vector store in local drive
os.makedirs(audi_qdrant_vector_store_path, exist_ok=True)

### Qdrant Format support
audi_table_documents = [
    Document(page_content=record["markdown"], metadata=record["metadata"])
    for record in audi_spec_table_records
]

audi_caption_documents = [
    Document(page_content=record["metadata"]["caption"], metadata=record["metadata"])
    for record in audi_spec_image_records
]

audi_all_text_documents = audi_spec_chunks + audi_table_documents + audi_caption_documents

### Injecting text, table and image captions to Vector store
audi_qdrant_text_store = QdrantVectorStore.from_documents(
    documents=audi_all_text_documents,
    embedding=embeddings,
    path=audi_qdrant_vector_store_path,
    collection_name=audi_qdrant_vector_store_collection,
)
audi_qdrant_text_store.client.close()

print(f"Upserted {len(audi_all_text_documents)} text-space points "
      f"({len(audi_spec_chunks)} chunks, {len(audi_table_documents)} tables, "
      f"{len(audi_caption_documents)} image captions) "
      f"into '{audi_qdrant_vector_store_collection}'")

audi_qdrant_image_collection = f"{audi_qdrant_vector_store_collection}_images"

if audi_spec_image_records:
    audi_image_documents = [
        Document(page_content=record["image_path"], metadata=record["metadata"])
        for record in audi_spec_image_records
    ]

    audi_qdrant_image_store = QdrantVectorStore.from_documents(
        documents=audi_image_documents,
        embedding=image_embeddings,
        path=audi_qdrant_vector_store_path,
        collection_name=audi_qdrant_image_collection,
    )
    audi_qdrant_image_store.client.close()

    print(f"Upserted {len(audi_image_documents)} image points into '{audi_qdrant_image_collection}'")
else:
    print("No images found - skipped creating the image collection")
