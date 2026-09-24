import os
import glob
import chromadb
from chromadb.utils import embedding_functions

PERSIST_DIRECTORY = "chromadb_store"
COLLECTION_NAME = "zepto_policies"
DOCS_DIR = "docs"

def ingest_documents():
    print("Initializing embedding function with all-MiniLM-L6-v2...")
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    
    client = chromadb.PersistentClient(path=PERSIST_DIRECTORY)
    
    # Reset collection if exists
    try:
        client.delete_collection(name=COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"}
    )

    doc_files = sorted(glob.glob(os.path.join(DOCS_DIR, "doc_*.txt")))
    if not doc_files:
        raise FileNotFoundError(f"No document files found in {DOCS_DIR}/")

    documents = []
    metadatas = []
    ids = []

    for file_path in doc_files:
        file_name = os.path.basename(file_path)
        doc_id = os.path.splitext(file_name)[0]  # e.g., 'doc_01'
        
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read().strip()

        documents.append(content)
        metadatas.append({"source": doc_id, "filename": file_name})
        ids.append(f"{doc_id}_chunk_0")

    collection.add(
        documents=documents,
        metadatas=metadatas,
        ids=ids
    )

    print(f"Successfully ingested {len(documents)} documents into ChromaDB collection '{COLLECTION_NAME}'.")

if __name__ == "__main__":
    ingest_documents()
