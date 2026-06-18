from langchain_community.vectorstores import FAISS
from vector_store import embeddings, FAISS_PATH
from datetime import datetime
import os
import json
import shutil

REGISTRY_PATH = os.path.join(FAISS_PATH, "registry.json")


def load_registry() -> dict:
    """Load the document registry (tracks what's been ingested)."""

    if not os.path.exists(REGISTRY_PATH):
        return {"documents": [], "total_chunks": 0}

    with open(REGISTRY_PATH, "r") as f:
        return json.load(f)


def save_registry(registry: dict):
    """Save the document registry to disk."""

    os.makedirs(FAISS_PATH, exist_ok=True)
    with open(REGISTRY_PATH, "w") as f:
        json.dump(registry, f, indent=2)


def add_document_to_index(chunks: list, doc_name: str, source_type: str) -> dict:
    """
    Embed new chunks and merge into the existing FAISS index.
    Creates a new index if none exists yet.
    """

    print(f"Embedding {len(chunks)} chunks for: {doc_name}")

    new_db = FAISS.from_documents(chunks, embeddings)

    index_file = os.path.join(FAISS_PATH, "index.faiss")

    if os.path.exists(index_file):
        print("  Merging with existing index...")
        existing_db = FAISS.load_local(
            FAISS_PATH, embeddings, allow_dangerous_deserialization=True
        )
        existing_db.merge_from(new_db)
        existing_db.save_local(FAISS_PATH)
        print("  ✅ Merged successfully")
    else:
        new_db.save_local(FAISS_PATH)
        print("  ✅ New index created")

    registry = load_registry()
    registry["documents"].append({
        "name": doc_name,
        "source_type": source_type,
        "chunks": len(chunks),
        "ingested_at": datetime.utcnow().isoformat()
    })
    registry["total_chunks"] = sum(d["chunks"] for d in registry["documents"])
    save_registry(registry)

    return registry


def load_index() -> FAISS:
    """Load the merged FAISS index."""

    index_file = os.path.join(FAISS_PATH, "index.faiss")
    if not os.path.exists(index_file):
        raise FileNotFoundError("No index found. Upload a document first.")

    return FAISS.load_local(
        FAISS_PATH, embeddings, allow_dangerous_deserialization=True
    )


def clear_index():
    """Delete the index and registry — full reset."""

    if os.path.exists(FAISS_PATH):
        shutil.rmtree(FAISS_PATH)
    print("✅ Index cleared")


def list_documents() -> list:
    """Return list of all ingested documents."""

    return load_registry().get("documents", [])