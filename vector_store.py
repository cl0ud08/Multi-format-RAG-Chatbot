from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from ingestion.pdf_loader import load_and_chunk_pdf
from dotenv import load_dotenv
import os

load_dotenv()

FAISS_PATH = "faiss_index"

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


def build_vector_store(pdf_path: str):
    """Chunk a PDF and embed into a FAISS index, saved to disk."""

    print("Loading and chunking PDF...")
    chunks = load_and_chunk_pdf(pdf_path)

    print(f"Embedding {len(chunks)} chunks... (this may take a moment)")
    db = FAISS.from_documents(chunks, embeddings)

    db.save_local(FAISS_PATH)
    print(f"✅ Vector store saved to {FAISS_PATH}/")
    return db


def load_vector_store():
    """Load existing FAISS index from disk."""

    if not os.path.exists(FAISS_PATH):
        raise FileNotFoundError(f"No FAISS index found at {FAISS_PATH}. Run build first.")

    db = FAISS.load_local(FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    print(f"✅ Vector store loaded from {FAISS_PATH}/")
    return db


def query_vector_store(db, question: str, k: int = 4):
    """Retrieve top-k most relevant chunks for a question."""

    results = db.similarity_search_with_score(question, k=k)
    print(f"\nQuery: {question}")
    print("-" * 50)
    for i, (doc, score) in enumerate(results):
        print(f"\nResult {i+1} (score: {score:.4f})")
        print(f"Page   : {doc.metadata.get('page', 'N/A')}")
        print(f"Content: {doc.page_content[:300]}...")
    return results


if __name__ == "__main__":
    PDF_PATH = "data/sample.pdf"

    # Build and save the index
    db = build_vector_store(PDF_PATH)

    # Reload it from disk (proves persistence works)
    print("\n--- Reloading index from disk ---")
    db = load_vector_store()

    # Test with 3 questions
    test_questions = [
        "What is the main topic of this document?",
        "What are the key concepts explained?",
        "Give me a summary of the introduction."
    ]

    for q in test_questions:
        query_vector_store(db, q)
        print("\n" + "=" * 60)