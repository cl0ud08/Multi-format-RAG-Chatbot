from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
    Settings
)
from llama_index.llms.gemini import Gemini
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SentenceSplitter
from dotenv import load_dotenv
import os

load_dotenv()

if not os.getenv("GOOGLE_API_KEY"):
    raise ValueError("GOOGLE_API_KEY not found in .env file.")

LLAMA_INDEX_PATH = "llama_index_storage"

# Configure global settings — same Gemini model + same HuggingFace embedding model as your LangChain pipeline
Settings.llm = Gemini(model="models/gemini-2.5-flash", temperature=0)
Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
Settings.node_parser = SentenceSplitter(chunk_size=500, chunk_overlap=50)


def build_llama_index(data_dir: str = "data") -> VectorStoreIndex:
    """Load all documents from a folder and build LlamaIndex."""

    print(f"Loading documents from: {data_dir}/")
    reader = SimpleDirectoryReader(data_dir)
    documents = reader.load_data()
    print(f"  Loaded {len(documents)} document pages")

    print("Building LlamaIndex vector store...")
    index = VectorStoreIndex.from_documents(documents, show_progress=True)

    index.storage_context.persist(persist_dir=LLAMA_INDEX_PATH)
    print(f"✅ LlamaIndex saved to {LLAMA_INDEX_PATH}/")

    return index


def load_llama_index() -> VectorStoreIndex:
    """Load persisted LlamaIndex from disk."""

    if not os.path.exists(LLAMA_INDEX_PATH):
        raise FileNotFoundError("No LlamaIndex found. Run build first.")

    storage_context = StorageContext.from_defaults(persist_dir=LLAMA_INDEX_PATH)
    index = load_index_from_storage(storage_context)
    print(f"✅ LlamaIndex loaded from {LLAMA_INDEX_PATH}/")
    return index


def query_llama(index: VectorStoreIndex, question: str, k: int = 4) -> dict:
    """Query LlamaIndex and return answer + sources."""

    query_engine = index.as_query_engine(similarity_top_k=k)
    response = query_engine.query(question)

    sources = []
    for node in response.source_nodes:
        sources.append({
            "score": round(node.score, 4) if node.score else "N/A",
            "file": node.metadata.get("file_name", "N/A"),
            "content": node.text[:200]
        })

    return {
        "answer": str(response),
        "sources": sources
    }


if __name__ == "__main__":
    if os.path.exists(LLAMA_INDEX_PATH):
        print("Found existing LlamaIndex — loading...")
        index = load_llama_index()
    else:
        index = build_llama_index("data")

    result = query_llama(index, "What is the main topic of this document?")
    print(f"\nAnswer: {result['answer']}")
    print(f"Sources: {len(result['sources'])} chunks used")