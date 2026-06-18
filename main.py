from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from chat_chain import build_chat_chain
from ingestion.loader_router import load_and_chunk
from ingestion.index_manager import (
    add_document_to_index,
    clear_index,
    list_documents,
    load_registry
)
from dotenv import load_dotenv
import os
import shutil
import uuid

load_dotenv()

app = FastAPI(title="RAG Chatbot API", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

chat_chain = None
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


# --- Models ---

class ChatRequest(BaseModel):
    question: str

class ChatResponse(BaseModel):
    answer: str
    sources: list[dict]

class URLRequest(BaseModel):
    url: str


# --- Helper ---

def refresh_chain():
    """Rebuild chat chain from latest merged index."""
    global chat_chain
    chat_chain = build_chat_chain()


# --- Routes ---

@app.get("/")
def root():
    return {"status": "RAG Chatbot API v2 running"}


@app.get("/documents")
def get_documents():
    """List all ingested documents."""
    return {
        "documents": list_documents(),
        "registry": load_registry()
    }


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Upload a PDF, DOCX, or CSV file — merges into existing index."""
    global chat_chain

    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in {".pdf", ".docx", ".csv"}:
        raise HTTPException(400, f"Unsupported type: {ext}")

    file_id = str(uuid.uuid4())[:8]
    save_path = os.path.join(UPLOAD_DIR, f"{file_id}_{file.filename}")

    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        chunks = load_and_chunk(save_path)
        registry = add_document_to_index(chunks, file.filename, ext[1:])
        refresh_chain()

        return {
            "message": f"{ext[1:].upper()} ingested successfully.",
            "filename": file.filename,
            "chunks_added": len(chunks),
            "total_docs": len(registry["documents"]),
            "total_chunks": registry["total_chunks"]
        }
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/upload/url")
async def upload_url(request: URLRequest):
    """Ingest a webpage by URL — merges into existing index."""
    global chat_chain

    try:
        chunks = load_and_chunk(request.url)
        registry = add_document_to_index(chunks, request.url, "url")
        refresh_chain()

        return {
            "message": "URL ingested successfully.",
            "url": request.url,
            "chunks_added": len(chunks),
            "total_docs": len(registry["documents"]),
            "total_chunks": registry["total_chunks"]
        }
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """Ask a question across all ingested documents."""
    global chat_chain

    if not request.question.strip():
        raise HTTPException(400, "Question cannot be empty.")

    if chat_chain is None:
        try:
            refresh_chain()
        except FileNotFoundError:
            raise HTTPException(400, "No documents uploaded yet.")

    try:
        result = chat_chain.invoke({"question": request.question})

        sources = [
            {
                "filename": doc.metadata.get("filename", doc.metadata.get("url", "N/A")),
                "source_type": doc.metadata.get("source_type", "unknown"),
                "page": doc.metadata.get("page", "N/A"),
                "chunk_index": doc.metadata.get("chunk_index", "N/A"),
                "content": doc.page_content[:200]
            }
            for doc in result.get("source_documents", [])
        ]

        return ChatResponse(answer=result["answer"], sources=sources)

    except Exception as e:
        err_str = str(e)
        if "RESOURCE_EXHAUSTED" in err_str or "429" in err_str:
            raise HTTPException(429, "Gemini API quota exhausted. Try again later.")
        raise HTTPException(500, err_str)


@app.delete("/reset")
def reset():
    """Clear everything — index, uploads, chain."""
    global chat_chain
    clear_index()
    chat_chain = None

    for f in os.listdir(UPLOAD_DIR):
        os.remove(os.path.join(UPLOAD_DIR, f))

    return {"message": "All documents and index cleared."}


@app.get("/health")
def health():
    docs = list_documents()
    return {
        "status": "ok",
        "chain_loaded": chat_chain is not None,
        "docs_count": len(docs),
        "documents": [d["name"] for d in docs]
    }