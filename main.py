from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from chat_chain import build_chat_chain
from vector_store import embeddings, FAISS_PATH
from ingestion.loader_router import load_and_chunk
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
import os
import shutil
import uuid

load_dotenv()

app = FastAPI(title="RAG Chatbot API", version="1.1")

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

class UploadResponse(BaseModel):
    message: str
    filename: str
    chunks: int

class URLRequest(BaseModel):
    url: str


# --- Helper: embed chunks and save FAISS index (used by all upload routes) ---

def embed_and_save(chunks) -> int:
    """Embed chunks with HuggingFace, save FAISS index, return chunk count."""
    db = FAISS.from_documents(chunks, embeddings)
    db.save_local(FAISS_PATH)
    return len(chunks)


# --- Routes ---

@app.get("/")
def root():
    return {"status": "RAG Chatbot API is running"}


@app.post("/upload", response_model=UploadResponse)
async def upload_file(file: UploadFile = File(...)):
    """Upload a PDF, DOCX, or CSV file."""
    global chat_chain

    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in {".pdf", ".docx", ".csv"}:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {ext}. Use PDF, DOCX, or CSV."
        )

    file_id = str(uuid.uuid4())[:8]
    save_path = os.path.join(UPLOAD_DIR, f"{file_id}_{file.filename}")

    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        chunks = load_and_chunk(save_path)
        chunk_count = embed_and_save(chunks)

        chat_chain = build_chat_chain()

        return UploadResponse(
            message=f"{ext[1:].upper()} processed successfully.",
            filename=file.filename,
            chunks=chunk_count
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload/url")
async def upload_url(request: URLRequest):
    """Ingest a webpage by URL."""
    global chat_chain

    try:
        chunks = load_and_chunk(request.url)
        chunk_count = embed_and_save(chunks)

        chat_chain = build_chat_chain()

        return {
            "message": "URL ingested successfully.",
            "url": request.url,
            "chunks": chunk_count
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """Answer a question using the current vector store + Gemini."""
    global chat_chain

    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    if chat_chain is None:
        try:
            chat_chain = build_chat_chain()
        except FileNotFoundError:
            raise HTTPException(
                status_code=400,
                detail="No document uploaded yet. Please upload a file or URL first."
            )

    try:
        result = chat_chain.invoke({"question": request.question})

        sources = [
            {
                "page": doc.metadata.get("page", "N/A"),
                "source": doc.metadata.get("source", "N/A"),
                "content": doc.page_content[:200]
            }
            for doc in result.get("source_documents", [])
        ]

        return ChatResponse(answer=result["answer"], sources=sources)

    except Exception as e:
        err_str = str(e)
        if "RESOURCE_EXHAUSTED" in err_str or "429" in err_str:
            raise HTTPException(
                status_code=429,
                detail="Gemini API quota exhausted (daily or per-minute limit). Try again later."
            )
        raise HTTPException(status_code=500, detail=err_str)


@app.delete("/reset")
def reset():
    """Clear memory and reset the chain."""
    global chat_chain
    chat_chain = None
    return {"message": "Session reset successfully."}


@app.get("/health")
def health():
    return {
        "status": "ok",
        "chain_loaded": chat_chain is not None
    }