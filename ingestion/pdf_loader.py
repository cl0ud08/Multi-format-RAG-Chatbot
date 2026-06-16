from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import json
import os

def load_and_chunk_pdf(pdf_path: str, chunk_size: int = 500, chunk_overlap: int = 50):
    """Load a PDF and split it into chunks."""

    # Step 1: Load the PDF
    print(f"Loading PDF: {pdf_path}")
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    print(f"✅ Loaded {len(pages)} pages")

    # Step 2: Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""]  # tries these in order
    )
    chunks = splitter.split_documents(pages)
    print(f"✅ Split into {len(chunks)} chunks")

    return chunks


def preview_chunks(chunks, n=5):
    """Print the first n chunks."""
    print(f"\n--- First {n} chunks ---")
    for i, chunk in enumerate(chunks[:n]):
        print(f"\nChunk {i+1}:")
        print(f"  Content : {chunk.page_content[:200]}...")
        print(f"  Metadata: {chunk.metadata}")


def save_chunks_to_json(chunks, output_path: str = "chunks_output.json"):
    """Save chunks to JSON for inspection."""
    data = [
        {
            "chunk_index": i,
            "content": chunk.page_content,
            "metadata": chunk.metadata
        }
        for i, chunk in enumerate(chunks)
    ]
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"\n✅ Saved {len(chunks)} chunks to {output_path}")


if __name__ == "__main__":
    PDF_PATH = "data/sample.pdf"

    if not os.path.exists(PDF_PATH):
        print(f"❌ PDF not found at {PDF_PATH}. Add a PDF file first.")
        exit(1)

    chunks = load_and_chunk_pdf(PDF_PATH)
    preview_chunks(chunks, n=5)
    save_chunks_to_json(chunks)