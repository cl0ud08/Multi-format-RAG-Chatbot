from ingestion.pdf_loader import load_and_chunk_pdf
from ingestion.docx_loader import load_and_chunk_docx
from ingestion.csv_loader import load_and_chunk_csv
from ingestion.url_loader import load_and_chunk_url
import os

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".csv"}


def load_and_chunk(source: str) -> list:
    """
    Route to the correct loader based on source type.
    Accepts: file path (PDF/DOCX/CSV) or a URL string.
    """

    if source.startswith("http://") or source.startswith("https://"):
        return load_and_chunk_url(source)

    if not os.path.exists(source):
        raise FileNotFoundError(f"File not found: {source}")

    ext = os.path.splitext(source)[1].lower()

    if ext == ".pdf":
        return load_and_chunk_pdf(source)
    elif ext == ".docx":
        return load_and_chunk_docx(source)
    elif ext == ".csv":
        return load_and_chunk_csv(source)
    else:
        raise ValueError(f"Unsupported format: {ext}. Supported: {SUPPORTED_EXTENSIONS}")


if __name__ == "__main__":
    import sys
    source = sys.argv[1] if len(sys.argv) > 1 else "data/sample.pdf"
    chunks = load_and_chunk(source)
    print(f"\nTotal chunks: {len(chunks)}")
    print(f"First chunk : {chunks[0].page_content[:300]}")