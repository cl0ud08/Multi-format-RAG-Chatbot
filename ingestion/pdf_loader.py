from langchain_community.document_loaders import PyPDFLoader
from ingestion.splitters import get_splitter
from ingestion.metadata import enrich_metadata, filter_short_chunks, deduplicate_chunks
import os


def load_and_chunk_pdf(pdf_path: str) -> list:
    print(f"Loading PDF: {pdf_path}")
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    print(f"  Pages loaded: {len(pages)}")

    splitter = get_splitter("pdf")
    chunks = splitter.split_documents(pages)

    chunks = filter_short_chunks(chunks)
    chunks = deduplicate_chunks(chunks)
    chunks = enrich_metadata(chunks, source_type="pdf", extra={
        "filename": os.path.basename(pdf_path),
        "filepath": pdf_path
    })

    print(f"✅ PDF: {len(chunks)} final chunks")
    return chunks


if __name__ == "__main__":
    chunks = load_and_chunk_pdf("data/sample.pdf")
    print(f"\nTotal chunks: {len(chunks)}")