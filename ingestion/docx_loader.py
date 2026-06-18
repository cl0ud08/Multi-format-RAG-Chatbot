from langchain_community.document_loaders import Docx2txtLoader
from ingestion.splitters import get_splitter
from ingestion.metadata import enrich_metadata, filter_short_chunks, deduplicate_chunks
import os


def load_and_chunk_docx(docx_path: str) -> list:
    print(f"Loading DOCX: {docx_path}")
    loader = Docx2txtLoader(docx_path)
    docs = loader.load()

    splitter = get_splitter("docx")
    chunks = splitter.split_documents(docs)

    chunks = filter_short_chunks(chunks)
    chunks = deduplicate_chunks(chunks)
    chunks = enrich_metadata(chunks, source_type="docx", extra={
        "filename": os.path.basename(docx_path),
        "filepath": docx_path
    })

    print(f"✅ DOCX: {len(chunks)} final chunks")
    return chunks


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "data/sample.docx"
    chunks = load_and_chunk_docx(path)
    print(f"\nTotal chunks: {len(chunks)}")