from langchain_core.documents import Document
from datetime import datetime
import hashlib


def enrich_metadata(chunks: list[Document], source_type: str, extra: dict = {}) -> list[Document]:
    """
    Add consistent metadata to every chunk.
    Fields added:
      - source_type : pdf | docx | csv | url
      - ingested_at : ISO timestamp
      - chunk_index : position in the list
      - chunk_hash  : unique ID for deduplication
      - char_count  : length of chunk text
    """

    now = datetime.utcnow().isoformat()

    for i, chunk in enumerate(chunks):
        content_hash = hashlib.md5(chunk.page_content.encode()).hexdigest()[:10]

        chunk.metadata.update({
            "source_type": source_type,
            "ingested_at": now,
            "chunk_index": i,
            "chunk_hash": content_hash,
            "char_count": len(chunk.page_content),
            **extra
        })

    return chunks


def filter_short_chunks(chunks: list[Document], min_chars: int = 80) -> list[Document]:
    """Remove chunks that are too short to be useful."""

    before = len(chunks)
    chunks = [c for c in chunks if len(c.page_content.strip()) >= min_chars]
    after = len(chunks)

    if before != after:
        print(f"  Filtered {before - after} short chunks (< {min_chars} chars)")

    return chunks


def deduplicate_chunks(chunks: list[Document]) -> list[Document]:
    """Remove exact duplicate chunks by content hash."""

    seen = set()
    unique = []

    for chunk in chunks:
        h = hashlib.md5(chunk.page_content.encode()).hexdigest()
        if h not in seen:
            seen.add(h)
            unique.append(chunk)

    removed = len(chunks) - len(unique)
    if removed:
        print(f"  Removed {removed} duplicate chunks")

    return unique


def preview_metadata(chunks: list[Document], n: int = 3):
    """Print metadata for the first n chunks."""

    print(f"\n--- Metadata preview (first {n} chunks) ---")
    for i, chunk in enumerate(chunks[:n]):
        print(f"\nChunk {i+1}:")
        for k, v in chunk.metadata.items():
            print(f"  {k:15}: {v}")
        print(f"  {'content':15}: {chunk.page_content[:120]}...")