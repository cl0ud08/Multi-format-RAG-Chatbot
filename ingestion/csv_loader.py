from langchain_core.documents import Document
from ingestion.splitters import get_splitter
from ingestion.metadata import enrich_metadata, filter_short_chunks, deduplicate_chunks
import pandas as pd
import os


def load_and_chunk_csv(csv_path: str) -> list:
    print(f"Loading CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"  Rows: {len(df)} | Columns: {list(df.columns)}")

    docs = []
    for i, row in df.iterrows():
        content = "\n".join([f"{col}: {val}" for col, val in row.items()])
        docs.append(Document(
            page_content=content,
            metadata={"row": i, "source": csv_path}
        ))

    splitter = get_splitter("csv")
    chunks = splitter.split_documents(docs)

    chunks = filter_short_chunks(chunks)
    chunks = deduplicate_chunks(chunks)
    chunks = enrich_metadata(chunks, source_type="csv", extra={
        "filename": os.path.basename(csv_path),
        "columns": str(list(df.columns))
    })

    print(f"✅ CSV: {len(chunks)} final chunks")
    return chunks


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "data/sample.csv"
    chunks = load_and_chunk_csv(path)
    print(f"\nTotal chunks: {len(chunks)}")