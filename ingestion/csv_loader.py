from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
import pandas as pd


def load_and_chunk_csv(csv_path: str, chunk_size: int = 500, chunk_overlap: int = 50):
    """Load a CSV file — each row becomes a document."""

    print(f"Loading CSV: {csv_path}")

    df = pd.read_csv(csv_path)
    print(f"  Columns: {list(df.columns)}")
    print(f"  Rows   : {len(df)}")

    docs = []
    for i, row in df.iterrows():
        content = "\n".join([f"{col}: {val}" for col, val in row.items()])
        docs.append(Document(
            page_content=content,
            metadata={"source": csv_path, "row": i}
        ))

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = splitter.split_documents(docs)
    print(f"✅ CSV loaded: {len(chunks)} chunks from {len(df)} rows")
    return chunks


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "data/sample.csv"
    chunks = load_and_chunk_csv(path)
    for i, c in enumerate(chunks[:3]):
        print(f"\nChunk {i+1}: {c.page_content[:200]}...")