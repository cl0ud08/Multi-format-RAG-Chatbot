from langchain_community.document_loaders import Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_and_chunk_docx(docx_path: str, chunk_size: int = 500, chunk_overlap: int = 50):
    """Load a Word document and split into chunks."""

    print(f"Loading DOCX: {docx_path}")

    loader = Docx2txtLoader(docx_path)
    docs = loader.load()

    for doc in docs:
        doc.metadata["source"] = docx_path

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = splitter.split_documents(docs)
    print(f"✅ DOCX loaded: {len(chunks)} chunks")
    return chunks


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "data/sample.docx"
    chunks = load_and_chunk_docx(path)
    for i, c in enumerate(chunks[:3]):
        print(f"\nChunk {i+1}: {c.page_content[:200]}...")