from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import requests

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
}


def is_valid_url(url: str) -> bool:
    try:
        # GET instead of HEAD — many sites (including Wikipedia) block HEAD requests
        # stream=True avoids downloading the full page just to check reachability
        response = requests.get(url, headers=HEADERS, timeout=5, stream=True)
        return response.status_code < 400
    except Exception:
        return False


def load_and_chunk_url(url: str, chunk_size: int = 500, chunk_overlap: int = 50):
    """Load a webpage and split into chunks."""

    print(f"Loading URL: {url}")

    if not is_valid_url(url):
        raise ValueError(f"URL is not reachable: {url}")

    loader = WebBaseLoader(url, header_template=HEADERS)
    docs = loader.load()

    for doc in docs:
        doc.page_content = " ".join(doc.page_content.split())
        doc.metadata["source"] = url

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = splitter.split_documents(docs)
    print(f"✅ URL loaded: {len(chunks)} chunks")
    return chunks


if __name__ == "__main__":
    chunks = load_and_chunk_url("https://en.wikipedia.org/wiki/Retrieval-augmented_generation")
    for i, c in enumerate(chunks[:3]):
        print(f"\nChunk {i+1}: {c.page_content[:200]}...")