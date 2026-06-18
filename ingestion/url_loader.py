from langchain_community.document_loaders import WebBaseLoader
from ingestion.splitters import get_splitter
from ingestion.metadata import enrich_metadata, filter_short_chunks, deduplicate_chunks
import requests

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
}


def is_valid_url(url: str) -> bool:
    try:
        response = requests.get(url, headers=HEADERS, timeout=5, stream=True)
        return response.status_code < 400
    except Exception:
        return False


def load_and_chunk_url(url: str) -> list:
    print(f"Loading URL: {url}")

    if not is_valid_url(url):
        raise ValueError(f"URL is not reachable: {url}")

    loader = WebBaseLoader(url, header_template=HEADERS)
    docs = loader.load()

    for doc in docs:
        doc.page_content = " ".join(doc.page_content.split())

    splitter = get_splitter("url")
    chunks = splitter.split_documents(docs)

    chunks = filter_short_chunks(chunks)
    chunks = deduplicate_chunks(chunks)
    chunks = enrich_metadata(chunks, source_type="url", extra={"url": url})

    print(f"✅ URL: {len(chunks)} final chunks")
    return chunks


if __name__ == "__main__":
    chunks = load_and_chunk_url("https://en.wikipedia.org/wiki/Retrieval-augmented_generation")
    print(f"\nTotal chunks: {len(chunks)}")