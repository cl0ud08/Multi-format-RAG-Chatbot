from langchain_text_splitters import RecursiveCharacterTextSplitter


def get_pdf_splitter():
    """PDFs have headers, paragraphs, sentences — recursive split with generous overlap."""
    return RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ".", "!", "?", " ", ""]
    )


def get_docx_splitter():
    """Word docs tend to have clear paragraphs — slightly larger chunks."""
    return RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=60,
        separators=["\n\n", "\n", ". ", " ", ""]
    )


def get_csv_splitter():
    """CSV rows are already atomic units — larger chunks group several rows for context."""
    return RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=0,
        separators=["\n\n", "\n"]
    )


def get_url_splitter():
    """Web pages are noisy (nav/footer/ads) — smaller chunks isolate actual content."""
    return RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=40,
        separators=["\n\n", "\n", ". ", " ", ""]
    )


SPLITTER_MAP = {
    "pdf": get_pdf_splitter,
    "docx": get_docx_splitter,
    "csv": get_csv_splitter,
    "url": get_url_splitter,
}


def get_splitter(source_type: str):
    """Return the right splitter for a given source type."""

    fn = SPLITTER_MAP.get(source_type)
    if not fn:
        raise ValueError(f"Unknown source type: {source_type}. Choose from {list(SPLITTER_MAP)}")
    return fn()