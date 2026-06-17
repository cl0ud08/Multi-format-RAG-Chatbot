from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from vector_store import load_vector_store

load_dotenv()


def build_rag_chain(k: int = 4):
    """
    Load FAISS vector store and Gemini model.
    """

    # Load FAISS index
    db = load_vector_store()

    # Create retriever
    retriever = db.as_retriever(
    search_kwargs={"k": k}
)

    # Gemini LLM
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0
    )

    return retriever, llm


def ask(retriever, llm, question: str):
    """
    Retrieve relevant chunks and ask Gemini.
    """

    print(f"\nQuestion: {question}")
    print("-" * 60)

    # Retrieve relevant documents
    docs = retriever.invoke(question)

    # Debug: Show retrieved chunks
    print("\nRetrieved Context:")
    print("=" * 60)

    for i, doc in enumerate(docs):
        print(f"\nChunk {i+1}")
        print(f"Page: {doc.metadata.get('page', 'N/A')}")
        print(doc.page_content[:500])
        print("-" * 60)

    # Build context
    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )

    # Prompt
    prompt = f"""
You are a helpful assistant.

Use ONLY the context below to answer the question.

If the answer is not present in the context, respond exactly:

I don't have enough information in the document to answer this.

Context:
{context}

Question:
{question}

Answer:
"""

    # Generate answer
    response = llm.invoke(prompt)

    print("\nAnswer:")
    print(response.content)

    print("\nSources Used:")
    for i, doc in enumerate(docs, start=1):
        print(
            f"[{i}] Page {doc.metadata.get('page', 'N/A')}"
        )

    return response


if __name__ == "__main__":

    retriever, llm = build_rag_chain()

    questions = [
    "What is an embedding?",
    "Why do we use chunking?",
    "What are the three steps of RAG?",
    "What metadata does each chunk contain?",
    "Why persist the FAISS index?"
]

    for q in questions:
        ask(retriever, llm, q)
        print("\n" + "=" * 80)