from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from vector_store import load_vector_store

load_dotenv()


def build_rag_chain():
    """
    Load FAISS vector store and Gemini model.
    """

    # Load saved FAISS index
    db = load_vector_store()

    # Create retriever
    retriever = db.as_retriever(
        search_kwargs={"k": 4}
    )

    # Gemini model
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0
    )

    return retriever, llm


def log_retrieved_chunks(retriever, question):
    """
    Display chunks returned by FAISS.
    Useful for debugging retrieval quality.
    """

    docs = retriever.invoke(question)

    print("\nRetrieved Context:")
    print("=" * 60)

    for i, doc in enumerate(docs, start=1):
        print(f"\nChunk {i}")
        print(f"Page: {doc.metadata.get('page', 'N/A')}")
        print(doc.page_content[:500])
        print("-" * 60)

    return docs


def ask(retriever, llm, question):
    """
    Ask a question using RAG.
    """

    print(f"\nQuestion: {question}")
    print("-" * 60)

    # Retrieve relevant chunks
    docs = log_retrieved_chunks(
        retriever,
        question
    )

    # Build context
    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )

    # Prompt
    prompt = f"""
You are a helpful assistant.

Answer ONLY using the provided context.

If the answer is not present in the context, reply exactly:

I don't have enough information in the document to answer this.

Context:
{context}

Question:
{question}

Answer:
"""

    # Gemini response
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

    while True:

        question = input(
            "\nAsk a question (type 'exit' to quit): "
        )

        if question.lower() == "exit":
            print("Goodbye!")
            break

        ask(
            retriever,
            llm,
            question
        )

        print("\n" + "=" * 80)