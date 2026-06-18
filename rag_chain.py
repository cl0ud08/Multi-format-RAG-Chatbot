from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from vector_store import load_vector_store
from dotenv import load_dotenv
import os
import time

load_dotenv()

if not os.getenv("GOOGLE_API_KEY"):
    raise ValueError(
        "GOOGLE_API_KEY not found in .env file. "
        "Get one at https://aistudio.google.com/app/apikey"
    )


def build_rag_chain(k: int = 4):
    """Load vector store and build the RetrievalQA chain using Gemini."""

    db = load_vector_store()
    retriever = db.as_retriever(search_kwargs={"k": k})

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0,
        max_retries=2,
    )

    prompt_template = """You are a helpful assistant. Use ONLY the context below to answer
the question. If the answer is not in the context, say "I don't have enough
information in the document to answer this."

Context:
{context}

Question: {question}

Answer:"""

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )

    return chain


def ask(chain, question: str, retries: int = 1):
    """Run a question through the chain, with backoff on rate-limit errors."""

    print(f"\nQuestion: {question}")
    print("-" * 50)

    for attempt in range(retries + 1):
        try:
            result = chain.invoke({"query": question})

            print(f"Answer: {result['result']}")
            print(f"\nSources used ({len(result['source_documents'])} chunks):")
            for i, doc in enumerate(result['source_documents']):
                print(f"  [{i+1}] Page {doc.metadata.get('page', 'N/A')}: {doc.page_content[:150]}...")

            return result

        except Exception as e:
            if "RESOURCE_EXHAUSTED" in str(e) or "429" in str(e):
                if "RetryInfo" in str(e) and attempt < retries:
                    wait_time = 35
                    print(f"⚠️  Rate limit hit. Waiting {wait_time}s before retry ({attempt+1}/{retries})...")
                    time.sleep(wait_time)
                else:
                    print("❌ Quota exhausted (daily or per-minute limit reached).")
                    print("   Wait for the quota to reset before trying again.")
                    return None
            else:
                print(f"❌ Error: {e}")
                return None

    return None


def test_retrieval_only(question: str, k: int = 4):
    """
    Check what chunks would be retrieved WITHOUT calling Gemini.
    Use this freely — it costs zero API quota (HuggingFace embeddings are local).
    """
    db = load_vector_store()
    results = db.similarity_search(question, k=k)

    print(f"\n[Retrieval test — no Gemini call] Question: {question}")
    print("-" * 50)
    for i, doc in enumerate(results):
        print(f"\nChunk {i+1} (page {doc.metadata.get('page', 'N/A')}):")
        print(f"  {doc.page_content[:200]}...")
    return results


if __name__ == "__main__":
    chain = build_rag_chain()

    # Kept short (2 questions) to conserve daily Gemini quota during testing.
    # Expand back to 5 once you've confirmed everything works.
    questions = [
        "What is the main topic of this document?",
        "What are the key points in the introduction?",
    ]

    for q in questions:
        ask(chain, q)
        print("\n" + "=" * 60)
        time.sleep(3)   # small pause between calls