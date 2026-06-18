from langchain_classic.chains import ConversationalRetrievalChain
from langchain_classic.memory import ConversationBufferWindowMemory
from langchain_core.prompts import (
    PromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate
)
from langchain_google_genai import ChatGoogleGenerativeAI
from ingestion.index_manager import load_index
from dotenv import load_dotenv
import os
import time

load_dotenv()

if not os.getenv("GOOGLE_API_KEY"):
    raise ValueError(
        "GOOGLE_API_KEY not found in .env file. "
        "Get one at https://aistudio.google.com/app/apikey"
    )


def build_chat_chain(k: int = 4, memory_window: int = 3):
    """Build a conversational RAG chain with memory."""

    # Load retriever
    db = load_index()
    retriever = db.as_retriever(search_kwargs={"k": k})

    # Gemini LLM
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0,
        max_retries=2,
    )

    # Memory — keeps last `memory_window` exchanges only
    memory = ConversationBufferWindowMemory(
        k=memory_window,
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )

    # System prompt — controls personality + grounding behavior
    system_template = """You are a helpful assistant with access to one or more documents.
Answer using ONLY the context provided. When answering:
1. Cite which document the information comes from using its filename or URL if relevant.
2. If multiple documents are relevant, combine their information clearly.
3. If the answer is not in any document, say "I don't have that information in the document."
4. Never make up facts.

Context:
{context}"""

    human_template = """Previous conversation:
{chat_history}

New question: {question}
Answer:"""

    messages = [
        SystemMessagePromptTemplate.from_template(system_template),
        HumanMessagePromptTemplate.from_template(human_template)
    ]
    qa_prompt = ChatPromptTemplate.from_messages(messages)

    # Conversational chain
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": qa_prompt},
        verbose=False
    )

    return chain


def chat(chain, question: str, retries: int = 1):
    """Send a message and print the response, with quota-aware retry."""

    print(f"\nYou: {question}")
    print("-" * 50)

    for attempt in range(retries + 1):
        try:
            result = chain.invoke({"question": question})

            print(f"Bot: {result['answer']}")
            print(f"\n[Sources: {len(result['source_documents'])} chunks used]")
            for i, doc in enumerate(result['source_documents']):
                print(f"  [{i+1}] Page {doc.metadata.get('page', 'N/A')}: {doc.page_content[:120]}...")

            return result

        except Exception as e:
            if "RESOURCE_EXHAUSTED" in str(e) or "429" in str(e):
                if attempt < retries:
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


def print_chat_history(chain):
    """Print the full memory buffer."""

    print("\n--- Chat History in Memory ---")
    messages = chain.memory.chat_memory.messages
    if not messages:
        print("  (empty)")
    for msg in messages:
        role = "You" if msg.type == "human" else "Bot"
        print(f"  {role}: {msg.content[:120]}...")


if __name__ == "__main__":
    chain = build_chat_chain()

    print("RAG Chatbot with Memory — type your questions below.")
    print("=" * 60)

    # Kept to 3 questions to conserve daily Gemini quota.
    # Each question here uses ~2 API calls (reformulation + answer).
    conversation = [
        "What is the main topic of this document?",
        "Can you elaborate on that?",              # tests memory — 'that' refers to previous answer
        "Is there anything about neural networks?", # tests fallback if not in doc
    ]

    for question in conversation:
        chat(chain, question)
        print()
        time.sleep(3)   # avoid tripping per-minute rate limit

    # Print full history after the conversation
    print_chat_history(chain)