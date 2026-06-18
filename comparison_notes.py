from llama_rag import build_llama_index, load_llama_index, query_llama
from chat_chain import build_chat_chain, chat
import os
import json
import time

# Kept to 2 questions to stay within daily Gemini quota
QUESTIONS = [
    "What is the main topic of this document?",
    "What are the key concepts explained?",
]


def run_langchain(questions: list) -> list:
    """Run all questions through your existing LangChain chat chain."""
    print("\n" + "=" * 60)
    print("LANGCHAIN RESULTS")
    print("=" * 60)

    chain = build_chat_chain()
    results = []

    for q in questions:
        start = time.time()
        result = chat(chain, q)
        elapsed = round(time.time() - start, 2)

        if result:
            results.append({
                "question": q,
                "answer": result["answer"],
                "num_sources": len(result["source_documents"]),
                "time_sec": elapsed
            })
        time.sleep(3)

    return results


def run_llamaindex(questions: list) -> list:
    """Run all questions through LlamaIndex."""
    print("\n" + "=" * 60)
    print("LLAMAINDEX RESULTS")
    print("=" * 60)

    if os.path.exists("llama_index_storage"):
        index = load_llama_index()
    else:
        index = build_llama_index("data")

    results = []

    for q in questions:
        start = time.time()
        result = query_llama(index, q)
        elapsed = round(time.time() - start, 2)

        print(f"\nQ: {q}")
        print(f"A: {result['answer'][:300]}")
        print(f"   Sources: {len(result['sources'])} | Time: {elapsed}s")

        results.append({
            "question": q,
            "answer": result["answer"],
            "num_sources": len(result["sources"]),
            "time_sec": elapsed
        })
        time.sleep(3)

    return results


def compare(lc_results: list, ll_results: list):
    """Print side-by-side comparison table."""

    print("\n" + "=" * 60)
    print("SIDE-BY-SIDE COMPARISON")
    print("=" * 60)

    for i, q in enumerate(QUESTIONS):
        if i >= len(lc_results) or i >= len(ll_results):
            continue
        lc = lc_results[i]
        ll = ll_results[i]

        print(f"\nQ{i+1}: {q}")
        print(f"\n  LangChain  ({lc['time_sec']}s | {lc['num_sources']} sources):")
        print(f"  {lc['answer'][:250]}")
        print(f"\n  LlamaIndex ({ll['time_sec']}s | {ll['num_sources']} sources):")
        print(f"  {ll['answer'][:250]}")
        print("-" * 60)


def save_comparison(lc_results: list, ll_results: list):
    """Save full comparison to JSON."""

    output = {
        "questions": QUESTIONS,
        "langchain": lc_results,
        "llamaindex": ll_results,
    }

    with open("comparison_notes.json", "w") as f:
        json.dump(output, f, indent=2)

    print("\n✅ Full comparison saved to comparison_notes.json")


if __name__ == "__main__":
    lc_results = run_langchain(QUESTIONS)
    ll_results = run_llamaindex(QUESTIONS)
    compare(lc_results, ll_results)
    save_comparison(lc_results, ll_results)