from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)

query = "Where is the Eiffel Tower?"

print(f"\nQuery: {query}")

results_with_scores = db.similarity_search_with_score(
    query,
    k=2
)

for i, (doc, score) in enumerate(results_with_scores):
    print(f"\nResult {i+1}")
    print(f"Score: {score:.4f}")
    print(f"Text : {doc.page_content}")