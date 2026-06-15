from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

sentences = [
    "The Eiffel Tower is located in Paris, France.",
    "Python is a popular programming language for data science.",
    "The Amazon rainforest produces 20% of the world's oxygen.",
    "Neural networks are inspired by the human brain.",
    "The speed of light is approximately 299,792 km per second."
]

docs = [
    Document(page_content=s, metadata={"index": i})
    for i, s in enumerate(sentences)
]

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("Embedding sentences...")

db = FAISS.from_documents(
    docs,
    embeddings
)

db.save_local("faiss_index")

print("✅ FAISS index saved!")