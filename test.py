from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from fastapi import FastAPI
from dotenv import load_dotenv
import os

load_dotenv()

print("✅ All imports successful")
print("API Key loaded:", bool(os.getenv("OPENAI_API_KEY")))