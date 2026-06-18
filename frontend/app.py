import streamlit as st
import requests

API_URL = "http://localhost:8000"

st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Modern Custom CSS ---
st.markdown("""
<style>
    /* Clean up top padding */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Modern Source Cards that adapt to Dark/Light mode */
    .source-card {
        background-color: rgba(128, 128, 128, 0.05);
        border: 1px solid rgba(128, 128, 128, 0.2);
        border-left: 4px solid #4CAF50;
        border-radius: 6px;
        padding: 12px 16px;
        margin: 8px 0;
        font-size: 0.9em;
        line-height: 1.4;
        transition: all 0.2s ease;
    }
    .source-card:hover {
        background-color: rgba(128, 128, 128, 0.1);
        border-color: rgba(128, 128, 128, 0.3);
    }
    .source-title {
        font-weight: 600;
        color: #4CAF50;
        margin-bottom: 4px;
    }
    .source-content {
        opacity: 0.85;
    }
    
    /* Clean up default Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# --- Session state ---
if "messages" not in st.session_state:
    st.session_state.messages = []


# --- Helper functions ---
def check_api():
    try:
        r = requests.get(f"{API_URL}/health", timeout=3)
        return r.status_code == 200, r.json()
    except Exception:
        return False, {}

def upload_file(file) -> dict:
    files = {"file": (file.name, file.getvalue(), file.type)}
    r = requests.post(f"{API_URL}/upload", files=files, timeout=120)
    r.raise_for_status()
    return r.json()

def upload_url(url: str) -> dict:
    r = requests.post(f"{API_URL}/upload/url", json={"url": url}, timeout=60)
    r.raise_for_status()
    return r.json()

def ask_question(question: str) -> dict:
    r = requests.post(f"{API_URL}/chat", json={"question": question}, timeout=60)
    if r.status_code == 429:
        return {"answer": "⚠️ Gemini quota exhausted for now. Please try again later.", "sources": []}
    r.raise_for_status()
    return r.json()

def reset_session():
    try:
        requests.delete(f"{API_URL}/reset")
    except Exception:
        pass
    st.session_state.messages = []


def fetch_documents() -> list:
    try:
        r = requests.get(f"{API_URL}/documents", timeout=5)
        return r.json().get("documents", [])
    except Exception:
        return []


# --- Sidebar ---
with st.sidebar:
    st.markdown("## 🤖 RAG Knowledge Base")
    
    is_up, health = check_api()
    if is_up:
        docs_count = health.get('docs_count', 0)
        st.caption(f"🟢 API Online • {docs_count} documents loaded")
    else:
        st.error("❌ API Offline")
        st.caption("Start your backend:")
        st.code("uvicorn main:app --reload")
        st.stop()

    st.divider()

    # --- Upload Section ---
    st.markdown("#### 📄 Add Documents")
    
    uploaded_file = st.file_uploader(
        "Upload files",
        type=["pdf", "docx", "csv"],
        help="Supported formats: PDF, Word, CSV",
        label_visibility="collapsed"
    )

    if uploaded_file:
        if st.button("Upload File", type="primary", use_container_width=True):
            with st.spinner(f"Ingesting {uploaded_file.name}..."):
                try:
                    result = upload_file(uploaded_file)
                    st.toast(f"✅ Added {result['chunks_added']} chunks from {result['filename']}")
                except Exception as e:
                    st.error(f"Upload failed: {e}")

    url_input = st.text_input("Or paste a URL", placeholder="https://example.com/article")
    if url_input and st.button("Ingest URL", use_container_width=True):
        with st.spinner("Extracting webpage..."):
            try:
                result = upload_url(url_input)
                st.toast(f"✅ Ingested {result['chunks_added']} chunks from URL")
            except Exception as e:
                st.error(f"URL failed: {e}")

    st.divider()

    # --- Document Library ---
    st.markdown("#### 📚 Library")
    docs = fetch_documents()
    if docs:
        for doc in docs:
            icon = {"pdf": "📄", "docx": "📝", "csv": "📊", "url": "🌐"}.get(doc["source_type"], "📁")
            name = doc["name"][:25] + "..." if len(doc["name"]) > 25 else doc["name"]
            st.markdown(f"{icon} **{name}** \n<span style='font-size:0.8em; color:gray;'>{doc['chunks']} chunks</span>", unsafe_allow_html=True)
    else:
        st.caption("Your library is empty. Upload a file above.")

    st.divider()
    
    # --- Settings/Reset ---
    st.caption("⚙️ Settings")
    if st.button("🗑️ Clear Chat & Knowledge Base", use_container_width=True):
        reset_session()
        st.toast("🧹 Memory cleared successfully.")
        st.rerun()


# --- Main chat area ---
st.title("Chat with your Data")
st.caption("Powered by Gemini API & FastAPI RAG Backend")

# Beautiful Empty State
if not docs and not st.session_state.messages:
    st.markdown("<br><br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.info("👋 **Welcome to your RAG Chatbot!**\n\nTo get started, head over to the sidebar and upload a PDF, Word document, CSV, or paste a web URL. Once ingested, you can ask questions directly against your data.")

# Render Chat History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander(f"📑 View Sources ({len(msg['sources'])})"):
                for i, src in enumerate(msg["sources"]):
                    file_label = src.get("filename", src.get("file", "Unknown"))
                    page_label = f"• Page {src['page']}" if src.get("page") not in [None, "N/A", ""] else ""
                    st.markdown(f"""
                    <div class='source-card'>
                        <div class='source-title'>[{i+1}] {file_label} {page_label}</div>
                        <div class='source-content'>{src['content']}...</div>
                    </div>
                    """, unsafe_allow_html=True)

# Chat Input
if prompt := st.chat_input("Ask a question about your documents..."):
    if not docs:
        st.warning("Please upload a document to the knowledge base first.")
        st.stop()

    # User message
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Assistant response
    with st.chat_message("assistant"):
        with st.spinner("Analyzing documents..."):
            try:
                result = ask_question(prompt)
                answer = result["answer"]
                sources = result.get("sources", [])

                st.markdown(answer)

                if sources:
                    with st.expander(f"📑 View Sources ({len(sources)})"):
                        for i, src in enumerate(sources):
                            file_label = src.get("filename", src.get("file", "Unknown"))
                            page_label = f"• Page {src['page']}" if src.get("page") not in [None, "N/A", ""] else ""
                            st.markdown(f"""
                            <div class='source-card'>
                                <div class='source-title'>[{i+1}] {file_label} {page_label}</div>
                                <div class='source-content'>{src['content']}...</div>
                            </div>
                            """, unsafe_allow_html=True)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources
                })

            except Exception as e:
                err = f"❌ Error: {e}"
                st.error(err)
                st.session_state.messages.append({
                    "role": "assistant", "content": err, "sources": []
                })