"""
Команди в термінал
cd rag-engine
streamlit run ui/streamlit_app.py
"""
import sys
from pathlib import Path

# Додаємо корінь проєкту в PYTHONPATH
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
from pathlib import Path

# --------------------------------------------------
# ІМПОРТ RAG SERVICE
# --------------------------------------------------

from services.rag_service import RAGService


# --------------------------------------------------
# STREAMLIT CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="Local RAG Engine",
    layout="wide"
)

st.title("🧠 Local RAG Engine")
st.caption("Engineering UI for testing RAG pipeline")


# --------------------------------------------------
# PATHS & CONFIG
# --------------------------------------------------

BASE_PATH = Path(__file__).resolve().parents[1]

DATA_PATH = BASE_PATH / "data"
DOCS_PATH = DATA_PATH / "documents"
CHUNKS_PATH = DATA_PATH / "chunks"
FEEDBACK_PATH = DATA_PATH / "feedback"

for p in [DOCS_PATH, CHUNKS_PATH, FEEDBACK_PATH]:
    p.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------

if "rag" not in st.session_state:
    st.session_state.rag = RAGService(
        documents_path=str(DOCS_PATH),
        chunks_path=str(CHUNKS_PATH),
        feedback_path=str(FEEDBACK_PATH),
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        llm_model="phi3:latest"
    )

if "last_response" not in st.session_state:
    st.session_state.last_response = None


rag = st.session_state.rag


# --------------------------------------------------
# SIDEBAR — INGESTION
# --------------------------------------------------

st.sidebar.header("📄 Document ingestion")

uploaded_file = st.sidebar.file_uploader(
    "Upload PDF",
    type=["pdf"]
)

if uploaded_file:
    temp_path = DATA_PATH / uploaded_file.name
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    if st.sidebar.button("📥 Ingest document"):
        with st.spinner("Indexing document..."):
            result = rag.ingest_document(
                source=str(temp_path),
                file_type="pdf"
            )
        st.sidebar.success("Document indexed")
        st.sidebar.json(result)


# --------------------------------------------------
# MAIN — QA INTERFACE
# --------------------------------------------------

st.subheader("💬 Ask a question")

question = st.text_input(
    "Your question",
    placeholder="Ask something about the uploaded documents..."
)

if st.button("Ask") and question:
    with st.spinner("Thinking..."):
        response = rag.ask(question)

    st.session_state.last_response = response

    st.markdown("### 🧠 Answer")
    st.write(response["answer"])

    st.markdown("### 📊 Evaluation")
    st.json(response["evaluation"])

    st.markdown("### 📚 Sources")
    st.write(response["sources"])


# --------------------------------------------------
# FEEDBACK LOOP
# --------------------------------------------------

if st.session_state.last_response:
    st.divider()
    st.subheader("🗳 Feedback")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("👍 Helpful"):
            rag.submit_feedback(
                feedback_id=st.session_state.last_response["feedback_id"],
                rating=1
            )
            st.success("Thanks for your feedback!")

    with col2:
        if st.button("👎 Not helpful"):
            rag.submit_feedback(
                feedback_id=st.session_state.last_response["feedback_id"],
                rating=-1
            )
            st.warning("Feedback recorded")
