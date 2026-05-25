"""AgencyMind — RAG Knowledge Base Chatbot for Digital Agencies.

Main Streamlit application entry point.
Handles the sidebar (document management, API key input) and the
main chat interface with source citations.
"""

import os
from typing import List, Optional

import streamlit as st

from document_processor import load_sample_documents, process_uploaded_files
from rag_engine import AgencyMindRAG

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="AgencyMind",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------
def init_session_state() -> None:
    """Initialise all required session_state variables on first load."""
    if "rag" not in st.session_state:
        st.session_state.rag = AgencyMindRAG()
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []  # list of (user, assistant)
    if "uploaded_docs" not in st.session_state:
        st.session_state.uploaded_docs: List[str] = []
    if "messages" not in st.session_state:
        st.session_state.messages = []  # list of dicts {"role": ..., "content": ...}


init_session_state()

# ---------------------------------------------------------------------------
# Helper — rebuild index from currently tracked uploaded docs
# ---------------------------------------------------------------------------
def rebuild_index() -> None:
    """Re-process all currently tracked uploaded files and rebuild the FAISS index.

    Reads files from the uploads tracker persisted in session_state.
    """
    if not st.session_state.uploaded_docs:
        st.session_state.rag = AgencyMindRAG()
        return

    uploaded_files = []
    for meta in st.session_state.uploaded_docs:
        file_path = meta.get("path", "")
        if os.path.exists(file_path):
            import io
            import mimetypes

            with open(file_path, "rb") as fh:
                buf = io.BytesIO(fh.read())
                buf.name = meta["name"]
                uploaded_files.append(buf)

    if uploaded_files:
        try:
            docs = process_uploaded_files(uploaded_files)
            st.session_state.rag.build_index(docs)
        except Exception as e:
            st.error(f"Error rebuilding index: {e}")


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
col_title, col_clear = st.columns([4, 1])
with col_title:
    st.title("AgencyMind 🧠")
    st.caption("Your Agency Knowledge Base — Powered by AI")
with col_clear:
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.messages = []
        st.rerun()

# ---------------------------------------------------------------------------
# Sidebar — Document Manager + API Key
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("📂 Document Manager")

    # --- API Key ---
    st.markdown("#### 🔑 Groq API Key")
    api_key = st.text_input(
        "Groq API Key",
        type="password",
        placeholder="gsk_...",
        label_visibility="collapsed",
        value=st.session_state.get("GROQ_API_KEY", ""),
    )
    if api_key:
        st.session_state["GROQ_API_KEY"] = api_key
        st.session_state.rag.set_api_key(api_key)

    if not st.session_state.rag.is_llm_ready:
        st.warning(
            "⚠️ No API key set. Add your Groq API key above to enable the chatbot. "
            "Get a free key at [console.groq.com](https://console.groq.com)."
        )

    st.divider()

    # --- File Uploader ---
    uploaded_files = st.file_uploader(
        "Upload PDF or TXT files",
        type=["pdf", "txt"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded_files:
        new_files = []
        for uf in uploaded_files:
            # Avoid re-processing already-tracked files (compare by name + size)
            if not any(
                meta["name"] == uf.name for meta in st.session_state.uploaded_docs
            ):
                new_files.append(uf)

        if new_files:
            with st.spinner("Processing uploaded documents..."):
                try:
                    docs = process_uploaded_files(new_files)

                    # Persist uploaded files so we can re-index on delete
                    import tempfile

                    for nf in new_files:
                        tmp = tempfile.NamedTemporaryFile(
                            delete=False, suffix=os.path.splitext(nf.name)[1]
                        )
                        tmp.write(nf.read())
                        tmp.flush()
                        st.session_state.uploaded_docs.append({
                            "name": nf.name,
                            "path": tmp.name,
                        })

                    st.session_state.rag.build_index(docs)
                    st.success(f"✅ {len(new_files)} file(s) indexed.")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Error processing files: {e}")

    # --- Load Sample Docs ---
    if st.button("📚 Load Sample Agency Docs", use_container_width=True):
        with st.spinner("Loading sample documents..."):
            try:
                sample_docs = load_sample_documents()
                st.session_state.rag.build_index(sample_docs)

                # Track sample docs in upload list
                sample_names = {"agency_sop.txt", "wordpress_guide.txt", "seo_checklist.txt"}
                existing_names = {
                    meta["name"] for meta in st.session_state.uploaded_docs
                }
                for sname in sample_names:
                    if sname not in existing_names:
                        st.session_state.uploaded_docs.append({
                            "name": sname,
                            "path": os.path.join("sample_docs", sname),
                        })

                st.success("✅ 3 sample documents loaded.")
                st.rerun()
            except Exception as e:
                st.error(f"❌ Error loading sample docs: {e}")

    st.divider()

    # --- Document List + Delete ---
    st.markdown("#### 📄 Loaded Documents")
    if st.session_state.uploaded_docs:
        for meta in st.session_state.uploaded_docs[:]:
            col1, col2 = st.columns([4, 1])
            col1.write(f"📄 {meta['name']}")
            if col2.button("🗑️", key=f"del_{meta['name']}"):
                try:
                    st.session_state.rag.delete_document(meta["name"])
                    st.session_state.uploaded_docs = [
                        m for m in st.session_state.uploaded_docs
                        if m["name"] != meta["name"]
                    ]
                    st.rerun()
                except Exception as e:
                    st.error(f"Error deleting document: {e}")
    else:
        st.info("No documents loaded yet.")

    # --- Stats ---
    stats = st.session_state.rag.get_stats()
    if stats["chunks"] > 0:
        st.markdown(
            f"**✅ {stats['documents']} documents indexed | "
            f"{stats['chunks']} chunks ready**"
        )
    else:
        st.markdown(
            "**No documents indexed yet. Upload files or load sample docs.**"
        )

# ---------------------------------------------------------------------------
# Main area — Chat interface
# ---------------------------------------------------------------------------
# If no documents, show empty state
if not st.session_state.rag.get_stats()["chunks"]:
    st.info(
        "📂 **No documents loaded yet.**\n\n"
        "Upload your agency SOPs, service guides, or client briefs — "
        "or click **'Load Sample Docs'** in the sidebar to try a demo."
    )

# Display message history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg and msg["sources"]:
            with st.expander("📄 Sources Used"):
                for src in msg["sources"]:
                    st.markdown(f"**📁 {src['source']}**")
                    st.caption(f"_{src['excerpt']}_")

# Chat input — disabled when no API key or no docs
no_api = not st.session_state.rag.is_llm_ready
no_docs = not st.session_state.rag.get_stats()["chunks"]

if prompt := st.chat_input(
    "Ask anything about your agency documents...",
    disabled=no_api or no_docs,
):
    # Guard: empty query
    if not prompt.strip():
        st.stop()

    # Guard: no documents
    if no_docs:
        with st.chat_message("assistant"):
            st.markdown(
                "Please upload documents first before asking questions."
            )
        st.stop()

    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                answer, sources = st.session_state.rag.query(
                    question=prompt,
                    chat_history=st.session_state.chat_history,
                )

                st.markdown(answer)
                if sources:
                    with st.expander("📄 Sources Used"):
                        for src in sources:
                            st.markdown(f"**📁 {src['source']}**")
                            st.caption(f"_{src['excerpt']}_")

                # Persist
                st.session_state.chat_history.append((prompt, answer))
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                })

            except RuntimeError as e:
                st.error(str(e))
            except Exception as e:
                st.error(f"API error, please retry: {e}")

# ---------------------------------------------------------------------------
# Footer — Download Chat
# ---------------------------------------------------------------------------
if st.session_state.chat_history:
    st.divider()
    chat_text_lines = []
    for user_msg, ai_msg in st.session_state.chat_history:
        chat_text_lines.append(f"User: {user_msg}")
        chat_text_lines.append(f"AgencyMind: {ai_msg}")
        chat_text_lines.append("---")
    chat_text = "\n".join(chat_text_lines)

    st.download_button(
        label="💾 Download Chat",
        data=chat_text,
        file_name="agencymind_chat.txt",
        mime="text/plain",
    )
