"""RAG pipeline for AgencyMind — embedding, retrieval, and LLM querying.

Orchestrates:
  1. HuggingFace embeddings (all-MiniLM-L6-v2) — local, free, no API key.
  2. FAISS vector store — in-memory, fast similarity search.
  3. Groq LLaMA 3.3-70B — free-tier LLM for answer generation.
  4. Chat history management — kept in session_state, passed to pipeline.
"""

import os
from typing import Dict, List, Optional, Tuple

import streamlit as st
from dotenv import load_dotenv
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq

from prompts import AGENCY_SYSTEM_PROMPT

load_dotenv()


class AgencyMindRAG:
    """Manages the full RAG lifecycle: indexing, retrieval, and generation.

    Usage:
        rag = AgencyMindRAG(api_key="gsk_...")
        rag.build_index(documents)
        answer, sources = rag.query("What's our SOP for onboarding?")
    """

    def __init__(self, api_key: Optional[str] = None):
        """Initialize the RAG engine with embeddings and optionally the LLM.

        Args:
            api_key: Groq API key. Falls back to GROQ_API_KEY env var or
                     session_state if not provided.
        """
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
        )
        self.vector_store: Optional[FAISS] = None
        self.llm: Optional[ChatGroq] = None

        self._documents: List[Document] = []
        self._source_filenames: set = set()

        if api_key or os.getenv("GROQ_API_KEY"):
            self._init_llm(api_key)

    def _init_llm(self, api_key: Optional[str] = None) -> None:
        """Configure the Groq LLaMA 3.3-70B LLM.

        Args:
            api_key: Groq API key. Falls back to env var.
        """
        key = api_key or os.getenv("GROQ_API_KEY") or st.session_state.get("GROQ_API_KEY", "")
        if key:
            self.llm = ChatGroq(
                api_key=key,
                model="llama-3.3-70b-versatile",
                temperature=0.3,
                max_tokens=1024,
            )

    def set_api_key(self, api_key: str) -> None:
        """Set or update the Groq API key and re-initialize the LLM.

        Args:
            api_key: Groq API key string.
        """
        self._init_llm(api_key)

    @property
    def is_llm_ready(self) -> bool:
        """Check whether the LLM has been configured with a valid API key."""
        return self.llm is not None

    def build_index(self, documents: List[Document]) -> None:
        """Embed all document chunks and build (or rebuild) the FAISS index.

        Args:
            documents: List of LangChain Document chunks from the processor.
        """
        if not documents:
            return

        self._documents = documents
        self._source_filenames = {
            doc.metadata.get("source", "unknown") for doc in documents
        }

        self.vector_store = FAISS.from_documents(
            documents=documents,
            embedding=self.embeddings,
        )

    def delete_document(self, source_filename: str) -> None:
        """Remove all chunks belonging to a given source file from the index.

        The FAISS index is rebuilt from scratch excluding the deleted file's
        chunks.

        Args:
            source_filename: The source filename to remove (e.g. "agency_sop.txt").
        """
        if self.vector_store is None:
            return

        filtered = [
            doc
            for doc in self._documents
            if doc.metadata.get("source") != source_filename
        ]
        self._source_filenames.discard(source_filename)
        self._documents = filtered

        if filtered:
            self.vector_store = FAISS.from_documents(
                documents=filtered,
                embedding=self.embeddings,
            )
        else:
            self.vector_store = None

    def get_stats(self) -> Dict[str, int]:
        """Return indexing statistics.

        Returns:
            Dict with keys 'documents' (unique source files) and 'chunks'
            (total chunk count).
        """
        return {
            "documents": len(self._source_filenames),
            "chunks": len(self._documents),
        }

    def query(
        self,
        question: str,
        chat_history: Optional[List[Tuple[str, str]]] = None,
    ) -> Tuple[str, List[Dict[str, str]]]:
        """Run a full RAG query — retrieve context, call LLM, return answer.

        Args:
            question: The user's question string.
            chat_history: List of (user_message, assistant_message) tuples
                          from the current session.

        Returns:
            Tuple of (answer_text, sources_list) where sources_list is a list
            of dicts with keys 'source' and 'excerpt'.

        Raises:
            RuntimeError: If the vector store or LLM is not initialized.
        """
        if self.vector_store is None:
            raise RuntimeError("No documents indexed. Please upload documents first.")

        if self.llm is None:
            raise RuntimeError(
                "Groq API key not configured. Please add your API key in the sidebar."
            )

        # --- 1. Build conversation history for LangChain ---
        langchain_history = []
        if chat_history:
            for user_msg, ai_msg in chat_history:
                langchain_history.append(HumanMessage(content=user_msg))
                langchain_history.append(AIMessage(content=ai_msg))

        # --- 2. Contextualize question with history ---
        contextualize_prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "Given a chat history and the latest user question, "
                "rephrase the question to be standalone if it references "
                "prior context. Otherwise reply with the question as-is.",
            ),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])

        retriever = self.vector_store.as_retriever(
            search_kwargs={"k": 4},
        )

        history_aware_retriever = create_history_aware_retriever(
            llm=self.llm,
            retriever=retriever,
            prompt=contextualize_prompt,
        )

        # --- 3. Create QA chain with system prompt ---
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", AGENCY_SYSTEM_PROMPT),
            ("system", "Context:\n{context}"),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])

        combine_docs_chain = create_stuff_documents_chain(
            llm=self.llm,
            prompt=qa_prompt,
        )

        rag_chain = create_retrieval_chain(
            retriever=history_aware_retriever,
            combine_docs_chain=combine_docs_chain,
        )

        # --- 4. Execute ---
        result = rag_chain.invoke({
            "input": question,
            "chat_history": langchain_history,
        })

        answer = result.get("answer", "No answer generated.")
        context_docs: List[Document] = result.get("context", [])

        # --- 5. Extract source info ---
        seen = set()
        sources = []
        for doc in context_docs:
            source = doc.metadata.get("source", "Unknown")
            if source not in seen:
                seen.add(source)
                excerpt = doc.page_content[:150].replace("\n", " ")
                sources.append({
                    "source": source,
                    "excerpt": excerpt + ("..." if len(doc.page_content) > 150 else ""),
                })

        return answer, sources
