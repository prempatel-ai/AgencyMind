"""Document ingestion and text chunking for AgencyMind RAG.

Handles PDF and TXT file uploads:
  - Extracts text via pdfplumber (PDF) or direct read (TXT)
  - Splits into chunks using LangChain's RecursiveCharacterTextSplitter
  - Attaches metadata {source, chunk_id} to each chunk
  - Returns list of LangChain Document objects
"""

import os
import tempfile
from typing import List, Optional

import pdfplumber
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader


def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extract and return all text from a PDF file's bytes using pdfplumber.

    Args:
        file_bytes: Raw bytes of the uploaded PDF file.

    Returns:
        Concatenated text from all pages.

    Raises:
        ValueError: If no extractable text is found in the PDF.
    """
    text_parts = []
    with pdfplumber.open(file_bytes) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)
    full_text = "\n".join(text_parts).strip()
    if not full_text:
        raise ValueError("No extractable text found in this PDF.")
    return full_text


def extract_text_from_txt(file_bytes: bytes) -> str:
    """Decode and return text from a TXT file's bytes.

    Args:
        file_bytes: Raw bytes of the uploaded TXT file.

    Returns:
        Decoded UTF-8 text content.
    """
    return file_bytes.decode("utf-8", errors="replace").strip()


def chunk_documents(
    documents: List[Document],
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> List[Document]:
    """Split LangChain Document objects into smaller overlapping chunks.

    Each output chunk receives metadata with its source filename and
    a sequential chunk_id.

    Args:
        documents: List of LangChain Documents (one per file).
        chunk_size: Maximum characters per chunk.
        chunk_overlap: Overlap characters between consecutive chunks.

    Returns:
        List of chunked Document objects with updated metadata.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    all_chunks = splitter.split_documents(documents)

    for i, chunk in enumerate(all_chunks):
        chunk.metadata["chunk_id"] = i

    return all_chunks


def process_uploaded_files(
    uploaded_files: List,
) -> List[Document]:
    """Process a list of Streamlit UploadedFile objects into chunked documents.

    For each file:
      1. Detects type by extension (.pdf / .txt).
      2. Extracts text.
      3. Wraps in a LangChain Document with source metadata.

    After extraction, all documents are chunked via chunk_documents().

    Args:
        uploaded_files: List of Streamlit UploadedFile objects from
                        st.file_uploader.

    Returns:
        List of LangChain Document chunks ready for embedding.

    Raises:
        ValueError: If unsupported file type is encountered.
    """
    raw_documents: List[Document] = []

    for uploaded_file in uploaded_files:
        filename = uploaded_file.name
        file_bytes = uploaded_file.read()
        ext = os.path.splitext(filename)[1].lower()

        try:
            if ext == ".pdf":
                text = extract_text_from_pdf(file_bytes)
            elif ext == ".txt":
                text = extract_text_from_txt(file_bytes)
            else:
                raise ValueError(f"Unsupported file type: {ext}")

            doc = Document(
                page_content=text,
                metadata={"source": filename},
            )
            raw_documents.append(doc)
        except Exception as e:
            # Re-raise with filename context so the caller can surface it
            raise RuntimeError(f"Failed to process '{filename}': {e}") from e

    return chunk_documents(raw_documents)


def load_sample_documents(
    sample_dir: str = "sample_docs",
) -> List[Document]:
    """Load the pre-bundled sample .txt documents from the sample_docs folder.

    Each file is read via TextLoader to maintain consistency with how
    user-uploaded TXT files are handled.

    Args:
        sample_dir: Path to the sample_docs directory relative to project root.

    Returns:
        List of chunked Document objects ready for embedding.
    """
    sample_files = [
        "agency_sop.txt",
        "wordpress_guide.txt",
        "seo_checklist.txt",
    ]

    raw_documents: List[Document] = []
    for fname in sample_files:
        path = os.path.join(sample_dir, fname)
        if not os.path.exists(path):
            continue
        loader = TextLoader(path, encoding="utf-8")
        for doc in loader.load():
            doc.metadata["source"] = fname
            raw_documents.append(doc)

    return chunk_documents(raw_documents)
