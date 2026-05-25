# AgencyMind 🧠 — RAG Knowledge Base for Digital Agencies

> Upload your agency SOPs, service docs, and client briefs.  
> Ask anything. Get answers with sources — instantly.

## What It Does
AgencyMind lets digital agencies turn their documentation into an intelligent Q&A assistant. Built with RAG (Retrieval-Augmented Generation) — it only answers from YOUR documents, never hallucinates.

## Use Cases (E2M-Specific)
- "What's our WordPress handoff checklist?"
- "How many revision rounds do we include?"
- "What's the SEO reporting structure for monthly clients?"

## Tech Stack
[Python] [LangChain] [FAISS] [Groq LLaMA3] [HuggingFace] [Streamlit]

## Features
- Multi-document upload (PDF + TXT)
- Source citations with every answer
- Free to run — Groq API + local HuggingFace embeddings
- Pre-loaded agency sample documents for demo
- Export full conversation

## Quick Start
```bash
git clone https://github.com/YOUR_USERNAME/agencymind-rag
cd agencymind-rag
pip install -r requirements.txt
cp .env.example .env
# Add your Groq API key to .env
streamlit run app.py
```

## Live Demo
[Deployed on Hugging Face Spaces] → LINK HERE
