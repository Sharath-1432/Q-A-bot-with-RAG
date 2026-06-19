# Q-A-bot-with-RAG
RAG Question Answering Bot

Overview
This project uses:
- LangChain
- ChromaDB
- HuggingFace Embeddings
- Google Gemini

Installation
1. Install Python
2. Install requirements
3. Add Gemini API key
4. Run ingest.py
5. Run main.py

Architecture

Chunking Strategy
Documents are split into chunks with overlap to preserve context.

Embedding Model
all-MiniLM-L6-v2

Vector Database
ChromaDB

LLM
Gemini 2.5 Flash
