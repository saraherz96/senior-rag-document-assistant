# 🧠 Senior RAG Document Assistant

An end-to-end AI application that enables users to upload documents and interact with them using natural language queries through a Retrieval-Augmented Generation (RAG) pipeline.

---

## 🚀 Overview

This project implements a production-style RAG system that processes documents, stores them as embeddings in a vector database, retrieves relevant context, and generates grounded responses using a Large Language Model (LLM).

The goal is to demonstrate real-world AI engineering skills including semantic search, prompt engineering, and modular system design.

---

## 🏗️ Architecture
User Upload (PDF)
↓
Text Extraction (pypdf)
↓
Chunking (overlap strategy)
↓
Embedding Generation (OpenAI)
↓
Vector Storage (ChromaDB)
↓
Semantic Retrieval
↓
Context Construction
↓
LLM Response Generation
↓
Answer + Sources

## 🧠 Tech Stack

- Python
- OpenAI API (Embeddings + LLM)
- ChromaDB (Vector Database)
- Streamlit (Frontend)
- PyPDF (Document parsing)

---
## What This Project Demonstrates
Retrieval-Augmented Generation (RAG) pipelines
Embeddings and semantic similarity
Vector databases (ChromaDB)
Prompt engineering for grounded responses
Modular AI system design
End-to-end LLM application development

