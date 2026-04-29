# 🚀 AI Study Copilot

AI Study Copilot is an AI-powered study assistant designed to accelerate the learning process by analyzing PDF lecture notes. The project parses documents, divides the text into meaningful semantic chunks, retrieves the most relevant information using vector search, and automatically generates cloze-type (fill-in-the-blank) quiz questions.

This approach transforms the passive "document reading" experience into an active, interactive learning session, ensuring better retention of complex subjects.

## 🎯 Why This Project?
* Extracting specific, accurate information from lengthy PDF notes is often time-consuming.
* Passive reading methods generally lead to low knowledge retention.
* The learning process needs to be more efficient, measurable, and intelligent through modern AI capabilities.

## ⚙️ How It Works (Architecture)
1. **Document Ingestion:** Reads and parses PDF files (`pdf_reader`).
2. **Text Processing:** Cleans the extracted text and splits it into manageable pieces (`chunker`).
3. **Vectorization & Indexing:** Converts text chunks into embeddings and indexes them using FAISS for rapid retrieval (`retriever`).
4. **Contextual Retrieval:** Finds the most relevant document sections based on the user's query using Retrieval-Augmented Generation (RAG).
5. **Quiz Generation:** Automatically generates interactive quiz questions from the retrieved context (`quiz_generator`).

## 🛠️ Tech Stack
* **Language:** Python
* **AI & NLP:** RAG Architecture, Local LLMs (Ollama), Text Embeddings
* **Vector Database:** FAISS
* **Frontend/UI:** Streamlit

## 🚧 Project Status
This project is currently in the Minimum Viable Product (MVP) phase. The core architecture is functional, and it is actively being expanded to build a more robust, feature-rich study copilot experience.
