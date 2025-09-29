# 🧠 NeuroMed AI – MedicalSearch  

**Smart Medical Research Assistant powered by Retrieval-Augmented Generation (RAG)**  

This project builds a RAG pipeline for **summarization** and **question-answering** over scientific / medical papers (mainly from arXiv).  
It integrates **web scraping, embeddings, vector databases, and LLMs** to deliver concise research summaries, relevant URLs, and detailed answers from PDFs.  

---

## 📊 System Architecture

![Architecture Diagram](Extracted_Images/architecture.png)

---

## 📂 Repository Structure
```bash
medicalSearch/
│
├── Scraping/ # Web scraping modules for arXiv / papers
├── Rag_Summary/ # Summarization pipeline
├── Qa-Bot/ # Question-answering over PDFs
├── Neuro-Med-app/ # Application / API / frontend
├── Murag/ # Experiments / alternative implementations
├── Extracted_Images/ # Assets & diagrams
├── .env # Environment variables
├── .gitignore
└── README.md # Project documentation
```
---

## 🚀 Features

- **Web Scraping**: Collects abstracts and PDFs from arXiv.  
- **Embedding Models**: Supports `sentence-transformers`, `bioBERT`, and `deepseek-embed`.  
- **Vector Databases**:  
  - **Elasticsearch** → Abstracts & metadata storage.  
  - **Chromedb** → PDF embeddings & retrieval.  
- **Summarization (RAG)**: Retrieves abstracts and generates concise summaries with paper URLs.  
- **Q&A (RAG)**: Answers detailed questions by retrieving relevant PDF passages.  
- **LLM Augmentation**: Uses **LLaMA 3.2** and **DeepSeek (medical / fine-tuned)** for improved responses.  

---

## ⚙️ Installation

### Prerequisites
- Python ≥ 3.8  
- pip / conda  
- Running **Elasticsearch** instance  
- **Chromedb** (or other vector DB)  
- GPU (recommended for embeddings & LLMs)  

### Setup

1. Clone the repository:
   ```sh
   git clone https://github.com/sara-bm/medicalSearch.git
   cd medicalSearch
```
2. Create a virtual environment:
   ```bash
python -m venv venv
source venv/bin/activate   # macOS/Linux
# .\venv\Scripts\activate  # Windows
```
