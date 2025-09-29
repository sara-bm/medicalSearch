# 🧠 NeuroMed AI – MedicalSearch  

**Smart Medical Research Assistant powered by Retrieval-Augmented Generation (RAG)**  

This project builds a RAG pipeline for **summarization** and **question-answering** over scientific / medical papers (mainly from arXiv).  
It integrates **web scraping, embeddings, vector databases, and LLMs** to deliver concise research summaries, relevant URLs, and detailed answers from PDFs.  

---

## 📊 System Architecture

![Architecture Diagram](Extracted_Images/architecture.png)

---

## 📂 Repository Structure

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

markdown
Copier le code

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
   ```bash
   git clone https://github.com/sara-bm/medicalSearch.git
   cd medicalSearch
Create a virtual environment:

bash
Copier le code
python -m venv venv
source venv/bin/activate   # macOS/Linux
# .\venv\Scripts\activate  # Windows
Install dependencies:

bash
Copier le code
pip install -r requirements.txt
Configure environment variables in a .env file:

ini
Copier le code
ELASTICSEARCH_URL=http://localhost:9200
ELASTICSEARCH_USER=your_user
ELASTICSEARCH_PASSWORD=your_password
CHROMEDB_PATH=./chromedb
OPENAI_API_KEY=your_api_key
MODEL_PATH=./models
▶️ Usage
1. Scraping Papers
Run the scraper to fetch abstracts & PDFs:

bash
Copier le code
python Scraping/run_scraper.py
2. Summarization
Get a summary + paper URLs for a query:

bash
Copier le code
python Rag_Summary/summarize.py --query "latest research on mRNA vaccines"
3. Q&A
Answer detailed questions from PDFs:

bash
Copier le code
python Qa-Bot/qa.py --query "What side effects were reported in mRNA vaccine studies?"
4. Run Web App (if available)
bash
Copier le code
cd Neuro-Med-app
uvicorn app:app --reload
🔧 Customization
Embeddings: Swap between BioBERT, SciBERT, or Sentence-BERT.

LLMs: Replace or fine-tune LLaMA / DeepSeek.

Retrieval: Adjust similarity thresholds & top-k results.

PDF Splitting: Tune chunk size for document embeddings.

📦 Dependencies
Key libraries:

transformers, sentence-transformers, bioBERT

elasticsearch, chromadb

uvicorn, fastapi (for API)

pdfplumber / PyPDF2 (PDF parsing)

scikit-learn or faiss (similarity search)

Install all via:

bash
Copier le code
pip install -r requirements.txt
✅ Example Workflow
Scrape new papers from arXiv.

Index abstracts in Elasticsearch.

Embed full-text PDFs into Chromedb.

User asks: “What are the risks of long-term AI use in radiology?”

Summarizer returns summary + URLs.

QA module retrieves PDFs & answers in detail.

📌 Limitations
Depends on scraped datasets (limited coverage).

Risk of hallucinations from LLM.

PDF parsing may introduce noise.

Latency when processing large PDFs.

🤝 Contributing
Fork the project

Create your feature branch (git checkout -b feature/new-feature)

Commit your changes (git commit -m 'Add new feature')

Push to the branch (git push origin feature/new-feature)

Open a Pull Request

📜 License
MIT License

✨ Citation
If you use this project in research, please cite:

nginx
Copier le code
NeuroMed / medicalSearch (2025).
"RAG pipeline for medical literature summarization & Q&A."
GitHub Repository: https://github.com/sara-bm/medicalSearch
pgsql
Copier le code

👉 Rename your uploaded diagram to `architecture.png` and put it inside `Extracted_Images/` so it renders correctly in GitHub.  

Do you also want me to **add badges** (Python version, license, stars, issues) at the top for a more professional look?






