# JioPay Chatbot – Retrieval-Augmented Generation (RAG) Assistant

A **production-ready Retrieval-Augmented Generation (RAG) chatbot** built to automate **customer support for JioPay**.
This assistant leverages a curated knowledge base scraped from the official JioPay website, ensuring **accurate, contextual, and up-to-date answers** while minimizing hallucinations.

The project features a **modern backend** powered by **FastAPI and LangChain**, and a **responsive frontend** built with **React + Vite**.

---

## ✨ Key Features

* **Interactive Chat Interface** – A smooth and intuitive web-based chat UI.
* **Retrieval-Augmented Generation** – Answers grounded in a JioPay-specific knowledge base.
* **Automated Data Pipeline** – End-to-end scripts for scraping, processing, and indexing website content.
* **Searchable Vector Store** – Enables fast and precise information retrieval using FAISS.
* **Scalable Tech Stack** – Combines FastAPI, LangChain, HuggingFace models, and React for full-stack functionality.

---

## 🛠️ Tech Stack

**Backend**

* **Framework**: FastAPI
* **RAG Orchestration**: LangChain (LCEL)
* **Vector Database**: FAISS
* **Embedding Model**: `BAAI/bge-small-en-v1.5` (via Sentence-Transformers)
* **LLM**: `mistralai/Mistral-7B-Instruct-v0.2` (via Hugging Face Hub)
* **Web Scraping**: BeautifulSoup4, Requests
* **Data Validation**: Pydantic

**Frontend**

* **Framework**: React
* **Build Tool**: Vite
* **Styling**: Tailwind CSS
* **API Client**: Axios

---

## 🚀 Getting Started

Follow these steps to run the project locally.

### Prerequisites

* **Git**
* **Python 3.9+**
* **Node.js + npm** (or yarn)
* **Hugging Face API token** (required for LLM access)

### 1. Clone the Repository

```bash
git clone https://github.com/byterotom/JioPay_chatBot.git
cd JioPay_chatBot
```

### 2. Configure Environment Variables

Navigate to the backend folder and create a `.env` file:

```bash
cd backend
cp .env.example .env
```

Edit `.env` and add your Hugging Face API token:

```
HUGGINGFACEHUB_API_TOKEN="hf_your_token_here"
```

### 3. Backend Setup

Create and activate a virtual environment:

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

#### Build the Knowledge Base

Run the scraper to collect data:

```bash
python scrapper.py
```

This generates `jiopay_data.txt`.

Build the FAISS vector index:

```bash
python ingest.py
```

This creates the `faiss_index/` directory.

#### Start the Backend

```bash
uvicorn main:app --reload
```

Backend runs at: [http://localhost:8000](http://localhost:8000)

---

### 4. Frontend Setup

Open a new terminal and go to the frontend folder:

```bash
cd frontend
npm install
npm run dev
```

Frontend runs at: [http://localhost:5173](http://localhost:5173)

---

## 💻 Usage

Once both servers are running:

1. Visit [http://localhost:5173](http://localhost:5173).
2. Ask the JioPay Assistant any question.
3. The chatbot retrieves answers from its knowledge base and provides contextual responses.

---

## 📂 Project Structure

```
.
├── backend/
│   ├── faiss_index/      # Vector index store
│   ├── ingest.py         # Build FAISS index
│   ├── main.py           # FastAPI app + RAG pipeline
│   ├── scrapper.py       # JioPay website scraper
│   ├── requirements.txt  # Backend dependencies
│   └── .env.example      # Environment variables template
└── frontend/
    ├── src/
    │   └── App.jsx       # Main React component
    └── package.json      # Frontend dependencies
```

---

## ☁️ Deployment

* **Frontend**: Deployable on [Vercel](https://vercel.com/).
* **Backend**: Requires hosting with sufficient memory.

  * Not compatible with Render free tier (due to embedding model memory usage).
  * Works well on cloud services with >= 8GB RAM (AWS, GCP, Azure, or paid Render plan).

---

## 🤝 Contributing

Contributions are welcome! To contribute:

1. Fork the repository
2. Create a new branch (`feature/your-feature`)
3. Commit changes and push
4. Open a pull request

---

## 📜 License

This project is licensed under the **MIT License**.