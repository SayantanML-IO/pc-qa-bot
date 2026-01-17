# PC Hardware Q&A Bot 🤖

A Retrieval-Augmented Generation (RAG) system that scrapes real-time PC hardware news, builds a vector database, and answers user questions using Groq's Llama-3 model.

## 🚀 Key Features

* **📰 Real-Time RAG Pipeline:** Automatically scrapes 10+ major hardware news sources (Tom's Hardware, TechSpot, etc.) via RSS.
* **⚡ High-Performance Scraping:** Uses `ThreadPoolExecutor` for parallel article fetching and processing.
* **🧠 Vector Search:** Built on **LangChain** and **ChromaDB** for semantic search and retrieval.
* **🤖 Fast Inference:** Powered by **Groq API (Llama-3 70B)** for near-instant answers.
* **📊 Metrics & Monitoring:** Tracks scraping success rates, token usage, and processing times in a local SQLite database.

## 🛠️ Tech Stack

* **LLM:** Groq API (Llama-3.3 70B)
* **Framework:** LangChain
* **Vector DB:** ChromaDB
* **Scraper:** BeautifulSoup4 + Feedparser
* **Processing:** PyTorch (for embeddings)

## ⚙️ Setup & Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/SayantanDe04/pc-qa-bot.git](https://github.com/SayantanDe04/pc-qa-bot.git)
    cd pc-qa-bot
    ```

2.  **Install dependencies:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    pip install -r requirements.txt
    ```

3.  **Configure API Keys:**
    Create a `.env` file in the root directory:
    ```ini
    GROQ_API_KEY=your_groq_api_key_here
    ```

4.  **Usage:**
    
    **Step 1: Build the Knowledge Base**
    Run the indexer to scrape news and populate the vector database:
    ```bash
    python main.py
    ```

    **Step 2: Chat with the Bot**
    Launch the CLI to ask questions:
    ```bash
    python cli.py
    ```