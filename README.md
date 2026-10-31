# RAG Telecom Chatbot

A Retrieval-Augmented Generation (RAG) customer care chatbot for telecom support. It answers questions about mobile connectivity, billing, SIM issues, and roaming by retrieving relevant context from three knowledge sources and generating responses with a Groq-hosted language model.

## Architecture

```
User question
     │
     ▼
Merged Retriever (top-k from each store)
  ├── ChromaDB · faq        (FAQ entries from CSV)
  ├── ChromaDB · tickets    (resolved support tickets from SQLite)
  └── ChromaDB · guides     (PDF guide chunks)
     │
     ▼
ChatPromptTemplate → Groq-hosted language model → Answer
```

**Embedding model:** `sentence-transformers/all-MiniLM-L6-v2` (runs locally via HuggingFace)  
**LLM:** configured through `GROQ_MODEL` and served by [Groq](https://groq.com)

## Project Structure

```
rag-telecom-chatbot/
├── app.py              # Streamlit web UI
├── main.py             # CLI entry point
├── rag_chain.py        # Builds the LangChain RAG chain
├── retriever.py        # Merges the three Chroma retrievers
├── config.py           # Validates secrets and defines project paths
├── ingest_faq.py       # Loads data/faq.csv → Chroma 'faq' collection
├── ingest_tickets.py   # Loads data/tickets.db → Chroma 'tickets' collection
├── ingest_pdf.py       # Loads data/telecom_guide.pdf → Chroma 'guides' collection
├── data/
│   ├── faq.csv             # FAQ question/answer pairs
│   ├── tickets.db          # SQLite database of resolved support tickets
│   ├── telecom_guide.pdf   # Telecom user guide (chunked at ingest)
│   ├── seed_tickets.py     # Script to seed the tickets database
│   └── generate_pdf.py     # Script to generate the telecom guide PDF
├── chroma_store/       # Persisted Chroma vector database (created at ingest)
├── pyproject.toml
├── requirements.txt    # Streamlit Cloud dependencies
├── uv.lock
└── .env.example
```

## Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip
- A [Groq API key](https://console.groq.com)
- A [HuggingFace token](https://huggingface.co/settings/tokens) (for downloading the embedding model)

Only `GROQ_API_KEY` is required for the chatbot. `HF_TOKEN` is optional, but
recommended for reliable Hugging Face model downloads. `GOOGLE_API_KEY` is not
used by this project.

## Setup

**1. Clone and install dependencies**

```bash
git clone <repo-url>
cd rag-telecom-chatbot
uv sync          # or: pip install -e .
```

**2. Configure environment variables**

```bash
cp .env.example .env
```

Edit `.env` and fill in your keys:

```
GROQ_API_KEY=your_groq_api_key_here
HF_TOKEN=your_huggingface_token_here
```

**3. Ingest data into Chroma**

Run the three ingestion scripts once to build the vector store:

```bash
python ingest_faq.py
python ingest_tickets.py
python ingest_pdf.py
```

Each script embeds the source data and persists it to `chroma_store/`. Re-run a script only when its source data changes.
The vector store is generated locally and is intentionally ignored by Git.

## Running the App

**Streamlit web UI**

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`. The sidebar has one-click sample questions and a button to clear the conversation history.

## Streamlit Cloud deployment

1. Push the project to GitHub without committing `.env` or any API key.
2. Create a Streamlit Community Cloud app using `app.py` as the entry point.
3. Add the following values in **App settings → Secrets**:

```toml
GROQ_API_KEY = "your_groq_api_key"
HF_TOKEN = "your_huggingface_token"
```

4. Ensure the generated `chroma_store/` is available to the deployment. The
   current setup creates it from the source files with the ingestion commands
   above, so either commit the generated store for a small demo or add a
   deployment step that runs all three ingestion scripts before the app starts.

Never put secrets in `app.py`, `rag_chain.py`, `.env.example`, or GitHub.

**CLI**

```bash
python main.py
```

Interactive prompt — type a question and press Enter. Type `quit` to exit.

## Data Sources

| Collection | Source file | Granularity |
|---|---|---|
| `faq` | `data/faq.csv` | 1 document per FAQ row |
| `tickets` | `data/tickets.db` | 1 document per resolved ticket |
| `guides` | `data/telecom_guide.pdf` | Chunks of 600 chars with 100-char overlap |

The retriever fetches the top 3 results from each collection (9 context documents total) for every query.

## Regenerating Seed Data

```bash
# Seed the SQLite ticket database
python data/seed_tickets.py

# Regenerate the PDF guide
python data/generate_pdf.py
```

After regenerating, re-run the corresponding ingest script.
