# QueryMind

Conversational BI agent that translates natural-language questions to SQL using RAG-powered schema retrieval and frontier LLM generation.

> **Status:** 🚧 Under active development — Phase 0 (Environment Setup)

## Quick Start

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/querymind.git
cd querymind

# Create virtual environment and install dependencies
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux
pip install -e ".[dev]"

# Copy environment template and add your API key
cp .env.example .env

# Run the app
streamlit run app/streamlit_app.py
```

## Architecture

*Architecture diagram will be added in Phase 5.*

## Tech Stack

Python · SQLite · SQLAlchemy · OpenAI API · ChromaDB · sentence-transformers · sqlglot · pandas · Plotly · Streamlit · pytest · Docker
