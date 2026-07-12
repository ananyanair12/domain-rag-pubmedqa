# Domain-Specific RAG with PubMedQA

A polished end-to-end retrieval-augmented generation (RAG) project for biomedical question answering over PubMedQA, built to compare baseline embeddings against contrastively fine-tuned embeddings.

## Why this project matters

This repo demonstrates how to build a domain-specific RAG system that:
- retrieves relevant biomedical passages from a chunked corpus,
- compares off-the-shelf versus fine-tuned embedding models,
- reranks candidates with a cross-encoder,
- generates grounded answers with an LLM backend,
- and evaluates retrieval quality with simple, reproducible metrics.

## Key features

- FastAPI backend for querying the RAG pipeline
- Streamlit dashboard for interactive demos
- Chunk-size ablation for 256 / 512 / 1024 token settings
- Retrieval evaluation with precision@k and related metrics
- Optional Groq-based answer generation
- Lightweight test suite for regression safety

## Project structure

```text
.
├── api/                  # FastAPI backend
├── dashboard/            # Streamlit UI
├── data/                 # PubMedQA download and preprocessing scripts
├── notebooks/            # Colab notebook for GPU fine-tuning
├── scripts/              # Local pipeline helpers
├── src/                  # Core pipeline logic
├── tests/                # Pytest sanity tests
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── .env.example
```

## Quick start

### 1) Clone and install

```bash
git clone <your-repo-url>
cd rag-project
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
```

### 2) Run the API

```bash
uvicorn api.main:app --reload --port 8000
```

Then open:
- http://localhost:8000/health
- http://localhost:8000/docs

### 3) Launch the dashboard

```bash
streamlit run dashboard/app.py
```

### 4) Run the tests

```bash
pytest -q
```

## Research goal

The core thesis question is simple:

> Does contrastive fine-tuning of the embedding model improve retrieval quality and answer faithfulness over general-purpose embeddings in a biomedical RAG system?

## Evaluation outputs

After running the evaluation scripts, the project produces:
- results/before_after_finetuning.json
- results/chunk_size_ablation.json
- results/ragas_comparison.json

## Notes

The project uses a weak-labeling approach for retrieval evaluation because PubMedQA does not ship explicit chunk-level relevance judgments. That makes the setup practical for experimentation while remaining transparent about its limitations.

## License

This project is intended for research and educational use.
