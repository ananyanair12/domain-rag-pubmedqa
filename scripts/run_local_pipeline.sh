#!/usr/bin/env bash
# Runs the parts of the pipeline that do NOT need a GPU.
# Run finetune_embeddings.py separately on Colab/Kaggle first (see
# notebooks/finetune_embeddings_colab.ipynb), then pull the resulting
# `models/` folder down before running this script.
set -e

echo "== 1. Download & split PubMedQA =="
python data/download_pubmedqa.py

echo "== 2. Build chunked corpus versions (256/512/1024) =="
python src/chunking.py

echo "== 3. Build baseline + fine-tuned vector collections =="
python src/vector_store.py

echo "== 4. Evaluate retrieval precision@5 (before vs after fine-tuning) =="
python src/evaluate_retrieval.py

echo "== 5. Run RAGAs evaluation suite =="
python src/evaluate_ragas.py

echo "All done. Launch the API with:  uvicorn api.main:app --reload"
echo "Launch the dashboard with:      streamlit run dashboard/app.py"
