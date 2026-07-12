"""
Streamlit dashboard for the domain-specific RAG system.

Run:
    streamlit run dashboard/app.py

Features:
    - Ask a question, see the generated answer + retrieved chunks + scores
    - Toggle between baseline and fine-tuned embeddings to compare live
    - View saved evaluation results (precision@5 before/after, RAGAs scores,
      chunk-size ablation) if src/evaluate_retrieval.py / evaluate_ragas.py
      have already been run
"""
import json
import os
import sys

import streamlit as st

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import config
from src.pipeline import run_query

st.set_page_config(page_title="Domain RAG Dashboard", layout="wide")
st.title("🔬 Domain-Specific RAG — PubMedQA")
st.caption("Contrastively fine-tuned embeddings vs. off-the-shelf baseline")

tab_query, tab_results = st.tabs(["Ask a question", "Evaluation results"])

with tab_query:
    col1, col2 = st.columns([3, 1])
    with col1:
        question = st.text_input("Ask a biomedical question:", placeholder="e.g. Does aspirin reduce risk of colorectal cancer?")
    with col2:
        embedder_choice = st.selectbox("Embedder", ["finetuned", "baseline"])
        use_reranker = st.checkbox("Use reranker", value=True)

    if st.button("Run query", type="primary") and question:
        with st.spinner("Retrieving, reranking, and generating..."):
            try:
                result = run_query(question, embedder_kind=embedder_choice, use_reranker=use_reranker)
            except Exception as e:
                st.error(f"Error: {e}")
                result = None

        if result:
            st.subheader("Answer")
            st.write(result["answer"])

            m1, m2, m3 = st.columns(3)
            m1.metric("Embedder used", result["embedder_used"])
            m2.metric("Reranked", str(result["reranked"]))
            m3.metric("Latency (s)", result["latency_seconds"])

            st.subheader("Retrieved chunks")
            for i, chunk in enumerate(result["retrieved_chunks"]):
                score_label = chunk.get("rerank_score", chunk.get("distance"))
                with st.expander(f"Chunk {i+1} — score: {score_label:.4f}"):
                    st.write(chunk["text"])

with tab_results:
    st.subheader("Precision@5 — Before vs. After Fine-Tuning")
    ba_path = f"{config.RESULTS_DIR}/before_after_finetuning.json"
    if os.path.exists(ba_path):
        with open(ba_path) as f:
            ba = json.load(f)
        c1, c2, c3 = st.columns(3)
        c1.metric("Baseline P@5", f"{ba['baseline_precision_at_5']:.3f}")
        c2.metric("Fine-tuned P@5", f"{ba['finetuned_precision_at_5']:.3f}")
        c3.metric("Delta", f"{ba['delta']:+.3f}")
    else:
        st.info("Run `python src/evaluate_retrieval.py` to generate this comparison.")

    st.subheader("Chunk-Size Ablation")
    ab_path = f"{config.RESULTS_DIR}/chunk_size_ablation.json"
    if os.path.exists(ab_path):
        with open(ab_path) as f:
            ab = json.load(f)
        st.bar_chart(ab["results"])
        st.write(f"Best chunk size: **{ab['best_chunk_size']}**")
    else:
        st.info("Run `python src/evaluate_retrieval.py::run_chunk_size_ablation` to generate this.")

    st.subheader("RAGAs Scores (baseline vs. fine-tuned)")
    ragas_path = f"{config.RESULTS_DIR}/ragas_comparison.json"
    if os.path.exists(ragas_path):
        with open(ragas_path) as f:
            ragas_results = json.load(f)
        st.table(ragas_results)
    else:
        st.info("Run `python src/evaluate_ragas.py` to generate this comparison.")
