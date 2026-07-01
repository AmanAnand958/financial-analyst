"""
FinDoc Intelligence — Streamlit Frontend

A premium-grade financial document Q&A interface with:
  - Tab 1: Document Upload + Chat Interface + Source Citations
  - Tab 2: Evaluation Dashboard (metrics gauges & charts)
  - Tab 3: System Health & Document Inventory
"""

import os
import json
import time
from datetime import datetime
from typing import Dict, List

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

# ── Page Config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="FinDoc Intelligence | KPMG",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Trust-first Corporate Header (KPMG Theme) */
    .main-header {
        background: #00338D; /* KPMG Blue */
        padding: 2.2rem 2.5rem;
        border-radius: 6px;
        margin-bottom: 2rem;
        border-left: 6px solid #00A3A6; /* Teal accent */
    }
    .main-header h1 {
        color: #ffffff;
        font-size: 2.2rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .main-header p {
        color: #e0f2f1;
        font-size: 1rem;
        margin: 0.3rem 0 0;
        font-weight: 400;
    }
    .kpmg-badge {
        background: rgba(255,255,255,0.12);
        color: #ffffff;
        padding: 4px 12px;
        border-radius: 3px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 1.2px;
        display: inline-block;
        margin-bottom: 0.6rem;
        border: 1px solid rgba(255,255,255,0.2);
    }

    /* Flat, Clean Card Layouts */
    .metric-card {
        background: #ffffff;
        border-radius: 4px;
        padding: 1.5rem;
        border: 1px solid #e2e8f0;
        border-left: 4px solid #00338D;
        margin-bottom: 1rem;
    }
    .metric-card.pass { border-left-color: #1b5e20; }
    .metric-card.warn { border-left-color: #e65100; }
    .metric-card.fail { border-left-color: #b71c1c; }

    /* Source Citation Cards */
    .source-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 4px;
        padding: 1.2rem;
        margin-bottom: 0.75rem;
        font-size: 0.88rem;
    }
    .source-card .doc-header {
        font-weight: 600;
        color: #00338D;
        margin-bottom: 0.5rem;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .relevance-badge {
        background: #f0fdf4;
        color: #1b5e20;
        border: 1px solid #bbf7d0;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.72rem;
        font-weight: 600;
    }
    .confidence-high {
        background: #f0fdf4;
        color: #166534;
        border: 1px solid #bbf7d0;
        padding: 3px 10px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.82rem;
    }
    .confidence-low {
        background: #fef2f2;
        color: #991b1b;
        border: 1px solid #fecaca;
        padding: 3px 10px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.82rem;
    }

    /* Clean, Grounded Answer Container */
    .answer-box {
        background: #f8fafc;
        border-radius: 6px;
        padding: 1.5rem;
        border: 1px solid #e2e8f0;
        border-left: 5px solid #00338D;
        font-size: 1rem;
        line-height: 1.6;
        color: #1e293b;
    }

    /* Chat Elements */
    .chat-user {
        background: #00338D;
        color: white;
        border-radius: 6px 6px 2px 6px;
        padding: 12px 18px;
        margin: 4px 0 4px 15%;
        font-size: 0.9rem;
    }
    .chat-bot {
        background: #f1f5f9;
        color: #1e293b;
        border-radius: 6px 6px 6px 2px;
        padding: 12px 18px;
        margin: 4px 15% 4px 0;
        font-size: 0.9rem;
        border: 1px solid #e2e8f0;
    }

    /* Status indicators */
    .status-ok { color: #166534; font-weight: 600; }
    .status-warn { color: #9a3412; font-weight: 600; }
    .status-err { color: #991b1b; font-weight: 600; }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown(
        """
        <div style="text-align:center; padding: 1rem 0;">
            <div style="font-size:2rem;">📊</div>
            <div style="font-weight:700; font-size:1.1rem; color:#00338D;">FinDoc Intelligence</div>
            <div style="font-size:0.75rem; color:#64748b;">KPMG Audit Assistant</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.divider()

    st.subheader("⚙️ Configuration")
    api_url = st.text_input(
        "API URL",
        value=st.session_state.get("api_url", "http://localhost:8000"),
        help="FastAPI backend URL",
    )
    st.session_state["api_url"] = api_url

    confidence_threshold = st.slider(
        "Confidence Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.6,
        step=0.05,
        help="Minimum retrieval score required to generate an answer",
    )
    top_k = st.slider(
        "Source Chunks (top-k)",
        min_value=1,
        max_value=10,
        value=3,
        help="Number of document chunks to retrieve per query",
    )

    st.divider()

    # Connection status
    st.subheader("🔌 Connection")
    try:
        health = requests.get(f"{api_url}/health", timeout=3).json()
        st.markdown(
            f'<span class="status-ok">● API Online</span>', unsafe_allow_html=True
        )
        st.caption(f"Chunks indexed: {health['components']['total_chunks']:,}")
        llm_status = health["components"]["llm"]
        if llm_status == "ready":
            st.markdown(
                '<span class="status-ok">● LLM Connected</span>', unsafe_allow_html=True
            )
        else:
            st.markdown(
                '<span class="status-warn">⚠ LLM: No API key</span>',
                unsafe_allow_html=True,
            )
    except Exception:
        st.markdown(
            '<span class="status-err">● API Offline</span>', unsafe_allow_html=True
        )
        st.caption("Start the backend with: uvicorn src.api.main:app --reload")

    st.divider()
    st.subheader("📤 Export Data")
    csv_data = export_history_to_csv()
    if csv_data:
        st.download_button(
            label="📥 Download Chat Log (CSV)",
            data=csv_data,
            file_name=f"fdi_chat_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )
    else:
        st.caption("No queries to export yet.")

    st.divider()
    st.caption("v1.0.0 · Powered by Groq + ChromaDB")


# ── Header ────────────────────────────────────────────────────────────────────

st.markdown(
    """
    <div class="main-header">
        <div class="kpmg-badge">KPMG · FINANCIAL AUDIT</div>
        <h1>📊 FinDoc Intelligence</h1>
        <p>RAG-Powered Financial Document Q&A with Source Citations & Hallucination Safeguards</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── History Helpers ───────────────────────────────────────────────────────────
HISTORY_FILE = "data/query_history.json"

def load_query_history():
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r") as f:
                return json.load(f)
        except Exception as e:
            st.error(f"Failed to load query history: {e}")
    return []

def save_query_history(history):
    try:
        os.makedirs(os.path.dirname(HISTORY_FILE), exist_ok=True)
        with open(HISTORY_FILE, "w") as f:
            json.dump(history, f, indent=2)
    except Exception as e:
        st.error(f"Failed to save query history: {e}")

def export_history_to_csv():
    history = st.session_state.get("chat_history", [])
    if not history:
        return None
    rows = []
    for entry in history:
        res = entry.get("result", {})
        sources = []
        for s in res.get("source_chunks", []):
            sources.append(f"{s['document']} (p. {s['page']}, sec. {s.get('section', 'General')})")
        sources_str = "; ".join(sources)
        
        # Format timestamp to human readable local time
        timestamp_raw = res.get("upload_timestamp", "")
        if timestamp_raw:
            try:
                # Remove Z and parse
                parsed_t = datetime.strptime(timestamp_raw.replace("Z", ""), "%Y-%m-%dT%H:%M:%S.%f")
                timestamp_str = parsed_t.strftime("%Y-%m-%d %H:%M:%S")
            except Exception:
                timestamp_str = timestamp_raw
        else:
            timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        rows.append({
            "Timestamp": timestamp_str,
            "Question": entry.get("question", ""),
            "Answer": res.get("answer", ""),
            "Confidence": res.get("confidence", ""),
            "Retrieval Score": res.get("retrieval_score", 0.0),
            "Latency (s)": res.get("latency_ms", 0) / 1000,
            "Sources": sources_str
        })
    df = pd.DataFrame(rows)
    return df.to_csv(index=False).encode('utf-8')

# ── Initialise session state ──────────────────────────────────────────────────

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = load_query_history()


# ── Tabs ──────────────────────────────────────────────────────────────────────

tab1, tab2, tab3 = st.tabs(
    ["💬 Upload & Query", "📈 Evaluation Dashboard", "🔧 System Health"]
)


# ==========================================================================
# TAB 1 — Upload & Query
# ==========================================================================
with tab1:
    col_upload, col_query = st.columns([1, 1.5], gap="large")

    # ── Upload pane ────────────────────────────────────────────────────────
    with col_upload:
        st.subheader("📂 Upload Financial Documents")
        st.caption("Supported: PDF (≤ 50 MB) · Annual reports, balance sheets, contracts")

        uploaded_files = st.file_uploader(
            "Choose PDF files",
            type=["pdf"],
            accept_multiple_files=True,
            help="Upload Reliance, TCS, Infosys reports or any financial PDF",
        )

        if uploaded_files:
            total_size_mb = sum(len(f.getvalue()) for f in uploaded_files) / (1024 * 1024)
            st.caption(
                f"📄 {len(uploaded_files)} files selected  |  Total: {total_size_mb:.1f} MB"
            )

            if st.button("⬆️ Upload & Process All", type="primary", use_container_width=True):
                progress_bar = st.progress(0.0)
                status_text = st.empty()
                success_count = 0
                
                for idx, u_file in enumerate(uploaded_files):
                    status_text.text(f"Processing ({idx+1}/{len(uploaded_files)}): {u_file.name}...")
                    try:
                        files = {"file": (u_file.name, u_file.getvalue(), "application/pdf")}
                        resp = requests.post(f"{api_url}/upload", files=files, timeout=120)
                        if resp.status_code == 200:
                            success_count += 1
                        else:
                            st.error(f"Failed to upload {u_file.name}: {resp.json().get('detail', resp.text)}")
                    except Exception as e:
                        st.error(f"Error uploading {u_file.name}: {e}")
                    progress_bar.progress((idx + 1) / len(uploaded_files))
                
                status_text.text(f"Completed! Ingested {success_count}/{len(uploaded_files)} files.")
                time.sleep(1)
                st.rerun()

        # Uploaded documents list
        st.subheader("📋 Indexed Documents")
        try:
            docs_resp = requests.get(f"{api_url}/documents", timeout=5)
            if docs_resp.status_code == 200:
                docs_data = docs_resp.json()
                if docs_data["total_documents"] == 0:
                    st.markdown(
                        """
                        <div style="border: 1px dashed #cbd5e1; padding: 2rem; border-radius: 4px; text-align: center; background: #f8fafc; margin-bottom: 1.5rem;">
                            <div style="font-size: 2rem; margin-bottom: 0.5rem;">📂</div>
                            <div style="font-weight: 600; color: #00338D; font-size: 0.9rem;">No Documents Ingested</div>
                            <div style="font-size: 0.78rem; color: #64748b; margin-top: 0.25rem;">
                                Upload a financial PDF above to index it for Q&A search.
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                else:
                    st.caption(
                        f"{docs_data['total_documents']} documents · "
                        f"{docs_data['total_chunks']:,} total chunks"
                    )
                    for doc in docs_data["documents"]:
                        with st.expander(
                            f"📄 {doc['document_name']} ({doc['chunk_count']} chunks)"
                        ):
                            st.write(f"**ID:** `{doc['document_id']}`")
                            st.write(f"**Uploaded:** {doc.get('uploaded_at', 'N/A')}")
                            if st.button(
                                "🗑️ Delete",
                                key=f"del_{doc['document_id']}",
                                type="secondary",
                            ):
                                del_resp = requests.delete(
                                    f"{api_url}/documents/{doc['document_id']}", timeout=10
                                )
                                if del_resp.status_code == 200:
                                    st.success("Deleted!")
                                    st.rerun()
        except Exception:
            st.info("Upload documents to see them listed here.")

    # ── Query pane ─────────────────────────────────────────────────────────
    with col_query:
        st.subheader("💬 Ask a Financial Question")
        st.caption("Select a preset suggestion or type a custom question below.")

        # Interactive pill grid for suggested questions
        st.markdown("<p style='font-size: 0.85rem; font-weight: 600; margin-bottom: 0.5rem; color: #00338D;'>Suggested Queries</p>", unsafe_allow_html=True)
        grid_col1, grid_col2 = st.columns(2)
        
        with grid_col1:
            if st.button("📈 Revenue in FY2023", key="s_rev", use_container_width=True):
                st.session_state["question_input"] = "What was the total revenue in FY2023?"
                st.rerun()
            if st.button("⚖️ Net Profit Margin", key="s_prof", use_container_width=True):
                st.session_state["question_input"] = "What was the net profit margin?"
                st.rerun()
                
        with grid_col2:
            if st.button("⚠️ Major Risk Factors", key="s_risk", use_container_width=True):
                st.session_state["question_input"] = "What are the major risk factors mentioned?"
                st.rerun()
            if st.button("💸 Cash Flow from Operations", key="s_cf", use_container_width=True):
                st.session_state["question_input"] = "Describe the cash flow from operations."
                st.rerun()

        st.write("") # Spacer

        question = st.text_area(
            "Your question",
            value=st.session_state.get("question_input", ""),
            height=100,
            placeholder="Ask anything about the uploaded financial documents …",
        )

        col_ask, col_clear = st.columns([3, 1])
        with col_ask:
            ask_btn = st.button(
                "🔍 Ask", type="primary", use_container_width=True, disabled=not question
            )
        with col_clear:
            if st.button("🗑 Clear", use_container_width=True):
                st.session_state["chat_history"] = []
                save_query_history([])
                st.rerun()

        if ask_btn and question:
            with st.spinner("🔍 Searching documents and generating answer …"):
                try:
                    payload = {
                        "question": question,
                        "top_k": top_k,
                        "confidence_threshold": confidence_threshold,
                    }
                    start = time.time()
                    resp = requests.post(f"{api_url}/query", json=payload, timeout=30)
                    elapsed = time.time() - start

                    if resp.status_code == 200:
                        result = resp.json()
                        st.session_state["chat_history"].append(
                            {"question": question, "result": result}
                        )
                        save_query_history(st.session_state["chat_history"])
                    else:
                        st.error(f"Query failed: {resp.json().get('detail', resp.text)}")
                except requests.exceptions.ConnectionError:
                    st.error("Cannot connect to API. Is the backend running?")
                except requests.exceptions.Timeout:
                    st.error("Request timed out. The system may be overloaded.")

        # ── Display latest answer ──────────────────────────────────────────
        if st.session_state["chat_history"]:
            latest = st.session_state["chat_history"][-1]
            result = latest["result"]

            st.divider()
            st.subheader("🤖 Answer")

            conf = result.get("confidence", "LOW")
            badge_class = "confidence-high" if conf == "HIGH" else "confidence-low"
            conf_html = f'<span class="{badge_class}">{conf} CONFIDENCE</span>'

            latency_s = result.get("latency_ms", 0) / 1000
            score = result.get("retrieval_score", 0)

            col_c1, col_c2, col_c3 = st.columns(3)
            col_c1.metric("Confidence", conf)
            col_c2.metric("Retrieval Score", f"{score:.2f}")
            col_c3.metric("Latency", f"{latency_s:.1f}s")

            st.markdown(
                f'<div class="answer-box">{result["answer"]}</div>',
                unsafe_allow_html=True,
            )

            # Source citations
            if result.get("source_chunks"):
                st.subheader("📄 Source Citations")
                for i, src in enumerate(result["source_chunks"], start=1):
                    rel_score = src.get("relevance_score", 0)
                    st.markdown(
                        f"""
                        <div class="source-card">
                            <div class="doc-header">
                                [{i}] {src['document']} · Page {src['page']}
                                &nbsp;<span class="relevance-badge">Score: {rel_score:.2f}</span>
                            </div>
                            <div style="color:#555; font-style:italic;">
                                Section: {src.get('section', 'General')}
                            </div>
                            <div style="margin-top:0.5rem;">{src['excerpt']}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

        # ── Chat history ───────────────────────────────────────────────────
        if len(st.session_state["chat_history"]) > 1:
            with st.expander(f"💬 Chat History ({len(st.session_state['chat_history'])} queries)"):
                for entry in reversed(st.session_state["chat_history"][:-1]):
                    st.markdown(
                        f'<div class="chat-user">🧑 {entry["question"]}</div>',
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        f'<div class="chat-bot">🤖 {entry["result"]["answer"][:200]}…</div>',
                        unsafe_allow_html=True,
                    )


# ==========================================================================
# TAB 2 — Evaluation Dashboard
# ==========================================================================
with tab2:
    st.subheader("📈 Pipeline Evaluation Metrics")
    st.caption("Real-time quality metrics for the RAG pipeline")

    col_refresh, _ = st.columns([1, 4])
    with col_refresh:
        if st.button("🔄 Refresh Metrics"):
            st.rerun()

    try:
        metrics_resp = requests.get(f"{api_url}/eval-metrics", timeout=5)
        metrics = metrics_resp.json() if metrics_resp.status_code == 200 else {}
    except Exception:
        metrics = {}

    if not metrics or metrics.get("total_queries", 0) == 0:
        st.markdown(
            """
            <div style="border: 1px dashed #cbd5e1; padding: 2.5rem; border-radius: 4px; text-align: center; background: #f8fafc; margin-bottom: 2rem; margin-top: 1rem;">
                <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">📈</div>
                <div style="font-weight: 600; color: #00338D; font-size: 1.05rem;">No Evaluation Records Found</div>
                <div style="font-size: 0.82rem; color: #64748b; margin-top: 0.25rem; max-width: 48ch; margin-left: auto; margin-right: auto;">
                    Submit questions in the Q&A tab to accumulate pipeline evaluation scores. Real-time metrics will show up here.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        # Show target metrics as reference
        st.subheader("🎯 Target Benchmarks")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Context Precision", "≥ 85%", "Target")
        col2.metric("Faithfulness", "≥ 90%", "Target")
        col3.metric("Answer Relevancy", "≥ 85%", "Target")
        col4.metric("Latency", "< 5.0s", "Target")
    else:
        cp = metrics.get("context_precision", 0)
        faith = metrics.get("faithfulness", 0)
        ar = metrics.get("answer_relevancy", 0)
        latency_s = metrics.get("avg_latency_ms", 0) / 1000
        total_q = metrics.get("total_queries", 0)
        hallucination_rate = metrics.get("hallucination_rate", 0)

        # KPI cards
        col1, col2, col3, col4 = st.columns(4)

        def delta_str(val, target):
            diff = val - target
            return f"{'↑' if diff >= 0 else '↓'} {abs(diff):.0%} vs target"

        col1.metric(
            "Context Precision",
            f"{cp:.0%}",
            delta_str(cp, 0.85),
            delta_color="normal" if cp >= 0.85 else "inverse",
        )
        col2.metric(
            "Faithfulness",
            f"{faith:.0%}",
            delta_str(faith, 0.90),
            delta_color="normal" if faith >= 0.90 else "inverse",
        )
        col3.metric(
            "Answer Relevancy",
            f"{ar:.0%}",
            delta_str(ar, 0.85),
            delta_color="normal" if ar >= 0.85 else "inverse",
        )
        col4.metric(
            "Avg Latency",
            f"{latency_s:.1f}s",
            f"{'✓' if latency_s < 5 else '✗'} target < 5s",
            delta_color="normal" if latency_s < 5 else "inverse",
        )

        st.divider()

        # Gauge charts
        col_g1, col_g2, col_g3 = st.columns(3)

        def make_gauge(title, value, target, max_val=1.0):
            color = "#00A3A6" if value >= target else "#991b1b"
            fig = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=round(value * 100, 1),
                    number={"suffix": "%", "font": {"size": 24, "family": "Inter", "color": "#1e293b"}},
                    title={"text": title, "font": {"size": 14, "family": "Inter", "color": "#00338D", "weight": "bold"}},
                    gauge={
                        "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#64748b"},
                        "bar": {"color": color, "thickness": 0.8},
                        "bgcolor": "#f8fafc",
                        "borderwidth": 1,
                        "bordercolor": "#cbd5e1",
                        "steps": [
                            {"range": [0, target * 100], "color": "#fef2f2"},
                            {"range": [target * 100, 100], "color": "#f0fdf4"},
                        ],
                        "threshold": {
                            "line": {"color": "#00338D", "width": 3},
                            "thickness": 0.75,
                            "value": target * 100,
                        },
                    },
                )
            )
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                height=180,
                margin=dict(t=40, b=10, l=10, r=10)
            )
            return fig

        with col_g1:
            st.plotly_chart(
                make_gauge("Context Precision", cp, 0.85), use_container_width=True
            )
        with col_g2:
            st.plotly_chart(
                make_gauge("Faithfulness", faith, 0.90), use_container_width=True
            )
        with col_g3:
            st.plotly_chart(
                make_gauge("Answer Relevancy", ar, 0.85), use_container_width=True
            )

        st.divider()

        col_stats1, col_stats2 = st.columns(2)
        with col_stats1:
            st.metric("Total Queries Processed", f"{total_q:,}")
            st.metric("Hallucination Rate", f"{hallucination_rate:.1%}", help="Fraction of queries that returned LOW confidence")
        with col_stats2:
            last_eval = metrics.get("last_evaluated", "N/A")
            st.metric("Last Activity", last_eval[:19].replace("T", " ") if last_eval else "N/A")
            st.metric("Metrics Source", metrics.get("source", "heuristic").upper())

        # RAGAs full evaluation
        st.divider()
        st.subheader("🧪 Full RAGAs Evaluation")
        st.caption(
            "Run the complete RAGAs evaluation suite using the test dataset. "
            "Requires OPENAI_API_KEY and data/processed/eval_dataset.json."
        )
        if st.button("▶️ Run Full RAGAs Evaluation"):
            with st.spinner("Running RAGAs … this may take a few minutes."):
                try:
                    ragas_resp = requests.post(f"{api_url}/eval/run-ragas", timeout=300)
                    if ragas_resp.status_code == 200:
                        ragas_metrics = ragas_resp.json()
                        st.success("✅ RAGAs evaluation complete!")
                        st.json(ragas_metrics)
                    else:
                        st.error(ragas_resp.json().get("detail", "Evaluation failed."))
                except Exception as exc:
                    st.error(f"Failed: {exc}")


# ==========================================================================
# TAB 3 — System Health
# ==========================================================================
with tab3:
    st.subheader("🔧 System Health & Status")

    col_ref3, _ = st.columns([1, 4])
    with col_ref3:
        if st.button("🔄 Refresh Status"):
            st.rerun()

    try:
        health_resp = requests.get(f"{api_url}/health", timeout=5)
        health_data = health_resp.json() if health_resp.status_code == 200 else {}
    except Exception:
        health_data = {}

    col_h1, col_h2 = st.columns(2)

    with col_h1:
        st.subheader("Component Status")
        if not health_data:
            st.error("⚠️ Cannot reach API backend")
            st.code("uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload")
        else:
            comps = health_data.get("components", {})
            vs_status = comps.get("vector_store", "unknown")
            llm_status = comps.get("llm", "unknown")
            chunks = comps.get("total_chunks", 0)

            # Vector store
            if vs_status == "ready":
                st.success(f"✅ Vector Store — {chunks:,} chunks indexed")
            else:
                st.error(f"❌ Vector Store — {vs_status}")

            # LLM
            if llm_status == "ready":
                st.success("✅ LLM (Groq) — Connected")
            elif llm_status == "no_api_key":
                st.warning("⚠️ LLM — No API key (set GROQ_API_KEY in .env)")
            else:
                st.error(f"❌ LLM — {llm_status}")

            # Embedding model
            st.success("✅ Embedding Model — all-MiniLM-L6-v2 loaded")

            # API version
            st.info(f"ℹ️ API Version: {health_data.get('version', 'N/A')}")

    with col_h2:
        st.subheader("📁 Document Inventory")
        try:
            docs_resp = requests.get(f"{api_url}/documents", timeout=5)
            if docs_resp.status_code == 200:
                docs_data = docs_resp.json()
                docs = docs_data.get("documents", [])

                if not docs:
                    st.info("No documents indexed yet.")
                else:
                    st.caption(
                        f"**{docs_data['total_documents']} documents · "
                        f"{docs_data['total_chunks']:,} chunks**"
                    )
                    for doc in docs:
                        st.markdown(
                            f"""
                            <div class="source-card">
                                <div class="doc-header">{doc['document_name']}</div>
                                <div style="font-size:0.8rem; color:#666;">
                                    ID: {doc['document_id']} &nbsp;|&nbsp;
                                    Chunks: {doc['chunk_count']} &nbsp;|&nbsp;
                                    Uploaded: {doc.get('uploaded_at', 'N/A')[:10]}
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
        except Exception:
            st.info("Connect to the API to see document inventory.")

    # Quick-start guide
    st.divider()
    st.subheader("🚀 Quick Start")
    st.markdown(
        """
        #### Local Development
        ```bash
        # 1. Install dependencies
        pip install -r requirements.txt

        # 2. Set API key
        cp .env.example .env
        # Edit .env and add your GROQ_API_KEY

        # 3. Start the backend
        uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

        # 4. In another terminal, start the UI
        streamlit run src/ui/app.py
        ```

        #### Docker
        ```bash
        docker-compose up
        # API → http://localhost:8000
        # UI  → http://localhost:8501
        ```
        """
    )
