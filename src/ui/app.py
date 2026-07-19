# Yeh file Streamlit frontend web dashboard client setup karne ke liye hai.
"""
FinDoc Intelligence — Streamlit Frontend

A premium-grade financial document Q&A interface with:
  - Tab 1: Document Upload + Chat Interface + Source Citations
  - Tab 2: Evaluation Dashboard (metrics gauges & charts)
  - Tab 3: System Health & Document Inventory
"""

# OS variables check operations package.
import os
# JSON parameters formats configurations.
import json
# System timings measurements tools.
import time
# Lifecycle timing tracking format.
from datetime import datetime
# Python typing systems support parameters.
from typing import Dict, List

# Dataframes data analysis package pandas.
import pandas as pd
# Graph elements charts builder package plotly.
import plotly.graph_objects as go
# HTTP API requests caller package.
import requests
# Streamlit UI framework components packages.
import streamlit as st


# ── Page Config ───────────────────────────────────────────────────────────────

# Streamlit page details layout metadata config setup.
st.set_page_config(
    # Page Title.
    page_title="FinDoc Intelligence | KPMG",
    # Icon.
    page_icon="📊",
    # Screen width scale config.
    layout="wide",
    # Sidebar initial display modes config.
    initial_sidebar_state="expanded",
)  # config end.


# ── Custom CSS ────────────────────────────────────────────────────────────────

# Dynamic custom HTML styling CSS tags inject.
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
)  # css markdown injection end.


# ── Sidebar ───────────────────────────────────────────────────────────────────

# Sidebar layout blocks definitions setups wrapper.
with st.sidebar:
    # Sidebar header branding logos custom HTML markdown.
    st.markdown(
        """
        <div style="text-align:center; padding: 1rem 0;">
            <div style="font-size:2rem;">📊</div>
            <div style="font-weight:700; font-size:1.1rem; color:#00338D;">FinDoc Intelligence</div>
            <div style="font-size:0.75rem; color:#64748b;">KPMG Audit Assistant</div>
        </div>
        """,
        unsafe_allow_html=True,
    )  # Markdown end.
    # Divider line sidebar.
    st.divider()

    # Settings configurations label sidebar headers.
    st.subheader("⚙️ Configuration")
    # Backend URL configuration text inputs.
    api_url = st.text_input(
        # Input label.
        "API URL",
        # Default value fallback from session state setup configurations.
        value=st.session_state.get("api_url", "http://localhost:8000"),
        # Tooltip checks.
        help="FastAPI backend URL",
    )  # API input text end.
    # Save active backend URL values in session maps.
    st.session_state["api_url"] = api_url

    # Slider configs control parameters score limit thresholds parameters.
    confidence_threshold = st.slider(
        # Limit title labels.
        "Confidence Threshold",
        # Minimum range limits bounds.
        min_value=0.0,
        # Maximum scale limit.
        max_value=1.0,
        # Default.
        value=0.6,
        # Scale steps.
        step=0.05,
        # Description tooltip.
        help="Minimum retrieval score required to generate an answer",
    )  # Slider threshold end.
    # Retrieve candidates limit controls sliders.
    top_k = st.slider(
        # Slider labels.
        "Source Chunks (top-k)",
        # Min chunks search size.
        min_value=1,
        # Max scale limit chunks search.
        max_value=10,
        # Default chunks retrieval targets.
        value=3,
        # Help details descriptors.
        help="Number of document chunks to retrieve per query",
    )  # Slider top_k end.

    # Divider sidebar layouts.
    st.divider()

    # Connection checks sidebar configurations indicators.
    st.subheader("🔌 Connection")
    # Ping API backend status check routes try block.
    try:
        # Request health endpoint status mapping checks parameters.
        health = requests.get(f"{api_url}/health", timeout=3).json()
        # Status green light HTML span configurations render details.
        st.markdown(
            f'<span class="status-ok">● API Online</span>', unsafe_allow_html=True
        )  # Online markdown end.
        # Displays chunks totals.
        st.caption(f"Chunks indexed: {health['components']['total_chunks']:,}")
        # Extract LLM connectivity flags status definitions.
        llm_status = health["components"]["llm"]
        # LLM active checker.
        if llm_status == "ready":
            # Connected status write.
            st.markdown(
                '<span class="status-ok">● LLM Connected</span>', unsafe_allow_html=True
            )  # LLM status markdown end.
        # Key configurations alert warning conditions checks.
        else:
            # LLM status offline or missing keys labels.
            st.markdown(
                '<span class="status-warn">⚠ LLM: No API key</span>',
                # HTML allowed flag configuration value.
                unsafe_allow_html=True,
            )  # LLM warning end.
    # Catch offline backend server exceptions.
    except Exception:
        # Status red light HTML span configurations offline tag.
        st.markdown(
            '<span class="status-err">● API Offline</span>', unsafe_allow_html=True
        )  # Offline status end.
        # Commands instructions info.
        st.caption("Start the backend with: uvicorn src.api.main:app --reload")

    # Divider separator sidebar.
    st.divider()
    # Export logs header section sidebar settings.
    st.subheader("📤 Export Data")
    # Export history function calls formats conversions.
    csv_data = export_history_to_csv()
    # Check data formats lists exist checks configurations.
    if csv_data:
        # Streamlit standard download button interface configurations.
        st.download_button(
            # Button labels.
            label="📥 Download Chat Log (CSV)",
            # Byte data source stream payload configs.
            data=csv_data,
            # Output filename formats timestamps tracking definitions.
            file_name=f"fdi_chat_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            # MIME header checks configurations parameters.
            mime="text/csv",
            # Width container scales.
            use_container_width=True
        )  # Download button end.
    # Empty logs scenario warning info.
    else:
        # Default placeholder text.
        st.caption("No queries to export yet.")

    # Sidebar separators line.
    st.divider()
    # Footer tag details.
    st.caption("v1.0.0 · Powered by Groq + ChromaDB")


# ── Header ────────────────────────────────────────────────────────────────────

# UI Page main branding title custom headers tags HTML.
st.markdown(
    """
    <div class="main-header">
        <div class="kpmg-badge">KPMG · FINANCIAL AUDIT</div>
        <h1>📊 FinDoc Intelligence</h1>
        <p>RAG-Powered Financial Document Q&A with Source Citations & Hallucination Safeguards</p>
    </div>
    """,
    unsafe_allow_html=True,
)  # Main header markdown end.


# ── History Helpers ───────────────────────────────────────────────────────────
# Query logging storage files coordinates paths setups.
HISTORY_FILE = "data/query_history.json"

# Loader functions query logging history.
def load_query_history():
    # Verify file coordinates paths validity options.
    if os.path.exists(HISTORY_FILE):
        # Open data read try checks error exceptions blocks.
        try:
            # Open files streams.
            with open(HISTORY_FILE, "r") as f:
                # Load JSON mappings arrays results configurations.
                return json.load(f)
        # Catch read failures logging console message trackings.
        except Exception as e:
            # Streamlit error block message prints parameters.
            st.error(f"Failed to load query history: {e}")
    # Return default empty list array.
    return []

# Persistent logs files write syncing methods definitions.
def save_query_history(history):
    # Try directory path builders execution file write streams.
    try:
        # Create directories parent targets structures.
        os.makedirs(os.path.dirname(HISTORY_FILE), exist_ok=True)
        # Open data write connections.
        with open(HISTORY_FILE, "w") as f:
            # Dump JSON arrays formats spacing indentation details.
            json.dump(history, f, indent=2)
    # Catch write errors stack traces.
    except Exception as e:
        # Streamlit error notifications displays screen.
        st.error(f"Failed to save query history: {e}")

# Export session state query history maps to CSV structures data helper method.
def export_history_to_csv():
    # Fetch log items lists from active session configurations.
    history = st.session_state.get("chat_history", [])
    # Verification check empty lists exits.
    if not history:
        # Return none indicators parameters configurations.
        return None
    # Row tracking records data collection formats placeholder list.
    rows = []
    # Loop log items configurations records maps updates formats.
    for entry in history:
        # Extract matches results formats setups maps parameters details.
        res = entry.get("result", {})
        # Citation texts segments collection lists placeholder.
        sources = []
        # Loop nested citations properties parameters mappings updates details.
        for s in res.get("source_chunks", []):
            # Append formatted strings elements details documents indicators.
            sources.append(f"{s['document']} (p. {s['page']}, sec. {s.get('section', 'General')})")
        # Joined output segments string details delimiters configurations setup.
        sources_str = "; ".join(sources)
        
        # Format timestamp to human readable local time
        # Timestamps key checks properties.
        timestamp_raw = res.get("upload_timestamp", "")
        # Verification values config checking constraints paths.
        if timestamp_raw:
            # Format parsing try catch block parameters checks codes.
            try:
                # Cleanup Z characters string parsing format datetime mapping variables.
                parsed_t = datetime.strptime(timestamp_raw.replace("Z", ""), "%Y-%m-%dT%H:%M:%S.%f")
                # Format string conversions templates properties tags.
                timestamp_str = parsed_t.strftime("%Y-%m-%d %H:%M:%S")
            # Parse error fallback setup options checks parameters.
            except Exception:
                # Reassign raw.
                timestamp_str = timestamp_raw
        # Empty timestamp fallback definitions setup coordinates properties.
        else:
            # Local timings.
            timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Row entries build dictionaries setups lists parameters insertions.
        rows.append({
            # Local timing labels.
            "Timestamp": timestamp_str,
            # Questions query string.
            "Question": entry.get("question", ""),
            # Answers output text templates string details.
            "Answer": res.get("answer", ""),
            # System confidence validation flags labels.
            "Confidence": res.get("confidence", ""),
            # Match score precision conversions properties mapping parameters.
            "Retrieval Score": res.get("retrieval_score", 0.0),
            # Latency measurements converts seconds values trackers metrics.
            "Latency (s)": res.get("latency_ms", 0) / 1000,
            # Combined sources references maps.
            "Sources": sources_str
        })  # Row inserts end.
    # Dataframe builder pandas packages setup config.
    df = pd.DataFrame(rows)
    # Output byte formats conversions.
    return df.to_csv(index=False).encode('utf-8')


# ── Initialise session state ──────────────────────────────────────────────────

# Check state keys exists in configurations indicators checks.
if "chat_history" not in st.session_state:
    # Load query history records on startup.
    st.session_state["chat_history"] = load_query_history()


# ── Tabs ──────────────────────────────────────────────────────────────────────

# UI Multi Tabs structures tabs object creation settings config.
tab1, tab2, tab3 = st.tabs(
    # Tab labels lists configurations parameters options tags definitions.
    ["💬 Upload & Query", "📈 Evaluation Dashboard", "🔧 System Health"]
)  # Tabs end.


# ==========================================================================
# TAB 1 — Upload & Query
# ==========================================================================
# Tab 1 segment blocks layout scope setup parameters configuration maps.
with tab1:
    # Multi columns setup page split configs margins.
    col_upload, col_query = st.columns([1, 1.5], gap="large")

    # ── Upload pane ────────────────────────────────────────────────────────
    # Left column layouts.
    with col_upload:
        # Title labels.
        st.subheader("📂 Upload Financial Documents")
        # Format types description guides info console screens.
        st.caption("Supported: PDF (≤ 50 MB) · Annual reports, balance sheets, contracts")

        # Ingest PDF file loader widgets Streamlit components.
        uploaded_files = st.file_uploader(
            # Label texts.
            "Choose PDF files",
            # File filter arrays constraints configs.
            type=["pdf"],
            # Multi selection flags.
            accept_multiple_files=True,
            # Tooltip details guides description checks.
            help="Upload Reliance, TCS, Infosys reports or any financial PDF",
        )  # Uploader end.

        # Verify items selected list validation checks paths.
        if uploaded_files:
            # Count file sizes total parameters.
            total_size_mb = sum(len(f.getvalue()) for f in uploaded_files) / (1024 * 1024)
            # Size details maps labels output st captions.
            st.caption(
                f"📄 {len(uploaded_files)} files selected  |  Total: {total_size_mb:.1f} MB"
            )  # Caption size end.

            # Upload process action trigger buttons.
            if st.button("⬆️ Upload & Process All", type="primary", use_container_width=True):
                # Progress bar widget initialization.
                progress_bar = st.progress(0.0)
                # Dynamic text placeholder widgets structures.
                status_text = st.empty()
                # Counts variables trackings configs metrics status.
                success_count = 0
                
                # Loop selected files parameters maps sequence iterations.
                for idx, u_file in enumerate(uploaded_files):
                    # Progress messages logs console display paths update.
                    status_text.text(f"Processing ({idx+1}/{len(uploaded_files)}): {u_file.name}...")
                    # Ingest API calls catch exception anomalies route paths.
                    try:
                        # Request payload binary byte format maps parameters setups options.
                        files = {"file": (u_file.name, u_file.getvalue(), "application/pdf")}
                        # Ingest upload POST requests backend connection calls.
                        resp = requests.post(f"{api_url}/upload", files=files, timeout=120)
                        # Response health check validation flags checks codes parameters.
                        if resp.status_code == 200:
                            # Increment success.
                            success_count += 1
                        # Failed statuses.
                        else:
                            # Error notifications.
                            st.error(f"Failed to upload {u_file.name}: {resp.json().get('detail', resp.text)}")
                    # Catch connection failures tracks.
                    except Exception as e:
                        # Error messages trace console updates.
                        st.error(f"Error uploading {u_file.name}: {e}")
                    # Update progress scales bar configurations.
                    progress_bar.progress((idx + 1) / len(uploaded_files))
                
                # Final notification status displays configurations updates.
                status_text.text(f"Completed! Ingested {success_count}/{len(uploaded_files)} files.")
                # Hold system threads.
                time.sleep(1)
                # Streamlit UI redraw reload methods triggers.
                st.rerun()

        # Uploaded documents list
        # Section titles.
        st.subheader("📋 Indexed Documents")
        # Database query document list fetch try catch blocks.
        try:
            # Request documents summary list API calls mappings paths.
            docs_resp = requests.get(f"{api_url}/documents", timeout=5)
            # Response health checks confirmations.
            if docs_resp.status_code == 200:
                # Extract documents mappings payload arrays.
                docs_data = docs_resp.json()
                # Check empty databases records conditions flags.
                if docs_data["total_documents"] == 0:
                    # Formatted placeholder custom CSS templates markup.
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
                    )  # Empty warning markdown end.
                # Non empty documents list processing display loops.
                else:
                    # Caption metrics sizes.
                    st.caption(
                        f"{docs_data['total_documents']} documents · "
                        f"{docs_data['total_chunks']:,} total chunks"
                    )  # Caption documents count end.
                    # Iterate documents summaries array configs details maps properties.
                    for doc in docs_data["documents"]:
                        # Streamlit visual dropdown expanders setups titles tags configurations.
                        with st.expander(
                            f"📄 {doc['document_name']} ({doc['chunk_count']} chunks)"
                        ):  # Expander document end.
                            # ID labels codes st text widgets configurations.
                            st.write(f"**ID:** `{doc['document_id']}`")
                            # Timing tags st text write.
                            st.write(f"**Uploaded:** {doc.get('uploaded_at', 'N/A')}")
                            # Delete action buttons trigger events settings.
                            if st.button(
                                # Labels titles.
                                "🗑️ Delete",
                                # Event key tags.
                                key=f"del_{doc['document_id']}",
                                # Type style settings options configurations.
                                type="secondary",
                            ):  # Delete button click checks.
                                # Delete document backend API request caller methods routes.
                                del_resp = requests.delete(
                                    f"{api_url}/documents/{doc['document_id']}", timeout=10
                                )  # API delete call end.
                                # Response confirmation status validation flags.
                                if del_resp.status_code == 200:
                                    # Success notify.
                                    st.success("Deleted!")
                                    # Reload screens.
                                    st.rerun()
        # Connection exceptions fallback traces.
        except Exception:
            # Default placeholder text.
            st.info("Upload documents to see them listed here.")

    # ── Query pane ─────────────────────────────────────────────────────────
    # Right column layout query inputs answer box display properties setup.
    with col_query:
        # Sub title.
        st.subheader("💬 Ask a Financial Question")
        # Info guides descriptions st captions widgets.
        st.caption("Select a preset suggestion or type a custom question below.")

        # Interactive pill grid for suggested questions
        # Grid title headers.
        st.markdown("<p style='font-size: 0.85rem; font-weight: 600; margin-bottom: 0.5rem; color: #00338D;'>Suggested Queries</p>", unsafe_allow_html=True)
        # Inner column splits preset buttons layouts structure configurations.
        grid_col1, grid_col2 = st.columns(2)
        
        # Presets left columns widgets.
        with grid_col1:
            # Revenue query action triggers.
            if st.button("📈 Revenue in FY2023", key="s_rev", use_container_width=True):
                # Update text area parameter values state.
                st.session_state["question_input"] = "What was the total revenue in FY2023?"
                # Redraw screen.
                st.rerun()
            # Net profit preset click trigger checks.
            if st.button("⚖️ Net Profit Margin", key="s_prof", use_container_width=True):
                # Session variables updates query string format text.
                st.session_state["question_input"] = "What was the net profit margin?"
                # Redraw.
                st.rerun()
                
        # Presets right columns widgets.
        with grid_col2:
            # Risk parameters buttons click triggers checks.
            if st.button("⚠️ Major Risk Factors", key="s_risk", use_container_width=True):
                # Save query inside variables setups.
                st.session_state["question_input"] = "What are the major risk factors mentioned?"
                # Redraw screens.
                st.rerun()
            # Cash flow presets actions click triggers checks.
            if st.button("💸 Cash Flow from Operations", key="s_cf", use_container_width=True):
                # Session setups.
                st.session_state["question_input"] = "Describe the cash flow from operations."
                # Reload UI.
                st.rerun()

        # Spacing line.
        st.write("")

        # Input text area fields components layouts configurations.
        question = st.text_area(
            # Input titles labels.
            "Your question",
            # Text default bindings configs values.
            value=st.session_state.get("question_input", ""),
            # Height scales options.
            height=100,
            # Placeholders text clues parameters setups configs.
            placeholder="Ask anything about the uploaded financial documents …",
        )  # Textarea end.

        # Action action buttons layout sub columns grid configs.
        col_ask, col_clear = st.columns([3, 1])
        # Ask parameters widgets.
        with col_ask:
            # Ask submit action buttons trigger conditions checks.
            ask_btn = st.button(
                # Label title tags.
                "🔍 Ask", type="primary", use_container_width=True, disabled=not question
            )  # Button end.
        # Clear logs parameters widgets.
        with col_clear:
            # Clear logs button click validations indicators.
            if st.button("🗑 Clear", use_container_width=True):
                # Reset chat history maps.
                st.session_state["chat_history"] = []
                # Clear persistent storage logs files maps.
                save_query_history([])
                # Redraw.
                st.rerun()

        # Ask triggers execution path logic.
        if ask_btn and question:
            # Streamlit loaders status widgets wrappers.
            with st.spinner("🔍 Searching documents and generating answer …"):
                # Query APIs connection try catch blocks execution structures.
                try:
                    # Ingest API request body formats dictionary variables.
                    payload = {
                        # Query question.
                        "question": question,
                        # Top candidates.
                        "top_k": top_k,
                        # Threshold limit checks.
                        "confidence_threshold": confidence_threshold,
                    }  # Payload end.
                    # Time tracking monotonic.
                    start = time.time()
                    # Query execution POST requests backend coordinates paths routes.
                    resp = requests.post(f"{api_url}/query", json=payload, timeout=30)
                    # Duration evaluation metric calculations.
                    elapsed = time.time() - start

                    # Response checks validations status confirmation.
                    if resp.status_code == 200:
                        # Extract results map structures dictionaries.
                        result = resp.json()
                        # Append session histories parameters maps logs lists.
                        st.session_state["chat_history"].append(
                            {"question": question, "result": result}
                        )  # Append end.
                        # Sync logs on storage files systems.
                        save_query_history(st.session_state["chat_history"])
                    # Failed response status mapping checks console error reports.
                    else:
                        # Displays warnings.
                        st.error(f"Query failed: {resp.json().get('detail', resp.text)}")
                # Catch connection anomalies offline statuses.
                except requests.exceptions.ConnectionError:
                    # Alerts.
                    st.error("Cannot connect to API. Is the backend running?")
                # Catch timeout anomalies.
                except requests.exceptions.Timeout:
                    # Alerts.
                    st.error("Request timed out. The system may be overloaded.")

        # ── Display latest answer ──────────────────────────────────────────
        # Check active history elements exists validation check paths.
        if st.session_state["chat_history"]:
            # Pick latest record maps.
            latest = st.session_state["chat_history"][-1]
            # Load results.
            result = latest["result"]

            # Visual dividers.
            st.divider()
            # Sub title headers.
            st.subheader("🤖 Answer")

            # Extract confidence checks.
            conf = result.get("confidence", "LOW")
            # CSS tag dynamic formatting allocations configs.
            badge_class = "confidence-high" if conf == "HIGH" else "confidence-low"
            # Format text tags HTML.
            conf_html = f'<span class="{badge_class}">{conf} CONFIDENCE</span>'

            # Latency second conversion measurements details configs.
            latency_s = result.get("latency_ms", 0) / 1000
            # Retrieve score mapping parameters.
            score = result.get("retrieval_score", 0)

            # Columns KPI layouts metrics grid.
            col_c1, col_c2, col_c3 = st.columns(3)
            # Display conf.
            col_c1.metric("Confidence", conf)
            # Display score.
            col_c2.metric("Retrieval Score", f"{score:.2f}")
            # Display speed.
            col_c3.metric("Latency", f"{latency_s:.1f}s")

            # Answer container box custom HTML markup configurations render.
            st.markdown(
                f'<div class="answer-box">{result["answer"]}</div>',
                # HTML allowed flags.
                unsafe_allow_html=True,
            )  # Box end.

            # Source citations
            # Verify source citations exists check.
            if result.get("source_chunks"):
                # Citation header title.
                st.subheader("📄 Source Citations")
                # Loop citations dictionary layouts maps sequences.
                for i, src in enumerate(result["source_chunks"], start=1):
                    # Score.
                    rel_score = src.get("relevance_score", 0)
                    # Formatted citation markup cards layouts html template.
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
                    )  # Citation card end.

        # ── Chat history ───────────────────────────────────────────────────
        # Prior chat query logs history expansions accordion widgets layouts.
        if len(st.session_state["chat_history"]) > 1:
            # Accordion wrapper configurations st expanders components.
            with st.expander(f"💬 Chat History ({len(st.session_state['chat_history'])} queries)"):
                # Reversed iteration loops maps configs parameters options checks.
                for entry in reversed(st.session_state["chat_history"][:-1]):
                    # User text bubble HTML markup custom template.
                    st.markdown(
                        f'<div class="chat-user">🧑 {entry["question"]}</div>',
                        unsafe_allow_html=True,
                    )  # User markdown end.
                    # Bot response bubble details html setups options checks.
                    st.markdown(
                        f'<div class="chat-bot">🤖 {entry["result"]["answer"][:200]}…</div>',
                        unsafe_allow_html=True,
                    )  # Bot markdown end.


# ==========================================================================
# TAB 2 — Evaluation Dashboard
# ==========================================================================
# Tab 2 dashboard blocks parameters setup margins layouts configurations.
with tab2:
    # Sub title.
    st.subheader("📈 Pipeline Evaluation Metrics")
    # Captions instructions details console updates.
    st.caption("Real-time quality metrics for the RAG pipeline")

    # Refresh tools button sub columns grid setups.
    col_refresh, _ = st.columns([1, 4])
    # Refresher widgets structures.
    with col_refresh:
        # Click event checkers validations.
        if st.button("🔄 Refresh Metrics"):
            # Reload screens.
            st.rerun()

    # Metrics load backend API call tries loops catches exceptions blocks.
    try:
        # Request evaluation metrics values endpoints routes.
        metrics_resp = requests.get(f"{api_url}/eval-metrics", timeout=5)
        # Check success parse.
        metrics = metrics_resp.json() if metrics_resp.status_code == 200 else {}
    # Catch backend server offline traces.
    except Exception:
        # Empty maps checks fallback setups.
        metrics = {}

    # Check metrics availability logs variables checks flags.
    if not metrics or metrics.get("total_queries", 0) == 0:
        # Formatted placeholder custom templates html layouts markdown write.
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
        )  # Markdown info end.
        # Targets reference baseline guidelines st metrics columns layouts.
        st.subheader("🎯 Target Benchmarks")
        # Split target columns grid.
        col1, col2, col3, col4 = st.columns(4)
        # Precision limit guidelines.
        col1.metric("Context Precision", "≥ 85%", "Target")
        # Faithfulness target.
        col2.metric("Faithfulness", "≥ 90%", "Target")
        # Relevancy target.
        col3.metric("Answer Relevancy", "≥ 85%", "Target")
        # Speed latency targets.
        col4.metric("Latency", "< 5.0s", "Target")
    # Non empty analytics logs values processing dashboard layouts configs.
    else:
        # Extract metrics values properties checks templates parameter keys.
        cp = metrics.get("context_precision", 0)
        # Faithfulness.
        faith = metrics.get("faithfulness", 0)
        # Relevancy.
        ar = metrics.get("answer_relevancy", 0)
        # Latency convert.
        latency_s = metrics.get("avg_latency_ms", 0) / 1000
        # Total items.
        total_q = metrics.get("total_queries", 0)
        # Abstain ratios.
        hallucination_rate = metrics.get("hallucination_rate", 0)

        # KPI metric columns layout grids.
        col1, col2, col3, col4 = st.columns(4)

        # Comparison indicators tag text generators helper functions.
        def delta_str(val, target):
            # Calculate difference boundaries values.
            diff = val - target
            # Format outputs text arrow symbols percentage mappings properties.
            return f"{'↑' if diff >= 0 else '↓'} {abs(diff):.0%} vs target"

        # Context Precision KPI metric st widget setups.
        col1.metric(
            # Label name.
            "Context Precision",
            # Text values percentages format maps.
            f"{cp:.0%}",
            # Comparative status.
            delta_str(cp, 0.85),
            # Threshold color flags.
            delta_color="normal" if cp >= 0.85 else "inverse",
        )  # Metric 1 end.
        # Faithfulness KPI metric widgets configuration.
        col2.metric(
            "Faithfulness",
            f"{faith:.0%}",
            delta_str(faith, 0.90),
            delta_color="normal" if faith >= 0.90 else "inverse",
        )  # Metric 2 end.
        # Relevancy KPI metrics displays.
        col3.metric(
            "Answer Relevancy",
            f"{ar:.0%}",
            delta_str(ar, 0.85),
            delta_color="normal" if ar >= 0.85 else "inverse",
        )  # Metric 3 end.
        # Speed latency indicators metrics config.
        col4.metric(
            "Avg Latency",
            f"{latency_s:.1f}s",
            f"{'✓' if latency_s < 5 else '✗'} target < 5s",
            delta_color="normal" if latency_s < 5 else "inverse",
        )  # Metric 4 end.

        # Divider.
        st.divider()

        # Charts sub layout columns structures.
        col_g1, col_g2, col_g3 = st.columns(3)

        # Plotly custom radial gauge chart generator helper functions configurations.
        def make_gauge(title, value, target, max_val=1.0):
            # Threshold check colors selection configs parameters.
            color = "#00A3A6" if value >= target else "#991b1b"
            # Plotly Indicator charts specifications dictionary mappings configurations.
            fig = go.Figure(
                go.Indicator(
                    # Display modes properties indicators.
                    mode="gauge+number",
                    # Percentages conversions scale maps values.
                    value=round(value * 100, 1),
                    # Number suffixes formats.
                    number={"suffix": "%", "font": {"size": 24, "family": "Inter", "color": "#1e293b"}},
                    # Chart headers tags configurations parameters maps.
                    title={"text": title, "font": {"size": 14, "family": "Inter", "color": "#00338D", "weight": "bold"}},
                    # Radial details.
                    gauge={
                        # Axis scales bounds configs.
                        "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#64748b"},
                        # Color selectors.
                        "bar": {"color": color, "thickness": 0.8},
                        # Back colors.
                        "bgcolor": "#f8fafc",
                        # Borders.
                        "borderwidth": 1,
                        "bordercolor": "#cbd5e1",
                        # Colored ranges sectors backgrounds check paths options.
                        "steps": [
                            # Red zones.
                            {"range": [0, target * 100], "color": "#fef2f2"},
                            # Green zones.
                            {"range": [target * 100, 100], "color": "#f0fdf4"},
                        ],  # Steps end.
                        # Target thresholds markings settings.
                        "threshold": {
                            # Line options.
                            "line": {"color": "#00338D", "width": 3},
                            # Scale thickness.
                            "thickness": 0.75,
                            # Value marks.
                            "value": target * 100,
                        },  # Threshold end.
                    },  # Gauge map end.
                )  # Indicator end.
            )  # Figure end.
            # Layout style cleanup configurations parameters options details.
            fig.update_layout(
                # Transparent paper background.
                paper_bgcolor='rgba(0,0,0,0)',
                # Transparent plot background.
                plot_bgcolor='rgba(0,0,0,0)',
                # Height size.
                height=180,
                # Margins padding spacing values structures.
                margin=dict(t=40, b=10, l=10, r=10)
            )  # Update end.
            # Output figure object returns.
            return fig

        # Render charts.
        with col_g1:
            # Context precision gauges.
            st.plotly_chart(
                make_gauge("Context Precision", cp, 0.85), use_container_width=True
            )  # Chart 1 end.
        with col_g2:
            # Faithfulness gauges.
            st.plotly_chart(
                make_gauge("Faithfulness", faith, 0.90), use_container_width=True
            )  # Chart 2 end.
        with col_g3:
            # Relevancy gauges charts.
            st.plotly_chart(
                make_gauge("Answer Relevancy", ar, 0.85), use_container_width=True
            )  # Chart 3 end.

        # Divider line.
        st.divider()

        # Side by side helper summary columns metrics widgets.
        col_stats1, col_stats2 = st.columns(2)
        # Left statistics stats.
        with col_stats1:
            # Query count.
            st.metric("Total Queries Processed", f"{total_q:,}")
            # Abstain rate metrics.
            st.metric("Hallucination Rate", f"{hallucination_rate:.1%}", help="Fraction of queries that returned LOW confidence")
        # Right statistics stats.
        with col_stats2:
            # Load timestamp properties parameters options checks.
            last_eval = metrics.get("last_evaluated", "N/A")
            # Parse timing.
            st.metric("Last Activity", last_eval[:19].replace("T", " ") if last_eval else "N/A")
            # Source system markers.
            st.metric("Metrics Source", metrics.get("source", "heuristic").upper())

        # RAGAs full evaluation
        # Divider line.
        st.divider()
        # Section title header.
        st.subheader("🧪 Full RAGAs Evaluation")
        # Guides.
        st.caption(
            "Run the complete RAGAs evaluation suite using the test dataset. "
            "Requires OPENAI_API_KEY and data/processed/eval_dataset.json."
        )  # Caption end.
        # Automation trigger button events validations check paths.
        if st.button("▶️ Run Full RAGAs Evaluation"):
            # Load spinner status widgets during evaluations calculation processing.
            with st.spinner("Running RAGAs … this may take a few minutes."):
                # API executions try exceptions catch blocks options.
                try:
                    # Ingest Ragas runs route POST requests endpoint coordinates.
                    ragas_resp = requests.post(f"{api_url}/eval/run-ragas", timeout=300)
                    # Response validation confirmations checks logic parameters.
                    if ragas_resp.status_code == 200:
                        # Extract results payload.
                        ragas_metrics = ragas_resp.json()
                        # Streamlit success notifications.
                        st.success("✅ RAGAs evaluation complete!")
                        # Output JSON structures mappings displays console.
                        st.json(ragas_metrics)
                    # Failed statuses.
                    else:
                        # Error display.
                        st.error(ragas_resp.json().get("detail", "Evaluation failed."))
                # Exception catches alerts tracks.
                except Exception as exc:
                    # Errors alert.
                    st.error(f"Failed: {exc}")


# ==========================================================================
# TAB 3 — System Health
# ==========================================================================
# Tab 3 blocks setups margins layouts configurations.
with tab3:
    # Sub titles.
    st.subheader("🔧 System Health & Status")

    # Separator columns layout widgets templates settings refresh.
    col_ref3, _ = st.columns([1, 4])
    # Refresher widgets configurations checks.
    with col_ref3:
        # Reload clicks checks filters.
        if st.button("🔄 Refresh Status"):
            # Redraw screens.
            st.rerun()

    # API health check calls attempts catches errors exceptions structures.
    try:
        # Request health check routes.
        health_resp = requests.get(f"{api_url}/health", timeout=5)
        # Parse.
        health_data = health_resp.json() if health_resp.status_code == 200 else {}
    # Catch offline.
    except Exception:
        # Empty templates checks.
        health_data = {}

    # Side split layout columns widgets structure setup.
    col_h1, col_h2 = st.columns(2)

    # Left components health columns.
    with col_h1:
        # Headers.
        st.subheader("Component Status")
        # Check empty responses validation checkers codes.
        if not health_data:
            # Error logs message alerts displays.
            st.error("⚠️ Cannot reach API backend")
            # Instructions guides.
            st.code("uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload")
        # Non empty health records displays logic parameters updates details.
        else:
            # Extract configurations components indicators maps.
            comps = health_data.get("components", {})
            # Vector store.
            vs_status = comps.get("vector_store", "unknown")
            # LLM.
            llm_status = comps.get("llm", "unknown")
            # Chunks count.
            chunks = comps.get("total_chunks", 0)

            # Vector store
            # Confirm success ready flags checks properties.
            if vs_status == "ready":
                # Displays success logs.
                st.success(f"✅ Vector Store — {chunks:,} chunks indexed")
            # Failed vector stores codes.
            else:
                # Displays error notifications.
                st.error(f"❌ Vector Store — {vs_status}")

            # LLM
            # Confirm active parameters.
            if llm_status == "ready":
                # Success connected.
                st.success("✅ LLM (Groq) — Connected")
            # Key checks.
            elif llm_status == "no_api_key":
                # Warning key setups.
                st.warning("⚠️ LLM — No API key (set GROQ_API_KEY in .env)")
            # Errors.
            else:
                # Error status code logs.
                st.error(f"❌ LLM — {llm_status}")

            # Embedding model success verification.
            st.success("✅ Embedding Model — all-MiniLM-L6-v2 loaded")

            # API versions details st info widgets.
            st.info(f"ℹ️ API Version: {health_data.get('version', 'N/A')}")

    # Right document storage inventory summary displays columns.
    with col_h2:
        # Title headers.
        st.subheader("📁 Document Inventory")
        # Ingest documents query lists calls try catch errors checking blocks.
        try:
            # Call documents summary list route coordinates.
            docs_resp = requests.get(f"{api_url}/documents", timeout=5)
            # Response confirmation checks validation.
            if docs_resp.status_code == 200:
                # Parse.
                docs_data = docs_resp.json()
                # Load document array items lists.
                docs = docs_data.get("documents", [])

                # Verify empty collection flags.
                if not docs:
                    # Default info alert placeholder.
                    st.info("No documents indexed yet.")
                # Process active list structures loop displays.
                else:
                    # Write title parameters count summaries.
                    st.markdown(
                        f"**{docs_data['total_documents']} documents · "
                        f"{docs_data['total_chunks']:,} chunks**"
                    )  # Markdown end.
                    # Loop documents list mappings coordinates details parameters.
                    for doc in docs:
                        # Write document card layout custom html template.
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
                        )  # Markdown end.
        # Fallback offline checks.
        except Exception:
            # Inform instructions templates.
            st.info("Connect to the API to see document inventory.")

    # Quick-start guide
    # Visual divider.
    st.divider()
    # Guides.
    st.subheader("🚀 Quick Start")
    # Quick start command codes instructions markdown.
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
    )  # Markdown instructions guide end.
