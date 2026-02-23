"""
Influence-Aware RAG — Diagnostic Dashboard
Streamlit application for visualizing document contribution analysis.

Supports two modes:
  1. Simulation — generates synthetic data for demo purposes
  2. Load Results — reads real pipeline output from CSV
"""

import os
import random
import time
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from scipy import stats

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Influence-Aware RAG",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ──────────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=DM+Sans:wght@300;400;500;600&display=swap');

/* ── Root theme · Light mode · 60:30:10 ── */
:root {
    --bg: #ffffff;
    --surface: #f5f5f5;
    --surface2: #eeeeee;
    --border: #d4d4d4;
    --accent: #2563eb;
    --accent2: #dc2626;
    --accent3: #16a34a;
    --warn: #777777;
    --text: #111111;
    --muted: #666666;
    --mono: 'IBM Plex Mono', monospace;
    --sans: 'DM Sans', sans-serif;
}

/* ── Base ── */
html, body, [class*="css"] {
    font-family: var(--sans);
    background-color: var(--bg);
    color: var(--text);
}

.stApp { background-color: var(--bg); }

/* ── Hide default Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background-color: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { color: var(--text) !important; }

/* ── Metric cards ── */
[data-testid="metric-container"] {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 16px;
}
[data-testid="metric-container"] label { color: var(--muted) !important; font-family: var(--mono); font-size: 11px; }
[data-testid="metric-container"] [data-testid="stMetricValue"] { color: var(--text) !important; font-family: var(--mono); }
[data-testid="metric-container"] [data-testid="stMetricDelta"] { color: var(--accent3) !important; }

/* ── Buttons ── */
.stButton > button {
    background: var(--surface2);
    color: var(--text);
    border: 1px solid var(--border);
    border-radius: 6px;
    font-family: var(--mono);
    font-size: 13px;
    font-weight: 600;
    letter-spacing: 0.05em;
    padding: 10px 24px;
    transition: all 0.2s ease;
}
.stButton > button:hover {
    background: var(--accent);
    color: #ffffff;
    border-color: var(--accent);
}

/* ── Select / number inputs ── */
[data-testid="stSelectbox"] > div > div,
[data-testid="stNumberInput"] input,
[data-testid="stTextInput"] input {
    background-color: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    font-family: var(--mono) !important;
    border-radius: 6px !important;
}

/* ── Progress bar ── */
.stProgress > div > div > div {
    background: var(--accent);
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: var(--surface2);
    border-radius: 8px;
    border: 1px solid var(--border);
    gap: 4px;
    padding: 4px;
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    color: var(--muted);
    font-family: var(--mono);
    font-size: 12px;
    border-radius: 6px;
}
.stTabs [aria-selected="true"] {
    background: var(--text) !important;
    color: var(--bg) !important;
    font-weight: 700 !important;
}

/* ── Dataframe ── */
[data-testid="stDataFrame"] { border: 1px solid var(--border); border-radius: 8px; }

/* ── Custom card components ── */
.panel {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 20px 24px;
    margin-bottom: 16px;
}
.panel-accent { border-left: 3px solid var(--accent); }
.panel-warn   { border-left: 3px solid var(--muted); }
.panel-good   { border-left: 3px solid var(--accent3); }
.panel-bad    { border-left: 3px solid var(--accent2); }

.label {
    font-family: var(--mono);
    font-size: 10px;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 4px;
}
.value {
    font-family: var(--mono);
    font-size: 22px;
    font-weight: 600;
    color: var(--text);
}
.value-warn  { color: var(--muted);   }
.value-good  { color: var(--accent3); }
.value-bad   { color: var(--accent2); }

.mono { font-family: var(--mono); font-size: 12px; color: var(--muted); }
.tag {
    display: inline-block;
    background: var(--surface);
    border: 1px solid var(--border);
    color: var(--text);
    font-family: var(--mono);
    font-size: 11px;
    padding: 2px 8px;
    border-radius: 4px;
    margin-right: 4px;
}
.tag-bad { border-color: var(--accent2); color: var(--accent2); background: #fef2f2; }
.tag-good { border-color: var(--accent3); color: var(--accent3); background: #f0fdf4; }

.step-box {
    background: var(--surface);
    border: 1px solid var(--border);
    border-left: 3px solid var(--border);
    border-radius: 8px;
    padding: 14px 18px;
    margin-bottom: 10px;
    font-family: var(--mono);
    font-size: 12px;
}
.step-box .step-num {
    color: var(--muted);
    font-size: 10px;
    letter-spacing: 0.1em;
    margin-bottom: 4px;
}
.step-box .step-title {
    color: var(--text);
    font-size: 13px;
    font-weight: 600;
    margin-bottom: 4px;
}
.step-box .step-desc { color: var(--muted); font-size: 11px; }

.bar-wrap { margin-bottom: 8px; }
.bar-label {
    display: flex;
    justify-content: space-between;
    font-family: var(--mono);
    font-size: 11px;
    color: var(--muted);
    margin-bottom: 3px;
}
.bar-track {
    height: 8px;
    background: var(--surface);
    border-radius: 4px;
    overflow: hidden;
}
.bar-fill {
    height: 100%;
    border-radius: 4px;
    transition: width 0.6s ease;
}

.heatmap-cell {
    display: inline-block;
    width: 36px;
    height: 36px;
    border-radius: 4px;
    margin: 2px;
    text-align: center;
    line-height: 36px;
    font-family: var(--mono);
    font-size: 10px;
    font-weight: 600;
    cursor: pointer;
    transition: transform 0.15s;
}
.heatmap-cell:hover { transform: scale(1.15); }

/* ── Divider ── */
hr { border-color: var(--border) !important; }
</style>
""",
    unsafe_allow_html=True,
)



# ── Helpers ────────────────────────────────────────────────────────────────────


def make_bar(label, value, max_val=1.0, color_var="var(--accent)", doc_id=None):
    pct = min(value / max_val * 100, 100) if max_val > 0 else 0
    prefix = f"D{doc_id} " if doc_id is not None else ""
    return f"""
    <div class="bar-wrap">
      <div class="bar-label">
        <span>{prefix}{label}</span>
        <span>{value:.3f}</span>
      </div>
      <div class="bar-track">
        <div class="bar-fill" style="width:{pct:.1f}%;background:{color_var};"></div>
      </div>
    </div>"""


def spearman_color(rho):
    if rho >= 0.7:
        return "value-good", "ALIGNED"
    if rho >= 0.4:
        return "value-warn", "PARTIAL"
    return "value-bad", "DIVERGENT"


import re

def highlight_query_terms(text: str, query: str) -> str:
    """
    Highlights words in `text` that appear in `query` (case-insensitive)
    using HTML/CSS. Excludes common stop words to avoid highlighting 'the', 'is', 'in'.
    """
    if not text or not query:
        return text
        
    stop_words = {
        "the", "is", "in", "at", "which", "on", "for", "a", "an", "and", 
        "or", "but", "to", "with", "of", "how", "what", "where", "when", 
        "who", "why", "did", "does", "do", "has", "have", "had", "been", "was", "were"
    }
    
    # Extract alphanumeric query terms
    query_terms = [t.lower() for t in re.findall(r'\b\w+\b', query)]
    query_terms = [t for t in query_terms if t not in stop_words and len(t) > 1]
    
    if not query_terms:
        return text
        
    # Build regex to find any of the query terms (case insensitive)
    # Using negative lookbehind/lookahead for letters so it catches words next to punctuation
    pattern = r'(?<![a-zA-Z])(' + '|'.join(map(re.escape, query_terms)) + r')(?![a-zA-Z])'
    
    # Replace with highlighted version
    highlighted = re.sub(
        pattern, 
        r'<strong style="background-color:rgba(37,99,235,0.2);color:#111111;padding:0 2px;border-radius:2px;">\1</strong>', 
        text, 
        flags=re.IGNORECASE
    )
    
    return highlighted


def simulate_query(query_text: str, n_docs=10, seed=None):
    """Simulate one query's retrieval + ablation results."""
    rng = np.random.default_rng(seed)
    retrieval_scores = np.sort(rng.exponential(2.0, n_docs))[::-1]
    retrieval_scores = retrieval_scores / retrieval_scores.max()

    # Influence weakly correlated with retrieval, but with noise
    noise = rng.normal(0, 0.35, n_docs)
    influence_scores = np.clip(
        retrieval_scores * 0.5 + noise + rng.uniform(0, 0.3, n_docs), 0.01, 1.0
    )

    # Extract non-stop words from query to inject into passages
    stop_words = {"what", "is", "the", "in", "who", "played", "when", "was", "where", "did", "how", "many", "are", "of", "to", "and", "a"}
    query_words = [w for w in re.findall(r'\b\w+\b', query_text.lower()) if w not in stop_words and len(w) > 2]
    
    passages = []
    for i in range(n_docs):
        base_sentence = rng.choice(
            [
                "The study examined the relationship between neural network depth and generalization error...",
                "Recent advances in transformer architectures have demonstrated significant improvements...",
                "Legal precedents establish that liability requires both duty of care and causation...",
                "Clinical trial data indicates a statistically significant reduction in adverse events...",
                "Financial models incorporating macroeconomic indicators outperform baseline strategies...",
                "Urban planning frameworks must account for population density and transit corridors...",
                "The algorithm converges in O(n log n) time under the assumption of sparse retrieval...",
                "Empirical results across five benchmark datasets confirm the proposed method's efficacy...",
                "Regulatory compliance frameworks require documentation of data lineage and provenance...",
                "Longitudinal studies over 12 months reveal consistent patterns in user behaviour...",
            ]
        )
        # Randomly inject 1-3 query words into the passage to simulate a match
        if query_words:
            num_inject = rng.integers(1, min(len(query_words) + 1, 4))
            injected_words = rng.choice(query_words, size=num_inject, replace=False)
            passage = f"Document {i + 1}: {' '.join(injected_words)} — {base_sentence}"
        else:
            passage = f"Document {i + 1}: {base_sentence}"
        passages.append(passage)

    ret_ranks = np.argsort(np.argsort(-retrieval_scores)) + 1
    inf_ranks = np.argsort(np.argsort(-influence_scores)) + 1
    rho, pval = stats.spearmanr(ret_ranks, inf_ranks)

    dominance = influence_scores.max() / influence_scores.sum()
    probs = influence_scores / influence_scores.sum()
    entropy = float(-np.sum(probs * np.log(probs + 1e-9)))

    docs = []
    for i in range(n_docs):
        docs.append(
            {
                "doc_id": i + 1,
                "passage": passages[i],
                "retrieval_score": float(retrieval_scores[i]),
                "retrieval_rank": int(ret_ranks[i]),
                "influence_score": float(influence_scores[i]),
                "influence_rank": int(inf_ranks[i]),
                "rouge_l": float(1.0 - influence_scores[i]),
            }
        )

    return {
        "docs": docs,
        "spearman_rho": float(rho),
        "pvalue": float(pval),
        "dominance_ratio": float(dominance),
        "influence_entropy": float(entropy),
        "is_divergent": rho < 0.7,
    }


def run_full_simulation(n_queries, n_docs, seed=42):
    results = []
    
    # A set of real-world queries for the simulation to cycle through
    real_queries = [
        "what is the population of new york city in 2020",
        "who played the main character in the movie gladiator",
        "when was the first iphone released",
        "where did they film high school musical two",
        "how many episodes are in friends season 3",
        "who won the super bowl in 1999",
        "what is the capital city of australia",
        "when did the cold war start and end",
        "what is the distance from the earth to the moon",
        "who wrote the book 1984",
        "how long does it take for light from the sun to reach earth",
        "what is the tallest mountain in the world",
        "who is the ceo of microsoft",
        "where was the declaration of independence signed",
        "what is the main ingredient in guacamole",
    ]
    
    for i in range(n_queries):
        # Pick a real query, cycling if n_queries > len(real_queries)
        q_text = real_queries[i % len(real_queries)]
        # Add a slight variation if we cycle to make it unique
        if i >= len(real_queries):
            q_text = f"{q_text} ({i + 1})"
            
        r = simulate_query(query_text=q_text, n_docs=n_docs, seed=seed + i)
        r["query_id"] = i + 1
        r["query"] = q_text
        results.append(r)
    return results


def load_results_from_csv(csv_path: str, div_threshold: float = 0.7):
    """
    Parse the pipeline CSV output back into the internal results format
    that the dashboard expects.
    """
    df = pd.read_csv(csv_path)

    # Group by query
    results = []
    for query_idx, (query_text, group) in enumerate(df.groupby("query", sort=False)):
        docs = []
        for _, row in group.iterrows():
            docs.append(
                {
                    "doc_id": int(row.get("doc_id", 0)),
                    "passage": str(row.get("text", row.get("passage", ""))),
                    "retrieval_score": float(row.get("retrieval_score", 0)),
                    "retrieval_rank": int(row.get("retrieval_rank", 0)),
                    "influence_score": float(row.get("influence_score", 0)),
                    "influence_rank": int(row.get("influence_rank", 0)),
                    "rouge_l": float(row.get("rouge_l", 0)),
                }
            )

        rho = float(group["spearman_rho"].iloc[0])
        results.append(
            {
                "query_id": query_idx + 1,
                "query": str(query_text),
                "docs": docs,
                "spearman_rho": rho,
                "pvalue": float(group.get("spearman_pvalue", group.get("pvalue", pd.Series([0.0]))).iloc[0]),
                "dominance_ratio": float(group["dominance_ratio"].iloc[0]),
                "influence_entropy": float(group["influence_entropy"].iloc[0]),
                "is_divergent": rho < div_threshold,
            }
        )

    return results


# ── Sidebar ─────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        """
    <div style="padding:4px 0 20px;">
      <div style="font-family:'DM Sans',sans-serif;font-size:18px;font-weight:600;color:#111111;">
        Influence-Aware RAG
      </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # ── Data Source toggle ──
    st.markdown('<div class="label">Data Source</div>', unsafe_allow_html=True)
    data_mode = st.radio(
        "Mode",
        ["Simulation", "Load Results (CSV)"],
        index=0,
        label_visibility="collapsed",
    )

    st.markdown("---")

    if data_mode == "Simulation":
        st.markdown(
            '<div class="label">Simulation Parameters</div>', unsafe_allow_html=True
        )
        n_queries = st.number_input(
            "Number of queries", min_value=10, max_value=200, value=50, step=10
        )
        n_docs = st.number_input(
            "Documents per query (top-k)", min_value=3, max_value=15, value=10
        )
        seed = st.number_input("Random seed", min_value=0, max_value=9999, value=42)
    else:
        st.markdown(
            '<div class="label">CSV File</div>', unsafe_allow_html=True
        )
        csv_path = st.text_input(
            "Path to results CSV",
            value="outputs/rag_influence_results.csv",
            label_visibility="collapsed",
        )
        # Also allow file upload
        uploaded = st.file_uploader("Or upload CSV", type=["csv"])
        n_docs = 10  # default for failure-mode threshold calc

    st.markdown("---")
    st.markdown('<div class="label">Divergence Threshold</div>', unsafe_allow_html=True)
    div_threshold = st.slider("ρ threshold", 0.0, 1.0, 0.7, 0.05)

    st.markdown("---")

    if data_mode == "Simulation":
        run_btn = st.button("▶  RUN SIMULATION", width="stretch")
    else:
        run_btn = st.button("▶  LOAD DATA", width="stretch")

    st.markdown(
        """
    <div style="margin-top:32px;">
      <div class="label" style="margin-bottom:8px;">Pipeline</div>
    """,
        unsafe_allow_html=True,
    )
    for step in [
        ("01", "RETRIEVE", "BM25 top-k per query"),
        ("02", "GENERATE", "LLM baseline answer"),
        ("03", "ABLATE", "Remove each doc, re-generate"),
        ("04", "SCORE", "ROUGE-L delta → influence"),
        ("05", "ANALYSE", "Spearman ρ, entropy, dominance"),
    ]:
        st.markdown(
            f"""
        <div class="step-box" style="margin-bottom:6px;padding:10px 14px;">
          <div class="step-num">{step[0]}</div>
          <div class="step-title">{step[1]}</div>
          <div class="step-desc">{step[2]}</div>
        </div>""",
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)


# ── Session state ───────────────────────────────────────────────────────────────
if "results" not in st.session_state:
    st.session_state.results = None
if "selected_query" not in st.session_state:
    st.session_state.selected_query = 0
if "data_source_label" not in st.session_state:
    st.session_state.data_source_label = ""


# ── Header ──────────────────────────────────────────────────────────────────────
st.markdown(
    """
<div style="padding: 8px 0 24px;">
  <div style="font-family:'DM Sans',sans-serif;font-size:28px;font-weight:600;color:#111111;
              line-height:1.2;">
    Influence-Aware RAG
    <span style="font-size:14px;font-weight:300;color:#666666;margin-left:12px;vertical-align:middle;">
        An ablation-based diagnostic tool for RAG pipelines. (in development)
    </span>
  </div>
</div>
""",
    unsafe_allow_html=True,
)


# ── Run / Load handler ──────────────────────────────────────────────────────────
if run_btn:
    if data_mode == "Simulation":
        progress_bar = st.progress(0, text="Initialising pipeline...")
        status_text = st.empty()

        steps = [
            (0.10, "Loading Natural Questions dataset..."),
            (0.20, "Building BM25 index over passages..."),
            (0.35, "Retrieving top-k documents per query..."),
            (0.55, "Generating baseline answers via LLM..."),
            (0.80, "Running ablation loop (10 passes × query)..."),
            (0.92, "Computing Spearman ρ, dominance, entropy..."),
            (1.00, "Aggregating results..."),
        ]

        for pct, msg in steps:
            time.sleep(0.35)
            progress_bar.progress(pct, text=msg)
            status_text.markdown(
                f'<span class="mono">⟩ {msg}</span>', unsafe_allow_html=True
            )

        st.session_state.results = run_full_simulation(n_queries, n_docs, seed)
        st.session_state.selected_query = 0
        st.session_state.data_source_label = f"Simulation · {n_queries} queries × {n_docs} docs"
        progress_bar.empty()
        status_text.empty()
        st.success(
            f"Simulation complete — {n_queries} queries × {n_docs} documents = {n_queries * n_docs} ablation calls"
        )

    else:
        # Load from CSV
        try:
            if uploaded is not None:
                df_tmp = pd.read_csv(uploaded)
                # Save temporarily so load_results_from_csv can read
                tmp_path = "/tmp/_rag_uploaded.csv"
                df_tmp.to_csv(tmp_path, index=False)
                loaded = load_results_from_csv(tmp_path, div_threshold)
                source = "uploaded file"
            else:
                if not os.path.exists(csv_path):
                    st.error(f"File not found: `{csv_path}`. Run the pipeline first with `python run_pipeline.py`.")
                    st.stop()
                loaded = load_results_from_csv(csv_path, div_threshold)
                source = csv_path

            # Determine n_docs from data
            if loaded and loaded[0]["docs"]:
                n_docs = len(loaded[0]["docs"])

            st.session_state.results = loaded
            st.session_state.selected_query = 0
            st.session_state.data_source_label = f"CSV · {source} · {len(loaded)} queries"
            st.success(f"Loaded {len(loaded)} queries from {source}")
        except Exception as exc:
            st.error(f"Failed to load CSV: {exc}")
            st.stop()


# ── Main content (only shown after run) ─────────────────────────────────────────
if st.session_state.results is None:
    # Welcome state
    st.markdown(
        """

    <div style="margin-top:32px;display:grid;grid-template-columns:1fr 1fr 1fr;gap:16px;max-width:800px;">
    """,
        unsafe_allow_html=True,
    )

    st.markdown("</div>", unsafe_allow_html=True)
    st.stop()


results = st.session_state.results

# Data source badge
if st.session_state.data_source_label:
    st.markdown(
        f'<div class="mono" style="margin-bottom:8px;">📊 {st.session_state.data_source_label}</div>',
        unsafe_allow_html=True,
    )

# Build summary dataframe
summary = pd.DataFrame(
    [
        {
            "Query ID": r["query_id"],
            "Spearman ρ": round(r["spearman_rho"], 3),
            "Dominance Ratio": round(r["dominance_ratio"], 3),
            "Entropy": round(r["influence_entropy"], 3),
            "Divergent": r["spearman_rho"] < div_threshold,  # dynamically computed
            "p-value": round(r["pvalue"], 4),
        }
        for r in results
    ]
)

n_divergent = summary["Divergent"].sum()
mean_rho = summary["Spearman ρ"].mean()
mean_dom = summary["Dominance Ratio"].mean()
mean_entropy = summary["Entropy"].mean()


# ── Top KPI row ──────────────────────────────────────────────────────────────────
k1, k2, k3, k4 = st.columns(4)

with k1:
    cls, lbl = spearman_color(mean_rho)
    st.markdown(
        f"""
    <div class="panel">
      <div class="label">Mean Spearman ρ</div>
      <div class="value {cls}">{mean_rho:.3f}</div>
      <div style="font-family:'IBM Plex Mono',monospace;font-size:10px;margin-top:4px;">
        <span class="tag {"tag-bad" if lbl == "DIVERGENT" else "tag-good" if lbl == "ALIGNED" else ""}">{lbl}</span>
      </div>
    </div>""",
        unsafe_allow_html=True,
    )

with k2:
    pct = n_divergent / len(results) * 100
    st.markdown(
        f"""
    <div class="panel">
      <div class="label">Divergent Queries (ρ &lt; {div_threshold})</div>
      <div class="value {"value-bad" if pct > 40 else "value-warn" if pct > 20 else "value-good"}">{pct:.1f}%</div>
      <div class="mono" style="margin-top:4px;">{n_divergent} / {len(results)} queries</div>
    </div>""",
        unsafe_allow_html=True,
    )

with k3:
    st.markdown(
        f"""
    <div class="panel">
      <div class="label">Mean Dominance Ratio</div>
      <div class="value">{mean_dom:.3f}</div>
      <div class="mono" style="margin-top:4px;">max(inf) / Σ(inf)</div>
    </div>""",
        unsafe_allow_html=True,
    )

with k4:
    st.markdown(
        f"""
    <div class="panel">
      <div class="label">Mean Influence Entropy</div>
      <div class="value">{mean_entropy:.3f}</div>
      <div class="mono" style="margin-top:4px;">distribution spread</div>
    </div>""",
        unsafe_allow_html=True,
    )


st.markdown("---")

# ── Tabs ─────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "  OVERVIEW  ",
        "  QUERY INSPECTOR  ",
        "  QUERY ANALYZER  ",
        "  FAILURE MODES  ",
        "  METHODOLOGY  ",
    ]
)


# ═══════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ═══════════════════════════════════════════════════════
with tab1:
    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        st.markdown(
            '<div class="label" style="margin-bottom:12px;">Spearman ρ Distribution</div>',
            unsafe_allow_html=True,
        )

        rhos = summary["Spearman ρ"].values
        bins = np.linspace(-1, 1, 21)
        counts, edges = np.histogram(rhos, bins=bins)
        max_count = counts.max()

        # Manual histogram bars in HTML
        bar_html = '<div style="display:flex;align-items:flex-end;gap:3px;height:120px;padding-bottom:4px;">'
        for i, (cnt, edge) in enumerate(zip(counts, edges[:-1])):
            h_pct = cnt / max_count * 100 if max_count > 0 else 0
            is_div = edge < div_threshold
            color = "var(--accent2)" if is_div else "var(--accent3)"
            bar_html += f'<div title="ρ≈{edge:.1f}: {cnt} queries" style="flex:1;height:{h_pct:.1f}%;background:{color};border-radius:2px 2px 0 0;opacity:0.85;"></div>'
        bar_html += "</div>"
        bar_html += f"""
        <div style="display:flex;justify-content:space-between;font-family:'IBM Plex Mono',monospace;font-size:10px;color:#666666;border-top:1px solid #d4d4d4;padding-top:4px;">
          <span>−1.0</span><span>−0.5</span><span>0.0</span>
          <span style="color:#dc2626;">{div_threshold} ▲</span>
          <span>0.5</span><span>1.0</span>
        </div>
        <div style="margin-top:8px;font-family:'IBM Plex Mono',monospace;font-size:10px;">
          <span style="color:#dc2626;">■</span> <span style="color:#666666;">Divergent (ρ &lt; {div_threshold})</span>
          &nbsp;&nbsp;
          <span style="color:#16a34a;">■</span> <span style="color:#666666;">Aligned (ρ ≥ {div_threshold})</span>
        </div>
        """
        st.markdown(f'<div class="panel">{bar_html}</div>', unsafe_allow_html=True)

    with col_right:
        st.markdown(
            '<div class="label" style="margin-bottom:12px;">Influence Entropy vs. Dominance Ratio</div>',
            unsafe_allow_html=True,
        )

        ent_vals = summary["Entropy"].values
        dom_vals = summary["Dominance Ratio"].values
        div_mask = summary["Divergent"].values

        # Simple scatter SVG
        W, H = 340, 180

        def scale_x(v):
            return int(40 + (v / (ent_vals.max() + 0.1)) * (W - 60))

        def scale_y(v):
            return int(H - 20 - (v / (dom_vals.max() + 0.05)) * (H - 40))

        svg = f'<svg width="{W}" height="{H}" style="overflow:visible">'
        svg += f'<line x1="40" y1="{H - 20}" x2="{W - 20}" y2="{H - 20}" stroke="#d4d4d4" stroke-width="1"/>'
        svg += f'<line x1="40" y1="10" x2="40" y2="{H - 20}" stroke="#d4d4d4" stroke-width="1"/>'
        svg += f'<text x="20" y="{H // 2}" fill="#6b7280" font-size="9" transform="rotate(-90,20,{H // 2})" text-anchor="middle" font-family="monospace">Dominance</text>'
        svg += f'<text x="{W // 2}" y="{H - 4}" fill="#6b7280" font-size="9" text-anchor="middle" font-family="monospace">Entropy</text>'
        for e, d, div in zip(ent_vals, dom_vals, div_mask):
            cx, cy = scale_x(e), scale_y(d)
            color = "#dc2626" if div else "#2563eb"
            svg += f'<circle cx="{cx}" cy="{cy}" r="4" fill="{color}" opacity="0.7"/>'
        svg += "</svg>"
        st.markdown(f'<div class="panel">{svg}</div>', unsafe_allow_html=True)

    # Full results table
    st.markdown(
        '<div class="label" style="margin:16px 0 8px;">All Query Results</div>',
        unsafe_allow_html=True,
    )
    display_df = summary.copy()
    display_df["Status"] = display_df["Divergent"].map(
        {True: "⚠ DIVERGENT", False: "✓ ALIGNED"}
    )
    st.dataframe(
        display_df[
            [
                "Query ID",
                "Spearman ρ",
                "Dominance Ratio",
                "Entropy",
                "p-value",
                "Status",
            ]
        ],
        width="stretch",
        height=280,
    )


# ═══════════════════════════════════════════════════════
# TAB 2 — QUERY INSPECTOR
# ═══════════════════════════════════════════════════════
with tab2:
    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    query_options = [f"Q{r['query_id']}: {r['query'][:60]}..." if len(str(r['query'])) > 60 else f"Q{r['query_id']}: {str(r['query'])}" for r in results]
    sel_label = st.selectbox("Select query to inspect", query_options)
    
    # Extract the id from "QX: ..."
    sel_id_str = sel_label.split(":")[0][1:] # e.g. "Q1" -> "1"
    sel_idx = int(sel_id_str) - 1
    r = results[sel_idx]
    docs = r["docs"]
    
    st.markdown(f"**Query:** {r['query']}")

    cls, lbl = spearman_color(r["spearman_rho"])
    is_div_dynamic = r["spearman_rho"] < div_threshold
    
    st.markdown(
        f"""
    <div class="panel panel-accent" style="display:flex;gap:40px;align-items:center;flex-wrap:wrap;">
      <div>
        <div class="label">Spearman ρ</div>
        <div class="value {cls}">{r["spearman_rho"]:.3f}</div>
      </div>
      <div>
        <div class="label">Status</div>
        <div style="margin-top:4px;"><span class="tag {"tag-bad" if is_div_dynamic else "tag-good"}">{lbl}</span></div>
      </div>
      <div>
        <div class="label">Dominance Ratio</div>
        <div class="value">{r["dominance_ratio"]:.3f}</div>
      </div>
      <div>
        <div class="label">Influence Entropy</div>
        <div class="value">{r["influence_entropy"]:.3f}</div>
      </div>
      <div>
        <div class="label">p-value</div>
        <div class="mono" style="color:#111111;font-size:14px;">{r["pvalue"]:.4f}</div>
      </div>
    </div>""",
        unsafe_allow_html=True,
    )

    col_a, col_b = st.columns(2, gap="large")

    with col_a:
        st.markdown(
            '<div class="label" style="margin-bottom:10px;">Retrieval Scores (BM25)</div>',
            unsafe_allow_html=True,
        )
        html = '<div class="panel">'
        sorted_by_ret = sorted(docs, key=lambda x: x["retrieval_rank"])
        for d in sorted_by_ret:
            html += make_bar(
                f"Rank {d['retrieval_rank']}",
                d["retrieval_score"],
                max_val=1.0,
                color_var="var(--accent)",
                doc_id=d["doc_id"],
            )
        html += "</div>"
        st.markdown(html, unsafe_allow_html=True)

    with col_b:
        st.markdown(
            '<div class="label" style="margin-bottom:10px;">Influence Scores (Ablation)</div>',
            unsafe_allow_html=True,
        )
        html = '<div class="panel">'
        sorted_by_inf = sorted(docs, key=lambda x: x["influence_rank"])
        max_inf = max(d["influence_score"] for d in docs) if docs else 1
        for d in sorted_by_inf:
            is_fail = d["retrieval_rank"] <= 3 and d["influence_rank"] >= (n_docs - 2)
            color = "var(--accent2)" if is_fail else "var(--accent3)"
            html += make_bar(
                f"Rank {d['influence_rank']}",
                d["influence_score"],
                max_val=max_inf,
                color_var=color,
                doc_id=d["doc_id"],
            )
        html += "</div>"
        st.markdown(html, unsafe_allow_html=True)

    # Rank divergence heatmap
    st.markdown(
        '<div class="label" style="margin:16px 0 8px;">Rank Divergence Map · Retrieval rank → Influence rank</div>',
        unsafe_allow_html=True,
    )
    cell_html = '<div class="panel"><div style="display:flex;flex-wrap:wrap;gap:4px;">'
    for d in sorted(docs, key=lambda x: x["doc_id"]):
        diff = abs(d["retrieval_rank"] - d["influence_rank"])
        # Color from green (0 diff) to red (max diff)
        max_diff = max(n_docs - 1, 1)
        red = min(255, int(diff / max_diff * 255))
        green = 255 - red
        bg = f"rgb({red},{green},80)"
        cell_html += f"""
        <div class="heatmap-cell" title="D{d['doc_id']}: ret={d['retrieval_rank']}, inf={d['influence_rank']}, Δ={diff}"
             style="background:{bg};color:#111111;">
          D{d['doc_id']}
        </div>"""
    cell_html += "</div>"
    cell_html += "<div style=\"margin-top:10px;font-family:'IBM Plex Mono',monospace;font-size:10px;color:#666666;\">Colour = |retrieval rank − influence rank| · Green = aligned · Red = divergent</div>"
    cell_html += "</div>"
    st.markdown(cell_html, unsafe_allow_html=True)

    # Document detail table
    st.markdown(
        '<div class="label" style="margin:16px 0 8px;">Document Detail</div>',
        unsafe_allow_html=True,
    )
    doc_df = pd.DataFrame(
        [
            {
                "Doc": f"D{d['doc_id']}",
                "Retrieval Score": round(d["retrieval_score"], 3),
                "Retrieval Rank": d["retrieval_rank"],
                "Influence Score": round(d["influence_score"], 3),
                "Influence Rank": d["influence_rank"],
                "Rank Δ": abs(d["retrieval_rank"] - d["influence_rank"]),
                "ROUGE-L": round(d["rouge_l"], 3),
                "Failure Mode": "⚠"
                if (d["retrieval_rank"] <= 3 and d["influence_rank"] >= n_docs - 2)
                else "",
            }
            for d in docs
        ]
    )
    st.dataframe(doc_df, width="stretch")


# ═══════════════════════════════════════════════════════
# TAB 3 — QUERY ANALYZER
# ═══════════════════════════════════════════════════════
with tab3:
    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
    
    col_qlist, col_docs, col_pie = st.columns([1, 2, 2], gap="large")
    
    with col_qlist:
        st.markdown('<div class="label" style="margin-bottom:10px;">Select Query</div>', unsafe_allow_html=True)
        query_options = [f"Q{r['query_id']}: {r['query'][:40]}..." if len(str(r['query'])) > 40 else f"Q{r['query_id']}: {str(r['query'])}" for r in results]
        
        # Add custom CSS for making the radio button container scrollable if there are many queries
        st.markdown(
            """
            <style>
            [data-testid="stRadio"] {
                max-height: 600px;
                overflow-y: auto;
            }
            </style>
            """,
            unsafe_allow_html=True
        )
        sel_q_label = st.radio("Queries", query_options, label_visibility="collapsed")
        sel_q_idx = query_options.index(sel_q_label)
        selected_r = results[sel_q_idx]
        
    with col_docs:
        st.markdown('<div class="label" style="margin-bottom:10px;">Retrieved Documents</div>', unsafe_allow_html=True)
        st.markdown(f"**Query {selected_r['query_id']}:** {selected_r['query']}")
        
        if "baseline_answer" in selected_r and selected_r["baseline_answer"]:
            st.markdown(f"**Baseline Answer:** {selected_r['baseline_answer']}")

        st.markdown("<hr style='margin:10px 0'>", unsafe_allow_html=True)
        
        # List of documents
        docs_sorted = sorted(selected_r["docs"], key=lambda x: x["influence_rank"])
        for d in docs_sorted:
            inf_score = d['influence_score']
            color = "#16a34a" if inf_score > 0 else "#dc2626" if inf_score < 0 else "#666666"
            
            st.markdown(f"**Doc {d['doc_id']}** (Influence Rank: **{d['influence_rank']}**, Score: <span style='color:{color}'>{inf_score:.3f}</span>)", unsafe_allow_html=True)
            
            passage_text = d.get("passage", "")
            if passage_text:
                with st.expander("View Passage Text"):
                    highlighted_text = highlight_query_terms(passage_text, selected_r['query'])
                    st.markdown(highlighted_text, unsafe_allow_html=True)
            else:
                st.caption("No passage text available in results.")

    with col_pie:
        st.markdown('<div class="label" style="margin-bottom:10px;">Explicit Ablation Results (Influence)</div>', unsafe_allow_html=True)
        
        labels = []
        sizes = []
        colors = []
        
        cmap = plt.get_cmap("tab20")
        
        # For a pie chart, we only plot positive influence scores
        for i, d in enumerate(docs_sorted):
            if d['influence_score'] > 0:
                labels.append(f"Doc {d['doc_id']} (Inf: {d['influence_score']:.2f})")
                sizes.append(d['influence_score'])
                colors.append(cmap(i % 20))
                
        if sum(sizes) > 0:
            fig, ax = plt.subplots(figsize=(6, 6))
            wedges, texts, autotexts = ax.pie(
                sizes, 
                labels=labels, 
                autopct='%1.1f%%', 
                startangle=90, 
                colors=colors,
                wedgeprops=dict(width=0.6, edgecolor='#ffffff', linewidth=1) # Donut chart style
            )
            ax.axis('equal')
            fig.patch.set_alpha(0.0)
            
            # Label styling
            for text in texts:
                text.set_color('#111111')
                text.set_fontsize(9)
                text.set_fontfamily('monospace')
            for autotext in autotexts:
                autotext.set_color('#111111')
                autotext.set_fontsize(8)
                autotext.set_fontweight('bold')
                
            st.pyplot(fig, transparent=True)
            plt.close(fig)
            
            st.markdown(
                """
                <div style="font-family:'IBM Plex Mono',monospace;font-size:10px;color:#666666;line-height:1.6;margin-top:12px;">
                Note: Only documents with positive influence scores (>0) are included in the pie chart. Documents that had zero or negative impact are excluded.
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.info("No documents had a positive influence score for this query.")


# ═══════════════════════════════════════════════════════
# TAB 4 — FAILURE MODES
# ═══════════════════════════════════════════════════════
with tab4:
    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    # Collect all failure mode instances
    failures = []
    for r in results:
        for d in r["docs"]:
            if d["retrieval_rank"] <= 3 and d["influence_rank"] >= (n_docs - 2):
                failures.append(
                    {
                        "Query ID": r["query_id"],
                        "Doc": f"D{d['doc_id']}",
                        "Retrieval Rank": d["retrieval_rank"],
                        "Influence Rank": d["influence_rank"],
                        "Influence Score": round(d["influence_score"], 3),
                        "Query ρ": round(r["spearman_rho"], 3),
                    }
                )

    st.markdown(
        f"""
    <div class="panel panel-bad">
      <div class="label">Failure Mode Definition</div>
      <div style="margin-top:8px;font-size:13px;color:#444444;line-height:1.7;">
        A <strong style="color:#dc2626;">failure mode</strong> occurs when a document ranks in the
        <strong>top 3 by BM25 retrieval</strong> but ranks in the
        <strong>bottom 3 by ablation influence</strong>. This indicates the retriever is surfacing
        documents the LLM does not actually use to construct its answer.
      </div>
      <div style="margin-top:12px;">
        <span class="tag tag-bad">⚠ {len(failures)} instances detected</span>
        <span class="tag tag-bad">{len(failures) / max(len(results), 1):.1f} avg per query</span>
      </div>
    </div>""",
        unsafe_allow_html=True,
    )

    if failures:
        fail_df = pd.DataFrame(failures)
        st.dataframe(fail_df, width="stretch")

        # Distribution of failure modes across queries
        st.markdown(
            '<div class="label" style="margin:20px 0 8px;">Failure Mode Frequency by Query</div>',
            unsafe_allow_html=True,
        )
        freq = Counter(f["Query ID"] for f in failures)
        top_queries = sorted(freq.items(), key=lambda x: -x[1])[:15]

        bar_html = '<div class="panel">'
        for qid, cnt in top_queries:
            bar_html += make_bar(
                f"Query #{qid}",
                cnt,
                max_val=max(freq.values()),
                color_var="var(--accent2)",
            )
        bar_html += "</div>"
        st.markdown(bar_html, unsafe_allow_html=True)
    else:
        st.markdown(
            '<div class="panel"><div style="color:#666666;font-family:\'IBM Plex Mono\',monospace;font-size:12px;">No failure modes detected in this run.</div></div>',
            unsafe_allow_html=True,
        )

    # Divergent queries breakdown
    st.markdown(
        '<div class="label" style="margin:20px 0 8px;">Divergent Query Deep-Dive</div>',
        unsafe_allow_html=True,
    )
    div_results = [r for r in results if r["is_divergent"]]

    if div_results:
        cols_div = st.columns(min(3, len(div_results)))
        for i, r in enumerate(div_results[:3]):
            with cols_div[i]:
                cls, lbl = spearman_color(r["spearman_rho"])
                st.markdown(
                    f"""
                <div class="panel panel-bad">
                  <div class="label">Query #{r["query_id"]}</div>
                  <div class="value {cls}" style="font-size:18px;">{r["spearman_rho"]:.3f}</div>
                  <div class="mono" style="margin-top:8px;">
                    dom={r["dominance_ratio"]:.2f} · H={r["influence_entropy"]:.2f}
                  </div>
                </div>""",
                    unsafe_allow_html=True,
                )
    else:
        st.markdown(
            '<div class="panel panel-good"><div style="color:#16a34a;font-size:13px;">✓ No divergent queries in this run.</div></div>',
            unsafe_allow_html=True,
        )


# ═══════════════════════════════════════════════════════
# TAB 5 — METHODOLOGY
# ═══════════════════════════════════════════════════════
with tab5:
    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
    col_m1, col_m2 = st.columns([1, 1], gap="large")

    with col_m1:
        st.markdown(
            '<div class="label" style="margin-bottom:12px;">Pipeline Steps</div>',
            unsafe_allow_html=True,
        )
        for step in [
            (
                "01",
                "Dataset & Setup",
                "Sample 100–200 queries from Natural Questions (short-answer subset). Build BM25 index over passages extracted from each query's document.",
                "panel-accent",
            ),
            (
                "02",
                "BM25 Retrieval",
                "For each query, retrieve top-k documents via BM25Okapi. Record retrieval scores and assign retrieval ranks (1 = highest BM25 score).",
                "panel-accent",
            ),
            (
                "03",
                "Baseline Generation",
                "Feed all k documents + query to the LLM (via OpenRouter). Generate and store the baseline answer. Temperature = 0 for reproducibility.",
                "panel-accent",
            ),
            (
                "04",
                "Ablation Loop",
                "For each document Di, remove it from context and regenerate the answer. Compute ROUGE-L against baseline. influence_i = 1 − ROUGE-L.",
                "panel-warn",
            ),
            (
                "05",
                "Rank & Correlate",
                "Rank documents by influence score. Compute Spearman ρ between retrieval ranks and influence ranks. Flag if ρ < 0.7 (divergent).",
                "panel-warn",
            ),
            (
                "06",
                "Aggregate & Report",
                "Compute mean ρ, % divergent queries, dominance ratios, entropy. Run Mann-Whitney U test for statistical significance (p < 0.05).",
                "panel-good",
            ),
        ]:
            st.markdown(
                f"""
            <div class="panel {step[3]}">
              <div class="step-num">{step[0]}</div>
              <div class="step-title" style="font-family:'DM Sans',sans-serif;font-size:14px;">{step[1]}</div>
              <div style="font-family:'IBM Plex Mono',monospace;font-size:11px;color:#666666;margin-top:6px;line-height:1.6;">{step[2]}</div>
            </div>""",
                unsafe_allow_html=True,
            )

    with col_m2:
        st.markdown(
            '<div class="label" style="margin-bottom:12px;">Metrics Reference</div>',
            unsafe_allow_html=True,
        )

        metrics_info = [
            (
                "Influence Score",
                "1 − ROUGE-L(baseline, ablated_i)",
                "Magnitude of output change when document i is removed. Higher = more influential.",
                "var(--accent)",
            ),
            (
                "Spearman ρ",
                "spearmanr(retrieval_ranks, influence_ranks)",
                "Rank correlation between BM25 order and ablation-derived influence order. ρ < 0.7 = divergent.",
                "var(--accent3)",
            ),
            (
                "Dominance Ratio",
                "max(influence) / Σ(influence)",
                "Whether one document dominates generation. High ratio = unbalanced retrieval.",
                "var(--warn)",
            ),
            (
                "Influence Entropy",
                "−Σ p_i · log(p_i)",
                "Shannon entropy of normalised influence scores. Low entropy = concentrated influence.",
                "var(--accent2)",
            ),
            (
                "Failure Mode",
                "ret_rank ≤ 3 AND inf_rank ≥ k−2",
                "Document retrieved highly but barely influences output. Key diagnostic signal.",
                "var(--accent2)",
            ),
        ]

        for name, formula, desc, color in metrics_info:
            st.markdown(
                f"""
            <div class="panel" style="border-left:3px solid {color};margin-bottom:10px;">
              <div style="font-family:'DM Sans',sans-serif;font-weight:600;color:#111111;font-size:13px;">{name}</div>
              <div style="font-family:'IBM Plex Mono',monospace;font-size:11px;color:{color};margin:6px 0;padding:4px 8px;background:var(--surface);border-radius:4px;">{formula}</div>
              <div style="font-size:12px;color:#666666;line-height:1.6;">{desc}</div>
            </div>""",
                unsafe_allow_html=True,
            )

        st.markdown(
            """
        <div class="panel" style="margin-top:8px;">
          <div class="label" style="margin-bottom:8px;">References</div>
          <div style="font-family:'IBM Plex Mono',monospace;font-size:10px;color:#666666;line-height:1.8;">
            [1] ArXiv 2507.04480 — Source Attribution in RAG<br>
            [2] Wu et al. (2024) — MedGraphRAG<br>
            [3] Frangou et al. (2025) — SLR of RAG Techniques<br>
            [4] Preprints 202512.0359 — RAG for Enterprise KM
          </div>
        </div>""",
            unsafe_allow_html=True,
        )
