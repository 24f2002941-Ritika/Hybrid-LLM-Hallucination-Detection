"""
Multi-page Hallucination Detection System.
Pages: Analyzer | Explanation | Evaluation | Detailed Metrics | Heatmap | History
"""

import streamlit as st

import ui_pages.page_analyzer as p1
import ui_pages.page_evaluation as p3
import ui_pages.page_explanation as p2
import ui_pages.page_heatmap as p6
import ui_pages.page_history as p5
import ui_pages.page_metrics as p4


st.set_page_config(
    page_title="Inside the Mind of an LLM",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif;
}

.stApp {
    background:
        radial-gradient(circle at top left, rgba(14, 165, 233, 0.18), transparent 35%),
        radial-gradient(circle at top right, rgba(16, 185, 129, 0.14), transparent 30%),
        linear-gradient(135deg, #f8fafc, #edf6ff 45%, #ecfdf5);
}

section[data-testid="stSidebar"] {
    background: rgba(255, 255, 255, 0.82) !important;
    border-right: 1px solid rgba(15, 23, 42, 0.08);
}

.card {
    background: rgba(255, 255, 255, 0.9);
    border: 1px solid rgba(15, 23, 42, 0.08);
    border-radius: 18px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1rem;
    backdrop-filter: blur(10px);
    box-shadow: 0 12px 30px rgba(15, 23, 42, 0.06);
}

.hero {
    text-align: center;
    padding: 1.8rem 0 1rem 0;
}

.hero h1 {
    margin-bottom: 0.35rem;
    font-size: 2.7rem;
    font-weight: 700;
    letter-spacing: -0.04em;
    color: #0f172a;
}

.hero p {
    color: rgba(15, 23, 42, 0.68);
    font-size: 1.02rem;
}

.badge-reliable,
.badge-uncertain,
.badge-hallucination,
.badge-confident-hall {
    display: inline-block;
    padding: 6px 18px;
    border-radius: 999px;
    color: #fff;
    font-weight: 700;
    font-size: 0.98rem;
}

.badge-reliable { background: linear-gradient(90deg, #059669, #10b981); }
.badge-uncertain { background: linear-gradient(90deg, #d97706, #f59e0b); }
.badge-hallucination { background: linear-gradient(90deg, #dc2626, #ef4444); }
.badge-confident-hall { background: linear-gradient(90deg, #b91c1c, #ef4444); }

.meter-wrap {
    background: rgba(15, 23, 42, 0.08);
    border-radius: 999px;
    height: 22px;
    width: 100%;
    overflow: hidden;
    margin: 8px 0;
}

.meter-fill {
    height: 100%;
    border-radius: 999px;
    transition: width 0.6s ease;
}

.reason-item {
    padding: 8px 12px;
    margin: 4px 0;
    border-left: 3px solid #0284c7;
    background: rgba(2, 132, 199, 0.08);
    border-radius: 0 8px 8px 0;
    color: rgba(15, 23, 42, 0.88);
    font-size: 0.92rem;
}

div[data-baseweb="tab-list"] {
    background: rgba(15, 23, 42, 0.04);
    border-radius: 12px;
    padding: 4px;
    gap: 4px;
}

div[data-baseweb="tab"] {
    border-radius: 8px !important;
    font-weight: 500;
}

[data-testid="stMetricLabel"] { color: rgba(15, 23, 42, 0.6) !important; }
[data-testid="stMetricValue"] { color: #111827 !important; }

textarea, input {
    background: rgba(255, 255, 255, 0.92) !important;
    color: #111827 !important;
    border-radius: 10px !important;
    border: 1px solid rgba(15, 23, 42, 0.1) !important;
}

.stButton > button {
    background: linear-gradient(135deg, #0284c7, #0f766e);
    color: white;
    border: none;
    border-radius: 10px;
    font-weight: 700;
    padding: 0.6rem 1.2rem;
    transition: transform 0.15s, box-shadow 0.15s;
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 10px 24px rgba(2, 132, 199, 0.22);
}

details summary {
    color: #0369a1 !important;
    font-weight: 700;
}

.footer {
    text-align: center;
    color: rgba(15, 23, 42, 0.45);
    font-size: 0.8rem;
    padding: 1.5rem 0 0.5rem;
}
</style>
""",
    unsafe_allow_html=True,
)

st.sidebar.markdown(
    """
<div style='text-align:center; padding: 1rem 0 0.5rem;'>
    <h2 style='color:#0369a1; margin:0; font-size:1.2rem; font-weight:700;'>Inside the Mind of an LLM</h2>
    <p style='color:rgba(15,23,42,0.5); font-size:0.75rem; margin:0;'>Hallucination Detection System</p>
</div>
<hr style='border-color:rgba(15,23,42,0.1); margin: 0.8rem 0;'/>
""",
    unsafe_allow_html=True,
)

PAGE_ICONS = {
    "Analyzer": p1,
    "Explanation": p2,
    "Evaluation": p3,
    "Detailed Metrics": p4,
    "Hallucination Heatmap": p6,
    "History": p5,
}

page = st.sidebar.radio("Navigate", list(PAGE_ICONS.keys()), label_visibility="collapsed")

st.sidebar.markdown("<hr style='border-color:rgba(15,23,42,0.1);'/>", unsafe_allow_html=True)
st.sidebar.subheader("Model Settings")

MODEL_LABELS = {
    "gpt2": "GPT-2 (117M)",
    "gpt2-medium": "GPT-2 Medium (345M)",
    "gpt2-large": "GPT-2 Large (774M)",
    "EleutherAI/gpt-neo-125M": "GPT-Neo 125M",
    "EleutherAI/gpt-neo-1.3B": "GPT-Neo 1.3B",
    "EleutherAI/gpt-neo-2.7B": "GPT-Neo 2.7B",
    "EleutherAI/pythia-2.8b": "Pythia 2.8B",
    "facebook/opt-6.7b": "OPT 6.7B (High VRAM/RAM)",
}

model_name = st.sidebar.selectbox(
    "Model",
    list(MODEL_LABELS.keys()),
    format_func=lambda x: MODEL_LABELS[x],
)

semantic_threshold = st.sidebar.slider("Semantic Match Threshold", 0.40, 1.00, 0.55, 0.05)
num_responses = st.sidebar.slider("Number of Responses", 1, 10, 5)
max_length = st.sidebar.slider("Max Generation Length", 10, 100, 50)
temperature = st.sidebar.slider("Temperature", 0.1, 2.0, 0.8, 0.1)

st.sidebar.subheader("Risk Weights")
alpha = st.sidebar.slider("Alpha (Internal)", 0.0, 1.0, 0.6, 0.1)
beta = st.sidebar.slider("Beta (External)", 0.0, 1.0, 0.4, 0.1)

st.sidebar.subheader("Metric Weights")
w1 = st.sidebar.slider("w1 - EigenScore", 0.0, 1.0, 0.4, 0.1)
w2 = st.sidebar.slider("w2 - Stability", 0.0, 1.0, 0.3, 0.1)
w3 = st.sidebar.slider("w3 - Grounding", 0.0, 1.0, 0.3, 0.1)

cfg = dict(
    model_name=model_name,
    semantic_threshold=semantic_threshold,
    num_responses=num_responses,
    max_length=max_length,
    temperature=temperature,
    alpha=alpha,
    beta=beta,
    w1=w1,
    w2=w2,
    w3=w3,
)

PAGE_ICONS[page].render(cfg) if page in {"Analyzer", "Evaluation"} else PAGE_ICONS[page].render()

st.markdown(
    "<div class='footer'>Inside the Mind of an LLM | Powered by GPT-2 | GPT-Neo | TransformerLens | TruthfulQA</div>",
    unsafe_allow_html=True,
)
