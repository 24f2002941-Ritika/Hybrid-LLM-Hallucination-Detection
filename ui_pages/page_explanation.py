"""
Page 2 - Explanation Page.
Converts raw metrics into user-friendly reasoning.
"""

import math

import streamlit as st


def _eigen_level(eigen_score: float) -> str:
    norm = 1.0 / (1.0 + math.exp(-eigen_score / max(1.0, abs(eigen_score) + 1e-9)))
    return "High" if norm > 0.5 else "Low"


def build_reasons(results: dict) -> list[str]:
    reasons = []
    eigen_score = results["eigen"]["eigen_score"]
    stability = results["stability"]["stability_score"]
    grounding = results["grounding"]["grounding_score"]
    ext_sim = results["external"]["external_consistency"]

    if _eigen_level(eigen_score) == "High":
        reasons.append("The model gave different answers across samples, which suggests uncertainty.")
    else:
        reasons.append("The model gave fairly consistent answers across samples, which suggests confidence.")

    if stability < 0.5:
        reasons.append("Its internal reasoning shifted noticeably across layers, so the answer path was unstable.")
    else:
        reasons.append("Its internal reasoning stayed relatively stable across layers.")

    if grounding < 0.5:
        reasons.append("The response was only weakly grounded in the original question.")
    else:
        reasons.append("The response stayed well grounded in the original question.")

    if ext_sim < 0.5:
        reasons.append("The answer did not match known facts strongly enough.")
    else:
        reasons.append("The answer matched the reference facts reasonably well.")

    return reasons


def classify_verbose(results: dict) -> tuple[str, str, str]:
    eigen_score = results["eigen"]["eigen_score"]
    ext_sim = results["external"]["external_consistency"]

    norm_eigen = 1.0 / (1.0 + math.exp(-eigen_score / max(1.0, abs(eigen_score) + 1e-9)))
    eigen_high = norm_eigen > 0.5

    if ext_sim > 0.5:
        return (
            "This answer appears to be correct and reliable.",
            "Reliable",
            "The model's answer aligns with known reference information and did not show strong warning signs internally.",
        )
    if eigen_high:
        return (
            "This answer looks uncertain.",
            "Uncertain Hallucination",
            "The model varied across samples and the result also failed to match the reference facts closely.",
        )
    return (
        "This answer is likely incorrect despite sounding confident.",
        "Confident but Incorrect Answer",
        "The model stayed internally consistent, but the answer still did not line up with the reference facts.",
    )


def _metric_bar(label: str, value: float, color: str, tooltip: str):
    pct = round(value * 100, 1)
    st.markdown(
        f"""
    <div style='margin-bottom:14px;'>
        <div style='display:flex; justify-content:space-between; margin-bottom:4px;'>
            <span style='color:#334155; font-size:0.9rem; font-weight:600;'>{label}</span>
            <span style='color:#0f172a; font-size:0.9rem; font-weight:700;'>{pct}%</span>
        </div>
        <div class='meter-wrap'>
            <div class='meter-fill' style='width:{pct}%; background:{color};'></div>
        </div>
        <span style='color:rgba(15,23,42,0.55); font-size:0.78rem;'>{tooltip}</span>
    </div>
    """,
        unsafe_allow_html=True,
    )


def render():
    st.markdown(
        """
    <div class='hero'>
        <h1>Plain-Language Explanation</h1>
        <p>What the system found, explained in simple terms.</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    results = st.session_state.get("last_results")
    prompt = st.session_state.get("last_prompt", "")

    if results is None:
        st.markdown(
            """
        <div class='card' style='text-align:center; padding:2.5rem;'>
            <h3 style='color:#0369a1;'>No Analysis Yet</h3>
            <p style='color:rgba(15,23,42,0.55);'>
                Run an analysis on the <strong>Analyzer</strong> page first, then come back here.
            </p>
        </div>""",
            unsafe_allow_html=True,
        )
        return

    headline, sub_label, conclusion = classify_verbose(results)
    reasons = build_reasons(results)

    st.markdown(
        f"""
    <div class='card'>
        <p style='color:#0369a1; font-size:0.8rem; margin:0 0 4px; font-weight:700;'>YOUR QUESTION</p>
        <p style='color:#0f172a; margin:0; font-size:1rem;'>"{prompt}"</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    verdict_color = {
        "Reliable": "#059669",
        "Uncertain Hallucination": "#d97706",
        "Confident but Incorrect Answer": "#dc2626",
    }.get(sub_label, "#0369a1")

    st.markdown(
        f"""
    <div class='card' style='border-left: 4px solid {verdict_color};'>
        <h3 style='color:#0f172a; margin:0 0 6px;'>{headline}</h3>
        <span style='background:{verdict_color}22; color:{verdict_color};
              border:1px solid {verdict_color}55; padding:4px 14px;
              border-radius:30px; font-size:0.85rem; font-weight:700;'>
            {sub_label}
        </span>
    </div>
    """,
        unsafe_allow_html=True,
    )

    st.subheader("Reasons")
    for reason in reasons:
        st.markdown(f"<div class='reason-item'>{reason}</div>", unsafe_allow_html=True)

    st.subheader("What the Numbers Mean")

    eigen_score = results["eigen"]["eigen_score"]
    stability = results["stability"]["stability_score"]
    grounding = results["grounding"]["grounding_score"]
    ext_sim = results["external"]["external_consistency"]
    norm_eigen = 1.0 / (1.0 + math.exp(-eigen_score / max(1.0, abs(eigen_score) + 1e-9)))

    col1, col2 = st.columns(2)
    with col1:
        _metric_bar(
            "Answer Consistency (1 - EigenScore)",
            1 - norm_eigen,
            "linear-gradient(90deg,#0284c7,#38bdf8)",
            "Higher means the model repeated itself more consistently across samples.",
        )
        _metric_bar(
            "Layer Stability",
            stability,
            "linear-gradient(90deg,#2563eb,#60a5fa)",
            "Higher means the hidden-state trajectory changed less from layer to layer.",
        )
    with col2:
        _metric_bar(
            "Grounding",
            grounding,
            "linear-gradient(90deg,#0f766e,#2dd4bf)",
            "Higher means the answer stayed closer to the original question.",
        )
        _metric_bar(
            "Factual Similarity",
            ext_sim,
            "linear-gradient(90deg,#059669,#34d399)",
            "Higher means the answer better matched the reference ground truth (TruthfulQA, CoQA, SQuAD, NQ, TriviaQA).",
        )

    st.markdown(
        f"""
    <div class='card' style='margin-top:1rem; border-top:3px solid {verdict_color};'>
        <p style='color:#0369a1; font-weight:700; margin:0 0 6px;'>Conclusion</p>
        <p style='color:#0f172a; margin:0; line-height:1.6;'>{conclusion}</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    with st.expander("Metric Glossary"):
        st.markdown(
            """
| Metric | When High | When Low |
|---|---|---|
| **EigenScore** | Model gave more varied answers | Model gave more consistent answers |
| **Stability** | Reasoning stayed stable across layers | Reasoning changed more across layers |
| **Grounding** | Answer stayed connected to the question | Answer drifted away from the question |
| **External Similarity** | Answer matched known facts | Answer did not match known facts |
"""
        )
