"""
Page 1 - Main Analyzer.
Prompt input -> run analysis -> show classification, confidence, and responses.
"""

from datetime import datetime
import math

import plotly.graph_objects as go
import streamlit as st

from analyzer import HallucinationAnalyzer


@st.cache_resource
def load_analyzer(model_name, semantic_threshold):
    return HallucinationAnalyzer(
        model_name=model_name,
        semantic_threshold=semantic_threshold,
    )


def classify(final_risk: float, eigen_score: float, external_similarity: float) -> tuple[str, str, float]:
    """Return the headline label, badge CSS class, and confidence percentage."""
    norm_eigen = 1.0 / (1.0 + math.exp(-eigen_score / max(1.0, abs(eigen_score) + 1e-9)))
    eigen_high = norm_eigen > 0.5
    confidence = max(0.0, min(100.0, round((1 - final_risk) * 100, 1)))

    if external_similarity > 0.5:
        return "Reliable", "badge-reliable", confidence
    if eigen_high:
        return "Uncertain Hallucination", "badge-uncertain", confidence
    return "Confident Hallucination", "badge-confident-hall", confidence


def confidence_color(pct: float) -> str:
    if pct >= 70:
        return "linear-gradient(90deg,#059669,#34d399)"
    if pct >= 40:
        return "linear-gradient(90deg,#d97706,#fbbf24)"
    return "linear-gradient(90deg,#dc2626,#f87171)"


def gauge(pct: float, label: str) -> go.Figure:
    color = "#10b981" if pct >= 70 else ("#f59e0b" if pct >= 40 else "#ef4444")
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=pct,
            number={"suffix": "%", "font": {"size": 36, "color": "#111827"}},
            title={"text": label, "font": {"size": 15, "color": "#4b5563"}},
            gauge={
                "axis": {
                    "range": [0, 100],
                    "tickcolor": "#9ca3af",
                    "tickfont": {"color": "#4b5563"},
                },
                "bar": {"color": color, "thickness": 0.25},
                "bgcolor": "rgba(0,0,0,0.05)",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 40], "color": "rgba(239,68,68,0.15)"},
                    {"range": [40, 70], "color": "rgba(245,158,11,0.15)"},
                    {"range": [70, 100], "color": "rgba(16,185,129,0.15)"},
                ],
            },
        )
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=40, b=20, l=20, r=20),
        height=230,
    )
    return fig


def plot_eigenvalues(eigenvalues):
    if not eigenvalues:
        return go.Figure()

    sorted_eigenvalues = sorted((float(v) for v in eigenvalues), reverse=True)
    x_values = list(range(1, len(sorted_eigenvalues) + 1))

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=x_values,
            y=sorted_eigenvalues,
            marker=dict(
                color=sorted_eigenvalues,
                colorscale="Tealgrn",
                showscale=True,
                colorbar=dict(title="Magnitude", tickfont=dict(color="#475569")),
                line=dict(color="rgba(15,23,42,0.18)", width=1),
            ),
            hovertemplate="Eigenvalue %{x}<br>Magnitude %{y:.4f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x_values,
            y=sorted_eigenvalues,
            mode="lines+markers",
            line=dict(color="#0f766e", width=2),
            marker=dict(size=7, color="#0f766e"),
            hoverinfo="skip",
            showlegend=False,
        )
    )
    fig.update_layout(
        title=dict(
            text="Hidden-State Covariance Eigenvalue Spectrum",
            font=dict(color="#111827", size=14),
        ),
        xaxis=dict(
            title="Eigenvalue Rank",
            color="#4b5563",
            tickmode="linear",
            dtick=1,
        ),
        yaxis=dict(title="Magnitude", color="#4b5563"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0.03)",
        height=340,
        margin=dict(t=50, b=40, l=50, r=20),
    )
    return fig


def render(cfg: dict):
    st.markdown(
        """
    <div class='hero'>
        <h1>Inside the Mind of an LLM</h1>
        <p>Detect hallucinations in AI responses using internal activations and factual grounding.</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # ── State machine ────────────────────────────────────────────────────────
    # Branch A: results exist in session → show them + Clear button
    # Branch B: no results → show input form
    has_results = bool(st.session_state.get("last_results"))

    # ── Branch A: display persisted results ──────────────────────────────────
    if has_results:
        results = st.session_state["last_results"]
        last_prompt = st.session_state.get("last_prompt", "")

        # Prompt header + Clear button
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown(
            f"<p style='color:rgba(15,23,42,0.5); font-size:0.8rem; margin:0 0 4px; font-weight:600; "
            f"letter-spacing:0.06em;'>ANALYZED PROMPT</p>"
            f"<p style='color:#111827; font-size:1rem; font-weight:600; margin:0 0 14px;'>{last_prompt}</p>",
            unsafe_allow_html=True,
        )
        if st.button("🗑️  Clear — Analyze a New Prompt", use_container_width=True, key="clear_results"):
            st.session_state["last_results"] = None
            st.session_state["last_prompt"] = ""
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

        # ── Results display ──────────────────────────────────────────────────
        final_risk = results["final_risk"]
        eigen_score = results["eigen"]["eigen_score"]
        ext_sim = results["external"]["external_consistency"]
        label, badge, conf_pct = classify(final_risk, eigen_score, ext_sim)

        st.success("Analysis complete.")

        col_a, col_b, col_c = st.columns([1, 1, 1])
        with col_a:
            st.markdown(
                f"<div class='card' style='text-align:center;'>"
                f"<p style='color:rgba(0,0,0,0.5); margin:0; font-size:0.85rem;'>CLASSIFICATION</p>"
                f"<span class='{badge}' style='font-size:1.1rem; margin-top:8px; display:inline-block;'>{label}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )
        with col_b:
            fill_color = confidence_color(conf_pct)
            st.markdown(
                f"<div class='card' style='text-align:center;'>"
                f"<p style='color:rgba(0,0,0,0.5); margin:0 0 6px; font-size:0.85rem;'>CONFIDENCE</p>"
                f"<span style='font-size:2rem; font-weight:700; color:#111827;'>{conf_pct}%</span>"
                f"<div class='meter-wrap' style='margin-top:8px;'>"
                f"<div class='meter-fill' style='width:{conf_pct}%; background:{fill_color};'></div>"
                f"</div></div>",
                unsafe_allow_html=True,
            )
        with col_c:
            st.plotly_chart(gauge(conf_pct, "Confidence Meter"), use_container_width=True)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Final Risk", f"{final_risk:.4f}")
        m2.metric("EigenScore", f"{eigen_score:.4f}")
        m3.metric("Stability", f"{results['stability']['stability_score']:.4f}")
        m4.metric("External Similarity", f"{ext_sim:.4f}")

        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Generated Responses")
        for i, response in enumerate(results["responses"], 1):
            with st.expander(f"Response {i}", expanded=(i == 1)):
                st.write(response)
        st.markdown("</div>", unsafe_allow_html=True)

        if results["external"]["ground_truth"] != "N/A":
            st.markdown(
                f"<div class='card'>"
                f"<p style='color:#0369a1; font-weight:700; margin:0 0 4px;'>"
                f"Ground Truth ({results['external'].get('ground_truth_source', 'N/A')})</p>"
                f"<p style='color:#111827; margin:0;'>{results['external']['ground_truth']}</p>"
                f"</div>",
                unsafe_allow_html=True,
            )
        else:
            st.info(
                "No ground truth was found across datasets "
                "(TruthfulQA, CoQA, SQuAD, NQ, TriviaQA) for this prompt. "
                "External similarity was set to a neutral fallback."
            )

        st.subheader("Eigenvalue Spectrum")
        st.plotly_chart(
            plot_eigenvalues(results["eigen"]["eigenvalues"]),
            use_container_width=True,
        )
        st.caption("Eigenvalues are shown in descending order and ranked from 1 to N for easier reading.")
        st.info("Switch to Explanation in the sidebar for a plain-language breakdown of these results.")
        return  # nothing more to do

    # ── Branch B: no results yet — show input form ───────────────────────────
    prompt = st.text_area(
        "Enter your question or prompt:",
        height=110,
        placeholder="e.g., What is the capital of Australia?",
        key="main_prompt",
    )
    analyze_btn = st.button("Run Analysis", use_container_width=True, key="run_analysis")

    if not analyze_btn:
        _show_example()
        return

    if not prompt.strip():
        st.warning("Please enter a prompt before running analysis.")
        return

    with st.spinner("Loading model and running analysis. This may take a minute."):
        try:
            analyzer = load_analyzer(cfg["model_name"], cfg["semantic_threshold"])
            results = analyzer.analyze(
                prompt=prompt,
                num_responses=cfg["num_responses"],
                max_length=cfg["max_length"],
                temperature=cfg["temperature"],
                alpha=cfg["alpha"],
                beta=cfg["beta"],
                w1=cfg["w1"],
                w2=cfg["w2"],
                w3=cfg["w3"],
            )
        except Exception as exc:
            st.error(f"Analysis failed: {exc}")
            st.exception(exc)
            return

    # Persist results and record history
    st.session_state["last_results"] = results
    st.session_state["last_prompt"] = prompt

    if "analysis_history" not in st.session_state:
        st.session_state["analysis_history"] = []
    st.session_state["analysis_history"].append(
        {
            "prompt": prompt,
            "final_risk": results["final_risk"],
            "eigen_score": results["eigen"]["eigen_score"],
            "stability": results["stability"]["stability_score"],
            "grounding": results["grounding"]["grounding_score"],
            "ext_sim": results["external"]["external_consistency"],
            "gt_source": results["external"].get("ground_truth_source", "N/A"),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
    )

    # Rerun so the results branch renders cleanly (avoids double-render on same run)
    st.rerun()


def _show_example():
    st.markdown(
        """
    <div class='card' style='text-align:center; padding:2.5rem 1rem;'>
        <span style='font-size:3rem;'>Analyze</span>
        <h3 style='color:#0369a1; margin:0.5rem 0;'>Ready to Analyze</h3>
        <p style='color:rgba(0,0,0,0.6); max-width:520px; margin:0 auto;'>
            Type a question above and click <strong>Run Analysis</strong>.<br/>
            The system will generate multiple responses, compute internal metrics
            (EigenScore, Stability, Grounding), and cross-check with reference datasets.
        </p>
    </div>""",
        unsafe_allow_html=True,
    )
