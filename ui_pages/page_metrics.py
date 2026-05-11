"""
Page 4 - Detailed Metrics.
Shows EigenScore, Stability, Grounding, and External Similarity with breakdowns.
"""

import json
import math

import plotly.graph_objects as go
import streamlit as st


LIGHT_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0.03)",
    font=dict(color="#4b5563"),
    margin=dict(t=50, b=40, l=50, r=20),
)


def _norm_eigen(eigen_score: float) -> float:
    return 1.0 / (1.0 + math.exp(-eigen_score / max(1.0, abs(eigen_score) + 1e-9)))


def _pill(text: str, color: str) -> str:
    return (
        f"<span style='background:{color}22; color:{color}; border:1px solid {color}55; "
        f"padding:3px 12px; border-radius:30px; font-size:0.82rem; font-weight:700;'>{text}</span>"
    )


def _status_pill(value: float, high_good: bool = True) -> str:
    good = value > 0.65 if high_good else value < 0.35
    medium = 0.35 <= value <= 0.65
    if good:
        return _pill("Good", "#10b981")
    if medium:
        return _pill("Medium", "#f59e0b")
    return _pill("Poor", "#ef4444")


def radar_chart(metrics: dict) -> go.Figure:
    categories = ["Consistency", "Stability", "Grounding", "External Similarity", "Confidence"]
    values = [
        metrics["consistency"],
        metrics["stability"],
        metrics["grounding"],
        metrics["ext_sim"],
        metrics["confidence"] / 100,
    ]
    values_closed = values + [values[0]]
    cats_closed = categories + [categories[0]]

    fig = go.Figure()
    fig.add_trace(
        go.Scatterpolar(
            r=values_closed,
            theta=cats_closed,
            fill="toself",
            fillcolor="rgba(2,132,199,0.16)",
            line=dict(color="#0284c7", width=2),
            name="Metrics",
        )
    )
    fig.update_layout(
        polar=dict(
            bgcolor="rgba(0,0,0,0.03)",
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                tickfont=dict(color="#4b5563"),
                gridcolor="rgba(0,0,0,0.1)",
            ),
            angularaxis=dict(
                tickfont=dict(color="#111827"),
                gridcolor="rgba(0,0,0,0.1)"),
        ),
        height=380,
        showlegend=False,
        title=dict(text="Metric Radar", font=dict(color="#111827", size=14)),
        **{k: v for k, v in LIGHT_LAYOUT.items() if k != "plot_bgcolor"},
    )
    return fig


def layer_stability_chart(layer_sims: list) -> go.Figure:
    x_values = [f"L{i}->L{i + 1}" for i in range(len(layer_sims))]
    fig = go.Figure(
        go.Scatter(
            x=x_values,
            y=layer_sims,
            mode="lines+markers",
            line=dict(color="#60a5fa", width=2),
            marker=dict(size=7, color="#60a5fa", line=dict(color="#1e3a5f", width=1.5)),
            fill="tozeroy",
            fillcolor="rgba(96,165,250,0.08)",
            hovertemplate="%{x}: %{y:.4f}<extra></extra>",
        )
    )
    fig.add_hline(
        y=0.5,
        line_dash="dash",
        line_color="#f59e0b",
        annotation_text="Stability Threshold (0.5)",
        annotation_font_color="#f59e0b",
    )
    fig.update_layout(
        title=dict(
            text="Layer-wise Stability",
            font=dict(color="#111827", size=13),
        ),
        xaxis=dict(title="Layer transition"),
        yaxis=dict(title="Similarity", range=[0, 1.05]),
        height=300,
        **LIGHT_LAYOUT,
    )
    return fig


def similarity_bar(similarities: list) -> go.Figure:
    colors = ["#10b981" if score > 0.5 else "#ef4444" for score in similarities]
    fig = go.Figure(
        go.Bar(
            x=[f"Response {i + 1}" for i in range(len(similarities))],
            y=similarities,
            marker_color=colors,
            text=[f"{score:.3f}" for score in similarities],
            textposition="outside",
            textfont=dict(color="#111827"),
        )
    )
    fig.add_hline(
        y=0.5,
        line_dash="dash",
        line_color="#f59e0b",
        annotation_text="Similarity threshold (0.5)",
        annotation_font_color="#f59e0b",
    )
    fig.update_layout(
        title=dict(text="Per-response Similarity to Ground Truth", font=dict(color="#111827", size=13)),
        yaxis=dict(range=[0, 1.15]),
        height=300,
        **LIGHT_LAYOUT,
    )
    return fig


def internal_risk_breakdown(results: dict) -> go.Figure:
    ir = results["internal_risk"]
    labels = ["EigenScore Component", "Stability Component", "Grounding Component"]
    values = [
        ir["eigen_score_component"],
        ir["stability_component"],
        ir["grounding_component"],
    ]
    colors = ["#0284c7", "#60a5fa", "#34d399"]
    fig = go.Figure(
        go.Bar(
            x=labels,
            y=values,
            marker_color=colors,
            text=[f"{value:.4f}" for value in values],
            textposition="outside",
            textfont=dict(color="#111827"),
        )
    )
    fig.update_layout(
        title=dict(text="Internal Risk Component Breakdown", font=dict(color="#111827", size=13)),
        yaxis=dict(title="Risk contribution"),
        height=300,
        **LIGHT_LAYOUT,
    )
    return fig


def weights_pie(results: dict) -> go.Figure:
    alpha = results["weights"]["alpha"]
    beta = results["weights"]["beta"]
    internal_risk = results["internal_risk"]["internal_risk"]
    external_risk = results["external"]["external_risk"]
    fig = go.Figure(
        go.Pie(
            labels=["Internal Risk (alpha)", "External Risk (beta)"],
            values=[alpha * internal_risk, beta * external_risk],
            marker=dict(colors=["#0284c7", "#0f766e"], line=dict(color="rgba(0,0,0,0)", width=0)),
            hole=0.5,
            textfont=dict(color="#111827"),
        )
    )
    fig.update_layout(
        title=dict(text="Final Risk Composition", font=dict(color="#111827", size=13)),
        legend=dict(font=dict(color="#4b5563")),
        height=300,
        **{k: v for k, v in LIGHT_LAYOUT.items() if k != "plot_bgcolor"},
    )
    return fig


def _detail_card(title: str, value_str: str, pill_html: str, meaning: str, border_color: str):
    st.markdown(
        f"""
    <div class='card' style='border-left:4px solid {border_color};'>
        <div style='display:flex; justify-content:space-between; align-items:center;'>
            <span style='color:#111827; font-weight:700; font-size:1rem;'>{title}</span>
            {pill_html}
        </div>
        <p style='color:#0369a1; font-size:1.5rem; font-weight:700; margin:4px 0;'>{value_str}</p>
        <p style='color:rgba(0,0,0,0.6); font-size:0.85rem; margin:0;'>{meaning}</p>
    </div>""",
        unsafe_allow_html=True,
    )


def render():
    st.markdown(
        """
    <div class='hero'>
        <h1>Detailed Metrics</h1>
        <p>Advanced view of EigenScore, Stability, Grounding, and External Similarity.</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    results = st.session_state.get("last_results")

    if results is None:
        st.markdown(
            """
        <div class='card' style='text-align:center; padding:2.5rem;'>
            <h3 style='color:#0369a1;'>No Analysis Yet</h3>
            <p style='color:rgba(0,0,0,0.6);'>
                Run an analysis on the <strong>Analyzer</strong> page first.
            </p>
        </div>""",
            unsafe_allow_html=True,
        )
        return

    eigen_score = results["eigen"]["eigen_score"]
    stability = results["stability"]["stability_score"]
    grounding = results["grounding"]["grounding_score"]
    ext_sim = results["external"]["external_consistency"]
    final_risk = results["final_risk"]
    confidence = round((1 - final_risk) * 100, 1)
    norm_eigen = _norm_eigen(eigen_score)
    consistency = 1 - norm_eigen

    c1, c2 = st.columns(2)
    with c1:
        _detail_card(
            "EigenScore (raw)",
            f"{eigen_score:.4f}",
            _status_pill(consistency, high_good=True),
            "Lower and more negative values usually mean the sampled answers stayed closer together.",
            "#0284c7",
        )
        _detail_card(
            "Grounding Score",
            f"{grounding:.4f}",
            _status_pill(grounding, high_good=True),
            "Measures how strongly the generated answer attends back to the question tokens.",
            "#34d399",
        )
    with c2:
        _detail_card(
            "Stability Score",
            f"{stability:.4f}",
            _status_pill(stability, high_good=True),
            "Measures similarity between adjacent transformer layers across the response.",
            "#60a5fa",
        )
        _detail_card(
            "External Similarity",
            f"{ext_sim:.4f}",
            _status_pill(ext_sim, high_good=True),
            "Measures how closely the generated responses match the reference answer.",
            "#f59e0b",
        )

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    fr1, fr2, fr3 = st.columns(3)
    fr1.metric("Final Risk Score", f"{final_risk:.4f}")
    fr2.metric("Confidence", f"{confidence}%")
    fr3.metric("Samples Generated", results["eigen"]["num_responses"])
    fr4, fr5 = st.columns(2)
    fr4.metric("Internal Risk", f"{results['internal_risk']['internal_risk']:.4f}")
    fr5.metric("External Risk", f"{results['external']['external_risk']:.4f}")
    st.markdown("</div>", unsafe_allow_html=True)

    col_rad, col_pie = st.columns(2)
    with col_rad:
        st.plotly_chart(
            radar_chart(
                {
                    "consistency": consistency,
                    "stability": stability,
                    "grounding": grounding,
                    "ext_sim": ext_sim,
                    "confidence": confidence,
                }
            ),
            use_container_width=True,
        )
    with col_pie:
        st.plotly_chart(weights_pie(results), use_container_width=True)

    st.plotly_chart(internal_risk_breakdown(results), use_container_width=True)

    if "layer_similarities" in results["stability"]:
        st.plotly_chart(
            layer_stability_chart(results["stability"]["layer_similarities"]),
            use_container_width=True,
        )

    if results["external"]["ground_truth"] != "N/A":
        st.plotly_chart(
            similarity_bar(results["external"]["similarities"]),
            use_container_width=True,
        )
        st.markdown(
            f"""
        <div class='card'>
            <p style='color:#0369a1; font-weight:700; margin:0 0 4px;'>Ground Truth Used</p>
            <p style='color:#111827; margin:0;'>{results['external']['ground_truth']}</p>
        </div>""",
            unsafe_allow_html=True,
        )

    with st.expander("Feature Clipping (INSIDE paper)"):
        clipped = results["eigen"].get("clipping_applied", False)
        st.markdown(
            f"""
- **Clipping applied:** {"Yes" if clipped else "No (memory bank was empty)"}
- Hidden-state activations are clipped per feature dimension using percentile thresholds
  derived from the accumulated memory bank.
- This helps keep outlier activations from dominating the EigenScore.
- Reference: *INSIDE: LLMs' Internal States Are Good Indicators of Factual Accuracy* (ICLR 2024)
"""
        )

    with st.expander("Raw Results (JSON)"):
        safe = {
            "prompt": results["prompt"],
            "final_risk": results["final_risk"],
            "eigen_score": results["eigen"]["eigen_score"],
            "stability": results["stability"]["stability_score"],
            "grounding": results["grounding"]["grounding_score"],
            "ext_sim": results["external"]["external_consistency"],
            "weights": results["weights"],
            "internal_risk": results["internal_risk"],
            "external_risk": results["external"]["external_risk"],
        }
        st.code(json.dumps(safe, indent=2), language="json")
