"""
Page 5 - Analysis History.
Tracks past analyses in session state.
"""

import json
import math

import pandas as pd
import plotly.graph_objects as go
import streamlit as st


LIGHT_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0.03)",
    font=dict(color="#475569"),
    margin=dict(t=50, b=40, l=50, r=20),
)


def _classify(final_risk: float, eigen_score: float, ext_sim: float) -> tuple[str, str]:
    norm_eigen = 1.0 / (1.0 + math.exp(-eigen_score / max(1.0, abs(eigen_score) + 1e-9)))
    eigen_high = norm_eigen > 0.5
    if ext_sim > 0.5:
        return "Reliable", "badge-reliable"
    if eigen_high:
        return "Uncertain", "badge-uncertain"
    return "Hallucination", "badge-confident-hall"


def _trend_chart(history: list) -> go.Figure:
    labels = [f"#{i + 1}" for i in range(len(history))]
    risks = [item["final_risk"] for item in history]
    stabilities = [item["stability"] for item in history]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=labels,
            y=risks,
            mode="lines+markers",
            name="Final Risk",
            line=dict(color="#ef4444", width=2),
            marker=dict(size=8),
            hovertemplate="Run %{x}<br>Risk: %{y:.4f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=labels,
            y=stabilities,
            mode="lines+markers",
            name="Stability",
            line=dict(color="#60a5fa", width=2, dash="dot"),
            marker=dict(size=6),
            hovertemplate="Run %{x}<br>Stability: %{y:.4f}<extra></extra>",
        )
    )
    fig.add_hline(
        y=0.5,
        line_dash="dash",
        line_color="#f59e0b",
        annotation_text="Risk = 0.5",
        annotation_font_color="#f59e0b",
    )
    fig.update_layout(
        title=dict(text="Risk Score Trend Across Runs", font=dict(color="#111827", size=14)),
        xaxis=dict(title="Run"),
        yaxis=dict(title="Score", range=[0, 1.05]),
        legend=dict(font=dict(color="#475569")),
        height=320,
        **LIGHT_LAYOUT,
    )
    return fig


def _radar_comparison(a: dict, b: dict, label_a: str, label_b: str) -> go.Figure:
    categories = ["Consistency", "Stability", "Grounding", "Ext. Similarity", "Confidence"]

    def values(item):
        norm = 1.0 / (1.0 + math.exp(-item["eigen_score"] / max(1.0, abs(item["eigen_score"]) + 1e-9)))
        return [
            1 - norm,
            item["stability"],
            item["grounding"],
            item["ext_sim"],
            1 - item["final_risk"],
        ]

    values_a = values(a)
    values_b = values(b)
    fig = go.Figure()
    fig.add_trace(
        go.Scatterpolar(
            r=values_a + [values_a[0]],
            theta=categories + [categories[0]],
            fill="toself",
            fillcolor="rgba(2,132,199,0.14)",
            line=dict(color="#0284c7", width=2),
            name=label_a,
        )
    )
    fig.add_trace(
        go.Scatterpolar(
            r=values_b + [values_b[0]],
            theta=categories + [categories[0]],
            fill="toself",
            fillcolor="rgba(16,185,129,0.12)",
            line=dict(color="#10b981", width=2),
            name=label_b,
        )
    )
    fig.update_layout(
        polar=dict(
            bgcolor="rgba(0,0,0,0.03)",
            radialaxis=dict(visible=True, range=[0, 1], tickfont=dict(color="#475569")),
            angularaxis=dict(tickfont=dict(color="#111827")),
        ),
        height=400,
        showlegend=True,
        title=dict(text="Metric Comparison", font=dict(color="#111827", size=14)),
        legend=dict(font=dict(color="#475569")),
        **{k: v for k, v in LIGHT_LAYOUT.items() if k != "plot_bgcolor"},
    )
    return fig


def render():
    st.markdown(
        """
    <div class='hero'>
        <h1>Analysis History</h1>
        <p>Past runs in this session, with trends and side-by-side comparison.</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    history = st.session_state.get("analysis_history", [])

    if not history:
        st.markdown(
            """
        <div class='card' style='text-align:center; padding:2.5rem;'>
            <h3 style='color:#0369a1;'>No History Yet</h3>
            <p style='color:rgba(15,23,42,0.55);'>
                Run at least one analysis on the <strong>Analyzer</strong> page first.
            </p>
        </div>""",
            unsafe_allow_html=True,
        )
        return

    st.subheader(f"Session Summary - {len(history)} run(s)")

    rows = []
    for i, item in enumerate(history):
        label, _ = _classify(item["final_risk"], item["eigen_score"], item["ext_sim"])
        rows.append(
            {
                "Run": f"#{i + 1}",
                "Prompt": item["prompt"][:60] + ("..." if len(item["prompt"]) > 60 else ""),
                "Classification": label,
                "Final Risk": round(item["final_risk"], 4),
                "EigenScore": round(item["eigen_score"], 4),
                "Stability": round(item["stability"], 4),
                "Grounding": round(item["grounding"], 4),
                "Ext. Sim": round(item["ext_sim"], 4),
                "Confidence %": round((1 - item["final_risk"]) * 100, 1),
                "GT Source": item.get("gt_source", "-"),
                "Timestamp": item.get("timestamp", "-"),
            }
        )

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True)

    if len(history) > 1:
        st.plotly_chart(_trend_chart(history), use_container_width=True)
    else:
        st.info("Run at least 2 analyses to see the trend chart.")

    st.markdown("<hr style='border-color:rgba(15,23,42,0.1);'/>", unsafe_allow_html=True)
    st.subheader("Side-by-Side Comparison")

    run_options = [f"#{i + 1} - {item['prompt'][:50]}..." for i, item in enumerate(history)]

    if len(history) < 2:
        st.info("Run at least 2 analyses to enable comparison.")
    else:
        c1, c2 = st.columns(2)
        with c1:
            idx_a = st.selectbox("Run A", range(len(history)), format_func=lambda i: run_options[i], key="cmp_a")
        with c2:
            idx_b = st.selectbox(
                "Run B",
                range(len(history)),
                format_func=lambda i: run_options[i],
                index=min(1, len(history) - 1),
                key="cmp_b",
            )

        run_a = history[idx_a]
        run_b = history[idx_b]

        metrics_def = [
            ("Final Risk", "final_risk", False),
            ("EigenScore", "eigen_score", False),
            ("Stability", "stability", True),
            ("Grounding", "grounding", True),
            ("Ext. Sim", "ext_sim", True),
        ]

        cols = st.columns(len(metrics_def))
        for col, (name, key, high_good) in zip(cols, metrics_def):
            value_a = run_a[key]
            value_b = run_b[key]
            delta = value_b - value_a
            better = (delta > 0) == high_good
            arrow = "Improved" if abs(delta) <= 0.001 else ("Up" if better else "Down")
            col.markdown(
                f"""
            <div class='card' style='text-align:center;'>
                <p style='color:rgba(0,0,0,0.45);font-size:0.75rem;margin:0 0 2px;font-weight:700;'>{name}</p>
                <p style='color:#111827;font-size:1rem;font-weight:700;margin:0;'>
                    {value_a:.4f} -> {value_b:.4f}
                </p>
                <p style='font-size:0.85rem;margin:2px 0 0;'>{arrow} {delta:+.4f}</p>
            </div>""",
                unsafe_allow_html=True,
            )

        st.plotly_chart(
            _radar_comparison(run_a, run_b, f"Run #{idx_a + 1}", f"Run #{idx_b + 1}"),
            use_container_width=True,
        )

    st.markdown("<hr style='border-color:rgba(15,23,42,0.1);'/>", unsafe_allow_html=True)
    st.subheader("Export")
    c1, c2, c3 = st.columns(3)

    with c1:
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download CSV",
            csv,
            file_name="halluciScan_history.csv",
            mime="text/csv",
            use_container_width=True,
            key="dl_csv",
        )

    with c2:
        json_data = json.dumps(history, indent=2, default=str).encode("utf-8")
        st.download_button(
            "Download JSON",
            json_data,
            file_name="halluciScan_history.json",
            mime="application/json",
            use_container_width=True,
            key="dl_json",
        )

    with c3:
        if st.button("Clear History", use_container_width=True, key="clear_hist"):
            st.session_state["analysis_history"] = []
            st.rerun()
