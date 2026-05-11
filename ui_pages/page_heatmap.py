"""
Page 6 - Hallucination Heatmap.
Visualizes which response, which layer, and which metric had the
highest hallucination contribution.
"""

import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st


# ---------------------------------------------------------------------------
# Shared layout helpers
# ---------------------------------------------------------------------------

LIGHT_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0.02)",
    font=dict(color="#4b5563"),
    margin=dict(t=60, b=50, l=70, r=20),
)

_RED_GREEN = [
    [0.00, "#10b981"],   # green  → low risk / high metric
    [0.35, "#34d399"],
    [0.50, "#fbbf24"],   # amber  → medium
    [0.65, "#f97316"],
    [1.00, "#ef4444"],   # red    → high risk / low metric
]

_GREEN_RED = [
    [0.00, "#ef4444"],
    [0.35, "#f97316"],
    [0.50, "#fbbf24"],
    [0.65, "#34d399"],
    [1.00, "#10b981"],
]


def _label(value: float, high_good: bool = True) -> str:
    """Convert a 0-1 float to a Low / Medium / High text label."""
    if high_good:
        if value >= 0.65:
            return "High"
        if value >= 0.35:
            return "Medium"
        return "Low"
    else:  # lower = better (risk)
        if value <= 0.35:
            return "Low"
        if value <= 0.65:
            return "Medium"
        return "High"


def _pill_html(text: str) -> str:
    colors = {
        "High":   ("#10b981", "#d1fae5"),
        "Medium": ("#d97706", "#fef3c7"),
        "Low":    ("#dc2626", "#fee2e2"),
    }
    fg, bg = colors.get(text, ("#6b7280", "#f3f4f6"))
    return (
        f"<span style='background:{bg}; color:{fg}; border:1px solid {fg}55; "
        f"padding:2px 10px; border-radius:30px; font-size:0.82rem; font-weight:700;'>"
        f"{text}</span>"
    )


# ---------------------------------------------------------------------------
# Chart builders
# ---------------------------------------------------------------------------

def _response_metric_heatmap(
    responses: list[str],
    ext_sims: list[float],
    global_stability: float,
    global_grounding: float,
) -> go.Figure:
    """
    Heatmap: rows = responses, cols = [Stability, Grounding, Hallucination Risk].
    Stability and Grounding are global (same for all responses); Risk is per-response.
    """
    n = len(responses)
    row_labels = [f"R{i+1}" for i in range(n)]

    stability_col   = [global_stability]   * n          # global
    grounding_col   = [global_grounding]   * n          # global
    risk_col        = [1.0 - s for s in ext_sims]      # per-response
    ext_match_col   = list(ext_sims)                    # per-response

    # z matrix: rows = responses, cols = metrics
    # We want red = high risk → invert stability/grounding/ext_match so they scale the same way
    col_names = ["Stability", "Grounding", "Ext. Match", "Halluc. Risk"]
    z = np.array([
        stability_col,
        grounding_col,
        ext_match_col,
        risk_col,
    ], dtype=float).T  # shape (n_responses, 4)

    # Build text annotations
    text = []
    for i in range(n):
        row_text = [
            f"{stability_col[i]:.3f}",
            f"{grounding_col[i]:.3f}",
            f"{ext_match_col[i]:.3f}",
            f"{risk_col[i]:.3f}",
        ]
        text.append(row_text)

    # For colorscale: 0=green(good), 1=red(bad).
    # Stability/Grounding/Ext.Match: high value = green (good) → use GREEN_RED for those
    # Risk: high value = red (bad) → use RED_GREEN
    # Plotly heatmap uses a single colorscale, so we'll flip the risk column manually
    # by storing risk as-is and using RED_GREEN colorscale

    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=col_names,
            y=row_labels,
            text=text,
            texttemplate="%{text}",
            textfont=dict(size=13, color="white"),
            colorscale=_GREEN_RED,
            zmin=0,
            zmax=1,
            colorbar=dict(
                title=dict(text="Score", font=dict(color="#4b5563")),
                tickfont=dict(color="#4b5563"),
                thickness=14,
                len=0.8,
            ),
            hovertemplate=(
                "Response: %{y}<br>Metric: %{x}<br>Value: %{text}<extra></extra>"
            ),
        )
    )

    # Overlay a second invisible heatmap for the Risk column with flipped colorscale
    risk_z = np.full((n, 4), np.nan)
    risk_z[:, 3] = risk_col
    fig.add_trace(
        go.Heatmap(
            z=risk_z,
            x=col_names,
            y=row_labels,
            text=[[None, None, None, f"{risk_col[i]:.3f}"] for i in range(n)],
            texttemplate="%{text}",
            textfont=dict(size=13, color="white"),
            colorscale=_RED_GREEN,
            zmin=0,
            zmax=1,
            showscale=False,
            hoverinfo="skip",
        )
    )

    fig.update_layout(
        title=dict(
            text="🔥 Response × Metric Hallucination Heatmap",
            font=dict(color="#111827", size=15),
        ),
        xaxis=dict(title="Metric", side="top", tickfont=dict(color="#111827", size=12)),
        yaxis=dict(
            title="Response",
            tickfont=dict(color="#111827", size=12),
            autorange="reversed",
        ),
        height=max(300, 80 + 55 * n),
        **LIGHT_LAYOUT,
    )
    return fig


def _layer_heatmap(layer_sims: list[float]) -> go.Figure:
    """
    Heatmap: rows = layer transitions, single col = Stability similarity.
    Red = low stability = high hallucination risk in that layer.
    """
    n = len(layer_sims)
    row_labels = [f"L{i}→L{i+1}" for i in range(n)]
    z = np.array([[s] for s in layer_sims])  # shape (n, 1)
    text = [[f"{s:.3f}"] for s in layer_sims]

    risk_level = ["High" if s < 0.35 else ("Medium" if s < 0.65 else "Low") for s in layer_sims]
    hover = [
        f"Layer: {row_labels[i]}<br>Stability: {layer_sims[i]:.3f}<br>Risk: {risk_level[i]}"
        for i in range(n)
    ]

    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=["Layer Stability"],
            y=row_labels,
            text=text,
            texttemplate="%{text}",
            textfont=dict(size=12, color="white"),
            colorscale=_GREEN_RED,
            zmin=0,
            zmax=1,
            customdata=[[r] for r in risk_level],
            hovertemplate=(
                "%{y}<br>Stability: %{text}<br>Risk: %{customdata[0]}<extra></extra>"
            ),
            colorbar=dict(
                title=dict(text="Stability", font=dict(color="#4b5563")),
                tickfont=dict(color="#4b5563"),
                thickness=14,
                len=0.8,
            ),
        )
    )
    fig.update_layout(
        title=dict(
            text="🧠 Layer-wise Stability Heatmap",
            font=dict(color="#111827", size=15),
        ),
        xaxis=dict(tickfont=dict(color="#111827")),
        yaxis=dict(
            title="Layer Transition",
            tickfont=dict(color="#111827", size=11),
            autorange="reversed",
        ),
        height=max(300, 80 + 38 * n),
        **LIGHT_LAYOUT,
    )
    return fig


def _metric_contribution_bar(
    global_stability: float,
    global_grounding: float,
    ext_sim: float,
    eigen_score: float,
) -> go.Figure:
    """Horizontal bar showing which metric contributed most to hallucination risk."""
    norm_eigen = 1.0 / (1.0 + math.exp(-eigen_score / max(1.0, abs(eigen_score) + 1e-9)))
    inconsistency = norm_eigen  # high → inconsistent

    metrics = ["EigenScore\n(Inconsistency)", "Instability\n(1-Stability)", "Low Grounding\n(1-Grounding)", "Low Ext. Match\n(1-Ext.Sim)"]
    values  = [inconsistency, 1 - global_stability, 1 - global_grounding, 1 - ext_sim]
    colors  = ["#7c3aed", "#ef4444", "#f97316", "#eab308"]

    fig = go.Figure(
        go.Bar(
            x=values,
            y=metrics,
            orientation="h",
            marker_color=colors,
            text=[f"{v:.3f}" for v in values],
            textposition="outside",
            textfont=dict(color="#111827", size=12),
            hovertemplate="%{y}: %{x:.4f}<extra></extra>",
        )
    )
    fig.update_layout(
        title=dict(
            text="📊 Metric Hallucination Contribution (higher = worse)",
            font=dict(color="#111827", size=15),
        ),
        xaxis=dict(title="Contribution to Hallucination Risk", range=[0, 1.15]),
        yaxis=dict(tickfont=dict(color="#111827", size=12)),
        height=300,
        **LIGHT_LAYOUT,
    )
    return fig


# ---------------------------------------------------------------------------
# Summary table builder
# ---------------------------------------------------------------------------

def _summary_table_html(
    responses: list[str],
    ext_sims: list[float],
    global_stability: float,
    global_grounding: float,
) -> str:
    rows_html = ""
    for i, sim in enumerate(ext_sims):
        risk = 1.0 - sim
        stab_label  = _label(global_stability, high_good=True)
        grnd_label  = _label(global_grounding, high_good=True)
        risk_label  = _label(risk, high_good=False)

        # Highlight the dominant risk factor
        highlight = max(
            [("Stability", 1 - global_stability), ("Grounding", 1 - global_grounding), ("Risk", risk)],
            key=lambda x: x[1]
        )[0]

        def cell(label, metric_name):
            is_worst = metric_name == highlight and i == ext_sims.index(max(ext_sims, key=lambda s: 1 - s))
            bg = "rgba(239,68,68,0.12)" if is_worst else "transparent"
            return (
                f"<td style='padding:10px 14px; border-bottom:1px solid rgba(15,23,42,0.07); "
                f"background:{bg}; text-align:center;'>{_pill_html(label)}</td>"
            )

        rows_html += (
            f"<tr>"
            f"<td style='padding:10px 14px; border-bottom:1px solid rgba(15,23,42,0.07); "
            f"font-weight:700; color:#0369a1;'>R{i+1}</td>"
            f"{cell(stab_label, 'Stability')}"
            f"{cell(grnd_label, 'Grounding')}"
            f"<td style='padding:10px 14px; border-bottom:1px solid rgba(15,23,42,0.07); "
            f"text-align:center;'>"
            f"<span style='font-size:1.05rem; font-weight:700; "
            f"color:{'#ef4444' if risk>0.65 else ('#f97316' if risk>0.35 else '#10b981')};'>"
            f"{risk:.2f}</span></td>"
            f"</tr>"
        )

    return f"""
    <div class='card' style='overflow-x:auto;'>
        <p style='font-weight:700; color:#111827; font-size:1rem; margin:0 0 12px;'>
            🗂️ Per-Response Hallucination Summary
        </p>
        <table style='width:100%; border-collapse:collapse; font-size:0.9rem; color:#374151;'>
            <thead>
                <tr style='background:rgba(2,132,199,0.08);'>
                    <th style='padding:10px 14px; text-align:left; color:#0369a1;'>Response</th>
                    <th style='padding:10px 14px; text-align:center; color:#0369a1;'>Stability</th>
                    <th style='padding:10px 14px; text-align:center; color:#0369a1;'>Grounding</th>
                    <th style='padding:10px 14px; text-align:center; color:#0369a1;'>Risk Score ↑</th>
                </tr>
            </thead>
            <tbody>
                {rows_html}
            </tbody>
        </table>
    </div>
    """


# ---------------------------------------------------------------------------
# Top-contributor badge
# ---------------------------------------------------------------------------

def _worst_response_card(responses, ext_sims, global_stability, global_grounding) -> str:
    risks = [1 - s for s in ext_sims]
    worst_idx = int(np.argmax(risks))
    worst_risk = risks[worst_idx]

    # Dominant metric
    candidates = {
        "Stability":  1 - global_stability,
        "Grounding":  1 - global_grounding,
        "Ext. Match": 1 - ext_sims[worst_idx],
    }
    dominant = max(candidates, key=lambda k: candidates[k])

    risk_color = "#ef4444" if worst_risk > 0.65 else ("#f97316" if worst_risk > 0.35 else "#10b981")

    return f"""
    <div class='card' style='border-left:4px solid {risk_color}; padding:1.2rem 1.4rem;'>
        <p style='color:rgba(0,0,0,0.45); font-size:0.78rem; margin:0 0 6px; font-weight:700;
                  letter-spacing:0.06em;'>HIGHEST HALLUCINATION CONTRIBUTION</p>
        <div style='display:flex; align-items:baseline; gap:16px; flex-wrap:wrap;'>
            <span style='font-size:1.6rem; font-weight:700; color:{risk_color};'>
                R{worst_idx+1}
            </span>
            <span style='color:#6b7280; font-size:0.9rem;'>
                Risk Score: <strong style='color:{risk_color};'>{worst_risk:.3f}</strong>
            </span>
            <span style='color:#6b7280; font-size:0.9rem;'>
                Dominant Metric: <strong style='color:#7c3aed;'>{dominant}</strong>
            </span>
        </div>
        <p style='color:rgba(0,0,0,0.55); font-size:0.82rem; margin:8px 0 0;'>
            "{responses[worst_idx][:120]}{'...' if len(responses[worst_idx]) > 120 else ''}"
        </p>
    </div>
    """


# ---------------------------------------------------------------------------
# Worst layer card
# ---------------------------------------------------------------------------

def _worst_layer_card(layer_sims: list[float]) -> str:
    worst_idx = int(np.argmin(layer_sims))
    worst_val = layer_sims[worst_idx]
    risk_color = "#ef4444" if worst_val < 0.35 else ("#f97316" if worst_val < 0.65 else "#10b981")
    return f"""
    <div class='card' style='border-left:4px solid {risk_color}; padding:1.2rem 1.4rem;'>
        <p style='color:rgba(0,0,0,0.45); font-size:0.78rem; margin:0 0 6px; font-weight:700;
                  letter-spacing:0.06em;'>MOST UNSTABLE LAYER TRANSITION</p>
        <div style='display:flex; align-items:baseline; gap:16px; flex-wrap:wrap;'>
            <span style='font-size:1.6rem; font-weight:700; color:{risk_color};'>
                L{worst_idx}→L{worst_idx+1}
            </span>
            <span style='color:#6b7280; font-size:0.9rem;'>
                Stability: <strong style='color:{risk_color};'>{worst_val:.3f}</strong>
            </span>
        </div>
        <p style='color:rgba(0,0,0,0.55); font-size:0.82rem; margin:8px 0 0;'>
            This layer had the lowest hidden-state similarity between consecutive layers,
            indicating the highest representational drift.
        </p>
    </div>
    """


# ---------------------------------------------------------------------------
# Main render
# ---------------------------------------------------------------------------

def render():
    st.markdown(
        """
    <div class='hero'>
        <h1>Hallucination Heatmap</h1>
        <p>Which response · which layer · which metric had the highest hallucination contribution.</p>
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
                Run an analysis on the <strong>Analyzer</strong> page first, then return here.
            </p>
        </div>""",
            unsafe_allow_html=True,
        )
        return

    # ── Extract data ──────────────────────────────────────────────────────────
    responses       = results["responses"]
    ext_sims        = results["external"].get("similarities", [])
    global_stability= results["stability"]["stability_score"]
    global_grounding= results["grounding"]["grounding_score"]
    ext_sim         = results["external"]["external_consistency"]
    eigen_score     = results["eigen"]["eigen_score"]
    layer_sims      = results["stability"].get("layer_similarities", [])

    # If no per-response similarities available, fall back to uniform
    if not ext_sims:
        ext_sims = [ext_sim] * len(responses)

    # ── Top-contributor cards ─────────────────────────────────────────────────
    col_resp, col_layer = st.columns(2)
    with col_resp:
        st.markdown(
            _worst_response_card(responses, ext_sims, global_stability, global_grounding),
            unsafe_allow_html=True,
        )
    with col_layer:
        if layer_sims:
            st.markdown(_worst_layer_card(layer_sims), unsafe_allow_html=True)
        else:
            st.info("Layer-level data not available for this analysis.")

    # ── Summary table ─────────────────────────────────────────────────────────
    st.markdown(
        _summary_table_html(responses, ext_sims, global_stability, global_grounding),
        unsafe_allow_html=True,
    )

    # ── Metric contribution bar ───────────────────────────────────────────────
    st.plotly_chart(
        _metric_contribution_bar(global_stability, global_grounding, ext_sim, eigen_score),
        use_container_width=True,
    )

    # ── Response × Metric heatmap ─────────────────────────────────────────────
    st.plotly_chart(
        _response_metric_heatmap(responses, ext_sims, global_stability, global_grounding),
        use_container_width=True,
    )

    # ── Layer heatmap ─────────────────────────────────────────────────────────
    if layer_sims:
        st.plotly_chart(_layer_heatmap(layer_sims), use_container_width=True)
        st.caption(
            "Green = stable (low hallucination risk) · Red = unstable (high representational drift)"
        )
    else:
        st.info("Layer-similarity data not available — run on a model that exposes hidden-state cache.")

    # ── How to read ───────────────────────────────────────────────────────────
    with st.expander("📖 How to read this page"):
        st.markdown(
            """
| Term | Meaning |
|---|---|
| **Stability** | How similar adjacent transformer layers are — low = representations drift = more hallucination |
| **Grounding** | How much the model attends back to question tokens while generating — low = drifting off-topic |
| **Ext. Match** | Cosine similarity of each response to the ground-truth answer — low = factually divergent |
| **Halluc. Risk** | `1 – Ext. Match` per response — the primary per-response hallucination signal |
| **EigenScore (Inconsistency)** | Normalized logistic of the raw EigenScore — high = responses were inconsistent |
| **L{i}→L{i+1}** | Hidden-state similarity between layer i and layer i+1; drops signal instability in that transition |

**Color guide:** 🟢 Green = good (low risk) · 🟡 Amber = uncertain · 🔴 Red = high hallucination contribution
"""
        )

    with st.expander("🔢 Raw numbers"):
        df_resp = pd.DataFrame(
            {
                "Response":    [f"R{i+1}" for i in range(len(ext_sims))],
                "Ext. Match":  [round(s, 4) for s in ext_sims],
                "Halluc. Risk":[round(1 - s, 4) for s in ext_sims],
                "Stability":   [round(global_stability, 4)] * len(ext_sims),
                "Grounding":   [round(global_grounding, 4)] * len(ext_sims),
            }
        )
        st.dataframe(df_resp, use_container_width=True)

        if layer_sims:
            df_layer = pd.DataFrame(
                {
                    "Layer Transition": [f"L{i}→L{i+1}" for i in range(len(layer_sims))],
                    "Stability":        [round(s, 4) for s in layer_sims],
                    "Risk (1-Stab)":    [round(1 - s, 4) for s in layer_sims],
                }
            )
            st.dataframe(df_layer, use_container_width=True)
