"""
Model Risk Dashboard — Streamlit App (Phase 4)

Run: streamlit run dashboard/app.py

This dashboard reads the JSON validation reports produced in Phase 2
and the SR 11-7 HTML report from Phase 3, and presents everything
in an interactive, tabbed interface.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import joblib
import streamlit as st
import plotly.graph_objects as go

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    BASELINE_METRICS, PERF_REPORT_PATH, DRIFT_REPORT_PATH,
    ADV_REPORT_PATH, SHAP_REPORT_PATH, REPORT_OUTPUT_PATH,
    TRAIN_PATH, TEST_PATH, OOT_PATH,
    MODEL_PATH, FEATURE_NAMES_PATH,
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Model Risk Dashboard",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Colour palette ────────────────────────────────────────────────────────────
RED    = "#c0392b"
YELLOW = "#e67e22"
GREEN  = "#27ae60"
NAVY   = "#0a1628"

FLAG_COLORS = {"RED": RED, "YELLOW": YELLOW, "GREEN": GREEN}

# ── Load data (cached so it only runs once per session) ───────────────────────
@st.cache_data
def load_all():
    def r(p):
        with open(p) as f:
            return json.load(f)
    return {
        "baseline":    r(BASELINE_METRICS),
        "performance": r(PERF_REPORT_PATH),
        "drift":       r(DRIFT_REPORT_PATH),
        "adversarial": r(ADV_REPORT_PATH),
        "shap":        r(SHAP_REPORT_PATH),
    }

@st.cache_data
def load_splits():
    train = pd.read_csv(TRAIN_PATH)
    test  = pd.read_csv(TEST_PATH)
    oot   = pd.read_csv(OOT_PATH)
    with open(FEATURE_NAMES_PATH) as f:
        features = json.load(f)
    return train, test, oot, features

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

data     = load_all()
train_df, test_df, oot_df, features = load_splits()
model    = load_model()

# Pre-compute score arrays once
train_scores = model.predict_proba(train_df[features])[:, 1]
test_scores  = model.predict_proba(test_df[features])[:, 1]
oot_scores   = model.predict_proba(oot_df[features])[:, 1]


# ── Recession waterfall: real per-factor contributions ────────────────────────
@st.cache_data
def compute_recession_contributions():
    """Apply each stress factor incrementally and measure the actual risk delta."""
    X_s  = oot_df[features].astype(float).copy()
    base = float(np.mean(model.predict_proba(X_s)[:, 1]))
    prev = base
    steps = []

    stress_sequence = []
    if "credit_amount" in features:
        stress_sequence.append(("credit_amount ×1.5",
                                 lambda df: df.assign(credit_amount=df["credit_amount"] * 1.5)))
    if "savings_status" in features:
        stress_sequence.append(("savings → worst",
                                 lambda df: df.assign(savings_status=float(df["savings_status"].min()))))
    if "employment" in features:
        stress_sequence.append(("employment → worst",
                                 lambda df: df.assign(employment=float(df["employment"].min()))))
    if "duration" in features:
        stress_sequence.append(("duration ×1.25",
                                 lambda df: df.assign(duration=df["duration"] * 1.25)))

    for label, fn in stress_sequence:
        X_s      = fn(X_s)
        new_risk = float(np.mean(model.predict_proba(X_s)[:, 1]))
        steps.append({"label": label, "contribution": round(new_risk - prev, 4)})
        prev = new_risk

    return base, round(prev, 4), steps

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"<h2 style='color:{NAVY};margin-bottom:4px;'>🏦 Model Risk</h2>", unsafe_allow_html=True)
    st.markdown(f"<p style='color:#666;font-size:13px;margin-top:0;'>SR 11-7 Validation Dashboard</p>", unsafe_allow_html=True)
    st.divider()

    # Compute rating
    perf = data["performance"]; drift = data["drift"]; adv = data["adversarial"]; shap = data["shap"]
    n_red = sum([
        perf["score_distribution"]["psi_oot_flag"] == "RED",
        perf["calibration"]["calibration_flag"] == "RED",
        data["performance"]["rolling_window"]["auc_variance"] > 0.010,
        adv["boundary_probing"]["flag"] == "RED",
    ])
    n_yellow = sum([
        drift["feature_drift_oot"]["n_yellow"] > 0,
    ])
    rating = "HIGH" if n_red >= 3 else ("MEDIUM" if n_red >= 1 or n_yellow >= 3 else "LOW")
    rating_color = {"HIGH": RED, "MEDIUM": YELLOW, "LOW": GREEN}[rating]

    st.markdown(f"""
    <div style='text-align:center;padding:14px;background:{rating_color}20;border:2px solid {rating_color};border-radius:6px;'>
      <div style='font-size:11px;color:#888;text-transform:uppercase;letter-spacing:1px;'>Overall Risk Rating</div>
      <div style='font-size:28px;font-weight:bold;color:{rating_color};'>{rating}</div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    st.markdown("**Model**")
    st.caption("XGBoost Credit Scoring v1.0")
    st.markdown("**Dataset**")
    st.caption("German Credit (UCI / OpenML)")
    st.markdown("**Findings**")
    cols = st.columns(3)
    cols[0].metric("🔴 RED",    n_red)
    cols[1].metric("🟡 YELLOW", n_yellow)
    cols[2].metric("🟢 GREEN",  10 - n_red - n_yellow)

    st.divider()
    st.caption("Phases: Data → Train → Validate → Report → Dashboard")

# ── Main Tabs ─────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Overview",
    "📉 Drift Monitor",
    "⚡ Stress Testing",
    "🔍 SHAP Explorer",
    "📄 Report",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown(f"<h2 style='color:{NAVY};'>Model Performance Overview</h2>", unsafe_allow_html=True)

    # ── Headline metrics ──
    b = data["baseline"]
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("OOT AUC",   f"{b['oot']['auc']:.4f}",   delta=f"{b['oot']['auc'] - b['train']['auc']:.4f} vs Train", delta_color="normal")
    c2.metric("OOT KS",    f"{b['oot']['ks']:.4f}")
    c3.metric("OOT Gini",  f"{b['oot']['gini']:.4f}")
    c4.metric("Overfit Gap (AUC)", f"{b['overfit_gap_auc']:.4f}", delta="High" if b['overfit_gap_auc'] > 0.10 else "OK", delta_color="inverse")
    c5.metric("PSI (Train→OOT)", f"{b['psi_train_vs_oot']:.4f}", delta="RED" if b['psi_train_vs_oot'] > 0.20 else "OK", delta_color="inverse")

    st.divider()

    # ── Performance across splits ──
    left, right = st.columns([1.4, 1])

    with left:
        st.markdown("#### Discrimination Metrics by Split")
        splits   = ["Train", "Test", "OOT"]
        auc_vals = [b["train"]["auc"], b["test"]["auc"], b["oot"]["auc"]]
        ks_vals  = [b["train"]["ks"],  b["test"]["ks"],  b["oot"]["ks"]]
        gin_vals = [b["train"]["gini"],b["test"]["gini"],b["oot"]["gini"]]

        fig = go.Figure()
        fig.add_trace(go.Bar(name="AUC",  x=splits, y=auc_vals, marker_color=NAVY,   text=[f"{v:.3f}" for v in auc_vals], textposition="outside"))
        fig.add_trace(go.Bar(name="KS",   x=splits, y=ks_vals,  marker_color="#2980b9", text=[f"{v:.3f}" for v in ks_vals],  textposition="outside"))
        fig.add_trace(go.Bar(name="Gini", x=splits, y=gin_vals, marker_color="#5dade2", text=[f"{v:.3f}" for v in gin_vals], textposition="outside"))
        fig.add_hline(y=0.65, line_dash="dot", line_color=RED,   annotation_text="Min AUC threshold", annotation_position="right")
        fig.add_hline(y=0.20, line_dash="dot", line_color=YELLOW, annotation_text="Min KS threshold",  annotation_position="right")
        fig.update_layout(
            barmode="group", height=340, margin=dict(t=60, b=30, l=10, r=170),
            legend=dict(orientation="h", y=1.15, x=0),
            yaxis_range=[0, 1.20],
            plot_bgcolor="white", paper_bgcolor="white",
        )
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.markdown("#### Threshold Sensitivity (OOT)")
        t_data = data["performance"]["threshold_sensitivity"]["thresholds"]
        t_df   = pd.DataFrame(t_data)
        fig2   = go.Figure()
        fig2.add_trace(go.Scatter(x=t_df["threshold"], y=t_df["precision"], name="Precision", line=dict(color=NAVY)))
        fig2.add_trace(go.Scatter(x=t_df["threshold"], y=t_df["recall"],    name="Recall",    line=dict(color=RED)))
        fig2.add_trace(go.Scatter(x=t_df["threshold"], y=t_df["f1"],        name="F1",        line=dict(color=GREEN, dash="dash")))
        opt_t = data["performance"]["threshold_sensitivity"]["optimal_threshold"]["threshold"]
        fig2.add_vline(x=opt_t, line_dash="dot", line_color=YELLOW,
                       annotation_text=f"Optimal={opt_t}",
                       annotation_yref="paper", annotation_y=0.98,
                       annotation_xanchor="left")
        fig2.add_vline(x=0.50,  line_dash="dot", line_color="#aaa",
                       annotation_text="Default=0.5",
                       annotation_yref="paper", annotation_y=0.85,
                       annotation_xanchor="left")
        fig2.update_layout(
            height=340, margin=dict(t=60, b=30, l=10, r=30),
            legend=dict(orientation="h", y=1.15, x=0), yaxis_range=[0, 1.10],
            xaxis_title="Decision Threshold", yaxis_title="Score",
            plot_bgcolor="white", paper_bgcolor="white",
        )
        st.plotly_chart(fig2, use_container_width=True)

    # ── Score distributions ──
    st.markdown("#### Score Distributions: Train vs Test vs OOT")
    st.caption("A shift between the curves indicates the model is scoring the populations differently — PSI quantifies this gap.")

    fig3 = go.Figure()
    for scores, name, color in [(train_scores, "Train", NAVY), (test_scores, "Test", "#2980b9"), (oot_scores, "OOT", RED)]:
        fig3.add_trace(go.Histogram(
            x=scores, name=name, nbinsx=30, histnorm="probability density",
            marker_color=color, opacity=0.55
        ))
    fig3.update_layout(
        barmode="overlay", height=300, margin=dict(t=20, b=40, l=50, r=20),
        xaxis_title="Predicted Default Probability", yaxis_title="Density",
        legend=dict(orientation="h", y=1.05, x=0),
        plot_bgcolor="white", paper_bgcolor="white",
    )
    st.plotly_chart(fig3, use_container_width=True)

    # ── Findings summary table ──
    st.markdown("#### All Findings")
    findings_data = [
        {"ID": "PERF-01", "Category": "Performance",    "Finding": "Score Distribution Shift (PSI)",        "Flag": data["performance"]["score_distribution"]["psi_oot_flag"]},
        {"ID": "PERF-02", "Category": "Performance",    "Finding": "Probability Calibration",               "Flag": data["performance"]["calibration"]["calibration_flag"]},
        {"ID": "PERF-03", "Category": "Performance",    "Finding": "Rolling Window AUC Stability",          "Flag": "RED" if b["overfit_gap_auc"] > 0.10 else "YELLOW"},
        {"ID": "DRIFT-01","Category": "Data Drift",     "Finding": "Feature Distribution Drift (OOT)",      "Flag": "YELLOW" if data["drift"]["feature_drift_oot"]["n_yellow"] > 0 else "GREEN"},
        {"ID": "DRIFT-02","Category": "Data Drift",     "Finding": "Covariate Shift (Train vs OOT)",        "Flag": data["drift"]["covariate_shift"]["flag"]},
        {"ID": "ADV-01",  "Category": "Adversarial",    "Finding": "Decision Boundary Instability",         "Flag": data["adversarial"]["boundary_probing"]["flag"]},
        {"ID": "ADV-02",  "Category": "Adversarial",    "Finding": "Directional Sanity Checks",             "Flag": "GREEN" if data["adversarial"]["directional_sanity"]["n_fail"] == 0 else "RED"},
        {"ID": "ADV-03",  "Category": "Adversarial",    "Finding": "Recession Stress Scenario",             "Flag": data["adversarial"]["recession_stress"]["flag"]},
        {"ID": "EXPL-01", "Category": "Explainability", "Finding": "SHAP Bootstrap Stability",              "Flag": data["shap"]["bootstrap_stability"]["overall_flag"]},
        {"ID": "EXPL-02", "Category": "Explainability", "Finding": "Feature Importance Shift (Train→OOT)",  "Flag": data["shap"]["importance_shift"]["flag"]},
    ]
    f_df = pd.DataFrame(findings_data)
    # Colour-code the Flag column
    def highlight_flag(val):
        colors = {"RED": "background-color:#fdecea;color:#c0392b;font-weight:bold",
                  "YELLOW": "background-color:#fef6e7;color:#d35400;font-weight:bold",
                  "GREEN": "background-color:#eafaf1;color:#1e8449;font-weight:bold"}
        return colors.get(val, "")
    st.dataframe(
        f_df.style.map(highlight_flag, subset=["Flag"]),
        use_container_width=True, hide_index=True, height=390
    )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — DRIFT MONITOR
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown(f"<h2 style='color:{NAVY};'>Data Drift Monitor</h2>", unsafe_allow_html=True)
    st.caption("PSI < 0.10 = GREEN (stable) | 0.10–0.20 = YELLOW (monitor) | > 0.20 = RED (investigate)")

    drift_d = data["drift"]
    feat_drift = drift_d["feature_drift_oot"]["features"]

    # ── PSI bar chart ──
    st.markdown("#### Feature PSI — Training vs OOT")
    psi_df = pd.DataFrame([
        {"Feature": f["feature"], "PSI": f["psi"] if f["psi"] else 0, "Flag": f["drift_flag"]}
        for f in feat_drift
    ]).sort_values("PSI", ascending=True)

    psi_df["Color"] = psi_df["Flag"].map(FLAG_COLORS)
    fig_psi = go.Figure(go.Bar(
        x=psi_df["PSI"], y=psi_df["Feature"],
        orientation="h",
        marker_color=psi_df["Color"].tolist(),
        text=[f"{v:.4f}" for v in psi_df["PSI"]],
        textposition="outside",
    ))
    fig_psi.add_vline(x=0.10, line_dash="dot", line_color=YELLOW,
                      annotation_text="0.10", annotation_yref="paper", annotation_y=1.01,
                      annotation_xanchor="center")
    fig_psi.add_vline(x=0.20, line_dash="dot", line_color=RED,
                      annotation_text="0.20", annotation_yref="paper", annotation_y=1.01,
                      annotation_xanchor="center")
    max_psi = psi_df["PSI"].max()
    fig_psi.update_layout(
        height=520, margin=dict(t=40, b=20, l=180, r=100),
        xaxis=dict(title="PSI Value", range=[0, max_psi * 1.25]),
        plot_bgcolor="white", paper_bgcolor="white",
    )
    st.plotly_chart(fig_psi, use_container_width=True)

    left2, right2 = st.columns(2)

    with left2:
        # ── Drift heatmap (flag matrix) ──
        st.markdown("#### Drift Flag Heatmap")
        st.caption("Each cell = PSI flag for that feature. Colour = severity.")
        flag_map = {"GREEN": 0, "YELLOW": 1, "RED": 2, "N/A": -1}
        feat_names = [f["feature"] for f in feat_drift]
        psi_flags  = [flag_map.get(f["drift_flag"], -1) for f in feat_drift]
        n = len(feat_names)
        ncols_h = 4
        nrows_h = (n + ncols_h - 1) // ncols_h
        z = np.full((nrows_h, ncols_h), -1, dtype=float)
        labels = [[""] * ncols_h for _ in range(nrows_h)]
        for i, (feat, val) in enumerate(zip(feat_names, psi_flags)):
            r, c = divmod(i, ncols_h)
            z[r][c] = val
            labels[r][c] = feat

        # Truncate long names so they fit inside heatmap cells
        short_labels = [[""] * ncols_h for _ in range(nrows_h)]
        for i, feat in enumerate(feat_names):
            r_i, c_i = divmod(i, ncols_h)
            short_labels[r_i][c_i] = feat[:13] + "…" if len(feat) > 13 else feat

        fig_heat = go.Figure(go.Heatmap(
            z=z,
            text=short_labels,
            texttemplate="%{text}",
            textfont=dict(size=9),
            colorscale=[[0, "#e8e8e8"], [0.33, GREEN], [0.66, YELLOW], [1.0, RED]],
            zmin=-1, zmax=2,
            showscale=False,
            xgap=4, ygap=4,
        ))
        fig_heat.update_layout(
            height=300, margin=dict(t=10, b=10, l=10, r=10),
            xaxis=dict(showticklabels=False), yaxis=dict(showticklabels=False),
        )
        st.plotly_chart(fig_heat, use_container_width=True)

    with right2:
        # ── Feature distribution explorer ──
        st.markdown("#### Feature Distribution: Train vs OOT")
        selected_feat = st.selectbox("Select feature to compare:", features, index=features.index("credit_amount"))
        fig_dist = go.Figure()
        fig_dist.add_trace(go.Histogram(
            x=train_df[selected_feat], name="Train", histnorm="probability",
            marker_color=NAVY, opacity=0.6, nbinsx=25
        ))
        fig_dist.add_trace(go.Histogram(
            x=oot_df[selected_feat], name="OOT", histnorm="probability",
            marker_color=RED, opacity=0.6, nbinsx=25
        ))
        psi_val = next((f["psi"] for f in feat_drift if f["feature"] == selected_feat), None)
        flag_val = next((f["drift_flag"] for f in feat_drift if f["feature"] == selected_feat), "N/A")
        fig_dist.update_layout(
            barmode="overlay", height=280, margin=dict(t=50, b=40, l=50, r=20),
            xaxis_title=selected_feat, yaxis_title="Probability",
            legend=dict(orientation="h", y=1.12, x=0),
            plot_bgcolor="white", paper_bgcolor="white",
            title=dict(text=f"PSI = {psi_val:.4f}  [{flag_val}]" if psi_val else "",
                       font=dict(size=13), x=0),
        )
        st.plotly_chart(fig_dist, use_container_width=True)

    # ── Target drift + covariate shift summary ──
    st.markdown("#### Population Summary")
    td = drift_d["target_drift"]
    cs = drift_d["covariate_shift"]
    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("Train Default Rate", f"{td['train_default_rate']:.1%}")
    mc2.metric("OOT Default Rate",   f"{td['oot_default_rate']:.1%}",
               delta=f"{(td['oot_default_rate']-td['train_default_rate']):.1%}")
    mc3.metric("Target Drift Flag",  td["flag_oot"])
    mc4.metric("Covariate Shift AUC", f"{cs['discriminative_auc']:.4f}",
               help="AUC of a classifier trained to distinguish train from OOT. Near 0.5 = no shift.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — STRESS TESTING
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown(f"<h2 style='color:{NAVY};'>Stress Testing & Adversarial Analysis</h2>", unsafe_allow_html=True)

    adv = data["adversarial"]

    top_row = st.columns(3)
    with top_row[0]:
        inst_rate = adv["boundary_probing"]["instability_rate"]
        st.metric(
            "Boundary Instability Rate",
            f"{inst_rate:.1%}",
            delta="RED — all borderline samples flip" if adv["boundary_probing"]["flag"] == "RED" else "Stable",
            delta_color="inverse"
        )
    with top_row[1]:
        rec_inc = adv["recession_stress"]["relative_increase_pct"]
        st.metric("Recession Risk Increase", f"{rec_inc:.1%}", delta=adv["recession_stress"]["flag"], delta_color="inverse")
    with top_row[2]:
        n_fail = adv["directional_sanity"]["n_fail"]
        st.metric("Directional Violations", f"{n_fail}/5", delta="PASS" if n_fail == 0 else "FAIL", delta_color="normal" if n_fail == 0 else "inverse")

    st.divider()
    left3, right3 = st.columns(2)

    with left3:
        # ── Directional sanity chart ──
        st.markdown("#### Directional Sanity Checks")
        st.caption("Spearman correlation between feature value and model risk score. Expected direction shown.")
        dir_checks = adv["directional_sanity"]["checks"]
        dir_df = pd.DataFrame(dir_checks)
        colors_dir = [GREEN if r else RED for r in dir_df["direction_match"]]
        fig_dir = go.Figure(go.Bar(
            x=dir_df["actual_corr"],
            y=dir_df["feature"],
            orientation="h",
            marker_color=colors_dir,
            text=[f"{v:+.4f}" for v in dir_df["actual_corr"]],
            textposition="outside",
        ))
        fig_dir.add_vline(x=0, line_color="#ccc")
        max_abs_corr = dir_df["actual_corr"].abs().max()
        fig_dir.update_layout(
            height=300, margin=dict(t=20, b=50, l=210, r=90),
            xaxis=dict(title="Spearman Correlation (Feature → Risk Score)",
                       range=[-(max_abs_corr * 1.4), max_abs_corr * 1.4]),
            plot_bgcolor="white", paper_bgcolor="white",
        )
        st.plotly_chart(fig_dir, use_container_width=True)

    with right3:
        # ── Recession stress waterfall ──
        st.markdown("#### Recession Stress: Risk Score Change")

        wf_baseline, wf_stressed, wf_steps = compute_recession_contributions()

        wf_x       = ["Baseline"] + [s["label"] for s in wf_steps] + ["Stressed Total"]
        wf_y       = [wf_baseline] + [s["contribution"] for s in wf_steps] + [0]
        wf_measure = ["absolute"] + ["relative"] * len(wf_steps) + ["total"]
        wf_text    = [f"{wf_baseline:.4f}"] + [f"{s['contribution']:+.4f}" for s in wf_steps] + [f"{wf_stressed:.4f}"]

        fig_rec = go.Figure(go.Waterfall(
            x=wf_x, y=wf_y, measure=wf_measure, text=wf_text,
            textposition="outside",
            connector={"line": {"color": "#ccc"}},
            decreasing_marker_color=GREEN,
            increasing_marker_color=RED,
            totals_marker_color=NAVY,
        ))
        fig_rec.update_layout(
            height=360, margin=dict(t=40, b=100, l=10, r=10),
            yaxis=dict(title="Average Risk Score",
                       range=[wf_baseline * 0.95, wf_stressed * 1.08]),
            xaxis=dict(tickangle=-30, tickfont=dict(size=11)),
            plot_bgcolor="white", paper_bgcolor="white",
        )
        st.plotly_chart(fig_rec, use_container_width=True)

    # ── Feature sensitivity under extreme inputs ──
    st.markdown("#### Feature Sensitivity Under Extreme Inputs")
    st.caption("Mean absolute change in predicted score when each feature is set to 0 (simulating missing data).")
    miss_df = pd.DataFrame(adv["missing_data"]["feature_impact"]).sort_values("mean_abs_delta", ascending=False)
    miss_df["Color"] = miss_df["fragile"].map({True: RED, False: "#bdc3c7"})
    fig_miss = go.Figure(go.Bar(
        x=miss_df["mean_abs_delta"],
        y=miss_df["feature"],
        orientation="h",
        marker_color=miss_df["Color"].tolist(),
        text=[f"{v:.4f}" for v in miss_df["mean_abs_delta"]],
        textposition="outside",
    ))
    fig_miss.add_vline(x=0.10, line_dash="dot", line_color=RED,
                       annotation_text="Fragile (0.10)",
                       annotation_yref="paper", annotation_y=1.01,
                       annotation_xanchor="center")
    max_miss = miss_df["mean_abs_delta"].max()
    fig_miss.update_layout(
        height=500, margin=dict(t=40, b=20, l=180, r=100),
        xaxis=dict(title="Mean |Δ Score| when feature = 0", range=[0, max_miss * 1.25]),
        plot_bgcolor="white", paper_bgcolor="white",
    )
    st.plotly_chart(fig_miss, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — SHAP EXPLORER
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown(f"<h2 style='color:{NAVY};'>SHAP Feature Importance & Explainability</h2>", unsafe_allow_html=True)
    st.caption("SHAP (SHapley Additive exPlanations) — how much each feature contributes to each prediction.")

    shap_d = data["shap"]
    top_left, top_right = st.columns(2)

    with top_left:
        # ── Global importance — OOT ──
        st.markdown("#### Global Feature Importance (OOT)")
        imp_data = sorted(shap_d["global_importance_oot"]["importance"], key=lambda x: x["mean_abs_shap"], reverse=True)[:15]
        imp_df   = pd.DataFrame(imp_data)
        fig_imp  = go.Figure(go.Bar(
            x=imp_df["mean_abs_shap"],
            y=imp_df["feature"],
            orientation="h",
            marker_color=NAVY,
            marker_line_color="white",
            text=[f"{v:.4f}" for v in imp_df["mean_abs_shap"]],
            textposition="outside",
        ))
        max_shap_val = imp_df["mean_abs_shap"].max()
        fig_imp.update_layout(
            height=440, margin=dict(t=20, b=20, l=180, r=90),
            xaxis=dict(title="Mean |SHAP Value|", range=[0, max_shap_val * 1.2]),
            plot_bgcolor="white", paper_bgcolor="white",
            yaxis={"categoryorder": "total ascending"},
        )
        st.plotly_chart(fig_imp, use_container_width=True)

    with top_right:
        # ── Train vs OOT importance comparison ──
        st.markdown("#### Importance: Train vs OOT Comparison")
        train_imp = {d["feature"]: d["mean_abs_shap"] for d in shap_d["global_importance_test"]["importance"]}
        oot_imp   = {d["feature"]: d["mean_abs_shap"] for d in shap_d["global_importance_oot"]["importance"]}
        common    = sorted(train_imp.keys(), key=lambda f: oot_imp.get(f, 0), reverse=True)[:12]
        fig_comp  = go.Figure()
        fig_comp.add_trace(go.Bar(name="Train", x=[train_imp[f] for f in common], y=common, orientation="h", marker_color=NAVY, opacity=0.7))
        fig_comp.add_trace(go.Bar(name="OOT",   x=[oot_imp[f]   for f in common], y=common, orientation="h", marker_color=RED,  opacity=0.7))
        fig_comp.update_layout(
            barmode="group", height=440, margin=dict(t=60, b=20, l=180, r=20),
            xaxis_title="Mean |SHAP Value|",
            plot_bgcolor="white", paper_bgcolor="white",
            legend=dict(orientation="h", y=1.12, x=0),
            yaxis={"categoryorder": "total ascending"},
        )
        st.plotly_chart(fig_comp, use_container_width=True)

    # ── Bootstrap stability chart ──
    st.markdown("#### Bootstrap Stability (Rank Distribution across 30 Subsamples)")
    st.caption("Low rank standard deviation = the feature's importance ranking is consistent. A std > 2.0 would be flagged.")
    stab_df = pd.DataFrame(shap_d["bootstrap_stability"]["features"]).sort_values("mean_rank")
    stab_df["Color"] = stab_df["stable"].map({True: GREEN, False: RED})
    fig_stab = go.Figure()
    fig_stab.add_trace(go.Scatter(
        x=stab_df["mean_rank"],
        y=stab_df["feature"],
        error_x=dict(type="data", array=stab_df["rank_std"].tolist(), visible=True, color="#aaa"),
        mode="markers",
        marker=dict(color=stab_df["Color"].tolist(), size=10),
        text=[f"Mean rank={r:.1f}  std={s:.2f}" for r, s in zip(stab_df["mean_rank"], stab_df["rank_std"])],
        hovertemplate="%{y}<br>%{text}<extra></extra>",
    ))
    fig_stab.update_layout(
        height=440, margin=dict(t=20, b=20, l=180, r=20),
        xaxis=dict(title="Mean Rank (1 = most important)", autorange="reversed"),
        plot_bgcolor="white", paper_bgcolor="white",
        yaxis={"categoryorder": "total ascending"},
    )
    st.plotly_chart(fig_stab, use_container_width=True)

    # ── Local explanations ──
    st.markdown("#### Individual Prediction Explanations")
    st.caption("Select a sample to see why the model gave that specific prediction.")

    local_exps = shap_d["local_explanations"]["explanations"]
    exp_labels = [f"Sample {e['sample_index']} — {e['label']} (score={e['predicted_prob']:.4f}, actual={'DEFAULT' if e['actual_target']==1 else 'GOOD'})"
                  for e in local_exps]
    selected_exp_idx = st.selectbox("Choose prediction:", range(len(exp_labels)), format_func=lambda i: exp_labels[i])
    exp = local_exps[selected_exp_idx]

    col_l, col_r = st.columns([2, 1])
    with col_l:
        contribs = exp["top_contributors"]
        c_df = pd.DataFrame(contribs)
        colors_c = [RED if v > 0 else GREEN for v in c_df["shap_value"]]
        fig_local = go.Figure(go.Bar(
            x=c_df["shap_value"],
            y=c_df["feature"],
            orientation="h",
            marker_color=colors_c,
            text=[f"{v:+.4f}" for v in c_df["shap_value"]],
            textposition="outside",
        ))
        fig_local.add_vline(x=0, line_color="#ccc")
        max_abs_shap_local = c_df["shap_value"].abs().max()
        fig_local.update_layout(
            height=300, margin=dict(t=20, b=50, l=210, r=100),
            xaxis=dict(title="SHAP Value (positive = increases default risk)",
                       range=[-(max_abs_shap_local * 1.5), max_abs_shap_local * 1.5]),
            plot_bgcolor="white", paper_bgcolor="white",
        )
        st.plotly_chart(fig_local, use_container_width=True)

    with col_r:
        st.markdown("**Prediction Summary**")
        st.markdown(f"""
        | | |
        |---|---|
        | **Predicted Score** | `{exp['predicted_prob']:.4f}` |
        | **Decision** | {'REJECT' if exp['predicted_prob'] >= 0.5 else 'APPROVE'} |
        | **Actual Outcome** | {'DEFAULT ❌' if exp['actual_target'] == 1 else 'GOOD ✅'} |
        | **Sample Type** | {exp['label']} |
        | **Base Value** | `{exp['base_value']:.4f}` |
        """)
        if exp["predicted_prob"] >= 0.5 and exp["actual_target"] == 0:
            st.warning("False Positive — model incorrectly flagged this applicant as high risk.")
        elif exp["predicted_prob"] < 0.5 and exp["actual_target"] == 1:
            st.error("False Negative — model missed a defaulter.")
        elif exp["predicted_prob"] >= 0.5 and exp["actual_target"] == 1:
            st.success("True Positive — model correctly identified defaulter.")
        else:
            st.success("True Negative — model correctly approved good applicant.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — REPORT
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown(f"<h2 style='color:{NAVY};'>SR 11-7 Validation Report</h2>", unsafe_allow_html=True)

    if os.path.exists(REPORT_OUTPUT_PATH):
        with open(REPORT_OUTPUT_PATH, "rb") as f:
            report_bytes = f.read()

        st.download_button(
            label="⬇ Download Full SR 11-7 Report (HTML)",
            data=report_bytes,
            file_name="sr117_validation_report.html",
            mime="text/html",
            use_container_width=True,
        )

        st.divider()
        st.markdown("#### Report Preview")
        st.caption("Full report rendered below. Download for best viewing experience.")
        report_html = report_bytes.decode("utf-8")
        st.components.v1.html(report_html, height=900, scrolling=True)
    else:
        st.error("Report not found. Run `python run_phase3.py` first to generate it.")
        st.code("python run_phase3.py")
