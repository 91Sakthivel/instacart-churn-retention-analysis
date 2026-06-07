# ============================================================
# streamlit/app.py -- Instacart Revenue Protection Dashboard
# ============================================================
import os
from pathlib import Path

# All data files live in the same directory as app.py.
_HERE = Path(__file__).parent

DATA_RISK_RANKED  = _HERE / "risk_ranked_top5k.csv"
DATA_RFM          = _HERE / "rfm_segments.csv"
DATA_TRAJ         = _HERE / "trajectory_segments.csv"
DATA_ROI          = _HERE / "retention_roi.csv"
DATA_DEPT_RES     = _HERE / "department_residuals.csv"
DATA_CAT_RISK     = _HERE / "category_risk_report.csv"
DATA_METRICS      = _HERE / "model_metrics.csv"
CHART_FEAT_IMP    = _HERE / "feature_importance_classifier.png"

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# ============================================================
# Page config — must be first Streamlit call
# ============================================================
st.set_page_config(
    page_title="Instacart Revenue Protection",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# Global CSS
# ============================================================
st.markdown("""
<style>
  .stApp { background-color: #ffffff; }
  h1 { color: #1a1a2e; font-weight: 700; font-size: 1.7rem; }
  h2, h3 { color: #1a1a2e; font-weight: 600; }
  div[data-testid="metric-container"] {
      background: #f4f8fc;
      border-left: 4px solid #2980b9;
      border-radius: 6px;
      padding: 12px 16px;
  }
  div[data-testid="metric-container"] label {
      color: #5a6a7a;
      font-size: 0.82rem;
      font-weight: 500;
  }
  div[data-testid="metric-container"] div[data-testid="metric-value"] {
      color: #1a1a2e;
      font-size: 1.35rem;
      font-weight: 700;
  }
  .alert-box {
      background: #fff5f5;
      border-left: 4px solid #e74c3c;
      border-radius: 6px;
      padding: 14px 18px;
      margin: 12px 0;
      font-size: 0.95rem;
  }
  .info-box {
      background: #f0f7ff;
      border-left: 4px solid #2980b9;
      border-radius: 6px;
      padding: 14px 18px;
      margin: 12px 0;
      font-size: 0.95rem;
  }
  .check-row {
      padding: 10px 0;
      border-bottom: 1px solid #f0f0f0;
      line-height: 1.6;
  }
  .tag-validated { color: #27ae60; font-weight: 700; }
  .tag-fail { color: #e74c3c; font-weight: 700; }
  .leak-row {
      border-left: 4px solid #e74c3c;
      padding: 10px 16px;
      margin: 8px 0;
      background: #fff9f9;
      border-radius: 4px;
      font-size: 0.92rem;
      line-height: 1.6;
  }
  .window-row {
      padding: 10px 16px;
      margin: 6px 0;
      background: #fafafa;
      border-radius: 4px;
      font-size: 0.93rem;
  }
</style>
""", unsafe_allow_html=True)

ACCENT   = "#2980b9"
C_TRAJ   = {"B": "#d32f2f", "D": "#e65100", "A": "#f57f17", "C": "#388e3c"}
T_LABELS = {
    "B": "B: Sudden Stop",
    "D": "D: High Value Drifting",
    "A": "A: Fading Frequency",
    "C": "C: Low Loyalty",
}

# ============================================================
# Data loading
# ============================================================
def _safe(path, **kw):
    p = Path(path)
    try:
        if p.exists():
            return pd.read_csv(p, **kw)
    except Exception as e:
        st.warning(f"Could not load {p.name}: {e}")
    return pd.DataFrame()

@st.cache_data
def load_data():
    risk     = _safe(DATA_RISK_RANKED)
    rfm      = _safe(DATA_RFM)
    traj     = _safe(DATA_TRAJ)
    roi      = _safe(DATA_ROI)
    dept_res = _safe(DATA_DEPT_RES)
    cat_risk = _safe(DATA_CAT_RISK)
    metrics  = _safe(DATA_METRICS)
    return risk, rfm, traj, roi, dept_res, cat_risk, metrics

risk, rfm, traj, roi, dept_res, cat_risk, metrics = load_data()

@st.cache_data
def build_explorer_df():
    base = risk.copy()
    if len(rfm) > 0 and "segment" in rfm.columns and len(base) > 0:
        seg = rfm[["user_id", "segment"]].rename(columns={"segment": "rfm_segment"})
        base = base.merge(seg, on="user_id", how="left")
    if "rfm_segment" not in base.columns:
        base["rfm_segment"] = "Unknown"
    base["rfm_segment"] = base["rfm_segment"].fillna("Unknown")
    return base

explorer = build_explorer_df()

# ============================================================
# Helpers
# ============================================================
def fc(v):
    try: return f"${float(v):,.0f}"
    except: return "N/A"

def fp(v, decimals=1):
    try: return f"{float(v):.{decimals}f}%"
    except: return "N/A"

def chart_layout(fig, height=None):
    upd = dict(
        plot_bgcolor="white", paper_bgcolor="white",
        font_color="#1a1a2e",
        margin=dict(t=20, b=50, l=50, r=20),
        showlegend=False,
    )
    if height:
        upd["height"] = height
    fig.update_layout(**upd)
    return fig

# ============================================================
# Sidebar
# ============================================================
with st.sidebar:
    st.markdown("## Instacart Revenue Protection")
    st.markdown("---")
    page = st.radio(
        "Navigation",
        options=[
            "Executive Dashboard",
            "Business Findings & Recommendations",
            "Customer Risk Explorer",
            "Intervention & ROI",
            "Category Exposure",
            "Model Performance",
            "Behavioral Monitor",
            "Cohort Retention",
        ],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown("**Dataset**")
    st.caption("206,209 customers")
    st.caption("33.8M order rows")
    st.markdown("---")
    st.markdown("**GitHub**")
    st.markdown("[instacart-churn-retention-analysis](https://github.com/YOUR_USERNAME/instacart-churn-retention-analysis)")

# ============================================================
# PAGE 1 — Executive Dashboard
# ============================================================
if page == "Executive Dashboard":
    st.title("Instacart Revenue Protection — Executive View")

    exec_view = st.selectbox(
        "View mode",
        ["All Sections", "Finance View"],
        key="exec_view",
        help="Finance View: revenue impact and cost-benefit numbers only — no segment charts.",
    )

    # --- Executive KPI Scorecard ---
    st.subheader("Executive KPI Scorecard")
    st.markdown("*Live metrics from the retention analytics pipeline*")

    # Recovery rates per trajectory type (validated in pipeline/05b_trajectory.py)
    _KPI_RR = {"A": 0.35, "B": 0.20, "D": 0.40, "C": 0.05}
    _TOTAL_MODELLED = 175072

    # Compute KPI values from already-loaded data
    # FIX 1: high-risk = rows where P_churn >= 0.5 (not total file length)
    _high_risk_n    = int((risk["P_churn"] >= 0.5).sum()) if (len(risk) > 0 and "P_churn" in risk.columns) else 0
    _high_risk_pct  = _high_risk_n / _TOTAL_MODELLED * 100

    # FIX 2: avg churn prob from P_churn mean, displayed as percentage
    _avg_churn_prob = 0.0
    if len(risk) > 0 and "P_churn" in risk.columns:
        _avg_churn_prob = float(risk["P_churn"].mean())

    _est_saved = 0
    if len(roi) > 0 and "n_customers" in roi.columns and "trajectory_type" in roi.columns:
        for _, _row in roi.iterrows():
            _est_saved += float(_row["n_customers"]) * _KPI_RR.get(_row["trajectory_type"], 0)

    _est_rev = 0.0
    if len(risk) > 0 and "predicted_loss" in risk.columns and "trajectory_type" in risk.columns:
        _r_tmp = risk.copy()
        _r_tmp["_rr"] = _r_tmp["trajectory_type"].map(_KPI_RR).fillna(0)
        _est_rev = float((_r_tmp["predicted_loss"] * _r_tmp["_rr"]).sum())

    # Row 1 — 4 cards
    k1, k2, k3, k4 = st.columns([1.2, 1.2, 1, 1], gap="medium")
    k1.metric("Total Customers Analyzed",  f"{_TOTAL_MODELLED:,}")
    k2.metric("High-Risk Customers",       f"{_high_risk_n:,}")
    k3.metric("High-Risk %",               f"{_high_risk_pct:.1f}%")
    k4.metric("Avg Churn Probability",     f"{_avg_churn_prob * 100:.1f}%")

    # Row 2 — 3 cards
    k5, k6, k7 = st.columns(3)
    k5.metric("Priority Segment",          "Type A — Fading Frequency (44.5% churn)")
    k6.metric("Est. Customers Saved",      f"{int(_est_saved):,}")
    k7.metric("Est. Revenue Protected",    fc(_est_rev))

    if exec_view == "All Sections":
        st.markdown("---")

        # --- Two charts side by side ---
        col_l, col_r = st.columns(2)

        with col_l:
            st.subheader("RFM Segment Distribution")
            if len(rfm) > 0 and "segment" in rfm.columns:
                seg_cnt = rfm["segment"].value_counts().reset_index()
                seg_cnt.columns = ["Segment", "Customers"]
                fig = px.bar(
                    seg_cnt, x="Segment", y="Customers",
                    color_discrete_sequence=[ACCENT], text="Customers",
                )
                fig.update_traces(texttemplate="%{text:,}", textposition="outside")
                chart_layout(fig)
                fig.update_yaxes(title="")
                st.plotly_chart(fig, width=700)
            else:
                st.info("rfm_segments.csv not loaded.")

        with col_r:
            st.subheader("Trajectory Type — Churn Rate")
            if len(traj) > 0 and "trajectory_type" in traj.columns and "churned" in traj.columns:
                tt = (
                    traj.groupby("trajectory_type")
                    .agg(customers=("user_id", "count"), churn_pct=("churned", "mean"))
                    .reset_index()
                )
                tt["churn_pct"] = tt["churn_pct"] * 100
                tt["label"] = tt["trajectory_type"].map(T_LABELS).fillna(tt["trajectory_type"])
                tt["_ord"] = tt["trajectory_type"].map({"B": 0, "D": 1, "A": 2, "C": 3}).fillna(9)
                tt = tt.sort_values("_ord")
                fig2 = px.bar(
                    tt, x="label", y="churn_pct", text="churn_pct",
                    color="trajectory_type",
                    color_discrete_map=C_TRAJ,
                )
                fig2.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
                chart_layout(fig2)
                fig2.update_yaxes(title="Churn Rate (%)")
                st.plotly_chart(fig2, width=700)
            else:
                st.info("trajectory_segments.csv not loaded.")

    st.markdown("---")

    # --- Top 10 risk table ---
    st.subheader("Top 10 Customers by Risk Score")
    if len(risk) > 0:
        disp = risk.head(10)[
            [c for c in ["user_id", "trajectory_type", "P_churn", "predicted_loss", "risk_score"]
             if c in risk.columns]
        ].copy()
        disp = disp.rename(columns={
            "user_id": "User ID", "trajectory_type": "Trajectory",
            "P_churn": "P(Churn)", "predicted_loss": "Predicted Loss",
            "risk_score": "Risk Score",
        })
        if "P(Churn)" in disp.columns:
            disp["P(Churn)"] = disp["P(Churn)"].map(lambda x: f"{x:.3f}")
        if "Predicted Loss" in disp.columns:
            disp["Predicted Loss"] = disp["Predicted Loss"].map(fc)
        if "Risk Score" in disp.columns:
            disp["Risk Score"] = disp["Risk Score"].map(fc)
        st.dataframe(disp, width=700, hide_index=True)
    else:
        st.info("risk_ranked_top5k.csv not loaded.")

    # --- Finance View: Revenue Impact Summary ---
    if exec_view == "Finance View":
        st.markdown("---")
        st.subheader("Revenue Impact by Trajectory Type")
        st.markdown("*Intervention budget, projected recovery, and net return per segment*")
        _FIN_COSTS  = {"A": 11.00, "B": 15.00, "D": 8.50,  "C": 0.50}
        _FIN_LABELS = {"A": "Fading Frequency", "B": "Sudden Stop",
                       "D": "High-Value Drifting", "C": "Low Loyalty"}
        if len(roi) > 0 and "trajectory_type" in roi.columns:
            _fin = roi.copy()
            _fin["Segment"]         = _fin["trajectory_type"].map(_FIN_LABELS).fillna(_fin["trajectory_type"])
            _fin["Customers"]       = _fin["n_customers"].map(lambda x: f"{int(x):,}")
            _fin["Cost / Customer"] = _fin["trajectory_type"].map(
                                          lambda t: f"${_FIN_COSTS.get(t, 0):.2f}")
            _fin["Total Budget"]    = (
                                          _fin["n_customers"]
                                          * _fin["trajectory_type"].map(_FIN_COSTS)
                                      ).map(lambda x: f"${x:,.0f}")
            _fin["Est. Rev Saved"]  = _fin["total_revenue_saved"].map(
                                          lambda x: f"${float(x):,.0f}")
            _fin["Net Benefit"]     = (
                                          _fin["total_revenue_saved"] - _fin["total_offer_cost"]
                                      ).map(lambda x: f"${float(x):,.0f}")
            _fin["Avg ROI"]         = _fin["avg_ROI_pct"].map(lambda x: f"{float(x):,.0f}%")
            _ord_map = {"B": 0, "D": 1, "A": 2, "C": 3}
            _fin["_o"] = _fin["trajectory_type"].map(_ord_map).fillna(9)
            _fin = _fin.sort_values("_o")
            st.dataframe(
                _fin[["Segment", "Customers", "Cost / Customer",
                      "Total Budget", "Est. Rev Saved", "Net Benefit", "Avg ROI"]],
                width=800, hide_index=True,
            )
            _total_budget  = (_fin["n_customers"] * _fin["trajectory_type"].map(_FIN_COSTS)).sum()
            _total_saved   = float(_fin["total_revenue_saved"].sum())
            _total_net     = _total_saved - float(_fin["total_offer_cost"].sum())
            fb1, fb2, fb3 = st.columns(3)
            fb1.metric("Total Campaign Budget",   fc(_total_budget))
            fb2.metric("Total Est. Rev Saved",    fc(_total_saved))
            fb3.metric("Total Net Benefit",       fc(_total_net))
        else:
            st.info("retention_roi.csv not loaded — run pipeline/05b_trajectory.py.")

# ============================================================
# PAGE 2 — Business Findings & Recommendations
# ============================================================
elif page == "Business Findings & Recommendations":
    st.title("Key Business Findings & Recommendations")
    st.markdown("*What the data actually showed and what the retention team should do Monday morning*")

    # ----------------------------------------------------------
    # SECTION 1 — The 3 Core Findings
    # ----------------------------------------------------------
    st.subheader("The 3 Core Findings")

    f1, f2, f3 = st.columns(3)

    with f1:
        st.markdown("""
        <div style="background:#fff5f5;border-left:4px solid #e74c3c;border-radius:6px;
                    padding:20px;height:100%;">
          <div style="font-size:2rem;font-weight:700;color:#e74c3c;margin-bottom:6px;">$980,584</div>
          <div style="font-weight:600;color:#1a1a2e;margin-bottom:10px;">The Waste Problem</div>
          <div style="font-size:0.9rem;color:#444;line-height:1.6;">
            Mistargeting 89,144 low-loyalty customers with an $11 discount wastes nearly $1M annually.
            These customers (Type C) have a 14.9% churn rate and respond only to light-touch push
            notifications costing $0.50 each. Standard retention campaigns treat all at-risk customers
            equally — this analysis shows that is expensive and wrong.
          </div>
        </div>
        """, unsafe_allow_html=True)

    with f2:
        st.markdown("""
        <div style="background:#fff8f0;border-left:4px solid #e65100;border-radius:6px;
                    padding:20px;height:100%;">
          <div style="font-size:2rem;font-weight:700;color:#e65100;margin-bottom:6px;">9,786 customers</div>
          <div style="font-weight:600;color:#1a1a2e;margin-bottom:10px;">Best Customers Quietly Leaving</div>
          <div style="font-size:0.9rem;color:#444;line-height:1.6;">
            High-value customers showing gap drift (Type D) have only 22.3% churn rate — easily
            missed by standard churn models. But they carry the highest annual value. A loyalty
            reward at $8.50 cost recovers 40% of them at 17,891% ROI.
            These are the customers worth fighting for.
          </div>
        </div>
        """, unsafe_allow_html=True)

    with f3:
        st.markdown("""
        <div style="background:#f0f7ff;border-left:4px solid #2980b9;border-radius:6px;
                    padding:20px;height:100%;">
          <div style="font-size:2rem;font-weight:700;color:#2980b9;margin-bottom:6px;">$183,384</div>
          <div style="font-weight:600;color:#1a1a2e;margin-bottom:10px;">Produce Is the Revenue Anchor</div>
          <div style="font-size:0.9rem;color:#444;line-height:1.6;">
            Produce carries 29% of total 60-day revenue exposure — not because produce shoppers
            churn at unusual rates, but because it has the most customers. The residual analysis
            confirmed no operational anomaly. This is pure volume concentration risk.
          </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("---")

    # ----------------------------------------------------------
    # SECTION 2 — Monday Morning Action Plan
    # ----------------------------------------------------------
    st.subheader("Monday Morning Action Plan")

    a1, a2, a3 = st.columns(3)

    with a1:
        st.markdown("""
        <div style="background:#f0fff4;border:1px solid #27ae60;border-top:4px solid #27ae60;
                    border-radius:6px;padding:20px;">
          <div style="font-size:0.75rem;font-weight:700;letter-spacing:0.08em;
                      color:#27ae60;text-transform:uppercase;margin-bottom:8px;">Do This Today</div>
          <div style="font-weight:700;font-size:1.05rem;color:#1a1a2e;margin-bottom:10px;">
            Type D Loyalty Reward Campaign
          </div>
          <div style="font-size:0.85rem;color:#444;line-height:1.7;">
            <b>Trigger:</b> reorder_rate &ge; 0.55 AND gap_trend &gt; 5<br>
            <b>Customers:</b> 9,786<br>
            <b>Cost:</b> $8.50 per customer<br>
            <b>Recovery:</b> 40% &nbsp;|&nbsp; <b>ROI:</b> 17,891%<br><br>
            Send personalized loyalty reward before gap hits 20 days.
          </div>
        </div>
        """, unsafe_allow_html=True)

    with a2:
        st.markdown("""
        <div style="background:#fffbf0;border:1px solid #e65100;border-top:4px solid #e65100;
                    border-radius:6px;padding:20px;">
          <div style="font-size:0.75rem;font-weight:700;letter-spacing:0.08em;
                      color:#e65100;text-transform:uppercase;margin-bottom:8px;">Do This This Week</div>
          <div style="font-weight:700;font-size:1.05rem;color:#1a1a2e;margin-bottom:10px;">
            Type B Sudden Stop Outreach
          </div>
          <div style="font-size:0.85rem;color:#444;line-height:1.7;">
            <b>Trigger:</b> early_avg_gap &lt; 12 AND late_avg_gap_clean &gt; 20<br>
            <b>Customers:</b> 6,010<br>
            <b>Cost:</b> $15.00 per customer<br>
            <b>Recovery:</b> 20% &nbsp;|&nbsp; <b>ROI:</b> 3,820%<br><br>
            Personal phone call within 48 hours of detection.
          </div>
        </div>
        """, unsafe_allow_html=True)

    with a3:
        st.markdown("""
        <div style="background:#fff5f5;border:1px solid #e74c3c;border-top:4px solid #e74c3c;
                    border-radius:6px;padding:20px;">
          <div style="font-size:0.75rem;font-weight:700;letter-spacing:0.08em;
                      color:#e74c3c;text-transform:uppercase;margin-bottom:8px;">Stop Doing This</div>
          <div style="font-weight:700;font-size:1.05rem;color:#1a1a2e;margin-bottom:10px;">
            Type C Blanket Discounting
          </div>
          <div style="font-size:0.85rem;color:#444;line-height:1.7;">
            <b>At-risk customers:</b> 89,144<br>
            <b>Annual waste:</b> $980,584 if sent $11 discount<br>
            <b>Correct offer:</b> $0.50 push notification<br><br>
            Replace discount campaigns with push notifications for this segment only.
          </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("---")

    # ----------------------------------------------------------
    # SECTION 3 — Honest Limitations
    # ----------------------------------------------------------
    st.subheader("What This Model Does Not Do")
    st.markdown("""
    <div style="background:#f8f9fa;border:1px solid #dee2e6;border-radius:6px;padding:20px;
                font-size:0.92rem;color:#444;line-height:1.8;">
      This model predicts churn probability from behavioral signals. It does not account for
      external factors (competitor promotions, pricing changes, app issues). The revenue
      regressor has R&#178;=0.18 — revenue loss per customer is uncertain and should be treated
      as a directional estimate, not a precise forecast. Retrain monthly on fresh order data.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("---")

    # ----------------------------------------------------------
    # SECTION 4 — Key Insights & Recommendations
    # ----------------------------------------------------------
    st.subheader("Key Insights & Recommendations")

    ins_col, rec_col = st.columns(2)

    with ins_col:
        st.markdown("""
        <div style="background:#f0f7ff;border-left:4px solid #2980b9;border-radius:6px;
                    padding:20px;height:100%;">
          <div style="font-weight:700;font-size:1.0rem;color:#2980b9;margin-bottom:12px;">
            &#128202; Key Insights
          </div>
          <ul style="font-size:0.9rem;color:#444;line-height:1.8;padding-left:18px;margin:0;">
            <li>Gap widening is the primary churn signal — customers who widen order gaps by
                5+ days show significantly elevated churn risk</li>
            <li>Basket size does NOT decline before churn — the basket is flat or slightly
                rising into the final order (9.79 → 10.15 items)</li>
            <li>Churn is behavioral, not categorical — residual analysis found no
                department-level anomalies across all 21 product categories</li>
            <li>The top 53% of customers drive 80% of total revenue risk — precise targeting
                outperforms blanket retention campaigns by a wide margin</li>
          </ul>
        </div>
        """, unsafe_allow_html=True)

    with rec_col:
        st.markdown("""
        <div style="background:#f0fff4;border-left:4px solid #27ae60;border-radius:6px;
                    padding:20px;height:100%;">
          <div style="font-weight:700;font-size:1.0rem;color:#27ae60;margin-bottom:12px;">
            &#9989; Recommendations
          </div>
          <ul style="font-size:0.9rem;color:#444;line-height:1.8;padding-left:18px;margin:0;">
            <li>Start with Type D loyalty rewards — 17,891% ROI and 40% recovery on the
                highest-value customers; act before gap exceeds 20 days</li>
            <li>Replace Type C discount spend with $0.50 push notifications immediately —
                saves $980,584 annually with no measurable recovery loss</li>
            <li>Deploy Type B phone calls within 48 hours of sudden-stop detection —
                recoverable only with immediate, personal outreach</li>
            <li>Retrain the model monthly on fresh order data — behavioral patterns shift
                with seasons, promotions, and competitive activity</li>
          </ul>
        </div>
        """, unsafe_allow_html=True)

# ============================================================
# PAGE 3 — Customer Risk Explorer
# ============================================================
elif page == "Customer Risk Explorer":
    st.title("Customer Risk Explorer")
    st.markdown("*Filter and identify at-risk customers for retention team action*")

    if len(explorer) == 0:
        st.warning("Risk data not available. Run pipeline/08_churn_model.py first.")
        st.stop()

    mktg_view = st.selectbox(
        "View mode",
        ["Default View", "Marketing View"],
        key="mktg_view",
        help="Marketing View: campaign actions and budget per segment — no raw probability scores.",
    )

    # --- Sidebar filters ---
    with st.sidebar:
        st.markdown("### Filters")
        all_traj  = sorted(explorer["trajectory_type"].dropna().unique().tolist())
        sel_traj  = st.multiselect("Trajectory Type", all_traj, default=all_traj)
        min_p     = st.slider("Minimum P(Churn)", 0.0, 1.0, 0.0, step=0.05)
        all_rfm   = sorted(explorer["rfm_segment"].dropna().unique().tolist())
        sel_rfm   = st.multiselect("RFM Segment", all_rfm, default=all_rfm)

    # --- Apply filters ---
    filtered = explorer.copy()
    if sel_traj:
        filtered = filtered[filtered["trajectory_type"].isin(sel_traj)]
    if "P_churn" in filtered.columns:
        filtered = filtered[filtered["P_churn"] >= min_p]
    if sel_rfm:
        filtered = filtered[filtered["rfm_segment"].isin(sel_rfm)]

    # --- Summary metrics ---
    s1, s2 = st.columns(2)
    s1.metric("Filtered Customers", f"{len(filtered):,}")
    risk_col = "risk_score" if "risk_score" in filtered.columns else None
    s2.metric("Total Risk Exposure", fc(filtered[risk_col].sum()) if risk_col else "N/A")

    st.markdown("---")

    if mktg_view == "Default View":
        # --- Filtered table ---
        want = ["user_id", "trajectory_type", "rfm_segment", "P_churn", "predicted_loss", "risk_score"]
        show_cols = [c for c in want if c in filtered.columns]
        show = filtered[show_cols].head(500).copy()
        show = show.rename(columns={
            "user_id": "User ID", "trajectory_type": "Trajectory",
            "rfm_segment": "RFM Segment", "P_churn": "P(Churn)",
            "predicted_loss": "Predicted Loss ($)", "risk_score": "Risk Score ($)",
        })
        if "P(Churn)" in show.columns:
            show["P(Churn)"] = show["P(Churn)"].map(lambda x: f"{x:.3f}")
        for col in ["Predicted Loss ($)", "Risk Score ($)"]:
            if col in show.columns:
                show[col] = show[col].map(fc)

        st.dataframe(show, width=700, hide_index=True)
        if len(filtered) > 500:
            st.caption(f"Showing top 500 of {len(filtered):,} filtered customers.")

        # --- Download ---
        csv_bytes = filtered[show_cols].to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download filtered results as CSV",
            data=csv_bytes,
            file_name="at_risk_customers_filtered.csv",
            mime="text/csv",
        )

    else:  # Marketing View
        _MKTG_ACTIONS = {
            "B": "Phone call within 48h",
            "D": "Loyalty reward",
            "A": "Product discount",
            "C": "Push notification",
        }
        _MKTG_COSTS = {"B": 15.00, "D": 8.50, "A": 11.00, "C": 0.50}

        # --- Campaign breakdown by trajectory type ---
        st.subheader("Campaign Breakdown by Trajectory Type")
        if "trajectory_type" in filtered.columns:
            _camp = (
                filtered.groupby("trajectory_type", sort=False)
                .agg(n_customers=("user_id", "count"))
                .reset_index()
            )
            _camp["Type"]            = _camp["trajectory_type"].map(T_LABELS).fillna(_camp["trajectory_type"])
            _camp["Suggested Action"]= _camp["trajectory_type"].map(_MKTG_ACTIONS).fillna("N/A")
            _camp["Cost / Customer"] = _camp["trajectory_type"].map(
                                           lambda t: f"${_MKTG_COSTS.get(t, 0):.2f}")
            _camp["Total Budget"]    = (
                                           _camp["n_customers"]
                                           * _camp["trajectory_type"].map(_MKTG_COSTS)
                                       ).map(lambda x: f"${x:,.0f}")
            _camp["Customers"]       = _camp["n_customers"].map(lambda x: f"{int(x):,}")
            _ord_m = {"B": 0, "D": 1, "A": 2, "C": 3}
            _camp["_o"] = _camp["trajectory_type"].map(_ord_m).fillna(9)
            _camp = _camp.sort_values("_o")
            st.dataframe(
                _camp[["Type", "Customers", "Suggested Action", "Cost / Customer", "Total Budget"]],
                width=750, hide_index=True,
            )
            _total_budget = float(
                (filtered["trajectory_type"].map(_MKTG_COSTS).fillna(0)).sum()
            )
            st.markdown(f"""
            <div class="info-box">
              <b>Total campaign budget for {len(filtered):,} filtered customers:</b>
              &nbsp; {fc(_total_budget)}
            </div>
            """, unsafe_allow_html=True)

        # --- Action list ---
        st.markdown("---")
        st.subheader("Action List")
        st.caption("Simplified view: trajectory, segment, and suggested action only.")
        _mktg_base_cols = ["user_id", "trajectory_type", "rfm_segment"]
        _mktg_base_cols = [c for c in _mktg_base_cols if c in filtered.columns]
        _mktg_show = filtered[_mktg_base_cols].head(500).copy()
        _mktg_show["Suggested Action"] = (
            filtered["trajectory_type"].head(500).map(_MKTG_ACTIONS).fillna("N/A").values
        )
        _mktg_show = _mktg_show.rename(columns={
            "user_id": "User ID", "trajectory_type": "Trajectory",
            "rfm_segment": "RFM Segment",
        })
        st.dataframe(_mktg_show, width=700, hide_index=True)
        if len(filtered) > 500:
            st.caption(f"Showing top 500 of {len(filtered):,} filtered customers.")

        # --- Download ---
        _mktg_dl = filtered[_mktg_base_cols].copy()
        _mktg_dl["suggested_action"] = _mktg_dl["trajectory_type"].map(_MKTG_ACTIONS).fillna("N/A")
        st.download_button(
            label="Download marketing action list as CSV",
            data=_mktg_dl.to_csv(index=False).encode("utf-8"),
            file_name="marketing_action_list.csv",
            mime="text/csv",
        )

# ============================================================
# PAGE 3 — Intervention & ROI
# ============================================================
elif page == "Intervention & ROI":
    st.title("Intervention Strategy by Trajectory Type")
    st.markdown("*When to act, what to offer, and what it costs*")

    OFFER_META = {
        "B": {"hook": "Phone call within 48h",   "cost": 15.00},
        "D": {"hook": "Loyalty reward",           "cost": 8.50},
        "A": {"hook": "Product discount",         "cost": 11.00},
        "C": {"hook": "Push notification",        "cost": 0.50},
    }
    ORD = {"B": 0, "D": 1, "A": 2, "C": 3}

    # --- ROI summary table ---
    if len(roi) > 0:
        r = roi.copy()
        r["_ord"]        = r["trajectory_type"].map(ORD).fillna(9)
        r["Type"]        = r["trajectory_type"].map(T_LABELS).fillna(r["trajectory_type"])
        r["Intervention"]= r["trajectory_type"].map(lambda t: OFFER_META.get(t, {}).get("hook", ""))
        r["Cost"]        = r["trajectory_type"].map(lambda t: f"${OFFER_META.get(t, {}).get('cost', 0):.2f}")
        r = r.sort_values("_ord")
        disp_roi = r[["Type", "n_customers", "actual_churn_pct", "Intervention", "Cost", "avg_ROI_pct"]].copy()
        disp_roi.columns = ["Type", "Customers", "Churn %", "Intervention", "Cost", "ROI %"]
        disp_roi["Customers"] = disp_roi["Customers"].map(lambda x: f"{int(x):,}")
        disp_roi["Churn %"]   = disp_roi["Churn %"].map(lambda x: f"{float(x):.1f}%")
        disp_roi["ROI %"]     = disp_roi["ROI %"].map(lambda x: f"{float(x):,.0f}%")
        st.dataframe(disp_roi, width=700, hide_index=True)
    else:
        fallback = pd.DataFrame({
            "Type":         ["B: Sudden Stop", "D: High Value Drifting", "A: Fading Frequency", "C: Low Loyalty"],
            "Customers":    ["6,010", "9,786", "2,498", "89,144"],
            "Churn %":      ["35.0%", "22.3%", "44.5%", "14.9%"],
            "Intervention": ["Phone call within 48h", "Loyalty reward", "Product discount", "Push notification"],
            "Cost":         ["$15.00", "$8.50", "$11.00", "$0.50"],
            "ROI %":        ["3,820%", "17,891%", "6,075%", "43,849%"],
        })
        st.dataframe(fallback, width=700, hide_index=True)
        r = None

    st.markdown("---")

    # --- ROI bar chart ---
    st.subheader("Average ROI by Trajectory Type")
    if len(roi) > 0:
        rc = roi.copy()
        rc["_ord"]  = rc["trajectory_type"].map(ORD).fillna(9)
        rc["label"] = rc["trajectory_type"].map(T_LABELS).fillna(rc["trajectory_type"])
        rc = rc.sort_values("_ord")
        fig = px.bar(
            rc, x="label", y="avg_ROI_pct", text="avg_ROI_pct",
            color="trajectory_type",
            color_discrete_map=C_TRAJ,
        )
        fig.update_traces(texttemplate="%{text:,.0f}%", textposition="outside")
        chart_layout(fig)
        fig.update_yaxes(title="Average ROI (%)")
        st.plotly_chart(fig, width=700)

    # --- Budget alert ---
    if len(roi) > 0 and "C" in roi["trajectory_type"].values:
        c_n = int(roi.loc[roi["trajectory_type"] == "C", "n_customers"].values[0])
    else:
        c_n = 89144
    wasted = c_n * 11
    st.markdown(f"""
    <div class="alert-box">
      <b>Budget Alert:</b> Mistargeting {c_n:,} Type C customers with an $11 discount
      wastes <b>${wasted:,}</b> annually.
      Type C has a 14.9% churn rate and responds to $0.50 push notifications only.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # --- Intervention windows ---
    st.subheader("Intervention Windows")
    windows = [
        ("B", "#d32f2f", "Act within 48 hours of sudden stop detected"),
        ("D", "#e65100", "Act before gap exceeds 20 days"),
        ("A", "#f57f17", "Act within next 4-6 orders"),
        ("C", "#388e3c", "Light touch only — push notification"),
    ]
    for tt, col, msg in windows:
        st.markdown(f"""
        <div class="window-row" style="border-left: 4px solid {col};">
          <b>Type {tt} ({T_LABELS[tt].split(': ')[1]}):</b> {msg}
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # --- Retention Campaign ROI Simulator ---
    st.subheader("Retention Campaign ROI Simulator")
    st.markdown("*Adjust campaign parameters to estimate impact before committing budget*")

    sim_c1, sim_c2, sim_c3 = st.columns(3)
    with sim_c1:
        sim_customers = st.slider(
            "Customers Targeted", min_value=100, max_value=2498, value=2498, step=50
        )
    with sim_c2:
        sim_cost = st.slider(
            "Campaign Cost Per Customer ($)", min_value=0.50, max_value=15.00,
            value=8.50, step=0.50
        )
    with sim_c3:
        sim_save_rate = st.slider(
            "Expected Save Rate (%)", min_value=5, max_value=40, value=20, step=5
        )

    # Compute outcomes
    _customers_saved = int(sim_customers * sim_save_rate / 100)
    _avg_rev_per_cust = 3.50 * 52 * 0.25   # $3.50/item × ~52 items/yr × 25% at risk
    _rev_protected    = _customers_saved * _avg_rev_per_cust
    _total_cost       = sim_customers * sim_cost
    _net_roi          = ((_rev_protected - _total_cost) / _total_cost * 100) if _total_cost > 0 else 0

    sim_m1, sim_m2, sim_m3 = st.columns(3)
    sim_m1.metric("Customers Saved",    f"{_customers_saved:,}")
    sim_m2.metric("Revenue Protected",  fc(_rev_protected))
    sim_m3.metric("Net ROI",            f"{_net_roi:,.0f}%")

    # Campaign Cost vs Revenue Protected bar chart
    _sim_bar = pd.DataFrame({
        "Metric": ["Campaign Cost", "Revenue Protected"],
        "Value":  [_total_cost, _rev_protected],
    })
    fig_sim = px.bar(
        _sim_bar, x="Metric", y="Value", text="Value",
        color="Metric",
        color_discrete_map={"Campaign Cost": "#e74c3c", "Revenue Protected": "#27ae60"},
    )
    fig_sim.update_traces(texttemplate="$%{text:,.0f}", textposition="outside")
    chart_layout(fig_sim, height=320)
    fig_sim.update_yaxes(title="Amount ($)")
    st.plotly_chart(fig_sim, width=700)

    # Colour-coded result box
    if _net_roi > 500:
        _res_bg, _res_border, _res_icon = "#f0fff4", "#27ae60", "✅"
        _res_msg = "Excellent ROI — campaign is well justified."
    elif _net_roi >= 100:
        _res_bg, _res_border, _res_icon = "#fffbf0", "#e65100", "⚠️"
        _res_msg = "Acceptable ROI — consider targeting higher-value segments."
    else:
        _res_bg, _res_border, _res_icon = "#fff5f5", "#e74c3c", "🚨"
        _res_msg = "Poor ROI — re-evaluate targeting parameters before spend."

    st.markdown(f"""
    <div style="background:{_res_bg};border-left:4px solid {_res_border};border-radius:6px;
                padding:14px 18px;margin:12px 0;font-size:0.95rem;">
      <b>{_res_icon} Campaign Result:</b> {_res_msg}
      Net ROI of <b>{_net_roi:,.0f}%</b> on {sim_customers:,} customers targeted,
      {_customers_saved:,} saved at ${sim_cost:.2f}/customer.
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# PAGE 4 — Category Exposure
# ============================================================
elif page == "Category Exposure":
    st.title("Department Revenue Exposure")
    st.markdown("*Which product categories carry the most revenue risk from declining Champions and Loyal customers*")

    ops_view = st.selectbox(
        "View mode",
        ["Default View", "Ops View"],
        key="ops_view",
        help="Ops View: Core Risk departments and residual watch-list — no full bar chart.",
    )

    if len(cat_risk) > 0 and "dept_total_risk" in cat_risk.columns:
        total_risk  = cat_risk["dept_total_risk"].sum()
        top7_risk   = cat_risk.sort_values("dept_total_risk", ascending=False).head(7)["dept_total_risk"].sum()
        prod_mask   = cat_risk["department"].str.lower().str.strip() == "produce"
        prod_risk   = float(cat_risk.loc[prod_mask, "dept_total_risk"].values[0]) if prod_mask.any() else 0

        e1, e2, e3 = st.columns(3)
        e1.metric("Total 60-Day Revenue at Risk", fc(total_risk))
        e2.metric("Top 7 Departments",            f"{top7_risk / total_risk * 100:.0f}% of total exposure")
        e3.metric("Produce Share",                f"{prod_risk / total_risk * 100:.0f}% of total exposure")

        st.markdown("---")

        if ops_view == "Default View":
            st.subheader("60-Day Revenue at Risk by Department")
            cat_sorted = cat_risk.sort_values("dept_total_risk", ascending=True).copy()
            fig = px.bar(
                cat_sorted, x="dept_total_risk", y="department",
                orientation="h", text="dept_total_risk",
                color="dept_total_risk",
                color_continuous_scale=[[0, "#d6eaf8"], [1, ACCENT]],
            )
            fig.update_traces(texttemplate="$%{text:,.0f}", textposition="outside")
            chart_layout(fig, height=max(380, len(cat_sorted) * 28))
            fig.update_xaxes(title="60-Day Revenue at Risk ($)")
            fig.update_yaxes(title="")
            fig.update_coloraxes(showscale=False)
            st.plotly_chart(fig, width=700)

        else:  # Ops View
            st.subheader("Core Risk Departments — Top 7 by Revenue Exposure")
            st.caption("Departments accounting for ~80% of total 60-day revenue exposure.")
            _top7  = cat_risk.sort_values("dept_total_risk", ascending=False).head(7).copy()
            _rest  = cat_risk.sort_values("dept_total_risk", ascending=False).iloc[7:].copy()
            _top7["60-Day Risk"]  = _top7["dept_total_risk"].map(lambda x: f"${x:,.0f}")
            _top7["% of Total"]   = (_top7["dept_total_risk"] / total_risk * 100).map(
                                        lambda x: f"{x:.1f}%")
            _top7["Priority"]     = "Core Risk"
            st.dataframe(
                _top7[["department", "60-Day Risk", "% of Total", "Priority"]].rename(
                    columns={"department": "Department"}),
                width=600, hide_index=True,
            )
            if len(_rest) > 0:
                _rest_total = _rest["dept_total_risk"].sum()
                st.caption(
                    f"Remaining {len(_rest)} departments: {fc(_rest_total)} combined "
                    f"({_rest_total / total_risk * 100:.1f}% of total exposure) — Standard priority."
                )

    else:
        st.info("category_risk_report.csv not available. Run pipeline/07_category_exposure.py first.")

    st.markdown("---")
    st.subheader("Residual Analysis — Actual vs Predicted Churn by Department")

    if len(dept_res) > 0:
        dr = dept_res.copy()
        dr["Actual Churn %"]    = dr["actual_churn_rate"].map(lambda x: f"{x*100:.1f}%")
        dr["Predicted %"]       = dr["predicted_churn_rate"].map(lambda x: f"{x*100:.1f}%")
        dr["Residual"]          = dr["residual"].map(lambda x: f"{x*100:+.1f}%")
        dr = dr.rename(columns={
            "primary_department": "Department",
            "customer_count":     "Customers",
            "status":             "Status",
        })
        show_dr = dr[["Department", "Customers", "Actual Churn %", "Predicted %", "Residual", "Status"]].copy()
        show_dr["Customers"] = show_dr["Customers"].map(lambda x: f"{int(x):,}")
        st.dataframe(show_dr, width=700, hide_index=True)

        if ops_view == "Ops View" and "residual" in dept_res.columns:
            _watch = dept_res[dept_res["residual"].abs() > 0.03].copy()
            if len(_watch) > 0:
                _watch_names = ", ".join(
                    _watch.sort_values("residual", ascending=False)["primary_department"].tolist()
                )
                st.markdown(f"""
                <div class="alert-box">
                  <b>Watch List ({len(_watch)} department{'s' if len(_watch) != 1 else ''}):</b>
                  {_watch_names} — residual exceeds 3pp. Investigate for operational issues
                  or data quality before next model retrain.
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style="background:#f0fff4;border-left:4px solid #27ae60;border-radius:6px;
                            padding:12px 16px;margin:12px 0;font-size:0.93rem;">
                  <b>All Clear:</b> No department residuals exceed 3pp.
                  Churn is explained by behavioral signals across all categories.
                </div>
                """, unsafe_allow_html=True)
    else:
        st.info("department_residuals.csv not available.")

    st.markdown("""
    <div class="info-box">
      <b>Finding:</b> Behavioral model explains all department churn variation — no operational
      anomalies detected. Churn is driven by customer behavior, not category-level issues.
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# PAGE 5 — Model Performance
# ============================================================
elif page == "Model Performance":
    st.title("Model Validation & Methodology")
    st.markdown("*How the model was built and why the results are honest*")

    # --- Methodology notes ---
    st.subheader("Methodology Notes")
    st.markdown("""
    <div class="info-box">
      Features were validated against actual churn patterns before model training.
      Order gap widening and reorder rate decline were confirmed as the primary behavioral signals.
      Features that did not show statistically significant separation between churned and active
      customers were excluded. Three rounds of leakage detection were performed to ensure model integrity.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # --- Final feature list ---
    st.subheader("Features Used in Final Model")
    feat_df = pd.DataFrame({
        "Feature": [
            "avg_days_between_orders",
            "order_gap_std",
            "order_gap_cv",
            "early_avg_gap",
            "late_avg_gap_clean",
            "order_count",
            "gap_trend",
            "reorder_rate",
        ],
        "Description": [
            "Mean days between all orders",
            "Standard deviation of order gaps",
            "Coefficient of variation in order gaps",
            "Mean of first 3 inter-order gaps",
            "Mean of gaps at positions -4, -3, -2 (final gap excluded)",
            "Total orders placed by customer",
            "late_avg_gap_clean minus early_avg_gap",
            "Proportion of items that are reorders",
        ],
    })
    st.dataframe(feat_df, width=700, hide_index=True)

    st.markdown("---")

    # --- Leakage fixes summary ---
    st.subheader("Leakage Fixes — 3 Rounds")
    LEAKS = [
        ("Round 1", "Removed days_since_last_order — direct label proxy (AUC 1.0000 to 0.9420)"),
        ("Round 2", "Replaced late_avg_gap with late_avg_gap_clean — indirect label proxy via final gap (AUC 0.9420 to 0.9059)"),
        ("Round 3", "Rebuilt Type B on pure gap signals — days_since_last_order >= 25 made trajectory_enc a 79% churn proxy; redefined as early_avg_gap < 12 AND late_avg_gap_clean > 20"),
    ]
    for rnd, line in LEAKS:
        st.markdown(f"""
        <div class="leak-row">
          <b>{rnd}:</b> {line}
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # --- Model comparison table ---
    st.subheader("Model Comparison")
    model_comp = pd.DataFrame({
        "Model":       ["Logistic Regression", "Random Forest", "XGBoost"],
        "Train AUC":   [0.9078, 0.9457, 0.9497],
        "Test AUC":    [0.9066, 0.9390, 0.9420],
        "Gap":         [0.001,  0.007,  0.008],
        "Precision":   [0.602,  0.619,  0.563],
        "Recall":      [0.844,  0.926,  0.988],
        "F1":          [0.703,  0.742,  0.717],
    })
    st.dataframe(model_comp, width=700, hide_index=True)

    st.markdown("""
    <div class="info-box">
      <b>Note:</b> Comparison metrics shown reflect the post-leakage-fix run on behavioral
      signals only. days_since_last_order was excluded. Overfitting gap is below 0.01 for all models.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # --- Final metrics ---
    st.subheader("Final Model Metrics")
    if len(metrics) > 0:
        m = metrics.iloc[0]
        fn1, fn2, fn3, fn4 = st.columns([1.2, 1.2, 1, 1], gap="medium")
        fn1.metric("Best Model",    str(m.get("best_model_name", "XGBoost")))
        fn2.metric("Test AUC",      f"{float(m.get('test_auc', 0.9059)):.4f}")
        fn3.metric("5-Fold CV AUC", f"{float(m.get('cv_mean', 0.907)):.4f} +/- {float(m.get('cv_std', 0.001)):.4f}")
        fn4.metric("Regressor R2",  f"{float(m.get('reg_r2', 0.1826)):.4f}")
        rr1, rr2 = st.columns(2)
        rr1.metric("Regressor RMSE",       f"${float(m.get('reg_rmse', 23.98)):.2f}")
        rr2.metric("Pareto (80% of risk)", f"Top {int(m.get('pareto_80pct_customers', 0)):,} customers")
    else:
        fn1, fn2, fn3, fn4 = st.columns([1.2, 1.2, 1, 1], gap="medium")
        fn1.metric("Best Model",    "XGBoost")
        fn2.metric("Test AUC",      "0.9059")
        fn3.metric("5-Fold CV AUC", "0.9070 +/- 0.0013")
        fn4.metric("Regressor R2",  "0.1826")

    st.markdown("---")

    # --- Feature importance chart ---
    st.subheader("Feature Importance — Churn Classifier")
    fi_path = CHART_FEAT_IMP
    if os.path.exists(fi_path):
        st.image(str(fi_path), use_container_width=True)
    else:
        st.info(f"Feature importance chart not found at {fi_path}.")

# ============================================================
# PAGE 7 — Behavioral Monitor
# ============================================================
elif page == "Behavioral Monitor":
    st.title("Behavioral Monitor — Order Sequence Analysis")
    st.markdown(
        "*Simulates weekly re-scoring using order sequence as time proxy. "
        "In production this runs on fresh transaction data.*"
    )

    # ----------------------------------------------------------
    # Load monitor data files (must live in streamlit/ directory)
    # ----------------------------------------------------------
    _DATA_DELTA = _HERE / "monitor_delta.csv"
    _DATA_SNAP1 = _HERE / "snapshot_1.csv"
    _DATA_SNAP3 = _HERE / "snapshot_3.csv"

    if not (_DATA_DELTA.exists() and _DATA_SNAP1.exists() and _DATA_SNAP3.exists()):
        st.warning(
            "Run pipeline/10_weekly_monitor.py first to generate monitor data"
        )
        st.stop()

    @st.cache_data
    def _load_monitor():
        delta = pd.read_csv(_DATA_DELTA)
        snap1 = pd.read_csv(_DATA_SNAP1)
        snap3 = pd.read_csv(_DATA_SNAP3)
        return delta, snap1, snap3

    delta, snap1, snap3 = _load_monitor()

    # Count each status category for the metric cards
    n_newly       = int((delta["status"] == "newly_at_risk").sum())
    n_recovering  = int((delta["status"] == "recovering").sum())
    n_stable_high = int((delta["status"] == "stable_high").sum())
    n_stable_low  = int((delta["status"] == "stable_low").sum())

    # ----------------------------------------------------------
    # Row 1 — Metric cards
    # ----------------------------------------------------------
    mc1, mc2, mc3, mc4 = st.columns([1.2, 1.2, 1, 1], gap="medium")
    mc1.metric("🔴 Newly At-Risk",   f"{n_newly:,}",       help="Was Low/Medium in S1, now High in S3")
    mc2.metric("🟢 Recovering",      f"{n_recovering:,}",  help="Was High in S1, now Low/Medium in S3")
    mc3.metric("⚠️ Stable High",     f"{n_stable_high:,}", help="High risk in both snapshots")
    mc4.metric("✅ Stable Low",      f"{n_stable_low:,}",  help="Low/Medium risk in both snapshots")

    st.markdown("---")

    # ----------------------------------------------------------
    # Row 2 — Two charts side by side
    # ----------------------------------------------------------
    _STATUS_LABELS = {
        "newly_at_risk": "Newly At-Risk",
        "recovering":    "Recovering",
        "stable_high":   "Stable High",
        "stable_low":    "Stable Low",
    }
    _STATUS_COLORS = {
        "Newly At-Risk": "#d32f2f",
        "Recovering":    "#27ae60",
        "Stable High":   "#e65100",
        "Stable Low":    "#388e3c",
    }

    row2_l, row2_r = st.columns(2)

    with row2_l:
        st.subheader("Customers by Status")
        status_cnt = delta["status"].value_counts().reset_index()
        status_cnt.columns = ["status", "count"]
        status_cnt["label"] = status_cnt["status"].map(_STATUS_LABELS).fillna(status_cnt["status"])
        fig_cnt = px.bar(
            status_cnt, x="label", y="count",
            text="count",
            color="label",
            color_discrete_map=_STATUS_COLORS,
        )
        fig_cnt.update_traces(texttemplate="%{text:,}", textposition="outside")
        chart_layout(fig_cnt, height=340)
        fig_cnt.update_yaxes(title="Customers")
        fig_cnt.update_xaxes(title="")
        st.plotly_chart(fig_cnt, width=700)

    with row2_r:
        st.subheader("Avg Reorder Rate by Status")
        avg_rr = (
            delta.groupby("status")["reorder_rate"]
            .mean()
            .reset_index()
        )
        avg_rr["label"] = avg_rr["status"].map(_STATUS_LABELS).fillna(avg_rr["status"])
        avg_rr["reorder_rate"] = avg_rr["reorder_rate"].round(3)
        fig_rr = px.bar(
            avg_rr, x="label", y="reorder_rate",
            text="reorder_rate",
            color="label",
            color_discrete_map=_STATUS_COLORS,
        )
        fig_rr.update_traces(texttemplate="%{text:.3f}", textposition="outside")
        chart_layout(fig_rr, height=340)
        fig_rr.update_yaxes(title="Avg Reorder Rate", range=[0, 1])
        fig_rr.update_xaxes(title="")
        st.plotly_chart(fig_rr, width=700)

    st.markdown("---")

    # ----------------------------------------------------------
    # Row 3 — Newly at-risk customers table
    # ----------------------------------------------------------
    st.subheader("Newly At-Risk Customers — Act Now")

    newly_df = delta[delta["status"] == "newly_at_risk"].copy()

    if len(newly_df) == 0:
        st.success("No newly at-risk customers detected this cycle.")
    else:
        newly_sorted = newly_df.sort_values("revenue_at_risk_90d", ascending=False)
        show_cols_mon = [
            "user_id", "avg_days_between_orders", "reorder_rate",
            "gap_trend", "order_count", "revenue_at_risk_90d",
        ]
        show_cols_mon = [c for c in show_cols_mon if c in newly_sorted.columns]
        disp_newly = newly_sorted[show_cols_mon].copy()
        disp_newly = disp_newly.rename(columns={
            "user_id":                 "User ID",
            "avg_days_between_orders": "Avg Days",
            "reorder_rate":            "Reorder Rate",
            "gap_trend":               "Gap Trend",
            "order_count":             "Orders",
            "revenue_at_risk_90d":     "Revenue at Risk (90d)",
        })
        if "Avg Days" in disp_newly.columns:
            disp_newly["Avg Days"] = disp_newly["Avg Days"].round(1)
        if "Reorder Rate" in disp_newly.columns:
            disp_newly["Reorder Rate"] = disp_newly["Reorder Rate"].round(3)
        if "Gap Trend" in disp_newly.columns:
            disp_newly["Gap Trend"] = disp_newly["Gap Trend"].round(2)
        if "Revenue at Risk (90d)" in disp_newly.columns:
            disp_newly["Revenue at Risk (90d)"] = disp_newly["Revenue at Risk (90d)"].map(fc)

        st.dataframe(disp_newly, width=700, hide_index=True)

        csv_mon = newly_sorted[show_cols_mon].to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download newly at-risk customers as CSV",
            data=csv_mon,
            file_name="newly_at_risk_customers.csv",
            mime="text/csv",
        )

    st.markdown("---")

    # ----------------------------------------------------------
    # Row 4 — Risk tier shift grouped bar chart
    # ----------------------------------------------------------
    st.subheader("Risk Tier Shift — Snapshot 1 vs Snapshot 3")
    st.caption(
        "Shows how many customers moved between risk tiers as order history accumulated. "
        "Bars are grouped by Snapshot 1 tier; colours show their Snapshot 3 tier."
    )

    tier_order = ["Low", "Medium", "High"]
    tier_colors = {"Low": "#388e3c", "Medium": "#e65100", "High": "#d32f2f"}

    tier_shift = (
        delta.groupby(["s1_risk_tier", "s3_risk_tier"])
        .size()
        .reset_index(name="count")
    )
    # Enforce ordering
    tier_shift["s1_risk_tier"] = pd.Categorical(
        tier_shift["s1_risk_tier"], categories=tier_order, ordered=True
    )
    tier_shift["s3_risk_tier"] = pd.Categorical(
        tier_shift["s3_risk_tier"], categories=tier_order, ordered=True
    )
    tier_shift = tier_shift.sort_values(["s1_risk_tier", "s3_risk_tier"])

    fig_shift = px.bar(
        tier_shift,
        x="s1_risk_tier", y="count",
        color="s3_risk_tier",
        barmode="group",
        text="count",
        labels={
            "s1_risk_tier": "Snapshot 1 Risk Tier",
            "count":        "Number of Customers",
            "s3_risk_tier": "Snapshot 3 Tier",
        },
        color_discrete_map=tier_colors,
    )
    fig_shift.update_traces(texttemplate="%{text:,}", textposition="outside")
    chart_layout(fig_shift, height=420)
    fig_shift.update_layout(showlegend=True, legend_title_text="Snapshot 3 Tier")
    st.plotly_chart(fig_shift, width=700)

# ============================================================
# PAGE 8 — Cohort Retention
# ============================================================
elif page == "Cohort Retention":
    st.title("Cohort Retention Analysis")
    st.markdown("---")

    # Load cohort data
    cohort_csv = pd.read_csv(str(_HERE / "cohort_retention.csv"))

    # Display heatmap
    st.subheader("Retention Heatmap by Customer Cohort")
    st.image(str(_HERE / "cohort_heatmap.png"), use_container_width=True)

    # Key insight box - readable formatting
    st.info("""
### 📊 Key Finding: 64.8 Percentage Point Gap

**Very Frequent Customers** (≤7 days between orders)
- Retention at order 10: **83.9%**
- Customer count: 23,147 (13.2% of cohort)

**Infrequent Customers** (>21 days between orders)
- Retention at order 10: **19.1%**
- Customer count: 32,045 (18.3% of cohort)

**Gap**: 83.9% - 19.1% = **64.8 percentage points**

This massive gap suggests ordering cadence is the strongest predictor of long-term retention.
""")

    # Cohort table with proper width
    st.subheader("Cohort Breakdown")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Cohort Distribution**")
        st.dataframe(cohort_csv, use_container_width=True)

    with col2:
        st.markdown("**What This Means**")
        st.markdown("""
- Group customers by average days between orders
- Track how many return for order 2, 3, 4... up to 10
- Calculate retention % at each step

**Insight**: Frequent buyers are sticky. Infrequent buyers drop off fast.
""")

    # Download button
    st.download_button(
        label="📥 Download Cohort Retention Data (CSV)",
        data=(_HERE / "cohort_retention.csv").read_bytes(),
        file_name="cohort_retention.csv",
        mime="text/csv",
    )
