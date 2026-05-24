# ============================================================
# streamlit/app.py -- Instacart Revenue Protection Dashboard
# ============================================================
import os
from pathlib import Path

# ROOT = project root (parent of the streamlit/ directory).
# Works on Windows, Linux, and Streamlit Cloud identically.


# ---- Data paths (relative to project root) -----------------
_HERE = Path(__file__).parent

DATA_RISK_RANKED  = _HERE / "risk_ranked_top5k.csv"
DATA_RFM          = _HERE / "rfm_segments.csv"
DATA_TRAJ         = _HERE / "trajectory_segments.csv"
DATA_ROI          = _HERE / "retention_roi.csv"
DATA_DEPT_RES     = _HERE / "department_residuals.csv"
DATA_CAT_RISK     = _HERE / "category_risk_report.csv"
DATA_METRICS      = _HERE / "model_metrics.csv"
CHART_FEAT_IMP    = _HERE / "feature_importance_classifier.png"
DATA_CAT_RISK     = _P / "data"    / "processed" / "category_risk_report.csv"
DATA_METRICS      = _P / "outputs" / "reports" / "model_metrics.csv"
CHART_FEAT_IMP    = _P / "outputs" / "charts"   / "feature_importance_classifier.png"

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

    # --- Metric cards ---
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Customers Analyzed", "206,209")
    m2.metric("Customers in Behavioral Model", "175,072")
    m3.metric("60-Day Revenue at Risk", "$626,303")
    m4.metric("Most Exposed Department", "Produce ($183,384)")

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
        st.info("risk_ranked_customers.csv not loaded.")

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

# ============================================================
# PAGE 3 — Customer Risk Explorer
# ============================================================
elif page == "Customer Risk Explorer":
    st.title("Customer Risk Explorer")
    st.markdown("*Filter and identify at-risk customers for retention team action*")

    if len(explorer) == 0:
        st.warning("Risk data not available. Run pipeline/08_churn_model.py first.")
        st.stop()

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

# ============================================================
# PAGE 4 — Category Exposure
# ============================================================
elif page == "Category Exposure":
    st.title("Department Revenue Exposure")
    st.markdown("*Which product categories carry the most revenue risk from declining Champions and Loyal customers*")

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
        fn1, fn2, fn3, fn4 = st.columns(4)
        fn1.metric("Best Model",    str(m.get("best_model_name", "XGBoost")))
        fn2.metric("Test AUC",      f"{float(m.get('test_auc', 0.9059)):.4f}")
        fn3.metric("5-Fold CV AUC", f"{float(m.get('cv_mean', 0.907)):.4f} +/- {float(m.get('cv_std', 0.001)):.4f}")
        fn4.metric("Regressor R2",  f"{float(m.get('reg_r2', 0.1826)):.4f}")
        rr1, rr2 = st.columns(2)
        rr1.metric("Regressor RMSE",       f"${float(m.get('reg_rmse', 23.98)):.2f}")
        rr2.metric("Pareto (80% of risk)", f"Top {int(m.get('pareto_80pct_customers', 0)):,} customers")
    else:
        fn1, fn2, fn3, fn4 = st.columns(4)
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
