# ============================================================
# pipeline/10_weekly_monitor.py — Behavioral Monitor
# ============================================================
# Simulates a weekly monitoring pipeline using order sequence
# as a time proxy. Customers are re-scored at three "snapshots"
# based on how much order history we can see (early / mid / full).
#
# The idea: in production this runs on fresh transaction data each week.
# Here we proxy time with order_count cutoffs so we can show how risk
# profiles shift as customer history accumulates.
#
# Outputs:
#   outputs/reports/snapshot_1.csv   — early-stage customers (<=10 orders)
#   outputs/reports/snapshot_2.csv   — mid-stage customers (<=15 orders)
#   outputs/reports/snapshot_3.csv   — all customers (full history)
#   outputs/reports/monitor_delta.csv — tier changes S1 vs S3
#   streamlit/ copies of the above   — for dashboard
#
# Run from project root:
#   D:/Users/Sakthi/anaconda3/python.exe pipeline/10_weekly_monitor.py
# ============================================================

import sys
import os
import pandas as pd
import numpy as np

# Add project root to path so config.py is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# ============================================================
# Paths
# ============================================================
_BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SNAP1_PATH    = os.path.join(config.REPORTS_DIR, "snapshot_1.csv")
SNAP2_PATH    = os.path.join(config.REPORTS_DIR, "snapshot_2.csv")
SNAP3_PATH    = os.path.join(config.REPORTS_DIR, "snapshot_3.csv")
DELTA_PATH    = os.path.join(config.REPORTS_DIR, "monitor_delta.csv")

STREAMLIT_DIR = os.path.join(_BASE_DIR, "streamlit")

# Create output directories if they don't exist yet
os.makedirs(config.REPORTS_DIR, exist_ok=True)
os.makedirs(STREAMLIT_DIR, exist_ok=True)

# ============================================================
# Load customer features
# ============================================================
# Only load the columns we need for speed — customer_features.csv
# is 61 MB so usecols makes a real difference.
COLS_NEEDED = [
    "user_id",
    "order_count",
    "gap_trend",
    "reorder_rate",
    "avg_days_between_orders",
    "revenue_at_risk_90d",
]

print("Loading customer_features.csv ...")
cf = pd.read_csv(config.CUSTOMER_FEATURES, usecols=COLS_NEEDED)
print(f"  Loaded {len(cf):,} customers.")

# Fill any NaN values with column medians so the risk formula
# doesn't silently drop rows. These are rare (pipeline 03 cleaned
# the dataset) but we want the monitor to be robust.
for col in ["gap_trend", "reorder_rate", "avg_days_between_orders", "revenue_at_risk_90d"]:
    if cf[col].isna().any():
        fill_val = cf[col].median()
        cf[col] = cf[col].fillna(fill_val)

# ============================================================
# Part 1 — Build risk snapshots
# ============================================================
# Risk formula: combines gap_trend (are gaps growing?) and
# reorder_rate (is loyalty declining?).
# Weights: 0.4 for gap trend, 0.6 for reorder rate drop.
# Both dimensions were validated as the strongest churn signals
# in pipeline/validate_signals.py.

def make_snapshot(df_subset):
    """
    Compute risk scores and tiers for a given customer subset.
    Returns a DataFrame with user_id, order_count, risk_score, risk_tier.
    """
    raw = df_subset["gap_trend"] * 0.4 + (1 - df_subset["reorder_rate"]) * 0.6

    # Min-max normalization to [0, 1] within this snapshot's population.
    # Normalizing per snapshot means tier thresholds mean the same thing
    # across snapshots regardless of absolute value differences.
    r_min = raw.min()
    r_max = raw.max()
    denom = r_max - r_min
    if denom < 1e-9:
        score = pd.Series(0.5, index=raw.index)
    else:
        score = (raw - r_min) / denom

    # Assign risk tiers: thresholds are relative to the [0,1] scale.
    def _tier(s):
        if s >= 0.6:
            return "High"
        elif s >= 0.3:
            return "Medium"
        else:
            return "Low"

    tier = score.map(_tier)

    result = df_subset[["user_id", "order_count"]].copy()
    result["risk_score"] = score.round(4).values
    result["risk_tier"]  = tier.values
    return result.reset_index(drop=True)


# Snapshot 1 — early behavior (order_count <= 10)
snap1_input = cf[cf["order_count"] <= 10].copy()
snap2_input = cf[cf["order_count"] <= 15].copy()
snap3_input = cf.copy()

print(f"\nBuilding snapshots...")
print(f"  Snapshot 1 (order_count <= 10): {len(snap1_input):,} customers")
print(f"  Snapshot 2 (order_count <= 15): {len(snap2_input):,} customers")
print(f"  Snapshot 3 (all customers):     {len(snap3_input):,} customers")

snap1 = make_snapshot(snap1_input)
snap2 = make_snapshot(snap2_input)
snap3 = make_snapshot(snap3_input)

# Save snapshots to outputs/reports/
snap1.to_csv(SNAP1_PATH, index=False)
snap2.to_csv(SNAP2_PATH, index=False)
snap3.to_csv(SNAP3_PATH, index=False)

# Also copy to streamlit/ so the dashboard can find them
snap1.to_csv(os.path.join(STREAMLIT_DIR, "snapshot_1.csv"), index=False)
snap3.to_csv(os.path.join(STREAMLIT_DIR, "snapshot_3.csv"), index=False)

print(f"  Snapshots saved.")

# ============================================================
# Part 2 — Delta report: Snapshot 1 vs Snapshot 3
# ============================================================
# For each customer in Snapshot 1 (early-stage), compare their
# risk tier from the early snapshot vs their tier in the full
# customer scoring. This shows who deteriorated, who improved.
#
# Snapshot 1 and Snapshot 3 overlap on user_id because Snapshot 3
# includes everyone. The inner join gives us all S1 customers
# re-evaluated against the full population's normalization scale.

compare = (
    snap1[["user_id", "risk_tier"]]
    .rename(columns={"risk_tier": "s1_risk_tier"})
    .merge(
        snap3[["user_id", "risk_tier"]].rename(columns={"risk_tier": "s3_risk_tier"}),
        on="user_id",
        how="inner",
    )
)

# Pull in behavioral columns and revenue for context
extra = cf[["user_id", "reorder_rate", "gap_trend",
             "avg_days_between_orders", "order_count", "revenue_at_risk_90d"]].copy()
compare = compare.merge(extra, on="user_id", how="left")


def _assign_status(s1t, s3t):
    """
    Classify the tier change between Snapshot 1 and Snapshot 3.
      newly_at_risk — customer was Low/Medium in S1, now High in S3
      recovering    — customer was High in S1, now Low/Medium in S3
      stable_high   — High in both (ongoing concern)
      stable_low    — all others (not deteriorating, not critical)
    """
    if s1t in ("Low", "Medium") and s3t == "High":
        return "newly_at_risk"
    elif s1t == "High" and s3t in ("Low", "Medium"):
        return "recovering"
    elif s1t == "High" and s3t == "High":
        return "stable_high"
    else:
        return "stable_low"


compare["status"] = compare.apply(
    lambda r: _assign_status(r["s1_risk_tier"], r["s3_risk_tier"]), axis=1
)

# Final column order as specified
delta = compare[[
    "user_id", "s1_risk_tier", "s3_risk_tier", "status",
    "reorder_rate", "gap_trend", "avg_days_between_orders",
    "order_count", "revenue_at_risk_90d",
]].copy()

# Save delta report
delta.to_csv(DELTA_PATH, index=False)
delta.to_csv(os.path.join(STREAMLIT_DIR, "monitor_delta.csv"), index=False)

# ============================================================
# Part 3 — Monday Morning Monitor Report
# ============================================================
# Count each status category
n_total       = len(delta)
n_newly       = int((delta["status"] == "newly_at_risk").sum())
n_recovering  = int((delta["status"] == "recovering").sum())
n_stable_high = int((delta["status"] == "stable_high").sum())
n_stable_low  = int((delta["status"] == "stable_low").sum())

# Top 5 newly at-risk customers by revenue_at_risk_90d
newly_df = delta[delta["status"] == "newly_at_risk"].copy()
top5_cols = ["user_id", "avg_days_between_orders", "reorder_rate", "revenue_at_risk_90d"]
top5 = newly_df.nlargest(min(5, len(newly_df)), "revenue_at_risk_90d")[top5_cols].copy()

# Print the report
print()
print("=" * 60)
print("WEEKLY BEHAVIORAL MONITOR — INSTACART RETENTION TEAM")
print("Snapshot window: Order sequence proxy (early / mid / full)")
print("=" * 60)
print(f"Total customers monitored: {n_total:,}")
print()
print(f"ALERT — Newly At-Risk:     {n_newly:,} customers deteriorated")
print(f"GOOD  — Recovering:        {n_recovering:,} customers improving")
print(f"INFO  — Stable High Risk:  {n_stable_high:,} customers remain critical")
print(f"INFO  — Stable Low Risk:   {n_stable_low:,} customers remain healthy")
print()

if len(top5) > 0:
    print("Top 5 newly at-risk customers by revenue_at_risk_90d:")
    # Format for clean console output
    top5_fmt = top5.copy()
    top5_fmt["avg_days_between_orders"] = top5_fmt["avg_days_between_orders"].round(1)
    top5_fmt["reorder_rate"]            = top5_fmt["reorder_rate"].round(3)
    top5_fmt["revenue_at_risk_90d"]     = top5_fmt["revenue_at_risk_90d"].map(
        lambda x: f"${x:,.2f}"
    )
    top5_fmt = top5_fmt.rename(columns={
        "user_id":                  "user_id",
        "avg_days_between_orders":  "avg_days",
        "reorder_rate":             "reorder_rate",
        "revenue_at_risk_90d":      "revenue_90d",
    })
    print(top5_fmt.to_string(index=False))
else:
    print("No newly at-risk customers detected.")

print("=" * 60)

print(f"\nSaved:")
print(f"  {SNAP1_PATH}")
print(f"  {SNAP2_PATH}")
print(f"  {SNAP3_PATH}")
print(f"  {DELTA_PATH}")
print(f"  streamlit/ copies written for dashboard.")
print("\nDone.")
