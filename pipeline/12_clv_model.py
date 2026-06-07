# ============================================================
# clv_model.py — 12-Month Customer Lifetime Value Scoring
# ============================================================
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

import pandas as pd
import numpy as np

print("=" * 60)
print("CLV Model — clv_model.py")
print("=" * 60)

# ----------------------------------------------------------------
# 1. Load data
# ----------------------------------------------------------------
print(f"\nLoading {config.TRAJECTORY_SEGMENTS}...")
df = pd.read_csv(config.TRAJECTORY_SEGMENTS)
print(f"  Loaded: {len(df):,} customers")
print(f"  Columns: {list(df.columns)}")

# Guard against bad rows
df = df.dropna(subset=['user_id', 'reorder_rate', 'avg_days_between_orders', 'annual_value'])
df = df[df['avg_days_between_orders'] > 0].copy().reset_index(drop=True)
print(f"  After quality filter: {len(df):,} customers")

# ----------------------------------------------------------------
# 2. Calculate 12-month CLV
#
#   annual_frequency  = 365 / avg_days_between_orders   (orders/year)
#   basket_value      = annual_value / annual_frequency  ($ per order, derived from file)
#   clv_12m           = reorder_rate × annual_frequency × basket_value
#                     = reorder_rate × annual_value      (algebraic simplification)
#
#   reorder_rate acts as a loyalty discount:
#   a customer who reorders 80% of items retains 80% of their expected annual value.
# ----------------------------------------------------------------
df['annual_frequency'] = 365.0 / df['avg_days_between_orders']
df['basket_value']     = df['annual_value'] / df['annual_frequency']
df['clv_12m']          = df['reorder_rate'] * df['annual_frequency'] * df['basket_value']

print(f"\n  CLV range: ${df['clv_12m'].min():,.2f} – ${df['clv_12m'].max():,.2f}")
print(f"  Median CLV: ${df['clv_12m'].median():,.2f}")

# ----------------------------------------------------------------
# 3. Segment CLV into 4 tiers using quartile thresholds
# ----------------------------------------------------------------
q1 = df['clv_12m'].quantile(0.25)
q2 = df['clv_12m'].quantile(0.50)
q3 = df['clv_12m'].quantile(0.75)

print(f"\n  CLV tier thresholds (quartile-based):")
print(f"    At-Risk : CLV < ${q1:,.2f}  (bottom 25%)")
print(f"    Low     : ${q1:,.2f} – ${q2:,.2f}")
print(f"    Medium  : ${q2:,.2f} – ${q3:,.2f}")
print(f"    High    : CLV >= ${q3:,.2f}  (top 25%)")

def assign_tier(clv):
    if clv >= q3:
        return 'High'
    elif clv >= q2:
        return 'Medium'
    elif clv >= q1:
        return 'Low'
    else:
        return 'At-Risk'

df['clv_tier'] = df['clv_12m'].apply(assign_tier)

# ----------------------------------------------------------------
# 4. Average CLV per trajectory_type (A/B/C/D)
# ----------------------------------------------------------------
print("\n" + "-" * 55)
print(f"{'Trajectory':<12} {'Customers':>10} {'Avg CLV':>12} {'Avg Reorder':>12}")
print("-" * 55)
traj_grp = df.groupby('trajectory_type').agg(
    n_customers=('clv_12m', 'count'),
    avg_clv=('clv_12m', 'mean'),
    avg_reorder=('reorder_rate', 'mean'),
).sort_values('avg_clv', ascending=False)

for t, row in traj_grp.iterrows():
    print(f"  {t:<10} {int(row['n_customers']):>10,} ${row['avg_clv']:>10,.2f}  {row['avg_reorder']:>10.1%}")
print("-" * 55)

# ----------------------------------------------------------------
# 5. Save outputs/clv_scores.csv
# ----------------------------------------------------------------
os.makedirs(config.OUTPUTS_DIR, exist_ok=True)
clv_out_path = os.path.join(config.OUTPUTS_DIR, 'clv_scores.csv')
output = df[['user_id', 'clv_12m', 'clv_tier', 'trajectory_type']].copy()
output.to_csv(clv_out_path, index=False)
print(f"\nSaved: {clv_out_path}  ({len(output):,} rows)")

# ----------------------------------------------------------------
# 6. Summary report
# ----------------------------------------------------------------
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)

# Total customers scored
print(f"\nTotal customers scored: {len(df):,}")

# Average CLV per tier
tier_order = ['High', 'Medium', 'Low', 'At-Risk']
print("\nAverage CLV per tier:")
tier_grp = df.groupby('clv_tier').agg(
    count=('clv_12m', 'count'),
    avg_clv=('clv_12m', 'mean'),
    total_clv=('clv_12m', 'sum'),
)
for tier in tier_order:
    if tier in tier_grp.index:
        row = tier_grp.loc[tier]
        print(f"  {tier:<8}: {int(row['count']):>7,} customers | "
              f"avg ${row['avg_clv']:>8,.2f} | "
              f"total ${row['total_clv']:>12,.0f}")

# Average CLV per trajectory type
print("\nAverage CLV per trajectory type:")
for t, row in traj_grp.iterrows():
    label = {
        'A': 'Fading Frequency',
        'B': 'Sudden Stop',
        'C': 'Low Loyalty',
        'D': 'High-Value Drifting',
    }.get(t, t)
    print(f"  Type {t} ({label:<20}): ${row['avg_clv']:>10,.2f}  ({int(row['n_customers']):,} customers)")

# Top 10 highest value customers
print("\nTop 10 highest-value customers:")
top10 = df.nlargest(10, 'clv_12m')[['user_id', 'clv_12m', 'clv_tier', 'trajectory_type']].reset_index(drop=True)
top10.index += 1
for i, row in top10.iterrows():
    print(f"  {i:>2}. user_id={int(row['user_id']):<8} "
          f"CLV=${row['clv_12m']:>10,.2f}  "
          f"tier={row['clv_tier']:<8} "
          f"type={row['trajectory_type']}")

print("\n" + "=" * 60)
print("CLV scoring complete.")
print("=" * 60)
