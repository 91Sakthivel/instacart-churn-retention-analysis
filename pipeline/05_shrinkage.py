# ============================================================
# 03_shrinkage.py — Basket Shrinkage Detection
# ============================================================
# Core question: who is quietly buying less and less?
#
# Per customer, we fit a simple linear trend on their last
# 5 basket sizes. If the slope is negative AND statistically
# significant (p < 0.05), they're flagged as genuinely
# shrinking — not just random noise.
#
# Design choices:
#   - 5-order window: recent enough to be actionable, enough
#     points to fit a line with meaningful degrees of freedom
#   - Vectorized OLS: same math as scipy.linregress but ~100x
#     faster — runs 182K regressions in under a second
#   - $3.50/item: reasonable grocery average, honest proxy
#     since Instacart's public dataset has no price data
#   - Triangular sum for items at risk: accounts for the
#     cumulative decline across all 90-day orders, not just
#     the endpoint
# ============================================================

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

import pandas as pd
import numpy as np
from scipy.stats import t as t_dist
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


AVG_ITEM_PRICE     = 3.50   # $/item proxy — no price data in this dataset
REGRESSION_WINDOW  = 5      # last N orders to fit the trend on
P_VALUE_THRESHOLD  = 0.05   # significance cutoff
DATASET_AVG_GAP    = 11.1   # fallback avg days between orders (from 01_explore)


# -------------------------------------------------------
# 1. Load — only numeric columns needed for basket sizing
# -------------------------------------------------------
print("=" * 60)
print("Instacart Basket Shrinkage — 03 Shrinkage Detection")
print("=" * 60)

if not os.path.exists(config.MASTER_ORDERS):
    print(f"\n  ERROR: {config.MASTER_ORDERS} not found.")
    print("  Run pipeline/01_load_explore.py first.")
    sys.exit(1)

print("\nLoading master_orders.csv (numeric columns only)...")

master = pd.read_csv(
    config.MASTER_ORDERS,
    usecols=['user_id', 'order_id', 'order_number', 'days_since_prior_order'],
    dtype={'user_id': 'int32', 'order_id': 'int32', 'order_number': 'int16'}
)
print(f"  {len(master):,} rows | {master['user_id'].nunique():,} customers")
print(f"  Memory: {master.memory_usage(deep=True).sum() / 1e6:.0f} MB")


# -------------------------------------------------------
# 2. Basket size + collapse to order level
#    Count rows per order_id before deduplication —
#    that row count is the basket size.
# -------------------------------------------------------
print("\nComputing basket sizes per order...")

basket_sizes = master.groupby('order_id').size().reset_index(name='basket_size')

orders = (master
          .drop_duplicates(subset='order_id')
          [['user_id', 'order_id', 'order_number', 'days_since_prior_order']]
          .merge(basket_sizes, on='order_id'))

del master
print(f"  {len(orders):,} orders | avg basket: {orders['basket_size'].mean():.1f} items")


# -------------------------------------------------------
# 3. Filter to customers with >= 5 orders
# -------------------------------------------------------
order_counts  = orders.groupby('user_id')['order_id'].count()
eligible_ids  = order_counts[order_counts >= REGRESSION_WINDOW].index
orders_elig   = orders[orders['user_id'].isin(eligible_ids)].copy()

pct_elig = len(eligible_ids) / orders['user_id'].nunique() * 100
print(f"\n  {len(eligible_ids):,} customers qualify ({pct_elig:.1f}%) — have {REGRESSION_WINDOW}+ orders")


# -------------------------------------------------------
# 4. Average order gap per customer
#    Used to project how many orders land in 90 days.
#    First orders have NaN days_since — pandas mean() skips
#    them automatically, so the average is over real gaps.
# -------------------------------------------------------
avg_days = (orders_elig
            .groupby('user_id')['days_since_prior_order']
            .mean()
            .fillna(DATASET_AVG_GAP)
            .clip(lower=1.0)
            .reset_index()
            .rename(columns={'days_since_prior_order': 'avg_days_between_orders'}))


# -------------------------------------------------------
# 5. Build regression input matrix Y — shape (n, 5)
#    Rows = customers, columns = basket size at orders
#    t-4, t-3, t-2, t-1, t (last 5 in chronological order).
# -------------------------------------------------------
print(f"\nBuilding last-{REGRESSION_WINDOW} matrix for vectorized regression...")

last5 = (orders_elig
         .sort_values(['user_id', 'order_number'])
         .groupby('user_id')
         .tail(REGRESSION_WINDOW)
         .copy())

last5['pos'] = last5.groupby('user_id').cumcount() + 1

# Sanity check — every eligible user must contribute exactly 5 rows
counts_ok = (last5.groupby('user_id').size() == REGRESSION_WINDOW).all()
if not counts_ok:
    print("  WARNING: some users have unexpected row counts — filtering to clean subset")
    good_users = last5.groupby('user_id').size()
    good_users = good_users[good_users == REGRESSION_WINDOW].index
    last5 = last5[last5['user_id'].isin(good_users)]

Y_df     = last5.pivot(index='user_id', columns='pos', values='basket_size')
Y        = Y_df.values.astype(float)   # (n_customers, 5)
user_ids = Y_df.index.values

print(f"  Matrix shape: {Y.shape[0]:,} × {Y.shape[1]}  (customers × last-5 orders)")


# -------------------------------------------------------
# 6. Vectorized OLS
#
#    x = [1,2,3,4,5], fixed for all customers.
#    Sxx = Σ(x - x̄)² = 10.0  (constant — compute once)
#
#    slope = Sxy / Sxx
#    Residuals → se_residuals → se_slope → t-stat → p-val
#
#    t-test with df = 3 (n-2), two-tailed.
#    This is identical to scipy.linregress, just vectorized.
# -------------------------------------------------------
print("Running vectorized OLS across all customers...")

x      = np.arange(1, REGRESSION_WINDOW + 1, dtype=float)
x_mean = x.mean()                          # 3.0
x_c    = x - x_mean                        # [-2, -1, 0, 1, 2]
Sxx    = (x_c ** 2).sum()                  # 10.0

y_means  = Y.mean(axis=1, keepdims=True)
y_c      = Y - y_means
Sxy      = (x_c * y_c).sum(axis=1)
slopes   = Sxy / Sxx

Y_hat     = y_means + slopes[:, np.newaxis] * x_c
residuals = Y - Y_hat
sse       = (residuals ** 2).sum(axis=1)
sst       = (y_c ** 2).sum(axis=1)

se_res    = np.sqrt(sse / (REGRESSION_WINDOW - 2))
se_slope  = se_res / np.sqrt(Sxx)

# Guard against flat lines (identical basket sizes) — set t=0 for those
t_stats  = np.where(se_slope > 1e-10, slopes / se_slope, 0.0)
p_values = 2 * t_dist.sf(np.abs(t_stats), df=REGRESSION_WINDOW - 2)
r_sq     = np.where(sst > 1e-10, 1 - sse / sst, 0.0)

print(f"  Done — {len(slopes):,} regressions")


# -------------------------------------------------------
# 7. Assemble results DataFrame
# -------------------------------------------------------
results = pd.DataFrame({
    'user_id':          user_ids,
    'slope':            slopes,
    'p_value':          p_values,
    'r_squared':        r_sq,
    'avg_basket_size':  Y.mean(axis=1),
    'last_basket_size': Y[:, -1],
})

results = results.merge(avg_days, on='user_id')

results['shrinkage_velocity'] = np.where(
    results['avg_basket_size'] > 0,
    results['slope'] / results['avg_basket_size'],
    0.0
)

results['shrinking_flag'] = (
    (results['slope'] < 0) &
    (results['p_value'] < P_VALUE_THRESHOLD)
)


# -------------------------------------------------------
# 8. 90-day revenue at risk
#
#    N = orders in 90 days = floor(90 / avg_gap)
#    Items at risk = |slope| * N*(N+1)/2  [triangular sum]
#    — this sums the cumulative basket decline across all N
#    future orders, not just the final endpoint.
#    Revenue proxy: items × $3.50
# -------------------------------------------------------
N = (90.0 / results['avg_days_between_orders']).clip(lower=0)
results['orders_in_90d'] = N

items_at_risk = np.where(
    results['shrinking_flag'],
    np.abs(results['slope']) * N * (N + 1) / 2,
    0.0
)
results['items_at_risk_90d']   = np.maximum(0, items_at_risk)
results['revenue_at_risk_90d'] = results['items_at_risk_90d'] * AVG_ITEM_PRICE
results['projected_basket_90d'] = np.maximum(
    0,
    results['last_basket_size'] + results['slope'] * N
)


# -------------------------------------------------------
# 9. Print findings
# -------------------------------------------------------
n_shrinking   = results['shrinking_flag'].sum()
pct_shrinking = n_shrinking / len(results) * 100
total_risk    = results['revenue_at_risk_90d'].sum()
avg_vel       = results.loc[results['shrinking_flag'], 'shrinkage_velocity'].mean()

print(f"\nFound {n_shrinking:,} customers in real decline — not just random noise")
print(f"That's {pct_shrinking:.1f}% of customers with sufficient order history")
print(f"Projected 90-day revenue at risk: ${total_risk:,.0f}")
print(f"Avg shrinkage velocity (shrinking customers): {avg_vel:.3f}")

print("\nSlope distribution (all eligible customers):")
print(f"  Mean:   {slopes.mean():.3f} items/order")
print(f"  Median: {np.median(slopes):.3f}")
print(f"  Min:    {slopes.min():.3f}  (fastest decline)")
print(f"  Max:    {slopes.max():.3f}  (fastest growth)")

print(f"\nP-value distribution:")
pct_sig = (p_values < P_VALUE_THRESHOLD).mean() * 100
print(f"  {pct_sig:.1f}% of customers have significant trend (p < {P_VALUE_THRESHOLD})")
sig_neg = ((p_values < P_VALUE_THRESHOLD) & (slopes < 0)).sum()
sig_pos = ((p_values < P_VALUE_THRESHOLD) & (slopes > 0)).sum()
print(f"  Significant negative: {sig_neg:,}  |  Significant positive: {sig_pos:,}")


# -------------------------------------------------------
# 10. Save customer_features.csv
# -------------------------------------------------------
print(f"\nSaving to {config.CUSTOMER_FEATURES}...")
os.makedirs(config.PROCESSED_DIR, exist_ok=True)
results.to_csv(config.CUSTOMER_FEATURES, index=False)
print(f"  Saved: {len(results):,} customers, {results.shape[1]} columns")


# -------------------------------------------------------
# 11. Chart — top 20 by revenue at risk
#    Horizontal bar, colored by shrinkage velocity.
#    Darker = faster decline. Sorted biggest-risk at top.
# -------------------------------------------------------
print("\nBuilding top-20 chart...")

shrinking = results[results['shrinking_flag']].copy()
top20 = shrinking.nlargest(20, 'revenue_at_risk_90d').reset_index(drop=True)

vel_vals = top20['shrinkage_velocity'].abs()
norm     = plt.Normalize(vel_vals.min(), vel_vals.max())
colors   = plt.cm.YlOrRd(norm(vel_vals.values))

fig, ax = plt.subplots(figsize=(11, 7))

bars = ax.barh(
    range(len(top20)),
    top20['revenue_at_risk_90d'],
    color=colors, edgecolor='white', linewidth=0.4
)

# Label each bar with dollar value and slope
max_val = top20['revenue_at_risk_90d'].max()
for i, (bar, row) in enumerate(zip(bars, top20.itertuples())):
    ax.text(
        bar.get_width() + max_val * 0.01,
        bar.get_y() + bar.get_height() / 2,
        f"${row.revenue_at_risk_90d:,.0f}   slope {row.slope:.2f}   p={row.p_value:.3f}",
        va='center', fontsize=7.5, color='#2c3e50'
    )

ax.set_yticks(range(len(top20)))
ax.set_yticklabels([f"Customer {uid}" for uid in top20['user_id']], fontsize=8)
ax.invert_yaxis()

ax.set_xlabel("Projected Revenue at Risk — 90 Days  ($)", fontsize=11, labelpad=8)
ax.set_title(
    "Top 20 Customers by Basket Shrinkage Risk\n"
    "Color intensity = shrinkage velocity  (darker = faster decline)",
    fontsize=12, fontweight='bold', pad=14, color='#1a252f'
)

ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${int(x):,}"))
ax.set_xlim(0, max_val * 1.35)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(axis='x', linestyle='--', alpha=0.35)

plt.tight_layout()
os.makedirs(config.CHARTS_DIR, exist_ok=True)
plt.savefig(config.SHRINKAGE_CHART, dpi=config.CHART_DPI, bbox_inches='tight')
plt.close()
print(f"  Chart saved: {config.SHRINKAGE_CHART}")


# -------------------------------------------------------
# 12. Wrap up
# -------------------------------------------------------
print("\n" + "=" * 60)
print("SHRINKAGE DETECTION COMPLETE")
print("=" * 60)
print(f"  {n_shrinking:,} customers with statistically significant basket decline")
print(f"  ${total_risk:,.0f} projected 90-day revenue at risk")
print(f"  Top customer at risk: ${top20['revenue_at_risk_90d'].iloc[0]:,.0f}")
print()
print("Next up — run pipeline/04_intervention.py")
