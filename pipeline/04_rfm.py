# ============================================================
# 02_rfm.py — RFM Segmentation
# ============================================================
# Score every customer on Recency, Frequency, and Monetary
# value, then slot them into one of four business segments:
# Champions, Loyal, At Risk, Lost.
#
# Quick note on what "Monetary" means here — Instacart has
# no price data in the public dataset, so we use total items
# purchased as the spend proxy. It's not perfect but it
# correlates tightly with actual spend (big baskets = big
# tickets) and it's what we have to work with.
#
# Recency uses days_since_prior_order at the customer's last
# recorded order — lower gap = shopping more recently = higher
# score. Instacart caps this column at 30 days.
# ============================================================

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')   # no display needed — we're just writing a PNG
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


# -------------------------------------------------------
# 1. Load master_orders — only the 4 columns we need.
#    The string columns (product_name, department, aisle)
#    are what makes the file 3GB — skip them here.
#    Basket size comes from counting rows per order_id
#    before we collapse to order level.
# -------------------------------------------------------
print("=" * 60)
print("Instacart Basket Shrinkage — 02 RFM Segmentation")
print("=" * 60)

if not os.path.exists(config.MASTER_ORDERS):
    print(f"\n  ERROR: {config.MASTER_ORDERS} not found.")
    print("  Run pipeline/01_load_explore.py first.")
    sys.exit(1)

print("\nLoading master_orders.csv (numeric columns only, skipping strings)...")

dtype_map = {
    'user_id':               'int32',
    'order_id':              'int32',
    'order_number':          'int16',
}

master = pd.read_csv(
    config.MASTER_ORDERS,
    usecols=['user_id', 'order_id', 'order_number', 'days_since_prior_order'],
    dtype=dtype_map
)

print(f"  {len(master):,} rows | "
      f"{master['user_id'].nunique():,} customers | "
      f"{master['order_id'].nunique():,} orders")
print(f"  Memory footprint: {master.memory_usage(deep=True).sum() / 1e6:.0f} MB")


# -------------------------------------------------------
# 2. Basket size — count rows per order_id now, while we
#    still have one row per (order, product). We'll use
#    this as the Monetary input.
# -------------------------------------------------------
print("\nCounting items per order (for Monetary dimension)...")
basket_sizes = master.groupby('order_id').size().reset_index(name='basket_size')


# -------------------------------------------------------
# 3. Collapse to order level — one row per (user, order).
#    We need this for recency and frequency calculations.
# -------------------------------------------------------
orders = (master
          .drop_duplicates(subset=['order_id'])
          [['user_id', 'order_id', 'order_number', 'days_since_prior_order']]
          .copy())

orders = orders.merge(basket_sizes, on='order_id')

del master   # done with the product-level detail, free the memory
print(f"  Order-level table: {len(orders):,} orders")


# -------------------------------------------------------
# 4. Compute the three raw RFM metrics.
#
#    RECENCY  — days_since_prior_order at the customer's
#               most recent order (max order_number).
#               NaN = single-order customer → fill with 30.
#
#    FREQUENCY — count of unique orders. Straightforward.
#
#    MONETARY  — sum of basket_size across all orders.
#                Captures whether someone is a big-basket
#                shopper or a grab-one-thing-and-leave type.
# -------------------------------------------------------
print("\nCalculating Recency, Frequency, Monetary...")

# Sort once so .last() gives us the most recent order per customer
orders_sorted = orders.sort_values(['user_id', 'order_number'])

recency = (orders_sorted
           .groupby('user_id')['days_since_prior_order']
           .last()
           .reset_index()
           .rename(columns={'days_since_prior_order': 'recency_days'}))

# Single-order customers have NaN (no prior order to measure from).
# Cap at 30 — same as Instacart's own ceiling — so they land in the
# "least recent" bucket without distorting the distribution.
recency['recency_days'] = recency['recency_days'].fillna(30).astype(float)

frequency = (orders
             .groupby('user_id')['order_id']
             .nunique()
             .reset_index()
             .rename(columns={'order_id': 'frequency'}))

monetary = (orders
            .groupby('user_id')['basket_size']
            .sum()
            .reset_index()
            .rename(columns={'basket_size': 'monetary'}))

rfm = (recency
       .merge(frequency, on='user_id')
       .merge(monetary,  on='user_id'))

print(f"  Customers in RFM table: {len(rfm):,}")
print(f"  Recency range:   {rfm['recency_days'].min():.0f}–{rfm['recency_days'].max():.0f} days")
print(f"  Frequency range: {rfm['frequency'].min()}–{rfm['frequency'].max()} orders")
print(f"  Monetary range:  {rfm['monetary'].min():,}–{rfm['monetary'].max():,} items")


# -------------------------------------------------------
# 5. Score each dimension 1–4.
#
#    Approach: rank(pct=True) converts raw values into
#    uniform percentiles, then pd.cut splits at 25/50/75.
#    This handles ties gracefully — qcut can error when
#    too many customers share the same value (common here
#    since Instacart caps days_since_prior_order at 30).
#
#    Recency is inverted: lower days = more recent = 4.
#    Frequency and Monetary are natural: higher = 4.
# -------------------------------------------------------
print("\nScoring dimensions 1–4...")

QUARTILE_BINS = [0, 0.25, 0.50, 0.75, 1.0]

rfm['r_score'] = pd.cut(
    rfm['recency_days'].rank(pct=True, method='average'),
    bins=QUARTILE_BINS,
    labels=[4, 3, 2, 1],   # inverted — low gap → 4, high gap → 1
    include_lowest=True
).astype(int)

rfm['f_score'] = pd.cut(
    rfm['frequency'].rank(pct=True, method='average'),
    bins=QUARTILE_BINS,
    labels=[1, 2, 3, 4],
    include_lowest=True
).astype(int)

rfm['m_score'] = pd.cut(
    rfm['monetary'].rank(pct=True, method='average'),
    bins=QUARTILE_BINS,
    labels=[1, 2, 3, 4],
    include_lowest=True
).astype(int)

rfm['rfm_score'] = (rfm['r_score'].astype(str)
                    + rfm['f_score'].astype(str)
                    + rfm['m_score'].astype(str))

# Quick sanity check — each score should be roughly 25% each
print("  Score distributions (expect ~25% per bucket):")
for col in ['r_score', 'f_score', 'm_score']:
    dist = rfm[col].value_counts(normalize=True).sort_index()
    line = "  ".join(f"{k}={v:.0%}" for k, v in dist.items())
    print(f"    {col}:  {line}")


# -------------------------------------------------------
# 6. Assign business segments.
#
#    Champions — recent, frequent, heavy baskets.
#                These are the customers we cannot afford
#                to lose. Any shrinkage here is a red flag.
#
#    Loyal     — recent and reasonably frequent. The stable
#                middle — they show up, they buy, they leave.
#
#    At Risk   — used to order a lot but haven't recently.
#                Classic pre-churn pattern. Good intervention
#                target if caught early enough.
#
#    Lost      — low engagement overall. Either one-timers
#                or customers who've already left quietly.
#
#    np.select checks conditions in order and takes the
#    first match — Champions condition must come before
#    Loyal since Champions also satisfy the Loyal rule.
# -------------------------------------------------------
print("\nAssigning segments...")

conditions = [
    (rfm['r_score'] >= 3) & (rfm['f_score'] >= 3) & (rfm['m_score'] >= 3),  # Champions
    (rfm['r_score'] >= 3) & (rfm['f_score'] >= 2),                           # Loyal
    (rfm['r_score'] <= 2) & (rfm['f_score'] >= 3),                           # At Risk
]
choices = ['Champions', 'Loyal', 'At Risk']
rfm['segment'] = np.select(conditions, choices, default='Lost')

total = len(rfm)
seg_counts = rfm['segment'].value_counts()
seg_pcts   = rfm['segment'].value_counts(normalize=True) * 100

print("\n  Segment breakdown:")
for seg in ['Champions', 'Loyal', 'At Risk', 'Lost']:
    n   = seg_counts.get(seg, 0)
    pct = seg_pcts.get(seg, 0)
    print(f"    {seg:<12}  {n:>7,}  ({pct:.1f}%)")

# Average raw metrics by segment — good reality check that
# the scoring logic produced sensible separations
print("\n  Average raw metrics per segment:")
seg_stats = (rfm.groupby('segment')[['recency_days', 'frequency', 'monetary']]
               .mean()
               .reindex(['Champions', 'Loyal', 'At Risk', 'Lost'])
               .round(1))
print(seg_stats.to_string())


# -------------------------------------------------------
# 7. Save rfm_segments.csv
# -------------------------------------------------------
print(f"\nSaving to {config.RFM_SEGMENTS}...")
os.makedirs(config.PROCESSED_DIR, exist_ok=True)
rfm.to_csv(config.RFM_SEGMENTS, index=False)
print(f"  Done — {len(rfm):,} customers, {rfm.shape[1]} columns")


# -------------------------------------------------------
# 8. Bar chart — segment distribution.
#    Sorted Champions → Loyal → At Risk → Lost.
#    Each bar labeled with count and share of base.
#    Clean, minimal, presentation-ready.
# -------------------------------------------------------
print("\nBuilding segment distribution chart...")

SEGMENT_ORDER  = ['Champions', 'Loyal', 'At Risk', 'Lost']
SEGMENT_COLORS = {
    'Champions': '#27ae60',
    'Loyal':     '#2980b9',
    'At Risk':   '#e67e22',
    'Lost':      '#c0392b',
}

bar_heights = [seg_counts.get(s, 0) for s in SEGMENT_ORDER]
bar_colors  = [SEGMENT_COLORS[s] for s in SEGMENT_ORDER]

fig, ax = plt.subplots(figsize=(9, 5.5))

bars = ax.bar(
    SEGMENT_ORDER, bar_heights,
    color=bar_colors, edgecolor='white', linewidth=0.8, zorder=3
)

# Annotate each bar — count on top, percentage below it
for bar, height in zip(bars, bar_heights):
    pct = height / total * 100
    x   = bar.get_x() + bar.get_width() / 2
    ax.text(x, height + total * 0.003, f"{height:,}",
            ha='center', va='bottom', fontsize=10.5,
            fontweight='bold', color='#1a252f')
    ax.text(x, height + total * 0.003 + max(bar_heights) * 0.03,
            f"({pct:.1f}%)",
            ha='center', va='bottom', fontsize=9, color='#555')

ax.set_title(
    "Customer RFM Segments  ·  Instacart Purchase History",
    fontsize=13, fontweight='bold', pad=16, color='#1a252f'
)
ax.set_xlabel("Segment", fontsize=11, labelpad=8, color='#333')
ax.set_ylabel("Number of Customers", fontsize=11, labelpad=8, color='#333')

ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
ax.set_ylim(0, max(bar_heights) * 1.22)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(axis='y', linestyle='--', alpha=0.35, zorder=0)
ax.tick_params(labelsize=10)

plt.tight_layout()
os.makedirs(config.CHARTS_DIR, exist_ok=True)
plt.savefig(config.RFM_CHART, dpi=config.CHART_DPI, bbox_inches='tight')
plt.close()

print(f"  Chart saved: {config.RFM_CHART}")


# -------------------------------------------------------
# 9. Wrap up
# -------------------------------------------------------
at_risk_n   = seg_counts.get('At Risk', 0)
at_risk_pct = seg_pcts.get('At Risk', 0)
champ_n     = seg_counts.get('Champions', 0)

print("\n" + "=" * 60)
print("RFM COMPLETE")
print("=" * 60)
print(f"  {total:,} customers scored and segmented")
print(f"  {champ_n:,} Champions — the revenue base we're protecting")
print(f"  {at_risk_n:,} At Risk ({at_risk_pct:.1f}%) — frequent buyers who've gone quiet")
print(f"  Segment chart saved to outputs/charts/")
print()
print("Next up — run pipeline/03_shrinkage.py")
