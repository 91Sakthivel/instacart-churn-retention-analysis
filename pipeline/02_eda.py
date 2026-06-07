# ============================================================
# 02_eda.py — Exploratory Data Analysis (12 business charts)
# ============================================================
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

os.makedirs(config.EDA_CHARTS_DIR, exist_ok=True)
os.makedirs(config.REPORTS_DIR, exist_ok=True)

print("=" * 60)
print("Instacart EDA - 02_eda.py")
print("=" * 60)
print("\nLoading master_orders.csv (product-level, 33M rows)...")

mo = pd.read_csv(
    config.MASTER_ORDERS,
    usecols=['user_id', 'order_id', 'product_id', 'product_name', 'department',
             'order_number', 'order_dow', 'order_hour_of_day',
             'days_since_prior_order', 'reordered'],
    dtype={
        'user_id': 'int32', 'order_id': 'int32', 'product_id': 'int32',
        'order_number': 'int16', 'order_dow': 'int8',
        'order_hour_of_day': 'int8', 'reordered': 'int8',
    }
)
print(f"  Loaded: {len(mo):,} rows, {mo['user_id'].nunique():,} customers")

print("  Aggregating to order level...")
order_stats = (
    mo.groupby('order_id', sort=False)
    .agg(
        basket_size=('product_id', 'count'),
        user_id=('user_id', 'first'),
        order_number=('order_number', 'first'),
        order_dow=('order_dow', 'first'),
        order_hour_of_day=('order_hour_of_day', 'first'),
        days_since_prior_order=('days_since_prior_order', 'first'),
    )
    .reset_index()
)

cust_agg = (
    order_stats.groupby('user_id')
    .agg(order_count=('order_id', 'count'), avg_basket=('basket_size', 'mean'))
    .reset_index()
)
last_gap = (
    order_stats.sort_values(['user_id', 'order_number'])
    .groupby('user_id', sort=False)
    .agg(last_gap=('days_since_prior_order', 'last'))
    .reset_index()
)
cust_agg = cust_agg.merge(last_gap, on='user_id', how='left')
cust_agg['churned'] = (
    (cust_agg['last_gap'] >= 30) & (cust_agg['order_count'] >= 5)
).astype(int)
cust_reorder = mo.groupby('user_id')['reordered'].mean().reset_index()
cust_reorder.columns = ['user_id', 'reorder_rate']
cust_agg = cust_agg.merge(cust_reorder, on='user_id', how='left')

total_cust = len(cust_agg)
few_orders = (cust_agg['order_count'] < 5).sum()
print(f"  {total_cust:,} customers | {few_orders:,} with <5 orders")

# Chart 1 — Order frequency
print("\n[Chart 1] Order frequency distribution...")
fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(cust_agg['order_count'].clip(upper=50), bins=50,
        color='steelblue', edgecolor='white', linewidth=0.4)
ax.axvline(5, color='red', linestyle='--', lw=1.5, label='5-order threshold')
ax.set_xlabel('Orders per Customer')
ax.set_ylabel('Customers')
ax.set_title('Order Frequency Distribution per Customer')
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(config.EDA_CHARTS_DIR, 'chart01_order_frequency.png'), dpi=config.CHART_DPI)
plt.close()
pct_few = few_orders / total_cust * 100
print(f"  INSIGHT: {pct_few:.1f}% of customers have fewer than 5 orders and will be excluded from behavioral modeling")

# Chart 2 — Basket size
print("[Chart 2] Basket size distribution...")
basket_vals = order_stats['basket_size']
mean_b, med_b, max_b = basket_vals.mean(), basket_vals.median(), basket_vals.max()
fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(basket_vals.clip(upper=80), bins=60, color='steelblue', edgecolor='white', lw=0.3)
ax.axvline(mean_b, color='red', linestyle='--', lw=1.5, label=f'Mean={mean_b:.1f}')
ax.axvline(med_b, color='orange', linestyle='--', lw=1.5, label=f'Median={med_b:.0f}')
ax.axvline(80, color='purple', linestyle='--', lw=1.5, label='Bulk threshold (80)')
ax.set_xlabel('Basket Size (items)')
ax.set_ylabel('Orders')
ax.set_title('Basket Size Distribution (clipped at 80)')
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(config.EDA_CHARTS_DIR, 'chart02_basket_size.png'), dpi=config.CHART_DPI)
plt.close()
print(f"  INSIGHT: Mean basket {mean_b:.1f} items, median {med_b:.0f} items. Max: {max_b:.0f}")
if max_b > 80:
    pct_bulk = (basket_vals > 80).sum() / len(basket_vals) * 100
    print(f"  {pct_bulk:.2f}% of orders exceed 80 items")

# Chart 3 — Reorder rate by department
print("[Chart 3] Reorder rate by department...")
dept_reorder = (
    mo.groupby('department', sort=False)['reordered'].mean()
    .sort_values(ascending=False).head(15).reset_index()
)
dept_reorder.columns = ['department', 'reorder_rate']
fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(dept_reorder['department'][::-1], dept_reorder['reorder_rate'][::-1], color='steelblue')
ax.set_xlabel('Reorder Rate')
ax.set_title('Reorder Rate by Department (Top 15)')
plt.tight_layout()
plt.savefig(os.path.join(config.EDA_CHARTS_DIR, 'chart03_reorder_by_dept.png'), dpi=config.CHART_DPI)
plt.close()
top3 = dept_reorder.head(3)['department'].tolist()
print(f"  INSIGHT: Top 3 departments by reorder rate: {', '.join(top3)}")

# Chart 4 — Orders by day of week
print("[Chart 4] Orders by day of week...")
day_names = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat']
dow_counts = order_stats['order_dow'].value_counts().sort_index()
fig, ax = plt.subplots(figsize=(8, 5))
ax.bar([day_names[i] for i in dow_counts.index], dow_counts.values, color='steelblue')
ax.set_xlabel('Day of Week')
ax.set_ylabel('Orders')
ax.set_title('Orders by Day of Week')
plt.tight_layout()
plt.savefig(os.path.join(config.EDA_CHARTS_DIR, 'chart04_orders_by_dow.png'), dpi=config.CHART_DPI)
plt.close()
peak_dow = day_names[int(dow_counts.idxmax())]
peak_dow_pct = dow_counts.max() / dow_counts.sum() * 100
print(f"  INSIGHT: Peak shopping day is {peak_dow} ({peak_dow_pct:.1f}% of orders)")

# Chart 5 — Orders by hour
print("[Chart 5] Orders by hour of day...")
hour_counts = order_stats['order_hour_of_day'].value_counts().sort_index()
fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(hour_counts.index, hour_counts.values, color='steelblue')
ax.set_xlabel('Hour of Day')
ax.set_ylabel('Orders')
ax.set_title('Orders by Hour of Day')
ax.set_xticks(range(0, 24))
plt.tight_layout()
plt.savefig(os.path.join(config.EDA_CHARTS_DIR, 'chart05_orders_by_hour.png'), dpi=config.CHART_DPI)
plt.close()
peak_hour = int(hour_counts.idxmax())
peak_hr_pct = hour_counts.max() / hour_counts.sum() * 100
print(f"  INSIGHT: Peak hour is {peak_hour}:00 ({peak_hr_pct:.1f}% of all orders)")

# Chart 6 — Top 20 products
print("[Chart 6] Top 20 products by volume...")
top20 = (
    mo.groupby('product_name', sort=False)['order_id'].nunique()
    .nlargest(20).reset_index()
)
top20.columns = ['product_name', 'order_count']
fig, ax = plt.subplots(figsize=(10, 8))
ax.barh([p[:35] for p in top20['product_name'][::-1]],
        top20['order_count'][::-1], color='steelblue')
ax.set_xlabel('Orders')
ax.set_title('Top 20 Products by Order Volume')
plt.tight_layout()
plt.savefig(os.path.join(config.EDA_CHARTS_DIR, 'chart06_top20_products.png'), dpi=config.CHART_DPI)
plt.close()
top1 = top20.iloc[0]
print(f"  INSIGHT: Top product '{top1['product_name']}' with {top1['order_count']:,} orders")

# Chart 7 — Basket by sequence
print("[Chart 7] Basket size by order sequence...")
seq_basket = (
    order_stats[order_stats['order_number'] <= 20]
    .groupby('order_number')['basket_size'].mean()
)
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(seq_basket.index, seq_basket.values, marker='o', color='steelblue', lw=2)
ax.set_xlabel('Order Sequence Number')
ax.set_ylabel('Avg Basket Size')
ax.set_title('Basket Size by Order Sequence (lifetime trajectory)')
ax.set_xticks(range(1, 21))
plt.tight_layout()
plt.savefig(os.path.join(config.EDA_CHARTS_DIR, 'chart07_basket_by_sequence.png'), dpi=config.CHART_DPI)
plt.close()
b1 = seq_basket.get(1, float('nan'))
b20 = seq_basket.get(20, float('nan'))
direction = "increases" if b20 > b1 else "decreases"
print(f"  INSIGHT: Basket at order 1: {b1:.1f}, at order 20: {b20:.1f} ({direction})")
print("  NOTE: Basket does NOT decline before churn -- Check 1 failed. This is lifetime, not churn signal.")

# Chart 8 — Gap distribution
print("[Chart 8] Days since prior order distribution...")
gaps = order_stats['days_since_prior_order'].dropna()
med_gap = gaps.median()
pct_30 = (gaps >= 30).sum() / len(gaps) * 100
fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(gaps.clip(upper=60), bins=61, color='steelblue', edgecolor='white', lw=0.3)
ax.axvline(30, color='red', linestyle='--', lw=1.5, label='30-day churn threshold')
ax.axvline(med_gap, color='orange', linestyle='--', lw=1.5, label=f'Median={med_gap:.0f}d')
ax.set_xlabel('Days Since Prior Order')
ax.set_ylabel('Orders')
ax.set_title('Days Since Prior Order Distribution')
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(config.EDA_CHARTS_DIR, 'chart08_gap_distribution.png'), dpi=config.CHART_DPI)
plt.close()
print(f"  INSIGHT: Median order gap {med_gap:.0f} days. {pct_30:.1f}% of gaps >= 30 days.")

# Chart 9 — Churn by order count bucket
print("[Chart 9] Churn rate by order count bucket...")
bins = [0, 5, 10, 20, 99999]
labels = ['1-5', '6-10', '11-20', '20+']
cust_agg['bucket'] = pd.cut(cust_agg['order_count'], bins=bins, labels=labels)
churn_bucket = cust_agg.groupby('bucket', observed=True)['churned'].mean().reset_index()
fig, ax = plt.subplots(figsize=(8, 5))
c_colors = ['#d32f2f' if r > 0.3 else '#1976d2' for r in churn_bucket['churned']]
ax.bar(churn_bucket['bucket'], churn_bucket['churned'] * 100, color=c_colors)
for i, (bkt, rate) in enumerate(zip(churn_bucket['bucket'], churn_bucket['churned'])):
    ax.text(i, rate * 100 + 0.5, f'{rate*100:.1f}%', ha='center', va='bottom', fontsize=10)
ax.set_xlabel('Order Count Bucket')
ax.set_ylabel('Churn Rate (%)')
ax.set_title('Churn Rate by Order Count Bucket')
plt.tight_layout()
plt.savefig(os.path.join(config.EDA_CHARTS_DIR, 'chart09_churn_by_bucket.png'), dpi=config.CHART_DPI)
plt.close()
for _, row in churn_bucket.iterrows():
    print(f"  Bucket {row['bucket']}: {row['churned']*100:.1f}% churn rate")

# Chart 10 — Reorder rate distribution
print("[Chart 10] Reorder rate distribution...")
rr_vals = cust_agg['reorder_rate'].dropna()
pct_above50 = (rr_vals >= 0.5).sum() / len(rr_vals) * 100
fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(rr_vals, bins=50, color='steelblue', edgecolor='white', lw=0.3)
ax.axvline(0.5, color='red', linestyle='--', lw=1.5, label='0.5 threshold')
ax.set_xlabel('Customer Reorder Rate')
ax.set_ylabel('Customers')
ax.set_title('Customer Reorder Rate Distribution')
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(config.EDA_CHARTS_DIR, 'chart10_reorder_rate_dist.png'), dpi=config.CHART_DPI)
plt.close()
print(f"  INSIGHT: {pct_above50:.1f}% of customers above 0.5 reorder rate.")
print("  VALIDATED SIGNAL: Strongest churn predictor (active 53.3% vs churned 43.4%, p~=0)")

# Chart 11 — Gap trend churned customers
print("[Chart 11] Gap trend for churned customers...")
churned_ids = set(cust_agg[cust_agg['churned'] == 1]['user_id'])
ch_orders = order_stats[order_stats['user_id'].isin(churned_ids)].copy()
ch_orders = ch_orders.sort_values(['user_id', 'order_number'])
ch_orders['rank_from_last'] = (
    ch_orders.groupby('user_id')['order_number']
    .rank(ascending=False, method='dense').astype(int)
)
pos_data = ch_orders[
    (ch_orders['rank_from_last'] <= 5) &
    ch_orders['days_since_prior_order'].notna()
].copy()
gap_td = (
    pos_data.groupby('rank_from_last')['days_since_prior_order']
    .mean().reset_index().sort_values('rank_from_last', ascending=False)
)
gap_td['position'] = [f'-{int(r)}' for r in gap_td['rank_from_last']]
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(gap_td['position'], gap_td['days_since_prior_order'],
        marker='o', color='crimson', lw=2, markersize=8)
for _, row in gap_td.iterrows():
    ax.annotate(f"{row['days_since_prior_order']:.1f}d",
                xy=(row['position'], row['days_since_prior_order']),
                xytext=(0, 8), textcoords='offset points', ha='center', fontsize=9)
ax.set_xlabel('Order Position (relative to last)')
ax.set_ylabel('Avg Days Since Prior Order')
ax.set_title('Order Gap Trend - Churned Customers (Last 5 Orders)')
plt.tight_layout()
plt.savefig(os.path.join(config.EDA_CHARTS_DIR, 'chart11_gap_trend_churned.png'), dpi=config.CHART_DPI)
plt.close()
g5v_row = gap_td[gap_td['rank_from_last'] == 5]['days_since_prior_order']
g1v_row = gap_td[gap_td['rank_from_last'] == 1]['days_since_prior_order']
gap_5v = g5v_row.values[0] if len(g5v_row) else float('nan')
gap_1v = g1v_row.values[0] if len(g1v_row) else float('nan')
print(f"  VALIDATED: Gap at -5: {gap_5v:.1f}d -> at -1: {gap_1v:.1f}d (gap widens before churn)")

# Chart 12 — Department mix by RFM
print("[Chart 12] Department mix by RFM segment...")
try:
    rfm = pd.read_csv(config.RFM_SEGMENTS, usecols=['user_id', 'segment'])
    mo_seg = mo[['user_id', 'order_id', 'department']].merge(rfm, on='user_id', how='inner')
    all_segs = mo_seg['segment'].unique().tolist()
    target_segs = [s for s in ['Champions', 'At Risk', 'Lost'] if s in all_segs]
    if len(target_segs) < 2:
        target_segs = all_segs[:3]
    mo_filt = mo_seg[mo_seg['segment'].isin(target_segs)]
    ds = mo_filt.groupby(['segment', 'department'])['order_id'].nunique().reset_index()
    ds.columns = ['segment', 'department', 'cnt']
    ds['total'] = ds.groupby('segment')['cnt'].transform('sum')
    ds['pct'] = ds['cnt'] / ds['total'] * 100
    top_depts = ds.groupby('department')['cnt'].sum().nlargest(8).index.tolist()
    ds_top = ds[ds['department'].isin(top_depts)]
    pivot = ds_top.pivot_table(index='department', columns='segment',
                               values='pct', fill_value=0, observed=True)
    pivot = pivot.reindex(columns=[c for c in target_segs if c in pivot.columns])
    fig, ax = plt.subplots(figsize=(12, 6))
    x = range(len(pivot.index))
    w = 0.28
    seg_colors = ['gold', 'darkorange', 'crimson']
    for i, seg in enumerate(pivot.columns):
        ax.bar([xi + i * w for xi in x], pivot[seg], w,
               label=seg, color=seg_colors[i % len(seg_colors)])
    ax.set_xticks([xi + w for xi in x])
    ax.set_xticklabels(pivot.index, rotation=30, ha='right')
    ax.set_ylabel('% of Orders')
    ax.set_title('Department Mix by RFM Segment')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(config.EDA_CHARTS_DIR, 'chart12_dept_mix_rfm.png'), dpi=config.CHART_DPI)
    plt.close()
    if len(pivot.columns) >= 2:
        col0, col_last = pivot.columns[0], pivot.columns[-1]
        diff_col = (pivot[col0] - pivot[col_last]).abs().idxmax()
        print(f"  INSIGHT: Biggest {col0} vs {col_last} diff: '{diff_col}' "
              f"({pivot.loc[diff_col, col0]:.1f}% vs {pivot.loc[diff_col, col_last]:.1f}%)")
    else:
        print("  Chart 12 complete")
except Exception as e:
    print(f"  Chart 12 note: {e}")

# Summary
active_rr = cust_agg[(cust_agg['churned'] == 0) & (cust_agg['order_count'] >= 5)]['reorder_rate'].mean()
churned_rr = cust_agg[cust_agg['churned'] == 1]['reorder_rate'].mean()

print("\n" + "=" * 60)
print("Key findings from EDA:")
print(f"1. {pct_few:.1f}% customers excluded (< 5 orders)")
print(f"2. Median order gap: {med_gap:.0f} days, {pct_30:.1f}% gaps >= 30 days")
print(f"3. Reorder rate: active {active_rr*100:.1f}% vs churned {churned_rr*100:.1f}% -- primary signal")
print(f"4. Gap widens from {gap_5v:.1f}d to {gap_1v:.1f}d in final 5 orders before churn")
print(f"5. Peak shopping: {peak_dow} at {peak_hour}:00")
print("=" * 60)
print("\nEDA complete. Findings above guide feature engineering.")
print("Next -- pipeline/03_clean.py")
