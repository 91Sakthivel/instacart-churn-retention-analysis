# ============================================================
# 11_cohort_analysis.py — Customer Cohort Retention Analysis
# ============================================================
# Cohort definition: ordering frequency (avg_days_between_orders)
#   — faster orderers are more loyal; this heatmap quantifies that
# Retention at order N: % of each cohort that reached N total orders
# Output: streamlit/cohort_heatmap.png
# ============================================================
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

STREAMLIT_DIR = os.path.join(config.BASE_DIR, 'streamlit')
OUT_PATH      = os.path.join(STREAMLIT_DIR, 'cohort_heatmap.png')

print("=" * 60)
print("Cohort Retention Analysis - 11_cohort_analysis.py")
print("=" * 60)

# ----------------------------------------------------------------
# 1. Load data and confirm column names
# ----------------------------------------------------------------
print(f"\nLoading {config.CUSTOMER_FEATURES}...")
cf = pd.read_csv(
    config.CUSTOMER_FEATURES,
    usecols=['user_id', 'order_count', 'avg_days_between_orders', 'churned'],
)
print(f"  Loaded: {len(cf):,} customers")
print(f"  Columns used: {list(cf.columns)}")
print(f"  order_count  range: {cf['order_count'].min():.0f} – {cf['order_count'].max():.0f}")
print(f"  avg_days range:     {cf['avg_days_between_orders'].min():.1f} – "
      f"{cf['avg_days_between_orders'].max():.1f} days")

# Drop the tiny fraction of rows with NaN in either signal
cf = cf.dropna(subset=['order_count', 'avg_days_between_orders']).copy()
print(f"  After NaN drop: {len(cf):,} customers")

# ----------------------------------------------------------------
# 2. Define cohorts from ordering frequency
#    (avg_days_between_orders → how often does this customer shop?)
# ----------------------------------------------------------------
bins   = [0, 7, 14, 21, float('inf')]
labels = ['<=7 days\n(Very Frequent)', '8-14 days\n(Frequent)',
          '15-21 days\n(Moderate)', '>21 days\n(Infrequent)']

cf['cohort'] = pd.cut(
    cf['avg_days_between_orders'],
    bins=bins,
    labels=labels,
    right=True,
)

print("\nCohort distribution (ordering frequency):")
cohort_counts = cf['cohort'].value_counts().sort_index()
for label in labels:
    n    = int(cohort_counts.get(label, 0))
    pct  = n / len(cf) * 100
    churn = cf.loc[cf['cohort'] == label, 'churned'].mean() * 100
    print(f"  {label.replace(chr(10), ' '):<30} {n:>7,} customers "
          f"({pct:>4.1f}%)   churn rate: {churn:>4.1f}%")

# ----------------------------------------------------------------
# 3. Build retention matrix
#    cell[cohort, N] = % of cohort that placed >= N total orders
# ----------------------------------------------------------------
ORDER_SEQUENCES = list(range(1, 11))
col_names = [f'Order {n}' for n in ORDER_SEQUENCES]

retention_rows = {}
for label in labels:
    mask = cf['cohort'] == label
    group = cf.loc[mask, 'order_count']
    if len(group) == 0:
        retention_rows[label] = [0.0] * len(ORDER_SEQUENCES)
    else:
        retention_rows[label] = [
            (group >= n).mean() * 100 for n in ORDER_SEQUENCES
        ]

retention_df = pd.DataFrame.from_dict(
    retention_rows, orient='index', columns=col_names
)
retention_df.index.name = 'Cohort (Avg Days Between Orders)'

print("\nRetention matrix (%):")
print(retention_df.round(1).to_string())

# ----------------------------------------------------------------
# 4. Seaborn heatmap with percentage annotations
# ----------------------------------------------------------------
# Build annotation array: "85.3%"
annot_arr = retention_df.map(lambda v: f'{v:.1f}%')

fig, ax = plt.subplots(figsize=(13, 5))

sns.heatmap(
    retention_df,
    annot=annot_arr,
    fmt='',
    cmap='Blues',
    vmin=0,
    vmax=100,
    linewidths=0.6,
    linecolor='#e0e0e0',
    ax=ax,
    annot_kws={'size': 9, 'weight': 'normal'},
    cbar_kws={'label': 'Retention Rate (%)', 'shrink': 0.8},
)

# --- Labels ---
ax.set_title(
    'Customer Cohort Retention by Order Sequence\n'
    'Cohorts defined by ordering frequency (avg days between orders)',
    fontsize=13, fontweight='bold', pad=14,
)
ax.set_xlabel('Order Sequence Number', fontsize=11, labelpad=10)
ax.set_ylabel('')
ax.tick_params(axis='x', rotation=0, labelsize=9)
ax.tick_params(axis='y', rotation=0, labelsize=9)

# Highlight column divider after order 5 (mid-point marker)
ax.axvline(x=5, color='#555', linewidth=1.5, linestyle='--', alpha=0.5)
ax.text(5.1, -0.35, '<-- first 5 orders | orders 6-10 -->',
        fontsize=7.5, color='#666', ha='left', va='top',
        transform=ax.get_xaxis_transform())

# Summary stats in a footer note
n_total = len(cf)
overall_5plus = (cf['order_count'] >= 5).mean() * 100
overall_10plus = (cf['order_count'] >= 10).mean() * 100
fig.text(
    0.01, -0.02,
    f'Dataset: {n_total:,} customers  |  '
    f'Reached order 5: {overall_5plus:.1f}%  |  '
    f'Reached order 10: {overall_10plus:.1f}%',
    fontsize=8, color='#666',
)

plt.tight_layout()

# ----------------------------------------------------------------
# 5. Save PNG
# ----------------------------------------------------------------
os.makedirs(STREAMLIT_DIR, exist_ok=True)
plt.savefig(OUT_PATH, dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print(f"\nSaved: {OUT_PATH}")

# ----------------------------------------------------------------
# 6. Key insights summary
# ----------------------------------------------------------------
print("\n" + "=" * 60)
print("KEY INSIGHTS")
print("=" * 60)

for label in labels:
    if label not in retention_df.index:
        continue
    row  = retention_df.loc[label]
    r5   = row['Order 5']
    r10  = row['Order 10']
    n    = int(cohort_counts.get(label, 0))
    churn = cf.loc[cf['cohort'] == label, 'churned'].mean() * 100
    short = label.split('\n')[0]
    print(f"  {short:<12}  n={n:>7,}  "
          f"order 5: {r5:>5.1f}%   order 10: {r10:>5.1f}%   "
          f"churn: {churn:>4.1f}%")

freq_r10  = retention_df.loc[labels[0], 'Order 10']
infreq_r10 = retention_df.loc[labels[-1], 'Order 10']
print(f"\nRetention gap at order 10: "
      f"Very Frequent {freq_r10:.1f}% vs Infrequent {infreq_r10:.1f}% "
      f"({freq_r10 - infreq_r10:.1f}pp difference)")

print("\n" + "=" * 60)
print("Cohort analysis complete.")
print("Next: open streamlit/app.py -> Cohort Retention page")
print("=" * 60)
