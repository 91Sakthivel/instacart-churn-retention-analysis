# ============================================================
# validate_signals.py — Pre-model Signal Validation
# ============================================================
# Before trusting any trajectory assumptions or feature
# engineering choices, this script asks the data directly
# whether the behavioural signals actually exist.
#
# Five checks:
#   1. Does basket size decline before churn?
#   2. Do order gaps widen before churn?
#   3. Does high-value category share drop before churn?
#   4. Is reorder rate different between churned / active?
#   5. Do natural clusters match A/B/C/D assumptions?
#
# Churn label (same definition as 08_churn_model.py):
#   churned = last_order_gap >= 30  AND  order_count >= 5
# ============================================================

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
import numpy as np
from scipy.stats import ttest_ind, ttest_rel
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

os.makedirs(os.path.join(config.CHARTS_DIR, 'validation'), exist_ok=True)
VAL_DIR = os.path.join(config.CHARTS_DIR, 'validation')

print("=" * 62)
print("Signal Validation — validate_signals.py")
print("=" * 62)

# -------------------------------------------------------
# Load product-level rows (minimal columns)
# -------------------------------------------------------
print("\nLoading master_orders.csv...")
mo = pd.read_csv(
    config.MASTER_ORDERS,
    usecols=['user_id', 'order_id', 'order_number',
             'days_since_prior_order', 'reordered', 'department'],
    dtype={
        'user_id':      'int32',
        'order_id':     'int32',
        'order_number': 'int16',
        'reordered':    'int8',
    }
)
print(f"  {len(mo):,} product rows | "
      f"{mo['user_id'].nunique():,} customers | "
      f"{mo['order_id'].nunique():,} orders")

# Premium categories: produce + dairy eggs (confirmed present in data)
# meat seafood included if column exists in the department values
premium_depts = {'produce', 'dairy eggs', 'meat seafood'}
mo['is_premium'] = mo['department'].isin(premium_depts).astype('int8')
mo['is_produce'] = (mo['department'] == 'produce').astype('int8')
mo['is_dairy']   = (mo['department'] == 'dairy eggs').astype('int8')

# -------------------------------------------------------
# Collapse to order level
# -------------------------------------------------------
print("Aggregating to order level...")
order_meta = (
    mo[['order_id', 'user_id', 'order_number', 'days_since_prior_order']]
    .drop_duplicates('order_id')
    .copy()
)
order_agg = (
    mo.groupby('order_id', sort=False)
    .agg(
        basket_size    = ('reordered',    'count'),
        reordered_sum  = ('reordered',    'sum'),
        premium_items  = ('is_premium',   'sum'),
        produce_items  = ('is_produce',   'sum'),
        dairy_items    = ('is_dairy',     'sum'),
    )
    .reset_index()
)
orders = order_meta.merge(order_agg, on='order_id', how='left')
orders['reorder_rate']   = orders['reordered_sum']  / orders['basket_size']
orders['premium_share']  = orders['premium_items']  / orders['basket_size']
orders['produce_share']  = orders['produce_items']  / orders['basket_size']

orders = orders.sort_values(['user_id', 'order_number'])
print(f"  {len(orders):,} order rows")

# -------------------------------------------------------
# Customer-level summary
# -------------------------------------------------------
avg_gap_per_cust = (
    orders[orders['days_since_prior_order'].notna() &
           (orders['days_since_prior_order'] > 0)]
    .groupby('user_id')['days_since_prior_order']
    .mean()
    .rename('avg_days_between_orders')
)

cust = (
    orders.groupby('user_id')
    .agg(
        order_count   = ('order_id', 'count'),
        last_gap      = ('days_since_prior_order', 'last'),
        avg_basket    = ('basket_size',   'mean'),
        reorder_rate  = ('reorder_rate',  'mean'),
    )
    .reset_index()
    .join(avg_gap_per_cust, on='user_id')
)

cust['churned'] = (
    cust['last_gap'].fillna(0).ge(30) & cust['order_count'].ge(5)
).astype(int)

n_churn  = cust['churned'].sum()
n_active = (cust['churned'] == 0).sum()
print(f"\n  Churn label: {n_churn:,} churned ({n_churn/len(cust)*100:.1f}%) | "
      f"{n_active:,} active")

# Attach churn flag and order count to order-level table
orders = orders.merge(cust[['user_id', 'churned', 'order_count']], on='user_id', how='left')

# Reverse order index: rev=1 → last order, rev=2 → 2nd from last
orders['rev'] = orders.groupby('user_id').cumcount(ascending=False) + 1
# Forward order index: fwd=1 → first order
orders['fwd'] = orders.groupby('user_id').cumcount(ascending=True)  + 1

print("\n" + "=" * 62)
print("CHECK 1 — Does basket size decline before churn?")
print("=" * 62)

# Churned customers with 10+ orders
churn10 = orders[(orders['churned'] == 1) & (orders['order_count'] >= 10)]
n_cust_c10 = churn10['user_id'].nunique()

basket_by_rev = (
    churn10[churn10['rev'] <= 10]
    .groupby('rev')['basket_size']
    .mean()
    .sort_index()
)

print(f"\n  Churned customers with 10+ orders: {n_cust_c10:,}")
print(f"\n  Avg basket size at each position (rev=1 is last order):")
print(f"  {'Pos':>5}  {'Label':>8}  {'Avg Basket':>11}")
print(f"  {'-'*30}")
for rev, val in basket_by_rev.items():
    label = f"-{rev}"
    print(f"  {rev:>5}  {label:>8}  {val:>11.2f}")

slope_val = np.polyfit(basket_by_rev.index, basket_by_rev.values, 1)[0]
early_avg = basket_by_rev[basket_by_rev.index >= 8].mean()   # positions -10 to -8
late_avg  = basket_by_rev[basket_by_rev.index <= 3].mean()   # positions -3 to -1

print(f"\n  Early avg (positions -10 to -8): {early_avg:.2f} items")
print(f"  Late avg  (positions  -3 to -1): {late_avg:.2f} items")
print(f"  Slope across positions:          {slope_val:+.4f} items/position")

CHECK1_VALIDATED = slope_val > 0.05   # rev index goes 1→10 (recent→old), so positive slope = decline toward churn
if late_avg < early_avg and abs(early_avg - late_avg) > 0.3:
    print(f"\n  RESULT: YES — decline exists ({early_avg:.2f} -> {late_avg:.2f}, "
          f"drop of {early_avg - late_avg:.2f} items)")
    CHECK1_VALIDATED = True
else:
    print(f"\n  RESULT: NO CLEAR DECLINE — difference {early_avg - late_avg:.2f} items "
          f"(below 0.3 threshold)")
    CHECK1_VALIDATED = False

# Chart 1
fig, ax = plt.subplots(figsize=(9, 4))
labels = [f"-{r}" for r in basket_by_rev.index]
ax.plot(range(len(basket_by_rev)), basket_by_rev.values,
        marker='o', color='#E91E63', linewidth=2)
ax.fill_between(range(len(basket_by_rev)), basket_by_rev.values,
                alpha=0.12, color='#E91E63')
ax.set_xticks(range(len(basket_by_rev)))
ax.set_xticklabels(labels[::-1])   # show as -10, -9, ..., -1
ax.set_xlabel('Orders before churn')
ax.set_ylabel('Avg basket size (items)')
ax.set_title('CHECK 1: Basket Size Before Churn', fontsize=12, fontweight='bold')
ax.axhline(early_avg, color='grey', linestyle='--', alpha=0.6, label=f'Early avg {early_avg:.1f}')
ax.axhline(late_avg,  color='red',  linestyle='--', alpha=0.6, label=f'Late avg {late_avg:.1f}')
ax.legend(fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(VAL_DIR, 'check1_basket_decline.png'), dpi=config.CHART_DPI, bbox_inches='tight')
plt.close(fig)


print("\n" + "=" * 62)
print("CHECK 2 — Do order gaps widen before churn?")
print("=" * 62)

gap_by_rev = (
    churn10[(churn10['rev'] <= 5) & (churn10['days_since_prior_order'].notna())]
    .groupby('rev')['days_since_prior_order']
    .mean()
    .sort_index()
)

print(f"\n  Avg days_since_prior_order at last 5 positions:")
print(f"  {'Pos':>5}  {'Label':>8}  {'Avg Gap (days)':>15}")
print(f"  {'-'*33}")
for rev, val in gap_by_rev.items():
    label = f"-{rev}"
    print(f"  {rev:>5}  {label:>8}  {val:>15.2f}")

if len(gap_by_rev) >= 2:
    early_gap = gap_by_rev.iloc[-1]   # furthest from churn
    late_gap  = gap_by_rev.iloc[0]    # closest to churn (rev=1)
    gap_change = late_gap - early_gap
    CHECK2_VALIDATED = gap_change > 1.0
    status = "YES — gaps widen" if CHECK2_VALIDATED else "NO CLEAR WIDENING"
    print(f"\n  Gap at -5: {early_gap:.2f} days  ->  Gap at -1: {late_gap:.2f} days")
    print(f"  Change: {gap_change:+.2f} days")
    print(f"  RESULT: {status} (change > 1.0 day threshold: {CHECK2_VALIDATED})")
else:
    CHECK2_VALIDATED = False
    print("  RESULT: insufficient data")

# Chart 2
fig, ax = plt.subplots(figsize=(7, 4))
labels2 = [f"-{r}" for r in gap_by_rev.index]
ax.bar(labels2, gap_by_rev.values, color='#FF5722', edgecolor='white')
ax.set_xlabel('Orders before churn')
ax.set_ylabel('Avg days since prior order')
ax.set_title('CHECK 2: Order Gap Before Churn', fontsize=12, fontweight='bold')
fig.tight_layout()
fig.savefig(os.path.join(VAL_DIR, 'check2_gap_widening.png'), dpi=config.CHART_DPI, bbox_inches='tight')
plt.close(fig)


print("\n" + "=" * 62)
print("CHECK 3 — Does high-value category share drop before churn?")
print("=" * 62)

# Customers with 6+ orders to have non-overlapping first-3 and last-3 windows
cust6 = cust[cust['order_count'] >= 6][['user_id', 'churned']].copy()
orders6 = orders[orders['user_id'].isin(cust6['user_id'])]

first3_prem = (
    orders6[orders6['fwd'] <= 3]
    .groupby('user_id')['premium_share']
    .mean()
    .rename('early_prem')
)
last3_prem = (
    orders6[orders6['rev'] <= 3]
    .groupby('user_id')['premium_share']
    .mean()
    .rename('late_prem')
)

prem_df = cust6.join(first3_prem, on='user_id').join(last3_prem, on='user_id').dropna()

churned_early = prem_df[prem_df['churned'] == 1]['early_prem'].mean()
churned_late  = prem_df[prem_df['churned'] == 1]['late_prem'].mean()
active_early  = prem_df[prem_df['churned'] == 0]['early_prem'].mean()
active_late   = prem_df[prem_df['churned'] == 0]['late_prem'].mean()

# Paired t-test on churned customers: did their premium share drop?
churn_paired = prem_df[prem_df['churned'] == 1]
t_stat, p_val = ttest_rel(churn_paired['early_prem'], churn_paired['late_prem'])

print(f"\n  Customers with 6+ orders analysed: {len(prem_df):,}")
print(f"\n  High-value category share (produce + dairy + meat):")
print(f"  {'Group':<20}  {'Early (first 3)':>16}  {'Late (last 3)':>14}  {'Change':>8}")
print(f"  {'-'*62}")
print(f"  {'Churned':<20}  {churned_early:>16.1%}  {churned_late:>14.1%}  "
      f"  {churned_late - churned_early:>+7.1%}")
print(f"  {'Active':<20}  {active_early:>16.1%}  {active_late:>14.1%}  "
      f"  {active_late - active_early:>+7.1%}")
print(f"\n  Paired t-test (churned early vs late): "
      f"t={t_stat:.3f}, p={p_val:.4f}")

CHECK3_VALIDATED = p_val < 0.05 and (churned_early > churned_late)
if CHECK3_VALIDATED:
    print(f"  RESULT: YES — drop is statistically significant (p={p_val:.4f} < 0.05)")
else:
    if p_val >= 0.05:
        print(f"  RESULT: NOT SIGNIFICANT (p={p_val:.4f} >= 0.05)")
    else:
        print(f"  RESULT: SHARE INCREASES before churn — assumption incorrect")
    CHECK3_VALIDATED = False

# Chart 3
fig, ax = plt.subplots(figsize=(7, 4))
x = np.arange(2)
w = 0.3
ax.bar(x - w/2, [churned_early, churned_late],  w, label='Churned', color='#F44336', edgecolor='white')
ax.bar(x + w/2, [active_early,  active_late],   w, label='Active',  color='#4CAF50', edgecolor='white')
ax.set_xticks(x)
ax.set_xticklabels(['Early (first 3 orders)', 'Late (last 3 orders)'])
ax.set_ylabel('Avg premium category share')
ax.set_title('CHECK 3: High-Value Category Share\nChurned vs Active Customers',
             fontsize=11, fontweight='bold')
ax.legend()
ax.set_ylim(0, max(churned_early, active_early) * 1.3)
star = f"p={p_val:.4f}{'*' if p_val < 0.05 else ' (ns)'}"
ax.text(0.5, 0.92, star, transform=ax.transAxes, ha='center', fontsize=10, color='black')
fig.tight_layout()
fig.savefig(os.path.join(VAL_DIR, 'check3_category_shift.png'), dpi=config.CHART_DPI, bbox_inches='tight')
plt.close(fig)


print("\n" + "=" * 62)
print("CHECK 4 — Is reorder rate different between churned / active?")
print("=" * 62)

eligible = cust[cust['order_count'] >= 5]
churn_rr  = eligible[eligible['churned'] == 1]['reorder_rate'].dropna()
active_rr = eligible[eligible['churned'] == 0]['reorder_rate'].dropna()

t4, p4 = ttest_ind(churn_rr, active_rr, equal_var=False)

print(f"\n  Churned customers:  n={len(churn_rr):,}  "
      f"avg reorder rate = {churn_rr.mean():.4f}  std={churn_rr.std():.4f}")
print(f"  Active  customers:  n={len(active_rr):,}  "
      f"avg reorder rate = {active_rr.mean():.4f}  std={active_rr.std():.4f}")
print(f"\n  Welch t-test: t={t4:.3f}, p={p4:.6f}")

CHECK4_VALIDATED = p4 < 0.05
diff4 = active_rr.mean() - churn_rr.mean()
if CHECK4_VALIDATED:
    direction = "Active customers reorder MORE" if diff4 > 0 else "Churned customers reorder MORE"
    print(f"  RESULT: YES — significant difference (p={p4:.6f})")
    print(f"          {direction} (delta = {abs(diff4):.4f})")
else:
    print(f"  RESULT: NOT SIGNIFICANT (p={p4:.6f})")

# Chart 4
fig, ax = plt.subplots(figsize=(6, 4))
ax.bar(['Churned', 'Active'], [churn_rr.mean(), active_rr.mean()],
       color=['#F44336', '#4CAF50'], edgecolor='white', width=0.4)
ax.errorbar(['Churned', 'Active'],
            [churn_rr.mean(), active_rr.mean()],
            yerr=[churn_rr.sem(), active_rr.sem()],
            fmt='none', color='black', capsize=6)
ax.set_ylabel('Avg reorder rate')
ax.set_ylim(0, max(churn_rr.mean(), active_rr.mean()) * 1.3)
ax.set_title('CHECK 4: Reorder Rate — Churned vs Active',
             fontsize=11, fontweight='bold')
p_label = f"p={p4:.4f}{'***' if p4 < 0.001 else '*' if p4 < 0.05 else ' (ns)'}"
ax.text(0.5, 0.92, p_label, transform=ax.transAxes, ha='center', fontsize=10)
fig.tight_layout()
fig.savefig(os.path.join(VAL_DIR, 'check4_reorder_rate.png'), dpi=config.CHART_DPI, bbox_inches='tight')
plt.close(fig)


print("\n" + "=" * 62)
print("CHECK 5 — Do natural clusters match A/B/C/D assumptions?")
print("=" * 62)

# Features for clustering — eligible customers only (5+ orders)
clust_df = eligible[['user_id', 'avg_basket', 'order_count',
                      'avg_days_between_orders', 'reorder_rate']].dropna()

X_raw = clust_df[['avg_basket', 'order_count', 'avg_days_between_orders']].values
scaler = StandardScaler()
X_sc   = scaler.fit_transform(X_raw)

print(f"\n  Running k-means (k=4) on "
      f"avg_basket, order_count, avg_days_between_orders...")
km = KMeans(n_clusters=4, random_state=config.RANDOM_STATE, n_init=20)
clust_df = clust_df.copy()
clust_df['cluster'] = km.fit_predict(X_sc)

# Cluster centers in original scale
centers_scaled = km.cluster_centers_
centers_orig   = scaler.inverse_transform(centers_scaled)

print(f"\n  Cluster centers (original scale):")
print(f"  {'Cluster':>9}  {'Avg Basket':>12}  {'Order Count':>12}  "
      f"{'Avg Gap (d)':>12}  {'Size':>7}  {'Churn %':>8}")
print(f"  {'-'*70}")

clust_merge = clust_df[['user_id', 'cluster']].merge(
    cust[['user_id', 'churned']], on='user_id', how='left'
)
for c in range(4):
    c_mask = clust_df['cluster'] == c
    n      = c_mask.sum()
    cx, cy, cz = centers_orig[c]
    churn_pct  = clust_merge[clust_merge['cluster'] == c]['churned'].mean() * 100
    print(f"  {'C'+str(c):>9}  {cx:>12.1f}  {cy:>12.1f}  {cz:>12.1f}  "
          f"{n:>7,}  {churn_pct:>7.1f}%")

# Check if cluster patterns match trajectory assumptions:
# A = gradual decline: moderate basket, high order_count, low gap
# B = cliff drop:      high basket early, now low basket
# C = oscillating:     irregular, moderate everything
# D = category collapse: stable basket, normal gap, category shift
# Since k-means only has 3 features it can't distinguish C vs D perfectly
print(f"\n  Trajectory-to-cluster mapping interpretation:")
for c in range(4):
    cx, cy, cz = centers_orig[c]
    churn_pct  = clust_merge[clust_merge['cluster'] == c]['churned'].mean() * 100
    # Characterise based on centers
    if cz < 12 and cy > 15:
        profile = "frequent buyers, small-medium basket (likely A — gradual)"
    elif cz > 18 and cy < 10:
        profile = "infrequent, fewer orders (likely B or D — drop/collapse)"
    elif cz < 12 and cy < 10:
        profile = "newer/lower frequency, regular (likely C — oscillating)"
    else:
        profile = "moderate on all dimensions (mixed)"
    print(f"  C{c} ({churn_pct:.1f}% churn): {profile}")

# Chart 5 — scatter of first 2 features coloured by cluster
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

sample_n = min(20000, len(clust_df))
samp = clust_df.sample(sample_n, random_state=42)
colors5 = ['#2196F3', '#4CAF50', '#FF9800', '#F44336']

ax = axes[0]
for c in range(4):
    mask = samp['cluster'] == c
    n_c  = clust_df[clust_df['cluster'] == c].shape[0]
    ax.scatter(samp[mask]['avg_basket'], samp[mask]['order_count'],
               s=4, alpha=0.4, color=colors5[c],
               label=f"C{c} (n={n_c:,})")
ax.set_xlabel('Avg basket size')
ax.set_ylabel('Order count')
ax.set_title('Clusters: Basket Size vs Order Count', fontsize=11, fontweight='bold')
ax.legend(fontsize=8, markerscale=4)

ax = axes[1]
for c in range(4):
    mask = samp['cluster'] == c
    n_c  = clust_df[clust_df['cluster'] == c].shape[0]
    ax.scatter(samp[mask]['avg_days_between_orders'], samp[mask]['avg_basket'],
               s=4, alpha=0.4, color=colors5[c],
               label=f"C{c} (n={n_c:,})")
ax.set_xlabel('Avg days between orders')
ax.set_ylabel('Avg basket size')
ax.set_title('Clusters: Order Gap vs Basket Size', fontsize=11, fontweight='bold')
ax.legend(fontsize=8, markerscale=4)

fig.tight_layout()
fig.savefig(os.path.join(VAL_DIR, 'check5_clusters.png'), dpi=config.CHART_DPI, bbox_inches='tight')
plt.close(fig)

# -------------------------------------------------------
# Inertia / elbow check (k=2 to 7)
# -------------------------------------------------------
inertias = []
ks = range(2, 8)
for k in ks:
    inertias.append(KMeans(n_clusters=k, random_state=42, n_init=10).fit(X_sc).inertia_)

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(list(ks), inertias, marker='o', color='#673AB7', linewidth=2)
ax.axvline(4, color='red', linestyle='--', alpha=0.6, label='k=4 chosen')
ax.set_xlabel('k (number of clusters)')
ax.set_ylabel('Inertia (within-cluster SS)')
ax.set_title('Elbow Plot — Optimal k?', fontsize=11, fontweight='bold')
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(VAL_DIR, 'check5_elbow.png'), dpi=config.CHART_DPI, bbox_inches='tight')
plt.close(fig)


# -------------------------------------------------------
# Summary
# -------------------------------------------------------
print("\n" + "=" * 62)
print("VALIDATION SUMMARY")
print("=" * 62)

validated   = []
not_supported = []

c1 = ("Basket size decline before churn", CHECK1_VALIDATED)
c2 = ("Order gap widens before churn",    CHECK2_VALIDATED)
c3 = ("Category share drops before churn", CHECK3_VALIDATED)
c4 = ("Reorder rate differs churned/active", CHECK4_VALIDATED)

for name, result in [c1, c2, c3, c4]:
    if result:
        validated.append(name)
        print(f"  VALIDATED     {name}")
    else:
        not_supported.append(name)
        print(f"  NOT SUPPORTED {name}")

print(f"\nCluster check (Check 5) requires manual interpretation above.")

print(f"\n--- What this means for feature engineering ---")
if CHECK1_VALIDATED:
    print("  basket_slope and last_3_avg_basket ARE useful features.")
else:
    print("  basket_slope may be weak — declining basket signal is subtle.")
    print("  Consider using BOTH slope AND last_3/first_3 ratio instead.")

if CHECK2_VALIDATED:
    print("  avg_days_between_orders and order_gap trend ARE useful features.")
else:
    print("  Order gap widening is NOT a reliable pre-churn signal in this data.")
    print("  order_gap_std may still capture variability but don't rely on trending gap.")

if CHECK3_VALIDATED:
    print("  Category shift (produce/dairy share) IS a real signal.")
    print("  pct_produce_orders and pct_dairy_orders belong in the feature set.")
else:
    print("  Category shift assumption is NOT supported — produce/dairy share")
    print("  does not significantly drop before churn in this dataset.")
    print("  These features can stay but don't expect high importance.")

if CHECK4_VALIDATED:
    print("  reorder_rate IS a useful discriminator — keep it.")
else:
    print("  reorder_rate does NOT separate churned from active reliably.")

# Specific trajectory honesty check
print(f"\n--- Trajectory type assumptions ---")
print(f"  Type A (Gradual Decline): "
      + ("supported — slope signal exists" if CHECK1_VALIDATED
         else "WEAK — slope decline not clearly visible in data"))
print(f"  Type B (Cliff Drop): requires last_3 vs first_3 comparison — "
      + ("plausible" if CHECK1_VALIDATED else "uncertain"))
print(f"  Type C (Oscillating): CV > 0.4 is a reasonable variability flag — "
      "NOT validated here but conceptually sound")
print(f"  Type D (Category Collapse): "
      + ("supported — category share difference exists" if CHECK3_VALIDATED
         else "NOT supported by t-test — difference not significant"))

print(f"\nAll validation charts saved to: {VAL_DIR}")
chart_count = len([f for f in os.listdir(VAL_DIR) if f.endswith('.png')])
print(f"  {chart_count} charts written")
print()
