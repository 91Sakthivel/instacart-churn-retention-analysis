# ============================================================
# 04_feature_engineering.py — Validated Feature Engineering
# ============================================================
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

os.makedirs(config.PROCESSED_DIR, exist_ok=True)

print("=" * 60)
print("Feature Engineering - 04_feature_engineering.py")
print("=" * 60)
print()
print("Building features from validated signals only.")
print("Basket slope and category shift excluded --")
print("validation showed these do not predict churn")
print("in this dataset (Check 1 p>0.05, Check 3 p=0.237).")
print("Primary signals: order gap trend and reorder rate.")

print("\nLoading cleaned_orders.csv...")
mo = pd.read_csv(
    config.CLEANED_ORDERS,
    usecols=['user_id', 'order_id', 'product_id', 'order_number',
             'days_since_prior_order', 'reordered', 'department',
             'basket_size', 'bulk_buyer', 'first_order',
             'single_dept_customer', 'zero_reorder_customer'],
    dtype={
        'user_id': 'int32', 'order_id': 'int32', 'product_id': 'int32',
        'order_number': 'int16', 'reordered': 'int8',
        'bulk_buyer': 'int8', 'first_order': 'int8',
        'single_dept_customer': 'int8', 'zero_reorder_customer': 'int8',
    }
)
print(f"  Loaded: {len(mo):,} rows, {mo['user_id'].nunique():,} customers")

# Order-level dedup (one row per order for gap/basket calculations)
order_lvl = (
    mo.groupby(['user_id', 'order_id', 'order_number'], sort=False)
    .agg(
        basket_size_ord=('basket_size', 'first'),
        days_gap=('days_since_prior_order', 'first'),
        first_order_flag=('first_order', 'first'),
        reordered_mean=('reordered', 'mean'),
    )
    .reset_index()
    .sort_values(['user_id', 'order_number'])
)

n_customers = mo['user_id'].nunique()
print(f"  Order-level: {len(order_lvl):,} orders for {n_customers:,} customers")

# ================================================================
# GROUP 1 — Basket Size Features (descriptive)
# ================================================================
print("\n[Group 1] Basket size features...")
basket_feats = (
    order_lvl.groupby('user_id', sort=False)
    .agg(
        avg_basket_size=('basket_size_ord', 'mean'),
        median_basket_size=('basket_size_ord', 'median'),
        basket_std=('basket_size_ord', 'std'),
        max_basket_size=('basket_size_ord', 'max'),
    )
    .reset_index()
)
basket_feats['basket_std'] = basket_feats['basket_std'].fillna(0)
basket_feats['basket_cv'] = np.where(
    basket_feats['avg_basket_size'] > 0,
    basket_feats['basket_std'] / basket_feats['avg_basket_size'],
    0
)
print("  Basket features built -- used for revenue estimation not churn prediction")
print("  (basket decline not validated in Check 1)")

# ================================================================
# GROUP 2 — Order Gap Features (VALIDATED signal)
# ================================================================
print("\n[Group 2] Order gap features (validated signal)...")

# Work with non-first orders only for gap averages
gap_data = order_lvl[order_lvl['first_order_flag'] == 0].copy()
gap_data = gap_data.sort_values(['user_id', 'order_number'])

# Rank within user: ascending (1=earliest gap) and descending (1=latest gap)
gap_data['rank_asc'] = (
    gap_data.groupby('user_id')['order_number']
    .rank(method='dense', ascending=True).astype(int)
)
gap_data['rank_desc'] = (
    gap_data.groupby('user_id')['order_number']
    .rank(method='dense', ascending=False).astype(int)
)

# avg and std of all gaps
gap_avg = (
    gap_data.groupby('user_id', sort=False)['days_gap']
    .agg(avg_days_between_orders='mean', order_gap_std='std')
    .reset_index()
)
gap_avg['order_gap_std'] = gap_avg['order_gap_std'].fillna(0)
gap_avg['order_gap_cv'] = np.where(
    gap_avg['avg_days_between_orders'] > 0,
    gap_avg['order_gap_std'] / gap_avg['avg_days_between_orders'],
    0
)

# Early avg gap (first 3 gaps)
early_gap = (
    gap_data[gap_data['rank_asc'] <= 3]
    .groupby('user_id', sort=False)['days_gap']
    .mean().reset_index().rename(columns={'days_gap': 'early_avg_gap'})
)

# Late avg gap CLEAN: mean of gaps at positions -4,-3,-2 (rank_desc 2-4).
# Excludes rank_desc=1 (the final gap = days_since_last_order) to avoid
# embedding the churn label threshold directly in the trend signal.
late_gap = (
    gap_data[gap_data['rank_desc'].between(2, 4)]
    .groupby('user_id', sort=False)['days_gap']
    .mean().reset_index().rename(columns={'days_gap': 'late_avg_gap_clean'})
)

# days_since_last_order = last order's gap value
last_order_gap = (
    order_lvl.sort_values(['user_id', 'order_number'])
    .groupby('user_id', sort=False)
    .agg(days_since_last_order=('days_gap', 'last'),
         order_count=('order_id', 'count'))
    .reset_index()
)

# gap_acceleration: 1 if each of last 3 gaps is longer than previous
last4 = gap_data[gap_data['rank_desc'] <= 4].copy()
last4 = last4.sort_values(['user_id', 'order_number'])
last4['prev_gap'] = last4.groupby('user_id')['days_gap'].shift(1)
last4['is_inc'] = (last4['days_gap'] > last4['prev_gap']).fillna(True)
gap_accel = (
    last4.groupby('user_id', sort=False)['is_inc'].all()
    .astype(int).reset_index().rename(columns={'is_inc': 'gap_acceleration'})
)

gap_feats = (
    gap_avg
    .merge(early_gap, on='user_id', how='left')
    .merge(late_gap, on='user_id', how='left')    # late_avg_gap_clean
    .merge(last_order_gap, on='user_id', how='left')
    .merge(gap_accel, on='user_id', how='left')
)
gap_feats['early_avg_gap'] = gap_feats['early_avg_gap'].fillna(gap_feats['avg_days_between_orders'])
gap_feats['late_avg_gap_clean'] = gap_feats['late_avg_gap_clean'].fillna(gap_feats['avg_days_between_orders'])
gap_feats['gap_trend'] = gap_feats['late_avg_gap_clean'] - gap_feats['early_avg_gap']
gap_feats['gap_acceleration'] = gap_feats['gap_acceleration'].fillna(0).astype(int)

print("  Gap features built -- validation confirmed gap widens from 15.3d to 30.0d")
print("  in final 5 orders before churn. Gap trend is primary temporal signal.")

# ================================================================
# GROUP 3 — Reorder Behavior (STRONGEST validated signal)
# ================================================================
print("\n[Group 3] Reorder behavior features...")

# Overall reorder rate
reorder_overall = (
    mo.groupby('user_id', sort=False)['reordered'].mean()
    .reset_index().rename(columns={'reordered': 'reorder_rate'})
)

# Order-level reorder rate
order_rr = order_lvl[['user_id', 'order_number', 'reordered_mean']].copy()
cust_max_ord = order_rr.groupby('user_id', sort=False)['order_number'].max().reset_index()
cust_max_ord.columns = ['user_id', 'max_order_number']
order_rr = order_rr.merge(cust_max_ord, on='user_id', how='left')
order_rr['half'] = np.where(
    order_rr['order_number'] <= order_rr['max_order_number'] / 2, 'early', 'late'
)
reorder_half = (
    order_rr.groupby(['user_id', 'half'], sort=False)['reordered_mean']
    .mean().unstack(fill_value=np.nan).reset_index()
)
reorder_half.columns.name = None
if 'early' not in reorder_half.columns:
    reorder_half['early'] = np.nan
if 'late' not in reorder_half.columns:
    reorder_half['late'] = np.nan
reorder_half = reorder_half.rename(columns={
    'early': 'early_reorder_rate', 'late': 'late_reorder_rate'
})
reorder_half['early_reorder_rate'] = reorder_half['early_reorder_rate'].fillna(reorder_overall.set_index('user_id')['reorder_rate'])
reorder_half['late_reorder_rate'] = reorder_half['late_reorder_rate'].fillna(reorder_overall.set_index('user_id')['reorder_rate'])
reorder_half['reorder_trend'] = reorder_half['late_reorder_rate'] - reorder_half['early_reorder_rate']

print("  Reorder features built -- validation confirmed active: 53.3% vs churned: 43.4%.")
print("  10-point delta with p~=0. Strongest signal in dataset.")

# ================================================================
# GROUP 4 — Department Loyalty Features
# ================================================================
print("\n[Group 4] Department loyalty features...")
total_orders_per_user = order_lvl.groupby('user_id', sort=False)['order_id'].nunique().reset_index()
total_orders_per_user.columns = ['user_id', 'total_order_count']

dept_feats_list = []
key_depts = {
    'produce': 'pct_produce_orders',
    'dairy eggs': 'pct_dairy_orders',
    'snacks': 'pct_snack_orders',
    'frozen': 'pct_frozen_orders',
    'beverages': 'pct_beverage_orders',
}
for dept_name, col_name in key_depts.items():
    dept_cnt = (
        mo[mo['department'] == dept_name]
        .groupby('user_id', sort=False)['order_id'].nunique()
        .reset_index().rename(columns={'order_id': 'dept_orders'})
    )
    dept_merged = total_orders_per_user.merge(dept_cnt, on='user_id', how='left')
    dept_merged['dept_orders'] = dept_merged['dept_orders'].fillna(0)
    dept_merged[col_name] = dept_merged['dept_orders'] / dept_merged['total_order_count']
    dept_feats_list.append(dept_merged[['user_id', col_name]])

dept_feats = dept_feats_list[0]
for df in dept_feats_list[1:]:
    dept_feats = dept_feats.merge(df, on='user_id', how='left')

# Top department per customer
top_dept = (
    mo.groupby(['user_id', 'department'], sort=False)['order_id'].nunique()
    .reset_index()
)
top_dept = (
    top_dept.sort_values(['user_id', 'order_id'], ascending=[True, False])
    .groupby('user_id', sort=False).first()
    .reset_index()[['user_id', 'department']]
    .rename(columns={'department': 'top_department'})
)
dept_feats = dept_feats.merge(top_dept, on='user_id', how='left')
print("  Department features built -- used for intervention hook recommendation")

# ================================================================
# GROUP 5 — Customer Segment Features
# ================================================================
print("\n[Group 5] Customer segment features...")
seg_feats = (
    mo.groupby('user_id', sort=False)
    .agg(
        bulk_buyer=('bulk_buyer', 'first'),
        single_dept_customer=('single_dept_customer', 'first'),
        zero_reorder_customer=('zero_reorder_customer', 'first'),
    )
    .reset_index()
)

# ================================================================
# GROUP 6 — Business Metric Features
# ================================================================
print("\n[Group 6] Business metric features...")

# ================================================================
# GROUP 7 — Churn Label
# ================================================================
print("\n[Group 7] Churn label...")

# ================================================================
# Merge all feature groups
# ================================================================
print("\nMerging all feature groups...")
features = (
    basket_feats
    .merge(gap_feats, on='user_id', how='left')
    .merge(reorder_overall, on='user_id', how='left')
    .merge(reorder_half[['user_id', 'early_reorder_rate', 'late_reorder_rate', 'reorder_trend']],
           on='user_id', how='left')
    .merge(dept_feats, on='user_id', how='left')
    .merge(seg_feats, on='user_id', how='left')
)

# Business metrics
features['annual_orders_est'] = np.where(
    features['avg_days_between_orders'] > 0,
    365.0 / features['avg_days_between_orders'],
    0
)
features['annual_value_est'] = features['annual_orders_est'] * 110.0
features['revenue_at_risk_90d'] = (
    features['avg_basket_size'] * features['annual_orders_est'] * 0.25
)

# Churn label
features['churned'] = (
    (features['days_since_last_order'] >= 30) &
    (features['order_count'] >= 5)
).astype(int)

# Fill any remaining NaNs
numeric_cols = features.select_dtypes(include=[np.number]).columns
features[numeric_cols] = features[numeric_cols].fillna(0)

# ================================================================
# Print feature summary
# ================================================================
print("\nFeature summary:")
summary_cols = [
    'avg_basket_size', 'avg_days_between_orders', 'order_gap_cv',
    'gap_trend', 'reorder_rate', 'early_reorder_rate', 'late_reorder_rate',
    'reorder_trend', 'early_avg_gap', 'late_avg_gap_clean', 'gap_acceleration',
    'days_since_last_order', 'order_count', 'annual_value_est',
    'pct_produce_orders', 'pct_dairy_orders',
]
print(f"{'Feature':<30} | {'Mean':>8} | {'Std':>8} | {'% Missing':>10}")
print("-" * 65)
for col in summary_cols:
    if col in features.columns:
        m = features[col].mean()
        s = features[col].std()
        pct_miss = features[col].isna().sum() / len(features) * 100
        print(f"  {col:<28} | {m:>8.2f} | {s:>8.2f} | {pct_miss:>9.1f}%")

n_widening = (features['gap_trend'] > 5).sum()
n_churned = features['churned'].sum()
n_low_rr = (features['reorder_rate'] < 0.35).sum()

print()
print(f"Customers with gap_trend > 5 (widening): {n_widening:,} ({n_widening/len(features)*100:.1f}%)")
print(f"Customers churned: {n_churned:,} ({n_churned/len(features)*100:.1f}%)")
print(f"Customers with reorder_rate < 0.35: {n_low_rr:,} ({n_low_rr/len(features)*100:.1f}%)")

n_features = len([c for c in features.columns if c != 'user_id'])
print(f"\nTotal features built: {n_features}")

print(f"\nSaving customer_features.csv...")
features.to_csv(config.CUSTOMER_FEATURES, index=False)
size_mb = os.path.getsize(config.CUSTOMER_FEATURES) / 1024 / 1024
print(f"  Saved: {config.CUSTOMER_FEATURES}")
print(f"  Size: {size_mb:.1f} MB, {len(features):,} customers")

print("\n" + "=" * 60)
print(f"Feature engineering done. {n_features} features built")
print(f"for {len(features):,} customers. Validated signals only.")
print("Next -- pipeline/04_rfm.py")
print("=" * 60)
