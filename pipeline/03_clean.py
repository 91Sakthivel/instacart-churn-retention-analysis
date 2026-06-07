# ============================================================
# 03_clean.py — Data Cleaning (7 documented decisions)
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
print("Instacart Data Cleaning - 03_clean.py")
print("=" * 60)

print("\nLoading master_orders.csv...")
mo = pd.read_csv(
    config.MASTER_ORDERS,
    dtype={
        'user_id': 'int32', 'order_id': 'int32', 'product_id': 'int32',
        'order_number': 'int16', 'order_dow': 'int8',
        'order_hour_of_day': 'int8', 'reordered': 'int8',
    }
)
rows_raw = len(mo)
cust_raw = mo['user_id'].nunique()
print(f"  Loaded: {rows_raw:,} rows, {cust_raw:,} customers")

# -------------------------------------------------------
# DECISION 1 — days_since_prior_order nulls = first orders
# -------------------------------------------------------
print("\n[Decision 1] days_since_prior_order nulls...")
null_gaps = mo['days_since_prior_order'].isna().sum()
mo['first_order'] = mo['days_since_prior_order'].isna().astype('int8')
mo['days_since_prior_order'] = mo['days_since_prior_order'].fillna(0)
pct_first = null_gaps / rows_raw * 100
print(f"  {null_gaps:,} first orders flagged ({pct_first:.1f}% of all orders) --")
print("  keeping these, they represent customer acquisition")

# -------------------------------------------------------
# DECISION 2 — Duplicate order_id + product_id check
# -------------------------------------------------------
print("\n[Decision 2] Duplicate order check...")
before = len(mo)
mo = mo.drop_duplicates(subset=['order_id', 'product_id'])
dupes = before - len(mo)
if dupes == 0:
    print("  No duplicates found -- dataset is clean")
else:
    print(f"  {dupes:,} duplicate rows removed")

# -------------------------------------------------------
# DECISION 3 — Basket size outliers (bulk buyers)
# -------------------------------------------------------
print("\n[Decision 3] Basket size outliers...")
order_size = mo.groupby('order_id', sort=False)['product_id'].count().reset_index()
order_size.columns = ['order_id', 'basket_size']
mo = mo.merge(order_size, on='order_id', how='left')

# Per-customer max basket for bulk flag
cust_max_basket = mo.groupby('user_id', sort=False)['basket_size'].max().reset_index()
cust_max_basket.columns = ['user_id', 'max_basket_size']
mo = mo.merge(cust_max_basket, on='user_id', how='left')
mo['bulk_buyer'] = (mo['max_basket_size'] > 80).astype('int8')
mo = mo.drop(columns=['max_basket_size'])

# Compute churn label for reporting
order_cnt_u = mo.groupby('user_id', sort=False)['order_number'].max().reset_index()
order_cnt_u.columns = ['user_id', 'max_order']
last_gap_u = (
    mo.sort_values(['user_id', 'order_number'])
    .groupby('user_id', sort=False)
    .last()[['days_since_prior_order']]
    .reset_index()
    .rename(columns={'days_since_prior_order': 'last_gap'})
)
churn_lu = order_cnt_u.merge(last_gap_u, on='user_id')
churn_lu['churned'] = ((churn_lu['last_gap'] >= 30) & (churn_lu['max_order'] >= 5)).astype(int)
mo = mo.merge(churn_lu[['user_id', 'churned']], on='user_id', how='left')

bulk_custs = mo[mo['bulk_buyer'] == 1]['user_id'].nunique()
normal_custs = mo[mo['bulk_buyer'] == 0]['user_id'].nunique()
bulk_churn = mo[mo['bulk_buyer'] == 1].groupby('user_id')['churned'].first().mean()
normal_churn = mo[mo['bulk_buyer'] == 0].groupby('user_id')['churned'].first().mean()
cmp = "higher" if bulk_churn > normal_churn else "lower"
print(f"  {bulk_custs:,} customers flagged as bulk buyers (max basket > 80 items)")
print(f"  Bulk buyer churn rate: {bulk_churn*100:.1f}% vs normal {normal_churn*100:.1f}% -- {cmp}")
print("  Decision: kept -- bulk buyers are real customers with different but valid behavior")

# -------------------------------------------------------
# DECISION 4 — Customers with < 5 orders
# -------------------------------------------------------
print("\n[Decision 4] Customers with fewer than 5 orders...")
before_rows = len(mo)
before_cust = mo['user_id'].nunique()

order_count_u = mo.groupby('user_id', sort=False)['order_number'].max().reset_index()
order_count_u.columns = ['user_id', 'order_count']
mo = mo.merge(order_count_u, on='user_id', how='left')

excluded_users = mo[mo['order_count'] < 5][['user_id', 'order_count']].drop_duplicates()
excl_churn = excluded_users.merge(churn_lu[['user_id', 'churned']], on='user_id', how='left')
excl_avg_basket = mo[mo['order_count'] < 5].groupby('user_id', sort=False)['basket_size'].mean().mean()

excluded_df = excluded_users.merge(churn_lu[['user_id', 'churned']], on='user_id', how='left')
excluded_df['avg_basket'] = excluded_users.merge(
    mo.groupby('user_id', sort=False)['basket_size'].mean().reset_index(),
    on='user_id', how='left'
)['basket_size']

os.makedirs(os.path.dirname(config.EXCLUDED_CUSTOMERS), exist_ok=True)
excluded_users.to_csv(config.EXCLUDED_CUSTOMERS, index=False)

mo = mo[mo['order_count'] >= 5].copy()
after_rows = len(mo)
after_cust = mo['user_id'].nunique()
excl_pct = (before_cust - after_cust) / before_cust * 100

print(f"  {before_cust - after_cust:,} customers excluded ({excl_pct:.1f}% of base) --")
print("  insufficient order history for behavioral modeling")
print(f"  Excluded customers: avg basket {excl_avg_basket:.1f} items, "
      f"churn rate {excl_churn['churned'].mean()*100:.1f}%")
print(f"  Saved to: {config.EXCLUDED_CUSTOMERS}")

# -------------------------------------------------------
# DECISION 5 — Single department customers
# -------------------------------------------------------
print("\n[Decision 5] Single department customers...")
dept_per_user = mo.groupby('user_id', sort=False)['department'].nunique().reset_index()
dept_per_user.columns = ['user_id', 'dept_count']
mo = mo.merge(dept_per_user, on='user_id', how='left')
mo['single_dept_customer'] = (mo['dept_count'] == 1).astype('int8')
mo = mo.drop(columns=['dept_count'])
single_dept_n = mo[mo['single_dept_customer'] == 1]['user_id'].nunique()
single_dept_pct = single_dept_n / mo['user_id'].nunique() * 100
print(f"  {single_dept_n:,} single-department customers ({single_dept_pct:.1f}%) -- flagged, not excluded")

# -------------------------------------------------------
# DECISION 6 — Zero reorder customers
# -------------------------------------------------------
print("\n[Decision 6] Zero reorder customers...")
reorder_sum = mo.groupby('user_id', sort=False)['reordered'].sum().reset_index()
reorder_sum.columns = ['user_id', 'total_reorders']
mo = mo.merge(reorder_sum, on='user_id', how='left')
mo['zero_reorder_customer'] = (mo['total_reorders'] == 0).astype('int8')
mo = mo.drop(columns=['total_reorders'])
zero_rr_n = mo[mo['zero_reorder_customer'] == 1]['user_id'].nunique()
print(f"  {zero_rr_n:,} zero-reorder customers -- flagged as one-time shoppers, different behavior pattern")

# -------------------------------------------------------
# DECISION 7 — Final null check
# -------------------------------------------------------
print("\n[Decision 7] Final null check...")
key_cols = ['user_id', 'order_id', 'product_id', 'department', 'order_number']
null_report = mo[key_cols].isnull().sum()
print("  Null counts per key column:")
for col, n in null_report.items():
    status = "OK" if n == 0 else "PROBLEM"
    print(f"    {col:<20}: {n:>6} nulls  [{status}]")

# -------------------------------------------------------
# Summary
# -------------------------------------------------------
print("\n" + "-" * 55)
print(f"{'Step':<30} | {'Before':>8} | {'After':>8} | {'Removed':>8}")
print("-" * 55)
print(f"{'Original rows':<30} | {rows_raw:>8,} | {'--':>8} | {'--':>8}")
print(f"{'After dedup':<30} | {rows_raw:>8,} | {rows_raw-dupes:>8,} | {dupes:>8,}")
print(f"{'Remove <5 order customers':<30} | {before_rows:>8,} | {after_rows:>8,} | {before_rows-after_rows:>8,}")
print(f"{'Final dataset':<30} | {after_rows:>8,} | {mo['user_id'].nunique():>8,} | {'--':>8}")
print("-" * 55)

print(f"\nSaving cleaned_orders.csv...")
mo.to_csv(config.CLEANED_ORDERS, index=False)
size_mb = os.path.getsize(config.CLEANED_ORDERS) / 1024 / 1024
print(f"  Saved: {config.CLEANED_ORDERS}")
print(f"  Size: {size_mb:.0f} MB, {after_rows:,} rows")

print("\n" + "=" * 60)
print(f"Cleaning done. 7 decisions documented.")
print(f"{after_rows:,} rows, {mo['user_id'].nunique():,} customers ready for feature engineering.")
print("Next -- pipeline/04_feature_engineering.py")
print("=" * 60)
