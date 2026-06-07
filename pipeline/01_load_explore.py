# ============================================================
# 01_load_explore.py — Data Loading & Initial Orientation
# ============================================================
# First stop in the pipeline. Load all 6 source files, get a
# clear picture of what we're working with, then write out the
# merged master file that every downstream script pulls from.
#
# Nothing fancy here — just making sure we understand the data
# before we start doing anything clever with it.
# ============================================================

import sys
import os

# Pull in paths from config — never hardcode these
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

import pandas as pd
import numpy as np


# -------------------------------------------------------
# Helper — fail loudly if a source file is missing.
# Better to crash here with a clear message than get a
# mysterious KeyError three scripts later.
# -------------------------------------------------------
def require_file(path):
    if not os.path.exists(path):
        print(f"\n  ERROR: Can't find {path}")
        print("  Drop the Instacart CSV files into data/raw/ and re-run.")
        sys.exit(1)


# -------------------------------------------------------
# 1. Load the small lookup tables first — these are tiny
#    and we want them in memory before the big joins.
# -------------------------------------------------------
print("=" * 60)
print("Instacart Basket Shrinkage — 01 Load & Explore")
print("=" * 60)

print("\nLoading lookup tables...")

for path in [config.PRODUCTS, config.AISLES, config.DEPARTMENTS,
             config.ORDERS, config.ORDER_PRODUCTS_TRAIN, config.ORDER_PRODUCTS_PRIOR]:
    require_file(path)

products    = pd.read_csv(config.PRODUCTS)
aisles      = pd.read_csv(config.AISLES)
departments = pd.read_csv(config.DEPARTMENTS)

print(f"  products:    {len(products):>7,} rows")
print(f"  aisles:      {len(aisles):>7,} rows")
print(f"  departments: {len(departments):>7,} rows")


# -------------------------------------------------------
# 2. Orders — medium-sized, straightforward load.
#    days_since_prior_order has nulls for first orders —
#    that's expected, not dirty data.
# -------------------------------------------------------
print("\nLoading orders... ")
orders = pd.read_csv(config.ORDERS)
print(f"  Got {len(orders):,} orders from {orders['user_id'].nunique():,} customers")

# Sanity check on the eval_set column — tells us which users
# are in train vs test vs prior history
eval_counts = orders['eval_set'].value_counts()
print(f"  eval_set breakdown: {dict(eval_counts)}")

# The date column doesn't exist directly — Instacart gives us
# order_dow (day of week) and order_hour_of_day instead.
# days_since_prior_order is what we'll use to reconstruct
# a relative timeline per customer.
null_days = orders['days_since_prior_order'].isna().sum()
pct_null  = null_days / len(orders) * 100
print(f"  days_since_prior_order nulls: {null_days:,} ({pct_null:.1f}%) — these are first orders, expected")


# -------------------------------------------------------
# 3. order_products__train — the labeled set.
#    Smaller than prior, loads fine in one shot.
# -------------------------------------------------------
print("\nLoading order_products__train...")
op_train = pd.read_csv(config.ORDER_PRODUCTS_TRAIN)
print(f"  {len(op_train):,} order-product rows")


# -------------------------------------------------------
# 4. order_products__prior — this is the big one at ~577MB.
#    Load in chunks and concat. We cast to smaller dtypes
#    upfront to cut memory roughly in half.
#
#    Why int32 instead of int64? These IDs top out around
#    ~50M which fits easily in int32. Saves ~200MB.
# -------------------------------------------------------
print("\nLoading order_products__prior... this one's big, give it a sec")

dtype_map = {
    'order_id':            'int32',
    'product_id':          'int32',
    'add_to_cart_order':   'int16',
    'reordered':           'int8',
}

chunks = []
rows_loaded = 0
for chunk in pd.read_csv(config.ORDER_PRODUCTS_PRIOR,
                         dtype=dtype_map,
                         chunksize=config.CHUNK_SIZE):
    chunks.append(chunk)
    rows_loaded += len(chunk)
    # Print a dot every million rows so we know it's still alive
    if rows_loaded % 1_000_000 == 0:
        print(f"  ...{rows_loaded:,} rows loaded so far")

op_prior = pd.concat(chunks, ignore_index=True)
del chunks   # free memory immediately — we don't need the list anymore

print(f"  Done — {len(op_prior):,} rows, {op_prior['order_id'].nunique():,} unique orders")
print(f"  Memory: {op_prior.memory_usage(deep=True).sum() / 1e6:.0f} MB")


# -------------------------------------------------------
# 5. Stack train + prior into one product-level table.
#    We want the full purchase history per customer, not
#    just the training snapshot.
# -------------------------------------------------------
print("\nCombining prior and train order-product rows...")
op_all = pd.concat([op_prior, op_train], ignore_index=True)
print(f"  Combined: {len(op_all):,} order-product rows")
del op_prior, op_train   # done with the pieces


# -------------------------------------------------------
# 6. Build the master table.
#    Join chain: order-products → orders → products →
#    aisles → departments
#
#    We want one row per (user, order, product) with all
#    the context fields attached. This is what every
#    downstream script will start from.
# -------------------------------------------------------
print("\nBuilding master order table...")

# Bring in product names and their aisle/dept IDs
op_enriched = op_all.merge(
    products[['product_id', 'product_name', 'aisle_id', 'department_id']],
    on='product_id',
    how='left'
)

# Attach aisle labels
op_enriched = op_enriched.merge(
    aisles[['aisle_id', 'aisle']],
    on='aisle_id',
    how='left'
)

# Attach department labels — this is the grouping level we care
# about most for the category exposure analysis in script 05
op_enriched = op_enriched.merge(
    departments[['department_id', 'department']],
    on='department_id',
    how='left'
)

# Now pull in the order-level context: user, day, hour, sequence
master = op_enriched.merge(
    orders[['order_id', 'user_id', 'order_number', 'order_dow',
            'order_hour_of_day', 'days_since_prior_order', 'eval_set']],
    on='order_id',
    how='left'
)

print(f"  Master table: {len(master):,} rows x {master.shape[1]} columns")
print(f"  Customers: {master['user_id'].nunique():,}")
print(f"  Orders:    {master['order_id'].nunique():,}")
print(f"  Products:  {master['product_id'].nunique():,}")


# -------------------------------------------------------
# 7. Quick data quality pass — print null counts for
#    anything that looks suspicious. A few nulls in
#    days_since_prior_order are fine (first orders).
#    Nulls in user_id or product_id are a problem.
# -------------------------------------------------------
print("\nNull check on key columns:")
key_cols = ['user_id', 'order_id', 'product_id', 'department',
            'aisle', 'order_number', 'days_since_prior_order']
for col in key_cols:
    n = master[col].isna().sum()
    flag = "  <-- check this" if (n > 0 and col not in ['days_since_prior_order']) else ""
    print(f"  {col:<30} {n:>10,} nulls{flag}")


# -------------------------------------------------------
# 8. Basket-level summary — collapse to one row per order
#    so we can see what average basket sizes look like.
# -------------------------------------------------------
print("\nBasket size summary (items per order):")
basket_sizes = master.groupby('order_id')['product_id'].count()
print(f"  Mean:   {basket_sizes.mean():.1f} items")
print(f"  Median: {basket_sizes.median():.0f} items")
print(f"  Min:    {basket_sizes.min()} items")
print(f"  Max:    {basket_sizes.max()} items")
print(f"  Std:    {basket_sizes.std():.1f}")


# -------------------------------------------------------
# 9. Order cadence — how often do customers shop?
#    Exclude NaN (first orders have no prior interval).
# -------------------------------------------------------
print("\nDays between orders (excluding first orders):")
cadence = orders['days_since_prior_order'].dropna()
print(f"  Mean:   {cadence.mean():.1f} days")
print(f"  Median: {cadence.median():.0f} days")
print(f"  Mode:   {cadence.mode()[0]:.0f} days")


# -------------------------------------------------------
# 10. Orders per customer distribution — gives us a sense
#     of how many customers have enough history for the
#     regression analysis in script 03.
# -------------------------------------------------------
print("\nOrders per customer distribution:")
orders_per_cust = orders.groupby('user_id')['order_number'].max()
print(f"  Mean:     {orders_per_cust.mean():.1f} orders")
print(f"  Median:   {orders_per_cust.median():.0f} orders")
print(f"  >= 5 orders (needed for regression): "
      f"{(orders_per_cust >= 5).sum():,} customers "
      f"({(orders_per_cust >= 5).mean() * 100:.1f}%)")
print(f"  >= 10 orders: {(orders_per_cust >= 10).sum():,} customers")


# -------------------------------------------------------
# 11. Department distribution — top 10 by order volume.
#     Good sanity check before the category analysis.
# -------------------------------------------------------
print("\nTop 10 departments by order-product rows:")
dept_counts = master['department'].value_counts().head(10)
for dept, cnt in dept_counts.items():
    print(f"  {dept:<25} {cnt:>10,}")


# -------------------------------------------------------
# 12. Reorder rate — what fraction of items are repeat buys?
#     High reorder rates mean loyal customers have sticky
#     habits, which matters for the shrinkage analysis.
# -------------------------------------------------------
reorder_rate = master['reordered'].mean()
print(f"\nOverall reorder rate: {reorder_rate:.1%}  "
      f"(of all items placed, this fraction was a repeat purchase)")


# -------------------------------------------------------
# 13. Save the master file.
#     This is the only large write in this script —
#     everything downstream reads from here.
# -------------------------------------------------------
print(f"\nSaving master_orders.csv to {config.MASTER_ORDERS}...")
os.makedirs(config.PROCESSED_DIR, exist_ok=True)

master.to_csv(config.MASTER_ORDERS, index=False)

size_mb = os.path.getsize(config.MASTER_ORDERS) / 1e6
print(f"  Written — {size_mb:.0f} MB on disk")


# -------------------------------------------------------
# 14. Final summary
# -------------------------------------------------------
print("\n" + "=" * 60)
print("ORIENTATION COMPLETE")
print("=" * 60)
print(f"  {master['user_id'].nunique():,} customers")
print(f"  {master['order_id'].nunique():,} orders")
print(f"  {len(master):,} order-product rows")
print(f"  {master['department'].nunique()} departments")
print(f"  Avg basket size: {basket_sizes.mean():.1f} items")
print(f"  Avg order gap:   {cadence.mean():.1f} days")
print(f"  Reorder rate:    {reorder_rate:.1%}")
print()
print("Looks good. Next up — run pipeline/02_rfm.py")
