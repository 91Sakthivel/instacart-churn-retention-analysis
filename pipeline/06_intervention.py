# ============================================================
# 04_intervention.py — Intervention Timing & Last-Hook Analysis
# ============================================================
# Two questions answered here:
#
#   1. WHEN is it too late?
#      Look at every customer's order history, find consecutive
#      declining-basket streaks, and measure what fraction of
#      customers placed another order after each streak length.
#      The point where that recovery rate drops below 20% is
#      the intervention deadline.
#
#   2. WHAT are they still buying?
#      For churned (Lost) customers, find the department they
#      were purchasing from most in their last 3 orders. That's
#      their last hook — a promotion in that category is a far
#      better re-engagement lever than a generic coupon.
#
# The streak analysis runs on ALL customers with enough history
# (not just Lost) to get a reliable statistical base.
# ============================================================

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

import pandas as pd
import numpy as np


RECOVERY_THRESHOLD = 0.20   # flag deadline where recovery drops below this
MIN_OBS_PER_STREAK = 50     # ignore streak lengths with fewer observations (noisy)


# -------------------------------------------------------
# 1. Check dependencies
# -------------------------------------------------------
print("=" * 60)
print("Instacart Basket Shrinkage — 04 Intervention Timing")
print("=" * 60)

for path, name in [(config.MASTER_ORDERS, 'master_orders.csv'),
                   (config.RFM_SEGMENTS,   'rfm_segments.csv')]:
    if not os.path.exists(path):
        print(f"\n  ERROR: {path} not found.")
        print(f"  Run the earlier pipeline scripts first.")
        sys.exit(1)

rfm = pd.read_csv(config.RFM_SEGMENTS, usecols=['user_id', 'segment'])
print(f"\nRFM segments loaded: {len(rfm):,} customers")
print(f"  Segment counts: {rfm['segment'].value_counts().to_dict()}")

lost_users = set(rfm[rfm['segment'] == 'Lost']['user_id'].values)
print(f"  Lost customers (churn analysis target): {len(lost_users):,}")


# -------------------------------------------------------
# 2. Load master_orders — numeric + department.
#    Department as 'category' dtype saves significant RAM
#    vs object (21 unique values across 33M rows).
# -------------------------------------------------------
print("\nLoading master_orders.csv (with department column)...")
print("  This one has a string column so it's a bit heavier — give it a moment")

master = pd.read_csv(
    config.MASTER_ORDERS,
    usecols=['user_id', 'order_id', 'order_number', 'days_since_prior_order', 'department'],
    dtype={
        'user_id':      'int32',
        'order_id':     'int32',
        'order_number': 'int16',
        'department':   'category',
    }
)
print(f"  {len(master):,} rows | memory: {master.memory_usage(deep=True).sum() / 1e6:.0f} MB")


# -------------------------------------------------------
# 3. Basket size per order + collapse to order level
# -------------------------------------------------------
print("\nComputing basket sizes...")

basket_sizes = master.groupby('order_id').size().reset_index(name='basket_size')

orders = (master
          .drop_duplicates(subset='order_id')
          [['user_id', 'order_id', 'order_number', 'days_since_prior_order']]
          .merge(basket_sizes, on='order_id'))

print(f"  Order-level: {len(orders):,} rows")


# -------------------------------------------------------
# 4. Streak analysis — consecutive declining basket orders
#
#    For each order, we compute how many consecutive
#    declining orders precede it (including itself).
#
#    Vectorized approach:
#      is_declining = (basket < prev_basket)
#      reset_point  = ~is_declining  (True where streak breaks)
#      group_id     = cumsum of reset_point within each user
#                     — increments at each reset, so declining
#                     runs share a group_id with the reset point
#                     that started them
#      streak       = cumcount within (user, group_id)
#                     — gives 0 at the reset, 1/2/3 for declines
# -------------------------------------------------------
print("\nComputing consecutive decline streaks across all orders...")

orders_s = orders.sort_values(['user_id', 'order_number']).reset_index(drop=True)

orders_s['prev_basket'] = orders_s.groupby('user_id')['basket_size'].shift(1)
orders_s['is_declining'] = (
    orders_s['basket_size'] < orders_s['prev_basket']
).fillna(False)

# reset_point is True at the start of every non-declining run
orders_s['reset_point'] = ~orders_s['is_declining']

# group_id increments each time we hit a reset — consecutive declining
# rows share the group_id of the reset that began their run
orders_s['group_id'] = (orders_s
    .groupby('user_id')['reset_point']
    .cumsum())

# cumcount within (user, group_id) = streak length at each position
orders_s['decline_streak'] = orders_s.groupby(
    ['user_id', 'group_id']
).cumcount()

# has_future_order: not the last recorded order for this customer
max_order_num = orders_s.groupby('user_id')['order_number'].transform('max')
orders_s['has_future_order'] = orders_s['order_number'] < max_order_num

print(f"  Streak computation complete")
print(f"  Orders with a declining streak >= 1: "
      f"{(orders_s['decline_streak'] >= 1).sum():,}")


# -------------------------------------------------------
# 5. Recovery rate by streak length
#    Among orders where the customer was N deep into a
#    consecutive decline, what fraction ever ordered again?
# -------------------------------------------------------
print("\nComputing recovery rates by streak length...")

recovery = (orders_s[orders_s['decline_streak'] >= 1]
            .groupby('decline_streak')['has_future_order']
            .agg(recovery_rate='mean', n_observations='count')
            .reset_index())

# Drop streak lengths with too few observations — rates are unreliable there
recovery = recovery[recovery['n_observations'] >= MIN_OBS_PER_STREAK].copy()
recovery['recovery_pct'] = recovery['recovery_rate'] * 100

print("\n  Recovery rate by consecutive declining orders:")
print(f"  {'Streak':>6}  {'Recovery':>10}  {'Observations':>14}")
print("  " + "-" * 36)
for _, row in recovery.iterrows():
    flag = " <-- deadline" if row['recovery_rate'] < RECOVERY_THRESHOLD else ""
    print(f"  {int(row['decline_streak']):>6}  "
          f"{row['recovery_pct']:>9.1f}%  "
          f"{int(row['n_observations']):>14,}{flag}")


# -------------------------------------------------------
# 6. Find the intervention deadline
# -------------------------------------------------------
below_threshold = recovery[recovery['recovery_rate'] < RECOVERY_THRESHOLD]

if len(below_threshold) > 0:
    intervention_deadline = int(below_threshold['decline_streak'].min())
    deadline_recovery_rate = below_threshold.loc[
        below_threshold['decline_streak'] == intervention_deadline, 'recovery_pct'
    ].values[0]
    print(f"\n  INTERVENTION DEADLINE: {intervention_deadline} consecutive declining orders")
    print(f"  At that point, recovery rate = {deadline_recovery_rate:.1f}%")
    print(f"  After {intervention_deadline} declining orders, {100 - deadline_recovery_rate:.1f}% "
          f"of customers never order again")
else:
    intervention_deadline = None
    print(f"\n  Recovery rate stays above {RECOVERY_THRESHOLD:.0%} across all streak lengths observed")
    print("  Intervention window is broader than this dataset shows")


# -------------------------------------------------------
# 7. Last-hook department analysis — Lost customers only
#
#    For each churned customer, find their most-purchased
#    department across their last 3 orders. This is the
#    product category they were still engaged with right
#    before they went quiet — the best re-engagement hook.
# -------------------------------------------------------
print("\nFinding last-hook departments for churned customers...")

# Filter master down to Lost users and the dept column we need
master_lost = master[master['user_id'].isin(lost_users)][
    ['user_id', 'order_id', 'order_number', 'department']
].copy()

# Get the order_ids for each Lost user's last 3 orders
last3_orders = (master_lost
                .drop_duplicates(subset=['user_id', 'order_id'])
                [['user_id', 'order_id', 'order_number']]
                .sort_values(['user_id', 'order_number'])
                .groupby('user_id')
                .tail(3))

last3_order_ids = set(last3_orders['order_id'].values)

# Items bought in those last 3 orders — grouped by department
last3_items = master_lost[master_lost['order_id'].isin(last3_order_ids)].copy()

# Most frequent department per customer in their last 3 orders
hook_dept = (last3_items
             .groupby(['user_id', 'department'], observed=True)
             .size()
             .reset_index(name='item_count')
             .sort_values(['user_id', 'item_count'], ascending=[True, False])
             .drop_duplicates(subset='user_id')
             [['user_id', 'department']]
             .rename(columns={'department': 'last_hook_department'}))

print(f"  Hook departments computed for {len(hook_dept):,} churned customers")

# What are the most common hooks across the Lost segment?
print("\n  Most common last-hook departments:")
hook_counts = hook_dept['last_hook_department'].value_counts().head(8)
for dept, cnt in hook_counts.items():
    pct = cnt / len(hook_dept) * 100
    print(f"    {str(dept):<25} {cnt:>6,}  ({pct:.1f}%)")


# -------------------------------------------------------
# 8. Max decline streak per churned customer at their
#    last recorded order — tells us how deep in the hole
#    they were when we lost them.
# -------------------------------------------------------
last_order_streaks = (orders_s
    .sort_values(['user_id', 'order_number'])
    .groupby('user_id')[['order_number', 'decline_streak', 'basket_size']]
    .last()
    .reset_index()
    .rename(columns={
        'decline_streak': 'streak_at_last_order',
        'basket_size': 'last_basket_size',
        'order_number': 'total_orders'
    }))

lost_streaks = last_order_streaks[last_order_streaks['user_id'].isin(lost_users)]


# -------------------------------------------------------
# 9. Build intervention_report.csv
# -------------------------------------------------------
intervention_report = (rfm[rfm['segment'] == 'Lost']
    .merge(hook_dept, on='user_id', how='left')
    .merge(lost_streaks[['user_id', 'total_orders', 'streak_at_last_order', 'last_basket_size']],
           on='user_id', how='left'))

intervention_report['intervention_deadline_orders'] = intervention_deadline

print(f"\nBuilt intervention report: {len(intervention_report):,} churned customers")

print("\nSample stats on churned customers:")
print(f"  Avg total orders before churning:  {intervention_report['total_orders'].mean():.1f}")
print(f"  Avg streak at last order:          {intervention_report['streak_at_last_order'].mean():.1f}")
pct_past_deadline = (
    (intervention_report['streak_at_last_order'] >= (intervention_deadline or 99)).mean() * 100
)
if intervention_deadline:
    print(f"  % who were past the intervention deadline at last order: {pct_past_deadline:.1f}%")

os.makedirs(config.PROCESSED_DIR, exist_ok=True)
intervention_report.to_csv(config.INTERVENTION_REPORT, index=False)
print(f"\nSaved: {config.INTERVENTION_REPORT}")


# -------------------------------------------------------
# 10. Wrap up
# -------------------------------------------------------
top_hook = hook_counts.index[0] if len(hook_counts) > 0 else "unknown"

print("\n" + "=" * 60)
print("INTERVENTION ANALYSIS COMPLETE")
print("=" * 60)
if intervention_deadline:
    print(f"  Intervention deadline: {intervention_deadline} consecutive declining orders")
    print(f"  After that, <{RECOVERY_THRESHOLD:.0%} of customers ever return")
print(f"  Top last-hook department: {top_hook}")
print(f"  {len(intervention_report):,} churned customers profiled")
print()
print("Next up — run pipeline/05_category_exposure.py")
