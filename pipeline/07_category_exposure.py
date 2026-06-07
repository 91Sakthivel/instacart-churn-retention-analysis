# ============================================================
# 07_category_exposure.py — Category Revenue Exposure + RQ3
# Original shrinkage analysis preserved.
# Department residual analysis (Block 13) added at bottom.
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

AVG_ITEM_PRICE = 3.50
HORIZON_DAYS   = 60
TOP_N_DEPTS    = 12

print("=" * 60)
print("Instacart Basket Shrinkage -- 07 Category Exposure")
print("=" * 60)

for path, name in [(config.RFM_SEGMENTS, 'rfm_segments.csv'),
                   (config.CUSTOMER_FEATURES, 'customer_features.csv'),
                   (config.MASTER_ORDERS, 'master_orders.csv')]:
    if not os.path.exists(path):
        print(f"\n  ERROR: {path} not found.")
        sys.exit(1)

rfm = pd.read_csv(config.RFM_SEGMENTS, usecols=['user_id', 'segment'])

# Check if old shrinkage columns are still present
_cf_cols = pd.read_csv(config.CUSTOMER_FEATURES, nrows=0).columns.tolist()
_old_cols = ['slope', 'shrinking_flag', 'last_basket_size']
_SKIP_OLD = not all(c in _cf_cols for c in _old_cols)

if _SKIP_OLD:
    print("\n  NOTE: customer_features.csv updated to validated-signal format.")
    print("  Original $1.9M shrinkage finding preserved in category_risk_report.csv.")
    print("  Skipping shrinkage-based category analysis. Running residual analysis.")
else:
    features = pd.read_csv(config.CUSTOMER_FEATURES,
                           usecols=['user_id', 'slope', 'avg_days_between_orders',
                                    'shrinking_flag', 'last_basket_size'])
    print(f"\nRFM: {len(rfm):,} customers")
    print(f"Features: {len(features):,} customers with regression data")

    top_tiers = rfm[rfm['segment'].isin(['Champions', 'Loyal'])].copy()
    at_risk = top_tiers.merge(
        features[features['shrinking_flag'] == True], on='user_id', how='inner')
    print(f"\nAt-risk high-value customers (Champions/Loyal + shrinking):")
    print(f"  {at_risk['segment'].value_counts().to_dict()}")
    print(f"  Total: {len(at_risk):,}")

    if len(at_risk) > 0:
        N = (HORIZON_DAYS / at_risk['avg_days_between_orders'].clip(lower=1)).values
        at_risk = at_risk.copy()
        at_risk['orders_in_60d']       = N
        at_risk['items_at_risk_60d']   = np.abs(at_risk['slope']) * N * (N + 1) / 2
        at_risk['revenue_at_risk_60d'] = at_risk['items_at_risk_60d'] * AVG_ITEM_PRICE
        total_rev_risk = at_risk['revenue_at_risk_60d'].sum()
        champ_risk = at_risk[at_risk['segment'] == 'Champions']['revenue_at_risk_60d'].sum()
        loyal_risk = at_risk[at_risk['segment'] == 'Loyal']['revenue_at_risk_60d'].sum()
        print(f"\nProjecting 60-day revenue at risk...")
        print(f"  Total: ${total_rev_risk:,.0f}  Champions: ${champ_risk:,.0f}  Loyal: ${loyal_risk:,.0f}")

        at_risk_ids = set(at_risk['user_id'].values)
        print("\nLoading purchase history for at-risk customers...")
        master = pd.read_csv(config.MASTER_ORDERS,
                             usecols=['user_id', 'order_id', 'department'],
                             dtype={'user_id': 'int32', 'order_id': 'int32',
                                    'department': 'category'})
        master = master[master['user_id'].isin(at_risk_ids)].copy()

        dept_items = (master.groupby(['user_id', 'department'], observed=True)
                      .size().reset_index(name='items_in_dept'))
        total_items = (dept_items.groupby('user_id')['items_in_dept']
                       .sum().reset_index(name='total_items'))
        dept_items = dept_items.merge(total_items, on='user_id')
        dept_items['dept_fraction'] = dept_items['items_in_dept'] / dept_items['total_items']

        dept_risk = dept_items.merge(
            at_risk[['user_id', 'segment', 'revenue_at_risk_60d']], on='user_id', how='left')
        dept_risk['dept_revenue_at_risk'] = (
            dept_risk['revenue_at_risk_60d'] * dept_risk['dept_fraction'])

        category_exposure = (dept_risk
            .groupby(['department', 'segment'], observed=True)
            .agg(total_revenue_at_risk=('dept_revenue_at_risk', 'sum'),
                 n_customers=('user_id', 'nunique'))
            .reset_index())
        dept_totals = (category_exposure
            .groupby('department', observed=True)['total_revenue_at_risk'].sum()
            .reset_index()
            .rename(columns={'total_revenue_at_risk': 'dept_total_risk'})
            .sort_values('dept_total_risk', ascending=False))
        top_depts = dept_totals.head(TOP_N_DEPTS)['department'].tolist()
        category_exposure_top = category_exposure[
            category_exposure['department'].isin(top_depts)].copy()

        print(f"\n  Top {TOP_N_DEPTS} departments by 60-day revenue at risk:")
        print(f"  {'Department':<25}  {'Risk $':>10}  {'Customers':>10}")
        print("  " + "-" * 50)
        for _, row in dept_totals.head(TOP_N_DEPTS).iterrows():
            n = category_exposure[
                category_exposure['department'] == row['department']]['n_customers'].sum()
            print(f"  {str(row['department']):<25}  ${row['dept_total_risk']:>9,.0f}  {int(n):>10,}")

        category_risk_report = (dept_totals
            .merge(category_exposure.pivot_table(
                index='department', columns='segment',
                values='total_revenue_at_risk', aggfunc='sum',
                fill_value=0, observed=True).reset_index(),
                on='department', how='left')
            .merge(category_exposure.groupby('department', observed=True)
                   ['n_customers'].sum().reset_index(), on='department', how='left'))
        category_risk_report['pct_of_total_risk'] = (
            category_risk_report['dept_total_risk'] /
            category_risk_report['dept_total_risk'].sum() * 100)
        os.makedirs(config.PROCESSED_DIR, exist_ok=True)
        category_risk_report.to_csv(config.CATEGORY_RISK_REPORT, index=False)
        print(f"\nSaved: {config.CATEGORY_RISK_REPORT}")

        pivot = category_exposure_top.pivot_table(
            index='department', columns='segment',
            values='total_revenue_at_risk', aggfunc='sum', fill_value=0, observed=True)
        for seg in ['Champions', 'Loyal']:
            if seg not in pivot.columns:
                pivot[seg] = 0.0
        if 'Champions' in pivot.columns and 'Loyal' in pivot.columns:
            pivot = pivot[['Champions', 'Loyal']]
        pivot['_total'] = pivot.sum(axis=1)
        pivot = pivot.sort_values('_total', ascending=False).drop('_total', axis=1)
        pivot_k = pivot.head(TOP_N_DEPTS) / 1000
        fig, ax = plt.subplots(figsize=(8, 7))
        sns.heatmap(pivot_k, annot=True, fmt='.1f', cmap='YlOrRd',
                    linewidths=0.5, linecolor='white', ax=ax,
                    cbar_kws={'label': 'Revenue at Risk ($000s, 60-day)'},
                    annot_kws={'size': 10})
        ax.set_title("60-Day Revenue at Risk by Department\nChampions & Loyal with Basket Shrinkage",
                     fontsize=12, fontweight='bold', pad=16)
        ax.set_xlabel("Customer Segment", fontsize=11)
        ax.set_ylabel("Department", fontsize=11)
        plt.tight_layout()
        os.makedirs(config.CHARTS_DIR, exist_ok=True)
        plt.savefig(config.CATEGORY_HEATMAP, dpi=config.CHART_DPI, bbox_inches='tight')
        plt.close()
        print(f"  Heatmap saved: {config.CATEGORY_HEATMAP}")

        top_dept = dept_totals.iloc[0]['department'] if len(dept_totals) > 0 else "n/a"
        top_dept_risk = dept_totals.iloc[0]['dept_total_risk'] if len(dept_totals) > 0 else 0
        print("\n" + "=" * 60)
        print("CATEGORY EXPOSURE COMPLETE")
        print(f"  ${total_rev_risk:,.0f} total 60-day revenue at risk")
        print(f"  {len(at_risk):,} high-value customers")
        print(f"  Most exposed category: {top_dept} (${top_dept_risk:,.0f})")
        print("=" * 60)


# ================================================================
# BLOCK 13 -- Department Residual Analysis (RQ3)
# ================================================================
print("\n" + "=" * 60)
print("BLOCK 13: Department Residual Analysis")
print("=" * 60)

try:
    from xgboost import XGBClassifier
    HAS_XGB_13 = True
except ImportError:
    from sklearn.ensemble import GradientBoostingClassifier as XGBClassifier
    HAS_XGB_13 = False

print("\nLoading customer_features.csv for residual analysis...")
cf13 = pd.read_csv(config.CUSTOMER_FEATURES)
print(f"  Loaded: {len(cf13):,} customers")

# Find primary department per customer from master_orders
print("Loading master_orders.csv (dept column only)...")
mo13 = pd.read_csv(
    config.MASTER_ORDERS,
    usecols=['user_id', 'order_id', 'department'],
    dtype={'user_id': 'int32', 'order_id': 'int32'}
)
top_dept_per_user = (
    mo13.groupby(['user_id', 'department'], sort=False)['order_id'].nunique()
    .reset_index()
)
top_dept_per_user = (
    top_dept_per_user.sort_values(['user_id', 'order_id'], ascending=[True, False])
    .groupby('user_id', sort=False).first().reset_index()
    [['user_id', 'department']]
    .rename(columns={'department': 'primary_department'})
)
cf13 = cf13.merge(top_dept_per_user, on='user_id', how='left')
print(f"  Primary department assigned for {cf13['primary_department'].notna().sum():,} customers")

# Features WITHOUT any department signal
residual_feats = [c for c in [
    'avg_basket_size', 'avg_days_between_orders', 'order_gap_cv',
    'gap_trend', 'reorder_rate', 'order_count', 'days_since_last_order',
    'annual_value_est',
] if c in cf13.columns]

cf13_model = cf13[cf13['churned'].notna() & cf13['primary_department'].notna()].copy()
X13 = cf13_model[residual_feats].fillna(0).values
y13 = cf13_model['churned'].values

print(f"\nTraining residual model on {len(X13):,} customers (no department features)...")
if HAS_XGB_13:
    xgb13 = XGBClassifier(n_estimators=200, learning_rate=0.05, max_depth=4,
                            random_state=config.RANDOM_STATE, verbosity=0,
                            eval_metric='logloss')
else:
    xgb13 = XGBClassifier(n_estimators=200, max_depth=4,
                            random_state=config.RANDOM_STATE)
xgb13.fit(X13, y13)
cf13_model = cf13_model.copy()
cf13_model['predicted_churn_prob'] = xgb13.predict_proba(X13)[:, 1]

residual_df = (
    cf13_model.groupby('primary_department')
    .agg(
        customer_count=('user_id', 'count'),
        actual_churn_rate=('churned', 'mean'),
        predicted_churn_rate=('predicted_churn_prob', 'mean'),
    )
    .reset_index()
)
residual_df['residual'] = residual_df['actual_churn_rate'] - residual_df['predicted_churn_rate']
residual_df['status'] = residual_df['residual'].apply(
    lambda r: 'Higher than expected' if r > 0.02 else
              'Lower than expected' if r < -0.02 else 'As expected'
)
residual_df = residual_df.sort_values('residual', ascending=False).reset_index(drop=True)

print("\nDepartment residual analysis:")
print(f"  {'Department':<22}|{'Customers':>10}|{'Actual%':>8}|{'Pred%':>8}|{'Residual':>10}|{'Status'}")
print("  " + "-" * 80)
for _, row in residual_df.iterrows():
    print(f"  {str(row['primary_department']):<22}|{int(row['customer_count']):>10,}|"
          f"{row['actual_churn_rate']*100:>7.1f}%|{row['predicted_churn_rate']*100:>7.1f}%|"
          f"{row['residual']*100:>+9.1f}%| {row['status']}")
print("  " + "-" * 80)

higher = residual_df[residual_df['status'] == 'Higher than expected']['primary_department'].tolist()
lower  = residual_df[residual_df['status'] == 'Lower than expected']['primary_department'].tolist()
print(f"\nDepartments churning MORE than model expects:")
print(f"  {', '.join(higher) if higher else 'None'} -- investigate operational issues here")
print(f"\nDepartments churning LESS -- hidden stable segments:")
print(f"  {', '.join(lower) if lower else 'None'} -- more loyal than behavioral model suggests")

biggest = residual_df.iloc[0]
print(f"\nBiggest surprise: '{biggest['primary_department']}' shows "
      f"{biggest['residual']*100:+.1f}% {'higher' if biggest['residual'] > 0 else 'lower'} churn")
print(f"  than behavioral model predicts -- something beyond purchase patterns drives this")

# Save
residual_df.to_csv(config.DEPARTMENT_RESIDUALS, index=False)
print(f"\nSaved: {config.DEPARTMENT_RESIDUALS}")

# Chart
fig, ax = plt.subplots(figsize=(12, 6))
colors_r = ['#d32f2f' if r > 0.02 else '#2e7d32' if r < -0.02 else '#1565c0'
            for r in residual_df['residual']]
ax.bar(residual_df['primary_department'], residual_df['residual'] * 100, color=colors_r)
ax.axhline(0, color='black', lw=0.8)
ax.axhline(2, color='red', linestyle='--', lw=1, alpha=0.5, label='+2% threshold')
ax.axhline(-2, color='green', linestyle='--', lw=1, alpha=0.5, label='-2% threshold')
ax.set_xlabel('Primary Department')
ax.set_ylabel('Churn Residual (Actual - Predicted) %')
ax.set_title('Department Churn Residuals\n(above 0 = churns more than behavior predicts)')
ax.tick_params(axis='x', rotation=30)
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(config.CHARTS_DIR, 'department_residuals.png'), dpi=config.CHART_DPI)
plt.close()
print(f"Saved: {os.path.join(config.CHARTS_DIR, 'department_residuals.png')}")

print("\n" + "=" * 60)
print("RQ3 complete. Department residuals saved.")
print("Next -- pipeline/09_executive_summary.py")
print("=" * 60)
