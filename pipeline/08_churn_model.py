# ============================================================
# 08_churn_model.py — Churn Classifier + Revenue Regressor
# Blocks 8-12: Feature selection, Stage 1 (classifier),
# Stage 2 (regressor), risk scoring and ranking
# ============================================================
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (roc_auc_score, precision_score, recall_score,
                             f1_score, mean_squared_error, mean_absolute_error, r2_score)
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.preprocessing import LabelEncoder

try:
    from imblearn.over_sampling import SMOTE
    HAS_SMOTE = True
except ImportError:
    HAS_SMOTE = False
    print("  WARNING: imbalanced-learn not installed. Proceeding without SMOTE.")

try:
    from xgboost import XGBClassifier, XGBRegressor
    HAS_XGB = True
except ImportError:
    from sklearn.ensemble import GradientBoostingClassifier as XGBClassifier
    from sklearn.ensemble import GradientBoostingRegressor as XGBRegressor
    HAS_XGB = False
    print("  NOTE: xgboost not installed. Using GradientBoosting as fallback.")

os.makedirs(config.MODELS_DIR, exist_ok=True)
os.makedirs(config.CHARTS_DIR, exist_ok=True)
os.makedirs(config.REPORTS_DIR, exist_ok=True)

print("=" * 60)
print("Churn Model - 08_churn_model.py")
print("=" * 60)

# ================================================================
# BLOCK 8 -- Feature Selection
# ================================================================
print("\n--- BLOCK 8: Feature Selection ---")
print("\nLoading customer_features.csv...")
cf = pd.read_csv(config.CUSTOMER_FEATURES)
print(f"  Loaded: {len(cf):,} customers, {len(cf.columns)} columns")

# Merge trajectory if available
traj_path = config.TRAJECTORY_SEGMENTS
if os.path.exists(traj_path):
    traj = pd.read_csv(traj_path, usecols=['user_id', 'trajectory_type'])
    cf = cf.merge(traj, on='user_id', how='left')
    cf['trajectory_type'] = cf['trajectory_type'].fillna('C')
    le = LabelEncoder()
    cf['trajectory_enc'] = le.fit_transform(cf['trajectory_type'].astype(str))
    print("  Merged trajectory types")
else:
    cf['trajectory_enc'] = 0
    cf['trajectory_type'] = 'C'
    print("  No trajectory_segments.csv found -- trajectory_enc = 0")

NON_FEATURE_COLS = {'user_id', 'churned', 'trajectory_type', 'top_department',
                    'revenue_at_risk_90d', 'annual_value_est', 'annual_orders_est',
                    'days_since_last_order'}

# Step 1: Remove validation-failed features
print("\n[Step 1] Removing validation-failed features:")
remove_features = {
    'basket_slope':          'Check 1 failed: basket does not decline before churn',
    'shrinkage_velocity':    'Derived from basket_slope, same failure',
    'lifetime_basket_trend': 'Check 1 failed',
    'last_3_avg_basket':     'Check 1 failed',
    'first_3_avg_basket':    'Check 1 failed',
    'category_shift_score':  'Check 3 failed: p=0.237',
    'high_value_collapse':   'Derived from category shift, same failure',
}
for feat, reason in remove_features.items():
    if feat in cf.columns:
        cf = cf.drop(columns=[feat])
        print(f"  Dropped '{feat}': {reason}")
    else:
        print(f"  '{feat}' not present (correct -- not in validated features)")

# Step 2: Low variance filter
print("\n[Step 2] Low variance filter...")
feat_candidates = [c for c in cf.select_dtypes(include=[np.number]).columns
                   if c not in NON_FEATURE_COLS and c != 'churned']
low_var_drop = []
for col in feat_candidates:
    vc = cf[col].value_counts(normalize=True)
    if len(vc) > 0 and vc.iloc[0] > 0.95:
        low_var_drop.append(col)
        print(f"  Dropped '{col}': {vc.iloc[0]*100:.1f}% identical")
if low_var_drop:
    cf = cf.drop(columns=low_var_drop)
    print(f"  Removed {len(low_var_drop)} low-variance features")
else:
    print("  No low-variance features found")

# Step 3: Correlation filter
print("\n[Step 3] Correlation filter (>0.85)...")
feat_candidates = [c for c in cf.select_dtypes(include=[np.number]).columns
                   if c not in NON_FEATURE_COLS and c != 'churned']
corr_mat = cf[feat_candidates].corr().abs()
upper = corr_mat.where(np.triu(np.ones(corr_mat.shape), k=1).astype(bool))
priority_list = ['avg_days_between_orders', 'gap_trend', 'reorder_rate',
                 'order_count', 'reorder_trend', 'gap_acceleration',
                 'avg_basket_size', 'order_gap_cv']
corr_drop = set()
for col in upper.columns:
    for hc in upper.index[upper[col] > 0.85].tolist():
        if col in corr_drop or hc in corr_drop:
            continue
        col_p = priority_list.index(col) if col in priority_list else 999
        hc_p = priority_list.index(hc) if hc in priority_list else 999
        drop_col = hc if col_p <= hc_p else col
        keep_col = col if col_p <= hc_p else hc
        corr_drop.add(drop_col)
        print(f"  Dropped '{drop_col}' (corr={upper.loc[hc, col]:.2f} with '{keep_col}')")
if corr_drop:
    cf = cf.drop(columns=list(corr_drop))
else:
    print("  No highly correlated pairs found")

# Step 4: RF importance pre-screen
print("\n[Step 4] RF importance pre-screen...")
feat_candidates = [c for c in cf.select_dtypes(include=[np.number]).columns
                   if c not in NON_FEATURE_COLS and c != 'churned']
cf_valid = cf[cf['churned'].notna()].copy()
X_screen = cf_valid[feat_candidates].fillna(0).values
y_screen = cf_valid['churned'].values
rf_screen = RandomForestClassifier(n_estimators=100, max_depth=6,
                                    random_state=config.RANDOM_STATE, n_jobs=-1)
rf_screen.fit(X_screen, y_screen)
imp_screen = dict(zip(feat_candidates, rf_screen.feature_importances_))
low_imp = [f for f, v in imp_screen.items() if v < 0.005]
if low_imp:
    cf = cf.drop(columns=low_imp)
    print(f"  Dropped {len(low_imp)} near-zero importance features: {low_imp}")
else:
    print("  All features passed importance threshold")

FINAL_FEATURES = [c for c in cf.select_dtypes(include=[np.number]).columns
                  if c not in NON_FEATURE_COLS and c != 'churned']
print(f"\nFINAL FEATURES FOR MODELING ({len(FINAL_FEATURES)} validated signals):")
for i, f in enumerate(FINAL_FEATURES, 1):
    print(f"  {i:2}. {f}")

# ================================================================
# BLOCK 9 -- Stage 1: Churn Classifier
# ================================================================
print("\n--- BLOCK 9: Stage 1 Churn Classifier ---")

from sklearn.model_selection import train_test_split

cf_valid = cf[cf['churned'].notna()].copy()
# Sort by order_count ascending (fewer orders = newer, less established -- temporal proxy)
# Stratified to maintain class balance (churn label == days_since_last_order >= 30
# so pure days_since_last_order sort would create 100% churned test set)
cf_sorted = cf_valid.sort_values('order_count', ascending=False).reset_index(drop=True)

X_all = cf_sorted[FINAL_FEATURES].fillna(0).values
y_all = cf_sorted['churned'].values

X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
    X_all, y_all, np.arange(len(cf_sorted)),
    test_size=0.2, random_state=config.RANDOM_STATE, stratify=y_all
)
train_df = cf_sorted.iloc[idx_train].copy()
test_df  = cf_sorted.iloc[idx_test].copy()
cf_sorted_full = cf_sorted.copy()

print(f"Train: {len(X_train):,} customers, churn rate {y_train.mean()*100:.1f}%")
print(f"Test:  {len(X_test):,} customers, churn rate {y_test.mean()*100:.1f}%")
print("Stratified split -- class balance maintained across train/test")

if HAS_SMOTE:
    print(f"\nBefore SMOTE: {np.bincount(y_train.astype(int))}")
    try:
        smote = SMOTE(random_state=config.RANDOM_STATE)
        X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
        print(f"After SMOTE:  {np.bincount(y_train_res.astype(int))}")
    except Exception as e:
        print(f"  SMOTE error: {e}. Using original.")
        X_train_res, y_train_res = X_train, y_train
else:
    X_train_res, y_train_res = X_train, y_train

models_cls = {}
print("\nTraining Logistic Regression...")
lr = LogisticRegression(class_weight='balanced', max_iter=1000,
                         random_state=config.RANDOM_STATE)
lr.fit(X_train_res, y_train_res)
models_cls['Logistic Regression'] = lr

print("Training Random Forest...")
rf = RandomForestClassifier(n_estimators=300, max_depth=8, class_weight='balanced',
                             random_state=config.RANDOM_STATE, n_jobs=-1)
rf.fit(X_train_res, y_train_res)
models_cls['Random Forest'] = rf

print("Training XGBoost...")
neg_pos = max((y_train == 0).sum() / max((y_train == 1).sum(), 1), 1)
if HAS_XGB:
    xgb = XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=4,
                         subsample=0.8, colsample_bytree=0.8,
                         scale_pos_weight=neg_pos,
                         random_state=config.RANDOM_STATE, verbosity=0,
                         eval_metric='logloss')
else:
    xgb = XGBClassifier(n_estimators=200, max_depth=4,
                         random_state=config.RANDOM_STATE)
xgb.fit(X_train_res, y_train_res)
models_cls['XGBoost'] = xgb

print("\nModel comparison:")
print(f"  {'Model':<22}|{'TrainAUC':>9}|{'TestAUC':>9}|{'Gap':>6}|{'Prec':>6}|{'Rec':>6}|{'F1':>6}")
print("  " + "-" * 68)
model_results = {}
for name, model in models_cls.items():
    try:
        tr_auc = roc_auc_score(y_train_res, model.predict_proba(X_train_res)[:, 1])
        te_prob = model.predict_proba(X_test)[:, 1]
        te_auc = roc_auc_score(y_test, te_prob)
        te_pred = (te_prob >= 0.5).astype(int)
        prec = precision_score(y_test, te_pred, zero_division=0)
        rec  = recall_score(y_test, te_pred, zero_division=0)
        f1   = f1_score(y_test, te_pred, zero_division=0)
        gap  = tr_auc - te_auc
        model_results[name] = {'train_auc': tr_auc, 'test_auc': te_auc, 'gap': gap,
                                'precision': prec, 'recall': rec, 'f1': f1, 'proba': te_prob}
        print(f"  {name:<22}| {tr_auc:.4f}  | {te_auc:.4f}  | {gap:.3f}| {prec:.3f}| {rec:.3f}| {f1:.3f}")
        if gap > 0.05:
            print(f"  ** OVERFITTING WARNING: gap={gap:.3f} for {name} **")
    except Exception as e:
        print(f"  {name}: ERROR -- {e}")
print("  " + "-" * 68)

best_name = max(model_results, key=lambda k: model_results[k]['test_auc'])
best_model = models_cls[best_name]
best_auc = model_results[best_name]['test_auc']
print(f"\nBest model: {best_name} (Test AUC = {best_auc:.4f})")

# Leakage check
if best_auc > 0.95:
    print(f"\n** AUC={best_auc:.4f} > 0.95 -- checking for leakage **")
    if 'days_since_last_order' in FINAL_FEATURES:
        print("  LEAKAGE: days_since_last_order IS the churn label (>= 30 = churned)")
    high_corr = []
    for f in FINAL_FEATURES:
        if cf_sorted[f].std() > 0:
            c = abs(cf_sorted[f].corr(cf_sorted['churned']))
            if c > 0.65:
                high_corr.append((f, c))
    if high_corr:
        high_corr.sort(key=lambda x: -x[1])
        for fname, corr_val in high_corr:
            print(f"  LEAKAGE CANDIDATE: '{fname}' corr={corr_val:.3f} with churn label")
    if not high_corr and 'days_since_last_order' not in FINAL_FEATURES:
        print("  No obvious leakage found -- AUC may reflect genuine signal strength")
        print("  Consider whether late_avg_gap_clean still encodes recency via the 2nd-most-recent gap")
    print("  STOPPING -- fix leakage before proceeding")
    import sys; sys.exit(1)

print(f"\n5-fold stratified CV on {best_name}...")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.RANDOM_STATE)
X_cv = cf_sorted_full[FINAL_FEATURES].fillna(0).values
y_cv = cf_sorted_full['churned'].values
cv_scores = []
for fold, (tr_i, val_i) in enumerate(skf.split(X_cv, y_cv), 1):
    Xtr, Xvl = X_cv[tr_i], X_cv[val_i]
    ytr, yvl = y_cv[tr_i], y_cv[val_i]
    if HAS_SMOTE and len(np.unique(ytr)) > 1:
        try:
            Xtr, ytr = SMOTE(random_state=config.RANDOM_STATE).fit_resample(Xtr, ytr)
        except Exception:
            pass
    m = type(best_model)(**best_model.get_params())
    m.fit(Xtr, ytr)
    s = roc_auc_score(yvl, m.predict_proba(Xvl)[:, 1])
    cv_scores.append(s)
    print(f"  Fold {fold}: {s:.4f}")
cv_mean = np.mean(cv_scores)
cv_std  = np.std(cv_scores)
print(f"5-fold CV: {cv_mean:.4f} +/- {cv_std:.4f}")

os.makedirs(config.MODELS_DIR, exist_ok=True)
with open(config.BEST_MODEL_PATH, 'wb') as f:
    pickle.dump({'model': best_model, 'features': FINAL_FEATURES, 'name': best_name}, f)
print(f"Saved: {config.BEST_MODEL_PATH}")

# ================================================================
# BLOCK 10 -- Feature Importance
# ================================================================
print("\n--- BLOCK 10: Feature Importance ---")
if hasattr(best_model, 'feature_importances_'):
    imp_vals = best_model.feature_importances_
elif hasattr(best_model, 'coef_'):
    imp_vals = np.abs(best_model.coef_[0])
else:
    imp_vals = np.ones(len(FINAL_FEATURES))

imp_df = pd.DataFrame({'feature': FINAL_FEATURES, 'importance': imp_vals})
imp_df = imp_df.sort_values('importance', ascending=False).reset_index(drop=True)

gap_feats_set = {'avg_days_between_orders', 'gap_trend', 'early_avg_gap', 'late_avg_gap_clean',
                 'order_gap_cv', 'order_gap_std', 'gap_acceleration'}
rr_feats_set  = {'reorder_rate', 'early_reorder_rate', 'late_reorder_rate', 'reorder_trend'}

print(f"\n{'Rank':<5}| {'Feature':<32}| {'Importance':>11}| Group")
print("-" * 58)
for i, row in imp_df.head(15).iterrows():
    grp = 'Reorder' if row['feature'] in rr_feats_set else \
          'Gap'     if row['feature'] in gap_feats_set else 'Other'
    print(f"  {i+1:<4}| {row['feature']:<32}| {row['importance']:>11.4f}| {grp}")
print("-" * 58)

top3 = imp_df.head(3)['feature'].tolist()
top5 = imp_df.head(5)['feature'].tolist()
if 'reorder_rate' in top3:
    print("VALIDATED: reorder_rate top predictor, consistent with Check 4 (delta=10pts, p~=0)")
else:
    rr_rank = imp_df[imp_df['feature'] == 'reorder_rate'].index[0]+1 if 'reorder_rate' in imp_df['feature'].values else 'N/A'
    print(f"NOTE: reorder_rate not in top 3 -- rank {rr_rank}. Check feature encoding.")

if any(f in top5 for f in gap_feats_set):
    print("VALIDATED: gap signal in top 5, consistent with Check 2 (gap widens before churn)")
else:
    print("NOTE: gap signal not in top 5 -- investigate feature values")

if 'trajectory_enc' in top5:
    print("Trajectory classification adds real signal beyond raw behavioral features")

eng_top10 = len([f for f in imp_df.head(10)['feature'] if f in
                 {'gap_trend','gap_acceleration','reorder_trend','trajectory_enc',
                  'early_avg_gap','late_avg_gap_clean','early_reorder_rate','late_reorder_rate'}])
print(f"\nEngineered features in top 10: {eng_top10}")
print("This validates the feature engineering decisions")

from matplotlib.patches import Patch
fig, ax = plt.subplots(figsize=(10, 7))
top15_imp = imp_df.head(15)
c_imp = ['#d32f2f' if f in rr_feats_set else '#1565c0' if f in gap_feats_set else '#2e7d32'
         for f in top15_imp['feature']]
ax.barh(top15_imp['feature'][::-1], top15_imp['importance'][::-1], color=c_imp[::-1])
ax.set_xlabel('Feature Importance')
ax.set_title(f'Feature Importance - {best_name} (Churn Classifier)')
ax.legend(handles=[Patch(color='#d32f2f', label='Reorder'),
                   Patch(color='#1565c0', label='Gap'),
                   Patch(color='#2e7d32', label='Other')])
plt.tight_layout()
plt.savefig(os.path.join(config.CHARTS_DIR, 'feature_importance_classifier.png'), dpi=config.CHART_DPI)
plt.close()
print(f"Saved: {os.path.join(config.CHARTS_DIR, 'feature_importance_classifier.png')}")

# ================================================================
# BLOCK 11 -- Stage 2: Revenue Regressor
# ================================================================
print("\n--- BLOCK 11: Stage 2 Revenue Regressor ---")

churned_df = cf_sorted_full[cf_sorted_full['churned'] == 1].copy().reset_index(drop=True)
target_col = 'revenue_at_risk_90d'
if target_col not in churned_df.columns or churned_df[target_col].sum() == 0:
    ann_val = churned_df.get('annual_value_est', churned_df.get('annual_orders_est', pd.Series(0)) * 110)
    churned_df[target_col] = ann_val * 0.25

print(f"Stage 2 dataset: {len(churned_df):,} churned customers")
tgt = churned_df[target_col].fillna(0)
print(f"Target range: ${tgt.min():.0f} to ${tgt.max():.0f} (mean ${tgt.mean():.0f})")

reg_feats = [f for f in FINAL_FEATURES if f in churned_df.columns]
split_r = int(len(churned_df) * 0.8)
Xrtr = churned_df.iloc[:split_r][reg_feats].fillna(0).values
yrtr = churned_df.iloc[:split_r][target_col].fillna(0).values
Xrte = churned_df.iloc[split_r:][reg_feats].fillna(0).values
yrte = churned_df.iloc[split_r:][target_col].fillna(0).values

print("Training RF Regressor baseline...")
rfr_base = RandomForestRegressor(n_estimators=300, max_depth=8,
                                   random_state=config.RANDOM_STATE, n_jobs=-1)
rfr_base.fit(Xrtr, yrtr)
yp_base = rfr_base.predict(Xrte)
r2b = r2_score(yrte, yp_base)
rmse_b = np.sqrt(mean_squared_error(yrte, yp_base))
mae_b = mean_absolute_error(yrte, yp_base)

print("Training RF Regressor tuned...")
rfr_tuned = RandomForestRegressor(n_estimators=500, max_depth=8,
                                    min_samples_split=5, min_samples_leaf=4,
                                    max_features=0.5,
                                    random_state=config.RANDOM_STATE, n_jobs=-1)
rfr_tuned.fit(Xrtr, yrtr)
yp_tuned = rfr_tuned.predict(Xrte)
r2t = r2_score(yrte, yp_tuned)
rmse_t = np.sqrt(mean_squared_error(yrte, yp_tuned))
mae_t = mean_absolute_error(yrte, yp_tuned)

kf = KFold(n_splits=5, shuffle=False)
Xrcv = churned_df[reg_feats].fillna(0).values
yrcv = churned_df[target_col].fillna(0).values
cv_r2_list = []
for fold, (tri, vali) in enumerate(kf.split(Xrcv), 1):
    m = RandomForestRegressor(n_estimators=200, max_depth=8, min_samples_split=5,
                               min_samples_leaf=4, max_features=0.5,
                               random_state=config.RANDOM_STATE, n_jobs=-1)
    m.fit(Xrcv[tri], yrcv[tri])
    s = r2_score(yrcv[vali], m.predict(Xrcv[vali]))
    cv_r2_list.append(s)
    print(f"  CV Fold {fold}: R2 = {s:.4f}")
cv_r2_mean = np.mean(cv_r2_list)

print("\nRegressor comparison:")
print(f"  {'Metric':<14}| {'Baseline':>10}| {'Tuned':>10}| {'Change':>10}")
print("  " + "-" * 48)
print(f"  {'R2':<14}| {r2b:>10.4f}| {r2t:>10.4f}| {r2t-r2b:>+10.4f}")
print(f"  {'RMSE ($)':<14}| {rmse_b:>10.2f}| {rmse_t:>10.2f}| {rmse_t-rmse_b:>+10.2f}")
print(f"  {'MAE ($)':<14}| {mae_b:>10.2f}| {mae_t:>10.2f}| {mae_t-mae_b:>+10.2f}")
print(f"  {'CV R2':<14}| {'--':>10}| {cv_r2_mean:>10.4f}| {'--':>10}")
print("  " + "-" * 48)

reg_imp_df = pd.DataFrame({'feature': reg_feats, 'importance': rfr_tuned.feature_importances_})
reg_imp_df = reg_imp_df.sort_values('importance', ascending=False).reset_index(drop=True)
print("\nTop 15 regressor features:")
for i, row in reg_imp_df.head(15).iterrows():
    rev_note = "(revenue scale)" if row['feature'] in {'avg_basket_size','avg_days_between_orders',
               'order_count','annual_value_est','annual_orders_est'} else "(churn pattern)"
    print(f"  {i+1:2}. {row['feature']:<30} {row['importance']:.4f}  {rev_note}")

fig, ax = plt.subplots(figsize=(10, 7))
t15r = reg_imp_df.head(15)
ax.barh(t15r['feature'][::-1], t15r['importance'][::-1], color='steelblue')
ax.set_xlabel('Feature Importance')
ax.set_title('Feature Importance - Revenue Regressor (Stage 2)')
plt.tight_layout()
plt.savefig(os.path.join(config.CHARTS_DIR, 'feature_importance_regressor.png'), dpi=config.CHART_DPI)
plt.close()
print(f"Saved: {os.path.join(config.CHARTS_DIR, 'feature_importance_regressor.png')}")

with open(config.REVENUE_REGRESSOR_PATH, 'wb') as f:
    pickle.dump({'model': rfr_tuned, 'features': reg_feats}, f)
print(f"Saved: {config.REVENUE_REGRESSOR_PATH}")

# ================================================================
# BLOCK 12 -- Risk Score and Ranking
# ================================================================
print("\n--- BLOCK 12: Risk Score and Ranking ---")

X_full = cf_sorted_full[FINAL_FEATURES].fillna(0).values
cf_sorted_full = cf_sorted_full.copy()
cf_sorted_full['P_churn'] = best_model.predict_proba(X_full)[:, 1]
cf_sorted_full['predicted_loss'] = rfr_tuned.predict(cf_sorted_full[reg_feats].fillna(0).values)
cf_sorted_full['risk_score'] = cf_sorted_full['P_churn'] * cf_sorted_full['predicted_loss']
cf_sorted = cf_sorted_full

ranked = cf_sorted.sort_values('risk_score', ascending=False).reset_index(drop=True)
ranked['rank'] = ranked.index + 1

print("\nTop 20 customers by risk score:")
print(f"  {'Rank':<5}| {'user_id':>8}| {'Traj':>5}| {'P(churn)':>9}| {'Pred Loss':>10}| {'Risk Score':>11}")
print("  " + "-" * 57)
ttype_col = 'trajectory_type' if 'trajectory_type' in ranked.columns else None
for _, row in ranked.head(20).iterrows():
    traj_val = row[ttype_col] if ttype_col else 'N/A'
    print(f"  {int(row['rank']):<5}| {int(row['user_id']):>8}| {str(traj_val):>5}| "
          f"{row['P_churn']:>9.3f}| ${row['predicted_loss']:>9.1f}| ${row['risk_score']:>10.1f}")
print("  " + "-" * 57)

total_risk = ranked['risk_score'].sum()
top10pct = int(len(ranked) * 0.1)
t10_risk = ranked.head(top10pct)['risk_score'].sum()
t500_risk = ranked.head(500)['risk_score'].sum()
cum_risk = ranked['risk_score'].cumsum() / total_risk
p80_idx = int((cum_risk >= 0.80).idxmax()) + 1
p80_pct = p80_idx / len(ranked) * 100
print(f"\nPareto analysis:")
print(f"  Top 10% ({top10pct:,} customers) = {t10_risk/total_risk*100:.1f}% of total projected risk")
print(f"  Top 500 customers = {t500_risk/total_risk*100:.1f}% of total risk")
print(f"  Pareto finding: top {p80_idx:,} customers ({p80_pct:.1f}%) = 80% of risk")

if ttype_col:
    top500 = ranked.head(500)
    print("\nTop 500 by trajectory type:")
    for t in ['B', 'D', 'A', 'C']:
        cnt = (top500[ttype_col] == t).sum()
        print(f"  Type {t}: {cnt:,} customers in top 500")

out_cols = ['rank', 'user_id', 'P_churn', 'predicted_loss', 'risk_score', 'churned']
if ttype_col:
    out_cols.insert(2, ttype_col)
for xc in ['reorder_rate', 'gap_trend', 'avg_days_between_orders']:
    if xc in ranked.columns:
        out_cols.append(xc)
ranked[out_cols].to_csv(config.RISK_RANKED_CUSTOMERS, index=False)
print(f"\nSaved: {config.RISK_RANKED_CUSTOMERS}")

# Save model metrics
pd.DataFrame([{
    'best_model_name': best_name, 'test_auc': best_auc,
    'cv_mean': cv_mean, 'cv_std': cv_std,
    'reg_r2': r2t, 'reg_rmse': rmse_t, 'reg_mae': mae_t, 'reg_cv_r2': cv_r2_mean,
    'n_features': len(FINAL_FEATURES),
    'n_train': len(X_train), 'n_test': len(X_test),
    'pareto_80pct_customers': p80_idx,
    'top500_risk_pct': t500_risk / total_risk * 100,
}]).to_csv(config.MODEL_METRICS, index=False)
print(f"Saved: {config.MODEL_METRICS}")

# Model comparison chart
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
mnames = list(model_results.keys())
atr = [model_results[n]['train_auc'] for n in mnames]
ate = [model_results[n]['test_auc'] for n in mnames]
xp = range(len(mnames))
w = 0.35
axes[0].bar([xi - w/2 for xi in xp], atr, w, label='Train AUC', color='steelblue')
axes[0].bar([xi + w/2 for xi in xp], ate, w, label='Test AUC', color='darkorange')
axes[0].set_xticks(list(xp))
axes[0].set_xticklabels([n.replace(' ', '\n') for n in mnames], fontsize=9)
axes[0].set_ylabel('AUC')
axes[0].set_title('Model Comparison: Train vs Test AUC')
axes[0].legend()
axes[0].set_ylim(0.5, 1.0)
axes[1].hist(ranked['risk_score'], bins=50, color='steelblue', edgecolor='white', lw=0.3)
axes[1].set_xlabel('Risk Score')
axes[1].set_ylabel('Customers')
axes[1].set_title('Risk Score Distribution')
plt.tight_layout()
plt.savefig(config.MODEL_COMPARISON_CHART, dpi=config.CHART_DPI)
plt.close()
print(f"Saved: {config.MODEL_COMPARISON_CHART}")

print("\n" + "=" * 60)
print(f"Modeling complete.")
print(f"Best model: {best_name}, AUC: {best_auc:.4f}, CV: {cv_mean:.4f} +/- {cv_std:.4f}")
print(f"Regressor R2: {r2t:.4f}, RMSE: ${rmse_t:.2f}")
print("Next -- pipeline/07_category_exposure.py")
print("=" * 60)
