# ============================================================
# config.py — Instacart Basket Shrinkage & Revenue Protection
# ============================================================
# Single source of truth for all file paths. Every pipeline
# script imports from here so nothing breaks if the project
# moves to a new machine or directory.
# ============================================================

import os

BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
RAW_DIR       = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
OUTPUTS_DIR   = os.path.join(BASE_DIR, "outputs")
CHARTS_DIR    = os.path.join(BASE_DIR, "outputs", "charts")
SQL_DIR       = os.path.join(BASE_DIR, "sql")
DOCS_DIR      = os.path.join(BASE_DIR, "docs")

# --- Raw source files ---
ORDERS             = os.path.join(RAW_DIR, "orders.csv")
ORDER_PRODUCTS_PRIOR = os.path.join(RAW_DIR, "order_products__prior.csv")   # 577MB — chunked load
ORDER_PRODUCTS_TRAIN = os.path.join(RAW_DIR, "order_products__train.csv")
PRODUCTS           = os.path.join(RAW_DIR, "products.csv")
AISLES             = os.path.join(RAW_DIR, "aisles.csv")
DEPARTMENTS        = os.path.join(RAW_DIR, "departments.csv")

# --- Processed / intermediate outputs ---
MASTER_ORDERS      = os.path.join(PROCESSED_DIR, "master_orders.csv")
RFM_SEGMENTS       = os.path.join(PROCESSED_DIR, "rfm_segments.csv")
CUSTOMER_FEATURES  = os.path.join(PROCESSED_DIR, "customer_features.csv")
INTERVENTION_REPORT  = os.path.join(PROCESSED_DIR, "intervention_report.csv")
CATEGORY_RISK_REPORT = os.path.join(PROCESSED_DIR, "category_risk_report.csv")

# --- Chart outputs ---
RFM_CHART          = os.path.join(CHARTS_DIR, "rfm_segments.png")
SHRINKAGE_CHART    = os.path.join(CHARTS_DIR, "top20_shrinking_customers.png")
CATEGORY_HEATMAP   = os.path.join(CHARTS_DIR, "category_heatmap.png")
EDA_CHARTS_DIR     = os.path.join(CHARTS_DIR, "eda")
MODEL_COMPARISON_CHART = os.path.join(CHARTS_DIR, "model_comparison.png")

# --- Database & report outputs ---
DB_PATH            = os.path.join(PROCESSED_DIR, "instacart.db")
REPORTS_DIR        = os.path.join(OUTPUTS_DIR, "reports")

# --- Processed CSVs (pipeline 03+) ---
CLEANED_ORDERS     = os.path.join(PROCESSED_DIR, "cleaned_orders.csv")
CHURN_PREDICTIONS  = os.path.join(REPORTS_DIR, "churn_predictions.csv")

# --- Model persistence ---
MODELS_DIR         = os.path.join(BASE_DIR, "models")
BEST_MODEL_PATH    = os.path.join(MODELS_DIR, "best_model.pkl")

# --- SQL file paths (used by run_all_queries.py) ---
SQL_RFM            = os.path.join(SQL_DIR, "01_rfm_queries.sql")
SQL_SHRINKAGE      = os.path.join(SQL_DIR, "02_shrinkage_queries.sql")
SQL_CATEGORY       = os.path.join(SQL_DIR, "03_category_exposure.sql")

# --- New pipeline paths (14-block rebuild) ---
TRAJECTORY_SEGMENTS   = os.path.join(PROCESSED_DIR, "trajectory_segments.csv")
EXCLUDED_CUSTOMERS    = os.path.join(PROCESSED_DIR, "excluded_customers.csv")
RETENTION_ROI         = os.path.join(REPORTS_DIR, "retention_roi.csv")
RISK_RANKED_CUSTOMERS = os.path.join(REPORTS_DIR, "risk_ranked_customers.csv")
DEPARTMENT_RESIDUALS  = os.path.join(REPORTS_DIR, "department_residuals.csv")
EXECUTIVE_SUMMARY_TXT = os.path.join(REPORTS_DIR, "executive_summary.txt")
MODEL_METRICS         = os.path.join(REPORTS_DIR, "model_metrics.csv")
REVENUE_REGRESSOR_PATH = os.path.join(MODELS_DIR, "revenue_regressor.pkl")

# --- Analysis parameters ---
CHUNK_SIZE         = 500_000   # rows per chunk for order_products__prior.csv
CHART_DPI          = 150
RANDOM_STATE       = 42
