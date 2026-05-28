# Instacart Churn & Customer Retention Analysis

**End-to-end retention analytics — XGBoost · dbt · BigQuery · Streamlit**

[![Live Dashboard](https://img.shields.io/badge/Live%20Dashboard-Streamlit-FF4B4B?style=flat&logo=streamlit)](https://instacart-churn-retention-analysis-dashboard.streamlit.app/)

---

## Business Problem

Some customers don't cancel — they just slowly buy less until they disappear. By the time standard churn reports flag them, the window to act has usually closed. This project identifies them while they're still ordering, ranks them by projected revenue loss, and assigns each customer to a trajectory type with a specific, costed intervention.

The honest constraint: the Instacart public dataset has no prices and no real timestamps — revenue is estimated at $3.50 per item, and time is proxied by order sequence. That limitation is documented throughout and doesn't change the relative ranking of customers at risk.

**Live dashboard:** https://instacart-churn-retention-analysis-dashboard.streamlit.app/

---

## Key Findings

| Metric | Value |
|---|---|
| Total customers analyzed | 206,209 |
| Customers in behavioral model (≥5 orders) | 175,072 |
| Excluded (<5 orders) | 31,137 (15.1%) |
| XGBoost churn model — Test AUC | **0.9059** |
| 5-fold cross-validation AUC | 0.9070 ± 0.0013 |
| Revenue regressor R² | 0.1826 · RMSE $23.98 |
| 60-day category revenue at risk | **$626,303** |
| Most exposed department | Produce — $183,384 (29% of total) |
| Pareto: customers driving 80% of risk | Top 93,043 of 175,072 |
| Wasted spend from mistargeting Type C | **$980,584/year** (89,144 × $11 discount) |
| Type D loyalty reward ROI | 17,891% |
| Type B phone call ROI | 3,820% |
| Type A product discount ROI | 6,075% |
| Type C push notification ROI | 43,849% |

### Trajectory Breakdown (107,438 customers with 10+ orders)

| Type | Name | Customers | Churn Rate | Intervention | Cost | Recovery |
|---|---|---|---|---|---|---|
| B | Sudden Stop | 6,010 | 35.0% | Personal phone call | $15.00 | 20% |
| D | High Value Drifting | 9,786 | 22.3% | Loyalty reward | $8.50 | 40% |
| A | Fading Frequency | 2,498 | 44.5% | Product discount | $11.00 | 35% |
| C | Low Loyalty | 89,144 | 14.9% | Push notification | $0.50 | 5% |

**Type D has the highest ROI** despite only 22% churn — they're the highest-value customers. **Type C should never receive expensive interventions.** Blanket discount campaigns that treat all four types the same waste most of the budget.

---

## Tech Stack

| Layer | Tools |
|---|---|
| Data pipeline | Python — pandas, numpy, scikit-learn, XGBoost, imbalanced-learn |
| Churn model | XGBoost classifier + RandomForest revenue regressor |
| Local SQL | SQLite (~1.6 GB), executemany() bulk load, 9 indexes |
| Cloud warehouse | dbt + BigQuery — 10 models, 15 data tests, star schema |
| Dashboard | Streamlit, Plotly — 7 pages, deployed on Streamlit Cloud |
| Documentation | dbt docs (lineage graph + column descriptions) |

---

## Project Architecture

```
Raw Data (6 CSV files — 33.8M order-product rows, 577 MB compressed)
│
├── 01_load_explore.py      Join chain: order_products → orders → products
│                           → aisles → departments. Output: master_orders.csv (3.2 GB)
│
├── 02_eda.py               12 EDA charts. Validates 5 behavioral signal hypotheses.
│                           Key finding: basket size does NOT decline before churn.
│                           Gap widening (15.3d → 30.0d) and reorder rate drop validated.
│
├── 03_clean.py             7 documented cleaning decisions (D1–D7).
│                           Customers with <5 orders saved to excluded_customers.csv.
│
├── validate_signals.py     5 statistical checks before any feature engineering.
│                           basket_slope and category_shift_score removed (p=0.237).
│                           gap_trend and reorder_rate confirmed as primary signals.
│
├── 04_feature_engineering  29 features across 7 groups.
│                           late_avg_gap_clean built to exclude rank_desc=1 (churn label).
│                           gap_trend = late_avg_gap_clean − early_avg_gap.
│                           revenue_at_risk_90d = basket × annual_orders × 0.25 (no gap dependency).
│
├── 04_rfm.py               RFM quartile scoring for all 206,209 customers.
│                           Champions: 55,022 | Loyal: 31,249 | At Risk: 35,050 | Lost: 84,888
│
├── 05b_trajectory.py       Classifies 107,438 customers (order_count ≥ 10) into A/B/C/D.
│                           Type B trigger: early_avg_gap < 12 AND late_avg_gap_clean > 20.
│                           ROI calculations per trajectory type.
│
├── 07_category_exposure.py XGBoost trained WITHOUT department features.
│                           Residual = actual − predicted churn by department.
│                           Result: all 21 departments "As expected" — churn is behavioral.
│
├── 08_churn_model.py       4-step feature selection. Final 8 features.
│                           3 leakage fix rounds: AUC 1.0000 → 0.9420 → 0.9456 → 0.9059.
│                           SMOTE on training set only. Stratified split.
│                           Leakage guard: sys.exit(1) if AUC > 0.95.
│
├── 09_executive_summary.py Monday morning briefing — all 3 RQs answered.
│
└── 10_weekly_monitor.py    Behavioral monitor. 3 order-sequence snapshots.
                            Delta report: newly_at_risk / recovering / stable_high / stable_low.
         │
         ├── SQL Layer (SQLite — local)
         │   00_create_database.py    1.6 GB · 33.8M rows · 9 indexes · executemany() load
         │   01_rfm_queries.sql       NTILE(10), SUM() OVER(), conditional aggregation
         │   02_shrinkage_queries.sql Severity tiers, segment-level risk
         │   03_category_exposure.sql ROW_NUMBER(), running Pareto totals
         │
         ├── dbt Layer (BigQuery — cloud)
         │   staging/        3 views — source wrappers, column aliases, no joins
         │   intermediate/   1 view — int_customer_risk, all joins + risk_tier CASE
         │   marts/          6 tables — star schema for BI tools
         │                   fact_customer_risk (175K rows, central fact)
         │                   dim_segment · dim_trajectory · dim_department
         │                   mart_retention_priority · mart_segment_summary
         │   15 data tests (unique · not_null · accepted_values + quote:false for INT64)
         │
         └── Streamlit Dashboard (7 pages)
             1. Executive Dashboard      — KPI cards, RFM bar, trajectory churn, top-10 table
             2. Business Findings        — 3 core findings, action plan, honest limitations
             3. Customer Risk Explorer   — filters by trajectory/P(churn)/RFM, CSV export
             4. Intervention & ROI       — ROI table, budget alert, intervention windows
             5. Category Exposure        — department risk bars, residual analysis table
             6. Model Performance        — leakage fixes, feature table, model comparison
             7. Behavioral Monitor       — 3 snapshots, delta report, tier shift chart
```

---

## dbt Layer — Star Schema on BigQuery

The dbt project (`instacart_dbt/`) transforms the Python pipeline outputs into a repeatable, tested analytics layer on BigQuery.

**Three-tier architecture:**

| Layer | Materialisation | Models | Purpose |
|---|---|---|---|
| Staging | Views | 3 | Column aliases, source declarations, no joins |
| Intermediate | View | 1 | All joins + derived `risk_tier` column |
| Marts | Tables | 6 | Business-facing outputs for dashboards and BI tools |

**Key mart — `fact_customer_risk`:** 175,072 rows · one row per modelled customer · joins all dimensions · includes `estimated_save_value = predicted_revenue_loss × recovery_rate`.

**Data tests:** 15 tests across `unique`, `not_null`, `accepted_values`. Integer columns tested with `quote: false` to match BigQuery's strict INT64/STRING type enforcement.

**Docs:** Run `dbt docs generate && dbt docs serve` to browse the full lineage graph with column-level documentation.

**BigQuery datasets:** `instacart_raw` (source) · `instacart_dbt_staging` · `instacart_dbt_intermediate` · `instacart_dbt_marts`

---

## Analytical Decisions

The full reasoning is in `docs/decision_log.md`. Three that are worth calling out:

**Leakage — 3 rounds.** `days_since_last_order` IS the churn label (≥30 = churned). It appeared directly as a feature (Round 1, AUC 1.000 → 0.942), indirectly via `late_avg_gap` which included the final gap (Round 2), and indirectly again via the Type B definition which was initially `days_since_last_order ≥ 25` (Round 3, AUC → 0.9059). Each fix was identified by checking feature importance and re-examining the signal source.

**Basket slope removed.** The original hypothesis was that basket size declines before churn. It doesn't — basket is flat or slightly rising into the final order (9.79 → 10.15 items). The gap-widening signal is what's real.

**Type C budget discipline.** 89,144 customers fall into the low-loyalty group. Sending them the same $11 product discount used for Type A wastes $980,584. The analysis shows they respond to $0.50 push notifications. This is the highest-impact recommendation in the project.

---

## How to Run

**Prerequisites:** Drop the 6 Instacart CSV files into `data/raw/` (download from [Kaggle](https://www.kaggle.com/c/instacart-market-basket-analysis/data)). Install dependencies with `pip install -r requirements.txt`.

```bash
# Pipeline — run from project root in order
python pipeline/01_load_explore.py          # Build master_orders.csv from 6 source files
python pipeline/02_eda.py                   # EDA charts, validate 5 behavioral hypotheses
python pipeline/03_clean.py                 # 7 cleaning decisions, exclude <5 order customers
python pipeline/validate_signals.py         # Confirm gap and reorder rate signals
python pipeline/04_feature_engineering.py   # Build 29 features, leakage-safe gap metrics
python pipeline/04_rfm.py                   # RFM quartile scoring, all 206K customers
python pipeline/05b_trajectory.py           # Trajectory classification + ROI modelling
python pipeline/07_category_exposure.py     # Department residual analysis
python pipeline/08_churn_model.py           # XGBoost model, 3 leakage fix rounds
python pipeline/09_executive_summary.py     # Executive briefing report
python pipeline/10_weekly_monitor.py        # Behavioral monitor, snapshot delta report

# SQL layer (local SQLite)
python sql/00_create_database.py            # Build instacart.db (~1.6 GB, 33.8M rows)

# dbt layer (BigQuery — requires ~/.dbt/profiles.yml configured)
cd instacart_dbt
dbt run                                     # Build all 10 models
dbt test                                    # Run 15 data quality tests
dbt docs generate && dbt docs serve         # Browse lineage graph + column docs

# Dashboard (local)
streamlit run streamlit/app.py
```

---

## Project Structure

```
instacart-analytics1/
├── config.py                       Single source of truth for all file paths
├── requirements.txt
│
├── pipeline/
│   ├── 01_load_explore.py          Load + join chain, chunked for 577MB source file
│   ├── 02_eda.py                   12 EDA charts
│   ├── 03_clean.py                 Data cleaning (D1–D7)
│   ├── validate_signals.py         5 pre-modeling statistical checks
│   ├── 04_feature_engineering.py   Behavioral features, leakage-safe gap metrics
│   ├── 04_rfm.py                   RFM scoring
│   ├── 05b_trajectory.py           Trajectory classification A/B/C/D
│   ├── 07_category_exposure.py     Department residual analysis
│   ├── 08_churn_model.py           XGBoost + RF regressor, leakage guard
│   ├── 09_executive_summary.py     Business briefing output
│   └── 10_weekly_monitor.py        Behavioral monitor, snapshot comparisons
│
├── sql/
│   ├── 00_create_database.py       SQLite build — executemany() bulk load, 9 indexes
│   ├── 01_rfm_queries.sql          NTILE(), SUM() OVER(), window functions
│   ├── 02_shrinkage_queries.sql    Severity classification, segment risk
│   └── 03_category_exposure.sql    ROW_NUMBER(), running Pareto totals
│
├── instacart_dbt/
│   ├── models/
│   │   ├── staging/                stg_customer_features, stg_rfm_segments,
│   │   │                           stg_trajectory_segments + sources.yml + schema.yml
│   │   ├── intermediate/           int_customer_risk (joins + risk_tier)
│   │   └── marts/                  fact_customer_risk, dim_segment, dim_trajectory,
│   │                               dim_department, mart_retention_priority,
│   │                               mart_segment_summary + schema.yml
│   └── dbt_project.yml
│
├── streamlit/
│   ├── app.py                      7-page dashboard (data files co-located for Cloud deploy)
│   └── *.csv                       Pipeline outputs copied here for Streamlit Cloud
│
├── data/
│   ├── raw/                        Source CSVs (not committed — download from Kaggle)
│   └── processed/                  Pipeline outputs: customer_features.csv (61 MB), etc.
│
├── outputs/
│   ├── charts/                     EDA and model charts (PNG)
│   └── reports/                    executive_summary.txt, risk_ranked_customers.csv, etc.
│
├── models/
│   ├── best_model.pkl              XGBoost (0.5 MB)
│   └── revenue_regressor.pkl       RandomForestRegressor (14.2 MB)
│
└── docs/
    ├── decision_log.md             8 analytical decisions with full rationale
    └── analyst_notes.md            Interview-style project walkthrough
```

---

## Dataset

Instacart Market Basket Analysis — publicly released by Instacart for a 2017 Kaggle competition. 206,209 anonymised customers, ~3.4M orders, 33.8M order-product rows. No prices, no real timestamps.

Download: [Kaggle — Instacart Market Basket Analysis](https://www.kaggle.com/c/instacart-market-basket-analysis/data)

Files needed in `data/raw/`: `orders.csv` · `order_products__prior.csv` (~577 MB) · `order_products__train.csv` · `products.csv` · `aisles.csv` · `departments.csv`

---

## Resume Bullets

> Copy-paste ready. Numbers are from the actual pipeline output.

- Built an XGBoost churn model across 175,072 Instacart customers (33.8M order rows), achieving AUC 0.9059 after three rounds of leakage detection that identified `days_since_last_order` as a direct label proxy and `late_avg_gap` as an indirect one.

- Designed a dbt analytics layer on BigQuery with 10 models and 15 automated data tests (unique, not_null, accepted_values), implementing a star schema with `fact_customer_risk` as the central 175K-row fact table feeding Streamlit and BI tools.

- Identified a $980,584 annual campaign waste: 89,144 low-loyalty customers were being targeted with $11 discounts when $0.50 push notifications produce equivalent response (Type C, 43,849% ROI vs 6,075% for product discounts).

- Classified 107,438 customers into four behavioral trajectory types with distinct intervention strategies — Type D "High Value Drifting" customers carry 22% churn rate but highest annual value; loyalty rewards at $8.50 recover 40% at 17,891% ROI.

- Engineered leakage-safe gap features using order sequence position rather than recency timestamps: `late_avg_gap_clean` explicitly excludes the final gap (which equals the churn label threshold), and `gap_trend` is derived entirely from behavioral signals.

- Built and deployed a 7-page Streamlit dashboard on Streamlit Cloud with a behavioral monitor that simulates weekly re-scoring using order sequence as a time proxy, producing tier-shift delta reports across 76,294 early-stage customers.
