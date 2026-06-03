Here's the complete README in one block — copy everything between the lines:

---

```markdown
# Instacart Churn & Customer Retention Analysis

[![Streamlit](https://img.shields.io/badge/Live%20Dashboard-Streamlit-FF4B4B?style=flat&logo=streamlit)](https://instacart-churn-retention-analysis-dashboard.streamlit.app/)
[![Power BI](https://img.shields.io/badge/Power%20BI-Live%20Report-F2C811?style=flat&logo=powerbi)](https://app.powerbi.com/links/phH_dmePJX?ctid=70de1992-07c6-480f-a318-a1afcba03983&pbi_source=linkShare)
[![dbt Tests](https://github.com/91Sakthivel/instacart-churn-retention-analysis/actions/workflows/dbt_tests.yml/badge.svg)](https://github.com/91Sakthivel/instacart-churn-retention-analysis/actions/workflows/dbt_tests.yml)

---

## What this project is about

I wanted to build something that went beyond predicting churn. Most churn projects stop at "here's who might leave." This one tries to answer the harder question: what do you actually do about it, and is it worth the cost?

The dataset is Instacart's public grocery transaction data — 33.8 million order rows, 206,209 customers. No prices, no timestamps. Revenue is estimated at $3.50 per item and time is proxied by order sequence. I documented those constraints upfront and built around them rather than pretending they don't exist.

The three questions I set out to answer:

1. Who is quietly leaving, and how much revenue is at risk?
2. When should you intervene, and what's the ROI of each option?
3. Which product categories are sitting on the most exposure?

---

## The finding I'm most proud of

Before building any model, I ran five statistical tests on the signals I planned to use. Two of them failed.

Basket size decline — the most intuitive churn signal — turned out to be flat or slightly rising right into the final order (p=0.237, not significant). I removed it from the model entirely. Gap widening and reorder rate drop were the real signals (p≈0).

That validation step changed everything downstream. It's documented in `pipeline/validate_signals.py` and `docs/decision_log.md`.

---

## Key numbers

| What | Number |
|---|---|
| Customers analyzed | 175,072 (of 206,209 total) |
| XGBoost AUC | 0.9059 (CV: 0.9070 ± 0.0013) |
| 60-day revenue at risk | $626,303 |
| Most exposed category | Produce — $183,384 |
| Annual budget waste found | $980,584 (Type C blanket discounting) |
| Best ROI intervention | Type C push notification — 43,849% |
| Best value intervention | Type D loyalty reward — 17,891% ROI, 40% recovery |

The $980,584 waste finding is the one that matters most for business. 89,144 low-loyalty customers were being sent $11 discount offers. A $0.50 push notification recovers the same percentage at a fraction of the cost.

---

## What I built

**Python pipeline (11 scripts)**
End-to-end from raw CSVs to model outputs. Includes hypothesis validation before feature engineering, three rounds of leakage detection (AUC went from 1.0 to 0.9059 across the fixes), and an XGBoost classifier plus a Random Forest revenue regressor combined into a customer risk ranking.

**SQL layer**
A 1.6 GB SQLite database with 33.8M rows, built using executemany() for performance. Three SQL files demonstrating window functions, CTEs, NTILE(), ROW_NUMBER(), and running Pareto totals.

**dbt + BigQuery**
I rebuilt the transformation layer as a proper three-tier dbt project on BigQuery — staging views, an intermediate join layer, and a star schema mart with six tables. 15 automated data tests, sources.yml, schema.yml with column-level documentation, and a lineage graph. The fact table is 175,072 rows with one row per customer and every dimension joined in.

**Power BI report (4 pages, live)**
Connected directly to the BigQuery mart via DirectQuery. No data import — every refresh queries the warehouse live. Four pages: executive KPI scorecard, customer risk by trajectory type, intervention ROI comparison, and department exposure heatmap. Slicers on every page.

🔗 https://app.powerbi.com/links/phH_dmePJX?ctid=70de1992-07c6-480f-a318-a1afcba03983&pbi_source=linkShare

**Streamlit dashboard (7 pages, live)**
Includes a behavioral monitor that simulates weekly re-scoring using order sequence as a time proxy, and a live ROI simulator where you can adjust campaign parameters and see revenue impact update in real time.

🔗 https://instacart-churn-retention-analysis-dashboard.streamlit.app/

---

## The leakage story

This is worth explaining because it's the part that took the most work.

The first model hit AUC 1.0. That's always wrong. I traced it back to `days_since_last_order` being in the feature matrix — it IS the churn label (churned = gap ≥ 30 days), so the model was predicting its own definition. Removed it. AUC dropped to 0.9420.

Second run still looked suspicious. Found that `late_avg_gap` was computed from the last 5 gaps including the final one — which equals `days_since_last_order`. Fixed it by computing `late_avg_gap_clean` from rank positions 2–4, explicitly excluding the final gap. AUC moved to 0.9456.

Third issue was subtler. The Type B trajectory definition used `days_since_last_order ≥ 25` as a trigger, which made it a near-perfect churn proxy (79.1% churn rate). Redefined Type B using pure behavioral gap signals. AUC settled at 0.9059 — stable, honest, and defensible.

All three rounds are documented in `pipeline/08_churn_model.py` and `docs/decision_log.md`.

---

## Trajectory types

I classified 107,438 customers (those with 10+ orders) into four behavioral trajectories:

| Type | Name | Customers | Churn | Best offer | Cost | ROI |
|---|---|---|---|---|---|---|
| A | Fading Frequency | 2,498 | 44.5% | Product discount | $11.00 | 6,075% |
| B | Sudden Stop | 6,010 | 35.0% | Phone call | $15.00 | 3,820% |
| C | Low Loyalty | 89,144 | 14.9% | Push notification | $0.50 | 43,849% |
| D | High Value Drifting | 9,786 | 22.3% | Loyalty reward | $8.50 | 17,891% |

Type D is the most important group. Low churn rate but highest annual value — they're the customers worth fighting for. Act before their gap exceeds 20 days.

---

## Tech stack

Python · SQL · XGBoost · Scikit-learn · dbt · BigQuery · Power BI · Streamlit · Plotly · SQLite · Git

---

## How to run it

You'll need the 6 Instacart CSV files from [Kaggle](https://www.kaggle.com/c/instacart-market-basket-analysis/data) in `data/raw/`. Then:

```bash
pip install -r requirements.txt

python pipeline/01_load_explore.py
python pipeline/02_eda.py
python pipeline/03_clean.py
python pipeline/validate_signals.py
python pipeline/04_feature_engineering.py
python pipeline/04_rfm.py
python pipeline/05b_trajectory.py
python pipeline/07_category_exposure.py
python pipeline/08_churn_model.py
python pipeline/09_executive_summary.py
python pipeline/10_weekly_monitor.py

python sql/00_create_database.py

cd instacart_dbt
dbt run
dbt test
dbt docs generate && dbt docs serve

streamlit run streamlit/app.py
```

For the dbt layer you'll need `~/.dbt/profiles.yml` configured with your BigQuery credentials.

---

## Project structure

```
instacart-analytics1/
├── pipeline/          11 scripts — load, clean, validate, model, monitor
├── sql/               SQLite database + 3 SQL query files
├── instacart_dbt/     dbt project — staging, intermediate, marts, tests, docs
├── streamlit/         7-page dashboard + data files for Cloud deployment
├── data/processed/    Pipeline outputs (raw CSVs not committed)
├── outputs/           Charts and reports
├── models/            best_model.pkl (XGBoost) + revenue_regressor.pkl (RF)
├── docs/              decision_log.md + analyst_notes.md
└── Instacart_Retention_Analysis.pbix   Power BI report file
```

---

## Honest limitations

The revenue figures use a $3.50/item proxy — there are no real prices in this dataset. The regressor R² is 0.18, which is intentional — revenue per customer is genuinely uncertain and the model reflects that. Recovery rates (20–40% by trajectory type) are based on industry benchmarks, not observed outcomes from this data. Retrain monthly on fresh order data in a production setting.

---

*Dataset: [Instacart Market Basket Analysis](https://www.kaggle.com/c/instacart-market-basket-analysis/data) — publicly released for a 2017 Kaggle competition.*
```