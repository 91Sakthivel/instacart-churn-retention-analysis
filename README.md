# Instacart Basket Shrinkage & Revenue Protection System

An end-to-end analytics pipeline that detects customers quietly reducing their grocery spend, estimates the revenue impact, and identifies when and how to intervene — before they're gone.

---

## The Business Problem

Some customers don't cancel. They just slowly buy less and less until they disappear. By the time they show up in a churn report, it's already too late to do anything about it. This project catches them while they're still active.

---

## Three Business Questions

**Question 1 — Who is quietly leaving, and how much money are we about to lose?**

For every customer with sufficient order history, fit a linear trend on their last 5 basket sizes. Flag the ones with a statistically significant negative slope (p < 0.05). Project their spend over the next 90 days. Rank by revenue at risk.

**Question 2 — When is it too late to save them, and what's the last thing they still care about?**

Look at customers who churned and work backwards: how many consecutive declining orders did they have before they stopped? Find where recovery probability drops below 20% — that's the intervention deadline. Identify what product category they were still buying right before they left — that's the re-engagement hook.

**Question 3 — Which product categories are sitting on a ticking revenue bomb?**

Take Champions and Loyal customers who are also showing basket shrinkage. Map their spend by department. Calculate how much category revenue disappears over 60 days if these specific customers keep declining.

---

## Key Findings

| Metric | Value |
|---|---|
| Total customers | 206,209 |
| Champions (top RFM tier) | 55,022 (26.7%) |
| Customers with significant basket decline | 4,898 (2.8%) |
| Projected 90-day revenue at risk | $1,918,140 |
| High-value customers driving 60-day risk | 2,390 (Champions + Loyal) |
| 60-day category revenue at risk | $626,303 |
| Most exposed department | Produce ($183,384) |
| Top win-back hook department | Produce (48.4% of churned customers) |
| Avg basket size | 10.1 items |
| Avg order gap | 11.1 days |
| Reorder rate | 59% |

---

## How to Run

**Prerequisites:** Drop the 6 Instacart CSV files into `data/raw/` before running.

```bash
# Run in order
python pipeline/01_load_explore.py   # Load & explore, build master_orders.csv
python pipeline/02_rfm.py            # RFM segmentation
python pipeline/03_shrinkage.py      # Basket shrinkage regression
python pipeline/04_intervention.py   # Intervention timing + last-hook analysis
python pipeline/05_category_exposure.py  # Category revenue exposure
python pipeline/06_executive_summary.py  # Business summary printout
```

Each script is standalone but reads outputs from prior scripts. Run them in order.

---

## Project Structure

```
instacart-analytics1/
├── config.py                    # All file paths — never hardcoded elsewhere
├── requirements.txt
│
├── data/
│   ├── raw/                     # Source CSV files (not included — download separately)
│   └── processed/               # Pipeline outputs
│       ├── master_orders.csv    # Merged order-product-department table (~3.3GB)
│       ├── rfm_segments.csv     # 206K customers with RFM scores and segments
│       ├── customer_features.csv  # Regression results, shrinkage flags, revenue projections
│       ├── intervention_report.csv  # Churned customers with last-hook departments
│       └── category_risk_report.csv  # Department-level revenue exposure
│
├── pipeline/
│   ├── 01_load_explore.py       # Data loading, validation, master file build
│   ├── 02_rfm.py                # RFM scoring (1–4) and segmentation
│   ├── 03_shrinkage.py          # Vectorized OLS basket trend regression
│   ├── 04_intervention.py       # Decline streak analysis + department hook
│   ├── 05_category_exposure.py  # Category risk heatmap
│   └── 06_executive_summary.py  # Monday morning summary print
│
├── sql/
│   ├── 01_rfm_queries.sql       # RFM in SQL with NTILE() and window functions
│   ├── 02_shrinkage_queries.sql # Basket trend detection with LAG() chains
│   └── 03_category_exposure.sql # Revenue concentration by department + customer tier
│
├── outputs/
│   └── charts/
│       ├── rfm_segments.png         # Segment distribution bar chart
│       ├── top20_shrinking_customers.png  # Revenue at risk — top 20
│       └── category_heatmap.png     # Department × segment heatmap
│
└── docs/
    ├── decision_log.md          # 8 analytical decisions and their rationale
    └── analyst_notes.md         # Interview-style project walkthrough
```

---

## Tech Stack

| Tool | Version | Purpose |
|---|---|---|
| Python | 3.11+ | Pipeline orchestration |
| pandas | 2.x | Data wrangling |
| NumPy | 1.26+ | Vectorized regression |
| SciPy | 1.12+ | t-distribution for p-values |
| matplotlib | 3.8+ | Charts |
| seaborn | 0.13+ | Heatmap |

---

## Dataset

Instacart Market Basket Analysis — publicly released by Instacart for a 2017 Kaggle competition.

The dataset contains anonymized order histories from ~206K users, including product names, aisles, departments, and order timing (but no timestamps or prices). Download from: [Kaggle — Instacart Market Basket Analysis](https://www.kaggle.com/c/instacart-market-basket-analysis/data)

**Files needed in `data/raw/`:**
- `orders.csv` (~130MB)
- `order_products__prior.csv` (~577MB — handled with chunked loading)
- `order_products__train.csv` (~25MB)
- `products.csv`
- `aisles.csv`
- `departments.csv`

---

## Technical Notes

- `order_products__prior.csv` is loaded in 500K-row chunks with int32/int16/int8 dtype casting, reducing memory from ~4GB to ~600MB
- Downstream scripts use `usecols=` to load only needed columns from the 3.3GB master file
- Revenue projections use $3.50/item as a grocery spend proxy (no price data in source)
- The regression window (5 orders) and significance threshold (p < 0.05) are configurable in `03_shrinkage.py`

---

## Author

Built as a portfolio project demonstrating end-to-end analytics engineering: from raw data ingestion through statistical modeling, business interpretation, and SQL translation.
