# instacart_dbt

dbt project for the Instacart Churn & Customer Retention Analysis.

This layer sits on top of the Python pipeline outputs that were loaded into BigQuery,
and transforms them into analysis-ready tables for the Streamlit dashboard and
ad-hoc stakeholder queries.

**Live dashboard:** https://instacart-churn-retention-analysis-dashboard.streamlit.app/  
**GitHub (full project):** https://github.com/91Sakthivel/instacart-churn-retention-analysis

---

## What this project does

The Python pipeline (scripts 01–09) built a churn prediction model across 206,209
Instacart customers — XGBoost AUC 0.9059, trained on 8 behavioral features with
three rounds of leakage checking. The pipeline outputs 7 CSV files that were loaded
into BigQuery as raw tables.

This dbt project takes those raw tables and produces:
- Clean, tested staging views over each source table
- A joined intermediate model (`int_customer_risk`) combining behavioral features,
  RFM segments, and trajectory classifications into one customer-level risk profile
- Two mart tables for the retention team and executive dashboards

---

## Project structure

```
models/
├── staging/           # Thin wrappers over raw source tables. Views only.
│   ├── sources.yml    # Source declarations for instacart_raw dataset
│   ├── schema.yml     # Column docs and data tests for staging models
│   ├── stg_customer_features.sql
│   ├── stg_rfm_segments.sql
│   └── stg_trajectory_segments.sql
│
├── intermediate/      # Joins and derived columns. Views only.
│   └── int_customer_risk.sql
│
└── marts/             # Final outputs for dashboards and reports. Tables.
    ├── schema.yml
    ├── mart_retention_priority.sql
    └── mart_segment_summary.sql
```

### Staging
Light-touch models — select and rename columns, no joins, no business logic.
Each model maps directly to one source table in `instacart_raw`.

### Intermediate
`int_customer_risk` joins the three staging models into one wide customer table.
It adds `risk_tier` (Critical / High / Medium / Low), a business-friendly bucketing
of the composite risk score. All mart models reference this one intermediate model.

### Marts
- **mart_retention_priority** — Critical and High risk customers sorted by risk score.
  This is the working list for the retention team.
- **mart_segment_summary** — Aggregated by segment × trajectory × risk_tier.
  Feeds executive dashboards and top-level charts.

---

## Source data

**BigQuery project:** `windy-container-451804-n4`  
**Source dataset:** `instacart_raw`  
**Output dataset:** `instacart_dbt`

| Table | Rows | Description |
|---|---|---|
| customer_features | 175,072 | Behavioral features per customer (5+ orders only) |
| rfm_segments | 206,209 | RFM scoring for all customers |
| trajectory_segments | 107,438 | Trajectory type A/B/C/D (10+ orders only) |
| category_risk_report | 21 | Department-level revenue risk |

---

## How to run

Make sure your dbt profile (`~/.dbt/profiles.yml`) is configured for BigQuery
with the `instacart_dbt` profile name.

```bash
# Run all models
dbt run --project-dir C:\Users\Hp\Desktop\instacart-analytics1\instacart_dbt

# Run only staging models
dbt run --select staging

# Run only marts
dbt run --select marts

# Run data quality tests
dbt test

# Generate and serve docs
dbt docs generate
dbt docs serve
```

Run order is handled by dbt's DAG automatically:
`stg_* → int_customer_risk → mart_*`

---

## Data tests

Tests are defined in `staging/schema.yml` and `marts/schema.yml`. Key ones:

- `stg_customer_features.user_id` — unique, not_null
- `stg_rfm_segments.user_id` — unique, not_null
- `stg_rfm_segments.segment` — accepted values: Champions, Loyal, At Risk, Lost
- `stg_trajectory_segments.trajectory_type` — accepted values: A, B, C, D
- `stg_customer_features.churn_label` — accepted values: 0, 1
- `mart_retention_priority.user_id` — unique, not_null

---

## Key numbers (from final pipeline run)

- Best model: XGBoost, AUC = 0.9059, 5-fold CV = 0.9070 ± 0.0013
- Trajectory breakdown: B=6,010 (35% churn), D=9,786 (22.3%), A=2,498 (44.5%), C=89,144 (14.9%)
- 60-day category revenue at risk: $626,303 (produce most exposed: $183,384)
- Top 93,043 customers account for 80% of projected revenue at risk (Pareto)
- Mistargeting Type C with $11 discount = $980,584 wasted per campaign

---

## Tech stack

- **dbt** 1.x with BigQuery adapter (`dbt-bigquery`)
- **BigQuery** — source and output
- **Python pipeline** — XGBoost, scikit-learn, pandas (see `pipeline/` directory)
- **Streamlit** — dashboard (see `streamlit/` directory)
