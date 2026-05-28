-- stg_customer_features.sql
--
-- Staging layer for customer behavioral features.
-- Selects the columns used by the churn model and downstream marts.
-- 175,072 customers who placed 5+ orders — customers below that threshold
-- were excluded from the pipeline and are not in this source table.
--
-- Key leakage note: days_since_last_order is NOT selected here.
-- It equals the churn label threshold (>= 30 = churned) and must never
-- appear as a model feature. See pipeline/08_churn_model.py for details.
--
-- Column rename: source table uses `churned` (pipeline convention).
-- We alias it as `churn_label` here to make the binary nature clear
-- and distinguish it from model-predicted churn probabilities downstream.

select
    user_id,
    avg_days_between_orders,
    order_gap_std,
    order_gap_cv,
    early_avg_gap,
    late_avg_gap_clean,
    order_count,
    gap_trend,
    reorder_rate,
    churned                     as churn_label,
    revenue_at_risk_90d,
    annual_value_est

from {{ source('instacart_raw', 'customer_features') }}
