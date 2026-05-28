-- stg_trajectory_segments.sql
--
-- Staging layer for customer trajectory classifications.
-- Only covers customers with 10+ orders (107,438 of the 175,072 in the model).
-- Customers with fewer orders don't have enough history to classify reliably.
--
-- The four trajectory types and their validated churn rates (from pipeline validation):
--   B (Sudden Stop)         — early_avg_gap < 12 AND late_avg_gap_clean > 20.
--                             Was ordering frequently, then stopped abruptly.
--                             35.0% empirical churn rate.
--   D (High Value Drifting) — reorder_rate >= 0.55 AND gap_trend > 5.
--                             Loyal customer whose gaps are slowly growing.
--                             22.3% empirical churn rate.
--   A (Fading Frequency)    — avg_days > 18 AND gap_trend > 3.
--                             Gaps steadily widening. 44.5% empirical churn rate.
--   C (Low Loyalty)         — default for customers not meeting A/B/D criteria.
--                             Never showed strong engagement. 14.9% churn rate.
--
-- Column derivations:
--   churn_probability   — assigned using empirical churn rate per trajectory type.
--                         The source table stores only the binary churned flag;
--                         per-customer XGBoost scores were not loaded to this table.
--                         Using group-level rates is the analytically honest proxy.
--   predicted_revenue_loss — annual_value * churn_probability (expected revenue lost).
--   risk_score          — churn_probability * annual_value * 0.25 (90-day exposure),
--                         consistent with the revenue_at_risk_90d methodology in
--                         customer_features (avg_basket_size × annual_orders_est × 0.25).

select
    user_id,
    trajectory_type,

    -- Empirical churn probability derived from trajectory group rates
    -- (validated in pipeline/validate_signals.py and executive_summary.txt)
    case trajectory_type
        when 'A' then 0.445
        when 'B' then 0.350
        when 'D' then 0.223
        when 'C' then 0.149
        else 0.0
    end                                                         as churn_probability,

    -- Expected revenue loss: annual value × group churn rate
    annual_value * case trajectory_type
        when 'A' then 0.445
        when 'B' then 0.350
        when 'D' then 0.223
        when 'C' then 0.149
        else 0.0
    end                                                         as predicted_revenue_loss,

    -- risk_score is churn_probability itself (kept in [0,1]).
    -- The risk_tier CASE in int_customer_risk uses thresholds 0.7 / 0.4 / 0.2,
    -- so risk_score must stay on a probability scale.
    -- Dollar exposure is captured separately in predicted_revenue_loss.
    case trajectory_type
        when 'A' then 0.445
        when 'B' then 0.350
        when 'D' then 0.223
        when 'C' then 0.149
        else 0.0
    end                                                         as risk_score

from {{ source('instacart_raw', 'trajectory_segments') }}
