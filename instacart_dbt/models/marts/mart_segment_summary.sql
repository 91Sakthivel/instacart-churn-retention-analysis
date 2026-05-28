-- mart_segment_summary.sql
--
-- Executive dashboard summary — one row per segment × trajectory_type × risk_tier
-- combination. Designed to feed the top-level charts in the Streamlit dashboard
-- and any BI tool connected to BigQuery.
--
-- Use this for: "how many High-risk Champions do we have?",
-- "what's the average churn probability for Type B Loyal customers?",
-- "where is the most revenue at risk across segments?"
--
-- Note: trajectory_type is NULL for customers with fewer than 10 orders
-- (they were not classified). Those rows are included and group separately
-- so you can see the risk profile of the unclassified population too.

select
    segment,
    trajectory_type,
    risk_tier,
    count(user_id)                  as n_customers,
    avg(churn_probability)          as avg_churn_probability,
    avg(risk_score)                 as avg_risk_score,
    sum(predicted_revenue_loss)     as total_predicted_revenue_loss,
    avg(reorder_rate)               as avg_reorder_rate

from {{ ref('int_customer_risk') }}

group by
    segment,
    trajectory_type,
    risk_tier

order by
    total_predicted_revenue_loss desc
