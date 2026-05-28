-- int_customer_risk.sql
--
-- Intermediate model that brings together behavioral features, RFM segments,
-- and trajectory classifications into a single customer-level risk profile.
--
-- Why this intermediate layer exists:
-- The staging tables serve different populations — customer_features has 175,072
-- customers, rfm_segments has 206,209, trajectory_segments has 107,438.
-- Joining them here means the mart models don't each need to repeat the join logic.
-- Customers not in trajectory_segments get NULL for risk columns, which the
-- risk_tier CASE maps to 'Low' — a reasonable default for unclassified customers.
--
-- risk_tier is a business-friendly bucketing of risk_score for dashboards
-- and intervention prioritisation. Thresholds were set based on the score
-- distribution across the 175K modelled customers.

with customer_features as (

    select * from {{ ref('stg_customer_features') }}

),

rfm as (

    select * from {{ ref('stg_rfm_segments') }}

),

trajectory as (

    select * from {{ ref('stg_trajectory_segments') }}

),

joined as (

    select
        cf.user_id,

        -- Behavioral features
        cf.avg_days_between_orders,
        cf.order_gap_std,
        cf.order_gap_cv,
        cf.early_avg_gap,
        cf.late_avg_gap_clean,
        cf.order_count,
        cf.gap_trend,
        cf.reorder_rate,
        cf.churn_label,
        cf.revenue_at_risk_90d,
        cf.annual_value_est,

        -- RFM segment
        rfm.segment,
        rfm.r_score,
        rfm.f_score,
        rfm.m_score,
        rfm.rfm_score,

        -- Trajectory and model outputs (null for customers with <10 orders)
        traj.trajectory_type,
        traj.churn_probability,
        traj.predicted_revenue_loss,
        traj.risk_score,

        -- Business-friendly risk tier derived from composite risk_score
        case
            when traj.risk_score >= 0.7 then 'Critical'
            when traj.risk_score >= 0.4 then 'High'
            when traj.risk_score >= 0.2 then 'Medium'
            else 'Low'
        end as risk_tier

    from customer_features as cf
    inner join rfm
        on cf.user_id = rfm.user_id
    left join trajectory as traj
        on cf.user_id = traj.user_id

)

select * from joined
