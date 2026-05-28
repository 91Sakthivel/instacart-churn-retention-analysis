-- fact_customer_risk.sql
--
-- Central fact table — the single source of truth for the Streamlit dashboard
-- and any BI tool connected to BigQuery.
--
-- One row per customer (175,072 total — all customers in the churn model).
-- Joins int_customer_risk (behavioral features + RFM + trajectory risk scores)
-- to dim_segment and dim_trajectory to enrich each customer row with
-- plain-English labels, intervention guidance, and ROI-based save estimates.
--
-- Key computed column:
--   estimated_save_value = predicted_revenue_loss × recovery_rate
--   This is the expected dollar value recovered per customer if the
--   recommended intervention is applied and succeeds at the modelled rate.
--   NULL for customers without a trajectory classification (<10 orders).
--
-- Join logic:
--   int_customer_risk  → dim_segment   (INNER — every modelled customer has a segment)
--   int_customer_risk  → dim_trajectory (LEFT  — customers with <10 orders have NULL type)

with customer_risk as (

    select * from {{ ref('int_customer_risk') }}

),

segments as (

    select * from {{ ref('dim_segment') }}

),

trajectories as (

    select * from {{ ref('dim_trajectory') }}

),

enriched as (

    select
        -- Identity
        cr.user_id,

        -- Behavioral features (from customer_features pipeline output)
        cr.avg_days_between_orders,
        cr.order_count,
        cr.reorder_rate,
        cr.gap_trend,
        cr.churn_label,
        cr.revenue_at_risk_90d,

        -- Model outputs (from trajectory_segments / stg_trajectory_segments)
        cr.churn_probability,
        cr.predicted_revenue_loss,
        cr.risk_score,
        cr.risk_tier,

        -- RFM segment identity
        cr.segment,
        cr.rfm_score,

        -- Segment dimension enrichment
        seg.segment_description,
        seg.intervention_priority,
        seg.recommended_action,

        -- Trajectory dimension enrichment
        cr.trajectory_type,
        traj.trajectory_name,
        traj.recommended_offer,
        traj.offer_cost,
        traj.recovery_rate,
        traj.roi_pct,
        traj.intervention_window,

        -- Estimated value of a successful intervention
        -- NULL where trajectory_type is NULL (customers with <10 orders)
        round(
            cr.predicted_revenue_loss * traj.recovery_rate,
            2
        )                                       as estimated_save_value

    from customer_risk          as cr

    inner join segments         as seg
        on cr.segment = seg.segment

    left join trajectories      as traj
        on cr.trajectory_type = traj.trajectory_type

)

select * from enriched
order by risk_score desc
