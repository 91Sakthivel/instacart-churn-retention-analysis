-- dim_trajectory.sql
--
-- Dimension table for the four customer trajectory types (A / B / C / D).
-- This drives the Intervention & ROI page in the Streamlit dashboard and
-- provides the lookup values for offer costs, recovery rates, and ROI
-- that feed into fact_customer_risk.estimated_save_value.
--
-- All financial and probability figures come from the validated pipeline:
--   - Churn rates: validated in pipeline/validate_signals.py
--   - ROI figures: modelled in pipeline/05b_trajectory.py
--   - Offer costs: business assumptions documented in executive_summary.txt
--
-- Trajectory classification rules (for reference):
--   A — Fading Frequency:    avg_days > 18 AND gap_trend > 3 AND reorder_rate 0.35–0.55
--   B — Sudden Stop:         early_avg_gap < 12 AND late_avg_gap_clean > 20
--   D — High Value Drifting: reorder_rate >= 0.55 AND gap_trend > 5
--   C — Low Loyalty:         default (does not meet A, B, or D criteria)
--
-- recovery_rate is stored as a decimal (e.g. 0.35 = 35%) so that
-- fact_customer_risk can compute estimated_save_value directly:
--   estimated_save_value = predicted_revenue_loss * recovery_rate

with trajectory_metadata as (

    select
        'A'                                 as trajectory_type,
        'Fading Frequency'                  as trajectory_name,
        0.445                               as churn_probability,
        'Product discount'                  as recommended_offer,
        cast(11.00 as float64)              as offer_cost,
        cast(0.35  as float64)              as recovery_rate,
        6075                                as roi_pct,
        'Trigger within the next 4 to 6 orders'
                                            as intervention_window

    union all

    select
        'B',
        'Sudden Stop',
        0.350,
        'Personal phone call',
        cast(15.00 as float64),
        cast(0.20  as float64),
        3820,
        'Within 48 hours of stop being detected'

    union all

    select
        'D',
        'High Value Drifting',
        0.223,
        'Loyalty reward',
        cast(8.50 as float64),
        cast(0.40 as float64),
        17891,
        'Before gap exceeds 20 days'

    union all

    select
        'C',
        'Low Loyalty',
        0.149,
        'Push notification',
        cast(0.50 as float64),
        cast(0.05 as float64),
        43849,
        'Light touch only — do not trigger expensive campaigns'

),

customer_counts as (

    select
        trajectory_type,
        count(user_id)  as n_customers

    from {{ ref('stg_trajectory_segments') }}
    group by trajectory_type

)

select
    m.trajectory_type,
    m.trajectory_name,
    m.churn_probability,
    m.recommended_offer,
    m.offer_cost,
    m.recovery_rate,
    m.roi_pct,
    m.intervention_window,
    coalesce(c.n_customers, 0)  as n_customers

from trajectory_metadata           as m
left join customer_counts          as c
    on m.trajectory_type = c.trajectory_type

order by m.churn_probability desc
