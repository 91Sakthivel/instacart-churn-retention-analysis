-- mart_retention_priority.sql
--
-- Retention priority list for the customer success and marketing teams.
-- Contains only Critical and High risk customers — the ones worth spending
-- real intervention budget on.
--
-- Sorted by risk_score descending so the highest-priority customers are
-- always at the top. This is what the retention team works from daily.
--
-- Exclusion logic: Medium and Low risk_tier customers are intentionally
-- excluded. Sending expensive interventions (phone calls, loyalty rewards)
-- to Type C / Low Loyalty customers has been modelled to waste up to
-- $980,584 in a single campaign cycle. See executive_summary.txt for detail.

select
    icr.user_id,
    icr.segment,
    icr.trajectory_type,
    icr.risk_tier,
    icr.risk_score,
    icr.churn_probability,
    icr.predicted_revenue_loss,
    icr.revenue_at_risk_90d,
    icr.avg_days_between_orders,
    icr.reorder_rate

from {{ ref('int_customer_risk') }} as icr

where icr.risk_tier in ('Critical', 'High')

order by icr.risk_score desc
