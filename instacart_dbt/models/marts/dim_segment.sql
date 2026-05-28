-- dim_segment.sql
--
-- Dimension table for the four RFM segments.
-- Designed for BI tools — one row per segment, stable keys, human-readable labels.
-- Join this to fact_customer_risk on segment to enrich dashboards with
-- plain-English descriptions and intervention guidance without repeating
-- that logic in every downstream model or report.
--
-- Segment counts are pulled live from stg_rfm_segments so they update
-- automatically if the pipeline reruns with new data.

with segment_metadata as (

    -- Hardcoded business rules for each segment.
    -- Champions = high on all three RFM dimensions.
    -- Loyal = frequent + recent but not top spenders.
    -- At Risk = were strong, now going quiet — highest urgency.
    -- Lost = low across the board — lowest ROI to target aggressively.

    select 'Champions'  as segment,
           'Most engaged customers — recent, frequent, high basket value. Core of the business.'
                        as segment_description,
           'High'       as intervention_priority,
           'Loyalty rewards and early access. Protect this segment above all others.'
                        as recommended_action

    union all

    select 'Loyal',
           'Frequent buyers with good recency. Not top spenders but consistent revenue.',
           'Medium',
           'Personalised product recommendations. Upsell basket size.'

    union all

    select 'At Risk',
           'Were active, now ordering less. Recency declining but frequency history is strong.',
           'High',
           'Re-engagement campaign before they slide to Lost. Act within the next 2 orders.'

    union all

    select 'Lost',
           'Low recency, frequency, and monetary. Minimal engagement history.',
           'Low',
           'Light touch only — a single win-back email or push notification. Do not overspend.'

),

customer_counts as (

    select
        segment,
        count(user_id)  as n_customers

    from {{ ref('stg_rfm_segments') }}
    group by segment

)

select
    m.segment,
    m.segment_description,
    m.intervention_priority,
    m.recommended_action,
    coalesce(c.n_customers, 0)  as n_customers

from segment_metadata              as m
left join customer_counts          as c
    on m.segment = c.segment

order by
    case m.intervention_priority
        when 'High'   then 1
        when 'Medium' then 2
        when 'Low'    then 3
        else 4
    end
