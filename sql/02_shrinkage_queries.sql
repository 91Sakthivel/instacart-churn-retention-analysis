-- ============================================================
-- 02_shrinkage_queries.sql  —  Basket Shrinkage Risk by Segment
-- ============================================================
-- Joins customer_features (regression outputs from Python)
-- with rfm_segments to produce a segment-level risk dashboard.
--
-- Business question answered:
--   Which customer segments have the most basket shrinkage,
--   how severe is it, and what is the projected revenue impact?
--
-- Demonstrates: JOIN, conditional aggregation (CASE WHEN inside
-- SUM/AVG), ROUND arithmetic, window SUM for percentage base,
-- CASE for business-readable severity labels.
-- ============================================================

WITH

-- -------------------------------------------------------
-- Classify each customer's shrinkage severity before
-- aggregating, so the GROUP BY can count severity tiers
-- directly without nested CASE in the aggregate.
-- -------------------------------------------------------
enriched AS (
    SELECT
        cf.user_id,
        rf.segment,
        cf.slope,
        cf.p_value,
        cf.avg_basket_size,
        cf.last_basket_size,
        cf.avg_days_between_orders,
        cf.shrinkage_velocity,
        cf.revenue_at_risk_90d,
        cf.orders_in_90d,
        -- Shrinking = statistically significant negative slope
        CASE WHEN cf.shrinking_flag = 1 THEN 1 ELSE 0 END AS is_shrinking,
        CASE
            WHEN cf.shrinking_flag = 0         THEN 'None'
            WHEN cf.shrinkage_velocity > -0.10 THEN 'Mild'
            WHEN cf.shrinkage_velocity > -0.20 THEN 'Moderate'
            ELSE                                    'Severe'
        END AS severity
    FROM customer_features cf
    JOIN rfm_segments rf ON cf.user_id = rf.user_id
),

-- -------------------------------------------------------
-- Segment-level aggregation.
-- Conditional SUM/AVG with CASE WHEN is the SQL equivalent
-- of pandas groupby with filtered aggregations.
-- -------------------------------------------------------
segment_risk AS (
    SELECT
        segment,
        COUNT(*)                                                        AS eligible_customers,
        SUM(is_shrinking)                                               AS shrinking_count,
        ROUND(SUM(is_shrinking) * 100.0 / COUNT(*), 1)                  AS pct_shrinking,

        -- Revenue totals — only shrinking customers contribute non-zero risk
        ROUND(SUM(revenue_at_risk_90d), 0)                              AS total_revenue_at_risk,
        ROUND(AVG(CASE WHEN is_shrinking = 1
                       THEN revenue_at_risk_90d END), 0)                AS avg_risk_per_customer,

        -- Severity breakdown
        SUM(CASE WHEN severity = 'Severe'   THEN 1 ELSE 0 END)          AS severe_count,
        SUM(CASE WHEN severity = 'Moderate' THEN 1 ELSE 0 END)          AS moderate_count,
        SUM(CASE WHEN severity = 'Mild'     THEN 1 ELSE 0 END)          AS mild_count,

        -- Avg metrics for shrinking customers only
        ROUND(AVG(CASE WHEN is_shrinking = 1 THEN slope              END), 4) AS avg_slope,
        ROUND(AVG(CASE WHEN is_shrinking = 1 THEN avg_days_between_orders END), 1)
                                                                        AS avg_order_gap_days,
        ROUND(AVG(CASE WHEN is_shrinking = 1 THEN avg_basket_size    END), 1) AS avg_basket_size
    FROM enriched
    GROUP BY segment
)

-- -------------------------------------------------------
-- Final output: add cross-segment % share of total risk
-- using a window SUM so we don't need a separate subquery.
-- -------------------------------------------------------
SELECT
    segment,
    eligible_customers,
    shrinking_count,
    pct_shrinking,
    total_revenue_at_risk,

    -- Each segment's share of the total portfolio revenue risk
    ROUND(
        total_revenue_at_risk * 100.0 / SUM(total_revenue_at_risk) OVER (),
        1
    )                                                                   AS pct_of_total_risk,

    avg_risk_per_customer,
    severe_count,
    moderate_count,
    mild_count,
    avg_slope,
    avg_order_gap_days,
    avg_basket_size

FROM segment_risk
ORDER BY total_revenue_at_risk DESC
