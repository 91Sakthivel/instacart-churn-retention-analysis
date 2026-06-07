-- ============================================================
-- 01_rfm_queries.sql  —  RFM Cohort Analysis
-- ============================================================
-- Reads from the rfm_segments table built by pipeline/02_rfm.py.
-- Demonstrates: NTILE(), window aggregates, conditional sums,
-- and multi-level GROUP BY with percentage calculations.
--
-- Business question answered:
--   How do our four customer segments compare on spending
--   depth, purchase frequency, and recency? Where are the
--   monetary decile concentrations per segment?
--
-- Returns one row per segment (4 rows total) plus a breakdown
-- of each segment's share of the top-20% spenders.
-- ============================================================

WITH

-- Assign each customer to a monetary decile (1=bottom, 10=top).
-- We use NTILE(10) here rather than NTILE(4) so we can measure
-- concentration — a segment that's "Loyal" in name should be
-- stacking up in deciles 7–10, not 1–3.
monetary_deciles AS (
    SELECT
        user_id,
        segment,
        recency_days,
        frequency,
        monetary,
        r_score,
        f_score,
        m_score,
        NTILE(10) OVER (ORDER BY monetary ASC)  AS monetary_decile,
        NTILE(10) OVER (ORDER BY frequency ASC) AS frequency_decile
    FROM rfm_segments
),

-- Segment-level aggregates. FILTER isn't in SQLite so we use CASE.
segment_summary AS (
    SELECT
        segment,
        COUNT(*)                                              AS n_customers,
        ROUND(AVG(recency_days), 1)                          AS avg_recency_days,
        ROUND(AVG(frequency),   1)                           AS avg_orders,
        ROUND(AVG(monetary),    0)                           AS avg_items_total,
        MIN(monetary)                                        AS min_items,
        MAX(monetary)                                        AS max_items,
        -- Top-20% spenders in each segment (decile >= 9)
        SUM(CASE WHEN monetary_decile >= 9 THEN 1 ELSE 0 END) AS n_top20pct_spenders,
        -- Customers with above-median frequency (decile >= 6)
        SUM(CASE WHEN frequency_decile >= 6 THEN 1 ELSE 0 END) AS n_above_median_freq,
        -- Average RFM component scores (sanity check — Champions should be 3–4 across)
        ROUND(AVG(CAST(r_score AS REAL)), 2)                 AS avg_r_score,
        ROUND(AVG(CAST(f_score AS REAL)), 2)                 AS avg_f_score,
        ROUND(AVG(CAST(m_score AS REAL)), 2)                 AS avg_m_score
    FROM monetary_deciles
    GROUP BY segment
),

-- Add percentage columns using a window SUM for the total denominator.
-- This avoids a subquery or join to get the overall total.
with_pcts AS (
    SELECT
        segment,
        n_customers,
        ROUND(
            n_customers * 100.0 / SUM(n_customers) OVER (),
            1
        )                                                    AS pct_of_base,
        avg_recency_days,
        avg_orders,
        avg_items_total,
        min_items,
        max_items,
        n_top20pct_spenders,
        ROUND(
            n_top20pct_spenders * 100.0 / n_customers,
            1
        )                                                    AS pct_top20pct_spenders,
        n_above_median_freq,
        avg_r_score,
        avg_f_score,
        avg_m_score
    FROM segment_summary
)

SELECT *
FROM with_pcts
ORDER BY
    -- Sort in business priority order, not alphabetical
    CASE segment
        WHEN 'Champions' THEN 1
        WHEN 'Loyal'     THEN 2
        WHEN 'At Risk'   THEN 3
        WHEN 'Lost'      THEN 4
        ELSE                  5
    END
