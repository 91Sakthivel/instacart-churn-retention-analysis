-- stg_rfm_segments.sql
--
-- Staging layer for RFM (Recency, Frequency, Monetary) segments.
-- Covers all 206,209 customers — including those with fewer than 5 orders
-- who were excluded from the churn model but still have an RFM profile.
--
-- Scoring: each dimension was ranked with rank(pct=True) and cut into
-- quartiles, producing scores 1–4. Segments were then assigned:
--   Champions  → r_score >= 3 AND f_score >= 3 AND m_score >= 3
--   Loyal      → r_score >= 3 AND f_score >= 2
--   At Risk    → r_score <= 2 AND f_score >= 3
--   Lost       → everything else

select
    user_id,
    segment,
    r_score,
    f_score,
    m_score,
    rfm_score

from {{ source('instacart_raw', 'rfm_segments') }}
