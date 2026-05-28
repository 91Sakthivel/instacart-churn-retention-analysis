-- dim_department.sql
--
-- Dimension table for the 21 product departments.
-- Drives the Category Exposure page in the Streamlit dashboard.
-- Source of truth for department-level revenue risk, segment concentration,
-- and Pareto tier classification (top 7 departments account for ~80% of
-- the $626,303 in total 60-day category revenue at risk).
--
-- champions_share_pct = Champions-segment dollar risk / total dept risk.
-- This tells you what fraction of each department's at-risk revenue comes
-- from your best customers. High champions_share means losing even a few
-- Champions would disproportionately hurt this department.
--
-- Note: stg_customer_features contains top_department per customer.
-- For customer-level churn by department, join stg_customer_features
-- to this table on department_name = top_department.

with dept_raw as (

    select
        department                                                  as department_name,
        dept_total_risk                                             as risk_60d,
        Champions                                                   as champions_risk,
        n_customers,
        pct_of_total_risk,

        -- Rank departments by 60-day risk exposure (highest = rank 1)
        row_number() over (order by dept_total_risk desc)           as dept_rank

    from {{ source('instacart_raw', 'category_risk_report') }}

)

select
    department_name,
    round(risk_60d, 2)                                              as risk_60d,

    -- Top 7 departments account for ~80% of total exposure (Pareto)
    case
        when dept_rank <= 7 then 'Core Risk'
        else 'Standard'
    end                                                             as risk_tier,

    -- Share of this department's risk attributable to Champions customers
    round(
        safe_divide(champions_risk, risk_60d) * 100,
        1
    )                                                               as champions_share_pct,

    n_customers,
    round(pct_of_total_risk, 2)                                     as pct_of_total_risk,
    dept_rank

from dept_raw
order by dept_rank
