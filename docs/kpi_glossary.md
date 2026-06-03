\# KPI Glossary — Instacart Churn \& Customer Retention Analysis



\*\*Project:\*\* Instacart Churn \& Customer Retention Analysis  

\*\*Author:\*\* Sakthivel Sivagurunathan  

\*\*Last Updated:\*\* June 2026  

\*\*Data Source:\*\* BigQuery — `windy-container-451804-n4.instacart\_dbt`



\---



\## 1. Churn Rate



| Field | Detail |

|---|---|

| \*\*Definition\*\* | Percentage of customers who have not placed an order within 90 days |

| \*\*Formula\*\* | `Churned Customers / Total Customers × 100` |

| \*\*Data Source\*\* | `fact\_customer\_risk` — `churn\_label` column |

| \*\*Business Interpretation\*\* | Core health metric. A rising churn rate signals deteriorating retention. Used to set baseline for all intervention strategies. |



\---



\## 2. Churn Probability Score



| Field | Detail |

|---|---|

| \*\*Definition\*\* | XGBoost model output — probability (0 to 1) that a customer will churn |

| \*\*Formula\*\* | XGBoost classifier output: `model.predict\_proba(X)\[:, 1]` |

| \*\*Data Source\*\* | `fact\_customer\_risk` — `churn\_probability` column |

| \*\*Business Interpretation\*\* | Used to rank customers by risk. Customers above 0.5 threshold are classified as high-risk and targeted for retention interventions. |



\---



\## 3. Risk Tier



| Field | Detail |

|---|---|

| \*\*Definition\*\* | Categorical bucketing of churn probability into four tiers |

| \*\*Formula\*\* | High: ≥0.75 / Medium-High: 0.50–0.74 / Medium-Low: 0.25–0.49 / Low: <0.25 |

| \*\*Data Source\*\* | `fact\_customer\_risk` — `risk\_tier` column |

| \*\*Business Interpretation\*\* | Enables tiered marketing spend — high-risk customers receive aggressive retention offers; low-risk customers receive lightweight engagement nudges. |



\---



\## 4. Misallocated Retention Spend



| Field | Detail |

|---|---|

| \*\*Definition\*\* | Retention budget spent on customers who were not actually at risk of churning |

| \*\*Formula\*\* | `Spend on Low-Risk Customers / Total Retention Spend × 100` |

| \*\*Data Source\*\* | `mart\_retention\_priority` — derived from `risk\_tier` and spend allocation |

| \*\*Business Interpretation\*\* | $980,584 identified as misallocated in the baseline period. Reallocating this spend to high-risk tiers directly improves ROI. |



\---



\## 5. Retention ROI



| Field | Detail |

|---|---|

| \*\*Definition\*\* | Return on investment from retention interventions targeted at high-risk customers |

| \*\*Formula\*\* | `(Revenue Saved − Intervention Cost) / Intervention Cost × 100` |

| \*\*Data Source\*\* | `mart\_retention\_priority` |

| \*\*Business Interpretation\*\* | Baseline intervention playbook yielded 17,891% ROI by focusing spend on customers with highest churn probability and highest order value. |



\---



\## 6. Customer Lifetime Value (CLV)



| Field | Detail |

|---|---|

| \*\*Definition\*\* | Predicted total revenue a customer will generate over their remaining lifetime |

| \*\*Formula\*\* | `Average Order Value × Purchase Frequency × Expected Customer Lifespan` |

| \*\*Data Source\*\* | `fact\_customer\_risk` — `clv\_score` column |

| \*\*Business Interpretation\*\* | Used to prioritize retention spend — high CLV + high churn risk customers are the highest priority for intervention. |



\---



\## 7. Average Order Value (AOV)



| Field | Detail |

|---|---|

| \*\*Definition\*\* | Average revenue per order for a customer |

| \*\*Formula\*\* | `Total Revenue / Total Orders` |

| \*\*Data Source\*\* | `fact\_customer\_risk` — `avg\_order\_value` column |

| \*\*Business Interpretation\*\* | Higher AOV customers generate more revenue per transaction. Combined with churn risk, identifies high-value customers most worth retaining. |



\---



\## 8. Order Frequency



| Field | Detail |

|---|---|

| \*\*Definition\*\* | Average number of orders placed per customer per month |

| \*\*Formula\*\* | `Total Orders / Active Months` |

| \*\*Data Source\*\* | `fact\_customer\_risk` — `order\_frequency` column |

| \*\*Business Interpretation\*\* | Declining order frequency is a leading indicator of churn. Used as a key feature in the XGBoost churn model. |



\---



\## 9. Days Since Last Order



| Field | Detail |

|---|---|

| \*\*Definition\*\* | Number of days elapsed since the customer's most recent order |

| \*\*Formula\*\* | `Current Date − MAX(order\_date)` |

| \*\*Data Source\*\* | `fact\_customer\_risk` — `days\_since\_last\_order` column |

| \*\*Business Interpretation\*\* | The single strongest churn signal. Customers beyond 60 days since last order enter the high-risk window. |



\---



\## 10. Customer Segment



| Field | Detail |

|---|---|

| \*\*Definition\*\* | RFM-based behavioral cluster assigned to each customer |

| \*\*Formula\*\* | K-Means clustering on Recency, Frequency, Monetary value scores |

| \*\*Data Source\*\* | `dim\_segment` — `segment\_name` column |

| \*\*Business Interpretation\*\* | Segments include Champions, Loyal, At-Risk, Hibernating, and Lost. Each segment receives a distinct retention strategy in the intervention playbook. |



\---



\## 11. Reorder Rate



| Field | Detail |

|---|---|

| \*\*Definition\*\* | Proportion of orders containing at least one reordered product |

| \*\*Formula\*\* | `Orders with reordered = 1 / Total Orders` |

| \*\*Data Source\*\* | `staging` layer — `stg\_order\_products` |

| \*\*Business Interpretation\*\* | High reorder rate signals habitual purchasing behavior — a strong retention indicator. Declining reorder rate precedes churn. |



\---



\## 12. Department Exposure Score



| Field | Detail |

|---|---|

| \*\*Definition\*\* | Number of distinct product departments a customer has ordered from |

| \*\*Formula\*\* | `COUNT(DISTINCT department\_id) per customer` |

| \*\*Data Source\*\* | `dim\_department` |

| \*\*Business Interpretation\*\* | Customers purchasing across more departments show broader engagement and lower churn risk. Narrow department exposure signals vulnerability. |



\---



\## 13. Trajectory Label



| Field | Detail |

|---|---|

| \*\*Definition\*\* | Directional classification of a customer's order frequency trend over time |

| \*\*Formula\*\* | Linear slope of order frequency across rolling 30-day windows — classified as Growing / Stable / Declining |

| \*\*Data Source\*\* | `dim\_trajectory` — `trajectory\_label` column |

| \*\*Business Interpretation\*\* | Declining trajectory customers are flagged for early intervention before they reach the churn threshold. |



\---



\## 14. Intervention Priority Score



| Field | Detail |

|---|---|

| \*\*Definition\*\* | Composite score ranking customers by urgency of retention intervention |

| \*\*Formula\*\* | `Churn Probability × CLV Score` — normalized 0 to 100 |

| \*\*Data Source\*\* | `mart\_retention\_priority` — `priority\_score` column |

| \*\*Business Interpretation\*\* | Combines risk and value into a single actionable rank. Top 20% by priority score receive the highest-intensity interventions. |



\---



\## 15. Model AUC Score



| Field | Detail |

|---|---|

| \*\*Definition\*\* | Area Under the ROC Curve — measures churn model discrimination ability |

| \*\*Formula\*\* | Sklearn `roc\_auc\_score(y\_test, y\_prob)` |

| \*\*Data Source\*\* | Model evaluation — `outputs/model\_evaluation/` |

| \*\*Business Interpretation\*\* | Final XGBoost model achieved AUC 0.906 — meaning 90.6% of the time the model correctly ranks a churner above a non-churner. Industry benchmark for churn models is 0.75+. |

