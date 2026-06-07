# Decision Log — Instacart Basket Shrinkage & Revenue Protection

Eight choices I made in this project, and why I made them.

---

## 1. Linear regression over moving average for shrinkage detection

I chose linear regression because it gives me two things a moving average can't: a direction (slope) and a confidence level (p-value). With a moving average, I can tell a customer's basket is getting smaller, but I can't distinguish between a real trend and random week-to-week noise. A customer who ordered 8, 9, 7, 11, 6 items might look like they're declining on the last order, but the regression says the trend is basically flat (and the p-value confirms it). That distinction matters — I don't want to send a retention offer to someone who's fine.

The downside is that OLS on 5 data points is noisy. A few outlier baskets can flip the slope. I accepted that tradeoff because the p-value filter catches it — only slopes significant at p < 0.05 make it into the flagged set.

---

## 2. Five-order window for the regression

I tried 3, 5, 7, and 10 orders. Three is too sensitive — one big shop followed by one small shop and you're flagged. Ten orders dilutes recent signals; a customer who was stable for 8 months and has been declining for 2 looks fine in a 10-order window.

Five struck the right balance: recent enough to catch something that started a few weeks ago, but enough data points to have meaningful degrees of freedom (df = n - 2 = 3 for the t-test). It also made practical sense — customers with fewer than 5 orders don't have enough history to trend yet.

The 88.4% eligibility rate (182K of 206K customers) was a good signal that 5 was the right floor.

---

## 3. Item count as the "Monetary" proxy

The Instacart public dataset has no price data. I could have used basket size alone, or tried to weight by department (produce vs. alcohol have very different average prices), but those approaches introduce assumptions I can't validate.

I went with raw item count because it's honest about what it is — a proxy for engagement depth, not actual dollars. When I multiply by $3.50/item for the revenue projection, I'm explicit about the assumption. A hiring manager reviewing this is going to ask "where did $3.50 come from?" and I want to have a clean answer: it's a conservative grocery average, not a made-up number, and the analysis is directionally correct regardless of the exact value.

---

## 4. Vectorized OLS instead of groupby-apply

The naive implementation — `df.groupby('user_id').apply(scipy.linregress)` — would take 3–5 minutes for 175K customers because it's a Python loop under the hood. I replaced it with a fully vectorized NumPy implementation that runs 175K regressions in under a second.

The math is identical to `scipy.linregress` for a fixed x = [1,2,3,4,5]: I precompute the constant terms (x̄ = 3, Sxx = 10), then do a single matrix multiply across all customers at once. Slopes, residuals, standard errors, t-stats, and p-values all computed as array operations. The tradeoff is readability — someone unfamiliar with OLS math will need to read the comments carefully to follow it. I added detailed comments specifically to address that.

---

## 5. Four RFM segments instead of the classic eight

The traditional RFM framework has 8–10 segments (Champions, Loyal, Potential Loyalists, Recent Customers, Promising, Needing Attention, At Risk, Can't Lose, Hibernating, Lost). I collapsed this to four because:

- This project has a specific revenue-protection goal, not a general CRM segmentation exercise
- The business actions collapse to roughly four responses anyway: retain Champions, develop Loyal, re-engage At Risk, try to win back Lost
- More segments would require more segment-specific business rules to be useful, and I'd be inventing them without real data to back them

The four segments are meaningfully differentiated — the average metrics table in the script output confirms they separate cleanly.

---

## 6. Triangular sum for 90-day revenue projection

When projecting revenue at risk over 90 days, I used `|slope| * N*(N+1)/2` rather than just `|slope| * N * last_basket`.

The simpler version (slope × number of orders) assumes the customer loses one basket-size unit at every single future order — it measures the total loss as if the decline hits all at once. The triangular sum is the actual cumulative area under the decline curve: the first future order loses `slope` items, the second loses `2 * slope`, and so on.

For a customer with slope = -2 and 9 orders in 90 days, the simple version overstates risk by 5x. The triangular version gives the correct cumulative answer. It's a small detail but it matters when you're handing a number to someone who's going to put it in a slide.

---

## 7. Loading the 577MB file in chunks with dtype optimization

`order_products__prior.csv` is 577MB on disk and would load to ~4GB in memory with default int64/float64 dtypes. I handled it two ways:

First, I cast integer columns to int32 and int16 at load time, cutting memory roughly in half. Product IDs max out around 50K and order IDs around 3.4M — both fit in int32. The `reordered` column is 0/1 — int8.

Second, I loaded in 500K-row chunks and concatenated, printing progress every million rows. This is slower than a single read but keeps peak memory bounded and gives the user feedback so they don't assume the process is frozen.

For downstream scripts that only need a subset of columns, I used `usecols=` to skip the string columns entirely — that's where most of the memory goes.

---

## 8. Using recency_days as the recency metric instead of a true timestamp

Instacart's public dataset doesn't include actual order dates. The `days_since_prior_order` column gives the gap between consecutive orders, not the distance from today.

I used the `days_since_prior_order` at each customer's most recent order as the recency proxy. This captures the gap between their last and second-to-last purchase — a customer with a 3-day gap is more "in the habit" than one with a 28-day gap, which is the behavior we care about for RFM scoring.

The limitation I'd flag in a presentation: this measures recency of habit, not recency from today. A customer who ordered 10 days ago after a 7-day gap looks "more recent" than one who ordered 3 days ago after a 30-day gap. In production with a live database this would be a real timestamp. For this dataset it's the closest available proxy, and the segment averages came out as expected (Champions: 6.7 day average gap, Lost: 23.1 days).
