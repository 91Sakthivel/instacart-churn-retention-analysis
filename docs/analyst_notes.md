# Analyst Notes — Project Walkthrough

*Written as if I'm walking a hiring manager through this project in an interview.*

---

## What this project is and why I built it

I wanted to demonstrate something beyond "clean the data, train a model, report accuracy." This project asks a business question that a real grocery retailer would actually care about: which customers are quietly buying less, and what are we about to lose?

The framing matters. It's not a churn model — it's a revenue protection system. Those are different things with different actions. A churn model tells you someone left. This project tells you someone is *leaving*, before they're gone, with a dollar value attached.

---

## The dataset

Instacart released 3 million+ orders from ~206K customers as a Kaggle competition dataset. It has the full purchase history — every item in every order, with the day of week, hour, and days since the prior order. What it *doesn't* have is actual timestamps or prices. Working around those limitations is part of what makes this interesting.

The 33.8 million order-product rows took some care to handle efficiently. The `order_products__prior.csv` file is 577MB and would bloat to 4GB with default pandas dtypes, so I used dtype optimization and chunked loading. For downstream scripts I used `usecols` to load only what I needed.

---

## What I actually found

**On shrinkage:** About 2.8% of customers with sufficient order history (roughly 4,900 people) show a statistically significant downward trend in basket size. That sounds small, but the projected 90-day revenue impact is $1.9M — and that's using a conservative $3.50/item proxy. The distribution is skewed too: the top customer alone accounts for nearly $60K of that risk, with orders averaging just 1 day apart and a slope of -4.1 items per order.

**On segments:** The RFM breakdown came out clean — 27% Champions, 15% Loyal, 17% At Risk, 41% Lost. The separation makes sense: Champions average 34 orders and 362 items total, Lost customers average 6 orders and 55 items. The scoring approach (rank percentile → quartile cut) is more robust than qcut for this dataset because `days_since_prior_order` is capped at 30, creating a lot of ties that qcut handles poorly.

**On category exposure:** When I filtered to Champions and Loyal customers who are also shrinking, I got 2,390 customers driving $626K in 60-day exposure. Produce is the most exposed category ($183K), followed by dairy/eggs ($101K). This is actually a meaningful finding — if you're a produce buyer at a grocery chain and your numbers are softening, this analysis tells you it's not a shelf presentation problem, it's a customer engagement problem.

**On the intervention timing:** This is where the dataset limitation is most honest. The Instacart data doesn't contain true churn — it's a snapshot of purchase history, and "Lost" is an RFM label, not a confirmed unsubscribe. The recovery rate analysis (looking at customers who had consecutive declining orders) showed ~93% of customers ordered again regardless of streak length. That's not a surprising result for a dataset of active purchasers. In a production setting with real churn data, this analysis would give a sharp number. Here, the honest answer is "the intervention window is broader than this dataset can show" — and saying that in an interview demonstrates you're not just reporting numbers you don't trust.

---

## What I'd do differently with real production data

**Real timestamps.** The days_since_prior_order proxy works, but a real timestamp would let me segment by recency from today, model seasonal effects, and identify customers who went quiet during a specific event (e.g., after a price change).

**Transaction values.** The $3.50/item proxy is directionally correct but hard to act on with precision. With real spend data, I'd weight the revenue projections by customer lifetime value rather than item count.

**More sophisticated trend detection.** OLS on 5 points is a clean starting point, but a CUSUM (cumulative sum) detector or exponentially weighted moving average would be more sensitive to recent changes. Worth testing if this were a live system with daily updates.

**Labelled churn.** The intervention deadline analysis (script 04) is the weakest part of this project precisely because Instacart doesn't have true churn labels. In a real retail system, I'd define churn as "no order in 90 days" and work backwards from there to find the pre-churn signals.

---

## What the business would actually do with this

The 4,900 flagged customers would go into a CRM workflow — probably an automated email series triggered by the shrinkage flag, offering something relevant to their purchase history rather than a generic coupon. The intervention should happen at the 2nd or 3rd declining order, not after 5.

The category exposure report ($626K across produce, dairy, snacks, beverages) would go to the category management team. "Your produce numbers are softening because your best customers are buying less, not because you're losing shelf share" is a very different conversation than a standard sales variance report.

The last-hook department analysis (48% of churned customers last bought from produce) gives the win-back campaign a clear hook. A produce-focused offer to churned customers is more likely to resonate than a blanket discount.

---

## Why I'd put this in a portfolio

Most junior analyst portfolios show the same things: EDA on a Kaggle dataset, a classification model, accuracy metrics. This project shows something different — the ability to frame a business problem, design an analysis that produces actionable outputs, and be honest about the limitations of the data. That's what separates someone who can run code from someone who can actually help a business make decisions.
