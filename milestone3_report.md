# Milestone 3 — Customer features and leakage validation
**Project:** How Much Purchase History Is Needed to Predict Future Customer Revenue? A Comparison of Simple and Machine Learning Models  
**Date:** 23 November 2026  
**Status:** Completed; **no ML models have been trained and no accuracy results have been reported.**

## What changed from Milestone 2
The 791,045 cleaned purchase-line rows were transformed into **41,625 customer–prediction-date records**. Each one has alternative features based on 30, 60 and 90 preceding days, plus one shared **actual spending amount in the following 30 days**. For model training these become **124,875 customer–date–history-length rows**.

| Split | Customer–date profiles | Rows counting all three windows | Actually purchased in next 30d |
|---|---:|---:|---:|
| Training | 29,339 | 88,017 | 8,951 |
| Validation | 5,980 | 17,940 | 1,871 |
| Final test | 6,306 | 18,918 | 2,387 |
| **Total** | **41,625** | **124,875** | **13,209** |

Each customer can appear at different historical prediction dates. These are **not 41,625 different customers**; the cleaned source has 5,878 distinct customer IDs. Dependence across dates is an analysis limitation.

## Five features calculated for each history window
1. **Historical spending:** sum of positive invoiced line amounts in `[cutoff-W, cutoff)`.
2. **Orders:** number of distinct invoice IDs in that window.
3. **Recency:** calendar days since the last observed positive sale; if no purchase, use **W+1** (31/61/91 days).
4. **Average order value:** historical gross spending divided by distinct invoices; zero for no orders.
5. **Distinct products:** number of unique stock codes purchased in that window.

**Target:** gross positive invoiced GBP spending during `[cutoff, cutoff+30 days)`. This is a known historical outcome for training/evaluation, never an input feature. It excludes cancellations and refunds and is **not net customer revenue, profit, or causal uplift from advertising**.

### Zero-history cases are important
**21,175** customer–date observations had **no purchases during the previous 30 days**; **8,770** had no purchases during the previous 60 days. These records were retained, because all customers were required to have at least one valid purchase within the common 90-day eligibility window. This is crucial for fair comparisons and enables the model to represent inactive short-window customers.

## Exact validation evidence
- All **20** fixed prediction dates matched Milestone 2 counts, positive future buyer counts, and actual gross future spending **exactly**.
- Customer–date–history rows are unique; all 30/60/90 windows share the same customer population and the same target per date.
- Source dates used in every history window are **on/after cutoff-W and strictly before the cutoff**. Source dates in nonempty outcome windows are on/after cutoff and strictly before cutoff+30.
- Feature spending is nondecreasing with history length (30-day ≤ 60-day ≤ 90-day).
- Independent validator compared all 124,875 long rows and 41,625 wide rows and independently recomputed **144 sampled history windows from the cleaned transaction source**, covering training, validation, test and zero-history cases. All checks passed.
- Training, validation and test dates are unchanged from Milestone 2, with no label periods crossing into the next split.

**Note:** Passing these guards prevents the particular date-related leakages audited. Model-level preprocessing and hyperparameter tuning in Milestone 4 must still be fitted using training/validation only, never future final-test labels.

## Files and how to reproduce
`milestone3_build_features.py` uses only standard Python. Place it beside `online_retail_clean_sales_m2.csv.gz` and `milestone2_cutoff_schedule.csv`. Run `python milestone3_build_features.py`, then `python verify_milestone3.py`.

- `milestone3_features_wide.csv.gz`: 41,625 customer–date rows with 30/60/90 features side-by-side.
- `milestone3_features_long.csv.gz`: 124,875 rows, one per customer–date–history length. **Recommended for model training**; only the five feature columns are predictors.
- `milestone3_customer_profiles_easy_preview.xlsx`: readable Excel example and glossary; full datasets are compressed CSVs.
- `milestone3_feature_quality.json` and `milestone3_verification.json`: machine-checkable quality results.
- `verify_milestone3.py`: independently recomputes sampled features from raw cleaned transactions and checks the full dataset consistency.

**Milestone 4:** Train historical-spending baseline, Ridge regression and Random Forest using the **same five features** within each history window. Tune with validation; keep the final test untouched until Milestone 5.
