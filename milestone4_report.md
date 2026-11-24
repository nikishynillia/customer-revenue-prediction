# Milestone 4 — Model training and validation (completed)
**Project:** How Much Purchase History Is Needed to Predict Future Customer Revenue? A Comparison of Simple and Machine Learning Models  
**Date:** 23 November 2026  
**Status:** Training and validation completed. **Final test has NOT been evaluated.**

## 1. Dataset and model inputs
Starting point: `milestone3_features_long.csv.gz` built from the cleaned original Online Retail II workbook.  
For EACH historical window (30, 60, 90 days): **29,339 training** customer-date profiles and **5,980 validation** profiles; all three window sizes refer to the same customers at the same cutoff dates.  

Each ML model receives exactly **five past-only variables**: historical positive sales (£), order count, recency (days), average order value (£), and number of distinct products.

**Excluded from predictor inputs:** `future_30d_spending_gbp` (correct answer), customer ID, prediction date, data split and history-length identifier. The correct answer is used as `y` in training and only for validation scoring.

**Target:** actual positive invoiced sales during the 30 days after each prediction date. These data exclude cancellations/returns. Accordingly, "customer revenue" in this project means **gross recorded positive invoiced sales**, not net revenue, margin, profit or an advertising effect.

## 2. Models and what was compared
1. **Historical-spending baseline:** forecast = 30 / previous-window-days × past invoiced spending. No ML training.
2. **Ridge regression:** standardize the five features using TRAINING data only; fit a regularized linear model to the training outcomes. Validate regularization alpha ∈ {1,100,10000}.
3. **Random Forest regression:** train 120-tree ensembles with fixed random seed 2026. Compare four combinations of tree depth and minimum leaf size (all specified in the training script).

Nine main selected combinations: three methods × three windows. All use exactly the same eligibility population and forecast horizon; the more complex methods use five features, whereas the historical-spending baseline intentionally uses only past spending, making Ridge a relevant simpler multivariable reference.

Nonnegative predictions: all model outputs are clipped at zero, because gross positive sales cannot be negative.

**Selection procedure:** minimize MAE on VALIDATION data, RMSE as a tie-break. Training records are earlier than validation, and no FINAL-TEST label was loaded or used in Milestone 4.

## 3. Actual validation results
Mean absolute error (MAE) in GBP — the average absolute difference between predicted and actual 30-day future spending, **lower is better**.

| History length | Spending baseline | Ridge regression | Random Forest |
|---|---:|---:|---:|
| 30 days | 304.05 | 270.18 | **264.05** |
| 60 days | 278.25 | 248.70 | **241.99** |
| 90 days | 261.49 | 232.24 | **231.85** |

**Validation only:** More historical data improved MAE in all three model families for these particular three validation periods. Random Forest's advantage over Ridge at 90 days is **£0.39 MAE per customer**, much too small to declare it consistently superior without additional evaluation.

**Alternative business metric:** The chosen top 10% of customers ranked by 90-day Random Forest predictions accounted for approximately **57.3%** of actual future positive sales on the three validation periods. This is a ranking evaluation only; it does not establish marketing uplift or prove the model would increase revenue when acted upon.

### Important sanity check: predicting zero
**68.7%** of these 5,980 validation customer-date profiles had **no recorded purchase** in the subsequent 30 days. Therefore, predicting zero for everyone has MAE **£237.21**. This trivial rule beats the 30- and 60-day versions of all three methods on MAE, and comes close to the 90-day results. MAE alone is not an adequate basis for the business-value claim; Milestone 5 must consider revenue-ranking, robustness and a zero baseline. Note that always predicting zero would be useless for identifying high-spending customers, which is the purpose of the ranking metric.

### Three validation cutoffs for the 90-day models
| Prediction date | Ridge MAE | Forest MAE |
|---|---:|---:|
| 2011-05-26 | 220.64 | 223.27 |
| 2011-06-25 | 213.91 | 212.94 |
| 2011-07-25 | 262.38 | 259.56 |

The winning model even changes between these three validation cutoffs. These dates do **not** represent three independent modern e-commerce campaigns.

## 4. Quality and leakage safeguards
- `milestone4_train_models.py` explicitly whitelists the **five** feature columns; no other fields enter a model.
- All candidates are fitted using `split='train'` only.
- StandardScaler for Ridge is fitted only on training values, inside a pipeline.
- Models and settings selected using only `split='validation'` outcomes.
- `split='test'` rows are skipped **before parsing their feature or target values**: 18,918 final-test window rows ignored.
- The script asserts identical customer/date keys and identical future spending across 30/60/90-day windows for each split.
- Independent verifier reloaded all **six saved ML models**, reproduced their validation MAEs and checked **17,940** per-window validation prediction records against the saved files; all checks passed.
- This verifies the code path and temporal split, but cannot establish absence of every possible source-data artifact or business-data bias.

## 5. Selected training settings
| Historical window | Ridge alpha | Random Forest maximum depth | RF minimum samples per leaf |
|---|---:|---:|---:|
| 30 days | 100 | 10 | 10 |
| 60 days | 100 | 10 | 10 |
| 90 days | 1 | 16 | 20 |

The models are frozen after validation; **do not change these settings in response to final-test results**. Milestone 5 evaluates these saved models on the held-out final dates 2011-08-24, 2011-09-23, and 2011-10-23.

## 6. Reproducibility
Required: Python 3, `numpy==2.3.5`, `scikit-learn==1.8.0`, `joblib==1.5.3`. May work on other versions but these are the verified environment versions.

To regenerate from the M3 feature file, run `python milestone4_train_models.py`, then `python milestone4_verify.py`. The training output includes six `.joblib` model files, the all-candidate validation table, chosen-model validation predictions, results JSON, and an independent check JSON. Training times depend on the local machine.

**Caveats:** Customer IDs recur across separate prediction dates; rows are not independent people. There are only three validation dates. This is an old UK retail dataset, with many customers who have zero future purchases, and high-value purchase outliers. Findings should be interpreted for this dataset and its gross positive sales target, not uncritically extrapolated to every contemporary e-commerce store.

## Next step
Milestone 5: Apply the **frozen** settings/models to the untouched chronological final-test period, compute MAE/RMSE, compare zero and historical-spending baselines, evaluate top-decile revenue capture, and report whether validation patterns persist. Results, charts, uncertainty and sensitivity checks belong in that milestone; none are asserted here.
