# Milestone 5 — Locked final-test evaluation
**Study:** How Much Purchase History Is Needed to Predict Future Customer Revenue? A Comparison of Simple and Machine Learning Models  
**Run date:** 23 November 2026  
**Status:** COMPLETE — final held-out evaluation was run once, with no model refitting or test-based tuning.

## 1. What we tested
- Original dataset: UCI Online Retail II, transactions from 2009–2011, with steps M1–M3 used to prepare 791,045 usable gross-positive sales records and customer snapshots.
- **Final held-out prediction dates:** 2011-08-24, 2011-09-23 and 2011-10-23. Each has 30 days of subsequently recorded purchases to predict.
- **Test population:** 6,306 customer/date snapshots (2,885 distinct customers), identical at 30-, 60- and 90-day history windows.
- **True future sales:** £2,120,820.70 across the 6,306 outcomes, 62.1% of which were £0.
- Prediction target: **positive invoiced sales**, not net revenue/profit; cancellations are excluded.
- Models were frozen after M4 selection based only on validation MAE: proportional historical-sales baseline, Ridge Regression and Random Forest. An additional always-£0 sanity baseline was evaluated.
- Only `historical_spend_gbp`, `orders`, `recency_days`, `average_order_value_gbp`, and `distinct_products` were passed to the ML models. Test answers were used only for evaluation after predictions.

## 2. Primary results: mean absolute error (MAE, GBP; lower is better)

| History | Always predict £0 | Historical-spending rule | Ridge Regression | Random Forest |
|---|---:|---:|---:|---:|
| 30 days | £336.32 | £418.09 | £348.15 | £347.35 |
| 60 days | £336.32 | £358.06 | £319.28 | £323.19 |
| 90 days | £336.32 | £332.99 | **£304.50** | £305.79 |

The **90-day Ridge model had the lowest observed final-test MAE**, £304.50, compared with £348.15 for the 30-day version. This does not mean the 90-day Ridge model was chosen *using* test performance; the M4 hyperparameter selection was already frozen.

Relative improvement in MAE: Ridge 90 days vs Ridge 30 days: **£43.66 (12.5%)**; Ridge 90 days vs always-zero: **£31.82 (9.5%)**.

A paired, customer-clustered bootstrap (2,000 resamples) yielded an exploratory 95% interval for the MAE change, 90 days minus 30 days, of **[-£59.72, -£31.54]** for Ridge. It does not fully reflect temporal dependence (only three test cutoffs).

## 3. Identifying the customers likely to generate the most revenue

| History | Historical-spending rule | Ridge Regression | Random Forest |
|---|---:|---:|---:|
| 30 days | 46.3% | 47.3% | 46.9% |
| 60 days | 53.0% | 53.3% | 53.6% |
| 90 days | 56.3% | 56.5% | **56.6%** |

Numbers are the percentage of future recorded sales earned by the **top 10% of test customer/date snapshots ranked by the model's prediction**, pooled over the three test dates. For a uniformly random 10% group, the *expected* revenue fraction is 10%. Constant-£0 predictions have no meaningful ranking ability and are excluded from this comparison.

**Important:** Once 90 days of history is available, the plain spending rule identifies nearly as much of the future revenue concentration as either ML model. The ranking benefit of sophisticated ML over that spending rule is very small in this test; offline ranking is not evidence of higher advertising return.

## 4. Robustness and limitations

- **Ridge vs Random Forest at 90 days:** £304.50 vs £305.79 MAE. Estimated Random Forest minus Ridge difference £1.29, exploratory customer-cluster-bootstrap 95% interval **[-£5.12, £7.92]**. This small difference is not convincing evidence that either is consistently superior.
- **Other metric changes the ranking:** 90-day *historical-spending baseline* had the lowest root mean squared error (RMSE) **£1,323.22**, versus **£1,339.49** for Ridge and **£1,367.61** for Random Forest. Large spenders meaningfully affect conclusions.
- **Three chronological test dates differ:** 90-day Ridge MAEs were **£302.21, £326.79 and £287.50**; 90-day Random Forest MAEs were **£306.29, £328.74 and £285.96** respectively.
- **Extreme values:** Excluding the top 1% of outcome values (post-hoc sensitivity only), 90-day MAEs become £263.09 (simple spending), £229.51 (Ridge), £230.84 (Random Forest). The 1% cutoff is ~£4,091.40. The post-hoc filtering must not be presented as the primary result.
- **Systematic underprediction:** Test average real spending per customer/date was **£336.32**. The Ridge 90-day model on average underpredicted by approximately **£91.40** per profile; Random Forest by approximately **£95.51**. Ranking may still be useful, but forecast totals are biased on the test dates.
- Repeated snapshots of the same customer are not independent; customer-clustered intervals only partially address dependence. Three test dates do not allow confident generalization to different years or markets. Most of the old dataset is UK retail with wholesale characteristics. Customer IDs missing for many raw rows; never equate the target with profit or ad-driven uplift.

## 5. Leakage protection and independent verification

1. M2 fixed train/validation/test chronological cutoffs before model training.
2. M3 computed past-only features and checked their source dates; outcomes cover only `[cutoff, cutoff + 30 days)`.
3. M4 used training outcomes for fitting and validation outcomes for selecting hyperparameters; it ignored all final-test labels.
4. M5 evaluated all *six saved models unchanged* and the two non-ML baselines, with no retraining or test-based selection. Model artifact SHA-256 hashes remained unchanged.
5. `milestone5_verify.py` independently recalculated every MAE, RMSE and R² (plus every prediction-date MAE) from the **18,918 saved test prediction rows**; all checks passed. Models must still be independently rerun and reviewed by the student before submission.

## 6. Evidence files
- `milestone5_evaluate.py`: frozen test evaluation; charts and raw outputs.
- `milestone5_verify.py`: independent results replay.
- `milestone5_results/test_scores_all_methods.csv`: exact overall results for 12 conditions (3 windows × 4 methods).
- `milestone5_results/test_scores_by_prediction_date.csv`: per-date scores for all methods.
- `milestone5_results/test_predictions_all_methods.csv.gz`: all 18,918 test predictions and recorded outcomes.
- `milestone5_results/test_outlier_sensitivity.csv`: sensitivity analysis.
- `milestone5_results/milestone5_results.json`: full metrics and exploratory paired customer bootstrap.
- `milestone5_results/figure1_test_mae.png`, `figure2_test_top10_capture.png`, `figure3_90d_per_date.png`: publication-ready charts.
- `milestone5_results/milestone5_independent_verification.json`: verification log.
- `milestone4_models/*.joblib`: unmodified models from M4; `milestone3_features_long.csv.gz`: original M3 inputs.

**Conclusion:** More purchase history improved forecast MAE and revenue ranking on these held-out periods. A 90-day linear Ridge model slightly outperformed 90-day Random Forest on MAE, while a simple historical-spending rule almost matched ML's top-decile revenue ranking and achieved the lowest 90-day RMSE. We cannot claim that ML always dominates, or that these results prove business impact on modern campaigns.
