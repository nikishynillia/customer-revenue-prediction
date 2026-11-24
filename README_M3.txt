MILESTONE 3 — 23 November 2026
Research: How Much Purchase History Is Needed to Predict Future Customer Revenue?

Files:
- milestone3_features_long.csv.gz: 124,875 customer-date-history length rows; use for ML.
- milestone3_features_wide.csv.gz: 41,625 customer-date rows, history variants side by side.
- milestone3_customer_profiles_easy_preview.xlsx: user-friendly Excel sample and explanations.
- milestone3_feature_quality.json: numeric data-quality audit.
- milestone3_verification.json: independent verification results.
- milestone3_report.md: methodology and definitions.
- research_roadmap_v3.md: latest milestone statuses.
- milestone3_build_features.py, verify_milestone3.py: scripts to regenerate features and validate.
- online_retail_clean_sales_m2.csv.gz, milestone2_cutoff_schedule.csv: required frozen Milestone 2 inputs, included.

Reproduce:
  python milestone3_build_features.py
  python verify_milestone3.py

Dependencies for building/validating features: Python 3 standard library only.
The Excel preview was generated using artifact_tool; its generator is included
but is not necessary to reproduce the model-ready CSV files.
No models were trained; no test-prediction results were calculated.
Do NOT use customer_id, prediction_date, split or future_30d_spending_gbp as model predictors.
