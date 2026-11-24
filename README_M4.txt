MILESTONE 4 — TRAINING + VALIDATION ONLY
23 November 2026

Research: How Much Purchase History Is Needed to Predict Future Customer Revenue?

Start here:
- research_roadmap_v4.md (status and next step)
- milestone4_report.md (methods and real validation findings)
- milestone4_models/validation_scores_all_candidates.csv (all model settings + validation metrics)
- milestone4_models/milestone4_validation_summary.json (detailed scores and selected settings)
- milestone4_models/milestone4_independent_verification.json (replay and leakage safeguards)

Model files:
- milestone4_models/window_{30,60,90}d_ridge.joblib
- milestone4_models/window_{30,60,90}d_forest.joblib

Reproduce:
1. Install versions listed in milestone4_requirements.txt.
2. Place milestone3_features_long.csv.gz in the same folder as the scripts.
3. Run `python milestone4_train_models.py`.
4. Run `python milestone4_verify.py`.

IMPORTANT: No final-test targets or outputs were used for model fitting, selection or evaluation.
The final test belongs to Milestone 5.

The package contains the necessary model-ready M3 feature data as an input. For full reproduction from the original workbook, combine with the earlier M2/M3 pipeline files.
