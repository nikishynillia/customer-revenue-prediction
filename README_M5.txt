MILESTONE 5 — FROZEN FINAL TEST RESULTS
23 November 2026

START
- research_roadmap_v5.md — milestone tracker.
- milestone5_report.md — results and research limitations.
- milestone5_results/ — CSV metrics, raw predictions, JSON evidence, PNG charts.

REPRODUCE MILESTONE 5 (python >= 3.10)
1) Extract this ZIP into one directory without changing its internal paths.
2) Install numpy==2.3.5, scikit-learn==1.8.0, joblib==1.5.3, and matplotlib.
   For example: python -m pip install -r milestone5_requirements.txt
3) Run python milestone5_evaluate.py
4) Run python milestone5_verify.py

Note: the frozen six joblib models and M3 features are included in this package.
Training methodology/source:
- milestone4_train_models.py fits 6 models using M3 train/validation only.
- milestone4_verify.py replays validation with saved M4 models.
- Neither script should be used for choosing new models after inspecting M5 outcomes.

FOR REPRODUCTION FROM THE ORIGINAL EXCEL DATA
Use the user's original UCI Online Retail II workbook and the earlier M2/M3
cleaning/feature-engineering scripts. The full raw-to-results package and
reviewer-accessible repository will be finalized in Milestone 7.
Official public source: https://archive.ics.uci.edu/dataset/502/online+retail+ii

IMPORTANT:
- Data predicts GROSS POSITIVE INVOICED SALES, not net revenue/profit.
- Same customers may appear at multiple cutoffs.
- The zero-prediction baseline has no valid top-decile ranking.
- Do not tune models on the final test; 3 final test cutoffs offer limited temporal inference.
- Students must independently reproduce results, check code and declare AI help.
