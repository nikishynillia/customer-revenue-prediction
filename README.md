# How Much Purchase History Is Needed to Predict Future Customer Revenue?

**A Comparison of Simple and Machine Learning Models**

Student Research Challenge · Architecture of Computers · Vistula University · 2026/27

## Research question

How does 30, 60 or 90 days of historical transaction information affect prediction of gross positive invoiced sales over the next 30 days? Compare a proportional historical-spending rule, Ridge regression and Random Forest, with an always-zero reference baseline. This is a controlled replication-style study, not an advertising intervention.

## Main held-out results

MAE in GBP, lower is better (6,306 customer-date observations per history window):

| History | Always £0 | Spending rule | Ridge | Random Forest |
|---|---:|---:|---:|---:|
| 30 days | 336.32 | 418.09 | 348.15 | 347.35 |
| 60 days | 336.32 | 358.06 | 319.28 | 323.19 |
| 90 days | 336.32 | 332.99 | 304.50 | 305.79 |

The 90-day spending rule has the lowest RMSE (£1,323.22). Ridge has the lowest observed MAE; the small 90-day difference between Ridge and Forest is uncertain. No causal marketing effect is measured.

## Contents

All core scripts intentionally remain at repository root, because they use paths relative to their own location.

| File / directory | Purpose |
|---|---|
| `online_retail_II.xlsx` | Unmodified original UCI workbook |
| `audit_retail.py`, `online_retail_audit.json` | Initial exploratory audit; script made portable for this package |
| `milestone2_clean_design.py`, `verify_milestone2.py` | Clean raw workbook, create fixed cutoff schedule, verify data |
| `online_retail_clean_sales_m2.csv.gz` | Cleaned transaction-level dataset |
| `milestone2_*` JSON/CSV files | Protocol, schedule, quality checks and verification |
| `milestone3_build_features.py`, `verify_milestone3.py` | Build and verify model-ready features |
| `milestone3_features_long.csv.gz`, `milestone3_features_wide.csv.gz` | 124,875 window rows / 41,625 customer-date snapshots |
| `milestone4_train_models.py`, `milestone4_verify.py` | Fit and select six ML models using training/validation only |
| `milestone4_models/` | Frozen models, validation predictions, candidate scores and verification |
| `milestone5_evaluate.py`, `milestone5_verify.py` | Evaluate fixed models and independently recheck calculations |
| `milestone5_results/` | Raw held-out predictions, numerical metrics, uncertainty, plots |
| `reference_results/` | Archived reference score table for comparison after rerunning |
| `compare_reference_results.py` | Compare regenerated scores with the archived original |
| `REPRODUCTION_RECORD.md` | Independent student rerun record |
| `DATA_LICENSE.md` | Source data attribution, license and checksum |

The optional `milestone3_make_preview_xlsx.py` created the illustrative Excel preview with artifact_tool; it is retained for transparency and is not part of the computational research pipeline or required to reproduce results. The Excel preview itself is supplied. The core experiment uses standard Python packages.

## Reproduce from the original workbook

Download/clone this repository and open a terminal at its root. Python 3.12 (64-bit) is recommended. The dependency versions in `requirements.txt` reflect the packages visible in the student's successful Windows rerun; the original milestone requirement files are also retained. Run all stages in order. On Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe milestone2_clean_design.py
.\.venv\Scripts\python.exe verify_milestone2.py
.\.venv\Scripts\python.exe milestone3_build_features.py
.\.venv\Scripts\python.exe verify_milestone3.py
.\.venv\Scripts\python.exe milestone4_train_models.py
.\.venv\Scripts\python.exe milestone4_verify.py
.\.venv\Scripts\python.exe milestone5_evaluate.py
.\.venv\Scripts\python.exe milestone5_verify.py
.\.venv\Scripts\python.exe compare_reference_results.py
```

For macOS/Linux, create a virtual environment using `python3.12 -m venv .venv` and substitute `.venv/bin/python` for the Windows executable above. The original workbook is already included. If needed, download it from the [official UCI record](https://archive.ics.uci.edu/dataset/502/online+retail+ii) and place it at the repository root with the same filename. Do not manually decompress `.csv.gz` files.

Expected checkpoints: 791,045 cleaned rows; 5,878 customers; 41,625 customer-date profiles; 124,875 window rows; 29,339 train / 5,980 validation / 6,306 test profiles per window; 18,918 held-out prediction rows verified. The original results are supplied; rerunning may overwrite generated outputs, so use a fresh working copy. Binary model files should only be loaded from a trusted copy of the repository; retraining from source is preferable. Tiny numerical differences can arise across library versions or platforms. The reference comparison uses a numerical tolerance.

## Experimental safeguards

Prediction dates: 20 cutoffs 30 days apart. Features use [t-W,t); target uses [t,t+30 days). Eligibility is defined by a valid purchase during the preceding 90 days for all models. Training cutoffs are Apr 2010–Apr 2011, validation May–Jul 2011, test Aug–Oct 2011. The fixed training and validation code ignores final-test targets, and only historical features enter prediction arrays. Validation selects settings; the final test is not used for tuning. Reproduction repeats that fixed evaluation, not model selection based on test outcomes.

## Limitations

One older UK retailer with wholesale customers; missing customer IDs; unresolved within-sheet repeated line signatures; gross positive invoice sales exclude returns and are not profit. Only three final test cutoffs and repeated customers limit generalization. The simple rule uses one feature while ML methods use five. Top-decile revenue capture pools the test snapshots and does not prove incremental advertising revenue.

## Data source and AI transparency

Chen, D. (2012). Online Retail II [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5CG6D. CC BY 4.0; see `DATA_LICENSE.md`. ChatGPT assisted with design, Python scripts, initial analysis, figures and the manuscript. The student reran the full core pipeline independently, as described in `REPRODUCTION_RECORD.md`.
