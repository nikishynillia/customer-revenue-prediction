# Milestone 2 — Cleaning and experimental design (finalized)
**Research topic:** How Much Purchase History Is Needed to Predict Future Customer Revenue? A Comparison of Simple and Machine Learning Models  
**Run date:** 23 November 2026  
**Input:** original uploaded `online_retail_II.xlsx` (read-only; two sheets)  
**Status:** Milestone 2 complete. **NO model training, feature rows or prediction results yet.**

## A. Audited data and exact reconciliation
| Measure | Count |
|---|---:|
| Raw worksheet records | 1,067,371 |
| Verified exact cross-sheet overlap | 22,523 |
| Records after overlap reconciliation | 1,044,848 |
| Excluded: No customer identifier | 235,287 |
| Excluded: Cancellation invoice with identifiable customer | 18,446 |
| Excluded: Nonpositive price after earlier exclusions | 70 |
| **Retained positive sale line items** | **791,045** |
| Retained customers | 5,878 |
| Customer-invoice pairs | 36,969 |
| Within-sheet identical line signatures retained, not deleted | 11,620 |

Check: 1,067,371 − 22,523 − 253,803 exclusions = 791,045 retained sales rows. The overlap rows matched **exactly as a multiset**, including repeated invoice-line records, not merely by matching dates or invoice IDs.

**Cleaning decisions:**
1. Keep worksheet 1 as source of truth for 1–9 December 2010; skip exactly matching entries from worksheet 2. No unmatched overlap remained.
2. Exclude records with missing customer IDs. Customer-level prediction is impossible without a customer key.
3. Exclude cancellation invoices (invoice begins with `C`), nonpositive quantities and nonpositive prices. Exclusion counts are hierarchical: a row is counted only for its first failure, so other problems may exist within an already excluded record.
4. Normalize source Excel floating-point price artifacts to **four decimal places**. This changes no retained unit price by more than 0.0000001 GBP. Compute line revenue as quantity × normalized unit price, preserving four decimal places.
5. Keep identical full line signatures within each worksheet (**11,620 additional instances**): no unique invoice-line ID exists, so removing these could incorrectly remove legitimate multiple line items. Flag as an ambiguity and consider a deduplication sensitivity test.
6. Retain transaction provenance (worksheet name and original row number), and preserve the original workbook unchanged.

**Target definition:** The study predicts **gross recorded positive invoiced sales in GBP** over the next 30 calendar days, excluding cancellation/return amounts. This is not net realized revenue, marketing ROI or profit. A label of zero means **no recorded positive purchases in the observed 30-day future period**.

**Important data concern:** A few individual purchase lines are extremely large (the largest is GBP 168469.6000). Some retained stock codes represent manual adjustments or postage; these are not systematically classified as goods. We will report these limitations and inspect their effect on errors in Milestone 5 rather than quietly modifying the target now.

## B. Fixed experimental design
- **Customer eligibility:** at least one observed valid positive sale during the 90 days before a snapshot date. Keep this customer set identical for 30/60/90-day feature comparisons. A customer can have no purchases in a shorter window; this is informative and should be recorded as zero history, not excluded.
- **Snapshot schedule:** one date every 30 calendar days from **2010-04-01 to 2011-10-23** (20 cutoffs). 
- **Feature windows:** 30, 60 and 90 calendar days immediately BEFORE the cutoff, ending strictly before it. The same target is used for all three windows.
- **Prediction horizon:** subsequent 30 days: `[cutoff, cutoff + 30 days)`.
- **Customer-level features planned for Milestone 3:** spending, invoice count, recency (if no earlier transaction in window use a designated missing-value rule), average invoice value and number of unique stock codes. Use only observed purchases in the permitted window.

### Chronological partitions
| Split | First snapshot | Last snapshot | Snapshot dates | Customer–date observations | Positive future spend observations |
|---|---|---|---:|---:|---:|
| Train | 2010-04-01 | 2011-04-26 | 14 | 29,339 | 8,951 |
| Validation | 2011-05-26 | 2011-07-25 | 3 | 5,980 | 1,871 |
| Test | 2011-08-24 | 2011-10-23 | 3 | 6,306 | 2,387 |

The customer–date totals are **not independent customers**: a customer may appear at multiple snapshot dates. This dependence must be accounted for when interpreting uncertainty.

**Leakage checks already passed:**
- Earliest 90-day history window lies inside the original 2009–2011 data period.
- Every future 30-day label period is fully observed in the dataset.
- Last training target ends **2011-05-26**, no later than the first validation cutoff **2011-05-26**.
- Last validation target ends **2011-08-24**, no later than the first test cutoff **2011-08-24**.
- Cleaned file independently re-read and all retained rows checked for valid customer, positive quantity, positive price, exact price × quantity revenue and no retained second-sheet overlap.

**Still to test in Milestone 3:** feature-level assertions on *actual generated features* (all their source dates must precede cutoff), one customer × cutoff × history-window feature row, no future target included in features, and consistent customer populations. These cannot be claimed completed until features exist.

## C. Comparisons and measurements
- **Baseline:** multiply gross spend in history window by `30 / history_window_days` (e.g., 30/60/90-day normalization). Also consider a predict-zero baseline as an interpretability check.
- **Simple ML:** Ridge regression, clamp negative predicted revenues to zero.
- **Complex ML:** Random Forest regression, with identical observed history horizon and eligible customers.
- **Selection:** configure features and tune settings using training/validation data only. Do not inspect final test prediction errors until settings are frozen.
- **Primary:** mean absolute error (GBP) on 30-day future gross purchases.
- **Secondary:** RMSE/median absolute error for extremes, revenue captured by top-decile predicted customers versus actual, bias and heterogeneity by history length.
- **Reporting:** compare all 9 scenarios, report numerical results, distributions, and limitations without claiming more customers would buy if targeted (no randomized marketing intervention in this dataset).

## D. Files and execution
- `milestone2_clean_design.py`: reproducible cleaning and split calculation from original workbook.
- `verify_milestone2.py`: independent checks on the exported CSV and protocol.
- `online_retail_clean_sales_m2.csv.gz`: full cleaned transaction-level export (11 columns including provenance).
- `online_retail_clean_sales_preview_1000.csv`: readable sample for Excel.
- `milestone2_cutoff_schedule.csv`: exact 20-date plan and eligibility counts.
- `milestone2_quality_checks.json`: quality measurements and cutoff statistics.
- `milestone2_experiment_protocol.json`: fixed machine-readable modeling specification.
- `milestone2_verification.json`: verification results.

Reproduce this milestone from the same folder by running `python milestone2_clean_design.py` followed by `python verify_milestone2.py`. The `lxml` Python package is used for XML parsing inside the xlsx container; the rest uses the Python standard library. The original xlsx is not overwritten.
