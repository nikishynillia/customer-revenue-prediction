# Online Retail II — Initial dataset audit

**Audit date:** 23 November 2026  
**Source:** Original user-uploaded `/mnt/data/online_retail_II.xlsx` (not a third-party CSV copy)

| Finding | Measured value |
|---|---:|
| Source transaction rows across both sheets | 1,067,371 |
| Cross-sheet repeated transaction rows (1–9 Dec 2010) | 22,523 |
| Rows after removing cross-sheet overlap | 1,044,848 |
| Rows with no customer ID | 235,287 |
| Preliminary usable positive-sales line items | 791,045 |
| Customers associated with positive-sales rows | 5,878 |
| Unique customer/invoice combinations | 36,969 |
| Customers with two or more observed orders | 4,255 |
| Customers with five or more observed orders | 2,161 |
| Earliest date | 2009-12-01 |
| Latest date | 2011-12-09 |

## Preliminary filtering logic

Count a transaction line for positive-sales prediction only if customer ID is present, quantity > 0, price > 0, and invoice does not start with `C`. This is a **preliminary** choice; final outcome definition, treatment of returns and possible within-sheet exact duplicate rows will be documented in the next milestone.

## Research feasibility

There are thousands of recurring customers, approximately 20 candidate non-overlapping 30-day outcome cutoffs, and an observed future-spending outcome available from actual transactions. This supports a 30/60/90-day history comparison. Accuracy or improvement has **not** been measured yet. No claim about modern customers or the causal value of marketing can be inferred from these historical transaction records.

## Source and audit reproducibility

- Original data: `online_retail_II.xlsx`, two Excel worksheets.
- Read-only audit code: `audit_retail.py`.
- Machine-readable summary: `online_retail_audit.json`.
- Interim daily positive-sales aggregation: `online_retail_clean_customer_daily.csv.gz`. This extract does not retain all potential modelling features, such as distinct products or invoices, and is **not** a final cleaned modelling dataset.
