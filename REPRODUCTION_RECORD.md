# Independent student reproduction

On 24 November 2026, the student reran the computational pipeline on a Windows computer with Python 3.12. Terminal screenshots shared in the project discussion show:

- 791,045 positive sales rows and 5,878 customers after cleaning; temporal checks passed.
- 41,625 customer-date profiles, 124,875 window-specific rows and 144 independently checked source-feature windows.
- Six models retrained, all validation MAEs reproduced, and 17,940 validation prediction rows verified.
- Final held-out evaluation completed without retraining; 18,918 test prediction rows independently rechecked by the verification script.
- The displayed final-test MAE, RMSE and top-decile capture values match the archived study values at the displayed precision.

The record describes the observed independent rerun; it does not imply an independent manual audit of every source or every line of code. Screenshots remain with the student and are not included in the public repository to avoid exposing unrelated personal screen content. The source-to-results scripts and numerical evidence are included here.
