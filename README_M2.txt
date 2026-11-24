Milestone 2: data cleaning and experimental design

Source workbook is NOT bundled; place your original online_retail_II.xlsx next to the scripts.

Run:
  python -m pip install -r requirements_m2.txt
  python milestone2_clean_design.py
  python verify_milestone2.py

Validation verifies all 791,045 cleaned positive-sales rows.
The CSV is gzip-compressed; Python gzip.open handles it.
Open the preview CSV in Excel if you want a quick inspection.
Feature engineering / machine learning not included; those are milestones 3 and 4.
