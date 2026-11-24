"""Compare regenerated held-out results with the original archived numerical reference."""
import csv, math
from pathlib import Path
ROOT = Path(__file__).resolve().parent
def load(path):
    with path.open(encoding="utf-8", newline="") as f:
        return {(r["window_days"],r["method"]):r for r in csv.DictReader(f)}
a=load(ROOT/"reference_results/test_scores_all_methods.csv")
b=load(ROOT/"milestone5_results/test_scores_all_methods.csv")
assert a.keys()==b.keys() and len(a)==12
for key in a:
    assert a[key]["n"]==b[key]["n"]=="6306"
    for metric in ("mae_gbp","rmse_gbp","r2","mean_prediction_gbp","mean_actual_gbp","top_decile_revenue_capture"):
        x,y=float(a[key][metric]),float(b[key][metric])
        assert math.isclose(x,y,rel_tol=1e-7,abs_tol=1e-6), (key,metric,x,y)
print("PASS: all 12 held-out method/window rows match archived reference metrics within tolerance.")
