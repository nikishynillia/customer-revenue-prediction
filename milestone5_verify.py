#!/usr/bin/env python3
"""Standalone Milestone-5 result replay from saved customer predictions.
No fitting or parameter selection. Verifies all output rows and key findings.
"""
import csv,gzip,json, math
from pathlib import Path
import numpy as np
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'milestone5_results'
meta=json.loads((OUT/'milestone5_results.json').read_text())
expected={(r['window_days'],r['method']):r for r in meta['results']}
column={'Always predict £0':'zero_pred_gbp','Historical spending baseline':'baseline_pred_gbp',
        'Ridge regression':'ridge_pred_gbp','Random Forest':'forest_pred_gbp'}
with gzip.open(OUT/'test_predictions_all_methods.csv.gz','rt',newline='') as f:
    rows=list(csv.DictReader(f))
assert len(rows)==18918, len(rows)
for w in [30,60,90]:
    sub=[r for r in rows if int(r['history_days'])==w]
    assert len(sub)==6306
    y=np.array([float(r['actual_future_30d_gbp']) for r in sub])
    assert math.isclose(y.sum(),2120820.70,abs_tol=.3)
    assert len(set((r['prediction_date'],r['customer_id']) for r in sub))==6306
    for method,col in column.items():
        p=np.array([float(r[col]) for r in sub]);s=expected[(w,method)]
        # Predictions stored rounded to 4 dp.
        assert abs(mean_absolute_error(y,p)-s['mae_gbp'])<.0002,(w,method,'mae')
        assert abs(np.sqrt(mean_squared_error(y,p))-s['rmse_gbp'])<.0002,(w,method,'rmse')
        assert abs(r2_score(y,p)-s['r2'])<.0000001,(w,method,'r2')
        for dt in ['2011-08-24','2011-09-23','2011-10-23']:
            index=np.array([r['prediction_date']==dt for r in sub])
            d=next(q for q in meta['per_date_results'] if q['prediction_date']==dt and q['window_days']==w and q['method']==method)
            assert abs(mean_absolute_error(y[index],p[index])-d['mae_gbp'])<.0002
    print('Verified',w,'day window: predictions, MAE, RMSE, R2, and three per-date results')
assert math.isclose(meta['zero_spending_profile_share'],float((y==0).mean()),abs_tol=1e-10)
assert meta['training_performed_in_m5'] is False and meta['selected_hyperparams_unchanged'] is True
report={'verified':True,'prediction_rows':len(rows),'test_profiles_per_history_window':6306,
        'verified_windows':[30,60,90],'verified_methods':list(column),
        'checked':['every test MAE','every test RMSE','every test R2','all per-date MAEs',
                   'cohort and outcome total','no fitting in M5 script'],
        'limitations':'Replays outputs, not a substitute for independent rerunning full source-to-model pipeline.'}
(OUT/'milestone5_independent_verification.json').write_text(json.dumps(report,indent=2))
print('PASS: 18,918 rows of held-out predictions independently rechecked.')
