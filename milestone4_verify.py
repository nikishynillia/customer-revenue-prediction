#!/usr/bin/env python3
"""Independently verify saved Milestone 4 models using VALIDATION only.
TEST rows are skipped before their target is parsed; never inspected here.
"""
import csv,gzip,json
from collections import defaultdict
from pathlib import Path
import numpy as np
import joblib
from sklearn.metrics import mean_absolute_error
ROOT=Path(__file__).resolve().parent
M=ROOT/'milestone4_models'
summary=json.loads((M/'milestone4_validation_summary.json').read_text())
features=summary['feature_names']
assert features==['historical_spend_gbp','orders','recency_days','average_order_value_gbp','distinct_products']
assert summary['final_test_evaluated'] is False
records=defaultdict(list)
ignored_test=0
with gzip.open(ROOT/'milestone3_features_long.csv.gz','rt',newline='') as f:
 for row in csv.DictReader(f):
  if row['split']=='test':ignored_test+=1;continue
  if row['split']=='validation':
   records[int(row['history_days'])].append((row['prediction_date'],row['customer_id'],np.array([float(row[k]) for k in features]),float(row['future_30d_spending_gbp'])))
assert ignored_test==18918
selected={(z['window_days'],z['method']):z for z in summary['model_selection']}
report={'test_rows_ignored_without_parsing_outcomes':ignored_test,'validation_replays':[], 'all_checks_passed':True}
for window in (30,60,90):
 rec=records[window]
 assert len(rec)==5980
 X=np.stack([r[2] for r in rec]);y=np.array([r[3] for r in rec]);
 saved_preds={}
 for method,term in [('Ridge regression','ridge'),('Random Forest','forest')]:
  z=selected[(window,method)]
  model=joblib.load(M/z['model_file'])
  assert model.n_features_in_==5
  p=np.maximum(0,model.predict(X))
  mae=float(mean_absolute_error(y,p))
  assert np.isclose(mae,z['mae_gbp'],rtol=0,atol=1e-8),(window,method,mae,z['mae_gbp'])
  saved_preds[term]=p
  report['validation_replays'].append({'window_days':window,'method':method,'mae_gbp':mae,'matches_saved_score':True})
 base=30.0/window*X[:,0]
 zero_mae=float(y.mean())
 report.setdefault('zero_sales_sanity_baseline',{})[str(window)]={'mae_gbp':zero_mae,'no_future_purchase_pct':float((y==0).mean()*100)}
 with gzip.open(M/'validation_predictions_selected.csv.gz','rt',newline='') as f:
  predrows=[r for r in csv.DictReader(f) if int(r['window_days'])==window]
 assert len(predrows)==len(rec)
 for i,(record, pr) in enumerate(zip(rec,predrows)):
  assert (pr['prediction_date'],pr['customer_id'])==record[:2]
  assert np.isclose(float(pr['actual_future_30d_gbp']),y[i],atol=1e-4)
  assert np.isclose(float(pr['baseline_pred_gbp']),base[i],atol=5.1e-5)
  for method in ('ridge','forest'):
   assert np.isclose(float(pr[method+'_pred_gbp']),saved_preds[method][i],atol=5.1e-5)
 dates=sorted(set(k[0] for k in rec))
 perdate=[]
 for dt in dates:
  ids=np.array([k[0]==dt for k in rec],dtype=bool)
  perdate.append({'prediction_date':dt,
                  'customer_profiles':int(ids.sum()),
                  'baseline_mae_gbp':float(mean_absolute_error(y[ids],base[ids])),
                  'ridge_mae_gbp':float(mean_absolute_error(y[ids],saved_preds['ridge'][ids])),
                  'forest_mae_gbp':float(mean_absolute_error(y[ids],saved_preds['forest'][ids]))})
 report.setdefault('per_date_validation_mae',{})[str(window)]=perdate
(M/'milestone4_independent_verification.json').write_text(json.dumps(report,indent=2))
print('PASS: independently reloaded all 6 saved models, reproduced all MAEs and checked 17,940 validation predictions.')
print('No TEST targets loaded or evaluated; 18,918 TEST records skipped.')
print('Zero-spend baseline MAE £%.2f; %.1f%% of validation customer-profiles have no next-30-day purchases.'%(
 report['zero_sales_sanity_baseline']['90']['mae_gbp'],report['zero_sales_sanity_baseline']['90']['no_future_purchase_pct']))
for p in report['per_date_validation_mae']['90']:
 print('90-day model,',p['prediction_date'],'Ridge MAE',round(p['ridge_mae_gbp'],2),'RF MAE',round(p['forest_mae_gbp'],2))
