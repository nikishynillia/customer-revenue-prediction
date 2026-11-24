#!/usr/bin/env python3
"""Milestone 4: train spending baseline, Ridge, and Random Forest.

Only uses TRAIN for fitting and VALIDATION for selecting models.
The TEST records, including their target, are ignored entirely in this script.
No new features are created from outcome values. Python >=3.10, numpy, sklearn, joblib.
"""
from __future__ import annotations
import csv, gzip, json, time
from pathlib import Path
import numpy as np
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'milestone3_features_long.csv.gz'
OUT = ROOT / 'milestone4_models'
OUT.mkdir(exist_ok=True)
FEATURES = ('historical_spend_gbp', 'orders', 'recency_days', 'average_order_value_gbp', 'distinct_products')
TARGET = 'future_30d_spending_gbp'
NON_FEATURES = ('split', 'prediction_date', 'customer_id', 'history_days', TARGET)
WINDOWS = (30,60,90)
SEED = 2026


def load_data():
    records = {w: {s: {'X': [], 'y': [], 'keys': []} for s in ('train','validation')} for w in WINDOWS}
    with gzip.open(DATA, 'rt', newline='', encoding='utf-8') as stream:
        reader = csv.DictReader(stream)
        assert tuple(reader.fieldnames) == NON_FEATURES[:4] + FEATURES + (TARGET,), reader.fieldnames
        assert not set(FEATURES).intersection(NON_FEATURES)
        ignored_test = 0
        for row in reader:
            split = row['split']
            if split == 'test':
                ignored_test += 1
                continue  # no TEST y read, parsed, evaluated or used for tuning
            assert split in ('train', 'validation')
            w = int(row['history_days'])
            r = records[w][split]
            r['X'].append([float(row[k]) for k in FEATURES])
            r['y'].append(float(row[TARGET]))
            r['keys'].append((row['prediction_date'], row['customer_id']))
    assert ignored_test == 6306*3
    for w in WINDOWS:
        for split, expected in (('train',29339),('validation',5980)):
            r = records[w][split]
            assert len(r['X']) == expected
            r['X'] = np.asarray(r['X'], dtype=np.float64)
            r['y'] = np.asarray(r['y'], dtype=np.float64)
            assert np.isfinite(r['X']).all() and np.isfinite(r['y']).all()
            assert (r['X']>=0).all() and (r['y']>=0).all()
    for split in ('train','validation'):
        keys = records[30][split]['keys']; target = records[30][split]['y']
        for w in (60,90):
            assert records[w][split]['keys'] == keys, 'misaligned customer cohorts'
            assert np.array_equal(records[w][split]['y'],target), 'future target differs by window'
        dates = [k[0] for k in keys]
        assert (max(dates) <= '2011-04-26') if split == 'train' else (min(dates)>='2011-05-26' and max(dates)<='2011-07-25')
    return records, ignored_test


def top_decile_metrics(y, pred):
    # Stable tie-break based on row order, not outcome or feature values
    k = max(1, int(np.ceil(len(y)*0.10)))
    pred_rank = np.argsort(-pred, kind='stable')[:k]
    actual_rank = np.argsort(-y, kind='stable')[:k]
    fraction_future_sales = float(y[pred_rank].sum() / y.sum()) if y.sum()>0 else 0.0
    overlap = float(len(set(pred_rank)&set(actual_rank))/k)
    return {'top_decile_revenue_capture':fraction_future_sales,
            'top_decile_actual_high_spenders_overlap':overlap,
            'top_decile_customer_count':k}


def scores(y, preds):
    p = np.clip(np.asarray(preds,dtype=float),0,None)
    assert np.isfinite(p).all()
    return {'mae_gbp':float(mean_absolute_error(y,p)),
            'rmse_gbp':float(np.sqrt(mean_squared_error(y,p))),
            'r2':float(r2_score(y,p)),
            'prediction_mean_gbp':float(p.mean()),
            'actual_mean_gbp':float(y.mean()),
            **top_decile_metrics(y,p)}


def make_ridge(alpha):
    return Pipeline([('scaler',StandardScaler()),('ridge',Ridge(alpha=alpha))])


def make_forest(depth, leaf):
    return RandomForestRegressor(n_estimators=120, max_depth=depth,
                                 min_samples_leaf=leaf, max_features=1.0,
                                 random_state=SEED, n_jobs=-1)


def train():
    t0 = time.monotonic()
    rec, skipped_test = load_data()
    print(f'Loaded 29,339 training and 5,980 validation customer profiles x3; ignored {skipped_test} final test rows.',flush=True)
    all_scores=[]; selected=[]; validation_predictions=[]
    for w in WINDOWS:
        tr, va = rec[w]['train'], rec[w]['validation']
        x, y = tr['X'], tr['y']
        xv, yv = va['X'], va['y']
        assert x.shape==(29339,5) and xv.shape==(5980,5)
        # Baseline: assume future daily spending rate = previous spending / window length.
        base_pred = np.clip(xv[:,0]*30.0/w,0,None)
        b_score = scores(yv,base_pred)
        all_scores.append({'window_days':w,'method':'Historical spending baseline', 'config':'30 / window * past spending','selected':True, **b_score})
        val_predictions_base = np.asarray(base_pred)
        print(f'{w}-day baseline: MAE {b_score["mae_gbp"]:.2f}',flush=True)

        for method, configs in (
            ('Ridge regression', [ {'alpha':a} for a in (1.0,100.0,10000.0) ]),
            ('Random Forest', [
                {'max_depth':10,'min_samples_leaf':10},
                {'max_depth':16,'min_samples_leaf':20},
                {'max_depth':12,'min_samples_leaf':50},
                {'max_depth':16,'min_samples_leaf':50},
            ])
        ):
            candidates=[]
            for conf in configs:
                model = make_ridge(**conf) if method.startswith('Ridge') else make_forest(conf['max_depth'],conf['min_samples_leaf'])
                fitstart=time.monotonic()
                model.fit(x,y)
                p=np.clip(model.predict(xv),0,None)
                metric=scores(yv,p)
                candidates.append((conf,model,p,metric))
                all_scores.append({'window_days':w,'method':method, 'config':json.dumps(conf,sort_keys=True), 'selected':False,**metric})
                print(f'{w}-day {method} {conf}: MAE {metric["mae_gbp"]:.2f}; RMSE {metric["rmse_gbp"]:.2f}; top10 capture {metric["top_decile_revenue_capture"]:.3f}; {time.monotonic()-fitstart:.1f}s',flush=True)
            # Primary selection metric MAE; the test set is never used
            conf, model, p, metric = min(candidates,key=lambda item: (item[3]['mae_gbp'],item[3]['rmse_gbp']))
            chosen_config=json.dumps(conf,sort_keys=True)
            for s in all_scores:
                if s['window_days']==w and s['method']==method and s['config']==chosen_config:s['selected']=True
            artifact=f'window_{w}d_'+('ridge' if method.startswith('Ridge') else 'forest')+'.joblib'
            joblib.dump(model,OUT/artifact,compress=3)
            selected.append({'window_days':w,'method':method,'config':conf,'model_file':artifact,'selection_rule':'minimum validation MAE; RMSE tie-break',**metric})
            if method.startswith('Ridge'):
                ridge_pred=p
            else:
                forest_pred=p
        for i,key in enumerate(va['keys']):
            validation_predictions.append({'prediction_date':key[0], 'customer_id':key[1], 'window_days':w,
                                           'actual_future_30d_gbp':f'{yv[i]:.4f}',
                                           'baseline_pred_gbp':f'{val_predictions_base[i]:.4f}',
                                           'ridge_pred_gbp':f'{ridge_pred[i]:.4f}',
                                           'forest_pred_gbp':f'{forest_pred[i]:.4f}'})
        print(f'Completed {w}-day history window.',flush=True)

    with (OUT/'validation_scores_all_candidates.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=all_scores[0].keys());writer.writeheader();writer.writerows(all_scores)
    with gzip.open(OUT/'validation_predictions_selected.csv.gz','wt',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=validation_predictions[0].keys());writer.writeheader();writer.writerows(validation_predictions)
    summary={
        'status':'milestone4_complete','created_at_utc':'2026-11-23',
        'dataset':'milestone3_features_long.csv.gz',
        'feature_names':FEATURES,'target_name':TARGET,
        'excluded_from_features':NON_FEATURES,
        'split_counts_per_window':{'train':29339,'validation':5980,'test':6306},
        'final_test_rows_ignored_without_target_parse':skipped_test,
        'training_date_range':['2010-04-01','2011-04-26'],
        'validation_date_range':['2011-05-26','2011-07-25'],
        'test_date_range_for_next_milestone':['2011-08-24','2011-10-23'],
        'model_selection_primary_metric':'validation MAE',
        'baseline_definition':'30 / history_window_days * historical_spend_gbp',
        'model_selection':selected,
        'validation_scores_all_candidates':all_scores,
        'final_test_evaluated':False,
        'all_models_fitted_on_train_only':True,
        'total_seconds':time.monotonic()-t0}
    (OUT/'milestone4_validation_summary.json').write_text(json.dumps(summary,indent=2))
    print(f'Finished milestone 4 in {summary["total_seconds"]:.1f}s. NO FINAL TEST evaluated.',flush=True)
    return summary

if __name__=='__main__':train()
