#!/usr/bin/env python3
"""Milestone 5: a SINGLE frozen-model, held-out test evaluation.

Uses ONLY selected models from M4; no fitting, no hyperparameter selection,
no results-driven model modification. Read test y solely for scoring.
Python 3.10+, numpy, scikit-learn, joblib, matplotlib.
"""
import csv, gzip, json, hashlib
from pathlib import Path
from collections import defaultdict
import numpy as np
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
FEATURE_CSV = ROOT / 'milestone3_features_long.csv.gz'
MODEL_DIR = ROOT / 'milestone4_models'
OUT = ROOT / 'milestone5_results'
OUT.mkdir(exist_ok=True)
WINDOWS=(30,60,90)
FEATURES=('historical_spend_gbp','orders','recency_days','average_order_value_gbp','distinct_products')
TARGET='future_30d_spending_gbp'
METHODS=('Always predict £0','Historical spending baseline','Ridge regression','Random Forest')


def money(v): return round(float(v),4)
def digest(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def top10(y,p,keys,per_period=False):
    # Tie-break is a SHA-256-derived stable key, not a target or future covariate.
    # Top-decile results are based on PRE-registered individual 30-day periods.
    n=len(y); k=max(1,int(np.ceil(.10*n)))
    tiebreak=np.array([int(hashlib.sha256(f'{date}|{customer}'.encode()).hexdigest()[:16],16)
                       for date,customer in keys],dtype=np.uint64)
    idx=np.lexsort((tiebreak,-p))[:k]
    oracle=np.lexsort((tiebreak,-y))[:k]
    return {'k':k,'capture':float(y[idx].sum()/y.sum()) if y.sum()>0 else None,
            'overlap_with_true_top10':float(np.intersect1d(idx,oracle).size/k)}

def calc(y,p,keys):
    ab=np.abs(y-p)
    return {'n':len(y),'mae_gbp':float(ab.mean()),'rmse_gbp':float(np.sqrt(np.square(y-p).mean())),
            'median_absolute_error_gbp':float(np.median(ab)),
            'r2':float(r2_score(y,p)),
            'mean_prediction_gbp':float(p.mean()),'mean_actual_gbp':float(y.mean()),
            'prediction_bias_gbp':float((p-y).mean()),
            'zero_future_purchase_fraction':float((y==0).mean()),
            'top_decile':top10(y,p,keys)}

def bootstrapped_diff(values_a,values_b,groups,B=2000,seed=2026):
    """Paired clustered customer bootstrap for mean |error| differences (A-B).
    Note: repeated customers are clustered; temporal dependence from just 3 cutoff
    periods is NOT fully captured. Should not be treated as generalization to 2026.
    """
    g_ids,inv=np.unique(np.array(groups),return_inverse=True)
    n=len(g_ids)
    dif=np.asarray(values_a)-np.asarray(values_b)
    counts=np.bincount(inv,minlength=n)
    sums=np.bincount(inv,weights=dif,minlength=n)
    rng=np.random.default_rng(seed)
    output=np.empty(B)
    for b in range(B):
        sel=rng.integers(0,n,size=n)
        output[b]=sums[sel].sum()/counts[sel].sum()
    return {'estimated_mae_difference_gbp':float(dif.mean()),
            'bootstrap_95pct_interval_gbp':[float(z) for z in np.percentile(output,[2.5,97.5])],
            'bootstrap_resamples':B,'cluster':'customer_id',
            'interval_limitations':'Does not fully account for having only three final chronological cutoffs.'}


def main():
    frozen=json.loads((MODEL_DIR/'milestone4_validation_summary.json').read_text())
    assert frozen['status']=='milestone4_complete' and frozen['final_test_evaluated'] is False
    assert frozen['feature_names']==list(FEATURES) and frozen['target_name']==TARGET
    assert frozen['split_counts_per_window']['test']==6306
    selected={(int(z['window_days']),z['method']):z for z in frozen['model_selection']}
    assert len(selected)==6
    hashes={z['model_file']:digest(MODEL_DIR/z['model_file']) for z in selected.values()}

    data={w:{'X':[],'y':[],'keys':[]} for w in WINDOWS}
    with gzip.open(FEATURE_CSV,'rt',newline='',encoding='utf-8') as f:
        reader=csv.DictReader(f)
        assert reader.fieldnames==['split','prediction_date','customer_id','history_days',*FEATURES,TARGET]
        for row in reader:
            if row['split']!='test':continue
            w=int(row['history_days']);r=data[w]
            assert row['prediction_date'] in ['2011-08-24','2011-09-23','2011-10-23']
            r['X'].append([float(row[k]) for k in FEATURES]);r['y'].append(float(row[TARGET]))
            r['keys'].append((row['prediction_date'],row['customer_id']))
    for w in WINDOWS:
        r=data[w]
        assert len(r['y'])==6306
        r['X']=np.asarray(r['X'],dtype=np.float64);r['y']=np.asarray(r['y'],dtype=np.float64)
        assert np.isfinite(r['X']).all() and np.isfinite(r['y']).all()
        assert (r['X']>=0).all() and (r['y']>=0).all()
    assert data[30]['keys']==data[60]['keys']==data[90]['keys']
    assert np.array_equal(data[30]['y'],data[60]['y']) and np.array_equal(data[30]['y'],data[90]['y'])
    # Split boundaries were specified and audited in M2/M3; no training or validation targets loaded here.
    keys=data[30]['keys'];y=data[30]['y']
    assert len(set(keys))==len(keys)
    assert np.isclose(y.sum(),2120820.70,atol=.001)

    results=[]; per_date=[]; predrows=[]; arrays={}
    for w in WINDOWS:
        X=data[w]['X']
        preds={'Always predict £0':np.zeros(len(y)),
               'Historical spending baseline':X[:,0]*30.0/w}
        for method in ('Ridge regression','Random Forest'):
            z=selected[(w,method)]
            model=joblib.load(MODEL_DIR/z['model_file'])
            assert model.n_features_in_==len(FEATURES)
            preds[method]=np.clip(model.predict(X),0,None)
        for method in METHODS:
            p=np.asarray(preds[method],dtype=np.float64)
            assert len(p)==len(y) and np.isfinite(p).all() and (p>=0).all()
            arr=calc(y,p,keys)
            results.append({'window_days':w,'method':method,**arr})
            arrays[(w,method)]=p
            for d in sorted(set(k[0] for k in keys)):
                idx=np.asarray([key[0]==d for key in keys])
                a=calc(y[idx],p[idx],[key for key,keep in zip(keys,idx) if keep])
                per_date.append({'prediction_date':d,'window_days':w,'method':method,**a})
        for i,(date,customer) in enumerate(keys):
            predrows.append({'prediction_date':date,'customer_id':customer,'history_days':w,
                             'actual_future_30d_gbp':f'{y[i]:.4f}',
                             'zero_pred_gbp':'0.0000',
                             'baseline_pred_gbp':f'{preds["Historical spending baseline"][i]:.4f}',
                             'ridge_pred_gbp':f'{preds["Ridge regression"][i]:.4f}',
                             'forest_pred_gbp':f'{preds["Random Forest"][i]:.4f}'})

    # Simple sensitivity analysis (post-hoc, clearly identified, not used to select models)
    threshold=float(np.quantile(y,.99))
    low=y<=threshold
    robustness=[]
    for w in WINDOWS:
        for m in METHODS:
            p=arrays[(w,m)]
            robustness.append({'window_days':w,'method':m,'test_MAE_gbp':float(np.abs(p-y).mean()),
                               'mae_excluding_top_1pct_outcomes_gbp':float(np.abs(p[low]-y[low]).mean()),
                               'top1pct_threshold_actual_future_spending_gbp':threshold,
                               'excluded_rows':int((~low).sum())})

    comparisons={}
    for m in ('Historical spending baseline','Ridge regression','Random Forest'):
        comparisons[f'{m} | 90 vs 30 days'] = bootstrapped_diff(
            np.abs(arrays[(90,m)]-y),np.abs(arrays[(30,m)]-y),
            [k[1] for k in keys])
    for w in WINDOWS:
        for m in ('Ridge regression','Random Forest'):
            comparisons[f'{m} | versus zero | {w} days'] = bootstrapped_diff(
                np.abs(arrays[(w,m)]-y),y, [k[1] for k in keys])
    comparisons['Random Forest vs Ridge | 90 days'] = bootstrapped_diff(
        np.abs(arrays[(90,'Random Forest')]-y),
        np.abs(arrays[(90,'Ridge regression')]-y), [k[1] for k in keys])

    assert {k:digest(MODEL_DIR/k) for k in hashes}==hashes,'A model artifact unexpectedly changed'
    # Write machine-readable data
    def write_csv(path,records):
        with path.open('w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=list(records[0].keys()));w.writeheader();w.writerows(records)
    summaryrows=[]; date_rows=[]; robrows=[]
    for rows,outlist in [(results,summaryrows),(per_date,date_rows)]:
        for r in rows:
            outlist.append({k:(v['capture'] if v['capture'] is not None else '' ) if k=='top_decile_revenue_capture' else v
                             for k,v in r.items() if k!='top_decile'})
            outlist[-1]['top_decile_revenue_capture']=r['top_decile']['capture']
            outlist[-1]['top_decile_oracle_overlap']=r['top_decile']['overlap_with_true_top10']
            outlist[-1]['top_decile_k']=r['top_decile']['k']
    write_csv(OUT/'test_scores_all_methods.csv',summaryrows)
    write_csv(OUT/'test_scores_by_prediction_date.csv',date_rows)
    write_csv(OUT/'test_outlier_sensitivity.csv',robustness)
    with gzip.open(OUT/'test_predictions_all_methods.csv.gz','wt',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(predrows[0].keys()));w.writeheader();w.writerows(predrows)

    # Charts, neutral matplotlib default palette; no manual colors.
    plt.figure(figsize=(8.4,5.0))
    for method in METHODS:
        plt.plot(WINDOWS,[next(z['mae_gbp'] for z in results if z['window_days']==wd and z['method']==method) for wd in WINDOWS],marker='o',label=method)
    plt.xlabel('History used (days)');plt.ylabel('MAE, £ — lower is better');plt.xticks(WINDOWS)
    plt.title('Final unseen test: average customer revenue prediction error')
    plt.grid(alpha=.25);plt.legend(fontsize=8);plt.tight_layout()
    plt.savefig(OUT/'figure1_test_mae.png',dpi=180);plt.close()

    plt.figure(figsize=(8.4,5.0))
    for method in METHODS[1:]:
        plt.plot(WINDOWS,[next(z['top_decile']['capture'] for z in results if z['window_days']==wd and z['method']==method) for wd in WINDOWS],marker='o',label=method)
    plt.axhline(.1,linestyle='--',label='Random selection: 10% expected')
    plt.xlabel('History used (days)');plt.ylabel('Share of future revenue captured — higher is better')
    plt.xticks(WINDOWS);plt.ylim(bottom=0,top=1)
    plt.title('Final unseen test: revenue captured by highest-ranked 10%')
    plt.grid(alpha=.25);plt.legend(fontsize=8);plt.tight_layout()
    plt.savefig(OUT/'figure2_test_top10_capture.png',dpi=180);plt.close()

    plt.figure(figsize=(8.4,5.0))
    for method in METHODS:
        plt.plot(sorted(set(k[0] for k in keys)),
                 [next(z['mae_gbp'] for z in per_date if z['window_days']==90 and z['method']==method and z['prediction_date']==d) for d in sorted(set(k[0] for k in keys))],
                 marker='o',label=method)
    plt.xlabel('Final-test prediction date');plt.ylabel('MAE, £ — lower is better')
    plt.xticks(rotation=15);plt.title('90-day history: stability across the three test dates')
    plt.grid(alpha=.25);plt.legend(fontsize=8);plt.tight_layout()
    plt.savefig(OUT/'figure3_90d_per_date.png',dpi=180);plt.close()

    resultmeta={'completed':'milestone5_heldout_test_complete','test_cutoffs':sorted(set(k[0] for k in keys)),
                'test_customer_date_profiles_per_window':len(y),'unique_customers_in_test':len(set(k[1] for k in keys)),
                'actual_sales_total_gbp':float(y.sum()),'zero_spending_profile_share':float((y==0).mean()),
                'model_files_sha256_before_and_after_equal':True,'selected_hyperparams_unchanged':True,
                'input_features':FEATURES,'target':TARGET,'training_performed_in_m5':False,
                'results':results,'per_date_results':per_date,'robustness':robustness,
                'exploratory_paired_bootstrap':comparisons,
                'test_used_once':'M5 completed without tuning models after viewing outcomes',
                'caveats':['Only three future chronological test cutoffs',
                           'Model selection based on validation MAE, not ranking capture',
                           'Customer profiles repeated at multiple dates; cluster bootstrap approximates within-customer dependence only',
                           'Gross positive sales, not net revenue or profit',
                           'Zero-spend method has no useful ranking. Its deterministic top-decile is not meaningful.',
                           'The 10% capture chart ranks customer-date profiles pooled across three test dates.']}
    (OUT/'milestone5_results.json').write_text(json.dumps(resultmeta,indent=2),encoding='utf-8')
    print('Held-out test completed once with frozen M4 artifacts. No retraining.')
    for w in WINDOWS:
        print(f'\nHistory window: {w}d')
        for m in METHODS:
            r=next(z for z in results if z['window_days']==w and z['method']==m)
            print(f" {m:<29} MAE £{r['mae_gbp']:,.2f}   RMSE £{r['rmse_gbp']:,.2f}   top-10 capture {r['top_decile']['capture']:.3f}")
    print('\nTOP 90d forest vs Ridge:',comparisons['Random Forest vs Ridge | 90 days'])
    print('Unique customers',resultmeta['unique_customers_in_test'],'zero spending pct',round(resultmeta['zero_spending_profile_share']*100,1),'outlier threshold',threshold)
    return resultmeta

if __name__=='__main__':main()
