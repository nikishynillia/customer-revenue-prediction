#!/usr/bin/env python3
"""Independently verify M3 features, dates, keys, and sample raw transactions."""
import csv,gzip,json,random
from collections import defaultdict,Counter
from decimal import Decimal,ROUND_HALF_UP
from datetime import date,timedelta
from pathlib import Path
R=Path(__file__).resolve().parent
wins=(30,60,90)
fields=('historical_spend_gbp','orders','recency_days','average_order_value_gbp','distinct_products')
with (R/'milestone2_cutoff_schedule.csv').open(newline='') as f:schedule={r['prediction_date']:r for r in csv.DictReader(f)}
long={}
with gzip.open(R/'milestone3_features_long.csv.gz','rt',newline='') as f:
    for x in csv.DictReader(f):
        k=(x['prediction_date'],x['customer_id'],int(x['history_days']))
        assert k not in long
        long[k]=x
assert len(long)==124875
unique={k[:2] for k in long}
assert len(unique)==41625
counter=Counter();eligible=Counter();future_by_cut=defaultdict(dict)
with gzip.open(R/'milestone3_features_wide.csv.gz','rt',newline='') as f:
    for x in csv.DictReader(f):
        k=(x['prediction_date'],x['customer_id']);assert k in unique
        for w in wins:
            rec=long[k+(w,)]
            assert x['future_30d_spending_gbp']==rec['future_30d_spending_gbp']
            assert all(x[f'{field}_{w}d']==rec[field] for field in fields)
            assert (int(rec['orders'])==0)==(Decimal(rec['historical_spend_gbp'])==0)
            assert 1<=int(rec['recency_days'])<=w+1
            if int(rec['orders'])==0:assert int(rec['recency_days'])==w+1
        assert Decimal(x['historical_spend_gbp_30d'])<=Decimal(x['historical_spend_gbp_60d'])<=Decimal(x['historical_spend_gbp_90d'])
        eligible[k[0]]+=1;counter[x['split']]+=1;future_by_cut[k[0]][k[1]]=Decimal(x['future_30d_spending_gbp'])
assert counter==Counter(train=29339,validation=5980,test=6306)
for dt,sch in schedule.items():
    assert eligible[dt]==int(sch['eligible_customers'])
    assert sum(v>0 for v in future_by_cut[dt].values())==int(sch['positive_future_spenders'])
    assert sum(future_by_cut[dt].values())==Decimal(sch['future_gross_sales_GBP'])
# Independent source spot checks from raw sales records, without using builder calculations.
rng=random.Random(20261123)
chosen=set()
for spl in ('train','validation','test'):
    avail=[k for k in unique if schedule[k[0]]['split']==spl]
    chosen.update(rng.sample(avail,12))
zero30=[k for k in unique if int(long[k+(30,)]['orders'])==0]
chosen.update(rng.sample(zero30,12))
ids={cu for _,cu in chosen};source=defaultdict(list)
with gzip.open(R/'online_retail_clean_sales_m2.csv.gz','rt',newline='') as f:
    for r in csv.DictReader(f):
        if r['customer_id'] in ids:source[r['customer_id']].append(r)
checks=0
for dt,cu in chosen:
    cutoff=date.fromisoformat(dt);ent=source[cu]
    future=sum((Decimal(r['line_revenue_gbp']) for r in ent if cutoff<=date.fromisoformat(r['invoice_date'])<cutoff+timedelta(days=30)),Decimal('0'))
    assert future==Decimal(long[(dt,cu,30)]['future_30d_spending_gbp'])
    for w in wins:
        hist=[r for r in ent if cutoff-timedelta(days=w)<=date.fromisoformat(r['invoice_date'])<cutoff]
        rec=long[(dt,cu,w)]
        spent=sum((Decimal(r['line_revenue_gbp']) for r in hist),Decimal('0'))
        invoices={r['invoice'] for r in hist}
        codes={r['stock_code'] for r in hist}
        recent=(cutoff-max(date.fromisoformat(r['invoice_date']) for r in hist)).days if hist else w+1
        assert spent==Decimal(rec['historical_spend_gbp'])
        assert len(invoices)==int(rec['orders'])
        assert len(codes)==int(rec['distinct_products'])
        assert recent==int(rec['recency_days'])
        mean=(spent/len(invoices)).quantize(Decimal('0.0001'),rounding=ROUND_HALF_UP) if invoices else Decimal(0)
        assert mean==Decimal(rec['average_order_value_gbp'])
        checks+=1
out={'all_checks_passed':True,'wide_profiles':len(unique),'long_rows':len(long),'training_profiles':counter['train'],
'validation_profiles':counter['validation'],'test_profiles':counter['test'],
'schedule_cutoffs_verified':len(schedule),'source_feature_windows_independently_verified':checks,
'checks':['unique customer-date-history combinations','full long/wide equality','20 cutoff customer, buyer, gross totals match Milestone 2','30/60/90 nested purchase history','date-based source spot checks','shared future target per customer and cutoff'],'no_ml_training':True}
(R/'milestone3_verification.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
print('PASS independent verification',len(unique),'customer-date profiles and',checks,'source-feature windows')
