#!/usr/bin/env python3
"""Build reproducible, leakage-safe customer features from Milestone 2 sales.
Requires Python 3 standard library only; no machine learning training is performed.
"""
import bisect,csv,gzip,json
from pathlib import Path
from collections import defaultdict,Counter
from datetime import date
ROOT=Path(__file__).resolve().parent
SOURCE=ROOT/'online_retail_clean_sales_m2.csv.gz'
SCHEDULE=ROOT/'milestone2_cutoff_schedule.csv'
LONG=ROOT/'milestone3_features_long.csv.gz'
WIDE=ROOT/'milestone3_features_wide.csv.gz'
PREVIEW=ROOT/'milestone3_customer_profiles_preview.csv'
QUALITY=ROOT/'milestone3_feature_quality.json'
WINDOWS=(30,60,90)
KEYS=('historical_spend_gbp','orders','recency_days','average_order_value_gbp','distinct_products')

def amount(raw):
    whole,_,fraction=raw.partition('.')
    assert not whole.startswith('-') and len(fraction)<=4
    return int(whole)*10000+int((fraction+'0000')[:4])
def as_gbp(n):return f'{n//10000}.{n%10000:04d}'
def mean_gbp(n,cnt):return as_gbp((n+cnt//2)//cnt) if cnt else '0.0000'

def main():
    with SCHEDULE.open(newline='',encoding='utf-8') as f:schedule=list(csv.DictReader(f))
    assert len(schedule)==20
    customers=defaultdict(list)
    nraw=0
    with gzip.open(SOURCE,'rt',encoding='utf-8',newline='') as f:
        for row in csv.DictReader(f):
            customers[row['customer_id']].append((date.fromisoformat(row['invoice_date']).toordinal(),row['invoice'],row['stock_code'],amount(row['line_revenue_gbp'])))
            nraw+=1
    assert nraw==791045 and len(customers)==5878
    data={}
    for cu,items in customers.items():
        items.sort(key=lambda r:r[0]);dd=[r[0] for r in items];cums=[0]
        for r in items:cums.append(cums[-1]+r[3])
        data[cu]=(items,dd,cums)
    del customers
    wide_fields=['split','prediction_date','customer_id']+[f'{k}_{w}d' for w in WINDOWS for k in KEYS]+['future_30d_spending_gbp']
    long_fields=['split','prediction_date','customer_id','history_days']+list(KEYS)+['future_30d_spending_gbp']
    totals=Counter();zero=Counter();positive=Counter();snapshot_by_split=Counter();bycut=[];samples=[];nlong=0
    with gzip.open(WIDE,'wt',encoding='utf-8',newline='',compresslevel=6) as fw,gzip.open(LONG,'wt',encoding='utf-8',newline='',compresslevel=6) as fl:
        ww=csv.DictWriter(fw,wide_fields);lw=csv.DictWriter(fl,long_fields);ww.writeheader();lw.writeheader()
        for sche in schedule:
            c=date.fromisoformat(sche['prediction_date']).toordinal();split=sche['split'];ct=0;nonzero=0;gross=0
            for cust,(items,dates,prefix) in data.items():
                lo=bisect.bisect_left(dates,c-90);hi=bisect.bisect_left(dates,c)
                if lo==hi:continue # 90d eligibility is fixed for all 3 windows
                future_hi=bisect.bisect_left(dates,c+30,hi)
                target=prefix[future_hi]-prefix[hi]
                assert target>=0 and (hi==future_hi or (c<=dates[hi] and dates[future_hi-1]<c+30))
                ct+=1;gross+=target;nonzero+=(target>0)
                row={'split':split,'prediction_date':sche['prediction_date'],'customer_id':cust,'future_30d_spending_gbp':as_gbp(target)}
                for w in WINDOWS:
                    idx=bisect.bisect_left(dates,c-w,lo,hi)
                    hist=items[idx:hi];spent=prefix[hi]-prefix[idx]
                    assert not hist or (c-w<=hist[0][0]<=hist[-1][0]<c)
                    orders=len({x[1] for x in hist});products=len({x[2] for x in hist})
                    recency=(c-hist[-1][0]) if hist else w+1
                    assert 1<=recency<=w+1 and (orders>0)==(spent>0)
                    vals=(as_gbp(spent),orders,recency,mean_gbp(spent,orders),products)
                    for k,v in zip(KEYS,vals):row[f'{k}_{w}d']=v
                    lw.writerow({'split':split,'prediction_date':sche['prediction_date'],'customer_id':cust,'history_days':w,**dict(zip(KEYS,vals)),'future_30d_spending_gbp':as_gbp(target)})
                    zero[w]+=(orders==0);nlong+=1
                assert amount(row['historical_spend_gbp_30d'])<=amount(row['historical_spend_gbp_60d'])<=amount(row['historical_spend_gbp_90d'])
                ww.writerow(row)
                if len(samples)<18:samples.append(row)
                snapshot_by_split[split]+=1
            assert ct==int(sche['eligible_customers']),(sche['prediction_date'],ct,sche['eligible_customers'])
            assert nonzero==int(sche['positive_future_spenders'])
            assert as_gbp(gross)==sche['future_gross_sales_GBP'],(sche['prediction_date'],as_gbp(gross),sche['future_gross_sales_GBP'])
            positive[split]+=nonzero;totals[split]+=gross
            bycut.append({'date':sche['prediction_date'],'split':split,'profiles':ct,'future_buyers':nonzero,'future_sales_gbp':as_gbp(gross)})
    assert snapshot_by_split==Counter(train=29339,validation=5980,test=6306)
    assert positive==Counter(train=8951,validation=1871,test=2387)
    assert nlong==124875
    with PREVIEW.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,wide_fields);w.writeheader();w.writerows(samples)
    quality={'milestone':'3','completed':True,'original_clean_sales_rows':nraw,'source_customers':len(data),'cutoff_dates':len(schedule),
             'customer_date_profiles':sum(snapshot_by_split.values()),'all_history_window_rows':nlong,
             'customer_date_profiles_by_split':dict(snapshot_by_split),'positive_future_buyers_by_split':dict(positive),
             'zero_history_profiles_by_window':{str(k):v for k,v in zero.items()},
             'total_next_30d_sales_gbp_by_split':{k:as_gbp(v) for k,v in totals.items()},
             'source_date_leakage_check':'passed: all feature observations are before cutoff; all outcomes at/after cutoff and strictly before cutoff+30d',
             'same_cohort_and_target_checks':'passed: all history lengths based on one common customer-date record',
             'schedule_reconciliation':'passed: all 20 cutoffs match M2 counts, buyers and actual future gross sales exactly',
             'no_ml_training':True,'split_details':bycut}
    QUALITY.write_text(json.dumps(quality,indent=2),encoding='utf-8')
    print('Generated:',sum(snapshot_by_split.values()),'profiles,',nlong,'per-window rows. Source dates and 20 cutoff checks passed.')
if __name__=='__main__':main()
