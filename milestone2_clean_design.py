#!/usr/bin/env python3
"""Read-only, reproducible Milestone 2 cleaner and split design for UCI Online Retail II.

Inputs: original two-sheet Excel workbook; no workbook modifications.
Outputs: gzipped positive-sales lines, full exclusion counts, snapshot eligibility
and leakage-check report, plus machine-readable experiment protocol.
No ML models are fitted here.
"""
from __future__ import annotations
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta
from decimal import Decimal, InvalidOperation
from pathlib import Path
from zipfile import ZipFile
from lxml import etree
import argparse, csv, gzip, hashlib, json, math, re, sys

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'online_retail_II.xlsx'
DATA_OUT = ROOT / 'online_retail_clean_sales_m2.csv.gz'
STATS_OUT = ROOT / 'milestone2_quality_checks.json'
PROTOCOL_OUT = ROOT / 'milestone2_experiment_protocol.json'
NS = '{http://schemas.openxmlformats.org/spreadsheetml/2006/main}'
ROW_RE = re.compile(rb'<row r="(\d+)"[^>]*>(.*?)</row>', re.S)
CELL_RE = re.compile(rb'<c r="([A-H])\d+"([^>]*)><v>([^<]*)</v></c>')
XL_EPOCH = date(1899,12,30)
SHEETS = [('Year 2009-2010','xl/worksheets/sheet1.xml'),('Year 2010-2011','xl/worksheets/sheet2.xml')]
CUTOFF_FIRST = date(2010,4,1)
STEP_DAYS = 30
HISTORY_WINDOWS = [30,60,90]
OUTCOME_DAYS = 30
TRAIN_END = date(2011,4,26)
VALIDATION_END = date(2011,7,25)


def as_str(entry, strings):
    if entry is None: return ''
    attrs, raw = entry
    if b't="s"' in attrs:
        return strings[int(raw)]
    return raw.decode('utf-8','replace')


def clean_num(s):
    x=Decimal(s)
    if not x.is_finite(): raise InvalidOperation('nonfinite')
    return x


def norm_id(s):
    if not s.strip():return ''
    try:
        d=Decimal(s)
        return str(int(d)) if d == d.to_integral_value() else format(d,'f')
    except InvalidOperation:
        return s.strip()


def process():
    stats = {
        'source_file': SOURCE.name, 'sheet_raw_data_rows': {},
        'raw_data_rows':0, 'exact_cross_sheet_overlap_dropped':0,
        'cross_sheet_overlap_unmatched':0, 'sheet1_overlap_rows':0,
        'valid_positive_sale_rows':0, 'exclusion_reason_priority':{},
        'invalid_date_rows':0, 'retained_exact_duplicate_line_signatures':0,
        'earliest_date':None,'latest_date':None,
        'notes': ['Exclusion reasons are mutually exclusive and prioritized.',
                  'Retain identical line items inside a source worksheet; they may represent valid repeated purchase lines.',
                  'Gross positive invoiced sales: no refund or cancellation netting; NOT net revenue or profit.',
                  'Unit price normalized to 4 decimal places from Excel floating-point text; checked absolute difference under 1e-7 GBP.',
                  'Calendar dates used; time-of-day discarded for midnight-cutoff analyses.']
    }
    why = Counter()
    overlap = Counter()
    duplicate_set=set()
    daily = defaultdict(Decimal)
    dates_by_cust=defaultdict(set)
    order_keys=set()
    sale_customers=set()
    monthly_count=Counter()
    row_source_count=Counter()
    invalid_examples=[]
    counters=Counter()
    with ZipFile(SOURCE) as z:
        strings=[''.join(elem.itertext()) for elem in etree.fromstring(z.read('xl/sharedStrings.xml')).findall(NS+'si')]
        with gzip.open(DATA_OUT,'wt',encoding='utf-8',newline='',compresslevel=6) as f:
            w=csv.writer(f)
            w.writerow(['invoice','stock_code','description','quantity','invoice_date','unit_price_gbp','customer_id','country','line_revenue_gbp','source_sheet','source_row'])
            for sheet_name,xml_path in SHEETS:
                print('PROCESSING',sheet_name,flush=True)
                src_count=0
                chunk = z.read(xml_path)
                for row in ROW_RE.finditer(chunk):
                    row_number=int(row.group(1))
                    if row_number==1:continue
                    src_count+=1;stats['raw_data_rows']+=1
                    cells={m.group(1).decode():(m.group(2),m.group(3)) for m in CELL_RE.finditer(row.group(2))}
                    data=[as_str(cells.get(col),strings) for col in 'ABCDEFGH']
                    inv,stock,description,qty_str,dt_str,pr_str,cust_str,country=data
                    try:
                        serial = float(dt_str)
                        if not math.isfinite(serial):raise ValueError('nonfinite date')
                        day= XL_EPOCH+timedelta(days=int(serial))
                    except (ValueError,OverflowError):
                        stats['invalid_date_rows']+=1;why['invalid_date']+=1
                        invalid_examples.append({'sheet':sheet_name,'row':row_number}) if len(invalid_examples)<10 else None
                        continue
                    # By design, worksheet 1 is authoritative for dates <= Dec 9 2010.
                    # Exact multiset verification of the cross-sheet overlap before dropping.
                    if date(2010,12,1)<=day<=date(2010,12,9):
                        signature=tuple(data)
                        if sheet_name==SHEETS[0][0]:
                            overlap[signature]+=1;stats['sheet1_overlap_rows']+=1
                        else:
                            if overlap[signature]:
                                overlap[signature]-=1
                                stats['exact_cross_sheet_overlap_dropped']+=1
                                continue
                            stats['cross_sheet_overlap_unmatched']+=1
                    di=day.isoformat()
                    if not stats['earliest_date'] or di<stats['earliest_date']:stats['earliest_date']=di
                    if not stats['latest_date'] or di>stats['latest_date']:stats['latest_date']=di
                    try:
                        qty=clean_num(qty_str)
                        price=clean_num(pr_str)
                    except (InvalidOperation,ValueError,TypeError):
                        why['invalid_numeric_fields']+=1;continue
                    # Disjoint exclusion priority, with nonexclusive flags also counted below.
                    cust=norm_id(cust_str)
                    if not cust:why['missing_customer_id']+=1; continue
                    if inv.upper().startswith('C'):why['cancellation_invoice']+=1;continue
                    if qty<=0:why['nonpositive_quantity']+=1;continue
                    if price<=0:why['nonpositive_price']+=1;continue
                    # Source Excel numbers sometimes contain binary display artifacts
                    # (e.g. 1.799999999999); retain the underlying 4-decimal precision.
                    assert abs(price-price.quantize(Decimal('0.0001'))) < Decimal('0.0000001'), (sheet_name,row_number,price)
                    price=price.quantize(Decimal('0.0001'))
                    revenue = (qty*price).quantize(Decimal('0.0001'))
                    if revenue<=0:why['nonpositive_revenue']+=1;continue
                    signature=(sheet_name,inv,stock,description,qty_str,dt_str,pr_str,cust,country)
                    signature_hash=hashlib.blake2b(repr(signature).encode(),digest_size=16).digest()
                    if signature_hash in duplicate_set:stats['retained_exact_duplicate_line_signatures']+=1
                    else:duplicate_set.add(signature_hash)
                    w.writerow([inv,stock,description,format(qty,'f'),di,format(price,'f'),cust,country,format(revenue,'f'),sheet_name,row_number])
                    stats['valid_positive_sale_rows']+=1
                    counters['total_revenue_cents_approx'] += int((revenue*Decimal('100')).quantize(Decimal('1')))
                    daily[(cust,day)] += revenue
                    dates_by_cust[cust].add(day)
                    sale_customers.add(cust)
                    order_keys.add((cust,inv))
                    monthly_count[di[:7]]+=1
                    row_source_count[sheet_name]+=1
                stats['sheet_raw_data_rows'][sheet_name]=src_count
                del chunk
                print('READ',src_count,'valid so far',stats['valid_positive_sale_rows'],flush=True)
    unmatched_sheet1=sum(overlap.values())
    assert unmatched_sheet1 == 0, f'Unmatched sheet1 overlap rows {unmatched_sheet1}'
    assert stats['cross_sheet_overlap_unmatched'] == 0, 'Unmatched second-sheet overlap rows; requires manual reconciliation.'
    assert stats['exact_cross_sheet_overlap_dropped']==stats['sheet1_overlap_rows'],'Cross-sheet overlap mismatch'
    stats['overlap_check']='passed: matching transaction multisets, no unmatched rows'
    stats['exclusion_reason_priority']=dict(why)
    stats['retained_source_rows']=stats['raw_data_rows']-stats['exact_cross_sheet_overlap_dropped']
    assert sum(why.values())+stats['valid_positive_sale_rows']==stats['retained_source_rows'], 'Counts do not reconcile'
    stats['unique_customers']=len(sale_customers)
    stats['unique_customer_invoice_pairs']=len(order_keys)
    stats['retained_distinct_customer_dates']=len(daily)
    stats['valid_source_rows_by_sheet']=dict(row_source_count)
    stats['monthly_positive_sale_rows']=dict(sorted(monthly_count.items()))
    stats['invalid_date_examples']=invalid_examples
    allcuts=[]
    c=CUTOFF_FIRST
    last_day=date.fromisoformat(stats['latest_date'])
    while c+timedelta(days=OUTCOME_DAYS-1)<=last_day:
        histstart=c-timedelta(days=max(HISTORY_WINDOWS))
        eligible=[cu for cu, ds in dates_by_cust.items() if any(histstart<=dd<c for dd in ds)]
        outcome_pos=sum(1 for cu in eligible if any(c<=dd<c+timedelta(days=30) for dd in dates_by_cust[cu]))
        # Positive sales only: retained outcomes where customer is eligible based on prior 90d.
        future_gross = sum((sum((daily[(cu,d)] for d in dates_by_cust[cu] if c<=d<c+timedelta(days=30)),Decimal('0')) for cu in eligible),Decimal('0'))
        subset='train' if c<=TRAIN_END else ('validation' if c<=VALIDATION_END else 'test')
        allcuts.append(dict(cutoff=c.isoformat(),history_start_90d=histstart.isoformat(),outcome_start=c.isoformat(),outcome_end_exclusive=(c+timedelta(days=30)).isoformat(),split=subset,eligible_customers=len(eligible),customers_with_positive_future_sales=outcome_pos,share_positive_future_sales=round(outcome_pos/len(eligible),5) if eligible else None,gross_future_sales_gbp=format(future_gross,'f')))
        c+=timedelta(days=STEP_DAYS)
    grouped=defaultdict(list)
    for x in allcuts: grouped[x['split']].append(x)
    assert set(grouped)=={'train','validation','test'}
    assert len(allcuts)>=18
    for x in allcuts:
        assert date.fromisoformat(x['history_start_90d']) >= date.fromisoformat(stats['earliest_date'])
        assert date.fromisoformat(x['outcome_end_exclusive'])-timedelta(days=1) <= last_day
    for s in ('train','validation'):
        later=('validation' if s=='train' else 'test')
        earlier_end=max(date.fromisoformat(x['outcome_end_exclusive']) for x in grouped[s])
        later_start=min(date.fromisoformat(x['outcome_start']) for x in grouped[later])
        assert earlier_end<=later_start,(s,earlier_end,later_start)
    stats['cutoff_rows']=allcuts
    stats['split_overview']={s:dict(first_cutoff=v[0]['cutoff'],last_cutoff=v[-1]['cutoff'],number_of_cutoffs=len(v),customer_snapshots=sum(x['eligible_customers'] for x in v),positive_outcome_snapshots=sum(x['customers_with_positive_future_sales'] for x in v),unique_customer_snapshots_not_independent=True) for s,v in grouped.items()}
    stats['date_checks']='passed: no future label windows overlapping later split starts; 90d history complete; 30d labels in source bounds'
    protocol={
        'research_question':'How Much Purchase History Is Needed to Predict Future Customer Revenue? A Comparison of Simple and Machine Learning Models',
        'source':'Original Online Retail II workbook, two sheets, dates 2009-12-01 to 2011-12-09',
        'status':'experimental design frozen at Milestone 2; features/models not built',
        'target':'sum of positive quantity × positive unit price across known customer transactions for 30 calendar days [cutoff, cutoff+30); GBP; gross recorded sales, not net of refunds',
        'population':'customers with >=1 positive sale in [cutoff-90d, cutoff); all are evaluated at all history windows, even if 30d/60d inputs have zero purchases',
        'features_to_generate_next_milestone':['30/60/90d positive sales total','30/60/90d invoice counts (customer ID + invoice)','purchase recency capped to window+1 days when no purchases','mean invoice value','distinct stock-code count'],
        'history_windows_days':HISTORY_WINDOWS,
        'prediction_horizon_days':30,
        'snapshot_step_days':STEP_DAYS,
        'snapshot_cutoffs':[x['cutoff'] for x in allcuts],
        'split_cutoffs':{s:[x['cutoff'] for x in grouped[s]] for s in ('train','validation','test')},
        'split_logic':'chronological by cutoff; last training label end <= first validation cutoff; last validation label end <= first test cutoff',
        'baseline':'gross historical spend × (30 / history_window_days), zero for no transactions within window',
        'models':['baseline','Ridge regression (nonnegative clipped predictions)','RandomForestRegressor'],
        'primary_metric':'MAE (GBP) on untouched chronological test snapshots',
        'secondary_metrics':['RMSE and median absolute error for sensitivity to extreme buyers','top 10% predicted customer future revenue capture vs actual','mean error/bias and positive-outcome share'],
        'fairness':['identical eligible snapshots at all 3 history windows','shared target, test snapshots and feature definitions','hyperparameters chosen with validation only','same historical information horizon for all compared models'],
        'leakage_guards':['per snapshot feature date < cutoff and feature date >= cutoff-W','target date >= cutoff and < cutoff+30','no final test outcomes in feature fitting, tuning, scaling or feature selection','earlier outcome windows do not overlap subsequent split start','one snapshot row per customer/cutoff/window'],
        'limitations':['Repeated customer snapshots correlated; account for dependence in uncertainty','Dataset is 2009-2011 UK retail with wholesale characteristics, not modern DTC advertising','Returns/cancellations excluded, so prediction target is gross positive invoiced sales not net realized revenue',
                       'Extreme single-line and customer spending outliers will be disclosed; robustness tests of prediction errors on and excluding top-1%-revenue customers may be informative','Customer IDs missing on many rows; population restricted to identifiable customers','Historical purchasing of new customers / customer churn outside 90-day activity window not evaluated'],
        'disclosure':['No ML fitting or trained results in Milestone 2','Source CSV records include customer IDs present in the public dataset; do not upload proprietary customer data'],
    }
    STATS_OUT.write_text(json.dumps(stats,indent=2),encoding='utf-8')
    PROTOCOL_OUT.write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    print('DONE CLEANED',stats['valid_positive_sale_rows'],'CUSTOMERS',stats['unique_customers'],'DUPLICATES',stats['retained_exact_duplicate_line_signatures'],flush=True)
    print('SPLITS',json.dumps(stats['split_overview']),flush=True)
    print('EXCLUSIONS',json.dumps(stats['exclusion_reason_priority']),flush=True)

if __name__=='__main__':
    process()
