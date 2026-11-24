#!/usr/bin/env python3
"""Independent validation of Milestone 2 output. Does not train models."""
from __future__ import annotations
import csv,gzip,heapq,json,collections
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path
ROOT=Path(__file__).resolve().parent
q=json.loads((ROOT/'milestone2_quality_checks.json').read_text())
p=json.loads((ROOT/'milestone2_experiment_protocol.json').read_text())
rows=0; revenue=Decimal('0'); customers=set(); orders=set(); countries=collections.Counter(); code_counts=collections.Counter(); unusual=[]; top=[]; missing=[]
for f in [ROOT/'online_retail_clean_sales_m2.csv.gz']:
    with gzip.open(f,'rt',encoding='utf-8',newline='') as stream:
        reader=csv.DictReader(stream)
        assert reader.fieldnames == ['invoice','stock_code','description','quantity','invoice_date','unit_price_gbp','customer_id','country','line_revenue_gbp','source_sheet','source_row']
        for r in reader:
            rows+=1
            d=date.fromisoformat(r['invoice_date'])
            qty=Decimal(r['quantity']); price=Decimal(r['unit_price_gbp']); amt=Decimal(r['line_revenue_gbp'])
            assert amt==qty*price, (rows,r)
            assert qty>0 and price>0 and amt>0
            assert r['customer_id'] and not r['invoice'].upper().startswith('C')
            assert r['source_sheet'] in ('Year 2009-2010','Year 2010-2011')
            if r['source_sheet']=='Year 2010-2011':assert d>date(2010,12,9)
            customers.add(r['customer_id']); orders.add((r['customer_id'],r['invoice']))
            countries[r['country']]+=1; code_counts[r['stock_code']]+=1
            revenue+=amt
            if len(top)<12:heapq.heappush(top,(amt,rows,r['invoice'],r['stock_code'],r['description'][:60]))
            elif amt>top[0][0]:heapq.heapreplace(top,(amt,rows,r['invoice'],r['stock_code'],r['description'][:60]))
            if r['invoice']=='' or r['stock_code']=='':missing.append(rows) if len(missing)<5 else None
assert rows==q['valid_positive_sale_rows']
assert len(customers)==q['unique_customers']
assert len(orders)==q['unique_customer_invoice_pairs']
assert q['raw_data_rows']-q['exact_cross_sheet_overlap_dropped']==q['retained_source_rows']
assert q['retained_source_rows']-sum(q['exclusion_reason_priority'].values())==rows
assert len(p['snapshot_cutoffs'])==20
splits=p['split_cutoffs'];assert {s:len(v) for s,v in splits.items()}=={'train':14,'validation':3,'test':3}
for s,cutoffs in splits.items():
    for cutoff in cutoffs:
        dd=date.fromisoformat(cutoff)
        assert dd-timedelta(days=90) >= date(2009,12,1)
        assert dd+timedelta(days=30) <= date(2011,12,10)
assert date.fromisoformat(splits['train'][-1])+timedelta(days=30)<=date.fromisoformat(splits['validation'][0])
assert date.fromisoformat(splits['validation'][-1])+timedelta(days=30)<=date.fromisoformat(splits['test'][0])
answer={
 'verification_passed':True,'cleaned_rows':rows,'customers':len(customers),'distinct_customer_invoices':len(orders),'gross_recorded_positive_sales_gbp':str(revenue),
 'top_10_line_values':[{ 'line_revenue_gbp':str(v),'invoice':inv,'stock_code':stock,'description':desc} for v,_,inv,stock,desc in sorted(top,reverse=True)[:10]],
 'top_countries_by_lines':countries.most_common(8),'top_stock_codes_by_lines':code_counts.most_common(14),'blank_invoice_or_sku_example_row_numbers':missing,
 'split_dates':{s:dict(first=v[0],last=v[-1],snapshots=len(v)) for s,v in splits.items()},
 'automatic_temporal_checks':'PASSED (schema, positive sales, preserved provenance, overlap, cutoff coverage, split label boundaries)',
 'not_checked_yet':'Feature-level leak tests must run in Milestone 3 on generated customer features; ML accuracy cannot be claimed at Milestone 2.'}
(ROOT/'milestone2_verification.json').write_text(json.dumps(answer,indent=2),encoding='utf-8')
print(json.dumps(answer,indent=2))
