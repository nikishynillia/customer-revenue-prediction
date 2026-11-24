import re, json, csv, gzip, collections
from pathlib import Path
from zipfile import ZipFile
from datetime import date, timedelta
from lxml import etree
ROOT=Path(__file__).resolve().parent
src=ROOT/'online_retail_II.xlsx'
NS='{http://schemas.openxmlformats.org/spreadsheetml/2006/main}'
R=re.compile(rb'<row r="\d+"[^>]*>(.*?)</row>',re.S)
C=re.compile(rb'<c r="([A-H])\d+"([^>]*)><v>([^<]*)</v></c>')
stats={'source_file':src.name,'sheet_counts':{},'total_rows':0,'missing_customer_id_rows':0,'nonpositive_quantity_rows':0,'nonpositive_price_rows':0,'cancelled_invoice_rows':0,'valid_sale_rows':0,'invalid_date_rows':0,'valid_sales_gross_amount':0.0,'duplicate_cross_sheet_rows_skipped':0}
daily=collections.defaultdict(float); orders=set(); custord=collections.defaultdict(set); custdates=collections.defaultdict(set); months=collections.Counter()
with ZipFile(src) as z:
    strs=[''.join(s.itertext()) for s in etree.fromstring(z.read('xl/sharedStrings.xml')).findall(NS+'si')]
    for name,p in [('Year 2009-2010','xl/worksheets/sheet1.xml'),('Year 2010-2011','xl/worksheets/sheet2.xml')]:
        ss={'rows':0,'missing_customer_id':0,'valid_sale_rows':0,'min_date':None,'max_date':None}
        raw=z.read(p)
        for row in R.finditer(raw):
            dct={m.group(1): (m.group(2),m.group(3)) for m in C.finditer(row.group(1))}
            if ss['rows']==0 and dct.get(b'A',(b'',b''))[1]==b'44' and b't="s"' in dct.get(b'A',(b'',b''))[0]: continue
            ss['rows']+=1;stats['total_rows']+=1
            try:dt=date(1899,12,30)+timedelta(days=int(float(dct[b'E'][1])))
            except (KeyError,ValueError,OverflowError):stats['invalid_date_rows']+=1;continue
            di=dt.isoformat()
            if name=='Year 2010-2011' and date(2010,12,1)<=dt<=date(2010,12,9):
                stats['duplicate_cross_sheet_rows_skipped']+=1
                stats['total_rows']-=1;ss['rows']-=1
                continue
            ss['min_date']=min(ss['min_date'] or di,di);ss['max_date']=max(ss['max_date'] or di,di)
            cust=dct.get(b'G',(b'',b''))[1]
            if not cust:stats['missing_customer_id_rows']+=1;ss['missing_customer_id']+=1
            invattr,invraw=dct.get(b'A',(b'',b''));inv=invraw.decode()
            if b't="s"' in invattr:inv=strs[int(inv)]
            cancelled=inv.upper().startswith('C')
            if cancelled:stats['cancelled_invoice_rows']+=1
            try:qty=float(dct[b'D'][1]);pr=float(dct[b'F'][1])
            except (ValueError,KeyError):continue
            if qty<=0:stats['nonpositive_quantity_rows']+=1
            if pr<=0:stats['nonpositive_price_rows']+=1
            if cust and qty>0 and pr>0 and not cancelled:
                cs=cust.decode()
                if b't="s"' in dct[b'G'][0]:cs=strs[int(cs)]
                if cs.replace('.0','').isdigit():cs=str(int(float(cs)))
                revenue=qty*pr;stats['valid_sales_gross_amount']+=revenue;stats['valid_sale_rows']+=1;ss['valid_sale_rows']+=1
                orders.add((cs,inv));custord[cs].add(inv);custdates[cs].add(dt);daily[(cs,di)]+=revenue;months[di[:7]]+=1
        del raw
        stats['sheet_counts'][name]=ss
        print('SHEET',name,ss,flush=True)
stats['raw_rows_including_overlap']=stats['total_rows']+stats['duplicate_cross_sheet_rows_skipped']
stats['overall_min_date']=min(x['min_date'] for x in stats['sheet_counts'].values())
stats['overall_max_date']=max(x['max_date'] for x in stats['sheet_counts'].values())
stats['valid_unique_customers']=len(custord);stats['valid_unique_customer_invoice_pairs']=len(orders)
stats['customers_with_2plus_orders']=sum(len(x)>=2 for x in custord.values());stats['customers_with_5plus_orders']=sum(len(x)>=5 for x in custord.values())
stats['monthly_valid_sale_rows']=dict(sorted(months.items()));stats['valid_unique_customer_days']=len(daily)
start=date(2010,4,1);end=date.fromisoformat(stats['overall_max_date']);cuts=[]
while start+timedelta(days=30)<=end:
    en=start+timedelta(days=30);beg=start-timedelta(days=90)
    eligible=[c for c,dates in custdates.items() if any(beg<=dt<start for dt in dates)]
    outcome=sum(any(start<=dt<en for dt in custdates[c]) for c in eligible)
    cuts.append(dict(cutoff=start.isoformat(),eligible_customers_past90d=len(eligible),positive_next30d=outcome,positive_fraction=round(outcome/len(eligible),4) if eligible else None))
    start+=timedelta(days=30)
stats['candidate_cutoffs_30day']=cuts
with gzip.open(ROOT/'online_retail_clean_customer_daily.csv.gz','wt',newline='') as out:
    w=csv.writer(out);w.writerow(['customer_id','date','gross_positive_sale_revenue'])
    for (cid,d),v in sorted(daily.items(),key=lambda a:(a[0][1],a[0][0])):w.writerow([cid,d,round(v,2)])
with open(ROOT/'online_retail_audit.json','w') as out:json.dump(stats,out,indent=2)
print('FINISH',json.dumps({k:v for k,v in stats.items() if k not in ('monthly_valid_sale_rows','candidate_cutoffs_30day','sheet_counts')}),flush=True)
print('FINAL CUTOFFS',cuts[-5:],flush=True)
