#!/usr/bin/env python3
"""Build easy-to-open Excel preview of M3 model-ready data with artifact_tool."""
import csv,json
from pathlib import Path
from artifact_tool import Workbook,SpreadsheetFile
R=Path(__file__).resolve().parent
with (R/'milestone3_customer_profiles_preview.csv').open(newline='',encoding='utf-8') as f: samples=list(csv.DictReader(f))
stat=json.loads((R/'milestone3_feature_quality.json').read_text())
wb=Workbook.create();s=wb.worksheets.add('Overview')
s.get_range('A1:B1').values=[['MILESTONE 3','CUSTOMER PROFILES READY']]
s.get_range('A1:B1').format={'fill':'#0F766E','font':{'bold':True,'color':'#FFFFFF'},'row_height':30}
rows=[['What we measured','Count'],['Original cleaned sale lines',stat['original_clean_sales_rows']],['Different source customers',stat['source_customers']],['Prediction dates',stat['cutoff_dates']],['Customer + prediction-date profiles',stat['customer_date_profiles']],['30/60/90-day feature rows combined',stat['all_history_window_rows']],['Training profiles',stat['customer_date_profiles_by_split']['train']],['Validation profiles',stat['customer_date_profiles_by_split']['validation']],['Test profiles',stat['customer_date_profiles_by_split']['test']],['No purchases in past 30 days',stat['zero_history_profiles_by_window']['30']]]
s.get_range('A3:B12').values=rows
s.get_range('A3:B3').format={'fill':'#0F766E','font':{'bold':True,'color':'#FFFFFF'}}
s.get_range('A1:A12').format.column_width=46;s.get_range('B1:B12').format.column_width=36
s.freeze_panes.freeze_rows(3)
example=wb.worksheets.add('Real Customer Examples')
headers=['Split','Prediction date','Customer ID','Past 30d spending (£)','Past 30d orders','Past 60d spending (£)','Past 60d orders','Past 90d spending (£)','Past 90d orders','Actual next 30d spending (£)']
vals=[headers]
for d in samples:
    vals.append([d['split'],d['prediction_date'],d['customer_id'],float(d['historical_spend_gbp_30d']),int(d['orders_30d']),float(d['historical_spend_gbp_60d']),int(d['orders_60d']),float(d['historical_spend_gbp_90d']),int(d['orders_90d']),float(d['future_30d_spending_gbp'])])
last=len(vals)
example.get_range(f'A1:J{last}').values=vals
example.get_range('A1:J1').format={'fill':'#0F766E','font':{'bold':True,'color':'#FFFFFF'},'row_height':36,'wrap_text':True}
example.get_range(f'A2:C{last}').format.column_width=16
example.get_range(f'D1:J{last}').format.column_width=23
for column in ['D','F','H','J']:example.get_range(f'{column}2:{column}{last}').set_number_format('#,##0.00')
example.freeze_panes.freeze_rows(1)
help_sheet=wb.worksheets.add('How to read')
notes=[['Label','Meaning'],['Prediction date','The date as of which the customer spending forecast would be made'],['Past 30, 60 and 90 days','Three competing lengths of historical spending data'],['Orders','Number of unique invoices in that historical window'],['Recency','Days since the most recent observed purchase'],['Distinct products','Number of different stock codes in the historical window'],['Actual next 30d spending','Correct known answer used to evaluate future predictions'],['Gross positive sales','Excludes returns; not profit, net revenue or incremental advertising lift'],['To avoid data leakage','Never use actual future 30d spending as an ML input feature']]
help_sheet.get_range(f'A1:B{len(notes)}').values=notes
help_sheet.get_range('A1:B1').format={'fill':'#0F766E','font':{'bold':True,'color':'#FFFFFF'}}
help_sheet.get_range(f'A1:A{len(notes)}').format.column_width=31
help_sheet.get_range(f'B1:B{len(notes)}').format.column_width=94
help_sheet.get_range(f'B2:B{len(notes)}').format.wrap_text=True
help_sheet.get_range(f'A1:B{len(notes)}').format.row_height=33
path=R/'milestone3_customer_profiles_easy_preview.xlsx'
SpreadsheetFile.export_xlsx(wb).save(str(path))
assert path.stat().st_size>1000
print('Excel ready:',path.name,path.stat().st_size,'bytes')
print(wb.inspect({'kind':'table','range':'Overview!A3:B8','include':'values','table_max_rows':6,'table_max_cols':2}).ndjson[:500])
