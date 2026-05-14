import pandas as pd

earnings_call = pd.read_parquet('data/Predictors/sm-calls_with_connectors.parquet')
earnings_call['date'] = pd.to_datetime(earnings_call['mostimportantdateutc'], format='%Y-%m-%d').dt.to_period("M")
earnings_call = earnings_call.sort_values(by=['permno', 'date'])

target = pd.read_csv('data/Targets/monthly_crsp.csv')
target = target.rename(columns={'PERMNO': 'permno'})
target['date'] = pd.to_datetime(target['MthCalDt'], format='%Y-%m-%d').dt.to_period("M")
target = target[['permno', 'date', 'MthRet', 'SICCD']]
target = target.sort_values(by=['permno', 'date'])
target['ret_3M_shifted'] = target.groupby('permno')['MthRet'].shift(-4)

merged = pd.merge(earnings_call, target, on=['permno', 'date'], how='left')
df = merged.drop(columns=['mostimportantdateutc', 'mostimportanttimeutc', 'MthRet'])
df.to_csv('data/merged_data.csv', index=False)

print(f"Saved {len(df)} rows to data/merged_data.csv")
