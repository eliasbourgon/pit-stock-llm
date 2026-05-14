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
df = merged[['date','text', 'SICCD', 'ret_3M_shifted']]

prefix_to_industry = {
    "01": "Agriculture, Forestry & Fishing",
    "02": "Agriculture, Forestry & Fishing",
    "07": "Agriculture, Forestry & Fishing",
    "08": "Agriculture, Forestry & Fishing",
    "09": "Agriculture, Forestry & Fishing",
    "10": "Metal Mining",
    "11": "Coal Mining",
    "12": "Coal Mining",
    "13": "Oil & Gas",
    "14": "Mining & Quarrying (Non-Metal)",
    "15": "Construction",
    "16": "Construction",
    "17": "Construction",
    "20": "Food & Beverage Manufacturing",
    "21": "Tobacco Manufacturing",
    "22": "Textile Manufacturing",
    "23": "Apparel & Accessories Manufacturing",
    "24": "Lumber & Wood Products",
    "25": "Furniture & Fixtures Manufacturing",
    "26": "Paper & Allied Products",
    "27": "Printing & Publishing",
    "28": "Chemicals & Allied Products",
    "29": "Petroleum Refining",
    "30": "Rubber & Plastics Manufacturing",
    "31": "Leather & Leather Products",
    "32": "Stone, Clay & Glass Products",
    "33": "Primary Metal Manufacturing",
    "34": "Fabricated Metal Products",
    "35": "Industrial Machinery & Equipment",
    "36": "Electronic & Electrical Equipment",
    "37": "Transportation Equipment Manufacturing",
    "38": "Instruments & Related Products",
    "39": "Miscellaneous Manufacturing",
    "40": "Railroad Transportation",
    "41": "Local & Urban Transit",
    "42": "Trucking & Warehousing",
    "44": "Water Transportation",
    "45": "Air Transportation",
    "46": "Pipelines",
    "47": "Transportation Services",
    "48": "Communications",
    "49": "Electric, Gas & Sanitary Services",
    "50": "Wholesale Trade - Durable Goods",
    "51": "Wholesale Trade - Non-Durable Goods",
    "52": "Retail - Building Materials & Hardware",
    "53": "Retail - General Merchandise",
    "54": "Retail - Food Stores",
    "55": "Retail - Auto Dealers & Gas Stations",
    "56": "Retail - Apparel & Accessory Stores",
    "57": "Retail - Home Furniture & Equipment",
    "58": "Retail - Eating & Drinking Places",
    "59": "Retail - Miscellaneous",
    "60": "Banking",
    "61": "Credit & Lending",
    "62": "Securities & Commodity Brokers",
    "63": "Insurance",
    "64": "Insurance",
    "65": "Real Estate",
    "67": "Holding & Investment Companies",
    "70": "Hotels & Lodging",
    "72": "Personal Services",
    "73": "Business Services",
    "75": "Auto Repair & Services",
    "76": "Miscellaneous Repair Services",
    "78": "Motion Picture & Entertainment",
    "79": "Amusement & Recreation",
    "80": "Health Services",
    "81": "Legal Services",
    "82": "Educational Services",
    "83": "Social Services",
    "84": "Museums & Art Galleries",
    "86": "Membership Organizations",
    "87": "Engineering & Management Services",
    "89": "Services - Miscellaneous",
    "91": "Public Administration",
    "92": "Public Administration",
    "93": "Public Administration",
    "94": "Public Administration",
    "95": "Public Administration",
    "96": "Public Administration",
    "97": "Public Administration",
    "99": "Public Administration",
}

def map_sic(sic_code):
    prefix = str(int(sic_code)).zfill(4)[:2]
    return prefix_to_industry.get(prefix, "Other / Unclassified")


df["industry"] = df["SICCD"].apply(map_sic)
df= df.drop(columns=['SICCD'])


df.to_parquet('data/merged_data.parquet', index=False)

print(f"Saved {len(df)} rows to data/merged_data.csv")
