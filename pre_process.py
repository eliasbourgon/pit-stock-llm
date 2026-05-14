"""
pre_process.py
──────────────
Merges summarized earnings calls with CRSP returns and maps SIC codes to industries.

Input:
  --input   : .parquet from preprocess_summarize.py  (must contain ec_summary or text, permno, mostimportantdateutc)
  --returns : monthly_crsp.csv                        (columns: PERMNO, MthCalDt, MthRet, SICCD)
  --output  : merged_data.parquet

Output columns: [date, text, industry, ret_3M_shifted]
  text = ec_summary if present, else original text column
"""

import argparse

import pandas as pd


PREFIX_TO_INDUSTRY = {
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


def map_sic(sic_code) -> str:
    prefix = str(int(sic_code)).zfill(4)[:2]
    return PREFIX_TO_INDUSTRY.get(prefix, "Other / Unclassified")


def main(args: argparse.Namespace) -> None:
    # ── Load earnings calls ───────────────────────────────────────────────────
    ec = pd.read_parquet(args.input)
    ec["date"] = pd.to_datetime(ec["mostimportantdateutc"], format="%Y-%m-%d").dt.to_period("M")
    ec = ec.sort_values(by=["permno", "date"])

    # Use ec_summary (from preprocess_summarize) if available, else fall back to raw text
    if "ec_summary" in ec.columns:
        ec = ec.drop(columns=["text"], errors="ignore")
        ec = ec.rename(columns={"ec_summary": "text"})
        print(f"Using 'ec_summary' column as text ({ec['text'].notna().sum()} non-null)")
    elif "text" not in ec.columns:
        raise ValueError("Input parquet must have either 'ec_summary' or 'text' column.")
    else:
        print("Using raw 'text' column (no ec_summary found)")

    # ── Load returns ──────────────────────────────────────────────────────────
    target = pd.read_csv(args.returns)
    target = target.rename(columns={"PERMNO": "permno"})
    target["date"] = pd.to_datetime(target["MthCalDt"], format="%Y-%m-%d").dt.to_period("M")
    target = target[["permno", "date", "MthRet", "SICCD"]].sort_values(by=["permno", "date"])
    target["ret_3M_shifted"] = target.groupby("permno")["MthRet"].shift(-4)

    # ── Merge ─────────────────────────────────────────────────────────────────
    merged = pd.merge(ec, target, on=["permno", "date"], how="left")
    merged["industry"] = merged["SICCD"].apply(map_sic)

    df = merged[["date", "text", "industry", "ret_3M_shifted"]]

    # ── Save ──────────────────────────────────────────────────────────────────
    df.to_parquet(args.output, index=False)
    print(f"Saved {len(df)} rows to {args.output}")
    print(f"  ret_3M_shifted non-null: {df['ret_3M_shifted'].notna().sum()} / {len(df)}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Merge earnings calls with CRSP returns")
    p.add_argument("--input",   required=True, help="Summarized .parquet (from preprocess_summarize.py)")
    p.add_argument("--returns", required=True, help="Path to monthly_crsp.csv")
    p.add_argument("--output",  default="data/merged_data.parquet")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
