# pit-stock-llm

## Setup

1. Clone the repo
2. Recreate the `data/` folder (not tracked by git) with the structure below
3. Run `python pre_process.py` to generate `data/merged_data.csv`

## Data structure

```
data/
├── Predictors/
│   ├── sm-calls_with_connectors.parquet   # Earnings call transcripts
│   ├── 10K_fillings.parquet               # SEC 10-K filings
└── Targets/
    ├── monthly_crsp.csv                   # Monthly CRSP stock returns (required)
```

`pre_process.py` uses `data/Predictors/sm-calls_with_connectors.parquet` and `data/Targets/monthly_crsp.csv` as inputs and outputs `data/merged_data.csv`.
