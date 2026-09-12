# Logistics Data Preprocessing Pipeline

Simulated data collection and preprocessing workflow for a logistics
(shipment tracking) dataset, built with pandas and numpy.

## Structure
```
logistics-data-preprocessing/
├── data/
│   ├── raw_logistics_shipments.csv       # simulated raw dataset (with issues)
│   ├── cleaned_logistics_shipments.csv   # output after pipeline runs
│   └── cleaning_summary.txt              # generated run summary
├── src/
│   └── preprocess.py                     # main preprocessing pipeline
├── docs/
│   └── Logistics_Data_Preprocessing_Report.docx
├── requirements.txt
└── README.md
```

## Dataset
200+ simulated shipment records with fields: `shipment_id`, `carrier`,
`origin`, `destination`, `distance_km`, `weight_kg`,
`transit_time_hours`, `delivery_cost_inr`, `on_time_flag`. Deliberately
injected with missing values, outliers, duplicate rows, and
inconsistent text casing to mimic real-world logistics data (modeled
on public datasets such as DOT/Kaggle shipment and freight records).

## Pipeline stages (`src/preprocess.py`)
1. Load & inspect raw data
2. Standardize categorical text formatting
3. Remove duplicate records
4. Handle missing values (median imputation, grouped by carrier)
5. Detect & treat outliers (IQR method, winsorized/capped)
6. Min-max normalize numeric features
7. Export cleaned dataset + summary

## Usage
```bash
pip install -r requirements.txt
python src/preprocess.py
```

## Report
See `docs/Logistics_Data_Preprocessing_Report.docx` for the full
methodology write-up, rationale for each technique, and reflection on
data quality's impact on logistics analytics.
