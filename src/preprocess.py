"""
Logistics Data Preprocessing Pipeline
======================================
Simulates the data collection, cleaning, and preprocessing workflow for a
hyperlocal logistics/shipment dataset (shipment records with carrier,
route, weight, transit time, and delivery cost fields).

Pipeline stages:
    1. Load & inspect raw data
    2. Standardize formatting (categorical text, dtypes)
    3. Remove duplicate records
    4. Handle missing values
    5. Detect and treat outliers
    6. Normalize numeric features
    7. Export cleaned dataset + summary report

Run:
    python src/preprocess.py
"""

import pandas as pd
import numpy as np
from pathlib import Path

RAW_PATH = Path("data/raw_logistics_shipments.csv")
CLEAN_PATH = Path("data/cleaned_logistics_shipments.csv")
SUMMARY_PATH = Path("data/cleaning_summary.txt")

NUMERIC_COLS = ["distance_km", "weight_kg", "transit_time_hours", "delivery_cost_inr"]


def load_data(path: Path) -> pd.DataFrame:
    """Step 1: Load raw CSV and report basic shape/dtype info."""
    df = pd.read_csv(path)
    print(f"[Load] {len(df)} rows, {len(df.columns)} columns loaded from {path.name}")
    return df


def standardize_formatting(df: pd.DataFrame) -> pd.DataFrame:
    """Step 2: Normalize text casing/whitespace in categorical columns."""
    cat_cols = ["carrier", "origin", "destination", "on_time_flag"]
    for col in cat_cols:
        df[col] = df[col].astype(str).str.strip().str.title()
    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Step 3: Drop exact duplicate shipment records."""
    before = len(df)
    df = df.drop_duplicates(subset="shipment_id", keep="first")
    print(f"[Duplicates] Removed {before - len(df)} duplicate rows")
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Step 4: Impute missing numeric values using median imputation
    (robust to skew/outliers compared to mean), grouped by carrier
    where possible so imputed values reflect that carrier's typical
    performance rather than a global average.
    """
    missing_before = df[NUMERIC_COLS].isna().sum().sum()
    for col in NUMERIC_COLS:
        df[col] = df.groupby("carrier")[col].transform(
            lambda s: s.fillna(s.median())
        )
        # fallback for any carrier-group that was entirely NaN
        df[col] = df[col].fillna(df[col].median())
    missing_after = df[NUMERIC_COLS].isna().sum().sum()
    print(f"[Missing Values] Imputed {missing_before - missing_after} missing entries "
          f"(median-by-carrier strategy)")
    return df


def detect_and_treat_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Step 5: Detect outliers using the IQR method and cap them
    (winsorize) rather than dropping rows, to preserve sample size
    while limiting the influence of extreme values.
    """
    outlier_counts = {}
    for col in NUMERIC_COLS:
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n_outliers = ((df[col] < lower) | (df[col] > upper)).sum()
        outlier_counts[col] = int(n_outliers)
        df[col] = df[col].clip(lower=lower, upper=upper)
    print(f"[Outliers] Detected & capped (IQR method): {outlier_counts}")
    return df


def normalize_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Step 6: Min-max normalize numeric columns to a [0, 1] range so
    features with different units/scales (km, kg, hours, INR)
    contribute comparably to downstream models.
    """
    for col in NUMERIC_COLS:
        col_min, col_max = df[col].min(), df[col].max()
        df[f"{col}_norm"] = (df[col] - col_min) / (col_max - col_min)
    print("[Normalization] Min-max scaled columns added with '_norm' suffix")
    return df


def main():
    df = load_data(RAW_PATH)
    df = standardize_formatting(df)
    df = remove_duplicates(df)
    df = handle_missing_values(df)
    df = detect_and_treat_outliers(df)
    df = normalize_features(df)

    df.to_csv(CLEAN_PATH, index=False)

    summary = (
        f"Cleaned dataset: {len(df)} rows x {len(df.columns)} columns\n"
        f"Numeric columns processed: {NUMERIC_COLS}\n"
        f"Output written to: {CLEAN_PATH}\n"
    )
    SUMMARY_PATH.write_text(summary)
    print("\n[Done] Cleaned dataset saved to", CLEAN_PATH)


if __name__ == "__main__":
    main()
