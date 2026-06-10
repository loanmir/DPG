"""
create_dataset.py
=================

Equivalent of `examples/create_dataset.ipynb`.

Steps performed:
1. Load `DPG/datasets/credit_card_approval/loan_data.csv`.
2. Print the dataframe shape, columns, dtypes, missing values and cardinality.
3. Classify every column into a semantic type
   (numerical_continuous, numerical_discrete, categorical,
   high_cardinality_categorical, binary, datetime, constant)
   using pandas dtypes + cardinality heuristics.
4. Print the feature-type distribution and a stats table for the
   high-cardinality numerical columns.
5. Build a small "toy" dataset by sampling a configurable subset of
   features (2 categorical + 1 numerical + 1 target by default) and
   save it to `datasets/dummy_dataset/dummy_dataset.csv`.

Run with:
    python examples/create_dataset.py
"""

from __future__ import annotations

import os

import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CSV_PATH = "/root/gitgud/temp/DPG/datasets/credit_card_approval/loan_data.csv"

# Features to keep in the toy dataset.  The last entry is treated as the
# target / class column.  Comment/uncomment entries to change the mix.
KEEP_FEATURES = [
    "person_gender",         # categorical (binary)
    "person_home_ownership", # categorical
    # "person_income",       # numerical
    # "loan_amnt",           # numerical
    "person_age",            # numerical
    "loan_status",           # target / class
]

RANDOM_STATE = 42
N_SAMPLES = 30

OUT_DIR = "/root/gitgud/temp/DPG/datasets/dummy_dataset"
OUT_PATH = os.path.join(OUT_DIR, "dummy_dataset.csv")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def classify_feature(series: pd.Series) -> str:
    """Return a semantic type for a pandas Series."""
    n = len(series)
    nunique = series.nunique(dropna=True)
    dtype = series.dtype
    is_numeric = pd.api.types.is_numeric_dtype(dtype)
    is_bool = pd.api.types.is_bool_dtype(dtype)
    is_datetime = pd.api.types.is_datetime64_any_dtype(dtype)

    if is_datetime:
        return "datetime"
    if is_bool or (is_numeric and nunique == 2):
        return "binary"
    if nunique <= 1:
        return "constant"

    # cardinality-relative heuristics
    rel = nunique / max(n, 1)

    if is_numeric:
        if rel > 0.5 and nunique > 50:
            return "numerical_continuous"  # likely id-like or continuous
        return "numerical_discrete"

    # non-numeric
    if nunique == 2:
        return "binary"
    if nunique <= 20 or rel < 0.05:
        return "categorical"
    if rel > 0.5:
        return "high_cardinality_categorical"
    return "categorical"


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    # 1. Load the raw dataset
    df = pd.read_csv(CSV_PATH)
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(df.head())

    # 2. Basic metadata
    print("\ndtypes:")
    print(df.dtypes)

    print("\nMissing values per column:")
    print(df.isna().sum())

    print("\nCardinality per column:")
    print(df.nunique().sort_values())

    # 3. Feature-type classification
    summary = []
    for col in df.columns:
        s = df[col]
        summary.append(
            {
                "feature": col,
                "pandas_dtype": str(s.dtype),
                "n_unique": s.nunique(dropna=True),
                "n_missing": int(s.isna().sum()),
                "example_values": list(s.dropna().unique()[:3]),
                "feature_type": classify_feature(s),
            }
        )

    feature_summary = pd.DataFrame(summary)
    print("\nFeature summary:")
    print(feature_summary)

    # 4. Feature-type distribution
    type_counts = feature_summary["feature_type"].value_counts()
    print("\nFeature-type distribution:")
    print(type_counts)

    # 5. Descriptive stats for high-cardinality numerical columns
    numeric_cols = [
        c for c in df.columns
        if pd.api.types.is_numeric_dtype(df[c]) and df[c].nunique() > 10
    ]
    if numeric_cols:
        desc = df[numeric_cols].describe().T
        print("\nNumerical column statistics:")
        print(desc[["count", "mean", "std", "min", "25%", "50%", "75%", "max"]])

    # 6. Build and persist the toy dataset
    dummy_df = (
        df[KEEP_FEATURES]
        .sample(n=N_SAMPLES, random_state=RANDOM_STATE)
        .reset_index(drop=True)
    )

    os.makedirs(OUT_DIR, exist_ok=True)
    dummy_df.to_csv(OUT_PATH, index=False)

    print(f"\nSaved {len(dummy_df)} rows to: {OUT_PATH}")
    print(dummy_df)


if __name__ == "__main__":
    main()
