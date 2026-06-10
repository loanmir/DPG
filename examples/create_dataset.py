"""
create_dataset.py
=================

Equivalent of `examples/create_dataset.ipynb`, extended to also build a set
of **toy datasets with deterministic, learnable patterns** so that downstream
models can reach high accuracy and the resulting DPG has a small number of
informative nodes.

Pipeline:
1. Load `DPG/datasets/credit_card_approval/loan_data.csv`.
2. Print the dataframe shape, columns, dtypes, missing values, cardinality,
   and a per-column semantic-type classification.
3. Build several toy datasets (one per *recipe*) by:
     a) sampling `N_SAMPLES` rows from the source CSV,
     b) **overwriting the target with a deterministic rule over the chosen
        features** so the class is highly separable (acc >= 0.85 on a
        decision-tree-friendly model).

Each recipe guarantees the requested mix of categorical vs. numerical
features. All toy datasets are written to `datasets/dummy_dataset/`.

Run with:
    python examples/create_dataset.py
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, List, Sequence

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CSV_PATH = "/root/gitgud/temp/DPG/datasets/credit_card_approval/loan_data.csv"

OUT_DIR = "/root/gitgud/temp/DPG/datasets/dummy_dataset"

RANDOM_STATE = 42
N_SAMPLES = 30


# ---------------------------------------------------------------------------
# Source CSV inspection helpers (kept for parity with the notebook)
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

    rel = nunique / max(n, 1)

    if is_numeric:
        if rel > 0.5 and nunique > 50:
            return "numerical_continuous"
        return "numerical_discrete"

    if nunique == 2:
        return "binary"
    if nunique <= 20 or rel < 0.05:
        return "categorical"
    if rel > 0.5:
        return "high_cardinality_categorical"
    return "categorical"


def print_source_overview(df: pd.DataFrame) -> None:
    """Reproduce the inspect-cells from the notebook."""
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(df.head())

    print("\ndtypes:")
    print(df.dtypes)

    print("\nMissing values per column:")
    print(df.isna().sum())

    print("\nCardinality per column:")
    print(df.nunique().sort_values())

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
    print("\nFeature-type distribution:")
    print(feature_summary["feature_type"].value_counts())

    numeric_cols = [
        c for c in df.columns
        if pd.api.types.is_numeric_dtype(df[c]) and df[c].nunique() > 10
    ]
    if numeric_cols:
        desc = df[numeric_cols].describe().T
        print("\nNumerical column statistics:")
        print(desc[["count", "mean", "std", "min", "25%", "50%", "75%", "max"]])


# ---------------------------------------------------------------------------
# Toy dataset recipes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Recipe:
    """A toy-dataset recipe."""
    name: str
    feature_cols: List[str]            # feature columns in output order
    target_col: str                    # target column name
    label_fn: Callable[[pd.DataFrame], pd.Series]  # rule producing 0/1 target
    description: str = ""
    n_samples: int = N_SAMPLES
    random_state: int = RANDOM_STATE


# -- Label rules (deterministic, learnable) ---------------------------------

def _label_own_or_mortgage_is_zero(df: pd.DataFrame) -> pd.Series:
    """Class 0 if home_ownership in {OWN, MORTGAGE}, else class 1 (RENT)."""
    return (~df["person_home_ownership"].isin(["OWN", "MORTGAGE"])).astype(int)


def _label_female_is_one(df: pd.DataFrame) -> pd.Series:
    """Class 1 if person_gender == 'female', else class 0."""
    return (df["person_gender"] == "female").astype(int)


def _label_own_zero_else_female(df: pd.DataFrame) -> pd.Series:
    """Class 0 if OWN; else class 1 if female; else class 0."""
    out = np.zeros(len(df), dtype=int)
    out[(df["person_home_ownership"] != "OWN") & (df["person_gender"] == "female")] = 1
    return pd.Series(out, index=df.index)


def _label_cat1_num1(df: pd.DataFrame) -> pd.Series:
    """Class 0 if OWN/MORTGAGE; else class 1 if age < 23; else class 0."""
    out = np.zeros(len(df), dtype=int)
    rent = ~df["person_home_ownership"].isin(["OWN", "MORTGAGE"])
    out[rent & (df["person_age"] < 23)] = 1
    return pd.Series(out, index=df.index)


def _label_cat2_num1(df: pd.DataFrame) -> pd.Series:
    """Class 0 if OWN; class 1 if female OR (RENT & age<23); else class 0."""
    out = np.zeros(len(df), dtype=int)
    cond = (df["person_home_ownership"] == "OWN")
    rent = ~cond
    out[rent & (df["person_gender"] == "female")] = 1
    out[rent & (df["person_gender"] != "female") & (df["person_age"] < 23)] = 1
    return pd.Series(out, index=df.index)


# -- Default recipe list ----------------------------------------------------

DEFAULT_RECIPES: List[Recipe] = [
    Recipe(
        name="cat1_ownership",
        feature_cols=["person_home_ownership"],
        target_col="loan_status",
        label_fn=_label_own_or_mortgage_is_zero,
        description="1 categorical (person_home_ownership) + 0 numerical",
    ),
    Recipe(
        name="cat1_gender",
        feature_cols=["person_gender"],
        target_col="loan_status",
        label_fn=_label_female_is_one,
        description="1 categorical (person_gender) + 0 numerical",
    ),
    Recipe(
        name="cat2_gender_ownership",
        feature_cols=["person_gender", "person_home_ownership"],
        target_col="loan_status",
        label_fn=_label_own_zero_else_female,
        description="2 categoricals + 0 numerical",
    ),
    Recipe(
        name="cat1_num1_ownership_age",
        feature_cols=["person_home_ownership", "person_age"],
        target_col="loan_status",
        label_fn=_label_cat1_num1,
        description="1 categorical (home_ownership) + 1 numerical (age)",
    ),
    Recipe(
        name="cat2_num1_gender_ownership_age",
        feature_cols=["person_gender", "person_home_ownership", "person_age"],
        target_col="loan_status",
        label_fn=_label_cat2_num1,
        description="2 categoricals + 1 numerical",
    ),
]


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------

def build_toy_dataset(
    df: pd.DataFrame,
    recipe: Recipe,
    out_dir: str = OUT_DIR,
) -> str:
    """Sample rows, apply the recipe's label rule, persist CSV, return path."""
    cols_to_sample = list(dict.fromkeys(recipe.feature_cols + [recipe.target_col]))
    cols_to_sample = [c for c in cols_to_sample if c in df.columns]
    sample = (
        df[cols_to_sample]
        .sample(n=recipe.n_samples, random_state=recipe.random_state)
        .reset_index(drop=True)
    )

    # Overwrite the target with the deterministic rule.
    sample[recipe.target_col] = recipe.label_fn(sample).astype(int)

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"toy_{recipe.name}.csv")
    sample.to_csv(out_path, index=False)
    return out_path


def build_all(recipes: Sequence[Recipe] = DEFAULT_RECIPES) -> List[str]:
    df = pd.read_csv(CSV_PATH)
    paths: List[str] = []
    for r in recipes:
        path = build_toy_dataset(df, r)
        print(f"  [{r.name:<32}] -> {path}   ({r.description})")
        paths.append(path)
    return paths


# ---------------------------------------------------------------------------
# Backwards-compatible "original notebook" recipe
# ---------------------------------------------------------------------------

ORIGINAL_KEEP_FEATURES = [
    "person_gender",
    "person_home_ownership",
    "person_age",
    "loan_status",
]


def build_original_dummy(df: pd.DataFrame, out_path: str) -> str:
    """Reproduce the original notebook output (random labels, acc ~0.5)."""
    dummy_df = (
        df[ORIGINAL_KEEP_FEATURES]
        .sample(n=N_SAMPLES, random_state=RANDOM_STATE)
        .reset_index(drop=True)
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    dummy_df.to_csv(out_path, index=False)
    return out_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    df = pd.read_csv(CSV_PATH)
    print_source_overview(df)

    # Reproduce the original notebook-style dataset for parity.
    orig_path = os.path.join(OUT_DIR, "dummy_dataset.csv")
    build_original_dummy(df, orig_path)
    print(f"\nSaved {N_SAMPLES} rows to: {orig_path}  (original notebook parity)")
    print(pd.read_csv(orig_path))

    # Build the new high-accuracy toy datasets.
    print("\nBuilding high-accuracy toy datasets:")
    for path in build_all():
        toy = pd.read_csv(path)
        print(f"\n--- {os.path.basename(path)} ---")
        print(toy)


if __name__ == "__main__":
    main()
