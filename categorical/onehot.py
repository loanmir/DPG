"""One-hot encoding facts shared by the categorical pipeline.

Which columns are categorical is decided exactly the way the training
pipeline's ``pd.get_dummies(features, drop_first=False)`` decides it, so
the answer does not depend on the pandas version's text dtype (``object``
under pandas 2, ``str`` under pandas 3).

The map from a one-hot dummy column back to its ``(original column,
category)`` pair is built at encoding time (``build_onehot_map``) and
stored in the BASIC structure JSON as ``onehot_map``. Splitting a dummy
name at its last ``_`` is ambiguous (``job_self_employed``, ``has_loan``),
so the name-based guess is only a fallback for JSONs without the map.
"""
from __future__ import annotations

from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import pandas as pd


def categorical_columns(df: pd.DataFrame) -> List[str]:
    """Columns of ``df`` that ``pd.get_dummies`` one-hot encodes, in column order.

    Raises ``ValueError`` when a dummy name produced by one column equals
    the name of another original column: the encoded frame would then hold
    two columns with the same name.
    """
    # Which columns get encoded depends on the dtype only, so one row is
    # enough (and avoids encoding a 381k-row dataset to read names). Each
    # column is checked on its own, so a dummy of another column can never
    # hide that a column was encoded.
    head = df.head(1)
    categorical = [
        c for c in df.columns
        if list(pd.get_dummies(head[[c]], drop_first=False).columns) != [c]
    ]
    if not categorical:
        return []
    dummies = set(pd.get_dummies(head[categorical], drop_first=False).columns)
    clashes = sorted(str(c) for c in dummies & set(df.columns))
    if clashes:
        raise ValueError(f"one-hot dummy name(s) equal to an original column name: {clashes}")
    return categorical


# How a missing categorical value is written inside an ``IN``/``NOT IN``
# label. It is what ``str()`` of a missing value read by ``pd.read_csv``
# (a float NaN) gives, which is how ``dpg_equivalence`` evaluates labels.
MISSING_CATEGORY = "nan"


def missing_value_columns(features: pd.DataFrame) -> List[str]:
    """Categorical columns of ``features`` with at least one missing value.

    ``pd.get_dummies(drop_first=False)`` encodes a missing value as all-zero
    dummies: for such a row every ``base NOT IN {...}`` holds and every
    ``base IN {...}`` fails. Rewriting one operator into the other is only
    exact if the missing value is counted as one more category of ``base``
    (``MISSING_CATEGORY``), so the pipeline records these columns in the
    structure JSON as ``missing_categorical``.
    """
    return [c for c in categorical_columns(features) if features[c].isna().any()]


def build_onehot_map(features: pd.DataFrame) -> Dict[str, Tuple[str, str]]:
    """``{dummy column: (original column, category)}`` for every column that
    ``pd.get_dummies(features, drop_first=False)`` creates. The original column
    is known here, so the split never has to be guessed from the name."""
    categorical = categorical_columns(features)
    numeric = set(features.columns) - set(categorical)
    onehot_map: Dict[str, Tuple[str, str]] = {}
    for col in categorical:
        prefix = f"{col}_"
        # ``get_dummies`` itself (not ``f"{col}_{value}"``) guarantees the
        # spelling of the training columns, including how non-string
        # values are printed. Unique rows give the same columns, faster.
        unique_rows = features[[col]].drop_duplicates()
        for dummy in pd.get_dummies(unique_rows, drop_first=False).columns:
            dummy = str(dummy)
            if not dummy.startswith(prefix):
                raise ValueError(f"dummy {dummy!r} does not start with {prefix!r}")
            category = dummy[len(prefix):]
            if dummy in onehot_map:
                raise ValueError(
                    f"one-hot name collision: {dummy!r} is produced by both "
                    f"{onehot_map[dummy]} and {(col, category)}"
                )
            if dummy in numeric:
                raise ValueError(
                    f"one-hot name collision: {dummy!r} (from {(col, category)}) "
                    f"is also a numeric column"
                )
            if any(ch in category for ch in ",{}"):
                print(
                    f"[warn] category {category!r} of column {col!r} contains ',', '{{' "
                    f"or '}}'; the 'base IN {{a, b}}' label grammar cannot represent it"
                )
            onehot_map[dummy] = (str(col), category)
    return onehot_map


def _guess_split(col: str) -> Optional[Tuple[str, str]]:
    """Legacy guess: split ``person_home_ownership_OWN`` at the last ``_``
    into ``('person_home_ownership', 'OWN')``. ``None`` when there is no
    underscore or the trailing token is numeric (e.g. ``x_1``)."""
    if "_" not in col:
        return None
    base, category = col.rsplit("_", 1)
    base = base.strip()
    category = category.strip()
    if not base or not category:
        return None
    try:
        float(category)
        return None
    except ValueError:
        pass
    return base, category


def split_one_hot_column(col: str, onehot_map: Optional[Mapping[str, Sequence[str]]] = None
                         ) -> Optional[Tuple[str, str]]:
    """``(original column, category)`` of a one-hot dummy, ``None`` for a numeric column.
    Uses ``onehot_map`` when given; otherwise falls back to the legacy guess
    (split at the last ``_``, numeric last token = not one-hot)."""
    if onehot_map is None:
        return _guess_split(col)
    entry = onehot_map.get(col)
    # JSON stores the map values as 2-element lists.
    return tuple(entry) if entry is not None else None
