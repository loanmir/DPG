"""
gridsearch_dpg_catsim.py
========================

A gridsearch script, sibling to ``gridsearch_dpg.py``, that is tuned for
producing DPG structures with **sequential, non-disjoint categorical
predicates on the same path** -- i.e. the model has to refine the same
categorical feature multiple times along a root-to-leaf path with sets that
*overlap* (e.g. ``cat_feat IN {A}`` followed by ``cat_feat IN {A, B}``)
rather than partitioning it (e.g. ``cat_feat NOT IN {X}`` followed by
``cat_feat IN {Y}``).

Scope
-----
This variant operates **exclusively on the ``toy_chain_*`` datasets** in
``datasets/dummy_dataset/``. Those are the synthetic CSVs specifically
engineered to elicit overlapping same-feature categorical chains. The
curated ``toy_cat*`` list from earlier versions of this script has been
removed: the chain datasets are the single source of truth and they
exercise a richer mix of categorical + numerical feature combinations.

It mirrors ``gridsearch_dpg.py`` in every other respect:

  * Trains a RandomForest on each candidate dataset,
  * Sweeps ``perc_var`` x ``decimal_threshold`` x ``community_threshold``,
  * Saves a per-run subdirectory with the JSON structure, node/edge CSVs,
    class boundaries, communities, plots and a per-run ``config.yaml``,
  * Appends a summary CSV.

Chain recipes (7 total)
-----------------------
Original 3 (kept verbatim):
  * ``chain_intent_a_ab``       - 2-step chain on ``loan_intent`` + age gate
  * ``chain_education_abc``     - 3-step chain on ``person_education`` only
  * ``chain_intent_with_age``   - 3-step chain on ``loan_intent`` + age +
                                  gender gates

New 4 (added in this revision):
  * ``chain_ownership_rent_mortgage_own`` - 3-step chain on
        ``person_home_ownership`` (``{RENT}`` -> ``{RENT, MORTGAGE}`` ->
        ``{RENT, MORTGAGE, OWN}``), gated by ``loan_amnt > 15000``.
  * ``chain_intent_emp_exp``    - 2-step chain on ``loan_intent``
        (``{VENTURE}`` -> ``{VENTURE, EDUCATION}``), gated by
        ``person_emp_exp >= 3``.
  * ``chain_education_emp_exp`` - 3-step chain on ``person_education``
        (``{Master}`` -> ``{Master, Bachelor}`` ->
        ``{Master, Bachelor, High School}``), gated by
        ``person_income < 40000`` and ``loan_int_rate > 12``.
  * ``chain_gender_intent_income`` - 3-step chain on ``loan_intent``
        (``{VENTURE}`` -> ``{VENTURE, EDUCATION}`` ->
        ``{VENTURE, EDUCATION, HOMEIMPROVEMENT}``), gated by
        ``person_income > 60000`` and ``person_gender == female``.

Usage:
    python examples/gridsearch_dpg_catsim.py
    python examples/gridsearch_dpg_catsim.py --generate-chain-datasets
    python examples/gridsearch_dpg_catsim.py \\
        --datasets toy_chain_intent_a_ab.csv,toy_chain_ownership_rent_mortgage_own.csv \\
        --mode random --n-samples 25
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import yaml
from networkx.readwrite import json_graph
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import KFold

# --- Make the project root importable (so `dpg` and `metrics` resolve) -------
SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
sys.path.insert(0, PROJECT_ROOT)

from dpg import DPGExplainer  # noqa: E402
from metrics.graph import GraphMetrics  # noqa: E402


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATA_DIR = os.path.join(PROJECT_ROOT, "datasets", "dummy_dataset")
LOAN_CSV = os.path.join(
    PROJECT_ROOT, "datasets", "credit_card_approval", "loan_data.csv"
)
BASE_CONFIG_PATH = os.path.join(PROJECT_ROOT, "config.yaml")
RESULTS_ROOT = os.path.join(SCRIPT_DIR, "results_cat")
SUMMARY_PATH = os.path.join(RESULTS_ROOT, "_summary.csv")

# Chain-friendly datasets (must live in DATA_DIR and follow the
# ``toy_chain_*.csv`` naming convention). When ``--datasets`` is omitted we
# auto-discover every ``toy_chain_*.csv`` on disk, in sorted order, so newly
# added chain recipes are picked up automatically.
DEFAULT_DATASET_PREFIX = "toy_chain_"


def discover_chain_datasets(data_dir: str = DATA_DIR,
                            prefix: str = DEFAULT_DATASET_PREFIX) -> List[str]:
    """Return the sorted list of ``toy_chain_*.csv`` filenames in ``data_dir``."""
    if not os.path.isdir(data_dir):
        return []
    return sorted(
        f for f in os.listdir(data_dir)
        if f.startswith(prefix) and f.lower().endswith(".csv")
    )

# Chain recipes that this script can synthesize at runtime. Their filenames
# follow the ``toy_chain_<recipe>.csv`` convention and they are emitted into
# ``DATA_DIR``. The 7 recipes below cover a mix of:
#   * pure-categorical chains
#   * cat chain + single numerical gate
#   * cat chain + numerical gate + 2nd categorical gate
CHAIN_RECIPES = [
    # --- Original 3 -------------------------------------------------------
    "chain_intent_a_ab",       # 2-step chain on loan_intent + age gate
    "chain_education_abc",     # 3-step chain on person_education only
    "chain_intent_with_age",   # 3-step chain on loan_intent + age + gender
    # --- New 4 (this revision) -------------------------------------------
    "chain_ownership_rent_mortgage_own",  # 3-step on person_home_ownership
    "chain_intent_emp_exp",     # 2-step on loan_intent + emp_exp gate
    "chain_education_emp_exp",  # 3-step on person_education + income + rate
    "chain_gender_intent_income",  # 3-step on loan_intent + income + gender
]

# Hyper-parameter grid (kept narrow for quick iteration; widen as needed).
DEFAULT_PERC_VARS = [0.01, 0.075]
DEFAULT_DECIMAL_THRESHOLDS = [2, 3]
DEFAULT_COMMUNITY_THRESHOLDS = [0.20, 0.30]

# Random search sampling budget.
DEFAULT_N_RANDOM_SAMPLES = 50

RANDOM_STATE = 27
NUM_TREES = 10
N_SPLITS = 5


# ---------------------------------------------------------------------------
# Chain-friendly dataset synthesis
# ---------------------------------------------------------------------------
#
# These recipes are deliberately crafted so that a decision-tree-friendly
# model must place **non-disjoint** categorical predicates of the same
# feature one after the other on a single root-to-leaf path.  Concretely:
#
#   path: cat_feat == A  ->  cat_feat in {A, B}  ->  class
#
# vs. the "partition" pattern this gridsearch tries to avoid:
#
#   path: cat_feat NOT IN {X}  ->  cat_feat in {Y}  ->  class

CHAIN_N_SAMPLES = 60
CHAIN_RANDOM_STATE = 11


def _label_chain_intent_a_ab(df: pd.DataFrame) -> pd.Series:
    """Class 1 iff loan_intent in {'VENTURE'} OR
                (loan_intent in {'VENTURE', 'EDUCATION'} AND person_age < 25).
    Encourages: ``loan_intent_VENTURE == 1`` then ``loan_intent_EDUCATION`` on
    the same path -- overlapping predicates.
    """
    venture = (df["loan_intent"] == "VENTURE")
    edu = (df["loan_intent"] == "EDUCATION")
    young = (df["person_age"] < 25)
    return (venture | (venture & edu & young)).astype(int)


def _label_chain_education_abc(df: pd.DataFrame) -> pd.Series:
    """Class 1 iff person_education == 'Master'
                OR person_education in {'Master', 'Bachelor'}
                OR person_education in {'Master', 'Bachelor', 'Associate'}.
    Three overlapping predicates of the same feature on the same path.
    """
    master = (df["person_education"] == "Master")
    bach = (df["person_education"] == "Bachelor")
    assoc = (df["person_education"] == "Associate")
    return (master | (master & bach) | (master & bach & assoc)).astype(int)


def _label_chain_intent_with_age(df: pd.DataFrame) -> pd.Series:
    """Class 1 iff loan_intent in {'VENTURE'}
                OR (loan_intent in {'VENTURE', 'EDUCATION'} AND age < 25)
                OR (loan_intent in {'VENTURE', 'EDUCATION', 'HOMEIMPROVEMENT'}
                    AND age < 25 AND gender == 'female').
    Three same-feature predicates + a numerical gate + a 2nd cat gate.
    """
    venture = (df["loan_intent"] == "VENTURE")
    edu = (df["loan_intent"] == "EDUCATION")
    home = (df["loan_intent"] == "HOMEIMPROVEMENT")
    young = (df["person_age"] < 25)
    female = (df["person_gender"] == "female")
    return (venture | (venture & edu & young) | (venture & edu & home & young & female)).astype(int)


def _label_chain_ownership_rent_mortgage_own(df: pd.DataFrame) -> pd.Series:
    """Class 1 iff person_home_ownership == 'RENT'
                OR (person_home_ownership in {'RENT', 'MORTGAGE'}
                    AND loan_amnt > 15000)
                OR (person_home_ownership in {'RENT', 'MORTGAGE', 'OWN'}
                    AND loan_amnt > 15000).
    Three same-feature predicates (RENT subset chain) gated by a numerical.
    The numerical gate must fire on the same path -> it appears as a
    sibling/refinement node right after the categorical chain.
    """
    rent = (df["person_home_ownership"] == "RENT")
    mort = (df["person_home_ownership"] == "MORTGAGE")
    own = (df["person_home_ownership"] == "OWN")
    big_loan = (df["loan_amnt"] > 15000)
    return (
        rent | ((rent | mort) & big_loan) | ((rent | mort | own) & big_loan)
    ).astype(int)


def _label_chain_intent_emp_exp(df: pd.DataFrame) -> pd.Series:
    """Class 1 iff loan_intent == 'VENTURE'
                OR (loan_intent in {'VENTURE', 'EDUCATION'}
                    AND person_emp_exp >= 3).
    Two same-feature predicates gated by a numerical.
    """
    venture = (df["loan_intent"] == "VENTURE")
    edu = (df["loan_intent"] == "EDUCATION")
    experienced = (df["person_emp_exp"] >= 3)
    return (venture | (venture & edu & experienced)).astype(int)


def _label_chain_education_emp_exp(df: pd.DataFrame) -> pd.Series:
    """Class 1 iff person_education == 'Master'
                OR (person_education in {'Master', 'Bachelor'}
                    AND person_income < 40000)
                OR (person_education in {'Master', 'Bachelor', 'High School'}
                    AND person_income < 40000 AND loan_int_rate > 12).
    Three same-feature predicates gated by one or two numericals.
    Encourages a long chain of same-cat refinements plus numerical
    refinements.
    """
    master = (df["person_education"] == "Master")
    bach = (df["person_education"] == "Bachelor")
    hs = (df["person_education"] == "High School")
    low_income = (df["person_income"] < 40000)
    high_rate = (df["loan_int_rate"] > 12)
    return (
        master
        | ((master | bach) & low_income)
        | ((master | bach | hs) & low_income & high_rate)
    ).astype(int)


def _label_chain_gender_intent_income(df: pd.DataFrame) -> pd.Series:
    """Class 1 iff loan_intent == 'VENTURE'
                OR (loan_intent in {'VENTURE', 'EDUCATION'}
                    AND person_income > 60000)
                OR (loan_intent in {'VENTURE', 'EDUCATION', 'HOMEIMPROVEMENT'}
                    AND person_income > 60000
                    AND person_gender == 'female').
    Three same-feature predicates gated by a numerical AND a 2nd cat.
    """
    venture = (df["loan_intent"] == "VENTURE")
    edu = (df["loan_intent"] == "EDUCATION")
    home = (df["loan_intent"] == "HOMEIMPROVEMENT")
    high_income = (df["person_income"] > 60000)
    female = (df["person_gender"] == "female")
    return (
        venture
        | ((venture | edu) & high_income)
        | ((venture | edu | home) & high_income & female)
    ).astype(int)


CHAIN_LABEL_FNS = {
    "chain_intent_a_ab": _label_chain_intent_a_ab,
    "chain_education_abc": _label_chain_education_abc,
    "chain_intent_with_age": _label_chain_intent_with_age,
    "chain_ownership_rent_mortgage_own": _label_chain_ownership_rent_mortgage_own,
    "chain_intent_emp_exp": _label_chain_intent_emp_exp,
    "chain_education_emp_exp": _label_chain_education_emp_exp,
    "chain_gender_intent_income": _label_chain_gender_intent_income,
}

# Default feature subsets per chain recipe.
CHAIN_FEATURES = {
    "chain_intent_a_ab": ["loan_intent", "person_age"],
    "chain_education_abc": ["person_education"],
    "chain_intent_with_age": ["loan_intent", "person_age", "person_gender"],
    "chain_ownership_rent_mortgage_own": ["person_home_ownership", "loan_amnt"],
    "chain_intent_emp_exp": ["loan_intent", "person_emp_exp"],
    "chain_education_emp_exp": ["person_education", "person_income", "loan_int_rate"],
    "chain_gender_intent_income": ["loan_intent", "person_income", "person_gender"],
}


def generate_chain_datasets(force: bool = False) -> List[str]:
    """Build the chain-friendly CSVs into DATA_DIR. Returns written paths."""
    if not os.path.exists(LOAN_CSV):
        print(f"[chain-gen] WARNING: source CSV not found at {LOAN_CSV}; "
              f"skipping chain dataset generation.")
        return []

    os.makedirs(DATA_DIR, exist_ok=True)
    src = pd.read_csv(LOAN_CSV)
    written: List[str] = []
    for recipe in CHAIN_RECIPES:
        out_path = os.path.join(DATA_DIR, f"toy_{recipe}.csv")
        if os.path.exists(out_path) and not force:
            written.append(out_path)
            continue
        feat_cols = CHAIN_FEATURES[recipe]
        needed = list(dict.fromkeys(feat_cols + ["loan_status"]))
        sample = (
            src[needed]
            .sample(n=CHAIN_N_SAMPLES, random_state=CHAIN_RANDOM_STATE)
            .reset_index(drop=True)
        )
        sample["loan_status"] = CHAIN_LABEL_FNS[recipe](sample).astype(int)
        sample.to_csv(out_path, index=False)
        written.append(out_path)
        print(f"[chain-gen] wrote {out_path}")
    return written


# ---------------------------------------------------------------------------
# Helpers (mirrors gridsearch_dpg.py)
# ---------------------------------------------------------------------------

def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _short_feature_tag(features: Sequence[str]) -> str:
    """Compact, filename-safe representation of a feature subset."""
    cleaned = [str(f).replace(" ", "_").replace(os.sep, "_") for f in features]
    head = cleaned[:3]
    extra = len(cleaned) - len(head)
    base = "_".join(head) if head else "none"
    return f"{base}__{extra}more" if extra > 0 else base


def _with_csv(name: str) -> str:
    return name if name.lower().endswith(".csv") else f"{name}.csv"


def load_dataset(dataset_name: str):
    """Load a CSV from DATA_DIR, return (X, y, feature_names, original_names)."""
    path = os.path.join(DATA_DIR, _with_csv(dataset_name))
    df = pd.read_csv(path)
    features = df.iloc[:, :-1]
    target_col = df.columns[-1]
    labels = df[target_col]

    original_feature_names = list(features.columns)
    features_enc = pd.get_dummies(features, drop_first=False)
    features_enc = features_enc.replace([np.inf, -np.inf], np.nan).fillna(features_enc.mean())
    features_enc = np.round(features_enc, 3)
    feature_names = list(features_enc.columns)
    return features_enc, labels, feature_names, original_feature_names


def train_cv(model, X, y, n_splits: int = N_SPLITS, random_state: int = RANDOM_STATE):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    accs, f1s = [], []
    last_train = None
    for tr, te in kf.split(X):
        X_tr, X_te = X.iloc[tr], X.iloc[te]
        y_tr, y_te = y.iloc[tr], y.iloc[te]
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_te)
        accs.append(accuracy_score(y_te, y_pred))
        f1s.append(f1_score(y_te, y_pred, average="weighted"))
        last_train = (X_tr, y_tr)
    return float(np.mean(accs)), float(np.mean(f1s)), last_train


def write_run_config(
    out_dir: str,
    perc_var: float,
    decimal_threshold: int,
) -> str:
    """Copy BASE_CONFIG_PATH into out_dir/config.yaml with overridden values."""
    with open(BASE_CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg.setdefault("dpg", {}).setdefault("default", {})
    cfg["dpg"]["default"]["perc_var"] = float(perc_var)
    cfg["dpg"]["default"]["decimal_threshold"] = int(decimal_threshold)
    cfg["dpg"]["default"]["n_jobs"] = 1
    cfg_path = os.path.join(out_dir, "config.yaml")
    with open(cfg_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return cfg_path


def save_dpg_structure_json(explanation, out_path: str, run_id: str,
                            feature_names, target_names) -> None:
    graph_data = json_graph.node_link_data(explanation.graph)
    labeled_nodes = [{"id": nid, "label": lab} for nid, lab in explanation.nodes]
    payload = {
        "run_id": run_id,
        "feature_names": [str(n) for n in feature_names],
        "target_names": [str(n) for n in target_names],
        "community_threshold": explanation.community_threshold,
        "nodes": labeled_nodes,
        "graph": graph_data,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=_json_default)


# ---------------------------------------------------------------------------
# Core run (mirrors gridsearch_dpg.py)
# ---------------------------------------------------------------------------

@dataclass
class RunResult:
    run_id: str
    dataset: str
    perc_var: float
    decimal_threshold: int
    community_threshold: float
    accuracy: float
    f1: float
    n_nodes: int
    n_edges: int
    n_communities: int
    out_dir: str
    n_features_used: int
    features_used: str


def run_one(
    dataset_name: str,
    perc_var: float,
    decimal_threshold: int,
    community_threshold: float,
    *,
    feature_subset_rng: Optional[random.Random] = None,
    feature_frac_min: float = 0.5,
    feature_frac_max: float = 1.0,
    feature_min: int = 1,
) -> RunResult:
    short_ds = dataset_name.replace("toy_", "").replace(".csv", "")
    run_id = (
        f"ds={short_ds}_pv={perc_var}_dt={decimal_threshold}_ct={community_threshold}"
    )
    out_dir = os.path.join(RESULTS_ROOT, run_id)
    os.makedirs(out_dir, exist_ok=True)

    cfg_path = write_run_config(out_dir, perc_var, decimal_threshold)

    X, y, feature_names, original_feature_names = load_dataset(dataset_name)
    all_features = list(feature_names)
    chosen_features: List[str] = all_features
    if feature_subset_rng is not None and len(all_features) > feature_min:
        frac_lo = max(0.0, min(1.0, feature_frac_min))
        frac_hi = max(frac_lo, min(1.0, feature_frac_max))
        frac = feature_subset_rng.uniform(frac_lo, frac_hi)
        k = max(feature_min, int(round(frac * len(all_features))))
        k = min(k, len(all_features))
        chosen_features = feature_subset_rng.sample(all_features, k=k)
        short_feats = _short_feature_tag(chosen_features)
        run_id = (
            f"ds={short_ds}_pv={perc_var}_dt={decimal_threshold}"
            f"_ct={community_threshold}_feats={short_feats}"
        )
        out_dir = os.path.join(RESULTS_ROOT, run_id)
        os.makedirs(out_dir, exist_ok=True)
        cfg_path = write_run_config(out_dir, perc_var, decimal_threshold)
        X = X[chosen_features]
        feature_names = chosen_features
    metric_suffix = f"acc_{round(0, 2)}"  # filled below

    model = RandomForestClassifier(n_estimators=NUM_TREES, random_state=RANDOM_STATE)
    acc, f1, (X_train, y_train) = train_cv(model, X, y)
    metric_suffix = f"acc_{round(acc, 2)}"

    target_names = np.unique(y).astype(str).tolist()
    explainer = DPGExplainer(
        model=model,
        feature_names=feature_names,
        target_names=target_names,
        config_file=cfg_path,
    )
    explanation = explainer.explain_global(
        X_train.values,
        communities=True,
        community_threshold=community_threshold,
    )

    save_dpg_structure_json(
        explanation,
        os.path.join(out_dir, f"{run_id}_dpg_structure.json"),
        run_id,
        feature_names,
        target_names,
    )
    with open(os.path.join(out_dir, f"{run_id}_dpg_class_boundaries.txt"), "w") as f:
        for key, value in explanation.class_boundaries.items():
            f.write(f"{key}: {value}\n")
    explanation.node_metrics.to_csv(
        os.path.join(out_dir, f"{run_id}_node_metrics.csv"), encoding="utf-8"
    )
    explanation.edge_metrics.to_csv(
        os.path.join(out_dir, f"{run_id}_edge_metrics.csv"), encoding="utf-8"
    )
    if explanation.communities is not None:
        GraphMetrics.communities_to_csv(
            explanation.communities,
            os.path.join(out_dir, f"{run_id}_dpg_communities.txt"),
        )

    if chosen_features and chosen_features != all_features:
        with open(os.path.join(out_dir, f"{run_id}_features.txt"), "w") as f:
            for feat in chosen_features:
                f.write(f"{feat}\n")

    explainer.plot(
        run_id,
        explanation=explanation,
        save_dir=out_dir,
        class_flag=False,
        export_pdf=True,
        show=False,
    )
    explainer.plot_communities(
        run_id,
        explanation=explanation,
        save_dir=out_dir,
        class_flag=True,
        export_pdf=True,
        show=False,
    )

    n_nodes = len(explanation.node_metrics)
    n_edges = len(explanation.edge_metrics)
    n_comms = 0
    if explanation.communities and "Clusters" in explanation.communities:
        n_comms = sum(
            1 for k, v in explanation.communities["Clusters"].items()
            if k != "Ambiguous" and v
        )

    return RunResult(
        run_id=run_id,
        dataset=dataset_name,
        perc_var=perc_var,
        decimal_threshold=decimal_threshold,
        community_threshold=community_threshold,
        accuracy=acc,
        f1=f1,
        n_nodes=n_nodes,
        n_edges=n_edges,
        n_communities=n_comms,
        out_dir=out_dir,
        n_features_used=len(chosen_features),
        features_used=";".join(chosen_features) if chosen_features != all_features else "",
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "DPG gridsearch for chain-style (overlapping) categorical "
            "predicates. Operates exclusively on the toy_chain_*.csv "
            "datasets in datasets/dummy_dataset/."
        )
    )
    parser.add_argument(
        "--datasets",
        default=None,
        help=(
            "Comma-separated ``toy_chain_*.csv`` filenames inside "
            "``datasets/dummy_dataset/``. If omitted, every "
            "``toy_chain_*.csv`` currently on disk is auto-discovered."
        ),
    )
    parser.add_argument(
        "--generate-chain-datasets",
        action="store_true",
        help=(
            "If set, synthesize (or refresh) the chain-friendly toy CSVs "
            "into datasets/dummy_dataset/ before sweeping. Idempotent unless "
            "--force-chain-regen is also given."
        ),
    )
    parser.add_argument(
        "--force-chain-regen",
        action="store_true",
        help="Force regeneration of chain datasets even if they already exist.",
    )
    parser.add_argument("--perc-var", default=",".join(map(str, DEFAULT_PERC_VARS)))
    parser.add_argument("--decimal-threshold", default=",".join(map(str, DEFAULT_DECIMAL_THRESHOLDS)))
    parser.add_argument("--community-threshold", default=",".join(map(str, DEFAULT_COMMUNITY_THRESHOLDS)))
    parser.add_argument(
        "--mode",
        choices=("random", "full"),
        default="random",
        help=(
            "Search strategy. 'random' (default) samples N combinations from "
            "the candidate ranges; 'full' reproduces the exhaustive grid."
        ),
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=DEFAULT_N_RANDOM_SAMPLES,
        help="Number of random hyper-parameter combinations per dataset (random mode).",
    )
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    parser.add_argument(
        "--random-features",
        action="store_true",
        help="If set, each run samples a random feature subset instead of using all features.",
    )
    parser.add_argument("--feature-frac-min", type=float, default=0.5)
    parser.add_argument("--feature-frac-max", type=float, default=1.0)
    parser.add_argument("--feature-min", type=int, default=1)
    args = parser.parse_args(argv)

    # --- Optional: synthesize chain-friendly datasets ---------------------
    if args.generate_chain_datasets:
        generate_chain_datasets(force=args.force_chain_regen)

    # --- Resolve dataset list --------------------------------------------
    # This script is chain-only: any --datasets entry that doesn't follow
    # the ``toy_chain_*.csv`` convention is filtered out (with a warning).
    if args.datasets:
        requested = [s.strip() for s in args.datasets.split(",") if s.strip()]
    else:
        requested = discover_chain_datasets()

    datasets: List[str] = []
    for name in requested:
        if not name.startswith(DEFAULT_DATASET_PREFIX):
            print(
                f"  ! SKIP: '{name}' does not match the "
                f"'{DEFAULT_DATASET_PREFIX}*.csv' convention required by "
                f"this script."
            )
            continue
        datasets.append(_with_csv(name))

    # Deduplicate while preserving order.
    datasets = list(dict.fromkeys(datasets))

    perc_vars = [float(s) for s in args.perc_var.split(",") if s.strip()]
    dec_thr = [int(s) for s in args.decimal_threshold.split(",") if s.strip()]
    com_thr = [float(s) for s in args.community_threshold.split(",") if s.strip()]

    os.makedirs(RESULTS_ROOT, exist_ok=True)
    print(f"[gridsearch-cat] results root: {RESULTS_ROOT}")

    rng = random.Random(args.seed)
    if args.mode == "full":
        combos: List[tuple] = [(pv, dt, ct) for pv in perc_vars for dt in dec_thr for ct in com_thr]
        print(
            f"[gridsearch-cat] mode=full: {len(datasets)} datasets x {len(combos)} combos "
            f"= {len(datasets) * len(combos)} runs"
        )
    else:
        n_samples = max(1, min(args.n_samples, len(perc_vars) * len(dec_thr) * len(com_thr)))
        combos = [
            (rng.choice(perc_vars), rng.choice(dec_thr), rng.choice(com_thr))
            for _ in range(n_samples)
        ]
        print(
            f"[gridsearch-cat] mode=random (seed={args.seed}): {len(datasets)} datasets x "
            f"{len(combos)} sampled combos = {len(datasets) * len(combos)} runs"
        )

    results: List[Dict] = []
    feature_rng = random.Random(args.seed + 1) if args.random_features else None
    for ds in datasets:
        ds_path = os.path.join(DATA_DIR, _with_csv(ds))
        if not os.path.exists(ds_path):
            print(f"  ! SKIP: dataset not found -> {ds_path}")
            continue
        for pv, dt, ct in combos:
            tag = f"ds={ds.replace('toy_','').replace('.csv','')} pv={pv} dt={dt} ct={ct}"
            print(f"\n[run] {tag}")
            try:
                r = run_one(
                    ds,
                    pv,
                    dt,
                    ct,
                    feature_subset_rng=feature_rng,
                    feature_frac_min=args.feature_frac_min,
                    feature_frac_max=args.feature_frac_max,
                    feature_min=args.feature_min,
                )
            except Exception as exc:  # noqa: BLE001
                print(f"  ! FAILED: {exc}")
                continue
            print(
                f"  acc={r.accuracy:.3f}  f1={r.f1:.3f}  "
                f"nodes={r.n_nodes}  edges={r.n_edges}  comms={r.n_communities}  "
                f"feats={r.n_features_used}"
            )
            results.append(r.__dict__)

    df = pd.DataFrame(results)
    write_header = not os.path.exists(SUMMARY_PATH)
    df.to_csv(SUMMARY_PATH, mode="a", header=write_header, index=False)
    print(f"\n[gridsearch-cat] wrote summary -> {SUMMARY_PATH}")
    if not df.empty:
        print("\n=== summary (sorted by accuracy desc, then n_nodes asc) ===")
        show = df.sort_values(
            ["accuracy", "n_nodes"], ascending=[False, True]
        )[
            [
                "dataset", "perc_var", "decimal_threshold",
                "community_threshold", "accuracy", "f1",
                "n_nodes", "n_edges", "n_communities",
                "n_features_used", "out_dir",
            ]
        ]
        print(show.to_string(index=False))


if __name__ == "__main__":
    main()
