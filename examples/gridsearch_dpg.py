"""
gridsearch_dpg.py
=================

A small pseudo-gridsearch around `examples/quickstart_categorical.py`.

It trains a RandomForest on each generated toy dataset, sweeps over
`perc_var` x `decimal_threshold` x `community_threshold`, and writes a
per-run subdirectory under `examples/results_gridsearch/<run_id>/` with:

  * <run_id>_dpg_structure.json
  * <run_id>_node_metrics.csv
  * <run_id>_edge_metrics.csv
  * <run_id>_dpg_class_boundaries.txt
  * <run_id>_dpg_communities.txt
  * <run_id>_DPG.png / .pdf
  * <run_id>_DPG_communities.png / .pdf
  * config.yaml  (the exact config used for this run)

A summary CSV (`examples/results_gridsearch/_summary.csv`) is appended on
every run so we can compare candidates afterwards.

Usage:
    python examples/gridsearch_dpg.py
    python examples/gridsearch_dpg.py --datasets toy_cat1_ownership,toy_cat2_gender_ownership
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
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
BASE_CONFIG_PATH = os.path.join(PROJECT_ROOT, "config.yaml")
RESULTS_ROOT = os.path.join(SCRIPT_DIR, "results_gridsearch")
SUMMARY_PATH = os.path.join(RESULTS_ROOT, "_summary.csv")

# Default toy dataset filenames (created by create_dataset.py)
DEFAULT_DATASETS = [
    "toy_cat1_ownership.csv",
    "toy_cat1_gender.csv",
    "toy_cat2_gender_ownership.csv",
    "toy_cat1_num1_ownership_age.csv",
    "toy_cat2_num1_gender_ownership_age.csv",
    "toy_cat2_3way_ownership_gender.csv",
    "toy_cat2_num1_age_interaction.csv",
    "toy_cat2_num1_age_required.csv",
]

# Default hyper-parameter grid
DEFAULT_PERC_VARS = [0.005, 0.01, 0.025, 0.05, 0.075, 0.10]
DEFAULT_DECIMAL_THRESHOLDS = [1, 2, 3]
DEFAULT_COMMUNITY_THRESHOLDS = [0.10, 0.20, 0.30]

RANDOM_STATE = 27
NUM_TREES = 10
N_SPLITS = 5


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def load_dataset(dataset_name: str):
    """Load a CSV from DATA_DIR, return (X, y, feature_names, original_names)."""
    path = os.path.join(DATA_DIR, dataset_name)
    df = pd.read_csv(path)
    features = df.iloc[:, :-1]
    target_col = df.columns[-1]
    labels = df[target_col]

    original_feature_names = list(features.columns)
    features_enc = pd.get_dummies(features, drop_first=False)
    features_enc = features_enc.replace([np.inf, -np.inf], np.nan).fillna(features_enc.mean())
    features_enc = np.round(features_enc, 3)  # internal precision; the per-run
    # decimal_threshold is applied separately by the DPG config.
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
    cfg["dpg"]["default"]["n_jobs"] = 1  # keep child processes predictable
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
# Core run
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


def run_one(
    dataset_name: str,
    perc_var: float,
    decimal_threshold: int,
    community_threshold: float,
) -> RunResult:
    # Output directory
    short_ds = dataset_name.replace("toy_", "").replace(".csv", "")
    run_id = (
        f"ds={short_ds}_pv={perc_var}_dt={decimal_threshold}_ct={community_threshold}"
    )
    out_dir = os.path.join(RESULTS_ROOT, run_id)
    os.makedirs(out_dir, exist_ok=True)

    # Per-run config.yaml
    cfg_path = write_run_config(out_dir, perc_var, decimal_threshold)

    # Load + train
    X, y, feature_names, original_feature_names = load_dataset(dataset_name)
    model = RandomForestClassifier(n_estimators=NUM_TREES, random_state=RANDOM_STATE)
    acc, f1, (X_train, y_train) = train_cv(model, X, y)
    metric_suffix = f"acc_{round(acc, 2)}"

    # Explain
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

    # Persist artifacts
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

    # Plots (no blocking show)
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
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="DPG gridsearch runner")
    parser.add_argument(
        "--datasets",
        default=",".join(DEFAULT_DATASETS),
        help="Comma-separated dataset filenames inside datasets/dummy_dataset/",
    )
    parser.add_argument("--perc-var", default=",".join(map(str, DEFAULT_PERC_VARS)))
    parser.add_argument("--decimal-threshold", default=",".join(map(str, DEFAULT_DECIMAL_THRESHOLDS)))
    parser.add_argument("--community-threshold", default=",".join(map(str, DEFAULT_COMMUNITY_THRESHOLDS)))
    args = parser.parse_args(argv)

    datasets = [s.strip() for s in args.datasets.split(",") if s.strip()]
    perc_vars = [float(s) for s in args.perc_var.split(",") if s.strip()]
    dec_thr = [int(s) for s in args.decimal_threshold.split(",") if s.strip()]
    com_thr = [float(s) for s in args.community_threshold.split(",") if s.strip()]

    os.makedirs(RESULTS_ROOT, exist_ok=True)
    print(f"[gridsearch] results root: {RESULTS_ROOT}")
    print(f"[gridsearch] {len(datasets)} datasets x {len(perc_vars)} perc_var x "
          f"{len(dec_thr)} decimal_threshold x {len(com_thr)} community_threshold "
          f"= {len(datasets)*len(perc_vars)*len(dec_thr)*len(com_thr)} runs")

    results: List[Dict] = []
    for ds in datasets:
        for pv in perc_vars:
            for dt in dec_thr:
                for ct in com_thr:
                    tag = f"ds={ds.replace('toy_','').replace('.csv','')} pv={pv} dt={dt} ct={ct}"
                    print(f"\n[run] {tag}")
                    try:
                        r = run_one(ds, pv, dt, ct)
                    except Exception as exc:  # noqa: BLE001
                        print(f"  ! FAILED: {exc}")
                        continue
                    print(
                        f"  acc={r.accuracy:.3f}  f1={r.f1:.3f}  "
                        f"nodes={r.n_nodes}  edges={r.n_edges}  comms={r.n_communities}"
                    )
                    results.append(r.__dict__)

    # Append to the summary CSV (write header only if the file doesn't exist)
    df = pd.DataFrame(results)
    write_header = not os.path.exists(SUMMARY_PATH)
    df.to_csv(SUMMARY_PATH, mode="a", header=write_header, index=False)
    print(f"\n[gridsearch] wrote summary -> {SUMMARY_PATH}")
    if not df.empty:
        print("\n=== summary (sorted by accuracy desc, then n_nodes asc) ===")
        show = df.sort_values(
            ["accuracy", "n_nodes"], ascending=[False, True]
        )[
            [
                "dataset", "perc_var", "decimal_threshold",
                "community_threshold", "accuracy", "f1",
                "n_nodes", "n_edges", "n_communities", "out_dir",
            ]
        ]
        print(show.to_string(index=False))


if __name__ == "__main__":
    main()
