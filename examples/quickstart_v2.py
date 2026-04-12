"""
DPG Analysis Pipeline - Example Script
=====================================
This script demonstrates a complete workflow for:
1. Training a Random Forest model
2. Generating Decision Predicate Graphs (DPG)
3. Extracting and visualizing interpretability metrics
"""
import sys
import os

SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
sys.path.insert(0, PROJECT_ROOT)

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import KFold
from dpg import DPGExplainer
import yaml
from counterfactual.utils.dataset_loader import load_dataset as load_dataset_from_loader
from counterfactual.utils.config_manager import DictConfig


def build_perc_var_candidates(base_perc_var):
    # Start from the configured value and progressively relax sparsity if plotting times out.
    env_candidates = os.getenv("DPG_PERC_VAR_CANDIDATES")
    if env_candidates:
        raw_values = [v.strip() for v in env_candidates.split(",") if v.strip()]
        candidates = [float(v) for v in raw_values]
    else:
        # If configured perc_var is extremely small, quickly jump to practical values.
        if base_perc_var < 1e-6:
            candidates = [base_perc_var, 1e-4, 1e-3, 1e-2, 5e-2, 1e-1]
        else:
            candidates = [
                base_perc_var,
                base_perc_var * 10,
                base_perc_var * 100,
                base_perc_var * 1_000,
                1e-3,
                1e-2,
                5e-2,
                1e-1,
            ]

    cleaned = []
    for value in candidates:
        value = float(value)
        if value <= 0:
            continue
        # Keep values in a practical range for this library.
        value = min(value, 0.5)
        if value not in cleaned:
            cleaned.append(value)
    return cleaned

def load_config(config_path):
    # Read YAML configuration used by the DPG library (percentile, thresholds, etc.)
    try:
        with open(config_path) as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Config file not found at {config_path}")
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Invalid YAML in config file: {str(e)}")


def load_dataset(dataset_name, base_dir):
    # Reuse the shared counterfactual dataset loader to keep ingestion logic centralized.
    loader_config = DictConfig(
        {
            "data": {
                "dataset": dataset_name,
                "dataset_url": "https://huggingface.co/datasets/MLLab-TS/german_credit/resolve/main/dataset.csv",
                "target_column": "foreign_worker",
                "categorical_encoding": "onehot",
                # Preserve prior quickstart behavior where first CSV column was used as index.
                "drop_columns": ["default"],
                "missing_values": "fill",
                "separator": ",",
            }
        }
    )
    dataset_info = load_dataset_from_loader(loader_config, repo_root=base_dir)

    features_df = dataset_info["features_df"]
    labels = pd.Series(dataset_info["labels"])
    feature_names = features_df.columns

    print("Size of X", features_df.shape)
    return features_df, labels, feature_names


def train_model_cv(model, features_matrix, labels, random_state):
    # Run K-Fold CV to report performance and keep the last trained split
    kf = KFold(n_splits=5, shuffle=True, random_state=random_state)
    accuracy_scores, f1_scores = [], []
    last_train = None

    for train_index, test_index in kf.split(features_matrix):
        X_train, X_test = features_matrix.iloc[train_index], features_matrix.iloc[test_index]
        y_train, y_test = labels.iloc[train_index], labels.iloc[test_index]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')

        accuracy_scores.append(accuracy)
        f1_scores.append(f1)
        last_train = (X_train, y_train)

        print(f"Fold - Accuracy: {accuracy}, F1-Score: {f1}")

    mean_accuracy = np.mean(accuracy_scores)
    metric_suffix = f"acc_{np.round(mean_accuracy, 2)}"
    return metric_suffix, last_train

def main():
    # =============================================================================
    # CONFIGURATION SECTION
    # =============================================================================
    # Tutorial note: adjust these values to point at your dataset and control the run.
    config = {
        "dataset_name": "german_credit",
        "num_trees": 10,
        "run_tag": "CustomDPG",
        "random_state": 27,
        "config_path": os.path.join(PROJECT_ROOT, "config.yaml"),
        "results_dir": os.path.join(SCRIPT_DIR, "results_v2"),
    }

    # Load DPG defaults from config.yaml
    config_data = load_config(config["config_path"])
    configured_perc_var = float(config_data["dpg"]["default"]["perc_var"])
    perc_var_candidates = build_perc_var_candidates(configured_perc_var)

    base_dir = PROJECT_ROOT
    # Load and clean the dataset
    features_matrix, labels, feature_names = load_dataset(
        config["dataset_name"], base_dir
    )

    # Train a Random Forest and estimate performance via CV
    model = RandomForestClassifier(
        n_estimators=config["num_trees"], random_state=config["random_state"]
    )

    metric_suffix, last_train = train_model_cv(
        model, features_matrix, labels, random_state=42
    )
    if last_train is None:
        raise RuntimeError("K-Fold training did not produce any train split.")
    X_train, y_train = last_train

    os.makedirs(config["results_dir"], exist_ok=True)
    target_names = np.unique(labels).astype(str).tolist()

    last_timeout_error = None
    for effective_perc_var in perc_var_candidates:
        run_id = (
            f"{model.__class__.__name__}_{config['run_tag']}_s{features_matrix.shape[0]}"
            f"_bl{config['num_trees']}_{metric_suffix}_perc_{effective_perc_var}"
        )

        print(f"INFO: Attempting DPG explain/plot with perc_var={effective_perc_var}")
        explainer = DPGExplainer(
            model=model,
            feature_names=feature_names,
            target_names=target_names,
            config_file=config["config_path"],
            dpg_config={
                "dpg": {
                    "default": {
                        "perc_var": effective_perc_var,
                        "decimal_threshold": config_data["dpg"]["default"].get("decimal_threshold", 6),
                        "n_jobs": config_data["dpg"]["default"].get("n_jobs", -1),
                    }
                }
            },
        )
        explanation = explainer.explain_global(
            X_train.values,
            communities=True,
            community_threshold=0.2,
        )

        # Save class boundary summary
        class_boundaries_path = os.path.join(
            config["results_dir"], f"{run_id}_dpg_class_boundaries.txt"
        )
        with open(class_boundaries_path, "w") as f:
            for key, value in explanation.class_boundaries.items():
                f.write(f"{key}: {value}\n")

        # Save node and edge metrics
        node_metrics_path = os.path.join(
            config["results_dir"], f"{run_id}_node_metrics.csv"
        )
        explanation.node_metrics.to_csv(node_metrics_path, encoding="utf-8")

        edge_metrics_path = os.path.join(
            config["results_dir"], f"{run_id}_edge_metrics.csv"
        )
        explanation.edge_metrics.to_csv(edge_metrics_path, encoding="utf-8")

        # Save communities
        communities_path = os.path.join(
            config["results_dir"], f"{run_id}_dpg_communities.txt"
        )
        if explanation.communities is not None:
            from metrics.graph import GraphMetrics

            GraphMetrics.communities_to_csv(explanation.communities, communities_path)

        run_name = f"{run_id}_DPG"
        try:
            # Render plots; on Graphviz timeout try a higher perc_var.
            explainer.plot(
                run_name,
                explanation=explanation,
                save_dir=config["results_dir"],
                class_flag=False,
                show=False,
                export_pdf=True,
            )
            explainer.plot_communities(
                run_name,
                explanation=explanation,
                save_dir=config["results_dir"],
                class_flag=True,
                show=False,
                export_pdf=True,
            )
            print(f"INFO: Plotting completed with perc_var={effective_perc_var}")
            return
        except TimeoutError as err:
            last_timeout_error = err
            print(
                f"WARNING: Plotting timed out with perc_var={effective_perc_var}. "
                "Retrying with a higher perc_var..."
            )

    raise RuntimeError(
        "All perc_var attempts timed out during plotting. "
        "Set DPG_PERC_VAR_CANDIDATES to a more aggressive list (e.g. 0.01,0.05,0.1)."
    ) from last_timeout_error


if __name__ == "__main__":
    main()