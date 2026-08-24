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
import json
import re

SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
sys.path.insert(0, PROJECT_ROOT)

import pandas as pd
import numpy as np
import networkx as nx
from networkx.readwrite import json_graph
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import OrdinalEncoder
from dpg import DPGExplainer
import yaml

def load_config(config_path):
    # Read YAML configuration used by the DPG library (percentile, thresholds, etc.)
    try:
        with open(config_path) as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Config file not found at {config_path}")
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Invalid YAML in config file: {str(e)}")


def load_dataset(dataset_name, base_dir, categorical_encoding="onehot", target_column=None):
    # Load CSV dataset from datasets/ and split features/labels.
    # target_column lets callers point at the real label column explicitly -
    # not every dataset puts it last (e.g. german_credit has it first).
    dataset_path = os.path.join(base_dir, "datasets/credit_card_approval", dataset_name)
    dataset_raw = pd.read_csv(dataset_path)

    if target_column is None:
        target_column = dataset_raw.columns[-1]

    features = dataset_raw.drop(columns=[target_column]).copy()
    labels = dataset_raw[target_column]

    # Capture original feature names before encoding (so the rest of the
    # pipeline still sees the original column names, e.g. for plots / DPG metrics).
    original_feature_names = list(features.columns)

    categorical_cols = [
        col
        for col in features.columns
        if features[col].dtype == "object" or features[col].dtype.name == "category"
    ]
    bool_cols = [col for col in features.columns if features[col].dtype == "bool"]

    # Encode categorical (object/string/bool) columns so they survive downstream
    # numeric operations like .mean() / .fillna(). Numeric columns are passed through.
    # feature_origin_map traces each post-encoding column back to the original
    # column it came from - needed to compare/aggregate explanations across
    # encodings that expand columns differently (e.g. one-hot).
    if categorical_encoding == "onehot":
        # Encode column-by-column (instead of one bulk pd.get_dummies call) so
        # we know exactly which original column produced each dummy column.
        encoded_parts = []
        feature_origin_map = {}
        for col in features.columns:
            if col in categorical_cols:
                dummies = pd.get_dummies(features[[col]], drop_first=False)
                for dcol in dummies.columns:
                    feature_origin_map[dcol] = col
                encoded_parts.append(dummies)
            else:
                feature_origin_map[col] = col
                encoded_parts.append(features[[col]])
        features = pd.concat(encoded_parts, axis=1)
    elif categorical_encoding == "ordinal":
        # Integer codes assigned per unique category value (alphabetical by
        # default), i.e. sklearn's OrdinalEncoder. Codes carry no inherent
        # magnitude/order relative to the original categories unless the
        # column happens to already be alphabetically ordered by meaning.
        if categorical_cols:
            features[categorical_cols] = OrdinalEncoder(dtype=np.float64).fit_transform(
                features[categorical_cols].astype(str)
            )
        for col in bool_cols:
            features[col] = features[col].astype(int)
        feature_origin_map = {col: col for col in features.columns}
    elif categorical_encoding == "target":
        # Mean/"target-like" encoding: each category is replaced by the mean
        # of the (numeric-coded) target among rows with that category.
        # Computed on the full dataset before the CV split in
        # train_model_cv, so this leaks target information across folds -
        # acceptable for this exploratory quickstart, not for a real model.
        target_numeric, _ = pd.factorize(labels)
        target_numeric = pd.Series(target_numeric, index=features.index)
        for col in categorical_cols:
            category_means = target_numeric.groupby(features[col]).mean()
            features[col] = features[col].map(category_means).astype(float)
        for col in bool_cols:
            features[col] = features[col].astype(int)
        feature_origin_map = {col: col for col in features.columns}
    else:
        raise ValueError(
            f"Unknown categorical_encoding '{categorical_encoding}', "
            "expected 'onehot', 'ordinal', or 'target'"
        )

    # Clean data: handle infinities and missing values
    features = features.replace([np.inf, -np.inf], np.nan).fillna(features.mean())
    features_matrix = np.round(features, 2)

    # Use the post-encoding column names so DPG and the visualizer see the dummy
    # columns (e.g. person_home_ownership_RENT) rather than the originals.
    feature_names = list(features_matrix.columns)

    print("Size of X", features_matrix.shape)
    return features_matrix, labels, feature_names, original_feature_names, feature_origin_map


def train_model_cv(model, features_matrix, labels, random_state):
    # Run K-Fold CV to report performance and keep the last trained split
    kf = KFold(n_splits=5, shuffle=True, random_state=random_state)
    accuracy_scores, f1_scores = [], []
    last_train = None
    last_test = None

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
        # Kept so callers can measure fidelity on data the DPG never saw
        # (it's only ever built from last_train).
        last_test = (X_test, y_test)

        print(f"Fold - Accuracy: {accuracy}, F1-Score: {f1}")

    mean_accuracy = np.mean(accuracy_scores)
    metric_suffix = f"acc_{np.round(mean_accuracy, 2)}"
    return metric_suffix, last_train, last_test, float(mean_accuracy)


##### LUCAS JAKIN
def extract_predicate_columns(explanation, feature_origin_map):
    """Return (encoded_column, original_column) for every predicate (split)
    node in the DPG graph - class/leaf nodes are skipped. Mirrors the same
    ' <= '/' > ' split used by GraphMetrics.calculate_class_boundaries."""
    pairs = []
    for label in explanation.node_metrics["Label"]:
        parts = re.split(' <= | > ', str(label))
        if len(parts) != 2:
            continue
        encoded_col = parts[0].strip()
        pairs.append((encoded_col, feature_origin_map.get(encoded_col, encoded_col)))
    return pairs


##### LUCAS JAKIN
def compute_clarity_metrics(explanation, feature_origin_map):
    """Heuristic 'semantic clarity' numbers for a DPG graph:
    - how fragmented the original features are across predicate nodes
      (one-hot splits a single original feature into many nodes)
    - what fraction of predicates are self-explanatory from their column
      name alone (true for one-hot dummy columns, e.g.
      "gender_female <= 0.5"; false for ordinal/target codes where the
      split value needs an external lookup table to mean anything).
    """
    pairs = extract_predicate_columns(explanation, feature_origin_map)
    total = len(pairs)
    original_features = {orig for _, orig in pairs}
    self_explanatory = sum(1 for enc, orig in pairs if enc != orig)
    return {
        "num_predicate_nodes": total,
        "num_original_features_referenced": len(original_features),
        "avg_predicate_nodes_per_feature": (
            total / len(original_features) if original_features else float("nan")
        ),
        "pct_self_explanatory_predicates": (
            self_explanatory / total if total else float("nan")
        ),
    }


##### LUCAS JAKIN
def compute_graph_depth(explanation):
    # Longest root-to-leaf predicate chain; only well-defined if the merged
    # graph is acyclic (predicate nodes shared across trees with differing
    # split order can in principle create cycles).
    graph = explanation.graph
    if nx.is_directed_acyclic_graph(graph):
        # weight=None: count hops, don't sum the edges' path-occurrence
        # "weight" attribute (dag_longest_path_length sums it by default).
        return nx.dag_longest_path_length(graph, weight=None)
    return float("nan")


##### LUCAS JAKIN
def node_label_set(explanation):
    return set(explanation.node_metrics["Label"].astype(str))


##### LUCAS JAKIN
def edge_pair_set(explanation):
    return set(
        zip(
            explanation.edge_metrics["Node_u_label"].astype(str),
            explanation.edge_metrics["Node_v_label"].astype(str),
        )
    )


##### LUCAS JAKIN
def original_feature_set(explanation, feature_origin_map):
    return {orig for _, orig in extract_predicate_columns(explanation, feature_origin_map)}


##### LUCAS JAKIN
def jaccard(set_a, set_b):
    union = set_a | set_b
    if not union:
        return float("nan")
    return len(set_a & set_b) / len(union)


##### LUCAS JAKIN
def parse_predicate_label(label):
    """Split a DPG node label into (feature, operator, threshold).
    Returns None for class/leaf nodes (e.g. "Class 0"), which have no
    ' <= '/' > ' predicate to parse. Node labels are single, literal
    conditions (see dpg/core.py tracing_ensemble: "feature <= threshold"
    for the left/true branch, "feature > threshold" for the right/false
    branch) - so unlike a normal tree node, there is no hidden direction
    to resolve: the label alone says exactly what must hold.
    """
    label = str(label)
    if ' <= ' in label:
        feature, threshold = label.split(' <= ', 1)
        operator = '<='
    elif ' > ' in label:
        feature, threshold = label.split(' > ', 1)
        operator = '>'
    else:
        return None
    try:
        threshold = float(threshold.strip())
    except ValueError:
        return None
    return feature.strip(), operator, threshold


##### LUCAS JAKIN
def node_holds_for_sample(label, sample_row):
    """True/False if `sample_row` satisfies this predicate node's condition,
    or None if the node is a class/leaf node or references a feature not
    present in sample_row."""
    parsed = parse_predicate_label(label)
    if parsed is None:
        return None
    feature, operator, threshold = parsed
    if feature not in sample_row.index:
        return None
    value = sample_row[feature]
    if operator == '<=':
        return bool(value <= threshold)
    return bool(value > threshold)


##### LUCAS JAKIN
def predict_with_dpg(explanation, sample_row):
    """Predict a class for one sample by walking the DPG graph itself,
    instead of asking the Random Forest.

    Starts only from nodes with in-degree 0 (the closest thing this
    merged, root-less graph has to "first predicate tested") whose
    condition the sample satisfies, then keeps following an edge only
    while the sample also satisfies the next node's condition, until a
    class node is reached. A per-path visited-set prevents infinite loops
    on the cycles this graph can contain. If several distinct paths reach
    different classes, the one backed by the larger total edge weight
    (i.e. supported by more of the original training paths) wins.

    Returns the predicted class as a string, or None if no path in the
    (pruned) graph matches this sample at all - the "coverage" gap that
    perc_var pruning can create. Because in-degree 0 can only under-count
    true start nodes (a node can end up with incoming edges just by also
    appearing mid-path in some other tree), this is a conservative, never
    over-optimistic, estimate of what the graph can predict.
    """
    graph = explanation.graph
    label_by_id = dict(explanation.nodes)

    class_weights = {}
    for node_id, label in explanation.nodes:
        label = str(label)
        if label.startswith("Class "):
            continue  # a class node can't be a starting predicate
        if graph.in_degree(node_id) != 0:
            continue  # only root-like nodes count as valid starts
        if node_holds_for_sample(label, sample_row) is not True:
            continue

        stack = [(node_id, frozenset([node_id]))]
        while stack:
            current_id, visited = stack.pop()
            for _, next_id, edge_data in graph.out_edges(current_id, data=True):
                if next_id in visited:
                    continue
                next_label = str(label_by_id.get(next_id, ""))
                weight = float(edge_data.get("weight", 1.0))
                if next_label.startswith("Class "):
                    class_name = next_label[len("Class "):]
                    class_weights[class_name] = class_weights.get(class_name, 0.0) + weight
                elif node_holds_for_sample(next_label, sample_row) is True:
                    stack.append((next_id, visited | {next_id}))

    if not class_weights:
        return None
    return max(class_weights, key=class_weights.get)


##### LUCAS JAKIN
def compute_fidelity(explanation, model, X_test):
    """How often the DPG graph's own prediction agrees with the Random
    Forest's prediction, on held-out data the graph never saw.

    Reported as two separate numbers rather than one blended score:
    - fidelity_coverage: fraction of test samples the graph could
      classify at all (a heavily pruned graph can have low coverage
      without that meaning its covered predictions are wrong).
    - fidelity: agreement rate with the Random Forest, among covered
      samples only.
    """
    model_preds = [str(p) for p in model.predict(X_test)]
    dpg_preds = [predict_with_dpg(explanation, X_test.iloc[i]) for i in range(len(X_test))]

    covered = [i for i, p in enumerate(dpg_preds) if p is not None]
    coverage = len(covered) / len(dpg_preds) if dpg_preds else float("nan")
    if covered:
        agreements = sum(1 for i in covered if dpg_preds[i] == model_preds[i])
        fidelity = agreements / len(covered)
    else:
        fidelity = float("nan")

    return {
        "fidelity": fidelity,
        "fidelity_coverage": coverage,
        "fidelity_n_test": len(dpg_preds),
    }


##### LUCAS JAKIN
def _json_default(value):
    # Convert NumPy/Pandas scalars and arrays into standard JSON-compatible types.
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


##### LUCAS JAKIN
def save_dpg_structure_json(explanation, output_path, run_id, feature_names, target_names):
    graph_data = json_graph.node_link_data(explanation.graph)
    labeled_nodes = [
        {"id": node_id, "label": label}
        for node_id, label in explanation.nodes
    ]

    payload = {
        "run_id": run_id,
        "feature_names": [str(name) for name in feature_names],
        "target_names": [str(name) for name in target_names],
        "community_threshold": explanation.community_threshold,
        "nodes": labeled_nodes,
        "graph": graph_data,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=_json_default)

##### LUCAS JAKIN
def run_single_encoding(base_dir, base_config, config_data, categorical_encoding):
    """Run the full train + DPG-explain + save pipeline for one categorical
    encoding, writing outputs under base_config['results_dir']/<encoding>/.
    Returns a summary dict used to compare encodings afterward.
    """
    config = dict(base_config)
    config["categorical_encoding"] = categorical_encoding
    config["results_dir"] = os.path.join(base_config["results_dir"], categorical_encoding)

    perc_var = config_data["dpg"]["default"]["perc_var"]

    features_matrix, labels, feature_names, original_feature_names, feature_origin_map = load_dataset(
        config["dataset_name"], base_dir, categorical_encoding=categorical_encoding,
        target_column=config["target_column"],
    )

    model = RandomForestClassifier(
        n_estimators=config["num_trees"], random_state=config["random_state"]
    )

    metric_suffix, last_train, last_test, mean_accuracy = train_model_cv(
        model, features_matrix, labels, random_state=42
    )
    X_train, y_train = last_train
    X_test, y_test = last_test

    run_id = (
        f"{model.__class__.__name__}_{config['run_tag']}_s{features_matrix.shape[0]}"
        f"_bl{config['num_trees']}_{metric_suffix}_perc_{perc_var}"
        f"_enc{categorical_encoding}"
    )

    target_names = np.unique(labels).astype(str).tolist()
    explainer = DPGExplainer(
        model=model,
        feature_names=feature_names,
        target_names=target_names,
        config_file=config["config_path"],
    )

    explanation = explainer.explain_global(
        X_train.values,
        communities=True,
        community_threshold=0.2,
    )

    os.makedirs(config["results_dir"], exist_ok=True)

    dpg_structure_path = os.path.join(
        config["results_dir"], f"{run_id}_dpg_structure.json"
    )
    save_dpg_structure_json(
        explanation=explanation,
        output_path=dpg_structure_path,
        run_id=run_id,
        feature_names=feature_names,
        target_names=target_names,
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

    # Render plots
    run_name = f"{run_id}_DPG"
    explainer.plot(
        run_name,
        explanation=explanation,
        save_dir=config["results_dir"],
        class_flag=False,
        export_pdf=True,
    )
    explainer.plot_communities(
        run_name,
        explanation=explanation,
        save_dir=config["results_dir"],
        class_flag=True,
        export_pdf=True,
    )

    clarity_metrics = compute_clarity_metrics(explanation, feature_origin_map)
    fidelity_metrics = compute_fidelity(explanation, model, X_test)

    return {
        "categorical_encoding": categorical_encoding,
        "cv_accuracy": mean_accuracy,
        "cv_accuracy_suffix": metric_suffix,
        "num_original_features": len(original_feature_names),
        "num_encoded_features": len(feature_names),
        "num_dpg_nodes": len(explanation.node_metrics),
        "num_dpg_edges": len(explanation.edge_metrics),
        "graph_depth": compute_graph_depth(explanation),
        **clarity_metrics,
        **fidelity_metrics,
        # Not written to the scalar comparison CSV - kept for the
        # cross-encoding agreement comparison in main().
        "_node_labels": node_label_set(explanation),
        "_edge_pairs": edge_pair_set(explanation),
        "_original_features_in_graph": original_feature_set(explanation, feature_origin_map),
    }


##### LUCAS JAKIN
def run_seed_stability_sweep(base_dir, base_config, config_data, categorical_encoding, seeds):
    """Rebuild the RF + DPG explanation across multiple random seeds (same
    seed drives both the RF and the CV split) without saving/plotting, then
    measure how much the graph's predicate structure changes run to run.
    Higher mean Jaccard = more stable explanation across seeds.
    """
    runs = []
    for seed in seeds:
        features_matrix, labels, feature_names, _original_feature_names, feature_origin_map = load_dataset(
            base_config["dataset_name"], base_dir, categorical_encoding=categorical_encoding,
            target_column=base_config["target_column"],
        )

        model = RandomForestClassifier(n_estimators=base_config["num_trees"], random_state=seed)
        _metric_suffix, last_train, _last_test, mean_accuracy = train_model_cv(
            model, features_matrix, labels, random_state=seed
        )
        X_train, _y_train = last_train

        target_names = np.unique(labels).astype(str).tolist()
        explainer = DPGExplainer(
            model=model,
            feature_names=feature_names,
            target_names=target_names,
            config_file=base_config["config_path"],
        )
        try:
            explanation = explainer.explain_global(X_train.values, communities=False)
        except Exception as err:
            # Some seeds can produce a tree ensemble with no path frequent
            # enough to clear perc_var - skip that seed rather than losing
            # the whole sweep, since we only need >=2 successful runs to
            # compute pairwise stability.
            print(f"WARNING: seed={seed} failed to build a DPG ({err}); skipping it")
            continue

        runs.append({
            "seed": seed,
            "cv_accuracy": mean_accuracy,
            "node_labels": node_label_set(explanation),
            "edge_pairs": edge_pair_set(explanation),
            "original_features": original_feature_set(explanation, feature_origin_map),
        })

    exact_label_jaccards, edge_jaccards, feature_jaccards = [], [], []
    for i in range(len(runs)):
        for j in range(i + 1, len(runs)):
            exact_label_jaccards.append(jaccard(runs[i]["node_labels"], runs[j]["node_labels"]))
            edge_jaccards.append(jaccard(runs[i]["edge_pairs"], runs[j]["edge_pairs"]))
            feature_jaccards.append(jaccard(runs[i]["original_features"], runs[j]["original_features"]))

    return {
        "categorical_encoding": categorical_encoding,
        "num_seeds_requested": len(seeds),
        "num_seeds_succeeded": len(runs),
        "mean_cv_accuracy": float(np.mean([r["cv_accuracy"] for r in runs])) if runs else float("nan"),
        "std_cv_accuracy": float(np.std([r["cv_accuracy"] for r in runs])) if runs else float("nan"),
        # Exact-label Jaccard: did the identical predicate string reappear
        # across seeds (strictest - sensitive to tiny threshold shifts).
        "mean_exact_label_jaccard": float(np.nanmean(exact_label_jaccards)),
        # Edge Jaccard: did the same predicate-to-predicate connections reappear.
        "mean_edge_jaccard": float(np.nanmean(edge_jaccards)),
        # Feature Jaccard: did the graph keep splitting on the same original
        # features, regardless of exact threshold (coarsest, most forgiving).
        "mean_feature_jaccard": float(np.nanmean(feature_jaccards)),
    }


##### LUCAS JAKIN
def main():
    # =============================================================================
    # CONFIGURATION SECTION
    # =============================================================================
    # Tutorial note: adjust these values to point at your dataset and control the run.
    config = {
        "dataset_name": "loan_data.csv",
        "target_column": "loan_status",
        "num_trees": 10,
        "run_tag": "CustomDPG",
        "random_state": 27,
        "config_path": os.path.join(PROJECT_ROOT, "config.yaml"),
        "results_dir": os.path.join(SCRIPT_DIR, "results_workPackage5"),
        # Encodings to compare. Each runs the full RF + DPG pipeline and is
        # written to its own results_dir/<encoding>/ subfolder.
        "categorical_encodings": ["onehot", "ordinal", "target"],
        # Seeds used to check how stable each encoding's DPG graph is across
        # independent train/CV runs (same dataset, different randomness).
        "stability_seeds": [27, 42, 123, 7, 99],
        # Two encodings are only compared for graph agreement if their mean
        # CV accuracy is within this tolerance of each other - otherwise a
        # low overlap could just mean "one model is worse", not "the
        # explanations disagree".
        "agreement_accuracy_tolerance": 0.05,
    }

    # Load DPG defaults from config.yaml
    config_data = load_config(config["config_path"])
    base_dir = PROJECT_ROOT

    ##### LUCAS JAKIN
    # --- Full pipeline per encoding: train, build DPG, save + plot ---
    results = []
    for encoding in config["categorical_encodings"]:
        print(f"\n=== Running pipeline with categorical_encoding={encoding} ===")
        results.append(run_single_encoding(base_dir, config, config_data, encoding))

    os.makedirs(config["results_dir"], exist_ok=True)
    set_fields = {"_node_labels", "_edge_pairs", "_original_features_in_graph"}
    comparison_rows = [
        {k: v for k, v in r.items() if k not in set_fields} for r in results
    ]
    comparison_path = os.path.join(config["results_dir"], "encoding_comparison.csv")
    pd.DataFrame(comparison_rows).to_csv(comparison_path, index=False)
    print(f"\nSaved encoding comparison summary to {comparison_path}")

    # --- Stability across seeds, per encoding ---
    stability_rows = []
    for encoding in config["categorical_encodings"]:
        print(f"\n=== Seed-stability sweep for categorical_encoding={encoding} ===")
        stability_rows.append(
            run_seed_stability_sweep(
                base_dir, config, config_data, encoding, config["stability_seeds"]
            )
        )
    stability_path = os.path.join(config["results_dir"], "encoding_stability.csv")
    pd.DataFrame(stability_rows).to_csv(stability_path, index=False)
    print(f"Saved seed-stability summary to {stability_path}")

    # --- Agreement between explanations under similar predictive performance ---
    agreement_rows = []
    tolerance = config["agreement_accuracy_tolerance"]
    for i in range(len(results)):
        for j in range(i + 1, len(results)):
            a, b = results[i], results[j]
            if abs(a["cv_accuracy"] - b["cv_accuracy"]) <= tolerance:
                agreement_rows.append({
                    "encoding_a": a["categorical_encoding"],
                    "encoding_b": b["categorical_encoding"],
                    "accuracy_a": a["cv_accuracy"],
                    "accuracy_b": b["cv_accuracy"],
                    # Compared at the level of original (pre-encoding)
                    # features, since raw predicate label strings aren't
                    # comparable across different encodings.
                    "feature_jaccard": jaccard(
                        a["_original_features_in_graph"], b["_original_features_in_graph"]
                    ),
                })

    agreement_path = os.path.join(config["results_dir"], "encoding_agreement.csv")
    if agreement_rows:
        pd.DataFrame(agreement_rows).to_csv(agreement_path, index=False)
        print(f"Saved cross-encoding agreement summary to {agreement_path}")
    else:
        print(
            f"No encoding pairs within accuracy tolerance {tolerance}; "
            f"skipped {agreement_path}"
        )
    ##### LUCAS JAKIN


if __name__ == "__main__":
    main()
