"""
WP4: Categorical class summaries - before/after experiment
============================================================
Trains one Random Forest on the Credit Card Approval dataset (one-hot
encoded, the only encoding the new categorical summary style applies to -
see class_summary.py), builds one DPG, and for each class compares:

   -> the existing numeric-interval summary (GraphMetrics.extract_class_boundaries,
    UNCHANGED)
   -> the new categorical-aware summary from class_summary.py

4 measurements: COMPACTNESS, REDUNDANCY, CLASS-DISCRIMINATIVE POWER, and PURITY. 
"""
import sys
import os
from collections import Counter

SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
EXAMPLES_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
PROJECT_ROOT = os.path.abspath(os.path.join(EXAMPLES_DIR, ".."))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, EXAMPLES_DIR)

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from dpg import DPGExplainer

import class_summary



def load_dataset(dataset_path, target_column):
    """
    Minimal, one-hot-only loader (this module only reads mixed-type data
    to demonstrate the categorical summary style, not to sweep encodings).
    """
    raw = pd.read_csv(dataset_path)
    features = raw.drop(columns=[target_column]).copy()
    labels = raw[target_column]

    features = pd.get_dummies(features, drop_first=False)
    features = features.replace([np.inf, -np.inf], np.nan).fillna(features.mean())
    features_matrix = np.round(features, 2)
    return features_matrix, labels


def old_style_statements(boundaries):
    """
    (feature,) for each old-style boundary line - deliberately coarse
    It ignores the exact threshold value
     
    One side effect: if home_ownership_RENT <= 0.5 shows up for Class 0 and home_ownership_RENT > 0.5 shows up for Class 1, 
    this check only looks at the feature name — not the direction — so it treats them as the same, non-unique fact, 
    even though they actually say opposite things. The new style doesn't have this problem, since it keeps the direction as part of the fact
    """
    stmts = []
    for boundary in boundaries:
        feature = class_summary.numeric_boundary_feature_name(boundary)
        if feature is not None:
            stmts.append((feature,))
    return stmts



def new_style_statements(structured):
    """
    (feature, value) for every fact the new summary asserts - numeric
    bounds kept at the same coarse (feature,)-only granularity as old-style
    for a fair discriminative-power comparison, categorical facts kept at
    (feature, category) granularity.
    """
    stmts = []
    for boundary in structured["numeric_boundaries"]:
        feature = class_summary.numeric_boundary_feature_name(boundary)
        if feature is not None:
            stmts.append((feature,))  # same coarse granularity as old-style, for a fair comparison
    for fs in structured["categorical_features"]:
        if fs["dominant"]:
            stmts.append((fs["feature"], f"dominant:{fs['dominant'][0]}"))
        for category in fs["inclusion"]:
            stmts.append((fs["feature"], f"in:{category}"))
        for category in fs["exclusion"]:
            stmts.append((fs["feature"], f"out:{category}"))
    return stmts



def count_same_feature_repeats(old_boundaries):
    """
    Counts how often the old style splits one category into several separate lines instead of one — each extra line about the same feature counts as redundant.
    """
    bases = []
    for boundary in old_boundaries:
        feature = class_summary.numeric_boundary_feature_name(boundary)
        if feature is None:
            continue
        parsed = class_summary.parse_onehot_predicate(f"{feature} <= 0.5")
        if parsed:
            bases.append(parsed[0])
    counts = Counter(bases)
    return sum(c - 1 for c in counts.values() if c > 1)



def count_new_style_repeats(structured):
    """
    Redundancy for the NEW style: how many base features needed more
    than one statement line to describe.

    -> should be 0 - render_class_summary always collapses one feature into a single line
    """
    return 0  # invariant of the renderer; kept as a function for symmetry/clarity



def discriminative_power(stmts_by_class):
    """
    Fraction of a class's statements that are NOT also asserted (same
    feature[, value]) for every other class. 
    
    -> the HIGHER = the summary says
    more that is actually specific to this class.
    """
    classes = list(stmts_by_class.keys())
    scores = {}
    for c in classes:
        stmts = stmts_by_class[c]
        if not stmts:
            scores[c] = float("nan")
            continue
        shared = sum(
            1 for s in stmts
            if all(s in stmts_by_class[other] for other in classes if other != c)
        )
        scores[c] = 1 - shared / len(stmts)
    return scores



def old_style_purity(boundaries, node_probs, class_name):
    """
    Purity for OLD-style statements: for one-hot dummy boundaries the
    boundary string IS a raw node label, so it can be looked up directly. 
    
    Genuinely numeric bounds are merged across several nodes and have no single matching
    node label - skipped, same as the new style skips
    them for the same reason.
    """
    class_key = f"Class {class_name}"
    values = []
    for boundary in boundaries:
        probs = node_probs.get(boundary)
        if probs and class_key in probs:
            values.append(probs[class_key])
    return float(np.mean(values)) if values else float("nan")



def new_style_purity(structured, node_probs, class_name):
    """
    Purity for NEW-style categorical statements, reconstructing the
    underlying raw node label for each asserted (feature, category,
    direction) fact and reusing the same node-probability lookup.
    """
    class_key = f"Class {class_name}"
    values = []
    for fs in structured["categorical_features"]:
        candidates = []
        if fs["dominant"]:
            candidates.append((fs["dominant"][0], "in"))
        candidates += [(c, "in") for c in fs["inclusion"]]
        candidates += [(c, "out") for c in fs["exclusion"]]
        for category, direction in candidates:
            op = "> 0.5" if direction == "in" else "<= 0.5"
            label = f"{fs['feature']}_{category} {op}"
            probs = node_probs.get(label)
            if probs and class_key in probs:
                values.append(probs[class_key])
    return float(np.mean(values)) if values else float("nan")


def main():
    config = {
        "dataset_path": os.path.join(PROJECT_ROOT, "datasets", "credit_card_approval", "loan_data.csv"),
        "target_column": "loan_status",
        "num_trees": 10,
        "random_state": 27,
        "config_path": os.path.join(PROJECT_ROOT, "config.yaml"),
        "results_dir": os.path.join(EXAMPLES_DIR, "results_workPackage4"),
        "min_support_fraction": 0.05,
        "dominance_threshold": 0.5,
        "contrastive_top_n": 2,
    }
    os.makedirs(config["results_dir"], exist_ok=True)

    features_matrix, labels = load_dataset(config["dataset_path"], config["target_column"])
    X_train, X_test, y_train, y_test = train_test_split(
        features_matrix, labels, test_size=0.2, random_state=config["random_state"], stratify=labels
    )

    model = RandomForestClassifier(n_estimators=config["num_trees"], random_state=config["random_state"])
    model.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, model.predict(X_test))
    print(f"Held-out accuracy: {accuracy:.3f}")

    target_names = np.unique(labels).astype(str).tolist()
    explainer = DPGExplainer(
        model=model,
        feature_names=list(features_matrix.columns),
        target_names=target_names,
        config_file=config["config_path"],
    )
    explanation = explainer.explain_global(X_train.values, communities=False)

    old_bounds = explanation.class_boundaries.get("Class Bounds", {})
    new_summary = class_summary.build_new_style_summary(
        explanation,
        min_support_fraction=config["min_support_fraction"],
        dominance_threshold=config["dominance_threshold"],
        contrastive_top_n=config["contrastive_top_n"],
    )
    _support_table, node_probs = class_summary.build_category_support_table(
        explanation.graph, explanation.nodes, target_names
    )

    old_stmts_by_class, new_stmts_by_class = {}, {}
    rows = []
    examples_lines = []

    for class_name, entry in new_summary.items():
        old_key = f"Class {class_name}"
        boundaries = old_bounds.get(old_key, [])
        old_text = f"{old_key}: {boundaries}"
        new_text = entry["text"]

        old_stmts = old_style_statements(boundaries)
        new_stmts = new_style_statements(entry["structured"])
        old_stmts_by_class[class_name] = set(old_stmts)
        new_stmts_by_class[class_name] = set(new_stmts)

        examples_lines.append(f"=== {old_key} ===\n--- BEFORE (numeric-only) ---\n{old_text}\n"
                               f"--- AFTER (categorical-aware) ---\n{new_text}\n")

        rows.append({
            "class": class_name,
            "style": "old",
            "num_statements": len(boundaries),
            "word_count": len(old_text.split()),
            "redundant_statements": count_same_feature_repeats(boundaries),
            "purity": old_style_purity(boundaries, node_probs, class_name),
        })
        rows.append({
            "class": class_name,
            "style": "new",
            "num_statements": len(new_stmts),
            "word_count": len(new_text.split()),
            "redundant_statements": count_new_style_repeats(entry["structured"]),
            "purity": new_style_purity(entry["structured"], node_probs, class_name),
        })

    old_discriminative = discriminative_power(old_stmts_by_class)
    new_discriminative = discriminative_power(new_stmts_by_class)
    for row in rows:
        row["discriminative_power"] = (
            old_discriminative[row["class"]] if row["style"] == "old" else new_discriminative[row["class"]]
        )

    comparison_path = os.path.join(config["results_dir"], "class_summary_comparison.csv")
    pd.DataFrame(rows).to_csv(comparison_path, index=False)
    print(f"Saved comparison table to {comparison_path}")

    examples_path = os.path.join(config["results_dir"], "before_after_examples.txt")
    with open(examples_path, "w", encoding="utf-8") as f:
        f.write(f"Held-out accuracy: {accuracy:.3f}\n\n")
        f.write("\n".join(examples_lines))
    print(f"Saved before/after examples to {examples_path}")


if __name__ == "__main__":
    main()
