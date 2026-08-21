"""
WP4: Categorical class summaries for DPG
=========================================
Turns the existing numeric-interval class summary (GraphMetrics.extract_class_boundaries,
which treats every predicate - one-hot dummy columns included - as a plain numeric
threshold) into a summary that speaks in category terms for one-hot encoded features,
while leaving genuinely numeric/ordinal/target-encoded features untouched.

This module only reads a live DPGExplanation (graph + nodes + target_names); it does
not modify anything in dpg/ or metrics/. It reuses GraphMetrics.clustering - the same
class<->node association already used internally by extract_class_boundaries - instead
of re-deriving which predicate nodes belong to which class.
"""
import re
import math

from metrics.graph import GraphMetrics



def is_categorical_predicate(label):
    """A predicate node is treated as a one-hot dummy column iff it has the
    form '<base>_<category> <= 0.5' or '<base>_<category> > 0.5' - the only
    threshold a tree can use to split a 0/1 dummy column - and the column
    name splits at its last underscore into a base name and a non-numeric
    category (so plain numeric columns like 'feature_2' are not mistaken
    for one-hot columns). Mirrors the detection rule from WP3's
    categorical_view_conversion.py.
    """
    return parse_onehot_predicate(label) is not None



def parse_onehot_predicate(label):
    """Return (base_feature, category, direction) for a one-hot predicate
    node, where direction is 'in' (feature_CAT > 0.5, i.e. feature IN {CAT})
    or 'out' (feature_CAT <= 0.5, i.e. feature NOT IN {CAT}). Returns None
    for anything else (numeric predicates, class/leaf nodes, malformed
    labels).
    """
    match = re.match(r"^\s*(.+?)\s*(<=|>)\s*([-+]?\d*\.?\d+)\s*$", str(label))
    if not match:
        return None
    column, operator, threshold_str = match.groups()
    try:
        threshold = float(threshold_str)
    except ValueError:
        return None
    if threshold != 0.5:
        return None

    if "_" not in column:
        return None
    base, category = column.rsplit("_", 1)
    if not base or not category:
        return None
    # A genuinely numeric column split at its last underscore (e.g.
    # "feature_2") would give a purely numeric "category" - not a real
    # category name, so treat it as a non-match.
    try:
        float(category)
        return None
    except ValueError:
        pass

    direction = "in" if operator == ">" else "out"
    return base, category, direction



def node_support(graph, node_id):
    """How many original decision-path instances passed through this node,
    approximated as the total weight of its outgoing edges (every
    non-leaf predicate node has at least one)."""
    return sum(
        float(data.get("weight", 1.0))
        for _, _, data in graph.out_edges(node_id, data=True)
    )



def build_category_support_table(graph, nodes, target_names, threshold=0.2):
    """Associate each one-hot categorical predicate node with the class(es)
    it supports, using the same clustering-based class<->node association
    GraphMetrics.extract_class_boundaries relies on internally.

    Returns {class_name: {base_feature: {category: {"in": support, "out": support}}}}
    and, alongside it, {node_label: {class_name: probability}} (the purity
    signal from GraphMetrics.clustering, reused as-is rather than
    recomputed).
    """
    node_label_to_id = {label: node_id for node_id, label in nodes if "->" not in node_id}
    node_id_to_label = {v: k for k, v in node_label_to_id.items()}
    class_nodes = {
        node_id: label
        for label, node_id in node_label_to_id.items()
        if str(label).startswith("Class ")
    }

    table = {}
    node_probs_by_label = {}
    if not class_nodes:
        return table, node_probs_by_label

    clusters, node_prob, _confidence = GraphMetrics.clustering(graph, class_nodes, threshold)
    node_probs_by_label = {
        node_id_to_label.get(node_id, str(node_id)): probs
        for node_id, probs in node_prob.items()
    }

    for class_label, member_ids in clusters.items():
        if str(class_label).lower() == "ambiguous":
            continue
        class_name = GraphMetrics._normalize_class_label(class_label)
        feature_table = table.setdefault(class_name, {})
        for node_id in member_ids:
            label = node_id_to_label.get(node_id, node_id_to_label.get(str(node_id)))
            if label is None:
                continue
            parsed = parse_onehot_predicate(label)
            if parsed is None:
                continue
            base, category, direction = parsed
            support = node_support(graph, node_id)
            entry = feature_table.setdefault(base, {}).setdefault(
                category, {"in": 0.0, "out": 0.0}
            )
            entry[direction] += support

    return table, node_probs_by_label



def filter_rare_categories(feature_table, min_support_fraction=0.05):
    """Bucket categories whose total support (in + out) falls under
    min_support_fraction of the feature's total support into "OTHER",
    so a long tail of rare categories can't dominate the summary.
    Returns (kept, other_totals, other_count).
    """
    total = sum(v["in"] + v["out"] for v in feature_table.values())
    if total == 0:
        return {}, {"in": 0.0, "out": 0.0}, 0

    kept, other = {}, {"in": 0.0, "out": 0.0}
    other_count = 0
    for category, values in feature_table.items():
        share = (values["in"] + values["out"]) / total
        if share >= min_support_fraction:
            kept[category] = values
        else:
            other["in"] += values["in"]
            other["out"] += values["out"]
            other_count += 1
    return kept, other, other_count



def summarize_feature_for_class(base_feature, feature_table, min_support_fraction=0.05,
                                 dominance_threshold=0.5):
    """Turn one (class, base_feature)'s raw support table into dominant
    value / inclusion set / exclusion set facts. Returns None if nothing
    survives filtering."""
    kept, other, other_count = filter_rare_categories(feature_table, min_support_fraction)
    total = sum(v["in"] + v["out"] for v in kept.values()) + other["in"] + other["out"]
    if total == 0:
        return None

    dominant = None
    if kept:
        best_category, best = max(kept.items(), key=lambda kv: kv[1]["in"])
        if best["in"] / total >= dominance_threshold:
            dominant = (best_category, best["in"] / total)

    # Inclusion/exclusion skip whatever the dominant statement already said,
    # and a category already flagged "in" is not also flagged "out" for the
    # same class - avoids asserting the same fact twice (see redundancy
    # measurement in the experiment script).
    inclusion = [
        category for category, v in kept.items()
        if v["in"] > 0 and (dominant is None or category != dominant[0])
    ]
    exclusion = [category for category, v in kept.items() if v["out"] > 0 and v["in"] == 0]

    return {
        "feature": base_feature,
        "dominant": dominant,
        "inclusion": inclusion,
        "exclusion": exclusion,
        "other_count": other_count,
        "total_support": total,
    }



def contrastive_categories(feature_table_a, feature_table_b, top_n=2, min_gap=0.15):
    """Compare the same base_feature's raw (pre-filter) support between two
    classes and return the categories with the largest share difference -
    the "this class leans towards CAT far more than the other" statements.
    Only returns gaps of at least min_gap (a share-difference floor, so
    near-identical distributions produce no contrastive statement)."""
    categories = set(feature_table_a) | set(feature_table_b)
    total_a = sum(v["in"] for v in feature_table_a.values()) or 1.0
    total_b = sum(v["in"] for v in feature_table_b.values()) or 1.0

    diffs = []
    for category in categories:
        share_a = feature_table_a.get(category, {"in": 0.0})["in"] / total_a
        share_b = feature_table_b.get(category, {"in": 0.0})["in"] / total_b
        diffs.append((category, share_a, share_b, share_a - share_b))

    diffs = [d for d in diffs if abs(d[3]) >= min_gap]
    diffs.sort(key=lambda d: abs(d[3]), reverse=True)
    return diffs[:top_n]



def numeric_boundary_feature_name(boundary):
    """Extract the feature name out of a formatted boundary string, in any
    of the three shapes GraphMetrics.calculate_class_boundaries /
    extract_class_boundaries can produce."""
    match = re.match(r"^\s*[-\d.eE+]+\s*<\s*(.+?)\s*<=", str(boundary))
    if match:
        return match.group(1).strip()
    match = re.match(r"^\s*(.+?)\s*(<=|>)\s*[-\d.eE+]+\s*$", str(boundary))
    if match:
        return match.group(1).strip()
    return None



def build_new_style_summary(explanation, min_support_fraction=0.05,
                             dominance_threshold=0.5, contrastive_top_n=2,
                             clustering_threshold=0.2):
    """Top-level entry point: build the new categorical-aware class summary
    for every class in `explanation`.

    Returns {class_name: {"text": str, "structured": {...}}}.
    """
    old_bounds = explanation.class_boundaries.get("Class Bounds", {})
    support_table, node_probs = build_category_support_table(
        explanation.graph, explanation.nodes, None, threshold=clustering_threshold
    )

    class_names = list(support_table.keys()) or [
        GraphMetrics._normalize_class_label(k) for k in old_bounds.keys()
    ]

    results = {}
    for class_name in class_names:
        old_key = f"Class {class_name}" if not str(class_name).startswith("Class ") else class_name
        boundaries = old_bounds.get(old_key, [])

        # Keep a boundary as "numeric" iff it does NOT itself parse as a
        # one-hot dummy predicate - those are superseded by the
        # categorical statements built below.
        numeric_boundaries = [b for b in boundaries if parse_onehot_predicate(b) is None]

        feature_summaries = []
        for base_feature, feature_table in support_table.get(class_name, {}).items():
            summary = summarize_feature_for_class(
                base_feature, feature_table, min_support_fraction, dominance_threshold
            )
            if summary:
                feature_summaries.append(summary)

        contrastive_notes = []
        for base_feature, feature_table in support_table.get(class_name, {}).items():
            for other_class in class_names:
                if other_class == class_name:
                    continue
                other_table = support_table.get(other_class, {}).get(base_feature)
                if not other_table:
                    continue
                for category, share_here, share_other, gap in contrastive_categories(
                    feature_table, other_table, top_n=contrastive_top_n
                ):
                    if gap > 0:  # only report gaps in THIS class's favor to avoid double-reporting
                        contrastive_notes.append(
                            f"{base_feature} = {category} is more common here ({share_here:.0%}) "
                            f"than in Class {other_class} ({share_other:.0%})"
                        )

        text = render_class_summary(class_name, numeric_boundaries, feature_summaries, contrastive_notes)
        results[class_name] = {
            "text": text,
            "structured": {
                "numeric_boundaries": numeric_boundaries,
                "categorical_features": feature_summaries,
                "contrastive_notes": contrastive_notes,
            },
        }
    return results



def render_class_summary(class_name, numeric_boundaries, feature_summaries, contrastive_notes):
    """Compact text template combining unchanged numeric bounds with the
    new categorical statements."""
    lines = [f"Class {class_name}:"]
    if numeric_boundaries:
        lines.append("  Numeric: " + "; ".join(numeric_boundaries) + ".")

    for fs in feature_summaries:
        parts = []
        if fs["dominant"]:
            category, share = fs["dominant"]
            parts.append(f"{fs['feature']} is mainly {category} ({share:.0%})")
        if fs["inclusion"]:
            parts.append(f"{fs['feature']} in {{{', '.join(fs['inclusion'])}}}")
        if fs["exclusion"]:
            parts.append(f"{fs['feature']} never {{{', '.join(fs['exclusion'])}}}")
        if fs["other_count"]:
            parts.append(f"{fs['other_count']} rare {fs['feature']} categories grouped as OTHER")
        if parts:
            lines.append("  " + "; ".join(parts) + ".")

    for note in contrastive_notes:
        lines.append(f"  {note}.")

    return "\n".join(lines)
