"""Semantic-equivalence checks between DPG grouping variants.

``run_all_metrics.py``'s ``metric_validity`` only checks graph structure
(has a class sink, no orphan nodes) -- it says nothing about whether a
grouped DPG still *predicts* the same way as the original BASIC DPG.

``examples/quickstart_categorical_WP5.py`` already has the traversal +
fidelity machinery to answer that (``predict_with_dpg`` / ``compute_fidelity``),
but its predicate parser only understands plain numeric labels
(``feature <= t``). It doesn't know how to evaluate the categorical
``base IN {cats}`` / ``base NOT IN {cats}`` labels ``cat_grouping*.py``
writes, or the ``AND``-joined conjunctions ``cat_grouping_split_conjunction.py``
produces. This module extends that evaluator to understand both grammars
so the exact same traversal logic runs unmodified on BASIC and every
grouped variant, and adds a direct variant-vs-variant agreement check
(the real equivalence signal -- two variants can each independently match
the model's fidelity without ever agreeing with each other on the same
sample).
"""

from __future__ import annotations

import inspect
import json
import re
from typing import Dict, List, Optional, Sequence, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from networkx.readwrite import json_graph

# Same grammar as categorical/cat_grouping.py's _PREDICATE_PATTERN / _IN_PATTERN.
_PREDICATE_PATTERN = re.compile(
    r"^\s*(?P<feature>.+?)\s*(?P<op><=|>|<|>=|==|!=)\s*(?P<value>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*$"
)
_IN_PATTERN = re.compile(
    r"^\s*(?P<base>.+?)\s+(?P<op>IN|NOT\s+IN)\s+\{(?P<cats>[^}]+)\}\s*$"
)


# ---------------------------------------------------------------------------
# Structure JSON -> networkx graph (same shape as run_all_metrics.py's
# _nx_graph / _real_nodes, duplicated here so this module has no
# orchestration-script dependency).
# ---------------------------------------------------------------------------


def load_structure_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def real_nodes(structure: dict) -> List[Tuple[str, str]]:
    """(id, label) pairs for real predicate/class nodes -- drops the
    empty-label pseudo-nodes the BASIC dump stores per edge."""
    return [
        (str(n["id"]), n["label"])
        for n in structure.get("nodes", [])
        if n.get("label") and "->" not in str(n["id"])
    ]


def structure_to_graph(structure: dict) -> nx.DiGraph:
    """Rebuild a DiGraph from a DPG structure JSON's ``graph`` field.
    NetworkX has renamed ``json_graph.node_link_graph``'s edges-list kwarg
    across versions (``edges`` in <3.4, ``link`` in 3.4-3.5, default
    flipped in 3.6+), so the right kwarg is picked at runtime."""
    graph = structure.get("graph", {})
    params = inspect.signature(json_graph.node_link_graph).parameters
    if "link" in params:
        return json_graph.node_link_graph(graph, link="links")
    if "edges" in params:
        return json_graph.node_link_graph(graph, edges="links")
    return json_graph.node_link_graph(graph)


def load_variant(structure_path: str) -> Tuple[nx.DiGraph, List[Tuple[str, str]]]:
    """Convenience: load a structure JSON straight into (graph, nodes),
    ready for ``predict_with_dpg``."""
    structure = load_structure_json(structure_path)
    return structure_to_graph(structure), real_nodes(structure)


# ---------------------------------------------------------------------------
# Sample rows
# ---------------------------------------------------------------------------


def build_eval_row(raw_row: pd.Series, encoded_row: pd.Series) -> pd.Series:
    """Combine one sample's raw (pre-one-hot) values with its one-hot
    dummy values into a single row.

    BASIC-DPG labels reference one-hot dummy columns (e.g.
    ``gender_Male <= 0.5``); grouped-variant labels reference the
    original categorical column (e.g. ``gender NOT IN {Male}``); numeric
    (non-categorical) predicates reference the same raw column name in
    both. One combined row lets a single predicate label, whichever
    grammar it uses, always be evaluated.

    Numeric columns survive one-hot encoding untouched, so they appear in
    BOTH inputs under the same name. The encoded value wins those
    collisions: it is the rounded / NaN-filled value the trees were
    actually fitted on, so it is what the graph's numeric thresholds were
    derived from. Leaving both would give the row a duplicated index and
    make ``sample_row[feature]`` return a Series instead of a scalar.
    """
    combined = pd.concat([raw_row, encoded_row])
    return combined[~combined.index.duplicated(keep="last")]


def load_eval_rows(
    csv_path: str,
    drop_columns: Optional[Sequence[str]] = None,
    n_samples: Optional[int] = 200,
    random_state: int = 27,
) -> List[pd.Series]:
    """Load a dataset CSV straight into rows ready for graph walking.

    Applies the same preprocessing the training pipeline uses (delimiter
    sniffing, dropped ID/free-text columns, rows with a missing target
    removed, one-hot encoding, rounding), then pairs each raw row with
    its encoded counterpart via ``build_eval_row``.

    Note these rows do NOT have to be the same rows the DPG was built
    from. The decision/explanation comparisons are graph-against-graph:
    both variants are walked with identical inputs, so any representative
    sample of the dataset answers the question "do these two graphs
    behave the same". Only fidelity-against-the-model would require the
    original split, and that needs the trained model anyway.

    ``n_samples`` draws a reproducible random subset (all rows if None or
    if the dataset is smaller).
    """
    df = pd.read_csv(csv_path, sep=None, engine="python", encoding="utf-8-sig")

    if drop_columns:
        present = [c for c in drop_columns if c in df.columns]
        if present:
            df = df.drop(columns=present)

    target_col = df.columns[-1]
    df = df.dropna(subset=[target_col]).reset_index(drop=True)

    features_raw = df.iloc[:, :-1]
    features_enc = pd.get_dummies(features_raw, drop_first=False)
    features_enc = features_enc.replace([np.inf, -np.inf], np.nan).fillna(features_enc.mean())
    features_enc = np.round(features_enc, 3)

    if n_samples is not None and len(features_raw) > n_samples:
        index = features_raw.sample(n=n_samples, random_state=random_state).index
    else:
        index = features_raw.index

    return [build_eval_row(features_raw.loc[i], features_enc.loc[i]) for i in index]


def compare_variants(
    graph_a: nx.DiGraph,
    nodes_a: Sequence[Tuple[str, str]],
    graph_b: nx.DiGraph,
    nodes_b: Sequence[Tuple[str, str]],
    eval_rows: Sequence[pd.Series],
) -> Dict[str, object]:
    """Full graph-against-graph comparison of two DPG variants.

    Walks both graphs over the same rows and returns the decision-level
    and explanation-level numbers together:
    ``decision_agreement``, ``explanation_agreement``,
    ``explanation_overlap``, ``explanation_partial``.

    No model is needed -- this compares two graphs to each other, not
    either graph to the RandomForest.
    """
    preds_a = dpg_predictions(graph_a, nodes_a, eval_rows)
    preds_b = dpg_predictions(graph_b, nodes_b, eval_rows)
    routes_a = [dpg_routes(graph_a, nodes_a, row) for row in eval_rows]
    routes_b = [dpg_routes(graph_b, nodes_b, row) for row in eval_rows]

    result: Dict[str, object] = {}
    result.update(compute_agreement(preds_a, preds_b))
    result.update(compute_path_agreement(routes_a, routes_b))
    return result


# ---------------------------------------------------------------------------
# Predicate evaluation
# ---------------------------------------------------------------------------


def _clause_holds(clause: str, sample_row: pd.Series) -> Optional[bool]:
    """Evaluate one predicate clause (no ``AND``). Returns None if it
    can't be parsed or references a column missing from ``sample_row``."""
    m = _IN_PATTERN.match(clause)
    if m is not None:
        base = m.group("base").strip()
        op = "NOT IN" if m.group("op").upper().startswith("NOT") else "IN"
        cats = {c.strip() for c in m.group("cats").split(",") if c.strip()}
        if base not in sample_row.index:
            return None
        is_in = str(sample_row[base]) in cats
        return is_in if op == "IN" else not is_in

    parsed = _PREDICATE_PATTERN.match(clause)
    if parsed is None:
        return None
    feature = parsed.group("feature").strip()
    op = parsed.group("op")
    value = float(parsed.group("value"))
    if feature not in sample_row.index:
        return None
    sample_val = sample_row[feature]
    if op == "<=":
        return bool(sample_val <= value)
    if op == "<":
        return bool(sample_val < value)
    if op == ">":
        return bool(sample_val > value)
    if op == ">=":
        return bool(sample_val >= value)
    if op == "==":
        return bool(sample_val == value)
    if op == "!=":
        return bool(sample_val != value)
    return None


def node_holds_for_sample(label: str, sample_row: pd.Series) -> Optional[bool]:
    """True/False if ``sample_row`` satisfies this predicate node's
    condition, or None if unknown (class/leaf node, unparseable label, or
    a referenced column missing from ``sample_row``).

    Understands three label grammars: plain numeric (``feature <= t``),
    categorical membership (``base IN {cats}`` / ``base NOT IN {cats}``),
    and ``AND``-joined chains of either (cat_grouping_split_conjunction.py
    output). Every clause must hold for the node to hold; a clause that
    evaluates to False short-circuits to False even if another clause in
    the same label is unparseable, since the node provably does not hold
    regardless of that unknown.
    """
    label = str(label)
    if label.startswith("Class "):
        return None

    saw_unknown = False
    for clause in label.split(" AND "):
        result = _clause_holds(clause.strip(), sample_row)
        if result is False:
            return False
        if result is None:
            saw_unknown = True
    return None if saw_unknown else True


# ---------------------------------------------------------------------------
# Graph traversal / prediction
# ---------------------------------------------------------------------------


def predict_with_dpg(
    graph: nx.DiGraph, nodes: Sequence[Tuple[str, str]], sample_row: pd.Series
) -> Optional[str]:
    """Predict a class for one sample by walking a DPG graph.

    Generic over BASIC and every grouped variant -- it only depends on
    ``node_holds_for_sample`` understanding the label grammar in play.
    Mirrors ``examples/quickstart_categorical_WP5.predict_with_dpg``
    exactly, decoupled from the ``DPGExplainer`` explanation object so it
    also runs on a graph rebuilt from a grouped-variant structure JSON.

    Starts only from in-degree-0 nodes whose condition the sample
    satisfies, follows an edge only while the sample also satisfies the
    next node's condition, until a class node is reached (a per-path
    visited-set guards against the cycles a merged DPG can contain). If
    several paths reach different classes, the one backed by the larger
    total edge weight wins. Returns None if no path matches at all.
    """
    label_by_id: Dict[str, str] = {str(nid): label for nid, label in nodes}

    class_weights: Dict[str, float] = {}
    for node_id, label in nodes:
        node_id = str(node_id)
        label = str(label)
        if label.startswith("Class "):
            continue
        if node_id not in graph or graph.in_degree(node_id) != 0:
            continue
        if node_holds_for_sample(label, sample_row) is not True:
            continue

        stack = [(node_id, frozenset([node_id]))]
        while stack:
            current_id, visited = stack.pop()
            for _, next_id, edge_data in graph.out_edges(current_id, data=True):
                next_id = str(next_id)
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


# ---------------------------------------------------------------------------
# Route canonicalisation ("does it EXPLAIN the same?", not just "decide")
# ---------------------------------------------------------------------------


def _split_one_hot_column(col: str) -> Optional[Tuple[str, str]]:
    """``person_home_ownership_OWN`` -> ``('person_home_ownership', 'OWN')``.
    None for non-OHE columns (no underscore, or a numeric trailing token
    like ``x_1``). Same rule as cat_grouping._split_one_hot_column."""
    if "_" not in col:
        return None
    base, category = col.rsplit("_", 1)
    base = base.strip()
    category = category.strip()
    if not base or not category:
        return None
    try:
        float(category)
    except ValueError:
        return base, category
    return None


def _canonical_clause(clause: str) -> Optional[tuple]:
    """Rewrite one clause into a variant-independent form, so the same
    condition written in either grammar compares equal.

    ``person_gender_female <= 0.5`` and ``person_gender NOT IN {female}``
    both become ``("cat", "person_gender", "NOT IN", {"female"})``.
    Numeric predicates (untouched by grouping) become
    ``("num", feature, op, threshold)``. Returns None if unparseable.
    """
    m = _IN_PATTERN.match(clause)
    if m is not None:
        base = m.group("base").strip()
        op = "NOT IN" if m.group("op").upper().startswith("NOT") else "IN"
        cats = frozenset(c.strip() for c in m.group("cats").split(",") if c.strip())
        return ("cat", base, op, cats)

    parsed = _PREDICATE_PATTERN.match(clause)
    if parsed is None:
        return None
    feature = parsed.group("feature").strip()
    op = parsed.group("op")
    value = float(parsed.group("value"))

    # A one-hot dummy split at 0.5 is really a categorical membership test
    # -- the exact rewrite cat_grouping._to_categorical_label performs.
    split = _split_one_hot_column(feature)
    if split is not None and abs(value - 0.5) <= 1e-9:
        base, category = split
        cat_op = "IN" if op in (">", ">=") else "NOT IN"
        return ("cat", base, cat_op, frozenset({category}))

    return ("num", feature, op, value)


def canonical_constraints(labels: Sequence[str]) -> frozenset:
    """Turn the labels visited along one route into a canonical set of
    constraints.

    Two normalisations happen here, and both are needed for a fair
    comparison between variants:

    1. Every clause is rewritten into a grammar-independent form (see
       ``_canonical_clause``), so one-hot and ``IN``/``NOT IN`` spellings
       of the same test collapse together.
    2. Clauses on the same feature with the same operator are merged by
       unioning their categories -- exactly what ``cat_grouping``'s chain
       collapsing does. So BASIC visiting ``X NOT IN {OWN}`` then
       ``X NOT IN {RENT}`` as two separate nodes yields the same
       constraint as GROUPED's single ``X NOT IN {OWN, RENT}`` node.

    Without step 2 the metric would report every successful chain
    collapse as a difference, i.e. it would punish grouping for doing its
    job.
    """
    cat_buckets: Dict[Tuple[str, str], set] = {}
    others = set()
    for label in labels:
        for clause in str(label).split(" AND "):
            canon = _canonical_clause(clause.strip())
            if canon is None:
                continue
            if canon[0] == "cat":
                _, base, op, cats = canon
                cat_buckets.setdefault((base, op), set()).update(cats)
            else:
                others.add(canon)

    merged = {("cat", base, op, frozenset(cats)) for (base, op), cats in cat_buckets.items()}
    return frozenset(merged | others)


def dpg_routes(
    graph: nx.DiGraph,
    nodes: Sequence[Tuple[str, str]],
    sample_row: pd.Series,
    max_routes: int = 500,
) -> Tuple[set, bool]:
    """Every complete route this sample can take through the graph.

    Same walk as ``predict_with_dpg``, but instead of collapsing
    everything into one predicted class it records what each route
    actually *tested* on the way. Returns
    ``({(constraints, class), ...}, truncated)`` where ``constraints`` is
    the canonical constraint set from ``canonical_constraints``.

    ``max_routes`` caps the enumeration: a densely shared DPG can offer a
    large number of distinct routes for one row, and the flag tells the
    caller the answer is partial rather than silently wrong.
    """
    label_by_id: Dict[str, str] = {str(nid): label for nid, label in nodes}
    routes: set = set()
    truncated = False

    for node_id, label in nodes:
        node_id = str(node_id)
        label = str(label)
        if label.startswith("Class "):
            continue
        if node_id not in graph or graph.in_degree(node_id) != 0:
            continue
        if node_holds_for_sample(label, sample_row) is not True:
            continue

        stack = [(node_id, frozenset([node_id]), (label,))]
        while stack:
            current_id, visited, seen_labels = stack.pop()
            for _, next_id, _data in graph.out_edges(current_id, data=True):
                next_id = str(next_id)
                if next_id in visited:
                    continue
                next_label = str(label_by_id.get(next_id, ""))
                if next_label.startswith("Class "):
                    routes.add(
                        (canonical_constraints(seen_labels), next_label[len("Class "):])
                    )
                    if len(routes) >= max_routes:
                        return routes, True
                elif node_holds_for_sample(next_label, sample_row) is True:
                    stack.append(
                        (next_id, visited | {next_id}, seen_labels + (next_label,))
                    )

    return routes, truncated


# ---------------------------------------------------------------------------
# Fidelity (DPG vs model) and agreement (DPG variant vs DPG variant)
# ---------------------------------------------------------------------------


def dpg_predictions(
    graph: nx.DiGraph, nodes: Sequence[Tuple[str, str]], eval_rows: Sequence[pd.Series]
) -> List[Optional[str]]:
    return [predict_with_dpg(graph, nodes, row) for row in eval_rows]


def compute_fidelity(
    dpg_preds: Sequence[Optional[str]], model_preds: Sequence[str]
) -> Dict[str, object]:
    """How often a DPG's own predictions agree with the model's
    predictions, over the same samples in the same order.

    Reported as two numbers: ``fidelity_coverage`` (fraction of samples
    the graph could classify at all -- a heavily pruned/grouped graph can
    have low coverage without that meaning its covered predictions are
    wrong) and ``fidelity`` (agreement rate among covered samples only).
    """
    if len(dpg_preds) != len(model_preds):
        raise ValueError("dpg_preds and model_preds must be the same length")
    covered = [i for i, p in enumerate(dpg_preds) if p is not None]
    coverage = len(covered) / len(dpg_preds) if dpg_preds else float("nan")
    if covered:
        agreements = sum(1 for i in covered if dpg_preds[i] == str(model_preds[i]))
        fidelity = agreements / len(covered)
    else:
        fidelity = float("nan")
    return {
        "fidelity": fidelity,
        "fidelity_coverage": coverage,
        "fidelity_n_samples": len(dpg_preds),
    }


def compute_path_agreement(
    routes_a: Sequence[Tuple[set, bool]], routes_b: Sequence[Tuple[set, bool]]
) -> Dict[str, object]:
    """Do two variants EXPLAIN each row the same way, not merely decide it
    the same way?

    ``compute_agreement`` only compares the final class, so two graphs
    that reach the same answer through entirely different predicates
    still score 1.000. This compares the canonical constraint sets of the
    routes themselves, which is the stronger claim -- in a DPG the path
    *is* the explanation.

    Reported per row over rows where both variants produced at least one
    route:

    * ``explanation_agreement`` -- fraction of rows whose full set of
      routes is identical between the two variants. The counterpart to
      ``compute_agreement``'s ``decision_agreement``.
    * ``explanation_overlap``   -- mean Jaccard overlap of the two route
      sets, giving partial credit when most but not all routes line up.
    * ``explanation_partial``   -- fraction of rows where route
      enumeration hit the cap in either variant, so that row's
      comparison is partial rather than complete.
    """
    if len(routes_a) != len(routes_b):
        raise ValueError("routes_a and routes_b must be the same length (same samples, same order)")

    exact_hits = 0
    jaccards: List[float] = []
    truncated = 0
    compared = 0

    for (set_a, trunc_a), (set_b, trunc_b) in zip(routes_a, routes_b):
        if not set_a and not set_b:
            continue  # neither variant could route this row -- nothing to compare
        compared += 1
        if trunc_a or trunc_b:
            truncated += 1
        if set_a == set_b:
            exact_hits += 1
        union = set_a | set_b
        jaccards.append(len(set_a & set_b) / len(union) if union else float("nan"))

    return {
        "explanation_agreement": exact_hits / compared if compared else float("nan"),
        "explanation_overlap": float(np.mean(jaccards)) if jaccards else float("nan"),
        "explanation_partial": truncated / compared if compared else float("nan"),
        "explanation_n_compared": compared,
    }


def compute_agreement(
    preds_a: Sequence[Optional[str]], preds_b: Sequence[Optional[str]]
) -> Dict[str, object]:
    """Pairwise prediction agreement between two DPG variants (e.g. BASIC
    vs GROUPED) on the same samples, in the same order.

    This is the decision-level equivalence signal: fidelity-to-model
    alone can't catch two variants that each independently match the
    model ~90% of the time on *different* 90%s of the data. Only a direct
    per-sample comparison does.

    It is deliberately the *weaker* half of the pair -- it compares only
    the final class, so two graphs reaching the same answer by different
    reasoning still score 1.000. ``compute_path_agreement`` covers that
    gap with ``explanation_agreement``.
    """
    if len(preds_a) != len(preds_b):
        raise ValueError("preds_a and preds_b must be the same length (same samples, same order)")
    both_covered = [i for i in range(len(preds_a)) if preds_a[i] is not None and preds_b[i] is not None]
    coverage = len(both_covered) / len(preds_a) if preds_a else float("nan")
    if both_covered:
        agreements = sum(1 for i in both_covered if preds_a[i] == preds_b[i])
        agreement = agreements / len(both_covered)
    else:
        agreement = float("nan")
    return {
        "decision_agreement": agreement,
        "decision_agreement_coverage": coverage,
        "decision_n_samples": len(preds_a),
    }
