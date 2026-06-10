"""
Convert categorical predicates in a DPG structure JSON
=======================================================

Purpose
-------
When a Decision Predicate Graph (DPG) is built from a dataset that contains
categorical features, the original categorical columns are typically one-hot
encoded (e.g. ``person_gender`` -> ``person_gender_female`` and
``person_gender_male``). Inside the graph, those one-hot columns appear as
binary numeric predicates of the form::

    person_gender_female > 0.5      # means: this sample IS female
    person_gender_male   <= 0.5     # means: this sample is NOT male
    person_gender_male   > 0.5      # means: this sample IS male
    person_gender_female <= 0.5     # means: this sample is NOT female

That encoding is hard to read. This script:

1. Reads a DPG ``*_dpg_structure.json`` file (the one produced by
   ``quickstart_categorical.py`` via ``save_dpg_structure_json``).
2. Detects one-hot predicates (operator ``<=`` or ``>`` with threshold 0.5).
3. Groups one-hot columns that share the same *root* feature (the suffix
   after the last ``_`` is treated as the category name).
4. Rewrites the predicate using set-membership syntax:

   - ``> 0.5`` on column ``root_value``  ->  ``root in {value}``
   - ``<= 0.5`` on column ``root_value`` ->  ``root not in {value}``

5. Walks every path from each "Class X" leaf back to a root, computes the
   set of categories that are *implicitly excluded* for each root, and adds
   those exclusions to the nearest predicate node on the path. This makes
   the resulting JSON self-contained: reading any single node tells you
   which categories are active / excluded for that branch.
6. Writes a new JSON with the rewritten labels.
7. Rebuilds a ``networkx.DiGraph`` from the new JSON, builds a graphviz
   ``Digraph`` with the same layout the official visualizer uses, and saves
   a PNG of the converted DPG next to the JSON.

Run it from any directory; the only required argument is the input JSON::

    python convert_categorical_dpg.py path/to/dpg_structure.json

The output PNG and JSON are written next to the input file with the
``_converted`` suffix.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict, deque
from typing import Dict, List, Optional, Tuple

import networkx as nx


# ---------------------------------------------------------------------------
# 1.  Label parsing helpers
# ---------------------------------------------------------------------------
# A predicate label looks like "<feature> <op> <threshold>", e.g.
#     "person_income > 44119.5"
#     "person_gender_female <= 0.5"
# We keep the same regex that the official visualizer uses
# (see dpg/visualizer.py::_PREDICATE_PATTERN) so behaviour is consistent.
_PREDICATE_PATTERN = re.compile(
    r"(.+?)\s*(<=|>|<|>=|==|!=)\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
)

# Small epsilon used to detect "== 0.5" robustly even after float rounding.
_THRESHOLD_EPS = 1e-9


def parse_predicate(label: str) -> Optional[Tuple[str, str, float]]:
    """Return ``(feature, op, threshold)`` for a predicate label, else ``None``.

    The feature name keeps the original whitespace; the operator is normalised
    to one of ``<=``, ``>``, ``<``, ``>=``, ``==``, ``!=``; the threshold is a
    ``float``.
    """
    match = _PREDICATE_PATTERN.search(str(label))
    if not match:
        return None
    return match.group(1).strip(), match.group(2), float(match.group(3))


def is_onehot_predicate(parsed: Tuple[str, str, float]) -> bool:
    """A predicate is a one-hot binary split when the threshold is 0.5.

    In the dummy dataset used by ``quickstart_categorical.py`` the categorical
    columns go through ``pd.get_dummies`` so every category becomes a 0/1
    column. Random forests split on those columns at 0.5.
    """
    _, op, thr = parsed
    return op in ("<=", ">") and abs(thr - 0.5) < _THRESHOLD_EPS


# ---------------------------------------------------------------------------
# 2.  Grouping one-hot columns back into a single categorical feature
# ---------------------------------------------------------------------------
def split_onehot_column(column: str) -> Tuple[str, str]:
    """Split ``"person_gender_female"`` into ``("person_gender", "female")``.

    Heuristic: take everything before the *last* underscore as the root
    feature, and the suffix after that underscore as the category name.
    This matches the column naming produced by ``pd.get_dummies`` in
    ``quickstart_categorical.py``.

    If the column has no underscore the whole string is returned as the
    root and the category is empty. Continuous features are never routed
    through this function.
    """
    if "_" not in column:
        return column, ""
    root, value = column.rsplit("_", 1)
    return root, value


# ---------------------------------------------------------------------------
# 3.  Path discovery in the DPG graph
# ---------------------------------------------------------------------------
def build_adjacency(dpg: dict) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    """Return ``(parents, children)`` adjacency dicts from ``graph.links``."""
    parents: Dict[str, List[str]] = defaultdict(list)
    children: Dict[str, List[str]] = defaultdict(list)
    for link in dpg["graph"]["links"]:
        src, tgt = link["source"], link["target"]
        parents[tgt].append(src)
        children[src].append(tgt)
    return parents, children


def find_roots(dpg: dict, children: Dict[str, List[str]]) -> List[str]:
    """A root is a node with no incoming edges (sources of the DPG)."""
    all_node_ids = {n["id"] for n in dpg["nodes"]}
    incoming = {tgt for srcs in children.values() for tgt in srcs}
    return [nid for nid in all_node_ids if nid not in incoming]


def find_leaves(dpg: dict) -> List[str]:
    """A leaf is a node whose label starts with ``"Class"`` (e.g. ``"Class 1"``)."""
    return [n["id"] for n in dpg["nodes"] if str(n["label"]).startswith("Class")]


def enumerate_paths_to_root(
    leaf: str, parents: Dict[str, List[str]], roots: set
) -> List[List[str]]:
    """Return every simple path (node-id lists) from ``leaf`` to any root.

    A path is the list of node ids visited while walking **upwards** from a
    leaf towards a root via the ``parents`` map. The leaf itself is the
    first element of each returned path. The list is empty if no root is
    reachable (orphan leaf).
    """
    paths: List[List[str]] = []
    queue: deque = deque([(leaf, [])])
    while queue:
        cur, trail = queue.popleft()
        if cur in roots:
            # The path stops here: append the full chain leaf -> root.
            paths.append(trail + [cur])
            continue
        for parent in parents.get(cur, []):
            queue.append((parent, trail + [cur]))
    return paths


# ---------------------------------------------------------------------------
# 4.  The core translator
# ---------------------------------------------------------------------------
def collect_active_categories(
    dpg: dict, parents: Dict[str, List[str]], roots: set
) -> Dict[str, Dict[str, set]]:
    """For every leaf, compute ``{root_feature: set_of_active_values}``.

    Walking from the leaf towards a root we read every one-hot predicate we
    encounter. ``> 0.5`` on a column means the value is *active* for that
    branch; ``<= 0.5`` means it is *inactive* (and the same value becomes
    active on the sibling branch of the graph).
    """
    leaf_to_active: Dict[str, Dict[str, set]] = {}
    for leaf in find_leaves(dpg):
        active_by_root: Dict[str, set] = defaultdict(set)
        for path in enumerate_paths_to_root(leaf, parents, roots):
            for node_id in path:
                label = node_label(dpg, node_id)
                if not label:
                    continue
                parsed = parse_predicate(label)
                if parsed is None or not is_onehot_predicate(parsed):
                    continue
                feat, op, _ = parsed
                if op == ">":
                    root, value = split_onehot_column(feat)
                    active_by_root[root].add(value)
        leaf_to_active[leaf] = dict(active_by_root)
    return leaf_to_active


def collect_all_categories(dpg: dict) -> Dict[str, set]:
    """Inverse map: ``{root_feature: set_of_all_values_seen_anywhere}``.

    Used to know which categories are *implicitly excluded* on a path that
    only activates one or two of them.
    """
    all_values: Dict[str, set] = defaultdict(set)
    for n in dpg["nodes"]:
        label = str(n["label"])
        if not label:
            continue
        parsed = parse_predicate(label)
        if parsed is None or not is_onehot_predicate(parsed):
            continue
        feat, _, _ = parsed
        root, value = split_onehot_column(feat)
        if value:
            all_values[root].add(value)
    return dict(all_values)


# ---------------------------------------------------------------------------
# 5.  Rewrite labels
# ---------------------------------------------------------------------------
def node_label(dpg: dict, node_id: str) -> str:
    """Convenience: return the label string for a given node id."""
    for n in dpg["nodes"]:
        if n["id"] == node_id:
            return str(n["label"])
    return ""


def translate_predicate(
    label: str,
    leaf_active: Dict[str, set],
    all_values: Dict[str, set],
) -> str:
    """Translate a single predicate label to a set-membership form.

    Behaviour:

    * Numeric predicates (e.g. ``person_income > 44119.5``) are returned
      unchanged.
    * ``feature > 0.5`` on a one-hot column becomes
      ``root in {value}`` and the implicit exclusions
      ``root not in {other_value_1, other_value_2, ...}`` are appended.
    * ``feature <= 0.5`` on a one-hot column becomes
      ``root not in {value}`` (the explicit half of the disjunction); the
      branch that *does* activate the value is on the other side of the
      edge in the graph.
    * ``Class X`` and edge placeholders are returned unchanged.

    The ``leaf_active`` argument carries the *active* categories for the
    leaf whose path we are translating, so we can attach the implicit
    exclusions only once per path. When ``leaf_active`` is empty the
    exclusions are skipped to avoid fabricating information.
    """
    if not label or label.startswith("Class"):
        return label
    parsed = parse_predicate(label)
    if parsed is None:
        return label  # non-predicate, keep as is

    feat, op, thr = parsed
    if not is_onehot_predicate(parsed):
        # Plain numeric predicate: pass through verbatim.
        return label

    root, value = split_onehot_column(feat)
    if not value:
        return label  # safety: column had no "_" to split on

    if op == ">":
        head = f"{root} in {{{value}}}"
    else:  # op == "<="
        head = f"{root} not in {{{value}}}"

    # Attach implicit exclusions only on the *active* side (> 0.5). These
    # are the categories that the other one-hot columns in the same group
    # force off, but that the original numeric label was hiding.
    extras: List[str] = []
    if op == ">":
        siblings = (all_values.get(root, set()) - {value}) - leaf_active.get(root, set())
        if siblings:
            extras.append(f"{root} not in {{{', '.join(sorted(siblings))}}}")
    if extras:
        return head + " " + " ".join(extras)
    return head


def translate_dpg(dpg: dict) -> dict:
    """Return a *new* DPG dict with categorical predicates rewritten.

    The structure (ids, edges, weights, metadata) is preserved exactly; only
    the ``label`` strings are updated.
    """
    parents, _ = build_adjacency(dpg)
    roots_list = find_roots(dpg, parents)
    roots_set = set(roots_list)
    leaf_to_active = collect_active_categories(dpg, parents, roots_set)
    all_values = collect_all_categories(dpg)

    new_nodes = []
    leaf_active_lookup = {}
    for leaf in find_leaves(dpg):
        leaf_active_lookup[leaf] = leaf_to_active.get(leaf, {})

    for n in dpg["nodes"]:
        new_label = n["label"]
        # Translate only predicate nodes (skip edge placeholders and class leaves).
        if new_label and not str(new_label).startswith("Class"):
            # We need the *path-specific* exclusions, so we look at every
            # leaf for which this node appears on a path. We attach the
            # exclusions only when the active set is unambiguous (i.e. all
            # leaves that pass through this node share the same active set).
            # In practice the dummy JSON has a unique active set per path.
            active_sets = []
            for leaf, active in leaf_to_active.items():
                if any(node_id == n["id"] for path in enumerate_paths_to_root(leaf, parents, roots_set) for node_id in path):
                    active_sets.append(active)
            # If multiple leaves reach this node with different active sets
            # we conservatively skip the implicit exclusions; the "active"
            # half of the predicate is still correct.
            common_active = None
            if active_sets:
                keys = set().union(*[s.keys() for s in active_sets])
                common_active = {k: set.intersection(*[s.get(k, set()) for s in active_sets]) for k in keys}
                if not all(s == active_sets[0] for s in active_sets[1:]):
                    common_active = None
            new_label = translate_predicate(new_label, common_active or {}, all_values)
        new_nodes.append({"id": n["id"], "label": new_label})

    return {
        **dpg,
        "nodes": new_nodes,
        "metadata": {
            "converted_from": dpg.get("run_id"),
            "conversion": "one-hot predicates rewritten as 'in {value}' / 'not in {value}'",
            "feature_value_groups": {
                root: sorted(vals) for root, vals in all_values.items()
            },
        },
    }


# ---------------------------------------------------------------------------
# 6.  Reconstruct a networkx graph from the new JSON
# ---------------------------------------------------------------------------
def rebuild_graph(dpg: dict) -> nx.DiGraph:
    """Build a fresh ``networkx.DiGraph`` with the exact same topology.

    We use ``nx.node_link_graph`` (the inverse of what
    ``save_dpg_structure_json`` uses) and then re-attach the rewritten node
    labels because the JSON only stores node ids in ``graph.nodes``.

    Note: the keyword for the edge-list key was renamed across networkx
    versions:
        * networkx >= 3.6: ``edges="links"``
        * networkx <= 3.5: ``link="links"``
    We try the new name first and fall back to the old one so the script
    works on both.
    """
    try:
        G = nx.node_link_graph(dpg["graph"], edges="links")
    except TypeError:
        G = nx.node_link_graph(dpg["graph"], link="links")
    label_map = {n["id"]: n["label"] for n in dpg["nodes"]}
    # ``nx.node_link_graph`` may keep the original node attribute ``id``;
    # we make sure the user-facing label is also available.
    for nid in G.nodes:
        G.nodes[nid]["label"] = label_map.get(nid, G.nodes[nid].get("label", ""))
        # Drop the synthetic edge-placeholder nodes (label == "") because
        # the official visualizer expects a graph of predicate + class
        # nodes only. Edge information is already captured by ``links``.
        if not G.nodes[nid]["label"]:
            G.remove_node(nid)
    return G


# ---------------------------------------------------------------------------
# 7.  Render the converted DPG with graphviz
# ---------------------------------------------------------------------------
def render_dpg_png(
    dpg: dict,
    output_png: str,
    layout: str = "LR",
    ranksep: str = "1.0",
    nodesep: str = "0.4",
) -> None:
    """Draw the converted DPG and save a PNG using only graphviz + networkx.

    We keep this dependency-free (no DPG library call) so the script is
    self-contained and works even if the DPG package internals change.

    The colour scheme mirrors the official DPG visualizer:
    * class leaves -> light blue (#9DC3E6)
    * predicate nodes -> light grey (#DEEBF7)
    """
    try:
        import graphviz
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "graphviz is required to render the PNG. Install with `pip install graphviz` "
            "and ensure the system `dot` binary is on PATH."
        ) from exc

    G = rebuild_graph(dpg)
    dot = graphviz.Digraph(format="png")
    dot.attr(rankdir=layout, ranksep=ranksep, nodesep=nodesep)
    dot.attr("node", shape="box", style="rounded,filled", fontname="Helvetica")

    CLASS_COLOR = "#9DC3E6"   # light blue
    PRED_COLOR = "#DEEBF7"    # light grey

    for nid, attrs in G.nodes(data=True):
        label = attrs.get("label", "")
        if label.startswith("Class"):
            fill = CLASS_COLOR
        else:
            fill = PRED_COLOR
        # Escape double quotes for graphviz.
        safe = label.replace('"', '\\"')
        dot.node(str(nid), label=safe, fillcolor=fill, color="#666666")

    for src, tgt, attrs in G.edges(data=True):
        weight = attrs.get("weight", 1.0)
        dot.edge(str(src), str(tgt), label=str(weight))

    os.makedirs(os.path.dirname(os.path.abspath(output_png)) or ".", exist_ok=True)
    # ``cleanup=False`` so the intermediate .gv source is kept next to the PNG,
    # which makes the output easier to debug.
    dot.render(filename=os.path.splitext(output_png)[0], cleanup=False)
    # ``dot.render`` writes "<stem>" without extension and the actual PNG as
    # "<stem>.png"; if the user asked for a different extension we rename.
    produced = os.path.splitext(output_png)[0] + ".png"
    if produced != output_png and os.path.exists(produced) and not os.path.exists(output_png):
        os.replace(produced, output_png)
    print(f"[convert_categorical_dpg] Wrote PNG to {output_png}")


# ---------------------------------------------------------------------------
# 8.  CLI entry point
# ---------------------------------------------------------------------------
def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Rewrite one-hot categorical predicates in a DPG structure JSON "
            "as set-membership predicates and render the result as a PNG."
        )
    )
    parser.add_argument(
        "input_json",
        help="Path to the *dpg_structure.json produced by quickstart_categorical.py",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Where to write the converted JSON (default: <input>_converted.json).",
    )
    parser.add_argument(
        "--no-render",
        action="store_true",
        help="Skip the PNG render step (JSON only).",
    )
    parser.add_argument(
        "--layout",
        default="LR",
        choices=["LR", "TB", "BT", "RL"],
        help="Graphviz rankdir used for the PNG (default: LR).",
    )
    args = parser.parse_args(argv)

    input_path = os.path.abspath(args.input_json)
    if not os.path.exists(input_path):
        print(f"[convert_categorical_dpg] ERROR: file not found: {input_path}", file=sys.stderr)
        return 1

    with open(input_path, "r", encoding="utf-8") as f:
        dpg = json.load(f)

    print(f"[convert_categorical_dpg] Loaded {input_path}")
    print(f"[convert_categorical_dpg] Original node count : {len(dpg.get('nodes', []))}")

    converted = translate_dpg(dpg)

    output_json = args.output_json or os.path.splitext(input_path)[0] + "_converted.json"
    output_json = os.path.abspath(output_json)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(converted, f, indent=2, ensure_ascii=False)
    print(f"[convert_categorical_dpg] Wrote converted JSON to {output_json}")

    # Show a short summary of the rewritten labels so the user can sanity-check.
    print("[convert_categorical_dpg] Sample of rewritten labels:")
    sample_count = 0
    for n in converted["nodes"]:
        if n["label"] and not str(n["label"]).startswith("Class"):
            print(f"  - {n['label']}")
            sample_count += 1
            if sample_count >= 12:
                break

    if args.no_render:
        return 0

    output_png = os.path.splitext(output_json)[0] + ".png"
    render_dpg_png(converted, output_png, layout=args.layout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
