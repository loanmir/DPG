"""
cat_grouping.py
================

For each subdirectory of ``examples/results_cat/`` (default; configurable via
``--root``) produce a re-labelled, *grouped* DPG PNG inside a ``wip/`` subdir.

Compared to ``categorical_view_dpg.py`` this script does **two** things on top
of the one-hot-to-IN/NOT-IN rewrite:

1. **Categorical rewrite.** Every node whose label is a one-hot encoded
   categorical predicate of the form ``<base>_<CAT> <op> 0.5`` is rewritten
   to ``<base> IN {<CAT>}`` (op ``>``/``>=``) or
   ``<base> NOT IN {<CAT>}`` (op ``<=``/``<``), matching
   ``categorical_view_dpg.py``.
2. **Sequential grouping.** When two consecutive nodes on a *single*
   root-to-leaf path share both the same base feature and the same operator
   (``IN`` or ``NOT IN``), they are collapsed into a single node with the
   categories merged.

   Example (from ``toy_chain_intent_with_age``)::

       loan_intent NOT IN {PERSONAL}  ->  loan_intent NOT IN {VENTURE}  ->  Class 0
                                          (edge weight 56)              (weight 212)

   becomes::

       loan_intent NOT IN {PERSONAL, VENTURE}  ->  Class 0      (weight 56 + 212 = 268)

   The grouping walk is iterative: any run of length > 2 collapses in one
   pass, and the resulting merged node can still be merged with the next
   neighbour on the path if the operator matches.

   Branching nodes (out-degree > 1) and non-categorical nodes
   (``Class 0``/``Class 1``, ``person_age <= 25``, ...) **break** the chain
   and prevent grouping across themselves.

Usage
-----
    python examples/cat_grouping.py
    python examples/cat_grouping.py --amount 10
    python examples/cat_grouping.py --root examples/results_cat --amount 5
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import Dict, List, Optional, Set, Tuple

# --- Make the project root importable (so `dpg` and `metrics` resolve) -------
SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
sys.path.insert(0, PROJECT_ROOT)

import graphviz
import pandas as pd
from networkx.readwrite import json_graph

from dpg.visualizer import plot_dpg


# ---------------------------------------------------------------------------
# Heuristics for detecting one-hot encoded categorical predicates
# ---------------------------------------------------------------------------

# Standard predicate: "<feature> <op> <number>"  e.g. "petal_length <= 1.23"
_PREDICATE_PATTERN = re.compile(
    r"^\s*(?P<feature>.+?)\s*(?P<op><=|>|<|>=|==|!=)\s*(?P<value>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*$"
)

# IN / NOT IN labels produced by the rewrite (and to be re-parsed for grouping)
_IN_PATTERN = re.compile(
    r"^\s*(?P<base>.+?)\s+(?P<op>IN|NOT\s+IN)\s+\{(?P<cats>[^}]+)\}\s*$"
)


def _split_one_hot_column(col: str) -> Optional[Tuple[str, str]]:
    """Split ``person_home_ownership_OWN`` into ``('person_home_ownership',
    'OWN')``. Returns ``None`` for non-OHE columns (no underscore, or trailing
    token is numeric, e.g. ``x_1``).
    """
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


def _is_categorical_predicate(label: str) -> bool:
    """``True`` when the label looks like a one-hot OHE predicate split on
    the 0.5 threshold."""
    parsed = _PREDICATE_PATTERN.match(label)
    if parsed is None:
        return False
    feature = parsed.group("feature")
    value = float(parsed.group("value"))
    if not _split_one_hot_column(feature):
        return False
    if abs(value - 0.5) > 1e-9:
        return False
    return True


def _to_categorical_label(label: str) -> str:
    """Rewrite a one-hot OHE predicate to ``base IN {CAT}`` /
    ``base NOT IN {CAT}``. Non-OHE labels are returned unchanged.
    """
    parsed = _PREDICATE_PATTERN.match(label)
    if parsed is None:
        return label
    feature, op, value = (
        parsed.group("feature"),
        parsed.group("op"),
        parsed.group("value"),
    )
    split = _split_one_hot_column(feature)
    if split is None or abs(float(value) - 0.5) > 1e-9:
        return label
    base, category = split
    if op in (">", ">="):
        return f"{base} IN {{{category}}}"
    if op in ("<=", "<"):
        return f"{base} NOT IN {{{category}}}"
    return label


def _parse_in_label(label: str) -> Optional[Tuple[str, str, List[str]]]:
    """Inverse of the rewrite: parse ``base IN {A, B}`` /
    ``base NOT IN {A, B}`` into ``(base, op_normalised, [cats])``.

    ``op_normalised`` is ``"IN"`` or ``"NOT IN"`` (single canonical form).
    Returns ``None`` for non-IN labels.
    """
    m = _IN_PATTERN.match(label)
    if m is None:
        return None
    base = m.group("base").strip()
    op = m.group("op").upper().replace("  ", " ")  # canonicalise spacing
    op = "NOT IN" if op == "NOT IN" else "IN"
    cats = [c.strip() for c in m.group("cats").split(",") if c.strip()]
    return base, op, cats


# ---------------------------------------------------------------------------
# Graph rebuilding helpers
# ---------------------------------------------------------------------------

def _load_structure_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _build_label_map(structure: dict) -> Tuple[Dict[str, str], Dict[str, str]]:
    """Return ``(id -> new_label, id -> old_label)`` for every node in the
    structure payload (the IN/NOT-IN rewrite only)."""
    new_labels: Dict[str, str] = {}
    old_labels: Dict[str, str] = {}
    for node in structure.get("nodes", []):
        node_id = str(node["id"])
        old_label = str(node.get("label", ""))
        new_labels[node_id] = _to_categorical_label(old_label)
        old_labels[node_id] = old_label
    return new_labels, old_labels


# ---------------------------------------------------------------------------
# Sequential grouping of IN / NOT IN nodes
# ---------------------------------------------------------------------------

def _root_to_leaf_paths(
    sources: List[str], succ: Dict[str, List[str]]
) -> List[List[str]]:
    """Enumerate every directed root-to-leaf path in the DAG.

    A path ends when the current node has no outgoing successors. Cycles are
    detected and broken to keep the algorithm total: a node already on the
    current path is treated as a leaf.

    Returns the list of paths, each path being a list of node ids from root
    to leaf (inclusive). Leaves are repeated across paths that branch.
    """
    paths: List[List[str]] = []

    def walk(node: str, acc: List[str], on_path: Set[str]) -> None:
        if node in on_path:
            # cycle: emit current prefix as a closed path
            paths.append(acc[:])
            return
        new_acc = acc + [node]
        children = succ.get(node, [])
        if not children:
            paths.append(new_acc)
            return
        new_on_path = on_path | {node}
        for c in children:
            walk(c, new_acc, new_on_path)

    for s in sources:
        walk(s, [], set())
    return paths


def _format_in_label(base: str, op: str, cats: List[str]) -> str:
    """Canonical ``base IN {A, B}`` / ``base NOT IN {A, B}`` rendering with
    stable category order (preserves first-seen, deduped)."""
    seen: Set[str] = set()
    deduped: List[str] = []
    for c in cats:
        if c not in seen:
            seen.add(c)
            deduped.append(c)
    return f"{base} {op} {{{', '.join(deduped)}}}"


def _collapse_path(
    path: List[str],
    label_map: Dict[str, str],
    weight_map: Dict[Tuple[str, str], float],
) -> List[Tuple[str, str]]:
    """Return the list of ``(node_id, label)`` pairs that should appear in
    the collapsed graph, in path order, after merging consecutive
    same-base+same-op categorical nodes.

    Edges along a collapsed segment have their weights summed and stored
    in ``weight_map`` keyed by ``(incoming_node_id, outgoing_node_id)``.

    The first/last nodes of the path are always preserved. A path of length
    1 (a leaf) yields the single node unchanged.
    """
    if len(path) <= 1:
        return [(path[0], label_map[path[0]])]

    out: List[Tuple[str, str]] = []
    i = 0
    n = len(path)
    while i < n:
        nid = path[i]
        cur_label = label_map[nid]
        cur_parsed = _parse_in_label(cur_label)
        if cur_parsed is None:
            # Non-categorical node: emit as-is and move on.
            out.append((nid, cur_label))
            i += 1
            continue
        base, op, cats = cur_parsed
        # Greedily extend the run while the next node is IN/NOT IN on the
        # same base with the same operator.
        j = i + 1
        merged_cats: List[str] = list(cats)
        # Track the sum of edge weights that the merged segment absorbs:
        # the weight of edge (path[k-1] -> path[k]) for k in (i+1 .. j).
        absorbed_weight = 0.0
        while j < n:
            next_label = label_map[path[j]]
            next_parsed = _parse_in_label(next_label)
            if next_parsed is None:
                break
            nbase, nop, ncats = next_parsed
            if nbase != base or nop != op:
                break
            merged_cats.extend(ncats)
            # Absorb the weight of the edge (path[j-1] -> path[j]).
            absorbed_weight += weight_map.get((path[j - 1], path[j]), 0.0)
            j += 1
        # The merged segment is path[i .. j-1]. We collapse it into a
        # single canonical node keyed by path[i] (so the id is stable and
        # we don't need to invent new node ids).
        if j - i == 1:
            # No neighbours to merge with; emit original.
            out.append((nid, cur_label))
        else:
            out.append((nid, _format_in_label(base, op, merged_cats)))
            # The outgoing edge weight from the merged node is the sum of
            # the original consecutive edges. We don't know the final
            # destination here (it's the next node *after* the segment),
            # so the caller computes that.
            # We store a per-segment flag instead.
            weight_map[("__merge_in__", path[j - 1])] = absorbed_weight
        i = j
    return out


def _apply_grouping(
    structure: dict,
    label_map: Dict[str, str],
) -> Tuple[Dict[str, str], Dict[Tuple[str, str], float], Set[str]]:
    """Apply the sequential grouping pass.

    Returns
    -------
    grouped_labels : dict
        ``node_id -> final_label`` (the IN/NOT-IN rewrite + merged run).
    grouped_weights : dict
        ``(src_id, dst_id) -> weight`` for the collapsed graph.
    collapsed_ids : set
        Set of node ids that were *absorbed* into a neighbour (i.e. the
        non-leading nodes of every merged run). The caller should drop
        these from the rendered graph.
    """
    # --- Build the directed graph (NetworkX may or may not be available
    # --- as a hard dep, but the script's project root makes it available).
    graph_data = structure.get("graph", {})
    nx_graph = json_graph.node_link_graph(graph_data)

    succ: Dict[str, List[str]] = {str(n): [str(c) for c in nx_graph.successors(str(n))]
                                  for n in nx_graph.nodes()}
    # Root nodes: in-degree 0.
    indeg: Dict[str, int] = {str(n): nx_graph.in_degree(str(n)) for n in nx_graph.nodes()}
    sources: List[str] = [n for n, d in indeg.items() if d == 0]

    # --- Original edge weights, indexed by (src, dst) -------------------
    raw_edges = graph_data.get("edges", graph_data.get("links", []))
    weight_map: Dict[Tuple[str, str], float] = {}
    for link in raw_edges:
        src = str(link["source"])
        dst = str(link["target"])
        try:
            w = float(link.get("weight", 0.0))
        except (TypeError, ValueError):
            w = 0.0
        weight_map[(src, dst)] = w

    paths = _root_to_leaf_paths(sources, succ)

    # --- Walk each path and collapse runs -------------------------------
    # Use a *fresh* per-call absorbed-weight map. The ``_collapse_path``
    # helper stashes intermediate values under a sentinel key
    # ``("__merge_in__", last_id_of_segment)`` -- we resolve them here by
    # re-walking the collapsed path.
    collapsed_ids: Set[str] = set()
    grouped_labels: Dict[str, str] = dict(label_map)  # shallow copy
    grouped_weights: Dict[Tuple[str, str], float] = dict(weight_map)

    for path in paths:
        # Build a clean per-path collapse to know which nodes are merged
        # into their leader. A node at index k (k > 0) of the merged
        # run is "absorbed" if its label was rewritten in this pass.
        i = 0
        n = len(path)
        while i < n:
            nid = path[i]
            cur_label = label_map[nid]
            cur_parsed = _parse_in_label(cur_label)
            if cur_parsed is None:
                i += 1
                continue
            base, op, cats = cur_parsed
            j = i + 1
            merged_cats: List[str] = list(cats)
            while j < n:
                nxt_label = label_map[path[j]]
                nxt_parsed = _parse_in_label(nxt_label)
                if nxt_parsed is None:
                    break
                nbase, nop, ncats = nxt_parsed
                if nbase != base or nop != op:
                    break
                merged_cats.extend(ncats)
                j += 1
            if j - i > 1:
                # Collapse path[i .. j-1] into path[i].
                grouped_labels[nid] = _format_in_label(base, op, merged_cats)
                # Drop the edges INSIDE the segment AND the outgoing edge
                # of the last absorbed node (it gets replaced by the
                # leader's outgoing edge with the summed weight).
                for k in range(i, j - 1):
                    grouped_weights.pop((path[k], path[k + 1]), None)
                grouped_weights.pop((path[j - 1], path[j]), None)
                # Sum the absorbed edge weights onto the OUTGOING edge
                # of the merged node (i.e. (path[i], path[j])). This
                # holds whether path[j] is an internal node or a leaf.
                absorbed = sum(
                    weight_map.get((path[k], path[k + 1]), 0.0)
                    for k in range(i, j - 1)
                )
                key = (path[i], path[j])
                grouped_weights[key] = grouped_weights.get(key, 0.0) + absorbed
                # Mark the absorbed nodes.
                for k in range(i + 1, j):
                    collapsed_ids.add(path[k])
            i = j

    return grouped_labels, grouped_weights, collapsed_ids


# ---------------------------------------------------------------------------
# DOT rendering of the (rewritten + grouped) graph
# ---------------------------------------------------------------------------

def _build_dot(
    structure: dict,
    grouped_labels: Dict[str, str],
    grouped_weights: Dict[Tuple[str, str], float],
    collapsed_ids: Set[str],
    visualization_config: dict,
) -> graphviz.Digraph:
    """Rebuild a Graphviz Digraph using the rewritten + grouped labels and
    edge weights, dropping nodes that have been collapsed into their leader.
    """
    viz = visualization_config.get("dpg", {}).get("visualization", {})
    graph_attrs = viz.get("graph_attrs", {})
    node_attrs = viz.get("node_attrs", {})
    class_node_attrs = viz.get("class_node", {})

    final_graph_attr = {
        "bgcolor": graph_attrs.get("bgcolor"),
        "rankdir": graph_attrs.get("rankdir"),
        "overlap": "false",
        "fontsize": "20",
    }
    final_graph_attr = {k: v for k, v in final_graph_attr.items() if v is not None}

    final_node_attr = {"shape": node_attrs.get("shape")}
    final_node_attr = {k: v for k, v in final_node_attr.items() if v is not None}

    default_fillcolor = node_attrs.get("fillcolor")
    class_fillcolor = class_node_attrs.get("fillcolor") or default_fillcolor

    dot = graphviz.Digraph(
        "dpg_cat_grouped",
        engine="dot",
        graph_attr=final_graph_attr,
        node_attr=final_node_attr if final_node_attr else None,
    )

    def _escape(label: str) -> str:
        return (
            label.replace("\\", "\\\\")
            .replace('"', '\\"')
            .replace("[", "\\[")
            .replace("]", "\\]")
        )

    # First pass: declare every surviving node with its grouped label.
    for node in structure.get("nodes", []):
        node_id = str(node["id"])
        if "->" in node_id:
            continue  # edge pseudo-nodes
        if node_id in collapsed_ids:
            continue  # absorbed by leader
        label = grouped_labels.get(node_id, str(node.get("label", "")))
        if label.startswith("Class"):
            fillcolor = class_fillcolor or default_fillcolor
        else:
            fillcolor = default_fillcolor
        dot.node(
            node_id,
            label=_escape(label),
            style="filled",
            fontsize="20",
            fillcolor=fillcolor,
        )

    # Second pass: emit edges in weight-descending order using the
    # *grouped* weight map (collapsed-segment weights have been summed).
    sorted_edges = sorted(
        grouped_weights.items(),
        key=lambda kv: kv[1],
        reverse=True,
    )
    for (src, dst), weight in sorted_edges:
        if src in collapsed_ids or dst in collapsed_ids:
            continue
        dot.edge(
            src,
            dst,
            label=str(weight),
            penwidth="1",
            fontsize="18",
        )

    return dot


# ---------------------------------------------------------------------------
# Per-run processing
# ---------------------------------------------------------------------------

def _load_node_metrics(path: str) -> pd.DataFrame:
    return pd.read_csv(path, index_col=0)


def _load_edge_metrics(path: str) -> pd.DataFrame:
    return pd.read_csv(path, index_col=0)


def _build_synthetic_metrics(
    structure: dict,
    grouped_labels: Dict[str, str],
    grouped_weights: Dict[Tuple[str, str], float],
    collapsed_ids: Set[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Build minimal node/edge metric DataFrames from the structure + grouped
    label/weight maps. Used as a fallback when the on-disk CSVs are missing.
    """
    nodes = []
    for node in structure.get("nodes", []):
        nid = str(node["id"])
        if "->" in nid or nid in collapsed_ids:
            continue
        nodes.append(
            {
                "Node": nid,
                "Label": grouped_labels.get(nid, str(node.get("label", ""))),
            }
        )
    edges = []
    for (src, dst), w in grouped_weights.items():
        if src in collapsed_ids or dst in collapsed_ids:
            continue
        edges.append({"Source_id": src, "Target_id": dst, "Weight": w})
    return pd.DataFrame(nodes), pd.DataFrame(edges)


def _process_subdir(
    subdir: str,
    wip_dir: str,
    visualization_config: dict,
) -> Optional[str]:
    """Rebuild a categorically-rewritten + grouped DPG PNG inside
    ``wip_dir``. Returns the path of the produced PNG, or ``None`` on
    failure.
    """
    run_id = os.path.basename(subdir.rstrip(os.sep))
    structure_path = os.path.join(subdir, f"{run_id}_dpg_structure.json")
    node_metrics_path = os.path.join(subdir, f"{run_id}_node_metrics.csv")
    edge_metrics_path = os.path.join(subdir, f"{run_id}_edge_metrics.csv")

    if not os.path.isfile(structure_path):
        print(f"  [skip] {run_id}: missing {os.path.basename(structure_path)}")
        return None

    structure = _load_structure_json(structure_path)
    label_map, _ = _build_label_map(structure)
    grouped_labels, grouped_weights, collapsed_ids = _apply_grouping(
        structure, label_map
    )
    dot = _build_dot(
        structure, grouped_labels, grouped_weights, collapsed_ids,
        visualization_config,
    )

    # Build or load node/edge metric DataFrames.
    if os.path.isfile(node_metrics_path):
        df_nodes = _load_node_metrics(node_metrics_path).copy()
        df_nodes["Node"] = df_nodes["Node"].astype(str)
        # Drop rows for nodes that were absorbed.
        df_nodes = df_nodes[~df_nodes["Node"].isin(collapsed_ids)].copy()
        # Re-label the surviving nodes with the grouped labels.
        original_labels = dict(zip(df_nodes["Node"], df_nodes["Label"]))
        df_nodes["Label"] = df_nodes["Node"].map(
            lambda nid: grouped_labels.get(nid, original_labels.get(nid, ""))
        )
    else:
        df_nodes, _ = _build_synthetic_metrics(
            structure, grouped_labels, grouped_weights, collapsed_ids
        )

    if os.path.isfile(edge_metrics_path):
        df_edges = _load_edge_metrics(edge_metrics_path).copy()
        if "Source_id" in df_edges.columns:
            df_edges["Source_id"] = df_edges["Source_id"].astype(str)
        if "Target_id" in df_edges.columns:
            df_edges["Target_id"] = df_edges["Target_id"].astype(str)
        # Drop rows whose endpoints were absorbed.
        df_edges = df_edges[
            ~df_edges["Source_id"].isin(collapsed_ids)
            & ~df_edges["Target_id"].isin(collapsed_ids)
        ].copy()
    else:
        _, df_edges = _build_synthetic_metrics(
            structure, grouped_labels, grouped_weights, collapsed_ids
        )

    os.makedirs(wip_dir, exist_ok=True)
    output_name = f"{run_id}_DPG_grouped"
    plot_dpg(
        output_name,
        dot,
        df_nodes,
        df_edges,
        save_dir=wip_dir,
        attribute=None,
        class_flag=False,
        layout_template="default",
        show=False,
        export_pdf=False,
    )

    out_path = os.path.join(wip_dir, f"{output_name}.png")
    print(f"  [ok]   {run_id} -> {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Subdir iteration + main
# ---------------------------------------------------------------------------

def _iter_subdirs(root: str) -> List[str]:
    """Return subdirs sorted by ``(dataset asc, perc_var desc, dt asc, ct asc)``.

    Mirrors the sort order used by ``categorical_view_dpg.py`` so the
    ``wip/`` outputs appear in a predictable order.
    """
    def _sort_key(entry: str):
        name = os.path.basename(entry.rstrip(os.sep))
        ds = name
        ct_s = dt_s = pv_s = None
        if "_ct=" in ds:
            ds, ct_s = ds.rsplit("_ct=", 1)
        if "_dt=" in ds:
            ds, dt_s = ds.rsplit("_dt=", 1)
        if "_pv=" in ds:
            ds, pv_s = ds.rsplit("_pv=", 1)
        if ds.startswith("ds="):
            ds = ds[len("ds="):]
        try:
            pv = float(pv_s) if pv_s is not None else 0.0
        except ValueError:
            pv = 0.0
        try:
            dt = int(dt_s) if dt_s is not None else 0
        except ValueError:
            dt = 0
        try:
            ct = float(ct_s) if ct_s is not None else 0.0
        except ValueError:
            ct = 0.0
        return (0, ds, -pv, dt, ct, name)

    subdirs = [
        entry
        for entry in (os.path.join(root, name) for name in os.listdir(root))
        if os.path.isdir(entry)
    ]
    subdirs.sort(key=_sort_key)
    return subdirs


def _load_visualization_config(config_path: str) -> dict:
    import yaml
    with open(config_path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        default=os.path.join(SCRIPT_DIR, "results_cat"),
        help="Root directory containing one subdirectory per gridsearch run.",
    )
    parser.add_argument(
        "--amount",
        type=int,
        default=None,
        help="If set, only process the first N subdirectories (sorted order).",
    )
    parser.add_argument(
        "--config",
        default=os.path.join(PROJECT_ROOT, "config.yaml"),
        help="Path to the YAML config used to style the rebuilt DPG.",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.root):
        print(f"ERROR: root directory not found: {args.root}")
        return 1

    visualization_config = _load_visualization_config(args.config)
    subdirs = _iter_subdirs(args.root)
    if args.amount is not None:
        subdirs = subdirs[: max(0, args.amount)]

    print(
        f"Processing {len(subdirs)} subdir(s) of {args.root} "
        f"(amount={args.amount if args.amount is not None else 'all'})."
    )
    for subdir in subdirs:
        wip_dir = os.path.join(subdir, "wip")
        try:
            _process_subdir(subdir, wip_dir, visualization_config)
        except Exception as exc:  # pragma: no cover - best-effort reporting
            print(f"  [err]  {os.path.basename(subdir)}: {exc}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
