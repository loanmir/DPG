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

For each processed subdirectory the script writes **both** the
``..._DPG_grouped.png`` image and a ``..._DPG_grouped_structure.json``
payload (absorbed nodes dropped, edges reweighted) into the ``wip/``
subdir, so downstream consumers can load the merged graph directly.

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


def _apply_grouping(
    structure: dict,
    label_map: Dict[str, str],
) -> Tuple[Dict[str, str], Dict[Tuple[str, str], float], Set[str]]:
    """Apply the sequential grouping pass directly on the graph structure.

    A run of nodes ``n0 -> n1 -> ... -> nk`` collapses into ``n0`` when,
    for every consecutive pair ``(n_i, n_{i+1})``:

    * both nodes parse as an ``IN``/``NOT IN`` categorical predicate on the
      *same* base feature and the *same* operator, and
    * ``n_i`` has exactly one outgoing edge (to ``n_{i+1}``), and
    * ``n_{i+1}`` has exactly one incoming edge (from ``n_i``).

    These two degree checks are what make the collapse structurally safe:
    if ``n_i`` had another child, relabelling it to the merged predicate
    would silently change the meaning of that other branch too; if
    ``n_{i+1}`` had another parent, dropping it would disconnect that
    parent's branch. A node that fails either check simply ends the run
    instead of merging, so a run is always the longest *simple* chain
    available -- matching the "branching nodes break the chain" rule.

    The run's merged outgoing edge(s) carry every weight the run absorbs:
    the edges internal to the run *and* the edge(s) leaving the run's tail
    node to whatever comes next (a Class node or another predicate).

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
    graph_data = structure.get("graph", {})
    nx_graph = json_graph.node_link_graph(graph_data)

    succ: Dict[str, List[str]] = {
        str(n): [str(c) for c in nx_graph.successors(str(n))] for n in nx_graph.nodes()
    }
    pred: Dict[str, List[str]] = {
        str(n): [str(p) for p in nx_graph.predecessors(str(n))] for n in nx_graph.nodes()
    }

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

    parsed_map: Dict[str, Optional[Tuple[str, str, List[str]]]] = {
        nid: _parse_in_label(label) for nid, label in label_map.items()
    }

    def _mergeable(u: str, v: str) -> bool:
        pu, pv = parsed_map.get(u), parsed_map.get(v)
        if pu is None or pv is None:
            return False
        if len(succ.get(u, [])) != 1 or succ[u][0] != v:
            return False
        if len(pred.get(v, [])) != 1 or pred[v][0] != u:
            return False
        return pu[0] == pv[0] and pu[1] == pv[1]

    # A node is absorbed iff its single predecessor is mergeable with it.
    # Every node that never shows up as a value here is the head of its own
    # (possibly length-1) run.
    absorbed_by: Dict[str, str] = {}
    for u in succ:
        if len(succ.get(u, [])) == 1:
            v = succ[u][0]
            if _mergeable(u, v):
                absorbed_by[v] = u
    collapsed_ids: Set[str] = set(absorbed_by.keys())

    grouped_labels: Dict[str, str] = dict(label_map)
    grouped_weights: Dict[Tuple[str, str], float] = dict(weight_map)

    heads = [
        n for n in succ
        if n not in collapsed_ids and parsed_map.get(n) is not None
    ]
    for head in heads:
        run = [head]
        node = head
        while len(succ.get(node, [])) == 1 and _mergeable(node, succ[node][0]):
            node = succ[node][0]
            run.append(node)
        if len(run) <= 1:
            continue

        base, op, _ = parsed_map[head]
        merged_cats: List[str] = []
        for nid in run:
            merged_cats.extend(parsed_map[nid][2])
        grouped_labels[head] = _format_in_label(base, op, merged_cats)

        internal_sum = sum(
            weight_map.get((run[k], run[k + 1]), 0.0) for k in range(len(run) - 1)
        )
        for k in range(len(run) - 1):
            grouped_weights.pop((run[k], run[k + 1]), None)

        tail = run[-1]
        for nxt in succ.get(tail, []):
            key = (tail, nxt)
            original = grouped_weights.pop(key, weight_map.get(key, 0.0))
            new_key = (head, nxt)
            grouped_weights[new_key] = (
                grouped_weights.get(new_key, 0.0) + original + internal_sum
            )

    return grouped_labels, grouped_weights, collapsed_ids


# ---------------------------------------------------------------------------
# Grouped structure JSON payload
# ---------------------------------------------------------------------------

def _build_grouped_structure(
    structure: dict,
    grouped_labels: Dict[str, str],
    grouped_weights: Dict[Tuple[str, str], float],
    collapsed_ids: Set[str],
) -> dict:
    """Rebuild the ``structure`` dict after the IN/NOT-IN rewrite and the
    sequential grouping pass.

    Nodes whose ids appear in ``collapsed_ids`` are dropped (they have been
    absorbed by the leader of their merged run). Surviving nodes keep every
    original field but have their ``label`` replaced by the value in
    ``grouped_labels``. Edges are taken from ``grouped_weights`` (which
    already sums the absorbed weights onto the leader's outgoing edge);
    any edge whose endpoints were absorbed is filtered out. Non-endpoint
    metadata (e.g. link ``id``) is preserved by looking the link up in
    the original ``edges``/``links`` list.

    Returns a new dict that mirrors the top-level shape of ``structure``
    (``nodes`` + ``graph``) and is safe to ``json.dump`` straight to disk.
    """
    # --- Surviving nodes (label rewritten) ------------------------------
    new_nodes: List[dict] = []
    for node in structure.get("nodes", []):
        node_id = str(node.get("id", ""))
        if not node_id or "->" in node_id:
            continue  # edge pseudo-nodes
        if node_id in collapsed_ids:
            continue  # absorbed into leader
        new_node = dict(node)
        new_node["label"] = grouped_labels.get(
            node_id, str(node.get("label", ""))
        )
        new_nodes.append(new_node)

    # --- Build a (src, dst) -> original-link template so we keep the
    # --- edge metadata (id, etc.) of the first raw occurrence of each key.
    graph_data = structure.get("graph", {})
    raw_edges = graph_data.get("edges") or graph_data.get("links") or []
    template_by_key: Dict[Tuple[str, str], dict] = {}
    for link in raw_edges:
        key = (str(link.get("source", "")), str(link.get("target", "")))
        template_by_key.setdefault(key, dict(link))

    # --- Surviving edges with the grouped weights ----------------------
    new_edges: List[dict] = []
    for (src, dst), weight in grouped_weights.items():
        if src in collapsed_ids or dst in collapsed_ids:
            continue
        template = template_by_key.get((src, dst), {})
        new_link = dict(template)
        new_link["source"] = src
        new_link["target"] = dst
        new_link["weight"] = weight
        new_edges.append(new_link)

    # --- Stitch the new payload together, preserving every other
    # --- top-level key that the structure may carry -------------------
    new_structure: dict = dict(structure)
    new_structure["nodes"] = new_nodes
    if "graph" in new_structure:
        new_graph = dict(new_structure["graph"])
        new_graph["edges"] = new_edges
        new_graph["links"] = new_edges
        new_structure["graph"] = new_graph
    else:
        # Fall back to the flat layout (edges under the root) so the
        # output is still self-consistent.
        new_structure["edges"] = new_edges
        new_structure["links"] = new_edges

    return new_structure


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

    # Persist the grouped structure as JSON so downstream tooling can
    # consume the merged graph without re-running the rewrite + grouping
    # pipeline.
    grouped_structure = _build_grouped_structure(
        structure, grouped_labels, grouped_weights, collapsed_ids
    )
    grouped_json_path = os.path.join(wip_dir, f"{output_name}_structure.json")
    with open(grouped_json_path, "w", encoding="utf-8") as fh:
        json.dump(grouped_structure, fh, indent=2, ensure_ascii=False)
    print(f"  [ok]   {run_id} -> {grouped_json_path}")

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
