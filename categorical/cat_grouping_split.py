"""
cat_grouping_split.py
======================

A more aggressive variant of ``cat_grouping.py``'s sequential grouping pass.

``cat_grouping._apply_grouping`` only merges two consecutive same-base /
same-op categorical nodes when the chain between them is *strictly private*:
the parent has exactly one outgoing edge (to the child) and the child has
exactly one incoming edge (from the parent). That's the only shape where you
can delete the child, fold its weight into the parent, and be sure you
haven't corrupted some other branch that also relies on either node.

In a real DPG that shape is rare: DPG construction deliberately *unifies*
identical predicate nodes that recur across many different rule paths, so a
predicate like ``loan_intent NOT IN {VENTURE}`` is typically reached from
several different parents (in-degree > 1), and a predicate like
``loan_intent NOT IN {PERSONAL}`` typically leads to several different next
tests (out-degree > 1). Under the strict rule almost nothing merges.

This script relaxes the rule by *splitting* the node that's in the way
instead of refusing to merge:

* If the parent ``u`` has other children, ``u`` can't disappear -- but the
  child ``v`` can still be relabelled in place to show the union of
  categories (edge weight untouched), *provided* ``v`` belongs to ``u``
  alone. If ``v`` is also shared by other parents, a private copy of ``v``
  is cloned first (identical label + outgoing edges), so the original ``v``
  is left completely untouched for its other parents, and the clone is what
  gets relabelled.
* If the parent ``u`` has exactly one outgoing edge (to ``v``), the merge is
  identical to ``cat_grouping``'s strict rule: ``u`` absorbs ``v`` (cloning
  it first if ``v`` is shared), ``u``'s id survives with the combined label,
  and the weight folds forward onto whatever comes after ``v``.

The run is iterative and continues until no adjacent same-base/same-op pair
remains, so chains of length 3+ still collapse to a single node exactly as
in ``cat_grouping``.

Trade-off: this can grow the node/edge count of a heavily-shared DPG --
a shared predicate hub now shows up once per distinct merged chain it
belongs to, rather than as a single canonical node -- in exchange for
grouping every eligible chain instead of only the (rare) fully private ones.

For each processed subdirectory the script writes both the
``..._DPG_split_grouped.png`` image and a
``..._DPG_split_grouped_structure.json`` payload into the
``wip/grouping_split/`` subdir, so it never overwrites whatever
``cat_grouping.py`` (or anything else) already produced directly in
``wip/``.

Usage
-----
    python examples/cat_grouping_split.py
    python examples/cat_grouping_split.py --amount 10
    python examples/cat_grouping_split.py --root examples/results_cat --amount 5
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Optional, Set, Tuple

# --- Make the project root + this script's own directory importable --------
SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, SCRIPT_DIR)

import graphviz
import pandas as pd
from networkx.readwrite import json_graph

from dpg.visualizer import plot_dpg

# Reuse the (already-correct) one-hot rewrite + label parsing/formatting from
# cat_grouping.py instead of re-implementing them here.
from .cat_grouping import (
    _build_label_map,
    _format_in_label,
    _iter_subdirs,
    _load_structure_json,
    _load_visualization_config,
    _parse_in_label,
)


# ---------------------------------------------------------------------------
# Split-then-merge grouping
# ---------------------------------------------------------------------------

def _split_then_merge(
    structure: dict,
    label_map: Dict[str, str],
) -> Tuple[Dict[str, str], Dict[Tuple[str, str], float]]:
    """Group consecutive same-base/same-op categorical nodes as aggressively
    as possible, cloning shared nodes so a merge never corrupts a branch it
    doesn't belong to.

    Returns ``(labels, edges)`` describing the *entire* rebuilt graph (not a
    diff against ``structure``): every surviving node id maps to its final
    label, and every surviving edge to its final weight. New node ids
    introduced by cloning look like ``"<original_id>#<n>"``.
    """
    graph_data = structure.get("graph", {})
    nx_graph = json_graph.node_link_graph(graph_data)

    # ``label_map`` mirrors ``structure["nodes"]``, which also carries a few
    # edge-label pseudo-entries whose id contains "->" and that never appear
    # in ``structure["graph"]`` (the actual node-link graph). Drop them here
    # so they don't show up as disconnected, empty-label boxes downstream.
    labels: Dict[str, str] = {
        nid: label for nid, label in label_map.items() if "->" not in nid
    }
    for n in nx_graph.nodes():
        labels.setdefault(str(n), label_map.get(str(n), ""))

    edges: Dict[Tuple[str, str], float] = {}
    for link in graph_data.get("edges", graph_data.get("links", [])):
        src, dst = str(link["source"]), str(link["target"])
        try:
            w = float(link.get("weight", 0.0))
        except (TypeError, ValueError):
            w = 0.0
        edges[(src, dst)] = edges.get((src, dst), 0.0) + w

    settled: Set[Tuple[str, str]] = set()
    clone_counter = 0

    def _degrees() -> Tuple[Dict[str, int], Dict[str, int]]:
        out_count: Dict[str, int] = {}
        in_count: Dict[str, int] = {}
        for (s, d) in edges:
            out_count[s] = out_count.get(s, 0) + 1
            in_count[d] = in_count.get(d, 0) + 1
        return out_count, in_count

    def _clone_node(v: str) -> str:
        """Private copy of ``v``: same label, same *current* outgoing
        edges. Nothing points to it yet -- the caller wires up its single
        incoming edge."""
        nonlocal clone_counter
        clone_counter += 1
        v_copy = f"{v}#{clone_counter}"
        labels[v_copy] = labels[v]
        for (s, d), w in list(edges.items()):
            if s == v:
                edges[(v_copy, d)] = edges.get((v_copy, d), 0.0) + w
        return v_copy

    def _fold_forward(
        u: str, target: str, w_uv: float, base: str, op: str,
        cats_u: List[str], cats_v: List[str],
    ) -> None:
        """``u`` absorbs ``target`` (``v`` or a private clone of it): the
        label combines, ``target``'s outgoing edges move onto ``u`` with
        ``w_uv`` folded into each, and ``target`` disappears. Requires
        ``edges[(u, target)] == w_uv`` to already be the case."""
        labels[u] = _format_in_label(base, op, cats_u + cats_v)
        del edges[(u, target)]
        for (s, d), w in list(edges.items()):
            if s == target:
                del edges[(s, d)]
                key = (u, d)
                edges[key] = edges.get(key, 0.0) + w + w_uv
        labels.pop(target, None)

    max_iterations = 10 * (len(edges) + len(labels) + 10)
    for _ in range(max_iterations):
        out_count, in_count = _degrees()

        candidate = None
        for (u, v), w_uv in edges.items():
            if (u, v) in settled:
                continue
            pu, pv = _parse_in_label(labels[u]), _parse_in_label(labels[v])
            if pu is None or pv is None or pu[0] != pv[0] or pu[1] != pv[1]:
                settled.add((u, v))
                continue
            candidate = (u, v, w_uv, pu, pv)
            break
        if candidate is None:
            break

        u, v, w_uv, pu, pv = candidate
        base, op = pu[0], pu[1]

        if out_count.get(u, 0) == 1:
            # u has nowhere else to go: it can safely disappear.
            if in_count.get(v, 0) == 1:
                target = v
            else:
                target = _clone_node(v)
                del edges[(u, v)]
                edges[(u, target)] = w_uv
            _fold_forward(u, target, w_uv, base, op, pu[2], pv[2])
        elif in_count.get(v, 0) == 1:
            # v belongs to u alone: merge in place, weight untouched.
            labels[v] = _format_in_label(base, op, pu[2] + pv[2])
            settled.add((u, v))
        else:
            # Both shared: clone v privately for u's branch and merge into
            # the clone, leaving the original v untouched for its other
            # parents.
            v_copy = _clone_node(v)
            labels[v_copy] = _format_in_label(base, op, pu[2] + pv[2])
            del edges[(u, v)]
            edges[(u, v_copy)] = w_uv
            settled.add((u, v_copy))
    else:
        print(
            "  [warn] split-then-merge hit its iteration cap "
            f"({max_iterations}); the input graph may contain a cycle."
        )

    return labels, edges


# ---------------------------------------------------------------------------
# Rendering / export from a plain (labels, edges) graph
# ---------------------------------------------------------------------------

def _build_dot_from_graph(
    labels: Dict[str, str],
    edges: Dict[Tuple[str, str], float],
    visualization_config: dict,
) -> graphviz.Digraph:
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
        "dpg_cat_split_grouped",
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

    for node_id, label in labels.items():
        fillcolor = class_fillcolor if label.startswith("Class") else default_fillcolor
        dot.node(
            node_id,
            label=_escape(label),
            style="filled",
            fontsize="20",
            fillcolor=fillcolor,
        )

    for (src, dst), weight in sorted(edges.items(), key=lambda kv: kv[1], reverse=True):
        dot.edge(src, dst, label=str(weight), penwidth="1", fontsize="18")

    return dot


def _build_split_structure(
    structure: dict,
    labels: Dict[str, str],
    edges: Dict[Tuple[str, str], float],
) -> dict:
    """Rebuild a ``structure``-shaped payload from the final (labels, edges)
    graph. Cloned node ids (``"<original_id>#<n>"``) have no counterpart in
    the original structure, so surviving nodes only carry ``id``/``label``
    rather than every original field.
    """
    new_nodes = [{"id": nid, "label": lbl} for nid, lbl in labels.items()]
    new_edges = [
        {"source": src, "target": dst, "weight": weight}
        for (src, dst), weight in edges.items()
    ]

    new_structure: dict = dict(structure)
    new_structure["nodes"] = new_nodes
    if "graph" in new_structure:
        new_graph = dict(new_structure["graph"])
        new_graph["nodes"] = [{"id": nid} for nid in labels]
        new_graph["edges"] = new_edges
        new_graph["links"] = new_edges
        new_structure["graph"] = new_graph
    else:
        new_structure["edges"] = new_edges
        new_structure["links"] = new_edges

    return new_structure


def _build_synthetic_metrics_from_graph(
    labels: Dict[str, str],
    edges: Dict[Tuple[str, str], float],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Minimal node/edge metric DataFrames for ``plot_dpg``. Real per-node
    CSV metrics (impurity, sample counts, ...) don't have a clean mapping
    onto cloned ids, so this script always uses the synthetic fallback
    rather than trying to reconcile the two.
    """
    nodes = [{"Node": nid, "Label": lbl} for nid, lbl in labels.items()]
    edge_rows = [
        {"Source_id": src, "Target_id": dst, "Weight": w}
        for (src, dst), w in edges.items()
    ]
    return pd.DataFrame(nodes), pd.DataFrame(edge_rows)


# ---------------------------------------------------------------------------
# Per-run processing
# ---------------------------------------------------------------------------

def _process_subdir(
    subdir: str,
    wip_dir: str,
    visualization_config: dict,
) -> Optional[str]:
    """Rebuild a categorically-rewritten + split-then-merge-grouped DPG PNG
    inside ``wip_dir``. Returns the path of the produced PNG, or ``None`` on
    failure.
    """
    run_id = os.path.basename(subdir.rstrip(os.sep))
    structure_path = os.path.join(subdir, f"{run_id}_dpg_structure.json")

    if not os.path.isfile(structure_path):
        print(f"  [skip] {run_id}: missing {os.path.basename(structure_path)}")
        return None

    structure = _load_structure_json(structure_path)
    label_map, _ = _build_label_map(structure)
    labels, edges = _split_then_merge(structure, label_map)

    dot = _build_dot_from_graph(labels, edges, visualization_config)
    df_nodes, df_edges = _build_synthetic_metrics_from_graph(labels, edges)

    os.makedirs(wip_dir, exist_ok=True)
    output_name = f"{run_id}_DPG_split_grouped"
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

    split_structure = _build_split_structure(structure, labels, edges)
    split_json_path = os.path.join(wip_dir, f"{output_name}_structure.json")
    with open(split_json_path, "w", encoding="utf-8") as fh:
        json.dump(split_structure, fh, indent=2, ensure_ascii=False)
    print(f"  [ok]   {run_id} -> {split_json_path}")

    return out_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

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
        wip_dir = os.path.join(subdir, "wip", "grouping_split")
        try:
            _process_subdir(subdir, wip_dir, visualization_config)
        except Exception as exc:  # pragma: no cover - best-effort reporting
            print(f"  [err]  {os.path.basename(subdir)}: {exc}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
