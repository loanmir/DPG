"""
cat_grouping_split_conjunction.py
===================================

A cross-feature variant of ``cat_grouping_split.py``.

``cat_grouping_split._split_then_merge`` only merges two categorical nodes
when they test the *same* base feature with the *same* operator (``IN`` /
``NOT IN``), because that's the only case where the combined predicate can
be written as a single compact set (``base IN {A, B}``). Two categorical
nodes on *different* features (or the same feature with a different
operator) never merge, even when they sit on a strictly private chain.

That restriction was about label *formatting*, not structural safety: the
in/out-degree privacy check that makes a merge safe to perform (see
``cat_grouping_split``'s module docstring) never depended on the two nodes
sharing a feature. Walking node 1 then node 2 on a single path *is* the
logical conjunction of both conditions holding, regardless of what each one
tests. So this script relaxes the merge criterion to "both nodes are
categorical" (rewritten to ``IN``/``NOT IN`` form), and renders the combined
predicate as a chain of clauses joined by ``AND``, compacting any
consecutive clauses that *do* share a base feature + operator into one set
just like before, e.g.::

    person_education NOT IN {Bachelor} AND loan_intent IN {EDUCATION}
    person_education NOT IN {Bachelor, Master} AND loan_intent IN {EDUCATION, VENTURE}

Numeric (non-categorical) nodes are deliberately excluded from every merge:
a node only ever has a "clause" if its label parses as ``IN``/``NOT IN``
(``cat_grouping._parse_in_label`` returns ``None`` for a plain numeric
threshold like ``loan_amnt <= 16500.0``), so a numeric node always breaks
the chain on both sides -- it can neither absorb a categorical neighbour nor
be absorbed by one. Categorical-with-categorical is the only thing that
ever merges here.

Dropping the same-feature restriction has one sharp edge: some of these
DPGs are, once you stop caring which feature each edge tests, not actually
acyclic. They're built by unifying identical predicate nodes across many
different rules/trees, and different rules can test the same features in a
different order, so a loop like ``loan_intent NOT IN {PERSONAL} ->
person_gender IN {male} -> loan_intent NOT IN {VENTURE} -> loan_intent NOT
IN {PERSONAL}`` genuinely exists in about a third of the bundled examples.
``cat_grouping_split.py`` never noticed because its same-base/same-op
restriction meant such a cross-feature loop could never be walked in the
first place. Here it can, and folding/cloning around a real cycle has no
fixed point -- it just keeps growing. So before merging anything, this
script computes strongly connected components of the raw graph and
permanently excludes any edge whose endpoints are in the same non-trivial
component: those nodes are left exactly as the one-hot rewrite produced
them, un-grouped, while every acyclic part of the graph still gets grouped
as aggressively as described above.

Everything else -- the split-then-merge mechanics (cloning a shared node
rather than refusing to merge), the three merge cases (parent absorbs
child, child relabelled in place, both cloned), and the weight-folding
rules -- is identical to ``cat_grouping_split.py``; only the merge
*criterion* and the label *formatting* change.

For each processed subdirectory the script writes both the
``..._DPG_split_grouped_conjunction.png`` image and a
``..._DPG_split_grouped_conjunction_structure.json`` payload into the
``wip/grouping_split_conjunction/`` subdir, alongside (not overwriting)
whatever ``cat_grouping.py`` / ``cat_grouping_split.py`` already produced.

Usage
-----
    python examples/cat_grouping_split_conjunction.py
    python examples/cat_grouping_split_conjunction.py --amount 10
    python examples/cat_grouping_split_conjunction.py --root examples/results_cat --amount 5
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
import networkx as nx
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

Clause = Tuple[str, str, List[str]]  # (base, op, cats)


# ---------------------------------------------------------------------------
# Clause helpers: a merged node's predicate is a list of (base, op, cats)
# clauses joined by AND, with adjacent same-base/same-op clauses compacted
# into one set.
# ---------------------------------------------------------------------------

def _combine_clauses(a: List[Clause], b: List[Clause]) -> List[Clause]:
    """``a`` followed by ``b`` (in path order), compacting the boundary if
    ``a``'s last clause and ``b``'s first clause share a base + operator."""
    combined: List[Clause] = [(base, op, list(cats)) for base, op, cats in a]
    for base, op, cats in b:
        if combined and combined[-1][0] == base and combined[-1][1] == op:
            lb, lo, lc = combined[-1]
            combined[-1] = (lb, lo, lc + list(cats))
        else:
            combined.append((base, op, list(cats)))
    return combined


def _format_clauses(clauses: List[Clause]) -> str:
    return " AND ".join(_format_in_label(base, op, cats) for base, op, cats in clauses)


# ---------------------------------------------------------------------------
# Split-then-merge grouping (cross-feature conjunction variant)
# ---------------------------------------------------------------------------

def _split_then_merge(
    structure: dict,
    label_map: Dict[str, str],
) -> Tuple[Dict[str, str], Dict[Tuple[str, str], float]]:
    """Group consecutive categorical nodes as aggressively as possible,
    regardless of whether they share a base feature or operator, cloning
    shared nodes so a merge never corrupts a branch it doesn't belong to.
    Numeric nodes never participate (see module docstring).

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

    # Every categorical node starts life as a single-clause conjunction.
    # Numeric / Class nodes have no entry here at all -- that's what keeps
    # them out of every merge below.
    clauses: Dict[str, List[Clause]] = {}
    for nid, label in labels.items():
        parsed = _parse_in_label(label)
        if parsed is not None:
            base, op, cats = parsed
            clauses[nid] = [(base, op, list(cats))]

    edges: Dict[Tuple[str, str], float] = {}
    for link in graph_data.get("edges", graph_data.get("links", [])):
        src, dst = str(link["source"]), str(link["target"])
        try:
            w = float(link.get("weight", 0.0))
        except (TypeError, ValueError):
            w = 0.0
        edges[(src, dst)] = edges.get((src, dst), 0.0) + w

    settled: Set[Tuple[str, str]] = set()

    # Cross-feature merging can walk all the way around a real cycle (see
    # module docstring) with no fixed point, since folding/cloning never
    # makes progress on a loop. Find every strongly connected component of
    # size > 1 up front and permanently refuse to merge across its edges --
    # those nodes stay exactly as the one-hot rewrite produced them.
    scc_id: Dict[str, int] = {}
    scc_size: Dict[int, int] = {}
    for i, component in enumerate(nx.strongly_connected_components(nx_graph)):
        component = {str(n) for n in component}
        scc_size[i] = len(component)
        for n in component:
            scc_id[n] = i
    for (u0, v0) in edges:
        cid = scc_id.get(u0)
        if cid is not None and cid == scc_id.get(v0) and scc_size[cid] > 1:
            settled.add((u0, v0))

    clone_counter = 0

    def _degrees() -> Tuple[Dict[str, int], Dict[str, int]]:
        out_count: Dict[str, int] = {}
        in_count: Dict[str, int] = {}
        for (s, d) in edges:
            out_count[s] = out_count.get(s, 0) + 1
            in_count[d] = in_count.get(d, 0) + 1
        return out_count, in_count

    def _clone_node(v: str) -> str:
        """Private copy of ``v``: same label, same clauses, same *current*
        outgoing edges. Nothing points to it yet -- the caller wires up its
        single incoming edge."""
        nonlocal clone_counter
        clone_counter += 1
        v_copy = f"{v}#{clone_counter}"
        labels[v_copy] = labels[v]
        clauses[v_copy] = list(clauses[v])
        for (s, d), w in list(edges.items()):
            if s == v:
                edges[(v_copy, d)] = edges.get((v_copy, d), 0.0) + w
        return v_copy

    def _fold_forward(u: str, target: str, w_uv: float) -> None:
        """``u`` absorbs ``target`` (``v`` or a private clone of it): the
        clauses combine, ``target``'s outgoing edges move onto ``u`` with
        ``w_uv`` folded into each, and ``target`` disappears. Requires
        ``edges[(u, target)] == w_uv`` to already be the case."""
        clauses[u] = _combine_clauses(clauses[u], clauses[target])
        labels[u] = _format_clauses(clauses[u])
        del edges[(u, target)]
        for (s, d), w in list(edges.items()):
            if s == target:
                del edges[(s, d)]
                key = (u, d)
                edges[key] = edges.get(key, 0.0) + w + w_uv
        labels.pop(target, None)
        clauses.pop(target, None)

    max_iterations = 10 * (len(edges) + len(labels) + 10)
    for _ in range(max_iterations):
        out_count, in_count = _degrees()

        candidate = None
        for (u, v), w_uv in edges.items():
            if (u, v) in settled:
                continue
            if u not in clauses or v not in clauses:
                settled.add((u, v))
                continue
            candidate = (u, v, w_uv)
            break
        if candidate is None:
            break

        u, v, w_uv = candidate

        if out_count.get(u, 0) == 1:
            # u has nowhere else to go: it can safely disappear.
            if in_count.get(v, 0) == 1:
                target = v
            else:
                target = _clone_node(v)
                del edges[(u, v)]
                edges[(u, target)] = w_uv
            _fold_forward(u, target, w_uv)
        elif in_count.get(v, 0) == 1:
            # v belongs to u alone: merge in place, weight untouched.
            clauses[v] = _combine_clauses(clauses[u], clauses[v])
            labels[v] = _format_clauses(clauses[v])
            settled.add((u, v))
        else:
            # Both shared: clone v privately for u's branch and merge into
            # the clone, leaving the original v untouched for its other
            # parents.
            v_copy = _clone_node(v)
            clauses[v_copy] = _combine_clauses(clauses[u], clauses[v])
            labels[v_copy] = _format_clauses(clauses[v_copy])
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
        "dpg_cat_split_grouped_conjunction",
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
    """Rebuild a categorically-rewritten + cross-feature split-then-merge
    grouped DPG PNG inside ``wip_dir``. Returns the path of the produced
    PNG, or ``None`` on failure.
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
    output_name = f"{run_id}_DPG_split_grouped_conjunction"
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
        wip_dir = os.path.join(subdir, "wip", "grouping_split_conjunction")
        try:
            _process_subdir(subdir, wip_dir, visualization_config)
        except Exception as exc:  # pragma: no cover - best-effort reporting
            print(f"  [err]  {os.path.basename(subdir)}: {exc}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
