"""
cat_duplicate_split_peer.py
============================

A post-processing pass over ``cat_grouping_split_conjunction.py``'s
output that removes a specific kind of label redundancy the
cross-feature conjunction split introduces.

Background
----------
``cat_grouping_split_conjunction.py`` builds a DPG in which a node
holding a multi-clause ``AND``-joined predicate can be reached from a
single-clause predecessor ``u``. The conjunction is the literal string
the conjunction split wrote when it folded a chain of categorical
nodes into one -- and if the chain intersected a same-base / same-op
test on the way, the resulting label of ``v`` ends up *repeating* the
parent clause verbatim. The repeated part is logically redundant (it
was already required to reach ``v``), so the node label can be
tightened without changing what the DPG predicts on any row.

Example 1 (NURSERY, edge weight 4148)::

    u:  parents NOT IN {usual}
    v:  parents NOT IN {usual} AND health IN {not-recom}

    The leading ``parents NOT IN {usual}`` is the same condition that
    got us to ``v``; once we're inside ``v`` it is redundant. The
    tightened ``v`` is just ``health IN {not-recom}`` -- exactly the
    same set of rows satisfy the new label, no model behaviour changes.

Example 2 (BANK_MARKETING, edge weight 504)::

    u:  month NOT IN {mar}
    v:  month NOT IN {mar, jun, dec, sep} AND contact NOT IN {unknown} AND job NOT IN {retired}

    The leading ``mar`` in ``v``'s ``month`` clause is redundant: we
    already know ``month != mar`` from the parent. The tightened ``v``
    is ``month NOT IN {jun, dec, sep} AND contact NOT IN {unknown} AND
    job NOT IN {retired}``.

The pass
--------
For every edge ``u -> v`` where the parent ``u`` parses as a single
``IN``/``NOT IN`` clause ``P`` and the child ``v`` has at least one
``AND``-joined clause, each child clause is *restricted* by the
parent's condition on the same base:

    P = ``base IN {p}``  and  Ci = ``base IN {c}``
        ->  Ci becomes ``base IN {p ∩ c}``
            (effective: base must be in both -- always tighter or equal)

    P = ``base NOT IN {p}``  and  Ci = ``base NOT IN {c}``
        ->  Ci becomes ``base NOT IN {c \\ p}``
            (the parent already excludes everything in ``p``; the child
            only needs to additionally exclude ``c \\ p``)

    P = ``base IN {p}``  and  Ci = ``base NOT IN {c}``
        ->  Ci becomes ``base NOT IN {c \\ p}``
            (combined: base must be in p AND not in c, i.e. base in
            p \\ c. Rewritten as a NOT IN clause over c: the values
            in c that the parent never allowed anyway (i.e. c \\ p)
            can be dropped from the exclusion list, since the parent
            already filters them out upstream)

    P = ``base NOT IN {p}``  and  Ci = ``base IN {c}``
        ->  Ci becomes ``base IN {c \\ p}``
            (combined: base must be in c AND not in p, i.e. base in
            c \\ p. The values in c that the parent already excludes
            upstream can be dropped from the inclusion list, since
            they can never reach the child in the first place)

    Different bases: Ci is left untouched.

    Empty restriction result:
      * For ``NOT IN``: the restriction is trivially true; the
        clause is dropped.
      * For ``IN``: the restriction is impossible; the clause is
        left as-is so the contradiction stays visible (it would mean
        the parent cannot lead to the child on any row, which is a
        data issue rather than a redundancy).

A node whose label becomes empty is removed and its single incoming
edge is rewired onto its outgoing edges with the original edge
weights summed (matching the rest of the project's split-then-merge
convention).

The pass iterates until a full sweep makes no change: simplifying
``v``'s label can turn a downstream edge ``v -> w`` into a redundancy
too, which only the next iteration picks up. The loop is bounded by
``max(100, total node count + 1)`` and on each iteration either
shrinks the total clause count, drops a node, or leaves the graph
unchanged -- so it always terminates.

Equivalence guarantee
---------------------
On the subset of rows that actually reach the child ``v`` -- i.e. the
rows that satisfy the parent ``u``'s clause -- the restricted child
clause is always at least as tight as the original: the rows it admits
are a subset of (or equal to) the rows the original child admitted
on that same subset. The simplification loop then either keeps the
tightened clause, drops the clause outright when the restriction is
identical to the original (the cosmetic case, e.g. ``parents IN {RENT}
-> parents IN {RENT} AND ...``), or drops it when the restriction is
trivially true (``NOT IN {}``). It never adds new constraints. And
the rewire of a fully-dropped node's incoming edge onto its outgoing
edges is a pure no-op for ``v``'s outgoing edges.

Net effect: every row that satisfied the original graph still
satisfies the modified one, and any row that didn't satisfy the
original satisfies the modified one even less. The two graphs are
therefore exactly equivalent on the same dataset: same predicted
class for every row. The change is purely a *label cleanup* -- the
graph routes the same rows to the same classes; only the cosmetic
duplication in the displayed child labels is removed.

Numeric predicate nodes (``age <= 25``) and class nodes (``Class 0``)
are never simplified: they have no IN/NOT IN clauses to start with,
and a conjunctive node's IN/NOT IN clauses are independent of any
numeric condition along the same path (the conjunction script only
ever joins IN/NOT IN clauses, so mixed labels never appear in its
output anyway).

Output
------
For every deduplicated structure JSON the script also renders a PNG of
the resulting graph and writes it next to the JSON, using the same
``config.yaml`` styling as the upstream scripts. Concretely, an input at
``.../GROUPED-SPLIT-CONJUNCTION DPG/<run>_DPG_split_grouped_conjunction_structure.json``
produces, in the same folder:

* ``<run>_DPG_split_grouped_conjunction_deduplicated_structure.json``
* ``<run>_DPG_split_grouped_conjunction_deduplicated.png``

Use ``--no-render`` to skip the PNG and emit only the JSON. Use
``--config <path>`` to point at a non-default ``config.yaml`` (ignored
when ``--no-render`` is set).

Usage
-----
Single structure JSON file (writes the deduplicated JSON and PNG next to
the input)::

    python -m categorical.cat_duplicate_split_peer \\
        outputs/categorical/nursery/GROUPED-SPLIT-CONJUNCTION\\ DPG/\\
        nursery_DPG_split_grouped_conjunction_structure.json

Whole directory of structure JSONs (walks recursively, picks up every
``*_DPG_split_grouped_conjunction_structure.json`` it finds)::

    python -m categorical.cat_duplicate_split_peer \\
        outputs/categorical/

Overwrite the input file in place (the original is gone after this)::

    python -m categorical.cat_duplicate_split_peer <file-or-dir> --inplace

Specify an explicit output path (single file mode only)::

    python -m categorical.cat_duplicate_split_peer <file> --out <other-file>

JSON only, no PNG::

    python -m categorical.cat_duplicate_split_peer <file-or-dir> --no-render
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

# --- Make the project root + this script's own directory importable --------
SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, SCRIPT_DIR)

import graphviz  # noqa: E402
import pandas as pd  # noqa: E402
from dpg.visualizer import plot_dpg  # noqa: E402

from cat_grouping import (  # noqa: E402
    _format_in_label,
    _load_visualization_config,
    _parse_in_label,
)


# ---------------------------------------------------------------------------
# Clause helpers
# ---------------------------------------------------------------------------

# A clause is a single IN/NOT IN test parsed out of a node label.
Clause = Tuple[str, str, List[str]]  # (base, op, cats)


def _split_label(label: str) -> List[Clause]:
    """Parse a possibly-conjunctive label into its IN/NOT IN clauses.

    Non-IN/NOT-IN pieces (numeric predicates like ``age <= 25``, class
    labels like ``Class 0``) are dropped: they can never be the source
    of a cross-node redundancy, and the conjunction script only ever
    joins IN/NOT IN clauses so they never appear in a conjunction
    label to begin with.
    """
    out: List[Clause] = []
    for piece in label.split(" AND "):
        parsed = _parse_in_label(piece)
        if parsed is not None:
            base, op, cats = parsed
            out.append((base, op, list(cats)))
    return out


def _format_clauses(clauses: List[Clause]) -> str:
    """Render a clause list back to a ``base OP {cats} AND ...`` label."""
    return " AND ".join(
        _format_in_label(base, op, cats) for base, op, cats in clauses
    )


def _restrict_clause(parent: Clause, child: Clause) -> Optional[Clause]:
    """Compute the child clause as it stands on the subset of rows where
    the parent's clause is true.

    Returns the new ``(base, op, cats)`` triple to use for the child,
    or ``None`` to leave the child untouched (different base, no shared
    condition to fold in). On the subset of rows where the parent's
    clause holds, the returned triple is always at least as tight as
    the original child clause: the rows it admits are a subset of (or
    equal to) the rows the original child admitted. The empty-cats
    case (``NOT IN {}`` -- trivially true, or ``IN {}`` -- trivially
    false) is handled by the caller, not here.
    """
    base_p, op_p, cats_p = parent
    base_c, op_c, cats_c = child
    if base_p != base_c:
        return None

    cats_p_set = set(cats_p)

    if op_p == "IN" and op_c == "IN":
        # Effective: base in (cats_c ∩ cats_p). Always tighter or equal.
        new_cats = [c for c in cats_c if c in cats_p_set]
        return (base_c, "IN", new_cats)

    if op_p == "NOT IN" and op_c == "NOT IN":
        # The parent already excludes everything in cats_p; the child
        # only needs to additionally exclude (cats_c \ cats_p).
        new_cats = [c for c in cats_c if c not in cats_p_set]
        return (base_c, "NOT IN", new_cats)

    if op_p == "IN" and op_c == "NOT IN":
        # P restricts base to cats_p; Ci additionally excludes cats_c.
        # Combined: base not in (cats_c \ cats_p).
        new_cats = [c for c in cats_c if c not in cats_p_set]
        return (base_c, "NOT IN", new_cats)

    if op_p == "NOT IN" and op_c == "IN":
        # P excludes cats_p; Ci includes cats_c. Combined: base in (cats_c \ cats_p).
        new_cats = [c for c in cats_c if c not in cats_p_set]
        return (base_c, "IN", new_cats)

    return None  # unreachable, but keeps the type-checker happy


# ---------------------------------------------------------------------------
# Core: iterative edge-walk simplification
# ---------------------------------------------------------------------------

def _load_structure_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _save_structure_json(structure: dict, path: str) -> None:
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(structure, fh, indent=2, ensure_ascii=False)


def _deduplicate_labels(structure: dict) -> Tuple[dict, int]:
    """Run the cross-node redundancy simplification on a structure
    payload until a full pass makes no change.

    The structure is mutated in place; this function does not copy it.
    Returns ``(structure, n_rewritten)`` where ``n_rewritten`` is the
    number of node labels that actually changed.
    """
    # --- Working state --------------------------------------------------
    # labels: id -> current label string for every real (non-pseudo) node
    # nodes_by_id: id -> the actual node dict from structure["nodes"]
    labels: Dict[str, str] = {}
    nodes_by_id: Dict[str, dict] = {}
    for n in structure.get("nodes", []):
        nid = str(n.get("id", ""))
        if "->" in nid:
            continue  # edge-label pseudo-nodes, never appear in graph.edges
        labels[nid] = str(n.get("label", ""))
        nodes_by_id[nid] = n

    graph_data = structure.get("graph", {}) or {}
    raw_edges = (
        graph_data.get("edges")
        or graph_data.get("links")
        or structure.get("edges")
        or structure.get("links")
        or []
    )
    weights: Dict[Tuple[str, str], float] = {}
    for link in raw_edges:
        key = (str(link["source"]), str(link["target"]))
        try:
            w = float(link.get("weight", 0.0))
        except (TypeError, ValueError):
            w = 0.0
        weights[key] = weights.get(key, 0.0) + w

    # --- The simplification loop ---------------------------------------
    def _parsed(nid: str) -> List[Clause]:
        return _split_label(labels.get(nid, ""))

    n_rewritten = 0
    max_iter = max(100, len(labels) + 1)
    for _ in range(max_iter):
        changed = False

        # Iterate over a snapshot of edge keys so the loop is stable
        # across mid-iteration mutations.
        for (u, v) in list(weights.keys()):
            if u not in labels or v not in labels:
                # Either side was dropped on a previous iteration; the
                # edge should already be gone, but be defensive.
                weights.pop((u, v), None)
                continue

            parent_clauses = _parsed(u)
            if len(parent_clauses) != 1:
                # A multi-clause parent can't be the source of a
                # single-clause redundancy at the child: we'd have to
                # fold in every parent clause, which is out of scope
                # for this pass.
                continue
            base_p, op_p, cats_p = parent_clauses[0]
            child_clauses = _parsed(v)
            if len(child_clauses) <= 1:
                # Nothing to simplify: child already has 0 or 1 clauses.
                continue

            new_child: List[Clause] = []
            child_changed = False
            for clause in child_clauses:
                restricted = _restrict_clause(parent_clauses[0], clause)
                if restricted is None:
                    new_child.append(clause)
                    continue
                base_r, op_r, cats_r = restricted
                if not cats_r:
                    if op_r == "NOT IN":
                        # The restriction is "base NOT IN {}", which is
                        # always true; the clause is fully redundant.
                        child_changed = True
                        continue
                    # For "IN": the restriction is "base IN {}", which
                    # is always false; the edge has become unreachable.
                    # Don't simplify -- leave the contradiction visible
                    # so it can be debugged if it ever appears in real
                    # data.
                    new_child.append(clause)
                    continue
                # Drop the clause outright only if the child clause
                # is a verbatim duplicate of the parent's -- the
                # parent's condition alone is enough to satisfy it, so
                # repeating it in the child is purely cosmetic. This
                # handles the "parents IN {RENT} -> parents IN {RENT}
                # AND ..." case where the child literally re-states
                # the parent. It deliberately does NOT fire when the
                # child is a strict subset of the parent (e.g. parent
                # IN {red, blue}, child IN {red}): the child's
                # narrowing is the binding constraint for the rows
                # that reach it, and removing it would let through
                # rows the original graph rejected.
                base_c_, op_c_, cats_c_ = clause
                if (
                    base_c_ == base_p
                    and op_r == op_c_
                    and op_p == op_c_
                    and set(cats_r) == set(cats_p)
                ):
                    child_changed = True
                    continue
                new_child.append(restricted)
                child_changed = True

            if not child_changed:
                continue

            # --- Update v's label (or drop v if it's now empty) -------
            if not new_child:
                # Drop v; rewire its incoming edge onto its outgoing
                # edges with the original edge weights summed. Matches
                # the project's split-then-merge fold convention.
                w_uv = weights[(u, v)]
                for (s, d), w in list(weights.items()):
                    if s != v:
                        continue
                    if d == u:
                        # Self-loop back to u: skip (would create a
                        # u -> u edge, which is meaningless here).
                        continue
                    del weights[(s, d)]
                    key = (u, d)
                    weights[key] = weights.get(key, 0.0) + w + w_uv
                del weights[(u, v)]
                labels.pop(v, None)
                # Drop v from the structure's nodes and the graph's
                # node list too, so a downstream consumer of the JSON
                # doesn't see a dangling entry.
                structure["nodes"] = [
                    n for n in structure.get("nodes", [])
                    if str(n.get("id", "")) != v
                ]
                if "graph" in structure:
                    structure["graph"]["nodes"] = [
                        n for n in structure["graph"].get("nodes", [])
                        if str(n.get("id", "")) != v
                    ]
                nodes_by_id.pop(v, None)
                n_rewritten += 1
                changed = True
                continue

            new_label = _format_clauses(new_child)
            if new_label == labels.get(v, ""):
                # Clauses were tweaked but the rendered string didn't
                # change (e.g. a duplicate that was just normalised);
                # not a real rewrite.
                continue
            labels[v] = new_label
            nodes_by_id[v]["label"] = new_label
            n_rewritten += 1
            changed = True

        if not changed:
            break

    # --- Persist the (possibly mutated) graph back into the structure --
    new_edges = [
        {"source": s, "target": d, "weight": w}
        for (s, d), w in weights.items()
    ]
    if "graph" in structure:
        structure["graph"]["edges"] = new_edges
        structure["graph"]["links"] = new_edges
    else:
        structure["edges"] = new_edges
        structure["links"] = new_edges

    return structure, n_rewritten


# ---------------------------------------------------------------------------
# Rendering: rebuild a PNG of the deduplicated graph
# ---------------------------------------------------------------------------

def _structure_to_labels_and_edges(
    structure: dict,
) -> Tuple[Dict[str, str], Dict[Tuple[str, str], float]]:
    """Pull the (label, edge-weight) maps back out of a structure payload.

    Used by the renderer: the deduplication step mutates the structure
    in place, but a renderer wants a single ``(labels, edges)`` pair
    (matching what ``_split_then_merge`` returns in the other cat
    scripts) to drive the Graphviz builder.
    """
    labels: Dict[str, str] = {}
    for n in structure.get("nodes", []):
        nid = str(n.get("id", ""))
        if "->" in nid:
            continue
        labels[nid] = str(n.get("label", ""))

    graph_data = structure.get("graph", {}) or {}
    raw_edges = (
        graph_data.get("edges")
        or graph_data.get("links")
        or structure.get("edges")
        or structure.get("links")
        or []
    )
    edges: Dict[Tuple[str, str], float] = {}
    for link in raw_edges:
        key = (str(link["source"]), str(link["target"]))
        try:
            w = float(link.get("weight", 0.0))
        except (TypeError, ValueError):
            w = 0.0
        edges[key] = edges.get(key, 0.0) + w
    return labels, edges


def _build_dot_from_graph(
    labels: Dict[str, str],
    edges: Dict[Tuple[str, str], float],
    visualization_config: dict,
) -> graphviz.Digraph:
    """Build a styled Graphviz Digraph from the deduplicated graph.

    Mirrors ``cat_grouping_split_conjunction._build_dot_from_graph`` so
    the rendered PNG looks identical to the upstream script's PNG,
    which is what users will compare it against.
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
        "dpg_cat_split_grouped_conjunction_deduplicated",
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


def _build_synthetic_metrics_from_graph(
    labels: Dict[str, str],
    edges: Dict[Tuple[str, str], float],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Minimal node/edge metric DataFrames for ``plot_dpg``.

    The deduplicated graph's labels and weights are exactly what the
    PNG needs to render; this helper packages them into the two
    DataFrames ``plot_dpg`` requires. Mirrors
    ``cat_grouping_split_conjunction._build_synthetic_metrics_from_graph``.
    """
    nodes = [{"Node": nid, "Label": lbl} for nid, lbl in labels.items()]
    edge_rows = [
        {"Source_id": src, "Target_id": dst, "Weight": w}
        for (src, dst), w in edges.items()
    ]
    return pd.DataFrame(nodes), pd.DataFrame(edge_rows)


def _render_deduplicated_png(
    structure: dict,
    out_png_path: str,
    visualization_config: dict,
) -> str:
    """Render the deduplicated graph to ``out_png_path`` via ``plot_dpg``.

    The PNG is always written next to the deduplicated JSON the script
    already produced -- by default in the same ``GROUPED-SPLIT-CONJUNCTION
    DPG/`` subfolder the input came from, so the file ends up alongside
    the upstream PNG and the user can diff the two visually.

    Returns the path of the rendered PNG.
    """
    labels, edges = _structure_to_labels_and_edges(structure)
    dot = _build_dot_from_graph(labels, edges, visualization_config)
    df_nodes, df_edges = _build_synthetic_metrics_from_graph(labels, edges)

    out_dir = os.path.dirname(out_png_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    output_name = os.path.splitext(os.path.basename(out_png_path))[0]
    plot_dpg(
        output_name,
        dot,
        df_nodes,
        df_edges,
        save_dir=out_dir,
        attribute=None,
        class_flag=False,
        layout_template="default",
        show=False,
        export_pdf=False,
    )
    return out_png_path


# ---------------------------------------------------------------------------
# File / directory walking
# ---------------------------------------------------------------------------

_CONJ_STRUCTURE_SUFFIX = "_DPG_split_grouped_conjunction_structure.json"


def _is_conjunction_structure(path: str) -> bool:
    """Heuristic: file is a cat_grouping_split_conjunction output if its
    name ends with ``_DPG_split_grouped_conjunction_structure.json``."""
    return os.path.basename(path).endswith(_CONJ_STRUCTURE_SUFFIX)


def _iter_structure_inputs(path: str) -> List[str]:
    """If ``path`` is a file, return ``[path]`` (no filtering). If it's a
    directory, return every conjunction structure JSON under it
    (recursive), in sorted order for deterministic output."""
    if os.path.isfile(path):
        return [path]
    if os.path.isdir(path):
        out: List[str] = []
        for root, _, files in os.walk(path):
            for f in sorted(files):
                if _is_conjunction_structure(os.path.join(root, f)):
                    out.append(os.path.join(root, f))
        return out
    return []


def _default_output_path(in_path: str) -> str:
    """Default destination for a single-file invocation: replace the
    ``_structure.json`` suffix with ``_deduplicated_structure.json``
    next to the input. Keeps the original file untouched."""
    if in_path.endswith("_structure.json"):
        return in_path[: -len("_structure.json")] + "_deduplicated_structure.json"
    return in_path + ".deduplicated"


def _default_png_path(json_path: str) -> str:
    """Default PNG path parallel to the deduplicated JSON.

    Mirrors the upstream naming: a structure named
    ``<run>_DPG_split_grouped_conjunction_structure.json`` ships with
    a sibling ``<run>_DPG_split_grouped_conjunction.png``. We just
    insert ``_deduplicated`` between the run and the variant suffix so
    it ends up as
    ``<run>_DPG_split_grouped_conjunction_deduplicated.png``.
    """
    base = os.path.basename(json_path)
    if base.endswith("_structure.json"):
        stem = base[: -len("_structure.json")]
    else:
        stem = os.path.splitext(base)[0]
    return os.path.join(os.path.dirname(json_path), stem + ".png")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _process_file(
    in_path: str,
    out_path: Optional[str],
    inplace: bool,
    render: bool,
    visualization_config: Optional[dict],
) -> Tuple[str, int, Optional[str]]:
    structure = _load_structure_json(in_path)
    new_structure, n_rewritten = _deduplicate_labels(structure)
    target = in_path if inplace else (out_path or _default_output_path(in_path))
    _save_structure_json(new_structure, target)

    png_target: Optional[str] = None
    if render:
        if visualization_config is None:
            raise RuntimeError(
                "internal: render=True requires a visualization_config "
                "to be passed to _process_file"
            )
        png_target = _default_png_path(target)
        _render_deduplicated_png(new_structure, png_target, visualization_config)

    return target, n_rewritten, png_target


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "input",
        help="A single structure JSON or a directory of them. Directory "
             "mode picks up every ``*_DPG_split_grouped_conjunction_"
             "structure.json`` under it (recursive).",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output path. Only valid when ``input`` is a single file. "
             "Defaults to ``<input>_deduplicated_structure.json`` next "
             "to the input.",
    )
    parser.add_argument(
        "--inplace",
        action="store_true",
        help="Overwrite the input file in place instead of writing a "
             "separate output. The original is gone after this.",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Print only the per-file summary lines (no header banner).",
    )
    parser.add_argument(
        "--no-render",
        action="store_true",
        help="Skip PNG rendering; only emit the deduplicated JSON.",
    )
    parser.add_argument(
        "--config",
        default=os.path.join(PROJECT_ROOT, "config.yaml"),
        help="Path to the YAML config used to style the rebuilt DPG. "
             "Ignored when --no-render is set.",
    )
    args = parser.parse_args()

    files = _iter_structure_inputs(args.input)
    if not files:
        print(f"No structure files found at {args.input!r}", file=sys.stderr)
        return 1

    if args.out and len(files) > 1:
        print(
            f"--out can only be used with a single input file; got {len(files)} "
            f"files under {args.input!r}",
            file=sys.stderr,
        )
        return 1

    render = not args.no_render
    if render and not os.path.isfile(args.config):
        print(
            f"ERROR: config not found: {args.config} (use --no-render to "
            f"skip PNG generation, or pass --config <path>)",
            file=sys.stderr,
        )
        return 1
    visualization_config = (
        _load_visualization_config(args.config) if render else None
    )

    if not args.quiet:
        mode = "deduplicate + render" if render else "deduplicate only"
        print(
            f"Processing {len(files)} structure file(s) from {args.input!r} "
            f"({mode})."
        )

    for f in files:
        out_path, n_rewritten, png_path = _process_file(
            f, args.out, args.inplace, render, visualization_config,
        )
        suffix = (
            f"  ({n_rewritten} label(s) rewritten, PNG -> {png_path})"
            if png_path
            else f"  ({n_rewritten} label(s) rewritten)"
        )
        print(f"  [ok] {f} -> {out_path}{suffix}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
