"""
categorical_view_dpg.py
========================

For each subdirectory of ``examples/results_gridsearch/`` create a ``wip/``
subdirectory and render a re-labelled copy of the DPG PNG.

Compared to the original graph, every node whose label is a *one-hot encoded
categorical predicate* of the form::

    <prefix>_<CATEGORY> <op> 0.5      (e.g. ``person_home_ownership_OWN > 0.5``)

is rewritten to a single set-membership label of the form::

    <base_feature> \u2208 {<CATEGORY>}  (e.g. ``person_home_ownership \u2208 {OWN}``)

where ``<base_feature>`` is the prefix (everything before the last ``_`` of the
feature column name) and ``<CATEGORY>`` is the trailing value of the encoded
column.

The output PNG keeps the **same name** as the original graph (just placed
inside ``wip/``) and is rendered as PNG only (no PDF, no communities plot).

Usage
-----
    python examples/categorical_view_dpg.py
    python examples/categorical_view_dpg.py --amount 10
    python examples/categorical_view_dpg.py --root examples/results_gridsearch --amount 5
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import Dict, List, Optional, Tuple

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


def _split_one_hot_column(col: str) -> Optional[Tuple[str, str]]:
    """Split a one-hot encoded column name into ``(base_feature, category)``.

    We treat the *last* underscore-separated token as the category and the
    rest as the base feature. ``None`` is returned when the column is clearly
    not a one-hot encoded categorical (no underscore, or the prefix is
    empty / the suffix looks numeric).
    """
    if "_" not in col:
        return None
    base, category = col.rsplit("_", 1)
    base = base.strip()
    category = category.strip()
    if not base or not category:
        return None
    # Avoid splitting genuinely numeric features such as ``x_1`` or ``col_42``.
    try:
        float(category)
        return None
    except ValueError:
        pass
    return base, category


def _is_categorical_predicate(label: str) -> bool:
    """``True`` when the label looks like a one-hot encoded categorical
    predicate, i.e. ``feature_CAT op 0.5``."""
    parsed = _PREDICATE_PATTERN.match(label)
    if parsed is None:
        return False
    feature = parsed.group("feature")
    value = float(parsed.group("value"))
    if not _split_one_hot_column(feature):
        return False
    # One-hot encoded columns are typically split on a 0.5 threshold.
    if abs(value - 0.5) > 1e-9:
        return False
    return True


def _to_categorical_label(label: str) -> str:
    """Convert a one-hot predicate label to a set-membership label.

    Example::

        "person_home_ownership_OWN > 0.5"  ->  "person_home_ownership \u2208 {OWN}"
        "person_home_ownership_RENT <= 0.5" -> "person_home_ownership \u2208 {RENT}"

    If the label does not look like a one-hot predicate, it is returned
    unchanged.
    """
    parsed = _PREDICATE_PATTERN.match(label)
    if parsed is None:
        return label
    feature, op, value = parsed.group("feature"), parsed.group("op"), parsed.group("value")
    split = _split_one_hot_column(feature)
    if split is None or abs(float(value) - 0.5) > 1e-9:
        return label
    base, category = split
    # op ``>`` means the dummy is set (1), ``<=`` means it is unset (0).
    if op in (">", ">="):
        return f"{base} \u2208 {{{category}}}"
    if op in ("<=", "<"):
        return f"{base} \u2209 {{{category}}}"
    return label


# ---------------------------------------------------------------------------
# Graph rebuilding helpers
# ---------------------------------------------------------------------------

def _load_structure_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _build_label_map(structure: dict) -> Tuple[Dict[str, str], Dict[str, str]]:
    """Return ``(id -> new_label, id -> old_label)`` for every node in the
    structure payload.
    """
    new_labels: Dict[str, str] = {}
    old_labels: Dict[str, str] = {}
    for node in structure.get("nodes", []):
        node_id = str(node["id"])
        old_label = str(node.get("label", ""))
        new_labels[node_id] = _to_categorical_label(old_label)
        old_labels[node_id] = old_label
    return new_labels, old_labels


def _build_edges(structure: dict) -> List[Tuple[str, str]]:
    """Edge list as ``(source_id, target_id)`` strings, in the same order
    produced by ``json_graph.node_link_data``."""
    graph_data = structure.get("graph", {})
    graph = json_graph.node_link_graph(graph_data)
    return [(str(u), str(v)) for u, v in graph.edges()]


def _build_dot(
    structure: dict,
    label_map: Dict[str, str],
    visualization_config: dict,
) -> graphviz.Digraph:
    """Rebuild a Graphviz Digraph using the (possibly rewritten) labels."""
    graph_attrs = visualization_config.get("graph", {}).get("graph_attrs", {})
    node_attrs = visualization_config.get("graph", {}).get("node_attrs", {})

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
    class_fillcolor = (
        visualization_config.get("graph", {})
        .get("class_node", {})
        .get("fillcolor")
    )

    dot = graphviz.Digraph(
        "dpg_categorical",
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

    # First pass: declare every node with its (possibly rewritten) label.
    for node in structure.get("nodes", []):
        node_id = str(node["id"])
        label = label_map.get(node_id, str(node.get("label", "")))
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

    # Second pass: add edges, mapping the encoded source/target strings to
    # the actual node ids (json_graph uses ``source``/``target`` keys).
    graph_data = structure.get("graph", {})
    for link in graph_data.get("links", []):
        src = str(link["source"])
        dst = str(link["target"])
        dot.edge(src, dst, penwidth="1", fontsize="18")

    return dot


# ---------------------------------------------------------------------------
# Per-run processing
# ---------------------------------------------------------------------------

def _load_node_metrics(path: str) -> pd.DataFrame:
    return pd.read_csv(path, index_col=0)


def _load_edge_metrics(path: str) -> pd.DataFrame:
    return pd.read_csv(path, index_col=0)


def _process_subdir(
    subdir: str,
    wip_dir: str,
    visualization_config: dict,
) -> Optional[str]:
    """Rebuild a categorically-rewritten DPG PNG inside ``wip_dir``.

    Returns the path of the produced PNG, or ``None`` on failure.
    """
    # The original graph name is the subdir basename (e.g. "ds=..._pv=...").
    run_id = os.path.basename(subdir.rstrip(os.sep))
    structure_path = os.path.join(subdir, f"{run_id}_dpg_structure.json")
    node_metrics_path = os.path.join(subdir, f"{run_id}_node_metrics.csv")
    edge_metrics_path = os.path.join(subdir, f"{run_id}_edge_metrics.csv")

    if not os.path.isfile(structure_path):
        print(f"  [skip] {run_id}: missing {os.path.basename(structure_path)}")
        return None

    structure = _load_structure_json(structure_path)
    label_map, _ = _build_label_map(structure)
    dot = _build_dot(structure, label_map, visualization_config)

    # ``plot_dpg`` needs node + edge metric DataFrames.  When unavailable,
    # build minimal ones from the structure itself so the call still works.
    if os.path.isfile(node_metrics_path):
        df_nodes = _load_node_metrics(node_metrics_path)
        # Coerce ids to str so they match the rebuilt dot and graphviz's
        # expectations (graphviz rejects non-string node identifiers).
        df_nodes = df_nodes.copy()
        df_nodes["Node"] = df_nodes["Node"].astype(str)
        # Update the "Label" column with the rewritten label so the legend /
        # any downstream column-based styling stays consistent.
        original_labels = dict(zip(df_nodes["Node"], df_nodes["Label"]))
        df_nodes["Label"] = df_nodes["Node"].map(
            lambda nid: label_map.get(nid, original_labels.get(nid, ""))
        )
    else:
        df_nodes = pd.DataFrame(
            [
                {
                    "Node": str(node["id"]),
                    "Label": label_map.get(str(node["id"]), str(node.get("label", ""))),
                }
                for node in structure.get("nodes", [])
            ]
        )

    if os.path.isfile(edge_metrics_path):
        df_edges = _load_edge_metrics(edge_metrics_path)
        df_edges = df_edges.copy()
        if "Source_id" in df_edges.columns:
            df_edges["Source_id"] = df_edges["Source_id"].astype(str)
        if "Target_id" in df_edges.columns:
            df_edges["Target_id"] = df_edges["Target_id"].astype(str)
    else:
        df_edges = pd.DataFrame(
            [
                {
                    "Source_id": src,
                    "Target_id": dst,
                    "Weight": 1.0,
                }
                for src, dst in _build_edges(structure)
            ]
        )

    os.makedirs(wip_dir, exist_ok=True)
    output_name = f"{run_id}_DPG"
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
# Main
# ---------------------------------------------------------------------------

def _iter_subdirs(root: str) -> List[str]:
    return sorted(
        entry
        for entry in (os.path.join(root, name) for name in os.listdir(root))
        if os.path.isdir(entry)
    )


def _load_visualization_config(config_path: str) -> dict:
    try:
        import yaml
    except ImportError:  # pragma: no cover - yaml is a hard dep of the project
        raise
    with open(config_path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        default=os.path.join(SCRIPT_DIR, "results_gridsearch"),
        help="Root directory containing one subdirectory per gridsearch run.",
    )
    parser.add_argument(
        "--amount",
        type=int,
        default=None,
        help="If set, only process the first N subdirectories (alphabetical order).",
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
