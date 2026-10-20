"""
cat_MINIMAL_peer.py
===================

Builds the *minimal* DPG from the outputs of ``cat_NEGATIVE_peer.py`` and
``cat_POSITIVE_peer.py``: for every categorical clause of every node,
keep whichever of its two forms (``NOT IN`` or ``IN``) lists fewer
values.

Why this is possible
--------------------
The NEGATIVE and POSITIVE graphs are both built from the same
deduplicated graph and only rewrite labels: they have the same node ids,
the same edges and the same weights, and each clause's two forms admit
exactly the same rows. So every clause can take either form without
changing what the graph does.

The choice
----------
The choice is made per clause, i.e. per attribute of a node: a
conjunctive node can mix both operators. For each attribute, the
version whose clause lists fewer values is kept; on a tie the ``IN``
version is kept::

    NEGATIVE:  marital NOT IN {divorced, single}   (2 values)
    POSITIVE:  marital IN {married}                (1 value)    <- kept

    NEGATIVE:  sex NOT IN {Female}                 (1 value)
    POSITIVE:  sex IN {Male}                       (1 value)    <- kept (tie: IN)

    NEGATIVE:  parents NOT IN {usual} AND health NOT IN {priority, recommended}
    POSITIVE:  parents IN {great-pret, pretentious} AND health IN {not_recom}
    MINIMAL:   parents NOT IN {usual} AND health IN {not_recom}

Clauses are matched by attribute, not by position. When a version holds
more than one clause on the same attribute (a contradiction POSITIVE
keeps as separate ``IN`` clauses, or a clause the ``--max-values`` cap
left in its original operator), that attribute's clauses are compared
as a group, by their total number of values, and the winning group is
kept whole.

Pieces that are not ``IN``/``NOT IN`` clauses (numeric predicates such
as ``age <= 25``, class nodes) are identical in both versions and stay
where they are.

The two inputs must describe the same graph: if their node ids, their
``graph`` sections, a node's numeric pieces or a node's set of
categorical attributes differ, the pair is rejected.

Output
------
For an input pair
``<stem>_negative_structure.json`` / ``<stem>_positive_structure.json``
the script writes, next to the NEGATIVE input:

* ``<stem>_minimal_structure.json``
* ``<stem>_minimal.png``

Every top-level field other than ``nodes`` (``graph``, ``onehot_map``,
``missing_categorical``, ...) is copied from the NEGATIVE input.

Usage
-----
One pair of files::

    python -m categorical.cat_MINIMAL_peer <negative.json> <positive.json>
    python -m categorical.cat_MINIMAL_peer <negative.json> <positive.json> --out <minimal.json>

A whole directory (walks recursively; every ``*_negative_structure.json``
is paired with the ``*_positive_structure.json`` at the same path with
``negative`` replaced by ``positive``, which covers both the
``...-NEGATIVE DPG/`` / ``...-POSITIVE DPG/`` folders of the pipeline and
two files side by side)::

    python -m categorical.cat_MINIMAL_peer outputs/categorical/

JSON only, no PNG::

    python -m categorical.cat_MINIMAL_peer <input(s)> --no-render
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Optional, Tuple

# --- Make the project root + this script's own directory importable --------
SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, SCRIPT_DIR)

from cat_grouping import (  # noqa: E402
    _format_in_label,
    _load_visualization_config,
    _parse_in_label,
)
from cat_duplicate_split_peer import (  # noqa: E402
    _load_structure_json,
    _render_deduplicated_png,
    _save_structure_json,
)


class MismatchedPairError(ValueError):
    """The NEGATIVE and POSITIVE structures do not describe the same graph."""


# ---------------------------------------------------------------------------
# Core: pick the form with fewer values per clause
# ---------------------------------------------------------------------------

# A label split into its ordered pieces and its categorical clauses:
#   pieces:  [("raw", text) | ("cat", base), ...] -- one "cat" slot per base,
#            at the position of the base's first clause
#   clauses: {base: [clause text, ...]}            -- in label order
#   values:  {base: total number of listed values}
SplitLabel = Tuple[List[Tuple[str, str]], Dict[str, List[str]], Dict[str, int]]


def _labels_by_id(structure: dict) -> Dict[str, str]:
    return {
        str(n.get("id", "")): str(n.get("label", ""))
        for n in structure.get("nodes", [])
    }


def _split_label(label: str) -> SplitLabel:
    """Split ``label`` on ``" AND "`` and group its IN/NOT IN clauses by base."""
    pieces: List[Tuple[str, str]] = []
    clauses: Dict[str, List[str]] = {}
    values: Dict[str, int] = {}
    for piece in label.split(" AND "):
        parsed = _parse_in_label(piece)
        if parsed is None:
            pieces.append(("raw", piece))
            continue
        base, op, cats = parsed
        if base not in clauses:
            clauses[base] = []
            values[base] = 0
            pieces.append(("cat", base))
        # Re-rendered so both versions are spelled the same way.
        clauses[base].append(_format_in_label(base, op, cats))
        values[base] += len(cats)
    return pieces, clauses, values


def _minimize_label(
    nid: str, neg_label: str, pos_label: str, stats: Dict[str, int]
) -> str:
    """Combine a node's NEGATIVE and POSITIVE labels clause by clause.

    For each base, the version whose clause(s) list fewer values is kept;
    on a tie the POSITIVE (``IN``) version is kept. Raw pieces (numeric
    predicates, class labels) must be identical in both versions and keep
    their position, which follows the NEGATIVE label.

    Raises ``MismatchedPairError`` when the two labels do not have the
    same raw pieces or the same set of bases.
    """
    neg_pieces, neg_clauses, neg_values = _split_label(neg_label)
    pos_pieces, pos_clauses, pos_values = _split_label(pos_label)
    if [p for p in neg_pieces if p[0] == "raw"] != [p for p in pos_pieces if p[0] == "raw"]:
        raise MismatchedPairError(f"node {nid}: numeric/class pieces differ between versions")
    if set(neg_clauses) != set(pos_clauses):
        raise MismatchedPairError(f"node {nid}: categorical attributes differ between versions")

    out: List[str] = []
    for kind, value in neg_pieces:
        if kind == "raw":
            out.append(value)
            continue
        neg, pos = neg_clauses[value], pos_clauses[value]
        stats["values_negative"] += neg_values[value]
        stats["values_positive"] += pos_values[value]
        if neg == pos:
            chosen = neg
            stats["identical"] += 1
        elif neg_values[value] < pos_values[value]:
            chosen = neg
            stats["from_negative"] += 1
        else:
            chosen = pos
            stats["from_positive"] += 1
            stats["ties"] += neg_values[value] == pos_values[value]
        stats["values_minimal"] += neg_values[value] if chosen is neg else pos_values[value]
        out.extend(chosen)
    return " AND ".join(out)


def _build_minimal(negative: dict, positive: dict) -> Tuple[dict, Dict[str, int]]:
    """Return ``(minimal_structure, stats)``.

    ``minimal_structure`` is a copy of ``negative`` in which every
    categorical clause takes the form (``NOT IN`` from NEGATIVE or ``IN``
    from POSITIVE) that lists fewer values, ``IN`` on a tie. ``stats``
    counts, per clause (one per base per node), where the clause came
    from, plus the listed values and predicate characters of each version.

    Raises ``MismatchedPairError`` when the two structures do not have the
    same node ids, the same ``graph`` section, or matching node labels.
    """
    neg_labels = _labels_by_id(negative)
    pos_labels = _labels_by_id(positive)
    if set(neg_labels) != set(pos_labels):
        raise MismatchedPairError("NEGATIVE and POSITIVE structures have different node ids")
    if negative.get("graph") != positive.get("graph"):
        raise MismatchedPairError("NEGATIVE and POSITIVE structures have different graphs")

    stats = {"from_negative": 0, "from_positive": 0, "identical": 0, "ties": 0,
             "values_negative": 0, "values_positive": 0, "values_minimal": 0,
             "chars_negative": 0, "chars_positive": 0, "chars_minimal": 0}
    minimal = dict(negative)
    minimal_nodes: List[dict] = []
    for node in negative.get("nodes", []):
        node = dict(node)
        nid = str(node.get("id", ""))
        if "->" not in nid:  # edge-label pseudo-nodes carry no predicate
            neg_label, pos_label = neg_labels[nid], pos_labels[nid]
            node["label"] = _minimize_label(nid, neg_label, pos_label, stats)
            stats["chars_negative"] += len(neg_label)
            stats["chars_positive"] += len(pos_label)
            stats["chars_minimal"] += len(node["label"])
        minimal_nodes.append(node)
    minimal["nodes"] = minimal_nodes
    return minimal, stats


# ---------------------------------------------------------------------------
# File / directory walking
# ---------------------------------------------------------------------------

_NEGATIVE_SUFFIX = "_negative_structure.json"


def _positive_path_for(negative_path: str) -> str:
    """The POSITIVE counterpart of a NEGATIVE structure path: ``negative``
    becomes ``positive`` in the file name and ``NEGATIVE`` becomes
    ``POSITIVE`` in the folder name (the pipeline's
    ``...-DEDUPLICATED-NEGATIVE DPG/`` folder)."""
    folder, name = os.path.split(negative_path)
    folder = os.path.join(
        os.path.dirname(folder), os.path.basename(folder).replace("NEGATIVE", "POSITIVE")
    )
    return os.path.join(folder, name[: -len(_NEGATIVE_SUFFIX)] + "_positive_structure.json")


def _iter_pairs(path: str) -> List[Tuple[str, str]]:
    """Every ``(negative, positive)`` structure pair under ``path``
    (recursive), in sorted order. NEGATIVE files without a POSITIVE
    counterpart are reported and skipped."""
    pairs: List[Tuple[str, str]] = []
    for root, dirs, files in os.walk(path):
        dirs.sort()
        for f in sorted(files):
            if not f.endswith(_NEGATIVE_SUFFIX):
                continue
            neg = os.path.join(root, f)
            pos = _positive_path_for(neg)
            if os.path.isfile(pos):
                pairs.append((neg, pos))
            else:
                print(f"  [skip] {neg}: no POSITIVE counterpart at {pos}", file=sys.stderr)
    return pairs


def _default_output_path(negative_path: str) -> str:
    """``<stem>_negative_structure.json`` -> ``<stem>_minimal_structure.json``
    next to the NEGATIVE input."""
    if negative_path.endswith(_NEGATIVE_SUFFIX):
        return negative_path[: -len(_NEGATIVE_SUFFIX)] + "_minimal_structure.json"
    return negative_path + ".minimal"


def _default_png_path(json_path: str) -> str:
    """``<stem>_structure.json`` -> ``<stem>.png`` in the same folder."""
    base = os.path.basename(json_path)
    if base.endswith("_structure.json"):
        stem = base[: -len("_structure.json")]
    else:
        stem = os.path.splitext(base)[0]
    return os.path.join(os.path.dirname(json_path), stem + ".png")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _process_pair(
    negative_path: str,
    positive_path: str,
    out_path: Optional[str],
    render: bool,
    visualization_config: Optional[dict],
) -> Tuple[str, Dict[str, int], Optional[str]]:
    negative = _load_structure_json(negative_path)
    positive = _load_structure_json(positive_path)
    minimal, stats = _build_minimal(negative, positive)
    target = out_path or _default_output_path(negative_path)
    _save_structure_json(minimal, target)

    png_target: Optional[str] = None
    if render:
        if visualization_config is None:
            raise RuntimeError(
                "internal: render=True requires a visualization_config "
                "to be passed to _process_pair"
            )
        png_target = _default_png_path(target)
        _render_deduplicated_png(minimal, png_target, visualization_config)

    return target, stats, png_target


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="Either a NEGATIVE and a POSITIVE structure JSON (in this "
             "order), or one directory to search for pairs (recursive).",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output path. Only valid with a single pair of files. "
             "Defaults to ``<stem>_minimal_structure.json`` next to the "
             "NEGATIVE input.",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Print only the per-pair summary lines (no header banner).",
    )
    parser.add_argument(
        "--no-render",
        action="store_true",
        help="Skip PNG rendering; only emit the minimal JSON.",
    )
    parser.add_argument(
        "--config",
        default=os.path.join(PROJECT_ROOT, "config.yaml"),
        help="Path to the YAML config used to style the rebuilt DPG. "
             "Ignored when --no-render is set.",
    )
    args = parser.parse_args()

    if len(args.inputs) == 2 and all(os.path.isfile(p) for p in args.inputs):
        pairs = [(args.inputs[0], args.inputs[1])]
    elif len(args.inputs) == 1 and os.path.isdir(args.inputs[0]):
        pairs = _iter_pairs(args.inputs[0])
    else:
        print(
            "Pass either two structure files (NEGATIVE then POSITIVE) or one directory.",
            file=sys.stderr,
        )
        return 1
    if not pairs:
        print(f"No NEGATIVE/POSITIVE pairs found in {args.inputs!r}", file=sys.stderr)
        return 1

    if args.out and len(pairs) > 1:
        print(
            f"--out can only be used with a single pair; got {len(pairs)} pairs",
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
        mode = "minimize + render" if render else "minimize only"
        print(f"Processing {len(pairs)} NEGATIVE/POSITIVE pair(s) ({mode}).")

    n_failed = 0
    for neg, pos in pairs:
        try:
            out_path, stats, png_path = _process_pair(
                neg, pos, args.out, render, visualization_config,
            )
        except MismatchedPairError as exc:
            print(f"  [skip] {neg}: {exc}", file=sys.stderr)
            n_failed += 1
            continue
        print(
            f"  [ok] {neg} -> {out_path}  "
            f"(clauses: {stats['from_negative']} NOT IN, {stats['from_positive']} IN "
            f"[{stats['ties']} by tie], {stats['identical']} identical; "
            f"values {stats['values_negative']} NEG / {stats['values_positive']} POS -> "
            f"{stats['values_minimal']}; "
            f"chars {stats['chars_negative']} NEG / {stats['chars_positive']} POS -> "
            f"{stats['chars_minimal']}"
            + (f"; PNG -> {png_path})" if png_path else ")")
        )

    return 1 if n_failed == len(pairs) else 0


if __name__ == "__main__":
    raise SystemExit(main())
