"""
cat_NEGATIVE_peer.py
====================

A post-processing pass over ``cat_duplicate_split_peer.py``'s output
that rewrites every categorical clause of the DPG into its ``NOT IN``
form, producing a fully *negative* graph.

The pass
--------
Each node label is split on ``" AND "`` and every ``IN`` clause is
replaced by the equivalent ``NOT IN`` over the complement of its
categories within the attribute's domain ``D``::

    base IN {S}   ->   base NOT IN {D \\ S}

Example (``status`` with domain ``{single, married, divorced}``)::

    status IN {single}                  ->  status NOT IN {married, divorced}
    status IN {single} AND job IN {it}  ->  status NOT IN {married, divorced} AND job NOT IN {<rest of job>}

``NOT IN`` clauses are left as they are. Numeric predicates
(``age <= 25``) and class nodes (``Class 0``) are never touched. If a
node ends up with more than one ``NOT IN`` clause on the same base,
they are merged into one clause over the union of their categories
(``x NOT IN A AND x NOT IN B`` == ``x NOT IN A ∪ B``).

The graph topology (nodes, edges, weights) is not modified: only labels
change, so the negative graph routes every row exactly as the input
graph does.

Attribute domains
-----------------
The domain of each categorical attribute is read from the structure
JSON's ``onehot_map`` (``{dummy: [base, category]}``), built with
``pd.get_dummies(drop_first=False)``, so it lists every category.
Structures without ``onehot_map`` (outputs that predate it) are skipped
with a message; domains are never guessed from feature names.

Clauses that cannot be converted are left as ``IN`` and reported:

* the base has no known domain, or ``S`` holds categories outside it;
* ``S`` covers the whole domain -- the complement is empty and the
  ``base NOT IN {...}`` label grammar cannot express ``NOT IN {}``;
* the complement holds a category the label grammar cannot round-trip
  (leading/trailing whitespace such as ``"\\tno"``, or ``,{}``): the
  parser would strip or split it and the clause would change meaning.

Output
------
For an input ``<run>_DPG_split_grouped_conjunction_deduplicated_structure.json``
the script writes, in the same folder:

* ``<run>_DPG_split_grouped_conjunction_deduplicated_negative_structure.json``
* ``<run>_DPG_split_grouped_conjunction_deduplicated_negative.png``

Usage
-----
Single structure JSON file::

    python -m categorical.cat_NEGATIVE_peer \\
        outputs/categorical/nursery/GROUPED-SPLIT-CONJUNCTION-DEDUPLICATED\\ DPG/\\
        nursery_DPG_split_grouped_conjunction_deduplicated_structure.json

Whole directory (walks recursively, picks up every
``*_DPG_split_grouped_conjunction_deduplicated_structure.json``)::

    python -m categorical.cat_NEGATIVE_peer outputs/categorical/

Explicit output path (single file mode only), overwrite in place, or
JSON only without the PNG::

    python -m categorical.cat_NEGATIVE_peer <file> --out <other-file>
    python -m categorical.cat_NEGATIVE_peer <file-or-dir> --inplace
    python -m categorical.cat_NEGATIVE_peer <file-or-dir> --no-render
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


# ---------------------------------------------------------------------------
# Attribute domains
# ---------------------------------------------------------------------------

def _domains_from_onehot_map(onehot_map: dict) -> Dict[str, List[str]]:
    """``{base: [categories]}`` from the structure's ``onehot_map``, in
    one-hot column order."""
    domains: Dict[str, List[str]] = {}
    for base, category in onehot_map.values():
        domains.setdefault(str(base), []).append(str(category))
    return domains


class MissingOnehotMapError(ValueError):
    """The structure JSON has no ``onehot_map``, so attribute domains are unknown."""


# ---------------------------------------------------------------------------
# Label conversion
# ---------------------------------------------------------------------------

def _is_representable(category: str) -> bool:
    """``True`` when ``category`` survives a write/parse round trip of the
    ``base OP {a, b}`` grammar: the parser strips whitespace around each
    category and splits on ``,`` up to the first ``}``. Writing e.g.
    ``"\\tno"`` into a complement would silently turn it into ``"no"``."""
    return category == category.strip() and not any(ch in category for ch in ",{}")


def _negate_label(label: str, domains: Dict[str, List[str]], problems: List[str]) -> str:
    """Rewrite every ``IN`` clause of ``label`` as ``NOT IN`` over the
    complement, merging same-base ``NOT IN`` clauses. Pieces that are not
    IN/NOT IN clauses are kept verbatim and in place. Clauses that cannot
    be converted are kept as ``IN`` and described in ``problems``."""
    pieces = label.split(" AND ")
    # Each output slot is either a verbatim string or a base whose merged
    # NOT IN clause is rendered at the position of its first occurrence.
    slots: List[Tuple[str, str]] = []  # ("raw", text) | ("not_in", base)
    not_in_cats: Dict[str, List[str]] = {}

    for piece in pieces:
        parsed = _parse_in_label(piece)
        if parsed is None:
            slots.append(("raw", piece))
            continue
        base, op, cats = parsed
        if op == "IN":
            domain = domains.get(base, [])
            unknown = [c for c in cats if c not in domain]
            if not domain or unknown:
                problems.append(
                    f"{piece!r}: unknown domain or categories {unknown or cats} "
                    f"for {base!r}; kept as IN"
                )
                slots.append(("raw", piece))
                continue
            cats_set = set(cats)
            cats = [c for c in domain if c not in cats_set]
            if not cats:
                problems.append(f"{piece!r}: covers the whole domain of {base!r}; kept as IN")
                slots.append(("raw", piece))
                continue
            unrepresentable = [c for c in cats if not _is_representable(c)]
            if unrepresentable:
                problems.append(
                    f"{piece!r}: complement holds categories {unrepresentable} the "
                    f"label grammar cannot round-trip; kept as IN"
                )
                slots.append(("raw", piece))
                continue
        if base not in not_in_cats:
            not_in_cats[base] = []
            slots.append(("not_in", base))
        merged = not_in_cats[base]
        merged.extend(c for c in cats if c not in merged)

    return " AND ".join(
        value if kind == "raw" else _format_in_label(value, "NOT IN", not_in_cats[value])
        for kind, value in slots
    )


def _negate_structure(structure: dict) -> Tuple[dict, int, List[str]]:
    """Convert every node label of ``structure`` to its negative form.

    The structure is mutated in place. Returns ``(structure,
    n_rewritten, problems)``: the number of labels that changed and a
    list of clauses that could not be converted.

    Raises ``MissingOnehotMapError`` when the structure has no
    ``onehot_map``: the domains are never guessed from feature names.
    """
    onehot_map = structure.get("onehot_map")
    if not onehot_map:
        raise MissingOnehotMapError(
            "structure has no onehot_map; regenerate it with the current pipeline"
        )
    domains = _domains_from_onehot_map(onehot_map)
    problems: List[str] = []

    n_rewritten = 0
    for node in structure.get("nodes", []):
        if "->" in str(node.get("id", "")):
            continue  # edge-label pseudo-nodes carry no predicate
        label = str(node.get("label", ""))
        new_label = _negate_label(label, domains, problems)
        if new_label != label:
            node["label"] = new_label
            n_rewritten += 1
    return structure, n_rewritten, problems


# ---------------------------------------------------------------------------
# File / directory walking
# ---------------------------------------------------------------------------

_DEDUP_STRUCTURE_SUFFIX = "_DPG_split_grouped_conjunction_deduplicated_structure.json"


def _iter_structure_inputs(path: str) -> List[str]:
    """If ``path`` is a file, return ``[path]`` (no filtering). If it's a
    directory, return every deduplicated structure JSON under it
    (recursive), in sorted order for deterministic output."""
    if os.path.isfile(path):
        return [path]
    if os.path.isdir(path):
        out: List[str] = []
        for root, dirs, files in os.walk(path):
            dirs.sort()
            for f in sorted(files):
                if f.endswith(_DEDUP_STRUCTURE_SUFFIX):
                    out.append(os.path.join(root, f))
        return out
    return []


def _default_output_path(in_path: str) -> str:
    """``<stem>_structure.json`` -> ``<stem>_negative_structure.json``."""
    if in_path.endswith("_structure.json"):
        return in_path[: -len("_structure.json")] + "_negative_structure.json"
    return in_path + ".negative"


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

def _process_file(
    in_path: str,
    out_path: Optional[str],
    inplace: bool,
    render: bool,
    visualization_config: Optional[dict],
) -> Tuple[str, int, List[str], Optional[str]]:
    structure = _load_structure_json(in_path)
    new_structure, n_rewritten, problems = _negate_structure(structure)
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

    return target, n_rewritten, problems, png_target


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "input",
        help="A single structure JSON or a directory of them. Directory "
             "mode picks up every ``*_DPG_split_grouped_conjunction_"
             "deduplicated_structure.json`` under it (recursive).",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output path. Only valid when ``input`` is a single file. "
             "Defaults to ``<input>_negative_structure.json`` next to the input.",
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
        help="Skip PNG rendering; only emit the negative JSON.",
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
        mode = "negate + render" if render else "negate only"
        print(
            f"Processing {len(files)} structure file(s) from {args.input!r} "
            f"({mode})."
        )

    n_skipped = 0
    for f in files:
        try:
            out_path, n_rewritten, problems, png_path = _process_file(
                f, args.out, args.inplace, render, visualization_config,
            )
        except MissingOnehotMapError as exc:
            print(f"  [skip] {f}: {exc}", file=sys.stderr)
            n_skipped += 1
            continue
        suffix = (
            f"  ({n_rewritten} label(s) rewritten, PNG -> {png_path})"
            if png_path
            else f"  ({n_rewritten} label(s) rewritten)"
        )
        print(f"  [ok] {f} -> {out_path}{suffix}")
        for problem in problems:
            print(f"       [warn] {problem}")

    return 1 if n_skipped == len(files) else 0


if __name__ == "__main__":
    raise SystemExit(main())
