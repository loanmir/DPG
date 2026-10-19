#!/usr/bin/env python3
"""Aggregate DPG-structure metrics across the categorical grouping variants.

Pulls every structure JSON directly from **Weights & Biases** (no local
output-cache dependency) and computes the 5 user-requested metrics for
each of the 6 grouping variants the experiments produce (4 grouping
passes + the deduplication post-processing of the conjunction variant +
the NOT IN-only rewrite of the deduplicated variant).

For each dataset folder under ``DPG/datasets/`` (skipping ``dummy_dataset``):

1. Connect to W&B (project ``dpg-categorical``, entity
   ``mllab-ts-universit-di-trieste``) and pick the **latest finished run**
   whose ``display_name == dataset_name`` AND that has all six
   ``dpg_structure`` artifacts attached (or use ``--include-incomplete``
   to accept runs that predate the deduplication / negative passes and
   only have some of them). The artifacts are produced by
   ``run_all_categorical_experiments.py``'s ``wandb_log_artifacts``:

       BASIC_DPG__<dataset>_dpg_structure                (basic DPG)
       GROUPED_DPG__<dataset>_DPG_grouped_structure      (cat_grouping.py)
       GROUPED-SPLIT_DPG__<dataset>_DPG_split_grouped_structure
       GROUPED-SPLIT-CONJUNCTION_DPG__<dataset>_DPG_split_grouped_conjunction_structure
       GROUPED-SPLIT-CONJUNCTION-DEDUPLICATED_DPG__<dataset>_DPG_split_grouped_conjunction_deduplicated_structure
       GROUPED-SPLIT-CONJUNCTION-DEDUPLICATED-NEGATIVE_DPG__<dataset>_DPG_negative_structure

2. Download each ``dpg_structure`` artifact into a per-run cache folder,
   parse the JSON, and compute:
       1. Number of nodes                  (graph.node count)
       2. Number of edges                  (graph.link count)
       3. Number of unique nodes           (count of distinct labels)
       4. Validity of the graph            (structural: has class sinks,
                                            no orphan nodes)
       5. Sum of predicate characters      (sum of len(label) including
                                            ``Class X`` sinks)
2b. Compute the graph-vs-graph equivalence metrics from
   ``dpg_equivalence.py``, comparing every variant against ``BASIC DPG``:

       6. decision_agreement     -- same predicted class on the same rows
       7. explanation_agreement  -- same conditions tested along the way
       8. explanation_overlap    -- partial credit for partly-matching routes
       9. explanation_partial    -- rows where route enumeration hit its cap

   These walk both graphs with real data rows, so they need the dataset
   CSV on local disk (``DPG/datasets/<dataset>/``). They do NOT need the
   trained RandomForest -- the comparison is graph against graph, not
   graph against model, which is why this script can compute them
   despite never training anything. Datasets with no local CSV simply
   get empty equivalence columns. Use ``--no-equivalence`` to skip.

   Note ``decision_agreement`` is the weaker of the pair: a graph with a
   flipped predicate can change the reasoning on most rows while still
   scoring 1.000 on decisions. Read ``explanation_agreement`` alongside it.
3. Log a ``wandb.Table`` to the same run with one row per variant.
4. Print a console summary and write a markdown report to
   ``DPG/outputs/categorical/metrics_report.md`` (the latter always,
   even when wandb logging succeeds -- useful as a local copy).

Usage
-----
    python DPG/categorical/run_all_metrics.py
    python DPG/categorical/run_all_metrics.py --datasets iris titanic
    python DPG/categorical/run_all_metrics.py --no-wandb          # no upload
    python DPG/categorical/run_all_metrics.py --dry-run           # no wandb call at all
"""

from __future__ import annotations

import argparse
import json
import pathlib
import shutil
import sys
import tempfile
import time
from datetime import datetime
from typing import Dict, List, Optional, Sequence, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from networkx.readwrite import json_graph

# --- Repo / import setup --------------------------------------------------
SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent  # DPG/

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from categorical import dpg_equivalence as eq  # noqa: E402

# Per-dataset columns dropped before one-hot encoding (IDs / near-unique
# free text). Imported from the experiment runner so the evaluation rows
# are encoded the same way the graphs were built. That module pulls in
# matplotlib/sklearn/dpg, so a failure to import it must not take the
# metrics run down -- equivalence just falls back to no dropped columns.
try:
    from categorical.run_all_categorical_experiments import DROP_COLUMNS_OVERRIDES
except Exception:  # noqa: BLE001 - optional, see above
    DROP_COLUMNS_OVERRIDES = {}

# --- Constants ------------------------------------------------------------

WANDB_PROJECT = "dpg-categorical"
WANDB_ENTITY = "mllab-ts-universit-di-trieste"

DEFAULT_DATASETS_DIR = REPO_ROOT / "datasets"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs" / "categorical"

# Folder that holds toy/sweep data, not a real dataset.
# ``health_insurance`` is also excluded -- the DPG it produces has 231
# nodes / 610 edges on a 381k-row dataset with a non-trivial cycle, so
# the route walker takes ~15-20s per variant on it (vs ~1-3s for every
# other dataset) and frequently hits its 500-route cap, which makes
# the explanation metrics noisy. Re-enable it by removing the entry
# below; the rest of the pipeline still works on it.
EXCLUDED_DATASET_DIRS = {"dummy_dataset", "health_insurance"}

# The six variants the experiments produce, in display order. The first
# entry is the "ground truth" DPG the other five are derived from. The
# deduplicated variant comes after the conjunction variant it is built
# from, and the negative variant after the deduplicated one.
VARIANTS: Tuple[Tuple[str, str], ...] = (
    ("BASIC DPG", "BASIC_DPG"),
    ("GROUPED DPG", "GROUPED_DPG"),
    ("GROUPED-SPLIT DPG", "GROUPED-SPLIT_DPG"),
    ("GROUPED-SPLIT-CONJUNCTION DPG", "GROUPED-SPLIT-CONJUNCTION_DPG"),
    ("GROUPED-SPLIT-CONJUNCTION-DEDUPLICATED DPG", "GROUPED-SPLIT-CONJUNCTION-DEDUPLICATED_DPG"),
    ("GROUPED-SPLIT-CONJUNCTION-DEDUPLICATED-NEGATIVE DPG", "GROUPED-SPLIT-CONJUNCTION-DEDUPLICATED-NEGATIVE_DPG"),
)

# W&B artifact name fragments per variant. The full artifact name is
# ``<ARTIFACT_NAMESPACE>__<dataset><ARTIFACT_FILE_STEM>`` (the
# double-underscore separates the "namespace" -- the variant label -- from
# the file stem). See ``run_all_categorical_experiments.wandb_log_artifacts``.
# Source: inspecting the artifact names on a real run:
#   BASIC_DPG__bank_marketing_dpg_structure
#   GROUPED_DPG__bank_marketing_DPG_grouped_structure
#   ...
_ARTIFACT_FILE_STEM_BY_VARIANT: Dict[str, str] = {
    "BASIC_DPG": "_dpg_structure",
    "GROUPED_DPG": "_DPG_grouped_structure",
    "GROUPED-SPLIT_DPG": "_DPG_split_grouped_structure",
    "GROUPED-SPLIT-CONJUNCTION_DPG": "_DPG_split_grouped_conjunction_structure",
    "GROUPED-SPLIT-CONJUNCTION-DEDUPLICATED_DPG": "_DPG_split_grouped_conjunction_deduplicated_structure",
    # Shorter than the other stems so the full artifact name stays under
    # W&B's 128-character limit (see ``run_negative_pass``).
    "GROUPED-SPLIT-CONJUNCTION-DEDUPLICATED-NEGATIVE_DPG": "_DPG_negative_structure",
}

# Type tag all structure-JSON artifacts use -- lets us find them on a run
# even if their exact name differs slightly.
_STRUCTURE_ARTIFACT_TYPE = "dpg_structure"

# W&B artifact *type* tag the deduplication PNG uses, set by
# ``run_all_categorical_experiments.wandb_log_artifacts`` for any
# ``*.png`` file. Listed here so a future search-by-type can scope to
# image artifacts only without hard-coding the string elsewhere.
_IMAGE_ARTIFACT_TYPE = "dpg_image"


# ---------------------------------------------------------------------------
# WandB (optional)
# ---------------------------------------------------------------------------

# ``import wandb`` may pick up an empty local directory named ``wandb``
# (e.g. ``DPG/wandb/`` holding offline run caches) instead of the real
# package when this script is launched with ``DPG/`` on ``sys.path``. Probe
# for an actually usable module -- one with ``Api`` + ``Table`` -- before
# claiming wandb is available.
WANDB_AVAILABLE = False
wandb = None  # type: ignore[assignment]
try:
    import wandb as _wandb  # noqa: F401

    if hasattr(_wandb, "Api") and hasattr(_wandb, "Table"):
        wandb = _wandb
        WANDB_AVAILABLE = True
except ImportError:  # pragma: no cover
    pass


# ---------------------------------------------------------------------------
# Dataset discovery
# ---------------------------------------------------------------------------


def discover_dataset_names(datasets_dir: pathlib.Path, only: Optional[Sequence[str]]) -> List[str]:
    """Return dataset names (i.e. subdir names) in sorted order, skipping
    ``dummy_dataset`` / ``health_insurance`` and any subfolder without
    a CSV. If ``only`` is given
    the result is filtered to that subset (preserving sorted order)."""
    if not datasets_dir.exists():
        return []
    only_set = set(only) if only else None
    names: List[str] = []
    for entry in sorted(datasets_dir.iterdir()):
        if not entry.is_dir():
            continue
        if only_set is None and entry.name in EXCLUDED_DATASET_DIRS:
            continue
        if not list(entry.glob("*.csv")):
            continue
        if only_set is not None and entry.name not in only_set:
            continue
        names.append(entry.name)
    return names


# ---------------------------------------------------------------------------
# Loading structure JSONs (from W&B artifacts, not local disk)
# ---------------------------------------------------------------------------


def find_structure_artifact(run, dataset_name: str, variant_namespace: str) -> Optional[object]:
    """Find the ``dpg_structure`` artifact on ``run`` whose name matches
    ``<variant_namespace>__<dataset_name>...``. Returns the artifact object
    or ``None``. Falls back to scanning all ``dpg_structure`` artifacts on
    the run for a substring match (artifacts are addressed by their full
    ``name:version`` string)."""
    expected_stem = _ARTIFACT_FILE_STEM_BY_VARIANT[variant_namespace]
    expected_prefix = f"{variant_namespace}__{dataset_name}"
    candidates = []
    for art in run.logged_artifacts():
        if art.type != _STRUCTURE_ARTIFACT_TYPE:
            continue
        name_no_ver = art.name.split(":")[0]
        if not name_no_ver.startswith(expected_prefix):
            continue
        # Tie-break: prefer the one whose name ends with the right stem.
        if name_no_ver.endswith(expected_stem):
            return art
        candidates.append(art)
    return candidates[0] if candidates else None


def download_structure_artifact(artifact, dest_dir: pathlib.Path) -> pathlib.Path:
    """Download a W&B artifact's contents to ``dest_dir`` and return the
    path to its ``*.json`` structure file. Raises ``FileNotFoundError`` if
    no JSON ended up in the downloaded folder."""
    if dest_dir.exists():
        shutil.rmtree(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    artifact.download(root=str(dest_dir))
    jsons = sorted(dest_dir.rglob("*.json"))
    if not jsons:
        raise FileNotFoundError(
            f"No JSON inside downloaded artifact at {dest_dir}"
        )
    return jsons[0]


def load_structure(json_path: pathlib.Path) -> dict:
    """Load a structure JSON, returning the raw ``dict``."""
    with open(json_path, "r", encoding="utf-8") as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# Metric extraction
# ---------------------------------------------------------------------------


def _real_nodes(structure: dict) -> List[Dict[str, str]]:
    """Top-level ``nodes`` list filtered to real (non-pseudo) nodes -- i.e.
    those with a non-empty ``label`` and no ``->`` in their id. The BASIC
    DPG dumps 2x as many entries because it also stores an empty-label
    pseudo-node per edge for visualisation; we always drop those."""
    return [
        n
        for n in structure.get("nodes", [])
        if n.get("label") and "->" not in str(n["id"])
    ]


def _nx_graph(structure: dict) -> nx.DiGraph:
    """Rebuild a ``networkx.DiGraph`` from the ``graph`` field, the same
    way the grouping scripts do.

    The DPG JSONs always use ``links`` (BASIC) or both ``links`` and
    ``edges`` (grouped variants). NetworkX changed the kwarg name three
    times across recent releases (``edges`` in <3.4, ``link`` in 3.4-3.5,
    default flipped in 3.6+). We pin it explicitly here to be portable.
    """
    graph = structure.get("graph", {})
    # Determine which kwarg name this NetworkX version expects.
    import inspect as _inspect

    params = _inspect.signature(json_graph.node_link_graph).parameters
    if "link" in params:
        return json_graph.node_link_graph(graph, link="links")
    if "edges" in params:
        return json_graph.node_link_graph(graph, edges="links")
    # Pre-2.x NetworkX: positional only.
    return json_graph.node_link_graph(graph)


def metric_node_count(structure: dict) -> int:
    """1) Number of nodes (graph-level, after grouping)."""
    return len(structure.get("graph", {}).get("nodes", []))


def metric_edge_count(structure: dict) -> int:
    """2) Number of edges (graph-level). The grouping variants store both
    ``links`` and ``edges``; BASIC only has ``links``. Always pick ``links``
    -- they're the same content where both exist."""
    g = structure.get("graph", {})
    return len(g.get("links", []))


def metric_unique_node_count(structure: dict) -> int:
    """3) Number of unique nodes -- distinct predicate/Class labels in
    the top-level ``nodes`` array. Two structural nodes with the same label
    (the same predicate reached via different parents) count as one."""
    return len({n["label"] for n in _real_nodes(structure)})


def metric_predicate_char_sum(structure: dict) -> int:
    """5) Sum of ``len(label)`` for every labelled node. Includes ``Class X``
    labels -- those are terminal sinks and count towards "characters in
    predicates for all the predicates in graph" (the original Notes
    phrasing)."""
    return sum(len(n["label"]) for n in _real_nodes(structure))


def metric_validity(structure: dict) -> bool:
    """4) Structural validity of the (possibly grouped) DPG.

    A DPG is structurally well-formed when it satisfies ALL of:

        * Reconstructs as a graph with at least one node.
        * Has at least one ``Class X`` sink (no class nodes == the rewrite
          dropped a terminal and the graph no longer represents a
          classification flow).
        * No orphan nodes: every non-root, non-sink node has in-degree >= 1
          AND out-degree >= 1. Roots (no incoming edges) are the entry
          points of the random forest's paths and are exempt; Class sinks
          (no outgoing edges) are exempt.

    Note on DAG-ness: a real DPG is NOT guaranteed to be a DAG -- a
    predicate node that's shared between trees with differing split order
    can close a cycle (see ``compute_graph_depth`` in
    ``examples/quickstart_categorical_WP5.py``). We deliberately do NOT
    require DAG-ness here; otherwise a perfectly valid credit_card_approval
    or titanic DPG would always report ``False``. The check exists to
    catch the rewrite/grouping pass corrupting the graph (lost nodes,
    dangling edges), not the inherent sharing structure of DPGs.
    """
    g = _nx_graph(structure)
    if g.number_of_nodes() == 0:
        return False

    real = _real_nodes(structure)
    labels = {n["id"]: n["label"] for n in real}

    # Class sinks: any node whose label starts with "Class".
    has_class_sink = any(str(lbl).startswith("Class ") for lbl in labels.values())
    if not has_class_sink:
        return False

    # No orphan nodes: every non-root (in-degree 0) and non-sink
    # (out-degree 0) node must still be wired into the graph on both
    # sides. Roots (the entry points of the random forest paths) and
    # Class sinks (terminal nodes) are exempt.
    for nid in g.nodes():
        nid_str = str(nid)
        if nid_str not in labels:
            # Pseudo-node from the BASIC dump; ignore.
            continue
        if str(labels[nid_str]).startswith("Class "):
            continue
        in_deg = g.in_degree(nid)
        out_deg = g.out_degree(nid)
        # Reject: a node with no incoming AND no outgoing (orphan), or a
        # non-root node with no outgoing (dead end that isn't a Class sink).
        if in_deg == 0 and out_deg == 0:
            return False
        if in_deg >= 1 and out_deg == 0:
            return False
    return True


METRIC_FUNCTIONS = (
    ("nodes", metric_node_count),
    ("edges", metric_edge_count),
    ("unique_nodes", metric_unique_node_count),
    ("valid", metric_validity),
    ("predicate_chars", metric_predicate_char_sum),
)


def compute_metrics(structure: dict) -> Dict[str, object]:
    """Apply all five metric extractors to a structure dict."""
    return {key: fn(structure) for key, fn in METRIC_FUNCTIONS}


# ---------------------------------------------------------------------------
# Per-dataset metrics (all from wandb)
# ---------------------------------------------------------------------------


def compute_dataset_metrics_from_run(
    run,
    dataset_name: str,
    cache_root: pathlib.Path,
) -> Dict[str, Dict[str, object]]:
    """Compute every (variant -> metrics) pair for one dataset by pulling
    each ``dpg_structure`` artifact off ``run``. Missing variants are
    reported with ``_missing=True`` so they still show up as a (missing)
    row in the output table.

    Each variant's artifact is downloaded into ``cache_root/<variant>/``
    so a re-run (or a follow-up metric extraction on the same run) can
    skip the network round-trip.
    """
    out: Dict[str, Dict[str, object]] = {}
    for variant_label, variant_namespace in VARIANTS:
        art = find_structure_artifact(run, dataset_name, variant_namespace)
        if art is None:
            out[variant_label] = {"_missing": True}
            continue
        variant_cache = cache_root / variant_namespace
        try:
            json_path = download_structure_artifact(art, variant_cache)
            structure = load_structure(json_path)
            metrics = compute_metrics(structure)
            metrics["_json_path"] = str(json_path)
            metrics["_artifact_name"] = art.name
            # Kept so the equivalence pass can rebuild this variant's
            # graph without downloading it a second time. Stripped before
            # the metrics reach the table/report.
            metrics["_structure"] = structure
            out[variant_label] = metrics
        except Exception as exc:  # noqa: BLE001
            out[variant_label] = {
                "_missing": True,
                "_error": f"{type(exc).__name__}: {exc}",
            }
    return out


# ---------------------------------------------------------------------------
# Equivalence metrics (graph vs graph -- no model needed)
# ---------------------------------------------------------------------------

# The variant every other variant is compared against.
BASELINE_VARIANT = "BASIC DPG"

EQUIVALENCE_KEYS = (
    "decision_agreement",
    "explanation_agreement",
    "explanation_overlap",
    "explanation_partial",
)

# Columns produced by ``add_faithfulness_metrics`` and shipped in the
# wandb table. ``faithfulness_score`` is the weighted composite from
# ``DPGExplainer.evaluate_faithfulness``; the rest are the three
# components we surface separately so a regression in one weight can be
# diagnosed without re-running the metric.
FAITHFULNESS_KEYS = (
    "faithfulness_score",
    "output_fidelity",
    "mean_trace_coverage_score",
    "mean_recombination_rate",
)


def find_dataset_csv(datasets_dir: pathlib.Path, dataset_name: str) -> Optional[pathlib.Path]:
    """Locate a dataset's CSV on local disk, or None if it isn't there.

    The structure JSONs come from W&B, but the equivalence metrics need
    real rows to walk the graphs with, and those only exist locally.
    """
    folder = datasets_dir / dataset_name
    if not folder.is_dir():
        return None
    csvs = sorted(folder.glob("*.csv"))
    return csvs[0] if csvs else None


def add_equivalence_metrics(
    dataset_name: str,
    variant_metrics: Dict[str, Dict[str, object]],
    datasets_dir: pathlib.Path,
    n_samples: int,
) -> None:
    """Fill in the equivalence columns for every variant, in place.

    Each variant is compared against ``BASELINE_VARIANT`` by walking both
    graphs over the same rows. This needs no RandomForest: the comparison
    is graph-against-graph, which is exactly the question "did grouping
    change anything". (Fidelity-against-the-model is deliberately not
    computed here -- it would require the trained model, which this
    script never has.)

    Missing dataset CSV, missing baseline, or an unparseable graph leaves
    the columns as None rather than failing the dataset.
    """
    baseline = variant_metrics.get(BASELINE_VARIANT)
    if not baseline or baseline.get("_missing") or "_structure" not in baseline:
        print(f"  [equiv] no {BASELINE_VARIANT} structure; skipping equivalence")
        return

    csv_path = find_dataset_csv(datasets_dir, dataset_name)
    if csv_path is None:
        print(f"  [equiv] no local CSV under {datasets_dir / dataset_name}; skipping equivalence")
        return

    try:
        eval_rows = eq.load_eval_rows(
            str(csv_path),
            drop_columns=DROP_COLUMNS_OVERRIDES.get(dataset_name),
            n_samples=n_samples,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"  [equiv] could not load rows from {csv_path.name}: {exc}")
        return

    # Domain-aware canonicalisation: lets the metric recognise that
    # variants emitting ``NOT IN {a, b}`` and ``NOT IN {a} AND IN {b}``
    # are equivalent (see ``canonical_constraints``). Failures here are
    # non-fatal -- the metric falls back to no-domain behaviour.
    domains: Optional[Dict[str, frozenset]] = None
    try:
        domains = eq.load_categorical_domains(
            str(csv_path),
            drop_columns=DROP_COLUMNS_OVERRIDES.get(dataset_name),
        )
        print(f"  [equiv] domain-aware canonicalisation on for {len(domains)} categorical column(s)")
    except Exception as exc:  # noqa: BLE001
        print(f"  [equiv] could not load categorical domains: {exc}; falling back to no-domain comparison")
        domains = None

    # The real dummy -> (column, category) map, so one-hot columns are not
    # guessed from their names. Non-fatal like the domains above.
    onehot_map: Optional[Dict[str, Tuple[str, str]]] = None
    try:
        onehot_map = eq.load_onehot_map(
            str(csv_path),
            drop_columns=DROP_COLUMNS_OVERRIDES.get(dataset_name),
        )
    except Exception as exc:  # noqa: BLE001
        print(f"  [equiv] could not build one-hot map: {exc}; one-hot columns are guessed from their names")
        onehot_map = None

    try:
        base_graph = eq.structure_to_graph(baseline["_structure"])
        base_nodes = eq.real_nodes(baseline["_structure"])
    except Exception as exc:  # noqa: BLE001
        print(f"  [equiv] could not rebuild {BASELINE_VARIANT} graph: {exc}")
        return

    print(f"  [equiv] comparing against {BASELINE_VARIANT} on {len(eval_rows)} rows")
    for variant_label, metrics in variant_metrics.items():
        if metrics.get("_missing") or "_structure" not in metrics:
            continue
        try:
            graph = eq.structure_to_graph(metrics["_structure"])
            nodes = eq.real_nodes(metrics["_structure"])
            result = eq.compare_variants(
                base_graph, base_nodes, graph, nodes, eval_rows,
                domains=domains, onehot_map=onehot_map,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"  [equiv] {variant_label}: failed ({type(exc).__name__}: {exc})")
            continue
        for key in EQUIVALENCE_KEYS:
            metrics[key] = result.get(key)
        print(
            f"  [equiv] {variant_label}: "
            f"decision={result.get('decision_agreement')}, "
            f"explanation={result.get('explanation_agreement')}"
        )


def strip_internal_keys(variant_metrics: Dict[str, Dict[str, object]]) -> None:
    """Drop the cached structure dicts once equivalence is done -- they
    are large and must not end up in the report or the wandb table."""
    for metrics in variant_metrics.values():
        metrics.pop("_structure", None)


# ---------------------------------------------------------------------------
# Faithfulness (graph walk vs black-box model)
# ---------------------------------------------------------------------------
#
# ``evaluate_faithfulness`` on ``DPGExplainer`` needs:
#   * the fitted model (RandomForestClassifier) -- not stored in the
#     structure artifact, so we retrain it here with the exact same
#     hyperparameters / seed the experiment runner uses, and
#   * a sample of evaluation rows -- we reuse ``load_eval_rows`` so the
#     faithfulness rows are encoded the same way as the equivalence rows.
#
# The retrained model is byte-identical across runs because both this
# script and ``run_all_categorical_experiments.py`` use
# ``RANDOM_STATE = 27`` / ``NUM_TREES = 100`` and the same
# ``train_test_split`` order via ``train_cv`` (sklearn ``KFold`` with
# the same seed). The first KFold split's training fold is what the
# original ``DPGExplainer.fit()`` saw; we use the *same* split to keep
# the trained model on a distribution that produced the stored graph.
#
# Cost: each variant retrains a RandomForest and walks N rows through it
# (``N = --faith-samples``, default 50). Use ``--no-faithfulness`` to
# skip this entirely if you're only after the structural metrics.


def _retrain_baseline_model(
    csv_path: pathlib.Path,
    feature_names: List[str],
    target_names: List[str],
    dataset_name: str,
):
    """Re-fit a RandomForestClassifier on the same train fold the
    original experiment used.

    Returns ``(model, X_train_df, feature_names_used, target_names_used)``
    or ``None`` if the dataset CSV is missing or the columns don't line
    up with the structure JSON. Failures are non-fatal: the caller
    surfaces them as ``None`` faithfulness columns in the table.
    """
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split
        from categorical.run_all_categorical_experiments import (
            DROP_COLUMNS_OVERRIDES,
            NUM_TREES,
            RANDOM_STATE,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"  [faith] could not import training helpers: {exc}")
        return None

    try:
        df = pd.read_csv(csv_path, sep=None, engine="python", encoding="utf-8-sig")
    except Exception as exc:  # noqa: BLE001
        print(f"  [faith] could not read {csv_path.name}: {exc}")
        return None

    drop_cols = [c for c in DROP_COLUMNS_OVERRIDES.get(dataset_name, []) if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    target_col = df.columns[-1]
    df = df.dropna(subset=[target_col]).reset_index(drop=True)

    # Same preprocessing order as ``run_all_categorical_experiments.load_dataset``.
    features_raw = df.iloc[:, :-1]
    y = df[target_col].values
    X_enc = pd.get_dummies(features_raw, drop_first=False)
    X_enc = X_enc.replace([np.inf, -np.inf], np.nan).fillna(X_enc.mean())
    X_enc = np.round(X_enc, 3)

    # Match the experiment's train/test split order. The runner uses
    # ``KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)`` for
    # CV; the first split's training fold is what ``DPGExplainer.fit()``
    # saw. ``train_test_split`` with the same seed reproduces the same
    # 80/20 partition when ``test_size=0.2``.
    X_train, _X_test, y_train, _y_test = train_test_split(
        X_enc, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    X_train = X_train.reset_index(drop=True)

    model = RandomForestClassifier(n_estimators=NUM_TREES, random_state=RANDOM_STATE)
    model.fit(X_train, y_train)
    return model, X_train, list(X_enc.columns), [str(t) for t in target_names]


def _faithfulness_for_variant(
    variant_label: str,
    metrics: Dict[str, object],
    csv_path: pathlib.Path,
    dataset_name: str,
    n_samples: int,
) -> None:
    """Fill in the faithfulness columns for one variant, in place.

    Reads the cached structure dict (``metrics["_structure"]``), rebuilds
    a NetworkX graph from it, re-fits a RandomForest on the same training
    fold, wraps it in a ``DPGExplainer``, then runs
    ``evaluate_faithfulness`` on a sample of the dataset CSV.
    """
    structure = metrics.get("_structure")
    if structure is None or metrics.get("_missing"):
        return

    feature_names = structure.get("feature_names")
    target_names = structure.get("target_names")
    if not feature_names or not target_names:
        print(
            f"  [faith] {variant_label}: structure JSON lacks "
            f"feature_names/target_names; skipping (older run?)"
        )
        return

    # Re-fit the RF on the same training fold.
    retrained = _retrain_baseline_model(
        csv_path, feature_names, target_names, dataset_name
    )
    if retrained is None:
        return
    model, _X_train_df, used_feature_names, used_target_names = retrained

    # The encoding pipeline may have produced more columns than
    # ``feature_names`` (because of one-hot expansion). Build the
    # explainer with the encoded column names so it matches what the
    # model was actually fitted on.
    try:
        from dpg import DPGExplainer
    except Exception as exc:  # noqa: BLE001
        print(f"  [faith] could not import DPGExplainer: {exc}")
        return

    try:
        explainer = DPGExplainer(
            model=model,
            feature_names=used_feature_names,
            target_names=used_target_names,
        )
        # ``evaluate_faithfulness`` needs the original (un-normalized)
        # model so ``model.predict`` goes through sklearn's real path.
        # DecisionPredicateGraph shallow-copies the model internally; for
        # ``GradientBoostingClassifier`` that copy can break ``predict``,
        # so we keep a reference to the model we actually trained.
        explainer._original_model = model
        # The structure JSON is the single source of truth for the graph;
        # inject it directly so we don't pay the (slow) extraction cost
        # twice. ``eq.structure_to_graph`` returns a NetworkX DiGraph
        # keyed under the same SHA1 hashing scheme used at training
        # time, so the trace recorder in ``explain_local`` will match
        # graph nodes by id.
        explainer._is_fitted = True
        explainer._graph = eq.structure_to_graph(structure)
        explainer._nodes = eq.real_nodes(structure)
        explainer._dot = None  # not needed for faithfulness
    except Exception as exc:  # noqa: BLE001
        print(f"  [faith] {variant_label}: could not build explainer "
              f"({type(exc).__name__}: {exc})")
        return

    # Pull a reproducible sample of evaluation rows, same loader the
    # equivalence step uses. ``y`` (real labels) is needed for the
    # ``local_accuracy`` sub-metric.
    try:
        from categorical import dpg_equivalence as _eq
        rows = _eq.load_eval_rows(
            str(csv_path),
            drop_columns=DROP_COLUMNS_OVERRIDES.get(dataset_name),
            n_samples=n_samples,
        )
        # ``load_eval_rows`` returns Series whose index contains BOTH
        # the raw column names AND the encoded column names (the loader
        # deliberately keeps both to handle "is this a numeric threshold?"
        # look-ups). The DPG was trained on encoded columns only, so we
        # must keep just the encoded subset.
        keep = [c for c in used_feature_names if c in rows[0].index]
        X_eval = pd.DataFrame(
            [{c: row[c] for c in keep} for row in rows],
            columns=keep,
        )
        # Recover labels for ``y_true``. Re-run the same preprocessing
        # the loader uses so the row indices line up (the loader
        # ``reset_index(drop=True)``s the dataframe, so positionally
        # ``rows[i]`` corresponds to the i-th row of the processed df).
        raw_df = pd.read_csv(csv_path, sep=None, engine="python",
                             encoding="utf-8-sig")
        drop_cols = [c for c in DROP_COLUMNS_OVERRIDES.get(dataset_name, [])
                     if c in raw_df.columns]
        if drop_cols:
            raw_df = raw_df.drop(columns=drop_cols)
        target_col = raw_df.columns[-1]
        raw_df = raw_df.dropna(subset=[target_col]).reset_index(drop=True)
        # The loader samples without replacement using random_state=27
        # on the post-dropna reset-index DataFrame. Reproduce that.
        sampled_pos = (
            raw_df.iloc[:, :-1]
            .sample(n=min(n_samples, len(raw_df)), random_state=27)
            .index
            .tolist()
        )
        y_eval = raw_df.loc[sampled_pos, target_col].tolist()
    except Exception as exc:  # noqa: BLE001
        print(f"  [faith] {variant_label}: could not load eval rows "
              f"({type(exc).__name__}: {exc})")
        return

    try:
        details = explainer.evaluate_faithfulness(
            X_eval,
            y_true=y_eval,
            max_samples=n_samples,
            return_details=True,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"  [faith] {variant_label}: evaluate_faithfulness failed "
              f"({type(exc).__name__}: {exc})")
        return

    metrics["faithfulness_score"] = details.get("faithfulness_score")
    metrics["output_fidelity"] = details.get("output_fidelity")
    metrics["mean_trace_coverage_score"] = details.get("mean_trace_coverage_score")
    metrics["mean_recombination_rate"] = details.get("mean_recombination_rate")
    print(
        f"  [faith] {variant_label}: faithfulness={details.get('faithfulness_score')}, "
        f"output_fidelity={details.get('output_fidelity')}"
    )


def add_faithfulness_metrics(
    dataset_name: str,
    variant_metrics: Dict[str, Dict[str, object]],
    datasets_dir: pathlib.Path,
    n_samples: int,
) -> None:
    """Fill in the faithfulness columns for every variant, in place.

    Re-fits a baseline RandomForest on each variant and walks the same
    CSV sample the equivalence pass used. Errors per-variant are
    non-fatal: the columns stay ``None`` so a single bad variant cannot
    take the whole dataset's row down.
    """
    csv_path = find_dataset_csv(datasets_dir, dataset_name)
    if csv_path is None:
        print(f"  [faith] no local CSV under {datasets_dir / dataset_name}; skipping")
        return
    print(f"  [faith] computing faithfulness on {len(variant_metrics)} variant(s) "
          f"({n_samples} samples each)")
    for variant_label, metrics in variant_metrics.items():
        _faithfulness_for_variant(
            variant_label, metrics, csv_path, dataset_name, n_samples
        )


# ---------------------------------------------------------------------------
# WandB: latest finished run per dataset
# ---------------------------------------------------------------------------


def run_has_all_structure_artifacts(run, dataset_name: str) -> Tuple[bool, List[str]]:
    """Return ``(ok, missing_namespaces)``: ``True`` if ``run`` has a
    ``dpg_structure`` artifact for every variant in ``VARIANTS`` matching
    ``dataset_name``. Used to skip runs that started but never finished
    all four grouping passes (artifact upload is the last step).
    """
    present_namespaces: set = set()
    for art in run.logged_artifacts():
        if art.type != _STRUCTURE_ARTIFACT_TYPE:
            continue
        name_no_ver = art.name.split(":")[0]
        for _label, namespace in VARIANTS:
            if name_no_ver.startswith(f"{namespace}__{dataset_name}"):
                present_namespaces.add(namespace)
                break
    missing = [ns for _label, ns in VARIANTS if ns not in present_namespaces]
    return (len(missing) == 0, missing)


def fetch_run_by_id(
    run_id: str,
    entity: str,
    project: str,
) -> Optional[object]:
    """Fetch a specific wandb ``Run`` by its id. Returns ``None`` if the
    run is missing, in a non-finished state, or wandb is unavailable.

    Used by ``--run-id`` to target a specific experiment (e.g. when
    re-running metrics after a bugfix without sweeping every dataset).
    """
    if not WANDB_AVAILABLE:
        return None
    api = wandb.Api()
    try:
        run = api.run(f"{entity}/{project}/{run_id}")
    except Exception as exc:  # noqa: BLE001
        print(f"  [wandb] could not fetch run {run_id}: {exc}")
        return None
    if run.state != "finished":
        print(
            f"  [wandb] run {run_id} is in state '{run.state}' (expected 'finished'); "
            f"proceeding anyway -- the structure artifacts may still be readable."
        )
    return run


def fetch_latest_finished_run(
    dataset_name: str,
    entity: str,
    project: str,
    offline: bool,
    require_all_artifacts: bool = True,
    per_page: int = 20,
):
    """Return the most recent wandb ``Run`` whose ``display_name ==
    dataset_name`` and ``state == 'finished'``. If
    ``require_all_artifacts`` is True, the run is only accepted if it has
    all 4 ``dpg_structure`` artifacts attached; otherwise we skip past
    incomplete runs and keep looking at older ones (up to ``per_page``).

    Returns ``None`` if no qualifying run is found.
    """
    if not WANDB_AVAILABLE:
        return None
    api = wandb.Api()
    filters = {
        "display_name": dataset_name,
        "state": "finished",
    }
    try:
        runs = api.runs(
            f"{entity}/{project}",
            filters=filters,
            order="-created_at",
            per_page=per_page,
        )
        for run in runs:
            if not (run.display_name == dataset_name or run.name == dataset_name):
                continue
            if not require_all_artifacts:
                return run
            ok, missing = run_has_all_structure_artifacts(run, dataset_name)
            if ok:
                return run
            print(
                f"  [wandb] skipping {run.id} ({run.display_name}, "
                f"{run.created_at}): missing structure artifacts {missing}"
            )
        return None
    except Exception as exc:  # noqa: BLE001
        print(f"  [wandb] failed to fetch runs for {dataset_name}: {exc}")
        return None


# ---------------------------------------------------------------------------
# WandB table logging
# ---------------------------------------------------------------------------


def _build_summary_rows(
    dataset_name: str,
    variant_metrics: Dict[str, Dict[str, object]],
    run_url: str,
) -> List[List[object]]:
    """Build the wandb.Table rows for a single dataset. One row per variant.

    Columns mirror the metrics the user asked for so the table renders
    self-explanatory on the wandb dashboard.
    """
    rows: List[List[object]] = []
    for variant_label, _short in VARIANTS:
        metrics = variant_metrics.get(variant_label, {})
        rows.append(
            [
                dataset_name,
                variant_label,
                metrics.get("nodes"),
                metrics.get("edges"),
                metrics.get("unique_nodes"),
                metrics.get("valid"),
                metrics.get("predicate_chars"),
                metrics.get("decision_agreement"),
                metrics.get("explanation_agreement"),
                metrics.get("explanation_overlap"),
                metrics.get("explanation_partial"),
                # Faithfulness (graph walk vs black-box model). Composite
                # first, then the three components we surface so a drop
                # in one weight is diagnosable without re-running.
                metrics.get("faithfulness_score"),
                metrics.get("output_fidelity"),
                metrics.get("mean_trace_coverage_score"),
                metrics.get("mean_recombination_rate"),
                run_url,
            ]
        )
    return rows


SUMMARY_COLUMNS = [
    "dataset",
    "variant",
    "n_nodes",
    "n_edges",
    "n_unique_nodes",
    "valid",
    "predicate_char_sum",
    # Graph-vs-graph equivalence, each variant against BASIC DPG.
    # decision_*  -- same predicted class (weaker: a broken graph can score 1.0)
    # explanation_* -- same conditions tested along the way (stronger)
    "decision_agreement",
    "explanation_agreement",
    "explanation_overlap",
    "explanation_partial",
    # Faithfulness (graph walk vs black-box model).
    # faithfulness_score: weighted composite of the four signals.
    # output_fidelity: share of rows where DPG walk agrees with the model.
    # mean_trace_coverage_score: how much of the graph the explanation touched.
    # mean_recombination_rate: how often the explanation reused the same nodes.
    "faithfulness_score",
    "output_fidelity",
    "mean_trace_coverage_score",
    "mean_recombination_rate",
    "wandb_run_url",
]


def log_table_to_run(
    run_id: str,
    dataset_name: str,
    variant_metrics: Dict[str, Dict[str, object]],
    run_url: str,
    entity: str,
    project: str,
) -> None:
    """Log a ``wandb.Table`` to the run with the given ``run_id`` by
    reattaching to it as a writer via ``wandb.init(resume=...)``.

    The ``Run`` object returned by ``api.runs()`` is read-only -- it has
    no ``.log()`` method. To append new metrics/tables to an existing
    finished run you must re-init wandb with ``resume="allow"`` and the
    run's id; this re-opens the run as a writer, ``log()`` adds the table,
    and ``finish()`` flushes it back to the server.
    """
    if not WANDB_AVAILABLE:
        return
    # Reattach to the existing run as a writer. ``reinit="finish_previous"``
    # lets us attach to multiple runs in one process (one per dataset) by
    # finishing the previous init first. ``resume="allow"`` lets us target
    # a finished run by id (newer wandb versions deprecate the bool form).
    attached = wandb.init(
        entity=entity,
        project=project,
        id=run_id,
        resume="allow",
        reinit="finish_previous",
    )
    try:
        table = wandb.Table(columns=SUMMARY_COLUMNS)
        for row in _build_summary_rows(dataset_name, variant_metrics, run_url):
            table.add_data(*row)
        attached.log({f"metrics/{dataset_name}_grouping_comparison": table})
    finally:
        attached.finish()


# ---------------------------------------------------------------------------
# Local fallback: markdown report
# ---------------------------------------------------------------------------


def _fmt_ratio(value: object) -> str:
    """Format an equivalence ratio for display. ``None`` (equivalence not
    computed for this dataset) and NaN both render as ``-``."""
    if value is None:
        return "-"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if number != number:  # NaN
        return "-"
    return f"{number:.3f}"


def render_markdown_report(per_dataset: Dict[str, Dict[str, object]]) -> str:
    """Render the collected metrics as a markdown table per dataset."""
    lines: List[str] = []
    lines.append("# Categorical DPG Metrics Report")
    lines.append("")
    lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")

    for dataset_name in sorted(per_dataset.keys()):
        payload = per_dataset[dataset_name]
        variant_metrics: Dict[str, Dict[str, object]] = payload["metrics"]
        run_url: str = payload.get("run_url", "")

        lines.append(f"## `{dataset_name}`")
        if run_url:
            lines.append(f"W&B run: <{run_url}>")
        lines.append("")
        lines.append(
            "| Variant | Nodes | Edges | Unique Nodes | Valid | Predicate Chars | "
            "Decision Agr. | Explanation Agr. | Explanation Overlap |"
            " Faithfulness | Output Fidelity | Trace Coverage | Recombination |"
        )
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for variant_label, _short in VARIANTS:
            m = variant_metrics.get(variant_label, {})
            if m.get("_missing"):
                lines.append(
                    f"| {variant_label} | (missing) | - | - | - | - | - | - | - | - | - | - | - |"
                )
                continue
            valid = m.get("valid")
            valid_str = "TRUE" if valid is True else "FALSE" if valid is False else str(valid)
            lines.append(
                f"| {variant_label} | {m.get('nodes')} | {m.get('edges')} | "
                f"{m.get('unique_nodes')} | {valid_str} | {m.get('predicate_chars')} | "
                f"{_fmt_ratio(m.get('decision_agreement'))} | "
                f"{_fmt_ratio(m.get('explanation_agreement'))} | "
                f"{_fmt_ratio(m.get('explanation_overlap'))} | "
                f"{_fmt_ratio(m.get('faithfulness_score'))} | "
                f"{_fmt_ratio(m.get('output_fidelity'))} | "
                f"{_fmt_ratio(m.get('mean_trace_coverage_score'))} | "
                f"{_fmt_ratio(m.get('mean_recombination_rate'))} |"
            )
        lines.append("")

    lines.append("## Column notes")
    lines.append("")
    lines.append("- **Valid** -- structural only: has a class sink, no orphan nodes.")
    lines.append("- **Decision Agr.** -- same predicted class as `BASIC DPG` on the same rows.")
    lines.append("  Weaker than it looks: a graph with a flipped predicate can still score 1.000.")
    lines.append("- **Explanation Agr.** -- same conditions tested along the way as `BASIC DPG`,")
    lines.append("  normalised so a collapsed chain counts as equal to the nodes it replaced.")
    lines.append("  This is the metric that actually shows grouping preserved the reasoning.")
    lines.append("- **Explanation Overlap** -- partial credit when routes only partly match.")
    lines.append("- **Faithfulness** -- weighted composite "
                 "(0.35 output_fidelity + 0.30 trace_coverage + 0.20 anti_recombination + 0.15 evidence_margin).")
    lines.append("- **Output Fidelity** -- fraction of rows where the DPG walk agrees with the model.")
    lines.append("- **Trace Coverage** -- how much of the graph the local explanation actually touched.")
    lines.append("- **Recombination** -- how often the explanation reused the same nodes "
                 "(lower is better; appears as 1 - recombination in the composite).")
    lines.append("")

    return "\n".join(lines)


def render_console_table(per_dataset: Dict[str, Dict[str, object]]) -> str:
    """Plain-text table suitable for stdout."""
    headers = ("Variant", "Nodes", "Edges", "Unique", "Valid", "PredChars",
               "DecisAgr", "ExplAgr", "ExplOvlp",
               "Faith", "OutFid", "TraceCov", "Recomb")
    widths = (28, 7, 7, 7, 7, 10, 9, 9, 9, 7, 7, 9, 8)
    out_lines: List[str] = []
    for dataset_name in sorted(per_dataset.keys()):
        variant_metrics = per_dataset[dataset_name]["metrics"]
        out_lines.append(f"\n[{dataset_name}]")
        out_lines.append("  " + " | ".join(f"{h:>{w}}" for h, w in zip(headers, widths)))
        out_lines.append("  " + "-+-".join("-" * w for w in widths))
        for variant_label, _short in VARIANTS:
            m = variant_metrics.get(variant_label, {})
            if m.get("_missing"):
                row = (f"{variant_label} (missing)",
                       "-", "-", "-", "-", "-", "-", "-", "-",
                       "-", "-", "-", "-")
            else:
                valid = m.get("valid")
                valid_str = "TRUE" if valid is True else "FALSE" if valid is False else str(valid)
                row = (
                    variant_label,
                    str(m.get("nodes")),
                    str(m.get("edges")),
                    str(m.get("unique_nodes")),
                    valid_str,
                    str(m.get("predicate_chars")),
                    _fmt_ratio(m.get("decision_agreement")),
                    _fmt_ratio(m.get("explanation_agreement")),
                    _fmt_ratio(m.get("explanation_overlap")),
                    _fmt_ratio(m.get("faithfulness_score")),
                    _fmt_ratio(m.get("output_fidelity")),
                    _fmt_ratio(m.get("mean_trace_coverage_score")),
                    _fmt_ratio(m.get("mean_recombination_rate")),
                )
            out_lines.append("  " + " | ".join(f"{c:>{w}}" for c, w in zip(row, widths)))
    return "\n".join(out_lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--datasets-dir",
        type=pathlib.Path,
        default=DEFAULT_DATASETS_DIR,
        help="Root containing one subdir per dataset (default: DPG/datasets). Used only to discover which dataset names exist.",
    )
    parser.add_argument(
        "--output-root",
        type=pathlib.Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Where to write metrics_report.md (default: DPG/outputs/categorical).",
    )
    parser.add_argument(
        "--artifact-cache",
        type=pathlib.Path,
        default=None,
        help="Where to download W&B structure-artifact JSONs (default: a temp dir, auto-deleted on exit).",
    )
    parser.add_argument(
        "--datasets", nargs="+", default=None,
        help="Restrict to a subset of dataset names (default: all, except dummy_dataset and health_insurance). Pass it explicitly to run on health_insurance despite the default exclusion.",
    )
    parser.add_argument(
        "--run-id", default=None,
        help="Target a specific wandb run by its id (e.g. 'xx5iw3ft'). When set, "
             "the script only processes the dataset name reported by that run's "
             "display_name, ignoring --datasets and the discovery loop. Useful "
             "for re-running metrics on a specific experiment after a fix.",
    )
    parser.add_argument(
        "--entity", default=WANDB_ENTITY,
        help=f"W&B entity (default: {WANDB_ENTITY}).",
    )
    parser.add_argument(
        "--project", default=WANDB_PROJECT,
        help=f"W&B project (default: {WANDB_PROJECT}).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Fetch runs from wandb and compute metrics, but don't upload tables.",
    )
    parser.add_argument(
        "--no-wandb", action="store_true",
        help="Skip wandb entirely; only write the local markdown report. Useless without local JSONs, since the script no longer reads from DPG/outputs/.",
    )
    parser.add_argument(
        "--offline", action="store_true",
        help="Use ``wandb`` in offline mode (sync later with ``wandb sync``).",
    )
    parser.add_argument(
        "--include-incomplete", action="store_true",
        help="Also accept wandb runs that don't have all 6 structure artifacts (rows will show (missing) for the absent variants). Useful for re-running metrics on a run that predates the deduplication / negative passes.",
    )
    parser.add_argument(
        "--equiv-samples", type=int, default=200,
        help="Rows sampled from the local dataset CSV to compute the equivalence "
             "metrics (default: 200). Graphs are walked once per row, so this "
             "drives the equivalence runtime.",
    )
    parser.add_argument(
        "--no-equivalence", action="store_true",
        help="Skip the decision/explanation agreement columns (they need the "
             "dataset CSV on local disk; the other metrics only need wandb).",
    )
    parser.add_argument(
        "--faith-samples", type=int, default=50,
        help="Rows sampled from the local dataset CSV to compute the faithfulness "
             "metrics (default: 50). Smaller than --equiv-samples because each "
             "sample requires re-fitting a RandomForest and walking it. "
             "Ignored when --no-faithfulness is set.",
    )
    parser.add_argument(
        "--no-faithfulness", action="store_true",
        help="Skip the faithfulness columns (they need the dataset CSV on local "
             "disk AND a re-trained RandomForest, which is the slowest step).",
    )
    args = parser.parse_args(argv)

    # When --run-id is given, we only ever process the one dataset that
    # run belongs to. The dataset name is taken from the run itself, so
    # the local datasets-dir scan and the --datasets filter are
    # bypassed -- the run id is the entire input.
    if args.run_id:
        print(f"Targeting single wandb run id: {args.run_id}")
        run = fetch_run_by_id(args.run_id, args.entity, args.project)
        if run is None:
            print(f"ERROR: could not load run {args.run_id} from {args.entity}/{args.project}")
            return 1
        # ``display_name`` is the dataset folder the experiment was
        # trained on; fall back to ``name`` for older runs that predate
        # the display_name field.
        target_dataset = run.display_name or run.name
        if not target_dataset:
            print(f"ERROR: run {args.run_id} has no display_name; cannot infer dataset.")
            return 1
        print(f"Run dataset (from display_name): {target_dataset}")
        dataset_names = [target_dataset]
    else:
        dataset_names = discover_dataset_names(args.datasets_dir, args.datasets)
        if not dataset_names:
            print(f"ERROR: no datasets found under {args.datasets_dir}")
            return 1

    print("=" * 64)
    print("CATEGORICAL DPG METRICS (basic vs 4 grouping variants)")
    print("=" * 64)
    print(f"Datasets:        {len(dataset_names)}  ({', '.join(dataset_names)})")
    print(f"W&B project:     {args.project}  (entity: {args.entity or '(account default)'})")
    print(f"Dry run:         {args.dry_run}")
    print(f"No wandb:        {args.no_wandb}")
    print(f"Report output:   {args.output_root / 'metrics_report.md'}")
    print("=" * 64)

    use_wandb = WANDB_AVAILABLE and not args.no_wandb
    if not use_wandb and not args.no_wandb:
        print("wandb not available; cannot pull structure JSONs without wandb.")
        print("(Re-run with --no-wandb suppressed once wandb is importable.)")
        return 1

    per_dataset: Dict[str, Dict[str, object]] = {}
    total_start = time.time()
    own_cache = False
    cache_root = args.artifact_cache
    if cache_root is None:
        cache_root = pathlib.Path(tempfile.mkdtemp(prefix="dpg_metric_cache_"))
        own_cache = True

    try:
        for idx, dataset_name in enumerate(dataset_names, 1):
            print(f"\n[{idx}/{len(dataset_names)}] {dataset_name}")
            print("-" * 40)

            if args.run_id:
                # Already loaded above; reuse it for every dataset in the
                # single-run path (there should be exactly one).
                pass
            else:
                run = fetch_latest_finished_run(
                    dataset_name,
                    args.entity,
                    args.project,
                    args.offline,
                    require_all_artifacts=not args.include_incomplete,
                )
                if run is None:
                    print(f"  [wandb] no qualifying finished run for {dataset_name}; skipping")
                    continue

            run_url = (
                run.url
                or f"https://wandb.ai/{args.entity}/{args.project}/runs/{run.id}"
            )
            print(f"  [wandb] latest qualifying run: {run.id}  ({run_url})")

            dataset_cache = cache_root / f"{dataset_name}__{run.id}"
            variant_metrics = compute_dataset_metrics_from_run(
                run, dataset_name, dataset_cache
            )

            # Report per-variant artifact presence + missing-variant warning.
            for variant_label, variant_ns in VARIANTS:
                m = variant_metrics.get(variant_label, {})
                if m.get("_missing"):
                    err = m.get("_error", "")
                    print(f"  - {variant_label} ({variant_ns}): missing{(' -- ' + err) if err else ''}")
                else:
                    print(f"  - {variant_label}: ok")

            # Graph-vs-graph equivalence against BASIC. Needs the local
            # dataset CSV for rows; the structures themselves came from
            # wandb above.
            if not args.no_equivalence:
                add_equivalence_metrics(
                    dataset_name, variant_metrics, args.datasets_dir, args.equiv_samples
                )

            # Faithfulness (graph walk vs black-box model). Runs BEFORE
            # ``strip_internal_keys`` because it needs ``_structure``.
            # Per-variant failures are non-fatal -- they leave the
            # columns as None rather than taking the whole row down.
            if not args.no_faithfulness:
                add_faithfulness_metrics(
                    dataset_name,
                    variant_metrics,
                    args.datasets_dir,
                    args.faith_samples,
                )
            strip_internal_keys(variant_metrics)

            per_dataset[dataset_name] = {
                "metrics": variant_metrics,
                "run_url": run_url,
                "run_id": run.id,
            }

            # Push the table to wandb.
            if not args.dry_run and use_wandb:
                try:
                    log_table_to_run(
                        run.id,
                        dataset_name,
                        variant_metrics,
                        run_url,
                        entity=args.entity,
                        project=args.project,
                    )
                    print(f"  [wandb] logged metrics table to run {run.id}")
                except Exception as exc:  # noqa: BLE001
                    print(f"  [wandb] failed to log table: {exc}")
    finally:
        if own_cache and cache_root.exists():
            shutil.rmtree(cache_root, ignore_errors=True)

    total_elapsed = time.time() - total_start

    # --- Console summary --------------------------------------------------
    print("\n" + "=" * 64)
    print("SUMMARY")
    print("=" * 64)
    print(render_console_table(per_dataset))
    print(f"\nElapsed: {total_elapsed:.1f}s")

    # --- Local markdown report -------------------------------------------
    if per_dataset:
        # When targeting a single run by id, write a per-run file so
        # successive ``--run-id`` invocations don't clobber each other.
        # The all-datasets path keeps the canonical ``metrics_report.md``
        # name for backwards compatibility.
        if args.run_id:
            report_path = args.output_root / f"metrics_report_{args.run_id}.md"
        else:
            report_path = args.output_root / "metrics_report.md"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(render_markdown_report(per_dataset), encoding="utf-8")
        print(f"\nMarkdown report: {report_path}")
    else:
        print("\nNothing to report (no datasets produced metrics).")

    return 0


if __name__ == "__main__":
    sys.exit(main())
