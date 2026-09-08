#!/usr/bin/env python3
"""Run the fidelity + agreement equivalence checks on ONE dataset.

A standalone driver for ``dpg_equivalence.py`` -- it does not touch the
batch pipeline (``run_all_categorical_experiments.py`` /
``run_all_metrics.py``) and writes nothing to Weights & Biases. Use it to
sanity-check the equivalence numbers on a single dataset before deciding
whether to fold these metrics into the batch runs.

What it does, in order:

1. Loads the dataset CSV twice over: once one-hot encoded (what the
   RandomForest and the BASIC DPG's labels use) and once raw / un-encoded
   (what the grouped variants' ``base IN {cats}`` labels reference).
   Both are kept row-aligned so one sample can be evaluated against
   either label grammar.
2. Trains a RandomForest on the last CV fold and builds the BASIC DPG
   from that fold's training rows -- the same recipe as
   ``run_all_categorical_experiments.run_one_dataset``.
3. Runs the three grouping passes (``cat_grouping``,
   ``cat_grouping_split``, ``cat_grouping_split_conjunction``) over the
   BASIC structure JSON, exactly as the batch runner does.
4. Reloads all four variants' structure JSONs as graphs and, for every
   held-out test sample, walks each graph to get its own predicted class.
5. Reports per variant:
     * fidelity      -- how often the graph agrees with the RandomForest
     * coverage      -- how often the graph could classify the row at all
     * agreement     -- how often the graph agrees with BASIC (the actual
                        equivalence signal; needs no model at all)

Usage
-----
    python DPG/categorical/check_equivalence.py
    python DPG/categorical/check_equivalence.py --dataset titanic
    python DPG/categorical/check_equivalence.py --n-samples 500
    python DPG/categorical/check_equivalence.py --keep-workdir
"""

from __future__ import annotations

import argparse
import importlib
import pathlib
import shutil
import sys
import tempfile
import time
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

# Save plots to disk only; never open a window (see the same note in
# run_all_categorical_experiments.py).
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent  # DPG/

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dpg import DPGExplainer  # noqa: E402
from categorical import dpg_equivalence as eq  # noqa: E402
from categorical.run_all_categorical_experiments import (  # noqa: E402
    DROP_COLUMNS_OVERRIDES,
    GROUPING_MODULES,
    NUM_TREES,
    N_SPLITS,
    RANDOM_STATE,
    COMMUNITY_THRESHOLD,
    DEFAULT_CONFIG_PATH,
    DEFAULT_DATASETS_DIR,
    resolve_config_path,
    save_dpg_structure_json,
)

BASIC_LABEL = "BASIC DPG"


# ---------------------------------------------------------------------------
# Dataset loading (encoded + raw, row-aligned)
# ---------------------------------------------------------------------------


def load_dataset_both_views(
    csv_path: pathlib.Path, dataset_name: str
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, List[str]]:
    """Return ``(X_encoded, X_raw, y, feature_names)``.

    Mirrors ``run_all_categorical_experiments.load_dataset`` step for step
    (same delimiter sniffing, same dropped columns, same missing-target
    handling, same rounding) but additionally hands back the raw
    pre-one-hot feature frame. Both frames share a clean 0..n-1 index so
    row ``i`` is the same sample in each.
    """
    df = pd.read_csv(csv_path, sep=None, engine="python", encoding="utf-8-sig")

    drop_cols = [c for c in DROP_COLUMNS_OVERRIDES.get(dataset_name, []) if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    target_col = df.columns[-1]
    n_before = len(df)
    df = df.dropna(subset=[target_col]).reset_index(drop=True)
    if len(df) != n_before:
        print(f"  [info] dropped {n_before - len(df)} row(s) with missing target")

    features_raw = df.iloc[:, :-1]
    labels = df.iloc[:, -1]

    features_enc = pd.get_dummies(features_raw, drop_first=False)
    features_enc = features_enc.replace([np.inf, -np.inf], np.nan).fillna(features_enc.mean())
    features_enc = np.round(features_enc, 3)

    return features_enc, features_raw, labels, list(features_enc.columns)


# ---------------------------------------------------------------------------
# Build the four variants
# ---------------------------------------------------------------------------


def build_variants(
    dataset_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    feature_names: List[str],
    target_names: List[str],
    model: RandomForestClassifier,
    workdir: pathlib.Path,
    config_path: pathlib.Path,
    visualization_config: dict,
) -> Dict[str, Tuple[object, List[Tuple[str, str]]]]:
    """Build the BASIC DPG plus the three grouped variants inside
    ``workdir``, and return ``{variant_label: (graph, nodes)}``.

    The BASIC run is staged in a folder named exactly ``run_id`` because
    the grouping scripts' ``_process_subdir`` derives the run id (and the
    filenames it looks for) from the subdir's basename -- same constraint
    the batch runner works around.
    """
    run_id = dataset_name
    staging_dir = workdir / run_id
    staging_dir.mkdir(parents=True, exist_ok=True)

    effective_config_path = resolve_config_path(dataset_name, config_path, staging_dir)

    explainer = DPGExplainer(
        model=model,
        feature_names=feature_names,
        target_names=target_names,
        config_file=str(effective_config_path),
    )
    explanation = explainer.explain_global(
        X_train.values,
        communities=True,
        community_threshold=COMMUNITY_THRESHOLD,
    )

    basic_json = staging_dir / f"{run_id}_dpg_structure.json"
    save_dpg_structure_json(explanation, basic_json, run_id, feature_names, target_names)
    # The grouping scripts reuse these CSVs when present (and synthesise
    # equivalents when absent) -- write them so this run matches the
    # batch pipeline's inputs exactly.
    explanation.node_metrics.to_csv(staging_dir / f"{run_id}_node_metrics.csv", encoding="utf-8")
    explanation.edge_metrics.to_csv(staging_dir / f"{run_id}_edge_metrics.csv", encoding="utf-8")

    variants: Dict[str, Tuple[object, List[Tuple[str, str]]]] = {}
    variants[BASIC_LABEL] = eq.load_variant(str(basic_json))
    print(f"  [ok]   {BASIC_LABEL}: "
          f"{variants[BASIC_LABEL][0].number_of_nodes()} nodes, "
          f"{variants[BASIC_LABEL][0].number_of_edges()} edges")

    for module_name, label in GROUPING_MODULES:
        out_dir = workdir / label
        out_dir.mkdir(parents=True, exist_ok=True)
        module = importlib.import_module(module_name)
        try:
            module._process_subdir(str(staging_dir), str(out_dir), visualization_config)
        except Exception as exc:  # noqa: BLE001 - report and continue
            print(f"  [err]  {module_name}: {exc}")
            continue

        grouped_jsons = sorted(out_dir.glob("*_structure.json"))
        if not grouped_jsons:
            print(f"  [skip] {label}: no structure JSON produced")
            continue
        graph, nodes = eq.load_variant(str(grouped_jsons[0]))
        variants[label] = (graph, nodes)
        print(f"  [ok]   {label}: {graph.number_of_nodes()} nodes, "
              f"{graph.number_of_edges()} edges")

    return variants


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--dataset", default="credit_card_approval",
                        help="Dataset subdir name under DPG/datasets (default: credit_card_approval).")
    parser.add_argument("--datasets-dir", type=pathlib.Path, default=DEFAULT_DATASETS_DIR)
    parser.add_argument("--config", type=pathlib.Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--n-samples", type=int, default=200,
                        help="How many held-out test rows to check (default: 200). "
                             "Graph traversal is per-sample, so this drives the runtime.")
    parser.add_argument("--workdir", type=pathlib.Path, default=None,
                        help="Where to build the four variants (default: a temp dir).")
    parser.add_argument("--keep-workdir", action="store_true",
                        help="Don't delete the temp workdir on exit (useful to inspect the JSONs).")
    parser.add_argument("--csv-out", type=pathlib.Path, default=None,
                        help="Optional path to write the results table as CSV.")
    args = parser.parse_args(argv)

    dataset_dir = args.datasets_dir / args.dataset
    csvs = sorted(dataset_dir.glob("*.csv"))
    if not csvs:
        print(f"ERROR: no CSV found under {dataset_dir}")
        return 1
    csv_path = csvs[0]

    with open(args.config, "r", encoding="utf-8") as fh:
        visualization_config = yaml.safe_load(fh)

    own_workdir = args.workdir is None
    workdir = args.workdir or pathlib.Path(tempfile.mkdtemp(prefix="dpg_equiv_"))
    workdir.mkdir(parents=True, exist_ok=True)

    print("=" * 64)
    print("DPG EQUIVALENCE CHECK (fidelity vs model, agreement vs BASIC)")
    print("=" * 64)
    print(f"Dataset:    {args.dataset}  ({csv_path.name})")
    print(f"Workdir:    {workdir}")
    print(f"Samples:    {args.n_samples}")
    print("=" * 64)

    total_start = time.time()

    try:
        # --- Load both views, train, split ------------------------------
        X_enc, X_raw, y, feature_names = load_dataset_both_views(csv_path, args.dataset)
        print(f"  [info] {len(X_enc)} rows, {X_raw.shape[1]} raw cols -> {X_enc.shape[1]} encoded cols")

        kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
        train_idx, test_idx = list(kf.split(X_enc))[-1]

        model = RandomForestClassifier(n_estimators=NUM_TREES, random_state=RANDOM_STATE)
        model.fit(X_enc.iloc[train_idx], y.iloc[train_idx])

        target_names = np.unique(y).astype(str).tolist()

        # --- Build the four variants ------------------------------------
        print("\nBuilding variants...")
        variants = build_variants(
            args.dataset,
            X_enc.iloc[train_idx], y.iloc[train_idx],
            feature_names, target_names, model,
            workdir, args.config, visualization_config,
        )
        if BASIC_LABEL not in variants:
            print("ERROR: BASIC DPG could not be built; nothing to compare against.")
            return 1

        # --- Evaluation rows (raw + encoded, one per sample) ------------
        eval_idx = test_idx[: args.n_samples]
        eval_rows = [
            eq.build_eval_row(X_raw.iloc[i], X_enc.iloc[i]) for i in eval_idx
        ]
        model_preds = [str(p) for p in model.predict(X_enc.iloc[eval_idx])]
        print(f"\nChecking {len(eval_rows)} held-out samples...")

        # --- Walk every variant, once per sample ------------------------
        # Two walks per variant: one collapsing to a predicted class
        # (decision-level), one recording the routes taken
        # (explanation-level).
        preds_by_variant: Dict[str, List[Optional[str]]] = {}
        routes_by_variant: Dict[str, List] = {}
        for label, (graph, nodes) in variants.items():
            t0 = time.time()
            preds_by_variant[label] = eq.dpg_predictions(graph, nodes, eval_rows)
            routes_by_variant[label] = [
                eq.dpg_routes(graph, nodes, row) for row in eval_rows
            ]
            print(f"  [ok]   {label}: traversed in {time.time() - t0:.1f}s")

        # --- Fidelity + agreement ---------------------------------------
        basic_preds = preds_by_variant[BASIC_LABEL]
        basic_routes = routes_by_variant[BASIC_LABEL]
        rows = []
        for label, preds in preds_by_variant.items():
            fid = eq.compute_fidelity(preds, model_preds)
            agr = eq.compute_agreement(basic_preds, preds)
            pth = eq.compute_path_agreement(basic_routes, routes_by_variant[label])
            rows.append({
                "variant": label,
                "nodes": variants[label][0].number_of_nodes(),
                "edges": variants[label][0].number_of_edges(),
                "fidelity_vs_RF": fid["fidelity"],
                "coverage": fid["fidelity_coverage"],
                "decision_agreement": agr["decision_agreement"],
                "explanation_agreement": pth["explanation_agreement"],
                "explanation_overlap": pth["explanation_overlap"],
                "explanation_partial": pth["explanation_partial"],
                "n_samples": fid["fidelity_n_samples"],
            })

        df = pd.DataFrame(rows)

        print("\n" + "=" * 64)
        print("RESULTS")
        print("=" * 64)
        with pd.option_context("display.width", 200, "display.max_columns", None):
            print(df.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
        print(f"\nElapsed: {time.time() - total_start:.1f}s")

        print("\nHow to read this:")
        print("  fidelity_vs_RF        -- graph's own prediction vs the RandomForest's")
        print("  coverage              -- fraction of rows the graph could classify at all")
        print("  decision_agreement    -- same DECISION as BASIC on the same rows.")
        print("                           1.000 => grouping preserved the predictions.")
        print("                           WEAKER: a broken graph can still score 1.000 here.")
        print("  explanation_agreement -- same EXPLANATION as BASIC: the conditions each row")
        print("                           was tested against, normalised so a collapsed")
        print("                           chain counts as equal to the nodes it replaced.")
        print("                           1.000 => grouping preserved the reasoning too.")
        print("  explanation_overlap   -- partial credit when routes only partly overlap.")
        print("  explanation_partial   -- rows where route enumeration hit the cap")
        print("                           (comparison is partial for those rows).")

        if args.csv_out:
            args.csv_out.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(args.csv_out, index=False)
            print(f"\nSaved: {args.csv_out}")

    finally:
        if own_workdir and not args.keep_workdir and workdir.exists():
            shutil.rmtree(workdir, ignore_errors=True)
        elif args.keep_workdir:
            print(f"\nWorkdir kept at: {workdir}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
