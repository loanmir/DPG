#!/usr/bin/env python3
"""Run a full basic-DPG + categorical-grouping pipeline for every dataset.

For each dataset folder under ``DPG/datasets/`` (every entry except
``dummy_dataset``, which is toy data used only for gridsearch sweeps), this
script:

1. Trains a RandomForest on the dataset's CSV and builds a "basic" DPG
   explanation -- the same recipe as ``examples/quickstart.py`` (one-hot
   encode categorical columns, 5-fold CV, ``DPGExplainer.explain_global``
   with ``communities=True``).
2. Feeds the resulting DPG structure JSON through the three grouping
   scripts living next to this one: ``cat_grouping.py``,
   ``cat_grouping_split.py`` and ``cat_grouping_split_conjunction.py``.
3. Writes everything under ``DPG/outputs/categorical/<dataset>/`` in four fixed
   subfolders:

     * ``BASIC DPG``                      -- the raw DPGExplainer output
     * ``GROUPED DPG``                    -- cat_grouping.py output
     * ``GROUPED-SPLIT DPG``              -- cat_grouping_split.py output
     * ``GROUPED-SPLIT-CONJUCTION DPG``   -- cat_grouping_split_conjunction.py output

Every dataset's target column is assumed to be its last CSV column (titanic's
``titanic.csv`` was reordered on disk so ``Survived`` is last, matching every
other dataset). Feature columns are one-hot encoded as-is (same as
``gridsearch_dpg.py``), except columns listed in ``DROP_COLUMNS_OVERRIDES``
-- currently titanic's ``PassengerId``, ``Name``, ``Ticket`` and ``Cabin``,
which are IDs / near-unique free text that would otherwise explode into
hundreds of one-hot columns. Datasets listed in ``PERC_VAR_OVERRIDES`` use a
smaller ``perc_var`` than ``config.yaml``'s default -- titanic's RandomForest
otherwise produces zero surviving decision paths at the default 0.01.

Optionally logs everything to Weights & Biases (project ``dpg-categorical``,
same entity as the counterfactual pipeline), one run per dataset.

Usage
-----
    python DPG/categorical/run_all_categorical_experiments.py
    python DPG/categorical/run_all_categorical_experiments.py --datasets iris titanic
    python DPG/categorical/run_all_categorical_experiments.py --dry-run
    python DPG/categorical/run_all_categorical_experiments.py --no-wandb
"""

from __future__ import annotations

import argparse
import importlib
import json
import pathlib
import shutil
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

# This script only ever saves plots to disk (show=False everywhere) and
# never displays them, so force the non-interactive Agg backend before
# ``dpg`` (which imports matplotlib.pyplot) is loaded. Otherwise matplotlib
# defaults to the interactive TkAgg backend whenever tkinter is available,
# and garbage-collecting the leftover Tk widget objects at interpreter
# shutdown prints harmless but noisy "Exception ignored in ... main thread
# is not in main loop" tracebacks.
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import yaml
from networkx.readwrite import json_graph
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import KFold

# --- Repo / import setup --------------------------------------------------
SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent  # DPG/

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dpg import DPGExplainer  # noqa: E402
from metrics.graph import GraphMetrics  # noqa: E402

# --- Constants --------------------------------------------------------------

WANDB_PROJECT = "dpg-categorical"
WANDB_ENTITY = "mllab-ts-universit-di-trieste"
WANDB_GROUP = "run_all_categorical_basic"

DEFAULT_DATASETS_DIR = REPO_ROOT / "datasets"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs" / "categorical"
DEFAULT_CONFIG_PATH = REPO_ROOT / "config.yaml"

# Folder that holds toy/sweep data, not a real dataset.
EXCLUDED_DATASET_DIRS = {"dummy_dataset"}

# Fixed output subfolder names, exactly as requested.
BASIC_LABEL = "BASIC DPG"
GROUPED_LABEL = "GROUPED DPG"
SPLIT_LABEL = "GROUPED-SPLIT DPG"
CONJUNCTION_LABEL = "GROUPED-SPLIT-CONJUCTION DPG"

# The three grouping scripts to run against each basic DPG structure,
# keyed by their importable module name (relative to the ``categorical``
# package) and mapped to the output subfolder they write into.
GROUPING_MODULES: Tuple[Tuple[str, str], ...] = (
    ("categorical.cat_grouping", GROUPED_LABEL),
    ("categorical.cat_grouping_split", SPLIT_LABEL),
    ("categorical.cat_grouping_split_conjunction", CONJUNCTION_LABEL),
)

# Columns to drop before one-hot encoding: pure row IDs or near-unique
# free text that would otherwise explode into hundreds of one-hot columns.
DROP_COLUMNS_OVERRIDES: Dict[str, List[str]] = {
    "titanic": ["PassengerId", "Name", "Ticket", "Cabin"],
}

# Per-dataset perc_var override (see config.yaml's ``dpg.default.perc_var``).
# perc_var is a *minimum* fraction of paths a pattern must appear in to be
# kept, so a smaller value keeps MORE paths. The repo default (0.01) leaves
# titanic with zero surviving paths (its RandomForest produces too many
# distinct root-to-leaf paths for any one to clear a 1% bar), so it needs a
# smaller perc_var than the default.
PERC_VAR_OVERRIDES: Dict[str, float] = {
    "titanic": 0.005,
}

NUM_TREES = 10
RANDOM_STATE = 27
N_SPLITS = 5
COMMUNITY_THRESHOLD = 0.2


# ---------------------------------------------------------------------------
# WandB (optional)
# ---------------------------------------------------------------------------

try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:  # pragma: no cover - wandb is a hard dep of the project
    WANDB_AVAILABLE = False
    wandb = None  # type: ignore[assignment]


def wandb_init_for_dataset(dataset_name: str, csv_path: pathlib.Path, offline: bool):
    if not WANDB_AVAILABLE:
        return None
    mode = "offline" if offline else "online"
    return wandb.init(
        entity=WANDB_ENTITY,
        project=WANDB_PROJECT,
        group=WANDB_GROUP,
        job_type="basic_dpg_and_grouping",
        name=dataset_name,
        config={"dataset": dataset_name, "csv": str(csv_path)},
        mode=mode,
    )


def wandb_log_artifacts(run, artifacts: List[pathlib.Path]) -> None:
    """Log each path as a wandb artifact (and image, for PNGs)."""
    if run is None:
        return
    for path in artifacts:
        if not path.is_file():
            continue
        ext = path.suffix.lower()
        artifact_name = f"{path.parent.name}__{path.stem}".replace(" ", "_")
        if ext == ".png":
            try:
                run.log({f"images/{path.parent.name}/{path.stem}": wandb.Image(str(path))})
            except Exception as exc:  # pragma: no cover - best effort
                print(f"  [wandb] could not log image {path.name}: {exc}")
            try:
                art = wandb.Artifact(artifact_name, type="dpg_image")
                art.add_file(str(path))
                run.log_artifact(art)
            except Exception as exc:  # pragma: no cover - best effort
                print(f"  [wandb] could not log artifact {path.name}: {exc}")
        elif ext == ".json":
            try:
                art = wandb.Artifact(artifact_name, type="dpg_structure")
                art.add_file(str(path))
                run.log_artifact(art)
            except Exception as exc:  # pragma: no cover - best effort
                print(f"  [wandb] could not log artifact {path.name}: {exc}")
        else:
            try:
                art = wandb.Artifact(artifact_name, type="dpg_aux")
                art.add_file(str(path))
                run.log_artifact(art)
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Dataset discovery + loading
# ---------------------------------------------------------------------------


def discover_dataset_csvs(
    datasets_dir: pathlib.Path, only: Optional[Sequence[str]]
) -> List[Tuple[str, pathlib.Path]]:
    """Return ``(dataset_name, csv_path)`` for every dataset subfolder.

    Skips ``dummy_dataset`` and any subfolder with no CSV inside. If a
    subfolder has more than one CSV, the first (sorted) one is used and a
    warning is printed.
    """
    if not datasets_dir.exists():
        return []
    only_set = set(only) if only else None
    pairs: List[Tuple[str, pathlib.Path]] = []
    for entry in sorted(datasets_dir.iterdir()):
        if not entry.is_dir() or entry.name in EXCLUDED_DATASET_DIRS:
            continue
        if only_set is not None and entry.name not in only_set:
            continue
        csvs = sorted(entry.glob("*.csv"))
        if not csvs:
            print(f"  [skip] {entry.name}: no CSV file found")
            continue
        if len(csvs) > 1:
            print(
                f"  [warn] {entry.name}: {len(csvs)} CSVs found, "
                f"using {csvs[0].name}"
            )
        pairs.append((entry.name, csvs[0]))
    return pairs


def load_dataset(csv_path: pathlib.Path, dataset_name: str):
    """Load a CSV, split features/labels, one-hot encode the features.

    The target column is assumed to be the last CSV column (true for every
    dataset under ``DPG/datasets/``). Columns listed in
    ``DROP_COLUMNS_OVERRIDES`` (IDs / near-unique free text) are dropped
    before encoding. The delimiter (``,`` vs ``;``) is auto-detected since
    datasets in this repo use both.
    """
    df = pd.read_csv(csv_path, sep=None, engine="python", encoding="utf-8-sig")

    drop_cols = [c for c in DROP_COLUMNS_OVERRIDES.get(dataset_name, []) if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    features = df.iloc[:, :-1]
    labels = df.iloc[:, -1]

    features_enc = pd.get_dummies(features, drop_first=False)
    features_enc = features_enc.replace([np.inf, -np.inf], np.nan).fillna(features_enc.mean())
    features_enc = np.round(features_enc, 3)
    feature_names = list(features_enc.columns)
    return features_enc, labels, feature_names


def train_cv(model, X, y, n_splits: int = N_SPLITS, random_state: int = RANDOM_STATE):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    accs, f1s = [], []
    last_train = None
    for tr, te in kf.split(X):
        X_tr, X_te = X.iloc[tr], X.iloc[te]
        y_tr, y_te = y.iloc[tr], y.iloc[te]
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_te)
        accs.append(accuracy_score(y_te, y_pred))
        f1s.append(f1_score(y_te, y_pred, average="weighted"))
        last_train = (X_tr, y_tr)
    return float(np.mean(accs)), float(np.mean(f1s)), last_train


# ---------------------------------------------------------------------------
# Structure JSON persistence (needed as input for the grouping scripts)
# ---------------------------------------------------------------------------


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def save_dpg_structure_json(explanation, out_path: pathlib.Path, run_id: str,
                             feature_names, target_names) -> None:
    graph_data = json_graph.node_link_data(explanation.graph)
    labeled_nodes = [{"id": nid, "label": lab} for nid, lab in explanation.nodes]
    payload = {
        "run_id": run_id,
        "feature_names": [str(n) for n in feature_names],
        "target_names": [str(n) for n in target_names],
        "community_threshold": explanation.community_threshold,
        "nodes": labeled_nodes,
        "graph": graph_data,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=_json_default)


# ---------------------------------------------------------------------------
# Per-dataset DPG config (perc_var override)
# ---------------------------------------------------------------------------


def resolve_config_path(
    dataset_name: str, base_config_path: pathlib.Path, staging_dir: pathlib.Path
) -> pathlib.Path:
    """Return the config.yaml path to use for this dataset's DPG run.

    Datasets in ``PERC_VAR_OVERRIDES`` get a copy of the base config with
    ``dpg.default.perc_var`` overridden, written into ``staging_dir`` so it
    travels alongside that dataset's other artifacts. Everything else just
    uses ``base_config_path`` unchanged.
    """
    override = PERC_VAR_OVERRIDES.get(dataset_name)
    if override is None:
        return base_config_path
    with open(base_config_path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    cfg.setdefault("dpg", {}).setdefault("default", {})["perc_var"] = float(override)
    out_path = staging_dir / "config.yaml"
    with open(out_path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(cfg, fh, sort_keys=False)
    return out_path


# ---------------------------------------------------------------------------
# Per-dataset run
# ---------------------------------------------------------------------------


def run_one_dataset(
    dataset_name: str,
    csv_path: pathlib.Path,
    output_root: pathlib.Path,
    visualization_config: dict,
    config_path: pathlib.Path,
) -> dict:
    """Produce the basic DPG run for one dataset, then the three grouped
    variants, writing everything under ``output_root/<dataset_name>/``.
    """
    dataset_out = output_root / dataset_name
    run_id = dataset_name

    # The basic run is staged in a folder named exactly ``run_id`` because
    # the grouping scripts' ``_process_subdir`` derives the run id (and the
    # structure/metrics filenames it looks for) from the subdir's basename.
    # Once grouping is done, this staging folder is renamed to "BASIC DPG".
    staging_dir = dataset_out / run_id
    staging_dir.mkdir(parents=True, exist_ok=True)

    X, y, feature_names = load_dataset(csv_path, dataset_name)

    model = RandomForestClassifier(n_estimators=NUM_TREES, random_state=RANDOM_STATE)
    acc, f1, (X_train, y_train) = train_cv(model, X, y)

    effective_config_path = resolve_config_path(dataset_name, config_path, staging_dir)

    target_names = np.unique(y).astype(str).tolist()
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

    save_dpg_structure_json(
        explanation,
        staging_dir / f"{run_id}_dpg_structure.json",
        run_id,
        feature_names,
        target_names,
    )
    with open(staging_dir / f"{run_id}_dpg_class_boundaries.txt", "w") as f:
        for key, value in explanation.class_boundaries.items():
            f.write(f"{key}: {value}\n")
    explanation.node_metrics.to_csv(
        staging_dir / f"{run_id}_node_metrics.csv", encoding="utf-8"
    )
    explanation.edge_metrics.to_csv(
        staging_dir / f"{run_id}_edge_metrics.csv", encoding="utf-8"
    )
    if explanation.communities is not None:
        GraphMetrics.communities_to_csv(
            explanation.communities,
            str(staging_dir / f"{run_id}_dpg_communities.txt"),
        )

    run_name = f"{run_id}_DPG"
    explainer.plot(
        run_name,
        explanation=explanation,
        save_dir=str(staging_dir),
        class_flag=False,
        export_pdf=True,
        show=False,
    )
    explainer.plot_communities(
        run_name,
        explanation=explanation,
        save_dir=str(staging_dir),
        class_flag=True,
        export_pdf=True,
        show=False,
    )

    n_nodes = len(explanation.node_metrics)
    n_edges = len(explanation.edge_metrics)
    n_comms = 0
    if explanation.communities and "Clusters" in explanation.communities:
        n_comms = sum(
            1 for k, v in explanation.communities["Clusters"].items()
            if k != "Ambiguous" and v
        )

    # --- Three grouping passes, each reading the staged structure JSON ---
    for module_name, label in GROUPING_MODULES:
        out_dir = dataset_out / label
        out_dir.mkdir(parents=True, exist_ok=True)
        module = importlib.import_module(module_name)
        try:
            out_png = module._process_subdir(
                str(staging_dir), str(out_dir), visualization_config
            )
        except Exception as exc:  # noqa: BLE001 - report and continue
            print(f"  [err]  {module_name} on {dataset_name}: {exc}")
            continue
        if out_png is None:
            print(f"  [skip] {module_name} on {dataset_name}: no structure JSON")
        else:
            print(f"  [ok]   {module_name} -> {out_dir}")

    # --- Move the staging folder into its final "BASIC DPG" name ---------
    basic_dir = dataset_out / BASIC_LABEL
    if basic_dir.exists():
        shutil.rmtree(basic_dir)
    shutil.move(str(staging_dir), str(basic_dir))

    artifacts = [p for p in dataset_out.rglob("*") if p.is_file()]

    return {
        "dataset": dataset_name,
        "accuracy": acc,
        "f1": f1,
        "n_nodes": n_nodes,
        "n_edges": n_edges,
        "n_communities": n_comms,
        "output_dir": dataset_out,
        "artifacts": artifacts,
    }


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def generate_report(
    results: Dict[str, list],
    total_elapsed: float,
    output_path: pathlib.Path,
) -> None:
    lines: List[str] = []
    lines.append("# Categorical DPG Batch Run Report")
    lines.append("")
    lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"**Total Duration:** {total_elapsed:.1f}s ({total_elapsed/60:.1f} minutes)")
    lines.append(f"**W&B project:** `{WANDB_PROJECT}`")
    lines.append(f"**W&B entity:**  `{WANDB_ENTITY or '(account default)'}`")
    lines.append(f"**W&B group:**   `{WANDB_GROUP}`")
    lines.append("")

    lines.append("## Summary")
    lines.append("")
    lines.append(f"- Successful: {len(results['success'])}")
    lines.append(f"- Failed:     {len(results['failed'])}")
    lines.append("")

    if results["success"]:
        lines.append("## Successful Runs")
        lines.append("")
        lines.append("| Dataset | Accuracy | F1 | Nodes | Edges | Communities | Output |")
        lines.append("|---|---|---|---|---|---|---|")
        for info in sorted(results["success"], key=lambda i: i["dataset"]):
            lines.append(
                f"| {info['dataset']} | {info['accuracy']:.3f} | {info['f1']:.3f} | "
                f"{info['n_nodes']} | {info['n_edges']} | {info['n_communities']} | "
                f"`{info['output_dir']}` |"
            )
        lines.append("")

    if results["failed"]:
        lines.append("## Failed Runs")
        lines.append("")
        for info in sorted(results["failed"], key=lambda i: i["dataset"]):
            lines.append(f"- `{info['dataset']}`: {info.get('error', 'unknown error')}")
        lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")


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
        help="Root containing one subdir per dataset (default: DPG/datasets).",
    )
    parser.add_argument(
        "--datasets", nargs="+", default=None,
        help="Restrict to a subset of dataset subdir names (default: all, except dummy_dataset).",
    )
    parser.add_argument(
        "--output-root",
        type=pathlib.Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Root under which each dataset gets its own subfolder (default: DPG/outputs/categorical).",
    )
    parser.add_argument(
        "--config",
        type=pathlib.Path,
        default=DEFAULT_CONFIG_PATH,
        help="DPG config YAML (perc_var/decimal_threshold + visualization styling).",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Cap the number of datasets processed.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print what would run; don't actually train models or invoke grouping scripts.",
    )
    parser.add_argument(
        "--offline", action="store_true",
        help="Run wandb in offline mode (sync later with `wandb sync`).",
    )
    parser.add_argument(
        "--no-wandb", action="store_true",
        help="Disable wandb entirely; just produce the local outputs.",
    )
    args = parser.parse_args(argv)

    if not args.config.exists():
        print(f"ERROR: config file not found: {args.config}")
        return 1
    with open(args.config, "r", encoding="utf-8") as fh:
        visualization_config = yaml.safe_load(fh)

    pairs = discover_dataset_csvs(args.datasets_dir, args.datasets)
    if args.datasets and not pairs:
        print(f"ERROR: none of the requested datasets exist under {args.datasets_dir}")
        return 1
    if not pairs:
        print(f"ERROR: no dataset CSVs found under {args.datasets_dir}")
        return 1
    if args.limit is not None:
        pairs = pairs[: args.limit]

    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)

    print("=" * 64)
    print("CATEGORICAL DPG BATCH RUNNER (basic run + 3 grouping passes)")
    print("=" * 64)
    print(f"Start time:       {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Datasets to run:  {len(pairs)}  ({', '.join(name for name, _ in pairs)})")
    print(f"Output root:      {output_root}")
    print(f"Config:           {args.config}")
    print(f"Dry run:          {args.dry_run}")
    print(f"W&B project:      {WANDB_PROJECT}  (entity: {WANDB_ENTITY or '(account default)'})")
    print("=" * 64)

    use_wandb = WANDB_AVAILABLE and not args.no_wandb
    if not use_wandb and not args.no_wandb:
        print("wandb not installed; running without W&B logging.")

    results: Dict[str, list] = {"success": [], "failed": []}
    total_start = time.time()

    for idx, (dataset_name, csv_path) in enumerate(pairs, 1):
        print(f"\n[{idx}/{len(pairs)}] {dataset_name}  ({csv_path})")
        print("-" * 40)

        if args.dry_run:
            print(f"  [dry-run] would train + explain + group {dataset_name}")
            results["success"].append({"dataset": dataset_name})
            continue

        run = None
        if use_wandb:
            try:
                run = wandb_init_for_dataset(dataset_name, csv_path, args.offline)
            except Exception as exc:  # noqa: BLE001
                print(f"  [wandb] init failed: {exc}; continuing without wandb")
                run = None

        try:
            info = run_one_dataset(
                dataset_name, csv_path, output_root, visualization_config, args.config
            )
        except Exception as exc:  # noqa: BLE001
            print(f"  FAIL {dataset_name}: {exc}")
            results["failed"].append({"dataset": dataset_name, "error": str(exc)})
            if run is not None:
                try:
                    run.finish()
                except Exception:
                    pass
            continue

        print(
            f"  OK -> {info['output_dir']}  "
            f"acc={info['accuracy']:.3f}  f1={info['f1']:.3f}  "
            f"nodes={info['n_nodes']}  edges={info['n_edges']}  comms={info['n_communities']}"
        )
        results["success"].append(info)

        if run is not None:
            try:
                wandb_log_artifacts(run, info["artifacts"])
                run.log({
                    "accuracy": info["accuracy"],
                    "f1": info["f1"],
                    "n_nodes": info["n_nodes"],
                    "n_edges": info["n_edges"],
                    "n_communities": info["n_communities"],
                })
            except Exception as exc:  # noqa: BLE001
                print(f"  [wandb] log failed: {exc}")
            try:
                run.finish()
            except Exception:
                pass

    total_elapsed = time.time() - total_start

    print("\n" + "=" * 64)
    print("SUMMARY")
    print("=" * 64)
    print(f"End time:    {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total time:  {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")
    print(f"Successful:  {len(results['success'])}")
    print(f"Failed:      {len(results['failed'])}")

    if results["failed"]:
        print("\nFailed datasets:")
        for info in results["failed"]:
            print(f"  - {info['dataset']}: {info.get('error', 'unknown error')}")

    if not args.dry_run:
        report_path = output_root / "report.md"
        generate_report(results, total_elapsed, report_path)
        print(f"\nReport saved to {report_path}")
        print(f"Artifacts under {output_root}")

    return 1 if results["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
