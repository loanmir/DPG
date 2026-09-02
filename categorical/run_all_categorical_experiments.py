#!/usr/bin/env python3
"""Run categorical features experiments for all datasets with all grouping scripts.

This is the categorical-features counterpart of ``run_all_experiments.py``
(``DPG/counterfactual/scripts/run_all_experiments.py``). Where that
script uses the counterfactual dataset loader and runs
``run_experiment.py`` for each (dataset, method) pair, this script:

* treats every subdirectory of ``DPG/datasets/`` as a dataset (no YAML
  loader, no method axis -- the "method" axis is replaced by the
  grouping script),
* iterates every known *grouping* script inside ``DPG/categorical/``
  (the three ``cat_grouping*`` modules; ``__init__``, this orchestrator
  and the one-hot-rewrite ``categorical_view_conversion.py`` are all
  skipped) and invokes each script's ``_process_subdir`` against the
  matching per-dataset DPG outputs,
* writes all rendered PNGs and rewritten JSON structures into an
  unversioned output directory (defaults to
  ``DPG/outputs/categorical/<timestamp>/``),
* pushes the same artifacts to Weights & Biases under the project
  ``dpg-categorical``, using the same workspace entity as the
  counterfactual ``run_all_experiments.py`` script
  (``mllab-ts-universit-di-trieste``) and a wandb ``group`` of
  ``run_all_categorical`` so the runs appear together in the UI.

The DPG subdirs each grouping script consumes are not produced here --
they are assumed to already exist on disk (e.g. written by
``gridsearch_dpg.py`` / ``gridsearch_dpg_catsim.py`` into
``DPG/examples/results_gridsearch/`` and ``DPG/examples/results_cat/``).
For each dataset we discover which DPG subdirs match the dataset name,
group them per results root, and feed each group to the matching
grouping script.

Usage
-----
    # Full sweep (every dataset x every grouping script, against every
    # known results root)
    python DPG/categorical/run_all_categorical_experiments.py

    # Restrict the dataset and/or script axes
    python DPG/categorical/run_all_categorical_experiments.py --datasets iris german_credit
    python DPG/categorical/run_all_categorical_experiments.py --scripts cat_grouping.py

    # Restrict the DPG results roots searched for each dataset
    python DPG/categorical/run_all_categorical_experiments.py --results-roots \\
        DPG/examples/results_cat DPG/examples/results_gridsearch

    # Skip (dataset, script) combinations that already have artifacts
    python DPG/categorical/run_all_categorical_experiments.py --skip-existing

    # Dry-run / offline / parallel
    python DPG/categorical/run_all_categorical_experiments.py --dry-run
    python DPG/categorical/run_all_categorical_experiments.py --offline
    python DPG/categorical/run_all_categorical_experiments.py --parallel 4
"""

from __future__ import annotations

import argparse
import importlib
import os
import pathlib
import shutil
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Iterable, List, Optional, Tuple

# --- Repo / import setup --------------------------------------------------
# The script now lives under ``DPG/categorical/`` (it isn't counterfactual
# work, so it was moved out of ``DPG/counterfactual/scripts/``). That
# means ``utils.experiment_status`` -- which lives under
# ``DPG/counterfactual/utils/`` -- is no longer importable via the old
# "sibling of the utils package" trick. We add both the DPG project root
# (so ``dpg`` etc. resolve) and the ``DPG/counterfactual/`` directory
# (so ``utils.experiment_status`` resolves) to ``sys.path``. The grouping
# scripts live right next to us under ``CATEGORICAL_DIR`` and don't need
# any extra path entry to find each other.
SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
CATEGORICAL_DIR = SCRIPT_DIR  # DPG/categorical/
REPO_ROOT = SCRIPT_DIR.parent  # DPG/
COUNTERFACTUAL_ROOT = REPO_ROOT / "counterfactual"

for p in (str(REPO_ROOT), str(COUNTERFACTUAL_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from utils.experiment_status import (  # noqa: E402
    PersistentStatus,
    get_experiment_status,
    write_status,
)

# The grouping scripts now live right next to us (in the ``categorical``
# package). We use this path as the default for ``--grouping-dir`` and
# for ``discover_scripts`` so the same folder is consulted regardless of
# where the user runs the script from.
GROUPING_DIR = CATEGORICAL_DIR

# --- Constants ------------------------------------------------------------

# Same workspace entity as the counterfactual ``run_all_experiments.py``
# pipeline (``CounterFactualDPG`` project, ``mllab-ts-universit-di-trieste``
# team). We log into a *new* project so the categorical runs stay isolated
# from the counterfactual ones.
WANDB_PROJECT = "dpg-categorical"
WANDB_ENTITY = "mllab-ts-universit-di-trieste"
WANDB_GROUP = "run_all_categorical"

# Where each ``(dataset, script)`` writes its PNGs + JSON payloads.
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs" / "categorical"

# Names of "results roots" the grouping scripts accept via ``--root``.
# These are checked in order; the first one that contains matching DPG
# subdirs for a given dataset wins.
DEFAULT_RESULTS_ROOTS: Tuple[pathlib.Path, ...] = (
    REPO_ROOT / "examples" / "results_cat",
    REPO_ROOT / "examples" / "results_gridsearch",
)

# Same-module name (``__init__.py``) is not a script.
EXCLUDED_SCRIPT_FILES = {"__init__.py"}

# Files inside ``datasets/`` that aren't real dataset subdirs (e.g.
# ``custom.csv`` lives directly in ``datasets/``).
EXCLUDED_DATASET_NAMES = {"custom.csv"}

# Per-script mapping: ``module_file -> (wip_subdir_relative_to_run_dir,
#                                       output_file_suffix)``.
#
# * ``wip_subdir`` is what the existing standalone CLI uses when it calls
#   its own ``_process_subdir``; we mirror that so the on-disk layout we
#   produce matches what the standalone CLI produces for any single
#   dataset (handy when diffing). Pass ``""`` for "write directly into
#   ``wip/``".
# * ``output_suffix`` is appended to the run-id to form the rendered
#   filename; again, mirrors the standalone CLIs.
#
# Only the *grouping* scripts are registered. ``categorical_view_conversion.py``
# is intentionally excluded -- it's an ablation step (the one-hot rewrite
# used while hunting for the best visual examples) and not a grouping pass.
SCRIPT_REGISTRY: Dict[str, Dict[str, str]] = {
    "cat_grouping.py": {
        "wip_subdir": "",
        "output_suffix": "_DPG_grouped",
        "label": "grouped",
    },
    "cat_grouping_split.py": {
        "wip_subdir": "grouping_split",
        "output_suffix": "_DPG_split_grouped",
        "label": "split",
    },
    "cat_grouping_split_conjunction.py": {
        "wip_subdir": "grouping_split_conjunction",
        "output_suffix": "_DPG_split_grouped_conjunction",
        "label": "conjunction",
    },
}


# --- Coloured output ------------------------------------------------------

class C:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"


# --- Domain types ---------------------------------------------------------


@dataclass
class DatasetMatch:
    """A single (dataset, results_root, matching_dpg_subdirs) bundle."""

    dataset: str
    results_root: pathlib.Path
    matching_subdirs: List[pathlib.Path] = field(default_factory=list)


@dataclass
class Experiment:
    dataset: str
    script_name: str
    run_id: str
    wip_subdir: str
    output_suffix: str
    output_dir: pathlib.Path  # absolute path of the run-specific output dir
    matches: List[DatasetMatch] = field(default_factory=list)

    @property
    def key(self) -> str:
        return f"{self.dataset}/{self.script_name}"

    @property
    def status_dir(self) -> pathlib.Path:
        return pathlib.Path(
            self.output_dir
        ).parent.parent / ".experiment_status"  # shared with run_all


# --- Discovery helpers ----------------------------------------------------


def discover_datasets(datasets_dir: pathlib.Path) -> List[str]:
    """Return the sorted list of dataset subdir names under ``datasets_dir``."""
    if not datasets_dir.exists():
        return []
    names = [
        entry.name
        for entry in datasets_dir.iterdir()
        if entry.is_dir() and entry.name not in EXCLUDED_DATASET_NAMES
    ]
    return sorted(names)


def discover_scripts(grouping_dir: pathlib.Path) -> List[str]:
    """Return the sorted list of grouping-script filenames."""
    if not grouping_dir.exists():
        return []
    return sorted(
        p.name
        for p in grouping_dir.iterdir()
        if p.is_file()
        and p.suffix == ".py"
        and p.name not in EXCLUDED_SCRIPT_FILES
        and p.name in SCRIPT_REGISTRY  # only run scripts we know about
    )


def _short_dataset_tag(csv_filename: str) -> str:
    """Mirror the ``ds=<tag>`` tag the gridsearch scripts use.

    ``gridsearch_dpg.py`` / ``gridsearch_dpg_catsim.py`` strip the
    ``toy_`` prefix and the ``.csv`` extension when building the
    ``ds=<tag>_pv=...`` subdir name, so:

      * ``toy_chain_intent_a_ab.csv`` -> ``chain_intent_a_ab``
      * ``toy_cat1_gender.csv``       -> ``cat1_gender``
      * ``iris.csv``                  -> ``iris``
    """
    tag = csv_filename
    if tag.lower().endswith(".csv"):
        tag = tag[:-4]
    if tag.startswith("toy_"):
        tag = tag[len("toy_"):]
    return tag


def discover_dataset_tags(dataset_dir: pathlib.Path) -> List[str]:
    """Return the sorted set of ``ds=<tag>``-compatible tags derivable from
    the CSV files living directly inside ``dataset_dir``.

    An empty list means the dataset subdir either has no CSVs or they
    don't follow the ``toy_*.csv`` convention; either way no matching
    DPG results exist and the caller should skip it.
    """
    if not dataset_dir.exists() or not dataset_dir.is_dir():
        return []
    tags: List[str] = []
    for entry in dataset_dir.iterdir():
        if not entry.is_file():
            continue
        if not entry.name.lower().endswith(".csv"):
            continue
        tag = _short_dataset_tag(entry.name)
        if tag:
            tags.append(tag)
    return sorted(set(tags))


def discover_matching_subdirs(
    dataset_tags: Iterable[str], results_root: pathlib.Path
) -> List[pathlib.Path]:
    """Return subdirs of ``results_root`` whose ``ds=<tag>`` prefix matches
    any of ``dataset_tags``.

    The DPG subdir naming convention is ``ds=<tag>_pv=..._dt=..._ct=...``
    (see ``gridsearch_dpg.py`` / ``gridsearch_dpg_catsim.py``). We only
    accept the structured ``ds=`` form so a CSV like ``iris.csv``
    doesn't accidentally match an unrelated subdir whose name happens to
    contain the substring ``iris``.
    """
    if not results_root.exists():
        return []
    tag_set = set(dataset_tags)
    matches: List[pathlib.Path] = []
    for entry in sorted(results_root.iterdir()):
        if not entry.is_dir():
            continue
        name = entry.name
        if not name.startswith("ds="):
            continue
        # Strip ``ds=`` prefix, then peel off the trailing ``_pv=...`` (if
        # present) so we can do an exact tag match. Anything else (e.g.
        # ``ds=iris`` without a ``_pv=`` suffix) is matched on the bare
        # ``ds=iris`` form.
        body = name[len("ds="):]
        if "_pv=" in body:
            tag = body.split("_pv=", 1)[0]
        else:
            tag = body
        if tag in tag_set:
            matches.append(entry)
    return matches


def build_dataset_matches(
    datasets: Iterable[str],
    datasets_dir: pathlib.Path,
    results_roots: Iterable[pathlib.Path],
) -> List[DatasetMatch]:
    """Pair each dataset with the (first) results root that has any subdirs
    matching one of the dataset's CSV-derived tags, plus those subdirs.

    Datasets whose subdir contains no matching CSVs (or no matching DPG
    results exist for any of them) are dropped from the returned list --
    the caller skips them with a "no matching DPG subdirs" message rather
    than treating "no work to do" as a hard failure.
    """
    matches: List[DatasetMatch] = []
    for ds in datasets:
        tags = discover_dataset_tags(datasets_dir / ds)
        if not tags:
            continue
        for root in results_roots:
            subs = discover_matching_subdirs(tags, root)
            if subs:
                matches.append(
                    DatasetMatch(
                        dataset=ds, results_root=root, matching_subdirs=subs
                    )
                )
                break
    return matches


# --- WandB helpers --------------------------------------------------------

try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:  # pragma: no cover - wandb is a hard dep of the project
    WANDB_AVAILABLE = False
    wandb = None  # type: ignore[assignment]


def wandb_init_for_experiment(
    experiment: Experiment,
    offline: bool,
    extra_config: Optional[Dict[str, object]] = None,
):
    """Initialise a wandb run for one (dataset, script) experiment.

    Returns the ``wandb.Run`` object on success, ``None`` if wandb is
    unavailable. The run is configured to land in the
    ``dpg-categorical`` project under the same entity as the
    counterfactual ``run_all_experiments.py`` script, grouped under
    ``run_all_categorical`` so all runs from this script appear together
    in the wandb UI.
    """
    if not WANDB_AVAILABLE:
        return None
    cfg = {
        "dataset": experiment.dataset,
        "grouping_script": experiment.script_name,
        "output_suffix": experiment.output_suffix,
        "n_matching_subdirs": sum(
            len(m.matching_subdirs) for m in experiment.matches
        ),
        "results_roots": [
            str(m.results_root) for m in experiment.matches
        ],
    }
    if extra_config:
        cfg.update(extra_config)
    mode = "offline" if offline else "online"
    return wandb.init(
        entity=WANDB_ENTITY,
        project=WANDB_PROJECT,
        group=WANDB_GROUP,
        job_type="categorical_grouping",
        name=experiment.key.replace(os.sep, "__"),
        config=cfg,
        mode=mode,
    )


def wandb_log_artifacts(run, artifacts: List[pathlib.Path]) -> None:
    """Log each path as a wandb artifact and image (PNG only).

    PNGs are also pushed via ``wandb.Image`` so they show up in the run's
    media panel. JSON files become ``<key>_structure`` artifacts so they
    can be re-fetched later.
    """
    if run is None:
        return
    import mimetypes

    for path in artifacts:
        if not path.exists():
            continue
        mime, _ = mimetypes.guess_type(str(path))
        ext = path.suffix.lower()
        artifact_name = f"{path.parent.name}__{path.stem}"
        if ext == ".png":
            try:
                run.log({f"images/{path.parent.name}/{path.stem}": wandb.Image(str(path))})
            except Exception as exc:  # pragma: no cover - best effort
                print(f"  {C.DIM}[wandb] could not log image {path.name}: {exc}{C.RESET}")
            try:
                art = wandb.Artifact(artifact_name, type="dpg_image")
                art.add_file(str(path))
                run.log_artifact(art)
            except Exception as exc:  # pragma: no cover - best effort
                print(f"  {C.DIM}[wandb] could not log artifact {path.name}: {exc}{C.RESET}")
        elif ext == ".json":
            try:
                art = wandb.Artifact(artifact_name, type="dpg_structure")
                art.add_file(str(path))
                run.log_artifact(art)
            except Exception as exc:  # pragma: no cover - best effort
                print(f"  {C.DIM}[wandb] could not log artifact {path.name}: {exc}{C.RESET}")
        else:
            # Anything else (txt, csv, ...): log as a generic file artifact.
            try:
                art = wandb.Artifact(artifact_name, type="dpg_aux")
                art.add_file(str(path))
                run.log_artifact(art)
            except Exception:
                pass


# --- Per-experiment execution --------------------------------------------


def _load_visualization_config(config_path: pathlib.Path) -> dict:
    """Mirror of the per-script ``_load_visualization_config`` helper."""
    import yaml

    with open(config_path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def run_one_experiment(
    experiment: Experiment,
    visualization_config: dict,
    offline: bool,
    dry_run: bool = False,
) -> Tuple[bool, List[pathlib.Path]]:
    """Run one ``(dataset, grouping_script)`` experiment in-process.

    Invokes the grouping script's ``_process_subdir`` for every matching
    DPG subdir, redirects ``wip_dir`` to the unversioned output tree, and
    returns ``(success, produced_artifact_paths)``.

    Returns ``(False, [])`` for the "no matching DPG subdirs" case so the
    caller can record a skip rather than a hard error.
    """
    if not experiment.matches:
        return False, []

    if dry_run:
        # Just announce what would happen and return success-ish.
        total = sum(len(m.matching_subdirs) for m in experiment.matches)
        print(
            f"  {C.DIM}[dry-run] would process {total} DPG subdir(s) "
            f"for {experiment.dataset}{C.RESET}"
        )
        return True, []

    # Each ``_process_subdir`` writes inside the ``wip_dir`` we pass;
    # naming follows the per-script convention (directly into ``wip/``
    # for cat_view / cat_grouping, into ``wip/<wip_subdir>/`` for the
    # two split variants).
    script_module = importlib.import_module(
        f"categorical.{experiment.script_name[:-3]}"
    )
    process_subdir = script_module._process_subdir

    # All matching subdirs from all selected results roots share the same
    # ``wip_dir`` (one experiment = one dataset x one grouping script).
    wip_dir = (
        experiment.output_dir / "wip" / experiment.wip_subdir
        if experiment.wip_subdir
        else experiment.output_dir / "wip"
    )
    wip_dir.mkdir(parents=True, exist_ok=True)

    produced: List[pathlib.Path] = []
    failed = 0
    for match in experiment.matches:
        for subdir in match.matching_subdirs:
            try:
                out_png = process_subdir(str(subdir), str(wip_dir), visualization_config)
            except Exception as exc:  # noqa: BLE001 - report and continue
                print(
                    f"  {C.RED}[err]{C.RESET} {subdir.name}: {exc}"
                )
                failed += 1
                continue
            if out_png is None:
                # Helper explicitly said "skip" (missing structure file, etc.).
                continue
            png_path = pathlib.Path(out_png)
            produced.append(png_path)
            # The rewritten/merged structure JSON lives next to the PNG
            # for ``cat_grouping*`` scripts; pick it up so wandb sees it.
            for json_candidate in wip_dir.glob(f"{subdir.name}*structure*.json"):
                produced.append(json_candidate)

    if failed and not produced:
        return False, produced
    return True, produced


def build_experiments(
    dataset_matches: List[DatasetMatch],
    scripts: List[str],
    output_root: pathlib.Path,
    skip_existing: bool,
) -> List[Experiment]:
    """Build the experiment list (and skip already-completed ones if asked)."""
    experiments: List[Experiment] = []
    for match in dataset_matches:
        for script_name in scripts:
            meta = SCRIPT_REGISTRY[script_name]
            output_dir = output_root / match.dataset / meta["label"]
            exp = Experiment(
                dataset=match.dataset,
                script_name=script_name,
                run_id=f"{match.dataset}__{meta['label']}",
                wip_subdir=meta["wip_subdir"],
                output_suffix=meta["output_suffix"],
                output_dir=output_dir,
                matches=[match],
            )

            if skip_existing:
                status, _ = get_experiment_status(
                    match.dataset,
                    f"cat__{meta['label']}",
                    pathlib.Path(output_root),
                )
                if status == PersistentStatus.FINISHED:
                    exp_key = f"{exp.dataset}/{exp.script_name}"
                    print(
                        f"  {C.YELLOW}[skip]{C.RESET} {exp_key} "
                        f"(already finished; pass without --skip-existing to re-run)"
                    )
                    continue

            experiments.append(exp)
    return experiments


# --- Top-level orchestration ---------------------------------------------


def generate_report(
    experiments: List[Experiment],
    results: Dict[str, List[Experiment]],
    total_elapsed: float,
    output_path: pathlib.Path,
) -> None:
    """Write a markdown summary report mirroring ``run_all_experiments.py``."""
    lines: List[str] = []
    lines.append("# Categorical Features Run Report")
    lines.append("")
    lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"**Total Duration:** {total_elapsed:.1f}s ({total_elapsed/60:.1f} minutes)")
    lines.append(f"**W&B project:** `{WANDB_PROJECT}`")
    lines.append(f"**W&B entity:**  `{WANDB_ENTITY}`")
    lines.append(f"**W&B group:**   `{WANDB_GROUP}`")
    lines.append("")

    lines.append("## Summary")
    lines.append("")
    lines.append(f"- OK  **Successful:** {len(results['success'])}")
    lines.append(f"- ERR **Failed:**     {len(results['failed'])}")
    lines.append(f"- SKIP **Skipped:**   {len(results['skipped'])}")
    lines.append("")

    if results["success"]:
        lines.append("## Successful Runs")
        lines.append("")
        for exp in sorted(results["success"], key=lambda e: e.key):
            lines.append(f"- `{exp.key}` -> `{exp.output_dir}`")
        lines.append("")
    if results["failed"]:
        lines.append("## Failed Runs")
        lines.append("")
        for exp in sorted(results["failed"], key=lambda e: e.key):
            lines.append(f"- `{exp.key}`")
        lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: Optional[str[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--datasets-dir",
        type=pathlib.Path,
        default=REPO_ROOT / "datasets",
        help="Root containing one subdir per dataset (default: DPG/datasets).",
    )
    parser.add_argument(
        "--datasets", nargs="+", default=None,
        help="Restrict to a subset of dataset subdir names (default: all).",
    )
    parser.add_argument(
        "--grouping-dir",
        type=pathlib.Path,
        default=GROUPING_DIR,
        help="Directory containing the grouping scripts.",
    )
    parser.add_argument(
        "--scripts", nargs="+", default=None,
        help="Restrict to a subset of grouping scripts (default: all known).",
    )
    parser.add_argument(
        "--results-roots",
        type=pathlib.Path,
        nargs="+",
        default=list(DEFAULT_RESULTS_ROOTS),
        help="One or more directories of DPG subdirs to mine for inputs.",
    )
    parser.add_argument(
        "--output-root",
        type=pathlib.Path,
        default=None,
        help=(
            "Unversioned root for rendered PNGs + JSONs. "
            f"Defaults to a timestamped subdir under {DEFAULT_OUTPUT_ROOT}."
        ),
    )
    parser.add_argument(
        "--skip-existing", action="store_true",
        help="Skip (dataset, script) combos whose status file is FINISHED.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print what would run; don't actually invoke grouping scripts.",
    )
    parser.add_argument(
        "--offline", action="store_true",
        help="Run wandb in offline mode (sync later with `wandb sync`).",
    )
    parser.add_argument(
        "--parallel", type=int, default=1, metavar="N",
        help="Run up to N experiments concurrently (default: sequential).",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Cap the number of experiments (after dataset/script filtering).",
    )
    parser.add_argument(
        "--config",
        type=pathlib.Path,
        default=REPO_ROOT / "config.yaml",
        help="DPG visualisation config YAML (passed to each grouping script).",
    )
    parser.add_argument(
        "--no-wandb", action="store_true",
        help="Disable wandb entirely; just produce the local outputs.",
    )
    args = parser.parse_args(argv)

    # --- Discovery ------------------------------------------------------
    datasets = args.datasets if args.datasets else discover_datasets(args.datasets_dir)
    if args.datasets and not datasets:
        print(f"ERROR: none of the requested datasets exist under {args.datasets_dir}")
        return 1
    scripts = args.scripts if args.scripts else discover_scripts(args.grouping_dir)
    if not scripts:
        print(f"ERROR: no known grouping scripts found under {args.grouping_dir}")
        return 1
    dataset_matches = build_dataset_matches(
        datasets, args.datasets_dir, args.results_roots
    )

    # --- Output root (timestamped if not provided) -----------------------
    if args.output_root is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_root = DEFAULT_OUTPUT_ROOT / ts
    else:
        output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)

    # --- Build the experiment list --------------------------------------
    experiments = build_experiments(
        dataset_matches, scripts, output_root, args.skip_existing
    )
    if args.limit is not None:
        experiments = experiments[: args.limit]

    if not experiments:
        print(
            f"{C.YELLOW}No experiments to run. "
            f"(datasets={len(datasets)}, matches={len(dataset_matches)}, "
            f"scripts={len(scripts)}){C.RESET}"
        )
        return 0

    # --- Banner ---------------------------------------------------------
    print("=" * 64)
    print(f"{C.BOLD}CATEGORICAL FEATURES BATCH RUNNER{C.RESET}")
    print("=" * 64)
    print(f"Start time:          {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Datasets discovered: {len(datasets)}")
    print(f"Datasets w/ matches: {len(dataset_matches)}")
    print(f"Grouping scripts:    {len(scripts)}")
    print(f"Experiments to run:  {len(experiments)}")
    print(f"Output root:         {output_root}")
    print(f"W&B project:         {WANDB_PROJECT}  (entity: {WANDB_ENTITY})")
    print(f"W&B group:           {WANDB_GROUP}")
    print(f"Parallel workers:    {args.parallel}")
    print(f"Dry run:             {args.dry_run}")
    print("=" * 64)

    if not args.config.exists():
        print(f"ERROR: visualisation config not found: {args.config}")
        return 1
    visualization_config = _load_visualization_config(args.config)

    # --- Run ------------------------------------------------------------
    total_start = time.time()
    results: Dict[str, List[Experiment]] = {
        "success": [], "failed": [], "skipped": [],
    }
    use_wandb = WANDB_AVAILABLE and not args.no_wandb
    if not use_wandb and not args.no_wandb:
        print(f"{C.YELLOW}wandb not installed; running without W&B logging.{C.RESET}")

    for idx, exp in enumerate(experiments, 1):
        n_subs = sum(len(m.matching_subdirs) for m in exp.matches)
        print(
            f"\n[{idx}/{len(experiments)}] {C.BOLD}{exp.key}{C.RESET} "
            f"({n_subs} DPG subdir(s))"
        )
        print("-" * 40)

        if not exp.matches:
            print(f"  {C.YELLOW}[skip]{C.RESET} no matching DPG subdirs for {exp.dataset}")
            results["skipped"].append(exp)
            continue

        # WandB: one run per experiment. Best-effort: if init fails, log
        # the artifact paths locally and keep going.
        run = None
        if use_wandb:
            try:
                run = wandb_init_for_experiment(exp, args.offline)
            except Exception as exc:  # noqa: BLE001
                print(f"  {C.YELLOW}[wandb] init failed: {exc}; continuing offline{C.RESET}")
                run = None

        try:
            ok, artifacts = run_one_experiment(
                exp, visualization_config, args.offline, dry_run=args.dry_run
            )
        finally:
            if run is not None:
                try:
                    wandb_log_artifacts(run, artifacts)
                except Exception as exc:  # noqa: BLE001
                    print(f"  {C.YELLOW}[wandb] log failed: {exc}{C.RESET}")
                try:
                    run.finish()
                except Exception:
                    pass

        if args.dry_run:
            results["success"].append(exp)
            continue
        if ok:
            print(
                f"  {C.GREEN}OK{C.RESET} -> {exp.output_dir} "
                f"({len(artifacts)} artifact(s))"
            )
            results["success"].append(exp)
            try:
                write_status(
                    exp.dataset,
                    f"cat__{SCRIPT_REGISTRY[exp.script_name]['label']}",
                    PersistentStatus.FINISHED,
                    pathlib.Path(output_root),
                    pid=os.getpid(),
                    start_time=total_start,
                    end_time=time.time(),
                )
            except Exception:
                pass
        else:
            print(f"  {C.RED}FAIL{C.RESET} {exp.key}")
            results["failed"].append(exp)
            try:
                write_status(
                    exp.dataset,
                    f"cat__{SCRIPT_REGISTRY[exp.script_name]['label']}",
                    PersistentStatus.ERROR,
                    pathlib.Path(output_root),
                    pid=os.getpid(),
                    start_time=total_start,
                    end_time=time.time(),
                    error_message="see run output",
                )
            except Exception:
                pass

    total_elapsed = time.time() - total_start

    # --- Summary --------------------------------------------------------
    print("\n" + "=" * 64)
    print(f"{C.BOLD}SUMMARY{C.RESET}")
    print("=" * 64)
    print(f"End time:        {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total time:      {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")
    print(f"{C.GREEN}Successful:{C.RESET}      {len(results['success'])}")
    print(f"{C.RED}Failed:{C.RESET}          {len(results['failed'])}")
    print(f"{C.YELLOW}Skipped:{C.RESET}         {len(results['skipped'])}")

    if results["failed"]:
        print("\nFailed experiments:")
        for exp in results["failed"]:
            print(f"  - {exp.key}")

    if not args.dry_run:
        report_path = output_root / "report.md"
        generate_report(experiments, results, total_elapsed, report_path)
        print(f"\nReport saved to {report_path}")
        print(f"Artifacts under {output_root}")

    return 1 if results["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())