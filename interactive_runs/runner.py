"""
runner.py
=========

Single high-level entry point used by the interactive website:
``run_one(dataset, perc_var, decimal_threshold, community_threshold)`` runs
the *full* pipeline the existing shell of CLI scripts does, but condensed
into one in-process call so the Flask app can return PNG files to the
browser without spawning subprocesses.

Pipeline stages (mirrors what was previously executed manually):

    1. ``gridsearch_dpg_catsim.run_one`` for the requested single combo
       (trains RF, builds DPG, dumps structure JSON, node/edge CSVs,
       communities CSV/TXT, class boundaries, raw PNG + PDF,
       communities PNG + PDF, writes ``wip/<run>_DPG.png`` via
       categorical_view_conversion, etc.).
    2. ``categorical_view_conversion._process_subdir`` to materialise the
       one-hot-rewritten ``wip/<run>_DPG.png``.
    3. ``cat_grouping._process_subdir`` for ``wip/<run>_DPG_grouped.png``
       and ``wip/<run>_DPG_grouped_structure.json``.
    4. ``cat_grouping_split._process_subdir`` for
       ``wip/grouping_split/<run>_DPG_split_grouped.{png,json}``.
    5. ``cat_grouping_split_conjunction._process_subdir`` for
       ``wip/grouping_split_conjunction/<run>_DPG_split_grouped_conjunction.{png,json}``.

Run output is written under ``INTERACTIVE_RUNS_ROOT/<run_id>/`` (defaults
to ``<repo>/interactive_runs``). The directory name follows the same
``ds=<short>_pv=<pv>_dt=<dt>_ct=<ct>`` convention used under
``examples/results_cat/`` so downstream tooling can treat both uniformly.

The defaults baked into the constants below reproduce the curated
reference run that exists under ``examples/results_cat/``:
``toy_chain_education_abc.csv`` with ``perc_var=0.01``,
``decimal_threshold=2``, ``community_threshold=0.3``.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import traceback
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# --- Path setup ------------------------------------------------------------
# Make the project root + grouping scripts directory importable so we can
# call the existing pipeline functions directly (no subprocesses).
INTERACTIVE_DIR = os.path.abspath(os.path.dirname(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(INTERACTIVE_DIR, ".."))
GROUPING_DIR = os.path.join(PROJECT_ROOT, "categorical")
for p in (PROJECT_ROOT, GROUPING_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

# Allow running *either* as ``python runner.py`` from the interactive_runs/
# directory or as an imported module from the Flask app.
PARENT_DIR = os.path.abspath(os.path.join(PROJECT_ROOT, ".."))
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

# Pipeline constants (mirror gridsearch_dpg_catsim.py) ----------------------
RANDOM_STATE = 27
NUM_TREES = 10

DEFAULT_DATASET = "toy_chain_education_abc.csv"
DEFAULT_PERC_VAR = 0.01
DEFAULT_DECIMAL_THRESHOLD = 2
DEFAULT_COMMUNITY_THRESHOLD = 0.3

# Where the website-generated runs go (kept distinct from the curated
# examples/results_cat/ tree so exploratory regenerations don't clobber
# the canonical artifacts).
INTERACTIVE_RUNS_ROOT = os.path.join(INTERACTIVE_DIR, "runs")
DATA_DIR = os.path.join(PROJECT_ROOT, "datasets", "dummy_dataset")
BASE_CONFIG_PATH = os.path.join(PROJECT_ROOT, "config.yaml")


@dataclass
class RunRequest:
    """One combination of (dataset, perc_var, decimal_threshold, ct).

    Defaults reproduce the curated run in ``examples/results_cat/``.
    """

    dataset: str = DEFAULT_DATASET
    perc_var: float = DEFAULT_PERC_VAR
    decimal_threshold: int = DEFAULT_DECIMAL_THRESHOLD
    community_threshold: float = DEFAULT_COMMUNITY_THRESHOLD

    def short_dataset(self) -> str:
        """Stable ``ds=<recipe>`` tag (strips ``toy_`` prefix + ``.csv``)."""
        name = self.dataset
        if name.lower().endswith(".csv"):
            name = name[:-4]
        if name.startswith("toy_"):
            name = name[len("toy_"):]
        return name

    def run_id(self) -> str:
        """Directory name; matches the existing examples/results_cat scheme."""
        return (
            f"ds={self.short_dataset()}"
            f"_pv={self.perc_var}"
            f"_dt={self.decimal_threshold}"
            f"_ct={self.community_threshold}"
        )

    def out_dir(self) -> str:
        return os.path.join(INTERACTIVE_RUNS_ROOT, self.run_id())

    def config_path(self) -> str:
        return os.path.join(self.out_dir(), "config.yaml")

    def as_form_dict(self) -> Dict[str, Any]:
        """Default values for the HTML form (serialisable)."""
        return {
            "dataset": self.dataset,
            "perc_var": self.perc_var,
            "decimal_threshold": self.decimal_threshold,
            "community_threshold": self.community_threshold,
        }


@dataclass
class RunResult:
    """What the website needs to render the page after one pipeline run."""

    run_id: str
    out_dir: str
    artifacts: Dict[str, str] = field(default_factory=dict)
    """Map of short label (raw/grouped/split/conjunction) -> absolute PNG path."""

    metrics: Dict[str, Any] = field(default_factory=dict)
    """acc / f1 / n_nodes / n_edges / n_communities / n_features_used."""

    errors: List[str] = field(default_factory=list)
    """Pipeline-stage errors (does not abort the whole run)."""


# ---------------------------------------------------------------------------
# Pipeline (5 stages)
# ---------------------------------------------------------------------------


def _write_run_config(req: RunRequest) -> str:
    """Copy BASE_CONFIG_PATH into out_dir/config.yaml with overridden values."""
    import yaml

    with open(BASE_CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg.setdefault("dpg", {}).setdefault("default", {})
    cfg["dpg"]["default"]["perc_var"] = float(req.perc_var)
    cfg["dpg"]["default"]["decimal_threshold"] = int(req.decimal_threshold)
    cfg["dpg"]["default"]["n_jobs"] = 1
    out = req.config_path()
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return out


def _stage_gridsearch(req: RunRequest) -> Dict[str, Any]:
    """Stage 1: train RF, build the DPG, dump the top-level artifacts.

    Wraps ``gridsearch_dpg_catsim.run_one``. Returns the RunResult dataclass
    it produced so we can pull acc/f1/n_nodes/etc. without re-reading.

    Note: ``gridsearch_dpg_catsim.run_one`` writes to the module-level
    ``RESULTS_ROOT`` constant. We override that constant for the duration
    of the call so the artifacts land in our ``INTERACTIVE_RUNS_ROOT``
    instead of clobbering ``examples/results_cat/``.
    """
    import examples.gridsearch_dpg_catsim as gs_mod  # noqa: F401  (path-setup)

    # Force the gridsearch module to look at our directory tree.
    original_results_root = gs_mod.RESULTS_ROOT
    original_data_dir = gs_mod.DATA_DIR
    original_base_cfg = gs_mod.BASE_CONFIG_PATH
    try:
        gs_mod.RESULTS_ROOT = os.path.dirname(req.out_dir())
        gs_mod.DATA_DIR = DATA_DIR  # already correct, but keep explicit
        gs_mod.BASE_CONFIG_PATH = req.config_path()  # pre-written config
        rr = gs_mod.run_one(
            req.dataset,
            req.perc_var,
            req.decimal_threshold,
            req.community_threshold,
            feature_subset_rng=None,
        )
    finally:
        gs_mod.RESULTS_ROOT = original_results_root
        gs_mod.DATA_DIR = original_data_dir
        gs_mod.BASE_CONFIG_PATH = original_base_cfg

    return {
        "accuracy": rr.accuracy,
        "f1": rr.f1,
        "n_nodes": rr.n_nodes,
        "n_edges": rr.n_edges,
        "n_communities": rr.n_communities,
        "n_features_used": rr.n_features_used,
    }


def _stage_wip(
    script_name: str,
    fn_name: str,
    req: RunRequest,
    wip_subdir: str = "",
) -> Optional[str]:
    """Run a single ``wip``-producing ``_process_subdir`` helper.

    ``wip_subdir`` lets us route each script to its own per-stage subdir so
    outputs don't overlap. Empty (the default) = write directly into
    ``wip/`` (``categorical_view_conversion`` and ``cat_grouping`` both use
    that). ``"grouping_split"`` and ``"grouping_split_conjunction"`` match
    the names the standalone CLIs use in their ``main()``.

    Each grouping script lives next door under ``DPG/categorical/`` and
    does ``from .cat_grouping import …`` (relative import inside the
    ``categorical`` package). We've already added the project root to
    ``sys.path`` at module load time, so the import resolves as
    ``categorical.<script_name>``.
    """
    import importlib

    full_mod = f"categorical.{script_name}"
    mod = importlib.import_module(full_mod)
    fn = getattr(mod, fn_name)
    out_dir = req.out_dir()
    wip_dir = (
        os.path.join(out_dir, "wip", wip_subdir)
        if wip_subdir
        else os.path.join(out_dir, "wip")
    )
    fn(out_dir, wip_dir, _load_visualization_config())
    return wip_dir


def _load_visualization_config() -> dict:
    """Load the visualisation section of the project config (used by all
    three grouping scripts to style their DOT output)."""
    import yaml

    with open(BASE_CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_one(req: RunRequest) -> RunResult:
    """Run the full 5-stage pipeline for one combination.

    Never raises: collects per-stage errors into ``RunResult.errors`` and
    keeps going so the page can show whichever stages succeeded.
    """
    os.makedirs(req.out_dir(), exist_ok=True)
    result = RunResult(run_id=req.run_id(), out_dir=req.out_dir())

    try:
        # Stage 0: write per-run config.yaml up-front so stage 1 can use it.
        _write_run_config(req)
    except Exception as exc:  # noqa: BLE001
        result.errors.append(f"stage0_config: {exc}")

    # Stage 1: raw gridsearch (trains model, builds DPG, dumps everything
    # the gridsearch script writes, plus wip/_DPG.png via its internal
    # plot call -- see gridsearch_dpg_catsim.run_one).
    try:
        result.metrics = _stage_gridsearch(req)
    except Exception as exc:  # noqa: BLE001
        result.errors.append(f"stage1_gridsearch: {exc}\n{traceback.format_exc()}")

    # Stage 2: categorical_view_conversion (writes wip/<run>_DPG.png,
    # the one-hot-rewritten view). It already ran inside stage 1's
    # explainer.plot path, but we call it explicitly so it's idempotent
    # and matches what the grouping scripts expect on disk.
    try:
        _stage_wip("categorical_view_conversion", "_process_subdir", req)
    except Exception as exc:  # noqa: BLE001
        result.errors.append(f"stage2_cat_view: {exc}")

    # Stage 3: cat_grouping (wip/<run>_DPG_grouped.{png,json}).
    try:
        _stage_wip("cat_grouping", "_process_subdir", req)
    except Exception as exc:  # noqa: BLE001
        result.errors.append(f"stage3_grouping: {exc}")

    # Stage 4: cat_grouping_split
    # (wip/grouping_split/<run>_DPG_split_grouped.{png,json}).
    try:
        _stage_wip(
            "cat_grouping_split",
            "_process_subdir",
            req,
            wip_subdir="grouping_split",
        )
    except Exception as exc:  # noqa: BLE001
        result.errors.append(f"stage4_split: {exc}")

    # Stage 5: cat_grouping_split_conjunction
    # (wip/grouping_split_conjunction/<run>_DPG_split_grouped_conjunction.{png,json}).
    try:
        _stage_wip(
            "cat_grouping_split_conjunction",
            "_process_subdir",
            req,
            wip_subdir="grouping_split_conjunction",
        )
    except Exception as exc:  # noqa: BLE001
        result.errors.append(f"stage5_conjunction: {exc}")

    # Collect whatever PNGs the stages emitted so the page can render them.
    # Display labels (left) map directly to the (run_id-prefixed) file name
    # the underlying scripts write (right) so copy/pasting the path into
    # ``examples/results_cat/…`` produces the exact same artifact naming.
    candidates = {
        "raw": (
            f"{result.run_id}.png"
        ),
        "cat_view": (
            f"wip/{result.run_id}_DPG.png"
        ),
        "grouped": (
            f"wip/{result.run_id}_DPG_grouped.png"
        ),
        "split": (
            f"wip/grouping_split/{result.run_id}_DPG_split_grouped.png"
        ),
        "conjunction": (
            f"wip/grouping_split_conjunction/"
            f"{result.run_id}_DPG_split_grouped_conjunction.png"
        ),
    }
    for label, rel in candidates.items():
        abs_path = os.path.join(result.out_dir, rel)
        if os.path.isfile(abs_path):
            result.artifacts[label] = abs_path

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _discover_datasets() -> List[str]:
    """Return sorted list of ``toy_chain_*.csv`` files actually on disk."""
    if not os.path.isdir(DATA_DIR):
        return []
    files = sorted(
        f for f in os.listdir(DATA_DIR)
        if f.startswith("toy_chain_") and f.endswith(".csv")
    )
    return files


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        default=DEFAULT_DATASET,
        help="toy_chain_*.csv name (drop-down options in the website).",
    )
    parser.add_argument("--perc-var", type=float, default=DEFAULT_PERC_VAR)
    parser.add_argument(
        "--decimal-threshold", type=int, default=DEFAULT_DECIMAL_THRESHOLD
    )
    parser.add_argument(
        "--community-threshold",
        type=float,
        default=DEFAULT_COMMUNITY_THRESHOLD,
    )
    args = parser.parse_args(argv)

    req = RunRequest(
        dataset=args.dataset,
        perc_var=args.perc_var,
        decimal_threshold=args.decimal_threshold,
        community_threshold=args.community_threshold,
    )
    out = run_one(req)
    print(json.dumps(
        {
            "run_id": out.run_id,
            "artifacts": out.artifacts,
            "metrics": out.metrics,
            "errors": out.errors,
        },
        indent=2,
    ))
    return 0 if not out.errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
