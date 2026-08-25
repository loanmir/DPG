"""
app.py
======

Flask server for the ``interactive_runs`` website.

Routes
-----
``GET  /``                       -> HTML page
``GET  /api/datasets``           -> list of available toy_chain_*.csv files
``POST /api/run``                -> run the full 5-stage pipeline for one
                                   (dataset, perc_var, decimal_threshold,
                                   community_threshold) combo, return JSON
                                   with metrics + absolute PNG paths.
``GET  /api/image?run_id=<r>&label=<l>``
                                -> serve the PNG for one of the five
                                   pipeline outputs (``raw`` / ``cat_view`` /
                                   ``grouped`` / ``split`` / ``conjunction``).

PNG serving is keyed by ``(run_id, label)`` so the frontend never sees
absolute filesystem paths and gets a stable URL regardless of how many
``interactive_runs`` directories exist on disk.
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Any, Dict

from flask import (
    Flask,
    Response,
    abort,
    jsonify,
    render_template,
    request,
    send_file,
    url_for,
)

# --- Make the runner module importable ------------------------------------
INTERACTIVE_DIR = os.path.abspath(os.path.dirname(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(INTERACTIVE_DIR, ".."))
PARENT_DIR = os.path.abspath(os.path.join(PROJECT_ROOT, ".."))
for p in (PROJECT_ROOT, PARENT_DIR, INTERACTIVE_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from interactive_runs.runner import (  # noqa: E402
    INTERACTIVE_RUNS_ROOT,
    DEFAULT_COMMUNITY_THRESHOLD,
    DEFAULT_DATASET,
    DEFAULT_DECIMAL_THRESHOLD,
    DEFAULT_PERC_VAR,
    RunRequest,
    _discover_datasets,
    run_one,
)

# `interactive_runs` was inserted as a non-package module above; expose it
# as a package attribute so ``templates`` + ``static`` resolve.
INTERACTIVE_PACKAGE = "interactive_runs_pkg"
if INTERACTIVE_PACKAGE not in sys.modules:
    import importlib.util

    pkg_spec = importlib.util.spec_from_file_location(
        INTERACTIVE_PACKAGE,
        os.path.join(INTERACTIVE_DIR, "__init__.py"),
    )
    if pkg_spec is None:
        # Fall back to constructing an empty package so Flask template /
        # static folder discovery still works.
        import types

        pkg = types.ModuleType(INTERACTIVE_PACKAGE)
        pkg.__path__ = [INTERACTIVE_DIR]
        sys.modules[INTERACTIVE_PACKAGE] = pkg
    else:
        pkg = importlib.util.module_from_spec(pkg_spec)
        pkg.__path__ = [INTERACTIVE_DIR]
        sys.modules[INTERACTIVE_PACKAGE] = pkg


app = Flask(
    __name__,
    template_folder="templates",
    static_folder="static",
)


# --- Helpers --------------------------------------------------------------


def _parse_request_payload(payload: Dict[str, Any]) -> RunRequest:
    """Validate a form-encoded / JSON payload into a RunRequest.

    Missing keys fall back to the curated defaults so the page can be
    driven via a simple HTML form without JS.
    """
    dataset = str(payload.get("dataset", DEFAULT_DATASET))
    try:
        perc_var = float(payload.get("perc_var", DEFAULT_PERC_VAR))
    except (TypeError, ValueError):
        perc_var = DEFAULT_PERC_VAR
    try:
        decimal_threshold = int(payload.get("decimal_threshold", DEFAULT_DECIMAL_THRESHOLD))
    except (TypeError, ValueError):
        decimal_threshold = DEFAULT_DECIMAL_THRESHOLD
    try:
        community_threshold = float(payload.get("community_threshold", DEFAULT_COMMUNITY_THRESHOLD))
    except (TypeError, ValueError):
        community_threshold = DEFAULT_COMMUNITY_THRESHOLD

    return RunRequest(
        dataset=dataset,
        perc_var=perc_var,
        decimal_threshold=decimal_threshold,
        community_threshold=community_threshold,
    )


def _stage_labels() -> Dict[str, str]:
    """Display labels for each of the 5 pipeline outputs in stable order."""
    return {
        "raw": "Raw DPG (root)",
        "cat_view": "Categorical view (one-hot rewrite)",
        "grouped": "Sequential grouping",
        "split": "Split-then-merge grouping",
        "conjunction": "Split-then-merge w/ cross-feature AND",
    }


# --- Routes ---------------------------------------------------------------


@app.route("/")
def index() -> Any:
    return render_template(
        "index.html",
        defaults={
            "dataset": DEFAULT_DATASET,
            "perc_var": DEFAULT_PERC_VAR,
            "decimal_threshold": DEFAULT_DECIMAL_THRESHOLD,
            "community_threshold": DEFAULT_COMMUNITY_THRESHOLD,
        },
        datasets=_discover_datasets(),
        stage_labels=_stage_labels(),
    )


@app.route("/api/datasets")
def api_datasets() -> Any:
    return jsonify({"datasets": _discover_datasets()})


@app.route("/api/run", methods=["POST"])
def api_run() -> Any:
    # Accept both JSON and form-encoded payloads (the page uses form-encoded
    # so browser submissions work without JS).
    payload: Dict[str, Any]
    if request.is_json:
        payload = request.get_json(silent=True) or {}
    else:
        payload = request.form.to_dict()

    req = _parse_request_payload(payload)

    started = time.time()
    result = run_one(req)
    elapsed = time.time() - started

    # Build a path-relative URL for each artifact so the client never
    # deals with absolute filesystem paths.
    image_urls: Dict[str, str] = {}
    for label in _stage_labels().keys():
        image_urls[label] = (
            url_for(
                "api_image",
                run_id=result.run_id,
                label=label,
                ts=int(time.time() * 1000),  # cache-buster
            )
        )

    return jsonify(
        {
            "run_id": result.run_id,
            "out_dir": result.out_dir,
            "request": {
                "dataset": req.dataset,
                "perc_var": req.perc_var,
                "decimal_threshold": req.decimal_threshold,
                "community_threshold": req.community_threshold,
            },
            "metrics": result.metrics,
            "errors": result.errors,
            "image_urls": image_urls,
            "elapsed_seconds": round(elapsed, 3),
        }
    )


@app.route("/api/image")
def api_image() -> Any:
    """Serve one PNG from a completed run. Args: ``run_id``, ``label``."""
    run_id = request.args.get("run_id", "")
    label = request.args.get("label", "")
    if not run_id or not label or label not in _stage_labels():
        abort(404)

    run_dir = os.path.join(INTERACTIVE_RUNS_ROOT, run_id)
    # The PNG paths in the runner's candidate map use the run_id prefix
    # and ``_`` suffixes; mirror that map here so the two stay in sync.
    relative = {
        "raw": f"{run_id}.png",
        "cat_view": f"wip/{run_id}_DPG.png",
        "grouped": f"wip/{run_id}_DPG_grouped.png",
        "split": f"wip/grouping_split/{run_id}_DPG_split_grouped.png",
        "conjunction": (
            f"wip/grouping_split_conjunction/{run_id}_DPG_split_grouped_conjunction.png"
        ),
    }[label]
    abs_path = os.path.join(run_dir, relative)
    if not os.path.isfile(abs_path):
        abort(404)
    return send_file(abs_path, mimetype="image/png")


if __name__ == "__main__":
    # ``debug=False`` because the pipeline imports carry module state that
    # doesn't survive an autoreload cleanly; the user just refreshes the
    # page after an edit.
    app.run(host="0.0.0.0", port=5050, debug=False)
