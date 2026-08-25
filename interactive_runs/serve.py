#!/usr/bin/env python
"""
Tiny launcher: ``python serve.py`` from anywhere inside the repo (with the
repo parent on PYTHONPATH, see the README) starts the Flask server on
http://localhost:5050.

For convenience this is what
``PYTHONPATH=/root/gitgud/temp python interactive_runs/serve.py`` runs.
"""

import os
import sys

# Add the parent of the repo to sys.path so the repo itself resolves as
# the ``DPG`` package (the existing grouping scripts use
# ``from DPG.examples.grouping_scripts.…``).
HERE = os.path.abspath(os.path.dirname(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(HERE, ".."))
PARENT_DIR = os.path.abspath(os.path.join(PROJECT_ROOT, ".."))
for p in (PARENT_DIR, PROJECT_ROOT, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

from interactive_runs.app import app  # noqa: E402


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5050"))
    app.run(host="0.0.0.0", port=port, debug=False)
