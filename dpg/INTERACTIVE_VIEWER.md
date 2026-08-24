# DPG interactive viewer

[`dpg/interactive_visualizer.py`](interactive_visualizer.py) is a small,
dependency-free launcher that turns any of the structure JSONs produced by
the gridsearch + grouping pipeline into a self-contained HTML page you can
open in your browser.

## What you get

A page with:

* The **full graph** rendered by default (drag / zoom / hover tooltips).
* **Edge-weight slider** — hides edges whose weight is below the threshold.
* **Node-weight slider** — hides nodes whose *total* edge weight
  (sum of in + out) is below the threshold.
* **Label filter** — text box. Keeps only nodes whose label contains the
  substring (case-insensitive). Empty = no filter.
* **Always-keep `Class *` nodes** checkbox — class destinations are pinned
  visible regardless of the node-weight threshold. Toggle off to filter
  them too.
* **Reset filters** button — restores defaults (full graph).
* **Refit view** button — re-centres the camera.
* **Variant switcher** — buttons for `Raw`, `Grouped (strict)`,
  `Split-grouped`. The launcher auto-detects which siblings exist on disk
  and disables the others. Variant switching requires the page to be
  served over HTTP (or a browser that allows `file://` fetches); see
  below.
* **Physics iterations slider** — controls vis-network's stabilisation
  budget for the force-atlas layout.

The vis-network UMD bundle is **vendored** at
[`dpg/static/vis-network.min.js`](static/vis-network.min.js) and the
launcher copies it next to every generated HTML. The page works offline,
under `file://`, and behind firewalls that block public CDNs.

## Quick start

### From the CLI

```bash
# Open the first *DPG_grouped_structure.json under examples/results_cat/.
.venv/bin/python -m dpg.interactive_visualizer

# Or point it at a specific JSON:
.venv/bin/python -m dpg.interactive_visualizer \
    examples/results_cat/ds=chain_intent_with_age_pv=0.075_dt=3_ct=0.3/wip/ds=chain_intent_with_age_pv=0.075_dt=3_ct=0.3_DPG_grouped_structure.json
```

The script writes `<run_id>_interactive.html` next to the JSON and opens
it in your default browser. If your browser blocks `file://` (some
setups do), serve the directory over HTTP and open it from there:

```bash
.venv/bin/python -m http.server 8765 \
    --directory examples/results_cat/ds=chain_intent_with_age_pv=0.075_dt=3_ct=0.3/wip
# then visit http://localhost:8765/<run_id>_interactive.html
```

### From Python

```python
from dpg.interactive_visualizer import open_interactive_view
open_interactive_view(
    "examples/results_cat/ds=.../wip/ds=..._DPG_grouped_structure.json"
)
```

## What JSON shapes does it accept?

Anything that follows the DPG `*_structure.json` convention:

* `examples/results_cat/<run>/<run>_dpg_structure.json` — raw DPG (one-hot
  encoded categorical predicates).
* `examples/results_cat/<run>/wip/<run>_DPG_grouped_structure.json` —
  strict sequential grouping.
* `examples/results_cat/<run>/wip/grouping_split/<run>_DPG_split_grouped_structure.json` —
  aggressive (split-then-merge) grouping.

The schema it expects (everything else is ignored):

```json
{
  "nodes": [{"id": "...", "label": "..."}],
  "graph": {
    "nodes": [{"id": "..."}],
    "edges": [{"source": "...", "target": "...", "weight": 12.5}]
  },
  "run_id": "...",
  "feature_names": ["..."],
  "target_names": ["..."]
}
```

Pseudo edge-label entries (those whose `id` contains `"->"`) are dropped,
and any node that ends up with zero incident edges is dropped too (some
grouping outputs leave behind a stranded id).

## Variant switching

The variant switcher fetches each variant via `fetch()` to its
`file://` URI. Browsers vary on whether they allow that:

* **Chrome / Edge** — usually blocked by default for `file://` URLs.
* **Firefox** — sometimes allowed.
* **Any browser over HTTP** — always works.

If variant switching doesn't work for you, simply open each variant's
`_interactive.html` directly (one per variant). The launcher writes one
HTML per run; if you want one per variant you can call
`open_interactive_view(path)` once per JSON.