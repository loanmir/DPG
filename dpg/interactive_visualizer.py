"""
dpg.interactive_visualizer
==========================

A small, dependency-free launcher that turns one of the DPG structure JSONs
produced by :mod:`examples.cat_grouping` (or
:mod:`examples.cat_grouping_split`, or the raw gridsearch ``*_dpg_structure.json``)
into an interactive HTML view.

What you get in the browser
---------------------------

* **Default view** — the *full* graph is rendered (no nodes or edges hidden).
* **Node-weight slider** — hides every node whose *total* edge weight (sum
  of incoming + outgoing edges) is below the threshold. Class nodes are
  always kept visible because they're the destination of every meaningful
  path; toggleable via a checkbox.
* **Edge-weight slider** — hides every edge whose weight is below the
  threshold.
* **Feature filter** — text box. Only nodes whose label contains the
  substring (case-insensitive) stay visible. Empty box = no filter.
* **Variant switcher** — buttons that load one of three JSON flavours
  produced alongside the run:

    * ``*_dpg_structure.json``           — raw DPG (no rewrite, no grouping)
    * ``*_DPG_grouped_structure.json``   — strict sequential grouping
    * ``*_DPG_split_grouped_structure.json`` — aggressive (split-then-merge)

  The launcher auto-detects which ones exist next to your chosen JSON and
  enables the corresponding button.
* **Hover tooltip** — full label, in/out degree, total weight.
* **Drag / zoom / pan** — provided by the underlying graph engine.

The visualizer is pure HTML + JS. It loads ``vis-network`` from a public
CDN, so as long as the user has internet on first open it works offline
thereafter (browser cache). No Python server is required.

Quick start
-----------

From a Python session::

    from dpg.interactive_visualizer import open_interactive_view
    open_interactive_view(
        "examples/results_cat/ds=.../wip/ds=..._DPG_grouped_structure.json"
    )

From the CLI (opens the same browser tab)::

    python -m dpg.interactive_visualizer \\
        examples/results_cat/ds=chain_intent_with_age_pv=0.075_dt=3_ct=0.3/wip/ds=chain_intent_with_age_pv=0.075_dt=3_ct=0.3_DPG_grouped_structure.json

If you don't pass a JSON, the launcher walks ``examples/results_cat/`` and
opens the *first* ``*_DPG_grouped_structure.json`` it finds.

Implementation
--------------

The HTML/JS payload is embedded as ``_HTML_TEMPLATE`` so the whole feature
ships in a single module. ``build_html_payload`` injects the JSON (and a
small companion manifest pointing at the other available variants) into the
template, ``write_payload`` writes it to disk and ``open_payload`` shells
out to the OS to open the resulting file in the default browser.
"""

from __future__ import annotations

import json
import os
import sys
import webbrowser
from pathlib import Path
from typing import Iterable, List, Optional, Sequence


# ---------------------------------------------------------------------------
# JSON loading / normalisation
# ---------------------------------------------------------------------------

# Node-id pseudo-entries the grouping scripts use to carry edge labels. They
# show up in ``structure["nodes"]`` (so the original DPG can carry "what
# predicate split is this edge labelled with?") but they never appear in
# ``structure["graph"]`` (the actual node-link graph). The visualizer should
# ignore them so they don't render as disconnected, empty-label boxes.
_EDGE_LABEL_PSEUDO_SEP = "->"


def _is_pseudo_edge_label_node(node_id: str) -> bool:
    return _EDGE_LABEL_PSEUDO_SEP in str(node_id)


def _coerce_weight(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _normalise_payload(structure: dict) -> dict:
    """Turn a DPG ``*_structure.json`` into the {nodes, edges, meta} shape
    the HTML payload expects.

    Returns
    -------
    dict
        ``nodes`` — list of ``{"id": str, "label": str, "is_class": bool}``.
        ``edges`` — list of ``{"source": str, "target": str, "weight": float}``.
        ``meta``  — small dict with stats (counts, weight bounds) used by
                    the UI to set slider defaults and to display a header.
    """
    raw_nodes: Sequence[dict] = structure.get("nodes", []) or []
    graph = structure.get("graph", {}) or {}

    # ---- nodes ----------------------------------------------------------
    # The actual node-link graph nodes are the authoritative list (they are
    # what's wired up by the edges below). The top-level ``nodes`` array
    # adds labels for them. We merge the two, dropping pseudo edge-label
    # entries.
    label_map = {
        str(n.get("id")): (n.get("label") or "")
        for n in raw_nodes
        if not _is_pseudo_edge_label_node(str(n.get("id")))
    }

    graph_node_ids = {str(n.get("id")) for n in graph.get("nodes", []) or []}

    nodes: List[dict] = []
    seen_ids = set()
    for nid in sorted(graph_node_ids | set(label_map.keys())):
        if _is_pseudo_edge_label_node(nid):
            continue
        label = label_map.get(nid, "")
        nodes.append(
            {
                "id": nid,
                "label": label or nid,
                "is_class": label.strip().lower().startswith("class"),
            }
        )
        seen_ids.add(nid)

    # ---- edges ----------------------------------------------------------
    raw_edges = graph.get("edges") or graph.get("links") or []
    edges: List[dict] = []
    for link in raw_edges:
        src = str(link.get("source"))
        dst = str(link.get("target"))
        # Some serializers include self-loops or reference pseudo edge-label
        # ids; both are meaningless for rendering.
        if _is_pseudo_edge_label_node(src) or _is_pseudo_edge_label_node(dst):
            continue
        edges.append(
            {
                "source": src,
                "target": dst,
                "weight": _coerce_weight(link.get("weight", 0.0)),
            }
        )

    # ---- drop stranded nodes -------------------------------------------
    # Some grouping outputs list a node in ``graph.nodes`` even though it
    # no longer participates in any edge (an absorbed pseudo node that
    # didn't get cleaned out). Rendering those as floating ellipses with
    # no purpose is confusing, so drop any node whose in/out edge set is
    # empty. Class nodes are always kept because they're the destination
    # of every meaningful path.
    incident_ids = set()
    for e in edges:
        incident_ids.add(e["source"])
        incident_ids.add(e["target"])
    nodes = [n for n in nodes if n["is_class"] or n["id"] in incident_ids]

    # ---- meta -----------------------------------------------------------
    weights = [e["weight"] for e in edges]
    node_weight_totals = {n["id"]: 0.0 for n in nodes}
    for e in edges:
        node_weight_totals[e["source"]] = (
            node_weight_totals.get(e["source"], 0.0) + e["weight"]
        )
        node_weight_totals[e["target"]] = (
            node_weight_totals.get(e["target"], 0.0) + e["weight"]
        )

    meta = {
        "n_nodes": len(nodes),
        "n_edges": len(edges),
        "min_edge_weight": min(weights) if weights else 0.0,
        "max_edge_weight": max(weights) if weights else 0.0,
        "min_node_weight": min(node_weight_totals.values())
        if node_weight_totals
        else 0.0,
        "max_node_weight": max(node_weight_totals.values())
        if node_weight_totals
        else 0.0,
        "run_id": structure.get("run_id", ""),
        "feature_names": structure.get("feature_names", []),
        "target_names": structure.get("target_names", []),
    }

    return {"nodes": nodes, "edges": edges, "meta": meta}


# ---------------------------------------------------------------------------
# Variant discovery
# ---------------------------------------------------------------------------

# The gridsearch output root contains ``<run_subdir>/<files>`` and the
# grouping scripts write ``*_DPG_grouped_structure.json`` and
# ``*_DPG_split_grouped_structure.json`` into ``<run_subdir>/wip/`` (the
# split variant goes into ``wip/grouping_split/``). The raw
# ``*_dpg_structure.json`` lives directly in ``<run_subdir>``. Given any of
# these three, we can compute the other two paths if and only if they
# actually exist on disk.

_VARIANT_SUFFIXES = {
    "raw": "_dpg_structure.json",
    "grouped": "_DPG_grouped_structure.json",
    "split": "_DPG_split_grouped_structure.json",
}


def _strip_known_suffix(name: str) -> Optional[str]:
    """Return the run_id (everything before ``_dpg_structure.json`` /
    ``_DPG_grouped_structure.json`` / ``_DPG_split_grouped_structure.json``)
    if the name matches one of the three known shapes, else ``None``.
    """
    for suffix in _VARIANT_SUFFIXES.values():
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return None


def discover_variants(json_path: str | os.PathLike) -> dict:
    """Given one of the three structure JSONs, return a dict with the
    ``raw`` / ``grouped`` / ``split`` paths that *exist on disk*, keyed by
    variant name. Paths are returned as ``file://`` URIs so the HTML can
    ``fetch()`` them directly without needing a server.

    If only the input file exists (no siblings), the dict contains exactly
    one entry pointing back at that input file under every key it matches.
    """
    p = Path(json_path).resolve()
    if not p.exists():
        raise FileNotFoundError(f"JSON not found: {p}")

    name = p.name
    run_id = _strip_known_suffix(name)
    if run_id is None:
        # Unknown shape: expose as the only "raw" variant.
        return {"raw": p.as_uri()}

    parent = p.parent
    # The split variant lives one level deeper.
    wip_split = parent / "grouping_split" / f"{run_id}_DPG_split_grouped_structure.json"
    wip = parent / f"{run_id}_DPG_grouped_structure.json"
    raw = parent / f"{run_id}_dpg_structure.json"

    # If the input was inside wip/grouping_split/, ``parent`` is already that
    # subdir, so ``raw`` would resolve to
    # ``wip/grouping_split/<run>_dpg_structure.json`` which doesn't exist.
    # In that case we have to walk one level up.
    if not raw.exists() and parent.name == "grouping_split":
        grand = parent.parent
        raw = grand / f"{run_id}_dpg_structure.json"
        wip = grand / f"{run_id}_DPG_grouped_structure.json"
        wip_split = grand / "grouping_split" / f"{run_id}_DPG_split_grouped_structure.json"

    found: dict = {}
    if raw.exists():
        found["raw"] = raw.as_uri()
    if wip.exists():
        found["grouped"] = wip.as_uri()
    if wip_split.exists():
        found["split"] = wip_split.as_uri()
    # Fallback: at minimum, the input itself must be reachable.
    if not found:
        found["input"] = p.as_uri()
    return found


# ---------------------------------------------------------------------------
# HTML payload
# ---------------------------------------------------------------------------
#
# Everything below is one self-contained HTML page. It uses vis-network
# from a CDN; everything else is inline. The Python side injects:
#
#   __JSON_PAYLOAD__   -> JSON.stringify({ nodes, edges, meta, variants })
#   __PAGE_TITLE__     -> human-readable title for the <title> tag.
#
# The page boots by:
#   1) parsing the injected payload,
#   2) building a vis-network DataSet for nodes/edges,
#   3) rendering the *full* graph,
#   4) wiring up three sliders + a text filter that re-render the visible
#      set on every change.
#
# We rebuild the DataSet (rather than mutating it in place) on every
# filter change so the initial render is always deterministic and the
# "Reset" button is trivial.

_HTML_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>__PAGE_TITLE__</title>
<!--
  vis-network: interactive graph engine (drag / zoom / hover tooltips).
  The launcher copies the vendored UMD bundle from
  ``dpg/static/vis-network.min.js`` next to this HTML so the page works
  offline, under ``file://``, and in sandboxes that block public CDNs. The
  same file is served as a relative URL so the very same page works
  whether it's opened locally or served over HTTP.
-->
<script src="vis-network.min.js"></script>
<style>
  :root {
    --bg: #fafafa;
    --panel-bg: #ffffff;
    --panel-border: #e0e0e0;
    --accent: #2563eb;
    --accent-soft: #dbeafe;
    --text: #1f2937;
    --muted: #6b7280;
    --warn: #b45309;
  }
  * { box-sizing: border-box; }
  html, body { margin: 0; padding: 0; height: 100%; font-family: -apple-system,
    BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; color: var(--text);
    background: var(--bg); }
  header {
    padding: 10px 16px; border-bottom: 1px solid var(--panel-border);
    background: var(--panel-bg);
    display: flex; flex-wrap: wrap; align-items: center; gap: 16px;
  }
  header h1 { font-size: 14px; font-weight: 600; margin: 0; }
  header .meta { font-size: 12px; color: var(--muted); }
  header .meta code { background: #f3f4f6; padding: 1px 5px; border-radius: 3px; }
  header .variants { display: flex; gap: 6px; }
  header .variants button {
    font-size: 12px; padding: 4px 10px; border-radius: 4px;
    border: 1px solid var(--panel-border); background: white; cursor: pointer;
  }
  header .variants button.active {
    background: var(--accent); color: white; border-color: var(--accent);
  }
  header .variants button:disabled { opacity: 0.4; cursor: not-allowed; }

  .layout { display: grid; grid-template-columns: 320px 1fr; height: calc(100vh - 49px); }
  aside {
    background: var(--panel-bg); border-right: 1px solid var(--panel-border);
    padding: 14px 16px; overflow-y: auto;
  }
  aside h2 { font-size: 12px; font-weight: 600; text-transform: uppercase;
    color: var(--muted); letter-spacing: 0.05em; margin: 18px 0 6px; }
  aside h2:first-child { margin-top: 0; }
  aside label { display: block; font-size: 12px; color: var(--muted);
    margin-bottom: 4px; }
  aside .row { margin-bottom: 12px; }
  aside input[type="range"] { width: 100%; }
  aside .value {
    font-size: 12px; color: var(--text); font-variant-numeric: tabular-nums;
    margin-left: 6px;
  }
  aside input[type="text"] {
    width: 100%; padding: 6px 8px; font-size: 13px; border-radius: 4px;
    border: 1px solid var(--panel-border);
  }
  aside .checkbox { display: flex; align-items: center; gap: 6px; font-size: 12px; }
  aside .actions { display: flex; gap: 8px; margin-top: 14px; }
  aside .actions button {
    flex: 1; font-size: 12px; padding: 6px 10px; border-radius: 4px;
    border: 1px solid var(--panel-border); background: white; cursor: pointer;
  }
  aside .actions button:hover { background: var(--accent-soft); }
  aside .stats {
    font-size: 11px; color: var(--muted); margin-top: 6px;
    font-variant-numeric: tabular-nums;
  }
  aside .warn {
    font-size: 11px; color: var(--warn); margin-top: 8px;
  }

  #graph { width: 100%; height: 100%; background: #ffffff; }
  /* Tooltip styling applied to the default vis-network tooltip element. */
  .vis-tooltip {
    background: rgba(31,41,55,0.95); color: white; padding: 6px 10px;
    border-radius: 4px; font-size: 12px; max-width: 360px;
    white-space: pre-wrap; word-break: break-word;
  }
</style>
</head>
<body>
<header>
  <h1 id="title">DPG interactive viewer</h1>
  <div class="meta" id="meta"></div>
  <div class="variants" id="variants">
    <button data-variant="raw" disabled>Raw</button>
    <button data-variant="grouped" disabled>Grouped (strict)</button>
    <button data-variant="split" disabled>Split-grouped</button>
  </div>
</header>
<div class="layout">
  <aside>
    <h2>Edge filter</h2>
    <div class="row">
      <label>Min edge weight <span class="value" id="edgeThresholdVal">0</span></label>
      <input type="range" id="edgeThreshold" min="0" max="0" step="1" value="0" />
      <div class="stats" id="edgeStats"></div>
    </div>

    <h2>Node filter</h2>
    <div class="row">
      <label>Min total node weight <span class="value" id="nodeThresholdVal">0</span></label>
      <input type="range" id="nodeThreshold" min="0" max="0" step="1" value="0" />
      <div class="stats" id="nodeStats"></div>
    </div>
    <div class="row">
      <label>Label contains (case-insensitive)</label>
      <input type="text" id="labelFilter" placeholder="e.g. loan_intent" />
    </div>
    <div class="row checkbox">
      <input type="checkbox" id="keepClassNodes" checked />
      <label for="keepClassNodes" style="margin: 0;">Always keep <code>Class *</code> nodes</label>
    </div>

    <h2>Physics</h2>
    <div class="row">
      <label>Solver iterations <span class="value" id="physicsIterVal">0</span></label>
      <input type="range" id="physicsIter" min="0" max="500" step="10" value="120" />
      <div class="stats">vis-network may need a few iterations before settling.</div>
    </div>

    <div class="actions">
      <button id="resetBtn">Reset filters</button>
      <button id="fitBtn">Refit view</button>
    </div>
  </aside>
  <div id="graph"></div>
</div>

<script>
"use strict";

// ---------------------------------------------------------------------------
// Injected payload
// ---------------------------------------------------------------------------
const PAYLOAD = __JSON_PAYLOAD__;

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------
const state = {
  nodes: [],            // master copy of the loaded nodes (id, label, is_class)
  edges: [],            // master copy of the loaded edges (src, dst, weight)
  meta: {},             // stats from the JSON
  variants: {},         // {variant_name: file:// URI}  for switching
  activeVariant: "input",
  // filter values
  edgeThreshold: 0,
  nodeThreshold: 0,
  labelFilter: "",
  keepClassNodes: true,
  // rendered artefacts (rebuilt on every filter change)
  network: null,
  nodesDS: null,
  edgesDS: null,
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
function clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }

function nodeWeightTotal(nodeId) {
  let s = 0;
  for (const e of state.edges) {
    if (e.source === nodeId || e.target === nodeId) s += e.weight;
  }
  return s;
}

function fmt(n) {
  if (!isFinite(n)) return "0";
  if (Math.abs(n) >= 1000) return n.toFixed(0);
  if (Math.abs(n) >= 1) return n.toFixed(2);
  return n.toFixed(3);
}

function escapeHtml(s) {
  return String(s)
    .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

// ---------------------------------------------------------------------------
// Payload loading
// ---------------------------------------------------------------------------
async function loadPayloadFromUri(uri, variantName) {
  const res = await fetch(uri);
  if (!res.ok) throw new Error("HTTP " + res.status + " for " + uri);
  const json = await res.json();
  return normalisePayload(json, variantName);
}

function normalisePayload(structure, variantName) {
  const rawNodes = structure.nodes || [];
  const graph = structure.graph || {};

  // Drop pseudo "edge label" nodes whose id contains "->". They show up in
  // the top-level nodes list to carry the edge's *text* label but never
  // participate in the actual graph (and never have a corresponding entry
  // in graph.nodes).
  const PSEUDO = "->";
  const labelMap = {};
  for (const n of rawNodes) {
    const id = String(n.id);
    if (id.includes(PSEUDO)) continue;
    labelMap[id] = (n.label != null) ? String(n.label) : "";
  }

  const graphNodeIds = new Set();
  for (const n of (graph.nodes || [])) graphNodeIds.add(String(n.id));

  const nodes = [];
  const allIds = new Set([...graphNodeIds, ...Object.keys(labelMap)]);
  for (const id of allIds) {
    if (id.includes(PSEUDO)) continue;
    const label = labelMap[id] || id;
    nodes.push({
      id,
      label,
      is_class: label.trim().toLowerCase().startsWith("class"),
    });
  }

  const rawEdges = graph.edges || graph.links || [];
  const edges = [];
  for (const e of rawEdges) {
    const src = String(e.source), dst = String(e.target);
    if (src.includes(PSEUDO) || dst.includes(PSEUDO)) continue;
    const w = Number(e.weight != null ? e.weight : 0) || 0;
    edges.push({ source: src, target: dst, weight: w });
  }

  const weights = edges.map(e => e.weight);
  const nodeW = {};
  for (const n of nodes) nodeW[n.id] = 0;
  for (const e of edges) {
    if (nodeW[e.source] != null) nodeW[e.source] += e.weight;
    if (nodeW[e.target] != null) nodeW[e.target] += e.weight;
  }

  // Drop stranded nodes (no incoming or outgoing edges), but always keep
  // Class nodes because they're the destination of every meaningful path.
  const incident = new Set();
  for (const e of edges) { incident.add(e.source); incident.add(e.target); }
  const liveNodes = nodes.filter(n => n.is_class || incident.has(n.id));

  return {
    nodes: liveNodes,
    edges,
    meta: {
      n_nodes: liveNodes.length,
      n_edges: edges.length,
      min_edge_weight: weights.length ? Math.min(...weights) : 0,
      max_edge_weight: weights.length ? Math.max(...weights) : 0,
      min_node_weight: Math.min(...Object.values(nodeW)),
      max_node_weight: Math.max(...Object.values(nodeW)),
      run_id: structure.run_id || "",
      variant: variantName,
    },
    nodeWeights: nodeW,
  };
}

// ---------------------------------------------------------------------------
// Rendering
// ---------------------------------------------------------------------------
function buildVisibleSets() {
  const labelLc = state.labelFilter.trim().toLowerCase();
  const visibleNodes = new Set();
  for (const n of state.nodes) {
    // 1. keep class nodes by default
    if (n.is_class && state.keepClassNodes) { visibleNodes.add(n.id); continue; }
    // 2. node weight threshold
    const w = state.nodeWeights[n.id] || 0;
    if (w < state.nodeThreshold) continue;
    // 3. label substring filter
    if (labelLc && !n.label.toLowerCase().includes(labelLc)) continue;
    visibleNodes.add(n.id);
  }

  const visibleEdges = [];
  for (const e of state.edges) {
    if (e.weight < state.edgeThreshold) continue;
    if (!visibleNodes.has(e.source) || !visibleNodes.has(e.target)) continue;
    visibleEdges.push(e);
  }

  return { visibleNodes, visibleEdges };
}

function rerender() {
  const { visibleNodes, visibleEdges } = buildVisibleSets();

  const nodeRecords = [];
  for (const n of state.nodes) {
    if (!visibleNodes.has(n.id)) continue;
    const w = state.nodeWeights[n.id] || 0;
    // Node size: linear in total weight, clamped so a single dominant
    // node doesn't dwarf the others.
    const size = 12 + clamp(Math.sqrt(w) * 2, 0, 22);
    const color = n.is_class
      ? { background: "#fde68a", border: "#b45309", highlight: { background: "#fcd34d", border: "#92400e" } }
      : { background: "#dbeafe", border: "#1d4ed8", highlight: { background: "#bfdbfe", border: "#1e3a8a" } };
    nodeRecords.push({
      id: n.id,
      label: n.label,
      title: n.label + "\n" +
             "id: " + n.id + "\n" +
             "total weight: " + fmt(w) + "\n" +
             (n.is_class ? "(class node — always visible when toggled)\n" : ""),
      shape: n.is_class ? "box" : "ellipse",
      size,
      color,
      font: { face: "monospace", size: 11 },
    });
  }

  const edgeRecords = [];
  for (const e of visibleEdges) {
    const w = e.weight;
    // Edge thickness: linear in weight, clamped.
    const width = clamp(0.5 + Math.log10(Math.max(w, 1)) * 1.4, 0.5, 8);
    edgeRecords.push({
      id: e.source + "->" + e.target,
      from: e.source,
      to: e.target,
      width,
      label: fmt(w),
      title: "weight: " + fmt(w),
      arrows: "to",
      color: { color: "#6b7280", highlight: "#111827" },
      font: { size: 9, color: "#374151", strokeWidth: 2, strokeColor: "#ffffff" },
      smooth: { enabled: true, type: "dynamic", roundness: 0.2 },
    });
  }

  state.nodesDS.clear(); state.nodesDS.add(nodeRecords);
  state.edgesDS.clear(); state.edgesDS.add(edgeRecords);

  // Update sidebar stats
  document.getElementById("edgeStats").textContent =
    "showing " + edgeRecords.length + " / " + state.edges.length + " edges";
  document.getElementById("nodeStats").textContent =
    "showing " + nodeRecords.length + " / " + state.nodes.length + " nodes";
  document.getElementById("meta").innerHTML =
    "<code>" + state.activeVariant + "</code> · " +
    state.nodes.length + " nodes · " +
    state.edges.length + " edges";
}

function setSliderRange(slider, lo, hi, decimals) {
  slider.min = lo;
  slider.max = hi;
  if (decimals === 0) slider.step = 1;
  else slider.step = (hi - lo) / 200 || 0.01;
}

function applyFilterDefaults() {
  setSliderRange(document.getElementById("edgeThreshold"),
                 0, Math.ceil(state.meta.max_edge_weight || 0), 0);
  document.getElementById("edgeThreshold").value = 0;
  state.edgeThreshold = 0;
  document.getElementById("edgeThresholdVal").textContent = 0;

  setSliderRange(document.getElementById("nodeThreshold"),
                 0, Math.ceil(state.meta.max_node_weight || 0), 0);
  document.getElementById("nodeThreshold").value = 0;
  state.nodeThreshold = 0;
  document.getElementById("nodeThresholdVal").textContent = 0;

  document.getElementById("labelFilter").value = "";
  state.labelFilter = "";
  document.getElementById("keepClassNodes").checked = true;
  state.keepClassNodes = true;
}

function attachEvents() {
  const edgeSlider = document.getElementById("edgeThreshold");
  edgeSlider.addEventListener("input", (ev) => {
    state.edgeThreshold = Number(ev.target.value);
    document.getElementById("edgeThresholdVal").textContent = fmt(state.edgeThreshold);
    rerender();
  });

  const nodeSlider = document.getElementById("nodeThreshold");
  nodeSlider.addEventListener("input", (ev) => {
    state.nodeThreshold = Number(ev.target.value);
    document.getElementById("nodeThresholdVal").textContent = fmt(state.nodeThreshold);
    rerender();
  });

  document.getElementById("labelFilter").addEventListener("input", (ev) => {
    state.labelFilter = ev.target.value;
    rerender();
  });

  document.getElementById("keepClassNodes").addEventListener("change", (ev) => {
    state.keepClassNodes = ev.target.checked;
    rerender();
  });

  document.getElementById("physicsIter").addEventListener("input", (ev) => {
    const v = Number(ev.target.value);
    document.getElementById("physicsIterVal").textContent = v;
    if (state.network) {
      state.network.setOptions({
        physics: { stabilization: { iterations: v } },
      });
      state.network.stabilize();
    }
  });

  document.getElementById("resetBtn").addEventListener("click", () => {
    applyFilterDefaults();
    rerender();
  });
  document.getElementById("fitBtn").addEventListener("click", () => {
    if (state.network) state.network.fit({ animation: { duration: 300 } });
  });
}

// ---------------------------------------------------------------------------
// Boot
// ---------------------------------------------------------------------------
async function boot() {
  // The injected payload provides the *initial* nodes/edges/meta so the
  // page is useful even when opened via file:// on a single JSON (where
  // fetch() would fail due to CORS). Variants are exposed as URIs but the
  // user must run a local server (or click the file directly with a
  // browser that allows file:// fetches) to actually switch between them.
  const initial = PAYLOAD;
  state.nodes = initial.nodes;
  state.edges = initial.edges;
  state.meta = initial.meta;
  state.nodeWeights = initial.nodeWeights;
  state.variants = initial.variants;
  state.activeVariant = initial.meta.variant || "input";

  // Build vis-network DataSets
  state.nodesDS = new vis.DataSet([]);
  state.edgesDS = new vis.DataSet([]);

  const container = document.getElementById("graph");
  state.network = new vis.Network(container, {
    nodes: state.nodesDS,
    edges: state.edgesDS,
  }, {
    layout: { improvedLayout: true, hierarchical: false },
    physics: {
      enabled: true,
      solver: "forceAtlas2Based",
      forceAtlas2Based: { gravitationalConstant: -45, springLength: 90,
                          springConstant: 0.08, damping: 0.9, avoidOverlap: 0.5 },
      stabilization: { iterations: 120, fit: true },
    },
    interaction: { hover: true, tooltipDelay: 100, multiselect: true,
                   dragNodes: true, zoomView: true },
    edges: { selectionWidth: 1.5 },
  });

  attachEvents();
  applyFilterDefaults();
  rerender();
  updateVariantButtons();

  // Variant buttons: re-fetch the JSON for the chosen variant and re-render.
  document.querySelectorAll("#variants button").forEach((btn) => {
    btn.addEventListener("click", async () => {
      const variant = btn.dataset.variant;
      const uri = state.variants[variant];
      if (!uri) return;
      try {
        const loaded = await loadPayloadFromUri(uri, variant);
        state.nodes = loaded.nodes;
        state.edges = loaded.edges;
        state.meta = loaded.meta;
        state.nodeWeights = loaded.nodeWeights;
        state.activeVariant = variant;
        applyFilterDefaults();
        rerender();
        updateVariantButtons();
      } catch (err) {
        alert("Failed to load variant '" + variant + "':\\n" + err.message +
              "\\n\\nTip: open this page through `python -m dpg.interactive_visualizer`," +
              " which starts a tiny local server that allows fetch().");
      }
    });
  });
}

function updateVariantButtons() {
  document.querySelectorAll("#variants button").forEach((btn) => {
    const v = btn.dataset.variant;
    btn.disabled = !state.variants[v];
    btn.classList.toggle("active", v === state.activeVariant);
  });
}

window.addEventListener("DOMContentLoaded", boot);
</script>
</body>
</html>
"""


def build_html_payload(structure: dict, *, title: str,
                       variants: Optional[dict] = None) -> str:
    """Render the HTML page with the given structure JSON baked in.

    Parameters
    ----------
    structure : dict
        The raw ``*_structure.json`` payload.
    title : str
        Page title (shown in the browser tab).
    variants : dict, optional
        ``{variant_name: "file://…"}`` so the user can switch between the
        raw / grouped / split-grouped variants. ``None`` disables the
        switcher.

    Returns
    -------
    str
        Full HTML source.
    """
    normalised = _normalise_payload(structure)
    payload = {
        "nodes": normalised["nodes"],
        "edges": normalised["edges"],
        "meta": normalised["meta"],
        "nodeWeights": {
            n["id"]: sum(
                e["weight"] for e in normalised["edges"]
                if e["source"] == n["id"] or e["target"] == n["id"]
            )
            for n in normalised["nodes"]
        },
        "variants": variants or {},
    }
    html = _HTML_TEMPLATE
    html = html.replace("__JSON_PAYLOAD__", json.dumps(payload))
    html = html.replace("__PAGE_TITLE__", title.replace("</", "<\\/"))
    return html


# ---------------------------------------------------------------------------
# Disk + browser plumbing
# ---------------------------------------------------------------------------

# Path to the vendored vis-network UMD bundle. The HTML template references
# it via a relative URL, so we have to copy it next to every generated
# HTML file (the JSONs live in arbitrary places: ``wip/``,
# ``wip/grouping_split/``, etc., so a single source directory can't serve
# them all).
_STATIC_DIR = Path(__file__).resolve().parent / "static"
_VENDORED_JS = _STATIC_DIR / "vis-network.min.js"


def _copy_static_assets(dest_dir: str | os.PathLike) -> None:
    """Copy the vendored vis-network UMD bundle into ``dest_dir`` so the
    HTML can load it via a relative ``<script src>``.

    Falls back silently (with a printed warning) if the source bundle is
    missing — the HTML will still try to load the public CDN.
    """
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    if _VENDORED_JS.exists():
        target = dest_dir / _VENDORED_JS.name
        if not target.exists() or _VENDORED_JS.stat().st_mtime > target.stat().st_mtime:
            target.write_bytes(_VENDORED_JS.read_bytes())
    else:
        print(
            f"  ! Note: vendored vis-network bundle not found at {_VENDORED_JS}; "
            f"the HTML will try to load it from the public CDN instead.",
            file=sys.stderr,
        )


def write_payload(html: str, dest_path: str | os.PathLike) -> Path:
    p = Path(dest_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(html, encoding="utf-8")
    return p


def open_payload(html_path: str | os.PathLike) -> None:
    webbrowser.open(Path(html_path).resolve().as_uri())


# ---------------------------------------------------------------------------
# High-level entry points
# ---------------------------------------------------------------------------

def open_interactive_view(json_path: str | os.PathLike,
                          *, open_browser: bool = True) -> Path:
    """Read a DPG ``*_structure.json`` and open an interactive HTML view.

    The HTML is written next to the JSON (same directory) as
    ``<run_id>_interactive.html``. If sibling variants (raw / grouped /
    split-grouped) exist, the user can switch between them via buttons.

    Returns the path to the generated HTML file.
    """
    json_path = Path(json_path).resolve()
    if not json_path.exists():
        raise FileNotFoundError(f"JSON not found: {json_path}")

    structure = json.loads(json_path.read_text(encoding="utf-8"))
    variants = discover_variants(json_path)

    # Build a sensible default title from the run subdirectory + variant.
    run_dir_name = json_path.parent.name
    if run_dir_name in ("wip", "grouping_split"):
        run_dir_name = json_path.parent.parent.name
    title = f"DPG viewer — {run_dir_name} ({json_path.name})"

    html = build_html_payload(structure, title=title, variants=variants)

    run_id = _strip_known_suffix(json_path.name) or json_path.stem
    out_html = json_path.parent / f"{run_id}_interactive.html"
    # If the JSON lives in wip/grouping_split/, keep the HTML next to it.
    out_path = write_payload(html, out_html)
    # Copy the vendored vis-network UMD bundle next to the HTML so the
    # page works offline / under file:// / behind firewalls that block
    # public CDNs.
    _copy_static_assets(out_path.parent)

    if open_browser:
        open_payload(out_path)
    return out_path


def _walk_for_default_json(root: str | os.PathLike) -> Optional[Path]:
    """Walk ``root`` and return the first ``*_DPG_grouped_structure.json``,
    or failing that the first ``*_dpg_structure.json``, or ``None``.
    """
    root = Path(root)
    if not root.is_dir():
        return None
    for pat in ("**/*_DPG_grouped_structure.json",
                "**/*_dpg_structure.json",
                "**/*_DPG_split_grouped_structure.json"):
        hits = sorted(root.glob(pat))
        if hits:
            return hits[0]
    return None


def main(argv: Optional[Iterable[str]] = None) -> int:
    args: List[str] = list(argv if argv is not None else sys.argv[1:])
    if not args:
        # Try to find a default in examples/results_cat/.
        project_root = Path(__file__).resolve().parent.parent
        candidates = [
            project_root / "examples" / "results_cat",
            project_root / "examples" / "results_gridsearch",
        ]
        for c in candidates:
            hit = _walk_for_default_json(c)
            if hit:
                args = [str(hit)]
                break
        if not args:
            print(
                "Usage: python -m dpg.interactive_visualizer <path/to/structure.json>\n"
                "\n"
                "No <structure.json> argument supplied and no default found\n"
                "under examples/results_cat/ or examples/results_gridsearch/.",
                file=sys.stderr,
            )
            return 2

    out_path = open_interactive_view(args[0])
    print(f"Interactive viewer written to: {out_path}")
    print("Opening in your default browser...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())