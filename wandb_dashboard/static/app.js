"use strict";

// ---------------------------------------------------------------------------
// small utilities
// ---------------------------------------------------------------------------

const $ = (sel, root = document) => root.querySelector(sel);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const cssVar = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const SERIES = () => [1, 2, 3, 4, 5, 6, 7, 8].map((i) => cssVar(`--series-${i}`));

const store = {
  get(k, d = null) { try { const v = localStorage.getItem(k); return v === null ? d : JSON.parse(v); } catch { return d; } },
  set(k, v) { try { localStorage.setItem(k, JSON.stringify(v)); } catch { /* storage unavailable */ } },
  del(k) { try { localStorage.removeItem(k); } catch { /* storage unavailable */ } },
};

function fmtNum(v) {
  if (typeof v !== "number" || !isFinite(v)) return String(v);
  if (Number.isInteger(v)) return v.toLocaleString();
  const a = Math.abs(v);
  if (a !== 0 && (a < 1e-3 || a >= 1e6)) return v.toExponential(3);
  if (a < 1) return v.toFixed(4);
  return v.toLocaleString(undefined, { maximumFractionDigits: 3 });
}
function fmtBytes(n) {
  if (n == null) return "";
  const u = ["B", "KB", "MB", "GB"]; let i = 0;
  while (n >= 1024 && i < u.length - 1) { n /= 1024; i++; }
  return `${n.toFixed(i ? 1 : 0)} ${u[i]}`;
}
function fmtDuration(s) {
  if (s == null || !isFinite(s)) return "—";
  s = Math.round(s);
  const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), sec = s % 60;
  return h ? `${h}h ${m}m ${sec}s` : m ? `${m}m ${sec}s` : `${sec}s`;
}
function fmtDate(iso) {
  if (!iso) return "—";
  const d = new Date(/Z|[+-]\d\d:?\d\d$/.test(iso) ? iso : iso + "Z");
  return isNaN(d) ? iso : d.toLocaleString();
}

function parseCSV(text) {
  const rows = []; let row = [], field = "", q = false;
  for (let i = 0; i < text.length; i++) {
    const c = text[i];
    if (q) {
      if (c === '"') { if (text[i + 1] === '"') { field += '"'; i++; } else q = false; }
      else field += c;
    } else if (c === '"') q = true;
    else if (c === ",") { row.push(field); field = ""; }
    else if (c === "\n" || c === "\r") {
      if (c === "\r" && text[i + 1] === "\n") i++;
      row.push(field); rows.push(row); row = []; field = "";
    } else field += c;
  }
  if (field || row.length) { row.push(field); rows.push(row); }
  return rows.filter((r) => r.length > 1 || r[0] !== "");
}
// Long digit strings (DPG node ids are 48-digit hashes) must stay strings to keep precision.
const numOrStr = (s) => (s !== "" && !isNaN(+s) && !/^-?\d{16,}$/.test(s.trim()) ? +s : s);

// Python repr helpers for DPG's communities txt ("['a', 'b']", "{'Class X': 0.7}")
const pyList = (s) => [...String(s).matchAll(/'((?:[^'\\]|\\.)*)'|"((?:[^"\\]|\\.)*)"/g)].map((m) => m[1] ?? m[2]);
const pyDict = (s) => Object.fromEntries([...String(s).matchAll(/'([^']*)'\s*:\s*([-\d.eE+]+)/g)].map((m) => [m[1], +m[2]]));

function textOn(hex) {
  const m = /^#?([\da-f]{2})([\da-f]{2})([\da-f]{2})/i.exec(hex || "");
  if (!m) return "#0b0b0b";
  const [r, g, b] = [m[1], m[2], m[3]].map((x) => { const c = parseInt(x, 16) / 255; return c <= 0.03928 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4; });
  return 0.2126 * r + 0.7152 * g + 0.0722 * b > 0.36 ? "#0b0b0b" : "#ffffff";
}

const tooltip = $("#tooltip");
function showTip(html, x, y) {
  tooltip.innerHTML = html; tooltip.hidden = false;
  const r = tooltip.getBoundingClientRect();
  tooltip.style.left = Math.min(x + 14, innerWidth - r.width - 8) + "px";
  tooltip.style.top = Math.min(y + 14, innerHeight - r.height - 8) + "px";
}
const hideTip = () => { tooltip.hidden = true; };

// ---------------------------------------------------------------------------
// settings + API
// ---------------------------------------------------------------------------

const settings = () => store.get("wbv.settings", { key: "", entity: "", project: "" });

async function api(path, params = {}) {
  const qs = new URLSearchParams(params).toString();
  const resp = await fetch(`${path}${qs ? "?" + qs : ""}`, { headers: { "X-Wandb-Key": settings().key || "" } });
  const body = await resp.json().catch(() => ({ error: `HTTP ${resp.status}` }));
  if (!resp.ok) throw new Error(body.error || `HTTP ${resp.status}`);
  return body;
}

const dlg = $("#settings");
function openSettings() {
  const s = settings();
  $("#set-key").value = s.key || ""; $("#set-entity").value = s.entity || ""; $("#set-project").value = s.project || "";
  $("#whoami").textContent = "";
  dlg.showModal();
}
$("#settings-btn").onclick = openSettings;
$("#toggle-key").onclick = () => {
  const i = $("#set-key"); i.type = i.type === "password" ? "text" : "password";
  $("#toggle-key").textContent = i.type === "password" ? "Show" : "Hide";
};
$("#clear-key").onclick = () => { $("#set-key").value = ""; };
function readSettingsForm() {
  return { key: $("#set-key").value.trim(), entity: $("#set-entity").value.trim(), project: $("#set-project").value.trim() };
}
$("#test-key").onclick = async () => {
  const out = $("#whoami"); out.className = "small muted"; out.textContent = "Checking…";
  const prev = settings(); store.set("wbv.settings", { ...prev, key: $("#set-key").value.trim() });
  try {
    const w = await api("/api/whoami");
    out.className = "small ok";
    out.textContent = `Signed in as ${w.username} · default entity ${w.entity}`;
    $("#entity-list").innerHTML = Object.keys(w.projects).map((e) => `<option value="${esc(e)}">`).join("");
    $("#project-list").innerHTML = [...new Set(Object.values(w.projects).flat())].map((p) => `<option value="${esc(p)}">`).join("");
  } catch (e) { out.className = "small bad"; out.textContent = e.message; }
  finally { store.set("wbv.settings", prev); }
};
$("#save-settings").onclick = () => { store.set("wbv.settings", readSettingsForm()); dlg.close(); updateKeyHint(); };

function updateKeyHint() {
  $("#key-hint").innerHTML = settings().key
    ? "API key set in Settings."
    : 'No API key yet. Open <a href="#" id="hint-settings">Settings</a> to add one.';
  const a = $("#hint-settings"); if (a) a.onclick = (e) => { e.preventDefault(); openSettings(); };
}

// theme toggle (follows OS until the viewer picks one)
function applyTheme(t) { if (t) document.documentElement.dataset.theme = t; else delete document.documentElement.dataset.theme; }
applyTheme(store.get("wbv.theme"));
$("#theme-btn").onclick = () => {
  const dark = cssVar("--surface-1") === "#1a1a19";
  const next = dark ? "light" : "dark"; store.set("wbv.theme", next); applyTheme(next); rerenderThemed();
};
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => rerenderThemed());

// ---------------------------------------------------------------------------
// loading a run
// ---------------------------------------------------------------------------

let current = null;           // last loaded payload
const themedRenderers = [];    // re-run on theme change (graphs + plots read CSS vars)
function rerenderThemed() { themedRenderers.forEach((fn) => fn()); }

function renderRecent() {
  const list = store.get("wbv.recent", []);
  $("#recent").innerHTML = list.map((r) => `<button class="chip" data-run="${esc(r.ref)}" title="${esc(r.ref)}">${esc(r.label)}</button>`).join("");
  $("#recent").querySelectorAll(".chip").forEach((b) => (b.onclick = () => { $("#run-input").value = b.dataset.run; loadRun(b.dataset.run); }));
}
function remember(d) {
  const ref = `${d.entity}/${d.project}/${d.run.name}`;
  const list = store.get("wbv.recent", []).filter((r) => r.ref !== ref);
  list.unshift({ ref, label: `${d.run.displayName || d.run.name} · ${d.run.name}` });
  store.set("wbv.recent", list.slice(0, 8)); renderRecent();
}

async function loadRun(ref, refresh = false) {
  ref = (ref || "").trim(); if (!ref) return;
  if (!settings().key) { openSettings(); return; }
  const s = settings(), status = $("#status");
  $("#empty").hidden = true;
  status.hidden = false; status.className = "status";
  status.innerHTML = `<span class="spinner"></span>Fetching <b>${esc(ref)}</b> from W&amp;B…`;
  $("#load-btn").disabled = true;
  try {
    const d = await api("/api/run", { run: ref, entity: s.entity, project: s.project, ...(refresh ? { refresh: 1 } : {}) });
    status.hidden = true;
    history.replaceState(null, "", `#run=${encodeURIComponent(`${d.entity}/${d.project}/${d.run.name}`)}`);
    current = d; remember(d); render(d);
    $("#refresh-btn").hidden = false;
  } catch (e) {
    status.className = "status error"; status.textContent = e.message;
  } finally { $("#load-btn").disabled = false; }
}
$("#run-form").onsubmit = (e) => { e.preventDefault(); loadRun($("#run-input").value); };
$("#refresh-btn").onclick = () => current && loadRun(`${current.entity}/${current.project}/${current.run.name}`, true);

// ---------------------------------------------------------------------------
// rendering
// ---------------------------------------------------------------------------

function render(d) {
  themedRenderers.length = 0;
  cyInstances.forEach((cy) => cy.destroy()); cyInstances.length = 0;
  const c = $("#content"); c.hidden = false; c.innerHTML = "";

  const variants = collectVariants(d);
  const usedMedia = new Set(variants.flatMap((v) => v.images.map((i) => i.key)));
  const sections = [
    ["overview", "Overview", renderOverview],
    ...(variants.length ? [["graphs", "DPG graphs", (el) => renderGraphs(el, variants)]] : []),
    ["metrics", "Metrics", renderMetrics],
    ...(Object.values(d.media).some((m) => m._type === "table-file") ? [["tables", "Tables", renderTables]] : []),
    ...(Object.entries(d.media).some(([k, m]) => m._type === "image-file" && !usedMedia.has(k)) ? [["media", "Media", (el) => renderMedia(el, usedMedia)]] : []),
    ["artifacts", "Artifacts", renderArtifacts],
    ["config", "Config", renderConfig],
    ["files", "Files & logs", renderFiles],
  ];
  const nav = document.createElement("nav"); nav.className = "section-nav";
  nav.innerHTML = sections.map(([id, t]) => `<a href="#sec-${id}" data-sec="${id}">${esc(t)}</a>`).join("");
  nav.querySelectorAll("a").forEach((a) => (a.onclick = (e) => { e.preventDefault(); $(`#sec-${a.dataset.sec}`).scrollIntoView({ behavior: "smooth", block: "start" }); }));
  c.appendChild(nav);
  for (const [id, , fn] of sections) {
    const el = document.createElement("section"); el.className = "card"; el.id = `sec-${id}`; el.style.scrollMarginTop = "70px";
    c.appendChild(el); fn(el, d);
  }
}

function statePill(state) {
  const map = { finished: ["good", "✓"], running: ["warning", "●"], crashed: ["critical", "✕"], failed: ["critical", "✕"], killed: ["critical", "■"] };
  const [cls, icon] = map[state] || ["neutral", "○"];
  return `<span class="pill ${cls}">${icon} ${esc(state)}</span>`;
}

function renderOverview(el, d) {
  const r = d.run, s = d.summary;
  const runtime = s._runtime ?? s["_wandb.runtime"] ?? (s._wandb && s._wandb.runtime);
  const meta = [
    ["Run id", `<span class="mono">${esc(r.name)}</span>`],
    ["Project", `${esc(d.entity)}/<b>${esc(d.project)}</b>`],
    ["Group", esc(r.group || "—")], ["Job type", esc(r.jobType || "—")],
    ["User", esc(r.user?.username || "—")], ["Host", esc(r.host || "—")],
    ["Created", fmtDate(r.createdAt)], ["Last heartbeat", fmtDate(r.heartbeatAt)],
    ["Runtime", fmtDuration(runtime)], ["Steps logged", esc(r.historyLineCount ?? "—")],
    ["wandb / python", esc(`${r.wandb?.cli_version || "?"} / ${r.wandb?.python_version || "?"}`)],
    ...(r.commit ? [["Commit", `<span class="mono">${esc(r.commit.slice(0, 10))}</span>`]] : []),
    ...(r.tags?.length ? [["Tags", r.tags.map((t) => `<span class="tag">${esc(t)}</span>`).join("")]] : []),
  ];
  const scalars = Object.entries(s).filter(([k, v]) => !k.startsWith("_") && (typeof v === "number" || typeof v === "string" || typeof v === "boolean"));
  el.innerHTML = `
    <div class="run-head">
      <div class="run-title"><h1>${esc(r.displayName || r.name)}</h1>${statePill(r.state)}</div>
      <a class="btn sm" href="${esc(d.url)}" target="_blank" rel="noopener">Open in W&amp;B ↗</a>
    </div>
    ${r.notes ? `<p class="muted">${esc(r.notes)}</p>` : ""}
    <dl class="meta-grid">${meta.map(([k, v]) => `<div><dt>${esc(k)}</dt><dd>${v}</dd></div>`).join("")}</dl>
    ${scalars.length ? `<div class="subhead">Summary <span class="muted">${scalars.length} values</span></div><div class="tiles">${scalars.map(([k, v], i) => `
      <div class="tile"${i >= 24 ? " hidden" : ""}><div class="k">${esc(k)}</div><div class="v ${typeof v === "number" ? "" : "text"}">${typeof v === "number" ? fmtNum(v) : esc(v)}</div></div>`).join("")}</div>
      ${scalars.length > 24 ? `<button class="btn sm more-tiles" style="margin-top:10px">Show all ${scalars.length}</button>` : ""}` : ""}`;
  const more = $(".more-tiles", el);
  if (more) more.onclick = () => { const hid = el.querySelector(".tile[hidden]"); el.querySelectorAll(".tile").forEach((t, i) => { t.hidden = hid ? false : i >= 24; }); more.textContent = hid ? "Show fewer" : `Show all ${scalars.length}`; };
}

// ---------------------------------------------------------------------------
// DPG graphs
// ---------------------------------------------------------------------------

const artText = (a, re) => { const f = a.files.find((f) => re.test(f.name)); return f && f.content; };

function collectVariants(d) {
  const byVariant = new Map();
  for (const a of d.artifacts) {
    const key = a.variant || a.name;
    if (!byVariant.has(key)) byVariant.set(key, []);
    byVariant.get(key).push(a);
  }
  const variants = [];
  for (const [key, arts] of byVariant) {
    for (const sa of arts.filter((a) => a.type === "dpg_structure")) {
      const raw = artText(sa, /\.json$/i); if (!raw) continue;
      let structure; try { structure = JSON.parse(raw); } catch { continue; }
      if (!structure.graph || !structure.nodes) continue;
      const pretty = key.replace(/_/g, " ");
      const aux = (re) => arts.map((a) => artText(a, re)).find(Boolean);
      const images = Object.entries(d.media)
        .filter(([k, m]) => m._type === "image-file" && k.startsWith(`images/${pretty}/`))
        .map(([k, m]) => ({ key: k, url: m.url, name: k.split("/").pop() }));
      variants.push({
        key, label: pretty, structure,
        communities: aux(/_communities\.txt$/i),
        nodeMetrics: aux(/_node_metrics\.csv$/i),
        classBounds: aux(/_class_boundaries\.txt$/i),
        images,
      });
    }
  }
  const order = (v) => (/^BASIC/i.test(v.key) ? 0 : 1);
  return variants.sort((a, b) => order(a) - order(b) || a.key.length - b.key.length || a.key.localeCompare(b.key));
}

function parseCommunities(text) {
  if (!text) return null;
  const clusters = [], probs = {};
  for (const [section, key, value] of parseCSV(text).slice(1)) {
    if (section === "Clusters") clusters.push({ name: key, members: pyList(value) });
    else if (section === "Probability") probs[key] = pyDict(value);
  }
  return clusters.length ? { clusters, probs } : null;
}

function parseNodeMetrics(text) {
  if (!text) return null;
  const rows = parseCSV(text); const head = rows[0]; const out = {};
  for (const r of rows.slice(1)) {
    const o = {}; head.forEach((h, i) => { if (h) o[h] = numOrStr(r[i]); });
    if (o.Node != null) out[String(o.Node)] = o;
  }
  return out;
}

// ---- cross-graph predicate matching -------------------------------------
// Reduce a node label to the set of basic-DPG predicates it covers, mirroring
// categorical/cat_grouping*.py:
//   one-hot  "<base>_<CAT> <= 0.5"  ->  "<base> NOT IN {CAT}"   (> 0.5 -> IN)
//   same-feature chains merge into "{A, B}", cross-feature ones join with " AND ".
// Numeric predicates are never rewritten. Two nodes match when their atom sets overlap.
const PRED_RE = /^\s*(.+?)\s*(<=|>=|<|>)\s*(-?[\d.]+(?:e[-+]?\d+)?)\s*$/i;
const IN_RE = /^\s*(.+?)\s+(NOT\s+IN|IN)\s+\{([^}]*)\}\s*$/;
// Same rule as cat_grouping._split_one_hot_column: split on the last "_", tail must not be numeric.
function isOneHot(col) {
  const i = col.lastIndexOf("_"); if (i <= 0 || i === col.length - 1) return false;
  return isNaN(Number(col.slice(i + 1)));
}
function predicateAtoms(label) {
  label = String(label ?? "").trim();
  if (/^Class /.test(label)) return [label];
  const atoms = [];
  for (const clause of label.split(/\s+AND\s+/)) {
    const inM = IN_RE.exec(clause);
    if (inM) {
      const side = /NOT/.test(inM[2]) ? "le" : "gt";
      inM[3].split(",").map((c) => c.trim()).filter(Boolean).forEach((c) => atoms.push(`${inM[1].trim()}_${c}|${side}`));
      continue;
    }
    const p = PRED_RE.exec(clause);
    if (p) {
      const [, feat, op, val] = p;
      // A 0.5 threshold on a one-hot column: key it the same way as the IN form above.
      if (Math.abs(+val - 0.5) < 1e-9 && isOneHot(feat)) atoms.push(`${feat.trim()}|${op.startsWith(">") ? "gt" : "le"}`);
      else atoms.push(`${feat.trim()}|${op}|${+val}`);
      continue;
    }
    atoms.push(clause.trim());
  }
  return atoms;
}

const cyInstances = [];
const FONT = "ui-sans-serif, system-ui, sans-serif";
const _measure = document.createElement("canvas").getContext("2d");
function textWidth(text, bold) { _measure.font = `${bold ? 700 : 400} 12px ${FONT}`; return _measure.measureText(text).width; }
const SEQ_BLUE = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"];

function renderGraphs(el, variants) {
  const isBasic = (v) => /^BASIC/i.test(v.key);
  const basic = variants.find(isBasic);
  const others = variants.filter((v) => v !== basic);
  el.innerHTML = `<h2>DPG graphs ${basic && others.length ? `<button class="btn sm layout-toggle" type="button"></button>` : ""}<span class="muted">rebuilt from the dpg_structure artifacts. Scroll to zoom, drag to pan, hover a node to trace its paths, click it for details. Selecting a node in one graph selects every node covering the same predicate in the other (e.g. <span class="mono">parents_usual &lt;= 0.5</span> ↔ <span class="mono">parents NOT IN {usual} AND …</span>).</span></h2>
    <div class="compare"></div>`;
  const compare = $(".compare", el);
  const panes = [];
  const link = (from) => (label) => panes.forEach((p) => p !== from && p.graph?.syncSelect(label));
  // Edge-weight thresholds: while locked, moving either slider moves the other to the same weight.
  let locked = store.get("wbv.thresholdLocked", true);
  const linkThreshold = (from) => (t) => { if (locked) panes.forEach((p) => p !== from && p.graph?.setThreshold(t)); };
  const lock = {
    get: () => locked,
    toggle: () => {
      locked = !locked; store.set("wbv.thresholdLocked", locked);
      if (locked && panes[0]?.graph) linkThreshold(panes[0])(panes[0].graph.threshold());
      return locked;
    },
  };

  const makePane = (title) => {
    const pane = document.createElement("div"); pane.className = "pane";
    pane.innerHTML = `<div class="variant-tabs" role="tablist"></div><div class="variant-body"></div>`;
    compare.appendChild(pane);
    const obj = { el: pane, graph: null, title };
    panes.push(obj);
    return obj;
  };
  const mountInto = (pane, v) => {
    if (pane.graph) {
      pane.graph.destroy();
      const idx = themedRenderers.indexOf(pane.graph.rethemer); if (idx >= 0) themedRenderers.splice(idx, 1);
    }
    const isLeft = pane === panes[0] && basic && others.length > 0;
    pane.graph = mountGraph($(".variant-body", pane.el), v, { onPin: link(pane), onThreshold: linkThreshold(pane), lock: isLeft ? lock : null });
    // A newly opened tab inherits the locked threshold.
    const src = panes.find((p) => p !== pane && p.graph);
    if (locked && src) pane.graph.setThreshold(src.graph.threshold());
    // Carry a selection made in the other graph over to the newly opened tab.
    const peer = panes.find((p) => p !== pane && p.graph?.ownLabel());
    if (peer) pane.graph.syncSelect(peer.graph.ownLabel());
  };

  // Left: the basic DPG, always shown.
  if (basic) {
    const left = makePane("basic");
    $(".variant-tabs", left.el).innerHTML = `<button role="tab" aria-selected="true" class="static">${esc(basic.label)}</button>`;
  }
  // Right: tabs over the grouped variants, defaulting to the conjunction one.
  let right = null;
  if (others.length) {
    right = makePane("grouped");
    const tabs = $(".variant-tabs", right.el);
    const def = Math.max(0, others.findIndex((v) => /CONJ/i.test(v.key)));
    const show = (i) => {
      tabs.querySelectorAll("button").forEach((b, j) => b.setAttribute("aria-selected", String(i === j)));
      mountInto(right, others[i]);
    };
    others.forEach((v, i) => {
      const b = document.createElement("button"); b.setAttribute("role", "tab"); b.textContent = v.label; b.onclick = () => show(i); tabs.appendChild(b);
    });
    if (basic) mountInto(panes[0], basic);
    show(def);
  } else if (basic) mountInto(panes[0], basic);
  compare.classList.toggle("two", panes.length > 1);

  // Side by side vs stacked (wide graphs read better stacked). Remembered per browser.
  const toggle = $(".layout-toggle", el);
  if (toggle) {
    const apply = (stacked) => {
      compare.classList.toggle("sbs", !stacked);
      toggle.textContent = stacked ? "⇆ Side by side" : "⇅ Stack vertically";
      toggle.title = stacked ? "Show the two graphs next to each other" : "Show the two graphs one above the other";
      panes.forEach((p) => p.graph?.refit());
    };
    let stacked = store.get("wbv.graphsStacked", false);
    apply(stacked);
    toggle.onclick = () => { stacked = !stacked; store.set("wbv.graphsStacked", stacked); apply(stacked); };
  } else compare.classList.add("sbs");
}

function mountGraph(host, v, { onPin = null, onThreshold = null, lock = null } = {}) {
  const s = v.structure;
  const labels = new Map(s.nodes.filter((n) => !String(n.id).includes("->")).map((n) => [String(n.id), n.label]));
  const comm = parseCommunities(v.communities);
  const metrics = parseNodeMetrics(v.nodeMetrics);
  const linkKey = s.graph.links ? "links" : "edges";
  const links = s.graph[linkKey] || [];
  const weights = links.map((l) => +l.weight || 0);
  const wMin = Math.min(...weights), wMax = Math.max(...weights);
  const uniqW = [...new Set(weights)].sort((x, y) => x - y); // slider steps through the distinct weights
  const nodeIds = new Set(s.graph.nodes.map((n) => String(n.id)));

  // community lookup: label -> cluster index
  const clusterOf = new Map();
  if (comm) comm.clusters.forEach((c, i) => c.members.forEach((m) => { if (!clusterOf.has(m)) clusterOf.set(m, i); }));
  const clusterColor = (i) => {
    const c = comm.clusters[i]; if (!c || /ambiguous/i.test(c.name)) return cssVar("--neutral-mark");
    const nonAmb = comm.clusters.filter((x) => !/ambiguous/i.test(x.name)); const k = nonAmb.indexOf(c);
    return k < 8 ? SERIES()[k] : cssVar("--neutral-mark");
  };

  const modes = [["default", "Default"]];
  if (comm) modes.push(["community", "Communities"]);
  if (metrics) {
    const cols = Object.keys(Object.values(metrics)[0] || {}).filter((k) => !["Node", "Label", ""].includes(k) && typeof Object.values(metrics)[0][k] === "number");
    cols.forEach((c) => modes.push([`metric:${c}`, c]));
  }
  const bigGraph = links.length > 250;

  host.innerHTML = `
    <div class="graph-toolbar">
      <div class="grp"><label>Color <select class="mode">${modes.map(([k, t]) => `<option value="${esc(k)}">${esc(t)}</option>`).join("")}</select></label></div>
      <div class="grp"><label>Layout <select class="dir"><option value="LR">Left → right</option><option value="TB">Top → bottom</option></select></label></div>
      <div class="grp"><input class="search" type="text" placeholder="Find predicate or class…" aria-label="Find node"></div>
      <div class="grp">
        <button class="btn sm wlabels" aria-pressed="${!bigGraph}">Edge weights</button>
        <button class="btn sm fit">Fit</button>
        <button class="btn sm tallbtn" aria-pressed="false">Taller</button>
        <button class="btn sm png">Export PNG</button>
        ${v.images.length ? `<button class="btn sm orig" aria-pressed="false">Original image${v.images.length > 1 ? "s" : ""}</button>` : ""}
      </div>
      <div class="grp thr">
        <label>Min edge weight <input type="range" class="thr-range" min="0" max="${Math.max(0, uniqW.length - 1)}" step="1" value="0" aria-label="Minimum edge weight"></label>
        <span class="thr-val small"></span>
        ${lock ? `<button class="btn sm thr-lock" type="button"></button>` : ""}
      </div>
    </div>
    <div class="graph-wrap">
      <div class="cy"></div>
      <aside class="side"></aside>
    </div>
    <div class="legend"></div>
    <div class="graph-stats"></div>
    <div class="orig-images" hidden></div>`;
  const cyEl = $(".cy", host), side = $(".side", host), legend = $(".legend", host);

  const elements = [];
  for (const id of nodeIds) {
    const label = labels.get(id) ?? id;
    const isClass = /^Class /.test(label);
    elements.push({ data: { id, label, atoms: predicateAtoms(label), isClass: isClass ? 1 : 0, cluster: clusterOf.has(label) ? clusterOf.get(label) : -1 } });
  }
  links.forEach((l, i) => {
    const w = +l.weight || 0;
    elements.push({ data: { id: `e${i}`, source: String(l.source), target: String(l.target), weight: w, wlabel: Number.isInteger(w) ? String(w) : fmtNum(w) } });
  });

  const classCount = elements.filter((e) => e.data.isClass).length;
  $(".graph-stats", host).innerHTML = `<span><b>${nodeIds.size}</b> nodes</span><span><b>${links.length}</b> edges</span><span><b>${classCount}</b> class nodes</span>
    ${s.community_threshold != null ? `<span>community threshold ${esc(s.community_threshold)}</span>` : ""}
    ${s.target_names ? `<span>${s.target_names.length} target classes</span>` : ""}
    ${s.feature_names ? `<span>${s.feature_names.length} features</span>` : ""}`;

  let showWeights = !bigGraph, mode = "default", dir = "LR", pinned = null;
  let ownLabel = null; // label the user picked in *this* graph (vs one synced in from the other)

  const style = () => {
    const nodeFill = cssVar("--node-fill"), nodeText = cssVar("--node-text"), nodeBorder = cssVar("--node-border");
    const edgeLo = cssVar("--edge-light"), edgeHi = cssVar("--edge-dark");
    const same = wMin === wMax;
    return [
      { selector: "node", style: {
        shape: "round-rectangle", "background-color": nodeFill, "border-width": 1, "border-color": nodeBorder,
        label: "data(label)", color: nodeText, "font-size": 12, "text-valign": "center", "text-halign": "center",
        width: (n) => textWidth(n.data("label"), n.data("isClass")) + 16, height: 28, padding: 0, "font-family": FONT,
      } },
      { selector: "node[isClass = 1]", style: {
        "background-color": cssVar("--class-fill"), color: cssVar("--class-text"), "border-color": cssVar("--class-border"),
        "border-width": 2, "font-weight": 700, shape: "round-rectangle", height: 32,
      } },
      { selector: "edge", style: {
        width: same ? 2 : `mapData(weight, ${wMin}, ${wMax}, 1, 5)`,
        "line-color": same ? edgeHi : `mapData(weight, ${wMin}, ${wMax}, ${edgeLo}, ${edgeHi})`,
        "target-arrow-color": same ? edgeHi : `mapData(weight, ${wMin}, ${wMax}, ${edgeLo}, ${edgeHi})`,
        "target-arrow-shape": "triangle", "arrow-scale": 0.9, "curve-style": "bezier",
        label: showWeights ? "data(wlabel)" : "", "font-size": 10, color: cssVar("--text-secondary"),
        "text-background-color": cssVar("--graph-bg"), "text-background-opacity": 0.85, "text-background-padding": "1px",
        "text-rotation": "autorotate",
      } },
      { selector: ".faded", style: { opacity: 0.12 } },
      { selector: ".pruned", style: { display: "none" } },
      { selector: "edge.hl", style: { "line-color": cssVar("--accent"), "target-arrow-color": cssVar("--accent"), "z-index": 9 } },
      { selector: "node.focus", style: { "border-width": 3, "border-color": cssVar("--accent") } },
      { selector: "node.match", style: { "border-width": 3, "border-color": cssVar("--series-2") } },
      { selector: "node[?cfill]", style: { "background-color": "data(cfill)", color: "data(ctext)" } },
    ];
  };

  const cy = cytoscape({
    container: cyEl, elements, style: style(), minZoom: 0.05, maxZoom: 4,
    layout: { name: "dagre", rankDir: dir, nodeSep: 14, rankSep: 55, edgeSep: 8, fit: false },
  });
  window.__dpgCy = cy; // handy from the devtools console
  cyInstances.push(cy);

  const relayout = () => { cy.layout({ name: "dagre", rankDir: dir, nodeSep: 14, rankSep: 55, edgeSep: 8, fit: false, animate: false }).run(); initialView(); };

  // Wide DPGs (hundreds of px per rank) are unreadable when fit whole; open at a readable zoom
  // anchored at the entry side and size the canvas to the graph. "Fit" shows the whole thing.
  const READABLE = 0.75;
  // When the details panel sits beside the canvas, cap it at the canvas height (no dead space below).
  function fitSide() {
    const beside = getComputedStyle(cyEl.parentElement).gridTemplateColumns.trim().split(/\s+/).length > 1;
    side.style.maxHeight = beside ? cyEl.style.height || "" : "";
  }
  function initialView() {
    if (cyEl.classList.contains("tall")) { cy.resize(); cy.fit(undefined, 20); side.style.maxHeight = ""; return; }
    const bb = cy.elements().boundingBox(), W = cyEl.clientWidth;
    const fitZoom = Math.min((W - 40) / bb.w, 600 / bb.h);
    const z = Math.min(Math.max(fitZoom, 0.05), 1.2) < READABLE ? READABLE : Math.min(fitZoom, 1.2);
    cyEl.style.height = Math.round(Math.min(Math.max(bb.h * z + 60, 420), 640)) + "px";
    cy.resize(); cy.zoom(z);
    fitSide();
    if (dir === "LR") cy.pan({ x: 20 - bb.x1 * z, y: cyEl.clientHeight / 2 - (bb.y1 + bb.h / 2) * z });
    else cy.pan({ x: W / 2 - (bb.x1 + bb.w / 2) * z, y: 20 - bb.y1 * z });
  }

  function applyMode() {
    cy.batch(() => {
      cy.nodes().forEach((n) => n.data({ cfill: null, ctext: null }));
      if (mode === "community" && comm) {
        cy.nodes("[isClass = 0]").forEach((n) => {
          const c = n.data("cluster"); const fill = c >= 0 ? clusterColor(c) : cssVar("--neutral-mark");
          n.data({ cfill: fill, ctext: textOn(fill) });
        });
      } else if (mode.startsWith("metric:")) {
        const col = mode.slice(7);
        const vals = cy.nodes("[isClass = 0]").map((n) => metrics[n.id()]?.[col]).filter((x) => typeof x === "number");
        const lo = Math.min(...vals), hi = Math.max(...vals);
        cy.nodes("[isClass = 0]").forEach((n) => {
          const x = metrics[n.id()]?.[col]; if (typeof x !== "number") return;
          const t = hi > lo ? (x - lo) / (hi - lo) : 0;
          const fill = SEQ_BLUE[Math.round(t * (SEQ_BLUE.length - 1))];
          n.data({ cfill: fill, ctext: textOn(fill) });
        });
      }
    });
    renderLegend();
  }

  function renderLegend() {
    const sw = (c) => `<span class="sw" style="background:${c}"></span>`;
    let html = `<span>${sw(cssVar("--class-fill"))}class node</span>`;
    if (mode === "default") html += `<span>${sw(cssVar("--node-fill"))}predicate</span>`;
    if (mode === "community" && comm) {
      html += comm.clusters.map((c, i) => {
        const count = cy.nodes(`[isClass = 0][cluster = ${i}]`).length;
        return `<span>${sw(clusterColor(i))}${esc(c.name)} <span class="muted">(${count})</span></span>`;
      }).join("");
      const none = cy.nodes("[isClass = 0][cluster = -1]").length;
      if (none) html += `<span>${sw(cssVar("--neutral-mark"))}unassigned <span class="muted">(${none})</span></span>`;
    }
    if (mode.startsWith("metric:")) {
      const col = mode.slice(7);
      const vals = Object.values(metrics).map((m) => m[col]).filter((x) => typeof x === "number");
      html += `<span>${esc(col)}: ${fmtNum(Math.min(...vals))}<span class="ramp" style="background:linear-gradient(90deg, ${SEQ_BLUE.join(",")})"></span>${fmtNum(Math.max(...vals))}</span>`;
    }
    html += `<span>edge width and shade = weight (${fmtNum(wMin)}–${fmtNum(wMax)})</span>`;
    legend.innerHTML = html;
  }

  function highlight(node) {
    cy.batch(() => {
      cy.elements().removeClass("faded hl focus");
      if (!node) return;
      const path = node.predecessors().union(node.successors()).union(node);
      cy.elements().not(path).addClass("faded");
      path.edges().addClass("hl");
      node.addClass("focus");
    });
  }

  const syncNote = (from, n) => from ? `<div class="sync-note">Matched from the other graph: <b>${esc(from)}</b>${n > 1 ? ` · ${n} nodes here` : ""}</div>` : "";
  function details(node, from = null) {
    if (node && node.length > 1) {
      side.innerHTML = `${syncNote(from, node.length)}<h3>${node.length} matching nodes</h3>
        <ul>${node.map((n) => `<li><button data-id="${esc(n.id())}">${esc(n.data("label"))}</button><span class="num">in ${n.indegree()} · out ${n.outdegree()}</span></li>`).join("")}</ul>`;
      side.querySelectorAll("button[data-id]").forEach((b) => (b.onclick = () => select(cy.getElementById(b.dataset.id), true)));
      return;
    }
    if (!node && from) {
      side.innerHTML = `${syncNote(from, 0)}<p class="muted">No node in this graph covers that predicate.</p>`;
      return;
    }
    if (!node) {
      side.innerHTML = `<h3>${esc(v.label)}</h3><p class="muted">Hover a node to trace every path through it. Click it to pin that view and see its connections${comm ? ", community and class probabilities" : ""}${metrics ? " and graph metrics" : ""}. Click empty space to clear.</p>
        ${v.classBounds ? `<h4>Class boundaries</h4>${renderClassBounds(v.classBounds)}` : ""}
        ${s.target_names ? `<h4>Target classes</h4><div class="small">${s.target_names.map(esc).join(", ")}</div>` : ""}`;
      bindClassBounds();
      return;
    }
    const label = node.data("label");
    const edgeItem = (e, other) => `<li><button data-id="${esc(other.id())}">${esc(other.data("label"))}</button><span class="num">${esc(e.data("wlabel"))}</span></li>`;
    const ins = node.incomers("edge").sort((a, b) => b.data("weight") - a.data("weight"));
    const outs = node.outgoers("edge").sort((a, b) => b.data("weight") - a.data("weight"));
    const m = metrics?.[node.id()];
    const pr = comm?.probs[label];
    const cl = node.data("cluster");
    const nonAmbIdx = comm ? comm.clusters.map((c, i) => [c, i]) : [];
    side.innerHTML = `${syncNote(from, 1)}
      <h3>${esc(label)}</h3>
      <div class="small muted">${node.data("isClass") ? "Class node" : "Predicate"} · in ${ins.length} · out ${outs.length}</div>
      ${comm && cl >= 0 ? `<div class="small" style="margin-top:6px"><span class="legend"><span><span class="sw" style="background:${clusterColor(cl)}"></span>${esc(comm.clusters[cl].name)}</span></span></div>` : ""}
      ${pr && Object.keys(pr).length ? `<h4>Class probability</h4>${Object.entries(pr).sort((a, b) => b[1] - a[1]).map(([k, p]) => {
        const hit = nonAmbIdx.find(([c]) => c.name === k); const col = hit ? clusterColor(hit[1]) : cssVar("--accent");
        return `<div class="prob"><span class="lbl small">${esc(k.replace(/^Class /, ""))}</span><span class="bar"><span style="width:${(p * 100).toFixed(1)}%;background:${col}"></span></span><span class="num small">${(p * 100).toFixed(0)}%</span></div>`;
      }).join("")}` : ""}
      ${m ? `<h4>Graph metrics</h4><ul>${Object.entries(m).filter(([k]) => !["Node", "Label", ""].includes(k)).map(([k, val]) => `<li><span>${esc(k)}</span><span class="num">${typeof val === "number" ? fmtNum(val) : esc(val)}</span></li>`).join("")}</ul>` : ""}
      <h4>Incoming (${ins.length})</h4><ul>${ins.map((e) => edgeItem(e, e.source())).join("") || '<li class="muted">none (entry point)</li>'}</ul>
      <h4>Outgoing (${outs.length})</h4><ul>${outs.map((e) => edgeItem(e, e.target())).join("") || '<li class="muted">none</li>'}</ul>`;
    side.querySelectorAll("button[data-id]").forEach((b) => (b.onclick = () => select(cy.getElementById(b.dataset.id), true)));
  }
  function renderClassBounds(text) {
    const m = /\{([\s\S]*)\}/.exec(text); if (!m) return `<pre class="text">${esc(text)}</pre>`;
    const parts = [...m[1].matchAll(/'(Class [^']+)'\s*:\s*\[([^\]]*)\]/g)];
    if (!parts.length) return `<pre class="text">${esc(text)}</pre>`;
    return parts.map(([, cls, list]) => `<div class="small" style="margin-bottom:8px"><button class="btn sm cb" data-label="${esc(cls)}" style="margin-bottom:3px">${esc(cls.replace(/^Class /, ""))}</button><div class="muted">${pyList(list).map(esc).join(" · ")}</div></div>`).join("");
  }
  function bindClassBounds() {
    side.querySelectorAll("button.cb").forEach((b) => (b.onclick = () => {
      const n = cy.nodes().filter((x) => x.data("label") === b.dataset.label); if (n.length) select(n[0], true);
    }));
  }

  function select(node, center = false) {
    pinned = node && node.length ? node : null;
    ownLabel = pinned ? pinned.data("label") : null;
    if (onPin) onPin(ownLabel);
    highlight(pinned); details(pinned);
    if (pinned && center) cy.animate({ center: { eles: pinned }, duration: 250 });
  }

  // Selection pushed from the other graph: select every node whose predicates overlap.
  function syncSelect(label) {
    ownLabel = null;
    if (!label) { pinned = null; highlight(null); details(null); return; }
    const want = new Set(predicateAtoms(label));
    const hits = cy.nodes().filter((n) => n.data("atoms").some((a) => want.has(a)));
    pinned = hits.length ? hits : null;
    highlight(pinned); details(pinned, label);
    if (!hits.length) return;
    // Keep the current zoom if every match fits in view; otherwise zoom out just enough.
    const bb = hits.boundingBox(), z = cy.zoom(), pad = 60;
    const fits = bb.w * z <= cy.width() - 2 * pad && bb.h * z <= cy.height() - 2 * pad;
    cy.animate(fits ? { center: { eles: hits }, duration: 250 } : { fit: { eles: hits, padding: pad }, duration: 250 });
  }

  cy.on("mouseover", "node", (e) => { if (!pinned) highlight(e.target); cyEl.style.cursor = "pointer"; });
  cy.on("mouseout", "node", () => { if (!pinned) highlight(null); cyEl.style.cursor = ""; hideTip(); });
  cy.on("mousemove", "node", (e) => {
    const n = e.target, m = metrics?.[n.id()];
    showTip(`<b>${esc(n.data("label"))}</b><div class="t">in ${n.indegree()} · out ${n.outdegree()}${m && m["Betweenness centrality"] != null ? ` · betweenness ${fmtNum(m["Betweenness centrality"])}` : ""}</div>`, e.originalEvent.clientX, e.originalEvent.clientY);
  });
  cy.on("mousemove", "edge", (e) => {
    const ed = e.target;
    showTip(`<div>${esc(ed.source().data("label"))} → ${esc(ed.target().data("label"))}</div><div class="t">weight <b>${esc(ed.data("wlabel"))}</b></div>`, e.originalEvent.clientX, e.originalEvent.clientY);
  });
  cy.on("mouseout", "edge", hideTip);
  cyEl.addEventListener("mouseleave", () => { hideTip(); if (!pinned) highlight(null); });
  cy.on("tap", "node", (e) => select(e.target));
  cy.on("tap", (e) => { if (e.target === cy) select(null); });

  $(".mode", host).onchange = (e) => { mode = e.target.value; applyMode(); };
  $(".dir", host).onchange = (e) => { dir = e.target.value; relayout(); };
  $(".fit", host).onclick = () => cy.fit(undefined, 20);
  $(".wlabels", host).onclick = (e) => { showWeights = !showWeights; e.target.setAttribute("aria-pressed", String(showWeights)); cy.style(style()); };
  $(".tallbtn", host).onclick = (e) => {
    const on = !cyEl.classList.contains("tall"); cyEl.classList.toggle("tall", on); e.target.setAttribute("aria-pressed", String(on));
    cyEl.style.height = ""; if (on) { cy.resize(); cy.fit(undefined, 20); } else initialView();
  };
  $(".png", host).onclick = () => {
    const a = document.createElement("a");
    a.href = cy.png({ full: true, scale: 2, bg: cssVar("--graph-bg") });
    a.download = `${current?.run?.displayName || "dpg"}_${v.key}.png`; a.click();
  };
  const orig = $(".orig", host);
  if (orig) orig.onclick = () => {
    const box = $(".orig-images", host), on = box.hidden;
    box.hidden = !on; orig.setAttribute("aria-pressed", String(on));
    if (on && !box.innerHTML) box.innerHTML = v.images.map((i) => `<figure><figcaption>${esc(i.name)} · <a href="${esc(i.url)}" target="_blank" rel="noopener">open full size ↗</a></figcaption><img src="${esc(i.url)}" alt="${esc(i.name)}" loading="lazy"></figure>`).join("");
  };
  let searchT;
  $(".search", host).oninput = (e) => {
    clearTimeout(searchT);
    searchT = setTimeout(() => {
      const q = e.target.value.trim().toLowerCase();
      cy.nodes().removeClass("match");
      if (!q) return;
      const hits = cy.nodes().filter((n) => n.data("label").toLowerCase().includes(q));
      hits.addClass("match");
      if (hits.length) cy.animate({ fit: { eles: hits, padding: 60 }, duration: 250 });
    }, 150);
  };

  // Open with communities colouring when this variant has them (mirrors the *_communities PNG).
  if (comm) { mode = "community"; $(".mode", host).value = "community"; }
  applyMode(); details(null); initialView();

  // ---- edge-weight threshold: hide edges below it, then nodes left without a visible edge ----
  let threshold = wMin;
  const thrRange = $(".thr-range", host), thrVal = $(".thr-val", host);
  function applyThreshold(t) {
    threshold = t;
    // Slider sits on the first distinct weight >= t (the end if t is above this graph's range).
    const idx = uniqW.findIndex((w) => w >= t);
    thrRange.value = String(idx < 0 ? uniqW.length - 1 : idx);
    cy.batch(() => {
      cy.edges().forEach((e) => e.toggleClass("pruned", e.data("weight") < t));
      cy.nodes().forEach((n) => n.toggleClass("pruned", t > wMin && !n.connectedEdges().some((e) => !e.hasClass("pruned"))));
    });
    const ve = cy.edges().filter((e) => !e.hasClass("pruned")).length, vn = cy.nodes().filter((n) => !n.hasClass("pruned")).length;
    thrVal.innerHTML = t <= wMin
      ? `<span class="muted">showing all</span>`
      : `≥ <b>${fmtNum(t)}</b> <span class="muted">· ${ve}/${cy.edges().length} edges · ${vn}/${cy.nodes().length} nodes</span>`;
  }
  thrRange.oninput = () => {
    const t = uniqW[+thrRange.value] ?? wMin;
    applyThreshold(t);
    if (onThreshold) onThreshold(t <= wMin ? -Infinity : t); // at the minimum = "no threshold" for the peer too
  };
  const lockBtn = $(".thr-lock", host);
  const drawLock = () => {
    const on = lock.get();
    // Padlock drawn inline (emoji fonts aren't everywhere); open shackle when unlocked.
    const shackle = on ? "M5 7V5a3 3 0 0 1 6 0v2" : "M5 7V5a3 3 0 0 1 5.8-1";
    lockBtn.innerHTML = `<svg viewBox="0 0 16 16" width="13" height="13" aria-hidden="true" style="vertical-align:-2px;margin-right:5px"><rect x="3" y="7" width="10" height="7" rx="1.5" fill="currentColor"/><path d="${shackle}" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"/></svg>${on ? "Synced" : "Independent"}`;
    lockBtn.setAttribute("aria-pressed", String(on));
    lockBtn.title = on ? "Both graphs use the same minimum edge weight. Click to set them separately." : "Each graph has its own minimum edge weight. Click to sync the grouped graph to this one.";
  };
  if (lockBtn) { drawLock(); lockBtn.onclick = () => { lock.toggle(); drawLock(); }; }
  applyThreshold(wMin);

  const rethemer = () => { cy.style(style()); applyMode(); };
  themedRenderers.push(rethemer);
  return { destroy: () => { hideTip(); const i = cyInstances.indexOf(cy); if (i >= 0) cyInstances.splice(i, 1); cy.destroy(); }, rethemer, syncSelect, ownLabel: () => ownLabel, setThreshold: (t) => applyThreshold(t), threshold: () => (threshold <= wMin ? -Infinity : threshold), refit: () => { cyEl.style.height = ""; cy.resize(); initialView(); } };
}

// ---------------------------------------------------------------------------
// metrics (history + system events)
// ---------------------------------------------------------------------------

function numericSeries(rows, xKey) {
  const keys = new Set(); rows.forEach((r) => Object.keys(r).forEach((k) => keys.add(k)));
  const out = [];
  for (const k of keys) {
    if (k.startsWith("_")) continue;
    if (k === xKey) continue;
    const pts = rows.filter((r) => typeof r[k] === "number" && isFinite(r[k]) && typeof r[xKey] === "number").map((r) => [r[xKey], r[k]]);
    if (pts.length >= 2) out.push({ key: k, pts });
  }
  return out.sort((a, b) => a.key.localeCompare(b.key));
}

function plotLine(el, pts, xTitle, color) {
  const draw = () => {
    const ink = cssVar("--text-secondary"), grid = cssVar("--grid");
    Plotly.react(el, [{
      x: pts.map((p) => p[0]), y: pts.map((p) => p[1]), type: "scatter", mode: pts.length <= 40 ? "lines+markers" : "lines",
      line: { color: color(), width: 2 }, marker: { size: 8, color: color(), line: { color: cssVar("--surface-1"), width: 2 } },
      hovertemplate: `${xTitle} %{x}<br><b>%{y:.4~g}</b><extra></extra>`,
    }], {
      margin: { l: 48, r: 12, t: 6, b: 34 }, paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
      font: { color: ink, size: 11, family: "ui-sans-serif, system-ui, sans-serif" },
      xaxis: { title: { text: xTitle, standoff: 4 }, gridcolor: grid, zeroline: false, linecolor: grid },
      yaxis: { gridcolor: grid, zeroline: false, linecolor: grid },
      hovermode: "x", hoverlabel: { bgcolor: cssVar("--surface-1"), bordercolor: cssVar("--border-strong"), font: { color: cssVar("--text-primary") } },
      showlegend: false,
    }, { displaylogo: false, responsive: true, modeBarButtonsToRemove: ["lasso2d", "select2d"] });
  };
  draw(); themedRenderers.push(draw);
}

function chartGrid(host, series, xTitle) {
  const grid = document.createElement("div"); grid.className = "charts"; host.appendChild(grid);
  for (const s of series) {
    const box = document.createElement("div"); box.className = "chart"; box.dataset.key = s.key.toLowerCase();
    box.innerHTML = `<h3>${esc(s.key)}</h3><div class="plot"></div>`;
    grid.appendChild(box);
    plotLine($(".plot", box), s.pts, xTitle, () => cssVar("--series-1"));
  }
}

// Charts grouped by key prefix ("train/loss" -> "train"); a group renders its plots on first open.
function chartGroups(host, series, xTitle, { openFirst = 2, strip = null } = {}) {
  const groups = new Map();
  for (const s of series) {
    const key = strip ? s.key.replace(strip, "") : s.key;
    const g = key.includes("/") ? key.split("/")[0] : key.includes(".") ? key.split(".")[0] : "(ungrouped)";
    if (!groups.has(g)) groups.set(g, []);
    groups.get(g).push({ ...s, key });
  }
  const flat = groups.size > 1 && [...groups.values()].every((g) => g.length === 1);
  if (groups.size === 1 || flat || series.length <= 6) { chartGrid(host, series.map((s) => ({ ...s, key: strip ? s.key.replace(strip, "") : s.key })), xTitle); return; }
  let i = 0;
  for (const [name, items] of groups) {
    const det = document.createElement("details"); det.className = "art";
    det.innerHTML = `<summary><span class="art-name">${esc(name)}</span><span class="muted small">${items.length} chart${items.length === 1 ? "" : "s"}</span></summary><div class="body"></div>`;
    host.appendChild(det);
    const draw = () => { const b = $(".body", det); if (!det.open || b.dataset.done) return; b.dataset.done = 1; chartGrid(b, items, xTitle); };
    det.addEventListener("toggle", draw);
    if (i++ < openFirst) det.open = true;
  }
}

function renderMetrics(el, d) {
  el.innerHTML = `<h2>Metrics <span class="muted">${d.history.length} history rows · ${d.events.length} system samples</span></h2><div class="body"></div>`;
  const body = $(".body", el);
  const hist = numericSeries(d.history, "_step");
  const single = [];
  const histKeys = new Set(hist.map((h) => h.key));
  const allKeys = new Set(); d.history.forEach((r) => Object.keys(r).forEach((k) => allKeys.add(k)));
  for (const k of allKeys) {
    if (k.startsWith("_") || histKeys.has(k)) continue;
    const vals = d.history.filter((r) => typeof r[k] === "number");
    if (vals.length === 1) single.push(k);
  }
  if (hist.length) {
    body.insertAdjacentHTML("beforeend", `<div class="tbl-tools"><input type="text" class="mfilter" placeholder="Filter ${hist.length} metrics…" aria-label="Filter metrics"><span class="muted small mcount"></span></div><div class="subhead">Logged metrics (vs step)</div>`);
    const holder = document.createElement("div"); body.appendChild(holder);
    const draw = (q) => {
      holder.innerHTML = "";
      const list = q ? hist.filter((h) => h.key.toLowerCase().includes(q)) : hist;
      $(".mcount", body).textContent = q ? `${list.length} match` : "";
      if (list.length) (q ? chartGrid(holder, list, "step") : chartGroups(holder, list, "step"));
      else holder.innerHTML = `<p class="muted small">No metric matches.</p>`;
    };
    let t; $(".mfilter", body).oninput = (e) => { clearTimeout(t); t = setTimeout(() => draw(e.target.value.trim().toLowerCase()), 200); };
    draw("");
  }
  if (single.length) body.insertAdjacentHTML("beforeend", `<p class="muted small">Logged once, so no chart: ${single.slice(0, 40).map((k) => `<span class="mono">${esc(k)}</span>`).join(", ")}${single.length > 40 ? ` and ${single.length - 40} more` : ""}. Their values are in the Summary tiles above.</p>`);
  if (!hist.length && !single.length) body.insertAdjacentHTML("beforeend", `<p class="muted">No numeric history logged.</p>`);

  const sys = numericSeries(d.events, "_runtime");
  if (sys.length) {
    const det = document.createElement("details"); det.className = "art"; det.style.marginTop = "16px";
    det.innerHTML = `<summary><span class="art-name">System metrics</span><span class="muted small">${sys.length} series · vs runtime (s)</span></summary><div class="body"></div>`;
    body.appendChild(det);
    det.addEventListener("toggle", () => { const b = $(".body", det); if (!det.open || b.dataset.done) return; b.dataset.done = 1; chartGroups(b, sys, "runtime (s)", { openFirst: 0, strip: /^system\./ }); });
  } else body.insertAdjacentHTML("beforeend", `<p class="muted small">No system metrics recorded for this run.</p>`);
}

// ---------------------------------------------------------------------------
// tables
// ---------------------------------------------------------------------------

function renderCell(v) {
  if (v === null || v === undefined) return { html: '<span class="muted">—</span>', n: false };
  if (typeof v === "boolean") return { html: v ? '<span class="ok">✓ true</span>' : '<span class="bad">✕ false</span>', n: false };
  if (typeof v === "number") return { html: fmtNum(v), n: true };
  if (typeof v === "object") return { html: `<span class="obj mono">${esc(JSON.stringify(v))}</span>`, n: false };
  const s = String(v);
  if (/^https?:\/\/storage\.googleapis\.com\//.test(s)) return { html: `<a href="${esc(s)}" target="_blank" rel="noopener">download</a>`, n: false };
  if (/^https?:\/\//.test(s)) return { html: `<a href="${esc(s)}" target="_blank" rel="noopener">${esc(s.replace(/^https?:\/\/(www\.)?/, "").slice(0, 60))}</a>`, n: false };
  return { html: esc(s), n: false };
}

function dataTable(host, columns, rows) {
  host.insertAdjacentHTML("beforeend", `<div class="tbl-tools"><input type="text" placeholder="Filter rows…" aria-label="Filter rows"><span class="muted small cnt"></span></div><div class="tbl-wrap"><table><thead></thead><tbody></tbody></table></div>`);
  const tools = host.lastElementChild.previousElementSibling, wrap = host.lastElementChild;
  const numCol = columns.map((_, i) => rows.length > 0 && rows.every((r) => r[i] == null || typeof r[i] === "number"));
  let sortI = -1, sortDir = 1, filter = "";
  const thead = $("thead", wrap), tbody = $("tbody", wrap);
  const drawHead = () => {
    thead.innerHTML = `<tr>${columns.map((c, i) => `<th class="${numCol[i] ? "n" : ""}" data-i="${i}">${esc(c)}${sortI === i ? (sortDir > 0 ? " ▲" : " ▼") : ""}</th>`).join("")}</tr>`;
    thead.querySelectorAll("th").forEach((th) => (th.onclick = () => { const i = +th.dataset.i; sortDir = sortI === i ? -sortDir : 1; sortI = i; drawHead(); drawBody(); }));
  };
  const drawBody = () => {
    let rs = rows;
    if (filter) rs = rs.filter((r) => r.some((c) => String(c ?? "").toLowerCase().includes(filter)));
    if (sortI >= 0) rs = [...rs].sort((a, b) => { const x = a[sortI], y = b[sortI]; return (x == null) - (y == null) || (x < y ? -1 : x > y ? 1 : 0) * sortDir; });
    const shown = rs.slice(0, 2000);
    tbody.innerHTML = shown.map((r) => `<tr>${r.map((c) => { const { html, n } = renderCell(c); return `<td class="${n ? "n" : ""}">${html}</td>`; }).join("")}</tr>`).join("");
    $(".cnt", tools).textContent = `${rs.length} of ${rows.length} rows${rs.length > shown.length ? " (first 2000 shown)" : ""}`;
  };
  $("input", tools).oninput = (e) => { filter = e.target.value.trim().toLowerCase(); drawBody(); };
  drawHead(); drawBody();
}

function renderTables(el, d) {
  el.innerHTML = `<h2>Tables</h2>`;
  for (const [k, m] of Object.entries(d.media)) {
    if (m._type !== "table-file") continue;
    el.insertAdjacentHTML("beforeend", `<div class="subhead">${esc(k)} <span class="muted">${m.nrows ?? "?"} × ${m.ncols ?? "?"}</span></div>`);
    const box = document.createElement("div"); el.appendChild(box);
    if (m.table?.columns) dataTable(box, m.table.columns, m.table.data || []);
    else box.innerHTML = `<p class="muted">Could not load table. <a href="${esc(m.url)}" target="_blank" rel="noopener">Download JSON</a></p>`;
  }
}

function renderMedia(el, used) {
  const imgs = Object.entries(current.media).filter(([k, m]) => m._type === "image-file" && !used.has(k));
  el.innerHTML = `<h2>Media <span class="muted">${imgs.length} images</span></h2><div class="gallery">${imgs.map(([k, m]) =>
    `<figure><a href="${esc(m.url)}" target="_blank" rel="noopener"><img src="${esc(m.url)}" alt="${esc(k)}" loading="lazy"></a><figcaption>${esc(k)}${m.width ? ` · ${m.width}×${m.height}` : ""}</figcaption></figure>`).join("")}</div>`;
}

// ---------------------------------------------------------------------------
// artifacts, config, files
// ---------------------------------------------------------------------------

function filePreview(host, f) {
  if (f.content == null) return;
  if (/\.csv$/i.test(f.name)) {
    const rows = parseCSV(f.content); if (!rows.length) return;
    let head = rows[0], data = rows.slice(1).map((r) => r.map(numOrStr));
    if (head[0] === "") { head = head.slice(1); data = data.map((r) => r.slice(1)); } // drop pandas index
    dataTable(host, head, data);
  } else if (/\.json$/i.test(f.name)) {
    let pretty = f.content; try { pretty = JSON.stringify(JSON.parse(f.content), null, 2); } catch { /* raw */ }
    host.insertAdjacentHTML("beforeend", `<pre class="text">${esc(pretty.length > 200000 ? pretty.slice(0, 200000) + "\n…" : pretty)}</pre>`);
  } else host.insertAdjacentHTML("beforeend", `<pre class="text">${esc(f.content)}</pre>`);
}

function renderArtifacts(el, d) {
  const arts = [...d.artifacts].sort((a, b) => a.type.localeCompare(b.type) || a.name.localeCompare(b.name));
  el.innerHTML = `<h2>Artifacts <span class="muted">${arts.length} logged${d.inputArtifacts.length ? ` · ${d.inputArtifacts.length} used` : ""}</span></h2>`;
  for (const a of arts) {
    const det = document.createElement("details"); det.className = "art";
    const size = a.files.reduce((s, f) => s + (f.size || 0), 0);
    det.innerHTML = `<summary><span class="type-badge">${esc(a.type)}</span><span class="art-name">${esc(a.name)}</span><span class="muted small">v${esc(a.version)} · ${a.files.length} file${a.files.length === 1 ? "" : "s"} · ${fmtBytes(size)}</span></summary><div class="body"></div>`;
    el.appendChild(det);
    det.addEventListener("toggle", () => {
      const body = $(".body", det); if (!det.open || body.dataset.done) return; body.dataset.done = 1;
      for (const f of a.files) {
        body.insertAdjacentHTML("beforeend", `<div class="file-row"><span class="mono">${esc(f.name)}</span><span class="muted small">${fmtBytes(f.size)}</span><a class="small" href="${esc(f.url)}" target="_blank" rel="noopener">download</a></div>`);
        if (/\.(png|jpe?g|gif|svg|webp)$/i.test(f.name)) body.insertAdjacentHTML("beforeend", `<div class="orig-images"><figure><img src="${esc(f.url)}" alt="${esc(f.name)}" loading="lazy"></figure></div>`);
        else filePreview(body, f);
      }
    });
  }
  if (d.inputArtifacts.length) {
    el.insertAdjacentHTML("beforeend", `<div class="subhead">Used (inputs)</div><ul>${d.inputArtifacts.map((a) => `<li><span class="type-badge">${esc(a.type)}</span> ${esc(a.name)} <span class="muted">v${a.version}</span></li>`).join("")}</ul>`);
  }
}

function kvTable(host, obj) {
  const rows = Object.entries(obj);
  host.insertAdjacentHTML("beforeend", `<div class="tbl-wrap"><table><tbody>${rows.map(([k, v]) => {
    const { html, n } = renderCell(v); return `<tr><td class="k mono">${esc(k)}</td><td class="${n ? "n" : ""}">${html}</td></tr>`;
  }).join("") || '<tr><td class="muted">empty</td></tr>'}</tbody></table></div>`);
}

function renderConfig(el, d) {
  el.innerHTML = `<h2>Config <span class="muted">${Object.keys(d.config).length} keys</span></h2>`;
  kvTable(el, d.config);
  const other = Object.fromEntries(Object.entries(d.summary).filter(([k, v]) => !(typeof v === "number" || typeof v === "string" || typeof v === "boolean") || k.startsWith("_")));
  if (Object.keys(other).length) { el.insertAdjacentHTML("beforeend", `<div class="subhead">Other summary fields</div>`); kvTable(el, other); }
}

function renderFiles(el, d) {
  el.innerHTML = `<h2>Files &amp; logs <span class="muted">${d.files.length} files</span></h2>`;
  const meta = d.files.find((f) => f.name === "wandb-metadata.json");
  if (meta?.content) {
    try {
      const m = JSON.parse(meta.content);
      const pick = Object.fromEntries(Object.entries(m).filter(([, v]) => v !== "" && v != null));
      el.insertAdjacentHTML("beforeend", `<div class="subhead">Environment (wandb-metadata.json)</div>`); kvTable(el, pick);
    } catch { /* ignore */ }
  }
  const log = d.files.find((f) => f.name === "output.log");
  if (log?.content != null) el.insertAdjacentHTML("beforeend", `<div class="subhead">output.log</div><pre class="text">${esc(log.content) || '<span class="muted">empty</span>'}</pre>`);
  el.insertAdjacentHTML("beforeend", `<div class="subhead">All files</div>`);
  const box = document.createElement("div"); el.appendChild(box);
  dataTable(box, ["name", "size (bytes)", "type", "link"], d.files.map((f) => [f.name, f.size, f.mimetype, f.url]));
}

// ---------------------------------------------------------------------------
// boot
// ---------------------------------------------------------------------------

updateKeyHint(); renderRecent();
const fromHash = new URLSearchParams(location.hash.slice(1)).get("run");
if (fromHash) { $("#run-input").value = fromHash; loadRun(fromHash); }
