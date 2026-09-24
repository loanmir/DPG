"""W&B run dashboard: a thin proxy over the wandb GraphQL API.

The browser sends the user's API key in the ``X-Wandb-Key`` header on every
request; the server never stores it. Small text files (DPG structure JSONs,
CSVs, tables, logs) are fetched server-side and inlined so the page can render
interactive graphs and tables without touching wandb's storage itself.

Run:  .venv/bin/python app.py            (listens on 0.0.0.0:8050)
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlparse

import requests
from flask import Flask, jsonify, request, send_from_directory

WANDB_BASE_URL = os.environ.get("WANDB_BASE_URL", "https://api.wandb.ai").rstrip("/")
# Optional server-side fallback key. Off unless explicitly set in the environment.
FALLBACK_KEY = os.environ.get("DASHBOARD_WANDB_API_KEY")

INLINE_EXTS = (".json", ".txt", ".csv", ".log", ".yaml", ".yml", ".md")
INLINE_MAX_BYTES = 3 * 1024 * 1024
ALLOWED_FILE_HOSTS = ("storage.googleapis.com", "api.wandb.ai")
CACHE_TTL_S = 300

app = Flask(__name__, static_folder="static", static_url_path="/static")
_http = requests.Session()
_cache: dict[tuple, tuple[float, dict]] = {}


class ApiError(Exception):
    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.status = status


@app.errorhandler(ApiError)
def _handle_api_error(exc: ApiError):
    return jsonify({"error": str(exc)}), exc.status


# ---------------------------------------------------------------------------
# wandb helpers
# ---------------------------------------------------------------------------


def _api_key() -> str:
    key = request.headers.get("X-Wandb-Key", "").strip() or FALLBACK_KEY
    if not key:
        raise ApiError("No W&B API key. Open Settings and paste your key.", 401)
    return key


def gql(key: str, query: str, variables: dict | None = None) -> dict:
    try:
        resp = _http.post(
            f"{WANDB_BASE_URL}/graphql",
            json={"query": query, "variables": variables or {}},
            auth=("api", key),
            timeout=60,
        )
    except requests.RequestException as exc:
        raise ApiError(f"Could not reach W&B: {exc}", 502)
    if resp.status_code == 401:
        raise ApiError("W&B rejected the API key (401).", 401)
    if not resp.ok:
        raise ApiError(f"W&B returned HTTP {resp.status_code}: {resp.text[:300]}", 502)
    body = resp.json()
    if body.get("errors"):
        raise ApiError("W&B error: " + "; ".join(e.get("message", "?") for e in body["errors"]), 502)
    return body["data"]


def fetch_text(url: str) -> str | None:
    host = urlparse(url).hostname or ""
    if host not in ALLOWED_FILE_HOSTS:
        return None
    try:
        resp = _http.get(url, timeout=60)
        resp.raise_for_status()
        return resp.content.decode("utf-8", errors="replace")
    except requests.RequestException:
        return None


def _loads(value, default=None):
    if value is None:
        return default
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return default


VIEWER_QUERY = """
query {
  viewer { username entity teams { edges { node { name } } } }
}
"""

PROJECTS_QUERY = """
query($e: String!) {
  entity(name: $e) { projects(first: 200) { edges { node { name } } } }
}
"""

RUN_EXISTS_QUERY = """
query($e: String!, $p: String!, $r: String!) {
  project(name: $p, entityName: $e) { run(name: $r) { id } }
}
"""

RUN_QUERY = """
query($e: String!, $p: String!, $r: String!) {
  project(name: $p, entityName: $e) {
    run(name: $r) {
      id name displayName state createdAt heartbeatAt updatedAt
      group jobType tags notes host commit
      config summaryMetrics historyLineCount
      history(samples: 2000)
      events(samples: 2000)
      user { username name }
      files(first: 500) { edges { node { name sizeBytes mimetype directUrl updatedAt } } }
      outputArtifacts(first: 200) {
        edges { node {
          id digest versionIndex createdAt size description
          artifactType { name }
          artifactSequence { name }
          aliases { alias }
          files(first: 50) { edges { node { name sizeBytes directUrl } } }
        } }
      }
      inputArtifacts(first: 100) {
        edges { node {
          versionIndex artifactType { name } artifactSequence { name }
        } }
      }
    }
  }
}
"""


def parse_run_ref(raw: str, entity: str, project: str) -> tuple[str, str, str]:
    """Accept ``id``, ``entity/project/id``, ``project/id`` or a wandb.ai URL."""
    raw = raw.strip().strip("/")
    m = re.search(r"wandb\.ai/([^/]+)/([^/]+)/runs/([^/?#]+)", raw)
    if m:
        return m.group(1), m.group(2), m.group(3)
    parts = [p for p in raw.split("/") if p and p != "runs"]
    if len(parts) >= 3:
        return parts[-3], parts[-2], parts[-1]
    if len(parts) == 2:
        return entity, parts[0], parts[1]
    return entity, project, parts[0] if parts else ""


def locate_run(key: str, entity: str, project: str, run_id: str) -> tuple[str, str]:
    if not entity:
        entity = gql(key, VIEWER_QUERY)["viewer"]["entity"]
    if project:
        return entity, project
    projects = [
        e["node"]["name"]
        for e in (gql(key, PROJECTS_QUERY, {"e": entity})["entity"] or {"projects": {"edges": []}})["projects"]["edges"]
    ]

    def has_run(p):
        try:
            return p if gql(key, RUN_EXISTS_QUERY, {"e": entity, "p": p, "r": run_id})["project"]["run"] else None
        except ApiError:
            return None

    with ThreadPoolExecutor(max_workers=8) as pool:
        for hit in pool.map(has_run, projects):
            if hit:
                return entity, hit
    raise ApiError(f"Run '{run_id}' not found in any project of entity '{entity}'.", 404)


def build_run(key: str, entity: str, project: str, run_id: str) -> dict:
    data = gql(key, RUN_QUERY, {"e": entity, "p": project, "r": run_id})
    run = (data.get("project") or {}).get("run")
    if not run:
        raise ApiError(f"Run '{run_id}' not found in {entity}/{project}.", 404)

    config = {
        k: (v.get("value") if isinstance(v, dict) and "value" in v else v)
        for k, v in (_loads(run["config"], {}) or {}).items()
        if k != "_wandb"
    }
    wandb_meta = (_loads(run["config"], {}) or {}).get("_wandb", {}).get("value", {})
    summary = _loads(run["summaryMetrics"], {}) or {}
    history = [_loads(row, {}) for row in run.get("history") or []]
    events = [_loads(row, {}) for row in run.get("events") or []]

    files = [e["node"] for e in run["files"]["edges"]]
    files_by_name = {f["name"]: f for f in files}

    artifacts = []
    for e in run["outputArtifacts"]["edges"]:
        n = e["node"]
        seq = n["artifactSequence"]["name"]
        artifacts.append({
            "name": seq,
            "type": n["artifactType"]["name"],
            "version": n["versionIndex"],
            "aliases": [a["alias"] for a in n.get("aliases") or []],
            "createdAt": n.get("createdAt"),
            "size": n.get("size"),
            "description": n.get("description"),
            # Artifact names are "<VARIANT_DIR>__<file stem>" in this repo's logger.
            "variant": seq.split("__", 1)[0] if "__" in seq else None,
            "files": [
                {"name": f["node"]["name"], "size": f["node"]["sizeBytes"], "url": f["node"]["directUrl"]}
                for f in n["files"]["edges"]
            ],
        })
    input_artifacts = [
        {"name": e["node"]["artifactSequence"]["name"], "type": e["node"]["artifactType"]["name"],
         "version": e["node"]["versionIndex"]}
        for e in run["inputArtifacts"]["edges"]
    ]

    # Summary media (images, tables, ...) point at run files by path.
    media = {}
    for k, v in summary.items():
        if isinstance(v, dict) and v.get("_type") and v.get("path") in files_by_name:
            media[k] = {**v, "url": files_by_name[v["path"]]["directUrl"]}

    # Inline small text content (artifact files, summary tables, logs).
    jobs = []
    for art in artifacts:
        if art["type"] == "wandb-history":
            continue
        for f in art["files"]:
            if f["name"].lower().endswith(INLINE_EXTS) and (f["size"] or 0) <= INLINE_MAX_BYTES:
                jobs.append((f, f["url"]))
    for m in media.values():
        if m["_type"] == "table-file":
            jobs.append((m, m["url"]))
    for name in ("output.log", "wandb-metadata.json", "requirements.txt"):
        f = files_by_name.get(name)
        if f and (f["sizeBytes"] or 0) <= INLINE_MAX_BYTES:
            jobs.append((f, f["directUrl"]))
    with ThreadPoolExecutor(max_workers=12) as pool:
        for (target, _), text in zip(jobs, pool.map(lambda j: fetch_text(j[1]), jobs)):
            target["content"] = text

    for m in media.values():
        if m["_type"] == "table-file":
            m["table"] = _loads(m.pop("content", None))

    return {
        "entity": entity,
        "project": project,
        "url": f"https://wandb.ai/{entity}/{project}/runs/{run['name']}",
        "run": {
            k: run.get(k)
            for k in ("id", "name", "displayName", "state", "createdAt", "heartbeatAt", "updatedAt",
                      "group", "jobType", "tags", "notes", "host", "commit", "historyLineCount")
        } | {"user": run.get("user"), "wandb": wandb_meta},
        "config": config,
        "summary": {k: v for k, v in summary.items() if k not in media},
        "media": media,
        "history": history,
        "events": events,
        "files": [
            {"name": f["name"], "size": f["sizeBytes"], "mimetype": f["mimetype"], "url": f["directUrl"],
             "content": f.get("content")}
            for f in files
        ],
        "artifacts": artifacts,
        "inputArtifacts": input_artifacts,
        "fetchedAt": time.time(),
    }


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.get("/api/whoami")
def whoami():
    key = _api_key()
    viewer = gql(key, VIEWER_QUERY)["viewer"]
    entities = [viewer["entity"]] + [
        t["node"]["name"] for t in viewer["teams"]["edges"] if t["node"]["name"] != viewer["entity"]
    ]
    projects = {}
    for ent in entities:
        try:
            projects[ent] = [e["node"]["name"] for e in gql(key, PROJECTS_QUERY, {"e": ent})["entity"]["projects"]["edges"]]
        except (ApiError, TypeError):
            projects[ent] = []
    return jsonify({"username": viewer["username"], "entity": viewer["entity"], "projects": projects})


@app.get("/api/run")
def get_run():
    key = _api_key()
    entity, project, run_id = parse_run_ref(
        request.args.get("run", ""), request.args.get("entity", "").strip(), request.args.get("project", "").strip()
    )
    if not re.fullmatch(r"[A-Za-z0-9_\-]+", run_id or ""):
        raise ApiError("Enter a run id like 'xx5iw3ft' (or paste a wandb.ai run URL).")
    entity, project = locate_run(key, entity, project, run_id)

    cache_key = (hashlib.sha256(key.encode()).hexdigest(), entity, project, run_id)
    hit = _cache.get(cache_key)
    if hit and time.time() - hit[0] < CACHE_TTL_S and request.args.get("refresh") != "1":
        return jsonify(hit[1])
    result = build_run(key, entity, project, run_id)
    _cache[cache_key] = (time.time(), result)
    return jsonify(result)


if __name__ == "__main__":
    from waitress import serve

    host = os.environ.get("DASHBOARD_HOST", "0.0.0.0")
    port = int(os.environ.get("DASHBOARD_PORT", "8050"))
    print(f"W&B dashboard on http://{host}:{port}")
    serve(app, host=host, port=port, threads=8)
