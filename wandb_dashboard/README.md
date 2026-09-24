# W&B run dashboard

A small Flask app that loads a single W&B run by id and shows everything about it.
DPG graphs are rebuilt from their `dpg_structure` JSON artifacts as interactive
Cytoscape graphs instead of being shown as the logged PNGs.

- **Overview**: state, project, group, job type, user, host, timings, commit, tags, and all summary scalars.
- **DPG graphs**: one tab per variant (BASIC / GROUPED / GROUPED-SPLIT / ...).
  - Zoom and pan the graph.
  - Hover a node to trace every path through it; click it for in/out edges with weights.
  - The detail panel also shows community, class probabilities and node metrics.
  - Colour nodes by community or by any node metric (e.g. betweenness).
  - Search nodes, switch between LR and TB layout, export a PNG, or toggle the original image.
- **Metrics**: history charts (grouped by prefix, filterable) and system metrics.
- **Tables**: `wandb.Table`s, sortable and filterable.
- **Artifacts**: every logged artifact, with CSV/JSON/text previews and download links.
- **Config, environment, output.log**, and every run file.

## Run

```bash
cd wandb_dashboard
/usr/bin/python3 -m venv .venv && .venv/bin/pip install flask requests waitress
.venv/bin/python app.py          # http://0.0.0.0:8050
```

Open the page, go to **Settings** and paste your API key (wandb.ai/authorize).
The key is stored in the browser's localStorage and sent with each request in
the `X-Wandb-Key` header. The server passes it on to W&B and never writes it anywhere.
Entity and project are optional: if you leave the project blank, the app searches every
project of the entity. The input also accepts `project/id`, `entity/project/id`
or a full wandb.ai run URL, and `/#run=<ref>` links load a run directly.

Environment variables:

| Variable | Default | Meaning |
|---|---|---|
| `DASHBOARD_HOST` / `DASHBOARD_PORT` | `0.0.0.0` / `8050` | bind address |
| `WANDB_BASE_URL` | `https://api.wandb.ai` | for self-hosted W&B |
| `DASHBOARD_WANDB_API_KEY` | unset | server-side fallback key. Anyone who can reach the port can then read your W&B data, so only set it on a trusted network. |

## Keep it running (systemd)

```ini
# /etc/systemd/system/wandb-dashboard.service
[Unit]
Description=W&B run dashboard (DPG)
After=network-online.target

[Service]
WorkingDirectory=/root/gitgud/temp/DPG/wandb_dashboard
ExecStart=/root/gitgud/temp/DPG/wandb_dashboard/.venv/bin/python app.py
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

```bash
systemctl daemon-reload && systemctl enable --now wandb-dashboard
```

JS libraries (Cytoscape, dagre, cytoscape-dagre, Plotly basic) are vendored in
`static/vendor/`, so the page needs no CDN.
