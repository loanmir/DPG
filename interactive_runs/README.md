# interactive_runs

Single-page Flask website that runs the existing DPG pipeline
(gridsearch + 4 grouping scripts) on demand, displays all 5 PNG outputs
side by side, and persists runs to a separate `interactive_runs/runs/`
directory (alongside `examples/results_cat/` rather than inside it).

## Run it

The grouping scripts use `from DPG.examples.grouping_scripts...` -- so
the parent of this repo has to be on `PYTHONPATH` (that's what makes the
repo importable as the top-level package `DPG`).

```bash
PYTHONPATH=/root/gitgud/temp \
    ./.venv/bin/python interactive_runs/serve.py
```

Then open http://localhost:5050.

## Files

| File | Purpose |
| --- | --- |
| `runner.py` | One-call wrapper around `examples/gridsearch_dpg_catsim.run_one` + the 4 grouping scripts. Importable as `from interactive_runs.runner import run_one`. |
| `app.py` | Flask server: 3 routes (`/`, `/api/run`, `/api/image`) plus `/api/datasets`. |
| `serve.py` | Tiny launcher so you can `python interactive_runs/serve.py`. |
| `templates/index.html` | The page. |
| `static/style.css` | The look. |
| `static/app.js` | The behaviour: debounced auto-rerun on any input change. |
| `runs/` | Generated outputs (one subdir per run, named like `examples/results_cat/`). Gitignored. |

## Pipeline stages (mirrors the manual one-line-per-script flow)

For every form submission, `runner.run_one` executes **all 5 stages in
order**:

1. `examples/gridsearch_dpg_catsim.run_one(...)` -- trains a 10-tree
   `RandomForestClassifier`, builds the DPG, writes the raw DPG PNG/PDF
   + communities PNG/PDF + structure JSON + node/edge CSVs into
   `runs/<run_id>/`.
2. `examples/grouping_scripts/categorical_view_conversion._process_subdir(...)`
   -- one-hot rewrite, writes `runs/<run_id>/wip/<run_id>_DPG.png`.
3. `examples/grouping_scripts/cat_grouping._process_subdir(...)` --
   sequential grouping, writes `runs/<run_id>/wip/<run_id>_DPG_grouped.png`
   + `_structure.json`.
4. `examples/grouping_scripts/cat_grouping_split._process_subdir(...)`
   -- split-then-merge grouping, writes
   `runs/<run_id>/wip/grouping_split/<run_id>_DPG_split_grouped.png`.
5. `examples/grouping_scripts/cat_grouping_split_conjunction._process_subdir(...)`
   -- cross-feature AND variant, writes
   `runs/<run_id>/wip/grouping_split_conjunction/<run_id>_DPG_split_grouped_conjunction.png`.

Any stage that errors does NOT abort the rest -- the errors accumulate
in the JSON response and the `wip/` siblings of the failed stage remain
stale. (This mirrors the explicit design in
`examples/gridsearch_dpg_catsim.run_one` and the standalone grouping
scripts.)

## Form inputs

| Field | Range / values | Default |
| --- | --- | --- |
| `dataset` | Dropdown of `datasets/dummy_dataset/toy_chain_*.csv` | `toy_chain_education_abc.csv` |
| `perc_var` | float, `[0.005, 0.5]`, step 0.005 | `0.01` |
| `decimal_threshold` | int, `[1, 6]`, step 1 | `2` |
| `community_threshold` | float, `[0.05, 1.0]`, step 0.05 | `0.3` |

Changing **any** single field triggers a full re-run of all 5 stages.
The browser side debounces by 250 ms so slider drags don't fire dozens
of requests in a row.

## Defaults match the curated reference run

The four default field values are precisely the parameters encoded in
`examples/results_cat/ds=chain_education_abc_pv=0.01_dt=2_ct=0.3/`, so
loading the page with defaults should reproduce the same artifacts in
`interactive_runs/runs/ds=chain_education_abc_pv=0.01_dt=2_ct=0.3/`.
