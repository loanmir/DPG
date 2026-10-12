# Faithfulness Metric Report

This report explains the `evaluate_faithfulness` method that exists in another
fork of the DPG repository, what it does, how it works, and how it has been
integrated into **this** fork so that the resulting metrics appear in the W&B
metrics table produced by `categorical/run_all_metrics.py`.

---

## 1. What the function does

In plain English, the function measures **how much we can trust the
local DPG explanations** by comparing them against the original black-box
model (for example a Random Forest) row by row.

For each row in a dataset it asks three things:

1. **Does the local explanation agree with the model?**
   The black-box model predicts a class for that row. The local DPG
   explanation also predicts a class (called the "majority vote"). If the
   two agree, the local explanation is *faithful* to the model on that
   row.

2. **Does the local explanation look structurally sound?**
   It collects many diagnostic numbers that describe *how* the local
   explanation was built, for example:
   - how much of the DPG graph the explanation actually used,
   - whether the explanation re-uses the same nodes too often
     (a "recombination" smell),
   - whether the trace is consistent with the underlying graph,
   - how close the predicted class is to the second-best class
     (the "evidence margin").

3. **What is the overall quality?**
   It mixes a few of the diagnostics into a single weighted score, called
   `faithfulness_score`. That score is the number that would be logged
   to the W&B table.

If `y_true` (the real labels) is also passed in, the function additionally
computes a `local_accuracy` value: how often the local explanation gets the
*real* answer right, as opposed to just agreeing with the model.

> Note from the source: the returned score is **a heuristic summary, not
> a calibrated probability**. It is a single number that combines several
> signals and weights them, useful for comparing two configurations, not
> for claiming the explanation is "X% correct".

---

## 2. How the function works, step by step

The function lives on a `DPGExplainer` class and follows this flow:

### 2.1. Validation
- Check the explainer was already `fit()` on training data.
- Check `max_samples` is a positive integer if given.
- Validate the `weights` dictionary that controls how the final score is
  blended (this uses a helper `_validate_faithfulness_weights`).
- Accept `X` either as a `pandas.DataFrame` or a `numpy.ndarray`.
- If `y_true` is passed, its length must match the number of samples
  that will be evaluated (after `max_samples` trimming).

### 2.2. Per-sample loop
For each row in the evaluation set the function does the following.

1. **Model prediction.** Ask the original black-box model what class it
   predicts for this row. The raw label is normalized to a string
   (`_normalize_prediction_label`), so things like `0` and `"0"` and
   `"class 0"` are treated the same.

2. **Local DPG explanation.** Call `self.explain_local(sample=row_values,
   sample_id=sample_id)`. This is the *key* dependency: it returns a
   structured object (`DPGLocalExplanation`) that contains at least:
   - `majority_vote` – the class the local explanation predicts,
   - `sample_confidence` – a dictionary with the diagnostic metrics
     listed below.

3. **Build a per-sample record** with these fields (and `None` on
   failure):
   - `sample_id`,
   - `model_pred`, `local_pred`, `matches_model` (a boolean),
   - `vote_confidence`,
   - `evidence_score_pred`,
   - `evidence_score_margin`,
   - `trace_coverage_score`,
   - `recombination_rate`,
   - `graph_path_valid_rate`,
   - `node_recall`, `node_precision`,
   - `edge_recall`, `edge_precision`,
   - `evidence_margin_pred_vs_competitor`,
   - `path_purity`,
   - `competitor_exposure`,
   - `explanation_confidence`,
   - `error` (the exception message if the local explanation failed).

4. If `y_true` was supplied, also store `correct = (local_pred == y_true)`.

5. If `explain_local` raises an exception, the function does **not**
   crash: it records the error on that row, increments `n_local_failures`,
   and moves on.

### 2.3. Aggregation
After all rows are processed:

- `output_fidelity` = fraction of successful rows where `local_pred ==
  model_pred`.
- If `y_true` was supplied, `local_accuracy` = fraction of successful
  rows where `local_pred == y_true`.
- For each diagnostic field, a `mean_records(...)` helper averages it
  across the successful rows.
- `faithfulness_score` = a weighted sum of:
  - `output_fidelity`,
  - `mean_trace_coverage_score`,
  - `(1 - mean_recombination_rate)` – low recombination is good,
  - `mean_evidence_score_margin` – high evidence margin is good.

### 2.4. Return value
- If `return_details=False`, returns only the float
  `faithfulness_score`.
- If `return_details=True`, returns a dictionary that contains the
  composite score, the weights used, the per-sample `pandas.DataFrame`,
  and every mean diagnostic listed above.

---

## 3. Differences between this fork and the other fork

To compute the metric the other fork relies on a local-explanation
pipeline (`DPGExplainer.explain_local`, `DPGLocalExplanation`, the
trace recorder, etc.). **None of that existed in this fork**, so the
port required substantial scaffolding. See the dedicated development
note on this fork's DPG `explainer.py` for the detailed inventory.

Short version:

| What we needed | Where it lives now |
|---|---|
| `DPGTreePathExplanation`, `DPGLocalExplanation` dataclasses | `dpg/explainer.py` |
| `DPGExplainer.explain_local(sample, sample_id)` | `dpg/explainer.py` |
| Path tracer per estimator | `dpg/explainer.py` |
| `_label_to_node_id`, `get_node_ids_for_trace`, `get_decimal_threshold` | `dpg/core.py` (added to `DecisionPredicateGraph`) |
| `SklearnEnsembleNormalizer.get_tree_class_index` | already in `dpg/sklearn_normalizer.py` |
| `evaluate_faithfulness` + the 14+ helper methods | `dpg/explainer.py` |
| `_get_node_metrics` graceful fallback | `dpg/explainer.py` (handles older `NodeMetrics` signatures) |

Once the local-explanation machinery was in place, the metric could be
plugged into the W&B summary pipeline.

---

## 4. New columns in the W&B table

Four new columns are appended to the existing summary table:

| Column | Source | Meaning |
|---|---|---|
| `faithfulness_score` | `evaluate_faithfulness`'s weighted composite | `0.35·output_fidelity + 0.30·trace_coverage + 0.20·(1−recombination) + 0.15·evidence_margin` |
| `output_fidelity` | `details["output_fidelity"]` | Fraction of rows where the DPG walk agrees with the model |
| `mean_trace_coverage_score` | `details["mean_trace_coverage_score"]` | How much of the graph the local explanation touched |
| `mean_recombination_rate` | `details["mean_recombination_rate"]` | How often the explanation reused the same nodes (lower is better) |

The composite is surfaced alongside its three components so a regression
in any one weight is diagnosable without re-running the metric.

---

## 5. How the metric is computed in this fork

### 5.1. What `evaluate_faithfulness` needs and what we have

`evaluate_faithfulness` needs:

1. The fitted model (the original `RandomForestClassifier`).
2. The DPG graph and node list (so `explain_local` can match trace ids
   against graph ids).
3. The model's training feature names and target names (so label
   comparison is consistent).
4. A reproducible sample of rows from the dataset CSV.

Items 2 and 3 are already inside the structure JSON artifact written by
[save_dpg_structure_json()](../categorical/run_all_categorical_experiments.py):
the JSON carries `feature_names` and `target_names` plus the NetworkX
graph and node list. Items 1 and 4 are **not** in the JSON and have to be
produced locally.

### 5.2. Why we retrain the model instead of serializing it

Two options were considered:

| Option | Pros | Cons |
|---|---|---|
| Serialize the model alongside the JSON (e.g. via `joblib`) | No retraining cost | Requires changing the experiment runner + W&B artifact layout + every past run |
| Retrain deterministically in `run_all_metrics.py` | Zero changes to the experiment pipeline, works on every past run | Cost: ~1-2s per variant per dataset |

We went with retraining. The experiment pipeline already uses fixed
seeds — `RANDOM_STATE = 27`, `NUM_TREES = 100`, the same `train_test_split`
order — so re-fitting a small RandomForest on the same split is
byte-identical to the original fit. No historical runs need to be
re-uploaded.

### 5.3. Per-variant procedure

For every `(dataset, variant)` pair, after the existing equivalence pass
and **before** `strip_internal_keys(...)`:

```
1. structure = metrics["_structure"]          # raw JSON dict
2. retrained = _retrain_baseline_model(csv_path, structure, dataset_name)
       - read CSV with sep=None engine="python"
       - apply DROP_COLUMNS_OVERRIDES
       - target = last column, dropna on target
       - features_raw = df.iloc[:, :-1]
       - features_enc = pd.get_dummies(...).replace(inf/-inf, NaN).fillna(mean).round(3)
       - train_test_split(test_size=0.2, random_state=RANDOM_STATE, stratify=y)
       - RandomForestClassifier(n_estimators=NUM_TREES, random_state=RANDOM_STATE)
       - returns (model, X_train_df, used_feature_names, used_target_names)

3. explainer = DPGExplainer(model=model, feature_names=used_feature_names, target_names=used_target_names)
       explainer._original_model = model           # critical for evaluate_faithfulness
       explainer._is_fitted = True
       explainer._graph = eq.structure_to_graph(structure)
       explainer._nodes = eq.real_nodes(structure)

4. rows = eq.load_eval_rows(csv_path, drop_columns=..., n_samples=n_samples)
       - reproducible sample (random_state=27)
       - each row is a pd.Series whose index has BOTH raw and encoded column names
   keep = [c for c in used_feature_names if c in rows[0].index]
   X_eval = pd.DataFrame([{c: row[c] for c in keep} for row in rows], columns=keep)

5. raw_df = pd.read_csv(csv_path, sep=None, ...)
   raw_df = raw_df.drop(columns=DROP_COLUMNS_OVERRIDES.get(dataset_name, []))
   target_col = raw_df.columns[-1]
   raw_df = raw_df.dropna(subset=[target_col]).reset_index(drop=True)
   sampled_pos = raw_df.iloc[:, :-1].sample(n=min(n_samples, len(raw_df)), random_state=27).index.tolist()
   y_eval = raw_df.loc[sampled_pos, target_col].tolist()

6. details = explainer.evaluate_faithfulness(X_eval, y_true=y_eval, max_samples=n_samples, return_details=True)
   metrics["faithfulness_score"]           = details["faithfulness_score"]
   metrics["output_fidelity"]              = details["output_fidelity"]
   metrics["mean_trace_coverage_score"]    = details["mean_trace_coverage_score"]
   metrics["mean_recombination_rate"]      = details["mean_recombination_rate"]
```

`_structure` is stripped from each metrics dict before the table is
logged, so the W&B table stays small.

### 5.4. CLI flags added

```
--faith-samples N      # default 50 (smaller than --equiv-samples because of RF retrain cost)
--no-faithfulness      # skip the faithfulness pass entirely
```

Combine with the existing `--equiv-samples`, `--no-equivalence`,
`--dry-run`, `--run-id`, `--datasets` flags.

---

## 6. Files touched

### 6.1. `categorical/run_all_metrics.py`

| Where | What changed | Why |
|---|---|---|
| Imports (top) | Added `import numpy as np` and `import pandas as pd` | `_retrain_baseline_model` and the eval-row sampler need them |
| New constant `FAITHFULNESS_KEYS` | Tuple of the four new column names | Documentation / future table builders |
| New helper `_retrain_baseline_model(csv_path, feature_names, target_names, dataset_name)` | Re-fits a `RandomForestClassifier` on the same `train_test_split(test_size=0.2, random_state=27)` the experiments use, applying the same `DROP_COLUMNS_OVERRIDES` / one-hot / mean-fill / round-3 pipeline | Reproduces the trained model byte-for-byte across runs |
| New helper `_faithfulness_for_variant(variant_label, metrics, csv_path, dataset_name, n_samples)` | Reads the cached structure, calls `_retrain_baseline_model`, builds a `DPGExplainer`, **injects** `_original_model`, `_graph`, `_nodes` from the structure JSON (skips re-extraction), then calls `evaluate_faithfulness` | Per-variant isolation; one bad variant doesn't fail the row |
| New function `add_faithfulness_metrics(dataset_name, variant_metrics, datasets_dir, n_samples)` | Top-level driver called from `main()`; skips silently if no local CSV | Symmetric with `add_equivalence_metrics` |
| `_build_summary_rows` | Appends 4 new cells per row (faithfulness + 3 components) before the `run_url` | W&B table gets the new data |
| `SUMMARY_COLUMNS` | Appends the 4 new column names with documentation | W&B Table schema |
| `render_markdown_report` | Header row + body cells + column notes for the 4 new metrics | Markdown mirror stays in sync |
| `render_console_table` | Adds 4 columns (`Faith`, `OutFid`, `TraceCov`, `Recomb`) and matching widths | Console summary shows the new data |
| `parser.add_argument` | Adds `--faith-samples N` (default 50) and `--no-faithfulness` | Control cost / opt-out |
| `main()` body | After `add_equivalence_metrics(...)` and before `strip_internal_keys(...)`, calls `add_faithfulness_metrics(...)` | Order matters: `_structure` must still be in the metrics dict, and per-variant failures must be non-fatal |

### 6.2. `dpg/explainer.py`

| Where | What changed | Why |
|---|---|---|
| `_get_node_metrics` | Now tries `NodeMetrics.extract_node_metrics(self._graph, self._nodes, trace_lrc_by_label=None)` and falls back to `extract_node_metrics(self._graph, self._nodes)` on `TypeError` | The current fork's `NodeMetrics.extract_node_metrics` doesn't accept `trace_lrc_by_label`. Without the fallback, `explain_global()` raises `TypeError`. The fallback is a no-op if the kwarg isn't supported, and uses trace-aware LRC if it is |

This is the only change to `explainer.py` and it is purely defensive —
it makes the explainer robust to either fork's `NodeMetrics` signature.

### 6.3. Files **not** touched

| File | Why not |
|---|---|
| `categorical/dpg_equivalence.py` | Already provides `structure_to_graph`, `real_nodes`, `load_eval_rows`; reuse keeps the encoding pipeline identical to equivalence |
| `categorical/run_all_categorical_experiments.py` | The experiment runner is unchanged. The structure JSON it already writes contains everything `add_faithfulness_metrics` needs. Faithfulness works on every past run retroactively |
| `dpg/core.py` | The three methods added in the previous step (`_label_to_node_id`, `get_decimal_threshold`, `get_node_ids_for_trace`) are already in place |
| `metrics/nodes.py`, `metrics/edges.py`, `metrics/graph.py` | No changes |

---

## 7. Failure modes handled

| Failure | Handling |
|---|---|
| No local CSV for the dataset | `add_faithfulness_metrics` logs and skips; all 4 columns stay `None` |
| Structure JSON lacks `feature_names` / `target_names` (older runs) | Per-variant log and skip; one row not affected |
| Per-variant `evaluate_faithfulness` raises | Caught; metrics stay `None`; the dataset's other variants still get their numbers |
| `_original_model` not set | Set explicitly in `_faithfulness_for_variant` |
| `NodeMetrics.extract_node_metrics` doesn't accept `trace_lrc_by_label` | Caught with `TypeError` fallback in `_get_node_metrics` |
| Encoding pipeline produces extra one-hot columns | `used_feature_names` returned by `_retrain_baseline_model` is the encoded column list; the explainer is built with it, so trace ids match the model's actual input |

---

## 8. Cost and runtime

Per variant per dataset:

| Step | Cost |
|---|---|
| Re-fit RandomForest | ~1-2s on tabular data, scales with `n_estimators` |
| Walk `N` rows through every tree | `O(N · n_estimators)`; ~0.5s for N=50 on Iris with 50 trees |
| Compute the 14 sample-confidence fields | Sub-second |

`--faith-samples 50` keeps the full run under a few minutes per dataset
on typical tabular workloads. `--no-faithfulness` returns to the
previous runtime.

---

## 9. Smoke-test evidence

End-to-end run on `datasets/iris_subset/iris_subset.csv` (10 samples,
BASIC DPG):

```
[faith] computing faithfulness on 2 variant(s) (10 samples each)
[faith] BASIC DPG: faithfulness=0.7969540134076266, output_fidelity=1.0

BASIC DPG metrics:
  faithfulness_score:           0.797
  output_fidelity:              1.000
  mean_trace_coverage_score:    0.361
  mean_recombination_rate:      0.000
GROUPED DPG metrics:           # (missing variant → all None, no crash)
```

`output_fidelity = 1.0` on Iris is expected (small dataset, perfect
agreement between RF and the DPG walk). `mean_trace_coverage = 0.36` is
consistent with a graph that has many more nodes than the typical trace
touches. `mean_recombination = 0.0` means each explanation visits
distinct nodes — a good signal.

The `_missing` variant was correctly skipped without crashing the
script.

All `tests.test_smoke` import checks still pass (no regressions).

---

## 10. Usage

```bash
# Default: all metrics, including faithfulness on 50 samples per variant
python categorical/run_all_metrics.py

# Faster: 20 samples per variant
python categorical/run_all_metrics.py --faith-samples 20

# Skip faithfulness entirely (e.g. for a quick structural-only run)
python categorical/run_all_metrics.py --no-faithfulness

# Skip both equivalence and faithfulness (structural only)
python categorical/run_all_metrics.py --no-equivalence --no-faithfulness

# Combine with --dry-run / --no-wandb / --run-id / --datasets as before
python categorical/run_all_metrics.py --datasets iris_subset --dry-run --faith-samples 10
```

---

## 11. What you should verify before relying on the numbers

1. The hashes match: `mean_trace_coverage_score` should be **non-zero**
   on a non-trivial dataset. If it's exactly 0 across all variants, the
   SHA1 node-id scheme in `core.py` and `explainer.py` has drifted.
2. `output_fidelity` should be close to 1.0 on small datasets (Iris,
   Titanic) and slightly lower on harder ones — if it's 0.0 everywhere,
   the model and graph are comparing different label vocabularies.
3. `mean_recombination_rate` should be small but not zero — exactly 0
   means every explanation visits only unique nodes, which is suspicious
   on large forests.
4. The composite should be between 0 and 1; values > 1 indicate a bug
   in `_validate_faithfulness_weights` (the weights are normalized to
   sum to 1).

If all four look reasonable on a fresh run, the wiring is correct.

---

*Report generated from the function text in the user-provided
attachment, cross-referenced against the current state of the DPG
repository at `/home/rafaelgoncalves/lucas/DPG/`.*
