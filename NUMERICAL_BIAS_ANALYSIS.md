# DPG Numerical Bias Analysis

Scope: analysis of numeric assumptions and bias points in DPG core/metrics/visualization pipeline.

## 1) Predicate representation in the code

### Finding 1.1: Predicates are serialized as numeric threshold comparisons only - PROBLEMATIC
- Description: decision predicates are emitted as string labels with only ordered numeric operators (`<=`, `>`).
- Why this is numerically biased: this representation privileges continuous/ordinal numeric splits and excludes unordered/equality/set predicates in the canonical predicate language.
- Evidence:
  - `DecisionPredicateGraph.tracing_ensemble`: `condition = f"{feature_name} <= {threshold}"` and `condition = f"{feature_name} > {threshold}"` in `DPG/dpg/core.py:201`, `DPG/dpg/core.py:204`
  - `DecisionPredicateGraph.tracing_ensemble_parallel`: same pattern in `DPG/dpg/core.py:244`, `DPG/dpg/core.py:247`

### Finding 1.2: Thresholds and predictions are rounded before becoming node labels - PROBLEMATIC
- Description: thresholds are rounded by `decimal_threshold`, regressor leaf predictions rounded to 2 decimals.
- Why this is numerically biased: quantization changes predicate granularity and can merge nearby but distinct splits.
- Evidence:
  - `threshold = round(tree_.threshold[node_index], self.decimal_threshold)` in `DPG/dpg/core.py:197`, `DPG/dpg/core.py:241`
  - `pred = round(tree_.value[node_index][0][0], 2)` in `DPG/dpg/core.py:187`, `DPG/dpg/core.py:232`
  - default `decimal_threshold` config in `DPG/dpg/core.py:31`, `DPG/dpg/core.py:107`

### Finding 1.3: Demo/training helper rounds custom input data to 2 decimals - PROBLEMATIC
- Description: custom CSV data in sklearn helper is forcibly rounded.
- Why this is numerically biased: upstream quantization can alter split learning and downstream DPG predicates.
- Evidence:
  - `data = np.round(df.values, 2).astype(np.float64)` in `DPG/dpg/sklearn_dpg.py:67`
  - inside **select_dataset()** function, it handles missing data as numerical (mean) + it rounds all the data.
  - Does not have any process for treating the categorical values directly!

## 2) Node canonicalization logic

### Finding 2.1: Canonicalization is text-based (string label -> node identity) - NOT SURE TO BE RELEVANT
- Description: node IDs are `sha1(activity_string)` and deduplicated by exact string match.
- Why this is numerically biased: numeric formatting/rounding directly controls node identity; semantically close numeric predicates can collapse or fragment depending on formatting.
- Evidence:
  - string-hash node IDs in `DPG/dpg/core.py:358`, `DPG/dpg/core.py:366`, `DPG/dpg/core.py:367`
  - deduplication set keyed by predicate string in `DPG/dpg/core.py:353`, `DPG/dpg/core.py:356`

### Finding 2.2: Label-to-ID maps in metrics collapse by label text - NOT SURE TO BE RELEVANT
- Description: metrics build `label -> id` dictionaries, preserving only one ID per label.
- Why this is numerically biased: canonicalization relies on exact string labels and assumes label uniqueness is semantically correct.
- Evidence:
  - `node_label_to_id = {node[1]: node[0] ...}` in `DPG/metrics/graph.py:90`, `DPG/metrics/graph.py:197`

### Finding 2.3: Edge weight parsing is numeric-format fragile - NOT SURE TO BE RELEVANT
- Description: DOT parsing accepts only a simplified numeric shape (`attr.replace(".", "").isdigit()`).
- Why this matters: negative or scientific notation can be dropped as non-numeric and become `None` weights.
- Evidence:
  - `weight = float(attr) if attr.replace(".", "").isdigit() else None` in `DPG/dpg/core.py:396`

## 3) Class constraint extraction

### Finding 3.1: Constraint extraction parser only accepts (`<=`, `>`) + numeric literal - PROBLEMATIC
- Description: parser regex for class-boundary extraction recognizes only two operators and numeric thresholds.
- Why this is numerically biased: class constraints are constrained to scalar threshold logic and cannot encode non-ordered conditions.
- Evidence:
  - `_parse_predicate` regex `(...)(<=|>)(number)` in `DPG/metrics/graph.py:62`, `DPG/metrics/graph.py:68`
- Function **parse_predicate()** parses the predicates/labels in a form (feature, operator, threshold) that can be accessed
- NO PROCESS for dealing with CATEGORICAL LABELS/PREDICATES!

### Finding 3.2: Class boundaries are intervalized with infinite bounds - PROBLEMATIC
- Description: per feature/class, bounds are computed as lower/upper and emitted as interval strings with optional `-inf/+inf` semantics.
- Why this is numerically biased: summaries force interval interpretations even when predicate distributions are multi-modal/disjoint.
- Evidence:
  - lower/upper construction in `DPG/metrics/graph.py:147`, `DPG/metrics/graph.py:148`
  - interval output strings in `DPG/metrics/graph.py:154`, `DPG/metrics/graph.py:156`, `DPG/metrics/graph.py:158`



### Finding 3.3: Legacy boundary path is also interval-only and operator-string based - PROBLEMATIC
- Description: legacy path splits strings by `' <= | > '` and updates min/max interval endpoints.
- Why this is numerically biased: relies on ordered numeric inequalities and scalar intervals.
- Evidence:
  - `calculate_class_boundaries` with `re.split(' <= | > ', node)` in `DPG/metrics/graph.py:18`, `DPG/metrics/graph.py:23`
  - interval output in `DPG/metrics/graph.py:44`, `DPG/metrics/graph.py:46`, `DPG/metrics/graph.py:48`
- Just splitting by "<= AND >" --> No INCLUSION, EXCLUSION operators!

## 4) Visualization labels

### Finding 4.1: Visualization parsing is constrained to numeric threshold predicates - PROBLEMATIC
- Description: visualization regex parser extracts `(feature, operator, float(threshold))` only for `<=` and `>`.
- Why this is numerically biased: all downstream label-aware plots assume continuous threshold predicates.
- Evidence:
  - `_PREDICATE_PATTERN` in `DPG/dpg/visualizer.py:23`, `DPG/dpg/visualizer.py:24`
  - `parse_predicate_parts` in `DPG/dpg/visualizer.py:893`
- Parse predicate labels like 'feature <= 1.23' or 'feature > 0.7'. --> No categorical

### Finding 4.2: LRC predicate summaries only include labels containing `<=` or `>` - PROBLEMATIC!
- Description: predicate rows are filtered by operator text presence.
- Why this is numerically biased: non-threshold predicate forms are excluded from ranking and comparative plots.
- Evidence:
  - filter mask in `DPG/dpg/visualizer.py:927`, `DPG/dpg/visualizer.py:928`, `DPG/dpg/visualizer.py:929`

### Finding 4.3: Plot glyph semantics encode interval directionality - PROBLEMATIC
- Description: operator drives line style and directional density glyphs (`^` for `>`, `v` for `<=`).
- Why this is numerically biased: plot grammar assumes ordered left/right threshold behavior.
- Evidence:
  - line style by operator in `DPG/dpg/visualizer.py:1147`
  - operator branching in interval aggregation in `DPG/dpg/visualizer.py:1388`, `DPG/dpg/visualizer.py:1390`

## 5) Path assumptions: ordered comparisons (`<=` and `>`) only

### Finding 5.1: Decision-path extraction uses exactly binary ordered split logic - PROBLEMATIC!
- Description: branch selection is hardcoded to `sample_val <= threshold` else `>`.
- Why this is numerically biased: path semantics are tied to total order over numeric values.
- Evidence:
  - branch condition in `DPG/dpg/core.py:200` and label creation in `DPG/dpg/core.py:201`, `DPG/dpg/core.py:204`
  - parallel variant in `DPG/dpg/core.py:244`, `DPG/dpg/core.py:247`

### Finding 5.2: Parsers and tests codify the same two-operator assumption - PROBLEMATIC!
- Description: regexes in metrics/visualizer/tests explicitly restrict predicate forms.
- Why this is numerically biased: the accepted predicate language is structurally numeric-threshold-only.
- Evidence:
  - metrics parser regex in `DPG/metrics/graph.py:68`
    - `\s*(<=|>)\s*`
  - visualizer parser regex in `DPG/dpg/visualizer.py:24`
    - `_PREDICATE_PATTERN = re.compile(
          r"(.+?)\s*(<=|>)\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
    )`
  - boundary format test regex in `DPG/tests/test_metrics.py:239`
    - Actually very similar to metrics parser regex in graph.py!

## 6) Summaries and metrics assuming intervals

### Finding 6.1: Core graph metrics produce interval summaries by design - PROBLEMATIC!
- Description: feature summaries are represented as min/max intervals.
- Why this is numerically biased: interval abstraction is the default metric output for class boundaries.
- Evidence:
  - `calculate_class_boundaries` interval outputs in `DPG/metrics/graph.py:44`, `DPG/metrics/graph.py:46`, `DPG/metrics/graph.py:48`
    - **for feature, (min_greater, max_lessequal) in feature_bounds.items():
            if min_greater == math.inf:
                boundary = f"{feature} <= {max_lessequal}"
            elif max_lessequal == -math.inf:
                boundary = f"{feature} > {min_greater}"
            else:
                boundary = f"{min_greater} < {feature} <= {max_lessequal}"
            boundaries.append(boundary)**
  - `extract_feature_intervals` initializes `{"min": -inf, "max": inf}` and updates by operators in `DPG/metrics/graph.py:430`, `DPG/metrics/graph.py:442`, `DPG/metrics/graph.py:444`
    - **regex = r'([a-zA-Z0-9_]+)\s*([<=|>]+)\s*([-+]?[\d.]+)'**
    - Code:
    - 
            for decision in decisions:
                match = re.search(regex, decision)
                if match:
                    feature, operator, value = match.groups()
                    value = float(value)
                    feature_count[feature] += 1

                    if '>' in operator:
                        feature_intervals[feature]["min"] = max(feature_intervals[feature]["min"], value)
                    elif '<=' in operator:
                        feature_intervals[feature]["max"] = min(feature_intervals[feature]["max"], value)
                        
            return feature_count, feature_intervals**

### Finding 6.2: Community-based bounds are aggregated into `(lower_bound, upper_bound, range_width)` - PROBLEMATIC!
- Description: modern visualization pipeline computes finite/unbounded intervals and width statistics.
- Why this is numerically biased: summary objects explicitly encode interval geometry.
- Evidence:
  - bounds dataframe fields in `DPG/dpg/visualizer.py:1407`, `DPG/dpg/visualizer.py:1408`, `DPG/dpg/visualizer.py:1409`
  - Code:
  
        lower = min(values["gt"]) if values["gt"] else float("-inf")
        upper = max(values["le"]) if values["le"] else float("inf")
        if lower > upper:
            lower = min(values["all"]) if values["all"] else float("-inf")
            upper = max(values["all"]) if values["all"] else float("inf")
        width = (upper - lower) if (np.isfinite(lower) and np.isfinite(upper)) else np.nan
        rows.append(
            {
                "class_name": cls,
                "community_id": community_id,
                "feature": feature,
                "lower_bound": float(lower),
                "upper_bound": float(upper),
                "range_width": float(width) if pd.notna(width) else np.nan,
            }
  - grouped interval aggregation (`min` lower, `max` upper, width) in `DPG/dpg/visualizer.py:1599`, `DPG/dpg/visualizer.py:1600`, `DPG/dpg/visualizer.py:1601`

### Finding 6.3: Dataset comparison plots and constraint overview are min/max range-centric
- Description: empirical class summaries and visual overlays rely on per-feature min/max ranges.
- Why this is numerically biased: summary narratives become interval overlap/non-overlap analyses.
- Evidence:
  - dataset bounds `ds_lower_bound`, `ds_upper_bound` in `DPG/dpg/visualizer.py:1537`, `DPG/dpg/visualizer.py:1550`, `DPG/dpg/visualizer.py:1551`
    - Code:
    - 
          for feature in X_df.columns:
            rows.append(
                {
                    "class_name": str(cls),
                    "feature": str(feature),
                    "ds_lower_bound": float(class_frame[feature].min()),
                    "ds_upper_bound": float(class_frame[feature].max()),
                }
            )
  - constraints overview min/max rendering and non-overlap checks in `DPG/dpg/visualizer.py:686`, `DPG/dpg/visualizer.py:713`, `DPG/dpg/visualizer.py:747` -> NOT SO SURE TO BE RELEVANT!

## Additional notes

- Class-node detection itself is label-string based (`"Class "` prefix / `"Class"` containment), which is a structural text convention relied on throughout metrics/visualization:
  - `DPG/metrics/graph.py:97`, `DPG/dpg/utils.py:50`, `DPG/dpg/sklearn_dpg.py:176`
- This analysis identifies where numeric bias is encoded; it does not claim those choices are necessarily wrong for tree-threshold models. They are strong design assumptions that should be made explicit in API/docs.
