# DPG graph analysis: original vs transformed views

**Date:** 2026-10-15 · **Pipeline version:** `9861198` (dedup logic for IN/IN clauses)
**Scope:** the 11 datasets from the latest `run_all_categorical_experiments.py` run.
`rain_australia` is not covered (its run hit the RAM limit of this machine and was skipped);
`health_insurance` and `dummy_dataset` are excluded by the pipeline itself.

## How the graphs were checked

Every source → `Class` path of a graph encodes a rule: the conjunction of the node labels along it.
An independent checker (not reusing any parsing from `categorical/`) converted every path of every
view into the exact constraint it puts on the raw features (allowed value set per categorical
feature, interval per numeric feature), using the dataset CSV for the categorical domains. It then:

* checked the **original (BASIC) graph** for missing classes, cycles, unreachable nodes,
  contradictory paths (no row can satisfy them) and redundant tests;
* matched every path of each **transformed view** back to the BASIC path(s) it came from (following
  node ids, including the chains that were folded away) and compared the two constraints.

All paths were enumerated exactly, except for `titanic` whose 46-node cycle makes that impossible;
there 5,000 random paths per view were checked instead.

To confirm suspected bugs that the current data does not trigger, the transformation functions
themselves were called on small hand-built graphs (outputs reproduced below).

## Summary

**No transformed view changes the meaning of any path on any of the 11 datasets.** Every rule in
GROUPED, GROUPED-SPLIT, CONJUNCTION and DEDUPLICATED matches a rule of the BASIC graph, with the same
class. The problems are elsewhere:

| # | Problem | Where | Status | Impact |
|---|---|---|---|---|
| 1 | Edge weights are summed when chains are folded | split, conjunction, dedup | **observed** (5/11 datasets) | weights inflated up to 4.5×; edge thickness and weight-based metrics no longer comparable to BASIC |
| 2 | `explanation_agreement` / `explanation_overlap` are wrong under pandas 3 | `dpg_equivalence.py` | **observed** (values on W&B) | conjunction views reported as 0.43–0.59 when they are actually 1.00 |
| 3 | Two `IN` tests on the same feature are merged by **union** | grouping, split, conjunction | latent | contradictory path becomes a valid rule |
| 4 | Opposite-operator clauses keep the "smaller set" | conjunction `_simplify_same_base` | latent | contradiction hidden, or a constraint dropped |
| 5 | Dedup restriction for parent `IN` / child `NOT IN` is wrong | dedup `_restrict_clause` | latent | contradiction hidden |
| 6 | Dedup rewrites a node from one parent's condition while it has others | dedup `_deduplicate_labels` | latent | rule loosened for the other parents |
| 7 | One-hot column split with `rsplit("_", 1)` | all grouping scripts + `dpg_equivalence.py` | latent (mitigated by renaming data) | wrong feature/value in labels |
| 8 | Cloning makes the "grouped" graph bigger | split, conjunction | **observed** | adult_subset: 24 → 35 nodes, 583 → 2080 predicate chars |
| 9 | BASIC graphs explain only some classes | original DPG | **observed** (7/11) | e.g. nursery shows 1 of 5 classes |
| 10 | BASIC graphs contain cycles and contradictory paths | original DPG | **observed** (5/11) | titanic: 61% of paths impossible |
| 11 | Nursery: 9 of 12 nodes unreachable | original DPG | **observed** | only one usable rule |
| 12 | `credit_card_visual` duplicates `credit_card_approval` | datasets | **observed** | same graphs, double-counted in aggregates |
| 13 | Absorbed nodes left in `graph["nodes"]` | `cat_grouping._build_grouped_structure` | **observed** (bank_marketing) | orphan unlabelled node; GROUPED node count off by one per merge |

"Latent" means the code is wrong but none of the current graphs hit the case. Problems 3–5 only
affect paths that already combine contradictory tests (BASIC graphs do contain such paths, see #10),
so they would surface as soon as such a path falls outside a cycle.

---

## Problems in the transformed views

### 1. Edge weights are summed when chains are folded (observed)

When a node absorbs its successor, the weight of the absorbed edge is **added** to every outgoing
edge (`edges[(u, d)] += w + w_uv` in `_fold_forward` of `cat_grouping_split*.py`, same rule in
`cat_grouping._apply_grouping` and in the dedup rewire). A chain `A -43-> B -43-> C -43-> Class`
becomes `A∧B∧C -129-> Class`, although the same 43 samples flow through it.

Total weight flowing into the class nodes, relative to BASIC:

| dataset | GROUPED | SPLIT | CONJUNCTION | DEDUPLICATED |
|---|---|---|---|---|
| accidents_mining | 1.00× | 1.00× | **2.68×** | **2.68×** |
| adult_subset | 1.00× | **1.51×** | **3.81×** | **3.81×** |
| bank_marketing | 1.00× | 1.00× | **4.46×** | **4.46×** |
| credit_card_approval / _visual | 1.00× | 1.00× | **1.92×** | **1.92×** |
| nursery | 1.00× | 1.00× | **3.22×** | **3.22×** |
| other 6 datasets | 1.00× | 1.00× | 1.00× | 1.00× |

Inflated edges are drawn **red** in the figures (label: new weight, BASIC weight in brackets).
Because longer chains accumulate more, the inflation is uneven within one graph: in
accidents_mining the three edges into `Class 0` go from 42 / 43 / 46 to 84 / 129 / 138, so their
relative order now reflects chain length, not traffic. Anything weight-based (edge thickness in the
renders, communities, edge metrics) is distorted. Keeping the weight of the outgoing edge (or the
minimum along the chain) would preserve the meaning.

### 2. Explanation-agreement metric broken by pandas 3 (observed)

`dpg_equivalence.load_categorical_domains` selects categorical columns with
`df[col].dtype == object`. The project venv runs **pandas 3.0.3** (`requirements.txt` pins 2.2.3),
where text columns have dtype `str`, so the function returns **no domains**. The metrics run logs
this as `domain-aware canonicalisation on for 0 categorical column(s)`. Without domains, mixed
`IN`/`NOT IN` clauses are not normalised, and equivalent routes are counted as different.

Recomputed for BASIC vs DEDUPLICATED, same 200 rows:

| dataset | as uploaded to W&B | with the dtype check fixed |
|---|---|---|
| accidents_mining | 0.593 (overlap 0.599) | **1.000** (overlap 1.000) |
| bank_marketing | 0.433 (overlap 0.570) | **1.000** (overlap 1.000) |

This agrees with the checker: these views are exact rewrites. The `explanation_agreement` and
`explanation_overlap` columns on W&B for this run should not be trusted. Fix: use
`pd.api.types.is_string_dtype(df[col]) or df[col].dtype == object`, or pin pandas to 2.2.3.

Note also that step 2 of `canonical_constraints` unions same-operator `IN` sets, i.e. the metric
shares bug #3 and could not detect it.

### 3. Two `IN` tests on the same feature are merged by union (latent)

`X IN {a}` then `X IN {b}` means `X = a AND X = b` (impossible), but the chain merge writes
`X IN {a, b}` (either). Unioning categories is only correct for `NOT IN`. Affected:
`cat_grouping._apply_grouping`, `cat_grouping_split._split_then_merge` (both branches),
`cat_grouping_split_conjunction._combine_clauses` and `_simplify_same_base` case 1.

```
BASIC:        X_a > 0.5 -> X_b > 0.5 -> Class 1      (no row can follow this)
GROUPED:      X IN {a, b} -> Class 1                  (every row with X=a or X=b)
SPLIT:        X IN {a, b} -> Class 1
CONJUNCTION:  X IN {a, b} -> Class 1
```

For `IN` the merged set must be the **intersection**.

### 4. Opposite-operator clauses: "keep the smaller set" (latent)

`_simplify_same_base` (conjunction), case 2: when a node has both `X IN {...}` and
`X NOT IN {...}`, it keeps whichever set is smaller and drops the other. The correct result is
`X IN (in_set \ not_in_set)`.

```
BASIC:        X_a > 0.5 -> X_a <= 0.5 -> Class 1     (impossible)
CONJUNCTION:  X IN {a} -> Class 1                     (tie -> keeps IN, contradiction gone)

BASIC:        X_b <= 0.5 -> Y_p > 0.5 -> X_a > 0.5 -> X_c > 0.5 -> Class 1
CONJUNCTION:  X NOT IN {b} AND Y IN {p} -> Class 1    (the X IN {a, c} test is dropped entirely)
```

### 5. Dedup restriction for parent `IN` / child `NOT IN` (latent)

`cat_duplicate_split_peer._restrict_clause`, case parent `X IN {p}`, child `X NOT IN {c}`: the code
(and docstring) keep `c \ p`; the values that still matter under the parent are `c ∩ p`.

```
_restrict_clause(("X","IN",["a"]), ("X","NOT IN",["a","c"]))  ->  ('X', 'NOT IN', ['c'])
correct: ('X', 'NOT IN', ['a'])   (parent X=a, child excludes a: impossible, should stay visible)
```

### 6. Dedup ignores the child's other parents (latent)

The dedup pass tightens `v`'s label using the condition of one parent `u`, but never checks whether
`v` has other incoming edges. The rewritten label is then wrong for every path through the other
parents. The docstring's equivalence guarantee assumes `v` has a single parent.

```
before:  u1 = X NOT IN {a} -> v,   u2 = W > 3.0 -> v,   v = X NOT IN {a, b} AND Y IN {p}
after:   v  = X NOT IN {b} AND Y IN {p}      (path via u2 no longer excludes X = a)
```

No current conjunction output has this shape, but the script is also a standalone CLI that accepts
any structure JSON. Restricting the rewrite to `in_degree(v) == 1` (or cloning `v` first, as the
split scripts do) would fix it.

### 7. One-hot columns split with `rsplit("_", 1)` (latent, mitigated)

`_split_one_hot_column` assumes the category value has no `_` and that any `name_word <= 0.5` is a
one-hot dummy:

```
_split_one_hot_column('job_self_employed') -> ('job_self', 'employed')   true: ('job', 'self_employed')
_split_one_hot_column('has_loan')          -> ('has', 'loan')            true: numeric 0/1 column 'has_loan'
```

The recent commits work around this by editing dataset values (`Govt_job` → `Govt-job`,
`Travel_Rarely` → `Travel-Rarely`, renaming `heart_disease`), and no current label is mis-split.
New datasets will hit it again. The real column list is available at encoding time
(`pd.get_dummies` input), so the split could use it instead of guessing.

### 8. Cloning makes the "grouped" graphs bigger (observed)

The split strategy clones shared nodes so merges don't affect other branches. On adult_subset this
creates 15 clones in the CONJUNCTION view: **24 → 35 nodes, 33 → 52 edges, 583 → 2080 predicate
characters**, with labels up to 124 characters. On nursery the conjunction view grows from 12 to 14
nodes, all of the growth inside the unreachable region (#11). This runs against the purpose of
grouping (a more readable graph) and is worth weighing per dataset.

### 13. GROUPED keeps absorbed nodes in `graph["nodes"]` (observed)

`cat_grouping._build_grouped_structure` drops absorbed nodes from the top-level `nodes` list and from
the edges, but copies `graph["nodes"]` unchanged. Every absorbed node therefore survives as an
isolated node without a label (the empty grey box in the bank_marketing GROUPED figure). Metrics that
count `graph["nodes"]` are off by one per merge: the metrics table reports 28 nodes for
bank_marketing GROUPED, the real graph has 27. The split/conjunction scripts rebuild the graph from
scratch and are not affected.

---

## Problems in the original (BASIC) graphs

| dataset | classes in graph | nodes | paths | contradictory paths | paths with a redundant test | nodes in cycles | unreachable nodes |
|---|---|---|---|---|---|---|---|
| accidents_mining | 1/2 (0) | 11 | 3 | 0 | 0 | 0 | 0 |
| adult_subset | 1/2 (0) | 24 | 21 | 0 | 0 | 0 | 0 |
| bank_marketing | 1/2 (no) | 28 | 8 | 0 | 5 (62%) | 0 | 0 |
| credit_card_approval | 2/2 | 51 | 671 | 33 (5%) | 515 (77%) | 29 | 0 |
| credit_card_visual | 2/2 | 51 | 671 | 33 (5%) | 515 (77%) | 29 | 0 |
| healthcare_stroke | 1/2 (0) | 36 | 112 | 29 (26%) | 44 (39%) | 13 | 0 |
| ibm_employee_attrition | 1/2 (No) | 44 | 98 | 0 | 30 (31%) | 12 | 0 |
| nursery | **1/5** (not-recom) | 12 | **1** | 0 | 0 | 9 | **9** |
| obesity_levels_uci | **2/7** (Obesity_Type_II, III) | 29 | 15 | 0 | 0 | 0 | 0 |
| students_dropout_success | 2/3 (Dropout, Graduate) | 25 | 2 | 0 | 0 | 0 | 0 |
| titanic | 2/2 | 69 | 3,421 sampled | **2,099 (61%)** | 1,622 (47%) | 46 | 0 |

### 9. Most graphs explain only some classes

7 of 11 graphs have no node for at least one class. On imbalanced binary datasets only the majority
class survives (accidents, adult, bank, healthcare_stroke, ibm). Nursery keeps 1 of 5 classes and
obesity 2 of 7. These graphs cannot explain a prediction of a missing class, and every metric
computed on them only covers the majority class. This follows from the `perc_var` filter keeping
only frequent paths. A per-class threshold, or reporting class coverage next to the metrics, would
make this visible.

### 10. Cycles and contradictory paths

DPG merges identical predicates across trees, so different trees that test features in different
orders create cycles. Walking through a cycle produces paths no row can follow:

```
healthcare_stroke: ... work_type_children <= 0.5 -> ... -> work_type_children > 0.5 -> Class 0
credit_card:       ... previous_loan_defaults_on_file_Yes <= 0.5 -> ... -> previous_loan_defaults_on_file_No <= 0.5 -> Class 0
titanic:           ... Pclass > 2.5 -> Parch <= 1.5 -> Pclass <= 1.5 -> ... -> Class 1
```

On titanic 61% of the sampled paths are impossible, and 47% contain a test already implied by an
earlier one. Nodes that trigger a contradiction have a **red border** in the figures; nodes in a cycle
are **orange**.

The cycles are also why problems #3–#5 stay latent. The conjunction script refuses to merge inside a
cycle, so the contradictory paths are never merged. The contradictions do exist, though, and on a
dataset where one sits outside a cycle those bugs would turn an impossible rule into a valid one.

### 11. Nursery: 9 of 12 nodes cannot be reached

Nursery's graph has one source node (`parents_usual > 0.5`) and one usable path:
`parents_usual > 0.5 → health_not-recom > 0.5 → Class not-recom`. The other 9 predicates form a
cycle with **no entry from any source**, so neither the equivalence walker nor a reader starting
from a root can ever reach them (grey dashed nodes in the figure).

### 12. `credit_card_visual` is a copy of `credit_card_approval`

The CSVs differ by one trailing row, and all five views are identical. If both stay in the
benchmark, any average across datasets counts this graph twice.

---

## Figures

Each figure re-renders the five views top-to-bottom, left to right:
BASIC · GROUPED · GROUPED-SPLIT · CONJUNCTION · DEDUPLICATED.

Legend: **orange fill** = node in a cycle · **red border** = node where a path becomes contradictory ·
**grey dashed** = node on no source → class path · **double border** = clone created by the split ·
**red edge** = weight differs from BASIC (BASIC value in brackets) · green ellipse = class.

### accidents_mining
Only `Class 0`; conjunction collapses three chains and inflates their weights 2–3×.

![accidents_mining](graph_analysis/accidents_mining_side_by_side.png)


### adult_subset
Only `Class 0`; conjunction clones 15 nodes (24 → 35) and weights ×3.8.

![adult_subset](graph_analysis/adult_subset_side_by_side.png)


### bank_marketing
Only `Class no`; weights ×4.5 in conjunction; 5 of 8 paths repeat a test already implied by an earlier one (e.g. `duration <= 638.0` after a tighter `duration` test).

![bank_marketing](graph_analysis/bank_marketing_side_by_side.png)


### credit_card_approval
29-node cycle, 33 contradictory paths; weights into `Class 0` ×1.9 in conjunction.

![credit_card_approval](graph_analysis/credit_card_approval_side_by_side.png)


### credit_card_visual
Identical to credit_card_approval (#12).

![credit_card_visual](graph_analysis/credit_card_visual_side_by_side.png)

### healthcare_stroke
Only `Class 0`; 13-node cycle; 26% of paths contradictory. All views identical to BASIC apart
from relabelling (everything sits in or next to the cycle).

![healthcare_stroke](graph_analysis/healthcare_stroke_side_by_side.png)


### ibm_employee_attrition
Only `Class No`; two 6-node cycles.

![ibm_employee_attrition](graph_analysis/ibm_employee_attrition_side_by_side.png)


### nursery
1 of 5 classes, 1 usable path, 9 unreachable nodes; conjunction adds 2 clones inside the
unreachable part and inflates weights ×3.2.

![nursery](graph_analysis/nursery_side_by_side.png)


### obesity_levels_uci
2 of 7 classes.

![obesity_levels_uci](graph_analysis/obesity_levels_uci_side_by_side.png)


### students_dropout_success
2 of 3 classes (no `Enrolled`); the graph is just two long chains, one per class, of mostly numeric tests, so the transformations barely change it.

![students_dropout_success](graph_analysis/students_dropout_success_side_by_side.png)


### titanic
46-node cycle; 61% of sampled paths contradictory; only numeric features plus `Sex`/`Embarked`,
so the transformations change almost nothing (SPLIT adds 1 clone).

![titanic](graph_analysis/titanic_side_by_side.png)


---

## Reproducing

Run the pipeline first so `outputs/categorical/<dataset>/` holds the five structure JSONs, then from
the repo root:

```
DS="accidents_mining adult_subset bank_marketing credit_card_approval credit_card_visual \
    healthcare_stroke ibm_employee_attrition nursery obesity_levels_uci \
    students_dropout_success titanic"
python categorical/graph_analysis/analyse_graphs.py <out_dir> $DS                         # per-path findings.json
python categorical/graph_analysis/render_side_by_side.py docs/development/graph_analysis $DS  # figures
```

The bug reproductions in problems 3–7 call the functions in `categorical/` directly on the small
graphs shown.
