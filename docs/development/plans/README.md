# Implementation plans for the graph-analysis problems

These plans fix (or, where no fix is possible, measure) the 16 problems listed in
[`../graph_analysis_report.md`](../graph_analysis_report.md) (revision 2, commit `5524a96`).
Each plan is written to be executed by a coding agent: one plan per agent run, in the
order given below. Read this file completely before starting any plan.

File names: `<problem number><letter>.md`. The letter is the approach; when a problem has
several approaches, `a` is the recommended one unless the plan says otherwise. Two extra
plans have no problem number: `0a.md` (shared validation tooling, needed by everything)
and `R1.md` (regenerate all outputs and update the report once the fixes are in).

## Plans at a glance

| Plan | Problem | Approach | Recommended | Depends on | Size |
|---|---|---|---|---|---|
| [0a](0a.md) | – | Shared validation tooling: baseline snapshot, `check_views.py`, mini stress grid, test package | **yes, first** | – | M |
| [1a](1a.md) | #1 weights summed on fold | A folded edge keeps the weight of the edge it replaces (no `+ w_uv`) | **yes** | 0a | S |
| [1b](1b.md) | #1 (clones) | Clones take a proportional share of the original's outgoing weight (flow conservation) | **yes** | 1a, 8a | M |
| [2a](2a.md) | #2 pandas 3 domains | Detect categorical columns the way `get_dummies` does (`categorical/onehot.py`) | **yes** | 0a | S |
| [3a](3a.md) | #3 `IN` ∪ `IN` | Intersection for `IN`, contradictions kept visible; same fix in the metric | **yes** | 0a | M |
| [3b](3b.md) | #3 (follow-up) | Flag unsatisfiable merged nodes (`unsat: true`), optional pruning | optional | 3a, 4a | S |
| [4a](4a.md) | #4 opposite operators | `IN (in \ not_in)`, contradictions kept visible | **yes** | 3a | S |
| [5a](5a.md) | #5 dedup `IN`/`NOT IN` | Keep `c ∩ p` instead of `c \ p` | **yes** | 0a | S |
| [6a](6a.md) | #6 dedup multi-parent | Rewrite / drop a node only when it has exactly one parent | **yes** | 5a | S |
| [6b](6b.md) | #6 (more simplification) | Multi-parent rewrite when every parent yields the same result | optional | 6a | S |
| [7a](7a.md) | #7 one-hot guessing | Store the real one-hot map in the structure JSON and use it everywhere | **yes** | 2a | L |
| [7b](7b.md) | #7 (alternative) | Unambiguous one-hot separator (`prefix_sep`) | no | – | M |
| [8a](8a.md) | #8 runaway cloning | Never merge into a node whose original id lies in a cycle (validated: 34 → 37 nodes instead of 1,386) | **yes** | 0a | S |
| [8b](8b.md) | #8 (growth from cloning) | Optional clone budget / no-clone mode | optional | 8a | M |
| [9a](9a.md) | #9 missing classes | Report class coverage as a metric | **yes** | 0a | S |
| [9b](9b.md) | #9 (more classes) | Optional per-class `perc_var` filter in `filter_log` | optional | 9a, 14a | M |
| [10a](10a.md) | #10 cycles / contradictions | Diagnostics metrics (cycle size, contradictory and redundant path shares) | **yes** | 0a | M |
| [11a](11a.md) | #11 unreachable nodes | Record where traces start; materialise a virtual `Start` node in the structure JSON | **yes** | 14a, 6a | L |
| [11b](11b.md) | #11 (alternative) | `start_nodes` attribute plus guards in every transformation and walker | no | 14a | L |
| [12a](12a.md) | #12 duplicate dataset | Exclude `credit_card_visual` by default (data kept) | **yes** | – | S |
| [12b](12b.md) | #12 (alternative) | Delete the dataset | needs user decision | – | S |
| [13a](13a.md) | #13 orphan nodes in GROUPED | Filter absorbed ids out of `graph["nodes"]` | **yes** | 0a | S |
| [14a](14a.md) | #14 trace scrambling | Remove the per-trace sort in `discover_dfg` | **yes** | 0a | S |
| [15a](15a.md) | #15 SPLIT never terminates | Same cycle guard as 8a in the split script | **yes** | 8a | S |
| [16a](16a.md) | #16 `n_jobs: 1` crash | Make the sequential paths return per-sample lists | **yes** | 0a | S |
| [R1](R1.md) | all | Regenerate outputs, re-run checker and stress test, update the report and W&B | **yes, last** | all chosen plans | L |

Size: S = under an hour of agent work, M = a few hours, L = most of a day.

## Execution order

Work on **one integration branch** (`fix/graph-analysis-plans`, created from `main`), one plan
at a time, one or more commits per plan. Several plans edit the same functions
(`_split_then_merge`, `_fold_forward`, `_clone_node`, `_simplify_same_base`), so running
them in parallel would create conflicts. Recommended order:

1. **0a** (tooling; everything else validates with it)
2. Small, independent fixes: **13a**, **16a**, **14a**, **2a**, **12a**
3. Clause algebra: **3a** → **4a** → **5a** → **6a** (→ 3b, 6b if wanted)
4. Structure: **8a** → **15a** → **1a** → **1b** → (8b if wanted)
5. Labels: **7a**
6. Graph properties: **9a**, **10a**, **11a** (→ 9b if wanted)
7. **R1**

Plans marked "optional" or "no" in the table are not executed unless the user asks for them.

## Environment (all plans)

* Repository root: `/root/gitgud/temp/DPG`. Run every command from there. All paths in the plans
  are relative to it.
* Python: always `.venv/bin/python` (Python 3.12.9, pinned `requirements.txt`: pandas 2.2.3,
  numpy 1.26.4, scikit-learn 1.5.1, networkx 3.3). **Never** use the system `python`/`python3`.
  If `.venv` is missing or broken, stop and tell the user.
* pytest is not installed in the venv yet. Install it once: `.venv/bin/pip install pytest`.
* Run tests with `PYTHONPATH=. .venv/bin/python -m pytest -q <path>`. The full existing suite is
  `tests/`; the new categorical tests (created by 0a) live in `tests/categorical/`.
* **Disk is almost full (about 2.5 GB free).** Do not create new venvs and do not copy
  `outputs/` more than once. Put scratch outputs under `outputs/plans_scratch/<plan id>/`
  (`outputs/` is git-ignored) and delete them when the plan is done, except the files the plan
  says to keep.
* **Compute budget.** Anything you estimate at more than 1 hour on this machine (7 cores, 13 GB
  RAM), or needing more than about 6 GB of RAM, must not run here. Stop and tell the user. The
  rules for the shared server are in `/root/gitgud/.github/instructions/malelab-compute.instructions.md`
  (only R1 may need it). The mini stress grid from 0a is sized to stay well below one hour.
* **No W&B uploads.** Every pipeline run in plans 0a–16a uses `--no-wandb`. Only R1 uploads, and
  only after the user has said yes.

## Code conventions

* Match the style of the file you edit: docstring density, comment tone, naming, type hints.
  Update docstrings and module docstrings that describe behaviour you change (several of them
  document the buggy behaviour explicitly; the plans list them).
* Locate code by **function name**, not line number. Line numbers in the plans refer to commit
  `5524a96` and drift as plans land.
* Keep these module-level names: `categorical/graph_analysis/latent_reachability.py` monkeypatches
  `cat_grouping_split_conjunction._combine_clauses`, `cat_grouping_split_conjunction._simplify_same_base`
  and `cat_duplicate_split_peer._restrict_clause`. They must stay module-level functions with the
  same signatures, and the code must keep calling them through the module globals (no local
  aliases, no `functools.partial` captured at import time).
* Keep the return signatures of `_split_then_merge` (`(labels, edges)`), `_apply_grouping`
  (`(labels, weights, collapsed_ids)`) and `_deduplicate_labels` (`(structure, n_rewritten)`).
  The runners and the stress-test harness unpack them.
* Do not edit anything under `outputs/` by hand, and do not commit it (it is git-ignored).
* Do not edit `docs/development/graph_analysis_report.md` in plans 0a–16a. R1 updates it once.

## Git

* Branch: `fix/graph-analysis-plans` from `main`. Do not push and do not open PRs unless the user
  asks.
* Commit message style of this repo: `fix: ...`, `test: ...`, `docs: ...`, `update: ...`, with the
  plan id in the subject, e.g. `fix(3a): Intersect IN sets when merging categorical clauses`.
* **No AI attribution anywhere**: no `Co-Authored-By` trailers, no "Generated with" lines, in
  commits, code, comments or docs.
* Commit the code, the tests and the plan file's "Execution log" together.

## Shared validation toolkit (created by 0a)

| Tool | What it answers |
|---|---|
| `PYTHONPATH=. .venv/bin/python -m pytest -q tests/categorical` | the unit tests of every plan done so far |
| `.venv/bin/python categorical/run_all_categorical_experiments.py --no-wandb --datasets $DS` | regenerates the 5 views of the 11 pipeline datasets into `outputs/categorical/` |
| `DPG_OUTPUTS_ROOT=<dir> .venv/bin/python categorical/graph_analysis/analyse_graphs.py <out> $DS` | per path: does every transformed view still mean the same as BASIC? (`n_changed_paths` must stay 0) |
| `.venv/bin/python categorical/graph_analysis/check_views.py <outputs root> <out.json> $DS` | per view: node/edge counts, orphan nodes, class coverage, total weight into class nodes vs BASIC, flow violations |
| `bash categorical/graph_analysis/mini_grid.sh <out dir>` | reduced stress test (4 datasets × 2 seeds × 2 tree counts × 3 `perc_var`), then the `latent_reachability.py summary` table |

`$DS` is always the 11 pipeline datasets:

```text
DS="accidents_mining adult_subset bank_marketing credit_card_approval credit_card_visual \
    healthcare_stroke ibm_employee_attrition nursery obesity_levels_uci \
    students_dropout_success titanic"
```

(After 12a, `credit_card_visual` is excluded by default but can still be passed explicitly.)

The baseline (revision-2 outputs, before any fix) is `outputs/categorical_baseline_rev2/`, and the
baseline numbers of every tool are in `outputs/plans_baseline/`. Compare against them, and against
the previous plan's results, never against your memory of the report.

## Definition of done (every plan)

1. The plan's implementation steps are done, nothing else changed.
2. The tests the plan lists exist and pass, and the existing suite still passes
   (`PYTHONPATH=. .venv/bin/python -m pytest -q tests/`). If an existing test fails, find out
   whether the failure is an expected consequence of the fix (the plan says which ones are). Never
   loosen or delete a test just to make it pass.
3. Every acceptance criterion in the plan's "Validation" section is met. If one is not met, stop,
   do not commit, and write in the execution log what you saw.
4. An `## Execution log` section is appended to the plan file with: date, commit hash(es),
   commands run, the measured numbers next to the expected ones, and any deviation from the plan
   with its reason.
5. Scratch outputs deleted (`outputs/plans_scratch/<plan id>/`), except the files the plan asks
   to keep.

```{toctree}
:hidden:

0a
1a
1b
2a
3a
3b
4a
5a
6a
6b
7a
7b
8a
8b
9a
9b
10a
11a
11b
12a
12b
13a
14a
15a
16a
R1
```
