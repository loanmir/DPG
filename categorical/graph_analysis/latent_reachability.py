"""Stress test: can a real DPG trigger the latent transformation bugs?

For one (dataset, seed, n_trees) this trains the pipeline's RandomForest
(``load_dataset`` + ``train_cv`` of run_all_categorical_experiments), extracts
the trace log with the repo's own ``DecisionPredicateGraph.fit``, and for each
``perc_var`` builds the BASIC structure exactly as ``generate_dot`` /
``to_networkx`` write it -- once with DPG's own ``discover_dfg`` (unstable
per-trace sort, report #14) and once with every trace in its true order.

Each structure goes through GROUPED / SPLIT / CONJUNCTION / DEDUP with the
suspect functions wrapped. Every call is checked semantically against the
dataset's category domains (the constraint algebra of analyse_graphs.py):
does the output mean the same as the input, and if not, was the input
already unsatisfiable (a contradiction gets hidden) or satisfiable (a rule
real rows follow changes)? DEDUP is checked edge by edge against every parent
of each rewritten node, which is exact for whole paths by induction.

Usage (from the repo root)::

    python categorical/graph_analysis/latent_reachability.py run <dataset> <seed> <n_trees> > out.jsonl
    python categorical/graph_analysis/latent_reachability.py summary <dir-of-jsonl> [<branch> [dpg|true_order]]

Environment switches: ``PERC_VARS`` (comma list, default 0.01,0.005,0.001),
``VIEW_TIME_LIMIT`` (seconds per transformation, default 5; SPLIT never
finishes on same-feature cycles, report #15), ``SKIP_SPLIT=1``, and
``FIXED_CONJ=1`` (replace #3/#4 by correct versions, to isolate #5/#6).

See docs/development/graph_analysis_report.md for the results.
"""
import contextlib
import glob
import hashlib
import io
import json
import os
import pathlib
import signal
import sys
from collections import Counter, defaultdict

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "categorical"), str(ROOT / "categorical" / "graph_analysis")]

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import networkx as nx  # noqa: E402
import numpy as np  # noqa: E402
from networkx.readwrite import json_graph  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402

import analyse_graphs as A  # noqa: E402
from categorical import cat_duplicate_split_peer as DD  # noqa: E402
from categorical import cat_grouping as G1  # noqa: E402
from categorical import cat_grouping_split as G2  # noqa: E402
from categorical import cat_grouping_split_conjunction as G3  # noqa: E402
from categorical import run_all_categorical_experiments as R  # noqa: E402
from dpg.core import DecisionPredicateGraph  # noqa: E402

PERC_VARS = tuple(float(x) for x in os.environ.get("PERC_VARS", "0.01,0.005,0.001").split(","))
VIEW_TIME_LIMIT = int(os.environ.get("VIEW_TIME_LIMIT", "5"))
SKIP_SPLIT = os.environ.get("SKIP_SPLIT") == "1"
SORT_SCRAMBLE_LEN = 16  # numpy's quicksort is a stable insertion sort up to 16 items


# --------------------------------------------------------------------------
# DPG generation (repo code) -> structure payload
# --------------------------------------------------------------------------
def nid(label):
    return str(int(hashlib.sha1(label.encode()).hexdigest(), 16))


def dfg_to_structure(dfg):
    """Mirror generate_dot (edges added in ascending-weight order), to_networkx
    and save_dpg_structure_json."""
    G = nx.DiGraph()
    labels = {}
    for (a, b), w in sorted(dfg.items(), key=lambda kv: kv[1]):
        labels[nid(a)], labels[nid(b)] = a, b
        G.add_edge(nid(a), nid(b), weight=float(w))
    nodes = [{"id": k, "label": v} for k, v in sorted(labels.items())]
    return {"nodes": nodes, "graph": json_graph.node_link_data(G)}


def true_order_dfg(flog):
    """discover_dfg without the per-trace sort (groupby keeps row order)."""
    dfg = {}
    for _, seq in flog.groupby("case:concept:name", sort=False)["concept:name"]:
        seq = list(seq)
        for a, b in zip(seq, seq[1:]):
            dfg[(a, b)] = dfg.get((a, b), 0) + 1
    return dfg


def fit_log(ds, seed, n_trees):
    """Unfiltered trace log of the pipeline's model (perc_var=0 skips filter_log)."""
    csv = sorted((R.DEFAULT_DATASETS_DIR / ds).glob("*.csv"))[0]
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        X, y, fn = R.load_dataset(csv, ds)
        model = RandomForestClassifier(n_estimators=n_trees, random_state=seed)
        _, _, (Xtr, _) = R.train_cv(model, X, y)
        tn = np.unique(y).astype(str).tolist()
        # n_jobs=2: the parallel path the pipeline uses (n_jobs=1 crashes, report #16)
        dpg = DecisionPredicateGraph(
            model, fn, tn,
            dpg_config={"dpg": {"default": {"perc_var": 0, "decimal_threshold": 2, "n_jobs": 2}}},
        )
        cap = {}
        dpg.discover_dfg = lambda log: cap.setdefault("log", log) is None or {}
        dpg.fit(Xtr.values)
    return dpg, cap["log"], list(fn)


# --------------------------------------------------------------------------
# Semantics
# --------------------------------------------------------------------------
class Sem:
    def __init__(self, ds):
        self.onehot, self.domain, self.numeric = A.load_domain(ds)

    def key(self, label):
        return A.label_cons(label, self.domain, self.onehot, self.numeric)[0].key()

    def clauses_key(self, clauses):
        return self.key(" AND ".join(f"{b} {op} {{{', '.join(c)}}}" for b, op, c in clauses))


# --------------------------------------------------------------------------
# Correct versions of #3/#4 (FIXED_CONJ=1), to isolate the dedup problems
# --------------------------------------------------------------------------
def _combine_fixed(a, b):
    combined = [(base, op, list(cats)) for base, op, cats in a]
    for base, op, cats in b:
        if combined and combined[-1][0] == base and combined[-1][1] == op:
            lb, lo, lc = combined[-1]
            merged = [c for c in lc if c in set(cats)] if op == "IN" else lc + list(cats)
            if op == "IN" and not merged:  # contradiction: keep both clauses visible
                combined.append((base, op, list(cats)))
                continue
            combined[-1] = (lb, lo, merged)
        else:
            combined.append((base, op, list(cats)))
    return combined


def _simplify_fixed(clauses):
    by = {}
    for base, op, cats in clauses:
        by.setdefault(base, []).append((op, list(cats)))
    out = []
    for base, items in by.items():
        ins = [set(c) for op, c in items if op == "IN"]
        excl = list(dict.fromkeys(x for op, c in items if op == "NOT IN" for x in c))
        if ins:
            first = next(c for op, c in items if op == "IN")
            allowed = [x for x in first if all(x in s for s in ins) and x not in excl]
            if allowed:
                out.append((base, "IN", allowed))
            else:  # contradiction: keep the original clauses so it stays visible
                out.extend((base, op, c) for op, c in items)
        else:
            out.append((base, "NOT IN", excl))
    return out


if os.environ.get("FIXED_CONJ") == "1":
    G3._combine_clauses, G3._simplify_same_base = _combine_fixed, _simplify_fixed


# --------------------------------------------------------------------------
# Instrumentation
# --------------------------------------------------------------------------
class Probe:
    def __init__(self, sem):
        self.sem = sem
        self.c = Counter()
        self.ex = defaultdict(list)

    def hit(self, name, wrong, example, before=None):
        """``before`` is the semantic key of the input: a wrong result on an
        UNSAT input hides a contradiction, on a SAT input it changes a rule
        that real rows can follow; None marks a structural finding."""
        self.c[f"{name}.reached"] += 1
        if wrong:
            kind = ("wrong" if before is None else
                    "wrong_from_unsat" if before == "UNSAT" else "wrong_from_sat")
            self.c[f"{name}.{kind}"] += 1
            if len(self.ex[f"{name}.{kind}"]) < 3:
                self.ex[f"{name}.{kind}"].append(example)


@contextlib.contextmanager
def instrumented(probe):
    orig_comb, orig_simp, orig_restr = G3._combine_clauses, G3._simplify_same_base, DD._restrict_clause

    def comb(a, b):
        out = orig_comb(a, b)
        if a and b and a[-1][0] == b[0][0] and a[-1][1] == b[0][1]:
            kind = "item3_conj_boundary_IN_union" if b[0][1] == "IN" else "conj_boundary_NOTIN_union"
            before = probe.sem.clauses_key(a + b)
            probe.hit(kind, before != probe.sem.clauses_key(out), {"in": a + b, "out": out}, before)
        return out

    def simp(clauses):
        out = orig_simp(clauses)
        by = defaultdict(list)
        for b, op, c in clauses:
            by[b].append((op, c))
        for b, items in by.items():
            if len(items) < 2:
                continue
            ops = {op for op, _ in items}
            sub_in = [(b, op, c) for op, c in items]
            sub_out = [x for x in out if x[0] == b]
            before = probe.sem.clauses_key(sub_in)
            kind = ("item4_conj_opposite_ops" if len(ops) == 2 else
                    "item3_conj_same_base_IN_union" if ops == {"IN"} else
                    "conj_same_base_NOTIN_union")
            probe.hit(kind, before != probe.sem.clauses_key(sub_out), {"in": sub_in, "out": sub_out}, before)
        return out

    def restr(parent, child):
        out = orig_restr(parent, child)
        if out is not None:
            name = f"dedup_restrict_{parent[1].replace(' ', '')}_{child[1].replace(' ', '')}"
            before = probe.sem.clauses_key([parent, child])
            if out[1] == "IN" and not out[2]:
                wrong = False  # the caller keeps the original clause
            else:
                kept = [] if (out[1] == "NOT IN" and not out[2]) else [out]
                wrong = before != probe.sem.clauses_key([parent] + kept)
            probe.hit(name, wrong, {"parent": parent, "child": child, "out": out}, before)
        return out

    G3._combine_clauses, G3._simplify_same_base, DD._restrict_clause = comb, simp, restr
    try:
        yield
    finally:
        G3._combine_clauses, G3._simplify_same_base, DD._restrict_clause = orig_comb, orig_simp, orig_restr


def multi_in_labels(labels):
    out = []
    for lab in labels.values():
        for piece in lab.split(" AND "):
            p = G1._parse_in_label(piece)
            if p and p[1] == "IN" and len(set(p[2])) > 1:
                out.append(piece)
    return out


def dedup_check(probe, conj, dd):
    """CONJ -> DEDUP, edge by edge: for every rewritten node v and every parent
    w, label(w) AND old(v) must mean the same as label(w) AND new(v); every
    CONJ edge must survive (or be rewired past a dropped node)."""
    sem = probe.sem

    def lab_edges(s):
        labels = {str(n["id"]): str(n["label"]) for n in s["nodes"] if "->" not in str(n["id"])}
        g = s["graph"]
        edges = {(str(e["source"]), str(e["target"])) for e in (g.get("edges") or g.get("links") or [])}
        return labels, edges

    cl, ce = lab_edges(conj)
    dl, de = lab_edges(dd)
    par, succ = defaultdict(set), defaultdict(set)
    for s, d in ce:
        par[d].add(s)
        succ[s].add(d)
    for v, old in cl.items():
        new = dl.get(v)
        if new == old:
            continue
        multi = len({cl[x] for x in par[v]}) > 1
        for w in par[v]:
            a = sem.key(f"{cl[w]} AND {old}")
            b = sem.key(f"{cl[w]} AND {new}" if new else cl[w])
            probe.hit("dedup_rewrite_vs_parent" + ("_multiparent" if multi else ""), a != b,
                      {"parent": cl[w], "old": old, "new": new, "parents": len(par[v])}, a)
    for w, v in ce:
        if w not in dl:
            continue
        ok = (w, v) in de if v in dl else all((w, d) in de for d in succ[v] if d != w)
        probe.hit("dedup_edge_kept", not ok, {"edge": [cl[w], cl[v]], "in_deg_v": len(par[v])})


def basic_checks(probe, structure, sem):
    """F2 of the report on BASIC edges, plus #7 on the labels actually used."""
    labels = {n["id"]: n["label"] for n in structure["nodes"]}
    for lab in labels.values():
        m = A.NUM_RE.match(lab)
        if not m or abs(float(m["v"]) - 0.5) > 1e-9:
            continue
        guess = G1._split_one_hot_column(m["f"])
        if m["f"] in sem.onehot and guess != sem.onehot[m["f"]]:
            probe.hit("item7_missplit_label", True, lab)
        elif m["f"] in sem.numeric and guess is not None:
            probe.hit("item7_numeric_as_category", True, lab)
    for e in structure["graph"]["links"]:
        a, b = labels[str(e["source"])], labels[str(e["target"])]
        if a.startswith("Class "):
            probe.hit("basic_edge_out_of_class", True, [a, b])
        ma, mb = A.NUM_RE.match(a), A.NUM_RE.match(b)
        if not (ma and mb and ma["f"] in sem.onehot and mb["f"] in sem.onehot):
            continue
        (ba, va), (bb, vb) = sem.onehot[ma["f"]], sem.onehot[mb["f"]]
        if ba != bb:
            continue
        if ma["op"] == ">":  # after X = a every X_* dummy is constant
            probe.hit("basic_edge_after_IN_same_feature", True, [a, b])
        elif va == vb:  # after X_a <= 0.5, X_a is constant
            probe.hit("basic_edge_after_NOTIN_same_dummy", True, [a, b])


# --------------------------------------------------------------------------
# Views
# --------------------------------------------------------------------------
class ViewTimeout(Exception):
    pass


@contextlib.contextmanager
def time_limit(seconds):
    def handler(signum, frame):
        raise ViewTimeout()
    old = signal.signal(signal.SIGALRM, handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


def run_views(probe, structure, limit=VIEW_TIME_LIMIT):
    sizes = {"basic": len(structure["nodes"])}
    lm, _ = G1._build_label_map(structure)
    gl, _, absorbed = G1._apply_grouping(structure, lm)
    for p in multi_in_labels({k: v for k, v in gl.items() if k not in absorbed}):
        probe.hit("item3_grouped_IN_union", True, p, "UNSAT")
    if not SKIP_SPLIT:
        try:
            with time_limit(limit), contextlib.redirect_stdout(io.StringIO()) as out:
                l2, _ = G2._split_then_merge(structure, lm)
            sizes["split"] = len(l2)
            if "iteration cap" in out.getvalue():
                probe.hit("split_iteration_cap", True, len(l2))
            for p in multi_in_labels(l2):
                probe.hit("item3_split_IN_union", True, p, "UNSAT")
        except ViewTimeout:
            probe.hit("split_timeout", True, f">{limit}s")
    try:
        with time_limit(limit), contextlib.redirect_stdout(io.StringIO()) as out:
            l3, e3 = G3._split_then_merge(structure, lm)
            sizes["conj"] = len(l3)
            conj = json.loads(json.dumps(G3._build_split_structure(structure, l3, e3)))
            dd, _ = DD._deduplicate_labels(json.loads(json.dumps(conj)))
        if "iteration cap" in out.getvalue():
            probe.hit("conj_iteration_cap", True, len(l3))
        dedup_check(probe, conj, dd)
    except ViewTimeout:
        probe.hit("conj_or_dedup_timeout", True, f">{limit}s")
    return sizes


def run_one(ds, seed, n_trees):
    out = []
    dpg, log, _ = fit_log(ds, seed, n_trees)
    sem = Sem(ds)
    for pv in PERC_VARS:
        dpg.perc_var = pv
        rec = {"dataset": ds, "seed": seed, "n_trees": n_trees, "perc_var": pv}
        flog = dpg.filter_log(log)
        if flog.empty:
            rec["empty"] = True
            out.append(rec)
            continue
        lens = flog.groupby("case:concept:name", sort=False).size()
        rec.update(n_traces=int(len(lens)), max_trace_len=int(lens.max()),
                   n_traces_scrambled=int((lens > SORT_SCRAMBLE_LEN).sum()))
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            dfg_dpg = DecisionPredicateGraph.discover_dfg(dpg, flog)
        dfg_true = true_order_dfg(flog)
        rec["edges_dpg"], rec["edges_true_order"] = len(dfg_dpg), len(dfg_true)
        rec["edges_only_in_dpg"] = len(set(dfg_dpg) - set(dfg_true))
        for tag, dfg in (("dpg", dfg_dpg), ("true_order", dfg_true)):
            probe = Probe(sem)
            structure = dfg_to_structure(dfg)
            basic_checks(probe, structure, sem)
            with instrumented(probe), contextlib.redirect_stdout(io.StringIO()):
                rec[f"{tag}_sizes"] = run_views(probe, structure)
            rec[f"{tag}_counts"] = dict(probe.c)
            rec[f"{tag}_examples"] = dict(probe.ex)
        out.append(rec)
    return out


# --------------------------------------------------------------------------
# Summary
# --------------------------------------------------------------------------
def summary(directory, branch=None, tag="dpg"):
    recs = [json.loads(l) for f in sorted(glob.glob(f"{directory}/*.jsonl")) for l in open(f) if l.strip()]
    ok = [r for r in recs if not r.get("empty")]
    print(f"configs: {len(recs)}  non-empty DPGs: {len(ok)}  datasets: {len({r['dataset'] for r in recs})}")
    by = defaultdict(Counter)
    for r in ok:
        c = by[r["perc_var"]]
        c["graphs"] += 1
        c["scrambled"] += r["n_traces_scrambled"] > 0
        c["spurious_edges"] += r["edges_only_in_dpg"]
    print("\n#14 trace scrambling per perc_var:", {k: dict(v) for k, v in sorted(by.items())})
    for t in ("dpg", "true_order"):
        cnt, graphs, dsets = Counter(), Counter(), defaultdict(set)
        for r in ok:
            for k, v in r.get(f"{t}_counts", {}).items():
                cnt[k] += v
                graphs[k] += 1
                dsets[k].add(r["dataset"])
        print(f"\n== {t}: count (graphs) ==")
        print(f"  {'branch':40s} {'reached':>16s} {'wrong from UNSAT':>18s} {'wrong from SAT':>16s} {'structural':>14s}")
        for n in sorted({k.rsplit('.', 1)[0] for k in cnt}):
            cell = lambda kind: f"{cnt[f'{n}.{kind}']} ({graphs[f'{n}.{kind}']})"
            bad = sorted(set().union(*(dsets[f"{n}.{k}"] for k in ("wrong_from_sat", "wrong"))))
            print(f"  {n:40s} {cell('reached'):>16s} {cell('wrong_from_unsat'):>18s} "
                  f"{cell('wrong_from_sat'):>16s} {cell('wrong'):>14s} {bad if bad else ''}")
    if branch:
        for r in ok:
            for ex in r.get(f"{tag}_examples", {}).get(branch, []):
                print(r["dataset"], r["seed"], r["n_trees"], r["perc_var"], json.dumps(ex, ensure_ascii=False)[:400])


if __name__ == "__main__":
    if sys.argv[1] == "run":
        for rec in run_one(sys.argv[2], int(sys.argv[3]), int(sys.argv[4])):
            print(json.dumps(rec, default=str))
    else:
        summary(*sys.argv[2:])
