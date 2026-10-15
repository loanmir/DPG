"""Independent semantic checker for DPG variants.

For every dataset under outputs/categorical/, load the BASIC DPG and each
transformed view, turn every source -> Class path into the exact logical
constraint it encodes over the *raw* features (categorical allowed-sets,
numeric intervals), and compare the views path by path.

Writes JSON findings to <out>/findings.json.

Usage (from the repo root, after run_all_categorical_experiments.py)::

    python categorical/graph_analysis/analyse_graphs.py <out_dir> <dataset> [<dataset> ...]

See docs/development/graph_analysis_report.md for what the checks mean.
"""
import json
import pathlib
import re
import sys
from collections import Counter, defaultdict

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[2]



sys.path.insert(0, str(ROOT / "categorical"))
from run_all_categorical_experiments import DROP_COLUMNS_OVERRIDES  # noqa: E402

VARIANTS = [
    ("BASIC DPG", "*_dpg_structure.json"),
    ("GROUPED DPG", "*_DPG_grouped_structure.json"),
    ("GROUPED-SPLIT DPG", "*_DPG_split_grouped_structure.json"),
    ("GROUPED-SPLIT-CONJUNCTION DPG", "*_DPG_split_grouped_conjunction_structure.json"),
    ("GROUPED-SPLIT-CONJUNCTION-DEDUPLICATED DPG", "*_deduplicated_structure.json"),
]
NUM_RE = re.compile(r"^(?P<f>.+?) (?P<op><=|>=|<|>) (?P<v>-?[0-9.eE+-]+)$")
IN_RE = re.compile(r"^(?P<base>.+?) (?P<op>NOT IN|IN) \{(?P<cats>.*)\}$")
CLASS_RE = re.compile(r"^Class (.+)$")
PATH_CAP = 200_000


# --------------------------------------------------------------------------
# Domain
# --------------------------------------------------------------------------
def load_domain(name):
    csv = next((ROOT / "datasets" / name).glob("*.csv"))
    df = pd.read_csv(csv, sep=None, engine="python", encoding="utf-8-sig")
    df = df.drop(columns=[c for c in DROP_COLUMNS_OVERRIDES.get(name, []) if c in df.columns])
    feats = df.iloc[:, :-1]
    enc = pd.get_dummies(feats, drop_first=False)
    cat_cols = [c for c in feats.columns if c not in enc.columns]
    onehot, domain = {}, {}
    for c in cat_cols:
        vals = [str(v) for v in feats[c].dropna().unique()]
        domain[c] = frozenset(vals)
        for v in vals:
            onehot[f"{c}_{v}"] = (c, v)
    numeric = set(enc.columns) - set(onehot)
    return onehot, domain, numeric


# --------------------------------------------------------------------------
# Constraint algebra
# --------------------------------------------------------------------------
class Cons:
    """Conjunction of per-feature constraints."""

    def __init__(self, domain):
        self.domain = domain
        self.cat = {}   # base -> frozenset allowed
        self.num = {}   # feat -> [lo_excl, hi_incl]
        self.unrep = False

    def copy(self):
        c = Cons(self.domain)
        c.cat, c.num, c.unrep = dict(self.cat), {k: list(v) for k, v in self.num.items()}, self.unrep
        return c

    def allow(self, base, vals):
        cur = self.cat.get(base, self.domain[base])
        self.cat[base] = cur & frozenset(vals)

    def exclude(self, base, vals):
        cur = self.cat.get(base, self.domain[base])
        self.cat[base] = cur - frozenset(vals)

    def numeric(self, f, op, t):
        lo, hi = self.num.get(f, [float("-inf"), float("inf")])
        if op in ("<=", "<"):
            hi = min(hi, t)
        else:
            lo = max(lo, t)
        self.num[f] = [lo, hi]

    def sat(self):
        return all(self.cat.values()) and all(lo < hi for lo, hi in self.num.values())

    def key(self):
        if not self.sat():
            return "UNSAT"
        cats = tuple(sorted((b, tuple(sorted(v))) for b, v in self.cat.items() if v != self.domain[b]))
        nums = tuple(sorted((f, lo, hi) for f, (lo, hi) in self.num.items()))
        return (cats, nums, self.unrep)

    def implies(self, other):
        """Is every row satisfying self also satisfying other?"""
        for b, v in other.cat.items():
            if not self.cat.get(b, self.domain[b]) <= v:
                return False
        for f, (lo, hi) in other.num.items():
            slo, shi = self.num.get(f, [float("-inf"), float("inf")])
            if slo < lo or shi > hi:
                return False
        return True


def pretty(cons_key):
    if cons_key == "UNSAT":
        return "UNSATISFIABLE (no row can follow this path)"
    cats, nums, unrep = cons_key
    parts = [f"{b} ∈ {{{', '.join(v)}}}" for b, v in cats]
    for f, lo, hi in nums:
        if lo == float("-inf"):
            parts.append(f"{f} ≤ {hi:g}")
        elif hi == float("inf"):
            parts.append(f"{f} > {lo:g}")
        else:
            parts.append(f"{lo:g} < {f} ≤ {hi:g}")
    return " ∧ ".join(parts) if parts else "TRUE (no constraint)"


# --------------------------------------------------------------------------
# Label -> constraint
# --------------------------------------------------------------------------
def apply_label(cons, label, onehot, numeric, anomalies):
    """Fold one node label into ``cons`` (in place)."""
    for piece in label.split(" AND "):
        piece = piece.strip()
        m = IN_RE.match(piece)
        if m:
            base_s, op = m["base"].strip(), m["op"]
            cats = [c.strip() for c in m["cats"].split(",") if c.strip()]
            # Reconstruct the one-hot column the grouping script split apart,
            # so a mis-split base/value still gets its true meaning.
            cols = [f"{base_s}_{c}" for c in cats]
            mapped = []
            for col in cols:
                if col in onehot:
                    mapped.append(("cat",) + onehot[col])
                elif col in numeric:
                    mapped.append(("num", col, None))
                else:
                    anomalies.append(f"unknown column '{col}' in label '{piece}'")
                    cons.unrep = True
            for kind, b, v in mapped:
                if kind == "cat" and b != base_s:
                    anomalies.append(
                        f"label base '{base_s}' is really feature '{b}' (value '{v}') in '{piece}'"
                    )
                if kind == "num":
                    anomalies.append(
                        f"numeric column '{b}' rendered as categorical in '{piece}'"
                    )
            if op == "NOT IN":
                for kind, b, v in mapped:
                    if kind == "cat":
                        cons.exclude(b, [v])
                    else:
                        cons.numeric(b, "<=", 0.5)
            else:
                bases = {(k, b) for k, b, _ in mapped}
                if len(bases) == 1 and mapped and mapped[0][0] == "cat":
                    cons.allow(mapped[0][1], [v for _, _, v in mapped])
                elif len(mapped) == 1:
                    cons.numeric(mapped[0][1], ">", 0.5)
                elif mapped:
                    anomalies.append(f"IN over several real features in '{piece}' (not representable)")
                    cons.unrep = True
            continue
        m = NUM_RE.match(piece)
        if m:
            f, op, t = m["f"], m["op"], float(m["v"])
            if f in onehot:
                b, v = onehot[f]
                (cons.allow if op in (">", ">=") else cons.exclude)(b, [v])
            else:
                cons.numeric(f, op, t)
            continue
        anomalies.append(f"unparseable label piece '{piece}'")
        cons.unrep = True


def label_cons(label, domain, onehot, numeric):
    c, a = Cons(domain), []
    apply_label(c, label, onehot, numeric, a)
    return c, a


# --------------------------------------------------------------------------
# Graph loading / path enumeration
# --------------------------------------------------------------------------
def load_graph(path):
    d = json.loads(path.read_text())
    labels = {str(n["id"]): str(n["label"]) for n in d["nodes"] if "->" not in str(n["id"])}
    g = d["graph"]
    edges = {}
    for e in g.get("links") or g.get("edges") or []:
        k = (str(e["source"]), str(e["target"]))
        edges[k] = edges.get(k, 0.0) + float(e.get("weight", 0) or 0)
    nodes = {str(n["id"]) for n in g["nodes"]} | {s for s, _ in edges} | {t for _, t in edges}
    for n in nodes:
        labels.setdefault(n, "")
    return d, labels, edges, nodes


EXPANSION_BUDGET = 1_000_000
N_SAMPLES = 5_000


def enumerate_paths(labels, edges, nodes):
    """All simple source -> Class paths; falls back to random-walk sampling
    when the graph is too cyclic to enumerate (returns capped=True)."""
    succ, indeg = defaultdict(list), Counter()
    for s, t in edges:
        succ[s].append(t)
        indeg[t] += 1
    is_class = {n: bool(CLASS_RE.match(labels[n])) for n in nodes}
    sources = sorted(n for n in nodes if indeg[n] == 0 and not is_class[n])
    paths, pops = [], 0
    for src in sources:
        stack = [(src, (src,))]
        while stack:
            pops += 1
            if pops > EXPANSION_BUDGET:
                return sample_paths(succ, is_class, sources), sources, True
            n, p = stack.pop()
            if is_class[n]:
                paths.append(list(p))
                continue
            for c in succ[n]:
                if c not in p:
                    stack.append((c, p + (c,)))
    return paths, sources, False


def sample_paths(succ, is_class, sources):
    import random
    rng = random.Random(0)
    out, seen = [], set()
    for _ in range(N_SAMPLES * 5):
        if len(out) >= N_SAMPLES or not sources:
            break
        n = rng.choice(sources)
        p = [n]
        while not is_class[n]:
            nxt = [c for c in succ[n] if c not in p]
            if not nxt:
                p = None
                break
            n = rng.choice(nxt)
            p.append(n)
        if p and tuple(p) not in seen:
            seen.add(tuple(p))
            out.append(p)
    return out


def basic_expansions(vpath_ids, bsucc, max_mid=8, cap=400, direct_first=False):
    """Basic-graph paths that visit ``vpath_ids`` in order, where the gap
    between two consecutive ids is a chain of folded-away nodes.

    ``direct_first`` uses the direct basic edge wherever one exists and only
    searches for a folded chain where it doesn't (one cheap candidate)."""
    segs = []
    for a, b in zip(vpath_ids, vpath_ids[1:]):
        if direct_first and b in bsucc.get(a, ()):
            segs.append([()])
            continue
        found, stack = [], [(a, (a,))]
        while stack and len(found) < 50:
            n, p = stack.pop()
            for c in bsucc.get(n, ()):
                if c == b:
                    found.append(p[1:])
                elif c not in p and len(p) <= max_mid and not CLASS_RE.match(c):
                    stack.append((c, p + (c,)))
        if not found:
            return []
        found.sort(key=len)
        segs.append(found)
    combos = [[vpath_ids[0]]]
    for seg, nxt in zip(segs, vpath_ids[1:]):
        combos = [c + list(mid) + [nxt] for c in combos for mid in seg][:cap]
    return combos


def path_cons(path, labels, domain, onehot, numeric):
    c, anomalies = Cons(domain), []
    redundant = []
    for n in path[:-1]:
        before = c.copy()
        lc, _ = label_cons(labels[n], domain, onehot, numeric)
        apply_label(c, labels[n], onehot, numeric, anomalies)
        if before.sat() and lc.sat() and before.implies(lc):
            redundant.append(n)
    return c, redundant


def strip_clone(nid):
    return nid.split("#")[0]


def is_subseq(short, long):
    it = iter(long)
    return all(x in it for x in short)


# --------------------------------------------------------------------------
# Main analysis
# --------------------------------------------------------------------------
def analyse(name):
    onehot, domain, numeric = load_domain(name)
    base_dir = ROOT / "outputs" / "categorical" / name
    res = {"dataset": name, "variants": {}}
    graphs = {}
    for vname, pat in VARIANTS:
        f = next((base_dir / vname).glob(pat))
        d, labels, edges, nodes = load_graph(f)
        paths, sources, capped = enumerate_paths(labels, edges, nodes)
        info = []
        for p in paths:
            c, red = path_cons(p, labels, domain, onehot, numeric)
            info.append({"path": p, "cls": labels[p[-1]], "key": c.key(), "redundant": red})
        graphs[vname] = (d, labels, edges, nodes, info)

        # node-level checks
        anomalies, taut, contra, dup_base = [], [], [], []
        for n in nodes:
            lab = labels[n]
            if CLASS_RE.match(lab) or not lab:
                continue
            c, a = label_cons(lab, domain, onehot, numeric)
            anomalies += a
            if not c.unrep:
                if not c.sat():
                    contra.append(lab)
                elif c.key() == ((), (), False):
                    taut.append(lab)
            bases = [m["base"] for m in (IN_RE.match(x.strip()) for x in lab.split(" AND ")) if m]
            if len(bases) != len(set(bases)):
                dup_base.append(lab)

        succ_any = defaultdict(list)
        for s, t in edges:
            succ_any[s].append(t)
        import networkx as nx
        G = nx.DiGraph(list(edges))
        G.add_nodes_from(nodes)
        sccs = [sorted(labels[n] for n in s) for s in nx.strongly_connected_components(G) if len(s) > 1]
        class_nodes = [n for n in nodes if CLASS_RE.match(labels[n])]
        reach_cls = set()
        for cn in class_nodes:
            reach_cls |= nx.ancestors(G, cn) | {cn}
        dead_ends = sorted(labels[n] for n in nodes if n not in reach_cls)
        on_path = {n for i in info for n in i["path"]}
        not_on_any_path = sorted(labels[n] for n in nodes if n not in on_path)
        class_in = defaultdict(float)
        for (s, t), w in edges.items():
            if CLASS_RE.match(labels[t]):
                class_in[labels[t]] += w

        res["variants"][vname] = {
            "file": str(f.relative_to(ROOT)),
            "n_nodes": len(nodes), "n_edges": len(edges),
            "n_paths": len(info), "capped": capped, "n_sources": len(sources),
            "classes": sorted(labels[n] for n in class_nodes),
            "n_unsat": sum(i["key"] == "UNSAT" for i in info),
            "unsat_examples": [[labels[n] for n in i["path"]] for i in info if i["key"] == "UNSAT"][:5],
            "n_redundant_paths": sum(bool(i["redundant"]) for i in info),
            "redundant_examples": [
                {"path": [labels[n] for n in i["path"]], "redundant": [labels[n] for n in i["redundant"]]}
                for i in info if i["redundant"]
            ][:5],
            "anomalies": sorted(set(anomalies)),
            "tautology_nodes": sorted(set(taut)),
            "contradiction_nodes": sorted(set(contra)),
            "dup_base_labels": sorted(set(dup_base)),
            "sccs": sccs,
            "dead_end_nodes": dead_ends,
            "unreached_nodes": not_on_any_path,
            "class_in_weight": dict(class_in),
            "total_weight": sum(edges.values()),
            "max_weight": max(edges.values()) if edges else 0,
        }

    # ---- path-level comparison against BASIC -------------------------
    _, blabels, bedges, _, binfo = graphs["BASIC DPG"]
    bsucc = defaultdict(list)
    for s, t in bedges:
        bsucc[s].append(t)
    bset = Counter((i["key"], i["cls"]) for i in binfo)
    bcapped = res["variants"]["BASIC DPG"]["capped"]
    key_cache = {}

    def bkey(path):
        t = tuple(path)
        if t not in key_cache:
            key_cache[t] = path_cons(path, blabels, domain, onehot, numeric)[0].key()
        return key_cache[t]

    for vname, _ in VARIANTS[1:]:
        _, vlabels, _, _, vinfo = graphs[vname]
        vcapped = res["variants"][vname]["capped"]
        vset = Counter((i["key"], i["cls"]) for i in vinfo)
        exact = not (bcapped or vcapped)
        lost = [k for k in bset if k not in vset] if exact else []
        gained = [k for k in vset if k not in bset] if exact else []
        examples, orphan = [], 0
        for i in vinfo:
            ids = [strip_clone(n) for n in i["path"]]
            quick = basic_expansions(ids, bsucc, direct_first=True, cap=1)
            if quick and bkey(quick[0]) == i["key"]:
                continue
            cands = basic_expansions(ids, bsucc)
            if not cands:
                orphan += 1
                continue
            keys = [bkey(c) for c in cands]
            if i["key"] in keys:
                continue
            b, bk = cands[0], keys[0]
            examples.append({
                "class": i["cls"],
                "basic_path": [blabels[n] for n in b],
                "basic_meaning": pretty(bk),
                "variant_path": [vlabels[n] for n in i["path"]],
                "variant_meaning": pretty(i["key"]),
                "change": ("unsat->sat" if bk == "UNSAT" else
                           "sat->unsat" if i["key"] == "UNSAT" else "different"),
            })
        # de-duplicate examples on (basic_meaning, variant_meaning)
        seen, uniq = set(), []
        for e in examples:
            k = (e["basic_meaning"], e["variant_meaning"], e["class"])
            if k not in seen:
                seen.add(k)
                uniq.append(e)
        res["variants"][vname]["vs_basic"] = {
            "exact": exact,
            "n_orphan_paths": orphan,
            "lost": [(pretty(k), c) for k, c in lost],
            "gained": [(pretty(k), c) for k, c in gained],
            "n_changed_paths": len(examples),
            "changed_examples": uniq,
            "change_kinds": dict(Counter(e["change"] for e in examples)),
        }
    return res


if __name__ == "__main__":
    OUT, DATASETS = pathlib.Path(sys.argv[1]), sys.argv[2:]
    OUT.mkdir(parents=True, exist_ok=True)
    allres = []
    for ds in DATASETS:
        print("analysing", ds, flush=True)
        allres.append(analyse(ds))
    (OUT / "findings.json").write_text(json.dumps(allres, indent=1, default=str, ensure_ascii=False))
    print("ok")
