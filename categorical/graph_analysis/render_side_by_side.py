"""Top-to-bottom re-renders of every DPG view, problems highlighted, composed
side by side per dataset.

Usage (from the repo root, after run_all_categorical_experiments.py)::

    python categorical/graph_analysis/render_side_by_side.py <out_img_dir> <dataset> [<dataset> ...]

Writes ``<out_img_dir>/<dataset>_side_by_side.png``. The figures in
docs/development/graph_analysis/ were produced with
``<out_img_dir> = docs/development/graph_analysis``.
"""
import pathlib
import sys
import tempfile
import textwrap
from collections import defaultdict

import graphviz
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import analyse_graphs as A  # noqa: E402

SHORT = {
    "BASIC DPG": "BASIC",
    "GROUPED DPG": "GROUPED",
    "GROUPED-SPLIT DPG": "GROUPED-SPLIT",
    "GROUPED-SPLIT-CONJUNCTION DPG": "CONJUNCTION",
    "GROUPED-SPLIT-CONJUNCTION-DEDUPLICATED DPG": "DEDUPLICATED",
}
MAX_W = 12000
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def wrap(label):
    parts = label.split(" AND ")
    lines = []
    for i, p in enumerate(parts):
        p = ("AND " if i else "") + p
        lines += textwrap.wrap(p, 38) or [p]
    return "\n".join(lines)


def analyse_variant(name, vname, pat, domain, onehot, numeric):
    f = next((A.ROOT / "outputs" / "categorical" / name / vname).glob(pat))
    _, labels, edges, nodes = A.load_graph(f)
    paths, _, _ = A.enumerate_paths(labels, edges, nodes)
    trigger = defaultdict(int)
    on_path = set()
    for p in paths:
        on_path.update(p)
        c = A.Cons(domain)
        for n in p[:-1]:
            A.apply_label(c, labels[n], onehot, numeric, [])
            if not c.sat():
                trigger[n] += 1
                break
    import networkx as nx
    G = nx.DiGraph(list(edges))
    G.add_nodes_from(nodes)
    scc = set().union(*[s for s in nx.strongly_connected_components(G) if len(s) > 1] or [set()])
    return labels, edges, nodes, trigger, on_path, scc


def render(name, vname, data, base_edges, out_png):
    labels, edges, nodes, trigger, on_path, scc = data
    g = graphviz.Digraph(format="png")
    g.attr(rankdir="TB", dpi="72", nodesep="0.25", ranksep="0.35")
    g.attr("node", shape="box", style="rounded,filled", fontname="DejaVu Sans", fontsize="11",
           fillcolor="#ffffff", color="#555555")
    g.attr("edge", fontname="DejaVu Sans", fontsize="9", color="#777777", fontcolor="#555555")
    for n in nodes:
        lab = labels[n]
        attrs = {}
        if A.CLASS_RE.match(lab):
            attrs.update(shape="ellipse", fillcolor="#c8e6c9", color="#2e7d32")
        elif n not in on_path:
            attrs.update(fillcolor="#e0e0e0", style="rounded,filled,dashed", fontcolor="#777777")
        elif n in scc:
            attrs.update(fillcolor="#ffe0b2")
        if trigger.get(n):
            attrs.update(color="#d32f2f", penwidth="3")
        if "#" in n:
            attrs.update(peripheries="2")
        g.node(n, wrap(lab), **attrs)
    for (s, t), w in edges.items():
        bw = base_edges.get((A.strip_clone(s), A.strip_clone(t)))
        if vname != "BASIC DPG" and (bw is None or abs(bw - w) > 1e-6):
            note = f"{w:g}" + (f" (basic {bw:g})" if bw is not None else " (new)")
            g.edge(s, t, label=note, color="#d32f2f", fontcolor="#d32f2f", penwidth="2")
        else:
            g.edge(s, t, label=f"{w:g}")
    g.render(out_png.with_suffix(""), cleanup=True)


def compose(pngs, titles, out):
    imgs = [Image.open(p).convert("RGB") for p in pngs]
    try:
        font = ImageFont.truetype(FONT, 30)
    except OSError:
        font = ImageFont.load_default(size=30)
    band, gap = 90, 40
    widths = [max(i.width, int(font.getlength(max(t.split("\n"), key=len))) + 20)
              for i, t in zip(imgs, titles)]
    h = max(i.height for i in imgs) + band
    w = sum(widths) + gap * (len(imgs) - 1)
    canvas = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(canvas)
    x = 0
    for img, t, pw in zip(imgs, titles, widths):
        canvas.paste(img, (x + (pw - img.width) // 2, band))
        d.multiline_text((x + 10, 8), t, fill="#111111", font=font, spacing=6)
        if x:
            d.line([(x - gap // 2, 0), (x - gap // 2, h)], fill="#bbbbbb", width=3)
        x += pw + gap
    if canvas.width > MAX_W:
        r = MAX_W / canvas.width
        canvas = canvas.resize((MAX_W, int(canvas.height * r)), Image.LANCZOS)
    canvas.quantize(colors=64, method=Image.Quantize.MEDIANCUT).save(out, optimize=True)


def main():
    out_dir = pathlib.Path(sys.argv[1])
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="dpg_render_"))
    for name in sys.argv[2:]:
        onehot, domain, numeric = A.load_domain(name)
        datas = {v: analyse_variant(name, v, p, domain, onehot, numeric) for v, p in A.VARIANTS}
        base_edges = datas["BASIC DPG"][1]
        pngs, titles = [], []
        for v, _ in A.VARIANTS:
            png = tmp / f"{name}__{SHORT[v]}.png"
            render(name, v, datas[v], base_edges, png)
            pngs.append(png)
            titles.append(f"{SHORT[v]}\n({len(datas[v][2])} nodes, {len(datas[v][1])} edges)")
        compose(pngs, titles, out_dir / f"{name}_side_by_side.png")
        print("rendered", name, flush=True)


if __name__ == "__main__":
    main()
