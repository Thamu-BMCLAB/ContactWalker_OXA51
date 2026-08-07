#python contactwalker_network_graphviz.py table2_RotA_all_contacts.csv --mutation ILE129 --gv-engine dot --scale 4.0 --node-size 2500 --font-size 14
#!/usr/bin/env python3
"""

Sign convention (must match your change column):
    change = WT - Mutant   (percentage points)
    negative  ->  contact STRENGTHENED in mutant (mutant-stabilized)  ->  GREEN
    positive  ->  contact WEAKENED  in mutant (mutant-destabilized)   ->  ORANGE

CSV format expected
-------------------
* First TWO columns: the residue-pair labels, e.g. "ILE129", "ASP163".
* A signed-change column: named via --change-col, else a column whose name
  contains "Min Change (all pairs)" is preferred (this is the "unchanged" half
  of the path-intermediate scaffold gate, Min Change == 0), else the first column
  whose name contains "change" (case-insensitive). REQUIRED -- the script errors
  out if it cannot find one (it does not recompute change from replicates).
* Wild-type replicate occupancy columns: identified by the --wt-prefix prefix
  (default "WT", e.g. WT1/WT2/WT3, case-insensitive). REQUIRED -- their row-wise
  mean is the "present in WT" test (>= --wt-occ) that forms the other half of the
  scaffold gate; the script errors out if none are found.
  Mutant / other columns are ignored (the mutant side is not used for scaffolding).

NOTE on layout: node positions come from a force-directed layout and are VISUAL
ONLY -- they are not physically or structurally meaningful. Edge connectivity
and the shortest paths, however, are all real measured contacts.

Example
-------
    python contactwalker_network.py table2_RotA_all_contacts.csv \
        --mutation ILE129 --sig 9 --trav 6 --depth 2

    # spread nodes out and enlarge labels
    python contactwalker_network.py contacts.csv --mutation ILE129 \
        --scale 1.4 --edge-length 1.2 --node-size 800 --font-size 10

Dependencies
------------
  Python  : pandas, numpy, networkx, matplotlib, pydot
  System  : Graphviz binaries (provides neato/sfdp/dot/... on PATH)
            Ubuntu/Debian : sudo apt-get install graphviz
            conda         : conda install -c conda-forge graphviz pydot
            macOS         : brew install graphviz   (then pip install pydot)
  (scipy is no longer required: the Kamada-Kawai layout was removed.)
"""

import argparse
import re
import sys

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.cm import ScalarMappable


# --------------------------------------------------------------------------- #
# Editable color / style constants (swap these to restyle the figure)
# --------------------------------------------------------------------------- #
COLOR_STABILIZED   = "#2E8B2E"   # green  : change < 0 (mutant-stabilized)
COLOR_DESTABILIZED = "#FF9400"   # orange : change > 0 (mutant-destabilized)
COLOR_MIDPOINT     = "#f7f7f7"   # near-white center of the diverging colormap
COLOR_BACKBONE     = "#222222"   # backbone bonds
COLOR_PATH         = "#9E9E9E"   # structural contacts on shortest paths
COLOR_TRAVERSAL    = "#FF69B4"   # traversal-threshold (disrupted contact) edges – pink
COLOR_NODE_DISRUPT = "#ECE9E2"   # disrupted residue node fill
COLOR_NODE_PATH    = "#D6D6D6"   # path-intermediate residue node fill
COLOR_MUTATION     = "#FD9BED"   # mutation/center residue node fill
COLOR_NODE_OUTLINE = "#222222"   # disrupted/mutation node outline
COLOR_PATH_OUTLINE = "#777777"   # path-intermediate node outline
HALO_COLOR         = "white"     # halo behind nodes so edges stop at borders
COLOR_CLAMP        = 0.16        # push diverging colors >=this away from white

THREE2ONE = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C', 'GLN': 'Q',
    'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LEU': 'L', 'LYS': 'K',
    'MET': 'M', 'PHE': 'F', 'PRO': 'P', 'SER': 'S', 'THR': 'T', 'TRP': 'W',
    'TYR': 'Y', 'VAL': 'V',
}


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def resnum(label):
    """Integer residue number inside a label like 'ILE129' -> 129."""
    m = re.search(r"(\d+)", str(label))
    if not m:
        raise ValueError(f"Could not parse a residue number from '{label}'")
    return int(m.group(1))


def res3(label):
    """Leading alpha residue code, e.g. 'ILE129' -> 'ILE'."""
    m = re.match(r"([A-Za-z]+)", str(label))
    return m.group(1).upper() if m else "UNK"


def short_label(label):
    """One-letter + number label, e.g. 'ILE129' -> 'I129'."""
    return f"{THREE2ONE.get(res3(label), 'KCX')}{resnum(label)}"


def display_short_label(label):
    """Short label for a user-supplied DISPLAY label (e.g. 'LEU129' -> 'L129').

    Unlike short_label(), this does NOT require the label to exist in the data —
    it is used purely for figure/legend/title text when --mutation-label overrides
    the displayed name of the center residue. Validates that the label parses to
    a residue number; raises a clear error otherwise.
    """
    try:
        num = resnum(label)
    except ValueError as exc:
        fail(f"--mutation-label '{label}' is not a valid residue label "
             f"(need a 3-letter code + number, e.g. 'LEU129'): {exc}")
    return f"{THREE2ONE.get(res3(label), 'KCX')}{num}"


def norm_residue_key(label):
    """Normalized residue identity for matching (e.g. 'ile129' -> 'ILE129')."""
    return f"{res3(label)}{resnum(label)}"


def eprint(*a, **k):
    print(*a, file=sys.stderr, **k)


def fail(msg, code=2):
    eprint(f"ERROR: {msg}")
    sys.exit(code)


# --------------------------------------------------------------------------- #
# CSV parsing
# --------------------------------------------------------------------------- #
def parse_csv(path, wt_prefix, change_col):
    try:
        raw = pd.read_csv(path)
    except Exception as exc:
        fail(f"Could not read CSV '{path}': {exc}")

    if raw.shape[1] < 3:
        fail(f"CSV needs >= 3 columns (2 residues + a change column); "
             f"found {raw.shape[1]}.")

    cols = list(raw.columns)
    r1col, r2col = cols[0], cols[1]

    # detect wild-type replicate columns by prefix (case-insensitive). These are
    # used to test "present in WT" (mean WT occupancy >= --wt-occ) for the
    # path-intermediate scaffold. Mutant columns are NOT used.
    #
    # CAUTION: a bare startswith(prefix) also matches per-replicate DIFFERENCE
    # columns such as "WT1-M1 (%)", "WT2-M3 (%)" (WT-vs-mutant deltas), whose
    # values are signed differences, not occupancies. Averaging those into
    # wt_mean corrupts it (can even go negative) and silently empties the
    # scaffold. So: keep WT-prefixed columns, but drop any that also reference a
    # mutant after a separator (e.g. "WT1-M1", "WT - M2", "WT1 minus M3") or are
    # explicitly named as a difference/delta column.
    wt_p = wt_prefix.lower()
    _diff_pat = re.compile(
        r"[-/_\u2013\u2014]\s*m\s*\d"      # WT1-M1, WT2/M3, WT_M2, WT3\u2013M1 ...
        r"|\bminus\b|\bdiff\b|\bdelta\b|\bchange\b|\bvs\.?\b",  # WT1 minus M2, *diff*, ...
        re.IGNORECASE)
    wt_all = [c for c in cols[2:] if str(c).lower().startswith(wt_p)]
    wt_cols = [c for c in wt_all if not _diff_pat.search(str(c))]
    wt_dropped = [c for c in wt_all if c not in wt_cols]
    if wt_dropped:
        eprint(f"Ignoring {len(wt_dropped)} WT-prefixed difference/delta column(s) "
               f"for the 'present in WT' test (these are WT-vs-mutant deltas, not "
               f"occupancies): {wt_dropped}")

    # locate change column
    #
    # The change column drives BOTH the diverging edge colors AND the
    # "unchanged" gate that selects path-intermediate (scaffold) residues.
    # We want the dedicated "Min Change (all pairs)" column for that, NOT one of
    # the per-replicate WTi-Mj difference columns. Selection priority:
    #   1. explicit --change-col, if given
    #   2. a column whose name contains "min change" (case-insensitive)
    #   3. fall back to the first column containing "change"
    if change_col is not None:
        if change_col not in raw.columns:
            fail(f"--change-col '{change_col}' not found. Columns: {cols}")
        chg_col = change_col
    else:
        min_cand = [c for c in cols if "min change" in str(c).lower()]
        any_cand = [c for c in cols if "change" in str(c).lower()]
        if min_cand:
            chg_col = min_cand[0]
            if len(any_cand) > 1:
                eprint(f"Multiple 'change' columns found {any_cand}; using the "
                       f"'Min Change' column '{chg_col}' for the scaffold gate. "
                       f"Override with --change-col if this is wrong.")
        elif any_cand:
            chg_col = any_cand[0]
            eprint(f"WARNING: no 'Min Change (all pairs)' column found; falling "
                   f"back to the first change-like column '{chg_col}'. If this is "
                   f"a per-replicate difference column, the path-intermediate "
                   f"(scaffold) selection may be wrong. Set --change-col "
                   f"\"Min Change (all pairs) (%)\" to be explicit.")
        else:
            fail("No change column found. Provide one with --change-col "
                 f"(no recomputation is done). Columns: {cols}")

    # the change column may have been mis-detected as a WT column by prefix;
    # never let that happen.
    wt_cols = [c for c in wt_cols if c != chg_col]
    if not wt_cols:
        fail(f"No wild-type columns detected (prefix '{wt_prefix}'). WT occupancy "
             f"is required to test 'present in WT' for the path-intermediate "
             f"scaffold. Set --wt-prefix to match your WT column names. "
             f"Columns: {cols}")

    df = pd.DataFrame({
        "r1": raw[r1col].astype(str).str.strip(),
        "r2": raw[r2col].astype(str).str.strip(),
        "change": pd.to_numeric(raw[chg_col], errors="coerce"),
    })
    # mean occupancy across all WT replicates -> "present in WT" test
    df["wt_mean"] = pd.to_numeric(
        raw[wt_cols].apply(pd.to_numeric, errors="coerce").mean(axis=1),
        errors="coerce")

    # drop rows with unparseable residues / change
    def parseable(lbl):
        return bool(re.search(r"\d", str(lbl))) and bool(re.match(r"[A-Za-z]", str(lbl)))
    before = len(df)
    df = df[df["r1"].map(parseable) & df["r2"].map(parseable)].copy()
    df = df.dropna(subset=["change"]).copy()
    dropped = before - len(df)
    if dropped:
        eprint(f"WARNING: dropped {dropped} row(s) with unparseable residues "
               f"or non-numeric change.")
    if df.empty:
        fail("No usable rows after parsing.")

    meta = dict(change_col=chg_col, wt_cols=wt_cols, r1col=r1col, r2col=r2col)
    return df, meta


# --------------------------------------------------------------------------- #
# Graph construction
# --------------------------------------------------------------------------- #
def resolve_mutation(df, mutation):
    """Match the requested mutation label to the actual residue label in data."""
    want = norm_residue_key(mutation)
    all_res = pd.unique(pd.concat([df["r1"], df["r2"]], ignore_index=True))
    for r in all_res:
        if norm_residue_key(r) == want:
            return r
    examples = ", ".join(map(str, all_res[:8]))
    fail(f"Mutation residue '{mutation}' not found in data. "
         f"Example residues present: {examples} ...")


def build_full_graph(df):
    """All contacts (max |change| per pair) + backbone bonds."""
    G = nx.Graph()
    for r in df.itertuples(index=False):
        a, b = r.r1, r.r2
        if a == b:
            continue
        aw = abs(float(r.change))
        if G.has_edge(a, b) and aw <= G[a][b]["aw"]:
            continue
        G.add_edge(a, b, w=float(r.change), aw=aw, backbone=False)
    present = {resnum(n): n for n in G.nodes()}
    for num in sorted(present):
        if num + 1 in present:
            a, b = present[num], present[num + 1]
            if G.has_edge(a, b):
                G[a][b]["backbone"] = True
            else:
                G.add_edge(a, b, w=0.0, aw=0.0, backbone=True)
    return G


def dfs_disruption_network(G, df, mutation_res, sig, trav, depth):
    targets = set()
    for r in df.itertuples(index=False):
        if abs(float(r.change)) >= sig:
            targets.add(r.r1)
            targets.add(r.r2)
    targets.add(mutation_res)
    targets = {t for t in targets if t in G}

    def eligible(u, v):
        e = G[u][v]
        return e["backbone"] or e["aw"] >= trav

    kept = set()

    def dfs(cur, tgt, d, path):
        if d > depth:
            return
        for nb in G.neighbors(cur):
            if nb in path or not eligible(cur, nb):
                continue
            npath = path + [nb]
            if nb == tgt:
                for i in range(len(npath) - 1):
                    kept.add(frozenset((npath[i], npath[i + 1])))
            else:
                dfs(nb, tgt, d + 1, npath)

    tl = sorted(targets, key=resnum)
    for i in range(len(tl)):
        for j in range(i + 1, len(tl)):
            dfs(tl[i], tl[j], 1, [tl[i]])

    sub = nx.Graph()
    sub.add_nodes_from(targets)          # keep isolated targets too
    for e in kept:
        u, v = tuple(e)
        edge_data = dict(G[u][v])
        # tag non-backbone edges: "sig" if |change| >= sig threshold, else "trav"
        if not edge_data["backbone"]:
            edge_data["disrupt_kind"] = "sig" if edge_data["aw"] >= sig else "trav"
        sub.add_edge(u, v, **edge_data)
    return sub, targets


def build_structural_graph(df, wt_occ, zero_tol=0.0):
    """Contacts PRESENT IN WT and UNCHANGED by the mutation, + backbone.

    The path-intermediate (scaffold) residues are exactly the endpoints of the
    contacts retained here. A contact is retained ONLY if it is simultaneously:
      * PRESENT IN WILD-TYPE   -> mean WT occupancy >= ``wt_occ`` (default 75%),
        i.e. the contact genuinely forms in the wild-type simulation; AND
      * UNCHANGED by mutation  -> |Min Change (all pairs)| <= zero_tol. With the
        default zero_tol=0.0 this is the strict "Min Change == 0" criterion.

    Why the WT-presence gate matters: among Min Change == 0 rows, WT occupancy is
    strongly bimodal (~0% or ~100%). The high-WT rows are contacts formed in WT
    and left intact by the mutation -- the stable scaffold connector paths should
    traverse. The near-zero-WT rows are pairs that never contact in either state;
    their zero change is an artifact of absence, so routing paths through them
    would be physically meaningless. There is NO condition on the mutant side
    (selection is "present in WT only", per the requested definition).

    Edge weights
    ------------
    All retained contact edges are equally valid, so each gets uniform
    ``weight`` = CONTACT_WEIGHT (1.0): among them the path takes the fewest
    through-space hops. Backbone bonds get an EXPENSIVE weight
    (BACKBONE_WEIGHT = 10) so the path uses the chain only as a last resort and
    prefers genuine through-space contacts -- otherwise the path degenerates into
    a long crawl along the sequence (this sparse graph made that the cheapest
    route when backbone was free).
    """
    BACKBONE_WEIGHT = 10.0
    CONTACT_WEIGHT = 1.0
    SB = nx.Graph()
    n_kept = 0
    for r in df.itertuples(index=False):
        if r.r1 == r.r2:
            continue
        chg = float(r.change)
        wt_mean = float(r.wt_mean) if not pd.isna(r.wt_mean) else -np.inf
        # present in WT (mean WT occupancy >= wt_occ) AND unchanged (Min Change==0)
        if wt_mean >= wt_occ and abs(chg) <= zero_tol:
            SB.add_edge(r.r1, r.r2, weight=CONTACT_WEIGHT)
            n_kept += 1
    present = {resnum(n): n for n in SB.nodes()}
    for num in sorted(present):
        if num + 1 in present:
            a, b = present[num], present[num + 1]
            if not SB.has_edge(a, b):
                # covalent backbone: costly, so paths prefer through-space contacts
                SB.add_edge(a, b, weight=BACKBONE_WEIGHT)
    gate = ("Min Change == 0" if zero_tol == 0.0
            else f"|Min Change| <= {zero_tol:g}")
    eprint(f"Structural (path-intermediate) graph: {n_kept} contacts "
           f"(WT mean >= {wt_occ:g}% AND {gate}) + backbone; "
           f"{SB.number_of_nodes()} nodes, {SB.number_of_edges()} edges.")
    return SB


def add_shortest_path_connectors(sub, SB, mutation_res, change_lookup):
    """Link each disconnected cluster to the mutation via shortest structural path.

    Returns (final_graph, paths_dict, new_path_nodes_set).
    """
    FG = nx.Graph()
    for u, v, d in sub.edges(data=True):
        FG.add_edge(u, v,
                    kind=("backbone" if d["backbone"] else d.get("disrupt_kind", "sig")),
                    change=d["w"])
    FG.add_nodes_from(sub.nodes())
    disrupt_nodes = set(sub.nodes())
    disrupted_pairs = {frozenset((u, v)) for u, v, d in sub.edges(data=True)
                       if not d["backbone"]}

    comps = sorted(nx.connected_components(sub), key=len, reverse=True)
    other = [c for c in comps if mutation_res not in c]

    paths = {}
    path_nodes = set()

    # Weighted shortest paths from the mutation to EVERY reachable residue,
    # computed once. Edge weights are CONTACT_WEIGHT=1.0 for real scaffold
    # contacts and BACKBONE_WEIGHT=10.0 for covalent backbone bonds, so Dijkstra
    # prefers routes through measured through-space contacts and only uses the
    # backbone when no contact route exists. dist[n] = summed weight of that
    # route; sp[n] = the route itself.
    if mutation_res in SB:
        dist, sp = nx.single_source_dijkstra(SB, mutation_res, weight="weight")
    else:
        eprint(f"WARNING: mutation residue {short_label(mutation_res)} is not in "
               f"the structural scaffold (no WT-present, unchanged contact "
               f"involves it); cannot connect any cluster.")
        dist, sp = {}, {}

    for cluster in other:
        # anchor = cluster residue genuinely CLOSEST to the mutation through the
        # structural contact network (not the closest residue number).
        reachable = [n for n in cluster if n in dist]
        if not reachable:
            members = ", ".join(short_label(n) for n in sorted(cluster, key=resnum))
            eprint(f"WARNING: disrupted cluster {{{members}}} has no path to "
                   f"{short_label(mutation_res)} through the scaffold (WT-present "
                   f"AND unchanged contacts); it will remain disconnected in the "
                   f"figure. Loosen --wt-occ or --minchange-tol to admit more "
                   f"scaffold contacts if a connection is expected.")
            continue
        anchor = min(reachable, key=lambda n: dist[n])
        p = sp[anchor]              # the actual weighted shortest path
        paths[anchor] = p
        for u, v in zip(p, p[1:]):
            path_nodes.update([u, v])
            key = frozenset((u, v))
            if key in disrupted_pairs:
                continue  # already a colored disrupt edge
            if not FG.has_edge(u, v):
                FG.add_edge(u, v, kind="path",
                            change=(change_lookup.get(key, 0.0) or 0.0))

    new_path_nodes = (path_nodes - disrupt_nodes) - {mutation_res}
    return FG, paths, new_path_nodes, disrupt_nodes


# --------------------------------------------------------------------------- #
# Layout (Graphviz via pydot; engine selectable)
# --------------------------------------------------------------------------- #
GV_ENGINES = ("neato", "sfdp", "fdp", "dot", "twopi", "circo")


def graphviz_layout(FG, engine, scale, edge_length, seed):
    """Lay out FG with a Graphviz engine through the pydot interface.

    Parameters
    ----------
    engine : one of GV_ENGINES (neato/sfdp/fdp/dot/twopi/circo).
    scale  : multiplies final coordinates (spreads nodes in 2D, like the old flag).
    edge_length : target separation; passed to force engines as the spring
                  'len' (neato) / preferred edge length, scaled into inches.
    seed   : random start seed for the force engines (neato/fdp/sfdp) for
             reproducibility.

    Returns dict {node: np.array([x, y])}.  Raises RuntimeError with an
    actionable message if Graphviz/pydot are not available.
    """
    nodes = list(FG.nodes())
    if len(nodes) == 1:
        return {nodes[0]: np.array([0.0, 0.0])}

    # The pydot bridge (nx.nx_pydot.graphviz_layout) does NOT take an `args`
    # string. Graphviz attributes are passed via the graph's own attribute
    # dicts: FG.graph["graph"|"node"|"edge"]. Work on a copy so we don't mutate
    # the caller's graph or its rendering attributes.
    H = FG.copy()
    graph_attrs = {"overlap": "false", "splines": "true"}
    edge_attrs = {}

    # Per-engine Graphviz attributes.
    #   force engines (neato/fdp/sfdp): spring length + random start seed.
    #   dot: layered; ranksep/nodesep scaled by edge_length.
    #   twopi/circo: radial/circular; honor nodesep/ranksep where relevant.
    if engine in ("neato", "fdp", "sfdp"):
        graph_attrs.update({"sep": "+8", "start": str(int(seed))})
        edge_attrs["len"] = f"{edge_length:g}"
    elif engine == "dot":
        graph_attrs.update({"rankdir": "TB",
                            "nodesep": f"{0.5 * edge_length:g}",
                            "ranksep": f"{0.7 * edge_length:g}"})
    else:  # twopi, circo
        graph_attrs.update({"nodesep": f"{0.4 * edge_length:g}",
                            "ranksep": f"{0.6 * edge_length:g}"})

    H.graph["graph"] = graph_attrs
    if edge_attrs:
        H.graph["edge"] = edge_attrs

    try:
        from networkx.drawing.nx_pydot import graphviz_layout as _gv_layout
    except Exception as exc:  # networkx pydot bridge missing
        raise RuntimeError(
            "Could not import the pydot Graphviz bridge "
            "(networkx.drawing.nx_pydot). Install the Python interface with "
            "`pip install pydot`."
        ) from exc

    try:
        raw = _gv_layout(H, prog=engine)
    except Exception as exc:
        # Most common cause: the Graphviz *binaries* are not installed / not on PATH.
        raise RuntimeError(
            f"Graphviz layout with engine '{engine}' failed: {exc}\n"
            "If the message mentions a missing program (e.g. \"dot\"/\"neato\"), "
            "the Graphviz system binaries are not installed / not on PATH:\n"
            "  Ubuntu/Debian : sudo apt-get install graphviz\n"
            "  conda         : conda install -c conda-forge graphviz pydot\n"
            "  macOS         : brew install graphviz\n"
            "Then confirm `which dot` (or `dot -V`) works on the command line."
        ) from exc

    # Center at origin and apply the user's scale factor so --scale still works.
    pts = np.array([raw[n] for n in nodes], dtype=float)
    center = pts.mean(axis=0)
    pos = {n: scale * (np.array(raw[n], dtype=float) - center) for n in nodes}
    return pos


# --------------------------------------------------------------------------- #
# Overlap repair (iterative repulsion of nodes that are too close together)
# --------------------------------------------------------------------------- #
def overlap_repair(pos, mutation_res, disrupt_nodes, new_path_nodes, args):
    """Iteratively push apart any two nodes whose centers are within
    `args.overlap_padding * (r_i + r_j)` of each other.

    Node radii are derived from `args.node_size` (which is matplotlib `node_size`
    = marker area in points^2). We convert to a *data-space* radius using a
    heuristic that scales with the overall layout extent, so the repair is
    independent of figure DPI/inches.

    The mutation node is anchored (does not move) so the network stays centered
    on it.
    """
    nodes = list(pos.keys())
    if len(nodes) < 2:
        return pos

    # Data-space radius per node (same hierarchy as the renderer)
    # node_size is marker area in points^2. We need a radius in data units.
    # Use the diagonal of the current layout extent to set a sensible scale.
    P = np.array([pos[n] for n in nodes], dtype=float)
    extent = float(np.linalg.norm(P.max(axis=0) - P.min(axis=0)))
    if extent <= 0:
        return pos
    # Empirical mapping: r_data = sqrt(node_size) * extent / 200
    # (gives roughly the right radius for the matplotlib scatter rendering)
    def _r(n):
        if n == mutation_res:
            ns = args.node_size * 1.8
        elif n in new_path_nodes:
            ns = args.node_size * 0.62
        else:
            ns = args.node_size
        return np.sqrt(ns) * extent / 200.0

    radii = np.array([_r(n) for n in nodes])
    mut_mask = np.array([n == mutation_res for n in nodes])

    pad = float(args.overlap_padding)
    max_iter = int(args.overlap_iter)
    moved_any = True
    iters = 0
    while moved_any and iters < max_iter:
        moved_any = False
        iters += 1
        # Pairwise distances
        diff = P[:, None, :] - P[None, :, :]               # (N,N,2)
        dist = np.linalg.norm(diff, axis=-1)               # (N,N)
        min_d = pad * (radii[:, None] + radii[None, :])    # (N,N)
        np.fill_diagonal(dist, np.inf)
        too_close = dist < min_d
        if not np.any(too_close):
            break
        # For each overlapping pair, push both nodes apart half the deficit
        # along the unit vector.
        disp = np.zeros_like(P)
        for i in range(len(nodes)):
            for j in range(len(nodes)):
                if i == j or not too_close[i, j]:
                    continue
                d_ij = dist[i, j]
                if d_ij < 1e-9:
                    # Coincident: jitter
                    rng = np.random.default_rng(iters * 1000 + i * 31 + j)
                    u = rng.normal(size=2)
                    u /= max(np.linalg.norm(u), 1e-9)
                else:
                    u = diff[i, j] / d_ij
                deficit = min_d[i, j] - d_ij
                disp[i] += 0.5 * deficit * u
        # Mutation node anchored: zero its displacement, redistribute to partner
        if np.any(mut_mask):
            # For pairs involving the mutation, give the full deficit to the other
            disp[mut_mask] = 0.0
        if np.linalg.norm(disp) > 1e-9:
            P = P + disp
            moved_any = True
    return {nodes[i]: P[i] for i in range(len(nodes))}


# --------------------------------------------------------------------------- #
# Legend / colorbar standalone canvas
# --------------------------------------------------------------------------- #
def _make_legend_elems(args, mutation_res):
    elems = [
        Line2D([0], [0], color=COLOR_DESTABILIZED, lw=4, label="Destabilized contact (sig, mutant down)"),
        Line2D([0], [0], color=COLOR_STABILIZED,   lw=4, label="Stabilized contact (sig, mutant up)"),
        Line2D([0], [0], color=COLOR_TRAVERSAL,    lw=4, label="Traversal-threshold contact"),
        Line2D([0], [0], color=COLOR_BACKBONE, lw=3.0, label="Backbone bond"),
        Line2D([0], [0], color=COLOR_PATH, lw=1.9,
               label=f"Structural contact on shortest path (WT ≥ {args.wt_occ:g}%, unchanged)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=COLOR_NODE_DISRUPT,
               markeredgecolor=COLOR_NODE_OUTLINE, markersize=13, label="Disrupted residue"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=COLOR_NODE_PATH,
               markeredgecolor=COLOR_PATH_OUTLINE, markersize=10, label="Path-intermediate residue"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=COLOR_MUTATION,
               markeredgecolor=COLOR_NODE_OUTLINE, markersize=15,
               label=f"Mutation site ({args.mut_display})"),
    ]
    return elems


def render_legend_only(FG, mutation_res, args):
    """Render a portrait strip: vertical colorbar on top, legend symbols stacked below.
    No network is drawn. Intended for `--legend-only` mode.
    """
    cmap = LinearSegmentedColormap.from_list(
        "stab_destab", [COLOR_STABILIZED, COLOR_MIDPOINT, COLOR_DESTABILIZED])
    if FG is not None:
        sig_changes = [abs(d["change"]) for _, _, d in FG.edges(data=True)
                       if d["kind"] == "sig"]
        vmax = max(sig_changes) if sig_changes else 30.0
    else:
        vmax = 30.0
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

    fig = plt.figure(figsize=(3.2, 8.0))
    # Colorbar axis (top ~30% of canvas)
    cax = fig.add_axes([0.30, 0.62, 0.18, 0.32])
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax, orientation="vertical")
    cbar.set_label("Occupancy change\n(WT - Mutant, %)", fontsize=10)
    cax.text(0.5, 1.04, "mutant destabilized", transform=cax.transAxes,
             ha="center", va="bottom", fontsize=8)
    cax.text(0.5, -0.05, "mutant stabilized", transform=cax.transAxes,
             ha="center", va="top", fontsize=8)
    # Sign convention caption beneath colorbar
    fig.text(0.5, 0.55,
             "positive = contact lost in mutant\nnegative = gained in mutant",
             ha="center", va="top", fontsize=9, style="italic")

    # Legend axis (bottom ~50%)
    lax = fig.add_axes([0.02, 0.02, 0.96, 0.50])
    lax.axis("off")
    elems = _make_legend_elems(args, mutation_res)
    lax.legend(handles=elems, loc="center", fontsize=10,
               frameon=True, framealpha=1.0, borderpad=0.8,
               labelspacing=1.0, handlelength=2.4, handletextpad=0.8)

    plt.savefig(args.output, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Rendering (main network)
# --------------------------------------------------------------------------- #
def render(FG, pos, mutation_res, disrupt_nodes, new_path_nodes, args, n_comp_before):
    cmap = LinearSegmentedColormap.from_list(
        "stab_destab", [COLOR_STABILIZED, COLOR_MIDPOINT, COLOR_DESTABILIZED])
    sig_changes = [abs(d["change"]) for _, _, d in FG.edges(data=True)
                   if d["kind"] == "sig"]
    vmax = max(sig_changes) if sig_changes else 1.0
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

    def clamp_color(change):
        t = norm(change)
        t = 0.5 + max(t - 0.5, COLOR_CLAMP) if t >= 0.5 else 0.5 - max(0.5 - t, COLOR_CLAMP)
        return cmap(min(max(t, 0.0), 1.0))

    sig_e  = [(u, v, d) for u, v, d in FG.edges(data=True) if d["kind"] == "sig"]
    trav_e = [(u, v, d) for u, v, d in FG.edges(data=True) if d["kind"] == "trav"]
    bb_e   = [(u, v, d) for u, v, d in FG.edges(data=True) if d["kind"] == "backbone"]
    path_e = [(u, v, d) for u, v, d in FG.edges(data=True) if d["kind"] == "path"]

    node_disrupt = [n for n in FG.nodes() if n in disrupt_nodes and n != mutation_res]
    node_path = [n for n in FG.nodes() if n in new_path_nodes]
    labels = {n: short_label(n) for n in FG.nodes()}
    # Override the center-residue node label with the display name (e.g. L129)
    # when --mutation-label is set; other nodes keep their data-derived labels.
    if getattr(args, "mut_display", None):
        labels[mutation_res] = args.mut_display

    ns_d = args.node_size
    ns_p = args.node_size * 0.62
    ns_m = args.node_size * 1.8

    # Figure size from --fig-inches (default 8.27 x 11.69 = A4 portrait)
    fig, ax = plt.subplots(figsize=tuple(args.fig_inches))

    nx.draw_networkx_edges(FG, pos, edgelist=[(u, v) for u, v, _ in bb_e],
                           width=3.0, edge_color=COLOR_BACKBONE, ax=ax)
    nx.draw_networkx_edges(FG, pos, edgelist=[(u, v) for u, v, _ in path_e],
                           width=1.9, edge_color=COLOR_PATH, ax=ax)
    for u, v, d in sig_e:
        w = 1.8 + 6.0 * (abs(d["change"]) / vmax)
        nx.draw_networkx_edges(FG, pos, edgelist=[(u, v)], width=w,
                               edge_color=[clamp_color(d["change"])], ax=ax)
    for u, v, d in trav_e:
        w = 1.8 + 6.0 * (abs(d["change"]) / vmax)
        nx.draw_networkx_edges(FG, pos, edgelist=[(u, v)], width=w,
                               edge_color=[COLOR_TRAVERSAL], ax=ax)

    def draw_nodes(nl, fc, ec, lw, size):
        if not nl:
            return
        nx.draw_networkx_nodes(FG, pos, nodelist=nl, node_color=HALO_COLOR,
                               edgecolors=HALO_COLOR, linewidths=0,
                               node_size=size * 1.05, ax=ax)
        nx.draw_networkx_nodes(FG, pos, nodelist=nl, node_color=fc,
                               edgecolors=ec, linewidths=lw, node_size=size, ax=ax)

    draw_nodes(node_disrupt, COLOR_NODE_DISRUPT, COLOR_NODE_OUTLINE, 1.4, ns_d)
    draw_nodes(node_path, COLOR_NODE_PATH, COLOR_PATH_OUTLINE, 1.1, ns_p)
    draw_nodes([mutation_res], COLOR_MUTATION, COLOR_NODE_OUTLINE, 2.2, ns_m)

    nx.draw_networkx_labels(FG, pos, labels=labels,
                            font_size=args.font_size, font_weight="bold", ax=ax)

    # Colorbar — suppressed when --no-colorbar
    if not args.no_colorbar:
        sm = ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, fraction=0.035, pad=0.02)
        cbar.set_label("Occupancy change  (WT - Mutant, %)", fontsize=10)
        cbar.ax.text(0.5, 1.02, "mutant destabilized", transform=cbar.ax.transAxes,
                     ha="center", va="bottom", fontsize=8)
        cbar.ax.text(0.5, -0.04, "mutant stabilized", transform=cbar.ax.transAxes,
                     ha="center", va="top", fontsize=8)

    # Legend — suppressed when --no-legend
    if not args.no_legend:
        legend_elems = _make_legend_elems(args, mutation_res)
        ax.legend(handles=legend_elems, loc="lower center", fontsize=12,
                  frameon=True, framealpha=0.95)

    # Title — suppressed when --no-title
    if not args.no_title:
        ax.set_title(
            f"ContactWalker disruption network - {args.mut_display}  "
            f"(Graphviz: {args.gv_engine})\n"
            f"significance {args.sig:g}%, traversal {args.trav:g}%, depth {args.depth}  |  "
            f"{FG.number_of_nodes()} nodes, {FG.number_of_edges()} edges  |  "
            f"clusters before linking: {n_comp_before}",
            fontsize=13)

    ax.axis("off")
    ax.margins(args.margin)
    plt.tight_layout()
    plt.savefig(args.output, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Console summary
# --------------------------------------------------------------------------- #
def print_summary(FG, paths, mutation_res, n_comp_before, n_comp_after, args=None):
    n_disrupt = sum(1 for _, _, d in FG.edges(data=True) if d["kind"] == "disrupt")
    n_bb = sum(1 for _, _, d in FG.edges(data=True) if d["kind"] == "backbone")
    n_path = sum(1 for _, _, d in FG.edges(data=True) if d["kind"] == "path")

    print("\n" + "=" * 70)
    mut_lbl = args.mut_display if (args is not None and getattr(args, "mut_display", None)) else short_label(mutation_res)
    print(f"ContactWalker network summary  (center residue: {mut_lbl})")
    print("=" * 70)
    print(f"Nodes: {FG.number_of_nodes()}   Edges: {FG.number_of_edges()}")
    print(f"  disrupted-contact edges : {n_disrupt}")
    print(f"  backbone bonds          : {n_bb}")
    print(f"  shortest-path contacts  : {n_path}")
    print(f"Connected components before path linking: {n_comp_before}")
    print(f"Connected components after  path linking: {n_comp_after}")
    print(f"Center-residue degree: {FG.degree(mutation_res)}")

    if paths:
        print("\nShortest structural paths (center -> cluster anchor):")
        for anchor, p in sorted(paths.items(), key=lambda kv: len(kv[1])):
            seg = " -> ".join(short_label(x) for x in p)
            print(f"  to {short_label(anchor):>6} ({len(p)-1} steps): {seg}")

    disrupt = [(u, v, d["change"]) for u, v, d in FG.edges(data=True)
               if d["kind"] == "disrupt"]
    disrupt.sort(key=lambda t: abs(t[2]), reverse=True)
    if disrupt:
        print("\nTop disrupted contacts by |change|:")
        for u, v, c in disrupt[:12]:
            tag = "mutant-stabilized" if c < 0 else "mutant-destabilized"
            print(f"  {short_label(u):>6} - {short_label(v):<6}  "
                  f"{c:+7.2f}%   ({tag})")
    print("=" * 70 + "\n")


# --------------------------------------------------------------------------- #
# Main / CLI
# --------------------------------------------------------------------------- #
def build_arg_parser():
    p = argparse.ArgumentParser(
        description="Recreate a ContactWalker contact-disruption network figure "
                    "from a contact-occupancy CSV.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("input_csv", help="Path to the contacts CSV.")
    p.add_argument("--mutation", required=True,
                   help="Mutation/center residue label as in the data, e.g. ILE129. "
                        "This is used for DATA MATCHING only (it must exist in the CSV).")
    p.add_argument("--mutation-label", default=None,
                   help="Optional DISPLAY label for the center residue in the figure, "
                        "legend, title, and console summary (e.g. LEU129 to show the "
                        "mutant Leu instead of the WT Ile). Does NOT affect data matching "
                        "or the --mutation residue. If omitted, the --mutation label is "
                        "used for display as before.")
    p.add_argument("--sig", type=float, default=2.2,
                   help="Significance threshold (percentage points) for target residues.")
    p.add_argument("--trav", type=float, default=1.2,
                   help="Allowable traversal threshold (percentage points) for DFS edges.")
    p.add_argument("--depth", type=int, default=1, help="DFS search depth.")
    p.add_argument("--minchange-tol", type=float, default=0.0,
                   help="Tolerance for the 'unchanged' half of the path-intermediate "
                        "(scaffold) gate. A contact passes this half when "
                        "|Min Change (all pairs)| <= this value. Default 0.0 means "
                        "strict Min Change == 0. Raise (e.g. 0.05) to also admit "
                        "near-zero rounding as unchanged. Combined (AND) with --wt-occ.")
    p.add_argument("--wt-prefix", default="WT",
                   help="Case-insensitive column-name prefix identifying the "
                        "wild-type replicate occupancy columns (e.g. WT1, WT2, WT3). "
                        "Their row-wise mean defines wild-type presence. Default 'WT'.")
    p.add_argument("--wt-occ", type=float, default=75.0,
                   help="Mean WT occupancy (%%) a contact must reach to count as "
                        "present in wild-type. Forms the second half of the "
                        "path-intermediate gate: a contact is a scaffold connector "
                        "only if WT mean >= this AND it is unchanged (--minchange-tol). "
                        "Default 75.0. No condition is placed on the mutant side.")
    p.add_argument("--change-col", default=None,
                   help="Name of the signed-change column used for edge colors AND "
                        "the path-intermediate scaffold gate. If omitted, a column "
                        "named like 'Min Change (all pairs)' is preferred, else the "
                        "first column containing 'change'.")
    # layout
    p.add_argument("--gv-engine", default="neato", choices=list(GV_ENGINES),
                   help="Graphviz layout engine (via pydot). neato/sfdp/fdp are "
                        "force-directed; dot is hierarchical; twopi/circo are "
                        "radial/circular.")
    # spacing / sizing
    p.add_argument("--scale", type=float, default=1.0,
                   help="Global layout scale factor (spreads nodes in 2D space).")
    p.add_argument("--edge-length", type=float, default=1.0,
                   help="Target edge length passed to the Graphviz engine "
                        "(spring 'len' for force engines; node/rank separation "
                        "for dot/twopi/circo).")
    p.add_argument("--node-size", type=float, default=640.0,
                   help="Base node area for disrupted residues "
                        "(mutation/path nodes scale from this).")
    p.add_argument("--font-size", type=float, default=8.5, help="Residue label font size.")
    # output
    p.add_argument("--output", default=None,
                   help="Output PNG path (default: <input_stem>_network.png).")
    p.add_argument("--dpi", type=int, default=200, help="Output resolution.")
    p.add_argument("--seed", type=int, default=0,
                   help="RNG seed for layout-initialization search (reproducibility).")
    # Figure shape and split-render flags (added in v2)
    p.add_argument("--fig-inches", type=float, nargs=2, default=[8.27, 11.69],
                   metavar=("W", "H"),
                   help="Figure size in inches (W H). Default 8.27 11.69 = A4 portrait.")
    p.add_argument("--no-colorbar", action="store_true",
                   help="Suppress the colorbar (use --legend-only to render it separately).")
    p.add_argument("--no-legend", action="store_true",
                   help="Suppress the categorical legend.")
    p.add_argument("--no-title", action="store_true",
                   help="Suppress the figure title.")
    p.add_argument("--legend-only", action="store_true",
                   help="Render only the colorbar + categorical legend on a small portrait "
                        "canvas; skip the network. CSV is still required (used to size the "
                        "colorbar range from the data); pass any CSV.")
    p.add_argument("--margin", type=float, default=0.08,
                   help="Axis margin around node centers (fraction of axis range).")
    # Overlap repair
    p.add_argument("--no-overlap-fix", action="store_true",
                   help="Skip the post-layout iterative repulsion pass.")
    p.add_argument("--overlap-iter", type=int, default=200,
                   help="Maximum iterations of the post-layout repulsion pass.")
    p.add_argument("--overlap-padding", type=float, default=1.05,
                   help="Minimum node-to-node distance as a multiple of (r_i + r_j).")
    return p


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    if args.output is None:
        stem = re.sub(r"\.csv$", "", args.input_csv, flags=re.IGNORECASE)
        args.output = f"{stem}_network.png"

    df, meta = parse_csv(args.input_csv, args.wt_prefix, args.change_col)
    eprint(f"Parsed {len(df)} contacts. change column='{meta['change_col']}'; "
           f"WT columns={meta['wt_cols']}.")

    mutation_res = resolve_mutation(df, args.mutation)

    # Display label for the center residue: --mutation-label overrides the shown
    # name (e.g. LEU129 for the mutant Leu) WITHOUT affecting data matching,
    # which still uses mutation_res (resolved from --mutation). Falls back to the
    # short form of the matched residue when --mutation-label is not given.
    if args.mutation_label:
        args.mut_display = display_short_label(args.mutation_label)
    else:
        args.mut_display = short_label(mutation_res)

    # change lookup (max-|change| per pair), for coloring path edges if disrupted
    change_lookup = {}
    for r in df.itertuples(index=False):
        k = frozenset((r.r1, r.r2))
        c = float(r.change)
        if k not in change_lookup or abs(c) > abs(change_lookup[k]):
            change_lookup[k] = c

    G = build_full_graph(df)
    sub, targets = dfs_disruption_network(G, df, mutation_res,
                                          args.sig, args.trav, args.depth)
    n_comp_before = nx.number_connected_components(sub)

    SB = build_structural_graph(df, args.wt_occ, zero_tol=args.minchange_tol)
    FG, paths, new_path_nodes, disrupt_nodes = add_shortest_path_connectors(
        sub, SB, mutation_res, change_lookup)
    n_comp_after = nx.number_connected_components(FG)

    # Legend-only mode: skip layout & network drawing, render legend strip and exit.
    if args.legend_only:
        render_legend_only(FG, mutation_res, args)
        print_summary(FG, paths, mutation_res, n_comp_before, n_comp_after, args)
        print(f"Legend strip written to: {args.output}")
        return

    pos = graphviz_layout(FG, args.gv_engine, args.scale, args.edge_length, args.seed)

    # Post-layout iterative overlap repair (unless --no-overlap-fix).
    if not args.no_overlap_fix:
        pos = overlap_repair(pos, mutation_res, disrupt_nodes, new_path_nodes, args)

    render(FG, pos, mutation_res, disrupt_nodes, new_path_nodes, args, n_comp_before)

    print_summary(FG, paths, mutation_res, n_comp_before, n_comp_after, args)
    print(f"Figure written to: {args.output}")


if __name__ == "__main__":
    main()
