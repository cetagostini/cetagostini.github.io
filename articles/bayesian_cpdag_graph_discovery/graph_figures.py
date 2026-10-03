"""Figures for the Bayesian CPDAG graph-discovery article.

Pure presentation: every public function takes finished draws (or raw data)
and returns a matplotlib ``Figure``. Nothing here fits, samples, writes
files, or mutates global style — the notebook displays the figures.

Conventions shared with ``graph_math``: pairs follow ``itertools.combinations``
order, pair states are 0 = absent, 1 = first to second, 2 = second to first,
and source arrays have shape (draws, pairs). DAG identifiers are Python
integers, little-endian base 3 over pair states; they are labels, never a
graph metric. Truth is optional everywhere and is never assumed to be
sampled, ranked, or of nonzero empirical mass.
"""

from collections import Counter
from itertools import combinations
from math import ceil
from textwrap import fill

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Rectangle

from cetagostini.style import COLORS

_SAGE = LinearSegmentedColormap.from_list(
    "sage", [COLORS["bg"], COLORS["primary"], COLORS["green_strong"]]
)
_STATE_COLORS = [COLORS["ink_muted"], COLORS["green_strong"], COLORS["brown"]]
_POS7 = {0: (-1.7, 1.9), 1: (-0.5, 2.4), 2: (0.6, 1.9),
         3: (1.9, 2.4), 4: (-1.25, 0.7), 5: (1.4, 0.8), 6: (0.0, -0.5)}
_MARK = 2200.0  # scatter points^2 per unit posterior mass on the graph map


# ------------------------------------------------------------- helpers ----
def _labels(labels):
    out = [str(item) for item in labels]
    if len(out) < 2:
        raise ValueError("labels must name at least two nodes")
    return out


def _pairs_for(n):
    return list(combinations(range(n), 2))


def _as_draws(states):
    arr = np.asarray(states)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2 or arr.shape[0] < 1 or arr.shape[1] < 1:
        raise ValueError("states must have shape (draws, pairs) with at least one draw")
    if arr.dtype.kind == "f":
        if not np.isfinite(arr).all() or not np.all(arr == np.floor(arr)):
            raise ValueError("pair states must be integers")
        arr = arr.astype(np.int64)
    if arr.dtype.kind not in "iu" or np.any((arr < 0) | (arr > 2)):
        raise ValueError("pair states must be 0 (absent), 1 (first to second) or 2 (second to first)")
    return arr.astype(np.int64)


def _as_truth(truth_states, n_pairs):
    if truth_states is None:
        return None
    row = np.asarray(truth_states)
    if row.ndim == 2 and row.shape[0] == 1:
        row = row[0]
    if row.ndim != 1 or row.shape[0] != n_pairs:
        raise ValueError(f"truth_states must have shape ({n_pairs},)")
    return _as_draws(row)[0]


def _check_pairs(n, n_pairs):
    if n_pairs != n * (n - 1) // 2:
        raise ValueError("states must cover every unordered pair of the labels")


def _dag_id(row):
    code = 0
    for k, state in enumerate(row):
        code += int(state) * 3 ** k
    return int(code)


def _skeleton_id(row):
    code = 0
    for k, state in enumerate(row):
        if int(state):
            code |= 1 << k
    return int(code)


def _masses(keys):
    """Empirical PMF over Python-int keys, sorted by mass then key."""
    counts = Counter(keys)
    total = len(keys)
    items = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return [k for k, _ in items], np.array([c / total for _, c in items], dtype=float)


def _state_masses(draws):
    """Unique state rows with masses, sorted by mass then DAG id."""
    uniq, counts = np.unique(draws, axis=0, return_counts=True)
    ids = [_dag_id(row) for row in uniq]
    order = sorted(range(len(ids)), key=lambda t: (-int(counts[t]), ids[t]))
    rows = [uniq[t].copy() for t in order]
    return rows, [ids[t] for t in order], np.array(
        [counts[t] / counts.sum() for t in order], dtype=float)


def _node_positions(n):
    if n == 7:
        return dict(_POS7)
    pos = nx.spring_layout(nx.empty_graph(n), seed=0)
    return {k: (2.0 * float(v[0]), 2.0 * float(v[1])) for k, v in pos.items()}


def _draw_dag(ax, row, labels, title=None):
    """Draw one DAG from its pair-state row at deterministic positions."""
    n = len(labels)
    pairs = _pairs_for(n)
    pos = _node_positions(n)
    graph = nx.empty_graph(n)
    nx.draw_networkx_nodes(graph, pos, node_size=650,
                           node_color=COLORS["surface_alt"],
                           edgecolors=COLORS["primary"], ax=ax)
    nx.draw_networkx_labels(graph, pos, labels=dict(enumerate(labels)),
                            font_color=COLORS["ink"], font_weight="bold",
                            font_size=9, ax=ax)
    for state, (i, j) in zip(row, pairs):
        if int(state) == 0:
            continue
        start, end = (i, j) if int(state) == 1 else (j, i)
        ax.add_patch(FancyArrowPatch(pos[start], pos[end], arrowstyle="-|>",
                                     mutation_scale=14, linewidth=1.6,
                                     shrinkA=16, shrinkB=16,
                                     color=COLORS["green_strong"]))
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    ax.set(xlim=(min(xs) - 0.4, max(xs) + 0.4), ylim=(min(ys) - 0.4, max(ys) + 0.4),
           aspect="equal")
    if title is not None:
        ax.set_title(title, fontsize=10, fontweight="bold", color=COLORS["ink"])
    ax.axis("off")


def _axes_style(ax):
    ax.set_facecolor(COLORS["bg"])
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(COLORS["line"])
    ax.grid(axis="y", color=COLORS["line"], alpha=0.6, linewidth=0.8)
    ax.tick_params(colors=COLORS["ink_muted"])


def _note(fig, text, y=0.008):
    text = fill(text, width=max(40, int(fig.get_figwidth() * 13)))
    fig.text(0.02, y, text, fontsize=8, color=COLORS["ink_muted"], ha="left", va="bottom")


def _parents_of(row, n, pairs):
    parents = [set() for _ in range(n)]
    for state, (i, j) in zip(row, pairs):
        if int(state) == 1:
            parents[j].add(i)
        elif int(state) == 2:
            parents[i].add(j)
    return parents


def _response_and_target(truth, n):
    """Pick a response node and one observed parent to display against it.

    The response is the unique sink of the generating graph when there is
    one, else the last column; the parent is its last parent in label order.
    """
    y_idx = n - 1
    if truth is not None:
        parents = _parents_of(truth, n, _pairs_for(n))
        sinks = [u for u in range(n)
                 if all(u not in ps for ps in parents)]
        if len(sinks) == 1:
            y_idx = sinks[0]
        pa = sorted(parents[y_idx])
    else:
        pa = []
    f_idx = pa[-1] if pa else (y_idx - 1 if y_idx > 0 else 1)
    return f_idx, y_idx


def _eval_response(response, grid):
    try:
        out = np.asarray(response(grid), dtype=float)
        if out.shape != grid.shape:
            raise ValueError
    except (TypeError, ValueError):
        out = np.array([float(response(float(x))) for x in grid])
    return out


# ------------------------------------------------------- 01 the process ----
def plot_process(data, labels, truth_states, response):
    """Generating DAG, response callable vs a linear reference, observed scatter.

    The scatter is the raw association between one parent and the response
    with every other parent and the noise still inside — not an isolated
    causal response.
    """
    names = _labels(labels)
    n = len(names)
    data = np.asarray(data, dtype=float)
    if data.ndim != 2 or data.shape[0] < 1 or data.shape[1] != n:
        raise ValueError(f"data must have shape (observations, {n})")
    if not callable(response):
        raise ValueError("response must be callable")
    pairs = _pairs_for(n)
    truth = _as_truth(truth_states, len(pairs))
    f_idx, y_idx = _response_and_target(truth, n)

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.7), layout="none")
    fig.set_facecolor(COLORS["bg"])

    ax = axes[0]
    if truth is None:
        ax.axis("off")
        ax.text(0.5, 0.5, "no generating graph supplied", ha="center", va="center",
                fontsize=10, color=COLORS["ink_muted"], transform=ax.transAxes)
    else:
        _draw_dag(ax, truth, names, "The generating graph")

    ax = axes[1]
    _axes_style(ax)
    inputs = sorted(_parents_of(truth, n, pairs)[y_idx]) if truth is not None else []
    input_data = data[:, inputs or [f_idx]]
    lo, hi = float(np.min(input_data)), float(np.max(input_data))
    if not lo < hi:
        lo, hi = lo - 1.0, hi + 1.0
    grid = np.linspace(lo, hi, 200)
    curve = _eval_response(response, grid)
    slope, intercept = np.polyfit(grid, curve, 1)
    ax.plot(grid, curve, color=COLORS["green_strong"], lw=2,
            label="Generating response h(x)")
    ax.plot(grid, slope * grid + intercept, color=COLORS["brown"], lw=1.8, ls="--",
            label="Least-squares line\n(over this plotted range)")
    ax.set(xlabel="Parent index x", ylabel="Response h(x)",
           title=f"Response into {names[y_idx]}")
    ax.legend(frameon=False, fontsize=8.5)

    ax = axes[2]
    _axes_style(ax)
    ax.scatter(data[:, f_idx], data[:, y_idx], s=8, alpha=0.45,
               color=COLORS["primary"], rasterized=True)
    ax.set(xlabel=names[f_idx], ylabel=names[y_idx],
           title=f"Observed {names[y_idx]} against {names[f_idx]}")
    ax.text(0.02, 0.02,
            "Other causes and noise remain.\nNot an isolated causal response.",
            transform=ax.transAxes, fontsize=8, color=COLORS["ink_muted"])

    fig.suptitle("The data-generating process", fontsize=12,
                 fontweight="bold", color=COLORS["ink"])
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
    return fig


# --------------------------------------------- 03 generating-state probs ---
def plot_truth_probabilities(states, linear_states, truth_states, labels):
    """Posterior probability of each pair's generating state, two likelihoods.

    Both posteriors share the same proper priors except for the likelihood
    basis; no probability threshold is guaranteed.
    """
    names = _labels(labels)
    n = len(names)
    pairs = _pairs_for(n)
    truth = _as_truth(truth_states, len(pairs))
    if truth is None:
        raise ValueError("truth_states is required: the figure shows each pair's generating state")
    draws = _as_draws(states)
    linear = _as_draws(linear_states)
    _check_pairs(n, draws.shape[1])
    if linear.shape[1] != draws.shape[1]:
        raise ValueError("states and linear_states must cover the same pairs")

    p_basis = np.mean(draws == truth[None, :], axis=0)
    p_linear = np.mean(linear == truth[None, :], axis=0)
    rows = np.arange(len(pairs))

    fig, ax = plt.subplots(figsize=(8.4, 0.44 * len(pairs) + 2.6), layout="none")
    fig.set_facecolor(COLORS["bg"])
    _axes_style(ax)
    ax.barh(rows - 0.19, p_basis, height=0.36, color=COLORS["green_strong"],
            label="nonlinear basis likelihood")
    ax.barh(rows + 0.19, p_linear, height=0.36, color=COLORS["brown"],
            label="linear likelihood")

    tick_labels = []
    for k, (i, j) in enumerate(pairs):
        if truth[k] == 1:
            desc = f"{names[i]} → {names[j]}"
        elif truth[k] == 2:
            desc = f"{names[j]} → {names[i]}"
        else:
            desc = "no arrow"
        tick_labels.append(f"{names[i]} – {names[j]}   ({desc})")
    ax.set(yticks=rows, yticklabels=tick_labels, xlim=(0, 1.02),
           xlabel="posterior probability of the pair's generating state",
           title="How often each pair lands on its generating state")
    ax.tick_params(axis="y", labelsize=8.5)
    ax.invert_yaxis()
    ax.legend(frameon=False, fontsize=9)
    _note(fig, "Same proper priors for both models except the likelihood basis; "
               "no decision threshold is guaranteed.", y=0.005)
    fig.tight_layout(rect=(0, 0.025, 1, 1))
    return fig


# ---------------------------------------------------- 04 arrow marginals ---
def plot_directions(states, labels):
    """Source-by-target matrix of marginal arrow probabilities.

    Off-diagonal zeros are arrows absent in the retained draws; diagonal
    cells are impossible self arrows. Nothing is thresholded into a DAG.
    """
    names = _labels(labels)
    n = len(names)
    pairs = _pairs_for(n)
    draws = _as_draws(states)
    _check_pairs(n, draws.shape[1])

    prob = np.zeros((n, n))
    for k, (i, j) in enumerate(pairs):
        prob[i, j] = np.mean(draws[:, k] == 1)
        prob[j, i] = np.mean(draws[:, k] == 2)

    fig, ax = plt.subplots(figsize=(7.4, 6.4), layout="none")
    fig.set_facecolor(COLORS["bg"])
    ax.set_facecolor(COLORS["bg"])
    masked = np.ma.masked_where(np.eye(n, dtype=bool), prob)
    vmax = float(prob.max()) if prob.max() > 0 else 1.0
    im = ax.imshow(masked, cmap=_SAGE, vmin=0.0, vmax=vmax)
    for i in range(n):
        ax.add_patch(Rectangle((i - 0.5, i - 0.5), 1, 1,
                               facecolor=COLORS["surface_alt"],
                               edgecolor=COLORS["ink_muted"], linewidth=0.8, hatch="///"))
    for i in range(n):
        for j in range(n):
            if i == j:
                text, color = "—", COLORS["ink_muted"]
            else:
                text = f"{prob[i, j]:.2f}"
                color = COLORS["bg"] if prob[i, j] > 0.55 * vmax else COLORS["ink"]
            ax.text(j, i, text, ha="center", va="center", fontsize=8, color=color)
    ax.set(xticks=range(n), yticks=range(n), xticklabels=names, yticklabels=names,
           xlabel="target", ylabel="source",
           title="Marginal posterior probability of each arrow")
    ax.grid(False)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    bar = fig.colorbar(im, ax=ax, shrink=0.82)
    bar.set_label("marginal arrow probability", fontsize=9, color=COLORS["ink"])
    bar.ax.tick_params(colors=COLORS["ink_muted"])
    _note(fig, "Each cell is the marginal probability of that arrow; the pair's absence "
               "probability is 1 minus the two opposing cells. Hatched cells are impossible "
               "self arrows.", y=0.005)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    return fig


# --------------------------------------------------- 05/06 ranked classes --
def plot_ranked_graphs(states, truth_states, labels, *, skeleton=False):
    """Ranked frequency PMF of DAGs or skeleton classes with truth located.

    Skeleton classes group on edge presence, summing every orientation; a
    skeleton class is not a CPDAG class.
    """
    names = _labels(labels)
    n = len(names)
    pairs = _pairs_for(n)
    draws = _as_draws(states)
    _check_pairs(n, draws.shape[1])
    truth = _as_truth(truth_states, len(pairs))

    if skeleton:
        keys = [_skeleton_id(row) for row in draws]
        truth_key = _skeleton_id(truth) if truth is not None else None
        kind, plural = "skeleton class", "skeleton classes"
    else:
        keys = [_dag_id(row) for row in draws]
        truth_key = _dag_id(truth) if truth is not None else None
        kind, plural = "DAG", "DAGs"
    ids, masses = _masses(keys)
    rank = np.arange(1, len(masses) + 1)

    fig, ax = plt.subplots(figsize=(8, 4.9), layout="none")
    _axes_style(ax)
    ax.plot(rank, masses, "-", lw=0.9, color=COLORS["ink_muted"], zorder=1)
    ax.plot(rank, masses, "o", ms=4, color=COLORS["green_strong"], zorder=2,
            label=f"visited {plural}")
    resolution = 1.0 / draws.shape[0]
    ax.axhline(resolution, color=COLORS["brown"], ls=":", lw=1.1,
               label="one retained draw")

    if truth_key is not None:
        if truth_key in ids:
            r = ids.index(truth_key) + 1
            m = float(masses[r - 1])
            ax.plot([r], [m], marker="o", ms=11, mfc="none", mec=COLORS["ink"],
                    mew=1.6, zorder=3, label=f"true {kind}")
            ax.annotate(f"true {kind} id={truth_key}: rank {r}, mass {m:.2%}",
                        xy=(r, m), xytext=(10, 12), textcoords="offset points",
                        fontsize=8.5, color=COLORS["ink"],
                        arrowprops=dict(arrowstyle="-", color=COLORS["ink"], lw=0.8))
        else:
            ax.text(0.03, 0.07,
                    f"true {kind} id={truth_key}: not among the visited {plural}",
                    transform=ax.transAxes, fontsize=9, color=COLORS["ink"])

    ax.set(xscale="log", yscale="log", xlabel="rank (most probable first)",
           ylabel="posterior probability",
           title=f"Ranked posterior over {plural}")
    ax.set_xlim(0.8, max(2.0, 1.4 * len(masses)))
    ax.set_ylim(max(0.4 * min(masses[-1], resolution), 1e-12), 2.2 * masses[0])
    ax.legend(frameon=False, fontsize=9)
    if skeleton:
        _note(fig, "A skeleton class sums every orientation of the same edges; "
                   "this is not a CPDAG class.", y=0.008)
    else:
        _note(fig, "Each point is one DAG identified by its base-3 id.", y=0.008)
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    return fig


# ------------------------------------------------------- 07 edge counts ----
def plot_edge_counts(states, prior_states, truth_states):
    """Empirical PMFs of arrow counts under the prior and the posterior."""
    draws = _as_draws(states)
    prior = _as_draws(prior_states)
    if prior.shape[1] != draws.shape[1]:
        raise ValueError("states and prior_states must cover the same pairs")
    truth = _as_truth(truth_states, draws.shape[1])

    post = (draws != 0).sum(axis=1)
    prior_counts = (prior != 0).sum(axis=1)
    true_count = int((truth != 0).sum()) if truth is not None else 0
    hi = int(max(post.max(), prior_counts.max(), true_count))
    xs = np.arange(hi + 1)
    post_p = np.bincount(post, minlength=hi + 1)[: hi + 1] / post.size
    prior_p = np.bincount(prior_counts, minlength=hi + 1)[: hi + 1] / prior_counts.size

    fig, ax = plt.subplots(figsize=(8, 4.9), layout="none")
    _axes_style(ax)
    ax.plot(xs, prior_p, "o-", ms=5, lw=1.5, color=COLORS["brown"], label="prior draws")
    ax.plot(xs, post_p, "o-", ms=5, lw=1.5, color=COLORS["green_strong"], label="posterior draws")
    if truth is not None:
        ax.axvline(true_count, color=COLORS["ink"], ls="--", lw=1.2,
                   label=f"true graph: {true_count} arrows")
    ax.set(xlabel="number of arrows in the graph", ylabel="probability",
           xticks=xs, ylim=(0, 1.05 * max(prior_p.max(), post_p.max())),
           title="Arrow counts: prior vs posterior")
    ax.legend(frameon=False, fontsize=9)
    _note(fig, "Prior counts come from proper acyclic graph draws — the pair states are "
               "not independent Bernoulli trials.", y=0.008)
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    return fig


# ---------------------------------------------------- 08 parent counts -----
def plot_parent_counts(states, truth_states, labels):
    """Small-multiple PMFs of each node's parent count, true count ringed."""
    names = _labels(labels)
    n = len(names)
    pairs = _pairs_for(n)
    draws = _as_draws(states)
    _check_pairs(n, draws.shape[1])
    truth = _as_truth(truth_states, len(pairs))

    parent = np.zeros((draws.shape[0], n), dtype=np.int64)
    for k, (i, j) in enumerate(pairs):
        parent[:, j] += draws[:, k] == 1
        parent[:, i] += draws[:, k] == 2
    if truth is not None:
        true_parent = np.zeros(n, dtype=np.int64)
        for k, (i, j) in enumerate(pairs):
            true_parent[j] += truth[k] == 1
            true_parent[i] += truth[k] == 2

    cols = min(4, n)
    rows = ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(3.1 * cols + 1.2, 2.7 * rows + 1.2),
                             sharex=True, sharey=True, layout="none")
    fig.set_facecolor(COLORS["bg"])
    axes = np.atleast_1d(axes).ravel()
    xs = np.arange(n)
    for node in range(n):
        ax = axes[node]
        _axes_style(ax)
        pmf = np.bincount(parent[:, node], minlength=n) / parent.shape[0]
        ax.plot(xs, pmf, "o-", ms=5, lw=1.5, color=COLORS["green_strong"])
        if truth is not None:
            tc = int(true_parent[node])
            ax.plot([tc], [pmf[tc]], marker="o", ms=11, mfc="none",
                    mec=COLORS["ink"], mew=1.5)
        ax.set(xticks=xs, xlim=(-0.5, n - 0.5))
        ax.set_title(names[node], fontsize=10, fontweight="bold", color=COLORS["ink"])
    for node in range(n, len(axes)):
        axes[node].axis("off")
    for node in range(n):
        if (node // cols) == rows - 1 or n <= cols:
            axes[node].set_xlabel("number of parents")
        if node % cols == 0:
            axes[node].set_ylabel("probability")
    handles = [Line2D([], [], color=COLORS["green_strong"], marker="o", lw=1.5,
                      label="posterior mass")]
    if truth is not None:
        handles.append(Line2D([], [], color=COLORS["ink"], marker="o", mfc="none",
                              lw=0, ms=10, mew=1.5, label="true count"))
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, fontsize=9)
    fig.suptitle("Parents per node", fontsize=12, fontweight="bold",
                 color=COLORS["ink"])
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    return fig


# ------------------------------------------------------- 09 the map --------
def plot_graph_map(states, truth_states, labels, *, limit=80):
    """Hamming-MDS map of the leading visited DAGs with the truth always drawn.

    Circle area is exactly proportional to empirical mass; the true code is
    added even when unvisited, as a zero-mass cross. Geometry is approximate.
    """
    names = _labels(labels)
    n = len(names)
    pairs = _pairs_for(n)
    draws = _as_draws(states)
    _check_pairs(n, draws.shape[1])
    truth = _as_truth(truth_states, len(pairs))
    limit = int(limit)
    if limit < 1:
        raise ValueError("limit must be a positive integer")

    rows, ids, masses = _state_masses(draws)
    top = min(limit, len(rows))
    chosen = list(range(top))
    truth_pos = None
    if truth is not None:
        for t, row in enumerate(rows):
            if np.array_equal(row, truth):
                truth_pos = t
                break
        if truth_pos is not None and truth_pos not in chosen:
            chosen.append(truth_pos)

    codes = [rows[t] for t in chosen]
    point_mass = [float(masses[t]) for t in chosen]
    truth_flags = [t == truth_pos for t in chosen]
    if truth is not None and truth_pos is None:
        codes.append(np.asarray(truth, dtype=np.int64))
        point_mass.append(0.0)
        truth_flags.append(True)

    xy = _mds_2d(np.asarray(codes, dtype=np.int64))
    visited = np.array([m > 0 for m in point_mass])
    truth_arr = np.array(truth_flags)

    fig, ax = plt.subplots(figsize=(8.2, 6.2), layout="none")
    _axes_style(ax)
    ax.grid(False)
    if visited.any():
        ax.scatter(xy[visited, 0], xy[visited, 1],
                   s=[_MARK * m for m, v in zip(point_mass, visited) if v],
                   color=COLORS["green_strong"], alpha=0.55,
                   edgecolors=COLORS["ink_muted"], linewidths=0.6, zorder=2)
    handles = [Line2D([], [], marker="o", lw=0, ms=11, mfc=COLORS["green_strong"],
                      alpha=0.55, mec=COLORS["ink_muted"],
                      label="visited DAG, area ∝ mass")]
    if truth is not None:
        t_idx = int(np.argmax(truth_arr))
        if point_mass[t_idx] > 0:
            ax.plot([xy[t_idx, 0]], [xy[t_idx, 1]], marker="o", ms=14, mfc="none",
                    mec=COLORS["ink"], mew=1.7, zorder=4)
            ax.annotate("true DAG", (xy[t_idx, 0], xy[t_idx, 1]),
                        xytext=(9, 9), textcoords="offset points",
                        fontsize=9, fontweight="bold", color=COLORS["ink"])
            handles.append(Line2D([], [], marker="o", lw=0, ms=11, mfc="none",
                                  mec=COLORS["ink"], mew=1.7, label="true DAG"))
        else:
            ax.plot([xy[t_idx, 0]], [xy[t_idx, 1]], marker="X", ms=13,
                    color=COLORS["ink"], zorder=4)
            ax.annotate("true DAG (not visited)", (xy[t_idx, 0], xy[t_idx, 1]),
                        xytext=(9, -14), textcoords="offset points",
                        fontsize=9, fontweight="bold", color=COLORS["ink"])
            handles.append(Line2D([], [], marker="X", lw=0, ms=11, color=COLORS["ink"],
                                  label="true DAG, not visited"))
    for rank_i in range(min(3, top)):
        if truth_arr[rank_i] and (truth is not None):
            continue
        ax.annotate(f"id={ids[chosen[rank_i]]}\n{point_mass[rank_i]:.1%}",
                    (xy[rank_i, 0], xy[rank_i, 1]), xytext=(7, 7),
                    textcoords="offset points", fontsize=8, color=COLORS["ink"])

    covered = float(sum(point_mass[t] for t in range(top)))
    _note(fig, f"Top {top} visited DAGs: {covered:.1%} of retained mass; circle area ∝ mass. "
               "Hamming-MDS geometry is approximate. The true graph is included separately "
               "if outside this subset.", y=0.005)
    ax.set(xlabel="MDS dimension 1 (Hamming, approximate)",
           ylabel="MDS dimension 2 (Hamming, approximate)",
           title="Map of visited DAGs")
    ax.legend(handles=handles, frameon=False, fontsize=9, loc="best")
    if len(codes) == 1:
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
    else:
        ax.margins(0.18)
        ax.set_aspect("equal", adjustable="datalim")
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    return fig


def _mds_2d(codes):
    """Classical MDS on Hamming distances between pair-state rows."""
    codes = np.asarray(codes)
    dist = (codes[:, None, :] != codes[None, :, :]).sum(axis=2).astype(float)
    if len(dist) == 1:
        return np.zeros((1, 2))
    center = np.eye(len(dist)) - 1.0 / len(dist)
    gram = -0.5 * center @ (dist ** 2) @ center
    w, v = np.linalg.eigh(gram)
    order = np.argsort(w)[::-1][:2]
    xy = v[:, order] * np.sqrt(np.maximum(w[order], 0.0))
    if xy.shape[1] < 2:
        xy = np.pad(xy, ((0, 0), (0, 2 - xy.shape[1])))
    for c in range(xy.shape[1]):
        if xy[np.argmax(np.abs(xy[:, c])), c] < 0:
            xy[:, c] *= -1
    return xy


# ------------------------------------------------------- 10 the decoder ----
def plot_decoder(states, truth_states, labels):
    """Draw the leading DAGs and the true graph behind their identifiers.

    A 2x2 grid on small screens: up to three leading alternatives plus the
    true DAG (deduplicated if it already leads). Fewer visited DAGs than that
    is fine.
    """
    names = _labels(labels)
    n = len(names)
    pairs = _pairs_for(n)
    draws = _as_draws(states)
    _check_pairs(n, draws.shape[1])
    truth = _as_truth(truth_states, len(pairs))

    rows, ids, masses = _state_masses(draws)
    truth_pos = None
    if truth is not None:
        for t, row in enumerate(rows):
            if np.array_equal(row, truth):
                truth_pos = t
                break

    leading = [t for t in range(len(rows)) if t != truth_pos][:3]
    panels = [(rows[t], ids[t], float(masses[t]), False, t + 1) for t in leading]
    if truth is not None:
        if truth_pos is not None:
            panels.append((rows[truth_pos], ids[truth_pos],
                           float(masses[truth_pos]), True, truth_pos + 1))
        else:
            panels.append((np.asarray(truth, dtype=np.int64), _dag_id(truth),
                           0.0, True, None))

    fig, axes = plt.subplots(2, 2, figsize=(9.6, 8.6), layout="none")
    fig.set_facecolor(COLORS["bg"])
    axes = axes.ravel()
    for ax, (row, code, mass, is_truth, rank) in zip(axes, panels):
        if is_truth and rank is None:
            title = f"true DAG — id={code}\nnot visited in the retained draws"
        elif is_truth:
            title = f"true DAG — id={code}\nrank {rank}, mass {mass:.1%}"
        else:
            title = f"id={code}\nrank {rank}, mass {mass:.1%}"
        _draw_dag(ax, row, names, title)
    for ax in axes[len(panels):]:
        ax.axis("off")
    fig.suptitle("ID decoder: the graph behind each identifier", fontsize=12,
                 fontweight="bold", color=COLORS["ink"])
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


# ----------------------------------------------------- 11 pair states ------
def plot_pair_states(states, truth_states, labels):
    """Small multiples of every pair's three-state posterior, truth ringed."""
    names = _labels(labels)
    n = len(names)
    pairs = _pairs_for(n)
    draws = _as_draws(states)
    _check_pairs(n, draws.shape[1])
    truth = _as_truth(truth_states, len(pairs))

    prob = np.zeros((len(pairs), 3))
    for k in range(len(pairs)):
        for s in range(3):
            prob[k, s] = np.mean(draws[:, k] == s)

    cols = min(3, len(pairs))
    rows = ceil(len(pairs) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(3.3 * cols + 0.6, 2.7 * rows + 1.4),
                             sharex=True, sharey=True, layout="none")
    fig.set_facecolor(COLORS["bg"])
    axes = np.atleast_1d(axes).ravel()
    for k, (i, j) in enumerate(pairs):
        ax = axes[k]
        _axes_style(ax)
        ax.plot([0, 1, 2], prob[k], "-", lw=1, color=COLORS["ink_muted"], alpha=0.5)
        for s in range(3):
            ax.plot([s], [prob[k, s]], "o", ms=7, color=_STATE_COLORS[s])
        if truth is not None:
            ax.plot([truth[k]], [prob[k, truth[k]]], marker="o", ms=13, mfc="none",
                    mec=COLORS["ink"], mew=1.5)
        ax.set(xticks=[0, 1, 2],
               xticklabels=["absent", f"{names[i]}→{names[j]}",
                            f"{names[j]}→{names[i]}"],
               xlim=(-0.55, 2.55), ylim=(-0.05, 1.12))
        ax.set_title(f"{names[i]} – {names[j]}", fontsize=10,
                     fontweight="bold", color=COLORS["ink"])
        plt.setp(ax.get_xticklabels(), rotation=25, ha="right", fontsize=7.5)
    for k in range(len(pairs), len(axes)):
        axes[k].axis("off")

    handles = [Line2D([], [], marker="o", lw=0, ms=8, color=_STATE_COLORS[s],
                      label=lab) for s, lab in enumerate(
                          ("absent", "first → second", "second → first"))]
    if truth is not None:
        handles.append(Line2D([], [], marker="o", lw=0, ms=11, mfc="none",
                              mec=COLORS["ink"], mew=1.5, label="true state"))
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=9)
    fig.suptitle("Every pair's state posterior", fontsize=12,
                 fontweight="bold", color=COLORS["ink"])
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    return fig


# ---------------------------------------------------- 12 cumulative mass ---
def plot_cumulative(states):
    """Cumulative ranked DAG mass with threshold points at achieved values."""
    draws = _as_draws(states)
    _, masses = _masses([_dag_id(row) for row in draws])
    cum = np.cumsum(masses)
    rank = np.arange(1, len(masses) + 1)

    fig, ax = plt.subplots(figsize=(8, 4.9), layout="none")
    _axes_style(ax)
    ax.plot(rank, cum, "-", lw=1.6, color=COLORS["green_strong"])
    ax.plot(rank, cum, "o", ms=3.5, color=COLORS["green_strong"])
    placed = []
    for level in (0.50, 0.90, 0.95):
        k = int(np.searchsorted(cum, level, side="left")) + 1
        achieved = float(cum[k - 1])
        ax.axhline(level, color=COLORS["ink_muted"], ls=":", lw=1)
        ax.plot([k], [achieved], "o", ms=8, color=COLORS["ink"], zorder=3)
        stacked = sum(1 for px, py in placed if abs(px - k) < 1e-9 and abs(py - achieved) < 1e-9)
        placed.append((k, achieved))
        noun = "DAG" if k == 1 else "DAGs"
        verb = "reaches" if k == 1 else "reach"
        ax.annotate(f"{k} {noun} {verb} ≥{level:.0%}", xy=(k, achieved),
                    xytext=(8, -10 - 17 * stacked), textcoords="offset points",
                    fontsize=9, color=COLORS["ink"])
    ax.set(xscale="log", xlabel="number of top-ranked DAGs kept",
           ylabel="cumulative posterior probability",
           title="Cumulative posterior mass of ranked DAGs",
           xlim=(0.8, max(2.0, 1.4 * len(masses))), ylim=(0, 1.05))
    fig.tight_layout()
    return fig


# ------------------------------------------------------- 13 ID PMF ---------
def plot_id_pmf(states, truth_states, *, top=12):
    """Categorical PMF over the leading DAG ids plus one 'Other' aggregate."""
    draws = _as_draws(states)
    truth = _as_truth(truth_states, draws.shape[1])
    ids, masses = _masses([_dag_id(row) for row in draws])
    top = int(top)
    if top < 1:
        raise ValueError("top must be a positive integer")

    shown = min(top, len(ids))
    vals = list(masses[:shown])
    labels_x = [str(code) for code in ids[:shown]]
    other = float(masses[shown:].sum()) if len(ids) > shown else 0.0
    if other > 0:
        vals.append(other)
        labels_x.append("Other visited DAGs")
    x = np.arange(len(vals))

    fig, ax = plt.subplots(figsize=(max(8.0, 0.72 * len(vals) + 3.2), 5.2), layout="none")
    _axes_style(ax)
    ax.vlines(x, 0, vals, color=COLORS["green_strong"], lw=2)
    ax.plot(x, vals, "o", ms=7, color=COLORS["green_strong"], label="visited DAG id")
    ax.set(xticks=x, xticklabels=labels_x,
           xlabel="DAG id (categories, not distances)",
           ylabel="posterior probability of this exact graph",
           title="Categorical posterior over DAG ids",
           ylim=(0, 1.12 * max(vals)))
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)

    if truth is not None:
        t_id = _dag_id(truth)
        if t_id in ids[:shown]:
            pos = ids.index(t_id)
            ax.plot([pos], [vals[pos]], marker="o", ms=14, mfc="none",
                    mec=COLORS["ink"], mew=1.8, label="true DAG id")
        elif t_id in ids:
            r = ids.index(t_id) + 1
            m = float(masses[r - 1])
            ax.text(0.02, 0.96,
                    f"true id={t_id}: rank {r}, mass {m:.2%} (inside the aggregate)",
                    transform=ax.transAxes, fontsize=8.5, va="top",
                    color=COLORS["ink"])
        else:
            ax.text(0.02, 0.96,
                    f"true id={t_id} was not visited in the retained draws — "
                    "empirical mass 0 here is not a claim that its posterior is zero",
                    transform=ax.transAxes, fontsize=8.5, va="top",
                    color=COLORS["ink"])
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    return fig


# ------------------------------------------------- 14 ID number line -------
def plot_id_numberline(states, truth_states):
    """Spike PMF of visited DAG ids on the number line.

    Only the sampled support is plotted; base-3 identifier values are labels,
    not structural distances.
    """
    draws = _as_draws(states)
    truth = _as_truth(truth_states, draws.shape[1])
    ids, masses = _masses([_dag_id(row) for row in draws])
    xs = [float(code) for code in ids]

    fig, ax = plt.subplots(figsize=(8.6, 4.9), layout="none")
    _axes_style(ax)
    ax.vlines(xs, 0, masses, color=COLORS["green_strong"], lw=1.2)
    ax.plot(xs, masses, "o", ms=4, color=COLORS["green_strong"], label="visited DAG id")
    ax.set(xlabel="the DAG's base-3 identifier (labels, not structural distances)",
           ylabel="posterior probability of this exact graph",
           title="Visited DAG IDs, not a graph metric")
    ax.ticklabel_format(style="plain", axis="x")
    ax.margins(x=0.04)

    if truth is not None:
        t_id = _dag_id(truth)
        if t_id in ids:
            pos = ids.index(t_id)
            ax.plot([xs[pos]], [masses[pos]], marker="o", ms=12, mfc="none",
                    mec=COLORS["ink"], mew=1.7, label="true DAG id")
        else:
            ax.text(0.02, 0.96,
                    f"true id={t_id} is not among the visited draws (not plotted)",
                    transform=ax.transAxes, fontsize=8.5, va="top",
                    color=COLORS["ink"])
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    return fig
