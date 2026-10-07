"""Frozen experiment and reusable graph workflow for the three-article series.

The analysis settings and algorithms match the original seven-node experiment.
Each article constructs and samples its own posterior; no execution state is shared.
"""
from collections import Counter

import arviz_base as azb
import arviz_stats as azs
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pymc as pm
import pymc.dims as pmd
import pytensor
import pytensor.tensor as pt
import pytensor.xtensor as ptx
from matplotlib.patches import FancyArrowPatch
from pymc_marketing.mmm import MichaelisMentenSaturation

from cetagostini.style import COLORS
from .graph_math import BasisScore, has_path, mec_key, pairs_for, states_to_parents
from .graph_sampling import TemperedGraphStep, initial_states


labels = ("a", "b", "c", "d", "e", "f", "y")
n_nodes, n_obs = len(labels), 1_000
pairs = pairs_for(n_nodes)
d_index, f_index, y_index = 3, 5, 6

# Seeds and sizes, frozen before any fit.
DATA_SEED, REPEAT_SEEDS = 20260909, (20260910, 20260911)
HELDOUT_SEED = 20260912
HELDOUT_PREDICTION_SEED = 20261012
SAMPLING_SEED, PRIOR_SEED = 20260929, 20260959
PREDICTIVE_SEED, EFFECT_SEED = 20261011, 20261010
N_REPEAT, N_HELDOUT = 300, 300
draws, tune, chains = 12_000, 3_000, 4
betas = np.r_[np.geomspace(1.0, 0.002, 15), 0.0]
EFFECT_DRAWS, NOISE_ROWS = 1_200, 2_048
PREDICTIVE_REPLICATES = 400

# Fixed feature dictionary, declared in z units; the target stays raw.
centers, width = (-1.0, 1.0), 1.5
loc, scale = 0.0, np.array([1.0, 1.0, 1.0, 1.0, 1.5, 1.5, 2.0])
tau0, lam, alpha0, beta0 = 0.01, 0.1, 2.0, 1.0
centers_wide, lam_tight = (-2.0, -1.0, 0.0, 1.0, 2.0), 1.0
sparse_direction = 0.1
SATURATION_ALPHA, SATURATION_LAM = 6.0, 1.0

input_index = pt.vector("input_index")
input_exposure = ptx.as_xtensor(pt.softplus(input_index), dims=("observation",))
saturation = MichaelisMentenSaturation()
saturation_response = pytensor.function(
    [input_index],
    saturation.function(input_exposure, alpha=SATURATION_ALPHA, lam=SATURATION_LAM).values,
)

def h(x):
    """Michaelis–Menten response on softplus exposure, compiled once."""
    values = np.asarray(x, dtype=float)
    return saturation_response(values.reshape(-1)).reshape(values.shape)


def make_world(seed, size):
    """Draw all seven errors row-wise; size 300 is a prefix of size 1000."""
    errors = np.random.default_rng(seed).normal(size=(size, n_nodes))
    a, b, c, d = errors[:, 0], errors[:, 1], errors[:, 2], errors[:, 3]
    e = 0.9 * a + 0.8 * b + errors[:, 4]
    f = 0.9 * d + errors[:, 5]
    y = 0.55 * h(a) + 0.5 * h(b) + 0.6 * h(c) + 0.7 * h(e) + 0.8 * h(f) + errors[:, 6]
    return np.column_stack([a, b, c, d, e, f, y])


def make_twin(seed, size):
    """Same observational law as make_world; f is a root and d regresses on f."""
    errors = np.random.default_rng(seed).normal(size=(size, n_nodes))
    a, b, c = errors[:, 0], errors[:, 1], errors[:, 2]
    e = 0.9 * a + 0.8 * b + errors[:, 4]
    f = np.sqrt(1.81) * errors[:, 5]
    d = (0.9 / 1.81) * f + errors[:, 3] / np.sqrt(1.81)
    y = 0.55 * h(a) + 0.5 * h(b) + 0.6 * h(c) + 0.7 * h(e) + 0.8 * h(f) + errors[:, 6]
    return np.column_stack([a, b, c, d, e, f, y])


def make_graph_model(score, pair_probs, labels):
    """One dataset, one unknown DAG; mechanisms are integrated out."""
    pairs = pairs_for(len(labels))
    coords = {
        "node": labels, "parent": labels, "child": labels,
        "pair": [f"{labels[i]}–{labels[j]}" for i, j in pairs],
        "state": ["absent", "forward", "backward"],
        "mask": np.arange(2 ** len(labels)),
    }
    with pm.Model(coords=coords) as model:
        probabilities = pmd.as_xtensor(pair_probs, dims=("pair", "state"))
        initial = initial_states(
            pair_probs, pairs, len(labels), np.random.default_rng(0), empty=True,
        )
        edge = pmd.Categorical("edge", p=probabilities, core_dims="state", initval=initial)
        i, j = pairs.T
        state = edge.values
        adjacency = pt.zeros((len(labels), len(labels)), dtype="int64")
        adjacency = pt.set_subtensor(adjacency[i, j], pt.eq(state, 1))
        adjacency = pt.set_subtensor(adjacency[j, i], pt.eq(state, 2))
        adjacency = pmd.as_xtensor(adjacency, dims=("parent", "child"))
        bits = pmd.as_xtensor(1 << np.arange(len(labels)), dims=("parent",))
        parents = (adjacency * bits).sum("parent").rename({"child": "node"})
        local_score = pmd.as_xtensor(score.table, dims=("node", "mask"))
        evidence = local_score.isel(mask=parents).sum("node")
        reach = adjacency.astype("bool")
        for k in range(len(labels)):
            reach = reach | (reach.isel(child=k) & reach.isel(parent=k))
        diagonal = pmd.as_xtensor(np.eye(len(labels), dtype=bool), dims=("parent", "child"))
        cycle = (reach & diagonal).any()
        pmd.Potential("dag_support", ptx.where(cycle, -np.inf, 0.0))
        pmd.Potential("marginal_likelihood", evidence)
        pmd.Deterministic("log_evidence", evidence)
        pmd.Deterministic("edge_count", ptx.math.neq(edge, 0).sum("pair"))
    return model


def fit_graphs(score, probabilities, labels, *, seed, draws, tune, betas, chains=4):
    """Sample unresolved core directions and reconstruct full posterior DAGs."""
    local_pairs = pairs_for(len(labels))
    starts_rng = np.random.default_rng(seed)
    starts = [
        {"edge": initial_states(
            probabilities, local_pairs, len(labels), starts_rng, empty=(chain == 0),
        )}
        for chain in range(chains)
    ]
    with make_graph_model(score, probabilities, labels) as model:
        step = TemperedGraphStep(
            [model["edge"]], score.table, local_pairs, probabilities, betas,
        )
        return pm.sample(
            draws=draws, tune=tune, chains=chains, cores=1,
            step=step, initvals=starts, random_seed=seed,
            progressbar=False, compute_convergence_checks=False,
        )


def graph_summary(trace, pairs, n_nodes, truth_key, d_index, y_index):
    """Aggregate DAGs into equivalence classes, retaining per-draw membership."""
    states = trace.posterior.edge.values
    unique, inverse, counts = np.unique(states.reshape(-1, len(pairs)), axis=0,
                                        return_inverse=True, return_counts=True)
    graphs = [states_to_parents(state, pairs, n_nodes) for state in unique]
    keys = [mec_key(graph) for graph in graphs]
    class_counts, class_members, representative = Counter(), Counter(), {}
    for key, graph, count in zip(keys, graphs, counts):
        class_counts[key] += int(count)
        class_members[key] += 1
        representative[key] = graph
    ranked = [key for key, _ in class_counts.most_common()]
    class_lookup = {key: index for index, key in enumerate(ranked)}
    class_id = np.array([class_lookup[key] for key in keys])[inverse].reshape(states.shape[:2])
    paths = np.array([
        has_path(graph, d_index, y_index) for graph in graphs
    ])[inverse].reshape(states.shape[:2])
    true_class = (class_id == class_lookup[truth_key]) if truth_key in class_lookup \
        else np.zeros_like(paths)
    return {"states": states, "graphs": graphs, "keys": keys, "ranked": ranked,
            "class_counts": class_counts, "class_members": class_members,
            "representative": representative, "class_id": class_id,
            "true_class": true_class, "path": paths}


def graph_diagnostic_data(trace, summary, probabilities, labels, pairs):
    """Prepare chain-by-draw graph events; leave all diagnostics to ArviZ."""
    states = summary["states"]
    names, events, fixed = [], [], []
    for k, (i, j) in enumerate(pairs):
        for state, name in [(0, f"{labels[i]}–{labels[j]} absent"),
                            (1, f"{labels[i]}→{labels[j]}"), (2, f"{labels[j]}→{labels[i]}")]:
            names.append(name)
            events.append(states[..., k] == state)
            fixed.append(probabilities[k, state] == 0 or np.count_nonzero(probabilities[k]) == 1)
    for index in range(min(4, len(summary["ranked"]))):
        names.append(f"Class {index + 1}")
        events.append(summary["class_id"] == index)
        fixed.append(False)
    names += ["Generating class", "d has a path to y"]
    events += [summary["true_class"], summary["path"]]
    fixed = np.asarray(fixed + [False, False])
    values = np.stack(events, axis=-1).astype(float)
    varying = np.ptp(values, axis=(0, 1)) > 0
    quantities = {"event": values[..., varying]}
    constant_summaries = []
    for name in ("edge_count", "log_evidence"):
        value = trace.posterior[name].values
        if np.ptp(value) > 0:
            quantities[name] = value
        else:
            constant_summaries.append(name)
    data = azb.from_dict(
        {"posterior": quantities},
        dims={"event": ["graph_event"]},
        coords={"graph_event": np.asarray(names)[varying]},
    )
    data["posterior"].attrs.update(
        fixed_indicators=int(fixed.sum()),
        other_constant_indicators=int((~varying & ~fixed).sum()),
        constant_summaries=", ".join(constant_summaries),
    )
    return data


def rank_and_mass(codes, target):
    """Empirical rank (1 = highest) and mass of a target code, or 'not visited'."""
    unique, counts = np.unique(codes, return_counts=True)
    if target not in set(unique.tolist()):
        return None, 0.0
    order = np.lexsort((unique, -counts))
    rank = int(np.where(unique[order] == target)[0][0]) + 1
    return rank, float(counts[list(unique).index(target)] / counts.sum())


def draw_graph(ax, adjacency, title):
    positions = {0: (-1.7, 1.9), 1: (-0.5, 2.4), 2: (0.6, 1.9),
                 3: (1.9, 2.4), 4: (-1.25, 0.7), 5: (1.4, 0.8), 6: (0.0, -0.5)}
    graph = nx.empty_graph(n_nodes)
    nx.draw_networkx_nodes(graph, positions, node_size=650,
                          node_color=COLORS["surface_alt"], edgecolors=COLORS["primary"], ax=ax)
    nx.draw_networkx_labels(graph, positions, labels=dict(enumerate(labels)),
                           font_color=COLORS["ink"], font_weight="bold", ax=ax)
    for i, j in pairs:
        if not (adjacency[i, j] or adjacency[j, i]):
            continue
        undirected = adjacency[i, j] and adjacency[j, i]
        start, end = (i, j) if adjacency[i, j] else (j, i)
        ax.add_patch(FancyArrowPatch(
            positions[start], positions[end], arrowstyle="-" if undirected else "-|>",
            mutation_scale=14, linewidth=1.6, shrinkA=16, shrinkB=16,
            linestyle="--" if undirected else "-",
            color=COLORS["brown"] if undirected else COLORS["green_strong"],
        ))
    ax.set(title=title, xlim=(-2.1, 2.3), ylim=(-0.9, 2.8), aspect="equal")
    ax.axis("off")


def adjacency_of(parents):
    return np.array([[bool(int(parents[j]) & (1 << i)) for j in range(n_nodes)]
                     for i in range(n_nodes)])


def build_score(current_data, *, lam_value=lam, centers_value=centers):
    return BasisScore(current_data, nonlinear_nodes=(y_index,), centers=centers_value,
                      width=width, loc=loc, scale=scale, tau0=tau0, lam=lam_value,
                      alpha0=alpha0, beta0=beta0)


def fit_metrics(trace, probabilities, truth_states, pairs, labels, truth_key):
    """Measured structural summaries and graph-event diagnostics for one run."""
    current = graph_summary(trace, pairs, n_nodes, truth_key, d_index, y_index)
    diag = azs.summary(
        graph_diagnostic_data(trace, current, probabilities, labels, pairs),
        ci_prob=0.95, round_to="none",
    )
    flat_states = current["states"].reshape(-1, len(pairs))
    powers_local = 3 ** np.arange(len(pairs))
    true_id = int((truth_states * powers_local).sum())
    binary_powers = 1 << np.arange(len(pairs))
    true_skeleton = int(((truth_states != 0) * binary_powers).sum())
    df_row = next(k for k, pair in enumerate(pairs) if tuple(pair) == (d_index, f_index))
    ids = (flat_states * powers_local).sum(axis=1)
    skeleton_codes = ((flat_states != 0) * binary_powers).sum(axis=1)
    dag_rank, dag_mass = rank_and_mass(ids, true_id)
    _, skel_mass = rank_and_mass(skeleton_codes, true_skeleton)
    pair3 = np.stack([(current["states"] == s).mean(axis=(0, 1)) for s in (0, 1, 2)], axis=1)
    arrow_rows = pair3[:, 1:].max(axis=1) >= 0.8
    df_text = f"{pair3[df_row, 1]:.2f}/{pair3[df_row, 2]:.2f}"
    if not arrow_rows[df_row]:
        df_text += " (below 0.8)"
    return {
        "True-DAG rank": dag_rank if dag_rank else "not visited",
        "True-DAG mass": dag_mass,
        "Skeleton mass": skel_mass,
        "Class mass": float(current["true_class"].mean()),
        "d→f / f→d": df_text,
        "Other arrows at 0.8": int(arrow_rows.sum() - arrow_rows[df_row]),
        "Max R-hat": float(diag.r_hat.max()),
        "Min bulk ESS": float(diag.ess_bulk.min()),
    }
