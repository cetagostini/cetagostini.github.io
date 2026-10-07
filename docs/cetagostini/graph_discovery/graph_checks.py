"""Small, exhaustive correctness oracle; never used to fit the seven-node example."""

from collections import defaultdict
from itertools import product
from types import SimpleNamespace

import numpy as np
import pymc as pm
from scipy.special import gammaln, softmax
from scipy.stats import invgamma, kstest, multivariate_t, t as student_t

from .graph_math import (
    BasisScore, BGeScore, cpdag, has_path, mec_key, pair_probabilities, pairs_for,
    states_to_parents, topological_order,
)
from .graph_sampling import (
    TemperedGraphStep, initial_states, terminal_parent_distributions,
    update_journeys,
)


def check_small_graphs(make_graph_model):
    """Check the actual model, CPDAG conversion, and sampler on all four-node DAGs."""
    def total_variation(left, right):
        return sum(abs(left.get(key, 0.0) - right.get(key, 0.0))
                   for key in left.keys() | right.keys()) / 2

    # Adjacent swaps alone are not full round trips; track physical replica IDs.
    journeys = np.array([1, 0, 0, 0])
    for labels in ([1, 0, 2, 3], [1, 2, 0, 3], [1, 2, 3, 0], [1, 2, 0, 3]):
        assert update_journeys(np.asarray(labels), journeys) == 0
    assert update_journeys(np.array([0, 1, 2, 3]), journeys) == 1
    assert update_journeys(np.array([0, 1, 2, 3]), journeys) == 0
    assert update_journeys(np.array([0]), np.array([1])) == 0

    n = 4
    pairs = pairs_for(n)
    pair_index = {tuple(map(int, pair)): index for index, pair in enumerate(pairs)}
    rng = np.random.default_rng(2401)
    data = rng.normal(size=(100, n))
    data[:, 2] += 0.8 * data[:, 0] + 0.7 * data[:, 1]
    data[:, 3] += 0.6 * data[:, 2]
    prior_sd = np.array([1.0, 1.3, 1.8, 2.0])
    score = BGeScore(data, prior_sd=prior_sd)
    probabilities = np.tile([0.5, 0.35, 0.15], (len(pairs), 1))
    model = make_graph_model(score, probabilities, tuple("abcd"))
    logp = model.compile_logp()
    states, parents, keys, evidence, prior = [], [], [], [], []
    groups = defaultdict(list)
    target_error = 0.0
    cycle_count = 0
    for state_tuple in product(range(3), repeat=len(pairs)):
        state = np.asarray(state_tuple, dtype="int64")
        graph = states_to_parents(state, pairs, n)
        actual = logp({"edge": state})
        try:
            topological_order(graph)
        except ValueError:
            assert np.isneginf(actual), "Model gave finite mass to a directed cycle"
            cycle_count += 1
            continue
        key = mec_key(graph)
        local_evidence = score.table[np.arange(n), graph].sum()
        local_prior = np.log(probabilities[np.arange(len(pairs)), state]).sum()
        target_error = max(target_error, abs(actual - local_evidence - local_prior))
        groups[key].append(len(states))
        states.append(state)
        parents.append(graph)
        keys.append(key)
        evidence.append(local_evidence)
        prior.append(local_prior)
    states, evidence, prior = np.asarray(states), np.asarray(evidence), np.asarray(prior)
    assert len(states) == 543 and len(groups) == 185
    assert target_error < 1e-9

    equivalence_error = 0.0
    for members in groups.values():
        directed = np.array([
            [[bool(int(parents[k][j]) & (1 << i)) for j in range(n)] for i in range(n)]
            for k in members
        ])
        skeleton = directed[0] | directed[0].T
        compelled = directed.all(axis=0)
        expected = skeleton & ~(compelled.T & ~compelled)
        for k in members:
            assert np.array_equal(cpdag(parents[k]), expected), "Incorrect compelled edge"
        equivalence_error = max(equivalence_error, float(np.ptp(evidence[members])))
    assert equivalence_error < 1e-9

    with model:
        step = TemperedGraphStep([model["edge"]], score.table, pairs, probabilities,
                                 np.r_[np.geomspace(1.0, 0.005, 10), 0.0])
        trace = pm.sample(draws=10_000, tune=2_000, chains=4, cores=1, step=step,
                          random_seed=2402, progressbar=False, compute_convergence_checks=False)
    exact = defaultdict(float)
    sampled = defaultdict(float)
    for key, weight in zip(keys, softmax(evidence + prior)):
        exact[key] += weight
    retained = trace.posterior.edge.values.reshape(-1, len(pairs))
    for state in retained:
        graph = states_to_parents(state, pairs, n)
        topological_order(graph)
        sampled[mec_key(graph)] += 1 / len(retained)
    tv = total_variation(exact, sampled)
    assert tv < 0.04, f"Sampler missed the exact four-node class posterior: TV={tv}"

    # Matrix semantics: values are [absent, left→right, right→left] in pairs_for order.
    direction_prior = np.zeros((n, n))
    direction_prior[0, 1], direction_prior[1, 0] = 0.6, 0.2
    direction_prior[0, 2], direction_prior[2, 0] = 0.2, 0.4
    direction_prior[1, 2], direction_prior[2, 1] = 0.1, 0.7
    direction_prior[0, 3], direction_prior[3, 0] = 0.5, 0.1
    direction_prior[1, 3], direction_prior[3, 1] = 0.1, 0.1
    direction_prior[2, 3], direction_prior[3, 2] = 0.1, 0.1
    matrix_probs = pair_probabilities(direction_prior)
    assert matrix_probs.shape == (len(pairs), 3)
    assert np.allclose(matrix_probs.sum(axis=1), 1.0)
    assert np.allclose(matrix_probs[pair_index[0, 1]], [0.2, 0.6, 0.2])
    assert np.allclose(matrix_probs[pair_index[0, 2]], [0.4, 0.2, 0.4])
    assert np.allclose(matrix_probs[pair_index[1, 2]], [0.2, 0.1, 0.7])
    assert np.allclose(matrix_probs[pair_index[0, 3]], [0.4, 0.5, 0.1])

    # A hard mask conditions on surviving states: it neither adds epsilon mass
    # nor forces the reverse direction. Both directions may be masked if absence remains.
    masked_source = np.array([[0.0, 0.3], [0.2, 0.0]])
    one_direction = pair_probabilities(
        masked_source,
        allowed=np.array([[False, True], [False, False]]),
    )
    both_directions = pair_probabilities(
        masked_source,
        allowed=np.zeros((2, 2), dtype=bool),
    )
    assert np.allclose(one_direction[0], [0.625, 0.375, 0.0])
    assert np.allclose(both_directions[0], [1.0, 0.0, 0.0])
    exact_zero_absence = pair_probabilities(np.array([[0.0, 0.25], [0.75, 0.0]]))
    assert exact_zero_absence[0, 0] == 0.0
    assert np.allclose(exact_zero_absence[0, 1:], [0.25, 0.75])
    try:
        pair_probabilities(
            np.array([[0.0, 0.5], [0.5, 0.0]]),
            allowed=np.zeros((2, 2), dtype=bool),
        )
        assert False, "A mask must reject a pair with no surviving state"
    except ValueError:
        pass

    # Required support cannot be discarded just because empty=True was requested.
    pairs3 = pairs_for(3)
    single_required = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    single_state = initial_states(single_required, pairs3, 3, np.random.default_rng(2404), empty=True)
    assert np.array_equal(single_state, np.array([1, 0, 0], dtype=np.int64))
    topological_order(states_to_parents(single_state, pairs3, 3))
    absent_only = np.array([[1.0, 0.0, 0.0], [0.5, 0.5, 0.0], [0.5, 0.5, 0.0]])
    absent_state = initial_states(absent_only, pairs3, 3, np.random.default_rng(2405), empty=True)
    assert np.array_equal(absent_state, np.zeros(len(pairs3), dtype=np.int64))
    assert np.all(absent_only[np.arange(3), absent_state] > 0)
    topological_order(states_to_parents(absent_state, pairs3, 3))
    try:
        initial_states(np.zeros((3, 3)), pairs3, 3, np.random.default_rng(2406))
        assert False, "Initial-state construction must reject an empty supported pair"
    except ValueError:
        pass

    # A truly required 0→1→2→0 cycle has no legal initialization.
    forced_cycle_prior = np.zeros((3, 3))
    forced_cycle_prior[0, 1] = 1.0
    forced_cycle_prior[1, 2] = 1.0
    forced_cycle_prior[2, 0] = 1.0
    forced_cycle_probs = pair_probabilities(forced_cycle_prior)
    assert np.array_equal(forced_cycle_probs[0], [0.0, 1.0, 0.0])
    assert np.array_equal(forced_cycle_probs[1], [0.0, 0.0, 1.0])
    assert np.array_equal(forced_cycle_probs[2], [0.0, 1.0, 0.0])
    try:
        initial_states(forced_cycle_probs, pairs3, 3, np.random.default_rng(2407), empty=True)
        assert False, "Required directed cycle must be rejected"
    except ValueError:
        pass

    # Masked support keeps a static symmetric proposal: choose one active pair,
    # then any supported state (including current), Q=1/(n_active*n_supported).
    masked_probs = np.array([[0.0, 0.7, 0.3], [0.4, 0.2, 0.4], [0.5, 0.5, 0.0]])
    score3 = BGeScore(data[:, :3], prior_sd=prior_sd[:3])
    model_masked = make_graph_model(score3, masked_probs, tuple("abc"))
    with model_masked:
        step_masked = TemperedGraphStep(
            [model_masked["edge"]], score3.table, pairs3, masked_probs,
            np.r_[np.geomspace(1.0, 0.01, 4), 0.0],
        )
    assert np.array_equal(step_masked.terminal_nodes, np.array([], dtype=np.int64))
    assert np.array_equal(step_masked.active_pairs, np.arange(len(pairs3), dtype=np.int64))
    masked_states, masked_evidence, masked_prior = [], [], []
    for state_tuple in product(range(3), repeat=len(pairs3)):
        state = np.asarray(state_tuple, dtype=np.int64)
        if np.any(masked_probs[np.arange(len(pairs3)), state] == 0):
            continue
        graph = states_to_parents(state, pairs3, 3)
        try:
            topological_order(graph)
        except ValueError:
            continue
        masked_states.append(state)
        masked_evidence.append(score3.table[np.arange(3), graph].sum())
        masked_prior.append(np.log(masked_probs[np.arange(len(pairs3)), state]).sum())
    masked_evidence = np.asarray(masked_evidence)
    masked_prior = np.asarray(masked_prior)
    # Exercise both kinds of two-state core support: required adjacency and
    # forbidden orientation. The compiled step must match the exact joint.
    with model_masked:
        masked_trace = pm.sample(
            draws=5_000, tune=1_000, chains=2, cores=1, step=step_masked,
            random_seed=2403, progressbar=False, compute_convergence_checks=False,
        )
    masked_exact = dict(zip(
        map(tuple, masked_states), softmax(masked_evidence + masked_prior),
    ))
    masked_draws = masked_trace.posterior.edge.values.reshape(-1, len(pairs3))
    masked_sampled = defaultdict(float)
    for state in masked_draws:
        key = tuple(state)
        assert key in masked_exact, "Core proposal violated support or acyclicity"
        masked_sampled[key] += 1 / len(masked_draws)
    masked_tv = total_variation(masked_exact, masked_sampled)
    assert masked_tv < 0.04, f"Two-state core proposals missed their target: TV={masked_tv}"

    # Offset tiny priors with a synthetic score so incorrect epsilon clipping
    # changes an observable posterior, rather than an unmeasurably rare event.
    tiny = 1e-315
    tiny_probs = np.array([[1.0, tiny, 2 * tiny]])
    tiny_table = np.zeros((2, 4))
    tiny_table[0, 2] = tiny_table[1, 1] = -np.log(tiny)
    tiny_model = make_graph_model(SimpleNamespace(table=tiny_table), tiny_probs, tuple("ab"))
    with tiny_model:
        tiny_step = TemperedGraphStep(
            [tiny_model["edge"]], tiny_table, pairs_for(2), tiny_probs, np.array([1.0]),
        )
        tiny_trace = pm.sample(
            draws=10_000, tune=500, chains=1, cores=1, step=tiny_step,
            random_seed=2510, progressbar=False, compute_convergence_checks=False,
        )
    tiny_frequency = np.bincount(tiny_trace.posterior.edge.values.ravel(), minlength=3) / 10_000
    np.testing.assert_allclose(tiny_frequency, [0.25, 0.25, 0.5], atol=0.025, rtol=0)

    # Always switching between two equally weighted states has period two.
    # Two sweeps per retained draw must not freeze the sampled orientation.
    periodic_probs = np.array([[0.0, 0.5, 0.5]])
    periodic_table = np.zeros((2, 4))
    periodic_model = make_graph_model(
        SimpleNamespace(table=periodic_table), periodic_probs, tuple("ab"),
    )
    with periodic_model:
        periodic_step = TemperedGraphStep(
            [periodic_model["edge"]], periodic_table, pairs_for(2), periodic_probs,
            np.array([1.0]), sweeps=2,
        )
        periodic_trace = pm.sample(
            draws=2_000, tune=0, chains=1, cores=1, step=periodic_step,
            random_seed=2512, progressbar=False, compute_convergence_checks=False,
        )
    periodic_frequency = np.bincount(periodic_trace.posterior.edge.values.ravel(), minlength=3) / 2_000
    assert periodic_frequency[0] == 0
    np.testing.assert_allclose(periodic_frequency, [0, 0.5, 0.5], atol=0.04, rtol=0)

    # Asymmetric four-node terminal example. Node 3 has no outgoing support;
    # core pairs retain three states, terminal pairs retain absent/incoming states.
    direction_r = np.zeros((n, n))
    direction_r[0, 1], direction_r[1, 0] = 0.5, 0.3
    direction_r[0, 2], direction_r[2, 0] = 0.2, 0.4
    direction_r[1, 2], direction_r[2, 1] = 0.1, 0.7
    direction_r[0, 3], direction_r[3, 0] = 0.3, 0.1
    direction_r[1, 3], direction_r[3, 1] = 0.2, 0.1
    direction_r[2, 3], direction_r[3, 2] = 0.2, 0.1
    allowed_r = np.ones((n, n), dtype=bool)
    allowed_r[3, :] = False
    allowed_r[np.diag_indices(n)] = False
    pair_probs_r = pair_probabilities(direction_r, allowed=allowed_r)
    for parent in range(3):
        pair = pair_index[parent, 3]
        assert pair_probs_r[pair, 1] > 0.0
        assert pair_probs_r[pair, 2] == 0.0

    initial_r = initial_states(pair_probs_r, pairs, n, np.random.default_rng(2408), empty=True)
    assert np.all(pair_probs_r[np.arange(len(pairs)), initial_r] > 0)
    initial_graph_r = states_to_parents(initial_r, pairs, n)
    topological_order(initial_graph_r)
    assert initial_graph_r[3] == 0

    valid_full, valid_parents, valid_keys = [], [], []
    valid_evidence, valid_prior = [], []
    for state_tuple in product(range(3), repeat=len(pairs)):
        state = np.asarray(state_tuple, dtype=np.int64)
        if np.any(pair_probs_r[np.arange(len(pairs)), state] == 0):
            continue
        graph = states_to_parents(state, pairs, n)
        try:
            topological_order(graph)
        except ValueError:
            continue
        assert not np.any(graph & (1 << 3)), "Terminal node acquired an outgoing arrow"
        valid_full.append(state)
        valid_parents.append(graph)
        valid_keys.append(mec_key(graph))
        valid_evidence.append(score.table[np.arange(n), graph].sum())
        valid_prior.append(np.log(pair_probs_r[np.arange(len(pairs)), state]).sum())
    assert len(valid_full) == 200, f"Expected 200 valid DAGs, got {len(valid_full)}"
    full_weights = softmax(np.asarray(valid_evidence) + np.asarray(valid_prior))

    model_r = make_graph_model(score, pair_probs_r, tuple("abcd"))
    restricted_logp = model_r.compile_logp()
    with model_r:
        step_r = TemperedGraphStep(
            [model_r["edge"]], score.table, pairs, pair_probs_r,
            np.r_[np.geomspace(1.0, 0.005, 10), 0.0],
        )
    assert np.array_equal(step_r.terminal_nodes, np.array([3], dtype=np.int64))
    expected_active = np.array([
        index for index, (left, right) in enumerate(pairs)
        if left != 3 and right != 3 and (pair_probs_r[index] > 0).sum() >= 2
    ], dtype=np.int64)
    assert np.array_equal(step_r.active_pairs, expected_active)
    assert np.isclose(
        restricted_logp({"edge": valid_full[0]}),
        valid_evidence[0] + valid_prior[0],
    )
    unsupported = valid_full[0].copy()
    unsupported[pair_index[0, 3]] = 2  # Forbidden 3→0 output edge.
    assert np.isneginf(restricted_logp({"edge": unsupported}))

    # Terminal masks use original global bits; their exact distribution is the
    # exhaustive posterior marginal over parent sets, not a fresh BGe draw.
    terminals = terminal_parent_distributions(score.table, pairs, pair_probs_r)
    assert set(terminals) == {3}
    terminal_masks, terminal_probs = terminals[3]
    assert isinstance(terminal_masks, np.ndarray)
    assert terminal_masks.dtype == np.int64
    assert terminal_masks.shape == (8,)
    assert np.array_equal(np.sort(terminal_masks), np.arange(8, dtype=np.int64))
    assert np.all(terminal_masks & (1 << 3) == 0)
    assert np.isclose(terminal_probs.sum(), 1.0)
    terminal_marginal = np.zeros(1 << n)
    for graph, weight in zip(valid_parents, full_weights):
        terminal_marginal[int(graph[3])] += weight
    np.testing.assert_allclose(
        terminal_probs, terminal_marginal[terminal_masks], rtol=0, atol=1e-12,
    )

    # Reconstruct every full DAG probability from its core graph and its
    # independent terminal parent set. This catches lost member multiplicity.
    core_pairs = np.array([
        index for index, (left, right) in enumerate(pairs) if left != 3 and right != 3
    ], dtype=np.int64)
    core_log_weights = {}
    for state, graph in zip(valid_full, valid_parents):
        core_state = tuple(int(state[index]) for index in core_pairs)
        core_log_weight = sum(score.table[node, int(graph[node])] for node in range(3))
        core_log_weight += sum(np.log(pair_probs_r[index, int(state[index])])
                               for index in core_pairs)
        previous = core_log_weights.setdefault(core_state, core_log_weight)
        assert np.isclose(previous, core_log_weight, rtol=0, atol=1e-12)
    assert len(core_log_weights) == 25
    core_states = tuple(core_log_weights)
    core_weights = dict(zip(core_states, softmax(np.array([
        core_log_weights[state] for state in core_states
    ]))))
    terminal_weight_map = dict(zip(map(int, terminal_masks), terminal_probs))
    assert len(valid_full) == len(core_weights) * len(terminal_weight_map)
    for state, graph, weight in zip(valid_full, valid_parents, full_weights):
        core_state = tuple(int(state[index]) for index in core_pairs)
        reconstructed = core_weights[core_state] * terminal_weight_map[int(graph[3])]
        assert np.isclose(reconstructed, weight, rtol=0, atol=1e-12)

    # Permute the original data, score scales, direction matrix, and support so
    # the same sink is node 0. Terminal status must follow support, not position.
    permutation = np.array([3, 0, 1, 2])
    score_nl = BGeScore(data[:, permutation], prior_sd=prior_sd[permutation])
    direction_nl = direction_r[np.ix_(permutation, permutation)]
    allowed_nl = allowed_r[np.ix_(permutation, permutation)]
    pair_probs_nl = pair_probabilities(direction_nl, allowed=allowed_nl)
    terminals_nl = terminal_parent_distributions(score_nl.table, pairs, pair_probs_nl)
    assert set(terminals_nl) == {0}
    masks_nl, probs_nl = terminals_nl[0]
    assert np.array_equal(np.sort(masks_nl), 2 * np.arange(8, dtype=np.int64))
    assert np.isclose(probs_nl.sum(), 1.0)
    model_nl = make_graph_model(score_nl, pair_probs_nl, tuple("dabc"))
    with model_nl:
        step_nl = TemperedGraphStep(
            [model_nl["edge"]], score_nl.table, pairs, pair_probs_nl,
            np.r_[np.geomspace(1.0, 0.01, 4), 0.0],
        )
    assert np.array_equal(step_nl.terminal_nodes, np.array([0], dtype=np.int64))
    expected_active_nl = np.array([
        index for index, (left, right) in enumerate(pairs)
        if left != 0 and right != 0 and (pair_probs_nl[index] > 0).sum() >= 2
    ], dtype=np.int64)
    assert np.array_equal(step_nl.active_pairs, expected_active_nl)

    # Two terminals have no edge between them, while their common core remains mutable.
    direction_mt = np.zeros((n, n))
    direction_mt[1, 0], direction_mt[3, 0] = 0.25, 0.35
    direction_mt[1, 2], direction_mt[3, 2] = 0.20, 0.30
    direction_mt[1, 3], direction_mt[3, 1] = 0.20, 0.40
    allowed_mt = np.ones((n, n), dtype=bool)
    allowed_mt[0, :] = False
    allowed_mt[2, :] = False
    allowed_mt[np.diag_indices(n)] = False
    pair_probs_mt = pair_probabilities(direction_mt, allowed=allowed_mt)
    terminals_mt = terminal_parent_distributions(score.table, pairs, pair_probs_mt)
    assert set(terminals_mt) == {0, 2}
    for terminal, other in ((0, 2), (2, 0)):
        masks_mt, probs_mt = terminals_mt[terminal]
        assert np.isclose(probs_mt.sum(), 1.0)
        assert np.all(masks_mt & (1 << other) == 0)
    model_mt = make_graph_model(score, pair_probs_mt, tuple("abcd"))
    with model_mt:
        step_mt = TemperedGraphStep(
            [model_mt["edge"]], score.table, pairs, pair_probs_mt,
            np.r_[np.geomspace(1.0, 0.01, 4), 0.0],
        )
    assert np.array_equal(step_mt.terminal_nodes, np.array([0, 2], dtype=np.int64))
    assert np.array_equal(step_mt.active_pairs, np.array([pair_index[1, 3]], dtype=np.int64))

    # Exercise reconstruction, not only terminal metadata, when sinks move or
    # share potential parents. Compare full draws with independently scored DAGs.
    variant_tvs = {}
    for index, (name, variant_model, variant_step, variant_score, variant_probs) in enumerate([
        ("non_last", model_nl, step_nl, score_nl, pair_probs_nl),
        ("multiple", model_mt, step_mt, score, pair_probs_mt),
    ]):
        variant_states, variant_log_weights = [], []
        for state, graph in zip(states, parents):
            if np.any(variant_probs[np.arange(len(pairs)), state] == 0):
                continue
            variant_states.append(tuple(map(int, state)))
            variant_log_weights.append(
                variant_score.table[np.arange(n), graph].sum()
                + np.log(variant_probs[np.arange(len(pairs)), state]).sum()
            )
        variant_exact = dict(zip(variant_states, softmax(variant_log_weights)))
        with variant_model:
            variant_trace = pm.sample(
                draws=3_000, tune=500, chains=2, cores=1, step=variant_step,
                random_seed=2414 + index, progressbar=False, compute_convergence_checks=False,
            )
        variant_draws = variant_trace.posterior.edge.values.reshape(-1, len(pairs))
        variant_sampled = defaultdict(float)
        for state in variant_draws:
            key = tuple(map(int, state))
            assert key in variant_exact, "Reconstructed a forbidden or cyclic graph"
            variant_sampled[key] += 1 / len(variant_draws)
        variant_tvs[name] = total_variation(variant_exact, variant_sampled)
        assert variant_tvs[name] < 0.10, f"{name} terminal reconstruction: TV={variant_tvs[name]}"

    # All-six-pair support can be fixed. Exact zeros must be accepted, required
    # arrows retained, and reconstruction must still emit the unique full graph.
    direction_fixed = np.zeros((n, n))
    direction_fixed[0, 1] = 1.0
    direction_fixed[0, 2] = 1.0
    direction_fixed[1, 2] = 1.0
    fixed_probs = pair_probabilities(direction_fixed)
    fixed_state = np.array([1, 1, 0, 1, 0, 0], dtype=np.int64)
    assert np.array_equal(initial_states(
        fixed_probs, pairs, n, np.random.default_rng(2409), empty=True,
    ), fixed_state)
    model_fixed = make_graph_model(score, fixed_probs, tuple("abcd"))
    with model_fixed:
        step_fixed = TemperedGraphStep(
            [model_fixed["edge"]], score.table, pairs, fixed_probs,
            np.r_[np.geomspace(1.0, 0.01, 4), 0.0],
        )
    assert np.array_equal(step_fixed.active_pairs, np.array([], dtype=np.int64))
    assert np.array_equal(step_fixed.terminal_nodes, np.array([2, 3], dtype=np.int64))
    fixed_term_masks, fixed_term_probs = terminal_parent_distributions(
        score.table, pairs, fixed_probs,
    )[2]
    assert np.isclose(fixed_term_probs[np.flatnonzero(fixed_term_masks == 3)][0], 1.0)
    with model_fixed:
        trace_fixed = pm.sample(
            draws=300, tune=100, chains=1, cores=1, step=step_fixed,
            random_seed=2410, progressbar=False, compute_convergence_checks=False,
        )
    retained_fixed = trace_fixed.posterior.edge.values.reshape(-1, len(pairs))
    assert np.all(retained_fixed == fixed_state)
    for name in ("cold_accept", "cold_invalid", "roundtrips",
                 *(f"swap_{i}" for i in range(len(step_fixed.betas) - 1))):
        assert np.all(trace_fixed.sample_stats[name].values == 0), "Fixed core reported exploration"

    # setup_chain must discard the previous replica state and honor the supplied
    # chain RNG. A reused step reproduces a fresh step's seeded full trajectory.
    reset_model = make_graph_model(score, pair_probs_r, tuple("abcd"))
    with reset_model:
        fresh_step = TemperedGraphStep(
            [reset_model["edge"]], score.table, pairs, pair_probs_r,
            np.r_[np.geomspace(1.0, 0.01, 4), 0.0], rng=np.random.default_rng(1),
        )
        reused_step = TemperedGraphStep(
            [reset_model["edge"]], score.table, pairs, pair_probs_r,
            np.r_[np.geomspace(1.0, 0.01, 4), 0.0], rng=np.random.default_rng(2),
        )

    reset_initial = reset_model.initial_point()

    def seeded_trajectory(step_method, seed):
        step_method.setup_chain(np.random.default_rng(seed), tune=0, draws=4)
        point = {name: np.array(value, copy=True) for name, value in reset_initial.items()}
        result = []
        for _ in range(4):
            point, _ = step_method.step(point)
            result.append(point["edge"].copy())
        return np.asarray(result)

    fresh_trajectory = seeded_trajectory(fresh_step, 2411)
    seeded_trajectory(reused_step, 2412)
    reset_trajectory = seeded_trajectory(reused_step, 2411)
    assert np.array_equal(fresh_trajectory, reset_trajectory)

    # Full output, not terminal marginals alone: compare reconstructed full DAG
    # draws and equivalence classes against all 200 exhaustive posterior weights.
    with model_r:
        trace_r = pm.sample(
            draws=5_000, tune=1_000, chains=4, cores=1, step=step_r,
            random_seed=2413, progressbar=False, compute_convergence_checks=False,
        )
    retained_r = trace_r.posterior.edge.values.reshape(-1, len(pairs))
    sampled_joint = defaultdict(float)
    sampled_classes = defaultdict(float)
    for state in retained_r:
        assert np.all(pair_probs_r[np.arange(len(pairs)), state] > 0)
        graph = states_to_parents(state, pairs, n)
        topological_order(graph)
        assert not np.any(graph & (1 << 3)), "Terminal node acquired an outgoing arrow"
        sampled_joint[tuple(map(int, state))] += 1 / len(retained_r)
        sampled_classes[mec_key(graph)] += 1 / len(retained_r)
    exact_joint = {tuple(map(int, state)): weight for state, weight in zip(valid_full, full_weights)}
    exact_classes = defaultdict(float)
    for key, weight in zip(valid_keys, full_weights):
        exact_classes[key] += weight
    joint_tv = total_variation(exact_joint, sampled_joint)
    class_tv = total_variation(exact_classes, sampled_classes)
    # 20,000 retained draws give an iid 200-cell multinomial TV scale near 0.04;
    # these thresholds allow Markov autocorrelation while rejecting a wrong target.
    assert joint_tv < 0.08, f"Sampler missed the exact restricted full joint: TV={joint_tv}"
    assert class_tv < 0.06, f"Sampler missed the exact restricted class posterior: TV={class_tv}"

    return {
        "four_node_DAGs": len(states),
        "four_node_CPDAGs": len(groups),
        "cyclic_states_rejected": cycle_count,
        "maximum_target_error": target_error,
        "maximum_score_equivalence_error": equivalence_error,
        "sampled_class_total_variation": tv,
        "masked_core_full_joint_TV": masked_tv,
        "subnormal_prior_max_probability_error": float(np.max(np.abs(tiny_frequency - [0.25, 0.25, 0.5]))),
        "two_state_periodicity_max_probability_error": float(np.max(np.abs(periodic_frequency - [0, 0.5, 0.5]))),
        "y_sink_valid_DAGs": len(valid_full),
        "y_sink_terminal_parent_masks": len(terminal_masks),
        "y_sink_full_joint_TV": joint_tv,
        "y_sink_class_TV": class_tv,
        "non_last_terminal": int(step_nl.terminal_nodes[0]),
        "multiple_terminal_count": len(step_mt.terminal_nodes),
        "non_last_full_joint_TV": variant_tvs["non_last"],
        "multiple_terminal_full_joint_TV": variant_tvs["multiple"],
        "fixed_graph_active_pairs": len(step_fixed.active_pairs),
        "forced_cycle_rejected": True,
        "reset_trajectory_agreement": True,
    }


def check_basis_score(make_graph_model):
    """Independent raw-unit oracles for BasisScore evidence, posteriors, and mechanisms."""

    def legal_masks(n_nodes, node):
        return [mask for mask in range(1 << n_nodes) if not (mask & (1 << node))]

    def dict_tv(left, right):
        return sum(abs(left.get(key, 0.0) - right.get(key, 0.0))
                   for key in left.keys() | right.keys()) / 2

    # The oracles rebuild the design from the declared raw-unit dictionary:
    # z=(x-loc)/scale with features [z, Gaussian bumps in z], gated by the
    # CHILD mechanism: a linear child sees z alone even when centers exist.
    # Every input is a fixture literal (the child flags included), never score
    # configuration, and the oracles must never call BasisScore.block or any
    # other feature path under test.
    def oracle_block(values, parent, loc, scale, centers, width, child, nonlinear_nodes):
        z = (np.asarray(values, dtype=float) - loc[parent]) / scale[parent]
        if child not in nonlinear_nodes:
            return np.column_stack([z])
        return np.column_stack(
            [z] + [np.exp(-0.5 * ((z - center) / width) ** 2) for center in centers]
        )

    def oracle_design(data, node, mask, loc, scale, centers, width, nonlinear_nodes):
        columns = [np.ones(len(data))]
        for parent in range(data.shape[1]):
            if mask & (1 << parent):
                columns.append(oracle_block(
                    data[:, parent], parent, loc, scale, centers, width,
                    node, nonlinear_nodes))
        return np.column_stack(columns)

    def oracle_posterior(data, node, mask, loc, scale, centers, width,
                         nonlinear_nodes, tau0, lam, alpha0, beta0):
        y = data[:, node]
        design = oracle_design(data, node, mask, loc, scale, centers, width, nonlinear_nodes)
        prior_precision = np.diag(np.r_[tau0, np.full(design.shape[1] - 1, lam)])
        precision = prior_precision + design.T @ design
        mean = np.linalg.solve(precision, design.T @ y)
        alpha_n = alpha0 + 0.5 * len(y)
        residual = y - design @ mean
        beta_n = beta0 + 0.5 * (residual @ residual + mean @ prior_precision @ mean)
        return mean, np.linalg.cholesky(precision), alpha_n, beta_n, prior_precision

    def oracle_evidence(prior_precision, chol_n, alpha0, beta0, alpha_n, beta_n, n_obs):
        logdet0 = np.linalg.slogdet(prior_precision)[1]
        logdet_n = np.linalg.slogdet(chol_n @ chol_n.T)[1]
        return (-0.5 * n_obs * np.log(2.0 * np.pi) + 0.5 * (logdet0 - logdet_n)
                + alpha0 * np.log(beta0) - alpha_n * np.log(beta_n)
                + gammaln(alpha_n) - gammaln(alpha0))

    def oracle_t_logpdf(data, node, mask, loc, scale, centers, width,
                        nonlinear_nodes, tau0, lam, alpha0, beta0):
        y = data[:, node]
        design = oracle_design(data, node, mask, loc, scale, centers, width, nonlinear_nodes)
        prior_precision = np.diag(np.r_[tau0, np.full(design.shape[1] - 1, lam)])
        shape = (beta0 / alpha0) * (
            np.eye(len(y)) + design @ np.linalg.solve(prior_precision, design.T)
        )
        return float(multivariate_t.logpdf(y, loc=np.zeros(len(y)), shape=shape,
                                           df=2.0 * alpha0))

    def sweep_families(score, data, label, centers, width, nonlinear_nodes,
                       loc, scale, tau0, lam, alpha0, beta0):
        """Every legal family of one fixture: t evidence, closed form, posterior."""
        n_nodes = data.shape[1]
        loc_v = np.broadcast_to(np.asarray(loc, dtype=float), (n_nodes,))
        scale_v = np.broadcast_to(np.asarray(scale, dtype=float), (n_nodes,))
        beta_v = np.broadcast_to(np.asarray(beta0, dtype=float), (n_nodes,))
        worst_t = worst_formula = worst_post = 0.0
        families = empty_families = 0
        for node in range(n_nodes):
            beta_node = float(beta_v[node])
            used = 1 + len(centers) if node in nonlinear_nodes else 1
            for mask in legal_masks(n_nodes, node):
                expected_size = 1 + used * bin(mask).count("1")
                actual = float(score.local(node, mask))
                expected_t = oracle_t_logpdf(
                    data, node, mask, loc_v, scale_v, centers, width, nonlinear_nodes,
                    tau0, lam, alpha0, beta_node)
                worst_t = max(worst_t, abs(actual - expected_t))
                assert abs(actual - expected_t) < 1e-8, (
                    f"{label} t-oracle node={node} mask={mask}")
                mean, chol_n, alpha_n, beta_n, prior_precision = oracle_posterior(
                    data, node, mask, loc_v, scale_v, centers, width, nonlinear_nodes,
                    tau0, lam, alpha0, beta_node)
                expected = oracle_evidence(
                    prior_precision, chol_n, alpha0, beta_node, alpha_n, beta_n, len(data))
                worst_formula = max(worst_formula, abs(actual - expected))
                assert abs(actual - expected) < 1e-8, (
                    f"{label} closed form node={node} mask={mask}")
                got_mean, got_chol, got_alpha, got_beta = score.local_posterior(node, mask)
                assert np.asarray(got_mean).shape == (expected_size,), (
                    f"{label} posterior mean dimension node={node} mask={mask}")
                assert np.asarray(got_chol).shape == (expected_size, expected_size), (
                    f"{label} posterior chol dimension node={node} mask={mask}")
                post_errors = (
                    float(np.max(np.abs(np.asarray(got_mean) - mean))),
                    float(np.max(np.abs(np.asarray(got_chol) - chol_n))),
                    abs(float(got_alpha) - alpha_n),
                    abs(float(got_beta) - beta_n),
                )
                worst_post = max(worst_post, *post_errors)
                assert max(post_errors) < 1e-8, (
                    f"{label} posterior node={node} mask={mask}")
                families += 1
                empty_families += int(mask == 0)
        return worst_t, worst_formula, worst_post, families, empty_families

    # Float64 closed forms agree to ~1e-11 here; 1e-8 absorbs exp/log reordering
    # while remaining orders of magnitude below any semantic error (O(1)).
    rng = np.random.default_rng(2610)
    n_nodes = 4
    raw = rng.normal(size=(6, n_nodes))
    raw[:, 1] += 1.4 * np.tanh(raw[:, 0])
    raw[:, 2] += -0.9 * np.tanh(raw[:, 1]) + 0.5 * raw[:, 0]
    raw[:, 3] += 0.8 * np.tanh(raw[:, 2]) - 0.7 * raw[:, 0]
    offsets = np.array([12.0, -7.5, 3.25, -0.75])
    # The shifted fixture with explicit raw-unit loc/scale catches implicit
    # response centering or re-estimated standardization; beta0 stays in raw
    # squared units (never replaced by scale**2).
    fixtures = [
        ("defaults", raw, (-1.0, 1.0), 1.5, 0.0, 1.0, 0.01, 0.1, 2.0, 1.0, range(4)),
        ("raw_units", raw + offsets, (-1.0, 1.0), 1.5, offsets,
         np.array([1.0, 1.5, 0.7, 2.0]), 0.05, 0.7, 3.5,
         np.array([1.0, 0.6, 1.7, 1.2]), range(4)),
        ("linear", raw, (), 1.5, 0.0, 1.0, 0.01, 0.1, 2.0, 1.0, ()),
        ("wide_centers", raw, (-2.0, -1.0, 0.0, 1.0, 2.0), 0.8, 0.0, 1.0,
         1.0, 1.0, 3.0, 0.7, range(4)),
        # The mixed fixture pins the child-resolved dictionary: nodes 1 and 3
        # keep the bumps while 0 and 2 stay linear, so edges with a linear
        # child and a nonlinear parent (and the reverse) are both scored.
        ("mixed", raw, (-1.0, 1.0), 1.5, 0.0, 1.0, 0.01, 0.1, 2.0, 1.0, (1, 3)),
    ]
    t_error = formula_error = post_error = 0.0
    families = empty_families = 0
    mixed_families = 0
    mixed_error = 0.0
    for (label, data, centers, width, loc, scale,
         tau0, lam, alpha0, beta0, nonlinear_nodes) in fixtures:
        score = BasisScore(data, centers=centers, width=width, loc=loc, scale=scale,
                           tau0=tau0, lam=lam, alpha0=alpha0, beta0=beta0,
                           nonlinear_nodes=nonlinear_nodes)
        worst_t, worst_formula, worst_post, count, empty_count = sweep_families(
            score, data, label, centers, width, nonlinear_nodes, loc, scale,
            tau0, lam, alpha0, beta0)
        t_error = max(t_error, worst_t)
        formula_error = max(formula_error, worst_formula)
        post_error = max(post_error, worst_post)
        families += count
        empty_families += empty_count
        if label == "mixed":
            mixed_families = count
            mixed_error = max(worst_t, worst_formula, worst_post)

    # N=1 keeps the proper prior decisive where the design has more columns than
    # observations; duplicated rows and collinear parents keep Lambda_n regular
    # only through the prior, so both stress the same formulas numerically. The
    # two fixtures also pin accepted nonlinear_nodes container forms (list and
    # integer array) through real evidence, not bare construction.
    single = BasisScore(raw[:1], centers=(-1.0, 1.0), width=1.5, loc=0.0, scale=1.0,
                        tau0=0.01, lam=0.1, alpha0=2.0, beta0=1.0,
                        nonlinear_nodes=[0, 1, 2, 3])
    worst_t, worst_formula, worst_post, count, empty_count = sweep_families(
        single, raw[:1], "N1_N_lt_p", (-1.0, 1.0), 1.5, [0, 1, 2, 3], 0.0, 1.0,
        0.01, 0.1, 2.0, 1.0)
    n1_error = max(worst_t, worst_formula, worst_post)
    families += count
    empty_families += empty_count
    duplicated = np.vstack([raw[:4], raw[:4]])
    duplicated[:, 1] = duplicated[:, 0]
    dup_score = BasisScore(duplicated, centers=(-1.0, 1.0), width=1.5, loc=0.0, scale=1.0,
                           tau0=0.01, lam=0.1, alpha0=2.0, beta0=1.0,
                           nonlinear_nodes=np.array([0, 1, 2, 3]))
    worst_t, worst_formula, worst_post, count, empty_count = sweep_families(
        dup_score, duplicated, "duplicated_inputs", (-1.0, 1.0), 1.5,
        np.array([0, 1, 2, 3]), 0.0, 1.0, 0.01, 0.1, 2.0, 1.0)
    duplicated_error = max(worst_t, worst_formula, worst_post)
    families += count
    empty_families += empty_count

    # N0 is a proper prior special case: log evidence exactly 0 and the
    # posterior tuple reduces exactly to the prior parameters.
    empty_data = np.empty((0, 3))
    centers_0 = (-1.0, 1.0)
    used_0 = 1 + len(centers_0)
    score_0 = BasisScore(empty_data, nonlinear_nodes=range(3))
    n0_error = n0_post_error = 0.0
    for node in range(3):
        for mask in legal_masks(3, node):
            value = float(score_0.local(node, mask))
            n0_error = max(n0_error, abs(value))
            assert value == 0.0, "N0 log evidence must be exactly 0"
            mean, chol_n, alpha_n, beta_n, _ = oracle_posterior(
                empty_data, node, mask, np.zeros(3), np.ones(3), centers_0, 1.5,
                range(3), 0.01, 0.1, 2.0, 1.0)
            got_mean, got_chol, got_alpha, got_beta = score_0.local_posterior(node, mask)
            prior_size = 1 + used_0 * bin(mask).count("1")
            assert np.asarray(got_mean).shape == (prior_size,), (
                "N0 prior mean dimension must follow the child dictionary")
            assert np.asarray(got_chol).shape == (prior_size, prior_size), (
                "N0 prior chol dimension must follow the child dictionary")
            n0_post_error = max(
                n0_post_error,
                float(np.max(np.abs(np.asarray(got_mean) - mean))),
                float(np.max(np.abs(np.asarray(got_chol) - chol_n))),
                abs(float(got_alpha) - alpha_n),
                abs(float(got_beta) - beta_n),
            )
    assert n0_post_error < 1e-12, "N0 local_posterior must return the exact prior"

    # Domain boundaries that the contract leaves genuinely uncertain: strict
    # positivity edges, finiteness, vector lengths, and the legal edges that a
    # sloppy validator might wrongly reject. Parent masks require integer dtype.
    good = dict(centers=(-1.0, 1.0), width=1.5, loc=0.0, scale=1.0,
                tau0=0.01, lam=0.1, alpha0=2.0, beta0=1.0,
                nonlinear_nodes=range(3))
    base = raw[:, :3]
    accepted = rejected = 0

    def build(data=None, **overrides):
        return BasisScore(base if data is None else data, **{**good, **overrides})

    def expect_reject(reason, data=None, **overrides):
        nonlocal rejected
        try:
            build(data=data, **overrides)
        except ValueError:
            rejected += 1
        else:
            raise AssertionError(f"Constructor must reject {reason}")

    def expect_accept(reason, data=None, **overrides):
        nonlocal accepted
        build(data=data, **overrides)
        accepted += 1

    def expect_value_error(reason, call):
        nonlocal rejected
        try:
            call()
        except ValueError:
            rejected += 1
        else:
            raise AssertionError(f"Must reject {reason}")

    expect_reject("scale=0", scale=0.0)
    expect_reject("negative scale", scale=-1.0)
    expect_reject("width=0", width=0.0)
    expect_reject("negative width", width=-0.5)
    expect_reject("tau0=0", tau0=0.0)
    expect_reject("negative tau0", tau0=-0.01)
    expect_reject("lam=0", lam=0.0)
    expect_reject("negative lam", lam=-1.0)
    expect_reject("alpha0=0", alpha0=0.0)
    expect_reject("negative alpha0", alpha0=-2.0)
    expect_reject("beta0=0", beta0=0.0)
    expect_reject("negative beta0", beta0=-1.0)
    nan_data = base.copy()
    nan_data[2, 1] = np.nan
    inf_data = base.copy()
    inf_data[0, 0] = np.inf
    expect_reject("NaN data", data=nan_data)
    expect_reject("infinite data", data=inf_data)
    expect_reject("NaN loc", loc=np.array([0.0, np.nan, 0.0]))
    expect_reject("infinite scale", scale=np.array([1.0, np.inf, 1.0]))
    expect_reject("NaN centers", centers=(0.0, np.nan))
    expect_reject("2D centers", centers=np.zeros((2, 2)))
    expect_reject("short scale vector", scale=np.ones(2))
    expect_reject("short beta0 vector", beta0=np.ones(2))
    expect_reject("zero beta0 entry", beta0=np.array([1.0, 0.0, 1.0]))
    expect_reject("bool nonlinear_nodes index", nonlinear_nodes=(True,))
    expect_reject("boolean nonlinear_nodes mask",
                  nonlinear_nodes=np.array([True, False, True]))
    expect_reject("float nonlinear_nodes index", nonlinear_nodes=(1.0,))
    expect_reject("negative nonlinear_nodes index", nonlinear_nodes=(-1,))
    expect_reject("out-of-range nonlinear_nodes index", nonlinear_nodes=(3,))
    expect_reject("duplicate nonlinear_nodes index", nonlinear_nodes=(1, 1))
    expect_reject("no columns", data=np.empty((3, 0)))
    expect_accept("N=0 rows", data=np.empty((0, 3)))
    expect_accept("empty centers", centers=())
    expect_accept("duplicated centers", centers=(0.5, 0.5))
    expect_accept("negative loc", loc=-3.5)
    expect_accept("N=1 row", data=base[:1])
    expect_accept("positive beta0 vector", beta0=np.array([1.0, 0.6, 1.7]))
    expect_accept("positive scale vector", scale=np.array([1.0, 1.5, 0.7]))
    score_b = build()
    expect_value_error("self mask", lambda: score_b.local(1, 0b010))
    expect_value_error("out-of-range mask", lambda: score_b.local(1, 1 << 5))
    expect_value_error("cyclic graph_score", lambda: score_b.graph_score(np.array([2, 4, 1])))
    expect_value_error("float parents array",
                       lambda: score_b.graph_score(np.array([0.0, 1.0, 2.0])))
    parents_zero = np.zeros(3, dtype=np.int64)
    weights_zero = np.zeros((3, 3, 3))
    params_zero = (np.zeros(3), weights_zero, np.ones(3))
    bad_noise = np.zeros((2, 3))
    bad_noise[0, 0] = np.nan
    expect_value_error("non-finite noise",
                       lambda: score_b.simulate(parents_zero, params_zero, 2, None,
                                                noise=bad_noise))
    expect_value_error("non-finite do",
                       lambda: score_b.simulate(parents_zero, params_zero, 2, None,
                                                do={0: np.inf}, noise=np.zeros((2, 3))))
    expect_value_error("n_obs=0",
                       lambda: score_b.simulate(parents_zero, params_zero, 0, None,
                                                noise=np.zeros((0, 3))))
    weights_off = np.zeros((3, 3, 3))
    weights_off[0, 1, 0] = 1.0
    expect_value_error("nonzero weights off the graph",
                       lambda: score_b.predict(parents_zero, (np.zeros(3), weights_off,
                                                              np.ones(3)), np.zeros((2, 3))))
    assert score_b.predict(parents_zero, params_zero, np.empty((0, 3))).shape == (0, 3)
    accepted += 1

    # Regression: the constructor must freeze caller-owned configuration arrays.
    # Mutating centers/loc/scale/beta0/nonlinear_nodes after construction must
    # leave the feature map, family evidence, posterior and predictions
    # bit-identical (aliasing bug fixed in integration). Consumer behavior only;
    # each array is corrupted in turn, the node-index array included.
    centers_f = np.array([-1.0, 1.0])
    loc_f = np.array([0.3, -0.4, 0.7])
    scale_f = np.array([1.2, 0.8, 1.6])
    beta0_f = np.array([1.0, 0.7, 1.9])
    nonlinear_f = np.array([0, 1, 2])
    score_f = BasisScore(base, centers=centers_f, width=1.5, loc=loc_f, scale=scale_f,
                         tau0=0.01, lam=0.1, alpha0=2.0, beta0=beta0_f,
                         nonlinear_nodes=nonlinear_f)
    probe = np.array([-0.7, 0.2, 1.4])
    parents_f = np.array([0, 1 << 0, (1 << 0) | (1 << 1)], dtype=np.int64)
    weights_f = np.zeros((3, 3, 3))
    for child in range(3):
        for parent in range(3):
            if parents_f[child] & (1 << parent):
                weights_f[child, parent] = (0.2 * (child + 1) - 0.1 * (parent + 1)
                                            + 0.05 * np.arange(3))
    params_f = (np.array([0.3, -0.2, 0.5]), weights_f, np.array([1.0, 0.7, 1.9]))
    values_f = np.array([[0.4, -1.0, 0.2], [-0.3, 0.6, 1.1], [1.2, 0.1, -0.8]])
    base_blocks = [score_f.block(node, probe, child=node) for node in range(3)]
    base_locals = np.array([score_f.local(node, mask)
                            for node in range(3) for mask in legal_masks(3, node)])
    base_post = score_f.local_posterior(2, 3)
    base_predict = score_f.predict(parents_f, params_f, values_f)
    targets = (centers_f, loc_f, scale_f, beta0_f, nonlinear_f)
    corruptions = (
        np.array([50.0, -50.0]), np.array([9.0, -9.0, 4.5]),
        np.array([1e6, 1e-6, 3.0]), np.array([7.0, 11.0, 13.0]),
        np.array([0]),
    )
    originals = tuple(target.copy() for target in targets)
    alias_delta = 0.0
    for corrupted, bad in zip(targets, corruptions):
        for target, original in zip(targets, originals):
            target[:] = original
        corrupted[:] = bad
        got_blocks = [score_f.block(node, probe, child=node) for node in range(3)]
        got_locals = np.array([score_f.local(node, mask)
                               for node in range(3) for mask in legal_masks(3, node)])
        got_post = score_f.local_posterior(2, 3)
        got_predict = score_f.predict(parents_f, params_f, values_f)
        for node in range(3):
            assert np.array_equal(got_blocks[node], base_blocks[node]), (
                "block must freeze caller centers/loc/scale/nonlinear_nodes")
        assert np.array_equal(got_locals, base_locals), (
            "local evidence must freeze caller config")
        for got, was in zip(got_post, base_post):
            assert np.array_equal(np.asarray(got), np.asarray(was)), (
                "local_posterior must freeze caller config")
        assert np.array_equal(got_predict, base_predict), (
            "predict must freeze caller config")
        alias_delta = max(alias_delta, float(np.max(np.abs(got_locals - base_locals))))
    assert alias_delta == 0.0

    # Parameter draws: shapes and exact off-edge zeros, then the exact posterior
    # and prior laws. 4,000 iid draws give an asymptotic 1% KS critical value
    # 1.63/sqrt(4000) ~= 0.026; the 0.05 threshold keeps a ~2x margin against
    # seed-level fluctuation while any wrong scale, mean, or family drives the
    # statistic toward 1.
    data_k = rng.normal(size=(6, 3))
    data_k[:, 1] += 1.3 * np.tanh(data_k[:, 0])
    data_k[:, 2] += -0.8 * np.tanh(data_k[:, 1])
    centers_k = (-1.0, 1.0)
    # The all-nonlinear declaration keeps node 2's theta layout at
    # weights_d[2, *]: the KS laws below concatenate the full padded rows of
    # parents 0 and 1, so node 2 must stay nonlinear.
    score_k = BasisScore(data_k, centers=centers_k, width=1.5, loc=0.0, scale=1.0,
                         tau0=0.01, lam=0.1, alpha0=2.0, beta0=1.0,
                         nonlinear_nodes=range(3))
    parents_k = np.array([0, 1, 3], dtype=np.int64)
    intercept_d, weights_d, variance_d = score_k.draw_parameters(
        parents_k, np.random.default_rng(2616))
    assert np.asarray(intercept_d).shape == (3,)
    assert np.asarray(weights_d).shape == (3, 3, 1 + len(centers_k))
    assert np.asarray(variance_d).shape == (3,) and np.all(variance_d > 0)
    for child in range(3):
        for parent in range(3):
            if not (parents_k[child] & (1 << parent)):
                assert np.all(weights_d[child, parent] == 0.0), "Off-edge weights must be 0"
    mean_o, chol_o, alpha_o, beta_o, _ = oracle_posterior(
        data_k, 2, 3, np.zeros(3), np.ones(3), centers_k, 1.5, range(3),
        0.01, 0.1, 2.0, 1.0)
    n_draws = 4_000
    draws_rng = np.random.default_rng(2615)
    v_draws = np.empty(n_draws)
    theta_draws = np.empty((n_draws, mean_o.size))
    for index in range(n_draws):
        intercept_d, weights_d, variance_d = score_k.draw_parameters(parents_k, draws_rng)
        v_draws[index] = variance_d[2]
        theta_draws[index] = np.concatenate(
            ([intercept_d[2]], weights_d[2, 0], weights_d[2, 1]))
    assert np.all(v_draws > 0)
    ks_variance = float(kstest(v_draws, invgamma(alpha_o, scale=beta_o).cdf).statistic)
    precision_o = chol_o @ chol_o.T
    directions = (np.eye(mean_o.size)[0], np.ones(mean_o.size) / np.sqrt(mean_o.size))
    ks_coefficients = 0.0
    for direction in directions:
        scale_u = np.sqrt(beta_o / alpha_o * direction @ np.linalg.solve(precision_o, direction))
        ks_coefficients = max(ks_coefficients, float(kstest(
            theta_draws @ direction,
            student_t(2.0 * alpha_o, loc=direction @ mean_o, scale=scale_u).cdf).statistic))
    assert ks_variance < 0.05 and ks_coefficients < 0.05

    # prior=True must ignore the observed data and reproduce the score's prior:
    # v ~ IG(alpha0, beta0), theta | v ~ Normal(0, v Lambda0^-1).
    diagonal0 = np.r_[0.01, np.full(mean_o.size - 1, 0.1)]
    prior_rng = np.random.default_rng(2616)
    prior_v = np.empty(n_draws)
    prior_theta = np.empty((n_draws, mean_o.size))
    for index in range(n_draws):
        intercept_d, weights_d, variance_d = score_k.draw_parameters(
            parents_k, prior_rng, prior=True)
        prior_v[index] = variance_d[2]
        prior_theta[index] = np.concatenate(
            ([intercept_d[2]], weights_d[2, 0], weights_d[2, 1]))
    ks_prior = float(kstest(prior_v, invgamma(2.0, scale=1.0).cdf).statistic)
    for direction in directions:
        scale_u = np.sqrt(0.5 * np.sum(direction ** 2 / diagonal0))
        ks_prior = max(ks_prior, float(kstest(
            prior_theta @ direction,
            student_t(4.0, loc=0.0, scale=scale_u).cdf).statistic))
    assert ks_prior < 0.05

    # N0 draws are exact prior draws as well.
    zero_rng = np.random.default_rng(2617)
    zero_v = np.empty(n_draws)
    zero_theta = np.empty((n_draws, mean_o.size))
    for index in range(n_draws):
        intercept_d, weights_d, variance_d = score_0.draw_parameters(parents_k, zero_rng)
        zero_v[index] = variance_d[2]
        zero_theta[index] = np.concatenate(
            ([intercept_d[2]], weights_d[2, 0], weights_d[2, 1]))
    ks_zero = float(kstest(zero_v, invgamma(2.0, scale=1.0).cdf).statistic)
    for direction in directions:
        scale_u = np.sqrt(0.5 * np.sum(direction ** 2 / diagonal0))
        ks_zero = max(ks_zero, float(kstest(
            zero_theta @ direction,
            student_t(4.0, loc=0.0, scale=scale_u).cdf).statistic))
    assert ks_zero < 0.05

    # Posterior means in MC standard-error units (variance-moment convergence is
    # left to the KS statistics: inverse-gamma tails converge slowly there).
    covariance_o = (beta_o / (alpha_o - 1.0)) * np.linalg.inv(precision_o)
    se_mean = np.sqrt(np.diag(covariance_o) / n_draws)
    mean_z = float(np.max(np.abs(theta_draws.mean(axis=0) - mean_o) / se_mean))
    expected_v = beta_o / (alpha_o - 1.0)
    var_v = beta_o ** 2 / ((alpha_o - 1.0) ** 2 * (alpha_o - 2.0))
    var_z = float(abs(v_draws.mean() - expected_v) / np.sqrt(var_v / n_draws))
    # A few reported statistics; 5-sigma bounds keep deterministic-seed flake
    # risk negligible while remaining far below any semantic shift.
    assert mean_z < 5.0 and var_z < 5.0

    # predict/simulate against explicit manual equations built from known
    # coefficients on a MIXED dictionary: children 1 and 3 keep the bumps while
    # children 0 and 2 are linear, so the used feature width follows the CHILD
    # and edges (1, 0) and (2, 1) each join parent and child of opposite kind.
    # Linear children are zero-padded in slots 1..K. The manual sums reassociate
    # float operations differently, so agreement is tight allclose; the
    # structural claims (hard do replacement, common-noise no-path outcomes)
    # are bit-identical.
    n_m = 4
    centers_m = (-1.0, 1.0)
    width_m = 1.5
    nonlinear_m = (1, 3)
    loc_m = np.array([0.2, -0.5, 0.9, -1.1])
    scale_m = np.array([1.0, 1.5, 0.7, 2.0])
    parents_m = np.array([0, 1 << 0, (1 << 0) | (1 << 1), 0], dtype=np.int64)
    intercept_m = np.array([0.3, -1.1, 0.7, 2.0])
    weights_m = np.zeros((n_m, n_m, 1 + len(centers_m)))
    for child in range(n_m):
        used_m = 1 + len(centers_m) if child in nonlinear_m else 1
        for parent in range(n_m):
            if parents_m[child] & (1 << parent):
                weights_m[child, parent, :used_m] = (
                    0.11 * (child + 1) - 0.07 * (parent + 1)
                    + 0.05 * np.arange(used_m))
    variance_m = np.array([0.4, 0.9, 1.3, 0.25])
    parameters_m = (intercept_m, weights_m, variance_m)
    score_m = BasisScore(np.zeros((2, n_m)), centers=centers_m, width=width_m,
                         loc=loc_m, scale=scale_m, nonlinear_nodes=nonlinear_m)
    values_in = np.random.default_rng(2611).normal(size=(5, n_m))
    noise = np.random.default_rng(2612).normal(size=(6, n_m))

    def manual_block(values, parent, child):
        z = (values[:, parent] - loc_m[parent]) / scale_m[parent]
        if child not in nonlinear_m:
            return np.column_stack([z])
        return np.column_stack(
            [z] + [np.exp(-0.5 * ((z - center) / width_m) ** 2) for center in centers_m]
        )

    def manual_predict(values):
        means = np.empty((len(values), n_m))
        for child in range(n_m):
            used_m = 1 + len(centers_m) if child in nonlinear_m else 1
            total = np.full(len(values), intercept_m[child])
            for parent in range(n_m):
                if parents_m[child] & (1 << parent):
                    total = total + (manual_block(values, parent, child)
                                     @ weights_m[child, parent, :used_m])
            means[:, child] = total
        return means

    def manual_simulate(noise_matrix, do=None):
        values = np.empty_like(noise_matrix)
        for child in range(n_m):
            if do is not None and child in do:
                values[:, child] = do[child]
                continue
            used_m = 1 + len(centers_m) if child in nonlinear_m else 1
            total = np.full(len(noise_matrix), intercept_m[child])
            for parent in range(n_m):
                if parents_m[child] & (1 << parent):
                    total = total + (manual_block(values, parent, child)
                                     @ weights_m[child, parent, :used_m])
            values[:, child] = total + np.sqrt(variance_m[child]) * noise_matrix[:, child]
        return values

    block_error = 0.0
    for node in range(n_m):
        for child in (1, 2):
            used_m = 1 + len(centers_m) if child in nonlinear_m else 1
            got = score_m.block(node, values_in[:, node], child=child)
            assert got.shape == (5, used_m), (
                "block width must follow the child: (5, 1+K) or (5, 1)")
            block_error = max(block_error, float(np.max(np.abs(
                got - manual_block(values_in, node, child)))))
    score_lin = BasisScore(np.zeros((2, n_m)), centers=(), width=width_m,
                           loc=loc_m, scale=scale_m, nonlinear_nodes=())
    z_only = score_lin.block(0, values_in[:, 0], child=0)
    assert z_only.shape == (5, 1)
    block_error = max(block_error, float(np.max(np.abs(
        z_only[:, 0] - (values_in[:, 0] - loc_m[0]) / scale_m[0]))))
    assert block_error < 1e-12

    # Padded slots of linear children are never written: posterior and prior
    # draws must both leave coefficients 1..K of the linear children at exactly
    # zero while the padded weights shape stays (n, n, 1+K).
    for prior_draw in (False, True):
        _, draw_w, _ = score_m.draw_parameters(
            parents_m, np.random.default_rng(2619), prior=prior_draw)
        assert np.asarray(draw_w).shape == (n_m, n_m, 1 + len(centers_m))
        for child in range(n_m):
            if child in nonlinear_m:
                continue
            assert np.all(draw_w[child, :, 1:] == 0.0), (
                "Linear children must draw exact-zero padded weights")

    predicted = score_m.predict(parents_m, parameters_m, values_in)
    predict_error = float(np.max(np.abs(predicted - manual_predict(values_in))))
    assert predict_error < 1e-12

    simulated = score_m.simulate(parents_m, parameters_m, 6, None, noise=noise)
    simulate_error = float(np.max(np.abs(simulated - manual_simulate(noise))))
    assert simulate_error < 1e-12

    do_value = 4.25
    simulated_do = score_m.simulate(parents_m, parameters_m, 6, None,
                                    do={0: do_value}, noise=noise)
    assert np.all(simulated_do[:, 0] == do_value), "do must hard-replace the equation"
    do_error = float(np.max(np.abs(simulated_do - manual_simulate(noise, do={0: do_value}))))
    simulated_do2 = score_m.simulate(parents_m, parameters_m, 6, None,
                                     do={2: -2.5}, noise=noise)
    assert np.all(simulated_do2[:, 2] == -2.5), "do must replace non-root equations too"
    do_error = max(do_error, float(np.max(np.abs(
        simulated_do2 - manual_simulate(noise, do={2: -2.5})))))
    assert do_error < 1e-12

    # Common noise + reachability: with no directed path from the intervened
    # node the outcome must be bit-identical, so the structural zero in the
    # finite effect is exact, not a threshold.
    alternate = score_m.simulate(parents_m, parameters_m, 6, None, do={0: -2.0}, noise=noise)
    assert not has_path(parents_m, 0, 3)
    assert has_path(parents_m, 0, 1) and has_path(parents_m, 1, 2)
    assert has_path(parents_m, 0, 0), "Identity reachability is documented True"
    assert np.array_equal(simulated_do[:, 3], alternate[:, 3]), "No-path outcome must be exact"
    assert not np.array_equal(simulated_do[:, 1], alternate[:, 1])
    no_path_effect = float(np.max(np.abs(simulated_do[:, 3] - alternate[:, 3])))
    assert no_path_effect == 0.0
    unaffected = score_m.simulate(parents_m, parameters_m, 6, None, do={2: 7.0}, noise=noise)
    assert np.array_equal(simulated_do2[:, [0, 1, 3]], unaffected[:, [0, 1, 3]])
    replay_a = score_m.simulate(parents_m, parameters_m, 6, np.random.default_rng(2618),
                                do={0: do_value})
    replay_b = score_m.simulate(parents_m, parameters_m, 6, np.random.default_rng(2618),
                                do={0: do_value})
    assert np.array_equal(replay_a, replay_b), "Seeded replay must be deterministic"

    # Coherent change of units: y' = c y, scale' = c scale, beta0' = c^2 beta0.
    # The density carries -N log c; graph ratios cancel that Jacobian. A
    # translation is NOT an invariance here: it would also require translating
    # the zero-mean intercept prior, not merely the predictor feature location.
    c_scale = 2.5
    data_u = raw[:, :3]
    score_a = BasisScore(data_u, centers=(-1.0, 1.0), width=1.5, loc=0.0, scale=1.0,
                         tau0=0.01, lam=0.1, alpha0=2.0, beta0=1.0,
                         nonlinear_nodes=range(3))
    data_b = data_u.copy()
    data_b[:, 1] = c_scale * data_u[:, 1]
    loc_b = np.zeros(3)
    scale_b = np.array([1.0, c_scale, 1.0])
    beta0_b = np.array([1.0, c_scale ** 2, 1.0])
    score_b2 = BasisScore(data_b, centers=(-1.0, 1.0), width=1.5, loc=loc_b, scale=scale_b,
                          tau0=0.01, lam=0.1, alpha0=2.0, beta0=beta0_b,
                          nonlinear_nodes=range(3))
    unit_error = 0.0
    jacobian = np.log(c_scale) * len(data_u)
    for node in range(3):
        for mask in legal_masks(3, node):
            expected = float(score_a.local(node, mask)) - (jacobian if node == 1 else 0.0)
            unit_error = max(unit_error, abs(float(score_b2.local(node, mask)) - expected))
    assert unit_error < 1e-8, "Coherent unit transform must shift only by the Jacobian"
    unit_table_error = float(np.max(np.abs(np.asarray(score_b2.table) - np.asarray(score_a.table))))
    assert unit_table_error < 1e-8, "table differences must cancel the Jacobian"
    score_c = BasisScore(data_b, centers=(-1.0, 1.0), width=1.5, loc=loc_b, scale=scale_b,
                         tau0=0.01, lam=0.1, alpha0=2.0, beta0=1.0,
                         nonlinear_nodes=range(3))
    noncoherent_gap = min(
        abs(float(score_c.local(1, mask)) - (float(score_a.local(1, mask)) - jacobian))
        for mask in legal_masks(3, 1))
    assert noncoherent_gap > 1e-3, "beta0 carries raw squared units and cannot be skipped"

    # Tiny nonlinear DAG: the compiled PyMC target equals the exact absolute
    # graph likelihood plus prior up to the constant the table subtracts (each
    # node's intercept-only evidence). The constant must cancel across states.
    score_target = BasisScore(data_k, nonlinear_nodes=range(3))
    pairs3 = pairs_for(3)
    probs3 = np.tile([0.5, 0.35, 0.15], (len(pairs3), 1))
    model_target = make_graph_model(score_target, probs3, tuple("abc"))
    logp_target = model_target.compile_logp()
    intercept_sum = sum(float(score_target.local(node, 0)) for node in range(3))
    residuals = []
    target_cycles = 0
    for state_tuple in product(range(3), repeat=len(pairs3)):
        state = np.asarray(state_tuple, dtype="int64")
        graph = states_to_parents(state, pairs3, 3)
        actual = float(logp_target({"edge": state}))
        try:
            topological_order(graph)
        except ValueError:
            assert np.isneginf(actual), "Model gave finite mass to a directed cycle"
            target_cycles += 1
            continue
        absolute = sum(float(score_target.local(node, int(graph[node]))) for node in range(3))
        prior_lp = float(np.log(probs3[np.arange(len(pairs3)), state]).sum())
        residuals.append(actual - absolute - prior_lp)
    target_error = max(abs(value + intercept_sum) for value in residuals)
    target_spread = float(np.ptp(residuals))
    assert target_error < 1e-9, "Compiled target must match the absolute likelihood + constant"
    assert target_spread < 1e-9, "The table constant must cancel across graphs"

    # One full-DAG posterior comparison with terminal y on the article's mixed
    # mechanism split (linear upstream, nonlinear y): exact enumeration gives
    # every supported acyclic state's absolute posterior; the sampler's full
    # states (including y's parent set) must match at DAG level, not only by
    # equivalence class.
    pair_probs = np.zeros((len(pairs3), 3))
    pair_index = {tuple(map(int, pair)): index for index, pair in enumerate(pairs3)}
    pair_probs[pair_index[0, 1]] = [0.2, 0.5, 0.3]
    pair_probs[pair_index[0, 2]] = [0.3, 0.7, 0.0]
    pair_probs[pair_index[1, 2]] = [0.4, 0.6, 0.0]
    score_y = BasisScore(data_k, nonlinear_nodes=(2,))
    exact_states, exact_log_weights = [], []
    for state_tuple in product(range(3), repeat=len(pairs3)):
        state = np.asarray(state_tuple, dtype="int64")
        if np.any(pair_probs[np.arange(len(pairs3)), state] == 0.0):
            continue
        graph = states_to_parents(state, pairs3, 3)
        try:
            topological_order(graph)
        except ValueError:
            continue
        exact_states.append(tuple(map(int, state)))
        exact_log_weights.append(
            sum(float(score_y.local(node, int(graph[node]))) for node in range(3))
            + float(np.log(pair_probs[np.arange(len(pairs3)), state]).sum()))
    exact_probs = dict(zip(exact_states, softmax(np.asarray(exact_log_weights))))
    model_y = make_graph_model(score_y, pair_probs, tuple("abc"))
    with model_y:
        step_y = TemperedGraphStep(
            [model_y["edge"]], score_y.table, pairs3, pair_probs,
            np.r_[np.geomspace(1.0, 0.01, 8), 0.0])
        trace_y = pm.sample(
            draws=4_000, tune=800, chains=2, cores=1, step=step_y,
            random_seed=2613, progressbar=False, compute_convergence_checks=False)
    retained_y = trace_y.posterior.edge.values.reshape(-1, len(pairs3))
    sampled_freq = defaultdict(float)
    sampled_class = defaultdict(float)
    sampled_terminal = defaultdict(float)
    exact_class = defaultdict(float)
    exact_terminal = defaultdict(float)
    for state in retained_y:
        key = tuple(map(int, state))
        assert key in exact_probs, "Sampler emitted a forbidden or cyclic graph"
        graph = states_to_parents(np.asarray(state), pairs3, 3)
        topological_order(graph)
        sampled_freq[key] += 1 / len(retained_y)
        sampled_class[mec_key(graph)] += 1 / len(retained_y)
        sampled_terminal[int(graph[2])] += 1 / len(retained_y)
    for key, weight in exact_probs.items():
        graph = states_to_parents(np.asarray(key), pairs3, 3)
        exact_class[mec_key(graph)] += weight
        exact_terminal[int(graph[2])] += weight
    # 2 chains x 4,000 retained = 8,000 draws over the enumerated full DAGs.
    # Inflating for autocorrelation by 2x gives per-cell sd <= 0.5/sqrt(4000)
    # ~= 0.008. With fixed seed 2613, allow TV 0.06, cell error 0.05,
    # and terminal marginal error 0.025 (about three marginal standard errors).
    # These are smoke tolerances, not a guarantee against every wrong target.
    sampler_tv = dict_tv(exact_probs, dict(sampled_freq))
    sampler_cell = max(abs(sampled_freq.get(key, 0.0) - weight)
                       for key, weight in exact_probs.items())
    assert sampler_tv < 0.06, f"Full-DAG sampler TV too large: {sampler_tv}"
    assert sampler_cell < 0.05, f"Full-DAG cell error too large: {sampler_cell}"
    terminal_error = max(
        abs(sampled_terminal.get(mask, 0.0) - probability)
        for mask, probability in exact_terminal.items())
    assert terminal_error < 0.025, f"Terminal parent marginal off: {terminal_error}"
    class_tv = dict_tv(dict(exact_class), dict(sampled_class))

    # Non-score-equivalence fixture: one Markov-equivalent triple (verified via
    # mec_key) receives measurably different absolute finite-basis evidence.
    # This pins a fact about THIS fixture only: no universal equality or
    # inequality across constructions is asserted, and no 50:50 claim about an
    # unidentified pair appears anywhere.
    score_mec = BasisScore(data_k, nonlinear_nodes=range(3))
    members = (np.array([0, 1, 2]), np.array([2, 0, 2]), np.array([2, 4, 0]))
    keys = {mec_key(member) for member in members}
    assert len(keys) == 1, "Fixture members must be Markov equivalent"
    member_scores = np.array([
        sum(float(score_mec.local(node, int(mask))) for node, mask in enumerate(member))
        for member in members
    ])
    mecs_gap = float(np.ptp(member_scores))
    assert mecs_gap > 1e-6, "This nonlinear fixture must not be score equivalent"

    return {
        "legal_families_checked": families,
        "t_oracle_max_abs_error": t_error,
        "closed_form_max_abs_error": formula_error,
        "local_posterior_max_abs_error": post_error,
        "mixed_families_checked": mixed_families,
        "mixed_oracle_max_abs_error": mixed_error,
        "intercept_only_families_checked": empty_families,
        "N1_N_lt_p_max_abs_error": n1_error,
        "duplicated_input_max_abs_error": duplicated_error,
        "N0_max_abs_log_evidence": n0_error,
        "N0_local_posterior_max_abs_error": n0_post_error,
        "N0_prior_draw_KS": ks_zero,
        "posterior_draw_variance_KS": ks_variance,
        "posterior_draw_coefficients_KS": ks_coefficients,
        "posterior_mean_max_z": mean_z,
        "posterior_mean_variance_max_z": var_z,
        "prior_draw_KS": ks_prior,
        "block_max_abs_error": block_error,
        "predict_max_abs_error": predict_error,
        "simulate_max_abs_error": simulate_error,
        "do_replacement_max_abs_error": do_error,
        "no_path_effect_max_abs": no_path_effect,
        "config_aliasing_max_delta": alias_delta,
        "unit_transform_jacobian_error": unit_error,
        "unit_transform_table_error": unit_table_error,
        "noncoherent_beta0_gap": noncoherent_gap,
        "domain_rejections_verified": rejected,
        "domain_acceptances_verified": accepted,
        "pymc_target_constant_error": target_error,
        "pymc_target_constant_spread": target_spread,
        "pymc_target_cycles_rejected": target_cycles,
        "full_DAGs_enumerated": len(exact_states),
        "full_DAG_sampler_TV": sampler_tv,
        "full_DAG_sampler_max_cell_error": sampler_cell,
        "terminal_parent_marginal_error": terminal_error,
        "class_TV_descriptive": class_tv,
        "non_score_equivalence_gap": mecs_gap,
    }
