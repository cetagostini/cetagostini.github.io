"""Small, exhaustive correctness oracle; never used to fit the seven-node example."""

from collections import defaultdict
from itertools import product
from types import SimpleNamespace

import numpy as np
import pymc as pm
from scipy.special import softmax

from graph_math import (
    BGeScore, cpdag, mec_key, pair_probabilities, pairs_for,
    states_to_parents, topological_order,
)
from graph_sampling import (
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
