"""Compiled, likelihood-tempered DAG Metropolis step for PyMC 6.

Support-aware proposals over hard-zero pair states, exact terminal-parent
marginalization, and rank-based initial-state generation.  Full graph
reconstruction at every retained draw preserves original node indices and
the complete seven-node BGe score table.
"""

import numpy as np
import pymc as pm
from numba import njit
from pymc.step_methods import BlockedStep
from pymc.util import get_random_generator, get_value_vars_from_user_vars

from graph_math import pairs_for, parents_to_states, states_to_parents


@njit(cache=True)
def acyclic(parents):
    """Kahn traversal of one integer parent mask per node."""
    seen = 0
    complete = (1 << len(parents)) - 1
    for _ in range(len(parents)):
        before = seen
        for node in range(len(parents)):
            if not (seen & (1 << node)) and not (parents[node] & ~seen):
                seen |= 1 << node
        if seen == complete:
            return True
        if seen == before:
            return False
    return False


@njit(cache=True)
def update_journeys(level_labels, journeys):
    """Count completed cold->hot->cold trips, not adjacent-temperature exchanges."""
    if len(level_labels) < 2:
        return 0
    cold, hot = level_labels[0], level_labels[-1]
    completed = int(journeys[cold] == 2)
    journeys[cold] = 1
    if journeys[hot] == 1:
        journeys[hot] = 2
    return completed


def _pair_support(pair_probs, pairs, n_nodes):
    """Validate canonical pair probabilities without changing their support."""
    if (isinstance(n_nodes, (bool, np.bool_))
            or not isinstance(n_nodes, (int, np.integer)) or not 1 <= n_nodes < 63):
        raise ValueError("n_nodes must be an integer between 1 and 62")
    pairs = np.asarray(pairs)
    if pairs.dtype.kind not in "iu" or not np.array_equal(pairs, pairs_for(n_nodes)):
        raise ValueError("pairs must contain all unordered pairs in canonical order")
    probabilities = np.asarray(pair_probs, dtype=float)
    if (probabilities.shape != (len(pairs), 3) or not np.isfinite(probabilities).all()
            or np.any(probabilities < 0) or not np.allclose(probabilities.sum(axis=1), 1)):
        raise ValueError("pair_probs must be finite non-negative normalized rows of three states")
    return probabilities, pairs.astype(np.int64, copy=False), probabilities > 0


def initial_states(pair_probs, pairs, n_nodes, rng, *, empty=False):
    """Construct a supported DAG directly from random topological ranks.

    ``empty=True`` removes every optional edge, but retains required
    adjacencies. Required directed edges constrain the random ordering.
    Starts need not follow the graph prior; they must lie in its support.
    """
    probabilities, pairs, support = _pair_support(pair_probs, pairs, n_nodes)
    if empty and support[:, 0].all():
        return np.zeros(len(pairs), dtype=np.int64)

    required = np.zeros(n_nodes, dtype=np.int64)
    forced = ~support[:, 0] & (support[:, 1] ^ support[:, 2])
    for pair in np.flatnonzero(forced):
        left, right = pairs[pair]
        parent, child = (left, right) if support[pair, 1] else (right, left)
        required[child] |= 1 << parent
    if required.any():
        order = _random_topological_order(required, rng)
        ranks = np.empty(n_nodes, dtype=np.int64)
        ranks[order] = np.arange(n_nodes)
    else:
        ranks = rng.permutation(n_nodes)

    left, right = pairs.T
    directions = np.where(ranks[left] < ranks[right], 1, 2)
    if empty:
        include = ~support[:, 0]
    else:
        directional = probabilities[np.arange(len(pairs)), directions]
        include = rng.random(len(pairs)) < directional / (probabilities[:, 0] + directional)
    return np.where(include, directions, 0).astype(np.int64)


def _random_topological_order(parents, rng):
    """Randomize a Kahn traversal of the required directed edges."""
    order, seen = [], 0
    for _ in parents:
        eligible = [
            node for node in range(len(parents))
            if not seen & (1 << node) and not int(parents[node]) & ~seen
        ]
        if not eligible:
            raise ValueError("Required directed edges form a cycle")
        node = int(rng.choice(eligible))
        order.append(node)
        seen |= 1 << node
    return order


def terminal_parent_distributions(table, pairs, pair_probs):
    """Return exact supported parent-set distributions for no-outgoing nodes.

    Parameters
    ----------
    table : array-like, shape (n, 2**n)
        Original full-model BGe local scores, not scores from a refitted core.
    pairs : array-like, shape (m, 2)
        Canonical node pairs from ``pairs_for``.
    pair_probs : array-like, shape (m, 3)
        Local pair probabilities after imposing hard restrictions.

    Returns
    -------
    dict[int, tuple[numpy.ndarray, numpy.ndarray]]
        Node index mapped to global-bit parent masks and their normalized
        probabilities. Isolated nodes have the single parent mask zero.
        Required parents occur in every supported mask.
    """
    table = np.asarray(table, dtype=float)
    if table.ndim != 2 or not 1 <= table.shape[0] < 63 or table.shape[1] != 1 << table.shape[0]:
        raise ValueError("table must have shape (n, 2**n), with 1 <= n < 63")
    n_nodes = table.shape[0]
    probabilities, pairs, support = _pair_support(pair_probs, pairs, n_nodes)
    has_outgoing = np.zeros(n_nodes, dtype=bool)
    has_outgoing[pairs[support[:, 1], 0]] = True
    has_outgoing[pairs[support[:, 2], 1]] = True

    result = {}
    for node in np.flatnonzero(~has_outgoing):
        required_mask, optional = 0, []
        for pair in np.flatnonzero((pairs[:, 0] == node) | (pairs[:, 1] == node)):
            left, right = pairs[pair]
            parent, direction = (left, 1) if right == node else (right, 2)
            if not support[pair, direction]:
                continue
            if not support[pair, 0]:
                required_mask |= 1 << int(parent)
            else:
                optional.append((
                    int(parent), np.log(probabilities[pair, 0]),
                    np.log(probabilities[pair, direction]),
                ))

        subsets = np.arange(1 << len(optional), dtype=np.int64)
        masks = np.full(len(subsets), required_mask, dtype=np.int64)
        log_weights = np.zeros(len(subsets))
        for bit, (parent, log_absent, log_present) in enumerate(optional):
            included = (subsets & (1 << bit)) != 0
            masks[included] |= 1 << parent
            log_weights += np.where(included, log_present, log_absent)
        log_weights += table[node, masks]
        if not np.isfinite(log_weights).all():
            raise ValueError("Supported terminal parent-set scores must be finite")
        weights = np.exp(log_weights - log_weights.max())
        weights /= weights.sum()
        result[int(node)] = (masks, weights)
    return result


@njit(cache=True)
def _core_sweep(parents, evidence, table, pairs, log_pair_probs, betas,
                rng, sweeps, swap_counts, level_labels, journeys,
                active_pairs, supported_states, support_counts, core_nodes):
    """Core-only MH sweep with support-aware proposals on active pairs.

    Parameters
    ----------
    active_pairs : int array
        Original indices of mutable, non-terminal pairs.
    supported_states : int array, shape (n_pairs, 3)
        Supported categories, including the current state.
    support_counts : int array, shape (n_pairs,)
        Number of supported categories for each pair.
    core_nodes : int array
        Original indices of non-terminal nodes whose evidence is summed.
    """
    accepted = invalid = roundtrips = 0
    n_active = len(active_pairs)
    if not n_active:
        return accepted, invalid, roundtrips
    n_rep = parents.shape[0]

    for _ in range(sweeps):
        for replica in range(n_rep):
            for _ in range(n_active):
                pair = active_pairs[rng.integers(0, n_active)]

                left, right = pairs[pair]
                old_left = parents[replica, left]
                old_right = parents[replica, right]
                old = (1 if old_right & (1 << left)
                       else 2 if old_left & (1 << right) else 0)
                new = supported_states[pair, rng.integers(0, support_counts[pair])]
                if new == old:
                    continue

                parents[replica, left] = old_left & ~(1 << right)
                parents[replica, right] = old_right & ~(1 << left)
                if new == 1:
                    parents[replica, right] |= 1 << left
                elif new == 2:
                    parents[replica, left] |= 1 << right

                if acyclic(parents[replica]):
                    delta = (table[left, parents[replica, left]] - table[left, old_left]
                             + table[right, parents[replica, right]] - table[right, old_right])
                    log_ratio = (betas[replica] * delta
                                 + log_pair_probs[pair, new] - log_pair_probs[pair, old])
                    if np.log(rng.random()) < log_ratio:
                        if replica == 0:
                            accepted += 1
                        continue
                elif replica == 0:
                    invalid += 1
                parents[replica, left] = old_left
                parents[replica, right] = old_right

            evidence[replica] = 0.0
            for c in range(len(core_nodes)):
                node = core_nodes[c]
                evidence[replica] += table[node, parents[replica, node]]

        for level in range(n_rep - 1):
            log_ratio = ((betas[level] - betas[level + 1])
                         * (evidence[level + 1] - evidence[level]))
            if np.log(rng.random()) < log_ratio:
                for node in range(parents.shape[1]):
                    parents[level, node], parents[level + 1, node] = (
                        parents[level + 1, node], parents[level, node])
                evidence[level], evidence[level + 1] = evidence[level + 1], evidence[level]
                level_labels[level], level_labels[level + 1] = level_labels[level + 1], level_labels[level]
                swap_counts[level] += 1
            roundtrips += update_journeys(level_labels, journeys)
    return accepted, invalid, roundtrips


class TemperedGraphStep(BlockedStep):
    """Sample one ``pmd.Categorical`` graph variable; retain only the cold replica.

    Proposes on mutable, non-terminal *active pairs* with support-aware
    uniform state draws, including self-proposals to prevent periodicity.
    Fixed pairs (single supported state) and
    terminal-incident pairs are excluded from proposals.  Terminal parent
    sets are drawn exactly at each retained cold-state update from their
    precomputed CDF.

    Stats report accepted graph changes and cycle rejections per attempted
    update, adjacent swap fractions, and full cold->hot->cold round trips.
    The fixed ladder never adapts. Warmup journeys do not count as retained trips.
    """

    name = "tempered_graph"
    stats_dtypes_shapes = {}

    def __init__(self, vars, table, pairs, pair_probs, betas, sweeps=1,
                 model=None, rng=None, blocked=True):
        model = pm.modelcontext(model)
        self.vars = get_value_vars_from_user_vars(vars, model)
        if len(self.vars) != 1 or not self.vars[0].dtype.startswith("int"):
            raise ValueError("Provide exactly one integer categorical graph variable")
        table = np.asarray(table, dtype=float)
        if table.ndim != 2:
            raise ValueError("table must be a node-by-parent-mask matrix")
        n = table.shape[0]
        if not 1 <= n < 63 or table.shape[1] != 1 << n:
            raise ValueError("table must have shape (n, 2**n), with 1 <= n < 63")
        valid = (np.arange(1 << n)[None, :] & (1 << np.arange(n))[:, None]) == 0
        if not np.isfinite(table[valid]).all():
            raise ValueError("All valid parent-set scores must be finite")
        pairs = np.asarray(pairs)
        if not np.array_equal(pairs, pairs_for(n)) or pairs.dtype.kind not in "iu":
            raise ValueError("pairs must contain all unordered pairs in canonical order")
        pair_probs = np.asarray(pair_probs, dtype=float)
        if (pair_probs.shape != (len(pairs), 3) or not np.isfinite(pair_probs).all()
                or np.any(pair_probs < 0) or not np.allclose(pair_probs.sum(axis=1), 1)):
            raise ValueError(
                "pair_probs must be non-negative finite normalized rows of three states"
            )
        support = pair_probs > 0
        if np.any(~support.any(axis=1)):
            raise ValueError("Every pair must have at least one supported state")
        betas = np.asarray(betas, dtype=float)
        if (betas.ndim != 1 or not len(betas) or not np.isfinite(betas).all()
                or betas[0] != 1 or np.any(betas < 0) or np.any(np.diff(betas) > 0)):
            raise ValueError("betas must decrease from 1 to a nonnegative value")
        if isinstance(sweeps, (bool, np.bool_)) or not isinstance(sweeps, (int, np.integer)) or sweeps < 1:
            raise ValueError("sweeps must be a positive integer")
        initial = model.initial_point()[self.vars[0].name]
        if initial.shape != (len(pairs),):
            raise ValueError("The graph variable must have one state per unordered pair")
        self.table = np.ascontiguousarray(table)
        self.pairs = np.ascontiguousarray(pairs, dtype=np.int64)
        self.pair_probs = np.ascontiguousarray(pair_probs)
        self.log_pair_probs = np.full_like(pair_probs, -np.inf)
        np.log(pair_probs, out=self.log_pair_probs, where=support)
        self.betas = betas.copy()
        self.sweeps = int(sweeps)
        self.rng = get_random_generator(rng)
        self.tune = True
        self._initialized = False

        # Detect terminals and build active-pair / support structures.
        self._terminal_distributions = terminal_parent_distributions(
            table, pairs, pair_probs
        )
        terminal_set = set(self._terminal_distributions.keys())
        self.terminal_nodes = np.array(sorted(terminal_set), dtype=np.int64)

        support_counts = support.sum(axis=1)
        incident_to_terminal = np.zeros(len(pairs), dtype=bool)
        for p in range(len(pairs)):
            if int(pairs[p, 0]) in terminal_set or int(pairs[p, 1]) in terminal_set:
                incident_to_terminal[p] = True
        self.active_pairs = np.where(
            (support_counts >= 2) & ~incident_to_terminal
        )[0].astype(np.int64)

        supported_states = np.full((len(pairs), 3), -1, dtype=np.int64)
        for pair in range(len(pairs)):
            supported_states[pair, :support_counts[pair]] = np.flatnonzero(support[pair])
        self._supported_states = supported_states
        self._support_counts = np.ascontiguousarray(support_counts, dtype=np.int64)
        self._core_nodes = np.array(
            [i for i in range(n) if i not in terminal_set], dtype=np.int64
        )

        # Precompute CDFs for terminal draws (internal detail).
        self._terminal_cdfs = {}
        for t, (masks, probs) in self._terminal_distributions.items():
            cdf = np.cumsum(probs)
            cdf[-1] = 1.0
            self._terminal_cdfs[t] = (masks, cdf)

        self.stats_dtypes_shapes = {
            "cold_accept": (float, []), "cold_invalid": (float, []),
            "roundtrips": (np.int64, []), "tune": (bool, []),
            **{f"swap_{i}": (float, []) for i in range(len(betas) - 1)},
        }
        self.stats_dtypes = [{name: dtype for name, (dtype, _) in self.stats_dtypes_shapes.items()}]

    def setup_chain(self, rng, tune, draws):
        super().setup_chain(rng, tune, draws)
        self.rng = get_random_generator(rng)
        self.tune = tune > 0
        self._initialized = False
        replicas, nodes = len(self.betas), self.table.shape[0]
        self._parents = np.zeros((replicas, nodes), dtype=np.int64)
        self._evidence = np.zeros(replicas)
        self._swaps = np.zeros(max(0, replicas - 1), dtype=np.int64)
        self._labels = np.arange(replicas)
        self._journeys = np.zeros(replicas, dtype=np.int64)
        self._journeys[0] = 1

    def stop_tuning(self):
        super().stop_tuning()
        self._journeys[:] = 0
        self._journeys[self._labels[0]] = 1

    def step(self, point):
        name = self.vars[0].name
        if not self._initialized:
            states = np.asarray(point[name])
            if (states.shape != (len(self.pairs),) or states.dtype.kind not in "iu"
                    or np.any((states < 0) | (states > 2))):
                raise ValueError("Initial graph must have one integer state in {0, 1, 2} per pair")
            if np.any(self.pair_probs[np.arange(len(self.pairs)), states] == 0):
                raise ValueError("Initial graph contains a zero-support pair state")
            cold = states_to_parents(states, self.pairs, self.table.shape[0])
            if not acyclic(cold):
                raise ValueError("Initial graph must be acyclic")
            cold[self.terminal_nodes] = 0
            self._parents[:] = cold
            self._initialized = True
        self._swaps[:] = 0
        n_active = len(self.active_pairs)
        attempts = max(1, self.sweeps * n_active)
        accepted, invalid, roundtrips = _core_sweep(
            self._parents, self._evidence, self.table, self.pairs,
            self.log_pair_probs, self.betas, self.rng, self.sweeps,
            self._swaps, self._labels, self._journeys,
            self.active_pairs, self._supported_states, self._support_counts,
            self._core_nodes,
        )
        full_states = self._reconstruct_full()
        updated = dict(point)
        updated[name] = full_states
        stats = {
            "cold_accept": accepted / attempts,
            "cold_invalid": invalid / attempts,
            "roundtrips": roundtrips,
            "tune": self.tune,
            **{f"swap_{i}": count / self.sweeps for i, count in enumerate(self._swaps)},
        }
        return updated, [stats]

    def _reconstruct_full(self):
        """Draw terminal parents without changing the cold core replica."""
        full_parents = self._parents[0].copy()
        for node, (masks, cdf) in self._terminal_cdfs.items():
            index = np.searchsorted(cdf, self.rng.random(), side="right")
            full_parents[node] = masks[index]
        return parents_to_states(full_parents, self.pairs)
