"""BGe scores, exact CPDAGs, and conjugate linear-Gaussian mechanisms.

Graph representation: parents[child] is a bit mask over parent nodes.
Coefficient matrices use [child, parent]; CPDAG adjacency uses [parent, child].
"""

from itertools import combinations

import numpy as np
from scipy.linalg import cho_solve, solve_triangular
from scipy.special import multigammaln


def _indices(mask):
    mask = int(mask)
    result = []
    while mask:
        bit = mask & -mask
        result.append(bit.bit_length() - 1)
        mask ^= bit
    return result


def _parent_array(parents):
    values = np.asarray(parents)
    if values.ndim != 1 or not 1 <= len(values) < 63 or values.dtype.kind not in "iu":
        raise ValueError("parents must be a one-dimensional integer array with 1–62 nodes")
    n = len(values)
    if np.any(values < 0) or np.any(values >= 1 << n):
        raise ValueError("Parent masks contain out-of-range node bits")
    return values.astype(np.int64, copy=False)


def pairs_for(n):
    return np.asarray(list(combinations(range(n), 2)), dtype=np.int64).reshape(-1, 2)


def pair_probabilities(direction_prior, allowed=None):
    """Convert source-by-target probabilities and hard masks to pair states.

    Parameters
    ----------
    direction_prior : array-like, shape (n, n)
        Local directional probabilities before conditioning on acyclicity.
        Opposing entries must sum to at most one; the diagonal must be zero.
    allowed : boolean array-like, shape (n, n), optional
        Admissible directions, with a false diagonal. Masking conditions the
        pair prior on its surviving states; it does not force the reverse edge.

    Returns
    -------
    numpy.ndarray
        Rows in ``pairs_for(n)`` order, with columns absent, forward, backward.
        Exact zeros remain hard exclusions, not small positive probabilities.
    """
    prior = np.asarray(direction_prior, dtype=float)
    if prior.ndim != 2 or prior.shape[0] != prior.shape[1] or not 1 <= len(prior) < 63:
        raise ValueError("direction_prior must be a square matrix with 1–62 nodes")
    if not np.isfinite(prior).all() or np.any((prior < 0) | (prior > 1)):
        raise ValueError("Directional probabilities must be finite and between zero and one")
    if np.any(np.diag(prior) != 0):
        raise ValueError("Self-edge probabilities must be zero")
    if allowed is not None:
        allowed = np.asarray(allowed)
        if allowed.shape != prior.shape or allowed.dtype.kind != "b":
            raise ValueError("allowed must be a Boolean matrix matching direction_prior")
        if np.any(np.diag(allowed)):
            raise ValueError("Self-edges cannot be allowed")
    left, right = pairs_for(len(prior)).T
    forward, backward = prior[left, right], prior[right, left]
    if np.any(forward + backward > 1 + 4 * np.finfo(float).eps):
        raise ValueError("Opposing directional probabilities must sum to at most one")
    probabilities = np.column_stack((np.maximum(1 - (forward + backward), 0), forward, backward))
    if allowed is not None:
        probabilities[:, 1] *= allowed[left, right]
        probabilities[:, 2] *= allowed[right, left]
    total = probabilities.sum(axis=1, keepdims=True)
    if np.any(total == 0):
        raise ValueError("The mask removes every supported state of a pair")
    return probabilities / total


def states_to_parents(states, pairs, n):
    states = np.asarray(states)
    if states.shape != (len(pairs),) or states.dtype.kind not in "iu" or np.any((states < 0) | (states > 2)):
        raise ValueError("Each pair needs an integer state: 0 absent, 1 forward, 2 backward")
    parents = np.zeros(n, dtype=np.int64)
    for state, (left, right) in zip(states, pairs):
        if state == 1:
            parents[right] |= 1 << int(left)
        elif state == 2:
            parents[left] |= 1 << int(right)
    return parents


def parents_to_states(parents, pairs):
    parents = _parent_array(parents)
    left, right = pairs.T
    forward = (parents[right] & (1 << left)) != 0
    backward = (parents[left] & (1 << right)) != 0
    if np.any(forward & backward):
        raise ValueError("A pair cannot carry both arrow directions")
    return np.where(forward, 1, np.where(backward, 2, 0)).astype(np.int64)


def topological_order(parents):
    parents = _parent_array(parents)
    seen, order = 0, []
    while len(order) < len(parents):
        before = seen
        for node, mask in enumerate(parents):
            if not seen & (1 << node) and not int(mask) & ~seen:
                order.append(node)
                seen |= 1 << node
        if seen == before:
            raise ValueError("Graph contains a directed cycle")
    return order


def mec_key(parents):
    """Canonical skeleton/collider key using arbitrary-precision Python integers."""
    parents = _parent_array(parents)
    skeleton, colliders = 0, 0
    n = len(parents)
    for pair, (left, right) in enumerate(combinations(range(n), 2)):
        if (int(parents[right]) & (1 << left)) or (int(parents[left]) & (1 << right)):
            skeleton |= 1 << pair
        else:
            for child in range(n):
                mask = int(parents[child])
                if mask & (1 << left) and mask & (1 << right):
                    colliders |= 1 << (pair * n + child)
    return skeleton, colliders


def cpdag(parents):
    """Chickering's ORDER-EDGES/LABEL-EDGES, not sampled-member intersection.

    See Chickering (2002), JMLR 2:445–498, Figures 4–5:
    https://www.jmlr.org/papers/volume2/chickering02a/chickering02a.pdf
    Both matrix entries true denotes a reversible undirected edge.
    """
    parents = _parent_array(parents)
    order = topological_order(parents)
    rank = {node: index for index, node in enumerate(order)}
    parent_sets = [set(_indices(mask)) for mask in parents]
    edges = [(parent, child) for child in order
             for parent in sorted(parent_sets[child], key=rank.get, reverse=True)]
    labels = {edge: 0 for edge in edges}  # 0 unknown, 1 compelled, 2 reversible
    for x, y in edges:
        if labels[x, y]:
            continue
        forced = False
        for w in parent_sets[x]:
            if labels[w, x] != 1:
                continue
            if w not in parent_sets[y]:
                for z in parent_sets[y]:
                    labels[z, y] = 1
                forced = True
                break
            labels[w, y] = 1
        if forced:
            continue
        status = 1 if any(z != x and z not in parent_sets[x] for z in parent_sets[y]) else 2
        for z in parent_sets[y]:
            if labels[z, y] == 0:
                labels[z, y] = status
    result = np.zeros((len(parents), len(parents)), dtype=bool)
    for (parent, child), status in labels.items():
        result[parent, child] = True
        if status == 2:
            result[child, parent] = True
    return result


class BGeScore:
    """Compatible normal-Wishart marginal likelihoods for complete Gaussian data.

    prior_sd specifies sqrt(diag(E[Sigma])) in the complete reference model.
    N=0 gives the same mechanism prior used by the data-conditioned score.
    Only local parent sets are cached; complete DAGs are never enumerated.
    """

    def __init__(self, data, prior_sd, alpha_mu=1.0, alpha_w=None, nu=None):
        data = np.asarray(data, dtype=float)
        if data.ndim != 2 or not np.isfinite(data).all() or not 1 <= data.shape[1] < 63:
            raise ValueError("BGe requires finite complete data with shape (N, n), 1 <= n < 63")
        self.N, self.n = data.shape
        self.alpha_mu = float(alpha_mu)
        self.alpha_w = float(self.n + 4 if alpha_w is None else alpha_w)
        prior_sd = np.asarray(prior_sd, dtype=float)
        nu = np.zeros(self.n) if nu is None else np.asarray(nu, dtype=float)
        if not np.isfinite(self.alpha_mu) or self.alpha_mu <= 0:
            raise ValueError("alpha_mu must be finite and positive")
        if not np.isfinite(self.alpha_w) or self.alpha_w <= self.n + 1:
            raise ValueError("alpha_w must exceed n + 1 so E[Sigma] exists")
        if prior_sd.shape != (self.n,) or not np.isfinite(prior_sd).all() or np.any(prior_sd <= 0):
            raise ValueError("prior_sd must contain one positive finite scale per node")
        if nu.shape != (self.n,) or not np.isfinite(nu).all():
            raise ValueError("nu must contain one finite prior mean per node")
        self.T = (self.alpha_w - self.n - 1) * np.diag(prior_sd ** 2)
        self.kappa = self.N + self.alpha_mu
        empirical_mean = data.mean(axis=0) if self.N else nu
        centered = data - empirical_mean
        shift = empirical_mean - nu
        self.R = (self.T + centered.T @ centered
                  + self.N * self.alpha_mu / self.kappa * np.outer(shift, shift))
        self.mean = (self.N * empirical_mean + self.alpha_mu * nu) / self.kappa
        self._log_m = np.zeros(1 << self.n)
        for mask in range(1, 1 << self.n):
            indices = _indices(mask)
            size = len(indices)
            adjusted = self.alpha_w - self.n + size
            _, log_t = np.linalg.slogdet(self.T[np.ix_(indices, indices)])
            sign, log_r = np.linalg.slogdet(self.R[np.ix_(indices, indices)])
            if sign <= 0:
                raise FloatingPointError("Updated normal-Wishart scale is not positive definite")
            self._log_m[mask] = (
                size / 2 * np.log(self.alpha_mu / self.kappa)
                + multigammaln((self.N + adjusted) / 2, size) - multigammaln(adjusted / 2, size)
                - self.N * size / 2 * np.log(np.pi)
                + adjusted / 2 * log_t - (self.N + adjusted) / 2 * log_r
            )
        self.table = np.full((self.n, 1 << self.n), -np.inf)
        for node in range(self.n):
            empty = self.local(node, 0)
            for mask in range(1 << self.n):
                if not mask & (1 << node):
                    self.table[node, mask] = self.local(node, mask) - empty

    def local(self, node, mask):
        mask = int(mask)
        if not 0 <= node < self.n or not 0 <= mask < 1 << self.n or mask & (1 << node):
            raise ValueError("Invalid node or parent mask")
        return float(self._log_m[mask | (1 << node)] - self._log_m[mask])

    def graph_score(self, parents):
        if len(parents) != self.n:
            raise ValueError("Graph and score must have the same nodes")
        topological_order(parents)
        return sum(self.local(node, mask) for node, mask in enumerate(parents))


def draw_parameters(score, parents, rng):
    """Draw compatible local intercepts, coefficients, and residual variances."""
    parents = _parent_array(parents)
    if len(parents) != score.n:
        raise ValueError("Graph and score must have the same nodes")
    topological_order(parents)
    intercept, coefficients, variance = np.empty(score.n), np.zeros((score.n, score.n)), np.empty(score.n)
    for node, mask in enumerate(parents):
        indices = _indices(mask)
        residual_scale = score.R[node, node]
        if indices:
            chol = np.linalg.cholesky(score.R[np.ix_(indices, indices)])
            beta_mean = cho_solve((chol, True), score.R[indices, node])
            residual_scale -= score.R[node, indices] @ beta_mean
        if residual_scale <= 0:
            raise FloatingPointError("Conditional residual scale must be positive")
        df = score.N + score.alpha_w - score.n + len(indices) + 1
        variance[node] = residual_scale / rng.chisquare(df)
        if indices:
            beta = beta_mean + np.sqrt(variance[node]) * solve_triangular(
                chol.T, rng.normal(size=len(indices)), lower=False)
            coefficients[node, indices] = beta
        intercept[node] = (score.mean[node] - coefficients[node] @ score.mean
                           + rng.normal(scale=np.sqrt(variance[node] / score.kappa)))
    return intercept, coefficients, variance


def _mechanism_order(parents, coefficients):
    order = topological_order(parents)
    n = len(parents)
    coefficients = np.asarray(coefficients)
    if coefficients.shape != (n, n) or not np.isfinite(coefficients).all():
        raise ValueError("Coefficients must be a finite child-by-parent matrix")
    permitted = np.array([[bool(int(mask) & (1 << parent)) for parent in range(n)] for mask in parents])
    if np.any(coefficients[~permitted] != 0):
        raise ValueError("Nonzero coefficient outside the graph")
    return order


def total_effects(parents, coefficients):
    """Return d E[X_child | do(X_intervention=t)] / dt, including identity effects."""
    order = _mechanism_order(parents, coefficients)
    effects = np.eye(len(parents))
    for node in order:
        indices = _indices(parents[node])
        if indices:
            effects[node] += coefficients[node, indices] @ effects[indices]
    return effects


def simulate_scm(parents, intercept, coefficients, variance, n_obs, rng):
    """Generate one dataset under a shared graph and independent Gaussian errors."""
    order = _mechanism_order(parents, coefficients)
    intercept, variance = np.asarray(intercept), np.asarray(variance)
    n = len(parents)
    if (intercept.shape != (n,) or variance.shape != (n,) or not np.isfinite(intercept).all()
            or not np.isfinite(variance).all() or np.any(variance <= 0)):
        raise ValueError("Intercepts must be finite and variances positive, one per node")
    values = rng.normal(size=(n_obs, n)) * np.sqrt(variance) + intercept
    for node in order:
        indices = _indices(parents[node])
        if indices:
            values[:, node] += values[:, indices] @ coefficients[node, indices]
    return values
