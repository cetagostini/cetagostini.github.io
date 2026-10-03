"""BGe scores, exact CPDAGs, and conjugate additive-mechanism models.

Graph representation: parents[child] is a bit mask over parent nodes.
Coefficient matrices use [child, parent]; BasisScore weight tensors carry a
trailing feature axis [child, parent, feature], where each child uses only its
first u(child) = 1+K (nonlinear) or 1 (linear) slots and the rest is zero
padding; CPDAG adjacency uses [parent, child].
"""

from itertools import combinations

import numpy as np
from scipy.linalg import cho_solve, solve_triangular
from scipy.special import gammaln, multigammaln


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


def _scalar(value, name, positive):
    array = np.asarray(value, dtype=float)
    if array.ndim != 0 or not np.isfinite(array) or (positive and array <= 0):
        raise ValueError(f"{name} must be a" + (" positive" if positive else "") + " finite scalar")
    return float(array)


def _per_node(value, n, name, positive):
    array = np.asarray(value, dtype=float)
    if array.ndim == 0:
        array = np.full(n, float(array))
    elif array.shape != (n,):
        raise ValueError(f"{name} must be a scalar or a vector of length n")
    if not np.isfinite(array).all() or (positive and np.any(array <= 0)):
        raise ValueError(f"{name} must be finite" + (" and positive" if positive else ""))
    return array.copy()


def _check_index(value, n, name):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or not 0 <= value < n:
        raise ValueError(f"{name} must be an integer in [0, {n})")
    return int(value)


def _check_rng(rng):
    if rng is None or not hasattr(rng, "normal") or not hasattr(rng, "gamma"):
        raise ValueError("rng must be a numpy random generator")
    return rng


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


def has_path(parents, src, dst):
    """Boolean directed reachability from ``src`` to ``dst`` in ``parents``.

    A node reaches itself, so ``has_path(parents, i, i)`` is True by
    convention; structural-zero summaries therefore treat self-effects as
    present. Only the graph is consulted — no path products or coefficients.
    """
    parents = _parent_array(parents)
    n = len(parents)
    src = _check_index(src, n, "src")
    dst = _check_index(dst, n, "dst")
    if src == dst:
        return True
    children = [[] for _ in range(n)]
    for child, mask in enumerate(parents):
        for parent in _indices(mask):
            children[parent].append(child)
    stack, seen = [src], {src}
    while stack:
        for child in children[stack.pop()]:
            if child == dst:
                return True
            if child not in seen:
                seen.add(child)
                stack.append(child)
    return False


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


class BasisScore:
    """Conjugate evidence for additive mechanisms in a fixed feature dictionary.

    Each node follows

        x_i = b_i0 + sum_{j in Pa(i)} sum_k b_jk phi_k(x_j) + eps_i,
        eps_i ~ Normal(0, sigma_i^2),

    where phi is the frozen map :meth:`block`, gated by the child i: the
    standardized linear term z_j = (x_j - loc_j) / scale_j with the parent's
    loc/scale alone for a linear child, plus one uncentered Gaussian bump
    exp(-.5 ((z_j - c) / width)^2) per fixed center c for a nonlinear child.
    ``nonlinear_nodes`` names the nonlinear child mechanisms: None (default)
    keeps the generic all-nonlinear dictionary, () makes every mechanism
    strictly linear, and a tuple of node indices selects the nonlinear
    children. Each child uses u(i) = 1 + K feature columns when nonlinear and
    u(i) = 1 when linear; the padded weight tensor stays n x n x (1+K) with
    unused slots exactly zero. The target is the raw node value; nothing is
    centered or re-estimated from the data. The prior is
    the conjugate normal-inverse-gamma pair: sigma_i^2 ~ InvGamma(alpha0,
    beta0) in the shape/scale convention p(v) = beta0^alpha0/Gamma(alpha0) *
    v^(-alpha0-1) * exp(-beta0/v), and given sigma_i^2 the coefficient vector
    (intercept first, then parent feature blocks in ascending parent order) is
    Normal(0, sigma_i^2 Lambda0^{-1}) with intercept precision tau0 and
    feature precision lam. Every family evidence is then closed form
    (:meth:`local`).

    Because each family is linear in its own fixed feature blocks, the score is
    not guaranteed to be score equivalent: Markov-equivalent DAGs can receive
    different evidence, and orientation gaps can reflect the dictionary as much
    as the data. beta0 is an inverse-gamma scale in raw squared target units.
    N = 0 data is valid: log evidence is exactly 0 and draws come from the
    prior. Posterior tuples are cached per (node, parent mask), so repeated
    parameter draws never refit from the N rows.
    """

    def __init__(self, data, *, nonlinear_nodes=None, centers=(-1.0, 1.0), width=1.5,
                 loc=0.0, scale=1.0, tau0=0.01, lam=0.1, alpha0=2.0, beta0=1.0):
        data = np.asarray(data, dtype=float)
        if data.ndim != 2 or not np.isfinite(data).all() or not 1 <= data.shape[1] < 63:
            raise ValueError("BasisScore requires finite complete data with shape (N, n), 1 <= n < 63")
        centers = np.asarray(centers, dtype=float)
        if centers.ndim != 1 or not np.isfinite(centers).all():
            raise ValueError("centers must be a finite one-dimensional array, possibly empty")
        self.data = data.copy()
        self.N, self.n = data.shape
        self.centers = centers.copy()
        self.K = len(centers)
        self.n_features = 1 + self.K
        if nonlinear_nodes is None:
            self.nonlinear_nodes = frozenset(range(self.n))
        else:
            try:
                requested = list(nonlinear_nodes)
            except TypeError:
                raise ValueError(
                    "nonlinear_nodes must be None or an iterable of node indices"
                ) from None
            checked = [_check_index(index, self.n, "nonlinear_nodes entry")
                       for index in requested]
            if len(set(checked)) != len(checked):
                raise ValueError("nonlinear_nodes must not contain duplicate indices")
            self.nonlinear_nodes = frozenset(checked)
        self.width = _scalar(width, "width", positive=True)
        self.loc = _per_node(loc, self.n, "loc", positive=False)
        self.scale = _per_node(scale, self.n, "scale", positive=True)
        self.tau0 = _scalar(tau0, "tau0", positive=True)
        self.lam = _scalar(lam, "lam", positive=True)
        self.alpha0 = _scalar(alpha0, "alpha0", positive=True)
        self.beta0 = _per_node(beta0, self.n, "beta0", positive=True)
        self._cache = {}
        self.table = np.zeros((self.n, 1 << self.n))
        for node in range(self.n):
            empty = self.local(node, 0)
            for mask in range(1 << self.n):
                if not mask & (1 << node):
                    self.table[node, mask] = self.local(node, mask) - empty

    def block(self, parent, values, *, child):
        """Frozen feature block of one parent's observations for one child.

        Columns are z = (values - loc[parent]) / scale[parent], always
        standardized with the parent's loc/scale, followed for a nonlinear
        ``child`` by one uncentered Gaussian bump
        exp(-.5 ((z - center) / width)^2) per entry of ``centers``, in centers
        order. A linear child receives the z column alone, so the block carries
        u(child) columns: 1 + K nonlinear, 1 linear. The child gate is applied
        before any bump is computed. The same map drives scoring, prediction
        and simulation.
        """
        parent = _check_index(parent, self.n, "parent")
        child = _check_index(child, self.n, "child")
        raw = np.asarray(values, dtype=float)
        if raw.ndim != 1 or not np.isfinite(raw).all():
            raise ValueError("values must be a finite one-dimensional array")
        z = (raw - self.loc[parent]) / self.scale[parent]
        if child not in self.nonlinear_nodes:
            return z[:, None]
        columns = [z] + [np.exp(-0.5 * ((z - center) / self.width) ** 2)
                         for center in self.centers]
        return np.column_stack(columns)

    def _check_family(self, node, mask):
        node = _check_index(node, self.n, "node")
        if (isinstance(mask, bool) or not isinstance(mask, (int, np.integer))
                or not 0 <= mask < 1 << self.n):
            raise ValueError(f"mask must be an integer in [0, {1 << self.n})")
        mask = int(mask)
        if mask & (1 << node):
            raise ValueError("A parent mask cannot contain the node itself")
        return node, mask

    def _posterior(self, node, mask):
        """Cached NIG posterior of one family on the raw target.

        The design carries one block per parent at the child's used width
        u(node): 1 + K columns for a nonlinear child, one for a linear child,
        matching the prior size 1 + u(node) * number of parents. Returns (mean,
        lower Cholesky of Lambda_n, alpha_n, beta_n) with the stable
        b_n = beta0 + .5 * (||y - X m_n||^2 + m_n' Lambda0 m_n).
        """
        key = (int(node), int(mask))
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        y = self.data[:, node]
        parents = _indices(mask)
        design = np.column_stack(
            [np.ones(self.N)]
            + [self.block(j, self.data[:, j], child=node) for j in parents])
        size = design.shape[1]
        lambda0 = np.empty(size)
        lambda0[0] = self.tau0
        lambda0[1:] = self.lam
        precision = design.T @ design
        precision[np.diag_indices(size)] += lambda0
        chol = np.linalg.cholesky(precision)
        mean = cho_solve((chol, True), design.T @ y)
        residual = y - design @ mean
        b_n = self.beta0[node] + 0.5 * (
            float(residual @ residual) + float((lambda0 * mean) @ mean))
        result = (mean, chol, self.alpha0 + self.N / 2.0, b_n)
        self._cache[key] = result
        return result

    def local(self, node, mask):
        """Absolute log marginal likelihood of one family.

        The intercept column is always included and the target is the raw node
        value; self-parent and out-of-range masks are rejected. N = 0 gives
        exactly 0. The table built in the constructor stores this value
        relative to each node's intercept-only family.
        """
        node, mask = self._check_family(node, mask)
        if self.N == 0:
            return 0.0
        mean, chol, a_n, b_n = self._posterior(node, mask)
        size = mean.shape[0]
        log_det_prior = np.log(self.tau0) + (size - 1) * np.log(self.lam)
        log_det_post = 2.0 * np.log(np.diag(chol)).sum()
        return float(
            -0.5 * self.N * np.log(2 * np.pi)
            + 0.5 * (log_det_prior - log_det_post)
            + self.alpha0 * np.log(self.beta0[node])
            - a_n * np.log(b_n)
            + gammaln(a_n) - gammaln(self.alpha0)
        )

    def local_posterior(self, node, mask):
        """NIG posterior tuple of one family: (mean, chol, alpha_n, beta_n).

        ``mean`` is the posterior coefficient mean and ``chol`` the lower
        Cholesky factor of the posterior precision Lambda_n, both ordered
        intercept first, then the used-width feature blocks of the parents in
        ascending order. ``alpha_n`` and ``beta_n`` are the inverse-gamma shape
        and scale of sigma_i^2 given the data. Copies are returned so the cache
        is never exposed.
        """
        node, mask = self._check_family(node, mask)
        mean, chol, a_n, b_n = self._posterior(node, mask)
        return mean.copy(), chol.copy(), float(a_n), float(b_n)

    def graph_score(self, parents):
        if len(parents) != self.n:
            raise ValueError("Graph and score must have the same nodes")
        topological_order(parents)
        return sum(self.local(node, mask) for node, mask in enumerate(parents))

    def _check_graph(self, parents):
        parents = _parent_array(parents)
        if len(parents) != self.n:
            raise ValueError("Graph and score must have the same nodes")
        return parents, topological_order(parents)

    def _check_parameters(self, parents, parameters):
        try:
            intercept, weights, variance = parameters
        except (TypeError, ValueError):
            raise ValueError(
                "parameters must be the (intercept, weights, variance) tuple"
            ) from None
        intercept = np.asarray(intercept, dtype=float)
        weights = np.asarray(weights, dtype=float)
        variance = np.asarray(variance, dtype=float)
        if (intercept.shape != (self.n,)
                or weights.shape != (self.n, self.n, self.n_features)
                or variance.shape != (self.n,)):
            raise ValueError(
                "Parameter shapes must be (n,), (n, n, 1+K) and (n,)")
        if not (np.isfinite(intercept).all() and np.isfinite(weights).all()
                and np.isfinite(variance).all()) or np.any(variance <= 0):
            raise ValueError("Parameters must be finite and variances positive")
        permitted = np.array(
            [[bool(int(mask) & (1 << parent)) for parent in range(self.n)]
             for mask in parents])
        if np.any(weights[~permitted] != 0):
            raise ValueError("Nonzero weight outside the graph")
        return intercept, weights, variance

    def draw_parameters(self, parents, rng, *, prior=False):
        """Draw one mechanism: (intercept, weights, variance).

        ``weights[i, j]`` holds the coefficients of parent j's feature block in
        child i's mechanism, in :meth:`block` column order, and is zero off the
        graph and in the padding slots beyond child i's used width u(i);
        ``intercept`` and ``variance`` have one entry per node. The draw
        is the exact conjugate update v = b_n / Gamma(a_n, rate 1),
        theta = m_n + sqrt(v) * solve(chol(Lambda_n).T, z) with z standard
        normal. ``prior=True`` ignores the observed data and draws from the
        mechanism prior at the same used width u(i), with the same feature map
        and prior as the score.
        """
        parents, _ = self._check_graph(parents)
        rng = _check_rng(rng)
        intercept = np.zeros(self.n)
        weights = np.zeros((self.n, self.n, self.n_features))
        variance = np.empty(self.n)
        for node, mask in enumerate(parents):
            mask = int(mask)
            p_indices = _indices(mask)
            used = self.n_features if node in self.nonlinear_nodes else 1
            if prior:
                size = 1 + used * len(p_indices)
                lambda0 = np.empty(size)
                lambda0[0] = self.tau0
                lambda0[1:] = self.lam
                mean = np.zeros(size)
                chol = np.diag(np.sqrt(lambda0))
                a_n, b_n = self.alpha0, self.beta0[node]
            else:
                mean, chol, a_n, b_n = self._posterior(node, mask)
            variance[node] = b_n / rng.gamma(a_n, 1.0)
            step = solve_triangular(
                chol.T, rng.normal(size=mean.shape[0]), lower=False)
            theta = mean + np.sqrt(variance[node]) * step
            intercept[node] = theta[0]
            for slot, parent in enumerate(p_indices):
                start = 1 + slot * used
                weights[node, parent, :used] = theta[start:start + used]
        return intercept, weights, variance

    def predict(self, parents, parameters, values):
        """Conditional means E[x_i | x_pa] under one mechanism draw.

        ``values`` is a finite raw (N, n) matrix; the result is the raw-scale
        mean matrix (N, n) with each mechanism's intercept included and no
        noise. Every term evaluates the same :meth:`block` map as scoring at
        the child's used width u(node), against the first u(node) weight slots.
        """
        parents, _ = self._check_graph(parents)
        intercept, weights, _ = self._check_parameters(parents, parameters)
        values = np.asarray(values, dtype=float)
        if values.ndim != 2 or values.shape[1] != self.n or not np.isfinite(values).all():
            raise ValueError(f"values must be a finite matrix with shape (N, {self.n})")
        means = np.empty((len(values), self.n))
        means[:] = intercept
        for node, mask in enumerate(parents):
            used = self.n_features if node in self.nonlinear_nodes else 1
            for parent in _indices(mask):
                means[:, node] += (
                    self.block(parent, values[:, parent], child=node)
                    @ weights[node, parent, :used])
        return means

    def simulate(self, parents, parameters, n_obs, rng, *, do=None, noise=None):
        """Generate raw values (n_obs, n) under one mechanism draw.

        Mechanisms are evaluated in topological order. Exogenous noise is
        standard normal per node, scaled by sqrt(variance); ``noise``, when
        given, is that (n_obs, n) standard-normal matrix and replaces the
        random draws, enabling common random numbers, so ``rng`` may then be
        None. ``do`` maps node indices to finite constants that replace both
        the mechanism equation and its noise at those nodes. Every mechanism
        term uses the same :meth:`block` map as scoring at the child's used
        width u(node), against the first u(node) weight slots.
        """
        parents, order = self._check_graph(parents)
        intercept, weights, variance = self._check_parameters(parents, parameters)
        if isinstance(n_obs, bool) or not isinstance(n_obs, (int, np.integer)) or n_obs < 1:
            raise ValueError("n_obs must be a positive integer")
        n_obs = int(n_obs)
        interventions = {}
        if do is not None:
            try:
                items = dict(do).items()
            except (TypeError, ValueError):
                raise ValueError("do must map node indices to finite constants") from None
            for key, value in items:
                node = _check_index(key, self.n, "do key")
                amount = _scalar(value, "do value", positive=False)
                interventions[node] = amount
        if noise is None:
            rng = _check_rng(rng)
            exogenous = rng.normal(size=(n_obs, self.n))
        else:
            exogenous = np.asarray(noise, dtype=float)
            if exogenous.shape != (n_obs, self.n) or not np.isfinite(exogenous).all():
                raise ValueError(
                    f"noise must be a finite matrix with shape ({n_obs}, {self.n})")
        values = np.empty((n_obs, self.n))
        for node in order:
            if node in interventions:
                values[:, node] = interventions[node]
                continue
            values[:, node] = intercept[node] + np.sqrt(variance[node]) * exogenous[:, node]
            used = self.n_features if node in self.nonlinear_nodes else 1
            for parent in _indices(parents[node]):
                values[:, node] += (
                    self.block(parent, values[:, parent], child=node)
                    @ weights[node, parent, :used])
        return values

