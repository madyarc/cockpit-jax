"""Matrix-free curvature. Everything is built on Hessian-vector products, so the
Hessian is never formed.

  loss_fn(params, state, batch) -> scalar

Flattening: a pytree is flattened by concatenating its leaves in tree order, each
raveled. `flat_hvp` returns that flat operator together with the pack/unpack pair,
for the routines that need a plain (n,) vector (LOBPCG, Lanczos).
"""
import jax
import jax.numpy as jnp
import numpy as np

from .tree import tree_dot, tree_norm, tree_scale, random_like


# ---------------------------------------------------------------------------
# the operator
# ---------------------------------------------------------------------------

def make_hvp(loss_fn, params, state, batch):
    """v -> H v at `params`, forward-over-reverse, jit-compiled."""
    @jax.jit
    def _hvp(p, s, b, v):
        return jax.jvp(lambda q: jax.grad(loss_fn)(q, s, b), (p,), (v,))[1]
    return lambda v: _hvp(params, state, batch, v)


def flatten_fns(tree):
    """(flatten, unflatten, n) for the leaf-concatenation layout."""
    leaves, treedef = jax.tree_util.tree_flatten(tree)
    shapes = [l.shape for l in leaves]
    sizes = [int(np.prod(s)) if s else 1 for s in shapes]
    n = int(sum(sizes))

    def flatten(t):
        return jnp.concatenate([jnp.ravel(l) for l in jax.tree_util.tree_leaves(t)])

    def unflatten(v):
        out, i = [], 0
        for shape, size in zip(shapes, sizes):
            out.append(jnp.reshape(v[i:i + size], shape))
            i += size
        return jax.tree_util.tree_unflatten(treedef, out)
    return flatten, unflatten, n


def flat_hvp(loss_fn, params, state, batch):
    """(matvec, flatten, unflatten, n) with matvec acting on a flat (n,) vector."""
    hvp = make_hvp(loss_fn, params, state, batch)
    flatten, unflatten, n = flatten_fns(params)
    return (lambda v: flatten(hvp(unflatten(v)))), flatten, unflatten, n


# ---------------------------------------------------------------------------
# eigenvalues
# ---------------------------------------------------------------------------

def top_hessian_eig(loss_fn, params, state, batch, key, iters=30, tol=1e-4):
    """Power iteration on H. Returns (lam, v, history).

    Converges to the eigenvalue of largest MAGNITUDE; lam is the Rayleigh quotient
    v^T H v, so its sign is the sign of that eigenvalue. Compare it with 2/lr only
    for plain gradient descent -- see top_preconditioned_eig for Adam-type steps.
    """
    hvp = make_hvp(loss_fn, params, state, batch)
    v = random_like(key, params)
    v = tree_scale(1.0 / tree_norm(v), v)
    lam, hist = None, []
    for _ in range(iters):
        hv = hvp(v)
        new_lam = float(tree_dot(v, hv))
        hist.append(new_lam)
        nrm = tree_norm(hv)
        if float(nrm) == 0.0:
            break
        v = tree_scale(1.0 / nrm, hv)
        if lam is not None and abs(new_lam - lam) <= tol * max(abs(lam), 1e-30):
            lam = new_lam
            break
        lam = new_lam
    return lam, v, hist


def top_k_hessian_eigs(loss_fn, params, state, batch, key, k=5, iters=100, tol=None):
    """LOBPCG for the k algebraically largest eigenvalues. Returns (eigvals, eigvecs)
    with eigvecs a list of k pytrees.

    LOBPCG finds the largest ALGEBRAIC eigenvalues, not the largest in magnitude, so
    it will not report a large negative eigenvalue. For that, run it again on -H by
    passing a negated loss_fn and flipping the signs.
    """
    from jax.experimental.sparse.linalg import lobpcg_standard

    matvec, flatten, unflatten, n = flat_hvp(loss_fn, params, state, batch)
    A = jax.jit(lambda X: jax.vmap(matvec, in_axes=1, out_axes=1)(X))
    X0 = jax.random.normal(key, (n, k), dtype=flatten(params).dtype)
    theta, U, _ = lobpcg_standard(A, X0, m=iters, tol=tol)
    return np.asarray(theta), [unflatten(U[:, i]) for i in range(U.shape[1])]


def top_preconditioned_eig(loss_fn, params, state, batch, apply_pinv, key,
                           iters=30, tol=1e-4):
    """Largest generalized eigenvalue of H v = lam P v, by power iteration on P^-1 H.

    `apply_pinv(v) -> P^-1 v` as a pytree map. For an Adam-type optimizer this is the
    quantity that the 2/lr stability threshold is about, not the raw Hessian
    eigenvalue. Returns (lam, v, history), lam = (v^T H v) / (v^T P v).
    """
    hvp = make_hvp(loss_fn, params, state, batch)
    v = random_like(key, params)
    v = tree_scale(1.0 / tree_norm(v), v)
    lam, hist = None, []
    for _ in range(iters):
        hv = hvp(v)
        pv_inv = apply_pinv(v)                       # P^-1 v, for the denominator
        den = float(tree_dot(v, v)) / max(float(tree_dot(v, pv_inv)), 1e-300)
        new_lam = float(tree_dot(v, hv)) / den if den != 0 else float("nan")
        hist.append(new_lam)
        w = apply_pinv(hv)
        nrm = tree_norm(w)
        if float(nrm) == 0.0:
            break
        v = tree_scale(1.0 / nrm, w)
        if lam is not None and abs(new_lam - lam) <= tol * max(abs(lam), 1e-30):
            lam = new_lam
            break
        lam = new_lam
    return lam, v, hist


def adam_pinv(opt_state, eps=1e-8):
    """apply_pinv for an optax Adam-type state: P = diag(sqrt(nu) + eps).

    Finds the first node in opt_state with a `nu` field (ScaleByAdamState and its
    relatives). Bias correction is NOT applied, so early in training this understates
    the preconditioner. Raises if no such node exists.
    """
    nu = None
    for node in jax.tree_util.tree_leaves(opt_state, is_leaf=lambda x: hasattr(x, "nu")):
        if hasattr(node, "nu"):
            nu = node.nu
            break
    if nu is None:
        raise ValueError("no optimizer state node with a 'nu' field; pass apply_pinv yourself")
    return lambda v: jax.tree_util.tree_map(lambda x, n: x / (jnp.sqrt(n) + eps), v, nu)


# ---------------------------------------------------------------------------
# trace and diagonal
# ---------------------------------------------------------------------------

def hessian_trace(loss_fn, params, state, batch, key, n_probes=10):
    """Hutchinson: mean over Rademacher z of z^T H z. Returns (mean, standard error)."""
    hvp = make_hvp(loss_fn, params, state, batch)
    vals = []
    for k in jax.random.split(key, n_probes):
        z = random_like(k, params, "rademacher")
        vals.append(float(tree_dot(z, hvp(z))))
    vals = np.asarray(vals)
    se = vals.std(ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else float("nan")
    return float(vals.mean()), float(se)


def hessian_diagonal(loss_fn, params, state, batch, key, n_probes=20):
    """Hutchinson diagonal: mean over Rademacher z of z * (H z), elementwise.

    Returns a pytree shaped like params. This is an estimator, not the exact diagonal;
    its error falls as 1/sqrt(n_probes).
    """
    hvp = make_hvp(loss_fn, params, state, batch)
    acc = jax.tree_util.tree_map(jnp.zeros_like, params)
    for k in jax.random.split(key, n_probes):
        z = random_like(k, params, "rademacher")
        hz = hvp(z)
        acc = jax.tree_util.tree_map(lambda a, zz, h: a + zz * h, acc, z, hz)
    return jax.tree_util.tree_map(lambda a: a / n_probes, acc)


def curvature_along(loss_fn, params, state, batch, direction):
    """d^T H d / d^T d: curvature along one direction, e.g. the last update."""
    hvp = make_hvp(loss_fn, params, state, batch)
    dd = float(tree_dot(direction, direction))
    return float(tree_dot(direction, hvp(direction))) / dd if dd > 0 else float("nan")


# ---------------------------------------------------------------------------
# spectral density (stochastic Lanczos quadrature)
# ---------------------------------------------------------------------------

def lanczos_tridiag(matvec, n, key, m=40, dtype=jnp.float64):
    """m steps of Lanczos with full reorthogonalisation. Returns (alphas, betas, V).

    Full reorthogonalisation costs m vectors of memory and m^2/2 dot products, and is
    what keeps the Ritz values from duplicating in floating point.
    """
    v = jax.random.normal(key, (n,), dtype=dtype)
    v = v / jnp.linalg.norm(v)
    V, alphas, betas = [v], [], []
    w = matvec(v)
    a = float(jnp.vdot(v, w))
    alphas.append(a)
    w = w - a * v
    for _ in range(m - 1):
        for u in V:                                   # full reorthogonalisation
            w = w - jnp.vdot(u, w) * u
        b = float(jnp.linalg.norm(w))
        if b < 1e-12:
            break
        betas.append(b)
        v = w / b
        V.append(v)
        w = matvec(v)
        a = float(jnp.vdot(v, w))
        alphas.append(a)
        w = w - a * v - b * V[-2]
    return np.asarray(alphas), np.asarray(betas), V


def spectral_density(loss_fn, params, state, batch, key, n_probes=8, m=40):
    """Stochastic Lanczos quadrature (Ghorbani, Krishnan, Xiao 2019).

    Returns (nodes, weights), each (n_probes, m'): the Ritz values of the Lanczos
    tridiagonal matrix and the squared first components of its eigenvectors. Feed them
    to density_curve for a plottable eigenvalue density, including the negative part
    that a single top eigenvalue hides.
    """
    matvec, flatten, _, n = flat_hvp(loss_fn, params, state, batch)
    dtype = flatten(params).dtype
    nodes, weights = [], []
    for k in jax.random.split(key, n_probes):
        alphas, betas, _ = lanczos_tridiag(matvec, n, k, m=m, dtype=dtype)
        T = np.diag(alphas) + np.diag(betas, 1) + np.diag(betas, -1)
        evals, evecs = np.linalg.eigh(T)
        nodes.append(evals)
        weights.append(evecs[0, :] ** 2)
    return nodes, weights


def density_curve(nodes, weights, grid=None, sigma=None, n_grid=1024):
    """Gaussian-smoothed density from spectral_density output. Returns (grid, density)."""
    all_nodes = np.concatenate([np.asarray(x) for x in nodes])
    if grid is None:
        lo, hi = all_nodes.min(), all_nodes.max()
        pad = 0.1 * (hi - lo + 1e-12)
        grid = np.linspace(lo - pad, hi + pad, n_grid)
    if sigma is None:
        sigma = max((grid[-1] - grid[0]) / 100.0, 1e-12)
    dens = np.zeros_like(grid)
    for nd, wt in zip(nodes, weights):
        nd, wt = np.asarray(nd), np.asarray(wt)
        dens += (wt[None, :] * np.exp(-0.5 * ((grid[:, None] - nd[None, :]) / sigma) ** 2)
                 ).sum(axis=1)
    dens /= (len(nodes) * sigma * np.sqrt(2.0 * np.pi))
    return grid, dens
