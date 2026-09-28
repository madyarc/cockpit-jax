"""Pytree helpers. All functions take any JAX pytree (e.g. a Flax params dict)."""
import jax
import jax.numpy as jnp


def leaves_with_paths(tree):
    """[(path_string, leaf), ...] in flatten order."""
    flat, _ = jax.tree_util.tree_flatten_with_path(tree)
    return [(jax.tree_util.keystr(p), l) for p, l in flat]


def tree_norm(tree):
    """Global L2 norm over all leaves."""
    return jnp.sqrt(sum(jnp.sum(jnp.square(l)) for l in jax.tree_util.tree_leaves(tree)))


def tree_dot(a, b):
    """Sum over leaves of <a_leaf, b_leaf>."""
    la, lb = jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b)
    return sum(jnp.sum(x * y) for x, y in zip(la, lb))


def tree_axpy(alpha, x, y):
    """alpha * x + y, leafwise."""
    return jax.tree_util.tree_map(lambda a, b: alpha * a + b, x, y)


def tree_scale(alpha, x):
    return jax.tree_util.tree_map(lambda a: alpha * a, x)


def tree_sub(a, b):
    return jax.tree_util.tree_map(lambda x, y: x - y, a, b)


def leaf_norms(tree):
    """{path: ||leaf||_2} as Python floats."""
    return {p: float(jnp.linalg.norm(jnp.ravel(l))) for p, l in leaves_with_paths(tree)}


def leaf_stats(tree):
    """{path: {norm, rms, absmax, mean}} as Python floats."""
    out = {}
    for p, l in leaves_with_paths(tree):
        v = jnp.ravel(l)
        out[p] = {
            "norm": float(jnp.linalg.norm(v)),
            "rms": float(jnp.sqrt(jnp.mean(v ** 2))),
            "absmax": float(jnp.max(jnp.abs(v))),
            "mean": float(jnp.mean(v)),
        }
    return out


def nonfinite_leaves(tree):
    """Paths of leaves that contain any NaN or inf, with the count of bad entries."""
    bad = {}
    for p, l in leaves_with_paths(tree):
        n = int(jnp.sum(~jnp.isfinite(l)))
        if n > 0:
            bad[p] = n
    return bad


def random_like(key, tree, kind="normal"):
    """Random pytree with the structure of `tree`. kind: 'normal' or 'rademacher'."""
    leaves, treedef = jax.tree_util.tree_flatten(tree)
    keys = jax.random.split(key, len(leaves))
    if kind == "normal":
        new = [jax.random.normal(k, l.shape, l.dtype) for k, l in zip(keys, leaves)]
    elif kind == "rademacher":
        new = [jax.random.rademacher(k, l.shape).astype(l.dtype) for k, l in zip(keys, leaves)]
    else:
        raise ValueError(kind)
    return jax.tree_util.tree_unflatten(treedef, new)


def diff_leaves(a, b, rel=True):
    """{path: ||a_leaf - b_leaf||} between two pytrees of the same structure.

    rel=True also divides by ||b_leaf||, giving {path: (abs, rel)}. Use it to see
    which parts of two checkpoints actually differ.
    """
    out = {}
    pa = leaves_with_paths(a)
    pb = dict(leaves_with_paths(b))
    for path, la in pa:
        lb = pb.get(path)
        if lb is None or la.shape != lb.shape:
            out[path] = ("missing or shape mismatch", float("nan"))
            continue
        d = float(jnp.linalg.norm(jnp.ravel(la - lb)))
        n = float(jnp.linalg.norm(jnp.ravel(lb)))
        out[path] = (d, d / n if (rel and n > 0) else float("nan"))
    return out
