"""Edits for counterfactual branches. All return new objects; inputs are not mutated.

State edits use state.replace(...), which works for Flax TrainState and any
flax.struct dataclass. set_dict_entry and set_field take the field name, so they
work whatever the state calls its weights or its auxiliary parameter copies.

Changing the learning rate means passing a different step_fn to Replay.branch, since
the optimizer lives inside the step. freeze_mask below builds the mask for
optax.masked, which is the piece you need to freeze part of the model in a branch.
"""
import jax
import jax.numpy as jnp
import numpy as np


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------

def keep_rows(batch, mask):
    """Rows of batch where mask is True. Changes N (triggers a recompile of jitted steps)."""
    return batch[np.asarray(mask, dtype=bool)]


def drop_top(batch, scores, frac):
    """Drop the fraction `frac` of rows with the largest scores. Returns (batch, dropped_idx)."""
    scores = np.asarray(scores)
    n_drop = int(round(frac * scores.shape[0]))
    idx = np.argsort(scores)[::-1][:n_drop]
    mask = np.ones(scores.shape[0], dtype=bool)
    mask[idx] = False
    return batch[mask], idx


def replace_top(batch, scores, frac, key, dom, sort_axis=None):
    """Replace the top-`frac` rows by uniform draws in the box dom (array (d, 2)).

    Keeps N fixed, so no recompile. sort_axis re-sorts the rows by that column
    afterwards, for loops whose batches are expected sorted."""
    scores = np.asarray(scores)
    n_rep = int(round(frac * scores.shape[0]))
    idx = np.argsort(scores)[::-1][:n_rep]
    dom = jnp.asarray(dom)
    new = jax.random.uniform(key, (n_rep, batch.shape[1]), minval=dom[:, 0], maxval=dom[:, 1],
                             dtype=batch.dtype)
    out = batch.at[jnp.asarray(idx)].set(new)
    if sort_axis is not None:
        out = out[jnp.argsort(out[:, sort_axis])]
    return out, idx


def restrict_region(batch, col, lo, hi, keep_inside=True):
    """Keep (or drop) rows with lo <= batch[:, col] <= hi."""
    b = np.asarray(batch[:, col])
    inside = (b >= lo) & (b <= hi)
    return keep_rows(batch, inside if keep_inside else ~inside)


# ---------------------------------------------------------------------------
# parameters
# ---------------------------------------------------------------------------

def map_leaves(params, pattern, fn):
    """Apply fn to every leaf whose path string contains `pattern`. Returns (params, hit_paths)."""
    hits = []

    def f(path, leaf):
        p = jax.tree_util.keystr(path)
        if pattern in p:
            hits.append(p)
            return fn(leaf)
        return leaf
    new = jax.tree_util.tree_map_with_path(f, params)
    return new, hits


def zero_leaves(params, pattern):
    return map_leaves(params, pattern, jnp.zeros_like)


def scale_leaves(params, pattern, alpha):
    return map_leaves(params, pattern, lambda l: alpha * l)


def perturb(params, key, eps, direction=None):
    """params + eps * direction; direction defaults to a unit-norm Gaussian pytree."""
    from .tree import random_like, tree_norm, tree_axpy, tree_scale
    if direction is None:
        direction = random_like(key, params)
        direction = tree_scale(1.0 / tree_norm(direction), direction)
    return tree_axpy(eps, direction, params)


# ---------------------------------------------------------------------------
# state
# ---------------------------------------------------------------------------

def set_params(state, params):
    return state.replace(params=params)


def set_dict_entry(state, field, key, value):
    """state.<field>[key] = value, e.g. field='loss_weights', key='res'."""
    d = dict(getattr(state, field))
    if key not in d:
        raise KeyError("{} not in state.{} (keys: {})".format(key, field, list(d)))
    d[key] = jnp.asarray(value, dtype=jnp.asarray(d[key]).dtype)
    return state.replace(**{field: d})


def set_loss_weight(state, key, value, field="loss_weights"):
    return set_dict_entry(state, field, key, value)


def set_field(state, field, value):
    """Replace any single field of the state, e.g. a second parameter copy used as a
    target, teacher or anchor."""
    return state.replace(**{field: value})


def freeze_mask(params, pattern, freeze=True):
    """Boolean mask pytree for optax.masked: True where the leaf is TRAINED.

    freeze=True marks leaves whose path contains `pattern` as frozen (False in the
    mask); freeze=False inverts it, training only those leaves. Use it when building
    the branch's optimizer:

        mask = freeze_mask(state.params, "layers_0")
        tx = optax.masked(optax.adam(lr), mask)

    The optimizer state of a masked tx differs from the unmasked one, so the branch
    needs its own tx.init(params) rather than the state's existing opt_state.
    """
    def f(path, leaf):
        hit = pattern in jax.tree_util.keystr(path)
        return (not hit) if freeze else hit
    return jax.tree_util.tree_map_with_path(f, params)
