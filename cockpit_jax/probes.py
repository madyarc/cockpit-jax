"""Diagnostics. Every probe is a plain function; nothing is stored.

Signatures used throughout
  loss_fn(params, state, batch)        -> scalar total loss (as the optimizer sees it)
  terms_fn(params, state, batch)       -> {term: scalar}   (unweighted loss terms)
  sample_fn(params, state, batch)      -> {name: (N,) value per sample}
  sample_loss_fn(params, state, batch) -> (N,) loss of each sample

Curvature lives in curvature.py and gradient noise in noise.py; both are re-exported
here, so `from cockpit_jax import probes as P` reaches everything.

Probe factories at the bottom return fn(prev_state, state, batch) -> dict, for
Replay(probes=...). Heavy probes (sharpness, spectral density, per-sample gradients)
are meant to be called by hand at the steps of interest.
"""
import jax
import jax.numpy as jnp
import numpy as np

from .tree import (tree_norm, tree_dot, tree_axpy, tree_scale, tree_sub,
                   leaf_norms, random_like, nonfinite_leaves)
from .curvature import (make_hvp, flat_hvp, flatten_fns, top_hessian_eig,
                        top_k_hessian_eigs, top_preconditioned_eig, adam_pinv,
                        hessian_trace, hessian_diagonal, curvature_along,
                        spectral_density, density_curve, lanczos_tridiag)
from .noise import grad_moments, gradient_tests, cabs, tic_diag, tic_trace


# ---------------------------------------------------------------------------
# gradients
# ---------------------------------------------------------------------------

def grad(loss_fn, params, state, batch):
    return jax.grad(loss_fn)(params, state, batch)


def layer_grad_norms(loss_fn, params, state, batch):
    """{leaf_path: ||dL/d leaf||}."""
    return leaf_norms(grad(loss_fn, params, state, batch))


def term_grads(terms_fn, params, state, batch):
    """{term: gradient pytree} of each unweighted loss term."""
    return jax.jacrev(terms_fn)(params, state, batch)


def term_grad_report(terms_fn, params, state, batch):
    """Per-term loss and gradient norm, and the cosine between every pair of term gradients.

    Negative cosine = the terms pull the parameters in opposing directions.
    """
    vals = terms_fn(params, state, batch)
    g = term_grads(terms_fn, params, state, batch)
    names = list(g.keys())
    out = {}
    norms = {n: float(tree_norm(g[n])) for n in names}
    for n in names:
        out["term/" + n] = float(vals[n])
        out["gnorm/" + n] = norms[n]
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            den = norms[a] * norms[b]
            out["gcos/{}|{}".format(a, b)] = float(tree_dot(g[a], g[b])) / den if den > 0 else float("nan")
    return out


# ---------------------------------------------------------------------------
# loss landscape
# ---------------------------------------------------------------------------

def loss_slice(loss_fn, params, state, batch, direction, alphas):
    """L(params + a * direction) for each a. Returns np.array of losses."""
    f = jax.jit(lambda a, p, d, s, b: loss_fn(tree_axpy(a, d, p), s, b))
    return np.array([float(f(a, params, direction, state, batch)) for a in alphas])


def loss_plane(loss_fn, params, state, batch, d1, d2, a1, a2):
    """L(params + a*d1 + b*d2) on the grid a1 x a2. Returns array (len(a1), len(a2))."""
    f = jax.jit(lambda a, c, p, u, v, s, b: loss_fn(tree_axpy(c, v, tree_axpy(a, u, p)), s, b))
    return np.array([[float(f(a, c, params, d1, d2, state, batch)) for c in a2] for a in a1])


# ---------------------------------------------------------------------------
# per sample
# ---------------------------------------------------------------------------

def per_sample_values(sample_fn, params, state, batch):
    """{name: np.array (N,)} -- whatever the model reports per sample."""
    return {k: np.asarray(v) for k, v in sample_fn(params, state, batch).items()}


def per_sample_grads(sample_loss_fn, params, state, batch, chunk=256):
    """Gradient of each sample's loss, summarised without storing N x P.

    Returns dict of np.arrays (N,):
      gnorm  : ||grad l_i||
      align  : <grad l_i, g> / ||g||^2, g = grad of mean_i l_i.
               Sums to 1 over the samples (up to rounding): each entry is that
               sample's share of the mean gradient, projected on g. Negative =
               the sample pulls against the mean gradient.
    """
    def single(p, x):
        return sample_loss_fn(p, state, x[None, :])[0]

    g_all = jax.grad(lambda p: jnp.mean(sample_loss_fn(p, state, batch)))(params)
    gg = float(tree_dot(g_all, g_all))

    @jax.jit
    def _chunk_stats(p, g_ref, xs):
        gs = jax.vmap(lambda x: jax.grad(single)(p, x))(xs)
        leaves_g = jax.tree_util.tree_leaves(gs)
        leaves_a = jax.tree_util.tree_leaves(g_ref)
        sq = sum(jnp.sum(l.reshape(l.shape[0], -1) ** 2, axis=1) for l in leaves_g)
        dot = sum(jnp.sum(l.reshape(l.shape[0], -1) * a.reshape(1, -1), axis=1)
                  for l, a in zip(leaves_g, leaves_a))
        return jnp.sqrt(sq), dot

    chunk_stats = lambda xs: _chunk_stats(params, g_all, xs)

    n = batch.shape[0]
    norms, dots = [], []
    for i in range(0, n, chunk):
        xs = batch[i:i + chunk]
        if xs.shape[0] < chunk:                       # pad to keep one compiled shape
            pad = jnp.repeat(xs[-1:], chunk - xs.shape[0], axis=0)
            nr, dt = chunk_stats(jnp.concatenate([xs, pad], axis=0))
            nr, dt = nr[:xs.shape[0]], dt[:xs.shape[0]]
        else:
            nr, dt = chunk_stats(xs)
        norms.append(np.asarray(nr))
        dots.append(np.asarray(dt))
    norms, dots = np.concatenate(norms), np.concatenate(dots)
    align = dots / (n * gg) if gg > 0 else np.full(n, np.nan)
    return {"gnorm": norms, "align": align}


# ---------------------------------------------------------------------------
# NaN / inf location
# ---------------------------------------------------------------------------

def locate_nan(fn, *args):
    """Run fn(*args) with jax_debug_nans on. Returns None if clean, else the error text,
    which names the primitive that first produced a NaN."""
    try:
        with jax.debug_nans(True):
            out = fn(*args)
            jax.block_until_ready(out)
    except FloatingPointError as e:
        return str(e)
    return None


def nonfinite_report(state):
    """{leaf_path: bad_count} over the whole state (params, opt_state, teacher, ...)."""
    return nonfinite_leaves(state)


# ---------------------------------------------------------------------------
# step quality (alpha)
# ---------------------------------------------------------------------------

def fit_cubic(f0, df0, f1, df1):
    """Hermite cubic through (0, f0, df0) and (1, f1, df1). Returns (a, b, c, e) of
    f(t) = a t^3 + b t^2 + c t + e."""
    e, c = f0, df0
    a = 2.0 * (f0 - f1) + (df0 + df1)
    b = 3.0 * (f1 - f0) - (2.0 * df0 + df1)
    return a, b, c, e


def _cubic_min(a, b, c):
    """Position of the local minimum of a t^3 + b t^2 + c t, or None."""
    if abs(a) < 1e-14:
        return (-c / (2.0 * b)) if b > 0 else None
    disc = 4.0 * b * b - 12.0 * a * c
    if disc < 0:
        return None
    r = np.sqrt(disc)
    for t in ((-2.0 * b + r) / (6.0 * a), (-2.0 * b - r) / (6.0 * a)):
        if 6.0 * a * t + 2.0 * b > 0:                 # the one that is a minimum
            return float(t)
    return None


def alpha(loss_fn, prev_state, state, batch, params_of=lambda s: s.params):
    """Step quality along the update just taken. Returns a dict.

    A cubic is fitted along the step from the loss and the directional derivative at
    both endpoints, with t = 0 at the pre-step parameters and t = 1 where the step
    landed. Both evaluations use prev_state, so the objective is the same at both ends.

      alpha = -1  the step went nowhere (it ended back at the starting loss level)
      alpha =  0  it landed at the minimum of the fit
      alpha = +1  it overshot to the point on the far side with the starting loss
      alpha > 1   it overshot past that point; alpha in (-1, 0) it undershot

    This is the noise-free version: Cockpit fits with the gradient variances as
    weights, which this does not, so the two will not agree exactly.
    """
    p0, p1 = params_of(prev_state), params_of(state)
    d = tree_sub(p1, p0)
    f0 = float(loss_fn(p0, prev_state, batch))
    f1 = float(loss_fn(p1, prev_state, batch))
    df0 = float(tree_dot(grad(loss_fn, p0, prev_state, batch), d))
    df1 = float(tree_dot(grad(loss_fn, p1, prev_state, batch), d))
    a, b, c, _ = fit_cubic(f0, df0, f1, df1)
    t_min = _cubic_min(a, b, c)
    out = {"alpha/f0": f0, "alpha/f1": f1, "alpha/df0": df0, "alpha/df1": df1,
           "alpha/t_min": float("nan") if t_min is None else t_min}
    if t_min is None or t_min <= 0:
        out["alpha"] = float("nan")
        return out
    if 1.0 <= t_min:
        out["alpha"] = (1.0 - t_min) / t_min           # undershoot, -1 at t = 0
        return out
    # the far side: the other root of f(t) = f0, i.e. of a t^2 + b t + c = 0
    if abs(a) < 1e-14:
        t_plus = (-c / b) if b != 0 else None
    else:
        disc = b * b - 4.0 * a * c
        if disc < 0:
            t_plus = None
        else:
            r = np.sqrt(disc)
            roots = [(-b + r) / (2.0 * a), (-b - r) / (2.0 * a)]
            roots = [t for t in roots if t > t_min]
            t_plus = min(roots) if roots else None
    if t_plus is None or t_plus <= t_min:
        out["alpha"] = float("nan")
    else:
        out["alpha"] = (1.0 - t_min) / (t_plus - t_min)
    return out


# ---------------------------------------------------------------------------
# probe factories for Replay(probes=...)
#
# Cost per step, on top of the step itself, since a probe set that doubles the
# step time is what makes a sweep unaffordable. Replay(probe_every=n) runs the
# whole set every n steps instead, which is the knob for the expensive ones.
#
#   probe_update      nothing beyond two norms
#   probe_distance    one norm
#   probe_grad        1 gradient
#   probe_terms       1 gradient PER LOSS TERM (jacrev over the term dict)
#   probe_term_values the terms only, no gradients: 1 forward pass
#   probe_layers      1 gradient, plus one column per leaf in the record
#   probe_step_curv   1 HVP
#   probe_alpha       2 losses + 2 gradients
#   probe_sharpness   `iters` HVPs (20 by default)
#   probe_noise       a full per-sample gradient pass, chunked
# ---------------------------------------------------------------------------

def probe_update(params_of=lambda s: s.params):
    """||d theta||, ||theta||, and their ratio for the step just taken."""
    def probe(prev_state, state, batch):
        p0, p1 = params_of(prev_state), params_of(state)
        d = tree_norm(tree_sub(p1, p0))
        n = tree_norm(p0)
        return {"upd/norm": d, "param/norm": n, "upd/ratio": d / n}
    return probe


def probe_grad(loss_fn, params_of=lambda s: s.params):
    """||grad L|| at the pre-step parameters, on the step's batch."""
    def probe(prev_state, state, batch):
        return {"grad/norm": tree_norm(grad(loss_fn, params_of(prev_state), prev_state, batch))}
    return probe


def probe_terms(terms_fn, params_of=lambda s: s.params):
    """Per-term loss, per-term gradient norm, and the cosine between every pair.

    One gradient per loss term. On a sweep, either raise Replay(probe_every=...) or
    use probe_term_values and keep this for the zoom.
    """
    def probe(prev_state, state, batch):
        return term_grad_report(terms_fn, params_of(prev_state), prev_state, batch)
    return probe


def probe_term_values(terms_fn, params_of=lambda s: s.params):
    """The unweighted loss terms only, no gradients: one forward pass."""
    def probe(prev_state, state, batch):
        vals = terms_fn(params_of(prev_state), prev_state, batch)
        return {"term/" + k: v for k, v in vals.items()}
    return probe


def probe_layers(loss_fn, params_of=lambda s: s.params):
    """Per-leaf gradient norms at the pre-step parameters (one column per leaf)."""
    def probe(prev_state, state, batch):
        return {"lgrad/" + k: v for k, v in
                layer_grad_norms(loss_fn, params_of(prev_state), prev_state, batch).items()}
    return probe


def probe_step_curvature(loss_fn, params_of=lambda s: s.params):
    """Curvature of L along the update just taken, at the pre-step parameters."""
    def probe(prev_state, state, batch):
        p0 = params_of(prev_state)
        d = tree_sub(params_of(state), p0)
        return {"curv/along_step": curvature_along(loss_fn, p0, prev_state, batch, d)}
    return probe


def probe_alpha(loss_fn, params_of=lambda s: s.params, full=False):
    """Step quality along the update just taken. full=True also records the fitted
    endpoint values, which is how you tell WHICH degenerate case produced a NaN.

    alpha is NaN whenever the fit has no minimum ahead of the start: the update was
    not a descent direction on this batch (df0 >= 0), the cubic is monotone, or there
    is no equal-loss point on the far side. That is an observation about the step, not
    an error -- Replay records it and carries on, since only `stop_keys` halts a run.
    Preconditioned and averaged optimizers produce it regularly.
    """
    def probe(prev_state, state, batch):
        out = alpha(loss_fn, prev_state, state, batch, params_of)
        return out if full else {"alpha": out["alpha"]}
    return probe


def probe_distance(params_ref, params_of=lambda s: s.params):
    """||theta - theta_ref||: how far the parameters have travelled from a reference,
    usually the state the replay started from."""
    def probe(prev_state, state, batch):
        return {"dist/from_ref": tree_norm(tree_sub(params_of(state), params_ref))}
    return probe


def probe_noise(sample_loss_fn, chunk=256, params_of=lambda s: s.params):
    """The gradient-noise instruments at the pre-step parameters: the three gradient
    tests, the gradient-covariance trace and the mean GSNR.

    Costs a full per-sample gradient pass every step. Use it on a short replay window,
    not on a long one.
    """
    def probe(prev_state, state, batch):
        m = grad_moments(sample_loss_fn, params_of(prev_state), prev_state, batch, chunk)
        return {"noise/norm_test": m["norm_test"], "noise/inner_test": m["inner_test"],
                "noise/ortho_test": m["ortho_test"], "noise/trace_sigma": m["trace_sigma"],
                "noise/mean_gsnr": m["mean_gsnr"]}
    return probe


def probe_sharpness(loss_fn, key, iters=20, params_of=lambda s: s.params, apply_pinv_of=None):
    """Top Hessian eigenvalue at the pre-step parameters, by power iteration.

    apply_pinv_of(state) -> apply_pinv turns this into the preconditioned eigenvalue,
    e.g. lambda s: adam_pinv(s.opt_state). Expensive: `iters` HVPs per step.
    """
    def probe(prev_state, state, batch):
        p0 = params_of(prev_state)
        if apply_pinv_of is None:
            lam, _, _ = top_hessian_eig(loss_fn, p0, prev_state, batch, key, iters=iters)
            return {"curv/top_eig": lam}
        lam, _, _ = top_preconditioned_eig(loss_fn, p0, prev_state, batch,
                                           apply_pinv_of(prev_state), key, iters=iters)
        return {"curv/top_eig_precond": lam}
    return probe
