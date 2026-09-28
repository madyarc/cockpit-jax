"""Gradient-noise instruments. All of them come from one chunked pass over the
per-sample gradients, so nothing of size N x P is ever stored.

  sample_loss_fn(params, state, batch) -> (N,) loss of each sample

The mean gradient here is grad of mean_i l_i, which is the batch gradient for a mean
reduction. If your loss reduces by sum, the scaling of every quantity below changes.

Definitions used (stated so the numbers are reproducible, since conventions differ):
  g      = mean gradient, g_i = gradient of sample i
  Var_j  = unbiased variance over samples of component j, divisor N-1
  norm test radius       = sqrt( sum_i ||g_i - g||^2 / (N (N-1)) ) / ||g||
  inner product width    = sqrt( sum_i ( <g_i, g>/||g||^2 - 1 )^2 / (N (N-1)) )
  orthogonality width    = sqrt( sum_i ||g_i - (<g_i,g>/||g||^2) g||^2 / (N (N-1)) ) / ||g||
Each is compared against the user's threshold (theta, nu): the batch is large enough
for that test when the quantity is below the threshold. Sources: Byrd et al. 2012
(norm test), Bollapragada et al. 2018 (inner product test), Bollapragada et al. 2018 /
Cockpit (orthogonality test).
"""
import jax
import jax.numpy as jnp
import numpy as np


def _flat(tree):
    return jnp.concatenate([jnp.ravel(l) for l in jax.tree_util.tree_leaves(tree)])


def _flat_rows(trees):
    """(chunk, P) from a pytree whose leaves carry a leading sample axis."""
    return jnp.concatenate([l.reshape(l.shape[0], -1)
                            for l in jax.tree_util.tree_leaves(trees)], axis=1)


def grad_moments(sample_loss_fn, params, state, batch, chunk=256):
    """One chunked pass over per-sample gradients. Returns a dict of raw accumulators
    and the derived statistics:

      mean        (P,) the batch gradient, flattened
      var         (P,) unbiased per-component variance over samples
      trace_sigma scalar, sum of var: the trace of the gradient covariance
      norm_test, inner_test, ortho_test   as defined in the module docstring
      mean_gsnr   mean over components of g_j^2 / var_j
      n           number of samples

    Cost: one gradient pass over the batch in chunks of `chunk` samples; peak memory
    is chunk x P.
    """
    n = int(batch.shape[0])

    def single(p, x):
        return sample_loss_fn(p, state, x[None, :])[0]

    g_mean_tree = jax.grad(lambda p: jnp.mean(sample_loss_fn(p, state, batch)))(params)
    g_mean = _flat(g_mean_tree)
    g_norm2 = float(jnp.vdot(g_mean, g_mean))

    @jax.jit
    def acc(p, gm, xs, w):
        G = _flat_rows(jax.vmap(lambda x: jax.grad(single)(p, x))(xs))
        dot = G @ gm
        return (jnp.sum(G * w[:, None], axis=0),          # sum g_i
                jnp.sum(G ** 2 * w[:, None], axis=0),     # sum g_i^2
                jnp.sum(jnp.sum(G ** 2, axis=1) * w),     # sum ||g_i||^2
                jnp.sum(dot * w),                         # sum <g_i, g>
                jnp.sum(dot ** 2 * w))                    # sum <g_i, g>^2

    S_g = jnp.zeros_like(g_mean)
    S_g2 = jnp.zeros_like(g_mean)
    S_nrm = S_dot = S_dot2 = 0.0
    for i in range(0, n, chunk):
        xs = batch[i:i + chunk]
        k = xs.shape[0]
        if k < chunk:                                     # pad, then mask the padding out
            xs = jnp.concatenate([xs, jnp.repeat(xs[-1:], chunk - k, axis=0)], axis=0)
        w = jnp.concatenate([jnp.ones((k,), xs.dtype), jnp.zeros((chunk - k,), xs.dtype)])
        a, b, c, d, e = acc(params, g_mean, xs, w)
        S_g, S_g2 = S_g + a, S_g2 + b
        S_nrm, S_dot, S_dot2 = S_nrm + float(c), S_dot + float(d), S_dot2 + float(e)

    S_g2 = np.asarray(S_g2)
    g = np.asarray(g_mean)
    var = (S_g2 - n * g ** 2) / max(n - 1, 1)
    var = np.maximum(var, 0.0)
    scale = n * max(n - 1, 1)

    dev2 = max(S_nrm - n * g_norm2, 0.0)                  # sum_i ||g_i - g||^2
    ortho2 = max(S_nrm - S_dot2 / g_norm2, 0.0) if g_norm2 > 0 else float("nan")
    inner2 = max(S_dot2 - n * g_norm2 ** 2, 0.0) if g_norm2 > 0 else float("nan")

    return {
        "n": n,
        "mean": g,
        "var": var,
        "grad_norm": float(np.sqrt(g_norm2)),
        "trace_sigma": float(var.sum()),
        "norm_test": float(np.sqrt(dev2 / scale) / np.sqrt(g_norm2)) if g_norm2 > 0 else float("nan"),
        "inner_test": float(np.sqrt(inner2 / scale) / g_norm2) if g_norm2 > 0 else float("nan"),
        "ortho_test": float(np.sqrt(ortho2 / scale) / np.sqrt(g_norm2)) if g_norm2 > 0 else float("nan"),
        "mean_gsnr": float(np.mean(g ** 2 / (var + 1e-30))),
        "_sum_sq_norms": S_nrm,
        "_sum_dot": S_dot,
        "_sum_dot_sq": S_dot2,
    }


def cabs(moments, lr, loss):
    """CABS batch-size suggestion (Balles, Romero, Hennig 2017): lr * tr(Sigma) / loss.

    Read as the batch size that balances the gradient noise against the progress the
    step can make. It is a suggestion from one batch, not a schedule.
    """
    return float(lr) * moments["trace_sigma"] / float(loss)


def tic_diag(moments, hess_diag, eps=1e-8):
    """Diagonal TIC: sum_j Var_j / (H_jj + eps).

    An approximation of tr(H^-1 Sigma), the Takeuchi information criterion term, with
    both matrices taken diagonal. hess_diag: pytree from curvature.hessian_diagonal.
    """
    d = np.asarray(_flat(hess_diag))
    return float(np.sum(moments["var"] / (d + eps)))


def tic_trace(moments, hess_trace, eps=1e-8):
    """Trace TIC: P * tr(Sigma) / tr(H).

    The same criterion with H approximated by (tr(H)/P) I, which is cheaper than the
    diagonal version but coarser. hess_trace: the scalar from curvature.hessian_trace.
    """
    p = moments["mean"].size
    return float(p * moments["trace_sigma"] / (float(hess_trace) + eps))


def gradient_tests(moments, theta_norm=1.0, theta_inner=1.0, nu_ortho=1.0):
    """The three tests as pass/fail against thresholds, plus their raw values.

    A test passes when its quantity is below the threshold, i.e. the batch is large
    enough for that criterion. Thresholds are a choice, not a property of the run.
    """
    return {
        "norm_test": moments["norm_test"],
        "norm_test_pass": bool(moments["norm_test"] <= theta_norm),
        "inner_test": moments["inner_test"],
        "inner_test_pass": bool(moments["inner_test"] <= theta_inner),
        "ortho_test": moments["ortho_test"],
        "ortho_test_pass": bool(moments["ortho_test"] <= nu_ortho),
    }
