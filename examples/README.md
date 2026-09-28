# Example

`replay_mlp.ipynb` -- a self-contained notebook, no data files and nothing beyond JAX,
optax, Orbax and matplotlib.

A small MLP fits a 2D function under a two-term loss (a fit term and a
first-derivative penalty, standing in for a data term and a residual term). The
learning rate ramps up until the run blows up. Then:

1. **Record.** A `FlightRecorder` keeps the last six states, one every 50 steps, and
   dumps them when the loss jumps over ten times its recent median, or the gradient
   norm over five times its own. The sampler key goes into the bundle's `host` dict, so
   the replay draws the same batches.
2. **Replay.** Load the earliest buffered state and step it forward with probes
   attached: gradient norm, update ratio, per-term gradient norms and their cosine,
   curvature along the step, alpha, distance travelled, the three gradient-noise tests
   and the top eigenvalue.
3. **Inspect one step.** Top Hessian eigenvalue, the preconditioned one against 2/lr,
   the top five by LOBPCG, the full eigenvalue density by Lanczos, the gradient tests
   with CABS and both TIC variants, per-sample gradient and loss maps over the inputs,
   per-leaf gradient bars, NaN location, and the loss along the update taken.
4. **Branch.** From one state: a tenth of the learning rate, the second loss term
   switched off, and the worst 5% of samples dropped -- all on identical batches,
   overlaid on one plot, then a leaf-by-leaf diff of two branches' parameters.

## Adapting it to your own run

Two things have to be yours:

- `step_fn(state, host, batch) -> (state, host, metrics)`: one iteration of your outer
  loop, not just the optimizer update. If the loop also updates loss weights, refreshes
  a second parameter copy, or advances a Python-side controller, all of it belongs
  inside `step_fn`.
- `host`: everything the step depends on that is not inside the train state. A mutable
  controller object should be snapshotted into `host` at the end of the step and
  reloaded from it at the top, or branches will share one object and corrupt each
  other.

The state only has to be a pytree carrying the fields the tools touch: `params`,
`loss_weights` if you use `set_loss_weight`, and `opt_state` if you use `adam_pinv`.

The probe dict in the notebook includes the expensive ones (`probe_noise` costs a
per-sample gradient pass per step, `probe_sharpness` costs 15 HVPs). Drop those two
before replaying more than a few hundred steps, and call them by hand at the steps
that matter instead.
