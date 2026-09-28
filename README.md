# cockpit-jax

An **unofficial, partial JAX implementation of Cockpit**, plus a replay layer that
Cockpit does not have: reload a checkpoint from just before a run diverged and step
it forward by hand, branch it under different interventions, and measure what
happens, instead of restarting the whole run.

Cockpit is by Frank Schneider, Felix Dangel and Philipp Hennig (NeurIPS 2021). The
original is PyTorch-only and built on BackPACK; this repository is an independent
implementation in JAX, not a translation of their code. It is not affiliated with or
endorsed by the Cockpit authors. Paper and original code are linked in
[Citing](#citing) below.

It is library-agnostic: everything takes plain callables and a pytree state, so it
works with any JAX training loop, whatever builds the model.

## Why

A run diverges at step 43,000. The usual answer is to change something and retrain
from zero. Instead: keep the last few full states in a ring buffer, dump them when the
loss spikes, then load the state from just before the spike and step forward one step
at a time with the expensive diagnostics switched on -- sharpness, per-term gradients,
per-sample maps -- and branch the same state under different fixes on identical
batches.

## What is here

| Module | Contents |
| --- | --- |
| `recorder.py` | `FlightRecorder`: ring buffer of full states, dumped on a NaN or a loss spike |
| `ckpt.py` | bundles: train state (Orbax) + host-side extras (pickle); loading an existing Orbax checkpoint |
| `replay.py` | `Replay`: step, `rewind`, `branch`, `zoom`, `save`/`load`; branches share one batch cache, so they see identical data |
| `store.py` | records, batches and metadata on disk, behind `Replay.save` and `.load` |
| `sweep.py` | run a set of cases from one start on the same batches, saving each as it finishes |
| `probes.py` | gradients per layer and per loss term, term-gradient cosines, alpha (step quality), 1D/2D loss slices, per-sample values and gradients, NaN location, distance travelled, and the probe factories; re-exports the two below |
| `curvature.py` | HVP, top eigenvalue by power iteration, top-k by LOBPCG, top preconditioned eigenvalue, Hutchinson trace and diagonal, eigenvalue spectral density by stochastic Lanczos quadrature |
| `noise.py` | per-sample gradient moments in one chunked pass, then the norm / inner-product / orthogonality tests, mean GSNR, CABS and TIC |
| `interventions.py` | drop or resample the worst samples, restrict a region of the input, zero or scale leaves, perturb, set a loss weight or any other state field |
| `plots.py` | trajectories, branch overlays, per-sample scatter over two input columns, per-leaf bars and histograms, slices and contours |
| `tree.py` | pytree norms, dots, per-leaf stats, non-finite search, random pytrees |

### Against Cockpit's instrument list

| Cockpit instrument | Here |
| --- | --- |
| Loss, GradNorm, UpdateSize, Distance | `probe_grad`, `probe_update`, `probe_distance` |
| Alpha | `probes.alpha`, `probe_alpha` -- noise-free variant, see below |
| NormTest, InnerTest, OrthoTest | `noise.grad_moments`, `noise.gradient_tests` |
| MeanGSNR | `grad_moments()["mean_gsnr"]` |
| CABS | `noise.cabs` |
| TICDiag, TICTrace | `noise.tic_diag`, `noise.tic_trace` |
| Trace, MaxEV | `curvature.hessian_trace`, `curvature.top_hessian_eig` |
| HistogramParam, HistogramGrad | `plots.plot_leaf_hists`, `plots.plot_values_hist` |
| Cockpit's live dashboard | **not here.** Plots are drawn afterwards from the records |
| EarlyStopping (the evidence-based criterion) | **not here** |
| GradHist2d | **not here** |

Two differences worth stating rather than burying:

- **Alpha is the noise-free version.** Cockpit fits the polynomial with the gradient
  variances as weights; this fits a Hermite cubic through the exact loss and
  directional derivative at both ends of the step. Same normalisation (-1 at the
  start, 0 at the fitted minimum, +1 at the equal-loss point on the far side), but the
  numbers will not match Cockpit's exactly.
- **The variance conventions are spelled out** in the `noise.py` docstring, because
  the divisor and the normalisation differ between papers.

Added, with no Cockpit counterpart: replay from a checkpoint, branching under
interventions, top-k eigenvalues and the full spectral density, the preconditioned top
eigenvalue for Adam-type steps, per-loss-term gradients and their pairwise cosines, and
per-sample value and gradient maps over the inputs.

## Install

```
pip install -r requirements.txt     # jax is NOT in there: install the build for your CUDA
pip install -e .
```

## Use

Two steps: record during the run, replay afterwards.

**1. Record.** In the training loop, after the step:

```python
from cockpit_jax.recorder import FlightRecorder

rec = FlightRecorder("runs/flight", keep=5, every=200)   # before the loop
...
host = {"sampler_key": sampler.key}          # plus any Python-side controller
if rec(step, state, float(loss), host=host):
    break     # a bundle is on disk; stop rather than burn GPU hours on a dead run
```

`host` is everything the step depends on that is **not** inside the train state:
PRNG keys, Python-side controllers, counters. Without it the replay is a different
trajectory from the run.

**2. Replay**, in a VS Code `# %%` session:

```python
state, host = cj.ckpt.load_bundle("runs/flight/step_40000", model.state)
rp = cj.replay.Replay(step_fn, state, host, batch_fn=next_batch, probes=probes,
                      keep_every=100, probe_every=10)   # see Memory, below
rp.step(2000, verbose=False)
br = rp.branch("term_off", state=I.set_loss_weight(rp.state, "smooth", 0.0))
br.step(2000, verbose=False)
V.plot_branches([rp, br], ["loss", "grad/norm", "upd/ratio"])

z = rp.zoom(1400, 50)          # that window again, every state kept, every probe on
rp.save("out/sweep", states="all")      # and back later with Replay.load
```

A set of variations from one starting state is `sweep.run_cases`, which runs them on
the same batches, saves each as it finishes, and frees its snapshots before the next
one starts.

`step_fn(state, host, batch) -> (state, host, metrics)` is one iteration of **your**
outer loop, not just the optimizer update: anything else the loop does per iteration
-- a second parameter copy, a loss-weight update, a Python-side controller -- belongs
inside it. See `examples/` for a full one.

## Example

`examples/replay_mlp.ipynb` -- a self-contained notebook: a small MLP on a 2D fitting
problem with a two-term loss, trained on a rising learning rate until it blows up,
then recorded, replayed, probed and branched. No data files, no model library.

## Caveats

- **Determinism.** Replay reproduces the original trajectory only if the bundle holds
  the whole state and GPU ops are deterministic. The examples set
  `--xla_gpu_deterministic_ops=true` and `TF_CUDNN_DETERMINISTIC=1`, at some cost in
  speed. Checkpoints written by a run usually lack the PRNG keys and any Python-side
  controller, so a replay from those is a valid trajectory but not *the* trajectory.
- **Sharpness and the optimizer.** `top_hessian_eig` returns an eigenvalue of the raw
  Hessian, and the 2/learning-rate threshold of Cohen et al. (2021) is a statement
  about plain gradient descent. For Adam-type steps use `top_preconditioned_eig` with
  `adam_pinv`, which is the matching operator -- though `adam_pinv` skips the bias
  correction, so it understates the preconditioner early in a run.
- **Cost.** The gradient-noise instruments need a per-sample gradient pass, and the
  spectral density needs `n_probes * m` HVPs. Both are replay-scale, not loop-scale.
- **Memory.** A state with an Adam-family optimizer runs to tens of MB, so keeping one
  per step is what makes a long replay run out of device memory. `keep_every=n`
  snapshots every n steps and `offload=True` (the default) puts those snapshots in host
  RAM; `zoom` then buys per-step resolution back over one window. `per_sample_grads` and
  `grad_moments` are chunked, but a chunk still materialises `chunk x n_params`, and
  Lanczos with full reorthogonalisation holds `m` flat vectors.
- **A probe is an observation, not a verdict.** Only the keys in `stop_keys` (default
  `("loss",)`) halt a run. A NaN from `alpha` on a step whose fit has no minimum, or
  from a cosine with a zero norm, is recorded and stepped past.
- **The probes measure the `loss_fn` you hand them.** If that is not the exact
  objective your step differentiates, the curvature numbers describe a different
  function from the one being minimised.
- None of this has been benchmarked for overhead. The probes are meant for replay, not
  for the hot loop.

## Citing

If this is useful, cite the original Cockpit paper, not this repository:

```bibtex
@inproceedings{schneider2021cockpit,
  title     = {Cockpit: A Practical Debugging Tool for the Training of Deep Neural Networks},
  author    = {Schneider, Frank and Dangel, Felix and Hennig, Philipp},
  booktitle = {Advances in Neural Information Processing Systems},
  volume    = {34},
  year      = {2021},
  eprint    = {2102.06604},
  archivePrefix = {arXiv}
}
```

- Paper: https://arxiv.org/abs/2102.06604
- Original (PyTorch) implementation: https://github.com/f-dangel/cockpit

Other work the probes come from:

- Cohen, Kaur, Li, Kolter, Talwalkar, *Gradient Descent on Neural Networks Typically
  Occurs at the Edge of Stability*, ICLR 2021 (arXiv:2103.00065) -- sharpness and 2/lr.
- Hutchinson, *A stochastic estimator of the trace of the influence matrix*, 1989 --
  the trace and diagonal estimators.
- Byrd, Chin, Nocedal, Wu, *Sample size selection in optimization methods for machine
  learning*, Math. Program. 2012 -- the norm test.
- Bollapragada, Byrd, Nocedal, *Adaptive sampling strategies for stochastic
  optimization*, SIAM J. Optim. 2018 -- the inner-product and orthogonality tests.
- Balles, Romero, Hennig, *Coupling adaptive batch sizes with learning rates*, UAI 2017
  -- CABS.
- Liu et al., *Understanding Why Neural Networks Generalize Well Through GSNR of
  Parameters*, ICLR 2020 -- the gradient signal-to-noise ratio.
- Thomas, Pedregosa, van Merrienboer et al., *On the interplay between noise and
  curvature*, AISTATS 2020 -- the TIC.
- Ghorbani, Krishnan, Xiao, *An investigation into neural net optimization via Hessian
  eigenvalue density*, ICML 2019 -- stochastic Lanczos quadrature.
- Li, Xu, Taylor, Studer, Goldstein, *Visualizing the Loss Landscape of Neural Nets*,
  NeurIPS 2018 (arXiv:1712.09913) -- loss slices.

## Authors

See [AUTHORS.md](AUTHORS.md).

## License

MIT, see [LICENSE](LICENSE). Cockpit itself is MIT-licensed; no Cockpit code is
included or derived from here.
