"""cockpit_jax: an unofficial, partial JAX implementation of Cockpit
(Schneider, Dangel and Hennig, NeurIPS 2021, arXiv:2102.06604), plus a
replay-from-checkpoint layer that Cockpit does not have.

Modules
  tree          pytree helpers (norms, per-leaf stats, non-finite search)
  ckpt          save / load a full bundle (state + host-side extras)
  recorder      FlightRecorder: ring buffer of states, dumps on a loss spike or NaN
  replay        Replay: step one at a time from a loaded state, rewind, branch,
                zoom into a window, save and reload
  store         records / batches / meta on disk, behind Replay.save and .load
  sweep         run a set of cases from one start, save each as it finishes
  probes        gradients per layer and per loss term, loss slices, per-sample
                values and gradients, alpha (step quality), NaN location, and the
                probe factories for Replay; re-exports curvature and noise
  curvature     HVP, top and top-k eigenvalues, preconditioned top eigenvalue,
                Hutchinson trace and diagonal, spectral density (Lanczos)
  noise         per-sample gradient moments, norm / inner-product / orthogonality
                tests, mean GSNR, CABS, TIC
  interventions edits to batch / params / state for counterfactual branches
  plots         matplotlib views of records, branches, sample maps, layers
"""
from . import (tree, ckpt, store, recorder, replay, sweep, curvature, noise,
               probes, interventions, plots)
