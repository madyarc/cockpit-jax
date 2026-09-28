"""FlightRecorder: keep the last few full states in host memory; dump them on a trigger.

Use inside the training loop, after each step:

    rec = FlightRecorder("runs/x/flight", keep=5, every=200)
    ...
    if rec(step, model.state, loss, host={"sampler_key": sampler.key, ...}):
        break   # optional: stop the run once the bundle is on disk

Triggers, in the order they are checked:
  - the loss is non-finite
  - the loss exceeds spike * median of the last `window` losses
  - any watched quantity in `aux` exceeds spike * its own running median
    (pass watch={"grad_norm": 5.0, "upd_ratio": 20.0} to set per-key factors)
  - a parameter or optimizer-state leaf is non-finite (checked on buffered steps only,
    since it costs a device sync)
  - trigger_fn(step, state, loss, aux) returns a reason string

The first three usually fire later than the last two: a gradient-norm spike normally
precedes the loss spike by a few steps, which is exactly the window the replay wants.
On trigger every buffered state is written as <out_dir>/step_<k>/ via ckpt.save_bundle,
and trigger.txt records why.
"""
import collections
import copy
import math
import os

import jax
import numpy as np

from .ckpt import save_bundle


class FlightRecorder:
    def __init__(self, out_dir, keep=5, every=200, spike=10.0, window=200,
                 min_history=50, watch=None, trigger_fn=None, check_nonfinite_state=True):
        self.out_dir = os.path.abspath(out_dir)
        self.every = every
        self.spike = spike
        self.min_history = min_history
        self.watch = dict(watch or {})
        self.trigger_fn = trigger_fn
        self.check_nonfinite_state = check_nonfinite_state
        self.ring = collections.deque(maxlen=keep)
        self.losses = collections.deque(maxlen=window)
        self.aux_hist = {k: collections.deque(maxlen=window) for k in self.watch}
        self.fired = False
        self.reason = None

    @staticmethod
    def _spiked(value, history, factor, min_history):
        if not math.isfinite(value):
            return "non-finite {}".format(value)
        if len(history) >= min_history:
            med = float(np.median(np.asarray(history)))
            if med > 0.0 and value > factor * med:
                return "{:.6e} > {} x median {:.6e}".format(value, factor, med)
        return None

    def _check(self, step, state, loss, aux, buffered):
        r = self._spiked(loss, self.losses, self.spike, self.min_history)
        if r:
            return "loss: " + r
        for key, factor in self.watch.items():
            if aux is None or key not in aux:
                continue
            r = self._spiked(float(aux[key]), self.aux_hist[key], factor, self.min_history)
            if r:
                return "{}: {}".format(key, r)
        if buffered and self.check_nonfinite_state:
            bad = [p for p, l in
                   [(jax.tree_util.keystr(p), l)
                    for p, l in jax.tree_util.tree_flatten_with_path(self.ring[-1][1])[0]]
                   if not bool(np.all(np.isfinite(np.asarray(l))))]
            if bad:
                return "non-finite state leaves: {}".format(bad[:5])
        if self.trigger_fn is not None:
            r = self.trigger_fn(step, state, loss, aux)
            if r:
                return str(r)
        return None

    def __call__(self, step, state, loss, host=None, aux=None):
        """Returns True on the step a trigger fires (only once).

        aux: any extra scalars the loop already has (gradient norm, update ratio),
        watched if their key is in `watch`.
        """
        if self.fired:
            return False
        loss = float(loss)
        buffered = step % self.every == 0
        if buffered:
            self.ring.append((step, jax.device_get(state), copy.deepcopy(jax.device_get(host))))
        reason = self._check(step, state, loss, aux, buffered)
        self.losses.append(loss)
        for key in self.watch:
            if aux is not None and key in aux:
                self.aux_hist[key].append(float(aux[key]))
        if reason is None:
            return False
        self.fired, self.reason = True, reason
        self.dump(step)
        return True

    def dump(self, trigger_step=None):
        """Write every buffered state to disk. Can also be called manually."""
        os.makedirs(self.out_dir, exist_ok=True)
        for step, state, host in self.ring:
            save_bundle(os.path.join(self.out_dir, "step_{}".format(step)), state, host)
        with open(os.path.join(self.out_dir, "trigger.txt"), "w") as f:
            f.write("trigger_step {}\nreason {}\nsaved {}\n".format(
                trigger_step, self.reason, [s for s, _, _ in self.ring]))
