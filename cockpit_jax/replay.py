"""Replay: step a loaded state forward one step at a time, rewind, branch, zoom.

step_fn(state, host, batch) -> (state, host, metrics)
    One training step exactly as the run does it (for a block-structured run,
    the whole block: inner steps, teacher refresh, clock update, weight updates).
    metrics: dict of scalars.
batch_fn() -> batch
    Called only for a step whose batch is not yet cached. Batches are cached by
    step index and shared with branches, so every branch sees identical data.
probes: {name: fn(prev_state, state, batch) -> dict of scalars}
    Run after a step; their outputs go into that step's record.

MEMORY. A long sweep cannot keep every state on the accelerator: one PirateNet-sized
state with an Adam-family optimizer runs to tens of MB, so a few thousand steps is
tens of GB. Three settings control this, and their defaults are the safe ones:

  keep_every=None   snapshot every `keep_every` steps instead of every step, so
                    `state_at` and `rewind` work at that resolution. None keeps only
                    the current state.
  offload=True      snapshots are moved to host memory (jax.device_get) and moved
                    back on use, so they cost RAM rather than VRAM.
  cache_batches     False drops each batch after its step. A branch that must see
                    the same data as the trunk needs the cache; one that only needs
                    statistically identical data does not.

The pattern this is built for: sweep wide and cheap with `keep_every` coarse and
`probe_every` sparse, then `zoom` into the window that matters, which re-runs those
steps from the nearest snapshot with every probe on and every state kept.

PERSISTENCE. `save` writes records, batches and snapshots under one directory, and
`load` rebuilds the object, so a later cell or a later kernel continues instead of
re-running. See store.py for the layout.
"""
import math
import os

import jax
import numpy as np

from . import store


def _to_float(v):
    try:
        a = np.asarray(jax.device_get(v))
    except Exception:
        return v
    if a.ndim == 0:
        return float(a)
    return v


def _nonfinite_keys(rec, watch=None):
    keys = rec.keys() if watch is None else watch
    return [k for k in keys
            if isinstance(rec.get(k), float) and not math.isfinite(rec[k])]


class Replay:
    """See the module docstring. `stop_keys` is the only thing that can halt a run:
    a non-finite value in one of those keys, default ("loss",). A diagnostic going
    NaN -- alpha on a step whose fit has no minimum, a cosine with a zero norm -- is
    recorded and stepped past, because a probe is an observation, not a verdict."""

    def __init__(self, step_fn, state, host=None, batch_fn=None, probes=None,
                 start_step=0, name="base", keep_every=None, offload=True,
                 cache_batches=True, probe_every=1, _batches=None):
        self.step_fn = step_fn
        self.batch_fn = batch_fn
        self.probes = dict(probes or {})
        self.name = name
        self.keep_every = keep_every
        self.offload = offload
        self.cache_batches = cache_batches
        self.probe_every = max(1, int(probe_every))
        self.batches = {} if _batches is None else _batches   # shared across branches
        self.start_step = int(start_step)
        self._k = int(start_step)
        self._state, self._host = state, host
        self.snapshots = {}                                   # step -> (state, host)
        self._snap(self._k, state, host)
        self.records = []
        self.stopped = None                                   # reason string, or None

    # -- state handling -------------------------------------------------------
    def _snap(self, k, state, host):
        """Keep a snapshot if the policy says so. Offloaded snapshots live in host
        memory; they are moved back to the device by state_at."""
        if self.keep_every is None and k != self.start_step:
            return
        if self.keep_every is not None and (k - self.start_step) % self.keep_every:
            return
        self.snapshots[k] = ((jax.device_get(state), jax.device_get(host))
                             if self.offload else (state, host))

    @property
    def k(self):
        return self._k

    @property
    def state(self):
        return self._state

    @property
    def host(self):
        return self._host

    def snapshot_steps(self):
        return sorted(self.snapshots)

    def state_at(self, k, exact=True):
        """(state, host) at step k, on the device.

        exact=False returns the nearest snapshot at or before k instead of raising,
        which is what `zoom` uses to find a starting point.
        """
        if k == self._k:
            return self._state, self._host
        if k not in self.snapshots:
            if exact:
                raise KeyError(
                    "no snapshot at step {}; kept {}. Set keep_every, or use "
                    "exact=False for the nearest earlier one.".format(
                        k, self.snapshot_steps()))
            earlier = [s for s in self.snapshots if s <= k]
            if not earlier:
                raise KeyError("no snapshot at or before step {}".format(k))
            k = max(earlier)
        state, host = self.snapshots[k]
        if self.offload:
            state = jax.device_put(state)
        return state, host

    def drop_states(self, keep=()):
        """Free every snapshot except `keep` (and the current state). The current
        state is never dropped -- the replay would have nowhere to continue from."""
        keep = set(int(x) for x in keep)
        self.snapshots = {s: v for s, v in self.snapshots.items() if s in keep}

    # -- batches --------------------------------------------------------------
    def batch_at(self, k):
        b = self.batches.get(k)
        if b is None:
            if self.batch_fn is None:
                raise RuntimeError("no batch cached for step {} and no batch_fn".format(k))
            b = self.batch_fn()
            if self.cache_batches:
                self.batches[k] = b
        elif isinstance(b, dict):
            b = {name: jax.device_put(arr) if isinstance(arr, np.ndarray) else arr
                 for name, arr in b.items()}
        elif isinstance(b, np.ndarray):
            b = jax.device_put(b)
        return b

    def set_batch(self, k, batch):
        """Pre-load or override the batch used at step k. The cache is shared with
        branches, so this changes that step for every branch that has not run it."""
        self.batches[k] = batch

    def drop_batches(self):
        """Free the shared batch cache. Branches that have not yet run those steps
        will draw fresh batches, which are then no longer identical to the trunk's."""
        self.batches.clear()

    # -- stepping -------------------------------------------------------------
    def step(self, n=1, batch=None, stop_keys=("loss",), verbose=True,
             probe_every=None, progress_every=None):
        """Advance n steps. Returns the number of steps actually taken.

        stop_keys: halt when one of these record keys is non-finite. Pass () to run
        through anything. Probe columns are never a stop condition unless named here.
        batch: use this batch for these steps instead of the cache.
        probe_every: override the instance setting for this call.
        progress_every: print one summary line every m steps (implies verbose=False).
        """
        pe = self.probe_every if probe_every is None else max(1, int(probe_every))
        taken = 0
        for i in range(int(n)):
            k = self._k
            b = self.batch_at(k) if batch is None else batch
            prev_state, host = self._state, self._host
            new_state, new_host, metrics = self.step_fn(prev_state, host, b)

            rec = {"branch": self.name, "step": k}
            rec.update({key: _to_float(v) for key, v in metrics.items()})
            if self.probes and (i % pe == 0 or i == n - 1):
                for probe in self.probes.values():
                    for key, v in probe(prev_state, new_state, b).items():
                        rec[key] = _to_float(v)
            self.records.append(rec)

            self._k, self._state, self._host = k + 1, new_state, new_host
            self._snap(self._k, new_state, new_host)
            if not self.cache_batches and batch is None:
                self.batches.pop(k, None)
            taken += 1

            if verbose:
                print(self._line(rec))
            elif progress_every and (taken % progress_every == 0 or i == n - 1):
                print(self._line(rec))

            bad = _nonfinite_keys(rec, stop_keys or ())
            if bad:
                self.stopped = "non-finite {} at step {}".format(bad, k)
                print(self.name + ": stopped, " + self.stopped)
                break
        return taken

    def run_until(self, cond, max_steps=1000, **kw):
        """Step until cond(record) is True, or max_steps, or a stop key goes
        non-finite. Returns the last record."""
        kw.setdefault("verbose", False)
        for _ in range(int(max_steps)):
            if self.step(1, **kw) == 0 or self.stopped:
                break
            if self.records and cond(self.records[-1]):
                break
        return self.records[-1] if self.records else None

    def rewind(self, k):
        """Go back to step k, which must be a kept snapshot. Records after k are
        dropped; snapshots after k are freed."""
        state, host = self.state_at(k)
        self._k, self._state, self._host = k, state, host
        self.records = [r for r in self.records if r["step"] < k]
        self.snapshots = {s: v for s, v in self.snapshots.items() if s <= k}
        self.stopped = None

    def branch(self, name, step_fn=None, state=None, host=None, probes=None,
               at=None, **kw):
        """New Replay from this one's current step, sharing the batch cache.

        step_fn / state / host / probes replace this branch's own (the intervention);
        `at` branches from an earlier snapshot instead of the current step; any
        keep_every / offload / cache_batches / probe_every is inherited unless given.
        """
        if at is None:
            k, st, ho = self._k, self._state, self._host
        else:
            k = int(at)
            st, ho = self.state_at(k)
        opts = dict(keep_every=self.keep_every, offload=self.offload,
                    cache_batches=self.cache_batches, probe_every=self.probe_every)
        opts.update(kw)
        return Replay(step_fn or self.step_fn,
                      st if state is None else state,
                      ho if host is None else host,
                      batch_fn=self.batch_fn,
                      probes=self.probes if probes is None else probes,
                      start_step=k, name=name, _batches=self.batches, **opts)

    def zoom(self, start, steps, name=None, probes=None, step_fn=None, **kw):
        """Re-run a window at full resolution: every state kept, every probe on.

        Branches from the nearest snapshot at or before `start` and runs `steps`
        steps on the same batches, so the window is the same trajectory, not a
        similar one -- provided the batches are still cached and the step is
        deterministic. If the nearest snapshot is earlier than `start`, the window
        begins there, and the returned object says so in its start_step.
        """
        s0 = start
        if start not in self.snapshots:
            earlier = [s for s in self.snapshots if s <= start]
            if not earlier:
                raise KeyError("no snapshot at or before step {}; kept {}".format(
                    start, self.snapshot_steps()))
            s0 = max(earlier)
            print("no snapshot at {}; starting the zoom at {}".format(start, s0))
        opts = dict(keep_every=1, offload=self.offload, cache_batches=True,
                    probe_every=1)
        opts.update(kw)
        br = self.branch(name or "{}_zoom{}".format(self.name, s0),
                         step_fn=step_fn, probes=probes, at=s0, **opts)
        br.step(steps + (start - s0), verbose=False)
        return br

    # -- output ---------------------------------------------------------------
    def column(self, key):
        """(steps, values) for one record key, skipping steps that lack it."""
        rows = [(r["step"], r[key]) for r in self.records
                if key in r and isinstance(r[key], float)]
        return (np.array([s for s, _ in rows]),
                np.array([v for _, v in rows], dtype=float))

    def last(self, key, default=float("nan")):
        for r in reversed(self.records):
            v = r.get(key)
            if isinstance(v, float):
                return v
        return default

    def to_csv(self, path):
        store.save_records(path, self.records)

    # -- persistence ----------------------------------------------------------
    def save(self, root, states="last", batches=True):
        """Write this branch under <root>/<name>/ so a later cell can reload it.

        states: "last" (enough to continue), "all" (every snapshot; enough to zoom
        into an earlier window later), or "none".
        batches: write the shared batch cache, so a reloaded branch replays the same
        data. It is shared, so saving two branches writes it twice.
        """
        d = os.path.join(root, self.name)
        os.makedirs(os.path.join(d, "states"), exist_ok=True)
        store.save_records(os.path.join(d, "records.csv"), self.records)
        if batches:
            store.save_batches(os.path.join(d, "batches.npz"), self.batches)

        saved = []
        if states == "all":
            want = sorted(set(self.snapshots) | {self._k})
        elif states == "last":
            want = [self._k]
        else:
            want = []
        for k in want:
            st, ho = (self._state, self._host) if k == self._k else self.snapshots[k]
            from .ckpt import save_bundle
            save_bundle(os.path.join(d, "states", "step_{}".format(k)), st, ho)
            saved.append(int(k))

        store.save_meta(os.path.join(d, "meta.json"), {
            "name": self.name, "start_step": self.start_step, "step": int(self._k),
            "keep_every": self.keep_every, "offload": self.offload,
            "cache_batches": self.cache_batches, "probe_every": self.probe_every,
            "stopped": self.stopped, "saved_states": saved,
            "n_records": len(self.records),
        })
        return d

    @classmethod
    def load(cls, root, name, template_state, step_fn, batch_fn=None, probes=None,
             step=None, **kw):
        """Rebuild a saved branch. `template_state` is a state with the same tree
        (e.g. model.state); `step_fn` and `probes` are rebuilt by the caller, since
        functions are not saved.

        step=None loads the latest saved state. Records and batches come back whole,
        so plots and `zoom` work without re-running anything.
        """
        from .ckpt import load_bundle
        d = os.path.join(root, name)
        meta = store.load_meta(os.path.join(d, "meta.json"))
        saved = meta.get("saved_states") or []
        if not saved:
            raise FileNotFoundError("no states saved for branch {} in {}".format(name, d))
        k = max(saved) if step is None else int(step)
        if k not in saved:
            raise KeyError("no saved state at step {}; saved {}".format(k, saved))
        state, host = load_bundle(os.path.join(d, "states", "step_{}".format(k)),
                                  template_state)

        opts = dict(keep_every=meta.get("keep_every"), offload=meta.get("offload", True),
                    cache_batches=meta.get("cache_batches", True),
                    probe_every=meta.get("probe_every", 1))
        opts.update(kw)
        rp = cls(step_fn, state, host, batch_fn=batch_fn, probes=probes,
                 start_step=k, name=name,
                 _batches=store.load_batches(os.path.join(d, "batches.npz")), **opts)
        rp.start_step = meta.get("start_step", k)
        rp.records = store.load_records(os.path.join(d, "records.csv"))
        rp.stopped = meta.get("stopped")
        for s in saved:
            if s == k:
                continue
            try:
                st, ho = load_bundle(os.path.join(d, "states", "step_{}".format(s)),
                                     template_state)
                rp.snapshots[s] = (jax.device_get(st), jax.device_get(ho)) if rp.offload \
                    else (st, ho)
            except Exception as e:
                print("could not load the snapshot at step {}: {}".format(s, e))
        return rp

    @staticmethod
    def _line(rec):
        parts = ["{}[{}]".format(rec["branch"], rec["step"])]
        for key, v in rec.items():
            if key in ("branch", "step") or not isinstance(v, float):
                continue
            parts.append("{}={:.3e}".format(key, v))
        return " ".join(parts)
