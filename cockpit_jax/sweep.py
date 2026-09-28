"""Running a set of cases from one starting state, and saving the result.

A case is a named variation on the step: a different step_fn, an edited state, or
both. `run_cases` steps them one after another from the same start, on the same
batches, and writes each to disk as it finishes so a later cell reloads it instead of
re-running.

Order matters. The first case is the trunk: it draws the batches, and every later case
replays them from the shared cache. So the trunk runs first and runs the full length;
if it stops early, the cases after it draw fresh batches from that point on and are no
longer comparable to it. `run_cases` says so when it happens.

Memory. Each case holds its current state on the accelerator and its snapshots in host
memory (`offload=True`). What makes a sweep run out of device memory is keeping every
step's state, so `keep_every` is coarse by default and `free_states` drops the host
copies once they are safely on disk.
"""
import time

import jax

from .replay import Replay


def _fmt(run, keys):
    return "  ".join("{} {:.3e}".format(k.split("/")[-1], run.last(k)) for k in keys)


def run_cases(cases, state, host, step_fn_of, batch_fn, steps, start_step=0,
              probes=None, save_root=None, save_states="last", free_states="saved",
              keep_every=None, offload=True, probe_every=1, stop_keys=("loss",),
              report_every=250, report_keys=("loss",), state_of=None, host_of=None):
    """Run every case for `steps` steps from the same start. Returns {name: Replay}.

    cases: an ordered mapping {name: case}. The case objects are yours; they reach
        your code only through step_fn_of / state_of / host_of.
    step_fn_of(name, case) -> step_fn      required
    state_of(name, case, state) -> state   optional, for an edited starting state
    host_of(name, case, host) -> host      optional, likewise
    keep_every: snapshot every n steps, so `zoom` and `state_at` work at that
        resolution. None keeps only the current state, which is the cheapest and
        leaves zoom nothing to start from.
    save_root: each case is saved under <save_root>/<name>/ as soon as it finishes, so
        a crash in case five does not cost cases one to four.
    save_states: "last" (enough to continue), "all" (every snapshot: what a later zoom
        needs), "none".
    free_states: after saving, "saved" drops the host-memory snapshots that are now on
        disk, "all" drops every snapshot, "none" keeps them. Dropped snapshots come
        back with Replay.load; kept ones need no reload.
    """
    names = list(cases)
    runs, shared, trunk_short = {}, None, None
    t0 = time.time()

    for name in names:
        case = cases[name]
        st = state if state_of is None else state_of(name, case, state)
        ho = host if host_of is None else host_of(name, case, host)

        # Every case starts from the caller's state, not from the trunk's history, so
        # freeing the trunk's snapshots cannot strand a later case. Only the batch
        # cache is shared, which is the point.
        run = Replay(step_fn_of(name, case), st, ho, batch_fn=batch_fn, probes=probes,
                     start_step=start_step, name=name, keep_every=keep_every,
                     offload=offload, cache_batches=True, probe_every=probe_every,
                     _batches=shared)
        if shared is None:
            shared = run.batches

        t1 = time.time()
        run.step(steps, stop_keys=stop_keys, verbose=False, progress_every=report_every)
        print("{:>12s} {:5d} steps in {:6.1f}s   {}{}".format(
            name, run.k - start_step, time.time() - t1, _fmt(run, report_keys),
            "  STOPPED: " + run.stopped if run.stopped else ""))
        if trunk_short is None:
            trunk_short = run.k - start_step
            if trunk_short < steps:
                print("    the trunk stopped at step {}; cases after it will draw "
                      "fresh batches beyond that point".format(run.k))

        saved = []
        if save_root is not None:
            run.save(save_root, states=save_states, batches=(len(runs) == 0))
            saved = sorted(run.snapshots) if save_states == "all" else [run.k]
        if free_states == "all":
            run.drop_states()
        elif free_states == "saved" and saved:
            run.drop_states(keep=[s for s in run.snapshots if s not in saved])
        jax.clear_caches()
        runs[name] = run

    print("{:.1f} s for {} cases x {} steps".format(time.time() - t0, len(runs), steps))
    return runs


def load_cases(save_root, template_state, step_fn_of, cases=None, batch_fn=None,
               probes=None, **kw):
    """Reload what run_cases wrote. Returns {name: Replay}.

    cases: the same mapping as before, so step_fn_of can rebuild each step function.
        When omitted, every saved branch under save_root is loaded with
        step_fn_of(name, None).
    """
    from . import store
    names = list(cases) if cases is not None else store.list_saved(save_root)
    runs = {}
    for name in names:
        case = cases[name] if cases is not None else None
        runs[name] = Replay.load(save_root, name, template_state,
                                 step_fn_of(name, case), batch_fn=batch_fn,
                                 probes=probes, **kw)
        print("{:>12s} loaded at step {} with {} records".format(
            name, runs[name].k, len(runs[name].records)))
    return runs


def summary(runs, keys, fmt="{:>12.4e}"):
    """Print the last value of each key for every run."""
    head = "{:>12s}".format("case") + "".join(
        "{:>14s}".format(k.split("/")[-1]) for k in keys)
    print(head)
    print("-" * len(head))
    for name, run in runs.items():
        row = "{:>12s}".format(name)
        for k in keys:
            row += "  " + fmt.format(run.last(k))
        print(row + ("  STOPPED" if run.stopped else ""))
