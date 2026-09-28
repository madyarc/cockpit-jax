"""Full-state bundles: the train state (Orbax) plus host-side extras (pickle).

Host extras = anything the step depends on that is NOT inside the train state:
sampler PRNG key, Python-side controllers (e.g. a tau clock), counters.
Without them a replay is not the same trajectory.
"""
import os
import pickle

import jax
import orbax.checkpoint as ocp


def save_bundle(path, state, host=None):
    """Write <path>/state (Orbax) and <path>/host.pkl. `path` must not already hold a state."""
    path = os.path.abspath(path)
    os.makedirs(path, exist_ok=True)
    ckptr = ocp.StandardCheckpointer()
    ckptr.save(os.path.join(path, "state"), jax.device_get(state))
    ckptr.wait_until_finished()
    with open(os.path.join(path, "host.pkl"), "wb") as f:
        pickle.dump(jax.device_get(host), f)


def load_bundle(path, template_state):
    """Return (state, host). template_state: a state with the same tree (e.g. model.state)."""
    path = os.path.abspath(path)
    ckptr = ocp.StandardCheckpointer()
    state = ckptr.restore(os.path.join(path, "state"), template_state)
    host = None
    hp = os.path.join(path, "host.pkl")
    if os.path.exists(hp):
        with open(hp, "rb") as f:
            host = pickle.load(f)
    return state, host


def restore_manager_ckpt(ckpt_dir, template_state, step=None, item=None, mngr=None):
    """Restore a checkpoint written by an Orbax CheckpointManager. Returns (state, step).

    step=None -> latest.

    ckpt_dir must be the directory that directly holds the numeric step directories.
    When the training library derives that path (jaxpi: <ckpt_path>/<wandb.name>/ckpt),
    call its function rather than assembling the path by hand -- a manager pointed one
    level too high sees no steps at all.

    item: only for a manager that registered named items, where Orbax requires Composite
    args. The common case is a single unnamed item (what ocp.args.StandardSave writes),
    which needs no name; this tries that first and asks for `item` only if Orbax refuses.

    mngr: pass the manager the training code builds when it uses non-default options.

    Two things this does NOT do, deliberately: it applies no sharding, so the arrays come
    back on one device, and it reads no host-side state. A library that sharded its step
    usually replicates after restoring (jaxpi's restore_checkpoint does, and it takes a
    step argument, so prefer it there), and anything the step needs that lives outside
    the train state has to come from elsewhere -- see save_bundle.
    """
    mngr = mngr or ocp.CheckpointManager(os.path.abspath(ckpt_dir))
    steps = sorted(mngr.all_steps())
    step = mngr.latest_step() if step is None else step
    if step not in steps:
        raise FileNotFoundError(
            "no step {} in {}; steps there: {}".format(step, mngr.directory, steps))

    if item is None:
        try:
            return mngr.restore(step, args=ocp.args.StandardRestore(template_state)), step
        except ValueError as e:
            raise ValueError(
                "{}\n\nThis checkpoint stores named items, so Orbax needs one: call again "
                "with item=<name>. The names are the subdirectories of {}.".format(
                    e, os.path.join(str(mngr.directory), str(step)))) from e

    restored = mngr.restore(
        step, args=ocp.args.Composite(**{item: ocp.args.StandardRestore(template_state)}))
    return restored[item], step


def list_manager_steps(ckpt_dir):
    return sorted(ocp.CheckpointManager(os.path.abspath(ckpt_dir)).all_steps())
