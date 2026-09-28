"""Persistence for a replay: records, batches and states to disk and back.

A cell that took ten minutes should not have to run twice. `Replay.save` writes
everything a later cell (or a later kernel) needs to pick the replay up where it
stopped, and `Replay.load` rebuilds it.

Layout under one directory per branch:

    <dir>/meta.json        name, start_step, current step, keep_every, stopped reason
    <dir>/records.csv      one row per step, every metric and probe column
    <dir>/batches.npz      the batches the branch consumed, keyed by step
    <dir>/states/step_<k>/ full state bundles (Orbax) for the snapshots kept

States are the expensive part; `Replay.save(states="last")` writes only the current
one, which is enough to continue, while "all" writes every snapshot, which is what a
later zoom into an earlier step needs.
"""
import csv
import json
import os

import jax
import numpy as np


# ---------------------------------------------------------------------------
# records
# ---------------------------------------------------------------------------

def save_records(path, records):
    keys = []
    for r in records:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in records:
            w.writerow({k: r.get(k, "") for k in keys})


def load_records(path):
    """Rows back as dicts, with numeric-looking fields as floats and step as int."""
    out = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            rec = {}
            for k, v in row.items():
                if v == "" or v is None:
                    continue
                if k == "step":
                    rec[k] = int(float(v))
                elif k == "branch":
                    rec[k] = v
                else:
                    try:
                        rec[k] = float(v)
                    except ValueError:
                        rec[k] = v
            out.append(rec)
    return out


# ---------------------------------------------------------------------------
# batches
# ---------------------------------------------------------------------------

def save_batches(path, batches):
    """batches: {step: array} or {step: {name: array}}. Written as one npz."""
    flat = {}
    for step, b in batches.items():
        b = jax.device_get(b)
        if isinstance(b, dict):
            for name, arr in b.items():
                flat["{}|{}".format(int(step), name)] = np.asarray(arr)
        else:
            flat["{}|".format(int(step))] = np.asarray(b)
    np.savez(path, **flat)


def load_batches(path):
    """The inverse. Arrays come back as numpy and are moved to the device on use."""
    out = {}
    if not os.path.exists(path):
        return out
    with np.load(path) as z:
        for key in z.files:
            s, name = key.split("|", 1)
            step = int(s)
            if name:
                out.setdefault(step, {})[name] = z[key]
            else:
                out[step] = z[key]
    return out


# ---------------------------------------------------------------------------
# meta
# ---------------------------------------------------------------------------

def save_meta(path, meta):
    with open(path, "w") as f:
        json.dump(meta, f, indent=1)


def load_meta(path):
    with open(path) as f:
        return json.load(f)


def list_saved(root):
    """Branch directories under `root`, i.e. the ones holding a meta.json."""
    if not os.path.isdir(root):
        return []
    return sorted(d for d in os.listdir(root)
                  if os.path.exists(os.path.join(root, d, "meta.json")))
