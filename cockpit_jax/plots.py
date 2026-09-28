"""matplotlib views. Every function returns the Figure; in a VS Code '# %%' cell it
renders inline in the Interactive window."""
import math

import matplotlib.pyplot as plt
import numpy as np


def _records_keys(records, prefix):
    keys = []
    for r in records:
        for k in r:
            if k.startswith(prefix) and k not in keys:
                keys.append(k)
    return keys


def plot_records(records, keys, logy=True, hline=None, title=None):
    """One panel per key, value vs step. hline: {key: y} draws a reference line (e.g. 2/lr)."""
    n = len(keys)
    cols = min(n, 3)
    rows = int(math.ceil(n / cols))
    fig, axs = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.0 * rows), squeeze=False)
    for ax, key in zip(axs.flat, keys):
        s = [r["step"] for r in records if key in r]
        v = [r[key] for r in records if key in r]
        ax.plot(s, np.abs(v) if logy else v, lw=1.2)
        if logy:
            ax.set_yscale("log")
        if hline and key in hline:
            ax.axhline(hline[key], color="k", ls="--", lw=0.8)
        ax.set_title(key, fontsize=9)
        ax.set_xlabel("step")
    for ax in list(axs.flat)[n:]:
        ax.axis("off")
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    return fig


def plot_prefix(records, prefix, logy=True):
    """Every key starting with prefix on one axis (e.g. 'lgrad/' for per-layer grad norms)."""
    keys = _records_keys(records, prefix)
    fig, ax = plt.subplots(figsize=(7, 4))
    for key in keys:
        s = [r["step"] for r in records if key in r]
        v = [r[key] for r in records if key in r]
        ax.plot(s, np.abs(v) if logy else v, lw=1.0, label=key[len(prefix):])
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel("step")
    ax.set_title(prefix)
    ax.legend(fontsize=6, ncol=2)
    fig.tight_layout()
    return fig


def plot_branches(replays, keys, logy=True):
    """Overlay the same keys for several Replay objects (branches from one checkpoint)."""
    n = len(keys)
    fig, axs = plt.subplots(1, n, figsize=(4.5 * n, 3.2), squeeze=False)
    for ax, key in zip(axs[0], keys):
        for rp in replays:
            s, v = rp.column(key)
            ax.plot(s, np.abs(v) if logy else v, lw=1.2, label=rp.name)
        if logy:
            ax.set_yscale("log")
        ax.set_title(key, fontsize=9)
        ax.set_xlabel("step")
        ax.legend(fontsize=7)
    fig.tight_layout()
    return fig


def plot_points(batch, values, cols=(0, 1), labels=("x1", "x2"), title=None, log=False, s=4):
    """Scatter of a per-sample value over two input columns (e.g. per-sample grad norm)."""
    b = np.asarray(batch)
    v = np.asarray(values)
    c = np.log10(np.abs(v) + 1e-300) if log else v
    fig, ax = plt.subplots(figsize=(6, 4))
    sc = ax.scatter(b[:, cols[0]], b[:, cols[1]], c=c, s=s, cmap="viridis")
    fig.colorbar(sc, ax=ax, label=("log10 |value|" if log else "value"))
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    if title:
        ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_leaf_bars(d, title=None, logx=True):
    """Horizontal bars of {leaf_path: value}, e.g. layer_grad_norms output."""
    keys = list(d.keys())
    vals = [d[k] for k in keys]
    fig, ax = plt.subplots(figsize=(7, 0.22 * len(keys) + 1))
    ax.barh(range(len(keys)), vals)
    ax.set_yticks(range(len(keys)))
    ax.set_yticklabels(keys, fontsize=6)
    ax.invert_yaxis()
    if logx:
        ax.set_xscale("log")
    if title:
        ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_leaf_hists(tree_leaves_with_paths, bins=60, max_panels=24):
    """Histogram per leaf. Pass tree.leaves_with_paths(params_or_grads)."""
    items = tree_leaves_with_paths[:max_panels]
    cols = 4
    rows = int(math.ceil(len(items) / cols))
    fig, axs = plt.subplots(rows, cols, figsize=(3.2 * cols, 2.2 * rows), squeeze=False)
    for ax, (p, l) in zip(axs.flat, items):
        ax.hist(np.ravel(np.asarray(l)), bins=bins)
        ax.set_title(p, fontsize=6)
    for ax in list(axs.flat)[len(items):]:
        ax.axis("off")
    fig.tight_layout()
    return fig


def plot_slice(alphas, losses, title=None):
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.plot(alphas, losses, "-o", ms=3)
    ax.axvline(0.0, color="k", lw=0.6)
    ax.set_yscale("log")
    ax.set_xlabel("alpha")
    ax.set_ylabel("loss")
    if title:
        ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_plane(a1, a2, grid, title=None):
    fig, ax = plt.subplots(figsize=(5, 4))
    cs = ax.contourf(a2, a1, np.log10(np.abs(grid) + 1e-300), levels=30)
    fig.colorbar(cs, ax=ax, label="log10 loss")
    ax.plot(0, 0, "r+")
    ax.set_xlabel("d2")
    ax.set_ylabel("d1")
    if title:
        ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_density(grid, dens, title=None, logy=True):
    """Eigenvalue density from curvature.density_curve. The negative part is the
    interesting one: it is what a single top eigenvalue cannot show."""
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(grid, dens, lw=1.2)
    ax.axvline(0.0, color="k", lw=0.6)
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel("eigenvalue")
    ax.set_ylabel("density")
    if title:
        ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_values_hist(values, bins=60, logx=False, title=None):
    """Histogram of a flat array, e.g. per-sample gradient norms or a variance vector."""
    v = np.ravel(np.asarray(values))
    if logx:
        v = np.log10(np.abs(v) + 1e-300)
    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.hist(v, bins=bins)
    ax.set_xlabel("log10 |value|" if logx else "value")
    if title:
        ax.set_title(title)
    fig.tight_layout()
    return fig
