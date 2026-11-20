"""Example 05's processed outcrop data, read the same way in every notebook.

``05a_hartland_quay_data`` cleans the raw files, picks the traces and the fold
axis, and writes ``data/processed/``; the fitting, meshing and figure notebooks
read it through here. numpy only.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

PROCESSED_DIR = Path("data/processed")
RESULTS_DIR = Path("results")


def unpack(xyz, offsets):
    """Ragged polylines stored as one array plus offsets: polyline i is ``xyz[off[i]:off[i + 1]]``."""
    return [xyz[a:b] for a, b in zip(offsets[:-1], offsets[1:])]


def load_processed(processed_dir: Path = PROCESSED_DIR) -> dict:
    """Everything 05a saved, in the modelling frame (add ``origin`` for the file frame).

    ``traces`` are the kept, thinned traces the fits use; ``all_traces`` every
    cleaned trace at full density. ``gp`` and ``gv`` are every cleaned bedding
    reading; ``gp_fit`` and ``gv_fit`` are the ones the fits use, and ``gp_test``
    and ``gv_test`` the ones held out for the bedding error. ``test_traces`` are
    the clubbed pairs held out of the fit (H22 with H25, H1 with H7), one trace
    each, kept at full density for the error metrics. ``hinge`` is the fold axis the
    restore uses (the π-axis of all bedding); ``u_ax``, ``w_ax`` span the fold
    profile.
    """
    processed_dir = Path(processed_dir)
    d = np.load(processed_dir / "hartland_processed.npz")
    return dict(
        origin=d["origin"],
        trace_names=[str(n) for n in d["trace_names"]],
        traces=unpack(d["trace_xyz"], d["trace_offsets"]),
        all_trace_names=[str(n) for n in d["all_trace_names"]],
        all_traces=unpack(d["all_trace_xyz"], d["all_trace_offsets"]),
        gp=d["gp"], gv=d["gv"], on_beach=d["on_beach"],
        gp_fit=d["gp_fit"], gv_fit=d["gv_fit"], on_beach_fit=d["on_beach_fit"],
        gp_test=d["gp_test"], gv_test=d["gv_test"], on_beach_test=d["on_beach_test"],
        test_trace_names=[str(n) for n in d["test_trace_names"]],
        test_traces=unpack(d["test_trace_xyz"], d["test_trace_offsets"]),
        gp_reversed=d["gp_reversed"], gv_reversed=d["gv_reversed"],
        cloud_xyz=d["cloud_xyz"].astype(float), cloud_rgb=d["cloud_rgb"] / 255.0,
        hinge=d["hinge"], u_ax=d["u_ax"], w_ax=d["w_ax"],
        settings=json.loads((processed_dir / "settings.json").read_text()),
    )


def heldout_traces(data: dict) -> tuple[list[str], list]:
    """The contact test, ``(names, polylines)``.

    Each polyline is one pair 05a held out of the fit and clubbed into a single
    trace, H22 with H25 and H1 with H7, at full density. The score is whether the
    points of a pair lie on one surface. ``data`` is :func:`load_processed`'s output.
    """
    return list(data["test_trace_names"]), list(data["test_traces"])


def trace_flatness(value_and_gradient, traces) -> np.ndarray:
    """How far each trace lies from one level surface of a field, in metres (RMS per trace).

    ``value_and_gradient(points)`` returns the field and its gradient at ``(n, 3)`` points.
    Each point's distance to the level ``c`` is first order with its own gradient,
    ``|ψ_i - c| / |∇ψ_i|``, and ``c`` is the level that minimises the squared distances,
    ``Σ w_i ψ_i / Σ w_i`` with ``w_i = 1 / |∇ψ_i|²``. Rescaling the field leaves the result
    unchanged, so fields of different models compare directly.
    """
    out = []
    for t in traces:
        v, g = value_and_gradient(np.asarray(t, dtype=float))
        v = np.asarray(v, dtype=float).ravel()
        w = 1.0 / np.maximum(np.einsum("ij,ij->i", np.asarray(g, dtype=float), np.asarray(g, dtype=float)), 1e-24)
        c = np.sum(w * v) / np.sum(w)
        out.append(float(np.sqrt(np.mean(w * (v - c) ** 2))))
    return np.asarray(out)


BLOCKS = dict(n_blocks=7, buffer=5.0)   # the spatial hold-out of 05c and 05i: 7 blocks, the 5 interior ones held out


def block_folds(traces, gp, u_ax, n_blocks: int = BLOCKS["n_blocks"], buffer: float = BLOCKS["buffer"]) -> list[dict]:
    """The spatial hold-out folds, one per interior block, identical in every notebook that uses them.

    The traces are split into ``n_blocks`` blocks of equal count along the fold profile ``u_ax``, by trace
    centroid. Each interior block is held out in turn; the two at the ends of the outcrop are always fitted,
    since holding them out would test extrapolation beyond the outcrop rather than interpolation across a gap.
    For each fold: ``test`` (indices of the held-out traces), ``train`` (the other traces, with any points within
    ``buffer`` m of the block, in u, cut; traces left with fewer than 2 points are dropped), ``train_index``
    (the trace each training polyline came from), ``bed_test`` and ``bed_train`` (masks over ``gp``: the
    bedding in the block, and the bedding outside the block and its buffer), and the block's ``lo``, ``hi`` in u.
    """
    cu = np.array([np.asarray(t).mean(0) @ u_ax for t in traces])
    edges = np.quantile(cu, np.linspace(0.0, 1.0, n_blocks + 1))
    edges[-1] += 1e-6
    block_of = np.clip(np.searchsorted(edges, cu, side="right") - 1, 0, n_blocks - 1)
    gu = np.asarray(gp) @ u_ax
    folds = []
    for b in range(1, n_blocks - 1):
        lo, hi = float(edges[b]), float(edges[b + 1])
        train, train_index = [], []
        for k in np.flatnonzero(block_of != b):
            s = np.asarray(traces[k]) @ u_ax
            keep = ~((s > lo - buffer) & (s < hi + buffer))
            if keep.sum() >= 2:
                train.append(np.asarray(traces[k])[keep])
                train_index.append(int(k))
        folds.append(dict(block=b, lo=lo, hi=hi, test=np.flatnonzero(block_of == b),
                          train=train, train_index=np.asarray(train_index),
                          bed_test=(gu >= lo) & (gu < hi), bed_train=(gu < lo - buffer) | (gu >= hi + buffer)))
    return folds
