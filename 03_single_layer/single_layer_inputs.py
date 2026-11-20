"""Example 03's digitised benchmark, observation split and ensemble measures, shared by every notebook.

``03a_single_layer_data`` writes the ground truth and the sparse-observation
split, ``03b_single_layer_fit`` (curlew) fits one LDI model without and one with strain
to tune the settings, ``03c_single_layer_ensemble`` (curlew) fits the ensembles with them and
``03d_single_layer_figures`` draws the results. All four read the data through
here, so they use the same shifted frame, the same horizon order (H0 at the
top) and the same observations. numpy only.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

DATA_DIR = Path("data")
GROUND_TRUTH = "single_layer_ground_truth.npz"
OBSERVATIONS = "single_layer_observations.npz"
HUDLESTON_S = 6.0           # bulk strain ratio X/Z of Hudleston's Fig. 3A (24:1)
MATRIX = (0, 1, 4, 5)       # matrix grid lines; H2 and H3 bound the competent layer
COMPETENT = (2, 3)


def _traces(xy, ids):
    return [xy[ids == i] for i in np.unique(ids)]


def load_truth(data_dir: Path = DATA_DIR) -> dict:
    """The digitised benchmark, shifted to the mean of the horizon points.

    Horizons are returned top to bottom (H0 uppermost); the file stores them
    bottom to top. Vertical marker lines run left to right.
    """
    gt = np.load(Path(data_dir) / GROUND_TRUTH)
    horizon_xy = gt["horizon_xy"].astype(float)
    origin = horizon_xy.mean(axis=0)
    horizons_full = _traces(horizon_xy - origin, gt["horizon_id"])[::-1]
    ratio = gt["strain_eigenvalue_ratio"].astype(float)
    return dict(
        origin=origin,
        horizons_full=horizons_full,
        horizon_names=[f"H{i}" for i in range(len(horizons_full))],
        verticals_full=_traces(gt["vertical_xy"].astype(float) - origin, gt["vertical_id"]),
        gradient_xy=gt["gradient_xy"].astype(float) - origin,
        gradient_vector=gt["gradient_vector"].astype(float),
        strain_xy=gt["strain_xy"].astype(float) - origin,
        strain_eigenvectors=gt["strain_eigenvectors"].astype(float),   # columns: minor, major (spatial)
        strain_eigenvalue_ratio=ratio,                                  # of B = F F^T
        q_truth=np.sqrt(ratio),                                         # stretch ratio
    )


def farthest_point_indices(points, n, seed):
    """Greedy farthest-point sample of ``n`` indices, starting from a random point."""
    points = np.asarray(points, dtype=float)
    rng = np.random.default_rng(seed)
    selected = [int(rng.integers(len(points)))]
    d2 = np.sum((points - points[selected[0]]) ** 2, axis=1)
    for _ in range(1, min(n, len(points))):
        selected.append(int(np.argmax(d2)))
        d2 = np.minimum(d2, np.sum((points - points[selected[-1]]) ** 2, axis=1))
    return np.asarray(selected, dtype=int)


def load_observations(truth: dict, data_dir: Path = DATA_DIR) -> dict:
    """The sparse observations the fits see, rebuilt from the indices 03a saved.

    Bedding normals are signed upward (younging up). ``sp``/``se``/``sr`` are the
    training strain cells; ``strain_test_idx`` are held out.
    """
    obs = np.load(Path(data_dir) / OBSERVATIONS)
    horizons = [trace[idx] for trace, idx in zip(truth["horizons_full"], obs["horizon_idx"])]
    gp = truth["gradient_xy"][obs["gradient_idx"]].copy()
    gv = truth["gradient_vector"][obs["gradient_idx"]].copy()
    gv[gv[:, 1] < 0] *= -1
    train, test = obs["strain_train_idx"], obs["strain_test_idx"]
    train_mask = np.zeros(len(truth["strain_xy"]), dtype=bool)
    train_mask[train] = True
    return dict(
        horizons=horizons, gp=gp, gv=gv,
        strain_train_idx=train, strain_test_idx=test, strain_train_mask=train_mask,
        sp=truth["strain_xy"][train], se=truth["strain_eigenvectors"][train],
        sr=truth["strain_eigenvalue_ratio"][train],
        seed=int(obs["seed"]),
    )


# ── Ensemble measures (numpy only) ──────────────────────────────────────────

def lithology_entropy(stack, classes):
    """Per-cell Shannon entropy (bits) of the members' lithology."""
    p = np.stack([(stack == c).mean(axis=0) for c in classes])
    return -(p * np.log2(np.clip(p, 1e-12, 1.0))).sum(axis=0)


def strain_envelope(B_stack, percentiles=(16, 84)):
    """Log-Euclidean mean strain shape over members, with ratio percentiles and axis coherence.

    ``B_stack`` is (members, points, 2, 2), unit determinant. Each member's shape
    is z = ln q (cos 2θ, sin 2θ), with q the stretch ratio and θ the major axis.
    """
    evals, evecs = np.linalg.eigh(B_stack)
    ln_q = 0.5 * np.log(np.clip(evals[..., 1] / evals[..., 0], 1.0, None))
    major = evecs[..., :, 1]
    two_theta = 2.0 * np.arctan2(major[..., 1], major[..., 0])
    z = ln_q[..., None] * np.stack([np.cos(two_theta), np.sin(two_theta)], axis=-1)

    z_mean = z.mean(axis=0)
    ln_q_mean = np.linalg.norm(z_mean, axis=-1)
    two_mean = np.arctan2(z_mean[:, 1], z_mean[:, 0])
    radial = np.einsum("mpk,pk->mp", z, np.stack([np.cos(two_mean), np.sin(two_mean)], axis=-1))
    radial_lo, radial_hi = np.percentile(radial, percentiles, axis=0)
    coherence = ln_q_mean / np.clip(np.linalg.norm(z, axis=-1).mean(axis=0), 1e-12, None)
    return {
        "theta": 0.5 * two_mean,
        "q": np.exp(ln_q_mean),
        "q_lo": np.exp(radial_lo),
        "q_hi": np.exp(radial_hi),
        "coherence": coherence,
        "log_ratio_sd": radial.std(axis=0),
    }


def restored_vertical_spread(nodes_restored, rows):
    """σx per horizon: the ensemble standard deviation of the restored x of each vertical marker
    line where it crosses that horizon, rms over the verticals.

    ``nodes_restored`` is (members, crossings, 2), the horizon–vertical crossings of
    :func:`grid_nodes` pulled back by every member; ``rows`` gives each crossing's horizon. Unlike
    the spread of restored width, which only sees a horizon's two ends, this follows material
    points across the whole section.
    """
    sd = np.asarray(nodes_restored, dtype=float)[..., 0].std(axis=0)
    rows = np.asarray(rows)
    return np.array([np.sqrt(np.mean(sd[rows == i] ** 2)) for i in np.unique(rows)])


def restored_horizon_spread(restored):
    """Ensemble spread of restored horizons: (members, horizons, samples, 2) in."""
    restored = np.asarray(restored, dtype=float)
    width = restored[..., 0].max(axis=2) - restored[..., 0].min(axis=2)
    depth = restored[..., 1].mean(axis=2)
    return {
        "mean": restored.mean(axis=0),
        "sigma_x": width.std(axis=0),
        "sigma_z": depth.std(axis=0),
        "width": width,
        "flatness": restored[..., 1].std(axis=2),
    }


def horizon_width(xy):
    xy = np.asarray(xy, dtype=float)
    return xy[..., 0].max(axis=-1) - xy[..., 0].min(axis=-1)


def bulk_shortening(present_traces, restored):
    """Shortening 1 - W_present/W_restored and S = (W_restored/W_present)^2 per member and horizon."""
    restored = np.asarray(restored, dtype=float)
    w_present = np.array([horizon_width(tr) for tr in present_traces])
    w_restored = horizon_width(restored)
    return {
        "W_present": w_present,
        "W_restored": w_restored,
        "shortening": 1.0 - w_present / np.clip(w_restored, 1e-12, None),
        "S": (w_restored / np.clip(w_present, 1e-12, None)) ** 2,
    }


def heldout_metrics(B_members, truth: dict, indices):
    """Per-member errors at the held-out cells.

    ``B_members`` is (members, cells, 2, 2), the unit-determinant spatial shape
    tensor at every digitised cell; only ``indices`` are scored. Returns the RMSE
    in log stretch ratio, the median major-axis error (degrees, 0–90) and the
    RMSE of the shape tensor.
    """
    q_true = truth["q_truth"][indices]
    frames = truth["strain_eigenvectors"][indices]
    major_true = frames[:, :, 1]
    B_true = np.stack([f @ np.diag([1.0 / q, q]) @ f.T for q, f in zip(q_true, frames)])
    out = {"log stretch RMSE": [], "axis error (degrees)": [], "shape-tensor RMSE": []}
    for B_all in B_members:
        B = B_all[indices]
        evals, evecs = np.linalg.eigh(B)
        q_pred = np.sqrt(np.clip(evals[:, 1] / evals[:, 0], 1.0, None))
        dots = np.abs(np.sum(evecs[:, :, 1] * major_true, axis=1))
        angle = np.degrees(np.arccos(np.clip(dots, 0.0, 1.0)))
        out["log stretch RMSE"].append(np.sqrt(np.mean((np.log(q_pred) - np.log(q_true)) ** 2)))
        out["axis error (degrees)"].append(np.median(angle))
        out["shape-tensor RMSE"].append(np.sqrt(np.mean((B - B_true) ** 2)))
    return {k: np.asarray(v) for k, v in out.items()}


# --- The vertical marker lines -----------------------------------------------------------------
# Hudleston's experiment carried vertical marker lines as well as the horizons. They are never
# fitted; where they cross the horizons they give material points across the whole section,
# whose restored positions measure the spread of the restorations (restored_vertical_spread).

def polyline_crossing(a, b):
    """First crossing of polylines ``a`` and ``b`` (both (n, 2)), or None."""
    p, r = a[:-1, None, :], (a[1:] - a[:-1])[:, None, :]
    q, s = b[None, :-1, :], (b[1:] - b[:-1])[None, :, :]
    cross = lambda u, v: u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0]
    den = cross(r, s)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = cross(q - p, s) / den
        u = cross(q - p, r) / den
    hit = (den != 0) & (t >= 0) & (t <= 1) & (u >= 0) & (u <= 1)
    if not hit.any():
        return None
    i, j = np.argwhere(hit)[0]
    return a[i] + t[i, j] * (a[i + 1] - a[i])


def grid_nodes(horizons, verticals):
    """Crossings of every horizon (row i, top to bottom) with every vertical (column j, left to right):
    (points, rows, columns)."""
    nodes, rows, cols = [], [], []
    for i, h in enumerate(horizons):
        for j, v in enumerate(verticals):
            x = polyline_crossing(np.asarray(h, float), np.asarray(v, float))
            if x is not None:
                nodes.append(x)
                rows.append(i)
                cols.append(j)
    return np.asarray(nodes), np.asarray(rows), np.asarray(cols)

