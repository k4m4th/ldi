"""Example 06's data and fit measures, shared by every notebook.

``06a_claudius_data`` reads the ImplicitBenchmark files in ``data/``, shifts them
to the box centre, turns the bedding poles to point up (younging) and writes
``data/processed/``, with the evaluation grid; the LDI and Surfe notebooks read
it through here. A fit keeps a fraction of each horizon's contacts
(``uniform_keep``), spaced evenly so the kept points do not clump, and the rest
is held out to score it. The benchmark's random ``Cutoff`` is stored but not
used for that split. There is no truth: the held-out contacts are the test.
numpy only (no curlew, torch or surfepy).

A fit can instead keep the contacts along hypothetical 2-D seismic lines (``seismic_keep``): vertical
lines of constant x (running east-west) and of constant y (north-south) across the box. Everything between
the lines is held out, so the score is how well the fit interpolates across the gaps. The bedding is
kept whole either way.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

PROCESSED_DIR = Path("data/processed")
RESULTS_DIR = Path("results")


def uniform_keep(points, fraction) -> np.ndarray:
    """Indices keeping ``fraction`` of ``points``, spaced evenly through the cloud.

    Farthest-point sampling in x, y, z: start at the point nearest the centroid
    and repeatedly add the point furthest from those already kept. Dense clumps
    therefore contribute one point, not a share of the fraction. The same cloud
    and fraction always gives the same subset. ``fraction`` of 1 keeps every
    point, in order.
    """
    points = np.asarray(points, dtype=float)
    n = len(points)
    if float(fraction) >= 1.0 or n == 0:
        return np.arange(n)
    n_keep = int(np.clip(np.round(float(fraction) * n), 0, n))
    if n_keep == 0:
        return np.empty(0, dtype=int)
    chosen = np.empty(n_keep, dtype=int)
    delta = points - points.mean(0)
    chosen[0] = int(np.argmin(np.einsum("ij,ij->i", delta, delta)))
    nearest = np.full(n, np.inf)
    for i in range(n_keep - 1):
        delta = points - points[chosen[i]]
        nearest = np.minimum(nearest, np.einsum("ij,ij->i", delta, delta))
        nearest[chosen[i]] = -1.0
        chosen[i + 1] = int(np.argmax(nearest))
    return np.sort(chosen)


SEISMIC_LINES = dict(n_x=5, n_y=5, half_width=20.0, bounds=True)   # 5 inner lines of each direction plus the two bounding
#                                                                    # lines (the extent of the picks); contacts within 20 m of a line


def pick_extent(points) -> np.ndarray:
    """``[[x_min, y_min], [x_max, y_max]]`` of the picks of every horizon pooled: the survey's own outline,
    a little inside the evaluation box."""
    p = np.vstack([np.asarray(q, dtype=float)[:, :2] for q in points])
    return np.array([p.min(axis=0), p.max(axis=0)])


def seismic_lines(extent, n_x, n_y, bounds=True):
    """Positions of the lines: ``n_x`` inner lines of constant x and ``n_y`` of constant y, evenly spaced across
    ``extent`` (``[[x_min, y_min], [x_max, y_max]]``, see ``pick_extent``), and with ``bounds`` the two bounding
    lines of each direction, at the extent itself, so the outermost picks are kept and the survey is closed."""
    def positions(lo, hi, n):
        inner = np.linspace(lo, hi, n + 2)[1:-1]
        return np.r_[lo, inner, hi] if bounds else inner
    return positions(extent[0, 0], extent[1, 0], n_x), positions(extent[0, 1], extent[1, 1], n_y)


def seismic_keep(points, line_x, line_y, half_width) -> np.ndarray:
    """Boolean mask of the ``points`` within ``half_width`` m (in plan) of any line of constant x or constant y.

    The picks sit on a lattice about 40 m apart, so a half-width of 20 m keeps one row of picks along each line.
    """
    points = np.asarray(points, dtype=float)
    keep = np.zeros(len(points), dtype=bool)
    for a in np.atleast_1d(line_x):
        keep |= np.abs(points[:, 0] - a) <= half_width
    for b in np.atleast_1d(line_y):
        keep |= np.abs(points[:, 1] - b) <= half_width
    return keep


def line_bedding(points, line_x, line_y, half_width, every=3, intersection_tol=None):
    """Bedding from the contacts along the seismic lines, taken as example 04 takes it from its digitised polylines.

    ``points`` is one array per horizon. On each line (of constant x or of constant y) the picks within ``half_width``
    are ordered along the line; the tangent at each is the central difference between its neighbours, in the vertical
    plane of the line, and the unit normal is that tangent turned 90°, pointing up. Every ``every``-th pick is kept,
    and none within ``intersection_tol`` (default twice ``half_width``) of a line of the other direction, where the two
    lines' apparent normals meet at one place. A normal is the gradient's direction projected on the line's plane (an
    apparent dip), not the true 3-D normal: it is for the spectral band, not for the fit's bedding constraints.

    Returns ``(gp, gv)``, positions and 3-D unit normals, both ``(n, 3)``.
    """
    tol = 2.0 * half_width if intersection_tol is None else intersection_tol
    gp, gv = [], []
    for positions, across, along, others in ((line_x, 0, 1, line_y), (line_y, 1, 0, line_x)):
        for position in np.atleast_1d(positions):
            for h in points:
                p = np.asarray(h, dtype=float)
                near = p[np.abs(p[:, across] - position) <= half_width]
                if len(near) < 4:
                    continue
                near = near[np.argsort(near[:, along])]
                xy = np.c_[near[:, along], near[:, 2]]
                tang = np.empty_like(xy)
                tang[1:-1] = xy[2:] - xy[:-2]
                tang[0], tang[-1] = xy[1] - xy[0], xy[-1] - xy[-2]
                n2 = np.c_[-tang[:, 1], tang[:, 0]]
                n2 /= np.linalg.norm(n2, axis=1, keepdims=True).clip(min=1e-8)
                n2[n2[:, 1] < 0] *= -1.0
                keep = np.zeros(len(near), dtype=bool)
                keep[::every] = True
                for other in np.atleast_1d(others):
                    keep &= np.abs(near[:, along] - other) > tol
                n3 = np.zeros((len(near), 3))
                n3[:, along], n3[:, 2] = n2[:, 0], n2[:, 1]
                gp.append(near[keep])
                gv.append(n3[keep])
    return np.vstack(gp), np.vstack(gv)


def sampling_tag(cutoff: float = 1.0, sampling: str = "uniform", lines: dict | None = None) -> str:
    """The name part a fit's outputs share: ``cut0.4``, or ``seis5x5b_w20`` for 5 + 5 inner lines and the bounding
    lines (``b``) within 20 m."""
    if sampling == "uniform":
        return f"cut{float(cutoff):g}"
    if sampling == "seismic":
        lines = SEISMIC_LINES if lines is None else lines
        return f"seis{lines['n_x']}x{lines['n_y']}{'b' if lines.get('bounds', True) else ''}_w{lines['half_width']:g}"
    raise ValueError(f"sampling must be 'uniform' or 'seismic', got {sampling!r}")


def load_processed(cutoff: float = 1.0, processed_dir: Path = PROCESSED_DIR, sampling: str = "uniform",
                   lines: dict | None = None) -> dict:
    """Everything 06a saved, in the modelling frame (add ``origin`` for UTM).

    ``sampling="uniform"`` (default): ``cutoff`` is the fraction of each horizon kept for the fit, by
    ``uniform_keep``. ``sampling="seismic"``: the contacts within ``lines["half_width"]`` of
    ``lines["n_x"]`` inner lines of constant x and ``lines["n_y"]`` of constant y, and the bounding lines (the extent of the
    picks) unless ``lines["bounds"]`` is False, are kept, by ``seismic_keep``
    (``cutoff`` is then unused; ``lines`` defaults to ``SEISMIC_LINES``). ``horizons[h]`` is that
    subset and ``held_out[h]`` the rest, youngest (A) first. ``line_x``, ``line_y`` are the lines'
    positions (empty for uniform sampling) and ``tag`` the name part for the outputs. ``gp``, ``gv`` are the bedding
    and its younging normals (all of them; there are few). ``x``, ``y``, ``z`` are
    the evaluation grid's axes and ``box`` its ``[[lo], [hi]]``. ``fault`` is the
    benchmark's fault (points), which the fits leave out.
    """
    processed_dir = Path(processed_dir)
    d = np.load(processed_dir / "claudius_processed.npz")
    names = [str(n) for n in d["horizon_names"]]
    points = [d[f"horizon_{h}"] for h in range(len(names))]
    lines = (SEISMIC_LINES if lines is None else lines) if sampling == "seismic" else None
    line_x, line_y = (seismic_lines(pick_extent(points), lines["n_x"], lines["n_y"], lines.get("bounds", True)) if lines
                      else (np.empty(0), np.empty(0)))
    horizons, held_out = [], []
    for p in points:
        if sampling == "seismic":
            keep = seismic_keep(p, line_x, line_y, lines["half_width"])
        else:
            keep = np.zeros(len(p), dtype=bool)
            keep[uniform_keep(p, cutoff)] = True
        horizons.append(p[keep])
        held_out.append(p[~keep])
    return dict(
        origin=d["origin"], horizon_names=names, cutoff=float(cutoff),
        sampling=sampling, lines=lines, line_x=line_x, line_y=line_y, tag=sampling_tag(cutoff, sampling, lines),
        horizons=horizons, held_out=held_out,
        gp=d["gp"], gv=d["gv"], fault=d["fault"],
        x=d["x"], y=d["y"], z=d["z"], box=d["box"],
        settings=json.loads((processed_dir / "settings.json").read_text()),
    )


def level_distance(values, gradients, isovalue):
    """First-order distance from points to the level surface ``isovalue``: ``|phi - iso| / |grad phi|``."""
    return np.abs(np.asarray(values) - isovalue) / np.maximum(np.linalg.norm(gradients, axis=1), 1e-12)


def summarise_distances(fit, held_out) -> dict:
    """Median and 90th percentile of the per-horizon distances, pooled, for the fit and held-out contacts."""
    out = {}
    for key, d in (("fit", fit), ("held_out", held_out)):
        d = np.concatenate([np.asarray(v) for v in d]) if len(d) else np.empty(0)
        out[f"{key}_m_median"] = float(np.median(d)) if d.size else float("nan")
        out[f"{key}_m_p90"] = float(np.percentile(d, 90)) if d.size else float("nan")
    return out
