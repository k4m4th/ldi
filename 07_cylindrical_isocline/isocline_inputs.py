"""Example 07's data and comparison measures, shared by every notebook.

``07a_cylindrical_isocline_data`` reads the raw benchmark files, decides the fold
axis and the modelling frame, orders the dense contacts into the truth lines and
writes ``data/processed/``; the fitting, meshing and figure notebooks read it
through here. The measures compare a fitted scalar field with the truth and test
how cylindrical it is; they need only the field on the evaluation grid, so the
LDI and Surfe notebooks score their fits the same way. numpy, scipy and
contourpy only (no curlew, torch or surfepy).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

PROCESSED_DIR = Path("data/processed")
RESULTS_DIR = Path("results")


def unpack(xyz, offsets):
    """Ragged polylines stored as one array plus offsets: polyline i is ``xyz[off[i]:off[i + 1]]``."""
    return [xyz[a:b] for a, b in zip(offsets[:-1], offsets[1:])]


def pack(lines):
    """The inverse of :func:`unpack`."""
    offsets = np.r_[0, np.cumsum([len(l) for l in lines])]
    return (np.vstack(lines) if lines else np.empty((0, 2))), offsets


def load_processed(processed_dir: Path = PROCESSED_DIR) -> dict:
    """Everything 07a saved, in the modelling frame (add ``origin`` for the file frame).

    ``horizons`` are the sparse 3-D contacts, youngest first; ``gp``, ``gv`` the
    bedding and its younging normals. ``hinge`` is the fold axis (the π-axis of
    the bedding) and ``u_ax``, ``w_ax`` span the fold profile normal to it.
    ``truth_points[h]`` are horizon h's dense contacts in the profile,
    ``(u, w)``; ``truth_lines[h]`` the same points ordered into polylines.
    """
    processed_dir = Path(processed_dir)
    d = np.load(processed_dir / "isocline_processed.npz")
    names = [str(n) for n in d["horizon_names"]]
    n = len(names)
    return dict(
        origin=d["origin"], horizon_names=names, levels=d["levels"],
        horizons=[d[f"horizon_{h}"] for h in range(n)],
        gp=d["gp"], gv=d["gv"],
        hinge=d["hinge"], u_ax=d["u_ax"], w_ax=d["w_ax"],
        truth_points=[d[f"truth_points_{h}"] for h in range(n)],
        truth_lines=[unpack(d[f"truth_lines_{h}"], d[f"truth_offsets_{h}"]) for h in range(n)],
        box=d["box"],
        settings=json.loads((processed_dir / "settings.json").read_text()),
    )


def chain(points, max_jump):
    """Order scattered points along a curve into polylines.

    Greedy nearest-neighbour walk from the point furthest from the centroid,
    extended at both ends; a gap over ``max_jump`` ends the polyline and the
    walk restarts from the furthest unvisited point. Good when the spacing along
    the curve is well below the distance between neighbouring limbs.
    """
    points = np.asarray(points, dtype=float)
    tree = cKDTree(points)
    free = np.ones(len(points), dtype=bool)

    def walk(i):
        path = [i]
        while True:
            free[path[-1]] = False
            k = min(len(points), 16)
            while True:
                dist, idx = tree.query(points[path[-1]], k=k)
                ok = free[idx] & (dist <= max_jump)
                if ok.any() or k == len(points) or dist[-1] > max_jump:
                    break
                k = min(len(points), 2 * k)
            if not ok.any():
                return path
            path.append(int(idx[np.argmax(ok)]))

    lines = []
    centre = points.mean(0)
    while free.any():
        cand = np.flatnonzero(free)
        start = int(cand[np.argmax(np.linalg.norm(points[cand] - centre, axis=1))])
        forward = walk(start)
        free[start] = True                    # walk the other way from the same start
        backward = walk(start)
        line = backward[::-1] + forward[1:]
        if len(line) >= 2:
            lines.append(points[line])
    return lines


def to_profile(xyz, u_ax, w_ax):
    """3-D points to fold-profile coordinates ``(u, w)``."""
    xyz = np.asarray(xyz, dtype=float)
    return np.c_[xyz @ u_ax, xyz @ w_ax]


def section_contours(a, b, field, level):
    """Isolines ``field = level`` of a section; ``field`` has shape ``(len(a), len(b))``."""
    from contourpy import LineType, contour_generator

    gen = contour_generator(x=np.asarray(a, float), y=np.asarray(b, float),
                            z=np.asarray(field, float).T, line_type=LineType.Separate)
    return [np.asarray(s, dtype=float) for s in gen.lines(float(level)) if len(s) >= 2]


def distance_to_polylines(points, lines):
    """Distance from each point to the nearest segment of ``lines`` (inf if there are none)."""
    points = np.asarray(points, dtype=float)
    if not lines:
        return np.full(len(points), np.inf)
    a = np.vstack([l[:-1] for l in lines])
    b = np.vstack([l[1:] for l in lines])
    ab = b - a
    best = np.full(len(points), np.inf)
    for s in range(0, len(a), 4096):
        A, AB = a[s:s + 4096], ab[s:s + 4096]
        t = ((points[:, None] - A[None]) * AB[None]).sum(-1) / np.maximum((AB ** 2).sum(-1), 1e-12)[None]
        q = A[None] + np.clip(t, 0.0, 1.0)[..., None] * AB[None]
        best = np.minimum(best, np.linalg.norm(points[:, None] - q, axis=-1).min(1))
    return best


def contact_misfit(x, y, z, phi, isovalues, truth_points, sections):
    """How far a fitted field's contacts lie from the truth, on sections along the hinge.

    For each section (the grid's x–z plane nearest each ``sections`` y) and each
    horizon, the field's isolines at that horizon's isovalue are compared with
    the horizon's dense truth contacts: ``to_model`` is the distance from each
    truth contact to the nearest isoline (a contact the model misses scores
    far), ``to_truth`` from each isoline vertex to the nearest truth contact (an
    isoline the truth does not have, e.g. a bubble, scores far). Valid for a
    hinge along y (07a checks this). Returns per-section, per-horizon medians
    and 90th percentiles, and the pooled distances.
    """
    out = {k: np.full((len(sections), len(isovalues)), np.nan)
           for k in ("to_model_median", "to_model_p90", "to_truth_median", "to_truth_p90")}
    pooled = {"to_model": [], "to_truth": []}
    trees = [cKDTree(t) for t in truth_points]
    for s, ys in enumerate(sections):
        j = int(np.argmin(np.abs(np.asarray(y) - ys)))
        for h, level in enumerate(isovalues):
            lines = section_contours(x, z, phi[:, j, :], level)
            d_model = distance_to_polylines(truth_points[h], lines)
            d_truth = trees[h].query(np.vstack(lines))[0] if lines else np.array([np.inf])
            out["to_model_median"][s, h], out["to_model_p90"][s, h] = np.median(d_model), np.percentile(d_model, 90)
            out["to_truth_median"][s, h], out["to_truth_p90"][s, h] = np.median(d_truth), np.percentile(d_truth, 90)
            pooled["to_model"].append(d_model)
            pooled["to_truth"].append(d_truth)
    out["to_model"] = np.concatenate(pooled["to_model"])
    out["to_truth"] = np.concatenate(pooled["to_truth"])
    out["sections"] = np.asarray(sections, dtype=float)
    return out


def closed_loops(x, y, z, phi, isovalues, sections, min_extent=None):
    """Closed isocontours of a fitted field on sections along the hinge: the bubbles.

    A stratigraphic contact in a section runs from edge to edge (or ends at the data's
    edge); an isoline that closes on itself is a pod, an interior extremum of the field.
    Counts them per section and horizon at that horizon's isovalue. Loops smaller than
    ``min_extent`` (default two grid steps) in both directions are not counted: a thin
    fold tip a cell wide can close a contour of a field with no extremum, an artefact of
    the grid. Valid for a hinge along y. Returns ``(len(sections), len(isovalues))`` integers.
    """
    x, z = np.asarray(x, dtype=float), np.asarray(z, dtype=float)
    if min_extent is None:
        min_extent = 2.0 * max(float(np.mean(np.diff(x))), float(np.mean(np.diff(z))))
    out = np.zeros((len(sections), len(isovalues)), dtype=int)
    for s, ys in enumerate(sections):
        j = int(np.argmin(np.abs(np.asarray(y) - ys)))
        for h, level in enumerate(isovalues):
            out[s, h] = sum(bool(np.allclose(line[0], line[-1])) and float(np.ptp(line, axis=0).max()) >= min_extent
                            for line in section_contours(x, z, phi[:, j, :], level))
    return out


def dip_error(x, y, z, phi, truth_lines, sections):
    """Angle between the fitted gradient and the true contact normal, along the dense true contacts.

    On each section (the grid's x–z plane nearest each ``sections`` y) the fit's ∇φ is
    interpolated to every vertex of every true contact and compared with that contact's
    normal. The sign of φ is arbitrary, so the majority sign counts as the right one;
    points against it are the reversed share. Unlike the bedding readings, the true
    contacts cover the whole fold. Valid for a hinge along y. Returns the pooled angles
    in degrees (0-180) and ``reversed``, the share of points past 90 degrees.
    """
    from scipy.interpolate import RegularGridInterpolator
    from scipy.ndimage import gaussian_filter1d

    x, z = np.asarray(x, dtype=float), np.asarray(z, dtype=float)
    cosines = []
    for ys in sections:
        j = int(np.argmin(np.abs(np.asarray(y) - ys)))
        gx, gz = np.gradient(np.asarray(phi[:, j, :], dtype=float), x, z)
        fx = RegularGridInterpolator((x, z), gx, bounds_error=False, fill_value=np.nan)
        fz = RegularGridInterpolator((x, z), gz, bounds_error=False, fill_value=np.nan)
        for lines in truth_lines:
            for line in lines:
                p = np.asarray(line, dtype=float)
                t = np.gradient(gaussian_filter1d(p, 1, axis=0), axis=0)
                t /= np.linalg.norm(t, axis=1, keepdims=True).clip(min=1e-12)
                n = np.c_[-t[:, 1], t[:, 0]]
                g = np.c_[fx(p), fz(p)]
                ok = np.isfinite(g).all(axis=1) & (np.linalg.norm(g, axis=1) > 0)
                cosines.append((g[ok] * n[ok]).sum(axis=1) / np.linalg.norm(g[ok], axis=1))
    cos = np.concatenate(cosines)
    cos = cos * (np.sign(np.median(cos)) or 1.0)
    return dict(angle_deg=np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))), reversed=float(np.mean(cos < 0)))


def unit_iou(x, y, z, phi, isovalues, truth_lines, sections, step=2.0):
    """Intersection over union of each unit between two horizons with the true one, on sections along the hinge.

    The truth lines are open, so a true unit is known only between two consecutive
    horizons: the polygon closed by horizon h's line and horizon h + 1's, reversed. On
    each section (the grid's x–z plane nearest each ``sections`` y), φ is sampled on a
    ``step`` m raster and the fit's unit between the same two isovalues is compared with
    that polygon, inside the union of the polygons (fitted cells outside every polygon
    are not scored). Valid for a hinge along y. Returns ``(len(sections), n_horizons - 1)``,
    units youngest first (H0–H1, H1–H2, ...).
    """
    from matplotlib.path import Path as MPath
    from scipy.interpolate import RegularGridInterpolator

    for h, lines in enumerate(truth_lines):
        if len(lines) != 1:
            raise ValueError(f"horizon {h} has {len(lines)} truth lines; the bands need one each")
    x, z = np.asarray(x, dtype=float), np.asarray(z, dtype=float)
    gx, gz = np.meshgrid(np.arange(x[0], x[-1] + step / 2, step), np.arange(z[0], z[-1] + step / 2, step),
                         indexing="ij")
    pts = np.c_[gx.ravel(), gz.ravel()]
    bands = [MPath(np.r_[truth_lines[h][0], truth_lines[h + 1][0][::-1]]).contains_points(pts)
             for h in range(len(truth_lines) - 1)]
    window = np.logical_or.reduce(bands)
    out = np.full((len(sections), len(bands)), np.nan)
    for s, ys in enumerate(sections):
        j = int(np.argmin(np.abs(np.asarray(y) - ys)))
        f = RegularGridInterpolator((x, z), np.asarray(phi[:, j, :], dtype=float))(pts)
        for h, band in enumerate(bands):
            lo, hi = sorted(np.asarray(isovalues, dtype=float)[h:h + 2])
            fit = (f >= lo) & (f < hi) & window
            out[s, h] = np.logical_and(fit, band).sum() / np.logical_or(fit, band).sum()
    return out


def units(phi, isovalues):
    """Unit index of each value: 0 below the lowest isovalue, then one more past each."""
    return np.digitize(phi, np.sort(np.asarray(isovalues, dtype=float)))


def unit_change_along_hinge(phi, isovalues, axis=1):
    """Fraction of each grid line along the hinge (grid ``axis``) whose unit differs from the line's commonest.

    Zero everywhere for a cylindrical model with its axis along the grid axis.
    Returns the fraction per line (the other two grid axes).
    """
    u = np.moveaxis(units(phi, isovalues), axis, -1)
    n_units = len(isovalues) + 1
    counts = np.stack([(u == k).sum(-1) for k in range(n_units)], axis=-1)
    return 1.0 - counts.max(-1) / u.shape[-1]


def hinge_alignment(gradient, hinge):
    """``|∇φ · h| / |∇φ|`` at each point: 0 where the level surface contains the hinge."""
    g = np.asarray(gradient, dtype=float).reshape(-1, 3)
    norm = np.linalg.norm(g, axis=1)
    keep = norm > 1e-12
    return np.abs(g[keep] @ np.asarray(hinge, dtype=float)) / norm[keep]


def summarise(misfit, change, alignment, iou=None):
    """The comparison numbers saved to ``<method>.json``; ``iou`` from :func:`unit_iou`."""
    finite = lambda v: v[np.isfinite(v)]
    extra = {} if iou is None else {"unit_iou_median": np.median(iou, axis=0).tolist()}
    return {
        "contact_to_model_m_median": float(np.median(misfit["to_model"])),
        "contact_to_model_m_p90": float(np.percentile(misfit["to_model"], 90)),
        "contact_missed_fraction": float(np.mean(~np.isfinite(misfit["to_model"]))),
        "isoline_to_truth_m_median": float(np.median(finite(misfit["to_truth"]))),
        "isoline_to_truth_m_p90": float(np.percentile(finite(misfit["to_truth"]), 90)),
        "unit_change_along_hinge_mean": float(change.mean()),
        "unit_change_along_hinge_lines_affected": float(np.mean(change > 0)),
        "hinge_alignment_median": float(np.median(alignment)),
        "hinge_alignment_p95": float(np.percentile(alignment, 95)),
        **extra,
    }


def figure_section(x, y, z, horizons, pad):
    """The section the figure draws: at the grid y nearest the middle of the contacts, with x
    widened by ``pad`` on the -x side (same step), as the renders are. Returns ``(xs, y, points)``,
    the points ordered ``(len(xs), len(z))``."""
    x, z = np.asarray(x, dtype=float), np.asarray(z, dtype=float)
    ys = float(y[np.argmin(np.abs(np.asarray(y) - np.median(np.vstack(horizons)[:, 1])))])
    step = x[1] - x[0]
    n = int(np.ceil(pad / step))
    xs = np.r_[x[0] - step * np.arange(n, 0, -1), x]
    return xs, ys, np.stack(np.meshgrid(xs, [ys], z, indexing="ij"), axis=-1).reshape(-1, 3)
