"""Example 01's section data and evaluation grid, built the same way in every notebook.

``01b_recumbent_fold_fit`` (curlew) and ``01c_recumbent_fold_surfe`` (Surfe) both
read the data through here, so they fit the same contacts and bedding in the
same shifted frame and evaluate on the same grid. With synthetic data, every
fit is scored against the truth here too (:func:`truth_reference`,
:func:`score_against_truth`). numpy, pandas and scipy only.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

SECTION_Y = 500.0           # m, the section the orientations lie on
GRID_HALF, GRID_STEP = 600.0, 5.0  # m, evaluation grid: [-600, 595] in x and z, centred on the data

CSV = {
    "synthetic": ("synthetic/surface_points.csv", "synthetic/orientations.csv"),
}


def normal_from_azimuth_dip(azimuth_deg, dip_deg):
    """Unit normal in GemPy coords (x, y, z up)."""
    az = np.deg2rad(np.asarray(azimuth_deg, dtype=float))
    dp = np.deg2rad(np.asarray(dip_deg, dtype=float))
    return np.stack([np.sin(az) * np.sin(dp), np.cos(az) * np.sin(dp), np.cos(dp)], axis=-1)


def load_section(dataset: str, data_dir: Path = Path("./data")) -> dict:
    """Contacts and bedding of the ``Y = SECTION_Y`` section, shifted to their centroid.

    Horizons are younger first: ``H0`` is ``rock2``, ``H1`` is ``rock1``.
    """
    sp_csv, ori_csv = (Path(data_dir) / f for f in CSV[dataset])
    sp = pd.read_csv(sp_csv)
    ori = pd.read_csv(ori_csv)
    section = sp[np.isclose(sp["Y"], SECTION_Y)]
    formations = sorted(section["formation"].astype(str).unique())
    assert formations == ["rock1", "rock2"], formations
    horizons = [section.loc[section["formation"] == f, ["X", "Z"]].to_numpy(dtype=float) for f in formations]
    assert all(len(h) >= 3 for h in horizons), [len(h) for h in horizons]
    horizons = horizons[::-1]  # younger (rock2) first

    gp = ori[["X", "Z"]].to_numpy(dtype=float)
    gv = normal_from_azimuth_dip(ori["azimuth"], ori["dip"])[:, (0, 2)]
    gv /= np.linalg.norm(gv, axis=1, keepdims=True)

    origin = np.vstack(horizons + [gp]).mean(axis=0)
    return dict(
        horizon_names=[f"H{i}" for i in range(len(horizons))],
        horizons=[h - origin for h in horizons],
        gp=gp - origin, gv=gv, origin=origin,
        sp_csv=sp_csv, ori_csv=ori_csv,
    )


def grid_axes():
    """x and z axes of the evaluation grid (shifted frame)."""
    a = np.arange(-GRID_HALF, GRID_HALF, GRID_STEP)
    return a, a.copy()


def true_units(x, z, origin, data_dir: Path = Path("./data")):
    """True units of the synthetic fold on the grid ``x`` by ``z`` (shifted frame), shape (nx, nz).

    1 below rock1 (oldest), 2 between, 3 above rock2 (youngest).
    """
    from synthetic_fold import from_settings

    fold = from_settings(json.loads((Path(data_dir) / "synthetic" / "settings.json").read_text()))
    gx, gz = np.meshgrid(np.asarray(x) + origin[0], np.asarray(z) + origin[1], indexing="ij")
    return fold.units(gx, gz)


def truth_reference(origin, data_dir: Path = Path("./data"), margin=None) -> dict:
    """What a synthetic fit is scored against, in the shifted frame.

    ``holes``: the drillholes' x. ``window``: the x range the holes constrain, from the
    outermost holes widened by ``margin`` m on each side (default half the mean hole
    spacing). ``contacts``: the true contacts, younger first (``H0`` = rock2, ``H1`` =
    rock1), as dense polylines.
    """
    data_dir = Path(data_dir)
    settings = json.loads((data_dir / "synthetic" / "settings.json").read_text())
    holes = np.sort(np.asarray(settings["sampling"]["holes"], dtype=float)) - origin[0]
    if margin is None:
        margin = 0.5 * float(np.diff(holes).mean())
    truth = np.load(data_dir / "synthetic" / "truth.npz")
    contacts = [truth["contact_rock2"] - origin, truth["contact_rock1"] - origin]
    return dict(holes=holes, window=(holes[0] - margin, holes[-1] + margin), contacts=contacts)


def match_units(lithology, unit_true):
    """A fit's lithology IDs as true units (1 oldest .. 3 youngest): flip the order if it
    runs against the truth."""
    lithology = np.asarray(lithology, dtype=int)
    n = int(unit_true.max())
    corr = np.corrcoef(lithology.ravel(), unit_true.ravel())[0, 1]
    return n + 1 - lithology if corr < 0 else lithology


def _boundary(units, x, z, top):
    """Vertices of the boundary between the units below ``top`` and ``top`` and above."""
    from skimage.measure import find_contours

    mask = (units >= top).astype(float)
    pts = [np.c_[np.interp(c[:, 0], np.arange(len(x)), x), np.interp(c[:, 1], np.arange(len(z)), z)]
           for c in find_contours(mask, 0.5)]
    return np.vstack(pts) if pts else np.empty((0, 2))


def score_against_truth(units, x, z, unit_true, reference) -> dict:
    """A fit's geometry against the truth, where the data can decide it.

    ``units`` are the fit's units on the grid ``x`` by ``z`` (from :func:`match_units`).

    - ``contact_distance``: per horizon (younger first), the median distance between the
      fit's contact and the true one inside ``reference["window"]``, both ways round (true
      to fit and fit to true, pooled), so a contact that is missing and one that is extra
      both count. In the units of ``x`` (m);
    - ``iou``: per unit, intersection over union with the true unit inside the window.
    """
    from scipy.spatial import cKDTree

    units = np.asarray(units, dtype=int)
    n = int(unit_true.max())
    lo, hi = reference["window"]
    inside_x = (x >= lo) & (x <= hi)
    distance = []
    for k, true_line in enumerate(reference["contacts"]):       # H0 sits on unit n, H1 on unit n - 1
        top = n - k
        fit_line = _boundary(units, x, z, top)
        true_line = true_line[(true_line[:, 0] >= lo) & (true_line[:, 0] <= hi)
                              & (true_line[:, 1] >= z[0]) & (true_line[:, 1] <= z[-1])]
        fit_line = fit_line[(fit_line[:, 0] >= lo) & (fit_line[:, 0] <= hi)]
        if not len(fit_line) or not len(true_line):
            distance.append(float("nan"))
            continue
        both = np.r_[cKDTree(fit_line).query(true_line)[0], cKDTree(true_line).query(fit_line)[0]]
        distance.append(float(np.median(both)))
    iou = []
    for u in range(1, n + 1):
        a, b = units[inside_x] == u, unit_true[inside_x] == u
        union = np.logical_or(a, b).sum()
        iou.append(float(np.logical_and(a, b).sum() / union) if union else float("nan"))
    return dict(contact_distance=distance, iou=iou)
