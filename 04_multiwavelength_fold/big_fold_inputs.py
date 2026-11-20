"""Example 04's digitised fold train and observations, read the same way in every notebook.

``04a_multiwavelength_fold_data`` writes the observations, ``04b_multiwavelength_fold_fit``
(curlew) fits them and ``04c_multiwavelength_fold_figures`` draws the results.
numpy only.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

DATA_DIR = Path("data")
CONTACTS = "big_fold_contacts.csv"            # ImplicitBenchmark BigFoldDigitized.svg, y already flipped
OBSERVATIONS = "big_fold_observations.npz"
TRUE_UNDEFORMED, TRUE_FOLDED = 2141.0, 1546.0  # m, the experiment's layer length before and after
INTEGRATORS = {"shear": 0, "rk4": 5}         # integrator -> n_steps


def results_dir(integrator: str, eq_form: str = "variance") -> Path:
    """Where a fit is saved: ``results/<integrator>``, or ``results/<integrator>_<eq_form>``
    for an equality term other than the variance (``results/shear_log``, ...)."""
    if integrator not in INTEGRATORS:
        raise ValueError(f"integrator must be one of {sorted(INTEGRATORS)}, got {integrator!r}")
    return Path("results") / (integrator if eq_form == "variance" else f"{integrator}_{eq_form}")


def figure_suffix(integrator: str) -> str:
    """Figures from the shear fit carry no suffix; others are tagged, e.g. ``_rk4``."""
    return "" if integrator == "shear" else f"_{integrator}"


def load_observations(data_dir: Path = DATA_DIR) -> dict:
    """The six horizons (full traces and the clumps fitted), the bedding and the origin, H0 youngest."""
    d = np.load(Path(data_dir) / OBSERVATIONS)
    names = [str(n) for n in d["horizon_names"]]
    return dict(
        origin=d["origin"],
        horizon_names=names,
        source_names=[str(n) for n in d["source_names"]],
        horizons_full=[d[f"full_{i}"] for i in range(len(names))],
        horizons=[d[f"clumps_{i}"] for i in range(len(names))],
        gp=d["gp"], gv=d["gv"],
        clump_size=int(d["clump_size"]),
    )


def split_clumps(points, clump_size):
    """A horizon's equality points as its separate clumps (each ``clump_size`` consecutive points)."""
    points = np.asarray(points)
    return [points[i:i + clump_size] for i in range(0, len(points), clump_size)]


def younging_scale(zlim, fraction=0.05):
    """T-bar length for ``figstyle.overlay_constraints`` as a fraction of section height."""
    return 1.0 / max(fraction * abs(float(zlim[1]) - float(zlim[0])), 1e-8)


def width(xy) -> float:
    return float(np.ptp(np.asarray(xy)[:, 0]))


def arc_length(xy) -> float:
    xy = np.asarray(xy, dtype=float)
    return float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum()) if len(xy) > 1 else 0.0


def layer_shortening(present, restored) -> dict:
    """Per horizon: present width W, restored width W0, present arc length, 1 - W/W0 and 1 - W/arc (%)."""
    w = np.array([width(h) for h in present])
    w0 = np.array([width(r) for r in restored])
    arc = np.array([arc_length(h) for h in present])
    return dict(W=w, W0=w0, arc=arc, from_restored=100 * (1 - w / w0), from_arc=100 * (1 - w / arc),
                benchmark=100 * (1 - TRUE_FOLDED / TRUE_UNDEFORMED))
