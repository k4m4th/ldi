"""Reusable diagnostics for the recumbent-fold reconstruction notebook.

Critical points are found from roots of the unnormalised scalar-field
gradient.  Raster-neighbour comparisons are deliberately not used.

``torch`` and ``curlew`` are optional. Everything that works on arrays or on a
plain gradient callable (``find_critical_points(None, x, z, gradient=...)``,
the Surfe helpers, contour packing, the drawing functions) needs only numpy
and matplotlib; only the helpers that take a curlew ``event`` or ``model``
need torch.
"""

from __future__ import annotations

import contextlib
import math
import os
import sys
from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
try:
    import torch
except ImportError:  # array-only use, e.g. publication figures or Surfe
    torch = None
from matplotlib.collections import EllipseCollection, LineCollection
from matplotlib.colors import LogNorm, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Arc, Circle, Patch
from matplotlib.patheffects import withStroke
from matplotlib import ticker

import figstyle


Array = np.ndarray


@dataclass(frozen=True)
class CriticalPointConfig:
    """Controls candidate generation, root refinement and classification."""

    root_tolerance: float = 1e-8
    residual_tolerance: float = 1e-5
    deduplication_distance: float | None = None
    hessian_step: float | None = None
    eigenvalue_tolerance: float = 1e-7
    boundary_margin: float | None = None
    batch_size: int = 65_536
    newton_iterations: int = 12
    stability_samples: int = 8_192


@dataclass
class CriticalPoints:
    """Refined critical points and their Hessian classifications."""

    points: Array
    kinds: Array
    residuals: Array
    eigenvalues: Array
    candidates: Array

    def of_kind(self, kind: str) -> Array:
        return self.points[self.kinds == kind]

    @property
    def counts(self) -> dict[str, int]:
        return {
            kind: int(np.count_nonzero(self.kinds == kind))
            for kind in ("maximum", "minimum", "saddle", "degenerate")
        }

    @property
    def extrema_count(self) -> int:
        return self.counts["maximum"] + self.counts["minimum"]


def _as_numpy(value) -> Array:
    if torch is not None and isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _require_torch() -> None:
    if torch is None:
        raise ImportError("this diagnostic evaluates a curlew event and needs torch")


def _as_points(points) -> Array:
    points = _as_numpy(points)
    if points.size == 0:
        dim = int(points.shape[-1]) if points.ndim >= 2 else 2
        return np.empty((0, dim), dtype=float)
    return np.asarray(points, dtype=float).reshape(-1, points.shape[-1])


def _field_device_dtype(field) -> tuple[torch.device, torch.dtype]:
    _require_torch()
    parameter = None
    for owner in (field, getattr(field, "integrator", None)):
        params = getattr(owner, "parameters", None)
        if not callable(params):
            continue
        parameter = next(iter(params()), None)
        if parameter is not None:
            break
    if parameter is None:
        return torch.device("cpu"), torch.get_default_dtype()
    return parameter.device, parameter.dtype


def _batched_tensor_call(
    function: Callable[[torch.Tensor], object],
    points: Array,
    *,
    field=None,
    batch_size: int = 65_536,
) -> Array:
    _require_torch()
    points = _as_points(points)
    if not len(points):
        return np.empty((0, points.shape[-1]), dtype=float)
    if field is None:
        device, dtype = torch.device("cpu"), torch.get_default_dtype()
    else:
        device, dtype = _field_device_dtype(field)
    values = []
    with torch.no_grad():
        for start in range(0, len(points), batch_size):
            tensor = torch.as_tensor(
                points[start : start + batch_size], device=device, dtype=dtype
            )
            value = function(tensor)
            if hasattr(value, "scalar"):
                value = value.scalar
            values.append(_as_numpy(value))
    return np.concatenate(values, axis=0)


def evaluate_model(
    model,
    points: Array,
    *,
    grid_shape: tuple[int, ...] | None = None,
    batch_size: int = 65_536,
) -> tuple[Array, Array]:
    """Evaluate scalar and lithology predictions in bounded batches."""
    _require_torch()

    scalar_parts, lithology_parts = [], []
    device, dtype = _field_device_dtype(model)
    points = _as_points(points)
    with torch.no_grad():
        for start in range(0, len(points), batch_size):
            tensor = torch.as_tensor(
                points[start : start + batch_size], device=device, dtype=dtype
            )
            prediction = model.predict(tensor, litho=True)
            scalar_parts.append(_as_numpy(prediction.scalar))
            scalar_parts[-1] = scalar_parts[-1].reshape(-1)
            lithology_parts.append(_as_numpy(prediction.lithoID).reshape(-1))
    scalar = np.concatenate(scalar_parts)
    lithology = np.concatenate(lithology_parts)
    if grid_shape is not None:
        scalar = scalar.reshape(grid_shape)
        lithology = lithology.reshape(grid_shape)
    return scalar, lithology


def _wrap_numpy_gradient(
    function: Callable[[Array], Array], dim: int = 2
) -> Callable[[Array], Array]:
    def gradient(points: Array) -> Array:
        values = function(np.atleast_2d(_as_points(points)))
        values = np.asarray(_as_numpy(values), dtype=float)
        if values.ndim == 1:
            values = values.reshape(1, -1)
        return values[:, :dim]

    return gradient


def make_gradient_function(event, *, dim: int = 2, batch_size: int = 65_536):
    """Return an unnormalised gradient callable with ``dim`` components."""

    field = event.field
    if hasattr(field, "stratigraphy_gradient"):
        call = field.stratigraphy_gradient
    elif hasattr(field, "gradient"):
        call = lambda x: field.gradient(
            x, transform=False, normalize=False, accumulate=False
        )
    else:
        raise TypeError(f"{type(field).__name__} exposes no supported gradient")

    def gradient(points: Array) -> Array:
        values = _batched_tensor_call(
            call, np.atleast_2d(points), field=field, batch_size=batch_size
        )
        return np.asarray(values, dtype=float)[:, :dim]

    return gradient


def evaluate_gradient(
    event,
    points: Array,
    *,
    batch_size: int = 65_536,
) -> Array:
    return make_gradient_function(event, batch_size=batch_size)(points)


def _value_and_unnormalised_gradient(field, tensor: torch.Tensor):
    """Scalar and unnormalised gradient from one field evaluation."""

    if not hasattr(field, "gradient"):
        raise TypeError(f"{type(field).__name__} exposes no supported gradient")
    gradient, value = field.gradient(
        tensor,
        transform=False,
        normalize=False,
        accumulate=False,
        return_value=True,
    )
    return value, gradient


def evaluate_fields(
    model,
    event,
    points: Array,
    *,
    grid_shape: tuple[int, ...] | None = None,
    batch_size: int = 65_536,
) -> tuple[Array, Array, Array]:
    """Evaluate scalar, lithology and unnormalised gradient in one batched pass."""
    _require_torch()

    field = event.field
    device, dtype = _field_device_dtype(field)
    points = _as_points(points)
    scalar_parts, lithology_parts, gradient_parts = [], [], []
    with torch.no_grad():
        for start in range(0, len(points), batch_size):
            tensor = torch.as_tensor(
                points[start : start + batch_size], device=device, dtype=dtype
            )
            value, gradient = _value_and_unnormalised_gradient(field, tensor)
            value = value.reshape(-1)
            prediction = model.predict(tensor, litho=True, values=value)
            scalar_parts.append(_as_numpy(value).reshape(-1))
            gradient_parts.append(np.asarray(_as_numpy(gradient), dtype=float)[:, :2])
            lithology_parts.append(_as_numpy(prediction.lithoID).reshape(-1))
    scalar = np.concatenate(scalar_parts)
    lithology = np.concatenate(lithology_parts)
    gradient = np.concatenate(gradient_parts, axis=0)
    if grid_shape is not None:
        scalar = scalar.reshape(grid_shape)
        lithology = lithology.reshape(grid_shape)
        gradient = gradient.reshape(*grid_shape, 2)
    return scalar, lithology, gradient


def winding_candidates(x: Array, z: Array, gradient_grid: Array) -> Array:
    """Return centres of grid cells whose gradient direction winds around zero."""

    gradient_grid = np.asarray(gradient_grid, dtype=float)
    if gradient_grid.shape != (len(x), len(z), 2):
        raise ValueError("gradient_grid must have shape (len(x), len(z), 2)")
    gx, gz = gradient_grid[..., 0], gradient_grid[..., 1]
    angles = np.arctan2(gz, gx)
    corners = (
        angles[:-1, :-1],
        angles[1:, :-1],
        angles[1:, 1:],
        angles[:-1, 1:],
    )
    winding = np.zeros((len(x) - 1, len(z) - 1), dtype=float)
    for first, second in zip(corners, corners[1:] + corners[:1]):
        winding += np.angle(np.exp(1j * (second - first)))
    finite = np.all(
        np.stack(
            [
                np.isfinite(gx[:-1, :-1]) & np.isfinite(gz[:-1, :-1]),
                np.isfinite(gx[1:, :-1]) & np.isfinite(gz[1:, :-1]),
                np.isfinite(gx[1:, 1:]) & np.isfinite(gz[1:, 1:]),
                np.isfinite(gx[:-1, 1:]) & np.isfinite(gz[:-1, 1:]),
            ]
        ),
        axis=0,
    )
    ii, jj = np.where(finite & (np.abs(winding) > np.pi))
    return np.column_stack(
        [0.5 * (x[ii] + x[ii + 1]), 0.5 * (z[jj] + z[jj + 1])]
    )


# Outward-oriented triangles on the six faces of a voxel. Vertex order is
# 000, 100, 010, 110, 001, 101, 011, 111.
_VOXEL_TRIANGLES = np.array(
    [
        [0, 4, 6],
        [0, 6, 2],
        [1, 3, 7],
        [1, 7, 5],
        [0, 1, 5],
        [0, 5, 4],
        [2, 6, 7],
        [2, 7, 3],
        [0, 2, 3],
        [0, 3, 1],
        [4, 5, 7],
        [4, 7, 6],
    ],
    dtype=int,
)


def _triangle_solid_angles(a: Array, b: Array, c: Array) -> Array:
    """Oriented solid angle of spherical triangles with unit vertices *a*, *b*, *c*."""

    numer = np.einsum("...i,...i->...", a, np.cross(b, c))
    denom = (
        1.0
        + np.einsum("...i,...i->...", a, b)
        + np.einsum("...i,...i->...", b, c)
        + np.einsum("...i,...i->...", c, a)
    )
    return 2.0 * np.arctan2(numer, denom)


def degree_candidates(x: Array, y: Array, z: Array, gradient_grid: Array) -> Array:
    """Return centres of voxels whose Gauss map has nonzero Brouwer degree at 0."""

    gradient_grid = np.asarray(gradient_grid, dtype=float)
    if gradient_grid.shape != (len(x), len(y), len(z), 3):
        raise ValueError("gradient_grid must have shape (len(x), len(y), len(z), 3)")
    corners = np.stack(
        [
            gradient_grid[:-1, :-1, :-1],
            gradient_grid[1:, :-1, :-1],
            gradient_grid[:-1, 1:, :-1],
            gradient_grid[1:, 1:, :-1],
            gradient_grid[:-1, :-1, 1:],
            gradient_grid[1:, :-1, 1:],
            gradient_grid[:-1, 1:, 1:],
            gradient_grid[1:, 1:, 1:],
        ],
        axis=-2,
    )
    mag = np.linalg.norm(corners, axis=-1, keepdims=True)
    finite = np.all(np.isfinite(corners), axis=(-1, -2)) & np.all(
        mag[..., 0] > 1e-30, axis=-1
    )
    unit = np.divide(corners, mag, where=mag > 1e-30, out=np.zeros_like(corners))
    a = unit[..., _VOXEL_TRIANGLES[:, 0], :]
    b = unit[..., _VOXEL_TRIANGLES[:, 1], :]
    c = unit[..., _VOXEL_TRIANGLES[:, 2], :]
    degree = _triangle_solid_angles(a, b, c).sum(axis=-1) / (4.0 * np.pi)
    ii, jj, kk = np.where(finite & (np.abs(degree) > 0.5))
    return np.column_stack(
        [
            0.5 * (x[ii] + x[ii + 1]),
            0.5 * (y[jj] + y[jj + 1]),
            0.5 * (z[kk] + z[kk + 1]),
        ]
    )


def _norm_local_minima(x: Array, y: Array, z: Array, gradient_grid: Array) -> Array:
    """Interior vertices that are 6-neighbour minima of ||∇φ||."""

    norm = np.linalg.norm(np.asarray(gradient_grid, dtype=float), axis=-1)
    local = np.ones(norm.shape, dtype=bool)
    local[0] = False
    local[-1] = False
    local[:, 0] = False
    local[:, -1] = False
    local[:, :, 0] = False
    local[:, :, -1] = False
    local[1:] &= norm[1:] <= norm[:-1]
    local[:-1] &= norm[:-1] <= norm[1:]
    local[:, 1:] &= norm[:, 1:] <= norm[:, :-1]
    local[:, :-1] &= norm[:, :-1] <= norm[:, 1:]
    local[:, :, 1:] &= norm[:, :, 1:] <= norm[:, :, :-1]
    local[:, :, :-1] &= norm[:, :, :-1] <= norm[:, :, 1:]
    ii, jj, kk = np.where(local)
    return np.column_stack([x[ii], y[jj], z[kk]])


def _deduplicate(points: Array, residuals: Array, distance: float) -> tuple[Array, Array]:
    if not len(points):
        dim = int(points.shape[-1]) if points.ndim >= 2 else 2
        return points.reshape(0, dim), residuals.reshape(0)
    keep: list[int] = []
    for index in np.argsort(residuals):
        if all(np.linalg.norm(points[index] - points[other]) > distance for other in keep):
            keep.append(int(index))
    keep_array = np.asarray(keep, dtype=int)
    return points[keep_array], residuals[keep_array]


def _finite_difference_hessian(
    gradient: Callable[[Array], Array], points: Array, step: float
) -> Array:
    points = np.atleast_2d(np.asarray(points, dtype=float))
    dim = int(points.shape[-1])
    if not len(points):
        return np.empty((0, dim, dim))
    columns = []
    for axis in range(dim):
        shift = np.zeros(dim, dtype=float)
        shift[axis] = step
        columns.append((gradient(points + shift) - gradient(points - shift)) / (2 * step))
    hessian = np.stack(columns, axis=-1)
    return 0.5 * (hessian + np.swapaxes(hessian, 1, 2))


def _newton_step(gradient_values: Array, hessian: Array) -> Array:
    """Batched Newton step; singular Hessians contribute a zero update."""

    delta = np.zeros_like(gradient_values)
    if not len(gradient_values):
        return delta
    det = np.linalg.det(hessian)
    good = np.abs(det) > 1e-18
    if not np.any(good):
        return delta
    delta[good] = np.linalg.solve(hessian[good], gradient_values[good][..., None])[..., 0]
    return delta


def _refine_roots(
    gradient: Callable[[Array], Array],
    starts: Array,
    *,
    step: float,
    n_iter: int,
    max_step: float,
) -> tuple[Array, Array]:
    starts = np.asarray(starts, dtype=float)
    dim = int(starts.shape[-1]) if starts.ndim >= 2 else 2
    points = starts.reshape(-1, dim)
    if not len(points):
        return points, np.empty(0)
    for _ in range(n_iter):
        values = gradient(points)
        hessian = _finite_difference_hessian(gradient, points, step)
        delta = _newton_step(values, hessian)
        scale = np.minimum(
            1.0,
            max_step / np.linalg.norm(delta, axis=1).clip(1e-30),
        )
        points = points - delta * scale[:, None]
    residuals = np.linalg.norm(gradient(points), axis=1)
    return points, residuals


def classify_hessians(
    hessian: Array, *, eigenvalue_tolerance: float = 1e-7
) -> tuple[Array, Array]:
    """Classify critical points from Hessians as max/min/saddle/degenerate.

    In 3-D, a single near-zero eigenvalue with a definite in-plane pair is
    Morse–Bott (a cylindrical generator) and is still labelled max/min/saddle.
    """

    hessian = np.asarray(hessian, dtype=float)
    if hessian.size == 0:
        dim = int(hessian.shape[-1]) if hessian.ndim >= 2 else 2
        return np.empty(0, dtype="<U10"), np.empty((0, dim))
    dim = int(hessian.shape[-1])
    hessian = hessian.reshape(-1, dim, dim)
    hessian = 0.5 * (hessian + np.swapaxes(hessian, 1, 2))
    eigenvalues = np.linalg.eigvalsh(hessian)
    kinds = np.full(len(eigenvalues), "degenerate", dtype="<U10")
    n_pos = np.sum(eigenvalues > eigenvalue_tolerance, axis=1)
    n_neg = np.sum(eigenvalues < -eigenvalue_tolerance, axis=1)
    n_small = dim - n_pos - n_neg
    kinds[(n_small == 0) & (n_pos == dim)] = "minimum"
    kinds[(n_small == 0) & (n_neg == dim)] = "maximum"
    kinds[(n_small == 0) & (n_pos > 0) & (n_neg > 0)] = "saddle"
    if dim == 3:
        kinds[(n_small == 1) & (n_pos == 2)] = "minimum"
        kinds[(n_small == 1) & (n_neg == 2)] = "maximum"
        kinds[(n_small == 1) & (n_pos == 1) & (n_neg == 1)] = "saddle"
    return kinds, eigenvalues


def _analytic_hessian(event, points: Array) -> Array | None:
    if event is None or not len(points):
        return None
    _require_torch()
    field = event.field
    if not hasattr(field, "gradient_and_hessian"):
        return None
    device, dtype = _field_device_dtype(field)
    with torch.no_grad():
        tensor = torch.as_tensor(points, device=device, dtype=dtype)
        output = field.gradient_and_hessian(tensor)
    hessian = output[-1] if isinstance(output, tuple) else output
    hessian = _as_numpy(hessian)
    if hessian.ndim != 3:
        return None
    dim = int(points.shape[-1])
    if hessian.shape[-1] != dim:
        return None
    return hessian


def _grid_from_axes(axes: tuple[Array, ...]) -> Array:
    meshes = np.meshgrid(*axes, indexing="ij")
    return np.column_stack([mesh.ravel() for mesh in meshes])


def _points_inside(points: Array, axes: tuple[Array, ...], margin: float) -> Array:
    if not len(points):
        return np.empty(0, dtype=bool)
    inside = np.ones(len(points), dtype=bool)
    for axis, bound in enumerate(axes):
        inside &= points[:, axis] >= bound[0] + margin
        inside &= points[:, axis] <= bound[-1] - margin
    return inside


def find_critical_points(
    event,
    *axes: Array,
    config: CriticalPointConfig | None = None,
    gradient: Callable[[Array], Array] | None = None,
    gradient_grid: Array | None = None,
) -> CriticalPoints:
    """Find, refine and classify zeros of ∇φ.

    Two axes ``(x, z)`` use 2-D winding on a section. Three axes
    ``(x, y, z)`` use the Brouwer degree on voxels, plus 6-neighbour
    minima of ``||∇φ||`` so cylindrical generators are not missed.
    """

    config = config or CriticalPointConfig()
    axes = tuple(np.asarray(axis, dtype=float) for axis in axes)
    dim = len(axes)
    if dim not in (2, 3):
        raise ValueError("find_critical_points expects 2 or 3 axes")
    if any(len(axis) < 2 for axis in axes):
        raise ValueError("each axis must contain at least two samples")
    if gradient is None:
        if event is None:
            raise TypeError("find_critical_points requires an event or a gradient")
        gradient_fn = make_gradient_function(
            event, dim=dim, batch_size=config.batch_size
        )
    else:
        gradient_fn = _wrap_numpy_gradient(gradient, dim=dim)
    expected = tuple(len(axis) for axis in axes) + (dim,)
    if gradient_grid is None:
        gradient_grid = gradient_fn(_grid_from_axes(axes)).reshape(expected)
    else:
        gradient_grid = np.asarray(gradient_grid, dtype=float)
        if gradient_grid.shape != expected:
            raise ValueError(f"gradient_grid must have shape {expected}")
    if dim == 2:
        candidates = winding_candidates(axes[0], axes[1], gradient_grid)
    else:
        candidates = np.vstack(
            [
                degree_candidates(*axes, gradient_grid),
                _norm_local_minima(*axes, gradient_grid),
            ]
        )
    gradient_norm = np.linalg.norm(gradient_grid, axis=-1)
    scale = max(float(np.median(gradient_norm)), 1e-12)
    residual_limit = config.residual_tolerance * scale
    spacing = min(float(np.min(np.diff(axis))) for axis in axes)
    margin = config.boundary_margin
    if margin is None:
        margin = spacing
    step = config.hessian_step or 0.25 * spacing
    root_array, residual_array = _refine_roots(
        gradient_fn,
        candidates,
        step=step,
        n_iter=config.newton_iterations,
        max_step=4.0 * spacing,
    )
    keep = _points_inside(root_array, axes, margin) & (residual_array <= residual_limit)
    root_array, residual_array = root_array[keep], residual_array[keep]
    distance = config.deduplication_distance
    if distance is None:
        distance = 0.25 * spacing
    root_array, residual_array = _deduplicate(root_array, residual_array, distance)
    hessian = _analytic_hessian(event, root_array)
    if hessian is None:
        hessian = _finite_difference_hessian(gradient_fn, root_array, step)
    kinds, eigenvalues = classify_hessians(
        hessian, eigenvalue_tolerance=config.eigenvalue_tolerance
    )
    return CriticalPoints(root_array, kinds, residual_array, eigenvalues, candidates)


def lithology_entropy(stack: Array, classes: Sequence | None = None) -> Array:
    """Shannon entropy, in bits, across ensemble lithology predictions."""

    stack = np.asarray(stack)
    if classes is None:
        classes = np.unique(stack)
    probabilities = np.stack([(stack == value).mean(axis=0) for value in classes])
    return -(probabilities * np.log2(np.clip(probabilities, 1e-12, 1.0))).sum(axis=0)


def _regular_grid_sample(x: Array, z: Array, max_points: int) -> Array:
    """Coarse regular lattice from 1-D axes, not a ravelled stride."""

    x = np.asarray(x, dtype=float)
    z = np.asarray(z, dtype=float)
    nx, nz = len(x), len(z)
    if nx == 0 or nz == 0:
        return np.empty((0, 2), dtype=float)
    n = nx * nz
    if n <= max_points:
        xx, zz = np.meshgrid(x, z, indexing="ij")
        return np.column_stack([xx.ravel(), zz.ravel()])
    aspect = nx / max(nz, 1)
    nxs = max(1, int(round(math.sqrt(max_points * aspect))))
    nzs = max(1, int(round(max_points / nxs)))
    sx = max(1, int(math.ceil(nx / nxs)))
    sz = max(1, int(math.ceil(nz / nzs)))
    xx, zz = np.meshgrid(x[::sx], z[::sz], indexing="ij")
    return np.column_stack([xx.ravel(), zz.ravel()])


def stability_sample_points(
    grid=None,
    x: Array | None = None,
    z: Array | None = None,
    max_points: int = 8_192,
    seed: int = 0,
) -> Array:
    """Poisson-disk samples on ``grid``, else a subsampled regular lattice."""

    max_points = int(max_points)
    if max_points <= 0:
        return np.empty((0, 2), dtype=float)
    if grid is not None:
        coords = np.asarray(grid.coords(), dtype=float)
        if len(coords) == 0:
            return coords.reshape(0, coords.shape[-1] if coords.ndim == 2 else 2)
        if len(coords) <= max_points:
            return coords
        extent = np.clip(coords.max(axis=0) - coords.min(axis=0), 1e-30, None)
        volume = float(np.prod(extent))
        dim = int(coords.shape[1])
        radius = 0.75 * (volume / max_points) ** (1.0 / dim)
        sampled = np.asarray(
            grid.sample(N=max_points, poissonDisk=(radius, max_points, seed)),
            dtype=float,
        )
        if len(sampled) >= max(1, max_points // 8):
            return sampled
    if x is not None and z is not None:
        return _regular_grid_sample(x, z, max_points)
    raise ValueError("stability_sample_points needs a grid or x, z axes")


def evaluate_stability(
    event,
    *,
    grid=None,
    x: Array | None = None,
    z: Array | None = None,
    points: Array | None = None,
    max_points: int | None = None,
    seed: int = 0,
) -> float:
    """Residual Lipschitz constant ``max ||J_h - I||_2`` of the discrete step."""

    integrator = getattr(getattr(event, "field", None), "integrator", None)
    if integrator is None:
        return float("nan")
    if points is None:
        if max_points is None:
            max_points = CriticalPointConfig.stability_samples
        points = stability_sample_points(
            grid=grid, x=x, z=z, max_points=max_points, seed=seed
        )
    device, dtype = _field_device_dtype(event.field)
    sample = torch.as_tensor(points, device=device, dtype=dtype)
    return float(integrator.stability(sample))


# Order of the explicit RK schemes FlowOffset runs. For both, the chain rule
# through the Butcher tableau gives sup||D Phi_h - I|| <= sum_{k=1}^{p} (hL)^k / k!
# with L >= sup||Dv||: hL (1 + hL/2) for the midpoint rule, and
# hL (1 + hL/2 + (hL)^2/6 + (hL)^3/24) for classic RK4.
_RK_ORDER = {"rk2": 2, "rk4": 4}


def velocity_lipschitz_bound(velocity) -> float:
    """
    Global bound ``G1 = sum_k |C_k| |Om_k| >= sup_x ||Dv(x)||_2`` over all of R^d.

    Uses the shear form ``v = sum_k C_k cos(Om_k . x + off_k)``, for which
    ``Dv = -sum_k sin(.) C_k Om_k^T`` and each rank-one term has spectral norm
    ``|C_k| |Om_k|``. The phase ``off_k`` (lift labels, learned phase) never
    enters, so the bound holds everywhere and for every label.
    """
    with torch.no_grad():
        C, om, _ = velocity.shear_decomposition()
        return float((C.norm(dim=-1) * om.norm(dim=-1)).sum())


def rk_step_rho_bound(
    lipschitz: float, n_steps: int, integrator: str = "rk4", duration: float = 1.0
) -> float:
    """Upper bound on ``sup_x ||D Phi_h(x) - I||_2`` for one RK step, ``h = duration / n_steps``."""
    order = _RK_ORDER[integrator]
    x = abs(float(duration)) * float(lipschitz) / int(n_steps)
    return sum(x**k / math.factorial(k) for k in range(1, order + 1))


def _rk_threshold(integrator: str, target: float = 1.0) -> float:
    """Largest ``hL`` with ``rk_step_rho_bound < target`` (bisection; the bound is increasing)."""
    lo, hi = 0.0, 1.0
    while rk_step_rho_bound(hi, 1, integrator) < target:
        hi *= 2.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if rk_step_rho_bound(mid, 1, integrator) < target:
            lo = mid
        else:
            hi = mid
    return lo


def required_steps(
    lipschitz: float, integrator: str = "rk4", duration: float = 1.0, target: float = 1.0
) -> int:
    """Smallest ``N`` with ``rk_step_rho_bound(lipschitz, N) < target``."""
    n = max(1, math.ceil(abs(float(duration)) * float(lipschitz) / _rk_threshold(integrator, target)))
    while rk_step_rho_bound(lipschitz, n, integrator, duration) >= target:
        n += 1
    return n


def check_step_injectivity(event, *, duration: float = 1.0, target: float = 1.0) -> dict:
    """
    Global, sample-free check that every integration step is injective.

    For ``rk2`` / ``rk4``, ``sup||D Phi_h - I|| < 1`` everywhere makes each step
    ``Phi_h = id + g`` a contraction perturbation of the identity, so
    ``|Phi_h(x) - Phi_h(y)| >= (1 - rho)|x - y| > 0`` on all of R^d, and the
    composition of steps is injective. ``rho`` is bounded with
    :func:`velocity_lipschitz_bound` and :func:`rk_step_rho_bound`; no points,
    domain or sampling are involved.

    ``shear`` steps are explicit shears ``x -> x + U(d.x)`` with ``U ⊥ d`` and
    inverse ``x - U(d.x)``: injective at any ``N``. ``nambu`` (implicit
    midpoint on split sub-flows) needs a different bound and is reported as
    ``certified=None``.

    Returns ``{"certified", "required_steps", "factors": [...]}``, one factor
    per velocity of a composite flow, each with ``n_steps``, ``lipschitz``
    (``G1``), ``rho_bound`` and ``required_steps``. ``certified`` is ``None``
    if any factor is not covered.
    """
    integrator = getattr(getattr(event, "field", None), "integrator", None)
    if integrator is None:
        raise ValueError("event.field has no integrator to check")
    if getattr(integrator, "fault_lift", None) is not None:
        raise NotImplementedError(
            "fault lifts modify the velocity (confinement, mobility) and make the "
            "map discontinuous; the global bound does not cover them"
        )
    scheme = integrator.integrator
    factors = []
    for velocity, n in zip(integrator.velocities, integrator.steps):
        if scheme == "shear":
            factors.append(dict(n_steps=n, certified=True, required_steps=1))
            continue
        if scheme not in _RK_ORDER:
            factors.append(dict(n_steps=n, certified=None, required_steps=None))
            continue
        lip = velocity_lipschitz_bound(velocity)
        rho = rk_step_rho_bound(lip, n, scheme, duration)
        factors.append(dict(
            n_steps=n,
            lipschitz=lip,
            rho_bound=rho,
            required_steps=required_steps(lip, scheme, duration, target),
            certified=rho < target,
        ))
    flags = [f["certified"] for f in factors]
    needed = [f["required_steps"] for f in factors]
    return {
        "integrator": scheme,
        "certified": None if None in flags else all(flags),
        "required_steps": None if None in needed else max(needed),
        "factors": factors,
    }


def evaluate_ensemble(
    models: Sequence,
    events: Sequence,
    points: Array,
    grid_shape: tuple[int, int],
    x: Array,
    z: Array,
    *,
    grid=None,
    critical_config: CriticalPointConfig | None = None,
    batch_size: int = 65_536,
) -> dict:
    """Evaluate ensemble fields and all diagnostics used by the notebook."""

    config = critical_config or CriticalPointConfig()
    points = _as_points(points)
    x, z = np.asarray(x, dtype=float), np.asarray(z, dtype=float)
    stab_pts = stability_sample_points(
        grid=grid, x=x, z=z, max_points=config.stability_samples
    )
    scalar, lithology, critical, stability, min_gradient_norm = [], [], [], [], []
    for model, event in zip(models, events):
        member_scalar, member_lithology, member_gradient = evaluate_fields(
            model, event, points, grid_shape=grid_shape, batch_size=batch_size
        )
        scalar.append(member_scalar)
        lithology.append(member_lithology)
        result = find_critical_points(
            event,
            x,
            z,
            config=config,
            gradient_grid=member_gradient,
        )
        critical.append(result)
        min_gradient_norm.append(
            float(np.linalg.norm(member_gradient.reshape(-1, 2), axis=1).min())
        )
        stability.append(evaluate_stability(event, points=stab_pts))
    extrema = np.asarray(
        [[item.counts["maximum"], item.counts["minimum"]] for item in critical],
        dtype=int,
    )
    saddles = np.asarray([item.counts["saddle"] for item in critical], dtype=int)
    return {
        "scalar": np.stack(scalar),
        "lithology": np.stack(lithology),
        "critical_points": critical,
        "extrema": extrema,
        "saddles": saddles,
        "minimum_gradient_norm": np.asarray(min_gradient_norm),
        "stability": np.asarray(stability),
    }


def select_diagnostic_member(diagnostics: dict) -> int:
    """Select most-extreme member, then lowest sampled gradient norm on a tie."""

    counts = np.asarray(diagnostics["extrema"], dtype=int).sum(axis=1)
    candidates = np.flatnonzero(counts == counts.max())
    if len(candidates) == 1:
        return int(candidates[0])
    gradient_norms = np.asarray(diagnostics["minimum_gradient_norm"], dtype=float)
    return int(candidates[np.nanargmin(gradient_norms[candidates])])


def gradient_resolution_check(
    event,
    resolutions: Iterable[tuple[int, int]],
    bounds: tuple[tuple[float, float], tuple[float, float]],
    *,
    config: CriticalPointConfig | None = None,
    gradient=None,
) -> list[dict]:
    """Repeat gradient-root detection on successively finer candidate grids."""

    output = []
    for nx, nz in resolutions:
        x = np.linspace(*bounds[0], int(nx))
        z = np.linspace(*bounds[1], int(nz))
        result = find_critical_points(event, x, z, config=config, gradient=gradient)
        output.append({"shape": (int(nx), int(nz)), "x": x, "z": z, "critical": result})
    return output


def sample_regular_grid(x: Array, z: Array, field: Array, points: Array) -> Array:
    """Bilinearly sample a field stored with shape ``(x, z)``."""

    points = np.asarray(points, dtype=float)
    ix = np.clip(np.searchsorted(x, points[:, 0]) - 1, 0, len(x) - 2)
    iz = np.clip(np.searchsorted(z, points[:, 1]) - 1, 0, len(z) - 2)
    tx = (points[:, 0] - x[ix]) / (x[ix + 1] - x[ix])
    tz = (points[:, 1] - z[iz]) / (z[iz + 1] - z[iz])
    return (
        (1 - tx) * (1 - tz) * field[ix, iz]
        + tx * (1 - tz) * field[ix + 1, iz]
        + (1 - tx) * tz * field[ix, iz + 1]
        + tx * tz * field[ix + 1, iz + 1]
    )


def isocontour_polyline(x: Array, z: Array, field: Array, level: float) -> Array:
    """Longest polyline of the isocontour ``field = level``.

    ``field`` has shape ``(len(x), len(z))``. Strain glyphs sit on these
    fitted contacts rather than on digitised traces.
    """

    x = np.asarray(x, dtype=float).ravel()
    z = np.asarray(z, dtype=float).ravel()
    field = np.asarray(field, dtype=float)
    if field.shape != (x.size, z.size):
        raise ValueError(
            f"field shape {field.shape} does not match "
            f"(len(x), len(z))=({x.size}, {z.size})"
        )
    lines: list[Array] = []
    try:
        from contourpy import LineType, contour_generator

        generated = contour_generator(
            x=x, y=z, z=field.T, line_type=LineType.Separate
        ).lines(float(level))
        lines = [np.asarray(seg, dtype=float) for seg in generated if len(seg)]
    except Exception:
        fig, ax = plt.subplots()
        try:
            cs = ax.contour(x, z, field.T, levels=[float(level)])
            if getattr(cs, "allsegs", None):
                lines = [
                    np.asarray(seg, dtype=float) for seg in cs.allsegs[0] if len(seg)
                ]
            else:
                lines = [
                    np.asarray(path.vertices, dtype=float)
                    for path in cs.get_paths()
                    if len(path.vertices)
                ]
        finally:
            plt.close(fig)
    if not lines:
        raise ValueError(f"no isocontour at level {level}")

    def _arclength(seg: Array) -> float:
        if len(seg) < 2:
            return 0.0
        return float(np.linalg.norm(np.diff(seg, axis=0), axis=1).sum())

    return max(lines, key=_arclength)


# ---------------------------------------------------------------------------
# Array-only helpers: Surfe interpolants, saved contacts and critical points.
# No torch / curlew required.
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def quiet():
    """Silence C/C++ stdout and stderr (e.g. Surfe's progress printing)."""

    sys.stdout.flush()
    sys.stderr.flush()
    saved = [os.dup(1), os.dup(2)]
    null = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(null, 1)
        os.dup2(null, 2)
        yield
    finally:
        os.dup2(saved[0], 1)
        os.dup2(saved[1], 2)
        for fd in (*saved, null):
            os.close(fd)


class AnisotropicSurfe:
    """A Surfe interpolant whose range along a chosen direction is ``ratio`` times its range across it.

    Surfe's own ``SetGlobalAnisotropy`` estimates the plunge from the bedding;
    this takes the direction as given (e.g. the fold hinge a restoration uses),
    by fitting in coordinates with the ``hinge`` component compressed by
    ``ratio``, where the kernel is isotropic. Constraints are mapped in
    (positions by ``T``, bedding normals as covectors by ``T^-1``) and values
    out (scalar unchanged, gradient by ``T``), so the object can be used
    wherever a ``surfepy.Surfe_API`` is: the ``Add*`` and ``Set*`` calls, and
    ``EvaluateInterpolantAtPoint(s)`` and ``EvaluateVectorInterpolantAtPoint(s)``.
    ``ratio = 1`` is the ordinary isotropic fit.
    """

    def __init__(self, model, hinge: Array, ratio: float = 1.0):
        h = np.asarray(hinge, dtype=float)
        h = h / np.linalg.norm(h)
        self._model = model
        self.ratio = float(ratio)
        self.hinge = h
        self._T = np.eye(3) - (1.0 - 1.0 / self.ratio) * np.outer(h, h)    # symmetric
        self._Tinv = np.linalg.inv(self._T)

    def __getattr__(self, name):                       # SetRBFKernel, ComputeInterpolant, ...
        return getattr(self._model, name)

    def AddInterfaceConstraint(self, x, y, z, level):
        self._model.AddInterfaceConstraint(*(self._T @ [x, y, z]), level)

    def AddPlanarConstraintwNormal(self, x, y, z, nx, ny, nz):
        n = self._Tinv @ [nx, ny, nz]
        self._model.AddPlanarConstraintwNormal(*(self._T @ [x, y, z]), *(n / np.linalg.norm(n)))

    def AddInequalityConstraint(self, x, y, z, level):
        self._model.AddInequalityConstraint(*(self._T @ [x, y, z]), level)

    def AddTangentConstraint(self, x, y, z, tx, ty, tz):
        self._model.AddTangentConstraint(*(self._T @ [x, y, z]), *(self._T @ [tx, ty, tz]))

    def EvaluateInterpolantAtPoint(self, x, y, z):
        return self._model.EvaluateInterpolantAtPoint(*(self._T @ [x, y, z]))

    def EvaluateInterpolantAtPoints(self, points):
        return self._model.EvaluateInterpolantAtPoints(np.asarray(points, dtype=float) @ self._T.T)

    def EvaluateVectorInterpolantAtPoint(self, x, y, z):
        return self._T @ np.asarray(self._model.EvaluateVectorInterpolantAtPoint(*(self._T @ [x, y, z])))

    def EvaluateVectorInterpolantAtPoints(self, points):
        g = np.asarray(self._model.EvaluateVectorInterpolantAtPoints(np.asarray(points, dtype=float) @ self._T.T))
        return g @ self._T.T                           # grad u = T^T grad' u' (T symmetric)


def _surfe_points(points: Array, y: float) -> Array:
    """Surfe's 3-D frame: ``(x, z)`` section points are embedded at ``y``; 3-D points pass through."""

    points = np.atleast_2d(np.asarray(points, dtype=float))
    if points.shape[1] == 3:
        return points
    return np.column_stack([points[:, 0], np.full(len(points), float(y)), points[:, 1]])


def surfe_scalar(model, points: Array, *, y: float = 0.0, batch_size: int = 200_000) -> Array:
    """Scalar of a Surfe interpolant at ``(x, z)`` section points or ``(x, y, z)`` points."""

    points = np.atleast_2d(np.asarray(points, dtype=float))
    out = []
    with quiet():
        for start in range(0, len(points), batch_size):
            chunk = _surfe_points(points[start : start + batch_size], y)
            out.append(np.asarray(model.EvaluateInterpolantAtPoints(chunk), dtype=float).ravel())
    return np.concatenate(out) if out else np.empty(0)


def surfe_isovalues(model, horizons: Sequence[Array], *, y: float = 0.0) -> Array:
    """Isovalue of each horizon for a Surfe interpolant, as curlew sets it.

    curlew's ``GeoEvent.getIsovalues`` evaluates the field exactly at an
    isosurface's seed points (here the horizon's contacts) and takes their
    mean. This does the same with Surfe's exact values, so Surfe's horizons and
    lithology are set the same way as NF's and LDI's.
    """

    return np.asarray([float(np.mean(surfe_scalar(model, h, y=y))) for h in horizons])


def make_surfe_gradient_function(model, *, y: float = 0.0, batch_size: int = 200_000):
    """Unnormalised gradient callable for a Surfe interpolant.

    Takes ``(x, z)`` section points (returning two components, on the plane at
    ``y``) or full ``(x, y, z)`` points (returning three).

    Pass it to :func:`find_critical_points` as ``gradient=`` (with ``None``
    as the event). Surfe evaluates the analytic gradient of the RBF
    interpolant anywhere, so critical points are refined without a raster.
    """

    def gradient(points: Array) -> Array:
        points = np.atleast_2d(np.asarray(points, dtype=float))
        out = []
        with quiet():
            for start in range(0, len(points), batch_size):
                chunk = _surfe_points(points[start : start + batch_size], y)
                vec = np.asarray(model.EvaluateVectorInterpolantAtPoints(chunk), dtype=float)
                out.append(vec if points.shape[1] == 3 else vec[:, (0, 2)])
        return np.concatenate(out, axis=0) if out else np.empty((0, 2))

    return gradient


def evaluate_surfe(
    model, *axes: Array, y: float = 0.0, batch_size: int = 200_000
) -> tuple[Array, Array]:
    """Scalar and gradient of Surfe on a grid.

    Two axes ``(x, z)`` evaluate the section at ``y`` and return shapes
    ``(nx, nz)`` and ``(nx, nz, 2)``; three axes ``(x, y, z)`` evaluate the
    volume and return ``(nx, ny, nz)`` and ``(nx, ny, nz, 3)``.
    """

    axes = tuple(np.asarray(a, dtype=float) for a in axes)
    shape = tuple(len(a) for a in axes)
    points = _grid_from_axes(axes)
    scalar = surfe_scalar(model, points, y=y, batch_size=batch_size).reshape(shape)
    gradient = make_surfe_gradient_function(model, y=y, batch_size=batch_size)(points)
    return scalar, gradient.reshape(*shape, len(axes))


def surfe_critical_points(
    model,
    *axes: Array,
    y: float = 0.0,
    config: CriticalPointConfig | None = None,
    gradient_grid: Array | None = None,
) -> CriticalPoints:
    """Critical points of a Surfe interpolant on an ``x``-``z`` section at ``y`` (two
    axes) or in the ``x``-``y``-``z`` volume (three axes)."""

    return find_critical_points(
        None,
        *axes,
        config=config,
        gradient=make_surfe_gradient_function(model, y=y),
        gradient_grid=gradient_grid,
    )


def surfe_mesher(model, step: float, *, origin=0.0, batch_size: int = 200_000):
    """``mesher(level, bounds) -> (vertices, faces)`` for a 3-D Surfe interpolant.

    Marching cubes (scikit-image) on a regular grid of spacing ``step`` over
    ``bounds`` (``[[x0, y0, z0], [x1, y1, z1]]`` in the model frame). ``origin``
    is the shift between the model frame and the frame the interpolant was
    fitted in. A level the field never reaches in ``bounds`` gives no faces.
    The field on a given box is evaluated once and reused for every level.
    """

    from skimage.measure import marching_cubes

    origin = np.asarray(origin, dtype=float)
    cache: dict[bytes, Array] = {}

    def mesh(level: float, bounds: Array) -> tuple[Array, Array]:
        lo, hi = (np.asarray(b, dtype=float) - origin for b in bounds)
        key = np.concatenate([np.asarray(lo, np.float64).ravel(), np.asarray(hi, np.float64).ravel()]).tobytes()
        field = cache.get(key)
        if field is None:
            axes = [np.arange(a, b + 0.5 * step, step) for a, b in zip(lo, hi)]
            points = _grid_from_axes(tuple(axes))
            field = surfe_scalar(model, points, batch_size=batch_size).reshape(tuple(map(len, axes)))
            cache[key] = field
        try:
            verts, faces, _, _ = marching_cubes(field, level=float(level), spacing=(step,) * 3)
        except ValueError:
            return np.empty((0, 3)), np.empty((0, 3), dtype=int)
        return verts + lo + origin, faces

    return mesh


def export_meshes(
    path,
    mesher,
    isovalues: Sequence[float],
    box: Array,
    *,
    extrema: Sequence[tuple] = (),
    shell_delta: float,
    shell_half: float,
    shell_tries: int = 4,
) -> None:
    """Mesh each contact surface, and the small level surfaces round each extremum, to ``path``.

    ``mesher(level, bounds)`` returns ``(vertices, faces)``. ``extrema`` is a
    list of ``(point, kind, value)`` with ``value`` the field at the extremum.
    The level surface just inside an extremum (``value -/+ shell_delta`` around a
    maximum / minimum, halved on each try) is meshed within ``shell_half`` of it
    for ``shell_tries`` levels, so the figure notebook can keep the first that
    closes without needing the field. Raw arrays only; no mesh library needed.
    """

    arrays: dict[str, Array] = {
        "box": np.asarray(box, dtype=float),
        "isovalues": np.asarray(isovalues, dtype=float),
        "n_tries": np.asarray(shell_tries),
    }
    for h, level in enumerate(isovalues):
        v, f = mesher(float(level), box)
        arrays[f"sheet{h}_v"], arrays[f"sheet{h}_f"] = np.asarray(v, np.float32), np.asarray(f, np.int32)
    arrays["ext_points"] = np.asarray([np.asarray(e[0], dtype=float) for e in extrema]).reshape(-1, 3)
    arrays["ext_kinds"] = np.asarray([e[1] for e in extrema]).astype("U10")
    arrays["ext_values"] = np.asarray([float(e[2]) for e in extrema])
    for e, (p, kind, value) in enumerate(extrema):
        p = np.asarray(p, dtype=float)
        bounds = np.stack([p - shell_half, p + shell_half])
        delta = float(shell_delta)
        for t in range(shell_tries):
            level = value - delta if kind == "maximum" else value + delta
            v, f = mesher(level, bounds)
            arrays[f"shell{e}_{t}_v"], arrays[f"shell{e}_{t}_f"] = np.asarray(v, np.float32), np.asarray(f, np.int32)
            delta *= 0.5
    np.savez_compressed(path, **arrays)


def evenly_spaced_levels(values: Sequence[float], n: int, *, trim: float = 0.03) -> Array:
    """``n`` levels evenly spaced across the range of ``values``, highest first.

    For drawing a few nicely spaced isosurfaces through a stack of many traces:
    the range is the traces' isovalues, pulled in by ``trim`` of it at each end
    so the outermost surfaces stay inside the data. Highest first, i.e. youngest
    first when the scalar increases upward.
    """

    values = np.asarray(values, dtype=float)
    lo, hi = values.min(), values.max()
    pad = trim * (hi - lo)
    return np.linspace(hi - pad, lo + pad, n)


def plot_cloud_scalar(
    xyz: Array,
    values: Array,
    levels: Sequence[float],
    *,
    lines: Sequence[Array] = (),
    title: str = "",
    unit: str = "m",
    size: float = 1.0,
):
    """A scalar field on a point cloud in discrete bands, as y–z and x–y views.

    ``levels`` are the band edges (any order); values beyond the outer edges get
    a band of their own at each end, so there are ``len(levels) + 1`` colours in
    the ramp of the other figures (low red, high blue).     Points nearest the
    viewer are drawn last: low x in the y–z view, high z in the x–y view.
    ``lines`` (e.g. the digitised traces) are drawn on top as points, in black.
    """

    from matplotlib.colors import BoundaryNorm, ListedColormap

    xyz, values = np.asarray(xyz, dtype=float), np.asarray(values, dtype=float).ravel()
    bounds = np.sort(np.asarray(levels, dtype=float))
    cmap = ListedColormap(figstyle.ramp()(np.linspace(0.0, 1.0, len(bounds) + 1)))
    norm = BoundaryNorm(bounds, cmap.N, extend="both")
    span = np.ptp(xyz, axis=0)
    fig, axs = plt.subplots(
        2, 1, figsize=(12, 12 * (span[2] / span[1] + span[0] / span[1]) + 1.2),
        height_ratios=(span[2], span[0]), constrained_layout=True,
    )
    views = (((1, 2), np.argsort(-xyz[:, 0]), "y–z"), ((1, 0), np.argsort(xyz[:, 2]), "x–y"))
    for ax, ((i, j), order, name) in zip(axs, views):
        sc = ax.scatter(xyz[order, i], xyz[order, j], c=values[order], cmap=cmap, norm=norm,
                        s=size, linewidths=0, rasterized=True)
        for line in lines:
            ax.scatter(line[:, i], line[:, j], s=3, color="k", linewidths=0, zorder=3)
        ax.set_aspect("equal")
        ax.set(xlabel=f"{'xyz'[i]} ({unit})", ylabel=f"{'xyz'[j]} ({unit})", title=f"{title}, {name}".strip(", "))
    fig.colorbar(sc, ax=axs, shrink=0.5, spacing="proportional", label="scalar")
    return fig, axs


def horizon_isovalues(
    x: Array, z: Array, scalar: Array, horizons: Sequence[Array]
) -> list[float]:
    """Isovalue of each horizon: median of the field sampled along its contacts."""

    return [float(np.median(sample_regular_grid(x, z, scalar, h))) for h in horizons]


def lithology_from_isovalues(
    scalar: Array, isovalues: Sequence[float], *, first_id: int = 1
) -> Array:
    """Integer lithology: ``first_id`` plus the number of isovalues below each value.

    The default ``first_id=1`` matches curlew's ``lithoID`` numbering when the
    scalar increases with younging on both sides.
    """

    return first_id + np.digitize(np.asarray(scalar), np.sort(np.asarray(isovalues, dtype=float)))


def _contour_lines(x: Array, z: Array, field: Array, level: float) -> list[Array]:
    x = np.asarray(x, dtype=float).ravel()
    z = np.asarray(z, dtype=float).ravel()
    try:
        from contourpy import LineType, contour_generator

        generated = contour_generator(
            x=x, y=z, z=np.asarray(field, dtype=float).T, line_type=LineType.Separate
        ).lines(float(level))
        return [np.asarray(seg, dtype=float) for seg in generated if len(seg)]
    except ImportError:
        fig, ax = plt.subplots()
        try:
            cs = ax.contour(x, z, np.asarray(field).T, levels=[float(level)])
            return [np.asarray(seg, dtype=float) for seg in cs.allsegs[0] if len(seg)]
        finally:
            plt.close(fig)


def ensemble_contacts(
    scalar_stack: Array, x: Array, z: Array, horizons: Sequence[Array]
) -> list[list[list[Array]]]:
    """Fitted contact polylines ``[member][horizon] -> [polyline, ...]``.

    Each horizon's level is the median of a member's scalar along the observed
    contacts. These polylines are all that the ensemble figure needs, so they
    can be saved instead of the full scalar stack.
    """

    contacts = []
    for field in scalar_stack:
        levels = horizon_isovalues(x, z, field, horizons)
        contacts.append([_contour_lines(x, z, field, level) for level in levels])
    return contacts


def pack_contacts(contacts) -> dict[str, Array]:
    """Flatten :func:`ensemble_contacts` output into arrays for ``np.savez``."""

    vertices, lengths, member, horizon = [], [], [], []
    for m, per_member in enumerate(contacts):
        for h, lines in enumerate(per_member):
            for line in lines:
                vertices.append(np.asarray(line, dtype=np.float32))
                lengths.append(len(line))
                member.append(m)
                horizon.append(h)
    return {
        "contact_vertices": np.concatenate(vertices) if vertices else np.empty((0, 2), np.float32),
        "contact_lengths": np.asarray(lengths, dtype=np.int64),
        "contact_member": np.asarray(member, dtype=np.int64),
        "contact_horizon": np.asarray(horizon, dtype=np.int64),
    }


def unpack_contacts(arrays, n_members: int, n_horizons: int):
    """Inverse of :func:`pack_contacts`."""

    contacts = [[[] for _ in range(n_horizons)] for _ in range(n_members)]
    offsets = np.concatenate([[0], np.cumsum(arrays["contact_lengths"])])
    for k, (m, h) in enumerate(zip(arrays["contact_member"], arrays["contact_horizon"])):
        contacts[int(m)][int(h)].append(
            np.asarray(arrays["contact_vertices"][offsets[k] : offsets[k + 1]], dtype=float)
        )
    return contacts


def pack_critical(critical: CriticalPoints, prefix: str = "critical_") -> dict[str, Array]:
    """Arrays for ``np.savez`` that restore a :class:`CriticalPoints`."""

    return {
        prefix + "points": np.asarray(critical.points, dtype=float),
        prefix + "kinds": np.asarray(critical.kinds).astype("U10"),
        prefix + "residuals": np.asarray(critical.residuals, dtype=float),
        prefix + "eigenvalues": np.asarray(critical.eigenvalues, dtype=float),
    }


def unpack_critical(arrays, prefix: str = "critical_") -> CriticalPoints:
    """Inverse of :func:`pack_critical`."""

    return CriticalPoints(
        points=np.asarray(arrays[prefix + "points"], dtype=float),
        kinds=np.asarray(arrays[prefix + "kinds"]),
        residuals=np.asarray(arrays[prefix + "residuals"], dtype=float),
        eigenvalues=np.asarray(arrays[prefix + "eigenvalues"], dtype=float),
        candidates=np.empty((0, np.shape(arrays[prefix + "points"])[-1])),
    )


def draw_contact_ensemble(
    ax,
    scalar_stack: Array,
    x: Array,
    z: Array,
    horizons: Sequence[Array],
    horizon_names: Sequence[str],
    gradient_points: Array,
    gradient_vectors: Array,
    *,
    colors: dict | None = None,
    contacts=None,
) -> None:
    """Draw every member's fitted contacts over the constraints.

    Give either ``scalar_stack`` (contoured here) or ``contacts``, the output
    of :func:`ensemble_contacts` (or :func:`unpack_contacts`), in which case
    ``scalar_stack`` may be ``None``.
    """

    if colors is None:
        colors = {
            name: figstyle.HORIZON_PALETTE[i % len(figstyle.HORIZON_PALETTE)]
            for i, name in enumerate(horizon_names)
        }
    if contacts is not None:
        for per_member in contacts:
            for index, (name, lines) in enumerate(zip(horizon_names, per_member)):
                for line in lines:
                    ax.plot(
                        line[:, 0], line[:, 1],
                        color=figstyle._horizon_color(name, index, colors),
                        lw=0.4, alpha=0.55, zorder=2,
                    )
        scalar_stack = ()
    for field in scalar_stack:
        for index, (name, horizon) in enumerate(zip(horizon_names, horizons)):
            level = float(np.median(sample_regular_grid(x, z, field, horizon)))
            ax.contour(
                x,
                z,
                field.T,
                levels=[level],
                colors=[figstyle._horizon_color(name, index, colors)],
                linewidths=0.4,
                # A single-colour contour dashes negative levels by default,
                # which would split one horizon into two line styles.
                linestyles="solid",
                alpha=0.55,
                zorder=2,
            )
    figstyle.overlay_constraints(
        ax,
        horizons,
        horizon_names,
        gradient_points,
        gradient_vectors,
        colors=colors,
        scatter_s=8,
        quiver_scale=0.013,
    )


def extrema_title(stem: str, critical: CriticalPoints) -> str:
    n = critical.extrema_count
    word = "extremum" if n == 1 else "extrema"
    return rf"{stem} $\psi$ ({n} {word})"


# Interior extrema in every figure: Tol purple, apart from curlew's colour ramp.
# Triangles up for maxima and down for minima in 2-D; closed enclosures in 3-D.
EXTREMUM_COLOR = "#AA3377"

_CRITICAL_STYLES = {
    "maximum": dict(marker="^", facecolor=EXTREMUM_COLOR, edgecolors="white", linewidths=0.5),
    "minimum": dict(marker="v", facecolor=EXTREMUM_COLOR, edgecolors="white", linewidths=0.5),
    "saddle": dict(marker="x", facecolor="none", edgecolors="#009E73", linewidths=0.9),
    "degenerate": dict(marker="o", facecolor="none", edgecolors="0.45", linewidths=0.7),
}


def extrema_handles(
    *critical: CriticalPoints,
    include_saddles: bool = False,
    include_degenerate: bool = False,
):
    """Legend proxies for the critical points that were drawn.

    With no sets, maxima and minima are both listed. Pass the sets given to
    :func:`overlay_critical_points` to omit a kind that has no points.
    """

    kinds = ["maximum", "minimum"]
    if include_saddles:
        kinds.append("saddle")
    if include_degenerate:
        kinds.append("degenerate")
    if critical:
        kinds = [
            kind for kind in kinds
            if any(len(item.of_kind(kind)) for item in critical)
        ]
    handles = []
    for kind in kinds:
        style = _CRITICAL_STYLES[kind]
        handles.append(
            Line2D(
                [0],
                [0],
                linestyle="none",
                marker=style["marker"],
                markerfacecolor=style["facecolor"],
                markeredgecolor=style["edgecolors"],
                markeredgewidth=style["linewidths"],
                markersize=5,
                label=kind.capitalize(),
            )
        )
    return handles


def overlay_critical_points(
    ax,
    critical: CriticalPoints,
    *,
    include_saddles: bool = False,
    include_degenerate: bool = False,
    size: float = 16,
    coords: tuple[int, int] = (0, 1),
    slice_axis: int | None = None,
    slice_value: float | None = None,
    slice_atol: float | None = None,
) -> None:
    """Overlay classified critical points on a 2-D section.

    ``coords`` picks the plotted columns. For a 3-D volume result on a
    *y*–*z* panel use ``coords=(1, 2)``; ``slice_axis`` / ``slice_value``
    / ``slice_atol`` keep only points in a slab.
    """

    kinds = ["maximum", "minimum"]
    if include_saddles:
        kinds.append("saddle")
    if include_degenerate:
        kinds.append("degenerate")
    i, j = coords
    for kind in kinds:
        points = critical.of_kind(kind)
        if slice_axis is not None and len(points):
            if slice_value is None:
                raise TypeError("slice_value is required when slice_axis is set")
            half = 0.0 if slice_atol is None else float(slice_atol)
            points = points[np.abs(points[:, slice_axis] - float(slice_value)) <= half]
        if not len(points):
            continue
        style = _CRITICAL_STYLES[kind]
        ax.scatter(
            points[:, i],
            points[:, j],
            s=size,
            marker=style["marker"],
            facecolors=style["facecolor"],
            edgecolors=style["edgecolors"],
            linewidths=style["linewidths"],
            label=kind.capitalize(),
            zorder=8,
        )


def pathline_seeds(
    x: Array | None = None,
    z: Array | None = None,
    *,
    grid=None,
    n_seeds: int = 64,
    extra: Array | None = None,
    seed: int = 0,
) -> Array:
    """Poisson-disk seeds via :meth:`~curlew.geometry.Grid.sample`.

    Same path as :func:`stability_sample_points`: ``grid.sample(..., poissonDisk=)``
    when a grid is given, otherwise a subsampled regular lattice.
    """

    seeds = stability_sample_points(
        grid=grid, x=x, z=z, max_points=n_seeds, seed=seed
    )
    if extra is None or not len(seeds):
        return seeds
    extra = np.asarray(extra, dtype=float).reshape(-1, 2)
    if len(seeds) >= 2:
        gap = 0.35 * float(np.median(np.sort(np.linalg.norm(
            seeds[:, None, :] - seeds[None, :, :], axis=-1
        ), axis=1)[:, 1]))
    else:
        gap = 0.0
    keep = [
        point
        for point in extra
        if np.min(np.linalg.norm(seeds - point, axis=1)) > gap
    ]
    if not keep:
        return seeds
    return np.vstack([seeds, np.asarray(keep)])


def _integrator_step_count(integrator) -> int:
    steps = getattr(integrator, "steps", None)
    if steps is not None:
        return int(sum(int(n) for n in steps))
    n_steps = integrator.n_steps
    if isinstance(n_steps, (list, tuple)):
        return int(sum(int(n) for n in n_steps))
    return int(n_steps)


def _shear_inverse_frames(
    integrator, seeds: torch.Tensor, *, velocity=None, n_steps: int | None = None
) -> Array:
    """Discrete inverse orbit of the shear splitting: one vertex per translation.

    Each sub-step is ``x ← x + h cos(Ω·x) C`` with ``C ⊥ Ω``. The polyline is
    the map actually applied, not a sample of the continuum pathline of
    ``v = Σ shears``.
    """

    x = seeds
    frames = [x.detach().cpu().numpy()[:, :2]]
    with torch.no_grad():
        if velocity is not None:
            record: list = []
            x = integrator._integrate(
                x, sign=-1.0, jac=False, duration=1.0, n_steps=n_steps,
                velocity=velocity, record=record,
            )
            frames.extend(y.detach().cpu().numpy()[:, :2] for y in record)
        else:
            for vel, n, span in integrator._plan(-1.0):
                record = []
                x = integrator._integrate(
                    x, sign=-1.0, jac=False, duration=1.0, n_steps=n,
                    velocity=vel, t_span=span, record=record,
                )
                frames.extend(y.detach().cpu().numpy()[:, :2] for y in record)
    return np.stack(frames, axis=0)


def _inverse_frames_for_velocity(integrator, seeds: torch.Tensor, velocity, n_steps: int) -> Array:
    """Discrete inverse orbit of one velocity, starting from ``seeds``."""

    if getattr(integrator, "integrator", None) == "shear":
        return _shear_inverse_frames(
            integrator, seeds, velocity=velocity, n_steps=n_steps
        )
    x = seeds
    frames = [x.detach().cpu().numpy()[:, :2]]
    n = max(1, int(n_steps))
    with torch.no_grad():
        for k in range(n):
            t0 = 1.0 - k / n
            t1 = t0 - 1.0 / n
            kwargs = dict(
                sign=-1.0, jac=False, duration=1.0 / n, n_steps=1, t_span=(t0, t1),
            )
            if velocity is not None:
                kwargs["velocity"] = velocity
            x = integrator._integrate(x, **kwargs)
            frames.append(x.detach().cpu().numpy()[:, :2])
    return np.stack(frames, axis=0)


def factor_pathlines(event, seeds: Array) -> list[Array]:
    """Inverse orbits of each velocity factor, each from the same present-day seeds.

    The bands are independent maps. Particle ``i`` is advected by ``-v_k``
    alone for factor ``k``; it is not handed from one band to the next.
    Returns one ``(n_frames, n_seeds, 2)`` stack per factor, coarse first
    (RK: ``n_k + 1``; shear: one vertex per translation).
    """

    integrator = getattr(getattr(event, "field", None), "integrator", None)
    if integrator is None:
        raise TypeError("event.field.integrator is required")
    device, dtype = _field_device_dtype(event.field)
    x0 = torch.as_tensor(_as_points(seeds)[:, :2], device=device, dtype=dtype)
    velocities = event_velocities(event)
    if not velocities:
        raise TypeError("event field exposes no velocity")
    steps = getattr(integrator, "steps", None)
    if steps is None:
        n = integrator.n_steps
        n = int(n[0] if isinstance(n, (list, tuple)) else n)
        steps = [n] * len(velocities)
    return [
        _inverse_frames_for_velocity(integrator, x0.clone(), velocity, n)
        for velocity, n in zip(velocities, steps)
    ]


def factor_labels(event) -> list[str]:
    """Panel titles for :func:`factor_pathlines` (coarse first)."""

    n = len(event_velocities(event))
    if n == 2:
        return ["Coarse", "Fine"]
    return [f"Band {i + 1}" for i in range(max(n, 1))]


def inverse_pathlines(event, seeds: Array) -> Array:
    """Record one composed inverse orbit per present-day seed.

    A list of velocity factors is a single map ``Φ = Φ_{K-1} ∘ ⋯ ∘ Φ_0``,
    not K separate flows. Frames are ``x(τ) = Φ_τ^{-1}(x_1)`` as the
    composition parameter ``τ`` runs from 0 to 1 — fine band first on the
    way back, then coarse, carrying the same particle.

    For ``rk2`` / ``rk4`` the shape is ``(N + 1, n_seeds, 2)`` with ``N``
    the total RK step count. For ``shear`` each vertex is a closed-form
    translation along one direction group (Strang-swept), so there are as
    many segments as shear sub-steps, not as many as ``n_steps``. The pin
    is *not* applied; subtract ``integrator.frame_offset`` from the last
    frame for pinned restored coordinates.
    """

    integrator = getattr(getattr(event, "field", None), "integrator", None)
    if integrator is None:
        raise TypeError("event.field.integrator is required")
    device, dtype = _field_device_dtype(event.field)
    x = torch.as_tensor(_as_points(seeds)[:, :2], device=device, dtype=dtype)
    if getattr(integrator, "integrator", None) == "shear":
        return _shear_inverse_frames(integrator, x)
    n_total = max(1, _integrator_step_count(integrator))
    frames = [x.detach().cpu().numpy()[:, :2]]
    with torch.no_grad():
        if hasattr(integrator, "map_to_time"):
            for i in range(1, n_total + 1):
                y = integrator.map_to_time(x, i / n_total, forward=False)
                frames.append(y.detach().cpu().numpy()[:, :2])
        else:
            dt = 1.0 / n_total
            for k in range(n_total):
                t0 = 1.0 - k * dt
                t1 = t0 - dt
                x = integrator._integrate(
                    x, sign=-1.0, jac=False, duration=dt, n_steps=1, t_span=(t0, t1)
                )
                frames.append(x.detach().cpu().numpy()[:, :2])
    return np.stack(frames, axis=0)


def pathline_speed(frames: Array) -> Array:
    """Per-vertex speed from discrete inverse RK frames, length per unit ``τ``.

    The composed inverse occupies ``τ ∈ [0, 1]``, so each recorded step is
    ``1 / (n_frames - 1)``. Use this when the event has several sequential
    velocity factors and there is no single ``v(x)`` to sample.
    """

    frames = np.asarray(frames, dtype=float)
    if frames.ndim != 3 or frames.shape[0] < 2:
        return np.zeros(frames.shape[:2], dtype=float)
    dtau = 1.0 / (len(frames) - 1)
    segment = np.linalg.norm(np.diff(frames[..., :2], axis=0), axis=-1) / dtau
    speed = np.empty(frames.shape[:2], dtype=float)
    speed[0] = segment[0]
    speed[-1] = segment[-1]
    if len(segment) > 1:
        speed[1:-1] = 0.5 * (segment[:-1] + segment[1:])
    return speed


_PATHLINE_CASING = "0.15"
_ANCHOR_CASING = "#CC3311"


def horizon_envelope(horizons: Sequence[Array], x: Array) -> tuple[Array, Array]:
    """Lowest and highest horizon z at each ``x``, by piecewise-linear traces."""

    query = np.asarray(x, dtype=float)
    low = np.full(query.shape, np.nan)
    high = np.full(query.shape, np.nan)
    for horizon in horizons:
        pts = np.asarray(horizon, dtype=float)
        if len(pts) < 2:
            continue
        order = np.argsort(pts[:, 0])
        depth = np.interp(query, pts[order, 0], pts[order, 1], left=np.nan, right=np.nan)
        low = np.fmin(low, depth)
        high = np.fmax(high, depth)
    return low, high


def strain_sample_points(
    horizons: Sequence[Array],
    *,
    grid=None,
    x: Array | None = None,
    z: Array | None = None,
    spacing: float | None = None,
    pad: float = 0.15,
    max_points: int = 8_000,
    seed: int = 0,
) -> tuple[Array, float]:
    """Dense centres inside the fold package, not the padded evaluation box.

    Poisson-disk samples are kept only between the uppermost and lowermost
    contacts at that ``x`` (plus a small vertical pad so the outer beds are
    included). Returns ``(points, spacing)``.
    """

    if grid is not None:
        axes = getattr(grid, "tformed_axes", None) or grid.axes
        x = np.asarray(axes[0], dtype=float)
        z = np.asarray(axes[1], dtype=float)
    if x is None or z is None:
        raise ValueError("strain_sample_points needs a grid or x, z axes")
    if spacing is None:
        spacing = 0.045 * abs(float(z[-1] - z[0]))
    spacing = max(float(spacing), 1e-6)
    n_draw = max(int(max_points) * 5, 8_000)
    if grid is not None:
        points = np.asarray(
            grid.sample(N=n_draw, poissonDisk=(spacing, n_draw, seed)), dtype=float
        )
    else:
        points = _regular_grid_sample(x, z, n_draw)
    low, high = horizon_envelope(horizons, points[:, 0])
    thick = np.clip(high - low, 0.0, None)
    pad_z = float(pad) * thick
    keep = (
        np.isfinite(low)
        & np.isfinite(high)
        & (points[:, 1] >= low - pad_z)
        & (points[:, 1] <= high + pad_z)
    )
    points = points[keep]
    if len(points) > max_points:
        points = points[: int(max_points)]
    return points, spacing


def _median_nearest_spacing(points: Array) -> float:
    points = _as_points(points)[:, :2]
    if len(points) < 2:
        return 1.0
    delta = points[:, None, :] - points[None, :, :]
    dist = np.linalg.norm(delta, axis=-1)
    np.fill_diagonal(dist, np.inf)
    return float(np.median(dist.min(axis=1)))


def strain_sample_points_reference(
    event,
    horizons: Sequence[Array],
    *,
    n_along: int = 48,
    n_between: int = 6,
    t_min: float = 0.08,
    t_max: float = 0.92,
) -> tuple[Array, Array, float]:
    """Fill a layer in the restored frame and return its present-day images.

    Present-day curvature is not used. The two bounding contacts are
    inverse-mapped, the unit between them is sampled as a graph over restored
    *x*, and those markers are pushed through $\\Phi$. Glyphs should be drawn
    at the present-day points; ``radius`` is the restored-frame spacing so
    ellipse size is set in the undeformed metric.

    Returns ``(present, reference, radius)``.
    """

    if len(horizons) < 2:
        raise ValueError("strain_sample_points_reference needs two bounding contacts")
    integrator = getattr(getattr(event, "field", None), "integrator", None)
    if integrator is None:
        raise TypeError("event.field.integrator is required")
    from curlew.kinematics.constraints import points_between_horizons

    device, dtype = _field_device_dtype(event.field)
    traces = [
        torch.as_tensor(_as_points(h)[:, :2], device=device, dtype=dtype)
        for h in (horizons[0], horizons[-1])
    ]
    with torch.no_grad():
        restored = [integrator.inverse_map(h) for h in traces]
        reference = points_between_horizons(
            restored[0],
            restored[1],
            n_along=int(n_along),
            n_between=int(n_between),
            t_min=float(t_min),
            t_max=float(t_max),
            depth_axis=getattr(event.field, "depth_axis", 1),
            parameterization="graph",
        )
        present = integrator.forward_map(reference)
    present_np = _as_numpy(present)[:, :2]
    reference_np = _as_numpy(reference)[:, :2]
    axis = int(getattr(event.field, "depth_axis", 1)) % 2
    thick = float(np.ptp(reference_np[:, axis]))
    spacing = _median_nearest_spacing(reference_np)
    # Glyph size follows restored thickness, not the sample pitch, or a dense
    # fill collapses to a dashed stitch along the layer.
    radius = max(0.45 * thick, 3.0 * spacing, 1e-6)
    return present_np, reference_np, radius


def strain_measures(event, points: Array) -> dict:
    """Principal stretches of ``F = (DΦ^{-1})^{-1}`` at present-day ``points``.

    The spatial strain ellipse is the image of the unit circle under ``F``:
    left singular vectors give the axes, singular values the stretches.
    """

    integrator = getattr(getattr(event, "field", None), "integrator", None)
    if integrator is None:
        raise TypeError("event.field.integrator is required")
    device, dtype = _field_device_dtype(event.field)
    x = torch.as_tensor(_as_points(points)[:, :2], device=device, dtype=dtype)
    with torch.no_grad():
        _, jac = integrator.inverse_map_with_jacobian(x)
        deformation = torch.linalg.inv(jac)
        left, singular, _ = torch.linalg.svd(deformation)
        angle = torch.atan2(left[:, 1, 0], left[:, 0, 0])
        stretch = singular.clamp_min(1e-12)
    return dict(
        points=_as_numpy(x)[:, :2],
        stretches=_as_numpy(stretch),
        angles=np.degrees(_as_numpy(angle)),
        Rs=_as_numpy(stretch[:, 0] / stretch[:, 1]),
        det=_as_numpy(torch.linalg.det(deformation)),
    )


def overlay_strain_ellipses(
    ax,
    measures: dict,
    *,
    radius: float,
    normalize: bool = True,
    cmap="YlOrRd",
    vmax: float | None = None,
    facecolor=None,
    edgecolor="0.15",
    linewidth: float = 0.25,
):
    """Draw spatial strain ellipses.

    ``radius`` is the radius of the reference circle. With ``normalize=False``
    the glyph is that circle's image under ``F``, so two panels that pass the
    same radius share one scale. ``normalize=True`` divides by the median
    major stretch and draws the median long axis at ``0.9 * radius``.

    With ``facecolor`` set, glyphs are a flat fill (e.g. white on lithology).
    Otherwise colour is flattening ``R_s = λ₁/λ₂``.
    """

    stretch = np.asarray(measures["stretches"], dtype=float)
    rs = np.asarray(measures["Rs"], dtype=float)
    points = np.asarray(measures["points"], dtype=float)
    if normalize:
        scale = 0.9 * float(radius) / (2.0 * max(float(np.median(stretch[:, 0])), 1e-6))
    else:
        scale = float(radius)
    kwargs = dict(
        widths=2.0 * scale * stretch[:, 0],
        heights=2.0 * scale * stretch[:, 1],
        angles=measures["angles"],
        units="xy",
        offsets=points,
        offset_transform=ax.transData,
        edgecolors=edgecolor,
        linewidths=linewidth,
    )
    if facecolor is None:
        if vmax is None:
            vmax = float(max(np.percentile(rs, 98) if len(rs) else 1.01, 1.01))
        kwargs.update(array=rs, cmap=cmap, norm=LogNorm(vmin=1.0, vmax=vmax))
    else:
        kwargs["facecolors"] = facecolor
    collection = EllipseCollection(**kwargs)
    collection.set_zorder(4)
    ax.add_collection(collection)
    return collection


def strain_handles():
    """Legend proxy for filled white strain ellipses."""

    return [
        Patch(facecolor="white", edgecolor="0.2", linewidth=0.7, label="Strain"),
    ]


def pathline_handles():
    """Legend proxies matching the dark path casing and red anchor casing."""

    return [
        Line2D(
            [0],
            [0],
            color="0.45",
            lw=1.35,
            solid_capstyle="round",
            path_effects=[withStroke(linewidth=2.6, foreground=_PATHLINE_CASING)],
            label="Pathline",
        ),
        Line2D(
            [0],
            [0],
            color="0.45",
            lw=1.35,
            marker="s",
            markersize=6.5,
            markerfacecolor="white",
            markeredgecolor=_ANCHOR_CASING,
            markeredgewidth=0.9,
            solid_capstyle="round",
            path_effects=[withStroke(linewidth=3.1, foreground=_ANCHOR_CASING)],
            label="Anchor",
        ),
    ]


def overlay_pathlines(
    ax,
    frames: Array,
    *,
    values: Array | None = None,
    cmap="viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    highlight: Array | None = None,
    color: str = "0.2",
    linewidth: float = 1.05,
):
    """Draw inverse-flow polylines, optionally coloured by a scalar along the path.

    ``values`` has shape ``(n_steps+1, n_seeds)`` (sampled at the vertices).
    Each segment is coloured by the mean of its endpoints. Returns the
    :class:`~matplotlib.collections.LineCollection` when ``values`` is set,
    so it can drive a colour bar.
    """

    frames = np.asarray(frames, dtype=float)
    if frames.ndim != 3 or frames.shape[-1] < 2:
        raise ValueError("frames must have shape (n_steps+1, n_seeds, 2)")
    segs = np.moveaxis(frames[..., :2], 0, 1)
    starts = segs[:, 0]
    highlight_index = None
    if highlight is not None and len(starts):
        highlight = np.asarray(highlight, dtype=float).reshape(-1)[:2]
        highlight_index = int(np.argmin(np.linalg.norm(starts - highlight, axis=1)))

    collection = None
    if values is not None:
        values = np.asarray(values, dtype=float)
        if values.shape != frames.shape[:2]:
            raise ValueError(
                "values must have shape (n_steps+1, n_seeds) matching frames"
            )
        pieces = np.stack([segs[:, :-1], segs[:, 1:]], axis=2).reshape(-1, 2, 2)
        seg_val = 0.5 * (values[:-1] + values[1:])
        flat_val = np.moveaxis(seg_val, 0, 1).reshape(-1)
        if vmin is None:
            vmin = float(np.nanmin(values))
        if vmax is None:
            vmax = float(np.nanmax(values))
        ordinary = (
            np.delete(segs, highlight_index, axis=0)
            if highlight_index is not None
            else segs
        )
        if len(ordinary):
            ax.add_collection(
                LineCollection(
                    ordinary,
                    colors=_PATHLINE_CASING,
                    linewidths=linewidth + 0.9,
                    zorder=2.9,
                    capstyle="round",
                    joinstyle="round",
                )
            )
        if highlight_index is not None:
            ax.add_collection(
                LineCollection(
                    segs[highlight_index : highlight_index + 1],
                    colors=_ANCHOR_CASING,
                    linewidths=linewidth + 1.55,
                    zorder=2.95,
                    capstyle="round",
                    joinstyle="round",
                )
            )
        collection = LineCollection(
            pieces,
            cmap=cmap,
            norm=Normalize(vmin=vmin, vmax=vmax),
            linewidths=linewidth,
            zorder=3.2,
            capstyle="round",
            joinstyle="round",
        )
        collection.set_array(flat_val)
        ax.add_collection(collection)
        start_color = collection.cmap(collection.norm(values[0]))
        ax.scatter(
            starts[:, 0],
            starts[:, 1],
            s=11,
            marker="o",
            facecolors=start_color,
            edgecolors="0.15",
            linewidths=0.35,
            zorder=5,
        )
    else:
        collection = LineCollection(
            segs,
            colors=color,
            linewidths=linewidth,
            zorder=3,
            capstyle="round",
            joinstyle="round",
        )
        collection.set_path_effects(
            [withStroke(linewidth=linewidth + 0.85, foreground="0.15")]
        )
        ax.add_collection(collection)
        ax.scatter(
            starts[:, 0],
            starts[:, 1],
            s=11,
            marker="o",
            facecolors=color,
            edgecolors="0.15",
            linewidths=0.35,
            zorder=5,
        )

    if highlight_index is not None:
        ax.scatter(
            [highlight[0]],
            [highlight[1]],
            s=42,
            marker="s",
            facecolors="white",
            edgecolors=_ANCHOR_CASING,
            linewidths=0.8,
            zorder=9,
        )
    return collection


def plot_critical_comparison(
    cases: Sequence[tuple[str, Array, CriticalPoints]],
    x: Array,
    z: Array,
    *,
    xlim=None,
    zlim=None,
    cmap="viridis",
):
    """Side-by-side scalar fields with full critical-point classification."""

    fig, axes = plt.subplots(
        1, len(cases), figsize=(5.1 * len(cases), 4.3), sharex=True, sharey=True
    )
    axes = np.atleast_1d(axes)
    all_scalar = np.concatenate([np.ravel(case[1]) for case in cases])
    vmin, vmax = np.percentile(all_scalar, [1.0, 99.0])
    for ax, (label, scalar, critical) in zip(axes, cases):
        ax.pcolormesh(
            x,
            z,
            np.asarray(scalar).T,
            cmap=cmap,
            shading="auto",
            rasterized=True,
        )
        overlay_critical_points(
            ax, critical, include_saddles=True, include_degenerate=True, size=36
        )
        n_ext = critical.extrema_count
        ax.set_title(f"{label}: {n_ext} extrema")
        ax.set_aspect("equal")
        if xlim is not None and zlim is not None:
            ax.set_xlim(*xlim)
            ax.set_ylim(*zlim)
    return fig, axes


def plot_critical_resolution(checks: Sequence[dict], scalar: Array | None = None):
    """Plot roots found at each candidate-grid resolution."""

    fig, axes = plt.subplots(1, len(checks), figsize=(3.5 * len(checks), 3.1), squeeze=False)
    for ax, item in zip(axes[0], checks):
        if scalar is not None and scalar.shape == (len(item["x"]), len(item["z"])):
            ax.contourf(item["x"], item["z"], scalar.T, levels=25, cmap="viridis")
        overlay_critical_points(ax, item["critical"], include_saddles=True)
        ax.set_title(f'{item["shape"][0]} × {item["shape"][1]}')
        ax.set_aspect("equal")
    return fig, axes[0]


def event_velocities(event) -> list:
    """Velocity factors on ``event.field``, one module per composite band.

    Prefers ``integrator.velocities`` (the owner of a composite flow). A
    single-factor event still returns a one-element list. Empty if the field
    exposes neither an integrator nor a velocity.
    """

    field = getattr(event, "field", None)
    integrator = getattr(field, "integrator", None)
    if integrator is not None and hasattr(integrator, "velocities"):
        return list(integrator.velocities)
    velocity = getattr(field, "velocity", None)
    if velocity is None and integrator is not None:
        velocity = getattr(integrator, "velocity", None)
    if velocity is None:
        return []
    if isinstance(velocity, (list, tuple)) or type(velocity).__name__ == "ModuleList":
        return list(velocity)
    return [velocity]


def evaluate_velocities(event, points: Array, *, batch_size: int = 65_536) -> list[Array]:
    """Evaluate each velocity factor at ``points``. Returns a list of ``(N, d)``."""

    field = getattr(event, "field", None)
    return [
        _batched_tensor_call(velocity, points, field=field, batch_size=batch_size)
        for velocity in event_velocities(event)
    ]


def _event_velocity(event):
    velocities = event_velocities(event)
    if not velocities:
        return None
    if len(velocities) > 1:
        raise AttributeError(
            f"composite flow with {len(velocities)} velocity factors; "
            "use event_velocities(event)"
        )
    return velocities[0]


def _fourier_slot(velocity):
    fsf = getattr(velocity, "stream", None)
    if fsf is not None and hasattr(fsf, "omega_spatial"):
        return fsf
    return next(
        (
            slot
            for slot in (getattr(velocity, "alpha", None), getattr(velocity, "beta", None))
            if hasattr(slot, "omega_spatial")
        ),
        None,
    )


def _band_title(index: int, n: int, wavelength: Array) -> str:
    if n == 2:
        name = "Coarse" if index == 0 else "Fine"
    else:
        name = f"Band {index + 1}"
    wavelength = np.asarray(wavelength)
    if not len(wavelength):
        return name
    lo, hi = float(np.min(wavelength)), float(np.max(wavelength))
    return rf"{name}: $\lambda$ {lo / 1000:.2g}–{hi / 1000:.2g} km"


def _draw_wavevector_panel(ax, velocity, *, cmap: str, title: str | None):
    fsf = _fourier_slot(velocity)
    if fsf is None:
        raise TypeError("velocity contains no Fourier scalar field")
    with torch.no_grad():
        vectors = _as_numpy(fsf.omega_spatial)
        amplitude = _as_numpy(fsf._effective_amplitudes()).reshape(-1)
    norm = Normalize(vmin=float(amplitude.min()), vmax=float(amplitude.max()) or 1.0)
    kx, kz = vectors[:, 0], vectors[:, 1]
    theta = np.mod(np.arctan2(kz, kx), np.pi)
    wavelength_m = 2 * np.pi / np.maximum(np.hypot(kx, kz), 1e-30)
    scatter = ax.scatter(
        theta,
        wavelength_m / 1000,
        c=amplitude,
        cmap=cmap,
        norm=norm,
        s=28,
        edgecolors="white",
        linewidths=0.35,
    )
    ax.set_theta_zero_location("E")
    ax.set_theta_direction(1)
    ax.set_thetamin(0)
    ax.set_thetamax(180)
    ax.set_thetagrids([0, 45, 90, 135, 180], labels=[r"$+x$", "", r"$+z$", "", r"$-x$"])
    if title:
        ax.set_title(title)
    ax.figure.colorbar(scatter, ax=ax, label=r"Effective fitted amplitude")
    return scatter, wavelength_m


def plot_wavevector_spectrum(
    event,
    *,
    ax=None,
    cmap: str = "viridis",
    title: str = "Wavevector spectrum",
):
    """Plot Euler-potential wavevectors, coloured by their phase amplitudes.

    One polar panel per velocity factor. A single-factor event keeps the
    original ``(ax, scatter)`` return; several factors return ``(fig, axes)``.
    """

    velocities = event_velocities(event)
    if not velocities:
        raise TypeError("event field exposes no velocity")
    n = len(velocities)
    if n == 1:
        if ax is None:
            _, ax = plt.subplots(figsize=(5.2, 4.2), subplot_kw=dict(projection="polar"))
        elif getattr(ax, "name", "") != "polar":
            ax.remove()
            ax = ax.figure.add_subplot(111, projection="polar")
        scatter, _ = _draw_wavevector_panel(ax, velocities[0], cmap=cmap, title=title)
        return ax, scatter
    fig, axes = plt.subplots(
        1, n, figsize=(5.2 * n, 4.2), subplot_kw=dict(projection="polar")
    )
    for i, (axis, velocity) in enumerate(zip(np.atleast_1d(axes), velocities)):
        _, wavelength_m = _draw_wavevector_panel(axis, velocity, cmap=cmap, title=None)
        axis.set_title(_band_title(i, n, wavelength_m))
    return fig, axes


def _upper_hemisphere(vectors: Array) -> Array:
    unit = vectors / np.linalg.norm(vectors, axis=1, keepdims=True).clip(1e-30)
    return np.where(unit[:, 1:2] < 0, -unit, unit)


def _polar_kernel_sum(grid_n, grid_loglam, data_n, data_loglam, values, kappa, sigma):
    output = np.zeros(len(grid_n))
    for start in range(0, len(data_n), 256):
        direction = data_n[start : start + 256]
        loglam = data_loglam[start : start + 256]
        weight = np.asarray(values)[start : start + 256]
        angular = np.cosh(kappa * np.clip(grid_n @ direction.T, -1, 1))
        radial = (grid_loglam[:, None] - loglam[None, :]) / sigma
        output += (angular * np.exp(-0.5 * radial * radial)) @ weight
    return output


def _weighted_percentile(values, weights, percentile):
    values, weights = np.asarray(values), np.asarray(weights)
    keep = np.isfinite(values) & np.isfinite(weights) & (values > 0) & (weights > 0)
    values, weights = values[keep], weights[keep]
    if not len(values):
        return 1e-6
    order = np.argsort(values)
    cdf = np.cumsum(weights[order])
    cdf /= cdf[-1]
    return float(np.interp(percentile / 100, cdf, values[order]))


def _shear_modes_raw(velocity):
    """Every row of the velocity's shear decomposition: ``((direction, wavelength, amplitude), keep)``,
    with ``keep`` the rows :func:`shear_modes` reports (finite, non-zero)."""
    if not hasattr(velocity, "shear_decomposition"):
        raise TypeError("velocity has no cubic shear decomposition")
    with torch.no_grad():
        coefficient, omega, _ = velocity.shear_decomposition()
    amplitude = np.linalg.norm(_as_numpy(coefficient), axis=-1)
    omega = _as_numpy(omega)
    direction = _upper_hemisphere(omega)
    wavenumber = np.linalg.norm(omega, axis=1)
    wavelength = 2 * np.pi / np.clip(wavenumber, 1e-30, None)
    keep = np.isfinite(wavelength) & np.isfinite(amplitude) & (amplitude > 0)
    return (direction, wavelength, amplitude), keep


def _shear_modes(velocity):
    (direction, wavelength, amplitude), keep = _shear_modes_raw(velocity)
    return direction[keep], wavelength[keep], amplitude[keep]


def _as_band_ranges(length_scale_range) -> list[tuple[float, float]]:
    """Normalise ``length_scale_range`` to a list of ``(lo, hi)`` pairs.

    Accepts a single pair, a sequence of pairs (one per velocity factor), or
    ``None``. A flat two-vector is one band; nested sequences are per-factor
    bands, which may overlap or not tile.
    """

    if length_scale_range is None:
        return []
    arr = np.asarray(length_scale_range, dtype=float)
    if arr.size == 0:
        return []
    if arr.ndim == 1:
        if arr.size != 2:
            raise ValueError(
                "length_scale_range must be (lo, hi) or a sequence of (lo, hi) pairs"
            )
        return [(float(arr[0]), float(arr[1]))]
    if arr.ndim == 2 and arr.shape[-1] == 2:
        return [(float(lo), float(hi)) for lo, hi in arr]
    raise ValueError(
        "length_scale_range must be (lo, hi) or a sequence of (lo, hi) pairs"
    )


def _length_scale_edges(length_scale_range) -> list[float]:
    """Unique wavelengths to mark on the radial axis, metres."""

    edges = []
    for lo, hi in _as_band_ranges(length_scale_range):
        edges.extend((lo, hi))
    return edges


def _shear_spectrum_limits(wavelengths, length_scale_range=None):
    parts = [np.asarray(w).ravel() for w in wavelengths if len(np.asarray(w))]
    stacked = np.concatenate(parts) if parts else np.array([])
    bands = _as_band_ranges(length_scale_range)
    span = (
        (min(lo for lo, _ in bands), max(hi for _, hi in bands))
        if bands else None
    )
    if len(stacked):
        low, high = np.percentile(stacked, [2, 98])
        low, high = max(float(low), 1e-9), max(float(high), float(low) * 1.01)
    elif span is not None:
        low, high = span
    else:
        low, high = 1.0, 2.0
    if span is not None:
        low = min(low, float(span[0]))
        high = max(high, 1.05 * float(span[1]))
    return low, high


def _shear_spectrum_image(
    direction,
    wavelength,
    amplitude,
    *,
    n_grid,
    kappa,
    sigma_log,
    value_gamma,
    high,
    vlim=None,
    signed=False,
):
    rim = high
    grid = np.linspace(-1, 1, n_grid)
    xx, zz = np.meshgrid(grid, grid)
    rr = xx * xx + zz * zz
    inside = (rr <= 1) & (zz >= 0) & (rr > 1e-12)
    radius = np.sqrt(rr[inside])
    grid_direction = np.column_stack((xx[inside] / radius, zz[inside] / radius))
    # Radius maps linearly onto wavelength, origin at 0 and the rim at `high`.
    # Clamping the inside of the band up to `low` would give that whole disk
    # one wavelength, so a mode on the inner edge matches it all the way to
    # the origin. Leave the radius alone and let the kernel die inward.
    grid_wavelength = radius * rim
    density = _polar_kernel_sum(
        grid_direction, np.log(grid_wavelength), direction, np.log(wavelength),
        np.ones_like(amplitude), kappa, sigma_log,
    )
    weighted = _polar_kernel_sum(
        grid_direction, np.log(grid_wavelength), direction, np.log(wavelength),
        amplitude, kappa, sigma_log,
    )
    mean_amplitude = weighted / np.clip(density, 1e-30, None)
    if vlim is None:
        amin, amax = _spectrum_vlim(mean_amplitude, density, signed=signed)
    else:
        amin, amax = (float(vlim[0]), float(vlim[1]))
    # signed values (a displacement share) on a diverging scale centred on zero
    norm, cmap = Normalize(amin, amax), plt.colormaps["RdBu_r" if signed else "rainbow"]
    image = np.ones(xx.shape + (3,))
    brightness = np.clip(density / max(float(density.max()), 1e-30), 0, 1) ** value_gamma
    image[inside] = figstyle.bright_rgb(norm(mean_amplitude), brightness, cmap)
    return image, amin, amax, rim, norm, cmap


def _spectrum_vlim(mean_amplitude, density, *, signed=False) -> tuple[float, float]:
    """Density-weighted 2nd and 98th percentiles of the kernel-mean amplitude
    (``signed``: symmetric about zero, at the larger of the two)."""

    amin = _weighted_percentile(mean_amplitude, density, 2)
    amax = _weighted_percentile(mean_amplitude, density, 98)
    if signed:
        lim = max(abs(amin), abs(amax), 1e-30)
        return -float(lim), float(lim)
    if amax <= amin:
        amax = amin + 1.0
    return float(amin), float(amax)


def _shear_spectrum_vlim(
    direction,
    wavelength,
    amplitude,
    *,
    n_grid,
    kappa,
    sigma_log,
    high,
    signed=False,
) -> tuple[float, float]:
    """Amplitude colour limits for one panel, on the same kernel as the image."""

    _, amin, amax, *_ = _shear_spectrum_image(
        direction, wavelength, amplitude,
        n_grid=n_grid, kappa=kappa, sigma_log=sigma_log,
        value_gamma=1.0, high=high, signed=signed,
    )
    return amin, amax


def _spectrum_tick_label(value: float) -> str:
    """Plain numerals through the thousands; ``.2g`` would print 500 as ``5e+02``."""

    value = float(value)
    magnitude = abs(value)
    if magnitude >= 10000 or (0 < magnitude < 0.01):
        return f"{value:.2g}"
    if abs(value - round(value)) <= 1e-6 * max(magnitude, 1.0):
        return f"{value:.0f}"
    return f"{value:.2g}"


def _draw_shear_spectrum_panel(
    ax,
    direction,
    wavelength,
    amplitude,
    *,
    n_grid,
    kappa,
    sigma_log,
    value_gamma,
    low,
    high,
    length_scale_range,
    title=None,
    vlim=None,
    wavelength_label=None,
    signed=False,
    value_name=r"$|C_k|$",
):
    image, amin, amax, rim, norm, cmap = _shear_spectrum_image(
        direction, wavelength, amplitude,
        n_grid=n_grid, kappa=kappa, sigma_log=sigma_log,
        value_gamma=value_gamma, high=high, vlim=vlim, signed=signed,
    )
    ax.imshow(image, extent=(-1, 1, -1, 1), origin="lower", interpolation="bilinear")
    ax.set(xlim=(-1.28, 1.48), ylim=(-0.22, 1.38))
    ax.set_aspect("equal")
    ax.axis("off")
    for degrees in (30, 60, 90, 120, 150):
        angle = np.deg2rad(degrees)
        ax.plot([0, np.cos(angle)], [0, np.sin(angle)], color="white", lw=0.55)
    ax.plot([-1, 1], [0, 0], color="0.25", lw=0.7)
    ax.add_patch(Arc((0, 0), 2, 2, theta1=0, theta2=180, color="0.25", lw=0.7))
    _annotate_shear_axes(ax, rim, length_scale_range=length_scale_range, wavelength_label=wavelength_label)
    if title:
        ax.set_title(title, pad=8)
    ticks = ticker.MaxNLocator(nbins=4, min_n_ticks=3).tick_values(amin, amax)
    ticks = [float(value) for value in ticks if amin <= value <= amax]
    figstyle.arc_colorbar(
        ax,
        cmap,
        norm,
        ticks,
        [_spectrum_tick_label(value) for value in ticks],
        val_gamma=value_gamma,
        amp_name=value_name,
        density_name=None,   # brightness = mode density, explained in captions
    )
    return ax


def plot_shear_spectrum(
    event,
    *,
    n_grid: int = 401,
    kappa: float = 100.0,
    sigma_log: float = 0.28,
    value_gamma: float = 0.60,
    length_scale_range=None,
):
    """Plot the smoothed cubic shear decomposition in direction-wavelength space.

    One panel per velocity factor, sharing a radial wavelength scale. A
    single-factor event keeps the original figure size and ``(fig, ax)``
    return; several factors return ``(fig, axes)``.

    ``length_scale_range`` may be one ``(lo, hi)`` pair or a sequence of
    pairs, one per factor. The shared rim spans every band; dashed rings
    mark that panel's own edges.
    """

    return plot_shear_spectrum_modes(
        shear_modes(event), n_grid=n_grid, kappa=kappa, sigma_log=sigma_log,
        value_gamma=value_gamma, length_scale_range=length_scale_range,
    )


def shear_modes(event) -> list[tuple[Array, Array, Array]]:
    """``(direction, wavelength, amplitude)`` of the cubic shear modes, one tuple per velocity factor.

    Plain arrays, so a fitting notebook can save them and a figure notebook can
    draw them with :func:`plot_shear_spectrum_modes` without the model.
    """

    velocities = event_velocities(event)
    if not velocities:
        raise TypeError("event field exposes no velocity")
    return [_shear_modes(v) for v in velocities]


def displacement_spectrum(event, points, *, measure: str = "share", batch_size: int = 16_384,
                          return_residual: bool = False):
    """``(direction, wavelength, displacement)`` of the shear modes, as the fitted map realises them.

    The amplitude spectrum (:func:`shear_modes`) gives each mode's learned
    coefficient. This gives what each mode does in the forward map Φ (restored
    frame → present day). A one-way sweep is a fixed sequence of shears, each
    mode adding ``u_k = h C_k cos(Ω_k·y + φ_k)`` to a point ``y`` where it sits
    when that mode's group acts, so the map's displacement splits exactly,

        U = Φ(y) − y = Σ_k u_k.

    ``measure="share"`` (default): each mode's share of that displacement,
    ``mean(u_k · U) / rms(U)`` over ``points``. The shares sum to ``rms(U)``
    exactly; a mode that moves material along the map's net displacement counts
    positive, one that works against it (cancelled by the others) negative.
    ``measure="rms"``: ``rms(|u_k|)``, how far the mode moves material whatever
    the direction (close to ``|C_k| / √2`` wherever the points span many of its
    wavelengths, so it mostly restates the amplitude).

    ``points`` are present-day points, pulled back through Φ⁻¹ to start the sweep;
    values are in the units of the coordinates. The modes, their order,
    ``direction`` and ``wavelength`` are those of :func:`shear_modes`, so each tuple
    pairs with its amplitude tuple; draw both with :func:`plot_shear_spectrum_modes`
    (``quantity="displacement"`` for the signed share).

    One-way shear sweeps only (``integrator="shear"``, ``n_steps=0``, phase split,
    no fault lift). ``return_residual`` also returns the largest distance between
    the replayed sweep's end point and the original point: zero up to round-off and
    the integrator's tabulated profiles, a check that the split is the map.
    """

    _require_torch()
    if measure not in ("share", "rms"):
        raise ValueError("measure is 'share' or 'rms'")
    integrator = getattr(getattr(event, "field", None), "integrator", None)
    if getattr(integrator, "integrator", None) != "shear":
        raise ValueError("displacement_spectrum needs a shear integrator")
    if any(int(n) != 0 for n in integrator.steps):
        raise ValueError("displacement_spectrum needs a one-way sweep (n_steps=0) for every factor")
    if getattr(integrator, "fault_lift", None) is not None or getattr(integrator, "lift_w", None) is not None:
        raise NotImplementedError("displacement_spectrum does not handle a fault lift")
    velocities = event_velocities(event)
    device, dtype = _field_device_dtype(event.field)
    x_all = _as_points(points)
    sums = [np.zeros(len(_shear_modes_raw(v)[1])) for v in velocities]
    u_sq, residual, count = 0.0, 0.0, 0

    def sweep(y, visit):
        # the forward one-way sweep, mode by mode: visit(factor, mode indices, (N, k, d) moves)
        for vel, n, span in integrator._plan(1.0):
            i = next(j for j, v in enumerate(velocities) if v is vel)
            C, om, off = vel.shear_decomposition()
            groups = integrator._shear_groups(om.detach())
            h = integrator._shear_step(0, len(groups), 1.0, 1.0)
            for idx in groups:                                             # longest wavelength first
                omg = om[idx]
                dhat = omg[0] / omg[0].norm().clamp_min(1e-30)
                phase = (y @ dhat).unsqueeze(-1) * (omg @ dhat) + off[idx]
                step = h * torch.cos(phase).unsqueeze(-1) * C[idx]
                visit(i, idx, step)
                y = y + step.sum(1)
        return y

    for start in range(0, len(x_all), batch_size):
        x = torch.as_tensor(x_all[start:start + batch_size], device=device, dtype=dtype)
        with torch.no_grad():
            if getattr(integrator, "shear_split", None) == "normal":
                for vel, n, span in integrator._plan(1.0):
                    if integrator._ridge_plan(vel, x, None) is not None:
                        raise NotImplementedError("displacement_spectrum needs the phase split; "
                                                  "this velocity is integrated by normal")
            y0 = x.clone()
            for vel, n, span in integrator._plan(-1.0):                     # back to the restored frame
                y0 = integrator._integrate(y0, sign=-1.0, jac=False, duration=1.0, n_steps=n,
                                           velocity=vel, t_span=span)
            if measure == "rms":
                def add_length(i, idx, step):
                    sums[i][idx.cpu().numpy()] += _as_numpy((step ** 2).sum(-1).sum(0))

                end = sweep(y0, add_length)
            else:
                end = sweep(y0, lambda i, idx, step: None)                  # the map's displacement first
                U = end - y0
                u_sq += float((U ** 2).sum())

                def add_share(i, idx, step):
                    sums[i][idx.cpu().numpy()] += _as_numpy((step * U.unsqueeze(1)).sum(-1).sum(0))

                sweep(y0, add_share)
            residual = max(residual, float((end - x).norm(dim=-1).max()))
        count += len(x)

    out = []
    for vel, total in zip(velocities, sums):
        keep = _shear_modes_raw(vel)[1]
        if measure == "rms":
            value = np.sqrt(total / max(count, 1))
        else:
            value = (total / max(count, 1)) / max(np.sqrt(u_sq / max(count, 1)), 1e-30)
        out.append(tuple(_shear_modes(vel)[:2]) + (value[keep],))
    return (out, residual) if return_residual else out


def plot_shear_spectrum_modes(
    modes,
    *,
    n_grid: int = 401,
    kappa: float = 100.0,
    sigma_log: float = 0.28,
    value_gamma: float = 0.60,
    length_scale_range=None,
    titles=None,
    share_scale: bool = False,
    wavelength_label=None,
    quantity: str = "amplitude",
):
    """:func:`plot_shear_spectrum` from saved :func:`shear_modes` (numpy only).

    ``modes`` may also hold different fits side by side (e.g. two integrators
    on the same basis): ``titles`` names the panels, and ``share_scale`` puts
    every panel on one amplitude colour scale so they can be compared.
    ``wavelength_label`` replaces ``λ (km)`` when coordinates are not metres
    (ticks stay wavelength / 1000).

    ``quantity="amplitude"`` (default) draws the amplitude spectrum, the third
    entry of each tuple being ``|C_k|``. ``quantity="displacement"`` draws the
    displacement spectrum of :func:`displacement_spectrum`: the third entry is a
    signed share, drawn on a diverging scale centred on zero.
    """

    if quantity not in ("amplitude", "displacement"):
        raise ValueError("quantity is 'amplitude' or 'displacement'")
    signed = quantity == "displacement"
    value_name = r"$\bar{u}_k$" if signed else r"$|C_k|$"
    bands = _as_band_ranges(length_scale_range)
    low, high = _shear_spectrum_limits(
        [wavelength for _, wavelength, _ in modes], length_scale_range
    )
    n = len(modes)
    vlim = None
    if share_scale:
        limits = [
            _shear_spectrum_vlim(d, w, a, n_grid=n_grid, kappa=kappa, sigma_log=sigma_log, high=high,
                                 signed=signed)
            for d, w, a in modes
        ]
        vlim = (min(lo for lo, _ in limits), max(hi for _, hi in limits))
        if signed:
            lim = max(abs(vlim[0]), abs(vlim[1]))
            vlim = (-lim, lim)
    fig, axes = plt.subplots(
        1, n, figsize=(6.8 if n == 1 else 5.6 * n, 4.15), squeeze=False
    )
    drawn = []
    for i, ((direction, wavelength, amplitude), ax) in enumerate(zip(modes, axes[0])):
        if titles is not None:
            title = titles[i]
        else:
            title = _band_title(i, n, wavelength) if n > 1 else None
        panel_range = bands[i] if i < len(bands) else (
            (min(lo for lo, _ in bands), max(hi for _, hi in bands)) if bands else None
        )
        drawn.append(
            _draw_shear_spectrum_panel(
                ax, direction, wavelength, amplitude,
                n_grid=n_grid, kappa=kappa, sigma_log=sigma_log,
                value_gamma=value_gamma, low=low, high=high,
                length_scale_range=panel_range, title=title, vlim=vlim,
                wavelength_label=wavelength_label, signed=signed, value_name=value_name,
            )
        )
    return fig, drawn[0] if n == 1 else np.asarray(drawn)


def _annotate_shear_axes(ax, rim_metres, *, length_scale_range=None, wavelength_label=None) -> None:
    """Direction labels and radial wavelength ticks; Cartesian axes stay off.

    Ticks are wavelengths / 1000. ``wavelength_label`` replaces the default
    ``λ (km)`` when the coordinates are not metres.
    """

    font = figstyle._fp(size=9)
    tick_font = figstyle._fp(size=8)
    for degrees, label, ha, va in (
        (0.0, r"$+x$", "left", "center"),
        (90.0, r"$+z$", "center", "bottom"),
        (180.0, r"$-x$", "right", "center"),
    ):
        angle = np.deg2rad(degrees)
        radius = 1.12
        ax.text(
            radius * np.cos(angle),
            radius * np.sin(angle),
            label,
            ha=ha,
            va=va,
            color="0.15",
            fontproperties=font,
            clip_on=False,
        )

    rim_km = float(rim_metres) / 1000.0
    tick_km = []
    for metres in _length_scale_edges(length_scale_range):
        tick_km.append(float(metres) / 1000.0)
        radius = float(metres) / float(rim_metres)
        if 0.08 < radius <= 1.0:
            ax.add_patch(
                Arc(
                    (0, 0), 2 * radius, 2 * radius,
                    theta1=0, theta2=180, color="0.35", lw=0.7, ls="--",
                )
            )
    # Band edges first; round values only where they clear every placed tick,
    # so labels near the centre do not run together.
    placed = []
    for value in tick_km + [0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0]:
        if not 0.12 * rim_km <= value <= 0.98 * rim_km and value not in tick_km:
            continue
        if all(abs(value - p) / rim_km >= 0.11 for p in placed):
            placed.append(value)
    for value in sorted(placed):
        radius = value / rim_km
        ax.text(
            radius,
            -0.055,
            f"{value:g}",
            ha="center",
            va="top",
            color="0.2",
            fontproperties=tick_font,
            clip_on=False,
        )
    ax.text(
        0.0,
        -0.16,
        r"$\lambda\ (\mathrm{km})$" if wavelength_label is None else wavelength_label,
        ha="center",
        va="top",
        color="0.15",
        fontproperties=font,
        clip_on=False,
    )


def _lower_hemisphere(vectors: Array) -> Array:
    unit = vectors / np.linalg.norm(vectors, axis=1, keepdims=True).clip(1e-30)
    return np.where(unit[:, 2:3] > 0.0, -unit, unit)


def _schmidt_grid(n: int):
    """Lambert disk → unit vectors (lower hemisphere). Area is preserved."""
    g = np.linspace(-1.0, 1.0, n)
    xx, yy = np.meshgrid(g, g)
    rr = xx * xx + yy * yy
    nz = rr - 1.0
    den = np.sqrt(np.clip(2.0 - rr, 0.0, None))
    nn = np.stack((xx * den, yy * den, nz), axis=-1)
    return xx, yy, rr, nn


def _axial_kernel_sum(grid, data, values, kappa):
    """sum_i values_i cosh(κ n·c_i) on RP² (identifies ±C)."""
    v = np.asarray(values, dtype=np.float64)
    out = np.zeros(len(grid), dtype=np.float64)
    chunk = 512
    for i0 in range(0, len(data), chunk):
        d = data[i0 : i0 + chunk]
        vv = v[i0 : i0 + chunk]
        c = np.clip(grid @ d.T, -1.0, 1.0)
        out += np.cosh(kappa * c) @ vv
    return out


def _amp_tick_labels(ticks):
    labs = []
    for t in ticks:
        if t <= 0:
            labs.append("0")
        elif 0.01 <= t < 100:
            labs.append(f"{t:.2g}")
        else:
            mant, exp = f"{t:.0e}".split("e")
            labs.append(f"{mant}e{int(exp):d}")
    return labs


def _km_tick_label(metres):
    v = metres / 1000.0
    return f"{v:.0f}" if abs(v - round(v)) < 0.05 else f"{v:g}"


def _log_norm_from_density(field, dens, pctl):
    vmin = _weighted_percentile(field, dens, pctl[0])
    vmax = _weighted_percentile(field, dens, pctl[1])
    if not np.isfinite(vmin) or vmin <= 0.0:
        vmin = 1e-6
    if not np.isfinite(vmax) or vmax <= vmin:
        vmax = vmin * 10.0
    if vmax / vmin >= 10.0:
        subs = np.array([1.0, 2.0, 5.0])
        exps = np.arange(np.floor(np.log10(vmin)) - 1, np.ceil(np.log10(vmax)) + 2)
        cand = (10.0 ** exps)[:, None] * subs[None, :]
        ticks = [float(t) for t in cand.ravel() if vmin * 0.98 <= t <= vmax * 1.02]
    else:
        ticks = [
            float(t)
            for t in ticker.MaxNLocator(nbins=4, min_n_ticks=3).tick_values(vmin, vmax)
            if t > 0
        ]
    ticks = [t for t in ticks if vmin <= t <= vmax]
    if len(ticks) < 2:
        ticks = [vmin, vmax]
    return LogNorm(vmin=vmin, vmax=vmax), ticks, vmin, vmax


def _schmidt_xy(vectors: Array) -> Array:
    """Project 3-D vectors to the Lambert disk used by :func:`_schmidt_grid`."""
    n = _lower_hemisphere(np.asarray(vectors, dtype=float).reshape(-1, 3))
    den = np.sqrt(np.clip(1.0 - n[:, 2], 1e-30, None))
    return np.column_stack((n[:, 0] / den, n[:, 1] / den))


def _hinge_from_event(event, velocities) -> Array | None:
    for velocity in velocities:
        slot = getattr(velocity, "hinge", None)
        direction = getattr(slot, "direction", None)
        if direction is not None:
            return np.asarray(direction.detach().cpu() if hasattr(direction, "detach") else direction, dtype=float)
    return None


def _draw_schmidt_net(ax) -> None:
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-1.42, 1.42)
    ax.set_ylim(-1.42, 1.42)
    ax.set_axis_off()
    for theta in (np.deg2rad(30), np.deg2rad(60)):
        r = float(np.sqrt(1.0 - np.cos(theta)))
        ax.add_patch(Circle(
            (0.0, 0.0), r, fill=False, edgecolor="white", lw=0.55, zorder=4,
        ))
    ax.plot([-1.0, 1.0], [0.0, 0.0], color="white", lw=0.7, zorder=4, solid_capstyle="butt")
    ax.plot([0.0, 0.0], [-1.0, 1.0], color="white", lw=0.45, zorder=4, solid_capstyle="butt")
    ax.add_patch(Circle(
        (0.0, 0.0), 1.0, fill=False, edgecolor="0.25", lw=0.7, zorder=5,
    ))
    font = figstyle._fp(size=10)
    ax.text(1.08, 0.0, r"$+x$", ha="left", va="center", color="0.15", fontproperties=font)
    ax.text(-1.08, 0.0, r"$-x$", ha="right", va="center", color="0.15", fontproperties=font)
    ax.text(0.0, 1.08, r"$+y$", ha="center", va="bottom", color="0.15", fontproperties=font)
    ax.text(0.0, -1.08, r"$-y$", ha="center", va="top", color="0.15", fontproperties=font)


def plot_shear_spectrum_3d(
    event,
    *,
    hinge=None,
    n_top: int = 5000,
    n_grid: int = 301,
    bandwidth_deg: float = 6.0,
    value_gamma: float = 0.60,
    amp_percentile=(10.0, 90.0),
    lam_percentile=(10.0, 90.0),
    direction: str = "shear",
):
    """Equal-area (Schmidt) nets of the 3-D cubic shear decomposition.

    Same layout as the recumbent-pericline supplement: both panels share the
    directional density of the strongest ``n_top`` shears. Hue in (a) is the
    local kernel mean of ``|C_k|``; hue in (b) is the local kernel mean of
    wavelength ``λ = 2π/|Ω|``. Lower hemisphere; opposite vectors identified.
    ``hinge`` is drawn as a point on both nets (taken from the Euler slot if
    omitted).

    ``direction`` chooses where each shear is placed: ``"shear"`` (default) by
    its coefficient ``C_k``, the way material moves; ``"wavevector"`` by
    ``Ω_k``, the direction across which that motion varies. With a fixed hinge
    every ``C_k`` lies in the plane normal to it, so the ``"shear"`` net is one
    girdle whatever the fit; on the ``"wavevector"`` net a cylindrical bank
    stays on that girdle and any variation along the hinge moves density
    towards the hinge point.
    """
    if direction not in ("shear", "wavevector"):
        raise ValueError(f"direction must be 'shear' or 'wavevector', got {direction!r}")
    try:
        import cmasher as cmr
        cmap_lam = cmr.gem
    except ImportError:
        cmap_lam = plt.colormaps["viridis"]

    velocities = event_velocities(event)
    if not velocities:
        raise TypeError("event field exposes no velocity")
    coeffs, omegas = [], []
    with torch.no_grad():
        for velocity in velocities:
            if not hasattr(velocity, "shear_decomposition"):
                raise TypeError("velocity has no cubic shear decomposition")
            coefficient, omega, _ = velocity.shear_decomposition()
            coeffs.append(coefficient)
            omegas.append(omega)
        coefficient = torch.cat(coeffs, dim=0)
        omega = torch.cat(omegas, dim=0)
        amp_t = coefficient.norm(dim=-1)
        n_all = int(amp_t.numel())
        n_keep = min(int(n_top), n_all)
        idx = torch.topk(amp_t, n_keep).indices
        ck = coefficient[idx].detach().cpu().numpy()
        om = omega[idx].detach().cpu().numpy()
        amp = amp_t[idx].detach().cpu().numpy()

    nC = _lower_hemisphere(ck if direction == "shear" else om)
    kmag = np.linalg.norm(om, axis=1)
    lam = 2.0 * np.pi / np.maximum(kmag, 1e-30)
    loglam = np.log(np.maximum(lam, 1e-30))

    xx, yy, rr, nn = _schmidt_grid(n_grid)
    inside = rr <= 1.0 + 1e-12
    grid = nn[inside]
    kappa = 1.0 / np.deg2rad(float(bandwidth_deg)) ** 2

    dens = _axial_kernel_sum(grid, nC, np.ones(n_keep), kappa)
    mass_amp = _axial_kernel_sum(grid, nC, amp, kappa)
    mass_loglam = _axial_kernel_sum(grid, nC, loglam, kappa)
    mean_amp = np.divide(mass_amp, dens, out=np.zeros_like(dens), where=dens > 0)
    mean_loglam = np.divide(mass_loglam, dens, out=np.zeros_like(dens), where=dens > 0)
    mean_lam = np.zeros_like(mean_loglam)
    np.exp(mean_loglam, out=mean_lam, where=dens > 0)
    dens_peak = float(np.max(dens)) if dens.size else 0.0

    norm_amp, ticks_amp, _, _ = _log_norm_from_density(mean_amp, dens, amp_percentile)
    norm_lam, ticks_lam, _, _ = _log_norm_from_density(mean_lam, dens, lam_percentile)
    cmap_amp = plt.colormaps["rainbow"]

    fig = plt.figure(figsize=(8.8, 4.15), facecolor="white")
    axes = [
        fig.add_axes((0.02, 0.06, 0.46, 0.88)),
        fig.add_axes((0.52, 0.06, 0.46, 0.88)),
    ]
    panels = (
        (mean_amp, cmap_amp, norm_amp, ticks_amp, _amp_tick_labels(ticks_amp),
         r"$|C_k|$", "Amplitude"),
        (mean_lam, cmap_lam, norm_lam, ticks_lam, [_km_tick_label(t) for t in ticks_lam],
         r"$\lambda$ (km)", "Wavelength"),
    )
    for ax, (mean_hue, cmap, norm, ticks, labs, hue_name, title), letter in zip(
        axes, panels, "ab",
    ):
        ax.set_facecolor("white")
        rgb = np.ones(xx.shape + (3,), dtype=np.float64)
        if dens_peak > 0.0:
            amp01 = np.clip(np.ma.filled(norm(np.maximum(mean_hue, 0.0)), 0.0), 0.0, 1.0)
            val = np.clip(dens / dens_peak, 0.0, 1.0) ** float(value_gamma)
            rgb[inside] = figstyle.bright_rgb(amp01, val, cmap)
        im = ax.imshow(
            rgb, origin="lower", extent=(-1.0, 1.0, -1.0, 1.0),
            interpolation="bilinear", zorder=1, aspect="equal",
        )
        im.set_clip_path(Circle((0.0, 0.0), 1.0, transform=ax.transData))
        _draw_schmidt_net(ax)
        ax.set_title(title, color="0.15", pad=6)
        figstyle.panel_label(ax, letter)
        figstyle.arc_colorbar(
            ax, cmap, norm, ticks, labs,
            val_gamma=value_gamma, amp_name=hue_name, density_name=None,
        )
    hinge_vec = hinge if hinge is not None else _hinge_from_event(event, velocities)
    if hinge_vec is not None:
        hx, hy = _schmidt_xy(hinge_vec)[0]
        font = figstyle._fp(size=9)
        for ax in axes:
            ax.scatter(
                [hx], [hy], s=42, marker="o",
                facecolors="white", edgecolors="0.1", linewidths=1.0, zorder=8,
            )
            ax.annotate(
                "hinge", (hx, hy), xytext=(7, 7), textcoords="offset points",
                color="0.1", fontproperties=font, ha="left", va="bottom",
                zorder=9, clip_on=False,
            )
    return fig, axes


def plot_shear_tilt(event, *, hinge=None, n_top: int = 5000, within_deg: float = 10.0):
    """How far each shear's wavevector tilts out of the plane normal to the hinge, against its wavelength.

    The tilt is ``arcsin |Ω̂_k · ĥ|``: 0° for a shear that is constant along the
    hinge (a cylindrical mode), 90° for one that varies only along it. Points are
    the strongest ``n_top`` shears, coloured by ``|C_k|``; the side panel is the
    amplitude-weighted distribution of the tilt. Returns ``(fig, axes, share)``,
    with ``share`` the fraction of the summed amplitude within ``within_deg`` of
    the plane. ``hinge`` defaults to the Euler slot's.
    """
    velocities = event_velocities(event)
    if not velocities:
        raise TypeError("event field exposes no velocity")
    hinge_vec = hinge if hinge is not None else _hinge_from_event(event, velocities)
    if hinge_vec is None:
        raise ValueError("no hinge: pass `hinge` for a velocity without a fixed slot")
    h = np.asarray(hinge_vec, dtype=float)
    h = h / np.linalg.norm(h)
    coeffs, omegas = [], []
    with torch.no_grad():
        for velocity in velocities:
            coefficient, omega, _ = velocity.shear_decomposition()
            coeffs.append(coefficient)
            omegas.append(omega)
        coefficient, omega = torch.cat(coeffs, dim=0), torch.cat(omegas, dim=0)
        amp_t = coefficient.norm(dim=-1)
        idx = torch.topk(amp_t, min(int(n_top), int(amp_t.numel()))).indices
        om = omega[idx].detach().cpu().numpy()
        amp = amp_t[idx].detach().cpu().numpy()
    kmag = np.linalg.norm(om, axis=1)
    lam = 2.0 * np.pi / np.maximum(kmag, 1e-30)
    tilt = np.degrees(np.arcsin(np.clip(np.abs(om @ h) / np.maximum(kmag, 1e-30), 0.0, 1.0)))
    share = float(amp[tilt <= within_deg].sum() / max(amp.sum(), 1e-30))

    fig = plt.figure(figsize=(7.2, 3.6), facecolor="white")
    ax = fig.add_axes((0.10, 0.15, 0.62, 0.75))
    side = fig.add_axes((0.74, 0.15, 0.14, 0.75), sharey=ax)
    order = np.argsort(amp)
    pos = amp[amp > 0]
    norm = LogNorm(vmin=float(pos.min()) if pos.size else 1e-6, vmax=float(amp.max()) if amp.size else 1.0)
    sc = ax.scatter(lam[order], tilt[order], c=amp[order], s=6, cmap="rainbow", norm=norm, linewidths=0)
    ax.set_xscale("log")
    ax.set_ylim(0.0, 90.0)
    ax.axhline(within_deg, color="0.4", lw=0.6, ls=(0, (3, 2)))
    ax.set_xlabel(r"wavelength $\lambda$ (m)")
    ax.set_ylabel("tilt out of the hinge-normal plane (°)")
    ax.set_title(f"{100 * share:.0f}% of the amplitude within {within_deg:g}° of the plane", color="0.15")
    bins = np.linspace(0.0, 90.0, 37)
    side.hist(tilt, bins=bins, weights=amp / max(amp.sum(), 1e-30), orientation="horizontal", color="0.45")
    side.set_xlabel("amplitude share")
    side.tick_params(labelleft=False)
    cax = fig.add_axes((0.90, 0.15, 0.02, 0.75))
    fig.colorbar(sc, cax=cax, label=r"$|C_k|$")
    return fig, (ax, side), share


