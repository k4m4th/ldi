"""Synthetic 2-D recumbent fold pairs with a known answer (example 01).

Three ways to build the same kind of fold, x east, z up, metres. Each contact
gets two bends, an antiform closing east and a synform closing west below it,
joined by an overturned limb. Both give the true stratigraphic coordinate in
closed form, so the true units are exact everywhere.

``ShearFold``: flat layers folded in two area-preserving, invertible steps,
an upright Gaussian anticline (z -> z + b(x)) then a horizontal shear zone
(x -> x + g(z)). Where g'(z) |b'(x)| > 1 the anticline's east flank is carried
past vertical. The fold is disharmonic and the overturned limb thinned; the
coordinate is the restored height ``phi = z - b(x - g(z))``.

``ParallelFold``: no deformation history at all. A reference curve of straight
limbs and circular hinges is laid out directly, and every contact is a parallel
curve of it, so beds keep their orthogonal thickness round the hinges (class
1B, as in buckle folds of competent layers). The coordinate is the signed
distance from the reference curve. Far from the layers that distance has
creases (where two parts of the curve are equally near), but within the band
the contacts occupy it is smooth.

``BentFold``: a ``ParallelFold`` refolded by bending the whole package round a
gentle upright antiform, as a stiff package buckles (tangential longitudinal
strain: layers stretched on the outside of the bend, shortened on the inside,
no shear along it). The recumbent fold's flat axial surface is arched, a
type-3 (hook) interference pattern. Its truth is the first fold's truth at the
point the bending sends there, found by projection onto the bending curve.

numpy only, so the fitting notebooks can evaluate the truth on their own grid;
``from_settings`` rebuilds either from ``01a``'s ``settings.json``.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass
class ShearFold:
    levels: tuple = (250.0, 400.0)  # m, restored height of the (older, younger) contact
    bump_x: float = 100.0           # m, centre of the upright anticline
    bump_width: float = 130.0       # m, its Gaussian width (sigma)
    bump_height: float = 280.0      # m, its amplitude
    shear_z: float = 500.0          # m, centre of the shear zone
    shear_width: float = 150.0      # m, its half-thickness (tanh scale)
    shear_offset: float = 800.0     # m, eastward displacement of everything above it

    def asdict(self) -> dict:
        d = asdict(self)
        d["levels"] = list(d["levels"])
        return d

    # the two steps and their derivatives
    def bump(self, x):
        return self.bump_height * np.exp(-0.5 * ((np.asarray(x, dtype=float) - self.bump_x) / self.bump_width) ** 2)

    def dbump(self, x):
        return -(np.asarray(x, dtype=float) - self.bump_x) / self.bump_width ** 2 * self.bump(x)

    def shift(self, z):
        return 0.5 * self.shear_offset * (1.0 + np.tanh((np.asarray(z, dtype=float) - self.shear_z) / self.shear_width))

    def dshift(self, z):
        u = (np.asarray(z, dtype=float) - self.shear_z) / self.shear_width
        return 0.5 * self.shear_offset / self.shear_width / np.cosh(u) ** 2

    def deform(self, x0, z0):
        """Forward map: where a point of the flat layer cake ends up."""
        z = np.asarray(z0, dtype=float) + self.bump(x0)
        return np.asarray(x0, dtype=float) + self.shift(z), z

    def restored_height(self, x, z):
        """Stratigraphic coordinate: the height the point had before folding."""
        z = np.asarray(z, dtype=float)
        return z - self.bump(np.asarray(x, dtype=float) - self.shift(z))

    def units(self, x, z):
        """True unit: 1 below the older contact (oldest), 2 between, 3 above the younger (youngest)."""
        phi = self.restored_height(x, z)
        lo, hi = sorted(self.levels)
        return np.where(phi < lo, 1, np.where(phi < hi, 2, 3)).astype(np.int8)

    def younging(self, x, z):
        """Unit bed normal pointing to younger beds (up the restored height), shape (..., 2)."""
        z = np.asarray(z, dtype=float)
        db = self.dbump(np.asarray(x, dtype=float) - self.shift(z))
        g = np.stack([-db, 1.0 + db * self.dshift(z)], axis=-1)
        return g / np.linalg.norm(g, axis=-1, keepdims=True)

    def contact(self, level, x_range=(-1000.0, 2000.0), step=0.5):
        """Contact at restored height ``level`` as an ordered polyline (west to east in the
        flat state), its arc length, and a limb label per point: "west" before the first
        bend, "middle" between the bends (overturned here), "east" after. Without an overturn every point
        is "west". Also returns the arc length of each bend (where the limb turns back)."""
        x0 = np.arange(x_range[0], x_range[1], step)
        x, z = self.deform(x0, np.full_like(x0, float(level)))
        pts = np.column_stack([x, z])
        s = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]
        reversed_ = np.gradient(x, x0) < 0.0
        bends = np.flatnonzero(np.diff(reversed_.astype(int)) != 0)
        limb = np.full(len(x0), "west", dtype=object)
        if len(bends) >= 2:
            limb[bends[0] + 1:bends[1] + 1] = "middle"
            limb[bends[1] + 1:] = "east"
        return pts, s, limb, s[bends + 1]


@dataclass
class ParallelFold:
    """Recumbent fold pair built geometrically, with no deformation history.

    A reference curve is laid out like a turtle path (straight limbs joined by
    circular hinges) and every contact is a parallel curve of it: the true
    stratigraphic coordinate is the signed distance from the reference curve,
    positive to its left (up, on the west limb). Beds keep their orthogonal
    thickness all the way round each hinge (Ramsay class 1B, as in buckle folds
    of competent layers), unlike the sheared fold, which thins the overturned
    limb. Contacts must stay inside each hinge's radius on its concave side.

    ``path`` is a sequence of ``("line", length)`` and ``("turn", degrees, radius)``
    steps, left turns positive; the two turns of more than 90° are the hinges.

    ``thickness`` (optional, one factor per ``path`` step) lets the layers thicken
    and thin along the fold: every distance from the reference curve is scaled by a
    factor that runs smoothly (a smoothstep in arc length) between the steps' midpoints,
    e.g. thickened hinges and an attenuated middle limb. The stratigraphic coordinate
    is then the signed distance divided by that factor, so contacts stay its level
    sets. Contacts, scaled, must still fit inside each hinge's radius.
    """

    levels: tuple = (-60.0, 60.0)    # m, signed distance of the (older, younger) contact
    start: tuple = (-1500.0, 620.0)  # m, start of the reference curve (far west)
    path: tuple = (
        ("line", 1600.0),
        ("turn", 20.0, 400.0),        # west limb ramps up
        ("line", 300.0),
        ("turn", -175.0, 110.0),      # antiform, closing east
        ("line", 280.0),              # overturned limb, dipping 25° to the west
        ("turn", 155.0, 110.0),       # synform, closing west
        ("line", 2500.0),             # east limb
    )
    thickness: tuple = None           # optional factor per path step; None keeps beds parallel

    def asdict(self) -> dict:
        d = asdict(self)
        d["levels"], d["start"] = list(d["levels"]), list(d["start"])
        d["path"] = [list(step) for step in d["path"]]
        d["thickness"] = None if self.thickness is None else [float(t) for t in self.thickness]
        return d

    def _step_lengths(self):
        return np.array([float(st[1]) if st[0] == "line" else np.radians(abs(float(st[1]))) * float(st[2])
                         for st in self.path])

    def factor(self, s):
        """Thickness factor at arc length ``s`` along the reference curve (1 without ``thickness``)."""
        if self.thickness is None:
            return np.ones_like(np.asarray(s, dtype=float))
        if len(self.thickness) != len(self.path):
            raise ValueError("thickness needs one factor per path step")
        L = self._step_lengths()
        mid = np.cumsum(L) - 0.5 * L
        w = np.asarray(self.thickness, dtype=float)
        s = np.clip(np.asarray(s, dtype=float), mid[0], mid[-1])
        # Smoothstep between neighbouring midpoints: no kinks where the factor changes.
        i = np.clip(np.searchsorted(mid, s, side="right") - 1, 0, len(mid) - 2)
        t = (s - mid[i]) / (mid[i + 1] - mid[i])
        return w[i] + (w[i + 1] - w[i]) * t * t * (3.0 - 2.0 * t)

    def _elements(self):
        """The path as lines and arcs, with their end points and headings (rad)."""
        p = np.asarray(self.start, dtype=float)
        h = 0.0
        out = []
        for step in self.path:
            if step[0] == "line":
                u = np.array([np.cos(h), np.sin(h)])
                out.append(dict(kind="line", p0=p.copy(), u=u, n=np.array([-u[1], u[0]]), length=float(step[1])))
                p = p + float(step[1]) * u
            else:
                turn, R = np.radians(float(step[1])), float(step[2])
                s = np.sign(turn)
                n = np.array([-np.sin(h), np.cos(h)])
                c = p + s * R * n
                theta0 = np.arctan2(*(p - c)[::-1])
                out.append(dict(kind="arc", c=c, R=R, s=s, theta0=theta0, sweep=abs(turn),
                                hinge=abs(turn) > np.pi / 2))
                h = h + turn
                p = c - s * R * np.array([-np.sin(h), np.cos(h)])
        return out

    def _signed_distance(self, x, z):
        d, g, _ = self.project(x, z)
        return d, g

    def project(self, x, z):
        """Signed distance from the reference curve, its gradient, and the arc length along
        the curve of the nearest point (0 at ``start``)."""
        x, z = np.broadcast_arrays(np.asarray(x, dtype=float), np.asarray(z, dtype=float))
        P = np.stack([x.ravel(), z.ravel()], axis=1)
        best_d = np.full(len(P), np.inf)
        best_g = np.zeros((len(P), 2))
        best_s = np.zeros(len(P))
        s0 = 0.0
        for e in self._elements():
            if e["kind"] == "line":
                t = np.clip((P - e["p0"]) @ e["u"], 0.0, e["length"])
                along = s0 + t
                s0 += e["length"]
                q = e["p0"] + t[:, None] * e["u"]
                r = P - q
                side = np.sign(r @ e["n"])
                side[side == 0] = 1.0
                dist = np.linalg.norm(r, axis=1)
                g = np.where(dist[:, None] > 1e-9, r / dist.clip(1e-9)[:, None] * side[:, None], e["n"])
                d = side * dist
            else:
                r = P - e["c"]
                rho = np.linalg.norm(r, axis=1).clip(1e-9)
                theta = np.arctan2(r[:, 1], r[:, 0])
                a = (e["s"] * (theta - e["theta0"])) % (2 * np.pi)
                on = a <= e["sweep"]
                # n_left points to the centre on a left turn and away from it on a right turn
                d = np.where(on, e["s"] * (e["R"] - rho), np.inf)
                g = -e["s"] * r / rho[:, None]
                along = s0 + e["R"] * np.minimum(a, e["sweep"])
                s0 += e["R"] * e["sweep"]
            better = np.abs(d) < np.abs(best_d)
            best_d[better] = d[better]
            best_g[better] = g[better]
            best_s[better] = along[better]
        return best_d.reshape(x.shape), best_g.reshape(x.shape + (2,)), best_s.reshape(x.shape)

    def restored_height(self, x, z):
        """Stratigraphic coordinate: signed distance from the reference curve, over the
        local thickness factor."""
        d, _, s = self.project(x, z)
        return d / self.factor(s)

    def units(self, x, z):
        """True unit: 1 below the older contact (oldest), 2 between, 3 above the younger (youngest)."""
        phi = self.restored_height(x, z)
        lo, hi = sorted(self.levels)
        return np.where(phi < lo, 1, np.where(phi < hi, 2, 3)).astype(np.int8)

    def younging(self, x, z, h=0.05):
        """Unit bed normal pointing to younger beds, shape (..., 2)."""
        if self.thickness is None:
            return self._signed_distance(x, z)[1]
        x, z = np.asarray(x, dtype=float), np.asarray(z, dtype=float)
        g = np.stack([self.restored_height(x + h, z) - self.restored_height(x - h, z),
                      self.restored_height(x, z + h) - self.restored_height(x, z - h)], axis=-1)
        return g / np.linalg.norm(g, axis=-1, keepdims=True)

    def reference(self, ds=0.5):
        """Reference curve sampled every ``ds`` m: points, left normals, element index per point."""
        pts, nrm, idx = [], [], []
        for k, e in enumerate(self._elements()):
            if e["kind"] == "line":
                t = np.arange(0.0, e["length"], ds)
                pts.append(e["p0"] + t[:, None] * e["u"])
                nrm.append(np.broadcast_to(e["n"], (len(t), 2)))
            else:
                ang = e["theta0"] + e["s"] * np.arange(0.0, e["sweep"], ds / e["R"])
                rad = np.stack([np.cos(ang), np.sin(ang)], axis=1)
                pts.append(e["c"] + e["R"] * rad)
                nrm.append(-e["s"] * rad)
            idx.append(np.full(len(pts[-1]), k))
        return np.vstack(pts), np.vstack(nrm), np.concatenate(idx)

    def contact(self, level, ds=0.5):
        """Contact at signed distance ``level``: polyline, arc length, limb label per point
        ("west" before the first hinge, "middle" between the hinges, "east" after), and the
        arc length of each hinge's midpoint."""
        ref, nrm, idx = self.reference(ds)
        s_ref = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(ref, axis=0), axis=1))]
        offset = float(level) * self.factor(s_ref)
        for k, e in enumerate(self._elements()):
            if e["kind"] == "arc" and (e["s"] * offset[idx == k]).max(initial=-np.inf) >= e["R"]:
                raise ValueError(f"contact at {level} m (x thickness) does not fit inside a hinge of radius {e['R']} m")
        pts = ref + offset[:, None] * nrm
        s = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]
        hinges = [k for k, e in enumerate(self._elements()) if e["kind"] == "arc" and e["hinge"]]
        s_bends = np.array([s[idx == k].mean() for k in hinges])
        limb = np.full(len(pts), "west", dtype=object)
        if len(hinges) >= 2:
            limb[s >= s_bends[0]] = "middle"
            limb[s >= s_bends[1]] = "east"
        return pts, s, limb, s_bends


@dataclass
class SmoothFold:
    """Recumbent fold pair along a smoothly curving reference line, for a natural look.

    Like ``ParallelFold`` (contacts are offsets of a reference curve, the
    stratigraphic coordinate is the signed distance from it over the local
    thickness factor), but the curve has no straight segments or circular arcs.
    Its heading turns smoothly through each hinge (a logistic step of width
    ``width``, so curvature peaks at the hinge and fades along the limbs) and
    wanders gently along the limbs (``wander``: a few long sinusoids of random
    phase), and the thickness drifts smoothly too. Hinges are rounded and
    slightly asymmetric, limbs gently curved, beds neither perfectly parallel
    nor perfectly constant.

    ``hinges`` is a sequence of ``(arc length, turn in degrees, width)``, left turns
    positive; the turns of more than 90° are the hinges used for limb labels.
    ``thickness`` is a sequence of ``(arc length, factor)`` control points (smoothstep
    between them), ``thickness_noise`` the size of a smooth random drift on top.
    Contacts, scaled, must stay inside the local radius of curvature (checked).
    """

    levels: tuple = (-100.0, 30.0)
    start: tuple = (-1500.0, 560.0)
    length: float = 5200.0
    hinges: tuple = (
        (1680.0, 20.0, 120.0),     # west limb ramps up
        (2260.0, -175.0, 175.0),   # antiform, closing east
        (2840.0, 155.0, 160.0),    # synform, closing west
    )
    wander: float = 4.0            # deg, size of the heading's gentle wandering along the limbs
    wander_wavelength: float = 700.0  # m, its typical wavelength
    thickness: tuple = ((1500.0, 1.0), (2260.0, 1.15), (2550.0, 0.6), (2840.0, 1.15), (3300.0, 1.0))
    thickness_noise: float = 0.08  # relative size of a smooth random drift in thickness
    seed: int = 3
    ds: float = 1.0                # m, sampling of the reference curve

    def asdict(self) -> dict:
        d = asdict(self)
        d["levels"], d["start"] = list(d["levels"]), list(d["start"])
        d["hinges"] = [list(h) for h in d["hinges"]]
        d["thickness"] = None if self.thickness is None else [list(t) for t in self.thickness]
        return d

    def _noise(self, s, amplitude, wavelength, salt):
        # A few long sinusoids of random phase and wavelength: smooth, zero-mean wandering.
        rng = np.random.default_rng(self.seed + salt)
        out = np.zeros_like(s)
        for _ in range(4):
            lam = wavelength * rng.uniform(0.6, 1.6)
            out += np.sin(2 * np.pi * s / lam + rng.uniform(0, 2 * np.pi))
        return amplitude * out / 2.0

    def _curve(self):
        if getattr(self, "_cache", None) is not None:
            return self._cache
        s = np.arange(0.0, self.length, self.ds)
        h = self._noise(s, np.radians(self.wander), self.wander_wavelength, 0)
        for c, turn, w in self.hinges:
            h = h + np.radians(turn) * 0.5 * (1.0 + np.tanh((s - c) / w))
        dh = np.gradient(h, s)
        u = np.stack([np.cos(h), np.sin(h)], axis=1)
        pts = np.asarray(self.start, dtype=float) + np.vstack([[0.0, 0.0], np.cumsum(0.5 * (u[1:] + u[:-1]) * self.ds, axis=0)])
        nrm = np.stack([-u[:, 1], u[:, 0]], axis=1)
        from scipy.spatial import cKDTree
        object.__setattr__(self, "_cache", (s, pts, u, nrm, dh, cKDTree(pts)))
        return self._cache

    def factor(self, s):
        s = np.asarray(s, dtype=float)
        w = np.ones_like(s)
        if self.thickness is not None:
            cs = np.array([t[0] for t in self.thickness])
            ws = np.array([t[1] for t in self.thickness])
            sc = np.clip(s, cs[0], cs[-1])
            i = np.clip(np.searchsorted(cs, sc, side="right") - 1, 0, len(cs) - 2)
            t = (sc - cs[i]) / (cs[i + 1] - cs[i])
            w = ws[i] + (ws[i + 1] - ws[i]) * t * t * (3.0 - 2.0 * t)
        if self.thickness_noise:
            w = w * (1.0 + self._noise(s, self.thickness_noise, 400.0, 1))
        return w

    def project(self, x, z):
        """Signed distance from the reference curve, its normal, and the arc length of the nearest point."""
        s_ref, pts, u, nrm, _, tree = self._curve()
        x, z = np.broadcast_arrays(np.asarray(x, dtype=float), np.asarray(z, dtype=float))
        P = np.stack([x.ravel(), z.ravel()], axis=1)
        _, i = tree.query(P)
        r = P - pts[i]
        d = (r * nrm[i]).sum(1)
        s = s_ref[i] + (r * u[i]).sum(1)
        return d.reshape(x.shape), nrm[i].reshape(x.shape + (2,)), s.reshape(x.shape)

    def restored_height(self, x, z):
        d, _, s = self.project(x, z)
        return d / self.factor(s)

    def units(self, x, z):
        phi = self.restored_height(x, z)
        lo, hi = sorted(self.levels)
        return np.where(phi < lo, 1, np.where(phi < hi, 2, 3)).astype(np.int8)

    def younging(self, x, z, h=0.05):
        x, z = np.asarray(x, dtype=float), np.asarray(z, dtype=float)
        g = np.stack([self.restored_height(x + h, z) - self.restored_height(x - h, z),
                      self.restored_height(x, z + h) - self.restored_height(x, z - h)], axis=-1)
        return g / np.linalg.norm(g, axis=-1, keepdims=True)

    def contact(self, level, ds=None):
        s_ref, pts, u, nrm, dh, _ = self._curve()
        offset = float(level) * self.factor(s_ref)
        # A contact folds back on itself where it reaches past the centre of curvature.
        if (offset * dh).max() >= 1.0:
            raise ValueError(f"contact at {level} m (x thickness) reaches past a hinge's centre of curvature; "
                             "widen that hinge or thin the layer there")
        out = pts + offset[:, None] * nrm
        s = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(out, axis=0), axis=1))]
        hinge_s = [c for c, turn, _ in self.hinges if abs(turn) > 90.0]
        idx = [int(np.argmin(np.abs(s_ref - c))) for c in hinge_s]
        s_bends = s[idx]
        limb = np.full(len(out), "west", dtype=object)
        if len(s_bends) >= 2:
            limb[s >= s_bends[0]] = "middle"
            limb[s >= s_bends[1]] = "east"
        return out, s, limb, s_bends


@dataclass
class BentFold:
    """A fold pair refolded by bending the whole package round a gentle curve.

    The first fold is laid out on a straight, horizontal datum (``z = bend.start
    z``). Bending takes each point at distance ``x - x0`` along that datum and
    height ``n`` above it to the point at the same distance along the bending
    curve and the same distance ``n`` from it, along the curve's normal. That is
    how a stiff package buckles: lines across the package stay straight and
    normal to it, and layers are stretched on the outer side of each bend and
    shortened on the inner side (tangential longitudinal strain), with no shear
    along the bending curve. Bending the recumbent fold's flat-lying axial
    surface into an upright fold gives a type-3 (hook) interference pattern.

    ``bend`` is a ``ParallelFold`` used only for its curve (``start``, ``path``):
    keep its radii larger than the distance of anything of interest from it on
    the concave side, or the bending is not one-to-one there. The truth at a
    point is the first fold's truth at the point bending sends there, found by
    projecting onto the bending curve.
    """

    base: ParallelFold = None
    bend: ParallelFold = None

    def __post_init__(self):
        if self.base is None:
            self.base = ParallelFold()
        if self.bend is None:
            self.bend = ParallelFold(
                levels=(0.0, 0.0), start=(-3000.0, 330.0),
                path=(("line", 2600.0), ("turn", 25.0, 1100.0), ("line", 50.0),
                      ("turn", -50.0, 1100.0), ("line", 50.0), ("turn", 25.0, 1100.0), ("line", 4000.0)),
            )

    @property
    def levels(self):
        return self.base.levels

    def asdict(self) -> dict:
        return dict(base=self.base.asdict(), bend=self.bend.asdict())

    def _to_base(self, x, z):
        n, _, s = self.bend.project(x, z)
        return self.bend.start[0] + s, self.bend.start[1] + n

    def deform(self, x, z, ds=0.5):
        """Where bending sends a point of the first fold."""
        ref, nrm, _ = self.bend.reference(ds)
        s_ref = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(ref, axis=0), axis=1))]
        s = np.asarray(x, dtype=float) - self.bend.start[0]
        n = np.asarray(z, dtype=float) - self.bend.start[1]
        c = np.stack([np.interp(s, s_ref, ref[:, i]) for i in (0, 1)], axis=-1)
        nn = np.stack([np.interp(s, s_ref, nrm[:, i]) for i in (0, 1)], axis=-1)
        nn /= np.linalg.norm(nn, axis=-1, keepdims=True)
        out = c + n[..., None] * nn
        return out[..., 0], out[..., 1]

    def check(self, x, z):
        """Largest distance from the bending curve, as a share of the smallest bend radius;
        bending is one-to-one only below 1 (on the concave side)."""
        n, _, _ = self.bend.project(x, z)
        r = min(float(step[2]) for step in self.bend.path if step[0] == "turn")
        return float(np.abs(n).max() / r)

    def restored_height(self, x, z):
        return self.base.restored_height(*self._to_base(x, z))

    def units(self, x, z):
        return self.base.units(*self._to_base(x, z))

    def younging(self, x, z, h=0.05):
        # Central differences of the truth: exact enough, and free of curvature bookkeeping.
        x, z = np.asarray(x, dtype=float), np.asarray(z, dtype=float)
        gx = self.restored_height(x + h, z) - self.restored_height(x - h, z)
        gz = self.restored_height(x, z + h) - self.restored_height(x, z - h)
        g = np.stack([gx, gz], axis=-1)
        return g / np.linalg.norm(g, axis=-1, keepdims=True)

    def contact(self, level, ds=0.5):
        pts, s, limb, s_bends = self.base.contact(level, ds)
        bend_idx = np.searchsorted(s, s_bends)
        x, z = self.deform(pts[:, 0], pts[:, 1])
        pts = np.column_stack([x, z])
        s = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]
        return pts, s, limb, s[bend_idx]


MODELS = {"ShearFold": ShearFold, "ParallelFold": ParallelFold, "SmoothFold": SmoothFold, "BentFold": BentFold}


def _smooth_from_dict(kw):
    kw = dict(kw)
    kw["levels"], kw["start"] = tuple(kw["levels"]), tuple(kw["start"])
    kw["hinges"] = tuple(tuple(h) for h in kw["hinges"])
    if kw.get("thickness") is not None:
        kw["thickness"] = tuple(tuple(t) for t in kw["thickness"])
    return SmoothFold(**kw)


def _parallel_from_dict(kw):
    kw = dict(kw)
    kw["path"] = tuple(tuple(step) for step in kw["path"])
    kw["start"], kw["levels"] = tuple(kw["start"]), tuple(kw["levels"])
    if kw.get("thickness") is not None:
        kw["thickness"] = tuple(kw["thickness"])
    return ParallelFold(**kw)


@dataclass
class Inverted:
    """Any of the folds with its stratigraphy turned upside down.

    Same contacts, opposite way-up: the stratigraphic coordinate changes sign,
    so every limb that was upright is overturned and vice versa. For the fold
    pairs here the middle limb becomes the only upright one, as in parasitic
    folds on the inverted limb of a larger recumbent fold (a nappe).
    """

    base: object = None

    @property
    def levels(self):
        return tuple(sorted(-float(l) for l in self.base.levels))

    def asdict(self) -> dict:
        return self.base.asdict()

    def restored_height(self, x, z):
        return -self.base.restored_height(x, z)

    def units(self, x, z):
        phi = self.restored_height(x, z)
        lo, hi = self.levels
        return np.where(phi < lo, 1, np.where(phi < hi, 2, 3)).astype(np.int8)

    def younging(self, x, z):
        return -self.base.younging(x, z)

    def contact(self, level, ds=0.5):
        return self.base.contact(-level, ds)


def from_settings(settings: dict):
    """Rebuild the truth from ``01a``'s settings.json (``model``, ``fold`` and ``inverted``)."""
    model, kw = settings.get("model", "ShearFold"), dict(settings["fold"])
    if model == "ParallelFold":
        fold = _parallel_from_dict(kw)
    elif model == "SmoothFold":
        fold = _smooth_from_dict(kw)
    elif model == "BentFold":
        base = kw["base"]
        base = _smooth_from_dict(base) if "hinges" in base else _parallel_from_dict(base)
        fold = BentFold(base=base, bend=_parallel_from_dict(kw["bend"]))
    else:
        fold = MODELS[model](**kw)
    return Inverted(fold) if settings.get("inverted", False) else fold
