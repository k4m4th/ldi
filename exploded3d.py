"""Exploded slab views of 3-D contact surfaces, rendered with headless Blender.

The caller meshes each contact surface (e.g. with ``GeoEvent.mesh``, octree dual
contouring) and passes a ``mesher(level, bounds) -> (verts, faces)`` for the
small closed level surfaces around interior extrema. `build_scene` cuts the
surfaces into slabs along y, moves the slabs apart in a staircase, traces each
contact where a slab is cut and writes everything Blender needs; `render` runs
`render_exploded.py` in Blender without a window. Every setting comes from the
caller.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
import pyvista as pv

BLENDER = "/Applications/Blender.app/Contents/MacOS/Blender"
RENDER_SCRIPT = Path(__file__).with_name("render_exploded.py")


def polydata(verts, faces):
    faces = np.asarray(faces, dtype=np.int64)
    return pv.PolyData(np.asarray(verts, dtype=float), np.c_[np.full(len(faces), 3), faces].ravel())


def _bare(mesh):
    """The mesh without point or cell data. The clipping filters leave arrays such as ``vtkOriginalCellIds`` whose
    length no longer matches the cells after ``extract_surface``; the mesh is then invalid and a later filter can
    silently drop it (a whole horizon's half vanished from the opened block)."""
    if mesh is not None:
        mesh.clear_data()
    return mesh


def clean(mesh, *, min_area=0.0, smooth_iter=0, pass_band=0.1):
    """Merge duplicate vertices, drop pieces smaller than `min_area` (m²) and optionally smooth.

    Taubin smoothing (no shrinkage); a smaller `pass_band` removes broader wiggles.
    """
    if mesh is None or mesh.n_points == 0:
        return None
    mesh = mesh.clean().triangulate()
    if min_area > 0:
        parts = mesh.connectivity(extraction_mode="all").split_bodies()
        keep = [p.extract_surface(algorithm="dataset_surface") for p in parts
                if p.extract_surface(algorithm="dataset_surface").area >= min_area]
        if not keep:
            return None
        mesh = pv.merge(keep).clean()
    if smooth_iter:
        mesh = mesh.smooth_taubin(n_iter=smooth_iter, pass_band=pass_band)
    return _bare(mesh)


def is_closed(mesh):
    return mesh.n_points > 0 and mesh.extract_feature_edges(
        boundary_edges=True, feature_edges=False, manifold_edges=False, non_manifold_edges=False).n_cells == 0


def closed_shell(mesher, point, kind, value, delta, half, tries=4):
    """The level surface just inside an extremum, if it closes within ±`half` of it.

    `value` is the field at the extremum; the level is `value` - `delta` around a
    maximum and `value` + `delta` around a minimum. `delta` halves on each failed try.
    """
    point = np.asarray(point, dtype=float)
    bounds = np.stack([point - half, point + half])
    for _ in range(tries):
        level = value - delta if kind == "maximum" else value + delta
        v, f = mesher(level, bounds)
        if len(f):
            mesh = polydata(v, f).clean()
            comp = mesh.connectivity(extraction_mode="closest", closest_point=point)
            comp = comp.extract_surface(algorithm="dataset_surface")
            if is_closed(comp):
                return comp
        delta *= 0.5
    return None


def sheets_from_saved(meshes, *, min_area=0.0, smooth_iter=0, pass_band=0.1):
    """The contact surfaces written by ``diagnostics.export_meshes``, cleaned, youngest first."""
    n = len(meshes["isovalues"])
    return [clean(polydata(meshes[f"sheet{h}_v"], meshes[f"sheet{h}_f"]), min_area=min_area,
                  smooth_iter=smooth_iter, pass_band=pass_band) for h in range(n)]


def shell_from_saved(meshes, e):
    """The closed level surface just inside extremum ``e`` from ``diagnostics.export_meshes``.

    Tries the saved levels in order and keeps the first whose component nearest
    the extremum is closed; ``None`` if none closes (the extremum then gets a sphere).
    """
    point = np.asarray(meshes["ext_points"][e], dtype=float)
    for t in range(int(meshes["n_tries"])):
        v, f = meshes[f"shell{e}_{t}_v"], meshes[f"shell{e}_{t}_f"]
        if len(f):
            comp = polydata(v, f).clean().connectivity(extraction_mode="closest", closest_point=point)
            comp = comp.extract_surface(algorithm="dataset_surface")
            if is_closed(comp):
                return comp
    return None


def _clip(mesh, lo, hi):
    """Exact clip of a mesh to the box [lo, hi]."""
    if mesh is None or mesh.n_points == 0:
        return None
    out = mesh.clip_box(bounds=(lo[0], hi[0], lo[1], hi[1], lo[2], hi[2]), invert=False)
    if out.n_points == 0:
        return None
    out = _bare(_bare(out).extract_surface(pass_pointid=False, pass_cellid=False, algorithm="dataset_surface").triangulate())   # clear before: stale arrays make it invalid
    return out if out.n_points else None


def _trace(surf, y, x_range, z_view):
    """Polylines where a surface crosses the plane y, inside the x and z limits."""
    if surf is None:
        return []
    cut = surf.slice(normal=(0.0, 1.0, 0.0), origin=(0.0, y, 0.0))
    if cut.n_points == 0:
        return []
    cut = cut.clip_box(bounds=(x_range[0], x_range[1], y - 1, y + 1, z_view[0], z_view[1]), invert=False)
    if cut.n_points == 0:
        return []
    cut = cut.extract_surface(algorithm=None).strip()
    lines, out, i = cut.lines, [], 0
    while i < len(lines):
        n = lines[i]
        idx = lines[i + 1:i + 1 + n]
        if n >= 2:
            out.append(np.asarray(cut.points[idx], dtype=float))
        i += n + 1
    return out


def _mesh_arrays(mesh):
    faces = mesh.faces.reshape(-1, 4)[:, 1:]
    return np.asarray(mesh.points, dtype=np.float32), np.asarray(faces, dtype=np.int32)


def _sphere_cut(mesh, centre, radius):
    """The part of a mesh outside the sphere, and the polylines where the sphere cuts it."""
    if mesh is None or mesh.n_points == 0:
        return None, []
    mesh = mesh.copy()
    mesh.point_data["d"] = np.linalg.norm(np.asarray(mesh.points) - np.asarray(centre), axis=1) - radius
    kept = mesh.clip_scalar(scalars="d", value=0.0, invert=False)
    kept = _bare(_bare(kept).extract_surface(pass_pointid=False, pass_cellid=False, algorithm="dataset_surface").triangulate()) if kept.n_points else None
    rim = mesh.contour([0.0], scalars="d")
    rims = []
    if rim.n_points:
        rim = rim.strip()
        lines, i = rim.lines, 0
        while i < len(lines):
            n = lines[i]
            if n >= 2:
                rims.append(np.asarray(rim.points[lines[i + 1:i + 1 + n]], dtype=float))
            i += n + 1
    return (kept if kept is not None and kept.n_points else None), rims


def _scalar_cut(mesh, fn):
    """The part of a mesh where ``fn(points) >= 0``, and the polylines where ``fn`` is zero."""
    if mesh is None or mesh.n_points == 0:
        return None, []
    mesh = mesh.copy()
    s = np.asarray(fn(np.asarray(mesh.points)), dtype=float)
    mesh.point_data["s"] = np.where(np.isfinite(s), s, -1e6)
    kept = mesh.clip_scalar(scalars="s", value=0.0, invert=False)
    kept = _bare(_bare(kept).extract_surface(pass_pointid=False, pass_cellid=False, algorithm="dataset_surface").triangulate()) if kept.n_points else None
    rims = []
    rim = mesh.contour([0.0], scalars="s")
    if rim.n_points:
        rim = rim.strip()
        lines, i = rim.lines, 0
        while i < len(lines):
            n = lines[i]
            if n >= 2:
                rims.append(np.asarray(rim.points[lines[i + 1:i + 1 + n]], dtype=float))
            i += n + 1
    return (kept if kept is not None and kept.n_points else None), rims


def _looks(occlusion, matte, backface_darken, scale):
    """The scene.json entries for the optional looks that render_exploded.py applies."""
    return {"occlusion": occlusion, "matte": matte, "backface_darken": backface_darken, "scale": scale}


def _smooth_polyline(line, length, step):
    """A polyline resampled every `step` m and smoothed along itself over about `length` m.

    Contouring a triangle mesh gives lines that zig-zag across the triangle edges with uneven
    spacing; a tube drawn through every vertex shows that. Even resampling and a few passes of
    a moving average take it out. A closed loop stays closed; an open line keeps its ends.
    """
    line = np.asarray(line, dtype=float)
    keep = np.r_[True, np.linalg.norm(np.diff(line, axis=0), axis=1) > 1e-9]
    line = line[keep]
    if length <= 0 or len(line) < 3:
        return line
    closed = np.linalg.norm(line[0] - line[-1]) < step
    if closed:
        line = np.vstack([line, line[:1]])
    s = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(line, axis=0), axis=1))]
    n = max(int(np.ceil(s[-1] / step)), 2)
    t = np.linspace(0.0, s[-1], n + 1)
    pts = np.column_stack([np.interp(t, s, line[:, k]) for k in range(3)])
    half = max(int(round(0.5 * length / step / 2)), 1)        # window half-width; 3 passes ~ `length`
    if closed:
        # A kernel longer than the loop makes np.convolve swap the two arrays.
        half = min(half, (len(pts) - 2) // 2)
        if half < 1:
            return pts
    kernel = np.ones(2 * half + 1) / (2 * half + 1)
    for _ in range(3):
        if closed:
            body = pts[:-1]
            wrap = np.vstack([body[-half:], body, body[:half]])
            body = np.column_stack([np.convolve(wrap[:, k], kernel, mode="valid") for k in range(3)])
            pts = np.vstack([body, body[:1]])
        elif len(pts) > 2 * half + 2:
            inner = np.column_stack([np.convolve(pts[:, k], kernel, mode="valid") for k in range(3)])
            pts = np.vstack([pts[:half], inner, pts[-half:]])        # the ends stay where they are
    return pts


def _smooth_boundary(mesh, length, passes=3):
    """The mesh with its free edges smoothed along themselves over about `length` m.

    A cut through a triangle mesh leaves an edge that zig-zags across the triangles; a slab
    wall (or a rim) built on it shows that. Each boundary loop or chain is smoothed with a
    moving average along its own vertices; vertices where boundaries meet stay put, as do
    the ends of open chains. Only boundary vertices move, by about the zig-zag's size.
    """
    if mesh is None or length <= 0 or mesh.n_points == 0:
        return mesh
    faces = mesh.faces.reshape(-1, 4)[:, 1:]
    edges = np.sort(np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1)
    uniq, count = np.unique(edges, axis=0, return_counts=True)
    bnd = uniq[count == 1]
    if not len(bnd):
        return mesh
    nbrs = {}
    for a, b in bnd:
        nbrs.setdefault(int(a), []).append(int(b))
        nbrs.setdefault(int(b), []).append(int(a))
    pts = np.asarray(mesh.points, dtype=float).copy()
    seen = set()
    for start in list(nbrs):
        if start in seen or len(nbrs[start]) != 2:
            continue
        # walk both ways from `start` until a junction, an end, or back round to `start`
        chain, closed = [start], False
        seen.add(start)
        for direction in (0, 1):
            prev, cur = start, nbrs[start][direction]
            run = []
            while cur not in seen and len(nbrs[cur]) == 2:
                run.append(cur); seen.add(cur)
                nxt = nbrs[cur][0] if nbrs[cur][0] != prev else nbrs[cur][1]
                prev, cur = cur, nxt
            if cur == start:
                closed = True
            elif cur not in run:
                run.append(cur)                              # a junction or end: an anchor
            chain = chain + run if direction == 0 else run[::-1] + chain
            if closed:
                break
        idx = np.asarray(chain)
        if len(idx) < 5:
            continue
        seg = np.linalg.norm(np.diff(pts[idx], axis=0), axis=1).mean()
        half = max(int(round(0.25 * length / max(seg, 1e-9))), 1)
        if closed:
            # A kernel longer than the loop makes np.convolve swap the two arrays.
            half = min(half, (len(idx) - 1) // 2)
        kernel = np.ones(2 * half + 1) / (2 * half + 1)
        p = pts[idx]
        for _ in range(passes):
            if closed:
                wrap = np.vstack([p[-half:], p, p[:half]])
                p = np.column_stack([np.convolve(wrap[:, k], kernel, mode="valid") for k in range(3)])
            elif len(p) > 2 * half + 2:
                inner = np.column_stack([np.convolve(p[:, k], kernel, mode="valid") for k in range(3)])
                p = np.vstack([p[:half], inner, p[-half:]])  # the anchors at the ends stay
        pts[idx] = p
    out = mesh.copy()
    out.points = pts
    return out


def _split_polyline(line, fn, side):
    """The runs of a polyline where ``side * fn >= 0``, cut exactly where ``fn`` crosses zero."""
    line = np.asarray(line, dtype=float)
    s = side * np.asarray(fn(line), dtype=float)
    runs, run = [], []
    for i in range(len(line)):
        if i and (s[i - 1] >= 0) != (s[i] >= 0):        # crossing: add the zero point to both runs
            t = s[i - 1] / (s[i - 1] - s[i])
            x = line[i - 1] + t * (line[i] - line[i - 1])
            run.append(x)
            if s[i] < 0:
                runs.append(np.asarray(run)); run = []
            else:
                run = [x]
        if s[i] >= 0:
            run.append(line[i])
    if run:
        runs.append(np.asarray(run))
    return [r for r in runs if len(r) >= 2]


def plane_traces(surface, normal, origin, lift=0.0, smooth=0.0, step=2.0):
    """Polylines where a surface crosses the plane through ``origin`` with ``normal``, raised ``lift`` m in z (so
    a tube drawn along them does not sink into the sheet) and smoothed over about ``smooth`` m (0 = as cut)."""
    if surface is None or surface.n_points == 0:
        return []
    cut = surface.slice(normal=normal, origin=origin)
    if cut.n_points == 0:
        return []
    cut = cut.strip()
    out, lines, i = [], cut.lines, 0
    while i < len(lines):
        n = lines[i]
        if n >= 2:
            line = np.asarray(cut.points[lines[i + 1:i + 1 + n]], dtype=float)
            line = _smooth_polyline(line, smooth, step) if smooth > 0 else line
            line[:, 2] += lift
            out.append(line)
        i += n + 1
    return out


def _fill_pockets(mesh, fn, min_area):
    """``fn``, but positive on the mesh's points inside ``fn < 0`` pockets smaller than ``min_area`` (m²)."""
    if not min_area:
        return fn
    pockets, _ = _scalar_cut(mesh, lambda p: -np.asarray(fn(p), dtype=float))
    if pockets is None:
        return fn
    parts = pockets.connectivity(extraction_mode="all")
    ids = np.asarray(parts.cell_data["RegionId"])
    area = np.asarray(parts.compute_cell_sizes(length=False, volume=False).cell_data["Area"])
    small = [r for r in np.unique(ids) if area[ids == r].sum() < min_area]
    if not small:
        return fn
    pts = np.unique(np.concatenate([parts.extract_cells(np.flatnonzero(ids == r)).points for r in small]), axis=0)
    tree = cKDTree(pts)

    def filled(p):
        p = np.atleast_2d(np.asarray(p, dtype=float))
        v = np.asarray(fn(p), dtype=float)
        inside = (v < 0) & (tree.query(p)[0] < 1e-3)
        return np.where(inside, -v, v)
    return filled


def build_cutaway(out_dir, *, surfaces, box, keep, colors, lines=(), line_colors=None, extrema=(),
                  extremum_color="#FF1FA8", rim_radius=0.6, rim_smooth=2.0, edge_smooth=2.0, line_radius=1.0, glyph_radius=2.5,
                  min_area=0.0, rims=True, ghost=None, ghost_alpha=0.25, ghost_whiten=0.4,
                  ghost_rims=False, ghost_min_area=0.0, occlusion=None, matte=False, backface_darken=0.0, sheen=0.0, rim_light=0.0, slab=0.0,
                  subdiv=0, world_strength=0.9, sun_energy=2.5, scale=1e-3):
    """Write a cutaway of nested contact surfaces to `out_dir`, each cut by its own function.

    surfaces : one PyVista mesh per contact (model frame, m), youngest first.
    box      : [[x0, y0, z0], [x1, y1, z1]], the domain every surface is cropped to.
    keep     : one callable per surface, ``keep[h](points) -> (n,)``; the part where it is
               >= 0 is kept, and its zero set is drawn as a rim in a darker shade of the colour
               (``rims=False`` leaves the rims out). E.g. "below the ground surface".
               Rims are resampled and smoothed along themselves over about `rim_smooth` m
               (0 = drawn through the contour's vertices as they are). The surfaces' own edges are
               smoothed the same way over about `edge_smooth` m (0 = as cut), so slab walls run smooth.
    ghost    : optional callable (or one per surface), ``ghost(points) -> (n,)``: where it is
               >= 0 the kept surface is drawn solid, elsewhere as a ghost, see-through (opacity
               ``ghost_alpha``) and paler (its colour mixed ``ghost_whiten`` of the way to white).
               E.g. "behind the cliff face", so the face shows as the edge of the solid part.
               ``ghost_rims=True`` also draws that edge as a rim. ``ghost_alpha=0`` leaves the
               ghost side out altogether (a cut, with its rim), rather than drawing it invisible.
               Ghost pockets smaller than ``ghost_min_area`` (m²) are drawn solid (no pinholes).
    occlusion : if given (m), EEVEE screen-space ambient occlusion over that distance, and
               soft sun shadows, so the gaps between nested surfaces darken.
    matte    : rough, low-specular surfaces (no plastic sheen).
    backface_darken : the underside of each surface this much darker (0-1), so overturned
               parts read as such.
    sheen    : sheen weight (0-1): surfaces brighten where they turn away from the view, so
               closely stacked sheets separate.
    rim_light : energy of a second sun from behind the scene (0 = none): it catches the
               edges of the sheets and lifts them off the background.
    slab     : if > 0 (m), each surface is drawn as a slab of that thickness, its walls in the
               darker shade of its colour; the rim tubes can then be left out (``rims=False``).
    subdiv   : Catmull–Clark levels applied by the renderer to the sheets (0 = the mesh as cut).
               The stored mesh is unchanged; only the render is subdivided.
    lines    : (n, 3) polylines (e.g. digitised traces), drawn as shaded tubes of radius
               `line_radius`, in ``line_colors`` (one hex colour per line) or dark grey.
    extrema  : list of (point (3,), enclosure mesh or None) inside the box, drawn in
               `extremum_color`; an extremum without an enclosure gets a small sphere.
    world_strength, sun_energy : brightness of the white surroundings and of the sun.
    `scale` converts metres to Blender units.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    lo, hi = np.asarray(box, dtype=float)
    arrays, objects = {}, []
    materials = {f"contact{h}": c for h, c in enumerate(colors)}
    ghosts = {}                                       # ghost material -> (colour, opacity)
    ghost_fns = (list(ghost) if isinstance(ghost, (list, tuple)) else [ghost] * len(surfaces)) if ghost else None

    def add_mesh(name, mesh, material):
        v, f = _mesh_arrays(mesh)
        arrays[f"{name}_v"] = v * scale
        arrays[f"{name}_f"] = f
        objects.append({"name": name, "type": "mesh", "material": material})

    def add_rims(name, lines_, material):
        for i, rim in enumerate(lines_):
            rim = _smooth_polyline(rim, rim_smooth, step=max(rim_radius, 0.25))
            objects.append({"name": f"{name}_{i}", "type": "tube", "points": (rim * scale).tolist(),
                            "radius": rim_radius * scale, "material": material})

    for h, (surf, fn) in enumerate(zip(surfaces, keep)):
        kept, rim_lines = _scalar_cut(_clip(surf, lo, hi), fn)
        kept = clean(kept, min_area=min_area) if kept is not None else None
        if kept is None:
            continue
        # only the rims along what is kept (a dropped speck leaves no floating edge)
        near = cKDTree(np.asarray(kept.points))
        rim_lines = [r for r in rim_lines if np.median(near.query(r)[0]) < 1.0] if rims else []
        if ghost_fns is None:
            add_mesh(f"cut{h}", _smooth_boundary(kept, edge_smooth), f"contact{h}")
            add_rims(f"rim{h}", rim_lines, f"trace{h}")
            continue
        g = _fill_pockets(kept, ghost_fns[h], ghost_min_area)
        solid, edge = _scalar_cut(kept, g)
        faded, _ = _scalar_cut(kept, lambda p, g=g: -np.asarray(g(p), dtype=float))
        # the air cut can leave a speck detached from the sheet; judge each piece on its own area
        solid = clean(solid, min_area=min_area) if solid is not None else None
        faded = clean(faded, min_area=min_area) if faded is not None else None

        def _along(mesh, lines_):
            if mesh is None or mesh.n_points == 0 or not lines_:
                return []
            near_m = cKDTree(np.asarray(mesh.points))
            return [r for r in lines_ if len(r) >= 2 and np.median(near_m.query(r)[0]) < 1.0]

        rgb = np.array([int(colors[h].lstrip("#")[i:i + 2], 16) for i in (0, 2, 4)]) / 255
        rgb = rgb + ghost_whiten * (1.0 - rgb)
        ghosts[f"ghost{h}"] = ("#" + "".join(f"{int(round(255 * c)):02x}" for c in rgb), float(ghost_alpha))
        hide = ghost_alpha <= 0                       # opacity 0: cut the ghost side away altogether
        if solid is not None:
            add_mesh(f"cut{h}", _smooth_boundary(solid, edge_smooth), f"contact{h}")
        if faded is not None and not hide:
            add_mesh(f"ghost{h}", _smooth_boundary(faded, edge_smooth), f"ghost{h}")
        # each rim split where it crosses to the ghost side: solid there, faded (or gone) beyond
        add_rims(f"rim{h}", _along(solid, [run for r in rim_lines for run in _split_polyline(r, g, +1)]), f"trace{h}")
        if not hide:
            add_rims(f"ghostrim{h}", _along(faded, [run for r in rim_lines for run in _split_polyline(r, g, -1)]), f"ghost{h}")
        if (ghost_rims or hide) and rims and solid is not None:   # a cut edge, rimmed like the others
            add_rims(f"edge{h}", _along(solid, edge), f"trace{h}")
    for li, pts in enumerate(lines):
        pts = np.asarray(pts, dtype=float)
        material = "hole"
        if line_colors is not None:
            material = f"line{li}"
            materials[material] = line_colors[li]
        idx = np.flatnonzero(np.all((pts >= lo) & (pts <= hi), axis=1))
        for k, run in enumerate(np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1) if len(idx) else []):
            if len(run) >= 2:
                objects.append({"name": f"line{li}_{k}", "type": "tube", "points": (pts[run] * scale).tolist(),
                                "radius": line_radius * scale, "material": material})
    for e, (p, shell) in enumerate(extrema):
        p = np.asarray(p, dtype=float)
        if not np.all((p >= lo) & (p <= hi)):
            continue
        piece = _clip(shell, lo, hi) if shell is not None else None
        if piece is not None:
            add_mesh(f"shell{e}", piece, "shell")
        else:
            objects.append({"name": f"ext{e}", "type": "spheres", "points": [(p * scale).tolist()],
                            "radius": glyph_radius * scale, "material": "extremum"})

    np.savez_compressed(out / "meshes.npz", **arrays)
    (out / "scene.json").write_text(json.dumps({"objects": objects, "colors": materials, "ghosts": ghosts,
                                                "occlusion": occlusion, "matte": matte,
                                                "backface_darken": backface_darken, "sheen": sheen,
                                                "rim_light": rim_light, "slab": slab, "subdiv": int(subdiv), "scale": scale,
                                                "extremum_color": extremum_color,
                                                "world_strength": world_strength, "sun_energy": sun_energy},
                                               indent=1))
    return out


def build_opened(out_dir, *, surfaces, box, split_y, hinge_x, angle, colors, section_x=None, section_y=None,
                 ghost_alpha=0.15, glass=None, extrema=(), extremum_color="#FF1FA8", texture=0.0, texture_size=25.0,
                 lines=(), line_color="#FFFFFF", line_radius=3.0, **kw):
    """A block of nested contact surfaces split at ``y = split_y`` and opened like a book.

    The two halves turn about a vertical hinge on the split plane at ``x = hinge_x`` (the spine,
    away from the viewer), so their cut faces turn towards the viewer; the contacts are traced
    where each surface meets its cut face. ``angle`` is one angle (degrees) for both halves or
    ``(angle_a, angle_b)``: half ``a`` is ``y < split_y``, half ``b`` the rest. Turning one half
    a little and the other a lot opens a gap through which the first half's cut face is seen.

    Half ``a`` may also be cut by one vertical section, where its surfaces are traced and beyond
    which they are drawn see-through (opacity ``ghost_alpha``; ``0`` removes them):
    ``section_x``, a section at constant *x*, see-through between it and the viewer; or
    ``section_y``, a section at constant *y* parallel to the cut, see-through between it and the
    cut face, so it is seen through the cut face. ``glass`` marks that section with a pane of
    glass, framed, across half ``a``'s box: a dict with ``alpha`` (opacity), and optionally
    ``color``, ``frame`` (frame colour), ``frame_alpha``, ``frame_radius`` (m) and ``margin`` (share of the surfaces' height it extends
    above and below them); ``None`` = no pane.
    ``lines`` are polylines in the model frame (e.g. seismic lines draped on a surface with :func:`plane_traces`),
    drawn as tubes of ``line_radius`` m in ``line_color`` on the half they lie in; where half ``a`` is see-through
    they are left out.
    ``texture`` (0-1) gives the surfaces a fine rock grain of about ``texture_size`` m (shade and
    bump; 0 = smooth). Other keywords go to :func:`build_cutaway`, which
    builds each half (``rims``, ``occlusion``, ``matte``, ``backface_darken``, ...).
    Returns ``(scene_dir, corners)``: ``corners`` are the eight corners of each half after
    opening (model frame, m), for framing the camera.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    lo, hi = np.asarray(box, dtype=float)
    front = 1.0 if hinge_x <= 0.5 * (lo[0] + hi[0]) else -1.0     # +1: the viewer is on the +x side
    angle_a, angle_b = (angle, angle) if np.ndim(angle) == 0 else angle
    if section_x is not None and section_y is not None:
        raise ValueError("give section_x or section_y, not both")
    ghost_a = None
    if section_x is not None:
        ghost_a = lambda p: front * (section_x - p[:, 0])           # >= 0 (solid) away from the viewer
    elif section_y is not None:
        ghost_a = lambda p: section_y - p[:, 1]                     # >= 0 (solid) beyond the section
    halves = {
        # y < split: turns so its cut face (normal +y) swings towards the viewer
        "a": dict(keep=lambda p: split_y - p[:, 1], sign=-front, t=np.radians(angle_a), ghost=ghost_a,
                  y=(lo[1], split_y)),
        "b": dict(keep=lambda p: p[:, 1] - split_y, sign=front, t=np.radians(angle_b), ghost=None, y=(split_y, hi[1])),
    }
    arrays, objects, colors_all, ghosts, spec = {}, [], {}, {}, None
    corners = []
    scale = kw.get("scale", 1e-3)
    for key, h in halves.items():
        mine = [(p, sh) for p, sh in extrema if h["y"][0] <= np.asarray(p)[1] <= h["y"][1]]
        runs = []                                   # the lines on this half, and only where its surface is solid
        for ln in lines:
            for r in _split_polyline(ln, h["keep"], +1):
                runs += _split_polyline(r, h["ghost"], +1) if h["ghost"] is not None else [r]
        sub = build_cutaway(out / f"half_{key}", surfaces=surfaces, box=box, keep=[h["keep"]] * len(surfaces),
                            colors=colors, extrema=mine, extremum_color=extremum_color,
                            ghost=h["ghost"], ghost_alpha=ghost_alpha, ghost_rims=h["ghost"] is not None,
                            lines=runs, line_colors=[line_color] * len(runs) if runs else None,
                            line_radius=line_radius, **kw)
        a = h["sign"] * h["t"]
        rot = np.array([[np.cos(a), -np.sin(a), 0.0], [np.sin(a), np.cos(a), 0.0], [0.0, 0.0, 1.0]])
        pivot = np.array([hinge_x, split_y, 0.0])
        turn = lambda q, s=1.0: (np.asarray(q, dtype=float) - pivot * s) @ rot.T + pivot * s
        data = np.load(sub / "meshes.npz")
        sub_spec = json.loads((sub / "scene.json").read_text())
        for name in data.files:
            arr = data[name]
            arrays[f"{key}_{name}"] = turn(arr, scale).astype(np.float32) if name.endswith("_v") else arr
        for o in sub_spec["objects"]:
            o = dict(o, name=f"{key}_{o['name']}")
            if "points" in o:
                o["points"] = turn(o["points"], scale).tolist()
            objects.append(o)
        colors_all.update(sub_spec["colors"])
        ghosts.update(sub_spec.get("ghosts", {}))
        spec = spec or sub_spec
        cy = np.array(h["y"])
        box_corners = np.array(np.meshgrid([lo[0], hi[0]], cy, [lo[2], hi[2]], indexing="ij")).reshape(3, -1).T
        corners.append(turn(box_corners))
        if key == "a" and glass is not None and (section_x is not None or section_y is not None):
            # the pane spans the height of the surfaces (plus a margin), not the empty box below
            pts = np.vstack([np.asarray(s.points) for s in surfaces])
            pts = pts[np.all((pts[:, :2] >= lo[:2]) & (pts[:, :2] <= hi[:2]), axis=1) & (pts[:, 1] <= split_y)]
            band = 0.03 * np.ptp(pts[:, 0] if section_x is not None else pts[:, 1])
            near = (np.abs(pts[:, 0] - section_x) if section_x is not None else np.abs(pts[:, 1] - section_y)) < band
            zs = pts[near, 2] if near.any() else pts[:, 2]       # the surfaces' height along the section
            margin = glass.get("margin", 0.05) * np.ptp(zs)
            za, zb = max(zs.min() - margin, lo[2]), min(zs.max() + margin, hi[2])
            if section_x is not None:                      # the pane across the half at the section
                pane = np.array([[section_x, lo[1], za], [section_x, split_y, za],
                                 [section_x, split_y, zb], [section_x, lo[1], zb]])
            else:
                pane = np.array([[lo[0], section_y, za], [hi[0], section_y, za],
                                 [hi[0], section_y, zb], [lo[0], section_y, zb]])
            arrays["glass_v"] = turn(pane * scale, scale).astype(np.float32)
            arrays["glass_f"] = np.array([[0, 1, 2], [0, 2, 3]])
            objects.append({"name": "glass", "type": "mesh", "material": "glass"})
            frame = turn(np.vstack([pane, pane[:1]]) * scale, scale)
            objects.append({"name": "glassframe", "type": "tube", "points": frame.tolist(),
                            "radius": glass.get("frame_radius", 8.0) * scale, "material": "glassframe"})
    np.savez_compressed(out / "meshes.npz", **arrays)
    extra = {"texture": texture, "texture_size": texture_size}
    if glass is not None:
        extra["glass"] = {k: v for k, v in glass.items() if k not in ("frame_radius", "margin")}
    (out / "scene.json").write_text(json.dumps(dict(spec, objects=objects, colors=colors_all, ghosts=ghosts, **extra),
                                               indent=1))
    return out, np.vstack(corners)


def _boundary_lines(mesh):
    """The free edges of a surface (where it ends), as polylines."""
    edges = mesh.extract_feature_edges(boundary_edges=True, feature_edges=False, manifold_edges=False,
                                       non_manifold_edges=False)
    if edges.n_points == 0:
        return []
    edges = edges.strip()
    lines, out, i = edges.lines, [], 0
    while i < len(lines):
        n = lines[i]
        if n >= 2:
            out.append(np.asarray(edges.points[lines[i + 1:i + 1 + n]], dtype=float))
        i += n + 1
    return out


def _bedding_slabs(points, normals, size):
    """Two meshes for bedding readings: each a small slab of `size` = (length along strike, width
    down dip, thickness) m, centred on its point and lying in the bedding plane; the half on the
    side its normal points to (way up) is one mesh, the half below the other."""
    length, width, thick = size
    corners = np.array([[i, j, k] for i in (-0.5, 0.5) for j in (-0.5, 0.5) for k in (0.0, 1.0)])
    quads = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
    tris = np.array([t for a, b, c, d in quads for t in ((a, b, c), (a, c, d))])
    halves = {+1: ([], []), -1: ([], [])}
    for p, n in zip(np.asarray(points, float), np.asarray(normals, float)):
        n = n / np.linalg.norm(n)
        s = np.cross(n, [0.0, 0.0, 1.0])
        s = s / np.linalg.norm(s) if np.linalg.norm(s) > 1e-6 else np.array([1.0, 0.0, 0.0])
        d = np.cross(n, s)
        for side, (verts, faces) in halves.items():
            c = corners * [length, width, side * 0.5 * thick]
            faces.append(tris[:, ::side] + 8 * len(verts))      # flip the winding of the lower half
            verts.append(p + c[:, :1] * s + c[:, 1:2] * d + c[:, 2:] * n)
    return {side: (np.vstack(v), np.vstack(f)) for side, (v, f) in halves.items() if v}


def build_slab(out_dir, *, surfaces, box, colors, tubes=(), extrema=(), extremum_color="#FF1FA8", tube_radius=5.0,
               halo_radius=None, protrude=0.0, edge_radius=3.0, edge_smooth=None, glyph_radius=8.0, min_area=0.0,
               occlusion=None, matte=False, backface_darken=0.0, surface_alpha=1.0, contacts=(),
               point_radius=5.0, bedding=None, bedding_size=(40.0, 10.0, 4.0),
               bedding_colors=("#2EC4D6", "#E0302A"), world_strength=0.9, sun_energy=2.5, scale=1e-3):
    """Write one slab of nested contact surfaces, cropped to `box`, to `out_dir`.

    surfaces : one PyVista mesh per contact (model frame, m), youngest first.
    Every edge of each cropped surface (where the box cuts it, and any hole) is drawn as a
    tube of `edge_radius` in a darker shade of its colour, as the traces in `build_scene`.
    tubes    : (x, z) of lines along y (e.g. interior extrema of a cylindrical field), drawn
               through the slab as tubes of `tube_radius` in `extremum_color`, sticking out of the
               front face by `protrude` m; ``halo_radius`` adds a translucent sleeve around each.
    extrema  : list of (point (3,), enclosure mesh or None), as in `build_scene`: each one inside
               the box is drawn as its enclosure, clipped to the box, or a sphere of `glyph_radius`.
    surface_alpha : opacity of the surfaces (1 = opaque), e.g. to show data through them.
    contacts : one (n, 3) array of contact points per surface, drawn as opaque spheres of
               `point_radius` in that surface's colour.
    bedding  : optional (points (m, 3), younging normals (m, 3)): each reading a small slab
               (`bedding_size`), its way-up half in ``bedding_colors[0]``, the other half in
               ``bedding_colors[1]``.
    edge_smooth : each surface's edges, and the tubes along them, are smoothed along themselves
               over about this many m (default three edge radii; 0 = as cut).
    occlusion, matte, backface_darken : the looks of `build_cutaway`: EEVEE ambient occlusion over
               that distance (m) with soft sun shadows, rough low-sheen surfaces, and undersides
               this much darker (0-1). All off by default.
    `scale` converts metres to Blender units.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    lo, hi = np.asarray(box, dtype=float)
    arrays, objects = {}, []
    smooth = 3.0 * edge_radius if edge_smooth is None else edge_smooth
    for h, surf in enumerate(surfaces):
        piece = _clip(surf, lo, hi)
        piece = clean(piece, min_area=min_area) if piece is not None else None
        if piece is None:
            continue
        piece = _smooth_boundary(piece, smooth)
        v, f = _mesh_arrays(piece)
        arrays[f"contact{h}_v"], arrays[f"contact{h}_f"] = v * scale, f
        objects.append({"name": f"contact{h}", "type": "mesh", "material": f"contact{h}"})
        for i, line in enumerate(_boundary_lines(piece)):
            line = _smooth_polyline(line, smooth, step=max(edge_radius, 1e-3))
            objects.append({"name": f"edge{h}_{i}", "type": "tube", "points": (line * scale).tolist(),
                            "radius": edge_radius * scale, "material": f"trace{h}"})
    for t, (tx, tz) in enumerate(tubes):
        pts = np.array([[tx, lo[1] - protrude, tz], [tx, hi[1], tz]])
        objects.append({"name": f"tube{t}", "type": "tube", "points": (pts * scale).tolist(),
                        "radius": tube_radius * scale, "material": "extremum"})
        if halo_radius:
            objects.append({"name": f"halo{t}", "type": "tube", "points": (pts * scale).tolist(),
                            "radius": halo_radius * scale, "material": "shell"})
    for e, (p, shell) in enumerate(extrema):
        p = np.asarray(p, dtype=float)
        if not np.all((p >= lo) & (p <= hi)):
            continue
        piece = _clip(shell, lo, hi) if shell is not None else None
        if piece is not None:                        # the enclosure marks the extremum
            v, f = _mesh_arrays(piece)
            arrays[f"shell{e}_v"], arrays[f"shell{e}_f"] = v * scale, f
            objects.append({"name": f"shell{e}", "type": "mesh", "material": "shell"})
        else:                                        # no closed enclosure: a small sphere instead
            objects.append({"name": f"ext{e}", "type": "spheres", "points": [(p * scale).tolist()],
                            "radius": glyph_radius * scale, "material": "extremum"})
    for h, pts in enumerate(contacts):
        pts = np.asarray(pts, dtype=float)
        if len(pts):
            objects.append({"name": f"points{h}", "type": "spheres", "points": (pts * scale).tolist(),
                            "radius": point_radius * scale, "material": f"contact{h}"})
    colours = {f"contact{h}": c for h, c in enumerate(colors)}
    if bedding is not None and len(bedding[0]):
        for side, (v, f) in _bedding_slabs(*bedding, bedding_size).items():
            name = "bedding_up" if side > 0 else "bedding_down"
            arrays[f"{name}_v"], arrays[f"{name}_f"] = v * scale, f
            objects.append({"name": name, "type": "mesh", "material": name})
        colours.update(bedding_up=bedding_colors[0], bedding_down=bedding_colors[1])
    np.savez_compressed(out / "meshes.npz", **arrays)
    (out / "scene.json").write_text(json.dumps({"objects": objects, "colors": colours,
                                                "alphas": {f"contact{h}": surface_alpha for h in range(len(colors))},
                                                "extremum_color": extremum_color,
                                                "world_strength": world_strength, "sun_energy": sun_energy,
                                                **_looks(occlusion, matte, backface_darken, scale)}, indent=1))
    return out


def build_terraces(out_dir, *, surfaces, box, centre, radii, colors, lines=(), extrema=(),
                   extremum_color="#FF1FA8", rim_radius=0.6, line_radius=0.35, glyph_radius=2.5,
                   min_area=0.0, world_strength=0.9, sun_energy=2.5, scale=1e-3):
    """Write a terraced cutaway of nested contact surfaces to `out_dir`.

    surfaces : one PyVista mesh per contact (model frame, m), youngest first.
    box      : [[x0, y0, z0], [x1, y1, z1]], the domain every surface is cropped to.
    centre, radii : surface h keeps only its part outside the sphere of radius ``radii[h]``
               about ``centre``, so with radii shrinking from youngest to oldest each older
               surface reaches further into the cut, like terraces. The rim where a surface
               meets its sphere is drawn as a tube in a darker shade of its colour.
    lines    : (n, 3) polylines (e.g. digitised traces), drawn as thin dark tubes.
    extrema  : list of (point (3,), enclosure mesh or None) inside the box, drawn in
               `extremum_color`; an extremum without an enclosure gets a small sphere.
    world_strength, sun_energy : brightness of the white surroundings and of the sun; lower
               values give deeper colours.
    `scale` converts metres to Blender units.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    lo, hi = np.asarray(box, dtype=float)
    centre = np.asarray(centre, dtype=float)
    arrays, objects = {}, []

    def add_mesh(name, mesh, material):
        v, f = _mesh_arrays(mesh)
        arrays[f"{name}_v"] = v * scale
        arrays[f"{name}_f"] = f
        objects.append({"name": name, "type": "mesh", "material": material})

    for h, (surf, r) in enumerate(zip(surfaces, radii)):
        kept, rims = _sphere_cut(_clip(surf, lo, hi), centre, float(r))
        kept = clean(kept, min_area=min_area) if kept is not None else None
        if kept is not None:
            add_mesh(f"terrace{h}", kept, f"contact{h}")
        for i, rim in enumerate(rims):
            objects.append({"name": f"rim{h}_{i}", "type": "tube", "points": (rim * scale).tolist(),
                            "radius": rim_radius * scale, "material": f"trace{h}"})
    for li, pts in enumerate(lines):
        pts = np.asarray(pts, dtype=float)
        ok = np.all((pts >= lo) & (pts <= hi), axis=1)
        idx = np.flatnonzero(ok)
        for k, run in enumerate(np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1) if len(idx) else []):
            if len(run) >= 2:
                objects.append({"name": f"line{li}_{k}", "type": "tube", "points": (pts[run] * scale).tolist(),
                                "radius": line_radius * scale, "material": "hole"})
    for e, (p, shell) in enumerate(extrema):
        p = np.asarray(p, dtype=float)
        if not np.all((p >= lo) & (p <= hi)):
            continue
        piece = _clip(shell, lo, hi) if shell is not None else None
        if piece is not None:
            add_mesh(f"shell{e}", piece, "shell")
        else:
            objects.append({"name": f"ext{e}", "type": "spheres", "points": [(p * scale).tolist()],
                            "radius": glyph_radius * scale, "material": "extremum"})

    np.savez_compressed(out / "meshes.npz", **arrays)
    materials = {f"contact{h}": c for h, c in enumerate(colors)}
    (out / "scene.json").write_text(json.dumps({"objects": objects, "colors": materials,
                                                "extremum_color": extremum_color,
                                                "world_strength": world_strength, "sun_energy": sun_energy},
                                               indent=1))
    return out


def build_scene(out_dir, *, surfaces, x_range, y_range, cuts, z_view, offset, collars, contacts, colors,
                extrema=(), extremum_color="#AA3377", traces=True, trace_radius=5.0, hole_radius=6.0,
                point_radius=11.0, glyph_radius=16.0, lines=(), line_radius=1.5, contact_alpha=1.0, trace_smooth=None,
                occlusion=None, matte=False, backface_darken=0.0, scale=1e-3):
    """Write the exploded scene of one model to `out_dir`.

    surfaces : one PyVista mesh per contact (model frame, m), youngest first.
    x_range, y_range : extent of the slabs; cuts: y of the cuts between slabs.
    z_view   : (z_min, z_max) crop of every slab.
    offset   : (dy, dz) added per slab: the gap along y and the lift of the staircase.
    collars  : (n, 2) drillhole collars; contacts: list of (m, 3) points per horizon.
    extrema  : list of (point (3,), enclosure mesh or None), drawn in `extremum_color`;
               an extremum without an enclosure gets a small sphere.
    contact_alpha : opacity of the contact surfaces (1 = opaque); the traces and points stay opaque.
    lines    : list of (n, 3) polylines (e.g. digitised traces), drawn as thin dark tubes of
               radius `line_radius` where they fall inside a slab.
    trace_smooth : the traces, and the surfaces' cut edges under them, are smoothed along
               themselves over about this many m (default three trace radii; 0 = as cut).
    occlusion, matte, backface_darken : the looks of `build_cutaway`: EEVEE ambient occlusion over
               that distance (m) with soft sun shadows, rough low-sheen surfaces, and undersides
               this much darker (0-1). All off by default.
    `scale` converts metres to Blender units.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    y_edges = [y_range[0], *cuts, y_range[1]]
    arrays, objects = {}, []
    smooth = 3.0 * trace_radius if trace_smooth is None else trace_smooth

    def add_mesh(name, mesh, material, shift):
        v, f = _mesh_arrays(mesh)
        arrays[f"{name}_v"] = (v + shift) * scale
        arrays[f"{name}_f"] = f
        objects.append({"name": name, "type": "mesh", "material": material})

    for k, (ylo, yhi) in enumerate(zip(y_edges[:-1], y_edges[1:])):
        shift = np.array([0.0, k * offset[0], k * offset[1]])
        lo = np.array([x_range[0], ylo, z_view[0]])
        hi = np.array([x_range[1], yhi, z_view[1]])
        for h, surf in enumerate(surfaces):
            piece = _clip(surf, lo, hi)
            if piece is not None:
                add_mesh(f"slab{k}_contact{h}", _smooth_boundary(piece, smooth), f"contact{h}", shift)
            if traces:                               # each contact where the slab is cut
                for side, yy in (("front", ylo), ("back", yhi)):
                    for i, line in enumerate(_trace(surf, yy, x_range, z_view)):
                        line = _smooth_polyline(line, smooth, step=max(trace_radius, 1e-3))
                        objects.append({"name": f"slab{k}_{side}_trace{h}_{i}", "type": "tube",
                                        "points": ((line + shift) * scale).tolist(),
                                        "radius": trace_radius * scale, "material": f"trace{h}"})
        for i, (cx, cy) in enumerate(collars):
            if ylo - 1.0 <= cy < yhi - 1.0:
                pts = np.array([[cx, cy, z_view[0]], [cx, cy, z_view[1]]]) + shift
                objects.append({"name": f"slab{k}_hole{i}", "type": "tube", "points": (pts * scale).tolist(),
                                "radius": hole_radius * scale, "material": "hole"})
        for h, pts in enumerate(contacts):
            m = (pts[:, 1] >= ylo - 1.0) & (pts[:, 1] < yhi - 1.0) & (pts[:, 2] >= z_view[0]) & (pts[:, 2] <= z_view[1])
            if m.any():
                objects.append({"name": f"slab{k}_points{h}", "type": "spheres",
                                "points": ((pts[m] + shift) * scale).tolist(), "radius": point_radius * scale,
                                "material": f"contact{h}"})
        for li, pts in enumerate(lines):             # digitised traces, as thin tubes
            pts = np.asarray(pts, dtype=float)
            ok = ((pts[:, 1] >= ylo - 1.0) & (pts[:, 1] < yhi - 1.0) & (pts[:, 2] >= z_view[0]) & (pts[:, 2] <= z_view[1])
                  & (pts[:, 0] >= x_range[0]) & (pts[:, 0] <= x_range[1]))
            idx = np.flatnonzero(ok)
            for r, run in enumerate(np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1) if len(idx) else []):
                if len(run) >= 2:
                    objects.append({"name": f"slab{k}_line{li}_{r}", "type": "tube",
                                    "points": ((pts[run] + shift) * scale).tolist(),
                                    "radius": line_radius * scale, "material": "hole"})
        for e, (p, shell) in enumerate(extrema):
            p = np.asarray(p, dtype=float)
            if not (ylo <= p[1] < yhi and z_view[0] <= p[2] <= z_view[1]):
                continue
            piece = _clip(shell, lo, hi) if shell is not None else None
            if piece is not None:                    # the enclosure marks the extremum
                add_mesh(f"slab{k}_shell{e}", piece, "shell", shift)
            else:                                    # no closed enclosure: a small sphere instead
                objects.append({"name": f"slab{k}_ext{e}", "type": "spheres",
                                "points": [((p + shift) * scale).tolist()],
                                "radius": glyph_radius * scale, "material": "extremum"})

    np.savez_compressed(out / "meshes.npz", **arrays)
    materials = {f"contact{h}": c for h, c in enumerate(colors)}
    (out / "scene.json").write_text(json.dumps({"objects": objects, "colors": materials,
                                                "extremum_color": extremum_color,
                                                "contact_alpha": contact_alpha,
                                                **_looks(occlusion, matte, backface_darken, scale)}, indent=1))
    return out


def build_fence(out_dir, *, planes, images, x_range, z_view, offset, lines=(), band=15.0, markers=(),
                marker_color="#AA3377", frame_radius=0.6, line_radius=0.7, marker_radius=2.6,
                frame_color="#2b2b2b", outlines=None, line_mask=None, scale=1e-3):
    """Write an exploded fence diagram: textured sections at constant y, stepped apart.

    planes : y of each section, in order (nearest the camera first).
    images : one (rows, cols, 4) uint8 RGBA array per plane, rows running up from `z_view[0]`
             to `z_view[1]` (bottom to top) and columns from `x_range[0]` to `x_range[1]`.
    offset : (dy, dz) added per section, so they step away from, and up behind, the first.
    lines  : (n, 3) polylines (e.g. digitised beds). The parts within `band` of a plane are
             drawn on it as thin dark tubes, flattened onto the plane.
    markers: (m, 3) points (e.g. interior extrema) drawn on the nearest plane within `band`.
    Each section also gets a thin dark frame. Give ``outlines`` (one list of (n, 3) polylines
    per plane, at that plane's y) to outline something other than the rectangle, e.g. where
    the section's alpha is zero outside a domain. ``line_mask(k, points)`` may return a boolean
    array saying which polyline points may be drawn on section ``k`` (e.g. those inside its domain).
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    objects = []
    x0, x1 = x_range
    z0, z1 = z_view
    for k, (yk, img) in enumerate(zip(planes, images)):
        shift = np.array([0.0, k * offset[0], k * offset[1]])
        name = f"section{k}.png"
        from PIL import Image                                # local: only needed to write the textures
        Image.fromarray(np.asarray(img, dtype=np.uint8)[::-1]).save(out / name)   # image rows run top-down
        corners = np.array([[x0, yk, z0], [x1, yk, z0], [x1, yk, z1], [x0, yk, z1]]) + shift
        objects.append({"name": f"section{k}", "type": "quad", "corners": (corners * scale).tolist(), "image": name})
        rings = [np.vstack([corners, corners[:1]])] if outlines is None else \
            [np.asarray(o, dtype=float) + shift for o in outlines[k]]
        for ri, ring in enumerate(rings):
            objects.append({"name": f"frame{k}_{ri}", "type": "tube", "points": (ring * scale).tolist(),
                            "radius": frame_radius * scale, "material": "frame"})
        eps = np.array([0.0, -0.4, 0.0])                      # a hair toward the camera, so it sits on the face
        for li, pts in enumerate(lines):
            pts = np.asarray(pts, dtype=float)
            ok = ((np.abs(pts[:, 1] - yk) <= band) & (pts[:, 0] >= x0) & (pts[:, 0] <= x1)
                  & (pts[:, 2] >= z0) & (pts[:, 2] <= z1))
            if line_mask is not None:
                flat_all = pts.copy()
                flat_all[:, 1] = yk
                ok &= line_mask(k, flat_all)
            idx = np.flatnonzero(ok)
            for r, run in enumerate(np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1) if len(idx) else []):
                if len(run) >= 2:
                    flat = pts[run].copy()
                    flat[:, 1] = yk
                    objects.append({"name": f"section{k}_line{li}_{r}", "type": "tube",
                                    "points": ((flat + shift + eps) * scale).tolist(),
                                    "radius": line_radius * scale, "material": "hole"})
        for mi, p in enumerate(np.asarray(markers, dtype=float).reshape(-1, 3)):
            nearest = int(np.argmin(np.abs(np.asarray(planes) - p[1])))
            if nearest == k and abs(p[1] - yk) <= band and x0 <= p[0] <= x1 and z0 <= p[2] <= z1:
                objects.append({"name": f"section{k}_marker{mi}", "type": "spheres",
                                "points": [((np.array([p[0], yk, p[2]]) + shift + 2 * eps) * scale).tolist()],
                                "radius": marker_radius * scale, "material": "extremum"})
    np.savez_compressed(out / "meshes.npz", empty=np.zeros(1))
    (out / "scene.json").write_text(json.dumps({"objects": objects, "colors": {"frame": frame_color},
                                                "extremum_color": marker_color}, indent=1))
    return out


def build_layers(out_dir, *, meshes=(), photo_meshes=(), lines=(), colors, alphas=None, line_radius=0.7,
                 extrema=(), extremum_color="#AA3377", glyph_radius=3.0, occlusion=None, matte=False,
                 backface_darken=0.0, rim_light=0.0, world_strength=0.9, sun_energy=2.5, scale=1e-3):
    """Write a scene of whole (unsliced) layers, each moved to where it should sit.

    meshes       : list of dicts ``{"name", "mesh", "material"}`` (PyVista, already in the scene frame);
                   ``material`` is a key of ``colors``.
    photo_meshes : list of dicts ``{"name", "mesh", "rgb"}``: a surface with one 0-1 RGB colour per vertex.
    lines        : list of (n, 3) polylines, drawn as thin dark tubes of radius ``line_radius``.
    colors       : ``{material: hex}``; ``alphas`` optionally ``{material: opacity}``.
    extrema      : list of points, drawn as small spheres in ``extremum_color``.
    occlusion, matte, backface_darken, rim_light, world_strength, sun_energy : the looks of
                   `build_cutaway`, applied to the materials of ``meshes``.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    arrays, objects = {}, []
    for m in meshes:
        v, f = _mesh_arrays(m["mesh"])
        arrays[f"{m['name']}_v"], arrays[f"{m['name']}_f"] = v * scale, f
        objects.append({"name": m["name"], "type": "mesh", "material": m["material"]})
    for m in photo_meshes:
        v, f = _mesh_arrays(m["mesh"])
        arrays[f"{m['name']}_v"], arrays[f"{m['name']}_f"] = v * scale, f
        arrays[f"{m['name']}_c"] = np.asarray(m["rgb"], dtype=np.float32)
        objects.append({"name": m["name"], "type": "colormesh"})
    for i, pts in enumerate(lines):
        objects.append({"name": f"line{i}", "type": "tube", "points": (np.asarray(pts) * scale).tolist(),
                        "radius": line_radius * scale, "material": "hole"})
    for i, p in enumerate(extrema):
        objects.append({"name": f"ext{i}", "type": "spheres", "points": [(np.asarray(p) * scale).tolist()],
                        "radius": glyph_radius * scale, "material": "extremum"})
    np.savez_compressed(out / "meshes.npz", **arrays)
    (out / "scene.json").write_text(json.dumps({"objects": objects, "colors": colors, "alphas": alphas or {},
                                                "extremum_color": extremum_color,
                                                "look_materials": sorted({m["material"] for m in meshes}),
                                                "rim_light": rim_light, "world_strength": world_strength,
                                                "sun_energy": sun_energy,
                                                **_looks(occlusion, matte, backface_darken, scale)}, indent=1))
    return out


def shade_cloud(points, rgb, *, light, view, k=16, ambient=0.45, diffuse=0.75, chunk=500_000):
    """Relight a coloured point cloud by its own geometry, like a hillshade in 3-D.

    Each point's normal is the least-variance direction of its `k` nearest neighbours, turned to
    face the viewer (`view`: unit vector from the scene towards the camera). The colour is scaled
    by ``ambient + diffuse * max(n . light, 0)`` (`light`: unit vector towards the light), so the
    relief of the outcrop reads while the photograph's colours stay. Returns the new colours (0-1).
    """
    from scipy.spatial import cKDTree

    pts = np.asarray(points, dtype=np.float64)
    light = np.asarray(light, dtype=float) / np.linalg.norm(light)
    view = np.asarray(view, dtype=float) / np.linalg.norm(view)
    tree = cKDTree(pts)
    shade = np.empty(len(pts))
    for i in range(0, len(pts), chunk):
        _, idx = tree.query(pts[i:i + chunk], k=k, workers=-1)
        nb = pts[idx] - pts[idx].mean(1, keepdims=True)
        n = np.linalg.eigh(np.einsum("nki,nkj->nij", nb, nb))[1][:, :, 0]   # smallest eigenvector
        n *= np.sign(n @ view)[:, None]                                  # facing the viewer
        shade[i:i + chunk] = ambient + diffuse * np.clip(n @ light, 0.0, None)
    return np.clip(np.asarray(rgb, dtype=float) * shade[:, None], 0.0, 1.0)


def cast_shadows(points, rgb, casters, *, light, reach=20.0, strength=0.55, soft=0.35):
    """Darken a coloured point cloud where small objects (tubes, slabs) block the sun.

    casters : list of (samples (m, 3), radius m): points along a tube, or over a slab, each
              blocking a disc of `radius` across the light.
    A cloud point is in shadow when a caster sample lies within `radius` of the ray from it
    towards the sun (`light`, unit vector towards it), and no more than `reach` m along it. The
    shadow fades out over the last `soft` of the radius. Returns the darkened colours (0-1).
    """
    from scipy.spatial import cKDTree

    light = np.asarray(light, dtype=float) / np.linalg.norm(light)
    a = np.cross(light, [0.0, 0.0, 1.0]); a /= np.linalg.norm(a)
    b = np.cross(light, a)
    pts = np.asarray(points, dtype=np.float64)
    tree = cKDTree(np.c_[pts @ a, pts @ b])
    depth = pts @ light
    dark = np.zeros(len(pts))
    for samples, radius in casters:
        s = np.asarray(samples, dtype=float)
        for idx, q in zip(tree.query_ball_point(np.c_[s @ a, s @ b], radius, workers=-1), s):
            if not idx:
                continue
            idx = np.asarray(idx)
            along = q @ light - depth[idx]                       # caster towards the sun from the point
            ok = (along > 0.0) & (along < reach)
            r = np.hypot(pts[idx[ok]] @ a - q @ a, pts[idx[ok]] @ b - q @ b) / radius
            edge = np.clip((1.0 - r) / max(soft, 1e-6), 0.0, 1.0)
            np.maximum.at(dark, idx[ok], edge)
    return np.clip(np.asarray(rgb, dtype=float) * (1.0 - strength * dark)[:, None], 0.0, 1.0)


def resample_polyline(line, step):
    """Points every `step` m along a polyline."""
    line = np.asarray(line, dtype=float)
    s = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(line, axis=0), axis=1))]
    t = np.arange(0.0, s[-1] + 1e-9, step)
    return np.stack([np.interp(t, s, line[:, k]) for k in range(line.shape[1])], axis=1)


def slab_samples(points, normals, size, step):
    """Points over each bedding slab of `_bedding_slabs` (its mid-plane), every `step` m."""
    length, width, _ = size
    u, v = np.meshgrid(np.arange(-length / 2, length / 2 + 1e-9, step), np.arange(-width / 2, width / 2 + 1e-9, step))
    out = []
    for p, n in zip(np.asarray(points, float), np.asarray(normals, float)):
        n = n / np.linalg.norm(n)
        s_ = np.cross(n, [0.0, 0.0, 1.0])
        s_ = s_ / np.linalg.norm(s_) if np.linalg.norm(s_) > 1e-6 else np.array([1.0, 0.0, 0.0])
        d = np.cross(n, s_)
        out.append(p + u.reshape(-1, 1) * s_ + v.reshape(-1, 1) * d)
    return np.vstack(out) if out else np.empty((0, 3))


def build_cloud(out_dir, *, points, rgb, lines=(), line_color="#1f1f1f", point_radius=0.5, line_radius=0.8,
                bedding=None, bedding_size=(4.0, 1.5, 0.6), bedding_colors=("#2EC4D6", "#E0302A"),
                emission=0.6, unlit=False, world_strength=0.9, sun_energy=2.5, sun_direction=None,
                occlusion=None, scale=1e-3):
    """Write a scene of a coloured point cloud (e.g. photogrammetry) with polylines on it.

    points : (n, 3) model frame, m; rgb: (n, 3) in 0-1 (sRGB), or uint8 0-255. Each point is a
             small sphere of `point_radius`, in its own colour, lit by the sun and partly self-lit
             (`emission`), so it looks like the photograph rather than a shaded model.
    lines  : (m, 3) polylines (e.g. digitised traces), drawn as tubes of `line_radius` in
             `line_color`.
    bedding : optional (points (k, 3), younging normals (k, 3)): each reading a small slab
             (`bedding_size`, as `build_slab`), its way-up half in ``bedding_colors[0]``, the
             other half in ``bedding_colors[1]``.
    unlit  : draw each point in exactly its own colour (self-lit only), as the photograph; the
             tubes and slabs are still lit by the sun and the world.
    sun_direction : (3,) towards the sun (model frame), e.g. the light of `shade_cloud`, so baked
             relief and cast shadows agree (default: the renderer's fixed sun).
    occlusion : EEVEE ambient occlusion over that distance (m), with soft sun shadows (None = off);
             with ``unlit=False`` the points then receive the shadows of the tubes and slabs.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pts = np.asarray(points, dtype=float)
    rgb = np.asarray(rgb)
    rgb8 = rgb if rgb.dtype == np.uint8 else np.round(np.clip(rgb, 0, 1) * 255).astype(np.uint8)
    arrays = {"cloud_v": (pts * scale).astype(np.float32), "cloud_c8": rgb8}
    objects = [{"name": "cloud", "type": "points", "radius": point_radius * scale, "emission": emission,
                "unlit": bool(unlit)}]
    for li, line in enumerate(lines):
        line = np.asarray(line, dtype=float)
        if len(line) >= 2:
            objects.append({"name": f"line{li}", "type": "tube", "points": (line * scale).tolist(),
                            "radius": line_radius * scale, "material": "line"})
    colours = {"line": line_color}
    if bedding is not None and len(bedding[0]):
        for side, (v, f) in _bedding_slabs(*bedding, bedding_size).items():
            name = "bedding_up" if side > 0 else "bedding_down"
            arrays[f"{name}_v"], arrays[f"{name}_f"] = v * scale, f
            objects.append({"name": name, "type": "mesh", "material": name})
        colours.update(bedding_up=bedding_colors[0], bedding_down=bedding_colors[1])
    np.savez(out / "meshes.npz", **arrays)                    # uncompressed: millions of points load faster
    (out / "scene.json").write_text(json.dumps({"objects": objects, "colors": colours,
                                                "world_strength": world_strength, "sun_energy": sun_energy,
                                                "sun_direction": None if sun_direction is None else
                                                np.asarray(sun_direction, float).tolist(),
                                                **_looks(occlusion, False, 0.0, scale)}, indent=1))
    return out


def build_gnomon(out_dir, *, origin, length=250.0, shaft_radius=7.0, head_radius=18.0, head_length=55.0,
                 color="#3a3a3a", scale=1e-3):
    """A scene with three arrows of `length` (m) along x, y and z from `origin` (model frame).

    Returns the arrow tips (model frame), for labelling."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    origin = np.asarray(origin, dtype=float)
    objects, tips = [], {}
    for label, axis in zip("xyz", np.eye(3)):
        tip = origin + length * axis
        base = tip - head_length * axis
        objects.append({"name": f"shaft_{label}", "type": "tube", "points": (np.stack([origin, base]) * scale).tolist(),
                        "radius": shaft_radius * scale, "material": "axis"})
        objects.append({"name": f"head_{label}", "type": "cone", "base": (base * scale).tolist(),
                        "tip": (tip * scale).tolist(), "radius": head_radius * scale, "material": "axis"})
        tips[label] = tip
    objects.append({"name": "origin", "type": "spheres", "points": [(origin * scale).tolist()],
                    "radius": 1.3 * shaft_radius * scale, "material": "axis"})
    np.savez_compressed(out / "meshes.npz", empty=np.zeros(1))
    (out / "scene.json").write_text(json.dumps({"objects": objects, "colors": {}, "axis_color": color}, indent=1))
    return tips


def project(points, cam, resolution, scale=1e-3):
    """Pixel (column, row) of model-frame points in a render made with `cam` (orthographic, or
    perspective when `cam` has a ``lens``, as :func:`perspective_camera` makes)."""
    pts = np.atleast_2d(np.asarray(points, dtype=float)) * scale
    rx, ry = resolution
    if "lens" in cam:
        rel = pts - np.asarray(cam["location"])
        forward = np.asarray(cam["target"]) - np.asarray(cam["location"])
        forward /= np.linalg.norm(forward)
        depth = rel @ forward
        k = cam["lens"] * max(rx, ry) / cam.get("sensor", 36.0)      # pixels per unit of tan(angle)
        return np.c_[rx / 2 + k * (rel @ np.asarray(cam["right"])) / depth,
                     ry / 2 - k * (rel @ np.asarray(cam["up"])) / depth]
    rel = pts - np.asarray(cam["target"])
    u, v = rel @ np.asarray(cam["right"]), rel @ np.asarray(cam["up"])
    span = cam["ortho_scale"] / max(rx, ry)          # Blender units per pixel
    return np.c_[rx / 2 + u / span, ry / 2 - v / span]


def perspective_camera(points, azimuth, elevation, resolution, *, distance, margin=0.04, target=None, scale=1e-3):
    """Perspective camera framing ``points`` (model frame, m) from azimuth/elevation in degrees.

    The camera looks at ``target`` (model frame, m; default the centre of the points' bounding
    box) from ``distance`` m away (shorter = stronger perspective) and its lens is set so every
    point fits, with ``margin``. Azimuth as
    in :func:`camera`. Blender's default 36 mm sensor fits the larger image dimension.
    """
    pts = np.atleast_2d(np.asarray(points, dtype=float))
    target = 0.5 * (pts.min(0) + pts.max(0)) if target is None else np.asarray(target, dtype=float)
    a, e = np.radians(azimuth), np.radians(elevation)
    d = np.array([np.sin(a) * np.cos(e), -np.cos(a) * np.cos(e), np.sin(e)])
    loc = target + distance * d
    right = np.cross(-d, [0.0, 0.0, 1.0]); right /= np.linalg.norm(right)
    up = np.cross(right, -d)
    rel = pts - loc
    depth = rel @ -d
    tan_u, tan_v = np.abs(rel @ right / depth).max(), np.abs(rel @ up / depth).max()
    rx, ry = resolution
    tan_big = max(tan_u, tan_v * rx / ry) if rx >= ry else max(tan_v, tan_u * ry / rx)
    lens = 18.0 / (tan_big * (1.0 + 2.0 * margin))
    return {"location": (loc * scale).tolist(), "target": (target * scale).tolist(), "lens": float(lens),
            "sensor": 36.0, "right": right.tolist(), "up": up.tolist()}


def camera(target, azimuth, elevation, distance, ortho_scale, scale=1e-3):
    """Orthographic camera looking at `target` (model frame, m) from azimuth/elevation in degrees.

    Azimuth 0 looks along +y; positive azimuth swings the camera towards +x."""
    a, e = np.radians(azimuth), np.radians(elevation)
    d = np.array([np.sin(a) * np.cos(e), -np.cos(a) * np.cos(e), np.sin(e)])
    loc = np.asarray(target, dtype=float) + distance * d
    right = np.cross(-d, [0.0, 0.0, 1.0]); right /= np.linalg.norm(right)
    up = np.cross(right, -d)
    return {"location": (loc * scale).tolist(), "target": (np.asarray(target) * scale).tolist(),
            "ortho_scale": ortho_scale * scale, "right": right.tolist(), "up": up.tolist()}


def fit_camera(bounds, azimuth, elevation, resolution, margin=0.04, distance=8000.0, scale=1e-3):
    """Orthographic camera framing the box `bounds` ((x0, x1), (y0, y1), (z0, z1)) in the render."""
    a, e = np.radians(azimuth), np.radians(elevation)
    view = -np.array([np.sin(a) * np.cos(e), -np.cos(a) * np.cos(e), np.sin(e)])
    right = np.cross(view, [0.0, 0.0, 1.0]); right /= np.linalg.norm(right)
    up = np.cross(right, view)
    corners = np.array(np.meshgrid(*bounds, indexing="ij")).reshape(3, -1).T
    u, v = corners @ right, corners @ up
    centre = corners.mean(0)
    target = centre + right * (0.5 * (u.min() + u.max()) - centre @ right) + up * (0.5 * (v.min() + v.max()) - centre @ up)
    w, h = (u.max() - u.min()) * (1 + 2 * margin), (v.max() - v.min()) * (1 + 2 * margin)
    rx, ry = resolution
    ortho = max(w, h * rx / ry) if rx >= ry else max(h, w * ry / rx)
    return camera(target, azimuth, elevation, distance, ortho, scale=scale)


def exploded_bounds(x_range, y_range, cuts, z_view, offset):
    """Box around the exploded slabs."""
    n = len(cuts)
    return (tuple(x_range), (y_range[0], y_range[1] + n * offset[0]), (z_view[0], z_view[1] + n * offset[1]))


def render(scene_dir, png, *, cam, resolution, engine="EEVEE", samples=64, timeout=900):
    """Render a scene written by `build_scene` with headless Blender."""
    settings = {"camera": cam, "resolution": list(resolution), "engine": engine, "samples": samples}
    cfg = Path(scene_dir) / "render.json"
    cfg.write_text(json.dumps(settings, indent=1))
    cmd = [BLENDER, "-b", "--factory-startup", "-P", str(RENDER_SCRIPT), "--",
           str(Path(scene_dir).resolve()), str(Path(png).resolve())]
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if res.returncode != 0 or not Path(png).exists():
        raise RuntimeError(f"Blender failed:\n{res.stdout[-2000:]}\n{res.stderr[-2000:]}")
    return Path(png)
